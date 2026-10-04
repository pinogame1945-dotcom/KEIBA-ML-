#!/usr/bin/env python3
import argparse,json,math,multiprocessing,os,time
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l2_core_v1 import load_dataset,write_csv
from run_l2_pricer_latent_race_v1 import load_outcomes,build_races
from run_l2_pricer_contextual_volatility_v2 import (
    attach_features,feature_frame,antithetic_normal,antithetic_student,
    rows_for_action,loss_for_rows
)

FOLDS=((2023,(2022,)),(2024,(2022,2023)),(2025,(2022,2023,2024)))
BET_TYPES=("WIN","EXACTA","TRIO","TRIFECTA")
LABEL_DRAWS=128
FINAL_DRAWS=1536
SIGMA_TRAIN=(0.20,0.40,0.60,0.80)
LAMBDA_TRAIN=(0.50,1.00,1.50)
SIGMA_DENSE=tuple(round(x,2) for x in np.arange(0.05,0.851,0.05))
LAMBDA_DENSE=tuple(round(x,2) for x in np.arange(0.25,1.751,0.25))
SWITCH_MARGINS=(0.0,-0.001,-0.0025,-0.005,-0.01,-0.02,-0.04)
BASE_FALLBACK=(0.0,0.0025,0.005,0.01,0.02,0.04)


def parse_args():
    p=argparse.ArgumentParser(description="L2 PRICER Continuous Surface V3")
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--workers",type=int,default=0)
    return p.parse_args()


def action(name,family,sigma,lam=0.0,df=None):
    return {"name":name,"family":family,"sigma":float(sigma),"lambda":float(lam),"df":df}


TRAIN_ACTIONS=[action("BASE_PL","BASE_PL",0.0)]
for s in SIGMA_TRAIN:
    TRAIN_ACTIONS.append(action(f"NH_S{s:.2f}","NORMAL_HOMO",s))
for lam in LAMBDA_TRAIN:
    for s in SIGMA_TRAIN:
        TRAIN_ACTIONS.append(action(f"NHE_L{lam:.2f}_S{s:.2f}","NORMAL_HETERO",s,lam))
for lam in LAMBDA_TRAIN:
    for s in SIGMA_TRAIN:
        TRAIN_ACTIONS.append(action(f"ST_L{lam:.2f}_S{s:.2f}","STUDENTT_HETERO",s,lam,5.0))
TRAIN_BY_NAME={a["name"]:a for a in TRAIN_ACTIONS}


def dense_actions():
    out=[TRAIN_BY_NAME["BASE_PL"]]
    for s in SIGMA_DENSE:
        out.append(action(f"NH_S{s:.2f}","NORMAL_HOMO",s))
    for lam in LAMBDA_DENSE:
        for s in SIGMA_DENSE:
            out.append(action(f"NHE_L{lam:.2f}_S{s:.2f}","NORMAL_HETERO",s,lam))
    for lam in LAMBDA_DENSE:
        for s in SIGMA_DENSE:
            out.append(action(f"ST_L{lam:.2f}_S{s:.2f}","STUDENTT_HETERO",s,lam,5.0))
    return out


DENSE_ACTIONS=dense_actions()


def action_features(a):
    fam=a["family"]
    sigma=float(a["sigma"]); lam=float(a["lambda"])
    return {
        "a_sigma":sigma,
        "a_sigma2":sigma*sigma,
        "a_lambda":lam,
        "a_sigma_lambda":sigma*lam,
        "a_is_normal":1.0 if fam in ("NORMAL_HOMO","NORMAL_HETERO") else 0.0,
        "a_is_student":1.0 if fam=="STUDENTT_HETERO" else 0.0,
        "a_is_hetero":1.0 if fam in ("NORMAL_HETERO","STUDENTT_HETERO") else 0.0,
        "a_is_base":1.0 if fam=="BASE_PL" else 0.0,
    }


def compute_losses_one(race,draws,phase):
    n=len(race["base"])
    zn=antithetic_normal(race["race_id"],draws,n,phase)
    zt=antithetic_student(race["race_id"],draws,n,phase)
    out={}
    for a in TRAIN_ACTIONS:
        rows=rows_for_action(race,a,zn,zt)
        out[a["name"]]=loss_for_rows(race,rows)
    return out


def _loss_chunk(args):
    races,draws,phase=args
    return [(r["race_id"],compute_losses_one(r,draws,phase)) for r in races]


def parallel_losses(races,draws,phase,workers):
    if workers<=1:
        parts=[_loss_chunk((races,draws,phase))]
    else:
        chunks=[races[i::workers] for i in range(workers)]
        ctx=multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as ex:
            parts=list(ex.map(_loss_chunk,[(c,draws,phase) for c in chunks if c]))
    out={}
    for part in parts:
        for rid,z in part: out[rid]=z
    return out


def fit_val_split(train):
    years=sorted({r["year"] for r in train})
    if len(years)>=2:
        vy=years[-1]
        fit=[r for r in train if r["year"]<vy]
        val=[r for r in train if r["year"]==vy]
        return fit,val,f"latest_full_year_{vy}"
    q=sorted(train,key=lambda r:(r["race_date"],r["race_id"]))
    cut=max(1,min(len(q)-1,int(len(q)*0.75)))
    return q[:cut],q[cut:],"latest_25pct_single_year"


def surface_rows(races,losses,feature_names):
    rows=[]; y=[]
    for r in races:
        base=float(losses[r["race_id"]]["BASE_PL"]["TRIFECTA"])
        ctx=r["features"]
        for a in TRAIN_ACTIONS:
            z={k:float(ctx[k]) for k in feature_names}
            z.update(action_features(a))
            rows.append(z)
            delta=float(losses[r["race_id"]][a["name"]]["TRIFECTA"])-base
            y.append(max(-0.50,min(0.50,delta)))
    cols=feature_names+list(action_features(TRAIN_ACTIONS[0]).keys())
    return pd.DataFrame(rows,columns=cols,dtype=np.float32),np.asarray(y,dtype=np.float32),cols


def model_params(seed):
    return dict(
        objective="huber",
        alpha=0.90,
        n_estimators=420,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=180,
        subsample=0.90,
        colsample_bytree=0.75,
        reg_lambda=8.0,
        reg_alpha=0.5,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )


def fit_surface(races,losses,feature_names):
    X,y,cols=surface_rows(races,losses,feature_names)
    models=[]
    for seed in (20261004,20261005,20261006):
        m=lgb.LGBMRegressor(**model_params(seed))
        m.fit(X,y)
        models.append(m)
    return models,cols


def predict_df(models,X):
    ps=[np.asarray(m.predict(X),dtype=float) for m in models]
    return np.mean(np.vstack(ps),axis=0)


def prediction_rows_for_action(races,action_,feature_names,cols):
    af=action_features(action_)
    rows=[]
    for r in races:
        z={k:float(r["features"][k]) for k in feature_names}; z.update(af); rows.append(z)
    return pd.DataFrame(rows,columns=cols,dtype=np.float32)


def predict_specific(models,races,action_,feature_names,cols):
    if action_["family"]=="BASE_PL":
        return np.zeros(len(races),dtype=float)
    X=prediction_rows_for_action(races,action_,feature_names,cols)
    return predict_df(models,X)


def predict_best(models,races,candidates,feature_names,cols,batch=384):
    chosen=[]; bestvals=[]
    for start in range(0,len(races),batch):
        chunk=races[start:start+batch]
        n=len(chunk)
        best=np.zeros(n,dtype=float)
        bestname=np.array(["BASE_PL"]*n,dtype=object)
        ctx=[r["features"] for r in chunk]
        for a in candidates:
            if a["family"]=="BASE_PL": continue
            af=action_features(a)
            rows=[]
            for c in ctx:
                z={k:float(c[k]) for k in feature_names}; z.update(af); rows.append(z)
            X=pd.DataFrame(rows,columns=cols,dtype=np.float32)
            p=predict_df(models,X)
            mask=p<best
            if np.any(mask):
                best[mask]=p[mask]; bestname[mask]=a["name"]
        chosen.extend(bestname.tolist()); bestvals.extend(best.tolist())
    return chosen,np.asarray(bestvals,dtype=float)


def choose_static(val,losses):
    rows=[]
    for a in TRAIN_ACTIONS:
        vals=[losses[r["race_id"]][a["name"]]["TRIFECTA"] for r in val]
        rows.append({"action":a["name"],"nll":float(np.mean(vals))})
    rows.sort(key=lambda z:(z["nll"],z["action"]))
    return TRAIN_BY_NAME[rows[0]["action"]],rows


def action_loss(r,losses,name):
    return float(losses[r["race_id"]][name]["TRIFECTA"])


def tune_policy(models,val,losses,static,feature_names,cols):
    best_names,best_pred=predict_best(models,val,TRAIN_ACTIONS,feature_names,cols)
    static_pred=predict_specific(models,val,static,feature_names,cols)
    rows=[]; best=None
    for sm in SWITCH_MARGINS:
        for bf in BASE_FALLBACK:
            acts=[]
            for i,r in enumerate(val):
                cand=best_names[i]
                use=cand if (best_pred[i]-static_pred[i])<sm else static["name"]
                pred=best_pred[i] if use==cand else static_pred[i]
                if pred>bf: use="BASE_PL"
                acts.append(use)
            nll=float(np.mean([action_loss(r,losses,a) for r,a in zip(val,acts)]))
            base=float(np.mean([action_loss(r,losses,"BASE_PL") for r in val]))
            static_nll=float(np.mean([action_loss(r,losses,static["name"]) for r in val]))
            row={"switch_margin":sm,"base_fallback":bf,"val_nll":nll,"delta_vs_base":nll-base,"delta_vs_static":nll-static_nll,
                 "base_pct":100*sum(a=="BASE_PL" for a in acts)/len(acts),
                 "static_pct":100*sum(a==static["name"] for a in acts)/len(acts)}
            rows.append(row)
            key=(nll,abs(sm),bf)
            if best is None or key<best[0]: best=(key,row)
    return best[1],rows


def resolve_dense_name(name):
    if name=="BASE_PL": return TRAIN_BY_NAME["BASE_PL"]
    for a in DENSE_ACTIONS:
        if a["name"]==name: return a
    raise KeyError(name)


def apply_policy(models,races,static,tune,feature_names,cols):
    names,pred=predict_best(models,races,DENSE_ACTIONS,feature_names,cols)
    static_pred=predict_specific(models,races,static,feature_names,cols)
    actions=[]; selected_pred=[]
    sm=float(tune["switch_margin"]); bf=float(tune["base_fallback"])
    for i in range(len(races)):
        cand=names[i]
        use=cand if (pred[i]-static_pred[i])<sm else static["name"]
        p=pred[i] if use==cand else static_pred[i]
        if p>bf:
            use="BASE_PL"; p=0.0
        actions.append(resolve_dense_name(use)); selected_pred.append(float(p))
    return actions,np.asarray(selected_pred,dtype=float)


def oracle_train_grid(races,losses):
    out=[]
    for r in races:
        name=min((a["name"] for a in TRAIN_ACTIONS),key=lambda n:losses[r["race_id"]][n]["TRIFECTA"])
        out.append(TRAIN_BY_NAME[name])
    return out


def final_eval_one(r,policy_actions,draws):
    n=len(r["base"])
    zn=antithetic_normal(r["race_id"],draws,n,"FINAL_V3")
    zt=antithetic_student(r["race_id"],draws,n,"FINAL_V3")
    cache={}
    out=[]
    for policy,a in policy_actions.items():
        key=(a["family"],a["sigma"],a["lambda"],a.get("df"))
        if key not in cache:
            rows=rows_for_action(r,a,zn,zt)
            cache[key]=loss_for_rows(r,rows)
        for subset,active in r["subsets"].items():
            if not active: continue
            for bet in BET_TYPES:
                v=cache[key].get(bet)
                if v is not None: out.append((policy,subset,bet,float(v)))
    return out


def _final_chunk(args):
    races,maps,draws=args
    sums=defaultdict(float); counts=defaultdict(int)
    for r in races:
        pa={policy:maps[policy][r["race_id"]] for policy in maps}
        for policy,subset,bet,v in final_eval_one(r,pa,draws):
            sums[(policy,subset,bet)]+=v; counts[(policy,subset,bet)]+=1
    return sums,counts


def final_eval(races,maps,draws,workers):
    if workers<=1:
        parts=[_final_chunk((races,maps,draws))]
    else:
        chunks=[races[i::workers] for i in range(workers)]
        ctx=multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as ex:
            parts=list(ex.map(_final_chunk,[(c,maps,draws) for c in chunks if c]))
    sums=defaultdict(float); counts=defaultdict(int)
    for s,c in parts:
        for k,v in s.items(): sums[k]+=v
        for k,v in c.items(): counts[k]+=v
    base={(s,b):sums[("BASE_PL",s,b)]/counts[("BASE_PL",s,b)] for (p,s,b) in counts if p=="BASE_PL"}
    rows=[]
    for k in sorted(counts):
        p,s,b=k; n=counts[k]; val=sums[k]/n; bv=base[(s,b)]
        rows.append({"policy":p,"subset":s,"bet_type":b,"races":n,"nll":val,"base_pl_nll":bv,"delta_vs_base":val-bv})
    return rows


def dist(actions):
    c=Counter(a["name"] for a in actions); n=len(actions)
    return {k:{"count":v,"pct":100*v/n} for k,v in sorted(c.items())}


def aggregate_importance(models,feature_names):
    acc=defaultdict(float)
    for m in models:
        for name,g in zip(m.feature_name_,m.booster_.feature_importance(importance_type="gain")):
            if name.startswith("a_"): continue
            acc[name]+=float(g)
    total=sum(acc.values()) or 1.0
    return [{"feature":k,"gain":v,"gain_share":v/total} for k,v in sorted(acc.items(),key=lambda x:-x[1])]


def main():
    a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L2_PRICER_CONTINUOUS_SURFACE_V3": raise SystemExit("wrong contract")
    cp=contract["cost_policy"]
    if cp["github_standard_cpu_only"] is not True or cp["paid_runner"] or cp["gpu"] or cp["paid_artifact_or_cache"]:
        raise SystemExit("cost guard drift")
    if contract["data_policy"]["locked_years"]!=[2026]: raise SystemExit("2026 lock drift")

    manifest,df=load_dataset(a.dataset_dir)
    df["year"]=df["year"].astype(int); df["race_id"]=df["race_id"].astype(str)
    for c in set(manifest["feature_columns"])|{"mean_probability","rank_std_pct","probability_std","rank_range_pct","top1_vote_share","top3_support_share","top6_support_share","mean_rank_pct"}:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
    if 2026 in set(df["year"]): raise SystemExit("2026 sealed")

    started=time.perf_counter()
    workers=a.workers if a.workers>0 else max(1,min(4,os.cpu_count() or 2))
    outcomes=load_outcomes(df,a.backfill_root)
    races=build_races(df,outcomes,{2022,2023,2024,2025})
    feature_names=attach_features(races,df)
    print(f"SURFACE_SETUP races={len(races)} features={len(feature_names)} train_actions={len(TRAIN_ACTIONS)} dense_actions={len(DENSE_ACTIONS)} workers={workers}",flush=True)

    losses=parallel_losses(races,LABEL_DRAWS,"SURFACE_LABEL",workers)
    print("SURFACE_LABELS_READY",flush=True)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    metrics=[]; policy_rows=[]; tuning_rows=[]; static_rows=[]; importance_rows=[]; fold_summary=[]

    for test_year,train_years in FOLDS:
        train=[r for r in races if r["year"] in set(train_years)]
        test=[r for r in races if r["year"]==test_year]
        fit,val,val_rule=fit_val_split(train)
        static,sg=choose_static(val,losses)
        for z in sg: static_rows.append({"test_year":test_year,"validation_rule":val_rule,**z})

        pre_models,cols=fit_surface(fit,losses,feature_names)
        tune,trows=tune_policy(pre_models,val,losses,static,feature_names,cols)
        for z in trows: tuning_rows.append({"test_year":test_year,"validation_rule":val_rule,"static_action":static["name"],**z})
        print("SURFACE_TUNED "+json.dumps({"test_year":test_year,"static":static["name"],"validation":val_rule,**tune},separators=(",",":")),flush=True)

        models,cols=fit_surface(train,losses,feature_names)
        surface_actions,pred=apply_policy(models,test,static,tune,feature_names,cols)
        oracle_actions=oracle_train_grid(test,losses)

        maps={
            "BASE_PL":{r["race_id"]:TRAIN_BY_NAME["BASE_PL"] for r in test},
            "STATIC_PRIOR":{r["race_id"]:static for r in test},
            "SURFACE_POLICY":{r["race_id"]:ac for r,ac in zip(test,surface_actions)},
            "ORACLE_GRID_DIAGNOSTIC":{r["race_id"]:ac for r,ac in zip(test,oracle_actions)},
        }
        rows=final_eval(test,maps,FINAL_DRAWS,workers)
        for z in rows: z["test_year"]=test_year; metrics.append(z)

        d=dist(surface_actions); od=dist(oracle_actions)
        policy_rows.append({
            "test_year":test_year,"validation_rule":val_rule,"static_action":static["name"],
            "switch_margin":tune["switch_margin"],"base_fallback":tune["base_fallback"],
            "surface_base_pct":d.get("BASE_PL",{}).get("pct",0.0),
            "surface_static_pct":d.get(static["name"],{}).get("pct",0.0),
            "surface_actions_json":json.dumps(d,separators=(",",":")),
            "oracle_actions_json":json.dumps(od,separators=(",",":")),
            "mean_selected_pred_delta":float(np.mean(pred)),
        })
        for z in aggregate_importance(models,feature_names)[:30]:
            importance_rows.append({"test_year":test_year,**z})

        fs={"test_year":test_year,"static_action":static["name"]}
        for z in rows:
            if z["subset"]=="ALL" and z["bet_type"] in ("TRIFECTA","TRIO"):
                key=f"{z['policy'].lower()}_{z['bet_type'].lower()}"
                fs[f"{key}_nll"]=z["nll"]; fs[f"{key}_delta"]=z["delta_vs_base"]
        fold_summary.append(fs)
        print("SURFACE_FOLD_DONE "+json.dumps(fs,separators=(",",":")),flush=True)

    write_csv(out/"policy-nll-metrics.csv",metrics)
    write_csv(out/"policy-selection.csv",policy_rows)
    write_csv(out/"policy-tuning.csv",tuning_rows)
    write_csv(out/"static-grid.csv",static_rows)
    write_csv(out/"feature-importance.csv",importance_rows)
    write_csv(out/"fold-summary.csv",fold_summary)

    summary={
        "contract":"L2_PRICER_CONTINUOUS_SURFACE_V3_RESULT",
        "architecture":"ENSEMBLED_CONTEXT_ACTION_CONTINUOUS_LOSS_SURFACE",
        "training_actions":len(TRAIN_ACTIONS),"dense_prediction_actions":len(DENSE_ACTIONS),
        "label_draws":LABEL_DRAWS,"final_draws":FINAL_DRAWS,"parallel_workers":workers,
        "feature_count":len(feature_names),"2026_locked":True,"market_used":False,"roi_optimized":False,
        "policy_selection":policy_rows,"fold_summary":fold_summary,
        "elapsed_seconds":time.perf_counter()-started,"promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 PRICER Continuous Surface V3\n\n"
        "A pooled Huber LightGBM ensemble learns context + continuous volatility parameters -> trifecta NLL delta. "
        "The deployable policy starts from the prior-only static optimum and only switches when the learned surface predicts enough incremental benefit. "
        "Dense sigma/lambda candidates are used at prediction time. Odds, ROI, ticket selection and staking are excluded. 2026 sealed.\n",
        encoding="utf-8"
    )
    print("===== FOLD SUMMARY =====")
    print((out/"fold-summary.csv").read_text())
    print("===== POLICY SELECTION =====")
    print((out/"policy-selection.csv").read_text())
    print("L2_PRICER_CONTINUOUS_SURFACE_V3_READY")


if __name__=="__main__":
    main()
