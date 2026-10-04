#!/usr/bin/env python3
import argparse,hashlib,json,math,multiprocessing,os,time
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l2_core_v1 import load_dataset,write_csv
from run_l2_pricer_latent_race_v1 import load_outcomes,build_races,event_prob_from_rows

EPS=1e-15
FOLDS=((2023,(2022,)),(2024,(2022,2023)),(2025,(2022,2023,2024)))
BET_TYPES=("WIN","EXACTA","TRIO","TRIFECTA")
LABEL_DRAWS=192
FINAL_DRAWS=1536

ACTIONS=(
    {"name":"BASE_PL","family":"BASE_PL","sigma":0.0,"lambda":0.0,"df":None},
    {"name":"NH_020","family":"NORMAL_HOMO","sigma":0.20,"lambda":0.0,"df":None},
    {"name":"NH_040","family":"NORMAL_HOMO","sigma":0.40,"lambda":0.0,"df":None},
    {"name":"NH_060","family":"NORMAL_HOMO","sigma":0.60,"lambda":0.0,"df":None},
    {"name":"NH_080","family":"NORMAL_HOMO","sigma":0.80,"lambda":0.0,"df":None},
    {"name":"NHE_L05_S030","family":"NORMAL_HETERO","sigma":0.30,"lambda":0.50,"df":None},
    {"name":"NHE_L05_S050","family":"NORMAL_HETERO","sigma":0.50,"lambda":0.50,"df":None},
    {"name":"NHE_L15_S030","family":"NORMAL_HETERO","sigma":0.30,"lambda":1.50,"df":None},
    {"name":"NHE_L15_S050","family":"NORMAL_HETERO","sigma":0.50,"lambda":1.50,"df":None},
    {"name":"ST_L075_S030","family":"STUDENTT_HETERO","sigma":0.30,"lambda":0.75,"df":5.0},
    {"name":"ST_L075_S050","family":"STUDENTT_HETERO","sigma":0.50,"lambda":0.75,"df":5.0},
    {"name":"ST_L15_S030","family":"STUDENTT_HETERO","sigma":0.30,"lambda":1.50,"df":5.0},
    {"name":"ST_L15_S050","family":"STUDENTT_HETERO","sigma":0.50,"lambda":1.50,"df":5.0},
)
ACTION_BY_NAME={x["name"]:x for x in ACTIONS}
NONBASE=[x for x in ACTIONS if x["name"]!="BASE_PL"]
THRESHOLDS=(0.0,-0.001,-0.0025,-0.005,-0.01,-0.02,-0.04,-0.08)


def parse_args():
    p=argparse.ArgumentParser(description="L2 PRICER contextual volatility V2")
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--workers",type=int,default=0)
    return p.parse_args()


def normalize(v):
    x=np.clip(np.asarray(v,dtype=np.float64),EPS,None)
    return x/x.sum()


def seed_for(rid,kind,phase):
    raw=f"{rid}|{kind}|{phase}|20261004".encode()
    return int.from_bytes(hashlib.blake2b(raw,digest_size=8).digest(),"little") & 0x7fffffff


def antithetic_normal(rid,draws,n,phase):
    rng=np.random.default_rng(seed_for(rid,"NORMAL",phase))
    half=(draws+1)//2
    z=rng.standard_normal((half,n),dtype=np.float64)
    return np.concatenate((z,-z),axis=0)[:draws]


def antithetic_student(rid,draws,n,phase,df=5.0):
    rng=np.random.default_rng(seed_for(rid,"STUDENT5",phase))
    half=(draws+1)//2
    z=rng.standard_t(df,size=(half,n)).astype(np.float64)
    z*=math.sqrt((df-2.0)/df)
    return np.concatenate((z,-z),axis=0)[:draws]


def rows_for_action(race,action,z_normal=None,z_student=None):
    if action["family"]=="BASE_PL":
        return race["base"].reshape(1,-1)
    if action["family"].startswith("NORMAL"):
        z=z_normal
    else:
        z=z_student
    sigma=np.full(len(race["base"]),float(action["sigma"]),dtype=np.float64)
    if action["family"] in ("NORMAL_HETERO","STUDENTT_HETERO"):
        sigma*=1.0+float(action["lambda"])*race["uncertainty"]
    logw=np.log(np.clip(race["base"],EPS,None))[None,:]+z*sigma[None,:]
    logw-=np.max(logw,axis=1,keepdims=True)
    w=np.exp(logw)
    w/=np.sum(w,axis=1,keepdims=True)
    return w


def clip_prob(x):
    return min(max(float(x),EPS),1.0)


def loss_for_rows(race,rows):
    out={}
    for bet in BET_TYPES:
        p=event_prob_from_rows(rows,race["winning_idx"].get(bet),bet)
        out[bet]=None if p is None else -math.log(clip_prob(p))
    return out


def compute_action_losses_one(race,draws,phase):
    n=len(race["base"])
    zn=antithetic_normal(race["race_id"],draws,n,phase)
    zt=antithetic_student(race["race_id"],draws,n,phase)
    out={}
    for action in ACTIONS:
        rows=rows_for_action(race,action,zn,zt)
        out[action["name"]]=loss_for_rows(race,rows)
    return out


def _loss_chunk(args):
    races,draws,phase=args
    return [(r["race_id"],compute_action_losses_one(r,draws,phase)) for r in races]


def parallel_action_losses(races,draws,phase,workers):
    chunks=[races[i::workers] for i in range(workers)]
    if workers<=1:
        parts=[_loss_chunk((races,draws,phase))]
    else:
        ctx=multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as ex:
            parts=list(ex.map(_loss_chunk,[(c,draws,phase) for c in chunks if c]))
    out={}
    for part in parts:
        for rid,losses in part: out[rid]=losses
    return out


def q(x,p):
    return float(np.quantile(np.asarray(x,dtype=float),p)) if len(x) else 0.0


def make_features(sub):
    sub=sub.sort_values(["consensus_rank","horse_number"],kind="stable").reset_index(drop=True)
    p=normalize(sub["mean_probability"].to_numpy(dtype=float))
    f={
        "field_size":float(len(sub)),
        "race_entropy":float(sub["race_entropy"].iloc[0]),
        "race_top1_probability":float(sub["race_top1_probability"].iloc[0]),
        "race_top2_probability_sum":float(sub["race_top2_probability_sum"].iloc[0]),
        "race_top3_probability_sum":float(sub["race_top3_probability_sum"].iloc[0]),
        "race_top1_top2_gap":float(sub["race_top1_top2_gap"].iloc[0]),
        "race_rank_std_mean":float(sub["race_rank_std_mean"].iloc[0]),
        "race_rank_std_max":float(sub["race_rank_std_max"].iloc[0]),
        "race_probability_std_mean":float(sub["race_probability_std_mean"].iloc[0]),
        "race_probability_std_max":float(sub["race_probability_std_max"].iloc[0]),
        "prob_hhi":float(np.sum(p*p)),
        "effective_field":float(1.0/max(np.sum(p*p),EPS)),
        "top3_mass":float(np.sum(p[:3])),
        "top6_mass":float(np.sum(p[:6])),
        "top10_mass":float(np.sum(p[:10])),
        "tail11_mass":float(np.sum(p[10:])),
        "top1_top2_ratio":float(p[0]/max(p[1],EPS)) if len(p)>1 else 1.0,
    }
    cols=("rank_std_pct","probability_std","rank_range_pct","top1_vote_share","top3_support_share","top6_support_share","mean_rank_pct")
    for c in cols:
        vals=pd.to_numeric(sub[c],errors="coerce").fillna(0.0).to_numpy(dtype=float)
        f[f"{c}_mean"]=float(np.mean(vals))
        f[f"{c}_max"]=float(np.max(vals))
        f[f"{c}_q50"]=q(vals,0.50)
        f[f"{c}_q75"]=q(vals,0.75)
        f[f"{c}_q90"]=q(vals,0.90)
    for pos in range(3):
        row=sub.iloc[pos] if pos<len(sub) else None
        for c in ("mean_probability","rank_std_pct","probability_std","rank_range_pct","top1_vote_share","top3_support_share","top6_support_share","mean_rank_pct"):
            f[f"r{pos+1}_{c}"]=float(row[c]) if row is not None else 0.0
    return f


def attach_features(races,df):
    fmap={}
    for rid,sub in df.groupby("race_id",sort=False):
        fmap[str(rid)]=make_features(sub)
    for r in races:
        r["features"]=fmap[r["race_id"]]
    return sorted(next(iter(fmap.values())).keys())


def feature_frame(races,feature_names):
    return pd.DataFrame([{k:r["features"][k] for k in feature_names} for r in races],columns=feature_names)


def chronological_split(races,frac=0.75):
    q=sorted(races,key=lambda r:(r["race_date"],r["race_id"]))
    cut=max(1,min(len(q)-1,int(len(q)*frac)))
    return q[:cut],q[cut:]


def target_delta(races,losses,action):
    return np.asarray([
        losses[r["race_id"]][action]["TRIFECTA"]-losses[r["race_id"]]["BASE_PL"]["TRIFECTA"]
        for r in races
    ],dtype=float)


def model_params(seed):
    return dict(
        objective="regression_l1",
        n_estimators=260,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=90,
        subsample=0.90,
        colsample_bytree=0.80,
        reg_lambda=4.0,
        reg_alpha=0.2,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )


def fit_models(train_races,losses,feature_names):
    X=feature_frame(train_races,feature_names)
    models={}
    for i,a in enumerate(NONBASE):
        m=lgb.LGBMRegressor(**model_params(20261004+i))
        m.fit(X,target_delta(train_races,losses,a["name"]))
        models[a["name"]]=m
    return models


def predict_actions(models,races,feature_names,threshold):
    X=feature_frame(races,feature_names)
    pred={name:np.asarray(m.predict(X),dtype=float) for name,m in models.items()}
    chosen=[]
    pred_best=[]
    for i in range(len(races)):
        name,val=min(((name,float(v[i])) for name,v in pred.items()),key=lambda x:x[1])
        if val>=threshold:
            chosen.append("BASE_PL")
        else:
            chosen.append(name)
        pred_best.append(val)
    return chosen,pred_best


def mean_actual_loss(races,losses,actions):
    vals=[losses[r["race_id"]][a]["TRIFECTA"] for r,a in zip(races,actions)]
    return float(np.mean(vals))


def choose_threshold(models,val_races,losses,feature_names):
    rows=[]
    best=None
    for t in THRESHOLDS:
        actions,pred=predict_actions(models,val_races,feature_names,t)
        loss=mean_actual_loss(val_races,losses,actions)
        row={"threshold":t,"val_trifecta_nll":loss,"base_rate_pct":100*sum(a=="BASE_PL" for a in actions)/len(actions)}
        rows.append(row)
        if best is None or (loss,t) < (best["val_trifecta_nll"],best["threshold"]):
            best=row
    return best,rows


def choose_static(val_races,losses):
    rows=[]
    for a in ACTIONS:
        vals=[losses[r["race_id"]][a["name"]]["TRIFECTA"] for r in val_races]
        rows.append((float(np.mean(vals)),a["name"]))
    return min(rows)[1],rows


def oracle_actions(races,losses):
    return [
        min(ACTION_BY_NAME,key=lambda a:losses[r["race_id"]][a]["TRIFECTA"])
        for r in races
    ]


def _final_chunk(args):
    races,policy_actions,draws=args
    sums=defaultdict(float); counts=defaultdict(int)
    for r in races:
        rid=r["race_id"]; n=len(r["base"])
        zn=antithetic_normal(rid,draws,n,"FINAL")
        zt=antithetic_student(rid,draws,n,"FINAL")
        cache={}
        for policy,amap in policy_actions.items():
            action_name=amap[rid]
            if action_name not in cache:
                rows=rows_for_action(r,ACTION_BY_NAME[action_name],zn,zt)
                cache[action_name]=loss_for_rows(r,rows)
            losses=cache[action_name]
            for subset,active in r["subsets"].items():
                if not active: continue
                for bet in BET_TYPES:
                    v=losses.get(bet)
                    if v is None: continue
                    sums[(policy,subset,bet)]+=v
                    counts[(policy,subset,bet)]+=1
    return sums,counts


def final_eval(races,policy_actions,draws,workers):
    chunks=[races[i::workers] for i in range(workers)]
    if workers<=1:
        parts=[_final_chunk((races,policy_actions,draws))]
    else:
        ctx=multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as ex:
            parts=list(ex.map(_final_chunk,[(c,policy_actions,draws) for c in chunks if c]))
    sums=defaultdict(float); counts=defaultdict(int)
    for s,c in parts:
        for k,v in s.items(): sums[k]+=v
        for k,v in c.items(): counts[k]+=v
    rows=[]
    base={(subset,bet):sums[("BASE_PL",subset,bet)]/counts[("BASE_PL",subset,bet)] for subset,bet in {(k[1],k[2]) for k in sums if k[0]=="BASE_PL"}}
    for (policy,subset,bet),n in sorted(counts.items()):
        nll=sums[(policy,subset,bet)]/n
        b=base[(subset,bet)]
        rows.append({"policy":policy,"subset":subset,"bet_type":bet,"races":n,"nll":nll,"base_pl_nll":b,"delta_vs_base":nll-b})
    return rows


def action_distribution(actions):
    c=Counter(actions); n=len(actions)
    return {k:{"count":c[k],"pct":100*c[k]/n} for k in sorted(c)}


def aggregate_importance(models):
    acc=defaultdict(float)
    for m in models.values():
        for name,g in zip(m.feature_name_,m.booster_.feature_importance(importance_type="gain")):
            acc[name]+=float(g)
    total=sum(acc.values()) or 1.0
    return [{"feature":k,"gain":v,"gain_share":v/total} for k,v in sorted(acc.items(),key=lambda x:-x[1])]


def main():
    a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L2_PRICER_CONTEXTUAL_VOLATILITY_V2": raise SystemExit("wrong contract")
    cp=contract["cost_policy"]
    if cp["github_standard_cpu_only"] is not True or cp["paid_runner"] or cp["gpu"] or cp["paid_artifact_or_cache"]:
        raise SystemExit("cost guard drift")
    if contract["data_policy"]["locked_years"]!=[2026]: raise SystemExit("2026 lock drift")

    manifest,df=load_dataset(a.dataset_dir)
    df["year"]=df["year"].astype(int); df["race_id"]=df["race_id"].astype(str)
    numeric=set(manifest["feature_columns"])|{"mean_probability","rank_std_pct","probability_std","rank_range_pct","top1_vote_share","top3_support_share","top6_support_share","mean_rank_pct"}
    for c in numeric:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
    if 2026 in set(df["year"]): raise SystemExit("2026 sealed")

    started=time.perf_counter()
    workers=a.workers if a.workers>0 else max(1,min(4,os.cpu_count() or 2))
    outcomes=load_outcomes(df,a.backfill_root)
    races=build_races(df,outcomes,{2022,2023,2024,2025})
    feature_names=attach_features(races,df)
    print(f"CTX_FEATURES n={len(feature_names)} workers={workers}",flush=True)

    label_losses=parallel_action_losses(races,LABEL_DRAWS,"LABEL",workers)
    print(f"CTX_LABEL_MATRIX_READY races={len(label_losses)} actions={len(ACTIONS)}",flush=True)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    metric_rows=[]; selection_rows=[]; threshold_rows=[]; importance_rows=[]; fold_summary=[]

    for test_year,train_years in FOLDS:
        train=[r for r in races if r["year"] in set(train_years)]
        test=[r for r in races if r["year"]==test_year]
        fit,val=chronological_split(train,0.75)
        pre_models=fit_models(fit,label_losses,feature_names)
        best_thr,thr_rows=choose_threshold(pre_models,val,label_losses,feature_names)
        static_action,static_grid=choose_static(val,label_losses)
        for x in thr_rows: threshold_rows.append({"test_year":test_year,**x})
        print("CTX_POLICY_CAL "+json.dumps({"test_year":test_year,"threshold":best_thr["threshold"],"static_action":static_action,"val_nll":best_thr["val_trifecta_nll"]},separators=(",",":")),flush=True)

        models=fit_models(train,label_losses,feature_names)
        contextual,pred_best=predict_actions(models,test,feature_names,best_thr["threshold"])
        oracle=oracle_actions(test,label_losses)
        policies={
            "BASE_PL":{r["race_id"]:"BASE_PL" for r in test},
            "STATIC_PRIOR":{r["race_id"]:static_action for r in test},
            "CONTEXTUAL":{r["race_id"]:act for r,act in zip(test,contextual)},
            "ORACLE_DIAGNOSTIC":{r["race_id"]:act for r,act in zip(test,oracle)},
        }
        rows=final_eval(test,policies,FINAL_DRAWS,workers)
        for row in rows:
            row["test_year"]=test_year
            metric_rows.append(row)

        dist=action_distribution(contextual)
        odist=action_distribution(oracle)
        selection_rows.append({
            "test_year":test_year,
            "threshold":best_thr["threshold"],
            "static_action":static_action,
            "contextual_base_pct":dist.get("BASE_PL",{}).get("pct",0.0),
            "contextual_actions_json":json.dumps(dist,separators=(",",":")),
            "oracle_actions_json":json.dumps(odist,separators=(",",":")),
            "mean_predicted_best_delta":float(np.mean(pred_best)),
        })
        for row in aggregate_importance(models)[:30]:
            importance_rows.append({"test_year":test_year,**row})

        all_tri=[r for r in rows if r["subset"]=="ALL" and r["bet_type"]=="TRIFECTA"]
        all_trio=[r for r in rows if r["subset"]=="ALL" and r["bet_type"]=="TRIO"]
        fs={"test_year":test_year}
        for r in all_tri:
            fs[f"{r['policy'].lower()}_trifecta_nll"]=r["nll"]
            fs[f"{r['policy'].lower()}_trifecta_delta"]=r["delta_vs_base"]
        for r in all_trio:
            fs[f"{r['policy'].lower()}_trio_nll"]=r["nll"]
            fs[f"{r['policy'].lower()}_trio_delta"]=r["delta_vs_base"]
        fold_summary.append(fs)
        print("CTX_FOLD_DONE "+json.dumps(fs,separators=(",",":")),flush=True)

    write_csv(out/"policy-nll-metrics.csv",metric_rows)
    write_csv(out/"policy-selection.csv",selection_rows)
    write_csv(out/"threshold-search.csv",threshold_rows)
    write_csv(out/"feature-importance.csv",importance_rows)
    write_csv(out/"fold-summary.csv",fold_summary)
    summary={
        "contract":"L2_PRICER_CONTEXTUAL_VOLATILITY_V2_RESULT",
        "architecture":"FULL_INFORMATION_ACTION_VALUE_CONTEXTUAL_VOLATILITY",
        "actions":[x["name"] for x in ACTIONS],
        "label_draws":LABEL_DRAWS,"final_draws":FINAL_DRAWS,"parallel_workers":workers,
        "feature_count":len(feature_names),"2026_locked":True,"market_used":False,"roi_optimized":False,
        "fold_summary":fold_summary,"policy_selection":selection_rows,
        "elapsed_seconds":time.perf_counter()-started,"promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 PRICER Contextual Volatility V2\n\n"
        "One-shot comparison of deterministic PL, a prior-only static latent action, a learned contextual action-value policy, "
        "and a non-deployable test-outcome oracle. The contextual learner sees only pre-race frozen L1.7 diagnostics and predicts "
        "which volatility action minimizes expected trifecta log loss. Odds/ROI/ticket selection/staking are excluded. 2026 sealed.\n",
        encoding="utf-8",
    )
    print("===== FOLD SUMMARY =====")
    print((out/"fold-summary.csv").read_text())
    print("===== POLICY SELECTION =====")
    print((out/"policy-selection.csv").read_text())
    print("L2_PRICER_CONTEXTUAL_VOLATILITY_V2_READY")


if __name__=="__main__":
    main()
