#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROUTER_CONTRACT="ROUTER_FEATURE_SNAPSHOT_V1"
OUT_CONTRACT="L15_OUTSIDER_RESCUE_ROUTER_V1_FOLD"
FORBIDDEN_TOKENS=("odds","popularity","payout","finish_position","winner_horse","actual_is_win","target")
PRIMARY=("outsider_gatecourse","outsider_raceshape","outsider_daytrend")
SHADOW=("outsider_field","outsider_jockey")
LAMBDA_PROFILES={"recall":0.10,"balanced":0.50,"precision":1.00}

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--test-year",required=True,type=int)
    p.add_argument("--router-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--nonhorse-scorecard",required=True)
    p.add_argument("--out",required=True)
    return p.parse_args()

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def finite(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None

def parse_year_map(items):
    out={}
    for spec in items:
        y,path=spec.split(":",1)
        out[int(y)]=path
    return out

def scalarize(prefix,obj,out,allow_strings=True):
    if not isinstance(obj,dict):
        return
    for k,v in obj.items():
        key=f"{prefix}_{k}" if prefix else str(k)
        low=key.lower()
        if any(tok in low for tok in FORBIDDEN_TOKENS):
            raise ValueError(f"forbidden feature key: {key}")
        if isinstance(v,bool):
            out[key]=1.0 if v else 0.0
        elif isinstance(v,(int,float)):
            x=finite(v)
            if x is not None:
                out[key]=x
        elif isinstance(v,str) and allow_strings:
            out[key]=v
        elif isinstance(v,dict):
            scalarize(key,v,out,allow_strings=allow_strings)

def jaccard(a,b):
    a=set(map(str,a or [])); b=set(map(str,b or [])); u=a|b
    return len(a&b)/len(u) if u else 1.0

def router_features(rec):
    if rec.get("contract")!=ROUTER_CONTRACT:
        raise ValueError("bad router contract")
    out={}
    scalarize("race",rec.get("race") or {},out,allow_strings=True)
    scalarize("coverage",rec.get("data_coverage") or {},out,allow_strings=False)
    scalarize("consensus",rec.get("consensus") or {},out,allow_strings=False)

    experts=rec.get("experts") or {}
    if len(experts)!=7:
        raise ValueError(f"expected 7 experts, got {len(experts)} race={rec.get('race_id')}")
    top3=[]; top6=[]
    for name in sorted(experts):
        view=experts[name] or {}
        stem="king_"+name
        for k in (
            "top1_probability","top2_probability","top3_probability",
            "top1_top2_gap","top1_top3_gap","top3_probability_mass",
            "top6_probability_mass","normalized_entropy"
        ):
            x=finite(view.get(k))
            if x is not None:
                out[f"{stem}_{k}"]=x
        scalarize(f"{stem}_family",view.get("top1_family_abs_share") or {},out,allow_strings=False)
        top3.append(view.get("top3_horse_ids") or [])
        top6.append(view.get("top6_horse_ids") or [])

    union3=set().union(*(set(map(str,x)) for x in top3))
    union6=set().union(*(set(map(str,x)) for x in top6))
    out["derived_top3_union_size"]=float(len(union3))
    out["derived_top6_union_size"]=float(len(union6))
    p3=[]; p6=[]
    for i in range(len(top3)):
        for j in range(i+1,len(top3)):
            p3.append(jaccard(top3[i],top3[j]))
            p6.append(jaccard(top6[i],top6[j]))
    out["derived_top3_pair_jaccard_min"]=min(p3) if p3 else 1.0
    out["derived_top6_pair_jaccard_min"]=min(p6) if p6 else 1.0
    out["derived_top3_pair_jaccard_std"]=statistics.pstdev(p3) if len(p3)>1 else 0.0
    out["derived_top6_pair_jaccard_std"]=statistics.pstdev(p6) if len(p6)>1 else 0.0
    return out

def load_router(path):
    out={}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            rec=json.loads(line)
            rid=str(rec.get("race_id") or "")
            if not rid:
                raise ValueError("router row missing race_id")
            if rid in out:
                raise ValueError(f"duplicate router race {rid}")
            out[rid]={
                "features":router_features(rec),
                "race_date":str(rec.get("race_date") or "")[:10],
            }
    if not out:
        raise ValueError(f"empty router: {path}")
    return out

def fold_result_map(scorecard):
    rows=scorecard.get("fold_results") or []
    out={}
    for row in rows:
        if row.get("status")!="success":
            continue
        cand=str(row.get("candidate") or "")
        year=int(row.get("validation_year") or 0)
        out[(cand,year)]=row
    return out

def classes_for(row):
    rescue=row.get("rescue") or {}
    raw=(rescue.get("four_way_race_ids_by_topn") or {}).get("6") or {}
    return {k:set(map(str,raw.get(k) or [])) for k in ("both_hit","kings_only","outsider_only","both_miss")}

def build_labels(scorecard,years,candidates):
    fmap=fold_result_map(scorecard)
    labels={}
    for year in years:
        base=None
        for cand in candidates:
            row=fmap.get((cand,year))
            if row:
                base=row
                break
        if base is None:
            raise ValueError(f"no scorecard row for year {year}")
        base_cls=classes_for(base)
        universe=set().union(*base_cls.values())
        kings_hit=base_cls["both_hit"]|base_cls["kings_only"]
        blind=universe-kings_hit
        if not universe:
            raise ValueError(f"empty label universe year={year}")
        per={rid:{"blind":int(rid in blind),"rescue":{}} for rid in universe}

        for cand in candidates:
            row=fmap.get((cand,year))
            if row is None:
                raise ValueError(f"missing candidate={cand} year={year}")
            cls=classes_for(row)
            u=set().union(*cls.values())
            kh=cls["both_hit"]|cls["kings_only"]
            if u!=universe or kh!=kings_hit:
                raise ValueError(f"seven-king universe mismatch candidate={cand} year={year}")
            rescue=set(map(str,((row.get("rescue") or {}).get("rescue_race_ids_by_topn") or {}).get("6") or []))
            if rescue!=(cls["outsider_only"]):
                raise ValueError(f"rescue classification mismatch candidate={cand} year={year}")
            for rid in universe:
                per[rid]["rescue"][cand]=int(rid in rescue)
        labels[year]=per
    return labels

def build_rows(router_by_year,labels,years):
    rows=[]
    for y in years:
        rmap=router_by_year[y]
        lmap=labels[y]
        if set(rmap)!=set(lmap):
            raise ValueError(f"race coverage mismatch y{y} router={len(rmap)} labels={len(lmap)} common={len(set(rmap)&set(lmap))}")
        for rid in sorted(rmap):
            rows.append({
                "year":y,
                "race_id":rid,
                "race_date":rmap[rid]["race_date"],
                "x":rmap[rid]["features"],
                "blind":int(lmap[rid]["blind"]),
                "rescue":dict(lmap[rid]["rescue"]),
            })
    return rows

def encode(train_rows,test_rows):
    tr=pd.DataFrame([r["x"] for r in train_rows])
    te=pd.DataFrame([r["x"] for r in test_rows])
    cat=[c for c in tr.columns if not pd.api.types.is_numeric_dtype(tr[c])]
    tr=pd.get_dummies(tr,columns=cat,dummy_na=True,dtype=float)
    te_cat=[c for c in te.columns if not pd.api.types.is_numeric_dtype(te[c])]
    te=pd.get_dummies(te,columns=te_cat,dummy_na=True,dtype=float)
    te=te.reindex(columns=tr.columns,fill_value=0.0)
    tr=tr.replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    te=te.replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    return tr,te

def make_model(seed=1945):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=220,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=80,
        subsample=1.0,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        reg_alpha=0.2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )

def fit_predict_binary(train_rows,test_rows,target_getter,seed=1945):
    Xtr,Xte=encode(train_rows,test_rows)
    y=np.array([int(target_getter(r)) for r in train_rows],dtype=int)
    if len(set(y.tolist()))<2:
        p=float(y.mean()) if len(y) else 0.0
        return np.full(len(test_rows),p),{"mode":"constant","rate":p,"features":int(Xtr.shape[1])}
    m=make_model(seed)
    m.fit(Xtr,y)
    pred=m.predict_proba(Xte)[:,1]
    gain=m.booster_.feature_importance(importance_type="gain")
    names=m.booster_.feature_name()
    top=sorted(zip(names,gain),key=lambda x:-x[1])[:20]
    info={"mode":"lightgbm","features":int(Xtr.shape[1]),"top_features":[{"feature":n,"gain":float(g)} for n,g in top if g>0]}
    return pred,info

def oof_gate_predictions(train_rows):
    X,_=encode(train_rows,train_rows)
    y=np.array([r["blind"] for r in train_rows],dtype=int)
    counts=np.bincount(y,minlength=2)
    minority=int(min(counts))
    if len(set(y.tolist()))<2 or minority<2:
        return np.full(len(train_rows),float(y.mean()) if len(y) else 0.0)
    n_splits=min(5,minority)
    skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=1945)
    out=np.zeros(len(train_rows),dtype=float)
    for k,(a,b) in enumerate(skf.split(X,y)):
        m=make_model(1945+k)
        m.fit(X.iloc[a],y[a])
        out[b]=m.predict_proba(X.iloc[b])[:,1]
    return out

def safe_auc(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    if len(set(y.tolist()))<2:
        return None
    return float(roc_auc_score(y,p))

def safe_ap(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    if y.sum()==0:
        return None
    return float(average_precision_score(y,p))

def learn_threshold(y,p,lam):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    candidates=sorted(set([0.0,1.000001]+[float(x) for x in np.quantile(p,np.linspace(0.02,0.98,49))]))
    best=None
    for t in candidates:
        g=p>=t
        tp=int(((y==1)&g).sum())
        fp=int(((y==0)&g).sum())
        utility=tp-lam*fp
        rec=tp/int((y==1).sum()) if int((y==1).sum()) else 0.0
        row=(utility,rec,-fp,t,tp,fp)
        if best is None or row>best:
            best=row
    _,_,_,t,tp,fp=best
    return {"threshold":float(t),"train_tp":int(tp),"train_fp":int(fp),"lambda":float(lam)}

def gate_metrics(y,p,threshold):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    g=p>=threshold
    tp=int(((y==1)&g).sum()); fp=int(((y==0)&g).sum())
    fn=int(((y==1)&(~g)).sum()); tn=int(((y==0)&(~g)).sum())
    return {
        "races":int(len(y)),
        "blind_spots":int((y==1).sum()),
        "blind_prevalence":float((y==1).mean()) if len(y) else None,
        "roc_auc":safe_auc(y,p),
        "pr_auc":safe_ap(y,p),
        "threshold":float(threshold),
        "precision":tp/(tp+fp) if tp+fp else None,
        "recall":tp/(tp+fn) if tp+fn else None,
        "false_positive_rate":fp/(fp+tn) if fp+tn else None,
        "intervention_rate":float(g.mean()) if len(g) else None,
        "tp":tp,"fp":fp,"fn":fn,"tn":tn,
    }

def heuristic_score(rows):
    out=[]
    for r in rows:
        x=r["x"]
        jac=finite(x.get("consensus_top6_pairwise_jaccard_mean"))
        if jac is None:
            jac=finite(x.get("derived_top6_pair_jaccard_min"))
        out.append(1.0-(jac if jac is not None else 0.5))
    return np.array(out,dtype=float)

def best_fixed_primary(train_rows):
    blind=[r for r in train_rows if r["blind"]]
    rates={}
    for c in PRIMARY:
        rates[c]=sum(r["rescue"][c] for r in blind)/len(blind) if blind else 0.0
    best=sorted(PRIMARY,key=lambda c:(-rates[c],c))[0]
    return best,rates

def eval_policy(test_rows,gate_mask,outsider_probs,selected_n):
    blind_total=sum(r["blind"] for r in test_rows)
    rescues=0; interventions=0; outsider_calls=0
    choice_counts=defaultdict(int)
    blind_interventions=0; normal_interventions=0
    for i,r in enumerate(test_rows):
        if not gate_mask[i]:
            continue
        interventions+=1
        if r["blind"]: blind_interventions+=1
        else: normal_interventions+=1
        order=sorted(PRIMARY,key=lambda c:(-outsider_probs[c][i],c))[:selected_n]
        outsider_calls+=len(order)
        for c in order: choice_counts[c]+=1
        if r["blind"] and any(r["rescue"][c] for c in order):
            rescues+=1
    return {
        "blind_spots":blind_total,
        "rescued_blind_spots":rescues,
        "blind_rescue_rate":rescues/blind_total if blind_total else None,
        "remaining_blind_spots":blind_total-rescues,
        "interventions":interventions,
        "intervention_rate":interventions/len(test_rows) if test_rows else None,
        "blind_interventions":blind_interventions,
        "normal_interventions":normal_interventions,
        "avg_outsiders_called_per_race":outsider_calls/len(test_rows) if test_rows else None,
        "avg_outsiders_called_per_intervention":outsider_calls/interventions if interventions else 0.0,
        "rescue_per_100_interventions":100*rescues/interventions if interventions else 0.0,
        "choice_counts":dict(sorted(choice_counts.items())),
    }

def eval_always(test_rows,selected):
    blind_total=sum(r["blind"] for r in test_rows)
    rescues=sum(1 for r in test_rows if r["blind"] and any(r["rescue"][c] for c in selected))
    return {
        "blind_spots":blind_total,
        "rescued_blind_spots":rescues,
        "blind_rescue_rate":rescues/blind_total if blind_total else None,
        "remaining_blind_spots":blind_total-rescues,
        "interventions":len(test_rows),
        "intervention_rate":1.0,
        "normal_interventions":sum(1 for r in test_rows if not r["blind"]),
        "avg_outsiders_called_per_race":float(len(selected)),
        "rescue_per_100_interventions":100*rescues/len(test_rows) if test_rows else 0.0,
        "selected":list(selected),
    }

def shadow_diag(test_rows):
    blind=[r for r in test_rows if r["blind"]]
    primary_union=sum(1 for r in blind if any(r["rescue"][c] for c in PRIMARY))
    shadow_union=sum(1 for r in blind if any(r["rescue"][c] for c in SHADOW))
    incremental=sum(1 for r in blind if (not any(r["rescue"][c] for c in PRIMARY)) and any(r["rescue"][c] for c in SHADOW))
    return {
        "blind_spots":len(blind),
        "primary3_union_rescues":primary_union,
        "primary3_union_rate":primary_union/len(blind) if blind else None,
        "shadow2_union_rescues":shadow_union,
        "shadow_incremental_beyond_primary3":incremental,
        "shadow_incremental_rate":incremental/len(blind) if blind else None,
    }

def calibration_bins(y,p,bins=10):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    out=[]
    for lo in np.linspace(0,1,bins+1)[:-1]:
        hi=lo+1/bins
        mask=(p>=lo)&(p<(hi if hi<1 else 1.000001))
        if not mask.any(): continue
        out.append({
            "lo":float(lo),"hi":float(min(1,hi)),"n":int(mask.sum()),
            "mean_p":float(p[mask].mean()),"actual_rate":float(y[mask].mean()),
        })
    return out

def main():
    a=parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8"))
    if cfg.get("contract")!="L15_OUTSIDER_RESCUE_ROUTER_EXPERIMENT_V1":
        raise SystemExit("bad config")
    if cfg.get("ability_uses_odds") is not False or 2026 not in cfg.get("locked_years",[]):
        raise SystemExit("odds/lock contract broken")
    if a.test_year not in (2022,2023,2024,2025):
        raise SystemExit("invalid test year")

    train_years=list(range(2021,a.test_year))
    needed=set(train_years+[a.test_year])
    router_paths=parse_year_map(a.router_year)
    if set(router_paths)!=needed:
        raise SystemExit(f"router years mismatch need={sorted(needed)} got={sorted(router_paths)}")

    scorecard=json.loads(Path(a.nonhorse_scorecard).read_text(encoding="utf-8"))
    candidates=list(PRIMARY+SHADOW)
    labels=build_labels(scorecard,sorted(needed),candidates)
    router_by_year={y:load_router(router_paths[y]) for y in sorted(needed)}
    rows=build_rows(router_by_year,labels,sorted(needed))
    train=[r for r in rows if r["year"] in train_years]
    test=[r for r in rows if r["year"]==a.test_year]

    # Stage A: model and train-only OOF thresholds.
    gate_pred,gate_info=fit_predict_binary(train,test,lambda r:r["blind"],seed=1945)
    gate_oof=oof_gate_predictions(train)
    ytr=np.array([r["blind"] for r in train],dtype=int)
    yte=np.array([r["blind"] for r in test],dtype=int)
    thresholds={name:learn_threshold(ytr,gate_oof,lam) for name,lam in LAMBDA_PROFILES.items()}

    heuristic=heuristic_score(test)
    stage_a={
        "model":gate_info,
        "test_roc_auc":safe_auc(yte,gate_pred),
        "test_pr_auc":safe_ap(yte,gate_pred),
        "blind_prevalence":float(yte.mean()),
        "heuristic_disagreement_roc_auc":safe_auc(yte,heuristic),
        "heuristic_disagreement_pr_auc":safe_ap(yte,heuristic),
        "calibration_bins":calibration_bins(yte,gate_pred),
        "profiles":{},
    }
    for name,pol in thresholds.items():
        stage_a["profiles"][name]={
            "train_policy":pol,
            "test":gate_metrics(yte,gate_pred,pol["threshold"]),
        }

    # Stage B: independent rescue probabilities, trained only on blind historical rows.
    train_blind=[r for r in train if r["blind"]]
    outsider_probs={}
    outsider_models={}
    for j,c in enumerate(candidates):
        p,info=fit_predict_binary(train_blind,test,lambda r,c=c:r["rescue"][c],seed=2100+j)
        outsider_probs[c]=p
        outsider_models[c]=info

    best_fixed,train_rates=best_fixed_primary(train)
    policies={
        "P0":{"name":"seven_kings_only","blind_spots":int(yte.sum()),"rescued_blind_spots":0,
              "blind_rescue_rate":0.0,"remaining_blind_spots":int(yte.sum()),"interventions":0,
              "intervention_rate":0.0,"avg_outsiders_called_per_race":0.0},
        "P1":eval_always(test,[best_fixed]),
        "P2":eval_always(test,list(PRIMARY)),
        "P3":{},
        "P4":{},
    }
    policies["P1"]["name"]="always_best_fixed_primary"
    policies["P1"]["train_best_fixed"]=best_fixed
    policies["P1"]["train_primary_rescue_rates"]=train_rates
    policies["P2"]["name"]="always_all_three_primary"

    for prof,pol in thresholds.items():
        mask=gate_pred>=pol["threshold"]
        policies["P3"][prof]=eval_policy(test,mask,outsider_probs,1)
        policies["P4"][prof]=eval_policy(test,mask,outsider_probs,2)

    blind_test=[r for r in test if r["blind"]]
    candidate_test_rates={
        c:(sum(r["rescue"][c] for r in blind_test)/len(blind_test) if blind_test else None)
        for c in candidates
    }
    result={
        "contract":OUT_CONTRACT,
        "experiment_id":cfg["experiment_id"],
        "test_year":a.test_year,
        "train_years":train_years,
        "races":len(test),
        "blind_spots":int(yte.sum()),
        "odds_used":False,
        "locked_years":[2026],
        "feature_contract":ROUTER_CONTRACT,
        "feature_scope":"router_snapshot_v1_only",
        "stage_a":stage_a,
        "stage_b":{
            "models":outsider_models,
            "test_rescue_rates_on_true_blind_spots":candidate_test_rates,
            "shadow":shadow_diag(test),
        },
        "policies":policies,
        "candidate_inflation":{
            "status":"pending_additive_integration",
            "reason":"Outsider Top6 horse lists were intentionally ephemeral in source run; routing feasibility is evaluated first without recomputation.",
        },
        "reference":{
            "all_2021_2025_blind_spots":1735,
            "all_11_hindsight_union_rescues":1575,
            "all_11_hindsight_oracle_rate":1575/1735,
            "all_11_unresolved":160,
            "warning":"ORACLE / HINDSIGHT ONLY, not model performance",
        },
    }
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L15_OUTSIDER_RESCUE_ROUTER_RESULT")
    print(json.dumps(result,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
