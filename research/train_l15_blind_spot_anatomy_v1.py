#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import train_l15_outsider_rescue_router_v1 as v1

CONTRACT="L15_BLIND_SPOT_ANATOMY_V1_FOLD"
BUDGETS=(0.05,0.10,0.15,0.20,0.25)
PRIMARY_BUDGET=0.10
CANDIDATES=v1.PRIMARY+v1.SHADOW
GROUPS=("BASE","BREADTH","REDUNDANCY","CORE_FRINGE","PAIRWISE","TOP3","TOP1","ALL_ANATOMY")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--test-year",required=True,type=int)
    p.add_argument("--router-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--nonhorse-scorecard",required=True)
    p.add_argument("--out",required=True)
    return p.parse_args()

def safe_div(a,b):
    return float(a)/float(b) if b else 0.0

def support_shape(sets,field_size,prefix):
    support=Counter()
    total_slots=0
    for s in sets:
        uniq=set(map(str,s or []))
        total_slots+=len(uniq)
        support.update(uniq)
    vals=sorted(support.values(),reverse=True)
    union=len(vals)
    duplicate=max(0,total_slots-union)
    weights=[x/total_slots for x in vals] if total_slots else []
    hhi=sum(w*w for w in weights) if weights else 0.0
    entropy=0.0
    if len(weights)>1:
        entropy=-sum(w*math.log(w) for w in weights if w>0)/math.log(len(weights))
    out={
        f"{prefix}_union_size":float(union),
        f"{prefix}_union_to_field":safe_div(union,field_size),
        f"{prefix}_unseen_count":float(max(0,int(field_size or 0)-union)),
        f"{prefix}_unseen_share":safe_div(max(0,int(field_size or 0)-union),field_size),
        f"{prefix}_total_slots":float(total_slots),
        f"{prefix}_duplicate_slots":float(duplicate),
        f"{prefix}_redundancy_ratio":safe_div(duplicate,total_slots),
        f"{prefix}_support_hhi":float(hhi),
        f"{prefix}_effective_horse_count":safe_div(1.0,hhi) if hhi else 0.0,
        f"{prefix}_support_entropy":float(entropy),
        f"{prefix}_support_max":float(vals[0] if vals else 0),
        f"{prefix}_support_second":float(vals[1] if len(vals)>1 else 0),
        f"{prefix}_max_support_share":safe_div(vals[0] if vals else 0,7),
        f"{prefix}_top2_support_share":safe_div(sum(vals[:2]),14),
        f"{prefix}_support_mean":float(statistics.mean(vals)) if vals else 0.0,
        f"{prefix}_support_std":float(statistics.pstdev(vals)) if len(vals)>1 else 0.0,
    }
    for k in range(1,8):
        eq=sum(1 for x in vals if x==k)
        out[f"{prefix}_support_eq{k}_count"]=float(eq)
        out[f"{prefix}_support_eq{k}_share"]=safe_div(eq,union)
    for k in (2,3,4,5):
        ge=sum(1 for x in vals if x>=k)
        out[f"{prefix}_support_ge{k}_count"]=float(ge)
        out[f"{prefix}_support_ge{k}_share"]=safe_div(ge,union)
    singleton=sum(1 for x in vals if x==1)
    fringe=sum(1 for x in vals if x<=2)
    core=sum(1 for x in vals if x>=4)
    core_slots=sum(x for x in vals if x>=4)
    out[f"{prefix}_singleton_share"]=safe_div(singleton,union)
    out[f"{prefix}_fringe_share"]=safe_div(fringe,union)
    out[f"{prefix}_core_share"]=safe_div(core,union)
    out[f"{prefix}_singleton_slot_share"]=safe_div(singleton,total_slots)
    out[f"{prefix}_core_slot_share"]=safe_div(core_slots,total_slots)
    return out

def pairwise_features(sets,prefix):
    jacs=[]
    ints=[]
    for i,a in enumerate(sets):
        aa=set(map(str,a or []))
        for b in sets[i+1:]:
            bb=set(map(str,b or []))
            u=aa|bb
            inter=len(aa&bb)
            jacs.append(inter/len(u) if u else 1.0)
            ints.append(float(inter))
    def pack(vals,stem):
        if not vals:
            return {
                f"{prefix}_{stem}_mean":0.0,
                f"{prefix}_{stem}_min":0.0,
                f"{prefix}_{stem}_max":0.0,
                f"{prefix}_{stem}_std":0.0,
            }
        return {
            f"{prefix}_{stem}_mean":float(statistics.mean(vals)),
            f"{prefix}_{stem}_min":float(min(vals)),
            f"{prefix}_{stem}_max":float(max(vals)),
            f"{prefix}_{stem}_std":float(statistics.pstdev(vals)) if len(vals)>1 else 0.0,
        }
    out=pack(jacs,"jaccard")
    out.update(pack(ints,"intersection"))
    return out

def top1_features(horses):
    votes=Counter(map(str,horses))
    counts=sorted(votes.values(),reverse=True)
    total=sum(counts)
    weights=[x/total for x in counts] if total else []
    hhi=sum(x*x for x in weights) if weights else 0.0
    entropy=0.0
    if len(weights)>1:
        entropy=-sum(w*math.log(w) for w in weights if w>0)/math.log(len(weights))
    maxv=counts[0] if counts else 0
    second=counts[1] if len(counts)>1 else 0
    return {
        "top1_unique_horses":float(len(votes)),
        "top1_max_vote":float(maxv),
        "top1_second_vote":float(second),
        "top1_max_vote_share":safe_div(maxv,total),
        "top1_second_vote_share":safe_div(second,total),
        "top1_top2_vote_share":safe_div(maxv+second,total),
        "top1_vote_gap":safe_div(maxv-second,total),
        "top1_vote_hhi":float(hhi),
        "top1_vote_entropy":float(entropy),
        "top1_full_agreement":1.0 if maxv==7 else 0.0,
        "top1_majority_ge4":1.0 if maxv>=4 else 0.0,
        "top1_two_pole":1.0 if len(counts)>=2 and maxv+second>=6 else 0.0,
        "top1_fragmented":1.0 if len(votes)>=5 else 0.0,
    }

def anatomy_groups(rec):
    experts=rec.get("experts") or {}
    if len(experts)!=7:
        raise ValueError(f"expected 7 experts race={rec.get('race_id')}")
    race=rec.get("race") or {}
    try:
        field_size=int(float(race.get("field_size") or 0))
    except (TypeError,ValueError):
        field_size=0

    top1=[]
    top3=[]
    top6=[]
    for name in sorted(experts):
        v=experts[name] or {}
        if v.get("top1_horse_id"):
            top1.append(str(v["top1_horse_id"]))
        top3.append(v.get("top3_horse_ids") or [])
        top6.append(v.get("top6_horse_ids") or [])

    s6=support_shape(top6,field_size,"top6")
    s3=support_shape(top3,field_size,"top3")
    p6=pairwise_features(top6,"top6_pair")
    t1=top1_features(top1)

    breadth={k:v for k,v in s6.items() if any(x in k for x in (
        "union_size","union_to_field","unseen_count","unseen_share","total_slots"
    ))}
    redundancy={k:v for k,v in s6.items() if any(x in k for x in (
        "duplicate_slots","redundancy_ratio","support_hhi","effective_horse_count",
        "support_entropy","support_max","support_second","max_support_share",
        "top2_support_share","support_mean","support_std"
    ))}
    core={k:v for k,v in s6.items() if (
        "support_eq" in k or "support_ge" in k or
        "singleton_share" in k or "fringe_share" in k or "core_share" in k or
        "singleton_slot_share" in k or "core_slot_share" in k
    )}
    top3_compact={k:v for k,v in s3.items() if any(x in k for x in (
        "union_size","union_to_field","unseen_share","redundancy_ratio",
        "support_hhi","effective_horse_count","support_entropy",
        "singleton_share","core_slot_share","max_support_share"
    ))}
    return {
        "BREADTH":breadth,
        "REDUNDANCY":redundancy,
        "CORE_FRINGE":core,
        "PAIRWISE":p6,
        "TOP3":top3_compact,
        "TOP1":t1,
        "ALL_ANATOMY":{**breadth,**redundancy,**core,**p6,**top3_compact,**t1},
    }

def load_router(path):
    out={}
    with v1.open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            rec=json.loads(line)
            if rec.get("contract")!=v1.ROUTER_CONTRACT:
                raise ValueError("bad router contract")
            rid=str(rec.get("race_id") or "")
            if not rid or rid in out:
                raise ValueError(f"bad/duplicate race_id {rid}")
            out[rid]={
                "base":v1.router_features(rec),
                "groups":anatomy_groups(rec),
                "race_date":str(rec.get("race_date") or "")[:10],
            }
    return out

def build_rows(router_by_year,labels,years):
    rows=[]
    for y in years:
        rmap=router_by_year[y]
        lmap=labels[y]
        if set(rmap)!=set(lmap):
            raise ValueError(f"coverage mismatch y={y} router={len(rmap)} labels={len(lmap)}")
        for rid in sorted(rmap):
            rows.append({
                "year":y,
                "race_id":rid,
                "blind":int(lmap[rid]["blind"]),
                "base":rmap[rid]["base"],
                "groups":rmap[rid]["groups"],
            })
    return rows

def feature_dict(row,group):
    x=dict(row["base"])
    if group!="BASE":
        x.update(row["groups"][group])
    return x

def encode(train,test,group):
    tr=pd.DataFrame([feature_dict(r,group) for r in train])
    te=pd.DataFrame([feature_dict(r,group) for r in test])
    tr_cats=[c for c in tr.columns if not pd.api.types.is_numeric_dtype(tr[c])]
    te_cats=[c for c in te.columns if not pd.api.types.is_numeric_dtype(te[c])]
    tr=pd.get_dummies(tr,columns=tr_cats,dummy_na=True,dtype=float)
    te=pd.get_dummies(te,columns=te_cats,dummy_na=True,dtype=float)
    te=te.reindex(columns=tr.columns,fill_value=0.0)
    tr=tr.replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    te=te.replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    return tr.reset_index(drop=True),te.reset_index(drop=True)

def model(seed):
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

def fit_predict(train,test,group,seed):
    Xtr,Xte=encode(train,test,group)
    y=np.asarray([r["blind"] for r in train],dtype=int)
    m=model(seed)
    m.fit(Xtr,y)
    p=m.predict_proba(Xte)[:,1]
    gains=m.booster_.feature_importance(importance_type="gain")
    names=m.booster_.feature_name()
    top=sorted(zip(names,gains),key=lambda x:-x[1])[:25]
    return p,{
        "feature_count":int(Xtr.shape[1]),
        "top_features":[{"feature":n,"gain":float(g)} for n,g in top if g>0],
    }

def safe_auc(y,p):
    return float(roc_auc_score(y,p)) if len(set(map(int,y)))>1 else None

def safe_ap(y,p):
    return float(average_precision_score(y,p)) if sum(map(int,y)) else None

def select_budget(rows,scores,budget):
    n=len(rows)
    k=max(1,int(round(n*budget)))
    order=sorted(range(n),key=lambda i:(-float(scores[i]),rows[i]["race_id"]))
    sel=np.zeros(n,dtype=bool)
    sel[order[:k]]=True
    return sel

def budget_metrics(rows,sel):
    y=np.asarray([r["blind"] for r in rows],dtype=int)
    blind=int(y.sum())
    chosen=int(sel.sum())
    caught=int(((y==1)&sel).sum())
    precision=caught/chosen if chosen else None
    recall=caught/blind if blind else None
    rate=chosen/len(rows) if rows else None
    prev=blind/len(rows) if rows else None
    return {
        "races":len(rows),
        "blind_spots":blind,
        "selected":chosen,
        "intervention_rate":rate,
        "caught_blind_spots":caught,
        "blind_recall":recall,
        "precision":precision,
        "lift_vs_random_recall":recall/rate if recall is not None and rate else None,
        "enrichment_vs_prevalence":precision/prev if precision is not None and prev else None,
        "false_positives":int(((y==0)&sel).sum()),
        "false_negatives":int(((y==1)&(~sel)).sum()),
    }

def ids_for_10(rows,sel):
    y=np.asarray([r["blind"] for r in rows],dtype=int)
    return {
        "selected":[rows[i]["race_id"] for i in range(len(rows)) if sel[i]],
        "true_blind":[rows[i]["race_id"] for i in range(len(rows)) if y[i]==1],
        "caught_blind":[rows[i]["race_id"] for i in range(len(rows)) if sel[i] and y[i]==1],
        "false_positive":[rows[i]["race_id"] for i in range(len(rows)) if sel[i] and y[i]==0],
    }

def eval_group(train,test,group,seed):
    p,info=fit_predict(train,test,group,seed)
    y=[r["blind"] for r in test]
    out={
        "roc_auc":safe_auc(y,p),
        "pr_auc":safe_ap(y,p),
        "model":info,
        "budgets":{},
        "race_ids_10pct":{},
    }
    for budget in BUDGETS:
        key=f"{int(round(budget*100))}%"
        sel=select_budget(test,p,budget)
        out["budgets"][key]=budget_metrics(test,sel)
        if abs(budget-PRIMARY_BUDGET)<1e-12:
            out["race_ids_10pct"]=ids_for_10(test,sel)
    return out

def train_quantile_edges(vals,q=5):
    arr=np.asarray([float(v) for v in vals if v is not None and math.isfinite(float(v))],dtype=float)
    if len(arr)==0:
        return []
    raw=list(np.quantile(arr,np.linspace(0,1,q+1)))
    edges=[raw[0]]
    for x in raw[1:]:
        if x>edges[-1]:
            edges.append(x)
    return edges

def bin_index(x,edges):
    if x is None or not edges:
        return None
    x=float(x)
    if len(edges)==1:
        return 0
    for i in range(len(edges)-1):
        if i==len(edges)-2:
            if edges[i] <= x <= edges[i+1]:
                return i
        elif edges[i] <= x < edges[i+1]:
            return i
    if x<edges[0]: return 0
    if x>edges[-1]: return len(edges)-2
    return None

def univariate(train,test):
    keys=[
        ("BREADTH","top6_union_to_field"),
        ("BREADTH","top6_unseen_share"),
        ("REDUNDANCY","top6_redundancy_ratio"),
        ("REDUNDANCY","top6_effective_horse_count"),
        ("REDUNDANCY","top6_support_entropy"),
        ("REDUNDANCY","top6_support_hhi"),
        ("CORE_FRINGE","top6_singleton_share"),
        ("CORE_FRINGE","top6_core_slot_share"),
        ("TOP3","top3_union_to_field"),
        ("TOP1","top1_max_vote_share"),
        ("TOP1","top1_vote_entropy"),
    ]
    test_prev=sum(r["blind"] for r in test)/len(test)
    out={}
    for group,key in keys:
        trvals=[r["groups"][group].get(key) for r in train]
        edges=train_quantile_edges(trvals,5)
        bins=[]
        for i in range(max(0,len(edges)-1)):
            rows=[r for r in test if bin_index(r["groups"][group].get(key),edges)==i]
            blind=sum(r["blind"] for r in rows)
            rate=blind/len(rows) if rows else None
            bins.append({
                "bin":i+1,
                "lo":edges[i],
                "hi":edges[i+1],
                "races":len(rows),
                "blind_spots":blind,
                "blind_rate":rate,
                "lift_vs_test_prevalence":rate/test_prev if rate is not None and test_prev else None,
            })
        out[key]={"train_edges":edges,"test_bins":bins}
    return out

def main():
    a=parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8"))
    if cfg.get("contract")!="L15_BLIND_SPOT_ANATOMY_V1_EXPERIMENT":
        raise SystemExit("bad config")
    if cfg.get("ability_uses_odds") is not False or 2026 not in cfg.get("locked_years",[]):
        raise SystemExit("leakage contract broken")
    if a.test_year not in (2022,2023,2024,2025):
        raise SystemExit("invalid test year")

    years=list(range(2021,a.test_year+1))
    paths=v1.parse_year_map(a.router_year)
    if set(paths)!=set(years):
        raise SystemExit(f"router years mismatch expected={years} got={sorted(paths)}")

    score=json.loads(Path(a.nonhorse_scorecard).read_text(encoding="utf-8"))
    labels=v1.build_labels(score,years,list(CANDIDATES))
    routers={y:load_router(paths[y]) for y in years}
    rows=build_rows(routers,labels,years)
    train=[r for r in rows if r["year"]<a.test_year]
    test=[r for r in rows if r["year"]==a.test_year]

    results={}
    for i,group in enumerate(GROUPS):
        results[group]=eval_group(train,test,group,1945+i*100)

    base10=results["BASE"]["budgets"]["10%"]["blind_recall"]
    deltas={}
    for group in GROUPS:
        deltas[group]={
            "recall_delta_pp_10pct":100*(results[group]["budgets"]["10%"]["blind_recall"]-base10),
            "caught_delta_10pct":results[group]["budgets"]["10%"]["caught_blind_spots"]-results["BASE"]["budgets"]["10%"]["caught_blind_spots"],
        }

    result={
        "contract":CONTRACT,
        "experiment_id":cfg["experiment_id"],
        "test_year":a.test_year,
        "train_years":list(range(2021,a.test_year)),
        "races":len(test),
        "blind_spots":sum(r["blind"] for r in test),
        "blind_prevalence":sum(r["blind"] for r in test)/len(test),
        "odds_used":False,
        "locked_years":[2026],
        "groups":results,
        "delta_vs_base":deltas,
        "univariate":univariate(train,test),
        "notes":[
            "All group models share the same BASE pre-race features; one anatomy group is added at a time.",
            "Budget selection uses only model scores and fixed top-q%; test labels are never used to choose selected races.",
            "Univariate bin edges are learned from train-year feature distributions only.",
        ],
    }

    out=Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L15_BLIND_SPOT_ANATOMY_V1_RESULT")
    print(json.dumps(result,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
