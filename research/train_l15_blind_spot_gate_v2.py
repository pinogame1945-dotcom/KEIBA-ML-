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
from sklearn.model_selection import StratifiedKFold

import train_l15_outsider_rescue_router_v1 as v1

CONTRACT="L15_BLIND_SPOT_GATE_V2_FOLD"
PRIMARY_BUDGET=0.10
BUDGETS=(0.05,0.10,0.15,0.20,0.25)
CANDIDATES=v1.PRIMARY+v1.SHADOW

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--test-year",required=True,type=int)
    p.add_argument("--router-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--nonhorse-scorecard",required=True)
    p.add_argument("--out",required=True)
    return p.parse_args()

def norm_entropy(counts):
    vals=[int(x) for x in counts if int(x)>0]
    if len(vals)<=1:
        return 0.0
    s=sum(vals)
    q=[x/s for x in vals]
    return -sum(x*math.log(x) for x in q)/math.log(len(q))

def hhi(counts):
    vals=[int(x) for x in counts if int(x)>0]
    s=sum(vals)
    if not s:
        return 0.0
    return sum((x/s)**2 for x in vals)

def support_features(prefix,sets,out,field_size):
    support=Counter()
    for s in sets:
        support.update(set(map(str,s or [])))
    vals=sorted(support.values(),reverse=True)
    union=len(vals)
    out[f"pattern_{prefix}_union_size"]=float(union)
    out[f"pattern_{prefix}_support_max"]=float(max(vals) if vals else 0)
    out[f"pattern_{prefix}_support_mean"]=float(sum(vals)/len(vals)) if vals else 0.0
    out[f"pattern_{prefix}_support_std"]=float(statistics.pstdev(vals)) if len(vals)>1 else 0.0
    out[f"pattern_{prefix}_support_hhi"]=float(hhi(vals))
    out[f"pattern_{prefix}_support_entropy"]=float(norm_entropy(vals))
    for k in range(1,8):
        out[f"pattern_{prefix}_support_eq{k}_horses"]=float(sum(1 for x in vals if x==k))
    out[f"pattern_{prefix}_support_ge4_horses"]=float(sum(1 for x in vals if x>=4))
    out[f"pattern_{prefix}_support_le2_horses"]=float(sum(1 for x in vals if x<=2))
    out[f"pattern_{prefix}_unanimous_horses"]=float(sum(1 for x in vals if x==7))
    if field_size and field_size>0:
        out[f"pattern_{prefix}_union_to_field"]=float(union/field_size)

def pattern_features(rec):
    experts=rec.get("experts") or {}
    if len(experts)!=7:
        raise ValueError(f"expected seven experts race={rec.get('race_id')}")
    out={}
    top1=[]
    top3=[]
    top6=[]
    for name in sorted(experts):
        v=experts[name] or {}
        h=v.get("top1_horse_id")
        if h:
            top1.append(str(h))
        top3.append(list(map(str,v.get("top3_horse_ids") or [])))
        top6.append(list(map(str,v.get("top6_horse_ids") or [])))

    votes=Counter(top1)
    counts=sorted(votes.values(),reverse=True)
    max_vote=counts[0] if counts else 0
    second=counts[1] if len(counts)>1 else 0
    out["pattern_top1_unique_horses"]=float(len(votes))
    out["pattern_top1_max_vote"]=float(max_vote)
    out["pattern_top1_second_vote"]=float(second)
    out["pattern_top1_max_vote_share"]=max_vote/7.0
    out["pattern_top1_second_vote_share"]=second/7.0
    out["pattern_top1_top2_vote_share"]=(max_vote+second)/7.0
    out["pattern_top1_vote_gap"]=(max_vote-second)/7.0
    out["pattern_top1_vote_hhi"]=float(hhi(counts))
    out["pattern_top1_vote_entropy"]=float(norm_entropy(counts))
    out["pattern_top1_vote_pattern"]="-".join(map(str,counts)) if counts else "none"
    for k in range(1,8):
        out[f"pattern_top1_groups_size{k}"]=float(sum(1 for x in counts if x==k))

    race=rec.get("race") or {}
    try:
        field_size=float(race.get("field_size"))
    except (TypeError,ValueError):
        field_size=None
    support_features("top3",top3,out,field_size)
    support_features("top6",top6,out,field_size)

    # Explicit concentration / fragmentation interactions.
    out["pattern_top1_full_agreement"]=1.0 if max_vote==7 else 0.0
    out["pattern_top1_majority_ge4"]=1.0 if max_vote>=4 else 0.0
    out["pattern_top1_two_pole"]=1.0 if len(counts)>=2 and (max_vote+second)>=6 else 0.0
    out["pattern_top1_fragmented"]=1.0 if len(votes)>=5 else 0.0
    out["pattern_top6_narrow_union"]=1.0 if out.get("pattern_top6_union_size",99)<=8 else 0.0
    out["pattern_top6_wide_union"]=1.0 if out.get("pattern_top6_union_size",0)>=12 else 0.0
    return out

def load_router_dual(path):
    out={}
    with v1.open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec=json.loads(line)
            if rec.get("contract")!=v1.ROUTER_CONTRACT:
                raise ValueError("bad router contract")
            rid=str(rec.get("race_id") or "")
            if not rid or rid in out:
                raise ValueError(f"bad/duplicate race_id {rid}")
            base=v1.router_features(rec)
            pat=dict(base)
            pat.update(pattern_features(rec))
            out[rid]={
                "x_base":base,
                "x_pattern":pat,
                "race_date":str(rec.get("race_date") or "")[:10],
            }
    return out

def build_rows(router_by_year,labels,years):
    rows=[]
    for y in years:
        rmap=router_by_year[y]
        lmap=labels[y]
        if set(rmap)!=set(lmap):
            raise ValueError(f"coverage mismatch y{y} router={len(rmap)} labels={len(lmap)}")
        for rid in sorted(rmap):
            rows.append({
                "year":y,
                "race_id":rid,
                "race_date":rmap[rid]["race_date"],
                "x_base":rmap[rid]["x_base"],
                "x_pattern":rmap[rid]["x_pattern"],
                "blind":int(lmap[rid]["blind"]),
            })
    return rows

def encode(train,test,key):
    tr=pd.DataFrame([r[key] for r in train])
    te=pd.DataFrame([r[key] for r in test])
    both=pd.concat([tr,te],ignore_index=True)
    cats=[c for c in both.columns if not pd.api.types.is_numeric_dtype(both[c])]
    both=pd.get_dummies(both,columns=cats,dummy_na=True,dtype=float)
    both=both.replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    return both.iloc[:len(tr)].reset_index(drop=True),both.iloc[len(tr):].reset_index(drop=True)

def model(seed):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=260,
        learning_rate=0.025,
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

def fit_predict(train,test,key,seed):
    Xtr,Xte=encode(train,test,key)
    y=np.asarray([r["blind"] for r in train],dtype=int)
    m=model(seed)
    m.fit(Xtr,y)
    p=m.predict_proba(Xte)[:,1]
    gains=m.booster_.feature_importance(importance_type="gain")
    names=m.booster_.feature_name()
    top=sorted(zip(names,gains),key=lambda x:-x[1])[:30]
    return p,{
        "feature_count":int(Xtr.shape[1]),
        "top_features":[{"feature":n,"gain":float(g)} for n,g in top if g>0],
    }

def oof_predict(rows,key,seed):
    X,_=encode(rows,rows,key)
    y=np.asarray([r["blind"] for r in rows],dtype=int)
    counts=np.bincount(y,minlength=2)
    n_splits=min(5,int(min(counts)))
    if n_splits<2:
        return np.full(len(rows),float(y.mean()))
    skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    p=np.zeros(len(rows),dtype=float)
    for i,(a,b) in enumerate(skf.split(X,y)):
        m=model(seed+i)
        m.fit(X.iloc[a],y[a])
        p[b]=m.predict_proba(X.iloc[b])[:,1]
    return p

def auc(y,p):
    return float(roc_auc_score(y,p)) if len(set(map(int,y)))>1 else None

def ap(y,p):
    return float(average_precision_score(y,p)) if sum(map(int,y)) else None

def metrics(rows,scores,selected):
    y=np.asarray([r["blind"] for r in rows],dtype=int)
    sel=np.asarray(selected,dtype=bool)
    n=len(rows)
    blind=int(y.sum())
    picked=int(sel.sum())
    caught=int(((y==1)&sel).sum())
    fp=int(((y==0)&sel).sum())
    fn=blind-caught
    rate=picked/n if n else 0.0
    prev=blind/n if n else 0.0
    precision=caught/picked if picked else None
    recall=caught/blind if blind else None
    return {
        "races":n,
        "blind_spots":blind,
        "selected":picked,
        "intervention_rate":rate,
        "caught_blind_spots":caught,
        "blind_recall":recall,
        "precision":precision,
        "enrichment_vs_prevalence":(precision/prev) if precision is not None and prev else None,
        "lift_vs_random_recall":(recall/rate) if recall is not None and rate else None,
        "false_positives":fp,
        "false_negatives":fn,
    }

def select_test_rank(rows,scores,budget):
    n=len(rows)
    k=max(1,int(round(n*budget)))
    order=sorted(range(n),key=lambda i:(-float(scores[i]),rows[i]["race_id"]))
    sel=np.zeros(n,dtype=bool)
    sel[order[:k]]=True
    return sel

def train_quantile_threshold(train_scores,budget):
    # Train-only deployable threshold. Higher score = more dangerous.
    return float(np.quantile(np.asarray(train_scores,dtype=float),1.0-budget))

def race_ids(rows,sel):
    y=np.asarray([r["blind"] for r in rows],dtype=int)
    selected=[rows[i]["race_id"] for i in range(len(rows)) if sel[i]]
    caught=[rows[i]["race_id"] for i in range(len(rows)) if sel[i] and y[i]==1]
    fp=[rows[i]["race_id"] for i in range(len(rows)) if sel[i] and y[i]==0]
    true_blind=[rows[i]["race_id"] for i in range(len(rows)) if y[i]==1]
    return {
        "selected":selected,
        "true_blind":true_blind,
        "caught_blind":caught,
        "false_positive":fp,
    }

def evaluate_model(train,test,key,label,seed):
    test_p,info=fit_predict(train,test,key,seed)
    train_oof=oof_predict(train,key,seed+100)
    yte=[r["blind"] for r in test]

    out={
        "label":label,
        "roc_auc":auc(yte,test_p),
        "pr_auc":ap(yte,test_p),
        "model":info,
        "test_rank":{},
        "train_quantile":{},
        "race_ids_10pct":{},
    }
    for budget in BUDGETS:
        b=f"{int(round(budget*100))}%"
        sel_rank=select_test_rank(test,test_p,budget)
        m_rank=metrics(test,test_p,sel_rank)
        out["test_rank"][b]=m_rank

        thr=train_quantile_threshold(train_oof,budget)
        sel_trainq=np.asarray(test_p)>=thr
        m_q=metrics(test,test_p,sel_trainq)
        m_q["train_oof_threshold"]=thr
        m_q["target_budget"]=budget
        out["train_quantile"][b]=m_q

        if abs(budget-PRIMARY_BUDGET)<1e-9:
            out["race_ids_10pct"]={
                "test_rank":race_ids(test,sel_rank),
                "train_quantile":race_ids(test,sel_trainq),
            }
    return out

def main():
    a=parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8"))
    if cfg.get("contract")!="L15_BLIND_SPOT_GATE_V2_EXPERIMENT":
        raise SystemExit("bad config")
    if cfg.get("ability_uses_odds") is not False or 2026 not in cfg.get("locked_years",[]):
        raise SystemExit("cost/leakage contract broken")
    if a.test_year not in (2022,2023,2024,2025):
        raise SystemExit("invalid test year")

    years=list(range(2021,a.test_year+1))
    paths=v1.parse_year_map(a.router_year)
    if set(paths)!=set(years):
        raise SystemExit(f"router years mismatch expected={years} got={sorted(paths)}")

    score=json.loads(Path(a.nonhorse_scorecard).read_text(encoding="utf-8"))
    labels=v1.build_labels(score,years,list(CANDIDATES))
    routers={y:load_router_dual(paths[y]) for y in years}
    rows=build_rows(routers,labels,years)
    train=[r for r in rows if r["year"]<a.test_year]
    test=[r for r in rows if r["year"]==a.test_year]

    base=evaluate_model(train,test,"x_base","BASE",1945)
    pattern=evaluate_model(train,test,"x_pattern","PATTERN",2945)

    comparison={}
    for b in ("5%","10%","15%","20%","25%"):
        comparison[b]={
            "test_rank_recall_delta_pp":100*((pattern["test_rank"][b]["blind_recall"] or 0)-(base["test_rank"][b]["blind_recall"] or 0)),
            "train_quantile_recall_delta_pp":100*((pattern["train_quantile"][b]["blind_recall"] or 0)-(base["train_quantile"][b]["blind_recall"] or 0)),
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
        "budgets":list(BUDGETS),
        "primary_budget":PRIMARY_BUDGET,
        "base":base,
        "pattern":pattern,
        "comparison":comparison,
        "notes":[
            "TEST_RANK is a ranking diagnostic using test-year scores only; labels are never used to choose selected races.",
            "TRAIN_QUANTILE uses only train-year OOF score quantiles to set thresholds and is the deployable-style evaluation.",
            "PATTERN adds explicit seven-king concentration / polarization / support-shape features.",
        ],
    }
    out=Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L15_BLIND_SPOT_GATE_V2_RESULT")
    print(json.dumps(result,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
