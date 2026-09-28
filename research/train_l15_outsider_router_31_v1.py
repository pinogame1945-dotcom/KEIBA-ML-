#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from analyze_l15_consensus_outsider_join_v1 import LABELS
from train_l15_king_anxiety_gate_v1 import uncertainty_features, labels as blind_labels, op
from train_l15_outsider_rescue_router_v1 import router_features

CONTRACT="L15_OUTSIDER_ROUTER_31_V1_FOLD"
TARGET_LABELS=("当日傾向型","レース構造型","枠・コース型","メンバー構成型","騎手型")
CONSENSUS_KEEP={
    "top3_jaccard","top6_jaccard","top3_union","top6_union",
    "rank_diff_mean","rank_std_mean","prob_std_mean","prob_std_max",
    "top3_mass_mean","top6_mass_mean","entropy_mean","entropy_std",
    "vote_pattern","top1_unique","top1_max_vote","top1_max_vote_share",
}
ALL_TEST_BLIND=1366
GATE_TOTAL_ALERTS=1384
GATE_CAUGHT_BLIND=277
FIVE_ORACLE_TOTAL=244

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--test-year",required=True,type=int)
    p.add_argument("--router-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--scorecard",required=True)
    p.add_argument("--selected",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_year_map(items):
    out={}
    for s in items:
        y,path=s.split(":",1)
        out[int(y)]=path
    return out

def load_selected(path):
    out=defaultdict(list)
    with open(path,newline="",encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out[int(r["year"])].append({
                "year":int(r["year"]),
                "race_id":str(r["race_id"]),
                "gate_score":float(r["score"]),
                "blind_csv":str(r["blind"]).strip().lower()=="true",
            })
    return out

def rescue_sets(score):
    by_label={label:defaultdict(set) for label in TARGET_LABELS}
    for row in score.get("fold_results") or []:
        if row.get("status")!="success":
            continue
        cand=str(row.get("candidate") or "")
        label=LABELS.get(cand,cand)
        if label not in by_label:
            continue
        year=int(row.get("validation_year") or 0)
        if year not in (2021,2022,2023,2024,2025):
            continue
        ids=((row.get("rescue") or {}).get("rescue_race_ids_by_topn") or {}).get("6") or []
        by_label[label][year].update(map(str,ids))
    missing=[label for label in TARGET_LABELS if not by_label[label]]
    if missing:
        raise ValueError(f"missing target outsider labels: {missing}")
    return by_label

def compact_features(rec):
    full=router_features(rec)
    u=uncertainty_features(rec,True)
    out={f"cw_{k}":v for k,v in u.items() if k in CONSENSUS_KEEP}
    for k,v in full.items():
        if k.startswith("race_") or k.startswith("coverage_"):
            out[k]=v
    return out

def load_features(path):
    out={}
    with op(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec=json.loads(line)
            rid=str(rec.get("race_id") or "")
            if not rid:
                raise ValueError("router row missing race_id")
            if rid in out:
                raise ValueError(f"duplicate router race_id={rid}")
            out[rid]={
                "COMPACT":compact_features(rec),
                "FULL":router_features(rec),
            }
    return out

def build_rows(years,paths,blind,rescue):
    rows=[]
    for year in years:
        fmap=load_features(paths[year])
        lmap=blind[year]
        if set(fmap)!=set(lmap):
            raise ValueError(f"coverage mismatch y={year} features={len(fmap)} labels={len(lmap)}")
        for rid in sorted(fmap):
            y=int(lmap[rid])
            rescuers=tuple(label for label in TARGET_LABELS if rid in rescue[label][year]) if y else tuple()
            rows.append({
                "year":year,
                "race_id":rid,
                "blind":y,
                "rescuers":rescuers,
                "COMPACT":fmap[rid]["COMPACT"],
                "FULL":fmap[rid]["FULL"],
            })
    return rows

def encode(train,test,key):
    tr=pd.DataFrame([r[key] for r in train])
    te=pd.DataFrame([r[key] for r in test])
    tr_cats=[c for c in tr.columns if not pd.api.types.is_numeric_dtype(tr[c])]
    te_cats=[c for c in te.columns if not pd.api.types.is_numeric_dtype(te[c])]
    tr=pd.get_dummies(tr,columns=tr_cats,dummy_na=True,dtype=float)
    te=pd.get_dummies(te,columns=te_cats,dummy_na=True,dtype=float)
    te=te.reindex(columns=tr.columns,fill_value=0.0)
    tr=tr.replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    te=te.replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    return tr.reset_index(drop=True),te.reset_index(drop=True)

def make_model(seed):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=160,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=25,
        subsample=1.0,
        colsample_bytree=0.90,
        reg_lambda=2.0,
        reg_alpha=0.2,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )

def fit_binary(Xtr,Xte,y,seed):
    y=np.asarray(y,dtype=int)
    rate=float(y.mean()) if len(y) else 0.0
    if len(set(y.tolist()))<2:
        return np.full(len(Xte),rate,dtype=float),{"mode":"constant","train_rate":rate}
    m=make_model(seed)
    m.fit(Xtr,y)
    p=m.predict_proba(Xte)[:,1]
    gains=m.booster_.feature_importance(importance_type="gain")
    names=m.booster_.feature_name()
    top=sorted(zip(names,gains),key=lambda x:-x[1])[:12]
    return p,{
        "mode":"lightgbm",
        "train_rate":rate,
        "top_features":[{"feature":n,"gain":float(g)} for n,g in top if g>0],
    }

def combo_key(combo):
    return " + ".join(combo)

def all_combos():
    combos=[]
    for k in range(1,len(TARGET_LABELS)+1):
        combos.extend(itertools.combinations(TARGET_LABELS,k))
    return combos

def rescue_target(row,combo):
    return int(bool(set(row["rescuers"]) & set(combo)))

def best_fixed_combo(train_blind,k):
    best=None
    for combo in itertools.combinations(TARGET_LABELS,k):
        hit=sum(rescue_target(r,combo) for r in train_blind)
        row=(hit,tuple(combo))
        if best is None or hit>best[0] or (hit==best[0] and tuple(combo)<best[1]):
            best=row
    return best[1],best[0]

def policy_metrics(test_rows,decisions):
    # decisions: list[tuple(candidate labels)] aligned to test_rows.
    if len(test_rows)!=len(decisions):
        raise ValueError("decision length mismatch")
    blind_total=sum(r["blind"] for r in test_rows)
    rescued=0
    calls=0
    false_alert_calls=0
    true_alert_calls=0
    for r,combo in zip(test_rows,decisions):
        calls+=len(combo)
        if r["blind"]:
            true_alert_calls+=len(combo)
            if set(combo) & set(r["rescuers"]):
                rescued+=1
        else:
            false_alert_calls+=len(combo)
    alerts=len(test_rows)
    return {
        "alerts":alerts,
        "gate_caught_blind":blind_total,
        "rescued":rescued,
        "rescue_rate_within_gate_blind":rescued/blind_total if blind_total else None,
        "end_to_end_rescue_rate_all_blind":rescued/ALL_TEST_BLIND,
        "outsider_calls":calls,
        "avg_outsiders_per_alert":calls/alerts if alerts else None,
        "rescue_per_100_outsider_calls":100*rescued/calls if calls else 0.0,
        "rescue_per_100_gate_alerts":100*rescued/alerts if alerts else 0.0,
        "false_alert_calls":false_alert_calls,
        "true_alert_calls":true_alert_calls,
    }

def oracle_five(test_rows):
    return sum(1 for r in test_rows if r["blind"] and r["rescuers"])

def main():
    a=parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8"))
    if cfg.get("contract")!="L15_OUTSIDER_ROUTER_31_V1_EXPERIMENT":
        raise SystemExit("bad config")
    if cfg.get("ability_uses_odds") is not False or 2026 not in cfg.get("locked_years",[]):
        raise SystemExit("odds/2026 contract broken")
    if a.test_year not in (2022,2023,2024,2025):
        raise SystemExit("invalid test year")

    score=json.loads(Path(a.scorecard).read_text(encoding="utf-8"))
    blind=blind_labels(score)
    rescue=rescue_sets(score)
    selected=load_selected(a.selected)

    train_years=list(range(2021,a.test_year))
    needed=train_years+[a.test_year]
    paths=parse_year_map(a.router_year)
    if set(paths)!=set(needed):
        raise SystemExit(f"router years mismatch need={needed} got={sorted(paths)}")

    rows=build_rows(needed,paths,blind,rescue)
    train_blind=[r for r in rows if r["year"] in train_years and r["blind"]]
    test_all_by_id={r["race_id"]:r for r in rows if r["year"]==a.test_year}

    test_rows=[]
    for s in selected[a.test_year]:
        rid=s["race_id"]
        if rid not in test_all_by_id:
            raise ValueError(f"selected race absent from router snapshot {rid}")
        r=dict(test_all_by_id[rid])
        if bool(r["blind"]) != bool(s["blind_csv"]):
            raise ValueError(f"blind mismatch selected vs scorecard {rid}")
        r["gate_score"]=s["gate_score"]
        test_rows.append(r)

    if not test_rows:
        raise ValueError("empty selected test rows")

    # References on this fold.
    blind_in_alerts=sum(r["blind"] for r in test_rows)
    five_oracle=oracle_five(test_rows)

    fold={
        "contract":CONTRACT,
        "experiment_id":cfg["experiment_id"],
        "test_year":a.test_year,
        "train_years":train_years,
        "train_blind_rows":len(train_blind),
        "alerts":len(test_rows),
        "gate_caught_blind":blind_in_alerts,
        "five_outsider_oracle_on_gate_blind":five_oracle,
        "feature_sets":{},
        "fixed_baselines":{},
        "odds_used":False,
        "locked_years":[2026],
    }

    # Train-only fixed baselines are feature-independent.
    fixed_decisions={}
    for k in range(1,6):
        combo,train_hit=best_fixed_combo(train_blind,k)
        decisions=[combo for _ in test_rows]
        name=f"FIXED_K{k}"
        fixed_decisions[name]=decisions
        fold["fixed_baselines"][name]={
            "combo":list(combo),
            "train_rescued":train_hit,
            "train_rescue_rate":train_hit/len(train_blind) if train_blind else None,
            "test":policy_metrics(test_rows,decisions),
        }

    # Each feature set gets independent individual + 31-combination routers.
    decision_rows=[{
        "year":a.test_year,
        "race_id":r["race_id"],
        "gate_score":r["gate_score"],
        "blind":bool(r["blind"]),
        "actual_rescuers":"|".join(r["rescuers"]),
    } for r in test_rows]

    combos=all_combos()
    for fidx,feature_set in enumerate(("COMPACT","FULL")):
        Xtr,Xte=encode(train_blind,test_rows,feature_set)
        candidate_pred={}
        candidate_info={}

        # Size-1 candidate models.
        for cidx,label in enumerate(TARGET_LABELS):
            y=[int(label in r["rescuers"]) for r in train_blind]
            p,info=fit_binary(Xtr,Xte,y,10000+fidx*1000+cidx)
            candidate_pred[label]=p
            candidate_info[label]=info

        combo_pred={}
        combo_info={}
        # Reuse size-1 candidate predictions, train direct models for k>=2.
        for combo in combos:
            key=combo_key(combo)
            if len(combo)==1:
                combo_pred[combo]=candidate_pred[combo[0]]
                combo_info[key]=candidate_info[combo[0]]
            else:
                y=[rescue_target(r,combo) for r in train_blind]
                p,info=fit_binary(Xtr,Xte,y,20000+fidx*2000+len(combo)*100+combos.index(combo))
                combo_pred[combo]=p
                combo_info[key]=info

        policies={}

        # Independent top-K candidate router.
        for k in (1,2,3):
            decisions=[]
            for i in range(len(test_rows)):
                ranked=sorted(TARGET_LABELS,key=lambda label:(-float(candidate_pred[label][i]),label))
                decisions.append(tuple(ranked[:k]))
            name=f"INDIVIDUAL_TOP{k}"
            policies[name]={
                "test":policy_metrics(test_rows,decisions),
                "selection_type":"top_k_independent_candidate_scores",
            }
            for row,combo in zip(decision_rows,decisions):
                row[f"{feature_set}__{name}"]="|".join(combo)
                row[f"{feature_set}__{name}__hit"]=bool(row["blind"] and set(combo)&set(row["actual_rescuers"].split("|")))

        # Direct combo router, separately within each call-count K.
        for k in range(1,6):
            eligible=[combo for combo in combos if len(combo)==k]
            decisions=[]
            pred_scores=[]
            for i in range(len(test_rows)):
                best=max(eligible,key=lambda combo:(float(combo_pred[combo][i]),tuple(combo)))
                decisions.append(tuple(best))
                pred_scores.append(float(combo_pred[best][i]))
            name=f"COMBO_K{k}"
            policies[name]={
                "test":policy_metrics(test_rows,decisions),
                "selection_type":"direct_union_rescue_model_with_fixed_k",
                "mean_selected_combo_score":float(np.mean(pred_scores)) if pred_scores else None,
            }
            for row,combo,ps in zip(decision_rows,decisions,pred_scores):
                row[f"{feature_set}__{name}"]="|".join(combo)
                row[f"{feature_set}__{name}__score"]=ps
                actual=set(filter(None,row["actual_rescuers"].split("|")))
                row[f"{feature_set}__{name}__hit"]=bool(row["blind"] and set(combo)&actual)

        # Policy deltas vs corresponding train-only fixed baselines.
        for name,p in policies.items():
            if name.startswith("COMBO_K"):
                k=int(name.split("K")[1])
            elif name.startswith("INDIVIDUAL_TOP"):
                k=int(name.replace("INDIVIDUAL_TOP",""))
            else:
                continue
            fixed=fold["fixed_baselines"][f"FIXED_K{k}"]["test"]
            p["delta_rescued_vs_fixed"]=p["test"]["rescued"]-fixed["rescued"]
            p["delta_gate_blind_recall_pp_vs_fixed"]=100*((p["test"]["rescue_rate_within_gate_blind"] or 0)-(fixed["rescue_rate_within_gate_blind"] or 0))

        fold["feature_sets"][feature_set]={
            "feature_count":int(Xtr.shape[1]),
            "candidate_models":candidate_info,
            "combo_models":combo_info,
            "policies":policies,
        }

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(fold,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    # Compact race-level decision CSV for all Gate alerts.
    fields=["year","race_id","gate_score","blind","actual_rescuers"]
    for fs in ("COMPACT","FULL"):
        for name in ("INDIVIDUAL_TOP1","INDIVIDUAL_TOP2","INDIVIDUAL_TOP3","COMBO_K1","COMBO_K2","COMBO_K3","COMBO_K4","COMBO_K5"):
            fields.append(f"{fs}__{name}")
            if name.startswith("COMBO_K"):
                fields.append(f"{fs}__{name}__score")
            fields.append(f"{fs}__{name}__hit")
    with open(out/"decisions.csv","w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        for row in decision_rows:
            w.writerow(row)

    print("L15_OUTSIDER_ROUTER_31_RESULT")
    print(json.dumps({
        "test_year":a.test_year,
        "alerts":len(test_rows),
        "gate_caught_blind":blind_in_alerts,
        "five_oracle":five_oracle,
        "fixed":{k:v["test"] for k,v in fold["fixed_baselines"].items()},
        "compact":{k:v["test"] for k,v in fold["feature_sets"]["COMPACT"]["policies"].items()},
        "full":{k:v["test"] for k,v in fold["feature_sets"]["FULL"]["policies"].items()},
        "out_dir":str(out),
    },ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
