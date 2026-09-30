#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import build_year_frame, encode, evaluate_chosen
from run_l2_ticket_market_gap_v2 import add_market_features, market_feature_columns

YEARS=(2021,2022,2023,2024,2025)
TEST_YEARS=(2022,2023,2024,2025)

def parse_args():
    p=argparse.ArgumentParser()
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def train_probability_model(train,test,cols,seed):
    xtr,xte=encode(train,test,cols)
    y=train["hit"].astype(int).to_numpy()
    model=lgb.LGBMClassifier(
        objective="binary",
        n_estimators=320,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=4.0,
        reg_alpha=0.5,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(xtr,y)
    p=np.asarray(model.predict_proba(xte)[:,1],dtype=float)
    imp=pd.DataFrame({
        "feature":cols,
        "importance_split":model.feature_importances_,
    })
    booster=model.booster_
    gain=booster.feature_importance(importance_type="gain")
    imp["importance_gain"]=gain
    return p,imp,booster

def normalize_within_race(df,raw_col="raw_probability"):
    z=df.copy()
    denom=z.groupby(["year","race_id"])[raw_col].transform("sum")
    z["probability"]=z[raw_col]/denom.clip(lower=1e-12)
    return z

def apply_break_even_policy(df):
    z=df.copy()
    z["expected_return_mult"]=z["probability"]*z["odds"]
    z["edge"]=z["expected_return_mult"]-1.0
    return z[z["edge"]>0.0].copy()

def collect_splits(booster,test_year):
    dump=booster.dump_model()
    rows=[]
    def walk(node,tree_index):
        if "split_index" not in node:
            return
        fi=int(node["split_feature"])
        fname=dump["feature_names"][fi]
        rows.append({
            "test_year":test_year,
            "tree_index":tree_index,
            "split_index":int(node["split_index"]),
            "feature":fname,
            "threshold":str(node.get("threshold","")),
            "decision_type":str(node.get("decision_type","")),
            "split_gain":float(node.get("split_gain",0.0)),
            "internal_count":int(node.get("internal_count",0)),
        })
        walk(node["left_child"],tree_index)
        walk(node["right_child"],tree_index)
    for i,t in enumerate(dump["tree_info"]):
        walk(t["tree_structure"],i)
    return rows

def calibration_row(test,p):
    y=test["hit"].astype(int).to_numpy()
    eps=np.clip(p,1e-9,1-1e-9)
    return {
        "roc_auc":float(roc_auc_score(y,p)),
        "log_loss":float(log_loss(y,eps)),
        "brier":float(brier_score_loss(y,p)),
    }

def metrics_extra(chosen,source_races,label):
    m=evaluate_chosen(chosen,source_races,label)
    if chosen.empty:
        m.update({
            "mean_edge":None,
            "median_edge":None,
            "median_odds":None,
            "mean_tickets_per_executed_race":0.0,
        })
        return m
    executed=max(1,int(m["executed_races"]))
    m.update({
        "mean_edge":float(chosen["edge"].mean()),
        "median_edge":float(chosen["edge"].median()),
        "median_odds":float(chosen["odds"].median()),
        "mean_tickets_per_executed_race":float(len(chosen)/executed),
    })
    return m

def main():
    a=parse_args()
    paths={y:getattr(a,f"l17_{y}") for y in YEARS}
    l17={y:load_l17(paths[y],y) for y in YEARS}
    frames={y:add_market_features(build_year_frame(y,l17[y],a.backfill_root)) for y in YEARS}
    cols=market_feature_columns(frames[2022])

    fold_rows=[]
    metrics_rows=[]
    importances=[]
    split_rows=[]
    predictions={}

    for y in TEST_YEARS:
        train_years=[t for t in YEARS if t<y]
        train=pd.concat([frames[t] for t in train_years],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)

        raw_p,imp,booster=train_probability_model(train,test,cols,95000+y)
        test["raw_probability"]=raw_p
        test=normalize_within_race(test)
        chosen=apply_break_even_policy(test)
        predictions[y]=test

        c=calibration_row(test,test["probability"].to_numpy())
        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(map(str,train_years)),
            "train_pairs":len(train),
            "test_pairs":len(test),
            "source_races":int(test["race_id"].nunique()),
            "positive_pairs":int(test["hit"].sum()),
            "feature_count":len(cols),
            "probability_sum_mean":float(
                test.groupby(["year","race_id"])["probability"].sum().mean()
            ),
            **c,
        })
        source_races=int(test["race_id"].nunique())
        m=metrics_extra(chosen,source_races,f"LEARNED_BREAK_EVEN_{y}")
        m["year"]=y
        metrics_rows.append(m)

        imp["test_year"]=y
        importances.append(imp)
        split_rows.extend(collect_splits(booster,y))

    all_test=pd.concat([predictions[y] for y in TEST_YEARS],ignore_index=True)
    all_chosen=apply_break_even_policy(all_test)
    overall=metrics_extra(
        all_chosen,
        int(all_test[["year","race_id"]].drop_duplicates().shape[0]),
        "LEARNED_BREAK_EVEN_ALL",
    )

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(out/"folds.csv",index=False)
    pd.DataFrame(metrics_rows).to_csv(out/"metrics-by-year.csv",index=False)
    pd.concat(importances,ignore_index=True).to_csv(out/"feature-importance.csv",index=False)
    split_df=pd.DataFrame(split_rows)
    split_df.to_csv(out/"learned-splits.csv",index=False)

    top_splits=(
        split_df.sort_values(["test_year","split_gain"],ascending=[True,False])
        .groupby("test_year",sort=False)
        .head(30)
    )
    top_splits.to_csv(out/"top-learned-splits.csv",index=False)

    summary={
        "contract":"L2_QUINELLA_LEARNED_BOUNDARIES_V1",
        "source_variant":"ODDS_COMMA_CORRECTED",
        "manual_bins_used":False,
        "legacy_laws_seeded":False,
        "manual_rule_grid_used":False,
        "ticket_cap_used":False,
        "selection_threshold":"predicted_probability * final_odds > 1.0",
        "selection_threshold_reason":"mathematical break-even only",
        "probability_normalization":"within-race sum-to-one",
        "model":"LightGBM binary classifier; tree split boundaries learned from prior years only",
        "walk_forward":{
            "2022":[2021],
            "2023":[2021,2022],
            "2024":[2021,2022,2023],
            "2025":[2021,2022,2023,2024],
        },
        "metrics_by_year":metrics_rows,
        "overall_2022_2025":overall,
        "learned_split_count":int(len(split_df)),
        "feature_count":len(cols),
        "2025_used_for_selection":False,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )
    print("L2_QUINELLA_LEARNED_BOUNDARIES_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
