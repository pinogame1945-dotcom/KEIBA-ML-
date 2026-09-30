#!/usr/bin/env python3
import argparse
import gzip
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import (
    build_year_frame, feature_columns, encode, evaluate_chosen, write_csv
)
from run_l2_ticket_market_gap_v2 import add_market_features

YEARS=(2022,2023,2024,2025)
DEV_YEARS=(2023,2024)
CONFIRM_YEAR=2025
MIN_DEV_COVERAGE_PCT=20.0
MIN_DELTA=(0.0,0.25,0.50,0.75,1.0,1.5,2.0)
MAX_TICKETS=(1,2,3,5)

MARKET_CONTEXT_BASE=[
    "market_q_norm","market_log_q","market_rank",
    "race_month","venue_code","discipline","surface","direction","weather",
    "track_condition","course_layout","race_class_normalized","grade",
    "sex_condition","weight_rule","distance_m","field_size",
]

def args():
    p=argparse.ArgumentParser(description="V3 market-residual correction model.")
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def logit(p):
    x=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    return np.log(x/(1-x))

def add_relative_features(df):
    z=df.copy()
    ordered=z.sort_values(
        ["year","race_id","pair_prob_product","pair_rank_sum","pair_numbers"],
        ascending=[True,True,False,True,True]
    )
    r=ordered.groupby(["year","race_id"],sort=False).cumcount()+1
    z.loc[ordered.index,"l17_pair_rank"]=r.to_numpy()
    z["l17_pair_rank"]=z["l17_pair_rank"].astype(int)
    z["l17_market_rank_gap"]=z["market_rank"]-z["l17_pair_rank"]
    z["log_l17_pair_prob"]=np.log(z["pair_prob_product"].clip(lower=1e-12))
    z["l17_market_log_gap"]=z["log_l17_pair_prob"]-z["market_log_q"]
    return z

def full_feature_columns(df):
    base_source=df.drop(
        columns=[
            "market_inv_odds","market_q_norm","market_log_q","market_rank",
            "l17_pair_rank","l17_market_rank_gap","log_l17_pair_prob","l17_market_log_gap"
        ],
        errors="ignore"
    )
    cols=feature_columns(base_source)
    cols=[c for c in cols if c not in {
        "a_horse_number","b_horse_number","pair_horse_number_gap"
    }]
    cols += [
        "market_q_norm","market_log_q","market_rank",
        "l17_pair_rank","l17_market_rank_gap","log_l17_pair_prob","l17_market_log_gap"
    ]
    return cols

def market_context_columns(df):
    miss=[c for c in MARKET_CONTEXT_BASE if c not in df.columns]
    if miss:
        raise SystemExit(f"missing market-context features={miss}")
    return list(MARKET_CONTEXT_BASE)

def train_binary(train,test,cols,seed):
    xtr,xte=encode(train,test,cols)
    y=train["hit"].astype(int).to_numpy()
    model=lgb.LGBMClassifier(
        objective="binary",
        n_estimators=360,
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
    imp=pd.DataFrame({"feature":cols,"gain":model.feature_importances_})
    return p,imp

def add_residual_rank(df):
    z=df.copy()
    ordered=z.sort_values(
        ["year","race_id","delta_logit","p_full","l17_pair_rank","pair_numbers"],
        ascending=[True,True,False,False,True,True]
    )
    r=ordered.groupby(["year","race_id"],sort=False).cumcount()+1
    z.loc[ordered.index,"residual_rank"]=r.to_numpy()
    z["residual_rank"]=z["residual_rank"].astype(int)
    return z

def policy(df,min_delta,max_tickets):
    z=df[df["delta_logit"]>=min_delta].copy()
    if z.empty:
        return z
    z=z.sort_values(
        ["year","race_date","race_id","delta_logit","p_full","residual_rank"],
        ascending=[True,True,True,False,False,True]
    )
    return z.groupby(["year","race_id"],sort=False).head(max_tickets).copy()

def model_metrics(y,p,prefix):
    yy=y.astype(int).to_numpy()
    pp=np.clip(np.asarray(p,dtype=float),1e-9,1-1e-9)
    return {
        f"{prefix}_auc":float(roc_auc_score(yy,pp)),
        f"{prefix}_logloss":float(log_loss(yy,pp)),
        f"{prefix}_brier":float(brier_score_loss(yy,pp)),
    }

def eval_policy_by_year(preds,min_delta,max_tickets):
    rows=[]
    for y in DEV_YEARS:
        df=preds[y]
        c=policy(df,min_delta,max_tickets)
        m=evaluate_chosen(c,df["race_id"].nunique(),f"DELTA_{min_delta:.2f}_MAX{max_tickets}")
        m["year"]=y
        rows.append(m)
    pooled=pd.concat([policy(preds[y],min_delta,max_tickets) for y in DEV_YEARS],ignore_index=True)
    source=sum(preds[y]["race_id"].nunique() for y in DEV_YEARS)
    comb=evaluate_chosen(pooled,source,f"DELTA_{min_delta:.2f}_MAX{max_tickets}")
    return rows,comb

def residual_bins(df):
    z=df.copy()
    z["delta_bin"]=pd.cut(
        z["delta_logit"],
        [-np.inf,0,.25,.5,.75,1,1.5,2,np.inf],
        labels=["<0","0-.25",".25-.5",".5-.75",".75-1","1-1.5","1.5-2","2+"],
        right=False
    ).astype(str)
    rows=[]
    for b,g in z.groupby("delta_bin",sort=False):
        if not len(g): continue
        m=evaluate_chosen(g,g["race_id"].nunique(),str(b))
        m["delta_bin"]=b
        rows.append(m)
    return rows

def main():
    a=args()
    paths={y:getattr(a,f"l17_{y}") for y in YEARS}
    l17={y:load_l17(paths[y],y) for y in YEARS}
    frames={
        y:add_relative_features(add_market_features(build_year_frame(y,l17[y],a.backfill_root)))
        for y in YEARS
    }

    market_cols=market_context_columns(frames[2022])
    full_cols=full_feature_columns(frames[2022])
    preds={}
    fold_rows=[]
    imps=[]

    for y in (2023,2024,2025):
        train=pd.concat([frames[t] for t in YEARS if t<y],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)
        p_market,imp_market=train_binary(train,test,market_cols,83000+y)
        p_full,imp_full=train_binary(train,test,full_cols,93000+y)
        test["p_market_context"]=p_market
        test["p_full"]=p_full
        test["delta_logit"]=logit(p_full)-logit(p_market)
        test=add_residual_rank(test)
        preds[y]=test

        row={
            "test_year":y,
            "train_years":"|".join(str(t) for t in YEARS if t<y),
            "train_pairs":len(train),
            "test_pairs":len(test),
            "source_races":test["race_id"].nunique(),
            "positive_pairs":int(test["hit"].sum()),
            "mean_delta_logit":float(test["delta_logit"].mean()),
            "pct_positive_delta":100.0*float((test["delta_logit"]>0).mean()),
        }
        row.update(model_metrics(test["hit"],p_market,"market_context"))
        row.update(model_metrics(test["hit"],p_full,"full"))
        row["logloss_lift_full_minus_market"]=row["market_context_logloss"]-row["full_logloss"]
        row["brier_lift_full_minus_market"]=row["market_context_brier"]-row["full_brier"]
        row["auc_lift_full_minus_market"]=row["full_auc"]-row["market_context_auc"]
        fold_rows.append(row)

        imp_market["model"]="MARKET_CONTEXT";imp_market["test_year"]=y
        imp_full["model"]="FULL";imp_full["test_year"]=y
        imps.extend([imp_market,imp_full])

    grid=[]
    yearly_grid=[]
    for d in MIN_DELTA:
        for mx in MAX_TICKETS:
            yrs,comb=eval_policy_by_year(preds,d,mx)
            yearly_grid.extend([{**r,"min_delta":d,"max_tickets_per_race":mx} for r in yrs])
            min_roi=min(float(r["roi_pct"]) for r in yrs if r["roi_pct"] is not None)
            min_cov=min(float(r["execution_coverage_pct"]) for r in yrs)
            comb.update({
                "min_delta":d,
                "max_tickets_per_race":mx,
                "min_year_roi_pct":min_roi,
                "min_year_execution_coverage_pct":min_cov,
            })
            grid.append(comb)

    eligible=[
        r for r in grid
        if r["min_year_execution_coverage_pct"]>=MIN_DEV_COVERAGE_PCT
    ]
    pool=eligible or grid
    champion=max(pool,key=lambda r:(
        r["min_year_roi_pct"],
        r["roi_pct"] if r["roi_pct"] is not None else -1e18,
        r["profit_yen"],
        -r["max_drawdown_yen"],
    ))

    hold=preds[CONFIRM_YEAR]
    chosen=policy(
        hold,
        float(champion["min_delta"]),
        int(champion["max_tickets_per_race"])
    )
    hold_metrics=evaluate_chosen(chosen,hold["race_id"].nunique(),"FROZEN_DEV_CHAMPION")
    hold_metrics.update({
        "min_delta":float(champion["min_delta"]),
        "max_tickets_per_race":int(champion["max_tickets_per_race"]),
        "selected_from_dev_years":list(DEV_YEARS),
    })

    confirm_grid=[]
    for d in MIN_DELTA:
        for mx in MAX_TICKETS:
            c=policy(hold,d,mx)
            m=evaluate_chosen(c,hold["race_id"].nunique(),f"DELTA_{d:.2f}_MAX{mx}")
            m.update({"min_delta":d,"max_tickets_per_race":mx})
            confirm_grid.append(m)

    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"fold-model-comparison.csv",fold_rows)
    write_csv(out/"dev-policy-grid.csv",grid)
    write_csv(out/"dev-policy-grid-by-year.csv",yearly_grid)
    write_csv(out/"confirmation-2025-policy-grid-diagnostic.csv",confirm_grid)
    write_csv(out/"confirmation-2025-selected.csv",[hold_metrics])
    write_csv(out/"confirmation-2025-residual-bins.csv",residual_bins(chosen))
    write_csv(out/"feature-importance.csv",pd.concat(imps,ignore_index=True))

    with gzip.open(out/"confirmation-selected-tickets.csv.gz","wt",newline="",encoding="utf-8") as fh:
        keep=[
            "year","race_id","race_date","pair_horse_ids","pair_numbers",
            "market_q_norm","market_rank","l17_pair_rank","l17_market_rank_gap",
            "p_market_context","p_full","delta_logit","residual_rank",
            "hit","return_yen_per100","odds","a_consensus_rank","b_consensus_rank"
        ]
        chosen[keep].to_csv(fh,index=False)

    summary={
        "contract":"L2_MARKET_RESIDUAL_V3_RESULT",
        "architecture":"market_context_baseline_vs_l17_residual_correction",
        "raw_ev_multiplication":False,
        "candidate_horse_cut":False,
        "development_years":list(DEV_YEARS),
        "confirmation_year":CONFIRM_YEAR,
        "confirmation_used_for_policy_selection":False,
        "selection_objective":"maximize worst-year DEV ROI subject to >=20% execution coverage in both DEV years; tie combined ROI/profit/lower drawdown",
        "selected_policy":champion,
        "confirmation_2025":hold_metrics,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Market Residual V3\n\n"
        "A market-context model and a market+L1.7 model predict the same ticket-hit target. "
        "The decision signal is delta_logit = logit(p_full) - logit(p_market_context), so market odds are not multiplied into a raw EV score. "
        "Policy selection uses 2023-2024 only and maximizes the worse yearly ROI, with a minimum 20% execution coverage in both years. "
        "2025 is confirmation only and is not used to select thresholds. 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("L2_MARKET_RESIDUAL_V3_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
