#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_exacta_decomposed_v1 import (
    YEARS,
    TEST_YEARS,
    build_pair_year_frame,
    pair_feature_columns,
    direction_feature_columns,
)
from run_l2_exacta_auto_v0 import train_classifier

EPS=1e-9


def parse_args():
    p=argparse.ArgumentParser(
        description="EXACTA learned-boundary V1: OOF meta-calibrator learns where base EV estimates fail."
    )
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--auto-baseline",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1)
        out[int(y)]=p
    return out


def chronological_base_calib(train):
    years=sorted(train["year"].unique())
    latest=years[-1]
    if len(years)>=2:
        base=train[train["year"]<latest].copy()
        calib=train[train["year"]==latest].copy()
        mode=f"holdout_year_{latest}"
    else:
        dates=sorted(train["race_date"].astype(str).unique())
        if len(dates)<10:
            raise SystemExit("too few dates for chronological calibration split")
        cut=max(1,min(len(dates)-1,int(len(dates)*0.70)))
        base_dates=set(dates[:cut])
        calib_dates=set(dates[cut:])
        base=train[train["race_date"].astype(str).isin(base_dates)].copy()
        calib=train[train["race_date"].astype(str).isin(calib_dates)].copy()
        mode=f"chronological_{latest}_70_30"
    if base.empty or calib.empty:
        raise SystemExit(f"empty split mode={mode}")
    return base,calib,mode


def base_predictions(train,test,pair_cols,dir_cols,seed):
    pp,_,_=train_classifier(
        train,test,pair_cols,"pair_hit",seed+1,direction_only=False
    )
    pd_,_,_=train_classifier(
        train,test,dir_cols,"direction_label",seed+2,direction_only=True
    )
    return np.clip(pp,EPS,1-EPS),np.clip(pd_,EPS,1-EPS)


def expand_orientations(frame,p_pair,p_dir):
    z=frame.copy().reset_index(drop=True)
    p_pair=np.asarray(p_pair,dtype=float)
    p_dir=np.asarray(p_dir,dtype=float)
    rows=[]
    common_cols=[
        "year","race_id","race_date","field_size","race_month","distance_m",
        "pair_market_q","pair_market_rank",
        "pair_consensus_rank_sum","pair_consensus_rank_abs_gap",
        "pair_mean_rank_sum","pair_mean_rank_abs_gap",
        "pair_rank_std_sum","pair_rank_std_abs_gap",
        "pair_top1_votes_sum","pair_top3_support_sum","pair_top6_support_sum",
        "pair_mean_probability_sum","pair_mean_probability_product",
        "pair_mean_probability_abs_gap","pair_expert_direction_agreement",
    ]
    for i,r in z.iterrows():
        base={c:r[c] for c in common_cols if c in z.columns}
        pair_p=float(p_pair[i])
        dir_p=float(p_dir[i])
        for orientation in (1,0):
            if orientation==1:
                odds=float(r["odds_a_to_b"])
                market_q=float(r["market_q_a_to_b"])
                hit=int(r["hit_a_to_b"])
                ret=float(r["return_a_to_b"])
                direction_prob=dir_p
                sign=1.0
                ticket=f'{int(r["a_horse_number"])}>{int(r["b_horse_number"])}'
            else:
                odds=float(r["odds_b_to_a"])
                market_q=float(r["market_q_b_to_a"])
                hit=int(r["hit_b_to_a"])
                ret=float(r["return_b_to_a"])
                direction_prob=1.0-dir_p
                sign=-1.0
                ticket=f'{int(r["b_horse_number"])}>{int(r["a_horse_number"])}'

            naive_p=max(EPS,min(1-EPS,pair_p*direction_prob))
            row={
                **base,
                "ticket":ticket,
                "orientation":orientation,
                "hit":hit,
                "return_yen":ret,
                "odds":odds,
                "market_q_orientation":market_q,
                "market_log_q_orientation":math.log(max(market_q,EPS)),
                "base_pair_probability":pair_p,
                "base_direction_probability":direction_prob,
                "base_direction_confidence":abs(direction_prob-0.5)*2.0,
                "base_exacta_probability":naive_p,
                "base_exacta_logit":math.log(naive_p/(1.0-naive_p)),
                "base_naive_ev":naive_p*odds-1.0,
                "pair_market_share_of_orientation":market_q/max(float(r["pair_market_q"]),EPS),
            }
            # Directional features are canonical A->B in frame; flip for B->A.
            for c in z.columns:
                if c.startswith("dir_") and c not in {
                    "dir_l17_prob","dir_market_aware_prob"
                }:
                    row[c]=sign*float(r[c])
            rows.append(row)

    out=pd.DataFrame(rows)
    out["market_rank_orientation"]=out.groupby(
        ["year","race_id"]
    )["market_q_orientation"].rank(method="min",ascending=False).astype(int)
    out["base_probability_rank"]=out.groupby(
        ["year","race_id"]
    )["base_exacta_probability"].rank(method="min",ascending=False).astype(int)
    out["base_ev_rank"]=out.groupby(
        ["year","race_id"]
    )["base_naive_ev"].rank(method="min",ascending=False).astype(int)
    return out


def meta_columns(df):
    identity={
        "year","race_id","race_date","ticket","orientation","hit","return_yen",
    }
    cols=[c for c in df.columns if c not in identity]
    return cols


def encode_meta(train,test,cols):
    xtr=train[cols].copy()
    xte=test[cols].copy()
    for c in cols:
        if xtr[c].dtype=="object" or xte[c].dtype=="object":
            tr=xtr[c].fillna("__NA__").astype(str)
            te=xte[c].fillna("__NA__").astype(str)
            vals=sorted(tr.unique())
            mp={v:i for i,v in enumerate(vals)}
            xtr[c]=tr.map(mp).fillna(-1).astype("int32")
            xte[c]=te.map(mp).fillna(-1).astype("int32")
        else:
            xtr[c]=pd.to_numeric(xtr[c],errors="coerce").fillna(0.0).astype("float32")
            xte[c]=pd.to_numeric(xte[c],errors="coerce").fillna(0.0).astype("float32")
    return xtr,xte


def train_meta(calib,test,cols,seed):
    xtr,xte=encode_meta(calib,test,cols)
    y=calib["hit"].astype(int).to_numpy()
    if len(np.unique(y))!=2:
        raise SystemExit("meta calibration labels single class")
    model=lgb.LGBMClassifier(
        objective="binary",
        n_estimators=360,
        learning_rate=0.025,
        num_leaves=15,
        min_child_samples=120,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=6.0,
        reg_alpha=1.0,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(xtr,y)
    raw=np.asarray(model.predict_proba(xte)[:,1],dtype=float)
    imp=pd.DataFrame({
        "feature":cols,
        "importance_split":model.feature_importances_,
        "importance_gain":model.booster_.feature_importance(importance_type="gain"),
    })
    return np.clip(raw,EPS,1-EPS),imp,model.booster_


def normalize_within_race(df,raw_col):
    z=df.copy()
    denom=z.groupby(["year","race_id"])[raw_col].transform("sum")
    z["learned_probability"]=z[raw_col]/denom.clip(lower=EPS)
    return z


def collect_splits(booster,test_year):
    dump=booster.dump_model()
    rows=[]
    def walk(node,tree_idx):
        if "split_index" not in node:
            return
        fi=int(node["split_feature"])
        rows.append({
            "test_year":test_year,
            "tree_index":tree_idx,
            "split_index":int(node["split_index"]),
            "feature":dump["feature_names"][fi],
            "threshold":str(node.get("threshold","")),
            "decision_type":str(node.get("decision_type","")),
            "split_gain":float(node.get("split_gain",0.0)),
            "internal_count":int(node.get("internal_count",0)),
        })
        walk(node["left_child"],tree_idx)
        walk(node["right_child"],tree_idx)
    for i,t in enumerate(dump["tree_info"]):
        walk(t["tree_structure"],i)
    return rows


def score_policy(df,label):
    z=df.copy()
    z["edge"]=z["learned_probability"]*z["odds"]-1.0
    chosen=z[z["edge"]>0.0].copy()
    source_races=int(z["race_id"].nunique())
    bet_races=int(chosen["race_id"].nunique())
    hit=chosen[chosen["hit"]==1]
    hit_races=int(hit["race_id"].nunique())
    tickets=len(chosen)
    stake=100.0*tickets
    ret=float(chosen["return_yen"].sum())
    return {
        "label":label,
        "source_races":source_races,
        "executed_races":bet_races,
        "execution_coverage_pct":100.0*bet_races/source_races if source_races else 0.0,
        "tickets":tickets,
        "tickets_per_source_race":tickets/source_races if source_races else 0.0,
        "tickets_per_executed_race":tickets/bet_races if bet_races else 0.0,
        "hit_races":hit_races,
        "race_hit_rate_pct_all":100.0*hit_races/source_races if source_races else 0.0,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else 0.0,
        "mean_edge":float(chosen["edge"].mean()) if tickets else None,
        "median_edge":float(chosen["edge"].median()) if tickets else None,
        "median_odds":float(chosen["odds"].median()) if tickets else None,
    }


def calibration_metrics(df):
    y=df["hit"].astype(int).to_numpy()
    p=np.clip(df["learned_probability"].to_numpy(dtype=float),EPS,1-EPS)
    return {
        "roc_auc":float(roc_auc_score(y,p)),
        "log_loss":float(log_loss(y,p,labels=[0,1])),
        "brier":float(brier_score_loss(y,p)),
        "probability_sum_mean":float(
            df.groupby(["year","race_id"])["learned_probability"].sum().mean()
        ),
    }


def main():
    a=parse_args()
    started=time.perf_counter()
    lp=parse_paths(a.l17_year)
    if set(lp)!=set(YEARS) or 2026 in lp:
        raise SystemExit("L1.7 years must be exactly 2022-2025; 2026 sealed")

    frames={
        y:build_pair_year_frame(y,load_l17(lp[y],y),a.backfill_root)
        for y in YEARS
    }
    pair_cols=pair_feature_columns(frames[2022],market=True)
    dir_cols=direction_feature_columns(frames[2022],market=True)

    fold_rows=[]
    metrics=[]
    importances=[]
    splits=[]
    test_scored=[]

    for y in TEST_YEARS:
        prior=pd.concat([frames[t] for t in YEARS if t<y],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)
        base,calib,split_mode=chronological_base_calib(prior)

        # Create genuinely out-of-time base predictions for the meta learner.
        pp_cal,pd_cal=base_predictions(base,calib,pair_cols,dir_cols,101000+y)
        meta_train=expand_orientations(calib,pp_cal,pd_cal)

        # Refit base learners on all prior data before scoring the unseen test year.
        pp_test,pd_test=base_predictions(prior,test,pair_cols,dir_cols,102000+y)
        meta_test=expand_orientations(test,pp_test,pd_test)

        cols=meta_columns(meta_train)
        raw,imp,booster=train_meta(meta_train,meta_test,cols,103000+y)
        meta_test["meta_raw_probability"]=raw
        meta_test=normalize_within_race(meta_test,"meta_raw_probability")
        test_scored.append(meta_test)

        row=score_policy(meta_test,f"LEARNED_BOUNDARY_{y}")
        row["year"]=y
        metrics.append(row)

        c=calibration_metrics(meta_test)
        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(str(t) for t in YEARS if t<y),
            "meta_calibration_split":split_mode,
            "meta_train_tickets":len(meta_train),
            "test_tickets":len(meta_test),
            "meta_features":len(cols),
            **c,
        })

        imp["test_year"]=y
        importances.append(imp)
        splits.extend(collect_splits(booster,y))

    all_test=pd.concat(test_scored,ignore_index=True)
    overall=score_policy(all_test,"LEARNED_BOUNDARY_ALL")
    baseline=pd.read_csv(a.auto_baseline)
    baseline_row=baseline[baseline["variant"]=="AUTO_MARKET_AWARE_EV"]
    if len(baseline_row)!=1:
        raise SystemExit("AUTO_MARKET_AWARE_EV baseline missing")
    baseline_dict=baseline_row.iloc[0].to_dict()

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(metrics).to_csv(out/"metrics-by-year.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(out/"folds.csv",index=False)
    pd.concat(importances,ignore_index=True).to_csv(out/"feature-importance.csv",index=False)
    split_df=pd.DataFrame(splits)
    split_df.to_csv(out/"learned-splits.csv",index=False)
    (
        split_df.sort_values(["test_year","split_gain"],ascending=[True,False])
        .groupby("test_year",sort=False)
        .head(40)
        .to_csv(out/"top-learned-splits.csv",index=False)
    )

    comparison=pd.DataFrame([
        {
            "model":"AUTO_V0_MARKET_AWARE",
            "source_races":int(baseline_dict["source_races"]),
            "executed_races":int(baseline_dict["bet_races"]),
            "tickets":int(baseline_dict["selected_tickets"]),
            "tickets_per_source_race":float(baseline_dict["tickets_per_source_race"]),
            "hit_races":int(baseline_dict["hit_races"]),
            "roi_pct":float(baseline_dict["roi_pct_equal100"]),
            "profit_yen":float(baseline_dict["profit_yen_equal100"]),
        },
        {
            "model":"LEARNED_BOUNDARY_V1",
            "source_races":int(overall["source_races"]),
            "executed_races":int(overall["executed_races"]),
            "tickets":int(overall["tickets"]),
            "tickets_per_source_race":float(overall["tickets_per_source_race"]),
            "hit_races":int(overall["hit_races"]),
            "roi_pct":float(overall["roi_pct"]),
            "profit_yen":float(overall["profit_yen"]),
        },
    ])
    comparison.to_csv(out/"auto-v0-vs-learned-boundary.csv",index=False)

    summary={
        "contract":"L2_EXACTA_LEARNED_BOUNDARY_V1",
        "purpose":"learn where the prior AUTO model overestimates exacta value using out-of-time meta learning",
        "manual_topk":False,
        "manual_confidence_floor":False,
        "manual_odds_band":False,
        "manual_ticket_cap":False,
        "manual_skip_rate":False,
        "legacy_law_seeded":False,
        "meta_model":"LightGBM classifier over out-of-time base predictions, odds/market state, pair context, and direction context",
        "meta_probability_normalization":"within-race sum-to-one",
        "selection_rule":"learned_probability * final_odds > 1.0",
        "selection_rule_reason":"mathematical break-even only; all practical boundaries are learned by the meta model",
        "walk_forward":{"test_years":list(TEST_YEARS),"training":"prior years only"},
        "overall_2023_2025":overall,
        "frozen_auto_v0_market_aware":baseline_dict,
        "2026_locked":True,
        "script_total_seconds":time.perf_counter()-started,
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    (out/"README.md").write_text(
        "# L2 EXACTA Learned Boundary V1\n\n"
        "This lane does not impose Top-K, a confidence floor, an odds band, a ticket cap, "
        "or a target skip rate. A second-stage LightGBM model is trained only on out-of-time "
        "predictions from prior data so it can learn where the base pair/direction model "
        "systematically over- or under-estimates exacta hit probability. Its split thresholds "
        "are the learned boundaries. Final ticket selection uses only the mathematical break-even "
        "condition learned_probability * final_odds > 1.0. Probabilities are normalized within race. "
        "Flat 100-yen staking is evaluation only; L3 stake sizing remains separate. 2026 is sealed.\n",
        encoding="utf-8",
    )

    print("L2_EXACTA_LEARNED_BOUNDARY_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
    print("===== AUTO V0 VS LEARNED =====")
    print(comparison.to_csv(index=False))
    print("===== YEARLY =====")
    print(pd.DataFrame(metrics).to_csv(index=False))


if __name__=="__main__":
    main()
