#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_exacta_decomposed_v1 import (
    YEARS,
    TEST_YEARS,
    build_pair_year_frame,
    pair_feature_columns,
    direction_feature_columns,
    encode,
)

EPS=1e-8


def parse_args():
    p=argparse.ArgumentParser(
        description="EXACTA AUTO V0: no hand-tuned TopK/Law/confidence/odds bands; calibrated probability x final odds drives selection."
    )
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1)
        out[int(y)]=p
    return out


def train_classifier(train,test,cols,target,seed,direction_only=False):
    if direction_only:
        tr=train[train["strict_direction_eligible"]==1].copy()
    else:
        tr=train.copy()
    tr=tr.sort_values(["year","race_id","pair_numbers"]).reset_index(drop=True)
    te=test.copy()
    te["_orig_index"]=te.index
    te=te.sort_values(["year","race_id","pair_numbers"]).reset_index(drop=True)

    if tr[target].nunique()!=2:
        raise SystemExit(f"target class drift target={target} counts={tr[target].value_counts().to_dict()}")

    xtr,xte=encode(tr,te,cols)
    y=tr[target].astype(int).to_numpy()
    model=lgb.LGBMClassifier(
        objective="binary",
        n_estimators=260 if not direction_only else 220,
        learning_rate=0.035,
        num_leaves=31 if not direction_only else 15,
        min_child_samples=100 if not direction_only else 60,
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
    raw=np.asarray(model.predict_proba(xte)[:,1],dtype=float)
    s=pd.Series(raw,index=te["_orig_index"].astype(int).to_numpy())
    pred=s.reindex(test.index).to_numpy()
    imp=pd.DataFrame({"feature":cols,"gain":model.feature_importances_})
    return pred,imp,len(tr)


def calibration_split(train):
    years=sorted(train["year"].unique())
    latest=years[-1]
    if len(years)>=2:
        base=train[train["year"]<latest].copy()
        calib=train[train["year"]==latest].copy()
        mode=f"holdout_year_{latest}"
    else:
        dates=sorted(train["race_date"].astype(str).unique())
        if len(dates)<10:
            raise SystemExit("too few dates for calibration split")
        cut=max(1,min(len(dates)-1,int(len(dates)*0.70)))
        base_dates=set(dates[:cut])
        calib_dates=set(dates[cut:])
        base=train[train["race_date"].astype(str).isin(base_dates)].copy()
        calib=train[train["race_date"].astype(str).isin(calib_dates)].copy()
        mode=f"chronological_{latest}_70_30"
    if base.empty or calib.empty:
        raise SystemExit(f"empty calibration split mode={mode}")
    return base,calib,mode


def _logit(p):
    p=np.clip(np.asarray(p,dtype=float),EPS,1.0-EPS)
    return np.log(p/(1.0-p)).reshape(-1,1)


def fit_platt(raw,y):
    y=np.asarray(y,dtype=int)
    if len(np.unique(y))!=2:
        raise SystemExit(f"platt labels single-class: {np.unique(y)}")
    lr=LogisticRegression(C=1.0,max_iter=1000,solver="lbfgs")
    lr.fit(_logit(raw),y)
    return lr


def apply_platt(model,raw):
    return model.predict_proba(_logit(raw))[:,1]


def calibration_metrics(raw,cal,y,label):
    y=np.asarray(y,dtype=int)
    return {
        "model":label,
        "rows":len(y),
        "positives":int(y.sum()),
        "raw_brier":float(brier_score_loss(y,np.clip(raw,EPS,1-EPS))),
        "calibrated_brier":float(brier_score_loss(y,np.clip(cal,EPS,1-EPS))),
        "raw_logloss":float(log_loss(y,np.clip(raw,EPS,1-EPS),labels=[0,1])),
        "calibrated_logloss":float(log_loss(y,np.clip(cal,EPS,1-EPS),labels=[0,1])),
    }


def fit_calibrated_variant(train,test,pair_cols,dir_cols,seed,label):
    base,calib,split_mode=calibration_split(train)

    # Out-of-time calibration predictions.
    cal_pair_raw,_,_=train_classifier(
        base,calib,pair_cols,"pair_hit",seed+1,direction_only=False
    )
    cal_dir_raw,_,_=train_classifier(
        base,calib,dir_cols,"direction_label",seed+2,direction_only=True
    )
    pair_platt=fit_platt(cal_pair_raw,calib["pair_hit"].astype(int).to_numpy())
    strict_mask=calib["strict_direction_eligible"].astype(bool).to_numpy()
    dir_platt=fit_platt(
        cal_dir_raw[strict_mask],
        calib.loc[strict_mask,"direction_label"].astype(int).to_numpy(),
    )

    pair_cal=apply_platt(pair_platt,cal_pair_raw)
    dir_cal=apply_platt(dir_platt,cal_dir_raw[strict_mask])
    cal_rows=[
        {
            "variant":label,
            "calibration_split":split_mode,
            **calibration_metrics(
                cal_pair_raw,pair_cal,calib["pair_hit"].astype(int).to_numpy(),"PAIR"
            ),
        },
        {
            "variant":label,
            "calibration_split":split_mode,
            **calibration_metrics(
                cal_dir_raw[strict_mask],
                dir_cal,
                calib.loc[strict_mask,"direction_label"].astype(int).to_numpy(),
                "DIRECTION",
            ),
        },
    ]

    # Refit on every prior race, then apply the frozen out-of-time calibrators.
    pair_raw,pair_imp,n_pair=train_classifier(
        train,test,pair_cols,"pair_hit",seed+11,direction_only=False
    )
    dir_raw,dir_imp,n_dir=train_classifier(
        train,test,dir_cols,"direction_label",seed+12,direction_only=True
    )
    pair_prob=apply_platt(pair_platt,pair_raw)
    dir_prob=apply_platt(dir_platt,dir_raw)

    pair_imp["component"]="PAIR"
    dir_imp["component"]="DIRECTION"
    pair_imp["variant"]=label
    dir_imp["variant"]=label

    return pair_prob,dir_prob,cal_rows,pd.concat([pair_imp,dir_imp],ignore_index=True),{
        "calibration_split":split_mode,
        "train_pair_rows":n_pair,
        "train_direction_rows":n_dir,
    }


def evaluate_policy(test,year,label,p_pair,p_dir):
    z=test.copy()
    z["p_pair"]=np.clip(p_pair,EPS,1.0-EPS)
    z["p_dir_a_to_b"]=np.clip(p_dir,EPS,1.0-EPS)
    z["p_exact_a_to_b"]=z["p_pair"]*z["p_dir_a_to_b"]
    z["p_exact_b_to_a"]=z["p_pair"]*(1.0-z["p_dir_a_to_b"])
    z["edge_a_to_b"]=z["p_exact_a_to_b"]*z["odds_a_to_b"]-1.0
    z["edge_b_to_a"]=z["p_exact_b_to_a"]*z["odds_b_to_a"]-1.0

    # No human TopK / confidence / odds-band boundary.
    # Positive model-implied expected value is the sole action criterion.
    z["select_a_to_b"]=z["edge_a_to_b"]>0.0
    z["select_b_to_a"]=z["edge_b_to_a"]>0.0
    z["selected_count"]=z["select_a_to_b"].astype(int)+z["select_b_to_a"].astype(int)

    selected_ab=z["select_a_to_b"]
    selected_ba=z["select_b_to_a"]
    selected_tickets=int(selected_ab.sum()+selected_ba.sum())
    source_races=int(z["race_id"].nunique())
    bet_races=int(z.loc[z["selected_count"]>0,"race_id"].nunique())
    both_dir_pairs=int(((z["select_a_to_b"])&(z["select_b_to_a"])).sum())

    win_ab=selected_ab & (z["hit_a_to_b"]==1)
    win_ba=selected_ba & (z["hit_b_to_a"]==1)
    winning_tickets=int(win_ab.sum()+win_ba.sum())
    hit_races=int(z.loc[win_ab|win_ba,"race_id"].nunique())

    total_return=float(
        z.loc[selected_ab,"return_a_to_b"].sum()
        +z.loc[selected_ba,"return_b_to_a"].sum()
    )
    stake=100.0*selected_tickets
    profit=total_return-stake

    selected_edges=np.concatenate([
        z.loc[selected_ab,"edge_a_to_b"].to_numpy(dtype=float),
        z.loc[selected_ba,"edge_b_to_a"].to_numpy(dtype=float),
    ]) if selected_tickets else np.array([],dtype=float)

    # Strict direction diagnostic only on uniquely settled exacta races.
    strict=z[z["strict_direction_eligible"]==1].copy()
    if len(strict):
        predicted_dir=np.where(strict["p_dir_a_to_b"]>=0.5,1,0)
        dir_correct=int((predicted_dir==strict["direction_label"].to_numpy()).sum())
    else:
        dir_correct=0

    row={
        "year":year,
        "variant":label,
        "source_races":source_races,
        "bet_races":bet_races,
        "bet_race_pct":100.0*bet_races/source_races if source_races else 0.0,
        "skip_race_pct":100.0*(source_races-bet_races)/source_races if source_races else 0.0,
        "selected_tickets":selected_tickets,
        "tickets_per_source_race":selected_tickets/source_races if source_races else 0.0,
        "tickets_per_bet_race":selected_tickets/bet_races if bet_races else 0.0,
        "pairs_both_directions_selected":both_dir_pairs,
        "winning_tickets":winning_tickets,
        "hit_races":hit_races,
        "hit_race_pct_all":100.0*hit_races/source_races if source_races else 0.0,
        "hit_race_pct_bet_only":100.0*hit_races/bet_races if bet_races else 0.0,
        "stake_yen_equal100":stake,
        "return_yen":total_return,
        "profit_yen_equal100":profit,
        "roi_pct_equal100":100.0*total_return/stake if stake else 0.0,
        "mean_selected_edge":float(selected_edges.mean()) if len(selected_edges) else 0.0,
        "median_selected_edge":float(np.median(selected_edges)) if len(selected_edges) else 0.0,
        "strict_direction_races":len(strict),
        "strict_direction_correct":dir_correct,
        "strict_direction_correct_pct":100.0*dir_correct/len(strict) if len(strict) else 0.0,
        "manual_topk":False,
        "manual_confidence_threshold":False,
        "manual_odds_band":False,
        "manual_law":False,
        "action_rule":"model_calibrated_probability * final_odds > 1",
    }
    return row


def main():
    a=parse_args()
    t0=time.perf_counter()
    lp=parse_paths(a.l17_year)
    if set(lp)!=set(YEARS) or 2026 in lp:
        raise SystemExit("L1.7 years must be exactly 2022-2025; 2026 sealed")

    frames={}
    for y in YEARS:
        frames[y]=build_pair_year_frame(y,load_l17(lp[y],y),a.backfill_root)

    pair_l17=pair_feature_columns(frames[2022],market=False)
    pair_market=pair_feature_columns(frames[2022],market=True)
    dir_l17=direction_feature_columns(frames[2022],market=False)
    dir_market=direction_feature_columns(frames[2022],market=True)

    if any("market" in c.lower() for c in pair_l17+dir_l17):
        raise SystemExit("market leakage into AUTO_L17 probability model")
    if any(c.startswith("dir_") for c in pair_l17+pair_market):
        raise SystemExit("directional leakage into unordered pair probability model")

    policy_rows=[]
    calibration_rows=[]
    fold_rows=[]
    importance=[]

    for y in TEST_YEARS:
        train=pd.concat([frames[t] for t in YEARS if t<y],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)

        fold_start=time.perf_counter()
        for idx,(label,pcols,dcols) in enumerate([
            ("AUTO_L17_EV",pair_l17,dir_l17),
            ("AUTO_MARKET_AWARE_EV",pair_market,dir_market),
        ]):
            pp,pd_,cal,imp,meta=fit_calibrated_variant(
                train,test,pcols,dcols,99000+y*10+idx,label
            )
            policy_rows.append(evaluate_policy(test,y,label,pp,pd_))
            for r in cal:
                r["test_year"]=y
                calibration_rows.append(r)
            imp["test_year"]=y
            importance.append(imp)
            fold_rows.append({
                "test_year":y,
                "variant":label,
                "train_years":"|".join(str(t) for t in YEARS if t<y),
                "test_pair_rows":len(test),
                "pair_features":len(pcols),
                "direction_features":len(dcols),
                **meta,
                "fold_elapsed_seconds":time.perf_counter()-fold_start,
            })

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(policy_rows).to_csv(out/"policy-results.csv",index=False)
    pd.DataFrame(calibration_rows).to_csv(out/"calibration-audit.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(out/"fold-metrics.csv",index=False)
    pd.concat(importance,ignore_index=True).to_csv(out/"feature-importance.csv",index=False)

    agg=[]
    pr=pd.DataFrame(policy_rows)
    for variant,g in pr.groupby("variant"):
        stake=float(g["stake_yen_equal100"].sum())
        ret=float(g["return_yen"].sum())
        source=int(g["source_races"].sum())
        bet=int(g["bet_races"].sum())
        hit=int(g["hit_races"].sum())
        tickets=int(g["selected_tickets"].sum())
        agg.append({
            "variant":variant,
            "years":"2023|2024|2025",
            "source_races":source,
            "bet_races":bet,
            "bet_race_pct":100.0*bet/source if source else 0.0,
            "selected_tickets":tickets,
            "tickets_per_source_race":tickets/source if source else 0.0,
            "hit_races":hit,
            "hit_race_pct_all":100.0*hit/source if source else 0.0,
            "stake_yen_equal100":stake,
            "return_yen":ret,
            "profit_yen_equal100":ret-stake,
            "roi_pct_equal100":100.0*ret/stake if stake else 0.0,
        })
    pd.DataFrame(agg).to_csv(out/"aggregate.csv",index=False)

    summary={
        "contract":"L2_EXACTA_AUTO_V0_RESULT",
        "bet_type":"EXACTA",
        "meaning_of_ruleless":"no human-tuned TopK, confidence floor, odds band, LAW, ticket-count cap, or skip-rate target",
        "irreducible_decision_principle":"select any orientation whose calibrated predicted hit probability times final odds exceeds 1.0",
        "probability_model":"unordered pair hit probability x conditional direction probability",
        "calibration":"out-of-time Platt scaling using latest prior year; chronological tail split when only 2022 is available",
        "variants":["AUTO_L17_EV","AUTO_MARKET_AWARE_EV"],
        "stakes":"evaluation only at flat 100 yen per selected ticket; L3 stake sizing remains separate",
        "walk_forward":{"test_years":list(TEST_YEARS),"training":"prior years only"},
        "law_search":False,
        "manual_boundaries":False,
        "2026_locked":True,
        "script_total_seconds":time.perf_counter()-t0,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 EXACTA AUTO V0 — No Hand-Tuned Rules\n\n"
        "This is the no-human-fixed-rule comparison lane. The model estimates the probability that an unordered pair is the first-two pair, "
        "then the conditional probability of A->B versus B->A. Probabilities are calibrated only with prior out-of-time data. "
        "No TopK, confidence floor, odds band, LAW, ticket-count cap, or forced skip rate is specified. "
        "An orientation is selected only when calibrated probability multiplied by final odds is greater than 1.0, i.e. positive model-implied expected value. "
        "Flat 100-yen staking is used only to evaluate the ticket-selection layer; stake sizing remains L3. 2026 is sealed.\n",
        encoding="utf-8"
    )

    print("L2_EXACTA_AUTO_V0_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
    print("===== AGGREGATE =====")
    print(pd.DataFrame(agg).to_csv(index=False))
    print("===== YEARLY POLICY =====")
    print(pd.DataFrame(policy_rows).to_csv(index=False))


if __name__=="__main__":
    main()
