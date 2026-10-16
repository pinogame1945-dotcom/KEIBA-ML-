#!/usr/bin/env python3
"""Strictly-prior target horse state and residual-regime features.

Original source: frozen run_l1_state_transition_v1.py and
run_l1_residual_regime_full_pairwise_v1.py. Reproduce prior race summaries,
lag/transition features and original regime formulas for *one unscored race*.
The target's result is absent; date >= target date never enters the history.
Research-replay only unless upstream capture timestamps are verified.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from run_l1_state_transition_v1 import (
    RAW_LAG_METRICS, TRANSITION_METRICS, RESIDUAL_METRICS, MAX_LAG)
from run_l1_residual_regime_full_pairwise_v1 import RESIDUAL_AXES

TARGET="202505050812"
CONTRACT="KEIBA_ORIGINAL_TARGET_SEQ_REGIME_PRIOR_ONLY_V1"
UPDATE_COLUMNS=(
    "_finish","_is_win","_is_top3","distance_m","body_weight","carried_weight",
    "margin_lengths","network_expected_pairwise_score","finish_time_ms","last_3f",
    "field_size","finish_pct","speed_mps","last3f_speed_mps",
    "perf_resid","speed_resid","last3f_resid",
)

def require(condition, message):
    if not condition:raise ValueError(message)

def tofloat(value):
    try: v=float(value)
    except (TypeError,ValueError):return float("nan")
    return v if math.isfinite(v) else float("nan")

def clean(v):
    try: v=float(v)
    except (TypeError,ValueError):return None
    return v if math.isfinite(v) else None

def std0(values):
    finite=[tofloat(v) for v in values if math.isfinite(tofloat(v))]
    return float(np.std(finite,ddof=0)) if finite else float("nan")

def mean(values):
    a=[tofloat(v) for v in values if math.isfinite(tofloat(v))]
    return float(np.mean(a)) if a else float("nan")

def prior_events(history,target_date):
    require(isinstance(history,pd.DataFrame) and not history.empty,"EMPTY_PRIOR_HISTORY")
    for k in ("_race_id","_horse_id","_race_date",*UPDATE_COLUMNS[:10]):
        require(k in history.columns,"MISSING_ORIGINAL_STATE_SOURCE_"+k)
    h=history.copy()
    require(h["_race_id"].astype(str).ne(TARGET).all(),"TARGET_OUTCOME_IN_HISTORY")
    require(pd.to_datetime(h["_race_date"],errors="coerce").notna().all(),
            "INVALID_HISTORY_RACE_DATE")
    require(h["_race_date"].astype(str).lt(target_date).all(),
            "HISTORY_NOT_STRICT_PRIOR_DATE")
    require(h["_race_id"].astype(str).str[:4].astype(int).between(2019,2025).all(),
            "INVALID_OR_SEALED_PRIOR_YEAR")
    require(h["_horse_id"].astype(str).ne("").all(),"PRIOR_HORSE_ID_MISSING")
    require(not h.duplicated(["_race_id","_horse_id"]).any(),
            "DUPLICATE_PRIOR_RACE_HORSE")
    for k in ("_finish","_is_win","_is_top3","distance_m","body_weight",
              "carried_weight","margin_lengths","network_expected_pairwise_score",
              "finish_time_ms","last_3f"):
        h[k]=pd.to_numeric(h[k],errors="coerce").astype(float)
    require(h["_finish"].ge(1).all() and h["_finish"].notna().all(),
            "INVALID_FINISH_POSITION_IN_PRIOR_HISTORY")
    # Exact original formulas (run_l1_state_transition_v1 build_state_features).
    h["field_size"]=h.groupby("_race_id")["_horse_id"].transform("size").astype(float)
    denom=(h["field_size"]-1).clip(lower=1.0)
    h["finish_pct"]=(1.0-(h["_finish"]-1.0)/denom).clip(0.0,1.0).astype("float32")
    sec=h["finish_time_ms"]/1000.0
    h["speed_mps"]=np.where(
        np.isfinite(h["distance_m"])&np.isfinite(sec)&(sec>0),
        h["distance_m"]/sec,np.nan).astype("float32")
    h["last3f_speed_mps"]=np.where(
        np.isfinite(h["last_3f"])&(h["last_3f"]>0),
        600.0/h["last_3f"],np.nan).astype("float32")
    speed_med=h.groupby("_race_id")["speed_mps"].transform("median")
    last_med=h.groupby("_race_id")["last3f_speed_mps"].transform("median")
    h["perf_resid"]=(h["finish_pct"].astype(float)-
                     h["network_expected_pairwise_score"].astype(float)).astype("float32")
    h["speed_resid"]=(h["speed_mps"].astype(float)-speed_med.astype(float)).astype("float32")
    h["last3f_resid"]=(h["last3f_speed_mps"].astype(float)-last_med.astype(float)).astype("float32")
    duplicate=h.duplicated(["_horse_id","_race_date"],keep=False)
    if duplicate.any():
        for key, grp in h.loc[duplicate].groupby(["_horse_id","_race_date"]):
            for c in UPDATE_COLUMNS:
                vals=grp[c].dropna().to_numpy(dtype=float)
                require(len(vals)<2 or np.allclose(vals,vals[0],
                        rtol=1e-7,atol=1e-7,equal_nan=True),
                        "SAME_DAY_STATE_ROWS_CONFLICT")
    return (h.sort_values(["_horse_id","_race_date","_race_id"])
             .drop_duplicates(["_horse_id","_race_date"]).reset_index(drop=True))

def horse_features(history,target_date):
    events=history.sort_values(["_race_date","_race_id"],ascending=[False,False])
    recent=events.head(MAX_LAG)
    lags={k:{c:float("nan") for c in (
           list(RAW_LAG_METRICS)+list(RESIDUAL_METRICS)+["network_expected_pairwise_score"])}
          for k in range(1,MAX_LAG+1)}
    prior_dates={}
    for index,(_,row) in enumerate(recent.iterrows(),1):
        for metric in RAW_LAG_METRICS:
            src={"is_win":"_is_win","is_top3":"_is_top3"}.get(metric,metric)
            lags[index][metric]=tofloat(row.get(src))
        for metric in (*RESIDUAL_METRICS,"network_expected_pairwise_score"):
            lags[index][metric]=tofloat(row.get(metric))
        prior_dates[index]=pd.Timestamp(row["_race_date"])
    day=pd.Timestamp(target_date)
    result={"seq_history_count":float(len(events))}
    for k in range(1,MAX_LAG+1):
        result[f"seq_gap_current_lag{k}_days"]=(
            float((day-prior_dates[k]).days) if k in prior_dates else float("nan"))
        for metric in (*RAW_LAG_METRICS,*RESIDUAL_METRICS,"network_expected_pairwise_score"):
            result[f"seq_lag{k}_{metric}"]=lags[k][metric]
    for a,b in ((1,2),(2,3)):
        v=(float((prior_dates[a]-prior_dates[b]).days)
           if a in prior_dates and b in prior_dates else float("nan"))
        result[f"seq_gap_lag{a}_lag{b}_days"]=v
    for metric in (*TRANSITION_METRICS,*RESIDUAL_METRICS):
        l1,l2,l3=[lags[k][metric] for k in (1,2,3)]
        valid=all(math.isfinite(v) for v in (l1,l2,l3))
        v12=l1-l2
        v23=l2-l3
        stem=metric
        result[f"seq_delta12_{stem}"]=v12
        result[f"seq_delta23_{stem}"]=v23
        result[f"seq_accel_{stem}"]=v12-v23
        result[f"seq_reversal_{stem}"]=(
            float(v12*v23<0) if valid else float("nan"))
        trio=[l1,l2,l3]
        result[f"seq_std3_{stem}"]=std0(trio)
        if metric in RESIDUAL_METRICS:
            result[f"seq_positive_count3_{stem}"]=(
                float(sum(v>0 for v in trio)) if valid else float("nan"))
        else:
            valid_trio=[v for v in trio if math.isfinite(v)]
            result[f"seq_range3_{stem}"]=(
                max(valid_trio)-min(valid_trio) if valid_trio else float("nan"))
    for axis in RESIDUAL_AXES:
        vals=[lags[k][axis] for k in range(1,6)]
        for k,v in enumerate(vals,1):
            result[f"regime_{axis}_lag{k}"]=v
        l1,l2,l3,l4,l5=vals
        tr,fi=vals[:3],vals
        valid3=all(math.isfinite(v) for v in tr)
        valid5=all(math.isfinite(v) for v in fi)
        std3=std0(tr)
        result.update({
            f"regime_{axis}_mean3":mean(tr),
            f"regime_{axis}_mean5":mean(fi),
            f"regime_{axis}_std3":std3,
            f"regime_{axis}_std5":std0(fi),
            f"regime_{axis}_slope3":(l1-l3)/2,
            f"regime_{axis}_slope5":(l1-l5)/4,
            f"regime_{axis}_accel":(l1-l2)-(l2-l3),
            f"regime_{axis}_shock1":l1-(l2+l3)/2,
            f"regime_{axis}_positive_count3":(
                float(sum(v>0 for v in tr)) if valid3 else float("nan")),
            f"regime_{axis}_positive_count5":(
                float(sum(v>0 for v in fi)) if valid5 else float("nan")),
            f"regime_{axis}_uptrend3":(
                float(l1>l2>l3) if valid3 else float("nan")),
            f"regime_{axis}_downtrend3":(
                float(l1<l2<l3) if valid3 else float("nan")),
            f"regime_{axis}_persistent_positive3":(
                float(all(v>0 for v in tr)) if valid3 else float("nan")),
            f"regime_{axis}_persistent_negative3":(
                float(all(v<0 for v in tr)) if valid3 else float("nan")),
            f"regime_{axis}_rebound":(
                float(l2<l3 and l1>l2) if valid3 else float("nan")),
            f"regime_{axis}_fade":(
                float(l2>l3 and l1<l2) if valid3 else float("nan")),
        })
        m=result[f"regime_{axis}_mean3"]
        result[f"regime_{axis}_strength3"]=(
            float(np.clip(m/(abs(std3)+0.05),-10,10))
            if math.isfinite(m) else float("nan"))
    return result

def compute(history,unscored):
    require(unscored.get("race_id")==TARGET and
            unscored.get("contract")=="KEIBA_UNSCORED_RACE_FEATURE_EXTRACT_V1",
            "INVALID_TARGET_BLIND_CONTRACT")
    require(unscored.get("used_results_payouts_final_odds") is False and
            unscored.get("used_target_labels") is False and
            unscored.get("used_historical_oos_target_predictions") is False,
            "TARGET_RESULTS_OR_SAVED_PREDICTIONS_PROHIBITED")
    target=unscored.get("rows")
    require(isinstance(target,list) and len(target)==17,"TARGET_NOT_17_HORSES")
    identities=[str(h.get("_horse_id")) for h in target]
    require(len(set(identities))==17,"TARGET_DUPLICATE_HORSE")
    require({h.get("_race_id") for h in target}=={TARGET},
            "TARGET_RACE_ID_DRIFT")
    day={h.get("_race_date") for h in target}
    require(day=={"2025-11-30"},"TARGET_RACE_DATE_DRIFT")
    events=prior_events(history,"2025-11-30")
    result=[]
    cols=None
    for hid in identities:
        h=events[events["_horse_id"].astype(str)==hid]
        features=horse_features(h,"2025-11-30")
        if cols is None:cols=set(features)
        require(set(features)==cols,"ORIGINAL_L1_STATE_FEATURE_SET_DRIFT")
        result.append({"horse_id":hid,"features":{k:clean(v) for k,v in features.items()}})
    seq_cols=sorted(k for k in cols if k.startswith("seq_"))
    regime_cols=sorted(k for k in cols if k.startswith("regime_"))
    require(len(regime_cols)==3*(5+16+1),"ORIGINAL_REGIME_FEATURE_COUNT_DRIFT")
    return {
        "contract":CONTRACT,
        "race_id":TARGET,
        "status":"RESEARCH_STRICT_PRIOR_SEQ_REGIME_RECONSTRUCTED_ASOF_UNVERIFIED",
        "history_years":sorted(set(events["_race_date"].str[:4].astype(int))),
        "history_rows":int(len(events)),
        "target_horses":len(result),
        "seq_feature_count":len(seq_cols),
        "regime_feature_count":len(regime_cols),
        "feature_names":sorted(cols),
        "strict_prior_date_only":True,
        "target_outcome_used":False,
        "2026_results_used":False,
        "original_research_parity_fully_verified":False,
        "genuine_asof_proven":False,
        "horse_features":result,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--compact-prior",action="append",required=True)
    p.add_argument("--unscored-target",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    frames=[pd.read_csv(path,compression="infer",low_memory=False,
                        dtype={"_race_id":"string","_horse_id":"string"})
            for path in a.compact_prior]
    unscored=json.loads(Path(a.unscored_target).read_text())
    out=compute(pd.concat(frames,ignore_index=True),unscored)
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,allow_nan=False)+"\n")
    print("ORIGINAL_BLIND_PRIOR_SEQ_REGIME",out["race_id"],
          "horses",out["target_horses"],
          "state",out["seq_feature_count"],
          "regime",out["regime_feature_count"],
          "history_rows",out["history_rows"],
          "asof_verified",False,flush=True)

if __name__=="__main__":main()
