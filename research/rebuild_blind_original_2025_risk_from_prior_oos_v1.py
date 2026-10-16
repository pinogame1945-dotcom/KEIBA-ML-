#!/usr/bin/env python3
"""Rebuild genuine 2021-24 original OOS Risk teacher, then score blind 2025 race.

Requires 2019-24 true labelled history; target 202505050812 never enters
original fit_direct_rank() (which uses validation outcomes for OOS checks).
Every past-year fold uses only Y-2,Y-1 years for ability. Risk teacher uses
all past OOS 2021-24 and market outcomes FROM THOSE PAST YEARS ONLY.
"""
from __future__ import annotations
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_l1_residual_regime_full_pairwise_v1 import fit_direct_rank,top1_metrics
from run_l1_market_correction_distill_v1 import add_ai_context
from run_l1_market_correction_deepdive_v1 import fit_risk_full
from run_l1_sameday_track_bias_v1 import bias_cols,speed_cols
from run_l1_state_transition_v1 import attach_state
from run_market_fixed_anchor_residual_v2 import EXPECTED_ABILITY
from run_l1_objective_rebuild_v1 import META,FORBIDDEN_KEYS

TRAIN_YEARS=(2019,2020,2021,2022,2023,2024)
TEACHER_YEARS=(2021,2022,2023,2024)
TARGET_YEAR=2025
TARGET="202505050812"

def require(condition,why):
    if not condition:raise ValueError(why)

def reader(path):
    return pd.read_csv(path,compression="infer",low_memory=False,
                       dtype={"_race_id":"string","_horse_id":"string","_race_date":"string"})

def teacher_frames(root,regimes,regime_cols):
    root=Path(root)
    # The frozen training program always adds the same 19 original bias/fit
    # and the same original regime_speed_resid_* set to CORE4 base.
    speed=speed_cols(regime_cols)
    bias_example=reader(root/"original-2021-daybias.csv.gz")
    fit=bias_cols(bias_example,with_fit=True)
    require(speed and fit,"PRIOR_OOS_RISK_INPUT_GEOMETRY_MISSING")
    oos=[]
    checks=[]
    for year in TEACHER_YEARS:
        years=(year-2,year-1)
        train_frames=[];valid=None
        for y in (*years,year):
            base=reader(root/f"original-{y}-core4-base.csv.gz")
            bias=reader(root/f"original-{y}-daybias.csv.gz")
            require(not base.duplicated(["_race_id","_horse_id"]).any() and
                    not bias.duplicated(["_race_id","_horse_id"]).any(),
                    "PRIOR_OOS_INPUT_KEY_DUPLICATE")
            require(base["_race_id"].astype(str).str[:4].eq(str(y)).all(),
                    "PRIOR_OOS_TARGET_YEAR_LEAK")
            frame=bias.merge(regimes[y][["_race_id","_horse_id",*speed]],
                             on=["_race_id","_horse_id"],
                             how="left",validate="one_to_one")
            full=attach_state(base,frame,fit+speed)
            require(len(full)==len(base),"PRIOR_OOS_FEATURE_JOIN_LOSS")
            if y==year:valid=full
            else:train_frames.append(full)
            del base,bias,frame,full
        train=pd.concat(train_frames,ignore_index=True)
        del train_frames
        expected=set(train.columns)-META-FORBIDDEN_KEYS
        observed=set(valid.columns)-META-FORBIDDEN_KEYS
        require(expected==observed,"PRIOR_OOS_ABL_FEATURE_SET_MISMATCH")
        predicted,_=fit_direct_rank(train,valid,5100000+year*100)
        metric=top1_metrics(predicted,"_score")["top1_top3_pct"]
        require(abs(metric-EXPECTED_ABILITY[year])<1e-9,
                "FROZEN_ORIGINAL_OOS_ABILITY_METRIC_DRIFT_YEAR_"+str(year))
        ctx=add_ai_context(predicted)
        market=reader(root/f"original-{year}-prior-market.csv.gz")
        require(market["_race_id"].astype(str).str[:4].eq(str(year)).all(),
                "NOT_PRIOR_MARKET_YEAR")
        ctx=ctx.merge(market,on=["_race_id","_horse_id"],
                      how="left",validate="one_to_one")
        require(ctx["market_rank"].notna().mean()>=0.99,
                "PRIOR_OOS_RISK_TEACHER_MARKET_COVERAGE_LOW")
        ctx["market_rank"]=pd.to_numeric(ctx["market_rank"],errors="coerce")
        ctx["_target_gap"]=ctx["market_rank"]-ctx["distill_ai_rank"]
        ctx["_target_gap_clip"]=ctx["_target_gap"].clip(-6,6)
        ctx["_target_up"]=(ctx["_target_gap"]>0).astype(int)
        ctx["_target_over3"]=(ctx["_target_gap"]>=3).astype(int)
        model=valid.merge(ctx[[
             "_race_id","_horse_id","_score","distill_ai_rank",
             "distill_field_size","distill_ai_rank01","distill_score_z",
             "distill_score_gap_to_best","distill_is_ai_top1",
             "distill_is_ai_top3","distill_is_ai_top6","market_rank",
             "final_win_odds","_target_gap","_target_gap_clip",
             "_target_up","_target_over3"]],
             on=["_race_id","_horse_id"],how="left",validate="one_to_one")
        require(len(model)==len(valid) and model["_target_over3"].isin((0,1)).all(),
                "ORIGINAL_OOS_RISK_LABEL_JOIN_DRIFT")
        oos.append(model)
        checks.append({
          "oos_test_year":year,"train_years":list(years),
          "horses":len(model),"races":int(model["_race_id"].nunique()),
          "original_top1_top3_pct":float(metric),
          "metric_reference_exact":True,
          "only_prior_teacher_market_final_read":True
        })
        print("ORIGINAL_RISK_TEACHER_PRIOR_OOS_FOLD",json.dumps(checks[-1],sort_keys=True),
              flush=True)
        del train,valid,predicted,ctx,market,model
        gc.collect()
    teacher=pd.concat(oos,ignore_index=True)
    require(set(teacher["_race_id"].astype(str).str[:4].astype(int))==set(TEACHER_YEARS),
            "RISK_TEACHER_YEAR_LEAK")
    require(not teacher.duplicated(["_race_id","_horse_id"]).any(),
            "RISK_TEACHER_DUPLICATE")
    return teacher,checks

def train_and_score(root,regimes,regime_cols,valid,cold_ctx):
    require(len(valid)==17 and
            valid["_race_id"].astype(str).eq(TARGET).all(),
            "RISK_TARGET_MUST_BE_17_REAL_HORSES")
    require(not {"_finish","_is_win","_is_top3","final_win_odds",
                 "market_rank","_target_over3","_score",
                 "risk","P__ANCHOR_FULL"}.intersection(valid.columns),
            "RISK_TARGET_LOOKAHEAD_OR_CACHED_SCORE")
    require(cold_ctx["_race_id"].astype(str).eq(TARGET).all(),
            "RISK_COLD_CONTEXT_RACE_ID_DRIFT")
    teacher,checks=teacher_frames(root,regimes,regime_cols)
    source=valid.merge(cold_ctx[[
        "_race_id","_horse_id","_score","distill_ai_rank",
        "distill_field_size","distill_ai_rank01","distill_score_z",
        "distill_score_gap_to_best","distill_is_ai_top1",
        "distill_is_ai_top3","distill_is_ai_top6"]],
        on=["_race_id","_horse_id"],how="left",validate="one_to_one")
    require(len(source)==17 and source["distill_ai_rank"].notna().all(),
            "RISK_TARGET_ORIGINAL_NEW_SCORE_JOIN_FAIL")
    cols=[c for c in source.columns if not c.startswith("neg_")]
    target=source[cols].copy()
    tr=teacher.copy()
    train_cols=set(tr.columns) - set((
        "_finish","_is_win","_is_top3","_target_gap",
        "_target_gap_clip","_target_up","_target_over3",
        "market_rank","final_win_odds"))
    valid_cols=set(target.columns)
    require(train_cols==valid_cols,
            "RISK_FULL_PRERACE_TEACHER_VS_TARGET_FEATURE_SCHEMA_DRIFT")
    seed=8500000+2025*100
    risk,_=fit_risk_full(teacher[[
        *[c for c in target.columns if c in teacher.columns],
        "_target_over3"]],target,seed)
    require(len(risk)==17 and np.isfinite(risk).all() and
            np.all((risk>=0)&(risk<=1)),"CORRECTION_RISK_COLD_OUTPUT_INVALID")
    rank=source[["_horse_id","distill_ai_rank"]].copy()
    rank["correction_risk"]=risk
    result={
        "contract":"KEIBA_BLIND_202505050812_FROZEN_CORRECTION_RISK_V1",
        "race_id":TARGET,
        "status":"ORIGINAL_PAST_OOS_REBUILT_NEW_RISK_NOT_YET_FULL_BET",
        "risk_prior_teacher_years":list(TEACHER_YEARS),
        "risk_teacher_horse_rows":len(teacher),
        "risk_teacher_races":int(teacher["_race_id"].nunique()),
        "risk_seed":seed,
        "risk_model":"FULL_PRERACE",
        "original_oos_metric_reproduction":checks,
        "target_horse_count":17,
        "selected_2025_outcome_or_final_odds_used":False,
        "selected_2025_saved_prediction_used":False,
        "prior_2021_2024_historical_market_teacher_used":True,
        "genuine_target_prestart_capture_time_independently_verified":False,
        "l2_1_ready":False,
        "l3_v3_ready":False,
        "horse_risk":[{
           "horse_id":str(h["_horse_id"]),
           "ai_raw_rank":int(h["distill_ai_rank"]),
           "correction_risk":float(h["correction_risk"]),
        } for _,h in rank.sort_values("distill_ai_rank").iterrows()]
    }
    root=Path(root)
    (root/"BLIND-202505050812-original-risk-not-outcome-verified.json").write_text(
         json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n")
    print("REAL_202505050812_ORIGINAL_PAST_OOS_RISK_RETRAINED",
          json.dumps({k:v for k,v in result.items() if k!="horse_risk"},sort_keys=True),
          flush=True)
    print("REAL_202505050812_ORIGINAL_NEW_TOP5_WITH_RISK",
          json.dumps(result["horse_risk"][:5],ensure_ascii=False),flush=True)
    return result
