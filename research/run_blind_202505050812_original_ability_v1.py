#!/usr/bin/env python3
"""Blind 202505050812 ORIGINAL CORE4 LambdaRank, real source not saved predictions.

- Reads original 2023/24 labelled training source + original same-day bias.
- Reconstructs original 2019–2025 state/regime *from strictly-prior data*;
  selected target's source outcome is never supplied anywhere.
- The 2025 target PRESTART FINAL-phase has no target label or saved score.
- Runs original LGBMRanker hyperparameters/seed and add_ai_context, and
  prints a fresh horse-id ranking. No Risk/odds/L2.1/L3 claim at this stage.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import run_l1_state_transition_v1 as state_module
from run_l1_objective_rebuild_v1 import META,model_common,prepare,select_variant
from run_l1_ceiling_audit_v1 import strip_forbidden,FULL_FORBIDDEN
from run_l1_residual_regime_full_pairwise_v1 import build_regime_features
from run_l1_market_correction_distill_v1 import add_ai_context
from run_l1_sameday_track_bias_v1 import bias_cols,speed_cols

TARGET="202505050812"
TARGET_YEAR=2025
TARGET_DAY="2025-11-30"
BASE_SHA="eb3392b725560e77d46b28aef036b6188814125ed8f7a3bfde0c53522dbadf58"
YEARS=range(2019,2026)
FIT_YEARS=(2023,2024)

def check(cond,reason):
    if not cond:raise ValueError(reason)

def csv_frame(path):
    return pd.read_csv(path,compression="infer",low_memory=False,
                       dtype={"_race_id":"string","_horse_id":"string","_race_date":"string"})

def _csv_item(item):
    year,path=item
    f=csv_frame(path)
    check(f["_year"].eq(year).all(),"COMPACT_ORIGINAL_STATE_SOURCE_YEAR_DRIFT")
    check(f["_race_id"].astype(str).str[:4].eq(str(year)).all(),
          "ORIGINAL_STATE_RACE_ID_YEAR_DRIFT")
    if year==TARGET_YEAR:
        check(f["_race_date"].astype(str).lt(TARGET_DAY).all(),
              "2025_STATE_TARGET_OR_FUTURE_RESULT_LEAK")
        check(f["_race_id"].astype(str).ne(TARGET).all(),
              "2025_SELECTED_RESULT_IN_STATE_SOURCE")
    return year,f

def original_state_regime(compacts):
    check(set(compacts)==set(YEARS),"2019_2025_STATE_SOURCE_COMPLETENESS")
    # Supply EXACT original build_state_features() formulas, with compact
    # records already processed by its own read_compact_state_source().
    # Only adapt its reader + thread executor; never change formula code.
    original_reader=state_module._read_state_year_item
    original_executor=state_module.concurrent.futures.ProcessPoolExecutor
    try:
        state_module._read_state_year_item=_csv_item
        state_module.concurrent.futures.ProcessPoolExecutor=concurrent.futures.ThreadPoolExecutor
        state,_,_,_,coverage=state_module.build_state_features(compacts)
    finally:
        state_module._read_state_year_item=original_reader
        state_module.concurrent.futures.ProcessPoolExecutor=original_executor
    regimes,cols=build_regime_features(state)
    return state,regimes,cols,coverage

def keys(df):
    return set(tuple(row) for row in df[["_race_id","_horse_id"]].to_numpy())

def main_pipeline(root):
    root=Path(root)
    documents={
        "target":json.loads((root/"target-unscored-final.json").read_text()),
        "bias":json.loads((root/"original-prior-daybias.json").read_text()),
        "prior_state":json.loads((root/"target-original-seq-regime.json").read_text())
    }
    target=documents["target"]
    check(target.get("contract")=="KEIBA_UNSCORED_RACE_FEATURE_EXTRACT_V1" and
          target.get("race_id")==TARGET,"INVALID_UNSCORED_REAL_TARGET_CONTRACT")
    check(target.get("source_prediction_phase")==
          "FINAL_PRESTART_PROJECTED_FROM_HISTORICAL_SNAPSHOT",
          "ORIGINAL_PRESTART_FINAL_FEATURE_CONTRACT_REQUIRED")
    for field in ("used_results_payouts_final_odds","used_target_labels",
                  "used_historical_oos_target_predictions"):
        check(target.get(field) is False,"TARGET_OUTCOME_OR_CACHE_WAS_USED")
    check(target.get("actual_source_capture_time_independently_verified") is False,
          "SOURCE_TIMING_NOT_AUTHENTICATED_YET")
    check(target.get("source_manifest_count_and_date_range_verified") is True,
          "SOURCE_MANIFEST_NOT_VERIFIED")
    check(len(target["rows"])==17 and len({r["_horse_id"] for r in target["rows"]})==17,
          "TARGET_17_HORSE_IDENTITY_DRIFT")
    for field in ("bias","prior_state"):
        check(documents[field]["race_id"]==TARGET,"SOURCE_TARGET_ID_MISMATCH_"+field)
        check(len(documents[field]["horse_scores"] if field=="bias"
                  else documents[field]["horse_features"])==17,
              "SOURCE_17_HORSE_DRIFT_"+field)
    check(documents["bias"]["target_result_or_final_odds_used"] is False,
          "BIAS_USED_TARGET_OUTCOME")
    check(documents["prior_state"]["strict_prior_date_only"] and
          documents["prior_state"]["target_outcome_used"] is False,
          "SEQ_REGIME_HISTORY_NOT_PRIOR")

    original=strip_forbidden(pd.DataFrame.from_records(target["rows"]))
    selected=select_variant(original,"core4")
    feats=sorted(set(selected.columns)-META)
    digest=hashlib.sha256(json.dumps(feats,separators=(",",":")).encode()).hexdigest()
    check(len(feats)==649 and digest==BASE_SHA,
          "ORIGINAL_TARGET_649_FEATURES_NAME_OR_SHA_DRIFT")
    check(not (set(selected)&FULL_FORBIDDEN),
          "BLINDED_TARGET_HAS_FORBIDDEN_SOURCE_FEATURES")
    check(not {"_finish","_is_win","_is_top3","final_win_odds","risk",
               "_score","P__ANCHOR_FULL"}.intersection(selected.columns),
          "BLINDED_TARGET_FEATURE_POSTRACE_LEAK")

    base={y:csv_frame(root/f"original-{y}-core4-base.csv.gz") for y in FIT_YEARS}
    bias={y:csv_frame(root/f"original-{y}-daybias.csv.gz") for y in FIT_YEARS}
    for y in FIT_YEARS:
        check(base[y]["_race_id"].astype(str).str.startswith(str(y)).all(),
              "ORIGINAL_BASE_TRAIN_YEAR_DRIFT")
        check(len(base[y])==len(bias[y]) and keys(base[y])==keys(bias[y]),
              "BASE_BIAS_SET_DRIFT")
        cols=sorted(set(base[y].columns)-META)
        check(len(cols)==649 and hashlib.sha256(
            json.dumps(cols,separators=(",",":")).encode()).hexdigest()==BASE_SHA,
            "ORIGINAL_PRIOR_TRAIN_FEATURE_SHA_DRIFT")
        check(not (set(base[y])&FULL_FORBIDDEN),
              "PRIOR_TRAIN_FORBIDDEN_FEATURES")
    compact={y:str(root/f"compact-strict-prior-{y}.csv.gz") for y in YEARS}
    states,regimes,regime_columns,coverage=original_state_regime(compact)
    speed=speed_cols(regime_columns)
    fit=bias_cols(bias[2023],with_fit=True)
    check(fit and speed,"ORIGINAL_FIT_OR_SPEED_FEATURES_NOT_RECONSTRUCTED")

    train_parts=[]
    for y in FIT_YEARS:
        src=bias[y].merge(regimes[y][["_race_id","_horse_id",*speed]],
                          on=["_race_id","_horse_id"],how="left",
                          validate="one_to_one")
        piece=state_module.attach_state(base[y],src,fit+speed)
        check(len(piece)==len(base[y]),"ORIGINAL_ABILITY_TRAIN_JOIN_LOSS")
        train_parts.append(piece)
    train=pd.concat(train_parts,ignore_index=True)

    bias_rows=[{"_race_id":TARGET,"_horse_id":x["horse_id"],**x["features"]}
               for x in documents["bias"]["horse_scores"]]
    src_bias=pd.DataFrame.from_records(bias_rows)
    check(keys(selected)==keys(src_bias),"BLIND_TARGET_BIAS_HORSE_DRIFT")
    state_rows=[{"_race_id":TARGET,"_horse_id":x["horse_id"],**x["features"]}
               for x in documents["prior_state"]["horse_features"]]
    src_state=pd.DataFrame.from_records(state_rows)
    check(keys(selected)==keys(src_state),"BLIND_TARGET_STATE_HORSE_DRIFT")
    target_extra=src_bias.merge(src_state[["_race_id","_horse_id",*speed]],
                               on=["_race_id","_horse_id"],how="left",validate="one_to_one")
    check(len(target_extra)==17,"TARGET_SPEED_BIAS_JOIN_LOSS")
    valid=state_module.attach_state(selected,target_extra,fit+speed)
    check(len(valid)==17,"BLIND_17_HORSES_REMOVED")
    train_features=set(train.columns)-META
    valid_features=set(valid.columns)-{"_race_id","_horse_id","_race_date"}
    check(train_features==valid_features,
          "ORIGINAL_TRAIN_VALID_ENRICHED_FEATURE_SCHEMA_DRIFT")
    check(train["_race_id"].astype(str).str[:4].astype(int).isin(FIT_YEARS).all(),
          "BLIND_ABILITY_TRAIN_YEARS_DRIFT")

    # Exact original run_l1_residual_regime_full_pairwise_v1.fit_direct_rank()
    # model algebra; don't call original wrapper because it requires target
    # finish, top3, win labels that MUST NOT be supplied for blind prediction.
    train=train.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    valid=valid.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    xtr,xva,cats=prepare(train,valid)
    relevance=np.select([train["_finish"].eq(k) for k in range(1,7)],
                        [6,5,4,3,2,1],default=0).astype(int)
    groups=train.groupby("_race_id",sort=False).size().tolist()
    check(sum(groups)==len(train),"BLIND_LAMBDARANK_GROUP_DRIFT")
    seed=5100000+2025*100
    model=lgb.LGBMRanker(objective="lambdarank",metric="ndcg",
                         label_gain=[0,1,3,7,15,31,63],
                         **model_common(seed))
    model.fit(xtr,relevance,group=groups,categorical_feature=cats)
    scores=np.asarray(model.predict(xva),dtype=float)
    check(len(scores)==17 and np.isfinite(scores).all(),
          "COLD_SCORE_NONFINITE_OR_MISSING_HORSE")
    pred=valid[["_race_id","_horse_id"]].copy()
    pred["_score"]=scores
    ctx=add_ai_context(pred).sort_values("distill_ai_rank")
    check(ctx["distill_ai_rank"].tolist()==list(range(1,18)),
          "ORIGINAL_NEW_PREDICTION_RANK_DRIFT")

    result={
      "contract":"KEIBA_202505050812_REAL_BLIND_ORIGINAL_ABILITY_V1",
      "race_id":TARGET,
      "status":"REAL_SOURCE_ORIGINAL_L1_ABILITY_ONLY_RISK_AND_BET_PENDING",
      "training_years":[2023,2024],
      "training_horse_rows":len(train),
      "training_races":int(train["_race_id"].nunique()),
      "ability_feature_columns":len(xtr.columns),
      "ability_original_seed":seed,
      "original_years_state_history":[2019,2020,2021,2022,2023,2024,2025],
      "target_horse_count":17,
      "source_model_code_reused":True,
      "target_saved_historical_predictions_read":False,
      "selected_target_outcomes_or_final_odds_used":False,
      "genuine_asof_source_timestamps_verified":False,
      "risk_calculated":False,
      "l2_1_ready":False,"l3_v3_ready":False,
      "horse_scores":[{
        "horse_id":str(r["_horse_id"]),
        "ai_raw_rank":int(r["distill_ai_rank"]),
        "ability_score":float(r["_score"]),
        "ability_score_z":float(r["distill_score_z"]),
        "ability_gap_to_best":float(r["distill_score_gap_to_best"])
      } for _,r in ctx.iterrows()],
    }
    path=root/"BLIND-202505050812-original-ability-not-outcome-verified.json"
    path.write_text(json.dumps(result,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n")
    print("BLIND_REAL_202505050812_ORIGINAL_ABILITY_PRODUCED",json.dumps({
      k:v for k,v in result.items() if k!="horse_scores"},sort_keys=True),flush=True)
    print("BLIND_REAL_202505050812_NEW_TOP5_HORSE_IDS",json.dumps(
        result["horse_scores"][:5],ensure_ascii=False),flush=True)
    # Freeze fresh ability BEFORE opening any other sources for prior OOS Risk.
    # These OOS teacher files contain only 2021-24 historical market labels,
    # never the selected 2025 race outcome, price or saved prediction.
    from rebuild_blind_original_2025_risk_from_prior_oos_v1 import train_and_score
    require_risk_inputs=[root/f"original-{y}-core4-base.csv.gz" for y in range(2019,2025)]
    require_risk_inputs += [root/f"original-{y}-daybias.csv.gz" for y in range(2019,2025)]
    require_risk_inputs += [root/f"original-{y}-prior-market.csv.gz" for y in range(2021,2025)]
    check(all(p.is_file() for p in require_risk_inputs),
          "ORIGINAL_2021_2024_OOS_RISK_TRAIN_INPUTS_INCOMPLETE")
    risk_result=train_and_score(root,regimes,regime_columns,valid,ctx)
    check({h["horse_id"] for h in risk_result["horse_risk"]}==
          {h["horse_id"] for h in result["horse_scores"]},
          "NEW_ABILITY_VS_NEW_RISK_17_HORSE_SET_DRIFT")
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    x=p.parse_args()
    main_pipeline(x.root)

if __name__=="__main__":main()
