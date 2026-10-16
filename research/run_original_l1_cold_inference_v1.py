#!/usr/bin/env python3
"""Cold-score an UNSCORED race using the exact original L1 learning functions.

Training side: prior two-year labelled horse features + historical OOF
market-correction teachers (past years only). Target side: as-of pre-race
features for ALL starters, with NO final odds, results, payouts or saved
target predictions. Reuses original modules, hyperparameters and seeds.
This is NOT an estimator promotion or claim of parity with historical OOS:
feature-generation provenance must be audited separately.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from run_l1_objective_rebuild_v1 import prepare, model_common
from run_l1_market_correction_distill_v1 import add_ai_context
from run_l1_market_correction_deepdive_v1 import fit_risk_full
import lightgbm as lgb

CONTRACT="KEIBA_ORIGINAL_L1_COLD_INFERENCE_V1"
BANNED_TARGET={
    "_finish","_is_win","_is_top3","_target","_target_win","_target_over3",
    "_target_gap","_target_gap_clip","_target_up","target","finish",
    "finish_position","result","result_rank","payout","payout_yen",
    "return_yen","hit","hit_ticket","final_win_odds","final_place_odds",
    "market_rank","market_prob_norm","win_odds_at_cutoff",
    "P__ANCHOR_FULL","distill_ai_rank","distill_score_z",
    "distill_score_gap_to_best","_score","risk",
}
BANNED_PREFIXES=("target_", "payout_", "result_", "final_", "after_race_", "postrace_")
NUMBERS=re.compile(r"^20[0-9]{10}$")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def year(value):
    require(type(value) is int and 2023<=value<=2026, "INVALID_HOLDOUT_YEAR")
    return value


def validate(target, ability_train, teacher, target_year, *,strict_full_field=True):
    yr=year(target_year)
    for df,label in ((target,"TARGET"),(ability_train,"ABILITY_TRAIN"),
                     (teacher,"PRIOR_TEACHER")):
        require(isinstance(df,pd.DataFrame) and not df.empty,
                "EMPTY_"+label)
        require({"_race_id","_horse_id","_race_date"}.issubset(df.columns),
                "MISSING_IDENTITIES_"+label)
        require(not df.duplicated(["_race_id","_horse_id"]).any(),
                "DUPLICATE_HORSE_"+label)
        dates=pd.to_datetime(df["_race_date"],errors="coerce")
        require(dates.notna().all(), "INVALID_RACE_DATE_"+label)
        require(df["_race_id"].astype(str).map(
            lambda v:bool(NUMBERS.fullmatch(v))).all(),
            "NONCANONICAL_NETKEIBA_RACE_ID_"+label)
        require(all(str(rid)[:4]==str(date)[:4]
                    for rid,date in zip(df["_race_id"],df["_race_date"])),
                "RACE_YEAR_MISMATCH_"+label)

    target_cols=set(target.columns)
    banned=sorted(k for k in target_cols if k in BANNED_TARGET or
                  k.startswith(BANNED_PREFIXES) or
                  k.startswith("P__") or k.startswith("outcome_"))
    require(not banned,"TARGET_HAS_RESULT_MARKET_OR_CACHED_PREDICTION:"+",".join(banned))
    require(target["_race_id"].nunique()==1,"ONLY_ONE_UNSCORED_RACE_PER_INVOCATION")
    require(all(str(x)[:4]==str(yr) for x in target["_race_id"]),
            "TARGET_YEAR_MISMATCH")
    require(3<=len(target)<=24,"INVALID_FULL_STARTER_COUNT")
    require("_finish" in ability_train and "_is_win" in ability_train and
            "_is_top3" in ability_train,"ABILITY_PRIOR_YEAR_LABELS_REQUIRED")
    require(ability_train["_finish"].notna().all(),
            "ABILITY_TRAIN_FINISH_POSITION_MISSING")
    require({"_target_over3","distill_ai_rank","distill_score_z",
             "distill_score_gap_to_best"}.issubset(teacher.columns),
            "ORIGINAL_CORRECTION_RISK_TEACHER_COLUMNS_MISSING")
    require(teacher["_target_over3"].isin((0,1)).all(),
            "INVALID_CORRECTION_TEACHER")
    train_years=sorted(set(ability_train["_race_id"].astype(str).str[:4].astype(int)))
    teacher_years=sorted(set(teacher["_race_id"].astype(str).str[:4].astype(int)))
    require(train_years==[yr-2,yr-1],"ABILITY_MUST_TRAIN_TWO_PRIOR_YEARS")
    require(teacher_years==list(range(2021,yr)),
            "CORRECTION_RISK_MUST_USE_ALL_PRIOR_OOS_TEACHER_YEARS")
    if strict_full_field:
        require(target["_horse_id"].nunique()==len(target),
                "FULL_FIELD_DUPLICATE_HORSE")
        require(not target["_horse_id"].astype(str).eq("").any(),
                "FULL_FIELD_HORSE_ID_MISSING")
    # Source provenance is machine-checkable but does NOT authenticate a
    # self-declared timestamp; real upstream collectors need independent checks.
    return train_years,teacher_years


def _numeric_feature_columns(df):
    return {c for c in df if c not in
            {"_race_id","_horse_id","_race_date","_finish","_is_win","_is_top3",
             "_target_over3","_target_gap","_target_gap_clip","_target_up"}}


def score_one(target, ability_train, past_oos_risk_teacher, target_year):
    train_years,teacher_years=validate(target,ability_train,past_oos_risk_teacher,target_year)
    # EXACT source model: run_l1_residual_regime_full_pairwise_v1.fit_direct_rank
    # uses this prepare + lightgbm Ranker + model_common(seed), with the fixed
    # 6,5,...,1,0 relevance grading. We cannot use its function directly
    # because that research function requires target-year _finish and labels.
    train=ability_train.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    valid=target.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    train_features=_numeric_feature_columns(train)
    target_features=_numeric_feature_columns(valid)
    require(train_features==target_features,
            "ORIGINAL_ABILITY_FEATURE_SCHEMA_DRIFT")
    # Do not let target supply a future model's expected prior score as feature.
    require(not any(x.startswith(("distill_","risk_","market_")) for x in target_features),
            "TARGET_MODEL_OR_MARKET_FEATURE_LEAK")
    xtr,xva,cats=prepare(train,valid)
    relevance=np.select([
        train["_finish"].eq(1),train["_finish"].eq(2),
        train["_finish"].eq(3),train["_finish"].eq(4),
        train["_finish"].eq(5),train["_finish"].eq(6)],
        [6,5,4,3,2,1],default=0).astype(int)
    groups=train.groupby("_race_id",sort=False).size().tolist()
    require(int(sum(groups))==len(train),"ABILITY_GROUP_SIZE_DRIFT")
    original_ability_seed=5100000+target_year*100
    ability=lgb.LGBMRanker(
        objective="lambdarank",metric="ndcg",
        label_gain=[0,1,3,7,15,31,63],
        **model_common(original_ability_seed))
    ability.fit(xtr,relevance,group=groups,categorical_feature=cats)
    scores=np.asarray(ability.predict(xva),dtype=float)
    require(np.isfinite(scores).all(),"NONFINITE_ORIGINAL_ABILITY_OUTPUT")
    pre=pd.DataFrame({"_race_id":valid["_race_id"].astype(str),
                      "_horse_id":valid["_horse_id"].astype(str),
                      "_score":scores})
    context=add_ai_context(pre)
    # Risk training must be built from ORIGINAL prior-year OOS teacher data,
    # never relabel target with FINAL market and never fetch target OOS scores.
    prior=past_oos_risk_teacher.copy()
    drop_teacher=["_target_gap","_target_gap_clip","_target_up",
                  "_target_over3","market_rank","final_win_odds"]
    # attach the complete, unsupervised original ability context back to
    # pre-race target features. This is FULL_PRERACE, not a rank-only proxy.
    valid_risk=valid.merge(
        context[["_race_id","_horse_id","_score","distill_ai_rank",
                 "distill_field_size","distill_ai_rank01","distill_score_z",
                 "distill_score_gap_to_best","distill_is_ai_top1",
                 "distill_is_ai_top3","distill_is_ai_top6"]],
        on=["_race_id","_horse_id"],how="left",validate="one_to_one")
    # Expected source risk variant FULL_PRERACE excludes neg_ features.
    cols=[c for c in valid_risk.columns if not c.startswith("neg_")]
    valid_risk=valid_risk[cols].copy()
    expected_teacher_features=set(valid_risk.columns) - set(
        ["_race_id","_horse_id","_race_date"])
    actual_teacher_features=set(prior.columns) - set(
        ["_race_id","_horse_id","_race_date","_finish","_is_win","_is_top3",
         *drop_teacher])
    require(expected_teacher_features==actual_teacher_features,
            "ORIGINAL_RISK_FULL_PRERACE_FEATURE_SCHEMA_DRIFT")
    require(not any(c.startswith("neg_") for c in prior.columns),
            "ORIGINAL_RISK_MUST_USE_FULL_PRERACE_NO_NEG")
    original_risk_seed=8200000+target_year*100
    risks,_=fit_risk_full(prior,valid_risk,original_risk_seed)
    require(len(risks)==len(valid) and np.isfinite(risks).all() and
            ((risks>=0)&(risks<=1)).all(),"NONFINITE_CORRECTION_RISK_OUTPUT")
    bykey=context.set_index(["_race_id","_horse_id"])
    out=[]
    for i,row in valid.iterrows():
        rid=str(row["_race_id"]); hid=str(row["_horse_id"])
        scored=bykey.loc[(rid,hid)]
        out.append({"race_id":rid,"horse_id":hid,
                    "ai_raw_rank":int(scored["distill_ai_rank"]),
                    "distill_score_z":float(scored["distill_score_z"]),
                    "distill_score_gap_to_best":float(scored["distill_score_gap_to_best"]),
                    "correction_risk":float(risks[i])})
    out.sort(key=lambda r:(r["ai_raw_rank"],r["horse_id"]))
    require(sorted(r["ai_raw_rank"] for r in out)==list(range(1,len(out)+1)),
            "ORIGINAL_RANK_COVERAGE_MISSING")
    return {"contract":CONTRACT,"status":"COLD_ORIGINAL_L1_ABILITY_RISK_ONLY",
            "race_id":out[0]["race_id"],
            "target_year":target_year,
            "ability_prior_years":train_years,
            "risk_prior_oos_teacher_years":teacher_years,
            "models":{"ability":"LAMBDA_RANK_CORE4_SAME_DAY_STATE_REGIME",
                      "risk":"FULL_PRERACE_OVER3_CORRECTION",
                      "source_original_training_config":True,
                      "target_prebet_feature_provenance_independently_verified":False},
            "source_data_phase":"SUPPLIED_PREBET_FEATURES_NOT_INDEPENDENTLY_VERIFIED",
            "prediction_before_bet_verified":False,
            "l2_1_ready":False,
            "l3_v3_ready":False,
            "historical_oos_predictions_used_as_target_input":False,
            "target_finish_result_or_final_odds_used":False,
            "horse_scores":out}


def frame(path):
    p=Path(path)
    opener=gzip.open if p.suffix==".gz" else open
    with opener(p,"rt",encoding="utf-8") as handle:
        data=json.load(handle)
    require(isinstance(data,dict) and isinstance(data.get("rows"),list),
            "EXPECTED_ROWS_JSON")
    return pd.DataFrame(data["rows"])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target-year",type=int,required=True)
    ap.add_argument("--prior-ability",required=True)
    ap.add_argument("--prior-oos-risk-teacher",required=True)
    ap.add_argument("--unscored-target",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    output=score_one(frame(args.unscored_target),frame(args.prior_ability),
                     frame(args.prior_oos_risk_teacher),args.target_year)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",
                    encoding="utf-8")
    print("ORIGINAL_L1_COLD_SCORE",output["race_id"],output["status"],
          len(output["horse_scores"]),flush=True)


if __name__=="__main__":
    main()
