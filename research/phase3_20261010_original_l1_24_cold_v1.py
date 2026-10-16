#!/usr/bin/env python3
"""Original 2026 unknown-year CORE4+19+22 L1 LambdaRank: one model, 24 races.

Use only original source 2019-2025 labelled history to form the full
strict-prior 2024-2025 ability training fold. Every 2026 target row is
unscored, and original 690 inputs were independently verified in Phase3.
This is L1 original single LambdaRank ability, NOT original 7-king Council,
Outsider, Correction Risk, L1.75 EV, L2.1 or L3 outputs.
No 2026 evaluation, saved target predictions, target outcomes or odds.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

import run_l1_state_transition_v1 as state_module
import run_l1_residual_regime_full_pairwise_v1 as regime_module
from run_l1_objective_rebuild_v1 import META, model_common, prepare, read_year, select_variant
from run_l1_ceiling_audit_v1 import FULL_FORBIDDEN, strip_forbidden
from run_l1_sameday_track_bias_v1 import _build_same_day_bias_year, bias_cols, speed_cols
from run_l1_market_correction_distill_v1 import add_ai_context
from run_l1_state_transition_v1 import read_compact_state_source
from audit_2026_generated_core4_v1 import CORE4_SHA, EXPECTED, digest

PRIOR_YEARS=tuple(range(2019,2026))
FIT_YEARS=(2024,2025)
TARGET_YEAR=2026
TARGET_DATE="2026-10-10"
N_RACES=24
N_HORSES=315
N_FEATURES=690
IDENTITY={"_race_id","_horse_id","_race_date"}
TRAIN_MODEL_SEED=5100000+TARGET_YEAR*100


def need(cond,reason):
    if not cond:
        raise ValueError(reason)


def id_keys(df):
    return set(map(tuple,df[["_race_id","_horse_id"]].astype(str).values))


def csv(path):
    return pd.read_csv(path,compression="infer",low_memory=False,dtype={
        "_race_id":"string","_horse_id":"string","_race_date":"string"})


def export_one(source,year,dest):
    need(year in PRIOR_YEARS,"LABELLED_HISTORY_MUST_BE_PRE_2026")
    source=Path(source)
    dest=Path(dest)
    dest.mkdir(parents=True,exist_ok=True)
    # Original source reader accepts valid historical results; not a single
    # selected target (all 2026 results remain unseen).
    compact=read_compact_state_source(source,year)
    need(not compact.empty,"EMPTY_ORIGINAL_PRIOR_STATE")
    need(compact["_race_date"].astype(str).lt(TARGET_DATE).all(),
         "FUTURE_OUTCOME_IN_ORIGINAL_PRIOR_HISTORY")
    need(not compact.duplicated(["_race_id","_horse_id"]).any(),
         "DUPLICATE_HISTORICAL_COMPACT_KEY")
    compact.to_csv(dest/f"original-complete-prior-{year}.csv.gz",
                   index=False,compression="gzip")
    result={"year":year,"state_rows":len(compact),
            "state_races":int(compact["_race_id"].nunique()),
            "strictly_before_2026_target":True}
    if year in FIT_YEARS:
        # Frozen yearly CORE4 / day bias readers. Identical original formulas
        # to the successful 2025 Arima reconstruction; only fit years update
        # from 2023/24 to the 2026 holdout's true prior 2024/25.
        train=select_variant(strip_forbidden(read_year(source,year)),"core4")
        names=sorted(set(train)-META)
        need(len(names)==EXPECTED and digest(names)==CORE4_SHA,
             "ORIGINAL_PRIOR_TRAIN_CORE4_649_SHA_DRIFT:"+str(year))
        need(not(set(train)&FULL_FORBIDDEN),"FORBIDDEN_ORIGINAL_TRAIN_KEYS")
        need(train["_race_id"].astype(str).str[:4].eq(str(year)).all(),
             "PRIOR_TRAIN_FOLD_YEAR_DRIFT")
        bias_year,bias,coverage=_build_same_day_bias_year(year,source)
        need(bias_year==year and not bias.duplicated(["_race_id","_horse_id"]).any(),
             "TRAIN_DAYBIAS_SOURCE_DUPLICATE")
        need(id_keys(train).issubset(id_keys(bias)),"TRAIN_BIAS_HORSE_SET_DRIFT")
        fit=bias_cols(bias,with_fit=True)
        need(len(fit)==19,"FROZEN_TRAIN_19_DAYBIAS_DRIFT")
        train.to_csv(dest/f"original-core4-train-{year}.csv.gz",
                     index=False,compression="gzip")
        bias.to_csv(dest/f"original-daybias-{year}.csv.gz",
                    index=False,compression="gzip")
        result.update({"ability_train_rows":len(train),
                       "ability_train_races":int(train["_race_id"].nunique()),
                       "core4":len(names),"daybias":len(fit),
                       "original_formulas_unchanged":True})
    print("P3_L1_ORIGINAL_PRE_2026_EXPORT "+
          json.dumps(result,sort_keys=True),flush=True)


def _read_compact_pickle(item):
    year,path=item
    df=csv(path)
    need(df["_race_id"].astype(str).str[:4].eq(str(year)).all(),
         "COMPACT_RACE_ID_YEAR_DRIFT")
    need(df["_race_date"].astype(str).lt(TARGET_DATE).all(),
         "OUT_OF_FOLD_YEAR_HISTORY_TARGET_LEAK")
    return year,df


def reconstruct_train_regime(root):
    paths={y:str(root/f"original-complete-prior-{y}.csv.gz") for y in PRIOR_YEARS}
    need(all(Path(p).is_file() for p in paths.values()),
         "MISSING_FULL_2019_2025_PRIOR_COMPACT")
    # Only scoped YEARS and a read-only CSV adapter differ from the unchanged
    # original state/lag/regime formula. No source file or model is edited.
    with patch.object(state_module,"YEARS",PRIOR_YEARS), \
         patch.object(state_module,"_read_state_year_item",
                      side_effect=_read_compact_pickle), \
         patch.object(state_module.concurrent.futures,
                      "ProcessPoolExecutor",
                      concurrent.futures.ThreadPoolExecutor), \
         patch.object(regime_module,"YEARS",PRIOR_YEARS):
        state,_,_,_,coverage=state_module.build_state_features(paths)
        regime,regime_cols=regime_module.build_regime_features(state)
    speed=speed_cols(regime_cols)
    need(len(speed)==22,"FROZEN_L1_REGIME_SPEED_22_DRIFT")
    return regime,speed


def targets(root):
    docs=[]
    ids=set()
    for path in sorted(Path(root).glob("2026*.json")):
        doc=json.loads(path.read_text(encoding="utf-8"))
        rid=doc["race_id"]
        need(rid not in ids,"DUPLICATE_2026_TARGET_RACE")
        ids.add(rid)
        need(doc["contract"]=="KEIBA_20261010_24_ORIGINAL_690_UNSCORED_INPUT_V1",
             "TARGET_NOT_VERIFIED_ORIGINAL_690")
        need(doc["feature_count"]==N_FEATURES and
             doc["original_19_bias_values_verified"] is True and
             doc["original_22_speed_regime_values_verified"] is True,
             "690_NUMERICAL_PARITY_NOT_ATTESTED")
        need(doc["race_date"]==TARGET_DATE and rid.startswith(str(TARGET_YEAR)),
             "TARGET_ID_OR_DATE_DRIFT")
        for k in ("used_target_labels","used_result_payout_or_final_odds",
                  "used_2026_historical_oos_predictions"):
            need(doc[k] is False,"TARGET_FORBIDDEN_SOURCE_"+k)
        need(doc["l1_scored"] is False,"TARGET_PRECOMPUTED_SCORE_PRESENT")
        need(doc["source_capture_asof_independently_verified"] is False,
             "ASOF_NOT_ALLOWED_TO_BE_SELF_ASSERTED")
        need(3<=len(doc["rows"])<=24,"INVALID_STARTER_COUNT")
        docs.append(doc)
    need(len(docs)==N_RACES,"NOT_ALL_24_UNSCORED_TARGET_RACES_PRESENT")
    data=[]
    for doc in docs:
        for item in doc["rows"]:
            need(item["_race_id"]==doc["race_id"] and
                 item["_race_date"]==TARGET_DATE,
                 "UNSCORED_HORSE_RACE_DATE_MISMATCH")
            need(len(set(item)-IDENTITY)==N_FEATURES,
                 "TARGET_690_MODEL_COLUMNS_DRIFT")
            data.append(item)
    df=pd.DataFrame.from_records(data)
    need(len(df)==N_HORSES and
         df["_race_id"].nunique()==N_RACES and
         not df.duplicated(["_race_id","_horse_id"]).any(),
         "TARGET_24_RACE_POPULATION_DRIFT")
    need(not (set(df.columns)&FULL_FORBIDDEN),
         "2026_TARGET_HAS_ORIGINAL_FORBIDDEN_FEATURES")
    bad=[c for c in df.columns if c.startswith(("P__","distill_","risk_")) or
         c in ("_finish","_is_win","_is_top3","final_win_odds",
               "final_popularity","_score","target","market_rank")]
    need(not bad,"2026_TARGET_HAS_LABEL_PREDICTION_OR_PRICE:"+repr(bad))
    return docs,df


def score_all(root,infer_dir,out_dir):
    root=Path(root)
    docs,target=targets(infer_dir)
    regime,speed=reconstruct_train_regime(root)
    originals=[]
    fit=None
    for year in FIT_YEARS:
        base=csv(root/f"original-core4-train-{year}.csv.gz")
        bias=csv(root/f"original-daybias-{year}.csv.gz")
        need(base["_race_id"].astype(str).str[:4].eq(str(year)).all(),
             "TRAIN_YEAR_DRIFT")
        need(set(base)-META==set(target)-IDENTITY-set(speed)-
             {c for c in target if c.startswith(("daybias_","fit_","horse_front_","horse_inner_"))},
             "TRAIN_CORE4_649_DOES_NOT_MATCH_TARGET")
        # Full original 2024-25 training: historical year-wide same-day
        # bias and strictly prior state, using frozen 690 feature arithmetic.
        if fit is None:
            fit=bias_cols(bias,with_fit=True)
        need(fit==bias_cols(bias,with_fit=True) and len(fit)==19,
             "DAY_BIAS_TRAIN_19_SCHEMATIC_DRIFT")
        merged=bias[["_race_id","_horse_id",*fit]].merge(
            regime[year][["_race_id","_horse_id",*speed]],
            on=["_race_id","_horse_id"],how="left",validate="one_to_one")
        original=state_module.attach_state(base,merged,fit+speed)
        need(len(original)==len(base),"TRAIN_JOIN_DROPPED_A_HORSE")
        need(not original[["_race_id","_horse_id"]].duplicated().any(),
             "TRAIN_DUPLICATED_HORSE")
        originals.append(original)
        print("P3_L1_ORIGINAL_TRAIN_FOLD_READY "+
              json.dumps({"year":year,"horses":len(original),
                          "races":int(original["_race_id"].nunique()),
                          "690_model_features":len(set(original)-META)},
                         sort_keys=True),flush=True)
    train=pd.concat(originals,ignore_index=True)
    train_features=set(train)-META
    valid_features=set(target)-IDENTITY
    need(train_features==valid_features and len(train_features)==N_FEATURES,
         "ORIGINAL_2024_2025_TRAIN_VS_2026_TARGET_690_SCHEMA_DRIFT")
    need(train["_race_id"].astype(str).str[:4].astype(int).isin(FIT_YEARS).all(),
         "ABILITY_TRAIN_MUST_BE_ONLY_2024_2025")
    train=train.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    target=target.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    xtr,xva,categorical=prepare(train,target)
    need(xtr.shape==(len(train),N_FEATURES) and
         xva.shape==(N_HORSES,N_FEATURES),"ORIGINAL_PREPARE_690_MATRIX_SHAPE")
    need(not set(xva.columns).symmetric_difference(xtr.columns),
         "ORIGINAL_PREPARE_COLUMN_ORDER_DRIFT")
    rel=np.select([train["_finish"].eq(x) for x in range(1,7)],
                  [6,5,4,3,2,1],default=0).astype(int)
    groups=train.groupby("_race_id",sort=False).size().tolist()
    need(sum(groups)==len(train),"ORIGINAL_LAMBDARANK_GROUP_DRIFT")
    model=lgb.LGBMRanker(
        objective="lambdarank",metric="ndcg",
        label_gain=[0,1,3,7,15,31,63],
        **model_common(TRAIN_MODEL_SEED))
    model.fit(xtr,rel,group=groups,categorical_feature=categorical)
    scores=np.asarray(model.predict(xva),dtype=float)
    need(len(scores)==N_HORSES and np.isfinite(scores).all(),
         "NONFINITE_ORIGINAL_2026_L1_SCORE")
    prediction=target[["_race_id","_horse_id"]].copy()
    prediction["_score"]=scores
    ranked=add_ai_context(prediction)
    need(ranked[["_race_id","_horse_id"]].duplicated().sum()==0,
         "ORIGINAL_ABILITY_PRED_HORSE_ID_DUPLICATE")
    out=Path(out_dir)
    out.mkdir(parents=True,exist_ok=True)
    report=[]
    for doc in docs:
        rid=doc["race_id"]
        sub=ranked.loc[ranked["_race_id"].astype(str).eq(rid)].sort_values(
            ["distill_ai_rank","_horse_id"])
        need(len(sub)==len(doc["rows"]),"PREDICTED_RACE_HORSE_COUNT_DRIFT")
        need(sorted(sub["distill_ai_rank"].astype(int).tolist())==
             list(range(1,len(doc["rows"])+1)),
             "ORIGINAL_ABILITY_RACE_RANK_COVERAGE_DRIFT")
        need(set(sub["_horse_id"].astype(str))==
             {x["_horse_id"] for x in doc["rows"]},
             "ORIGINAL_ABILITY_RACE_HORSE_ID_DRIFT")
        horses=[{
            "horse_id":str(row["_horse_id"]),
            "ai_raw_rank":int(row["distill_ai_rank"]),
            "ability_score":float(row["_score"]),
            "ability_score_z":float(row["distill_score_z"]),
            "ability_gap_to_best":float(row["distill_score_gap_to_best"]),
        } for _,row in sub.iterrows()]
        prediction_doc={
            "contract":"KEIBA_20261010_24_REAL_ORIGINAL_L1_ABILITY_V1",
            "race_id":rid,"race_date":TARGET_DATE,
            "status":"ORIGINAL_L1_ABILITY_ONLY_NOT_COUNCIL_OR_L175_L3",
            "trained_once_across_all_24":True,
            "historical_training_years":[2024,2025],
            "2026_training_or_score_result_label_used":False,
            "source_prebet_asof_independently_verified":False,
            "target_result_or_final_odds_used":False,
            "model_feature_count":N_FEATURES,
            "model_seed":TRAIN_MODEL_SEED,
            "pure_original_L1_ability_not_seven_king":True,
            "original_7king_outsider_ready":False,
            "l175_p3_ready":False,
            "l21_l3_ready":False,
            "horse_scores":horses,
        }
        (out/(rid+".json")).write_text(
            json.dumps(prediction_doc,ensure_ascii=False,allow_nan=False)+"\n",
            encoding="utf-8")
        top3=[h["horse_id"] for h in horses[:3]]
        report.append({"race_id":rid,"horses":len(horses),
                       "raw_rank_coverage_ok":True,
                       "top3_ids":top3})
        print("P3_24_ORIGINAL_REAL_L1_RACE "+
              json.dumps(report[-1],sort_keys=True),flush=True)
    need(len(report)==N_RACES,"NOT_ALL_24_ORIGINAL_RACE_OUTPUTS")
    result={
        "contract":"KEIBA_20261010_24_REAL_ORIGINAL_L1_ABILITY_BATCH_V1",
        "status":"24_REAL_ORIGINAL_L1_ABILITY_RANKINGS_CREATED",
        "races":len(report),"target_horses":N_HORSES,
        "fit_years":list(FIT_YEARS),
        "train_horses":len(train),"train_races":int(train["_race_id"].nunique()),
        "model_features":N_FEATURES,"model_seed":TRAIN_MODEL_SEED,
        "real_original_lambdarank_trained_once":True,
        "original_seven_king_outsider_l175_and_l21_l3_not_yet_executed":True,
        "no_2026_training_or_performance_evaluation":True,
        "target_results_and_final_odds_excluded":True,
        "historical_source_asof_certified":False,
        "frozen_l1_model_math_changed":False,
        "race_results":report,
    }
    (out/"manifest.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("P3_24_ORIGINAL_L1_ABILITY_BATCH_SUCCESS "+
          json.dumps({k:v for k,v in result.items() if k!="race_results"},
                     sort_keys=True),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",required=True,choices=("export","score"))
    ap.add_argument("--source")
    ap.add_argument("--year",type=int)
    ap.add_argument("--prior-dir",required=True)
    ap.add_argument("--inputs-690")
    ap.add_argument("--output")
    args=ap.parse_args()
    if args.mode=="export":
        need(args.source is not None and args.year is not None,
             "EXPORT_YEAR_AND_SOURCE_REQUIRED")
        export_one(args.source,args.year,args.prior_dir)
    else:
        need(args.inputs_690 is not None and args.output is not None,
             "SCORE_24_INPUT_AND_OUTPUT_REQUIRED")
        score_all(args.prior_dir,args.inputs_690,args.output)


if __name__=="__main__":
    main()
