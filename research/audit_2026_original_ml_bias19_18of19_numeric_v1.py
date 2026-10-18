#!/usr/bin/env python3
"""Independent frozen ORIGINAL ML 2026 October 17 19-bias partial numeric parity.

Never invent absent target STYLE input. Re-execute the ORIGINAL-ML c782a7e
BiasState.snapshot() function as an unmodified AST node and independently
compute 18 of 19 stored daily-bias values for 471 horses, including numeric
inner-position geometry and four interaction-null outputs. The 19th front
score requires frozen original target STYLE_FIELDS not in cached CORE4 649;
it is explicitly NOT numerically certified. Speed 22, CORE4 649 not certified.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math
from pathlib import Path
import numpy as np

ML_BIAS_ORIGINAL_GIT_BLOB="6b2124558790b317063c1d48e5c3b0ce9b6952e6"
SRC_SHA="d6607946292efe79fc5eef8eaf143e689c41f6a9"
DAY="2026-10-17"
FROZEN_BIAS_COLUMNS={
    "daybias_prior_races","daybias_front_gap_mean","daybias_front_gap_ewma",
    "daybias_inner_gap_mean","daybias_inner_gap_ewma",
    "daybias_last3_gap_mean","daybias_last3_gap_ewma",
    "daybias_winner_front_mean","daybias_winner_front_ewma",
    "daybias_winner_inner_mean","daybias_winner_inner_ewma",
    "daybias_speed_adv_mean","daybias_speed_adv_ewma",
    "horse_front_score","horse_inner_score","fit_front_mean",
    "fit_front_ewma","fit_inner_mean","fit_inner_ewma"}
MISSING={"horse_front_score"}
FIT=("fit_front_mean","fit_front_ewma","fit_inner_mean","fit_inner_ewma")

def need(cond,msg):
    if not cond:raise ValueError(msg)
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for blob in iter(lambda:f.read(1048576),b""):h.update(blob)
    return h.hexdigest()
def git_blob(path):
    blob=path.read_bytes()
    return hashlib.sha1(b"blob "+str(len(blob)).encode()+b"\x00"+blob).hexdigest()
def extracted_original_snapshot(source):
    need(git_blob(source)==ML_BIAS_ORIGINAL_GIT_BLOB,
         "ORIGINAL_ML_BIAS_SOURCE_NOT_C782_LOCKED")
    tree=ast.parse(source.read_text(encoding="utf8"),filename=str(source))
    choices=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="BiasState"]
    need(len(choices)==1 and
         {m.name for m in choices[0].body if isinstance(m,ast.FunctionDef)}
         >= {"__init__","snapshot","update"},"ORIGINAL_ML_BIAS_STATE_ABSENT")
    scope={"np":np,"math":math}
    exec(compile(ast.Module(body=choices,type_ignores=[]),str(source),"exec"),scope)
    return scope["BiasState"]

def number_equals(a,b):
    if b is None:
        return a is None
    return type(a) in (int,float) and math.isfinite(a) and a==b

def run(reference,delivery,output):
    need(not output.exists(),"REFUSE_OVERWRITE_ORIGINAL_NUMERIC_AUDIT")
    BiasState=extracted_original_snapshot(reference)
    base=delivery/"ml_features/delivery/690"
    index=json.loads((base/"index.json").read_text(encoding="utf8"))
    need(index.get("contract")=="KEIBA_BACKFILL_TO_ML_690_INDEX_V1" and
         index.get("total_races")==60 and index.get("total_horses")==786 and
         index.get("original_ml_full_numerical_parity_verified") is False and
         index.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET",
         "SAVED_BACKFILL_INDEX_CONTRACT_DRIFT")
    by_day={rid:v for rid,v in index["races"].items() if v["date"]==DAY}
    need(len(by_day)==36 and sum(x["starters"] for x in by_day.values())==471,
         "REAL_PRESTART_RACE_COUNT_DRIFT")
    total=0;value_count=0;front_present=0;records=[]
    for rid,meta in sorted(by_day.items()):
        file=delivery/meta["path"]
        need(file.is_file() and digest(file)==meta["sha256"],
             "SAVED_BACKFILL_690_SHA_MISMATCH:"+rid)
        obj=json.loads(file.read_text(encoding="utf8"))
        need(obj.get("contract")=="KEIBA_BACKFILL_TO_ML_690_UNSCORED_V1" and
             obj.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET" and
             obj.get("source_capture_asof_independently_verified") is False and
             obj.get("used_target_results_or_final_odds") is False and
             obj.get("starters")==meta["starters"] and
             obj.get("race_date")==DAY and obj.get("race_id")==rid,
             "690_ORIGINAL_SOURCE_NOT_PRESTART_UNSCORED:"+rid)
        rows=obj["rows"]
        snap=BiasState().snapshot()
        need(len(snap)==13 and set(snap)==FROZEN_BIAS_COLUMNS-MISSING-
             {"horse_inner_score",*FIT},"PINNED_ORIGINAL_BIAS_SNAPSHOT_SCHEMA_DRIFT")
        n=len(rows)
        seen=set()
        for horse in rows:
            hid=str(horse.get("_horse_id") or "")
            hno=horse.get("horse_number")
            need(hid and hid not in seen and
                 type(hno)is int and 1<=hno<=18 and
                 horse.get("_race_date")==DAY and horse.get("_race_id")==rid,
                 "INVALID_HORSE_ID_OR_NUMBER:"+rid)
            seen.add(hid)
            need(FROZEN_BIAS_COLUMNS.issubset(horse),"MISSING_19_ORIGINAL_BIAS_FIELDS:"+rid)
            expected={k:(None if not math.isfinite(v) else v) for k,v in snap.items()}
            expected["horse_inner_score"]=max(0.,min(1.,1.-(hno-1.)/max(1.,float(n-1))))
            for k in FIT:expected[k]=None
            need(len(expected)==18,"INDEPENDENT_ORIGINAL_ML_FEATURE_COUNT_NOT_18")
            for k,val in expected.items():
                actual=horse[k]
                need(number_equals(actual,val),
                     "ORIGINAL_ML_INDEPENDENT_BIAS_NUMERIC_MISMATCH:"+rid+
                     ":"+hid+":"+k+":observed="+str(actual)+":expected="+str(val))
                value_count+=1
            front=horse["horse_front_score"]
            need(front is None or
                 (type(front)in (int,float) and math.isfinite(front) and 0<=front<=1),
                 "SOURCE_HORSE_FRONT_SCORE_INVALID:"+rid+":"+hid)
            front_present+=front is not None
            total+=1
        records.append({"race_id":rid,"horses":len(rows),"original_ml_18_numerical_values_verified":len(rows)*18,
                        "horse_front_score_original_style_input_absent":True,
                        "other_649_22_independent_parity_checked":False})
    need(total==471 and value_count==8478 and len(records)==36,
         "INDEPENDENT_ORIGINAL_ML_BIAS_SCOPE_DRIFT")
    result={
        "contract":"KEIBA_ORIGINAL_ML_2026_1017_BIAS18_OF19_INDEPENDENT_VALUES_READONLY_V1",
        "original_ml_source_git_blob":ML_BIAS_ORIGINAL_GIT_BLOB,
        "backfill_690_reference_sha":SRC_SHA,
        "races":36,"horses":471,
        "independently_recomputed_original_ml_bias_cells":value_count,
        "original_ml_recomputed_bias_columns_per_horse":18,
        "original_ml_style_source_missing_bias_column":"horse_front_score",
        "horse_front_values_validated_range_only":471,
        "horse_front_values_nonnull":front_present,
        "original_ml_core4_649_independent_numeric_parity":False,
        "original_ml_speed22_independent_numeric_parity":False,
        "all_690_independent_numeric_parity_certified":False,
        "same_day_results_used":False,
        "model_training_or_prediction_invoked":False,
        "frozen_l21_invoked":False,"l3_v3_invoked":False,
        "ready_for_betting":False,"per_race":records}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print("FROZEN_ORIGINAL_ML_BIAS_18_OF19_2026_REAL_VALUE_PARITY "+
          json.dumps({"races":36,"horses":471,"values":value_count,
                      "front_requires_original_style":True,
                      "690_original_independent_parity":False,
                      "betting":False},sort_keys=True),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("original-ml-bias","backfill-690","output"):
        parser.add_argument("--"+key,required=True,type=Path)
    a=parser.parse_args()
    run(a.original_ml_bias,a.backfill_690,a.output)
