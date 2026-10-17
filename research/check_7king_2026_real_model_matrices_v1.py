#!/usr/bin/env python3
"""Check actual frozen 7-King LightGBM matrix construction on 2026 blind data.

Reuses emit_l1_to_l2.load_validation_frame unmodified. This is ONLY an input
adapter check: no booster/model file is read; no scores or rankings are made.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import tempfile

import pandas as pd
from emit_l1_to_l2 import load_validation_frame

ALIAS = ("core4","pedlegacy","condition","full","light","pedv1","condrc")
DAYS = {"2026-10-10":(24,315),"2026-10-17":(36,471)}
JOIN_CONTRACT = "KEIBA_BACKFILL_690_PLUS_ORIGINAL_V4_238_UNSCORED_V1"
IDENTITY = {"_race_id","_horse_id","_race_date"}
DENIED = {"target","market_outcome","actual_finish_position","final_win_odds","payouts"}

def check(test,why):
    if not test:
        raise ValueError(why)

def matrix_gate(source: Path, schemas: Path, output: Path):
    source = source.resolve()
    schemas = schemas.resolve()
    result = {"contract":"KEIBA_FROZEN_7KING_2026_MODEL_MATRIX_V1",
              "source":"PERSISTED_690_PLUS_ORIGINAL_V4_238",
              "source_history_recalculated":False,
              "blind_year":2026,
              "2026_target_results_used":False,
              "model_loaded":False,
              "model_trained":False,
              "model_scored":False,
              "approved_for_betting":False,
              "races":0,"horse_rows":0,"kings":[]}
    with tempfile.TemporaryDirectory(prefix="l1-seven-king-2026-matrix-") as temp:
        temp_file = Path(temp)/"blind-targets.jsonl"
        row_set=set()
        with temp_file.open("w",encoding="utf-8") as fh:
            for day, (nr,nh) in DAYS.items():
                files=sorted((source/day).glob("*.json"))
                check(len(files)==nr,"SOURCE_DAY_RACE_COUNT:"+day)
                count=0
                for file in files:
                    doc=json.loads(file.read_text(encoding="utf-8"))
                    rid=file.stem
                    check(doc.get("contract")==JOIN_CONTRACT and
                          doc.get("race_id")==rid and doc.get("race_date")==day and
                          doc.get("feature_count")==928,
                          "EXPECTED_ORIGINAL_928_INPUT:"+rid)
                    check(doc.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET" and
                          doc.get("used_target_results_or_final_odds") is False and
                          doc.get("2026_results_read_or_evaluated") is False,
                          "2026_OUTCOMES_OR_READINESS_DRIFT:"+rid)
                    for row in doc["rows"]:
                        hid=str(row.get("_horse_id") or "")
                        check(hid.isdigit() and row.get("_race_id")==rid and
                              row.get("_race_date")==day and (rid,hid) not in row_set,
                              "HORSE_DATE_ID_DRIFT:"+rid)
                        row_set.add((rid,hid))
                        features={k:v for k,v in row.items() if k not in IDENTITY}
                        check(len(features)==928 and not set(features).intersection(DENIED),
                              "FORBIDDEN_OR_WRONG_FEATURES:"+rid)
                        payload={
                            "race_id":rid, "horse_id":hid,
                            "features":{"race_date":day, **features},
                        }
                        fh.write(json.dumps(payload,ensure_ascii=False,
                            separators=(",",":"),allow_nan=False)+"\n")
                        count+=1
                check(count==nh,"TARGET_DAY_HORSE_COUNT:"+day)
                result["races"]+=nr
                result["horse_rows"]+=count
        check(result["races"]==60 and result["horse_rows"]==786,
              "TARGET_COUNT_60_786")
        ordered_ids=None
        for alias in ALIAS:
            schema=json.loads((schemas/alias/"schema.json").read_text(encoding="utf-8"))
            expected=schema.get("feature_order")
            cats=schema.get("category_levels") or {}
            check(isinstance(expected,list) and bool(expected) and
                  len(expected)==len(set(expected)) and
                  isinstance(cats,dict),"INVALID_ARCHIVED_2025_SCHEMA:"+alias)
            # This unmodified original helper performs identical categorical/numeric
            # conversions and feature ordering as the frozen L1→L2 emitter.
            frame, ids=load_validation_frame(
                temp_file,"2026-10-10","2026-10-17",expected,cats
            )
            check(frame.shape==(786,len(expected)),"ORIGINAL_EMITTER_MATRIX_SHAPE:"+alias)
            identity=[(x["race_id"],x["horse_id"],x["race_date"]) for x in ids]
            check(set((r,h) for r,h,_ in identity)==row_set and
                  len(identity)==786 and
                  all(date in DAYS for _,_,date in identity),
                  "ORIGINAL_EMITTER_TARGET_ID_MISMATCH:"+alias)
            if ordered_ids is None:ordered_ids=identity
            check(ordered_ids==identity,"EXPERT_MATRIX_ROW_ORDER_DRIFT:"+alias)
            check(list(frame.columns)==expected,"EXPERT_MATRIX_COL_ORDER_DRIFT:"+alias)
            check(all(pd.api.types.is_numeric_dtype(frame[k]) or
                      isinstance(frame[k].dtype,pd.CategoricalDtype)
                      for k in expected),
                  "MATRIX_NON_NUMERIC_UNEXPECTED:"+alias)
            missing={name:int(frame[name].isna().sum()) for name in expected
                     if frame[name].isna().any()}
            categorical_unknown={name:missing.get(name,0) for name in cats
                                 if name in frame and missing.get(name,0)}
            completely_null=sorted(k for k,count in missing.items() if count==786)
            numeric_conversions=Counter()
            for k in expected:
                if k not in cats and not pd.api.types.is_numeric_dtype(frame[k]):
                    numeric_conversions[k]+=1
            check(not numeric_conversions,"NUMERIC_CONVERSION_FAILED:"+alias)
            row={"alias":alias,
                 "required_columns":len(expected),
                 "matrix_rows":len(frame),
                 "categorical_columns":len([k for k in expected if k in cats]),
                 "all_null_columns":completely_null,
                 "columns_with_some_null":len(missing),
                 "categorical_null_counts":categorical_unknown,
                 "model_loaded":False,
                 "model_scored":False}
            result["kings"].append(row)
            print("ORIGINAL_2026_SEVEN_KING_MATRIX "+json.dumps(
                row,ensure_ascii=False,sort_keys=True),flush=True)
    check(len(result["kings"])==7,"SEVEN_KING_MATRICES_NOT_SEVEN")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",
                      encoding="utf-8")
    print("FROZEN_2026_MATRIX_READY_NO_MODELS_OR_2026_OUTCOMES "+json.dumps({
        "races":result["races"],"rows":result["horse_rows"],
        "experts":len(result["kings"]),"model_scored":False
    },sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--joined",type=Path,required=True)
    p.add_argument("--schemas",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    matrix_gate(a.joined,a.schemas,a.report)

if __name__=="__main__":
    main()
