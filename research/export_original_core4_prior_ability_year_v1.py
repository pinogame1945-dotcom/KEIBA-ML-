#!/usr/bin/env python3
"""Build training-year CORE4 feature matrices with ORIGINAL frozen functions.

Only 2023/2024 labels are read, never the user's 2025 target race labels.
A full yearly source metadata row/date check protects Kaggle's .gz/.jsonl
compression-equivalent name resolution (not a substitute for blob SHA).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from run_l1_objective_rebuild_v1 import read_year, select_variant, META, open_text
from run_l1_ceiling_audit_v1 import strip_forbidden, FULL_FORBIDDEN

def ensure(ok,why):
    if not ok:raise ValueError(why)

def audit_source(path, year, expected_rows, first, last):
    count=0; observed_first=None; observed_last=None
    with open_text(path) as f:
        for line in f:
            if not line.strip():continue
            obj=json.loads(line)
            count+=1
            date=str((obj.get("features") or {}).get("race_date") or obj.get("race_date") or "")[:10]
            ensure(date.startswith(str(year)+"-"),"CROSS_YEAR_SOURCE_DRIFT")
            observed_first=min(date,observed_first) if observed_first else date
            observed_last=max(date,observed_last) if observed_last else date
    ensure((count,observed_first,observed_last)==(expected_rows,first,last),
           "ANNUAL_SNAPSHOT_MANIFEST_COUNT_OR_DATE_DRIFT")
    return count


def build(source, year, expected_rows, first,last,output_folder):
    ensure(year in (2023,2024),"ABILITY_YEAR_MUST_BE_2023_OR_2024")
    source=Path(source)
    folder=Path(output_folder);folder.mkdir(parents=True,exist_ok=True)
    rows=audit_source(source,year,expected_rows,first,last)
    raw=read_year(source,year)
    train=select_variant(strip_forbidden(raw),"core4")
    model_keys=sorted(set(train.columns)-set(META))
    ensure(model_keys and not(set(model_keys)&FULL_FORBIDDEN),
           "ORIGINAL_TRAIN_FORBIDDEN_FEATURES")
    ensure(train["_race_id"].astype(str).str[:4].eq(str(year)).all(),
           "SOURCE_TRAIN_FOLD_YEAR_LEAK")
    ensure(not train.duplicated(["_race_id","_horse_id"]).any(),
           "SOURCE_TRAIN_DUPLICATE_HORSE")
    ensure(train["_is_win"].isin([0,1]).all() and
           train["_is_top3"].isin([0,1]).all(),
           "SOURCE_TRAIN_LABELS_INVALID")
    # Exact frozen read_year accepts >=1 winner per race. Do not reject
    # legitimate dead heats (multiple horses both credited with first).
    winners=train.groupby("_race_id")["_is_win"].sum()
    ensure(winners.ge(1).all(),"SOURCE_TRAIN_WINNER_COVERAGE_DRIFT")
    multiple_winner_races=int(winners.gt(1).sum())
    schema_sha=hashlib.sha256(json.dumps(model_keys,separators=(",",":")).encode()).hexdigest()
    # Derived file ephemeral in the current free standard CPU runner only.
    dest=folder/f"original-core4-train-{year}.csv.gz"
    train.to_csv(dest,index=False,compression="gzip")
    meta={
        "contract":"ORIGINAL_FROZEN_ABILITY_CORE4_TRAIN_YEAR_SCHEMA_V1",
        "year":year,"source_snapshot_rows":rows,
        "training_horse_rows":len(train),"training_races":int(train["_race_id"].nunique()),
        "multi_winner_races_preserved_per_original":multiple_winner_races,
        "exact_original_read_year_used":True,
        "exact_original_strip_forbidden_used":True,
        "exact_original_select_variant_core4_used":True,
        "training_labels_are_past_only":True,
        "no_2025_target_label_or_saved_prediction_read":True,
        "feature_names":model_keys,
        "feature_name_count":len(model_keys),
        "feature_name_sha256":schema_sha,
        "output_relative_name":dest.name
    }
    out=folder/f"original-core4-train-{year}-schema.json"
    out.write_text(json.dumps(meta,sort_keys=True,ensure_ascii=False)+"\n")
    print("ORIGINAL_ABILITY_TRAIN_YEAR_AUDITED",json.dumps({
         k:v for k,v in meta.items() if k!="feature_names"},sort_keys=True),flush=True)
    return meta


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--year",type=int,required=True)
    ap.add_argument("--input",required=True)
    ap.add_argument("--expected-rows",type=int,required=True)
    ap.add_argument("--expected-min",required=True)
    ap.add_argument("--expected-max",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    build(a.input,a.year,a.expected_rows,a.expected_min,a.expected_max,a.out)


if __name__=="__main__":
    main()
