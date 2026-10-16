#!/usr/bin/env python3
"""Stream one real Kaggle year and emit only original compact strictly-prior state.

For the target year 2025: inspect date and race_id BEFORE accessing target
labels; omit the selected race and every date >= 2025-11-30 completely.
For 2019–2024: only prior-year labelled races are valid. Reuse exact frozen
read_compact_state_source() for outcomes, speed, etc. At no time access 2026.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from run_l1_objective_rebuild_v1 import open_text
from run_l1_state_transition_v1 import read_compact_state_source

TARGET="202505050812"
CUTOFF="2025-11-30"


def require(ok,msg):
    if not ok:raise ValueError(msg)


def export(source,year,manifest_rows,min_date,max_date,out):
    require(type(year) is int and 2019<=year<=2025,
            "STRICT_HISTORY_YEARS_ONLY_2019_TO_2025")
    source=Path(source);out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    require(manifest_rows>0,"INVALID_MANIFEST_ROWS")
    count=0;kept=0;first=None;last=None
    # Only transient filtered source is passed to the original compact
    # history parser. Never call that parser on a full 2025 source because
    # it would read the 202505050812 target outcome.
    with tempfile.TemporaryDirectory(dir=out) as temp:
        compact_input=Path(temp)/f"prior-{year}.jsonl"
        with compact_input.open("wt",encoding="utf-8") as fout:
            with open_text(source) as handle:
                for line in handle:
                    if not line.strip():continue
                    row=json.loads(line)
                    count+=1
                    date=str((row.get("features") or {}).get("race_date") or
                             row.get("race_date") or "")[:10]
                    require(len(date)==10 and date.startswith(f"{year}-"),
                            "STATE_HISTORY_SOURCE_YEAR_DATE_DRIFT")
                    first=min(first,date) if first is not None else date
                    last=max(last,date) if last is not None else date
                    if year==2025:
                        rid=str(row.get("race_id") or "")
                        # No selected race result accessed under ANY condition,
                        # including malformed date drift.
                        if rid==TARGET:continue
                        if date>=CUTOFF:continue
                    fout.write(line if line.endswith("\n") else line+"\n")
                    kept+=1
        require((count,first,last)==(manifest_rows,min_date,max_date),
                "STATE_HISTORY_KAGGLE_MANIFEST_ROWS_OR_DATES_DRIFT")
        require(kept>0,"NO_STRICTLY_PRIOR_SOURCE_DATA")
        prior=read_compact_state_source(compact_input,year)
    require((prior["_race_date"].astype(str)<CUTOFF).all(),
            "HISTORY_MUST_BE_STRICT_PRIOR_TO_TARGET_DATE")
    require(prior["_race_id"].astype(str).ne(TARGET).all(),
            "TARGET_RESULT_LEAKED_IN_ORIGINAL_STATE_READER")
    require(not prior.duplicated(["_race_id","_horse_id"]).any(),
            "DUPLICATE_STATE_HISTORY_HORSE_IN_RACE")
    data=out/f"compact-strict-prior-{year}.csv.gz"
    prior.to_csv(data,index=False,compression="gzip")
    digest=hashlib.sha256()
    with data.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            digest.update(block)
    attestation={
        "contract":"KEIBA_STRICT_PRIOR_ORIGINAL_STATE_SOURCE_YEAR_V1",
        "year":year,
        "target_race_id":TARGET,
        "target_date_exclusive":CUTOFF,
        "snapshot_manifest_rows":manifest_rows,
        "source_rows_scanned":count,
        "source_rows_before_cutoff":kept,
        "original_valid_finish_rows":int(len(prior)),
        "original_prior_races":int(prior["_race_id"].nunique()),
        "saved_prior_compact_sha256":digest.hexdigest(),
        "source_labels_all_prerace_relative_to_target":True,
        "selected_race_outcome_accessed":False,
        "date_on_or_after_cutoff_outcome_accessed":False,
        "2026_snapshot_accessed":False,
        "source_capture_timestamp_proven":False,
    }
    meta=out/f"compact-strict-prior-{year}-proof.json"
    meta.write_text(json.dumps(attestation,sort_keys=True)+"\n",encoding="utf-8")
    print("ORIGINAL_STATE_HISTORY_PRIOR_YEAR",json.dumps(attestation,sort_keys=True),
          flush=True)
    return data,attestation


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--year",type=int,required=True)
    p.add_argument("--expected-rows",type=int,required=True)
    p.add_argument("--expected-min",required=True)
    p.add_argument("--expected-max",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    export(a.input,a.year,a.expected_rows,a.expected_min,a.expected_max,a.out)

if __name__=="__main__":
    main()
