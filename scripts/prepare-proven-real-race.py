#!/usr/bin/env python3
"""Restore the EXACT real two-day BACKFILL smoke fixture found in prior CI.

Source commit, files and target race are proven by run 38731110047.
No Git clone, no filename probing, no arbitrary large downloads.
"""
from __future__ import annotations
import gzip
import importlib.util
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("github_fetch",HERE/"github-readonly-archive.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
REPO="pinogame1945-dotcom/KEIBA-BACKFILL"
REF="03950cfe132abeca15ef17dbc7c6885ceb1591b7"
DAYS=("2025-01-05","2025-01-06")
RACE_ID="202507010212"

def main():
    if len(sys.argv)!=2:raise SystemExit("usage: prepare-proven-real-race.py OUTPUT_ROOT")
    root=Path(sys.argv[1]).resolve()
    paths={}
    for date in DAYS:
        path=f"data/daily/{date}.jsonl.gz"
        local=root/path
        module.obtain(REPO,REF,path,local,1_000_000)
        paths[date]=local
    manifest_file=root/"manifest-origin.json"
    module.obtain(REPO,REF,"data/manifest.json",manifest_file,50_000_000)
    original=json.loads(manifest_file.read_text(encoding="utf-8"))
    days={}
    for date in DAYS:
        expected=f"data/daily/{date}.jsonl.gz"
        entry=original.get("days",{}).get(date)
        if entry is None or entry.get("status")!="SUCCESS" or entry.get("file")!=expected:
            raise ValueError("RACE_SOURCE_MANIFEST_DRIFT "+date)
        with gzip.open(paths[date],"rt",encoding="utf-8") as stream:
            rows=[json.loads(line) for line in stream if line.strip()]
        if date==DAYS[1]:
            target=[row for row in rows if str(row.get("race",{}).get("race_id"))==RACE_ID]
            if len(target)!=1 or len(target[0].get("entries",[]))!=16:
                raise ValueError("SOURCE_RACE_ID_OR_FIELD_DRIFT")
        days[date]=entry
    result={"days":days,"non_meeting_days":{},"schedule_exception_days":{}}
    out=root/"data"/"manifest.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False),encoding="utf-8")
    manifest_file.unlink(missing_ok=True)
    print("PROVEN_REAL_RACE_READY",json.dumps({
        "race_id":RACE_ID,"history_from":DAYS[0],
        "target_date":DAYS[1],"expected_entries":16,
        "source_commit":REF
    }),flush=True)

if __name__=="__main__":
    main()
