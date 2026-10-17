#!/usr/bin/env python3
"""Metadata-only gate for prior-dated 2026 history; never open race results.

This intentionally sees ONLY Git's tree filenames/blob IDs, not blob content.
It does not authorize using 2026 prior outcome fields, changing any frozen
rules, or treating the 2026 target outcome as accessible for calibration.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

PINNED_BACKFILL_SHA="9fe9f1742aa074a84c03a779e1ee949c04f78c0d"
EXPECTED_2026_TARGET_DAY="2026-10-17"
FILE=re.compile(r"^data/daily/(2026-\d{2}-\d{2})\.jsonl\.gz$")
TREE=re.compile(r"^([0-7]{6}) (blob|tree) ([0-9a-f]{40})\t(.*)$")

def guard(ok, why):
    if not ok:
        raise ValueError(why)

def select_prior(tree_lines, target_day):
    guard(target_day==EXPECTED_2026_TARGET_DAY, "TARGET_DATE_LOCK_DRIFT")
    names=[]
    seen=set()
    for line in tree_lines:
        match=TREE.fullmatch(line)
        guard(match is not None, "UNEXPECTED_GIT_TREE_LINE")
        mode, kind, blob_id, name=match.groups()
        m=FILE.fullmatch(name)
        if m is None:
            continue
        date=m.group(1)
        guard(kind=="blob" and mode=="100644" and name not in seen,
              "UNEXPECTED_DAILY_FILE_METADATA")
        seen.add(name)
        if date>=target_day:
            continue
        names.append({"path":name,"date":date,"git_blob_oid":blob_id})
    guard(names and all(p["date"]<target_day for p in names),
          "NO_STRICTLY_PRIOR_DATED_2026_FILE_NAMES")
    return sorted(names,key=lambda x:x["date"])

def run(repo, out, target_day):
    guard(not out.exists(),"REFUSE_OVERWRITE_PRIOR_HISTORY_METADATA")
    sha=subprocess.check_output(
      ["git","-C",str(repo),"rev-parse","FETCH_HEAD"],text=True).strip()
    guard(sha==PINNED_BACKFILL_SHA,"BACKFILL_PINNED_GIT_SHA_MISMATCH")
    # git ls-tree walks COMMIT TREE NAMES ONLY. Never checkout or cat-file.
    tree=subprocess.check_output(
      ["git","-C",str(repo),"ls-tree","-r","FETCH_HEAD","data/daily"],
      text=True)
    names=select_prior(tree.splitlines(),target_day)
    counts=dict(sorted(Counter(x["date"][:7] for x in names).items()))
    report={
      "contract":"KEIBA_2026_10_17_PRIOR_DATED_DAILY_METADATA_ONLY_V1",
      "source_backfill_git_sha":PINNED_BACKFILL_SHA,
      "target_date":target_day,
      "files_strictly_before_target_date":len(names),
      "months":counts,
      "source_race_blobs_opened":0,
      "2026_outcome_fields_read":False,
      "2026_target_results_read":False,
      "history_model_inputs_recovered":False,
      "2026_prior_results_use_authorized":False,
      "asof_capture_independently_proven":False,
      "ready_for_betting":False,
      "source_file_names_and_git_blob_ids":names
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("2026_PRIOR_HISTORY_TREE_ONLY_NO_RESULT_READ "+
          json.dumps({k:v for k,v in report.items() if k!="source_file_names_and_git_blob_ids"},
          sort_keys=True),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--target-date",default=EXPECTED_2026_TARGET_DAY)
    args=parser.parse_args()
    run(args.repo,args.out,args.target_date)
