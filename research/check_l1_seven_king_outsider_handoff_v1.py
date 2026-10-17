#!/usr/bin/env python3
"""Read-only identity/schema gate for seven independent kings and 13 Outsider ballots.

This never constructs missing expert scores, votes, a council ranking or a
2026 evaluation. Saved 2021-2025 OOF predictions cannot be substituted for
a new 2026 target.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import re

EXPERTS=("core4","pedlegacy","condition","full","light","pedv1","condrc")
OUTSIDERS=(
    "outsider_bodychange","outsider_chimera","outsider_classmove",
    "outsider_coursetransition","outsider_daytrend","outsider_disttransition",
    "outsider_field","outsider_gatecourse","outsider_jockey",
    "outsider_opponentgap","outsider_raceshape","outsider_rotation",
    "outsider_weightchange",
)
BACKFILL_CONTRACT="KEIBA_BACKFILL_TO_ML_690_UNSCORED_V1"
EXPERT_CONTRACT="L1_TO_L2_OUTPUT_CONTRACT_V1"
OUT_CONTRACT="KEIBA_ML_SEVEN_KING_OUTSIDER_READINESS_V1"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def safe_file(value):
    path=Path(value)
    require(path.is_file(), "SUPPLIED_FILE_NOT_FOUND")
    require(path.stat().st_size <= 12_000_000, "SUPPLIED_FILE_TOO_LARGE")
    return path


def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8-sig") if str(path).endswith(".gz") \
        else open(path,"rt",encoding="utf-8-sig")


def load_backfill(path):
    file=safe_file(path)
    obj=json.loads(file.read_text(encoding="utf-8"))
    require(obj.get("contract")==BACKFILL_CONTRACT, "BACKFILL_690_CONTRACT_DRIFT")
    rid=str(obj.get("race_id") or "")
    date=obj.get("race_date")
    require(bool(re.fullmatch(r"2026\\d{8}",rid)) and
            isinstance(date,str) and date.startswith("2026-"),
            "MUST_USE_2026_TARGET_RACE")
    require(obj.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET",
            "UNVERIFIED_ASOF_STATUS_MUST_BE_RETAINED")
    require(obj.get("used_target_results_or_final_odds") is False,
            "FORBIDDEN_TARGET_RESULT_FLAG")
    require(obj.get("source_capture_asof_independently_verified") is False,
            "PRESTART_NOT_INDEPENDENTLY_CERTIFIED")
    require(obj.get("feature_count")==690,"FEATURE_COUNT_NOT_690")
    rows=obj.get("rows")
    require(isinstance(rows,list) and len(rows) >= 2 and len(rows) <= 24,
            "INVALID_STARTER_COUNT")
    require(obj.get("starters")==len(rows),"STARTER_COUNT_DRIFT")
    ids=[]
    for row in rows:
        require(row.get("_race_id")==rid and row.get("_race_date")==date,
                "RACE_DATE_PER_HORSE_DRIFT")
        hid=str(row.get("_horse_id") or "")
        require(hid.isdigit(),"INVALID_HORSE_ID")
        ids.append(hid)
        require(len(set(row)-{"_race_id","_race_date","_horse_id"})==690,
                "HORSE_690_COLUMN_COUNT_DRIFT")
        require("target" not in row and "market_outcome" not in row,
                "FORBIDDEN_TARGET_LABEL")
    require(len(ids)==len(set(ids)), "DUPLICATE_HORSE_ID")
    return rid,date,set(ids)


def load_expert(path, expected_race, horse_ids):
    f=safe_file(path)
    seen={}
    with open_text(f) as fh:
        for raw in fh:
            if not raw.strip():
                continue
            row=json.loads(raw)
            require(row.get("contract")==EXPERT_CONTRACT,
                    "NOT_FROZEN_ORIGINAL_EXPERT_SCORE_CONTRACT")
            rid=str(row.get("race_id") or "")
            require(rid==expected_race, "HISTORICAL_OR_OTHER_RACE_SCORE_FORBIDDEN")
            hid=str(row.get("horse_id") or "")
            require(hid in horse_ids and hid not in seen,
                    "EXPERT_HORSE_ID_MISMATCH")
            if row.get("race_date") is not None:
                require(str(row["race_date"]).startswith("2026-"),
                        "HISTORICAL_EXPERT_DATE_FORBIDDEN")
            rank=row.get("predicted_rank")
            require(type(rank) is int and rank>=1, "INVALID_EXPERT_RANK")
            seen[hid]=rank
    require(set(seen)==horse_ids and
            sorted(seen.values())==list(range(1,len(horse_ids)+1)),
            "EXPERT_FULL_FIELD_OR_RANK_COVERAGE_DRIFT")
    return len(seen)


def load_outsiders(path,expected_race,horse_ids):
    f=safe_file(path)
    seen=set()
    with open_text(f) as fh:
        for row in csv.DictReader(fh):
            require(row.get("race_id")==expected_race,
                    "HISTORICAL_OR_OTHER_RACE_BALLOT_FORBIDDEN")
            cand=row.get("candidate")
            require(cand in OUTSIDERS and cand not in seen,
                    "OUTSIDER_CANDIDATE_UNKNOWN_OR_DUPLICATED")
            seen.add(cand)
            voted=[]
            for i in (1,2,3):
                hid=str(row.get(f"top{i}_horse_id") or "").strip()
                if hid:
                    require(hid in horse_ids,"OUTSIDER_HORSE_ID_MISMATCH")
                    voted.append(hid)
            require(len(voted)==len(set(voted)),
                    "OUTSIDER_DUPLICATE_TOP3_HORSE")
    require(seen==set(OUTSIDERS), "NOT_ALL_13_OUTSIDER_BALLOTS")
    return len(seen)


def preflight(backfill,experts,outsiders):
    rid,date,horses=load_backfill(backfill)
    require(set(experts).issubset(set(EXPERTS)),"EXPERT_ALIAS_NOT_IN_ORIGINAL_SEVEN")
    found={}
    for alias,path in sorted(experts.items()):
        found[alias]=load_expert(path,rid,horses)
    ballots_count=load_outsiders(outsiders,rid,horses) if outsiders else 0
    absent=[e for e in EXPERTS if e not in experts]
    ready=not absent and ballots_count==len(OUTSIDERS)
    report={
        "contract":OUT_CONTRACT,
        "race_id":rid,"race_date":date,"horses":len(horses),
        "seven_king_names":list(EXPERTS),
        "verified_experts":found,
        "missing_experts":absent,
        "outsider_candidates_required":list(OUTSIDERS),
        "outsider_ballots_verified":ballots_count,
        "status":"READY_FOR_FROZEN_REPLAY_INPUT_REVIEW" if ready else
                 "BLOCKED_ORIGINAL_7KING_OUTSIDER_2026_INPUTS",
        "ready_for_original_council_input_review":ready,
        "original_council_executed":False,
        "models_trained_or_scored":False,
        "2026_results_read_or_evaluated":False,
        "original_690_parity_verified":False,
        "prestart_asof_independently_verified":False,
        "approved_for_betting":False,
    }
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--backfill-690",required=True)
    p.add_argument("--expert",action="append",default=[],help="alias=2026-score.jsonl[.gz]")
    p.add_argument("--outsider-ballots",help="2026 race 13-candidate ballots.csv[.gz]")
    p.add_argument("--output")
    args=p.parse_args()
    experts={}
    for raw in args.expert:
        alias,sep,path=raw.partition("=")
        require(bool(sep) and alias not in experts and bool(path),
                "EXPERT_ARG_ALIAS_DUPLICATED_OR_INVALID")
        experts[alias]=path
    report=preflight(args.backfill_690,experts,args.outsider_ballots)
    if args.output:
        dst=Path(args.output)
        dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\\n",
                       encoding="utf-8")
    print("L1_7KING_OUTSIDER_READINESS "+json.dumps(
        report,ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
