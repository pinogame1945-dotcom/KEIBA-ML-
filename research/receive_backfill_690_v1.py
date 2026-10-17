#!/usr/bin/env python3
"""Receive existing BACKFILL 690-column race JSONs; never compute features or fit a model.

Data is copied only into caller-supplied local (ideally RUNNER_TEMP) workspace.
This is a structural pre-inference intake gate, NOT original numeric parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

ID_COLUMNS = {"_race_id", "_horse_id", "_race_date"}
CORE4_SHA = "eb3392b725560e77d46b28aef036b6188814125ed8f7a3bfde0c53522dbadf58"
ACCEPTED = {
    "KEIBA_20261010_24_ORIGINAL_690_UNSCORED_INPUT_V1",
    "KEIBA_BACKFILL_ORIGINAL_L1_690_INPUT_V1",
    "KEIBA_BACKFILL_20261017_ORIGINAL_690_UNSCORED_TRIAL_V1",
}
FORBIDDEN = {
    "_finish", "_is_win", "_is_top3", "target", "market_outcome",
    "final_win_odds", "final_popularity", "actual_start_time", "_score",
}
PROHIBITED_PREFIXES = ("P__", "distill_", "risk_")
VALID_ID = re.compile(r"^20[0-9]{10}$")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def schema_sha(columns):
    return hashlib.sha256(
        json.dumps(sorted(columns), separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def check_document(doc, filename):
    need(isinstance(doc, dict), "BACKFILL_DOCUMENT_NOT_OBJECT")
    need(doc.get("contract") in ACCEPTED, "UNKNOWN_BACKFILL_690_CONTRACT")
    race_id = str(doc.get("race_id", ""))
    need(bool(VALID_ID.fullmatch(race_id)), "INVALID_RACE_ID")
    need(filename == race_id + ".json", "RACE_ID_FILE_NAME_MISMATCH")
    date = str(doc.get("race_date", ""))
    need(bool(re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}", date)) and race_id[:4] == date[:4],
         "RACE_DATE_INVALID")
    need(doc.get("feature_count") == 690, "FEATURE_COUNT_NOT_690")
    # Historical and live documents expose different negative assertions.
    evidence = [doc.get(k) for k in (
        "used_target_labels", "used_result_payout_or_final_odds",
        "used_target_results_or_final_odds", "target_outcomes_used")]
    need(any(x is False for x in evidence), "MISSING_NO_OUTCOME_ATTESTATION")
    need(not any(x is True for x in evidence), "OUTCOME_LEAK_FLAG")
    need(doc.get("l1_scored") is not True, "SCORED_RESULT_NOT_FEATURE_INPUT")
    rows = doc.get("rows")
    need(isinstance(rows, list) and 1 <= len(rows) <= 24, "INVALID_STARTER_ROWS")
    if "starters" in doc:
        need(doc["starters"] == len(rows), "STARTER_COUNT_DRIFT")
    columns = None
    horse_ids = []
    for row in rows:
        need(isinstance(row, dict), "HORSE_ROW_NOT_OBJECT")
        need(row.get("_race_id") == race_id and row.get("_race_date") == date,
             "HORSE_RACE_ID_OR_DATE_DRIFT")
        horse_id = str(row.get("_horse_id", ""))
        need(horse_id.isdecimal() and len(horse_id) >= 8, "INVALID_HORSE_ID")
        horse_ids.append(horse_id)
        keys = set(row) - ID_COLUMNS
        need(len(keys) == 690, "HORSE_NOT_690_FEATURES")
        need(not keys.intersection(FORBIDDEN) and
             not any(key.startswith(PROHIBITED_PREFIXES) for key in keys),
             "HORSE_FORBIDDEN_TARGET_FEATURE")
        bias = {k for k in keys if k.startswith("daybias_") or
                k.startswith("horse_front_") or k.startswith("horse_inner_") or k.startswith("fit_")}
        speed = {k for k in keys if k.startswith("regime_speed_resid_")}
        need(len(bias) == 19 and len(speed) == 22, "BIAS19_SPEED22_SCHEMA_DRIFT")
        core = keys - bias - speed
        need(len(core) == 649 and schema_sha(core) == CORE4_SHA, "CORE4_SHA_DRIFT")
        if columns is None:
            columns = keys
        need(keys == columns, "INCONSISTENT_HORSE_FEATURE_COLUMNS")
        for key in keys:
            v = row[key]
            need(v is None or isinstance(v, (bool, str, int, float)), "FEATURE_NON_SCALAR")
            if isinstance(v, float):
                need(math.isfinite(v), "FEATURE_NON_FINITE")
    need(len(set(horse_ids)) == len(rows), "DUPLICATE_HORSE_ID")
    return race_id, date, len(rows), columns


def receive(input_dir, output_dir, expected_races, expected_horses):
    source = Path(input_dir)
    need(source.is_dir(), "BACKFILL_690_SOURCE_DIRECTORY_NOT_FOUND")
    files = sorted(p for p in source.glob("*.json") if p.name != "manifest.json")
    need(bool(files), "BACKFILL_690_FILES_NOT_PUBLISHED")
    need(all(p.stat().st_size <= 16 * 1024 * 1024 for p in files), "SOURCE_FILE_TOO_LARGE")
    if expected_races is not None:
        need(len(files) == expected_races, "RACE_COUNT_DRIFT")
    summaries, schema, dates = [], None, set()
    for p in files:
        doc = json.loads(p.read_text(encoding="utf-8"))
        rid, date, n, columns = check_document(doc, p.name)
        need(not any(d["race_id"] == rid for d in summaries), "DUPLICATE_RACE")
        if schema is None:
            schema = columns
        need(schema == columns, "ACROSS_RACES_690_SCHEMA_DRIFT")
        dates.add(date)
        summaries.append({"race_id": rid, "race_date": date, "starters": n})
    need(len(dates) == 1, "MIXED_RACE_DATES")
    count = sum(r["starters"] for r in summaries)
    if expected_horses is not None:
        need(count == expected_horses, "TOTAL_STARTERS_DRIFT")
    out = Path(output_dir)
    need(not out.resolve() == source.resolve(), "OUTPUT_MUST_DIFFER_FROM_SOURCE")
    out.mkdir(parents=True, exist_ok=True)
    for p in files:
        shutil.copyfile(p, out / p.name)
    report = {
        "contract": "KEIBA_ML_BACKFILL_690_RECEIPT_V1",
        "source_type": "BACKFILL_ORIGINAL_L1_690",
        "races": len(summaries),
        "horses": count,
        "features_per_horse": 690,
        "core4_sha256": CORE4_SHA,
        "input_schemas_verified": True,
        "numeric_original_parity_checked": False,
        "independent_prestart_timestamp_verified": False,
        "readiness": "STRUCTURE_ACCEPTED_RESEARCH_ONLY_NO_BETS",
        "model_executed": False,
        "receipts": summaries,
    }
    (out / "receipt.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Path to BACKFILL-produced 690 per-race JSON")
    p.add_argument("--output-dir", required=True, help="Runner-local received input folder")
    p.add_argument("--expected-races", type=int)
    p.add_argument("--expected-horses", type=int)
    a = p.parse_args()
    result = receive(a.input_dir, a.output_dir, a.expected_races, a.expected_horses)
    print("BACKFILL_690_RECEIVED " + json.dumps(
        {k: result[k] for k in ("races", "horses", "readiness")}, ensure_ascii=False
    ), flush=True)


if __name__ == "__main__":
    main()
