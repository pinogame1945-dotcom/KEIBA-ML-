#!/usr/bin/env python3
"""2026-10-10, both JRA venues: read-only 24-race unscored inference handoff.

Frozen 649 CORE4 feature names from verified original annual builder.
Target outcomes, postrace prices, saved scores and 2026 evaluation are not read.
The only materialization is absent CORE4 columns -> null (the unchanged
original Pandas prepare() maps them to NaN). All files runner-temporary only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path

from audit_2026_generated_core4_v1 import selected_columns, digest, CORE4_SHA, EXPECTED

TARGET_DAY = "2026-10-10"
EXPECTED_IDS = frozenset(
    f"2026{venue}0403{race_no:02d}"
    for venue in ("05", "08")
    for race_no in range(1, 13)
)
IDENTITY = {"_race_id", "_horse_id", "_race_date"}
WITHDRAWN = {"SCRATCHED", "EXCLUDED"}


def read_jsonlines_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def verify(check, reason):
    if not check:
        raise ValueError(reason)


def run(backfill_day, snapshot, output):
    year_union = set()
    chosen = defaultdict(list)
    year_rows = 0
    for row in read_jsonlines_gz(snapshot):
        feat = row.get("features") or {}
        dt = str(feat.get("race_date") or "")
        verify(dt.startswith("2026-"), "2026_SNAPSHOT_DATE_DRIFT")
        names = selected_columns(feat)
        year_union.update(names)
        year_rows += 1
        if dt != TARGET_DAY:
            continue
        rid = str(row.get("race_id") or "")
        hid = str(row.get("horse_id") or "")
        verify(rid in EXPECTED_IDS, "OTHER_DATE_OR_UNEXPECTED_RACE_ID_IN_TARGET_SNAPSHOT")
        verify(hid.isdigit() and 8 <= len(hid) <= 16, "HORSE_ID_INVALID")
        # Snapshot target labels/results/odds are NEVER fetched or inspected.
        selected = {k: feat.get(k) for k in names}
        selected["race_month"] = 10
        selected["race_day_of_year"] = date.fromisoformat(TARGET_DAY).timetuple().tm_yday
        chosen[rid].append({"_race_id": rid, "_horse_id": hid,
                            "_race_date": TARGET_DAY, **selected})
    verify(len(year_union) == EXPECTED and digest(sorted(year_union)) == CORE4_SHA,
           "FULL_YEAR_ORIGINAL_649_SHA_DRIFT")

    daily = {}
    for item in read_jsonlines_gz(backfill_day):
        race = item.get("race") or {}
        rid = str(race.get("race_id") or "")
        verify(rid in EXPECTED_IDS, "DAILY_WRONG_RACE_ID:" + rid)
        verify(str(race.get("actual_date")) == TARGET_DAY,
               "DAILY_RACE_DATE_MISMATCH:" + rid)
        verify(rid not in daily, "DUPLICATE_DAILY_RACE_ID:" + rid)
        entries = item.get("entries")
        verify(isinstance(entries, list), "DAILY_ENTRIES_NOT_LIST")
        horses = {
            str(e.get("horse_id"))
            for e in entries
            if str(e.get("entry_status") or "").upper() not in WITHDRAWN
        }
        verify(len(horses) > 0, "NO_DAILY_ELIGIBLE_STARTERS:" + rid)
        daily[rid] = {"horse_ids": horses, "race_no": race.get("race_no"),
                      "venue_code": race.get("venue_code")}
        # Deliberately do not read item["results"], item["odds"], payouts.
    verify(set(daily) == EXPECTED_IDS,
           "BACKFILL_EXPECTED_24_RACE_SET_MISMATCH:" +
           repr(sorted(EXPECTED_IDS - set(daily))) + ":" +
           repr(sorted(set(daily) - EXPECTED_IDS)))
    verify(set(chosen) == EXPECTED_IDS,
           "SNAPSHOT_EXPECTED_24_RACE_SET_MISMATCH:" +
           repr(sorted(EXPECTED_IDS - set(chosen))))
    verify(len(chosen) == 24, "NOT_24_TARGET_RACES")
    p = Path(output)
    p.mkdir(parents=True, exist_ok=True)
    reports = []
    count = Counter()
    for rid in sorted(EXPECTED_IDS):
        raw = chosen[rid]
        ids = [r["_horse_id"] for r in raw]
        verify(len(set(ids)) == len(raw), "DUPLICATE_TARGET_HORSE:" + rid)
        verify(set(ids) == daily[rid]["horse_ids"],
               "SOURCE_ENTRIES_VS_FROZEN_SNAPSHOT_STARTER_DRIFT:" + rid +
               ":only_daily=" + repr(sorted(daily[rid]["horse_ids"] - set(ids))) +
               ":only_snapshot=" + repr(sorted(set(ids) - daily[rid]["horse_ids"])))
        have = set().union(*(set(r) - IDENTITY for r in raw))
        missing = sorted(year_union - have)
        extras = sorted(have - year_union)
        verify(not extras, "CORE4_EXTRA_FEATURES:" + rid)
        verify(not missing or (
            len(missing) == 54 and
            sum(k.startswith("auto_field_") for k in missing) == 45 and
            sum(k.startswith("auto_pair_") for k in missing) == 9
        ), "UNEXPECTED_CORE4_MISSING_FIELDS:" + rid)
        normalized = []
        for row in sorted(raw, key=lambda x: (int(x.get("horse_number") or 0), x["_horse_id"])):
            fixed = {k: row.get(k) for k in sorted(year_union)}
            normalized.append({k: row[k] for k in ("_race_id","_horse_id","_race_date")} | fixed)
        verify(all(len(set(x) - IDENTITY) == 649 for x in normalized),
               "CORE4_POST_NORMALIZATION_WRONG_COLUMN_COUNT:" + rid)
        for col in missing:
            verify(all(x[col] is None for x in normalized),
                   "MISSING_AUTO_FIELD_NOT_NULL:" + rid + ":" + col)
        doc = {
            "contract": "KEIBA_20261010_P3_UNSCORED_CORE4_649_V1",
            "race_id": rid, "race_date": TARGET_DAY,
            "source": "PINNED_2026_BACKFILL_HISTORICAL_FINAL_PROJECTION",
            "prediction_phase": "FINAL_PROJECTED_ASOF_UNVERIFIED",
            "used_target_labels": False,
            "used_result_payout_or_final_odds": False,
            "used_2026_historical_oos_predictions": False,
            "source_capture_asof_independently_verified": False,
            "model_690_enrichment_completed": False,
            "l1_scored": False, "l175_scored": False,
            "l21_frozen_tickets_ready": False, "l3_v3_p3_proven": False,
            "core4_649_reference_sha256": CORE4_SHA,
            "core4_missing_columns_before_original_nan_alignment": missing,
            "rows": normalized,
        }
        (p / (rid + ".json")).write_text(
            json.dumps(doc, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        reports.append({
            "race_id": rid,
            "course": "東京" if rid[4:6] == "05" else "京都",
            "race_no": int(rid[-2:]),
            "starters": len(raw),
            "raw_core4_column_union": len(have),
            "missing_core4_features_normalized_to_nan": len(missing),
            "all_649_input_names_ready": True,
            "jump_race_requires_model_eligibility_review": rid == "202608040304",
        })
        count["starters"] += len(raw)
        count["races_with_54_missing_auto_columns"] += int(bool(missing))
    body = {
        "contract": "KEIBA_20261010_24_JRA_RACES_P3_INPUT_AUDIT_V1",
        "race_day": TARGET_DAY,
        "source_year_2026_total_rows_scanned": year_rows,
        "total_races": len(reports),
        "total_starters": count["starters"],
        "races_with_54_nan_auto_fields": count["races_with_54_missing_auto_columns"],
        "races": reports,
        "historical_snapshot_asof_source_timing_verified": False,
        "original_l1_l175_l21_l3_executed": False,
        "core4_649_name_and_population_contract_verified": True,
        "target_results_or_prices_used": False,
        "backfill_data_changed": False,
        "frozen_models_policies_changed": False,
    }
    (p / "manifest.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("PHASE3_20261010_24_JRA_INPUTS " +
          json.dumps(body, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill-day", required=True)
    ap.add_argument("--snapshot-2026", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    run(args.backfill_day, args.snapshot_2026, args.output)
