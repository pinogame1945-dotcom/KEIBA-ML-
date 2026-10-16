#!/usr/bin/env python3
"""Read-only Phase 1 audit: BACKFILL history vs race-day AI input readiness.

History is post-race source data, NEVER a pre-race feature panel. Only strictly
prior completed race days can enter a chosen prediction cutoff. The independent
panel mode checks the UNSCORED AI handoff, without inferring missing features or
claiming timestamp provenance. No training, odds or L2.1/L3 code is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path
import re

ID = re.compile(r"^[0-9]{12}$")
HORSE = re.compile(r"^[0-9]{8,16}$")
FORBIDDEN = re.compile(
    r"^(?:_finish$|_is_win$|_is_top3$|target(?:_|$)|finish(?:_|$)|"
    r"final(?:_|$)|market_final_|result(?:_|$)|payout(?:_|$)|"
    r"postrace_|after_race_|outcome_|P__|distill_|risk(?:_|$)|"
    r"actual_start_time$|market_rank$|win_odds_at_cutoff$)",
    re.I,
)
RACE_FIELDS = ("venue_code", "discipline", "surface", "distance_m",
               "direction", "weather", "track_condition")
ENTRY_FIELDS = ("horse_id", "horse_number", "sex", "age", "carried_weight",
                "jockey_id", "trainer_id", "body_weight")
RESULT_FIELDS = ("official_finish_position", "finish_time_ms", "last_3f")
MAX_PROBLEMS = 40
CANONICAL_CORE4_649_SHA256 = "eb3392b725560e77d46b28aef036b6188814125ed8f7a3bfde0c53522dbadf58"


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for i, line in enumerate(stream, 1):
            if line.strip():
                yield i, json.loads(line)


def good_date(value):
    try:
        parsed = date.fromisoformat(value)
        return parsed.isoformat() == value
    except (ValueError, TypeError):
        return False


def populated(value):
    return value is not None and value != ""


def summarize_history(paths, cutoff):
    if not good_date(cutoff):
        raise ValueError("PREDICTION_CUTOFF_INVALID")
    errors = []
    counts = Counter()
    coverage = Counter()
    race_ids = set()
    dates = set()
    for path in paths:
        if not Path(path).is_file():
            raise ValueError("HISTORY_FILE_NOT_FOUND:" + str(path))
        seen_file = 0
        for line, item in read_rows(path):
            seen_file += 1
            loc = f"{Path(path).name}:{line}"
            race = item.get("race") or {}
            rid = str(race.get("race_id") or "")
            race_date = str(race.get("actual_date") or "")
            if not ID.fullmatch(rid):
                errors.append(loc + ":RACE_ID_INVALID")
            if not good_date(race_date):
                errors.append(loc + ":ACTUAL_DATE_INVALID")
            elif race_date >= cutoff:
                errors.append(loc + ":HISTORY_NOT_STRICTLY_PRIOR_TO_CUTOFF")
            if rid in race_ids:
                errors.append(loc + ":DUPLICATE_RACE_ID")
            race_ids.add(rid)
            dates.add(race_date)
            counts["races"] += 1
            for key in RACE_FIELDS:
                coverage["race." + key + ".den"] += 1
                if populated(race.get(key)):
                    coverage["race." + key + ".num"] += 1
            entries = item.get("entries")
            results = item.get("results")
            if not isinstance(entries, list) or not entries:
                errors.append(loc + ":ENTRIES_MISSING")
                continue
            if not isinstance(results, list) or not results:
                errors.append(loc + ":RESULTS_MISSING_FOR_HISTORY")
                continue
            en_ids, en_nos, res_ids = [], [], []
            for entry in entries:
                counts["entries"] += 1
                hid = str(entry.get("horse_id") or "")
                if not HORSE.fullmatch(hid):
                    errors.append(loc + ":HORSE_ID_INVALID")
                en_ids.append(hid)
                en_nos.append(entry.get("horse_number"))
                for key in ENTRY_FIELDS:
                    coverage["entry." + key + ".den"] += 1
                    if populated(entry.get(key)):
                        coverage["entry." + key + ".num"] += 1
            if len(set(en_ids)) != len(en_ids):
                errors.append(loc + ":DUPLICATE_ENTRY_HORSE_ID")
            if None in en_nos or len(set(en_nos)) != len(en_nos):
                errors.append(loc + ":HORSE_NUMBER_MISSING_OR_DUPLICATED")
            for result in results:
                counts["results"] += 1
                res_ids.append(str(result.get("horse_id") or ""))
                for key in RESULT_FIELDS:
                    coverage["result." + key + ".den"] += 1
                    if populated(result.get(key)):
                        coverage["result." + key + ".num"] += 1
            if len(set(res_ids)) != len(res_ids) or set(en_ids) != set(res_ids):
                errors.append(loc + ":ENTRIES_RESULTS_HORSE_ID_MISMATCH")
            laps = item.get("laps")
            counts["races_with_laps"] += int(isinstance(laps, list) and bool(laps))
            counts["finished_horses"] += sum(
                r.get("result_status") == "FINISHED" for r in results
            )
        if not seen_file:
            errors.append(Path(path).name + ":EMPTY_SOURCE_FILE")
    return {
        "mode": "STRICT_PRIOR_BACKFILL_HISTORY",
        "cutoff_exclusive": cutoff,
        "source_date_range": [min(dates), max(dates)] if dates else [],
        "counts": dict(sorted(counts.items())),
        "coverage": {
            key[:-4]: {"present": coverage[key], "total": value,
                        "pct": round(coverage[key] / value * 100, 2) if value else 0}
            for key, value in sorted(coverage.items()) if key.endswith(".num")
            for value in [coverage[key[:-4] + ".den"]]
        },
        "errors": errors[:MAX_PROBLEMS],
        "total_errors": len(errors),
        "raw_history_valid": not errors and counts["races"] > 0,
        "core4_649_feature_schema_verified": False,
        "derived_state_bias_regime_verified": False,
        "asof_capture_timestamp_verified": False,
        "ai_prediction_ready": False,
        "status": ("RAW_HISTORY_CHECK_PASSED_FEATURE_BUILD_REQUIRED"
                   if not errors and counts["races"] else "RAW_HISTORY_CHECK_FAILED"),
    }


def summarize_panel(path, target_id, target_date, expected_keys_path=None):
    if not ID.fullmatch(target_id) or not good_date(target_date):
        raise ValueError("PANEL_RACE_ID_OR_DATE_INVALID")
    if target_id[:4] != target_date[:4]:
        raise ValueError("PANEL_RACE_ID_YEAR_DRIFT")
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("rows"), list):
        raise ValueError("UNSCORED_TARGET_DOCUMENT_REQUIRED")
    errors = []
    if doc.get("race_id") != target_id or doc.get("race_date") != target_date:
        errors.append("PANEL_RACE_ID_DATE_MISMATCH")
    if doc.get("used_target_labels") is not False:
        errors.append("TARGET_LABELS_NOT_EXPLICITLY_EXCLUDED")
    if doc.get("used_results_payouts_final_odds") is not False:
        errors.append("POSTRACE_RESULTS_NOT_EXPLICITLY_EXCLUDED")
    if doc.get("used_historical_oos_target_predictions") is not False:
        errors.append("SAVED_TARGET_PREDICTIONS_NOT_EXPLICITLY_EXCLUDED")
    if doc.get("contract") != "KEIBA_UNSCORED_RACE_FEATURE_EXTRACT_V1":
        errors.append("PANEL_CONTRACT_MISMATCH")
    rows = doc["rows"]
    ids = []
    source_keys = set()
    for i, row in enumerate(rows):
        rid, hid, dt = str(row.get("_race_id") or ""), str(row.get("_horse_id") or ""), row.get("_race_date")
        if rid != target_id or dt != target_date or not HORSE.fullmatch(hid):
            errors.append(f"ROW_{i}_IDENTITY_OR_DATE_MISMATCH")
        ids.append(hid)
        fields = set(row) - {"_race_id", "_horse_id", "_race_date"}
        source_keys.update(fields)
        leaks = sorted(key for key in fields if FORBIDDEN.search(key))
        if leaks:
            errors.append(f"ROW_{i}_POSTRACE_OR_SCORE_LEAK:" + ",".join(leaks[:5]))
    if not rows or len(set(ids)) != len(ids):
        errors.append("EMPTY_OR_DUPLICATED_PANEL_HORSES")
    if expected_keys_path:
        expected = json.loads(Path(expected_keys_path).read_text(encoding="utf-8"))
        if not isinstance(expected, list) or len(expected) != 649 or len(set(expected)) != 649:
            raise ValueError("CANONICAL_CORE4_649_FEATURE_LIST_REQUIRED")
        digest = hashlib.sha256(json.dumps(sorted(expected), separators=(",", ":")).encode()).hexdigest()
        if digest != CANONICAL_CORE4_649_SHA256:
            raise ValueError("FROZEN_CORE4_FEATURE_NAMES_SHA256_MISMATCH")
        expected_keys = set(expected)
        # Frozen inference uses pandas.DataFrame.from_records(rows), which
        # unions horse-level keys and fills absent cells with NaN. Only a
        # missing or additional key in the race-level UNION is schema drift.
        sparse_horses = sum(
            set(row) - {"_race_id", "_horse_id", "_race_date"} != expected_keys
            for row in rows
        )
        missing = sorted(expected_keys - source_keys)
        extra = sorted(source_keys - expected_keys)
        if missing or extra:
            errors.append("CORE4_649_FEATURE_SCHEMA_DRIFT")
    else:
        missing = []
        extra = []
    return {
        "mode": "UNSCORED_PREDICTION_FEATURE_PANEL",
        "race_id": target_id, "target_date": target_date,
        "horse_count": len(rows),
        "feature_names_seen": len(source_keys),
        "expected_core4_names_supplied": bool(expected_keys_path),
        "missing_feature_names": missing[:30],
        "extra_feature_names": extra[:30],
        "rows_with_sparsely_omitted_keys_filled_as_nan": sparse_horses if expected_keys_path else None,
        "errors": errors[:MAX_PROBLEMS], "total_errors": len(errors),
        "source_timing_verified": doc.get("actual_source_capture_time_independently_verified") is True,
        "raw_core4_schema_check_passed": bool(expected_keys_path) and not errors,
        "derived_state_bias_regime_verified": False,
        "ai_prediction_ready": False,
        "status": ("PANEL_SCHEMA_CHECK_PASSED_DERIVED_FEATURES_AND_ASOF_PENDING"
                   if not errors and expected_keys_path else
                   "PANEL_AUDIT_INCOMPLETE" if not errors else "PANEL_CHECK_FAILED"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-day", action="append", default=[])
    parser.add_argument("--prediction-cutoff")
    parser.add_argument("--panel")
    parser.add_argument("--target-race-id")
    parser.add_argument("--target-date")
    parser.add_argument("--expected-core4-keys")
    parser.add_argument("--output", default="-")
    a = parser.parse_args()
    if bool(a.history_day) == bool(a.panel):
        parser.error("choose history-day or panel")
    if a.history_day:
        if not a.prediction_cutoff:
            parser.error("--prediction-cutoff required for history")
        result = summarize_history(a.history_day, a.prediction_cutoff)
    else:
        if not a.target_date or not a.target_race_id:
            parser.error("--target-date and --target-race-id required for panel")
        result = summarize_panel(a.panel, a.target_race_id, a.target_date, a.expected_core4_keys)
    body = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if a.output == "-":
        print(body, end="")
    else:
        Path(a.output).write_text(body, encoding="utf-8")
    if result["total_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
