#!/usr/bin/env python3
"""Read-only parity probe for original frozen CORE4 feature names on 2026 history.

Use *existing* 2022..2026 source snapshot builder; do NOT fit any model or
evaluate outcomes for sealed 2026. No change to CORE4, 7 Kings, Outsider,
frozen L2.1/L3, source data, storage or captured as-of assertions.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path
from run_l1_objective_rebuild_v1 import keep_variant_column, FORBIDDEN_KEYS, META
from run_l1_ceiling_audit_v1 import FULL_FORBIDDEN

CORE4_SHA = "eb3392b725560e77d46b28aef036b6188814125ed8f7a3bfde0c53522dbadf58"
EXPECTED = 649
FIELDS = ("opponent_", "auto_", "actor_", "timepace_", "previous_", "recent_", "prior_")
ALLOWED_2026_FIRST = "2026-01-01"
ALLOWED_2026_LAST = "2026-10-12"

def iso(value):
    try:
        return date.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError):
        return None

def selected_columns(features):
    if not isinstance(features, dict):
        raise ValueError("NOT_A_FEATURE_OBJECT")
    columns = (set(features) - {"race_date"} - FORBIDDEN_KEYS - FULL_FORBIDDEN)
    columns.update(("race_month", "race_day_of_year"))
    return sorted(x for x in columns if x not in META and keep_variant_column(x, "core4"))

def digest(keys):
    return hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()

def scan(path, year):
    if year not in (2025, 2026):
        raise ValueError("ONLY_REFERENCE_2025_AND_SEALED_2026")
    encountered = Counter()
    populated = Counter()
    by_date = Counter()
    row_count = 0
    observed_first = None
    observed_last = None
    year_set = set()
    race_sets = {}
    sparse_row_count = 0
    problems = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get("race_id") or "")
            hid = str(row.get("horse_id") or "")
            features = row.get("features") or {}
            dt = iso(features.get("race_date"))
            if not dt or not dt.startswith(str(year) + "-"):
                problems.append("SOURCE_YEAR_DATE_DRIFT")
                continue
            if year == 2026 and not (ALLOWED_2026_FIRST <= dt <= ALLOWED_2026_LAST):
                problems.append("2026_SOURCE_OUTSIDE_VERIFIED_PAST_CUTOFF")
            if len(rid) != 12 or rid[:4] != str(year) or not hid:
                problems.append("RACE_HORSE_ID_INVALID")
            if row.get("leakage_policy") != "STRICT_PRIOR_DATE_ONLY":
                problems.append("MODEL_LEAKAGE_POLICY_DRIFT")
            if row.get("prediction_phase") != "FINAL":
                problems.append("PREDICTION_PHASE_DRIFT")
            keys = selected_columns(features)
            # The original Pandas DataFrame.from_records() uses union-of-keys
            # within its input cohort and fills absent horse-level cells with NaN.
            # Sparse per-horse objects are normal and must NOT be called schema drift.
            year_set.update(keys)
            race_sets.setdefault(rid, set()).update(keys)
            if len(keys) < EXPECTED:
                sparse_row_count += 1
            row_count += 1
            by_date[dt] += 1
            observed_first = min(observed_first, dt) if observed_first else dt
            observed_last = max(observed_last, dt) if observed_last else dt
            for key in keys:
                encountered[key] += 1
                val = features.get(key)
                if key == "race_month":
                    val = int(dt[5:7])
                elif key == "race_day_of_year":
                    val = date.fromisoformat(dt).timetuple().tm_yday
                if val is not None and val != "":
                    populated[key] += 1
    if not row_count:
        problems.append("NO_VERIFIED_SNAPSHOT_ROWS")
    if len(year_set) != EXPECTED or digest(sorted(year_set)) != CORE4_SHA:
        problems.append("FROZEN_649_YEAR_UNION_FEATURE_NAME_MISMATCH")
    # Original blind single-race inference constructs a NEW DataFrame from
    # that race alone, so record which race cohorts lack one or more columns.
    deficient_races = sorted(
        rid for rid, cols in race_sets.items()
        if len(cols) != EXPECTED or digest(sorted(cols)) != CORE4_SHA
    )
    family = {}
    for prefix in FIELDS:
        keys = [k for k in encountered if k.startswith(prefix)]
        if keys:
            n = sum(populated[k] for k in keys)
            total = row_count * len(keys)
            family[prefix] = {"features": len(keys), "populated_cells": n,
                              "total_cells": total, "coverage_pct": round(n * 100 / total, 2)}
    sparse = sorted(({"name": k, "present": populated[k], "rows": row_count}
                     for k in encountered),
                    key=lambda x: (x["present"], x["name"]))[:15]
    return {
        "year": year,
        "rows": row_count,
        "dates": len(by_date),
        "first_date": observed_first,
        "last_date": observed_last,
        "core4_feature_count": len(year_set),
        "core4_name_sha256": digest(sorted(year_set)) if year_set else None,
        "core4_name_equivalent_to_original": bool(year_set) and not problems,
        "per_horse_rows_with_omitted_feature_keys": sparse_row_count,
        "total_race_panels": len(race_sets),
        "race_panels_missing_original_649_name_union": len(deficient_races),
        "race_panels_with_complete_649_name_union": len(race_sets) - len(deficient_races),
        "first_10_incomplete_race_ids": deficient_races[:10],
        "feature_coverage_by_family": family,
        "lowest_coverage_features": sparse,
        "error_count": len(problems),
        "errors_first_10": problems[:10],
        "used_2026_outcomes_for_training_or_evaluation": False,
        "used_2026_outcomes_to_score_models": False,
        "independent_source_asof_timestamps_verified": False,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-2025", required=True)
    p.add_argument("--snapshot-2026", required=True)
    a = p.parse_args()
    data = [scan(a.snapshot_2025, 2025), scan(a.snapshot_2026, 2026)]
    report = {
        "contract": "KEIBA_2026_CORE4_GENERATION_PHASE2_AUDIT_V1",
        "status": "CORE4_NAME_SCHEMA_VERIFIED_ONLY" if all(x["core4_name_equivalent_to_original"] for x in data)
                  else "CORE4_NAME_SCHEMA_MISMATCH",
        "reference_core4_names_sha256": CORE4_SHA,
        "years": data,
        "core4_feature_values_equal_to_2025_canonical_snapshot": False,
        "state_bias_regime_690_parity_verified": False,
        "full_live_ai_ready": False,
    }
    print("PHASE2_2026_CORE4_AUDIT " + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    if report["status"] != "CORE4_NAME_SCHEMA_VERIFIED_ONLY":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
