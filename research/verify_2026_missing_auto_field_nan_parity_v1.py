#!/usr/bin/env python3
"""Verify actual 2026 missing-field one-race inference matrix parity.

READ ONLY. Exercises the UNCHANGED frozen original prepare() to compare
(1) historical year-batch records with Pandas union-of-keys/NaN behavior and
(2) a raw single-race DataFrame reindexed to the SAME frozen training schema.

2026 results/target/market_outcome are never loaded, accessed or used.
No 2026 training, score evaluation or bet recommendation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import pandas as pd

from audit_2026_generated_core4_v1 import CORE4_SHA, EXPECTED, digest, selected_columns
from run_l1_objective_rebuild_v1 import prepare, META

IDENTITY = ("_race_id", "_horse_id", "_race_date")
VERIFY_CONTROL_MAX = 40


def feature_records(path, year):
    """Load only unscored historical feature dictionaries (never target labels)."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            f = item["features"]
            rid, hid, d = str(item["race_id"]), str(item["horse_id"]), str(f.get("race_date") or "")
            if not d.startswith(str(year) + "-") or rid[:4] != str(year) or not hid:
                raise ValueError("RACE_YEAR_OR_HORSE_ID_DRIFT")
            names = selected_columns(f)
            entry = {key: f.get(key) for key in names}
            entry["race_month"] = int(d[5:7])
            entry["race_day_of_year"] = pd.Timestamp(d).dayofyear
            entry.update({"_race_id": rid, "_horse_id": hid, "_race_date": d})
            yield entry


def verify(reference_2025, snapshot_2026):
    # This is ONLY a categorical-code normalization probe, not the original
    # model training set. Same original prepare() is used for both paths.
    reference = list(feature_records(reference_2025, 2025))
    train = pd.DataFrame.from_records(reference)
    training_names = sorted(set(train) - set(META))
    if len(training_names) != EXPECTED or digest(training_names) != CORE4_SHA:
        raise ValueError("FROZEN_2025_TRAIN_REFERENCE_649_DRIFT")
    expected = set(training_names)
    by_race = defaultdict(list)
    key_union = set()
    total_horses = 0
    for entry in feature_records(snapshot_2026, 2026):
        by_race[entry["_race_id"]].append(entry)
        key_union.update(set(entry) - set(IDENTITY))
        total_horses += 1
    if key_union != expected or digest(sorted(key_union)) != CORE4_SHA:
        raise ValueError("FROZEN_2026_YEAR_FEATURE_NAMES_DRIFT")

    missing = {}
    healthy = []
    for rid, entries in sorted(by_race.items()):
        current = set().union(*(set(e) - set(IDENTITY) for e in entries))
        miss = sorted(expected - current)
        extra = sorted(current - expected)
        if extra:
            raise ValueError("EXTRA_RACE_FEATURE_KEYS:" + rid)
        if miss:
            missing[rid] = miss
        else:
            healthy.append(rid)
    if len(missing) != 207:
        raise ValueError(f"EXPECTED_207_PREVIOUSLY_VERIFIED_MISSING_RACES_GOT_{len(missing)}")
    if not all(len(keys) == 54 and all(k.startswith("auto_field_") for k in keys)
               for keys in missing.values()):
        raise ValueError("MISSING_FIELDS_DIFFER_FROM_FROZEN_54_AUTO_FIELD_DIAGNOSIS")

    control = healthy[:VERIFY_CONTROL_MAX // 2] + healthy[-VERIFY_CONTROL_MAX // 2:]
    checked = sorted([*missing, *set(control)])
    checked_records = []
    race_first_records = []
    matrix_na_added = 0
    for rid in checked:
        entries = by_race[rid]
        # Historical year-mode: from_records unions all available named
        # columns across the complete cohort; unavailable fields are NaN.
        checked_records.extend(entries)
        # One-race live-mode candidate; DO NOT change any calculation
        # or fill value. Only shape alignment, as already done in prepare().
        single = pd.DataFrame.from_records(entries)
        miss = missing.get(rid, [])
        if miss and any(x in single.columns for x in miss):
            raise ValueError("MISSING_COLUMN_ACCIDENTALLY_PRESENT")
        matrix_na_added += len(entries) * len(miss)
        candidate = single.reindex(columns=[*IDENTITY, *training_names])
        if miss and not candidate[miss].isna().all().all():
            raise ValueError("REINDEX_DID_NOT_MATERIALIZE_ONLY_NAN")
        race_first_records.append(candidate)
    historical = pd.DataFrame.from_records(checked_records)
    race_batched = pd.concat(race_first_records, ignore_index=True, copy=False)
    historical = historical.sort_values(["_race_id", "_horse_id"]).reset_index(drop=True)
    race_batched = race_batched.sort_values(["_race_id", "_horse_id"]).reset_index(drop=True)

    if historical[list(IDENTITY)].astype(str).to_numpy().tolist() != race_batched[list(IDENTITY)].astype(str).to_numpy().tolist():
        raise ValueError("HORSE_ORDER_OR_ID_DRIFT")
    if len(historical) != len(race_batched):
        raise ValueError("HORSE_ROW_COUNT_DRIFT")

    # The original prepare() enforces the precise 649 model features, LightGBM
    # float32 conversions and fixed training-side categorical levels.
    _, baseline, cat1 = prepare(train, historical)
    _, proposed, cat2 = prepare(train, race_batched)
    if cat1 != cat2:
        raise ValueError("MODEL_CATEGORICAL_COLUMN_DRIFT")
    if list(baseline.columns) != training_names or list(proposed.columns) != training_names:
        raise ValueError("MODEL_649_ORDER_DRIFT")
    # Exact equality compares float32 bit-equivalent numeric values and
    # identical categorical levels/codes; NaNs compare equal in Pandas.
    try:
        pd.testing.assert_frame_equal(
            baseline, proposed, check_dtype=True, check_exact=True,
            check_categorical=True, check_names=True,
        )
    except AssertionError as exc:
        raise ValueError("ORIGINAL_PREPARE_MATRIX_VALUE_NOT_EQUAL") from exc
    if baseline.shape != (len(checked_records), 649):
        raise ValueError("MODEL_MATRIX_SIZE_INVALID")
    return {
        "contract": "KEIBA_2026_MISSING_AUTO_FIELD_NAN_PREPARE_PARITY_V1",
        "status": "EXACT_PREPARE_MATRIX_PARITY_SUCCESS",
        "original_function": "run_l1_objective_rebuild_v1.prepare",
        "reference_categorical_levels_from": "2025-12-28_REF_PROBE_NOT_FROZEN_2023_2024_FIT_LEVELS",
        "2026_total_races_scanned": len(by_race),
        "2026_total_horses_scanned": total_horses,
        "2026_incomplete_races_verified": len(missing),
        "2026_complete_race_controls_checked": len(set(control)),
        "2026_actual_races_matrix_compared": len(checked),
        "2026_actual_horses_matrix_compared": len(checked_records),
        "per_incomplete_race_missing_features": 54,
        "only_missing_feature_family": "auto_field_",
        "nan_cells_materialized_by_649_schema": matrix_na_added,
        "matrix_features": baseline.shape[1],
        "matrix_numeric_and_category_values_equal": True,
        "train_model_fit_executed": False,
        "2026_outcomes_or_betting_used": False,
        "frozen_649_schema_or_feature_formula_changed": False,
        "extra_41_regime_and_bias_value_parity_tested": False,
        "production_code_changed": False,
        "prediction_or_bet_equality_tested": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-2025", required=True)
    ap.add_argument("--snapshot-2026", required=True)
    a = ap.parse_args()
    data = verify(a.reference_2025, a.snapshot_2026)
    print("PHASE2_2026_NAN_INPUT_PARITY " + json.dumps(data, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
