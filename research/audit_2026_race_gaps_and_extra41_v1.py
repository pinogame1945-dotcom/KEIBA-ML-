#!/usr/bin/env python3
"""Diagnostic only: 2026 per-race CORE4 missing-column causes and frozen 41 extras.

No 2026 labels used for model fit/selection/performance evaluation. The original
unchanged source feature and same-day-bias formulas are called. Historical race-day
FINAL phase is not proof of live as-of captures. No change to frozen contracts.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import json
from pathlib import Path
import tempfile
import pandas as pd

from audit_2026_generated_core4_v1 import selected_columns, digest, CORE4_SHA, EXPECTED
from run_l1_sameday_track_bias_v1 import BiasState, bias_cols, speed_cols, _build_same_day_bias_year
from compute_blind_target_seq_regime_v1 import horse_features


def readlines(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def family(key):
    for prefix in ("auto_", "opponent_", "actor_", "timepace_", "previous_",
                   "recent_", "prior_", "backfill_", "network_"):
        if key.startswith(prefix):
            return prefix
    return "BASE"


def audit(snapshot, max_date="2026-10-12"):
    race_keys = defaultdict(set)
    race_horses = Counter()
    race_dates = {}
    year_keys = set()
    all_2026_ids = set()
    last_day_records = []
    row_count = 0
    for row in readlines(snapshot):
        rdate = str((row.get("features") or {}).get("race_date", ""))
        if not rdate.startswith("2026-"):
            raise ValueError("NOT_2026_SNAPSHOT")
        if rdate > max_date:
            raise ValueError("2026_FUTURE_DATE_IN_SNAPSHOT")
        rid = str(row.get("race_id", ""))
        hid = str(row.get("horse_id", ""))
        if (rid, hid) in all_2026_ids:
            raise ValueError("DUPLICATE_2026_RACE_HORSE")
        all_2026_ids.add((rid,hid))
        keys = set(selected_columns(row.get("features") or {}))
        race_keys[rid].update(keys)
        year_keys.update(keys)
        race_dates[rid] = rdate
        race_horses[rid] += 1
        row_count += 1
        if rdate == max_date:
            last_day_records.append(row)
    if len(year_keys) != EXPECTED or digest(sorted(year_keys)) != CORE4_SHA:
        raise ValueError("2026_REFERENCE_FROZEN_649_MISMATCH")

    missing_races = {}
    key_freq = Counter()
    family_freq = Counter()
    month_freq = Counter()
    day_freq = Counter()
    race_summaries = []
    for rid, have in race_keys.items():
        miss = sorted(year_keys - have)
        if not miss:
            continue
        missing_races[rid] = miss
        key_freq.update(miss)
        family_freq.update(set(map(family, miss)))
        month_freq[race_dates[rid][:7]] += 1
        day_freq[race_dates[rid]] += 1
        race_summaries.append({
            "race_id": rid,
            "date": race_dates[rid],
            "horses": race_horses[rid],
            "missing_count": len(miss),
            "missing": miss[:30],
        })
    race_summaries.sort(key=lambda d: (-d["missing_count"], d["race_id"]))
    gap_counts = Counter(map(len, missing_races.values()))
    current = [rid for rid in missing_races if race_dates[rid] >= "2026-10-10"]
    report = {
        "rows": row_count,
        "races": len(race_keys),
        "canonical_core4_649": digest(sorted(year_keys)),
        "missing_race_count": len(missing_races),
        "missing_race_pct": round(100 * len(missing_races) / len(race_keys), 3),
        "distinct_missing_names": len(key_freq),
        "top_missing_names_by_race": key_freq.most_common(35),
        "missing_key_family_affected_race_count": family_freq.most_common(),
        "missing_key_count_histogram": sorted(gap_counts.items()),
        "missing_by_month": sorted(month_freq.items()),
        "missing_by_day_last_12": sorted(day_freq.items())[-12:],
        "missing_race_ids_last_2026_10_10_to_12": sorted(current),
        "worst_races_first_20": race_summaries[:20],
        "sample_last_day": max_date,
        "last_day_horses": len(last_day_records),
    }
    return report, last_day_records


def audit_extra_41(last_day, target_date):
    # Exact names are derived from frozen BiasState and regime helper,
    # not a fabricated list of similar-looking columns.
    bias = BiasState()
    bias_keys = list(bias.snapshot())
    horse_and_fit = (
        "horse_front_score", "horse_inner_score",
        "fit_front_mean", "fit_front_ewma", "fit_inner_mean", "fit_inner_ewma"
    )
    actual_bias = bias_cols(pd.DataFrame(columns=[*bias_keys, *horse_and_fit]), with_fit=True)

    # This is the unchanged helper used for original regime; empty historical
    # panel produces NaN values by design but gives the exact 22 column names.
    empty = pd.DataFrame(columns=["_race_id", "_race_date"])
    sample_features = horse_features(empty, target_date)
    actual_speed = speed_cols(list(sample_features))
    if len(actual_bias) != 19 or len(actual_speed) != 22:
        raise ValueError("ORIGINAL_19_PLUS_22_EXTRA_FEATURES_DRIFT")
    if len(set(actual_bias + actual_speed)) != 41:
        raise ValueError("ORIGINAL_41_EXTRAS_NOT_UNIQUE")

    day_bias_races = None
    day_bias_rows = 0
    day_bias_usable = 0
    bias_source_matches = False
    # Numerically exercise original same-day bias on one fully completed date,
    # not on a live race, and do not use outputs for model evaluation.
    if last_day:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "completed-day.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as out:
                for row in last_day:
                    out.write(json.dumps(row, ensure_ascii=False, separators=(",",":")) + "\n")
            y, panel, _ = _build_same_day_bias_year(2026, source)
            if y != 2026:
                raise ValueError("SAME_DAY_ORIGINAL_YEAR_DRIFT")
            cols = bias_cols(panel, with_fit=True)
            if cols != actual_bias:
                raise ValueError("ORIGINAL_BIAS_19_COLUMNS_MISMATCH")
            day_bias_rows = len(panel)
            day_bias_races = int(panel["_race_id"].nunique())
            day_bias_usable = int(panel["daybias_prior_races"].gt(0).sum())
            bias_source_matches = len(last_day) == len(panel)
    return {
        "original_day_bias_plus_fit_columns": actual_bias,
        "original_day_bias_column_count": len(actual_bias),
        "original_speed_regime_columns": actual_speed,
        "original_speed_regime_column_count": len(actual_speed),
        "total_extra_original_columns": len(actual_bias) + len(actual_speed),
        "total_original_core4_plus_extras": 649 + len(actual_bias) + len(actual_speed),
        "completed_day_original_bias_races": day_bias_races,
        "completed_day_original_bias_rows": day_bias_rows,
        "completed_day_horses_with_prior_races_on_same_day": day_bias_usable,
        "sample_bias_population_exactly_matched": bias_source_matches,
        "regime_values_parity_to_full_2019_2026_state": False,
        "all_690_values_equal_frozen_reference": False,
        "historical_2026_outcomes_used_for_model_selection_or_fit": False,
        "historical_final_phase_proves_prebet_timestamps": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-2026", required=True)
    p.add_argument("--cutoff-last-day", default="2026-10-12")
    a = p.parse_args()
    info, last = audit(a.snapshot_2026, max_date=a.cutoff_last_day)
    extras = audit_extra_41(last, a.cutoff_last_day)
    record = {
        "contract": "KEIBA_2026_RACE_GAP_EXTRA41_DIAGNOSTIC_V1",
        "core4": info,
        "extra41": extras,
        "full_original_690_value_parity_claim": False,
        "any_frozen_logic_modified": False,
    }
    print("PHASE2_2026_RACE_GAP_EXTRA41 " + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
    if info["missing_race_count"] == 0:
        print("2026_ALL_RACES_COMPLETE_FEATURE_UNION")
    else:
        print("2026_INCOMPLETE_RACE_PANELS_DIAGNOSED_COUNT",info["missing_race_count"])


if __name__ == "__main__":
    main()
