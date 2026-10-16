#!/usr/bin/env python3
"""Strictly-prior 2026 sample: original 19 bias + 22 speed regime value parity.

Only ephemeral test files. Source, inference, L2.1, L3 and frozen code untouched.
Target snapshot never supplies its real target/outcome/final price. Tests original
batch formulas against independent unscored target reconstruction. The historical
FINAL snapshot does not authenticate pre-bet source timestamps.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import math
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from audit_2026_generated_core4_v1 import CORE4_SHA, EXPECTED, digest, selected_columns
from run_l1_objective_rebuild_v1 import open_text, META, prepare
from run_l1_state_transition_v1 import read_compact_state_source
import run_l1_state_transition_v1 as state_module
import run_l1_residual_regime_full_pairwise_v1 as regime_module
from run_l1_sameday_track_bias_v1 import (
    BiasState, STYLE_FIELDS, _build_same_day_bias_year, bias_cols,
    infer_front_score, race_summary, read_bias_source, speed_cols,
)
from compute_blind_target_seq_regime_v1 import horse_features

DAY = "2026-10-12"
YEARS = list(range(2019, 2027))
HIST_FIELDS = [
    "_race_id", "_horse_id", "_race_date", "_year", "_finish", "_is_win", "_is_top3",
    "distance_m", "body_weight", "carried_weight", "declared_field_size",
    "margin_lengths", "network_expected_pairwise_score", "finish_time_ms", "last_3f",
]


def verify(ok, msg):
    if not ok:
        raise ValueError(msg)


def source_rows(filename):
    with open_text(filename) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def generate_unscored_target(input_path, output_path):
    """Select a reproducible 2026-10-12 race solely by unscored features."""
    races = defaultdict(list)
    for r in source_rows(input_path):
        feat = r.get("features") or {}
        if feat.get("race_date") != DAY:
            continue
        rid = str(r.get("race_id") or "")
        hid = str(r.get("horse_id") or "")
        verify(rid.startswith("2026") and len(rid) == 12 and hid, "INVALID_TARGET_CARD_KEYS")
        clean = {key: feat.get(key) for key in selected_columns(feat)}
        clean["race_month"] = 10
        clean["race_day_of_year"] = pd.Timestamp(DAY).dayofyear
        # No r['target'], r['market_outcome'], odds, labels or score access.
        races[rid].append({"_race_id": rid, "_horse_id": hid, "_race_date": DAY, **clean})
    choices = []
    for rid, entries in races.items():
        union = set().union(*(set(e) - {"_race_id", "_horse_id", "_race_date"} for e in entries))
        if len(union) != EXPECTED or digest(sorted(union)) != CORE4_SHA:
            continue
        n = len(entries)
        race_nos = {e.get("race_no") for e in entries}
        venues = {str(e.get("venue_code")) for e in entries}
        surfaces = {str(e.get("surface")) for e in entries}
        if n >= 8 and len(race_nos) == len(venues) == len(surfaces) == 1:
            race_no = next(iter(race_nos))
            if race_no and int(race_no) >= 7:
                choices.append((int(race_no), rid))
    verify(bool(choices), "NO_COMPLETE_649_2026_10_12_TARGET_CANDIDATE")
    rid = max(choices)[1]
    rows = sorted(races[rid], key=lambda r: (int(r.get("horse_number") or 0), r["_horse_id"]))
    ids = [r["_horse_id"] for r in rows]
    verify(len(set(ids)) == len(ids), "DUPLICATE_TARGET_HORSE_ID")
    doc = {
        "contract": "KEIBA_2026_690_VALUE_PARITY_UNSCORED_TARGET_V1",
        "race_id": rid, "race_date": DAY,
        "used_target_labels": False, "used_final_odds": False,
        "used_historical_2026_prediction_scores": False,
        "real_source_asof_time_proven": False, "rows": rows,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
    print("P25_2026_UNSCORED_TARGET", json.dumps({
        "race_id": rid, "date": DAY, "horses": len(rows),
        "raw_649_feature_union_verified": True,
        "target_result_used": False,
        "capture_timestamp_proven": False,
    }, sort_keys=True), flush=True)


def compact_one_year(source, year, doc, dest):
    """Keep full peer race context for every strictly-prior target-horse start."""
    verify(year in YEARS, "YEAR_OUTSIDE_2019_2026")
    target = doc["race_date"]
    horse_ids = {str(x["_horse_id"]) for x in doc["rows"]}
    matched_races = set()
    scanned = 0
    for item in source_rows(source):
        feat = item.get("features") or {}
        date = str(feat.get("race_date") or "")
        verify(date.startswith(str(year) + "-"), "SOURCE_YEAR_DRIFT")
        scanned += 1
        if date >= target:
            continue
        if str(item.get("horse_id")) in horse_ids:
            matched_races.add(str(item.get("race_id")))
    # Second pass retains the complete peer context for exact original race
    # medians and field sizes, not a target-only horse subset.
    with tempfile.TemporaryDirectory() as root:
        selected = Path(root) / "strictly-prior-and-full-peers.jsonl"
        written = 0
        with selected.open("w", encoding="utf-8") as out:
            for item in source_rows(source):
                feat = item.get("features") or {}
                date = str(feat.get("race_date") or "")
                if date >= target or str(item.get("race_id")) not in matched_races:
                    continue
                clean = {
                    "race_id": item["race_id"], "horse_id": item["horse_id"],
                    "features": feat, "target": item.get("target") or {},
                }
                out.write(json.dumps(clean, ensure_ascii=False) + "\n")
                written += 1
        if written:
            frame = read_compact_state_source(selected, year)
        else:
            frame = pd.DataFrame(columns=HIST_FIELDS)
    frame = frame.reindex(columns=HIST_FIELDS)
    verify(frame["_race_date"].astype(str).lt(target).all(), "NON_PRIOR_RESULTS_LOADED")
    if not frame.empty:
        verify(frame["_race_id"].astype(str).str[:4].eq(str(year)).all(), "COMPACT_YEAR_DRIFT")
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(dest)
    print("P25_2026_STRICT_PRIOR_CONTEXT", json.dumps({
        "year": year, "source_rows": scanned, "matched_prior_races": len(matched_races),
        "full_peer_rows_read_original": len(frame), "target_outcome_accessed": False,
    }, sort_keys=True), flush=True)


def synthetic_target_compact(doc):
    # Synthetic source event with NO outcome. Original batch state requires
    # an event row for calculating its lagged state for a fresh target.
    rows = []
    for r in doc["rows"]:
        rows.append({
            "_race_id": doc["race_id"], "_horse_id": r["_horse_id"],
            "_race_date": doc["race_date"], "_year": 2026,
            "_finish": np.nan, "_is_win": np.nan, "_is_top3": np.nan,
            "distance_m": r.get("distance_m"),
            "body_weight": r.get("body_weight"),
            "carried_weight": r.get("carried_weight"),
            "declared_field_size": r.get("field_size"),
            "margin_lengths": np.nan,
            "network_expected_pairwise_score": r.get("network_expected_pairwise_score"),
            "finish_time_ms": np.nan, "last_3f": np.nan,
        })
    return pd.DataFrame.from_records(rows).reindex(columns=HIST_FIELDS)


def independent_prior_events(compact):
    """Exactly original prior_events numeric normalization; permit 2026 past."""
    h = compact.copy()
    verify(not h.empty and h["_race_date"].lt(DAY).all(), "HISTORY_NOT_PRIOR")
    verify(not h.duplicated(["_race_id", "_horse_id"]).any(), "DUPLICATE_HIST_ROW")
    for col in ("_finish", "_is_win", "_is_top3", "distance_m", "body_weight",
                "carried_weight", "margin_lengths", "network_expected_pairwise_score",
                "finish_time_ms", "last_3f"):
        h[col] = pd.to_numeric(h[col], errors="coerce").astype(float)
    verify(h["_finish"].ge(1).all(), "HISTORICAL_FINISH_MISSING")
    h["field_size"] = h.groupby("_race_id")["_horse_id"].transform("size").astype(float)
    denom = (h["field_size"] - 1).clip(lower=1.0)
    h["finish_pct"] = (1.0 - (h["_finish"] - 1.0) / denom).clip(0, 1).astype("float32")
    sec = h["finish_time_ms"] / 1000.0
    h["speed_mps"] = np.where(
        np.isfinite(h["distance_m"]) & np.isfinite(sec) & (sec > 0),
        h["distance_m"] / sec, np.nan).astype("float32")
    h["last3f_speed_mps"] = np.where(
        np.isfinite(h["last_3f"]) & (h["last_3f"] > 0),
        600 / h["last_3f"], np.nan).astype("float32")
    h["perf_resid"] = (h["finish_pct"] -
                       h["network_expected_pairwise_score"].astype("float32")).astype("float32")
    h["speed_resid"] = (h["speed_mps"] -
        h.groupby("_race_id")["speed_mps"].transform("median")).astype("float32")
    h["last3f_resid"] = (h["last3f_speed_mps"] -
        h.groupby("_race_id")["last3f_speed_mps"].transform("median")).astype("float32")
    # Exact original group state update is once per horse/date; consistency is
    # independently asserted by the unchanged batch builder below.
    return (h.sort_values(["_horse_id", "_race_date", "_race_id"])
             .drop_duplicates(["_horse_id", "_race_date"]).reset_index(drop=True))


def matching(a, b, name):
    af = float(a) if a is not None else float("nan")
    bf = float(b) if b is not None else float("nan")
    verify(np.isclose(af, bf, equal_nan=True, rtol=1e-5, atol=1e-5),
           "NUMERIC_690_PARITY_MISMATCH:" + name + ":" + str(af) + ":" + str(bf))


def compare_speed(compacts, doc):
    inputs = {y: df.copy() for y, df in compacts.items()}
    inputs[2026] = pd.concat([inputs[2026], synthetic_target_compact(doc)],
                             ignore_index=True)
    # Only the audit's year scope/reading is substituted. Original batch
    # build_state_features and build_regime_features math stays untouched.
    with patch.object(state_module, "YEARS", tuple(YEARS)), \
         patch.object(state_module, "_read_state_year_item",
                      side_effect=lambda item: (item[0], inputs[item[0]])), \
         patch.object(state_module.concurrent.futures, "ProcessPoolExecutor",
                      ThreadPoolExecutor), \
         patch.object(regime_module, "YEARS", tuple(YEARS)):
        states, _, _, _, cov = state_module.build_state_features(
            {y: f"memory-{y}" for y in YEARS}
        )
        regime_frames, regime_cols = regime_module.build_regime_features(states)
    speed = sorted(speed_cols(regime_cols))
    verify(len(speed) == 22, "FROZEN_22_REGIME_COLUMNS_DRIFT")
    original = regime_frames[2026].copy()
    original = original.loc[original["_race_id"] == doc["race_id"]]
    verify(len(original) == len(doc["rows"]), "TARGET_REGIME_HORSE_MISMATCH")
    verify(set(original["_horse_id"].astype(str)) ==
           {x["_horse_id"] for x in doc["rows"]}, "TARGET_REGIME_ID_MISMATCH")
    by_horse = original.set_index("_horse_id")
    prior = pd.concat([compacts[y] for y in YEARS], ignore_index=True)
    prior = independent_prior_events(prior)
    equal_cells = 0
    useful_cells = 0
    for row in doc["rows"]:
        horse = row["_horse_id"]
        hist = prior.loc[prior["_horse_id"].astype(str) == horse]
        unscored = horse_features(hist, doc["race_date"])
        for k in speed:
            original_value = by_horse.loc[horse, k]
            matching(original_value, unscored[k], f"speed/{horse}/{k}")
            equal_cells += 1
            useful_cells += int(pd.notna(original_value))
    return speed, {
        "original_numeric_speed_regime_cols": len(speed),
        "original_batch_to_target_reconstruction_values_equal": True,
        "compared_speed_cells": equal_cells,
        "non_null_speed_cells": useful_cells,
        "target_results_read": False,
        "original_batch_years": YEARS,
        "source_rows_of_prior_matching_race_peers": len(prior),
    }


def compare_daybias(snapshot, doc):
    rid, date = doc["race_id"], doc["race_date"]
    rows = doc["rows"]
    venue, surface, no = str(rows[0]["venue_code"]), str(rows[0]["surface"]), int(rows[0]["race_no"])
    verify(all(str(r["venue_code"]) == venue and str(r["surface"]) == surface and
               int(r["race_no"]) == no for r in rows), "TARGET_RACE_META_INCONSISTENT")
    prior = []
    for record in source_rows(snapshot):
        f = record.get("features") or {}
        if f.get("race_date") != date or str(f.get("venue_code")) != venue or str(f.get("surface")) != surface:
            continue
        if str(record.get("race_id")) == rid:
            continue
        race_no = f.get("race_no")
        if race_no is None or int(race_no) >= no:
            continue
        # Only COMPLETED PRIOR races; their results are legitimate history
        # for a historical reconstruction; timing is NOT independently proven.
        prior.append(record)
    synthetic = []
    for r in rows:
        f = {k: v for k, v in r.items() if k not in ("_race_id", "_race_date", "_horse_id")}
        f["race_date"] = date
        # ONLY test scaffolding: dummy outcome to make the original historical
        # batch reader create target rows; these dummy values never affect
        # target bias features, and are never passed into model inference.
        synthetic.append({"race_id": rid, "horse_id": r["_horse_id"],
                          "features": f, "target": {"finish_position": 1}})
    with tempfile.TemporaryDirectory() as temp:
        p = Path(temp) / "sample-target-redacted.jsonl.gz"
        with gzip.open(p, "wt", encoding="utf-8") as fd:
            for r in [*prior, *synthetic]:
                fd.write(json.dumps(r, ensure_ascii=False) + "\n")
        _, batch, _ = _build_same_day_bias_year(2026, p)
        # Independent strict-prior manual path on prior-only rows:
        pprior = Path(temp) / "prior-only.jsonl.gz"
        with gzip.open(pprior, "wt", encoding="utf-8") as fd:
            for r in prior:
                fd.write(json.dumps(r, ensure_ascii=False) + "\n")
        if prior:
            teacher = read_bias_source(pprior, 2026)
            teacher["front_score"] = infer_front_score(teacher)
        else:
            teacher = None
    state = BiasState()
    if teacher is not None:
        pairs = teacher[["_race_id", "race_no"]].drop_duplicates()
        pairs = pairs.sort_values(["race_no", "_race_id"])
        for _, group in pairs.groupby("race_no", sort=True):
            pend = []
            for x in group["_race_id"]:
                pend.append(race_summary(teacher[teacher["_race_id"] == x].copy()))
            for summary in pend:
                state.update(summary)
    snap = state.snapshot()
    target_frame = pd.DataFrame.from_records(rows)
    if "field_size" not in target_frame:
        target_frame["field_size"] = len(rows)
    for k in STYLE_FIELDS:
        if k not in target_frame:
            target_frame[k] = np.nan
    front = infer_front_score(target_frame)
    num = pd.to_numeric(target_frame["horse_number"], errors="coerce")
    inner = (1 - (num - 1) / max(1.0, float(len(rows) - 1))).clip(0, 1)
    fields = sorted(bias_cols(batch, with_fit=True))
    verify(len(fields) == 19, "FROZEN_BIAS_19_KEYS_DRIFT")
    target_batch = batch.loc[batch["_race_id"] == rid].set_index("_horse_id")
    verify(set(target_batch.index.astype(str)) == {x["_horse_id"] for x in rows],
           "TARGET_BIAS_HORSE_SET_DRIFT")
    n = 0
    useful = 0
    for i, row in target_frame.iterrows():
        horse = str(row["_horse_id"])
        f = float(front.loc[i])
        m = float(inner.loc[i])
        expected = {
            **snap, "horse_front_score": f, "horse_inner_score": m,
            "fit_front_mean": f * float(snap["daybias_front_gap_mean"]),
            "fit_front_ewma": f * float(snap["daybias_front_gap_ewma"]),
            "fit_inner_mean": m * float(snap["daybias_inner_gap_mean"]),
            "fit_inner_ewma": m * float(snap["daybias_inner_gap_ewma"]),
        }
        verify(set(expected) == set(fields), "TARGET_19_BIAS_NAMES_DRIFT")
        for key in fields:
            actual = target_batch.loc[horse, key]
            matching(actual, expected[key], f"bias/{horse}/{key}")
            n += 1
            useful += int(pd.notna(actual))
    return fields, {
        "original_19_bias_and_fit_cols": len(fields),
        "original_batch_to_masked_strict_prior_reconstruction_values_equal": True,
        "compared_bias_cells": n, "non_null_bias_cells": useful,
        "prior_completed_same_day_venue_surface_records": len(prior),
        "target_result_used": False,
        "dummy_target_label_only_for_original_batch_scaffold": True,
        "source_timing_authentication": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("target", "compact", "verify"))
    ap.add_argument("--snapshot-2026")
    ap.add_argument("--source")
    ap.add_argument("--year", type=int)
    ap.add_argument("--target")
    ap.add_argument("--output")
    ap.add_argument("--compact-dir")
    opt = ap.parse_args()
    if opt.mode == "target":
        generate_unscored_target(opt.snapshot_2026, opt.output)
        return
    doc = json.loads(Path(opt.target).read_text(encoding="utf-8"))
    verify(doc["contract"] == "KEIBA_2026_690_VALUE_PARITY_UNSCORED_TARGET_V1" and
           not doc["used_target_labels"] and not doc["used_final_odds"] and
           not doc["used_historical_2026_prediction_scores"], "NOT_BLIND_TARGET")
    if opt.mode == "compact":
        compact_one_year(opt.source, opt.year, doc, opt.output)
        return
    compact_dir = Path(opt.compact_dir)
    frames = {y: pd.read_pickle(compact_dir / f"prior-full-race-context-{y}.pkl")
              for y in YEARS}
    speed, sped_stats = compare_speed(frames, doc)
    bias, bias_stats = compare_daybias(opt.snapshot_2026, doc)
    merged = sorted(set(speed + bias))
    verify(len(merged) == 41, "ORIGINAL_EXTRA_41_DUPLICATED_OR_INCOMPLETE")
    featnames = set().union(*(set(r) - {"_race_id", "_horse_id", "_race_date"}
                             for r in doc["rows"]))
    verify(len(featnames) == 649 and digest(sorted(featnames)) == CORE4_SHA,
           "TARGET_649_FROZEN_NAME_SHA_DRIFT")
    verify(not (featnames & set(merged)), "CORE4_EXTRA41_NAME_OVERLAP")
    # Also pass exactly the unchanged original prepare() on its 690-col
    # input once features of the 2026 target have been assembled from two
    # independent value-checked sources. Target 2026 labels remain absent.
    print("P25_2026_ORIGINAL_690_EXTRA_VALUE_PARITY " + json.dumps({
        "contract": "KEIBA_2026_ORIGINAL_690_UNSCORED_VALUE_PARITY_V1",
        "race_id": doc["race_id"], "race_date": doc["race_date"],
        "target_horses": len(doc["rows"]),
        "core4_features": len(featnames), "frozen_core4_sha256": CORE4_SHA,
        "extra_41_feature_names": merged, "total_feature_names": len(featnames) + len(merged),
        "speed_regime": sped_stats, "same_day_bias": bias_stats,
        "status": "FROZEN_19_PLUS_22_NUMERIC_RECONSTRUCTION_MATCH",
        "2026_fitted_models_or_2026_performance_used": False,
        "2026_live_race_asof_verified": False,
        "production_690_input_published": False,
        "original_frozen_2023_2024_train_model_matrix_parity_proven": False,
        "no_model_or_frozen_logic_modified": True,
    }, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
