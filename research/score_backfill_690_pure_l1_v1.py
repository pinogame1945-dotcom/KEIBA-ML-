#!/usr/bin/env python3
"""Score published 2026 BACKFILL 690 inputs using the frozen ORIGINAL L1 LambdaRank.

The 2024/25 training fold, 690-feature preparation, model hyperparameters,
training seed, and rank/score normalization are those of the existing
phase3_20261010_original_l1_24_cold_v1.py; no 2026 labels or results enter fit.
Structural BACKFILL input, NOT numeric parity or 2026 performance evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import phase3_20261010_original_l1_24_cold_v1 as original
from run_l1_objective_rebuild_v1 import META, prepare, model_common
from run_l1_ceiling_audit_v1 import FULL_FORBIDDEN
from run_l1_sameday_track_bias_v1 import bias_cols
from run_l1_market_correction_distill_v1 import add_ai_context
from receive_backfill_690_v1 import check_document, ID_COLUMNS

FIT_YEARS = original.FIT_YEARS
TARGET_YEAR = original.TARGET_YEAR
TRAIN_MODEL_SEED = original.TRAIN_MODEL_SEED
N_FEATURES = original.N_FEATURES
EXPECTED = {"2026-10-10": (24, 315), "2026-10-17": (36, 471)}
CONTRACT = "KEIBA_BACKFILL_690_REAL_ORIGINAL_L1_ABILITY_V1"


def need(check, reason):
    if not check:
        raise ValueError(reason)


def load_delivered(root):
    root = Path(root)
    rows = []
    docs = []
    for date, (expected_races, expected_horses) in EXPECTED.items():
        source = root / date
        need(source.is_dir(), "DELIVERED_DATE_NOT_RECEIVED:" + date)
        names = sorted(source.glob("2026*.json"))
        need(len(names) == expected_races, "DELIVERED_RACE_POPULATION_DRIFT:" + date)
        active = 0
        for path in names:
            doc = json.loads(path.read_text(encoding="utf8"))
            rid, actual_date, n, _ = check_document(doc, path.name)
            need(actual_date == date, "WRONG_DATE_IN_DELIVERED_RACE")
            need(doc.get("readiness") == "ASOF_UNVERIFIED_DO_NOT_BET",
                 "UNKNOWN_DELIVERED_READINESS")
            need(doc.get("source_capture_asof_independently_verified") is False,
                 "UNVERIFIED_SOURCE_MUST_NOT_ASSERT_VERIFIED")
            need(doc.get("original_ml_full_numerical_parity_verified") is False,
                 "UNVERIFIED_ORIGINAL_PARITY_MUST_NOT_ASSERT_VERIFIED")
            need(doc.get("used_target_results_or_final_odds") is False,
                 "2026_TARGET_RESULTS_FORBIDDEN")
            need(doc.get("race_id") == rid, "RACE_ID_DRIFT")
            rows.extend(doc["rows"])
            docs.append({"race_id": rid, "race_date": date, "starters": n})
            active += n
        need(active == expected_horses, "DELIVERED_HORSE_POPULATION_DRIFT:" + date)
        print("REAL_L1_INPUT_DAY_READY " + json.dumps({
            "race_date": date, "races": expected_races, "horses": active,
            "target_labels_used": False, "numerical_parity_tested": False
        }, sort_keys=True), flush=True)
    df = pd.DataFrame.from_records(rows)
    need(len(df) == 786 and df["_race_id"].nunique() == 60,
         "2026_TARGET_TOTAL_POPULATION_DRIFT")
    need(not df.duplicated(["_race_id", "_horse_id"]).any(),
         "DUPLICATED_2026_HORSE")
    need(not (set(df.columns) & FULL_FORBIDDEN), "TARGET_FORBIDDEN_LABEL_KEY")
    need(not (set(df.columns) & {"_finish", "_is_win", "_is_top3", "target",
                                 "market_outcome", "_score", "final_win_odds",
                                 "final_popularity", "market_rank"}),
         "TARGET_HAS_OUTCOME_OR_ODDS")
    need(set(df.columns) == set(rows[0]), "TARGET_COLUMN_SCHEMA_DIFFERS")
    need(len(set(df) - ID_COLUMNS) == N_FEATURES, "TARGET_FEATURE_SCHEMA_DRIFT")
    return docs, df


def training_from_pinned_original(prior_dir, target):
    root = Path(prior_dir)
    regime, speed = original.reconstruct_train_regime(root)
    originals = []
    fit = None
    for year in FIT_YEARS:
        base = original.csv(root / f"original-core4-train-{year}.csv.gz")
        bias = original.csv(root / f"original-daybias-{year}.csv.gz")
        need(base["_race_id"].astype(str).str[:4].eq(str(year)).all(),
             "TRAIN_YEAR_DRIFT")
        need(set(base) - META == set(target) - ID_COLUMNS - set(speed) -
             {c for c in target if c.startswith(("daybias_", "fit_",
                                                "horse_front_", "horse_inner_"))},
             "ORIGINAL_TRAIN_CORE4_NOT_MATCH_DELIVERED_649")
        if fit is None:
            fit = bias_cols(bias, with_fit=True)
        need(fit == bias_cols(bias, with_fit=True) and len(fit) == 19,
             "ORIGINAL_TRAIN_DAYBIAS_19_SCHEMA_DRIFT")
        merged = bias[["_race_id", "_horse_id", *fit]].merge(
            regime[year][["_race_id", "_horse_id", *speed]],
            on=["_race_id", "_horse_id"], how="left", validate="one_to_one")
        original_train = original.state_module.attach_state(base, merged, fit + speed)
        need(len(original_train) == len(base), "TRAIN_JOIN_DROPPED_HORSE")
        need(not original_train[["_race_id", "_horse_id"]].duplicated().any(),
             "TRAIN_HORSE_DUPLICATE")
        originals.append(original_train)
        print("REAL_L1_FROZEN_TRAIN_FOLD " + json.dumps({
            "year": year, "horses": len(original_train),
            "races": int(original_train["_race_id"].nunique()),
            "feature_count": len(set(original_train) - META),
        }, sort_keys=True), flush=True)
    train = pd.concat(originals, ignore_index=True)
    need(set(train) - META == set(target) - ID_COLUMNS,
         "ORIGINAL_L1_TRAIN_TARGET_COLUMN_MISMATCH")
    need(len(set(train) - META) == N_FEATURES, "TRAIN_NOT_690")
    need(train["_race_id"].astype(str).str[:4].astype(int).isin(FIT_YEARS).all(),
         "TRAIN_LABEL_YEAR_MUST_BE_2024_2025")
    return train


def score(prior_dir, input_dir, output_dir):
    docs, target = load_delivered(input_dir)
    train = training_from_pinned_original(prior_dir, target)
    train = train.sort_values(["_race_id", "_horse_id"]).reset_index(drop=True)
    target = target.sort_values(["_race_id", "_horse_id"]).reset_index(drop=True)
    xtr, xva, cats = prepare(train, target)
    need(xtr.shape == (len(train), N_FEATURES) and
         xva.shape == (len(target), N_FEATURES), "PREPARE_SHAPE_NOT_690")
    need(list(xtr.columns) == list(xva.columns), "PREPARE_COLUMNS_DIFFER")
    rel = np.select([train["_finish"].eq(k) for k in range(1, 7)],
                    [6, 5, 4, 3, 2, 1], default=0).astype(int)
    groups = train.groupby("_race_id", sort=False).size().tolist()
    need(sum(groups) == len(train), "LAMBDA_GROUP_TOTAL_DRIFT")
    # Identical frozen LightGBM LambdaRank algorithm and model settings.
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg",
        label_gain=[0, 1, 3, 7, 15, 31, 63],
        **model_common(TRAIN_MODEL_SEED))
    print("REAL_L1_FROZEN_FIT_START " + json.dumps({
        "years": list(FIT_YEARS), "rows": len(train), "races": len(groups),
        "columns": xtr.shape[1], "categorical_columns": len(cats),
        "seed": TRAIN_MODEL_SEED, "target_year": 2026,
    }, sort_keys=True), flush=True)
    model.fit(xtr, rel, group=groups, categorical_feature=cats)
    scores = np.asarray(model.predict(xva), dtype=float)
    need(len(scores) == len(target) and np.isfinite(scores).all(),
         "NONFINITE_ORIGINAL_L1_SCORES")
    pred = target[["_race_id", "_horse_id"]].copy()
    pred["_score"] = scores
    ranked = add_ai_context(pred)
    need(not ranked.duplicated(["_race_id", "_horse_id"]).any(),
         "SCORES_DUPLICATED_RACE_HORSE")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for doc in docs:
        rid, day = doc["race_id"], doc["race_date"]
        sub = ranked.loc[ranked["_race_id"].astype(str).eq(rid)].sort_values(
            ["distill_ai_rank", "_horse_id"])
        need(len(sub) == doc["starters"], "SCORES_RACE_HEADCOUNT_DRIFT")
        need(sub["distill_ai_rank"].astype(int).tolist() ==
             list(range(1, len(sub) + 1)), "RAW_RANK_NOT_CONTIGUOUS")
        horses = [{
            "horse_id": str(row["_horse_id"]),
            "ai_raw_rank": int(row["distill_ai_rank"]),
            "ability_score": float(row["_score"]),
            "ability_score_z": float(row["distill_score_z"]),
            "ability_gap_to_best": float(row["distill_score_gap_to_best"]),
        } for _, row in sub.iterrows()]
        result = {
            "contract": CONTRACT, "race_id": rid, "race_date": day,
            "status": "ORIGINAL_L1_ABILITY_ONLY_NOT_COUNCIL_OR_L175_L3",
            "source": "BACKFILL_690_HANDOFF_PINNED",
            "model_origin": "ORIGINAL_PHASE3_20261010_LAMBDARANK",
            "historical_training_years": list(FIT_YEARS),
            "real_original_lambdarank_trained_once_for_60": True,
            "model_feature_count": N_FEATURES,
            "model_seed": TRAIN_MODEL_SEED,
            "2026_training_or_performance_evaluation": False,
            "target_result_or_final_odds_used": False,
            "source_prebet_asof_independently_verified": False,
            "numeric_original_690_parity_verified": False,
            "original_seven_king_outsider_ready": False,
            "l175_p3_ready": False,
            "l21_l3_ready": False,
            "horse_scores": horses,
        }
        (out / (rid + ".json")).write_text(
            json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8")
        summary = {"race_id": rid, "date": day, "horses": len(horses),
                   "top3": [h["horse_id"] for h in horses[:3]]}
        reports.append(summary)
        print("BACKFILL_REAL_ORIGINAL_L1_RACE " + json.dumps(
            summary, ensure_ascii=False, sort_keys=True), flush=True)
    need(len(reports) == 60 and len(ranked) == 786, "NOT_ALL_RACE_SCORES")
    manifest = {
        "contract": "KEIBA_BACKFILL_60_ORIGINAL_L1_ABILITY_BATCH_V1",
        "status": "60_RACES_ORIGINAL_L1_ABILITY_SCORED_RESEARCH_ONLY",
        "races": len(reports), "horses": len(ranked),
        "model_features": N_FEATURES, "historical_train_years": list(FIT_YEARS),
        "train_horses": len(train), "train_races": len(groups),
        "real_original_lambdarank_fit_once": True,
        "model_seed": TRAIN_MODEL_SEED,
        "2026_outcomes_not_used_or_evaluated": True,
        "original_690_numerical_parity_not_checked": True,
        "prestart_asof_not_independently_verified": True,
        "seven_king_outsider_l175_l21_l3_not_executed": True,
        "race_results": reports,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("BACKFILL_690_REAL_L1_60_SUCCESS " + json.dumps(
        {k: v for k, v in manifest.items() if k != "race_results"},
        sort_keys=True, ensure_ascii=False), flush=True)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-dir", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    score(args.prior_dir, args.input_dir, args.output_dir)
