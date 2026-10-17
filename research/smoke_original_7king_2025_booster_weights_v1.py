#!/usr/bin/env python3
"""Read-only smoke for the seven ACTUAL archived 2025 king boosters.

Loads archive model.txt/model.json/schema.json from existing 2025 Kaggle
datasets and verifies model hashes, feature lists, best iterations, LightGBM
prediction/TreeSHAP invariants using three SYNTHETIC rows. No real race
rows/2026 results are opened and NO historical booster is promoted or scored
against the 2026 races.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from audit_7king_2026_blind_training_assets_v1 import ALIASES, COLUMNS, CONFIG_NAMES
from emit_l1_to_l2 import sigmoid

GEN = "eb13d4096519167b"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def synthetic_frame(names, levels):
    # Inputs are intentionally artificial. No target or horse information.
    data = {}
    for name in names:
        if name in levels:
            choices = [str(x) for x in levels[name]]
            require(choices, "EMPTY_ARCHIVED_CATEGORICAL_LEVELS:" + name)
            data[name] = pd.Categorical(
                [choices[0], choices[-1], None],
                categories=choices,
            )
        else:
            data[name] = [0.0, 1.0, float("nan")]
    return pd.DataFrame(data, columns=names)


def check_archive(directory, alias, candidate, column_count):
    p = Path(directory) / alias
    paths = {name: p / name for name in ("model.txt", "model.json", "schema.json")}
    for name, path in paths.items():
        require(path.is_file() and path.stat().st_size > 0,
                "ARCHIVED_2025_FILE_MISSING:" + alias + ":" + name)
    meta = json.loads(paths["model.json"].read_text(encoding="utf-8"))
    schema = json.loads(paths["schema.json"].read_text(encoding="utf-8"))
    train = meta.get("split") or {}
    require(train.get("train_start") == "2023-01-01" and
            train.get("train_end") == "2024-12-31" and
            train.get("valid_start") == "2025-01-01" and
            train.get("valid_end") == "2025-12-31",
            "NOT_THE_ORIGINAL_2025_OUT_OF_SAMPLE_FOLD:" + alias)
    names = schema.get("feature_order")
    levels = schema.get("category_levels") or {}
    require(isinstance(names, list) and len(names) == column_count and
            len(set(names)) == column_count and
            meta.get("feature_count") == column_count and
            meta.get("features") == names and
            meta.get("category_levels") == levels,
            "ARCHIVED_2025_MODEL_SCHEMA_META_DRIFT:" + alias)
    require(meta.get("prediction_phase") == "FINAL" and
            meta.get("ability_uses_odds") is False,
            "ARCHIVED_2025_ODDS_OR_PREDICTION_PHASE_DRIFT:" + alias)
    expected_hash = (meta.get("reproducibility") or {}).get("model_sha256")
    require(isinstance(expected_hash, str) and len(expected_hash) == 64,
            "ARCHIVED_MODEL_SHA256_NOT_RECORDED:" + alias)
    observed_hash = sha256_file(paths["model.txt"])
    require(observed_hash == expected_hash,
            "ARCHIVED_2025_BOOSTER_HASH_MISMATCH:" + alias)
    iteration = meta.get("best_iteration")
    require(type(iteration) is int and 1 <= iteration <= 600,
            "ARCHIVED_2025_BEST_ITERATION_INVALID:" + alias)
    booster = lgb.Booster(model_file=str(paths["model.txt"]))
    require(booster.feature_name() == names and
            booster.num_feature() == column_count and
            booster.current_iteration() >= iteration,
            "ARCHIVED_2025_BOOSTER_FEATURES_OR_ITERATIONS_MISMATCH:" + alias)
    frame = synthetic_frame(names, levels)
    prediction = np.asarray(
        booster.predict(frame, num_iteration=iteration),
        dtype=float,
    )
    margins = np.asarray(
        booster.predict(frame, num_iteration=iteration, raw_score=True),
        dtype=float,
    )
    contributions = np.asarray(
        booster.predict(frame, num_iteration=iteration, pred_contrib=True),
        dtype=float,
    )
    require(prediction.shape == (3,) and margins.shape == (3,) and
            contributions.shape == (3, column_count + 1) and
            np.isfinite(prediction).all() and
            np.isfinite(margins).all() and
            np.isfinite(contributions).all() and
            (prediction >= 0).all() and (prediction <= 1).all(),
            "ARCHIVED_2025_BOOSTER_REAL_PREDICTION_BAD:" + alias)
    for i in range(3):
        require(abs(sigmoid(float(margins[i])) - prediction[i]) < 1e-6 and
                abs(float(contributions[i].sum()) - margins[i]) < 1e-5,
                "ARCHIVED_2025_BOOSTER_SHAP_RECONSTRUCTION_DRIFT:" + alias)
    return {
        "alias": alias,
        "candidate": candidate,
        "original_fold_year": 2025,
        "train_years": [2023, 2024],
        "features": column_count,
        "best_iteration": iteration,
        "booster_sha256_validated": True,
        "booster_loaded": True,
        "synthetic_inference_rows": 3,
        "tree_shap_logit_checked": True,
        "scored_actual_2026_races": False,
        "trained_2026_model": False,
        "deployable_2026": False,
    }


def smoke(models, report):
    freeze = json.loads(Path("research/l1-raceclass-absorption-walkforward-v1.json").read_text())
    require(freeze.get("snapshot_generation") == GEN and
            freeze.get("locked_years") == [2026] and
            freeze.get("training_window_years") == 2 and
            tuple(row["name"] for row in freeze["candidates"]) == CONFIG_NAMES,
            "ARCHIVED_MODEL_SMOKE_FROZEN_RESEARCH_DRIFT")
    results = []
    for alias, name, count in zip(ALIASES, CONFIG_NAMES, COLUMNS):
        checked = check_archive(models, alias, name, count)
        results.append(checked)
        print("ARCHIVED_2025_KING_REAL_BOOSTER_SMOKE " +
              json.dumps(checked, ensure_ascii=False, sort_keys=True),
              flush=True)
    require(len(results) == 7, "ARCHIVED_MODEL_SEVEN_OWNERS_MISSING")
    summary = {
        "contract": "KEIBA_ORIGINAL_2025_SEVEN_KING_ARCHIVED_BOOSTER_SMOKE_V1",
        "models": 7,
        "historical_year": 2025,
        "historical_models_promoted": False,
        "historical_boosters_loaded": True,
        "real_booster_sha256_checks": 7,
        "synthetic_prediction_rows_per_king": 3,
        "2026_outcomes_or_real_races_read": False,
        "2026_models_trained_or_scored": False,
        "asof_verified": False,
        "betting_ready": False,
        "kings": results,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print("ARCHIVED_2025_SEVEN_REAL_LIGHTGBM_BOOSTERS_7_OF_7_SMOKE_PASS",
          flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--models", type=Path, required=True)
    cli.add_argument("--report", type=Path, required=True)
    args = cli.parse_args()
    smoke(args.models, args.report)
