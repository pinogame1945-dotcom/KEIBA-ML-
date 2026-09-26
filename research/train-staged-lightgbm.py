#!/usr/bin/env python3
import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EXPECTED_DATASET_VERSION = 2
EXPECTED_FEATURE_SCHEMA_VERSION = 3
EXPECTED_LEAKAGE_POLICY = "STRICT_PRIOR_DATE_ONLY"

STAGES = {
    "base": [],
    "opponent_v1": ["opponent_"],
    "opponent_both": ["opponent_", "network_"],
    "lap": ["opponent_", "network_", "lap_"],
    "style": ["opponent_", "network_", "lap_", "style_"],
    "pedigree": ["opponent_", "network_", "lap_", "style_", "pedigree_"],
    "distance": ["opponent_", "network_", "lap_", "style_", "pedigree_", "distx_"],
}
EXTRA_PREFIXES = ["opponent_", "network_", "lap_", "style_", "pedigree_", "distx_"]

BASE_CATEGORICAL = [
    "venue_code", "discipline", "surface", "direction", "weather",
    "track_condition", "sex", "jockey_id", "trainer_id",
]
PEDIGREE_CATEGORICAL = [
    "pedigree_sire_id", "pedigree_dam_id",
    "pedigree_siresire_id", "pedigree_damsire_id",
]

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--stage", choices=list(STAGES), required=True)
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--valid-start", required=True)
    p.add_argument("--valid-end", required=True)
    p.add_argument("--meta-out", required=True)
    p.add_argument("--model-out", required=True)
    p.add_argument("--predictions-out")
    p.add_argument("--schema-out")
    p.add_argument("--model-version", default="L1_LIGHTGBM_EXPERIMENTAL")
    p.add_argument("--source-repo", default="pinogame1945-dotcom/KEIBA-BACKFILL")
    p.add_argument("--source-ref")
    p.add_argument("--source-sha")
    return p.parse_args()

def load(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("ml_dataset_version", 0)) != EXPECTED_DATASET_VERSION:
                raise ValueError("dataset version mismatch")
            if int(row.get("feature_schema_version", 0)) != EXPECTED_FEATURE_SCHEMA_VERSION:
                raise ValueError("feature schema mismatch")
            if row.get("leakage_policy") != EXPECTED_LEAKAGE_POLICY:
                raise ValueError("leakage policy mismatch")
            rows.append(row)
    if not rows:
        raise ValueError("empty dataset")
    return rows

def hhmm(value):
    if value is None:
        return np.nan
    text = str(value).strip()
    if ":" not in text:
        return np.nan
    try:
        h, m = map(int, text.split(":", 1))
    except ValueError:
        return np.nan
    return h * 60 + m if 0 <= h <= 23 and 0 <= m <= 59 else np.nan

def stage_features(features, stage):
    allowed = STAGES[stage]
    out = {}
    for key, value in features.items():
        matched = next((prefix for prefix in EXTRA_PREFIXES if key.startswith(prefix)), None)
        if matched is None or matched in allowed:
            out[key] = value
    return out

def flatten(rows, stage):
    records = []
    for row in rows:
        f = stage_features(dict(row["features"]), stage)
        date = f.pop("race_date", None)
        start = f.pop("actual_start_time", None)
        dt = pd.to_datetime(date, errors="coerce")
        f["race_month"] = int(dt.month) if not pd.isna(dt) else np.nan
        f["race_day_of_year"] = int(dt.dayofyear) if not pd.isna(dt) else np.nan
        f["start_minutes"] = hhmm(start)
        target = row.get("target") or {}
        is_win = target.get("is_win")
        if is_win is None:
            continue
        records.append({
            "_race_id": str(row.get("race_id") or ""),
            "_horse_id": str(row.get("horse_id") or ""),
            "_race_date": str(date or ""),
            "_target": 1 if bool(is_win) else 0,
            **f,
        })
    df = pd.DataFrame.from_records(records)
    df["_race_date_dt"] = pd.to_datetime(df["_race_date"], errors="coerce")
    return df[df["_race_id"].ne("") & df["_horse_id"].ne("") & df["_race_date_dt"].notna()].copy()

def complete_races(df):
    winners = df.groupby("_race_id")["_target"].sum()
    ids = winners[winners >= 1].index
    return df[df["_race_id"].isin(ids)].copy()

def split(df, a):
    train = df[
        (df["_race_date_dt"] >= pd.to_datetime(a.train_start)) &
        (df["_race_date_dt"] <= pd.to_datetime(a.train_end))
    ].copy()
    valid = df[
        (df["_race_date_dt"] >= pd.to_datetime(a.valid_start)) &
        (df["_race_date_dt"] <= pd.to_datetime(a.valid_end))
    ].copy()
    train, valid = complete_races(train), complete_races(valid)
    if train.empty or valid.empty:
        raise ValueError("empty split")
    return train, valid

def frames(train, valid):
    cols = [c for c in train.columns if not c.startswith("_")]
    xtr, xva = train[cols].copy(), valid[cols].copy()
    categorical = [c for c in BASE_CATEGORICAL + PEDIGREE_CATEGORICAL if c in cols]
    for c in categorical:
        tv = xtr[c].astype("string").fillna("__MISSING__")
        cats = sorted(set(tv.tolist()))
        xtr[c] = pd.Categorical(tv, categories=cats)
        vv = xva[c].astype("string").fillna("__MISSING__")
        xva[c] = pd.Categorical(vv, categories=cats)
    for c in cols:
        if c in categorical:
            continue
        xtr[c] = pd.to_numeric(xtr[c], errors="coerce")
        xva[c] = pd.to_numeric(xva[c], errors="coerce")
    category_levels = {
        c: [str(v) for v in xtr[c].cat.categories]
        for c in categorical
    }
    return xtr, xva, categorical, category_levels

def predictions(frame, raw):
    out = frame[["_race_id", "_horse_id", "_target"]].copy()
    out["p"] = np.asarray(raw, dtype=float)
    sums = out.groupby("_race_id")["p"].transform("sum")
    sizes = out.groupby("_race_id")["p"].transform("size")
    out["pn"] = np.where(sums > 0, out["p"] / sums, 1.0 / sizes)
    out["rank"] = out.groupby("_race_id")["p"].rank(method="first", ascending=False).astype(int)
    return out

def metrics(pred, y, raw):
    ranks, probs = [], []
    for _, g in pred.groupby("_race_id", sort=False):
        w = g[g["_target"] == 1]
        if w.empty:
            continue
        ranks.append(int(w["rank"].min()))
        probs.append(float(w["pn"].sum()))
    ranks = np.asarray(ranks, dtype=float)
    probs = np.clip(np.asarray(probs, dtype=float), 1e-15, 1.0)
    return {
        "races": int(len(ranks)),
        "top1_winner_capture": float(np.mean(ranks <= 1)),
        "top3_winner_capture": float(np.mean(ranks <= 3)),
        "top6_winner_capture": float(np.mean(ranks <= 6)),
        "mean_winner_rank": float(np.mean(ranks)),
        "mean_reciprocal_winner_rank": float(np.mean(1.0 / ranks)),
        "race_normalized_nll": float(np.mean(-np.log(probs))),
        "raw_logloss": float(log_loss(y, raw, labels=[0, 1])),
        "raw_brier": float(brier_score_loss(y, raw)),
        "raw_roc_auc": float(roc_auc_score(y, raw)),
    }

def main():
    a = args()
    if pd.to_datetime(a.train_end) >= pd.to_datetime(a.valid_start):
        raise ValueError("train_end must be before valid_start")
    if pd.to_datetime(a.valid_start) > pd.to_datetime(a.valid_end):
        raise ValueError("valid_start must be <= valid_end")
    df = flatten(load(a.dataset), a.stage)
    train, valid = split(df, a)
    xtr, xva, categorical, category_levels = frames(train, valid)
    ytr, yva = train["_target"].astype(int), valid["_target"].astype(int)

    params = {
        "objective": "binary",
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 30,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        xtr, ytr,
        eval_set=[(xva, yva)],
        eval_metric="binary_logloss",
        categorical_feature=categorical,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    raw = np.clip(model.predict_proba(xva, num_iteration=model.best_iteration_)[:, 1], 1e-15, 1 - 1e-15)
    pred = predictions(valid, raw)
    result_metrics = metrics(pred, yva, raw)

    Path(a.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(a.model_out)
    gain = model.booster_.feature_importance(importance_type="gain")
    names = model.booster_.feature_name()
    importance = sorted(
        [{"feature": n, "gain": float(g)} for n, g in zip(names, gain)],
        key=lambda x: x["gain"], reverse=True,
    )
    meta = {
        "model_version": a.model_version,
        "stage": a.stage,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "ability_uses_odds": False,
        "dataset_contract": {
            "ml_dataset_version": EXPECTED_DATASET_VERSION,
            "feature_schema_version": EXPECTED_FEATURE_SCHEMA_VERSION,
            "leakage_policy": EXPECTED_LEAKAGE_POLICY,
        },
        "source": {
            "repository": a.source_repo,
            "ref": a.source_ref,
            "sha": a.source_sha,
        },
        "split": {
            "train_start": a.train_start, "train_end": a.train_end,
            "valid_start": a.valid_start, "valid_end": a.valid_end,
            "train_rows": int(len(train)), "train_races": int(train["_race_id"].nunique()),
            "valid_rows": int(len(valid)), "valid_races": int(valid["_race_id"].nunique()),
        },
        "feature_count": len(names),
        "features": names,
        "categorical_features": categorical,
        "category_levels": category_levels,
        "params": params,
        "best_iteration": int(model.best_iteration_ or params["n_estimators"]),
        "metrics": result_metrics,
        "feature_importance_gain": importance,
    }

    if a.predictions_out:
        out_path = Path(a.predictions_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out = pred.copy()
        out["_race_date"] = valid["_race_date"].values
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            for record in out.sort_values(["_race_date", "_race_id", "rank"]).to_dict(orient="records"):
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    if a.schema_out:
        schema_path = Path(a.schema_out)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema = {
            "contract": "L1_FEATURE_SCHEMA_V1",
            "ml_dataset_version": EXPECTED_DATASET_VERSION,
            "feature_schema_version": EXPECTED_FEATURE_SCHEMA_VERSION,
            "leakage_policy": EXPECTED_LEAKAGE_POLICY,
            "ability_uses_odds": False,
            "stage": a.stage,
            "feature_count": len(names),
            "feature_order": names,
            "categorical_features": categorical,
            "category_levels": category_levels,
        }
        schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    Path(a.meta_out).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("ML_STAGE_RESULT")
    print(json.dumps({
        "stage": a.stage,
        "feature_count": len(names),
        "split": meta["split"],
        "metrics": result_metrics,
        "top_features": importance[:12],
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
