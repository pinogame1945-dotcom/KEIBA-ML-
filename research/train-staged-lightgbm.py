#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EXPECTED_DATASET_VERSION = 3
EXPECTED_FEATURE_SCHEMA_VERSION = 7
EXPECTED_LEAKAGE_POLICY = "STRICT_PRIOR_DATE_ONLY"

DEFAULT_FEATURE_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contracts" / "l1-feature-set-contract-v1.json"
DEFAULT_FEATURE_CONTRACT = json.loads(DEFAULT_FEATURE_CONTRACT_PATH.read_text(encoding="utf-8"))
LEGACY_STAGES = sorted(DEFAULT_FEATURE_CONTRACT["legacy_stage_map"])

BASE_CATEGORICAL = [
    "venue_code", "discipline", "surface", "direction", "weather",
    "track_condition", "sex",
    "backfill_course_layout", "backfill_race_class_normalized", "backfill_grade",
    "backfill_sex_condition", "backfill_weight_rule",
]

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--feature-sets", help="Comma-separated independent feature families, e.g. BASE,DISTANCE")
    p.add_argument("--stage", choices=LEGACY_STAGES, help="Deprecated cumulative stage; translated to Feature Sets")
    p.add_argument("--prediction-phase", default=DEFAULT_FEATURE_CONTRACT["default_prediction_phase"], choices=sorted(DEFAULT_FEATURE_CONTRACT["prediction_phases"]))
    p.add_argument("--history-windows-json", help="JSON object recording the dataset history-window contract")
    p.add_argument("--feature-contract", default=str(DEFAULT_FEATURE_CONTRACT_PATH))
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
    p.add_argument("--ml-source-sha")
    p.add_argument(
        "--feature-catalog",
        default=str(Path(__file__).resolve().parents[1] / "contracts" / "l1-feature-catalog-v1.json"),
    )
    p.add_argument("--diagnostics-out")
    p.add_argument("--contributions-out")
    return p.parse_args()

def load_feature_contract(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_feature_sets(raw, legacy_stage, contract):
    order = list(contract["feature_sets"])
    valid = set(order)
    if raw:
        requested = []
        for item in str(raw).split(","):
            name = item.strip().upper()
            if name and name not in requested:
                requested.append(name)
    elif legacy_stage:
        requested = list(contract["legacy_stage_map"][legacy_stage])
    else:
        requested = list(contract["default_feature_sets"])
    invalid = [name for name in requested if name not in valid]
    if invalid:
        raise ValueError("invalid feature set(s): " + ", ".join(invalid))
    if "BASE" not in requested:
        requested.append("BASE")
    canonical = [name for name in order if name in requested]
    if raw and legacy_stage:
        legacy = set(contract["legacy_stage_map"][legacy_stage])
        if set(canonical) != legacy:
            raise ValueError("--feature-sets conflicts with deprecated --stage mapping")
    return canonical


def normalize_prediction_phase(value, contract):
    phase = str(value or contract["default_prediction_phase"]).strip().upper()
    if phase not in contract["prediction_phases"]:
        raise ValueError("invalid prediction phase: " + phase)
    return phase


def normalize_history_windows(raw, contract):
    defaults = dict(contract["history_windows"])
    if not raw:
        return defaults
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--history-windows-json must be a JSON object")
    out = dict(defaults)
    for key, value in parsed.items():
        if key not in defaults:
            raise ValueError("unknown history window: " + str(key))
        if defaults[key] == "ALL":
            if str(value).upper() != "ALL":
                raise ValueError(f"{key} history window must stay ALL")
            out[key] = "ALL"
            continue
        lo, hi = contract["history_window_bounds"][key]
        numeric = float(value)
        if not numeric.is_integer():
            raise ValueError(f"{key} history window must be an integer")
        n = int(numeric)
        if n < lo or n > hi:
            raise ValueError(f"{key} history window must be from {lo} to {hi}")
        out[key] = n
    return out


def feature_set_features(features, feature_sets, contract):
    prefix_map = {name: list(spec.get("prefixes") or []) for name, spec in contract["feature_sets"].items()}
    all_prefixes = []
    for prefixes in prefix_map.values():
        for prefix in prefixes:
            if prefix not in all_prefixes:
                all_prefixes.append(prefix)
    allowed = {prefix for name in feature_sets for prefix in prefix_map[name]}
    out = {}
    for key, value in features.items():
        matched = next((prefix for prefix in all_prefixes if key.startswith(prefix)), None)
        if matched is None or matched in allowed:
            out[key] = value
    return out


def prediction_phase_blocked(columns, prediction_phase, contract):
    policy = contract["prediction_phases"][prediction_phase]
    exact = set(policy.get("blocked_exact_model_keys") or [])
    patterns = [re.compile(raw) for raw in policy.get("blocked_model_key_patterns") or []]
    return sorted(
        column for column in columns
        if column in exact or any(pattern.search(column) for pattern in patterns)
    )


def assert_prediction_phase_safety(columns, prediction_phase, contract):
    bad = prediction_phase_blocked(columns, prediction_phase, contract)
    if bad:
        raise ValueError(
            f"prediction phase {prediction_phase} blocked model columns: " + ", ".join(bad)
        )


def load_forbidden_model_keys(path):
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    forbidden = set()
    for row in catalog.get("features", []):
        if row.get("l1_allowed") is False:
            forbidden.update(str(key) for key in row.get("model_keys", []) if str(key))
    return forbidden


def assert_catalog_safety(columns, catalog_path):
    forbidden = load_forbidden_model_keys(catalog_path)
    bad = sorted(set(columns) & forbidden)
    if bad:
        raise ValueError("L1 feature catalog blocked model columns: " + ", ".join(bad))


def assert_catalog_availability(columns, catalog_path, prediction_phase, feature_contract):
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    allowed = set(feature_contract["prediction_phases"][prediction_phase]["allowed_available_at"])
    bad = []
    for row in catalog.get("features", []):
        keys = set(str(key) for key in row.get("model_keys", []) if str(key))
        if keys.intersection(columns) and row.get("available_at") not in allowed:
            bad.extend(sorted(keys.intersection(columns)))
    if bad:
        raise ValueError(
            f"prediction phase {prediction_phase} catalog availability blocked model columns: "
            + ", ".join(sorted(set(bad)))
        )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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

def flatten(rows, feature_sets, prediction_phase, history_windows, feature_contract, forbidden_model_keys=None):
    forbidden_model_keys = set(forbidden_model_keys or [])
    records = []
    for row in rows:
        row_phase = row.get("prediction_phase")
        if row_phase and str(row_phase).upper() != prediction_phase:
            raise ValueError("dataset prediction_phase mismatch")
        row_sets = row.get("feature_sets")
        if row_sets and list(row_sets) != feature_sets:
            raise ValueError("dataset feature_sets mismatch")
        row_windows = row.get("history_windows")
        if row_windows and dict(row_windows) != history_windows:
            raise ValueError("dataset history_windows mismatch")
        f = feature_set_features(dict(row["features"]), feature_sets, feature_contract)
        assert_prediction_phase_safety(list(f), prediction_phase, feature_contract)
        bad = sorted(set(f) & forbidden_model_keys)
        if bad:
            raise ValueError("L1 feature catalog blocked dataset columns: " + ", ".join(bad))
        date = f.pop("race_date", None)
        f.pop("actual_start_time", None)
        f.pop("jockey_id", None)
        f.pop("trainer_id", None)
        dt = pd.to_datetime(date, errors="coerce")
        f["race_month"] = int(dt.month) if not pd.isna(dt) else np.nan
        f["race_day_of_year"] = int(dt.dayofyear) if not pd.isna(dt) else np.nan
        target = row.get("target") or {}
        is_win = target.get("is_win")
        if is_win is None:
            continue
        records.append({
            "_race_id": str(row.get("race_id") or ""),
            "_horse_id": str(row.get("horse_id") or ""),
            "_race_date": str(date or ""),
            "_target": 1 if bool(is_win) else 0,
            "_finish_position": target.get("finish_position"),
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
    categorical = [c for c in BASE_CATEGORICAL if c in cols]
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
    out = frame[["_race_id", "_horse_id", "_target", "_finish_position"]].copy()
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


def race_capture_metrics(pred):
    ranks, probs = [], []
    for _, g in pred.groupby("_race_id", sort=False):
        w = g[g["_target"] == 1]
        if w.empty:
            continue
        ranks.append(int(w["rank"].min()))
        probs.append(float(w["pn"].sum()))
    if not ranks:
        return {"races": 0}
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
    }


def subgroup_metrics(valid, pred):
    context = valid.groupby("_race_id", sort=False).first().reset_index()
    field_sizes = valid.groupby("_race_id").size().rename("field_size")
    context = context.merge(field_sizes, left_on="_race_id", right_index=True, how="left")
    if "distance_m" in context.columns:
        distance = pd.to_numeric(context["distance_m"], errors="coerce")
        context["distance_band"] = pd.cut(
            distance,
            bins=[-np.inf, 1400, 1800, 2200, np.inf],
            labels=["<=1400", "1401-1800", "1801-2200", ">=2201"],
        ).astype("string")
    dimensions = {
        "surface": "surface",
        "venue": "venue_code",
        "distance_band": "distance_band",
        "race_class": "backfill_race_class_normalized",
        "grade": "backfill_grade",
        "course_layout": "backfill_course_layout",
        "field_size": "field_size",
    }
    report = {}
    for label, column in dimensions.items():
        if column not in context.columns:
            continue
        rows = []
        values = context[column].astype("string").fillna("__MISSING__")
        for value in sorted(values.unique().tolist()):
            race_ids = set(context.loc[values == value, "_race_id"].astype(str))
            metric = race_capture_metrics(pred[pred["_race_id"].astype(str).isin(race_ids)])
            rows.append({"value": str(value), **metric})
        report[label] = rows
    return report


def feature_coverage(frame, columns):
    out = {}
    for column in columns:
        series = frame[column]
        present = int(series.notna().sum())
        out[column] = {
            "rows": int(len(series)),
            "present": present,
            "non_null_rate": float(present / len(series)) if len(series) else None,
        }
    return out


def split_threshold_summary(model_dump):
    summary = {}

    def visit(node):
        if not isinstance(node, dict):
            return
        if "split_feature" in node:
            name = model_dump["feature_names"][int(node["split_feature"])]
            row = summary.setdefault(name, {
                "split_count": 0,
                "gain_sum": 0.0,
                "threshold_samples": [],
            })
            row["split_count"] += 1
            row["gain_sum"] += float(node.get("split_gain") or 0.0)
            if len(row["threshold_samples"]) < 25:
                row["threshold_samples"].append(str(node.get("threshold")))
            visit(node.get("left_child"))
            visit(node.get("right_child"))

    for tree in model_dump.get("tree_info", []):
        visit(tree.get("tree_structure"))
    return sorted(
        [{"feature": name, **row} for name, row in summary.items()],
        key=lambda row: (-row["gain_sum"], -row["split_count"], row["feature"]),
    )


def write_json_payload(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if str(target).endswith(".gz"):
        with gzip.open(target, "wt", encoding="utf-8") as fh:
            fh.write(raw)
            fh.write("\n")
    else:
        target.write_text(raw + "\n", encoding="utf-8")


def write_model_diagnostics(model, xva, valid, pred, names, diagnostics_out=None, contributions_out=None):
    if not diagnostics_out and not contributions_out:
        return None
    booster = model.booster_
    model_dump = booster.dump_model()
    sum_abs = np.zeros(len(names), dtype=float)
    rows_seen = 0
    contrib_handle = None
    try:
        if contributions_out:
            target = Path(contributions_out)
            target.parent.mkdir(parents=True, exist_ok=True)
            contrib_handle = gzip.open(target, "wt", encoding="utf-8") if str(target).endswith(".gz") else open(target, "w", encoding="utf-8")
        chunk = 2000
        for start in range(0, len(xva), chunk):
            stop = min(len(xva), start + chunk)
            matrix = booster.predict(
                xva.iloc[start:stop],
                pred_contrib=True,
                num_iteration=model.best_iteration_,
            )
            matrix = np.asarray(matrix, dtype=float)
            values = matrix[:, :len(names)]
            sum_abs += np.abs(values).sum(axis=0)
            rows_seen += len(values)
            if contrib_handle is not None:
                for offset, row_values in enumerate(values):
                    order = np.argsort(np.abs(row_values))[-5:][::-1]
                    p = pred.iloc[start + offset]
                    record = {
                        "race_id": str(p["_race_id"]),
                        "horse_id": str(p["_horse_id"]),
                        "predicted_rank": int(p["rank"]),
                        "top_contributions": [
                            {"feature": names[int(i)], "value": float(row_values[int(i)])}
                            for i in order
                        ],
                    }
                    contrib_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if contrib_handle is not None:
            contrib_handle.close()

    mean_abs = sum_abs / rows_seen if rows_seen else sum_abs
    contribution_summary = sorted(
        [{"feature": name, "mean_abs_contribution": float(value)} for name, value in zip(names, mean_abs)],
        key=lambda row: (-row["mean_abs_contribution"], row["feature"]),
    )
    if diagnostics_out:
        write_json_payload(diagnostics_out, {
            "contract": "L1_MODEL_DIAGNOSTICS_V1",
            "feature_count": len(names),
            "validation_rows": int(len(xva)),
            "feature_importance_gain": sorted(
                [
                    {"feature": n, "gain": float(g)}
                    for n, g in zip(
                        booster.feature_name(),
                        booster.feature_importance(importance_type="gain"),
                    )
                ],
                key=lambda row: (-row["gain"], row["feature"]),
            ),
            "mean_abs_contribution": contribution_summary,
            "split_threshold_summary": split_threshold_summary(model_dump),
            "model_dump": model_dump,
        })
    return contribution_summary



def main():
    a = args()
    if pd.to_datetime(a.train_end) >= pd.to_datetime(a.valid_start):
        raise ValueError("train_end must be before valid_start")
    if pd.to_datetime(a.valid_start) > pd.to_datetime(a.valid_end):
        raise ValueError("valid_start must be <= valid_end")
    feature_contract = load_feature_contract(a.feature_contract)
    feature_sets = normalize_feature_sets(a.feature_sets, a.stage, feature_contract)
    prediction_phase = normalize_prediction_phase(a.prediction_phase, feature_contract)
    history_windows = normalize_history_windows(a.history_windows_json, feature_contract)
    forbidden_model_keys = load_forbidden_model_keys(a.feature_catalog)
    df = flatten(
        load(a.dataset),
        feature_sets,
        prediction_phase,
        history_windows,
        feature_contract,
        forbidden_model_keys,
    )
    train, valid = split(df, a)
    xtr, xva, categorical, category_levels = frames(train, valid)
    assert_catalog_safety(list(xtr.columns), a.feature_catalog)
    assert_catalog_availability(list(xtr.columns), a.feature_catalog, prediction_phase, feature_contract)
    assert_prediction_phase_safety(list(xtr.columns), prediction_phase, feature_contract)
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
    subgroup_report = subgroup_metrics(valid, pred)

    Path(a.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(a.model_out)
    gain = model.booster_.feature_importance(importance_type="gain")
    names = model.booster_.feature_name()
    importance = sorted(
        [{"feature": n, "gain": float(g)} for n, g in zip(names, gain)],
        key=lambda x: x["gain"], reverse=True,
    )
    split_config = {
        "train_start": a.train_start,
        "train_end": a.train_end,
        "valid_start": a.valid_start,
        "valid_end": a.valid_end,
    }
    training_config = {
        "dataset_contract": {
            "ml_dataset_version": EXPECTED_DATASET_VERSION,
            "feature_schema_version": EXPECTED_FEATURE_SCHEMA_VERSION,
            "leakage_policy": EXPECTED_LEAKAGE_POLICY,
        },
        "prediction_phase": prediction_phase,
        "feature_sets": feature_sets,
        "history_windows": history_windows,
        "legacy_stage": a.stage,
        "split": split_config,
        "params": params,
        "feature_order": names,
        "categorical_features": categorical,
        "source": {
            "repository": a.source_repo,
            "ref": a.source_ref,
            "sha": a.source_sha,
            "ml_source_sha": a.ml_source_sha,
        },
    }
    model_sha256 = sha256_file(a.model_out)
    training_config_sha256 = sha256_json(training_config)
    catalog_sha256 = sha256_file(a.feature_catalog)
    feature_contract_sha256 = sha256_file(a.feature_contract)
    contribution_summary = write_model_diagnostics(
        model,
        xva,
        valid,
        pred,
        names,
        diagnostics_out=a.diagnostics_out,
        contributions_out=a.contributions_out,
    )
    meta = {
        "model_version": a.model_version,
        "prediction_phase": prediction_phase,
        "feature_sets": feature_sets,
        "history_windows": history_windows,
        "legacy_stage": a.stage,
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
            "ml_source_sha": a.ml_source_sha,
        },
        "runtime": {
            "python": platform.python_version(),
            "lightgbm": package_version("lightgbm"),
            "pandas": package_version("pandas"),
            "scikit_learn": package_version("scikit-learn"),
            "numpy": package_version("numpy"),
        },
        "split": {
            **split_config,
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
        "subgroup_metrics": subgroup_report,
        "feature_coverage": {
            "train": feature_coverage(xtr, names),
            "valid": feature_coverage(xva, names),
        },
        "reproducibility": {
            "model_sha256": model_sha256,
            "training_config_sha256": training_config_sha256,
            "feature_catalog_sha256": catalog_sha256,
            "feature_catalog": str(Path(a.feature_catalog)),
            "feature_contract_sha256": feature_contract_sha256,
            "feature_contract": str(Path(a.feature_contract)),
        },
        "diagnostics": {
            "contract": "L1_MODEL_DIAGNOSTICS_V1" if a.diagnostics_out else None,
            "diagnostics_out": a.diagnostics_out,
            "contributions_out": a.contributions_out,
            "mean_abs_contribution": contribution_summary,
        },
    }

    if a.predictions_out:
        out_path = Path(a.predictions_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out = pred.copy()
        out["_race_date"] = valid["_race_date"].values
        out["model_version"] = a.model_version
        out["prediction_phase"] = prediction_phase
        out["ml_dataset_version"] = EXPECTED_DATASET_VERSION
        out["feature_schema_version"] = EXPECTED_FEATURE_SCHEMA_VERSION
        out["leakage_policy"] = EXPECTED_LEAKAGE_POLICY
        out = out.rename(columns={
            "_race_date": "race_date",
            "_race_id": "race_id",
            "_horse_id": "horse_id",
            "_target": "actual_is_win",
            "_finish_position": "actual_finish_position",
            "p": "raw_win_probability",
            "pn": "race_normalized_win_probability",
            "rank": "predicted_rank",
        })
        cols = [
            "race_date", "race_id", "horse_id", "model_version", "prediction_phase",
            "ml_dataset_version", "feature_schema_version", "leakage_policy",
            "raw_win_probability", "race_normalized_win_probability", "predicted_rank",
            "actual_is_win", "actual_finish_position",
        ]
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            for record in out.sort_values(["race_date", "race_id", "predicted_rank"])[cols].to_dict(orient="records"):
                record["feature_sets"] = feature_sets
                record["history_windows"] = history_windows
                record["exact_feature_list"] = names
                record["feature_catalog_sha256"] = catalog_sha256
                record["feature_contract_sha256"] = feature_contract_sha256
                record["model_sha256"] = model_sha256
                record["training_config_sha256"] = training_config_sha256
                record["source_backfill_sha"] = a.source_sha
                record["ml_source_sha"] = a.ml_source_sha
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
            "prediction_phase": prediction_phase,
            "feature_sets": feature_sets,
            "history_windows": history_windows,
            "legacy_stage": a.stage,
            "feature_count": len(names),
            "feature_order": names,
            "categorical_features": categorical,
            "category_levels": category_levels,
            "reproducibility": {
                "model_sha256": model_sha256,
                "training_config_sha256": training_config_sha256,
                "feature_catalog_sha256": catalog_sha256,
                "feature_contract_sha256": feature_contract_sha256,
                "source_backfill_sha": a.source_sha,
                "ml_source_sha": a.ml_source_sha,
            },
        }
        schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    Path(a.meta_out).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("ML_FEATURE_SET_RESULT")
    print(json.dumps({
        "prediction_phase": prediction_phase,
        "feature_sets": feature_sets,
        "history_windows": history_windows,
        "legacy_stage": a.stage,
        "feature_count": len(names),
        "split": meta["split"],
        "metrics": result_metrics,
        "top_features": importance[:12],
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
