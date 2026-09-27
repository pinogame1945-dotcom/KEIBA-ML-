#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FAMILIES = [
    "BASE", "OPPONENT", "NETWORK", "LAP", "STYLE", "DISTANCE",
    "BACKFILL", "AUTO", "PEDIGREE", "ACTOR", "TIME_PACE",
]
CATEGORICAL = [
    "venue_code", "surface", "race_class", "discipline", "direction",
    "weather", "track_condition",
]


def parse_args():
    p = argparse.ArgumentParser(description="L2 king router V0: train on one year, frozen test on the next.")
    p.add_argument("--train-light", required=True)
    p.add_argument("--train-core", required=True)
    p.add_argument("--train-snapshot", required=True)
    p.add_argument("--test-light", required=True)
    p.add_argument("--test-core", required=True)
    p.add_argument("--test-snapshot", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--decisions-out", required=True)
    return p.parse_args()


def open_text(path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_expert(path):
    races = defaultdict(list)
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            races[str(row["race_id"])].append(row)
    for race_id, rows in races.items():
        rows.sort(key=lambda x: int(x["predicted_rank"]))
        if not rows or int(rows[0]["predicted_rank"]) != 1:
            raise ValueError(f"{race_id}: rank 1 missing")
    return races


def read_snapshot(path, wanted):
    races = {}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            race_id = str(row.get("race_id") or "")
            if race_id not in wanted:
                continue
            features = row.get("features") or {}
            target = row.get("target") or {}
            rec = races.setdefault(
                race_id,
                {
                    "winners": set(),
                    "race_date": str(features.get("race_date") or "")[:10],
                    "venue_code": features.get("venue_code"),
                    "surface": features.get("surface"),
                    "race_class": features.get("backfill_race_class_normalized"),
                    "discipline": features.get("discipline"),
                    "direction": features.get("direction"),
                    "weather": features.get("weather"),
                    "track_condition": features.get("track_condition"),
                    "distance_m": features.get("distance_m"),
                    "field_size": features.get("backfill_field_size"),
                },
            )
            if target.get("is_win") is True:
                rec["winners"].add(str(row.get("horse_id") or ""))
    missing = sorted(wanted - set(races))
    if missing:
        raise ValueError(f"snapshot missing {len(missing)} requested races")
    no_winner = [race_id for race_id, rec in races.items() if not rec["winners"]]
    if no_winner:
        raise ValueError(f"{len(no_winner)} requested races have no winner")
    return races


def as_float(value, default=0.0):
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def row_by_horse(rows):
    return {str(row["horse_id"]): row for row in rows}


def build_record(race_id, light_rows, core_rows, truth):
    light_top = light_rows[0]
    core_top = core_rows[0]
    light_pick = str(light_top["horse_id"])
    core_pick = str(core_top["horse_id"])
    light_map = row_by_horse(light_rows)
    core_map = row_by_horse(core_rows)
    if light_pick not in core_map or core_pick not in light_map:
        raise ValueError(f"{race_id}: cross-expert horse coverage differs")

    light_on_core = light_map[core_pick]
    core_on_light = core_map[light_pick]
    ls = light_top.get("race_summary") or {}
    cs = core_top.get("race_summary") or {}

    field_size = as_int(truth.get("field_size"), len(light_rows))
    rec = {
        "race_id": race_id,
        "race_date": truth.get("race_date"),
        "agreement": light_pick == core_pick,
        "light_pick": light_pick,
        "core_pick": core_pick,
        "venue_code": str(truth.get("venue_code") or "UNKNOWN"),
        "surface": str(truth.get("surface") or "UNKNOWN"),
        "race_class": str(truth.get("race_class") or "UNKNOWN"),
        "discipline": str(truth.get("discipline") or "UNKNOWN"),
        "direction": str(truth.get("direction") or "UNKNOWN"),
        "weather": str(truth.get("weather") or "UNKNOWN"),
        "track_condition": str(truth.get("track_condition") or "UNKNOWN"),
        "distance_m": as_float(truth.get("distance_m"), 0.0),
        "field_size": field_size,
        "light_top1_prob": as_float(light_top.get("race_normalized_win_probability")),
        "core_top1_prob": as_float(core_top.get("race_normalized_win_probability")),
        "light_raw_prob": as_float(light_top.get("raw_win_probability")),
        "core_raw_prob": as_float(core_top.get("raw_win_probability")),
        "light_gap": as_float(ls.get("top1_top2_gap")),
        "core_gap": as_float(cs.get("top1_top2_gap")),
        "light_entropy": as_float(ls.get("normalized_entropy")),
        "core_entropy": as_float(cs.get("normalized_entropy")),
        "light_top3_mass": as_float(ls.get("top3_probability_mass")),
        "core_top3_mass": as_float(cs.get("top3_probability_mass")),
        "core_prob_on_light_pick": as_float(core_on_light.get("race_normalized_win_probability")),
        "light_prob_on_core_pick": as_float(light_on_core.get("race_normalized_win_probability")),
        "core_rank_of_light_pick": as_int(core_on_light.get("predicted_rank")),
        "light_rank_of_core_pick": as_int(light_on_core.get("predicted_rank")),
    }
    rec["top1_prob_diff_core_minus_light"] = rec["core_top1_prob"] - rec["light_top1_prob"]
    rec["gap_diff_core_minus_light"] = rec["core_gap"] - rec["light_gap"]
    rec["entropy_diff_core_minus_light"] = rec["core_entropy"] - rec["light_entropy"]
    rec["light_pick_self_advantage"] = rec["light_top1_prob"] - rec["core_prob_on_light_pick"]
    rec["core_pick_self_advantage"] = rec["core_top1_prob"] - rec["light_prob_on_core_pick"]

    light_share = light_top.get("family_abs_share") or {}
    core_share = core_top.get("family_abs_share") or {}
    for family in FAMILIES:
        rec[f"light_share_{family}"] = as_float(light_share.get(family))
        rec[f"core_share_{family}"] = as_float(core_share.get(family))

    winners = truth["winners"]
    light_hit = light_pick in winners
    core_hit = core_pick in winners
    rec["light_hit"] = light_hit
    rec["core_hit"] = core_hit
    if rec["agreement"]:
        rec["target"] = "AGREE_CORRECT" if light_hit else "AGREE_WRONG"
    elif light_hit and core_hit:
        rec["target"] = "BOTH"
    elif light_hit:
        rec["target"] = "LIGHT"
    elif core_hit:
        rec["target"] = "CORE"
    else:
        rec["target"] = "NEITHER"
    return rec


def build_year(light_path, core_path, snapshot_path):
    light = read_expert(light_path)
    core = read_expert(core_path)
    if set(light) != set(core):
        raise ValueError("light/core race coverage differs")
    truth = read_snapshot(snapshot_path, set(light))
    rows = [
        build_record(race_id, light[race_id], core[race_id], truth[race_id])
        for race_id in sorted(light)
    ]
    return rows


def model_columns(frame):
    blocked = {
        "race_id", "race_date", "agreement", "light_pick", "core_pick",
        "light_hit", "core_hit", "target",
    }
    cols = [c for c in frame.columns if c not in blocked]
    categorical = [c for c in CATEGORICAL if c in cols]
    numeric = [c for c in cols if c not in categorical]
    return numeric, categorical


def baseline_metrics(rows, selector):
    disagreement = [r for r in rows if not r["agreement"]]
    hits = 0
    selected = 0
    skips = 0
    for r in disagreement:
        choice = selector(r)
        if choice == "NEITHER":
            skips += 1
            continue
        selected += 1
        if choice == "LIGHT":
            hits += int(r["light_hit"])
        elif choice == "CORE":
            hits += int(r["core_hit"])
        else:
            raise ValueError(f"bad baseline choice {choice}")
    return {
        "disagreement_races": len(disagreement),
        "selected": selected,
        "skipped": skips,
        "coverage": selected / len(disagreement) if disagreement else None,
        "hits": hits,
        "hit_rate_selected": hits / selected if selected else None,
        "hits_per_disagreement": hits / len(disagreement) if disagreement else None,
    }


def evaluate_router(test_rows, predictions, probabilities, classes):
    disagreement = [r for r in test_rows if not r["agreement"] and r["target"] != "BOTH"]
    if len(disagreement) != len(predictions):
        raise ValueError("prediction count mismatch")
    selected = hits = 0
    class_correct = 0
    decisions = []
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    for r, pred, probs in zip(disagreement, predictions, probabilities):
        pred = str(pred)
        true = str(r["target"])
        class_correct += int(pred == true)
        selected_hit = False
        if pred == "LIGHT":
            selected += 1
            selected_hit = bool(r["light_hit"])
        elif pred == "CORE":
            selected += 1
            selected_hit = bool(r["core_hit"])
        elif pred != "NEITHER":
            raise ValueError(f"unexpected router class {pred}")
        hits += int(selected_hit)
        decisions.append(
            {
                "race_id": r["race_id"],
                "race_date": r["race_date"],
                "true_class": true,
                "predicted_class": pred,
                "predicted_confidence": float(max(probs)),
                "prob_LIGHT": float(probs[class_to_idx["LIGHT"]]) if "LIGHT" in class_to_idx else 0.0,
                "prob_CORE": float(probs[class_to_idx["CORE"]]) if "CORE" in class_to_idx else 0.0,
                "prob_NEITHER": float(probs[class_to_idx["NEITHER"]]) if "NEITHER" in class_to_idx else 0.0,
                "selected_hit": selected_hit,
                "light_hit": bool(r["light_hit"]),
                "core_hit": bool(r["core_hit"]),
                "venue_code": r["venue_code"],
                "surface": r["surface"],
                "race_class": r["race_class"],
                "distance_m": r["distance_m"],
                "field_size": r["field_size"],
            }
        )
    agreement = [r for r in test_rows if r["agreement"]]
    agreement_hits = sum(int(r["light_hit"]) for r in agreement)
    return {
        "disagreement_races": len(disagreement),
        "class_accuracy": class_correct / len(disagreement) if disagreement else None,
        "selected": selected,
        "skipped": len(disagreement) - selected,
        "coverage": selected / len(disagreement) if disagreement else None,
        "selected_hits": hits,
        "hit_rate_selected": hits / selected if selected else None,
        "hits_per_disagreement": hits / len(disagreement) if disagreement else None,
        "agreement_races": len(agreement),
        "agreement_hits": agreement_hits,
        "agreement_top1_accuracy": agreement_hits / len(agreement) if agreement else None,
        "combined_hits": agreement_hits + hits,
        "combined_hits_per_all_races": (agreement_hits + hits) / len(test_rows) if test_rows else None,
    }, decisions


def main():
    a = parse_args()
    train_rows = build_year(a.train_light, a.train_core, a.train_snapshot)
    test_rows = build_year(a.test_light, a.test_core, a.test_snapshot)

    train_dis = [r for r in train_rows if not r["agreement"] and r["target"] != "BOTH"]
    test_dis = [r for r in test_rows if not r["agreement"] and r["target"] != "BOTH"]
    if len(train_dis) < 50 or len(test_dis) < 50:
        raise ValueError("not enough disagreement races for router prototype")

    train_df = pd.DataFrame(train_dis)
    test_df = pd.DataFrame(test_dis)
    numeric, categorical = model_columns(train_df)
    missing_test = sorted(set(numeric + categorical) - set(test_df.columns))
    if missing_test:
        raise ValueError(f"test columns missing: {missing_test}")

    x_train = train_df[numeric + categorical].copy()
    x_test = test_df[numeric + categorical].copy()
    y_train = train_df["target"].astype(str)
    y_test = test_df["target"].astype(str)

    expected = {"LIGHT", "CORE", "NEITHER"}
    if set(y_train) != expected:
        raise ValueError(f"router train classes must be {sorted(expected)}, got {sorted(set(y_train))}")

    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=3000,
        random_state=1945,
    )
    pipeline = Pipeline([("prep", prep), ("clf", clf)])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)
    classes = [str(x) for x in pipeline.named_steps["clf"].classes_]

    router_metrics, decisions = evaluate_router(test_rows, predictions, probabilities, classes)
    cm = confusion_matrix(y_test, predictions, labels=["LIGHT", "CORE", "NEITHER"])

    feature_names = pipeline.named_steps["prep"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_
    top_coefficients = {}
    for class_name, coef in zip(classes, coefs):
        pairs = sorted(
            [(str(name), float(value)) for name, value in zip(feature_names, coef)],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:20]
        top_coefficients[class_name] = [{"feature": n, "coefficient": v} for n, v in pairs]

    train_counts = Counter(r["target"] for r in train_dis)
    test_counts = Counter(r["target"] for r in test_dis)

    baselines = {
        "always_light": baseline_metrics(test_rows, lambda r: "LIGHT"),
        "always_core": baseline_metrics(test_rows, lambda r: "CORE"),
        "larger_top1_gap": baseline_metrics(
            test_rows,
            lambda r: "LIGHT" if r["light_gap"] > r["core_gap"] else "CORE",
        ),
        "higher_normalized_top1_probability": baseline_metrics(
            test_rows,
            lambda r: "LIGHT" if r["light_top1_prob"] > r["core_top1_prob"] else "CORE",
        ),
        "oracle_ceiling": baseline_metrics(
            test_rows,
            lambda r: "LIGHT" if r["light_hit"] else ("CORE" if r["core_hit"] else "NEITHER"),
        ),
    }

    summary = {
        "contract": "L2_KING_ROUTER_V0",
        "model": {
            "family": "multinomial_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "random_state": 1945,
            "sklearn_version": sklearn.__version__,
            "odds_used": False,
            "outcome_features_used": False,
            "router_classes": classes,
            "numeric_feature_count": len(numeric),
            "categorical_feature_count": len(categorical),
        },
        "train": {
            "races": len(train_rows),
            "agreement_races": sum(r["agreement"] for r in train_rows),
            "disagreement_races": len(train_dis),
            "target_counts": dict(train_counts),
        },
        "test": {
            "races": len(test_rows),
            "agreement_races": sum(r["agreement"] for r in test_rows),
            "disagreement_races": len(test_dis),
            "target_counts": dict(test_counts),
        },
        "router_test": router_metrics,
        "baselines_test": baselines,
        "confusion_matrix": {
            "labels": ["LIGHT", "CORE", "NEITHER"],
            "rows_true_cols_pred": cm.tolist(),
        },
        "top_coefficients_by_class": top_coefficients,
        "notes": [
            "Router is trained only on disagreement races; agreement races bypass it.",
            "NEITHER means abstain rather than force a king selection.",
            "2022 is used only as the frozen test set in this V0 run.",
            "This is a prototype result, not a production routing rule.",
        ],
    }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    decisions_out = Path(a.decisions_out)
    decisions_out.parent.mkdir(parents=True, exist_ok=True)
    write = gzip.open if str(decisions_out).endswith(".gz") else open
    with write(decisions_out, "wt", encoding="utf-8") as fh:
        for row in decisions:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print("L2_KING_ROUTER_V0_OK")
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
