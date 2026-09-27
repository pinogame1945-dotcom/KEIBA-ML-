#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE3_CONTRACT_PATH = ROOT / "contracts" / "l1-phase3-research-v1.json"
FEATURE_CONTRACT_PATH = ROOT / "contracts" / "l1-feature-set-contract-v1.json"
SMALL_SAMPLE_CONTRACT_PATH = ROOT / "contracts" / "l1-small-sample-contract-v1.json"

PHASE3 = json.loads(PHASE3_CONTRACT_PATH.read_text(encoding="utf-8"))
FEATURES = json.loads(FEATURE_CONTRACT_PATH.read_text(encoding="utf-8"))
SMALL = json.loads(SMALL_SAMPLE_CONTRACT_PATH.read_text(encoding="utf-8"))
LOCKED_YEAR = int(PHASE3["locked_year"])

WINDOW_REQUIREMENTS = {
    "recent_form": {"BASE"},
    "style_last3f": {"LAP", "STYLE"},
    "suitability": {"BASE"},
    "opponent": {"OPPONENT"},
    "auto_rolling": {"AUTO"},
    "actor_recent": {"ACTOR"},
    "time_pace": {"TIME_PACE"},
}


def parse_args():
    p = argparse.ArgumentParser(description="L1 Phase 3 research/selection matrix runner.")
    p.add_argument("--source-root", required=True)
    p.add_argument("--out-dir", default="out/phase3")
    p.add_argument("--first-holdout", type=int, default=2025)
    p.add_argument("--last-holdout", type=int, default=2025)
    p.add_argument("--train-years", type=int, default=3)
    p.add_argument("--warmup-years", type=int, default=1)
    p.add_argument("--prediction-phase", choices=["EARLY", "FINAL"], default="FINAL")
    p.add_argument(
        "--matrix",
        choices=["base_plus_one", "windows", "shrinkage", "feature_selection", "all"],
        default="base_plus_one",
    )
    p.add_argument(
        "--families",
        default=",".join(PHASE3["base_plus_one_families"]),
        help="Families used by BASE+1 matrix",
    )
    p.add_argument(
        "--reference-feature-sets",
        default="BASE,OPPONENT,NETWORK,LAP,STYLE,DISTANCE,PEDIGREE,ACTOR,TIME_PACE",
        help="Feature Sets used by window/shrinkage/feature-selection matrices",
    )
    p.add_argument("--window-grid-json", help="Override Phase 3 window grid JSON object")
    p.add_argument("--rate-strengths", help="Comma-separated rate shrinkage strengths")
    p.add_argument("--mean-strengths", help="Comma-separated mean shrinkage strengths")
    p.add_argument("--min-specific-grid", help="Comma-separated minimum specific observation counts")
    p.add_argument("--source-repo", default="pinogame1945-dotcom/KEIBA-BACKFILL")
    p.add_argument("--source-ref", default="main")
    p.add_argument("--source-sha")
    p.add_argument("--ml-source-sha")
    p.add_argument("--readiness-report")
    p.add_argument("--min-core-known-coverage", type=float, default=0.98)
    p.add_argument("--max-invalid-rate", type=float, default=0.0)
    p.add_argument("--max-year-gap", type=float, default=0.10)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max-execute-candidates", type=int, default=24)
    return p.parse_args()


def canonical_feature_sets(raw):
    requested = {x.strip().upper() for x in str(raw).split(",") if x.strip()}
    requested.add("BASE")
    order = list(FEATURES["feature_sets"])
    invalid = sorted(requested - set(order))
    if invalid:
        raise ValueError("invalid Feature Set(s): " + ", ".join(invalid))
    return [name for name in order if name in requested]


def csv_numbers(raw, default, *, integer=False):
    if not raw:
        return list(default)
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item) if integer else float(item)
        if value not in values:
            values.append(value)
    return values


def stable_id(payload):
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def defaults():
    windows = deepcopy(FEATURES["history_windows"])
    small = {
        "ratePriorStrength": float(SMALL["defaults"]["rate_prior_strength"]),
        "meanPriorStrength": float(SMALL["defaults"]["mean_prior_strength"]),
        "minSpecificObservations": int(SMALL["defaults"]["min_specific_observations"]),
    }
    return windows, small


def candidate(kind, label, feature_sets, windows, small_sample, feature_selection="none"):
    config = {
        "kind": kind,
        "label": label,
        "feature_sets": list(feature_sets),
        "history_windows": deepcopy(windows),
        "small_sample_policy": deepcopy(small_sample),
        "feature_selection": feature_selection,
    }
    config["candidate_id"] = stable_id(config)
    return config


def build_candidates(a):
    base_windows, base_small = defaults()
    candidates = []

    matrices = (
        ["base_plus_one", "windows", "shrinkage", "feature_selection"]
        if a.matrix == "all"
        else [a.matrix]
    )

    if "base_plus_one" in matrices:
        candidates.append(candidate(
            "base_plus_one", "BASE", ["BASE"], base_windows, base_small
        ))
        valid = set(PHASE3["base_plus_one_families"])
        requested = [x.strip().upper() for x in a.families.split(",") if x.strip()]
        invalid = sorted(set(requested) - valid)
        if invalid:
            raise ValueError("invalid BASE+1 family: " + ", ".join(invalid))
        for family in requested:
            candidates.append(candidate(
                "base_plus_one",
                f"BASE+{family}",
                canonical_feature_sets(f"BASE,{family}"),
                base_windows,
                base_small,
            ))

    reference = canonical_feature_sets(a.reference_feature_sets)
    reference_set = set(reference)

    if "windows" in matrices:
        grid = json.loads(a.window_grid_json) if a.window_grid_json else PHASE3["window_grid"]
        if not isinstance(grid, dict):
            raise ValueError("--window-grid-json must be a JSON object")
        for key, values in grid.items():
            if key not in FEATURES["history_window_bounds"]:
                raise ValueError("unknown window grid key: " + str(key))
            required = WINDOW_REQUIREMENTS.get(key, {"BASE"})
            if key not in {"recent_form", "suitability"} and not (reference_set & required):
                continue
            for value in values:
                n = int(value)
                lo, hi = FEATURES["history_window_bounds"][key]
                if n < lo or n > hi:
                    raise ValueError(f"{key} candidate {n} outside {lo}..{hi}")
                windows = deepcopy(base_windows)
                windows[key] = n
                candidates.append(candidate(
                    "windows",
                    f"{key}={n}",
                    reference,
                    windows,
                    base_small,
                ))

    if "shrinkage" in matrices:
        if not ({"PEDIGREE", "ACTOR"} & reference_set):
            raise ValueError("shrinkage matrix requires PEDIGREE and/or ACTOR in --reference-feature-sets")
        for value in csv_numbers(a.rate_strengths, PHASE3["shrinkage_grid"]["ratePriorStrength"]):
            small = deepcopy(base_small)
            small["ratePriorStrength"] = float(value)
            candidates.append(candidate(
                "shrinkage",
                f"ratePriorStrength={value}",
                reference,
                base_windows,
                small,
            ))
        for value in csv_numbers(a.mean_strengths, PHASE3["shrinkage_grid"]["meanPriorStrength"]):
            small = deepcopy(base_small)
            small["meanPriorStrength"] = float(value)
            candidates.append(candidate(
                "shrinkage",
                f"meanPriorStrength={value}",
                reference,
                base_windows,
                small,
            ))
        for value in csv_numbers(
            a.min_specific_grid,
            PHASE3["shrinkage_grid"]["minSpecificObservations"],
            integer=True,
        ):
            small = deepcopy(base_small)
            small["minSpecificObservations"] = int(value)
            candidates.append(candidate(
                "shrinkage",
                f"minSpecificObservations={value}",
                reference,
                base_windows,
                small,
            ))

    if "feature_selection" in matrices:
        for mode in PHASE3["feature_selection_modes"]:
            candidates.append(candidate(
                "feature_selection",
                f"feature_selection={mode}",
                reference,
                base_windows,
                base_small,
                feature_selection=mode,
            ))

    unique = []
    seen = set()
    for item in candidates:
        identity = json.dumps({
            "feature_sets": item["feature_sets"],
            "history_windows": item["history_windows"],
            "small_sample_policy": item["small_sample_policy"],
            "feature_selection": item["feature_selection"],
        }, sort_keys=True, separators=(",", ":"))
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
    return unique


def assert_2026_lock(a):
    if a.first_holdout > a.last_holdout:
        raise ValueError("first-holdout must be <= last-holdout")
    if a.last_holdout >= LOCKED_YEAR:
        raise ValueError(
            f"{LOCKED_YEAR} research lock: Phase 3 cannot inspect holdout {a.last_holdout}"
        )
    if a.train_years < 1 or a.warmup_years < 0:
        raise ValueError("invalid train/warmup years")


def source_range(a):
    start_year = a.first_holdout - a.train_years - a.warmup_years
    return f"{start_year}-01-01", f"{a.last_holdout}-12-31"


def run(cmd):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def prepare_readiness(a, root):
    if a.readiness_report:
        return Path(a.readiness_report)
    start, end = source_range(a)
    report = root / "backfill-readiness.json"
    cmd = [
        "node", "scripts/audit-backfill-readiness.mjs",
        "--source-root", a.source_root,
        "--start", start,
        "--end", end,
        "--report-out", report,
        "--min-core-known-coverage", a.min_core_known_coverage,
        "--max-invalid-rate", a.max_invalid_rate,
        "--max-year-gap", a.max_year_gap,
        "--require-source-integrity",
    ]
    if a.source_sha:
        cmd.extend(["--source-sha", a.source_sha])
    run(cmd)
    return report


def walk_forward_cmd(a, item, candidate_dir, readiness_report):
    windows = item["history_windows"]
    small = item["small_sample_policy"]
    return [
        sys.executable,
        "research/run_walk_forward.py",
        "--source-root", a.source_root,
        "--out-dir", candidate_dir,
        "--first-holdout", a.first_holdout,
        "--last-holdout", a.last_holdout,
        "--train-years", a.train_years,
        "--warmup-years", a.warmup_years,
        "--prediction-phase", a.prediction_phase,
        "--feature-sets", ",".join(item["feature_sets"]),
        "--history-recent-form", windows["recent_form"],
        "--history-style-last3f", windows["style_last3f"],
        "--history-suitability", windows["suitability"],
        "--history-opponent", windows["opponent"],
        "--history-auto-rolling", windows["auto_rolling"],
        "--history-actor-recent", windows["actor_recent"],
        "--history-time-pace", windows["time_pace"],
        "--shrinkage-rate-strength", small["ratePriorStrength"],
        "--shrinkage-mean-strength", small["meanPriorStrength"],
        "--min-specific-observations", small["minSpecificObservations"],
        "--feature-selection", item["feature_selection"],
        "--source-repo", a.source_repo,
        "--source-ref", a.source_ref,
        "--readiness-report", readiness_report,
    ] + (
        ["--source-sha", a.source_sha] if a.source_sha else []
    ) + (
        ["--ml-source-sha", a.ml_source_sha] if a.ml_source_sha else []
    )


def weighted_mean(rows, key, weight_key="races"):
    pairs = []
    for row in rows:
        value = row.get(key)
        weight = row.get(weight_key)
        if value is None or weight is None or weight <= 0:
            continue
        pairs.append((float(value), float(weight)))
    if not pairs:
        return None
    total = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total


def compact_candidate_result(item, summary):
    folds = []
    metric_rows = []
    for fold in summary.get("folds", []):
        metrics = fold.get("metrics") or {}
        metric_rows.append(metrics)
        fs = fold.get("feature_selection") or {}
        inner = dict(fs.get("inner_split") or {})
        inner.pop("gain_fraction", None)
        folds.append({
            "holdout_year": fold.get("holdout_year"),
            "metrics": metrics,
            "feature_count": fold.get("feature_count"),
            "features": fold.get("features") or [],
            "feature_selection": {
                "mode": fs.get("mode"),
                "input_feature_count": len(fs.get("input_features") or []),
                "selected_feature_count": len(fs.get("selected_features") or []),
                "dropped_missing": fs.get("dropped_missing") or [],
                "dropped_constant": fs.get("dropped_constant") or [],
                "dropped_correlation": fs.get("dropped_correlation") or [],
                "dropped_inner_gain": fs.get("dropped_inner_gain") or [],
                "inner_gain_abstained": bool(fs.get("inner_gain_abstained", False)),
                "inner_gain_abstain_reason": fs.get("inner_gain_abstain_reason"),
                "inner_split": inner or None,
                "parameters": fs.get("parameters") or {},
            },
            "subgroup_metrics": fold.get("subgroup_metrics"),
            "reproducibility": fold.get("reproducibility"),
        })
    aggregate = {}
    for key in [
        "top1_winner_capture",
        "top3_winner_capture",
        "top6_winner_capture",
        "mean_winner_rank",
        "mean_reciprocal_winner_rank",
        "race_normalized_nll",
        "raw_logloss",
        "raw_brier",
        "raw_roc_auc",
    ]:
        aggregate[key] = weighted_mean(metric_rows, key)
    aggregate["races"] = sum(int(row.get("races") or 0) for row in metric_rows)
    return {
        **item,
        "aggregate_metrics": aggregate,
        "folds": folds,
    }


def add_baseline_deltas(results):
    baselines = [
        row for row in results
        if row["kind"] == "base_plus_one" and row["feature_sets"] == ["BASE"]
    ]
    if not baselines:
        return
    baseline = baselines[0]["aggregate_metrics"]
    for row in results:
        deltas = {}
        for key, base_value in baseline.items():
            value = row["aggregate_metrics"].get(key)
            if key == "races" or base_value is None or value is None:
                continue
            deltas[key] = value - base_value
        row["delta_vs_base"] = deltas


def main():
    a = parse_args()
    assert_2026_lock(a)
    candidates = build_candidates(a)
    plan = {
        "contract": PHASE3["contract"],
        "locked_year": LOCKED_YEAR,
        "prediction_phase": a.prediction_phase,
        "matrix": a.matrix,
        "first_holdout": a.first_holdout,
        "last_holdout": a.last_holdout,
        "train_years": a.train_years,
        "warmup_years": a.warmup_years,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    print("L1_PHASE3_RESEARCH_PLAN")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not a.execute:
        return

    if a.max_execute_candidates < 1:
        raise ValueError("--max-execute-candidates must be >= 1")
    if len(candidates) > a.max_execute_candidates:
        raise ValueError(
            f"candidate count {len(candidates)} exceeds --max-execute-candidates={a.max_execute_candidates}; "
            "split the matrix or explicitly raise the limit"
        )
    if not a.source_sha:
        raise ValueError("--source-sha is required for Phase 3 execution reproducibility")

    root = Path(a.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    readiness = prepare_readiness(a, root)
    results = []
    for index, item in enumerate(candidates, start=1):
        candidate_dir = root / "candidates" / f"{index:03d}-{item['candidate_id']}"
        print(
            f"L1_PHASE3_CANDIDATE_START {index}/{len(candidates)} "
            f"{item['candidate_id']} {item['label']}",
            flush=True,
        )
        run(walk_forward_cmd(a, item, candidate_dir, readiness))
        summary_path = candidate_dir / "walk-forward-summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results.append(compact_candidate_result(item, summary))

    add_baseline_deltas(results)
    primary = PHASE3["primary_comparison_metric"]
    comparable = [
        row for row in results
        if row["aggregate_metrics"].get(primary) is not None
    ]
    order = [
        row["candidate_id"]
        for row in sorted(
            comparable,
            key=lambda row: (
                row["aggregate_metrics"][primary],
                row["candidate_id"],
            ),
        )
    ]
    output = {
        "contract": PHASE3["summary_contract"]["name"],
        "research_contract": PHASE3["contract"],
        "locked_year": LOCKED_YEAR,
        "production_promotion_allowed": False,
        "selection_statement": (
            "This summary compares research candidates only. "
            "It does not promote a production L1 model and it does not inspect 2026."
        ),
        "source": {
            "repository": a.source_repo,
            "ref": a.source_ref,
            "sha": a.source_sha,
            "ml_source_sha": a.ml_source_sha,
        },
        "prediction_phase": a.prediction_phase,
        "matrix": a.matrix,
        "primary_comparison_metric": primary,
        "secondary_metrics": PHASE3["secondary_metrics"],
        "ordered_candidate_ids_by_primary_metric": order,
        "candidate_count": len(results),
        "candidates": results,
    }
    output["summary_sha256"] = hashlib.sha256(
        json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path = root / "research-summary.json"
    payload = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    size = len(payload.encode("utf-8"))
    max_bytes = int(PHASE3["summary_contract"]["max_bytes"])
    if size > max_bytes:
        raise ValueError(
            f"compact research summary is {size} bytes, exceeding contract max {max_bytes}"
        )
    path.write_text(payload, encoding="utf-8")
    print("L1_PHASE3_RESEARCH_DONE")
    print(json.dumps({
        "summary": str(path),
        "candidate_count": len(results),
        "summary_sha256": output["summary_sha256"],
        "persistent_model_or_oof_upload": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
