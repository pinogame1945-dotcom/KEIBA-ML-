#!/usr/bin/env python3
import argparse
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_CONTRACT_PATH = ROOT / "contracts" / "l1-feature-set-contract-v1.json"
DEFAULT_FEATURE_CONTRACT = json.loads(DEFAULT_FEATURE_CONTRACT_PATH.read_text(encoding="utf-8"))


def parse_args():
    p = argparse.ArgumentParser(description="Run bounded-memory L1 walk-forward folds.")
    p.add_argument("--source-root", required=True)
    p.add_argument("--out-dir", default="out/walk-forward")
    p.add_argument("--first-holdout", type=int, required=True)
    p.add_argument("--last-holdout", type=int, required=True)
    p.add_argument("--train-years", type=int, default=3)
    p.add_argument("--warmup-years", type=int, default=1)
    p.add_argument("--feature-contract", default=str(DEFAULT_FEATURE_CONTRACT_PATH))
    p.add_argument("--feature-sets", help="Comma-separated independent Feature Sets")
    p.add_argument(
        "--stage",
        choices=sorted(DEFAULT_FEATURE_CONTRACT["legacy_stage_map"]),
        help="Deprecated cumulative stage; translated to Feature Sets",
    )
    p.add_argument(
        "--prediction-phase",
        default=DEFAULT_FEATURE_CONTRACT["default_prediction_phase"],
        choices=sorted(DEFAULT_FEATURE_CONTRACT["prediction_phases"]),
    )
    p.add_argument("--history-limit", type=int, help="Deprecated: set every bounded history window to the same value")
    p.add_argument("--history-recent-form", type=int)
    p.add_argument("--history-style-last3f", type=int)
    p.add_argument("--history-suitability", type=int)
    p.add_argument("--history-opponent", type=int)
    p.add_argument("--history-auto-rolling", type=int)
    p.add_argument("--source-repo", default="pinogame1945-dotcom/KEIBA-BACKFILL")
    p.add_argument("--source-ref", default="main")
    p.add_argument("--source-sha")
    p.add_argument("--ml-source-sha")
    p.add_argument("--readiness-report")
    p.add_argument("--keep-datasets", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--min-core-known-coverage", type=float, default=0.98)
    p.add_argument("--max-invalid-rate", type=float, default=0.0)
    p.add_argument("--max-year-gap", type=float, default=0.10)
    return p.parse_args()


def load_feature_contract(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_feature_sets(a, contract):
    if a.feature_sets:
        requested = []
        for item in a.feature_sets.split(","):
            name = item.strip().upper()
            if name and name not in requested:
                requested.append(name)
    elif a.stage:
        requested = list(contract["legacy_stage_map"][a.stage])
    else:
        requested = list(contract["default_feature_sets"])

    valid = set(contract["feature_sets"])
    invalid = [name for name in requested if name not in valid]
    if invalid:
        raise ValueError("invalid feature set(s): " + ", ".join(invalid))
    if "BASE" not in requested:
        requested.insert(0, "BASE")

    if a.feature_sets and a.stage:
        legacy = list(contract["legacy_stage_map"][a.stage])
        if requested != legacy:
            raise ValueError("--feature-sets conflicts with deprecated --stage mapping")
    return requested


def resolve_history_windows(a, contract):
    out = dict(contract["history_windows"])
    bounded = list(contract["history_window_bounds"])

    if a.history_limit is not None:
        if not 1 <= a.history_limit <= 100:
            raise ValueError("--history-limit must be from 1 to 100")
        for key in bounded:
            out[key] = a.history_limit

    explicit = {
        "recent_form": a.history_recent_form,
        "style_last3f": a.history_style_last3f,
        "suitability": a.history_suitability,
        "opponent": a.history_opponent,
        "auto_rolling": a.history_auto_rolling,
    }
    for key, value in explicit.items():
        if value is None:
            continue
        lo, hi = contract["history_window_bounds"][key]
        if value < lo or value > hi:
            raise ValueError(f"{key} history window must be from {lo} to {hi}")
        out[key] = value
    return out


def fold_spec(year, train_years, warmup_years):
    train_first = year - train_years
    return {
        "holdout_year": year,
        "source_start": f"{train_first - warmup_years}-01-01",
        "source_end": f"{year}-12-31",
        "emit_start": f"{train_first}-01-01",
        "emit_end": f"{year}-12-31",
        "train_start": f"{train_first}-01-01",
        "train_end": f"{year - 1}-12-31",
        "valid_start": f"{year}-01-01",
        "valid_end": f"{year}-12-31",
    }


def run(cmd):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def readiness_command(a, folds, report_path):
    source_start = min(fold["source_start"] for fold in folds)
    source_end = max(fold["source_end"] for fold in folds)
    cmd = [
        "node", "scripts/audit-backfill-readiness.mjs",
        "--source-root", a.source_root,
        "--start", source_start,
        "--end", source_end,
        "--report-out", report_path,
        "--min-core-known-coverage", a.min_core_known_coverage,
        "--max-invalid-rate", a.max_invalid_rate,
        "--max-year-gap", a.max_year_gap,
        "--require-source-integrity",
    ]
    if a.source_sha:
        cmd.extend(["--source-sha", a.source_sha])
    return cmd


def validate_readiness_report(path, a, folds):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_start = min(fold["source_start"] for fold in folds)
    expected_end = max(fold["source_end"] for fold in folds)
    source = report.get("source") or {}
    thresholds = report.get("thresholds") or {}
    checks = [
        (report.get("report_version") == "BACKFILL_READINESS_V3", "readiness version mismatch"),
        (report.get("source_integrity", {}).get("passed") is True, "source integrity not verified"),
        (report.get("ready_for_l1_research") is True, "report is not ready"),
        (source.get("start") == expected_start, "readiness start mismatch"),
        (source.get("end") == expected_end, "readiness end mismatch"),
        (thresholds.get("min_core_known_coverage") == a.min_core_known_coverage, "coverage threshold mismatch"),
        (thresholds.get("max_invalid_rate") == a.max_invalid_rate, "invalid threshold mismatch"),
        (thresholds.get("max_year_gap") == a.max_year_gap, "year-gap threshold mismatch"),
    ]
    if a.source_sha:
        checks.append((source.get("sha") == a.source_sha, "BACKFILL SHA mismatch"))
    failed = [message for ok, message in checks if not ok]
    if failed:
        raise ValueError("invalid readiness report: " + "; ".join(failed))
    return report


def append_history_args(cmd, windows):
    mapping = {
        "recent_form": "--history-recent-form",
        "style_last3f": "--history-style-last3f",
        "suitability": "--history-suitability",
        "opponent": "--history-opponent",
        "auto_rolling": "--history-auto-rolling",
    }
    for key, flag in mapping.items():
        cmd.extend([flag, windows[key]])


def main():
    a = parse_args()
    contract = load_feature_contract(a.feature_contract)
    feature_sets = resolve_feature_sets(a, contract)
    prediction_phase = str(a.prediction_phase).upper()
    history_windows = resolve_history_windows(a, contract)

    if a.first_holdout > a.last_holdout:
        raise ValueError("first-holdout must be <= last-holdout")
    if a.train_years < 1 or a.warmup_years < 0:
        raise ValueError("invalid train/warmup years")
    if not (0 <= a.min_core_known_coverage <= 1):
        raise ValueError("min-core-known-coverage must be from 0 to 1")
    if not (0 <= a.max_invalid_rate <= 1):
        raise ValueError("max-invalid-rate must be from 0 to 1")
    if not (0 <= a.max_year_gap <= 1):
        raise ValueError("max-year-gap must be from 0 to 1")

    folds = [
        fold_spec(y, a.train_years, a.warmup_years)
        for y in range(a.first_holdout, a.last_holdout + 1)
    ]
    experiment_tag = "-".join(name.lower() for name in feature_sets)
    print("L1_WALK_FORWARD_PLAN")
    print(json.dumps({
        "prediction_phase": prediction_phase,
        "feature_sets": feature_sets,
        "history_windows": history_windows,
        "legacy_stage": a.stage,
        "train_years": a.train_years,
        "warmup_years": a.warmup_years,
        "folds": folds,
    }, indent=2))
    if a.plan_only:
        return

    root = Path(a.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    readiness_report = root / "backfill-readiness.json"

    print("L1_BACKFILL_READINESS_GATE", flush=True)
    if a.readiness_report:
        try:
            report = validate_readiness_report(a.readiness_report, a, folds)
            readiness_report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as error:
            print(
                f"L1_BACKFILL_READINESS_BLOCKED: {error}. Training was not started.",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(2) from error
    else:
        try:
            run(readiness_command(a, folds, readiness_report))
        except subprocess.CalledProcessError as error:
            print(
                "L1_BACKFILL_READINESS_BLOCKED: BACKFILL is not ready. "
                f"See {readiness_report}. Training was not started.",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(error.returncode) from error

    print("L1_BACKFILL_READINESS_PASS", flush=True)

    datasets = root / "tmp-datasets"
    models = root / "models"
    oof_dir = root / "oof"
    schemas = root / "schemas"
    diagnostics = root / "diagnostics"
    for p in (datasets, models, oof_dir, schemas, diagnostics):
        p.mkdir(parents=True, exist_ok=True)

    summaries = []
    oof_paths = []

    for fold in folds:
        year = fold["holdout_year"]
        dataset = datasets / f"fold-{year}.jsonl.gz"
        model = models / f"l1-{experiment_tag}-{prediction_phase.lower()}-{year}.txt"
        meta = models / f"l1-{experiment_tag}-{prediction_phase.lower()}-{year}.json"
        schema = schemas / f"l1-{experiment_tag}-{prediction_phase.lower()}-{year}-schema.json"
        pred = oof_dir / f"oof-{year}.jsonl.gz"
        diagnostic = diagnostics / f"l1-{experiment_tag}-{prediction_phase.lower()}-{year}-diagnostics.json.gz"
        contributions = diagnostics / f"l1-{experiment_tag}-{prediction_phase.lower()}-{year}-contributions.jsonl.gz"

        dataset_cmd = [
            "node", "research/build-staged-dataset.mjs",
            "--source-root", a.source_root,
            "--output", dataset,
            "--source-start", fold["source_start"],
            "--source-end", fold["source_end"],
            "--emit-start", fold["emit_start"],
            "--emit-end", fold["emit_end"],
            "--feature-sets", ",".join(feature_sets),
            "--prediction-phase", prediction_phase,
        ]
        append_history_args(dataset_cmd, history_windows)
        run(dataset_cmd)

        cmd = [
            sys.executable, "research/train-staged-lightgbm.py",
            "--dataset", dataset,
            "--feature-sets", ",".join(feature_sets),
            "--prediction-phase", prediction_phase,
            "--history-windows-json", json.dumps(history_windows, separators=(",", ":")),
            "--feature-contract", a.feature_contract,
            "--train-start", fold["train_start"],
            "--train-end", fold["train_end"],
            "--valid-start", fold["valid_start"],
            "--valid-end", fold["valid_end"],
            "--model-out", model,
            "--meta-out", meta,
            "--schema-out", schema,
            "--predictions-out", pred,
            "--diagnostics-out", diagnostic,
            "--contributions-out", contributions,
            "--model-version", f"L1_{experiment_tag.upper()}_{prediction_phase}_WF_{year}",
            "--source-repo", a.source_repo,
            "--source-ref", a.source_ref,
        ]
        if a.source_sha:
            cmd.extend(["--source-sha", a.source_sha])
        if a.ml_source_sha:
            cmd.extend(["--ml-source-sha", a.ml_source_sha])
        run(cmd)

        metadata = json.loads(meta.read_text(encoding="utf-8"))
        summaries.append({
            "holdout_year": year,
            "model_version": metadata["model_version"],
            "prediction_phase": metadata["prediction_phase"],
            "feature_sets": metadata["feature_sets"],
            "history_windows": metadata["history_windows"],
            "split": metadata["split"],
            "metrics": metadata["metrics"],
            "feature_count": metadata["feature_count"],
            "reproducibility": metadata.get("reproducibility"),
            "subgroup_metrics": metadata.get("subgroup_metrics"),
            "diagnostics": metadata.get("diagnostics"),
        })
        oof_paths.append(pred)

        if not a.keep_datasets:
            dataset.unlink(missing_ok=True)

    combined = oof_dir / "l1-oof-all.jsonl.gz"
    with gzip.open(combined, "wt", encoding="utf-8") as dst:
        for p in oof_paths:
            with gzip.open(p, "rt", encoding="utf-8") as src:
                shutil.copyfileobj(src, dst)

    summary = {
        "contract": "L1_WALK_FORWARD_V2",
        "prediction_phase": prediction_phase,
        "feature_sets": feature_sets,
        "history_windows": history_windows,
        "legacy_stage": a.stage,
        "source": {
            "repository": a.source_repo,
            "ref": a.source_ref,
            "sha": a.source_sha,
        },
        "train_years": a.train_years,
        "warmup_years": a.warmup_years,
        "readiness_gate": {
            "report": str(readiness_report),
            "min_core_known_coverage": a.min_core_known_coverage,
            "max_invalid_rate": a.max_invalid_rate,
            "max_year_gap": a.max_year_gap,
        },
        "folds": summaries,
        "combined_oof": str(combined),
    }
    (root / "walk-forward-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("L1_WALK_FORWARD_DONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
