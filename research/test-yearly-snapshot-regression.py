#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURE_CONTRACT = json.loads((ROOT / "contracts/l1-feature-set-contract-v1.json").read_text(encoding="utf-8"))
SMALL_SAMPLE_CONTRACT = json.loads((ROOT / "contracts/l1-small-sample-contract-v1.json").read_text(encoding="utf-8"))

EXPECTED = {
    "BASE": {
        "feature_sets": ["BASE"],
        "feature_count": 38,
        "metrics": {
            "races": 3456,
            "top1_winner_capture": 0.2630208333333333,
            "top3_winner_capture": 0.5526620370370371,
            "top6_winner_capture": 0.7792245370370371,
            "mean_winner_rank": 4.158854166666667,
            "mean_reciprocal_winner_rank": 0.46091680747330765,
            "race_normalized_nll": 2.183908938624418,
            "raw_logloss": 0.2293778015054907,
            "raw_brier": 0.06251387768953563,
            "raw_roc_auc": 0.760220066415816,
        },
    },
    "BASE+OPPONENT": {
        "feature_sets": ["BASE", "OPPONENT"],
        "feature_count": 64,
        "metrics": {
            "races": 3456,
            "top1_winner_capture": 0.2644675925925926,
            "top3_winner_capture": 0.5584490740740741,
            "top6_winner_capture": 0.7876157407407407,
            "mean_winner_rank": 4.09056712962963,
            "mean_reciprocal_winner_rank": 0.46331526770724585,
            "race_normalized_nll": 2.169321474631118,
            "raw_logloss": 0.22684056124864643,
            "raw_brier": 0.062012472332226835,
            "raw_roc_auc": 0.7693981110885828,
        },
    },
    "BASE+PEDIGREE": {
        "feature_sets": ["BASE", "PEDIGREE"],
        "feature_count": 242,
        "metrics": {
            "races": 3456,
            "top1_winner_capture": 0.25810185185185186,
            "top3_winner_capture": 0.5604745370370371,
            "top6_winner_capture": 0.7939814814814815,
            "mean_winner_rank": 4.0532407407407405,
            "mean_reciprocal_winner_rank": 0.4620011121057893,
            "race_normalized_nll": 2.1767316749261036,
            "raw_logloss": 0.22906119315704807,
            "raw_brier": 0.0625427351986856,
            "raw_roc_auc": 0.7626440925814618,
        },
    },
    "BASE+ACTOR": {
        "feature_sets": ["BASE", "ACTOR"],
        "feature_count": 217,
        "metrics": {
            "races": 3456,
            "top1_winner_capture": 0.2647569444444444,
            "top3_winner_capture": 0.5818865740740741,
            "top6_winner_capture": 0.8052662037037037,
            "mean_winner_rank": 3.9288194444444446,
            "mean_reciprocal_winner_rank": 0.47178686326779157,
            "race_normalized_nll": 2.1418297729481535,
            "raw_logloss": 0.2263612822936886,
            "raw_brier": 0.06208400403486521,
            "raw_roc_auc": 0.7729911072659497,
        },
    },
    "BASE+TIME_PACE": {
        "feature_sets": ["BASE", "TIME_PACE"],
        "feature_count": 61,
        "metrics": {
            "races": 3456,
            "top1_winner_capture": 0.2664930555555556,
            "top3_winner_capture": 0.5665509259259259,
            "top6_winner_capture": 0.7861689814814815,
            "mean_winner_rank": 4.1001157407407405,
            "mean_reciprocal_winner_rank": 0.46552391911166924,
            "race_normalized_nll": 2.1685526014042185,
            "raw_logloss": 0.22849452831436437,
            "raw_brier": 0.06234584113704314,
            "raw_roc_auc": 0.7633677518694822,
        },
    },
}

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-root", required=True)
    p.add_argument("--out-dir", default="out/snapshot-regression")
    p.add_argument("--source-sha", required=True)
    p.add_argument("--ml-source-sha", required=True)
    p.add_argument("--tolerance", type=float, default=1e-12)
    return p.parse_args()

def run(cmd):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True)

def find_generation(root):
    manifests = sorted(Path(root).glob("*/manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"expected exactly one snapshot manifest, found {len(manifests)}")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    return manifests[0].parent, manifest

def almost_equal(actual, expected, tolerance):
    if isinstance(expected, int):
        return int(actual) == expected
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)

def main():
    a = args()
    generation_dir, manifest = find_generation(a.snapshot_root)
    source = manifest.get("generation") or {}
    if source.get("source_backfill_sha") != a.source_sha:
        raise ValueError("snapshot BACKFILL SHA mismatch")
    if source.get("leakage_policy") != "STRICT_PRIOR_DATE_ONLY":
        raise ValueError("snapshot leakage policy mismatch")
    if source.get("prediction_phase") != "FINAL":
        raise ValueError("snapshot prediction phase mismatch")

    by_year = {int(row["year"]): generation_dir / row["file"] for row in manifest["years"]}
    inputs = [by_year[2023], by_year[2024], by_year[2025]]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history_windows = dict(FEATURE_CONTRACT["history_windows"])
    small = SMALL_SAMPLE_CONTRACT["defaults"]
    small_sample_policy = {
        "ratePriorStrength": float(small["rate_prior_strength"]),
        "meanPriorStrength": float(small["mean_prior_strength"]),
        "minSpecificObservations": int(small["min_specific_observations"]),
        "actorRecentWindow": int(history_windows["actor_recent"]),
    }

    results = []
    failed = False
    for label, expected in EXPECTED.items():
        slug = label.lower().replace("+", "-").replace("_", "-")
        candidate_dir = out / slug
        candidate_dir.mkdir(parents=True, exist_ok=True)
        dataset = candidate_dir / "dataset.jsonl.gz"
        model = candidate_dir / "model.txt"
        meta = candidate_dir / "meta.json"
        schema = candidate_dir / "schema.json"

        run([
            "node", "research/project-yearly-snapshots.mjs",
            "--inputs", ",".join(str(p) for p in inputs),
            "--output", str(dataset),
            "--feature-sets", ",".join(expected["feature_sets"]),
            "--prediction-phase", "FINAL",
        ])
        run([
            sys.executable, "research/train-staged-lightgbm.py",
            "--dataset", str(dataset),
            "--feature-sets", ",".join(expected["feature_sets"]),
            "--prediction-phase", "FINAL",
            "--history-windows-json", json.dumps(history_windows, separators=(",", ":")),
            "--small-sample-policy-json", json.dumps(small_sample_policy, separators=(",", ":")),
            "--feature-selection", "none",
            "--train-start", "2023-01-01",
            "--train-end", "2024-12-31",
            "--valid-start", "2025-01-01",
            "--valid-end", "2025-12-31",
            "--model-out", str(model),
            "--meta-out", str(meta),
            "--schema-out", str(schema),
            "--model-version", "L1_SNAPSHOT_REGRESSION_" + slug.upper(),
            "--source-repo", "pinogame1945-dotcom/KEIBA-BACKFILL",
            "--source-ref", a.source_sha,
            "--source-sha", a.source_sha,
            "--ml-source-sha", a.ml_source_sha,
        ])

        actual = json.loads(meta.read_text(encoding="utf-8"))
        diffs = {}
        if int(actual["feature_count"]) != int(expected["feature_count"]):
            diffs["feature_count"] = {
                "expected": expected["feature_count"],
                "actual": actual["feature_count"],
            }
        for key, exp in expected["metrics"].items():
            got = actual["metrics"].get(key)
            if got is None or not almost_equal(got, exp, a.tolerance):
                diffs[key] = {"expected": exp, "actual": got}

        passed = not diffs
        failed = failed or not passed
        results.append({
            "label": label,
            "passed": passed,
            "feature_sets": expected["feature_sets"],
            "feature_count": actual["feature_count"],
            "metrics": actual["metrics"],
            "diffs": diffs,
        })

        dataset.unlink(missing_ok=True)
        model.unlink(missing_ok=True)
        schema.unlink(missing_ok=True)

    summary = {
        "contract": "L1_YEARLY_SNAPSHOT_PILOT_REGRESSION_V1",
        "expected_pilot_run": 36298187997,
        "source_backfill_sha": a.source_sha,
        "snapshot_generation_id": source.get("generation_id"),
        "tolerance": a.tolerance,
        "all_passed": not failed,
        "candidates": results,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("L1_YEARLY_SNAPSHOT_REGRESSION_RESULT")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
