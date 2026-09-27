#!/usr/bin/env python3
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "trainer-foundation-smoke"


def make_rows(forbidden=False, early_safe=False):
    rows = []
    for year, race_count in [(2024, 12), (2025, 4)]:
        for r in range(1, race_count + 1):
            date = f"{year}-01-{r:02d}"
            race_id = f"{year}050101{r:02d}"
            for h in range(1, 5):
                features = {
                    "race_date": date,
                    "venue_code": "05",
                    "surface": "TURF" if r % 2 else "DIRT",
                    "distance_m": 1200 + (r % 4) * 400,
                    "direction": "LEFT",
                    "weather": "晴",
                    "track_condition": "良",
                    "gate": h,
                    "horse_number": h,
                    "sex": "牡" if h % 2 else "牝",
                    "age": 3 + (h % 3),
                    "carried_weight": 55 + (h % 2),
                    "body_weight": 450 + h * 4,
                    "body_weight_diff": h - 2,
                    "prior_starts": r + h,
                    "recent_top3_rate": (5 - h) / 5,
                    "style_probe": h / 10,
                    "distx_probe": h * 100,
                }
                if early_safe:
                    for key in ("weather", "track_condition", "body_weight", "body_weight_diff"):
                        features.pop(key, None)
                if forbidden:
                    features["jockey_id"] = "J_FORBIDDEN"
                rows.append({
                    "ml_dataset_version": 3,
                    "feature_schema_version": 8,
                    "leakage_policy": "STRICT_PRIOR_DATE_ONLY",
                    "race_id": race_id,
                    "horse_id": f"H{year}{r:02d}{h:02d}",
                    "features": features,
                    "target": {
                        "is_win": h == 1,
                        "finish_position": h,
                    },
                    "market_outcome": {},
                })
    return rows


def write_dataset(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def trainer_cmd(
    dataset,
    stem,
    *,
    prediction_phase="FINAL",
    feature_sets=None,
    stage="base",
    feature_selection="none",
    train_start="2024-01-01",
    train_end="2024-12-31",
    valid_start="2025-01-01",
    valid_end="2025-12-31",
):
    cmd = [
        sys.executable,
        str(ROOT / "research" / "train-staged-lightgbm.py"),
        "--dataset", str(dataset),
        "--prediction-phase", prediction_phase,
        "--feature-selection", feature_selection,
        "--train-start", train_start,
        "--train-end", train_end,
        "--valid-start", valid_start,
        "--valid-end", valid_end,
        "--model-out", str(OUT / f"{stem}.txt"),
        "--meta-out", str(OUT / f"{stem}.json"),
        "--schema-out", str(OUT / f"{stem}-schema.json"),
        "--predictions-out", str(OUT / f"{stem}-oof.jsonl.gz"),
        "--diagnostics-out", str(OUT / f"{stem}-diagnostics.json.gz"),
        "--contributions-out", str(OUT / f"{stem}-contrib.jsonl.gz"),
        "--model-version", f"SMOKE_{stem.upper()}",
        "--source-sha", "BACKFILL_SMOKE_SHA",
        "--ml-source-sha", "ML_SMOKE_SHA",
    ]
    if feature_sets is not None:
        cmd.extend(["--feature-sets", feature_sets])
    if stage is not None:
        cmd.extend(["--stage", stage])
    return cmd


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)

    safe = OUT / "safe-dataset.jsonl.gz"
    write_dataset(safe, make_rows(False))
    subprocess.run(trainer_cmd(safe, "safe"), cwd=ROOT, check=True)

    meta = json.loads((OUT / "safe.json").read_text(encoding="utf-8"))
    assert len(meta["reproducibility"]["model_sha256"]) == 64
    assert len(meta["reproducibility"]["training_config_sha256"]) == 64
    assert len(meta["reproducibility"]["feature_catalog_sha256"]) == 64
    assert len(meta["reproducibility"]["feature_contract_sha256"]) == 64
    assert len(meta["reproducibility"]["small_sample_contract_sha256"]) == 64
    assert meta["small_sample_policy"]["ratePriorStrength"] == 20.0
    assert meta["small_sample_policy"]["meanPriorStrength"] == 10.0
    assert meta["small_sample_policy"]["minSpecificObservations"] == 5
    assert meta["small_sample_policy"]["actorRecentWindow"] == 30
    assert meta["prediction_phase"] == "FINAL"
    assert meta["feature_sets"] == ["BASE"]
    assert meta["history_windows"]["recent_form"] == 5
    assert meta["history_windows"]["style_last3f"] == 10
    assert "distx_probe" not in meta["features"]
    assert "style_probe" not in meta["features"]
    assert "surface" in meta["subgroup_metrics"]
    assert "distance_band" in meta["subgroup_metrics"]
    assert meta["diagnostics"]["contract"] == "L1_MODEL_DIAGNOSTICS_V1"

    with gzip.open(OUT / "safe-oof.jsonl.gz", "rt", encoding="utf-8") as fh:
        oof = json.loads(next(fh))
    assert oof["ml_dataset_version"] == 3
    assert oof["feature_schema_version"] == 8
    assert oof["leakage_policy"] == "STRICT_PRIOR_DATE_ONLY"
    assert oof["prediction_phase"] == "FINAL"
    assert oof["feature_sets"] == ["BASE"]
    assert oof["history_windows"]["suitability"] == 20
    assert "exact_feature_list" in oof
    assert len(oof["feature_catalog_sha256"]) == 64
    assert len(oof["feature_contract_sha256"]) == 64
    assert len(oof["small_sample_contract_sha256"]) == 64
    assert oof["small_sample_policy"]["actorRecentWindow"] == 30

    feature_set = OUT / "feature-set-dataset.jsonl.gz"
    write_dataset(feature_set, make_rows(False))
    subprocess.run(
        trainer_cmd(feature_set, "feature-set", feature_sets="BASE,DISTANCE", stage=None),
        cwd=ROOT,
        check=True,
    )
    feature_meta = json.loads((OUT / "feature-set.json").read_text(encoding="utf-8"))
    assert feature_meta["feature_sets"] == ["BASE", "DISTANCE"]
    assert "distx_probe" in feature_meta["features"]
    assert "style_probe" not in feature_meta["features"]

    selection_dataset = OUT / "selection-dataset.jsonl.gz"
    selection_rows = make_rows(False)
    for row in selection_rows:
        row["features"]["constant_probe"] = 1
        row["features"]["duplicate_distance_probe"] = row["features"]["distance_m"]
    write_dataset(selection_dataset, selection_rows)
    subprocess.run(
        trainer_cmd(
            selection_dataset,
            "selection",
            feature_sets="BASE",
            stage=None,
            feature_selection="train_v1",
        ),
        cwd=ROOT,
        check=True,
    )
    selection_meta = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    selection_report = selection_meta["feature_selection"]
    assert selection_report["contract"] == "L1_TRAIN_ONLY_FEATURE_SELECTION_V1"
    assert selection_report["mode"] == "train_v1"
    assert "constant_probe" in selection_report["dropped_constant"]
    assert "duplicate_distance_probe" in selection_report["dropped_correlation"]
    assert "constant_probe" not in selection_meta["features"]
    assert "duplicate_distance_probe" not in selection_meta["features"]

    early_safe = OUT / "early-safe-dataset.jsonl.gz"
    write_dataset(early_safe, make_rows(False, early_safe=True))
    subprocess.run(
        trainer_cmd(early_safe, "early-safe", prediction_phase="EARLY"),
        cwd=ROOT,
        check=True,
    )
    early_meta = json.loads((OUT / "early-safe.json").read_text(encoding="utf-8"))
    assert early_meta["prediction_phase"] == "EARLY"
    assert "body_weight" not in early_meta["features"]
    assert "weather" not in early_meta["features"]

    early_blocked = subprocess.run(
        trainer_cmd(safe, "early-blocked", prediction_phase="EARLY"),
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert early_blocked.returncode != 0
    early_text = (early_blocked.stdout or "") + "\n" + (early_blocked.stderr or "")
    assert "prediction phase EARLY blocked model columns" in early_text

    with gzip.open(OUT / "safe-diagnostics.json.gz", "rt", encoding="utf-8") as fh:
        diagnostics = json.loads(fh.read())
    assert diagnostics["contract"] == "L1_MODEL_DIAGNOSTICS_V1"
    assert "model_dump" in diagnostics
    assert "split_threshold_summary" in diagnostics
    assert "mean_abs_contribution" in diagnostics

    with gzip.open(OUT / "safe-contrib.jsonl.gz", "rt", encoding="utf-8") as fh:
        contrib = json.loads(next(fh))
    assert "race_id" in contrib
    assert "horse_id" in contrib
    assert len(contrib["top_contributions"]) <= 5

    forbidden = OUT / "forbidden-dataset.jsonl.gz"
    write_dataset(forbidden, make_rows(True))
    blocked = subprocess.run(
        trainer_cmd(forbidden, "forbidden"),
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert blocked.returncode != 0
    combined = (blocked.stdout or "") + "\n" + (blocked.stderr or "")
    assert "L1 feature catalog blocked dataset columns: jockey_id" in combined

    locked = subprocess.run(
        trainer_cmd(
            safe,
            "locked-2026",
            train_start="2025-01-01",
            train_end="2025-12-31",
            valid_start="2026-01-01",
            valid_end="2026-12-31",
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert locked.returncode != 0
    locked_text = (locked.stdout or "") + "\n" + (locked.stderr or "")
    assert "2026 research lock" in locked_text

    print("TRAINER_FOUNDATION_SMOKE_OK")
    shutil.rmtree(OUT, ignore_errors=True)


if __name__ == "__main__":
    main()
