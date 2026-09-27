#!/usr/bin/env python3
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "trainer-foundation-smoke"


def make_rows(forbidden=False):
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
                }
                if forbidden:
                    features["jockey_id"] = "J_FORBIDDEN"
                rows.append({
                    "ml_dataset_version": 3,
                    "feature_schema_version": 7,
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


def trainer_cmd(dataset, stem):
    return [
        sys.executable,
        str(ROOT / "research" / "train-staged-lightgbm.py"),
        "--dataset", str(dataset),
        "--stage", "base",
        "--train-start", "2024-01-01",
        "--train-end", "2024-12-31",
        "--valid-start", "2025-01-01",
        "--valid-end", "2025-12-31",
        "--model-out", str(OUT / f"{stem}.txt"),
        "--meta-out", str(OUT / f"{stem}.json"),
        "--schema-out", str(OUT / f"{stem}-schema.json"),
        "--predictions-out", str(OUT / f"{stem}-oof.jsonl.gz"),
        "--diagnostics-out", str(OUT / f"{stem}-diagnostics.json.gz"),
        "--model-version", f"SMOKE_{stem.upper()}",
        "--source-sha", "BACKFILL_SMOKE_SHA",
        "--ml-source-sha", "ML_SMOKE_SHA",
    ]


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
    assert "surface" in meta["subgroup_metrics"]
    assert "distance_band" in meta["subgroup_metrics"]
    assert meta["diagnostics"]["contract"] == "L1_MODEL_DIAGNOSTICS_V1"

    with gzip.open(OUT / "safe-oof.jsonl.gz", "rt", encoding="utf-8") as fh:
        oof = json.loads(next(fh))
    assert oof["ml_dataset_version"] == 3
    assert oof["feature_schema_version"] == 7
    assert oof["leakage_policy"] == "STRICT_PRIOR_DATE_ONLY"

    with gzip.open(OUT / "safe-diagnostics.json.gz", "rt", encoding="utf-8") as fh:
        diagnostics = json.loads(fh.read())
    assert diagnostics["contract"] == "L1_MODEL_DIAGNOSTICS_V1"
    assert "model_dump" in diagnostics
    assert "split_threshold_summary" in diagnostics
    assert "mean_abs_contribution" in diagnostics

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
    assert "L1 feature catalog blocked model columns: jockey_id" in combined

    print("TRAINER_FOUNDATION_SMOKE_OK")
    shutil.rmtree(OUT, ignore_errors=True)


if __name__ == "__main__":
    main()
