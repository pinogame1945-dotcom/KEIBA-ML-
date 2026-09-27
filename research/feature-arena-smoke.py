#!/usr/bin/env python3
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "feature-arena-smoke"
ALL_SETS = [
    "BASE", "OPPONENT", "NETWORK", "LAP", "STYLE", "DISTANCE",
    "BACKFILL", "AUTO", "PEDIGREE", "ACTOR", "TIME_PACE",
]
HISTORY_WINDOWS = {
    "recent_form": 5,
    "style_last3f": 10,
    "suitability": 20,
    "opponent": 10,
    "auto_rolling": 20,
    "career": "ALL",
    "elo": "ALL",
    "actor_recent": 30,
    "time_pace": 10,
}
SMALL_SAMPLE = {
    "ratePriorStrength": 20,
    "meanPriorStrength": 10,
    "minSpecificObservations": 5,
    "actorRecentWindow": 30,
}


def make_rows():
    rows = []
    for year, races in [(2024, 18), (2025, 6)]:
        for race_no in range(1, races + 1):
            date = f"{year}-01-{race_no:02d}"
            race_id = f"{year}050101{race_no:02d}"
            for horse_no in range(1, 5):
                quality = 5 - horse_no
                features = {
                    "race_date": date,
                    "venue_code": "05",
                    "surface": "TURF" if race_no % 2 else "DIRT",
                    "distance_m": 1200 + (race_no % 4) * 400,
                    "direction": "LEFT",
                    "weather": "晴",
                    "track_condition": "良",
                    "gate": horse_no,
                    "horse_number": horse_no,
                    "sex": "牡" if horse_no % 2 else "牝",
                    "age": 3 + (horse_no % 3),
                    "carried_weight": 55 + (horse_no % 2),
                    "body_weight": 450 + horse_no * 4,
                    "body_weight_diff": horse_no - 2,
                    "prior_starts": race_no + horse_no,
                    "recent_top3_rate": quality / 5,
                    "opponent_probe": quality * 0.10,
                    "network_probe": quality * 0.11,
                    "lap_probe": quality * 0.12,
                    "style_probe": quality * 0.13,
                    "distx_probe": quality * 10,
                    "backfill_probe": quality * 0.14,
                    "auto_probe": quality * 0.15,
                    "ped_probe": quality * 0.16,
                    "actor_probe": quality * 0.17,
                    "timepace_probe": quality * 0.18,
                }
                rows.append({
                    "ml_dataset_version": 3,
                    "feature_schema_version": 8,
                    "leakage_policy": "STRICT_PRIOR_DATE_ONLY",
                    "race_id": race_id,
                    "horse_id": f"H{year}{race_no:02d}{horse_no:02d}",
                    "race_date": date,
                    "prediction_phase": "FINAL",
                    "feature_sets": ALL_SETS,
                    "history_windows": HISTORY_WINDOWS,
                    "small_sample_policy": SMALL_SAMPLE,
                    "features": features,
                    "target": {
                        "is_win": horse_no == 1,
                        "finish_position": horse_no,
                    },
                    "market_outcome": {},
                })
    return rows


def write_gzip(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(cmd, check=True):
    return subprocess.run(
        [str(x) for x in cmd],
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=not check,
    )


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)

    source = OUT / "snapshot-2024-2025.jsonl.gz"
    write_gzip(source, make_rows())

    candidates = OUT / "candidates.json"
    candidates.write_text(json.dumps([
        {"name": "base", "feature_sets": ["BASE"]},
        {"name": "opponent", "feature_sets": ["BASE", "OPPONENT"]},
        {"name": "time_pace", "feature_sets": ["BASE", "TIME_PACE"]},
        {"name": "all", "feature_sets": ALL_SETS},
    ], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    arena_out = OUT / "arena"
    run([
        sys.executable,
        ROOT / "research" / "run_feature_arena.py",
        "--inputs", source,
        "--out-dir", arena_out,
        "--train-start", "2024-01-01",
        "--train-end", "2024-12-31",
        "--valid-start", "2025-01-01",
        "--valid-end", "2025-12-31",
        "--candidates-json", candidates,
        "--source-sha", "BACKFILL_SMOKE_SHA",
        "--ml-source-sha", "ML_SMOKE_SHA",
    ])

    summary = json.loads((arena_out / "feature-arena-summary.json").read_text(encoding="utf-8"))
    assert summary["contract"] == "L1_FEATURE_ARENA_WIN_V1"
    assert summary["model_family"] == "WIN_BINARY_LIGHTGBM"
    assert summary["ability_uses_odds"] is False
    assert summary["combined_winner_score"] is None
    assert len(summary["results"]) == 4
    assert summary["delta_vs_base"]["baseline_candidate"] == "base"
    assert "top3_winner_capture" in summary["metric_views"]

    by_name = {row["name"]: row for row in summary["results"]}
    base_meta = json.loads(Path(by_name["base"]["metadata"]).read_text(encoding="utf-8"))
    opponent_meta = json.loads(Path(by_name["opponent"]["metadata"]).read_text(encoding="utf-8"))
    time_meta = json.loads(Path(by_name["time_pace"]["metadata"]).read_text(encoding="utf-8"))
    all_meta = json.loads(Path(by_name["all"]["metadata"]).read_text(encoding="utf-8"))

    assert base_meta["feature_sets"] == ["BASE"]
    assert "opponent_probe" not in base_meta["features"]
    assert "timepace_probe" not in base_meta["features"]

    assert opponent_meta["feature_sets"] == ["BASE", "OPPONENT"]
    assert "opponent_probe" in opponent_meta["features"]
    assert "timepace_probe" not in opponent_meta["features"]

    assert time_meta["feature_sets"] == ["BASE", "TIME_PACE"]
    assert "timepace_probe" in time_meta["features"]
    assert "opponent_probe" not in time_meta["features"]

    assert all_meta["feature_sets"] == ALL_SETS
    for probe in (
        "opponent_probe", "network_probe", "lap_probe", "style_probe",
        "distx_probe", "backfill_probe", "auto_probe", "ped_probe",
        "actor_probe", "timepace_probe",
    ):
        assert probe in all_meta["features"]

    for projected in (arena_out / "projections").glob("*.jsonl.gz"):
        raise AssertionError(f"projection should have been deleted: {projected}")

    locked = run([
        sys.executable,
        ROOT / "research" / "run_feature_arena.py",
        "--inputs", source,
        "--out-dir", OUT / "locked",
        "--train-start", "2025-01-01",
        "--train-end", "2025-12-31",
        "--valid-start", "2026-01-01",
        "--valid-end", "2026-12-31",
        "--candidates-json", candidates,
        "--plan-only",
    ], check=False)
    assert locked.returncode != 0
    text = (locked.stdout or "") + "\n" + (locked.stderr or "")
    assert "2026 research lock" in text

    print("FEATURE_ARENA_SMOKE_OK")
    shutil.rmtree(OUT, ignore_errors=True)


if __name__ == "__main__":
    main()
