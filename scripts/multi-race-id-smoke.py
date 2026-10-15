#!/usr/bin/env python3
"""No-network contract tests for the one-or-many race ID entry point."""
from __future__ import annotations
import gzip
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

batch = load("research/run-races-by-id.py", "batch_ids")
extract = load("research/extract-races-snapshot.py", "batch_extract")
A = "202506050811"
B = "202507010212"

assert batch.parse_ids(A) == [A]
assert batch.parse_ids(A + " " + A + "," + B) == [A, B]
assert batch.parse_ids("https://race.netkeiba.com/race/shutuba.html?race_id=" + A) == [A]
for value in ("202606050811", "12345", "https://evil.example/?race_id=" + A):
    try:
        batch.parse_ids(value)
        raise AssertionError("unsafe ID passed: " + value)
    except ValueError:
        pass

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    source = root / "snapshot.jsonl.gz"
    with gzip.open(source, "wt", encoding="utf-8") as out:
        for rid in (A, B):
            for i in range(1, 5):
                out.write(json.dumps({
                    "race_id": rid,
                    "horse_id": f"{rid}-{i}",
                    "feature_schema_version": 8,
                    "leakage_policy": "STRICT_PRIOR_DATE_ONLY",
                    "features": {"race_date": "2025-12-28", "horse_form": i},
                    "finish_position": i,
                    "final_win_odds": 4.0,
                    "target_win": 0,
                }) + "\n")
    folder = root / "races"
    got = extract.extract_many(source, [A, B], folder)
    assert got == {A: 4, B: 4}
    for rid in (A, B):
        rows = [json.loads(x) for x in (folder / rid / "features.jsonl").read_text().splitlines()]
        assert len(rows) == 4
        assert all("finish_position" not in r and "target_win" not in r
                   and "final_win_odds" not in r for r in rows)
    try:
        extract.extract_many(source, [A, "202606050811"], folder)
        raise AssertionError("2026 target unsealed")
    except ValueError:
        pass

    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"contract": "KEIBA_ID_BATCH_INPUTS_V1",
                                    "races": {A: {"mobile_export": "absent.json"}}}), encoding="utf-8")
    out_dir = root / "out"
    run = subprocess.run(
        [sys.executable, str(ROOT / "research/run-races-by-id.py"),
         "--race-ids", A + ", " + B,
         "--skip-kings", "--input-manifest", str(manifest),
         "--session-dir", str(root / "session"), "--out-dir", str(out_dir)],
        capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0, (run.stdout, run.stderr)
    report = json.loads((out_dir / "batch-summary.json").read_text())
    assert report["requested_race_ids"] == [A, B]
    assert report["race_count"] == 2
    assert report["summary"]["l3_paper_ready"] == 0
    assert report["summary"]["l3_blocked_or_missing"] == 2
    assert report["summary"]["paper_total_stake_yen_ready_only"] == 0
    assert report["races"][0]["l21_l3_status"] == "BLOCKED_INPUT_OR_PROVENANCE"
    assert report["races"][1]["l21_l3_status"] == "BLOCKED_MISSING_INPUTS"
    assert report["actual_bets_placed"] is False
    assert report["replay_is_verified_prebet"] is False

    strict = subprocess.run(
        [sys.executable, str(ROOT / "research/run-races-by-id.py"),
         "--race-ids", A + " " + B, "--skip-kings", "--require-l3",
         "--session-dir", str(root / "session"), "--out-dir", str(root / "strict")],
        capture_output=True, text=True, cwd=ROOT)
    assert strict.returncode == 2, strict.stderr
print("MULTI_RACE_ID_INPUT_AND_FAIL_CLOSED_PAPER_TEST_PASS", A, B, flush=True)
