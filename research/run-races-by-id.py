#!/usr/bin/env python3
"""Batch race-ID entry point: one Kaggle snapshot pass per year, frozen L2.1/L3.

This is a READ-ONLY paper harness. It does not certify a historical FINAL-phase
snapshot as pre-bet; it never substitutes 7-King consensus for ANCHOR_FULL.
Each race produces a durable PASS/BLOCKED result; failures are isolated.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
ID = re.compile(r"202[2-5][0-9]{8}")
MAX_RACES = 60
MANIFEST_CONTRACT = "KEIBA_ID_BATCH_INPUTS_V1"
REPORT_CONTRACT = "KEIBA_ID_BATCH_REPORT_V1"

def parse_ids(raw: str) -> list[str]:
    unique = []
    for token in re.split(r"[,;\s]+", raw.strip()):
        if not token:
            continue
        if token.startswith("https://"):
            u = urlparse(token)
            if u.hostname not in ("race.netkeiba.com", "db.netkeiba.com"):
                raise ValueError("UNAPPROVED_RACE_URL_HOST")
            ids = parse_qs(u.query).get("race_id", [])
            if len(ids) != 1:
                raise ValueError("MISSING_OR_DUPLICATE_RACE_ID_URL")
            token = ids[0]
        if not ID.fullmatch(token):
            raise ValueError("INVALID_ID_OR_2026_SEALED_" + token[:30])
        if token not in unique:
            unique.append(token)
    if not 1 <= len(unique) <= MAX_RACES:
        raise ValueError("BATCH_SIZE_MUST_BE_1_TO_60")
    return unique

def read_manifest(path: str | None) -> tuple[dict, Path]:
    if not path:
        return {}, Path.cwd()
    file = Path(path).resolve()
    data = json.loads(file.read_text(encoding="utf-8"))
    if data.get("contract") != MANIFEST_CONTRACT or not isinstance(data.get("races"), dict):
        raise ValueError("INVALID_BATCH_INPUT_MANIFEST_CONTRACT")
    if any(not ID.fullmatch(str(x)) for x in data["races"]):
        raise ValueError("MANIFEST_CONTAINS_UNSUPPORTED_RACE_ID")
    return data["races"], file.parent

def local_path(raw: object, base: Path) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("NON_STRING_OR_EMPTY_LOCAL_SOURCE_PATH")
    file = Path(raw)
    file = (file if file.is_absolute() else base / file).resolve()
    if not file.is_file():
        raise ValueError("SOURCE_FILE_NOT_FOUND_" + file.name)
    return str(file)

def execute(argv: list[str], *, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=ROOT, text=True, capture_output=True,
                          env=env, timeout=2400, check=False)

def release_snapshot(session: Path, remote: str) -> None:
    """Only remove a metadata-resolved, runner-local large snapshot."""
    path = (session / "snapshot" / remote).resolve()
    folder = (session / "snapshot").resolve()
    if path.parent == folder and path.is_file():
        path.unlink()

def run_batch(race_ids: list[str], session: Path, out_root: Path,
              sources: dict, sources_root: Path,
              limits: str | None, skip_kings: bool) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    session.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {rid: {
        "race_id": rid,
        "seven_king_status": "NOT_STARTED",
        "l21_l3_status": "NOT_STARTED",
        "l3_paper_ticket_count": None,
        "l3_paper_stake_yen": None,
        "warnings": ["ARCHIVED_FINAL_SNAPSHOT_NOT_VERIFIED_PREBET",
                     "NO_RESULT_LABELS_ALLOWED", "NO_REAL_BETS_PLACED"],
    } for rid in race_ids}
    by_year: dict[str, list[str]] = defaultdict(list)
    for rid in race_ids:
        by_year[rid[:4]].append(rid)

    for year, ids in sorted(by_year.items()):
        if skip_kings:
            for rid in ids:
                results[rid]["seven_king_status"] = "SKIPPED_FOR_INPUT_OR_OFFLINE_TEST"
            continue
        if not os.environ.get("KAGGLE_API_TOKEN"):
            for rid in ids:
                results[rid]["seven_king_status"] = "BLOCKED_MISSING_KAGGLE_API_TOKEN"
            continue
        prep_env = dict(os.environ, KEIBA_PREPARE_ONLY="1")
        prep = execute(["bash", "scripts/score-known-race-kaggle.sh",
                        ids[0], str(session), str(session / "unused.json")], env=prep_env)
        if prep.returncode != 0:
            for rid in ids:
                results[rid]["seven_king_status"] = "BLOCKED_SNAPSHOT_PREPARATION"
                results[rid]["seven_king_error"] = (prep.stderr or prep.stdout)[-800:]
            continue
        resolved = session / "snapshot" / ("resolved-" + year + ".tsv")
        try:
            lines = [line.split("\t") for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(lines) != 1 or len(lines[0]) < 3:
                raise ValueError("SNAPSHOT_MANIFEST_RESOLUTION_NOT_UNIQUE")
            remote = lines[0][2]
            snapshot = (session / "snapshot" / remote).resolve()
            if snapshot.parent != (session / "snapshot").resolve() or not snapshot.is_file():
                raise ValueError("SNAPSHOT_PATH_NOT_METADATA_RESOLVED")
            target_ids = session / ("batch-" + year + "-ids.txt")
            target_ids.write_text("\n".join(ids) + "\n", encoding="utf-8")
            ext = execute([sys.executable, "research/extract-races-snapshot.py",
                           "--input", str(snapshot), "--ids-file", str(target_ids),
                           "--output-dir", str(session / "races")])
            if ext.returncode != 0:
                raise ValueError("SNAPSHOT_MULTIRACE_EXTRACT_FAILED: " + (ext.stderr or ext.stdout)[-650:])
        except (ValueError, OSError) as exc:
            for rid in ids:
                results[rid]["seven_king_status"] = "BLOCKED_SNAPSHOT_BATCH_EXTRACT"
                results[rid]["seven_king_error"] = str(exc)[-800:]
            continue
        for rid in ids:
            output = out_root / rid / "prior-year-seven-kings.json"
            score = execute(["bash", "scripts/score-known-race-kaggle.sh",
                             rid, str(session), str(output)])
            if score.returncode:
                results[rid]["seven_king_status"] = "BLOCKED_PRIOR_YEAR_MODEL"
                results[rid]["seven_king_error"] = (score.stderr or score.stdout)[-800:]
                continue
            try:
                data = json.loads(output.read_text(encoding="utf-8"))
                if (data.get("contract") != "KEIBA_PRIOR_YEAR_MODEL_SINGLE_RACE_V1"
                    or data.get("race_id") != rid or data.get("model_count") != 7
                    or data.get("input_outcome_labels_used") is not False
                    or data.get("snapshot_prediction_phase") != "FINAL_ARCHIVED_TIMESTAMP_NOT_VERIFIED"):
                    raise ValueError("PRIOR_YEAR_MODEL_OUTPUT_CONTRACT_DRIFT")
                results[rid]["seven_king_status"] = "SEVEN_PRIOR_YEAR_MODELS_SCORED_ARCHIVED_FINAL"
                results[rid]["seven_king_horse_count"] = data["horse_count"]
                results[rid]["rank_sum_comparator_top3"] = [
                    {"horse_id": x["horse_id"], "rank_sum": x["rank_sum"]}
                    for x in data["horses"][:3]
                ]
                results[rid]["seven_king_output"] = str(output)
            except (ValueError, KeyError, OSError) as exc:
                results[rid]["seven_king_status"] = "BLOCKED_KINGS_OUTPUT_VALIDATION"
                results[rid]["seven_king_error"] = str(exc)

        # Release the large downloaded year snapshot only after all selected
        # race features and seven scores have been computed. One year at a time
        # keeps even four target years within the free runner's local disk.
        if len(by_year) > 1:
            release_snapshot(session, remote)

    for rid in race_ids:
        record = results[rid]
        folder = out_root / rid
        folder.mkdir(parents=True, exist_ok=True)
        paper = [sys.executable, "research/run-one-race-paper-pipeline.py",
                 "--race", rid, "--out-dir", str(folder / "l21-l3")]
        entry = sources.get(rid) or {}
        if not isinstance(entry, dict):
            record["l21_l3_status"] = "BLOCKED_INVALID_MANIFEST_RACE_RECORD"
            continue
        fields = {
            "mobile_export": "--mobile-export",
            "original_features": "--original-features",
            "original_anchor_parameters": "--original-anchor-parameters",
            "original_ability_risk": "--original-ability-risk",
            "analysis": "--analysis",
            "l3_limits": "--limits",
        }
        try:
            for field, flag in fields.items():
                value = entry.get(field)
                if field == "l3_limits" and value is None:
                    value = limits
                    base = Path.cwd()
                else:
                    base = sources_root
                path = local_path(value, base)
                if path:
                    paper.extend([flag, path])
            process = execute(paper)
            status_path = folder / "l21-l3" / "pipeline-status.json"
            if process.returncode or not status_path.is_file():
                record["l21_l3_status"] = "BLOCKED_PIPELINE_PROCESS"
                record["l21_l3_error"] = (process.stderr or process.stdout)[-800:]
                continue
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            if payload.get("race_id") != rid or payload.get("contract") != "KEIBA_ONE_RACE_PAPER_ORCHESTRATION_V1":
                raise ValueError("L21_L3_OUTPUT_CONTRACT_DRIFT")
            record["l21_l3_status"] = payload["status"]
            record["missing_inputs"] = payload.get("required_inputs_missing", [])
            record["stages"] = payload.get("stages", {})
            record["l21_l3_status_file"] = str(status_path)
            if payload["status"] == "PAPER_ONLY_LIMITS_CHECKED":
                record["l3_paper_ticket_count"] = payload["result_ticket_count"]
                record["l3_paper_stake_yen"] = payload["result_total_baseline_yen"]
                record["l3_paper_portfolio"] = str(folder / "l21-l3" / payload["l3_output_path"])
        except (ValueError, KeyError, OSError) as exc:
            record["l21_l3_status"] = "BLOCKED_INPUT_OR_PROVENANCE"
            record["l21_l3_error"] = str(exc)[:800]

    rows = [results[rid] for rid in race_ids]
    ready = [x for x in rows if x["l21_l3_status"] == "PAPER_ONLY_LIMITS_CHECKED"]
    report = {
        "contract": REPORT_CONTRACT,
        "requested_race_ids": race_ids,
        "race_count": len(race_ids),
        "summary": {
            "prior_year_seven_king_scored": sum(x["seven_king_status"] == "SEVEN_PRIOR_YEAR_MODELS_SCORED_ARCHIVED_FINAL" for x in rows),
            "l3_paper_ready": len(ready),
            "l3_blocked_or_missing": len(rows) - len(ready),
            "paper_ticket_count_ready_only": sum(x["l3_paper_ticket_count"] for x in ready),
            "paper_total_stake_yen_ready_only": sum(x["l3_paper_stake_yen"] for x in ready),
        },
        "replay_is_verified_prebet": False,
        "actual_bets_placed": False,
        "freeze_l21_unmodified": True,
        "races": rows,
    }
    dest = out_root / "batch-summary.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for x in rows:
        print("RACE_ID_RESULT", x["race_id"],
              "7KING=" + x["seven_king_status"],
              "L2_L3=" + x["l21_l3_status"],
              "PAPER_TICKETS=" + str(x["l3_paper_ticket_count"]),
              "PAPER_YEN=" + str(x["l3_paper_stake_yen"]), flush=True)
        if x.get("rank_sum_comparator_top3"):
            print("  SAVED_FINAL_7KING_COMPARATOR_TOP3",
                  json.dumps(x["rank_sum_comparator_top3"], ensure_ascii=False), flush=True)
        if x.get("missing_inputs"):
            print("  REQUIRED_INPUTS_MISSING", ",".join(x["missing_inputs"]), flush=True)
    print("MULTIRACE_BATCH_SUMMARY", json.dumps(report["summary"], ensure_ascii=False),
          "file=" + str(dest), flush=True)
    return report

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--race-ids", help="One or many IDs, comma/space/newline separated")
    p.add_argument("--ids-file", help="File of IDs, one per line or separated by commas")
    p.add_argument("--input-manifest", help="Exact, separately captured per-race L2.1/L3 inputs")
    p.add_argument("--l3-limits", help="Common L3 flat100 budget/limits JSON")
    p.add_argument("--session-dir", default=os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), "keiba-id-batch-session"))
    p.add_argument("--out-dir", default="out/race-id-batch")
    p.add_argument("--skip-kings", action="store_true", help="Offline contract smoke only; never fake a 7King prediction")
    p.add_argument("--require-l3", action="store_true", help="Exit nonzero unless every race has a valid L3 paper portfolio")
    a = p.parse_args()
    raw = "\n".join(filter(None, [
        a.race_ids,
        Path(a.ids_file).read_text(encoding="utf-8") if a.ids_file else None,
    ]))
    try:
        ids = parse_ids(raw)
        src, root = read_manifest(a.input_manifest)
        report = run_batch(ids, Path(a.session_dir).resolve(),
                           Path(a.out_dir).resolve(), src, root,
                           a.l3_limits, a.skip_kings)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    if a.require_l3 and report["summary"]["l3_paper_ready"] != len(ids):
        raise SystemExit(2)

if __name__ == "__main__":
    main()
