#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULT_MARKERS = ("FEATURE_ARENA_RESULT", "FIRST_KODOKU_RESULT")
ANSI_RE = re.compile(r"\\x1b\\[[0-9;]*[A-Za-z]")
TIMESTAMP_RE = re.compile(r"^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?Z\\s+")

def parse_args():
    p = argparse.ArgumentParser(description="Archive Feature Arena run into immutable Git ledger.")
    p.add_argument("--repo", required=True)
    p.add_argument("--run-id", required=True, type=int)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--out-root", default="research-results/l1-feature-arena")
    p.add_argument("--snapshot-generation", required=True)
    p.add_argument("--backfill-sha", required=True)
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--valid-start", required=True)
    p.add_argument("--valid-end", required=True)
    p.add_argument("--prediction-phase", default="FINAL")
    p.add_argument("--model-family", default="WIN_BINARY_LIGHTGBM")
    p.add_argument("--feature-selection", default="none")
    p.add_argument("--expected-candidates", default="base,opponent,network,lap,style,distance,backfill,auto,pedigree,actor,time_pace,all")
    p.add_argument("--token-env", default="GH_TOKEN")
    return p.parse_args()

def gh_json(endpoint):
    raw = subprocess.check_output(["gh", "api", endpoint], text=True)
    return json.loads(raw)

def gh_log(repo, job_id):
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        return "", "GH_TOKEN/GITHUB_TOKEN is required for job log download"
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    url = f"{api}/repos/{repo}/actions/jobs/{job_id}/logs"
    proc = subprocess.run(
        [
            "curl", "-fsSL",
            "-H", f"Authorization: Bearer {token}",
            "-H", "Accept: application/vnd.github+json",
            "-H", "X-GitHub-Api-Version: 2022-11-28",
            url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return "", proc.stderr.decode("utf-8", errors="replace").strip()
    return proc.stdout.decode("utf-8", errors="replace"), None

def clean_line(line):
    line = ANSI_RE.sub("", line).replace("\\ufeff", "")
    line = TIMESTAMP_RE.sub("", line)
    return line.strip()

def extract_result(log_text):
    lines = [clean_line(x) for x in log_text.splitlines()]
    for i, line in enumerate(lines):
        if not any(marker in line for marker in RESULT_MARKERS):
            continue
        for candidate in lines[i + 1:i + 8]:
            start = candidate.find("{")
            if start < 0:
                continue
            try:
                value = json.loads(candidate[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("candidate"):
                return value
    return None

def extract_resource_usage(log_text):
    usages = []
    for raw in (log_text or "").splitlines():
        line = clean_line(raw)
        marker = "FEATURE_ARENA_RESOURCE_USAGE "
        at = line.find(marker)
        if at < 0:
            continue
        payload = line[at + len(marker):].strip()
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            usages.append(value)
    if not usages:
        return None
    peaks = [float(x.get("peak_rss_mib")) for x in usages if x.get("peak_rss_mib") is not None]
    elapsed = [float(x.get("elapsed_seconds")) for x in usages if x.get("elapsed_seconds") is not None]
    return {
        "stages": usages,
        "observed_peak_rss_mib": max(peaks) if peaks else None,
        "observed_stage_seconds": sum(elapsed) if elapsed else None,
    }


def failed_step(job):
    for step in job.get("steps") or []:
        if step.get("conclusion") == "failure":
            return step.get("name")
    return None

def short_error(log_text, limit=12):
    if not log_text:
        return None
    cleaned = [clean_line(x) for x in log_text.splitlines()]
    interesting = [line for line in cleaned if line and ("error" in line.lower() or "traceback" in line.lower() or "exception" in line.lower() or "failed" in line.lower() or "exit code" in line.lower())]
    return interesting[-limit:] or None

def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)

def main():
    a = parse_args()
    if not os.environ.get(a.token_env):
        raise SystemExit(f"{a.token_env} is required")

    run = gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}")
    jobs_payload = gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}/jobs?per_page=100")
    jobs = [j for j in jobs_payload.get("jobs", []) if str(j.get("name", "")).startswith("arena-")]
    if not jobs:
        raise SystemExit("no arena-* jobs found")

    candidates = []
    jobs_record = []
    for job in sorted(jobs, key=lambda x: str(x.get("name", ""))):
        log_text, log_error = gh_log(a.repo, job["id"])
        result = extract_result(log_text) if log_text else None
        jobs_record.append({
            "job_id": job.get("id"), "name": job.get("name"), "status": job.get("status"),
            "conclusion": job.get("conclusion"), "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"), "failed_step": failed_step(job),
            "log_fetch_error": log_error,
        })
        candidate_name = str(job.get("name", "")).removeprefix("arena-")
        if result:
            candidates.append({
                "candidate": result.get("candidate", candidate_name), "job_id": job.get("id"),
                "status": "success", "feature_sets": result.get("feature_sets") or [],
                "actor_prefixes": result.get("actor_prefixes") or [],
                "feature_count": result.get("feature_count"), "metrics": result.get("metrics") or {},
                "resource_usage": result.get("resource_usage") or extract_resource_usage(log_text),
            })
        else:
            candidates.append({
                "candidate": candidate_name, "job_id": job.get("id"),
                "status": job.get("conclusion") or job.get("status"), "feature_sets": [],
                "actor_prefixes": [],
                "feature_count": None, "metrics": {}, "failed_step": failed_step(job),
                "error_excerpt": short_error(log_text), "log_fetch_error": log_error,
                "resource_usage": extract_resource_usage(log_text),
            })

    expected = {x.strip() for x in a.expected_candidates.split(",") if x.strip()}
    if not expected:
        raise SystemExit("--expected-candidates resolved to an empty set")
    got = {row["candidate"] for row in candidates}
    missing = sorted(expected - got)

    run_dir = Path(a.out_root) / a.experiment_id / "attempts" / f"run-{a.run_id}"
    if run_dir.exists():
        raise SystemExit(f"immutable ledger path already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    experiment = {
        "contract": "L1_FEATURE_ARENA_LEDGER_V1", "experiment_id": a.experiment_id,
        "run_id": a.run_id, "run_url": run.get("html_url"), "workflow": run.get("name"),
        "event": run.get("event"), "head_branch": run.get("head_branch"), "head_sha": run.get("head_sha"),
        "run_attempt": run.get("run_attempt"), "run_status": run.get("status"),
        "run_conclusion": run.get("conclusion"), "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"), "archived_at": datetime.now(timezone.utc).isoformat(),
        "model_family": a.model_family, "ability_uses_odds": False,
        "prediction_phase": a.prediction_phase, "feature_selection": a.feature_selection,
        "snapshot_generation": a.snapshot_generation, "backfill_sha": a.backfill_sha,
        "train": {"start": a.train_start, "end": a.train_end},
        "holdout": {"start": a.valid_start, "end": a.valid_end},
        "expected_candidates": sorted(expected), "candidate_count": len(candidates),
        "missing_candidates": missing,
        "storage_policy": {"result_ledger": "git", "models": "not_stored_by_ledger", "github_artifacts": False, "github_cache": False},
    }
    scorecard = {
        "contract": "L1_FEATURE_ARENA_SCORECARD_V1", "experiment_id": a.experiment_id,
        "run_id": a.run_id,
        "complete": not missing and len(candidates) == 12 and all(r["status"] == "success" for r in candidates),
        "composite_winner": None,
        "candidates": sorted(candidates, key=lambda x: x["candidate"]),
    }

    (run_dir / "experiment.json").write_text(json.dumps(experiment, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    (run_dir / "scorecard.json").write_text(json.dumps(scorecard, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    (run_dir / "jobs.json").write_text(json.dumps({"run_id": a.run_id, "jobs": jobs_record}, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")

    lines = [
        f"# {a.experiment_id} — run {a.run_id}", "",
        f"- Run: {run.get('html_url')}",
        f"- Code SHA: {run.get('head_sha')}",
        f"- Snapshot: {a.snapshot_generation}",
        f"- BACKFILL SHA: {a.backfill_sha}",
        f"- Train: {a.train_start} .. {a.train_end}",
        f"- Holdout: {a.valid_start} .. {a.valid_end}",
        f"- Model: {a.model_family}", "- Odds in L1: NO",
        f"- Feature selection: {a.feature_selection}",
        f"- Run conclusion: {run.get('conclusion')}",
        f"- Ledger complete: {'YES' if scorecard['complete'] else 'NO'}",
        "", "## Scorecard", "",
        "| Candidate | Actor slice | Status | Features | Peak MiB | Seconds | Top1 | Top3 | Top6 | Mean rank | MRR | Race NLL | LogLoss | Brier | AUC |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(candidates, key=lambda x: x["candidate"]):
        m = row.get("metrics") or {}
        lines.append("| " + " | ".join([
            row["candidate"], ",".join(row.get("actor_prefixes") or []), row["status"], fmt(row.get("feature_count")),
            fmt((row.get("resource_usage") or {}).get("candidate_peak_rss_mib") or (row.get("resource_usage") or {}).get("observed_peak_rss_mib")),
            fmt((row.get("resource_usage") or {}).get("candidate_elapsed_seconds") or (row.get("resource_usage") or {}).get("observed_stage_seconds")),
            fmt(m.get("top1_winner_capture")), fmt(m.get("top3_winner_capture")),
            fmt(m.get("top6_winner_capture")), fmt(m.get("mean_winner_rank")),
            fmt(m.get("mean_reciprocal_winner_rank")), fmt(m.get("race_normalized_nll")),
            fmt(m.get("raw_logloss")), fmt(m.get("raw_brier")), fmt(m.get("raw_roc_auc")),
        ]) + " |")

    failures = [r for r in candidates if r["status"] != "success"]
    if failures:
        lines += ["", "## Failures", ""]
        for row in failures:
            lines += [f"### {row['candidate']}", "", f"- Status: {row['status']}"]
            if row.get("failed_step"):
                lines.append(f"- Failed step: {row['failed_step']}")
            if row.get("error_excerpt"):
                lines += ["- Error excerpt:", "", "    " + "\\n    ".join(row["error_excerpt"])]
            lines.append("")

    lines += ["", "## Interpretation policy", "", "No composite winner is assigned. Metrics are preserved as measured for later research.", ""]
    (run_dir / "README.md").write_text("\\n".join(lines), encoding="utf-8")

    print("FEATURE_ARENA_LEDGER_READY")
    print(json.dumps({"experiment_id": a.experiment_id, "run_id": a.run_id, "path": str(run_dir), "candidate_count": len(candidates), "complete": scorecard["complete"], "missing_candidates": missing}, ensure_ascii=False))

if __name__ == "__main__":
    main()
