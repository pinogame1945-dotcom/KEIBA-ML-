#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULT_MARKER = "NEWCOMER_DUEL_RESULT "
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s+")

def parse_args():
    p = argparse.ArgumentParser(description="Archive NEWCOMER specialist run into a separate immutable Git ledger.")
    p.add_argument("--repo", required=True)
    p.add_argument("--run-id", required=True, type=int)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--out-root", default="research-results/l1-newcomer-specialist")
    p.add_argument("--snapshot-generation", required=True)
    p.add_argument("--backfill-sha", required=True)
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--valid-start", required=True)
    p.add_argument("--valid-end", required=True)
    p.add_argument("--expected-candidates", required=True)
    return p.parse_args()

def gh_json(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))

def gh_log(repo, job_id):
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        return "", "GH_TOKEN/GITHUB_TOKEN is required"
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    url = f"{api}/repos/{repo}/actions/jobs/{job_id}/logs"
    proc = subprocess.run([
        "curl", "-fsSL",
        "-H", f"Authorization: Bearer {token}",
        "-H", "Accept: application/vnd.github+json",
        "-H", "X-GitHub-Api-Version: 2022-11-28",
        url,
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode:
        return "", proc.stderr.decode("utf-8", errors="replace").strip()
    return proc.stdout.decode("utf-8", errors="replace"), None

def clean(line):
    return TIMESTAMP_RE.sub("", ANSI_RE.sub("", line).replace("\ufeff", "")).strip()

def extract_result(log_text):
    for raw in log_text.splitlines():
        line = clean(raw)
        at = line.find(RESULT_MARKER)
        if at < 0:
            continue
        payload = line[at + len(RESULT_MARKER):].strip()
        try:
            row = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("candidate"):
            return row
    return None

def failed_step(job):
    for step in job.get("steps") or []:
        if step.get("conclusion") == "failure":
            return step.get("name")
    return None

def main():
    a = parse_args()
    run = gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}")
    payload = gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}/jobs?per_page=100")
    jobs = [j for j in payload.get("jobs", []) if str(j.get("name", "")).startswith("newcomer-")]
    if not jobs:
        raise SystemExit("no newcomer-* jobs found")

    candidates = []
    jobs_record = []
    for job in sorted(jobs, key=lambda x: str(x.get("name", ""))):
        log_text, log_error = gh_log(a.repo, job["id"])
        result = extract_result(log_text) if log_text else None
        name = str(job.get("name", "")).removeprefix("newcomer-")
        jobs_record.append({
            "job_id": job.get("id"),
            "name": job.get("name"),
            "status": job.get("status"),
            "conclusion": job.get("conclusion"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "failed_step": failed_step(job),
            "log_fetch_error": log_error,
        })
        if result:
            candidates.append({
                "candidate": result.get("candidate", name),
                "job_id": job.get("id"),
                "status": "success",
                "train_scope": result.get("train_scope"),
                "valid_scope": result.get("valid_scope"),
                "feature_sets": result.get("feature_sets") or [],
                "actor_prefixes": result.get("actor_prefixes") or [],
                "feature_count": result.get("feature_count"),
                "metrics": result.get("metrics") or {},
                "resource_usage": result.get("resource_usage") or {},
            })
        else:
            candidates.append({
                "candidate": name,
                "job_id": job.get("id"),
                "status": job.get("conclusion") or job.get("status"),
                "train_scope": None,
                "valid_scope": None,
                "feature_sets": [],
                "actor_prefixes": [],
                "feature_count": None,
                "metrics": {},
                "failed_step": failed_step(job),
                "log_fetch_error": log_error,
            })

    expected = {x.strip() for x in a.expected_candidates.split(",") if x.strip()}
    got = {r["candidate"] for r in candidates}
    missing = sorted(expected - got)
    complete = not missing and len(candidates) == len(expected) and all(r["status"] == "success" for r in candidates)

    run_dir = Path(a.out_root) / a.experiment_id / "attempts" / f"run-{a.run_id}"
    if run_dir.exists():
        raise SystemExit(f"immutable ledger path already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    experiment = {
        "contract": "L1_NEWCOMER_SPECIALIST_LEDGER_V1",
        "experiment_id": a.experiment_id,
        "run_id": a.run_id,
        "run_url": run.get("html_url"),
        "workflow": run.get("name"),
        "head_branch": run.get("head_branch"),
        "head_sha": run.get("head_sha"),
        "run_conclusion": run.get("conclusion"),
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "model_family": "WIN_BINARY_LIGHTGBM",
        "ability_uses_odds": False,
        "specialist_domain": "NEWCOMER",
        "snapshot_generation": a.snapshot_generation,
        "backfill_sha": a.backfill_sha,
        "train": {"start": a.train_start, "end": a.train_end},
        "holdout": {"start": a.valid_start, "end": a.valid_end, "race_class": "NEWCOMER"},
        "expected_candidates": sorted(expected),
        "candidate_count": len(candidates),
        "missing_candidates": missing,
        "separate_from_main_feature_arena": True,
        "storage_policy": {
            "ledger": "git",
            "models": "not_stored_by_ledger",
            "github_artifacts": False,
            "github_cache": False,
        },
    }
    scorecard = {
        "contract": "L1_NEWCOMER_SPECIALIST_SCORECARD_V1",
        "experiment_id": a.experiment_id,
        "run_id": a.run_id,
        "complete": complete,
        "candidates": sorted(candidates, key=lambda r: r["candidate"]),
    }
    (run_dir / "experiment.json").write_text(json.dumps(experiment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "scorecard.json").write_text(json.dumps(scorecard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "jobs.json").write_text(json.dumps({"run_id": a.run_id, "jobs": jobs_record}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {a.experiment_id} — run {a.run_id}", "",
        f"- Run: {run.get('html_url')}",
        f"- Code SHA: {run.get('head_sha')}",
        f"- Snapshot: {a.snapshot_generation}",
        f"- BACKFILL SHA: {a.backfill_sha}",
        f"- Train: {a.train_start} .. {a.train_end}",
        f"- Holdout: {a.valid_start} .. {a.valid_end} / NEWCOMER only",
        "- L1 odds: NO",
        "- Ledger: NEWCOMER specialist only; separate from l1-feature-arena",
        f"- Complete: {'YES' if complete else 'NO'}",
        "", "## Scorecard", "",
        "| Candidate | Train scope | Features | Top1 | Top3 | Top6 | Mean winner rank | Peak MiB | Seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(candidates, key=lambda x: x["candidate"]):
        m = r.get("metrics") or {}
        u = r.get("resource_usage") or {}
        lines.append(
            f"| {r['candidate']} | {r.get('train_scope') or ''} | {r.get('feature_count') or ''} | "
            f"{m.get('top1_winner_capture','')} | {m.get('top3_winner_capture','')} | "
            f"{m.get('top6_winner_capture','')} | {m.get('mean_winner_rank','')} | "
            f"{u.get('candidate_peak_rss_mib','')} | {u.get('candidate_elapsed_seconds','')} |"
        )
    lines += ["", "## Rule", "", "This ledger never mixes with the main L1 Feature Arena ledger.", ""]
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    root = Path(a.out_root)
    root.mkdir(parents=True, exist_ok=True)
    root_readme = root / "README.md"
    if not root_readme.exists():
        root_readme.write_text(
            "# L1 NEWCOMER Specialist Research Ledger\n\n"
            "新馬戦専用L1研究の永久台帳。\n\n"
            "- 本線 research-results/l1-feature-arena/ とは完全分離する。\n"
            "- 実験IDは NEWCOMER-001, NEWCOMER-002, ... を使う。\n"
            "- 各Actions実行は <experiment>/attempts/run-<run_id>/ に不変保存する。\n"
            "- 成功・失敗の両方を残す。\n"
            "- 大容量モデル/データセットは保存しない。\n"
            "- L1能力評価にオッズは使わない。\n",
            encoding="utf-8",
        )

    print("NEWCOMER_SPECIALIST_LEDGER_READY")
    print(json.dumps({
        "experiment_id": a.experiment_id,
        "run_id": a.run_id,
        "path": str(run_dir),
        "complete": complete,
        "candidate_count": len(candidates),
        "missing_candidates": missing,
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
