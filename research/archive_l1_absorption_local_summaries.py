#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--experiment-id",required=True)
    p.add_argument("--run-id",required=True,type=int)
    p.add_argument("--snapshot-generation",required=True)
    p.add_argument("--backfill-sha",required=True)
    p.add_argument("--summary",action="append",default=[])
    p.add_argument("--expected-candidates",required=True)
    p.add_argument("--out-root",default="research-results/l1-raceclass-absorption")
    p.add_argument("--train-start",required=True)
    p.add_argument("--train-end",required=True)
    p.add_argument("--valid-start",required=True)
    p.add_argument("--valid-end",required=True)
    return p.parse_args()

def main():
    a=parse_args()
    expected=[x.strip() for x in a.expected_candidates.split(",") if x.strip()]
    candidates=[]
    for raw in a.summary:
        path=Path(raw)
        if not path.is_file():
            continue
        payload=json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("results",[]):
            candidates.append({
                "candidate":row.get("name"),
                "status":"success",
                "train_scope":(row.get("race_scope") or {}).get("train"),
                "valid_scope":(row.get("race_scope") or {}).get("valid"),
                "feature_sets":row.get("feature_sets") or [],
                "actor_prefixes":row.get("actor_prefixes") or [],
                "auto_slices":row.get("auto_slices") or [],
                "pedigree_slices":row.get("pedigree_slices") or [],
                "feature_count":row.get("feature_count"),
                "metrics":row.get("metrics") or {},
                "resource_usage":row.get("resource_usage") or {},
            })
    got={r["candidate"] for r in candidates}
    missing=[x for x in expected if x not in got]
    complete=not missing and len(candidates)==len(expected)

    out=Path(a.out_root)/a.experiment_id/"attempts"/f"run-{a.run_id}"
    if out.exists():
        raise SystemExit(f"immutable ledger path already exists: {out}")
    out.mkdir(parents=True,exist_ok=False)
    run_url=f"{os.environ.get('GITHUB_SERVER_URL','https://github.com')}/{os.environ.get('GITHUB_REPOSITORY','')}/actions/runs/{a.run_id}"
    exp={
        "contract":"L1_ABSORPTION_SPECIALIST_LEDGER_V1",
        "experiment_id":a.experiment_id,
        "run_id":a.run_id,
        "run_url":run_url,
        "head_branch":os.environ.get("GITHUB_REF_NAME"),
        "head_sha":os.environ.get("GITHUB_SHA"),
        "archived_at":datetime.now(timezone.utc).isoformat(),
        "model_family":"WIN_BINARY_LIGHTGBM",
        "ability_uses_odds":False,
        "research_domain":"RACE_CLASS_ABSORPTION",
        "snapshot_generation":a.snapshot_generation,
        "snapshot_persistence":"runner_local_ephemeral",
        "backfill_sha":a.backfill_sha,
        "train":{"start":a.train_start,"end":a.train_end},
        "holdout":{"start":a.valid_start,"end":a.valid_end,"race_class":"ALL"},
        "expected_candidates":expected,
        "candidate_count":len(candidates),
        "missing_candidates":missing,
        "separate_from_main_feature_arena":True,
        "storage_policy":{"ledger":"git","snapshot":"ephemeral","github_artifacts":False,"github_cache":False},
    }
    score={
        "contract":"L1_ABSORPTION_SPECIALIST_SCORECARD_V1",
        "experiment_id":a.experiment_id,
        "run_id":a.run_id,
        "complete":complete,
        "candidates":sorted(candidates,key=lambda x:x["candidate"]),
    }
    (out/"experiment.json").write_text(json.dumps(exp,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"scorecard.json").write_text(json.dumps(score,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"jobs.json").write_text(json.dumps({"run_id":a.run_id,"mode":"single-job-local-summaries"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    def pct(v):
        return "" if v is None else f"{100*float(v):.2f}%"
    lines=[
        f"# {a.experiment_id} — run {a.run_id}","",
        f"- Run: {run_url}",
        f"- Code SHA: {os.environ.get('GITHUB_SHA','')}",
        f"- Snapshot generation: {a.snapshot_generation}",
        "- Snapshot persistence: runner-local ephemeral only",
        f"- BACKFILL SHA: {a.backfill_sha}",
        f"- Train: {a.train_start} .. {a.train_end}",
        f"- Holdout: {a.valid_start} .. {a.valid_end}  / ALL races",
        "- Main Feature Arena ledger: separate","",
        "## Scorecard","",
        "| Candidate | Train | Top1 | Top3 | Top6 | Mean winner rank | Features | Peak MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(candidates,key=lambda x:-(x.get("metrics") or {}).get("top1_winner_capture",0)):
        m=r.get("metrics") or {}
        u=r.get("resource_usage") or {}
        lines.append(
            f"| {r['candidate']} | {r.get('train_scope') or ''} | {pct(m.get('top1_winner_capture'))} | "
            f"{pct(m.get('top3_winner_capture'))} | {pct(m.get('top6_winner_capture'))} | "
            f"{m.get('mean_winner_rank','')} | {r.get('feature_count','')} | {u.get('candidate_peak_rss_mib','')} |"
        )
    if missing:
        lines+=["","## Missing","",*["- "+x for x in missing]]
    (out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("ABSORPTION_LOCAL_LEDGER_READY")
    print(json.dumps({"path":str(out),"complete":complete,"missing":missing},ensure_ascii=False))

if __name__=="__main__":
    main()
