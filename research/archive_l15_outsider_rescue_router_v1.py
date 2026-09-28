#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from datetime import datetime,timezone
from pathlib import Path

MARKER="L15_OUTSIDER_RESCUE_ROUTER_RESULT"
ANSI_RE=re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TS_RE=re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s+")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True)
    p.add_argument("--run-id",required=True,type=int)
    p.add_argument("--out-root",default="research-results/l15-outsider-rescue-router-v1")
    return p.parse_args()

def gh_json(ep):
    return json.loads(subprocess.check_output(["gh","api",ep],text=True))

def gh_log(repo,job_id):
    token=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        return "","token missing"
    api=os.environ.get("GITHUB_API_URL","https://api.github.com").rstrip("/")
    url=f"{api}/repos/{repo}/actions/jobs/{job_id}/logs"
    p=subprocess.run([
        "curl","-fsSL",
        "-H",f"Authorization: Bearer {token}",
        "-H","Accept: application/vnd.github+json",
        "-H","X-GitHub-Api-Version: 2022-11-28",
        url,
    ],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if p.returncode:
        return "",p.stderr.decode(errors="replace").strip()
    return p.stdout.decode(errors="replace"),None

def clean(s):
    return TS_RE.sub("",ANSI_RE.sub("",s).replace("\ufeff","")).strip()

def extract(log):
    lines=[clean(x) for x in log.splitlines()]
    for i,line in enumerate(lines):
        if MARKER not in line:
            continue
        for row in lines[i+1:i+10]:
            at=row.find("{")
            if at<0:
                continue
            try:
                x=json.loads(row[at:])
            except json.JSONDecodeError:
                continue
            if isinstance(x,dict) and x.get("contract")=="L15_OUTSIDER_RESCUE_ROUTER_V1_FOLD":
                return x
    return None

def aggregate_policy(folds,key,profile=None):
    rows=[]
    for f in folds:
        p=f["policies"][key]
        if profile is not None:
            p=p[profile]
        rows.append(p)
    blind=sum(int(x.get("blind_spots") or 0) for x in rows)
    rescued=sum(int(x.get("rescued_blind_spots") or 0) for x in rows)
    races=sum(int(f.get("races") or 0) for f in folds)
    interventions=sum(int(x.get("interventions") or 0) for x in rows)
    return {
        "races":races,
        "blind_spots":blind,
        "rescued_blind_spots":rescued,
        "blind_rescue_rate":rescued/blind if blind else None,
        "remaining_blind_spots":blind-rescued,
        "interventions":interventions,
        "intervention_rate":interventions/races if races else None,
        "rescue_per_100_interventions":100*rescued/interventions if interventions else 0.0,
    }

def weighted_metric(folds,path):
    num=den=0.0
    for f in folds:
        cur=f
        for k in path:
            cur=cur.get(k) if isinstance(cur,dict) else None
            if cur is None:
                break
        if cur is None:
            continue
        w=float(f.get("races") or 0)
        num+=float(cur)*w
        den+=w
    return num/den if den else None

def main():
    a=parse_args()
    run=gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}")
    jobs=gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}/jobs?per_page=100").get("jobs",[])
    folds=[]
    job_rows=[]
    for job in jobs:
        name=str(job.get("name") or "")
        if not name.startswith("rescue-router-y"):
            continue
        log,err=gh_log(a.repo,job["id"])
        result=extract(log)
        job_rows.append({
            "job_id":job.get("id"),
            "name":name,
            "conclusion":job.get("conclusion"),
            "log_fetch_error":err,
            "has_result":result is not None,
        })
        if result is not None:
            folds.append(result)

    folds=sorted(folds,key=lambda x:int(x["test_year"]))
    got={int(x["test_year"]) for x in folds}
    expected={2022,2023,2024,2025}
    complete=(got==expected and all(j.get("conclusion")=="success" for j in job_rows))

    profiles=("recall","balanced","precision")
    aggregate={
        "P0":aggregate_policy(folds,"P0") if folds else None,
        "P1":aggregate_policy(folds,"P1") if folds else None,
        "P2":aggregate_policy(folds,"P2") if folds else None,
        "P3":{p:aggregate_policy(folds,"P3",p) for p in profiles} if folds else {},
        "P4":{p:aggregate_policy(folds,"P4",p) for p in profiles} if folds else {},
    }
    summary={
        "contract":"L15_OUTSIDER_RESCUE_ROUTER_V1_SUMMARY",
        "run_id":a.run_id,
        "run_url":run.get("html_url"),
        "head_sha":run.get("head_sha"),
        "head_branch":run.get("head_branch"),
        "archived_at":datetime.now(timezone.utc).isoformat(),
        "complete":complete,
        "missing_years":sorted(expected-got),
        "folds":folds,
        "aggregate":aggregate,
        "stage_a_weighted":{
            "roc_auc":weighted_metric(folds,["stage_a","test_roc_auc"]),
            "pr_auc":weighted_metric(folds,["stage_a","test_pr_auc"]),
            "blind_prevalence":weighted_metric(folds,["stage_a","blind_prevalence"]),
            "heuristic_disagreement_roc_auc":weighted_metric(folds,["stage_a","heuristic_disagreement_roc_auc"]),
            "heuristic_disagreement_pr_auc":weighted_metric(folds,["stage_a","heuristic_disagreement_pr_auc"]),
        },
        "jobs":job_rows,
        "odds_used":False,
        "locked_years":[2026],
        "cost_policy":{
            "runner":"standard_ubuntu_cpu",
            "gpu":False,
            "github_artifact":False,
            "github_cache":False,
            "new_kaggle_persistence":False,
        },
        "candidate_inflation_status":"pending_additive_integration",
        "notes":[
            "This run measures routing feasibility from pre-race Router Feature Snapshot V1.",
            "P2 is an always-call-all-three practical-cost reference, not a hindsight selector.",
            "The 90.78% all-outsider union is hindsight-only oracle headroom.",
            "Production promotion still requires additive candidate integration and downstream L2/L3 EV/ROI validation.",
        ],
    }

    out=Path(a.out_root)/f"run-{a.run_id}"
    if out.exists():
        raise SystemExit(f"immutable output exists: {out}")
    out.mkdir(parents=True)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    lines=[
        f"# L1.5 Outsider Rescue Router V1 — run {a.run_id}",
        "",
        f"- Run: {run.get('html_url')}",
        f"- Head SHA: {run.get('head_sha')}",
        f"- Complete: {complete}",
        "- 2026 sealed",
        "- Odds: NO",
        "- Runner: standard GitHub CPU only",
        "- Artifact/Cache/new Kaggle persistence: NO",
        "",
        "## Fold results",
        "",
        "| Test | Races | Blind | Gate AUC | Gate PR-AUC | P1 rescue | P2 rescue | P3 balanced | P4 balanced |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in folds:
        def pct(v):
            return "" if v is None else f"{100*float(v):.2f}%"
        lines.append(
            f"| {f['test_year']} | {f['races']} | {f['blind_spots']} | "
            f"{f['stage_a']['test_roc_auc']:.4f} | {f['stage_a']['test_pr_auc']:.4f} | "
            f"{pct(f['policies']['P1']['blind_rescue_rate'])} | "
            f"{pct(f['policies']['P2']['blind_rescue_rate'])} | "
            f"{pct(f['policies']['P3']['balanced']['blind_rescue_rate'])} | "
            f"{pct(f['policies']['P4']['balanced']['blind_rescue_rate'])} |"
        )
    if folds:
        lines += [
            "",
            "## Aggregate",
            "",
            "| Policy | Rescued | Blind | Rescue rate | Interventions | Rescue / 100 interventions |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        display=[
            ("P0",aggregate["P0"]),
            ("P1",aggregate["P1"]),
            ("P2",aggregate["P2"]),
            ("P3 balanced",aggregate["P3"]["balanced"]),
            ("P4 balanced",aggregate["P4"]["balanced"]),
        ]
        for name,x in display:
            lines.append(
                f"| {name} | {x['rescued_blind_spots']} | {x['blind_spots']} | "
                f"{100*x['blind_rescue_rate']:.2f}% | {x['interventions']} | "
                f"{x['rescue_per_100_interventions']:.2f} |"
            )
    lines += [
        "",
        "Candidate inflation is intentionally pending: source outsider Top6 horse lists were ephemeral.",
        "No production promotion is permitted before additive integration and L2/L3 EV/ROI validation.",
        "",
    ]
    (out/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print("L15_OUTSIDER_RESCUE_ROUTER_LEDGER_READY")
    print(json.dumps({"run_id":a.run_id,"complete":complete,"missing_years":sorted(expected-got),"path":str(out)},ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
