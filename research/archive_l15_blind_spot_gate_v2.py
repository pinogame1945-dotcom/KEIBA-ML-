#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from datetime import datetime,timezone
from pathlib import Path

MARKER="L15_BLIND_SPOT_GATE_V2_RESULT"
CONTRACT="L15_BLIND_SPOT_GATE_V2_FOLD"
ANSI_RE=re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TS_RE=re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s+")

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True)
    p.add_argument("--run-id",required=True,type=int)
    p.add_argument("--out-root",default="research-results/l15-blind-spot-gate-v2")
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
            if isinstance(x,dict) and x.get("contract")==CONTRACT:
                return x
    return None

def agg_budget(folds,model_key,mode,budget):
    rows=[f[model_key][mode][budget] for f in folds]
    races=sum(int(x.get("races") or 0) for x in rows)
    blind=sum(int(x.get("blind_spots") or 0) for x in rows)
    selected=sum(int(x.get("selected") or 0) for x in rows)
    caught=sum(int(x.get("caught_blind_spots") or 0) for x in rows)
    fp=sum(int(x.get("false_positives") or 0) for x in rows)
    fn=sum(int(x.get("false_negatives") or 0) for x in rows)
    rate=selected/races if races else None
    recall=caught/blind if blind else None
    precision=caught/selected if selected else None
    prev=blind/races if races else None
    return {
        "races":races,
        "blind_spots":blind,
        "selected":selected,
        "intervention_rate":rate,
        "caught_blind_spots":caught,
        "blind_recall":recall,
        "precision":precision,
        "enrichment_vs_prevalence":precision/prev if precision is not None and prev else None,
        "lift_vs_random_recall":recall/rate if recall is not None and rate else None,
        "false_positives":fp,
        "false_negatives":fn,
    }

def weighted(folds,model_key,metric):
    num=den=0.0
    for f in folds:
        v=f[model_key].get(metric)
        if v is None:
            continue
        w=float(f.get("races") or 0)
        num+=float(v)*w
        den+=w
    return num/den if den else None

def main():
    a=args()
    run=gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}")
    jobs=gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}/jobs?per_page=100").get("jobs",[])
    folds=[]
    job_rows=[]
    for job in jobs:
        name=str(job.get("name") or "")
        if not name.startswith("blind-gate-v2-y"):
            continue
        log,err=gh_log(a.repo,job["id"])
        x=extract(log)
        job_rows.append({
            "job_id":job.get("id"),
            "name":name,
            "conclusion":job.get("conclusion"),
            "has_result":x is not None,
            "log_fetch_error":err,
        })
        if x is not None:
            folds.append(x)
    folds=sorted(folds,key=lambda x:int(x["test_year"]))
    expected={2022,2023,2024,2025}
    got={int(x["test_year"]) for x in folds}
    complete=got==expected and all(x.get("conclusion")=="success" for x in job_rows)

    budgets=("5%","10%","15%","20%","25%")
    aggregate={}
    for model_key in ("base","pattern"):
        aggregate[model_key]={
            "weighted_roc_auc":weighted(folds,model_key,"roc_auc"),
            "weighted_pr_auc":weighted(folds,model_key,"pr_auc"),
            "test_rank":{b:agg_budget(folds,model_key,"test_rank",b) for b in budgets},
            "train_quantile":{b:agg_budget(folds,model_key,"train_quantile",b) for b in budgets},
        }

    summary={
        "contract":"L15_BLIND_SPOT_GATE_V2_SUMMARY",
        "run_id":a.run_id,
        "run_url":run.get("html_url"),
        "head_sha":run.get("head_sha"),
        "head_branch":run.get("head_branch"),
        "archived_at":datetime.now(timezone.utc).isoformat(),
        "complete":complete,
        "missing_years":sorted(expected-got),
        "folds":folds,
        "aggregate":aggregate,
        "primary_budget":"10%",
        "odds_used":False,
        "locked_years":[2026],
        "cost_policy":{
            "runner":"standard_ubuntu_cpu",
            "gpu":False,
            "github_artifact":False,
            "github_cache":False,
            "new_kaggle_persistence":False,
        },
        "join_ready":{
            "race_ids_preserved":True,
            "fields":"folds.*.(base|pattern).race_ids_10pct.*",
            "purpose":"later join against independent user research by race_id without mixing models first",
        },
    }

    out=Path(a.out_root)/f"run-{a.run_id}"
    if out.exists():
        raise SystemExit(f"immutable output exists: {out}")
    out.mkdir(parents=True)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    def pct(v):
        return "" if v is None else f"{100*float(v):.2f}%"

    lines=[
        f"# L1.5 Blind-Spot Gate V2 — run {a.run_id}",
        "",
        f"- Run: {run.get('html_url')}",
        f"- Complete: {complete}",
        "- 2026 sealed / Odds NO",
        "- Standard CPU only / Artifact NO / Cache NO / New Kaggle persistence NO",
        "",
        "## 10% primary budget by year — TRAIN_QUANTILE",
        "",
        "| Year | Blind | BASE selected | BASE recall | PATTERN selected | PATTERN recall | Δ recall pp |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in folds:
        b=f["base"]["train_quantile"]["10%"]
        p=f["pattern"]["train_quantile"]["10%"]
        delta=100*((p.get("blind_recall") or 0)-(b.get("blind_recall") or 0))
        lines.append(
            f"| {f['test_year']} | {f['blind_spots']} | {b['selected']} | {pct(b['blind_recall'])} | "
            f"{p['selected']} | {pct(p['blind_recall'])} | {delta:+.2f} |"
        )

    lines += [
        "",
        "## Aggregate",
        "",
        "| Model | AUC | PR-AUC | Mode | Budget | Intervention | Blind recall | Precision | Enrichment |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for model_key,label in (("base","BASE"),("pattern","PATTERN")):
        for mode,label_mode in (("test_rank","TEST_RANK"),("train_quantile","TRAIN_QUANTILE")):
            for b in budgets:
                x=aggregate[model_key][mode][b]
                enrich="" if x["enrichment_vs_prevalence"] is None else f"{x['enrichment_vs_prevalence']:.2f}x"
                lines.append(
                    f"| {label} | {aggregate[model_key]['weighted_roc_auc']:.4f} | "
                    f"{aggregate[model_key]['weighted_pr_auc']:.4f} | {label_mode} | {b} | "
                    f"{pct(x['intervention_rate'])} | {pct(x['blind_recall'])} | {pct(x['precision'])} | "
                    f"{enrich} |"
                )

    lines += [
        "",
        "10% race IDs are preserved in summary.json for later independent-study joins.",
        "",
    ]
    (out/"README.md").write_text("\n".join(lines),encoding="utf-8")

    print("L15_BLIND_SPOT_GATE_V2_LEDGER_READY")
    print(json.dumps({
        "run_id":a.run_id,
        "complete":complete,
        "missing_years":sorted(expected-got),
        "path":str(out),
    },ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
