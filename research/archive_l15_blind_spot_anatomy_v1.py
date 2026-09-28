#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from datetime import datetime,timezone
from pathlib import Path

MARKER="L15_BLIND_SPOT_ANATOMY_V1_RESULT"
CONTRACT="L15_BLIND_SPOT_ANATOMY_V1_FOLD"
GROUPS=("BASE","BREADTH","REDUNDANCY","CORE_FRINGE","PAIRWISE","TOP3","TOP1","ALL_ANATOMY")
BUDGETS=("5%","10%","15%","20%","25%")
ANSI_RE=re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TS_RE=re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s+")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True)
    p.add_argument("--run-id",required=True,type=int)
    p.add_argument("--out-root",default="research-results/l15-blind-spot-anatomy-v1")
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

def aggregate_budget(folds,group,budget):
    rows=[f["groups"][group]["budgets"][budget] for f in folds]
    races=sum(int(x["races"]) for x in rows)
    blind=sum(int(x["blind_spots"]) for x in rows)
    selected=sum(int(x["selected"]) for x in rows)
    caught=sum(int(x["caught_blind_spots"]) for x in rows)
    fp=sum(int(x["false_positives"]) for x in rows)
    fn=sum(int(x["false_negatives"]) for x in rows)
    intervention=selected/races if races else None
    recall=caught/blind if blind else None
    precision=caught/selected if selected else None
    prevalence=blind/races if races else None
    return {
        "races":races,
        "blind_spots":blind,
        "selected":selected,
        "intervention_rate":intervention,
        "caught_blind_spots":caught,
        "blind_recall":recall,
        "precision":precision,
        "lift_vs_random_recall":recall/intervention if recall is not None and intervention else None,
        "enrichment_vs_prevalence":precision/prevalence if precision is not None and prevalence else None,
        "false_positives":fp,
        "false_negatives":fn,
    }

def weighted(folds,group,key):
    num=den=0.0
    for f in folds:
        v=f["groups"][group].get(key)
        if v is None:
            continue
        w=float(f["races"])
        num+=float(v)*w
        den+=w
    return num/den if den else None

def overlap10(fold,group):
    base=set(fold["groups"]["BASE"]["race_ids_10pct"]["selected"])
    cur=set(fold["groups"][group]["race_ids_10pct"]["selected"])
    union=base|cur
    return {
        "base_selected":len(base),
        "group_selected":len(cur),
        "intersection":len(base&cur),
        "base_only":len(base-cur),
        "group_only":len(cur-base),
        "jaccard":len(base&cur)/len(union) if union else None,
    }

def main():
    a=parse_args()
    run=gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}")
    jobs=gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}/jobs?per_page=100").get("jobs",[])
    folds=[]
    job_rows=[]
    for job in jobs:
        name=str(job.get("name") or "")
        if not name.startswith("blind-anatomy-y"):
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

    aggregate={}
    base10=None
    for group in GROUPS:
        aggregate[group]={
            "weighted_roc_auc":weighted(folds,group,"roc_auc"),
            "weighted_pr_auc":weighted(folds,group,"pr_auc"),
            "budgets":{b:aggregate_budget(folds,group,b) for b in BUDGETS},
            "yearly_10pct":[
                {
                    "year":f["test_year"],
                    "caught":f["groups"][group]["budgets"]["10%"]["caught_blind_spots"],
                    "blind_recall":f["groups"][group]["budgets"]["10%"]["blind_recall"],
                    "delta_vs_base_pp":f["delta_vs_base"][group]["recall_delta_pp_10pct"],
                    "caught_delta_vs_base":f["delta_vs_base"][group]["caught_delta_10pct"],
                    "selection_overlap_vs_base":overlap10(f,group),
                }
                for f in folds
            ],
        }
        if group=="BASE":
            base10=aggregate[group]["budgets"]["10%"]

    for group in GROUPS:
        x=aggregate[group]
        x10=x["budgets"]["10%"]
        x["aggregate_10pct_delta_vs_base_pp"]=100*((x10["blind_recall"] or 0)-(base10["blind_recall"] or 0))
        x["aggregate_10pct_caught_delta_vs_base"]=x10["caught_blind_spots"]-base10["caught_blind_spots"]
        if group=="BASE":
            x["positive_years_vs_base"]=0
            x["negative_years_vs_base"]=0
        else:
            deltas=[f["delta_vs_base"][group]["recall_delta_pp_10pct"] for f in folds]
            x["positive_years_vs_base"]=sum(d>1e-12 for d in deltas)
            x["negative_years_vs_base"]=sum(d<-1e-12 for d in deltas)

    summary={
        "contract":"L15_BLIND_SPOT_ANATOMY_V1_SUMMARY",
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
        "race_id_join_ready":True,
    }

    out=Path(a.out_root)/f"run-{a.run_id}"
    if out.exists():
        raise SystemExit(f"immutable output exists: {out}")
    out.mkdir(parents=True)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    def pct(v):
        return "" if v is None else f"{100*float(v):.2f}%"
    lines=[
        f"# L1.5 Blind-Spot Anatomy V1 — run {a.run_id}",
        "",
        f"- Run: {run.get('html_url')}",
        f"- Complete: {complete}",
        "- 2026 sealed / Odds NO",
        "- Standard CPU only / Artifact NO / Cache NO / New Kaggle persistence NO",
        "",
        "## 10% budget aggregate",
        "",
        "| Group | AUC | PR-AUC | Caught | Blind recall | Precision | Δ vs BASE | + years | - years |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        x=aggregate[group]
        b=x["budgets"]["10%"]
        lines.append(
            f"| {group} | {x['weighted_roc_auc']:.4f} | {x['weighted_pr_auc']:.4f} | "
            f"{b['caught_blind_spots']} | {pct(b['blind_recall'])} | {pct(b['precision'])} | "
            f"{x['aggregate_10pct_delta_vs_base_pp']:+.2f}pt | "
            f"{x['positive_years_vs_base']} | {x['negative_years_vs_base']} |"
        )
    lines += [
        "",
        "Fold-level univariate quintile tables and 10% race IDs are preserved in summary.json.",
        "No automatic production winner is declared; this is Gate V3 feature selection evidence.",
        "",
    ]
    (out/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print("L15_BLIND_SPOT_ANATOMY_V1_LEDGER_READY")
    print(json.dumps({
        "run_id":a.run_id,
        "complete":complete,
        "missing_years":sorted(expected-got),
        "path":str(out),
    },ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
