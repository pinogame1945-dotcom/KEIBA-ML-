#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ANSI_RE = re.compile(r"\\x1b\\[[0-9;]*[A-Za-z]")
TIMESTAMP_RE = re.compile(r"^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?Z\\s+")
RESULT_MARKER = "L1_WALKFORWARD_RESULT"
RESOURCE_MARKERS = ("FEATURE_ARENA_RESOURCE_USAGE ", "FEATURE_ARENA_RESOURCE_SAMPLE ")
METRICS = (
    "top1_winner_capture",
    "top3_winner_capture",
    "top6_winner_capture",
    "mean_winner_rank",
    "mean_reciprocal_winner_rank",
    "race_normalized_nll",
    "raw_logloss",
    "raw_brier",
    "raw_roc_auc",
)

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True)
    p.add_argument("--run-id",required=True,type=int)
    p.add_argument("--config",default="research/l1-final-walkforward-v1.json")
    p.add_argument("--out-root",default="research-results/l1-final-walkforward")
    return p.parse_args()

def gh_json(endpoint):
    return json.loads(subprocess.check_output(["gh","api",endpoint],text=True))

def gh_log(repo,job_id):
    token=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        return "","GH_TOKEN/GITHUB_TOKEN missing"
    api=os.environ.get("GITHUB_API_URL","https://api.github.com").rstrip("/")
    url=f"{api}/repos/{repo}/actions/jobs/{job_id}/logs"
    p=subprocess.run(
        ["curl","-fsSL","-H",f"Authorization: Bearer {token}",
         "-H","Accept: application/vnd.github+json",
         "-H","X-GitHub-Api-Version: 2022-11-28",url],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,
    )
    if p.returncode:
        return "",p.stderr.decode("utf-8",errors="replace").strip()
    return p.stdout.decode("utf-8",errors="replace"),None

def clean(line):
    return TIMESTAMP_RE.sub("",ANSI_RE.sub("",line).replace("\ufeff","")).strip()

def extract_result(log):
    lines=[clean(x) for x in (log or "").splitlines()]
    for i,line in enumerate(lines):
        if RESULT_MARKER not in line:
            continue
        for row in lines[i+1:i+8]:
            at=row.find("{")
            if at<0: continue
            try: value=json.loads(row[at:])
            except json.JSONDecodeError: continue
            if isinstance(value,dict) and value.get("candidate") and value.get("validation_year"):
                return value
    return None

def extract_resource(log):
    items=[]
    for raw in (log or "").splitlines():
        line=clean(raw)
        for marker in RESOURCE_MARKERS:
            at=line.find(marker)
            if at<0: continue
            try: value=json.loads(line[at+len(marker):].strip())
            except json.JSONDecodeError: break
            if isinstance(value,dict): items.append(value)
            break
    peaks=[float(x["peak_rss_mib"]) for x in items if x.get("peak_rss_mib") is not None]
    return {"observed_peak_rss_mib":max(peaks) if peaks else None,"samples":items} if items else None

def failed_step(job):
    for step in job.get("steps") or []:
        if step.get("conclusion")=="failure":
            return step.get("name")
    return None

def short_error(log,limit=10):
    rows=[]
    for raw in (log or "").splitlines():
        line=clean(raw)
        low=line.lower()
        if any(x in low for x in ("error","traceback","exception","failed","shutdown signal","operation was canceled","killed","out of memory")):
            rows.append(line)
    return rows[-limit:] or None

def mean(values):
    return float(sum(values)/len(values)) if values else None

def aggregate(records, expected_years):
    success=[r for r in records if r.get("status")=="success"]
    out={
        "expected_folds":len(expected_years),
        "successful_folds":len(success),
        "failed_folds":len(expected_years)-len(success),
        "years_success":[r["validation_year"] for r in success],
        "years_failed":[y for y in expected_years if y not in {r["validation_year"] for r in success}],
        "max_peak_rss_mib":None,
        "mean_metrics":{},
        "stdev_metrics":{},
        "worst_year_metrics":{},
        "best_year_metrics":{},
    }
    peaks=[r.get("peak_rss_mib") for r in success if r.get("peak_rss_mib") is not None]
    out["max_peak_rss_mib"]=max(peaks) if peaks else None
    for key in METRICS:
        vals=[float(r["metrics"][key]) for r in success if r.get("metrics",{}).get(key) is not None]
        out["mean_metrics"][key]=mean(vals)
        out["stdev_metrics"][key]=float(statistics.pstdev(vals)) if len(vals)>1 else (0.0 if vals else None)
        if vals:
            if key in ("mean_winner_rank","race_normalized_nll","raw_logloss","raw_brier"):
                out["best_year_metrics"][key]=min(vals)
                out["worst_year_metrics"][key]=max(vals)
            else:
                out["best_year_metrics"][key]=max(vals)
                out["worst_year_metrics"][key]=min(vals)
        else:
            out["best_year_metrics"][key]=None
            out["worst_year_metrics"][key]=None
    return out

def fmt(v):
    if v is None: return ""
    if isinstance(v,float): return f"{v:.6f}"
    return str(v)

def main():
    a=args()
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8"))
    exp=cfg["experiment_id"]
    expected_years=[int(x) for x in cfg["validation_years"]]
    expected_candidates=[x["name"] for x in cfg["candidates"]]
    labels={x["name"]:x.get("label_ja",x["name"]) for x in cfg["candidates"]}
    roles={x["name"]:x.get("role") for x in cfg["candidates"]}

    run=gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}")
    jobs_payload=gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}/jobs?per_page=100")
    jobs=[j for j in jobs_payload.get("jobs",[]) if "league-" in str(j.get("name",""))]

    rows=[]
    jobs_record=[]
    for job in sorted(jobs,key=lambda x:str(x.get("name",""))):
        log,log_error=gh_log(a.repo,job["id"])
        result=extract_result(log)
        resource=extract_resource(log)
        jobs_record.append({
            "job_id":job.get("id"),"name":job.get("name"),"status":job.get("status"),
            "conclusion":job.get("conclusion"),"started_at":job.get("started_at"),
            "completed_at":job.get("completed_at"),"failed_step":failed_step(job),
            "log_fetch_error":log_error,
        })
        if result:
            ru=result.get("resource_usage") or {}
            rows.append({
                "candidate":result["candidate"],
                "label_ja":labels.get(result["candidate"],result["candidate"]),
                "role":roles.get(result["candidate"]),
                "validation_year":int(result["validation_year"]),
                "train_start":result.get("train_start"),
                "train_end":result.get("train_end"),
                "status":"success",
                "feature_count":result.get("feature_count"),
                "metrics":result.get("metrics") or {},
                "peak_rss_mib":ru.get("candidate_peak_rss_mib") or (resource or {}).get("observed_peak_rss_mib"),
                "resource_usage":ru,
                "job_id":job.get("id"),
            })
        else:
            m=re.search(r"league-(.+?)-y(20\\d{2})(?:$|\\s|\\/)",str(job.get("name","")))
            candidate=m.group(1) if m else str(job.get("name",""))
            year=int(m.group(2)) if m else None
            rows.append({
                "candidate":candidate,"label_ja":labels.get(candidate,candidate),
                "role":roles.get(candidate),"validation_year":year,
                "status":job.get("conclusion") or job.get("status"),
                "feature_count":None,"metrics":{},
                "peak_rss_mib":(resource or {}).get("observed_peak_rss_mib"),
                "resource_usage":resource,
                "failed_step":failed_step(job),"error_excerpt":short_error(log),
                "log_fetch_error":log_error,"job_id":job.get("id"),
            })

    expected={(c,y) for c in expected_candidates for y in expected_years}
    got={(r["candidate"],r["validation_year"]) for r in rows if r.get("validation_year")}
    missing=sorted([{"candidate":c,"validation_year":y} for c,y in expected-got],key=lambda x:(x["candidate"],x["validation_year"]))

    summaries={}
    for candidate in expected_candidates:
        summaries[candidate]=aggregate(
            sorted([r for r in rows if r["candidate"]==candidate],key=lambda x:x.get("validation_year") or 0),
            expected_years,
        )
        summaries[candidate]["label_ja"]=labels.get(candidate,candidate)
        summaries[candidate]["role"]=roles.get(candidate)

    run_dir=Path(a.out_root)/exp/"attempts"/f"run-{a.run_id}"
    if run_dir.exists():
        raise SystemExit(f"immutable ledger path exists: {run_dir}")
    run_dir.mkdir(parents=True)

    experiment={
        "contract":"L1_FINAL_WALKFORWARD_LEDGER_V1",
        "experiment_id":exp,"run_id":a.run_id,"run_url":run.get("html_url"),
        "workflow":run.get("name"),"head_branch":run.get("head_branch"),
        "head_sha":run.get("head_sha"),"run_attempt":run.get("run_attempt"),
        "run_conclusion":run.get("conclusion"),"archived_at":datetime.now(timezone.utc).isoformat(),
        "snapshot_generation":cfg.get("snapshot_generation"),
        "training_window_years":cfg.get("training_window_years"),
        "validation_years":expected_years,"ability_uses_odds":False,
        "locked_years":cfg.get("locked_years") or [2026],
        "expected_candidates":expected_candidates,
        "expected_jobs":len(expected),"observed_jobs":len(rows),"missing_jobs":missing,
        "storage_policy":{"result_ledger":"git","github_artifacts":False,"github_cache":False,"models":"ephemeral"},
    }
    scorecard={
        "contract":"L1_FINAL_WALKFORWARD_SCORECARD_V1",
        "experiment_id":exp,"run_id":a.run_id,
        "complete":not missing and all((r.get("status")=="success") for r in rows) and len(rows)==len(expected),
        "composite_winner":None,
        "fold_results":sorted(rows,key=lambda x:(x["candidate"],x.get("validation_year") or 0)),
        "candidate_summaries":summaries,
    }
    (run_dir/"experiment.json").write_text(json.dumps(experiment,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (run_dir/"scorecard.json").write_text(json.dumps(scorecard,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (run_dir/"jobs.json").write_text(json.dumps({"run_id":a.run_id,"jobs":jobs_record},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    lines=[
        f"# {exp} — run {a.run_id}","",
        f"- Run: {run.get('html_url')}",
        f"- Code SHA: {run.get('head_sha')}",
        f"- Snapshot generation: {cfg.get('snapshot_generation')}",
        f"- Rolling train window: {cfg.get('training_window_years')} years",
        f"- Validation years: {', '.join(map(str,expected_years))}",
        "- 2026 lock: ON","- L1 odds: OFF",
        "- Composite winner: NONE","",
        "## Candidate summary","",
        "| Candidate | Role | Success | Peak MiB | Mean Top1 | Mean Top3 | Mean Top6 | Mean rank | Mean NLL | Mean AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in expected_candidates:
        s=summaries[candidate]; m=s["mean_metrics"]
        lines.append("| "+" | ".join([
            s["label_ja"],str(s.get("role") or ""),f"{s['successful_folds']}/{s['expected_folds']}",
            fmt(s.get("max_peak_rss_mib")),fmt(m.get("top1_winner_capture")),
            fmt(m.get("top3_winner_capture")),fmt(m.get("top6_winner_capture")),
            fmt(m.get("mean_winner_rank")),fmt(m.get("race_normalized_nll")),
            fmt(m.get("raw_roc_auc")),
        ])+" |")
    lines += ["","## Fold results","",
        "| Candidate | Year | Status | Features | Peak MiB | Top1 | Top3 | Top6 | Mean rank | NLL | AUC |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows,key=lambda x:(x["candidate"],x.get("validation_year") or 0)):
        m=r.get("metrics") or {}
        lines.append("| "+" | ".join([
            r.get("label_ja",r["candidate"]),fmt(r.get("validation_year")),str(r.get("status")),
            fmt(r.get("feature_count")),fmt(r.get("peak_rss_mib")),fmt(m.get("top1_winner_capture")),
            fmt(m.get("top3_winner_capture")),fmt(m.get("top6_winner_capture")),
            fmt(m.get("mean_winner_rank")),fmt(m.get("race_normalized_nll")),fmt(m.get("raw_roc_auc")),
        ])+" |")
    failures=[r for r in rows if r.get("status")!="success"]
    if failures:
        lines += ["","## Failures",""]
        for r in failures:
            lines += [f"### {r.get('label_ja',r['candidate'])} / {r.get('validation_year')}","",
                      f"- Status: {r.get('status')}",f"- Peak MiB: {fmt(r.get('peak_rss_mib'))}"]
            if r.get("failed_step"): lines.append(f"- Failed step: {r['failed_step']}")
            if r.get("error_excerpt"): lines += ["- Error excerpt:","","    "+"\n    ".join(r["error_excerpt"])]
            lines.append("")
    lines += ["","## Interpretation policy","",
              "No composite winner is assigned automatically. Compare cross-year stability, metrics, completion, memory, and later 2026 confirmation separately.",""]
    (run_dir/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print("L1_WALKFORWARD_LEDGER_READY")
    print(json.dumps({"experiment_id":exp,"run_id":a.run_id,"path":str(run_dir),"jobs":len(rows),"missing":missing,"complete":scorecard["complete"]},ensure_ascii=False))

if __name__=="__main__":
    main()
