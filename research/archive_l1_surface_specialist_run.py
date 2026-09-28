#!/usr/bin/env python3
import argparse,json,os,re,subprocess
from datetime import datetime,timezone
from pathlib import Path

MARKER="L1_SURFACE_SPECIALIST_RESULT"
ANSI_RE=re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TS_RE=re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s+")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True)
    p.add_argument("--run-id",required=True,type=int)
    p.add_argument("--config",default="research/l1-surface-specialists-v1.json")
    p.add_argument("--out-root",default="research-results/l1-surface-specialists")
    return p.parse_args()

def gh_json(endpoint):
    return json.loads(subprocess.check_output(["gh","api",endpoint],text=True))

def job_log(repo,job_id):
    token=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GH_TOKEN/GITHUB_TOKEN missing")
    api=os.environ.get("GITHUB_API_URL","https://api.github.com").rstrip("/")
    url=f"{api}/repos/{repo}/actions/jobs/{job_id}/logs"
    p=subprocess.run([
        "curl","-fsSL",
        "-H",f"Authorization: Bearer {token}",
        "-H","Accept: application/vnd.github+json",
        "-H","X-GitHub-Api-Version: 2022-11-28",
        url
    ],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if p.returncode:
        raise SystemExit(p.stderr.decode("utf-8",errors="replace"))
    return p.stdout.decode("utf-8",errors="replace")

def clean(s):
    return TS_RE.sub("",ANSI_RE.sub("",s).replace("\ufeff","")).strip()

def extract_all(log):
    lines=[clean(x) for x in log.splitlines()]
    out=[]
    seen=set()
    for i,line in enumerate(lines):
        if MARKER not in line:
            continue
        for row in lines[i+1:i+8]:
            at=row.find("{")
            if at<0:
                continue
            try:
                obj=json.loads(row[at:])
            except json.JSONDecodeError:
                continue
            if not (isinstance(obj,dict) and obj.get("candidate") and obj.get("surface") and obj.get("validation_year")):
                continue
            key=(str(obj["surface"]),str(obj["candidate"]),int(obj["validation_year"]))
            if key in seen:
                break
            seen.add(key)
            out.append(obj)
            break
    return out

def weighted(rows,key):
    pairs=[]
    for row in rows:
        m=row.get("metrics") or {}
        n=int(m.get("races") or 0)
        v=m.get(key)
        if n and v is not None:
            pairs.append((n,float(v)))
    total=sum(n for n,_ in pairs)
    return (sum(n*v for n,v in pairs)/total) if total else None

def fmt(v,pct=False):
    if v is None:
        return ""
    return f"{v*100:.2f}%" if pct else f"{v:.3f}"

def main():
    a=parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8"))
    years=[int(x) for x in cfg["validation_years"]]
    surfaces=list(cfg["surfaces"])
    candidates=[x["name"] for x in cfg["candidates"]]
    labels={x["name"]:x.get("label_ja",x["name"]) for x in cfg["candidates"]}
    expected={(s,c,y) for s in surfaces for c in candidates for y in years}

    run=gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}")
    jobs=gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}/jobs?per_page=100").get("jobs",[])
    rows=[]
    for job in jobs:
        if "specialist-" not in str(job.get("name","")):
            continue
        if job.get("conclusion")!="success":
            continue
        results=extract_all(job_log(a.repo,job["id"]))
        for result in results:
            result["job_id"]=job["id"]
            rows.append(result)

    got={(r["surface"],r["candidate"],int(r["validation_year"])) for r in rows}
    missing=sorted(
        [{"surface":s,"candidate":c,"validation_year":y} for s,c,y in expected-got],
        key=lambda x:(x["surface"],x["candidate"],x["validation_year"])
    )

    summaries={}
    for s in surfaces:
        summaries[s]={}
        for c in candidates:
            cr=[r for r in rows if r["surface"]==s and r["candidate"]==c]
            summaries[s][c]={
                "label_ja":labels[c],
                "races":sum(int((r.get("metrics") or {}).get("races") or 0) for r in cr),
                "top1_winner_capture":weighted(cr,"top1_winner_capture"),
                "top3_winner_capture":weighted(cr,"top3_winner_capture"),
                "top6_winner_capture":weighted(cr,"top6_winner_capture"),
                "mean_winner_rank":weighted(cr,"mean_winner_rank"),
                "mean_reciprocal_winner_rank":weighted(cr,"mean_reciprocal_winner_rank"),
                "race_normalized_nll":weighted(cr,"race_normalized_nll"),
            }

    exp=cfg["experiment_id"]
    run_dir=Path(a.out_root)/exp/"attempts"/f"run-{a.run_id}"
    if run_dir.exists():
        raise SystemExit(f"immutable ledger path exists: {run_dir}")
    run_dir.mkdir(parents=True)

    experiment={
        "contract":"L1_SURFACE_SPECIALIST_LEDGER_V1",
        "experiment_id":exp,
        "run_id":a.run_id,
        "run_url":run.get("html_url"),
        "head_branch":run.get("head_branch"),
        "head_sha":run.get("head_sha"),
        "archived_at":datetime.now(timezone.utc).isoformat(),
        "snapshot_generation":cfg["snapshot_generation"],
        "training_window_years":cfg["training_window_years"],
        "validation_years":years,
        "surfaces":surfaces,
        "ability_uses_odds":False,
        "locked_years":cfg["locked_years"],
        "expected_jobs":len(expected),
        "observed_jobs":len(rows),
        "missing_jobs":missing,
        "storage_policy":{
            "result_ledger":"git",
            "github_artifacts":False,
            "github_cache":False,
            "models":"ephemeral",
            "kaggle_snapshot":"read_only"
        }
    }
    scorecard={
        "contract":"L1_SURFACE_SPECIALIST_SCORECARD_V1",
        "experiment_id":exp,
        "run_id":a.run_id,
        "complete":not missing and len(rows)==len(expected),
        "fold_results":sorted(rows,key=lambda x:(x["surface"],x["candidate"],int(x["validation_year"]))),
        "surface_candidate_summaries":summaries
    }
    (run_dir/"experiment.json").write_text(json.dumps(experiment,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (run_dir/"scorecard.json").write_text(json.dumps(scorecard,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    md=[
        f"# {exp} — run {a.run_id}","",
        f"- Snapshot: {cfg['snapshot_generation']}",
        "- Method: previous 2 years SAME SURFACE train -> next year SAME SURFACE validation",
        "- Race class: ALL",
        "- 2026 lock: ON",
        "- L1 odds: OFF",
        "- Models are ephemeral; only small result ledgers are persisted to Git.",""
    ]
    for s in surfaces:
        md += [
            f"## {s} — five-year weighted summary","",
            "| Candidate | Races | Top1 | Top3 | Top6 | Mean winner rank |",
            "|---|---:|---:|---:|---:|---:|"
        ]
        for c in candidates:
            m=summaries[s][c]
            md.append(
                f"| {labels[c]} | {m['races']} | {fmt(m['top1_winner_capture'],True)} | "
                f"{fmt(m['top3_winner_capture'],True)} | {fmt(m['top6_winner_capture'],True)} | "
                f"{fmt(m['mean_winner_rank'])} |"
            )
        md.append("")
        for y in years:
            md += [
                f"### {s} {y}","",
                "| Candidate | Races | Top1 | Top3 | Top6 | Mean winner rank |",
                "|---|---:|---:|---:|---:|---:|"
            ]
            for c in candidates:
                row=next((r for r in rows if r["surface"]==s and r["candidate"]==c and int(r["validation_year"])==y),None)
                if not row:
                    continue
                m=row["metrics"]
                md.append(
                    f"| {labels[c]} | {m.get('races','')} | {fmt(m.get('top1_winner_capture'),True)} | "
                    f"{fmt(m.get('top3_winner_capture'),True)} | {fmt(m.get('top6_winner_capture'),True)} | "
                    f"{fmt(m.get('mean_winner_rank'))} |"
                )
            md.append("")

    (run_dir/"README.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print("L1_SURFACE_SPECIALIST_LEDGER_READY")
    print(json.dumps({
        "experiment_id":exp,
        "run_id":a.run_id,
        "path":str(run_dir),
        "complete":scorecard["complete"],
        "observed_jobs":len(rows),
        "expected_jobs":len(expected),
        "missing":missing
    },ensure_ascii=False))

if __name__=="__main__":
    main()
