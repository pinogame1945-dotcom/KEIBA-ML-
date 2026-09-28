#!/usr/bin/env python3
import argparse,json,os,re,subprocess
from datetime import datetime,timezone
from pathlib import Path

MARKER="L1_SURFACE_RESULT"
ANSI_RE=re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TS_RE=re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s+")

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True)
    p.add_argument("--run-id",required=True,type=int)
    p.add_argument("--config",default="research/l1-surface-routing-v1.json")
    p.add_argument("--out-root",default="research-results/l1-surface-routing")
    return p.parse_args()

def gh_json(endpoint):
    return json.loads(subprocess.check_output(["gh","api",endpoint],text=True))

def gh_log(repo,job_id):
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

def extract(log):
    lines=[clean(x) for x in log.splitlines()]
    for i,line in enumerate(lines):
        if MARKER not in line:
            continue
        for row in lines[i+1:i+8]:
            at=row.find("{")
            if at<0:
                continue
            try:
                v=json.loads(row[at:])
            except json.JSONDecodeError:
                continue
            if isinstance(v,dict) and v.get("candidate") and v.get("validation_year"):
                return v
    return None

def weighted(rows,surface,key):
    pairs=[]
    for row in rows:
        m=(row.get("surfaces") or {}).get(surface) or {}
        races=int(m.get("races") or 0)
        value=m.get(key)
        if races and value is not None:
            pairs.append((races,float(value)))
    total=sum(n for n,_ in pairs)
    return (sum(n*v for n,v in pairs)/total) if total else None

def fmt(v,pct=False):
    if v is None:
        return ""
    return f"{v*100:.2f}%" if pct else f"{v:.3f}"

def main():
    a=args()
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8"))
    exp=cfg["experiment_id"]
    years=[int(x) for x in cfg["validation_years"]]
    candidates=[x["name"] for x in cfg["candidates"]]
    labels={x["name"]:x.get("label_ja",x["name"]) for x in cfg["candidates"]}
    surfaces=list(cfg["surfaces"])

    run=gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}")
    payload=gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}/jobs?per_page=100")
    jobs=[j for j in payload.get("jobs",[]) if "surface-" in str(j.get("name",""))]

    rows=[]
    for job in jobs:
        result=extract(gh_log(a.repo,job["id"]))
        if result:
            result["status"]="success"
            result["job_id"]=job["id"]
            rows.append(result)

    expected={(c,y) for c in candidates for y in years}
    got={(r["candidate"],int(r["validation_year"])) for r in rows}
    missing=sorted(
        [{"candidate":c,"validation_year":y} for c,y in expected-got],
        key=lambda x:(x["candidate"],x["validation_year"])
    )

    summaries={}
    for c in candidates:
        cr=[r for r in rows if r["candidate"]==c]
        summaries[c]={"label_ja":labels[c],"surfaces":{}}
        for s in surfaces:
            summaries[c]["surfaces"][s]={
                "races":sum(int(((r.get("surfaces") or {}).get(s) or {}).get("races") or 0) for r in cr),
                "top1_winner_capture":weighted(cr,s,"top1_winner_capture"),
                "top3_winner_capture":weighted(cr,s,"top3_winner_capture"),
                "top6_winner_capture":weighted(cr,s,"top6_winner_capture"),
                "mean_winner_rank":weighted(cr,s,"mean_winner_rank"),
                "mean_reciprocal_winner_rank":weighted(cr,s,"mean_reciprocal_winner_rank"),
                "race_normalized_nll":weighted(cr,s,"race_normalized_nll"),
            }

    run_dir=Path(a.out_root)/exp/"attempts"/f"run-{a.run_id}"
    if run_dir.exists():
        raise SystemExit(f"immutable ledger path exists: {run_dir}")
    run_dir.mkdir(parents=True)

    experiment={
        "contract":"L1_SURFACE_ROUTING_LEDGER_V1",
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
        "locked_years":cfg.get("locked_years") or [2026],
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
        "contract":"L1_SURFACE_ROUTING_SCORECARD_V1",
        "experiment_id":exp,
        "run_id":a.run_id,
        "complete":not missing and len(rows)==len(expected),
        "fold_results":sorted(rows,key=lambda x:(x["candidate"],int(x["validation_year"]))),
        "candidate_summaries":summaries
    }

    (run_dir/"experiment.json").write_text(json.dumps(experiment,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (run_dir/"scorecard.json").write_text(json.dumps(scorecard,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    md=[
        f"# {exp} — run {a.run_id}","",
        f"- Snapshot: {cfg['snapshot_generation']}",
        "- Method: exact 2-year rolling train -> next-year validation",
        "- 2026 lock: ON",
        "- L1 odds: OFF",
        "- Purpose: compare existing L1 candidates by TURF / DIRT validation performance","",
        "## Five-year surface summary","",
        "| Candidate | Surface | Races | Top1 | Top3 | Top6 | Mean winner rank |",
        "|---|---|---:|---:|---:|---:|---:|"
    ]
    for c in candidates:
        for s in surfaces:
            m=summaries[c]["surfaces"][s]
            md.append(
                f"| {labels[c]} | {s} | {m['races']} | "
                f"{fmt(m['top1_winner_capture'],True)} | {fmt(m['top3_winner_capture'],True)} | "
                f"{fmt(m['top6_winner_capture'],True)} | {fmt(m['mean_winner_rank'])} |"
            )

    md += ["","## Annual surface results",""]
    for y in years:
        md += [f"### {y}","",
               "| Candidate | Surface | Races | Top1 | Top3 | Top6 | Mean winner rank |",
               "|---|---|---:|---:|---:|---:|---:|"]
        for c in candidates:
            row=next((r for r in rows if r["candidate"]==c and int(r["validation_year"])==y),None)
            if not row:
                continue
            for s in surfaces:
                m=(row.get("surfaces") or {}).get(s) or {}
                md.append(
                    f"| {labels[c]} | {s} | {m.get('races','')} | "
                    f"{fmt(m.get('top1_winner_capture'),True)} | {fmt(m.get('top3_winner_capture'),True)} | "
                    f"{fmt(m.get('top6_winner_capture'),True)} | {fmt(m.get('mean_winner_rank'))} |"
                )
        md.append("")

    (run_dir/"README.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print("L1_SURFACE_LEDGER_READY")
    print(json.dumps({
        "experiment_id":exp,
        "run_id":a.run_id,
        "path":str(run_dir),
        "complete":scorecard["complete"],
        "jobs":len(rows),
        "missing":missing
    },ensure_ascii=False))

if __name__=="__main__":
    main()
