#!/usr/bin/env python3
import argparse,json,os,re,statistics,subprocess
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

MARKER="L1_OUTSIDER_RESULT"
ANSI_RE=re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TS_RE=re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s+")

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True)
    p.add_argument("--run-id",required=True,type=int)
    p.add_argument("--config",default="research/l1-outsider-arena-v1.json")
    p.add_argument("--out-root",default="research-results/l1-outsider-arena-v1")
    return p.parse_args()

def gh_json(ep):
    return json.loads(subprocess.check_output(["gh","api",ep],text=True))

def gh_log(repo,job_id):
    token=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token: return "","token missing"
    api=os.environ.get("GITHUB_API_URL","https://api.github.com").rstrip("/")
    url=f"{api}/repos/{repo}/actions/jobs/{job_id}/logs"
    p=subprocess.run(["curl","-fsSL","-H",f"Authorization: Bearer {token}",
                      "-H","Accept: application/vnd.github+json",
                      "-H","X-GitHub-Api-Version: 2022-11-28",url],
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if p.returncode: return "",p.stderr.decode(errors="replace").strip()
    return p.stdout.decode(errors="replace"),None

def clean(s):
    return TS_RE.sub("",ANSI_RE.sub("",s).replace("\ufeff","")).strip()

def extract_all(log):
    lines=[clean(x) for x in log.splitlines()]
    out=[]
    seen=set()
    for i,line in enumerate(lines):
        if MARKER not in line: continue
        for row in lines[i+1:i+10]:
            at=row.find("{")
            if at<0: continue
            try: v=json.loads(row[at:])
            except json.JSONDecodeError: continue
            if isinstance(v,dict) and v.get("candidate") and v.get("validation_year"):
                key=(str(v.get("candidate")),int(v.get("validation_year")))
                if key not in seen:
                    out.append(v); seen.add(key)
                break
    return out

def error_excerpt(log,limit=12):
    out=[]
    for raw in log.splitlines():
        line=clean(raw); low=line.lower()
        if any(x in low for x in ("error","traceback","exception","failed","killed","out of memory","shutdown signal","operation was canceled")):
            out.append(line)
    return out[-limit:] or None

def mean(xs): return sum(xs)/len(xs) if xs else None

def outsider_overlap(rows, candidate_names, validation_years):
    out={}
    success=[r for r in rows if r.get("status")=="success"]
    for year in validation_years:
        year_rows=[r for r in success if int(r.get("validation_year") or 0)==int(year)]
        by_name={r.get("candidate"):r for r in year_rows}
        year_out={}
        for n in ("1","3","6"):
            rescue_sets={}
            class_sets={}
            for name in candidate_names:
                row=by_name.get(name) or {}
                rescue=row.get("rescue") or {}
                ids=(rescue.get("rescue_race_ids_by_topn") or {}).get(n) or []
                rescue_sets[name]=set(map(str,ids))
                raw_classes=(rescue.get("four_way_race_ids_by_topn") or {}).get(n) or {}
                class_sets[name]={
                    key:set(map(str,raw_classes.get(key) or []))
                    for key in ("both_hit","kings_only","outsider_only","both_miss")
                }

            union=set().union(*rescue_sets.values()) if rescue_sets else set()
            rescuers={}
            for rid in sorted(union):
                rescuers[rid]=[name for name in candidate_names if rid in rescue_sets[name]]
            exclusive={}
            for name in candidate_names:
                others=set().union(*(rescue_sets[x] for x in candidate_names if x!=name)) if len(candidate_names)>1 else set()
                exclusive[name]=len(rescue_sets[name]-others)
            pairs={}
            for i,left in enumerate(candidate_names):
                for right in candidate_names[i+1:]:
                    inter=rescue_sets[left]&rescue_sets[right]
                    uni=rescue_sets[left]|rescue_sets[right]
                    pairs[f"{left}__{right}"]={
                        "intersection_count":len(inter),
                        "union_count":len(uni),
                        "jaccard":len(inter)/len(uni) if uni else None,
                    }
            hist=defaultdict(int)
            for names in rescuers.values():
                hist[str(len(names))]+=1

            # Group-level four-way view: seven kings as one army vs every outsider as another.
            available=[name for name in candidate_names if any(class_sets[name].values())]
            group_vs_kings=None
            if available:
                first=available[0]
                universe=set().union(*class_sets[first].values())
                kings_hit=class_sets[first]["both_hit"]|class_sets[first]["kings_only"]
                for name in available[1:]:
                    u=set().union(*class_sets[name].values())
                    kh=class_sets[name]["both_hit"]|class_sets[name]["kings_only"]
                    if u!=universe:
                        raise SystemExit(f"{year} Top{n}: outsider classification universe mismatch")
                    if kh!=kings_hit:
                        raise SystemExit(f"{year} Top{n}: seven-king hit set changed across candidates")
                outsider_hit=set().union(*[
                    class_sets[name]["both_hit"]|class_sets[name]["outsider_only"]
                    for name in available
                ])
                grouped={
                    "both_hit":kings_hit&outsider_hit,
                    "kings_only":kings_hit-outsider_hit,
                    "outsiders_only":outsider_hit-kings_hit,
                    "both_miss":universe-(kings_hit|outsider_hit),
                }
                if sum(len(v) for v in grouped.values())!=len(universe):
                    raise SystemExit(f"{year} Top{n}: group four-way count mismatch")
                group_vs_kings={
                    "races":len(universe),
                    "counts":{key:len(value) for key,value in grouped.items()},
                    "race_ids":{key:sorted(value) for key,value in grouped.items()},
                }

            year_out[n]={
                "candidate_rescue_counts":{name:len(rescue_sets[name]) for name in candidate_names},
                "union_rescue_count":len(union),
                "union_rescue_race_ids":sorted(union),
                "exclusive_rescue_counts":exclusive,
                "rescuer_count_histogram":dict(sorted(hist.items(),key=lambda x:int(x[0]))),
                "rescuers_by_race":rescuers,
                "pairwise":pairs,
                "group_vs_kings":group_vs_kings,
            }
        out[str(year)]=year_out
    return out

def aggregate(rows):
    ok=[r for r in rows if r.get("status")=="success"]
    agg={"successful_folds":len(ok),"expected_folds":len(rows),"mean_metrics":{},"topn":{},"max_peak_rss_mib":None,
         "strong_rebellion_top1":{"count":0,"hit_count":0,"miss_count":0,"hit_rate":None,"miss_rate":None}}
    metrics=("top1_winner_capture","top3_winner_capture","top6_winner_capture","mean_winner_rank","race_normalized_nll","raw_roc_auc")
    for m in metrics:
        vals=[float(r["metrics"][m]) for r in ok if r.get("metrics",{}).get(m) is not None]
        agg["mean_metrics"][m]=mean(vals)
        agg["stdev_"+m]=statistics.pstdev(vals) if len(vals)>1 else (0.0 if vals else None)
    peaks=[]
    for r in ok:
        ru=r.get("resource_usage") or {}
        p=ru.get("candidate_peak_rss_mib")
        if p is not None: peaks.append(float(p))
    agg["max_peak_rss_mib"]=max(peaks) if peaks else None
    for n in ("1","3","6"):
        races=blind=rescues=hits=unionhits=0
        jacc=[]
        for r in ok:
            x=(r.get("rescue") or {}).get("topn",{}).get(n) or {}
            races+=int(x.get("races") or 0)
            blind+=int(x.get("seven_blind_spot_count") or 0)
            rescues+=int(x.get("rescue_count") or 0)
            hits+=int(x.get("candidate_hit_count") or 0)
            unionhits+=int(x.get("seven_union_hit_count") or 0)
            if x.get("mean_jaccard_vs_seven_union") is not None and x.get("races"):
                jacc.append((float(x["mean_jaccard_vs_seven_union"]),int(x["races"])))
        denom=sum(w for _,w in jacc)
        agg["topn"][n]={
            "races":races,"candidate_hits":hits,"seven_union_hits":unionhits,
            "seven_blind_spots":blind,"rescues":rescues,
            "rescue_rate_on_seven_blind_spots":rescues/blind if blind else None,
            "rescue_share_of_candidate_hits":rescues/hits if hits else None,
            "weighted_mean_jaccard_vs_seven_union":sum(v*w for v,w in jacc)/denom if denom else None,
        }
    rebellion_count=rebellion_hits=rebellion_misses=0
    for r in ok:
        x=(r.get("rescue") or {}).get("strong_rebellion_top1") or {}
        rebellion_count+=int(x.get("count") or 0)
        rebellion_hits+=int(x.get("hit_count") or 0)
        rebellion_misses+=int(x.get("miss_count") or 0)
    agg["strong_rebellion_top1"]={
        "count":rebellion_count,
        "hit_count":rebellion_hits,
        "miss_count":rebellion_misses,
        "hit_rate":rebellion_hits/rebellion_count if rebellion_count else None,
        "miss_rate":rebellion_misses/rebellion_count if rebellion_count else None,
    }
    return agg

def main():
    a=args()
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8"))
    exp=cfg["experiment_id"]
    expected={(c["name"],int(y)) for c in cfg["candidates"] for y in cfg["validation_years"]}
    labels={c["name"]:c.get("label_ja",c["name"]) for c in cfg["candidates"]}
    run=gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}")
    jobs=gh_json(f"/repos/{a.repo}/actions/runs/{a.run_id}/jobs?per_page=100").get("jobs",[])
    rows=[]; logs=[]
    for job in jobs:
        name=str(job.get("name") or "")
        if "outsider-" not in name: continue
        log,err=gh_log(a.repo,job["id"])
        results=extract_all(log)
        rec={"job_id":job.get("id"),"name":name,"status":job.get("status"),"conclusion":job.get("conclusion"),
             "started_at":job.get("started_at"),"completed_at":job.get("completed_at"),"log_fetch_error":err}
        if results:
            for result in results:
                rows.append({**result,"status":"success","job_id":job.get("id")})
            rec["structured_results"]=results
            # A candidate runner is fail-stop; if it failed after some years, the remaining
            # expected folds stay missing instead of being invented as successful.
            if job.get("conclusion")!="success":
                rec["error_excerpt"]=error_excerpt(log)
        else:
            m=re.search(r"outsider-(outsider_[a-z]+)(?:-all-years|-y(20\d{2}))?",name)
            cand=m.group(1) if m else name
            year=int(m.group(2)) if m and m.group(2) else None
            if year is not None:
                rows.append({"candidate":cand,"label_ja":labels.get(cand,cand),"validation_year":year,
                             "status":job.get("conclusion") or job.get("status"),"metrics":{},"rescue":{},
                             "job_id":job.get("id"),"error_excerpt":error_excerpt(log),"log_fetch_error":err})
            rec["error_excerpt"]=error_excerpt(log)
        logs.append(rec)

    got={(r.get("candidate"),r.get("validation_year")) for r in rows}
    missing=sorted([{"candidate":c,"validation_year":y} for c,y in expected-got],key=lambda x:(x["candidate"],x["validation_year"]))
    summaries={}
    for c in cfg["candidates"]:
        name=c["name"]
        subset=sorted([r for r in rows if r.get("candidate")==name],key=lambda x:x.get("validation_year") or 0)
        summaries[name]={"label_ja":labels[name],**aggregate(subset)}

    run_dir=Path(a.out_root)/exp/"attempts"/f"run-{a.run_id}"
    if run_dir.exists(): raise SystemExit(f"immutable ledger path exists: {run_dir}")
    run_dir.mkdir(parents=True)
    experiment={
        "contract":"L1_OUTSIDER_ARENA_LEDGER_V1","experiment_id":exp,"run_id":a.run_id,
        "run_url":run.get("html_url"),"head_branch":run.get("head_branch"),"head_sha":run.get("head_sha"),
        "archived_at":datetime.now(timezone.utc).isoformat(),"snapshot_generation":cfg["snapshot_generation"],
        "validation_years":cfg["validation_years"],"candidates":[c["name"] for c in cfg["candidates"]],
        "ability_uses_odds":False,"locked_years":cfg["locked_years"],"expected_jobs":len(expected),
        "observed_jobs":len(rows),"missing_jobs":missing,
        "storage_policy":{"raw_logs":"github_actions","structured_logs":"git_json","github_artifacts":False,
                          "github_cache":False,"candidate_models":"ephemeral","candidate_scores":"ephemeral"}
    }
    complete=(not missing and len(rows)==len(expected) and all(r.get("status")=="success" for r in rows))
    candidate_names=[c["name"] for c in cfg["candidates"]]
    score={"contract":"L1_OUTSIDER_ARENA_SCORECARD_V2","experiment_id":exp,"run_id":a.run_id,
           "complete":complete,"composite_winner":None,
           "fold_results":sorted(rows,key=lambda x:(x.get("candidate",""),x.get("validation_year") or 0)),
           "candidate_summaries":summaries,
           "outsider_overlap":outsider_overlap(rows,candidate_names,cfg["validation_years"])}
    (run_dir/"experiment.json").write_text(json.dumps(experiment,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (run_dir/"scorecard.json").write_text(json.dumps(score,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (run_dir/"logs.json").write_text(json.dumps({"run_id":a.run_id,"jobs":logs},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    lines=[f"# {exp} — run {a.run_id}","",f"- Run: {run.get('html_url')}",f"- Code SHA: {run.get('head_sha')}",
           f"- Snapshot: {cfg['snapshot_generation']}",f"- {len(cfg['candidates'])} outsider families × {len(cfg['validation_years'])} validation years","- 2026 sealed",
           "- Odds in L1: NO","- Raw logs: GitHub Actions","- Structured logs: logs.json","",
           "## Summary","",
           "| Candidate | Success | Mean Top1 | Mean Top3 | Mean Top6 | Mean rank | Top6 blind spots | Top6 rescues | Rescue rate |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cfg["candidates"]:
        s=summaries[c["name"]]; m=s["mean_metrics"]; t=s["topn"]["6"]
        def f(v): return "" if v is None else (f"{v:.6f}" if isinstance(v,float) else str(v))
        lines.append("| "+" | ".join([c["label_ja"],f"{s['successful_folds']}/{s['expected_folds']}",
            f(m.get("top1_winner_capture")),f(m.get("top3_winner_capture")),f(m.get("top6_winner_capture")),
            f(m.get("mean_winner_rank")),f(t.get("seven_blind_spots")),f(t.get("rescues")),
            f(t.get("rescue_rate_on_seven_blind_spots"))])+" |")
    lines += ["","## Top6 outsider overlap by year","",
              "| Year | Union rescues |", "|---:|---:|"]
    for year in cfg["validation_years"]:
        y=score["outsider_overlap"].get(str(year),{}).get("6",{})
        lines.append(f"| {year} | {y.get('union_rescue_count',0)} |")
    lines += ["","## Top6 seven kings vs outsider army","",
              "| Year | Both hit | Kings only | Outsiders only | Both miss |",
              "|---:|---:|---:|---:|---:|"]
    for year in cfg["validation_years"]:
        g=(score["outsider_overlap"].get(str(year),{}).get("6",{}).get("group_vs_kings") or {})
        counts=g.get("counts") or {}
        lines.append(
            f"| {year} | {counts.get('both_hit',0)} | {counts.get('kings_only',0)} | "
            f"{counts.get('outsiders_only',0)} | {counts.get('both_miss',0)} |"
        )
    lines += ["","## Interpretation","",
              "No automatic winner is assigned. A useful outsider may be weaker standalone if it consistently rescues seven-king blind spots on untouched years.",
              "Promotion still requires downstream L2/L3 EV/ROI validation.",""]
    (run_dir/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print("L1_OUTSIDER_LEDGER_READY")
    print(json.dumps({"run_id":a.run_id,"path":str(run_dir),"complete":complete,"missing":missing},ensure_ascii=False))

if __name__=="__main__":
    main()
