#!/usr/bin/env python3
import argparse,gzip,json
from collections import defaultdict
from pathlib import Path

from rank_utils import RANK_TIE_POLICY, deterministic_ranks, tie_diagnostics

def open_text(path,mode="rt"):
    return gzip.open(path,mode,encoding="utf-8") if str(path).endswith(".gz") else open(path,mode,encoding="utf-8")

def parse_args():
    p=argparse.ArgumentParser(description="Re-rank persisted L1 horse-level scores without result-order dependence.")
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--summary",required=False)
    return p.parse_args()

def main():
    a=parse_args()
    races=defaultdict(list)
    with open_text(a.input) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            hid=str(row.get("horse_id") or "")
            if not rid or not hid:
                raise ValueError("score row missing race_id/horse_id")
            score=row.get("raw_margin_logit")
            if score is None:
                score=row.get("raw_win_probability")
            if score is None:
                raise ValueError("score row missing raw_margin_logit/raw_win_probability")
            races[rid].append(row)

    if not races:
        raise ValueError("empty score input")

    diagnostics={
        "contract":"L1_TIE_SAFE_RERANK_V1",
        "policy":RANK_TIE_POLICY,
        "races":len(races),
        "rows":0,
        "any_tie_races":0,
        "boundary_ties":{"1":0,"3":0,"6":0},
        "changed_rank_rows":0,
        "changed_top1_races":0,
        "changed_top3_races":0,
        "changed_top6_races":0,
    }
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    ordered_races=[]

    for rid in sorted(races):
        rows=races[rid]
        scores=[r.get("raw_margin_logit") if r.get("raw_margin_logit") is not None else r.get("raw_win_probability") for r in rows]
        horse_ids=[str(r["horse_id"]) for r in rows]
        ranks=deterministic_ranks(scores,horse_ids)
        tie=tie_diagnostics(scores,horse_ids)
        diagnostics["rows"]+=len(rows)
        diagnostics["any_tie_races"]+=int(tie["any_tie"])
        for n in ("1","3","6"):
            diagnostics["boundary_ties"][n]+=int(tie["boundary_tie"][n])

        old_top={}
        new_top={}
        for n in (1,3,6):
            old_top[n]={str(r["horse_id"]) for r in rows if int(r.get("predicted_rank") or 999)<=n}
            new_top[n]={str(r["horse_id"]) for r,rank in zip(rows,ranks) if rank<=n}
        diagnostics["changed_top1_races"]+=int(old_top[1]!=new_top[1])
        diagnostics["changed_top3_races"]+=int(old_top[3]!=new_top[3])
        diagnostics["changed_top6_races"]+=int(old_top[6]!=new_top[6])

        rewritten=[]
        for row,rank in zip(rows,ranks):
            old=int(row.get("predicted_rank") or 0)
            diagnostics["changed_rank_rows"]+=int(old!=rank)
            rec=dict(row)
            rec["predicted_rank"]=int(rank)
            summary=dict(rec.get("race_summary") or {})
            summary["rank_tie_policy"]=RANK_TIE_POLICY
            summary["rank_tie_group_count"]=tie["tie_group_count"]
            summary["rank_tied_horse_count"]=tie["tied_horse_count"]
            summary["rank_max_tie_group_size"]=tie["max_tie_group_size"]
            summary["rank_top1_boundary_tie"]=tie["boundary_tie"]["1"]
            summary["rank_top3_boundary_tie"]=tie["boundary_tie"]["3"]
            summary["rank_top6_boundary_tie"]=tie["boundary_tie"]["6"]
            rec["race_summary"]=summary
            rewritten.append(rec)
        rewritten.sort(key=lambda r:(int(r["predicted_rank"]),str(r["horse_id"])))
        ordered_races.append((rid,rewritten))

    with open_text(out,"wt") as fh:
        for _,rows in ordered_races:
            for row in rows:
                fh.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")

    for key in ("any_tie_races","changed_top1_races","changed_top3_races","changed_top6_races"):
        diagnostics[key+"_rate"]=diagnostics[key]/diagnostics["races"]
    diagnostics["changed_rank_row_rate"]=diagnostics["changed_rank_rows"]/diagnostics["rows"]
    diagnostics["boundary_tie_rates"]={k:v/diagnostics["races"] for k,v in diagnostics["boundary_ties"].items()}

    if a.summary:
        p=Path(a.summary); p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L1_TIE_SAFE_RERANK_OK")
    print(json.dumps(diagnostics,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
