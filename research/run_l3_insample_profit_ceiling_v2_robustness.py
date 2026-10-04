#!/usr/bin/env python3
import argparse
import csv
import heapq
import json
from collections import defaultdict
from pathlib import Path

from run_l3_insample_profit_ceiling_v1 import (
    BET_TYPES,
    EXPECTED_RACES,
    blend,
    parse_key,
    prepare_year,
    read_jsonl_gz,
    ticket_prob,
)

INSAMPLE_YEARS=(2022,2023,2024,2025)
REMOVE_COUNTS=(0,1,5,10,20)
FLAT_STAKE=100.0
V2_MAX_K=20

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def load_coverage90_policies(path):
    js=json.loads(Path(path).read_text(encoding="utf-8"))
    rows=[r for r in js["selected_by_type"] if float(r["coverage_floor_pct"])==90.0]
    if {r["bet_type"] for r in rows} != set(BET_TYPES):
        raise SystemExit("coverage90 policy set incomplete")
    policies={}
    for r in rows:
        policies[r["bet_type"]]={
            "bet_type":r["bet_type"],
            "beta":float(r["beta"]),
            "edge_min":float(r["edge_min"]),
            "topk":int(r["topk"]),
            "expected_source_races":int(r["source_races"]),
            "expected_races_bet":int(r["races_bet"]),
            "expected_tickets":int(r["tickets"]),
            "expected_hits":int(r["hits"]),
            "expected_stake":float(r["stake"]),
            "expected_return":float(r["return"]),
            "expected_roi_pct":float(r["roi_pct"]),
            "expected_coverage_pct":float(r["race_coverage_pct"]),
        }
    return js,policies

def evaluate_policy(paths,policy):
    bet=policy["bet_type"]
    beta=policy["beta"]
    edge_min=policy["edge_min"]
    topk=policy["topk"]
    stats={"source_races":0,"races_bet":0,"tickets":0,"hits":0,"stake":0.0,"return":0.0}
    by_year={y:{"source_races":0,"races_bet":0,"tickets":0,"hits":0,"stake":0.0,"return":0.0} for y in INSAMPLE_YEARS}
    hit_rows=[]

    for y in INSAMPLE_YEARS:
        for rec in read_jsonl_gz(paths[y]):
            stats["source_races"]+=1
            by_year[y]["source_races"]+=1
            tickets=rec["tickets"].get(bet) or []
            if not tickets:
                continue
            no_index={int(n):i for i,n in enumerate(rec["horse_numbers"])}
            p=blend(rec["l1_probability"],rec["win_odds"],beta)
            winners={ks:float(v) for ks,v in (rec["payouts"].get(bet) or [])}
            scored=[]
            for ks,odd0 in tickets:
                odd=float(odd0)
                if odd<=1.0:
                    continue
                key=parse_key(ks)
                pr=ticket_prob(bet,key,no_index,p)
                if pr<=0.0:
                    continue
                edge=pr*odd-1.0
                ret=winners.get(ks,0.0)
                scored.append((edge,pr,odd,ret,ks))
            top=heapq.nlargest(V2_MAX_K,scored,key=lambda z:(z[0],z[1],z[2]))
            chosen=[z for z in top if z[0]>=edge_min][:topk]
            if not chosen:
                continue
            stats["races_bet"]+=1
            by_year[y]["races_bet"]+=1
            for edge,pr,odd,ret,ks in chosen:
                stats["tickets"]+=1; by_year[y]["tickets"]+=1
                stats["stake"]+=FLAT_STAKE; by_year[y]["stake"]+=FLAT_STAKE
                stats["return"]+=ret; by_year[y]["return"]+=ret
                if ret>0:
                    stats["hits"]+=1; by_year[y]["hits"]+=1
                    hit_rows.append({
                        "bet_type":bet,
                        "year":y,
                        "race_date":rec["race_date"],
                        "race_id":rec["race_id"],
                        "ticket":ks,
                        "odds":odd,
                        "model_probability":pr,
                        "model_edge":edge,
                        "return_per100":ret,
                    })
    return stats,by_year,hit_rows

def verify(policy,stats):
    checks={
        "source_races":(stats["source_races"],policy["expected_source_races"]),
        "races_bet":(stats["races_bet"],policy["expected_races_bet"]),
        "tickets":(stats["tickets"],policy["expected_tickets"]),
        "hits":(stats["hits"],policy["expected_hits"]),
        "stake":(stats["stake"],policy["expected_stake"]),
        "return":(stats["return"],policy["expected_return"]),
    }
    bad={k:v for k,v in checks.items() if abs(float(v[0])-float(v[1]))>1e-6}
    if bad:
        raise SystemExit(f"V2 reproduction mismatch bet={policy['bet_type']} bad={bad}")

def robustness_rows(policy,stats,hit_rows):
    hits=sorted(hit_rows,key=lambda r:(r["return_per100"],r["year"],r["race_id"]),reverse=True)
    out=[]
    base_return=stats["return"]; stake=stats["stake"]
    cumulative=0.0
    break_even_after=None
    for i,h in enumerate(hits,1):
        cumulative+=h["return_per100"]
        if break_even_after is None and base_return-cumulative < stake:
            break_even_after=i
    if base_return < stake:
        break_even_after=0

    for n in REMOVE_COUNTS:
        removed=hits[:min(n,len(hits))]
        removed_return=sum(x["return_per100"] for x in removed)
        adjusted_return=base_return-removed_return
        out.append({
            "bet_type":policy["bet_type"],
            "beta":policy["beta"],
            "edge_min":policy["edge_min"],
            "topk":policy["topk"],
            "remove_top_hits":n,
            "base_hits":stats["hits"],
            "remaining_hits":stats["hits"]-len(removed),
            "stake":stake,
            "base_return":base_return,
            "removed_return":removed_return,
            "removed_return_share_pct":100.0*removed_return/base_return if base_return else 0.0,
            "adjusted_return":adjusted_return,
            "adjusted_profit":adjusted_return-stake,
            "adjusted_roi_pct":100.0*adjusted_return/stake if stake else None,
            "hits_removed_to_fall_below_100_roi":break_even_after,
        })
    return out,hits,break_even_after

def by_year_robustness_rows(policy,by_year,hit_rows):
    out=[]
    for y in INSAMPLE_YEARS:
        st=by_year[y]
        hits=sorted([r for r in hit_rows if r["year"]==y],key=lambda r:r["return_per100"],reverse=True)
        for n in REMOVE_COUNTS:
            removed=hits[:min(n,len(hits))]
            removed_return=sum(x["return_per100"] for x in removed)
            adj=st["return"]-removed_return
            out.append({
                "bet_type":policy["bet_type"],
                "year":y,
                "remove_top_hits":n,
                "races_bet":st["races_bet"],
                "hits":st["hits"],
                "stake":st["stake"],
                "base_return":st["return"],
                "removed_return":removed_return,
                "adjusted_return":adj,
                "adjusted_profit":adj-st["stake"],
                "adjusted_roi_pct":100.0*adj/st["stake"] if st["stake"] else None,
            })
    return out

def main():
    p=argparse.ArgumentParser(description="Outlier sensitivity for fixed V2 broad-coverage policies.")
    for y in INSAMPLE_YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--work-dir",required=True)
    p.add_argument("--source-summary",required=True)
    p.add_argument("--out-dir",required=True)
    a=p.parse_args()

    source,policies=load_coverage90_policies(a.source_summary)
    work=Path(a.work_dir); work.mkdir(parents=True,exist_ok=True)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    compact={}; prep=[]
    for y in INSAMPLE_YEARS:
        path=work/f"y{y}.jsonl.gz"; compact[y]=path
        meta=prepare_year(y,getattr(a,f"l17_{y}"),a.backfill_root,path)
        prep.append(meta)
        print("PREPARED",meta,flush=True)
    total_source=sum(x["races"] for x in prep)
    if total_source<EXPECTED_RACES*len(INSAMPLE_YEARS)-40:
        raise SystemExit(f"pooled source coverage too low races={total_source}")

    robustness=[]
    by_year_rows=[]
    all_hits=[]
    summaries=[]
    for bet in BET_TYPES:
        policy=policies[bet]
        stats,by_year,hits=evaluate_policy(compact,policy)
        verify(policy,stats)
        rows,ordered,break_even=robustness_rows(policy,stats,hits)
        robustness.extend(rows)
        by_year_rows.extend(by_year_robustness_rows(policy,by_year,hits))
        for rank,row in enumerate(ordered,1):
            x=dict(row); x["payout_rank"]=rank
            all_hits.append(x)
        summaries.append({
            "bet_type":bet,
            "beta":policy["beta"],
            "edge_min":policy["edge_min"],
            "topk":policy["topk"],
            **stats,
            "roi_pct":100.0*stats["return"]/stats["stake"] if stats["stake"] else None,
            "profit_yen":stats["return"]-stats["stake"],
            "race_coverage_pct":100.0*stats["races_bet"]/stats["source_races"] if stats["source_races"] else 0.0,
            "hits_removed_to_fall_below_100_roi":break_even,
            "max_hit_return_per100":ordered[0]["return_per100"] if ordered else 0.0,
            "top5_return_share_pct":100.0*sum(x["return_per100"] for x in ordered[:5])/stats["return"] if stats["return"] else 0.0,
            "top10_return_share_pct":100.0*sum(x["return_per100"] for x in ordered[:10])/stats["return"] if stats["return"] else 0.0,
        })

    write_csv(out/"policy_summary.csv",summaries)
    write_csv(out/"robustness.csv",robustness)
    write_csv(out/"by_year_robustness.csv",by_year_rows)
    write_csv(out/"top_hits.csv",all_hits)

    result={
        "contract":"L3_INSAMPLE_PROFIT_CEILING_V2_ROBUSTNESS_RESULT",
        "purpose":"Remove the largest realized winning tickets from the fixed V2 90%-coverage policies without re-optimizing anything.",
        "source_contract":source.get("contract"),
        "source_summary":str(a.source_summary),
        "reoptimized":False,
        "remove_counts":list(REMOVE_COUNTS),
        "flat_stake_yen":FLAT_STAKE,
        "insample_years":list(INSAMPLE_YEARS),
        "source_races":total_source,
        "policies":summaries,
        "robustness":robustness,
        "2026_locked":True,
        "paid_compute":False,
        "runner":"ubuntu-latest standard CPU",
        "artifact_cache":False,
    }
    (out/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
