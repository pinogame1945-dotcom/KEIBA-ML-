#!/usr/bin/env python3
import argparse
import csv
import heapq
import json
import multiprocessing as mp
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
BETA_GRID=tuple(i/20.0 for i in range(21))
EDGE_GRID=(-1.0,-0.75,-0.50,-0.25,-0.10,0.0,0.03,0.10,0.25,0.50)
TOPK_GRID=(1,2,4,8,12,20)
COVERAGE_FLOORS=(50.0,70.0,90.0)
FLAT_STAKE=100.0

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def empty_stat():
    return {"source_races":0,"races_bet":0,"tickets":0,"hits":0,"stake":0.0,"return":0.0}

def evaluate_bet_type(args):
    bet,paths=args
    # key=(beta,edge_min,topk), value totals + per-year totals
    totals={}
    by_year={}
    for beta in BETA_GRID:
        for edge in EDGE_GRID:
            for topk in TOPK_GRID:
                key=(beta,edge,topk)
                totals[key]=empty_stat()
                by_year[key]={y:empty_stat() for y in INSAMPLE_YEARS}

    max_k=max(TOPK_GRID)
    for y in INSAMPLE_YEARS:
        seen=0
        for rec in read_jsonl_gz(paths[y]):
            seen+=1
            no_index={int(n):i for i,n in enumerate(rec["horse_numbers"])}
            winners={ks:float(v) for ks,v in (rec["payouts"].get(bet) or [])}
            tickets=rec["tickets"].get(bet) or []
            for key in totals:
                if key[0]==0.0: # source race count only once per policy is updated below by beta loop
                    pass
            if not tickets:
                # Still count source race for every policy.
                for beta in BETA_GRID:
                    for edge in EDGE_GRID:
                        for topk in TOPK_GRID:
                            k=(beta,edge,topk)
                            totals[k]["source_races"]+=1
                            by_year[k][y]["source_races"]+=1
                continue

            for beta in BETA_GRID:
                p=blend(rec["l1_probability"],rec["win_odds"],beta)
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
                top=heapq.nlargest(max_k,scored,key=lambda z:(z[0],z[1],z[2]))
                for edge_min in EDGE_GRID:
                    eligible=[z for z in top if z[0]>=edge_min]
                    for topk in TOPK_GRID:
                        k=(beta,edge_min,topk)
                        st=totals[k]; ys=by_year[k][y]
                        st["source_races"]+=1; ys["source_races"]+=1
                        chosen=eligible[:topk]
                        if not chosen:
                            continue
                        st["races_bet"]+=1; ys["races_bet"]+=1
                        n=len(chosen)
                        st["tickets"]+=n; ys["tickets"]+=n
                        st["stake"]+=FLAT_STAKE*n; ys["stake"]+=FLAT_STAKE*n
                        r=sum(z[3] for z in chosen)
                        h=sum(1 for z in chosen if z[3]>0)
                        st["return"]+=r; ys["return"]+=r
                        st["hits"]+=h; ys["hits"]+=h
        print(f"BET_DONE bet={bet} year={y} races={seen}",flush=True)

    rows=[]
    for (beta,edge_min,topk),st in totals.items():
        stake=st["stake"]; ret=st["return"]; src=st["source_races"]
        row={
            "bet_type":bet,"beta":beta,"edge_min":edge_min,"topk":topk,
            **st,
            "roi_pct":100.0*ret/stake if stake else None,
            "profit_yen":ret-stake,
            "race_coverage_pct":100.0*st["races_bet"]/src if src else 0.0,
        }
        for y in INSAMPLE_YEARS:
            z=by_year[(beta,edge_min,topk)][y]
            row[f"y{y}_roi_pct"]=100.0*z["return"]/z["stake"] if z["stake"] else None
            row[f"y{y}_race_coverage_pct"]=100.0*z["races_bet"]/z["source_races"] if z["source_races"] else 0.0
            row[f"y{y}_profit_yen"]=z["return"]-z["stake"]
        rows.append(row)
    return bet,rows

def choose(rows):
    selected=[]
    for bet in BET_TYPES:
        local=[r for r in rows if r["bet_type"]==bet and r["roi_pct"] is not None]
        for floor in COVERAGE_FLOORS:
            eligible=[r for r in local if r["race_coverage_pct"]>=floor]
            met=bool(eligible)
            pool=eligible if met else local
            if met:
                pool=sorted(pool,key=lambda r:(r["roi_pct"],r["profit_yen"],r["race_coverage_pct"]),reverse=True)
            else:
                pool=sorted(pool,key=lambda r:(r["race_coverage_pct"],r["roi_pct"],r["profit_yen"]),reverse=True)
            best=dict(pool[0])
            best["coverage_floor_pct"]=floor
            best["coverage_floor_met"]=met
            selected.append(best)
    overall=[]
    for floor in COVERAGE_FLOORS:
        candidates=[r for r in selected if r["coverage_floor_pct"]==floor and r["coverage_floor_met"]]
        if candidates:
            candidates=sorted(candidates,key=lambda r:(r["roi_pct"],r["profit_yen"]),reverse=True)
            overall.append(dict(candidates[0]))
        else:
            candidates=[r for r in selected if r["coverage_floor_pct"]==floor]
            candidates=sorted(candidates,key=lambda r:(r["race_coverage_pct"],r["roi_pct"]),reverse=True)
            overall.append(dict(candidates[0]))
    return selected,overall

def main():
    p=argparse.ArgumentParser(description="Direct in-sample flat-stake profit ceiling. Global knobs only; no subgroup mining.")
    for y in INSAMPLE_YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--work-dir",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if a.workers<1 or a.workers>4:
        raise SystemExit("workers must be 1..4 on standard CPU runner")

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

    jobs=[(bet,compact) for bet in BET_TYPES]
    rows=[]
    if a.workers>1:
        ctx=mp.get_context("fork")
        with ctx.Pool(processes=min(a.workers,len(jobs))) as pool:
            for bet,r in pool.imap_unordered(evaluate_bet_type,jobs):
                print(f"GRID_DONE bet={bet} rows={len(r)}",flush=True)
                rows.extend(r)
    else:
        for job in jobs:
            bet,r=evaluate_bet_type(job); rows.extend(r)

    selected,overall=choose(rows)
    write_csv(out/"global_grid.csv",rows)
    write_csv(out/"selected_by_type.csv",selected)
    write_csv(out/"best_overall.csv",overall)

    result={
        "contract":"L3_INSAMPLE_PROFIT_CEILING_V2_RESULT",
        "purpose":"Deliberately overfit the same 2022-2025 outcomes using only global ticket-selection knobs and flat 100-yen staking.",
        "deployable_backtest":False,
        "overfit_intentional":True,
        "subgroup_mining_allowed":False,
        "individual_race_memorization_allowed":False,
        "insample_years":list(INSAMPLE_YEARS),
        "bet_types":list(BET_TYPES),
        "beta_grid":list(BETA_GRID),
        "edge_grid":list(EDGE_GRID),
        "topk_grid":list(TOPK_GRID),
        "coverage_floors_pct":list(COVERAGE_FLOORS),
        "flat_stake_yen":FLAT_STAKE,
        "global_knobs_only":["bet_type","beta","edge_min","topk"],
        "negative_model_edge_candidates_allowed":True,
        "bankroll_ruin_can_reduce_coverage":False,
        "source_races":total_source,
        "selected_by_type":selected,
        "best_overall":overall,
        "historical_market_timing":"FINAL_POSTHOC; deliberate in-sample existence test, not a live deployable strategy.",
        "2026_locked":True,
        "paid_compute":False,
        "runner":"ubuntu-latest standard CPU",
        "artifact_cache":False,
    }
    (out/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    main()
