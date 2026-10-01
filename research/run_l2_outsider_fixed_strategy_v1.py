#!/usr/bin/env python3
import argparse,csv,gzip,itertools,json,time
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_l2_ticket_lab_v1 import load_market, write_csv
from run_l2_fixed_strategy_benchmark_v1 import STRATEGIES, tickets_for, max_drawdown
from build_l2_bet_kings_dataset_v1 import load_day, payout_map

YEARS=(2023,2024,2025)
PRIMARY_YEARS=(2024,2025)
SOURCES=("KING","OUTSIDER")

def args():
    p=argparse.ArgumentParser(description="Mirror fixed-strategy benchmark: Seven-King rank vs tie-safe 13-Outsider weighted council rank.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--ballots-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_year_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1); out[int(y)]=p
    if set(out)!=set(YEARS): raise SystemExit(f"ballot years mismatch: {sorted(out)}")
    return out

def load_outsider_orders(paths,market):
    # Exact safe weighted-council policy:
    # rank1=3, rank2=2, rank3=1; ties: top1 votes, support count,
    # top2 votes, top3 votes, horse_number, horse_id.
    by_market={(int(y),str(rid)):g.copy() for (y,rid),g in market.groupby(["year","race_id"],sort=False)}
    orders={}
    stats={}
    for year,path in sorted(paths.items()):
        votes=defaultdict(lambda:defaultdict(lambda:{"count":0,"weighted":0,"v1":0,"v2":0,"v3":0}))
        candidates=defaultdict(set)
        with gzip.open(path,"rt",encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rid=str(row["race_id"]); cand=str(row["candidate"])
                candidates[rid].add(cand)
                seen=set()
                for k,w in ((1,3),(2,2),(3,1)):
                    hid=str(row.get(f"top{k}_horse_id") or "")
                    if not hid: continue
                    if hid in seen: raise SystemExit(f"duplicate horse in candidate top3 y={year} race={rid} cand={cand}")
                    seen.add(hid)
                    d=votes[rid][hid]
                    d["count"]+=1; d["weighted"]+=w; d[f"v{k}"]+=1

        year_counts=[]
        for (yy,rid),g in by_market.items():
            if yy!=year: continue
            if len(candidates.get(rid,set()))!=13:
                raise SystemExit(f"expected 13 outsider candidates y={year} race={rid}, got={len(candidates.get(rid,set()))}")
            scored=[]
            for row in g.itertuples(index=False):
                hid=str(row.horse_id); num=int(row.horse_number)
                d=votes[rid].get(hid,{"count":0,"weighted":0,"v1":0,"v2":0,"v3":0})
                scored.append({
                    "horse_id":hid,"horse_number":num,
                    "count":int(d["count"]),"weighted":int(d["weighted"]),
                    "v1":int(d["v1"]),"v2":int(d["v2"]),"v3":int(d["v3"]),
                })
            scored.sort(key=lambda x:(-x["weighted"],-x["v1"],-x["count"],-x["v2"],-x["v3"],x["horse_number"],x["horse_id"]))
            orders[(year,rid)]={i+1:int(x["horse_number"]) for i,x in enumerate(scored)}
            year_counts.append(sum(x["weighted"]>0 for x in scored))
        stats[year]={
            "races":len(year_counts),
            "positive_vote_horses_min":min(year_counts) if year_counts else None,
            "positive_vote_horses_mean":float(np.mean(year_counts)) if year_counts else None,
            "positive_vote_horses_max":max(year_counts) if year_counts else None,
        }
    return orders,stats

def aggregate(rows,year,source,strategy):
    q=[r for r in rows if r["year"]==year and r["source"]==source and r["strategy"]==strategy]
    if not q: return None
    stake=sum(r["stake_yen"] for r in q); ret=sum(r["return_yen"] for r in q)
    points=[r["tickets"] for r in q]
    profits=[r["profit_yen"] for r in sorted(q,key=lambda x:(x["race_date"],x["race_id"]))]
    return {
        "year":year,"source":source,"strategy":strategy,
        "bet_type":STRATEGIES[strategy]["bet"],"label":STRATEGIES[strategy]["label"],
        "races":len(q),"tickets":sum(points),
        "avg_points":float(np.mean(points)),"median_points":float(np.median(points)),"max_points":max(points),
        "hit_races":sum(r["hit"] for r in q),
        "hit_rate_pct":100.0*sum(r["hit"] for r in q)/len(q),
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(profits),
    }

def main():
    a=args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    market=load_market(a.market_scored)
    market=market[market["year"].isin(YEARS)].copy()
    market["race_id"]=market["race_id"].astype(str)

    # KING rank is frozen consensus rank from the same market source.
    king_orders={}
    for (year,rid),g in market.groupby(["year","race_id"],sort=False):
        rr={}
        for row in g.itertuples(index=False):
            rank=int(row.consensus_rank); num=int(row.horse_number)
            if rank in rr: raise SystemExit(f"duplicate King rank y={year} race={rid} rank={rank}")
            rr[rank]=num
        king_orders[(int(year),str(rid))]=rr

    outsider_orders,outsider_stats=load_outsider_orders(parse_year_paths(a.ballots_year),market)
    if set(king_orders)!=set(outsider_orders):
        miss_king=set(outsider_orders)-set(king_orders)
        miss_out=set(king_orders)-set(outsider_orders)
        raise SystemExit(f"rank coverage mismatch king_missing={len(miss_king)} outsider_missing={len(miss_out)}")

    wanted={y:set(market.loc[market["year"]==y,"race_id"].astype(str)) for y in YEARS}
    root=Path(a.backfill_root)
    race_rows=[]; counters=defaultdict(int)

    for year in YEARS:
        seen_races=set()
        for day_path in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
            date=day_path.name[:10]
            day=load_day(day_path,wanted[year])
            if not day: continue
            for rid,pack in day.items():
                rid=str(rid); key=(year,rid)
                if key not in king_orders or key not in outsider_orders: continue
                payouts,present=payout_map(pack)
                seen_races.add(rid)
                for source,rank_to_num in (("KING",king_orders[key]),("OUTSIDER",outsider_orders[key])):
                    for strategy,meta in STRATEGIES.items():
                        bet=meta["bet"]
                        if bet not in present:
                            counters[f"missing_payout_{bet}"]+=1
                            continue
                        ts=tickets_for(strategy,rank_to_num)
                        if not ts:
                            counters[f"empty_{source}_{strategy}"]+=1
                            continue
                        ret=sum(float(payouts.get((bet,t),0.0)) for t in ts)
                        race_rows.append({
                            "year":year,"race_date":date,"race_id":rid,
                            "source":source,"strategy":strategy,"bet_type":bet,"label":meta["label"],
                            "tickets":len(ts),"stake_yen":100.0*len(ts),
                            "return_yen":ret,"profit_yen":ret-100.0*len(ts),"hit":int(ret>0),
                        })
        print("OUTSIDER_FIXED_YEAR_DONE "+json.dumps({
            "year":year,"races_seen":len(seen_races),
            "rows":sum(1 for r in race_rows if r["year"]==year)
        },separators=(",",":")),flush=True)

    yearly=[]
    for year in YEARS:
        for source in SOURCES:
            for s in STRATEGIES:
                r=aggregate(race_rows,year,source,s)
                if r: yearly.append(r)

    pooled=[]
    for source in SOURCES:
        for s in STRATEGIES:
            q=[r for r in race_rows if r["year"] in PRIMARY_YEARS and r["source"]==source and r["strategy"]==s]
            if not q: continue
            stake=sum(r["stake_yen"] for r in q); ret=sum(r["return_yen"] for r in q)
            profits=[r["profit_yen"] for r in sorted(q,key=lambda x:(x["year"],x["race_date"],x["race_id"]))]
            pooled.append({
                "years":"2024|2025","source":source,"strategy":s,
                "bet_type":STRATEGIES[s]["bet"],"label":STRATEGIES[s]["label"],
                "races":len(q),"tickets":sum(r["tickets"] for r in q),
                "avg_points":float(np.mean([r["tickets"] for r in q])),
                "hit_races":sum(r["hit"] for r in q),
                "hit_rate_pct":100.0*sum(r["hit"] for r in q)/len(q),
                "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
                "roi_pct":100.0*ret/stake if stake else None,
                "max_drawdown_yen":max_drawdown(profits),
            })

    # Exact apples-to-apples source delta for every strategy/year.
    pmap={(r["source"],r["strategy"]):r for r in pooled}
    deltas=[]
    for s in STRATEGIES:
        k=pmap.get(("KING",s)); o=pmap.get(("OUTSIDER",s))
        if not k or not o: continue
        deltas.append({
            "strategy":s,"bet_type":STRATEGIES[s]["bet"],"label":STRATEGIES[s]["label"],
            "king_hit_rate_pct":k["hit_rate_pct"],"outsider_hit_rate_pct":o["hit_rate_pct"],
            "delta_hit_rate_pp":o["hit_rate_pct"]-k["hit_rate_pct"],
            "king_roi_pct":k["roi_pct"],"outsider_roi_pct":o["roi_pct"],
            "delta_roi_pp":o["roi_pct"]-k["roi_pct"],
            "king_avg_points":k["avg_points"],"outsider_avg_points":o["avg_points"],
            "king_max_drawdown_yen":k["max_drawdown_yen"],"outsider_max_drawdown_yen":o["max_drawdown_yen"],
        })

    write_csv(out/"yearly-summary.csv",yearly)
    write_csv(out/"primary-2024-2025-summary.csv",pooled)
    write_csv(out/"king-vs-outsider.csv",deltas)
    if race_rows:
        with gzip.open(out/"race-detail.csv.gz","wt",newline="",encoding="utf-8") as fh:
            w=csv.DictWriter(fh,fieldnames=list(race_rows[0]))
            w.writeheader(); w.writerows(race_rows)

    summary={
        "contract":"L2_OUTSIDER_FIXED_STRATEGY_V1",
        "comparison":"Exact same six ticket shapes, same 100 yen/ticket, same races, KING rank vs OUTSIDER weighted council rank.",
        "outsider_rank_policy":"13 safe Outsiders: top1=3, top2=2, top3=1; ties: top1 votes, support count, top2 votes, top3 votes, horse_number, horse_id.",
        "strategy_contract":STRATEGIES,
        "stake_policy":"100 yen per generated ticket",
        "odds_gate":False,
        "threshold_search":False,
        "evaluation_years":list(YEARS),
        "primary_years":list(PRIMARY_YEARS),
        "outsider_stats":outsider_stats,
        "counters":dict(counters),
        "elapsed_seconds":time.perf_counter()-start,
        "promotion":False,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== KING VS OUTSIDER ====="); print((out/"king-vs-outsider.csv").read_text())
    print("===== PRIMARY ====="); print((out/"primary-2024-2025-summary.csv").read_text())
    print("L2_OUTSIDER_FIXED_STRATEGY_V1_READY")

if __name__=="__main__":
    main()
