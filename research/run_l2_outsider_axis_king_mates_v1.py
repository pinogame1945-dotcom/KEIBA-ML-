#!/usr/bin/env python3
import argparse,csv,gzip,itertools,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np

from run_l2_ticket_lab_v1 import load_market, write_csv
from run_l2_fixed_strategy_benchmark_v1 import max_drawdown
from run_l2_outsider_fixed_strategy_v1 import load_outsider_orders
from build_l2_bet_kings_dataset_v1 import load_day, payout_map, canonical_numbers

YEARS=(2023,2024,2025)
PRIMARY_YEARS=(2024,2025)
SOURCES=("KING","OUTSIDER","REVERSE_HYBRID")

STRATEGIES={
    "Q_AXIS1_5":{"bet":"QUINELLA","label":"馬連 1位軸→紐5頭","mate_n":5},
    "Q_TOP2_6":{"bet":"QUINELLA","label":"馬連 1・2位軸→紐6頭","mate_n":6},
    "T_AXIS1_7":{"bet":"TRIO","label":"三連複 1位軸＋紐7頭から2頭","mate_n":7},
    "T_AXIS12_6":{"bet":"TRIO","label":"三連複 1・2位固定＋紐6頭","mate_n":6},
}

def args():
    p=argparse.ArgumentParser(description="Reverse fixed hybrid benchmark: Outsider axes with KING mates.")
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

def ordered_numbers(rank_map):
    return [rank_map[r] for r in sorted(rank_map)]

def first_excluding(rank_map,n,excluded):
    out=[]
    for x in ordered_numbers(rank_map):
        if x in excluded or x in out: continue
        out.append(x)
        if len(out)>=n: break
    return out

def pure_tickets(rank,strategy):
    if strategy=="Q_AXIS1_5":
        if 1 not in rank: return []
        mates=[rank[r] for r in range(2,7) if r in rank]
        return [canonical_numbers("QUINELLA",(rank[1],m)) for m in mates if m!=rank[1]]
    if strategy=="Q_TOP2_6":
        axes=[rank[r] for r in (1,2) if r in rank]
        mates=[rank[r] for r in range(3,9) if r in rank]
        return sorted(set(canonical_numbers("QUINELLA",(a,m)) for a in axes for m in mates if a!=m))
    if strategy=="T_AXIS1_7":
        if 1 not in rank: return []
        mates=[rank[r] for r in range(2,9) if r in rank]
        return [canonical_numbers("TRIO",(rank[1],)+p) for p in itertools.combinations(mates,2)]
    if strategy=="T_AXIS12_6":
        if 1 not in rank or 2 not in rank: return []
        mates=[rank[r] for r in range(3,9) if r in rank]
        return [canonical_numbers("TRIO",(rank[1],rank[2],m)) for m in mates if m not in {rank[1],rank[2]}]
    raise ValueError(strategy)

def reverse_tickets(strategy,king,out):
    # Reverse HYBRID: Outsider is the axis source, KING is the mate source.
    if strategy=="Q_AXIS1_5":
        if 1 not in out: return [],0
        axes=[out[1]]
        mates=first_excluding(king,5,set(axes))
        return [canonical_numbers("QUINELLA",(axes[0],m)) for m in mates],len(mates)
    if strategy=="Q_TOP2_6":
        if 1 not in out or 2 not in out: return [],0
        axes=[out[1],out[2]]
        mates=first_excluding(king,6,set(axes))
        return sorted(set(canonical_numbers("QUINELLA",(a,m)) for a in axes for m in mates)),len(mates)
    if strategy=="T_AXIS1_7":
        if 1 not in out: return [],0
        axes=[out[1]]
        mates=first_excluding(king,7,set(axes))
        return [canonical_numbers("TRIO",(axes[0],)+p) for p in itertools.combinations(mates,2)],len(mates)
    if strategy=="T_AXIS12_6":
        if 1 not in out or 2 not in out: return [],0
        axes=[out[1],out[2]]
        mates=first_excluding(king,6,set(axes))
        return [canonical_numbers("TRIO",(axes[0],axes[1],m)) for m in mates],len(mates)
    raise ValueError(strategy)

def tickets_for(source,strategy,king,out):
    if source=="KING": return pure_tickets(king,strategy),None
    if source=="OUTSIDER": return pure_tickets(out,strategy),None
    if source=="REVERSE_HYBRID": return reverse_tickets(strategy,king,out)
    raise ValueError(source)

def aggregate(rows,year,source,strategy):
    q=[r for r in rows if r["year"]==year and r["source"]==source and r["strategy"]==strategy]
    if not q: return None
    stake=sum(r["stake_yen"] for r in q); ret=sum(r["return_yen"] for r in q)
    active=sum(r["active"] for r in q)
    profits=[r["profit_yen"] for r in sorted(q,key=lambda x:(x["race_date"],x["race_id"]))]
    return {
        "year":year,"source":source,"strategy":strategy,
        "bet_type":STRATEGIES[strategy]["bet"],"label":STRATEGIES[strategy]["label"],
        "races":len(q),"active_races":active,"coverage_pct":100.0*active/len(q),
        "tickets":sum(r["tickets"] for r in q),
        "avg_points":float(np.mean([r["tickets"] for r in q])),
        "hit_races":sum(r["hit"] for r in q),
        "hit_rate_pct":100.0*sum(r["hit"] for r in q)/len(q),
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(profits),
        "avg_mates":float(np.mean([r["mate_count"] for r in q if r["mate_count"] is not None])) if any(r["mate_count"] is not None for r in q) else None,
    }

def main():
    a=args(); outdir=Path(a.out_dir); outdir.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()

    market=load_market(a.market_scored)
    market=market[market["year"].isin(YEARS)].copy()
    market["race_id"]=market["race_id"].astype(str)

    king_orders={}
    for (year,rid),g in market.groupby(["year","race_id"],sort=False):
        rr={}
        for row in g.itertuples(index=False):
            rank=int(row.consensus_rank); num=int(row.horse_number)
            if rank in rr: raise SystemExit(f"duplicate KING rank y={year} race={rid} rank={rank}")
            rr[rank]=num
        king_orders[(int(year),str(rid))]=rr

    outsider_orders,outsider_stats=load_outsider_orders(parse_year_paths(a.ballots_year),market)
    if set(king_orders)!=set(outsider_orders): raise SystemExit("KING/OUTSIDER coverage mismatch")

    wanted={y:set(market.loc[market["year"]==y,"race_id"].astype(str)) for y in YEARS}
    root=Path(a.backfill_root)
    rows=[]; counters=defaultdict(int)

    for year in YEARS:
        seen=set()
        for day_path in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
            date=day_path.name[:10]
            day=load_day(day_path,wanted[year])
            if not day: continue
            for rid,pack in day.items():
                rid=str(rid); key=(year,rid)
                if key not in king_orders or key not in outsider_orders: continue
                payouts,present=payout_map(pack)
                seen.add(rid)
                king=king_orders[key]; outsider=outsider_orders[key]
                for source in SOURCES:
                    for strategy,meta in STRATEGIES.items():
                        bet=meta["bet"]
                        if bet not in present:
                            counters[f"missing_payout_{bet}"]+=1; continue
                        ts,mate_count=tickets_for(source,strategy,king,outsider)
                        if len(ts)!=len(set(ts)): raise SystemExit(f"duplicate tickets {year} {rid} {source} {strategy}")
                        active=int(len(ts)>0)
                        if not active: counters[f"zero_ticket_{source}_{strategy}"]+=1
                        ret=sum(float(payouts.get((bet,t),0.0)) for t in ts)
                        rows.append({
                            "year":year,"race_date":date,"race_id":rid,
                            "source":source,"strategy":strategy,"bet_type":bet,"label":meta["label"],
                            "tickets":len(ts),"active":active,"mate_count":mate_count,
                            "stake_yen":100.0*len(ts),"return_yen":ret,
                            "profit_yen":ret-100.0*len(ts),"hit":int(ret>0),
                        })
        print("REVERSE_HYBRID_YEAR_DONE "+json.dumps({"year":year,"races":len(seen)},separators=(",",":")),flush=True)

    yearly=[]
    for year in YEARS:
        for source in SOURCES:
            for s in STRATEGIES:
                r=aggregate(rows,year,source,s)
                if r: yearly.append(r)

    pooled=[]
    for source in SOURCES:
        for s in STRATEGIES:
            q=[r for r in rows if r["year"] in PRIMARY_YEARS and r["source"]==source and r["strategy"]==s]
            if not q: continue
            stake=sum(r["stake_yen"] for r in q); ret=sum(r["return_yen"] for r in q)
            active=sum(r["active"] for r in q)
            profits=[r["profit_yen"] for r in sorted(q,key=lambda x:(x["year"],x["race_date"],x["race_id"]))]
            pooled.append({
                "years":"2024|2025","source":source,"strategy":s,
                "bet_type":STRATEGIES[s]["bet"],"label":STRATEGIES[s]["label"],
                "races":len(q),"active_races":active,"coverage_pct":100.0*active/len(q),
                "tickets":sum(r["tickets"] for r in q),
                "avg_points":float(np.mean([r["tickets"] for r in q])),
                "hit_races":sum(r["hit"] for r in q),
                "hit_rate_pct":100.0*sum(r["hit"] for r in q)/len(q),
                "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
                "roi_pct":100.0*ret/stake if stake else None,
                "max_drawdown_yen":max_drawdown(profits),
                "avg_mates":float(np.mean([r["mate_count"] for r in q if r["mate_count"] is not None])) if any(r["mate_count"] is not None for r in q) else None,
            })

    pmap={(r["source"],r["strategy"]):r for r in pooled}
    compare=[]
    for s in STRATEGIES:
        k=pmap[("KING",s)]; o=pmap[("OUTSIDER",s)]; r=pmap[("REVERSE_HYBRID",s)]
        compare.append({
            "strategy":s,"bet_type":STRATEGIES[s]["bet"],"label":STRATEGIES[s]["label"],
            "king_hit_pct":k["hit_rate_pct"],"outsider_hit_pct":o["hit_rate_pct"],"reverse_hit_pct":r["hit_rate_pct"],
            "king_roi_pct":k["roi_pct"],"outsider_roi_pct":o["roi_pct"],"reverse_roi_pct":r["roi_pct"],
            "reverse_minus_king_hit_pp":r["hit_rate_pct"]-k["hit_rate_pct"],
            "reverse_minus_outsider_hit_pp":r["hit_rate_pct"]-o["hit_rate_pct"],
            "reverse_minus_king_roi_pp":r["roi_pct"]-k["roi_pct"],
            "reverse_minus_outsider_roi_pp":r["roi_pct"]-o["roi_pct"],
            "reverse_avg_points":r["avg_points"],"reverse_coverage_pct":r["coverage_pct"],
            "reverse_max_drawdown_yen":r["max_drawdown_yen"],
        })

    write_csv(outdir/"yearly-summary.csv",yearly)
    write_csv(outdir/"primary-2024-2025-summary.csv",pooled)
    write_csv(outdir/"three-way-comparison.csv",compare)

    if rows:
        with gzip.open(outdir/"race-detail.csv.gz","wt",newline="",encoding="utf-8") as fh:
            w=csv.DictWriter(fh,fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    summary={
        "contract":"L2_OUTSIDER_AXIS_KING_MATES_V1",
        "comparison":"KING-only and OUTSIDER-only baselines versus reverse hybrid: Outsider axes with KING mates.",
        "reverse_policy":"Outsider supplies axis horse(s); KING supplies mate horses, excluding only the Outsider axis horse(s).",
        "odds_gate":False,
        "threshold_search":False,
        "stake_policy":"100 yen per generated ticket",
        "evaluation_years":list(YEARS),
        "primary_years":list(PRIMARY_YEARS),
        "outsider_stats":outsider_stats,
        "counters":dict(counters),
        "elapsed_seconds":time.perf_counter()-start,
        "promotion":False,
        "2026_locked":True,
    }
    (outdir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== THREE WAY ====="); print((outdir/"three-way-comparison.csv").read_text())
    print("===== PRIMARY ====="); print((outdir/"primary-2024-2025-summary.csv").read_text())
    print("L2_OUTSIDER_AXIS_KING_MATES_V1_READY")

if __name__=="__main__":
    main()
