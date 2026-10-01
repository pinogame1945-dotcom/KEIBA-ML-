#!/usr/bin/env python3
import argparse,csv,gzip,itertools,json,math,time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_l2_ticket_lab_v1 import load_market, write_csv
from build_l2_bet_kings_dataset_v1 import load_day, load_odds_day, decode_odds, payout_map, canonical_numbers

YEARS=(2023,2024,2025)
PRIMARY_YEARS=(2024,2025)

STRATEGIES={
    "Q_AXIS1_TOP6":{"bet":"QUINELLA","label":"馬連 1位軸→2〜6位"},
    "Q_TOP2_TO8":{"bet":"QUINELLA","label":"馬連 1〜2位→3〜8位"},
    "Q_TOP6_BOX":{"bet":"QUINELLA","label":"馬連 Top6 BOX"},
    "T_AXIS1_TO8":{"bet":"TRIO","label":"三連複 1位軸＋2〜8位から2頭"},
    "T_AXIS12_TO8":{"bet":"TRIO","label":"三連複 1・2位固定＋3〜8位"},
    "T_TOP6_BOX":{"bet":"TRIO","label":"三連複 Top6 BOX"},
}

def args():
    p=argparse.ArgumentParser(description="Fixed human-defined quinella/trio strategy benchmark. No learned buy rules.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def max_drawdown(profits):
    eq=0.0; peak=0.0; dd=0.0
    for x in profits:
        eq+=float(x); peak=max(peak,eq); dd=max(dd,peak-eq)
    return dd

def tickets_for(strategy, rank_to_num):
    def nums(lo,hi):
        return [rank_to_num[r] for r in range(lo,hi+1) if r in rank_to_num]
    if strategy=="Q_AXIS1_TOP6":
        if 1 not in rank_to_num: return []
        return [canonical_numbers("QUINELLA",(rank_to_num[1],x)) for x in nums(2,6)]
    if strategy=="Q_TOP2_TO8":
        axes=nums(1,2); opp=nums(3,8)
        return sorted(set(canonical_numbers("QUINELLA",(a,b)) for a in axes for b in opp if a!=b))
    if strategy=="Q_TOP6_BOX":
        top=nums(1,6)
        return [canonical_numbers("QUINELLA",x) for x in itertools.combinations(top,2)]
    if strategy=="T_AXIS1_TO8":
        if 1 not in rank_to_num: return []
        opp=nums(2,8)
        return [canonical_numbers("TRIO",(rank_to_num[1],)+x) for x in itertools.combinations(opp,2)]
    if strategy=="T_AXIS12_TO8":
        if 1 not in rank_to_num or 2 not in rank_to_num: return []
        return [canonical_numbers("TRIO",(rank_to_num[1],rank_to_num[2],x)) for x in nums(3,8)]
    if strategy=="T_TOP6_BOX":
        top=nums(1,6)
        return [canonical_numbers("TRIO",x) for x in itertools.combinations(top,3)]
    raise ValueError(strategy)

def aggregate(rows,year,strategy):
    q=[r for r in rows if r["year"]==year and r["strategy"]==strategy]
    if not q: return None
    stake=sum(r["stake_yen"] for r in q); ret=sum(r["return_yen"] for r in q)
    profits=[r["profit_yen"] for r in sorted(q,key=lambda x:(x["race_date"],x["race_id"]))]
    points=[r["tickets"] for r in q]
    return {
        "year":year,"strategy":strategy,"bet_type":STRATEGIES[strategy]["bet"],"label":STRATEGIES[strategy]["label"],
        "races":len(q),"tickets":sum(points),"avg_points":float(np.mean(points)),
        "median_points":float(np.median(points)),"max_points":max(points),
        "hit_races":sum(r["hit"] for r in q),
        "hit_rate_pct":100.0*sum(r["hit"] for r in q)/len(q),
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(profits),
        "avg_return_on_hit_yen":float(np.mean([r["return_yen"] for r in q if r["hit"]])) if any(r["hit"] for r in q) else None,
    }

def paired_compare(summary_rows):
    rows=[]
    for bet in ("QUINELLA","TRIO"):
        names=[k for k,v in STRATEGIES.items() if v["bet"]==bet]
        for year in PRIMARY_YEARS:
            y=[r for r in summary_rows if r["year"]==year and r["strategy"] in names]
            for r in y:
                rows.append({
                    "year":year,"bet_type":bet,"strategy":r["strategy"],"label":r["label"],
                    "hit_rate_pct":r["hit_rate_pct"],"roi_pct":r["roi_pct"],
                    "avg_points":r["avg_points"],"max_drawdown_yen":r["max_drawdown_yen"],
                })
    return rows

def main():
    a=args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    market=load_market(a.market_scored)
    market=market[market["year"].isin(YEARS)].copy()
    market["race_id"]=market["race_id"].astype(str)

    by_race={(int(y),str(rid)):g.copy() for (y,rid),g in market.groupby(["year","race_id"],sort=False)}
    wanted={y:set(market.loc[market["year"]==y,"race_id"]) for y in YEARS}
    root=Path(a.backfill_root)

    race_rows=[]; counters=defaultdict(int)
    for year in YEARS:
        for day_path in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
            date=day_path.name[:10]
            odds_path=root/"data"/"odds"/"daily"/f"{date}.jsonl.gz"
            day=load_day(day_path,wanted[year])
            if not day: continue
            oddsday=load_odds_day(odds_path,set(day)) if odds_path.exists() else {}
            for rid,pack in day.items():
                g=by_race.get((year,str(rid)))
                if g is None: continue
                ranks={}
                bad=False
                for row in g.itertuples(index=False):
                    r=int(row.consensus_rank); n=int(row.horse_number)
                    if r in ranks or r<=0 or n<=0: bad=True; break
                    ranks[r]=n
                if bad or 1 not in ranks:
                    counters["bad_rank_map"]+=1; continue

                payouts,present=payout_map(pack)
                odds=decode_odds(oddsday.get(str(rid),{})) if str(rid) in oddsday else {}
                for strategy,meta in STRATEGIES.items():
                    bet=meta["bet"]
                    if bet not in present:
                        counters[f"missing_payout_{bet}"]+=1
                        continue
                    ts=tickets_for(strategy,ranks)
                    if not ts:
                        counters[f"empty_{strategy}"]+=1
                        continue
                    # Fixed strategy: every generated ticket gets 100 yen. Odds never gate selection.
                    ret=sum(float(payouts.get((bet,t),0.0)) for t in ts)
                    hit=int(ret>0)
                    winner_odds=[float(odds[(bet,t)]) for t in ts if (bet,t) in payouts and (bet,t) in odds]
                    race_rows.append({
                        "year":year,"race_date":date,"race_id":str(rid),
                        "strategy":strategy,"bet_type":bet,"label":meta["label"],
                        "field_size":len(ranks),"tickets":len(ts),
                        "stake_yen":100.0*len(ts),"return_yen":ret,
                        "profit_yen":ret-100.0*len(ts),"hit":hit,
                        "winning_ticket_odds":winner_odds[0] if winner_odds else None,
                    })
        print("FIXED_STRATEGY_YEAR_DONE "+json.dumps({
            "year":year,
            "races_by_strategy":{s:sum(1 for r in race_rows if r["year"]==year and r["strategy"]==s) for s in STRATEGIES}
        },separators=(",",":")),flush=True)

    summary=[]
    for year in YEARS:
        for s in STRATEGIES:
            r=aggregate(race_rows,year,s)
            if r: summary.append(r)

    # Pooled primary holdout-like years, with no tuning/selection.
    pooled=[]
    for s in STRATEGIES:
        q=[r for r in race_rows if r["year"] in PRIMARY_YEARS and r["strategy"]==s]
        if not q: continue
        stake=sum(r["stake_yen"] for r in q); ret=sum(r["return_yen"] for r in q)
        profits=[r["profit_yen"] for r in sorted(q,key=lambda x:(x["year"],x["race_date"],x["race_id"]))]
        pooled.append({
            "years":"2024|2025","strategy":s,"bet_type":STRATEGIES[s]["bet"],"label":STRATEGIES[s]["label"],
            "races":len(q),"tickets":sum(r["tickets"] for r in q),
            "avg_points":float(np.mean([r["tickets"] for r in q])),
            "hit_races":sum(r["hit"] for r in q),
            "hit_rate_pct":100.0*sum(r["hit"] for r in q)/len(q),
            "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
            "roi_pct":100.0*ret/stake if stake else None,
            "max_drawdown_yen":max_drawdown(profits),
        })

    write_csv(out/"yearly-summary.csv",summary)
    write_csv(out/"primary-2024-2025-summary.csv",pooled)
    write_csv(out/"side-by-side.csv",paired_compare(summary))
    if race_rows:
        with gzip.open(out/"race-detail.csv.gz","wt",newline="",encoding="utf-8") as fh:
            w=csv.DictWriter(fh,fieldnames=list(race_rows[0]))
            w.writeheader(); w.writerows(race_rows)

    result={
        "contract":"L2_FIXED_STRATEGY_BENCHMARK_V1",
        "philosophy":"Human defines ticket shape; AI/L1.75 supplies only the frozen horse ranking. No learned buy rule, no odds gate, no threshold search.",
        "strategies":STRATEGIES,
        "stake_policy":"100 yen per generated ticket",
        "odds_policy":"Odds never determine whether a ticket is bought.",
        "evaluation_years":list(YEARS),
        "primary_comparison_years":list(PRIMARY_YEARS),
        "payout_use":"evaluation only",
        "counters":dict(counters),
        "elapsed_seconds":time.perf_counter()-start,
        "promotion":False,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== YEARLY ====="); print((out/"yearly-summary.csv").read_text())
    print("===== PRIMARY ====="); print((out/"primary-2024-2025-summary.csv").read_text())
    print("L2_FIXED_STRATEGY_BENCHMARK_V1_READY")

if __name__=="__main__":
    main()
