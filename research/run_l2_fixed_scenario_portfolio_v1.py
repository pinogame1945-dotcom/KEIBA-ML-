#!/usr/bin/env python3
import argparse,csv,gzip,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np

from run_l2_ticket_lab_v1 import load_market, write_csv
from run_l2_fixed_strategy_benchmark_v1 import max_drawdown
from run_l2_outsider_fixed_strategy_v1 import load_outsider_orders
from build_l2_bet_kings_dataset_v1 import load_day, payout_map, canonical_numbers

YEARS=(2023,2024,2025)
PRIMARY_YEARS=(2024,2025)

# Human-defined, frozen portfolio. No search, no odds gate.
# MAINLINE: 5 quinella tickets = 500 yen
# HOLE_MIX: 3 trio tickets = 300 yen
# REVERSE: 2 quinella tickets = 200 yen
PORTFOLIO_BUDGET_YEN=1000

def args():
    p=argparse.ArgumentParser(description="Fixed three-scenario race portfolio benchmark.")
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

def scenario_tickets(king,out):
    # 1) MAINLINE: KING1 -> KING2..6, quinella, 5 x 100 = 500.
    if 1 not in king:
        return {"MAINLINE":[],"HOLE_MIX":[],"REVERSE":[]}
    mainline=[
        ("QUINELLA",canonical_numbers("QUINELLA",(king[1],king[r])),100)
        for r in range(2,7) if r in king and king[r]!=king[1]
    ]

    # 2) HOLE_MIX: KING1+KING2 fixed, third horse from Outsider.
    # Exclude KING1..3 so Outsider contributes genuinely different horses.
    hole=[]
    if 2 in king:
        excluded={king[r] for r in range(1,4) if r in king}
        mates=first_excluding(out,3,excluded)
        hole=[
            ("TRIO",canonical_numbers("TRIO",(king[1],king[2],m)),100)
            for m in mates
        ]

    # 3) REVERSE: Outsider1 as axis, top two distinct KING mates, quinella.
    reverse=[]
    if 1 in out:
        mates=first_excluding(king,2,{out[1]})
        reverse=[
            ("QUINELLA",canonical_numbers("QUINELLA",(out[1],m)),100)
            for m in mates
        ]

    return {"MAINLINE":mainline,"HOLE_MIX":hole,"REVERSE":reverse}

def summarize(rows,years,label):
    q=[r for r in rows if r["year"] in years]
    if not q: return None
    stake=sum(r["stake_yen"] for r in q); ret=sum(r["return_yen"] for r in q)
    profits=[r["profit_yen"] for r in sorted(q,key=lambda x:(x["year"],x["race_date"],x["race_id"]))]
    returns=np.array([r["return_yen"] for r in q],dtype=float)
    stakes=np.array([r["stake_yen"] for r in q],dtype=float)
    full=sum(r["stake_yen"]==PORTFOLIO_BUDGET_YEN for r in q)
    return {
        "period":label,
        "races":len(q),
        "full_1000_races":full,
        "full_1000_rate_pct":100.0*full/len(q),
        "avg_stake_yen":float(stakes.mean()),
        "avg_return_yen":float(returns.mean()),
        "median_return_yen":float(np.median(returns)),
        "p90_return_yen":float(np.quantile(returns,0.90)),
        "p95_return_yen":float(np.quantile(returns,0.95)),
        "any_hit_races":sum(r["any_hit"] for r in q),
        "any_hit_rate_pct":100.0*sum(r["any_hit"] for r in q)/len(q),
        "profitable_races":sum(r["return_yen"]>r["stake_yen"] for r in q),
        "profitable_race_rate_pct":100.0*sum(r["return_yen"]>r["stake_yen"] for r in q)/len(q),
        "break_even_or_better_races":sum(r["return_yen"]>=r["stake_yen"] for r in q),
        "break_even_or_better_rate_pct":100.0*sum(r["return_yen"]>=r["stake_yen"] for r in q)/len(q),
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(profits),
    }

def scenario_summary(rows,years,label,scenario):
    q=[r for r in rows if r["year"] in years]
    stake=sum(r[f"{scenario.lower()}_stake_yen"] for r in q)
    ret=sum(r[f"{scenario.lower()}_return_yen"] for r in q)
    return {
        "period":label,"scenario":scenario,"races":len(q),
        "avg_stake_yen":stake/len(q) if q else None,
        "hit_races":sum(r[f"{scenario.lower()}_hit"] for r in q),
        "hit_rate_pct":100.0*sum(r[f"{scenario.lower()}_hit"] for r in q)/len(q) if q else None,
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
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
                sc=scenario_tickets(king_orders[key],outsider_orders[key])

                row={"year":year,"race_date":date,"race_id":rid}
                total_stake=0.0; total_return=0.0
                all_tickets=[]
                for scenario in ("MAINLINE","HOLE_MIX","REVERSE"):
                    stake=0.0; ret=0.0; hit=0
                    for bet,ticket,unit in sc[scenario]:
                        if bet not in present:
                            counters[f"missing_{scenario}_{bet}"]+=1
                            continue
                        stake+=unit
                        payout100=float(payouts.get((bet,ticket),0.0))
                        ticket_return=payout100*(unit/100.0)
                        ret+=ticket_return
                        hit=max(hit,int(ticket_return>0))
                        all_tickets.append((scenario,bet,ticket,unit,ticket_return))
                    row[f"{scenario.lower()}_tickets"]=len(sc[scenario])
                    row[f"{scenario.lower()}_stake_yen"]=stake
                    row[f"{scenario.lower()}_return_yen"]=ret
                    row[f"{scenario.lower()}_hit"]=hit
                    total_stake+=stake; total_return+=ret

                # Duplicate ticket across scenarios is allowed: it means intentionally
                # allocating additional stake to the same combination.
                row["scenario_ticket_count"]=sum(len(sc[x]) for x in sc)
                row["unique_ticket_count"]=len(set((b,t) for _,b,t,_,_ in all_tickets))
                row["stake_yen"]=total_stake
                row["return_yen"]=total_return
                row["profit_yen"]=total_return-total_stake
                row["any_hit"]=int(total_return>0)
                row["exact_budget"]=int(total_stake==PORTFOLIO_BUDGET_YEN)
                rows.append(row)

        print("PORTFOLIO_YEAR_DONE "+json.dumps({"year":year,"races":len(seen)},separators=(",",":")),flush=True)

    summary_rows=[]
    for y in YEARS:
        s=summarize(rows,(y,),str(y))
        if s: summary_rows.append(s)
    pooled=summarize(rows,PRIMARY_YEARS,"2024|2025")
    if pooled: summary_rows.append(pooled)

    scenario_rows=[]
    for label,years in [("2023",(2023,)),("2024",(2024,)),("2025",(2025,)),("2024|2025",PRIMARY_YEARS)]:
        for sc in ("MAINLINE","HOLE_MIX","REVERSE"):
            scenario_rows.append(scenario_summary(rows,years,label,sc))

    write_csv(outdir/"portfolio-summary.csv",summary_rows)
    write_csv(outdir/"scenario-summary.csv",scenario_rows)

    if rows:
        with gzip.open(outdir/"race-detail.csv.gz","wt",newline="",encoding="utf-8") as fh:
            w=csv.DictWriter(fh,fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    summary={
        "contract":"L2_FIXED_SCENARIO_PORTFOLIO_V1",
        "portfolio":{
            "MAINLINE":"KING1 -> KING2..6 quinella, 5 tickets x 100 = 500 yen",
            "HOLE_MIX":"KING1+KING2 fixed + top3 Outsider mates after excluding KING1..3, trio, 3 x 100 = 300 yen",
            "REVERSE":"Outsider1 axis -> top2 distinct KING mates, quinella, 2 x 100 = 200 yen",
            "target_budget_yen":PORTFOLIO_BUDGET_YEN
        },
        "design":"Human-defined fixed scenario portfolio. No model search, no odds gate, no threshold tuning, no race selection.",
        "odds_gate":False,
        "threshold_search":False,
        "race_selection":False,
        "evaluation_years":list(YEARS),
        "primary_years":list(PRIMARY_YEARS),
        "outsider_stats":outsider_stats,
        "counters":dict(counters),
        "elapsed_seconds":time.perf_counter()-start,
        "promotion":False,
        "2026_locked":True,
    }
    (outdir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== PORTFOLIO =====")
    print((outdir/"portfolio-summary.csv").read_text())
    print("===== SCENARIOS =====")
    print((outdir/"scenario-summary.csv").read_text())
    print("L2_FIXED_SCENARIO_PORTFOLIO_V1_READY")

if __name__=="__main__":
    main()
