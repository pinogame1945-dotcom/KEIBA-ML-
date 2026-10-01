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
SOURCES=("KING","OUTSIDER","HYBRID_AXIS","HYBRID_TOP3","HYBRID_TOP8")

STRATEGIES={
    "Q_AXIS1_5":{"bet":"QUINELLA","label":"馬連 KING1位軸→紐5頭","mate_n":5},
    "Q_TOP2_6":{"bet":"QUINELLA","label":"馬連 KING1・2位軸→紐6頭","mate_n":6},
    "T_AXIS1_7":{"bet":"TRIO","label":"三連複 KING1位軸＋紐7頭から2頭","mate_n":7},
    "T_AXIS12_6":{"bet":"TRIO","label":"三連複 KING1・2位固定＋紐6頭","mate_n":6},
}

def args():
    p=argparse.ArgumentParser(description="Compare three KING-overlap exclusion levels for Outsider mates.")
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

def axes_for(strategy,king):
    if strategy in ("Q_AXIS1_5","T_AXIS1_7"):
        return [king[1]] if 1 in king else []
    if strategy in ("Q_TOP2_6","T_AXIS12_6"):
        return [king[r] for r in (1,2) if r in king]
    raise ValueError(strategy)

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

def hybrid_tickets(strategy,king,out,level):
    axes=axes_for(strategy,king)
    need_axes=1 if strategy in ("Q_AXIS1_5","T_AXIS1_7") else 2
    if len(axes)!=need_axes: return [],0

    if level=="AXIS":
        excluded=set(axes)
    elif level=="TOP3":
        excluded={king[r] for r in range(1,4) if r in king}
    elif level=="TOP8":
        excluded={king[r] for r in range(1,9) if r in king}
    else:
        raise ValueError(level)

    mates=first_excluding(out,STRATEGIES[strategy]["mate_n"],excluded)

    if strategy=="Q_AXIS1_5":
        ts=[canonical_numbers("QUINELLA",(axes[0],m)) for m in mates]
    elif strategy=="Q_TOP2_6":
        ts=sorted(set(canonical_numbers("QUINELLA",(a,m)) for a in axes for m in mates))
    elif strategy=="T_AXIS1_7":
        ts=[canonical_numbers("TRIO",(axes[0],)+p) for p in itertools.combinations(mates,2)]
    elif strategy=="T_AXIS12_6":
        ts=[canonical_numbers("TRIO",(axes[0],axes[1],m)) for m in mates]
    else:
        raise ValueError(strategy)
    return ts,len(mates)

def tickets_for(source,strategy,king,out):
    if source=="KING": return pure_tickets(king,strategy),None
    if source=="OUTSIDER": return pure_tickets(out,strategy),None
    if source=="HYBRID_AXIS": return hybrid_tickets(strategy,king,out,"AXIS")
    if source=="HYBRID_TOP3": return hybrid_tickets(strategy,king,out,"TOP3")
    if source=="HYBRID_TOP8": return hybrid_tickets(strategy,king,out,"TOP8")
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
        "avg_points_all_races":float(np.mean([r["tickets"] for r in q])),
        "avg_points_active":float(np.mean([r["tickets"] for r in q if r["active"]])) if active else 0.0,
        "hit_races":sum(r["hit"] for r in q),
        "hit_rate_all_races_pct":100.0*sum(r["hit"] for r in q)/len(q),
        "hit_rate_active_pct":100.0*sum(r["hit"] for r in q)/active if active else None,
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
        print("OVERLAP_LEVEL_YEAR_DONE "+json.dumps({"year":year,"races":len(seen)},separators=(",",":")),flush=True)

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
                "avg_points_all_races":float(np.mean([r["tickets"] for r in q])),
                "avg_points_active":float(np.mean([r["tickets"] for r in q if r["active"]])) if active else 0.0,
                "hit_races":sum(r["hit"] for r in q),
                "hit_rate_all_races_pct":100.0*sum(r["hit"] for r in q)/len(q),
                "hit_rate_active_pct":100.0*sum(r["hit"] for r in q)/active if active else None,
                "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
                "roi_pct":100.0*ret/stake if stake else None,
                "max_drawdown_yen":max_drawdown(profits),
                "avg_mates":float(np.mean([r["mate_count"] for r in q if r["mate_count"] is not None])) if any(r["mate_count"] is not None for r in q) else None,
            })

    pmap={(r["source"],r["strategy"]):r for r in pooled}
    compare=[]
    for s in STRATEGIES:
        row={"strategy":s,"bet_type":STRATEGIES[s]["bet"],"label":STRATEGIES[s]["label"]}
        for source in SOURCES:
            x=pmap[(source,s)]
            key=source.lower()
            row[f"{key}_hit_pct"]=x["hit_rate_all_races_pct"]
            row[f"{key}_roi_pct"]=x["roi_pct"]
            row[f"{key}_coverage_pct"]=x["coverage_pct"]
            row[f"{key}_avg_points"]=x["avg_points_all_races"]
            row[f"{key}_avg_mates"]=x["avg_mates"]
        row["top3_minus_axis_roi_pp"]=pmap[("HYBRID_TOP3",s)]["roi_pct"]-pmap[("HYBRID_AXIS",s)]["roi_pct"]
        row["top8_minus_top3_roi_pp"]=pmap[("HYBRID_TOP8",s)]["roi_pct"]-pmap[("HYBRID_TOP3",s)]["roi_pct"]
        compare.append(row)

    write_csv(outdir/"yearly-summary.csv",yearly)
    write_csv(outdir/"primary-2024-2025-summary.csv",pooled)
    write_csv(outdir/"five-way-comparison.csv",compare)

    if rows:
        with gzip.open(outdir/"race-detail.csv.gz","wt",newline="",encoding="utf-8") as fh:
            w=csv.DictWriter(fh,fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    summary={
        "contract":"L2_KING_AXIS_OUTSIDER_OVERLAP_LEVELS_V3",
        "comparison":"KING-only and OUTSIDER-only baselines plus three hybrid overlap exclusions: axis-only, KING Top3, KING Top8.",
        "hybrid_levels":{
            "HYBRID_AXIS":"exclude only the fixed KING axis horse(s)",
            "HYBRID_TOP3":"exclude KING ranks 1-3 before taking Outsider mates",
            "HYBRID_TOP8":"exclude KING ranks 1-8 before taking Outsider mates"
        },
        "short_field_policy":"Never refill with excluded KING horses. Use fewer mates/tickets and report coverage.",
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
    print("===== FIVE WAY ====="); print((outdir/"five-way-comparison.csv").read_text())
    print("===== PRIMARY ====="); print((outdir/"primary-2024-2025-summary.csv").read_text())
    print("L2_KING_AXIS_OUTSIDER_OVERLAP_LEVELS_V3_READY")

if __name__=="__main__":
    main()
