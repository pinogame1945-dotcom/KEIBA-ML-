#!/usr/bin/env python3
import argparse,csv,gzip,json,time
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import load_day, payout_map, canonical_numbers

YEARS=(2023,2024,2025)
PRIMARY_YEARS=(2024,2025)
LEVELS=(15,10,8,6)

def args():
    p=argparse.ArgumentParser(description="Fixed KING Top6 quinella ticket-compression benchmark.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def load_king_orders(path):
    orders={}
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf-8",newline="") as fh:
        r=csv.DictReader(fh)
        for row in r:
            y=int(float(row["year"]))
            if y not in YEARS: continue
            rid=str(row["race_id"])
            rank=int(float(row["consensus_rank"]))
            num=int(float(row["horse_number"]))
            key=(y,rid)
            d=orders.setdefault(key,{})
            if rank in d and d[rank]!=num:
                raise SystemExit(f"duplicate KING rank year={y} race={rid} rank={rank}")
            d[rank]=num
    return orders

def top6_rank_pairs(rank_map):
    ranks=[r for r in range(1,7) if r in rank_map]
    pairs=[]
    for i,a in enumerate(ranks):
        for b in ranks[i+1:]:
            pairs.append((a,b))
    # Fixed predeclared priority: lower KING rank-sum first;
    # tie-break by lower best rank, then lower worst rank.
    pairs.sort(key=lambda p:(p[0]+p[1],p[0],p[1]))
    return pairs

def tickets_for(rank_map,limit):
    pairs=top6_rank_pairs(rank_map)[:limit]
    return [
        (p,canonical_numbers("QUINELLA",(rank_map[p[0]],rank_map[p[1]])))
        for p in pairs
    ]

def max_drawdown(profits):
    peak=0.0
    equity=0.0
    mdd=0.0
    for p in profits:
        equity+=p
        if equity>peak: peak=equity
        dd=peak-equity
        if dd>mdd: mdd=dd
    return mdd

def aggregate(rows,years,label,limit):
    q=[x for x in rows if x["year"] in years and x["limit"]==limit]
    if not q: return None
    stake=sum(x["stake_yen"] for x in q)
    ret=sum(x["return_yen"] for x in q)
    ordered=sorted(q,key=lambda x:(x["year"],x["race_date"],x["race_id"]))
    profits=[x["profit_yen"] for x in ordered]
    hits=sum(x["hit"] for x in q)
    avg_pts=sum(x["tickets"] for x in q)/len(q)
    return {
        "period":label,
        "limit":limit,
        "races":len(q),
        "tickets":sum(x["tickets"] for x in q),
        "avg_points":avg_pts,
        "hit_races":hits,
        "hit_rate_pct":100.0*hits/len(q),
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(profits),
    }

def retention(rows,years,label,limit):
    full={x["race_id"]:(x["hit"],x["return_yen"],x["tickets"],x["stake_yen"])
          for x in rows if x["year"] in years and x["limit"]==15}
    comp={x["race_id"]:(x["hit"],x["return_yen"],x["tickets"],x["stake_yen"])
          for x in rows if x["year"] in years and x["limit"]==limit}
    # race_id is globally unique in this dataset; if not, year must be part of key.
    full_hits=sum(v[0] for v in full.values())
    retained=sum(1 for rid,v in full.items() if v[0] and rid in comp and comp[rid][0])
    full_return=sum(v[1] for v in full.values())
    comp_return=sum(v[1] for v in comp.values())
    full_stake=sum(v[3] for v in full.values())
    comp_stake=sum(v[3] for v in comp.values())
    return {
        "period":label,
        "limit":limit,
        "full15_hit_races":full_hits,
        "retained_full15_hits":retained,
        "full15_hit_retention_pct":100.0*retained/full_hits if full_hits else None,
        "stake_retention_pct":100.0*comp_stake/full_stake if full_stake else None,
        "stake_reduction_pct":100.0*(1.0-comp_stake/full_stake) if full_stake else None,
        "return_retention_pct":100.0*comp_return/full_return if full_return else None,
        "return_minus_stake_retention_pp":(
            100.0*comp_return/full_return - 100.0*comp_stake/full_stake
            if full_return and full_stake else None
        ),
    }

def main():
    a=args()
    outdir=Path(a.out_dir); outdir.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    king=load_king_orders(a.market_scored)
    wanted={y:{rid for (yy,rid) in king if yy==y} for y in YEARS}
    root=Path(a.backfill_root)
    rows=[]; counters=defaultdict(int)

    # Freeze and expose the exact full-field rank-pair priority.
    example={i:i for i in range(1,7)}
    priority=top6_rank_pairs(example)
    (outdir/"pair-priority.json").write_text(
        json.dumps({"rule":"rank_sum_asc_then_best_rank_then_worst_rank","pairs":priority},indent=2)+"\n",
        encoding="utf-8"
    )

    for year in YEARS:
        seen=set()
        for day_path in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
            date=day_path.name[:10]
            day=load_day(day_path,wanted[year])
            if not day: continue
            for rid,pack in day.items():
                rid=str(rid)
                key=(year,rid)
                if key not in king: continue
                payouts,present=payout_map(pack)
                if "QUINELLA" not in present:
                    counters["missing_quinella_payout"]+=1
                    continue
                seen.add(rid)
                rank_map=king[key]
                for limit in LEVELS:
                    tks=tickets_for(rank_map,limit)
                    stake=100.0*len(tks)
                    ret=sum(float(payouts.get(("QUINELLA",ticket),0.0)) for _,ticket in tks)
                    rows.append({
                        "year":year,"race_date":date,"race_id":rid,"limit":limit,
                        "tickets":len(tks),"stake_yen":stake,"return_yen":ret,
                        "profit_yen":ret-stake,"hit":int(ret>0),
                    })
        print("COMPRESSION_YEAR_DONE "+json.dumps({"year":year,"races":len(seen)},separators=(",",":")),flush=True)

    summary=[]
    retention_rows=[]
    for year in YEARS:
        for limit in LEVELS:
            summary.append(aggregate(rows,(year,),str(year),limit))
        for limit in (10,8,6):
            retention_rows.append(retention(rows,(year,),str(year),limit))
    for limit in LEVELS:
        summary.append(aggregate(rows,PRIMARY_YEARS,"2024|2025",limit))
    for limit in (10,8,6):
        retention_rows.append(retention(rows,PRIMARY_YEARS,"2024|2025",limit))

    def write_csv(path,data):
        data=[x for x in data if x is not None]
        with open(path,"w",newline="",encoding="utf-8") as fh:
            w=csv.DictWriter(fh,fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)

    write_csv(outdir/"summary.csv",summary)
    write_csv(outdir/"retention.csv",retention_rows)
    with gzip.open(outdir/"race-detail.csv.gz","wt",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    meta={
        "contract":"L2_KING_TOP6_QUINELLA_COMPRESSION_V1",
        "base":"KING Top6 quinella box",
        "levels":[15,10,8,6],
        "ticket_priority":"Within KING ranks 1..6, sort pair by rank-sum ascending, then best rank ascending, then worst rank ascending; take first N.",
        "unit_stake_yen":100,
        "odds_gate":False,
        "race_selection":False,
        "target_search":False,
        "evaluation_years":list(YEARS),
        "primary_years":list(PRIMARY_YEARS),
        "counters":dict(counters),
        "elapsed_seconds":time.perf_counter()-start,
        "promotion":False,
        "2026_locked":True,
    }
    (outdir/"meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== SUMMARY =====")
    print((outdir/"summary.csv").read_text())
    print("===== RETENTION =====")
    print((outdir/"retention.csv").read_text())
    print("L2_KING_TOP6_QUINELLA_COMPRESSION_V1_READY")

if __name__=="__main__":
    main()
