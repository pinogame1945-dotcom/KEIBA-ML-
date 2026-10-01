#!/usr/bin/env python3
import argparse,csv,gzip,json,time
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import load_day,payout_map,canonical_numbers

YEARS=(2023,2024,2025)
PRIMARY=(2024,2025)
LEVELS=(9,7,6,5)

BASE_PAIRS=[
    (1,2),(1,3),(1,4),(1,5),(1,6),
    (2,3),(2,4),(2,5),(2,6),
]

def parse_args():
    p=argparse.ArgumentParser(description="Fixed compression inside KING top2-involved 9-point quinella set.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def load_king(path):
    out={}
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf-8",newline="") as fh:
        r=csv.DictReader(fh)
        for row in r:
            y=int(float(row["year"]))
            if y not in YEARS:
                continue
            rid=str(row["race_id"])
            rank=int(float(row["consensus_rank"]))
            num=int(float(row["horse_number"]))
            d=out.setdefault((y,rid),{})
            if rank in d and d[rank]!=num:
                raise SystemExit(f"duplicate KING rank y={y} race={rid} rank={rank}")
            d[rank]=num
    return out

def priority_pairs():
    # Fixed structural priority only. No odds/results/ROI used.
    # Lower rank-sum first, then better minimum rank, then worse rank.
    return sorted(BASE_PAIRS,key=lambda p:(p[0]+p[1],p[0],p[1]))

def tickets(rank_map,limit):
    out=[]
    for pair in priority_pairs()[:limit]:
        a,b=pair
        if a not in rank_map or b not in rank_map:
            continue
        out.append((pair,canonical_numbers("QUINELLA",(rank_map[a],rank_map[b]))))
    return out

def max_drawdown(profits):
    eq=peak=mdd=0.0
    for p in profits:
        eq+=p
        peak=max(peak,eq)
        mdd=max(mdd,peak-eq)
    return mdd

def agg(rows,years,label,limit):
    q=[x for x in rows if x["year"] in years and x["limit"]==limit]
    if not q: return None
    stake=sum(x["stake_yen"] for x in q)
    ret=sum(x["return_yen"] for x in q)
    hits=sum(x["hit"] for x in q)
    profits=[x["profit_yen"] for x in sorted(q,key=lambda z:(z["year"],z["race_date"],z["race_id"]))]
    return {
        "period":label,"limit":limit,"races":len(q),
        "avg_points":sum(x["tickets"] for x in q)/len(q),
        "hit_races":hits,"hit_rate_pct":100.0*hits/len(q),
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(profits),
    }

def retention(rows,years,label,limit):
    full={(x["year"],x["race_id"]):x for x in rows if x["year"] in years and x["limit"]==9}
    comp={(x["year"],x["race_id"]):x for x in rows if x["year"] in years and x["limit"]==limit}
    full_hits=sum(x["hit"] for x in full.values())
    retained=sum(1 for k,x in full.items() if x["hit"] and comp.get(k,{}).get("hit",0))
    full_stake=sum(x["stake_yen"] for x in full.values())
    comp_stake=sum(x["stake_yen"] for x in comp.values())
    full_ret=sum(x["return_yen"] for x in full.values())
    comp_ret=sum(x["return_yen"] for x in comp.values())
    return {
        "period":label,"limit":limit,
        "base9_hit_races":full_hits,
        "retained_base9_hits":retained,
        "base9_hit_retention_pct":100.0*retained/full_hits if full_hits else None,
        "stake_reduction_pct":100.0*(1.0-comp_stake/full_stake) if full_stake else None,
        "return_retention_pct":100.0*comp_ret/full_ret if full_ret else None,
        "return_retention_minus_stake_retention_pp":(
            100.0*comp_ret/full_ret - 100.0*comp_stake/full_stake
            if full_ret and full_stake else None
        ),
    }

def pair_agg(pair_rows,years,label,pair):
    key=f"{pair[0]}-{pair[1]}"
    q=[x for x in pair_rows if x["year"] in years and x["rank_pair"]==key]
    if not q: return None
    stake=sum(x["stake_yen"] for x in q); ret=sum(x["return_yen"] for x in q)
    return {
        "period":label,"rank_pair":key,"priority":priority_pairs().index(pair)+1,
        "races":len(q),"hit_races":sum(x["hit"] for x in q),
        "hit_rate_pct":100.0*sum(x["hit"] for x in q)/len(q),
        "stake_yen":stake,"return_yen":ret,
        "roi_pct":100.0*ret/stake if stake else None,
    }

def write_csv(path,data):
    data=[x for x in data if x is not None]
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(data[0].keys()))
        w.writeheader(); w.writerows(data)

def main():
    a=parse_args()
    outdir=Path(a.out_dir); outdir.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    king=load_king(a.market_scored)
    wanted={y:{rid for yy,rid in king if yy==y} for y in YEARS}
    root=Path(a.backfill_root)
    rows=[]; pair_rows=[]; counters=defaultdict(int)

    pri=priority_pairs()
    (outdir/"pair-priority.json").write_text(
        json.dumps({
            "base_pairs":BASE_PAIRS,
            "rule":"rank_sum_asc_then_best_rank_then_worst_rank",
            "priority":pri,
            "levels":list(LEVELS),
        },indent=2)+"\n",encoding="utf-8"
    )

    for year in YEARS:
        seen=set()
        for day_path in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
            date=day_path.name[:10]
            day=load_day(day_path,wanted[year])
            if not day: continue
            for rid,pack in day.items():
                rid=str(rid); key=(year,rid)
                if key not in king: continue
                payouts,present=payout_map(pack)
                if "QUINELLA" not in present:
                    counters["missing_quinella"]+=1
                    continue
                ranks=king[key]
                seen.add(rid)

                pair_ret={}
                for p in pri:
                    a1,b1=p
                    if a1 not in ranks or b1 not in ranks:
                        continue
                    ticket=canonical_numbers("QUINELLA",(ranks[a1],ranks[b1]))
                    ret=float(payouts.get(("QUINELLA",ticket),0.0))
                    pair_ret[p]=ret
                    pair_rows.append({
                        "year":year,"race_date":date,"race_id":rid,
                        "rank_pair":f"{a1}-{b1}","stake_yen":100.0,
                        "return_yen":ret,"hit":int(ret>0),
                    })

                for limit in LEVELS:
                    chosen=[p for p in pri[:limit] if p in pair_ret]
                    stake=100.0*len(chosen)
                    ret=sum(pair_ret[p] for p in chosen)
                    rows.append({
                        "year":year,"race_date":date,"race_id":rid,"limit":limit,
                        "tickets":len(chosen),"stake_yen":stake,"return_yen":ret,
                        "profit_yen":ret-stake,"hit":int(ret>0),
                    })
        print("TOP2_NINE_YEAR_DONE "+json.dumps({"year":year,"races":len(seen)},separators=(",",":")),flush=True)

    summary=[]; retain=[]; pairs=[]
    for y in YEARS:
        for limit in LEVELS:
            summary.append(agg(rows,(y,),str(y),limit))
        for limit in (7,6,5):
            retain.append(retention(rows,(y,),str(y),limit))
        for p in pri:
            pairs.append(pair_agg(pair_rows,(y,),str(y),p))
    for limit in LEVELS:
        summary.append(agg(rows,PRIMARY,"2024|2025",limit))
    for limit in (7,6,5):
        retain.append(retention(rows,PRIMARY,"2024|2025",limit))
    for p in pri:
        pairs.append(pair_agg(pair_rows,PRIMARY,"2024|2025",p))

    write_csv(outdir/"summary.csv",summary)
    write_csv(outdir/"retention.csv",retain)
    write_csv(outdir/"pair-summary.csv",pairs)

    meta={
        "contract":"L2_KING_TOP2_NINE_COMPRESSION_V1",
        "base":"KING top2-involved quinella set: 1-2..6 plus 2-3..6 = 9 points",
        "levels":list(LEVELS),
        "priority_rule":"rank-sum ascending, then best rank ascending, then worst rank ascending",
        "unit_stake_yen":100,
        "odds_gate":False,"race_selection":False,"target_search":False,
        "evaluation_years":list(YEARS),"primary_years":list(PRIMARY),
        "counters":dict(counters),"elapsed_seconds":time.perf_counter()-start,
        "promotion":False,"2026_locked":True,
    }
    (outdir/"meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== SUMMARY ====="); print((outdir/"summary.csv").read_text())
    print("===== RETENTION ====="); print((outdir/"retention.csv").read_text())
    print("===== PAIRS ====="); print((outdir/"pair-summary.csv").read_text())
    print("L2_KING_TOP2_NINE_COMPRESSION_V1_READY")

if __name__=="__main__":
    main()
