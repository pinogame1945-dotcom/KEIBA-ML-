#!/usr/bin/env python3
import argparse,csv,gzip,json,time
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import load_day,payout_map,canonical_numbers

YEARS=(2023,2024,2025)
PRIMARY=(2024,2025)

# Fixed blocks inside KING Top6 quinella BOX.
BLOCKS={
    "G1_AXIS1":[(1,2),(1,3),(1,4),(1,5),(1,6)],
    "G2_AXIS2":[(2,3),(2,4),(2,5),(2,6)],
    "GLOW_3TO6":[(3,4),(3,5),(3,6),(4,5),(4,6),(5,6)],
}
SETS={
    "FULL15":("G1_AXIS1","G2_AXIS2","GLOW_3TO6"),
    "G1_ONLY":("G1_AXIS1",),
    "G2_ONLY":("G2_AXIS2",),
    "GLOW_ONLY":("GLOW_3TO6",),
    "G1_G2":("G1_AXIS1","G2_AXIS2"),
    "G1_GLOW":("G1_AXIS1","GLOW_3TO6"),
    "G2_GLOW":("G2_AXIS2","GLOW_3TO6"),
}

def parse_args():
    p=argparse.ArgumentParser(description="Fixed KING Top6 quinella block decomposition.")
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
            if y not in YEARS: continue
            rid=str(row["race_id"])
            rank=int(float(row["consensus_rank"]))
            num=int(float(row["horse_number"]))
            d=out.setdefault((y,rid),{})
            if rank in d and d[rank]!=num:
                raise SystemExit(f"duplicate rank y={y} race={rid} rank={rank}")
            d[rank]=num
    return out

def pairs_for_set(name):
    ans=[]
    for b in SETS[name]:
        ans.extend(BLOCKS[b])
    return ans

def max_drawdown(profits):
    eq=peak=mdd=0.0
    for p in profits:
        eq+=p
        peak=max(peak,eq)
        mdd=max(mdd,peak-eq)
    return mdd

def agg(rows,years,label,set_name):
    q=[x for x in rows if x["year"] in years and x["set_name"]==set_name]
    if not q: return None
    stake=sum(x["stake_yen"] for x in q); ret=sum(x["return_yen"] for x in q)
    hits=sum(x["hit"] for x in q)
    profits=[x["profit_yen"] for x in sorted(q,key=lambda z:(z["year"],z["race_date"],z["race_id"]))]
    return {
        "period":label,"set_name":set_name,
        "blocks":"+".join(SETS[set_name]),
        "races":len(q),"avg_points":sum(x["tickets"] for x in q)/len(q),
        "hit_races":hits,"hit_rate_pct":100.0*hits/len(q),
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(profits),
    }

def contribution(rows,years,label,block):
    full={(x["year"],x["race_id"]):x for x in rows if x["year"] in years and x["set_name"]=="FULL15"}
    without_name={"G1_AXIS1":"G2_GLOW","G2_AXIS2":"G1_GLOW","GLOW_3TO6":"G1_G2"}[block]
    wo={(x["year"],x["race_id"]):x for x in rows if x["year"] in years and x["set_name"]==without_name}
    full_hits=sum(x["hit"] for x in full.values())
    lost_hits=sum(1 for k,x in full.items() if x["hit"] and not wo[k]["hit"])
    full_return=sum(x["return_yen"] for x in full.values())
    wo_return=sum(x["return_yen"] for x in wo.values())
    full_stake=sum(x["stake_yen"] for x in full.values())
    wo_stake=sum(x["stake_yen"] for x in wo.values())
    return {
        "period":label,"removed_block":block,"without_set":without_name,
        "full15_hit_races":full_hits,
        "hits_lost_when_removed":lost_hits,
        "hit_loss_pct_of_full15_hits":100.0*lost_hits/full_hits if full_hits else None,
        "stake_saved_pct":100.0*(1.0-wo_stake/full_stake) if full_stake else None,
        "return_lost_pct":100.0*(1.0-wo_return/full_return) if full_return else None,
        "return_lost_minus_stake_saved_pp":(
            100.0*(1.0-wo_return/full_return)-100.0*(1.0-wo_stake/full_stake)
            if full_return and full_stake else None
        ),
    }

def pair_agg(pair_rows,years,label,pair):
    q=[x for x in pair_rows if x["year"] in years and x["rank_pair"]==pair]
    if not q: return None
    stake=sum(x["stake_yen"] for x in q); ret=sum(x["return_yen"] for x in q)
    return {
        "period":label,"rank_pair":pair,"races":len(q),
        "hit_races":sum(x["hit"] for x in q),
        "hit_rate_pct":100.0*sum(x["hit"] for x in q)/len(q),
        "stake_yen":stake,"return_yen":ret,"roi_pct":100.0*ret/stake if stake else None,
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

    pair_to_block={p:b for b,pairs in BLOCKS.items() for p in pairs}

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
                    counters["missing_quinella"]+=1; continue
                ranks=king[key]
                if any(r not in ranks for r in range(1,7)):
                    counters["short_top6"]+=1
                seen.add(rid)

                cache={}
                for a1 in range(1,7):
                    for b1 in range(a1+1,7):
                        if a1 not in ranks or b1 not in ranks: continue
                        ticket=canonical_numbers("QUINELLA",(ranks[a1],ranks[b1]))
                        ret=float(payouts.get(("QUINELLA",ticket),0.0))
                        cache[(a1,b1)]=ret
                        pair_rows.append({
                            "year":year,"race_date":date,"race_id":rid,
                            "rank_pair":f"{a1}-{b1}","block":pair_to_block[(a1,b1)],
                            "stake_yen":100.0,"return_yen":ret,"hit":int(ret>0),
                        })

                for set_name in SETS:
                    pairs=[p for p in pairs_for_set(set_name) if p in cache]
                    stake=100.0*len(pairs)
                    ret=sum(cache[p] for p in pairs)
                    rows.append({
                        "year":year,"race_date":date,"race_id":rid,"set_name":set_name,
                        "tickets":len(pairs),"stake_yen":stake,"return_yen":ret,
                        "profit_yen":ret-stake,"hit":int(ret>0),
                    })
        print("DECOMP_YEAR_DONE "+json.dumps({"year":year,"races":len(seen)},separators=(",",":")),flush=True)

    summary=[]; contrib=[]; pairsum=[]
    for y in YEARS:
        for s in SETS: summary.append(agg(rows,(y,),str(y),s))
        for b in BLOCKS: contrib.append(contribution(rows,(y,),str(y),b))
        for p in [f"{a}-{b}" for a in range(1,7) for b in range(a+1,7)]:
            pairsum.append(pair_agg(pair_rows,(y,),str(y),p))
    for s in SETS: summary.append(agg(rows,PRIMARY,"2024|2025",s))
    for b in BLOCKS: contrib.append(contribution(rows,PRIMARY,"2024|2025",b))
    for p in [f"{a}-{b}" for a in range(1,7) for b in range(a+1,7)]:
        pairsum.append(pair_agg(pair_rows,PRIMARY,"2024|2025",p))

    write_csv(outdir/"set-summary.csv",summary)
    write_csv(outdir/"block-removal-contribution.csv",contrib)
    write_csv(outdir/"pair-summary.csv",pairsum)

    meta={
        "contract":"L2_KING_TOP6_QUINELLA_BLOCK_DECOMP_V1",
        "blocks":BLOCKS,
        "sets":SETS,
        "unit_stake_yen":100,
        "odds_gate":False,"race_selection":False,"target_search":False,
        "evaluation_years":list(YEARS),"primary_years":list(PRIMARY),
        "counters":dict(counters),"elapsed_seconds":time.perf_counter()-start,
        "promotion":False,"2026_locked":True,
    }
    (outdir/"meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== SET SUMMARY ====="); print((outdir/"set-summary.csv").read_text())
    print("===== CONTRIBUTION ====="); print((outdir/"block-removal-contribution.csv").read_text())
    print("===== PAIRS ====="); print((outdir/"pair-summary.csv").read_text())
    print("L2_KING_TOP6_QUINELLA_BLOCK_DECOMP_V1_READY")

if __name__=="__main__":
    main()
