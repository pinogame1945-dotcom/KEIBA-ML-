#!/usr/bin/env python3
import argparse
import csv
import gzip
import itertools
import json
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import horse_number_map, payout_map
from run_l2_simple_rank_role_v1 import YEARS, load_l17
from run_l2_market_gap_v1 import market_map, gap_bucket, gap_direction

FOCAL_RANKS=(1,2,3)
POOLS=(3,6)
SEATS=(1,2,3)


def parse_args():
    p=argparse.ArgumentParser(description="Trifecta seat x market-gap audit for fixed L1.7 ranks 1-3.")
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def parse_year_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1)
        out[int(y)]=Path(p)
    if set(out)!=set(YEARS):
        raise SystemExit(f"year path mismatch got={sorted(out)} expected={list(YEARS)}")
    return out


def tickets_for_seat(pool,focal,seat):
    others=[x for x in pool if x!=focal]
    rows=[]
    for a,b in itertools.permutations(others,2):
        if seat==1:
            rows.append((focal,a,b))
        elif seat==2:
            rows.append((a,focal,b))
        elif seat==3:
            rows.append((a,b,focal))
        else:
            raise ValueError(seat)
    # permutations are unique by construction, but keep a deterministic guard.
    out=[]
    seen=set()
    for row in rows:
        if len(set(row))!=3 or row in seen:
            continue
        seen.add(row)
        out.append(row)
    return out


def empty_stat():
    return {
        "races":0,
        "tickets":0,
        "hit_races":0,
        "winning_tickets":0,
        "stake_yen":0.0,
        "return_yen":0.0,
    }


def update(stat,tickets,payouts):
    stake=100.0*len(tickets)
    ret=0.0
    wins=0
    for nums in tickets:
        p=float(payouts.get(("TRIFECTA",tuple(nums)),0.0))
        if p>0:
            wins+=1
            ret+=p
    stat["races"]+=1
    stat["tickets"]+=len(tickets)
    stat["hit_races"]+=int(wins>0)
    stat["winning_tickets"]+=wins
    stat["stake_yen"]+=stake
    stat["return_yen"]+=ret


def finalize(key,stat):
    year,rank,pool_k,seat,group=key
    n=stat["races"]
    stake=stat["stake_yen"]
    return {
        "year":year,
        "focal_rank":rank,
        "pool_k":pool_k,
        "seat":seat,
        "market_group":group,
        "races":n,
        "tickets":stat["tickets"],
        "avg_tickets_per_race":stat["tickets"]/n if n else None,
        "hit_races":stat["hit_races"],
        "hit_rate_pct":100.0*stat["hit_races"]/n if n else None,
        "winning_tickets":stat["winning_tickets"],
        "stake_yen":stake,
        "return_yen":stat["return_yen"],
        "profit_yen":stat["return_yen"]-stake,
        "roi_pct":100.0*stat["return_yen"]/stake if stake else None,
    }


def empty_horse():
    return {"n":0,"finish1":0,"finish2":0,"finish3":0,"top3":0}


def update_horse(stat,finish):
    stat["n"]+=1
    stat["finish1"]+=int(finish==1)
    stat["finish2"]+=int(finish==2)
    stat["finish3"]+=int(finish==3)
    stat["top3"]+=int(finish is not None and finish<=3)


def horse_row(key,stat):
    year,rank,group=key
    n=stat["n"]
    return {
        "year":year,
        "focal_rank":rank,
        "market_group":group,
        "horses":n,
        "finish1_rate_pct":100.0*stat["finish1"]/n if n else None,
        "finish2_rate_pct":100.0*stat["finish2"]/n if n else None,
        "finish3_rate_pct":100.0*stat["finish3"]/n if n else None,
        "top3_rate_pct":100.0*stat["top3"]/n if n else None,
    }


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    fields=list(rows[0].keys())
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    a=parse_args()
    paths=parse_year_paths(a.l17_year)
    l17,l17_counts=load_l17(paths)
    root=Path(a.backfill_root)/"data"/"daily"
    out_dir=Path(a.out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)

    stats=defaultdict(empty_stat)
    horse_stats=defaultdict(empty_horse)
    seen=set()
    files_scanned=0

    for year in YEARS:
        for path in sorted(root.glob(f"{year}-*.jsonl.gz")):
            files_scanned+=1
            with gzip.open(path,"rt",encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    pack=json.loads(line)
                    rid=str((pack.get("race") or {}).get("race_id") or "")
                    rec=l17.get(rid)
                    if rec is None:
                        continue
                    if rid in seen:
                        raise ValueError(f"duplicate race pack race={rid}")
                    if int(rec["year"])!=year:
                        raise ValueError(f"race year drift race={rid}")

                    order=[str(x) for x in rec["consensus_order"]]
                    hno=horse_number_map(pack)
                    market=market_map(pack)
                    payouts,present=payout_map(pack)
                    if "TRIFECTA" not in present:
                        seen.add(rid)
                        continue

                    trusted_ids=order[:min(6,len(order))]
                    if any(h not in hno for h in trusted_ids):
                        missing=[h for h in trusted_ids if h not in hno]
                        raise ValueError(f"horse-number mapping missing race={rid} horses={missing}")
                    trusted_numbers=[hno[h] for h in trusted_ids]

                    for rank in FOCAL_RANKS:
                        if rank>len(order):
                            continue
                        hid=order[rank-1]
                        m=market.get(hid)
                        if not m or m["popularity"] is None:
                            raise ValueError(f"market popularity missing race={rid} rank={rank} horse={hid}")
                        pop=int(m["popularity"])
                        gap=pop-rank
                        bucket=gap_bucket(gap)
                        direction=gap_direction(gap)
                        finish=m["finish"]

                        groups=["ALL",bucket,direction]
                        if gap>0:
                            groups.append("KING_HIGHER")
                        elif gap<0:
                            groups.append("MARKET_HIGHER")
                        else:
                            groups.append("SAME")
                        groups=list(dict.fromkeys(groups))

                        for ykey in (year,"ALL"):
                            for group in groups:
                                update_horse(horse_stats[(ykey,rank,group)],finish)

                        focal_no=hno[hid]
                        for pool_k in POOLS:
                            pool=trusted_numbers[:pool_k]
                            if len(pool)<pool_k or focal_no not in pool:
                                continue
                            for seat in SEATS:
                                tickets=tickets_for_seat(pool,focal_no,seat)
                                if not tickets:
                                    continue
                                for ykey in (year,"ALL"):
                                    for group in groups:
                                        update(stats[(ykey,rank,pool_k,seat,group)],tickets,payouts)

                    seen.add(rid)

    missing=set(l17)-seen
    if missing:
        raise SystemExit(f"missing race packs count={len(missing)} sample={sorted(missing)[:20]}")

    rows=[finalize(k,v) for k,v in stats.items()]
    rows.sort(key=lambda r:(str(r["year"]),r["focal_rank"],r["pool_k"],r["seat"],r["market_group"]))
    combined=[r for r in rows if r["year"]=="ALL"]
    yearly=[r for r in rows if r["year"]!="ALL"]

    hrows=[horse_row(k,v) for k,v in horse_stats.items()]
    hrows.sort(key=lambda r:(str(r["year"]),r["focal_rank"],r["market_group"]))

    # Fixed-bucket lift vs each rank/pool/seat baseline. This is diagnostic only.
    base={}
    yearly_index={}
    for r in combined:
        if r["market_group"]=="ALL":
            base[(r["focal_rank"],r["pool_k"],r["seat"])]=r
    for r in yearly:
        yearly_index[(int(r["year"]),r["focal_rank"],r["pool_k"],r["seat"],r["market_group"])]=r

    lifts=[]
    diagnostic_groups=["KING_HIGHER","MARKET_HIGHER","SAME","POS_1","POS_2_3","POS_4PLUS","NEG_1","NEG_2_3","NEG_4PLUS","ZERO"]
    for r in combined:
        group=r["market_group"]
        if group not in diagnostic_groups:
            continue
        b=base[(r["focal_rank"],r["pool_k"],r["seat"])]
        yearly_rois=[]
        yearly_deltas=[]
        yearly_ns=[]
        complete=True
        for y in YEARS:
            rr=yearly_index.get((y,r["focal_rank"],r["pool_k"],r["seat"],group))
            bb=yearly_index.get((y,r["focal_rank"],r["pool_k"],r["seat"],"ALL"))
            if rr is None or bb is None:
                complete=False
                yearly_rois.append(None)
                yearly_deltas.append(None)
                yearly_ns.append(0)
            else:
                yearly_rois.append(rr["roi_pct"])
                yearly_deltas.append(rr["roi_pct"]-bb["roi_pct"])
                yearly_ns.append(rr["races"])
        lifts.append({
            "focal_rank":r["focal_rank"],
            "pool_k":r["pool_k"],
            "seat":r["seat"],
            "market_group":group,
            "races":r["races"],
            "coverage_pct":100.0*r["races"]/b["races"] if b["races"] else None,
            "baseline_roi_pct":b["roi_pct"],
            "group_roi_pct":r["roi_pct"],
            "roi_delta_pp":r["roi_pct"]-b["roi_pct"],
            "hit_rate_pct":r["hit_rate_pct"],
            "profit_yen":r["profit_yen"],
            "y2022_roi_pct":yearly_rois[0],
            "y2023_roi_pct":yearly_rois[1],
            "y2024_roi_pct":yearly_rois[2],
            "y2025_roi_pct":yearly_rois[3],
            "y2022_delta_pp":yearly_deltas[0],
            "y2023_delta_pp":yearly_deltas[1],
            "y2024_delta_pp":yearly_deltas[2],
            "y2025_delta_pp":yearly_deltas[3],
            "years_improved":sum(1 for x in yearly_deltas if x is not None and x>0),
            "all_4_years_improved":complete and all(x>0 for x in yearly_deltas),
            "min_year_races":min(yearly_ns) if complete else 0,
        })
    lifts.sort(key=lambda r:(r["all_4_years_improved"],r["years_improved"],r["roi_delta_pp"],r["races"]),reverse=True)

    write_csv(out_dir/"seat-gap-combined.csv",combined)
    write_csv(out_dir/"seat-gap-yearly.csv",yearly)
    write_csv(out_dir/"horse-seat-outcomes.csv",hrows)
    write_csv(out_dir/"gap-seat-lift.csv",lifts)

    summary={
        "contract":"L2_MARKET_GAP_SEAT_V1",
        "idea":"Fix L1.7 rank 1-3 into trifecta seat 1/2/3 and measure market-gap buckets without learned thresholds.",
        "years":list(YEARS),
        "l17_races_by_year":l17_counts,
        "races_processed":len(seen),
        "daily_files_scanned":files_scanned,
        "focal_ranks":list(FOCAL_RANKS),
        "pools":list(POOLS),
        "seats":list(SEATS),
        "gap_definition":"market_popularity_rank - l17_consensus_rank",
        "fixed_gap_buckets":["NEG_4PLUS","NEG_2_3","NEG_1","ZERO","POS_1","POS_2_3","POS_4PLUS"],
        "learning":False,
        "threshold_learning":False,
        "rerank_l1":False,
        "gate":False,
        "outsider":False,
        "payout_evaluation_only":True,
        "stake_per_ticket_yen":100,
        "2026_locked":True,
        "production_promotion":False,
    }
    (out_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out_dir/"README.md").write_text(
        "# L2 Market Gap Seat V1\n\n"
        "Trifecta-only diagnostic. L1.7 ranks are fixed and never re-ranked.\n"
        "For each focal rank 1/2/3, place it in seat 1/2/3, fill remaining seats from Top3 or Top6, "
        "then compare fixed market-gap buckets. No learned threshold; payout is evaluation only; 2026 sealed.\n",
        encoding="utf-8",
    )
    print("L2_MARKET_GAP_SEAT_V1_DONE")
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__":
    main()
