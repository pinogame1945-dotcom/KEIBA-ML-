#!/usr/bin/env python3
import argparse
import csv
import gzip
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import horse_number_map, payout_map, final_odds_tuple
from run_l2_simple_rank_role_v1 import YEARS, load_l17
from run_l2_market_gap_v1 import market_map, gap_bucket, gap_direction

TEST_YEARS=(2023,2024,2025)
FOCAL_RANKS=(1,2,3)
POOLS=(3,6)
SEATS=(1,2,3)
VARIANTS=("RANK_ONLY","DIRECTION","BUCKET")
POLICIES=("PROB_ONLY","VALUE_RATIO","MARKET_CHEAPEST")


def parse_args():
    p=argparse.ArgumentParser(description="Walk-forward trifecta seat selection using prior formation hit probabilities and current final market odds.")
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


def finite(value):
    try:
        if isinstance(value,str):
            value=value.replace(",","").strip()
        x=float(value)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None


def load_odds_day(path,wanted):
    rows={}
    if not path.exists():
        raise FileNotFoundError(str(path))
    with gzip.open(path,"rt",encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            if rid in wanted:
                rows[rid]=row
    return rows


def decode_trifecta_odds(record):
    out={}
    root=record.get("odds") or {}
    data=root.get("8")
    if not isinstance(data,dict):
        return out
    for key,raw in data.items():
        key=str(key)
        if not re.fullmatch(r"\d{6}",key):
            continue
        nums=tuple(int(key[i:i+2]) for i in (0,2,4))
        if any(x<=0 for x in nums) or len(set(nums))!=3:
            continue
        tup=final_odds_tuple(raw)
        if not tup:
            continue
        price=finite(tup[0])
        if price is None or price<=0:
            continue
        out[nums]=price
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
    return rows


def market_group(variant,gap):
    if variant=="RANK_ONLY":
        return "ALL"
    if variant=="DIRECTION":
        return gap_direction(gap)
    if variant=="BUCKET":
        return gap_bucket(gap)
    raise ValueError(variant)


def empty_train():
    return {"races":0,"hits":0}


def empty_eval():
    return {
        "races":0,
        "tickets":0,
        "hit_races":0,
        "stake_yen":0.0,
        "return_yen":0.0,
        "sum_selected_market_mass":0.0,
        "sum_selected_prior_hit_rate":0.0,
        "sum_selected_value_ratio":0.0,
    }


def update_eval(stat,tickets,payouts,market_mass,prior_hit_rate,value_ratio):
    ret=0.0
    hit=False
    for t in tickets:
        p=float(payouts.get(("TRIFECTA",tuple(t)),0.0))
        if p>0:
            hit=True
            ret+=p
    stat["races"]+=1
    stat["tickets"]+=len(tickets)
    stat["hit_races"]+=int(hit)
    stat["stake_yen"]+=100.0*len(tickets)
    stat["return_yen"]+=ret
    stat["sum_selected_market_mass"]+=market_mass
    stat["sum_selected_prior_hit_rate"]+=prior_hit_rate
    stat["sum_selected_value_ratio"]+=value_ratio


def finalize_eval(key,stat):
    year,variant,policy,rank,pool=key
    n=stat["races"]
    stake=stat["stake_yen"]
    return {
        "year":year,
        "variant":variant,
        "policy":policy,
        "focal_rank":rank,
        "pool_k":pool,
        "races":n,
        "tickets":stat["tickets"],
        "avg_tickets_per_race":stat["tickets"]/n if n else None,
        "hit_races":stat["hit_races"],
        "hit_rate_pct":100.0*stat["hit_races"]/n if n else None,
        "stake_yen":stake,
        "return_yen":stat["return_yen"],
        "profit_yen":stat["return_yen"]-stake,
        "roi_pct":100.0*stat["return_yen"]/stake if stake else None,
        "avg_selected_market_mass":stat["sum_selected_market_mass"]/n if n else None,
        "avg_selected_prior_hit_rate_pct":100.0*stat["sum_selected_prior_hit_rate"]/n if n else None,
        "avg_selected_value_ratio":stat["sum_selected_value_ratio"]/n if n else None,
    }


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    a=parse_args()
    paths=parse_year_paths(a.l17_year)
    l17,l17_counts=load_l17(paths)
    root=Path(a.backfill_root)
    daily_root=root/"data"/"daily"
    odds_root=root/"data"/"odds"/"daily"
    out_dir=Path(a.out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)

    # First build race-level observations for all years.
    observations=[]
    daily_files=0
    odds_files=0
    counters=defaultdict(int)
    seen=set()

    # L1.7 intentionally carries race_id/rank output, not calendar race_date.
    # Pair the fixed BACKFILL daily and odds files by their canonical filename,
    # then select rows whose race_id exists in L1.7. This avoids reconstructing
    # or guessing a date from the race_id.
    for day_path in sorted(daily_root.glob("20??-??-??.jsonl.gz")):
        date=day_path.name.removesuffix(".jsonl.gz")
        try:
            year=int(date[:4])
        except ValueError:
            continue
        if year not in YEARS:
            continue
        odds_path=odds_root/day_path.name
        daily_files+=1
        if not odds_path.exists():
            counters["missing_odds_file"]+=1
            continue
        odds_files+=1

        day={}
        wanted=set()
        with gzip.open(day_path,"rt",encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                pack=json.loads(line)
                rid=str((pack.get("race") or {}).get("race_id") or "")
                if rid in l17 and int(l17[rid]["year"])==year:
                    day[rid]=pack
                    wanted.add(rid)
        if not wanted:
            continue
        odds_day=load_odds_day(odds_path,wanted)

        for rid in sorted(wanted):
            rec=l17[rid]
            year=int(rec["year"])
            pack=day.get(rid)
            odds_record=odds_day.get(rid)
            if pack is None:
                counters["missing_daily_row"]+=1
                continue
            if odds_record is None:
                counters["missing_odds_row"]+=1
                continue
            if rid in seen:
                raise ValueError(f"duplicate race={rid}")

            order=[str(x) for x in rec["consensus_order"]]
            hno=horse_number_map(pack)
            market=market_map(pack)
            payouts,present=payout_map(pack)
            tri_odds=decode_trifecta_odds(odds_record)
            if "TRIFECTA" not in present:
                counters["trifecta_payout_missing"]+=1
                continue
            if not tri_odds:
                counters["trifecta_odds_missing"]+=1
                continue

            trusted_ids=order[:min(6,len(order))]
            if any(h not in hno for h in trusted_ids):
                counters["horse_number_missing"]+=1
                continue
            trusted_numbers=[hno[h] for h in trusted_ids]

            for rank in FOCAL_RANKS:
                if rank>len(order):
                    continue
                hid=order[rank-1]
                m=market.get(hid)
                if not m or m["popularity"] is None:
                    counters["market_popularity_missing"]+=1
                    continue
                gap=int(m["popularity"])-rank
                focal_no=hno[hid]

                for pool_k in POOLS:
                    pool=trusted_numbers[:pool_k]
                    if len(pool)<pool_k or focal_no not in pool:
                        counters[f"pool_{pool_k}_ineligible"]+=1
                        continue
                    seat_data={}
                    complete=True
                    for seat in SEATS:
                        tickets=tickets_for_seat(pool,focal_no,seat)
                        prices=[]
                        for t in tickets:
                            price=tri_odds.get(tuple(t))
                            if price is None:
                                complete=False
                                break
                            prices.append(price)
                        if not complete:
                            break
                        mass=sum(1.0/p for p in prices)
                        hit=any(float(payouts.get(("TRIFECTA",tuple(t)),0.0))>0 for t in tickets)
                        seat_data[seat]={
                            "tickets":tickets,
                            "market_mass":mass,
                            "hit":hit,
                        }
                    if not complete:
                        counters[f"pool_{pool_k}_incomplete_odds"]+=1
                        continue

                    observations.append({
                        "race_id":rid,
                        "year":year,
                        "rank":rank,
                        "pool":pool_k,
                        "gap":gap,
                        "seat_data":seat_data,
                        "payouts":payouts,
                    })
            seen.add(rid)

    # Strict prior-year training counts.
    train=defaultdict(empty_train)
    for obs in observations:
        y=obs["year"]
        rank=obs["rank"]
        pool=obs["pool"]
        gap=obs["gap"]
        for variant in VARIANTS:
            group=market_group(variant,gap)
            for seat in SEATS:
                key=(y,variant,rank,pool,group,seat)
                train[key]["races"]+=1
                train[key]["hits"]+=int(obs["seat_data"][seat]["hit"])
                # Rank-only fallback ledger.
                if variant!="RANK_ONLY":
                    k2=(y,"RANK_ONLY",rank,pool,"ALL",seat)
                    # already populated on RANK_ONLY iteration, don't duplicate.
                    pass

    def prior_rates(test_year,variant,rank,pool,group):
        rates={}
        counts={}
        for seat in SEATS:
            n=h=0
            for y in YEARS:
                if y>=test_year:
                    continue
                st=train.get((y,variant,rank,pool,group,seat))
                if st:
                    n+=st["races"]; h+=st["hits"]
            if n==0 and variant!="RANK_ONLY":
                for y in YEARS:
                    if y>=test_year:
                        continue
                    st=train.get((y,"RANK_ONLY",rank,pool,"ALL",seat))
                    if st:
                        n+=st["races"]; h+=st["hits"]
            rates[seat]=h/n if n else None
            counts[seat]=n
        if any(rates[s] is None for s in SEATS):
            return None,None
        return rates,counts

    eval_stats=defaultdict(empty_eval)
    choice_counts=defaultdict(int)
    choice_score_sums=defaultdict(lambda:{"prior":0.0,"mass":0.0,"ratio":0.0})
    eligible_by_year=defaultdict(int)

    for obs in observations:
        test_year=obs["year"]
        if test_year not in TEST_YEARS:
            continue
        rank=obs["rank"]
        pool=obs["pool"]
        gap=obs["gap"]
        masses={s:obs["seat_data"][s]["market_mass"] for s in SEATS}
        total_mass=sum(masses.values())
        if total_mass<=0:
            counters["nonpositive_market_mass"]+=1
            continue
        eligible_by_year[(test_year,rank,pool)]+=1

        for variant in VARIANTS:
            group=market_group(variant,gap)
            rates,counts=prior_rates(test_year,variant,rank,pool,group)
            if rates is None:
                counters["training_missing"]+=1
                continue

            ratios={s:(rates[s]/masses[s] if masses[s]>0 else -1.0) for s in SEATS}
            choices={
                "PROB_ONLY":max(SEATS,key=lambda s:(rates[s],-s)),
                "VALUE_RATIO":max(SEATS,key=lambda s:(ratios[s],-s)),
                "MARKET_CHEAPEST":min(SEATS,key=lambda s:(masses[s],s)),
            }
            for policy,seat in choices.items():
                sd=obs["seat_data"][seat]
                key=(test_year,variant,policy,rank,pool)
                update_eval(eval_stats[key],sd["tickets"],obs["payouts"],masses[seat],rates[seat],ratios[seat])
                ck=(test_year,variant,policy,rank,pool,seat)
                choice_counts[ck]+=1
                choice_score_sums[ck]["prior"]+=rates[seat]
                choice_score_sums[ck]["mass"]+=masses[seat]
                choice_score_sums[ck]["ratio"]+=ratios[seat]

    yearly=[finalize_eval(k,v) for k,v in eval_stats.items()]
    yearly.sort(key=lambda r:(r["year"],r["variant"],r["policy"],r["focal_rank"],r["pool_k"]))

    # OOS combined 2023-2025.
    combined=[]
    for variant in VARIANTS:
        for policy in POLICIES:
            for rank in FOCAL_RANKS:
                for pool in POOLS:
                    sub=[r for r in yearly if r["variant"]==variant and r["policy"]==policy and r["focal_rank"]==rank and r["pool_k"]==pool]
                    races=sum(r["races"] for r in sub)
                    tickets=sum(r["tickets"] for r in sub)
                    hits=sum(r["hit_races"] for r in sub)
                    stake=sum(r["stake_yen"] for r in sub)
                    ret=sum(r["return_yen"] for r in sub)
                    combined.append({
                        "variant":variant,
                        "policy":policy,
                        "focal_rank":rank,
                        "pool_k":pool,
                        "test_years":"2023-2025",
                        "races":races,
                        "tickets":tickets,
                        "avg_tickets_per_race":tickets/races if races else None,
                        "hit_races":hits,
                        "hit_rate_pct":100.0*hits/races if races else None,
                        "stake_yen":stake,
                        "return_yen":ret,
                        "profit_yen":ret-stake,
                        "roi_pct":100.0*ret/stake if stake else None,
                    })
    combined.sort(key=lambda r:r["roi_pct"],reverse=True)

    choices=[]
    for key,n in choice_counts.items():
        y,variant,policy,rank,pool,seat=key
        sums=choice_score_sums[key]
        choices.append({
            "year":y,
            "variant":variant,
            "policy":policy,
            "focal_rank":rank,
            "pool_k":pool,
            "selected_seat":seat,
            "races":n,
            "share_pct_within_policy":None,
            "avg_prior_hit_rate_pct":100.0*sums["prior"]/n if n else None,
            "avg_market_implied_mass":sums["mass"]/n if n else None,
            "avg_value_ratio":sums["ratio"]/n if n else None,
        })
    group_totals=defaultdict(int)
    for r in choices:
        group_totals[(r["year"],r["variant"],r["policy"],r["focal_rank"],r["pool_k"])]+=r["races"]
    for r in choices:
        tot=group_totals[(r["year"],r["variant"],r["policy"],r["focal_rank"],r["pool_k"])]
        r["share_pct_within_policy"]=100.0*r["races"]/tot if tot else None
    choices.sort(key=lambda r:(r["year"],r["variant"],r["policy"],r["focal_rank"],r["pool_k"],r["selected_seat"]))

    coverage=[]
    for y in TEST_YEARS:
        for rank in FOCAL_RANKS:
            for pool in POOLS:
                n=eligible_by_year[(y,rank,pool)]
                denominator=3456 if pool==3 else l17_counts[str(y)] if isinstance(l17_counts,dict) and str(y) in l17_counts else 3456
                coverage.append({
                    "year":y,
                    "focal_rank":rank,
                    "pool_k":pool,
                    "eligible_races":n,
                    "coverage_pct_vs_3456":100.0*n/3456.0,
                })

    write_csv(out_dir/"walkforward-yearly.csv",yearly)
    write_csv(out_dir/"walkforward-combined.csv",combined)
    write_csv(out_dir/"seat-choice-counts.csv",choices)
    write_csv(out_dir/"coverage.csv",coverage)

    summary={
        "contract":"L2_SEAT_MARKET_VALUE_WF_V1",
        "idea":"Choose the trifecta seat using only prior-year formation hit probability and current final trifecta market odds; compare probability-only and market-cheapest controls.",
        "years":[2022,2023,2024,2025],
        "test_years":[2023,2024,2025],
        "l17_races_by_year":l17_counts,
        "focal_ranks":[1,2,3],
        "pools":[3,6],
        "variants":list(VARIANTS),
        "policies":{
            "PROB_ONLY":"max prior-year formation hit probability",
            "VALUE_RATIO":"max prior hit probability / current summed implied probability of the seat formation",
            "MARKET_CHEAPEST":"min current summed implied probability, ignoring prior hit probability"
        },
        "market_mass":"sum(1/final_trifecta_odds) over all tickets in the seat formation",
        "selection_uses_current_market_odds":True,
        "selection_uses_current_popularity_for_group":True,
        "selection_uses_test_outcome":False,
        "selection_uses_payout":False,
        "payout_evaluation_only":True,
        "rerank_l1":False,
        "threshold_filtering":False,
        "2026_locked":True,
        "daily_files_scanned":daily_files,
        "odds_files_scanned":odds_files,
        "counters":dict(counters),
        "best_oos_combined":combined[0] if combined else None,
    }
    (out_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out_dir/"README.md").write_text(
        "# L2 Seat Market Value WF V1\n\n"
        "Strict walk-forward seat selection for trifecta. L1.7 ranks are fixed. "
        "The VALUE_RATIO policy compares prior-year formation hit probability with the current market's summed implied probability for each seat. "
        "No test-year outcome or payout enters selection. No learned skip threshold. 2026 is sealed.\n",
        encoding="utf-8",
    )
    print("L2_SEAT_MARKET_VALUE_WF_V1_DONE")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print("TOP_COMBINED")
    for row in combined[:18]:
        print(row)


if __name__=="__main__":
    main()
