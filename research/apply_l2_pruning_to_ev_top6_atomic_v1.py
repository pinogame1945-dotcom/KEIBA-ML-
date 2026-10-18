#!/usr/bin/env python3
"""Reuse L2 ticket-pruning V1, apply ALL 20 historical policies to ALL exact-rank tickets.

No change to L2.1 frozen STANDARD/STRICT. Uses unchanged pruning functions:
create_cards, add_learning, create_thresholds, mask, POLICIES. Training source
is the identical prior original-152-universe 2023/2024 ticket pool. Evaluation
is chronological: train 2023/test 2024; train 2023+24/test 2025. All 6,870
EV1-6 x AI_RAW numbered ticket strategies are measured, including baseline
red/black -> after red/black for every policy. 2026 sealed.

One additional retrospective scan is needed: prior atomic rank CSV has settled
totals, but cannot reconstruct per-ticket probability/market/price scores after
aggregation. AI probabilities and raw ranks are reused, never recomputed.
"""
import argparse
import csv
import gzip
import itertools
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import run_l2_fixed_ticket_pruning_3040_v1 as prune
import run_l2_all_bet_types_ticket_odds_gate_v1 as prices
import run_l2_value_anchor_bet_frontier_v1 as old
from run_l2_ai_market_rank_betfrontier_v1 import load_predictions
from run_all_race_flow_ticket_audit_v1 import YEARS, RACES_YEAR, TOTAL_RACES, assert_ok
from build_l2_market_helpers_v1 import load_day, load_odds_day, horse_number_map

TEST_YEARS = (2024, 2025)
POLICIES = prune.POLICIES
FAMILIES = ("WIN", "PLACE", "QUINELLA", "WIDE", "EXACTA_FIRST",
            "EXACTA_SECOND", "TRIO", "TRIFECTA_FIRST", "TRIFECTA_SECOND",
            "TRIFECTA_THIRD")


def get_original_atomic(path):
    index = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") if str(path).endswith(".gz") else open(
            path, "r", encoding="utf-8", newline="") as fd:
        for row in csv.DictReader(fd):
            if row["year"] not in ("2024", "2025", "2023-2025"):
                continue
            k = (row["year"], int(row["ev_axis_rank"]), row["family"],
                 int(row["raw_ai_rank_1"]), int(row["raw_ai_rank_2"]))
            assert_ok(k not in index, "duplicate atomic baseline")
            index[k] = (int(row["purchased_tickets"]), int(row["hit_tickets"]),
                        int(float(row["return_yen"])))
    assert_ok(len(index) >= 6870, "atomic source incomplete")
    return index


def card_from_ticket(ticket, axis_num, market, nums_to_horse, positions, probs, price):
    kind, seq = ticket
    partners = [n for n in seq if n != axis_num]
    assert_ok(len(partners) == len(seq) - 1, "selected axis repeated")
    orientation = seq.index(axis_num) + 1 if kind in ("EXACTA","TRIFECTA") else 0
    mkt = (sum(positions["MARKET"][n] for n in partners)/len(partners)
           if partners else positions["MARKET"][axis_num])
    ai = (sum(positions["AI_PROB"][n] for n in partners)/len(partners)
          if partners else positions["AI_PROB"][axis_num])
    raw = (sum(positions["AI_RAW"][n] for n in partners)/len(partners)
           if partners else positions["AI_RAW"][axis_num])
    p = (sum(max(.000001, probs[nums_to_horse[n]]) for n in partners)/len(partners)
         if partners else probs[nums_to_horse[axis_num]])
    p = max(.00001, min(.99999, float(p)))
    bz = bisect_right(prune.BINS, price) - 1
    assert_ok(bz >= 0 and bz < len(prune.BINS), "invalid price bin")
    feat = (kind, bz, orientation, min(5,int((mkt-1)//3)),
            min(5,int((ai-1)//3)))
    pr = math.log(p) + .08*math.log(max(.00001,probs[nums_to_horse[axis_num]]))
    mk = -math.log1p(mkt) - .15*math.log1p(positions["MARKET"][axis_num])
    dis = (mkt-raw)/(mkt+raw+1)
    # Exact frozen field structure; the monetary payout is only for settlement,
    # and is never used inside prune.mask.
    return (float(price), 0, feat, float(pr), float(mk), float(dis))


def generate(axis_num, ranks):
    """Every named AI rank, no top-K collapse; physically duplicate-free."""
    yield ("WIN",0,0), ("WIN",(axis_num,))
    yield ("PLACE",0,0), ("PLACE",(axis_num,))
    others = [(i,n) for i,n in enumerate(ranks,1) if n != axis_num]
    for j,n in others:
        yield ("QUINELLA",j,0), ("QUINELLA",tuple(sorted((axis_num,n))))
        yield ("WIDE",j,0), ("WIDE",tuple(sorted((axis_num,n))))
        yield ("EXACTA_FIRST",j,0), ("EXACTA",(axis_num,n))
        yield ("EXACTA_SECOND",j,0), ("EXACTA",(n,axis_num))
    for (j,a),(k,b) in itertools.combinations(others,2):
        yield ("TRIO",j,k), ("TRIO",tuple(sorted((axis_num,a,b))))
        yield ("TRIFECTA_FIRST",j,k), ("TRIFECTA",(axis_num,a,b))
        yield ("TRIFECTA_FIRST",k,j), ("TRIFECTA",(axis_num,b,a))
        yield ("TRIFECTA_SECOND",j,k), ("TRIFECTA",(a,axis_num,b))
        yield ("TRIFECTA_SECOND",k,j), ("TRIFECTA",(b,axis_num,a))
        yield ("TRIFECTA_THIRD",j,k), ("TRIFECTA",(a,b,axis_num))
        yield ("TRIFECTA_THIRD",k,j), ("TRIFECTA",(b,a,axis_num))


def make_stats():
    return [[0,0,0] for _ in POLICIES]


def evaluate_race(year, rid, group, pack, market, stats, limits,
                  counters, missing, meta):
    number = horse_number_map(pack)
    nums_to_horse = {no:hid for hid,no in number.items()}
    rank = {}
    raw = group.sort_values(["distill_ai_rank","market_rank","_horse_id"],
                             ascending=[True,True,True],kind="mergesort")[
                                 "_horse_id"].astype(str).tolist()
    prob = group.sort_values(["P__ANCHOR_FULL","market_rank","_horse_id"],
                              ascending=[False,True,True],kind="mergesort")[
                                 "_horse_id"].astype(str).tolist()
    market_horse = group.sort_values(["market_rank","final_win_odds","_horse_id"],
                                     ascending=[True,True,True],kind="mergesort")[
                                         "_horse_id"].astype(str).tolist()
    ranked = {"MARKET":market_horse, "AI_PROB":prob, "AI_RAW":raw}
    assert_ok(set(raw)==set(prob)==set(market_horse), "ranking source diverged")
    assert_ok(set(raw).issubset(number), "horse number missing")
    raw_numbers = [number[h] for h in raw]
    for n in raw_numbers:
        assert_ok(n in nums_to_horse, "duplicate horse numbers")
    pos = {name:{number[h]:i+1 for i,h in enumerate(seq)}
           for name,seq in ranked.items()}
    probs = dict(zip(group["_horse_id"].astype(str),
                     group["P__ANCHOR_FULL"].astype(float)))
    vals = group.assign(__v=group["P__ANCHOR_FULL"]*group["final_win_odds"])
    ev_ids = vals.sort_values(["__v","P__ANCHOR_FULL","market_rank","_horse_id"],
                              ascending=[False,False,True,True],kind="mergesort")[
                                  "_horse_id"].astype(str).tolist()
    pays = old.payouts(pack,rid)
    price_cache = {}
    scored = 0
    for r,axis in enumerate(ev_ids[:6],1):
        an = number[axis]
        for spec,ticket in generate(an,raw_numbers):
            f,j,k = spec
            key = (r,f,j,k)
            arr = counters[key]
            if arr is None:
                arr = counters[key] = make_stats()
            value = pays.get(ticket,0)
            base = arr[0]
            base[0]+=1
            base[1]+=int(value>0)
            base[2]+=int(value)
            if ticket not in price_cache:
                price_cache[ticket] = prices.ticket_price(ticket,market)
            price = price_cache[ticket]
            if price is None:
                # No fabricated prices: leave an explicit unscorable tally.
                # Keep ticket in BASE; original old pruning would reject
                # missing-priced tickets. For policy evaluation, the unscorable
                # ticket is marked unretained; effect is separately quantified.
                missing[key]+=1
                meta["unscorable_ticket_appearances"] += 1
                continue
            card = card_from_ticket(ticket,an,market,nums_to_horse,pos,probs,price)
            flags = prune.mask(card,stats,limits)
            for i,policy in enumerate(POLICIES[1:],1):
                if flags[policy]:
                    row=arr[i]
                    row[0]+=1
                    row[1]+=int(value>0)
                    row[2]+=int(value)
            scored+=1
    meta["race_count"]+=1
    meta["scored_ticket_appearances"]+=scored


def scan_year(year, data, picks, racewise, root, stat, limits, train):
    frame = data[data.year.eq(year)]
    grps = frame.groupby("_race_id",sort=False)
    rs = racewise[racewise.year.eq(year)]
    assert_ok(len(rs)==RACES_YEAR, f"racewise {year}")
    records=rs.set_index("race_id")
    days=rs.groupby("race_date")["race_id"].apply(list)
    totals=defaultdict(make_stats)
    missing=Counter()
    meta=Counter()
    learned=defaultdict(lambda:[0,0,0.])
    future=[]
    for day,rids in days.items():
        packs=load_day(root/"data"/"daily"/(day+".jsonl.gz"),set(rids))
        odds=load_odds_day(root/"data"/"odds"/"daily"/(day+".jsonl.gz"),set(rids))
        for rid in rids:
            g=grps.get_group(rid)
            pack=packs.get(rid)
            market=odds.get(rid)
            assert_ok(pack is not None and market is not None, f"missing dated input {rid}")
            assert_ok(market.get("odds_kind")=="FINAL" and
                      market.get("source_status")=="result",f"non-final odds {rid}")
            info=records.loc[rid]
            # Training pool must be EXACTLY the already-fixed original-152
            # universe with same ticket de-dup and frozen feature extraction.
            if train:
                unique={}
                for tid,ticket,card in prune.create_cards(
                        g,pack,market,rid,picks["ANCHOR_FULL"][rid],info):
                    unique[ticket]=card
                for card in unique.values():
                    prune.add_learning(learned,card)
                    future.append(card)
                meta["original_training_cards"]+=len(unique)
            if year in TEST_YEARS:
                evaluate_race(year,rid,g,pack,market,stat,limits,
                              totals,missing,meta)
            else:
                meta["race_count"]+=1
        if len(meta) and meta["race_count"]%600 < len(rids):
            print("YEAR_SCAN",year,meta["race_count"],
                  "train_cards",meta["original_training_cards"],
                  "scored",meta["scored_ticket_appearances"],flush=True)
    assert_ok(meta["race_count"]==RACES_YEAR,
              f"race coverage {year}: {meta['race_count']}")
    return totals,missing,meta,learned,future


def export(records, baseline, peryear, unpriced, meta):
    report = []
    index = defaultdict(dict)
    for yr,totals in peryear.items():
        for (r,f,j,k),arr in totals.items():
            index[(yr,r,f,j,k)]=arr
    seen = set()
    for key, (base_n,base_h,base_return) in baseline.items():
        year,r,f,j,k = key
        if year == "2023-2025":
            continue
        if year not in ("2024","2025"):
            continue
        code=(r,f,j,k)
        seen.add(code)
        current=index.get(key)
        if current is None:
            assert_ok(base_n == 0, f"missing paid baseline group {key}")
            current=make_stats()
        assert_ok(current[0] == [base_n,base_h,base_return],
                  f"historical no-prune baseline drift {key}: expected {(base_n,base_h,base_return)}, got {current[0]}")
        for pi,policy in enumerate(POLICIES):
            n,h,ret=current[pi]
            assert_ok(n<=base_n and h<=base_h and ret<=base_return,
                      f"pruning created outcome {key} {policy}")
            row = {
                "year":year, "ev_axis_rank":r, "family":f,
                "raw_ai_rank_1":j, "raw_ai_rank_2":k,
                "policy":policy,
                "before_tickets":base_n, "before_hits":base_h,
                "before_hit_rate_pct":round(100*base_h/base_n,6) if base_n else None,
                "before_stake_yen":base_n*100, "before_return_yen":base_return,
                "before_profit_yen":base_return-100*base_n,
                "before_roi_pct":round(100*base_return/(100*base_n),6) if base_n else None,
                "after_tickets":n, "after_hits":h,
                "after_hit_rate_pct":round(100*h/n,6) if n else None,
                "after_stake_yen":n*100, "after_return_yen":ret,
                "after_profit_yen":ret-100*n,
                "after_roi_pct":round(100*ret/(100*n),6) if n else None,
                "no_price_tickets":unpriced[year].get(code,0),
                "profit_delta_yen":(ret-100*n)-(base_return-100*base_n),
                "red_to_black":int(base_return-100*base_n<0 and ret-100*n>0),
                "black_to_red":int(base_return-100*base_n>0 and ret-100*n<0),
                "positive_before":int(base_return-100*base_n>0),
                "positive_after":int(n>0 and ret-100*n>0),
            }
            report.append(row)
    assert_ok(len(seen)>=6000, f"insufficient tested configs {len(seen)}")
    # Pool 2024+2025, preserving exactly-one-policy-per-strategy comparison.
    pair=defaultdict(dict)
    for row in report:
        ix=(row["ev_axis_rank"],row["family"],row["raw_ai_rank_1"],
            row["raw_ai_rank_2"],row["policy"])
        pair[ix][row["year"]]=row
    pooled=[]
    for key,years in pair.items():
        if len(years)!=2:
            # An absent rank in a year is equivalent to zero exposure.
            pass
        base_n=sum(years[y]["before_tickets"] for y in years)
        base_h=sum(years[y]["before_hits"] for y in years)
        base_ret=sum(years[y]["before_return_yen"] for y in years)
        after_n=sum(years[y]["after_tickets"] for y in years)
        after_h=sum(years[y]["after_hits"] for y in years)
        after_ret=sum(years[y]["after_return_yen"] for y in years)
        bprofit=base_ret-base_n*100
        aprofit=after_ret-after_n*100
        r,f,j,k,policy=key
        pooled.append({
            "year":"2024-2025", "ev_axis_rank":r, "family":f,
            "raw_ai_rank_1":j,"raw_ai_rank_2":k,"policy":policy,
            "before_tickets":base_n,"before_hits":base_h,
            "before_hit_rate_pct":round(100*base_h/base_n,6) if base_n else None,
            "before_stake_yen":base_n*100, "before_return_yen":base_ret,
            "before_profit_yen":bprofit,
            "before_roi_pct":round(100*base_ret/(100*base_n),6) if base_n else None,
            "after_tickets":after_n,"after_hits":after_h,
            "after_hit_rate_pct":round(100*after_h/after_n,6) if after_n else None,
            "after_stake_yen":after_n*100, "after_return_yen":after_ret,
            "after_profit_yen":aprofit,
            "after_roi_pct":round(100*after_ret/(100*after_n),6) if after_n else None,
            "no_price_tickets":sum(years[y]["no_price_tickets"] for y in years),
            "profit_delta_yen":aprofit-bprofit,
            "red_to_black":int(bprofit<0 and aprofit>0),
            "black_to_red":int(bprofit>0 and aprofit<0),
            "positive_before":int(bprofit>0),
            "positive_after":int(after_n>0 and aprofit>0),
        })
    # Original 2023 was training only and should NOT silently be presented as
    # forward-evaluated. Full 2023-25 split baseline remains in atomic input.
    by_policy=[]
    for period,raw in (("2024",report),("2025",report),("2024-2025",pooled)):
        for policy in POLICIES:
            a=[r for r in raw if r["year"]==period and r["policy"]==policy]
            assert_ok(a, "missing policy output")
            by_policy.append({
                "year":period,"policy":policy,"atomic_settings":len(a),
                "before_black":sum(x["positive_before"] for x in a),
                "before_red":sum(x["before_profit_yen"]<0 for x in a),
                "after_black":sum(x["positive_after"] for x in a),
                "after_red":sum(x["after_profit_yen"]<0 for x in a),
                "red_to_black":sum(x["red_to_black"] for x in a),
                "black_to_red":sum(x["black_to_red"] for x in a),
                "before_tickets_sum_non_dedup":sum(x["before_tickets"] for x in a),
                "after_tickets_sum_non_dedup":sum(x["after_tickets"] for x in a),
                "lost_winning_tickets_non_dedup":sum(x["before_hits"]-x["after_hits"] for x in a),
                "saved_losing_tickets_non_dedup":sum(
                    (x["before_tickets"]-x["before_hits"]) -
                    (x["after_tickets"]-x["after_hits"]) for x in a),
                "sum_profit_delta_yen_non_dedup":sum(x["profit_delta_yen"] for x in a),
                "missing_price_tickets_non_dedup":sum(x["no_price_tickets"] for x in a),
            })
    return report, pooled, by_policy


def write_csv(path,rows):
    assert rows
    with open(path,"w",newline="",encoding="utf-8") as fd:
        writer=csv.DictWriter(fd,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    for name in ("predictions","picks","racewise","backfill-root",
                 "atomic-reference","out-dir"):
        ap.add_argument("--"+name,required=True)
    a=ap.parse_args()
    data,picks=load_predictions(a.predictions,a.picks)
    assert_ok(set(data.year)==set(YEARS),"2026 sealed")
    races=pd.read_csv(a.racewise,compression="infer",
                      dtype={"race_id":str,"VALUE_EV_horse":str,"AI_RAW_TOP_horse":str},
                      low_memory=False)
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert_ok(len(races)==TOTAL_RACES and races.race_id.is_unique, "racewise mismatch")
    baseline=get_original_atomic(a.atomic_reference)
    stat=defaultdict(lambda:[0,0,0.])
    training=[]
    peryear={}
    unpriced={}
    metas={}
    for y in YEARS:
        limits=prune.create_thresholds(training,stat) if y>2023 else {}
        totals,missing,meta,learned,future=scan_year(y,data,picks,races,
            Path(a.backfill_root),stat,limits,y!=2025)
        if y in TEST_YEARS:
            peryear[str(y)]=totals
            unpriced[str(y)]=missing
        metas[str(y)]=dict(meta)
        if y != 2025:
            prune.merge(stat,learned)
            training.extend(future)
        print("YEAR_END",y,"train",len(training),"baseline",len(totals),
              "unscorable",meta.get("unscorable_ticket_appearances",0),flush=True)
    report,pooled,by_policy=export(None,baseline,peryear,unpriced,metas)
    dest=Path(a.out_dir)
    dest.mkdir(parents=True,exist_ok=True)
    write_csv(dest/"pruning-all-exact-rank-lines-2024-25.csv",report)
    write_csv(dest/"pruning-all-exact-rank-lines-pooled-2024-25.csv",pooled)
    write_csv(dest/"red-black-transition-per-policy-and-year.csv",by_policy)
    summary={
        "contract":"L2_EV_TOP6_RAW_AI_EXACT_RANK_L2_PRUNING_APPLY_V1",
        "source_prior_pruning":"research/run_l2_fixed_ticket_pruning_3040_v1.py",
        "policies_unmodified":list(POLICIES),
        "historical_training_source":"original 152 template universe, 2023 then 2024",
        "years_2023":"learning only, not pruned",
        "years_2024_2025":"independently compared with prior-year train only",
        "no_ai_feature_recalculation":True,
        "historical_final_ticket_odds_required_for_existing_pruning":True,
        "unscorable_rule":"no ticket price -> BASE counted, skipped for all prune variants; quantified separately",
        "2026_sealed":True,
        "frozen_L2_1_unchanged":True,
        "baseline_parity_checked_all_exact_rank_tickets":True,
        "analysis_only_not_forward_deployable_due_to_final_odds":True,
        "training_meta":metas,"red_black_statistics":by_policy,
        "note":"Sums over 6870 atomic strategies contain overlapping physical tickets and cannot be interpreted as deduplicated portfolio profit.",
    }
    (dest/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("RED_BLACK_COUNTS",json.dumps(
        [r for r in by_policy if r["year"]=="2024-2025"],ensure_ascii=False),
        flush=True)
    print("ALL_6870_PRUNING_COMPLETE",flush=True)


if __name__=="__main__":
    main()
