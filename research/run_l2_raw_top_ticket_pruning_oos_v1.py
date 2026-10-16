#!/usr/bin/env python3
"""Frozen L2 original ticket pruning (20 variants) on RAW AI Top1 axis.

152 ORIGINAL ticket templates x 20 FINAL-odds bands; no alteration to frozen
L2.1. Prior-year-only feature thresholds, train=2023 / test=2024, then
train=2023+2024 / test=2025. 2026 intentionally sealed.
Input: pinned BACKFILL, previously audited RAW 3,040-band baseline.
Speed: one historical pass for all policies, masks cached once per physical
ticket, vectorized group totals, sort each training threshold group once
instead of five times. All formulas and tie semantics from frozen original.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import run_l2_fixed_ticket_pruning_3040_v1 as fixed
import run_l2_raw_top_odds_boundary_v1 as raw
from build_l2_market_helpers_v1 import load_day, load_odds_day
from run_all_race_flow_ticket_audit_v1 import YEARS, RACES_YEAR, TOTAL_RACES, assert_ok

POLICIES = tuple(fixed.POLICIES)  # BASE and 4 modes x 5 kept percentages.
MODES = tuple(fixed.MODES)
KEEPS = tuple(fixed.KEEP)
BANDS = tuple(raw.fixed.bin_name(z) for z in raw.BINS)
TEMPLATES = tuple(raw.TEMPLATES)
TEST_YEARS = ("2024", "2025", "2024-2025")
assert len(POLICIES) == 21 and len(TEMPLATES) == 152 and len(BANDS) == 20


def fast_prior_thresholds(history, stats):
    """Exact frozen create_thresholds() math, sorting each group only ONCE."""
    dist = defaultdict(list)
    whole = defaultdict(list)
    for card in history:
        group = (card[2][0], card[2][1])
        scores = fixed.calc_scores(card, stats)
        for mode, value in scores.items():
            dist[(mode, group)].append(value)
            whole[(mode, group[0])].append(value)
    sorted_whole = {key: sorted(val) for key, val in whole.items()}
    limits = {}
    for (mode, group), arr in dist.items():
        vals = sorted(arr) if len(arr) >= 100 else sorted_whole[(mode, group[0])]
        for keep in KEEPS:
            index = min(len(vals)-1, int(len(vals)*(100-keep)/100))
            limits[(mode, keep, group)] = vals[index]
    for (mode, kind), vals in sorted_whole.items():
        for keep in KEEPS:
            index = min(len(vals)-1, int(len(vals)*(100-keep)/100))
            limits[(mode, keep, kind)] = vals[index]
    return limits


def calc(n, h, payout):
    stake = 100*int(n)
    profit = int(payout) - stake
    color = ("NO_TICKETS" if n == 0 else "BLACK" if profit > 0
             else "RED" if profit < 0 else "BREAK_EVEN")
    return dict(tickets=int(n), hits=int(h), stake_yen=stake,
                payout_yen=int(payout), profit_yen=profit,
                roi_pct=round(100*int(payout)/stake, 6) if stake else None,
                color=color)


def read_baseline(path):
    baseline = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["period"] not in ("2023", "2024", "2025", "2023-2025"):
                raise ValueError("unexpected baseline period")
            key = (row["period"], row["partner_order"], int(row["other_horses_k"]),
                   row["family"], row["odds_band"])
            if key in baseline:
                raise ValueError("duplicated baseline key")
            baseline[key] = (int(row["tickets"]), int(row["hit_tickets"]),
                             int(row["payout_yen"]))
    assert_ok(len(baseline)==4*3040, "original raw baseline 3040 x 4 drift")
    return baseline


def process_year(year, prediction_group, picks, yearly_races, backfill_root,
                 statistics, thresholds, future_cards):
    """One physical-ticket score per race; process all 20 masks in one pass."""
    y=str(year)
    by_race=yearly_races.set_index("race_id")
    grouped_days=yearly_races.groupby("race_date")["race_id"].apply(list)
    line_stats=defaultdict(lambda: np.zeros((len(POLICIES), 3), dtype=np.int64))
    dedup_stats=np.zeros((len(POLICIES), 3), dtype=np.int64)
    train=defaultdict(lambda:[0,0,0.])
    num_races=0
    physical_count=0
    same_axis=0

    for date, race_ids in grouped_days.items():
        day=str(date)[:10]
        ids=set(race_ids)
        packs=load_day(backfill_root/"data"/"daily"/(day+".jsonl.gz"),ids)
        odds=load_odds_day(backfill_root/"data"/"odds"/"daily"/(day+".jsonl.gz"),ids)
        for rid in sorted(race_ids):
            pack=packs.get(rid)
            market=odds.get(rid)
            assert_ok(pack is not None and market is not None, "missing race "+str(rid))
            assert_ok(market.get("odds_kind")=="FINAL" and
                      market.get("source_status")=="result", "not FINAL "+str(rid))
            g=prediction_group.get_group(rid)
            info=by_race.loc[rid]
            axis=str(info["AI_RAW_TOP_horse"])
            assert_ok(axis==str(g.sort_values(
                ["distill_ai_rank","market_rank","_horse_id"],
                ascending=[True,True,True],kind="mergesort").iloc[0]["_horse_id"]),
                "raw ranking mismatch "+rid)
            assert_ok(picks["ANCHOR_FULL"][rid]==str(info["VALUE_EV_horse"]),
                      "EV historical pick mismatch "+rid)
            same_axis+=int(axis==picks["ANCHOR_FULL"][rid])
            # Strict adapter for the original frozen function's EV-axis assert.
            fake=info.copy()
            fake["VALUE_EV_horse"]=axis

            cards=[]
            positions={}
            line_positions=defaultdict(list)
            line_seen=defaultdict(set)
            for tid,ticket,card in fixed.create_cards(g,pack,market,rid,axis,fake):
                assert_ok(tid[:3] in TEMPLATES and tid[3] in BANDS, "template or band drift")
                if ticket in line_seen[tid]:
                    raise ValueError("duplicate physical ticket in template")
                line_seen[tid].add(ticket)
                n=positions.get(ticket)
                if n is None:
                    n=len(cards)
                    positions[ticket]=n
                    cards.append(card)
                else:
                    assert_ok(cards[n] == card, "repeated physical ticket card mismatch")
                line_positions[tid].append(n)
            del positions, line_seen
            count=len(cards)
            physical_count+=count
            if count:
                pays=np.fromiter((int(c[1]) for c in cards),dtype=np.int64,count=count)
                hits=(pays>0).astype(np.int64)
                votes=np.zeros((count,len(POLICIES)),dtype=np.int8)
                votes[:,0]=1
                if year>2023:
                    for idx,card in enumerate(cards):
                        flags=fixed.mask(card,statistics,thresholds)
                        for p,policy in enumerate(POLICIES[1:],1):
                            votes[idx,p]=int(flags[policy])
                # One computation for distinct physical tickets / policy / race.
                dedup_stats[:,0]+=votes.sum(axis=0,dtype=np.int64)
                dedup_stats[:,1]+=hits @ votes
                dedup_stats[:,2]+=pays @ votes
                for tid,inds in line_positions.items():
                    relevant=votes[inds]
                    output=line_stats[tid]
                    output[:,0]+=relevant.sum(axis=0,dtype=np.int64)
                    output[:,1]+=hits[inds] @ relevant
                    output[:,2]+=pays[inds] @ relevant

                # Exactly one addition per distinct ticket, AFTER evaluation.
                # Training results are only applied at NEXT YEAR'S boundary.
                for card in cards:
                    fixed.add_learning(train,card)
                # Feature history must contain every physical ticket, once,
                # from all prior years; do not learn from same-year payouts.
                if year < 2025:
                    future_cards.extend(cards)
            num_races+=1
            if num_races%1200==0:
                print("RAW_PRUNE_PROGRESS",year,num_races,"PHYSICAL",physical_count,
                      flush=True)
    assert_ok(num_races==RACES_YEAR,"yearly race coverage drift "+y)
    print("RAW_PRUNE_YEAR_DONE",year,"PHYSICAL",physical_count,
          "UNIQUE_LINE_KEYS",len(line_stats),flush=True)
    return line_stats,dedup_stats,train,physical_count,same_axis


def assert_exact_original(year, line_stats, old):
    """Every 3,040 line's BASE counts and payouts must match saved Phase 1."""
    errors=[]
    for order,k,family in TEMPLATES:
        for band in BANDS:
            key=(str(year),order,k,family,band)
            expected=old[key]
            arr=line_stats.get((order,k,family,band))
            actual=(0,0,0) if arr is None else tuple(int(z) for z in arr[0])
            if actual!=expected:
                errors.append((key,expected,actual))
                if len(errors)>=3: break
        if len(errors)>=3: break
    assert_ok(not errors,"baseline exact match failed "+repr(errors))


def write_csv(path,rows):
    assert_ok(bool(rows),"empty output file "+str(path))
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyze_baseline_and_pruning(results,old):
    """Compare same 2024-25 races/tickets (no 2023 in out-of-sample masks)."""
    band_rows=[]
    pooled_rows=[]
    border_rows=[]
    policy_rows=[]
    reference={}  # policy, period, tid -> metrics for odds band
    portfolio=defaultdict(lambda: np.zeros(3,dtype=np.int64))

    for year in (2024,2025):
        line_stats,dedup_stats,*_=results[year]
        for p,policy in enumerate(POLICIES):
            portfolio[(str(year),policy)]+=dedup_stats[p]
            portfolio[("2024-2025",policy)]+=dedup_stats[p]
        for order,k,family in TEMPLATES:
            for band in BANDS:
                tid=(order,k,family,band)
                key=(str(year),*tid)
                before=old[key]
                after=line_stats.get(tid)
                for p,policy in enumerate(POLICIES):
                    n,h,pay=(int(v) for v in after[p]) if after is not None else (0,0,0)
                    c=calc(n,h,pay)
                    reference[(str(year),policy,*tid)]=c
                    b=calc(*before)
                    band_rows.append(dict(
                        period=str(year),policy=policy,
                        partner_order=order,other_horses_k=k,family=family,
                        odds_band=band,band_floor=raw.BINS[BANDS.index(band)],
                        before_tickets=b["tickets"],before_hits=b["hits"],
                        before_stake_yen=b["stake_yen"],before_payout_yen=b["payout_yen"],
                        before_roi_pct=b["roi_pct"],before_color=b["color"],
                        after_tickets=c["tickets"],after_hits=c["hits"],
                        after_stake_yen=c["stake_yen"],after_payout_yen=c["payout_yen"],
                        after_profit_yen=c["profit_yen"],after_roi_pct=c["roi_pct"],
                        after_color=c["color"],
                        color_change=b["color"]+"->"+c["color"]))

    for order,k,family in TEMPLATES:
        for band in BANDS:
            pre=tuple(sum(old[(y,order,k,family,band)][i]
                          for y in ("2024","2025")) for i in range(3))
            b=calc(*pre)
            for policy in POLICIES:
                tuples=[reference[(y,policy,order,k,family,band)]
                        for y in ("2024","2025")]
                c=calc(*(sum(row[key] for row in tuples)
                         for key in ("tickets","hits","payout_yen")))
                reference[("2024-2025",policy,order,k,family,band)]=c
                pooled_rows.append(dict(
                    period="2024-2025",policy=policy,
                    partner_order=order,other_horses_k=k,family=family,
                    odds_band=band,band_floor=raw.BINS[BANDS.index(band)],
                    before_tickets=b["tickets"],before_hits=b["hits"],
                    before_roi_pct=b["roi_pct"],before_color=b["color"],
                    after_tickets=c["tickets"],after_hits=c["hits"],
                    after_roi_pct=c["roi_pct"],after_profit_yen=c["profit_yen"],
                    after_color=c["color"],
                    red_to_black=int(b["color"]=="RED" and c["color"]=="BLACK"),
                    black_to_red=int(b["color"]=="BLACK" and c["color"]=="RED")
                ))

    def first_black(arr):
        return next((row[0] for row in arr if row[1]=="BLACK"),None)

    for policy in POLICIES:
        lines=[r for r in pooled_rows if r["policy"]==policy]
        # Includes NO_TICKETS separately; no selecting on test outcomes.
        counts=Counter(row["after_color"] for row in lines)
        before=Counter(row["before_color"] for row in lines)
        before_black=sum(r["before_color"]=="BLACK" for r in lines)
        after_black=sum(r["after_color"]=="BLACK" for r in lines)
        count_redblack=sum(r["red_to_black"] for r in lines)
        count_blackred=sum(r["black_to_red"] for r in lines)
        portfolio_data=calc(*portfolio[("2024-2025",policy)])
        annual=[
            calc(*portfolio[(str(year),policy)]) for year in (2024,2025)
        ]
        policy_rows.append(dict(
            policy=policy,train="2023 then 2023-2024",test="2024-2025",
            baseline_black_lines=before_black,
            post_black_lines=after_black,
            red_to_black_lines=count_redblack,
            black_to_red_lines=count_blackred,
            post_red_lines=counts["RED"],
            post_no_tickets_lines=counts["NO_TICKETS"],
            post_break_even_lines=counts["BREAK_EVEN"],
            physical_tickets=portfolio_data["tickets"],
            physical_hit_tickets=portfolio_data["hits"],
            physical_stake_yen=portfolio_data["stake_yen"],
            physical_payout_yen=portfolio_data["payout_yen"],
            physical_roi_pct=portfolio_data["roi_pct"],
            physical_profit_yen=portfolio_data["profit_yen"],
            year_2024_roi_pct=annual[0]["roi_pct"],
            year_2025_roi_pct=annual[1]["roi_pct"],
            note="2024+2025 complete ticket union, not selected black lines."
        ))

        for order,k,family in TEMPLATES:
            base_bin=[];post_bin=[];base_floor=[];post_floor=[]
            cumulative_before=np.zeros(3,dtype=np.int64)
            cumulative_after=np.zeros(3,dtype=np.int64)
            for band in BANDS:
                b=calc(*(sum(old[(year,order,k,family,band)][i]
                             for year in ("2024","2025")) for i in range(3)))
                a=reference[("2024-2025",policy,order,k,family,band)]
                floor=raw.BINS[BANDS.index(band)]
                base_bin.append((floor,b["color"]))
                post_bin.append((floor,a["color"]))
            for band in reversed(BANDS):
                row=reference[("2024-2025",policy,order,k,family,band)]
                cumulative_after+=np.array([row[x] for x in ("tickets","hits","payout_yen")])
                cumulative_before+=np.array([
                    sum(old[(yr,order,k,family,band)][i] for yr in ("2024","2025"))
                    for i in range(3)])
                floor=raw.BINS[BANDS.index(band)]
                base_floor.append((floor,calc(*cumulative_before)["color"]))
                post_floor.append((floor,calc(*cumulative_after)["color"]))
            # For 'first profitable minimum odds floor', scan ascending.
            border_rows.append(dict(
                policy=policy,partner_order=order,other_horses_k=k,family=family,
                before_first_black_band=first_black(base_bin),
                after_first_black_band=first_black(post_bin),
                before_first_black_min_odds_floor=first_black(reversed(base_floor)),
                after_first_black_min_odds_floor=first_black(reversed(post_floor)),
                before_black_bands=sum(c=="BLACK" for _,c in base_bin),
                after_black_bands=sum(c=="BLACK" for _,c in post_bin),
                before_black_min_odds_floors=sum(c=="BLACK" for _,c in base_floor),
                after_black_min_odds_floors=sum(c=="BLACK" for _,c in post_floor),
                red_to_black_bands=sum(b=="RED" and a=="BLACK"
                                        for (_,b),(_,a) in zip(base_bin,post_bin)),
                black_to_red_bands=sum(b=="BLACK" and a=="RED"
                                        for (_,b),(_,a) in zip(base_bin,post_bin)),
            ))
    return band_rows,pooled_rows,border_rows,policy_rows


def main():
    ap=argparse.ArgumentParser()
    for arg in ("predictions","picks","racewise","old-raw-bands",
                "backfill-root","out-dir"):
        ap.add_argument("--"+arg,required=True)
    args=ap.parse_args()
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    old=read_baseline(Path(args.old_raw_bands))
    data,picks,races=raw.yearly_data(args.predictions,args.picks,args.racewise)
    history=[]
    statistics=defaultdict(lambda:[0,0,0.])
    results={}
    same=0
    for year in YEARS:
        thresholds=fast_prior_thresholds(history,statistics) if year>2023 else {}
        ry=races[races.year.eq(year)]
        groups=data[data.year.eq(year)].groupby("_race_id",sort=False)
        result=process_year(year,groups,picks,ry,Path(args.backfill_root),
                            statistics,thresholds,history)
        result_year={2023:0,2024:1,2025:2}[year]
        assert_exact_original(year,result[0],old)
        # The first pass and this pass must choose identical RAW horses.
        same+=result[4]
        results[year]=result
        if year<2025:
            fixed.merge(statistics,result[2])
        print("PRUNING_YEAR",year,"BASE_MATCHED_EXACTLY",True,
              "TRAIN_HISTORY",len(history),"DISTINCT_PHYSICAL",result[3],flush=True)
    assert_ok(same==2596,"RAW-vs-EV same-horse count drift")
    bands,pooled,borders,policies=analyze_baseline_and_pruning(results,old)
    assert_ok(len(bands)==2*21*3040,"yearly policy x line coverage")
    assert_ok(len(pooled)==21*3040,"pooled policy x line coverage")
    assert_ok(len(borders)==21*152,"policy x template boundaries")
    assert_ok(len(policies)==21,"all 20 prune policies")
    for year in (2024,2025):
        total=sum(int(results[year][0][tid][0,0]) for tid in results[year][0])
        assert_ok(total>0,"unexpected empty year")
    for y in (2024,2025):
        assert_ok(results[y][1][0,0]>0,"empty BASE union")
    # The original 2023-25 de-duplicated union is 3,972,199 tickets.
    assert_ok(sum(int(results[y][1][0,0]) for y in YEARS)==3972199,
              "physical BASE count diverged from phase 1")
    files={
        "pruning_2024_2025_all_3040_odds_bands_by_year.csv":bands,
        "pruning_2024_2025_pooled_3040_odds_bands.csv":pooled,
        "pruning_2024_2025_152_template_odds_border_moves.csv":borders,
        "pruning_2024_2025_20_policy_comparison.csv":policies,
    }
    for name,rows in files.items():
        write_csv(out/name,rows)
    s=dict(
        contract="L2_RAW_TOP_TICKET_PRUNING_OOS_V1",
        baseline_phase1_exact_3040x3_matched=True,
        physical_base_2023_2025=3972199,
        same_axis_raw_and_ev_races=same,
        policy_count=20, retained_target_rates=list(KEEPS),
        policy_modes=list(MODES), policy_count_including_baseline=len(POLICIES),
        training={"2024":[2023],"2025":[2023,2024]},
        holdout_years=[2024,2025],year_2026_sealed=True,
        previous_selection="fixed original 152 templates x 20 disjoint FINAL odds bins",
        selection_method="frozen fixed_ticket_pruning_3040 threshold/scores/mask",
        scoring_once_per_distinct_physical_ticket=True,
        load_historical_files_once=True,
        thresholds_sort_once_per_training_score_group=True,
        physical_stake_yen=100,
        no_L21_contract_or_axis_mutation=True,
        market_prices_are_final_not_prebet=True,
        no_test_year_labels_in_threshold_training=True,
        results_are_descriptive_not_new_frozen_rules=True,
        policy_comparison=policies
    )
    (out/"summary.json").write_text(json.dumps(s,ensure_ascii=False,indent=2)+"\n",
                                    encoding="utf-8")
    print("RAW_TICKET_PRUNING_OOS_COMPLETE",json.dumps({
        "baseline_exact":True,
        "policies":len(policies),
        "overall_base_2024_2025":next(p for p in policies if p["policy"]=="BASE"),
        "largest_red_to_black":sorted(policies,key=lambda p:p["red_to_black_lines"],
                                      reverse=True)[:3]},
        ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
