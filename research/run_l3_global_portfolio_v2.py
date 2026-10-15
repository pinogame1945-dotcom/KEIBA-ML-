#!/usr/bin/env python3
"""L3 GLOBAL PORTFOLIO V2: compare whole frozen L2.1 portfolio stake policies.

All tickets are purchased. This is a historical, in-sample sensitivity study:
L2.1 selected its rules using 2023-25 outcomes, prices are FINAL (post-race).
Neither L2.1 nor this analysis proves future ROI or live executable odds.
No 2026 data. No external services, numpy, pandas, packages or Kaggle.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
import run_l3_money_allocation_v1 as frozen

YEARS = (2023, 2024, 2025)
CAPS = (300, 500)
SCOPES = ("RACE", "DAY")
CORE = ("REVERSE_P3", "OUTSIDER", "P3_OUTSIDER")
DIAGNOSTICS = ("FORWARD_P3", "HASH")
POSTHOC = ("TRIFECTA_REVERSE", "TRIFECTA_OUTSIDER")
# Scope and caps are predeclared. TRIFECTA_* are explicitly exploratory,
# because V1's outcome attribution motivated them.
POLICIES = [
    (strategy, scope, cap)
    for strategy in CORE + POSTHOC + DIAGNOSTICS
    for scope in SCOPES
    for cap in (CAPS if strategy in CORE + POSTHOC else (300,))
]
REFS = ("FLAT100", "FLAT200")
BASE_STAKE = 200
MIN_STAKE = 100
FIELDS = ("policy", "year", "race_date", "race_id", "bet_type", "selection",
          "l175_complete", "stake_yen", "return_yen")
ALLOWED = {"min_p3", "support_min", "mean_p3_delta", "ticket_odds"}


def ticket_key(t):
    return (t["year"], t["race_id"], t["bet_type"], t["selection"])


def percentiles(group, field, reverse=False):
    """Midrank tied numeric values, missing unknown stays None."""
    vals = sorted({t[field] for t in group if t[field] is not None})
    order = {v: (.5 if len(vals) == 1 else i / (len(vals) - 1))
             for i, v in enumerate(vals)}
    return {ticket_key(t): (None if t[field] is None else
            (1 - order[t[field]] if reverse else order[t[field]])) for t in group}


def score_map(strategy, group):
    """No hit, payout or future outcome in score computation."""
    if strategy == "HASH":
        return {ticket_key(t): int(hashlib.sha256(
            "|".join(map(str, ticket_key(t))).encode("utf-8")
        ).hexdigest()[:12], 16) for t in group}
    p3 = percentiles(group, "min_p3", reverse=True)
    support = percentiles(group, "support_min")
    result = {}
    for t in group:
        k = ticket_key(t)
        a, b = p3[k], support[k]
        if strategy == "REVERSE_P3":
            value = a
        elif strategy == "FORWARD_P3":
            value = None if a is None else 1 - a
        elif strategy == "OUTSIDER":
            value = b
        elif strategy == "P3_OUTSIDER":
            value = None if a is None or b is None else (a + b) / 2
        elif strategy == "TRIFECTA_REVERSE":
            value = None if a is None else (2 + a if t["bet_type"] == "TRIFECTA" else a)
        elif strategy == "TRIFECTA_OUTSIDER":
            value = None if b is None else (2 + b if t["bet_type"] == "TRIFECTA" else b)
        else:
            raise ValueError(strategy)
        result[k] = value
    return result


def allocate(group, strategy, cap, budget):
    n = len(group)
    if not n:
        raise ValueError("empty group")
    if budget != BASE_STAKE * n:
        raise ValueError("capital mismatch")
    if cap not in CAPS:
        raise ValueError("unsupported max stake")
    stakes = {ticket_key(t): MIN_STAKE for t in group}
    scores = score_map(strategy, group)
    ordered = sorted(group, key=lambda t: (
        scores[ticket_key(t)] is None,
        -(scores[ticket_key(t)] if scores[ticket_key(t)] is not None else 0),
        ticket_key(t)))
    units = (budget - MIN_STAKE * n) // 100
    for t in ordered:
        take = min((cap-MIN_STAKE)//100, units)
        stakes[ticket_key(t)] += 100 * take
        units -= take
        if not units:
            break
    if units != 0 or any(s % 100 or s < 100 or s > cap for s in stakes.values()):
        raise AssertionError("invalid stakes")
    if sum(stakes.values()) != budget or len(stakes) != n:
        raise AssertionError("budget/ticket conservation failed")
    return stakes


def bankroll_stats(rows):
    total_in = sum(r["stake_yen"] for r in rows)
    total_out = sum(r["return_yen"] for r in rows)
    assert total_in > 0
    winners = sorted([r["return_yen"] for r in rows if r["return_yen"] > 0], reverse=True)
    dates = defaultdict(lambda: [0, 0])
    months = defaultdict(lambda: [0, 0])
    race_set = set()
    unknown = 0
    for r in rows:
        date = r["race_date"]
        dates[date][0] += r["stake_yen"]
        dates[date][1] += r["return_yen"]
        month = date[:7]
        months[month][0] += r["stake_yen"]
        months[month][1] += r["return_yen"]
        race_set.add((r["year"], r["race_id"]))
        unknown += (r["l175_complete"] == 0)
    pnl = [(d, o-i) for d, (i,o) in sorted(dates.items())]
    peak = cumulative = dd = 0
    losing_streak = longest_streak = 0
    for d, val in pnl:
        cumulative += val
        peak = max(peak, cumulative)
        dd = max(dd, peak-cumulative)
        losing_streak = losing_streak + 1 if val < 0 else 0
        longest_streak = max(longest_streak, losing_streak)
    return {
        "races": len(race_set), "tickets": len(rows), "hits": len(winners),
        "missing_signal_tickets": unknown,
        "stake_yen": total_in, "payout_yen": total_out,
        "profit_yen": total_out-total_in, "roi_pct": round(100*total_out/total_in, 5),
        "roi_without_top3_pct": round(100*(total_out-sum(winners[:3]))/total_in, 5),
        "roi_without_top10_pct": round(100*(total_out-sum(winners[:10]))/total_in, 5),
        "biggest_hit_share_pct": round(100*winners[0]/total_out, 5) if winners else 0,
        "max_day_close_drawdown_yen": dd,
        "largest_daily_stake_yen": max(i for i,o in dates.values()),
        "worst_day_profit_yen": min(v for d,v in pnl),
        "days_with_profit": sum(v>0 for d,v in pnl),
        "days_total": len(pnl),
        "monthly_profit_months": sum(o>i for i,o in months.values()),
        "months_total": len(months),
        "max_consecutive_losing_days": longest_streak,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_race = frozen.load()  # validated 13,479 tickets and yearly exact ROI.
    tickets = [t for group in by_race.values() for t in group]
    if len(tickets) != 13479 or len(by_race) != 6877:
        raise AssertionError("L2.1 full-port stake domain changed")
    byday = defaultdict(list)
    for t in tickets:
        byday[t["race_date"]].append(t)
    if len({ticket_key(t) for t in tickets}) != len(tickets):
        raise AssertionError("L2.1 physical-ticket identity changed")

    groups = {}
    for scope in SCOPES:
        groups[scope] = by_race if scope == "RACE" else byday
    policies = list(REFS) + [
        f"{strategy}_{scope}_MAX{cap}" for strategy,scope,cap in POLICIES]
    summaries = []
    annual = []
    months = []
    group_daily = {}
    by_policy = {}
    with gzip.open(out/"full_portfolio_stakes.csv.gz", "wt", newline="", encoding="utf-8", compresslevel=6) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for policy in policies:
            stakes = {}
            if policy in REFS:
                unit = 100 if policy == "FLAT100" else 200
                stakes = {ticket_key(t): unit for t in tickets}
            else:
                strategy,scope,cap = next((s,g,c) for s,g,c in POLICIES
                                          if f"{s}_{g}_MAX{c}" == policy)
                for group in groups[scope].values():
                    batch = allocate(group, strategy, cap, BASE_STAKE*len(group))
                    if set(stakes).intersection(batch):
                        raise AssertionError("duplicate group allocation")
                    stakes.update(batch)
            if len(stakes) != len(tickets):
                raise AssertionError(("ticket coverage",policy,len(stakes)))
            rows = []
            grouped = defaultdict(list)
            daily = defaultdict(int)
            monthly = defaultdict(int)
            for t in tickets:
                st = stakes[ticket_key(t)]
                if st % 100 or st < MIN_STAKE or (policy != "FLAT100" and st > 500):
                    raise AssertionError("invalid ticket amount")
                paid = t["payout_yen_per100"]*(st//100)
                r = {f: t[f] for f in ("year","race_date","race_id","bet_type","selection","l175_complete")}
                r.update({"policy":policy,"stake_yen":st,"return_yen":paid})
                writer.writerow(r)
                rows.append(r)
                grouped[r["year"]].append(r)
                daily[r["race_date"]] += paid-st
                monthly[r["race_date"][:7]] += paid-st
            result = bankroll_stats(rows)
            result["policy"] = policy
            result["delta_profit_vs_flat200_yen"] = (
                None if policy=="FLAT100" else result["profit_yen"]-247520)
            result["historical_scope"] = "IN_SAMPLE_L2_POSTHOC_FINAL_ODDS"
            summaries.append(result)
            by_policy[policy] = result
            group_daily[policy] = daily
            for year in YEARS:
                d=bankroll_stats(grouped[year])
                d["year"] = year
                d["policy"] = policy
                annual.append(d)
            for m,v in monthly.items():
                months.append({"policy":policy,"month":m,"profit_yen":v})
            if policy=="FLAT100" and (result["stake_yen"],result["payout_yen"]) != (1347900,1471660):
                raise AssertionError("FLAT100 differs from frozen contract")
            if policy!="FLAT100" and result["stake_yen"] != 2695800:
                raise AssertionError("capital differs between portfolios")

    # Paired day block bootstrap: illustrative sensitivity only; all predeclared
    # strategies are still subject to selection and multiple-comparison bias.
    rng = random.Random(20261015)
    days = sorted(byday)
    repeats = 800
    bootstrap_indices = [[rng.randrange(len(days)) for _ in days] for _ in range(repeats)]
    bootstrap = []
    for policy in policies:
        if policy in REFS:
            continue
        delta = [group_daily[policy][d]-group_daily["FLAT200"][d] for d in days]
        draw = sorted(sum(delta[i] for i in inds) for inds in bootstrap_indices)
        bootstrap.append({
            "policy":policy,
            "profit_diff_yen":sum(delta),
            "paired_day_resample_95pct_low_yen":draw[int(.025*(repeats-1))],
            "paired_day_resample_95pct_high_yen":draw[int(.975*(repeats-1))],
        })

    def write_csv(path, rows):
        with (out/path).open("w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    write_csv("whole_portfolio_comparison.csv", summaries)
    write_csv("yearly_portfolio_comparison.csv", annual)
    write_csv("monthly_portfolio_profit.csv", sorted(months, key=lambda x:(x["policy"],x["month"])))
    write_csv("paired_daily_bootstrap.csv",bootstrap)
    info = {
        "contract":"KEIBA_L3_GLOBAL_PORTFOLIO_V2",
        "source_frozen_sha":"9655c717333636b5cc931f672e12dbc377e396bc",
        "l2_is_in_sample_posthoc_on_2023_2024_2025": True,
        "all_2026_data_sealed": True,
        "all_historical_ticket_odds_are_final_after_race": True,
        "strong_caution":"These payouts were not a proven pre-race executable price.",
        "no_new_bets_no_skips_no_zero_stakes":True,
        "unknown_signal_215_kept":True,
        "equal_total_capital_for_all_non_flat100_policies": True,
        "daily_grouping_not_live_executable_proof":True,
        "signal_sources_only":"pre-race L1.75 rank<=10 OOS p3 and Outsider support; no outcome in decisions",
        "trifecta_policies_are_exploratory_post_v1_outcome_inspection": True,
        "scope":"Whole 13,479 physical frozen purchases, portfolio-level ROI, profit and risk.",
        "policies":policies,
        "race_count":len(by_race),
        "day_count":len(byday),
        "selected_strategy_claim":"NONE. All policies diagnostics; no model is production ready.",
        "bootstrap":"800 paired resamples of days, descriptive only, not a valid OOS guarantee",
        "all_fixed_bets_match_flat200":True,
    }
    (out/"summary.json").write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L3_GLOBAL_V2_COMPLETE",json.dumps({
        "policies":len(policies),"races":len(by_race),"days":len(byday),
        "tickets":len(tickets),"equal_capital":2695800,
        "top_roi_descriptive":sorted(
            [{"policy":x["policy"],"roi":x["roi_pct"],"profit":x["profit_yen"],
              "maxdd":x["max_day_close_drawdown_yen"],
              "roi_ex_top10":x["roi_without_top10_pct"]}
             for x in summaries if x["policy"]!="FLAT100"],
            key=lambda x:-x["roi"])[:8]
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
