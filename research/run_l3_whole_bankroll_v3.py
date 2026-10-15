#!/usr/bin/env python3
"""Whole-L3 bankroll feasibility and stress test on one FROZEN 13,479-ticket ledger.

NO ticket selection/reordering/repricing. Each bet has a preassigned stake
from prior V2. Bankroll validation never changes these stakes; if unable to
prefund a complete day, it reports INSUFFICIENT_BUDGET instead of skipping.
2026 untouched. L2.1 selected using 2023-25 posthoc outcomes, with FINAL
odds; these simulations CANNOT establish tradable or out-of-sample ROI.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "research-results/l3-global-portfolio-v2/run-38717294078/full_portfolio_stakes.csv.gz"
YEARS = (2023, 2024, 2025)
STARTING_CASH = (10000, 20000, 30000, 50000, 100000, 200000, 300000, 500000)
PAYMENT_HAIRCUTS = (0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25)
STRESS_SAMPLES = 1000
SEED = 20261015
COMPARATORS = ("FLAT200", "REVERSE_P3_RACE_MAX300",
               "P3_OUTSIDER_RACE_MAX300", "OUTSIDER_RACE_MAX300")
FIELDS = ("policy", "year", "race_date", "race_id", "bet_type", "selection",
          "l175_complete", "stake_yen", "return_yen")


def data():
    with gzip.open(INPUT, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if set(FIELDS) != set(reader.fieldnames or []):
            raise ValueError(("input schema drift", reader.fieldnames))
        yield from reader


def state():
    return {"days": defaultdict(lambda: [0, 0, 0]),
            "years": defaultdict(lambda: [0, 0, 0]),
            "tickets": 0, "wins": 0, "unknown": 0,
            "stake": 0, "payout": 0, "top10": [], "tickets_by_type": defaultdict(int),
            "policy": None}


def consume():
    all_states = {}
    original_keys = set()
    last_policy = None
    keys = set()
    seen_policy = set()
    for row in data():
        pol = row["policy"]
        if last_policy is not None and pol != last_policy:
            if keys != original_keys and last_policy != "FLAT100":
                raise AssertionError(("ticket set drift", last_policy, len(keys)))
            if last_policy == "FLAT100":
                original_keys = keys.copy()
            seen_policy.add(last_policy)
            keys = set()
        if pol in seen_policy:
            raise AssertionError(("interleaved policy rows", pol))
        last_policy = pol
        y = int(row["year"])
        if y not in YEARS:
            raise ValueError(("UNSEALED_YEAR", y))
        date = str(row["race_date"])
        if not date.startswith(str(y) + "-") or len(date) != 10:
            raise ValueError(("bad date", date))
        ticket = (y, str(row["race_id"]), str(row["bet_type"]), str(row["selection"]))
        if ticket in keys:
            raise AssertionError(("duplicate frozen ticket", pol, ticket))
        keys.add(ticket)
        st = int(row["stake_yen"])
        ret = int(row["return_yen"])
        if st < 100 or st % 100 or ret < 0 or st > (100 if pol == "FLAT100" else 500):
            raise ValueError(("bad stake/return", pol, ticket, st, ret))
        unknown = int(row["l175_complete"]) == 0
        current = all_states.setdefault(pol, state())
        current["policy"] = pol
        day = current["days"][date]
        day[0] += st
        day[1] += ret
        day[2] += 1
        year = current["years"][y]
        year[0] += st
        year[1] += ret
        year[2] += 1
        current["tickets"] += 1
        current["wins"] += (ret > 0)
        current["unknown"] += unknown
        current["stake"] += st
        current["payout"] += ret
        current["tickets_by_type"][row["bet_type"]] += 1
        if ret > 0:
            if len(current["top10"]) < 10:
                heapq.heappush(current["top10"], ret)
            elif ret > current["top10"][0]:
                heapq.heapreplace(current["top10"], ret)
    if last_policy != "FLAT100" and keys != original_keys:
        raise AssertionError("last policy ticket set mismatch")
    expected = 26
    if len(all_states) != expected:
        raise AssertionError(("policy count drift", len(all_states), expected))
    for pol, s in all_states.items():
        if (s["tickets"], s["wins"], s["unknown"]) != (13479, 376, 215):
            raise AssertionError(("L2.1 membership drift", pol,
                                 s["tickets"],s["wins"],s["unknown"]))
        if len(s["days"]) != 322:
            raise AssertionError(("date coverage drift", pol, len(s["days"])))
        if s["stake"] != (1347900 if pol == "FLAT100" else 2695800):
            raise AssertionError(("capital mismatch", pol,s["stake"]))
        if pol == "FLAT100" and s["payout"] != 1471660:
            raise AssertionError("L2.1 100-yen baseline mismatch")
        if pol == "FLAT200" and s["payout"] != 2943320:
            raise AssertionError("L2.1 200-yen baseline mismatch")
    return all_states


def minimum_initial_cash(days):
    """Exact for must-fund-all-bets-of-day-before-any-return assumption.

    Starting account C, preceding day realized profit P, day's stake S:
    funding possible iff C + P >= S. Minimum is max(0, max(S-P)).
    """
    pnl = 0
    minimum = 0
    peak = 0
    maxdd = 0
    for _,(stake,payout,_) in sorted(days.items()):
        minimum = max(minimum, stake-pnl)
        pnl += payout - stake
        peak = max(peak, pnl)
        maxdd = max(maxdd, peak-pnl)
    return minimum, maxdd


def first_shortage(days, capital):
    cash = capital
    completed = 0
    tickets = 0
    for date,(stake,payout,n) in sorted(days.items()):
        if cash < stake:
            return {
                "status": "INSUFFICIENT_BUDGET", "first_unfundable_date": date,
                "cash_before_day_yen": cash, "day_required_yen": stake,
                "shortage_yen": stake - cash, "days_completed": completed,
                "tickets_completed": tickets,
            }
        cash += payout-stake
        completed += 1
        tickets += n
    return {
        "status": "COMPLETE", "first_unfundable_date": "",
        "cash_before_day_yen": "", "day_required_yen": "",
        "shortage_yen": 0, "days_completed": completed,
        "tickets_completed": tickets, "final_cash_yen": cash,
    }


def percentile(values, p):
    vals=sorted(values)
    return vals[int((len(vals)-1)*p)]


def write_table(out, name, rows):
    if not rows:
        raise AssertionError(("empty report", name))
    with (out/name).open("w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    a=p.parse_args()
    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    states=consume()
    risk_rows=[]
    year_rows=[]
    feasibility=[]
    haircut_rows=[]
    sensitivity=[]
    rng=random.Random(SEED)
    base_days=sorted(states["FLAT200"]["days"])
    index_draws=[[rng.randrange(len(base_days)) for _ in base_days]
                 for _ in range(STRESS_SAMPLES)]
    baseline_by_day=states["FLAT200"]["days"]
    for policy,s in states.items():
        required,dd=minimum_initial_cash(s["days"])
        stake,returns=s["stake"],s["payout"]
        sorted_days=sorted(s["days"])
        if sorted_days != base_days:
            raise AssertionError(("day set drift",policy))
        biggest=max(v[0] for v in s["days"].values())
        if required < s["days"][sorted_days[0]][0]:
            raise AssertionError("minimum capital below the first day's purchase")
        top=sorted(s["top10"],reverse=True)
        year_rois=[]
        for year in YEARS:
            subset={d:v for d,v in s["days"].items() if d.startswith(str(year) + "-")}
            required_y,dd_y=minimum_initial_cash(subset)
            spent_y,paid_y,n_y=s["years"][year]
            year_rois.append(100*paid_y/spent_y)
            year_rows.append({
                "policy":policy,"year":year,"stake_yen":spent_y,
                "payout_yen":paid_y, "profit_yen":paid_y-spent_y,
                "roi_pct":round(100*paid_y/spent_y,6),
                "min_starting_cash_daily_prefund_yen":required_y,
                "max_daily_close_dd_yen":dd_y,"tickets":n_y,
            })
        risk_rows.append({
            "policy":policy,"stake_yen":stake,"payout_yen":returns,
            "profit_yen":returns-stake,
            "roi_pct":round(100*returns/stake,6),
            "min_starting_cash_daily_prefund_yen":required,
            "max_daily_close_dd_yen":dd,
            "max_single_day_stake_yen":biggest,
            "all_years_historical_positive": all(x>100 for x in year_rois),
            "year_roi_min_pct":round(min(year_rois),6),
            "year_roi_max_pct":round(max(year_rois),6),
            "roi_minus_top3_pct":round(100*(returns-sum(top[:3]))/stake,6),
            "roi_minus_top10_pct":round(100*(returns-sum(top))/stake,6),
            "top10_winning_return_share_pct":round(100*sum(top)/returns,6),
            "historical_profit_to_min_required_cash_pct":
                round(100*(returns-stake)/required,6) if required else "",
        })
        for money in STARTING_CASH:
            outcome=first_shortage(s["days"],money)
            feasibility.append({"policy":policy,"starting_cash_yen":money,**outcome})
            if money >= required and outcome["status"] != "COMPLETE":
                raise AssertionError(("wrong insolvency threshold",policy,money,required))
            if money < required and outcome["status"] != "INSUFFICIENT_BUDGET":
                raise AssertionError(("wrong feasible threshold",policy,money,required))
        for haircut in PAYMENT_HAIRCUTS:
            reduced=returns*(1-haircut)
            haircut_rows.append({
                "policy":policy,
                "hypothetical_payout_reduction_pct":round(100*haircut,2),
                "stress_return_yen":round(reduced,2),
                "stress_profit_yen":round(reduced-stake,2),
                "stress_roi_pct":round(100*reduced/stake,6),
                "assumption":"all positive payouts uniformly reduced; not JRA execution simulation",
            })
        if policy=="FLAT100":
            # Different funding baseline, shown for reference; other policies
            # have identical investment on each day, so paired comparisons work.
            continue
        # Day bootstrap with replacement. This is a historical resample, not
        # an independent chronological out-of-sample performance guarantee.
        day_arrays=[s["days"][d] for d in base_days]
        ref_array=[baseline_by_day[d] for d in base_days]
        required_trials=[]
        dd_trials=[]
        roi_diff_trials=[]
        for seq in index_draws:
            cum=0
            maximum=0
            peak=0
            maxdd=0
            payout_delta=0
            investment=0
            for i in seq:
                st,paid,_=day_arrays[i]
                ref_st,ref_paid,_=ref_array[i]
                if st!=ref_st:
                    raise AssertionError(("funding per-day mismatch",policy,base_days[i]))
                maximum=max(maximum,st-cum)
                cum+=paid-st
                peak=max(peak,cum)
                maxdd=max(maxdd,peak-cum)
                payout_delta+=paid-ref_paid
                investment+=st
            required_trials.append(maximum)
            dd_trials.append(maxdd)
            roi_diff_trials.append(100*payout_delta/investment)
        sensitivity.append({
            "policy":policy, "samples":STRESS_SAMPLES,
            "bootstrap_required_cash_p50_yen":percentile(required_trials,.5),
            "bootstrap_required_cash_p95_yen":percentile(required_trials,.95),
            "bootstrap_required_cash_p99_yen":percentile(required_trials,.99),
            "bootstrap_maxdd_p95_yen":percentile(dd_trials,.95),
            "bootstrap_roi_pp_difference_vs_flat200_p025":round(percentile(roi_diff_trials,.025),6),
            "bootstrap_roi_pp_difference_vs_flat200_p975":round(percentile(roi_diff_trials,.975),6),
            **{f"resampled_budget_survival_rate_at_{cash}_pct":
               round(100*sum(r<=cash for r in required_trials)/STRESS_SAMPLES,2)
               for cash in STARTING_CASH},
        })
    write_table(out,"bankroll_feasibility_all_policies.csv",feasibility)
    write_table(out,"global_capital_and_risk.csv",risk_rows)
    write_table(out,"annual_capital_and_risk.csv",year_rows)
    write_table(out,"hypothetical_payout_haircut_stress.csv",haircut_rows)
    write_table(out,"day_bootstrap_capital_and_roi_sensitivity.csv",sensitivity)
    summary={
        "contract":"KEIBA_L3_WHOLE_BANKROLL_V3",
        "core_objective":"Maximize ROI on all frozen 13,479 tickets, while assessing actual fixed-bet funding and tail risk.",
        "ticket_set_l2_1_fixed":True,
        "all_bets_minimum_100_yen_no_skips_no_rebets":True,
        "bankroll_amount_does_not_modify_stakes":True,
        "insufficient_budget_does_not_permit_partial_purchase":True,
        "funding_convention":"All stakes for a date reserved before any of that date's winnings credited.",
        "initial_cash_definition":"Exact minimum opening cash sufficient to follow a given frozen stake schedule across historical days.",
        "not_live_payout_settlement_order":True,
        "total_tickets":13479, "total_races":6877, "historical_days":322,
        "2026_sealed":True,
        "l2_1_was_selected_on_2023_2025_outcomes":True,
        "historical_odds_are_FINAL_not_cutoff":True,
        "no_tradable_OOS_profitability_claim":True,
        "bootstrap_draws":STRESS_SAMPLES,
        "bootstrap_is_a_nonindependent_historical_sensitivity_analysis":True,
        "cash_scenarios_yen":list(STARTING_CASH),
        "uniform_winning_payout_haircuts_are_hypothetical_not_real_market_repricing":True,
        "policies_count":len(states),
        "recommended_production_policy":"NONE: requires genuinely independent timed paper betting/pricing audit.",
        "illustrative_primary_policies":{
            p:next(v for v in risk_rows if v["policy"]==p)
            for p in COMPARATORS
        },
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L3_WHOLE_BANKROLL_V3_DONE",json.dumps({
        "policies":len(states),
        "results":summary["illustrative_primary_policies"],
        "stresses":STRESS_SAMPLES,
        "no_ticket_changes":True
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
