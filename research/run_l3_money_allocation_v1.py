#!/usr/bin/env python3
"""L3 V1: fixed-bets, equal-capital 100-yen-unit allocation research.

Research only. The L2.1 ticket set was chosen using 2023-25 outcomes; odds
in the frozen ledger are post-race FINAL. No OOS/live profitability claim.
Never use payout/hit in stake decisions. No bet may receive zero stake.
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

ROOT = Path(__file__).resolve().parents[1]
TRUST = ROOT / "research-results/l3-l175-frozen-ticket-trust-audit-v1/run-38710197503"
RACEWISE = ROOT / "research-results/l2-ai-market-rank-betfrontier-v1/run-38681301385/racewise-anchor-comparison.csv.gz"
BASELINE = {
    2023: (4363, 143, 474650, 2377),
    2024: (4584, 120, 533180, 2233),
    2025: (4532, 113, 463830, 2267),
}
BET_TYPES = {"QUINELLA", "TRIO", "TRIFECTA"}
# Predeclared before observing allocation outcomes. All except FLAT100
# spend precisely 200 yen per frozen ticket on average, within each race.
POLICIES = (
    "FLAT100", "FLAT200", "HASH_CONTROL",
    "MIN_P3", "SUPPORT_MIN", "DELTA_MEAN", "P3_SUPPORT",
    "REVERSE_P3", "LOW_FINAL_ODDS", "HIGH_FINAL_ODDS", "P3_HIGH_FINAL_ODDS",
)
FINAL_ODDS_ONLY = {"LOW_FINAL_ODDS", "HIGH_FINAL_ODDS", "P3_HIGH_FINAL_ODDS"}
SIGNALS = {"min_p3", "support_min", "mean_p3_delta", "ticket_odds"}
FIELDS = ("year", "race_id", "bet_type", "selection", "race_date")


def csv_rows(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def key(t):
    return (t["year"], t["race_id"], t["bet_type"], t["selection"])


def decimal(r, field):
    v = r.get(field)
    if not v:
        return None
    z = float(v)
    if not math.isfinite(z):
        raise ValueError(("nonfinite", field, key(r)))
    return z


def load():
    dates = {}
    for r in csv_rows(RACEWISE):
        year = int(r["year"])
        if year not in BASELINE:
            continue
        idx = (year, str(r["race_id"]))
        date = str(r["race_date"])[:10]
        if len(date) != 10 or not date.startswith(str(year) + "-"):
            raise ValueError(("bad race date", idx, date))
        if idx in dates and dates[idx] != date:
            raise ValueError(("conflicting race dates", idx))
        dates[idx] = date

    tickets = []
    for known, filename in ((True, "joined_ticket_trust_complete.csv.gz"),
                            (False, "tickets_missing_l175_signal.csv.gz")):
        for row in csv_rows(TRUST / filename):
            y = int(row["year"])
            rid = str(row["race_id"])
            idx = (y, rid)
            if y not in BASELINE or idx not in dates:
                raise ValueError(("race mismatch", idx))
            bet = str(row["bet_type"])
            if bet not in BET_TYPES:
                raise ValueError(("unexpected bet", bet))
            payout = int(row["payout_yen"])
            hit = int(row["hit"])
            if payout < 0 or hit != int(payout > 0):
                raise ValueError(("invalid settlement", idx, payout, hit))
            features = {f: (decimal(row, f) if known else None) for f in SIGNALS}
            t = {
                "year": y, "race_id": rid, "bet_type": bet,
                "selection": str(row["selection"]), "race_date": dates[idx],
                "hit": hit, "payout_yen_per100": payout,
                "l175_complete": int(known), **features,
            }
            if known and (t["min_p3"] is None or not 0 <= t["min_p3"] <= 1):
                raise ValueError(("invalid min_p3", key(t)))
            tickets.append(t)
    seen = set()
    by_race = defaultdict(list)
    for t in tickets:
        k = key(t)
        if k in seen:
            raise ValueError(("duplicate frozen physical ticket", k))
        seen.add(k)
        by_race[(t["year"], t["race_id"])].append(t)
    if len(tickets) != 13479 or sum(t["l175_complete"] == 0 for t in tickets) != 215:
        raise ValueError(("frozen count/coverage drift", len(tickets)))
    for year, (n, wins, ret, races) in BASELINE.items():
        a = [t for t in tickets if t["year"] == year]
        actual = (len(a), sum(t["hit"] for t in a),
                  sum(t["payout_yen_per100"] for t in a),
                  sum(y == year for y, _ in by_race))
        if actual != (n, wins, ret, races):
            raise ValueError(("L2.1 baseline drift", year, actual))
    return by_race


def percentile(values, field):
    observed = sorted({t[field] for t in values if t[field] is not None})
    if not observed:
        return {key(t): None for t in values}
    result = {}
    for t in values:
        v = t[field]
        result[key(t)] = None if v is None else (
            .5 if len(observed) == 1 else observed.index(v) / (len(observed) - 1))
    return result


def scores(policy, tickets):
    # No payout, hit, finish, rank learned from target results.
    if policy == "HASH_CONTROL":
        return {key(t): int(hashlib.sha256("|".join(map(str, key(t))).encode()).hexdigest()[:12], 16)
                for t in tickets}
    ranks = {f: percentile(tickets, f) for f in SIGNALS}
    result = {}
    for t in tickets:
        k = key(t)
        def rank(name):
            return ranks[name][k]
        def mix(*names):
            vals = [rank(name) for name in names]
            return None if any(v is None for v in vals) else sum(vals) / len(vals)
        if policy == "MIN_P3":
            val = rank("min_p3")
        elif policy == "REVERSE_P3":
            x = rank("min_p3")
            val = None if x is None else -x
        elif policy == "SUPPORT_MIN":
            val = rank("support_min")
        elif policy == "DELTA_MEAN":
            val = rank("mean_p3_delta")
        elif policy == "P3_SUPPORT":
            val = mix("min_p3", "support_min")
        elif policy == "HIGH_FINAL_ODDS":
            val = rank("ticket_odds")
        elif policy == "LOW_FINAL_ODDS":
            x = rank("ticket_odds")
            val = None if x is None else -x
        elif policy == "P3_HIGH_FINAL_ODDS":
            val = mix("min_p3", "ticket_odds")
        else:
            raise ValueError(policy)
        result[k] = val
    return result


def stakes(policy, tickets, race_budget_yen):
    n = len(tickets)
    if not n or race_budget_yen % 100:
        raise ValueError(("invalid race budget", race_budget_yen, n))
    minimum = 100 * n
    if race_budget_yen < minimum:
        raise ValueError("INSUFFICIENT_BUDGET")
    if race_budget_yen > 300 * n:
        raise ValueError("RACE_BUDGET_EXCEEDS_MAX_STAKE")
    if policy == "FLAT100":
        if race_budget_yen != minimum:
            raise ValueError("FLAT100_REQUIRES_MINIMUM_BUDGET")
        return {key(t): 100 for t in tickets}
    if policy == "FLAT200":
        if race_budget_yen != 200 * n:
            raise ValueError("FLAT200_REQUIRES_DOUBLE_BUDGET")
        return {key(t): 200 for t in tickets}
    weights = scores(policy, tickets)
    # Missing signals are not silently fabricated. They remain as 100-yen
    # purchases and sort below available evidence for the incremental pool.
    order = sorted(tickets, key=lambda t: (
        weights[key(t)] is None,
        -(weights[key(t)] if weights[key(t)] is not None else 0),
        key(t)))
    alloc = {key(t): 100 for t in tickets}
    units = (race_budget_yen - minimum) // 100
    for t in order:
        extra = min(2, units)
        alloc[key(t)] += 100 * extra
        units -= extra
        if units == 0:
            break
    if units or set(alloc) != {key(t) for t in tickets}:
        raise AssertionError("budget or frozen-key mismatch")
    if sum(alloc.values()) != race_budget_yen or not all(
            100 <= v <= 300 and v % 100 == 0 for v in alloc.values()):
        raise AssertionError("invalid stake vector")
    return alloc


def max_dd(daily):
    peak = total = maximum = 0
    for date, value in sorted(daily.items()):
        total += value
        peak = max(peak, total)
        maximum = max(maximum, peak - total)
    return maximum


def describe(rows):
    stake = sum(r["stake_yen"] for r in rows)
    returns = sum(r["return_yen"] for r in rows)
    wins = [r["return_yen"] for r in rows if r["return_yen"] > 0]
    daily = defaultdict(int)
    races = set()
    for r in rows:
        daily[r["race_date"]] += r["return_yen"] - r["stake_yen"]
        races.add((r["year"], r["race_id"]))
    minus3 = returns - sum(sorted(wins, reverse=True)[:3])
    return {
        "races": len(races), "tickets": len(rows), "hits": len(wins),
        "stake_yen": stake, "payout_yen": returns, "profit_yen": returns - stake,
        "roi_pct": round(100 * returns / stake, 6),
        "roi_ex_top3_win_pct": round(100 * minus3 / stake, 6),
        "max_daily_close_drawdown_yen": max_dd(daily),
        "worst_day_profit_yen": min(daily.values()),
        "daily_profit_stdev_yen": round(statistics.pstdev(daily.values()), 2),
    }


def percentile_ci(samples, alpha):
    s = sorted(samples)
    return (s[int((alpha/2) * (len(s)-1))],
            s[int((1-alpha/2) * (len(s)-1))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frozen = load()
    grouped = defaultdict(list)
    byday = defaultdict(dict)
    n_unknown = 0
    with gzip.open(out / "ticket_stakes.csv.gz", "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "policy", "year", "race_date", "race_id", "bet_type", "selection",
            "l175_complete", "stake_yen", "return_yen"])
        writer.writeheader()
        for (year, rid), tickets in sorted(frozen.items()):
            n = len(tickets)
            n_unknown += sum(t["l175_complete"] == 0 for t in tickets)
            for policy in POLICIES:
                budget = 100 * n if policy == "FLAT100" else 200 * n
                allocation = stakes(policy, tickets, budget)
                if set(allocation) != {key(t) for t in tickets}:
                    raise AssertionError("ticket-set drift")
                for t in tickets:
                    st = allocation[key(t)]
                    paid = t["payout_yen_per100"] * st // 100
                    row = {field: t[field] for field in FIELDS}
                    row.update(policy=policy, l175_complete=t["l175_complete"],
                               stake_yen=st, return_yen=paid)
                    writer.writerow(row)
                    grouped[policy, year].append(row)
                    byday[policy, t["race_date"]]["profit_yen"] = (
                        byday[policy, t["race_date"]].get("profit_yen", 0) + paid - st)
    if n_unknown != 215:
        raise AssertionError("missing ticket drift")
    comparison = []
    summary = {
        "contract": "KEIBA_L3_MONEY_ALLOCATION_V1",
        "source_l2_frozen_commit": "9655c717333636b5cc931f672e12dbc377e396bc",
        "l2_was_selected_on_test_period": True,
        "period": "2023-2025",
        "2026_sealed": True,
        "historical_odds_are_post_race_FINAL": True,
        "price_based_policies_descriptive_only": sorted(FINAL_ODDS_ONLY),
        "p3_is_not_exact_ticket_hit_probability": True,
        "no_ticket_selection_or_zero_stakes": True,
        "unknown_signal_ticket_count": n_unknown,
        "all_policies_spend_same_total_except_FLAT100": True,
        "allocation": "Every race receives exactly 200 yen per ticket on average, min 100/max 300, original L2 tickets unchanged.",
        "yearly": {},
    }
    for policy in POLICIES:
        for year in sorted(BASELINE):
            row = {"policy": policy, "year": year, **describe(grouped[policy, year])}
            comparison.append(row)
        combined = [r for year in BASELINE for r in grouped[policy, year]]
        row = {"policy": policy, "year": "2023-2025", **describe(combined)}
        comparison.append(row)
        summary["yearly"][policy] = {
            str(y): next(r for r in comparison if r["policy"] == policy and r["year"] == y)
            for y in BASELINE
        }
    flat100 = next(r for r in comparison if r["policy"] == "FLAT100" and r["year"] == "2023-2025")
    flat200 = next(r for r in comparison if r["policy"] == "FLAT200" and r["year"] == "2023-2025")
    if (flat100["stake_yen"], flat100["payout_yen"], flat100["hits"],
        flat100["tickets"], flat100["races"]) != (1347900, 1471660, 376, 13479, 6877):
        raise ValueError(("FLAT100 does not reproduce frozen baseline", flat100))
    if flat200["profit_yen"] != 247520:
        raise ValueError("FLAT200 arithmetic failure")
    for row in comparison:
        if row["policy"] != "FLAT100" and row["stake_yen"] != (
                flat200["stake_yen"] if row["year"] == "2023-2025"
                else next(x["stake_yen"] for x in comparison
                          if x["policy"] == "FLAT200" and x["year"] == row["year"])):
            raise ValueError(("comparison capital mismatch", row["policy"], row["year"]))
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    # Paired block bootstrap of *days*, not tickets, for descriptive
    # sensitivity only. NOT valid independent out-of-sample evidence.
    dates = sorted({date for policy, date in byday if policy == "FLAT200"})
    rng = random.Random(20261015)
    index_sets = [[rng.randrange(len(dates)) for _ in dates] for _ in range(1000)]
    for policy in POLICIES:
        if policy in ("FLAT100", "FLAT200"):
            continue
        daily_delta = [
            byday[policy, d].get("profit_yen", 0) -
            byday["FLAT200", d].get("profit_yen", 0) for d in dates]
        ci = percentile_ci([sum(daily_delta[i] for i in draw) for draw in index_sets], .05)
        summary.setdefault("paired_day_resample_vs_flat200", {})[policy] = {
            "observed_extra_profit_yen": sum(daily_delta),
            "resample_95pct_profit_diff_interval_yen": list(ci),
            "resamples": len(index_sets), "causal_or_oos_claim": False,
        }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("L3_MONEY_ALLOCATION_V1_DONE", json.dumps({
        "flat100_profit": flat100["profit_yen"], "flat200_profit": flat200["profit_yen"],
        "all_13k_included": True,
        "unknown_kept": n_unknown,
        "comparison": {p: next(x for x in comparison if x["policy"] == p
                              and x["year"] == "2023-2025")["roi_pct"] for p in POLICIES},
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
