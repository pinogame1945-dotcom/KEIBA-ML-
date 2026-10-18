#!/usr/bin/env python3
"""EV rank 1..6 axes x pure-AI counterpart rank x exact payouts (2023-25).

Uses only the frozen historical single-win EV ranking and pure distill_ai_rank.
NO exotic-bet price is consulted for ticket construction. 2026 is sealed.
Winning-payout prefix aggregation avoids materializing up to 816 combinations
per rank/template/race: O(paid tickets + ranks * families) per axis.
"""
import argparse
import concurrent.futures
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import run_l2_value_anchor_bet_frontier_v1 as old
from run_l2_ai_market_rank_betfrontier_v1 import load_predictions
from run_all_race_flow_ticket_audit_v1 import YEARS, RACES_YEAR, TOTAL_RACES, assert_ok
from build_l2_market_helpers_v1 import load_day, horse_number_map
from run_value_anchor_official_positions_v1 import positions

MAX_EV_RANK = 6
MAX_PARTNERS = 17
COMBO_FAMILIES = old.FAMILIES[2:]
ALL_FAMILIES = old.FAMILIES
BASELINE = dict(tickets=10368, return_yen=724820, hit_races=684)
MARKET_BUCKETS = ("1", "2", "3", "4-6", "7-10", "11+")


def bucket(market_rank):
    n = int(market_rank)
    assert_ok(n >= 1 and n == market_rank, "invalid market rank")
    return "1" if n == 1 else "2" if n == 2 else "3" if n == 3 else (
        "4-6" if n <= 6 else "7-10" if n <= 10 else "11+")


def tickets_count(family, k):
    if family in ("QUINELLA", "WIDE", "EXACTA_FIRST", "EXACTA_SECOND"):
        return k
    if family == "EXACTA_BOTH":
        return 2 * k
    if family == "TRIO":
        return k * (k - 1) // 2
    if family in ("TRIFECTA_FIRST", "TRIFECTA_SECOND", "TRIFECTA_THIRD"):
        return k * (k - 1)
    if family == "TRIFECTA_ANY":
        return 3 * k * (k - 1)
    raise ValueError("unexpected family " + family)


def payout_prefix(axis, partners, payout_map):
    """Only inspect payouts of tickets containing axis, indexed by rank needed."""
    rank = {n: i + 1 for i, n in enumerate(partners)}
    assert_ok(axis not in rank and len(rank) == len(partners), "partner duplication")
    deltas = {f: [0.] * (MAX_PARTNERS + 1) for f in ALL_FAMILIES}
    hits = {f: [0] * (MAX_PARTNERS + 1) for f in ALL_FAMILIES}
    for (bet, combo), amount in payout_map.items():
        if axis not in combo:
            continue
        if bet in ("WIN", "PLACE"):
            if combo == (axis,):
                deltas[bet][0] += amount
                hits[bet][0] += 1
            continue
        others = [n for n in combo if n != axis]
        assert_ok(len(others) == len(combo) - 1, "bad payout combination")
        if any(n not in rank for n in others):
            continue
        k = max(rank[n] for n in others)
        if bet in ("QUINELLA", "WIDE"):
            families = (bet,)
        elif bet == "EXACTA":
            families = ("EXACTA_FIRST", "EXACTA_BOTH") if combo[0] == axis else (
                "EXACTA_SECOND", "EXACTA_BOTH")
        elif bet == "TRIO":
            families = ("TRIO",)
        elif bet == "TRIFECTA":
            pos = combo.index(axis)
            families = (("TRIFECTA_FIRST", "TRIFECTA_SECOND", "TRIFECTA_THIRD")[pos],
                        "TRIFECTA_ANY")
        else:
            raise ValueError("unknown bet type " + bet)
        assert_ok(1 <= k <= MAX_PARTNERS, "rank out of bounds")
        for family in families:
            deltas[family][k] += amount
            hits[family][k] += 1
    for family in COMBO_FAMILIES:
        for k in range(1, MAX_PARTNERS + 1):
            deltas[family][k] += deltas[family][k - 1]
            hits[family][k] += hits[family][k - 1]
    return deltas, hits


def test_prefix():
    """Exhaustive old-generator comparison, including disjoint paid combinations."""
    import itertools
    for n in (3, 5, 10, 17):
        axis = 18
        partners = list(range(1, n + 1))
        cards = old.pair_templates(axis, partners)
        all_tickets = {(f, t) for f, ts in cards.items() for t in ts}
        # Multiple simultaneous payouts test dead-heat-like cases.
        paid = {(bet, combo): float((i % 29 + 1) * 100)
                for i, (_, (bet, combo)) in enumerate(sorted(all_tickets))
                if i % 7 == 0}
        paid[("WIN", (axis,))] = 210.
        paid[("PLACE", (axis,))] = 130.
        ret, hit = payout_prefix(axis, partners, paid)
        for k in range(1, MAX_PARTNERS + 1):
            sub = old.pair_templates(axis, partners[:k])
            for f in COMBO_FAMILIES:
                real = sum(paid.get(t, 0) for t in sub[f])
                real_hits = sum(t in paid for t in sub[f])
                assert_ok(tickets_count(f, min(k, n)) == len(sub[f]),
                          f"tickets parity {n} {k} {f}")
                assert_ok(ret[f][k] == real and hit[f][k] == real_hits,
                          f"payout parity {n} {k} {f}: {ret[f][k]} != {real}")
        assert_ok(ret["WIN"][0] == 210. and ret["PLACE"][0] == 130.,
                  "single-ticket payout parity")
    print("PREFIX_PARITY_SELF_TEST_OK", flush=True)


def collect(ct, tickets, amount, hit_tickets, finish):
    ct["races"] += 1
    ct["tickets"] += tickets
    ct["stake_yen"] += 100 * tickets
    ct["return_yen"] += amount
    ct["hit_races"] += int(amount > 0)
    ct["hit_tickets"] += hit_tickets
    ct["bet_races"] += int(tickets > 0)
    ct["stake_" + finish] += 100 * tickets
    ct["return_" + finish] += amount


def run_year(year, all_data, picks, backfill_root):
    df = all_data[all_data.year.eq(year)].copy()
    groups = df.groupby("_race_id", sort=False)
    dates = df[["_race_id", "_race_date"]].drop_duplicates()
    assert_ok(not dates.duplicated("_race_id").any(), "conflicting race dates")
    days = dates.groupby("_race_date")["_race_id"].apply(list)
    stats = defaultdict(Counter)
    axes = defaultdict(Counter)
    race_count = 0

    for day, rids in days.items():
        packs = load_day(backfill_root / "data" / "daily" / (day + ".jsonl.gz"),
                         set(rids))
        for rid in rids:
            pack = packs.get(rid)
            assert_ok(pack is not None, f"missing race {rid}")
            group = groups.get_group(rid)
            horses = horse_number_map(pack)
            finishes = positions(pack)
            payout_map = old.payouts(pack, rid)
            vals = group.assign(__ev=group["P__ANCHOR_FULL"] * group["final_win_odds"])
            ev = vals.sort_values(["__ev", "P__ANCHOR_FULL", "market_rank", "_horse_id"],
                                  ascending=[False, False, True, True],
                                  kind="mergesort")["_horse_id"].astype(str).tolist()
            raw = group.sort_values(["distill_ai_rank", "market_rank", "_horse_id"],
                                    ascending=[True, True, True],
                                    kind="mergesort")["_horse_id"].astype(str).tolist()
            assert_ok(set(ev) == set(raw) and len(set(ev)) == len(ev), "rank mismatch")
            assert_ok(ev[0] == picks["ANCHOR_FULL"][rid],
                      f"EV rank1 frozen pick mismatch {rid}")
            assert_ok(set(ev).issubset(horses), f"horse number missing {rid}")
            indexed = group.set_index("_horse_id")
            for r, axis in enumerate(ev[:MAX_EV_RANK], 1):
                finish = finishes.get(axis)
                actual_win = indexed.loc[axis, "target_win"]
                actual_top3 = indexed.loc[axis, "target_top3"]
                assert_ok(int(finish == 1) == int(actual_win), "finish/win target drift")
                assert_ok(int(finish is not None and finish <= 3) == int(actual_top3),
                          "finish/top3 target drift")
                finish_bucket = "NO_FINISH" if finish is None else (
                    "FIRST" if finish == 1 else "SECOND" if finish == 2 else
                    "THIRD" if finish == 3 else "FOURTH_PLUS")
                market_bucket = bucket(float(indexed.loc[axis, "market_rank"]))
                axes[(r, "ALL")]["races"] += 1
                axes[(r, market_bucket)]["races"] += 1
                axes[(r, "ALL")]["top3"] += int(actual_top3)
                axes[(r, market_bucket)]["top3"] += int(actual_top3)
                axes[(r, "ALL")]["wins"] += int(actual_win)
                axes[(r, market_bucket)]["wins"] += int(actual_win)
                axis_num = horses[axis]
                partners = [horses[h] for h in raw if h != axis]
                ret, hit = payout_prefix(axis_num, partners, payout_map)
                for label in ("ALL", market_bucket):
                    for family in ("WIN", "PLACE"):
                        collect(stats[(r, label, family, 0)], 1,
                                ret[family][0], hit[family][0], finish_bucket)
                for k in range(1, MAX_PARTNERS + 1):
                    effective_k = min(k, len(partners))
                    for family in COMBO_FAMILIES:
                        ticket_n = tickets_count(family, effective_k)
                        for label in ("ALL", market_bucket):
                            c = stats[(r, label, family, k)]
                            collect(c, ticket_n, ret[family][k], hit[family][k],
                                    finish_bucket)
                            c["saturated_races"] += int(k > len(partners))
            race_count += 1
    assert_ok(race_count == RACES_YEAR, f"{year}: {race_count} race count wrong")
    return stats, axes


def aggregate(chunks):
    counters = defaultdict(Counter)
    axes = defaultdict(Counter)
    for year, (rows, axis_rows) in zip(YEARS, chunks):
        for key, c in rows.items():
            counters[(str(year), *key)].update(c)
            counters[("2023-2025", *key)].update(c)
        for key, c in axis_rows.items():
            axes[(str(year), *key)].update(c)
            axes[("2023-2025", *key)].update(c)
    metrics = []
    for (year, r, b, family, k), c in counters.items():
        stake = c["stake_yen"]
        gross = c["return_yen"]
        n = c["races"]
        metrics.append({
            "year": year, "ev_axis_rank": r, "market_rank_bucket": b,
            "family": family, "partner_order": "AI_RAW", "other_horses_k": k,
            "races": n, "bet_races": c["bet_races"],
            "tickets": c["tickets"], "avg_tickets_per_race": round(c["tickets"] / n, 5),
            "stake_yen": stake, "return_yen": round(gross, 2),
            "profit_yen": round(gross - stake, 2),
            "roi_pct": round(100 * gross / stake, 6) if stake else None,
            "hit_races": c["hit_races"],
            "hit_rate_pct": round(100 * c["hit_races"] / n, 6),
            "hit_tickets": c["hit_tickets"],
            "saturated_races": c["saturated_races"],
        })
    rank_rows = []
    for (year, r, b), c in axes.items():
        rank_rows.append({"year": year, "ev_axis_rank": r, "market_rank_bucket": b,
                          "races": c["races"], "wins": c["wins"], "top3": c["top3"],
                          "win_pct": round(100 * c["wins"] / c["races"], 6),
                          "top3_pct": round(100 * c["top3"] / c["races"], 6)})
    frame = pd.DataFrame(metrics).sort_values(
        ["year", "ev_axis_rank", "market_rank_bucket", "family", "other_horses_k"])
    ranking = pd.DataFrame(rank_rows).sort_values(
        ["year", "ev_axis_rank", "market_rank_bucket"])
    return frame, ranking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions")
    ap.add_argument("--picks")
    ap.add_argument("--backfill-root")
    ap.add_argument("--out-dir")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        test_prefix()
        return
    assert_ok(all((args.predictions, args.picks, args.backfill_root, args.out_dir)),
              "missing mandatory file arguments")
    test_prefix()
    data, picks = load_predictions(args.predictions, args.picks)
    root = Path(args.backfill_root)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        chunks = list(pool.map(lambda y: run_year(y, data, picks, root), YEARS))
    results, ranks = aggregate(chunks)
    base = results[(results.year == "2023-2025") &
                   (results.ev_axis_rank == 1) &
                   (results.market_rank_bucket == "ALL") &
                   (results.family == "TRIO") &
                   (results.other_horses_k == 2)].iloc[0]
    assert_ok(int(base.tickets) == BASELINE["tickets"] and
              abs(base.return_yen - BASELINE["return_yen"]) < .01 and
              int(base.hit_races) == BASELINE["hit_races"],
              "frozen 2023-2025 EV1/RAW2 TRIO baseline parity failed")
    for r in range(1, MAX_EV_RANK + 1):
        expected = int((ranks[(ranks.year == "2023-2025") &
                               (ranks.ev_axis_rank == r) &
                               (ranks.market_rank_bucket == "ALL")].iloc[0]).races)
        assert_ok(expected > 0 and expected <= TOTAL_RACES, "missing EV rank")
        actual = results[(results.year == "2023-2025") &
                         (results.ev_axis_rank == r) &
                         (results.market_rank_bucket == "ALL")]
        assert_ok(len(actual) == 2 + 10 * MAX_PARTNERS, "missing template output")
        assert_ok(actual.races.eq(expected).all(), "races mismatch")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "all-ev-top6-raw-ai-templates.csv", index=False)
    ranks.to_csv(out / "ev-axis-rank-hit-and-market-buckets.csv", index=False)
    pooled = results[(results.year == "2023-2025") &
                     (results.market_rank_bucket == "ALL")]
    profitable = pooled[pooled.profit_yen.gt(0)]
    report = {
        "contract": "L2_EV_TOP6_RAW_AI_NO_TICKET_ODDS_V1",
        "source": "frozen ANCHOR_FULL predicted win probability x final single-win odds",
        "value_rank_tiebreak": ["EV descending", "p descending", "market_rank ascending",
                                "horse_id ascending"],
        "partner_ranking": "distill_ai_rank (pure AI) ascending, excluding selected axis",
        "years": list(YEARS), "races": TOTAL_RACES,
        "axis_ranks": list(range(1, MAX_EV_RANK + 1)),
        "partner_k": list(range(1, MAX_PARTNERS + 1)),
        "families": list(ALL_FAMILIES), "stake_yen_per_ticket": 100,
        "rule": "All races, fixed EV axis ranking, no ticket odds threshold, no ex-post skip",
        "market_rank_bucket_only": list(MARKET_BUCKETS),
        "historical_odds_warning": "EV selection uses retrospective final SINGLE WIN odds, not bet-cutoff win odds. Payouts for settlement only.",
        "model_refit": False, "new_selection_training": False,
        "zero_combination_tickets_at_k1": ["TRIO", "TRIFECTA_FIRST",
                                           "TRIFECTA_SECOND", "TRIFECTA_THIRD",
                                           "TRIFECTA_ANY"],
        "saturated_k_note": "When field is smaller than k+1, the partner set saturates and same tickets recur across larger k templates.",
        "historical_source_backfill_sha": old.SHA,
        "price_gate": "NONE (no exotic ticket price in selection)",
        "payout_settlement": "actual historical BACKFILL dividends, dead-heat aware",
        "ev1_trio_raw2_reproduced": dict(BASELINE),
        "templates_per_rank_all_bucket": 2 + 10 * MAX_PARTNERS,
        "profitable_templates_count_in_sample": int(len(profitable)),
        "profitable_templates_count_by_ev_rank_in_sample": {
            str(i): int((profitable.ev_axis_rank == i).sum())
            for i in range(1, MAX_EV_RANK + 1)},
        "highest_roi_observed_NOT_FORWARD_VALIDATED": pooled[
            pooled.stake_yen.gt(0)].sort_values(
                ["roi_pct", "stake_yen"], ascending=[False, False]).head(20)[
                ["ev_axis_rank", "family", "other_horses_k", "races",
                 "tickets", "hit_races", "profit_yen", "roi_pct"]].to_dict("records"),
        "yearly_not_oos_warning": "Selection of best strategies from these same years would be in-sample; this file does not declare a deployable strategy.",
        "sealed_years": [2026], "cost": "free GitHub Actions standard CPU only",
    }
    (out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("EV_TOP6_RAW_AI_ALL_BETS_DONE")
    print("PER_EV_RANK")
    print(ranks[(ranks.year == "2023-2025") &
                (ranks.market_rank_bucket == "ALL")].to_csv(index=False))
    print("TOP20_IN_SAMPLE_NOT_DEPLOYABLE")
    print(pooled[pooled.stake_yen.gt(0)].sort_values(
        ["roi_pct", "stake_yen"], ascending=[False, False]).head(20).to_csv(index=False))


if __name__ == "__main__":
    main()
