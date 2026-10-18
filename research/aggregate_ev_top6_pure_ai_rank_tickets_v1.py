#!/usr/bin/env python3
"""Exact single-ticket ROI for every EV 1..6 axis x PURE AI numbered partner.

NO feature/model recalculation, no ticket-odds gates. Reads frozen OOS prediction
ranks and pinned BACKFILL settled payouts. Enumerates *named raw AI ranks*,
not just previous 'top-K' template families. Results never used for selection.
Historical 2023-25 only; 2026 sealed.

Fast approach: count race exposure by (year, EV-rank, field-size, axis-AI-position).
Every paid ticket is mapped to its actual numbered AI partner position(s) once.
Generate all possible template exposures AFTER the scan using histograms. Thus
we do not generate millions of losing combinations per race.
"""
import argparse
import concurrent.futures
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from run_l2_ai_market_rank_betfrontier_v1 import load_predictions
from run_all_race_flow_ticket_audit_v1 import YEARS, RACES_YEAR, TOTAL_RACES, assert_ok
from build_l2_market_helpers_v1 import load_day, horse_number_map
from run_value_anchor_official_positions_v1 import positions
import run_l2_value_anchor_bet_frontier_v1 as old

FAMILIES = ("WIN", "PLACE", "QUINELLA", "WIDE", "EXACTA_FIRST",
            "EXACTA_SECOND", "TRIO", "TRIFECTA_FIRST", "TRIFECTA_SECOND",
            "TRIFECTA_THIRD")
DOUBLE = ("QUINELLA", "WIDE", "EXACTA_FIRST", "EXACTA_SECOND")
TRIPLE = ("TRIO", "TRIFECTA_FIRST", "TRIFECTA_SECOND", "TRIFECTA_THIRD")
MAX_FIELD = 18
MAX_EV_RANK = 6
MISSING = 0


def configs():
    for f in FAMILIES:
        if f in ("WIN", "PLACE"):
            yield f, 0, 0
        elif f in DOUBLE:
            for j in range(1, MAX_FIELD + 1):
                yield f, j, 0
        elif f == "TRIO":
            for j in range(1, MAX_FIELD + 1):
                for k in range(j + 1, MAX_FIELD + 1):
                    yield f, j, k
        else:
            for j in range(1, MAX_FIELD + 1):
                for k in range(1, MAX_FIELD + 1):
                    if j != k:
                        yield f, j, k


def winner_to_template(axis_num, numbers, rank_map, bet, combo):
    if axis_num not in combo:
        return None
    if bet in ("WIN", "PLACE"):
        return (bet, 0, 0) if combo == (axis_num,) else None
    other = [rank_map.get(n) for n in combo if n != axis_num]
    if None in other:
        return None
    if bet in ("QUINELLA", "WIDE"):
        return bet, other[0], 0
    if bet == "EXACTA":
        return ("EXACTA_FIRST" if combo[0] == axis_num else "EXACTA_SECOND",
                other[0], 0)
    if bet == "TRIO":
        return "TRIO", min(other), max(other)
    if bet == "TRIFECTA":
        assert_ok(len(other) == 2, "invalid trifecta paid ticket")
        return ("TRIFECTA_FIRST", "TRIFECTA_SECOND", "TRIFECTA_THIRD")[
            combo.index(axis_num)], other[0], other[1]
    raise ValueError(f"unrecognized bet category {bet}")


def scan_year(year, data, picks, root):
    year_df = data[data.year.eq(year)].copy()
    groups = year_df.groupby("_race_id", sort=False)
    dates = year_df[["_race_id", "_race_date"]].drop_duplicates()
    assert_ok(not dates.duplicated("_race_id").any(), "multiple dates per race")
    days = dates.groupby("_race_date")["_race_id"].apply(list)
    exposure = Counter()
    payouts = defaultdict(float)
    paid_hits = Counter()
    baseline = defaultdict(Counter)
    races = 0
    for date, rids in days.items():
        packs = load_day(root/"data"/"daily"/f"{date}.jsonl.gz", set(rids))
        for rid in rids:
            pack = packs.get(rid)
            assert_ok(pack is not None, f"missing historical pack {rid}")
            g = groups.get_group(rid)
            # identical EV ranking and deterministic tie policy to frozen research
            g = g.assign(_EV=g["P__ANCHOR_FULL"] * g["final_win_odds"])
            ev_order = g.sort_values(
                ["_EV", "P__ANCHOR_FULL", "market_rank", "_horse_id"],
                ascending=[False, False, True, True],
                kind="mergesort")["_horse_id"].astype(str).tolist()
            raw_order = g.sort_values(
                ["distill_ai_rank", "market_rank", "_horse_id"],
                ascending=[True, True, True], kind="mergesort")[
                    "_horse_id"].astype(str).tolist()
            assert_ok(len(ev_order) == len(raw_order) == len(set(raw_order)),
                      f"invalid horse set {rid}")
            assert_ok(set(ev_order) == set(raw_order), "different candidate sets")
            assert_ok(ev_order[0] == picks["ANCHOR_FULL"][rid],
                      f"EV1 does not reproduce frozen pick {rid}")
            no_by_horse = horse_number_map(pack)
            assert_ok(set(raw_order).issubset(no_by_horse), "unmapped horse number")
            pos = positions(pack)
            by_no_rank = {no_by_horse[h]: i for i, h in enumerate(raw_order, 1)}
            assert_ok(len(by_no_rank) == len(raw_order), "number collision")
            rank_by_horse = {h: i for i, h in enumerate(raw_order, 1)}
            paymap = old.payouts(pack, rid)
            for ev_rank, axis_id in enumerate(ev_order[:MAX_EV_RANK], 1):
                official = pos.get(axis_id)
                axis_target = g.loc[g._horse_id.eq(axis_id)].iloc[0]
                assert_ok(int(official == 1) == int(axis_target.target_win),
                          "target winner mismatch")
                assert_ok(int(official is not None and official <= 3) ==
                          int(axis_target.target_top3), "top3 target mismatch")
                axis_num = no_by_horse[axis_id]
                assert_ok(axis_num in by_no_rank, "missing axis raw rank")
                axis_raw = rank_by_horse[axis_id]
                exposure[(year, ev_rank, len(raw_order), axis_raw)] += 1
                for (bet, combo), amount in paymap.items():
                    key = winner_to_template(axis_num, no_by_horse, by_no_rank,
                                             bet, combo)
                    if key is not None:
                        f, j, k = key
                        payouts[(year, ev_rank, f, j, k)] += amount
                        paid_hits[(year, ev_rank, f, j, k)] += 1
                # Historical old top-K must exactly reproduce independent prior result.
                other = [no_by_horse[h] for h in raw_order if h != axis_id][:2]
                if len(other) == 2:
                    old_trio = old.pair_templates(axis_num, other)["TRIO"]
                    value = sum(paymap.get(t, 0) for t in old_trio)
                    b = baseline[(year, ev_rank)]
                    b["races"] += 1
                    b["tickets"] += len(old_trio)
                    b["return_yen"] += value
                    b["hit_races"] += int(value > 0)
            races += 1
    assert_ok(races == RACES_YEAR, f"incomplete year {year} {races}")
    return exposure, payouts, paid_hits, baseline


def eligible(n, axis_position, family, j, k):
    if family in ("WIN", "PLACE"):
        return True
    if j > n or j == axis_position:
        return False
    if family in DOUBLE:
        return True
    return k <= n and k != axis_position and k != j


def make_report(chunks, reference):
    expos = Counter()
    payout = defaultdict(float)
    wins = Counter()
    baselines = defaultdict(Counter)
    for e, p, w, b in chunks:
        expos.update(e)
        for key, amt in p.items():
            payout[key] += amt
        wins.update(w)
        for key, val in b.items():
            baselines[key].update(val)
    assert_ok(sum(expos.values()) <= 6*TOTAL_RACES, "exposure overflow")
    original = {}
    with Path(reference).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["market_rank_bucket"] != "ALL" or row["family"] != "TRIO":
                continue
            if int(row["other_horses_k"]) != 2:
                continue
            original[(row["year"], int(row["ev_axis_rank"]))] = row
    for yr in ("2023","2024","2025"):
        for rank in range(1, 7):
            prev = original[(yr, rank)]
            current = baselines[(int(yr), rank)]
            assert_ok(current["tickets"] == int(prev["tickets"]) and
                      current["hit_races"] == int(prev["hit_races"]) and
                      abs(current["return_yen"] - float(prev["return_yen"])) < .01,
                      f"old top-2 result differs from historical reference y={yr} rank={rank}")

    rank_buckets = defaultdict(list)
    for (year, rank, size, pos), count in expos.items():
        rank_buckets[(year, rank)].append((size, pos, count))
    report = []
    for yr in YEARS:
        for ev in range(1, MAX_EV_RANK + 1):
            field_hist = rank_buckets[(yr, ev)]
            for f, j, k in configs():
                races = sum(ncount for n, rawaxis, ncount in field_hist
                            if eligible(n, rawaxis, f, j, k))
                if races == 0:
                    continue
                payout_amount = payout.get((yr, ev, f, j, k), 0.)
                hit_count = wins.get((yr, ev, f, j, k), 0)
                assert_ok(0 <= hit_count <= races, "hit count exceeds race count")
                assert_ok(hit_count or not payout_amount, "paid amount without hit")
                assert_ok(not hit_count or payout_amount, "hit without payout")
                report.append({
                    "year": str(yr), "ev_axis_rank": ev, "family": f,
                    "raw_ai_rank_1": j, "raw_ai_rank_2": k,
                    "eligible_races": races, "purchased_tickets": races,
                    "hit_races": hit_count, "hit_tickets": hit_count,
                    "hit_rate_pct": round(100 * hit_count / races, 6),
                    "stake_yen": 100 * races,
                    "return_yen": round(payout_amount, 2),
                    "profit_yen": round(payout_amount - 100 * races, 2),
                    "roi_pct": round(payout_amount / races, 6),
                })
    # add 3yr totals from exact yearly entries - no re-evaluation.
    grouped = defaultdict(list)
    for r in report:
        key = (r["ev_axis_rank"], r["family"], r["raw_ai_rank_1"],
               r["raw_ai_rank_2"])
        grouped[key].append(r)
    for (rank, f, j, k), rows in grouped.items():
        nr = sum(x["purchased_tickets"] for x in rows)
        nh = sum(x["hit_tickets"] for x in rows)
        ret = sum(x["return_yen"] for x in rows)
        assert_ok(nr > 0 and len(rows) <= 3, "bad aggregate grouping")
        report.append({
            "year": "2023-2025", "ev_axis_rank": rank, "family": f,
            "raw_ai_rank_1": j, "raw_ai_rank_2": k,
            "eligible_races": nr, "purchased_tickets": nr,
            "hit_races": nh, "hit_tickets": nh,
            "hit_rate_pct": round(100 * nh / nr, 6),
            "stake_yen": nr * 100, "return_yen": round(ret, 2),
            "profit_yen": round(ret - nr * 100, 2),
            "roi_pct": round(ret / nr, 6)
        })
    pooled = [r for r in report if r["year"] == "2023-2025"]
    # Return on separate fixed ranked tickets, not a combined betting portfolio.
    top_roi = sorted(pooled, key=lambda x:(-x["roi_pct"], -x["eligible_races"]))
    top_profit = sorted(pooled, key=lambda x:(-x["profit_yen"], -x["eligible_races"]))
    sufficient = [r for r in pooled if r["eligible_races"] >= 1000]
    stable = []
    row_index = {(r["year"], r["ev_axis_rank"], r["family"],
                  r["raw_ai_rank_1"], r["raw_ai_rank_2"]): r for r in report}
    for p in pooled:
        keys = [(str(yr), p["ev_axis_rank"], p["family"],
                 p["raw_ai_rank_1"], p["raw_ai_rank_2"]) for yr in YEARS]
        if all(key in row_index and row_index[key]["profit_yen"] > 0 for key in keys):
            stable.append(p)
    summary = {
        "contract": "L2_EV_TOP6_PURE_AI_NAMED_RANK_ATOMIC_TICKETS_V1",
        "year_range": "2023-2025", "races": TOTAL_RACES,
        "ranking_1": "EV: frozen P__ANCHOR_FULL*historical final single-win odds",
        "ranking_2": "raw distill_ai_rank, deterministic ties",
        "condition": "Individual raw AI numeric rank combinations, not top-K reindexed",
        "all_selections": "6 EV ranks x 7 physical bet kinds and ordered trifecta positions",
        "tickets_are": "Single 100-yen specific race bet combination; one ticket per eligible race",
        "selection_does_not_use_exotic_odds": True,
        "2026_sealed": True,
        "historical_final_single_win_odds_warning": True,
        "all_atomic_bet_configs": len(pooled),
        "all_atomic_profitable_in_sample": sum(r["profit_yen"] > 0 for r in pooled),
        "all_atomic_profitable_each_year_in_sample": len(stable),
        "highest_roi_any_size": top_roi[:30],
        "highest_roi_min_1000_races": sorted(sufficient, key=lambda x:-x["roi_pct"])[:50],
        "highest_absolute_profit": top_profit[:30],
        "all_three_years_profitable": sorted(stable,key=lambda x:-x["profit_yen"])[:60],
        "reference_baseline_all_year_parity": True,
        "warning": "Selecting profitable combos from same retrospective years is in-sample. Baseline L2.1 not modified. Distinct single tickets are building blocks for further deduplicated portfolios.",
    }
    return report, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--backfill-root", required=True)
    ap.add_argument("--prior-full-result", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    data, picks = load_predictions(a.predictions, a.picks)
    assert_ok(set(data.year.unique()) == set(YEARS), "2026 sealed or missing year")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        chunks = list(pool.map(lambda yr:scan_year(yr, data, picks,
                                                  Path(a.backfill_root)), YEARS))
    report, summary = make_report(chunks, a.prior_full_result)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out/"atomic-exact-rank-yearly-returns.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
                                   encoding="utf-8")
    print("EXACT_RANK_TICKET_STATS_DONE",flush=True)
    print(json.dumps({k:summary[k] for k in (
        "all_atomic_bet_configs", "all_atomic_profitable_in_sample",
        "all_atomic_profitable_each_year_in_sample",
        "highest_roi_min_1000_races", "highest_absolute_profit")},
        ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
