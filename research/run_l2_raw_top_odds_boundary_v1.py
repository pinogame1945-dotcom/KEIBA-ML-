#!/usr/bin/env python3
"""RAW AI rank-1 anchor: exactly the frozen predecessor's 152 tickets templates x 20 odds bands.

PHASE 1: all ticket candidates, disjoint final-odds bands, paid results, 2023-25.
PHASE 2: same horse orders and families, reduce k=10,7,5,3,2, track black/red
         odds band and cumulative-minimum-odds break-even border movement.
Research only: same-period descriptive results, FINAL odds, no 2026, no L2.1 change.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import run_l2_fixed_ticket_pruning_3040_v1 as fixed
import run_l2_ai_market_rank_betfrontier_v1 as ranklab
from build_l2_market_helpers_v1 import load_day, load_odds_day
from run_all_race_flow_ticket_audit_v1 import YEARS, RACES_YEAR, TOTAL_RACES, assert_ok

K_VALUES = (2, 3, 5, 7, 10)
ORDERS = tuple(fixed.lab.ORDERS)
FAMILIES = tuple(fixed.FAMILIES)
BINS = tuple(fixed.BINS)
SINGLES = (("SINGLE", 0, "WIN"), ("SINGLE", 0, "PLACE"))
TEMPLATES = SINGLES + tuple((order, k, fam) for order in ORDERS
                            for k in K_VALUES for fam in FAMILIES)
assert len(TEMPLATES) == 152 and len(BINS) == 20
PERIODS = tuple(str(y) for y in YEARS) + ("2023-2025",)


def bump(store, key, c):
    store[key].update(c)


def calc(counts):
    n, hits, payout = counts["tickets"], counts["hits"], counts["payout_yen"]
    stake = n * 100
    gain = payout - stake
    return dict(tickets=n, hit_tickets=hits, stake_yen=stake,
                payout_yen=payout, profit_yen=gain,
                roi_pct=round(100 * payout / stake, 6) if stake else None,
                color=("NO_TICKETS" if not n else "BLACK" if gain > 0
                       else "RED" if gain < 0 else "BREAK_EVEN"),
                bought_races=counts["races"])


def combine(store, year, key, counter):
    bump(store, (str(year), *key), counter)
    bump(store, ("2023-2025", *key), counter)


def write_csv(path, rows):
    if not rows:
        raise ValueError("no rows " + str(path))
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def yearly_data(pred_path, picks_path, racewise_path):
    pred, picks = ranklab.load_predictions(pred_path, picks_path)
    rs = pd.read_csv(racewise_path, compression="infer",
                     dtype={"race_id": str, "AI_RAW_TOP_horse": str,
                            "VALUE_EV_horse": str})
    rs["year"] = pd.to_numeric(rs["year"], errors="raise").astype(int)
    assert_ok(len(rs) == TOTAL_RACES and rs.race_id.is_unique,
              "racewise coverage drift")
    assert_ok(set(rs.year) == set(YEARS), "unexpected years")
    assert_ok(set(rs.race_id) == set(pred._race_id.astype(str)),
              "racewise/prediction races drift")
    return pred, picks, rs


def analyze(pred, picks, races, backfill_root):
    lines = defaultdict(Counter)
    dedup = defaultdict(Counter)
    covered = Counter()
    equal_anchor = 0

    for year in YEARS:
        yearly = races.loc[races.year.eq(year)]
        grouped = pred.loc[pred.year.eq(year)].groupby("_race_id", sort=False)
        by_race = yearly.set_index("race_id")
        days = yearly.groupby("race_date")["race_id"].apply(list)
        for day, ids in days.items():
            day = str(day)[:10]
            packs = load_day(backfill_root / "data/daily" / (day + ".jsonl.gz"), set(ids))
            odds = load_odds_day(backfill_root / "data/odds/daily" / (day + ".jsonl.gz"), set(ids))
            for rid in sorted(ids):
                pack, market = packs.get(rid), odds.get(rid)
                assert_ok(pack is not None and market is not None, "missing race " + rid)
                assert_ok(market.get("odds_kind") == "FINAL" and
                          market.get("source_status") == "result", "odds not FINAL " + rid)
                g = grouped.get_group(rid)
                info = by_race.loc[rid]
                top = str(info["AI_RAW_TOP_horse"])
                assert_ok(top == str(g.sort_values(
                    ["distill_ai_rank", "market_rank", "_horse_id"],
                    ascending=[True, True, True], kind="mergesort"
                ).iloc[0]["_horse_id"]), "AI_RAW rank drift " + rid)
                assert_ok(str(info["VALUE_EV_horse"]) == picks["ANCHOR_FULL"][rid],
                          "frozen EV anchor drift " + rid)
                equal_anchor += int(top == picks["ANCHOR_FULL"][rid])

                # The original 152-template function only reads VALUE_EV_horse
                # for a guard assertion. Supply a temporary adapter Series
                # with RAW as its anchor, leaving persisted inputs untouched.
                # All 152 ticket templates and 20 price bands use original code.
                shim = info.copy()
                shim["VALUE_EV_horse"] = top
                per_line = defaultdict(Counter)
                physical = {}
                seen_in_line = set()
                for tid, ticket, card in fixed.create_cards(g, pack, market, rid, top, shim):
                    assert_ok(tid[:3] in TEMPLATES, "new template " + str(tid))
                    pair = (tid, ticket)
                    assert_ok(pair not in seen_in_line,
                              "duplicate physical ticket within line " + str(pair))
                    seen_in_line.add(pair)
                    c = per_line[tid]
                    c["tickets"] += 1
                    c["hits"] += int(card[1] > 0)
                    c["payout_yen"] += int(card[1])
                    if ticket in physical:
                        assert_ok(physical[ticket] == (float(card[0]), int(card[1])),
                                  "physical ticket price/settlement drift")
                    else:
                        physical[ticket] = (float(card[0]), int(card[1]))

                for tid, c in per_line.items():
                    c["races"] = 1
                    combine(lines, year, tid, c)
                physical_counters = defaultdict(Counter)
                for (bet_type, _selection), (price, amount) in physical.items():
                    band = fixed.bin_name(BINS[fixed.bisect_right(BINS, price) - 1])
                    for key in (("ALL", "ALL"), (bet_type, band)):
                        c = physical_counters[key]
                        c["tickets"] += 1
                        c["hits"] += int(amount > 0)
                        c["payout_yen"] += amount
                for key, c in physical_counters.items():
                    c["races"] = 1
                    combine(dedup, year, key, c)

                covered[str(year)] += 1
                if covered[str(year)] % 1200 == 0:
                    print("RAW_ANCHOR_PROGRESS", year, covered[str(year)], flush=True)
        assert_ok(covered[str(year)] == RACES_YEAR,
                  "race count drift " + str(year))
    assert_ok(sum(covered.values()) == TOTAL_RACES, "total races drift")
    return lines, dedup, covered, equal_anchor


def outputs(lines, dedup):
    band_rows = []
    floor_rows = []
    all_bands = tuple(fixed.bin_name(b) for b in BINS)
    for period in PERIODS:
        for order, k, family in TEMPLATES:
            suffix = [lines.get((period, order, k, family, band), Counter())
                      for band in all_bands]
            running = Counter()
            for i in range(len(BINS)-1, -1, -1):
                band = all_bands[i]
                base = {"period": period, "partner_order": order,
                        "other_horses_k": k, "family": family,
                        "odds_band": band, "band_floor": BINS[i],
                        "band_ceiling_exclusive": (BINS[i+1] if i+1 < len(BINS) else None)}
                band_rows.append({**base, **calc(suffix[i])})
                running.update(suffix[i])
                floor_rows.append({**base, **calc(running)})

    row_key = lambda r: (r["period"], r["partner_order"], r["other_horses_k"],
                         r["family"], r["odds_band"])
    bands = {row_key(r): r for r in band_rows}
    floors = {row_key(r): r for r in floor_rows}
    reductions = []
    for order in ORDERS:
        for family in FAMILIES:
            for k in (10, 7, 5, 3, 2):
                base_bands = [bands[("2023-2025", order, 10, family, b)] for b in all_bands]
                now_bands = [bands[("2023-2025", order, k, family, b)] for b in all_bands]
                base_floors = [floors[("2023-2025", order, 10, family, b)] for b in all_bands]
                now_floors = [floors[("2023-2025", order, k, family, b)] for b in all_bands]

                def first_black(rows):
                    for r in rows:
                        if r["color"] == "BLACK":
                            return r["band_floor"]
                    return None

                def stable(row):
                    b = row["odds_band"]
                    return all(bands[(str(year), order, k, family, b)]["color"] == "BLACK"
                               for year in YEARS)

                reductions.append(dict(
                    partner_order=order, family=family, other_horses_k=k,
                    baseline_k=10,
                    k10_lowest_black_disjoint_band=first_black(base_bands),
                    k_lowest_black_disjoint_band=first_black(now_bands),
                    k10_lowest_profitable_cumulative_floor=first_black(base_floors),
                    k_lowest_profitable_cumulative_floor=first_black(now_floors),
                    k10_black_bands=sum(r["color"] == "BLACK" for r in base_bands),
                    k_black_bands=sum(r["color"] == "BLACK" for r in now_bands),
                    red_to_black_vs_k10=sum(
                        a["color"] == "RED" and b["color"] == "BLACK"
                        for a, b in zip(base_bands, now_bands)),
                    black_to_red_vs_k10=sum(
                        a["color"] == "BLACK" and b["color"] == "RED"
                        for a, b in zip(base_bands, now_bands)),
                    all_years_black_bands=sum(stable(r) for r in now_bands),
                    all_tickets_in_20_bands=sum(r["tickets"] for r in now_bands),
                    all_payout_yen_in_20_bands=sum(r["payout_yen"] for r in now_bands),
                    # First positive threshold is descriptive, not a gate to deploy.
                ))

    dedup_rows = []
    for period in PERIODS:
        for bet, band in [("ALL", "ALL")] + [
            (kind, b) for kind in ("WIN", "PLACE", "QUINELLA", "WIDE",
                                  "EXACTA", "TRIO", "TRIFECTA") for b in all_bands]:
            dedup_rows.append(dict(period=period, bet_type=bet, odds_band=band,
                                   **calc(dedup.get((period, bet, band), Counter()))))

    overall = next(r for r in dedup_rows if r["period"] == "2023-2025" and
                   r["bet_type"] == "ALL")
    assert_ok(len(band_rows) == len(PERIODS)*3040 and
              len(floor_rows) == len(PERIODS)*3040, "3040 line grid drift")
    assert_ok(len(reductions) == len(ORDERS)*len(FAMILIES)*5,
              "k reduction grid drift")
    assert_ok(overall["tickets"] > 0, "no physical ticket coverage")
    assert_ok(sum(r["tickets"] for r in dedup_rows if r["period"]=="2023-2025"
                  and r["bet_type"] != "ALL") == overall["tickets"],
              "dedup type totals drift")
    return band_rows, floor_rows, reductions, dedup_rows


def main():
    p = argparse.ArgumentParser()
    for arg in ("predictions", "picks", "racewise", "backfill-root", "out-dir"):
        p.add_argument("--" + arg, required=True)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data, picks, races = yearly_data(a.predictions, a.picks, a.racewise)
    lines, dedup, covered, match = analyze(data, picks, races, Path(a.backfill_root))
    bands, floors, changes, dedup_rows = outputs(lines, dedup)
    files = {
        "phase1_all_3040_odds_bands_by_year.csv": bands,
        "phase1_all_3040_cumulative_odds_floors_by_year.csv": floors,
        "phase1_dedup_physical_tickets.csv": dedup_rows,
        "phase2_k10_to_7_5_3_2_border_moves.csv": changes,
    }
    for name, rows in files.items():
        write_csv(out/name, rows)
    summary = dict(
        contract="L2_RAW_TOP_152X20_ODDS_BORDER_V1",
        phase1="ALL_ORIGINAL_152_TEMPLATES_X_20_FINAL_ODDS_BANDS",
        phase2="SAME_PARTNER_ORDER_AND_FAMILY_K_REDUCTION_10_TO_7_TO_5_TO_3_TO_2",
        original_model_axis="distill_ai_rank_1_tie_market_rank_horse_id",
        original_ev_axis_or_frozen_l21_changed=False,
        year_counts=dict(covered), same_horse_raw_and_ev_races=match,
        full_template_lines=3040, original_templates=152, odds_bands=20,
        physical_ticket_baseline=next(r for r in dedup_rows
                                     if r["period"]=="2023-2025"
                                     and r["bet_type"]=="ALL"),
        phase1_pooled_black_lines=sum(r["color"]=="BLACK" for r in bands
                                      if r["period"]=="2023-2025"),
        phase1_pooled_red_lines=sum(r["color"]=="RED" for r in bands
                                    if r["period"]=="2023-2025"),
        phase1_pooled_no_ticket_lines=sum(r["color"]=="NO_TICKETS" for r in bands
                                          if r["period"]=="2023-2025"),
        phase2_comparison_rows=len(changes),
        phase2_red_to_black_vs_k10=sum(r["red_to_black_vs_k10"] for r in changes),
        phase2_black_to_red_vs_k10=sum(r["black_to_red_vs_k10"] for r in changes),
        per_line_tickets_repeated_across_templates=True,
        physical_union_is_deduplicated_per_race=True,
        same_period_profitability_selection_is_not_oos_proof=True,
        year_2026_sealed=True,
        pricing="HISTORICAL_FINAL_NOT_PURCHASABLE_CUTOFF_ODDS",
        result_years=[2023,2024,2025],
        line_black_definition="100YEN_TICKET_PAYOUT_GREATER_THAN_STAKE",
        boundary_is_descriptive_only=True,
        no_new_filter_or_operational_rule_adopted=True,
        no_paid_artifacts_or_gpu=True,
    )
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
                                     encoding="utf-8")
    print("RAW_TOP_3040_PHASE1_PHASE2_COMPLETE",json.dumps(summary,ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    main()
