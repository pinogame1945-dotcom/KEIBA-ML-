#!/usr/bin/env python3
"""All-race 2023-2025 same-ticket exact-payout ranking anchor comparison.

Four PRE-RACE anchor selections:
  AI_RAW_TOP       -> L1 distill_ai_rank == 1 (tie-safe)
  AI_PROB_TOP      -> highest P__ANCHOR_FULL (market-anchored calibrated probability)
  MARKET_TOP       -> market_rank / final odds favorite
  VALUE_EV         -> frozen historical argmax P__ANCHOR_FULL * final_win_odds

Use fixed templates, same 100-yen units, no skips, no odds thresholds.
Results only used for settlement. 2026 locked.
"""
import argparse
import concurrent.futures
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import run_l2_value_anchor_bet_frontier_v1 as prior
from run_all_race_flow_ticket_audit_v1 import (
    YEARS, RACES_YEAR, TOTAL_RACES, get_inputs, assert_ok,
)
from build_l2_market_helpers_v1 import load_day, horse_number_map
from run_value_anchor_official_positions_v1 import positions

AXES = ("AI_RAW_TOP", "AI_PROB_TOP", "MARKET_TOP", "VALUE_EV")
SOURCES = ("AI_RAW", "AI_PROB", "MARKET")
KS = (2, 3, 5, 7)
TOP_K = (1, 2, 3, 6)


def load_predictions(pred_path, picks_path):
    main, picks = get_inputs(pred_path, picks_path)
    extra = pd.read_csv(
        pred_path, compression="infer",
        usecols=["year", "_race_id", "_horse_id", "_race_date", "distill_ai_rank"],
        dtype={"_race_id": str, "_horse_id": str}, low_memory=False,
    )
    extra["year"] = pd.to_numeric(extra["year"], errors="raise").astype(int)
    extra["distill_ai_rank"] = pd.to_numeric(extra["distill_ai_rank"], errors="raise")
    assert_ok(extra["distill_ai_rank"].notna().all(), "missing raw AI rank")
    assert_ok(extra["distill_ai_rank"].ge(1).all(), "invalid raw AI rank")
    assert_ok(not extra.duplicated(["_race_id", "_horse_id"]).any(), "extra duplicated")
    result = main.merge(extra, on=["year", "_race_id", "_horse_id"],
                        validate="one_to_one")
    assert_ok(len(result) == len(main), "merge lost horses")
    assert_ok(result["_race_date"].notna().all(), "race date missing")
    result["_race_date"] = result["_race_date"].astype(str).str[:10]
    assert_ok(result["_race_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all(),
              "bad date")
    return result, picks


def rank_horse_ids(group):
    market = group.sort_values(
        ["market_rank", "final_win_odds", "_horse_id"],
        ascending=[True, True, True], kind="mergesort",
    )["_horse_id"].astype(str).tolist()
    prob = group.sort_values(
        ["P__ANCHOR_FULL", "market_rank", "_horse_id"],
        ascending=[False, True, True], kind="mergesort",
    )["_horse_id"].astype(str).tolist()
    raw = group.sort_values(
        ["distill_ai_rank", "market_rank", "_horse_id"],
        ascending=[True, True, True], kind="mergesort",
    )["_horse_id"].astype(str).tolist()
    assert_ok(len(market) == len(set(market)), "duplicate horse")
    assert_ok(set(market) == set(prob) == set(raw), "order mismatch")
    return {"MARKET": market, "AI_PROB": prob, "AI_RAW": raw}


def one_year(year, all_data, picks, root):
    df = all_data[all_data.year.eq(year)].copy()
    groups = df.groupby("_race_id", sort=False)
    days = df[["_race_id", "_race_date"]].drop_duplicates()
    assert_ok(not days.duplicated("_race_id").any(), "duplicate race date")
    grouped_days = days.groupby("_race_date")["_race_id"].apply(list)

    counters = defaultdict(Counter)
    records = []
    ranking_coverage = defaultdict(Counter)
    race_count = 0

    for day, rids in grouped_days.items():
        packs = load_day(root / "data" / "daily" / (day + ".jsonl.gz"), set(rids))
        for rid in rids:
            pack = packs.get(rid)
            assert_ok(pack is not None, "missing pack " + str(rid))
            group = groups.get_group(rid)
            horse_map = horse_number_map(pack)
            finishing = positions(pack)
            paid = prior.payouts(pack, rid)
            order = rank_horse_ids(group)
            value_id = picks["ANCHOR_FULL"][rid]
            market_id = picks["MARKET_FAVORITE"][rid]
            assert_ok(market_id == order["MARKET"][0], "market pick tie drift " + str(rid))
            selected = {
                "AI_RAW_TOP": order["AI_RAW"][0],
                "AI_PROB_TOP": order["AI_PROB"][0],
                "MARKET_TOP": market_id,
                "VALUE_EV": value_id,
            }
            assert_ok(all(h in horse_map and h in finishing for h in selected.values()),
                      "missing mapped horse " + str(rid))
            row = {"year": year, "race_id": rid, "race_date": day}
            by_horse = group.set_index("_horse_id")

            for source in SOURCES:
                for k in TOP_K:
                    top = order[source][:k]
                    ranking_coverage[(source, k)]["races"] += 1
                    ranking_coverage[(source, k)]["win_races"] += int(
                        any(int(by_horse.loc[h, "target_win"]) for h in top))
                    ranking_coverage[(source, k)]["top3_any_races"] += int(
                        any(int(by_horse.loc[h, "target_top3"]) for h in top))
                    ranking_coverage[(source, k)]["top3_horses"] += sum(
                        int(by_horse.loc[h, "target_top3"]) for h in top)

            for axis_name, hid in selected.items():
                official = finishing[hid]
                assert_ok(official is not None and official >= 1,
                          "no finish for selected " + str(rid))
                is_win = int(by_horse.loc[hid, "target_win"])
                is_top3 = int(by_horse.loc[hid, "target_top3"])
                assert_ok(is_win == int(official == 1), "win target drift " + str(rid))
                assert_ok(is_top3 == int(official <= 3), "top3 target drift " + str(rid))
                bucket = {1: "FIRST", 2: "SECOND", 3: "THIRD"}.get(
                    official, "FOURTH_PLUS")
                axis_no = horse_map[hid]
                row[axis_name + "_horse"] = hid
                row[axis_name + "_finish"] = official
                row[axis_name + "_win"] = is_win
                row[axis_name + "_top3"] = is_top3
                row[axis_name + "_market_rank"] = float(by_horse.loc[hid, "market_rank"])
                row[axis_name + "_odds"] = float(by_horse.loc[hid, "final_win_odds"])
                row[axis_name + "_p"] = float(by_horse.loc[hid, "P__ANCHOR_FULL"])

                for family in ("WIN", "PLACE"):
                    receipt = paid.get((family, (axis_no,)), 0)
                    prior.update(counters[(axis_name, family, "SINGLE", 0)],
                                 1, receipt, bucket)

                for source in SOURCES:
                    other_numbers = [
                        horse_map[h] for h in order[source] if h != hid
                    ]
                    for k in KS:
                        families = prior.pair_templates(axis_no, other_numbers[:k])
                        for family, tickets in families.items():
                            assert_ok(all(axis_no in combo for _, combo in tickets),
                                      "axis absent in ticket")
                            rewards = [paid.get(t, 0) for t in tickets]
                            gained = sum(rewards)
                            key = (axis_name, family, source, k)
                            prior.update(counters[key], len(tickets), gained, bucket)
                            counters[key]["extra_hit_tickets"] += max(
                                0, sum(v > 0 for v in rewards) - int(gained > 0))
            row["RAW_market_same"] = int(
                selected["AI_RAW_TOP"] == selected["MARKET_TOP"])
            row["PROB_market_same"] = int(
                selected["AI_PROB_TOP"] == selected["MARKET_TOP"])
            row["RAW_prob_same"] = int(
                selected["AI_RAW_TOP"] == selected["AI_PROB_TOP"])
            records.append(row)
            race_count += 1

    assert_ok(race_count == RACES_YEAR, "races mismatch " + str(year))
    for (axis, family, src, k), c in counters.items():
        assert_ok(c["races"] == RACES_YEAR, "template incomplete " + str((axis, family, src, k)))
    return counters, records, ranking_coverage


def metrics_frame(chunks):
    accumulated = defaultdict(Counter)
    for year, (stats, _, _) in zip(YEARS, chunks):
        for key, c in stats.items():
            accumulated[(str(year), *key)].update(c)
            accumulated[("2023-2025", *key)].update(c)
    rows = []
    for (year, axis, family, source, k), c in accumulated.items():
        n = c["races"]
        expected = TOTAL_RACES if year == "2023-2025" else RACES_YEAR
        assert_ok(n == expected, "missing races for " + str((year, axis, family, source, k)))
        stake = c["stake_yen"]
        returned = c["return_yen"]
        rows.append({
            "year": year, "axis": axis, "family": family,
            "partner_order": source, "other_horses_k": k,
            "races": n, "bet_races": c["bet_races"],
            "tickets": c["tickets"], "avg_tickets_per_race": round(c["tickets"] / n, 5),
            "stake_yen": stake, "return_yen": round(returned, 2),
            "profit_yen": round(returned - stake, 2),
            "roi_pct": round(100 * returned / stake, 6),
            "hit_races": c["hit_races"], "hit_rate_pct": round(100 * c["hit_races"] / n, 6),
            "hit_tickets": c["hit_tickets"] + c["extra_hit_tickets"],
            "miss_races": c["miss_races"],
            **{"stake_" + b: c["stake_" + b] for b in prior.BUCKETS},
            **{"return_" + b: round(c["return_" + b], 2) for b in prior.BUCKETS},
        })
    return pd.DataFrame(rows).sort_values(
        ["year", "axis", "family", "partner_order", "other_horses_k"]
    ).reset_index(drop=True)


def coverage_frame(chunks):
    pooled = defaultdict(Counter)
    output = []
    for year, (_, _, metrics) in zip(YEARS, chunks):
        for key, c in metrics.items():
            pooled[key].update(c)
            output.append({"year": str(year), "ranking": key[0], "top_k": key[1],
                           **coverage_metrics(c)})
    for key, c in pooled.items():
        output.append({"year": "2023-2025", "ranking": key[0], "top_k": key[1],
                       **coverage_metrics(c)})
    return pd.DataFrame(output).sort_values(["year", "ranking", "top_k"])


def coverage_metrics(c):
    n = c["races"]
    return {"races": n,
            "win_capture_races": c["win_races"],
            "win_capture_pct": round(100 * c["win_races"] / n, 6),
            "top3_any_races": c["top3_any_races"],
            "top3_any_pct": round(100 * c["top3_any_races"] / n, 6),
            "top3_horses": c["top3_horses"],
            "avg_top3_horses_per_race": round(c["top3_horses"] / n, 6)}


def selection_summary(race_df):
    rows = []
    for name, group in [("2023-2025", race_df)] + [
        (str(year), race_df[race_df.year.eq(year)]) for year in YEARS
    ]:
        for axis in AXES:
            wins = group[axis + "_win"]
            top3 = group[axis + "_top3"]
            n = len(group)
            rows.append({
                "year": name, "axis": axis, "races": n,
                "win_races": int(wins.sum()), "win_pct": round(100 * wins.mean(), 6),
                "top3_races": int(top3.sum()), "top3_pct": round(100 * top3.mean(), 6),
                "avg_final_odds": round(group[axis + "_odds"].mean(), 6),
                "median_final_odds": round(group[axis + "_odds"].median(), 6),
                "avg_market_rank": round(group[axis + "_market_rank"].mean(), 6),
            })
    return pd.DataFrame(rows)


def overlap_summary(race_df):
    pairs = (("AI_RAW_TOP", "MARKET_TOP"), ("AI_PROB_TOP", "MARKET_TOP"),
             ("VALUE_EV", "MARKET_TOP"), ("AI_RAW_TOP", "AI_PROB_TOP"))
    rows = []
    for name, group in [("2023-2025", race_df)] + [
        (str(year), race_df[race_df.year.eq(year)]) for year in YEARS
    ]:
        for a, b in pairs:
            different = group[group[a + "_horse"].ne(group[b + "_horse"])]
            rows.append({
                "year": name, "selector_a": a, "selector_b": b,
                "races": len(group), "same_horse_races": len(group) - len(different),
                "same_horse_pct": round(100 * (len(group) - len(different)) / len(group), 6),
                "disagree_races": len(different),
                "A_only_wins": int((different[a + "_win"].eq(1) & different[b + "_win"].eq(0)).sum()),
                "B_only_wins": int((different[b + "_win"].eq(1) & different[a + "_win"].eq(0)).sum()),
                "A_only_top3": int((different[a + "_top3"].eq(1) & different[b + "_top3"].eq(0)).sum()),
                "B_only_top3": int((different[b + "_top3"].eq(1) & different[a + "_top3"].eq(0)).sum()),
                "both_top3": int((different[a + "_top3"].eq(1) & different[b + "_top3"].eq(1)).sum()),
                "neither_top3": int((different[a + "_top3"].eq(0) & different[b + "_top3"].eq(0)).sum()),
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--picks", required=True)
    p.add_argument("--backfill-root", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data, picks = load_predictions(a.predictions, a.picks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        chunks = list(ex.map(
            lambda year: one_year(year, data, picks, Path(a.backfill_root)), YEARS
        ))
    raw = pd.DataFrame([r for _, rec, _ in chunks for r in rec])
    assert_ok(len(raw) == TOTAL_RACES and raw.race_id.is_unique,
              "per-race comparison incomplete")
    totals = metrics_frame(chunks)
    ranks = coverage_frame(chunks)
    selectors = selection_summary(raw)
    overlaps = overlap_summary(raw)
    all_years = totals[totals.year.eq("2023-2025")].copy()

    assert_ok(all_years.groupby("axis").size().to_dict() == {
        name: 122 for name in AXES}, "wrong strategy coverage")
    baseline = all_years[
        all_years.axis.eq("VALUE_EV") & all_years.family.eq("TRIO") &
        all_years.partner_order.eq("AI_PROB") &
        all_years.other_horses_k.eq(2)
    ].iloc[0]
    assert_ok(abs(baseline["roi_pct"] - 88.888889) < 1e-5,
              "EV original result did not reproduce")
    assert_ok(abs(baseline["hit_rate_pct"] - 8.352623) < 1e-5,
              "EV original hit rate did not reproduce")
    assert_ok(int(selectors[
        selectors.year.eq("2023-2025") & selectors.axis.eq("VALUE_EV")
    ].iloc[0]["top3_races"]) == 4512, "value baseline top3 drift")

    totals.to_csv(out / "all-anchor-fixed-template-results.csv", index=False)
    ranks.to_csv(out / "ranking-topk-coverage.csv", index=False)
    selectors.to_csv(out / "axis-hit-and-rank-comparison.csv", index=False)
    overlaps.to_csv(out / "selection-overlap-and-disagreement.csv", index=False)
    raw.to_csv(out / "racewise-anchor-comparison.csv.gz", index=False,
               compression="gzip")
    anchor_fronts = []
    leaders = []
    for axis in AXES:
        report = all_years[all_years.axis.eq(axis)].copy()
        frame = report.rename(columns={
            "axis": "anchor_kind",
            "partner_order": "partner_order",
        })
        front, bands = prior.pareto(frame)
        front["axis"] = axis
        bands["axis"] = axis
        anchor_fronts.append(front)
        leaders.append(bands)
    pd.concat(anchor_fronts, ignore_index=True).to_csv(
        out / "per-anchor-cost-hit-roi-frontier.csv", index=False)
    pd.concat(leaders, ignore_index=True).to_csv(
        out / "per-anchor-budget-leaders.csv", index=False)

    best = all_years.sort_values(["roi_pct", "hit_rate_pct"], ascending=False).groupby(
        "axis", sort=False).head(5)
    popular_types = all_years[
        (all_years.family.isin(["WIN", "PLACE"])) |
        ((all_years.other_horses_k.eq(2)) &
         (all_years.partner_order.isin(["AI_RAW", "MARKET"])) &
         (all_years.family.isin(["QUINELLA", "WIDE", "EXACTA_FIRST", "TRIO"])))
    ].sort_values(["family", "other_horses_k", "axis", "partner_order"])
    popular_types.to_csv(out / "easy-compare-common-tickets.csv", index=False)
    summary = {
        "contract": "L2_AI_MARKET_RANK_BET_FRONTIER_V1",
        "years": list(YEARS), "races": TOTAL_RACES,
        "axes": list(AXES), "partner_rankings": list(SOURCES),
        "fixed_templates_per_axis": 122,
        "no_skip": True, "no_odds_filter": True, "stake_per_ticket_yen": 100,
        "selection_definitions": {
            "AI_RAW_TOP": "minimum distill_ai_rank, deterministic market/id tie break",
            "AI_PROB_TOP": "maximum P__ANCHOR_FULL, deterministic market/id tie break",
            "MARKET_TOP": "minimum market_rank then final_win_odds then horse_id",
            "VALUE_EV": "frozen ANCHOR_FULL maximum P__ANCHOR_FULL * final_win_odds",
        },
        "warning_market_anchor": "P__ANCHOR_FULL is not market independent: q_market * exp(delta) normalized.",
        "raw_ai_rank_source": "distill_ai_rank, the L1 distillation ranking stored in frozen 2023-2025 OOS predictions",
        "post_result_leakage_exclusion": "No result or payout participates in anchor/partner ticket selection.",
        "all_race_price": "Exact BACKFILL payouts pinned SHA " + prior.SHA,
        "match_reference_roi": float(baseline["roi_pct"]),
        "best_five_per_axis": best[[
            "axis", "family", "partner_order", "other_horses_k",
            "avg_tickets_per_race", "hit_rate_pct", "roi_pct", "profit_yen"
        ]].to_dict("records"),
        "best_profitable_template_counts": all_years.groupby("axis")[
            "profit_yen"].apply(lambda s: int(s.gt(0).sum())).to_dict(),
        "per_axis_top1": selectors[selectors.year.eq("2023-2025")].to_dict("records"),
        "market_vs_ai_disagreement": overlaps[
            overlaps.year.eq("2023-2025")].to_dict("records"),
        "free_standard_runner_only": True, "2026_locked": True,
        "interpretation": "Rank and payout comparisons are observational; no forward-validated switching policy.",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    print("RANKING_TOP1")
    print(selectors[selectors.year.eq("2023-2025")].to_csv(index=False), flush=True)
    print("RANKING_TOPK")
    print(ranks[ranks.year.eq("2023-2025")].to_csv(index=False), flush=True)
    print("DISAGREEMENT")
    print(overlaps[overlaps.year.eq("2023-2025")].to_csv(index=False), flush=True)
    print("BEST_PER_AXIS")
    print(best.to_csv(index=False), flush=True)
    print("RANK_VS_MARKET_FRONTIER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
