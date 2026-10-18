#!/usr/bin/env python3
"""Present all historical EV1..6 x pure AI results at inspectable granularity.

Only reads completed settlement CSV, never reloads predictions/backfill or refits.
Keep full precision carried by the original file. 2026 sealed.
"""
import csv
import json
from collections import Counter
from pathlib import Path

RUN = "39112399214"
ROOT = Path(f"research-results/l2-ev-top6-raw-ai-allbets-v1/run-{RUN}")
SRC = ROOT / "all-ev-top6-raw-ai-templates.csv"
YEARS = ("2023", "2024", "2025")
FAMILIES = ("WIN", "PLACE", "QUINELLA", "WIDE", "EXACTA_FIRST",
            "EXACTA_SECOND", "EXACTA_BOTH", "TRIO",
            "TRIFECTA_FIRST", "TRIFECTA_SECOND", "TRIFECTA_THIRD", "TRIFECTA_ANY")
BUCKETS = ("1", "2", "3", "4-6", "7-10", "11+")
FIN = ("races", "bet_races", "tickets", "hit_races", "hit_tickets",
       "stake_yen", "return_yen", "profit_yen", "roi_pct", "hit_rate_pct")


def write_csv(path, rows):
    assert rows, f"empty output {path}"
    cols = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with SRC.open(newline="", encoding="utf-8") as fh:
        source = list(csv.DictReader(fh))
    assert len(source) > 10000, len(source)
    index = {}
    for r in source:
        key = (r["year"], int(r["ev_axis_rank"]), r["market_rank_bucket"],
               r["family"], int(r["other_horses_k"]))
        assert key not in index, key
        index[key] = r

    def num(r, name):
        val = r[name]
        return float(val) if val else float("nan")

    def parsed(r):
        return {
            "year": r["year"], "ev_axis_rank": int(r["ev_axis_rank"]),
            "single_win_popularity": r["market_rank_bucket"],
            "bet_family": r["family"], "pure_ai_partner_top_k": int(r["other_horses_k"]),
            **{k: r[k] for k in FIN},
        }

    pooled = sorted((r for r in source if r["year"] == "2023-2025" and
                     r["market_rank_bucket"] == "ALL" and int(r["stake_yen"]) > 0),
                    key=lambda r: (int(r["ev_axis_rank"]), r["family"],
                                   int(r["other_horses_k"])))
    assert len(pooled) == 6 * 172
    full = []
    for r in pooled:
        rank = int(r["ev_axis_rank"])
        k = int(r["other_horses_k"])
        family = r["family"]
        rec = {"ev_axis_rank": rank, "bet_family": family, "pure_ai_partner_top_k": k}
        for label, year in [("all3", "2023-2025"), ("y2023", "2023"),
                            ("y2024", "2024"), ("y2025", "2025")]:
            hit = index[(year, rank, "ALL", family, k)]
            for v in FIN:
                rec[label + "_" + v] = hit[v]
        full.append(rec)
    write_csv(ROOT / "detail-all-1032-combinations-yearly.csv", full)

    family_rows = []
    for rank in range(1, 7):
        for family in FAMILIES:
            matches = [r for r in pooled if int(r["ev_axis_rank"]) == rank and
                       r["family"] == family]
            best = max(matches, key=lambda r: (num(r, "roi_pct"),
                                                num(r, "profit_yen"),
                                                -int(r["other_horses_k"])))
            k = int(best["other_horses_k"])
            rec = {"ev_axis_rank": rank, "bet_family": family,
                   "pure_ai_partner_top_k": k}
            for label, year in [("all3", "2023-2025"), ("y2023", "2023"),
                                ("y2024", "2024"), ("y2025", "2025")]:
                hit = index[(year, rank, "ALL", family, k)]
                for v in FIN:
                    rec[label + "_" + v] = hit[v]
            family_rows.append(rec)
    assert len(family_rows) == 72
    write_csv(ROOT / "detail-best-k-per-rank-and-bet-family-72.csv", family_rows)

    positive, positive_all_three = [], []
    for r in source:
        if r["year"] != "2023-2025" or r["market_rank_bucket"] == "ALL":
            continue
        if not r["stake_yen"] or not r["profit_yen"] or float(r["profit_yen"]) <= 0:
            continue
        rank = int(r["ev_axis_rank"])
        bucket = r["market_rank_bucket"]
        family = r["family"]
        k = int(r["other_horses_k"])
        result = parsed(r)
        year_profit = {}
        for year in YEARS:
            detail = index[(year, rank, bucket, family, k)]
            year_profit[year] = detail["profit_yen"]
            for metric in FIN:
                result[year + "_" + metric] = detail[metric]
        positive.append(result)
        if all(float(year_profit[y]) > 0 for y in YEARS):
            positive_all_three.append(result)

    positive.sort(key=lambda r: -float(r["profit_yen"]))
    positive_all_three.sort(key=lambda r: -float(r["profit_yen"]))
    if positive:
        write_csv(ROOT / "detail-single-win-popularity-positive-3yr.csv", positive)
    if positive_all_three:
        write_csv(ROOT / "detail-single-win-popularity-each-year-positive.csv",
                  positive_all_three)

    compact = {
        "contract": "L2_EV_TOP6_RAW_AI_DETAILED_REPORT_V1",
        "input": str(SRC), "source_run": RUN,
        "years": list(YEARS), "total_races": 10368,
        "all_races_strategy_count": len(pooled),
        "all_races_positive_count": sum(float(r["profit_yen"]) > 0 for r in pooled),
        "per_rank_family_all_12": family_rows,
        "by_rank_pooled_highest_roi": {
            str(rank): parsed(max((r for r in pooled
                       if int(r["ev_axis_rank"]) == rank),
                       key=lambda x: num(x, "roi_pct")))
            for rank in range(1, 7)},
        "popularity_positive_count": len(positive),
        "popularity_each_year_positive_count": len(positive_all_three),
        "popularity_positive_by_rank": dict(Counter(str(r["ev_axis_rank"])
                                                  for r in positive)),
        "popularity_positive_by_bucket": dict(Counter(r["single_win_popularity"]
                                                    for r in positive)),
        "top_30_popularity_positive_by_absolute_profit": positive[:30],
        "all3_positive_min_300_tickets": [
            r for r in positive_all_three if int(r["tickets"]) >= 300][:30],
        "interpretation": [
            "All 2023-25 pooled positive templates in all-race slice are zero.",
            "Popularity slice results are in-sample diagnoses, not vetted bet rules.",
            "EV uses frozen historical final win odds; no exotic bet odds for selection.",
            "Full historical precision preserved as recorded in base source CSV.",
            "Results do not authorize changing frozen L2.1.",
        ],
        "sealed_2026": True
    }
    (ROOT / "detail-summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("DETAIL_DONE 1032 all-race combinations, 72 rank-family")
    print("POPULARITY", len(positive), "positive", len(positive_all_three),
          "each-year-positive")
    print("RANK_FAMILY_ROWS")
    for r in family_rows:
        print(",".join(str(r[x]) for x in (
            "ev_axis_rank", "bet_family", "pure_ai_partner_top_k",
            "all3_races", "all3_tickets", "all3_hit_races",
            "all3_stake_yen", "all3_return_yen", "all3_profit_yen",
            "all3_roi_pct", "y2023_roi_pct", "y2024_roi_pct", "y2025_roi_pct")))
    print("DETAIL_VERIFIED")


if __name__ == "__main__":
    main()
