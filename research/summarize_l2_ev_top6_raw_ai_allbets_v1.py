#!/usr/bin/env python3
"""Summarize the already-settled EV1..6 x pure AI results without rerunning models."""
import csv
import json
from collections import Counter
from pathlib import Path

RUN = "39112399214"
ROOT = Path("research-results/l2-ev-top6-raw-ai-allbets-v1/run-" + RUN)
PATH = ROOT / "all-ev-top6-raw-ai-templates.csv"


def main():
    with PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and len(rows) > 2000
    for r in rows:
        for key in ("ev_axis_rank", "other_horses_k", "races", "tickets",
                    "stake_yen", "hit_races"):
            r[key] = int(r[key])
        for key in ("roi_pct", "profit_yen", "return_yen"):
            r[key] = float(r[key]) if r[key] else None

    def show(r):
        return {key: r[key] for key in (
            "year", "ev_axis_rank", "market_rank_bucket", "family",
            "other_horses_k", "races", "tickets", "hit_races",
            "stake_yen", "return_yen", "profit_yen", "roi_pct")}

    def leading(group, count=10, by="roi_pct"):
        return [show(r) for r in sorted(
            (r for r in group if r["stake_yen"] and r["roi_pct"] is not None),
            key=lambda r: (r[by], r["stake_yen"]), reverse=True)[:count]]

    all_pooled = [r for r in rows if r["year"] == "2023-2025" and
                  r["market_rank_bucket"] == "ALL"]
    assert len(all_pooled) == 6 * 172
    rank_best = {str(rank): leading(
        [r for r in all_pooled if r["ev_axis_rank"] == rank], 3)
        for rank in range(1, 7)}
    by_year_best = {str(year): {
        str(rank): leading([r for r in rows if r["year"] == str(year) and
                           r["market_rank_bucket"] == "ALL" and
                           r["ev_axis_rank"] == rank], 1)
        for rank in range(1, 7)}
        for year in (2023, 2024, 2025)}

    bucket_pooled = [r for r in rows if r["year"] == "2023-2025" and
                     r["market_rank_bucket"] != "ALL" and r["stake_yen"] > 0]
    positive = [r for r in bucket_pooled if r["profit_yen"] > 0]
    stable = []
    for r in positive:
        yearly = [x for x in rows if x["year"] in ("2023", "2024", "2025") and
                  x["market_rank_bucket"] == r["market_rank_bucket"] and
                  x["ev_axis_rank"] == r["ev_axis_rank"] and
                  x["family"] == r["family"] and
                  x["other_horses_k"] == r["other_horses_k"]]
        if len(yearly) == 3 and all(x["profit_yen"] is not None and x["profit_yen"] > 0
                                     for x in yearly):
            stable.append(r)
    report = {
        "source_run": RUN, "source_csv": str(PATH), "2026_sealed": True,
        "all_race_pooled": {
            "templates_per_rank": 172,
            "positive_count": sum(r["profit_yen"] > 0 for r in all_pooled),
            "top3_by_ev_rank": rank_best,
            "best_by_year_and_rank": by_year_best,
        },
        "market_rank_bucket_diagnostics_ONLY_not_bet_rules": {
            "total_nonzero_bucket_templates": len(bucket_pooled),
            "three_year_pooled_positive_count": len(positive),
            "positive_with_each_individual_year_positive_count": len(stable),
            "positive_by_market_bucket": dict(Counter(
                r["market_rank_bucket"] for r in positive)),
            "positive_by_ev_rank": dict(Counter(
                str(r["ev_axis_rank"]) for r in positive)),
            "largest_profit_pooled_positive": leading(positive, 20, "profit_yen"),
            "highest_roi_pooled_positive_min300tickets": leading(
                [r for r in positive if r["tickets"] >= 300], 20, "roi_pct"),
            "each_year_positive": leading(stable, 20, "profit_yen"),
        },
        "caveat": "Buckets and leading strategies selected on same 2023-25 records. NOT forward validated. FINAL single-win odds were used for EV rank; ticket odds were NOT selection gates.",
    }
    dest = ROOT / "compact-finding.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print("COMPACT_L2_EV_TOP6_DONE", flush=True)
    print("ALL_RACE_TOP_1_EACH_RANK")
    for rank, top in rank_best.items():
        print(rank, top[:1], flush=True)
    print("MARKET_BUCKET_COUNTS", len(positive), "STABLE", len(stable), flush=True)


if __name__ == "__main__":
    main()
