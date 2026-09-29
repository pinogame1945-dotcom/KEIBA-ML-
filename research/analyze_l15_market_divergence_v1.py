#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

EXPECTED_YEARS = [2022, 2023, 2024, 2025]
EXPECTED_RACES_PER_YEAR = 3456

def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "rt", encoding="utf-8")

def parse_map(values, label):
    out = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"{label} must be YEAR=PATH: {raw}")
        y, p = raw.split("=", 1)
        y = int(y)
        out[y] = p
    return out

def finite(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def integer(v):
    x = finite(v)
    if x is None:
        return None
    i = int(round(x))
    return i if i >= 1 else None

def seven_order(router_row):
    experts = router_row.get("experts") or {}
    if len(experts) != 7:
        raise ValueError(f"expected 7 experts race_id={router_row.get('race_id')} got={len(experts)}")
    stats = defaultdict(lambda: {
        "support": 0,
        "borda": 0.0,
        "top1_votes": 0,
        "best_rank": 99,
        "rank_sum": 0.0,
    })
    for expert in experts.values():
        ids = [str(x) for x in (expert.get("top6_horse_ids") or []) if str(x)]
        if not ids:
            raise ValueError(f"expert missing Top6 race_id={router_row.get('race_id')}")
        for rank, hid in enumerate(ids, 1):
            s = stats[hid]
            s["support"] += 1
            s["borda"] += float(7-rank)
            s["top1_votes"] += int(rank == 1)
            s["best_rank"] = min(s["best_rank"], rank)
            s["rank_sum"] += rank
    for s in stats.values():
        s["mean_rank"] = s["rank_sum"] / s["support"]
    return sorted(
        stats,
        key=lambda hid: (
            -stats[hid]["borda"],
            -stats[hid]["support"],
            -stats[hid]["top1_votes"],
            stats[hid]["best_rank"],
            stats[hid]["mean_rank"],
            hid,
        ),
    )

def load_market(path):
    market = {}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get("race_id") or "")
            hid = str(row.get("horse_id") or "")
            if not rid or not hid:
                continue
            m = row.get("market_outcome") or {}
            t = row.get("target") or {}
            pop = integer(m.get("final_popularity"))
            odds = finite(m.get("final_win_odds"))
            finish = integer(t.get("finish_position"))
            if pop is None:
                pop = integer(row.get("final_popularity"))
            if odds is None:
                odds = finite(row.get("final_win_odds"))
            # compatibility fallback for older snapshot forms
            if pop is None:
                pop = integer(row.get("popularity"))
            if odds is None:
                odds = finite(row.get("win_odds"))
            market[(rid, hid)] = {"popularity": pop, "odds": odds, "finish": finish}
    return market

def load_router(path):
    rows = []
    with open_text(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def safe_mean(xs):
    return sum(xs) / len(xs) if xs else None

def safe_median(xs):
    return statistics.median(xs) if xs else None

def pearson(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = safe_mean(xs), safe_mean(ys)
    dx = [x-mx for x in xs]
    dy = [y-my for y in ys]
    den = math.sqrt(sum(x*x for x in dx) * sum(y*y for y in dy))
    if den == 0:
        return None
    return sum(a*b for a,b in zip(dx,dy)) / den

def pct(n, d):
    return (100.0*n/d) if d else None

def summarize(items, races):
    valid = [x for x in items if x["popularity"] is not None]
    gaps = [x["popularity"] - x["consensus_rank"] for x in valid]
    abs_gaps = [abs(g) for g in gaps]
    odds = [x["odds"] for x in valid if x["odds"] is not None]
    top1 = [x for x in valid if x["consensus_rank"] == 1]

    race_overlap = []
    race_exact6 = 0
    race_has_market1 = 0
    race_corrs = []
    six_horse_box_eligible_races = 0
    small_field_or_short_consensus_races = 0
    for r in races:
        vals = [x for x in r["items"] if x["popularity"] is not None]
        if len(vals) != 6:
            small_field_or_short_consensus_races += 1
            continue
        six_horse_box_eligible_races += 1
        overlap = sum(1 for x in vals if x["popularity"] <= 6)
        race_overlap.append(overlap / 6.0)
        race_exact6 += int(overlap == 6)
        race_has_market1 += int(any(x["popularity"] == 1 for x in vals))
        corr = pearson(
            [float(x["consensus_rank"]) for x in vals],
            [float(x["popularity"]) for x in vals],
        )
        if corr is not None:
            race_corrs.append(corr)

    return {
        "races": len(races),
        "top6_rows": len(items),
        "market_joined_rows": len(valid),
        "market_join_coverage_pct": pct(len(valid), len(items)),
        "top1_same_as_market_favorite_pct": pct(sum(1 for x in top1 if x["popularity"] == 1), len(top1)),
        "top1_market_popularity_mean": safe_mean([x["popularity"] for x in top1]),
        "top1_market_popularity_median": safe_median([x["popularity"] for x in top1]),
        "top1_final_odds_mean": safe_mean([x["odds"] for x in top1 if x["odds"] is not None]),
        "top1_final_odds_median": safe_median([x["odds"] for x in top1 if x["odds"] is not None]),
        "top6_member_is_market_top6_pct": pct(sum(1 for x in valid if x["popularity"] <= 6), len(valid)),
        "top6_member_is_market_top3_pct": pct(sum(1 for x in valid if x["popularity"] <= 3), len(valid)),
        "top6_member_market_7plus_pct": pct(sum(1 for x in valid if x["popularity"] >= 7), len(valid)),
        "top6_member_market_10plus_pct": pct(sum(1 for x in valid if x["popularity"] >= 10), len(valid)),
        "rank_gap_mean_market_minus_consensus": safe_mean(gaps),
        "rank_gap_median_market_minus_consensus": safe_median(gaps),
        "absolute_rank_gap_mean": safe_mean(abs_gaps),
        "absolute_rank_gap_median": safe_median(abs_gaps),
        "consensus_rates_horse_3plus_ranks_above_market_pct": pct(sum(1 for g in gaps if g >= 3), len(gaps)),
        "market_rates_horse_3plus_ranks_above_consensus_pct": pct(sum(1 for g in gaps if g <= -3), len(gaps)),
        "final_odds_mean_top6_members": safe_mean(odds),
        "final_odds_median_top6_members": safe_median(odds),
        "six_horse_box_eligible_races": six_horse_box_eligible_races,
        "small_field_or_short_consensus_races": small_field_or_short_consensus_races,
        "race_top6_overlap_with_market_top6_mean_pct": 100.0*safe_mean(race_overlap) if race_overlap else None,
        "race_exact_same_top6_set_pct": pct(race_exact6, len(race_overlap)),
        "race_consensus_top6_contains_market_favorite_pct": pct(race_has_market1, len(race_overlap)),
        "race_consensus_vs_market_rank_correlation_mean": safe_mean(race_corrs),
    }

def rank_summary(items):
    out = []
    for rank in range(1, 11):
        rows = [x for x in items if x["consensus_rank"] == rank and x["popularity"] is not None]
        pops = [x["popularity"] for x in rows]
        odds = [x["odds"] for x in rows if x["odds"] is not None]
        gaps = [x["popularity"] - rank for x in rows]
        finish_rows = [x for x in rows if x.get("finish") is not None]
        out.append({
            "consensus_rank": rank,
            "n": len(rows),
            "market_popularity_mean": safe_mean(pops),
            "market_popularity_median": safe_median(pops),
            "final_odds_mean": safe_mean(odds),
            "final_odds_median": safe_median(odds),
            "exact_same_rank_pct": pct(sum(1 for x in rows if x["popularity"] == rank), len(rows)),
            "market_top3_pct": pct(sum(1 for x in rows if x["popularity"] <= 3), len(rows)),
            "market_top6_pct": pct(sum(1 for x in rows if x["popularity"] <= 6), len(rows)),
            "market_7plus_pct": pct(sum(1 for x in rows if x["popularity"] >= 7), len(rows)),
            "market_10plus_pct": pct(sum(1 for x in rows if x["popularity"] >= 10), len(rows)),
            "gap_mean_market_minus_consensus": safe_mean(gaps),
            "abs_gap_mean": safe_mean([abs(x) for x in gaps]),
            "finish_known_n": len(finish_rows),
            "win_pct": pct(sum(1 for x in finish_rows if x["finish"] == 1), len(finish_rows)),
            "top3_pct": pct(sum(1 for x in finish_rows if x["finish"] <= 3), len(finish_rows)),
            "top6_finish_pct": pct(sum(1 for x in finish_rows if x["finish"] <= 6), len(finish_rows)),
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="append", required=True, help="YEAR=PATH")
    ap.add_argument("--router", action="append", required=True, help="YEAR=PATH")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    snapshots = parse_map(a.snapshot, "snapshot")
    routers = parse_map(a.router, "router")
    if sorted(snapshots) != EXPECTED_YEARS or sorted(routers) != EXPECTED_YEARS:
        raise SystemExit(f"expected years {EXPECTED_YEARS}; snapshots={sorted(snapshots)} routers={sorted(routers)}")

    all_items = []
    all_rank_items = []
    all_races = []
    year_summaries = []

    for year in EXPECTED_YEARS:
        market = load_market(snapshots[year])
        router_rows = load_router(routers[year])
        if len(router_rows) != EXPECTED_RACES_PER_YEAR:
            raise SystemExit(f"router race count regression year={year}: {len(router_rows)} != {EXPECTED_RACES_PER_YEAR}")
        year_items = []
        year_rank_items = []
        year_races = []
        for row in router_rows:
            rid = str(row.get("race_id") or "")
            order = seven_order(row)
            if not order:
                raise ValueError(f"consensus has no horses year={year} race_id={rid}")
            items = []
            for rank, hid in enumerate(order[:10], 1):
                m = market.get((rid, hid), {})
                rec = {
                    "year": year,
                    "race_id": rid,
                    "horse_id": hid,
                    "consensus_rank": rank,
                    "popularity": m.get("popularity"),
                    "odds": m.get("odds"),
                    "finish": m.get("finish"),
                }
                year_rank_items.append(rec)
                all_rank_items.append(rec)
                if rank <= 6:
                    items.append(rec)
                    year_items.append(rec)
                    all_items.append(rec)
            race = {"year": year, "race_id": rid, "items": items}
            year_races.append(race)
            all_races.append(race)
        s = summarize(year_items, year_races)
        s["year"] = year
        year_summaries.append(s)

    overall = summarize(all_items, all_races)
    overall["years"] = EXPECTED_YEARS
    overall["expected_total_races"] = EXPECTED_RACES_PER_YEAR * len(EXPECTED_YEARS)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": "L15_MARKET_DIVERGENCE_V1",
        "scope": "Seven-King consensus Top6 overlap metrics plus consensus ranks 1-10 vs final single-win market, post-hoc evaluation only",
        "odds_used_for_prediction": False,
        "years": EXPECTED_YEARS,
        "overall": overall,
        "by_year": year_summaries,
        "by_consensus_rank": rank_summary(all_rank_items),
        "definitions": {
            "rank_gap": "final_popularity - consensus_rank; positive means consensus rates the horse more highly than the market",
            "top6_overlap": "share of consensus Top6 horses whose final popularity is 1..6",
            "exact_same_top6_set": "all six consensus Top6 horses are final market popularity 1..6",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(out_dir / "rank-summary.csv", "w", newline="", encoding="utf-8") as fh:
        rows = payload["by_consensus_rank"]
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(out_dir / "year-summary.csv", "w", newline="", encoding="utf-8") as fh:
        rows = year_summaries
        keys = ["year"] + [k for k in rows[0].keys() if k != "year"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print("L15_MARKET_DIVERGENCE_V1_OK")
    print(json.dumps(payload["overall"], ensure_ascii=False, separators=(",", ":")))
    print("RANK_SUMMARY")
    for row in payload["by_consensus_rank"]:
        print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))

if __name__ == "__main__":
    main()
