#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import horse_number_map, payout_map
from run_l2_simple_rank_role_v1 import YEARS, load_l17, strategies


def parse_args():
    p = argparse.ArgumentParser(description="L2 market-gap audit: compare fixed L1.7 rank with final market popularity.")
    p.add_argument("--l17-year", action="append", required=True, help="YEAR:PATH")
    p.add_argument("--backfill-root", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def parse_year_paths(items):
    out = {}
    for spec in items:
        y, p = spec.split(":", 1)
        out[int(y)] = Path(p)
    if set(out) != set(YEARS):
        raise SystemExit(f"year path mismatch got={sorted(out)} expected={list(YEARS)}")
    return out


def finite(value):
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def int_pos(value):
    x = finite(value)
    if x is None:
        return None
    i = int(x)
    return i if i > 0 and i == x else None


def market_map(pack):
    out = {}
    for row in pack.get("results") or []:
        hid = str(row.get("horse_id") or "")
        pop = int_pos(row.get("popularity"))
        odds = finite(row.get("win_odds"))
        finish = int_pos(row.get("official_finish_position"))
        if not hid:
            continue
        out[hid] = {
            "popularity": pop,
            "win_odds": odds if odds is not None and odds > 0 else None,
            "finish": finish,
        }
    return out


def gap_bucket(gap):
    if gap <= -4:
        return "NEG_4PLUS"
    if gap <= -2:
        return "NEG_2_3"
    if gap == -1:
        return "NEG_1"
    if gap == 0:
        return "ZERO"
    if gap == 1:
        return "POS_1"
    if gap <= 3:
        return "POS_2_3"
    return "POS_4PLUS"


def gap_direction(gap):
    if gap > 0:
        return "KING_HIGHER"
    if gap < 0:
        return "MARKET_HIGHER"
    return "SAME"


def empty_horse_stat():
    return {
        "horses": 0,
        "wins": 0,
        "top3": 0,
        "stake_yen": 0.0,
        "return_yen": 0.0,
        "sum_popularity": 0.0,
        "sum_gap": 0.0,
        "sum_odds": 0.0,
        "odds_count": 0,
    }


def update_horse(stat, pop, gap, odds, finish, win_return):
    stat["horses"] += 1
    stat["wins"] += int(finish == 1)
    stat["top3"] += int(finish is not None and finish <= 3)
    stat["stake_yen"] += 100.0
    stat["return_yen"] += win_return
    stat["sum_popularity"] += pop
    stat["sum_gap"] += gap
    if odds is not None:
        stat["sum_odds"] += odds
        stat["odds_count"] += 1


def horse_row(year, rank_group, bucket, direction, stat):
    n = stat["horses"]
    stake = stat["stake_yen"]
    return {
        "year": year,
        "rank_group": rank_group,
        "gap_bucket": bucket,
        "gap_direction": direction,
        "horses": n,
        "wins": stat["wins"],
        "win_rate_pct": 100.0 * stat["wins"] / n if n else None,
        "top3": stat["top3"],
        "top3_rate_pct": 100.0 * stat["top3"] / n if n else None,
        "avg_market_popularity": stat["sum_popularity"] / n if n else None,
        "avg_gap": stat["sum_gap"] / n if n else None,
        "avg_win_odds": stat["sum_odds"] / stat["odds_count"] if stat["odds_count"] else None,
        "win_stake_yen": stake,
        "win_return_yen": stat["return_yen"],
        "win_profit_yen": stat["return_yen"] - stake,
        "win_roi_pct": 100.0 * stat["return_yen"] / stake if stake else None,
    }


def empty_strategy_stat():
    return {
        "races": 0,
        "tickets": 0,
        "hit_races": 0,
        "stake_yen": 0.0,
        "return_yen": 0.0,
    }


def update_strategy(stat, tickets, payouts, bet_type):
    stake = 100.0 * len(tickets)
    ret = 0.0
    hit = False
    for nums in tickets:
        p = float(payouts.get((bet_type, nums), 0.0))
        if p > 0:
            hit = True
            ret += p
    stat["races"] += 1
    stat["tickets"] += len(tickets)
    stat["hit_races"] += int(hit)
    stat["stake_yen"] += stake
    stat["return_yen"] += ret


def strategy_row(key, stat):
    year, strategy, role_class, focal_rank, pool_k, bet_type, market_group = key
    races = stat["races"]
    stake = stat["stake_yen"]
    return {
        "year": year,
        "strategy": strategy,
        "role_class": role_class,
        "focal_rank": focal_rank,
        "pool_k": pool_k,
        "bet_type": bet_type,
        "market_group": market_group,
        "races": races,
        "coverage_pct_of_full_13824": 100.0 * races / 13824.0 if year == "ALL" else 100.0 * races / 3456.0,
        "avg_tickets_per_race": stat["tickets"] / races if races else None,
        "hit_rate_pct": 100.0 * stat["hit_races"] / races if races else None,
        "stake_yen": stake,
        "return_yen": stat["return_yen"],
        "profit_yen": stat["return_yen"] - stake,
        "roi_pct": 100.0 * stat["return_yen"] / stake if stake else None,
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    a = parse_args()
    paths = parse_year_paths(a.l17_year)
    l17, l17_counts = load_l17(paths)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horse_stats = defaultdict(empty_horse_stat)
    strat_stats = defaultdict(empty_strategy_stat)
    seen = set()
    market_missing_races = 0
    market_missing_focal = 0
    root = Path(a.backfill_root) / "data" / "daily"
    files_scanned = 0

    for year in YEARS:
        for path in sorted(root.glob(f"{year}-*.jsonl.gz")):
            files_scanned += 1
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    pack = json.loads(line)
                    rid = str((pack.get("race") or {}).get("race_id") or "")
                    rec = l17.get(rid)
                    if rec is None:
                        continue
                    if rid in seen:
                        raise ValueError(f"duplicate race pack race={rid}")
                    if int(rec["year"]) != year:
                        raise ValueError(f"race year drift race={rid}")

                    order = [str(x) for x in rec["consensus_order"]]
                    hno = horse_number_map(pack)
                    market = market_map(pack)
                    payouts, present = payout_map(pack)
                    if not market:
                        market_missing_races += 1
                        continue

                    # Horse-level diagnostic. Market input is popularity/win_odds only.
                    for idx, hid in enumerate(order, 1):
                        m = market.get(hid)
                        if not m or m["popularity"] is None or hid not in hno:
                            continue
                        pop = int(m["popularity"])
                        gap = pop - idx
                        bucket = gap_bucket(gap)
                        direction = gap_direction(gap)
                        finish = m["finish"]
                        odds = m["win_odds"]
                        win_return = float(payouts.get(("WIN", (hno[hid],)), 0.0)) if "WIN" in present else 0.0

                        groups = [
                            ("ALL_RANKS",),
                            (f"R{idx}",),
                        ]
                        if idx <= 3:
                            groups.append(("TOP3",))
                        elif idx <= 6:
                            groups.append(("R4_6",))
                        else:
                            groups.append(("R7PLUS",))

                        for (rank_group,) in groups:
                            for ykey in (year, "ALL"):
                                update_horse(horse_stats[(ykey, rank_group, bucket, direction)], pop, gap, odds, finish, win_return)
                                update_horse(horse_stats[(ykey, rank_group, "ALL_BUCKETS", direction)], pop, gap, odds, finish, win_return)
                                update_horse(horse_stats[(ykey, rank_group, bucket, "ALL_DIRECTIONS")], pop, gap, odds, finish, win_return)

                    trusted_ids = order[:min(6, len(order))]
                    if any(h not in hno for h in trusted_ids):
                        missing = [h for h in trusted_ids if h not in hno]
                        raise ValueError(f"horse-number mapping missing race={rid} horses={missing}")
                    trusted_numbers = [hno[h] for h in trusted_ids]

                    # Apply user's market-gap idea to the existing rank-role strategies.
                    for s in strategies(trusted_numbers):
                        if s["role_class"] not in ("AXIS", "HIMO"):
                            continue
                        focal_rank = int(s["focal_rank"])
                        if focal_rank < 1 or focal_rank > len(order):
                            continue
                        focal_hid = order[focal_rank - 1]
                        m = market.get(focal_hid)
                        if not m or m["popularity"] is None:
                            market_missing_focal += 1
                            continue
                        gap = int(m["popularity"]) - focal_rank
                        direction = gap_direction(gap)
                        bucket = gap_bucket(gap)
                        bet = s["bet_type"]
                        if bet not in present:
                            continue

                        market_groups = ["ALL", direction, bucket]
                        if gap > 0:
                            market_groups.append("BUY_GAP_GT_0")
                        else:
                            market_groups.append("SKIP_GAP_LE_0")

                        for market_group in market_groups:
                            for ykey in (year, "ALL"):
                                key = (
                                    ykey, s["strategy"], s["role_class"], focal_rank,
                                    s["pool_k"], bet, market_group
                                )
                                update_strategy(strat_stats[key], s["tickets"], payouts, bet)
                    seen.add(rid)

    missing = set(l17) - seen
    if missing:
        raise SystemExit(f"missing race packs/market rows count={len(missing)} sample={sorted(missing)[:20]}")

    horse_rows = [horse_row(*k, v) for k, v in horse_stats.items()]
    horse_rows.sort(key=lambda r: (str(r["year"]), r["rank_group"], r["gap_direction"], r["gap_bucket"]))

    strat_rows = [strategy_row(k, v) for k, v in strat_stats.items()]
    strat_rows.sort(key=lambda r: (
        str(r["year"]), r["strategy"], r["market_group"]
    ))
    combined = [r for r in strat_rows if r["year"] == "ALL"]
    yearly = [r for r in strat_rows if r["year"] != "ALL"]

    # Direct baseline vs user's buy rule.
    by_strategy = defaultdict(dict)
    for r in combined:
        by_strategy[r["strategy"]][r["market_group"]] = r
    buy_compare = []
    for strategy, d in sorted(by_strategy.items()):
        base = d.get("ALL")
        buy = d.get("BUY_GAP_GT_0")
        skip = d.get("SKIP_GAP_LE_0")
        if not base or not buy:
            continue
        buy_compare.append({
            "strategy": strategy,
            "role_class": base["role_class"],
            "focal_rank": base["focal_rank"],
            "pool_k": base["pool_k"],
            "bet_type": base["bet_type"],
            "baseline_races": base["races"],
            "baseline_roi_pct": base["roi_pct"],
            "buy_gap_gt0_races": buy["races"],
            "buy_gap_gt0_coverage_pct": 100.0 * buy["races"] / base["races"] if base["races"] else None,
            "buy_gap_gt0_hit_rate_pct": buy["hit_rate_pct"],
            "buy_gap_gt0_roi_pct": buy["roi_pct"],
            "roi_delta_pp": buy["roi_pct"] - base["roi_pct"],
            "buy_gap_gt0_profit_yen": buy["profit_yen"],
            "skip_gap_le0_races": skip["races"] if skip else 0,
            "skip_gap_le0_roi_pct": skip["roi_pct"] if skip else None,
        })

    # Navigation only: highest ROI delta, not a production winner.
    buy_compare.sort(key=lambda r: (r["roi_delta_pp"], r["buy_gap_gt0_races"]), reverse=True)

    write_csv(out_dir / "horse-gap-summary.csv", horse_rows)
    write_csv(out_dir / "strategy-gap-combined.csv", combined)
    write_csv(out_dir / "strategy-gap-yearly.csv", yearly)
    write_csv(out_dir / "buy-rule-comparison.csv", buy_compare)

    summary = {
        "contract": "L2_MARKET_GAP_V1",
        "idea": "Keep L1.7 ranking fixed. Buy-side condition is market popularity rank worse than king rank: popularity - king_rank > 0.",
        "years": list(YEARS),
        "l17_races_by_year": l17_counts,
        "races_processed": len(seen),
        "daily_files_scanned": files_scanned,
        "market_source": "BACKFILL result rows: final popularity and win_odds, keyed by horse_id",
        "selection_inputs": ["L1.7 consensus rank", "final market popularity rank"],
        "outcome_or_payout_used_for_selection": False,
        "learning": False,
        "rerank_l1": False,
        "gap_definition": "market_popularity_rank - l17_consensus_rank",
        "buy_rule": "gap > 0",
        "skip_rule": "gap <= 0",
        "gap_buckets": ["NEG_4PLUS", "NEG_2_3", "NEG_1", "ZERO", "POS_1", "POS_2_3", "POS_4PLUS"],
        "market_missing_races": market_missing_races,
        "market_missing_focal_occurrences": market_missing_focal,
        "2026_locked": True,
        "production_promotion": False,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    readme = """# L2 Market Gap V1

Purpose: test the user's simple value hypothesis without re-ranking the Seven-King output.

- gap = final market popularity rank - L1.7 consensus rank
- gap > 0: Seven-King rates the horse higher than the market (BUY side)
- gap = 0: same rank
- gap < 0: market rates the horse higher (SKIP side)
- no learned threshold
- no Gate / Outsider / router
- no outcome or payout is used to decide whether to buy
- payout is evaluation only
- 2026 remains sealed
- outputs are diagnostics, not a production promotion decision
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print("L2_MARKET_GAP_V1_DONE")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
