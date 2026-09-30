#!/usr/bin/env python3
import argparse
import csv
import gzip
import itertools
import json
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import canonical_numbers, horse_number_map, payout_map

YEARS = (2022, 2023, 2024, 2025)
POOLS = (3, 6)
FOCAL_RANKS = (1, 2, 3)

BET_ARITY = {
    "WIN": 1,
    "QUINELLA": 2,
    "EXACTA": 2,
    "TRIO": 3,
    "TRIFECTA": 3,
}


def parse_args():
    p = argparse.ArgumentParser(description="Simple L2 rank-role audit: trust L1.7 ranks 1-3 and only test axis/himo usage.")
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


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "rt", encoding="utf-8")


def load_l17(paths):
    out = {}
    counts = {}
    for year in YEARS:
        n = 0
        with open_text(paths[year]) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("contract") != "L17_SEVEN_KING_FULLFIELD_OUTPUT_V1":
                    raise ValueError(f"bad L1.7 contract year={year}")
                if int(r.get("year")) != year:
                    raise ValueError(f"L1.7 year drift expected={year} got={r.get('year')}")
                rid = str(r.get("race_id") or "")
                if not rid or rid in out:
                    raise ValueError(f"bad/duplicate race_id={rid}")
                order = [str(x) for x in (r.get("consensus_order") or [])]
                horses = r.get("horses") or []
                if len(order) != len(horses) or len(order) < 6:
                    raise ValueError(f"bad L1.7 field race={rid} order={len(order)} horses={len(horses)}")
                if len(set(order)) != len(order):
                    raise ValueError(f"duplicate horse in L1.7 race={rid}")
                out[rid] = r
                n += 1
        counts[year] = n
    return out, counts


def uniq_tickets(bet_type, rows):
    seen = set()
    out = []
    arity = BET_ARITY[bet_type]
    for row in rows:
        vals = tuple(int(x) for x in row)
        if len(vals) != arity or len(set(vals)) != len(vals):
            continue
        key = canonical_numbers(bet_type, vals)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def strategies(top_numbers):
    out = []
    for pool_k in POOLS:
        pool = list(top_numbers[:pool_k])
        if len(pool) < pool_k:
            continue

        # Core-only baselines: no learning, no gate, no market filter.
        out.extend([
            {
                "strategy": f"CORE_TOP{pool_k}_QUINELLA_BOX",
                "role_class": "CORE",
                "focal_rank": 0,
                "pool_k": pool_k,
                "bet_type": "QUINELLA",
                "tickets": uniq_tickets("QUINELLA", itertools.combinations(pool, 2)),
            },
            {
                "strategy": f"CORE_TOP{pool_k}_EXACTA_BOX",
                "role_class": "CORE",
                "focal_rank": 0,
                "pool_k": pool_k,
                "bet_type": "EXACTA",
                "tickets": uniq_tickets("EXACTA", itertools.permutations(pool, 2)),
            },
            {
                "strategy": f"CORE_TOP{pool_k}_TRIO_BOX",
                "role_class": "CORE",
                "focal_rank": 0,
                "pool_k": pool_k,
                "bet_type": "TRIO",
                "tickets": uniq_tickets("TRIO", itertools.combinations(pool, 3)),
            },
            {
                "strategy": f"CORE_TOP{pool_k}_TRIFECTA_BOX",
                "role_class": "CORE",
                "focal_rank": 0,
                "pool_k": pool_k,
                "bet_type": "TRIFECTA",
                "tickets": uniq_tickets("TRIFECTA", itertools.permutations(pool, 3)),
            },
        ])

        for focal_rank in FOCAL_RANKS:
            focal = top_numbers[focal_rank - 1]
            others = [x for x in pool if x != focal]

            # AXIS = focal horse is fixed; the remaining legs come from the same TopK pool.
            axis_defs = [
                ("WIN_SINGLE", "WIN", [(focal,)]),
                ("QUINELLA_AXIS", "QUINELLA", [(focal, x) for x in others]),
                ("EXACTA_AXIS_FIRST", "EXACTA", [(focal, x) for x in others]),
                ("EXACTA_AXIS_SECOND", "EXACTA", [(x, focal) for x in others]),
                ("TRIO_AXIS", "TRIO", [(focal, a, b) for a, b in itertools.combinations(others, 2)]),
                ("TRIFECTA_AXIS_FIRST", "TRIFECTA", [(focal, a, b) for a, b in itertools.permutations(others, 2)]),
                ("TRIFECTA_AXIS_SECOND", "TRIFECTA", [(a, focal, b) for a, b in itertools.permutations(others, 2)]),
                ("TRIFECTA_AXIS_THIRD", "TRIFECTA", [(a, b, focal) for a, b in itertools.permutations(others, 2)]),
            ]
            for label, bet, rows in axis_defs:
                tickets = uniq_tickets(bet, rows)
                if tickets:
                    out.append({
                        "strategy": f"R{focal_rank}_{label}_TOP{pool_k}",
                        "role_class": "AXIS",
                        "focal_rank": focal_rank,
                        "pool_k": pool_k,
                        "bet_type": bet,
                        "tickets": tickets,
                    })

            # HIMO = focal horse is mandatory alongside the best-ranked other Top3 horse,
            # which acts as the anchor. This keeps the definition deterministic for R1/R2/R3.
            anchor_rank = 2 if focal_rank == 1 else 1
            anchor = top_numbers[anchor_rank - 1]
            mates = [x for x in pool if x not in {anchor, focal}]
            himo_defs = [
                ("QUINELLA_HIMO_BEST_OTHER", "QUINELLA", [(anchor, focal)]),
                ("EXACTA_HIMO_BEST_OTHER_FIRST", "EXACTA", [(anchor, focal)]),
                ("TRIO_HIMO_BEST_OTHER", "TRIO", [(anchor, focal, x) for x in mates]),
                ("TRIFECTA_HIMO_BEST_OTHER_FIRST", "TRIFECTA",
                 [(anchor, focal, x) for x in mates] + [(anchor, x, focal) for x in mates]),
            ]
            for label, bet, rows in himo_defs:
                tickets = uniq_tickets(bet, rows)
                if tickets:
                    out.append({
                        "strategy": f"R{focal_rank}_{label}_TOP{pool_k}",
                        "role_class": "HIMO",
                        "focal_rank": focal_rank,
                        "pool_k": pool_k,
                        "bet_type": bet,
                        "tickets": tickets,
                    })
    return out


def empty_stat():
    return {
        "races": 0,
        "tickets": 0,
        "hit_races": 0,
        "winning_tickets": 0,
        "positive_profit_races": 0,
        "stake_yen": 0.0,
        "return_yen": 0.0,
    }


def update(stat, tickets, payouts, bet_type):
    stake = 100.0 * len(tickets)
    ret = 0.0
    wins = 0
    for nums in tickets:
        p = float(payouts.get((bet_type, nums), 0.0))
        if p > 0:
            wins += 1
            ret += p
    stat["races"] += 1
    stat["tickets"] += len(tickets)
    stat["hit_races"] += int(wins > 0)
    stat["winning_tickets"] += wins
    stat["positive_profit_races"] += int(ret > stake)
    stat["stake_yen"] += stake
    stat["return_yen"] += ret


def finalize_row(key, stat):
    year, strategy, role_class, focal_rank, pool_k, bet_type = key
    races = stat["races"]
    stake = stat["stake_yen"]
    ret = stat["return_yen"]
    return {
        "year": year,
        "strategy": strategy,
        "role_class": role_class,
        "focal_rank": focal_rank,
        "pool_k": pool_k,
        "bet_type": bet_type,
        "races": races,
        "tickets": stat["tickets"],
        "avg_tickets_per_race": stat["tickets"] / races if races else None,
        "hit_races": stat["hit_races"],
        "hit_rate_pct": 100.0 * stat["hit_races"] / races if races else None,
        "winning_tickets": stat["winning_tickets"],
        "positive_profit_races": stat["positive_profit_races"],
        "positive_profit_race_pct": 100.0 * stat["positive_profit_races"] / races if races else None,
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi_pct": 100.0 * ret / stake if stake else None,
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

    stats = defaultdict(empty_stat)
    seen = set()
    files_scanned = 0
    root = Path(a.backfill_root) / "data" / "daily"

    for year in YEARS:
        for path in sorted(root.glob(f"{year}-*.jsonl.gz")):
            files_scanned += 1
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    pack = json.loads(line)
                    rid = str((pack.get("race") or {}).get("race_id") or "")
                    if rid not in l17:
                        continue
                    if rid in seen:
                        raise ValueError(f"duplicate race pack race={rid}")
                    rec = l17[rid]
                    rec_year = int(rec["year"])
                    if rec_year != year:
                        raise ValueError(f"race year drift race={rid} l17={rec_year} file_year={year}")

                    hno = horse_number_map(pack)
                    order = [str(x) for x in rec["consensus_order"]]
                    trusted_ids = order[:min(6, len(order))]
                    if any(h not in hno for h in trusted_ids):
                        missing = [h for h in trusted_ids if h not in hno]
                        raise ValueError(f"horse-number mapping missing race={rid} horses={missing}")
                    trusted_numbers = [hno[h] for h in trusted_ids]
                    if len(set(trusted_numbers)) != len(trusted_numbers):
                        raise ValueError(f"duplicate horse number in trusted pool race={rid} nums={trusted_numbers}")

                    payouts, present = payout_map(pack)
                    for s in strategies(trusted_numbers):
                        bet = s["bet_type"]
                        if bet not in present:
                            continue
                        tickets = s["tickets"]
                        key_year = (
                            year, s["strategy"], s["role_class"], s["focal_rank"], s["pool_k"], bet
                        )
                        key_all = (
                            "ALL", s["strategy"], s["role_class"], s["focal_rank"], s["pool_k"], bet
                        )
                        update(stats[key_year], tickets, payouts, bet)
                        update(stats[key_all], tickets, payouts, bet)
                    seen.add(rid)

    missing = set(l17) - seen
    if missing:
        raise SystemExit(f"missing race packs count={len(missing)} sample={sorted(missing)[:20]}")

    rows = [finalize_row(k, v) for k, v in stats.items()]
    rows.sort(key=lambda r: (
        str(r["year"]), r["role_class"], r["focal_rank"], r["bet_type"], r["pool_k"], r["strategy"]
    ))

    combined = [r for r in rows if r["year"] == "ALL"]
    yearly = [r for r in rows if r["year"] != "ALL"]

    # Direct AXIS vs HIMO comparison on the same focal rank / pool / bet family.
    role_summary = []
    groups = defaultdict(lambda: {"AXIS": [], "HIMO": []})
    for r in combined:
        if r["role_class"] in ("AXIS", "HIMO"):
            groups[(r["focal_rank"], r["pool_k"], r["bet_type"])][r["role_class"]].append(r)
    for (rank, pool_k, bet), d in sorted(groups.items()):
        for role in ("AXIS", "HIMO"):
            for r in d[role]:
                role_summary.append({
                    "focal_rank": rank,
                    "pool_k": pool_k,
                    "bet_type": bet,
                    "role_class": role,
                    "strategy": r["strategy"],
                    "races": r["races"],
                    "avg_tickets_per_race": r["avg_tickets_per_race"],
                    "hit_rate_pct": r["hit_rate_pct"],
                    "positive_profit_race_pct": r["positive_profit_race_pct"],
                    "roi_pct": r["roi_pct"],
                    "profit_yen": r["profit_yen"],
                })

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "metrics-yearly.csv", yearly)
    write_csv(out / "metrics-combined.csv", combined)
    write_csv(out / "role-comparison.csv", role_summary)

    # Small navigation table: not a selected policy, just highest combined ROI per rank/role/bet.
    best = {}
    for r in role_summary:
        k = (r["focal_rank"], r["role_class"], r["bet_type"])
        cur = best.get(k)
        if cur is None or float(r["roi_pct"]) > float(cur["roi_pct"]):
            best[k] = r
    best_rows = [best[k] for k in sorted(best)]
    write_csv(out / "best-per-rank-role-bet.csv", best_rows)

    summary = {
        "contract": "L2_SIMPLE_RANK_ROLE_V1",
        "idea": "Trust L1.7 ranks 1-3; do not re-rank. Only test deterministic AXIS/HIMO ticket roles.",
        "years": list(YEARS),
        "l17_races_by_year": l17_counts,
        "races_processed": len(seen),
        "daily_files_scanned": files_scanned,
        "focal_ranks": list(FOCAL_RANKS),
        "pools": list(POOLS),
        "stake_per_ticket_yen": 100,
        "learning": False,
        "race_filtering": False,
        "gate": False,
        "outsider": False,
        "router": False,
        "odds_used_for_selection": False,
        "popularity_used_for_selection": False,
        "payout_used_for_selection": False,
        "payout_used_for_evaluation_only": True,
        "2026_locked": True,
        "production_promotion": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# L2 Simple Rank-Role V1\n\n"
        "No model is trained here. L1.7 Seven-King full-field consensus is accepted as-is. "
        "Only ranks 1-3 are treated as focal horses, then deterministic AXIS/HIMO usage is evaluated "
        "with Top3 and Top6 partner pools. Every eligible race is evaluated; there is no gate, outsider, "
        "router, market filter, popularity feature, or race skipping. Payouts are used only after ticket "
        "construction for historical ROI measurement. 2026 remains sealed.\n",
        encoding="utf-8",
    )

    print("L2_SIMPLE_RANK_ROLE_V1_READY", flush=True)
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
