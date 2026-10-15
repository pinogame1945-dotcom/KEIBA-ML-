#!/usr/bin/env python3
"""All-race, OOS-only market vs ANCHOR_FULL outcome and winner-rank audit."""
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

YEARS = (2023, 2024, 2025)
N_PER_YEAR = 3456
N_TOTAL = 10368
PROB = "P__ANCHOR_FULL"
CASES = ("MARKET_ONLY", "AI_ONLY", "BOTH_HIT", "BOTH_MISS")
RANK_COLS = {
    "MARKET_RANK": "market_rank",
    "AI_PROB_RANK": "ai_prob_rank",
    "AI_VALUE_RANK": "ai_value_rank",
}
REQUIRED = [
    "year", "_race_id", "_horse_id", "target_win", "final_win_odds",
    "market_rank", PROB,
]
BASELINE_WINS = {
    2023: (1143, 753),
    2024: (1199, 575),
    2025: (1146, 589),
}

def check(condition, message):
    if not condition:
        raise ValueError(message)

def prepare(df):
    check(not df["year"].eq(2026).any(), "2026 sealed")
    check(set(df["year"].unique()) == set(YEARS), "unexpected/missing years")
    check(not df[REQUIRED].isna().any().any(), "missing input cells")
    check(not df.duplicated(["_race_id", "_horse_id"]).any(), "duplicate horse-race")
    check(df[["final_win_odds", PROB, "market_rank"]].apply(
        lambda x: np.isfinite(x.to_numpy(dtype=float)).all()).all(), "nonfinite inputs")
    check(df["final_win_odds"].gt(0).all(), "nonpositive final odds")
    check(df[PROB].between(0, 1).all(), "probability out of range")
    check(df["target_win"].isin([0, 1]).all(), "nonbinary win target")
    grp = df.groupby("_race_id", sort=False)
    check(df["_race_id"].nunique() == N_TOTAL, "race coverage mismatch")
    wins_per_race = grp["target_win"].sum()
    print("WINNER_COUNT_DISTRIBUTION", json.dumps(wins_per_race.value_counts().sort_index().to_dict()), flush=True)
    print("NON_SINGLE_WINNERS_EXAMPLES", wins_per_race[wins_per_race.ne(1)].head(15).to_dict(), flush=True)
    check(wins_per_race.isin([1, 2]).all(), "zero winner or >2 winners")
    check(grp["year"].nunique().eq(1).all(), "race spanning years")
    for y in YEARS:
        check(df.loc[df.year.eq(y), "_race_id"].nunique() == N_PER_YEAR,
              f"year {y}: race coverage mismatch")
    return df

def process_year(y, d):
    """No model fitting: all three years share the single loaded input frame."""
    q = d.copy()
    q["market_rank"] = pd.to_numeric(q["market_rank"], errors="raise")
    q["__value"] = q[PROB] * q["final_win_odds"]
    # Stable, precommitted tie rules. AI rank means probability rank, NOT EV rank.
    m = q.sort_values(["_race_id", "market_rank", "final_win_odds", "_horse_id"],
                      ascending=[True, True, True, True])
    a = q.sort_values(["_race_id", "__value", PROB, "market_rank", "_horse_id"],
                      ascending=[True, False, False, True, True])
    p = q.sort_values(["_race_id", PROB, "market_rank", "_horse_id"],
                      ascending=[True, False, True, True])
    q.loc[p.index, "ai_prob_rank"] = p.groupby("_race_id", sort=False).cumcount().to_numpy() + 1
    q.loc[a.index, "ai_value_rank"] = a.groupby("_race_id", sort=False).cumcount().to_numpy() + 1
    check(q["ai_prob_rank"].notna().all() and q["ai_value_rank"].notna().all(), "rank missing")
    market = m.drop_duplicates("_race_id").set_index("_race_id")
    ai = a.drop_duplicates("_race_id").set_index("_race_id")
    market_win = int(market["target_win"].sum())
    ai_win = int(ai["target_win"].sum())
    check((market_win, ai_win) == BASELINE_WINS[y],
          f"{y} baseline wins changed: {market_win}, {ai_win}")
    race = pd.DataFrame({
        "_race_id": market.index.to_numpy(),
        "year": y,
        "market_pick_horse": market["_horse_id"].to_numpy(),
        "ai_pick_horse": ai["_horse_id"].reindex(market.index).to_numpy(),
        "market_hit": market["target_win"].eq(1).to_numpy(),
        "ai_hit": ai["target_win"].reindex(market.index).eq(1).to_numpy(),
        "field_size": q.groupby("_race_id").size().reindex(market.index).to_numpy(),
        "winner_horses_count": q.groupby("_race_id")["target_win"].sum().reindex(market.index).to_numpy(),
    })
    conditions = [
        race["market_hit"] & ~race["ai_hit"],
        ~race["market_hit"] & race["ai_hit"],
        race["market_hit"] & race["ai_hit"],
        ~race["market_hit"] & ~race["ai_hit"],
    ]
    race["outcome"] = np.select(conditions, CASES, default="ERROR")
    check(len(race) == N_PER_YEAR and not race["outcome"].eq("ERROR").any(),
          "invalid race partition")
    check(race.loc[race["market_hit"] & race["ai_hit"] &
          race["market_pick_horse"].ne(race["ai_pick_horse"]),
          "winner_horses_count"].eq(2).all(), "both-hit different horses without dead heat")
    indexed = race.set_index("_race_id")
    w = q[q["target_win"].eq(1)].copy()
    check(w["_race_id"].nunique() == N_PER_YEAR, f"{y} missing winners")
    check(len(w) == int(race["winner_horses_count"].sum()), f"{y} winner-horse count")
    for c in ("market_pick_horse", "ai_pick_horse", "market_hit", "ai_hit",
              "outcome", "field_size", "winner_horses_count"):
        w[c] = w["_race_id"].map(indexed[c])
    w["year"] = y
    w["winner_final_odds"] = w["final_win_odds"]
    w["winner_ai_win_probability"] = w[PROB]
    w["market_pick_final_odds"] = w["_race_id"].map(market["final_win_odds"])
    w["ai_pick_final_odds"] = w["_race_id"].map(ai["final_win_odds"])
    return (race, w[[
        "year", "_race_id", "_horse_id", "outcome", "market_hit", "ai_hit",
        "market_pick_horse", "ai_pick_horse", "field_size", "winner_horses_count",
        "market_rank", "ai_prob_rank", "ai_value_rank", "winner_final_odds",
        "winner_ai_win_probability", "market_pick_final_odds", "ai_pick_final_odds"
    ]])


def rows_for_view(year, races, winners):
    case_rows, dist_rows, cross_rows, topk_rows = [], [], [], []
    upper = int(races["field_size"].max())
    for outcome in ("ALL", *CASES):
        r = races if outcome == "ALL" else races[races["outcome"].eq(outcome)]
        z = winners if outcome == "ALL" else winners[winners["outcome"].eq(outcome)]
        n_r, n_w = int(len(r)), int(len(z))
        check(int(r["winner_horses_count"].sum()) == n_w,
              f"{year}/{outcome}: winner-horse count discrepancy")
        case_rows.append({
            "year": year, "outcome": outcome, "races": n_r,
            "pct_of_all_races_in_year": 100 * n_r / len(races),
            "winning_horses": n_w,
            "dead_heat_races": int(r["winner_horses_count"].eq(2).sum()),
            "market_hits": int(r["market_hit"].sum()),
            "ai_hits": int(r["ai_hit"].sum()),
            "both_pick_same_horse": int(r["market_pick_horse"].eq(r["ai_pick_horse"]).sum()),
        })
        for label, col in RANK_COLS.items():
            v = z[col].astype(int)
            counts = v.value_counts()
            for rank in range(1, upper + 1):
                hits = int(counts.get(rank, 0))
                dist_rows.append({
                    "year": year, "outcome": outcome, "ranking": label,
                    "rank": rank, "winning_horses": hits,
                    "pct_within_outcome_winning_horses": 100 * hits / n_w if n_w else 0.0,
                    "pct_of_all_winning_horses": 100 * hits / len(winners),
                })
            best = z.groupby("_race_id")[col].min()
            for k in (1, 2, 3, 6, 10):
                captured_horses = int(v.le(k).sum())
                captured_races = int(best.le(k).sum())
                topk_rows.append({
                    "year": year, "outcome": outcome, "ranking": label,
                    "top_k": k, "captured_winning_horses": captured_horses,
                    "captured_races_any_winner": captured_races,
                    "race_capture_pct_within_outcome": 100 * captured_races / n_r if n_r else 0.0,
                    "horse_capture_pct_within_outcome": 100 * captured_horses / n_w if n_w else 0.0,
                })
        pair_counts = z.groupby(["market_rank", "ai_prob_rank"], sort=True).size()
        for (m, a), count in pair_counts.items():
            cross_rows.append({
                "year": year, "outcome": outcome,
                "winner_market_rank": int(m), "winner_ai_prob_rank": int(a),
                "winning_horses": int(count),
                "pct_within_outcome_winning_horses": 100 * count / n_w if n_w else 0.0,
            })
    return case_rows, dist_rows, cross_rows, topk_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions, compression="infer", usecols=REQUIRED,
                     dtype={"_race_id": str, "_horse_id": str}, low_memory=False)
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["target_win"] = pd.to_numeric(df["target_win"], errors="raise")
    df["final_win_odds"] = pd.to_numeric(df["final_win_odds"], errors="raise")
    df[PROB] = pd.to_numeric(df[PROB], errors="raise")
    df["market_rank"] = pd.to_numeric(df["market_rank"], errors="raise")
    prepare(df)
    # Bounded CPU parallelism by year. No additional data fetch or cache.
    max_workers = min(3, max(1, os.cpu_count() or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        per_year = list(executor.map(
            lambda y: process_year(y, df[df["year"].eq(y)]), YEARS))
    races = pd.concat([x[0] for x in per_year], ignore_index=True)
    winners = pd.concat([x[1] for x in per_year], ignore_index=True)
    check(len(races) == N_TOTAL and races["_race_id"].is_unique, "pooled race coverage")
    check(winners["_race_id"].nunique() == N_TOTAL, "pooled winner coverage")
    check(int(races["winner_horses_count"].sum()) == len(winners),
          "pooled dead heat count mismatch")
    check(int(races["market_hit"].sum()) == 3488, "pooled favorite win mismatch")
    check(int(races["ai_hit"].sum()) == 1917, "pooled AI win mismatch")
    case_rows, dist_rows, cross_rows, topk_rows = [], [], [], []
    for year, rr, ww in [
        *((str(y), races[races["year"].eq(y)], winners[winners["year"].eq(y)]) for y in YEARS),
        ("2023-2025", races, winners),
    ]:
        a, b, c, d = rows_for_view(year, rr, ww)
        case_rows.extend(a)
        dist_rows.extend(b)
        cross_rows.extend(c)
        topk_rows.extend(d)
    case_df = pd.DataFrame(case_rows)
    counts = case_df[case_df["year"].eq("2023-2025")].set_index("outcome")["races"]
    check(sum(int(counts[c]) for c in CASES) == N_TOTAL, "bad 4-class partition")
    check(int(counts["MARKET_ONLY"] + counts["BOTH_HIT"]) == 3488, "market class mismatch")
    check(int(counts["AI_ONLY"] + counts["BOTH_HIT"]) == 1917, "AI class mismatch")
    case_df.to_csv(out / "four-outcome-counts.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(out / "winner-rank-distributions.csv", index=False)
    pd.DataFrame(cross_rows).to_csv(out / "winner-market-vs-ai-rank.csv", index=False)
    pd.DataFrame(topk_rows).to_csv(out / "winner-topk-capture.csv", index=False)
    races.to_csv(out / "all-race-outcome-diagnostics.csv.gz", index=False, compression="gzip")
    winners.to_csv(out / "all-race-winner-diagnostics.csv.gz",
                   index=False, compression="gzip")
    summary = {
        "contract": "ALL_RACE_WINRANK_OUTCOMES_V1",
        "source": "MARKET_FIXED_ANCHOR_RESIDUAL_V2 OOS predictions, reused without model fitting",
        "benchmark": "ALL_RACE_ONE_PICK_V1 Run 38668174639 (exact original selectors)",
        "years": list(YEARS),
        "races": N_TOTAL,
        "winning_horses": int(len(winners)),
        "dead_heat_races": int(races["winner_horses_count"].eq(2).sum()),
        "no_skip": True,
        "winner_market_rank": "market_rank as supplied in OOS data (odds popularity rank)",
        "winner_ai_prob_rank": "P__ANCHOR_FULL descending, ties by market_rank then horse_id",
        "winner_ai_value_rank": "P__ANCHOR_FULL * final_win_odds descending (purchase selection rank)",
        "market_favorite_rule": "minimum market_rank, minimum final_win_odds, horse_id",
        "ai_pick_rule": "maximum P__ANCHOR_FULL * final_win_odds; ties by probability, market_rank, horse_id",
        "four_outcomes": {c: int(counts[c]) for c in CASES},
        "winner_rank_denominator": "winning horses (dead heats count as two); outcome denominator is races",
        "topk_race_capture": "counts races with at least one real winner in the top K",
        "market_hits": 3488, "ai_hits": 1917,
        "note": "Retrospective outcome and ranking audit only; no new betting strategy or profitable switch inferred.",
        "odds_caveat": "final_win_odds are after-race closing odds, not necessarily wager-time observable",
        "locked_year": 2026,
        "parallel_workers": max_workers,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=== ALL-RACE 4-WAY OUTCOMES (DEAD HEATS PRESERVED) ===")
    print(case_df.to_csv(index=False), flush=True)
    print("=== POOLED WINNER TOP-K RANK CAPTURE ===")
    t = pd.DataFrame(topk_rows)
    print(t[(t["year"].eq("2023-2025")) & (t["outcome"].eq("ALL"))]
          .to_csv(index=False), flush=True)
    print("ALL_RACE_WINRANK_OUTCOMES_V1_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
