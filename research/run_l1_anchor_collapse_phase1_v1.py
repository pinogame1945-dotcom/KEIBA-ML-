#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

VALID_YEARS = (2023, 2024, 2025)

def parse_args():
    p = argparse.ArgumentParser(description="Phase-1 census for triple-agreement anchor collapse.")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--market-scored", required=True)
    p.add_argument("--outsider-predictions", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()

def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def odds_band(x):
    x = float(x)
    if x <= 1.5:
        return "LE_1.5"
    if x <= 2.0:
        return "GT_1.5_LE_2.0"
    if x <= 3.0:
        return "GT_2.0_LE_3.0"
    if x <= 5.0:
        return "GT_3.0_LE_5.0"
    return "GT_5.0"

def summarize(name, q):
    n = len(q)
    collapse = int(q["collapse"].sum()) if n else 0
    survive = n - collapse
    return {
        "scope": name,
        "anchors": n,
        "survive_top3": survive,
        "collapse_outside_top3": collapse,
        "collapse_rate_pct": 100.0 * collapse / n if n else None,
        "mean_final_win_odds": float(q["final_win_odds"].mean()) if n else None,
        "median_final_win_odds": float(q["final_win_odds"].median()) if n else None,
        "mean_outsider_p3_delta": float(q["p3_delta"].mean()) if n else None,
        "median_outsider_p3_delta": float(q["p3_delta"].median()) if n else None,
    }

def main():
    a = parse_args()
    if a.year not in VALID_YEARS:
        raise SystemExit(f"unsupported year={a.year}; valid={VALID_YEARS}")
    if a.year == 2026:
        raise SystemExit("2026 sealed")

    market_cols = [
        "year", "race_id", "horse_id", "horse_number", "consensus_rank",
        "market_rank", "final_win_odds", "target_top3"
    ]
    outsider_cols = [
        "year", "race_id", "horse_id", "rank", "score", "p3",
        "p3_rank_baseline", "p3_delta"
    ]
    m = pd.read_csv(a.market_scored, compression="gzip", usecols=market_cols)
    p = pd.read_csv(a.outsider_predictions, compression="gzip", usecols=outsider_cols)

    m["year"] = pd.to_numeric(m["year"], errors="raise").astype(int)
    p["year"] = pd.to_numeric(p["year"], errors="raise").astype(int)
    if 2026 in set(m["year"]) or 2026 in set(p["year"]):
        raise SystemExit("2026 sealed")

    m = m[m["year"] == a.year].copy()
    p = p[p["year"] == a.year].copy()
    for c in ("consensus_rank", "market_rank", "final_win_odds", "target_top3"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    for c in ("rank", "score", "p3", "p3_rank_baseline", "p3_delta"):
        p[c] = pd.to_numeric(p[c], errors="coerce")

    keys = ["year", "race_id", "horse_id"]
    if m.duplicated(keys).any():
        raise SystemExit("duplicate market keys")
    if p.duplicated(keys).any():
        raise SystemExit("duplicate outsider keys")

    z = m.merge(p, on=keys, how="inner", validate="one_to_one", suffixes=("", "_outsider"))
    if z.empty:
        raise SystemExit("empty join")

    z["king_rank_mismatch"] = (z["consensus_rank"].astype(int) != z["rank"].astype(int)).astype(int)
    z["collapse"] = (z["target_top3"].astype(int) == 0).astype(int)

    # User definition: Seven-King, Outsider, and market all point at the same anchor.
    # Outsider is signal-only; p3_delta > 0 means it reinforces the historical King rank.
    anchors = z[
        (z["consensus_rank"] == 1)
        & (z["market_rank"] == 1)
        & (z["rank"] == 1)
        & (z["p3_delta"] > 0)
        & z["target_top3"].notna()
        & z["final_win_odds"].notna()
    ].copy()

    if anchors.empty:
        raise SystemExit(f"no triple-agreement anchors year={a.year}")

    anchors["odds_band"] = [odds_band(x) for x in anchors["final_win_odds"]]

    # Outcome-free support quantiles are descriptive only. They are never optimized on collapse labels.
    anchors["outsider_support_quantile"] = pd.qcut(
        anchors["p3_delta"].rank(method="first"),
        q=min(4, len(anchors)),
        labels=False,
        duplicates="drop",
    )
    anchors["outsider_support_quantile"] = anchors["outsider_support_quantile"].astype(int) + 1

    headline = [summarize("ALL_TRIPLE_AGREE", anchors)]
    for name, threshold in (("MARKET_ODDS_LE_3.0", 3.0), ("MARKET_ODDS_LE_2.0", 2.0), ("MARKET_ODDS_LE_1.5", 1.5)):
        headline.append(summarize(name, anchors[anchors["final_win_odds"] <= threshold]))

    odds_rows = []
    for band, q in anchors.groupby("odds_band", sort=False):
        r = summarize(str(band), q)
        r["odds_band"] = str(band)
        odds_rows.append(r)

    support_rows = []
    for bucket, q in anchors.groupby("outsider_support_quantile", sort=True):
        r = summarize(f"OUTSIDER_Q{int(bucket)}", q)
        r["outsider_support_quantile"] = int(bucket)
        r["p3_delta_min"] = float(q["p3_delta"].min())
        r["p3_delta_max"] = float(q["p3_delta"].max())
        support_rows.append(r)

    collapse_cols = [
        "year", "race_id", "horse_id", "horse_number", "consensus_rank", "market_rank",
        "rank", "final_win_odds", "score", "p3", "p3_rank_baseline", "p3_delta",
        "odds_band", "outsider_support_quantile", "target_top3", "collapse",
        "king_rank_mismatch"
    ]
    collapse_cases = anchors[anchors["collapse"] == 1][collapse_cols].sort_values(
        ["final_win_odds", "p3_delta"], ascending=[True, False]
    )
    anchors_out = anchors[collapse_cols].sort_values(["race_id", "horse_number"])

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "headline.csv", headline)
    write_csv(out / "odds-bands.csv", odds_rows)
    write_csv(out / "outsider-support-quantiles.csv", support_rows)
    collapse_cases.to_csv(out / "collapse-cases.csv", index=False)
    anchors_out.to_csv(out / "anchors.csv.gz", index=False, compression="gzip")

    summary = {
        "contract": "L1_ANCHOR_COLLAPSE_PHASE1_V1",
        "year": a.year,
        "definition": {
            "seven_king": "consensus_rank == 1 and frozen Outsider rank == 1",
            "market": "tie-safe final WIN market_rank == 1",
            "outsider": "p3_delta > 0; reinforces King podium belief",
            "survive": "target_top3 == 1",
            "collapse": "target_top3 == 0",
        },
        "purpose": "Census only: isolate cases where all three systems agree, then measure how often the anchor still misses the podium.",
        "anchors": int(len(anchors)),
        "collapse_cases": int(anchors["collapse"].sum()),
        "collapse_rate_pct": 100.0 * float(anchors["collapse"].mean()),
        "king_rank_mismatch_rows_in_join": int(z["king_rank_mismatch"].sum()),
        "market_rows_joined": int(len(z)),
        "outsider_signal_used_for_rerank": False,
        "market_used_inside_core_l1": False,
        "outcome_optimized_thresholds": False,
        "paid_compute": False,
        "2026_locked": True,
        "promotion": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("===== SUMMARY =====")
    print((out / "summary.json").read_text(encoding="utf-8"))
    print("===== HEADLINE =====")
    print((out / "headline.csv").read_text(encoding="utf-8"))
    print("===== ODDS BANDS =====")
    print((out / "odds-bands.csv").read_text(encoding="utf-8"))
    print("===== OUTSIDER SUPPORT QUANTILES =====")
    print((out / "outsider-support-quantiles.csv").read_text(encoding="utf-8"))
    print("===== COLLAPSE CASES FIRST 30 =====")
    print("\n".join((out / "collapse-cases.csv").read_text(encoding="utf-8").splitlines()[:31]))
    print("L1_ANCHOR_COLLAPSE_PHASE1_READY")

if __name__ == "__main__":
    main()
