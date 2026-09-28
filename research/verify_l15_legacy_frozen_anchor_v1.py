#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Verify frozen L1.5 legacy ANCHOR V1 summary")
    p.add_argument("--summary", required=True)
    p.add_argument(
        "--manifest",
        default="research/legacy_frozen/l15_legacy_frozen_v1_anchor.json",
    )
    return p.parse_args()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct3(rate):
    return round(float(rate) * 100.0, 3)


def pp3(value):
    return round(float(value), 3)


def expect_eq(label, actual, expected):
    if actual != expected:
        raise SystemExit(f"LEGACY_FROZEN_MISMATCH {label}: actual={actual} expected={expected}")


def main():
    a = parse_args()
    manifest = load(a.manifest)
    summary = load(a.summary)

    expect_eq("manifest.contract", manifest.get("contract"), "L15_LEGACY_FROZEN_V1")
    expect_eq("manifest.role", manifest.get("role"), "ANCHOR")
    expect_eq("summary.contract", summary.get("contract"), "L15_COUNCIL_ROUTER_V3")
    expect_eq("summary.role", summary.get("role"), "ANCHOR")

    protocol = summary.get("protocol") or {}
    expect_eq("summary.protocol.locked_years", protocol.get("locked_years"), [2026])
    expect_eq("summary.protocol.odds_used", protocol.get("odds_used"), False)

    expected = manifest["expected"]
    agg = summary["aggregate_all_cells"]
    exp_agg = expected["aggregate_all_cells"]
    expect_eq(
        "aggregate.router_v3_hit_rate_pct_3dp",
        pct3(agg["router_v3_hit_rate"]),
        exp_agg["router_v3_hit_rate_pct_3dp"],
    )
    expect_eq(
        "aggregate.baseline_hit_rate_pct_3dp",
        pct3(agg["baseline_hit_rate"]),
        exp_agg["baseline_hit_rate_pct_3dp"],
    )
    expect_eq(
        "aggregate.delta_vs_baseline_pp_3dp",
        pp3(agg["delta_vs_baseline_pp"]),
        exp_agg["delta_vs_baseline_pp_3dp"],
    )

    fold_map = {}
    for fold in summary.get("folds") or []:
        key = f'{int(fold["test_year"])}|{fold["role"]}|{int(fold["top_n"])}'
        fold_map[key] = fold

    for key, exp in expected["folds"].items():
        if key not in fold_map:
            raise SystemExit(f"LEGACY_FROZEN_MISSING_FOLD {key}")
        fold = fold_map[key]
        expect_eq(
            f"{key}.router_v3_pct_3dp",
            pct3(fold["router_v3_hit_rate"]),
            exp["router_v3_pct_3dp"],
        )
        expect_eq(
            f"{key}.baseline_pct_3dp",
            pct3(fold["baseline_hit_rate"]),
            exp["baseline_pct_3dp"],
        )
        expect_eq(
            f"{key}.delta_pp_3dp",
            pp3(fold["delta_vs_baseline_pp"]),
            exp["delta_pp_3dp"],
        )

    print(
        "L15_LEGACY_FROZEN_ANCHOR_V1_OK",
        f"summary={a.summary}",
        f"folds={len(expected['folds'])}",
        f"aggregate_v3={pct3(agg['router_v3_hit_rate']):.3f}%",
        f"aggregate_baseline={pct3(agg['baseline_hit_rate']):.3f}%",
        f"delta={pp3(agg['delta_vs_baseline_pp']):+.3f}pp",
    )


if __name__ == "__main__":
    main()
