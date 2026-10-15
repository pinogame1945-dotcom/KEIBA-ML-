#!/usr/bin/env python3
"""Offline, stdlib-only verifier for the immutable research snapshot KEIBA L2.1.

Run: python scripts/verify_keiba_l2_1_freeze.py
This verifies the exact selected line/gates and published historical aggregates.
It does NOT validate predictive profitability, future odds availability or a live executor.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "contracts/KEIBA_L2_1_FIXED_PURCHASE_V1.json"


def rows(path):
    with (ROOT / path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def git_blob_sha(b):
    return hashlib.sha1(b"blob " + str(len(b)).encode("ascii") + b"\0" + b).hexdigest()


def check(expected, actual, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected={expected!r}, actual={actual!r}")


def main():
    c = json.loads(C.read_text(encoding="utf-8"))
    check("KEIBA_L2_1_FIXED_PURCHASE_V1", c["contract"], "contract")
    check("2.1.0", c["version"], "version")
    check("STANDARD", c["primary_policy"], "primary policy")
    check("L3_STAKE_SIZING_ONLY", c["intended_next_stage"], "handoff")
    check(True, c["research_baseline"]["2026_sealed"], "2026 sealed")
    check(True, c["ticket_policy"]["avoid_any_extra_gate_or_skip"], "no discretionary skip")
    check(True, c["ticket_policy"]["strict_counts_against_standard_only"], "STRICT subset only")
    check(100, c["ticket_policy"]["stake_baseline_yen_per_ticket"], "100-yen evaluation")
    assert "FINAL prices" in c["research_baseline"]["validation_scope"]
    assert "FINAL" in c["ticket_policy"]["ticket_features_market_cutoff_guard"]

    for x in c["evidence_files"]:
        p = ROOT / x["path"]
        assert p.is_file(), ("missing immutable source", str(p))
        check(x["git_blob_sha"], git_blob_sha(p.read_bytes()), "source blob " + x["path"])

    selected = rows("research-results/l2-hindsight-stable-black-union-v1/"
                    "run-38701489066/retained_original_black_and_red_to_black_candidate_lines.csv")
    for pol,key,count in (("STABLE_MEDIUM","STANDARD",50),
                          ("STABLE_STRICT","STRICT_SUBSET",26)):
        from_csv = {(r["line"],r["gate"]) for r in selected if r["policy"] == pol}
        freeze = {(r["line"],r["gate"]) for r in c["frozen_rules"][key]}
        check(count, len(freeze), "freeze count " + key)
        check(count, len(from_csv), "source count " + key)
        check(from_csv, freeze, "rule+gate exact set " + key)

    medium = {(r["line"],r["gate"]) for r in c["frozen_rules"]["STANDARD"]}
    strict = {(r["line"],r["gate"]) for r in c["frozen_rules"]["STRICT_SUBSET"]}
    assert strict.issubset(medium), "STRICT must be a subset, not an extra portfolio"
    uniq = sorted({line for line,gate in medium})
    check(sorted(c["unique_standard_original_lines"]), uniq, "nine unique recipes")
    check(9, len(uniq), "nine original recipes")
    families = {name.split("|")[2] for name in uniq}
    check({"QUINELLA","TRIO","TRIFECTA_FIRST","TRIFECTA_THIRD"}, families,
          "specific recipe families")
    check(["QUINELLA","TRIFECTA","TRIO"],
          sorted(c["ticket_policy"]["bet_types_live_in_policy"]), "three physical bet types")

    pooled = rows("research-results/l2-hindsight-stable-black-union-v1/"
                  "run-38701489066/three_year_stable_black_only_dedup_portfolio.csv")
    labels = [("STABLE_MEDIUM","STANDARD"),("STABLE_STRICT","STRICT")]
    for policy,label in labels:
        for y in ("2023","2024","2025","2023-2025"):
            r = next((x for x in pooled if x["year"]==y and x["portfolio"]==policy), None)
            assert r, ("no historical baseline row",policy,y)
            target = c["research_baseline"][label + "_ALL"] if y=="2023-2025" else (
                c["research_baseline"][label + "_BY_YEAR"][y])
            for k in ("available_races","races_bought","tickets","winning_tickets",
                      "stake_yen","return_yen","profit_yen"):
                check(target[k], int(r[k]), f"{label} {y} {k}")
            for k in ("roi_pct","roi_minus_biggest3_pct"):
                assert abs(target[k]-float(r[k])) < 1e-10, (label,y,k)
    checks = c["research_baseline"]["STANDARD_ALL"]
    check(13479,checks["tickets"],"standard tickets")
    check(376,checks["winning_tickets"],"standard hits")
    check(1471660,checks["return_yen"],"standard returns")
    check(1347900,checks["stake_yen"],"standard stakes")
    check(123760,checks["profit_yen"],"standard profit")
    check(1431,c["research_baseline"]["STRICT_ALL"]["tickets"],"strict tickets")

    l175=json.loads((ROOT/"research-results/l3-l175-frozen-ticket-trust-audit-v1/"
                        "run-38710197503/summary.json").read_text(encoding="utf-8"))
    check(checks["tickets"],l175["standard_baseline"]["tickets"],"trust standard tickets")
    check(checks["winning_tickets"],l175["standard_baseline"]["hits"],"trust standard hits")
    check(checks["return_yen"],l175["standard_baseline"]["payout_yen"],"trust payout")
    check(13264,l175["l175_full_coverage"]["tickets"],"complete signals")
    check(215,l175["l175_full_coverage"]["missing_tickets"],"unknown signals")
    check(True,l175["2026_sealed"],"trust 2026 seal")
    tier=rows("research-results/l3-l175-frozen-ticket-trust-audit-v1/"
              "run-38710197503/trust_hit_roi_by_tier_year_and_context.csv")
    for which,hit,roi in (("Q4",3.00865,144.67469),("Q5",3.63499,113.15032)):
        r=next(x for x in tier if x["dimension"]=="p3_tier" and x["year"]=="2023-2025" and x["bucket"]==which)
        assert abs(float(r["hit_rate_pct"])-hit)<1e-5
        assert abs(float(r["roi_pct"])-roi)<1e-5

    print("KEIBA_L2_1_FROZEN_CONTRACT_VERIFIED")
    print("STANDARD: 50 rules, 9 source recipes, 13,479 historical tickets, ROI 109.18169%")
    print("STRICT: subset 26 rules, 1,431 tickets, ROI 118.30189%")
    print("L1.75: 13,264/13,479 joined, diagnostic-only; 2026 sealed")
    print("NOT OOS VALIDATED. HISTORICAL FINAL ODDS ARE NOT A LIVE CUTOFF PRICE PROOF.")


if __name__ == "__main__":
    main()
