#!/usr/bin/env python3
"""Read-only bridge audit: saved original 2026 AI/Risk/ANCHOR -> frozen L2.1 inputs.

This is NOT a betting adapter. It deliberately refuses to label historical
2026-10-17 prices as an independently timestamp-certified bet-cutoff snapshot.
No scores, model weights, ranks, frozen 50 rules, tickets, or stakes are rebuilt.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

PURE = "KEIBA_2026_ORIGINAL_PURE_AI_RISK_ANCHOR_RESEARCH_CANDIDATE_V1"
L175 = "KEIBA_L175_BACKFILL_2026_1017_RESEARCH_PRIVATE_V1"
L175_RACE = "KEIBA_2026_1017_L175_MARKET_JOIN_RESEARCH_V1"
FREEZE = "KEIBA_L2_1_FIXED_PURCHASE_V1"
DAY = "2026-10-17"
BLOCKERS = (
    "INDEPENDENT_PRESTART_FEATURE_AND_MARKET_ASOF_NOT_CERTIFIED",
    "AUTHENTIC_BET_CUTOFF_TIMESTAMPED_TICKET_ODDS_NOT_AVAILABLE",
    "FROZEN_L21_GATE_FEATURE_PANEL_NOT_JOINED",
    "ORIGINAL_2026_RESEARCH_CANDIDATES_NOT_RELEASE_APPROVED",
)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def lines(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def scalar(x):
    return type(x) in (float, int) and math.isfinite(x)

def audit(pure_idx, pure_scores, market_idx, market_scores, frozen_file, output):
    require(not output.exists(), "REFUSE_OVERWRITE_RESEARCH_REPORT")
    p, m, c = load(pure_idx), load(market_idx), load(frozen_file)
    require(
        p.get("contract") == PURE and p.get("races") == 36 and
        p.get("horses") == 471 and p.get("anchor_races_scored") == 36 and
        p.get("anchor_races_blocked_favorite_tie") == [] and
        p.get("2026_target_outcomes_used") is False and
        p.get("source_asof_independently_verified") is False and
        p.get("original_model_release_approved") is False and
        p.get("ready_for_betting") is False and
        p.get("frozen_l21_l3_invoked") is False and
        p.get("original_ability_fit_years") == [2024, 2025] and
        p.get("original_risk_oos_teacher_years") == [2022, 2023, 2024, 2025],
        "ORIGINAL_PURE_MODEL_METADATA_OR_RESEARCH_SEAL_DRIFT"
    )
    files = p.get("files", {})
    require(pure_scores.name in files and
            files[pure_scores.name]["sha256"] == digest(pure_scores) and
            files[pure_scores.name]["bytes"] == pure_scores.stat().st_size and
            len(files) == 4,
            "ORIGINAL_PURE_SCORE_SHA_OR_MODEL_FILE_MANIFEST_DRIFT")
    require(m.get("contract") == L175 and m.get("races") == 36 and
            m.get("horses") == 471 and
            m.get("horses_with_win_market") == 471 and
            m.get("ranking_policy") == "SEVEN_KING_UNCHANGED" and
            m.get("market_asof_independently_verified") is False and
            m.get("ready_for_betting") is False and
            m.get("2026_target_results_used") is False and
            m.get("files", {}).get(market_scores.name) == digest(market_scores),
            "ORIGINAL_L175_SAVED_MARKET_SHA_OR_PREBET_SEAL_DRIFT")
    require(c.get("contract") == FREEZE and c.get("version") == "2.1.0" and
            c.get("primary_policy") == "STANDARD" and
            c.get("rule_counts", {}).get("STANDARD") == 50 and
            c.get("rule_counts", {}).get("STRICT_SUBSET") == 26 and
            len(c["frozen_rules"]["STANDARD"]) == 50 and
            len(c["frozen_rules"]["STRICT_SUBSET"]) == 26 and
            c["ticket_policy"]["anchor"].startswith("EXACT_EXISTING_ANCHOR_FULL"),
            "ORIGINAL_FROZEN_L21_RULES_CHANGED")

    market = {}
    for row in lines(market_scores):
        rid = str(row.get("race_id") or "")
        hs = row.get("horses") or []
        require(rid not in market and len(rid) == 12 and rid.startswith("2026") and
                row.get("contract") == L175_RACE and
                row.get("horse_count") == len(hs) and
                row.get("win_market_source") == "JRA_OFFICIAL_POST" and
                row.get("market_asof_independently_verified") is False and
                row.get("ready_for_betting") is False and
                row.get("2026_target_results_used") is False and
                row.get("ranking_policy") == "SEVEN_KING_UNCHANGED",
                "SAVED_MARKET_RACE_OR_PROVENANCE_DRIFT:" + rid)
        by_horse = {str(h["horse_id"]): h for h in hs}
        require(len(by_horse) == len(hs), "DUPLICATE_SAVED_MARKET_HORSE:" + rid)
        market[rid] = by_horse
    require(len(market) == 36 and sum(map(len, market.values())) == 471,
            "SAVED_MARKET_36R_471H_SCOPE_DRIFT")

    seen = set()
    result_races = []
    total = pcount = anchor_count = 0
    for row in lines(pure_scores):
        rid = str(row.get("race_id") or "")
        hs = row.get("horses") or []
        require(rid in market and rid not in seen and row.get("race_date") == DAY and
                row.get("anchor_status") == "RESEARCH_SCORED_ASOF_UNVERIFIED" and
                row.get("ready_for_betting") is False and
                row.get("asof_independently_verified") is False and
                row.get("2026_target_results_used") is False and
                row.get("l21_l3_invoked") is False,
                "ORIGINAL_PURE_HISTORIC_ASOF_OR_SCOPE_DRIFT:" + rid)
        expected = market[rid]
        require(len(hs) == len(expected) and
                {str(h["horse_id"]) for h in hs} == set(expected) and
                sorted(h["distill_ai_rank"] for h in hs) == list(range(1, len(hs)+1)),
                "ORIGINAL_PURE_RANK_OR_HORSE_POPULATION_DRIFT:" + rid)
        axis = row["anchor_horse_number"]
        require(type(axis) is int and axis > 0, "ORIGINAL_ANCHOR_AXIS_MISSING:" + rid)
        vote = 0
        pp = []
        for h in hs:
            hid = str(h["horse_id"])
            old = expected[hid]
            require(type(h["horse_number"]) is int and
                    h["horse_number"] == old["horse_number"] and
                    h["market_rank"] == old["market_rank"] and
                    scalar(h["original_prior_market_win_odds"]) and
                    float(h["original_prior_market_win_odds"]) == float(old["market_win_odds"]),
                    "ORIGINAL_ANCHOR_MARKET_SAME_SOURCE_DRIFT:" + rid + ":" + hid)
            require(type(h["distill_ai_rank"]) is int and
                    scalar(h["distill_score_z"]) and
                    scalar(h["distill_score_gap_to_best"]) and
                    scalar(h["risk"]) and 0 <= h["risk"] <= 1 and
                    scalar(h["P__ANCHOR_FULL"]) and 0 <= h["P__ANCHOR_FULL"] <= 1 and
                    type(h["original_anchor"]) is bool,
                    "ORIGINAL_PURE_OR_RISK_OR_ANCHOR_NOT_REAL:" + rid + ":" + hid)
            vote += int(h["original_anchor"])
            pp.append(h["P__ANCHOR_FULL"])
            if h["original_anchor"]:
                require(h["horse_number"] == axis, "ORIGINAL_ANCHOR_AXIS_IDENTITY_DRIFT:" + rid)
        require(vote == 1 and abs(math.fsum(pp) - 1) < 1e-9,
                "ORIGINAL_ANCHOR_PROB_NORMALIZATION_DRIFT:" + rid)
        pcount += len(hs)
        total += len(hs)
        anchor_count += 1
        seen.add(rid)
        result_races.append({
            "race_id": rid, "horses": len(hs),
            "original_ai_rank_covered": len(hs),
            "original_risk_covered": len(hs),
            "original_anchor_prob_covered": len(hs),
            "original_anchor_horse_number": axis,
            "original_model_source_verified": True,
            "market_derived_from_saved_original_l175": True,
            "frozen_l21_run_status": "BLOCKED_RESEARCH_ONLY_MISSING_PREBET_SOURCES",
            "remaining_blockers": list(BLOCKERS),
            "ready_for_betting": False,
            "tickets": None,
        })
    require(len(seen) == 36 and seen == set(market) and total == pcount == 471 and
            anchor_count == 36, "2026_ORIGINAL_L21_JOIN_NOT_36R_471H")
    out = {
        "contract": "KEIBA_2026_1017_ORIGINAL_AI_RISK_ANCHOR_FROZEN_L21_READONLY_AUDIT_V1",
        "race_date": DAY,
        "real_races_joined": 36, "real_horses_joined": 471,
        "original_ai_rank_horses": 471,
        "original_correction_risk_horses": 471,
        "original_anchor_full_probability_horses": 471,
        "original_anchor_full_axis_races": 36,
        "frozen_standard_rules": 50, "frozen_strict_subset_rules": 26,
        "seven_king_rank_changed": False,
        "frozen_l21_run_status": "BLOCKED_RESEARCH_ONLY_MISSING_PREBET_SOURCES",
        "remaining_blockers": list(BLOCKERS),
        "2026_target_results_used": False,
        "2026_model_release_approved": False,
        "historical_2026_prebet_source_independently_verified": False,
        "frozen_l21_engine_invoked": False, "l3_v3_invoked": False,
        "tickets": None, "ready_for_betting": False,
        "races": sorted(result_races, key=lambda r: r["race_id"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True)+"\n",encoding="utf-8")
    print("ORIGINAL_REAL_2026_L21_SOURCE_JOIN_VERIFIED_36R_471H_36_ANCHOR_AXES_NOT_BETTABLE")

def main():
    p = argparse.ArgumentParser()
    for key in ("original-index","original-scores","l175-index","l175-scores","frozen-l21-contract","output"):
        p.add_argument("--"+key, required=True, type=Path)
    a = p.parse_args()
    audit(a.original_index,a.original_scores,a.l175_index,a.l175_scores,
          a.frozen_l21_contract,a.output)

if __name__ == "__main__":
    main()
