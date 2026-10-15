"""Fail-closed readiness audit for immutable L2.1; not a ticket executor.

L2.1 requires frozen ANCHOR_FULL and an auditable pre-bet-cutoff full-ticket
price snapshot. A historical FINAL odds file never counts as that evidence.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/"contracts"/"KEIBA_L2_1_FIXED_PURCHASE_V1.json"
REQUIRED_HORSE_SIGNAL_KEYS=(
    "ANCHOR_FULL","AI_PROB","AI_RAW","MARKET","P__ANCHOR_FULL",
    "risk","distill_score_z","market_prob_norm"
)
REQUIRED_ODDS_KEY="full_ticket_odds_snapshot_as_of_bet_cutoff"

def verified_policy(path=FROZEN):
    c=json.loads(Path(path).read_text(encoding="utf-8"))
    if c.get("contract")!="KEIBA_L2_1_FIXED_PURCHASE_V1" or c.get("version")!="2.1.0":
        raise ValueError("L2.1 immutable version mismatch")
    if c.get("status")!="FROZEN_RESEARCH_POLICY_FOR_L3_HANDOFF":
        raise ValueError("L2.1 is not marked frozen")
    if c.get("primary_policy")!="STANDARD" or c.get("ticket_policy",{}).get("anchor")!=(
        "EXACT_EXISTING_ANCHOR_FULL; no replacement, re-ranking or L1/L1.75 feedback"
    ):
        raise ValueError("L2.1 frozen anchor/policy drift")
    rules=c.get("frozen_rules",{})
    std={(r["line"],r["gate"]) for r in rules.get("STANDARD",[])}
    strict={(r["line"],r["gate"]) for r in rules.get("STRICT_SUBSET",[])}
    if len(std)!=50 or len(strict)!=26 or not strict.issubset(std):
        raise ValueError("L2.1 50/26 rule counts or strict subset drift")
    lines={line for line,_ in std}
    if len(lines)!=9 or lines!=set(c.get("unique_standard_original_lines",[])):
        raise ValueError("L2.1 9-template drift")
    if set(c["ticket_policy"]["bet_types_live_in_policy"])!={
        "QUINELLA","TRIO","TRIFECTA"
    }:
        raise ValueError("L2.1 physical bet family drift")
    if c["ticket_policy"]["stake_baseline_yen_per_ticket"]!=100:
        raise ValueError("L2.1 baseline unit drift")
    return c

def stage_status(race_report,signals=None,odds=None):
    c=verified_policy()
    provided=signals or {}
    # Do not infer signal availability from mere 'SCORED' placeholders; exact
    # original L2.1 contract is required, including the frozen ANCHOR_FULL.
    absent=[key for key in REQUIRED_HORSE_SIGNAL_KEYS if key not in provided]
    if (not isinstance(odds,dict) or odds.get("kind")!="PRE_RACE"
        or odds.get("complete") is not True
        or not odds.get("observed_at") or not odds.get("bet_cutoff")):
        absent.append(REQUIRED_ODDS_KEY)
    # No ticket generation in V1: don't claim the stage is READY with inputs.
    return {
        "stage":"L2.1 STANDARD",
        "kind":"FROZEN_FIXED_TICKET_POLICY",
        "contract":c["contract"],
        "version":c["version"],
        "policy":"STANDARD",
        "status":"BLOCKED_MISSING_INPUTS" if absent else "BLOCKED_TICKET_EXECUTOR_NOT_CONNECTED",
        "required_inputs_missing":absent,
        "anchor":"ANCHOR_FULL (frozen, never reselected)",
        "original_templates":9,
        "standard_line_gate_rules":50,
        "strict_subset_rules":26,
        "bet_types":["QUINELLA","TRIO","TRIFECTA"],
        "stake_baseline_yen":100,
        "confidence":"DIAGNOSTIC_ONLY_NO_TICKET_MUTATION",
        "historical_final_odds":"FORBIDDEN_AS_LIVE_PRICE",
        "note":"L2.1 remains frozen; no tickets were purchased or simulated by this stage",
        "race_id":race_report["race_id"],
    }
