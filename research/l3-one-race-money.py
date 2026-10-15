#!/usr/bin/env python3
"""L3 money-management *paper* handoff for immutable L2.1 physical tickets.

Only allocate the frozen 100-yen flat baseline to ALL tickets, or BLOCK.
No dropping/reranking, no ROI optimization based on same-year in-sample data,
no payout handling or betting execution. User budget applies per race.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,math,re
from pathlib import Path

IN="KEIBA_L21_PREBET_QUOTED_TICKETS_V1"
OUT="KEIBA_L3_FIXED100_PAPER_PORTFOLIO_V1"
TYPES={"QUINELLA":2,"TRIO":3,"TRIFECTA":3}

def integer(raw,label,minimum=0,maximum=10_000_000):
    if type(raw) is not int or not minimum<=raw<=maximum:
        raise ValueError("INVALID_"+label)
    return raw

def key(rid,kind,selection):
    return hashlib.sha256((rid+"|"+kind+"|"+selection).encode("ascii")).hexdigest()[:20]

def allocate(tickets,config):
    if tickets.get("contract")!=IN or tickets.get("frozen_policy")!="KEIBA_L2_1_FIXED_PURCHASE_V1" or tickets.get("frozen_version")!="2.1.0":
        raise ValueError("NOT_ORIGINAL_FROZEN_L21_TICKETS")
    rid=str(tickets.get("race_id") or "")
    if not re.fullmatch(r"202[2-6]\d{8}",rid):
        raise ValueError("RACE_ID_INVALID_OR_2026_SEALED")
    if tickets.get("policy")!="STANDARD":
        raise ValueError("L3_ONLY_FROZEN_STANDARD_POLICY")
    if tickets.get("historical_final_odds_used") is not False:
        raise ValueError("FINAL_ODDS_CANNOT_ENTER_L3")
    observed=tickets.get("quote_observed_at")
    cutoff=tickets.get("bet_cutoff_at")
    if not isinstance(observed,str) or not isinstance(cutoff,str) or not observed or not cutoff:
        raise ValueError("MISSING_ODDS_PROVENANCE")
    if rid.startswith("2026"):
        try:
            at=datetime.datetime.fromisoformat(observed.replace("Z","+00:00"))
            last=datetime.datetime.fromisoformat(cutoff.replace("Z","+00:00"))
        except ValueError as exc:
            raise ValueError("2026_INVALID_L3_ASOF_TIMESTAMPS") from exc
        if at.tzinfo is None or last.tzinfo is None:
            raise ValueError("2026_L3_TIMESTAMP_TIMEZONE_REQUIRED")
        now=datetime.datetime.now(datetime.timezone.utc)
        if not at<=now<last:
            raise ValueError("2026_ONLY_REAL_FUTURE_PREBET_ASOF_NO_RESULT_REPLAY")
    if config.get("contract")!="KEIBA_L3_FIXED100_LIMITS_V1":
        raise ValueError("L3_MONEY_CONFIG_CONTRACT_MISMATCH")
    budget=integer(config.get("race_budget_yen"),"RACE_BUDGET_YEN",0)
    cap=integer(config.get("max_race_exposure_yen"),"MAX_RACE_EXPOSURE_YEN",0)
    max_count=integer(config.get("max_ticket_count"),"MAX_TICKET_COUNT",0,1000)
    if budget%100 or cap%100:raise ValueError("L3_BUDGET_MUST_BE_100YEN_MULTIPLE")
    if config.get("stake_policy")!="BASELINE_FLAT_100_ALL_OR_NONE":
        raise ValueError("UNRESEARCHED_L3_STAKING_POLICY_FORBIDDEN")
    inputs=tickets.get("tickets")
    if not isinstance(inputs,list) or len(inputs)>1000:
        raise ValueError("INVALID_L21_PHYSICAL_TICKET_FIELD")
    n=integer(tickets.get("physical_ticket_count"),"TICKET_COUNT",0,1000)
    total=integer(tickets.get("baseline_total_stake_yen"),"BASELINE_TOTAL",0)
    if len(inputs)!=n or total!=100*n:
        raise ValueError("L21_TICKET_COUNT_OR_BASELINE_DRIFT")
    seen=set()
    output=[]
    for row in inputs:
        if row.get("race_id")!=rid or row.get("bet_type") not in TYPES or row.get("observed_at")!=observed:
            raise ValueError("TICKET_PROVENANCE_DRIFT")
        kind=row["bet_type"]
        nums=row.get("selection_numbers")
        if (not isinstance(nums,list) or len(nums)!=TYPES[kind] or
            any(type(x)is not int or x<1 or x>24 for x in nums) or len(set(nums))!=len(nums)):
            raise ValueError("BAD_TICKET_SELECTION_NUMBERS")
        if kind!="TRIFECTA" and nums!=sorted(nums):
            raise ValueError("UNORDERED_TICKET_NOT_CANONICAL")
        selection="-".join(map(str,nums))
        if row.get("canonical_selection")!=selection:
            raise ValueError("PHYSICAL_TICKET_IDENTITY_DRIFT")
        identity=(rid,kind,selection)
        if identity in seen:raise ValueError("DUPLICATE_PHYSICAL_TICKET_L3_MUST_NOT_DOUBLE_BUY")
        seen.add(identity)
        odds=row.get("decimal_odds")
        if isinstance(odds,bool):
            raise ValueError("INVALID_TICKET_ODDS")
        try:price=float(odds)
        except (TypeError,ValueError) as exc:raise ValueError("INVALID_TICKET_ODDS") from exc
        if not math.isfinite(price) or price<1 or price>1_000_000:
            raise ValueError("INVALID_TICKET_ODDS")
        if row.get("baseline_stake_yen")!=100:
            raise ValueError("STAKE_BASELINE_MUST_REMAIN_100_YEN")
        rules=row.get("matched_line_gates")
        if not isinstance(rules,list) or not rules or any(not isinstance(x,dict) or not x.get("line") or not x.get("gate") for x in rules):
            raise ValueError("MISSING_ORIGINAL_FROZEN_GATE_EVIDENCE")
        output.append({"L2_1_ticket_id":key(rid,kind,selection),
                       "race_id":rid,"bet_type":kind,"canonical_selection":selection,
                       "stake_yen":100,"decimal_odds":price,
                       "odds_timestamp":observed,
                       "source_line_gates":rules,
                       "strict_subset":row.get("strict_subset") is True,
                       "risk_reason":"UNWEIGHTED_FROZEN_BASELINE_NOT_CALIBRATED_BET_EV"})
    if tickets.get("status")!="PAPER_ONLY_PREBET_QUOTE_VALIDATED":
        if tickets.get("status")!="BLOCKED_INCOMPLETE_QUOTE_BOOK":
            raise ValueError("UNKNOWN_L21_STATUS")
        if n!=0 or output:raise ValueError("BLOCKED_L21_CAN_NEVER_HAVE_BUYABLE_TICKETS")
        status="BLOCKED_UPSTREAM_PRICE_COVERAGE"
    elif n>max_count:
        status="BLOCKED_TICKET_COUNT_CAP_DO_NOT_PRUNE"
    elif total>budget:
        status="INSUFFICIENT_BUDGET_DO_NOT_DROP_TICKETS"
    elif total>cap:
        status="BLOCKED_RACE_EXPOSURE_LIMIT_DO_NOT_DROP_TICKETS"
    else:
        status="PAPER_ONLY_BASELINE_READY"
    if status!="PAPER_ONLY_BASELINE_READY":output=[]
    # Keep a non-executable FULL key audit even in blocked states.
    return {"contract":OUT,"race_id":rid,"status":status,
            "stake_policy":"BASELINE_FLAT_100_ALL_OR_NONE",
            "l21_original_policy":"STANDARD","source_l21_contract":IN,
            "odds_timestamp":observed,
            "max_race_budget_yen":budget,"max_race_exposure_yen":cap,
            "max_ticket_count":max_count,
            "required_ticket_count":n,"required_total_yen":total,
            "portfolio_total_yen":sum(t["stake_yen"] for t in output),
            "ticket_count":len(output),
            "required_ticket_ids":[key(rid,t["bet_type"],t["canonical_selection"]) for t in inputs],
            "tickets":output,
            "was_any_frozen_ticket_added_removed_or_reweighted":False,
            "zero_stake_for_individual_ticket_allowed":False,
            "any_purchase_executed":False,
            "true_win_probabilities_or_edge_used":False,
            "historical_roi_used_to_allocate":False,
            "original_quote_timestamp_authenticity_independently_verified":False,
            "source_anchor_asof_independently_verified":False,
            "NOT_LIVE_BET_AUTHORIZED":True
            }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tickets",required=True)
    p.add_argument("--limits",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    q=allocate(json.loads(Path(a.tickets).read_text(encoding="utf-8")),
               json.loads(Path(a.limits).read_text(encoding="utf-8")))
    f=Path(a.out);f.parent.mkdir(parents=True,exist_ok=True)
    f.write_text(json.dumps(q,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L3_IMMUTABLE_L21_FLAT100",q["race_id"],q["status"],q["ticket_count"],
          q["portfolio_total_yen"],flush=True)

if __name__=="__main__":main()
