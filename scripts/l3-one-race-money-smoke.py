#!/usr/bin/env python3
"""Fixed 100 yen L3 paper allocator never hides/reranks/buys frozen tickets."""
import copy,importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location("money",Path("research/l3-one-race-money.py").resolve())
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rid="202507010212"
orig={"contract":"KEIBA_L21_PREBET_QUOTED_TICKETS_V1",
 "frozen_policy":"KEIBA_L2_1_FIXED_PURCHASE_V1","frozen_version":"2.1.0",
 "race_id":rid,"policy":"STANDARD","historical_final_odds_used":False,
 "status":"PAPER_ONLY_PREBET_QUOTE_VALIDATED",
 "quote_observed_at":"2025-01-06T13:00:00+09:00",
 "bet_cutoff_at":"2025-01-06T14:25:00+09:00",
 "physical_ticket_count":3,"baseline_total_stake_yen":300,
 "tickets":[{"race_id":rid,"bet_type":bt,"selection_numbers":numbers,
 "canonical_selection":"-".join(map(str,numbers)),
 "observed_at":"2025-01-06T13:00:00+09:00",
 "decimal_odds":price,"baseline_stake_yen":100,
 "matched_line_gates":[{"line":"AI_PROB|2|TRIO|15-20","gate":"ALL"}]}
 for bt,numbers,price in [
   ("QUINELLA",[2,5],17.2),("TRIO",[2,5,7],19.2),("TRIFECTA",[5,7,2],250.1)]]}
limits={"contract":"KEIBA_L3_FIXED100_LIMITS_V1",
 "race_budget_yen":500,"max_race_exposure_yen":300,
 "max_ticket_count":10,"stake_policy":"BASELINE_FLAT_100_ALL_OR_NONE"}
a=m.allocate(orig,limits)
assert a["status"]=="PAPER_ONLY_BASELINE_READY"
assert a["portfolio_total_yen"]==300 and a["ticket_count"]==3
assert a["required_ticket_count"]==a["ticket_count"]==3
assert [t["canonical_selection"] for t in a["tickets"]]==[
    "2-5","2-5-7","5-7-2"]
assert all(t["stake_yen"]==100 for t in a["tickets"])
assert len(set(t["L2_1_ticket_id"] for t in a["tickets"]))==3
assert a["any_purchase_executed"] is False
assert m.allocate(orig,{**limits,"race_budget_yen":200})["status"]=="INSUFFICIENT_BUDGET_DO_NOT_DROP_TICKETS"
assert m.allocate(orig,{**limits,"race_budget_yen":200})["tickets"]==[]
assert m.allocate(orig,{**limits,"max_race_exposure_yen":200})["status"]=="BLOCKED_RACE_EXPOSURE_LIMIT_DO_NOT_DROP_TICKETS"
assert m.allocate(orig,{**limits,"max_ticket_count":2})["status"]=="BLOCKED_TICKET_COUNT_CAP_DO_NOT_PRUNE"
b=copy.deepcopy(orig);b.update(status="BLOCKED_INCOMPLETE_QUOTE_BOOK",
                              physical_ticket_count=0,baseline_total_stake_yen=0,tickets=[])
assert m.allocate(b,limits)["status"]=="BLOCKED_UPSTREAM_PRICE_COVERAGE"
def wrong(edit,msg):
    b=copy.deepcopy(orig);c=copy.deepcopy(limits);edit(b,c)
    try:m.allocate(b,c)
    except ValueError as e:assert msg in str(e),(msg,str(e))
    else:raise AssertionError("DID_NOT_FAIL_CLOSED "+msg)
wrong(lambda b,c:b["tickets"].append(copy.deepcopy(b["tickets"][0])),"TICKET_COUNT_OR_BASELINE")
wrong(lambda b,c:b["tickets"][0].update(decimal_odds=float("nan")),"INVALID_TICKET_ODDS")
wrong(lambda b,c:b["tickets"][0].update(baseline_stake_yen=200),"STAKE_BASELINE")
wrong(lambda b,c:b["tickets"][0].update(canonical_selection="5-2"),"PHYSICAL_TICKET_IDENTITY")
wrong(lambda b,c:b["tickets"][0].update(selection_numbers=[5,2],canonical_selection="5-2"),"UNORDERED_TICKET")
wrong(lambda b,c:b["tickets"][0].update(observed_at="POST"),"TICKET_PROVENANCE")
wrong(lambda b,c:b.update(historical_final_odds_used=True),"FINAL_ODDS")
wrong(lambda b,c:b.update(race_id="202607010212"),"2026_ONLY_REAL_FUTURE_PREBET")
wrong(lambda b,c:c.update(stake_policy="KELLY"),"UNRESEARCHED_L3")
wrong(lambda b,c:c.update(race_budget_yen=155),"L3_BUDGET_MUST_BE_100YEN_MULTIPLE")
wrong(lambda b,c:b["tickets"][0].update(matched_line_gates=[]),"MISSING_ORIGINAL_FROZEN_GATE")
print("L3_IMMUTABLE_PAPER_100YEN_CAPS_NO_TICKET_MUTATION_PASS")
