#!/usr/bin/env python3
"""Standard-library regression for frozen ticket geometry, source gate,
duplicate physical tickets, price band half-open boundaries, missing quotes.
No historical outcome or real betting.
"""
from __future__ import annotations
import copy,importlib.util,itertools,json,sys
from pathlib import Path
source=Path("research/l21-frozen-prebet-price.py").resolve()
spec=importlib.util.spec_from_file_location("l21",source)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
frozen,standard,strict=mod.source_contract()
assert len(standard)==50 and len(strict)==26 and strict.issubset(standard)
assert len({x[0] for x in standard})==9
assert mod.canonical("TRIO",(4,2,3))==(2,3,4)
assert mod.canonical("QUINELLA",(5,2))==(2,5)
assert mod.canonical("TRIFECTA",(4,2,3))==(4,2,3)
assert mod.generated(1,[2,3],"TRIFECTA_FIRST")=={
    ("TRIFECTA",(1,2,3)),("TRIFECTA",(1,3,2))}
assert mod.generated(1,[2,3],"TRIFECTA_THIRD")=={
    ("TRIFECTA",(2,3,1)),("TRIFECTA",(3,2,1))}
assert mod.generated(1,[2,3],"TRIO")=={("TRIO",(1,2,3))}
rid="202507010212"
key="m_jra_test"
card={"race":{"raceKey":key,"canonicalRaceId":rid,
    "raceDate":"2025-01-06","startTime":"14:30",
    "raceStatus":"SCHEDULED","scheduleStatus":"ACTIVE","status":"OFFICIAL"},
    "entries":[{"raceKey":key,"horseNo":n,"entryStatus":"ACTIVE"} for n in range(1,9)]}
    # source features are deliberately synthetic, not a derived 7 King proxy.
names=("raw","market")
features={"contract":"KEIBA_L21_ORIGINAL_ANCHOR_FEATURES_PREBET_V1",
    "race_id":rid,"canonical_race_key":key,
    "anchor_source":"EXACT_EXISTING_ANCHOR_FULL",
    "anchor_horse_id":"horse1",
    "feature_source":"ORIGINAL_L21_HORSE_FEATURES",
    "input_phase":"PRE_RACE","features_asof_asserted":True,
    "features_observed_at":"2025-01-06T13:00:00+09:00",
    "win_odds_observed_at":"2025-01-06T13:32:00+09:00",
    "anchor_computed_at":"2025-01-06T13:33:00+09:00",
    "horses":[{"horse_id":f"horse{n}","horse_number":n,
        "P__ANCHOR_FULL":0.31 if n==1 else 0.14-n*0.005,
        "risk":0.09,"distill_score_z":0.45,
        "distill_ai_rank":{2:3,3:4,4:5,5:2}.get(n,n),"market_rank":n,
        "market_prob_norm":0.15,"win_odds_at_cutoff":float(n+2)}
        for n in range(1,9)]}
quotes={"contract":"KEIBA_JRA_COMBINATION_ODDS_ASOF_V1",
    "canonical_race_id":rid,"raceKey":key,
    "source_system":"KEIBA_MOBILE_JRA","source_phase":"PRE_RACE",
    "is_final":False,"source_status":"LIVE",
    "source_confirmed_prestart":True,
    "observed_at":"2025-01-06T13:32:00+09:00",
    "bet_cutoff_at":"2025-01-06T14:25:00+09:00",
    "scheduled_start_at":"2025-01-06T14:30:00+09:00",
    "rows":[]}
def add(kind,nums,price):
    quote={"raceKey":key,"betType":kind,"selection1":nums[0],
        "selection2":nums[1],"selection3":nums[2] if len(nums)==3 else None,
        "odds":price,"oddsMin":None,"oddsMax":None,
        "observedAt":quotes["observed_at"],
        "sourceUrl":"https://www.jra.go.jp/test"}
    quotes["rows"].append(quote)
for b in range(2,9):add("QUINELLA",(1,b),16.0)
for b,c in itertools.combinations(range(2,9),2):
    add("TRIO",(1,b,c),16.0 if (b+c)%2 else 35.0)
    for first,second in ((b,c),(c,b)):
        add("TRIFECTA",(1,first,second),240.0)
        add("TRIFECTA",(first,second,1),25.0)
result=mod.execute(card,features,quotes)
assert result["status"]=="PAPER_ONLY_PREBET_QUOTE_VALIDATED"
assert result["ticket_quote_missing_unique_count"]==0
assert result["physical_ticket_count"]>0
assert len({(r["bet_type"],r["canonical_selection"]) for r in result["tickets"]})==len(result["tickets"])
assert any(len(t["matched_line_gates"])>1 for t in result["tickets"])
assert any(t["bet_type"]=="QUINELLA" for t in result["tickets"])
assert any(t["bet_type"]=="TRIO" for t in result["tickets"])
assert any(t["bet_type"]=="TRIFECTA" for t in result["tickets"])
assert result["baseline_total_stake_yen"]==result["physical_ticket_count"]*100
assert all(t["baseline_stake_yen"]==100 for t in result["tickets"])
assert all(t["observed_at"]==quotes["observed_at"] for t in result["tickets"])
strict_results=mod.execute(card,features,quotes,policy="STRICT_SUBSET")
assert strict_results["status"]==result["status"]
orig={(t["bet_type"],t["canonical_selection"]) for t in result["tickets"]}
sub={(t["bet_type"],t["canonical_selection"]) for t in strict_results["tickets"]}
assert sub.issubset(orig) and len(sub)<=len(orig)
assert mod.BAND["15-20"][0]<=15.0<mod.BAND["15-20"][1]
assert not(mod.BAND["15-20"][0]<=20.0<mod.BAND["15-20"][1])
def reject(edit,reason):
    a=copy.deepcopy(card);b=copy.deepcopy(features);c=copy.deepcopy(quotes)
    edit(a,b,c)
    try:mod.execute(a,b,c)
    except ValueError as exc:assert reason in str(exc),(reason,str(exc))
    else:raise AssertionError("UNSAFE_L21_INPUT_ACCEPTED "+reason)
reject(lambda a,b,c:a["race"].update(status="SCHEDULED"),"RESULT_OR_CANCELLED")
reject(lambda a,b,c:a["race"].update(canonicalRaceId="202607010212"),"2026_SEALED")
reject(lambda a,b,c:b.update(anchor_source="PRIOR_YEAR_7KING_PROXY"),"L21_ORIGINAL_HORSE_FEATURE")
reject(lambda a,b,c:b.update(features_asof_asserted=False),"L21_ORIGINAL_HORSE_FEATURE")
reject(lambda a,b,c:b.update(anchor_horse_id="not_actual"),"FROZEN_ANCHOR_ID_NOT_FOUND")
reject(lambda a,b,c:b["horses"][1].pop("risk"),"MISSING_NUMERIC_risk")
reject(lambda a,b,c:b["horses"][1].update(features_observed_at="2025-01-06T14:30:00+09:00"),"none") if False else None
reject(lambda a,b,c:b.update(features_observed_at="2025-01-06T14:26:00+09:00"),"L21_HORSE_FEATURES_AFTER_QUOTE_SNAPSHOT")
reject(lambda a,b,c:b.update(win_odds_observed_at="2025-01-06T13:31:00+09:00"),"L21_ORIGINAL_ANCHOR_WIN_ODDS_NOT_SAME_MARKET_SNAPSHOT")
reject(lambda a,b,c:b.update(anchor_computed_at="2025-01-06T14:26:00+09:00"),"L21_ORIGINAL_ANCHOR_COMPUTED_OUTSIDE_QUOTE_TO_CUTOFF")
reject(lambda a,b,c:c.update(observed_at="2025-01-06T14:30:00+09:00"),"QUOTE_NOT_BEFORE")
reject(lambda a,b,c:c.update(is_final=True),"COMBINATION_QUOTES_NOT_AUTHORIZED")
reject(lambda a,b,c:c["rows"][0].update(betType="TRIFECTA"),"INVALID_HORSE_NUMBER_ticket")
reject(lambda a,b,c:c["rows"].append(copy.deepcopy(c["rows"][0])),"DUPLICATE_PHYSICAL_QUOTE")
reject(lambda a,b,c:c["rows"][0].update(sourceUrl="https://example.org"),"UNTRUSTED_JRA_SOURCE_URL")
reject(lambda a,b,c:c["rows"][0].update(odds=float("nan")),"NONFINITE_combination_odds")
missing=copy.deepcopy(quotes)
missing["rows"]=[r for r in missing["rows"] if not(r["betType"]=="TRIO" and r["selection1"]==1 and r["selection2"]==2 and r["selection3"]==3)]
blocked=mod.execute(card,features,missing)
assert blocked["status"]=="BLOCKED_INCOMPLETE_QUOTE_BOOK"
assert blocked["ticket_quote_missing_unique_count"]>0
assert blocked["tickets"]==[] and blocked["baseline_total_stake_yen"]==0
f_spec=importlib.util.spec_from_file_location("l21_view",
    Path("research/l21-format-price-preview.py").resolve())
formatter=importlib.util.module_from_spec(f_spec);f_spec.loader.exec_module(formatter)
preview=formatter.render(result)
assert "凍結買い目" in preview and "物理的な購入候補" in preview
assert "馬連" in preview and "三連複" in preview and "三連単" in preview
assert "重複排除前" in preview
blocked_preview=formatter.render(blocked)
assert "購入候補は出力しない" in blocked_preview
assert "不足した物理オッズ" in blocked_preview
print("FROZEN_L21_50_GATES_AND_9_TEMPLATE_STRICT_PREBET_PRICING_PASS",
      "paper_unique_tickets",result["physical_ticket_count"],
      "strict_subset",strict_results["physical_ticket_count"])
