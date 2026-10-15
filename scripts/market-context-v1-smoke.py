#!/usr/bin/env python3
"""Synthetic source-contract tests for independent pre-start win market overlay."""
import copy,datetime,json,sys
from pathlib import Path
sys.path.insert(0,str(Path("research").resolve()))
from market_context_v1 import attach_win_market

rid="202507010212"
names=("チャンネルトンネル","ブルーミンデザイン","ナムラエイハブ")
horses=[{
    "horse_id":str(100+i),"horse_number":i+1,"horse_name":names[i],
    "prior_year_king_raw_probabilities":{
        name:[0.5,0.3,0.2][i]+(0.01*j if i==0 else -0.005*j)
        for j,name in enumerate(("core4","pedlegacy","condition","full","light","pedv1","condrc"))
    }
} for i in range(3)]
analysis={
    "contract":"KEIBA_ONE_RACE_COMBINED_ANALYSIS_V1",
    "race_id":rid,
    "prior_year_new_model_inference":{"status":"ORIGINAL_MODELS_SCORED_NOT_VERIFIED_PREBET"},
    "horses":horses,
}
market={
    "contract":"KEIBA_JRA_WIN_ODDS_ASOF_V1","canonical_race_id":rid,
    "source_system":"KEIBA_MOBILE_JRA","source_phase":"PRE_RACE",
    "is_final":False,
    "observed_at":"2025-01-06T13:30:00+09:00",
    "scheduled_start_at":"2025-01-06T14:30:00+09:00",
    "rows":[{"betType":"WIN","selection1":i+1,"selection2":None,
        "selection3":None,"odds":odds,"oddsMin":None,"oddsMax":None,
        "observedAt":"2025-01-06T13:30:00+09:00",
        "sourceUrl":"https://www.jra.go.jp/example/odds"}
        for i,odds in enumerate([2.4,4.0,9.1])]
}
out=attach_win_market(analysis,market)
assert len(out["horses"])==3
assert sum(h["market_normalized_win_probability"] for h in out["horses"])==1 or abs(sum(h["market_normalized_win_probability"] for h in out["horses"])-1)<1e-12
assert len(out["horses"][0]["market_comparison"])==5
assert out["market_context"]["no_reorder_or_ticket_modification"] is True
assert out["market_context"]["not_l175"]
assert analysis["horses"][0].get("market_win_odds") is None # source not mutated

def rejected(mutator,reason):
    bad=copy.deepcopy(market)
    mutator(bad)
    try:attach_win_market(analysis,bad)
    except ValueError as exc:
        assert reason in str(exc),(reason,str(exc))
    else:raise AssertionError("UNSAFE_ODDS_ACCEPTED "+reason)

rejected(lambda m:m.update(is_final=True),"MUST_EXPLICITLY_ASSERT_NOT_FINAL")
rejected(lambda m:m.update(source_phase="FINAL"),"FINAL_OR_RESULTS_ODDS_FORBIDDEN")
rejected(lambda m:m.update(observed_at="2025-01-06T14:30:00+09:00"),"ODDS_SNAPSHOT_NOT_PRE_START")
rejected(lambda m:m.update(observed_at="2025-01-06T13:30:00"),"TIMEZONE_REQUIRED")
rejected(lambda m:m.update(canonical_race_id="202507010211"),"RACE_ID_DRIFT")
rejected(lambda m:m["rows"][0].update(sourceUrl="https://example.com/not-jra"),"ODDS_SOURCE_MUST_BE_OFFICIAL_JRA_HTTPS")
rejected(lambda m:m["rows"].pop(),"PARTIAL_OR_UNKNOWN_FIELD_ODDS_FORBIDDEN")
rejected(lambda m:m["rows"][0].update(betType="TRIFECTA"),"NON_WIN_OR_COMBINATION_ODDS_FORBIDDEN")
rejected(lambda m:m["rows"][0].update(observedAt="2025-01-06T14:00:00+09:00"),"MIXED_SNAPSHOT_TIMES")
rejected(lambda m:m["rows"][0].update(odds=float("nan")),"NONFINITE_WIN_ODDS")
rejected(lambda m:m["rows"][0].update(selection1=2),"MARKET_MISSING_OR_DUPLICATE_HORSE_NUMBER")
rejected(lambda m:m.update(source_system="BACKFILL_FINAL"),"UNVERIFIED_MARKET_SOURCE_SYSTEM")
safe=copy.deepcopy(analysis)
safe["prior_year_new_model_inference"]={"status":"UNAVAILABLE"}
no_model=attach_win_market(safe,market)
assert all("uncalibrated_win_probability_gap_pp" not in h["market_comparison"] for h in no_model["horses"])
assert no_model["market_context"]["model_comparison"]=="NONE"
seal=copy.deepcopy(market);seal["canonical_race_id"]="202607010212"
bad=copy.deepcopy(analysis);bad["race_id"]=seal["canonical_race_id"]
try:attach_win_market(bad,seal)
except ValueError as e:assert "2026_SEALED" in str(e)
else:raise AssertionError("2026 sealed race allowed")
print("STRICT_PREBET_MARKET_CONTEXT_TEST_PASS",len(out["horses"]),"runners", "13 source/time/leakage assertions")
