#!/usr/bin/env python3
"""Exhaustive 18-starter JRA 3-product quote book regression.

JRA max full field (18): 153 quinella, 816 trio, 4,896 ordered
trifecta + 18 win = 5,883 independently timestamped physical prices.
These are SYNTHETIC only, never fetched from JRA or used for actual bets.
"""
import importlib.util,itertools,json,sys
from pathlib import Path
def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path("research")/file)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj
    spec.loader.exec_module(obj)
    return obj
bridge=load("max18_bridge","bridge-mobile-combination-odds.py")
l21=load("max18_l21","l21-frozen-prebet-price.py")
rid="202507010212";key="EXHAUSTIVE_18_FIXTURE"
url="https://www.jra.go.jp/test";stamp="2025-01-06T13:00:00+09:00"
def row(kind,nums,price):
    return {"raceKey":key,"betType":kind,
      "selection1":nums[0],"selection2":nums[1] if len(nums)>1 else None,
      "selection3":nums[2] if len(nums)>2 else None,
      "odds":price,"oddsMin":None,"oddsMax":None,
      "observedAt":stamp,"sourceUrl":url}
nums=tuple(range(1,19))
rows=[row("WIN",(n,),float(n+2)) for n in nums]
rows.extend(row("QUINELLA",v,16.0) for v in itertools.combinations(nums,2))
rows.extend(row("TRIO",v,35.0) for v in itertools.combinations(nums,3))
rows.extend(row("TRIFECTA",v,240.0) for v in itertools.permutations(nums,3))
assert len(rows)==5883
card={"race":{"raceKey":key,"canonicalRaceId":rid,
  "raceDate":"2025-01-06","startTime":"14:30",
  "status":"OFFICIAL","raceStatus":"SCHEDULED","scheduleStatus":"ACTIVE",
  "sourceUrl":url},
  "entries":[{"raceKey":key,"horseNo":n,"entryStatus":"ACTIVE",
              "canonicalHorseId":f"horse{n}"} for n in nums],
  "capture":{"source_system":"KEIBA_MOBILE_JRA",
             "final_probe_checked":True,"is_final":False,"source_status":"LIVE",
             "final_probe_checked_at":"2025-01-06T13:01:00+09:00",
             "bet_cutoff_at":"2025-01-06T14:25:00+09:00"},
  "odds_rows":rows}
book=bridge.convert(card)
assert len(book["rows"])==5865 and len(book["win_odds_rows"])==18
model={"contract":"KEIBA_L21_ORIGINAL_ANCHOR_FEATURES_PREBET_V1",
 "race_id":rid,"canonical_race_key":key,
 "anchor_source":"EXACT_EXISTING_ANCHOR_FULL",
 "anchor_horse_id":"horse1",
 "feature_source":"ORIGINAL_L21_HORSE_FEATURES",
 "input_phase":"PRE_RACE","features_asof_asserted":True,
 "features_observed_at":"2025-01-06T12:50:00+09:00",
 "win_odds_observed_at":stamp,
 "anchor_computed_at":"2025-01-06T13:02:00+09:00",
 "horses":[{"horse_id":f"horse{n}","horse_number":n,
 "P__ANCHOR_FULL":0.31 if n==1 else 0.03,
 "risk":0.1,"distill_score_z":0.1,
 "distill_ai_rank":n,"market_rank":n,
 "market_prob_norm":0.03,
 "win_odds_at_cutoff":float(n+2)} for n in nums]}
out=l21.execute(card,model,book)
assert out["status"]=="PAPER_ONLY_PREBET_QUOTE_VALIDATED",out
assert out["ticket_quote_missing_unique_count"]==0
assert out["physical_ticket_count"]>=1
print("MAX_18_HORSE_JRA_WIN_AND_THREE_BETTYPE_SOURCE_COMPLETE",
      len(rows),"combo",len(book["rows"]),"physical_tickets",out["physical_ticket_count"])
