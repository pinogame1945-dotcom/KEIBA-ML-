#!/usr/bin/env python3
"""P4: strict schema guard for published official JRA 8 bet types.
This is NOT a live odds HTML parser; JRA currently returns error 013.
Never substitute predicted netkeiba odds or final odds for pre-bet prices.
"""
import json,math
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
RID="202605040611"
BET_TYPES={
 "WIN":(1,False,False),"PLACE":(1,False,True),
 "BRACKET_QUINELLA":(2,False,False),"QUINELLA":(2,False,False),
 "WIDE":(2,False,True),"EXACTA":(2,True,False),
 "TRIO":(3,False,False),"TRIFECTA":(3,True,False),
}
BASELINE=json.loads(Path("audit-results/fuji-20261017-149-asof-20261016.json").read_text("utf-8"))
CARD={x["horse_number"]:x for x in BASELINE["runners"]}
assert len(CARD)==14 and BASELINE["race_id"]==RID
def aware_iso(s):
 if not isinstance(s,str):return False
 try: d=datetime.fromisoformat(s.replace("Z","+00:00"))
 except ValueError:return False
 return d.tzinfo is not None and d.utcoffset() is not None
def validate_quote_batch(batch):
 if batch.get("race_id")!=RID:raise ValueError("RACE_ID_MISMATCH")
 if batch.get("source_error") or batch.get("placeholder") or batch.get("error013"):
  raise ValueError("ODDS_NOT_AVAILABLE")
 u=urlparse(batch.get("source_url",""))
 if u.scheme!="https" or u.hostname!="www.jra.go.jp" or u.path!="/JRADB/accessO.html":
  raise ValueError("NOT_OFFICIAL_JRA_ODDS_SOURCE")
 if batch.get("race_header_verified") is not True:raise ValueError("RACE_HEADER_NOT_VERIFIED")
 if batch.get("snapshot_type")!="PRE_RACE_LIVE_ODDS":raise ValueError("NOT_LIVE_PRE_RACE_ODDS")
 if not aware_iso(batch.get("fetched_at_utc")):raise ValueError("FETCHED_TIME_MISSING")
 if batch.get("page_updated_at") is not None and not aware_iso(batch["page_updated_at"]):
  raise ValueError("UNVERIFIED_UPDATED_TIME")
 quotes=batch.get("quotes")
 if not isinstance(quotes,list) or not quotes:raise ValueError("NO_REAL_ODDS_QUOTES")
 keys=set()
 result=[]
 frames={n:0 for n in range(1,9)}
 for x in CARD.values():frames[x["gate"]]+=1
 for q in quotes:
  typ=q.get("bet_type")
  if typ not in BET_TYPES:raise ValueError("BET_TYPE_UNSUPPORTED")
  n,ordered,ranged=BET_TYPES[typ]
  combination=q.get("combination")
  if not isinstance(combination,list) or len(combination)!=n or any(type(v)!=int for v in combination):
   raise ValueError("COMBINATION_LENGTH_OR_TYPE")
  limit=8 if typ=="BRACKET_QUINELLA" else 14
  if any(not 1<=v<=limit for v in combination):raise ValueError("COMBINATION_NUMBER_OUT_OF_RANGE")
  if typ=="BRACKET_QUINELLA":
   if len(set(combination))==1 and frames[combination[0]]<2:raise ValueError("IMPOSSIBLE_SAME_FRAME")
  elif len(set(combination))!=n:raise ValueError("DUPLICATE_HORSE_NUMBER")
  if not ordered and tuple(combination)!=tuple(sorted(combination)):raise ValueError("UNORDERED_COMBINATION_NOT_SORTED")
  k=(typ,tuple(combination))
  if k in keys:raise ValueError("DUPLICATE_QUOTE")
  keys.add(k)
  low=q.get("odds") if not ranged else q.get("odds_min")
  high=q.get("odds") if not ranged else q.get("odds_max")
  if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (low,high)):
   raise ValueError("PUBLISHED_ODDS_MISSING_OR_INVALID")
  if high<low:raise ValueError("INVALID_ODDS_RANGE")
  result.append({"bet_type":typ,"combination":combination,
    "odds":float(low) if not ranged else None,
    "odds_min":float(low) if ranged else None,"odds_max":float(high) if ranged else None})
 return {"contract":"FUJI_PREBET_P4_VALIDATED_ODDS_V1","race_id":RID,
   "source_url":batch["source_url"],"source_last_modified":batch.get("page_updated_at"),
   "fetched_at_utc":batch["fetched_at_utc"],
   "timestamp_is_observation_only":batch.get("page_updated_at") is None,
   "records":len(result),"by_bet_type":{typ:sum(x["bet_type"]==typ for x in result) for typ in BET_TYPES},
   "quotes":result}
