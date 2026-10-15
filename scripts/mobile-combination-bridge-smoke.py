#!/usr/bin/env python3
"""JRA combination odds export adapter refuses anything not auditably prestart."""
from __future__ import annotations
import copy,importlib.util,sys
from pathlib import Path
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,Path(path).resolve())
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
sys.path.insert(0,str(Path("research").resolve()))
m=load("research/bridge-mobile-combination-odds.py","bridge_combo")
rid="202507010212";race_key="SYNTH_JRA_KEY";url="https://www.jra.go.jp/test"
payload={
    "race":{"raceKey":race_key,"canonicalRaceId":rid,"raceDate":"2025-01-06",
            "startTime":"14:30","raceStatus":"SCHEDULED","scheduleStatus":"ACTIVE",
            "status":"OFFICIAL","sourceUrl":url},
    "entries":[{"raceKey":race_key,"horseNo":i,"entryStatus":"ACTIVE"} for i in (1,2,3)],
    "capture":{"source_system":"KEIBA_MOBILE_JRA","final_probe_checked":True,
               "is_final":False,"source_status":"LIVE",
               "final_probe_checked_at":"2025-01-06T13:31:00+09:00",
               "bet_cutoff_at":"2025-01-06T14:25:00+09:00"},
    "odds_rows":[{"raceKey":race_key,"betType":bt,"selection1":a,"selection2":b,
                  "selection3":c,"observedAt":"2025-01-06T13:30:00+09:00",
                  "odds":v,"oddsMin":None,"oddsMax":None,"sourceUrl":url}
                 for bt,a,b,c,v in [
                   ("QUINELLA",1,2,None,17.0),
                   ("TRIO",1,2,3,35.0),
                   ("TRIFECTA",1,2,3,250.0)
                 ]]
}
out=m.convert(payload)
assert out["contract"]=="KEIBA_JRA_COMBINATION_ODDS_ASOF_V1"
assert len(out["rows"])==3 and out["source_confirmed_prestart"] is True
def reject(fn,why):
    data=copy.deepcopy(payload);fn(data)
    try:m.convert(data)
    except ValueError as exc:assert why in str(exc),(why,str(exc))
    else:raise AssertionError("BAD_CAPTURE_ACCEPTED "+why)
reject(lambda x:x.pop("capture"),"MISSING_INDEPENDENT_JRA_FINAL_STATUS_CAPTURE")
reject(lambda x:x["capture"].update(final_probe_checked=False),"FINAL_OR_UNVERIFIED")
reject(lambda x:x["capture"].update(is_final=True),"FINAL_OR_UNVERIFIED")
reject(lambda x:x["capture"].update(final_probe_checked_at="2025-01-06T15:00:00+09:00"),"FINAL_PROBE_ODDS")
reject(lambda x:x["odds_rows"][0].update(observedAt="2025-01-06T12:00:00+09:00"),"ODDS_TIMES_MIXED")
reject(lambda x:x["odds_rows"][0].update(raceKey="WRONG"),"ODDS_RACE_KEY_MISMATCH")
reject(lambda x:x["odds_rows"][0].update(sourceUrl="https://example.net/"),"ODDS_JRA_RACE_CARD_SOURCE_URL")
reject(lambda x:x["odds_rows"].pop(1),"MISSING_ONE_OR_MORE_L21_BET_TYPES")
reject(lambda x:x["race"].update(status="OFFICIAL"),"RESULT_OR_CANCELLED_RACE")
reject(lambda x:x["race"].update(canonicalRaceId="202607010212"),"2026_SEALED")
print("MOBILE_JRA_COMBINATION_PREBET_PROBE_AND_ODDSROW_EXPORT_PASS")
