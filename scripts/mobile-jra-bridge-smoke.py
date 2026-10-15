#!/usr/bin/env python3
"""KEIBA-MOBILE JraRace/JraEntry/OddsRow structural adapter safety tests."""
from __future__ import annotations
import copy,importlib.util,sys
from pathlib import Path
sys.path.insert(0,str(Path("research").resolve()))
p=Path("research/bridge-keiba-mobile-odds.py").resolve()
spec=importlib.util.spec_from_file_location("keiba_mobile_bridge",p)
bridge=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
rowtime="2025-01-06T13:00:00+09:00"
payload={
    "race":{"raceKey":"2025_JRA_01_02_12","canonicalRaceId":"202507010212",
        "raceDate":"2025-01-06","startTime":"14:30","raceStatus":"SCHEDULED",
        "scheduleStatus":"ACTIVE","status":"OFFICIAL"},
    "entries":[{"raceKey":"2025_JRA_01_02_12","entryStatus":"ACTIVE","horseNo":i}
               for i in (1,2,3)]+
              [{"raceKey":"2025_JRA_01_02_12","entryStatus":"SCRATCHED","horseNo":4}],
    "odds_rows":[{"raceKey":"2025_JRA_01_02_12","betType":"WIN",
        "selection1":i,"selection2":None,"selection3":None,
        "odds":float(i+1),"oddsMin":None,"oddsMax":None,
        "observedAt":rowtime,"sourceUrl":"https://www.jra.go.jp/JRADB/odds/"}
        for i in (1,2,3)]+
        [{"raceKey":"2025_JRA_01_02_12","betType":"TRIO",
          "selection1":1,"selection2":2,"selection3":3,
          "odds":9.9,"observedAt":rowtime,"sourceUrl":"https://www.jra.go.jp/"}]
}
good=bridge.from_mobile(payload)
assert good["contract"]=="KEIBA_JRA_WIN_ODDS_ASOF_V1"
assert len(good["rows"])==3
assert good["is_final"] is False
assert good["scheduled_start_at"]=="2025-01-06T14:30:00+09:00"

def bad(fn,code):
    x=copy.deepcopy(payload);fn(x)
    try:bridge.from_mobile(x)
    except ValueError as e:assert code in str(e),f"{code} vs {e}"
    else:raise AssertionError("unsafe JRA mobile odds allowed "+code)

bad(lambda m:m["race"].update(canonicalRaceId=None),"CANONICAL_NETKEIBA")
bad(lambda m:m["race"].update(raceStatus="COMPLETED"),"CANCELLED_FINISHED")
bad(lambda m:m["race"].update(scheduleStatus="RESCHEDULED"),"CANCELLED_FINISHED")
bad(lambda m:m["race"].update(status="OFFICIAL"),"ALREADY_OFFICIAL")
bad(lambda m:m["race"].update(startTime=None),"SCHEDULED_START")
bad(lambda m:m["entries"][0].update(horseNo=None),"ACTIVE_ENTRY_NOT_NUMBERED")
bad(lambda m:m["odds_rows"][0].update(raceKey="WRONG"),"ODDS_RACE_KEY_DRIFT")
bad(lambda m:m["odds_rows"][0].update(observedAt="2025-01-06T13:10:00+09:00"),"MIXED_WIN_CAPTURE_TIMES")
bad(lambda m:m["odds_rows"][0].update(observedAt="2025-01-06T15:00:00+09:00"),"MIXED_WIN_CAPTURE_TIMES")
bad(lambda m:m["odds_rows"][0].update(selection1=4),"PARTIAL_OR_EXTRANEOUS")
bad(lambda m:m["odds_rows"].pop(0),"PARTIAL_OR_EXTRANEOUS")
bad(lambda m:m["race"].update(canonicalRaceId="202607010212"),"2026_SEALED")
print("KEIBA_MOBILE_JRA_WIN_MARKET_BRIDGE_TEST_PASS")
