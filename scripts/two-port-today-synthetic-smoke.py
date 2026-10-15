#!/usr/bin/env python3
"""Two synthetic pre-start KEIBA-MOBILE race cards through new today-ID port.

This proves the routing and L2/L3 geometry ONLY. No real publisher as-of
attestation and no actual future race prediction.
"""
from __future__ import annotations
import datetime as dt
import importlib.util
import itertools
import json
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("today_ports",ROOT/"research/execute-races-by-id.py")
port=importlib.util.module_from_spec(spec);spec.loader.exec_module(port)
jst=dt.timezone(dt.timedelta(hours=9))
now=dt.datetime.now(jst).replace(microsecond=0)
if now.hour==23 and now.minute>=30:
    # No race can remain in the current JST day past midnight.
    print("TWO_PORT_TODAY_SYNTH_SKIPPED_END_OF_JST_DAY")
    raise SystemExit(0)
start=now+dt.timedelta(minutes=55)
cutoff=now+dt.timedelta(minutes=40)
observed=now-dt.timedelta(seconds=70)
probe=now-dt.timedelta(seconds=25)
model_at=now-dt.timedelta(seconds=10)
features_at=now-dt.timedelta(minutes=5)
rids=["202606050811","202606050812"]
with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    capture=root/"captures";capture.mkdir()
    def row(key,url,bet,seq,price):
        return {"raceKey":key,"betType":bet,"selection1":seq[0],
                "selection2":seq[1] if len(seq)>1 else None,
                "selection3":seq[2] if len(seq)>2 else None,
                "odds":price,"oddsMin":None,"oddsMax":None,
                "observedAt":observed.isoformat(),"sourceUrl":url}
    for rid in rids:
        key="SYNTH_RACE_"+rid
        url="https://www.jra.go.jp/test"
        horses=range(1,9)
        entries=[{"raceKey":key,"horseNo":n,"canonicalHorseId":f"{rid}horse{n}",
                  "entryStatus":"ACTIVE"} for n in horses]
        odds=[row(key,url,"WIN",(n,),float(n+2)) for n in horses]
        odds +=[row(key,url,"QUINELLA",(a,b),16.0)
                for a,b in itertools.combinations(horses,2)]
        odds +=[row(key,url,"TRIO",(a,b,c),16.0 if (b+c)%2 else 35.0)
                for a,b,c in itertools.combinations(horses,3)]
        odds +=[row(key,url,"TRIFECTA",(a,b,c),
                   240.0 if a==1 else 25.0 if c==1 else 500.0)
                for a,b,c in itertools.permutations(horses,3)]
        assert len(odds)==428
        card={
            "race":{"raceKey":key,"canonicalRaceId":rid,
                    "raceDate":now.date().isoformat(),
                    "startTime":start.strftime("%H:%M"),
                    "raceStatus":"SCHEDULED","scheduleStatus":"ACTIVE",
                    "status":"OFFICIAL","sourceUrl":url},
            "entries":entries,
            "capture":{"source_system":"KEIBA_MOBILE_JRA",
                       "final_probe_checked":True,"is_final":False,
                       "source_status":"LIVE",
                       "final_probe_checked_at":probe.isoformat(),
                       "bet_cutoff_at":cutoff.isoformat()},
            "odds_rows":odds
        }
        original={
            "contract":"KEIBA_L21_ORIGINAL_ANCHOR_FEATURES_PREBET_V1",
            "race_id":rid,"canonical_race_key":key,
            "anchor_source":"EXACT_EXISTING_ANCHOR_FULL",
            "anchor_horse_id":f"{rid}horse1",
            "feature_source":"ORIGINAL_L21_HORSE_FEATURES","input_phase":"PRE_RACE",
            "features_asof_asserted":True,
            "features_observed_at":features_at.isoformat(),
            "win_odds_observed_at":observed.isoformat(),
            "anchor_computed_at":model_at.isoformat(),
            "horses":[{"horse_id":f"{rid}horse{n}","horse_number":n,
                "P__ANCHOR_FULL":0.31 if n==1 else 0.14-n*0.005,
                "risk":0.09,"distill_score_z":0.45,
                "distill_ai_rank":{2:3,3:4,4:5,5:2}.get(n,n),
                "market_rank":n,"market_prob_norm":0.15,
                "win_odds_at_cutoff":float(n+2)} for n in horses]
        }
        (capture/f"{rid}.json").write_text(json.dumps(card),encoding="utf-8")
        (capture/f"{rid}-original-features.json").write_text(json.dumps(original),encoding="utf-8")
    lim=root/"limits.json"
    lim.write_text(json.dumps({
        "contract":"KEIBA_L3_FIXED100_LIMITS_V1",
        "race_budget_yen":10000,"max_race_exposure_yen":10000,
        "max_ticket_count":1000,
        "stake_policy":"BASELINE_FLAT_100_ALL_OR_NONE"}),encoding="utf-8")
    results=port.today(rids,capture,root/"out",lim)
    assert len(results)==2,results
    for item in results:
        assert item["status"]=="PAPER_ONLY_LIMITS_CHECKED",item
        assert item["l3_status"]=="PAPER_ONLY_BASELINE_READY",item
        assert item["paper_ticket_count"]>0 and item["paper_stake_yen"]>0,item
        assert item["actual_bets_placed"] is False
        print("TODAY_ID_SYNTHETIC_L2_L3_PASS",item["race_id"],
              "tickets",item["paper_ticket_count"],"yen",item["paper_stake_yen"],flush=True)
    # Correctly refuse identical-looking previously started date in
    # TODAY port. Do not allow a historical file to masquerade as live.
    changed=json.loads((capture/f"{rids[0]}.json").read_text())
    changed["race"]["raceDate"]="2025-12-28"
    (capture/f"{rids[0]}.json").write_text(json.dumps(changed),encoding="utf-8")
    blocked=port.today([rids[0]],capture,root/"stale",lim)[0]
    assert blocked["status"]=="BLOCKED_ORIGINAL_PREBET_CONTRACT",blocked
    assert "TODAY_PORT_REQUIRES_CURRENT_JST_DATE" in blocked["reason"],blocked
print("TODAY_MULTI_RACE_ID_TO_L3_SYNTHETIC_ROUTE_AND_DATE_GATES_PASS",flush=True)
