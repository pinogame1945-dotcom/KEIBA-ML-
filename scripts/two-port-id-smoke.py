#!/usr/bin/env python3
"""Check BOTH entry points without network or real wagers."""
from __future__ import annotations
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("two_port",ROOT/"research/execute-races-by-id.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class Args:
    race_ids=["202506050811, 202507010212","202506050811"]
    ids_file=None
assert mod.parse_ids(Args())==["202506050811","202507010212"]
for selection in (["202606050811"],["202506050811","NOT_AN_ID"]):
    class Bad:pass
    arg=Bad();arg.race_ids=selection;arg.ids_file=None
    if selection==["202606050811"]:
        assert mod.parse_ids(arg)==selection
    else:
        try:mod.parse_ids(arg);raise AssertionError("bad id accepted")
        except ValueError:pass

config={
    "contract":"KEIBA_L3_FIXED100_LIMITS_V1",
    "race_budget_yen":1500,
    "max_race_exposure_yen":1500,
    "max_ticket_count":15,
    "stake_policy":"BASELINE_FLAT_100_ALL_OR_NONE",
}
tickets=[{"bet_type":"TRIFECTA","selection":s,"ticket_odds":x}
         for s,x in (("3-4-5",232.6),("3-4-9",285.4),("3-5-4",237.7),
                     ("3-5-9",250.1),("3-9-4",296.3),("3-9-5",271.3))]
archive={
    "race_id":"202506050811","source_odds_kind":"FINAL",
    "FORBIDDEN_FOR_LIVE_BETTING":True,
    "tickets":tickets,
}
r=mod.historical_l3_audit(archive,config)
assert r["paper_stake_yen"]==600 and r["paper_ticket_count"]==6
assert r["actual_L3_prebet_allocate_called"] is False
assert r["historical_final_odds_used"] is True
small=dict(config,race_budget_yen=500)
assert mod.historical_l3_audit(archive,small)["paper_stake_yen"]==0
try:
    mod.historical_l3_audit({**archive,"FORBIDDEN_FOR_LIVE_BETTING":False},config)
    raise AssertionError("historical market should never enter live L3")
except ValueError:pass

with tempfile.TemporaryDirectory() as td:
    tmp=Path(td)
    empty=mod.today(["202506050811","202507010212"],tmp/"captures",tmp/"out",tmp/"config.json")
    assert len(empty)==2
    assert all(x["status"]=="BLOCKED_MISSING_INPUTS" for x in empty),empty
    assert all("KEIBA_MOBILE_OFFICIAL_JRA_CAPTURE" in x["missing_inputs"] for x in empty)
    assert all(x["actual_bets_placed"] is False for x in empty)

print("TWO_PORT_REAL_ID_CONTRACTS_PASS single multi archive_cost today_failclosed no_pseudo_live",flush=True)
