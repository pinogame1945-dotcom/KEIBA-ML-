#!/usr/bin/env python3
"""Rank1 replay of frozen L2.1 STANDARD alongside fixed rank2-6 results."""

from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import pandas as pd
import run_l2_ev_rank2to6_pruning20_best_roi_v1 as ev
import run_l2_all_fixed_odds_band_lenses_v1 as lens
import run_l2_fixed_ticket_pruning_3040_v1 as fixed
import run_l2_hindsight_stable_black_union_v1 as frozen
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,assert_ok

def parse_line(line):
    v=line.split("|")
    assert_ok(len(v)==4,("invalid frozen rule",line))
    return v[0],int(v[1]),v[2],v[3]

def frozen_standard(path):
    o=json.loads(Path(path).read_text(encoding="utf-8"))
    assert_ok(o["contract"]=="KEIBA_L2_1_FIXED_PURCHASE_V1"
        and o["version"]=="2.1.0" and o["primary_policy"]=="STANDARD",
        "L2.1 contract/version mismatch")
    assert_ok(o["ticket_policy"]["anchor"].startswith("EXACT_EXISTING_ANCHOR_FULL"),
              "frozen rank1 anchor mismatch")
    rows=o["frozen_rules"]["STANDARD"]
    assert_ok(len(rows)==o["rule_counts"]["STANDARD"]==50,"frozen 50 rules changed")
    selected=defaultdict(set)
    for item in rows:
        assert_ok(item["gate"] in lens.GATES,"frozen gate not found")
        selected[parse_line(item["line"])].add(item["gate"])
    assert_ok(len(selected)==9 and sum(map(len,selected.values()))==50,
              "frozen line/gate signature changed")
    return o,dict(selected)

def blank():
    return dict(tickets=0,hits=0,payout_yen=0,races_bought=0)

def count(counter,tickets):
    if tickets:
        counter["races_bought"]+=1
        for payout in tickets.values():
            counter["tickets"]+=1
            counter["hits"]+=int(payout>0)
            counter["payout_yen"]+=int(payout)

def result(year,method,st,available):
    n=st["tickets"];ret=st["payout_yen"]
    return dict(rank=1,method=method,period=str(year),
        available_races=available,races_bought=st["races_bought"],
        tickets=n,winning_tickets=st["hits"],stake_yen=n*100,
        return_yen=ret,profit_yen=ret-n*100,
        roi_pct=100*ret/(n*100) if n else None)
