#!/usr/bin/env python3
"""Read-only replay of the frozen L2.1 archived horse ranks and ticket portfolio.

No models are fitted; do not regenerate frozen tickets or infer historical PREBET.
Uses only original 2023-25 OOS scores and L2.1 ticket trust sources, including
all 215 UNKNOWN p3 tickets. Actual settlement data is NEVER read for scoring.
Official operational V3 reverse-p3 stake order is called, not reimplemented.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
from l3_v3_operational import reverse_p3_stakes

EXPECTED_COUNTS={2023:4363,2024:4584,2025:4532}
EXPECTED_TOTAL=13479
KNOWN_P3=13264
UNKNOWN_P3=215
CONTRACT="KEIBA_L21_FROZEN_FINAL_ARCHIVE_V3_REPLAY_V1"


def stream(path):
    path=Path(path)
    open_file=gzip.open if path.suffix==".gz" else open
    with open_file(path,"rt",encoding="utf-8",newline="") as f:
        yield from csv.DictReader(f)


def checked_id(raw):
    if not isinstance(raw,str) or not re.fullmatch(r"202[345][0-9]{8}",raw):
        raise ValueError("FROZEN_HISTORICAL_ONLY_2023_2025_2026_SEALED")
    return raw


def finite(raw,name,lower=0,upper=1e12):
    try: val=float(raw)
    except (TypeError,ValueError) as exc:
        raise ValueError("INVALID_"+name) from exc
    if not (math.isfinite(val) and lower<=val<=upper):
        raise ValueError("INVALID_"+name)
    return val


def source_rows(known_file,missing_file):
    seen=set()
    counts=Counter()
    missing=0
    known=0
    bought=defaultdict(list)
    for file,with_p3 in ((known_file,True),(missing_file,False)):
        for r in stream(file):
            rid=checked_id(r.get("race_id"))
            year=int(r["year"])
            if year!=int(rid[:4]):
                raise ValueError("FROZEN_TICKET_YEAR_DRIFT")
            bt=r.get("bet_type")
            if bt not in ("QUINELLA","TRIO","TRIFECTA"):
                raise ValueError("FROZEN_TICKET_BET_TYPE_DRIFT")
            sel=r.get("selection","")
            parts=sel.split("-")
            if len(parts)!=(2 if bt=="QUINELLA" else 3) or any(
                not x.isdigit() or not 1<=int(x)<=24 for x in parts
            ) or len(set(parts))!=len(parts):
                raise ValueError("FROZEN_TICKET_SELECTION_DRIFT")
            if bt!="TRIFECTA" and list(map(int,parts))!=sorted(map(int,parts)):
                raise ValueError("FROZEN_TICKET_CANONICAL_ORDER_DRIFT")
            key=(rid,bt,sel)
            if key in seen:
                raise ValueError("FROZEN_DUPLICATE_PHYSICAL_TICKET")
            seen.add(key)
            p3=finite(r.get("min_p3"),"FROZEN_MIN_P3",0,1) if with_p3 else None
            if with_p3: known+=1
            else: missing+=1
            bought[rid].append(({"bet_type":bt,"canonical_selection":sel,
                                  "selection_numbers":list(map(int,parts))},p3))
            counts[year]+=1
    if dict(counts)!=EXPECTED_COUNTS or known!=KNOWN_P3 or missing!=UNKNOWN_P3 or len(seen)!=EXPECTED_TOTAL:
        raise ValueError("FROZEN_L2_COUNTS_OR_215_UNKNOWN_DRIFT "+
                         str(dict(counts))+" known "+str(known)+" missing "+str(missing))
    return bought


def load_picks(path, requested):
    picks=defaultdict(dict)
    for r in stream(path):
        rid=str(r.get("_race_id",""))
        if rid not in requested or r.get("strategy")!="ANCHOR_FULL":
            continue
        horse=str(r["_horse_id"])
        if rid in picks:
            raise ValueError("FROZEN_ANCHOR_DUPLICATE_FOR_RACE")
        picks[rid]=dict(horse_id=horse,
                        value=finite(r["selected_value"],"ANCHOR_SELECTED_VALUE"))
    if set(picks)!=set(requested):
        raise ValueError("FROZEN_ANCHOR_PICK_MISSING")
    return picks


def load_original_horses(path,ability_cache,requested):
    # Saved 2025 OOS compact predictions intentionally omit distill_score_z;
    # recover it from the original frozen fair-base cache by exact two-key join.
    # Never infer ability from rank, 7 kings, or any 2026 targets.
    z_by_pair={}
    for h in stream(ability_cache):
        rid=str(h.get("_race_id",""))
        if rid not in requested: continue
        k=(rid,str(h.get("_horse_id","")))
        if k in z_by_pair: raise ValueError("FROZEN_ABILITY_Z_DUPLICATE")
        z_by_pair[k]=finite(h.get("distill_score_z"),"ORIGINAL_ABILITY_Z",-10000,10000)
    groups=defaultdict(list)
    date_by_race={}
    for r in stream(path):
        rid=str(r.get("_race_id",""))
        if rid not in requested: continue
        if int(r.get("year",0))!=int(rid[:4]):
            raise ValueError("OOS_MODEL_YEAR_DRIFT")
        date=str(r.get("_race_date",""))[:10]
        if rid in date_by_race and date_by_race[rid]!=date:
            raise ValueError("OOS_RACE_DATE_DRIFT")
        date_by_race[rid]=date
        p=finite(r.get("P__ANCHOR_FULL"),"ORIGINAL_PWIN",0,1)
        odds=finite(r.get("final_win_odds"),"RESEARCH_FINAL_ODDS",1)
        rank=int(finite(r.get("distill_ai_rank"),"ORIGINAL_AI_RANK",1,24))
        mr=int(finite(r.get("market_rank"),"ORIGINAL_MARKET_RANK",1,24))
        risk=finite(r.get("risk"),"ORIGINAL_RISK",0,1)
        hid=str(r.get("_horse_id",""))
        if (rid,hid) not in z_by_pair:
            raise ValueError("FROZEN_ABILITY_Z_JOIN_MISSING")
        score=z_by_pair[(rid,hid)]
        if not hid or any(x["horse_id"]==hid for x in groups[rid]):
            raise ValueError("OOS_DUPLICATE_HORSE_ID")
        groups[rid].append(dict(horse_id=hid,
                                P__ANCHOR_FULL=p,final_win_odds=odds,
                                distill_ai_rank=rank,market_rank=mr,
                                risk=risk,distill_score_z=score,
                                value=p*odds))
    if set(groups)!=set(requested):
        raise ValueError("OOS_REQUESTED_RACE_NOT_FOUND")
    for rid in requested:
        if set(h["horse_id"] for h in groups[rid])!=set(
            hid for rr,hid in z_by_pair if rr==rid):
            raise ValueError("FROZEN_OOS_ABILITY_HORSE_SET_DRIFT")
    return groups,date_by_race


def run(ids,known,missing,oos,picks,ability_cache,race_budget_yen,bankroll_yen):
    requested=[checked_id(x) for x in ids]
    if len(requested)!=len(set(requested)) or not requested:
        raise ValueError("EMPTY_OR_DUPLICATE_RACE_IDS")
    # Reject invalid caps before loading potentially large source archives.
    for v,label in ((race_budget_yen,"RACE_BUDGET"),(bankroll_yen,"BANKROLL")):
        if type(v) is not int or v<0 or v%100:
            raise ValueError("INVALID_"+label)
    portfolios=source_rows(known,missing)
    selected=set(requested)
    axes=load_picks(picks,selected)
    horses,dates=load_original_horses(oos,ability_cache,selected)
    day_races=defaultdict(list)
    output=[]
    for rid in requested:
        group=horses[rid]
        by_id={h["horse_id"]:h for h in group}
        if len(group)<3 or len(group)>24:
            raise ValueError("HISTORICAL_OOS_STARTER_POPULATION_INVALID")
        if sorted(h["distill_ai_rank"] for h in group)!=list(range(1,len(group)+1)):
            raise ValueError("FROZEN_ORIGINAL_RAW_RANK_DRIFT")
        raw=sorted(group,key=lambda h:(h["distill_ai_rank"],h["market_rank"],h["horse_id"]))
        calibrated=sorted(group,key=lambda h:(-h["P__ANCHOR_FULL"],h["market_rank"],h["horse_id"]))
        ev=sorted(group,key=lambda h:(-h["value"],-h["P__ANCHOR_FULL"],h["market_rank"],h["horse_id"]))
        if axes[rid]["horse_id"]!=ev[0]["horse_id"] or abs(axes[rid]["value"]-ev[0]["value"])>1e-6:
            raise ValueError("FROZEN_ANCHOR_FULL_PICK_NOT_REPRODUCED")
        ticket_rows=portfolios.get(rid,[])
        n=len(ticket_rows)
        budget=min(race_budget_yen,200*n)
        if budget<100*n:
            status="HISTORICAL_STAKE_BUDGET_INSUFFICIENT"
            tickets=[]
        else:
            stakes=reverse_p3_stakes(ticket_rows,budget) if n else {}
            tickets=[{
                "bet_type":t["bet_type"],"canonical_selection":t["canonical_selection"],
                "stake_yen":stakes[(t["bet_type"],t["canonical_selection"])],
                "min_p3":p3,"p3_missing":p3 is None,
                "historical_price_phase":"FINAL_RESEARCH_ONLY"
            } for t,p3 in ticket_rows]
            status="HISTORICAL_FINAL_ONLY_UNAVAILABLE_AS_PREBET"
        if len(tickets)!=n and status=="HISTORICAL_FINAL_ONLY_UNAVAILABLE_AS_PREBET":
            raise AssertionError("L2_TICKET_PRESERVATION_FAILURE")
        day_races[dates[rid]].append(rid)
        output.append({
            "race_id":rid,"race_date":dates[rid],"status":status,
            "original_anchor_horse_id":ev[0]["horse_id"],
            "rankings":{
                "ai_raw":[{"rank":i,"horse_id":h["horse_id"],"distill_ai_rank":h["distill_ai_rank"]}
                          for i,h in enumerate(raw,1)],
                "ai_prob":[{"rank":i,"horse_id":h["horse_id"],"original_pwin":h["P__ANCHOR_FULL"]}
                           for i,h in enumerate(calibrated,1)],
                "value_ev":[{"rank":i,"horse_id":h["horse_id"],"historical_final_value":h["value"]}
                            for i,h in enumerate(ev,1)]
            },
            "original_risk_by_horse_id":{h["horse_id"]:h["risk"] for h in group},
            "original_ability_z_by_horse_id":{h["horse_id"]:h["distill_score_z"] for h in group},
            "frozen_physical_ticket_count":n,
            "missing_l175_ticket_p3":sum(p3 is None for _,p3 in ticket_rows),
            "l3_v3_policy":"REVERSE_P3_RACE_MAX300",
            "l3_v3_total_yen":sum(t["stake_yen"] for t in tickets),
            "tickets":tickets
        })
    by_id={r["race_id"]:r for r in output}
    days=[]
    for date,selected_races in sorted(day_races.items()):
        required=sum(by_id[rid]["l3_v3_total_yen"] for rid in selected_races)
        funded=(required<=bankroll_yen and all(
            by_id[rid]["status"]=="HISTORICAL_FINAL_ONLY_UNAVAILABLE_AS_PREBET"
            for rid in selected_races))
        if not funded:
            for rid in selected_races:
                by_id[rid]["tickets"]=[]
                by_id[rid]["l3_v3_total_yen"]=0
                by_id[rid]["status"]="HISTORICAL_V3_SAME_DAY_NOT_FUNDED"
        days.append({"race_date":date,"total_requested_before_payout_yen":required,
                     "available_paper_bankroll_yen":bankroll_yen,
                     "status":"PAPER_FUNDED_FINAL_ONLY" if funded else "PAPER_BLOCKED",
                     "same_day_payouts_credited_yen":0})
    return {"contract":CONTRACT,"status":"HISTORICAL_FINAL_RESEARCH_ONLY",
            "source_phase":"FINAL_RESEARCH_ARCHIVED_NOT_AT_BET_CUTOFF",
            "genuinely_prebet_validated":False,
            "historical_2023_2025_L2_selection_was_hindsight":True,
            "original_models_retrained":False,
            "year_2026_results_read":False,
            "real_money_bets_placed":False,
            "complete_same_day_JRA_schedule_certified":False,
            "rankings_source":"ORIGINAL_FROZEN_OOS_SCORES",
            "tickets_source":"ORIGINAL_FROZEN_L21_13479_PHYSICAL_TICKETS",
            "ticket_p3_source":"L17_SAVED_TOP10_P3; 215 UNKNOWN TICKETS PRESERVED",
            "ranking_without_horse_names_or_nos":True,
            "races":output,"daily_research_bankroll":days}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--race",action="append",required=True)
    p.add_argument("--frozen-oos",required=True)
    p.add_argument("--frozen-picks",required=True)
    p.add_argument("--frozen-ability-cache",required=True)
    p.add_argument("--frozen-known-p3-tickets",required=True)
    p.add_argument("--frozen-unknown-p3-tickets",required=True)
    p.add_argument("--race-budget-yen",type=int,default=20000)
    p.add_argument("--paper-bankroll-yen",type=int,default=100000)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    data=run(a.race,a.frozen_known_p3_tickets,a.frozen_unknown_p3_tickets,
             a.frozen_oos,a.frozen_picks,a.frozen_ability_cache,a.race_budget_yen,a.paper_bankroll_yen)
    target=Path(a.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for r in data["races"]:
        print("FROZEN_REPLAY_RACE",r["race_id"],r["status"],
              "original_AI_RAW",r["rankings"]["ai_raw"][0]["horse_id"],
              "frozen_anchor",r["original_anchor_horse_id"],
              "tickets",r["frozen_physical_ticket_count"],
              "L3_paper_yen",r["l3_v3_total_yen"])
    print("FROZEN_ARCHIVE_REPLAY_REAL_SCORES_NO_NEW_MODEL",
          len(data["races"]),data["source_phase"])


if __name__=="__main__":
    main()
