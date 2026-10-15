#!/usr/bin/env python3
"""Bridge existing KEIBA-MOBILE JraRace + OddsRow records to ML market contract.

Takes a locally exported JSON {race, entries, odds_rows}; no scraped HTML,
no net requests, no guessing a race or horse identity, no post-race fallback.
This is NOT a historical FINAL odds import.
"""
from __future__ import annotations
import argparse,datetime,json,re
from pathlib import Path

from market_context_v1 import CONTRACT,attach_win_market,read_iso

def from_mobile(payload):
    if not isinstance(payload,dict):raise ValueError("INVALID_MOBILE_PAYLOAD")
    race=payload.get("race")
    if not isinstance(race,dict):raise ValueError("MISSING_JRA_RACE")
    rid=str(race.get("canonicalRaceId") or "")
    if not re.fullmatch(r"202[1-6]\d{8}",rid):
        raise ValueError("CANONICAL_NETKEIBA_RACE_ID_REQUIRED_2026_SEALED")
    if race.get("raceStatus")!="SCHEDULED" or race.get("scheduleStatus")!="ACTIVE":
        raise ValueError("CANCELLED_FINISHED_OR_RESCHEDULED_RACE_FORBIDDEN")
    if race.get("status")!="OFFICIAL":
        raise ValueError("RACE_CARD_NOT_YET_OFFICIAL")
    iso_date=race.get("raceDate")
    time=race.get("startTime")
    if not isinstance(iso_date,str) or not re.fullmatch(r"\d{4}-\d\d-\d\d",iso_date):
        raise ValueError("MISSING_RACE_DATE")
    if not isinstance(time,str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d",time):
        raise ValueError("MISSING_SCHEDULED_START_CLOCK")
    scheduled=f"{iso_date}T{time}:00+09:00"
    read_iso(scheduled,"scheduled_start_at")
    entries=payload.get("entries")
    if not isinstance(entries,list) or len(entries)<3:
        raise ValueError("MISSING_NUMBERED_ENTRY_CARD")
    nums=[]
    for e in entries:
        if e.get("raceKey")!=race.get("raceKey"):
            raise ValueError("ENTRY_RACE_KEY_DRIFT")
        if e.get("entryStatus")=="ACTIVE":
            n=e.get("horseNo")
            if isinstance(n,bool) or not isinstance(n,int) or not 1<=n<=24:
                raise ValueError("ACTIVE_ENTRY_NOT_NUMBERED")
            nums.append(n)
    if len(nums)<3 or len(nums)!=len(set(nums)):
        raise ValueError("AMBIGUOUS_ACTIVE_ENTRY_NUMBERS")
    rows=payload.get("odds_rows")
    if not isinstance(rows,list) or not rows:
        raise ValueError("MISSING_JRA_WIN_ODDS")
    win=[r for r in rows if r.get("betType")=="WIN"]
    if not win:raise ValueError("MISSING_WIN_PRODUCT")
    quoted_numbers=[r.get("selection1") for r in win]
    if (any(isinstance(n,bool) or not isinstance(n,int) for n in quoted_numbers)
        or len(quoted_numbers)!=len(nums)
        or set(quoted_numbers)!=set(nums)):
        raise ValueError("PARTIAL_OR_EXTRANEOUS_WIN_QUOTE_VS_ACTIVE_ENTRIES")
    when={str(r.get("observedAt") or "") for r in win}
    if len(when)!=1:raise ValueError("MIXED_WIN_CAPTURE_TIMES")
    captured=next(iter(when))
    if not read_iso(captured,"win_observed_at")<read_iso(scheduled,"scheduled_start_at"):
        raise ValueError("MOBILE_WIN_ODDS_POST_START")
    if any(r.get("raceKey")!=race["raceKey"] for r in win):
        raise ValueError("ODDS_RACE_KEY_DRIFT")
    return {
        "contract":CONTRACT,
        "canonical_race_id":rid,
        "source_system":"KEIBA_MOBILE_JRA",
        "source_phase":"PRE_RACE",
        "is_final":False,
        "source_status":"LIVE",
        "observed_at":captured,
        "scheduled_start_at":scheduled,
        "rows":win,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mobile-export",required=True,
                    help="Offline JSON {race:JraRace, entries:JraEntry[], odds_rows:OddsRow[]}")
    ap.add_argument("--analysis",required=True,
                    help="Already produced one-race analysis.json with matching canonical IDs")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    source=json.loads(Path(args.mobile_export).read_text(encoding="utf-8"))
    quote=from_mobile(source)
    analysis=json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    result=attach_win_market(analysis,quote)
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("JRA_MOBILE_TO_ONE_RACE_MARKET_COMPLETE",result["race_id"],
          len(result["horses"]),"full quoted active field only")
if __name__=="__main__":main()
