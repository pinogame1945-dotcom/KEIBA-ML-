#!/usr/bin/env python3
"""Convert original KEIBA-MOBILE OddsRow export into an L2.1 quote-book.

Requires an independently supplied final-odds check envelope; cannot infer
source is not final merely from the status of an already completed race.
Only QUINELLA, TRIO, TRIFECTA. No missing-combination fabrication.
"""
from __future__ import annotations
import argparse,importlib.util,json
from pathlib import Path
P=Path(__file__).with_name("l21-frozen-prebet-price.py")
spec=importlib.util.spec_from_file_location("keiba_l21",P)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

TYPES=("QUINELLA","TRIO","TRIFECTA")

def convert(payload):
    race,rid,active,start=mod.validate_live_card(payload)
    provenance=payload.get("capture")
    if not isinstance(provenance,dict):
        raise ValueError("MISSING_INDEPENDENT_JRA_FINAL_STATUS_CAPTURE")
    if (provenance.get("source_system")!="KEIBA_MOBILE_JRA"
        or provenance.get("final_probe_checked") is not True
        or provenance.get("is_final") is not False
        or provenance.get("source_status")!="LIVE"):
        raise ValueError("FINAL_OR_UNVERIFIED_CAPTURE")
    rows=payload.get("odds_rows")
    if not isinstance(rows,list):raise ValueError("MISSING_ORIGINAL_ODDSROWS")
    wins=[r for r in rows if r.get("betType")=="WIN"]
    if len(wins)!=len(active):
        raise ValueError("MISSING_FULL_SAME_CAPTURE_WIN_ODDS")
    win_numbers=[r.get("selection1") for r in wins]
    if (any(type(n) is not int for n in win_numbers)
        or set(win_numbers)!=active or len(set(win_numbers))!=len(wins)):
        raise ValueError("WIN_PRICE_HORSE_NUMBER_DRIFT")
    if any(r.get("selection2") is not None or r.get("selection3") is not None
           for r in wins):
        raise ValueError("INVALID_WIN_PRICE_ARITY")
    combos=[r for r in rows if r.get("betType") in TYPES]
    if not combos or set(r["betType"] for r in combos)!=set(TYPES):
        raise ValueError("MISSING_ONE_OR_MORE_L21_BET_TYPES")
    times={str(r.get("observedAt") or "") for r in combos+wins}
    if len(times)!=1:raise ValueError("ODDS_TIMES_MIXED")
    stamp=next(iter(times))
    observed=mod.timestamp(stamp,"original_oddrow_observed")
    cutoff=mod.timestamp(provenance.get("bet_cutoff_at"),"declared_bet_cutoff")
    checked=mod.timestamp(provenance.get("final_probe_checked_at"),"final_probe")
    if not observed<=checked<cutoff<start:
        raise ValueError("FINAL_PROBE_ODDS_OR_CUTOFF_NOT_IN_TIME")
    if any(r.get("raceKey")!=race["raceKey"] for r in combos+wins):
        raise ValueError("ODDS_RACE_KEY_MISMATCH")
    if any(r.get("sourceUrl")!=race.get("sourceUrl") for r in combos+wins):
        raise ValueError("ODDS_JRA_RACE_CARD_SOURCE_URL_MISMATCH")
    out={
      "contract":"KEIBA_JRA_COMBINATION_ODDS_ASOF_V1",
      "canonical_race_id":rid,"raceKey":race["raceKey"],
      "source_system":"KEIBA_MOBILE_JRA",
      "source_phase":"PRE_RACE",
      "is_final":False,"source_status":"LIVE",
      "source_confirmed_prestart":True,
      "observed_at":stamp,
      "bet_cutoff_at":provenance["bet_cutoff_at"],
      "scheduled_start_at":f'{race["raceDate"]}T{race["startTime"]}:00+09:00',
      "rows":combos,
      "win_odds_rows":wins,
      "verification_note":"The JRA final-state probe and timestamps are asserted by the offline export. Authenticity and a server-side timestamp are not independently attested."
    }
    mod.combine_quote(out,rid,active,race["raceKey"],start)
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mobile-export",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    q=convert(json.loads(Path(a.mobile_export).read_text(encoding="utf-8")))
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(q,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("JRA_MOBILE_EXACT_COMBINATION_QUOTE_BOOK_READY",q["canonical_race_id"],
          len(q["rows"]),q["observed_at"])
if __name__=="__main__":main()
