#!/usr/bin/env python3
"""Diagnose five original OOS/actual BACKFILL runner-id drifts. No repair guess.

Compares original archived starting 2025 market correction panel, historical
race entry field, archived group-1 WIN price keys and snapshot raw metadata.
This is evidence inspection for 2025 ONLY, never a 2026 training input.
"""
from __future__ import annotations
import argparse,csv,gzip,json,sys
from collections import Counter
from pathlib import Path

MISSING={
"202506010203":"2022100262","202506010208":"2019106330",
"202506010209":"2018100247","202506010212":"2020100209",
"202507010203":"2022106707",
}
def panel(path):
    results={}
    with gzip.open(path,"rt",encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream):
            rid=row["_race_id"]
            if rid in MISSING:results.setdefault(rid,set()).add(row["_horse_id"])
    return results

def all_packs(path,ids,odds=False):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            record=json.loads(line)
            rid=str(record.get("race_id") if odds else (record.get("race") or {}).get("race_id"))
            if rid in ids:yield rid,record

def audit(pred,pack,oddspack):
    ids=set(MISSING)
    predicted=panel(pred)
    entries=dict(all_packs(pack,ids,False))
    prices=dict(all_packs(oddspack,ids,True))
    if set(entries)!=ids or set(prices)!=ids:
        raise ValueError("DAILY_PACK_TARGET_RACE_NOT_COMPLETE")
    cases=[]
    for rid,missing_id in MISSING.items():
        active=entries[rid].get("entries") or []
        exact=[x for x in active if str(x.get("horse_id"))==missing_id]
        if len(exact)!=1 or missing_id in predicted.get(rid,set()):
            raise ValueError("NOT_THE_SAME_FIVE_MISSING_HORSES")
        row=exact[0]
        no=int(row["horse_number"])
        race=entries[rid].get("race") or {}
        odds=prices[rid]
        winbook=(odds.get("odds") or {}).get("1") or {}
        match={k:v for k,v in winbook.items() if str(k).isdigit() and int(k)==no}
        # For historical investigation ONLY use statuses and pricing evidence.
        case={
           "race_id":rid,"horse_id":missing_id,"horse_number":no,
           "entry_status":row.get("entry_status"),
           "original_panel_horse_count":len(predicted.get(rid,set())),
           "backfill_start_entries":sum(x.get("entry_status")=="STARTED" for x in active),
           "backfill_numbered_entries":sum(x.get("horse_number") is not None for x in active),
           "odds_record_kind":odds.get("odds_kind"),
           "odds_record_source_status":odds.get("source_status"),
           "win_price_present_for_missing_horse":bool(match),
           "win_odds_raw_keys":sorted(match),
           "win_odds_raw_values":list(match.values()),
           "entry_nonresult_fields":{k:row.get(k) for k in (
               "horse_name","horse_id","horse_number","entry_status",
               "win_odds","popularity","odds","jockey_name","carried_weight")},
           "record_source_keys":sorted(x for x in odds if x in (
               "race_id","odds_kind","source_status","fetched_at","snapshot_at",
               "source_url","source","capture_at","odds")),
        }
        cases.append(case)
    return {"contract":"KEIBA_L21_OOS_MISSING_HORSE_DIAGNOSIS_2025_ONLY_V1",
        "year":2025,"races_checked":len(MISSING),"cases":cases,
        "no_silent_proxies":True,
        "no_2026_results":True,
        "not_repairing_original_frozen_training_dataset":True}

def main():
    p=argparse.ArgumentParser()
    for name in ("predictions","race-pack","odds-pack","output"):
        p.add_argument("--"+name,required=True)
    x=p.parse_args()
    z=audit(x.predictions,x.race_pack,x.odds_pack)
    dst=Path(x.output);dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(z,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for case in z["cases"]:
        print("STARTED_HORSE_OOS_MISSING",case["race_id"],case["horse_number"],
              case["horse_id"],"group1_odds_exists",case["win_price_present_for_missing_horse"],
              "price_keys",case["win_odds_raw_keys"],"entry",
              case["entry_nonresult_fields"],flush=True)
if __name__=="__main__":main()
