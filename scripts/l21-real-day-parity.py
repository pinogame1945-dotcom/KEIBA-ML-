#!/usr/bin/env python3
"""Broaden ORIGINAL frozen-L2.1 parity to ALL 2025-01-06 races, with the
same cached real feature, anchor, BACKFILL and FINAL odds inputs.
NO purchase price timing claim. Never score 2026.
"""
from __future__ import annotations
import argparse,gzip,importlib.util,json
from collections import Counter
from pathlib import Path

modfile=Path(__file__).resolve().parents[1]/"research/l21-replay-real-final-one-race.py"
spec=importlib.util.spec_from_file_location("real_l21",modfile)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def run_day(a):
    rows=[]
    with gzip.open(a.race_pack,"rt",encoding="utf-8") as src:
        for line in src:
            if not line.strip():continue
            race=json.loads(line).get("race") or {}
            rid=str(race.get("race_id") or "")
            if rid.startswith("2025") and race.get("race_date")=="2025-01-06":
                rows.append(rid)
    if len(rows)<8 or len(rows)!=len(set(rows)) or any(len(rid)!=12 for rid in rows):
        raise ValueError("PROVEN_HISTORICAL_DAY_RACE_SET_INCOMPLETE_OR_DUPLICATED")
    output=[]
    for rid in sorted(rows):
        result=m.replay_one(rid,a.predictions,a.ability_cache,a.picks,a.race_pack,a.odds_pack)
        result=m.compare(result,a.baseline_covered,a.baseline_uncovered,rid)
        output.append(result)
        print("REAL_L21_DATE_PARITY",rid,"axis",result["anchor_horse_number"],
              "tickets",result["ticket_count"],flush=True)
    count=sum(r["ticket_count"] for r in output)
    if count<3:raise ValueError("REAL_DAY_TOO_FEW_TICKETS_TO_PROVE_PORTFOLIO")
    return {
        "contract":"KEIBA_L21_ALL_RACES_ONE_DATE_REAL_FINAL_REPRODUCTION_V1",
        "date":"2025-01-06","races":len(output),"physical_tickets":count,
        "original_research_parity":all(r["original_saved_ticket_set_exact"] for r in output),
        "odds_phase":"FINAL_HISTORICAL_NOT_TRADEABLE",
        "live_quote_confirmed":False,
        "source_outcome_labels_used_to_generate":False,
        "race_summaries":[{"race_id":r["race_id"],
                           "anchor_horse_number":r["anchor_horse_number"],
                           "ticket_count":r["ticket_count"]} for r in output]
    }

def main():
    p=argparse.ArgumentParser()
    for f in ("predictions","ability-cache","picks","race-pack","odds-pack",
              "baseline-covered","baseline-uncovered","output"):
        p.add_argument("--"+f,required=True)
    a=p.parse_args()
    result=run_day(a)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("WHOLE_REAL_HISTORICAL_DATE_L21_FROZEN_TICKET_PARITY_PASS",
          result["date"],result["races"],result["physical_tickets"],flush=True)

if __name__=="__main__":main()
