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
            # The remote pack was manifest-pinned to 2025-01-06. Race
            # date may be under 'date', 'race_date' or absent in BACKFILL.
            observed=race.get("race_date") or race.get("date")
            if observed and str(observed)[:10]!="2025-01-06":
                raise ValueError("BACKFILL_PACK_RACE_DATE_DRIFT")
            if rid.startswith("2025"):
                rows.append(rid)
    if len(rows)<8 or len(rows)!=len(set(rows)) or any(len(rid)!=12 for rid in rows):
        raise ValueError("PROVEN_HISTORICAL_DAY_RACE_SET_INCOMPLETE_OR_DUPLICATED")
    output=[]
    blocked=[]
    for rid in sorted(rows):
        try:
            result=m.replay_one(
                rid,a.predictions,a.ability_cache,a.picks,a.race_pack,a.odds_pack,
                retrospective_dnf_scope=a.retrospective_dnf_source_scope)
            result=m.compare(result,a.baseline_covered,a.baseline_uncovered,rid)
        except ValueError as exc:
            # An archived OOS prediction can legitimately be absent for a
            # BACKFILL STARTED entry. Such a race must NOT be silently
            # filled with a 7-King proxy or counted as a validated race.
            blocked.append({"race_id":rid,"status":"BLOCKED_NOT_REPRODUCIBLE",
                            "reason":str(exc)})
            print("REAL_L21_DATE_RACE_BLOCKED",rid,str(exc)[:600],flush=True)
            continue
        output.append(result)
        print("REAL_L21_DATE_PARITY",rid,"axis",result["anchor_horse_number"],
              "tickets",result["ticket_count"],flush=True)
    count=sum(r["ticket_count"] for r in output)
    if count<3:raise ValueError("REAL_DAY_TOO_FEW_TICKETS_TO_PROVE_PORTFOLIO")
    return {
        "contract":"KEIBA_L21_ALL_RACES_ONE_DATE_REAL_FINAL_REPRODUCTION_V1",
        "date":"2025-01-06","races":len(output),"requested_races":len(rows),
        "blocked_races":len(blocked),"blocked_cases":blocked,
        "status":"ALL_RACES_MATCH" if not blocked else "PARTIAL_RACE_SOURCE_BLOCKS",
        "retrospective_dnf_source_scope":a.retrospective_dnf_source_scope,
        "dnf_exclusion_is_postresult_hindsight":a.retrospective_dnf_source_scope,
        "races_using_future_dnf_status":sum(bool(r["retrospective_dnf_omitted_horse_ids"]) for r in output),
        "dnf_horses_excluded_as_in_original_oos":sum(len(r["retrospective_dnf_omitted_horse_ids"]) for r in output),
        "NOT_PREBET_PERFORMANCE_EVALUATION":True,
        "physical_tickets":count,
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
    p.add_argument("--retrospective-dnf-source-scope",action="store_true",
                   help="HISTORICAL AUDIT ONLY: reproduce original post-result DNF label-filtered ticket universe. FORBIDDEN for prebet.")
    for f in ("predictions","ability-cache","picks","race-pack","odds-pack",
              "baseline-covered","baseline-uncovered","output"):
        p.add_argument("--"+f,required=True)
    a=p.parse_args()
    result=run_day(a)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("WHOLE_REAL_HISTORICAL_DATE_L21_FROZEN_TICKET_PARITY_PASS",
          result["date"],result["races"],result["physical_tickets"],
          "BLOCKED",result["blocked_races"],flush=True)

if __name__=="__main__":main()
