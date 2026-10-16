#!/usr/bin/env python3
"""Blind real L1 ability+Risk -> validated horse-level interface.

Never upgrades a historical EARLY/FINAL research projection to an authenticated
JRA PRE_RACE observation. No market probabilities, original L1.7 p3, odds, or
tickets are substituted. Emits actionable BLOCKED diagnostics instead.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

RACE="202505050812"
CONTRACT="KEIBA_BLIND_ORIGINAL_L1_TO_FROZEN_L175_BRIDGE_V1"
BLOCKERS=(
    "HISTORICAL_SNAPSHOT_CAPTURE_ASOF_NOT_INDEPENDENTLY_VERIFIED",
    "OFFICIAL_PRESTART_FULL_FIELD_NOT_INDEPENDENTLY_VERIFIED",
    "ORIGINAL_2025_ANCHOR_FULL_REQUIRES_AUTHENTIC_PREBET_WIN_QUOTES",
    "FROZEN_L21_REQUIRES_AUTHENTIC_SAME_SNAPSHOT_COMBINATION_QUOTES",
    "L3_V3_REQUIRES_ORIGINAL_L17_KING_OUTSIDER_PREBET_P3",
)

def check(ok,message):
    if not ok:raise ValueError(message)

def unique(source,field,value):
    check(source.get(field)==value,"BLIND_SOURCE_"+field+"_MISMATCH")

def bridge(ability,risk,target):
    unique(ability,"contract","KEIBA_202505050812_REAL_BLIND_ORIGINAL_ABILITY_V1")
    unique(risk,"contract","KEIBA_BLIND_202505050812_FROZEN_CORRECTION_RISK_V1")
    unique(target,"contract","KEIBA_UNSCORED_RACE_FEATURE_EXTRACT_V1")
    for entry in (ability,risk,target):
        unique(entry,"race_id",RACE)
    check(ability.get("training_years")==[2023,2024],"WRONG_ORIGINAL_L1_TRAIN_YEARS")
    check(risk.get("risk_prior_teacher_years")==[2021,2022,2023,2024],
          "WRONG_RISK_TEACHER_YEARS")
    check(risk.get("risk_seed")==8702500,"WRONG_ORIGINAL_RISK_SEED")
    check(len(risk.get("original_oos_metric_reproduction") or [])==4 and
          all(x.get("metric_reference_exact") is True for x in
              risk["original_oos_metric_reproduction"]),
          "ORIGINAL_OOS_PARITY_NOT_PROVEN")
    for entry,fields in (
        (ability,("selected_target_outcomes_or_final_odds_used",
                  "target_saved_historical_predictions_read")),
        (risk,("selected_2025_outcome_or_final_odds_used",
               "selected_2025_saved_prediction_used")),
        (target,("used_target_labels","used_results_payouts_final_odds",
                 "used_historical_oos_target_predictions"))):
        for field in fields:check(entry.get(field) is False,"BLIND_INPUT_LEAK_"+field)
    unique(target,"source_prediction_phase",
           "FINAL_PRESTART_PROJECTED_FROM_HISTORICAL_SNAPSHOT")
    check(target.get("source_manifest_count_and_date_range_verified") is True,
          "SOURCE_SNAPSHOT_MANIFEST_NOT_VERIFIED")
    # Source capture is not independently verified; rejecting assertions
    # of certification prevents a downstream adapter accidentally promoting
    # this research-only record.
    for record,key in ((ability,"genuine_asof_source_timestamps_verified"),
                       (risk,"genuine_target_prestart_capture_time_independently_verified"),
                       (target,"actual_source_capture_time_independently_verified")):
        check(record.get(key) is False,"UNSUPPORTED_HISTORICAL_ASOF_ASSERTION")
    horses=target.get("rows")
    raw=ability.get("horse_scores")
    risks=risk.get("horse_risk")
    check(all(isinstance(x,list) and len(x)==17 for x in (horses,raw,risks)),
          "BLIND_COMPLETE_17_HORSES_REQUIRED")
    forbidden_names={"_finish","_is_win","_is_top3","_score",
                     "finish","finish_position","market_rank",
                     "win_odds_at_cutoff","risk","distill_ai_rank",
                     "P__ANCHOR_FULL","actual_start_time"}
    forbidden_prefixes=("_target","target_","finish_","result_",
                       "payout_","final_","postrace_","after_race_",
                       "market_final_","distill_","outcome_")
    for horse in horses:
        check(isinstance(horse,dict),"BLIND_HORSE_RECORD_NOT_OBJECT")
        check(not any(k in forbidden_names or k.startswith(forbidden_prefixes)
                      for k in horse),"BLIND_TARGET_RESULT_FIELD_FORBIDDEN")
    def by_id(rows,field):
        indexed={}
        for row in rows:
            hid=row.get(field)
            check(isinstance(hid,str) and hid and hid not in indexed,
                  "BLIND_HORSE_KEY_MISSING_OR_DUPLICATE")
            indexed[hid]=row
        return indexed
    entries=by_id(horses,"_horse_id")
    ability_map=by_id(raw,"horse_id")
    risk_map=by_id(risks,"horse_id")
    check(set(entries)==set(ability_map)==set(risk_map),
          "BLIND_CANONICAL_17_HORSE_SET_MISMATCH")
    numbers=set();ranks=set();result=[]
    for hid,entry in entries.items():
        number=entry.get("horse_number")
        check(type(number) is int and 1<=number<=24 and number not in numbers,
              "BLIND_HORSE_NUMBER_NOT_CANONICAL")
        numbers.add(number)
        first=ability_map[hid];second=risk_map[hid]
        rank=first.get("ai_raw_rank")
        check(type(rank) is int and 1<=rank<=17 and rank not in ranks and
              second.get("ai_raw_rank")==rank,
              "BLIND_RISK_ABILITY_RANK_DRIFT")
        ranks.add(rank)
        for key in ("ability_score","ability_score_z","ability_gap_to_best"):
            value=first.get(key)
            check(type(value) in (float,int) and math.isfinite(value),
                  "BLIND_ABILITY_NONFINITE_"+key)
        risk_value=second.get("correction_risk")
        check(type(risk_value) in (float,int) and math.isfinite(risk_value)
              and 0<=risk_value<=1,"BLIND_CORRECTION_RISK_INVALID")
        result.append({
            "horse_id":hid,"horse_number":number,
            "ai_raw_rank":rank,
            "ability_score":float(first["ability_score"]),
            "distill_score_z":float(first["ability_score_z"]),
            "distill_score_gap_to_best":float(first["ability_gap_to_best"]),
            "correction_risk":float(risk_value),
        })
    check(ranks==set(range(1,18)),"BLIND_RANK_1_TO_17_NOT_COMPLETE")
    result.sort(key=lambda x:x["ai_raw_rank"])
    return {
      "contract":CONTRACT,"race_id":RACE,
      "status":"ORIGINAL_L1_AND_CORRECTION_RISK_JOINED_DOWNSTREAM_BLOCKED",
      "horse_count":len(result),"source_ability_training_years":[2023,2024],
      "source_risk_teacher_years":[2021,2022,2023,2024],
      "target_result_or_saved_prediction_read":False,
      "research_postrace_snapshot_is_not_verified_prebet":True,
      "l1_original_ability_and_risk_bridge_connected":True,
      "original_ability_market_anchor_full_available":False,
      "original_l17_outsider_p3_available":False,
      "frozen_l21_ready":False,"official_l3_v3_ready":False,
      "purchase_ready":False,"actual_bets_placed":False,
      "blocked_reasons":list(BLOCKERS),
      "horses":result,
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fresh-ability",required=True)
    parser.add_argument("--fresh-risk",required=True)
    parser.add_argument("--unscored-target",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    src=lambda p:json.loads(Path(p).read_text(encoding="utf-8"))
    out=bridge(src(args.fresh_ability),src(args.fresh_risk),src(args.unscored_target))
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,
                               allow_nan=False,indent=2)+"\n",encoding="utf-8")
    print("BLIND_REAL_L1_AND_RISK_BRIDGE_JOINED",
          "race",RACE,"horses",len(out["horses"]),
          "blockers",len(out["blocked_reasons"]),"purchase_ready",False,flush=True)

if __name__=="__main__":main()
