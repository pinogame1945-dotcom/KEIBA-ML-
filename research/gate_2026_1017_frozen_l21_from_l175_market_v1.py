#!/usr/bin/env python3
"""Fail-closed original L2.1 handoff gate for existing sealed 2026-10-17 L1.75.

Does NOT turn SevenKing ranks or Outsider p3 into original ANCHOR_FULL, original
distill_ai_rank, original Correction Risk, or as-of certified live pricing.
Does NOT invoke the original frozen 50-rule ticket engine or L3 V3 until all
required original inputs are independently established.
"""
import argparse,datetime,hashlib,json,math
from pathlib import Path
from collections import Counter
L175_INDEX="KEIBA_L175_BACKFILL_2026_1017_RESEARCH_PRIVATE_V1"
L175_RACE="KEIBA_2026_1017_L175_MARKET_JOIN_RESEARCH_V1"
FREEZE="KEIBA_L2_1_FIXED_PURCHASE_V1"
ORIGINAL_RACE_FIELDS=("P__ANCHOR_FULL","distill_ai_rank","distill_score_z",
                      "distill_score_gap_to_best","risk","market_prob_norm")
ORIGINAL_SOURCE_CONTRACT="KEIBA_L21_ORIGINAL_ANCHOR_FEATURES_PREBET_V1"
REQUIRED_PREBET=("canonical_race_key","anchor_horse_id","features_asof_asserted",
 "features_observed_at","anchor_computed_at","bet_cutoff_at",
 "scheduled_start_at","source_confirmed_prestart")
DAY="2026-10-17"

def verify(x,msg):
    if not x:raise ValueError(msg)

def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def check(frozen_idx,races_file,freeze_file,output):
    verify(not output.exists(),"NO_FROZEN_L21_GATE_OVERWRITE")
    index=json.loads(frozen_idx.read_text(encoding="utf-8"))
    frozen=json.loads(freeze_file.read_text(encoding="utf-8"))
    verify(index.get("contract")==L175_INDEX and index.get("races")==36 and
      index.get("horses")==471 and index.get("horses_with_win_market")==471 and
      index.get("races_with_full_win_market")==36 and
      index.get("ranking_policy")=="SEVEN_KING_UNCHANGED" and
      index.get("market_asof_independently_verified") is False and
      index.get("ready_for_betting") is False and
      index.get("l21_l3_invoked") is False and
      index.get("2026_target_results_used") is False and
      index.get("files",{}).get(races_file.name)==digest(races_file),
      "ORIGINAL_L175_BACKFILL_MARKET_INDEX_SHA_OR_RESEARCH_GUARD_INVALID")
    verify(frozen.get("contract")==FREEZE and frozen.get("version")=="2.1.0" and
      frozen.get("primary_policy")=="STANDARD" and
      frozen.get("rule_counts",{}).get("STANDARD")==50 and
      frozen.get("rule_counts",{}).get("STRICT_SUBSET")==26 and
      frozen.get("ticket_policy",{}).get("stake_baseline_yen_per_ticket")==100 and
      frozen.get("ticket_policy",{}).get("anchor","").startswith("EXACT_EXISTING_ANCHOR_FULL"),
      "FROZEN_L21_STANDARD_50_ANCHOR_RULES_CHANGED")
    lines={(x["line"],x["gate"]) for x in frozen["frozen_rules"]["STANDARD"]}
    subset={(x["line"],x["gate"]) for x in frozen["frozen_rules"]["STRICT_SUBSET"]}
    verify(len(lines)==50 and len(subset)==26 and subset.issubset(lines) and
      len({x[0] for x in lines})==9,"ORIGINAL_L21_50_LINE_GATE_DRIFT")
    seen=set(); total=0;p3count=0;withwin=0;race_out=[]
    metrics=Counter()
    with races_file.open("r",encoding="utf-8") as file:
        for line in file:
            if not line.strip():continue
            r=json.loads(line)
            rid=str(r.get("race_id") or "")
            horses=r.get("horses") or []
            verify(len(rid)==12 and rid.startswith("2026") and rid not in seen and
               r.get("contract")==L175_RACE and
               r.get("horse_count")==len(horses) and
               r.get("prediction_year")==2026 and
               r.get("win_market_source")=="JRA_OFFICIAL_POST" and
               r.get("win_market_snapshot_type")=="PRE_RACE_LIVE_ODDS" and
               r.get("win_market_readiness")=="PRE_RACE_OBSERVED_ASOF_UNVERIFIED" and
               r.get("market_asof_independently_verified") is False and
               r.get("market_readiness")=="RESEARCH_ONLY_INDEPENDENT_ASOF_NOT_VERIFIED" and
               r.get("market_field_complete") is True and
               r.get("ranking_policy")=="SEVEN_KING_UNCHANGED" and
               r.get("ready_for_betting") is False and
               r.get("2026_target_results_used") is False and
               r.get("2026_model_fit_or_evaluation") is False and
               r.get("l21_l3_invoked") is False,"L175_UNSAFE_OR_DRIFTED_HISTORICAL_2026_SOURCE:"+rid)
            verify([h["king_rank"] for h in horses]==list(range(1,len(horses)+1)) and
               len({h["horse_id"] for h in horses})==len(horses) and
               len({h["horse_number"] for h in horses})==len(horses),
               "ORIGINAL_L175_HORSE_ID_OR_RANK_DRIFT:"+rid)
            for horse in horses:
                v=horse.get("market_win_odds")
                verify(type(v) in (int,float) and math.isfinite(v) and v>=1 and
                  type(horse.get("market_rank")) is int and
                  1<=horse["market_rank"]<=len(horses) and
                  horse.get("king_rank")!=None and
                  not any(key in horse for key in ORIGINAL_RACE_FIELDS),
                  "L175_MUST_NOT_BE_MISREPRESENTED_AS_ORIGINAL_L21_FEATURES:"+rid)
                p3count+=int(horse.get("p3_calibrated") is not None)
                withwin+=1
            missing=["ORIGINAL_ANCHOR_FULL_AXIS_AND_P_PROBABILITIES",
              "ORIGINAL_DISTILL_AI_RAW_RANKS_AND_SCORE_Z",
              "ORIGINAL_CORRECTION_RISK",
              "ORIGINAL_ASOF_ANCHOR_FEATURE_PANEL",
              "INDEPENDENT_PRE_RACE_SOURCE_CAPTURE_CERTIFICATION",
              "AUTHENTIC_AT_CUTOFF_SAME_SNAPSHOT_JRA_BET_QUOTES",
              "FROZEN_2026_HISTORICAL_REPLAY_REQUIRES_SEPARATE_RESEARCH_AUTHORIZATION"]
            race_out.append({
              "race_id":rid,"horse_count":len(horses),
              "status":"BLOCKED_ORIGINAL_FROZEN_L21_INPUTS_INCOMPLETE",
              "l175_market_covered_horses":len(horses),
              "king_and_outsider_p3_covered_horses":sum(h["p3_calibrated"] is not None for h in horses),
              "original_l21_policy":FREEZE,"original_l21_frozen_version":"2.1.0",
              "policy":"STANDARD","approved_line_gate_rules":50,
              "original_l21_source_contract_required":ORIGINAL_SOURCE_CONTRACT,
              "missing_original_input_categories":missing,
              "seven_king_substituted_as_original_anchor":False,
              "expected_value_ai_axis_fabricated":False,
              "2026_results_or_payouts_used":False,
              "market_asof_independently_verified":False,
              "frozen_l21_engine_invoked":False,"l3_v3_invoked":False,
              "ticket_count":None,"ready_for_betting":False})
            metrics["blocked_races"]+=1;total+=len(horses);seen.add(rid)
    verify(len(seen)==36 and total==471 and withwin==471 and p3count==347,
           "L175_TO_L21_SOURCE_SCOPE_NOT_ORIGINAL_36R_471H_347P3")
    result={"contract":"KEIBA_2026_1017_FROZEN_L21_REAL_INPUT_GATE_V1",
      "research_source":"PINNED_PRIVATE_REAL_L175_JRA_WIN_MARKET",
      "race_date":DAY,"races":race_out,"race_count":36,"horse_count":471,
      "win_market_horse_count":471,"available_original_p3_horse_count":347,
      "frozen_rules_unchanged":True,"primary_policy":"STANDARD",
      "standard_line_gate_count":50,"strict_diagnostic_count":26,
      "original_l21_axis_available":False,
      "historical_2026_frozen_prebet_engine_blocked":True,
      "ready_for_betting":False,"actual_bets_placed":False,
      "no_guess_of_missing_original_models_or_asof":True,
      "frozen_l21_engine_invoked":False,"l3_v3_invoked":False,
      "all_races_status":"BLOCKED_ORIGINAL_FROZEN_L21_INPUTS_INCOMPLETE"}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("FROZEN_L21_REAL_2026_L175_GATE_PASS "+
      json.dumps({"races":36,"horses":total,"win_quotes":withwin,"original_p3":p3count,
       "frozen_standard_rules":50,"blocked_races":36,"ticket_generation":"NOT_INVOKED",
       "missing_original_ANCHOR_FULL":True,"2026_results_used":False},sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser()
    for x in ("l175-index","l175-races","frozen-l21-contract","output"):
        p.add_argument("--"+x,type=Path,required=True)
    a=p.parse_args()
    check(a.l175_index,a.l175_races,a.frozen_l21_contract,a.output)
if __name__=="__main__":main()
