#!/usr/bin/env python3
"""Fresh historical 202505050812 original L1->ANCHOR_FULL->frozen L2.1->V3.

Historical FINAL odds explicitly allowed as research ONLY. The PREBET engine
is NOT called with fabricated timestamps; its exact frozen line/gate/selection
functions are reused unchanged. Official V3 reverse_p3_stakes is reused.
Missing original L1.7 p3 is UNKNOWN, never estimated or substituted.
The NEW 17-horse ability + correction risk (not 16-horse OOS replay) is
the sole prediction basis.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import score_github_original_anchor_asof as SOURCE
import l21_github_frozen_prebet as L21
from l3_v3_operational import reverse_p3_stakes, POLICY

RID="202505050812"
DAY="2025-11-30"
CONTRACT="KEIBA_FRESH_HISTORICAL_L1_L21_L3_MOBILE_V1"
PRICE_MAP={"QUINELLA":"4","TRIO":"7","TRIFECTA":"8"}

def ensure(cond,why):
    if not cond: raise ValueError(why)

def selected_gzip(path,nested):
    found=[]
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            owner=row if not nested else (row.get("race") or {})
            if owner.get("race_id")==RID:found.append(row)
    ensure(len(found)==1,"USER_SELECTED_RACE_SOURCE_NOT_UNIQUE")
    return found[0]

def price(item,label):
    # Frozen BACKFILL odds pack 'finalOddsTuple()' uses index 0 for 3
    # element source arrays, as verified by original decoder.
    ensure(isinstance(item,list) and len(item)==3,
           "ORIGINAL_FINAL_PRICE_TUPLE_SCHEMA_"+label)
    try: val=float(str(item[0]).replace(",",""))
    except (ValueError,TypeError): raise ValueError("ORIGINAL_FINAL_PRICE_NOT_NUMERIC_"+label)
    ensure(math.isfinite(val) and 1<=val<=1_000_000,
           "INVALID_OR_UNAVAILABLE_FINAL_PRICE_"+label)
    return val

def final_price_table(archive,starters):
    ensure(archive.get("race_id")==RID and archive.get("actual_date")==DAY,
           "FINAL_ODDS_RACE_DATE_DRIFT")
    ensure(archive.get("source")=="NETKEIBA_FINAL" and
           archive.get("odds_kind")=="FINAL" and
           archive.get("source_status")=="result",
           "NOT_VERIFIED_BACKFILL_HISTORICAL_FINAL_SOURCE")
    rows=archive.get("odds")
    ensure(isinstance(rows,dict),"MISSING_HISTORICAL_FINAL_ODDS_GROUPS")
    numbers=set(starters)
    wins={}
    for n in numbers:
        raw=(rows.get("1") or {}).get(str(n).zfill(2))
        ensure(raw is not None,"FINAL_WIN_ODDS_MISSING_HORSE_"+str(n))
        wins[n]=price(raw,"WIN_"+str(n))
    available={}
    def kprice(group,selection):
        key="".join(str(n).zfill(2) for n in selection)
        raw=(rows.get(group) or {}).get(key)
        ensure(raw is not None,"FINAL_COMBINATION_QUOTE_MISSING_"+group+"_"+key)
        return price(raw,group+"_"+key)
    all_numbers=sorted(numbers)
    for a,b in itertools.combinations(all_numbers,2):
        available[("QUINELLA",(a,b))]=kprice("4",(a,b))
    for a,b,c in itertools.combinations(all_numbers,3):
        available[("TRIO",(a,b,c))]=kprice("7",(a,b,c))
    for nums in itertools.permutations(all_numbers,3):
        available[("TRIFECTA",nums)]=kprice("8",nums)
    ensure(len(available)==(math.comb(len(numbers),2)+math.comb(len(numbers),3)+
                              len(numbers)*(len(numbers)-1)*(len(numbers)-2)),
           "FINAL_COMBINATION_ACTIVE_FIELD_COVERAGE_DRIFT")
    return wins,available

def historical_full_frozen(bridge,daypack,odds,parameters,paper_budget=20000):
    ensure(bridge.get("contract")=="KEIBA_BLIND_ORIGINAL_L1_TO_FROZEN_L175_BRIDGE_V1" and
           bridge.get("race_id")==RID and
           bridge.get("l1_original_ability_and_risk_bridge_connected") is True,
           "FRESH_BLIND_ORIGINAL_ABILITY_RISK_NOT_CONNECTED")
    ensure(bridge.get("horse_count")==17 and
           bridge.get("target_result_or_saved_prediction_read") is False,
           "NEW_17_HORSE_SOURCE_OR_PREDICTION_LEAK")
    ensure(paper_budget>=0 and type(paper_budget) is int and paper_budget%100==0,
           "INVALID_PAPER_RACE_BUDGET")
    ensure((daypack.get("race") or {}).get("race_id")==RID,
           "BACKFILL_CARD_RACE_ID_DRIFT")
    entries=daypack.get("entries")
    ensure(isinstance(entries,list),"BACKFILL_CARD_ENTRIES_ABSENT")
    known={};scratched={}
    for row in entries:
        n=row.get("horse_number")
        hid=row.get("horse_id")
        ensure(type(n) is int and 1<=n<=24 and n not in known and n not in scratched,
               "HISTORICAL_CANONICAL_HORSE_NUMBER_DRIFT")
        ensure(isinstance(hid,str) and hid,"HISTORICAL_CANONICAL_HORSE_ID_MISSING")
        st=row.get("entry_status")
        if st=="STARTED":known[n]=row
        elif st in ("SCRATCHED","EXCLUDED"):scratched[n]=row
        else:raise ValueError("HISTORICAL_ENTRY_STATUS_UNKNOWN")
    ensure(len(known)==17 and len(entries)==18 and len(scratched)==1,
           "HISTORICAL_USER_RACE_STARTED_VS_SCRATCHED_DRIFT")
    source=bridge.get("horses")
    ensure(isinstance(source,list) and len(source)==17,"BLIND_SCORE_17_HORSES_ABSENT")
    mapped={}
    for horse in source:
        n=horse.get("horse_number")
        if type(n) is not int or n not in known or n in mapped:
            raise ValueError("BLIND_SCORE_HORSE_NUMBER_NOT_STARTED")
        ensure(horse.get("horse_id")==known[n].get("horse_id"),
               "BLIND_SCORE_HORSE_ID_VS_BACKFILL_CANONICAL_DRIFT")
        mapped[n]={
           "horse_id":horse["horse_id"],"horse_number":n,
           "distill_ai_rank":horse["ai_raw_rank"],
           "distill_score_z":horse["distill_score_z"],
           "distill_score_gap_to_best":horse["distill_score_gap_to_best"],
           "risk":horse["correction_risk"]}
    ensure(set(mapped)==set(known),"BLIND_17_HORSE_SET_INCOMPLETE")
    wins,combos=final_price_table(odds,known)
    ensure(parameters.get("contract")=="KEIBA_ORIGINAL_ANCHOR_FULL_PRIOR_TRAINED_PARAMS_V1",
           "EXACT_2025_ORIGINAL_ANCHOR_MODEL_REQUIRED")
    ensure(parameters.get("trained_only_on_years")==[2022,2023,2024] and
           parameters.get("variant")=="ANCHOR_FULL" and
           parameters.get("features_in_original_order")==list(SOURCE.MODEL.FEATURES),
           "ORIGINAL_ANCHOR_PARAM_FOLD_DRIFT")
    coeff=parameters["coefficients"]
    normal=parameters["standardizer"]
    ensure(set(coeff)==set(normal)==set(SOURCE.MODEL.FEATURES),
           "ORIGINAL_ANCHOR_PARAMETER_FEATURE_DRIFT")
    geometry=SOURCE.GEOM.build_geometry(mapped,wins)
    model_horses={}
    logits={}
    for n,feature in geometry.items():
        z=math.log(feature["market_prob_norm"])
        for k in SOURCE.MODEL.FEATURES:
            unit=normal[k]
            sd=float(unit["std"]);avg=float(unit["mean"]);beta=float(coeff[k])
            ensure(sd>0 and all(math.isfinite(v) for v in (sd,avg,beta)),
                   "ORIGINAL_ANCHOR_STANDARDIZER_INVALID")
            z+=(float(feature["features"][k])-avg)/sd*beta
        logits[n]=z
        model_horses[n]={**mapped[n],
                          "market_rank":feature["market_rank"],
                          "market_prob_norm":feature["market_prob_norm"],
                          "win_odds_at_cutoff":wins[n]}
    zmax=max(logits.values())
    denom=math.fsum(math.exp(z-zmax) for z in logits.values())
    ensure(denom>0 and math.isfinite(denom),"ORIGINAL_ANCHOR_PROBABILITIES_INVALID")
    for n in logits:
        model_horses[n]["P__ANCHOR_FULL"]=math.exp(logits[n]-zmax)/denom
    ensure(abs(sum(h["P__ANCHOR_FULL"] for h in model_horses.values())-1)<1e-9,
           "ORIGINAL_ANCHOR_PROBABILITY_NORMALIZATION_DRIFT")
    market_axis=sorted(known,key=lambda n:(-(model_horses[n]["P__ANCHOR_FULL"]*wins[n]),
                       -model_horses[n]["P__ANCHOR_FULL"],model_horses[n]["market_rank"],
                       model_horses[n]["horse_id"]))[0]
    active=set(known)
    order={
      "AI_PROB":sorted(active,key=lambda n:(-model_horses[n]["P__ANCHOR_FULL"],
                                             model_horses[n]["market_rank"],
                                             model_horses[n]["horse_id"])),
      "AI_RAW":sorted(active,key=lambda n:(model_horses[n]["distill_ai_rank"],
                                            model_horses[n]["market_rank"],
                                            model_horses[n]["horse_id"])),
      "MARKET":sorted(active,key=lambda n:(model_horses[n]["market_rank"],
                                            wins[n],model_horses[n]["horse_id"]))}
    for idx,n in enumerate(order["AI_PROB"],1):
        model_horses[n]["P_rank"]=idx
    source_contract,approved,_=L21.source_contract()
    assert source_contract["version"]=="2.1.0"
    physical=defaultdict(set);considered=0
    for (line,gate) in sorted(approved):
        listtype,cutoff,family,band=line.split("|")
        partners=[n for n in order[listtype] if n!=market_axis][:int(cutoff)]
        low,high=L21.BAND[band]
        for ticket in L21.generated(market_axis,partners,family):
            considered+=1
            ensure(ticket in combos,"ORIGINAL_FROZEN_L21_MISSING_HISTORICAL_QUOTE")
            if low<=combos[ticket]<high and L21.gate_values(ticket,model_horses,market_axis)[gate]:
                physical[ticket].add((line,gate))
    tickets=[]
    for ticket,proof in sorted(physical.items()):
        kind,selection=ticket
        tickets.append({
          "bet_type":kind,"selection_numbers":list(selection),
          "canonical_selection":"-".join(map(str,selection)),
          "decimal_odds_final":combos[ticket],
          "matched_line_gates":[{"line":line,"gate":gate}
                                for line,gate in sorted(proof)],
          "baseline_stake_yen":100,
          "min_p3":None,"p3_status":"ORIGINAL_L17_P3_NOT_AVAILABLE",
          "historical_price_phase":"FINAL_RESEARCH_ONLY"})
    # Same EXACT official historical V3 staking helper; keep unknown p3
    # instead of silently substituting ability/win probability/guess.
    bankroll=min(paper_budget,200*len(tickets))
    if tickets and bankroll<100*len(tickets):
        raise ValueError("V3_PAPER_BUDGET_BELOW_FROZEN_MINIMUM")
    assignment=reverse_p3_stakes([(row,None) for row in tickets],bankroll) if tickets else {}
    for row in tickets:
        key=(row["bet_type"],row["canonical_selection"])
        row["stake_yen"]=assignment[key]
    assert sum(x["stake_yen"] for x in tickets)==bankroll
    ranked={
        "ai_raw":[model_horses[n] for n in order["AI_RAW"]],
        "ai_probability":[model_horses[n] for n in order["AI_PROB"]],
        "market":[model_horses[n] for n in order["MARKET"]],
        "expected_value":[model_horses[n] for n in sorted(
            active,key=lambda n:(-(model_horses[n]["P__ANCHOR_FULL"]*wins[n]),
                                  -model_horses[n]["P__ANCHOR_FULL"],
                                  model_horses[n]["market_rank"],model_horses[n]["horse_id"]))]}
    # Include full names and numbers for mobile; do not expose target results.
    def display(n):
        h=model_horses[n]
        return {"horse_id":h["horse_id"],"horse_number":n,
                "horse_name":known[n]["horse_name"],
                "ai_raw_rank":h["distill_ai_rank"],
                "ai_probability_rank":h["P_rank"],
                "correction_risk":h["risk"],
                "ability_score_z":h["distill_score_z"],
                "market_final_rank":h["market_rank"],
                "historical_final_win_odds":wins[n],
                "original_anchor_win_probability":h["P__ANCHOR_FULL"],
                "historical_expected_value":h["P__ANCHOR_FULL"]*wins[n]}
    output={
      "contract":CONTRACT,"race_id":RID,"race_date":DAY,
      "status":"HISTORICAL_FINAL_ONLY_NEW_L1_L21_V3_P3_UNKNOWN",
      "race_entry_count":18,"started_horse_count":17,
      "scratched":[{"horse_id":h["horse_id"],"horse_name":h["horse_name"],
                    "horse_number":n} for n,h in sorted(scratched.items())],
      "scope":"HISTORICAL_RESEARCH_ONLY_NOT_LIVE",
      "source_price_phase":"NETKEIBA_FINAL_POSTRACE",
      "features_asof_independently_verified":False,
      "is_live_or_purchase_ready":False,"actual_bets_placed":False,
      "fresh_original_ability_risk_used":True,
      "historical_saved_2025_oos_scores_used":False,
      "historical_saved_l21_tickets_used":False,
      "original_anchor_parameter_train_years":[2022,2023,2024],
      "original_l2_1_frozen_version":"2.1.0",
      "original_l2_1_policy":"STANDARD",
      "original_l2_1_standard_line_gate_count":len(approved),
      "original_l2_1_candidate_rule_occurrences":considered,
      "original_anchor_horse_id":model_horses[market_axis]["horse_id"],
      "original_anchor_horse_number":market_axis,
      "original_l3_v3_policy":POLICY,
      "l3_p3_provenance":"UNKNOWN_NO_INVENTED_P3",
      "l3_p3_ranking_exactly_reproduced":False,
      "l3_staking_formula_original":True,
      "bet_count":len(tickets),"l3_v3_paper_stake_yen":bankroll,
      "horses":[display(n) for n in sorted(known)],
      "rankings":{k:[display(h["horse_number"]) for h in v]
                  for k,v in ranked.items()},
      "tickets":tickets,
    }
    return output

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bridge",required=True)
    ap.add_argument("--historical-day-pack",required=True)
    ap.add_argument("--historical-odds-pack",required=True)
    ap.add_argument("--original-anchor-parameters",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--paper-race-budget-yen",type=int,default=20000)
    a=ap.parse_args()
    def get(path):return json.loads(Path(path).read_text(encoding="utf-8"))
    card=selected_gzip(a.historical_day_pack,True)
    odds=selected_gzip(a.historical_odds_pack,False)
    data=historical_full_frozen(get(a.bridge),card,odds,
             get(a.original_anchor_parameters),a.paper_race_budget_yen)
    dest=Path(a.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,allow_nan=False,
                               separators=(",",":"))+"\n",encoding="utf8")
    print("NEW_REAL_BLIND_202505050812_ORIGINAL_L1_L21_L3_HISTORICAL",
          json.dumps({"race_id":RID,"horses":data["started_horse_count"],
          "anchor_no":data["original_anchor_horse_number"],
          "tickets":data["bet_count"],"paper_yen":data["l3_v3_paper_stake_yen"],
          "p3_all_unknown":True,"original_50_line_gates":len(L21.source_contract()[1]),
          "prior_only_coeffs":True,"final_odds_only":True},sort_keys=True),flush=True)
    print("HISTORICAL_NEW_L1_L3_TOP5",json.dumps(
          data["rankings"]["ai_raw"][:5],ensure_ascii=False),flush=True)

if __name__=="__main__":main()
