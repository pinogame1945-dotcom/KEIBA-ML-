#!/usr/bin/env python3
"""Source-transparent historical-year-trained ANCHOR_FULL as-of market scorer.

Uses ORIGINAL prior-only 2025 coefficients/normalizer recreated from true
frozen sources. Does not invent ability or risk: it REQUIRES complete,
independently derived, time-stamped ORIGINAL ability/risk companions.
A validated JRA same-snapshot WIN + three bet types is also required.
This is a paper adapter, NOT a promoted live betting model or an executable
bet. 2026 results are sealed; only true upcoming observation is permitted.
"""
from __future__ import annotations
import argparse,datetime,importlib.util,json,math,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def module(file,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"research"/file)
    out=importlib.util.module_from_spec(spec);sys.modules[name]=out
    spec.loader.exec_module(out);return out
BRIDGE=module("bridge-mobile-combination-odds.py","original_2025_anchor_mobile_bridge")
L21=module("l21-frozen-prebet-price.py","original_2025_anchor_l21")
MODEL=module("replay-original-anchor-coefficients-2025.py","original_2025_anchor_params")
CONTRACT="KEIBA_L1_ORIGINAL_ABILITY_RISK_ASOF_V1"
RESULT="KEIBA_ORIGINAL_ANCHOR_FULL_ASOF_PAPER_V1"

def calc(original,mobile,companions,now=None):
    if original.get("contract")!="KEIBA_ORIGINAL_ANCHOR_FULL_PRIOR_TRAINED_PARAMS_V1":
        raise ValueError("NOT_ORIGINAL_PRIOR_TRAINED_2025_FULL_ANCHOR")
    if original.get("trained_only_on_years")!=[2022,2023,2024]:
        raise ValueError("ANCHOR_TRAINING_YEAR_OR_ORIGINAL_SELECTION_DRIFT")
    if original.get("variant")!="ANCHOR_FULL" or original.get("features_in_original_order")!=list(MODEL.FEATURES):
        raise ValueError("ORIGINAL_ANCHOR_FEATURE_SCHEMA_CHANGED")
    if original.get("not_promoted_or_authenticated_for_live_betting") is not True:
        raise ValueError("FROZEN_ORIGINAL_MODEL_PROVENANCE_MISSING")
    beta=original.get("coefficients") or {}
    normal=original.get("standardizer") or {}
    if set(beta)!=set(MODEL.FEATURES) or set(normal)!=set(MODEL.FEATURES):
        raise ValueError("INCOMPLETE_ORIGINAL_ANCHOR_WEIGHTS")
    race,rid,active,start=L21.validate_live_card(mobile)
    quote=BRIDGE.convert(mobile)
    win=quote["win_odds_rows"]
    if companions.get("contract")!=CONTRACT or companions.get("race_id")!=rid or companions.get("canonical_race_key")!=race["raceKey"]:
        raise ValueError("REAL_ABILITY_RISK_COMPANION_RACE_DRIFT")
    if companions.get("feature_source")!="ORIGINAL_PREBET_ABILITY_AND_RISK":
        raise ValueError("SEVEN_KING_OR_ARCHIVED_POSTRACE_NOT_ALLOWED_AS_COMPANION")
    if companions.get("source_observed_before_bet") is not True:
        raise ValueError("ORIGINAL_ABILITY_RISK_ASOF_EVIDENCE_MISSING")
    stamp=L21.timestamp(companions.get("observed_at"),"companions_observed")
    quote_time=L21.timestamp(quote["observed_at"],"odds_observed")
    if stamp>quote_time:raise ValueError("COMPANION_FEATURES_OBSERVED_AFTER_MARKET_QUOTE")
    clock=now if now is not None else datetime.datetime.now(datetime.timezone.utc)
    if clock.tzinfo is None:raise ValueError("NOW_MUST_HAVE_TIMEZONE")
    clock=clock.astimezone(datetime.timezone.utc)
    cutoff=L21.timestamp(quote["bet_cutoff_at"],"bet_cutoff")
    if not quote_time<=clock<cutoff:
        raise ValueError("REAL_ANCHOR_INFERENCE_TIME_NOT_WITHIN_MARKET_TO_CUTOFF")
    if rid.startswith("2026") and clock>=start:
        raise ValueError("2026_RESULTS_SEALED_ONLY_FUTURE_UNSTARTED_RACE")
    horses=companions.get("horses")
    if not isinstance(horses,list) or len(horses)!=len(active):
        raise ValueError("ORIGINAL_ABILITY_RISK_FIELD_COVERAGE")
    records={};ids=set();ranks=set()
    for row in horses:
        number=L21.no(row.get("horse_number"),"ability_risk")
        hid=row.get("horse_id")
        if number in records or number not in active or not isinstance(hid,str) or not hid or hid in ids:
            raise ValueError("ORIGINAL_ABILITY_RISK_HORSE_OR_NUMBER_DRIFT")
        ids.add(hid)
        rank=row.get("distill_ai_rank")
        if type(rank) is not int or not 1<=rank<=len(active) or rank in ranks:
            raise ValueError("ORIGINAL_ABILITY_RANK_NOT_COMPLETE_UNIQUE")
        ranks.add(rank)
        clean=dict(row)
        for field in ("distill_score_z","distill_score_gap_to_best","risk"):
            clean[field]=L21.number(row.get(field),"companion_"+field)
        if clean["distill_score_gap_to_best"]<0 or not 0<=clean["risk"]<=1:
            raise ValueError("COMPANION_RISK_OR_SCORE_RANGE_INVALID")
        records[number]=clean
    if set(records)!=active or ranks!=set(range(1,len(active)+1)):
        raise ValueError("ORIGINAL_ABILITY_RISK_ACTIVE_FIELD_NOT_FULL")
    wins={}
    for row in win:
        n=row["selection1"]
        if n in wins:raise ValueError("DUPLICATED_WIN_ODDS")
        wins[n]=L21.number(row["odds"],"win")
    if set(wins)!=active:raise ValueError("JRA_WIN_QUOTE_NOT_COMPLETE")
    best=min(wins.values())
    if sum(v==best for v in wins.values())!=1:
        raise ValueError("MARKET_TOP1_TIE_WITHOUT_OFFICIAL_POPULARITY_UNRESOLVED")
    order=sorted(active,key=lambda n:(wins[n],records[n]["horse_id"]))
    market_rank={}
    for i,n in enumerate(order):
        market_rank[n]=1+next(j for j,m in enumerate(order) if wins[m]==wins[n])
    win_inv={n:1/wins[n] for n in active}
    inv_sum=math.fsum(win_inv.values())
    market={n:win_inv[n]/inv_sum for n in active}
    overlap=sum(records[n]["distill_ai_rank"]<=3 and market_rank[n]<=3 for n in active)
    top1_match=int(next(n for n in active if records[n]["distill_ai_rank"]==1)==
                   next(n for n in active if market_rank[n]==1))
    full=[]
    count=len(active)
    for n in sorted(active):
        h=records[n]
        ai=h["distill_ai_rank"];mr=market_rank[n]
        gap=(mr-ai)/count
        ngap=abs(mr-ai)/count
        disagree=(3-max(0,min(3,overlap)))/3
        feat={
          "field_size":count,
          "distill_ai_rank01":1-(ai-1)/max(1,count-1),
          "distill_score_z":h["distill_score_z"],
          "distill_score_gap_to_best":h["distill_score_gap_to_best"],
          "distill_is_ai_top1":float(ai==1),
          "distill_is_ai_top3":float(ai<=3),
          "distill_is_ai_top6":float(ai<=6),
          "risk":h["risk"],
          "rank_gap_norm":gap,"abs_rank_gap_norm":ngap,
          "top3_overlap_norm":overlap/3,
          "ai_unique_top3":int(ai<=3 and mr>3),
          "market_unique_top3":int(ai>3 and mr<=3),
          "top1_match":top1_match,
          "risk_x_abs_gap":h["risk"]*ngap,
          "ability_x_market_gap":h["distill_score_z"]*gap,
          "ability_x_disagree_level":h["distill_score_z"]*disagree,
          "risk_x_disagree_level":h["risk"]*disagree,
          "rankgap_x_disagree_level":gap*disagree,
        }
        z=math.log(market[n])
        for f in MODEL.FEATURES:
            unit=normal[f]
            sd=L21.number(unit["std"],"original_standardizer_sd")
            if sd<=0:raise ValueError("INVALID_ORIGINAL_STANDARDIZER_SD")
            mean=L21.number(unit["mean"],"original_standardizer_mean")
            weight=L21.number(beta[f],"original_beta")
            z+=(feat[f]-mean)/sd*weight
        full.append({"horse_id":h["horse_id"],"horse_number":n,
                     "win_odds_at_cutoff":wins[n],"market_rank":mr,
                     "market_prob_norm":market[n],
                     "distill_ai_rank":ai,
                     "distill_score_z":h["distill_score_z"],
                     "risk":h["risk"],"_logit":z})
    maximum=max(x["_logit"] for x in full)
    raw=[math.exp(x["_logit"]-maximum) for x in full]
    total=math.fsum(raw)
    for x,v in zip(full,raw):
        x["P__ANCHOR_FULL"]=v/total
        del x["_logit"]
    ranked=sorted(full,key=lambda h:(
        -(h["P__ANCHOR_FULL"]*h["win_odds_at_cutoff"]),
        -h["P__ANCHOR_FULL"],h["market_rank"],h["horse_id"]))
    axis=ranked[0]
    original_panel={
       "contract":"KEIBA_L21_ORIGINAL_ANCHOR_FEATURES_PREBET_V1",
       "race_id":rid,"canonical_race_key":race["raceKey"],
       "anchor_source":"EXACT_EXISTING_ANCHOR_FULL",
       "feature_source":"ORIGINAL_L21_HORSE_FEATURES",
       "input_phase":"PRE_RACE",
       "features_asof_asserted":True,
       "features_observed_at":companions["observed_at"],
       "win_odds_observed_at":quote["observed_at"],
       "anchor_computed_at":clock.isoformat(),
       "anchor_horse_id":axis["horse_id"],
       "horses":full,
       "model_provenance":"RECONSTRUCTED_PRIOR_OOS_COEFFICIENTS_WITH_ASSERTED_UPSTREAM_COMPANIONS",
       "not_model_promoted_for_live_betting":True,
       "score_requires_actual_asof_ability_risk":True,
       "market_odds_rank_ties_method":"MIN_RANK_WITH_UNIQUE_FIRST"
    }
    return {
       "contract":RESULT,"race_id":rid,
       "status":"PAPER_PRIOR_PARAMETERS_SCORED_WITH_ASSERTED_SOURCE_TIMES",
       "not_official_live_bet_prediction":True,
       "prior_only_model_no_2025_fit_or_selection":True,
       "original_companion_provenance_not_independently_verified":True,
       "model_full_probs":original_panel,
       "source_win_market_capture_at":quote["observed_at"],
       "original_anchor_number":axis["horse_number"],
       "original_anchor_horse_id":axis["horse_id"],
       "no_result_or_payout_read":True
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--parameters",required=True)
    p.add_argument("--mobile-export",required=True)
    p.add_argument("--original-ability-risk",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    result=calc(json.loads(Path(a.parameters).read_text()),
        json.loads(Path(a.mobile_export).read_text()),
        json.loads(Path(a.original_ability_risk).read_text()))
    target=Path(a.out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("PRIOR_ORIGINAL_ANCHOR_FULL_ASOF_PAPER_SCORE",result["race_id"],
          result["original_anchor_number"],flush=True)

if __name__=="__main__":main()
