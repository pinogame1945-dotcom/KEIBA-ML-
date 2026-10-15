#!/usr/bin/env python3
"""Historical OOS final-market anchored fair-value research replay (NOT live).

This source uses FINAL odds available only after the race. It is NOT
L1.75 promotion and CANNOT be used to select bets or infer pre-bet edge.
Outcome/finish/payout columns in original CSV are ignored entirely.
"""
from __future__ import annotations
import argparse,csv,gzip,json,math
from pathlib import Path

CONTRACT="KEIBA_MARKET_FINAL_OOS_RESEARCH_ONLY_V1"
SOURCE_CONTRACT="MARKET_FIXED_ANCHOR_RESIDUAL_V2_FROM_CACHE"
VARIANTS=("MARKET_DIRECT","ANCHOR_ABILITY","ANCHOR_ABILITY_RISK","ANCHOR_FULL")

def safe_number(row,name):
    try:v=float(row[name])
    except (TypeError,ValueError,KeyError) as e:raise ValueError("FAIR_RESEARCH_NUMERIC_MISSING "+name) from e
    if not math.isfinite(v) or not 0<=v<=1:
        raise ValueError("FAIR_RESEARCH_PROB_INVALID "+name)
    return v

def replay(analysis,pred_file,meta_file,lambda_file):
    if analysis.get("contract")!="KEIBA_ONE_RACE_COMBINED_ANALYSIS_V1":
        raise ValueError("ONE_RACE_REPORT_CONTRACT_REQUIRED")
    rid=str(analysis.get("race_id") or "")
    if len(rid)!=12 or not rid.isascii() or not rid.isdigit() or not 2023<=int(rid[:4])<=2025:
        raise ValueError("FAIR_OOS_ONLY_2023_TO_2025_2026_SEALED")
    info=json.loads(Path(meta_file).read_text(encoding="utf-8"))
    if (info.get("contract")!=SOURCE_CONTRACT or
        info.get("promotion") is not False or
        info.get("2026_locked") is not True or
        info.get("roi_target_used_for_training") is not False or
        info.get("lambda_selection")!="prior-data chronological 80/20 only"):
        raise ValueError("FAIR_RESEARCH_PROMOTION_OR_TIME_POLICY_DRIFT")
    year=int(rid[:4])
    # Verify no lambda is selected on this evaluation year's outcomes.
    selected={}
    with open(lambda_file,encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            if int(row.get("test_year") or -1)!=year:continue
            if str(row.get("selected") or "") not in ("1","1.0"):continue
            variant=row["variant"]
            if variant not in VARIANTS[1:] or variant in selected:
                raise ValueError("FAIR_OOS_LAMBDA_AMBIGUOUS")
            val=float(row["lambda"])
            if not math.isfinite(val) or val<=0:raise ValueError("FAIR_OOS_LAMBDA_INVALID")
            selected[variant]=val
    if set(selected)!=set(VARIANTS[1:]):
        raise ValueError("FAIR_OOS_SELECTED_LAMBDA_COVERAGE_DRIFT")
    horses=analysis.get("horses") or []
    expected={str(h["horse_id"]) for h in horses}
    if len(expected)!=len(horses) or len(expected)<3:
        raise ValueError("FAIR_TARGET_HORSE_ID_INVALID")
    results={}
    with gzip.open(pred_file,"rt",encoding="utf-8",newline="") if str(pred_file).endswith(".gz") else open(pred_file,encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("_race_id"))!=rid:continue
            if int(row.get("year") or -1)!=year or str(row.get("_race_date") or "")[:4]!=str(year):
                raise ValueError("FAIR_OOS_YEAR_DATE_DRIFT")
            hid=str(row.get("_horse_id") or "")
            if not hid or hid in results or hid not in expected:
                raise ValueError("FAIR_OOS_UNKNOWN_OR_DUPLICATE_HORSE")
            p={name:safe_number(row,"P__"+name) for name in VARIANTS}
            q=safe_number(row,"market_prob_norm")
            if abs(p["MARKET_DIRECT"]-q)>1e-7:
                raise ValueError("FAIR_OOS_MARKET_BASE_PROBABILITY_DRIFT")
            results[hid]={
                "source_kind":"FINAL_HISTORICAL_ODDS_ONLY",
                "market_direct_final":q,
                "oos_win_probabilities":p,
                "risk_score_historical":safe_number(row,"risk"),
            }
    if set(results)!=expected:raise ValueError("FAIR_OOS_FIELD_COVERAGE_DRIFT")
    for variant in VARIANTS:
        total=sum(h["oos_win_probabilities"][variant] for h in results.values())
        if abs(total-1)>1e-4:
            raise ValueError("FAIR_OOS_PROBS_RACE_NORMALIZATION_DRIFT "+variant)
    return {
        "contract":CONTRACT,"race_id":rid,
        "source_run":"38659856085","source_research_contract":SOURCE_CONTRACT,
        "status":"HISTORICAL_FINAL_ODDS_OOS_RESEARCH_REPLAY",
        "source_has_final_odds":True,
        "is_prestart_market":False,
        "is_production_l175":False,
        "promotion":False,
        "fair_probability_must_not_be_used_with_current_quote_without_retraining":True,
        "train_years_expected":[y for y in (2021,2022,2023,2024) if y<year],
        "lambda_prior_only":selected,
        "model_count":len(VARIANTS),"horse_count":len(results),
        "horse_results":results,
        "warning":"This saved prediction already incorporates the historical FINAL win odds. Do NOT merge with a timestamped live pre-race odds quote or use as a live value/EV signal."
    }

def main():
    p=argparse.ArgumentParser()
    for flag in ("analysis","predictions","summary","lambda-selection","output"):
        p.add_argument("--"+flag,required=True)
    a=p.parse_args()
    src=json.loads(Path(a.analysis).read_text(encoding="utf-8"))
    out=replay(src,a.predictions,a.summary,a.lambda_selection)
    dst=Path(a.output);dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FINAL_MARKET_OOS_RESEARCH_REPLAY_READY",out["race_id"],out["horse_count"],"not_live",flush=True)
if __name__=="__main__":main()
