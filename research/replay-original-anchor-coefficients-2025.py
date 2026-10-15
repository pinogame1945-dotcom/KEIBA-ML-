#!/usr/bin/env python3
"""Exact ORIGINAL 2025 ANCHOR_FULL residual replay on historical FINAL cache.

Recover saved model beta from residual-coefficients.csv and RECONSTRUCT
training-only (2022-2024) median/mean/population-std transform exactly.
No train/val refit, no 2025 labels/outcomes used; no 2026 historical records.
The 2025 feature panel used for parity contains retrospective FINAL odds,
so this is not yet an authentic live/prebet model deployment.
"""
from __future__ import annotations
import argparse,csv,gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path

VARIANT="ANCHOR_FULL"
FEATURES=("field_size","distill_ai_rank01","distill_score_z","distill_score_gap_to_best",
    "distill_is_ai_top1","distill_is_ai_top3","distill_is_ai_top6","risk",
    "rank_gap_norm","abs_rank_gap_norm","top3_overlap_norm","ai_unique_top3",
    "market_unique_top3","top1_match","risk_x_abs_gap","ability_x_market_gap",
    "ability_x_disagree_level","risk_x_disagree_level","rankgap_x_disagree_level")
OUTCOME_FIELDS=("target_win","target_top3","final_win_odds","_target_gap","_target_over3",
                "payout","finish","target","result","hit","roi")
SOURCE="MARKET_FIXED_ANCHOR_RESIDUAL_V2_FROM_CACHE"
CONTRACT="KEIBA_ORIGINAL_ANCHOR_FULL_COEFF_REPLAY_FINAL_ONLY_V1"

def file_rows(path):
    with (gzip.open(path,"rt",encoding="utf-8",newline="") if str(path).endswith(".gz")
          else open(path,encoding="utf-8",newline="")) as fd:
        yield from csv.DictReader(fd)

def safe_float(s,label):
    try:
        x=float(s)
    except (TypeError,ValueError) as e:
        raise ValueError("INVALID_FLOAT_"+label) from e
    if not math.isfinite(x):raise ValueError("NONFINITE_"+label)
    return x

def optional_float(s):
    if s is None or str(s).strip().lower() in ("","nan","none","null","inf","-inf"):
        return None
    try:f=float(s)
    except ValueError:return None
    return f if math.isfinite(f) else None

def build_2025_normalizer(train_rows):
    values={f:[] for f in FEATURES}
    count=0
    for item in train_rows:
        year=int(item["year"])
        if year>=2025:continue
        if not 2022<=year<=2024:raise ValueError("TRAIN_YEAR_INVALID_FOR_SAVED_2025_COEFF")
        count+=1
        for feature in FEATURES:
            if feature not in item and feature not in ("ability_x_disagree_level",
                    "risk_x_disagree_level","rankgap_x_disagree_level"):
                raise ValueError("FEATURE_MISSING_IN_SOURCE_"+feature)
            v=optional_float(compute_feature(item,feature))
            if v is not None:values[feature].append(v)
    if count<1000:raise ValueError("INSUFFICIENT_PRIOR_TRAINING_HORSES")
    med={name:(float(statistics.median(a)) if a else 0.0)
         for name,a in values.items()}
    normal={}
    for feature,a in values.items():
        v=med[feature]
        mu=math.fsum(a)/count+(count-len(a))*v/count
        variance=(math.fsum((n-mu)**2 for n in a)+
                  (count-len(a))*(v-mu)**2)/count
        sd=math.sqrt(max(variance,0.0)) or 1.0
        normal[feature]={"median":v,"mean":mu,"std":sd}
    return normal,count

def compute_feature(row,feature):
    if feature in ("ability_x_disagree_level","risk_x_disagree_level","rankgap_x_disagree_level"):
        overlap=optional_float(row.get("top3_overlap"))
        if overlap is None:return None
        dis=(3-max(0,min(3,overlap)))/3
        parent={"ability_x_disagree_level":"distill_score_z",
                "risk_x_disagree_level":"risk",
                "rankgap_x_disagree_level":"rank_gap_norm"}[feature]
        val=optional_float(row.get(parent))
        return None if val is None else val*dis
    return row.get(feature)

def read_beta(path,lambda_path):
    actual={}
    for row in file_rows(path):
        if row.get("test_year")!="2025" or row.get("variant")!=VARIANT:
            continue
        feature=row["feature"]
        if feature in actual:raise ValueError("COEFFICIENT_DUPLICATE")
        actual[feature]=safe_float(row["coef"],"coef_"+feature)
    if set(actual)!=set(FEATURES):
        raise ValueError("ORIGINAL_ANCHOR_FULL_COEFFICIENT_SCHEMA_DRIFT")
    selected=[]
    for r in file_rows(lambda_path):
        if (r.get("test_year")=="2025" and r.get("variant")==VARIANT and
            r.get("selected") in ("1","1.0")):
            selected.append(safe_float(r["lambda"],"selected_lambda"))
    if len(selected)!=1:raise ValueError("SELECTED_TRAIN_ONLY_LAMBDA_AMBIGUOUS")
    return actual,selected[0]

def predict_one_race(rows,normal,beta):
    seen=set();scores=[]
    for row in rows:
        hid=str(row.get("_horse_id") or "")
        if hid in seen or not hid:raise ValueError("HORSE_COVERAGE_BAD")
        seen.add(hid)
        p=safe_float(row["market_prob_norm"],"market_prob_norm")
        if not 0<p<=1:raise ValueError("MARKET_PROBABILITY_NOT_NORMALIZED")
        z=math.log(p)
        for feat in FEATURES:
            raw=optional_float(compute_feature(row,feat))
            m=normal[feat]
            if raw is None:raw=m["median"]
            z+=(raw-m["mean"])/m["std"]*beta[feat]
        scores.append((hid,z))
    if len(scores)<3 or len(scores)>24:raise ValueError("INVALID_RACE_SIZE")
    m=max(x[1] for x in scores)
    values=[math.exp(x-m) for _,x in scores]
    total=math.fsum(values)
    return {hid:v/total for (hid,_),v in zip(scores,values)}

def execute(cache,coeff,lambdas,predictions,race_id,fail_threshold=5e-7):
    if len(race_id)!=12 or not race_id.startswith("2025") or not race_id.isdigit():
        raise ValueError("ONLY_HISTORICAL_2025_PARITY_NO_2026_HINDSIGHT")
    # First pass trains only the *standardizer*, not any coefficient or
    # label-calibrated model. Coefficients are already frozen original OOS.
    normal,ntrain=build_2025_normalizer(file_rows(cache))
    beta,selected_lambda=read_beta(coeff,lambdas)
    panel=[row for row in file_rows(cache) if row.get("_race_id")==race_id]
    if not panel or any(int(row["year"])!=2025 for row in panel):
        raise ValueError("CACHED_2025_RACE_ABSENT")
    probabilities=predict_one_race(panel,normal,beta)
    expected={}
    for row in file_rows(predictions):
        if row.get("_race_id")!=race_id:continue
        h=row.get("_horse_id")
        if h in expected:raise ValueError("ORIGINAL_OOS_DUPLICATE_HORSE")
        expected[h]=safe_float(row["P__ANCHOR_FULL"],"P__ANCHOR_FULL")
    if set(expected)!=set(probabilities):
        raise ValueError("RECONSTRUCTION_HORSE_ID_DRIFT")
    error=max(abs(expected[h]-v) for h,v in probabilities.items())
    if error>fail_threshold:
        raise ValueError("ORIGINAL_OOS_COEFFICIENT_REPRODUCTION_DRIFT "+str(error))
    return {
        "contract":CONTRACT,
        "race_id":race_id,"training_data_years":[2022,2023,2024],
        "heldout_year":2025,"lambda_prior_only":selected_lambda,
        "original_source_phase":"FINAL",
        "prediction_prebet_time_verified":False,
        "original_risk_and_ability_prebet_time_verified":False,
        "market_anchor_prebet_time_verified":False,
        "not_a_promoted_l175_live_model":True,
        "no_2025_labels_read":True,"no_2026_data_read":True,
        "original_fit_coefficients_not_retrained":True,
        "standardizer_train_horses":ntrain,
        "original_anchored_probs_match":True,
        "max_absolute_prob_difference":error,
        "horse_count":len(panel),
        "horses":[{"horse_id":hid,"replayed_pwin":p,"original_saved_pwin":expected[hid]}
                  for hid,p in sorted(probabilities.items())],
    }

def main():
    p=argparse.ArgumentParser()
    for k in ("cache","coefficients","lambda-selection","predictions","race","out"):
        p.add_argument("--"+k,required=True)
    a=p.parse_args()
    out=execute(a.cache,a.coefficients,a.lambda_selection,a.predictions,a.race)
    target=Path(a.out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ORIGINAL_ANCHOR_FULL_OOS_COEFFICIENT_RECONSTRUCTION",out["race_id"],
          "horses",out["horse_count"],"MAX_DIFF",out["max_absolute_prob_difference"],flush=True)

if __name__=="__main__":main()
