#!/usr/bin/env python3
"""Audit already trained 7-king FINAL-phase Kaggle bundles; never promote
validation-year-early-stopped models as genuine same-year as-of inference.
Read-only, CPU-only and no 2026 material.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path

EXPERTS=("core4","pedlegacy","condition","full","light","pedv1","condrc")
DATASET="L1_FEATURE_SCHEMA_V1"

def audit_one(directory,alias,year,*,load_model=True):
    import lightgbm as lgb
    import pandas as pd
    d=Path(directory)
    paths={name:d/name for name in ("model.txt","model.json","schema.json","candidate.json")}
    if any(not p.is_file() for p in paths.values()):
        raise ValueError("MISSING_BUNDLE_FILE "+alias)
    meta=json.loads(paths["model.json"].read_text(encoding="utf-8"))
    schema=json.loads(paths["schema.json"].read_text(encoding="utf-8"))
    cand=json.loads(paths["candidate.json"].read_text(encoding="utf-8"))
    if schema.get("contract")!=DATASET or meta.get("prediction_phase")!="FINAL" or schema.get("prediction_phase")!="FINAL":
        raise ValueError("SOURCE_CONTRACT_OR_PHASE_DRIFT")
    if meta.get("feature_sets")!=schema.get("feature_sets"):
        raise ValueError("FEATURE_SET_DRIFT")
    if meta.get("features")!=schema.get("feature_order"):
        raise ValueError("FEATURE_ORDER_DRIFT")
    if meta.get("category_levels")!=schema.get("category_levels"):
        raise ValueError("CATEGORICAL_LEVELS_DRIFT")
    if meta.get("dataset_contract",{}).get("feature_schema_version")!=8:
        raise ValueError("FEATURE_SCHEMA_V8_DRIFT")
    if meta.get("ability_uses_odds") is not False or schema.get("ability_uses_odds") is not False:
        raise ValueError("ABILITY_MARKET_LEAKAGE")
    if meta.get("history_windows")!=schema.get("history_windows") or meta.get("small_sample_policy")!=schema.get("small_sample_policy"):
        raise ValueError("SOURCE_HISTORY_POLICY_DRIFT")
    cfg=cand[0] if isinstance(cand,list) and len(cand)==1 else None
    if not cfg or cfg.get("feature_sets")!=meta["feature_sets"]:
        raise ValueError("CANDIDATE_CONFIG_DRIFT")
    if alias not in EXPERTS or int(year) not in range(2021,2026):
        raise ValueError("UNKNOWN_CANDIDATE_OR_2026_LOCK")
    split=meta.get("split") or {}
    if str(split.get("valid_start"))!=f"{year}-01-01" or str(split.get("valid_end"))!=f"{year}-12-31":
        raise ValueError("VALID_FOLD_DRIFT")
    if str(split.get("train_end"))>=str(split.get("valid_start")):
        raise ValueError("TRAIN_VALID_SPLIT_DRIFT")
    checksum=hashlib.sha256(paths["model.txt"].read_bytes()).hexdigest()
    if checksum!=meta.get("reproducibility",{}).get("model_sha256"):
        raise ValueError("MODEL_FILE_SHA256_DRIFT")
    if checksum!=schema.get("reproducibility",{}).get("model_sha256"):
        raise ValueError("SCHEMA_MODEL_SHA256_DRIFT")
    feats=list(schema["feature_order"])
    if not feats or len(set(feats))!=len(feats):
        raise ValueError("INVALID_FEATURE_ORDER")
    checked=None
    if load_model:
        booster=lgb.Booster(model_file=str(paths["model.txt"]))
        if booster.feature_name()!=feats or booster.num_feature()!=len(feats):
            raise ValueError("LIGHTGBM_NATIVE_FEATURE_ORDER_DRIFT")
        levels=schema["category_levels"]
        data={name:(pd.Categorical(["__MISSING__"],categories=levels[name]) if name in levels else [float("nan")]) for name in feats}
        frame=pd.DataFrame(data)
        result=booster.predict(frame,num_iteration=int(meta["best_iteration"]))
        if len(result)!=1 or not math.isfinite(float(result[0])):
            raise ValueError("MODEL_SYNTHETIC_SCORE_NOT_FINITE")
        checked=round(float(result[0]),6)
    return {
        "alias":alias,"year":year,"feature_count":len(feats),
        "train_end":split["train_end"],"validation_end":split["valid_end"],
        "prediction_phase":meta["prediction_phase"],"model_sha256":checksum,
        "model_loaded_and_predicts":bool(load_model),
        "one_synthetic_missing_feature_score":checked,
        "target_year_asof_eligible":False,
        "reason":"EARLY_STOPPING_USED_TARGET_YEAR_VALIDATION_LABELS; also FINAL archived feature availability lacks pre-bet timestamps",
        "not_a_target_year_blind_model":True
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--year",type=int,default=2025)
    p.add_argument("--no-score",action="store_true")
    args=p.parse_args()
    if args.year not in range(2021,2026):raise ValueError("year outside allowed historical research")
    data=[]
    for alias in EXPERTS:
        row=audit_one(Path(args.root)/alias,alias,args.year,load_model=not args.no_score)
        print("MODEL_BUNDLE_AUDIT",json.dumps(row,separators=(",",":")),flush=True)
        data.append(row)
    if len(data)!=7 or any(item["target_year_asof_eligible"] for item in data):
        raise ValueError("AUDIT_FAILED")
    print("SAVED_7KING_MODEL_BUNDLES_PRESENT_AND_NOT_ASOF_ELIGIBLE_FOR_SAME_YEAR",
          len(data),flush=True)

if __name__=="__main__":main()
