#!/usr/bin/env python3
"""Exact saved 2025 LightGBM model-output reproduction on one archived snapshot
race. This is a MODEL REPLAY, NOT a historically blind pre-race prediction:
the persisted 2025 models used full-2025 validation labels for early stopping.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,math
from datetime import date
from pathlib import Path

EXPERTS=("core4","pedlegacy","condition","full","light","pedv1","condrc")
RACE_ID="202507010212"

def lines(path):
    op=gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)

def verify(snapshot,root,race_id,council_file):
    import lightgbm as lgb
    import pandas as pd
    if int(race_id[:4])>=2026:raise ValueError("2026_SEALED")
    rows=[]
    for row in lines(snapshot):
        if str(row.get("race_id") or "")==race_id:
            if row.get("leakage_policy")!="STRICT_PRIOR_DATE_ONLY" or row.get("feature_schema_version")!=8:
                raise ValueError("SNAPSHOT_CONTRACT_DRIFT")
            # Never copy target/finish labels or final odds into the scorer.
            rows.append({"horse_id":str(row.get("horse_id") or ""),
                "race_date":str((row.get("features") or {}).get("race_date") or row.get("race_date") or ""),
                "features":dict(row.get("features") or {})})
    if len(rows)<3 or len({x["horse_id"] for x in rows})!=len(rows):
        raise ValueError("TARGET_RACE_NOT_FOUND_OR_HORSE_COVERAGE_DRIFT")
    d=date.fromisoformat(rows[0]["race_date"][:10])
    if any(r["race_date"][:10]!=d.isoformat() for r in rows):
        raise ValueError("TARGET_RACE_DATE_MISMATCH")
    council=[r for r in lines(council_file) if str(r.get("race_id"))==race_id]
    if len(council)!=1:raise ValueError("ORIGINAL_SEVEN_KING_COUNCIL_SOURCE_RACE_MISSING")
    archived=council[0]
    if str(archived.get("race_date"))!=d.isoformat():
        raise ValueError("SAVED_COUNCIL_DATE_DRIFT")
    ids=[str(h) for h in archived.get("horses") or []]
    matrix=archived.get("rank_matrix") or []
    if len(matrix)!=7 or len(ids)!=len(rows) or set(ids)!={r["horse_id"] for r in rows}:
        raise ValueError("ARCHIVED_7KING_FIELD_OR_EXPERT_COVERAGE_DRIFT")
    report=[]
    for alias in EXPERTS:
        folder=Path(root)/alias
        meta=json.loads((folder/"model.json").read_text(encoding="utf-8"))
        schema=json.loads((folder/"schema.json").read_text(encoding="utf-8"))
        model=folder/"model.txt"
        sha=hashlib.sha256(model.read_bytes()).hexdigest()
        if sha!=meta.get("reproducibility",{}).get("model_sha256") or sha!=schema.get("reproducibility",{}).get("model_sha256"):
            raise ValueError("MODEL_ARTIFACT_CHECKSUM_DRIFT "+alias)
        if meta.get("prediction_phase")!="FINAL":
            raise ValueError("SOURCE_PHASE_DRIFT "+alias)
        split=meta.get("split") or {}
        if split.get("valid_start")!=f"{d.year}-01-01" or split.get("valid_end")!=f"{d.year}-12-31":
            raise ValueError("VALIDATION_FOLD_DRIFT "+alias)
        if split.get("train_end")>=split["valid_start"]:
            raise ValueError("MODEL_TRAIN_SPLIT_LEAK "+alias)
        features=schema["feature_order"]
        levels=schema["category_levels"]
        booster=lgb.Booster(model_file=str(model))
        if booster.feature_name()!=features:raise ValueError("MODEL_FEATURE_ORDER_DRIFT "+alias)
        values=[]
        for row in rows:
            raw={k:v for k,v in row["features"].items() if k not in
                 ("actual_start_time","jockey_id","trainer_id","race_date")}
            raw["race_month"]=d.month
            raw["race_day_of_year"]=d.timetuple().tm_yday
            values.append({key:raw.get(key) for key in features})
        dataframe=pd.DataFrame(values,columns=features)
        for key in features:
            if key in levels:
                text=dataframe[key].astype("string").fillna("__MISSING__")
                dataframe[key]=pd.Categorical(text,categories=levels[key])
            else:
                dataframe[key]=pd.to_numeric(dataframe[key],errors="coerce")
        preds=booster.predict(dataframe,num_iteration=int(meta["best_iteration"]))
        actual={hid:int(rank) for hid,rank in zip(ids,matrix[EXPERTS.index(alias)])}
        order=sorted(rows,key=lambda row:(-float(preds[[r["horse_id"] for r in rows].index(row["horse_id"])]),row["horse_id"]))
        guessed={row["horse_id"]:i+1 for i,row in enumerate(order)}
        if sorted(actual.values())!=list(range(1,len(rows)+1)):
            raise ValueError("BAD_ARCHIVED_EXPERT_RANK_PERMUTATION "+alias)
        delta=max(abs(guessed[hid]-actual[hid]) for hid in actual)
        if delta:
            raise ValueError("MODEL_VS_ORIGINAL_TIE_SAFE_COUNCIL_RANK_DRIFT "+alias+" "+str(delta))
        report.append({"model":alias,"feature_count":len(features),
                       "starters":len(rows),"max_abs_rank_difference":delta,
                       "source_phase":"FINAL",
                       "source_valid_year":d.year,
                       "genuine_asof_target_year":False,
                       "reason":"EARLY_STOPPING_SELECTED_USING_VALIDATION_YEAR_RESULTS"})
    return {"contract":"KEIBA_EXACT_RESEARCH_MODEL_REPRODUCTION_V1",
            "race_id":race_id,"starters":len(rows),"saved_models_reproduced":len(report),
            "target_labels_used_in_prediction_matrix":False,
            "genuine_pre_race_inference":False,
            "models":report}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snapshot",required=True)
    ap.add_argument("--bundles",required=True)
    ap.add_argument("--council-file",required=True)
    ap.add_argument("--race",default=RACE_ID)
    args=ap.parse_args()
    data=verify(args.snapshot,args.bundles,args.race,args.council_file)
    for r in data["models"]:
        print("EXACT_SAVED_KING_MODEL_REPRODUCED",json.dumps(r,separators=(",",":")),flush=True)
    print("SEVEN_KING_ACTUAL_MODEL_OUTPUT_MATCHES_ORIGINAL_SAVED_TIE_SAFE_RANKS",
          data["race_id"],data["saved_models_reproduced"],data["starters"],flush=True)

if __name__=="__main__":main()
