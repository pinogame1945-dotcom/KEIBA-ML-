#!/usr/bin/env python3
"""Operational 2026-blind refit of ONE original seven-king LightGBM expert.

Not a modification of the frozen walk-forward research. Reuses exact original
flatten(), complete_races(), frames(), safety checks and params from
train-staged-lightgbm.py. Uses the earlier 2025 OOS fold's best_iteration
as the fixed 2026 refit round count. Trains on 2024+2025 ONLY. No 2026
outcomes, validation, early stopping, or tuning. Outputs a strictly
provenanced, candidate deployment bundle; race betting stays disabled.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import lightgbm as lgb
import pandas as pd

GEN="eb13d4096519167b"
CONFIG="research/l1-raceclass-absorption-walkforward-v1.json"
ORIG="research/train-staged-lightgbm.py"
MODEL_RELEASE="KEIBA_7KING_2026_BLIND_MODEL_RELEASE_V1"
CANDIDATES={
 "core4":("core4_no_pedigree",649),
 "pedlegacy":("no_auto_full_pedigree_legacy",470),
 "condition":("jockey_trainer_condition",423),
 "full":("no_auto_full",266),
 "light":("no_auto_jockey",164),
 "pedv1":("no_auto_full_pedigree_v1",504),
 "condrc":("jockey_trainer_condition_plus_raceclass",457),
}
MODEL_PARAMS_KEY="original_2025_early_stop_best_iteration"

def check(ok,msg):
    if not ok: raise ValueError(msg)

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()

def frozen_trainer():
    src=Path(ORIG)
    spec=importlib.util.spec_from_file_location("keiba_frozen_train_module",src)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def frozen_params(module):
    source=Path(ORIG).read_text(encoding="utf-8")
    tree=ast.parse(source)
    blocks=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=="params" for t in node.targets)
            and isinstance(node.value,ast.Dict)]
    check(len(blocks)==1,"ORIGINAL_TRAINER_PARAMS_CANNOT_BE_UNIQUELY_EXTRACTED")
    output={}
    for k,v in zip(blocks[0].value.keys,blocks[0].value.values):
        name=ast.literal_eval(k)
        if isinstance(v,ast.Name) and v.id=="LGBM_N_JOBS":
            output[name]=module.LGBM_N_JOBS
        else:
            output[name]=ast.literal_eval(v)
    check(output.get("objective")=="binary" and output.get("n_estimators")==600
          and output.get("learning_rate")==0.03 and
          output.get("random_state")==42 and
          output.get("histogram_pool_size")==1024,
          "ORIGINAL_TRAINING_HYPERPARAMETER_DRIFT")
    return output

def train_one(king,prior_meta_file,prior_schema_file,snapshot_paths,destination,approval_ref):
    check(king in CANDIDATES,"UNRECOGNIZED_ORIGINAL_KING")
    check(len(snapshot_paths)==2 and
          {p.name for p in snapshot_paths}=={
              "snapshot-2024.jsonl.gz","snapshot-2025.jsonl.gz"},
          "EXACTLY_2024_AND_2025_PRIOR_SNAPSHOTS_REQUIRED")
    check(all(p.is_file() and p.stat().st_size>0 for p in snapshot_paths),
          "PRIOR_TRAINING_SNAPSHOT_FILE_MISSING")
    check(not destination.exists(),"REFUSE_OVERWRITE_TRAINED_KING")
    check(approval_ref.strip() and "2026" in approval_ref,
          "REFIT_PROTOCOL_REFERENCE_MISSING")
    frozen=read(CONFIG)
    check(frozen["contract"]=="L1_RACECLASS_ABSORPTION_WALKFORWARD_V1"
          and frozen["snapshot_generation"]==GEN and
          frozen["training_window_years"]==2 and
          frozen["locked_years"]==[2026] and
          frozen["feature_selection"]=="none" and
          frozen["prediction_phase"]=="FINAL" and
          frozen["train_race_class"]=="ALL" and
          frozen["valid_race_class"]=="ALL" and
          frozen["ability_uses_odds"] is False,
          "FROZEN_2026_YEAR_AND_CANDIDATE_CONTRACT_DRIFT")
    candidate_name,n_features=CANDIDATES[king]
    candidates=[c for c in frozen["candidates"] if c["name"]==candidate_name]
    check(len(candidates)==1,"ORIGINAL_KING_MISSING_FROM_FROZEN_CONFIG")
    candidate=candidates[0]
    prior=read(prior_meta_file)
    previous_schema=read(prior_schema_file)
    prior_split=prior.get("split") or {}
    check(prior_split.get("train_start")=="2023-01-01" and
          prior_split.get("train_end")=="2024-12-31" and
          prior_split.get("valid_start")=="2025-01-01" and
          prior_split.get("valid_end")=="2025-12-31" and
          prior.get("feature_count")==n_features and
          len(previous_schema.get("feature_order") or [])==n_features,
          "NOT_THE_FROZEN_2025_OUT_OF_SAMPLE_KING:"+king)
    rounds=prior.get("best_iteration")
    check(type(rounds) is int and 1<=rounds<=600,
          "2025_ORIGINAL_VALIDATED_BEST_ITERATION_REQUIRED:"+king)

    trainer=frozen_trainer()
    params=frozen_params(trainer)
    contract=trainer.load_feature_contract(trainer.DEFAULT_FEATURE_CONTRACT_PATH)
    sets=trainer.normalize_feature_sets(",".join(candidate["feature_sets"]),None,contract)
    check(sets==candidate["feature_sets"],"ORIGINAL_FEATURE_FAMILY_ORDER_DRIFT")
    import gzip
    with gzip.open(snapshot_paths[0],"rt",encoding="utf-8") as first_source:
        first=json.loads(next(line for line in first_source if line.strip()))
    history=trainer.normalize_history_windows(
        json.dumps(first.get("history_windows")),contract)
    small=trainer.normalize_small_sample_policy(
        json.dumps(first.get("small_sample_policy")),
        read(trainer.DEFAULT_SMALL_SAMPLE_CONTRACT_PATH),history)
    check(first.get("ml_dataset_version")==3 and
          first.get("feature_schema_version")==8 and
          first.get("leakage_policy")=="STRICT_PRIOR_DATE_ONLY",
          "PRIOR_2024_SOURCE_SNAPSHOT_CONTRACT_DRIFT")
    blocked=trainer.load_forbidden_model_keys(
        Path("contracts/l1-feature-catalog-v1.json"))
    # A copied/renamed or duplicated snapshot must never silently change
    # the two-year training population. Input names and row years must match.
    def prior_rows():
        years=set()
        seen=set()
        for path in snapshot_paths:
            expected_year=path.name.split("-")[1].split(".")[0]
            for row in trainer.iter_dataset_rows(path):
                date=str((row.get("features") or {}).get("race_date") or "")
                check(date[:4]==expected_year,
                      "TRAINING_ROW_YEAR_DOES_NOT_MATCH_SNAPSHOT:"+path.name)
                years.add(date[:4])
                identity=(str(row.get("race_id") or ""),str(row.get("horse_id") or ""))
                check(all(identity) and identity not in seen,
                      "PRIOR_TRAINING_HORSE_DUPLICATED_OR_UNIDENTIFIED")
                seen.add(identity)
                check((row.get("prediction_phase") or "FINAL")=="FINAL",
                      "NOT_THE_ORIGINAL_FINAL_PHASE")
                yield row
        check(years=={"2024","2025"},"TRAINING_DATA_YEAR_NOT_PRESENT")
    df=trainer.flatten(prior_rows(),sets,"FINAL",history,small,contract,blocked)
    check(not df.empty and
          set(df["_race_date_dt"].dt.year.unique())=={2024,2025},
          "TRAINING_OUTSIDE_2024_2025_OR_MISSING_YEAR")
    train=trainer.complete_races(df)
    check(not train.empty and train["_target"].nunique()==2 and
          set(train["_race_date_dt"].dt.year.unique())=={2024,2025},
          "TRAINING_WINNER_LABEL_INCOMPLETE")
    train_rows=len(train)
    train_races=train["_race_id"].nunique()
    xtr,_,categorical,category_levels=trainer.frames(
        train,train.iloc[0:0],None)
    check(xtr.shape==(train_rows,n_features),
          "ORIGINAL_SEVEN_KING_2026_TRAINING_FEATURE_COUNT_DRIFT:"+king)
    check(list(xtr.columns)==previous_schema["feature_order"],
          "FROZEN_2025_TO_2026_FEATURE_ORDER_DRIFT:"+king)
    trainer.assert_catalog_safety(list(xtr.columns),
                                  Path("contracts/l1-feature-catalog-v1.json"))
    trainer.assert_catalog_availability(list(xtr.columns),
        Path("contracts/l1-feature-catalog-v1.json"),"FINAL",contract)
    trainer.assert_prediction_phase_safety(list(xtr.columns),"FINAL",contract)
    labels=train["_target"].astype(int)
    del train, df

    dataset=lgb.Dataset(xtr,label=labels,categorical_feature=categorical,
                        free_raw_data=True)
    dataset.construct()
    del xtr,labels
    native=dict(params)
    native.pop("n_estimators")
    native["metric"]="binary_logloss"
    # A 2025 OOS validation-selected round count is reused; NO 2026
    # validation set, outcome or early-stopping callbacks are provided.
    booster=lgb.train(native,dataset,num_boost_round=rounds)
    check(booster.current_iteration()==rounds,
          "BLIND_2026_REFIT_ITERATION_DRIFT")
    check(booster.feature_name()==previous_schema["feature_order"],
          "ORIGINAL_TRAINED_BOOSTER_SCHEMA_DRIFT")
    dest=destination
    dest.mkdir(parents=True)
    model_file=dest/"model.txt"
    schema_file=dest/"schema.json"
    meta_file=dest/"model.json"
    booster.save_model(str(model_file))
    model_sha=hash_file(model_file)
    model_version="L1_7KING_2026_BLIND_REFIT_"+candidate_name.upper()
    schema={
        "contract":"L1_FEATURE_SCHEMA_V1",
        "model_version":model_version,
        "ml_dataset_version":3,
        "feature_schema_version":8,
        "leakage_policy":"STRICT_PRIOR_DATE_ONLY",
        "ability_uses_odds":False,
        "prediction_phase":"FINAL",
        "training_years":[2024,2025],
        "feature_order":booster.feature_name(),
        "category_levels":category_levels,
    }
    schema_file.write_text(json.dumps(schema,ensure_ascii=False,indent=2)+"\n",
                           encoding="utf-8")
    meta={
        "contract":MODEL_RELEASE,
        "deployment_status":"CANDIDATE_2026_BLIND_UNRELEASED",
        "candidate_name":candidate_name,
        "prediction_year":2026,
        "training_years":[2024,2025],
        "train_start":"2024-01-01",
        "train_end":"2025-12-31",
        "snapshot_generation":GEN,
        "2026_outcomes_used":False,
        "2026_validation_used":False,
        "ability_uses_odds":False,
        "feature_selection":"none",
        "prediction_phase":"FINAL",
        "training_rule_id":"REUSE_2025_OOS_BEST_ITERATION_REFIT_2024_2025",
        "training_rule_approval_ref":None,
        "refit_protocol_reference":approval_ref,
        "best_iteration":rounds,
        MODEL_PARAMS_KEY:rounds,
        "training_rows":train_rows,
        "training_races":int(train_races),
        "model_version":model_version,
        "model_sha256":model_sha,
        "schema_sha256":hash_file(schema_file),
        "feature_sets":sets,
        "actor_prefixes":candidate.get("actor_prefixes") or [],
        "auto_slices":candidate.get("auto_slices") or [],
        "pedigree_slices":candidate.get("pedigree_slices") or [],
        "params":params,
        "training_config_sha256":hashlib.sha256(json.dumps(
            {"king":candidate_name,"year":[2024,2025],"params":params,
             "best_iteration_source":"2025-OOS","rounds":rounds},
            sort_keys=True,separators=(",",":")).encode()).hexdigest(),
    }
    meta_file.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",
                         encoding="utf-8")
    print("FROZEN_2026_7KING_PRIOR_ONLY_REFIT_CANDIDATE_COMPLETED "+
          json.dumps({"king":king,"training_rows":train_rows,
                      "training_races":int(train_races),"rounds":rounds,
                      "model_sha256":model_sha,
                      "trained_on_2026_outcomes":False},sort_keys=True),
          flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--king",required=True,choices=tuple(CANDIDATES))
    parser.add_argument("--prior-model-meta",type=Path,required=True)
    parser.add_argument("--prior-model-schema",type=Path,required=True)
    parser.add_argument("--prior-2024-projection",type=Path,required=True)
    parser.add_argument("--prior-2025-projection",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--approval-ref",required=True)
    a=parser.parse_args()
    train_one(a.king,a.prior_model_meta,a.prior_model_schema,
              [a.prior_2024_projection,a.prior_2025_projection],
              a.out,a.approval_ref)

if __name__=="__main__":
    main()
