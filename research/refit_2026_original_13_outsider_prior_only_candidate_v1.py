#!/usr/bin/env python3
"""Original 13 Outsider candidate-only 2026 refit (NEVER a frozen model release).

The historical 2025 fold supplies each expert's actual best_iteration from
the exact original 2023-24 training / 2025 validation. Refit uses the SAME
frozen LightGBM implementation/params and the complete 2024-25 population.
The 2023 year is read solely for original feature schema stability: it is
NOT included in the 2026 refit training rows. 2026 rows/results never enter.
This is a research CANDIDATE, not an approved change to the frozen strategy.

Inputs: one original 2023-25 mode dataset and 2025 original OOS meta/schema.
Models must NOT be stored on public GitHub or exposed to L1.75/L2/L3.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb

from refit_2026_seven_king_prior_only_v1 import (
    frozen_trainer, frozen_params, hash_file, check,
)

GENERATION="eb13d4096519167b"
CONTRACT="KEIBA_ORIGINAL_13_OUTSIDER_2026_CANDIDATE_REFIT_V1"
TRAIN_YEARS={2024,2025}
SOURCE_YEARS={2023,2024,2025}
PRIOR_START="2023-01-01"
PRIOR_END="2024-12-31"
PRIOR_VALID_START="2025-01-01"
PRIOR_VALID_END="2025-12-31"

def config():
    entries={}
    for path in ("research/l1-outsider-nonhorse-v1.json",
                 "research/l1-outsider-transition-v1.json"):
        c=json.loads(Path(path).read_text(encoding="utf-8"))
        check(c.get("snapshot_generation")==GENERATION and
              c.get("training_window_years")==2 and
              c.get("prediction_phase")=="FINAL" and
              c.get("feature_selection")=="none" and
              c.get("ability_uses_odds") is False and
              c.get("locked_years")==[2026],
              "ORIGINAL_13_OUTSIDER_FROZEN_CONFIG_CHANGED")
        for item in c["candidates"]:
            key=item["name"]
            check(key not in entries and item.get("feature_sets")==["BASE"],
                  "ORIGINAL_13_OUTSIDER_FAMILY_CHANGED")
            entries[key]=item
    check(len(entries)==13,"ORIGINAL_13_OUTSIDER_EXPERT_COUNT_CHANGED")
    return entries

def prior_meta_gate(candidate, meta, schema):
    old=json.loads(meta.read_text(encoding="utf-8"))
    old_schema=json.loads(schema.read_text(encoding="utf-8"))
    split=old.get("split") or {}
    check(split.get("train_start")==PRIOR_START and
          split.get("train_end")==PRIOR_END and
          split.get("valid_start")==PRIOR_VALID_START and
          split.get("valid_end")==PRIOR_VALID_END,
          "OUTSIDER_2025_OOS_FOLD_DOES_NOT_MATCH_FROZEN_ORIGINAL:"+candidate)
    rounds=old.get("best_iteration")
    order=old_schema.get("feature_order") or []
    check(type(rounds) is int and 1<=rounds<=600 and
          len(order)>10 and len(order)==len(set(order)) and
          old.get("feature_count")==len(order),
          "OUTSIDER_2025_OOS_ROUNDS_OR_SCHEMA_UNVERIFIED:"+candidate)
    selection=old.get("feature_selection")
    check(isinstance(selection,dict) and
          selection.get("contract")=="L1_TRAIN_ONLY_FEATURE_SELECTION_V1" and
          selection.get("mode")=="none" and
          selection.get("selected_features")==order and
          old.get("prediction_phase")=="FINAL",
          "OUTSIDER_ORIGINAL_2025_MODEL_MODE_DRIFT:"+candidate)
    check(old_schema.get("prediction_phase","FINAL")=="FINAL",
          "OUTSIDER_2025_MODEL_SCHEMA_PHASE_DRIFT:"+candidate)
    return rounds,order,old,old_schema

def train(candidate, old_meta, old_schema, mode_dataset, output_dir,
          protocol_reference):
    frozen=config()
    check(candidate in frozen,"NOT_ONE_OF_FROZEN_13_OUTSIDERS")
    check(not output_dir.exists(),"OUTSIDER_MODEL_OVERWRITE_REFUSED")
    check(mode_dataset.is_file() and mode_dataset.stat().st_size>0 and
          str(protocol_reference).strip().startswith("2026_"),
          "OUTSIDER_ORIGINAL_PRIOR_ONLY_SOURCE_OR_PROTOCOL_MISSING")
    rounds,old_order,prior,prior_schema=prior_meta_gate(
        candidate,old_meta,old_schema
    )
    trainer=frozen_trainer()
    params=frozen_params(trainer)
    contract=trainer.load_feature_contract(trainer.DEFAULT_FEATURE_CONTRACT_PATH)
    sets=trainer.normalize_feature_sets("BASE",None,contract)
    check(sets==["BASE"],"ORIGINAL_OUTSIDER_BASE_FEATURE_SCOPE_CHANGED")
    blocked=trainer.load_forbidden_model_keys(
        Path("contracts/l1-feature-catalog-v1.json"))
    iterator=trainer.iter_dataset_rows(mode_dataset)
    first=next(iterator)
    check(first.get("ml_dataset_version")==3 and
          first.get("feature_schema_version")==8 and
          first.get("leakage_policy")=="STRICT_PRIOR_DATE_ONLY" and
          first.get("feature_sets")==["BASE"] and
          first.get("prediction_phase")=="FINAL" and
          str(first.get("features",{}).get("race_date","")).startswith("2023-"),
          "ORIGINAL_2023_25_OUTSIDER_SOURCE_CONTRACT_DRIFT")
    history=trainer.normalize_history_windows(
        json.dumps(first["history_windows"]),contract
    )
    small=trainer.normalize_small_sample_policy(
        json.dumps(first["small_sample_policy"]),
        json.loads(Path(trainer.DEFAULT_SMALL_SAMPLE_CONTRACT_PATH).read_text()),
        history
    )
    years=set()
    identified=set()
    def rows():
        import itertools
        for row in itertools.chain([first],iterator):
            rid=str(row.get("race_id") or "")
            hid=str(row.get("horse_id") or "")
            dt=str((row.get("features") or {}).get("race_date") or "")
            year=int(dt[:4]) if dt[:4].isdigit() else -1
            check(year in SOURCE_YEARS and rid.startswith(str(year)) and
                  row.get("leakage_policy")=="STRICT_PRIOR_DATE_ONLY" and
                  row.get("prediction_phase")=="FINAL" and
                  row.get("feature_sets")==["BASE"] and
                  (rid,hid) not in identified and bool(hid),
                  "OUTSIDER_2023_25_SOURCE_ROW_INVALID")
            identified.add((rid,hid))
            years.add(year)
            yield row
    df=trainer.flatten(rows(),sets,"FINAL",history,small,contract,blocked)
    check(years==SOURCE_YEARS and
          set(df["_race_date_dt"].dt.year.unique())==SOURCE_YEARS,
          "OUTSIDER_2025_HISTORY_SOURCE_YEARS_MISSING")
    training=df[df["_race_date_dt"].dt.year.isin(TRAIN_YEARS)].copy()
    training=trainer.complete_races(training)
    check(not training.empty and
          set(training["_race_date_dt"].dt.year.unique())==TRAIN_YEARS and
          training["_target"].nunique()==2,
          "OUTSIDER_2026_CANDIDATE_PRIOR_TRAINING_LABELS_INCOMPLETE")
    # Use 2023-25 source to keep the original 2025 model schema union;
    # only 2024-25 labels are included in the refit LightGBM Dataset.
    n_rows=len(training)
    n_races=int(training["_race_id"].nunique())
    x,_,categorical,levels=trainer.frames(
        training,training.iloc[:0],None
    )
    check(list(x.columns)==old_order,
          "OUTSIDER_ORIGINAL_2025_OOS_TO_2026_FEATURE_ORDER_DRIFT:"+candidate)
    trainer.assert_catalog_safety(list(x.columns),
                                  Path("contracts/l1-feature-catalog-v1.json"))
    trainer.assert_catalog_availability(list(x.columns),
        Path("contracts/l1-feature-catalog-v1.json"),"FINAL",contract)
    trainer.assert_prediction_phase_safety(list(x.columns),"FINAL",contract)
    y=training["_target"].astype(int)
    train_set=lgb.Dataset(x,label=y,categorical_feature=categorical,
                          free_raw_data=True)
    train_set.construct()
    del df,training,x,y
    native=dict(params)
    native.pop("n_estimators")
    native["metric"]="binary_logloss"
    model=lgb.train(native,train_set,num_boost_round=rounds)
    check(model.current_iteration()==rounds and
          model.feature_name()==old_order,
          "OUTSIDER_2026_CANDIDATE_BOOSTER_NOT_SCHEMA_COMPATIBLE")
    output_dir.mkdir(parents=True,exist_ok=False)
    model_path=output_dir/"model.txt"
    schema_path=output_dir/"schema.json"
    meta_path=output_dir/"model.json"
    model.save_model(str(model_path))
    schema={
      "contract":"L1_FEATURE_SCHEMA_V1",
      "candidate_name":candidate,
      "model_version":"OUTSIDER_2026_RESEARCH_"+candidate.upper(),
      "prediction_year":2026,
      "training_years":[2024,2025],
      "source_schema_years":[2023,2024,2025],
      "ml_dataset_version":3,"feature_schema_version":8,
      "leakage_policy":"STRICT_PRIOR_DATE_ONLY",
      "prediction_phase":"FINAL","ability_uses_odds":False,
      "feature_order":old_order,"category_levels":levels,
    }
    schema_path.write_text(json.dumps(schema,ensure_ascii=False,indent=2)+"\n")
    meta={
      "contract":CONTRACT,
      "candidate_name":candidate,
      "mode":frozen[candidate]["mode"],
      "deployment_status":"CANDIDATE_2026_BLIND_UNRELEASED",
      "training_rule_id":"2025_ORIGINAL_OOS_BEST_ITERATION_REFIT_2024_2025_CANDIDATE",
      "training_rule_approval_ref":None,
      "refit_protocol_reference":protocol_reference,
      "prediction_year":2026,"training_years":[2024,2025],
      "source_schema_years":[2023,2024,2025],
      "2026_outcomes_used":False,
      "2026_validation_used":False,
      "2026_training_evaluation_used":False,
      "prediction_phase":"FINAL","ability_uses_odds":False,
      "feature_selection":"none",
      "source_original_2025_oos_best_iteration":rounds,
      "source_original_2025_oos_meta_sha256":hash_file(old_meta),
      "source_original_2025_oos_schema_sha256":hash_file(old_schema),
      "source_original_2023_25_mode_dataset_sha256":hash_file(mode_dataset),
      "best_iteration":rounds,
      "training_rows":n_rows,"training_races":n_races,
      "feature_count":len(old_order),
      "feature_sets":["BASE"],
      "params":params,
      "model_sha256":hash_file(model_path),
      "schema_sha256":hash_file(schema_path),
      "snapshot_generation":GENERATION,
      "ready_for_betting":False,
      "asof_2026_target_independently_certified":False,
    }
    meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n")
    print("OUTSIDER_ORIGINAL_2026_UNRELEASED_REFIT_ONLY "+
          json.dumps({"candidate":candidate,"rounds":rounds,
                      "training_rows":n_rows,"training_races":n_races,
                      "source_2023_labels_in_refit":0,
                      "target_2026_results_used":False,
                      "approved":False},sort_keys=True),flush=True)
    return meta

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--candidate",required=True)
    p.add_argument("--prior-oos-meta",required=True,type=Path)
    p.add_argument("--prior-oos-schema",required=True,type=Path)
    p.add_argument("--original-mode-dataset",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--protocol-reference",required=True)
    a=p.parse_args()
    train(a.candidate,a.prior_oos_meta,a.prior_oos_schema,
          a.original_mode_dataset,a.out,a.protocol_reference)
