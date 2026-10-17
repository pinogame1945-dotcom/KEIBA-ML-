#!/usr/bin/env python3
"""Blind 2026 inference entry for the seven ORIGINAL frozen king architectures.

Uses unmodified emit_l1_to_l2 utilities (feature conversion, TreeSHAP family
mapping, sigmoid and SHA) but never the research 2026 validation/early-stopping
path. This will *refuse* historical fold models; it is not a 2026 trainer.
Only a future, explicitly approved, train-2024+2025 model release is accepted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd

from emit_l1_to_l2 import (
    FEATURE_CONTRACT, L2_CONTRACT, family_for_feature, load_validation_frame,
    sha256_json, sigmoid,
)

CANDIDATES = {
    "core4": ("core4_no_pedigree", 649),
    "pedlegacy": ("no_auto_full_pedigree_legacy", 470),
    "condition": ("jockey_trainer_condition", 423),
    "full": ("no_auto_full", 266),
    "light": ("no_auto_jockey", 164),
    "pedv1": ("no_auto_full_pedigree_v1", 504),
    "condrc": ("jockey_trainer_condition_plus_raceclass", 457),
}
DAY_EXPECTATIONS = {"2026-10-10": (24, 315), "2026-10-17": (36, 471)}
INPUT_CONTRACT = "KEIBA_BACKFILL_690_PLUS_ORIGINAL_V4_238_UNSCORED_V1"
RELEASE_CONTRACT = "KEIBA_7KING_2026_BLIND_MODEL_RELEASE_V1"
RELEASE_STATUS = "APPROVED_2026_BLIND"
DENIED = set(L2_CONTRACT["forbidden_payload_fields"]) | {
    "_target","_finish","_is_win","_is_top3","_score","target",
    "finish","result","result_rank","postrace_finish","after_race_outcome",
    "final_place_odds","market_rank","market_prob_norm","payout_yen",
}
IDENTITY = {"_race_id", "_horse_id", "_race_date"}


def require(ok: bool, msg: str):
    if not ok:
        raise ValueError(msg)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda:src.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def model_gate(king: str, folder: Path):
    name, expected_count = CANDIDATES[king]
    model_file = folder/"model.txt"
    meta_file = folder/"model.json"
    schema_file = folder/"schema.json"
    for f in (model_file, meta_file, schema_file):
        require(f.is_file() and f.stat().st_size > 0,
                "APPROVED_2026_RELEASE_FILE_MISSING:"+king+":"+f.name)
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    require(meta.get("contract")==RELEASE_CONTRACT and
            meta.get("deployment_status")==RELEASE_STATUS,
            "NOT_AN_APPROVED_2026_BLIND_MODEL:"+king)
    require(meta.get("candidate_name")==name and
            meta.get("prediction_year")==2026 and
            meta.get("training_years")==[2024,2025] and
            meta.get("train_start")=="2024-01-01" and
            meta.get("train_end")=="2025-12-31" and
            meta.get("snapshot_generation")=="eb13d4096519167b",
            "2026_BLIND_ORIGINAL_TWO_YEAR_MODEL_PROVENANCE_MISMATCH:"+king)
    require(meta.get("2026_outcomes_used") is False and
            meta.get("ability_uses_odds") is False and
            meta.get("feature_selection")=="none" and
            meta.get("prediction_phase")=="FINAL",
            "2026_RESULTS_OR_MODEL_SPEC_VIOLATION:"+king)
    require(isinstance(meta.get("training_rule_id"),str) and
            bool(meta["training_rule_id"].strip()) and
            isinstance(meta.get("training_rule_approval_ref"),str) and
            bool(meta["training_rule_approval_ref"].strip()),
            "2026_MODEL_TRAINING_RULE_NOT_APPROVED:"+king)
    require(schema.get("contract")=="L1_FEATURE_SCHEMA_V1",
            "FROZEN_L1_FEATURE_SCHEMA_CHANGED:"+king)
    order=schema.get("feature_order")
    categories=schema.get("category_levels") or {}
    require(isinstance(order,list) and len(order)==expected_count and
            len(set(order))==expected_count and isinstance(categories,dict),
            "SEVEN_KING_ORIGINAL_SCHEMA_DRIFT:"+king)
    require(schema.get("prediction_phase")=="FINAL" and
            schema.get("ability_uses_odds") is False and
            int(schema.get("ml_dataset_version",0))==3 and
            int(schema.get("feature_schema_version",0))==8 and
            schema.get("leakage_policy")=="STRICT_PRIOR_DATE_ONLY",
            "ORIGINAL_SCHEMA_LEAKAGE_CONTRACT_DRIFT:"+king)
    require(schema.get("training_years")==[2024,2025] and
            schema.get("model_version")==meta.get("model_version"),
            "ORIGINAL_MODEL_SCHEMA_RELEASE_MISMATCH:"+king)
    require(meta.get("model_sha256")==digest(model_file),
            "APPROVED_MODEL_SHA256_MISMATCH:"+king)
    require(meta.get("schema_sha256")==digest(schema_file),
            "APPROVED_SCHEMA_SHA256_MISMATCH:"+king)
    iteration=meta.get("best_iteration")
    require(type(iteration) is int and 1<=iteration<=600,
            "INVALID_APPROVED_ITERATION:"+king)
    require(bool(meta.get("model_version")),
            "APPROVED_MODEL_VERSION_MISSING:"+king)
    return model_file, meta, schema, iteration


def original_unscored_dataset(joined: Path, output: Path):
    race_ids=set()
    horses=set()
    count=0
    with output.open("w",encoding="utf-8") as fh:
        for day,(n_races,n_horses) in DAY_EXPECTATIONS.items():
            files=sorted((joined/day).glob("*.json"))
            require(len(files)==n_races,"JOINED_ORIGINAL_TARGET_RACES_MISSING:"+day)
            day_horses=0
            for file in files:
                record=json.loads(file.read_text(encoding="utf-8"))
                rid=file.stem
                require(record.get("contract")==INPUT_CONTRACT and
                        record.get("race_id")==rid and
                        record.get("race_date")==day and
                        record.get("feature_count")==928 and
                        record.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET" and
                        record.get("used_target_results_or_final_odds") is False and
                        record.get("2026_results_read_or_evaluated") is False,
                        "UNVERIFIED_INPUT_MUST_STAY_NOT_FOR_BETTING:"+rid)
                require(rid not in race_ids,"DUPLICATED_RACE:"+rid)
                race_ids.add(rid)
                rows=record.get("rows")
                require(isinstance(rows,list) and len(rows)==record.get("starters"),
                        "WRONG_ORIGINAL_STARTERS:"+rid)
                for row in rows:
                    hid=str(row.get("_horse_id") or "")
                    require(row.get("_race_id")==rid and
                            row.get("_race_date")==day and
                            hid.isdigit() and (rid,hid) not in horses,
                            "INVALID_TARGET_ROW_ID:"+rid+":"+hid)
                    horses.add((rid,hid))
                    features={key:value for key,value in row.items()
                              if key not in IDENTITY}
                    require(len(features)==928 and
                            not set(features).intersection(DENIED) and
                            not any(k.startswith(("result_","target_","payout_","postrace_","after_race_","final_"))
                                    for k in features),
                            "TARGET_RESULT_FINAL_ODDS_OR_SCHEMA_LEAK:"+rid)
                    fh.write(json.dumps({
                        "race_id":rid,"horse_id":hid,
                        "features":{"race_date":day,**features}
                    },ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")
                    count+=1
                    day_horses+=1
            require(day_horses==n_horses,"ORIGINAL_DAY_HORSE_ROWS_MISSING:"+day)
    require(len(race_ids)==60 and count==786,"TARGET_60_RACES_786_ROWS_DRIFT")
    return horses


def score_king(king: str, files, target_file: Path, expected_ids, output:Path):
    model_file,meta,schema,iterations = files
    booster = lgb.Booster(model_file=str(model_file))
    feature_order=schema["feature_order"]
    require(booster.feature_name()==feature_order,
            "ORIGINAL_LIGHTGBM_FEATURE_ORDER_DRIFT:"+king)
    require(0<iterations<=booster.current_iteration(),
            "APPROVED_ITERATION_NOT_PRESENT_IN_MODEL:"+king)
    frame,ids=load_validation_frame(target_file,
                                   "2026-10-10","2026-10-17",
                                   feature_order,schema.get("category_levels") or {})
    require(len(ids)==786 and
            {(x["race_id"],x["horse_id"]) for x in ids}==expected_ids,
            "MODEL_INPUT_LOST_OR_DUPLICATED_RUNNER:"+king)
    raw=booster.predict(frame,num_iteration=iterations)
    logits=booster.predict(frame,num_iteration=iterations,raw_score=True)
    contrib=np.asarray(booster.predict(frame,num_iteration=iterations,
                                      pred_contrib=True),dtype=float)
    require(len(raw)==786 and
            contrib.shape==(786,len(feature_order)+1) and
            np.isfinite(raw).all() and np.isfinite(logits).all() and
            np.isfinite(contrib).all(),
            "LIGHTGBM_OUTPUT_NOT_FINITE_OR_SHAP_SIZE_DRIFT:"+king)
    for i in range(786):
        require(abs(sigmoid(float(logits[i]))-float(raw[i]))<=1e-6 and
                abs(float(contrib[i].sum())-float(logits[i]))<=1e-5,
                "ORIGINAL_SHAP_RECONSTRUCTION_FAILURE:"+king)
    groups=defaultdict(list)
    for i,r in enumerate(ids):
        groups[r["race_id"]].append(i)
    require(len(groups)==60,"PREDICTED_RACE_COUNT_DRIFT:"+king)
    ranks={}
    norms={}
    summaries={}
    for rid,idxs in groups.items():
        ps=np.clip(raw[idxs],1e-15,1.0)
        pn=ps/ps.sum()
        order=np.argsort(-raw[idxs],kind="stable")
        field_ranks=np.empty(len(idxs),dtype=int)
        for rank,pos in enumerate(order,start=1):
            field_ranks[pos]=rank
        for j,i in enumerate(idxs):
            ranks[i]=int(field_ranks[j])
            norms[i]=float(pn[j])
        desc=np.sort(pn)[::-1]
        summaries[rid]={"field_size":len(idxs),"top1_probability":float(desc[0]),
                        "top2_probability":float(desc[1]) if len(desc)>1 else None,
                        "top1_top2_gap":float(desc[0]-desc[1]) if len(desc)>1 else None,
                        "top3_probability_mass":float(np.sum(desc[:3])),
                        "normalized_entropy":float(-np.sum(np.clip(pn,1e-15,1)*np.log(np.clip(pn,1e-15,1)))/np.log(len(idxs)))
                              if len(idxs)>1 else 0.0}
    families=list(FEATURE_CONTRACT["feature_sets"])
    family_of=[family_for_feature(x) for x in feature_order]
    cfg={"candidate_name":meta["candidate_name"],
         "prediction_phase":"FINAL",
         "feature_sets":meta["feature_sets"],
         "actor_prefixes":meta.get("actor_prefixes") or [],
         "auto_slices":meta.get("auto_slices") or [],
         "pedigree_slices":meta.get("pedigree_slices") or [],
         "feature_selection_mode":"none"}
    expert_id=sha256_json(cfg)[:16]
    forbidden=set(L2_CONTRACT["forbidden_payload_fields"])
    with gzip.open(output,"wt",encoding="utf-8",compresslevel=6) as dst:
        for i,ident in enumerate(ids):
            family_signed={x:0.0 for x in families}
            family_abs={x:0.0 for x in families}
            for j,family in enumerate(family_of):
                value=float(contrib[i,j])
                family_signed[family]+=value
                family_abs[family]+=abs(value)
            denom=sum(family_abs.values())
            row={
                "contract":"L1_TO_L2_OUTPUT_CONTRACT_V1",
                **ident,
                "model_version":meta["model_version"],"expert_id":expert_id,
                "prediction_phase":"FINAL","feature_sets":meta["feature_sets"],
                "ml_dataset_version":3,
                "feature_schema_version":8,
                "leakage_policy":"STRICT_PRIOR_DATE_ONLY",
                "raw_win_probability":float(raw[i]),
                "race_normalized_win_probability":norms[i],
                "predicted_rank":ranks[i],
                "raw_margin_logit":float(logits[i]),
                "bias_logit":float(contrib[i,-1]),
                "family_contribution_logit":family_signed,
                "family_abs_contribution":family_abs,
                "family_abs_share":{x:(family_abs[x]/denom if denom else 0.0) for x in families},
                "race_summary":summaries[ident["race_id"]],
                "model_sha256":meta["model_sha256"],
                "training_config_sha256":meta.get("training_config_sha256"),
                "feature_catalog_sha256":meta.get("feature_catalog_sha256"),
                "feature_contract_sha256":meta.get("feature_contract_sha256"),
                "source_backfill_sha":meta.get("source_backfill_sha"),
                "ml_source_sha":meta.get("ml_source_sha")
            }
            require(not forbidden.intersection(row),
                    "L1_TO_L2_ORIGINAL_FORBIDDEN_PREDICTION_PAYLOAD:"+king)
            dst.write(json.dumps(row,ensure_ascii=False,separators=(",",":"),
                                 allow_nan=False)+"\n")
    return {"alias":king,"expert_id":expert_id,
            "model_sha256":meta["model_sha256"],
            "model_version":meta["model_version"],
            "horse_rows":len(ids),"races":len(groups),
            "best_iteration":iterations}


def run(joined:Path,models:Path,output:Path):
    # Fail closed: check ALL seven released 2026 models BEFORE opening target data.
    preflight={alias:model_gate(alias,models/alias) for alias in CANDIDATES}
    require(len(set(v[1]["training_rule_id"] for v in preflight.values()))==1,
            "SEVEN_KING_TRAINING_RULE_RELEASE_INCONSISTENT")
    require(len(set(v[1]["training_rule_approval_ref"] for v in preflight.values()))==1,
            "SEVEN_KING_APPROVAL_REF_INCONSISTENT")
    require(not output.exists(),"REFUSE_OVERWRITE_EXISTING_SCORE_DIRECTORY")
    with tempfile.TemporaryDirectory(prefix="keiba-blind-7king-") as temp:
        source=Path(temp)/"target.jsonl"
        expected_ids=original_unscored_dataset(joined,source)
        staged=Path(temp)/"scores"
        staged.mkdir()
        scored=[]
        for alias,release in preflight.items():
            dst=staged/(alias+".jsonl.gz")
            record=score_king(alias,release,source,expected_ids,dst)
            scored.append(record)
            print("2026_FROZEN_KING_INFERENCE "+json.dumps(record,sort_keys=True),flush=True)
        summary={
            "contract":"KEIBA_2026_SEVEN_KING_BLIND_INFERENCE_V1",
            "model_release_id":preflight["core4"][1]["training_rule_approval_ref"],
            "models_trained_on_years":[2024,2025],
            "prediction_year":2026,
            "races":60,"horse_rows":786,"king_count":7,
            "2026_target_outcomes_used":False,
            "2026_model_validation_performed":False,
            "source_input_asof_independently_verified":False,
            "ready_for_betting":False,
            "frozen_l2_l3_invoked":False,
            "scores":scored,
        }
        (staged/"index.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
                                         encoding="utf-8")
        output.mkdir(parents=True)
        for item in staged.iterdir():
            item.replace(output/item.name)
    print("SEVEN_REAL_2026_BLIND_KING_SCORES_WRITTEN_NOT_FOR_BETTING",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--joined",type=Path,required=True)
    p.add_argument("--approved-models",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.joined,a.approved_models,a.output)


if __name__=="__main__":
    main()
