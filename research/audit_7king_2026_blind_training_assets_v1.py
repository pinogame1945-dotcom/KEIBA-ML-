#!/usr/bin/env python3
"""Read-only historical asset gate for seven 2026-blind experts.

Inspects only the existing eb13 snapshot manifest, file listing, and seven
archived 2025 model/schema METADATA. Does not fetch snapshots, model.txt,
scores, 2026 outcomes, or create a 2026 training prescription.
"""
import argparse
import csv
import json
from pathlib import Path

ALIASES = ("core4","pedlegacy","condition","full","light","pedv1","condrc")
COLUMNS = (649,470,423,266,164,504,457)
CONFIG_NAMES = (
    "core4_no_pedigree","no_auto_full_pedigree_legacy",
    "jockey_trainer_condition","no_auto_full","no_auto_jockey",
    "no_auto_full_pedigree_v1","jockey_trainer_condition_plus_raceclass"
)
GEN="eb13d4096519167b"

def require(ok,reason):
    if not ok: raise ValueError(reason)

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def audit(root:Path, config:Path, manifest_path:Path, file_listing:Path):
    config=load(config)
    require(config["contract"]=="L1_RACECLASS_ABSORPTION_WALKFORWARD_V1",
            "FROZEN_CONFIG_ID_MISMATCH")
    require(config["snapshot_generation"]==GEN and
            config["training_window_years"]==2 and config["locked_years"]==[2026],
            "FROZEN_2026_ROLLING_LOCK_MISMATCH")
    require(config["prediction_phase"]=="FINAL" and config["feature_selection"]=="none"
            and config["ability_uses_odds"] is False, "FROZEN_TRAINING_MODE_DRIFT")
    require(tuple(x["name"] for x in config["candidates"])==CONFIG_NAMES,
            "FROZEN_7K_EXPERT_ID_DRIFT")
    manifest=load(manifest_path)
    require((manifest.get("generation") or {}).get("generation_id")==GEN,
            "SNAPSHOT_MANIFEST_GENERATION_DRIFT")
    years=manifest.get("years")
    require(isinstance(years,list),"SNAPSHOT_MANIFEST_YEARS_INVALID")
    ymap={str(x.get("year")):x for x in years}
    require("2024" in ymap and "2025" in ymap,
            "SNAPSHOT_2024_2025_NOT_PRESENT_IN_EXISTING_MANIFEST")
    with open(file_listing,encoding="utf-8",newline="") as f:
        headers=list(csv.DictReader(f))
    names={str(row.get("name") or "") for row in headers}
    snapshot=[]
    for y in (2024,2025):
        record=ymap[str(y)]
        file=str(record.get("file") or "")
        require(file in (f"snapshot-{y}.jsonl.gz",f"snapshot-{y}.jsonl"),
                "SNAPSHOT_FILENAME_DRIFT:"+str(y))
        require(file in names or (file.endswith(".gz") and file[:-3] in names),
                "SNAPSHOT_FILE_NOT_IN_REMOTE_LIST:"+str(y))
        require(isinstance(record.get("sha256"),str) and
                len(record["sha256"])==64, "SNAPSHOT_MANIFEST_HASH_ABSENT:"+str(y))
        snapshot.append({"year":y,"listed_in_existing_kaggle_dataset":True,
                         "filename":file,"expected_sha256":record["sha256"],
                         "content_fetched_or_hash_verified":False})
    kings=[]
    for alias,candidate,column_count in zip(ALIASES,config["candidates"],COLUMNS):
        original=root/alias
        meta=load(original/"model.json")
        schema=load(original/"schema.json")
        fields=schema.get("feature_order")
        require(isinstance(fields,list) and len(fields)==column_count and
                len(set(fields))==column_count,"ARCHIVED_SCHEMA_COLUMNS_DRIFT:"+alias)
        require(meta.get("feature_count")==column_count and
                meta.get("prediction_phase")=="FINAL" and
                meta.get("ability_uses_odds") is False,
                "ARCHIVED_2025_MODEL_METADATA_DRIFT:"+alias)
        split=meta.get("split") or {}
        require(split.get("train_start")=="2023-01-01" and
                split.get("train_end")=="2024-12-31" and
                split.get("valid_start")=="2025-01-01" and
                split.get("valid_end")=="2025-12-31",
                "ARCHIVED_2025_FOLD_DATES_NOT_AS_EXPECTED:"+alias)
        params=meta.get("params") or {}
        require(params.get("objective")=="binary" and
                params.get("n_estimators")==600 and
                params.get("random_state")==42,
                "ARCHIVED_2025_TRAINING_PARAMETERS_DRIFT:"+alias)
        best=meta.get("best_iteration")
        require(type(best) is int and 1<=best<=600,
                "ARCHIVED_BEST_ITERATION_INVALID:"+alias)
        require(schema.get("feature_order")==meta.get("features"),
                "ARCHIVED_SCHEMA_META_COLUMN_ORDER_DIFF:"+alias)
        require(schema.get("category_levels")==meta.get("category_levels"),
                "ARCHIVED_SCHEMA_META_CATEGORICAL_DRIFT:"+alias)
        kings.append({"alias":alias,"candidate":candidate["name"],
                      "frozen_feature_count":column_count,
                      "archived_training_years":[2023,2024],
                      "archived_validation_year":2025,
                      "archived_early_stopping_best_iteration":best,
                      "reusable_for_training_2026":False,
                      "archive_is_deployable_2026":False})
        print("FROZEN_7KING_ARCHIVED_2025_METADATA "+json.dumps(
            kings[-1],sort_keys=True),flush=True)
    return {
        "contract":"KEIBA_FROZEN_7KING_BLIND_2026_TRAINING_ASSET_GATE_V1",
        "snapshot_generation":GEN,
        "training_years_needed_for_2026":[2024,2025],
        "2026_outcomes_accessed":False,
        "existing_snapshot_data_files_downloaded":False,
        "existing_snapshot_files_hash_verified":False,
        "snapshot_files":snapshot,
        "seven_original_2025_model_metadata_count":7,
        "kings":kings,
        "historical_fold_best_iterations_are_NOT_2026_training_policy":True,
        "blind_2026_early_stopping_rule_resolved":False,
        "trained_2026_kings":0,
        "scored_2026_kings":0,
        "approved_for_betting":False,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--models-metadata",type=Path,required=True)
    p.add_argument("--snapshot-manifest",type=Path,required=True)
    p.add_argument("--snapshot-files-list",type=Path,required=True)
    p.add_argument("--config",type=Path,default=Path(
        "research/l1-raceclass-absorption-walkforward-v1.json"))
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    result=audit(a.models_metadata,a.config,a.snapshot_manifest,
                 a.snapshot_files_list)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",
                        encoding="utf-8")
    print("2026_BLIND_SEVEN_KING_HISTORICAL_METADATA_GATE_PASS "+json.dumps(
        {"snapshot_years":[2024,2025],"kings":len(result["kings"]),
         "snapshots_downloaded":False,"2026_outcomes_read":False,
         "model_trained":False,"stopping_protocol_resolved":False},
        sort_keys=True),flush=True)

if __name__=="__main__":
    main()
