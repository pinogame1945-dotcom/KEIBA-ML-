#!/usr/bin/env python3
"""Frozen seven-king schema and live-2026 input preflight (read only).

The 2026 rows are UNSCORED. The archived 2025 schema.json files are used
solely as exact reference contracts. No 2025 booster is deployed, no 2026
outcome/odds/payout is read, and no model is trained or run.
"""
from __future__ import annotations
from collections import Counter
import argparse
import json
from pathlib import Path

ALIASES = ("core4", "pedlegacy", "condition", "full", "light", "pedv1", "condrc")
NAMES = (
    "core4_no_pedigree",
    "no_auto_full_pedigree_legacy",
    "jockey_trainer_condition",
    "no_auto_full",
    "no_auto_jockey",
    "no_auto_full_pedigree_v1",
    "jockey_trainer_condition_plus_raceclass",
)
DAY_COUNTS = {"2026-10-10": (24,315), "2026-10-17": (36,471)}
IDENTITY = {"_race_id", "_horse_id", "_race_date"}
FORBIDDEN = {"target", "market_outcome", "final_win_odds", "final_popularity"}
JOIN = "KEIBA_BACKFILL_690_PLUS_ORIGINAL_V4_238_UNSCORED_V1"
GEN = "eb13d4096519167b"

def require(ok, msg):
    if not ok: raise ValueError(msg)

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def inspect(joined, schemas, config):
    require(config["contract"] == "L1_RACECLASS_ABSORPTION_WALKFORWARD_V1", "FROZEN_CONFIG_CHANGED")
    require(config["snapshot_generation"] == GEN and config["training_window_years"] == 2, "FROZEN_ROLLING_CONTRACT")
    require(config["locked_years"] == [2026] and config["prediction_phase"] == "FINAL", "FROZEN_2026_LOCK")
    require(config["feature_selection"] == "none" and config["ability_uses_odds"] is False, "FROZEN_MODEL_SPEC_CHANGED")
    require(tuple(x["name"] for x in config["candidates"]) == NAMES, "SEVEN_KING_NAMES_CHANGED")

    models = {}
    for alias, king in zip(ALIASES,config["candidates"]):
        spec = read(schemas/alias/"schema.json")
        fields = spec.get("feature_order")
        require(isinstance(fields,list) and fields and len(fields) == len(set(fields)),
                "BAD_ARCHIVED_SCHEMA:"+alias)
        levels = spec.get("category_levels") or {}
        require(isinstance(levels,dict), "ARCHIVED_CATEGORY_SCHEMA_MALFORMED:"+alias)
        models[alias] = {
            "candidate":king["name"],
            "feature_order":fields,
            "required":set(fields),
            "category_levels":levels,
            "non_null":Counter(),
            "unknown_category":Counter(),
            "category_examples":{},
        }

    reports = {}
    shared_columns = None
    joined_counts = Counter()
    for day, (expected_races,expected_rows) in DAY_COUNTS.items():
        files = sorted((joined/day).glob("*.json"))
        require(len(files) == expected_races, "WRONG_JOINED_RACE_FILES:"+day)
        rows_checked=0
        for file in files:
            obj = read(file)
            rid = file.stem
            require(obj.get("contract") == JOIN and obj.get("feature_count") == 928,
                    "WRONG_JOINED_CONTRACT_OR_COLUMNS:"+rid)
            require(obj.get("race_id") == rid and obj.get("race_date") == day,
                    "WRONG_RACE_OR_DAY:"+rid)
            require(obj.get("readiness") == "ASOF_UNVERIFIED_DO_NOT_BET", "UNVERIFIED_ASOF_LABEL_LOST:"+rid)
            require(obj.get("used_target_results_or_final_odds") is False and
                    obj.get("2026_results_read_or_evaluated") is False and
                    obj.get("seven_king_models_executed") is False,
                    "FORBIDDEN_2026_SCORES_OR_OUTCOMES:"+rid)
            participants=set()
            rows = obj.get("rows") or []
            require(len(rows) == obj["starters"], "STARTER_COUNT_MISMATCH:"+rid)
            for row in rows:
                hid = str(row.get("_horse_id") or "")
                require(hid.isdigit() and hid not in participants, "HORSE_DUPLICATE:"+rid)
                participants.add(hid)
                require(row.get("_race_id") == rid and row.get("_race_date") == day,
                        "ROW_ID_OR_DAY_INCONSISTENT:"+rid)
                cols = set(row) - IDENTITY
                require(len(cols)==928 and not(cols & FORBIDDEN), "UNEXPECTED_COLUMN_OR_LABEL:"+rid)
                if shared_columns is None: shared_columns=cols
                require(shared_columns == cols, "INCONSISTENT_928_KEYS:"+rid)
                require(len([x for x in cols if x.startswith("ped_")])==238, "NOT_ORIGINAL_PEDIGREE_V4:"+rid)
                for alias, m in models.items():
                    missing = m["required"] - cols
                    require(not missing,
                        "FROZEN_EXPERT_COLUMNS_MISSING:"+alias+":"+rid+":"+repr(sorted(missing)[:10]))
                    for key in m["feature_order"]:
                        value=row.get(key)
                        if value is not None and value != "":
                            m["non_null"][key]+=1
                            if key in m["category_levels"]:
                                levels=m["category_levels"][key]
                                if str(value) not in {str(v) for v in levels}:
                                    m["unknown_category"][key]+=1
                                    m["category_examples"].setdefault(key,str(value))
                rows_checked+=1
        require(rows_checked==expected_rows, "DAY_HORSE_ROWS_WRONG:"+day)
        reports[day]={"races":len(files),"horse_rows":rows_checked}
        joined_counts["rows"]+=rows_checked
        joined_counts["races"]+=len(files)
    require(joined_counts["races"]==60 and joined_counts["rows"]==786, "ALL_TARGET_COUNTS_DRIFT")
    result = {
        "contract":"KEIBA_FROZEN_7KING_2026_928_INPUT_GATE_V1",
        "historical_rolling_years":[2024,2025],
        "blind_inference_year":2026,
        "archived_schema_reference_year":2025,
        "snapshot_generation":GEN,
        "rows":joined_counts["rows"],"races":joined_counts["races"],
        "joined_input_columns":len(shared_columns),
        "pedigree_input_columns":238,
        "days":reports,
        "seven_king_input_columns_complete":True,
        "frozen_2026_evaluation_lock_kept":True,
        "seven_king_2026_models_trained":False,
        "seven_king_2026_inference_performed":False,
        "original_690_numerical_parity_checked":False,
        "prestart_asof_independently_verified":False,
        "2026_race_results_or_odds_evaluated":False,
        "ready_for_betting":False,
        "king_schemas":[]
    }
    for alias in ALIASES:
        m=models[alias]
        required=m["required"]
        result["king_schemas"].append({
            "alias":alias,"candidate":m["candidate"],
            "required_columns":len(required),
            "present_columns":len(required),
            "missing_columns":0,
            "pedigree_columns":sum(x.startswith("ped_") for x in required),
            "all_null_columns":sorted(required-set(m["non_null"])),
            "unknown_category_counts":dict(m["unknown_category"]),
            "unknown_category_samples":m["category_examples"]
        })
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--joined",required=True,type=Path)
    p.add_argument("--schemas",required=True,type=Path)
    p.add_argument("--config",default="research/l1-raceclass-absorption-walkforward-v1.json")
    p.add_argument("--report",type=Path)
    a=p.parse_args()
    result=inspect(a.joined,a.schemas,read(a.config))
    if a.report:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for row in result["king_schemas"]:
        print("2026_FROZEN_SEVEN_KING_INPUT",json.dumps(row,ensure_ascii=False,sort_keys=True),flush=True)
    print("SEVEN_KING_2026_INPUT_GATE_PASS_NO_MODELS_OR_2026_RESULTS",json.dumps({
        "races":result["races"],"rows":result["rows"],"kings":len(result["king_schemas"]),
        "columns":result["joined_input_columns"],
        "all_schema_names_matched":result["seven_king_input_columns_complete"]
    },sort_keys=True),flush=True)

if __name__=="__main__":
    main()
