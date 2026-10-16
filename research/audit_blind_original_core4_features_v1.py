#!/usr/bin/env python3
"""Gate 202505050812 real EARLY features against ORIGINAL L1 CORE4 columns.

This does not train or score. No stored historical target OOS labels/scores.
It verifies source features have no forbidden columns and produces a schema
fingerprint to compare against 2023/24 training frames. No chance of silent
fallback to an invented feature schema.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import pandas as pd

from run_l1_objective_rebuild_v1 import (
    META, FORBIDDEN_KEYS, BASE_CATEGORICAL, VARIANTS, keep_variant_column)
from run_l1_ceiling_audit_v1 import FULL_FORBIDDEN
from run_original_l1_cold_inference_v1 import BANNED_TARGET, BANNED_PREFIXES


def require(ok, code):
    if not ok:
        raise ValueError(code)


def audit(payload):
    require(payload.get("contract") == "KEIBA_UNSCORED_RACE_FEATURE_EXTRACT_V1",
            "INVALID_UNSCORED_SNAPSHOT_CONTRACT")
    require(payload.get("race_id") == "202505050812",
            "WRONG_USER_BLIND_RACE")
    for field in ("used_target_labels", "used_results_payouts_final_odds",
                  "used_historical_oos_target_predictions"):
        require(payload.get(field) is False, "RESEARCH_LOOKAHEAD_"+field)
    require(payload.get("source_manifest_count_and_date_range_verified") is True,
            "SOURCE_MANIFEST_WAS_NOT_VERIFIED")
    require(payload.get("actual_source_capture_time_independently_verified") is False,
            "SOURCE_CAPTURE_CANNOT_BE_SELF_CERTIFIED")
    rows=payload.get("rows")
    require(isinstance(rows,list) and len(rows)==17,"USER_RACE_17_HORSES_EXPECTED")
    df=pd.DataFrame.from_records(rows)
    require(not df.duplicated(["_race_id","_horse_id"]).any(),
            "CORE4_HORSE_IDENTITY_DRIFT")
    require(set(df["_race_id"])=={"202505050812"},"OTHER_RACE_ENTERED_SNAPSHOT")
    forbidden=sorted(c for c in df if c in FULL_FORBIDDEN or
                     c in BANNED_TARGET or
                     c.startswith(BANNED_PREFIXES) or c.startswith(("P__","distill_","outcome_")))
    require(not forbidden, "ORIGINAL_CORE4_TARGET_LEAK:" + ",".join(forbidden))
    original=sorted(c for c in df if c not in META and c not in FORBIDDEN_KEYS)
    selected=sorted(c for c in original if keep_variant_column(c,"core4"))
    dropped=sorted(set(original)-set(selected))
    require(VARIANTS["core4"]["allowed_prefixes"]==
            {"opponent_","auto_","actor_","timepace_"},
            "ORIGINAL_CORE4_FAMILY_DEFINITION_CHANGED")
    require(set(("race_month","race_day_of_year")).issubset(selected),
            "ORIGINAL_SEASONAL_FEATURES_MISSING")
    require(not any(c.startswith(("network_","lap_","style_","distx_","backfill_","ped_")) for c in selected),
            "FORBIDDEN_NON_CORE4_FAMILY_IN_FEATURES")
    require(not any(c.startswith("_") for c in selected),
            "HIDDEN_MODEL_FEATURE_IN_CORE4")
    not_numeric = [c for c in selected if c not in BASE_CATEGORICAL and not
                   pd.api.types.is_numeric_dtype(df[c])]
    # Research prepare() calls pd.to_numeric(errors='coerce') for these.
    # Report but do not create a fake numerical representation.
    prefix_counts=Counter()
    for c in selected:
        p=next((x for x in
                 ("opponent_","auto_","actor_","timepace_") if c.startswith(x)),"BASE")
        prefix_counts[p]+=1
    stamp=hashlib.sha256(json.dumps(selected,sort_keys=True,
                        separators=(",",":")).encode()).hexdigest()
    answer={
        "contract":"KEIBA_BLIND_202505050812_ORIGINAL_CORE4_SCHEMA_AUDIT_V1",
        "race_id":"202505050812",
        "status":"ORIGINAL_CORE4_TARGET_FEATURE_NAMES_AUDITED_NOT_YET_TRAINED",
        "horses":len(rows),
        "input_feature_count":len(original),
        "core4_feature_count":len(selected),
        "excluded_non_core4_count":len(dropped),
        "core4_group_counts":dict(sorted(prefix_counts.items())),
        "categorical_count":len(set(selected)&BASE_CATEGORICAL),
        "noncategorical_non_numeric_names":not_numeric[:30],
        "core4_feature_name_sha256":stamp,
        "feature_columns":selected,
        "target_prediction_available":False,
        "year_2025_saved_AI_targets_used":False,
        "target_result_or_final_odds_used":False,
        "full_field_verified_with_official_entries":False,
        "source_capture_asof_verified":False,
        "training_2023_2024_feature_schema_verified":False,
        "trained_model_equivalence_verified":False
    }
    return answer


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    payload=json.loads(Path(args.input).read_text(encoding="utf-8"))
    out=audit(payload)
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    summary={k:v for k,v in out.items() if k!="feature_columns"}
    print("ORIGINAL_CORE4_BLIND_SCHEMA_AUDIT",json.dumps(summary,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
