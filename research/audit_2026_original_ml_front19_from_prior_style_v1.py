#!/usr/bin/env python3
"""Compare genuine pinned ORIGINAL ML front score with saved BACKFILL front.

Input STYLE14 originates from unmodified original HistoricalExtraProcessor and
strictly-prior historical BACKFILL daily sources through 2026-10-16. The target
2026-10-17 has no results/payouts/odds. Directly re-execute the frozen 2025 ML
infer_front_score function as an AST source node, with no copy of its math.

Only horse_front_score is numerically compared here. Prior separate audit
verified 18 other bias fields. This does NOT certify other 649+22 columns,
source-asof evidence, model approval, or live betting readiness.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

DAY="2026-10-17"
ML_STYLE_SOURCE_GIT_BLOB="6b2124558790b317063c1d48e5c3b0ce9b6952e6"
STYLE_NAMES=frozenset((
    "style_recent_avg_first_ratio","style_recent_avg_first_position",
    "style_previous_first_ratio","style_previous_first_position"))
REFERENCE="KEIBA_BACKFILL_TO_ML_690_INDEX_V1"

def require(ok,reason):
    if not ok:raise ValueError(reason)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def gitblob(path):
    b=Path(path).read_bytes()
    return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\\x00"+b).hexdigest()

def original_front(source):
    require(gitblob(source)==ML_STYLE_SOURCE_GIT_BLOB,
            "ORIGINAL_2025_ML_STYLE_FORMULA_SOURCE_NOT_PINNED")
    tree=ast.parse(source.read_text(encoding="utf8"),filename=str(source))
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef)
           and x.name=="infer_front_score"]
    require(len(nodes)==1,"PINNED_ORIGINAL_ML_FRONT_FORMULA_NOT_FOUND")
    scope={"pd":pd,"np":np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),scope)
    return scope["infer_front_score"]

def load_style(file):
    allrows={};date_keys=set()
    with file.open(encoding="utf8") as src:
        for line in src:
            if not line.strip():continue
            row=json.loads(line)
            rid=str(row.get("race_id",""));hid=str(row.get("horse_id",""))
            key=(rid,hid)
            require(len(rid)==12 and rid.startswith("2026") and
                    key not in allrows and
                    row.get("original_style_values_complete_from_strict_prior_dates") is True and
                    row.get("blocked_incomplete_prerace_history") is False and
                    row.get("style_complete") is not None and
                    row.get("model_ready") is False and
                    row.get("ready_for_betting") is False and
                    row.get("asof_full_features_verified") is False and
                    row.get("prior_starts_690")==row.get("strictly_prior_races_observed"),
                    "INCOMPLETE_OR_UNCERTIFIED_STRICT_PRIOR_STYLE_SOURCE:"+rid+":"+hid)
            style=row["style_complete"]
            require(type(style)is dict and len(style)==14 and
                    STYLE_NAMES.issubset(style),
                    "ORIGINAL_STYLE14_INPUT_INCOMPLETE:"+rid+":"+hid)
            allrows[key]=row
            date_keys.add(rid)
    require(len(allrows)==471 and len(date_keys)==36,
            "STRICT_PRIOR_SOURCE_NOT_REAL_36R_471H")
    return allrows

def equal(left,right):
    if right is None:return left is None
    return type(left) in (int,float) and math.isfinite(left) and left==right

def run(original,style_path,delivery,out):
    require(not out.exists(),"REFUSE_OVERWRITE_PRIOR_PARITY_AUDIT")
    infer_front_score=original_front(original)
    styles=load_style(style_path)
    idx=json.loads((delivery/"ml_features/delivery/690/index.json").read_text())
    require(idx.get("contract")==REFERENCE and idx.get("total_races")==60 and
            idx.get("total_horses")==786 and
            idx.get("original_ml_full_numerical_parity_verified") is False and
            idx.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET",
            "CACHED_690_REFERENCE_SHA_INDEX_DRIFT")
    races={rid:rec for rid,rec in idx["races"].items()
           if rec["date"]==DAY}
    require(len(races)==36 and sum(v["starters"] for v in races.values())==471,
            "CACHED_690_SOURCE_DAY_DRIFT")
    verified=0;nonnull=0;rows_by_race=[]
    for rid,meta in sorted(races.items()):
        f=delivery/meta["path"]
        require(f.is_file() and sha256(f)==meta["sha256"],
                "SAVED_690_SHA_NOT_MATCHING:"+rid)
        doc=json.loads(f.read_text(encoding="utf8"))
        require(doc["contract"]=="KEIBA_BACKFILL_TO_ML_690_UNSCORED_V1" and
                doc["race_id"]==rid and doc["race_date"]==DAY and
                doc["feature_count"]==690 and doc["starters"]==meta["starters"] and
                doc.get("used_target_results_or_final_odds") is False and
                doc.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET",
                "SAVED_ORIGINAL_690_SOURCE_INVALID:"+rid)
        inputs=[];targets=[]
        for row in doc["rows"]:
            hid=str(row["_horse_id"])
            key=(rid,hid)
            require(key in styles,"ORIGINAL_STYLE_AND_BACKFILL_HORSE_ID_DRIFT:"+rid+":"+hid)
            s=styles[key]
            require(s["source_690_sha256"]==meta["sha256"] and
                    s["prior_starts_690"]==row["prior_starts"] and
                    row["_race_id"]==rid and row["_race_date"]==DAY,
                    "ORIGINAL_STYLE_PRIOR_COUNT_OR_SOURCE_SHA_MISMATCH:"+rid+":"+hid)
            require(type(row.get("field_size"))in (int,float) and
                    3<=row["field_size"]<=18,
                    "FROZEN_ML_FRONT_FIELD_SIZE_ABSENT:"+rid+":"+hid)
            inputs.append({**{k:s["style_complete"][k] for k in STYLE_NAMES},
                           "field_size":row["field_size"]})
            targets.append(row)
        # Exact 2025 ML function, no copied formula, no weight/model inference.
        predicted=infer_front_score(pd.DataFrame.from_records(inputs))
        require(len(predicted)==len(targets),"ORIGINAL_ML_FRONT_POPULATION_INVALID:"+rid)
        for row,v in zip(targets,predicted):
            expected=float(v) if pd.notna(v) else None
            observed=row.get("horse_front_score")
            require(equal(observed,expected),
                    "GENUINE_ORIGINAL_ML_FRONT_NUMERIC_MISMATCH:"
                    +rid+":"+str(row["_horse_id"])+":observed="+str(observed)
                    +":original_ml="+str(expected))
            verified+=1;nonnull+=expected is not None
        rows_by_race.append({"race_id":rid,"verified":len(targets),
                             "full_pinned_style_prior_count_verified":True,
                             "numeric_parity_front_score":True})
    require(verified==471 and len(rows_by_race)==36,
            "ORIGINAL_ML_FRONT_NUMERIC_VERIFICATION_NOT_471")
    report={
        "contract":"KEIBA_ORIGINAL_ML_2026_1017_FRONT_19TH_FROM_PINNED_PRIOR_STYLE_PARITY_V1",
        "races":36,"horses":471,"numerically_verified_original_ml_horse_front_score":471,
        "non_null_original_ml_horse_front_score":nonnull,
        "original_ml_source_sha":ML_STYLE_SOURCE_GIT_BLOB,
        "other_18_bias_fields_separately_verified_run":39105758197,
        "all_19_bias_full_universe_separately_verified":True,
        "full_649_plus_22_numeric_parity_verified":False,
        "entire_690_numeric_parity_certified":False,
        "target_day_outcomes_used":False,
        "model_or_frozen_policy_changed":False,
        "l21_tickets_generated":False,"l3_money_generated":False,
        "ready_for_betting":False,"per_race":rows_by_race}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\\n",encoding="utf8")
    print("ORIGINAL_ML_FRONT19_STRICT_PRIOR_REAL_PARITY "
          +json.dumps({"races":36,"horses":verified,"nonnull_front":nonnull,
                       "original_ml_formula":True,"full_690_verified":False,
                       "betting":False},sort_keys=True),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for name in ("original-ml-bias","source-style14","backfill-690","output"):
        ap.add_argument("--"+name,required=True,type=Path)
    z=ap.parse_args()
    run(z.original_ml_bias,z.source_style14,z.backfill_690,z.output)
