#!/usr/bin/env python3
"""Join genuine original STYLE14 and BACKFILL13 as a compact 471-runner delta.

Inputs: an already run original v2 BACKFILL context and original historical
STYLE strictly-prior-date audit output; the immutable cached 690 index/files.
Not a new feature formula. Leaves 690 values on their original BACKFILL source.
2026 prior-date results are historical only; target-day result is never read.
"""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

DAY="2026-10-17"
BFPREFIX="ml_features/delivery/690/"
CONTEXT_CONTRACT="KEIBA_2026_10_17_ORIGINAL_OUTSIDER_RACE_CONTEXT_V1"
STYLE_FIELDS={
 "style_previous_first_position","style_previous_last_position",
 "style_previous_position_gain","style_previous_first_ratio",
 "style_previous_last_ratio","style_recent_races_measured",
 "style_recent_avg_first_position","style_recent_avg_last_position",
 "style_recent_avg_position_gain","style_recent_avg_first_ratio",
 "style_recent_avg_last_ratio","style_recent_front_rate",
 "style_recent_finish_front_rate","style_recent_improve_rate"
}
BF_FIELDS={
 "backfill_field_size","backfill_course_layout","backfill_course_laps",
 "backfill_race_class_normalized","backfill_grade","backfill_age_min",
 "backfill_age_max","backfill_sex_condition","backfill_weight_rule",
 "backfill_mixed","backfill_international","backfill_special_designated",
 "backfill_designated"
}
CONTRACT="KEIBA_2026_ORIGINAL_13_OUTSIDER_STYLE14_BACKFILL13_DELTA_V1"
def guard(condition,reason):
    if not condition:raise ValueError(reason)
def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()
def load_jsonl(file):
    with file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)
def run(root,context,style,output):
    guard(not output.exists(),"REFUSE_OVERWRITE_EXISTING_OUTSIDER_DELTA")
    index=json.loads((root/BFPREFIX/"index.json").read_text(encoding="utf-8"))
    guard(index.get("contract")=="KEIBA_BACKFILL_TO_ML_690_INDEX_V1" and
          index.get("feature_count")==690 and index.get("total_races")==60 and
          index.get("total_horses")==786 and
          index.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET",
          "PINNED_690_INDEX_DRIFT")
    races={rid:z for rid,z in index["races"].items() if z.get("date")==DAY}
    guard(len(races)==36 and sum(v["starters"] for v in races.values())==471,
          "PINNED_2026_10_17_36_471_UNIVERSE_DRIFT")
    original={}
    for row in load_jsonl(context):
        rid,hid=str(row.get("race_id") or ""),str(row.get("horse_id") or "")
        key=(rid,hid)
        values={k:v for k,v in (row.get("race_features") or {}).items()
                if k.startswith("backfill_")}
        guard(rid in races and key not in original and
              set(values)==BF_FIELDS and
              row.get("contract")==CONTEXT_CONTRACT and
              row.get("original_backfill_adapter_version")==2 and
              row.get("raw_race_context_observed_before_target_date") is True and
              row.get("derived_style_history_available") is False and
              row.get("ready_for_betting") is False,
              "UNVERIFIED_ORIGINAL_BACKFILL13_CONTEXT")
        original[key]=values
    guard(len(original)==471,"ORIGINAL_BACKFILL13_CONTEXT_MISSING_HORSES")
    styles={}
    measured=0
    for row in load_jsonl(style):
        rid,hid=str(row.get("race_id") or ""),str(row.get("horse_id") or "")
        key=(rid,hid)
        st=row.get("style_complete")
        guard(rid in races and key not in styles and
              row.get("original_style_values_complete_from_strict_prior_dates") is True and
              row.get("blocked_incomplete_prerace_history") is False and
              row.get("prior_starts_690")==row.get("strictly_prior_races_observed") and
              isinstance(st,dict) and set(st)==STYLE_FIELDS and
              row.get("model_ready") is False and
              row.get("ready_for_betting") is False and
              row.get("asof_full_features_verified") is False,
              "STRICT_PRIOR_ORIGINAL_STYLE14_NOT_COMPLETE:"+rid+":"+hid)
        styles[key]=(st,row.get("source_690_sha256"))
        if (st.get("style_recent_races_measured") or 0)>0:measured+=1
    guard(len(styles)==471 and measured==402,
          "STRICT_PRIOR_STYLE_471_OR_402_SOURCE_COVERAGE_DRIFT")
    rows=[]
    nonnull=Counter()
    for rid,item in sorted(races.items()):
        source=root/item["path"]
        guard(item["path"]==BFPREFIX+DAY+"/"+rid+".json" and
              source.is_file() and digest(source)==item["sha256"],
              "IMMUTABLE_690_RACE_SHA_MISMATCH:"+rid)
        doc=json.loads(source.read_text(encoding="utf-8"))
        guard(doc.get("race_id")==rid and doc.get("feature_count")==690 and
              doc.get("starters")==item["starters"] and
              doc.get("used_target_results_or_final_odds") is False and
              doc.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET",
              "IMMUTABLE_690_RACE_META_DRIFT:"+rid)
        for item690 in doc["rows"]:
            hid=str(item690.get("_horse_id") or "")
            key=(rid,hid)
            guard(key in original and key in styles and
                  item690.get("_race_id")==rid and item690.get("_race_date")==DAY,
                  "ORIGINAL_690_CONTEXT_AND_STYLE_HORSE_SET_MISMATCH")
            existing=set(item690)-{"_race_id","_horse_id","_race_date"}
            guard(len(existing)==690 and not(existing&(BF_FIELDS|STYLE_FIELDS)),
                  "690_ALREADY_CONTAINS_ADDITIONAL_STYLE_OR_BACKFILL_COLUMNS")
            st,source_sha=styles[key]
            guard(source_sha==item["sha256"],
                  "STYLE_PROVENANCE_690_RACE_HASH_MISMATCH")
            delta={**original[key],**st}
            guard(len(delta)==27 and set(delta)==(BF_FIELDS|STYLE_FIELDS),
                  "OUTSIDER_DELTA_13_14_FEATURE_COUNT_DRIFT")
            nonnull.update(k for k,v in delta.items() if v is not None)
            rows.append({
              "contract":CONTRACT,"race_id":rid,"horse_id":hid,
              "race_date":DAY,"cached_690_race_sha256":item["sha256"],
              "original_features_added":27,
              "style_history_source":"STRICT_PRIOR_DATE_ONLY_UP_TO_2026_10_16",
              "prior_2026_outcomes_used_only_for_history":True,
              "target_2026_results_used":False,
              "year_2026_outcome_evaluation_performed":False,
              "style_source_complete":True,
              "full_928_asof_proven":False,
              "ready_for_betting":False,
              "features_delta":delta
            })
    guard(len(rows)==471 and len({(x["race_id"],x["horse_id"]) for x in rows})==471,
          "OUTSIDER_DELTA_UNIVERSE_NOT_EXACTLY_471")
    output.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(output,"wt",encoding="utf-8") as f:
        for row in sorted(rows,key=lambda z:(z["race_id"],z["horse_id"])):
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")
    metrics={"races":36,"horses":471,"original_690_kept_unchanged":True,
             "extra_features":27,"combined_hypothetical_features":717,
             "style_runners_with_measured_corners":measured,
             "delta_non_null_cell_count":sum(nonnull.values()),
             "target_2026_results_used":False,
             "prior_2026_results_historical_only":True,
             "asof_all_717_verified":False,"ready_for_betting":False,
             "delta_bytes":output.stat().st_size,
             "delta_sha256":digest(output)}
    print("ORIGINAL_OUTSIDER_STYLE14_BACKFILL13_DELTA_36_471_PASS "+
          json.dumps(metrics,sort_keys=True),flush=True)
    return metrics
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--cached-690-root",type=Path,required=True)
    p.add_argument("--original-backfill-context",type=Path,required=True)
    p.add_argument("--original-style",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.cached_690_root,a.original_backfill_context,a.original_style,a.output)
