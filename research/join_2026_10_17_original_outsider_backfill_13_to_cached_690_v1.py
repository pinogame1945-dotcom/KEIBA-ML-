#!/usr/bin/env python3
"""Read-only 2026-10-17 690 + original BACKFILL v2 13 race-context merger.

Exact source SHA per original 36 cached race files; 471 horse-id matches.
NO 2026 outcomes, no wager-ready flag, no STYLE value fabrication. Retains
existing 690 values bit-for-bit, appending ONLY pinned-backfill-derived 13 keys.
One-time interim audit output is runner-local, never a model input release.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

PREFIX="ml_features/delivery/690"
DAY="2026-10-17"
ADDED={
 "backfill_field_size","backfill_course_layout","backfill_course_laps",
 "backfill_race_class_normalized","backfill_grade","backfill_age_min",
 "backfill_age_max","backfill_sex_condition","backfill_weight_rule",
 "backfill_mixed","backfill_international","backfill_special_designated",
 "backfill_designated"
}

def require(ok,msg):
    if not ok:raise ValueError(msg)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def join(root,context,output):
    require(not output.exists(),"NO_OVERWRITE_2026_OUTSIDER_SOURCE_MERGE")
    index=json.loads((root/PREFIX/"index.json").read_text(encoding="utf-8"))
    require(index.get("feature_count")==690 and
            index.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET" and
            index.get("total_races")==60 and
            index.get("total_horses")==786 and
            index.get("contract")=="KEIBA_BACKFILL_TO_ML_690_INDEX_V1",
            "PINNED_690_INPUT_INDEX_DRIFT")
    entries={rid:d for rid,d in index["races"].items() if d["date"]==DAY}
    require(len(entries)==36 and sum(e["starters"] for e in entries.values())==471,
            "PINNED_690_2026_10_17_SCOPE_DRIFT")
    context_data={}
    with context.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            d=json.loads(line)
            rid,hid=str(d.get("race_id") or ""),str(d.get("horse_id") or "")
            ident=(rid,hid)
            features=d.get("race_features") or {}
            require(ident not in context_data and rid in entries and hid.isdigit()
                and set(k for k in features if k.startswith("backfill_"))==ADDED
                and d.get("contract")=="KEIBA_2026_10_17_ORIGINAL_OUTSIDER_RACE_CONTEXT_V1"
                and d.get("raw_race_context_observed_before_target_date") is True
                and d.get("derived_style_history_available") is False
                and d.get("ready_for_betting") is False,
                "ORIGINAL_BACKFILL_CONTEXT_NOT_AUTHENTIC_OR_DUPLICATED")
            context_data[ident]=d
    require(len(context_data)==471,"OUTSIDER_BACKFILL_CONTEXT_NOT_471")
    matches=set()
    combined=[]
    nulls={k:0 for k in sorted(ADDED)}
    for rid,item in sorted(entries.items()):
        cached=root/item["path"]
        require(cached.is_file() and sha(cached)==item["sha256"],
                "PINNED_690_RACE_SHA_DRIFT:"+rid)
        doc=json.loads(cached.read_text(encoding="utf-8"))
        require(doc.get("race_id")==rid and doc.get("race_date")==DAY and
                doc.get("feature_count")==690 and
                doc.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET" and
                doc.get("used_target_results_or_final_odds") is False and
                doc.get("starters")==item["starters"],"PINNED_690_RACE_META_DRIFT:"+rid)
        rows=doc.get("rows") or []
        require(len(rows)==item["starters"],"PINNED_690_STARTERS_DRIFT:"+rid)
        for row in rows:
            hid=str(row.get("_horse_id") or "")
            ident=(rid,hid)
            require(ident in context_data and ident not in matches and
                    row.get("_race_id")==rid and row.get("_race_date")==DAY,
                    "690_VERSUS_BACKFILL_SOURCE_HORSE_MISMATCH:"+rid+":"+hid)
            base={k:v for k,v in row.items() if k not in ("_race_id","_horse_id","_race_date")}
            require(len(base)==690 and not (set(base)&ADDED),
                    "690_BASE_WIDTH_OR_BACKFILL_OVERLAP_DRIFT")
            saved=context_data[ident]
            other=saved["race_features"]
            for k in ("venue_code","meeting_no","meeting_day","race_no",
                      "discipline","surface","distance_m"):
                require(base.get(k)==other.get(k),
                        "PRE_RACE_JRA_METADATA_DIFFERS_FROM_690:"+rid+":"+k)
            enriched={**base,**{k:other[k] for k in sorted(ADDED)}}
            require(len(enriched)==703 and
                    all(enriched[k] is base[k] or enriched[k]==base[k] for k in base),
                    "690_EXISTING_NUMERIC_FEATURES_CHANGED")
            for k in ADDED:
                if enriched[k] is None:nulls[k]+=1
            combined.append({
                "contract":"KEIBA_2026_ORIGINAL_OUTSIDER_690_PLUS_13_BACKFILL_PARTIAL_V1",
                "race_id":rid,"horse_id":hid,"race_date":DAY,
                "base_690_source_sha256":item["sha256"],
                "live_race_context_captured_before_date":True,
                "full_2026_feature_asof_proven":False,
                "outsider_style_history_recovered":False,
                "outsider_all_13_ready":False,
                "ready_for_betting":False,
                "features":enriched,
            })
            matches.add(ident)
    require(len(matches)==471 and matches==set(context_data),
            "BACKFILL_AND_690_HORSE_UNIVERSE_NOT_EQUAL")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",encoding="utf-8") as f:
        for d in combined:
            f.write(json.dumps(d,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")
    result={
      "races":36,"horses":len(combined),"source_690_columns":690,
      "recovered_original_backfill_columns":len(ADDED),
      "interim_combined_feature_columns":703,
      "null_cells_by_feature":{k:v for k,v in nulls.items() if v},
      "2026_results_used":False,"style_historical_source_restored":False,
      "model_input_ready":False,"asof_independently_verified":False,
      "out":str(output),
    }
    print("ORIGINAL_BACKFILL_13_WITH_SAVED_690_36_471_JOIN_PASS "+
          json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--backfill-root",type=Path,required=True)
    p.add_argument("--original-context",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    x=p.parse_args()
    join(x.backfill_root,x.original_context,x.out)
