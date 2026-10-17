#!/usr/bin/env python3
"""Prepare existing 2026 BACKFILL 690 rows for frozen 13-outsider builder.

Does NOT recompute any 690 feature, fetch target labels, use 2026 results,
train models, or claim pre-start as-of certification. Produces two SEPARATE
target-day inputs so unknown 10/10 finishes never become 10/17 history.

The CORE4 649 keys come from the already saved unapproved seven-king schema,
NOT an independently invented features list. Historical metadata comes
read-only from the original 2025 snapshot first row.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path

DAYS={"2026-10-10":(24,315),"2026-10-17":(36,471)}
DAY_CONTRACT="KEIBA_BACKFILL_TO_ML_690_UNSCORED_V1"
INDEX_CONTRACT="KEIBA_BACKFILL_TO_ML_690_INDEX_V1"
UNSAFE={"target","payout","results","finish_position","final_win_odds","final_popularity"}

def check(ok,msg):
    if not ok:raise ValueError(msg)

def source(path):
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for piece in iter(lambda:f.read(1<<20),b""):h.update(piece)
    return h.hexdigest()

def prepare(root,schema_file,meta_snapshot,out):
    index=json.loads((root/"ml_features/delivery/690/index.json").read_text(encoding="utf-8"))
    schema=json.loads(schema_file.read_text(encoding="utf-8"))
    first=next(source(meta_snapshot))
    keys=list(schema.get("feature_order") or [])
    check(len(keys)==649 and len(set(keys))==649
          and not any(k in UNSAFE for k in keys)
          and index.get("contract")==INDEX_CONTRACT
          and index.get("feature_count")==690
          and index.get("total_races")==60
          and index.get("total_horses")==786
          and index.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET",
          "FROZEN_649_OR_690_INPUT_CONTRACT_INVALID")
    check(first.get("ml_dataset_version")==3
          and first.get("feature_schema_version")==8
          and first.get("leakage_policy")=="STRICT_PRIOR_DATE_ONLY"
          and str((first.get("features") or {}).get("race_date","")).startswith("2025-")
          and isinstance(first.get("history_windows"),dict)
          and first.get("history_windows")
          and isinstance(first.get("small_sample_policy"),dict)
          and first.get("small_sample_policy"),
          "2025_ORIGINAL_HISTORY_METADATA_NOT_VERIFIED")
    races=index.get("races") or {}
    check(len(races)==60,"2026_UNSCORED_RACE_SET_NOT_60")
    groups={day:[] for day in DAYS}
    all_horse_ids=set()
    for rid,item in sorted(races.items()):
        day=str(item.get("date") or "")
        check(day in DAYS and rid[:4]=="2026" and len(rid)==12
              and rid.isdigit() and item.get("path")==
                  f"ml_features/delivery/690/{day}/{rid}.json",
              "TARGET_RACE_IDENTIFICATION_INVALID:"+rid)
        path=root/item["path"]
        check(path.is_file() and sha(path)==item.get("sha256"),
              "690_SOURCE_SHA256_MISMATCH:"+rid)
        doc=json.loads(path.read_text(encoding="utf-8"))
        check(doc.get("contract")==DAY_CONTRACT and doc.get("race_id")==rid
              and doc.get("race_date")==day
              and doc.get("feature_count")==690
              and doc.get("starters")==item.get("starters")
              and doc.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET"
              and doc.get("used_target_results_or_final_odds") is False,
              "690_SOURCE_RESULT_OR_ASOF_POLICY_DRIFT:"+rid)
        rows=doc.get("rows") or []
        check(len(rows)==item["starters"],"690_SOURCE_POPULATION_DRIFT:"+rid)
        ids=set()
        for raw in rows:
            hid=str(raw.get("_horse_id") or "")
            check(hid.isdigit() and hid not in ids and
                  raw.get("_race_id")==rid and raw.get("_race_date")==day,
                  "690_SOURCE_HORSE_IDENTITY_DRIFT:"+rid)
            ids.add(hid)
            # Preserve only the 649 already ordered original model features.
            available=set(raw)-{"_race_id","_horse_id","_race_date"}
            check(len(available)==690 and set(keys).issubset(available),
                  "690_NOT_SUPERSET_OF_ORIGINAL_CORE4_649:"+rid)
            check(not any(k in UNSAFE for k in available),
                  "2026_RESULTS_OR_MARKET_OUTCOME_IN_TARGET_FEATURES:"+rid)
            features={"race_date":day}
            features.update({k:raw[k] for k in keys})
            record={
              "ml_dataset_version":3,"feature_schema_version":8,
              "leakage_policy":"STRICT_PRIOR_DATE_ONLY",
              "prediction_phase":"FINAL","feature_sets":["BASE"],
              "history_windows":first["history_windows"],
              "small_sample_policy":first["small_sample_policy"],
              "race_id":rid,"horse_id":hid,
              "features":features,"target":{},
            }
            groups[day].append(record)
            all_horse_ids.add((rid,hid))
    check(sum(map(len,groups.values()))==786 and len(all_horse_ids)==786,
          "2026_TARGET_POPULATION_NOT_786")
    check(not out.exists(),"OUTSIDER_UNSCORED_TARGET_REGENERATION_REFUSED")
    out.mkdir(parents=True)
    written={}
    for day,rows in groups.items():
        nr=len({x["race_id"] for x in rows})
        check((nr,len(rows))==DAYS[day],
              "2026_TARGET_DATE_SCOPE_COUNT_DRIFT:"+day)
        dest=out/("target-"+day+".jsonl.gz")
        with gzip.open(dest,"wt",encoding="utf-8") as sink:
            for row in sorted(rows,key=lambda z:(z["race_id"],z["horse_id"])):
                sink.write(json.dumps(row,ensure_ascii=False,
                    allow_nan=False,separators=(",",":"))+"\n")
        written[day]={"races":nr,"horses":len(rows),
                      "output_file":dest.name,"sha256":sha(dest)}
    report={
      "contract":"KEIBA_2026_13_OUTSIDER_ORIGINAL_649_UNSCORED_TARGETS_V1",
      "year":2026,"total_races":60,"total_horses":786,
      "source_690_sha256_checked":60,
      "input_original_core4_feature_count":649,
      "training_years_included":[],
      "target_2026_outcomes_used":False,
      "target_2026_odds_used":False,
      "independent_feature_asof_certified":False,
      "2026_models_trained":False,
      "history_10_10_results_imputed_for_10_17":False,
      "ready_for_betting":False,
      "targets":written,
    }
    (out/"index.json").write_text(json.dumps(report,ensure_ascii=False,
      indent=2)+"\n",encoding="utf-8")
    print("2026_ORIGINAL_13_OUTSIDER_UNSCORED_649_SOURCE_BRIDGE_READY "+
          json.dumps({"races":60,"horses":786,"days":list(DAYS),
                      "results_2026":False,"asof_certified":False},
                     sort_keys=True),flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--backfill",type=Path,required=True)
    p.add_argument("--original-core4-schema",type=Path,required=True)
    p.add_argument("--meta-snapshot-2025",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    prepare(a.backfill,a.original_core4_schema,a.meta_snapshot_2025,a.out)
