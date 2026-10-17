#!/usr/bin/env python3
"""Read-only, single-pass, original 13 Outsider 2023-2025 source/feature audit.

Uses unchanged process_day() from original build_l1_nonhorse_dataset_v1.py.
No model training, prediction, payout evaluation or 2026 outcomes; original
2025 OOS window remains train2023-24/validation2025, not refit/promotion.
Original 13 modes see identical rows and their own per-horse history state.
All outputs are aggregate counts/schemas (no runner or training model files).
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from build_l1_nonhorse_dataset_v1 import process_day, MODES

EXPECTED_GENERATION="eb13d4096519167b"
EXPECTED_SOURCES=(2023,2024,2025)
CONFIGS=("research/l1-outsider-nonhorse-v1.json",
         "research/l1-outsider-transition-v1.json")
FORBIDDEN={"odds","popularity","payout"}
HISTORY_CONTRACT="STRICT_PRIOR_DATE_ONLY"
def demand(ok,reason):
    if not ok:raise ValueError(reason)
def open_file(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else path.open("r",encoding="utf-8")
def candidates():
    found={}
    for file in CONFIGS:
        data=json.loads(Path(file).read_text(encoding="utf-8"))
        demand(data.get("snapshot_generation")==EXPECTED_GENERATION and
               data.get("training_window_years")==2 and
               data.get("locked_years")==[2026] and
               data.get("ability_uses_odds") is False and
               data.get("prediction_phase")=="FINAL" and
               data.get("feature_selection")=="none",
               "ORIGINAL_13_OUTSIDER_CONFIG_DRIFT")
        for one in data["candidates"]:
            name,mode=one["name"],one["mode"]
            demand(name not in found and mode in MODES and
                   one["feature_sets"]==["BASE"],
                   "ORIGINAL_13_OUTSIDER_CANDIDATE_INVALID")
            found[name]=mode
    demand(len(found)==13,"ORIGINAL_13_OUTSIDER_CANDIDATES_NOT_13")
    return found

class AuditWriter:
    def __init__(self):
        self.year_counts=Counter()
        self.races_by_year=defaultdict(set)
        self.feature_keys_by_year=defaultdict(set)
        self.feature_sets_by_year=defaultdict(set)
        self.nonnull_by_year=Counter()
        self.target_labels_by_year=Counter()
        self.target_winner_races=defaultdict(set)
        self.style_sample_count=Counter()
    def write(self,line):
        row=json.loads(line)
        rid=str(row.get("race_id") or "")
        date=str(row.get("features",{}).get("race_date") or "")
        year=int(date[:4])
        demand(year in EXPECTED_SOURCES and rid.startswith(str(year)) and
               row.get("leakage_policy")==HISTORY_CONTRACT and
               row.get("ml_dataset_version")==3 and
               row.get("feature_schema_version")==8,
               "ORIGINAL_2025_FOLD_FEATURE_OUTPUT_DRIFT")
        feature=row.get("features") or {}
        demand(bool(feature) and all(not any(t in k.lower() for t in FORBIDDEN) for k in feature),
               "ORIGINAL_OUTSIDER_ODDS_MARKET_LEAKAGE")
        self.year_counts[year]+=1
        self.races_by_year[year].add(rid)
        self.feature_keys_by_year[year].update(feature)
        self.feature_sets_by_year[year].add(tuple(sorted(feature)))
        self.nonnull_by_year[year]+=sum(v is not None for v in feature.values())
        target=row.get("target") or {}
        if target.get("is_win") is not None:
            self.target_labels_by_year[year]+=1
            if target.get("is_win"):self.target_winner_races[year].add(rid)
        if feature.get("nh_shape_front_heavy_share") is not None:
            self.style_sample_count[year]+=1

def validate_source_row(row,expected_year):
    features=row.get("features") or {}
    date=str(features.get("race_date") or "")
    demand(date[:4]==str(expected_year) and
           row.get("ml_dataset_version")==3 and
           row.get("feature_schema_version")==8 and
           row.get("leakage_policy")==HISTORY_CONTRACT and
           row.get("prediction_phase","FINAL")=="FINAL" and
           str(row.get("race_id","")).startswith(str(expected_year)),
           "2023_2025_HISTORICAL_SOURCE_OR_YEAR_DRIFT")
    demand(row.get("target") is not None and
           not any(t in features for t in ("target","payout_yen","final_win_odds")),
           "HISTORIC_ONLY_2025_SOURCE_CONTRACT_DRIFT")
    return date

def run(paths,output):
    demand(not output.exists(),"NO_OVERWRITE_2025_ORIGINAL_SOURCE_AUDIT")
    demand(len(paths)==3 and
           all(p.is_file() and
               p.name in {f"snapshot-{y}.jsonl",f"snapshot-{y}.jsonl.gz"}
               for y,p in zip(EXPECTED_SOURCES,paths)),
           "EXACT_ORIGINAL_2023_24_25_SNAPSHOTS_REQUIRED")
    modes=candidates()
    writers={name:AuditWriter() for name in modes}
    states={name:defaultdict(list) for name in modes}
    earliest=None
    previous=None
    per_year=Counter()
    races=defaultdict(set)
    current_day=[]
    meta=None
    def emit():
        if not current_day:return
        for name,mode in modes.items():
            count=process_day(current_day,mode,writers[name],meta,states[name])
            demand(count==len(current_day),
                   "ORIGINAL_13_OUTSIDER_SOURCE_ROW_LOSS:"+name)
    for yr,p in zip(EXPECTED_SOURCES,paths):
        with open_file(p) as source:
            for line in source:
                if not line.strip():continue
                row=json.loads(line)
                date=validate_source_row(row,yr)
                if meta is None:
                    meta={"history_windows":row.get("history_windows") or {},
                          "small_sample_policy":row.get("small_sample_policy") or {}}
                    demand(meta["history_windows"] and meta["small_sample_policy"],
                           "HISTORICAL_META_MISSING")
                else:
                    demand(row.get("history_windows")==meta["history_windows"] and
                           row.get("small_sample_policy")==meta["small_sample_policy"],
                           "HISTORICAL_WINDOW_POLICY_DRIFT")
                demand(previous is None or date>=previous,
                       "HISTORIC_FOLD_DATE_NOT_SORTED")
                if previous and date!=previous:
                    emit()
                    current_day.clear()
                previous=date
                earliest=date if earliest is None else earliest
                current_day.append(row)
                per_year[yr]+=1
                races[yr].add(str(row["race_id"]))
    emit()
    demand(set(per_year)==set(EXPECTED_SOURCES) and
           all(per_year[y]>10000 for y in EXPECTED_SOURCES),
           "2023_2025_ORIGINAL_SOURCE_COUNTS_TOO_SMALL")
    details={}
    for name,w in writers.items():
        demand(dict(w.year_counts)==dict(per_year) and
               all(w.races_by_year[y]==races[y] for y in EXPECTED_SOURCES),
               "ORIGINAL_13_OUTSIDER_FEATURE_POPULATION_MISMATCH:"+name)
        demand(w.target_labels_by_year[2025]>0 and
               len(w.target_winner_races[2025])>100,
               "FROZEN_2025_ORIGINAL_VALIDATION_LABELS_MISSING:"+name)
        details[name]={
            "mode":modes[name],
            "training_2023_2024_rows":per_year[2023]+per_year[2024],
            "validation_2025_rows":per_year[2025],
            "valid_2025_winner_races":len(w.target_winner_races[2025]),
            "train_2023_24_feature_union":len(w.feature_keys_by_year[2023]|w.feature_keys_by_year[2024]),
            "validation_2025_feature_union":len(w.feature_keys_by_year[2025]),
            "new_keys_only_2025":sorted(w.feature_keys_by_year[2025]-(w.feature_keys_by_year[2023]|w.feature_keys_by_year[2024])),
            "2025_distinct_feature_sets":len(w.feature_sets_by_year[2025]),
            "2025_non_null_cells":w.nonnull_by_year[2025],
            "2025_shape_front_heavy_nonnull":w.style_sample_count[2025],
        }
    result={
        "contract":"ORIGINAL_13_OUTSIDER_2025_OOS_SOURCE_COMPATIBILITY_V1",
        "snapshot_generation":EXPECTED_GENERATION,
        "source_years":[2023,2024,2025],
        "original_2025_oos_train_years":[2023,2024],
        "original_2025_oos_validation_year":[2025],
        "source_rows_by_year":dict(per_year),
        "source_races_by_year":{str(y):len(races[y]) for y in EXPECTED_SOURCES},
        "original_model_specs_unchanged":True,
        "source_snapshots_fetched_once":True,
        "features_calculated_via_original_code":True,
        "2026_result_rows_read":0,
        "model_training_invoked":False,
        "model_scoring_invoked":False,
        "candidate_2026_models_approved":False,
        "ready_for_betting":False,
        "candidates":details
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ORIGINAL_13_OUTSIDER_2025_OOS_SOURCE_AUDIT_PASS "+
          json.dumps({"candidates":len(details),"original_2025_valid_races":len(races[2025]),
                      "source_rows_by_year":dict(per_year),
                      "model_training":False,
                      "2026_outcomes_read":False},sort_keys=True),flush=True)
    for name,d in details.items():
        print("OUTSIDER_PRIOR_OOS_SOURCE_MODE "+
              json.dumps({"name":name,"feature_union_2025":d["validation_2025_feature_union"],
                          "new_keys_only_2025":len(d["new_keys_only_2025"]),
                          "ready_for_betting":False},sort_keys=True),flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot-2023",required=True,type=Path)
    p.add_argument("--snapshot-2024",required=True,type=Path)
    p.add_argument("--snapshot-2025",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    run([args.snapshot_2023,args.snapshot_2024,args.snapshot_2025],args.out)
