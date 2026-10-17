#!/usr/bin/env python3
"""Build all 13 ORIGINAL 2025-fold mode datasets in ONE 2023-25 source pass.

Original build_l1_nonhorse_dataset_v1.process_day() is UNCHANGED. This
replaces only duplicate source IO in the old one-candidate-at-a-time runner.
2026 data/results are categorically rejected; OOS 2025 labels are allowed
only for the original 2025 validation and to train an unapproved 2026 candidate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from audit_original_13_outsider_2025_oos_source_v1 import (
    EXPECTED_SOURCES, candidates, open_file, validate_source_row,
)
from build_l1_nonhorse_dataset_v1 import process_day

EXPECTED_COUNTS={2023:47493,2024:47009,2025:47731}
CONTRACT="ORIGINAL_13_OUTSIDER_2023_25_SINGLE_PASS_FROZEN_FEATURE_DATASETS_V1"

class CheckedWriter:
    def __init__(self, path, name):
        self.writer=gzip.open(path,"wt",encoding="utf-8")
        self.name=name
        self.rows=Counter()
        self.races=defaultdict(set)
        self.features=set()
        self.valid_winners=set()
    def write(self,line):
        row=json.loads(line)
        d=str((row.get("features") or {}).get("race_date") or "")
        year=int(d[:4]) if d[:4].isdigit() else -1
        rid=str(row.get("race_id") or "")
        if year not in EXPECTED_SOURCES or not rid.startswith(str(year)):
            raise ValueError("ORIGINAL_13_BAD_HISTORICAL_DATASET_OUTPUT:"+self.name)
        if (row.get("feature_sets")!=["BASE"] or
            row.get("prediction_phase")!="FINAL" or
            row.get("leakage_policy")!="STRICT_PRIOR_DATE_ONLY" or
            row.get("ml_dataset_version")!=3 or
            row.get("feature_schema_version")!=8):
            raise ValueError("ORIGINAL_13_OUTSIDER_SOURCE_CONTRACT_DRIFT:"+self.name)
        f=row.get("features") or {}
        if not f or any("odds" in str(k).lower() or "payout" in str(k).lower()
                        or "popularity" in str(k).lower() for k in f):
            raise ValueError("ORIGINAL_13_OUTSIDER_MARKET_LEAK:"+self.name)
        self.rows[year]+=1
        self.races[year].add(rid)
        self.features.update(f)
        if year==2025 and (row.get("target") or {}).get("is_win") is True:
            self.valid_winners.add(rid)
        self.writer.write(line)
    def close(self):self.writer.close()

def run(paths,out):
    if out.exists(): raise ValueError("REFUSE_OVERWRITE_EXISTING_OUTSIDER_MODE_DATASETS")
    if len(paths)!=3 or any(
        not f.is_file() or f.name not in
          {f"snapshot-{yr}.jsonl",f"snapshot-{yr}.jsonl.gz"}
        for yr,f in zip(EXPECTED_SOURCES,paths)
    ):raise ValueError("ONLY_PINNED_2023_2024_2025_SNAPSHOTS_ALLOWED")
    modes=candidates()
    if len(modes)!=13: raise ValueError("FROZEN_13_MODE_COUNT_CHANGED")
    out.mkdir(parents=True)
    writers={name:CheckedWriter(out/(name+".jsonl.gz"),name) for name in modes}
    states={name:defaultdict(list) for name in modes}
    counts=Counter()
    rac=defaultdict(set)
    day=[]
    prev=None
    meta=None
    def emit():
        if not day:return
        for name,mode in modes.items():
            n=process_day(day,mode,writers[name],meta,states[name])
            if n!=len(day):raise ValueError("ORIGINAL_13_OUTSIDER_MODE_ROW_LOSS:"+name)
    try:
        for year,file in zip(EXPECTED_SOURCES,paths):
            with open_file(file) as fh:
                for line in fh:
                    if not line.strip():continue
                    record=json.loads(line)
                    date=validate_source_row(record,year)
                    if prev is not None and date<prev:
                        raise ValueError("ORIGINAL_OOS_SOURCE_NOT_DATE_SORTED")
                    data_meta={"history_windows":record.get("history_windows") or {},
                               "small_sample_policy":record.get("small_sample_policy") or {}}
                    if meta is None:
                        if not all(data_meta.values()):
                            raise ValueError("ORIGINAL_OOS_SOURCE_META_MISSING")
                        meta=data_meta
                    elif data_meta!=meta:
                        raise ValueError("ORIGINAL_OOS_SOURCE_HISTORY_POLICY_DRIFT")
                    if prev is not None and date!=prev:
                        emit()
                        day.clear()
                    day.append(record)
                    prev=date
                    counts[year]+=1
                    rac[year].add(str(record["race_id"]))
        emit()
    finally:
        for writer in writers.values():writer.close()
    if dict(counts)!=EXPECTED_COUNTS:
        raise ValueError("ORIGINAL_SNAPSHOT_POPULATION_DRIFT:"+str(dict(counts)))
    summary={}
    for name,writer in writers.items():
        if dict(writer.rows)!=EXPECTED_COUNTS or any(
            writer.races[y]!=rac[y] for y in EXPECTED_SOURCES
        ) or len(writer.valid_winners)<100:
            raise ValueError("ORIGINAL_OOS_13_MODE_POPULATION_OR_2025_LABELS_MISSING:"+name)
        p=out/(name+".jsonl.gz")
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        summary[name]={
          "mode":modes[name],"rows":sum(writer.rows.values()),
          "2023_2024_train_rows":writer.rows[2023]+writer.rows[2024],
          "2025_validation_rows":writer.rows[2025],
          "2025_races":len(writer.races[2025]),
          "2025_winner_races":len(writer.valid_winners),
          "feature_keys_union":len(writer.features),
          "filename":p.name,"sha256":h,"bytes":p.stat().st_size,
        }
        print("OUTSIDER_ORIGINAL_2025_SOURCE_SINGLE_PASS_ONE_MODE "+
              json.dumps({"name":name,"rows":summary[name]["rows"],
                          "feature_keys":summary[name]["feature_keys_union"]},
                         sort_keys=True),flush=True)
    result={
        "contract":CONTRACT,"source_snapshot_generation":"eb13d4096519167b",
        "source_years":[2023,2024,2025],
        "original_2025_oos_training_years":[2023,2024],
        "original_2025_oos_validation_year":2025,
        "rows_by_year":dict(counts),
        "races_by_year":{str(y):len(rac[y]) for y in EXPECTED_SOURCES},
        "original_unmodified_builder":True,
        "2026_target_or_prior_result_rows_read":0,
        "2026_outcome_evaluation":False,
        "models_trained":0,
        "betting_release_approved":False,
        "candidate_modes":summary
    }
    (out/"index.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",
                                  encoding="utf-8")
    print("ORIGINAL_13_OUTSIDER_2025_SOURCE_SINGLE_PASS_COMPLETE "+
          json.dumps({"candidates":len(summary),
                      "source_rows":sum(counts.values()),
                      "2025_races":len(rac[2025]),
                      "no_2026_labels":True},sort_keys=True),flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source-2023",type=Path,required=True)
    p.add_argument("--source-2024",type=Path,required=True)
    p.add_argument("--source-2025",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run([a.source_2023,a.source_2024,a.source_2025],a.out)
