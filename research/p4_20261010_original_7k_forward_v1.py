#!/usr/bin/env python3
"""Strict-forward inference of the *saved original* seven 2025 kings on Oct 10 2026.

No new models, recalibration or algorithm modification. Models in Kaggle are
the original models trained on 2023-2024 and validated on 2025. Not a claim
that those are new 2024-2025 updated models. This isolated adapter never
consults 2026 result labels or final odds. Output is historical paper-only,
not a bet-ready as-of proof. This file does not calculate Outsider or p3.
"""
from __future__ import annotations
import argparse,gzip,hashlib,importlib.util,json,math,statistics,sys
from pathlib import Path
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

DATE="2026-10-10"
IDS={f"2026{venue}0403{n:02d}" for venue in ("05","08") for n in range(1,13)}
ALIASES=("core4","pedlegacy","condition","full","light","pedv1","condrc")
CANDIDATES=(
 "core4_no_pedigree","no_auto_full_pedigree_legacy",
 "jockey_trainer_condition","no_auto_full","no_auto_jockey",
 "no_auto_full_pedigree_v1",
 "jockey_trainer_condition_plus_raceclass")
N_HORSES=315

def need(ok,msg):
    if not ok:raise ValueError(msg)

def reader(path):
    with gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)

def dump_gz(path,rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,"wt",encoding="utf-8") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")

def scrub(source,out):
    rows=[];keys=set();counts=defaultdict(int)
    for row in reader(source):
        features=row.get("features") or {}
        if str(features.get("race_date") or "")[:10]!=DATE:continue
        rid=str(row.get("race_id") or "")
        hid=str(row.get("horse_id") or "")
        need(rid in IDS and hid.isdigit(),"UNEXPECTED_2026_TARGET_RACE_OR_HORSE")
        need((rid,hid) not in keys,"DUPLICATE_2026_7K_TARGET_KEY")
        keys.add((rid,hid));counts[rid]+=1
        need(row.get("ml_dataset_version")==3 and row.get("feature_schema_version")==8 and
             row.get("leakage_policy")=="STRICT_PRIOR_DATE_ONLY",
             "2026_FEATURE_SOURCE_CONTRACT_DRIFT")
        # Never use or copy target, results, payout or odds properties.
        # Only strictly-prior features from the source's feature projection.
        cols={k:v for k,v in features.items()
              if not any(token in k.lower() for token in
                  ("odds","popularity","payout","target","actual_finish"))}
        rows.append({k:row[k] for k in
                     ("ml_dataset_version","feature_schema_version","leakage_policy",
                      "prediction_phase","feature_sets","history_windows",
                      "small_sample_policy","race_id","horse_id")
                     if k in row}|{"features":cols,"target":{}})
    need(len(rows)==N_HORSES and set(counts)==IDS,"OCT10_2026_24_RACE_315_HORSE_SET_DRIFT")
    dump_gz(out,rows)
    print("L17_ORIGINAL_7K_SCRUBBED_TARGET",
          json.dumps({"races":len(counts),"horses":len(rows),
                      "target_labels_excluded":True,
                      "final_market_odds_excluded":True},
                     sort_keys=True),flush=True)

def load_original_emit(path):
    spec=importlib.util.spec_from_file_location("frozen_l1_emit_readonly",path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

def infer(alias,candidate,projected,root,original_emit,out):
    model_dir=Path(root)/alias
    schema=json.loads((model_dir/"schema.json").read_text())
    meta=json.loads((model_dir/"model.json").read_text())
    names=list(schema.get("feature_order") or [])
    cats=dict(schema.get("category_levels") or {})
    need(names and meta.get("prediction_phase")=="FINAL",
         "ORIGINAL_7K_SCHEMA_OR_PHASE_MISSING_"+alias)
    # The original metadata explicitly describes 2023/24 training, original
    # fitted weights and a 2025 OOS fold. Do not convert this to a fake 2026 fit.
    split=(meta.get("split") or {})
    need(split.get("train_start")=="2023-01-01" and
         split.get("train_end")=="2024-12-31" and
         split.get("valid_start")=="2025-01-01" and
         split.get("valid_end")=="2025-12-31",
         "ORIGINAL_7K_2025_MODEL_TRAIN_FOLD_DRIFT_"+alias)
    need(meta.get("feature_selection",{}).get("mode") in
         ("none",None),"ORIGINAL_7K_FEATURE_SELECTION_DRIFT_"+alias)
    raw_rows=list(reader(projected))
    need(len(raw_rows)==N_HORSES,
         "PROJECTED_ORIGINAL_7K_HORSE_COUNT_DRIFT_"+alias)
    found=set().union(*(set(r.get("features") or {}) for r in raw_rows))
    # Different starters legitimately have null/missing numeric values.
    # A column entirely absent from target dataset is structural drift.
    missing=set(names)-found-{"race_month","race_day_of_year"}
    need(not missing,
         "ORIGINAL_7K_2026_SCHEMA_MISSING_"+alias+":"+repr(sorted(missing)[:12]))
    df,ids=original_emit.load_validation_frame(
         projected,DATE,DATE,names,cats)
    need(len(ids)==N_HORSES and len(df)==N_HORSES,
         "ORIGINAL_7K_EMIT_READER_COUNT_DRIFT_"+alias)
    booster=lgb.Booster(model_file=str(model_dir/"model.txt"))
    need(booster.feature_name()==names,
         "ORIGINAL_7K_MODEL_FEATURE_ORDER_DRIFT_"+alias)
    best=int(meta.get("best_iteration") or booster.best_iteration or -1)
    probability=np.asarray(booster.predict(df,num_iteration=best),dtype=float)
    margins=np.asarray(booster.predict(df,raw_score=True,
                                       num_iteration=best),dtype=float)
    need(len(probability)==N_HORSES and
         np.isfinite(probability).all() and
         np.isfinite(margins).all() and
         (probability>0).all() and (probability<1).all(),
         "ORIGINAL_7K_PRED_NONFINITE_"+alias)
    group=defaultdict(list)
    for i,h in enumerate(ids):
        rid=str(h["race_id"]);hid=str(h["horse_id"])
        need(rid in IDS,"ORIGINAL_7K_PRED_ID_OUTSIDE_REQUEST")
        group[rid].append((i,hid,h["horse_number"]))
    result=[]
    for rid in sorted(group):
        inds=group[rid]
        total=sum(float(probability[i]) for i,_,_ in inds)
        need(total>0,"ORIGINAL_7K_EMPTY_SUM_PROBS")
        sorted_rows=sorted(inds,key=lambda x:(-float(margins[x[0]]),
                                   x[2] if x[2] is not None else 10**9,
                                   x[1]))
        for rank,(i,hid,num) in enumerate(sorted_rows,1):
            result.append({
                "contract":"L1_TO_L2_OUTPUT_CONTRACT_V1",
                "race_id":rid,"race_date":DATE,"horse_id":hid,
                "horse_number":num,
                "predicted_rank":rank,
                "raw_win_probability":float(probability[i]),
                "race_normalized_win_probability":float(probability[i]/total),
                "raw_margin_logit":float(margins[i]),
            })
    need(len(result)==N_HORSES,"ORIGINAL_7K_MODEL_RESULT_COUNT_DRIFT_"+alias)
    dump_gz(out,result)
    print("L17_ORIGINAL_KING_FORWARD_COMPLETE",
          json.dumps({"candidate":candidate,"alias":alias,"races":24,
                      "horses":315,"model_training_years":[2023,2024],
                      "original_model_saved_2025_used_without_refit":True},
                     sort_keys=True),flush=True)

def consensus(all_scores,original_materialize,out):
    # Reuse original archive's seven-king input parser (no modelling or
    # manually fabricated ballots). Its main() is sealed for research year
    # 2026, so only the pure, original parser is used in this forward adapter.
    expert={name:original_materialize.load_expert(
               name,all_scores[name]) for name in ALIASES}
    ids=sorted(next(iter(expert.values())))
    need(len(ids)==24 and all(set(t)==set(ids) for t in expert.values()),
         "SEVEN_KING_RACE_COVERAGE_DRIFT")
    result=[];total=0
    for rid in ids:
        horses={x["horse_id"] for x in expert[ALIASES[0]][rid]}
        need(all({x["horse_id"] for x in expert[e][rid]}==horses for e in ALIASES),
             "SEVEN_KING_HORSE_COVERAGE_DRIFT")
        scored=defaultdict(dict)
        horse_nums={}
        for name in ALIASES:
            for x in expert[name][rid]:
                hid=x["horse_id"]
                scored[hid][name]=x
                horse_nums.setdefault(hid,x.get("horse_number"))
        rows=[]
        for hid in sorted(horses):
            views=scored[hid]
            rank=[int(x["rank"]) for x in views.values()]
            prob=[float(x["probability"]) for x in views.values()]
            rows.append({
              "horse_id":hid,"horse_number":horse_nums[hid],
              "borda_score":float(sum(rank)),
              "normalized_borda":float(sum(rank)/(7*len(horses))),
              "mean_rank":float(sum(rank)/7),
              "rank_std":float(statistics.pstdev(rank)),
              "best_rank":min(rank),"worst_rank":max(rank),
              "top1_votes":sum(x==1 for x in rank),
              "top3_support":sum(x<=3 for x in rank),
              "top6_support":sum(x<=6 for x in rank),
              "mean_probability":float(sum(prob)/7),
              "probability_std":float(statistics.pstdev(prob)),
              "king_probability_mean":float(sum(prob)/7),
              "king_probability_std":float(statistics.pstdev(prob)),
              "experts":{e:{"rank":int(views[e]["rank"]),
                            "probability":float(views[e]["probability"])}
                          for e in ALIASES},
            })
        # Frozen original materialize_l17_seven_king_fullfield_v1 ordering.
        rows.sort(key=lambda x:(
            x["mean_rank"],-x["top6_support"],
            -x["top3_support"],-x["top1_votes"],
            -x["mean_probability"],
            x["horse_number"] if x["horse_number"] is not None else 10**9,
            x["horse_id"]))
        for rank,x in enumerate(rows,1):
            x["consensus_rank"]=rank
        need([x["consensus_rank"] for x in rows]==
             list(range(1,len(horses)+1)),
             "SEVEN_KING_RANK_COVERAGE_DRIFT")
        result.append({
           "contract":"L17_SEVEN_KING_FULLFIELD_OUTPUT_V1",
           "l17_version":"1.0.0","year":2026,
           "race_id":rid,"race_date":DATE,
           "model_ref":"ORIGINAL_SAVED_2025_7KING_TRAIN_2023_2024",
           "field_size":len(rows),"expert_count":7,
           "consensus_order":[x["horse_id"] for x in rows],
           "horses":rows})
        total+=len(rows)
    need(total==N_HORSES,"ORIGINAL_7K_CONSENSUS_TOTAL_DRIFT")
    dump_gz(out,result)
    print("L17_ORIGINAL_SEVEN_KING_CONSENSUS_READY",
         json.dumps({"races":24,"horses":315,
                     "frozen_order":"SEVEN_KING_UNCHANGED",
                     "original_2025_models_used":True,
                     "outsider_and_p3_not_yet_connected":True},
                    sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",required=True,choices=("scrub","infer","consensus"))
    p.add_argument("--source")
    p.add_argument("--out",required=True)
    p.add_argument("--alias",choices=ALIASES)
    p.add_argument("--projection")
    p.add_argument("--models")
    p.add_argument("--original-dir")
    p.add_argument("--scores-dir")
    a=p.parse_args()
    if a.mode=="scrub":
        need(a.source,"SCRUB_REQUIRES_SOURCE")
        scrub(a.source,a.out)
    elif a.mode=="infer":
        need(a.alias and a.projection and a.models and a.original_dir,
             "INFER_REQUIRES_EXACT_ORIGINAL_MODELS_AND_PROJECTION")
        emitter=load_original_emit(Path(a.original_dir)/"research/emit_l1_to_l2.py")
        infer(a.alias,CANDIDATES[ALIASES.index(a.alias)],a.projection,
              a.models,emitter,a.out)
    else:
        need(a.scores_dir and a.original_dir,
             "CONSENSUS_REQUIRES_SCORES_AND_ORIGINAL_CALCULATOR")
        old=load_original_emit(Path(a.original_dir)/"research/materialize_l17_seven_king_fullfield_v1.py")
        files={name:Path(a.scores_dir)/(name+".jsonl.gz") for name in ALIASES}
        need(all(path.is_file() for path in files.values()),
             "CONSENSUS_ORIGINAL_7K_OUTPUT_MISSING")
        consensus(files,old,a.out)

if __name__=="__main__":
    main()
