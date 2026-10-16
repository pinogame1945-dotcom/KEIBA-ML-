#!/usr/bin/env python3
"""Twenty-four October 10 2026 historical pre-result projections; original 690.

Audit only: input schema, 19 same-day bias + 22 speed-regime values;
not a certified actual pre-start feed or production/purchase inference.
Never open the selected target race's result or historical final market prices.
No 2026 model fit, evaluation, or existing model/policy modifications.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_l1_state_transition_v1 as state_module
import run_l1_residual_regime_full_pairwise_v1 as regime_module
from run_l1_sameday_track_bias_v1 import (
    BiasState, STYLE_FIELDS, bias_cols, infer_front_score, race_summary,
    read_bias_source, speed_cols)
from verify_2026_original_690_extra_values_v1 import (
    HIST_FIELDS, compact_one_year, independent_prior_events, matching,
    synthetic_target_compact, compare_daybias, verify, source_rows)
from compute_blind_target_seq_regime_v1 import horse_features
from audit_2026_generated_core4_v1 import EXPECTED, CORE4_SHA, digest

YEARS=tuple(range(2019,2027))
DAY="2026-10-10"
ID_FIELDS={"_race_id","_horse_id","_race_date"}
NUMERIC=("_finish","_is_win","_is_top3","distance_m","body_weight",
         "carried_weight","declared_field_size","margin_lengths",
         "network_expected_pairwise_score","finish_time_ms","last_3f")


def docs_at(root):
    path=Path(root)
    paths=sorted(p for p in path.glob("2026*.json") if p.is_file())
    verify(len(paths)==24,"REQUIRE_EXACTLY_24_RACE_INPUTS")
    docs=[json.loads(p.read_text(encoding="utf-8")) for p in paths]
    verify(len({d["race_id"] for d in docs})==24,"DUPLICATE_RACE_ID")
    verify(all(d["contract"]=="KEIBA_20261010_P3_UNSCORED_CORE4_649_V1" and
               d["race_date"]==DAY and
               d["used_target_labels"] is False and
               d["used_result_payout_or_final_odds"] is False and
               d["used_2026_historical_oos_predictions"] is False
               for d in docs),"FORBIDDEN_TARGET_SOURCE")
    for d in docs:
        verify(all(row["_race_id"]==d["race_id"] and row["_race_date"]==DAY
                   for row in d["rows"]),"TARGET_HORSE_RACE_MISMATCH")
        union=set().union(*(set(row)-ID_FIELDS for row in d["rows"]))
        verify(len(union)==649 and digest(sorted(union))==CORE4_SHA,
               "PREPARED_649_FEATURE_SHA_DRIFT:"+d["race_id"])
    return docs


def all_24_target(root, path):
    docs=docs_at(root)
    all_rows=[row for d in docs for row in d["rows"]]
    keys=[(row["_race_id"],row["_horse_id"]) for row in all_rows]
    verify(len(all_rows)==315 and len(set(keys))==315,
           "TARGET_24_HORSE_POPULATION_DRIFT")
    doc={"contract":"KEIBA_20261010_24_UNSCORED_COMPACT_HANDOFF_V1",
         "race_date":DAY,"race_id":"2026_24_BATCH",
         "rows":all_rows,"used_target_labels":False}
    Path(path).write_text(json.dumps(doc,ensure_ascii=False)+"\n",encoding="utf-8")
    print("P3_24_ALL_TARGET_HORSES_FOR_PRIOR_CONTEXT",len(all_rows),flush=True)


def year_context(source,year,batch,output):
    doc=json.loads(Path(batch).read_text(encoding="utf-8"))
    verify(doc["race_date"]==DAY and len(doc["rows"])==315,"WRONG_BATCH")
    compact_one_year(source,year,doc,output)


def safe(value):
    try:
        v=float(value)
    except (TypeError,ValueError):
        return None
    return v if np.isfinite(v) else None


def speed_24(inputs,docs):
    batch_rows=[row for d in docs for row in d["rows"]]
    batch={year:src.copy() for year,src in inputs.items()}
    synthetic=pd.concat(
        [synthetic_target_compact({
            "race_id":doc["race_id"],"race_date":DAY,"rows":doc["rows"]})
         for doc in docs],ignore_index=True)
    verify(synthetic["_finish"].isna().all() and
           synthetic["finish_time_ms"].isna().all() and
           synthetic["last_3f"].isna().all(),"SYNTHETIC_TARGET_OUTCOME_NOT_MASKED")
    batch[2026]=pd.concat([batch[2026],synthetic],ignore_index=True)
    for year,frame in batch.items():
        for col in NUMERIC:
            verify(col in frame,"MISSING_ORIGINAL_COMPACT_NUMERIC:"+col)
            frame[col]=pd.to_numeric(frame[col],errors="coerce").astype("float64")
        verify(frame["_race_date"].astype(str).lt(DAY).all() if year!=2026
               else frame["_race_date"].astype(str).le(DAY).all(),
               "HISTORY_CUTOFF_DRIFT")
    with patch.object(state_module,"YEARS",YEARS), \
         patch.object(state_module,"_read_state_year_item",
                      side_effect=lambda item:(item[0],batch[item[0]])), \
         patch.object(state_module.concurrent.futures,
                      "ProcessPoolExecutor",ThreadPoolExecutor), \
         patch.object(regime_module,"YEARS",YEARS):
        states,_,_,_,coverage=state_module.build_state_features({
            year:f"ephemeral-{year}" for year in YEARS})
        regimes,columns=regime_module.build_regime_features(states)
    speed=sorted(speed_cols(columns))
    verify(len(speed)==22,"SPEED_REGIME_FROZEN_22_DRIFT")
    actual=regimes[2026]
    current=actual.loc[actual["_race_date"].eq(DAY)] if "_race_date" in actual else (
        actual.loc[actual["_race_id"].isin([d["race_id"] for d in docs])])
    keys_target={(r["_race_id"],r["_horse_id"]) for r in batch_rows}
    keys_actual=set(zip(current["_race_id"].astype(str),current["_horse_id"].astype(str)))
    verify(keys_target==keys_actual,"SPEED_24_HORSE_KEY_MISMATCH")
    indexed=current.set_index(["_race_id","_horse_id"])
    prior=pd.concat([inputs[year] for year in YEARS],ignore_index=True)
    prior=independent_prior_events(prior)
    scored={}
    valid=0
    for row in batch_rows:
        key=row["_race_id"],row["_horse_id"]
        history=prior.loc[prior["_horse_id"].astype(str).eq(key[1])]
        independent=horse_features(history,DAY)
        original=indexed.loc[key]
        record={}
        for col in speed:
            matching(original[col],independent[col],f"{key}:{col}")
            record[col]=safe(original[col])
            valid+=int(pd.notna(original[col]))
        scored[key]=record
    return speed,scored,{"cells_verified":len(speed)*len(batch_rows),
                         "non_null_cells":valid,"source_priors":len(prior),
                         "target_result_used":False,"numeric_parity":True}


def bias_for_one(snapshot,doc):
    # Four original pre-race STYLE source fields are intentionally NOT
    # CORE4 inputs. Load them only for the original infer_front_score()
    # day-bias/fit calculation. No target result or market object is read.
    # Keep the frozen 649 / 690-column model matrix free of these raw fields.
    rid=doc["race_id"]
    styles={}
    for rec in source_rows(snapshot):
        if str(rec.get("race_id")) != rid:
            continue
        feats=rec.get("features") or {}
        verify(feats.get("race_date")==DAY,"STYLE_TARGET_DAY_DRIFT")
        hid=str(rec.get("horse_id"))
        verify(hid not in styles,"STYLE_DUPLICATE_TARGET_HORSE")
        styles[hid]={name:feats.get(name) for name in STYLE_FIELDS}
    verify(set(styles)=={str(r["_horse_id"]) for r in doc["rows"]},
           "STYLE_4_TARGET_HORSE_IDS_NOT_MATCHED")
    bias_doc={**doc,"rows":[
        {**row,**styles[str(row["_horse_id"])]} for row in doc["rows"]
    ]}
    # The original frozen formula test compares two reconstruction routes;
    # neither receives selected target outcome; earlier same-day race outcomes
    # are used only for subsequent race state.
    names,diagnostics=compare_daybias(snapshot,bias_doc)
    first=doc["rows"][0]
    venue,surface=str(first["venue_code"]),str(first["surface"])
    number=int(first["race_no"])
    matched=[]
    for r in source_rows(snapshot):
        f=r.get("features") or {}
        if (f.get("race_date")==DAY and str(f.get("venue_code"))==venue
                and str(f.get("surface"))==surface and
                int(f.get("race_no") or 0)<number):
            verify(r.get("race_id")!=rid,"TARGET_RESULT_LEAK")
            matched.append(r)
    if matched:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as t:
            tmp=Path(t)/"strictly-earlier-races.jsonl.gz"
            with gzip.open(tmp,"wt",encoding="utf-8") as fd:
                for r in matched:
                    fd.write(json.dumps(r,ensure_ascii=False)+"\n")
            teacher=read_bias_source(tmp,2026)
            teacher["front_score"]=infer_front_score(teacher)
    else:
        teacher=None
    b=BiasState()
    if teacher is not None:
        for race_no, group in teacher[["_race_id","race_no"]].drop_duplicates().sort_values(
            ["race_no","_race_id"]).groupby("race_no",sort=True):
            # Update in a batch per race number, like the original builder.
            pending=[race_summary(teacher.loc[teacher["_race_id"].eq(x)].copy())
                     for x in group["_race_id"].tolist()]
            for summary in pending:
                b.update(summary)
    snapshot=b.snapshot()
    df=pd.DataFrame.from_records(bias_doc["rows"])
    for col in STYLE_FIELDS:
        if col not in df.columns:
            df[col]=np.nan
    if "field_size" not in df.columns:
        df["field_size"]=len(df)
    front=infer_front_score(df)
    numbers=pd.to_numeric(df["horse_number"],errors="coerce")
    inner=(1-(numbers-1)/max(1.,float(len(df)-1))).clip(0,1)
    byhorse={}
    for i,row in df.iterrows():
        f=float(front.loc[i]);n=float(inner.loc[i])
        result={
            **snapshot,"horse_front_score":f,"horse_inner_score":n,
            "fit_front_mean":f*float(snapshot["daybias_front_gap_mean"]),
            "fit_front_ewma":f*float(snapshot["daybias_front_gap_ewma"]),
            "fit_inner_mean":n*float(snapshot["daybias_inner_gap_mean"]),
            "fit_inner_ewma":n*float(snapshot["daybias_inner_gap_ewma"]),
        }
        verify(set(result)==set(names),"BIAS_19_NAME_DRIFT")
        byhorse[(rid,str(row["_horse_id"]))]={k:safe(v) for k,v in result.items()}
    return names,byhorse,diagnostics


def run(panels,snapshot,compact_dir,outdir):
    docs=docs_at(panels)
    compact=Path(compact_dir)
    frames={y:pd.read_pickle(compact/f"prior-full-race-context-{y}.pkl") for y in YEARS}
    speedcols,speeds,speed_stats=speed_24(frames,docs)
    biascols=None
    biases={}
    count=Counter()
    reports=[]
    out=Path(outdir)
    out.mkdir(parents=True,exist_ok=True)
    for doc in docs:
        columns,vals,stats=bias_for_one(snapshot,doc)
        if biascols is None:biascols=columns
        verify(biascols==columns,"DIFFERENT_BIAS_COLUMNS_PER_RACE")
        biases.update(vals)
        rid=doc["race_id"]
        full=[]
        for row in doc["rows"]:
            k=rid,row["_horse_id"]
            verify(k in vals and k in speeds,"TARGET_NOT_PRESENT_IN_EXTRAS")
            restored={key:row[key] for key in row}
            restored.update(vals[k])
            restored.update(speeds[k])
            verify(set(restored)-ID_FIELDS==set(row)-ID_FIELDS | set(columns) | set(speedcols),
                   "FEATURE_690_SCHEMA_DRIFT")
            verify(len(set(restored)-ID_FIELDS)==690,
                   "FEATURE_690_COUNT_NOT_EXACT")
            full.append(restored)
        doc["contract"]="KEIBA_20261010_24_ORIGINAL_690_UNSCORED_INPUT_V1"
        doc["feature_count"]=690
        doc["rows"]=full
        doc["original_19_bias_values_verified"]=True
        doc["original_22_speed_regime_values_verified"]=True
        doc["frozen_2024_2025_model_training_schema_parity_verified"]=False
        doc["l1_scored"]=False
        (out/(rid+".json")).write_text(json.dumps(doc,ensure_ascii=False,allow_nan=False)+"\n",
                                       encoding="utf-8")
        count["bias_cells"]+=stats["compared_bias_cells"]
        count["bias_nonnull"]+=stats["non_null_bias_cells"]
        reports.append({"race_id":rid,"starters":len(full),
                        "daybias_cells_verified":stats["compared_bias_cells"],
                        "daybias_prior_same_day_rows":stats["prior_completed_same_day_venue_surface_records"],
                        "speed_regime_cells_verified":len(full)*22,
                        "model_690_matrix_prepared":True,
                        "model_eligibility_for_jump_race_unverified":rid=="202608040304"})
        print("P3_24_RACE_EXTRA41_READY "+json.dumps(reports[-1],sort_keys=True),flush=True)
    verify(len(reports)==24 and len(speeds)==315,"NOT_ALL_24_690_READY")
    verify(len(biascols)==19 and len(speedcols)==22,"EXTRA_41_SCHEMA_DRIFT")
    result={
        "contract":"KEIBA_20261010_24_ORIGINAL_690_NUMERICAL_INPUT_PROBE_V1",
        "status":"24_690_TARGET_INPUTS_VERIFIED_L1_NOT_YET_EXECUTED",
        "date":DAY,"races":len(reports),"starters":len(speeds),
        "per_race_results":reports,
        "core4":649,"daybias_features":len(biascols),"speed_regime_features":len(speedcols),
        "all_690_feature_names_verified_for_24":True,
        "bias_numeric_parity":{"cells":count["bias_cells"],"nonnull":count["bias_nonnull"]},
        "speed_numeric_parity":speed_stats,
        "all_target_results_and_final_odds_excluded":True,
        "2026_model_training_or_performance_evaluation":False,
        "2026_live_asof_certified":False,
        "l1_l175_l21_l3_execution_verified":False,
        "original_p3_available":False,
        "actual_prediction_stakes_or_tickets_claimed":False,
        "frozen_models_policies_modified":False,
    }
    (out/"manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",
                                     encoding="utf-8")
    print("P3_24_ORIGINAL_690_INPUT_PARITY "+json.dumps(result,ensure_ascii=False,sort_keys=True),
          flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",required=True,choices=("aggregate","compact","run"))
    p.add_argument("--panels",required=True)
    p.add_argument("--batch")
    p.add_argument("--source")
    p.add_argument("--year",type=int)
    p.add_argument("--compact-dir")
    p.add_argument("--out")
    args=p.parse_args()
    if args.mode=="aggregate":
        all_24_target(args.panels,args.out)
    elif args.mode=="compact":
        year_context(args.source,args.year,args.batch,args.out)
    else:
        run(args.panels,args.source,args.compact_dir,args.out)


if __name__=="__main__":
    main()
