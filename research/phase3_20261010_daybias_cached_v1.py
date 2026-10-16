#!/usr/bin/env python3
"""Performance-only Phase3 adapter: cache same-day source and bias for 24 races.

Proof: batch original _build_same_day_bias_year() once + an independent
strictly-earlier-race reconstruction of its 19 pre-race values for each
selected horse. Original model features/formula/policy unchanged.
All 2026 target finish results are excluded from model training, output
and evaluation. Earlier completed races can inform only later race bias.
Historical capture times are NOT authenticated by this retrospective audit.
No persistent data or upload.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import phase3_20261010_24_extra41_batch_v1 as base
from verify_2026_original_690_extra_values_v1 import matching,source_rows
from run_l1_sameday_track_bias_v1 import (
    BiasState,STYLE_FIELDS,_build_same_day_bias_year,bias_cols,
    infer_front_score,race_summary,read_bias_source
)


def require(ok,reason):
    if not ok:raise ValueError(reason)


def numeric(value):
    try:
        v=float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def precompute(source,docs):
    ids={d["race_id"] for d in docs}
    starters={(d["race_id"],r["_horse_id"]) for d in docs for r in d["rows"]}
    expected_n=sum(len(d["rows"]) for d in docs)
    require(len(ids)==24 and len(starters)==expected_n==315,
            "FAST_DAYBIAS_ALL_24_STARTER_IDENTITIES_DRIFT")
    original_styles={}
    seen=set()
    # Scan the year's snapshot once, instead of 24 times.
    # Only rows from the requested race day are materialized; 2026
    # target scores, payout objects and final prices are never opened.
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"2026-10-10-original-prior-only-projection.jsonl.gz"
        with gzip.open(path,"wt",encoding="utf-8") as fd:
            for obj in source_rows(source):
                feat=obj.get("features") or {}
                if feat.get("race_date")!=base.DAY:
                    continue
                rid=str(obj.get("race_id") or "")
                hid=str(obj.get("horse_id") or "")
                require(rid in ids,"FAST_DAYBIAS_UNEXPECTED_SAME_DAY_RACE_ID")
                key=(rid,hid)
                require(key in starters and key not in seen,
                        "FAST_DAYBIAS_SOURCE_HORSE_NOT_UNIQUE")
                seen.add(key)
                original_styles[key]={name:feat.get(name) for name in STYLE_FIELDS}
                # Historical finish labels are read *only by the original
                # state updater* for subsequently completed race numbers.
                # Snapshots for the selected race are calculated BEFORE its
                # own race outcome can update that state.
                fd.write(json.dumps(obj,ensure_ascii=False)+"\n")
        require(seen==starters,"FAST_DAYBIAS_SOURCE_ENTRY_SET_DRIFT")
        _, original, _ = _build_same_day_bias_year(2026,path)
        observed=read_bias_source(path,2026)
        observed["front_score"]=infer_front_score(observed)
    require(len(original)==expected_n and len(observed)==expected_n,
            "FAST_DAYBIAS_SOURCE_DROPPED_HORSE")
    require(not original.duplicated(["_race_id","_horse_id"]).any(),
            "FAST_DAYBIAS_ORIGINAL_DUPLICATE")
    require(original[["_race_id","_horse_id"]].astype(str).apply(tuple,axis=1).isin(starters).all(),
            "FAST_DAYBIAS_ORIGINAL_HORSE_NOT_IN_ALL_24")
    original=original.set_index(["_race_id","_horse_id"])
    fields=sorted(bias_cols(original,with_fit=True))
    require(len(fields)==19,"FAST_DAYBIAS_ORIGINAL_19_NAMES_DRIFT")
    print("P3_FAST_BIAS_ONCE_SOURCE_READY "+
          json.dumps({"races":24,"horses":expected_n,"fields":len(fields),
                      "scan_2026_snapshot_once":True,
                      "prior_by_venue_surface_and_race_no":True},
                     sort_keys=True),flush=True)
    return fields,original,observed,original_styles


def one_race(doc,fields,original,observed,styles):
    rid=doc["race_id"]
    rows=doc["rows"]
    for r in rows:
        require((rid,r["_horse_id"]) in styles,
                "FAST_BIAS_MISSING_TARGET_FRONT_STYLE")
    frame=pd.DataFrame.from_records([
        {**r, **styles[(rid,r["_horse_id"])]} for r in rows
    ])
    venue=str(frame.iloc[0]["venue_code"])
    surface=str(frame.iloc[0]["surface"])
    number=int(frame.iloc[0]["race_no"])
    require(all(frame["venue_code"].astype(str)==venue) and
            all(frame["surface"].astype(str)==surface) and
            pd.to_numeric(frame["race_no"],errors="coerce").eq(number).all(),
            "FAST_BIAS_TARGET_RACE_META_DRIFT")
    # Strictly earlier in SAME date/venue/surface, by race number.
    earlier=observed.loc[
        observed["_race_date"].astype(str).eq(base.DAY) &
        observed["venue_code"].astype(str).eq(venue) &
        observed["surface"].astype(str).eq(surface) &
        pd.to_numeric(observed["race_no"],errors="coerce").lt(number)
    ]
    require(not earlier["_race_id"].astype(str).eq(rid).any(),
            "FAST_BIAS_TARGET_OUTCOME_IN_OWN_CONTEXT")
    state=BiasState()
    order=earlier[["_race_id","race_no"]].drop_duplicates().sort_values(
        ["race_no","_race_id"])
    for _, group in order.groupby("race_no",sort=True):
        # Exact original race_summary / BiasState math; no same-race-no
        # result may enter a peer's state until this group completes.
        pending=[race_summary(earlier.loc[earlier["_race_id"].eq(x)].copy())
                 for x in group["_race_id"]]
        for item in pending:
            state.update(item)
    snap=state.snapshot()
    front=infer_front_score(frame)
    num=pd.to_numeric(frame["horse_number"],errors="coerce")
    inner=(1-(num-1)/max(1.0,float(len(frame)-1))).clip(0,1)
    vals={}
    nonnull=0
    for i,row in frame.iterrows():
        hid=str(row["_horse_id"])
        f=float(front.loc[i]); n=float(inner.loc[i])
        expected={
            **snap,
            "horse_front_score":f,
            "horse_inner_score":n,
            "fit_front_mean":f*float(snap["daybias_front_gap_mean"]),
            "fit_front_ewma":f*float(snap["daybias_front_gap_ewma"]),
            "fit_inner_mean":n*float(snap["daybias_inner_gap_mean"]),
            "fit_inner_ewma":n*float(snap["daybias_inner_gap_ewma"])
        }
        require(set(expected)==set(fields),
                "FAST_BIAS_FROZEN_19_COLUMN_SET_DRIFT")
        key=(rid,hid)
        require(key in original.index,
                "FAST_BIAS_ORIGINAL_BATCH_HORSE_MISSING")
        actual=original.loc[key]
        for col in fields:
            # Independent direct strict-prior projection must match original
            # same-day batch formula. Retain original float32 results.
            matching(actual[col],expected[col],f"cached/{rid}/{hid}/{col}")
            nonnull+=int(pd.notna(actual[col]))
        vals[key]={col:numeric(actual[col]) for col in fields}
    result={
        "original_19_bias_and_fit_cols":19,
        "original_batch_to_masked_strict_prior_reconstruction_values_equal":True,
        "compared_bias_cells":len(rows)*19,
        "non_null_bias_cells":nonnull,
        "prior_completed_same_day_venue_surface_records":len(earlier),
        "target_result_used_for_own_bias":False,
        "source_timing_authentication":False
    }
    return fields,vals,result


def run(args):
    docs=base.docs_at(args.panels)
    fields,original,prior,styles=precompute(args.source,docs)
    by_id={x["race_id"]:x for x in docs}
    calls=0
    def cached_bias_for_one(snapshot,doc):
        nonlocal calls
        require(doc["race_id"] in by_id and
                snapshot==args.source,"FAST_BIAS_SOURCE_DOES_NOT_MATCH")
        calls+=1
        return one_race(doc,fields,original,prior,styles)
    # Interpose an audited input adaptor ONLY; original LambdaRank,
    # L1 features and L2.1/L3 frozen source code are never changed.
    original_adapter=base.bias_for_one
    try:
        base.bias_for_one=cached_bias_for_one
        base.run(args.panels,args.source,args.compact_dir,args.out)
    finally:
        base.bias_for_one=original_adapter
    require(calls==24,"FAST_BIAS_ALL_24_NOT_CALLED")
    result=json.loads((Path(args.out)/"manifest.json").read_text(encoding="utf-8"))
    require(result["races"]==24 and result["starters"]==315 and
            result["bias_numeric_parity"]["cells"]==5985 and
            result["bias_numeric_parity"]["nonnull"]==4711 and
            result["speed_numeric_parity"]["cells_verified"]==6930,
            "FAST_BIAS_DOES_NOT_MATCH_PREVIOUS_VERIFIED_COUNTS")
    print("P3_FAST_DAYBIAS_SAME_NUMERIC_PARITY "+
          json.dumps({"all_24_races":True,"starters":315,
                      "bias_cells":5985,"bias_nonnull":4711,
                      "speed_cells":6930,"2026_source_scans_for_bias":1,
                      "original_frozen_formula_changed":False},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--panels",required=True)
    p.add_argument("--source",required=True)
    p.add_argument("--compact-dir",required=True)
    p.add_argument("--out",required=True)
    run(p.parse_args())
