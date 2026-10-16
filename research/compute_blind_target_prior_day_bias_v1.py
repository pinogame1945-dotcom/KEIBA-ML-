#!/usr/bin/env python3
"""Extract same-day bias for a blind target using ONLY earlier completed races.

Reuses frozen BiasState, read_bias_source, race_summary, infer_front_score.
The selected race's result is never opened, and race_no>=target number is
excluded before any target fields are inspected. This reconstructs the exact
daybias/horse-front/inner/fit formulas, not timestamp provenance proof.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from run_l1_objective_rebuild_v1 import open_text
from run_l1_sameday_track_bias_v1 import (
    BiasState, STYLE_FIELDS, finite, infer_front_score,
    read_bias_source, race_summary
)

CONTRACT="KEIBA_BLIND_TARGET_ORIGINAL_PRIOR_DAYBIAS_V1"
TARGET="202505050812"


def ensure(ok,msg):
    if not ok:raise ValueError(msg)


def single_value(horses,name):
    vals={str(h.get(name) or "") for h in horses}
    ensure(len(vals)==1,"TARGET_META_DRIFT_"+name)
    return next(iter(vals))


def calculate(source,target_payload):
    ensure(target_payload.get("contract")=="KEIBA_UNSCORED_RACE_FEATURE_EXTRACT_V1",
           "BLINDED_TARGET_FEATURE_CONTRACT_REQUIRED")
    ensure(target_payload.get("race_id")==TARGET,"WRONG_TARGET_RACE")
    ensure(target_payload.get("used_target_labels") is False and
           target_payload.get("used_results_payouts_final_odds") is False,
           "POSTRACE_TARGET_INPUT_FORBIDDEN")
    horses=target_payload.get("rows") or []
    ensure(len(horses)==17,"TARGET_FIELD_MUST_HAVE_17_HORSES")
    rid=single_value(horses,"_race_id")
    date=single_value(horses,"_race_date")
    venue=single_value(horses,"venue_code")
    surface=single_value(horses,"surface")
    no=single_value(horses,"race_no")
    ensure(rid==TARGET and date=="2025-11-30","TARGET_DATE_ID_DRIFT")
    ensure(venue and surface,"TARGET_VENUE_SURFACE_NOT_IDENTIFIED")
    try:target_race_no=int(float(no))
    except (TypeError,ValueError) as ex:raise ValueError("NO_TARGET_RACE_NUMBER") from ex
    ensure(target_race_no==12,"TARGET_RACE_NO_12_REQUIRED")

    # The selection check MUST happen BEFORE accessing r['target'].
    # Prior races from the SAME date, venue and surface are allowed to
    # supply historical outcomes as the original same-day bias teacher.
    prior_lines=[]
    seen=set()
    with open_text(source) as handle:
        for line in handle:
            if not line.strip():continue
            r=json.loads(line)
            if str(r.get("race_id") or "")==TARGET:continue
            feat=r.get("features") or {}
            d=str(feat.get("race_date") or r.get("race_date") or "")[:10]
            if d!=date or str(feat.get("venue_code") or "")!=venue or str(feat.get("surface") or "")!=surface:
                continue
            race_no=finite(feat.get("race_no"))
            if not math.isfinite(race_no) or not 1<=race_no<target_race_no:
                continue
            prior_key=(str(r.get("race_id")),str(r.get("horse_id")))
            ensure(prior_key not in seen,"DUPLICATE_PRIOR_SAMEDAY_HORSE")
            seen.add(prior_key)
            # Only these strictly earlier race results may be included.
            prior_lines.append(line)

    state=BiasState()
    prior_races=0
    previous_ids=[]
    if prior_lines:
        with tempfile.TemporaryDirectory() as temp:
            prior_file=Path(temp)/"earlier-completed-races-only.jsonl"
            prior_file.write_text("".join(prior_lines),encoding="utf-8")
            prior=read_bias_source(prior_file,2025)
        prior["front_score"]=infer_front_score(prior)
        pairs=(prior[["_race_id","race_no"]]
               .drop_duplicates().sort_values(["race_no","_race_id"]))
        for _,group in pairs.groupby("race_no",sort=True,dropna=False):
            pending=[]
            for id_ in group["_race_id"]:
                completed=prior[prior["_race_id"]==id_].copy()
                ensure(not completed.empty,"EMPTY_PRIOR_RACE")
                prior_races+=1
                previous_ids.append(str(id_))
                pending.append(race_summary(completed))
            for summary in pending:state.update(summary)

    snapshot=state.snapshot()
    frame=pd.DataFrame.from_records(horses)
    ensure(frame["_horse_id"].nunique()==17,"DUPLICATE_TARGET_HORSE")
    frame["field_size"]=pd.to_numeric(frame.get("field_size"),errors="coerce")
    for c in STYLE_FIELDS:
        if c not in frame: frame[c]=np.nan
    front=infer_front_score(frame)
    hno=pd.to_numeric(frame["horse_number"],errors="coerce")
    ensure(hno.notna().all(),"TARGET_HORSE_NUMBER_MISSING")
    ensure(hno.nunique()==17,"TARGET_HORSE_NUMBER_DUPLICATE")
    inner=(1.0-(hno-1.0)/max(1.0,float(len(horses)-1))).clip(0.0,1.0)
    results=[]
    for ix,row in frame.iterrows():
        data={k:float(v) if math.isfinite(float(v)) else None for k,v in snapshot.items()}
        f=float(front.loc[ix]); n=float(inner.loc[ix])
        data["horse_front_score"]=f if math.isfinite(f) else None
        data["horse_inner_score"]=n if math.isfinite(n) else None
        for field,base,axis in (
            ("fit_front_mean","daybias_front_gap_mean",f),
            ("fit_front_ewma","daybias_front_gap_ewma",f),
            ("fit_inner_mean","daybias_inner_gap_mean",n),
            ("fit_inner_ewma","daybias_inner_gap_ewma",n)):
            prior_val=snapshot[base]
            v=axis*float(prior_val)
            data[field]=float(v) if math.isfinite(v) else None
        results.append({"horse_id":str(row["_horse_id"]),
                        "race_id":TARGET,
                        "features":data})
    return {
      "contract":CONTRACT,
      "race_id":TARGET,"race_date":date,
      "source_phase":target_payload["source_prediction_phase"],
      "status":"PRIOR_RACE_DAY_BIAS_RECONSTRUCTED_CAPTURE_TIME_NOT_PROVEN",
      "source_prior_race_count":prior_races,
      "source_prior_race_ids":previous_ids,
      "target_result_or_final_odds_used":False,
      "prior_only_race_number_order_proven":True,
      "target_race_start_timestamp_independently_verified":False,
      "source_historical_outcome_revision_times_verified":False,
      "l1_ready":False,"l2_1_ready":False,"l3_v3_ready":False,
      "horse_scores":results,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--2025-source",required=True)
    p.add_argument("--unscored-target",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    data=calculate(a.__dict__["2025_source"],
                   json.loads(Path(a.unscored_target).read_text(encoding="utf-8")))
    out=Path(a.output)
    out.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,
                              sort_keys=True)+"\n",encoding="utf-8")
    print("BLIND_SAMEDAY_DAYBIAS_FROM_PRIOR_RACES_ONLY",data["race_id"],
          "prior_races",data["source_prior_race_count"],
          "horses",len(data["horse_scores"]),
          "certified_capture_time",False,flush=True)

if __name__=="__main__":
    main()
