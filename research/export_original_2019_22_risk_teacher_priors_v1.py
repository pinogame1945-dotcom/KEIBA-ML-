#!/usr/bin/env python3
"""Original 2019–2022 CORE4 + bias for reproducing only past 2021–24 OOS.

No target-year 2025 result, odds, or saved prediction is accessed.
"""
import argparse
import hashlib
import json
from pathlib import Path

from run_l1_objective_rebuild_v1 import META,read_year,select_variant
from run_l1_ceiling_audit_v1 import strip_forbidden
from run_l1_sameday_track_bias_v1 import _build_same_day_bias_year,bias_cols

SHA="eb3392b725560e77d46b28aef036b6188814125ed8f7a3bfde0c53522dbadf58"

def export(path,year,out):
    if year not in (2019,2020,2021,2022):
        raise ValueError("ONLY_PAST_2019_TO_2022_FOR_RISK_TEACHER")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    base=select_variant(strip_forbidden(read_year(path,year)),"core4")
    names=sorted(set(base.columns)-META)
    digest=hashlib.sha256(json.dumps(names,separators=(",",":")).encode()).hexdigest()
    if len(names)!=649 or digest!=SHA:
        raise ValueError("FROZEN_ORIGINAL_EARLIER_2019_22_CORE4_SCHEMA_DRIFT")
    y,bias,coverage=_build_same_day_bias_year(year,path)
    if y!=year or bias.duplicated(["_race_id","_horse_id"]).any():
        raise ValueError("FROZEN_ORIGINAL_EARLIER_DAYBIAS_KEY_DRIFT")
    baseline=set(map(tuple,base[["_race_id","_horse_id"]].to_numpy()))
    covered=set(map(tuple,bias[["_race_id","_horse_id"]].to_numpy()))
    if not baseline.issubset(covered):
        raise ValueError("ORIGINAL_DAYBIAS_SOURCE_HORSE_MISSING")
    (out/f"original-{year}-core4-base.csv.gz").parent.mkdir(parents=True,exist_ok=True)
    base.to_csv(out/f"original-{year}-core4-base.csv.gz",index=False,compression="gzip")
    bias.to_csv(out/f"original-{year}-daybias.csv.gz",index=False,compression="gzip")
    print("FROZEN_ORIGINAL_EARLIER_OOS_TRAIN_SOURCE",json.dumps({
        "year":year,"source_races":int(base["_race_id"].nunique()),
        "horses":len(base),"base_features":len(names),
        "daybias_features":len(bias_cols(bias,with_fit=True)),
        "source_snapshot_past_year":True,"2025_target_result_used":False
    },sort_keys=True),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--year",type=int,required=True)
    ap.add_argument("--source",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    export(a.source,a.year,a.out)

if __name__=="__main__":main()
