#!/usr/bin/env python3
"""One-pass authentic frozen source CORE4 and original same-day bias from prior years.

Only 2023 & 2024 training years. Must run while respective original snapshot
is available, prior to ephemeral raw deletion. Results from 2025 target
never read. The exact original research functions are called unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from run_l1_objective_rebuild_v1 import read_year,select_variant,META
from run_l1_ceiling_audit_v1 import strip_forbidden,FULL_FORBIDDEN
from run_l1_sameday_track_bias_v1 import _build_same_day_bias_year,bias_cols

def export(source,year,dest):
    if year not in (2023,2024):raise ValueError("ONLY_PRIOR_2023_2024_TRAIN")
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    base=select_variant(strip_forbidden(read_year(source,year)),"core4")
    features=sorted(c for c in base.columns if c not in META)
    if len(features)!=649 or set(features)&FULL_FORBIDDEN:
        raise ValueError("ORIGINAL_CORE4_649_SCHEMA_DRIFT")
    digest=hashlib.sha256(json.dumps(features,separators=(",",":")).encode()).hexdigest()
    if digest!="eb3392b725560e77d46b28aef036b6188814125ed8f7a3bfde0c53522dbadf58":
        raise ValueError("ORIGINAL_CORE4_FEATURE_SHA_DRIFT")
    y,bias,coverage=_build_same_day_bias_year(year,source)
    if y!=year or len(bias)!=len(base):raise ValueError("BIAS_ORIGINAL_ROW_COUNT_MISMATCH")
    if set(map(tuple,bias[["_race_id","_horse_id"]].values))!=set(map(tuple,base[["_race_id","_horse_id"]].values)):
        raise ValueError("BIAS_ORIGINAL_HORSE_POPULATION_DRIFT")
    if not len(bias_cols(bias,with_fit=True)):
        raise ValueError("ORIGINAL_DAYBIAS_MISSING")
    base_path=dest/f"original-{year}-core4-base.csv.gz"
    bias_path=dest/f"original-{year}-daybias.csv.gz"
    base.to_csv(base_path,index=False,compression="gzip")
    bias.to_csv(bias_path,index=False,compression="gzip")
    result={
        "contract":"KEIBA_ORIGINAL_PRIOR_ABILITY_CORE4_AND_DAYBIAS_V1",
        "year":year,"base_rows":len(base),"base_races":int(base["_race_id"].nunique()),
        "base_columns":len(features),"feature_sha256":digest,
        "bias_rows":len(bias),"bias_features":len(bias_cols(bias,with_fit=True)),
        "original_formulas_called":True,"target_race_id_not_loaded":True,
        "original_daybias_coverage":coverage
    }
    print("ORIGINAL_PRIOR_BASE_AND_BIAS_READY",json.dumps(result,sort_keys=True),flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--year",type=int,required=True)
    p.add_argument("--input",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    export(a.input,a.year,a.out)

if __name__=="__main__":main()
