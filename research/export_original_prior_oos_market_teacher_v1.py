#!/usr/bin/env python3
"""Extract original PRIOR-year market ranks for Correction Risk teacher ONLY.

This is historical teacher data from 2021-24. Selected 2025 target and
2026 are forbidden, and no postrace market price may enter inference.
Uses unchanged original read_market_year() + market_rank_frame().
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from run_l1_market_disagreement_audit_v1 import read_market_year
from run_l1_market_correction_distill_v1 import market_rank_frame

def export(path,year,output):
    if year not in (2021,2022,2023,2024):
        raise ValueError("HISTORICAL_OOS_TEACHER_MARKET_ONLY_PRIOR_2021_24")
    data=market_rank_frame(read_market_year(path,year))
    if data.empty or data.duplicated(["_race_id","_horse_id"]).any():
        raise ValueError("PRIOR_MARKET_TEACHER_KEY_DRIFT")
    if not data["_race_id"].astype(str).str[:4].eq(str(year)).all():
        raise ValueError("PRIOR_TEACHER_YEAR_MISMATCH")
    o=Path(output);o.parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(o,compression="gzip",index=False)
    print("FROZEN_ORIGINAL_OOS_MARKET_TEACHER_YEAR",json.dumps({
        "year":year,"horse_rows":len(data),
        "races":int(data["_race_id"].nunique()),
        "only_prior_teacher":True,
        "2025_selected_race_used":False,
        "2026_results_used":False},sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--year",type=int,required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    export(a.source,a.year,a.output)

if __name__=="__main__":main()
