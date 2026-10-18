#!/usr/bin/env python3
"""One-pass historical source projection for original 2026 ability/risk.

Uses *unchanged* original 2025 CORE4, track bias, compact state and market
teacher operators on 2019-2025 history ONLY. Never reads 2026 outcomes.
Historical source is consumed once per year then deleted by workflow.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import pandas as pd

from run_l1_objective_rebuild_v1 import read_year,select_variant,META
from run_l1_ceiling_audit_v1 import strip_forbidden,FULL_FORBIDDEN
from run_l1_sameday_track_bias_v1 import _build_same_day_bias_year,bias_cols
from run_l1_state_transition_v1 import read_compact_state_source
from run_l1_market_disagreement_audit_v1 import read_market_year
from run_l1_market_correction_distill_v1 import market_rank_frame

SHA="eb3392b725560e77d46b28aef036b6188814125ed8f7a3bfde0c53522dbadf58"

def need(x,reason):
    if not x:raise ValueError(reason)

def export(year,source,dest,manifest_year,manifest_rows,manifest_first,manifest_last):
    need(2019<=year<=2025 and year==manifest_year,"ONLY_HISTORICAL_2019_2025")
    need(source.is_file() and manifest_rows>0 and
         manifest_first.startswith(str(year)+"-") and manifest_last.startswith(str(year)+"-"),
         "HISTORICAL_SOURCE_MANIFEST_UNTRUSTED")
    dest.mkdir(parents=True,exist_ok=True)
    # Use the same exact original compact feature reader; all source outcomes
    # are in 2019..2025 (strictly prior to sealed 2026 target).
    comp=read_compact_state_source(source,year)
    need(len(comp)>0 and not comp.duplicated(["_race_id","_horse_id"]).any() and
         comp["_race_date"].astype(str).str[:4].eq(str(year)).all() and
         comp["_race_date"].astype(str).lt("2026-10-17").all(),
         "ORIGINAL_COMPACT_HISTORY_INVALID")
    compact=dest/f"compact-strict-prior-{year}.csv.gz"
    comp.to_csv(compact,index=False,compression="gzip")
    out={"year":year,"historical_compact_rows":len(comp),
         "historical_compact_races":int(comp["_race_id"].nunique()),
         "source_observed_historical_only":True,"2026_outcomes_used":False,
         "source_manifest_year":manifest_year,"source_manifest_rows":manifest_rows,
         "source_manifest_first":manifest_first,"source_manifest_last":manifest_last}
    del comp
    if year>=2020:
        # Original 2025 cold-replay CORE4 source projection unchanged.
        base=select_variant(strip_forbidden(read_year(source,year)),"core4")
        columns=sorted(set(base.columns)-META)
        sig=hashlib.sha256(json.dumps(columns,separators=(",",":")).encode()).hexdigest()
        need(len(columns)==649 and sig==SHA and
             not(set(base)&FULL_FORBIDDEN) and
             not base.duplicated(["_race_id","_horse_id"]).any(),
             "ORIGINAL_CORE4_HISTORICAL_SCHEMA_OR_UNIQUE_DRIFT:"+str(year))
        observed,bias,coverage=_build_same_day_bias_year(year,source)
        need(observed==year and not bias.duplicated(["_race_id","_horse_id"]).any(),
             "ORIGINAL_PRIOR_DAYBIAS_KEYS_DRIFT:"+str(year))
        k1=set(zip(base["_race_id"],base["_horse_id"]))
        k2=set(zip(bias["_race_id"],bias["_horse_id"]))
        need(k1==k2 and len(base)==len(bias),"ORIGINAL_CORE4_BIAS_POPULATION_DRIFT:"+str(year))
        base.to_csv(dest/f"original-{year}-core4-base.csv.gz",index=False,compression="gzip")
        bias.to_csv(dest/f"original-{year}-daybias.csv.gz",index=False,compression="gzip")
        out.update({"original_core4_training_rows":len(base),
                    "original_base649_names_sha256":sig,
                    "original_bias_columns":len(bias_cols(bias,with_fit=True)),
                    "past_day_bias_only":True})
        del base,bias
    if year>=2022:
        market=market_rank_frame(read_market_year(source,year))
        need(not market.empty and
             market["_race_id"].astype(str).str[:4].eq(str(year)).all() and
             not market.duplicated(["_race_id","_horse_id"]).any(),
             "ORIGINAL_HISTORICAL_MARKET_TEACHER_INVALID:"+str(year))
        market.to_csv(dest/f"original-{year}-prior-market.csv.gz",
                      index=False,compression="gzip")
        out["original_past_teacher_market_rows"]=len(market)
    (dest/f"original-{year}-projection-proof.json").write_text(
         json.dumps(out,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf8")
    print("ORIGINAL_2026_PAST_YEAR_ONCE_COMPLETE "+json.dumps(out,sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--year",type=int,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--manifest-year",type=int,required=True)
    p.add_argument("--manifest-rows",type=int,required=True)
    p.add_argument("--manifest-first",required=True)
    p.add_argument("--manifest-last",required=True)
    a=p.parse_args()
    export(a.year,a.source,a.out,a.manifest_year,a.manifest_rows,a.manifest_first,a.manifest_last)
if __name__=="__main__":main()
