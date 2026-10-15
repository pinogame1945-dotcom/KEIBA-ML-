#!/usr/bin/env python3
"""AUDIT ONLY: test original 19-feature market/AI geometry on real 2025 FINAL.

The original cache has result and FINAL quote columns. Read only allowlisted
earlier-computed score/risk/rank and FINAL odds for an independent RETROSPECTIVE
parity audit. Never export a prebet-labelled input or use archived outcomes.
"""
from __future__ import annotations
import argparse,csv,gzip,importlib.util,json,math,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def load(filename,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"research"/filename)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj
    spec.loader.exec_module(obj)
    return obj
G=load("anchor-original-feature-geometry.py","original_anchor_real_geometry")
def flt(raw):
    x=float(raw)
    if not math.isfinite(x):raise ValueError("REAL_CACHE_INPUT_NONFINITE")
    return x
def as_rank(raw):
    x=flt(raw)
    if not x.is_integer():raise ValueError("REAL_CACHE_RANK_NONINTEGER")
    return int(x)

def audit(cache,pred,parameters,race):
    if not (len(race)==12 and race[:4]=="2025" and race.isdigit()):
        raise ValueError("ONLY_2025_HISTORICAL_WITH_2026_HOLDOUT_SEALED")
    with gzip.open(cache,"rt",encoding="utf-8",newline="") as file:
        rows=[r for r in csv.DictReader(file) if r["_race_id"]==race]
    if not rows or any(r["year"]!="2025" for r in rows):
        raise ValueError("ACTUAL_HELDOUT_2025_CACHE_RACE_NOT_FOUND")
    by_no={}
    # This audit relies on BACKFILL original odds/canonical number map, which
    # must be provided explicitly for the target race; original cache has
    # horse_id but no horse_number.
    horse_nos=json.loads(Path(parameters).read_text())["audit_only_horse_numbers"] if False else None
    # Geometry requires numbered horses only to enforce uniqueness. For this
    # purely retrospective audit assign neutral arbitrary slot numbers to each
    # horse ID in sorted order. NEVER use the slots as real race horse numbers.
    ordered=sorted(rows,key=lambda r:r["_horse_id"])
    panel={};prices={};mapping={}
    for i,row in enumerate(ordered,1):
        hid=row["_horse_id"]
        mapping[i]=hid
        panel[i]={
          "horse_id":hid,
          "distill_ai_rank":as_rank(row["distill_ai_rank"]),
          "distill_score_z":flt(row["distill_score_z"]),
          "distill_score_gap_to_best":flt(row["distill_score_gap_to_best"]),
          "risk":flt(row["risk"]),
        }
        prices[i]=flt(row["final_win_odds"])
    actual=G.build_geometry(panel,prices)
    max_feat=0.0;max_market=0.0
    for i,row in enumerate(ordered,1):
        result=actual[i]
        if result["horse_id"]!=row["_horse_id"]:
            raise ValueError("GEOMETRY_HORSE_ID_DRIFT")
        if result["market_rank"]!=as_rank(row["market_rank"]):
            raise ValueError("HISTORICAL_FINAL_MARKET_RANK_TIE_DRIFT "+row["_horse_id"])
        max_market=max(max_market,abs(result["market_prob_norm"]-flt(row["market_prob_norm"])))
        for field in G.FIELDS:
            difference=abs(result["features"][field]-flt(row[field]))
            max_feat=max(max_feat,difference)
            if difference>1e-10:
                raise ValueError(f"ORIGINAL_{field}_FEATURE_FORMULA_DRIFT id={row['_horse_id']} difference={difference}")
    if max_market>1e-10:raise ValueError("NORMALIZED_WIN_MARKET_FORMULA_DRIFT "+str(max_market))
    model=json.loads(Path(parameters).read_text())
    if model.get("features_in_original_order")!=list(G.FIELDS):
        raise ValueError("ORIGINAL_EXPORTED_MODEL_FEATURE_ORDER_DRIFT")
    raw=[]
    for n,item in sorted(actual.items()):
        z=math.log(item["market_prob_norm"])
        for feat in G.FIELDS:
            unit=model["standardizer"][feat]
            z+=(item["features"][feat]-unit["mean"])/unit["std"]*model["coefficients"][feat]
        raw.append((item["horse_id"],z))
    top=max(z for _,z in raw)
    exponents=[math.exp(z-top) for _,z in raw]
    total=math.fsum(exponents)
    predicted={hid:ex/total for (hid,_),ex in zip(raw,exponents)}
    with gzip.open(pred,"rt",encoding="utf-8",newline="") as f:
        known={r["_horse_id"]:flt(r["P__ANCHOR_FULL"]) for r in csv.DictReader(f)
               if r["_race_id"]==race}
    if set(known)!=set(predicted):
        raise ValueError("FINAL_OOS_ANCHOR_HORSE_SET_DRIFT")
    max_prob=max(abs(known[hid]-p) for hid,p in predicted.items())
    if max_prob>5e-7:raise ValueError("EXACT_ANCHOR_GEOMETRY_TO_SAVED_PWIN_DRIFT "+str(max_prob))
    return {
      "contract":"KEIBA_ORIGINAL_19_FEATURES_REAL_FINAL_AUDIT_ONLY_V1",
      "race_id":race,"horses":len(rows),
      "max_19_feature_abs_error":max_feat,
      "max_final_market_normalization_error":max_market,
      "max_original_pwin_error":max_prob,
      "source":"2025_ORIGINAL_FINAL_FEATURES_AND_FINAL_WIN_PRICES",
      "never_validated_as_of_or_prebet":True,
      "synthetic_horse_slot_order_not_betting_horse_number":True,
      "target_win_and_finish_and_payout_fields_not_read":True,
      "2026_sealed":True,
    }
def main():
    p=argparse.ArgumentParser()
    for n in ("cache","predictions","parameters","race","output"):
        p.add_argument("--"+n,required=True)
    a=p.parse_args()
    result=audit(a.cache,a.predictions,a.parameters,a.race)
    dst=Path(a.output);dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ORIGINAL_19_REAL_FEATURE_GEOMETRY_AND_ANCHOR_PARITY",
          result["race_id"],result["horses"],result["max_19_feature_abs_error"],
          result["max_original_pwin_error"],flush=True)
if __name__=="__main__":main()
