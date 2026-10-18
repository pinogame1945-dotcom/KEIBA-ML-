#!/usr/bin/env python3
"""Read-only 2026/10/17 original CORE4 source parity gate against 2025 original.

Uses pre-existing BACKFILL 690 files, no model fitting or new feature generation.
Distinguishes parity of the base 649 from completeness of the original 2026
L1 CORE4+day bias+sequence/regime+Correction Risk+ANCHOR_FULL inference.
"""
from __future__ import annotations
import argparse, ast, datetime, hashlib, json, math
from pathlib import Path
from collections import Counter

BASE_COUNT=649
DAY_BIAS_COUNT=19
SPEED_COUNT=22
DAY="2026-10-17"
MODEL_CONTRACT="KEIBA_UNSCORED_RACE_FEATURE_EXTRACT_V1"
SOURCE_SHA="d6607946292efe79fc5eef8eaf143e689c41f6a9"
SOURCE_REF="handoff/690-feature-data-v1"
INDEX="KEIBA_BACKFILL_TO_ML_690_INDEX_V1"
OUT="KEIBA_ORIGINAL_CORE4_2026_1017_CACHED690_READONLY_SOURCE_AUDIT_V1"
MISSING=("2026 ORIGINAL CORE4 trained LambdaRank full-model artifact or exact-year original training replay",
         "original 2026 strict-prior sequential race-state and regime feature parity (not the 22 speed columns)",
         "2026 ORIGINAL past-OOS trained Correction Risk teacher and target inference",
         "2026 ORIGINAL ANCHOR_FULL coefficients/normalizer and full-field prior-only probability inference",
         "independently verified source capture as-of and same-snapshot market quote cutoff")

def need(ok,reason):
    if not ok:raise ValueError(reason)

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()

def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def original_hash(source):
    tree=ast.parse(Path(source).read_text(encoding="utf-8"))
    target={}
    for expr in tree.body:
        if isinstance(expr,ast.Assign):
            for t in expr.targets:
                if isinstance(t,ast.Name) and t.id in ("BASE_SHA","FIT_YEARS","TARGET_YEAR"):
                    try:target[t.id]=ast.literal_eval(expr.value)
                    except (ValueError,TypeError):pass
    need(target.get("BASE_SHA") and target.get("FIT_YEARS")== (2023,2024) and
      target.get("TARGET_YEAR")==2025,"ORIGINAL_2025_FROZEN_CORE4_GATE_SOURCE_DRIFT")
    return target["BASE_SHA"]

def audit(root,original,out):
    need(not out.exists(),"REFUSE_OVERWRITE_ORIGINAL_SOURCE_AUDIT")
    base=root/"ml_features/delivery"
    index=load(base/"690/index.json")
    need(index.get("contract")==INDEX and index.get("branch")==SOURCE_REF and
         len(index.get("races") or {})==60,"PINNED_BACKFILL_690_INDEX_DRIFT")
    selected={rid:v for rid,v in index["races"].items() if v["date"]==DAY}
    need(len(selected)==36,"BACKFILL_DAILY_36_RACES_MISSING")
    reference_hash=original_hash(original)
    first_keys=None;base_hash=None;total=0;race_digest={};identities=set()
    zero=Counter()
    for rid,meta in sorted(selected.items()):
        need(len(rid)==12 and rid.isdecimal() and rid.startswith("2026"),
             "INVALID_L1_ORIGINAL_TARGET_RACE_ID")
        path=base/"690"/DAY/(rid+".json")
        need(path.is_file() and
             meta.get("path")=="ml_features/delivery/690/"+DAY+"/"+rid+".json" and
             sha(path)==meta["sha256"],"CACHED_690_SOURCE_SHA256_DRIFT:"+rid)
        d=load(path)
        need(d.get("contract")=="KEIBA_BACKFILL_TO_ML_690_UNSCORED_V1" and
          d.get("race_id")==rid and d.get("race_date")==DAY and
          d.get("feature_count")==690 and
          d.get("starters")==meta["starters"] and
          d.get("used_target_results_or_final_odds") is False and
          d.get("source_capture_asof_independently_verified") is False and
          d.get("original_ml_full_numerical_parity_verified") is False,
          "2026_BACKFILL_PRETARGET_SOURCE_DRIFT:"+rid)
        horses=d.get("rows")
        need(isinstance(horses,list) and len(horses)==d["starters"],
             "CACHED_690_HORSE_POPULATION_DRIFT:"+rid)
        race_numbers=set();race_horses=set()
        for item in horses:
            keys=list(item)
            need(len(keys)==693 and keys[:3]==["_race_id","_horse_id","_race_date"],
                 "ORIGINAL_690_KEY_FIELDS_OR_ORDER_DRIFT:"+rid)
            observed=keys[3:]
            if first_keys is None: first_keys=observed
            need(observed==first_keys,"CACHED_690_SCHEMA_DIFFERS_BY_HORSE:"+rid)
            hid=str(item["_horse_id"])
            no=item.get("horse_number")
            need(item["_race_id"]==rid and item["_race_date"]==DAY and
                 len(hid)>0 and type(no) is int and 1<=no<=24 and
                 no not in race_numbers and hid not in race_horses and
                 (rid,hid) not in identities,"2026_HORSE_CANONICAL_ID_DRIFT:"+rid)
            need(not any(k in item for k in (
                "_finish","_is_win","_is_top3","result","payout",
                "final_win_odds","final_popularity","P__ANCHOR_FULL",
                "distill_ai_rank","risk","target")),
                 "ORIGINAL_2026_PRETARGET_LABEL_OR_FAKE_PROBABILITY_PRESENT:"+rid)
            race_numbers.add(no);race_horses.add(hid);identities.add((rid,hid));total+=1
        race_digest[rid]={"sha256":meta["sha256"],"horses":len(race_horses)}
        for item in horses:
            for k in first_keys:
                if item.get(k) is None:zero[k]+=1
    need(len(race_digest)==36 and total==471,"ORIGINAL_36R_471H_SCOPE_DRIFT")
    core=first_keys[:BASE_COUNT]
    daytime=first_keys[BASE_COUNT:BASE_COUNT+DAY_BIAS_COUNT]
    speed=first_keys[BASE_COUNT+DAY_BIAS_COUNT:]
    need(len(core)==BASE_COUNT and len(daytime)==DAY_BIAS_COUNT and len(speed)==SPEED_COUNT and
         len(set(first_keys))==690 and
         all(x.startswith("daybias_") or x.startswith("horse_") or x.startswith("fit_") for x in daytime) and
         all(x.startswith("regime_speed_") for x in speed),
         "KNOWN_649_PLUS_19_PLUS_22_SOURCE_STRUCTURE_CHANGED")
    current_hash=hashlib.sha256(json.dumps(sorted(core),separators=(",",":")).encode()).hexdigest()
    exact=current_hash==reference_hash
    analysis={
      "contract":OUT,"source_repository":"pinogame1945-dotcom/KEIBA-BACKFILL",
      "source_commit":SOURCE_SHA,"source_branch":SOURCE_REF,
      "target_date":DAY,"races":36,"horses":471,
      "feature_total":690,"base649":len(core),"daybias19":len(daytime),"speed22":len(speed),
      "original_2025_core4_base_feature_sha256":reference_hash,
      "observed_2026_base649_feature_sha256":current_hash,
      "base649_original_core4_name_set_exact_match":exact,
      "not_a_numerical_parity_certificate":True,
      "original_risk_model_run":False,"original_anchor_full_run":False,
      "original_core4_2026_trained_model_run":False,
      "original_2026_model_weights_available_verified":False,
      "all_36_races_win_odds_source_independently_asof_verified":False,
      "l21_original_50_rule_engine_invoked":False,"l3_v3_invoked":False,
      "research_only":True,"ready_for_betting":False,
      "2026_results_read":False,
      "remaining_original_l21_inputs":list(MISSING),
      "race_sources":race_digest,
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ORIGINAL_2026_REAL_CACHED690_SOURCE_AUDIT "+
        json.dumps({k:analysis[k] for k in ("races","horses","base649","daybias19","speed22",
        "base649_original_core4_name_set_exact_match","observed_2026_base649_feature_sha256",
        "original_2025_core4_base_feature_sha256","ready_for_betting")},sort_keys=True),flush=True)

def main():
    parser=argparse.ArgumentParser()
    for name in ("backfill-handoff","original-2025-source","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args()
    audit(a.backfill_handoff,a.original_2025_source,a.output)
if __name__=="__main__":main()
