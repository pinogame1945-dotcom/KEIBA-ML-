#!/usr/bin/env python3
"""Original 13 Outsider dependency audit + actual 8 direct-input dataset runs.

The original builder's build_features/process_day functions are never changed.
Only 8 frozen modes have no horse_history dependency; other 5 cannot be
faithfully materialized from target-day 717 without prior race signatures.
Even these 8 stay UNRELEASED; 2026 day-before snapshot lacks real-time earlier
winners and body-weight which must not be fabricated.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from build_l1_nonhorse_dataset_v1 import (
    process_day, horse_signature, MODES, STYLE_KEYS, RACE_BASE_KEYS,
)

SOURCE_CONTRACT="KEIBA_2026_ORIGINAL_13_OUTSIDER_STYLE14_BACKFILL13_DELTA_V1"
EXPECTED_STATeless={
    "outsider_daytrend":"DAY_TREND",
    "outsider_raceshape":"RACE_SHAPE",
    "outsider_gatecourse":"GATE_COURSE",
    "outsider_jockey":"JOCKEY",
    "outsider_field":"FIELD_COMPOSITION",
    "outsider_chimera":"CHIMERA",
    "outsider_rotation":"ROTATION",
    "outsider_opponentgap":"OPPONENT_GAP",
}
REQUIRES_HISTORY={
    "outsider_classmove":"CLASS_MOVE",
    "outsider_weightchange":"WEIGHT_CHANGE",
    "outsider_bodychange":"BODY_CHANGE",
    "outsider_disttransition":"DISTANCE_TRANSITION",
    "outsider_coursetransition":"COURSE_TRANSITION",
}
LIVE_DAY="2026-10-17"
MARKET=("odds","popularity","payout")
TARGET_RESULT_KEYS={"target","finish_position","is_win","market_outcome","_finish","_is_win","_is_top3"}

def require(x,why):
    if not x:raise ValueError(why)

def read_json(file):
    return json.loads(file.read_text(encoding="utf-8"))

def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()

def load_config():
    combos={}
    for fname in ("l1-outsider-nonhorse-v1.json","l1-outsider-transition-v1.json"):
        cfg=read_json(Path(__file__).parent/fname)
        require(cfg["locked_years"]==[2026] and
                cfg["training_window_years"]==2 and
                cfg["snapshot_generation"]=="eb13d4096519167b" and
                cfg["ability_uses_odds"] is False and
                cfg["prediction_phase"]=="FINAL",
                "ORIGINAL_13_OUTSIDER_FROZEN_CONFIG_DRIFT")
        for c in cfg["candidates"]:
            require(c["name"] not in combos,"DUPLICATED_ORIGINAL_OUTSIDER")
            combos[c["name"]]=c["mode"]
    require(len(combos)==13 and
            {**EXPECTED_STATeless,**REQUIRES_HISTORY}==combos and
            set(combos.values())<=MODES,"13_OUTSIDER_MODE_DEPENDENCY_DRIFT")
    return combos

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cached-backfill",type=Path,required=True)
    p.add_argument("--delta27",type=Path,required=True)
    p.add_argument("--feature-contract",type=Path,required=True)
    p.add_argument("--small-sample-contract",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--prior-signatures",type=Path,help="Optional strict-prior original last3 horse signatures, required for remaining 5")
    a=p.parse_args()
    require(not a.output.exists(),"NO_OUTSIDER_FEATURE_OUTPUT_OVERWRITE")
    modes=load_config()
    frozen=read_json(a.feature_contract)
    small=read_json(a.small_sample_contract)
    require(frozen["contract"]=="L1_FEATURE_SET_CONTRACT_V1" and
            frozen["history_windows"]["style_last3f"]==10 and
            small["contract"]=="L1_SMALL_SAMPLE_CONTRACT_V1",
            "ORIGINAL_BACKFILL_FEATURE_META_DRIFT")
    meta={"history_windows":frozen["history_windows"],
          "small_sample_policy":small["defaults"]}
    src=a.cached_backfill/"ml_features/delivery/690"
    index=read_json(src/"index.json")
    require(index["feature_count"]==690 and index["total_races"]==60 and
            index["total_horses"]==786 and
            index["readiness"]=="ASOF_UNVERIFIED_DO_NOT_BET",
            "PINNED_690_NOT_ASOF_UNVERIFIED_60_786")
    races={r:e for r,e in index["races"].items() if e["date"]==LIVE_DAY}
    require(len(races)==36 and sum(x["starters"] for x in races.values())==471,
            "LIVE_2026_10_17_36R_471H_DRIFT")
    delta={}
    with gzip.open(a.delta27,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            rid,hid=str(row.get("race_id") or ""),str(row.get("horse_id") or "")
            key=(rid,hid)
            fields=row.get("features_delta") or {}
            require(rid in races and key not in delta and len(fields)==27 and
                    row.get("contract")==SOURCE_CONTRACT and
                    row.get("style_source_complete") is True and
                    row.get("target_2026_results_used") is False and
                    row.get("ready_for_betting") is False,
                    "DELTA_27_PROVENANCE_OR_POPULATION_MISMATCH")
            delta[key]=(fields,row["cached_690_race_sha256"])
    require(len(delta)==471,"ORIGINAL_OUTSIDER_DELTA_NOT_471")
    day=[]
    matched=set()
    missing_counts=Counter()
    for rid,entry in sorted(races.items()):
        file=a.cached_backfill/entry["path"]
        require(file.is_file() and digest(file)==entry["sha256"],
                "ORIGINAL_690_FILE_SHA_DRIFT:"+rid)
        x=read_json(file)
        require(x["race_id"]==rid and x["feature_count"]==690 and
                x["starters"]==entry["starters"] and
                x["readiness"]=="ASOF_UNVERIFIED_DO_NOT_BET" and
                x["used_target_results_or_final_odds"] is False,
                "ORIGINAL_690_META_DRIFT:"+rid)
        for raw in x["rows"]:
            key=(rid,str(raw["_horse_id"]))
            require(key in delta and key not in matched and
                    raw["_race_id"]==rid and raw["_race_date"]==LIVE_DAY,
                    "ORIGINAL_690_DELTA_ID_DRIFT")
            fields,sha=delta[key]
            require(sha==entry["sha256"],"ORIGINAL_690_DELTA_SOURCE_SHA_DRIFT")
            features={k:v for k,v in raw.items() if not k.startswith("_")}
            require(len(features)==690 and not(set(features)&set(fields)),
                    "690_OR_ADDED_27_KEY_COLLISION")
            features.update(fields)
            # race_date is _race_date identity in cached 690, not a model column.
            # The original Outsider builder expects this date inside features.
            features["race_date"]=LIVE_DAY
            require(len(features)==718 and
                    not (set(features)&TARGET_RESULT_KEYS) and
                    not any(any(b in str(k).lower() for b in MARKET) for k in features),
                    "UNSAFE_RESULT_OR_MARKET_FEATURE")
            require(all(k in features for k in RACE_BASE_KEYS+STYLE_KEYS),
                    "ORIGINAL_13_OUTSIDER_COMMON_INPUT_MISSING")
            # preserve null values exactly: no imputation and no postrace data.
            missing_counts.update(k for k,v in fields.items() if v is None)
            day.append({
                "ml_dataset_version":3,"feature_schema_version":8,
                "leakage_policy":"STRICT_PRIOR_DATE_ONLY",
                "prediction_phase":"FINAL",
                "race_id":rid,"horse_id":key[1],
                "features":features,
                "target":{},
            })
            matched.add(key)
    require(len(day)==471 and len(matched)==len(delta),
            "TARGET_UNIVERSE_DID_NOT_MATCH_471")
    # Since raw live source is from 2026-10-16 UTC, no prior same-day winning
    # races exist. Do not fabricate an intraday observed result or odds.
    histories=defaultdict(list)
    if a.prior_signatures:
        rows_by_key={}
        with a.prior_signatures.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():continue
                row=json.loads(line)
                key=(str(row.get("race_id") or ""),str(row.get("horse_id") or ""))
                historic=row.get("last_three_records")
                require(key in matched and key not in rows_by_key and
                        row.get("contract")=="KEIBA_2026_10_17_ORIGINAL_OUTSIDER_5_PRIOR_HORSE_SIGNATURE_SOURCE_V1" and
                        row.get("strict_prior_source_dates")==["2022-01-01","2026-10-16"] and
                        row.get("2026_target_results_used") is False and
                        row.get("source_is_original_pinned_backfill") is True and
                        row.get("model_ready") is False and
                        row.get("ready_for_betting") is False and
                        isinstance(historic,list) and len(historic)<=3,
                        "FIVE_OUTSIDER_PRIOR_SIGNATURE_PROVENANCE_DRIFT")
                rows_by_key[key]=row
        require(set(rows_by_key)==matched,
                "FIVE_OUTSIDER_PRIOR_SIGNATURE_471_NOT_MATCHED")
        seen_prior={}
        for item in day:
            key=(item["race_id"],item["horse_id"])
            row=rows_by_key[key]
            hist=row["last_three_records"]
            expected=next(z for z in item["features"].items() if z[0]=="prior_starts")[1]
            require(row.get("observed_prior_starts")==expected and
                    len(hist)==min(3,int(expected)),
                    "FIVE_OUTSIDER_HISTORIC_STARTS_PARITY_DRIFT")
            prior=tuple(sorted(json.dumps(x,sort_keys=True) for x in hist))
            hid=item["horse_id"]
            if hid in seen_prior:
                require(prior==seen_prior[hid],
                        "FIVE_OUTSIDER_SAME_HORSE_CONFLICTING_HISTORY")
            else:
                histories[hid]=[horse_signature({"features":original}) for original in hist]
                seen_prior[hid]=prior
        require(len(histories)==len(seen_prior),
                "FIVE_OUTSIDER_HISTORIC_SOURCE_DUPLICATE_DRIFT")
    a.output.mkdir(parents=True)
    result={}
    all_modes={**EXPECTED_STATeless,
               **(REQUIRES_HISTORY if a.prior_signatures else {})}
    for cand,mode in all_modes.items():
        dest=a.output/(cand+".jsonl.gz")
        with gzip.open(dest,"wt",encoding="utf-8") as writer:
            output_rows=process_day(day,mode,writer,meta,defaultdict(list,{
                h:list(items) for h,items in histories.items()
            }))
        require(output_rows==471,"ORIGINAL_13_OUTSIDER_8_MODE_ROW_COUNT_DRIFT:"+cand)
        seen=set()
        feature_sets=set()
        feature_union=set()
        with gzip.open(dest,"rt",encoding="utf-8") as reader:
            for line in reader:
                if not line.strip():continue
                row=json.loads(line)
                key=(str(row["race_id"]),str(row["horse_id"]))
                require(key in matched and key not in seen and
                        row.get("leakage_policy")=="STRICT_PRIOR_DATE_ONLY" and
                        row.get("target")=={} and
                        row.get("prediction_phase")=="FINAL",
                        "OUTSIDER_2026_SOURCE_LABEL_LEAK_OR_DUPLICATE:"+cand)
                features=row["features"]
                require(len(features)>0 and
                        all(k in features for k in RACE_BASE_KEYS) and
                        not (set(features)&TARGET_RESULT_KEYS) and
                        not any(any(b in k.lower() for b in MARKET) for k in features),
                        "OUTSIDER_2026_RESULT_OR_MARKET_FEATURE:"+cand)
                if cand in {"outsider_daytrend","outsider_chimera"}:
                    require(features["nh_day_prior_races"]==0 and
                            features["nh_day_has_prior"]==0,
                            "OUTSIDER_PRE_DAY_CAPTURE_INTRADAY_RESULTS_NOT_PERMITTED")
                feature_sets.add(tuple(sorted(features)))
                feature_union.update(features)
                seen.add(key)
        require(seen==matched and len(feature_union)>0,
                "OUTSIDER_STATELESS_8_POPULATION_OR_FEATURE_SCHEMA_DRIFT:"+cand)
        # Original OPPONENT_GAP adds dynamic keys only for usable opponent
        # observations. Frozen downstream projector performs union/NaN align.
        if cand!="outsider_opponentgap":
            require(len(feature_sets)==1,
                    "OUTSIDER_STATIC_7_FEATURE_SCHEMA_DRIFT:"+cand)
        else:
            require(all(set(fs)<=feature_union for fs in feature_sets) and
                    all(k in RACE_BASE_KEYS or k.startswith("nh_")
                        for k in feature_union),
                    "OUTSIDER_DYNAMIC_OPPONENT_FEATURES_DRIFT")
        result[cand]={
            "mode":mode,"rows":471,
            "feature_count_union":len(feature_union),
            "distinct_feature_schemas":len(feature_sets),
            "output_sha256":digest(dest),
            "size_bytes":dest.stat().st_size,
            "source_target_before_day":True,
            "2026_training_or_evaluation":False,
            "asof_verified":False,
        }
        print("ORIGINAL_OUTSIDER_ONE_OF_EIGHT_TARGET_DATASET_READY "+
              json.dumps({"candidate":cand,"mode":mode,
                          "rows":471,"features_union":result[cand]["feature_count_union"],
                          "ready_for_betting":False},sort_keys=True),flush=True)
    manifest={
        "contract":"KEIBA_2026_10_17_ORIGINAL_OUTSIDER_8_STATIC_FEATURE_DATASETS_V1",
        "target_races":36,"target_horses":471,"day":LIVE_DAY,
        "input_features_690":690,"original_delta_features":27,
        "candidate_original_count":13,
        "candidate_source_feature_datasets_ready":len(all_modes),
        "candidate_source_feature_datasets_blocked":13-len(all_modes),
        "blocked_candidates":{} if a.prior_signatures else REQUIRES_HISTORY,
        "prior_history_recovered":bool(a.prior_signatures),
        "ready_candidates":result,
        "additional_null_cells_by_column":dict(sorted(missing_counts.items())),
        "original_builder": "build_l1_nonhorse_dataset_v1.py:process_day",
        "2026_prior_results_feature_history_only":True,
        "2026_target_results_used":False,
        "2026_training_or_evaluation":False,
        "actual_model_weights_loaded":False,
        "actual_prediction_scores_produced":False,
        "asof_full_input_audited":False,
        "ready_for_betting":False
    }
    (a.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,
                                                    indent=2)+"\n",encoding="utf-8")
    print("ORIGINAL_2026_13_OUTSIDER_FEATURE_INPUT_COVERAGE_AUDIT "+
          json.dumps({"races":36,"horses":471,
             "candidate_input_ready":len(all_modes),
             "candidate_blocked_history":13-len(all_modes),
             "model_scorings":0,"2026_target_results_used":False,
             "all_13_model_ready":False},sort_keys=True),flush=True)
if __name__=="__main__":
    main()
