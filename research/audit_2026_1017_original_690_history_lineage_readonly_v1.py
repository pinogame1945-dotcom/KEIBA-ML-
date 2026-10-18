#!/usr/bin/env python3
"""Independent read-only audit of ORIGINAL 690 saved-feature lineage.

Validate all 36 source-race files and 471 horse rows, exact immutable Git SHA
and per-file checksums, original 649 CORE4 column-set digest, original 19+22
names, strictly-prior target history source files and generator input guards.
Do NOT claim numerical parity, model release, or full independently proven
history/event as-of timing. No model fit, recalculation or 2026 outcome labels.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, subprocess
from pathlib import Path

DAY="2026-10-17"
HEAD="10a500619f4061d6c2b20cfbfa0ac01093720807"
DELIVERY="d6607946292efe79fc5eef8eaf143e689c41f6a9"
CORE4_SHA="eb3392b725560e77d46b28aef036b6188814125ed8f7a3bfde0c53522dbadf58"
FEATURE_SHA="1667293eed867b77bbccf9af7dea758f0c9252581de40aedb64fded938621384"
B19={
 "daybias_prior_races","daybias_front_gap_mean","daybias_front_gap_ewma",
 "daybias_inner_gap_mean","daybias_inner_gap_ewma",
 "daybias_last3_gap_mean","daybias_last3_gap_ewma",
 "daybias_winner_front_mean","daybias_winner_front_ewma",
 "daybias_winner_inner_mean","daybias_winner_inner_ewma",
 "daybias_speed_adv_mean","daybias_speed_adv_ewma",
 "horse_front_score","horse_inner_score","fit_front_mean",
 "fit_front_ewma","fit_inner_mean","fit_inner_ewma"}
FORBIDDEN={"_finish","_is_win","_is_top3","target","market_outcome",
 "final_win_odds","final_popularity","_score"}
GUARDS={
 "ml_features/research/auto_new_day_handoff_2026_v1.py":[
    'if "2022-01-01"<=dt<date_text:',
    '"STRICT_PRIOR_HISTORY_BOUNDARY"',
    '"LIVE_TARGET_RESULTS_OR_FINAL_MARKET_FOUND"',
    '"--allow-unresulted-targets"',
    '"--source-end",prior_day',
    '"--source-end",date_text'],
 "ml_features/research/export_690_delivery_dynamic_v1.py":[
    'frame["_race_date"].astype(str).lt(date).all()',
    '"HISTORY_LOOKAHEAD:',
    '"CURRENT_DAY_VIEW_NOT_BYTE_EQUIVALENT_TO_FULL_TARGET_ROWS"',
    'bias19(view, target)',
    'speed22(history,',
    '"used_target_results_or_final_odds": False'],
 "ml_features/research/build_690_any_race_v1.py":[
    '"FUTURE_HISTORY_USED"',
    'if feat.get("race_date") != day:',
    'int(feat.get("race_no") or 0) < number',
    'if completed:',
    'KEY_PREFIX_BLOCK = ("P__", "distill_", "risk_")'],
 "ml_features/research/build-yearly-superset-snapshots.mjs":[
    '"LIVE_TARGET_MUST_NOT_CONTAIN_RESULTS_PAYOUTS_OR_ODDS"',
    'allowUnresultedTargets && group.date === emitStart',
    'processDay(group.rows, group.date,'],
 "ml_features/research/live_json_to_daily_pack_dynamic_v1.py":[
    '"results": [], "payouts": [], "odds": [], "laps": [], "corners": []',
    '"source": "JRA_OFFICIAL_POST"' if False else '"live_market_data_stored_separately": True'],
}

def need(ok,reason):
    if not ok: raise ValueError(reason)

def sha(data):return hashlib.sha256(data).hexdigest()
def read(p):return json.loads(p.read_bytes())

def head(root):
    proc=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],
                        capture_output=True,text=True,check=True)
    return proc.stdout.strip()

def historic_tree_dates(root):
    proc=subprocess.run(["git","-C",str(root),"ls-tree","-r","--name-only","HEAD","data/daily"],
                        capture_output=True,text=True,check=True)
    days=[]
    for p in proc.stdout.splitlines():
        match=re.fullmatch(r"data/daily/(\d{4}-\d{2}-\d{2})\.jsonl\.gz",p)
        if match: days.append(match.group(1))
    need(len(days)>700 and days==sorted(set(days)) and
         max(days)<DAY,"HISTORIC_JRA_INPUT_INCLUDES_TARGET_OR_FUTURE_DAY")
    return {"source_daily_packs":len(days),
            "source_daily_latest":max(days),
            "source_daily_earliest":min(days)}

def check_code(root):
    seals={}
    for path,needles in GUARDS.items():
        p=root/path
        need(p.is_file(),"PINNED_ORIGINAL_BUILDER_SCRIPT_ABSENT:"+path)
        blob=p.read_bytes();txt=blob.decode("utf8")
        for needle in needles:
            need(needle in txt,"ORIGINAL_BUILDER_SOURCE_GUARD_DRIFT:"+path+":"+needle)
        seals[path]=sha(blob)
    return seals

def audit(source,delivery,output):
    need(not output.exists(),"REFUSE_TO_OVERWRITE_PREVIOUS_690_AUDIT")
    need(head(source)==HEAD and head(delivery)==DELIVERY,
         "PINNED_BACKFILL_REPOSITORY_COMMIT_SHA_MISMATCH")
    age=historic_tree_dates(source)
    guards=check_code(source)
    base=delivery/"ml_features/delivery/690"
    idx=read(base/"index.json")
    need(idx.get("contract")=="KEIBA_BACKFILL_TO_ML_690_INDEX_V1" and
         idx.get("feature_count")==690 and idx.get("total_races")==60 and
         idx.get("total_horses")==786 and
         idx.get("feature_name_sha256")==FEATURE_SHA and
         idx.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET" and
         idx.get("original_ml_full_numerical_parity_verified") is False,
         "ORIGINAL_690_INDEX_OR_ASOF_SEAL_DRIFT")
    targets={rid:info for rid,info in idx["races"].items()
             if info["date"]==DAY}
    need(len(targets)==36 and sum(x["starters"] for x in targets.values())==471,
         "FROZEN_2026_TARGET_SCOPE_MISMATCH")
    rawdir=source/"data/live"/DAY
    files=sorted(x for x in rawdir.glob("*.json") if x.name!="manifest.json")
    need(len(files)==36 and {x.stem for x in files}==set(targets),
         "SOURCE_AND_SAVED_FEATURE_RACE_IDS_MISMATCH")
    names=None;total=0;cells=0;invalid=0;nonzero_daybias=0
    race_audit=[];all_horses=set()
    for rid,item in sorted(targets.items()):
        source_doc=read(rawdir/(rid+".json"))
        fpath=delivery/item["path"]
        need(fpath.is_file() and sha(fpath.read_bytes())==item["sha256"],
             "SAVED_690_HORSE_FEATURE_BYTE_SHA_FAILED:"+rid)
        doc=read(fpath)
        need(source_doc.get("race_id")==rid and source_doc.get("race_date")==DAY and
             source_doc.get("includes_results") is False and
             source_doc.get("includes_final_odds") is False and
             source_doc.get("status")=="SOURCE_AND_ODDS_VERIFIED" and
             source_doc.get("backfill_history_features_computed") is False and
             doc.get("contract")=="KEIBA_BACKFILL_TO_ML_690_UNSCORED_V1" and
             doc.get("race_id")==rid and doc.get("race_date")==DAY and
             doc.get("feature_count")==690 and
             doc.get("source_contract")=="ORIGINAL_CORE4_649_DAYBIAS_19_SPEED_REGIME_22" and
             doc.get("used_target_results_or_final_odds") is False and
             doc.get("source_capture_asof_independently_verified") is False and
             doc.get("original_ml_full_numerical_parity_verified") is False and
             doc.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET",
             "ORIGINAL_FROZEN_690_SOURCE_DOCUMENT_INCOMPATIBLE:"+rid)
        orig=source_doc["netkeiba"]["runners"]
        official=source_doc["jra_card"]["official_horse_set"]
        named={str(h["horse_name"]):str(h["horse_id"]) for h in orig}
        active={named[h["horse_name"]] for h in official
                if h["entry_status"]=="LISTED"}
        rows=doc.get("rows")
        need(isinstance(rows,list) and
             len(rows)==item["starters"]==len(active) and
             {str(h.get("_horse_id")) for h in rows}==active,
             "ORIGINAL_690_HORSE_IDENTITY_DIFFERENT_FROM_OFFICIAL_CARD:"+rid)
        for row in rows:
            hid=str(row["_horse_id"])
            need((rid,hid) not in all_horses and
                 row["_race_id"]==rid and row["_race_date"]==DAY,
                 "690_DUPLICATE_OR_MISDATED_HORSE_ROW:"+rid+":"+hid)
            all_horses.add((rid,hid))
            feature_cols=set(row)-{"_race_id","_horse_id","_race_date"}
            need(len(feature_cols)==690 and
                 not (feature_cols & FORBIDDEN) and
                 not any(x.startswith(("P__","distill_","risk_")) for x in feature_cols),
                 "690_HAS_POSTRESULT_OR_MODEL_LEAKAGE_SCHEMA:"+rid)
            if names is None: names=feature_cols
            need(feature_cols==names,"690_FULL_SCHEMA_NOT_UNIFORM:"+rid)
            speed={x for x in feature_cols if x.startswith("regime_speed_resid_")}
            need(len(speed)==22 and B19.issubset(feature_cols) and
                 not (speed & B19) and len(feature_cols-B19-speed)==649,
                 "ORIGINAL_649_19_22_INPUT_SPLIT_DRIFT:"+rid)
            # No completed target-day results were supplied in live source.
            # Therefore same-day *outcome* bias must be empty; geometry based
            # only on announced entries may still exist in 6 horse_* / fit_*.
            if row["daybias_prior_races"] not in (0,0.0):nonzero_daybias+=1
            for val in row.values():
                if isinstance(val,float) and not math.isfinite(val):invalid+=1
            cells+=690;total+=1
        race_audit.append({"race_id":rid,"horses":len(rows),
            "source_race_id_card_matched":True,
            "original_feature_checksum_verified":True,
            "target_result_and_final_market_source_excluded":True,
            "690_numerical_parity_independently_verified":False})
    need(total==471 and cells==324990 and invalid==0 and nonzero_daybias==0,
         "REAL_690_SCHEMA_FINITE_OR_UNRESULTED_DAYBIAS_SCOPE_INVALID")
    d649=sorted(names-B19-{x for x in names if x.startswith("regime_speed_resid_")})
    core4=sha(json.dumps(d649,ensure_ascii=True,separators=(",",":")).encode())
    schema=sha("\n".join(sorted(names)).encode())
    need(core4==CORE4_SHA,"ORIGINAL_CORE4_649_NAME_DIGEST_NOT_EQUAL")
    need(schema==FEATURE_SHA,"ORIGINAL_690_FULL_SCHEMA_DIGEST_NOT_EQUAL")
    output_doc={
        "contract":"KEIBA_2026_1017_ORIGINAL_690_PRIOR_ONLY_LINEAGE_READONLY_AUDIT_V1",
        "races":36,"horses":471,"feature_columns":690,
        "original_core4_columns":649,"original_daybias_columns":19,
        "original_speed_regime_columns":22,
        "sealed_feature_cells":cells,
        "feature_files_sha256_verified":36,
        "horse_keys_linked_to_original_official_entries":471,
        "source_repo_sha":HEAD,"feature_repo_sha":DELIVERY,
        "source_dates":age,
        "immutable_builder_source_sha256":guards,
        "original_649_feature_name_sha256":core4,
        "original_690_feature_name_sha256":schema,
        "source_target_result_or_final_odds_used":False,
        "same_day_result_bias_nonzero_horses":nonzero_daybias,
        "history_input_source_date_strictly_prior":True,
        "original_history_generation_runtime_guards_present":True,
        "independent_numeric_690_parity_verified":False,
        "entire_historical_training_source_asof_certified":False,
        "all_model_candidate_release_approved":False,
        "frozen_l21_betting_cutoff_certified":False,
        "frozen_l21_engine_invoked":False,"l3_v3_invoked":False,
        "ready_for_betting":False,"race_evidence":race_audit}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(output_doc,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print("ORIGINAL_2026_PRIOR_ONLY_690_SAVED_LINEAGE_AUDIT "+
          json.dumps({"races":36,"horses":total,"cells":cells,
                      "history_last":age["source_daily_latest"],
                      "same_day_result_bias_nonzero":nonzero_daybias,
                      "core4_schema_sha_verified":True,
                      "full_numerical_parity_certified":False,
                      "betting":False},sort_keys=True))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for name in ("source","delivery","output"):
        ap.add_argument("--"+name,type=Path,required=True)
    a=ap.parse_args()
    audit(a.source,a.delivery,a.output)
