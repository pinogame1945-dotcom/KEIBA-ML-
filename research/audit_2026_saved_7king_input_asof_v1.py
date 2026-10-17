#!/usr/bin/env python3
"""Read-only 2026 seven-king as-of lineage audit of pinned saved inputs.

NEVER approves model/input for betting. Separates independent observation of
live 2026-10-17 raw card source timing from missing provenance of intermediate
CORE4/690+238 features. 2026-10-10 is a historical FINAL projection without
a verified as-of capture; cannot be grandfathered into pre-race status.

No feature recompute, Kaggle publish, model fit, calibration, rerank, or label read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import zipfile

EXPECTED={"2026-10-10":(24,315),"2026-10-17":(36,471)}
SOURCE_CUTOFF=datetime.fromisoformat("2026-10-16T15:00:00+00:00")
KINGS=("core4","pedlegacy","condition","full","light","pedv1","condrc")
CANDIDATE_CONTRACT="KEIBA_2026_UNRELEASED_SEVENKING_MODEL_SCORE_HANDOFF_V1"
EXPECTED_KAGGLE="pino1945/keiba-ml-2026-original-sevenking-candidate-v1"
MODEL_GENERATION="eb13d4096519167b"
BACKFILL_HASH="9fe9f1742aa074a84c03a779e1ee949c04f78c0d"
FORBIDDEN_SOURCE_KEYS={"results","payouts","final_odds"}
HOUR=timedelta(hours=1)

def require(ok,reason):
    if not ok:raise ValueError(reason)

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def stamp(value,reason):
    require(isinstance(value,str) and value,reason+"_ABSENT")
    dt=datetime.fromisoformat(value.replace("Z","+00:00"))
    require(dt.tzinfo is not None,reason+"_NAIVE")
    return dt.astimezone(timezone.utc)

def observed_prior(value,limit,reason):
    dt=stamp(value,reason)
    require(dt<limit,reason+"_NOT_BEFORE_CAPTURE_CUTOFF")
    return dt.isoformat()

def check_690(root):
    index=read(root/"ml_features/delivery/690/index.json")
    require(index.get("contract")=="KEIBA_BACKFILL_TO_ML_690_INDEX_V1" and
            index.get("feature_count")==690 and
            index.get("total_races")==60 and index.get("total_horses")==786 and
            index.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET",
            "ORIGINAL_690_INDEX_SCOPE_OR_READINESS_DRIFT")
    entries=index["races"]
    require(isinstance(entries,dict) and len(entries)==60,
            "ORIGINAL_690_INDEX_NOT_60")
    counts={d:[0,0] for d in EXPECTED}
    for rid,entry in sorted(entries.items()):
        day=entry.get("date")
        require(day in EXPECTED and rid.isdigit() and len(rid)==12 and
                rid[:4]=="2026" and str(entry.get("path","")).startswith(
                    "ml_features/delivery/690/"+day+"/"),
                "ORIGINAL_690_INDEX_BAD_RACE:"+rid)
        path=root/entry["path"]
        require(path.is_file() and digest(path)==entry.get("sha256"),
                "ORIGINAL_690_HASH_MISMATCH:"+rid)
        doc=read(path)
        require(doc.get("race_id")==rid and doc.get("race_date")==day and
                doc.get("feature_count")==690 and doc.get("starters")==entry.get("starters") and
                doc.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET" and
                doc.get("used_target_results_or_final_odds") is False,
                "ORIGINAL_690_METADATA_ASOF_DRIFT:"+rid)
        rows=doc.get("rows")
        require(isinstance(rows,list) and len(rows)==doc["starters"],
                "ORIGINAL_690_HORSE_ROWS_DRIFT:"+rid)
        ids=set()
        for row in rows:
            hid=str(row.get("_horse_id") or "")
            require(hid.isdigit() and hid not in ids and row.get("_race_id")==rid
                    and row.get("_race_date")==day,
                    "ORIGINAL_690_RACE_HORSE_MISMATCH:"+rid)
            ids.add(hid)
        counts[day][0]+=1
        counts[day][1]+=len(rows)
    require(all(tuple(counts[d])==EXPECTED[d] for d in EXPECTED),
            "ORIGINAL_690_60_786_COVERAGE_DRIFT")
    return {day:{"races":rs,"horses":hs,"cached_690_sha256_verified":True,
                 "original_690_asof_certified":False} for day,(rs,hs) in counts.items()}

def check_live(root):
    live=root/"data/live/2026-10-17"
    manifest=read(live/"manifest.json")
    require(manifest.get("contract")=="JRA_20261017_ALL_RACES_SOURCE_MANIFEST_V1"
            and manifest.get("day")=="2026-10-17"
            and manifest.get("actual_races_processed")==36,
            "LIVE_20261017_MANIFEST_DRIFT")
    manifest_at=observed_prior(manifest.get("generated_at_utc"),
                               SOURCE_CUTOFF,"LIVE_MANIFEST")
    races=manifest.get("races") or []
    require(len(races)==36 and len({str(e.get("race_id")) for e in races})==36,
            "LIVE_SOURCE_36_RACES_NOT_PRESENT")
    hours=[]
    counts={"races":0,"raw_captured_before_target_date":0,
            "raw_results_excluded":0,"netkeiba_prestart_observed":0,
            "jra_pages_prestart_observed":0}
    for item in races:
        rid=str(item.get("race_id") or "")
        require(rid.startswith("2026") and item.get("status")=="SOURCE_AND_ODDS_VERIFIED",
                "LIVE_RACE_ID_OR_SOURCE_STATUS_DRIFT:"+rid)
        doc=read(live/(rid+".json"))
        require(doc.get("race_id")==rid and doc.get("race_date")=="2026-10-17"
                and doc.get("includes_results") is False
                and doc.get("includes_final_odds") is False
                and doc.get("models_executed") is False
                and doc.get("backfill_history_features_computed") is False,
                "LIVE_RAW_CARD_NOT_PRE_RACE_ONLY:"+rid)
        hours.append(observed_prior(doc.get("captured_at_utc"),SOURCE_CUTOFF,
                                    "LIVE_CAPTURE:"+rid))
        counts["races"]+=1
        counts["raw_captured_before_target_date"]+=1
        counts["raw_results_excluded"]+=1
        net=(doc.get("netkeiba") or {}).get("meta") or {}
        observed_prior(net.get("fetched_at_utc"),SOURCE_CUTOFF,
                       "NETKEIBA_FETCH:"+rid)
        counts["netkeiba_prestart_observed"]+=1
        odds=doc.get("jra_odds") or {}
        require(odds.get("snapshot_type")=="PRE_RACE_LIVE_ODDS" and
                odds.get("combination_coverage_complete") is True and
                odds.get("publisher_update_display_consistent") is True,
                "LIVE_JRA_PAGE_SOURCE_NOT_PRESTART:"+rid)
        pages=odds.get("page_observations") or {}
        require(len(pages)>0,"LIVE_JRA_PAGE_OBSERVATIONS_MISSING:"+rid)
        for bet,details in pages.items():
            observed_prior(details.get("fetched_at_utc"),SOURCE_CUTOFF,
                           "JRA_PAGE_FETCH:"+rid+":"+str(bet))
            counts["jra_pages_prestart_observed"]+=1
    return {
      "source_observation_manifest_utc":manifest_at,
      "latest_raw_capture_utc":max(hours),
      "raw_live_source_timing_verified":counts["races"]==36,
      "counts":counts,
      "raw_source_data_has_no_target_results":True,
      "derived_690_and_238_asof_certified":False,
      "all_928_features_reconstructed_from_timestamped_source":False,
    }

def check_saved(saved):
    """Kaggle unpacks the original .zip and .jsonl.gz on ingest. Audit REAL
    remote files rather than claiming the archived ZIP itself exists remotely."""
    manifest=read(saved/"manifest.json")
    handoff=read(saved/"index.json")
    require(manifest.get("contract")=="KEIBA_L15_7KING_STAGE_V1"
            and manifest.get("ref")==EXPECTED_KAGGLE,
            "SAVED_KAGGLE_MANIFEST_PROVENANCE_DRIFT")
    require(handoff.get("contract")==CANDIDATE_CONTRACT
            and handoff.get("snapshot_generation")==MODEL_GENERATION
            and handoff.get("training_years")==[2024,2025]
            and handoff.get("prediction_year")==2026
            and handoff.get("races")==60 and handoff.get("horse_rows")==786
            and handoff.get("kings")==7
            and handoff.get("ready_for_betting") is False
            and handoff.get("asof_audited") is False
            and handoff.get("model_status")=="CANDIDATE_2026_BLIND_UNRELEASED",
            "SAVED_2026_MODEL_HANDOFF_NOT_UNRELEASED")
    listed=manifest.get("files") or []
    require({p.get("name") for p in listed}=={handoff["bundle"],"index.json"},
            "SAVED_KAGGLE_MANIFEST_EXPECTED_ORIGINAL_BUNDLE_MISMATCH")
    meta_index=next(item for item in listed if item["name"]=="index.json")
    require((saved/"index.json").stat().st_size==meta_index.get("bytes")
            and digest(saved/"index.json")==meta_index.get("sha256"),
            "SAVED_KAGGLE_REMOTE_INDEX_HASH_MISMATCH")
    # The Kaggle storage layer expands both ZIP and GZIP. The saved archive
    # SHA256 cannot be rehashed from the expanded entries and must NOT be
    # reported as verified. Check the actual inner files and row contracts.
    expanded=saved/"2026-sevenking-unreleased-candidate"
    require(expanded.is_dir(),"SAVED_KAGGLE_EXPANDED_BUNDLE_NOT_PRESENT")
    refit=read(expanded/"models/refit-index.json")
    summary=read(expanded/"scores/index.json")
    require(refit.get("models")==7 and refit.get("released") is False
            and refit.get("training_years")==[2024,2025]
            and summary.get("research_only") is True
            and summary.get("ready_for_betting") is False
            and summary.get("source_input_asof_independently_verified") is False
            and summary.get("races")==60 and summary.get("horse_rows")==786
            and summary.get("king_count")==7,
            "SAVED_2026_CANDIDATE_INTERNAL_SCOPE_DRIFT")
    for alias in KINGS:
        folder=expanded/"models"/alias
        model=folder/"model.txt"
        schema=folder/"schema.json"
        meta=read(folder/"model.json")
        require(meta.get("deployment_status")=="CANDIDATE_2026_BLIND_UNRELEASED"
                and meta.get("training_years")==[2024,2025]
                and meta.get("2026_outcomes_used") is False
                and meta.get("training_rule_approval_ref") is None,
                "SAVED_2026_CANDIDATE_SELF_APPROVAL:"+alias)
        require(model.is_file() and digest(model)==meta.get("model_sha256"),
                "SAVED_MODEL_WEIGHT_HASH_MISMATCH:"+alias)
        require(schema.is_file() and digest(schema)==meta.get("schema_sha256"),
                "SAVED_MODEL_SCHEMA_HASH_MISMATCH:"+alias)
        score=expanded/"scores"/(alias+".jsonl")
        require(score.is_file(),"SAVED_CANDIDATE_SCORE_MISSING:"+alias)
        rows=0
        race_horses={}
        with score.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():continue
                record=json.loads(line)
                require(record.get("contract")=="L1_TO_L2_OUTPUT_CONTRACT_V1"
                        and str(record.get("race_id") or "").startswith("2026")
                        and record.get("model_sha256")==meta.get("model_sha256"),
                        "SAVED_CANDIDATE_SCORE_PROVENANCE_MISMATCH:"+alias)
                rid=record["race_id"]
                race_horses.setdefault(rid,set())
                hid=str(record.get("horse_id") or "")
                require(hid.isdigit() and hid not in race_horses[rid],
                        "SAVED_CANDIDATE_SCORE_DUPLICATED_HORSE:"+alias)
                race_horses[rid].add(hid)
                rows+=1
        require(rows==786 and len(race_horses)==60,
                "SAVED_CANDIDATE_SCORE_60_786_DRIFT:"+alias)
    council=expanded/"council2026.jsonl"
    require(council.is_file(),"SAVED_CANDIDATE_L16_COUNCIL_MISSING")
    rows=0
    races=set()
    with council.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():continue
            record=json.loads(line)
            rid=str(record.get("race_id") or "")
            require(record.get("contract")=="L16_FULL_RANK_CONSENSUS_V1"
                    and rid.startswith("2026") and rid not in races
                    and record.get("expert_count")==7,
                    "SAVED_CANDIDATE_CONSENSUS_DRIFT")
            races.add(rid)
            horses=record.get("horses") or []
            require(record.get("field_size")==len(horses)
                    and [int(h["consensus_rank"]) for h in horses]==list(range(1,len(horses)+1)),
                    "SAVED_CANDIDATE_CONSENSUS_RANKS_DRIFT")
            rows+=len(horses)
    require(len(races)==60 and rows==786,"SAVED_CANDIDATE_COUNCIL_60_786_DRIFT")
    return {
      "remote_index_manifest_sha256_checked":True,
      "original_zip_remote_bytes_checked":False,
      "zip_auto_extracted_on_Kaggle_ingest":True,
      "expanded_seven_model_weight_hashes_checked":7,
      "expanded_seven_model_schema_hashes_checked":7,
      "expanded_score_files_with_786_rows_checked":7,
      "expanded_council_races":60,"expanded_council_horse_rows":786,
      "model_release_status":"CANDIDATE_2026_BLIND_UNRELEASED",
      "training_years":[2024,2025],
      "target_outcomes_used":False,
      "asof_independently_verified":False,
    }

def audit(backfill,saved,report):
    raw=check_live(backfill)
    per_day=check_690(backfill)
    persisted=check_saved(saved)
    out={
      "contract":"KEIBA_2026_SEVEN_KING_INDEPENDENT_SOURCE_ASOF_EVIDENCE_V1",
      "backfill_git_commit":BACKFILL_HASH,
      "historical_snapshot_date_2026_10_10_prestart_proven":False,
      "day_2026_10_17_live_source_prestart_proven":raw["raw_live_source_timing_verified"],
      "day_2026_10_17_690_238_feature_reconstruction_asof_proven":False,
      "daily_690":per_day,
      "live_source_2026_10_17":raw,
      "persisted_models":persisted,
      "original_690_numeric_parity_recomputed":False,
      "target_2026_outcomes_or_payouts_evaluated":False,
      "feature_matrix_928_independent_asof_verified":False,
      "formal_model_release_authorized":False,
      "ready_for_betting":False,
      "ready_for_l175_l21_l3_production":False,
      "blockers":[
          "2026-10-10 source is historical FINAL projection with no independent before-start timestamp",
          "2026-10-17 live page capture is before date, but 649 feature derivation and 238 pedigree lineage are not independently source-time certified",
          "2026 model is an unreleased refit candidate, not an approved L1 production release",
      ],
    }
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SEVEN_KING_2026_SOURCE_TIME_EVIDENCE_AUDIT "+json.dumps({
        "status":"PARTIAL_SOURCE_ONLY_NOT_MODEL_READY",
        "races":60,"horse_rows":786,
        "live_20261017_raw_races":raw["counts"]["races"],
        "20261010_asof_proven":False,
        "20261017_derived_features_asof_proven":False,
        "ready_for_betting":False,
        "report":str(report)},sort_keys=True),flush=True)
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--backfill-root",type=Path,required=True)
    p.add_argument("--saved-candidate",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    args=p.parse_args()
    audit(args.backfill_root,args.saved_candidate,args.report)
