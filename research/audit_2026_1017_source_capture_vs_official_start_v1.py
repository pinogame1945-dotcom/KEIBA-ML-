#!/usr/bin/env python3
"""Validate saved BACKFILL 2026-10-17 prestart source/odds capture evidence.

Separate raw race/odds page capture proof from 690 as-of numerical/provenance
certification, official betting-cutoff quotes, and unapproved research models.
Read pinned existing files ONLY. Never use 2026 outcomes/final odds/returns.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone,timedelta
from email.utils import parsedate_to_datetime
import hashlib,json,re
from pathlib import Path

DAY="2026-10-17"
DAY_START=datetime.fromisoformat(DAY+"T00:00:00+09:00").astimezone(timezone.utc)
INDEX_F="KEIBA_BACKFILL_TO_ML_690_INDEX_V1"
INDEX_O="KEIBA_BACKFILL_TO_ML_ODDS_INDEX_V1"
SOURCE="JRA_20261017_ALL_RACES_PRE_RACE_AUDIT_V1"
DELIV="KEIBA_BACKFILL_TO_ML_ODDS_V1"
FROZEN_BETS=("QUINELLA","TRIO","TRIFECTA")

def need(x,m):
    if not x:raise ValueError(m)
def data(path):return json.loads(path.read_text(encoding="utf-8"))
def checksum(path):
    h=hashlib.sha256()
    with path.open("rb") as fp:
        for buf in iter(lambda:fp.read(1048576),b""):h.update(buf)
    return h.hexdigest()
def utc(v):
    need(type(v)is str,"SOURCE_FETCHED_AT_UNDEFINED")
    a=datetime.fromisoformat(v.replace("Z","+00:00"))
    need(a.tzinfo is not None,"SOURCE_TIMEZONE_MISSING")
    return a.astimezone(timezone.utc)
def header_time(meta):
    need(type(meta)is dict and len(meta.get("sha256",""))==64 and
         re.fullmatch(r"[0-9a-f]{64}",meta["sha256"]),
         "RAW_PAGE_SHA256_SOURCE_EVIDENCE_MISSING")
    h=parsedate_to_datetime(meta["http_date"]).astimezone(timezone.utc)
    f=utc(meta["fetched_at_utc"])
    # HTTP Date has 1-second resolution. Reject incompatible page capture
    # time rather than trusting purported labels.
    need(abs((h-f).total_seconds())<=180,"PUBLISHER_HTTP_DATE_CAPTURE_DISAGREES")
    return f,h
def cutoff(meta):
    sched=meta.get("scheduled_start_time_jst")
    if sched is None:return DAY_START,"STRICT_RACE_DAY_MIDNIGHT"
    need(type(sched)is str and re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d",sched),
         "OFFICIAL_POST_TIME_BAD_FORMAT")
    return datetime.fromisoformat(DAY+"T"+sched+":00+09:00").astimezone(timezone.utc),"PUBLISHED_RACE_SCHEDULE"
def valid_before(times,limit):
    return all(x<limit for x in times)
def audit(source_root,delivery_root,output):
    need(not output.exists(),"REFUSE_CAPTURE_AUDIT_OVERWRITE")
    raw=source_root/"data/live"/DAY
    delivery=delivery_root/"ml_features/delivery"
    fidx=data(delivery/"690/index.json");oidx=data(delivery/"odds/index.json")
    need(fidx.get("contract")==INDEX_F and
         fidx.get("feature_count")==690 and fidx.get("total_races")==60 and
         fidx.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET" and
         fidx.get("original_ml_full_numerical_parity_verified") is False and
         oidx.get("contract")==INDEX_O and oidx.get("total_races")==60,
         "ORIGINAL_BACKFILL_PAIRED_INDEX_PROVENANCE_CHANGED")
    target={rid:r for rid,r in fidx["races"].items() if r["date"]==DAY}
    need(len(target)==36 and
         {rid for rid,o in oidx["races"].items() if o["date"]==DAY}==set(target),
         "2026_OCT17_REAL_RACES_NOT_36_PAIRED")
    records=[];horse_count=0;source_pages=0;types=Counter()
    for rid,info in sorted(target.items()):
        orig=raw/(rid+".json")
        feat=delivery_root/info["path"]
        oddmeta=oidx["races"][rid];odds=delivery_root/oddmeta["path"]
        need(orig.is_file() and feat.is_file() and odds.is_file() and
             checksum(feat)==info["sha256"] and
             checksum(odds)==oddmeta["sha256"] and
             oddmeta["feature_sha256"]==info["sha256"],
             "RACE_RAW_FEATURE_ODDS_SOURCE_NOT_INTEGRITY_MATCHED:"+rid)
        src=data(orig);fd=data(feat);od=data(odds)
        need(src.get("contract")==SOURCE and src.get("race_id")==rid and
             src.get("race_date")==DAY and
             src.get("status")=="SOURCE_AND_ODDS_VERIFIED" and
             src.get("includes_results") is False and
             src.get("includes_final_odds") is False and
             src.get("models_executed") is False and
             src.get("backfill_history_features_computed") is False and
             not src.get("errors") and
             fd.get("contract")=="KEIBA_BACKFILL_TO_ML_690_UNSCORED_V1" and
             fd.get("race_id")==rid and fd.get("race_date")==DAY and
             fd.get("feature_count")==690 and
             fd.get("used_target_results_or_final_odds") is False and
             fd.get("source_capture_asof_independently_verified") is False and
             fd.get("original_ml_full_numerical_parity_verified") is False and
             fd.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET",
             "ORIGINAL_SOURCE_OR_UNSCORED_690_DATA_IS_NOT_SEALED:"+rid)
        sd=src["source_discovery"]
        need(sd["race_id"]==rid and sd["race_date"]==DAY and
             sd["venue_code"]==rid[4:6] and
             sd["race_no"]==int(rid[-2:]),
             "OFFICIAL_DISCOVERY_RACE_ID_UNLINKED:"+rid)
        cutoff_at,method=cutoff(sd)
        times=[utc(src["captured_at_utc"])]
        times+=list(header_time(src["netkeiba"]["meta"]))
        times+=list(header_time(src["jra_card"]["meta"]))
        need(src["jra_card"]["identity_verified"] is True,
             "OFFICIAL_JRA_RUNNERS_NOT_IDENTITY_VERIFIED:"+rid)
        j=src["jra_odds"];o=od["odds"]
        need(j.get("snapshot_type")=="PRE_RACE_LIVE_ODDS" and
             j.get("source")=="JRA_OFFICIAL_POST" and
             j.get("publisher_update_display_consistent") is True and
             j.get("publisher_update_day_not_inferred") is True and
             j.get("combination_coverage_complete") is True and
             o.get("snapshot_type")=="PRE_RACE_LIVE_ODDS" and
             o.get("source")=="JRA_OFFICIAL_POST" and
             o.get("readiness")=="PRE_RACE_OBSERVED_ASOF_UNVERIFIED" and
             o.get("publisher_update_display_consistent") is True and
             o.get("publisher_update_day_not_inferred") is True and
             od.get("contract")==DELIV and od.get("race_id")==rid and
             od.get("race_date")==DAY and
             od.get("feature_sha256")==info["sha256"] and
             od.get("feature_reference")==info["path"],
             "OFFICIAL_ODDS_PRE_RACE_EVIDENCE_OR_DELIVERY_DRIFT:"+rid)
        need(o.get("source_captured_at_utc")==src["captured_at_utc"] and
             o.get("page_observations")==j.get("page_observations") and
             o.get("quotes")==j.get("quotes") and
             o.get("no_votes")==j.get("no_votes") and
             o.get("scratched_combinations")==j.get("scratched_combinations"),
             "HANDOFF_TICKET_QUOTES_NOT_EXACT_SOURCE_SNAPSHOT:"+rid)
        pages=j.get("page_observations") or {}
        need({"WIN_PLACE",*FROZEN_BETS}.issubset(pages) and
             all(x in pages for x in FROZEN_BETS),
             "OFFICIAL_REQUIRED_L21_ODDS_PAGE_CAPTURE_MISSING:"+rid)
        for k,v in pages.items():
            need(len(v.get("sha256",""))==64 and type(v.get("cname"))is str,
                 "JRA_ODDS_PAGE_SHA_OR_URL_MISSING:"+rid+":"+k)
            times+=list(header_time(v))
        source_pages+=len(pages)+2
        need(len(src["netkeiba"]["runners"])==src["netkeiba"]["runner_count"] and
             len(src["jra_card"]["official_horse_set"])==src["netkeiba"]["runner_count"],
             "JRA_NETKEIBA_ALL_ENTRIES_NOT_ALIGNED:"+rid)
        ours={int(h["horse_number"]):str(h["horse_id"]) for h in src["netkeiba"]["runners"]}
        names={int(h["horse_number"]):h["horse_name"] for h in src["netkeiba"]["runners"]}
        official=src["jra_card"]["official_horse_set"]
        need(len(ours)==len(names)==len(official) and
             all(int(h["horse_number"]) in names and
                 h["horse_name"]==names[int(h["horse_number"])] for h in official),
             "OFFICIAL_NETKEIBA_HORSE_ID_OR_NAME_NOT_SAME:"+rid)
        starters={ours[int(h["horse_number"])] for h in official
                  if h["entry_status"]=="LISTED"}
        rows=fd.get("rows") or []
        need(len(rows)==len(starters)==info["starters"] and
             len({str(h["_horse_id"]) for h in rows})==len(rows) and
             {str(h["_horse_id"]) for h in rows}==starters and
             all(h["_race_id"]==rid and h["_race_date"]==DAY and
                 h["horse_number"] in ours and
                 str(h["_horse_id"])==ours[h["horse_number"]] for h in rows),
             "690_STARTERS_DID_NOT_MATCH_PRE_RACE_OFFICIAL_ENTRY_SET:"+rid)
        need(not set(j.get("not_offered_bet_types") or []).intersection(FROZEN_BETS) and
             all(j["by_bet_type"][bet]["status"]=="COLLECTED"
                 for bet in FROZEN_BETS),
             "FROZEN_L21_TICKET_MARKET_NOT_OFFERED:"+rid)
        for bet in FROZEN_BETS:types[bet]+=j["by_bet_type"][bet]["quoted"]
        prior=valid_before(times,cutoff_at)
        # Distinguish proof of saved source capture from a full independently
        # proven per-feature past-history/as-of model/bet cutoff attestation.
        records.append({
            "race_id":rid,"starters":len(starters),
            "source_capture_before_official_scheduled_start":prior if method=="PUBLISHED_RACE_SCHEDULE" else None,
            "source_capture_before_race_date_midnight":prior if method=="STRICT_RACE_DAY_MIDNIGHT" else None,
            "source_cutoff_policy":method,
            "cutoff_utc":cutoff_at.isoformat(),
            "last_evidence_fetch_utc":max(times).isoformat(),
            "earliest_evidence_fetch_utc":min(times).isoformat(),
            "source_page_records_checked":len(pages)+2,
            "all_raw_pages_captured_before_evidence_cutoff":prior,
            "source_horse_ids_match_690":True,
            "same_8type_quotes_as_delivery":True,
            "any_2026_result_or_final_odds_used":False,
            "full_690_feature_asof_independently_certified":False,
            "ready_for_betting":False})
        horse_count+=len(starters)
    counts=Counter(r["source_cutoff_policy"] for r in records)
    ok=sum(r["all_raw_pages_captured_before_evidence_cutoff"] for r in records)
    need(horse_count==471 and
         sum(v["starters"] for v in records)==471 and
         len(records)==36 and
         source_pages>=36*5,
         "REAL_2026_SOURCE_CAPTURE_AUDIT_POPULATION_INVALID")
    summary={
        "contract":"KEIBA_2026_1017_RAW_SOURCE_PRESTART_CAPTURE_EVIDENCE_READONLY_V1",
        "race_date":DAY,"races":36,"horses":471,
        "raw_source_page_records_checked":source_pages,
        "observational_capture_before_evidence_cutoff_races":ok,
        "observation_after_cutoff_or_unproven_races":36-ok,
        "cutoff_policy_counts":dict(counts),
        "l21_three_bet_types_quoted_combinations":dict(types),
        "all_36_races_raw_source_feature_and_quote_integrity_checked":True,
        "full_690_feature_level_asof_independently_certified":False,
        "full_690_original_numerical_parity_certified":False,
        "2026_result_or_final_odds_used":False,
        "original_2026_model_release_approved":False,
        "live_bet_cutoff_ticket_odds_verified":False,
        "frozen_l21_engine_invoked":False,"l3_v3_invoked":False,
        "ready_for_betting":False,
        "races":records}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("RAW_2026_OFFICIAL_PRESTART_CAPTURE_SOURCE_EVIDENCE "+
          json.dumps({"races":36,"horses":471,"before_cutoff_races":ok,
                      "unproven_or_after_cutoff":36-ok,"cutoff_policies":dict(counts),
                      "raw_page_records":source_pages,
                      "l21_quoted":dict(types),"feature_asof_approved":False,
                      "betting":False},sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("source-root","delivery-root","output"):
        p.add_argument("--"+name,required=True,type=Path)
    x=p.parse_args()
    audit(x.source_root,x.delivery_root,x.output)
