#!/usr/bin/env python3
"""Publish NEW 17-started-horse 202505050812 historical FINAL-only L3 JSON.

Immutable distinct path; never overwrite the existing saved-prediction replay.
Bounded free public Git storage only. No Actions Artifacts, cache, paid storage.
"""
from __future__ import annotations
import argparse,base64,hashlib,json,os,sys,urllib.parse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import publish_frozen_historical_races as previous

DEST="historical-new/2025/202505050812.json"
MAX_BYTES=512*1024
MAX_TOTAL_HISTORICAL_GIT_BYTES=4*1024*1024

def validate(x):
    if not isinstance(x,dict) or x.get("contract")!="KEIBA_FRESH_HISTORICAL_L1_L21_L3_MOBILE_V1":
        raise ValueError("INVALID_NEW_17_HORSE_HISTORICAL_MOBILE_CONTRACT")
    if x.get("race_id")!="202505050812" or x.get("race_date")!="2025-11-30":
        raise ValueError("MOBILE_SELECTED_RACE_ID_DRIFT")
    exact={
        "started_horse_count":17,
        "race_entry_count":18,
        "source_price_phase":"NETKEIBA_FINAL_POSTRACE",
        "is_live_or_purchase_ready":False,
        "actual_bets_placed":False,
        "fresh_original_ability_risk_used":True,
        "historical_saved_2025_oos_scores_used":False,
        "historical_saved_l21_tickets_used":False,
        "original_l2_1_policy":"STANDARD",
        "original_l2_1_standard_line_gate_count":50,
        "original_l3_v3_policy":"REVERSE_P3_RACE_MAX300",
        "l3_p3_provenance":"UNKNOWN_NO_INVENTED_P3",
        "l3_staking_formula_original":True,
    }
    for k,v in exact.items():
        if x.get(k)!=v:raise ValueError("MOBILE_RESEARCH_PROVENANCE_DRIFT_"+k)
    horses=x.get("horses") or []
    if (len(horses)!=17 or len({h["horse_id"] for h in horses})!=17 or
        len({h["horse_number"] for h in horses})!=17):
        raise ValueError("MOBILE_REAL_STARTED_17_HORSE_POPULATION_DRIFT")
    rank=x.get("rankings") or {}
    if (set(rank)!={"ai_raw","ai_probability","market","expected_value"} or
        any(len(rows)!=17 or
            set(h["horse_id"] for h in rows)!=set(h["horse_id"] for h in horses)
            for rows in rank.values())):
        raise ValueError("MOBILE_FOUR_RANKINGS_17_HORSE_DRIFT")
    tickets=x.get("tickets")
    if not isinstance(tickets,list) or len(tickets)!=x.get("bet_count"):
        raise ValueError("MOBILE_L21_L3_TICKET_COUNT_DRIFT")
    if (not all(h.get("min_p3") is None and h.get("p3_status")==
                "ORIGINAL_L17_P3_NOT_AVAILABLE" for h in tickets) or
        not all(h.get("stake_yen") in (100,200,300) for h in tickets) or
        sum(h.get("stake_yen",0) for h in tickets)!=x.get("l3_v3_paper_stake_yen")):
        raise ValueError("MOBILE_OFFICIAL_L3_V3_ALLOCATION_OR_P3_DRIFT")
    return (json.dumps(x,sort_keys=True,ensure_ascii=False,
                       separators=(",",":"),allow_nan=False)+"\n").encode("utf-8")

def publish(raw,*,dry_run=False):
    if len(raw)>MAX_BYTES:raise ValueError("MOBILE_JSON_EXCEEDS_FREE_GIT_SMALL_FILE_CAP")
    if dry_run:
        print("NEW_17_HORSE_MOBILE_RESEARCH_JSON_DRY_RUN",DEST,len(raw))
        return
    client=previous.GithubContents(os.environ.get("GITHUB_TOKEN"))
    current=client.existing(DEST)
    if current:
        sha=hashlib.sha1(("blob "+str(len(raw))+"\0").encode()+raw).hexdigest()
        if current.get("sha")!=sha or current.get("size")!=len(raw):
            raise ValueError("IMMUTABLE_REAL_HISTORICAL_NEW_L1_L3_RECORD_ALREADY_DIFFERS")
        print("IMMUTABLE_HISTORICAL_NEW_RECORD_ALREADY_MATCHES",DEST)
        return
    branch=client.request("GET","git/ref/heads/"+previous.DEST_BRANCH)
    tree=client.request("GET","git/trees/"+branch["object"]["sha"]+"?recursive=1")
    if tree.get("truncated") is not False:
        raise ValueError("FREE_GIT_HISTORICAL_TREE_TRUNCATED_STOP")
    size=sum(int(x.get("size") or 0) for x in tree["tree"]
             if x.get("type")=="blob" and
             str(x.get("path","")).startswith(("historical/","historical-new/")))
    if size+len(raw)>MAX_TOTAL_HISTORICAL_GIT_BYTES:
        raise ValueError("FREE_GIT_HISTORICAL_TOTAL_4MB_GUARD_STOP")
    args={
       "message":"data(historical): publish new blind original 17-horse L1 L2.1 V3 202505050812 research result",
       "branch":previous.DEST_BRANCH,
       "content":base64.b64encode(raw).decode("ascii")}
    uri="contents/"+urllib.parse.quote(DEST,safe="/")
    result=client.request("PUT",uri,args)
    print("PUBLISHED_NEW_17_HORSE_HISTORICAL_MOBILE_JSON",
          DEST,len(raw),hashlib.sha256(raw).hexdigest(),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    data=json.loads(Path(a.input).read_text(encoding="utf-8"))
    publish(validate(data),dry_run=a.dry_run)

if __name__=="__main__":main()
