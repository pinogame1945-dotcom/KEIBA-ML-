#!/usr/bin/env python3
"""TWO ports, one race-ID interface, existing frozen models/rules only.

historical: original saved ANCHOR_FULL OOS + official archived FINAL price ->
            frozen L2.1 historical replay + non-tradable L3 flat100 audit.
today:      already collected KEIBA-MOBILE LIVE JRA odds + genuine original
            as-of ability/risk -> unchanged prebet L2.1 -> unchanged L3.
Neither port scrapes netkeiba. Never promote FINAL prices to LIVE.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO="pinogame1945-dotcom/KEIBA-ML-"
FROZEN="freeze/l2.1-standard50-to-l3-20261015"
BACKFILL="pinogame1945-dotcom/KEIBA-BACKFILL"
BACKFILL_COMMIT="03950cfe132abeca15ef17dbc7c6885ceb1591b7"
ARCHIVE_ROOT="research-results"
SOURCE_FILES={
    "predictions":(FROZEN,f"{ARCHIVE_ROOT}/market-fixed-anchor-residual-v2/run-38659856085/oos-fair-value-predictions.csv.gz",11_000_000),
    "ability":(FROZEN,"research-cache/market-fixed-anchor-residual-v2/fair-base-bf811fa2eab73db0.csv.gz",20_000_000),
    "picks":(FROZEN,f"{ARCHIVE_ROOT}/all-race-one-pick-v1/run-38668174639/all-race-one-picks.csv.gz",1_500_000),
}
PATTERN=re.compile(r"202[2-6]\d{8}")

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj
    spec.loader.exec_module(obj)
    return obj

# Existing, unmodified research and L3 engines.
HISTORY=load("research/l21-replay-real-final-one-race.py","exact_frozen_historical_replay")
L3=load("research/l3-one-race-money.py","existing_flat100_l3")
PAPER=load("research/run-one-race-paper-pipeline.py","existing_live_l21_l3")
FETCH=load("scripts/github-readonly-archive.py","metadata_verified_github_archive")

def parse_ids(args):
    text=" ".join(args.race_ids or [])
    if args.ids_file:
        text+="\n"+"\n".join(s.split("#",1)[0] for s in Path(args.ids_file).read_text(encoding="utf-8").splitlines())
    found=[]
    for token in re.split(r"[\s,;]+",text.strip()):
        if not token: continue
        if not PATTERN.fullmatch(token):
            raise ValueError("INVALID_12_DIGIT_RACE_ID_OR_YEAR_"+token[:20])
        if token not in found:found.append(token)
    if not 1<=len(found)<=60:
        raise ValueError("BATCH_RACE_ID_COUNT_MUST_BE_1_TO_60")
    return found

def archive_sources(session):
    paths={}
    for name,(ref,path,max_bytes) in SOURCE_FILES.items():
        dest=session/"verified-original"/name/Path(path).name
        FETCH.obtain(REPO,ref,path,dest,max_bytes)
        paths[name]=dest
    return paths

def archived_dates(pred, ids):
    selected={rid:set() for rid in ids}
    wanted=set(ids)
    with gzip.open(pred,"rt",encoding="utf-8",newline="") as file:
        reader=csv.DictReader(file)
        if "_race_date" not in (reader.fieldnames or []) or "_race_id" not in (reader.fieldnames or []):
            raise ValueError("ORIGINAL_OOS_NO_RACE_DATE_INDEX")
        for row in reader:
            rid=str(row.get("_race_id") or "")
            if rid in wanted:
                selected[rid].add(str(row.get("_race_date") or "")[:10])
    result={}
    for rid,days in selected.items():
        if len(days)!=1:continue
        d=next(iter(days))
        try:date=datetime.date.fromisoformat(d)
        except ValueError:continue
        if date.year!=int(rid[:4]):continue
        result[rid]=d
    return result

def daily_sources(session,date):
    paths={}
    for kind,remote in (
        ("race",f"data/daily/{date}.jsonl.gz"),
        ("odds",f"data/odds/daily/{date}.jsonl.gz")):
        out=session/"official-backfill"/remote
        FETCH.obtain(BACKFILL,BACKFILL_COMMIT,remote,out,35_000_000)
        paths[kind]=out
    return paths

def limits_file(value):
    p=Path(value).resolve()
    d=json.loads(p.read_text(encoding="utf-8"))
    if d.get("contract")!="KEIBA_L3_FIXED100_LIMITS_V1":
        raise ValueError("INVALID_OR_ABSENT_FROZEN_L3_LIMITS")
    return d

def historical_l3_audit(replayed,config):
    """Read-only historical cost math. NOT L3.allocate(): FINAL odds are banned there."""
    if replayed["source_odds_kind"]!="FINAL" or replayed["FORBIDDEN_FOR_LIVE_BETTING"] is not True:
        raise ValueError("HISTORY_PROVENANCE_NOT_EXPLICIT_FINAL")
    ids=set()
    for x in replayed["tickets"]:
        k=(x["bet_type"],x["selection"])
        if k in ids:raise ValueError("HISTORY_DUPLICATE_FROZEN_TICKET")
        ids.add(k)
        if x["bet_type"] not in L3.TYPES:
            raise ValueError("HISTORY_UNKNOWN_FROZEN_BET_TYPE")
        # Historical replay has no opportunity to alter the frozen set.
        if not 1<=float(x["ticket_odds"])<=1_000_000:
            raise ValueError("HISTORY_NONPOSITIVE_FINAL_PRICE")
    n=len(ids)
    budget=L3.integer(config["race_budget_yen"],"RACE_BUDGET_YEN")
    cap=L3.integer(config["max_race_exposure_yen"],"MAX_RACE_EXPOSURE_YEN")
    mx=L3.integer(config["max_ticket_count"],"MAX_TICKET_COUNT",0,1000)
    if config["stake_policy"]!="BASELINE_FLAT_100_ALL_OR_NONE" or budget%100 or cap%100:
        raise ValueError("FROZEN_L3_LIMITS_DRIFT")
    required=100*n
    if n>mx:status="BLOCKED_TICKET_COUNT_CAP_DO_NOT_PRUNE"
    elif required>budget:status="INSUFFICIENT_BUDGET_DO_NOT_DROP_TICKETS"
    elif required>cap:status="BLOCKED_RACE_EXPOSURE_LIMIT_DO_NOT_DROP_TICKETS"
    else:status="HISTORICAL_FINAL_FLAT100_AUDIT_ONLY"
    buy=status=="HISTORICAL_FINAL_FLAT100_AUDIT_ONLY"
    return {
        "contract":"KEIBA_L3_HISTORICAL_FINAL_COST_AUDIT_V1",
        "race_id":replayed["race_id"],"status":status,
        "required_ticket_count":n,"required_total_yen":required,
        "paper_ticket_count":n if buy else 0,
        "paper_stake_yen":required if buy else 0,
        "stake_policy":"BASELINE_FLAT_100_ALL_OR_NONE",
        "original_frozen_l21_ticket_set_unmodified":True,
        "historical_final_odds_used":True,
        "actual_L3_prebet_allocate_called":False,
        "is_live_prediction":False,
        "actual_bets_placed":False,
        "research_only":True,
    }

def result_top3(pack):
    """Outcome is read AFTER frozen ticket generation; no outcome ranking feed."""
    rows=(pack.get("results") or [])
    valid={}
    for row in rows:
        raw=row.get("finish_position")
        no=row.get("horse_number")
        if type(raw) is not int or type(no) is not int:continue
        if raw in (1,2,3):
            if raw in valid:raise ValueError("DUPLICATE_ACTUAL_FINISH_POSITION")
            valid[raw]=no
    return [valid.get(i) for i in (1,2,3)] if len(valid)==3 else None

def historical(ids, session, out, config):
    outcomes={rid:{"race_id":rid,"port":"historical","status":"BLOCKED_NOT_STARTED",
                    "not_verified_prebet":True,"actual_bets_placed":False} for rid in ids}
    allowed=[rid for rid in ids if rid[:4] in ("2023","2024","2025")]
    for rid in ids:
        if rid not in allowed:
            outcomes[rid]["status"]="BLOCKED_NO_ORIGINAL_HISTORICAL_OOS_YEAR"
    if not allowed:return list(outcomes.values())
    try:
        shared=archive_sources(session)
        dates=archived_dates(shared["predictions"],allowed)
    except Exception as exc:
        for rid in allowed:
            outcomes[rid]["status"]="BLOCKED_SHARED_ORIGINAL_ARCHIVE"
            outcomes[rid]["reason"]=str(exc)[:500]
        return list(outcomes.values())
    by_day={}
    for rid in allowed:
        if rid not in dates:
            outcomes[rid]["status"]="BLOCKED_NO_SAVED_ORIGINAL_ANCHOR_FOR_ID"
        else:
            by_day.setdefault(dates[rid],[]).append(rid)
    for day,group in sorted(by_day.items()):
        try: daily=daily_sources(session,day)
        except Exception as exc:
            for rid in group:
                outcomes[rid]["status"]="BLOCKED_DATED_BACKFILL_OR_ODDS"
                outcomes[rid]["reason"]=str(exc)[:500]
            continue
        for rid in group:
            try:
                # Real frozen rule implementation, genuine saved original
                # ANCHOR_FULL; do not invoke comparative seven-king proxy.
                replay=HISTORY.replay_one(rid,shared["predictions"],shared["ability"],shared["picks"],
                                          daily["race"],daily["odds"])
                if replay["retrospective_dnf_result_used_to_match_archived_population"]:
                    raise ValueError("POST_RESULT_DNF_EXCLUSION_NOT_ALLOWED")
                cost=historical_l3_audit(replay,config)
                pack=HISTORY.race_in_pack(daily["race"],rid)
                top3=result_top3(pack)
                hits=[]
                if top3:
                    for ticket in replay["tickets"]:
                        bet=ticket["bet_type"]
                        selection=list(map(int,ticket["selection"].split("-")))
                        good=(bet=="TRIFECTA" and selection==top3 or
                              bet=="TRIO" and sorted(selection)==sorted(top3) or
                              bet=="QUINELLA" and sorted(selection)==sorted(top3[:2]))
                        if good:hits.append(ticket)
                record=out/rid
                record.mkdir(parents=True,exist_ok=True)
                (record/"frozen-l21-historical-final.json").write_text(
                    json.dumps(replay,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
                (record/"l3-historical-final-flat100-audit.json").write_text(
                    json.dumps(cost,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
                outcomes[rid].update({
                    "status":cost["status"],"race_date":day,
                    "anchor_horse_number":replay["anchor_horse_number"],
                    "frozen_ticket_count":replay["ticket_count"],
                    "required_flat100_yen":cost["required_total_yen"],
                    "paper_stake_yen":cost["paper_stake_yen"],
                    "historical_winning_top3":top3,
                    "paper_winning_ticket_count":len(hits) if top3 else None,
                    "paper_estimated_return_from_final_odds_yen":(
                        round(sum(100*float(t["ticket_odds"]) for t in hits))
                        if top3 and cost["status"]=="HISTORICAL_FINAL_FLAT100_AUDIT_ONLY" else None),
                    "payout_is_officially_verified":False,
                    "original_anchor_from_prior_oos_archive":True,
                    "pricing_source":"BACKFILL_FINAL_NOT_PREBET",
                    "files":{
                        "l21":str(record/"frozen-l21-historical-final.json"),
                        "historical_l3_cost":str(record/"l3-historical-final-flat100-audit.json")
                    }
                })
            except Exception as exc:
                outcomes[rid]["status"]="BLOCKED_ORIGINAL_L21_REPLAY"
                outcomes[rid]["reason"]=str(exc)[:800]
    return [outcomes[rid] for rid in ids]

def today(ids, captures, out, limits):
    results=[]
    for rid in ids:
        folder=out/rid
        card=captures/(rid+".json")
        orig=captures/(rid+"-original-features.json")
        params=captures/"original-anchor-parameters.json"
        companion=captures/(rid+"-original-ability-risk.json")
        # Read already captured ORIGINAL market and ability sources.
        # No network access and no guesses/fallback from historical FINAL.
        arguments={
            "mobile_file":str(card) if card.is_file() else None,
            "original_file":str(orig) if orig.is_file() else None,
            "limits_file":str(limits),
            "original_parameters_file":str(params) if not orig.is_file() and params.is_file() else None,
            "original_ability_risk_file":str(companion) if not orig.is_file() and companion.is_file() else None,
        }
        try:
            # This is the actual-day port, not a second way to replay
            # past FINAL-phase market data or 2026 sealed results.
            if card.is_file():
                raw=json.loads(card.read_text(encoding="utf-8"))
                race=raw.get("race") or {}
                jst=datetime.timezone(datetime.timedelta(hours=9))
                now=datetime.datetime.now(datetime.timezone.utc)
                today_jst=now.astimezone(jst).date()
                race_date=datetime.date.fromisoformat(str(race.get("raceDate") or ""))
                begin=datetime.datetime.fromisoformat(
                    race_date.isoformat()+"T"+str(race.get("startTime"))+":00+09:00")
                if race_date!=today_jst or begin<=now:
                    raise ValueError("TODAY_PORT_REQUIRES_CURRENT_JST_DATE_AND_FUTURE_START")
                if str(race.get("canonicalRaceId"))!=rid:
                    raise ValueError("TODAY_CARD_ID_MISMATCH")
            result=PAPER.run(rid,folder,**arguments)
            row={"race_id":rid,"port":"today","status":result["status"],
                 "l21_status":result["stages"]["L2_1_TICKET"],
                 "l3_status":result["stages"]["L3_MONEY"],
                 "missing_inputs":result["required_inputs_missing"],
                 "actual_bets_placed":False,
                 "source_authenticity_independently_verified":False,
                 "files":{"pipeline":str(folder/"pipeline-status.json")}}
            if result.get("l3_output_path"):
                portfolio=json.loads((folder/result["l3_output_path"]).read_text(encoding="utf-8"))
                row.update({"paper_stake_yen":portfolio["portfolio_total_yen"],
                            "paper_ticket_count":portfolio["ticket_count"],
                            "l3_status":portfolio["status"],
                            "files":{**row["files"],"l3":str(folder/result["l3_output_path"])}})
        except Exception as exc:
            row={"race_id":rid,"port":"today","status":"BLOCKED_ORIGINAL_PREBET_CONTRACT",
                 "reason":str(exc)[:800],"actual_bets_placed":False}
        results.append(row)
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",required=True,choices=["historical","today"])
    p.add_argument("--race-ids",nargs="*",help="One or many IDs, separated by spaces or commas")
    p.add_argument("--ids-file")
    p.add_argument("--captures-dir",default="out/keiba-mobile-captures")
    p.add_argument("--limits",default="config/l3-one-race-example-fixed100-limits.json")
    p.add_argument("--session-dir",default="/tmp/keiba-race-session")
    p.add_argument("--out-dir",default="out/two-port-races")
    p.add_argument("--require-all",action="store_true")
    a=p.parse_args()
    ids=parse_ids(a)
    out=Path(a.out_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    config=limits_file(a.limits)
    if a.mode=="historical":
        if any(rid.startswith("2026") for rid in ids):
            p.error("2026_RESULTS_SEALED_HISTORICAL_REPLAY_FORBIDDEN")
        races=historical(ids,Path(a.session_dir).resolve(),out,config)
        success="HISTORICAL_FINAL_FLAT100_AUDIT_ONLY"
    else:
        races=today(ids,Path(a.captures_dir).resolve(),out,Path(a.limits).resolve())
        success="PAPER_ONLY_LIMITS_CHECKED"
    accepted=[r for r in races if r["status"]==success]
    summary={
        "contract":"KEIBA_TWO_PORT_ID_BATCH_REPORT_V1","mode":a.mode,
        "race_ids":ids,"requested":len(ids),"complete":len(accepted),
        "blocked":len(ids)-len(accepted),
        "paper_stake_yen":sum(x["paper_stake_yen"] for x in accepted),
        "true_live_betting_authorized":False,"bets_placed":False,
        "frozen_l21_and_l3_sources_modified":False,
        "historical_final_is_not_prebet":a.mode=="historical",
        "budget_note":"Default L3 limits are an EXAMPLE research budget, not an approved investment size.",
        "races":races
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for r in races:
        print("TWO_PORT_RACE",r["race_id"],a.mode,r["status"],
              "axis",r.get("anchor_horse_number"),"tickets",r.get("frozen_ticket_count",r.get("paper_ticket_count")),
              "paper_yen",r.get("paper_stake_yen"),"reason",r.get("reason",""),flush=True)
    print("TWO_PORT_BATCH",a.mode,"requested",len(ids),"complete",len(accepted),
          "blocked",len(ids)-len(accepted),"paper_yen",summary["paper_stake_yen"],flush=True)
    if a.require_all and len(accepted)!=len(ids):
        raise SystemExit(2)

if __name__=="__main__":
    main()
