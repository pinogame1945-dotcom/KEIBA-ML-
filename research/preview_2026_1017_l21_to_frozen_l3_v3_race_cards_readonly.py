#!/usr/bin/env python3
"""Read-only race-by-race 2026 historical L2.1 tickets -> frozen V3 stake ordering.

Genuine original L3 V3 reverse_p3_stakes is directly invoked from its immutable
2026-10-16 adopted source. This is a HISTORICAL RESEARCH PREVIEW ONLY, not
official operational run(): official V3 explicitly blocks a historical 2026
race from masquerading as a current prebet quote.
For each race, budget=200 yen per L2.1 physical ticket, V3 min100/max300.
Same-day cash is UNKNOWN, so funding/placement readiness is NOT attested.
L1.75 p3 is used AS STORED: null remains null, never inferred from win
probability or ranking. No 2026-10-17 results, payouts, hit/ROI evaluation.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

DAY="2026-10-17"
ORIGINAL_V3_BLOB="71ea06e8a4d1a9694fc3a113c3c376a6b0bb4289"
FROZEN_V3="REVERSE_P3_RACE_MAX300"
SOURCE_CONTRACT="KEIBA_ORIGINAL_L21_FROZEN_STANDARD_50_2026_1017_RESEARCH_REPLAY_V1"
MARKET_INDEX="KEIBA_L175_BACKFILL_2026_1017_RESEARCH_PRIVATE_V1"
MARKET_RACE="KEIBA_2026_1017_L175_MARKET_JOIN_RESEARCH_V1"
EXPECTED_2026_RACES=36
EXPECTED_2026_HORSES=471
EXPECTED_TICKETS=38
MAX_LIMIT=300

def require(cond,msg):
    if not cond:raise ValueError(msg)

def git_blob(path):
    data=Path(path).read_bytes()
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\x00"+data).hexdigest()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def frozen_stakes(v3_source):
    require(git_blob(v3_source)==ORIGINAL_V3_BLOB,"OFFICIAL_L3_V3_SOURCE_SHA_CHANGED")
    parsed=ast.parse(v3_source.read_text(encoding="utf-8"),filename=str(v3_source))
    selected=[f for f in parsed.body
              if isinstance(f,ast.FunctionDef) and f.name in
              ("yen","ticket_key","reverse_p3_stakes")]
    require({f.name for f in selected}=={"yen","ticket_key","reverse_p3_stakes"},
            "ORIGINAL_REVERSE_P3_ALLOCATOR_FUNCTIONS_MISSING")
    scope={}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(v3_source),"exec"),scope)
    return scope["reverse_p3_stakes"]

def saved_2026_market(index_path,scores_path):
    i=json.loads(index_path.read_text(encoding="utf-8"))
    require(i.get("contract")==MARKET_INDEX and i.get("races")==36 and
            i.get("horses")==471 and i.get("ready_for_betting") is False and
            i.get("2026_target_results_used") is False and
            i.get("market_asof_independently_verified") is False and
            i.get("files",{}).get(scores_path.name)==digest(scores_path),
            "SAVED_L175_CANDIDATE_INDEX_DRIFT")
    result={};total=nonnull=0
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():continue
        r=json.loads(line)
        rid=str(r.get("race_id") or "")
        horses=r.get("horses") or []
        require(rid not in result and r.get("contract")==MARKET_RACE and
                r.get("race_date")==DAY and r.get("horse_count")==len(horses) and
                r.get("ready_for_betting") is False and
                r.get("market_asof_independently_verified") is False and
                r.get("2026_target_results_used") is False,
                "SAVED_L175_REAL_RACE_OR_SOURCE_DRIFT:"+rid)
        by_no={}
        for h in horses:
            no=h.get("horse_number")
            p=h.get("p3_calibrated")
            require(type(no)is int and 1<=no<=18 and no not in by_no and
                    (p is None or (type(p)in (int,float) and
                     math.isfinite(p) and 0<=p<=1)),
                    "INVALID_SAVED_ORIGINAL_L175_P3:"+rid)
            by_no[no]=p
            total+=1
            nonnull+=p is not None
        result[rid]=by_no
    require(len(result)==EXPECTED_2026_RACES and
            total==EXPECTED_2026_HORSES and nonnull==347,
            "L175_ORIGINAL_UNCHANGED_2026_P3_COVERAGE_DRIFT")
    return result

def perform(replay,market_index,market_scores,v3_source,out):
    require(not out.exists(),"DO_NOT_OVERWRITE_PREVIOUS_V3_PREVIEW")
    original_stakes=frozen_stakes(v3_source)
    l2=json.loads(replay.read_text(encoding="utf-8"))
    require(l2.get("contract")==SOURCE_CONTRACT and l2.get("day")==DAY and
            l2.get("frozen_policy")=="STANDARD" and
            l2.get("frozen_standard_rule_count")==50 and
            l2.get("frozen_strict_subset_rule_count")==26 and
            l2.get("races")==36 and l2.get("horses")==471 and
            l2.get("total_standard_unique_physical_tickets")==38 and
            l2.get("payout_source_sealed_empty") is True and
            l2.get("target_2026_results_or_payouts_read") is False and
            l2.get("model_release_approved") is False and
            l2.get("ready_for_betting") is False,
            "REAL_L21_38_NO_RESULTS_AND_RESEARCH_GUARDS_REQUIRED")
    market=saved_2026_market(market_index,market_scores)
    cards=[];by_bet=Counter();stats=Counter();ids=set();count=0;sum_stake=0
    for race in l2["research_race_outputs"]:
        rid=race["race_id"];horses=market.get(rid)
        require(horses is not None and rid not in ids and
                race["horse_count"]==len(horses) and
                race["original_frozen_standard_rules"]==50 and
                race["ready_for_betting"] is False and
                race["research_only"] is True,
                "L21_RACE_SCOPE_SOURCE_NOT_AUTHENTIC:"+rid)
        ids.add(rid)
        enriched=[]
        for t in race["tickets"]:
            bt=t["bet_type"]
            ns=t["selection"]
            require(bt in ("QUINELLA","TRIO","TRIFECTA") and
                    isinstance(ns,list) and len(ns)==(2 if bt=="QUINELLA" else 3)
                    and all(type(n)is int and n in horses for n in ns)
                    and len(set(ns))==len(ns)
                    and (bt=="TRIFECTA" or ns==sorted(ns)) and
                    t["baseline_yen"]==100 and t["ready_for_betting"] is False and
                    isinstance(t["line_gates"],list) and len(t["line_gates"])>0,
                    "FROZEN_L21_TICKET_DATA_DRIFT:"+rid)
            odds=t["odds"]
            require(type(odds)in (int,float) and math.isfinite(odds) and odds>=1,
                    "INVALID_SAVED_TICKET_ODDS")
            p3=[horses[n] for n in ns]
            minp3=min(p3) if all(x is not None for x in p3) else None
            adapted={"bet_type":bt,"canonical_selection":"-".join(map(str,ns))}
            enriched.append((adapted,minp3))
        n=len(enriched)
        budget=200*n  # frozen V3 target, no bankroll fabricated
        require(0<=n<=3 and budget<=600,
                "L21_UNEXPECTED_RACE_TICKET_COUNT_OR_BUDGET")
        assignment=original_stakes(enriched,budget) if n else {}
        preview=[]
        for raw,(tag,minp3) in zip(race["tickets"],enriched):
            key=(tag["bet_type"],tag["canonical_selection"])
            stake=assignment[key]
            require(stake in (100,200,300),
                    "FROZEN_V3_STAKE_NOT_VALID")
            preview.append({
                "bet_type":raw["bet_type"],
                "selection":raw["selection"],
                "decimal_odds":raw["odds"],
                "original_L21_odds_band":raw["odds_band"],
                "stake_yen":stake,
                "stake_source":"PINNED_REVERSE_P3_RACE_MAX300",
                "p3_min":minp3,"p3_missing":minp3 is None,
                "strict_diagnostic_only":raw["strict"],
                "L21_line_gates":raw["line_gates"],
                "result":"NOT_EVALUATED_2026_SEALED",
                "hit":None,"payout_yen":None,
                "ready_for_betting":False})
            by_bet[raw["bet_type"]]+=1
            stats["p3_missing_tickets"]+=int(minp3 is None)
            count+=1
            sum_stake+=stake
        require(len(preview)==n and sum(t["stake_yen"] for t in preview)==budget
                and len({(t["bet_type"],tuple(t["selection"])) for t in preview})==n,
                "OFFICIAL_V3_RESEARCH_ALLOCATION_CONSERVATION_OR_DROPPED_TICKET")
        cards.append({"race_id":rid,"original_anchor_horse_number":
                    race["original_anchor_horse_number"],
                    "horse_count":race["horse_count"],
                    "L21_tickets":n,"L3_preview_budget_yen":budget,
                    "L3_preview_stake_total_yen":sum(t["stake_yen"] for t in preview),
                    "bankroll_assessment":"NOT_CHECKED_AVAILABLE_CASH_NOT_GIVEN",
                    "true_official_operational_run_invoked":False,
                    "historical_2026_quotes_cannot_pass_real_prebet_cutoff_gate":True,
                    "betting":False,"tickets":preview})
        print("L3_V3_RACE_CARD "+json.dumps({
            "race_id":rid,"anchor":race["original_anchor_horse_number"],
            "ticket_count":n,"stake_total_yen":budget,
            "tickets":[{"bet_type":t["bet_type"],"selection":t["selection"],
                        "odds":t["decimal_odds"],"yen":t["stake_yen"],
                        "p3_min":t["p3_min"],"hit":None} for t in preview]
        },ensure_ascii=False,sort_keys=True),flush=True)
    require(len(ids)==36 and count==38 and sum_stake==7600 and
            sum(x["L21_tickets"] for x in cards)==38 and
            all(x["L3_preview_stake_total_yen"]==x["L3_preview_budget_yen"] for x in cards),
            "OFFICIAL_L3_V3_RESEARCH_PREVIEW_NOT_EXACT_36R_38T_7600Y")
    out.parent.mkdir(parents=True,exist_ok=True)
    result={"contract":"KEIBA_2026_1017_L21_38_TO_FROZEN_V3_RESEARCH_PREVIEW_V1",
        "day":DAY,"races":36,"horses":471,"tickets":count,
        "frozen_l21_standard_rules":50,"frozen_l21_buy_set_unchanged":True,
        "l3_frozen_policy":FROZEN_V3,
        "official_v3_reverse_p3_stakes_function_directly_invoked":True,
        "official_v3_run_invoked":False,
        "v3_budget_formula":"200YEN_PER_L21_TICKET",
        "total_provisional_stake_yen":sum_stake,
        "per_race_max_exposure_yen":MAX_LIMIT,
        "funding_check":"UNVERIFIED_USER_AVAILABLE_CASH_NOT_PROVIDED",
        "bankroll_pre_funding_v3_applied":False,
        "2026_target_outcomes_used":False,"hit_or_roi_evaluated":False,
        "real_bet_cutoff_price_certified":False,
        "model_release_approved":False,
        "ready_for_betting":False,"actual_bets_placed":False,
        "bet_types":dict(by_bet),"p3_missing_ticket_count":stats["p3_missing_tickets"],
        "per_race":cards}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print("L3_V3_RESEARCH_PREVIEW_SUMMARY "+json.dumps({
        "races":36,"l21_tickets_unchanged":count,"total_stakes_yen":sum_stake,
        "p3_missing_tickets":stats["p3_missing_tickets"],
        "official_V3_run":False,"funding_verified":False,
        "hit_2026_evaluated":False},ensure_ascii=False,sort_keys=True),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    for s in ("l21-replay","l175-index","l175-scores","frozen-v3-source","output"):
        a.add_argument("--"+s,type=Path,required=True)
    x=a.parse_args()
    perform(x.l21_replay,x.l175_index,x.l175_scores,x.frozen_v3_source,x.output)
