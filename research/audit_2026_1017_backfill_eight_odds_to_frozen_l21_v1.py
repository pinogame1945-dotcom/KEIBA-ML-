#!/usr/bin/env python3
"""Read-only source audit of the true BACKFILL 8-bet odds for frozen L2.1.

Use the already-saved source, never regenerate features, ranks or models.
A complete combination table is not an independent bet-cutoff certification.
Never invoke frozen ticket engine/L3 or synthesize unavailable quotes.
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, math
from pathlib import Path

BETS=("WIN","PLACE","BRACKET_QUINELLA","QUINELLA","WIDE","EXACTA","TRIO","TRIFECTA")
L21_BETS=("QUINELLA","TRIO","TRIFECTA")
ODDS_CONTRACT="KEIBA_BACKFILL_TO_ML_ODDS_V1"
ODDS_INDEX="KEIBA_BACKFILL_TO_ML_ODDS_INDEX_V1"
FEATURE_INDEX="KEIBA_BACKFILL_TO_ML_690_INDEX_V1"
UPSTREAM="KEIBA_2026_1017_ORIGINAL_AI_RISK_ANCHOR_FROZEN_L21_READONLY_AUDIT_V1"
DATE="2026-10-17"
ROOT=Path("ml_features/delivery")

def require(test, err):
    if not test: raise ValueError(err)

def read(path):return json.loads(path.read_text(encoding="utf8"))

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda:stream.read(1048576),b""):h.update(b)
    return h.hexdigest()

def read_lines(path):
    with path.open("r",encoding="utf8") as file:
        for l in file:
            if l.strip():yield json.loads(l)

def valid_time(s):
    try: return datetime.datetime.fromisoformat(s.replace("Z","+00:00")).tzinfo is not None
    except (ValueError,AttributeError):return False

def key(e):
    typ=e["bet_type"];nums=e["combination"]
    require(typ in BETS and type(nums) is list and
            len(nums)==(1 if typ in ("WIN","PLACE") else 3 if typ in ("TRIO","TRIFECTA") else 2) and
            all(type(n) is int and 1<=n<=18 for n in nums) and
            (typ=="BRACKET_QUINELLA" or len(nums)==len(set(nums))), "UNEXPECTED_JRA_TICKET_COMBINATION_FORMAT")
    # Parentheses never silently re-sort ordered trifectas.
    if typ in ("QUINELLA","TRIO","WIDE","BRACKET_QUINELLA"):
        require(nums==sorted(nums),"JRA_UNORDERED_TICKET_NOT_CANONICAL")
    return typ,tuple(nums)

def audit(backfill, upstream_file, market_file, report):
    require(not report.exists(),"REFUSE_READONLY_AUDIT_REPORT_OVERWRITE")
    u=read(upstream_file)
    require(u["contract"]==UPSTREAM and u["real_races_joined"]==36 and
            u["real_horses_joined"]==471 and u["original_anchor_full_axis_races"]==36 and
            u["frozen_standard_rules"]==50 and u["frozen_strict_subset_rules"]==26 and
            u["frozen_l21_engine_invoked"] is False and u["ready_for_betting"] is False,
            "AUTHENTIC_ORIGINAL_AI_ANCHOR_L21_UPSTREAM_AUDIT_NOT_VERIFIED")
    fidx=read(backfill/ROOT/"690/index.json")
    oidx=read(backfill/ROOT/"odds/index.json")
    require(fidx["contract"]==FEATURE_INDEX and oidx["contract"]==ODDS_INDEX and
            oidx["total_races"]==60 and fidx["feature_count"]==690,
            "BACKFILL_SAVED_FEATURE_ODDS_INDEX_INVALID")
    market={}
    for r in read_lines(market_file):
        rid=str(r.get("race_id") or "")
        require(rid not in market and r.get("ready_for_betting") is False and
                r.get("market_asof_independently_verified") is False and
                r.get("win_market_source")=="JRA_OFFICIAL_POST" and
                r.get("2026_target_results_used") is False,"SAVED_L175_MARKET_RESEARCH_SOURCE_DRIFT")
        market[rid]={int(h["horse_number"]):float(h["market_win_odds"]) for h in r["horses"]}
        require(len(market[rid])==len(r["horses"]),"MARKET_DUPLICATE_HORSE_NUMBER")
    require(len(market)==36 and sum(map(len,market.values()))==471,
            "MARKET_RACE_POPULATION_INVALID")
    seen=set();rows=[];sumq=0;deltas=0
    for rec in u["races"]:
        rid=rec["race_id"]
        require(rid not in seen and rid in market and
                rid in fidx["races"] and rid in oidx["races"] and
                rec["original_anchor_horse_number"] in market[rid],
                "CROSS_SOURCE_RACE_ID_OR_ANCHOR_MISSING:"+rid)
        fi,oi=fidx["races"][rid],oidx["races"][rid]
        require(fi["date"]==oi["date"]==DATE and
                fi["sha256"]==oi["feature_sha256"] and
                oi["source"]=="JRA_OFFICIAL_POST" and
                oi["snapshot_type"]=="PRE_RACE_LIVE_ODDS" and
                oi["readiness"]=="PRE_RACE_OBSERVED_ASOF_UNVERIFIED",
                "FEATURE_ODDS_SAME_RACE_REFERENCE_MISMATCH:"+rid)
        odds_path=backfill/oi["path"]
        require(odds_path.is_file() and sha(odds_path)==oi["sha256"],
                "BACKFILL_ODDS_FILE_SHA256_MISMATCH:"+rid)
        o=read(odds_path)
        require(o["contract"]==ODDS_CONTRACT and o["race_id"]==rid and
                o["race_date"]==DATE and o["feature_reference"]==fi["path"] and
                o["feature_sha256"]==fi["sha256"] and
                o["original_is_unmodified_source"] is True,
                "SAVED_ODDS_FEATURE_IDENTITY_DRIFT:"+rid)
        d=o["odds"]
        require(d["source"]=="JRA_OFFICIAL_POST" and
                d["snapshot_type"]=="PRE_RACE_LIVE_ODDS" and
                d["readiness"]=="PRE_RACE_OBSERVED_ASOF_UNVERIFIED" and
                valid_time(d["source_captured_at_utc"]) and
                set(d["by_bet_type"])==set(BETS) and
                not d.get("not_offered_bet_types") and
                d["expected_combination_count"]==
                d["quoted_combination_count"]+d["no_votes_combination_count"]+
                d["scratched_combination_count"],
                "OFFICIAL_EIGHT_ODDS_COVERAGE_OR_PROVENANCE_DRIFT:"+rid)
        quotes=d["quotes"];novotes=d["no_votes"];scratches=d["scratched_combinations"]
        require(all(type(x) is list for x in (quotes,novotes,scratches)) and
                (len(quotes),len(novotes),len(scratches))==
                (d["quoted_combination_count"],d["no_votes_combination_count"],
                 d["scratched_combination_count"]),
                "ODDS_EXPECTED_VS_STORED_COUNT_DRIFT:"+rid)
        typed={typ:{"quoted":0,"no_votes":0,"scratched":0} for typ in BETS}
        keys=set();win={}
        for category,group in (("quoted",quotes),("no_votes",novotes),("scratched",scratches)):
            for e in group:
                kk=key(e)
                require(kk not in keys,"ODDS_DUPLICATE_STATE_OR_COMBINATION:"+rid)
                keys.add(kk);typed[e["bet_type"]][category]+=1
                if category=="quoted":
                    if e["bet_type"] in ("PLACE","WIDE"):
                        require("odds" not in e and
                                all(type(e.get(k)) in (float,int) and math.isfinite(e[k])
                                    for k in ("odds_min","odds_max")) and
                                1<=e["odds_min"]<=e["odds_max"],
                                "BAD_RANGE_MARKET_QUOTATION:"+rid)
                    else:
                        require(type(e.get("odds")) in (float,int) and
                                math.isfinite(e["odds"]) and e["odds"]>=1,
                                "BAD_MARKET_QUOTATION:"+rid)
                    if e["bet_type"]=="WIN": win[kk[1][0]]=float(e["odds"])
                else:
                    require(e.get("odds") is None,
                            "NO_VOTE_OR_SCRATCHED_HAS_IMAGINARY_ODDS:"+rid)
        for typ in BETS:
            t=d["by_bet_type"][typ]
            require(t["status"]=="COLLECTED" and
                    t["expected"]==t["listed"]==
                    typed[typ]["quoted"]+typed[typ]["no_votes"]+typed[typ]["scratched"] and
                    all(t[k]==typed[typ][k] for k in ("quoted","no_votes","scratched")),
                    "EIGHT_BET_GROUP_COMBINATION_LOSS:"+rid+":"+typ)
        require(set(win)==set(market[rid]),"SAVED_L175_WIN_HORSE_SET_VS_FULL8_MISMATCH:"+rid)
        changed=[n for n in win if not math.isclose(win[n],market[rid][n],rel_tol=0,abs_tol=1e-9)]
        sumq+=sum(typed[t]["quoted"] for t in L21_BETS)
        deltas+=len(changed)
        rows.append({"race_id":rid,"original_anchor_horse_number":rec["original_anchor_horse_number"],
                     "jra_full_eight_bet_types_verified":True,
                     "needed_frozen_l21_odds":{typ:typed[typ] for typ in L21_BETS},
                     "same_snapshot_win_odds_exact_match":not changed,
                     "different_win_price_horses":len(changed),
                     "saved_quote_capture":d["source_captured_at_utc"],
                     "asof_independently_certified":False,
                     "l21_ticket_engine_invoked":False,"ready_for_betting":False})
        seen.add(rid)
    require(len(seen)==36 and seen==set(market),"BACKFILL_36R_SAVED_MARKET_MISSING")
    blockers=[
        "FEATURE_AND_PRICE_ASOF_INDEPENDENT_PRESTART_CERTIFICATION_MISSING",
        "FROZEN_L21_ANCHOR_GATE_FEATURE_PANEL_NOT_YET_VALIDATED",
        "OFFICIAL_BET_CUTOFF_TICKET_ODDS_SNAPSHOT_NOT_PROVEN",
        "2026_MODEL_RELEASE_APPROVAL_NOT_GRANTED"]
    if deltas:blockers.insert(0,"L175_WIN_PRICES_DIFFER_FROM_BACKFILL_COMBINATION_SNAPSHOT")
    summary={"contract":"KEIBA_2026_1017_BACKFILL_EIGHT_ODDS_ORIGINAL_L21_SOURCE_AUDIT_V1",
        "real_races_verified":36,"real_horses_verified":471,
        "l21_bet_types":list(L21_BETS),"complete_stored_eight_bet_races":36,
        "frozen_l21_combinations_with_quotes":sumq,
        "same_market_snapshot_races":sum(r["same_snapshot_win_odds_exact_match"] for r in rows),
        "different_market_win_quote_horses":deltas,
        "frozen_standard_rules":50,"frozen_strict_subset_rules":26,
        "remaining_blockers":blockers,"2026_target_results_used":False,
        "frozen_l21_ticket_engine_invoked":False,"l3_v3_invoked":False,
        "ready_for_betting":False,"races":rows}
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print("BACKFILL_FULL_EIGHT_ODDS_TO_ORIGINAL_FROZEN_L21_SOURCE_AUDIT "+
          json.dumps({"races":36,"horses":471,"ticket_quotes":sumq,
                      "matching_market_races":summary["same_market_snapshot_races"],
                      "win_quote_price_mismatches":deltas,
                      "l21_betting":False},sort_keys=True),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for arg in ("backfill","upstream-audit","l175-scores","output"):
        ap.add_argument("--"+arg,type=Path,required=True)
    x=ap.parse_args()
    audit(x.backfill,x.upstream_audit,x.l175_scores,x.output)
