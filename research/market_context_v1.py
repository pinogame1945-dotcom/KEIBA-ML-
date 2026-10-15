#!/usr/bin/env python3
"""Strict one-race timestamped JRA win-odds MARKET CONTEXT, NOT L1.75.

A source snapshot must be traceable to an independent JRA pre-start capture.
Never silently use historical FINAL odds or post-race observations. Compare
market's normalized implied WIN probabilities to prior-year King projections
without presenting the difference as calibrated EV or a betting recommendation.
"""
from __future__ import annotations
import argparse,copy,datetime,json,math
from pathlib import Path
from urllib.parse import urlparse

CONTRACT="KEIBA_JRA_WIN_ODDS_ASOF_V1"
OUTPUT_CONTRACT="KEIBA_ONE_RACE_MARKET_CONTEXT_V1"
KINGS=("core4","pedlegacy","condition","full","light","pedv1","condrc")

def read_iso(raw,label):
    if not isinstance(raw,str) or not raw:
        raise ValueError("MISSING_"+label.upper())
    if raw.endswith("Z"):raw=raw[:-1]+"+00:00"
    try:
        timestamp=datetime.datetime.fromisoformat(raw)
    except ValueError as e:
        raise ValueError("INVALID_"+label.upper()) from e
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("TIMEZONE_REQUIRED_"+label.upper())
    return timestamp.astimezone(datetime.timezone.utc)

def finite(x,label):
    if isinstance(x,bool):raise ValueError("INVALID_"+label)
    try:v=float(x)
    except (TypeError,ValueError) as e:raise ValueError("INVALID_"+label) from e
    if not math.isfinite(v):raise ValueError("NONFINITE_"+label)
    return v

def attach_win_market(analysis,market):
    if analysis.get("contract")!="KEIBA_ONE_RACE_COMBINED_ANALYSIS_V1":
        raise ValueError("SINGLE_RACE_ANALYSIS_CONTRACT_MISMATCH")
    if market.get("contract")!=CONTRACT:
        raise ValueError("WIN_ODDS_SOURCE_CONTRACT_MISMATCH")
    rid=str(analysis.get("race_id") or "")
    if rid!=str(market.get("canonical_race_id") or ""):
        raise ValueError("RACE_ID_DRIFT_MARKET")
    if len(rid)!=12 or not rid.isascii() or not rid.isdigit() or int(rid[:4])>=2026:
        raise ValueError("2026_SEALED_OR_NONCANONICAL_RACE")
    if market.get("source_system")!="KEIBA_MOBILE_JRA":
        raise ValueError("UNVERIFIED_MARKET_SOURCE_SYSTEM")
    if market.get("source_phase")!="PRE_RACE":
        raise ValueError("FINAL_OR_RESULTS_ODDS_FORBIDDEN")
    if market.get("is_final") is not False:
        raise ValueError("MUST_EXPLICITLY_ASSERT_NOT_FINAL")
    if market.get("source_status") in ("FINAL","RESULT","POST_RACE"):
        raise ValueError("POST_RACE_ODDS_FORBIDDEN")
    start=read_iso(market.get("scheduled_start_at"),"scheduled_start_at")
    captured=read_iso(market.get("observed_at"),"observed_at")
    if not captured<start:
        raise ValueError("ODDS_SNAPSHOT_NOT_PRE_START")
    if start.date().year not in (int(rid[:4]),int(rid[:4])+1):
        raise ValueError("RACE_START_YEAR_DRIFT")
    if not isinstance(analysis.get("horses"),list) or len(analysis["horses"])<3:
        raise ValueError("MISSING_ANALYSIS_HORSE_COVERAGE")
    horse_by_no={}
    for horse in analysis["horses"]:
        raw=horse.get("horse_number")
        if isinstance(raw,bool):raise ValueError("INVALID_HORSE_NUMBER")
        try:number=int(raw)
        except (TypeError,ValueError) as e:raise ValueError("INVALID_HORSE_NUMBER") from e
        if number<1 or number>24 or number in horse_by_no:
            raise ValueError("HORSE_NUMBER_AMBIGUOUS")
        horse_by_no[number]=horse
    rows=market.get("rows")
    if not isinstance(rows,list) or not rows or len(rows)>24:
        raise ValueError("INVALID_WIN_ODDS_ROWS")
    odds={}
    url_hosts=[]
    for row in rows:
        if row.get("betType")!="WIN" or row.get("selection2") is not None or row.get("selection3") is not None:
            raise ValueError("NON_WIN_OR_COMBINATION_ODDS_FORBIDDEN")
        try:number=int(row["selection1"])
        except (TypeError,ValueError,KeyError) as e:raise ValueError("INVALID_WIN_SELECTION") from e
        if number not in horse_by_no or number in odds:
            raise ValueError("MARKET_MISSING_OR_DUPLICATE_HORSE_NUMBER")
        price=finite(row.get("odds"),"WIN_ODDS")
        if price<1.0 or price>9999.9:
            raise ValueError("MARKET_PRICE_OUT_OF_RANGE")
        if row.get("oddsMin") is not None or row.get("oddsMax") is not None:
            raise ValueError("RANGE_ODDS_CANNOT_BE_WIN")
        if read_iso(row.get("observedAt"),"win_row_observed_at")!=captured:
            raise ValueError("MIXED_SNAPSHOT_TIMES")
        link=row.get("sourceUrl")
        parsed=urlparse(str(link or ""))
        if parsed.scheme!="https" or parsed.hostname not in ("www.jra.go.jp","jra.go.jp","www.jra.jp","jra.jp"):
            raise ValueError("ODDS_SOURCE_MUST_BE_OFFICIAL_JRA_HTTPS")
        url_hosts.append(parsed.hostname)
        odds[number]=price
    if set(odds)!=set(horse_by_no):
        raise ValueError("PARTIAL_OR_UNKNOWN_FIELD_ODDS_FORBIDDEN")
    probabilities={number:1/price for number,price in odds.items()}
    inverse_sum=sum(probabilities.values())
    if not 0<inverse_sum<3:
        raise ValueError("UNREASONABLE_MARKET_OVERROUND")
    market_p={number:p/inverse_sum for number,p in probabilities.items()}
    # Rank ties use horse NUMBER, NEVER archive finish/result ordering.
    ranked_market=sorted(odds,key=lambda n:(odds[n],n))
    market_ranks={n:i+1 for i,n in enumerate(ranked_market)}
    # Use only real newly run PRIOR-YEAR model probabilities, never the
    # retrospectively selected 2025 archived Council model values.
    prior=analysis.get("prior_year_new_model_inference") or {}
    model_valid=prior.get("status")=="ORIGINAL_MODELS_SCORED_NOT_VERIFIED_PREBET"
    model_p={}
    if model_valid:
        sum_by_king={}
        for alias in KINGS:
            raw=[finite(h.get("prior_year_king_raw_probabilities",{}).get(alias),
                        "PRIOR_YEAR_"+alias) for h in analysis["horses"]]
            if any(p<0 or p>1 for p in raw) or sum(raw)<=0:
                raise ValueError("INVALID_PRIOR_YEAR_KING_PROBABILITIES")
            sum_by_king[alias]=sum(raw)
        for h in analysis["horses"]:
            hid=str(h["horse_id"])
            model_p[hid]=sum(
                finite(h["prior_year_king_raw_probabilities"][alias],
                    "PRIOR_YEAR_"+alias)/sum_by_king[alias]
                for alias in KINGS)/len(KINGS)
        if abs(sum(model_p.values())-1)>1e-9:
            raise ValueError("KING_MIXTURE_NOT_NORMALIZED")
    result=copy.deepcopy(analysis)
    for h in result["horses"]:
        number=int(h["horse_number"])
        h["market_win_odds"]=odds[number]
        h["market_normalized_win_probability"]=market_p[number]
        h["market_win_rank"]=market_ranks[number]
        h["market_quote_observed_at"]=market["observed_at"]
        h["market_comparison"]={
            "status":"MODEL_VS_MARKET_DESCRIPTIVE_ONLY" if model_valid else "PRIOR_YEAR_MODEL_NOT_AVAILABLE",
            "is_l175":False,
            "is_calibrated_EV":False,
        }
        if model_valid:
            p=model_p[str(h["horse_id"])]
            h["market_comparison"]["uncalibrated_prior_year_king_mean_win_probability"]=p
            h["market_comparison"]["uncalibrated_win_probability_gap_pp"]=100*(p-market_p[number])
    result["market_context"]={
        "contract":OUTPUT_CONTRACT,
        "status":"TIMESTAMP_CHECKED_SOURCE_AUTHENTICITY_NOT_INDEPENDENTLY_ATTESTED",
        "source_system":"KEIBA_MOBILE_JRA",
        "source_phase":"PRE_RACE",
        "source_market":"WIN_ODDS_ONLY",
        "bet_odds_not_final":True,
        "race_id":rid,
        "observed_at":market["observed_at"],
        "scheduled_start_at":market["scheduled_start_at"],
        "captured_before_scheduled_start":True,
        "source_hostname":sorted(set(url_hosts)),
        "field_coverage":len(odds),
        "market_overround":inverse_sum,
        "model_comparison":"UNCALIBRATED_PRIOR_YEAR_7KING_WIN_VS_MARKET" if model_valid else "NONE",
        "not_l175":True,
        "not_bet_roi_or_ev":True,
        "no_reorder_or_ticket_modification":True,
        "warning":"Timestamps/source attribution from snapshot are checked for internal consistency only. JRA server publication time not independently authenticated. No L1.75 market correction or profitable odds claim.",
    }
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--analysis",required=True)
    ap.add_argument("--market",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    x=attach_win_market(
        json.loads(Path(a.analysis).read_text(encoding="utf-8")),
        json.loads(Path(a.market).read_text(encoding="utf-8")))
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("PRESTART_MARKET_OVERLAY_READY",x["race_id"],len(x["horses"]),
          x["market_context"]["model_comparison"])

if __name__=="__main__":
    main()
