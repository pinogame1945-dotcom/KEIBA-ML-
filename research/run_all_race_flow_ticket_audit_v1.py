#!/usr/bin/env python3
"""All-race anchor-flow hit and ticket-cost audit; NO payouts guessed.

Inputs: frozen out-of-sample full-field predictions and exact V1 win picks.
All race selection happens independently of results; all 2023-2025 races included.
TRIO exactly settles only when there are exactly three top3-labelled horses.
QUINELLA top2 is unavailable: 'winner + top3 other' is an UPPER BOUND proxy.
"""
import argparse
import concurrent.futures
import itertools
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

YEARS=(2023,2024,2025)
RACES_YEAR=3456
TOTAL_RACES=RACES_YEAR*3
SOURCES=("MARKET","AI_PROB")
AXES=("FAVORITE_AXIS","VALUE_AXIS","BOTH_AXES_UNION","TWO_AXES_WITH_SINGLE_FALLBACK")
PARTNERS=(3,5,7,10)
PICK_STRATEGIES=("MARKET_FAVORITE","ANCHOR_FULL")
INPUT_PRED=["year","_race_id","_horse_id","market_rank","final_win_odds",
            "P__ANCHOR_FULL","target_top3","target_win"]
INPUT_PICK=["year","_race_id","_horse_id","strategy","target_win"]

def assert_ok(val,text):
    if not bool(val):raise ValueError(text)

def get_inputs(pred_path,picks_path):
    d=pd.read_csv(pred_path,usecols=INPUT_PRED,compression="infer",
                  dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    p=pd.read_csv(picks_path,usecols=INPUT_PICK,compression="infer",
                  dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    for name,x in (("pred",d),("pick",p)):
        x["year"]=pd.to_numeric(x["year"],errors="raise").astype(int)
        assert_ok(set(x["year"].unique())==set(YEARS),f"{name}: 2026 sealed / missing year")
        assert_ok(not x[["year","_race_id","_horse_id"]].isna().any().any(),f"{name}: missing horse")
    assert_ok(not d.duplicated(["_race_id","_horse_id"]).any(),"duplicate pred horse")
    for c in ("market_rank","final_win_odds","P__ANCHOR_FULL","target_top3","target_win"):
        d[c]=pd.to_numeric(d[c],errors="raise")
    assert_ok(not d[["market_rank","final_win_odds","P__ANCHOR_FULL",
                    "target_top3","target_win"]].isna().any().any(),"invalid prediction")
    assert_ok(d["target_win"].isin([0,1]).all() and
              d["target_top3"].isin([0,1]).all(),"invalid target")
    assert_ok(not (d["target_win"].gt(d["target_top3"])).any(),"winner not top3")
    p=p[p["strategy"].isin(PICK_STRATEGIES)].copy()
    assert_ok(len(p)==TOTAL_RACES*2,"picks incomplete")
    assert_ok(not p.duplicated(["_race_id","strategy"]).any(),"duplicate pick")
    p["target_win"]=pd.to_numeric(p["target_win"],errors="raise")
    joined=p.merge(d[["year","_race_id","_horse_id","target_win"]],
                   on=["year","_race_id","_horse_id"],
                   suffixes=("_pick","_pred"),how="left",validate="many_to_one")
    assert_ok(joined["target_win_pick"].eq(joined["target_win_pred"]).all(),"pick drift")
    assert_ok(d["_race_id"].nunique()==TOTAL_RACES,"OOS race count changed")
    result={}
    for name in PICK_STRATEGIES:
        s=p[p["strategy"].eq(name)]
        assert_ok(s["_race_id"].nunique()==TOTAL_RACES,f"{name} race coverage")
        result[name]=dict(zip(s["_race_id"],s["_horse_id"]))
    return d,result

def flow_tickets(kind,axis,market,value,ordered,k):
    seen=set()
    def one(a):
        others=[h for h in ordered if h!=a][:k]
        if kind=="TRIO":
            return [tuple(sorted((a,*pair))) for pair in itertools.combinations(others,2)]
        return [tuple(sorted((a,x))) for x in others]
    if axis=="FAVORITE_AXIS":
        seen.update(one(market))
    elif axis=="VALUE_AXIS":
        seen.update(one(value))
    elif axis=="BOTH_AXES_UNION":
        seen.update(one(market));seen.update(one(value))
    else:
        if market!=value:
            extra=[h for h in ordered if h!=market and h!=value][:k]
            if kind=="TRIO":
                seen.update(tuple(sorted((market,value,h))) for h in extra)
            else:
                seen.add(tuple(sorted((market,value))))
        else:
            seen.update(one(market))
    return seen

def eval_year(year, d, favorites, values):
    per_race=d[d["year"].eq(year)].groupby("_race_id",sort=False)
    sums=defaultdict(lambda:Counter())
    cases=defaultdict(lambda:Counter())
    detail=[]
    for race,df in per_race:
        m=favorites[race];v=values[race]
        horses=set(df["_horse_id"])
        assert_ok(m in horses and v in horses, f"missing anchor {race}")
        market_order=list(df.sort_values(["market_rank","final_win_odds","_horse_id"],
                                 ascending=[True,True,True])["_horse_id"])
        ai_order=list(df.sort_values(["P__ANCHOR_FULL","market_rank","_horse_id"],
                             ascending=[False,True,True])["_horse_id"])
        results=dict(zip(df["_horse_id"],zip(df["target_win"],df["target_top3"])))
        top3={h for h,z in results.items() if z[1]==1}
        winners={h for h,z in results.items() if z[0]==1}
        assert_ok(len(winners) in (1,2),f"winner count {race}")
        assert_ok(3<=len(top3)<=6,f"top3 count unexpected {race} count={len(top3)}")
        assert_ok(len(winners-top3)==0,f"winner outside top3 {race}")
        both_win_miss=(m not in winners and v not in winners)
        scope=("ALL","BOTH_WIN_MISS") if both_win_miss else ("ALL",)
        for kind in ("QUINELLA","TRIO"):
            for source in SOURCES:
                ordered=market_order if source=="MARKET" else ai_order
                for axis in AXES:
                    for k in PARTNERS:
                        combo=(kind,source,axis,k)
                        tickets=flow_tickets(kind,axis,m,v,ordered,k)
                        assert_ok(len(tickets)==len(set(tickets)),f"duplicate ticket {race}")
                        # QUINELLA is a generous necessary-condition proxy ONLY:
                        # actual runner-up not in saved OOS file. Never report as exact hit.
                        if kind=="QUINELLA":
                            potential=sum(int(any(w in t for w in winners) and
                                              all(h in top3 for h in t)) for t in tickets)
                            exact=0; ambiguous=potential
                        else:
                            potential=sum(int(all(h in top3 for h in t)) for t in tickets)
                            exact=potential if len(top3)==3 else 0
                            ambiguous=potential if len(top3)!=3 else 0
                        for sc in scope:
                            z=sums[(year,sc,*combo)]
                            z["races"]+=1
                            z["tickets"]+=len(tickets)
                            z["cost_yen"]+=100*len(tickets)
                            z["potential_hit_tickets"]+=potential
                            z["potential_hit_races"]+=int(potential>0)
                            z["exact_hit_tickets"]+=exact
                            z["exact_hit_races"]+=int(exact>0)
                            z["ambiguous_tickets"]+=ambiguous
                            z["ambiguous_races"]+=int(ambiguous>0)
                        if sc=="ALL" and kind=="TRIO" and source=="MARKET" and axis=="BOTH_AXES_UNION" and k==5:
                            detail.append({"year":year,"race_id":race,
                                           "both_win_miss":int(both_win_miss),
                                           "distinct_anchors":int(m!=v),
                                           "top3_horses":len(top3),
                                           "ticket_count":len(tickets),
                                           "possible_trio_hit":int(potential>0),
                                           "exact_three_places":int(len(top3)==3)})
    assert_ok(len(per_race)==RACES_YEAR,f"{year} races missing")
    return sums,detail

def aggregate(sums):
    combined=defaultdict(Counter)
    for (year,scope,kind,source,axis,k),c in sums.items():
        combined[(str(year),scope,kind,source,axis,k)].update(c)
        combined[("2023-2025",scope,kind,source,axis,k)].update(c)
    rows=[]
    for (year,scope,kind,source,axis,k),c in combined.items():
        n=c["races"];cost=c["cost_yen"];potential=c["potential_hit_races"]
        assert_ok(n==TOTAL_RACES if year=="2023-2025" and scope=="ALL" else
                  n==(5951 if year=="2023-2025" else
                      RACES_YEAR if scope=="ALL" else {2023:1962,2024:1959,2025:2030}[int(year)]),
                  f"scope coverage problem {year}/{scope}")
        rows.append({
            "year":year,"scope":scope,"bet_type":kind,"partner_order":source,
            "anchor_rule":axis,"other_horses_k":k,
            "races":n,"tickets":c["tickets"],"average_tickets_per_race":round(c["tickets"]/n,4),
            "stake_yen":cost,
            "potential_hit_races":potential,
            "potential_hit_pct":round(100*potential/n,5),
            "potential_hit_tickets":c["potential_hit_tickets"],
            "exact_three_places_hit_races":c["exact_hit_races"],
            "exact_three_places_hit_tickets":c["exact_hit_tickets"],
            "ambiguous_top3_races":c["ambiguous_races"],
            "break_even_avg_return_yen_per_potential_hit_race":round(cost/potential,2) if potential else None,
            "actual_payout_roi_pct":None,
            "settlement_warning":("upper_bound_only_runner_up_unknown" if kind=="QUINELLA"
                                  else "unsettled_exact_payout_missing"),
        })
    return pd.DataFrame(rows).sort_values(
        ["year","scope","bet_type","partner_order","anchor_rule","other_horses_k"]
    ).reset_index(drop=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--predictions",required=True)
    parser.add_argument("--picks",required=True)
    parser.add_argument("--out-dir",required=True)
    a=parser.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    d,picks=get_inputs(a.predictions,a.picks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda y:eval_year(y,d,picks["MARKET_FAVORITE"],
                                                picks["ANCHOR_FULL"]),YEARS))
    sums={};details=[]
    for c,dd in results:
        sums.update(c);details.extend(dd)
    summary=aggregate(sums)
    assert_ok(len(summary)>0,"empty output")
    summary.to_csv(out/"all-race-flow-ticket-coverage.csv",index=False)
    pd.DataFrame(details).to_csv(out/"sample-race-combinations.csv.gz",
                                 index=False,compression="gzip")
    preferred=summary[(summary.year=="2023-2025") &
                      (summary.scope=="ALL") &
                      (summary.bet_type=="TRIO") &
                      (summary.partner_order=="MARKET") &
                      (summary.anchor_rule=="BOTH_AXES_UNION")]
    print("=== TRIО MARKET PARTNERS, TWO AXES UNION, ALL RACES ===")
    print(preferred.to_csv(index=False),flush=True)
    important=summary[(summary.year=="2023-2025") &
                      (summary.scope=="BOTH_WIN_MISS") &
                      (summary.bet_type=="TRIO") &
                      (summary.partner_order=="MARKET") &
                      (summary.anchor_rule=="BOTH_AXES_UNION")]
    print("=== TWO WIN PICKS MISS, SAME PRESET (RETROSPECTIVE DIAGNOSTIC) ===")
    print(important.to_csv(index=False),flush=True)
    (out/"summary.json").write_text(json.dumps({
        "contract":"ALL_RACE_FLOW_TICKET_AUDIT_V1",
        "years":list(YEARS),"races":TOTAL_RACES,
        "rules":"fixed market-rank / AI win-prob partner lists; favorites/AI expected-value axes",
        "stake_per_ticket_yen":100,
        "total_strategies":int(summary[(summary.year=="2023-2025") & (summary.scope=="ALL")].shape[0]),
        "2026_locked":True,"model_refit":False,"trained_router":False,
        "no_after_result_selection":"BOTH_WIN_MISS slice is a diagnostic only; it CANNOT be identified at bet time",
        "quinella_limitation":"Exact second place absent: potential captures are upper bounds, not winning quinella tickets",
        "trio_limitation":"top3 labels recover candidate hit coverage; exact payouts not in source; ambiguous ties excluded from certain-hit column",
        "payout_and_roi":"UNAVAILABLE. Never infer ROI from win odds or call theoretical coverage profitable.",
        "payout_next_step":"Fetch matching authoritative BACKFILL exact payouts and horse number maps to settle generated tickets; need confirmation of free resource bounds before large data transfer."
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ALL_RACE_FLOW_TICKET_AUDIT_V1_COMPLETE",flush=True)

if __name__=="__main__":main()
