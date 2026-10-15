#!/usr/bin/env python3
"""2023-25 fixed EV/model/rank ticket-candidate price-floor lab.

Ticket-by-ticket gate uses the archived FINAL win/place/quinella/wide/exacta/
trio/trifecta *pre-settlement* prices, never the eventual winning dividend.
All payout values settle against archived official refunds. Entire experiment
uses pre-defined ticket templates and thresholds; after-selection inference
is carefully separated from chronological rolling decision results.

2026 outcomes sealed; CPU only; no paid storage or arbitrary API downloads.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import gzip
import itertools
import json
import math
from bisect import bisect_right
from collections import Counter,defaultdict
from pathlib import Path

import pandas as pd
import run_l2_value_anchor_bet_frontier_v1 as prior
from build_l2_market_helpers_v1 import load_day,load_odds_day,horse_number_map,canonical_numbers
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,get_inputs,assert_ok

FLOORS=(0.0,1.5,2.0,3.0,5.0,7.0,10.0,15.0,20.0,30.0,50.0,
        75.0,100.0,150.0,200.0,300.0,500.0,1000.0,2000.0,5000.0)
AXES=("EV","AI_PROB","MARKET","AI_RAW")
ORDERS=("AI_PROB","MARKET","AI_RAW")
COUNTS=(2,3,5,7)
FAMILIES=("QUINELLA","WIDE","EXACTA_FIRST","EXACTA_SECOND","EXACTA_BOTH",
          "TRIO","TRIFECTA_FIRST","TRIFECTA_SECOND","TRIFECTA_THIRD","TRIFECTA_ANY")
BOX_FAMILIES=("QUINELLA","WIDE","EXACTA_BOX","TRIO","TRIFECTA_BOX")
BET_TO_GROUP={"WIN":"1","PLACE":"2","QUINELLA":"4","WIDE":"5",
              "EXACTA":"6","TRIO":"7","TRIFECTA":"8"}
RANGE_FAMILIES={"PLACE","WIDE"}
PRESETS={
 "LOOSE":{"WIN":3,"PLACE":1.5,"QUINELLA":5,"WIDE":2,
          "EXACTA":10,"TRIO":10,"TRIFECTA":30},
 "BALANCED":{"WIN":5,"PLACE":2,"QUINELLA":10,"WIDE":3,
          "EXACTA":20,"TRIO":20,"TRIFECTA":75},
 "STRICT":{"WIN":10,"PLACE":3,"QUINELLA":20,"WIDE":5,
          "EXACTA":50,"TRIO":50,"TRIFECTA":200},
}
PORTFOLIO_LEGS=(
 ("EV","SINGLE",0,"WIN"),
 ("EV","SINGLE",0,"PLACE"),
 ("EV","AI_PROB",3,"QUINELLA"),
 ("EV","MARKET",3,"WIDE"),
 ("EV","AI_PROB",3,"EXACTA_FIRST"),
 ("EV","AI_PROB",3,"TRIO"),
 ("EV","AI_PROB",5,"TRIFECTA_FIRST"),
)
BENCHMARKS={
 ("EV","SINGLE",0,"WIN"):(900960,1917,10368),
 ("EV","SINGLE",0,"PLACE"):(865630,4467,10368),
 ("EV","AI_PROB",2,"TRIO"):(921600,866,10368),
 ("EV","MARKET",2,"EXACTA_FIRST"):(1839560,1000,20736),
 ("EV","MARKET",2,"WIDE"):(1782070,3561,20736),
}

def parse_price(raw,kind):
    if not isinstance(raw,list) or len(raw)<3:return None
    tup=raw[3:6] if len(raw)>=6 else raw[:3]
    try:
        low=float(str(tup[0]).replace(",",""))
        high=float(str(tup[1]).replace(",","")) if kind in RANGE_FAMILIES else None
    except (ValueError,TypeError):return None
    if not math.isfinite(low) or low<=0:return None
    if high is not None and (not math.isfinite(high) or high<low):return None
    return low

def sort_horses(g,kind):
    if kind=="AI_PROB":
        return g.sort_values(["P__ANCHOR_FULL","market_rank","_horse_id"],
              ascending=[False,True,True],kind="mergesort")["_horse_id"].astype(str).tolist()
    if kind=="MARKET":
        return g.sort_values(["market_rank","final_win_odds","_horse_id"],
              ascending=[True,True,True],kind="mergesort")["_horse_id"].astype(str).tolist()
    raise ValueError(kind)

def box_tickets(order,family):
    if family in ("QUINELLA","WIDE"):
        return {(family,x) for x in itertools.combinations(sorted(order),2)}
    if family=="TRIO":
        return {("TRIO",x) for x in itertools.combinations(sorted(order),3)}
    if family=="EXACTA_BOX":
        return {("EXACTA",x) for x in itertools.permutations(order,2)}
    if family=="TRIFECTA_BOX":
        return {("TRIFECTA",x) for x in itertools.permutations(order,3)}
    raise ValueError(family)

def ticket_price(ticket,odds):
    kind,nums=ticket
    key=(str(nums[0]) if len(nums)==1 else "".join(f"{x:02d}" for x in nums))
    group=(odds.get("odds") or {}).get(BET_TO_GROUP[kind]) or {}
    raw=group.get(key)
    return parse_price(raw,kind)

def template_candidates(anchors,orders,number):
    for axis in AXES:
        anchor=anchors[axis]
        an=number[anchor]
        for kind in ("WIN","PLACE"):
            yield (axis,"SINGLE",0,kind),{(kind,(an,))}
        for source in ORDERS:
            ordered=orders[source]
            others=[number[h] for h in ordered if h!=anchor]
            assert_ok(len(others)==len(ordered)-1,"partner/race key drift")
            for k in COUNTS+((10,) if axis=="EV" else ()):
                by=prior.pair_templates(an,others[:k])
                for family in FAMILIES:
                    yield (axis,source,k,family),by[family]
    for source in ORDERS:
        ordered=orders[source]
        for k in (3,5,7):
            union={}
            for anchor in sorted({anchors["EV"],anchors["MARKET"]}):
                others=[number[h] for h in ordered if h!=anchor]
                x=prior.pair_templates(number[anchor],others[:k])
                for fam in FAMILIES:
                    union.setdefault(fam,set()).update(x[fam])
            for fam in FAMILIES:
                yield ("EV_MARKET_UNION",source,k,fam),union[fam]
    for source in ORDERS:
        nums=[number[h] for h in orders[source]]
        for k in (4,6):
            if len(nums)<k:continue
            for fam in BOX_FAMILIES:
                yield ("BOX",source,k,fam),box_tickets(nums[:k],fam)

def year_task(year,data,picks,races,root):
    year_data=data[data.year.eq(year)]
    groups=year_data.groupby("_race_id",sort=False)
    year_races=races.loc[races.year.eq(year)]
    assert_ok(len(year_races)==RACES_YEAR,"race count drift")
    by_race=year_races.set_index("race_id")
    dates=year_races.groupby("race_date")["race_id"].apply(list)
    agg=defaultdict(Counter)
    baseline=defaultdict(Counter)
    portfolio=defaultdict(Counter)
    meta=Counter()
    family_price=defaultdict(Counter)
    for date,ids in dates.items():
        pays=load_day(root/"data"/"daily"/(str(date)+".jsonl.gz"),set(ids))
        odds=load_odds_day(root/"data"/"odds"/"daily"/(str(date)+".jsonl.gz"),set(ids))
        for rid in ids:
            meta["races"]+=1
            pack=pays.get(rid)
            assert_ok(pack is not None,"missing refund pack: "+rid)
            market=odds.get(rid)
            if market is None:
                meta["races_missing_final_odds"]+=1
                continue
            assert_ok(market.get("odds_kind")=="FINAL" and market.get("source_status")=="result",
                      "unsupported odds state "+rid)
            g=groups.get_group(rid)
            row=by_race.loc[rid]
            ordered={
                "AI_PROB":sort_horses(g,"AI_PROB"),
                "MARKET":sort_horses(g,"MARKET"),
                "AI_RAW":g.sort_values(["distill_ai_rank","market_rank","_horse_id"],
                                      ascending=[True,True,True],kind="mergesort")["_horse_id"].astype(str).tolist()
                    if "distill_ai_rank" in g.columns else [],}
            # The archived raw-AI axis is authoritative; raw full-field ranks
            # live in the independent all-race source, not residual predictions.
            if not ordered["AI_RAW"]:
                ordered["AI_RAW"]=sort_horses(g,"AI_PROB")
                meta["raw_order_unavailable_races"]+=1
            anchors={
                "EV":picks["ANCHOR_FULL"][rid],
                "AI_PROB":str(row["AI_PROB_TOP_horse"]),
                "MARKET":str(row["MARKET_TOP_horse"]),
                "AI_RAW":str(row["AI_RAW_TOP_horse"]),
            }
            no=horse_number_map(pack)
            assert_ok(set(anchors.values()).issubset(no),"selected horse missing from official pack: "+rid)
            assert_ok(all(set(seq).issubset(no) for seq in ordered.values()),"unmapped horse "+rid)
            refunds=prior.payouts(pack,rid)
            assert_ok(ordered["AI_PROB"][0]==anchors["AI_PROB"] and
                      ordered["MARKET"][0]==anchors["MARKET"] and
                      anchors["EV"]==row["VALUE_EV_horse"],"frozen selection drift "+rid)
            cache={}
            cached_templates={}
            for tpl,tickets in template_candidates(anchors,ordered,no):
                if not tickets:continue
                n=len(FLOORS)
                ct=[0]*n
                paid=[0.0]*n
                hits=[0]*n
                all_n=len(tickets)
                all_paid=0
                all_hits=0
                for ticket in tickets:
                    if ticket not in cache:
                        cache[ticket]=(ticket_price(ticket,market),refunds.get(ticket,0.0))
                    price,ret=cache[ticket]
                    all_paid+=ret
                    all_hits+=int(ret>0)
                    if price is None:
                        family_price[ticket[0]]["missing"]+=1
                        if ret>0:family_price[ticket[0]]["missing_paid"]+=1
                        continue
                    family_price[ticket[0]]["priced"]+=1
                    b=bisect_right(FLOORS,price)-1
                    if b>=0:
                        ct[b]+=1; paid[b]+=ret;hits[b]+=int(ret>0)
                c=baseline[tpl]
                c["races"]+=1;c["all_tickets"]+=all_n;c["all_stake_yen"]+=all_n*100
                c["all_refund_yen"]+=all_paid;c["all_hit_races"]+=int(all_hits>0)
                # Fold prefix: every floor below an observed ticket price purchases it.
                nbuy=0;ngot=0;nsuccess=0
                for z in range(n-1,-1,-1):
                    nbuy+=ct[z];ngot+=paid[z];nsuccess+=hits[z]
                    if not nbuy:continue
                    c2=agg[(tpl,z)]
                    c2["races_bought"]+=1
                    c2["tickets"]+=nbuy
                    c2["return_yen"]+=ngot
                    c2["hit_races"]+=int(nsuccess>0)
                    c2["hit_tickets"]+=nsuccess
                if tpl in PORTFOLIO_LEGS:
                    cached_templates[tpl]=tickets
            # Cross-family portfolios: allow a race if at least one individual
            # ticket clears *its own bet type's* fixed price floor.
            for preset,thresholds in PRESETS.items():
                merged={}
                for tpl in PORTFOLIO_LEGS:
                    for ticket in cached_templates[tpl]:
                        merged[ticket]=None
                for cap in (0,5,10,20,40):
                    purchased=[]
                    for ticket in merged:
                        price,ref=cache[ticket]
                        if price is None or price<thresholds[ticket[0]]:continue
                        purchased.append((price,ref,ticket))
                    purchased.sort(key=lambda x:(x[0],x[2]))
                    if cap:purchased=purchased[:cap]
                    c=portfolio[(preset,cap)]
                    if not purchased:continue
                    c["races_bought"]+=1
                    c["tickets"]+=len(purchased)
                    c["return_yen"]+=sum(t[1] for t in purchased)
                    c["hit_races"]+=int(any(t[1]>0 for t in purchased))
                    c["hit_tickets"]+=sum(t[1]>0 for t in purchased)
            if meta["races"]%1200==0:print("PROGRESS",year,meta["races"],flush=True)
    assert_ok(meta["races"]==RACES_YEAR,"missing races "+str(year))
    return year,agg,baseline,portfolio,meta,family_price

def summarize(records):
    agg,baseline,portfolio=defaultdict(Counter),defaultdict(Counter),defaultdict(Counter)
    meta=defaultdict(Counter); prices=defaultdict(Counter)
    for year,a,b,c,d,e in records:
        for tpl,cnt in a.items():
            agg[(str(year),*tpl)].update(cnt);agg[("2023-2025",*tpl)].update(cnt)
        for tpl,cnt in b.items():
            baseline[(str(year),tpl)].update(cnt);baseline[("2023-2025",tpl)].update(cnt)
        for k,cnt in c.items():
            portfolio[(str(year),*k)].update(cnt);portfolio[("2023-2025",*k)].update(cnt)
        meta[str(year)].update(d);meta["2023-2025"].update(d)
        for k,cnt in e.items():
            prices[(str(year),k)].update(cnt);prices[("2023-2025",k)].update(cnt)
    return agg,baseline,portfolio,meta,prices

def dataframe_totals(agg,baseline,meta):
    rows=[]
    for (year,axis,source,k,family,z),c in agg.items():
        available=meta[year]["races"]-meta[year]["races_missing_final_odds"]
        price_floors=FLOORS[z]
        n=c["tickets"];ret=c["return_yen"]
        rows.append(dict(year=year,axis=axis,order=source,k=k,family=family,
            price_floor=price_floors,usable_races=available,
            all_races=meta[year]["races"],races_bought=c["races_bought"],
            race_coverage_pct=round(100*c["races_bought"]/meta[year]["races"],5),
            hit_races=c["hit_races"],hit_tickets=c["hit_tickets"],
            hit_per_bought_race_pct=round(100*c["hit_races"]/c["races_bought"],5),
            tickets=n,avg_tickets_per_bought_race=round(n/c["races_bought"],4),
            stake_yen=100*n,return_yen=round(ret,2),profit_yen=round(ret-100*n,2),
            roi_pct=round(ret/n,6),
            odds_scope="FINAL lower-end for PLACE and WIDE"))
    return pd.DataFrame(rows)

def forward_test(df):
    # Price-cut and fixed-ticket-template tuning is retrospectively searched
    # only in previous years; test period never participates in selection.
    testing=[]
    for test_year,train_years in (("2024",["2023"]),("2025",["2023","2024"])):
        train=df.loc[df.year.isin(train_years)].groupby(
               ["axis","order","k","family","price_floor"],dropna=False,as_index=False)[
               ["races_bought","tickets","stake_yen","return_yen","hit_races"]].sum()
        train["roi"]=train.return_yen/train.stake_yen*100
        years=len(train_years)
        train["avg_races_per_year"]=train.races_bought/years
        train["avg_tickets_per_year"]=train.tickets/years
        # Predeclared per-bet-kind policy candidates: broad participation vs
        # low point-count; don't select 1-hit wonder/no-coverage policies.
        for axis in ("EV","AI_PROB","MARKET","AI_RAW","EV_MARKET_UNION","BOX"):
            for category,families in (
                ("SINGLES",["WIN","PLACE"]),
                ("PAIRS",["QUINELLA","WIDE","EXACTA_FIRST","EXACTA_SECOND","EXACTA_BOTH","EXACTA_BOX"]),
                ("TRIO",["TRIO"]),
                ("TRIFECTA",["TRIFECTA_FIRST","TRIFECTA_SECOND","TRIFECTA_THIRD","TRIFECTA_ANY","TRIFECTA_BOX"]),
                ("ANY",["WIN","PLACE","QUINELLA","WIDE","EXACTA_FIRST","EXACTA_SECOND","EXACTA_BOTH","EXACTA_BOX","TRIO",
                        "TRIFECTA_FIRST","TRIFECTA_SECOND","TRIFECTA_THIRD","TRIFECTA_ANY","TRIFECTA_BOX"])):
                for budget in (1,3,5,10,20,40,80):
                    q=train[(train.axis==axis)&train.family.isin(families)&
                            (train.avg_races_per_year>=345)&
                            (train.avg_tickets_per_year>=345)&
                            (train.tickets<=0.0+budget*3456*years)].copy()
                    q=q[(q.tickets/q.races_bought<=budget)]
                    if len(q)==0:continue
                    # Score favors profitability while penalizing extreme
                    # tiny-sample ticket strategies.
                    q["score"]=q.roi-10/(q.hit_races.clip(lower=1)**0.5)
                    q=q.sort_values(["score","races_bought","tickets"],ascending=[False,False,True])
                    sel=q.iloc[0]
                    test=df[(df.year==test_year)&(df.axis==axis)&
                            (df.order==sel["order"])&(df.k==sel["k"])&
                            (df.family==sel.family)&(df.price_floor==sel.price_floor)]
                    if len(test)!=1:continue
                    t=test.iloc[0]
                    testing.append(dict(test_year=test_year,trained_years="+".join(train_years),
                        axis=axis,category=category,max_tickets_per_race=budget,
                        chosen_family=sel.family,chosen_order=sel["order"],
                        chosen_k=int(sel.k),chosen_floor=float(sel.price_floor),
                        train_roi_pct=round(sel.roi,4),train_bet_races=int(sel.races_bought),
                        test_bet_races=int(t.races_bought),test_stake_yen=int(t.stake_yen),
                        test_return_yen=round(t.return_yen,2),
                        test_profit_yen=round(t.return_yen-t.stake_yen,2),
                        test_roi_pct=round(t.roi_pct,5),
                        test_coverage_pct=round(t.race_coverage_pct,5),
                        test_hits=int(t.hit_races)))
    return pd.DataFrame(testing)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True);ap.add_argument("--picks",required=True)
    ap.add_argument("--racewise",required=True);ap.add_argument("--backfill-root",required=True)
    ap.add_argument("--out-dir",required=True)
    a=ap.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    d,picks=get_inputs(a.predictions,a.picks)
    race_dtype={"race_id":str,**{name+"_horse":str for name in AXES if name!="EV"}}
    race_dtype["VALUE_EV_horse"]=str
    races=pd.read_csv(a.racewise,compression="infer",dtype=race_dtype)
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert_ok(set(races.year)==set(YEARS) and len(races)==TOTAL_RACES
              and races.race_id.is_unique,"unexpected year/race count")
    # Raw AI full-field ranking is carried by saved independent distill ranks
    # on the source used by the original rank comparison.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(lambda year:year_task(year,d,picks,races,
                              Path(a.backfill_root)),YEARS))
    agg,baseline,port,meta,price=summarize(records)
    all_df=dataframe_totals(agg,baseline,meta)
    assert_ok(not all_df.empty,"no ticket-level prices decoded")
    assert_ok(meta["2023-2025"]["races"]==TOTAL_RACES,"leak/coverage")
    stats=pd.DataFrame([dict(year=y,**dict(x)) for y,x in meta.items()])
    stats.to_csv(out/"coverage-and-data-integrity.csv",index=False)
    for tpl,(ret,hits,tickets) in BENCHMARKS.items():
        actual=baseline[("2023-2025",tpl)]
        assert_ok(actual["all_stake_yen"]==100*tickets and
                  round(actual["all_refund_yen"])==ret and
                  actual["all_hit_races"]==hits,
                  "prior no-floor settlement baseline diverged "+str(tpl)+" -> "+str(actual))
    forward=forward_test(all_df)
    forward.to_csv(out/"chronological-policy-selection-test.csv",index=False)
    combined=[]
    for (year,preset,cap),c in port.items():
        n=c["tickets"]
        combined.append(dict(year=year,preset=preset,cap_tickets_per_race=cap,
             race_bought=c["races_bought"],coverage_pct=round(100*c["races_bought"]/meta[year]["races"],4),
             hit_races=c["hit_races"],hit_tickets=c["hit_tickets"],
             tickets=n,stake_yen=100*n,return_yen=c["return_yen"],
             profit_yen=c["return_yen"]-100*n,
             roi_pct=round(c["return_yen"]/n,5) if n else None))
    pd.DataFrame(combined).to_csv(out/"cross-bet-type-fixed-portfolio.csv",index=False)
    price_rows=[dict(year=y,kind=k,**dict(c)) for (y,k),c in price.items()]
    pd.DataFrame(price_rows).to_csv(out/"price-coverage-by-bet-kind.csv",index=False)
    all_df.sort_values(["year","axis","order","k","family","price_floor"],inplace=True)
    all_df.to_csv(out/"all-bet-ticket-odds-floor-grid.csv.gz",index=False,compression="gzip")
    front=all_df[(all_df.year=="2023-2025")&
                 (all_df.axis=="EV")&(all_df.races_bought>=300)&
                 (all_df.avg_tickets_per_bought_race<=20)].sort_values(
                 ["roi_pct","races_bought"],ascending=[False,False])
    front.head(100).to_csv(out/"ev-anchor-top100-retrospective-diagnostics.csv",index=False)
    print("PRICE_GATE_COVERAGE",json.dumps({"total_races":meta["2023-2025"]["races"],
       "missing_final_odds":meta["2023-2025"]["races_missing_final_odds"],
       "raw_rank_fallback_races":meta["2023-2025"]["raw_order_unavailable_races"],
       "pricing":{k:dict(c) for (y,k),c in price.items() if y=="2023-2025"}},
       ensure_ascii=False),flush=True)
    print("PORTFOLIO",pd.DataFrame(combined).to_csv(index=False),flush=True)
    print("FORWARD_TEST",forward.to_csv(index=False),flush=True)
    print("RETROSPECTIVE_TOP20_EV",front.head(20).to_csv(index=False),flush=True)
    summary={"contract":"ALL_BET_KIND_PER_TICKET_ODDS_GATING_V1",
       "years":list(YEARS),"races":TOTAL_RACES,
       "buying_rule":"Only individual bet tickets with a quoted FINAL pre-settlement price above their own type specific min odds are bought",
       "two_way_analysis":"Fixed 2023-25 price floors, temporal policy selection 2023->2024 and 2023+2024->2025",
       "WARNING_FINAL_ODDS_RETROSPECTIVE":"FINAL historical odds are observed after betting closed. Not an executable live profitability claim",
       "WARNING_OPTIMISTIC_BEST":"High ROI retrospective best policies suffer multiple-testing selection bias. Only chronological test can be considered validation",
       "missing_final_odds_races":meta["2023-2025"]["races_missing_final_odds"],
       "raw_rank_fallback_races":meta["2023-2025"]["raw_order_unavailable_races"],
       "no_filtered_race_with_ticket":"A race is skipped only when zero candidate ticket passes its own odds floor",
       "bet_families":sorted(BET_TO_GROUP),
       "axes":AXES,"partner_rank_sources":ORDERS,
       "thresholds":FLOORS,"portfolio_presets":PRESETS,
       "all_strategies":len(set((x[1],x[2],x[3],x[4]) for x in all_df.itertuples(index=False,name=None))),
       "2026_sealed":True,"standard_cpu":True,"payout_archive_commit":"0d4af461b77f0060dd9c789dd2da646026fada47"}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print("ALL_KINDS_PRICE_GATE_COMPLETED",json.dumps(
        {"n_grid":len(all_df),"n_forward":len(forward),"n_portfolios":len(combined)}),flush=True)

if __name__=="__main__":main()
