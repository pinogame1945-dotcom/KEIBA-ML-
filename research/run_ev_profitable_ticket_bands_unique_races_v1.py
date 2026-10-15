#!/usr/bin/env python3
"""Exact unique-race and unique-ticket coverage of observed profitable odds bands.

Targets ALL 3-year positive exact 20-bin EV-anchor ticket rules (not only
three consecutive positive years), then tests individual rule race counts,
break-even ticket hit rates, and actual UNION portfolios.

The 2023/24/25 per-ticket profit table is archived and fixed before this scan.
This is descriptive in-sample analysis, not an OOS betting policy recommendation.
"""
import argparse
import concurrent.futures
import csv
import json
from bisect import bisect_right
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path

import pandas as pd
import run_l2_all_bet_types_ticket_odds_gate_v1 as odds_lab
import run_l2_value_anchor_bet_frontier_v1 as prior
import run_l2_ai_market_rank_betfrontier_v1 as priors_rank
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,assert_ok
from build_l2_market_helpers_v1 import load_day,load_odds_day,horse_number_map

STEPS=(0,1.5,2,3,5,7,10,15,20,30,50,75,100,150,200,300,500,1000,2000,5000)
CATEGORIES=("THREE_YEAR_TOTAL_PLUS","ALL_THREE_YEARS_PLUS",
            "THREE_YEAR_PLUS_N500_H10","ALL_THREE_YEARS_PLUS_N500_H10")
YEARS_TEXT=tuple(str(x) for x in YEARS)
YEAR_TOTAL="2023-2025"
BET_FAMILY={
    "WIN":"WIN","PLACE":"PLACE","QUINELLA":"QUINELLA","WIDE":"WIDE",
    "EXACTA_FIRST":"EXACTA","EXACTA_SECOND":"EXACTA","EXACTA_BOTH":"EXACTA",
    "TRIO":"TRIO","TRIFECTA_FIRST":"TRIFECTA","TRIFECTA_SECOND":"TRIFECTA",
    "TRIFECTA_THIRD":"TRIFECTA","TRIFECTA_ANY":"TRIFECTA"
}

def yen(s):
    v=Decimal(str(s))
    assert v==v.to_integral_value(),s
    return int(v)

def decimal_pct(n,d):
    if not d:return ""
    return str(Decimal(n)*Decimal(100)/Decimal(d))

def validate_bin(band):
    lo=float(band[:-1]) if band.endswith("+") else float(band.split("-")[0])
    assert lo in STEPS,("unknown odds band",band)
    hi=float("inf") if band.endswith("+") else float(band.split("-")[1])
    ix=STEPS.index(lo)
    assert hi==(STEPS[ix+1] if ix+1<len(STEPS) else float("inf")),(band,lo,hi)
    return lo,hi

def groups(path):
    with open(path,encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==152*20,len(rows)
    selected={}
    for r in rows:
        parts=r["買い方ID"].split("|")
        assert len(parts)==4 and parts[0]=="EV",parts
        order,k,family=parts[1],int(parts[2]),parts[3]
        band=r["馬券オッズ帯"]
        lo,hi=validate_bin(band)
        tid=(order,k,family,band)
        assert tid not in selected,tid
        record={"tpl":(order,k,family),"band":band,"lo":lo,"hi":hi,
                "race_count":{}, "original":r}
        record["plus_all"]=all(yen(r[f"{y}_損益円"])>0 for y in YEARS_TEXT)
        record["plus_total"]=yen(r[f"{YEAR_TOTAL}_損益円"])>0
        record["robust"]=(int(r[f"{YEAR_TOTAL}_購入点数"])>=500
                          and int(r[f"{YEAR_TOTAL}_的中点数"])>=10)
        selected[tid]=record
    positive={tid:r for tid,r in selected.items() if r["plus_total"]}
    assert len(positive)==245,len(positive)
    assert sum(r["plus_all"] for r in positive.values())==11
    assert sum(r["robust"] for r in positive.values())==39
    assert sum(r["plus_all"] and r["robust"] for r in positive.values())==7
    return positive,selected

def category_match(r,category):
    if category=="THREE_YEAR_TOTAL_PLUS":return True
    if category=="ALL_THREE_YEARS_PLUS":return r["plus_all"]
    if category=="THREE_YEAR_PLUS_N500_H10":return r["robust"]
    if category=="ALL_THREE_YEARS_PLUS_N500_H10":
        return r["plus_all"] and r["robust"]
    raise ValueError(category)

def evaluate_year(year,all_pred,picks,races,root,positive):
    pred=all_pred.loc[all_pred.year.eq(year)]
    grp=pred.groupby("_race_id",sort=False)
    ar=races.loc[races.year.eq(year)]
    assert_ok(len(ar)==RACES_YEAR,"races drift")
    ar_idx=ar.set_index("race_id")
    dates=ar.groupby("race_date")["race_id"].apply(list)
    # Build each family/partner-group tickets only ONCE per race.
    selections=defaultdict(list)
    for tid,r in positive.items():
        selections[r["tpl"]].append((tid,r["lo"],r["hi"]))
    stats=defaultdict(Counter)
    portfolio=defaultdict(Counter)
    meta=Counter()
    for day,ids in dates.items():
        pack_by_id=load_day(root/"data"/"daily"/f"{day}.jsonl.gz",set(ids))
        odds_by_id=load_odds_day(root/"data"/"odds"/"daily"/f"{day}.jsonl.gz",set(ids))
        for rid in ids:
            meta["races"]+=1
            pack=pack_by_id.get(rid)
            market=odds_by_id.get(rid)
            assert_ok(pack is not None and market is not None,"missing official race/odds: "+str(rid))
            g=grp.get_group(rid)
            anchor=picks["ANCHOR_FULL"][rid]
            info=ar_idx.loc[rid]
            ranked={
              "AI_PROB":odds_lab.sort_horses(g,"AI_PROB"),
              "MARKET":odds_lab.sort_horses(g,"MARKET"),
              "AI_RAW":g.sort_values(["distill_ai_rank","market_rank","_horse_id"],
                        ascending=[True,True,True],kind="mergesort")["_horse_id"].astype(str).tolist()}
            assert_ok(ranked["AI_RAW"][0]==str(info["AI_RAW_TOP_horse"]),"raw order")
            assert_ok(anchor==str(info["VALUE_EV_horse"]),"anchor selection drift")
            no=horse_number_map(pack)
            assert_ok(set(ranked["AI_RAW"])==set(ranked["AI_PROB"])==set(ranked["MARKET"]),
                      "ranked horse set mismatch")
            assert_ok(all(h in no for h in ranked["AI_RAW"]),"missing horse no")
            refunds=prior.payouts(pack,rid)
            axno=no[anchor]
            # Each physical ticket is independent but can be selected by many strategies.
            prices_and_refunds={}
            category_tickets={key:set() for key in CATEGORIES}
            def result_for(ticket):
                if ticket not in prices_and_refunds:
                    price=odds_lab.ticket_price(ticket,market)
                    ret=refunds.get(ticket,0)
                    prices_and_refunds[ticket]=(price,ret)
                return prices_and_refunds[ticket]
            for (order,k,fam),conditions in selections.items():
                if order=="SINGLE":
                    template={(fam,(axno,))}
                else:
                    others=[no[h] for h in ranked[order] if h!=anchor][:k]
                    template=prior.pair_templates(axno,others)[fam]
                # O(n ticket) via bisect then only matching profitable band,
                # rather than loops over all 20 groups.
                condition_by_low={lo:(tid,hi) for tid,lo,hi in conditions}
                local=defaultdict(lambda:[0,0,0])
                for ticket in template:
                    price,ret=result_for(ticket)
                    if price is None:continue
                    l=STEPS[bisect_right(STEPS,price)-1] if price>=0 else None
                    selected=condition_by_low.get(l)
                    if selected is None:continue
                    tid,high=selected
                    assert_ok(price<high or high==float("inf"),"band mismatch")
                    s=local[tid]
                    s[0]+=1
                    s[1]+=int(ret>0)
                    s[2]+=int(round(ret))
                    for cat in CATEGORIES:
                        if category_match(positive[tid],cat):
                            category_tickets[cat].add(ticket)
                for tid,(n,hits,ret) in local.items():
                    c=stats[tid]
                    c["races_bought"]+=1
                    c["tickets"]+=n
                    c["hit_tickets"]+=hits
                    c["hit_races"]+=int(hits>0)
                    c["return_yen"]+=ret
            for cat,ts in category_tickets.items():
                c=portfolio[cat]
                if not ts:continue
                hit=0;ret=0
                for ticket in ts:
                    _,returned=result_for(ticket)
                    hit+=int(returned>0)
                    ret+=int(round(returned))
                c["races_bought"]+=1
                c["hit_races"]+=int(hit>0)
                c["tickets"]+=len(ts)
                c["hit_tickets"]+=hit
                c["return_yen"]+=ret
            if meta["races"]%1200==0:
                print("YEAR_PROGRESS",year,meta["races"],flush=True)
    assert_ok(meta["races"]==RACES_YEAR,"race count")
    return year,stats,portfolio,meta

def make_row(tid,year,c,r,all_races):
    n=c["tickets"];h=c["hit_tickets"];ns=c["races_bought"];rh=c["hit_races"]
    total_ret=c["return_yen"]
    return {"year":year,"odds_band":r["band"],
        "order":r["tpl"][0],"other_horses_k":r["tpl"][1],"family":r["tpl"][2],
        "three_year_positive":r["plus_total"],"three_years_each_positive":r["plus_all"],
        "samples_500_tickets_10_hits":r["robust"],
        "races_bought":ns,
        "races_fraction_unevaluated":all_races-ns,
        "race_coverage_pct":decimal_pct(ns,all_races),
        "hit_races":rh,"race_hit_rate_pct":decimal_pct(rh,ns),
        "tickets":n,"hit_tickets":h,"ticket_hit_rate_pct":decimal_pct(h,n),
        "tickets_per_bought_race":str(Decimal(n)/Decimal(ns)) if ns else "",
        "stake_yen":n*100,"return_yen":total_ret,"profit_yen":total_ret-n*100,
        "roi_pct":decimal_pct(total_ret,n*100),
        "mean_winning_payout_yen":decimal_pct(total_ret,h*100) if h else "",
        "break_even_ticket_hit_rate_pct":decimal_pct(n*100*h,total_ret*n) if h and total_ret else "",
        "hit_excess_percentage_points":(
           str(Decimal(100)*Decimal(h)/Decimal(n)-Decimal(10000)*Decimal(h)/Decimal(total_ret))
           if n and h and total_ret else ""),
        }
# Note: mean winning payout multiplier = ret/(100*hits).
# Break-even ticket hit rate (%): 100*100*hits/ret.
def fix_math(row):
    h=row["hit_tickets"];ret=row["return_yen"];n=row["tickets"]
    row["mean_winning_dividend_multiplier"]=str(Decimal(ret)/Decimal(100*h)) if h else ""
    row["break_even_ticket_hit_rate_pct"]=str(Decimal(10000)*Decimal(h)/Decimal(ret)) if h and ret else ""
    row["ticket_hit_rate_pct"]=str(Decimal(100)*Decimal(h)/Decimal(n)) if n else ""
    row["hit_excess_percentage_points"]=str(
        Decimal(100)*Decimal(h)/Decimal(n)-Decimal(10000)*Decimal(h)/Decimal(ret)
        ) if n and h and ret else ""
    row.pop("mean_winning_payout_yen",None)
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--picks",required=True)
    ap.add_argument("--racewise",required=True)
    ap.add_argument("--backfill-root",required=True)
    ap.add_argument("--bands",required=True)
    ap.add_argument("--out-dir",required=True)
    args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    positive,all_rules=groups(args.bands)
    d,picks=priors_rank.load_predictions(args.predictions,args.picks)
    race_dtype={"race_id":str,
                "VALUE_EV_horse":str,
                "AI_RAW_TOP_horse":str}
    races=pd.read_csv(args.racewise,compression="infer",dtype=race_dtype)
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert_ok(set(races.year)==set(YEARS) and len(races)==TOTAL_RACES
              and races.race_id.is_unique,"racewise drift")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        parts=list(pool.map(lambda y:evaluate_year(y,d,picks,races,
             Path(args.backfill_root),positive),YEARS))
    totals=defaultdict(Counter)
    portfolio=defaultdict(Counter)
    meta=Counter()
    for y,stats,port,mm in parts:
        for tid,c in stats.items():totals[tid].update(c)
        for cat,c in port.items():portfolio[cat].update(c)
        meta.update(mm)
    assert_ok(meta["races"]==TOTAL_RACES,"full coverage")
    # Every profitable rule MUST match the exact archived ticket/settlement
    # baseline; race coverage is only newly measured attribute.
    rows=[]
    for tid,r in sorted(positive.items()):
        observed=totals[tid]
        original=r["original"]
        assert_ok(observed["tickets"]==int(original[f"{YEAR_TOTAL}_購入点数"]),
                  "source tickets mismatch "+str(tid))
        assert_ok(observed["hit_tickets"]==int(original[f"{YEAR_TOTAL}_的中点数"]),
                  "source hits mismatch "+str(tid))
        assert_ok(observed["return_yen"]==yen(original[f"{YEAR_TOTAL}_払戻円"]),
                  "source payouts mismatch "+str(tid))
        assert_ok(observed["races_bought"]<=TOTAL_RACES,"race overcount")
        for year,stats,port,mm in parts:
            c=stats[tid]
            assert_ok(c["tickets"]==int(original[f"{year}_購入点数"]),
                      "source annual tickets "+str(tid)+" "+str(year))
            assert_ok(c["return_yen"]==yen(original[f"{year}_払戻円"]),
                      "source annual payouts "+str(tid)+" "+str(year))
            rows.append(fix_math(make_row(tid,str(year),c,r,RACES_YEAR)))
        rows.append(fix_math(make_row(tid,YEAR_TOTAL,observed,r,TOTAL_RACES)))
    df=pd.DataFrame(rows)
    df.to_csv(out/"all-3year-positive-ticket-odds-bands-with-unique-races.csv",index=False)
    df[df.three_years_each_positive].to_csv(
        out/"all-three-years-profitable-bands-race-coverage.csv",index=False)
    df[df.samples_500_tickets_10_hits].to_csv(
        out/"sampled-3year-positive-bands-race-coverage.csv",index=False)
    cats=[]
    for cat in CATEGORIES:
        for year,stats,port,mm in [*parts,(YEAR_TOTAL,None,portfolio,meta)]:
            c=port[cat]
            n=c["tickets"]
            r={"group":cat,"year":str(year),
               "included_rules":sum(category_match(z,cat) for z in positive.values()),
               "races_all":mm["races"],"races_bought":c["races_bought"],
               "race_coverage_pct":decimal_pct(c["races_bought"],mm["races"]),
               "hit_races":c["hit_races"],"race_hit_rate_pct":decimal_pct(c["hit_races"],c["races_bought"]),
               "unique_tickets":n,"hit_tickets":c["hit_tickets"],
               "ticket_hit_rate_pct":decimal_pct(c["hit_tickets"],n),
               "tickets_per_bought_race":str(Decimal(n)/Decimal(c["races_bought"])) if c["races_bought"] else "",
               "stake_yen":n*100,"return_yen":c["return_yen"],
               "profit_yen":c["return_yen"]-n*100,
               "roi_pct":decimal_pct(c["return_yen"],n*100)}
            cats.append(r)
    pd.DataFrame(cats).to_csv(out/"union-unique-races-and-deduped-purchases-by-year.csv",index=False)
    print("ALL_PROFIT_CATEGORIES_UNIQUE_UNION",json.dumps(
      [x for x in cats if x["year"]==YEAR_TOTAL],ensure_ascii=False),flush=True)
    focus=df[df.three_years_each_positive & (df.year==YEAR_TOTAL)].sort_values(
      "profit_yen",ascending=False)
    print("ALL_11_CONSECUTIVE_PROFIT_RULES",focus.to_csv(index=False),flush=True)
    by_kind=df[(df.year==YEAR_TOTAL)&(df.samples_500_tickets_10_hits)]
    print("THREE_YEAR_POSITIVE_39_BY_TYPE",
        json.dumps(by_kind.groupby("family").agg(
             candidate_bands=("family","size"),
             sum_stakes_non_dedup=("stake_yen","sum"),
             sum_profits_non_dedup=("profit_yen","sum")).reset_index().to_dict("records"),
             ensure_ascii=False),flush=True)
    (out/"summary.json").write_text(json.dumps({
        "years":list(YEARS),"races":TOTAL_RACES,
        "price_bands":20,
        "all_rules_examined":len(all_rules),
        "3_year_positive_rules":len(positive),
        "3_consecutive_positive_rules":sum(r["plus_all"] for r in positive.values()),
        "positive_min500_10hits_rules":sum(r["robust"] for r in positive.values()),
        "consecutive_min500_10hits_rules":sum(r["robust"] and r["plus_all"] for r in positive.values()),
        "union_definitions":{
            "THREE_YEAR_TOTAL_PLUS":"All 245 retrospectively profitable rule/price combinations",
            "ALL_THREE_YEARS_PLUS":"All 11 rule/price combinations that were profitable in 2023, 2024, and 2025 individually",
            "THREE_YEAR_PLUS_N500_H10":"39 3-year pooled profitable combinations with >=500 tickets and >=10 winning tickets",
            "ALL_THREE_YEARS_PLUS_N500_H10":"7 3-year individually profitable combinations with >=500 tickets and >=10 hits"},
        "portfolio_note":"Union deduplicates the same physical bet (kind, horse combination) inside each race; exact unit 100 yen; per-template lines overlap",
        "retrospective_selection_warning":"These lines are chosen after seeing 2023-2025 returns; not a deployable future profitable strategy",
        "odds_warning":"Uses historically archived final odds; pre-closing tradeability not established",
        "year_2026_sealed":True,"cost":"GitHub Actions standard CPU; no purchased resources"
       },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("UNIQUE_RACES_COMPLETE",flush=True)

if __name__=="__main__":main()
