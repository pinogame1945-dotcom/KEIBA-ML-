#!/usr/bin/env python3
"""L2 ticket-level red-to-black, multi-kind WALK FORWARD v1.

Anchor ALWAYS: frozen EV horse (ANCHOR_FULL). All 7 physical ticket families,
all price ranges, and union of top10 partners in AI_PROB / MARKET / AI_RAW.
No retrospectively profitable price-band filter, no 2026 data.
Every 2024 ticket is scored with 2023 only; every 2025 ticket is scored with
2023+2024 only. 2023 is training/warmup, not an OOS evaluation.
Legacy FINAL settlement-market odds ARE NOT known before race close. Strict
'price-conditional retrospective research' and not executable live proof.

Run on free GitHub standard CPU. No training data persisted. Compressed daily
official prices/refunds are loaded from pinned BACKFILL commit within runner,
and small aggregate report files only are committed.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import itertools
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import run_l2_value_anchor_bet_frontier_v1 as prior
import run_l2_ai_market_rank_betfrontier_v1 as ranklab
import run_l2_all_bet_types_ticket_odds_gate_v1 as pricelab
from build_l2_market_helpers_v1 import load_day,load_odds_day,horse_number_map
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,assert_ok

YEAR_TEXT=("2023","2024","2025")
TYPES=("WIN","PLACE","QUINELLA","WIDE","EXACTA","TRIO","TRIFECTA")
FLOORS=pricelab.FLOORS
MODES=("HIST","RANK","BLEND")
BUDGETS=(1,3,5,10,20,40)
PSEUDO=5.0
# Price-bin/structural metrics are trained ONLY on earlier years. No
# in-year refit and no outcome-derived loss feature at scoring time.

def span(d):
    return {"WIN":0,"PLACE":0,"QUINELLA":1,"WIDE":1,"EXACTA":1,
            "TRIO":2,"TRIFECTA":2}[d]

def empty():
    return Counter()

def add_stat(to,key,p,h):
    c=to[key];c["n"]+=1;c["hits"]+=h;c["expected"]+=min(.98,.8/p)

def gap_bucket(g):
    return 0 if g<-3 else 1 if g<-1 else 2 if g<1 else 3 if g<3 else 4

def bucket(price):
    return bisect_right(FLOORS,price)-1

def feature(ticket,price,anchor_no,n2id,mrank,prank,rrank):
    kind,seq=ticket
    partners=[n for n in seq if n!=anchor_no]
    assert len(partners)==len(seq)-1,(ticket,anchor_no)
    delta=sum(mrank[n]-prank[n] for n in partners)/max(1,len(partners))
    rawdelta=sum(mrank[n]-rrank[n] for n in partners)/max(1,len(partners))
    # Omit raw model tie hazards by using only immutable stable ranks.
    quality=(delta+rawdelta)/2.0
    orientation=(seq.index(anchor_no)+1 if kind in ("EXACTA","TRIFECTA") else 0)
    return (kind,orientation,bucket(price),gap_bucket(quality)),quality

def candidate_tickets(anchor_no,orders,no):
    # All tickets anchored on EV; include what original 152 styles can select
    # up to 10 partner horses from any ranking; no results-based price cut.
    partner=set()
    for order in ("AI_PROB","MARKET","AI_RAW"):
        partner.update(no[h] for h in orders[order] if no[h]!=anchor_no
                       for _ in [0] if True) if False else None
        partner.update([no[h] for h in orders[order] if no[h]!=anchor_no][:10])
    others=sorted(partner)
    # Physical tickets unique per race. For unordered types, use sorted tuple.
    for kind in ("WIN","PLACE"):
        yield (kind,(anchor_no,))
    for b in others:
        pair=tuple(sorted((anchor_no,b)))
        yield ("QUINELLA",pair)
        yield ("WIDE",pair)
        yield ("EXACTA",(anchor_no,b))
        yield ("EXACTA",(b,anchor_no))
    for b,c in itertools.combinations(others,2):
        trio=tuple(sorted((anchor_no,b,c)))
        yield ("TRIO",trio)
        for ordered in itertools.permutations((anchor_no,b,c),3):
            yield ("TRIFECTA",ordered)

def process_ticket_input(g,pick,race,pack,market):
    rank={
      "AI_PROB":pricelab.sort_horses(g,"AI_PROB"),
      "MARKET":pricelab.sort_horses(g,"MARKET"),
      "AI_RAW":g.sort_values(["distill_ai_rank","market_rank","_horse_id"],
                ascending=[True,True,True],kind="mergesort")["_horse_id"].astype(str).tolist()}
    no=horse_number_map(pack)
    assert_ok(set(rank["MARKET"])==set(rank["AI_PROB"])==set(rank["AI_RAW"]),"rank set drift")
    assert_ok(all(h in no for h in rank["MARKET"]),"horse no set missing")
    assert_ok(rank["AI_RAW"][0]==str(race["AI_RAW_TOP_horse"]),"raw rank drift")
    assert_ok(pick==str(race["VALUE_EV_horse"]),"EV axis drift")
    anchor=no[pick]
    mrank={no[h]:i+1 for i,h in enumerate(rank["MARKET"])}
    prank={no[h]:i+1 for i,h in enumerate(rank["AI_PROB"])}
    rrank={no[h]:i+1 for i,h in enumerate(rank["AI_RAW"])}
    refunds=prior.payouts(pack,race["race_id"])
    cards=[]
    missing=Counter()
    for ticket in candidate_tickets(anchor,rank,no):
        kind=ticket[0]
        price=pricelab.ticket_price(ticket,market)
        if price is None:
            missing[kind]+=1
            continue
        assert math.isfinite(price) and price>0
        key,quality=feature(ticket,price,anchor,None,mrank,prank,rrank)
        # One official payout per actual physical ticket. No target leakage:
        # settle is used only for training updates AFTER the test decisions.
        ret=round(float(refunds.get(ticket,0)))
        cards.append((ticket,price,ret,key,quality))
    assert_ok(len(cards)>0,"all ticket prices absent: "+race["race_id"])
    return cards,missing

def train_update(stats,card):
    ticket,price,ret,feat,quality=card
    k,position,pricebin,qbin=feat
    priorhit=.8/price
    for key in (("FULL",k,position,pricebin,qbin),
                ("BIN",k,position,pricebin),("TYPE",k,position)):
        add_stat(stats,key,price,int(ret>0))

def correction(stats,key):
    c=stats.get(key)
    if not c:return 1.0
    # 5 pseudo-winners at the reference market odds ratio.
    # Empirical Bayes shrinkage attenuates rare large payouts.
    return (c["hits"]+PSEUDO)/(c["expected"]+PSEUDO)

def ticket_score(card,stats,mode):
    ticket,price,ret,feat,quality=card
    kind,position,pricebin,qbin=feat
    ratio=(.65*correction(stats,("FULL",kind,position,pricebin,qbin))+
           .25*correction(stats,("BIN",kind,position,pricebin))+
           .10*correction(stats,("TYPE",kind,position)))
    # Rank-only baseline uses zero outcomes and never chooses profitable bins.
    rank_ratio=math.exp(max(-6,min(6,quality))*.045)
    if mode=="HIST":relative=ratio
    elif mode=="RANK":relative=rank_ratio
    elif mode=="BLEND":relative=ratio**.65*rank_ratio**.35
    else:raise ValueError(mode)
    return .8*relative

def stable(cards,stats,mode):
    # Quotes are post-close FINAL; never use truth/payout in sorting.
    return sorted(cards,key=lambda card:
        (-ticket_score(card,stats,mode),card[1],card[0]))

def step_race(counter,cards,race_count,score_tag,weights=None):
    if not cards:return
    n=0;got=0;h=0;types=set()
    for i,c in enumerate(cards):
        ticket,price,ret,feat,qual=c
        units=(weights[i] if weights else 1)
        assert units in (1,2,3)
        n+=units;got+=int(ret)*units
        h+=int(ret>0)
        types.add(ticket[0])
    counter["races_bought"]+=1
    counter["tickets"]+=len(cards)
    counter["stake_units"]+=n
    counter["hit_tickets"]+=h
    counter["hit_races"]+=int(h>0)
    counter["return_yen"]+=got
    counter["race_pnl_last"]=got-n*100
    # running peak drawdown tracked in chronological race order
    counter["equity"]+=got-n*100
    counter["peak"]=max(counter["peak"],counter["equity"])
    counter["max_drawdown_yen"]=max(counter["max_drawdown_yen"],
                                      counter["peak"]-counter["equity"])
    counter["family_count"]+=len(types)

def result(year,kind,mode,cap,stat):
    stake=100*stat["stake_units"];ret=stat["return_yen"]
    return dict(year=year,bet_family=kind,mode=mode,tickets_cap=cap,
      all_races=RACES_YEAR,races_bought=stat["races_bought"],
      coverage_pct=round(stat["races_bought"]/RACES_YEAR*100,8),
      tickets=stat["tickets"],units=stat["stake_units"],
      avg_tickets_per_bought_race=round(stat["tickets"]/stat["races_bought"],8)
                          if stat["races_bought"] else 0,
      hit_races=stat["hit_races"],race_hit_rate_pct=
        round(stat["hit_races"]/stat["races_bought"]*100,8)
                          if stat["races_bought"] else 0,
      hit_tickets=stat["hit_tickets"],ticket_hit_rate_pct=
        round(stat["hit_tickets"]/stat["tickets"]*100,8)
                          if stat["tickets"] else 0,
      stake_yen=stake,return_yen=ret,profit_yen=ret-stake,
      roi_pct=round(ret/stake*100,8) if stake else None,
      max_drawdown_yen=stat["max_drawdown_yen"],avg_types_per_race=
        round(stat["family_count"]/stat["races_bought"],8)
                            if stat["races_bought"] else 0)

def run_year(y,all_pred,picks,races,root,stats,output):
    data=all_pred[all_pred.year.eq(y)]
    groups=data.groupby("_race_id",sort=False)
    yr=races[races.year.eq(y)]
    assert_ok(len(yr)==RACES_YEAR,"year mismatch")
    by_race=yr.set_index("race_id")
    dates=yr.groupby("race_date")["race_id"].apply(list)
    all_rows=defaultdict(Counter)
    year_stats=defaultdict(Counter)
    missing=Counter()
    total=Counter()
    for day,ids in dates.items():
        pays=load_day(root/"data"/"daily"/f"{day}.jsonl.gz",set(ids))
        odds=load_odds_day(root/"data"/"odds"/"daily"/f"{day}.jsonl.gz",set(ids))
        for rid in sorted(ids):
            total["races"]+=1
            pack=pays.get(rid);market=odds.get(rid)
            assert_ok(pack is not None and market is not None,"missing pack or FINAL odds "+rid)
            assert_ok(market.get("odds_kind")=="FINAL" and market.get("source_status")=="result",
                      "odds state drift "+rid)
            rec=by_race.loc[rid].to_dict();rec["race_id"]=rid
            cards,omitted=process_ticket_input(groups.get_group(rid),
                picks["ANCHOR_FULL"][rid],rec,pack,market)
            missing.update(omitted)
            total["priced_tickets"]+=len(cards)
            by_kind=defaultdict(list)
            for card in cards:by_kind[card[0][0]].append(card)
            for kind,candidates in by_kind.items():
                step_race(all_rows[(kind,"POOL",0)],candidates,total["races"],"POOL")
            step_race(all_rows[("ALL","POOL",0)],cards,total["races"],"POOL")
            # No model for 2023: it is entirely WARMUP, NOT OOS.
            if y in (2024,2025):
                for mode in MODES:
                    ordered=stable(cards,stats,mode)
                    for k in BUDGETS:
                        sub=ordered[:k]
                        step_race(all_rows[("ALL",mode,k)],sub,total["races"],mode)
                        if mode=="BLEND" and k in (3,5,10):
                            weights=[1 if ticket_score(card,stats,mode)<1.02
                                   else (2 if ticket_score(card,stats,mode)<1.15 else 3)
                                   for card in sub]
                            step_race(all_rows[("ALL","BLEND_WEIGHTED",k)],sub,
                                      total["races"],"BLEND_WEIGHTED",weights)
                    for kind,candidates in by_kind.items():
                        ordered_type=[t for t in ordered if t[0][0]==kind]
                        for k in (1,3):
                            step_race(all_rows[(kind,mode,k)],ordered_type[:k],
                                      total["races"],mode)
            # Update separately; the entire y=2024 results must NOT influence
            # its own test policy scores. Passed immutable previous-year stats.
            for card in cards:train_update(year_stats,card)
            if total["races"]%900==0:
                print("BATCH_YEAR_PROGRESS",y,total["races"],"priced",total["priced_tickets"],flush=True)
    assert_ok(total["races"]==RACES_YEAR,"year count drift")
    print("YEAR_COMPLETED",json.dumps({"year":y,"races":total["races"],
       "priced_tickets":total["priced_tickets"],"missing_quote_tickets":dict(missing),
       "portfolio_policies":len(all_rows)},ensure_ascii=False),flush=True)
    return year_stats,all_rows,total,missing

def merge_hist(prev,current):
    # The historical expected/hit statistics are additive; model NEVER sees
    # 2025 labels until 2025 is fully scored and discarded.
    for key,c in current.items():prev[key].update(c)

def export_csv(path,records):
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--picks",required=True)
    ap.add_argument("--racewise",required=True)
    ap.add_argument("--backfill-root",required=True)
    ap.add_argument("--out-dir",required=True)
    a=ap.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    data,picks=ranklab.load_predictions(a.predictions,a.picks)
    race_dtype={"race_id":str,"VALUE_EV_horse":str,"AI_RAW_TOP_horse":str}
    races=pd.read_csv(a.racewise,compression="infer",dtype=race_dtype)
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert_ok(set(races.year)==set(YEARS) and len(races)==TOTAL_RACES
              and races.race_id.is_unique,"racewise manifest drift")
    stats=defaultdict(Counter)
    rows=[];run_manifest={}
    for y in YEARS:
        year_model,results,metadata,missing=run_year(y,data,picks,races,
            Path(a.backfill_root),stats,out)
        for (kind,mode,k),v in results.items():
            rows.append(result(str(y),kind,mode,k,v))
        run_manifest[str(y)]=dict(priced_tickets=metadata["priced_tickets"],
                                 missing_quotes=dict(missing),
                                 model_training_years=[yy for yy in YEARS if yy<y])
        merge_hist(stats,year_model)
    rows.sort(key=lambda x:(x["year"],x["bet_family"],x["mode"],x["tickets_cap"]))
    export_csv(out/"all-policies-by-year.csv",rows)
    # Pool test 2024+2025 only, NEVER 2023 training.
    pooled=defaultdict(Counter)
    # Profit / money only; drawdowns do not add and are deliberately omitted
    for x in rows:
        if x["year"] not in ("2024","2025"):continue
        key=(x["bet_family"],x["mode"],x["tickets_cap"])
        for f in ("races_bought","tickets","units","hit_races",
                  "hit_tickets","stake_yen","return_yen","profit_yen"):
            pooled[key][f]+=x[f]
    combined=[]
    for (kind,mode,k),v in pooled.items():
        st=v["stake_yen"];n=v["tickets"];br=v["races_bought"]
        combined.append(dict(year="2024-2025_OOS",bet_family=kind,mode=mode,
             tickets_cap=k,all_races=6912,races_bought=br,
             coverage_pct=round(br/6912*100,8),
             tickets=n,units=v["units"],
             avg_tickets_per_bought_race=round(n/br,8) if br else 0,
             hit_races=v["hit_races"],race_hit_rate_pct=round(v["hit_races"]/br*100,8) if br else 0,
             hit_tickets=v["hit_tickets"],ticket_hit_rate_pct=
               round(v["hit_tickets"]/n*100,8) if n else 0,
             stake_yen=st,return_yen=v["return_yen"],profit_yen=v["profit_yen"],
             roi_pct=round(v["return_yen"]/st*100,8) if st else None,
             max_drawdown_yen=None,avg_types_per_race=None))
    combined.sort(key=lambda x:(x["bet_family"],x["mode"],x["tickets_cap"]))
    export_csv(out/"all-policies-2024-2025-oos-combined.csv",combined)
    # Explicit non-cherry-picked red->black check: pooled every-ticket baseline
    # is below 100%; learned fixed-rank budgets test 2024 and 2025 separately.
    converted=[]
    idx={(x["year"],x["bet_family"],x["mode"],x["tickets_cap"]):x for x in rows}
    for kind in (*TYPES,"ALL"):
        p=next((x for x in combined if x["bet_family"]==kind and x["mode"]=="POOL"),None)
        if not p:continue
        for r in combined:
            if r["bet_family"]!=kind or r["mode"]=="POOL":continue
            t24=idx[("2024",kind,r["mode"],r["tickets_cap"])]
            t25=idx[("2025",kind,r["mode"],r["tickets_cap"])]
            converted.append(dict(bet_family=kind,mode=r["mode"],tickets_cap=r["tickets_cap"],
                raw_pool_roi_2024_25=p["roi_pct"],
                optimized_roi_2024_25=r["roi_pct"],
                roi_2024=t24["roi_pct"],roi_2025=t25["roi_pct"],
                changed_red_to_black=bool(p["roi_pct"]<100 and r["roi_pct"]>100),
                both_oos_years_profit=bool(t24["profit_yen"]>0 and t25["profit_yen"]>0),
                bought_races=r["races_bought"],coverage_pct=r["coverage_pct"],
                tickets=r["tickets"],hit_tickets=r["hit_tickets"],
                profit_yen=r["profit_yen"],stake_yen=r["stake_yen"]))
    export_csv(out/"red-to-black-full-family-and-ticket-budget-audit.csv",converted)
    # Do not pick a final winner by test results; publish entire model/policy
    # grid (incl negative result), pre-defined before outcome evaluation.
    manifest=dict(contract="L2_EV_TICKET_LEVEL_RED_TO_BLACK_WALK_FORWARD_V1",
       evaluation_years=[2024,2025],training_year=2023,
       validation_2024_training=[2023],validation_2025_training=[2023,2024],
       year_2023_evaluation="NO: warmup training only",year_2026_sealed=True,
       races=TOTAL_RACES,eval_races=RACES_YEAR*2,
       anchor="ANCHOR_FULL expected-value-ranked top1 frozen",
       candidate_pool="ALL 7 types, ALL odds, anchored union of top10 partners from AI_PROB MARKET AI_RAW; no outcome-ranked blacklist",
       methods=MODES,budgets=BUDGETS,weighted_blend_caps=[3,5,10],
       actual_settlement="official payouts per physical ticket",
       final_odds_warning="Uses historical FINAL odds (observed only after betting closes). These are not executable live results",
       conditioned_selection_warning="2023 trained 2024 tested, 2023+2024 trained 2025 tested. Never calibrates on test-year payouts before scoring",
       no_charge="GitHub Actions standard CPU, temporary pinned BACKFILL, no paid artifacts/cache/data",
       data=run_manifest,all_policy_rows=len(rows),test_aggregate_rows=len(combined),
       red_to_black_count=sum(r["changed_red_to_black"] for r in converted),
       both_years_positive_count=sum(r["both_oos_years_profit"] for r in converted),
       selected_final_policy="NONE: no post-hoc OOS cherry-pick")
    (out/"summary.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("OOS_2024_2025_POLICY_TABLE",json.dumps([
      x for x in combined if x["bet_family"]=="ALL"],ensure_ascii=False),flush=True)
    print("RED_TO_BLACK_CANDIDATES",json.dumps(
        [r for r in converted if r["changed_red_to_black"]],
        ensure_ascii=False),flush=True)
    print("L2_EV_TICKET_LEVEL_RED_TO_BLACK_COMPLETE",json.dumps({
       "evaluated_years":[2024,2025],"rows":len(rows),
       "combo_rows":len(combined),
       "red_to_black_policies":manifest["red_to_black_count"],
       "both_years_positive":manifest["both_years_positive_count"]
       },ensure_ascii=False),flush=True)
if __name__=="__main__":main()
