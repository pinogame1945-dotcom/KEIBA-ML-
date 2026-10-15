#!/usr/bin/env python3
"""Frozen AI value-anchor, cross-bet hit/cost/return frontier (2023-2025).
All tickets must physically contain one preselected horse. Fixed odds are
retrospective selection context ONLY: cash returns use normalized race payouts.
Strategies are FIXED across races; oracle rows are labelled POST-RACE / nontradeable.
"""
import argparse
import concurrent.futures
import itertools
import json
import re
from collections import Counter,defaultdict
from pathlib import Path
import pandas as pd

from run_all_race_flow_ticket_audit_v1 import (
    YEARS, TOTAL_RACES, RACES_YEAR, SOURCES, PARTNERS, get_inputs, assert_ok
)
from build_l2_market_helpers_v1 import load_day, horse_number_map
from run_value_anchor_official_positions_v1 import positions

SHA="0d4af461b77f0060dd9c789dd2da646026fada47"
BET_ARITY={"WIN":1,"PLACE":1,"QUINELLA":2,"WIDE":2,
           "EXACTA":2,"TRIO":3,"TRIFECTA":3}
UNORDERED={"QUINELLA","WIDE","TRIO"}
FAMILIES=("WIN","PLACE","QUINELLA","WIDE","EXACTA_FIRST","EXACTA_SECOND",
          "EXACTA_BOTH","TRIO","TRIFECTA_FIRST","TRIFECTA_SECOND",
          "TRIFECTA_THIRD","TRIFECTA_ANY")
PARTNER_COUNTS=(2,3,5,7)
BUCKETS=("FIRST","SECOND","THIRD","FOURTH_PLUS","NO_FINISH")

def num_from_combination(raw,arity):
    nums=tuple(map(int,re.findall(r"\d+",str(raw))))
    if len(nums)!=arity or any(x<1 or x>18 for x in nums):
        return None
    if arity>=2 and len(set(nums))!=len(nums):
        return None
    return nums

def payouts(pack,rid):
    out={}
    seen=set()
    for row in pack.get("payouts") or []:
        bet=str(row.get("bet_type") or "")
        if bet not in BET_ARITY:
            continue
        seen.add(bet)
        tup=num_from_combination(row.get("combination"),BET_ARITY[bet])
        if tup is None:
            continue
        if bet in UNORDERED:
            tup=tuple(sorted(tup))
        try:
            val=float(str(row.get("payout_yen")).replace(",",""))
        except (TypeError,ValueError):
            continue
        if not (0 < val < 1e10):
            continue
        assert_ok((bet,tup) not in out, f"{rid}: duplicate paid ticket {bet} {tup}")
        out[(bet,tup)]=val
    missing=set(BET_ARITY)-seen
    assert_ok(not missing,f"{rid}: missing entire payout categories {sorted(missing)}")
    for kind in BET_ARITY:
        assert_ok(any(k[0]==kind for k in out),f"{rid}: no usable {kind} payout")
    return out

def pair_templates(axis,others):
    base=[(axis,b) for b in others]
    results={
      "QUINELLA":{("QUINELLA",tuple(sorted(x))) for x in base},
      "WIDE":{("WIDE",tuple(sorted(x))) for x in base},
      "EXACTA_FIRST":{("EXACTA",x) for x in base},
      "EXACTA_SECOND":{("EXACTA",(b,axis)) for _,b in base},
    }
    results["EXACTA_BOTH"]=results["EXACTA_FIRST"]|results["EXACTA_SECOND"]
    triples=list(itertools.combinations(others,2))
    results["TRIO"]={
      ("TRIO",tuple(sorted((axis,b,c)))) for b,c in triples
    }
    results["TRIFECTA_FIRST"]={
      ("TRIFECTA",(axis,x,y)) for b,c in triples for x,y in ((b,c),(c,b))
    }
    results["TRIFECTA_SECOND"]={
      ("TRIFECTA",(x,axis,y)) for b,c in triples for x,y in ((b,c),(c,b))
    }
    results["TRIFECTA_THIRD"]={
      ("TRIFECTA",(x,y,axis)) for b,c in triples for x,y in ((b,c),(c,b))
    }
    results["TRIFECTA_ANY"]=(
      results["TRIFECTA_FIRST"]|
      results["TRIFECTA_SECOND"]|
      results["TRIFECTA_THIRD"]
    )
    return results

def update(ct,n,return_yen,rank):
    assert_ok(n>=0,"negative tickets")
    ct["races"]+=1
    ct["tickets"]+=n
    ct["stake_yen"]+=100*n
    ct["return_yen"]+=return_yen
    ct["hit_races"]+=int(return_yen>0)
    ct["hit_tickets"]+=int(return_yen>0) # one paid hit in typical race; multi payouts audited below
    ct["bet_races"]+=int(n>0)
    ct["miss_races"]+=int(n>0 and return_yen==0)
    ct["stake_"+rank]+=100*n
    ct["return_"+rank]+=return_yen

def one_year(year,data,picks,root):
    df=data[data.year.eq(year)].copy()
    df["_race_date"]=df["_race_date"].astype(str).str[:10]
    by_date=df[["_race_id","_race_date"]].drop_duplicates().groupby("_race_date")
    groups=df.groupby("_race_id",sort=False)
    aggregate=defaultdict(Counter)
    ceilings=[]
    records=0
    for date,day_races in by_date:
        ids=day_races["_race_id"].astype(str).tolist()
        packs=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",set(ids))
        for rid in ids:
            pack=packs.get(rid)
            assert_ok(pack is not None,f"{year}: race pack absent {rid}")
            group=groups.get_group(rid)
            horse_num=horse_number_map(pack)
            axis=picks["ANCHOR_FULL"][rid]
            assert_ok(axis in horse_num,f"{rid}: selected AI horse missing number")
            nums=set(group["_horse_id"].astype(str))
            assert_ok(nums.issubset(horse_num),f"{rid}: missing horse numbers")
            p=payouts(pack,rid)
            rank=positions(pack).get(axis)
            assert_ok(rank is not None, f"{rid}: missing selected official finish")
            win=int(group.loc[group["_horse_id"].eq(axis),"target_win"].iloc[0])
            place=int(group.loc[group["_horse_id"].eq(axis),"target_top3"].iloc[0])
            assert_ok(win==int(rank==1) and place==int(rank<=3),
                      f"{rid}: historical prediction target mismatch")
            r_bucket={1:"FIRST",2:"SECOND",3:"THIRD"}.get(rank,"FOURTH_PLUS")
            axis_no=horse_num[axis]
            pays_for_axis=[(kind,tuple(t),price) for (kind,t),price in p.items()
                           if axis_no in t]
            oracle_max=max([price for _,_,price in pays_for_axis],default=0)
            oracle_kind=sorted(pays_for_axis,key=lambda row:(-row[2],row[0]))[0][0] if pays_for_axis else "NONE"
            ceiling_by_type={kind:max((amt for typ,_,amt in pays_for_axis if typ==kind),default=0)
                             for kind in BET_ARITY}
            if rank<=3:
                assert_ok(oracle_max>0,f"{rid}: top3 selected horse but no payout")
            else:
                assert_ok(oracle_max==0,f"{rid}: selected 4+ unexpectedly paid")
            ceilings.append({
              "year":year,"race_id":rid,"selected_horse_id":axis,
              "selected_horse_number":axis_no,"selected_finish":rank,
              "selected_finish_bucket":r_bucket,
              "best_possible_paid_ticket_yen":oracle_max,
              "best_possible_paid_ticket_kind":oracle_kind,
              **{"max_paid_"+kind+"_yen":amt for kind,amt in ceiling_by_type.items()},
            })
            for family in ("WIN","PLACE"):
                key=(family,"SINGLE",0)
                value=p.get((family,(axis_no,)),0)
                update(aggregate[key],1,value,r_bucket)
            for source in SOURCES:
                ranking=(group.sort_values(["market_rank","final_win_odds","_horse_id"],
                          ascending=[True,True,True]) if source=="MARKET" else
                         group.sort_values(["P__ANCHOR_FULL","market_rank","_horse_id"],
                          ascending=[False,True,True]))
                others=[horse_num[hid] for hid in ranking["_horse_id"].astype(str)
                        if hid!=axis]
                for n in PARTNER_COUNTS:
                    tickets_by_family=pair_templates(axis_no,others[:n])
                    for family,tickets in tickets_by_family.items():
                        assert_ok(all(axis_no in combo for _,combo in tickets),
                                  f"{rid}: ticket omits anchor {family}")
                        rewards=[p.get(ticket,0) for ticket in tickets]
                        update(aggregate[(family,source,n)],len(tickets),sum(rewards),r_bucket)
                        # Save multi-deadheat wins transparently for audit only.
                        aggregate[(family,source,n)]["extra_hit_tickets"]+=max(0,sum(v>0 for v in rewards)-int(sum(rewards)>0))
            records+=1
    assert_ok(records==RACES_YEAR,f"{year}: {records} races != {RACES_YEAR}")
    return aggregate,ceilings

def agg_years(chunks):
    accum=defaultdict(Counter)
    for year,(totals,_) in zip(YEARS,chunks):
        for (family,source,n),counter in totals.items():
            accum[(str(year),family,source,n)].update(counter)
            accum[("2023-2025",family,source,n)].update(counter)
    summary=[]
    for (year,family,source,k),c in accum.items():
        nr=c["races"]
        assert_ok(nr==TOTAL_RACES if year=="2023-2025" else nr==RACES_YEAR,
                  f"{year}/{family}/{source}/{k}: incomplete race coverage")
        cost=c["stake_yen"];gross=c["return_yen"];profit=gross-cost
        summary.append({
          "year":year,"family":family,"partner_order":source,
          "other_horses_k":k,"races":nr,"bet_races":c["bet_races"],
          "avg_tickets_per_race":round(c["tickets"]/nr,5),
          "tickets":c["tickets"],"stake_yen":cost,"return_yen":round(gross,2),
          "profit_yen":round(profit,2),
          "roi_pct":round(100*gross/cost,6) if cost else None,
          "hit_races":c["hit_races"],"hit_rate_pct":round(100*c["hit_races"]/nr,6),
          "miss_races":c["miss_races"],"hit_tickets":c["hit_tickets"]+c["extra_hit_tickets"],
          "budget_to_save_from_misses_for_break_even_yen":max(0,round(-profit,2)),
          **{"stake_"+a+"_yen":c["stake_"+a] for a in BUCKETS},
          **{"return_"+a+"_yen":round(c["return_"+a],2) for a in BUCKETS},
        })
    return pd.DataFrame(summary).sort_values(
        ["year","family","partner_order","other_horses_k"]).reset_index(drop=True)

def pareto(reports):
    rows=reports[reports.year.eq("2023-2025")].to_dict("records")
    fronts=[]
    for a in rows:
        a_cost=a["avg_tickets_per_race"]
        a_hit=a["hit_rate_pct"]
        a_roi=a["roi_pct"]
        dominated=False
        for b in rows:
            if a is b:continue
            if (b["avg_tickets_per_race"]<=a_cost and
                b["hit_rate_pct"]>=a_hit and b["roi_pct"]>=a_roi and
                (b["avg_tickets_per_race"]<a_cost or b["hit_rate_pct"]>a_hit or b["roi_pct"]>a_roi)):
                dominated=True;break
        if not dominated:
            fronts.append({k:a[k] for k in ("family","partner_order","other_horses_k",
                                            "avg_tickets_per_race","hit_rate_pct","roi_pct","profit_yen")})
    front=pd.DataFrame(fronts).sort_values(["avg_tickets_per_race","roi_pct"],
                                           ascending=[True,False])
    bands=[]
    for budget in (1,2,3,5,7,10,20,50,100):
        candidates=[r for r in rows if r["avg_tickets_per_race"]<=budget]
        if not candidates:continue
        for objective,key in (("ROI","roi_pct"),("HIT_CAPTURE","hit_rate_pct"),
                              ("TOTAL_PROFIT","profit_yen")):
            best=max(candidates,key=lambda r:(r[key],-r["avg_tickets_per_race"]))
            bands.append({
                "budget_avg_max_points_per_race":budget,
                "objective_hindsight_selection":objective,
                "family":best["family"],
                "partner_order":best["partner_order"],
                "other_horses_k":best["other_horses_k"],
                "avg_tickets_per_race":best["avg_tickets_per_race"],
                "hit_rate_pct":best["hit_rate_pct"],"roi_pct":best["roi_pct"],
                "profit_yen":best["profit_yen"],
            })
    return front,pd.DataFrame(bands)

def oracle_summary(ledger):
    out=[]
    for y,g in [("2023-2025",ledger)]+[(str(yr),ledger[ledger.year.eq(yr)]) for yr in YEARS]:
        winners=g[g.best_possible_paid_ticket_yen.gt(0)]
        out.append({
          "year":y,"races":len(g),
          "eligible_3rd_or_better":len(winners),
          "eligible_pct":round(100*len(winners)/len(g),5),
          "axis_4th_or_worse":int(g.selected_finish.gt(3).sum()),
          "one_ticket_per_race_stake_yen":len(g)*100,
          "post_result_max_single_ticket_return_yen":float(g.best_possible_paid_ticket_yen.sum()),
          "post_result_one_ticket_every_race_roi_pct":round(
             100*float(g.best_possible_paid_ticket_yen.sum())/(100*len(g)),3),
          "post_result_skip_all_misses_stake_yen":len(winners)*100,
          "post_result_skip_all_misses_roi_pct":round(
             100*float(g.best_possible_paid_ticket_yen.sum())/(100*len(winners)),3),
          **{"best_"+bet+"_sum_return_yen":float(g["max_paid_"+bet+"_yen"].sum())
             for bet in BET_ARITY},
        })
    return pd.DataFrame(out)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",required=True)
    p.add_argument("--picks",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    args=p.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    pred,picks=get_inputs(args.predictions,args.picks)
    dates=pd.read_csv(args.predictions,compression="infer",
        usecols=["year","_race_id","_horse_id","_race_date"],
        dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    assert_ok(not dates["_race_date"].isna().any(),"missing race dates")
    dates["_race_date"]=dates["_race_date"].astype(str).str[:10]
    assert_ok(dates["_race_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all(),"bad dates")
    pred=pred.merge(dates,on=["year","_race_id","_horse_id"],validate="one_to_one")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        chunks=list(pool.map(lambda y:one_year(y,pred,picks,Path(args.backfill_root)),YEARS))
    ledger=pd.DataFrame([r for _,ss in chunks for r in ss])
    assert_ok(len(ledger)==TOTAL_RACES and ledger.race_id.is_unique,"not all races")
    assert_ok(ledger.selected_finish.le(3).sum()==4512,"top3 baseline drift")
    assert_ok(ledger.selected_finish.eq(1).sum()==1917,"win baseline drift")
    templates=agg_years(chunks)
    allr=templates[templates.year.eq("2023-2025")].copy()
    assert_ok(len(allr)==82,f"wrong fixed template count {len(allr)}")
    front,bands=pareto(templates)
    oracle=oracle_summary(ledger)
    templates.to_csv(out/"fixed-all-race-bet-type-templates.csv",index=False)
    front.to_csv(out/"cost-hit-roi-pareto.csv",index=False)
    bands.to_csv(out/"budget-band-fixed-template-leaders.csv",index=False)
    oracle.to_csv(out/"post-result-theoretical-oracle.csv",index=False)
    ledger.to_csv(out/"per-race-max-payout-ceiling.csv.gz",index=False,compression="gzip")
    best=allr.sort_values("roi_pct",ascending=False).head(12)
    worst=allr.sort_values("roi_pct").head(5)
    (out/"summary.json").write_text(json.dumps({
       "contract":"L2_VALUE_ANCHOR_BET_FRONTIER_V1",
       "all_races":TOTAL_RACES,"train_years":[],"evaluation_years":list(YEARS),
       "same_AI_anchor":"ANCHOR_FULL max P__ANCHOR_FULL*final_win_odds, frozen OOS",
       "fixed_templates":len(allr),
       "all_templates_profit_over_zero":int(allr.profit_yen.gt(0).sum()),
       "top3_reach":4512,"top3_reach_pct":43.518519,
       "winning_bet_kinds":list(BET_ARITY),
       "winning_bet_note":"Seven physical horse-containing bet categories: WIN, PLACE, QUINELLA, WIDE, EXACTA, TRIO, TRIFECTA",
       "ignored_bracket_note":"BRACKET_QUINELLA would not necessarily contain this particular horse",
       "pareto_points":len(front),
       "top12_fixed_templates":best.to_dict("records"),
       "oracle_pool_total":oracle.iloc[0].to_dict(),
       "model_training":False,
       "explicit_post_result_oracle_warning":"HINDSIGHT CEILING ONLY: each result's maximum winning payout is chosen AFTER race, cannot be reproduced at bet time",
       "budget_bands_warning":"Best fixed-template each budget selected ex post over all years, not forward-validated",
       "final_odds_warning":"Selection involves FINAL win odds; real time execution / slippage not proven",
       "payout_source":"BACKFILL pinned to "+SHA,
       "ticket_cost_yen":100,
       "no_hindsight_within_fixed_template":True,
       "no_official_odds_price_filter":True,
       "year_2026_locked":True,
    },ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    print("SCOPE_RESULTS_START",flush=True)
    print("ORACLE_CEILING")
    print(oracle.to_csv(index=False))
    print("BEST_FIXED_TEMPLATES")
    print(best[["family","partner_order","other_horses_k","avg_tickets_per_race",
                "hit_rate_pct","roi_pct","profit_yen"]].to_csv(index=False))
    print("PARETO")
    print(front.to_csv(index=False))
    print("SCOPE_RESULTS_DONE",flush=True)

if __name__=="__main__":
    main()
