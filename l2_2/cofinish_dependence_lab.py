#!/usr/bin/env python3
"""L2.2 co-finish dependence lab: exact horse-combination probability vs actual payout.
Years 2023-2025 only. Calibration train 2023 -> test 2024, train 2023-24 -> test 2025.
No target in input features. Keep frozen L2.1 physical tickets read-only/excluded.
"""
from __future__ import annotations
import argparse,concurrent.futures,csv,json,itertools,math,sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

from market_distortion_lab import need,manifest_paths,l21,candidates,pl,FREEZE,BACKFILL,YEARS
from build_l2_market_helpers_v1 import load_day,load_odds_day,horse_number_map
from run_l2_all_bet_types_ticket_odds_gate_v1 import ticket_price
from run_l2_value_anchor_bet_frontier_v1 import payouts

EXPECTED_QUOTED={2023:651539,2024:665028,2025:665944}
PRED_COLS=["year","_race_id","_race_date","_horse_id","market_rank","final_win_odds","P__ANCHOR_FULL"]
STUDIES={
 "BASE_PL_115_CAP8":("raw",1.15,8),
 "RELATION_115_CAP8":("relation",1.15,8),
 "RELATION_130_CAP8":("relation",1.30,8),
 "RELATION_115_CAP20":("relation",1.15,20),
}
MAX_QUOTE=2000.0
SMOOTH_CELL_EXPECTED_WINS=25.0
SMOOTH_FAMILY_EXPECTED_WINS=40.0

def joint(kind,nums,p):
    if kind=="EXACTA":return pl(p,*nums)
    if kind=="TRIFECTA":return pl(p,*nums)
    if kind in ("QUINELLA","TRIO"):
        return sum(pl(p,*v) for v in itertools.permutations(nums))
    raise ValueError(kind)

def cohort(kind,nums,rank,ai_top,p,field_size):
    market_top=sum(rank[v]<=3 for v in nums)
    group="2PLUS_TOP3" if market_top>=2 else "AT_MOST_1_TOP3"
    agreement="ALL_AI_TOP6" if all(v in ai_top for v in nums) else "SOME_AI_OUTSIDE"
    # race-level competition and joint-profile are both pre-result.
    concentration="STRONG_FAV" if max(p.values())>=0.33 else "OPEN_FIELD"
    return (kind,group,agreement,concentration)

def bucket():
    return dict(count=0,actual_hits=0,expected_hits=0.0,return_yen=0,quote_expected_return=0.0)

def bump(group,prob,payout,odds):
    group["count"]+=1
    group["actual_hits"]+=int(payout>0)
    group["expected_hits"]+=prob
    group["return_yen"]+=payout
    group["quote_expected_return"]+=prob*odds*100

def combine(records):
    out=bucket()
    for r in records:
        for k in out:out[k]+=r[k]
    return out

def fit(train_years):
    combined=defaultdict(bucket)
    family=defaultdict(bucket)
    for year_stats in train_years:
        for key,record in year_stats.items():
            for field,value in record.items():combined[key][field]+=value
            kind=key[0]
            for field,value in record.items():family[kind][field]+=value
    factors={}
    for key,c in combined.items():
        f=family[key[0]]
        family_rate=(f["actual_hits"]+SMOOTH_FAMILY_EXPECTED_WINS)/(f["expected_hits"]+SMOOTH_FAMILY_EXPECTED_WINS)
        factor=(c["actual_hits"]+SMOOTH_CELL_EXPECTED_WINS*family_rate)/(c["expected_hits"]+SMOOTH_CELL_EXPECTED_WINS)
        factors[key]=max(.60,min(1.40,factor))
    return factors,combined,family

def stats(payouts):
    winners=sorted((int(x) for x in payouts if x>0),reverse=True)
    n=len(payouts);ret=sum(winners);largest=sum(winners[:3])
    return dict(tickets=n,hits=len(winners),stake_yen=100*n,return_yen=ret,
        profit_yen=ret-100*n,roi_pct=100*ret/(100*n) if n else 0.,
        biggest3_share_pct=100*largest/ret if ret else 0.,
        roi_without_biggest3_pct=100*(ret-largest)/(100*n) if n else 0.)

def worker(year,frame,root_path,mode,parameters,baseline):
    root=Path(root_path);byrace=frame.groupby("_race_id",sort=False)
    meta=frame[["_race_id","_race_date"]].drop_duplicates()
    need(len(meta)==3456,"expected all 3456 OOS races for "+str(year))
    day_races=(meta.assign(day=meta._race_date.astype(str).str[:10])
                  .groupby("day")["_race_id"].apply(list))
    manifest=manifest_paths(root,list(day_races.index))
    grouped=defaultdict(bucket)
    portfolio=defaultdict(list)
    trend=defaultdict(bucket)
    aud=defaultdict(int)
    for day_i,(day,ids) in enumerate(day_races.items(),1):
        rpath,opath=manifest[day]
        races=load_day(root/rpath,set(ids));odds=load_odds_day(root/opath,set(ids))
        for rid in sorted(ids):
            pack=races.get(rid);book=odds.get(rid)
            need(pack is not None and book is not None,"race or odds missing "+rid)
            need(book.get("odds_kind")=="FINAL" and book.get("source_status")=="result",
                 "non-FINAL quote: "+rid)
            group=byrace.get_group(rid)
            numbers=horse_number_map(pack)
            need(set(group._horse_id.astype(str)).issubset(numbers),"horse mapping "+rid)
            market=group.sort_values(["market_rank","final_win_odds","_horse_id"],
                                    ascending=[True,True,True],kind="mergesort")
            airoster=group.sort_values(["P__ANCHOR_FULL","market_rank","_horse_id"],
                                    ascending=[False,True,True],kind="mergesort")
            ranked_market=list(market._horse_id.astype(str))
            ranked_ai=list(airoster._horse_id.astype(str))
            candidate_ids=list(dict.fromkeys(ranked_market[:6]+ranked_ai[:6]))[:9]
            selected=tuple(numbers[h] for h in candidate_ids)
            need(len(selected)>=3 and len(set(selected))==len(selected),"candidate drift "+rid)
            ai_top={numbers[h] for h in ranked_ai[:6]}
            ranks={numbers[h]:i for i,h in enumerate(ranked_market,1)}
            pool={numbers[h]:1.0/float(o) for h,o in
                  group[["_horse_id","final_win_odds"]].itertuples(index=False,name=None)}
            norm=sum(pool.values())
            need(norm>0 and all(math.isfinite(x) and x>0 for x in pool.values()),
                 "win quotes incorrect "+rid)
            prob={n:x/norm for n,x in pool.items()}
            returns=payouts(pack,rid)
            options={key:[] for key in STUDIES}
            quote_count=0
            for ticket in candidates(selected):
                price=ticket_price(ticket,book)
                if price is None:
                    aud["quote_missing"]+=1
                    aud["winner_quote_missing"]+=int(bool(returns.get(ticket,0)))
                    continue
                aud["priced"]+=1;quote_count+=1
                actual=int(returns.get(ticket,0))
                p=joint(ticket[0],ticket[1],prob)
                need(0<p<=1 and math.isfinite(p),"joint outcome probability invalid "+rid)
                key=cohort(ticket[0],ticket[1],ranks,ai_top,prob,len(numbers))
                group_feature=(year,ticket[0],key[1],key[2],key[3])
                bump(trend[group_feature],p,actual,price)
                if mode=="train":
                    bump(grouped[key],p,actual,price)
                    continue
                require_key=(year,rid,ticket[0],ticket[1])
                if require_key in baseline:
                    aud["frozen_l21_overlap"]+=1
                    continue
                if price>MAX_QUOTE:
                    aud["quote_too_high"]+=1
                    continue
                calibrated=p*parameters.get(key,1.0)
                raw=p*price
                corrected=calibrated*price
                for name,(model,threshold,cap) in STUDIES.items():
                    ev=raw if model=="raw" else corrected
                    if ev>=threshold:
                        options[name].append((ev,ticket,actual))
            for name,rows in options.items():
                if mode!="test":continue
                cap=STUDIES[name][2]
                chosen=sorted(rows,key=lambda x:(-x[0],x[1]))[:cap]
                portfolio[name].extend(z[2] for z in chosen)
                aud["races_bought_"+name]+=int(bool(chosen))
            aud["races"]+=1
        if day_i%65==0:
            print("COFINISH",year,mode,"days",day_i,"priced",aud["priced"],flush=True)
    need(aud["races"]==3456,"race count missing")
    need(aud["priced"]==EXPECTED_QUOTED[year],
         "price universe regression "+str(year)+": "+str(aud["priced"]))
    if mode=="train":return dict(grouped),dict(trend),dict(aud)
    return {name:stats(returns) for name,returns in portfolio.items()},dict(trend),dict(aud)

def main(a):
    df=pd.read_csv(a.predictions,usecols=PRED_COLS,compression="infer",
                   dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    df.year=pd.to_numeric(df.year,errors="raise").astype(int)
    need(set(df.year.unique())==set(YEARS) and df._race_id.nunique()==10368,
         "year 2026 sealed and original OOS coverage required")
    need(not df.duplicated(["_race_id","_horse_id"]).any(),"duplicate OOS predictions")
    frozen=l21((a.joined,a.missing))
    years={y:df[df.year==y].copy() for y in YEARS}
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        tasks={y:pool.submit(worker,y,years[y],a.backfill_root,"train",{},set())
               for y in YEARS}
        train={y:tasks[y].result() for y in YEARS}
    model={2024:fit([train[2023][0]]),
           2025:fit([train[2023][0],train[2024][0]])}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        tasks={y:pool.submit(worker,y,years[y],a.backfill_root,"test",
                             model[y][0],set(frozen)) for y in (2024,2025)}
        tests={y:tasks[y].result() for y in (2024,2025)}
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/"joint_profile_by_year.csv").open("w",newline="") as f:
        fields=["year","bet_type","top3_mix","ai_consensus","race_concentration",
                "count","actual_hits","expected_hits","observed_to_expected",
                "return_yen","actual_roi_pct","harville_expected_roi_pct"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for year in YEARS:
            for (kind,mix,agree,hot),v in sorted(train[year][0].items()):
                n=v["count"]
                writer.writerow(dict(year=year,bet_type=kind,top3_mix=mix,
                  ai_consensus=agree,race_concentration=hot,count=n,
                  actual_hits=v["actual_hits"],expected_hits=round(v["expected_hits"],8),
                  observed_to_expected=round(v["actual_hits"]/v["expected_hits"],6),
                  return_yen=v["return_yen"],actual_roi_pct=100*v["return_yen"]/(100*n),
                  harville_expected_roi_pct=v["quote_expected_return"]/n))
    with (out/"forward_portfolios.csv").open("w",newline="") as f:
        fields=["year","policy","tickets","hits","stake_yen","return_yen",
                "profit_yen","roi_pct","biggest3_share_pct","roi_without_biggest3_pct",
                "l21_stake_yen","l21_profit_yen","combined_profit_yen","combined_roi_pct"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for y in (2024,2025):
            base=[v for key,v in frozen.items() if key[0]==y]
            bstake=100*len(base);breturn=sum(base)
            for name,z in sorted(tests[y][0].items()):
                writer.writerow(dict(year=y,policy=name,**z,
                    l21_stake_yen=bstake,l21_profit_yen=breturn-bstake,
                    combined_profit_yen=(breturn-bstake)+z["profit_yen"],
                    combined_roi_pct=100*(breturn+z["return_yen"])/(bstake+z["stake_yen"])))
    report=dict(contract="KEIBA_L2_2_PAIR_DEPENDENCE_V1",
      frozen_l21=FREEZE,pinned_backfill=BACKFILL,year_2026_sealed=True,
      retrospective_final_odds_not_live_executable=True,
      explicit_year_training={"2024":[2023],"2025":[2023,2024]},
      quoted_candidates=dict(EXPECTED_QUOTED),
      cohort_definition=["bet_type","market_top3_count_ge2","all_members_AI_top6","favorite_win_prob_ge0.33"],
      baseline_model="normalized market win odds => Plackett-Luce joint finish order",
      conditional_model="prior year observed / Harville expected, shrink 25 expected wins to family ratio; family shrink 40; clip .6..1.4",
      primary_policy="RELATION_115_CAP8",quote_cap=MAX_QUOTE,
      all_strategies=STUDIES,all_l21_original_tickets=len(frozen),
      selected_cell_counts={str(y):len(model[y][0]) for y in model},
      train_audit={str(y):train[y][2] for y in YEARS},
      test_audit={str(y):tests[y][2] for y in tests},
      forward={str(y):tests[y][0] for y in tests},
      warning="Exploratory multiple comparison; closing odds retrospective; 2026 held out. Cannot infer profitable live strategy.")
    (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("COFINISH_FORWARD",json.dumps(report["forward"],ensure_ascii=False),flush=True)

def selftest():
    probabilities={1:.50,2:.30,3:.20}
    need(abs(sum(pl(probabilities,*x) for x in itertools.permutations((1,2,3)))-1.)<1e-10,
         "full joint finish distribution must sum to 1")
    need(abs(sum(joint("QUINELLA",pair,probabilities)
                 for pair in itertools.combinations((1,2,3),2))-1.)<1e-10,
         "unordered winning two all outcomes")
    need(abs(joint("TRIO",(1,2,3),probabilities)-1.)<1e-10,"three-runner trio prob")
    need(len(set(candidates((1,2,3))))==16,"ticket shape")
    model,_,fam=fit([{("QUINELLA","2PLUS_TOP3","ALL_AI_TOP6","OPEN_FIELD"):
                      dict(count=100,actual_hits=13,expected_hits=10,
                           return_yen=12000,quote_expected_return=10000)}])
    need(len(model)==1 and .6<=next(iter(model.values()))<=1.4,"shrinkage")
    need(stats([0,250,0,0])["profit_yen"]==-150,"payout bookkeeping")
    print("COFINISH_SELFTEST_OK")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--selftest",action="store_true")
    for name in ("predictions","joined","missing","backfill-root","out-dir"):
        parser.add_argument("--"+name)
    args=parser.parse_args()
    if args.selftest:selftest()
    else:main(args)
