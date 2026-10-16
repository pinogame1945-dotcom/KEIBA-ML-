#!/usr/bin/env python3
"""EV-ranked anchors 2-6: all 20 frozen ticket-pruning methods, then pooled-black lines.

Per independently evaluated EV rank:
* Same 152 ticket templates and 20 final-odds intervals as prior baseline.
* BASE and exactly four historical frozen pruners x five keep thresholds.
* 2023 is original unpruned training year for ALL policies, 2024 trains on
  2023 only, 2025 trains on 2023+2024 only.
* For EACH policy identify original line/odds-band with >0 pooled 3-year
  yen profit after pruning (not annual-all-black or stability filters).
* Buy the full union of those black lines, removing identical physical
  tickets within the race. Select the policy with greatest union ROI.
* No model refits, no 2026, no modification of frozen L2.1.
* Black-line selection AND best-method choice use the same 2023-25 data:
  descriptive hindsight only, no prospective return claim.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import run_l2_fixed_ticket_pruning_3040_v1 as fixed
import run_l2_raw_top_ticket_pruning_oos_v1 as old_pruner
import run_l2_ai_market_rank_betfrontier_v1 as ranklab
import run_l2_ev_rank2to6_official_v1 as evlab
from build_l2_market_helpers_v1 import load_day,load_odds_day
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,assert_ok

RANKS=(2,3,4,5,6)
POLICIES=tuple(fixed.POLICIES)
TEMPLATES=tuple(old_pruner.TEMPLATES)
BANDS=tuple(old_pruner.BANDS)
LINE_KEYS=tuple((order,k,family,band) for order,k,family in TEMPLATES for band in BANDS)
assert len(POLICIES)==21 and len(POLICIES[1:])==20 and len(LINE_KEYS)==3040
assert fixed.MODES==("PROB","MARKET","LEARNED","HYBRID")
assert fixed.KEEP==(90,75,50,25,10)

def baseline_from_saved(path,rank):
    """Pre-prune, independently reproduced 15,200-conditions archival receipt."""
    out={}
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            if int(row["ev_rank"])!=rank:continue
            tid=(row["partner_order"],int(row["other_horses_k"]),
                 row["family"],row["odds_band"])
            assert_ok(tid in LINE_KEYS and tid not in out,("baseline key drift",tid))
            out[tid]={}
            for year in YEARS:
                out[tid][year]=(int(row[f"{year}_tickets"]),
                                 int(row[f"{year}_hits"]),
                                 int(row[f"{year}_tickets"])*100+int(row[f"{year}_profit_yen"]))
    assert_ok(len(out)==3040,("archived baseline 3040",rank,len(out)))
    return out

def choose_axis(rid,group,info,picks,rank):
    seq=evlab.order_ev(group)
    assert_ok(seq[0]==picks["ANCHOR_FULL"][rid],"original EV first-place drift")
    assert_ok(str(info["VALUE_EV_horse"])==seq[0],"racewise EV rank drift")
    if len(seq)<rank:return None
    return seq[rank-1]

def iterate_races(year,grouped,yr,root):
    """Fetch each day's race data and FINAL odds exactly once per pass."""
    index=yr.set_index("race_id")
    done=0
    for date,ids in yr.groupby("race_date")["race_id"].apply(list).items():
        day=str(date)[:10]
        packs=load_day(root/"data"/"daily"/(day+".jsonl.gz"),set(ids))
        odds=load_odds_day(root/"data"/"odds"/"daily"/(day+".jsonl.gz"),set(ids))
        for rid in sorted(ids):
            pack=packs.get(rid);price=odds.get(rid)
            assert_ok(pack is not None and price is not None,("missing historical race",rid))
            assert_ok(price.get("odds_kind")=="FINAL" and
                      price.get("source_status")=="result",("not final historical quote",rid))
            done+=1
            yield rid,pack,price,grouped.get_group(rid),index.loc[rid]
    assert_ok(done==RACES_YEAR,("yearly race total",year,done))

def ledger_pass(data,picks,races,root,rank,archived):
    """Historic masks each unique physical ticket, all 21 policies simultaneously."""
    history=[]
    training=defaultdict(lambda:[0,0,0.])
    result={}
    thresholds_by_year={}
    training_by_year={}
    distinct_by_year={}
    for year in YEARS:
        limits=old_pruner.fast_prior_thresholds(history,training) if year>2023 else {}
        thresholds_by_year[year]=limits
        training_by_year[year]={k:tuple(v) for k,v in training.items()}
        yr=races[races.year.eq(year)]
        assert_ok(len(yr)==RACES_YEAR,"missing year race records")
        groups=data[data.year.eq(year)].groupby("_race_id",sort=False)
        summary=defaultdict(lambda:np.zeros((21,3),dtype=np.int64))
        learn=defaultdict(lambda:[0,0,0.])
        physical=0;eligible=0;done=0
        for rid,pack,price,g,info in iterate_races(year,groups,yr,root):
            done+=1
            axis=choose_axis(rid,g,info,picks,rank)
            if axis is None:continue
            eligible+=1
            shim=info.copy()
            shim["VALUE_EV_horse"]=axis
            cards=[]
            positions={}
            line_positions=defaultdict(list)
            line_seen=defaultdict(set)
            for tid,ticket,card in fixed.create_cards(g,pack,price,rid,axis,shim):
                assert_ok(tid in archived,"unknown template/band")
                assert_ok(ticket not in line_seen[tid],
                          ("duplicate within one fixed line",rid,tid))
                line_seen[tid].add(ticket)
                at=positions.get(ticket)
                if at is None:
                    at=len(cards);positions[ticket]=at;cards.append(card)
                else:
                    assert_ok(cards[at]==card,("card drift",rid,ticket))
                line_positions[tid].append(at)
            del positions,line_seen
            n=len(cards)
            physical+=n
            if n:
                pays=np.fromiter((int(c[1]) for c in cards),dtype=np.int64,count=n)
                hits=(pays>0).astype(np.int64)
                votes=np.ones((n,21),dtype=np.int8) if year==2023 else np.zeros((n,21),dtype=np.int8)
                if year>2023:
                    votes[:,0]=1
                    for idx,card in enumerate(cards):
                        matched=fixed.mask(card,training,limits)
                        for p,policy in enumerate(POLICIES[1:],1):
                            votes[idx,p]=int(matched[policy])
                for tid,indices in line_positions.items():
                    m=votes[indices]
                    counts=summary[tid]
                    counts[:,0]+=m.sum(axis=0,dtype=np.int64)
                    counts[:,1]+=hits[indices]@m
                    counts[:,2]+=pays[indices]@m
                for card in cards:fixed.add_learning(learn,card)
                if year<2025:history.extend(cards)
            if done%1200==0:
                print("EV_PRUNE_LEDGER_PROGRESS",rank,year,done,
                      "physical",physical,flush=True)
        assert_ok(done==RACES_YEAR,("ledger race count",rank,year))
        for tid in LINE_KEYS:
            m=summary.get(tid)
            observed=(0,0,0) if m is None else tuple(int(x) for x in m[0])
            assert_ok(observed==archived[tid][year],
                      ("ORIGINAL_UNPRUNED_BASELINE_MISMATCH",rank,year,tid,
                       observed,archived[tid][year]))
        if year<2025:fixed.merge(training,learn)
        result[year]=summary
        distinct_by_year[str(year)]={"races":eligible,"physical_tickets":physical}
        print("EV_PRUNE_BASELINE_EXACT",rank,year,eligible,physical,
              "original_lines",len(summary),flush=True)
    return result,thresholds_by_year,training_by_year,distinct_by_year

def black_lines(all_lines):
    selected={p:set() for p in POLICIES}
    originals={}
    for tid in LINE_KEYS:
        src=[all_lines[y].get(tid) for y in YEARS]
        pooled=np.zeros((21,3),dtype=np.int64)
        for v in src:
            if v is not None:pooled+=v
        for p,policy in enumerate(POLICIES):
            n,h,pay=(int(t) for t in pooled[p])
            if n and pay>100*n:selected[policy].add(tid)
        originals[tid]=pooled
    assert_ok(selected["BASE"],"No historical baseline black line")
    return selected,originals

def physically_dedup_pass(data,picks,races,root,rank,selected,thresholds,training_by_year):
    """Only count one real ticket per race/policy across all profitable lines."""
    candidates=defaultdict(list)
    for p,policy in enumerate(POLICIES):
        for tid in selected[policy]:candidates[tid].append(p)
    records={(year,policy):dict(n=0,hits=0,payout=0,races=0,hit_races=0)
             for year in YEARS for policy in POLICIES}
    if not candidates:return records
    for year in YEARS:
        yr=races[races.year.eq(year)]
        groups=data[data.year.eq(year)].groupby("_race_id",sort=False)
        done=0
        for rid,pack,price,g,info in iterate_races(year,groups,yr,root):
            done+=1
            axis=choose_axis(rid,g,info,picks,rank)
            if axis is None:continue
            shim=info.copy();shim["VALUE_EV_horse"]=axis
            physical={}
            for tid,ticket,card in fixed.create_cards(g,pack,price,rid,axis,shim):
                req=candidates.get(tid)
                if req:
                    ret=int(card[1])
                    entry=physical.get(ticket)
                    if entry is None:
                        entry=[card,0];physical[ticket]=entry
                    else:assert_ok(entry[0]==card,("price/payout mismatch",rid,ticket))
                    entry[1]|=sum(1<<p for p in req)
            if not physical:continue
            bought=[0]*21;hit_races=[False]*21
            for ticket,(card,req) in physical.items():
                votes=(1<<21)-1 if year==2023 else None
                if votes is None:
                    flags=fixed.mask(card,training_by_year[year],thresholds[year])
                    votes=sum(1<<p for p,policy in enumerate(POLICIES)
                              if flags[policy])
                for p in range(21):
                    if req&(1<<p) and votes&(1<<p):
                        st=records[year,POLICIES[p]]
                        st["n"]+=1
                        ret=int(card[1])
                        st["hits"]+=int(ret>0);st["payout"]+=ret
                        bought[p]+=1
                        hit_races[p]|=ret>0
            for p in range(21):
                if bought[p]:
                    st=records[year,POLICIES[p]]
                    st["races"]+=1;st["hit_races"]+=int(hit_races[p])
            if done%1200==0:
                print("EV_PRUNE_UNION_PROGRESS",rank,year,done,flush=True)
        assert_ok(done==RACES_YEAR,"union race count")
    return records

def roi(entry):
    n=entry["n"]
    return 100*entry["payout"]/(100*n) if n else None

def summarize(all_lines,selected,union,rank,pre_stats,output):
    listing=[]
    winners=[]
    black_detail=[]
    with gzip.open(output/"all_20_policies_3040_lines_3years.csv.gz",
                   "wt",encoding="utf-8",newline="",compresslevel=6) as f:
        writer=csv.writer(f)
        writer.writerow(["ev_rank","policy","line","year",
                         "tickets","hits","stake_yen","payout_yen","profit_yen","roi_pct","black"])
        for idx,policy in enumerate(POLICIES):
            for tid in LINE_KEYS:
                annual=[]
                for year in YEARS:
                    mat=all_lines[year].get(tid)
                    x=(0,0,0) if mat is None else tuple(int(v) for v in mat[idx])
                    annual.append(x)
                    n,h,pay=x
                    writer.writerow([rank,policy,"|".join(map(str,tid)),year,n,h,100*n,
                                     pay,pay-100*n,100*pay/(100*n) if n else "",int(n>0 and pay>100*n)])
                n,h,pay=(sum(a[i] for a in annual) for i in range(3))
                writer.writerow([rank,policy,"|".join(map(str,tid)),"2023-2025",n,h,100*n,
                                 pay,pay-100*n,100*pay/(100*n) if n else "",int(n>0 and pay>100*n)])
                if tid in selected[policy]:
                    black_detail.append(dict(ev_rank=rank,policy=policy,
                                             line="|".join(map(str,tid)),
                                             tickets=n,hits=h,profit_yen=pay-100*n,
                                             roi_pct=100*pay/(100*n)))
    with gzip.open(output/"three_year_pooled_black_lines_by_policy.csv.gz",
                   "wt",encoding="utf-8",newline="",compresslevel=6) as f:
        w=csv.DictWriter(f,fieldnames=list(black_detail[0]) if black_detail else
                         ["ev_rank","policy","line","tickets","hits","profit_yen","roi_pct"])
        w.writeheader();w.writerows(black_detail)
    for policy in POLICIES:
        details=[]
        for year in YEARS:
            stats=union[(year,policy)]
            details.append(dict(ev_rank=rank,policy=policy,year=year,
                                selected_black_lines=len(selected[policy]),
                                **stats,roi_pct=roi(stats)))
        total={k:sum(row[k] for row in details) for k in
               ("n","hits","payout","races","hit_races")}
        summary=dict(ev_rank=rank,policy=policy,year="2023-2025",
                     selected_black_lines=len(selected[policy]),
                     **total,roi_pct=roi(total),profit_yen=total["payout"]-100*total["n"])
        for d in details:
            d["profit_yen"]=d["payout"]-100*d["n"]
        listing.extend(details+[summary])
        if policy!="BASE" and total["n"]>0 and total["payout"]>100*total["n"]:
            winners.append(summary)
    assert_ok(len(listing)==21*4,"policy-period count")
    # Max ROI (not max absolute profit); deterministic tie-breaks.
    best=sorted(winners,key=lambda d:(-d["roi_pct"],-d["profit_yen"],-d["n"],d["policy"]))
    winner=best[0] if best else None
    old_pruner.write_csv(output/"portfolio_by_policy_and_year.csv",listing)
    comparison=[d for d in listing if d["year"]=="2023-2025"]
    comparison=sorted(comparison,key=lambda r:(-(r["roi_pct"] or -1),r["policy"]))
    old_pruner.write_csv(output/"all_20_methods_roi_ranking.csv",comparison)
    if winner:
        output_lines=[d for d in black_detail if d["policy"]==winner["policy"]]
        old_pruner.write_csv(output/"adopted_method_black_odds_lines.csv",output_lines)
    result=dict(contract="L2_EV_RANK2TO6_TICKET_PRUNING20_BEST_3YR_BLACK_UNION_V1",
                ev_rank=rank,period=[2023,2024,2025],
                source_lines_per_method=3040,
                methods_evaluated=list(POLICIES[1:]),
                total_methods=20,baseline_method="BASE",
                selection="positive pooled three-year yen profit for each line, physical-dedup union, highest union ROI method independently for each EV rank",
                methodology="frozen create_cards/calc_scores/mask/thresholds; same period retrospective post-selection",
                training={"2023":"BASE_UNPRUNED_TRAINING","2024":[2023],"2025":[2023,2024]},
                original_unpruned_input_verified=True,
                observed_rank_races=pre_stats,
                all_policy_comparison=comparison,
                adopted_policy=(winner["policy"] if winner else None),
                adopted_portfolio=winner,
                stability_medium_strict_not_applied=True,
                all_2023_2025_black_same_period_hindsight=True,
                final_quotes_not_live_prebet=True,
                year2026_sealed=True,only_free_standard_cpu=True)
    (output/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("EV_RANK_PRUNING_20_METHODS_COMPLETE",
          json.dumps({"rank":rank,"best":winner,"methods":len(comparison),
                      "black":{p:len(selected[p]) for p in POLICIES}},ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser()
    for key in ("rank","predictions","picks","racewise","baseline",
                "backfill-root","out-dir"):
        ap.add_argument("--"+key,required=True)
    args=ap.parse_args()
    rank=int(args.rank)
    assert_ok(rank in RANKS,"rank outside 2-6")
    dest=Path(args.out_dir);dest.mkdir(parents=True,exist_ok=True)
    archived=baseline_from_saved(Path(args.baseline),rank)
    data,picks=ranklab.load_predictions(args.predictions,args.picks)
    races=pd.read_csv(args.racewise,compression="infer",
                      dtype={"race_id":str,"VALUE_EV_horse":str,"AI_RAW_TOP_horse":str})
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert_ok(len(races)==TOTAL_RACES and races.race_id.is_unique,"10368 archived races")
    assert_ok(set(races.year)==set(YEARS),"2026 sealed")
    root=Path(args.backfill_root)
    stats,limits,prior_training,coverage=ledger_pass(data,picks,races,root,rank,archived)
    selected,pooled=black_lines(stats)
    # Second pass recomputes the exact frozen score with prior training data;
    # do NOT infer market/LEARNED masks from pooled future-year receipts.
    union=physically_dedup_pass(data,picks,races,root,rank,selected,limits,prior_training)
    summarize(stats,selected,union,rank,coverage,dest)

if __name__=="__main__":main()
