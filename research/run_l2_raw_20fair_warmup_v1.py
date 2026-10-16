#!/usr/bin/env python3
"""Fair RAW top1 20-policy comparison with 2023 prior-only January warmup.

No 2022 matching OOS horse prediction data exists in fixed project input.
2023 January is a COMMON unscored calibration period for ALL policies including
BASE; Feb-Dec 2023 uses only January training; 2024 uses full 2023 data; 2025
uses full 2023+2024 data. All 2024-25 OOS line results are reused unchanged.
No hindsight labels used in thresholds, 2026 sealed. Report all lines after
equal-window application; select 3-year pooled positive candidates per policy.
"""
from __future__ import annotations
import argparse,csv,gzip,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

import run_l2_fixed_ticket_pruning_3040_v1 as fixed
import run_l2_raw_top_odds_boundary_v1 as raw
import run_l2_raw_top_ticket_pruning_oos_v1 as prune
from build_l2_market_helpers_v1 import load_day,load_odds_day
from run_all_race_flow_ticket_audit_v1 import RACES_YEAR,assert_ok

START="2023-02-01"
POLICIES=tuple(prune.POLICIES)
TEMPLATES=tuple(raw.TEMPLATES)
BANDS=tuple(prune.BANDS)
assert len(POLICIES)==21 and len(TEMPLATES)*len(BANDS)==3040


def csvrows(p):
    with (gzip.open(p,"rt",encoding="utf-8-sig",newline="") if str(p).endswith(".gz")
          else open(p,encoding="utf-8-sig",newline="")) as f:
        yield from csv.DictReader(f)


def out_csv(path, rows):
    assert rows
    with (gzip.open(path,"wt",encoding="utf-8",newline="") if str(path).endswith(".gz")
          else open(path,"w",encoding="utf-8",newline="")) as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def count_stats(v):
    n,h,p=(int(x) for x in v)
    return dict(tickets=n,hits=h,hit_rate_pct=100*h/n if n else None,
                stake_yen=100*n,payout_yen=p,profit_yen=p-100*n,
                roi_pct=100*p/(100*n) if n else None)


def year2023(pred,picks,races,root,baseline):
    df=races[races.year.eq(2023)]
    assert_ok(len(df)==RACES_YEAR,"2023 race coverage drift")
    gr=pred[pred.year.eq(2023)].groupby("_race_id",sort=False)
    by_race=df.set_index("race_id")
    days=df.groupby("race_date")["race_id"].apply(list)

    train_stats=defaultdict(lambda:[0,0,0.])
    history=[]
    records=defaultdict(lambda:np.zeros((len(POLICIES),3),dtype=np.int64))
    portfolio=np.zeros((len(POLICIES),3),dtype=np.int64)
    months=Counter()
    jan_base=defaultdict(Counter)
    years_base=defaultdict(Counter)
    trained=False
    limits={}
    feb_races=0

    for day,ids in days.items():
        date=str(day)[:10]
        if date>=START and not trained:
            assert len(history)>1000 and months["warmup"]>0,"no prior warmup"
            limits=prune.fast_prior_thresholds(history,train_stats)
            trained=True
            print("WARMUP_READY",dict(months),"physical_history",len(history),
                  "strictly_before",START,flush=True)
        packets=load_day(root/"data"/"daily"/(date+".jsonl.gz"),set(ids))
        odds=load_odds_day(root/"data/odds/daily"/(date+".jsonl.gz"),set(ids))
        for rid in sorted(ids):
            pack=packets.get(rid);market=odds.get(rid)
            assert_ok(pack is not None and market is not None,"source missing "+rid)
            assert_ok(market.get("odds_kind")=="FINAL" and market.get("source_status")=="result",
                      "FINAL quote needed "+rid)
            g=gr.get_group(rid)
            info=by_race.loc[rid]
            axis=str(info["AI_RAW_TOP_horse"])
            assert_ok(axis==str(g.sort_values(["distill_ai_rank","market_rank","_horse_id"],
                          ascending=[True,True,True],kind="mergesort").iloc[0]["_horse_id"]),
                      "rank1 drift "+rid)
            assert_ok(str(info["VALUE_EV_horse"])==picks["ANCHOR_FULL"][rid],
                      "EV pick drift "+rid)
            shim=info.copy();shim["VALUE_EV_horse"]=axis
            physical={}
            lines=defaultdict(list)
            seen=defaultdict(set)
            for tid,ticket,card in fixed.create_cards(g,pack,market,rid,axis,shim):
                if ticket in physical: assert physical[ticket]==card, "card duplicate mismatch"
                else:physical[ticket]=card
                assert ticket not in seen[tid],"repeated ticket in line"
                seen[tid].add(ticket)
                lines[tid].append(ticket)
                p=int(card[1])
                target=jan_base if date<START else years_base
                target[tid]["n"]+=1;target[tid]["hits"]+=int(p>0);target[tid]["payout"]+=p
            if date<START:
                months["warmup"]+=1
                # historical labels used only to train NEXT evaluation period
                for card in physical.values():
                    fixed.add_learning(train_stats,card)
                    history.append(card)
                continue
            feb_races+=1
            months["evaluated"]+=1
            tickets=list(physical)
            if not tickets:continue
            cards=[physical[t] for t in tickets]
            pos={t:i for i,t in enumerate(tickets)}
            n=len(cards)
            flags=np.zeros((n,len(POLICIES)),dtype=np.int8)
            flags[:,0]=1
            for i,c in enumerate(cards):
                decisions=fixed.mask(c,train_stats,limits)
                for p,name in enumerate(POLICIES[1:],1):
                    flags[i,p]=int(decisions[name])
            receipts=np.fromiter((int(c[1]) for c in cards),dtype=np.int64,count=n)
            hits=(receipts>0).astype(np.int64)
            portfolio[:,0]+=flags.sum(axis=0,dtype=np.int64)
            portfolio[:,1]+=hits@flags
            portfolio[:,2]+=receipts@flags
            for tid,tids in lines.items():
                ix=[pos[t] for t in tids]
                mask=flags[ix]
                v=records[tid]
                v[:,0]+=mask.sum(axis=0,dtype=np.int64)
                v[:,1]+=hits[ix]@mask
                v[:,2]+=receipts[ix]@mask
        if months["evaluated"] and months["evaluated"]%1200==0:
            print("2023_FAIR_PROGRESS",months["evaluated"],flush=True)
    assert trained and months["warmup"]+months["evaluated"]==RACES_YEAR
    assert feb_races==months["evaluated"]
    # All records must match 2023 original base minus Jan historical warmup.
    for tid in ((o,k,f,b) for o,k,f in TEMPLATES for b in BANDS):
        original=baseline[("2023",*tid)]
        jan=jan_base[tid]
        expected=tuple(int(original[i])-jan[key] for i,key in enumerate(("n","hits","payout")))
        assert all(x>=0 for x in expected),("invalid Jan subtraction",tid)
        observed=tuple(int(v) for v in records[tid][0])
        jan_total=tuple(jan[k] for k in ("n","hits","payout"))
        all2023=years_base[tid]
        assert tuple(jan_total[i]+all2023[key] for i,key in enumerate(("n","hits","payout")))==original,("2023 full replay drift",tid)
        assert observed==expected,("baseline Feb-Dec drift",tid,expected,observed)
    print("FAIR_2023_BASE_EXACT_MATCH",True,"warmup_races",months["warmup"],
          "eval_races",months["evaluated"],"physical",portfolio[0,0],flush=True)
    return records,portfolio,dict(months)


def archived_years(path,baseline):
    out={}
    for r in csvrows(path):
        y=r["period"];policy=r["policy"]
        assert y in ("2024","2025") and policy in POLICIES
        tid=(r["partner_order"],int(r["other_horses_k"]),r["family"],r["odds_band"])
        key=(int(y),policy,tid)
        assert key not in out,key
        b=baseline[(y,*tid)]
        assert tuple(int(r[x]) for x in ("before_tickets","before_hits","before_payout_yen"))==b,("old saved base mismatch",key)
        v=tuple(int(r[x]) for x in ("after_tickets","after_hits","after_payout_yen"))
        if policy=="BASE":assert v==b
        out[key]=v
    assert len(out)==2*len(POLICIES)*3040
    return out


def analyze(records,old,baseline):
    rankings=[]
    detail=[]
    for policy_index,policy in enumerate(POLICIES):
        black=[]
        all_stat=[0,0,0]
        for order,k,family in TEMPLATES:
            for band in BANDS:
                tid=(order,k,family,band)
                f23=tuple(int(x) for x in records[tid][policy_index])
                f24=old[(2024,policy,tid)]
                f25=old[(2025,policy,tid)]
                full=tuple(f23[i]+f24[i]+f25[i] for i in range(3))
                for i in range(3):all_stat[i]+=full[i]
                if not full[0] or full[2]<=full[0]*100:continue
                line=count_stats(full)
                b=count_stats(tuple(int(records[tid][0,i])+old[(2024,"BASE",tid)][i]+old[(2025,"BASE",tid)][i] for i in range(3)))
                row=dict(method=policy,partner_order=order,other_horses_k=k,
                         family=family,odds_band=band,**line,
                         base_same_window_tickets=b["tickets"],
                         base_same_window_roi_pct=b["roi_pct"],
                         year2023_tickets=f23[0],year2023_hits=f23[1],
                         year2023_payout_yen=f23[2],
                         year2024_tickets=f24[0],year2024_hits=f24[1],
                         year2024_payout_yen=f24[2],
                         year2025_tickets=f25[0],year2025_hits=f25[1],
                         year2025_payout_yen=f25[2])
                black.append(row)
        black.sort(key=lambda r:(-r["roi_pct"],-r["tickets"],r["family"],
                                 r["partner_order"],r["other_horses_k"],r["odds_band"]))
        for i,r in enumerate(black,1):r["rank_within_method"]=i
        detail.extend(black)
        black_total=[sum(x[stat] for x in black) for stat in ("tickets","hits","payout_yen")]
        b=count_stats(black_total)
        full=count_stats(all_stat)
        rankings.append(dict(method=policy,three_year_black_lines=len(black),
            black_only_line_sum_tickets_OVERLAPS=b["tickets"],
            black_only_line_sum_hits_OVERLAPS=b["hits"],
            black_only_line_sum_profit_yen_OVERLAPS=b["profit_yen"],
            black_only_line_sum_roi_pct_OVERLAPS=b["roi_pct"],
            all_lines_line_sum_tickets_OVERLAPS=full["tickets"],
            all_lines_line_sum_roi_pct_OVERLAPS=full["roi_pct"],
            year2023_eval_start=START,
            note="2023 Jan warmup excluded all methods; 2023 Feb-Dec and 2024-25 evaluated; positive 3yr posthoc; physical duplicates present"))
    rankings.sort(key=lambda r:(-r["black_only_line_sum_roi_pct_OVERLAPS"] or -1,r["method"]),reverse=True)
    for ix,row in enumerate(rankings,1):row["rank"]=ix
    return rankings,detail


def main():
    ap=argparse.ArgumentParser()
    for k in ("predictions","picks","racewise","historical",
              "baseline","pruned24-25","out-dir"):
        ap.add_argument("--"+k,required=True)
    a=ap.parse_args()
    root=Path(a.out_dir);root.mkdir(parents=True,exist_ok=True)
    original=prune.read_baseline(Path(a.baseline))
    pred,picks,races=raw.yearly_data(a.predictions,a.picks,a.racewise)
    jan_fair,physical,months=year2023(pred,picks,races,Path(a.historical),original)
    old=archived_years(Path(a.__dict__["pruned24-25"]),original)
    rankings,detail=analyze(jan_fair,old,original)
    assert len(rankings)==21
    assert len(detail)==sum(r["three_year_black_lines"] for r in rankings)
    assert all(r["profit_yen"]>0 for r in detail)
    out_csv(root/"all_20_methods_ranked_fair.csv",rankings)
    out_csv(root/"all_black_lines_by_method_fair.csv.gz",detail)
    # Every policy and original-template band, including those with no bets.
    every=[]
    for order,k,family in TEMPLATES:
        for band in BANDS:
            tid=(order,k,family,band)
            for p,name in enumerate(POLICIES):
                n,h,pay=(int(x) for x in jan_fair[tid][p])
                every.append(dict(method=name,partner_order=order,other_horses_k=k,
                                  family=family,odds_band=band,year="2023_FEB_DEC",
                                  tickets=n,hits=h,payout_yen=pay))
    out_csv(root/"2023_feb_dec_all3040_by21_policies.csv.gz",every)
    result=dict(contract="RAW_TOP1_ALL20_FAIR_2023_WARMUP_V1",
        years={"2023":"Jan training only; Feb-Dec scored all 20 + BASE",
               "2024":"original full year prior-2023 frozen masks from complete earlier run",
               "2025":"original full year prior-2023+2024 frozen masks from complete earlier run"},
        January_2023_excluded_from_all_methods_including_BASE=True,
        future_outcome_training_in_each_test_window=False,
        all20_methods_same_2023_evaluation_window=True,
        warmup_count=months,policy_ranks=rankings,
        best_method=next(r["method"] for r in rankings if r["method"]!="BASE"),
        warning="Not full 2023 all-race: no matching prior-2023 model scores exist, 2023 Jan excluded.",
        only_in_sample_selected_black_roi=True,
        ticket_line_sum_overlaps=True,no_2026=True)
    (root/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FAIR20_COMPLETE",json.dumps({"best":result["best_method"],"head":rankings[:5],"warmup":months},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
