#!/usr/bin/env python3
"""Historical 2023-2025 original-black plus red-to-black strategies, stabilize then merge.

The user-requested result: choose original 152x20 odds lines, filter bets by
existing pre-race horse-lens gates (no finish/outcome enters a ticket gate),
require each line-gate positive in EACH observed year 2023/24/25, screen hit
counts, remove jackpot-dependent or fragile options, and buy every surviving
physical ticket ONCE. This is deliberately an IN-SAMPLE DESCRIPTIVE STUDY:
selection on all 3 years, no claim of future profitability. No 2026.
"""

from __future__ import annotations
import argparse,csv,gzip,heapq,json,math
from collections import Counter,defaultdict
from pathlib import Path
import pandas as pd
import run_l2_all_fixed_odds_band_lenses_v1 as lens
import run_l2_fixed_ticket_pruning_3040_v1 as fixed
from build_l2_market_helpers_v1 import load_day,load_odds_day,horse_number_map
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,assert_ok

YRS=(2023,2024,2025)
GATES=tuple(lens.GATES)
# Different robust-ness requirements are pre-registered (not auto-picked from
# any withheld-year ROI); never choose the "best" based on test outcome.
POLICIES={
  "BASE_ORIGINAL_11":dict(n=0,hits=0,per_year_hits=0,max1=1.0,max3=1.0,minus3_roi=0),
  "ALL_YEAR_BLACK_CANDIDATES":dict(n=0,hits=0,per_year_hits=0,max1=1.0,max3=1.0,minus3_roi=0),
  "HITS30_NO_JACKPOT_GATE":dict(n=500,hits=30,per_year_hits=8,max1=1.0,max3=1.0,minus3_roi=0),
  "STABLE_MEDIUM":dict(n=500,hits=30,per_year_hits=8,max1=.40,max3=.70,minus3_roi=90),
  "STABLE_STRICT":dict(n=800,hits=50,per_year_hits=12,max1=.30,max3=.55,minus3_roi=100),
}

BASIC={"min_n":250,"min_hits":15,"min_roi_pct":100.0}
DIAG=("ALL_FIXED_3040","HINDSIGHT_BLACK_245")
YEAR_LOSSES=("2024","2025")

def read(path):
    with (gzip.open(path,"rt",encoding="utf-8",newline="") if str(path).endswith(".gz")
          else open(path,encoding="utf-8",newline="")) as f:
        return list(csv.DictReader(f))

def write(path,rows):
    assert rows
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def key_from_row(r):
    return (tuple(r["original_line"].split("|")),r["gate"])

def key_from_ticket(tid,gate):
    return (tuple(map(str,tid)),gate)

def historical_summary(path):
    years=defaultdict(dict)
    for row in read(path):
        y=row["year"]
        if y not in ("2023","2024","2025"):continue
        key=key_from_row(row)
        assert key not in years[y],(y,key)
        stats=dict(n=int(row["filtered_tickets"]),hits=int(row["filtered_hit_tickets"]),
                   return_yen=int(row["filtered_return_yen"]))
        assert stats["return_yen"]>=0 and 0<=stats["hits"]<=stats["n"]
        years[y][key]=stats
    assert all(len(years[str(y)])==3040*len(GATES) for y in (2023,2024,2025))
    return years

def prior_stats(by_year,years,key):
    c=Counter()
    for y in years:
        a=by_year[str(y)][key]
        for k,v in a.items():c[k]+=v
    c["profit_yen"]=c["return_yen"]-100*c["n"]
    c["roi_pct"]=100*c["return_yen"]/(100*c["n"]) if c["n"] else 0
    c["positive_years"]=sum(by_year[str(y)][key]["return_yen"]>
                   100*by_year[str(y)][key]["n"] for y in years)
    return c

def preliminary(by_year,years):
    first=by_year[str(years[0])]
    matching=set()
    for k in first:
        s=prior_stats(by_year,years,k)
        if (s["n"]>=BASIC["min_n"] and s["hits"]>=BASIC["min_hits"]
             and s["roi_pct"]>BASIC["min_roi_pct"]):
            matching.add(k)
    return matching

def top3_sum(values):
    return sum(values)

def decide(prelim,by_year,years,actual,policy):
    cfg=POLICIES[policy]; selected={};audits=[]
    for key in sorted(prelim):
        st=prior_stats(by_year,years,key)
        hist=actual.get(key)
        assert hist is not None,("no training hit concentration",key)
        # Historical line aggregates and exact settled training tickets MUST
        # reconcile; an out-of-contract price/quote or duplicate is fatal.
        assert hist["n"]==st["n"],("TRAIN_N_DRIFT",key,hist["n"],st["n"])
        assert hist["hits"]==st["hits"],("TRAIN_HIT_DRIFT",key,hist["hits"],st["hits"])
        assert hist["return_yen"]==st["return_yen"],("TRAIN_RET_DRIFT",key,hist["return_yen"],st["return_yen"])
        p3=hist["biggest3"]
        top1=(max(p3) if p3 else 0)
        top3=top3_sum(p3)
        share1=top1/st["return_yen"] if st["return_yen"] else 1.0
        share3=top3/st["return_yen"] if st["return_yen"] else 1.0
        minus3=(100*(st["return_yen"]-top3)/(100*st["n"])) if st["n"] else 0
        checks=dict(hits=st["hits"]>=cfg["hits"],n=st["n"]>=cfg["n"],
                    roi=st["roi_pct"]>cfg["min_roi"],
                    biggest1=share1<=cfg["max1"],
                    biggest3=share3<=cfg["max3"],
                    minus3=minus3>=cfg["minus3_roi"],
                    year_balance=not cfg["require_all_years_positive"]
                          or st["positive_years"]==len(years))
        accepted=all(checks.values())
        if accepted:selected[key]=dict(st)
        audits.append(dict(test_year=max(years)+1,policy=policy,
          train_years="-".join(map(str,years)),
          line="|".join(key[0]),gate=key[1],decision="ACCEPT" if accepted else "REJECT",
          n=st["n"],hits=st["hits"],profit_yen=st["profit_yen"],roi_pct=st["roi_pct"],
          positive_training_years=st["positive_years"],
          biggest_single_winning_payout_yen=top1,
          top3_winning_payout_total_yen=top3,
          biggest1_return_share=share1,top3_return_share=share3,
          exclude_top3_roi_pct=minus3,
          failures=",".join(k for k,v in checks.items() if not v)))
    return selected,audits

def harvest(history,key,returned):
    c=history[key]
    c["n"]+=1
    if returned>0:
        c["hits"]+=1;c["return_yen"]+=returned
        top=c["biggest3"]
        if len(top)<3:heapq.heappush(top,returned)
        elif returned>top[0]:heapq.heapreplace(top,returned)

def record_purchase(bucket,ticket,returned):
    # For selection portfolios, duplicates are suppressed at race level before
    # settling: exact 100-yen unit / physical ticket.
    bucket["tickets"]+=1
    bucket["hits"]+=int(returned>0)
    bucket["return_yen"]+=returned
    if returned>0:
        top=bucket["top3"]
        if len(top)<3:heapq.heappush(top,returned)
        elif returned>top[0]:heapq.heapreplace(top,returned)

def make_horse(race_pred,pack):
    no=horse_number_map(pack)
    pr=race_pred.sort_values(["P__ANCHOR_FULL","market_rank","_horse_id"],
                   ascending=[False,True,True],kind="mergesort")
    pranks=dict(zip(pr["_horse_id"].astype(str),range(1,len(pr)+1)))
    horse={}
    for _,row in race_pred.iterrows():
        hid=str(row["_horse_id"])
        horse[no[hid]]=dict(P__ANCHOR_FULL=float(row["P__ANCHOR_FULL"]),
           market_prob_norm=float(row["market_prob_norm"]),risk=float(row["risk"]),
           distill_score_z=float(row["distill_score_z"]),
           distill_ai_rank=int(row["distill_ai_rank"]),market_rank=int(row["market_rank"]),
           P_rank=pranks[hid])
    assert len(horse)==len(race_pred), "horse feature coverage drift"
    return horse,no

def add_race(result,allcards):
    if not allcards:return
    result["races_bought"]+=1
    had_hit=0
    for ticket,returned in allcards.items():
        record_purchase(result,ticket,returned)
        had_hit+=int(returned>0)
    result["hit_races"]+=int(had_hit>0)

def one_year(year,pred,picks,races,root,
             history_needed,accepted,plus_from_hindsight):
    # history_needed only used for years BEFORE each target evaluation fold
    g=pred[pred.year.eq(year)].groupby("_race_id",sort=False)
    rs=races[races.year.eq(year)]
    assert len(rs)==RACES_YEAR
    index=rs.set_index("race_id")
    dates=rs.groupby("race_date")["race_id"].apply(list)
    hist=defaultdict(lambda:dict(n=0,hits=0,return_yen=0,biggest3=[]))
    outputs=defaultdict(lambda:dict(races_bought=0,hit_races=0,tickets=0,
                                    hits=0,return_yen=0,top3=[]))
    eval_year=year in (2023,2024,2025)
    gate_by_line=defaultdict(list)
    for key in history_needed:
        gate_by_line[key[0]].append(key[1])
    buy_by_line=defaultdict(list)
    for pol,accepted_lines in accepted.items():
        for key in accepted_lines:
            buy_by_line[key[0]].append((pol,key[1]))
    recent_count=0;seen=0
    for day,ids in dates.items():
        packs=load_day(root/"data"/"daily"/(day+".jsonl.gz"),set(ids))
        prices=load_odds_day(root/"data"/"odds"/"daily"/(day+".jsonl.gz"),set(ids))
        for rid in sorted(ids):
            seen+=1
            pack=packs.get(rid);market=prices.get(rid)
            assert pack is not None and market is not None,("missing day/race",day,rid)
            assert market.get("odds_kind")=="FINAL" and market.get("source_status")=="result",rid
            gf=g.get_group(rid)
            info=dict(index.loc[rid]);info["race_id"]=rid
            axis=picks["ANCHOR_FULL"][rid]
            horse,no=make_horse(gf,pack)
            anchor_no=no[axis]
            settled={}
            votes={}
            qualifying=defaultdict(dict)
            baseline_black={}
            baseline_all={}
            for tid,ticket,card in fixed.create_cards(gf,pack,market,rid,axis,info):
                returned=int(card[1])
                if ticket in settled:assert settled[ticket]==returned,("conflicting payout",rid,ticket)
                settled[ticket]=returned
                if eval_year:
                    baseline_all[ticket]=returned
                    if tuple(map(str,tid)) in plus_from_hindsight:baseline_black[ticket]=returned
                tidkey=tuple(map(str,tid))
                req_hist=gate_by_line.get(tidkey,())
                req_buy=buy_by_line.get(tidkey,())
                if not req_hist and not req_buy:continue
                if ticket not in votes:votes[ticket]=lens.gate_flags(ticket,horse,anchor_no)
                check=votes[ticket]
                for gate in req_hist:
                    if check[gate]:harvest(hist,(tidkey,gate),returned)
                for pol,gate in req_buy:
                    if check[gate]:qualifying[pol][ticket]=returned
            if eval_year:
                add_race(outputs["HINDSIGHT_BLACK_245"],baseline_black)
                add_race(outputs["ALL_FIXED_3040"],baseline_all)
                for policy in POLICIES:add_race(outputs[policy],qualifying[policy])
            if seen%1400==0:
                print("STABLE_YEAR_PROGRESS",year,seen,
                  "train_subset",len(history_needed),
                  "qualifying_lines",sum(len(q) for q in accepted.values()),flush=True)
    assert seen==RACES_YEAR
    for key in history_needed:
        # A line can have zero candidate bets in 2023 but positive bets
        # in 2024. Preserve its genuine zero-count prior-year contribution.
        if key not in hist:
            hist[key]=dict(n=0,hits=0,return_yen=0,biggest3=[])
    print("STABLE_YEAR_DONE",year,"training_ticket_contribs",len(hist),
          "portfolio_policies",len(outputs),flush=True)
    return hist,outputs

def merge_hist(history_a,history_b):
    # Same gate×original line, different earlier years. 3 max payouts can
    # be merged exactly by keeping only largest 3 of each year's 3 winners.
    dst={}
    for key in (set(history_a)|set(history_b)):
        aa=history_a.get(key,{})
        bb=history_b.get(key,{})
        winners=heapq.nlargest(3,list(aa.get("biggest3",()))+list(bb.get("biggest3",())))
        dst[key]=dict(n=aa.get("n",0)+bb.get("n",0),
          hits=aa.get("hits",0)+bb.get("hits",0),
          return_yen=aa.get("return_yen",0)+bb.get("return_yen",0),
          biggest3=winners)
    return dst

def result_row(year,name,c,total_races):
    n=c["tickets"];hit=c["hits"];ret=c["return_yen"]
    return dict(year=year,portfolio=name,available_races=total_races,
      races_bought=c["races_bought"],race_coverage_pct=
        100*c["races_bought"]/total_races,
      tickets=n,winning_tickets=hit,losing_tickets=n-hit,
      winning_races=c["hit_races"],stake_yen=100*n,return_yen=ret,
      profit_yen=ret-100*n,roi_pct=100*ret/(100*n) if n else None,
      top1_winner_payout_yen=max(c["top3"]) if c["top3"] else 0,
      top3_winner_payouts_yen=sum(c["top3"]),
      payout_concentration_top3_pct=
        100*sum(c["top3"])/ret if ret else None,
      roi_minus_biggest3_pct=
        100*(ret-sum(c["top3"]))/(100*n) if n else None)

def pool(results):
    # per-year ROI/earnings/payouts are additive. Race counts represent
    # non-overlapping years, and top payouts combine the 3 largest in each.
    c=dict(races_bought=sum(x["races_bought"] for x in results),
        hit_races=sum(x["hit_races"] for x in results),
        tickets=sum(x["tickets"] for x in results),
        hits=sum(x["hits"] for x in results),
        return_yen=sum(x["return_yen"] for x in results),
        top3=heapq.nlargest(3,sum([x["top3"] for x in results],[])))
    return c

def main():
    ap=argparse.ArgumentParser()
    for name in ("predictions","picks","racewise","backfill-root","bands",
                 "original-lens-lines","old-union","out-dir"):
        ap.add_argument("--"+name,required=True)
    a=ap.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    original=lens.load_original(a.bands)
    originally_black={tuple(map(str,k)) for k,v in original.items()
                      if int(v["2023-2025_損益円"])>0}
    assert len(originally_black)==245
    by_year=historical_summary(a.original_lens_lines)
    first=by_year["2023"]
    positive_all_years=set()
    gold_old_11=set()
    for key in first:
        s=prior_stats(by_year,(2023,2024,2025),key)
        if all(by_year[str(y)][key]["return_yen"]>100*by_year[str(y)][key]["n"]
               and by_year[str(y)][key]["n"]>0 for y in (2023,2024,2025)):
            assert s["profit_yen"]>0
            positive_all_years.add(key)
            if key[1]=="ALL" and key[0] in originally_black:
                gold_old_11.add(key)
    assert len(gold_old_11)==11,("original 11 all year profitable audit drift",len(gold_old_11))
    print("HINDSIGHT_PRIOR_BLACK_FIXED_LINES",len(originally_black),
          "ALL_THREE_YEARS_BLACK_LINE_GATE_PAIRS",len(positive_all_years),
          "EXACT_OLD_GOLD_11",len(gold_old_11),flush=True)
    pred,picks=lens.frame(a.predictions,a.picks)
    races=pd.read_csv(a.racewise,compression="infer",dtype={
        "race_id":str,"VALUE_EV_horse":str,"AI_RAW_TOP_horse":str})
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert len(races)==TOTAL_RACES and races.race_id.is_unique
    assert set(races.year)==set(YRS)
    root=Path(a.backfill_root)
    # EXACT 2023/24/25 winner receipts; no ticket gate uses final position.
    # These settled amounts are used ONLY to assess which historical
    # consistently profitable lines are overdependent on 1-3 jackpots.
    historical={}
    for year in YRS:
        got,_=one_year(year,pred,picks,races,root,positive_all_years,{},originally_black)
        historical[str(year)]=got
        for key in positive_all_years:
            obs=by_year[str(year)][key]
            gotstats=got[key]
            assert gotstats["n"]==obs["n"],("line N drift",year,key,gotstats["n"],obs["n"])
            assert gotstats["hits"]==obs["hits"],("line HIT drift",year,key)
            assert gotstats["return_yen"]==obs["return_yen"],("line RETURN drift",year,key)
    pooled={}
    for key in positive_all_years:
        one=historical["2023"][key];two=historical["2024"][key]
        three=historical["2025"][key]
        winners=heapq.nlargest(3,one["biggest3"]+two["biggest3"]+three["biggest3"])
        pooled[key]=dict(n=sum(t["n"] for t in (one,two,three)),
                         hits=sum(t["hits"] for t in (one,two,three)),
                         return_yen=sum(t["return_yen"] for t in (one,two,three)),
                         biggest3=winners)
    selected={k:{} for k in POLICIES}
    audits=[]
    for key in sorted(positive_all_years):
        agg=pooled[key]
        total=prior_stats(by_year,(2023,2024,2025),key)
        assert (agg["n"],agg["hits"],agg["return_yen"])==(
                total["n"],total["hits"],total["return_yen"])
        largest=max(agg["biggest3"]) if agg["biggest3"] else 0
        sum3=sum(agg["biggest3"])
        share1=largest/agg["return_yen"] if agg["return_yen"] else 1.0
        share3=sum3/agg["return_yen"] if agg["return_yen"] else 1.0
        roi_minus3=100*(agg["return_yen"]-sum3)/(100*agg["n"]) if agg["n"] else 0
        per_year_hits=min(historical[str(y)][key]["hits"] for y in YRS)
        tid,gate=key
        orig=original[tuple([tid[0],int(tid[1]),tid[2],tid[3]])]
        origprofit=int(orig["2023-2025_損益円"])
        category="ORIGINAL_BLACK" if origprofit>0 else (
          "RED_TO_BLACK" if origprofit<0 else "ZERO_TO_BLACK")
        for name,conf in POLICIES.items():
            checks=dict(
                enough_tickets=agg["n"]>=conf["n"],
                enough_hits=agg["hits"]>=conf["hits"],
                each_year_hits=per_year_hits>=conf["per_year_hits"],
                one_jackpot_concentration=share1<=conf["max1"],
                three_jackpot_concentration=share3<=conf["max3"],
                remove_top3_roi=roi_minus3>=conf["minus3_roi"],
                old_original_11=(name!="BASE_ORIGINAL_11" or key in gold_old_11))
            keep=all(checks.values())
            if name=="ALL_YEAR_BLACK_CANDIDATES":keep=True
            if keep:selected[name][key]=dict(total)
            audits.append(dict(policy=name,line="|".join(tid),gate=gate,
               original_category=category,decision="ACCEPT" if keep else "REJECT",
               original_3year_profit_yen=origprofit,
               historical_profit_yen=total["profit_yen"],historical_roi_pct=total["roi_pct"],
               historical_tickets=agg["n"],historical_wins=agg["hits"],
               minimum_annual_winning_tickets=per_year_hits,
               top1_payout_yen=largest,top3_payout_yen=sum3,
               top1_return_share_pct=100*share1,top3_return_share_pct=100*share3,
               roi_excluding_top3_payouts=roi_minus3,
               rejected_because=",".join(k for k,v in checks.items() if not v)))
    print("STABILITY_SELECTION_COUNTS",json.dumps({name:{
       "line_gate_pairs":len(rows),
       "unique_original_lines":len(set(k[0] for k in rows)),
       "old_black_line_gates":sum(k[0] in originally_black for k in rows),
       "red_to_black_line_gates":sum(int(original[(k[0][0],int(k[0][1]),k[0][2],k[0][3])]["2023-2025_損益円"])<0 for k in rows)}
       for name,rows in selected.items()},ensure_ascii=False),flush=True)
    # Once robust rules are determined, BUY ALL eligible physical tickets.
    # Each source line is the original 152 method × original 20 odds band
    # and the horse gate is fixed prior to inspecting that race's result.
    # The entire physical ticket is purchased at most once per race.
    annual={}
    for y in YRS:
        _,outcomes=one_year(y,pred,picks,races,root,set(),selected,originally_black)
        annual[str(y)]=outcomes
    # Verify physical ticket baseline against independent historical runs.
    expected={r["year"]:r for r in read(a.old_union)
              if r["group"]=="THREE_YEAR_TOTAL_PLUS"}
    complete={r["year"]:r for r in read(
         "research-results/l2-frozen-odds-l1-lenses-v1/run-38694334351/union_black245_other2795_x_existing_axes_yearly.csv")
         if r["scope"]=="ALL_3040" and r["gate"]=="ALL"}
    for y in YRS:
        value=annual[str(y)]
        b=expected[str(y)]
        for field,metric in (("races_bought","races_bought"),("unique_tickets","tickets"),
                 ("hit_tickets","hits"),("return_yen","return_yen")):
            assert int(b[field])==value["HINDSIGHT_BLACK_245"][metric],(
                "245 GOLDEN RECONSTRUCTION FAIL",y,field)
        u=complete[str(y)]
        for field,metric in (("baseline_races_bought","races_bought"),
                ("baseline_tickets","tickets"),("baseline_hit_tickets","hits"),
                ("baseline_return_yen","return_yen")):
            assert int(u[field])==value["ALL_FIXED_3040"][metric],(
                "3040 FULL BASELINE RECONSTRUCTION FAIL",y,field)
    legacy={r["year"]:r for r in read(a.old_union) if r["group"]=="ALL_THREE_YEARS_PLUS"}
    for y in YRS:
        b=legacy[str(y)];g=annual[str(y)]["BASE_ORIGINAL_11"]
        for source,field in (("races_bought","races_bought"),
                            ("unique_tickets","tickets"),
                            ("hit_tickets","hits"),("return_yen","return_yen")):
            assert int(b[source])==g[field],("OLD GOLD 11 REPRO FAIL",y,source,
                                              b[source],g[field])
    print("ALL_3_YEARS_ORIGINAL_245_FULL_3040_AND_STABLE_11_GOLDEN_VERIFIED",flush=True)
    metrics=[]
    for name in (*POLICIES,*DIAG):
        for y in YRS:metrics.append(result_row(str(y),name,annual[str(y)][name],3456))
        metrics.append(result_row("2023-2025",name,
           pool([annual[str(y)][name] for y in YRS]),10368))
    write(out/"three_year_stable_black_only_dedup_portfolio.csv",metrics)
    write(out/"every_candidate_line_jackpot_concentration_and_decision.csv",audits)
    keep_rows=[]
    for name,selection in selected.items():
        for (tid,g),stats in sorted(selection.items()):
            orig=int(original[(tid[0],int(tid[1]),tid[2],tid[3])]["2023-2025_損益円"])
            keep_rows.append(dict(policy=name,line="|".join(tid),gate=g,
                     category="ORIGINAL_BLACK" if orig>0 else ("RED_TO_BLACK" if orig<0 else "ZERO_TO_BLACK"),
                     historical_stake_yen=stats["n"]*100,
                     historical_profit_yen=stats["profit_yen"],
                     historical_hit_tickets=stats["hits"],
                     year2023_profit_yen=by_year["2023"][tid,g]["return_yen"]-100*by_year["2023"][tid,g]["n"],
                     year2024_profit_yen=by_year["2024"][tid,g]["return_yen"]-100*by_year["2024"][tid,g]["n"],
                     year2025_profit_yen=by_year["2025"][tid,g]["return_yen"]-100*by_year["2025"][tid,g]["n"]))
    if not keep_rows:
        keep_rows=[dict(policy="",line="",gate="",category="",
                        historical_stake_yen=0,historical_profit_yen=0,historical_hit_tickets=0,
                        year2023_profit_yen=0,year2024_profit_yen=0,year2025_profit_yen=0)]
    write(out/"retained_original_black_and_red_to_black_candidate_lines.csv",keep_rows)
    result=dict(contract="L2_HINDSIGHT_ALL_3_YEARS_BLACK_PLUS_NEW_RED_TO_BLACK_THEN_STABILITY_V1",
       goal="Generate all 152 original EV-axis ticket methods × 20 odds bands; filter individual tickets by only pre-race horse signals, reject every 2023/24/25 red or jackpot-unstable strategy, deduplicate true physical tickets and measure exact cost, return, ROI.",
       selection_period=[2023,2024,2025],
       selection_is_hindsight=True,
       no_leaked_finish_in_ticket_gate=True,
       final_odds_warning="Historical final quoted odds are not guaranteed to be available at bet cutoff; feasibility not independently verified",
       evaluation="descriptive IN-SAMPLE final merged 2023-25 performance, not prospective heldout walkforward",
       original_black_lines_count=245,
       original_all_three_year_black_count=11,
       original_red_to_black_found_before_final_stability=621,
       originally_unstable_group_not_automatically_adopted=True,
       candidate_three_year_black_line_gate_pairs=len(positive_all_years),
       selection_thresholds=POLICIES,
       selected_line_gate_counts={k:len(v) for k,v in selected.items()},
       physical_ticket_baseline_validation="All 3 years exact against frozen original 245, all3040, and stable original11",
       sealed_year_2026=True,
       cost="only free standard CPU and no artifacts/cache")
    (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FINAL_STABLE_AND_BLACK_ONLY_MERGED_PORTFOLIOS",json.dumps(metrics,ensure_ascii=False),flush=True)
    print("ORIGINAL_BLACK_PLUS_RED_TO_BLACK_MERGE_COMPLETE",flush=True)

if __name__=="__main__":main()
