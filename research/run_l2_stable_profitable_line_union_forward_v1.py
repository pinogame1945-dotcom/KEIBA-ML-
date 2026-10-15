#!/usr/bin/env python3
"""Stable line selection -> exact unique physical bets portfolio, 2024/25 forward.

FROZEN 152 betting styles x 20 final quote odds bands, frozen EV axis horse.
Candidate rows are selected only using strictly earlier-year aggregated ticket
results.  Top-1/top-3 actual payoffs of training-year winners are inspected
to reject jackpot-dependent rules. Never touches 2026.

This is still retrospective market data: quote odds are final, unavailable
before market close, and pre-existing L1 model features must be pre-race.
The historical black245/converted621 classes are diagnostic-only and NEVER
used as training-rule eligibility. Never claim OOS in the full sense of
locked research hypothesis because these specs were designed after inspecting
2023-25 studies. Never select a final policy by current 2024/25 profit.
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
  "COVERAGE":dict(n=250,hits=15,min_roi=100.0,max1=.40,max3=.70,
                  minus3_roi=85.0, require_all_years_positive=False),
  "ROBUST":dict(n=500,hits=25,min_roi=105.0,max1=.30,max3=.55,
                minus3_roi=95.0, require_all_years_positive=False),
  "STRICT":dict(n=1000,hits=40,min_roi=110.0,max1=.20,max3=.40,
                minus3_roi=100.0, require_all_years_positive=True),
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
        if y not in ("2023","2024"):continue
        key=key_from_row(row)
        assert key not in years[y],(y,key)
        stats=dict(n=int(row["filtered_tickets"]),hits=int(row["filtered_hit_tickets"]),
                   return_yen=int(row["filtered_return_yen"]))
        assert stats["return_yen"]>=0 and 0<=stats["hits"]<=stats["n"]
        years[y][key]=stats
    assert all(len(years[str(y)])==3040*len(GATES) for y in (2023,2024))
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
    eval_year=year in (2024,2025)
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
    originals=lens.load_original(a.bands)
    plus={tuple(map(str,tid)) for tid,row in originals.items() if int(row["2023-2025_損益円"])>0}
    assert len(plus)==245
    by_year=historical_summary(a.original_lens_lines)
    prelim24=preliminary(by_year,(2023,))
    prelim25=preliminary(by_year,(2023,2024))
    print("PRELIM_PRIOR_ONLY",json.dumps(dict(test24=len(prelim24),test25=len(prelim25))),
          flush=True)
    assert all(key[1] in GATES for key in prelim24|prelim25)
    pred,picks=lens.frame(a.predictions,a.picks)
    races=pd.read_csv(a.racewise,compression="infer",dtype={
      "race_id":str,"VALUE_EV_horse":str,"AI_RAW_TOP_horse":str})
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert len(races)==TOTAL_RACES and races.race_id.is_unique and set(races.year)==set(YRS)
    root=Path(a.backfill_root)
    # Need exact max-3 payout contributions for both prospective folds.
    h23,_=one_year(2023,pred,picks,races,root,prelim24|prelim25,{},plus)
    sel24={};audits=[]
    for policy in POLICIES:
        sel,rows=decide(prelim24,by_year,(2023,),h23,policy)
        sel24[policy]=sel;audits+=rows
    print("STABLE_SELECTED_2024_TRAIN_2023",
          {p:len(s) for p,s in sel24.items()},flush=True)
    h24,out24=one_year(2024,pred,picks,races,root,prelim25,sel24,plus)
    merged=merge_hist(h23,h24)
    sel25={}
    for policy in POLICIES:
        sel,rows=decide(prelim25,by_year,(2023,2024),merged,policy)
        sel25[policy]=sel;audits+=rows
    print("STABLE_SELECTED_2025_TRAIN_2023_24",
          {p:len(s) for p,s in sel25.items()},flush=True)
    _,out25=one_year(2025,pred,picks,races,root,set(),sel25,plus)
    # Exact source 245 union, no change to old golden baseline before filters.
    expected={r["year"]:r for r in read(a.old_union) if r["group"]=="THREE_YEAR_TOTAL_PLUS"}
    for y,outputs in ((2024,out24),(2025,out25)):
        c=outputs["HINDSIGHT_BLACK_245"];r=expected[str(y)]
        for source,target in (("races_bought","races_bought"),("unique_tickets","tickets"),
                              ("hit_tickets","hits"),("return_yen","return_yen")):
            assert int(r[source])==c[target],("original black baseline drift",y,source,c[target],r[source])
    print("EXACT_245_HINDSIGHT_UNION_ORIGINAL_VERIFIED_2024_2025",flush=True)
    # Full universe initial portfolio also must match original 3040 unfiltered
    # physical tickets before any strategy selection.
    base_all={r["year"]:r for r in read(
      "research-results/l2-frozen-odds-l1-lenses-v1/run-38694334351/union_black245_other2795_x_existing_axes_yearly.csv")
          if r["scope"]=="ALL_3040" and r["gate"]=="ALL"}
    for y,outputs in ((2024,out24),(2025,out25)):
        c=outputs["ALL_FIXED_3040"];r=base_all[str(y)]
        for source,target in (("baseline_races_bought","races_bought"),
          ("baseline_tickets","tickets"),("baseline_hit_tickets","hits"),
          ("baseline_return_yen","return_yen")):
            assert int(r[source])==c[target],("all3040 baseline drift",y,source)
    print("EXACT_ALL3040_UNION_VERIFIED_2024_2025",flush=True)
    rows=[]
    for name in list(POLICIES)+list(DIAG):
        rows.append(result_row("2024",name,out24[name],3456))
        rows.append(result_row("2025",name,out25[name],3456))
        rows.append(result_row("2024-2025",name,pool([out24[name],out25[name]]),6912))
    write(out/"stable_portfolio_chronological_2024_2025.csv",rows)
    assert audits,("no prelim audit",len(prelim24),len(prelim25))
    write(out/"all_training_line_gate_stability_decisions.csv",audits)
    all_accepted=[]
    for y,s in ((2024,sel24),(2025,sel25)):
        for policy,selected in s.items():
            for (tid,g),stat in sorted(selected.items()):
                all_accepted.append(dict(test_year=y,policy=policy,original_line="|".join(tid),
                  gate=g,original_2023_2025_black_group=tid in plus,
                  training_tickets=stat["n"],training_wins=stat["hits"],
                  training_profit_yen=stat["profit_yen"],training_roi_pct=stat["roi_pct"]))
    if not all_accepted:all_accepted=[{
        "test_year":"","policy":"","original_line":"","gate":"","original_2023_2025_black_group":"",
        "training_tickets":0,"training_wins":0,"training_profit_yen":0,"training_roi_pct":""}]
    write(out/"accepted_stable_lines_by_training_year.csv",all_accepted)
    manifest=dict(contract="EV152X20_STABLE_PROFIT_LINES_ONLY_EXACT_UNION_WF_V1",
        policy_thresholds=POLICIES,
        prelim=BASIC,
        selected_counts={str(y):{k:len(v) for k,v in selections.items()}
                      for y,selections in ((2024,sel24),(2025,sel25))},
        initial_prefilter={"2024":len(prelim24),"2025":len(prelim25)},
        immutable_bet_templates="Original 152 styles x original 20 final-quote bins; no newly generated bet.",
        selection="previous years only, plus exact training payout top3 concentration. Reject unstable and nonprofitable lines, do not use future year for rule adoption.",
        diagnostic_only="2023-25 hindsight-black245 compared but NEVER used as selector",
        final_policy_selection="NONE: report all three prespecified variants; do not cherry-pick ROI from 2024/2025",
        no_2026=True,bet_unit_yen=100,
        warning_final_odds="Historical final odds only; untradeable as pre-close price proof",
        warning_research="Hypothesis and thresholds set after observing earlier 2023-2025 research; strictly prior-year algorithm alone is not a clean untouched external holdout",
        runner="free standard CPU. pinned BACKFILL, no paid artifacts/caches")
    (out/"summary.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("STABLE_POLICY_PORTFOLIOS",json.dumps(rows,ensure_ascii=False),flush=True)
    print("STABLE_EXACT_UNION_COMPLETE",flush=True)

if __name__=="__main__":main()
