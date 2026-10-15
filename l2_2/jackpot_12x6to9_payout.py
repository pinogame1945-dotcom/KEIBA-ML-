#!/usr/bin/env python3
"""Payout arithmetic for fixed 1st+2nd popularity anchors, 6th-9th opponents.

Uses existing complete 2023-25 retrospective archival exports only.
No predictions, no betting, no data downloads, no 2026 data.
Closing-market rank is not necessarily available before cutoff.
64 multi-paid outcomes in parent cohort are audited, not fabricated.
"""
import argparse
import csv
import gzip
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean,median

EXPECTED_RACES=10368
YEARS=(2023,2024,2025)
EXPECTED_JACKPOT_12_6TO9=592
SOURCE_ARCHIVE="research-results/l2-2-trifecta-jackpot-anatomy-v1/run-38761110958/all_races_outcome_anatomy.csv.gz"
SOURCE_PREDICTIONS="research-results/market-fixed-anchor-residual-v2/run-38659856085/oos-fair-value-predictions.csv.gz"
RESULT_NAMES=("2023-2025","2023","2024","2025")

def need(condition,msg):
    if not condition:raise ValueError(msg)

def buck(p):
    if p<10000:return "UNDER_10000"
    if p<30000:return "10000_29999"
    if p<100000:return "30000_99999"
    if p<1000000:return "100000_999999"
    return "1000000_PLUS"

def quantile(vals,q):
    if not vals:return None
    s=sorted(vals)
    ix=(len(s)-1)*q
    lo=math.floor(ix);hi=math.ceil(ix)
    return s[lo]*(hi-ix)+s[hi]*(ix-lo) if lo!=hi else float(s[lo])

def stored_data(path):
    with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
        d={r["race_id"]:r for r in csv.DictReader(f)}
    need(len(d)==EXPECTED_RACES,"saved fact table missing races or has duplicates")
    need(set(int(x["year"]) for x in d.values())==set(YEARS),"2026 sealed")
    return d

def rank_by_race(path):
    ranks=defaultdict(Counter)
    horses=defaultdict(set)
    with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            race=r["_race_id"]
            hid=r["_horse_id"]
            need(hid not in horses[race],"duplicate runner for "+race)
            horses[race].add(hid)
            raw=r.get("market_rank")
            need(raw not in (None,""),"missing rank for "+race)
            k=int(float(raw))
            need(1<=k<=30,"invalid rank "+str(k)+" for "+race)
            ranks[race][k]+=1
    need(len(ranks)==EXPECTED_RACES,"OOS per-horse races not complete")
    return ranks

def report_segment(rows):
    n=len(rows)
    stake=sum(x["stake_yen"] for x in rows)
    wins=[x for x in rows if x["return_yen"]>0]
    jackpot=[x for x in wins if x["return_yen"]>=10000]
    payouts=[x["return_yen"] for x in jackpot]
    all_pay=[x["return_yen"] for x in wins]
    gross=sum(all_pay)
    band=Counter(buck(x["return_yen"]) for x in wins)
    return dict(races=n,betting_races=sum(x["stake_yen"]>0 for x in rows),
        selection_total=sum(x["lines"] for x in rows),
        stake_yen=stake,winning_races=len(wins),jackpot_winning_races=len(jackpot),
        nonjackpot_winning_races=len(wins)-len(jackpot),
        jackpot_payout_total_yen=sum(payouts),
        nonjackpot_payout_total_yen=gross-sum(payouts),
        return_yen=gross,profit_yen=gross-stake,
        roi_pct=100*gross/stake if stake else None,
        jackpot_payout_mean_yen=mean(payouts) if payouts else None,
        jackpot_payout_median_yen=median(payouts) if payouts else None,
        jackpot_payout_min_yen=min(payouts) if payouts else None,
        jackpot_payout_max_yen=max(payouts) if payouts else None,
        jackpot_payout_p25_yen=quantile(payouts,.25),
        jackpot_payout_p75_yen=quantile(payouts,.75),
        break_even_avg_yen_if_only_jackpot_wins=stake/len(jackpot) if jackpot else None,
        break_even_avg_yen_all_wins=stake/len(wins) if wins else None,
        official_multi_payout_selected_rows=sum(x["multi_payout"] for x in wins),
        payouts_by_band=dict(sorted(band.items())))

def run(args):
    fact=stored_data(args.anatomy)
    ranks=rank_by_race(args.predictions)
    need(fact.keys()==ranks.keys(),"saved result and prediction race IDs mismatch")
    rows=[];audit=Counter()
    for rid,r in sorted(fact.items()):
        pop=ranks[rid]
        exact=pop.get(1,0)==1 and pop.get(2,0)==1
        candidates=sum(pop.get(i,0) for i in range(6,10))
        if not exact:audit["missing_or_ambiguous_rank1_rank2"]+=1
        if candidates!=4:audit["opponent_count_not_four"]+=1
        if not exact: candidates=0
        lines=6*candidates
        payment=int(r["max_trifecta_refund_yen"])
        winners=tuple(map(int,r["winner_popularity_ranks"].split("-")))
        expected=len(winners)==3 and 1 in winners and 2 in winners and any(6<=x<=9 for x in winners)
        jackpot=payment>=10000
        audit["postrace_12_6to9_any_payout"]+=int(expected)
        audit["postrace_12_6to9_jackpot"]+=int(expected and jackpot)
        selected=exact and candidates>0 and expected
        if expected and not selected:audit["matching_but_not_buyable"]+=1
        win_yen=payment if selected else 0
        multipaid=int(r["paid_trifecta_outcomes"])>1
        if selected and multipaid:audit["multi_paid_selected"]+=1
        if selected and jackpot:audit["selected_jackpot"]+=1
        rows.append(dict(year=int(r["year"]),race_id=rid,stake_yen=100*lines,
                         lines=lines,return_yen=win_yen,multi_payout=int(multipaid)))
    need(audit["postrace_12_6to9_jackpot"]==EXPECTED_JACKPOT_12_6TO9,
         "reference 592 should reconcile before reporting")
    need(len(rows)==EXPECTED_RACES,"wrong total")
    summaries={"2023-2025":report_segment(rows)}
    for y in YEARS:
        slice=[r for r in rows if r["year"]==y]
        need(len(slice)==3456,"year coverage wrong")
        summaries[str(y)]=report_segment(slice)
    result=dict(contract="KEIBA_L22_JACKPOT_12_6TO9_PAYOUT_V1",
        hypothesis="1st and 2nd market favorites as two-anchors 6-way TRIFECTA multi to 6th-9th ranks",
        immutable_source_fact=SOURCE_ARCHIVE,immutable_source_predictions=SOURCE_PREDICTIONS,
        years=list(YEARS),year_2026_sealed=True,
        output_basis="official post-race TRIFECTA settled refund JPY for 100-yen physical winning ticket",
        note_24_line="24 tickets only when all 4 market ranks 6-9 exist with unique rank1/rank2; no phantom bets for missing runners",
        disclaimer="This uses post-close ranks and maximum-payout representative in multi-paid dead heats. Because additional winning dead-heat tickets are omitted, gross return is a LOWER BOUND, not exact ledger. Never present as verified executable ROI.",
        hypothetical_fixed24_all_races_stake_yen=EXPECTED_RACES*24*100,
        hypothetical_fixed24_jackpot_only_required_mean_yen=EXPECTED_RACES*24*100/EXPECTED_JACKPOT_12_6TO9,
        audits=dict(audit),periods=summaries)
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print("JACKPOT_PAYOUT_REPORT_START",flush=True)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    print("JACKPOT_PAYOUT_REPORT_END",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--anatomy",default=SOURCE_ARCHIVE)
    p.add_argument("--predictions",default=SOURCE_PREDICTIONS)
    p.add_argument("--output",default="out/jackpot-12x6to9-payout-summary.json")
    args=p.parse_args()
    need(buck(9999)=="UNDER_10000" and buck(10000)=="10000_29999","threshold")
    run(args)
