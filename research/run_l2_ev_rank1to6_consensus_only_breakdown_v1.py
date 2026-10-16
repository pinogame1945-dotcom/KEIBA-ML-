#!/usr/bin/env python3
"""Read-only physical ticket agreement audit for fixed EV ranks 1-6.

Consensus = exactly the same (year, race ID, wager kind, ordered selection)
adopted by >=2 DISTINCT EV ranks, held ONCE (100 yen, one original payout).
No new source selection, gate tuning, strategy optimisation or 2026 evaluation.
"""
from __future__ import annotations
import argparse,csv,gzip,itertools,json
from collections import Counter,defaultdict
from pathlib import Path

import run_l2_ev_rank1to6_bet_type_breakdown_v1 as base

STAGES=base.STAGES
YEARS=base.YEARS
CATEGORIES=("CONSENSUS","SINGLE_RANK","TOTAL")
ALL_KINDS=tuple(sorted(base.JAPANESE))

def take(mask):
    return tuple(r for r in range(1,7) if mask&(1<<(r-1)))

def filter_category(stage_dict,group):
    if group=="TOTAL":return stage_dict
    return {k:v for k,v in stage_dict.items()
            if (v[1].bit_count()>1)==(group=="CONSENSUS")}

def audit_stage(stage,entries,source_summary,stable_summary):
    n=len(entries)
    hit=sum(v[0]>0 for v in entries.values())
    refund=sum(v[0] for v in entries.values())
    assert n==base.EXPECTED[stage][0],("ticket count drift",stage,n)
    assert hit==base.EXPECTED[stage][1],("hit count drift",stage,hit)
    assert refund==base.EXPECTED[stage][2],("return drift",stage,refund)
    counts=Counter(mask.bit_count() for ret,mask in entries.values())
    assert sum(counts.values())==n
    extra=sum((num-1)*freq for num,freq in counts.items())
    official=(source_summary["deduplicated"] if stage=="BASE"
              else stable_summary["after_stability"][stage])
    assert extra==official["cross_rank_duplicates_removed"],("cross rank duplicates",stage,extra)
    if stage=="BASE":
        assert {str(k):v for k,v in sorted(counts.items())}==source_summary["duplicate_multiplicity"],(
            "source multiplicity drift",counts)
    assert all(1<=num<=6 for num in counts)
    return counts,extra

def metric_row(stage,group,year,kind,entries):
    year_filter=set(YEARS) if year=="2023-2025" else {int(year)}
    events=[(k,v[0]) for k,v in entries.items()
            if k[0] in year_filter and (kind=="ALL" or k[2]==kind)]
    return base.metric(dict(stage=stage,subset=group,year=str(year),
                       ticket_type=kind,ticket_type_ja=base.JAPANESE.get(kind,"全券種")),
                       events)

def build_reports(grouped,stage):
    by_year=[];by_type=[];winners=[];pairs=[];rank_set=[]
    for category in CATEGORIES:
        d=filter_category(grouped,category)
        for year in (*YEARS,"2023-2025"):
            by_year.append(metric_row(stage,category,year,"ALL",d))
            for kind in ALL_KINDS:
                by_type.append(metric_row(stage,category,year,kind,d))
        # Largest actual winning physical tickets, independent of multiplicity
        if category=="CONSENSUS":
            winning=sorted(((k,v) for k,v in d.items() if v[0]>0),
                           key=lambda x:(-x[1][0],x[0]))[:30]
            for (year,rid,kind,selection),(ret,mask) in winning:
                winners.append(dict(stage=stage,year=year,race_id=rid,ticket_type=kind,
                    ticket_type_ja=base.JAPANESE[kind],horse_numbers=selection,
                    payout_yen=ret,payout_multiple=ret/100,
                    agreement_count=mask.bit_count(),ranks=",".join(map(str,take(mask)))))
    d=filter_category(grouped,"CONSENSUS")
    # Exact contributor combination (e.g. '1+4' vs '1+2+4').
    masks=defaultdict(dict)
    for k,(ret,mask) in d.items():
        masks[mask][k]=(ret,mask)
    for mask,selected in sorted(masks.items()):
        rec=metric_row(stage,"EXACT_RANK_SET","2023-2025","ALL",selected)
        rec.update(agree_ranks=",".join(map(str,take(mask))),
                   agreement_count=mask.bit_count())
        rank_set.append(rec)
    # Pairwise co-adoptions; a 3-rank ticket contributes to each relevant pair
    # within this separate diagnostic. Do not add these rows together.
    for a,b in itertools.combinations(range(1,7),2):
        mask=(1<<(a-1))|(1<<(b-1))
        selected={k:v for k,v in grouped.items() if v[1]&mask==mask}
        rec=metric_row(stage,"PAIRWISE_OVERLAP","2023-2025","ALL",selected)
        rec.update(rank_a=a,rank_b=b)
        pairs.append(rec)
    return by_year,by_type,winners,pairs,rank_set

def payout_bands(entries,stage):
    result=[]
    for kind in ALL_KINDS:
        hit=[ret for (y,r,t,s),(ret,mask) in entries.items() if t==kind and ret>0]
        for lower,upper,name in base.BANDS:
            selected=[v for v in hit if v>=lower and (upper is None or v<upper)]
            result.append(dict(stage=stage,ticket_type=kind,
                  ticket_type_ja=base.JAPANESE[kind],winning_payout_band=name,
                  winning_ticket_count=len(selected),winning_payout_yen=sum(selected)))
    return result

def physical_csv(path,entries,stage):
    with gzip.open(path,"wt",encoding="utf-8",newline="") as fh:
        writer=csv.writer(fh)
        writer.writerow(("stage","year","race_id","bet_type","selection",
                         "payout_yen","distinct_rank_count","contributing_ranks"))
        for (y,rid,kind,selection),(pay,mask) in sorted(entries.items()):
            writer.writerow((stage,y,rid,kind,selection,pay,mask.bit_count(),
                             ",".join(map(str,take(mask)))))

def write_csv(path,rows):
    assert rows,("empty CSV",path)
    with open(path,"w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",required=True)
    args=ap.parse_args()
    output=Path(args.out_dir);output.mkdir(parents=True,exist_ok=True)
    old=json.loads((base.SOURCE_BASE/"summary.json").read_text(encoding="utf-8"))
    stability=json.loads((base.SOURCE_STABLE/"summary.json").read_text(encoding="utf-8"))
    yearly=[];types=[];bigwins=[];pairs=[];ranksets=[];bands=[]
    results={};all_stages={}
    for stage in STAGES:
        entries,perrank=base.read_stage(stage)
        counts,extra=audit_stage(stage,entries,old,stability)
        all_stages[stage]=entries
        matching=filter_category(entries,"CONSENSUS")
        assert all(v[1].bit_count()>=2 for v in matching.values())
        physical_csv(output/f"{stage.lower()}_consensus_physical.csv.gz",matching,stage)
        a,b,c,d,e=build_reports(entries,stage)
        yearly+=a;types+=b;bigwins+=c;pairs+=d;ranksets+=e
        bands+=payout_bands(matching,stage)
        stage_row=next(x for x in a if x["subset"]=="CONSENSUS" and x["year"]=="2023-2025")
        solo_row=next(x for x in a if x["subset"]=="SINGLE_RANK" and x["year"]=="2023-2025")
        total_row=next(x for x in a if x["subset"]=="TOTAL" and x["year"]=="2023-2025")
        assert stage_row["tickets"]+solo_row["tickets"]==total_row["tickets"]
        assert stage_row["payout_yen"]+solo_row["payout_yen"]==total_row["payout_yen"]
        assert stage_row["hits"]+solo_row["hits"]==total_row["hits"]
        assert stage_row["tickets"]==sum(n for k,n in counts.items() if k>=2)
        for year in YEARS:
            n=next(x for x in a if x["subset"]=="TOTAL" and x["year"]==str(year))
            if stage=="BASE":
                ref=next(x for x in old["year_by_year"] if x["year"]==year)
                assert n["tickets"]==ref["unique_tickets"],("year source count drift",stage,year)
                assert n["payout_yen"]==ref["return_yen"],("year source payouts drift",stage,year)
        results[stage]=dict(consensus=stage_row,single_rank=solo_row,
                            all_tickets=total_row,
                            multiplicity={str(k):v for k,v in sorted(counts.items())},
                            cross_rank_duplicate_purchases_prevented=extra,
                            consensus_ticket_fraction=stage_row["tickets"]/len(entries) if entries else 0,
                            contributing_ranks_2_or_more=True)
        print("AGREED_PHYSICAL_TICKETS",stage,json.dumps({
             "tickets":stage_row["tickets"],"hits":stage_row["hits"],
             "ROI":stage_row["roi_pct"],"profit":stage_row["profit_yen"],
             "gross_duplicate_excess":extra},ensure_ascii=False),flush=True)
    # Stage inclusion does not imply agreement survives after filtering per rank.
    # It is essential not to wrongly assert matching <= base matching: a ticket
    # may have been adopted by two ranks pre-filter but only one post-filter.
    for s in STAGES[1:]:
        assert set(all_stages[s]).issubset(all_stages["BASE"]),("non-original ticket",s)
    write_csv(output/"consensus_vs_single_by_year.csv",yearly)
    write_csv(output/"consensus_vs_single_by_ticket_type_and_year.csv",types)
    write_csv(output/"consensus_exact_contributing_rank_sets.csv",ranksets)
    write_csv(output/"consensus_rank_pairwise_overlap.csv",pairs)
    write_csv(output/"consensus_largest_30_winners.csv",bigwins) if bigwins else None
    write_csv(output/"consensus_winning_payout_bands.csv",bands)
    info=dict(contract="EV_RANK1TO6_IDENTICAL_PHYSICAL_TICKETS_CONSENSUS_BREAKDOWN_V1",
              exact_ticket_identity="year,race_id,bet_type,ordered_selection",
              agreement="same physical ticket present for two or more DISTINCT EV ranks",
              one_100yen_bet_each="no duplicated purchases",
              original_fixed_rank_reductions_unchanged=True,
              no_pruning_or_selection_changes=True,
              read_only_historical_sources=[38922914983,38927480138],
              consensus_not_all_high_rank_agreement=True,
              same_three_year_period_posthoc_line_gate_selection=True,
              only_refunded_winner_bands_not_bet_time_odds=True,
              year_2026_sealed=True,results=results)
    (output/"summary.json").write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n",
                                        encoding="utf-8")
    print("ALL_CONSENSUS_AUDITS_PASS",flush=True)

if __name__=="__main__": main()
