#!/usr/bin/env python3
"""Union the six *already selected* EV-rank betting portfolios by real ticket.
No new bet selection, gate/stability filter, model refit, or 2026 evaluation.
"""
from __future__ import annotations
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path
import pandas as pd
import run_l2_ev_rank1_pruning20_same_stage_v1 as ev1
import run_l2_ev_rank2to6_pruning20_best_roi_v1 as ev2
from run_all_race_flow_ticket_audit_v1 import YEARS,TOTAL_RACES,assert_ok

SOURCE="research-results/l2-ev-rank1-same20-pruners-v1/run-38919159628/comparison_summary.json"
SOURCE_1="research-results/l2-ev-rank1-same20-pruners-v1/run-38919159628/rank-1"
SOURCE_26="research-results/l2-ev-rank2to6-pruning20-warmup-v1/run-38891800300"
ORIGINAL_EV1="research-results/l2-ticket-odds-band-profit-v1/EV軸_全152買い方_オッズ帯別_年度推移_詳細20区間.csv"
ORIGINAL_EV26="research-results/l2-ev-rank2to6-official-v1/run-38875364714/all_152x20_odds_rank2to6_red_black.csv.gz"
EXPECT={1:("MARKET_10",145224,17195580,454),
        2:("MARKET_25",108617,12570510,241),
        3:("PROB_25",164930,19919800,292),
        4:("MARKET_75",100930,12143790,206),
        5:("LEARNED_10",83894,10222240,519),
        6:("LEARNED_10",78997,9883500,292)}

def read_source():
    s=json.loads(Path(SOURCE).read_text(encoding="utf-8"))
    assert_ok(s["rank1_original_3040_input_exact"] and s["medium_strict_not_applied"],"source conditions mismatch")
    assert_ok(s["evaluate_2023_from"]=="2023-02-01" and s["evaluate_until"]=="2025-12-31","source period mismatch")
    a={int(r["ev_rank"]):r for r in s["comparisons"]}
    assert_ok(set(a)==set(EXPECT),"rank source missing")
    for rank,(method,n,pay,lines) in EXPECT.items():
        r=a[rank]
        assert_ok((r["adopted_method"],int(r["purchased_tickets"]),
                   int(r["return_yen"]),int(r["black_odds_lines"]))==(method,n,pay,lines),
                  ("adopted source drift",rank))
        assert_ok(int(r["stake_yen"])==n*100,"source stake drift")
    return a

def read_rank_lines(rank,method):
    source=Path(SOURCE_1 if rank==1 else f"{SOURCE_26}/rank-{rank}")
    out=set()
    with (source/"adopted_method_black_odds_lines.csv").open(encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            assert_ok(int(row["ev_rank"])==rank and row["policy"]==method,"method drift")
            order,k,fam,band=row["line"].split("|")
            out.add((order,int(k),fam,band))
    assert_ok(len(out)==EXPECT[rank][3],("selected line count",rank,len(out)))
    old={}
    with (source/"portfolio_by_policy_and_year.csv").open(encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            if row["policy"]==method:
                old[str(row["year"])]=row
    assert_ok(set(old)=={"2023","2024","2025","2023-2025"},"yearly source portfolio missing")
    return out,old

def compare_totals(observed,source):
    for field,key in (("n","n"),("hits","hits"),("payout","payout"),
                      ("races","races"),("hit_races","hit_races")):
        assert_ok(int(observed[field])==int(source[key]),
                  ("rank portfolio reconstruction mismatch",field,observed[field],source[key]))

def do_capture(a):
    rank=a.rank;policy,n,payout,num_lines=EXPECT[rank]
    source=read_source()[rank]
    active,reference_years=read_rank_lines(rank,policy)
    mod=ev1 if rank==1 else ev2
    archival=mod.baseline_from_saved(ORIGINAL_EV1 if rank==1 else ORIGINAL_EV26,rank)
    data,picks=mod.ranklab.load_predictions(a.predictions,a.picks)
    races=pd.read_csv(a.racewise,compression="infer",
                      dtype={"race_id":str,"VALUE_EV_horse":str,"AI_RAW_TOP_horse":str})
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert_ok(len(races)==TOTAL_RACES and races.race_id.is_unique and set(races.year)==set(YEARS),
              "archival race drift")
    stats,thresholds,training,coverage=mod.ledger_pass(
        data,picks,races,Path(a.backfill_root),rank,archival)
    black,_=mod.black_lines(stats)
    assert_ok(black[policy]==active,("adopted profitable lines mismatch",rank,policy))
    assert_ok(coverage["2023"]["training_only_january_races"]==312,"January warmup mismatch")
    target=Path(a.out_dir);target.mkdir(parents=True,exist_ok=True)
    totals={year:dict(n=0,hits=0,payout=0,races=0,hit_races=0) for year in YEARS}
    path=target/"adopted_physical_tickets.csv.gz"
    with gzip.open(path,"wt",newline="",encoding="utf-8",compresslevel=6) as f:
        w=csv.writer(f)
        w.writerow(("year","race_id","bet_type","selection","payout_yen"))
        for year in YEARS:
            yr=races[races.year.eq(year)]
            groups=data[data.year.eq(year)].groupby("_race_id",sort=False)
            ct=totals[year];seen_races=0
            for rid,pack,price,g,info in mod.iterate_races(year,groups,yr,Path(a.backfill_root)):
                if not mod.pruning_eval_eligible(year,info["race_date"]):continue
                axis=mod.choose_axis(rid,g,info,picks,rank)
                if axis is None:continue
                shim=info.copy();shim["VALUE_EV_horse"]=axis
                dedup={}
                for tid,ticket,card in mod.fixed.create_cards(g,pack,price,rid,axis,shim):
                    if tid not in active:continue
                    prev=dedup.get(ticket)
                    if prev is not None:
                        assert_ok(prev==card,("same-rank ticket amount mismatch",rank,rid,ticket))
                    else:dedup[ticket]=card
                r_bought=False;r_hit=False
                for (kind,seq),card in dedup.items():
                    if not mod.fixed.mask(card,training[year],thresholds[year])[policy]:continue
                    refund=int(card[1])
                    w.writerow((year,rid,kind,",".join(str(int(i)) for i in seq),refund))
                    ct["n"]+=1;ct["hits"]+=int(refund>0);ct["payout"]+=refund
                    r_bought=True;r_hit|=(refund>0)
                ct["races"]+=int(r_bought);ct["hit_races"]+=int(r_hit)
                seen_races+=1
                if seen_races%1300==0:print("UNION_CAPTURE_PROGRESS",rank,year,seen_races,flush=True)
            compare_totals(ct,reference_years[str(year)])
            print("UNION_RANK_YEAR_EXACT",rank,year,ct,flush=True)
    combined={k:sum(x[k] for x in totals.values()) for k in totals[YEARS[0]]}
    compare_totals(combined,reference_years["2023-2025"])
    compare_totals(combined,{"n":n,"hits":source["winning_tickets"],
                            "payout":payout,"races":reference_years["2023-2025"]["races"],
                            "hit_races":reference_years["2023-2025"]["hit_races"]})
    result=dict(rank=rank,method=policy,selected_lines=len(active),
                original_3040_per_year_exact=True,
                black_lines_match_prior_study=True,
                every_year_portfolio_exact=True,years={str(y):v for y,v in totals.items()},
                combined=combined,source_run=(38919159628 if rank==1 else 38891800300),
                january_2023_training_excluded=True,medium_strict_applied=False)
    (target/"capture_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("CAPTURED_RANK_EXACT",json.dumps(result,ensure_ascii=False),flush=True)

def merge_file(path,rnk,individual,unique):
    with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            year=int(row["year"]);assert_ok(year in YEARS,"2026 sealed")
            ret=int(row["payout_yen"])
            key=(row["race_id"],row["bet_type"],row["selection"])
            assert_ok(all(part.isdigit() for part in row["selection"].split(",")),
                      "ticket sequence invalid")
            c=individual[rnk,year]
            c["n"]+=1;c["hits"]+=int(ret>0);c["payout"]+=ret
            old=unique.get(key)
            if old is None:unique[key]=(year,ret,1<<(rnk-1))
            else:
                assert_ok((old[0],old[1])==(year,ret),("cross-rank payout drift",rnk,key,old,ret))
                assert_ok(not old[2]&(1<<(rnk-1)),("duplicate *within* rank",rnk,key))
                unique[key]=(year,ret,old[2]|(1<<(rnk-1)))

def do_merge(a):
    source=read_source()
    base=Path(a.out_dir);base.mkdir(parents=True,exist_ok=True)
    unique={}
    individual=defaultdict(lambda:dict(n=0,hits=0,payout=0))
    for rank in EXPECT:
        folder=base/f"rank-{rank}"
        cap=json.loads((folder/"capture_summary.json").read_text(encoding="utf-8"))
        assert_ok(cap["rank"]==rank and cap["every_year_portfolio_exact"]
                  and cap["original_3040_per_year_exact"] and cap["black_lines_match_prior_study"],
                  ("rank capture invariants",rank))
        merge_file(folder/"adopted_physical_tickets.csv.gz",rank,individual,unique)
        combined={k:sum(individual[rank,y][k] for y in YEARS) for k in ("n","hits","payout")}
        assert_ok(combined["n"]==source[rank]["purchased_tickets"]
                  and combined["hits"]==source[rank]["winning_tickets"]
                  and combined["payout"]==source[rank]["return_yen"],
                  ("source adopted portfolio disagreement",rank,combined))
        for year in YEARS:
            assert_ok(individual[rank,year]["n"]==cap["years"][str(year)]["n"],
                      ("capture CSV loss",rank,year))
    assert_ok(sum(source[r]["purchased_tickets"] for r in EXPECT)==682592,
              "pre-dedup input drift")
    totals=defaultdict(lambda:dict(tickets=0,hit_tickets=0,payout_yen=0,
                                   overlap_tickets=0,overlap_payout_yen=0,
                                   bought_race_ids=set(),winning_race_ids=set()))
    count_multiplicity=defaultdict(int)
    for (race_id,kind,sel),(year,ret,mask) in unique.items():
        count=mask.bit_count()
        u=totals[year]
        u["tickets"]+=1;u["hit_tickets"]+=int(ret>0);u["payout_yen"]+=ret
        u["bought_race_ids"].add(race_id)
        if ret>0:u["winning_race_ids"].add(race_id)
        u["overlap_tickets"]+=count-1
        u["overlap_payout_yen"]+=ret*(count-1)
        count_multiplicity[count]+=1
    rows=[]
    for year in YEARS:
        x=totals[year]
        row=dict(year=year,unique_tickets=x["tickets"],unique_hit_tickets=x["hit_tickets"],
                 unique_bought_races=len(x["bought_race_ids"]),
                 unique_winning_races=len(x["winning_race_ids"]),
                 cross_rank_duplicates_removed=x["overlap_tickets"],
                 stake_yen=x["tickets"]*100,return_yen=x["payout_yen"],
                 profit_yen=x["payout_yen"]-x["tickets"]*100,
                 roi_pct=100*x["payout_yen"]/(x["tickets"]*100),
                 duplicate_stake_removed_yen=x["overlap_tickets"]*100,
                 duplicate_payout_removed_yen=x["overlap_payout_yen"])
        assert_ok(row["unique_tickets"]+row["cross_rank_duplicates_removed"]==
                  sum(individual[rank,year]["n"] for rank in EXPECT),
                  ("gross/unique ticket accounting mismatch",year))
        rows.append(row)
    summary={}
    for name in ("unique_tickets","unique_hit_tickets","cross_rank_duplicates_removed",
                 "stake_yen","return_yen","profit_yen","duplicate_stake_removed_yen","duplicate_payout_removed_yen"):
        summary[name]=sum(row[name] for row in rows)
    summary["year"]="2023-2025"
    summary["roi_pct"]=100*summary["return_yen"]/summary["stake_yen"]
    summary["unique_bought_races"]=len(set().union(*(totals[y]["bought_race_ids"] for y in YEARS)))
    summary["unique_winning_races"]=len(set().union(*(totals[y]["winning_race_ids"] for y in YEARS)))
    assert_ok(summary["unique_tickets"]+summary["cross_rank_duplicates_removed"]==682592,
              "six-rank accounting mismatch")
    rows.append(summary)
    with (base/"physical_ticket_union_by_year.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    gross_n=682592;gross_pay=sum(source[r]["return_yen"] for r in EXPECT)
    assert_ok(gross_pay-summary["duplicate_payout_removed_yen"]==summary["return_yen"],
              "payout dedup accounting mismatch")
    result=dict(contract="EV_RANK1_TO6_ADOPTED_PHYSICAL_TICKET_UNION_V1",
        no_method_reselection=True,method_by_rank={str(r):EXPECT[r][0] for r in EXPECT},
        source_comparison_run=38919159628,source_other_ranks_run=38891800300,
        original_rank1_ANCHOR_FULL_kept=True,
        exactly_same_race_bet_type_ordered_selection_dedup=True,
        different_ticket_types_not_collapsed=True,
        no_stability_medium_strict=True,years_2023_feb_to_2025_dec_races=10056,
        gross_ticket_count=gross_n,gross_stake_yen=100*gross_n,
        gross_return_yen=gross_pay,gross_profit_yen=gross_pay-100*gross_n,
        gross_roi_pct=100*gross_pay/(100*gross_n),
        deduplicated=summary,year_by_year=rows[:-1],
        duplicate_multiplicity={str(k):v for k,v in sorted(count_multiplicity.items())},
        all_six_original_per_rank_annual_counts_and_payouts_exact=True,
        posthoc_profit_band_selection=True,final_odds_not_bet_time=True,year_2026_sealed=True)
    (base/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SIX_RANK_UNION_EXACT_COMPLETE",json.dumps(dict(gross=gross_n,unique=summary,
                         removed=summary["cross_rank_duplicates_removed"]),ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    s=p.add_subparsers(dest="mode",required=True)
    a=s.add_parser("capture")
    a.add_argument("--rank",type=int,choices=tuple(EXPECT),required=True)
    for key in ("predictions","picks","racewise","backfill-root","out-dir"):
        a.add_argument("--"+key,required=True)
    b=s.add_parser("combine")
    b.add_argument("--out-dir",required=True)
    arg=p.parse_args()
    if arg.mode=="capture":do_capture(arg)
    else:do_merge(arg)

if __name__=="__main__":main()
