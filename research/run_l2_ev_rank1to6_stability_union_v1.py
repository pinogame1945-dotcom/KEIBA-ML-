#!/usr/bin/env python3
"""Original STABLE_MEDIUM/STRICT and 3-year annual-black portfolios: 6-rank physical union."""
from __future__ import annotations
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path
from run_all_race_flow_ticket_audit_v1 import YEARS,assert_ok
import run_l2_ev_rank1to6_physical_union_v1 as previous

POLICIES=("ALL_YEAR_BLACK","STABLE_MEDIUM","STABLE_STRICT")
BASELINE_DIR="research-results/l2-ev-rank1to6-physical-union-v1/run-38922914983"
PREVIOUS_OFFICIAL={
    2:{"ALL_YEAR_BLACK":(34437,4314930),"STABLE_MEDIUM":(0,0),"STABLE_STRICT":(0,0)},
    3:{"ALL_YEAR_BLACK":(46987,5953720),"STABLE_MEDIUM":(0,0),"STABLE_STRICT":(0,0)},
    4:{"ALL_YEAR_BLACK":(14236,2143880),"STABLE_MEDIUM":(0,0),"STABLE_STRICT":(0,0)},
    5:{"ALL_YEAR_BLACK":(14066,1739900),"STABLE_MEDIUM":(6376,746870),"STABLE_STRICT":(0,0)},
    6:{"ALL_YEAR_BLACK":(5044,726410),"STABLE_MEDIUM":(0,0),"STABLE_STRICT":(0,0)},
}

def stage_record(path,rank,policy,ref):
    totals=defaultdict(lambda:dict(tickets=0,hits=0,payout=0))
    with gzip.open(path,"rt",encoding="utf-8",newline="") as h:
        reader=csv.DictReader(h)
        assert_ok(set(reader.fieldnames)==set(("year","race_id","bet_type","selection","payout_yen")),
                  "physical ticket output schema changed")
        for row in reader:
            year=int(row["year"])
            assert_ok(year in YEARS,"2026 must remain sealed")
            ret=int(row["payout_yen"])
            assert_ok(ret>=0,"invalid payout")
            k=(year,row["race_id"],row["bet_type"],row["selection"])
            assert_ok(k not in ref,("duplicate within same rank",rank,policy,k))
            ref[k]=ret
            r=totals[year];r["tickets"]+=1;r["hits"]+=int(ret>0);r["payout"]+=ret
    return totals

def merge_policy(root,policy,baseline):
    by_rank={}
    global_bets={}
    duplicates=defaultdict(lambda:dict(n=0,returns=0))
    for rank in range(1,7):
        directory=root/f"rank-{rank}"
        summ=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        assert_ok(summ["ev_rank"]==rank and summ["adopted_pruner"]==previous.EXPECT[rank][0],
                  ("rank or fixed source method changed",rank))
        assert_ok(summ["pruning_source_counts_and_payouts_exactly_reconciled"],
                  ("rank source unreconciled",rank))
        assert_ok(summ["source_original_black_lines"]==previous.EXPECT[rank][3],
                  ("original lines drift",rank))
        policy_results={r["policy"]:r for r in summ["three_year_results"].values()}
        expected=policy_results[policy]
        raw={}
        stats=stage_record(directory/(policy.lower()+"_physical.csv.gz"),rank,policy,raw)
        n=sum(r["tickets"] for r in stats.values())
        pay=sum(r["payout"] for r in stats.values())
        assert_ok((n,pay)==(int(expected["tickets"]),int(expected["payout_yen"])),
                  ("source stage count mismatch",rank,policy))
        if rank in PREVIOUS_OFFICIAL:
            assert_ok((n,pay)==PREVIOUS_OFFICIAL[rank][policy],
                      ("prior rank2-6 official stability study drift",rank,policy))
        by_rank[str(rank)]=dict(tickets=n,hit_tickets=sum(r["hits"] for r in stats.values()),
                              stake_yen=100*n,return_yen=pay,
                              roi_pct=100*pay/(100*n) if n else None)
        for (year,rid,kind,sel),ret in raw.items():
            key=(year,rid,kind,sel)
            old=global_bets.get(key)
            if old is None:global_bets[key]=(ret,1<<(rank-1))
            else:
                assert_ok(old[0]==ret,("different refund for same real ticket",rank,key,old,ret))
                assert_ok(not old[1]&(1<<(rank-1)),("double count same rank",rank,key))
                global_bets[key]=(ret,old[1]|(1<<(rank-1)))
                d=duplicates[year];d["n"]+=1;d["returns"]+=ret
    year_rows=[]
    for year in YEARS:
        bets=[(key,v) for key,v in global_bets.items() if key[0]==year]
        n=len(bets);payout=sum(v[0] for k,v in bets)
        hit=sum(v[0]>0 for k,v in bets)
        race_ids={k[1] for k,v in bets}
        hit_races={k[1] for k,v in bets if v[0]>0}
        dup=duplicates[year]
        year_rows.append(dict(policy=policy,period=str(year),purchased_races=len(race_ids),
                    hit_races=len(hit_races),unique_tickets=n,unique_hits=hit,
                    stake_yen=100*n,return_yen=payout,profit_yen=payout-100*n,
                    roi_pct=100*payout/(100*n) if n else None,
                    cross_rank_duplicates_removed=dup["n"],
                    duplicate_payout_yen=dup["returns"]))
    pooled={k:sum(r[k] for r in year_rows) for k in
            ("purchased_races","hit_races","unique_tickets","unique_hits",
             "stake_yen","return_yen","profit_yen",
             "cross_rank_duplicates_removed","duplicate_payout_yen")}
    pooled.update(dict(policy=policy,period="2023-2025",
                       roi_pct=100*pooled["return_yen"]/pooled["stake_yen"]
                       if pooled["stake_yen"] else None))
    assert_ok(pooled["unique_tickets"]+pooled["cross_rank_duplicates_removed"]==
              sum(x["tickets"] for x in by_rank.values()),
              ("gross vs dedup count mismatch",policy))
    assert_ok(pooled["return_yen"]+pooled["duplicate_payout_yen"]==
              sum(x["return_yen"] for x in by_rank.values()),
              ("gross vs dedup payouts mismatch",policy))
    assert_ok(set(global_bets).issubset(baseline),
              ("stabilization added a ticket outside the original 653734",policy))
    for key,(ret,_) in global_bets.items():
        assert_ok(baseline[key]==ret,("settled payout differs from original baseline",policy,key))
    print("STABILIZED_UNION",policy,pooled,flush=True)
    return year_rows,pooled,by_rank,set(global_bets)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",required=True)
    a=ap.parse_args()
    root=Path(a.out_dir);root.mkdir(parents=True,exist_ok=True)
    source=json.loads((Path(BASELINE_DIR)/"summary.json").read_text(encoding="utf-8"))
    assert_ok(source["deduplicated"]["unique_tickets"]==653734
              and source["deduplicated"]["return_yen"]==77239240,"original deduplicated baseline mismatch")
    baseline={}
    for rank in range(1,7):
        path=Path(BASELINE_DIR)/f"rank-{rank}"/"adopted_physical_tickets.csv.gz"
        with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
            for r in csv.DictReader(f):
                k=(int(r["year"]),r["race_id"],r["bet_type"],r["selection"])
                ret=int(r["payout_yen"])
                old=baseline.get(k)
                assert_ok(old is None or old==ret,"base physical refund differs")
                baseline[k]=ret
    assert_ok(len(baseline)==653734 and sum(baseline.values())==77239240,
              "original 6 rank physical inputs changed")
    rows=[];pools={};ranks={};sets={}
    for pol in POLICIES:
        annual,total,detail,keys=merge_policy(root,pol,baseline)
        rows.extend(annual+[total]);pools[pol]=total;ranks[pol]=detail;sets[pol]=keys
    assert_ok(sets["STABLE_STRICT"].issubset(sets["STABLE_MEDIUM"]),"strict not medium subset across ranks")
    assert_ok(sets["STABLE_MEDIUM"].issubset(sets["ALL_YEAR_BLACK"]),"medium not annual-black subset")
    with (root/"stability_union_by_year.csv").open("w",encoding="utf-8",newline="") as f:
        cols=list(rows[0]);w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    result=dict(contract="EV_RANK1TO6_ORIGINAL_OFFICIAL_STABILITY_POST_PRUNE_DEDUP_V1",
        original_untouched_union=source["deduplicated"],
        original_run=38922914983,fixed_comparison_source=38919159628,
        exact_official_17_gates_applied=True,
        official_selection_applied_to_rank_line_gate_not_directly_to_merged_tickets=True,
        official_medium=dict(n=500,hits=30,per_year_hits=8,max1_share=0.40,
                             max3_share=0.70,remove_top3_roi_floor=90),
        official_strict=dict(n=800,hits=50,per_year_hits=12,max1_share=0.30,
                             max3_share=0.55,remove_top3_roi_floor=100),
        annual_black_criterion="each 2023 Feb-Dec, full 2024 and 2025 year has >100% ROI",
        methods_unmodified={str(rank):previous.EXPECT[rank][0] for rank in range(1,7)},
        rank2to6_past_stability_reproduced=True,rank1_new_annual_gate_results=True,
        after_stability=pools,rank_specific=ranks,
        no_new_ticket_keys=True,no_med_strict_change=True,
        evaluation_races=10056,selection_uses_full_evaluation_period_posthoc=True,
        historical_final_odds_not_live=True,year_2026_sealed=True)
    (root/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("STABILITY_SIX_RANK_UNION_VERIFIED",json.dumps(pools,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
