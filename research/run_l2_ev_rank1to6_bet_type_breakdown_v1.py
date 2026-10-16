#!/usr/bin/env python3
"""Read-only audit of exact physical ticket portfolio (EV1-6) by ticket family.

All source tickets were already reconstructed and validated. This script DOES NOT
reselect source methods, profitable odds bands, stability gates, or any ticket.
No running of any model or historical BACKFILL. 2026 remains sealed.
"""
from __future__ import annotations
import argparse,csv,gzip,json,heapq
from collections import defaultdict,Counter
from pathlib import Path

SOURCE_BASE=Path("research-results/l2-ev-rank1to6-physical-union-v1/run-38922914983")
SOURCE_STABLE=Path("research-results/l2-ev-rank1to6-stability-union-v1/run-38927480138")
STAGES=("BASE","ALL_YEAR_BLACK","STABLE_MEDIUM","STABLE_STRICT")
YEARS=(2023,2024,2025)
EXPECTED={
 "BASE":(653734,3371,77239240),
 "ALL_YEAR_BLACK":(130958,774,16806420),
 "STABLE_MEDIUM":(7325,93,854060),
 "STABLE_STRICT":(949,62,107190),
}
JAPANESE={
 "WIN":"単勝","PLACE":"複勝",
 "QUINELLA":"馬連","WIDE":"ワイド","EXACTA":"馬単",
 "TRIO":"三連複","TRIFECTA":"三連単",
}
BANDS=((0,1000,"10倍未満"),(1000,3000,"10〜30倍未満"),
       (3000,10000,"30〜100倍未満"),(10000,30000,"100〜300倍未満"),
       (30000,100000,"300〜1000倍未満"),(100000,None,"1000倍以上"))

def assert_ok(ok,what):
    if not ok:raise ValueError(what)

def rows(file):
    with gzip.open(file,"rt",encoding="utf-8",newline="") as fh:
        reader=csv.DictReader(fh)
        assert_ok(tuple(reader.fieldnames)==("year","race_id","bet_type","selection","payout_yen"),
                  ("original ticket format",str(file)))
        for row in reader:
            year=int(row["year"])
            assert_ok(year in YEARS,("out-of-evaluation-year",year))
            p=int(row["payout_yen"])
            assert_ok(p>=0,("negative payout",row))
            key=(year,row["race_id"],row["bet_type"],row["selection"])
            assert_ok(row["bet_type"] in JAPANESE,("unknown original ticket kind",row["bet_type"]))
            yield key,p

def read_stage(stage):
    """Dedup true physical bets across rank; retain contributing rank mask for overlap."""
    distinct={}
    per_rank={}
    for rank in range(1,7):
        source= (SOURCE_BASE if stage=="BASE" else SOURCE_STABLE)
        fname=("adopted_physical_tickets.csv.gz" if stage=="BASE"
               else stage.lower()+"_physical.csv.gz")
        p=source/f"rank-{rank}"/fname
        assert_ok(p.is_file(),("source file missing",p))
        counted={}
        for key,refund in rows(p):
            assert_ok(key not in counted,("duplicate ticket within rank",rank,stage,key))
            counted[key]=refund
            prev=distinct.get(key)
            if prev is None:distinct[key]=(refund,1<<(rank-1))
            else:
                assert_ok(prev[0]==refund,("cross-rank payout mismatch",stage,key,prev,refund))
                assert_ok(not (prev[1]&(1<<(rank-1))),"repeated rank in contributor mask")
                distinct[key]=(refund,prev[1]|(1<<(rank-1)))
        per_rank[rank]=counted
    return distinct,per_rank

def metric(key,events):
    """Convert list of (ticket_key, refund) to audited row."""
    n=len(events); hits=[(r,k) for k,r in events if r>0]
    pay=sum(r for k,r in events);total_races=len({k[1] for k,r in events})
    hit_races=len({k[1] for r,k in hits})
    winners=heapq.nlargest(3,[r for r,k in hits])
    top1=winners[0] if winners else 0
    top3=sum(winners)
    return dict(**key,tickets=n,hits=len(hits),ticket_hit_rate_pct=100*len(hits)/n if n else None,
                races_bought=total_races,races_hit=hit_races,
                race_hit_rate_pct=100*hit_races/total_races if total_races else None,
                stake_yen=100*n,payout_yen=pay,profit_yen=pay-100*n,
                roi_pct=100*pay/(100*n) if n else None,
                biggest_win_payout_yen=top1,
                biggest3_win_payout_yen=top3,
                biggest3_share_of_payout_pct=100*top3/pay if pay else None,
                roi_without_biggest3_pct=100*(pay-top3)/(100*n) if n else None)

def make_metrics(events,stage,year,group_by_kind=False,rank=None):
    result=[]
    years=(str(year),) if year=="2023-2025" else (year,)
    ys=set(YEARS) if year=="2023-2025" else {int(year)}
    if group_by_kind:
        codes=sorted({k[2] for k,v in events})
    else:codes=("ALL",)
    for kind in codes:
        selected=[(k,ret) for k,ret in events
                  if k[0] in ys and (not group_by_kind or k[2]==kind)]
        name=(JAPANESE.get(kind,kind) if group_by_kind else "全券種")
        key={"stage":stage,"year":str(year),"ticket_type":kind,"ticket_type_ja":name}
        if rank is not None:key["ev_rank"]=rank
        result.append(metric(key,selected))
    return result

def csv_out(path,records):
    assert_ok(records,"empty rows cannot write")
    fields=list(records[0])
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader();w.writerows(records)

def payout_band(refund):
    for lo,hi,name in BANDS:
        if refund>=lo and (hi is None or refund<hi):return name
    raise ValueError("invalid refund")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True)
    a=p.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    source_json=json.loads((SOURCE_STABLE/"summary.json").read_text(encoding="utf-8"))
    assert_ok(source_json["original_untouched_union"]["unique_tickets"]==EXPECTED["BASE"][0],
              "source baseline count changed")
    categories=[];rank_types=[];payouts=[];top=[];overview=[];all_keys={}
    for stage in STAGES:
        unique,per_rank=read_stage(stage)
        assert_ok((len(unique),sum(int(v[0]>0) for v in unique.values()),
                   sum(v[0] for v in unique.values()))==EXPECTED[stage],
                  ("stage baseline mismatch",stage))
        all_keys[stage]=set(unique)
        union_events=[(k,v[0]) for k,v in unique.items()]
        rank_counts={str(rank):len(entries) for rank,entries in per_rank.items()}
        for year in (*YEARS,"2023-2025"):
            overview.extend(make_metrics(union_events,stage,year,False))
            categories.extend(make_metrics(union_events,stage,year,True))
            for rank,entry in per_rank.items():
                e=[(key,ret) for key,ret in entry.items()]
                rank_types.extend(make_metrics(e,stage,year,True,rank))
                rank_types.extend(make_metrics(e,stage,year,False,rank))
        multiplicity=Counter(mask.bit_count() for ret,mask in unique.values())
        # Payout distribution only among winning tickets, not purchase-odds ranges.
        for kind in sorted(JAPANESE):
            winners=[(k,r) for k,(r,mask) in unique.items() if k[2]==kind and r>0]
            grouped=defaultdict(list)
            for key,ret in winners:grouped[payout_band(ret)].append(ret)
            for lo,hi,name in BANDS:
                values=grouped[name]
                payouts.append(dict(stage=stage,ticket_type=kind,ticket_type_ja=JAPANESE[kind],
                    winning_payout_band=name,winning_ticket_count=len(values),
                    winning_payout_yen=sum(values)))
        for key,(ret,mask) in heapq.nlargest(30,
                     [(k,v) for k,v in unique.items() if v[0]>0],
                     key=lambda item:item[1][0]):
            year,rid,kind,sel=key
            top.append(dict(stage=stage,year=year,race_id=rid,
                            ticket_type=kind,ticket_type_ja=JAPANESE[kind],
                            horse_numbers=sel,refunded_per_100_yen=ret,
                            return_multiple=ret/100,ev_rank_contributors=",".join(
                                str(r) for r in range(1,7) if mask & (1<<(r-1)))))
        assert_ok(sum(v-1 for k,v in multiplicity.items() if k>1)==0
                  if stage in ("STABLE_MEDIUM","STABLE_STRICT") else True,
                  ("unexpected stable cross-rank overlap",stage))
        overview_rows=[x for x in overview if x["stage"]==stage and x["year"]=="2023-2025"]
        assert_ok(overview_rows[0]["tickets"]==EXPECTED[stage][0],"overview drift")
        print("TYPE_BREAKDOWN_STAGE",stage,overview_rows[0]["roi_pct"],
              "tickets",len(unique),"kinds",Counter(k[2] for k in unique),
              "ranks",rank_counts,flush=True)
    assert_ok(all_keys["STABLE_STRICT"] <= all_keys["STABLE_MEDIUM"] <= all_keys["ALL_YEAR_BLACK"] <= all_keys["BASE"],
              "not a proper nested subset of original")
    csv_out(out/"overview_by_year.csv",overview)
    csv_out(out/"ticket_type_by_year.csv",categories)
    csv_out(out/"rank_and_ticket_type_by_year.csv",rank_types)
    csv_out(out/"hit_payout_bands_by_type.csv",payouts)
    csv_out(out/"top_30_winning_tickets_each_stage.csv",top)
    big={
        "contract":"READ_ONLY_EXACT_SIX_RANK_TICKET_TYPE_AND_PAYOUT_BREAKDOWN_V1",
        "stages":list(STAGES),"original_source_run":38922914983,
        "stability_source_run":38927480138,
        "baseline_reconciled":True,"all_stage_bets_nested_exact_subsets":True,
        "source_frozen_selections_preserved":True,
        "years":list(YEARS),"jan_2023_evaluation_excluded":True,
        "no_new_strategy_selection":True,"no_2026":True,
        "payout_bins_are_winning_refund_multiples_not_prebet_odds_bins":True,
        "ticket_kinds":JAPANESE,
        "by_stage":{s:next(r for r in overview if r["stage"]==s and r["year"]=="2023-2025")
                    for s in STAGES}
    }
    (out/"summary.json").write_text(json.dumps(big,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("SIX_RANK_BET_TYPE_REPORT_COMPLETE",json.dumps(big["by_stage"],ensure_ascii=False),flush=True)

if __name__=="__main__":main()
