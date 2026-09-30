#!/usr/bin/env python3
import csv, json
from collections import defaultdict
from pathlib import Path

BASE=Path("research-results/l2-market-gap-seat-v1/run-36706811732")
YEARS=(2022,2023,2024,2025)
TEST_YEARS=(2023,2024,2025)
SEATS=(1,2,3)
DIRECTION_GROUPS=("KING_HIGHER","SAME","MARKET_HIGHER")
BUCKET_GROUPS=("NEG_4PLUS","NEG_2_3","NEG_1","ZERO","POS_1","POS_2_3","POS_4PLUS")


def read_csv(path):
    with open(path,newline="",encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(x):
    return float(x) if x not in ("",None) else 0.0


def i(x):
    return int(float(x)) if x not in ("",None) else 0


def weighted_probs(horse_rows, train_years, rank, group):
    n=0
    counts={1:0.0,2:0.0,3:0.0}
    for r in horse_rows:
        if r["year"] not in {str(y) for y in train_years}:
            continue
        if i(r["focal_rank"])!=rank or r["market_group"]!=group:
            continue
        rn=i(r["horses"])
        n+=rn
        counts[1]+=rn*f(r["finish1_rate_pct"])/100.0
        counts[2]+=rn*f(r["finish2_rate_pct"])/100.0
        counts[3]+=rn*f(r["finish3_rate_pct"])/100.0
    if n==0:
        return None
    probs={s:counts[s]/n for s in SEATS}
    return n,probs


def choose_seat(horse_rows, train_years, rank, group):
    got=weighted_probs(horse_rows,train_years,rank,group)
    fallback=False
    used_group=group
    if got is None:
        got=weighted_probs(horse_rows,train_years,rank,"ALL")
        fallback=True
        used_group="ALL"
    if got is None:
        raise RuntimeError(f"no training outcomes rank={rank} group={group}")
    n,probs=got
    # deterministic: highest empirical exact-finish probability; ties favor lower seat number.
    seat=max(SEATS,key=lambda s:(probs[s],-s))
    return {
        "seat":seat,
        "train_horses":n,
        "p1":probs[1],
        "p2":probs[2],
        "p3":probs[3],
        "fallback":fallback,
        "used_group":used_group,
    }


def index_metrics(rows):
    out={}
    for r in rows:
        key=(i(r["year"]),i(r["focal_rank"]),i(r["pool_k"]),i(r["seat"]),r["market_group"])
        out[key]=r
    return out


def aggregate(rows):
    races=sum(i(r["races"]) for r in rows)
    tickets=sum(i(r["tickets"]) for r in rows)
    hit=sum(i(r["hit_races"]) for r in rows)
    stake=sum(f(r["stake_yen"]) for r in rows)
    ret=sum(f(r["return_yen"]) for r in rows)
    return {
        "races":races,
        "tickets":tickets,
        "hit_races":hit,
        "hit_rate_pct":100.0*hit/races if races else None,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "avg_tickets_per_race":tickets/races if races else None,
    }


def main():
    horse=read_csv(BASE/"horse-seat-outcomes.csv")
    metrics=read_csv(BASE/"seat-gap-yearly.csv")
    idx=index_metrics(metrics)

    variants={
        "RANK_ONLY":("ALL",),
        "DIRECTION":DIRECTION_GROUPS,
        "BUCKET":BUCKET_GROUPS,
    }

    choice_rows=[]
    eval_rows=[]

    for test_year in TEST_YEARS:
        train_years=tuple(y for y in YEARS if y<test_year)
        for variant,groups in variants.items():
            for rank in (1,2,3):
                rank_choice={}
                for group in groups:
                    # Only choose groups that occur in the test year for this rank.
                    occurs=any(
                        r["year"]==str(test_year) and i(r["focal_rank"])==rank and r["market_group"]==group and i(r["horses"])>0
                        for r in horse
                    )
                    if not occurs:
                        continue
                    c=choose_seat(horse,train_years,rank,group)
                    rank_choice[group]=c
                    choice_rows.append({
                        "test_year":test_year,
                        "train_years":"-".join(map(str,train_years)),
                        "variant":variant,
                        "focal_rank":rank,
                        "market_group":group,
                        "selected_seat":c["seat"],
                        "train_group_used":c["used_group"],
                        "fallback_to_rank_only":c["fallback"],
                        "train_horses":c["train_horses"],
                        "p_finish1_pct":100*c["p1"],
                        "p_finish2_pct":100*c["p2"],
                        "p_finish3_pct":100*c["p3"],
                    })

                for pool in (3,6):
                    selected_metric_rows=[]
                    for group,c in rank_choice.items():
                        key=(test_year,rank,pool,c["seat"],group)
                        r=idx.get(key)
                        if r is None:
                            raise RuntimeError(f"missing metric row {key}")
                        selected_metric_rows.append(r)
                    agg=aggregate(selected_metric_rows)

                    # Comparator: best fixed seat is NOT used to choose policy; report all three fixed seats.
                    fixed={}
                    for seat in SEATS:
                        rs=[]
                        for group in rank_choice:
                            rr=idx.get((test_year,rank,pool,seat,group))
                            if rr is None:
                                raise RuntimeError(f"missing fixed metric {(test_year,rank,pool,seat,group)}")
                            rs.append(rr)
                        fixed[seat]=aggregate(rs)

                    eval_rows.append({
                        "test_year":test_year,
                        "train_years":"-".join(map(str,train_years)),
                        "variant":variant,
                        "focal_rank":rank,
                        "pool_k":pool,
                        **agg,
                        "fixed_seat1_roi_pct":fixed[1]["roi_pct"],
                        "fixed_seat2_roi_pct":fixed[2]["roi_pct"],
                        "fixed_seat3_roi_pct":fixed[3]["roi_pct"],
                    })

    # Combine strictly out-of-sample 2023-2025.
    combined=[]
    for variant in variants:
        for rank in (1,2,3):
            for pool in (3,6):
                sub=[r for r in eval_rows if r["variant"]==variant and r["focal_rank"]==rank and r["pool_k"]==pool]
                races=sum(r["races"] for r in sub)
                tickets=sum(r["tickets"] for r in sub)
                hit=sum(r["hit_races"] for r in sub)
                stake=sum(r["stake_yen"] for r in sub)
                ret=sum(r["return_yen"] for r in sub)
                combined.append({
                    "variant":variant,
                    "focal_rank":rank,
                    "pool_k":pool,
                    "test_years":"2023-2025",
                    "races":races,
                    "tickets":tickets,
                    "avg_tickets_per_race":tickets/races if races else None,
                    "hit_races":hit,
                    "hit_rate_pct":100.0*hit/races if races else None,
                    "stake_yen":stake,
                    "return_yen":ret,
                    "profit_yen":ret-stake,
                    "roi_pct":100.0*ret/stake if stake else None,
                })
    combined.sort(key=lambda r:r["roi_pct"],reverse=True)

    out=Path("research-results/l2-seat-by-finishprob-wf-v1")
    out.mkdir(parents=True,exist_ok=True)

    def write(name,rows):
        with open(out/name,"w",newline="",encoding="utf-8") as fh:
            w=csv.DictWriter(fh,fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    write("seat-choices.csv",choice_rows)
    write("walkforward-yearly.csv",eval_rows)
    write("walkforward-combined.csv",combined)

    summary={
        "contract":"L2_SEAT_BY_FINISHPROB_WF_V1",
        "source_run":36706811732,
        "idea":"Choose trifecta seat only from prior-year exact finish probabilities of the fixed L1.7 focal horse; never choose seat by ROI.",
        "test_years":[2023,2024,2025],
        "variants":{
            "RANK_ONLY":"rank only",
            "DIRECTION":"rank + KING_HIGHER/SAME/MARKET_HIGHER",
            "BUCKET":"rank + fixed market-gap bucket"
        },
        "pools":[3,6],
        "selection_target":"argmax(P(finish=1),P(finish=2),P(finish=3))",
        "selection_uses_roi":False,
        "selection_uses_payout":False,
        "selection_uses_test_year":False,
        "rerank_l1":False,
        "2026_locked":True,
        "best_oos_combined":combined[0],
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print("TOP")
    for r in combined[:12]:
        print(r)


if __name__=="__main__":
    main()
