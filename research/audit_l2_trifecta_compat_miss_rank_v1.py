#!/usr/bin/env python3
import argparse,csv,json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_l2_trifecta_fullfield_ability_v2 import (
    YEARS, TRAIN_YEARS, load_l17, parse_paths, sample_year, train, iter_races, matrix_for
)

BANDS=[
    (1,5,"01-05"),
    (6,10,"06-10"),
    (11,20,"11-20"),
    (21,50,"21-50"),
    (51,100,"51-100"),
    (101,200,"101-200"),
    (201,10**9,"201+"),
]

EXPECTED_RACES=3456
EXPECTED_TOP5_HITS=78
EXPECTED_TOP5_RETURN=1662740.0

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def band_for(rank):
    for lo,hi,label in BANDS:
        if lo<=rank<=hi:
            return label
    raise RuntimeError(rank)

def percentile(xs,p):
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs,dtype=np.float64),p))

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if hasattr(rows,"to_csv"):
        rows.to_csv(path,index=False)
        return
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def main():
    a=parse_args()
    paths=parse_paths(a.l17_year)
    l17={y:load_l17(paths[y],y) for y in YEARS}

    # Exact COMPAT training recipe used for the 2025 fold in Ability V2.
    samples={y:sample_year(y,l17[y],a.backfill_root,"COMPAT") for y in TRAIN_YEARS}
    model,importance,train_rows,train_races=train([samples[y] for y in TRAIN_YEARS],111020)

    details=[]
    band_stats=defaultdict(lambda:{
        "races":0,"winner_payouts":[],"winner_rank_sum":0,"candidate_sum":0,
        "top5_hit":0,"top10_hit":0,"top20_hit":0,"top50_hit":0,
    })
    all_winner_ranks=[]
    top5_hits=0
    top5_return=0.0
    ok_races=0
    counters=defaultdict(int)

    for race in iter_races(2025,l17[2025],a.backfill_root,with_odds=False):
        if race["status"]!="ok":
            counters[race["status"]]+=1
            continue
        ok_races+=1
        idx=np.arange(len(race["combos"]),dtype=np.int64)
        x=matrix_for(race,idx)
        pred=np.asarray(model.predict(x),dtype=np.float64)
        order=np.argsort(-pred,kind="mergesort")
        rank_of=np.empty(len(order),dtype=np.int32)
        rank_of[order]=np.arange(1,len(order)+1,dtype=np.int32)

        winner_indices=[int(i) for i in race["positive"]]
        winner_ranks=[int(rank_of[i]) for i in winner_indices]
        best_rank=min(winner_ranks)
        best_winner_idx=winner_indices[winner_ranks.index(best_rank)]
        payout=float(race["returns"][best_winner_idx])
        band=band_for(best_rank)
        all_winner_ranks.append(best_rank)

        chosen5=np.asarray(order[:5],dtype=np.int64)
        hit5=bool(np.any(race["returns"][chosen5]>0))
        ret5=float(race["returns"][chosen5].sum())
        top5_hits+=int(hit5)
        top5_return+=ret5

        a1,b1,c1,nums=race["combos"][best_winner_idx]
        top_score=float(pred[order[0]])
        winner_score=float(pred[best_winner_idx])
        fifth_score=float(pred[order[min(4,len(order)-1)]])
        tenth_score=float(pred[order[min(9,len(order)-1)]])
        margin_top_minus_winner=top_score-winner_score
        margin_fifth_minus_winner=fifth_score-winner_score
        margin_tenth_minus_winner=tenth_score-winner_score

        st=band_stats[band]
        st["races"]+=1
        st["winner_payouts"].append(payout)
        st["winner_rank_sum"]+=best_rank
        st["candidate_sum"]+=len(order)
        st["top5_hit"]+=int(best_rank<=5)
        st["top10_hit"]+=int(best_rank<=10)
        st["top20_hit"]+=int(best_rank<=20)
        st["top50_hit"]+=int(best_rank<=50)

        details.append({
            "race_id":race["race_id"],
            "race_date":race["race_date"],
            "field_size":len(race["horses"]),
            "candidate_count":len(order),
            "winner_rank":best_rank,
            "band":band,
            "winner_ticket":"-".join(map(str,nums)),
            "winner_payout_yen_per100":payout,
            "winning_payout_combo_count":len(winner_indices),
            "top5_hit":int(hit5),
            "top5_return_yen":ret5,
            "model_top_score":top_score,
            "winner_score":winner_score,
            "fifth_score":fifth_score,
            "tenth_score":tenth_score,
            "top_minus_winner_score_gap":margin_top_minus_winner,
            "fifth_minus_winner_score_gap":margin_fifth_minus_winner,
            "tenth_minus_winner_score_gap":margin_tenth_minus_winner,
            "winner_s1_consensus_rank":int(a1["consensus_rank"]),
            "winner_s2_consensus_rank":int(b1["consensus_rank"]),
            "winner_s3_consensus_rank":int(c1["consensus_rank"]),
        })

    if ok_races!=EXPECTED_RACES:
        raise SystemExit(f"coverage drift: {ok_races} != {EXPECTED_RACES}")
    if top5_hits!=EXPECTED_TOP5_HITS:
        raise SystemExit(f"repro top5 hits drift: {top5_hits} != {EXPECTED_TOP5_HITS}")
    if abs(top5_return-EXPECTED_TOP5_RETURN)>1e-6:
        raise SystemExit(f"repro top5 return drift: {top5_return} != {EXPECTED_TOP5_RETURN}")

    band_rows=[]
    for _,_,label in BANDS:
        st=band_stats[label]
        n=st["races"]
        payouts=st["winner_payouts"]
        band_rows.append({
            "band":label,
            "races":n,
            "share_pct":100.0*n/ok_races,
            "cumulative_through_band_pct":None,
            "mean_winner_rank":st["winner_rank_sum"]/n if n else None,
            "mean_candidates":st["candidate_sum"]/n if n else None,
            "mean_winner_payout_yen_per100":float(np.mean(payouts)) if payouts else None,
            "median_winner_payout_yen_per100":float(np.median(payouts)) if payouts else None,
            "p75_winner_payout_yen_per100":percentile(payouts,75),
            "p90_winner_payout_yen_per100":percentile(payouts,90),
            "max_winner_payout_yen_per100":max(payouts) if payouts else None,
        })
    cum=0
    for row in band_rows:
        cum+=row["races"]
        row["cumulative_through_band_pct"]=100.0*cum/ok_races

    topn_reach=[]
    for n in (1,2,3,5,10,20,50,100,200,500,1000):
        caught=sum(1 for r in all_winner_ranks if r<=n)
        topn_reach.append({
            "top_n":n,
            "caught_races":caught,
            "caught_rate_pct":100.0*caught/ok_races,
            "increment_from_top5_races":caught-top5_hits if n>=5 else None,
        })

    # Focus on misses just outside Top5, without inventing a new betting policy.
    near_miss=[r for r in details if 6<=r["winner_rank"]<=20]
    near_summary={
        "races":len(near_miss),
        "share_of_all_races_pct":100.0*len(near_miss)/ok_races,
        "share_of_top5_misses_pct":100.0*len(near_miss)/(ok_races-top5_hits),
        "median_winner_rank":float(np.median([r["winner_rank"] for r in near_miss])) if near_miss else None,
        "median_payout_yen_per100":float(np.median([r["winner_payout_yen_per100"] for r in near_miss])) if near_miss else None,
        "median_fifth_minus_winner_score_gap":float(np.median([r["fifth_minus_winner_score_gap"] for r in near_miss])) if near_miss else None,
    }

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"winner-rank-bands.csv",band_rows)
    write_csv(out/"topn-reach.csv",topn_reach)
    write_csv(out/"race-detail.csv",details)
    write_csv(out/"feature-importance.csv",importance)

    summary={
        "contract":"L2_TRIFECTA_COMPAT_MISS_RANK_AUDIT_V1",
        "year":2025,
        "source_races":ok_races,
        "training_years":[2022,2023,2024],
        "model_recipe":"Ability V2 COMPAT 2025 fold exact reproduction",
        "training_rows":train_rows,
        "training_races":train_races,
        "reproduction_guard":{
            "expected_top5_hits":EXPECTED_TOP5_HITS,
            "actual_top5_hits":top5_hits,
            "expected_top5_return_yen":EXPECTED_TOP5_RETURN,
            "actual_top5_return_yen":top5_return,
            "passed":True,
        },
        "top5":{
            "hit_races":top5_hits,
            "miss_races":ok_races-top5_hits,
            "hit_rate_pct":100.0*top5_hits/ok_races,
            "roi_pct":100.0*top5_return/(ok_races*5*100.0),
        },
        "near_miss_6_to_20":near_summary,
        "winner_rank_median":float(np.median(all_winner_ranks)),
        "winner_rank_p75":percentile(all_winner_ranks,75),
        "winner_rank_p90":percentile(all_winner_ranks,90),
        "winner_rank_p95":percentile(all_winner_ranks,95),
        "winner_rank_max":max(all_winner_ranks),
        "bands":band_rows,
        "topn_reach":topn_reach,
        "counters":dict(counters),
        "new_betting_rule_created":False,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Trifecta COMPAT Miss-Rank Audit V1\n\n"
        "Diagnostic only. Reproduces the 2025 Ability V2 COMPAT model exactly, then records where the actual winning trifecta ranked among every STARTED full-field ordered ticket. "
        "No new betting rule, filter, gate, or policy is selected. Bands are 1-5, 6-10, 11-20, 21-50, 51-100, 101-200, and 201+. "
        "The run must reproduce 78 Top5 hits and JPY 1,662,740 return before publishing diagnostics.\n",
        encoding="utf-8"
    )
    print("L2_TRIFECTA_COMPAT_MISS_RANK_AUDIT_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
