#!/usr/bin/env python3
import argparse,csv,itertools,json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

from run_l2_trifecta_fullfield_ability_v2 import (
    YEARS, TRAIN_YEARS, load_l17, parse_paths, sample_year, train, iter_races, matrix_for
)

EXPECTED_RACES=3456
EXPECTED_TOP5_HITS=78
DEEP_THRESHOLD=200

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if hasattr(rows,"to_csv"):
        rows.to_csv(path,index=False); return
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def rank_band(rank):
    if rank<=5:return "01-05"
    if rank<=20:return "06-20"
    if rank<=50:return "21-50"
    if rank<=100:return "51-100"
    if rank<=200:return "101-200"
    return "201+"

def main():
    a=parse_args()
    paths=parse_paths(a.l17_year)
    l17={y:load_l17(paths[y],y) for y in YEARS}

    # Exact 2025 COMPAT fold recipe.
    samples={y:sample_year(y,l17[y],a.backfill_root,"COMPAT") for y in TRAIN_YEARS}
    model,importance,train_rows,train_races=train([samples[y] for y in TRAIN_YEARS],111020)

    total=0; top5_hits=0; deep=0
    seat_out6=Counter(); seat_out10=Counter()
    outside6_count=Counter(); outside10_count=Counter()
    exact_band=Counter(); bestperm_band=Counter()
    ordering_rescue=Counter()
    pattern_counter=Counter()
    payout_by_out6=defaultdict(list)
    payout_by_bestband=defaultdict(list)
    deep_gaps=[]; deep_exact=[]; deep_best=[]
    deep_rows=[]

    for race in iter_races(2025,l17[2025],a.backfill_root,with_odds=False):
        if race["status"]!="ok":
            raise SystemExit(f"unexpected coverage status {race['status']} race={race.get('race_id')}")
        total+=1
        idx=np.arange(len(race["combos"]),dtype=np.int64)
        x=matrix_for(race,idx)
        pred=np.asarray(model.predict(x),dtype=np.float64)
        order=np.argsort(-pred,kind="mergesort")
        rank_of=np.empty(len(order),dtype=np.int32)
        rank_of[order]=np.arange(1,len(order)+1,dtype=np.int32)
        number_to_idx={combo[3]:i for i,combo in enumerate(race["combos"])}

        winner_indices=[int(i) for i in race["positive"]]
        winner_ranks=[int(rank_of[i]) for i in winner_indices]
        exact_rank=min(winner_ranks)
        chosen_winner_idx=winner_indices[winner_ranks.index(exact_rank)]
        if exact_rank<=5: top5_hits+=1
        exact_band[rank_band(exact_rank)]+=1

        a1,b1,c1,nums=race["combos"][chosen_winner_idx]
        trio=tuple(nums)
        trio_perms=list(itertools.permutations(trio,3))
        perm_ranks=[]
        for p in trio_perms:
            pi=number_to_idx.get(tuple(p))
            if pi is None:
                raise RuntimeError(f"missing permutation race={race['race_id']} perm={p}")
            perm_ranks.append((int(rank_of[pi]),tuple(p),int(pi)))
        perm_ranks.sort(key=lambda z:z[0])
        best_rank,best_perm,best_idx=perm_ranks[0]
        bestperm_band[rank_band(best_rank)]+=1

        if exact_rank>DEEP_THRESHOLD:
            deep+=1
            r1,r2,r3=(int(a1["consensus_rank"]),int(b1["consensus_rank"]),int(c1["consensus_rank"]))
            ranks=(r1,r2,r3)
            n_out6=sum(r>6 for r in ranks)
            n_out10=sum(r>10 for r in ranks)
            outside6_count[n_out6]+=1
            outside10_count[n_out10]+=1
            if r1>6: seat_out6["1st"]+=1
            if r2>6: seat_out6["2nd"]+=1
            if r3>6: seat_out6["3rd"]+=1
            if r1>10: seat_out10["1st"]+=1
            if r2>10: seat_out10["2nd"]+=1
            if r3>10: seat_out10["3rd"]+=1
            pattern_counter[(min(r1,99),min(r2,99),min(r3,99))]+=1

            payout=float(race["returns"][chosen_winner_idx])
            payout_by_out6[n_out6].append(payout)
            bb=rank_band(best_rank)
            payout_by_bestband[bb].append(payout)
            deep_exact.append(exact_rank); deep_best.append(best_rank); deep_gaps.append(exact_rank-best_rank)

            if best_rank<=5: ordering_rescue["bestperm_01_05"]+=1
            if best_rank<=20: ordering_rescue["bestperm_01_20"]+=1
            if best_rank<=50: ordering_rescue["bestperm_01_50"]+=1
            if best_rank<=100: ordering_rescue["bestperm_01_100"]+=1
            if best_rank<=200: ordering_rescue["bestperm_01_200"]+=1
            if best_rank>200: ordering_rescue["bestperm_still_201plus"]+=1

            deep_rows.append({
                "race_id":race["race_id"],"race_date":race["race_date"],
                "field_size":len(race["horses"]),"candidate_count":len(order),
                "exact_winner_rank":exact_rank,"best_permutation_rank":best_rank,
                "exact_minus_best_rank_gap":exact_rank-best_rank,
                "winner_ticket":"-".join(map(str,trio)),
                "best_permutation":"-".join(map(str,best_perm)),
                "winner_payout_yen_per100":payout,
                "first_consensus_rank":r1,"second_consensus_rank":r2,"third_consensus_rank":r3,
                "horses_outside_top6":n_out6,"horses_outside_top10":n_out10,
            })

    if total!=EXPECTED_RACES:
        raise SystemExit(f"coverage drift: {total} != {EXPECTED_RACES}")
    if top5_hits!=EXPECTED_TOP5_HITS:
        raise SystemExit(f"repro top5 hits drift: {top5_hits} != {EXPECTED_TOP5_HITS}")

    def pct(n,d=deep): return 100.0*n/d if d else None
    def stats(xs):
        return {
            "races":len(xs),
            "mean_payout_yen_per100":float(np.mean(xs)) if xs else None,
            "median_payout_yen_per100":float(np.median(xs)) if xs else None,
            "p90_payout_yen_per100":float(np.percentile(xs,90)) if xs else None,
        }

    seat_rows=[]
    for seat in ("1st","2nd","3rd"):
        seat_rows.append({
            "seat":seat,
            "outside_top6_races":seat_out6[seat],
            "outside_top6_pct":pct(seat_out6[seat]),
            "outside_top10_races":seat_out10[seat],
            "outside_top10_pct":pct(seat_out10[seat]),
        })

    out6_rows=[]
    for n in range(4):
        s=stats(payout_by_out6[n])
        out6_rows.append({
            "horses_outside_top6":n,
            "races":outside6_count[n],
            "share_of_deep_misses_pct":pct(outside6_count[n]),
            **{k:v for k,v in s.items() if k!="races"},
        })

    out10_rows=[]
    for n in range(4):
        out10_rows.append({
            "horses_outside_top10":n,
            "races":outside10_count[n],
            "share_of_deep_misses_pct":pct(outside10_count[n]),
        })

    bestband_rows=[]
    for label in ("01-05","06-20","21-50","51-100","101-200","201+"):
        s=stats(payout_by_bestband[label])
        bestband_rows.append({
            "best_permutation_band":label,
            "races":len(payout_by_bestband[label]),
            "share_of_deep_misses_pct":pct(len(payout_by_bestband[label])),
            **{k:v for k,v in s.items() if k!="races"},
        })

    pattern_rows=[]
    for ranks,n in pattern_counter.most_common(30):
        pattern_rows.append({
            "first_consensus_rank":ranks[0],
            "second_consensus_rank":ranks[1],
            "third_consensus_rank":ranks[2],
            "races":n,
            "share_of_deep_misses_pct":pct(n),
        })

    summary={
        "contract":"L2_TRIFECTA_COMPAT_DEEPMISS_AUDIT_V1",
        "year":2025,
        "source_races":total,
        "top5_hits":top5_hits,
        "deep_miss_definition":"exact winning trifecta COMPAT rank > 200",
        "deep_miss_races":deep,
        "deep_miss_share_pct":100.0*deep/total,
        "median_exact_rank":float(np.median(deep_exact)),
        "median_best_permutation_rank":float(np.median(deep_best)),
        "median_exact_minus_best_rank_gap":float(np.median(deep_gaps)),
        "ordering_rescue":{
            k:{"races":v,"share_of_deep_misses_pct":pct(v)}
            for k,v in ordering_rescue.items()
        },
        "seat_outside_top6":seat_rows,
        "outside_top6_count":out6_rows,
        "outside_top10_count":out10_rows,
        "best_permutation_bands":bestband_rows,
        "interpretation_contract":{
            "ordering_problem":"If the best permutation of the actual podium trio ranks much higher than the exact winning order.",
            "horse_set_problem":"If even the best permutation of the actual podium trio remains far down the ranking.",
            "no_new_betting_rule":True
        },
        "training_rows":train_rows,
        "training_races":train_races,
        "2026_locked":True,
    }

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"seat-outside-summary.csv",seat_rows)
    write_csv(out/"outside-top6-count.csv",out6_rows)
    write_csv(out/"outside-top10-count.csv",out10_rows)
    write_csv(out/"best-permutation-bands.csv",bestband_rows)
    write_csv(out/"top-consensus-rank-patterns.csv",pattern_rows)
    write_csv(out/"feature-importance.csv",importance)
    # Compact sample of the most severe cases, not the full matrix.
    severe=sorted(deep_rows,key=lambda r:(-r["exact_winner_rank"],-r["winner_payout_yen_per100"]))[:200]
    write_csv(out/"deepmiss-severe-sample.csv",severe)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Trifecta COMPAT Deep-Miss Audit V1\n\n"
        "Diagnostic only. Reproduces the frozen 2025 COMPAT model and inspects exact winning trifectas ranked below 200. "
        "For each deep miss, it checks the consensus ranks of the actual 1st/2nd/3rd horses and the best model rank among all six permutations of that same podium trio. "
        "This separates horse-set blindness from finishing-order/seat assignment error. No new betting rule or filter is selected.\n",
        encoding="utf-8"
    )
    print("L2_TRIFECTA_COMPAT_DEEPMISS_AUDIT_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
