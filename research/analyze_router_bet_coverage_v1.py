#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

CONTRACT="ROUTER_BET_COVERAGE_V1"


def parse_args():
    p=argparse.ArgumentParser(description="Analyze horse-set and ordered finish coverage for multiple L1 experts.")
    p.add_argument("--snapshot",required=True)
    p.add_argument("--expert",action="append",required=True,help="name=path")
    p.add_argument("--output",required=True)
    return p.parse_args()


def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")


def read_expert(path):
    races=defaultdict(list)
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            races[str(row["race_id"])].append(row)
    for rid,rows in races.items():
        rows.sort(key=lambda r:int(r["predicted_rank"]))
        ranks=[int(r["predicted_rank"]) for r in rows]
        if ranks != list(range(1,len(rows)+1)):
            raise ValueError(f"{rid}: ranks are not contiguous")
    return races


def ordered_outcomes(groups, slots):
    """Generate valid ordered podium outcomes, including dead heats."""
    import itertools
    groups={int(k):list(v) for k,v in groups.items()}
    pieces=[()]
    for rank in sorted(groups):
        if rank > slots:
            continue
        horses=groups[rank]
        occupied=min(len(horses), slots-rank+1)
        if occupied <= 0:
            continue
        choices=list(itertools.permutations(horses, occupied))
        pieces=[a+b for a in pieces for b in choices]
    return sorted(set(x for x in pieces if len(x)==slots))


def read_truth(path,wanted):
    groups=defaultdict(lambda: defaultdict(list))
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            if rid not in wanted: continue
            hid=str(row.get("horse_id") or "")
            target=row.get("target") or {}
            finish=target.get("finish_position")
            try:
                finish=int(float(finish))
            except (TypeError,ValueError):
                continue
            if finish <= 3:
                groups[rid][finish].append(hid)

    truth={}
    bad=[]
    dead_heat=0
    for rid in wanted:
        g=groups[rid]
        pair=ordered_outcomes(g,2)
        trio=ordered_outcomes(g,3)
        if not pair or not trio:
            bad.append({"race_id":rid,"groups":dict(g)})
            continue
        if any(len(v)>1 for v in g.values()):
            dead_heat+=1
        truth[rid]={"pair_outcomes":pair,"trio_outcomes":trio,"groups":dict(g)}
    if bad:
        raise ValueError(f"{len(bad)} races cannot form podium outcomes: {bad[:10]}")
    print("PODIUM_TRUTH_READY "+json.dumps({"races":len(truth),"dead_heat_races":dead_heat},ensure_ascii=False,separators=(",",":")))
    return truth


def rate(hit,total):
    return hit/total if total else None


def analyze_expert(races,truth):
    total=len(races)
    metrics={
        "top1_hit":0,
        "exact_top2_order":0,
        "exact_top3_order":0,
        "top2_set_in_topN":{str(n):0 for n in range(2,7)},
        "top3_set_in_topN":{str(n):0 for n in range(3,7)},
        "first_fixed_second_in_topN":{str(n):0 for n in range(2,7)},
        "first_fixed_top3_set_in_topN":{str(n):0 for n in range(3,7)},
        "winner_rank_distribution":defaultdict(int),
        "second_rank_distribution":defaultdict(int),
        "third_rank_distribution":defaultdict(int),
    }

    for rid,rows in races.items():
        by_horse={str(r["horse_id"]):int(r["predicted_rank"]) for r in rows}
        predicted=[str(r["horse_id"]) for r in rows]
        pair_outcomes=truth[rid]["pair_outcomes"]
        trio_outcomes=truth[rid]["trio_outcomes"]

        # A dead heat may make multiple ordered outcomes valid.
        winner_horses={x[0] for x in pair_outcomes}
        second_slot_horses={x[1] for x in pair_outcomes}
        third_slot_horses={x[2] for x in trio_outcomes}

        best_winner_rank=min(by_horse[h] for h in winner_horses)
        best_second_rank=min(by_horse[h] for h in second_slot_horses)
        best_third_rank=min(by_horse[h] for h in third_slot_horses)
        metrics["winner_rank_distribution"][str(best_winner_rank)]+=1
        metrics["second_rank_distribution"][str(best_second_rank)]+=1
        metrics["third_rank_distribution"][str(best_third_rank)]+=1

        metrics["top1_hit"] += int(predicted[0] in winner_horses)
        metrics["exact_top2_order"] += int(tuple(predicted[:2]) in pair_outcomes)
        metrics["exact_top3_order"] += int(tuple(predicted[:3]) in trio_outcomes)

        for n in range(2,7):
            topn=set(predicted[:n])
            metrics["top2_set_in_topN"][str(n)] += int(
                any(set(outcome).issubset(topn) for outcome in pair_outcomes)
            )
            metrics["first_fixed_second_in_topN"][str(n)] += int(
                any(predicted[0]==outcome[0] and outcome[1] in topn for outcome in pair_outcomes)
            )
        for n in range(3,7):
            topn=set(predicted[:n])
            metrics["top3_set_in_topN"][str(n)] += int(
                any(set(outcome).issubset(topn) for outcome in trio_outcomes)
            )
            metrics["first_fixed_top3_set_in_topN"][str(n)] += int(
                any(
                    predicted[0]==outcome[0]
                    and outcome[1] in topn
                    and outcome[2] in topn
                    for outcome in trio_outcomes
                )
            )

    return {
        "races":total,
        "top1_hit":{"hits":metrics["top1_hit"],"rate":rate(metrics["top1_hit"],total)},
        "ordered":{
            "exact_top2_order":{"hits":metrics["exact_top2_order"],"rate":rate(metrics["exact_top2_order"],total)},
            "exact_top3_order":{"hits":metrics["exact_top3_order"],"rate":rate(metrics["exact_top3_order"],total)},
        },
        "pair_capture":{
            n:{"hits":v,"rate":rate(v,total)}
            for n,v in metrics["top2_set_in_topN"].items()
        },
        "trio_capture":{
            n:{"hits":v,"rate":rate(v,total)}
            for n,v in metrics["top3_set_in_topN"].items()
        },
        "first_fixed_pair_capture":{
            n:{"hits":v,"rate":rate(v,total)}
            for n,v in metrics["first_fixed_second_in_topN"].items()
        },
        "first_fixed_trio_capture":{
            n:{"hits":v,"rate":rate(v,total)}
            for n,v in metrics["first_fixed_top3_set_in_topN"].items()
        },
        "finish_rank_distributions":{
            "winner":dict(sorted(metrics["winner_rank_distribution"].items(),key=lambda x:int(x[0]))),
            "second":dict(sorted(metrics["second_rank_distribution"].items(),key=lambda x:int(x[0]))),
            "third":dict(sorted(metrics["third_rank_distribution"].items(),key=lambda x:int(x[0]))),
        },
    }


def main():
    a=parse_args()
    experts={}
    for spec in a.expert:
        if "=" not in spec:
            raise ValueError("--expert must be name=path")
        name,path=spec.split("=",1)
        experts[name]=read_expert(path)
    if len(experts)<2:
        raise ValueError("need at least two experts")
    race_sets=[set(v) for v in experts.values()]
    if any(s!=race_sets[0] for s in race_sets[1:]):
        raise ValueError("expert race coverage differs")
    wanted=race_sets[0]
    truth=read_truth(a.snapshot,wanted)

    out={
        "contract":CONTRACT,
        "races":len(wanted),
        "experts":{name:analyze_expert(races,truth) for name,races in experts.items()},
        "definitions":{
            "pair_capture":"Actual 1st and 2nd are both inside expert TopN; order ignored.",
            "trio_capture":"Actual 1st, 2nd and 3rd are all inside expert TopN; order ignored.",
            "first_fixed_pair_capture":"Expert rank1 is actual winner and actual 2nd is inside expert TopN.",
            "first_fixed_trio_capture":"Expert rank1 is actual winner and actual 2nd/3rd are both inside expert TopN.",
            "exact_top2_order":"Expert ranks 1,2 exactly equal actual 1st,2nd.",
            "exact_top3_order":"Expert ranks 1,2,3 exactly equal actual 1st,2nd,3rd."
        },
        "notes":[
            "No odds are used.",
            "These are candidate-set coverage diagnostics, not ROI or ticket profitability.",
            "2020 is a pretest year in this workflow; 2026 remains untouched."
        ]
    }
    p=Path(a.output)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ROUTER_BET_COVERAGE_V1_OK")
    print(json.dumps(out,ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
