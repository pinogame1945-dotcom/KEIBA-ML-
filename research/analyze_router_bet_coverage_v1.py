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


def read_truth(path,wanted):
    races=defaultdict(dict)
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
            if finish in (1,2,3):
                races[rid][finish]=hid
    missing=[rid for rid in wanted if set(races[rid])!={1,2,3}]
    if missing:
        raise ValueError(f"{len(missing)} races missing complete 1st/2nd/3rd truth")
    return races


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
        actual=truth[rid]
        h1,h2,h3=actual[1],actual[2],actual[3]
        r1=by_horse.get(h1)
        r2=by_horse.get(h2)
        r3=by_horse.get(h3)
        if r1 is None or r2 is None or r3 is None:
            raise ValueError(f"{rid}: truth horse missing from expert rows")
        metrics["winner_rank_distribution"][str(r1)]+=1
        metrics["second_rank_distribution"][str(r2)]+=1
        metrics["third_rank_distribution"][str(r3)]+=1

        metrics["top1_hit"] += int(r1==1)
        metrics["exact_top2_order"] += int(r1==1 and r2==2)
        metrics["exact_top3_order"] += int(r1==1 and r2==2 and r3==3)

        for n in range(2,7):
            metrics["top2_set_in_topN"][str(n)] += int(r1<=n and r2<=n)
            metrics["first_fixed_second_in_topN"][str(n)] += int(r1==1 and r2<=n)
        for n in range(3,7):
            metrics["top3_set_in_topN"][str(n)] += int(r1<=n and r2<=n and r3<=n)
            metrics["first_fixed_top3_set_in_topN"][str(n)] += int(r1==1 and r2<=n and r3<=n)

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
