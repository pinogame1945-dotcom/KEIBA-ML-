#!/usr/bin/env python3
import argparse
import csv
import gzip
import itertools
import json
from collections import defaultdict
from pathlib import Path

YEARS=(2022,2023,2024,2025)
MAX_RANK=15

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot-root",required=True)
    p.add_argument("--router-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def find_one(root,patterns):
    root=Path(root)
    for pat in patterns:
        hits=sorted(root.glob(pat))
        if hits:return hits[0]
    raise FileNotFoundError((root,patterns))

def ordered_outcomes(groups,slots=3):
    pieces=[()]
    for rank in sorted(groups):
        if rank>slots: continue
        horses=list(groups[rank])
        occupied=min(len(horses),slots-rank+1)
        if occupied<=0: continue
        perms=list(itertools.permutations(horses,occupied))
        pieces=[a+b for a in pieces for b in perms]
    return sorted(set(x for x in pieces if len(x)==slots))

def read_truth(path):
    groups=defaultdict(lambda:defaultdict(list))
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            hid=str(row.get("horse_id") or "")
            target=row.get("target") or {}
            try: finish=int(float(target.get("finish_position")))
            except (TypeError,ValueError): continue
            if rid and hid and finish<=3:
                groups[rid][finish].append(hid)
    out={}
    for rid,g in groups.items():
        oc=ordered_outcomes(g,3)
        if oc: out[rid]=oc
    return out

def load_router(path):
    out={}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            if rid: out[rid]=row
    return out

def seven_order(router_row):
    experts=router_row.get("experts") or {}
    stats=defaultdict(lambda:{"support":0,"borda":0.0,"top1_votes":0,"best_rank":99,"rank_sum":0.0})
    for expert in experts.values():
        ids=[str(x) for x in (expert.get("top6_horse_ids") or []) if str(x)]
        for rank,hid in enumerate(ids,1):
            s=stats[hid]
            s["support"]+=1
            s["borda"]+=float(7-rank)
            s["top1_votes"]+=int(rank==1)
            s["best_rank"]=min(s["best_rank"],rank)
            s["rank_sum"]+=rank
    for s in stats.values():
        s["mean_rank"]=s["rank_sum"]/s["support"]
    return sorted(stats,key=lambda hid:(-stats[hid]["borda"],-stats[hid]["support"],-stats[hid]["top1_votes"],stats[hid]["best_rank"],stats[hid]["mean_rank"],hid))

def podium_set(outcomes):
    s=set()
    for o in outcomes: s.update(o)
    return s

def exact123(order,outcomes):
    if len(order)<3:return False
    return tuple(order[:3]) in set(outcomes)

def top3_box(order,outcomes):
    if len(order)<3:return False
    s=set(order[:3])
    return any(set(o)==s for o in outcomes)

def main():
    a=parse_args()
    out_dir=Path(a.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    overall=defaultdict(lambda:{"races_with_rank":0,"podium_hits":0})
    yearly={}
    total_races=0
    exact_hits=0
    box_hits=0

    for year in YEARS:
        snap=find_one(a.snapshot_root,[f"snapshot-{year}.jsonl.gz",f"snapshot-{year}.jsonl"])
        router=find_one(Path(a.router_root)/str(year),["router-7k.jsonl.gz","router-7k.jsonl"])
        truth=read_truth(snap)
        routers=load_router(router)
        if len(routers)!=3456: raise SystemExit(f"router count regression {year}: {len(routers)}")
        yr=defaultdict(lambda:{"races_with_rank":0,"podium_hits":0})
        yr_exact=yr_box=0
        for rid,row in routers.items():
            outcomes=truth[rid]
            pset=podium_set(outcomes)
            order=seven_order(row)
            total_races+=1
            yr_exact += int(exact123(order,outcomes))
            yr_box += int(top3_box(order,outcomes))
            exact_hits += int(exact123(order,outcomes))
            box_hits += int(top3_box(order,outcomes))
            for idx,hid in enumerate(order[:MAX_RANK],1):
                overall[idx]["races_with_rank"]+=1
                yr[idx]["races_with_rank"]+=1
                hit=int(hid in pset)
                overall[idx]["podium_hits"]+=hit
                yr[idx]["podium_hits"]+=hit
        yearly[str(year)]={
            "races":len(routers),
            "exact_123_hits":yr_exact,
            "exact_123_rate":yr_exact/len(routers),
            "top3_box_hits":yr_box,
            "top3_box_rate":yr_box/len(routers),
            "rank_podium_rate":{
                str(r):{
                    "races":yr[r]["races_with_rank"],
                    "hits":yr[r]["podium_hits"],
                    "rate":yr[r]["podium_hits"]/yr[r]["races_with_rank"] if yr[r]["races_with_rank"] else None
                } for r in sorted(yr)
            }
        }

    summary={
        "contract":"L15_CONSENSUS_RANK_PODIUM_V1",
        "years":list(YEARS),
        "locked_years":[2026],
        "ability_uses_odds":False,
        "overall":{
            "races":total_races,
            "exact_123_hits":exact_hits,
            "exact_123_rate":exact_hits/total_races,
            "top3_box_hits":box_hits,
            "top3_box_rate":box_hits/total_races,
            "rank_podium_rate":{
                str(r):{
                    "races":overall[r]["races_with_rank"],
                    "hits":overall[r]["podium_hits"],
                    "rate":overall[r]["podium_hits"]/overall[r]["races_with_rank"] if overall[r]["races_with_rank"] else None
                } for r in sorted(overall)
            }
        },
        "by_year":yearly,
        "notes":[
            "Rank is the frozen seven-king consensus order reconstructed with the L15_FIXED_V1 Borda/support rule.",
            "Podium hit means the horse at that consensus rank actually finished in an official top-3 position.",
            "Exact 1-2-3 is one straight trifecta ticket; Top3 box is the same three horses in any order.",
            "Dead heats are accepted through the same ordered-outcome logic used by the existing L1.5 role datasets."
        ]
    }
    (out_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with open(out_dir/"rank-podium.csv","w",newline="",encoding="utf-8") as fh:
        w=csv.writer(fh); w.writerow(["rank","races","podium_hits","podium_rate"])
        for r in sorted(overall):
            n=overall[r]["races_with_rank"]; h=overall[r]["podium_hits"]
            w.writerow([r,n,h,h/n if n else ""])
    print("L15_CONSENSUS_RANK_PODIUM_READY")
    print(json.dumps(summary["overall"],ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
