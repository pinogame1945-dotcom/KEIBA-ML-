#!/usr/bin/env python3
import argparse, gzip, json, itertools, csv
from collections import defaultdict
from pathlib import Path

YEARS=(2022,2023,2024,2025)
TOPN=10

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

def read_truth_and_field(path):
    races=defaultdict(lambda: {"finish":defaultdict(list),"field":set()})
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            hid=str(row.get("horse_id") or "")
            if not rid or not hid: continue
            races[rid]["field"].add(hid)
            target=row.get("target") or {}
            try: fin=int(float(target.get("finish_position")))
            except (TypeError,ValueError): continue
            if fin<=3:races[rid]["finish"][fin].append(hid)
    out={}
    for rid,d in races.items():
        outcomes=[()]
        for rank in sorted(d["finish"]):
            if rank>3: continue
            horses=list(d["finish"][rank])
            occupied=min(len(horses),3-rank+1)
            if occupied<=0: continue
            perms=list(itertools.permutations(horses,occupied))
            outcomes=[a+b for a in outcomes for b in perms]
        outcomes=sorted(set(x for x in outcomes if len(x)==3))
        if outcomes:
            out[rid]={"outcomes":outcomes,"field_size":len(d["field"])}
    return out

def load_router(path):
    out={}
    with open_text(path) as fh:
        for line in fh:
            if line.strip():
                r=json.loads(line); out[str(r["race_id"])]=r
    return out

def seven_order(router_row):
    experts=router_row.get("experts") or {}
    stats=defaultdict(lambda:{"support":0,"borda":0.0,"top1_votes":0,"best_rank":99,"rank_sum":0.0})
    for expert in experts.values():
        ids=[str(x) for x in (expert.get("top6_horse_ids") or []) if str(x)]
        for rank,hid in enumerate(ids,1):
            s=stats[hid]; s["support"]+=1; s["borda"]+=7-rank; s["top1_votes"]+=int(rank==1); s["best_rank"]=min(s["best_rank"],rank); s["rank_sum"]+=rank
    for s in stats.values(): s["mean_rank"]=s["rank_sum"]/s["support"]
    return sorted(stats,key=lambda h:(-stats[h]["borda"],-stats[h]["support"],-stats[h]["top1_votes"],stats[h]["best_rank"],stats[h]["mean_rank"],h))

def miss_profile(top10,outcomes):
    # choose valid dead-heat outcome with minimum misses; for equal misses keep first deterministically
    best=None
    s=set(top10)
    for o in outcomes:
        flags=tuple(h not in s for h in o)
        cand=(sum(flags),flags,o)
        if best is None or cand<best: best=cand
    return best

def main():
    a=parse_args(); outdir=Path(a.out_dir); outdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    agg={"races":0,"captured":0,"missed":0,"miss1":0,"miss2":0,"miss3":0,
         "miss_pos1":0,"miss_pos2":0,"miss_pos3":0}
    by_field=defaultdict(lambda:{"races":0,"captured":0,"missed":0})
    by_year={}
    combos=defaultdict(int)

    for y in YEARS:
        snap=find_one(a.snapshot_root,[f"snapshot-{y}.jsonl.gz",f"snapshot-{y}.jsonl"])
        router=find_one(Path(a.router_root)/str(y),["router-7k.jsonl.gz","router-7k.jsonl"])
        truth=read_truth_and_field(snap); routers=load_router(router)
        ya={"races":0,"captured":0,"missed":0,"miss1":0,"miss2":0,"miss3":0,"miss_pos1":0,"miss_pos2":0,"miss_pos3":0}
        for rid,rr in routers.items():
            t=truth[rid]; order=seven_order(rr); top10=order[:TOPN]
            nmiss,flags,outcome=miss_profile(top10,t["outcomes"])
            fs=t["field_size"]
            agg["races"]+=1; ya["races"]+=1; by_field[fs]["races"]+=1
            if nmiss==0:
                agg["captured"]+=1; ya["captured"]+=1; by_field[fs]["captured"]+=1
            else:
                agg["missed"]+=1; ya["missed"]+=1; by_field[fs]["missed"]+=1
                agg[f"miss{nmiss}"]+=1; ya[f"miss{nmiss}"]+=1
                key=[]
                for i,f in enumerate(flags,1):
                    if f:
                        agg[f"miss_pos{i}"]+=1; ya[f"miss_pos{i}"]+=1; key.append(str(i))
                combos["+".join(key)]+=1
            rows.append({"year":y,"race_id":rid,"field_size":fs,"candidate_count":len(order),"top10_size":len(top10),
                         "miss_count":nmiss,"miss_1st":int(flags[0]),"miss_2nd":int(flags[1]),"miss_3rd":int(flags[2])})
        by_year[str(y)]=ya

    for d in [agg,*by_year.values()]:
        d["capture_rate"]=d["captured"]/d["races"]
        d["miss_rate"]=d["missed"]/d["races"]
        d["miss1_rate_all"]=d["miss1"]/d["races"]
        d["miss2_rate_all"]=d["miss2"]/d["races"]
        d["miss3_rate_all"]=d["miss3"]/d["races"]
        d["miss_pos1_rate_all"]=d["miss_pos1"]/d["races"]
        d["miss_pos2_rate_all"]=d["miss_pos2"]/d["races"]
        d["miss_pos3_rate_all"]=d["miss_pos3"]/d["races"]

    field_summary={}
    for fs,d in sorted(by_field.items()):
        field_summary[str(fs)]={**d,"capture_rate":d["captured"]/d["races"],"miss_rate":d["missed"]/d["races"]}

    summary={
      "contract":"L15_TOP10_MISS_DIAG_V1","years":list(YEARS),"locked_years":[2026],"ability_uses_odds":False,
      "topn":TOPN,"overall":agg,"miss_position_combinations":dict(sorted(combos.items())),
      "by_year":by_year,"by_field_size":field_summary,
      "note":"For dead heats, uses the valid ordered 1-2-3 outcome requiring the fewest Top10 misses, so miss counts are conservative."
    }
    (outdir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with open(outdir/"race-misses.csv","w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__": main()
