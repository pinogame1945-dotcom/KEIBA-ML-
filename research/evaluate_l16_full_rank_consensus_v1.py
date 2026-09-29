#!/usr/bin/env python3
import argparse,gzip,itertools,json
from collections import defaultdict
from pathlib import Path

TOP_NS=(3,4,5,6,7,8,9,10,11,12)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--consensus",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--year",required=True,type=int)
    p.add_argument("--output",required=True)
    return p.parse_args()

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def truth(path):
    groups=defaultdict(lambda:defaultdict(list))
    fields=defaultdict(set)
    with open_text(path) as f:
        for line in f:
            if not line.strip():continue
            x=json.loads(line); rid=str(x.get("race_id") or ""); hid=str(x.get("horse_id") or "")
            if not rid or not hid:continue
            fields[rid].add(hid)
            try: pos=int(float((x.get("target") or {}).get("finish_position")))
            except (TypeError,ValueError):continue
            if pos<=3:groups[rid][pos].append(hid)
    out={}
    for rid,g in groups.items():
        pieces=[()]
        for rank in sorted(g):
            if rank>3:continue
            hs=list(g[rank]); occupied=min(len(hs),3-rank+1)
            perms=list(itertools.permutations(hs,occupied))
            pieces=[a+b for a in pieces for b in perms]
        outcomes=sorted(set(x for x in pieces if len(x)==3))
        if outcomes:out[rid]={"outcomes":outcomes,"field_size":len(fields[rid])}
    return out

def captured(order,outcomes,n):
    s=set(order[:n])
    return any(set(o).issubset(s) for o in outcomes)

def main():
    a=parse_args()
    if a.year==2026: raise ValueError("2026 is locked")
    t=truth(a.snapshot)
    total=0; hits={n:0 for n in TOP_NS}; exact=0
    by_field=defaultdict(lambda:{"races":0,**{f"top{n}_hits":0 for n in TOP_NS}})
    with open_text(a.consensus) as f:
        for line in f:
            if not line.strip():continue
            x=json.loads(line)
            if x.get("contract")!="L16_FULL_RANK_CONSENSUS_V1":raise ValueError("bad L1.6 contract")
            rid=str(x["race_id"]); rows=sorted(x["horses"],key=lambda z:int(z["consensus_rank"]))
            order=[str(z["horse_id"]) for z in rows]
            if len(order)!=int(x["field_size"]):raise ValueError(f"{rid}: field mismatch")
            if rid not in t:raise ValueError(f"truth missing {rid}")
            outcomes=t[rid]["outcomes"]; fs=t[rid]["field_size"]
            total+=1; by_field[fs]["races"]+=1
            for n in TOP_NS:
                h=captured(order,outcomes,n)
                hits[n]+=int(h); by_field[fs][f"top{n}_hits"]+=int(h)
            exact+=int(any(tuple(order[:3])==o for o in outcomes))
    if total!=3456:raise ValueError(f"race count {total} != 3456")
    summary={
      "contract":"L16_FULL_RANK_CONSENSUS_EVAL_V1","year":a.year,"locked_years":[2026],
      "races":total,
      "topn":{str(n):{"hits":hits[n],"rate":hits[n]/total} for n in TOP_NS},
      "exact_consensus_1_2_3":{"hits":exact,"rate":exact/total},
      "by_field_size":{}
    }
    for fs,d in sorted(by_field.items()):
        row={"races":d["races"]}
        for n in TOP_NS:
            row[f"top{n}_hits"]=d[f"top{n}_hits"]
            row[f"top{n}_rate"]=d[f"top{n}_hits"]/d["races"]
        summary["by_field_size"][str(fs)]=row
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":main()
