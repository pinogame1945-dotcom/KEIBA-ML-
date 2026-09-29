#!/usr/bin/env python3
import argparse,csv,gzip,itertools,json
from collections import defaultdict
from pathlib import Path

CANDS=["outsider_daytrend","outsider_raceshape","outsider_gatecourse","outsider_field","outsider_jockey"]

def ap():
    p=argparse.ArgumentParser()
    p.add_argument("--consensus",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--outsider-csv",required=True)
    p.add_argument("--year",type=int,required=True)
    p.add_argument("--output",required=True)
    return p.parse_args()

def open_text(p):
    return gzip.open(p,"rt",encoding="utf-8") if str(p).endswith(".gz") else open(p,"rt",encoding="utf-8")

def truth(path):
    g=defaultdict(lambda:defaultdict(list))
    with open_text(path) as f:
        for line in f:
            if not line.strip(): continue
            x=json.loads(line); rid=str(x.get("race_id") or ""); hid=str(x.get("horse_id") or "")
            if not rid or not hid: continue
            try: pos=int(float((x.get("target") or {}).get("finish_position")))
            except: continue
            if pos<=3:g[rid][pos].append(hid)
    out={}
    for rid,z in g.items():
        pieces=[()]
        for pos in sorted(z):
            hs=list(z[pos]); occupied=min(len(hs),3-pos+1)
            perms=list(itertools.permutations(hs,occupied))
            pieces=[a+b for a in pieces for b in perms]
        outcomes=sorted(set(p for p in pieces if len(p)==3))
        if outcomes: out[rid]=outcomes
    return out

def consensus(path):
    out={}
    with open_text(path) as f:
        for line in f:
            if not line.strip():continue
            x=json.loads(line)
            rows=sorted(x["horses"],key=lambda z:int(z["consensus_rank"]))
            out[str(x["race_id"])]=[str(z["horse_id"]) for z in rows]
    return out

def outsiders(path):
    out=defaultdict(dict)
    with open(path,newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ids=[x for x in r["top3_horse_ids"].split("|") if x]
            if not ids: raise ValueError("empty top3")
            out[str(r["race_id"])][r["candidate"]]=ids[0]
    return out

def full_hit(sel,outcomes):
    return any(set(o).issubset(sel) for o in outcomes)

def main():
    a=ap()
    if a.year==2026: raise ValueError("2026 locked")
    t=truth(a.snapshot); c=consensus(a.consensus); o=outsiders(a.outsider_csv)
    if set(t)!=set(c) or len(c)!=3456: raise ValueError("coverage mismatch")
    base_hits=0; union_hits=0; union_size=0; combined_size=0
    indiv={k:{"full_hits":0,"sum_size":0} for k in CANDS}
    for rid,order in c.items():
        top3=set(order[:3]); outcomes=t[rid]
        base_hits += int(full_hit(top3,outcomes))
        u=set()
        for cand in CANDS:
            if cand not in o[rid]: raise ValueError(f"missing {rid} {cand}")
            h=o[rid][cand]; u.add(h)
            s=top3|{h}
            indiv[cand]["full_hits"] += int(full_hit(s,outcomes))
            indiv[cand]["sum_size"] += len(s)
        union_size += len(u)
        combined=top3|u
        combined_size += len(combined)
        union_hits += int(full_hit(combined,outcomes))
    n=len(c)
    res={
      "contract":"L16_OUTSIDER_TOP1_MATRIX_V1","year":a.year,"races":n,
      "l16_top3_full_hits":base_hits,"l16_top3_full_rate":base_hits/n,
      "all5_top1_union_full_hits":union_hits,"all5_top1_union_full_rate":union_hits/n,
      "all5_top1_union_avg_size":union_size/n,
      "l16_top3_plus_all5_top1_avg_size":combined_size/n,
      "candidates":{k:{
        "full_hits":v["full_hits"],
        "full_rate":v["full_hits"]/n,
        "avg_size_with_l16_top3":v["sum_size"]/n
      } for k,v in indiv.items()}
    }
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(res,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(res,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":main()
