#!/usr/bin/env python3
import argparse,csv,gzip,json,math
from collections import defaultdict
from pathlib import Path

def ap():
    p=argparse.ArgumentParser()
    p.add_argument("--score",required=True)
    p.add_argument("--candidate",required=True)
    p.add_argument("--output",required=True)
    return p.parse_args()

def opent(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def finite(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None

def horse_number_key(v):
    try:
        return (0,int(float(v)))
    except (TypeError,ValueError):
        return (1,999999)

def main():
    a=ap()
    by=defaultdict(list)
    with opent(a.score) as fh:
        for line in fh:
            if not line.strip(): continue
            r=json.loads(line)
            rid=str(r.get("race_id") or "")
            hid=str(r.get("horse_id") or "")
            score=finite(r.get("raw_margin_logit"))
            if score is None: score=finite(r.get("raw_win_probability"))
            if not rid or not hid or score is None:
                raise ValueError("bad score row")
            by[rid].append({
                "horse_id":hid,
                "horse_number":r.get("horse_number"),
                "score":score,
            })

    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8-sig") as fh:
        w=csv.DictWriter(fh,fieldnames=["race_id","candidate","top1_horse_id","top2_horse_id","top3_horse_id"])
        w.writeheader()
        for rid in sorted(by):
            rows=sorted(by[rid],key=lambda r:(-r["score"],horse_number_key(r["horse_number"]),r["horse_id"]))
            ids=[r["horse_id"] for r in rows[:3]]
            while len(ids)<3: ids.append("")
            w.writerow({
                "race_id":rid,
                "candidate":a.candidate,
                "top1_horse_id":ids[0],
                "top2_horse_id":ids[1],
                "top3_horse_id":ids[2],
            })
    print(json.dumps({"candidate":a.candidate,"races":len(by),"output":str(out)},ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
