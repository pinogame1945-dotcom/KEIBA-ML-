#!/usr/bin/env python3
import argparse,gzip,json,math
from collections import defaultdict
from pathlib import Path

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    return p.parse_args()

def open_text(path,mode):
    return gzip.open(path,mode,encoding="utf-8") if str(path).endswith(".gz") else open(path,mode,encoding="utf-8")

def finite(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None

def main():
    a=args()
    races=defaultdict(list)
    with open_text(a.input,"rt") as fh:
        for line in fh:
            if not line.strip(): continue
            r=json.loads(line)
            rid=str(r.get("race_id") or "")
            hid=str(r.get("horse_id") or "")
            if not rid or not hid:
                raise ValueError("missing race_id/horse_id")
            score=finite(r.get("raw_margin_logit"))
            if score is None: score=finite(r.get("raw_win_probability"))
            if score is None: raise ValueError("missing score")
            r["_safe_score"]=score
            races[rid].append(r)

    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with open_text(out,"wt") as fh:
        for rid in sorted(races):
            rows=sorted(races[rid],key=lambda r:(-r["_safe_score"],str(r.get("horse_id") or "")))
            for rank,r in enumerate(rows,start=1):
                r["predicted_rank"]=rank
                r.pop("_safe_score",None)
                fh.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")
    print(json.dumps({"races":len(races),"rows":sum(map(len,races.values())),"output":str(out)},separators=(",",":")))

if __name__=="__main__":
    main()
