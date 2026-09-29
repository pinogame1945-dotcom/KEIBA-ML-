#!/usr/bin/env python3
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path

CANDS=["outsider_daytrend","outsider_raceshape","outsider_gatecourse","outsider_field","outsider_jockey"]
KS=[3,6]

def ap():
    p=argparse.ArgumentParser()
    p.add_argument("--consensus",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--outsider-csv",required=True,action="append")
    p.add_argument("--year",type=int,required=True)
    p.add_argument("--output",required=True)
    return p.parse_args()

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def load_truth(path):
    out=defaultdict(dict)
    with open_text(path) as f:
        for line in f:
            if not line.strip(): continue
            x=json.loads(line)
            rid=str(x.get("race_id") or ""); hid=str(x.get("horse_id") or "")
            if not rid or not hid: continue
            try: pos=int(float((x.get("target") or {}).get("finish_position")))
            except Exception: continue
            out[rid][hid]=pos
    return out

def load_consensus(path):
    out={}
    with open_text(path) as f:
        for line in f:
            if not line.strip(): continue
            x=json.loads(line)
            rows=sorted(x["horses"], key=lambda z:int(z["consensus_rank"]))
            out[str(x["race_id"])]={str(z["horse_id"]):int(z["consensus_rank"]) for z in rows}
    return out

def load_outsiders(paths):
    out=defaultdict(dict)
    for path in paths:
        with open(path,newline="",encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rid=str(r["race_id"]); cand=r["candidate"]
                t3=[x for x in r["top3_horse_ids"].split("|") if x]
                t6=[x for x in r["top6_horse_ids"].split("|") if x]
                out[rid][cand]={3:set(t3),6:set(t6)}
    return out

def stat():
    return {"n":0,"win":0,"podium":0}

def add(s,pos):
    s["n"]+=1
    if pos==1:s["win"]+=1
    if pos<=3:s["podium"]+=1

def finish(s):
    return {
      **s,
      "win_rate":s["win"]/s["n"] if s["n"] else None,
      "podium_rate":s["podium"]/s["n"] if s["n"] else None,
    }

def bucket(rank):
    if rank<=6:return "4-6"
    if rank<=10:return "7-10"
    return "11+"

def main():
    a=ap()
    if a.year==2026: raise ValueError("2026 locked")
    truth=load_truth(a.snapshot); cons=load_consensus(a.consensus); outs=load_outsiders(a.outsider_csv)
    if len(cons)!=3456 or set(cons)!=set(truth):
        raise ValueError(f"coverage mismatch consensus={len(cons)} truth={len(truth)}")
    for rid in cons:
        missing=[c for c in CANDS if c not in outs[rid]]
        if missing: raise ValueError(f"missing outsider rows race={rid} {missing}")

    result={"contract":"L16_OUTSIDER_AGREEMENT_V1","year":a.year,"races":len(cons),"thresholds":{}}

    for k in KS:
        per={}
        vote_groups={str(v):{"king_topk":stat(),"king_outside":stat()} for v in range(7)}
        any_groups={
          "king_topk_outsider_any":stat(),
          "king_topk_outsider_none":stat(),
          "king_outside_outsider_any":stat(),
        }
        outsider_only_buckets={b:stat() for b in ["4-6","7-10","11+"]}

        for cand in CANDS:
            per[cand]={
              "agreement":stat(),
              "outsider_only":stat(),
              "king_only":stat(),
              "intersection_total":0,
              "outsider_total":0,
              "king_total":0,
              "jaccard_sum":0.0,
              "races":0
            }

        for rid,ranks in cons.items():
            truth_r=truth[rid]
            king={h for h,r in ranks.items() if r<=k}
            outsider_sets={c:outs[rid][c][k] for c in CANDS}

            for cand,s in outsider_sets.items():
                inter=king&s
                oo=s-king
                ko=king-s
                p=per[cand]
                p["races"]+=1
                p["intersection_total"]+=len(inter)
                p["outsider_total"]+=len(s)
                p["king_total"]+=len(king)
                union=king|s
                p["jaccard_sum"]+=len(inter)/len(union) if union else 1.0
                for h in inter:
                    if h in truth_r:add(p["agreement"],truth_r[h])
                for h in oo:
                    if h in truth_r:add(p["outsider_only"],truth_r[h])
                for h in ko:
                    if h in truth_r:add(p["king_only"],truth_r[h])

            all_horses=set(ranks)
            for h in all_horses:
                votes=sum(h in outsider_sets[c] for c in CANDS)
                king_in=h in king
                pos=truth_r.get(h)
                if pos is None: continue
                add(vote_groups[str(votes)]["king_topk" if king_in else "king_outside"],pos)
                if king_in and votes>=1:add(any_groups["king_topk_outsider_any"],pos)
                elif king_in:add(any_groups["king_topk_outsider_none"],pos)
                elif votes>=1:
                    add(any_groups["king_outside_outsider_any"],pos)
                    add(outsider_only_buckets[bucket(ranks[h])],pos)

        for cand,p in per.items():
            p["agreement"]=finish(p["agreement"])
            p["outsider_only"]=finish(p["outsider_only"])
            p["king_only"]=finish(p["king_only"])
            p["avg_intersection_per_race"]=p["intersection_total"]/p["races"]
            p["overlap_share_of_outsider_selections"]=p["intersection_total"]/p["outsider_total"] if p["outsider_total"] else None
            p["mean_jaccard"]=p["jaccard_sum"]/p["races"]
            del p["jaccard_sum"]

        result["thresholds"][str(k)]={
          "per_candidate":per,
          "vote_groups":{v:{g:finish(s) for g,s in x.items()} for v,x in vote_groups.items()},
          "any_outsider_groups":{g:finish(s) for g,s in any_groups.items()},
          "outsider_only_l16_rank_buckets":{b:finish(s) for b,s in outsider_only_buckets.items()}
        }

    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__": main()
