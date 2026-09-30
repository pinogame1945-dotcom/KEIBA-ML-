#!/usr/bin/env python3
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path

def ap():
    p=argparse.ArgumentParser()
    p.add_argument("--consensus",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--top3-dir",required=True)
    p.add_argument("--year",type=int,required=True)
    p.add_argument("--output",required=True)
    return p.parse_args()

def opent(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def load_truth(path):
    out=defaultdict(dict)
    horse_num=defaultdict(dict)
    with opent(path) as fh:
        for line in fh:
            if not line.strip(): continue
            x=json.loads(line)
            rid=str(x.get("race_id") or ""); hid=str(x.get("horse_id") or "")
            if not rid or not hid: continue
            horse_num[rid][hid]=x.get("horse_number")
            try: pos=int(float((x.get("target") or {}).get("finish_position")))
            except (TypeError,ValueError): continue
            out[rid][hid]=pos
    return out,horse_num

def load_consensus(path):
    out={}
    with opent(path) as fh:
        for line in fh:
            if not line.strip(): continue
            x=json.loads(line)
            rows=sorted(x["horses"],key=lambda z:int(z["consensus_rank"]))
            out[str(x["race_id"])]={str(z["horse_id"]):int(z["consensus_rank"]) for z in rows}
    return out

def load_votes(path):
    out=defaultdict(dict); cands=set()
    for p in sorted(Path(path).glob("*.csv")):
        with p.open(newline="",encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                rid=str(r["race_id"]); c=str(r["candidate"]); cands.add(c)
                ranks={}
                for k in (1,2,3):
                    h=str(r.get(f"top{k}_horse_id") or "")
                    if h: ranks[h]=k
                out[rid][c]=ranks
    return out,sorted(cands)

def hnum(v):
    try: return (0,int(float(v)))
    except (TypeError,ValueError): return (1,999999)

def metric():
    return {"hits":0,"races":0}
def add(m,hit):
    m["races"]+=1; m["hits"]+=int(bool(hit))
def fin(m):
    return {**m,"rate":m["hits"]/m["races"] if m["races"] else None}

def main():
    a=ap()
    if a.year==2026: raise ValueError("2026 locked")
    truth,nums=load_truth(a.snapshot)
    kings=load_consensus(a.consensus)
    votes,cands=load_votes(a.top3_dir)
    if len(cands)!=13: raise ValueError(f"expected 13 candidates, got {len(cands)}")
    if set(kings)!=set(truth): raise ValueError("coverage mismatch")

    schemes=["weighted","count"]
    topns=(1,3,6)
    metrics={s:{str(k):metric() for k in topns} for s in schemes}
    blind3={s:{str(k):metric() for k in topns} for s in schemes}
    blind6={s:{str(k):metric() for k in topns} for s in schemes}
    unique_positive=[]

    for rid,kranks in kings.items():
        if any(c not in votes[rid] for c in cands):
            raise ValueError(f"missing candidate votes race={rid}")
        horses=list(kranks)
        scored=[]
        for hid in horses:
            c=0; w=0; v1=v2=v3=0
            for cand in cands:
                r=votes[rid][cand].get(hid)
                if r is None: continue
                c+=1; w+=(4-r)
                if r==1:v1+=1
                elif r==2:v2+=1
                elif r==3:v3+=1
            scored.append({"horse_id":hid,"count":c,"weighted":w,"v1":v1,"v2":v2,"v3":v3,"num":nums[rid].get(hid)})
        unique_positive.append(sum(x["weighted"]>0 for x in scored))
        order={
            "weighted":sorted(scored,key=lambda x:(-x["weighted"],-x["v1"],-x["count"],-x["v2"],-x["v3"],hnum(x["num"]),x["horse_id"])),
            "count":sorted(scored,key=lambda x:(-x["count"],-x["weighted"],-x["v1"],-x["v2"],-x["v3"],hnum(x["num"]),x["horse_id"])),
        }
        winner=next((h for h,p in truth[rid].items() if p==1),None)
        if winner is None: continue
        king3={h for h,r in kranks.items() if r<=3}
        king6={h for h,r in kranks.items() if r<=6}
        b3=winner not in king3
        b6=winner not in king6
        for s in schemes:
            ids=[x["horse_id"] for x in order[s]]
            for k in topns:
                hit=winner in set(ids[:k])
                add(metrics[s][str(k)],hit)
                if b3:add(blind3[s][str(k)],hit)
                if b6:add(blind6[s][str(k)],hit)

    out={
      "contract":"L16_OUTSIDER_COUNCIL_SAFE_V1",
      "year":a.year,
      "races":len(kings),
      "candidate_count":len(cands),
      "candidates":cands,
      "weighted_policy":"rank1=3, rank2=2, rank3=1, outside=0; ties: top1 votes, support count, top2 votes, top3 votes, horse_number, horse_id",
      "count_policy":"Top3 support count; ties: weighted score, top1 votes, top2 votes, top3 votes, horse_number, horse_id",
      "metrics":{s:{k:fin(v) for k,v in d.items()} for s,d in metrics.items()},
      "seven_top3_blind":{"count":sum(1 for rid,kr in kings.items() if next((h for h,p in truth[rid].items() if p==1),None) not in {h for h,r in kr.items() if r<=3}),
                          "council":{s:{k:fin(v) for k,v in d.items()} for s,d in blind3.items()}},
      "seven_top6_blind":{"count":sum(1 for rid,kr in kings.items() if next((h for h,p in truth[rid].items() if p==1),None) not in {h for h,r in kr.items() if r<=6}),
                          "council":{s:{k:fin(v) for k,v in d.items()} for s,d in blind6.items()}},
      "positive_vote_horses_per_race":{"min":min(unique_positive),"max":max(unique_positive),"mean":sum(unique_positive)/len(unique_positive)},
      "2026_sealed":True,
      "odds_used":False,
    }
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L16_OUTSIDER_COUNCIL_SAFE_OK",a.year)

if __name__=="__main__":
    main()
