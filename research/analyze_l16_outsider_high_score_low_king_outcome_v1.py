#!/usr/bin/env python3
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path

def ap():
    p=argparse.ArgumentParser()
    p.add_argument("--ballots",required=True)
    p.add_argument("--consensus",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--year",type=int,required=True)
    p.add_argument("--output",required=True)
    return p.parse_args()

def opent(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def score_bucket(s):
    if s<=5:return "1-5"
    if s<=10:return "6-10"
    if s<=15:return "11-15"
    if s<=20:return "16-20"
    if s<=25:return "21-25"
    return "26+"

def king_bucket(r):
    if r==1:return "1"
    if r==2:return "2"
    if r==3:return "3"
    if r<=6:return "4-6"
    if r<=10:return "7-10"
    return "11+"

def blank():
    return {"n":0,"wins":0,"top3":0,"top6":0,"finish_sum":0,"finish_n":0,"missing_finish":0}

def add(d,pos):
    d["n"]+=1
    if pos is None:
        d["missing_finish"]+=1
        return
    d["finish_sum"]+=pos; d["finish_n"]+=1
    d["wins"]+=int(pos==1)
    d["top3"]+=int(pos<=3)
    d["top6"]+=int(pos<=6)

def fin(d):
    n=d["finish_n"]
    return {**d,
      "win_rate":d["wins"]/n if n else None,
      "top3_rate":d["top3"]/n if n else None,
      "top6_rate":d["top6"]/n if n else None,
      "avg_finish":d["finish_sum"]/n if n else None,
    }

def main():
    a=ap()
    if a.year>=2026: raise ValueError("2026 sealed")

    pts=defaultdict(lambda:defaultdict(int))
    cands=set()
    with opent(a.ballots) as fh:
        for row in csv.DictReader(fh):
            rid=str(row["race_id"]); c=str(row["candidate"]); cands.add(c)
            for k,w in ((1,3),(2,2),(3,1)):
                hid=str(row.get(f"top{k}_horse_id") or "")
                if hid: pts[rid][hid]+=w
    if len(cands)!=13: raise ValueError(f"expected 13 candidates got {len(cands)}")

    ranks={}
    with opent(a.consensus) as fh:
        for line in fh:
            if not line.strip(): continue
            x=json.loads(line)
            ranks[str(x["race_id"])]={str(h["horse_id"]):int(h["consensus_rank"]) for h in x["horses"]}

    truth=defaultdict(dict)
    raw=defaultdict(dict)
    with opent(a.snapshot) as fh:
        for line in fh:
            if not line.strip(): continue
            x=json.loads(line); rid=str(x.get("race_id") or ""); hid=str(x.get("horse_id") or "")
            if not rid or not hid: continue
            v=(x.get("target") or {}).get("finish_position"); raw[rid][hid]=v
            try: truth[rid][hid]=int(float(v))
            except (TypeError,ValueError): pass

    cross=defaultdict(blank)
    focus={
      "score21plus_king7plus":blank(),
      "score21plus_king7to10":blank(),
      "score21plus_king11plus":blank(),
      "score16plus_king7plus":blank(),
      "score21plus_king4to6":blank(),
    }
    exact_score_lowrank=defaultdict(blank)
    exact456=defaultdict(blank)
    bucket456=defaultdict(blank)
    overall456=defaultdict(blank)
    exact7to10=defaultdict(blank)
    bucket7to10=defaultdict(blank)
    overall7to10=defaultdict(blank)

    for rid,hs in pts.items():
        for hid,s in hs.items():
            if s<=0: continue
            kr=ranks[rid].get(hid)
            if kr is None: raise ValueError(f"missing rank {rid} {hid}")
            pos=truth[rid].get(hid)
            add(cross[(score_bucket(s),king_bucket(kr))],pos)
            if s>=21 and kr>=7:add(focus["score21plus_king7plus"],pos)
            if s>=21 and 7<=kr<=10:add(focus["score21plus_king7to10"],pos)
            if s>=21 and kr>=11:add(focus["score21plus_king11plus"],pos)
            if s>=16 and kr>=7:add(focus["score16plus_king7plus"],pos)
            if s>=21 and 4<=kr<=6:add(focus["score21plus_king4to6"],pos)
            if kr>=7:add(exact_score_lowrank[s],pos)

    # Exact King ranks 4/5/6/7/8/9/10, including zero Outsider points.
    for rid,hs in ranks.items():
        for hid,kr in hs.items():
            if kr not in (4,5,6,7,8,9,10): continue
            s=pts[rid].get(hid,0)
            pos=truth[rid].get(hid)
            b="0" if s==0 else score_bucket(s)
            if kr in (4,5,6):
                add(overall456[kr],pos)
                add(exact456[(kr,s)],pos)
                add(bucket456[(kr,b)],pos)
            else:
                add(overall7to10[kr],pos)
                add(exact7to10[(kr,s)],pos)
                add(bucket7to10[(kr,b)],pos)

    order_s=("1-5","6-10","11-15","16-20","21-25","26+")
    order_k=("1","2","3","4-6","7-10","11+")
    out={
      "contract":"L16_OUTSIDER_HIGH_SCORE_LOW_KING_OUTCOME_V1",
      "year":a.year,
      "score_definition":"13 outsider Top3 ballots: 3/2/1 points",
      "cross":{sb:{kb:fin(cross[(sb,kb)]) for kb in order_k} for sb in order_s},
      "focus":{k:fin(v) for k,v in focus.items()},
      "exact_score_king7plus":{str(s):fin(v) for s,v in sorted(exact_score_lowrank.items())},
      "king7to10":{
        "overall":{str(k):fin(overall7to10[k]) for k in (7,8,9,10)},
        "score_bucket":{str(k):{b:fin(bucket7to10[(k,b)]) for b in ("0","1-5","6-10","11-15","16-20","21-25","26+")} for k in (7,8,9,10)},
        "score_exact":{str(k):{str(s):fin(exact7to10[(k,s)]) for s in range(40) if exact7to10[(k,s)]["n"]>0} for k in (7,8,9,10)}
      },
      "king456":{
        "overall":{str(k):fin(overall456[k]) for k in (4,5,6)},
        "score_bucket":{str(k):{b:fin(bucket456[(k,b)]) for b in ("0","1-5","6-10","11-15","16-20","21-25","26+")} for k in (4,5,6)},
        "score_exact":{str(k):{str(s):fin(exact456[(k,s)]) for s in range(40) if exact456[(k,s)]["n"]>0} for k in (4,5,6)}
      },
      "2026_sealed":True,
      "odds_used":False,
    }
    p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("HIGH_SCORE_LOW_KING_OUTCOME_OK",a.year,json.dumps(out["focus"],ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
