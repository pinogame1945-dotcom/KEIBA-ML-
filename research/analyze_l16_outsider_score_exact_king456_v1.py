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

def blank():
    return {"n":0,"wins":0,"top3":0,"top6":0,"finish_sum":0,"finish_n":0,"missing_finish":0}

def add(d,pos):
    d["n"]+=1
    if pos is None:
        d["missing_finish"]+=1
        return
    d["finish_sum"]+=pos; d["finish_n"]+=1
    d["wins"]+=int(pos==1); d["top3"]+=int(pos<=3); d["top6"]+=int(pos<=6)

def fin(d):
    n=d["finish_n"]
    return {**d,
      "win_rate":d["wins"]/n if n else None,
      "top3_rate":d["top3"]/n if n else None,
      "top6_rate":d["top6"]/n if n else None,
      "avg_finish":d["finish_sum"]/n if n else None}

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
    with opent(a.snapshot) as fh:
        for line in fh:
            if not line.strip(): continue
            x=json.loads(line); rid=str(x.get("race_id") or ""); hid=str(x.get("horse_id") or "")
            if not rid or not hid: continue
            try: truth[rid][hid]=int(float((x.get("target") or {}).get("finish_position")))
            except (TypeError,ValueError): pass

    exact=defaultdict(blank)
    bucket=defaultdict(blank)
    overall=defaultdict(blank)
    for rid,hs in ranks.items():
        for hid,kr in hs.items():
            if kr not in (4,5,6): continue
            s=pts[rid].get(hid,0)
            pos=truth[rid].get(hid)
            add(overall[kr],pos)
            add(exact[(kr,s)],pos)
            add(bucket[(kr,score_bucket(s) if s>0 else "0")],pos)

    out={"contract":"L16_OUTSIDER_SCORE_EXACT_KING456_OUTCOME_V1","year":a.year,
         "score_definition":"13 outsider Top3 ballots: rank1=3, rank2=2, rank3=1, otherwise 0",
         "overall_by_king_rank":{str(k):fin(overall[k]) for k in (4,5,6)},
         "score_bucket_by_king_rank":{
           str(k):{b:fin(bucket[(k,b)]) for b in ("0","1-5","6-10","11-15","16-20","21-25","26+")}
           for k in (4,5,6)},
         "score_exact_by_king_rank":{
           str(k):{str(s):fin(exact[(k,s)]) for s in range(0,40) if exact[(k,s)]["n"]>0}
           for k in (4,5,6)},
         "2026_sealed":True,"odds_used":False}
    p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("OUTSIDER_SCORE_EXACT_KING456_OK",a.year)

if __name__=="__main__":
    main()
