#!/usr/bin/env python3
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path

def ap():
    p=argparse.ArgumentParser()
    p.add_argument("--ballots",required=True)
    p.add_argument("--consensus",required=True)
    p.add_argument("--year",type=int,required=True)
    p.add_argument("--output",required=True)
    return p.parse_args()

def opent(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def king_bucket(r):
    if r==1:return "1"
    if r==2:return "2"
    if r==3:return "3"
    if r<=6:return "4-6"
    if r<=10:return "7-10"
    return "11+"

def score_bucket(s):
    if s<=5:return "1-5"
    if s<=10:return "6-10"
    if s<=15:return "11-15"
    if s<=20:return "16-20"
    if s<=25:return "21-25"
    return "26+"

def main():
    a=ap()
    if a.year>=2026: raise ValueError("2026 sealed")

    pts=defaultdict(lambda:defaultdict(int))
    cnt=defaultdict(lambda:defaultdict(int))
    candidates=set()
    with opent(a.ballots) as fh:
        r=csv.DictReader(fh)
        for row in r:
            rid=str(row["race_id"]); cand=str(row["candidate"]); candidates.add(cand)
            seen=set()
            for k,w in ((1,3),(2,2),(3,1)):
                hid=str(row.get(f"top{k}_horse_id") or "")
                if not hid: continue
                pts[rid][hid]+=w
                if hid not in seen:
                    cnt[rid][hid]+=1; seen.add(hid)
    if len(candidates)!=13:
        raise ValueError(f"expected 13 candidates, got {len(candidates)}")

    ranks={}
    with opent(a.consensus) as fh:
        for line in fh:
            if not line.strip(): continue
            x=json.loads(line)
            rid=str(x["race_id"])
            ranks[rid]={str(h["horse_id"]):int(h["consensus_rank"]) for h in x["horses"]}

    exact=defaultdict(lambda:{"horses":0,"score_sum":0,"support_count_sum":0})
    kb=defaultdict(lambda:{"horses":0,"score_sum":0})
    cross=defaultdict(lambda:defaultdict(int))
    byscore=defaultdict(lambda:defaultdict(int))
    total=0
    for rid,hs in pts.items():
        if rid not in ranks: raise ValueError(f"missing consensus race {rid}")
        for hid,s in hs.items():
            if s<=0: continue
            kr=ranks[rid].get(hid)
            if kr is None: raise ValueError(f"missing horse in consensus race={rid} horse={hid}")
            total+=1
            e=exact[kr]; e["horses"]+=1; e["score_sum"]+=s; e["support_count_sum"]+=cnt[rid][hid]
            b=king_bucket(kr); kb[b]["horses"]+=1; kb[b]["score_sum"]+=s
            cross[score_bucket(s)][b]+=1
            byscore[s][b]+=1

    exact_out={}
    for kr,d in sorted(exact.items()):
        exact_out[str(kr)]={
            **d,
            "share":d["horses"]/total,
            "avg_outsider_score":d["score_sum"]/d["horses"],
            "avg_support_count":d["support_count_sum"]/d["horses"],
        }
    kb_out={}
    for b in ("1","2","3","4-6","7-10","11+"):
        d=kb[b]
        kb_out[b]={
            **d,
            "share":d["horses"]/total if total else 0,
            "avg_outsider_score":d["score_sum"]/d["horses"] if d["horses"] else None,
        }
    cross_out={}
    for sb in ("1-5","6-10","11-15","16-20","21-25","26+"):
        row={b:int(cross[sb][b]) for b in ("1","2","3","4-6","7-10","11+")}
        n=sum(row.values())
        cross_out[sb]={"horses":n,"king_rank_buckets":row,
                       "shares":{b:(row[b]/n if n else 0) for b in row}}
    score_exact={}
    for s,d in sorted(byscore.items()):
        n=sum(d.values())
        score_exact[str(s)]={"horses":n,"king_rank_buckets":{b:int(d[b]) for b in ("1","2","3","4-6","7-10","11+")}}

    out={
      "contract":"L16_OUTSIDER_SCORE_X_KING_RANK_V1",
      "year":a.year,
      "races":len(ranks),
      "scored_horse_rows":total,
      "candidate_count":13,
      "score_definition":"sum of 13 outsider Top3 ballots: rank1=3, rank2=2, rank3=1",
      "king_rank_exact":exact_out,
      "king_rank_buckets":kb_out,
      "score_bucket_x_king_rank":cross_out,
      "score_exact_x_king_rank":score_exact,
      "2026_sealed":True,
      "odds_used":False,
    }
    p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("OUTSIDER_SCORE_X_KING_RANK_OK",a.year,total)

if __name__=="__main__":
    main()
