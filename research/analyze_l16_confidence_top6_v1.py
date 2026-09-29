#!/usr/bin/env python3
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def load(paths):
    out=[]
    for p in paths:
        with gzip.open(p,"rt",encoding="utf-8",newline="") as f:
            for r in csv.DictReader(f):
                out.append({
                    "year":int(r["year"]),"race_id":r["race_id"],"horse_id":r["horse_id"],
                    "king_rank":int(r["king_rank"]),"vote1":int(r["vote1"]),"vote2":int(r["vote2"]),"vote3":int(r["vote3"]),
                    "is_podium":int(r["is_podium"]),"is_win":int(r["is_win"])
                })
    return out

def fit(train,test):
    def feat(r): return {"king_rank":str(r["king_rank"]),"vote1":r["vote1"],"vote2":r["vote2"],"vote3":r["vote3"]}
    v=DictVectorizer(sparse=True)
    A=v.fit_transform([feat(r) for r in train]).tocsr()
    B=v.transform([feat(r) for r in test]).tocsr()
    A.indices=A.indices.astype(np.int32,copy=False); A.indptr=A.indptr.astype(np.int32,copy=False)
    B.indices=B.indices.astype(np.int32,copy=False); B.indptr=B.indptr.astype(np.int32,copy=False)
    m=LogisticRegression(C=1.0,solver="liblinear",max_iter=1000)
    m.fit(A,[r["is_podium"] for r in train])
    pr=m.predict_proba(B)[:,1]
    z=[]
    for r,p in zip(test,pr):
        q=dict(r); q["confidence"]=float(p); z.append(q)
    return z

def metrics(rows):
    byr=defaultdict(list)
    for r in rows: byr[r["race_id"]].append(r)
    n=0; king_win=conf_win=king_full=conf_full=0
    king_p=conf_p=truth_p=0
    conf_better=king_better=ties=0
    gained=lost=0
    for rid,rs in byr.items():
        truthw={r["horse_id"] for r in rs if r["is_win"]}
        truthpod={r["horse_id"] for r in rs if r["is_podium"]}
        if not truthw or not truthpod: continue
        king=[r for r in sorted(rs,key=lambda x:(x["king_rank"],x["horse_id"]))[:6]]
        conf=[r for r in sorted(rs,key=lambda x:(-x["confidence"],x["king_rank"],x["horse_id"]))[:6]]
        ks={r["horse_id"] for r in king}; cs={r["horse_id"] for r in conf}
        kp=len(ks&truthpod); cp=len(cs&truthpod)
        n+=1; truth_p+=len(truthpod); king_p+=kp; conf_p+=cp
        kw=int(bool(ks&truthw)); cw=int(bool(cs&truthw))
        kf=int(truthpod.issubset(ks)); cf=int(truthpod.issubset(cs))
        king_win+=kw; conf_win+=cw; king_full+=kf; conf_full+=cf
        if cp>kp: conf_better+=1
        elif kp>cp: king_better+=1
        else: ties+=1
        if cf and not kf: gained+=1
        if kf and not cf: lost+=1
    return {
      "races":n,
      "king_top6":{"winner_hits":king_win,"winner_rate":king_win/n,"full_podium_hits":king_full,"full_podium_rate":king_full/n,"podium_occurrence_hits":king_p,"podium_occurrence_recall":king_p/truth_p},
      "confidence_top6":{"winner_hits":conf_win,"winner_rate":conf_win/n,"full_podium_hits":conf_full,"full_podium_rate":conf_full/n,"podium_occurrence_hits":conf_p,"podium_occurrence_recall":conf_p/truth_p},
      "head_to_head":{"confidence_more_podium_races":conf_better,"king_more_podium_races":king_better,"ties":ties,"full_podium_gained":gained,"full_podium_lost":lost,"net_full_podium":gained-lost}
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inputs",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    rows=load(a.inputs.split(","))
    byy=defaultdict(list)
    for r in rows: byy[r["year"]].append(r)
    years={}; pooled=[]
    for y in (2023,2024,2025):
        tr=[r for yy in sorted(byy) if yy<y for r in byy[yy]]
        te=fit(tr,byy[y])
        years[str(y)]=metrics(te)
        pooled.extend(te)
    out={"contract":"L16_CONFIDENCE_TOP6_OOS_V1","scope":"strict walk-forward 2023-2025, fixed Top6, Chimera excluded","years":years,"pooled":metrics(pooled)}
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out["pooled"],ensure_ascii=False))

if __name__=="__main__": main()
