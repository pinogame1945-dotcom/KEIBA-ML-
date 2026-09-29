#!/usr/bin/env python3
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def load_rows(paths):
    out=[]
    for path in paths:
        with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
            for r in csv.DictReader(f):
                out.append({
                    "year":int(r["year"]),"race_id":str(r["race_id"]),"horse_id":str(r["horse_id"]),
                    "king_rank":int(r["king_rank"]),"vote1":int(r["vote1"]),"vote2":int(r["vote2"]),
                    "vote3":int(r["vote3"]),"is_podium":int(r["is_podium"]),"is_win":int(r["is_win"])
                })
    return out

def fit_predict(train,test):
    def feats(r):
        return {"king_rank":str(r["king_rank"]),"vote1":r["vote1"],"vote2":r["vote2"],"vote3":r["vote3"]}
    vec=DictVectorizer(sparse=True)
    A=vec.fit_transform([feats(r) for r in train]).tocsr()
    B=vec.transform([feats(r) for r in test]).tocsr()
    A.indices=A.indices.astype(np.int32,copy=False); A.indptr=A.indptr.astype(np.int32,copy=False)
    B.indices=B.indices.astype(np.int32,copy=False); B.indptr=B.indptr.astype(np.int32,copy=False)
    m=LogisticRegression(C=1.0,solver="liblinear",max_iter=1000)
    m.fit(A,[r["is_podium"] for r in train])
    probs=m.predict_proba(B)[:,1]
    z=[]
    for r,p in zip(test,probs):
        q=dict(r); q["confidence"]=float(p); z.append(q)
    return z

def load_fixed(path):
    d={}
    with open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line); d[str(x["race_id"])]=x
    return d

def eval_race(rows,current):
    field={r["horse_id"]:r for r in rows}
    truthp={r["horse_id"] for r in rows if r["is_podium"]}
    truthw={r["horse_id"] for r in rows if r["is_win"]}
    current_ids=[h for h in current["candidate_horse_ids"] if h in field]
    k=len(current_ids)
    conf=sorted(rows,key=lambda r:(-r["confidence"],r["king_rank"],r["horse_id"]))[:k]
    conf_ids=[r["horse_id"] for r in conf]
    a=set(current_ids); b=set(conf_ids)
    return {
      "race_id":current["race_id"],"alert":bool(current.get("gate_alert")),"k":k,
      "truth_p":len(truthp),
      "cur_p":len(a&truthp),"new_p":len(b&truthp),
      "cur_full":int(truthp.issubset(a)),"new_full":int(truthp.issubset(b)),
      "cur_win":int(bool(a&truthw)),"new_win":int(bool(b&truthw))
    }

def metric(rs):
    n=len(rs); tp=sum(x["truth_p"] for x in rs)
    cf=sum(x["cur_full"] for x in rs); nf=sum(x["new_full"] for x in rs)
    cw=sum(x["cur_win"] for x in rs); nw=sum(x["new_win"] for x in rs)
    cp=sum(x["cur_p"] for x in rs); npd=sum(x["new_p"] for x in rs)
    better=sum(x["new_p"]>x["cur_p"] for x in rs)
    worse=sum(x["cur_p"]>x["new_p"] for x in rs)
    gain=sum(x["new_full"] and not x["cur_full"] for x in rs)
    loss=sum(x["cur_full"] and not x["new_full"] for x in rs)
    return {
      "races":n,"avg_candidate_n":sum(x["k"] for x in rs)/n if n else None,
      "current":{"full_podium_hits":cf,"full_podium_rate":cf/n if n else None,
                 "winner_hits":cw,"winner_rate":cw/n if n else None,
                 "podium_occurrence_hits":cp,"podium_occurrence_recall":cp/tp if tp else None},
      "confidence":{"full_podium_hits":nf,"full_podium_rate":nf/n if n else None,
                    "winner_hits":nw,"winner_rate":nw/n if n else None,
                    "podium_occurrence_hits":npd,"podium_occurrence_recall":npd/tp if tp else None},
      "head_to_head":{"confidence_more_podium_races":better,"current_more_podium_races":worse,
                      "ties":n-better-worse,"full_podium_gained":gain,"full_podium_lost":loss,
                      "net_full_podium":gain-loss}
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rows",required=True)
    p.add_argument("--fixed-root",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    rows=load_rows(a.rows.split(","))
    byy=defaultdict(list)
    for r in rows: byy[r["year"]].append(r)

    all_oos=[]; years={}
    for y in (2023,2024,2025):
        train=[r for yy in sorted(byy) if yy<y for r in byy[yy]]
        test=fit_predict(train,byy[y])
        byr=defaultdict(list)
        for r in test: byr[r["race_id"]].append(r)
        fixed=load_fixed(Path(a.fixed_root)/f"y{y}.jsonl")
        rs=[]; missing=[]
        for rid,x in fixed.items():
            if rid not in byr:
                missing.append(rid); continue
            e=eval_race(byr[rid],x)
            if e["k"]>0: rs.append(e)
        if len(rs)!=3456:
            raise SystemExit(f"year {y} comparable race regression: {len(rs)} != 3456; missing={len(missing)}")
        years[str(y)]={
          "all":metric(rs),
          "alert":metric([x for x in rs if x["alert"]]),
          "non_alert":metric([x for x in rs if not x["alert"]])
        }
        all_oos.extend(rs)

    out={
      "contract":"L16_CONFIDENCE_VS_L15_FULL_OOS_V1",
      "scope":"2023-2025 strict walk-forward OOS; all 10,368 races; identical effective candidate count per race",
      "training":"2023 trained on 2022; 2024 on 2022-2023; 2025 on 2022-2024",
      "locked_year":2026,
      "years":years,
      "pooled":{
        "all":metric(all_oos),
        "alert":metric([x for x in all_oos if x["alert"]]),
        "non_alert":metric([x for x in all_oos if not x["alert"]])
      }
    }
    q=Path(a.output); q.parent.mkdir(parents=True,exist_ok=True)
    q.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out["pooled"],ensure_ascii=False))

if __name__=="__main__":
    main()
