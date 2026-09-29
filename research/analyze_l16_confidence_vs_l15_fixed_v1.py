#!/usr/bin/env python3
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def load_rows(paths):
    rows=[]
    for path in paths:
        with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "year":int(r["year"]),"race_id":r["race_id"],"horse_id":r["horse_id"],
                    "king_rank":int(r["king_rank"]),"vote1":int(r["vote1"]),"vote2":int(r["vote2"]),
                    "vote3":int(r["vote3"]),"out_votes":int(r["out_votes"]),
                    "is_podium":int(r["is_podium"]),"is_win":int(r["is_win"])
                })
    return rows

def fit_predict(train,test):
    Xtr=[{"king_rank":str(r["king_rank"]),"vote1":r["vote1"],"vote2":r["vote2"],"vote3":r["vote3"]} for r in train]
    Xte=[{"king_rank":str(r["king_rank"]),"vote1":r["vote1"],"vote2":r["vote2"],"vote3":r["vote3"]} for r in test]
    vec=DictVectorizer(sparse=True)
    A=vec.fit_transform(Xtr).tocsr(); B=vec.transform(Xte).tocsr()
    A.indices=A.indices.astype(np.int32,copy=False); A.indptr=A.indptr.astype(np.int32,copy=False)
    B.indices=B.indices.astype(np.int32,copy=False); B.indptr=B.indptr.astype(np.int32,copy=False)
    model=LogisticRegression(C=1.0,solver="liblinear",max_iter=1000)
    model.fit(A,[r["is_podium"] for r in train])
    p=model.predict_proba(B)[:,1]
    out=[]
    for r,pr in zip(test,p):
        z=dict(r); z["confidence"]=float(pr); out.append(z)
    return out

def load_fixed(path):
    out={}
    with open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x=json.loads(line); out[str(x["race_id"])]=x
    return out

def summarize_race(rows,current_ids):
    field={r["horse_id"]:r for r in rows}
    truth_podium={r["horse_id"] for r in rows if r["is_podium"]==1}
    truth_win={r["horse_id"] for r in rows if r["is_win"]==1}
    current=[h for h in current_ids if h in field]
    k=len(current)
    conf=sorted(rows,key=lambda r:(-r["confidence"],r["king_rank"],r["horse_id"]))[:k]
    conf_ids=[r["horse_id"] for r in conf]
    curset=set(current); confset=set(conf_ids)
    cur_p=len(curset & truth_podium); conf_p=len(confset & truth_podium)
    return {
        "k":k,"field_n":len(rows),"truth_podium_n":len(truth_podium),"truth_win_n":len(truth_win),
        "current_podium_hits":cur_p,"confidence_podium_hits":conf_p,
        "current_full_podium":int(truth_podium.issubset(curset)),
        "confidence_full_podium":int(truth_podium.issubset(confset)),
        "current_win_hit":int(bool(curset & truth_win)),
        "confidence_win_hit":int(bool(confset & truth_win)),
        "current_only_podium":len((curset-confset)&truth_podium),
        "confidence_only_podium":len((confset-curset)&truth_podium),
        "current_ids":current,"confidence_ids":conf_ids
    }

def pct(a,b): return a/b if b else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rows",required=True,help="comma-separated 2022-2025 row files")
    ap.add_argument("--fixed-root",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    rows=load_rows(a.rows.split(","))
    by_year=defaultdict(list)
    for r in rows: by_year[r["year"]].append(r)

    years={}
    pooled=[]
    for y in [2023,2024,2025]:
        train=[r for yr in by_year for r in by_year[yr] if yr<y]
        test=fit_predict(train,by_year[y])
        races=defaultdict(list)
        for r in test: races[r["race_id"]].append(r)
        fixed=load_fixed(Path(a.fixed_root)/f"y{y}.jsonl")
        details=[]
        skipped=[]
        for rid,x in fixed.items():
            if rid not in races:
                skipped.append({"race_id":rid,"reason":"missing confidence rows"}); continue
            s=summarize_race(races[rid],x["candidate_horse_ids"])
            if s["k"]==0:
                skipped.append({"race_id":rid,"reason":"zero effective current candidates"}); continue
            s["race_id"]=rid; details.append(s)
        n=len(details)
        cur_full=sum(d["current_full_podium"] for d in details)
        con_full=sum(d["confidence_full_podium"] for d in details)
        cur_win=sum(d["current_win_hit"] for d in details)
        con_win=sum(d["confidence_win_hit"] for d in details)
        cur_ph=sum(d["current_podium_hits"] for d in details)
        con_ph=sum(d["confidence_podium_hits"] for d in details)
        truth_p=sum(d["truth_podium_n"] for d in details)
        conf_better=sum(d["confidence_podium_hits"]>d["current_podium_hits"] for d in details)
        cur_better=sum(d["current_podium_hits"]>d["confidence_podium_hits"] for d in details)
        tie=n-conf_better-cur_better
        gained_full=sum(d["confidence_full_podium"] and not d["current_full_podium"] for d in details)
        lost_full=sum(d["current_full_podium"] and not d["confidence_full_podium"] for d in details)
        ys={
          "year":y,"alert_races":n,"skipped":skipped,
          "avg_candidate_n":sum(d["k"] for d in details)/n if n else None,
          "current":{
             "full_podium_hits":cur_full,"full_podium_rate":pct(cur_full,n),
             "winner_hits":cur_win,"winner_rate":pct(cur_win,n),
             "podium_occurrence_hits":cur_ph,"podium_occurrence_recall":pct(cur_ph,truth_p)
          },
          "confidence_same_k":{
             "full_podium_hits":con_full,"full_podium_rate":pct(con_full,n),
             "winner_hits":con_win,"winner_rate":pct(con_win,n),
             "podium_occurrence_hits":con_ph,"podium_occurrence_recall":pct(con_ph,truth_p)
          },
          "head_to_head":{
             "confidence_more_podium_races":conf_better,
             "current_more_podium_races":cur_better,
             "ties":tie,
             "full_podium_gained":gained_full,
             "full_podium_lost":lost_full,
             "net_full_podium":gained_full-lost_full
          },
          "details":details
        }
        years[str(y)]=ys; pooled.extend(details)

    n=len(pooled); truth_p=sum(d["truth_podium_n"] for d in pooled)
    cur_full=sum(d["current_full_podium"] for d in pooled); con_full=sum(d["confidence_full_podium"] for d in pooled)
    cur_win=sum(d["current_win_hit"] for d in pooled); con_win=sum(d["confidence_win_hit"] for d in pooled)
    cur_ph=sum(d["current_podium_hits"] for d in pooled); con_ph=sum(d["confidence_podium_hits"] for d in pooled)
    cb=sum(d["confidence_podium_hits"]>d["current_podium_hits"] for d in pooled)
    kb=sum(d["current_podium_hits"]>d["confidence_podium_hits"] for d in pooled)
    gf=sum(d["confidence_full_podium"] and not d["current_full_podium"] for d in pooled)
    lf=sum(d["current_full_podium"] and not d["confidence_full_podium"] for d in pooled)
    out={
      "contract":"L16_CONFIDENCE_VS_L15_FIXED_V1",
      "scope":"L15_FIXED_V1 alert races only; 2023-2025 walk-forward OOS; identical effective candidate count per race",
      "locked_year":2026,
      "years":years,
      "pooled":{
        "alert_races":n,"avg_candidate_n":sum(d["k"] for d in pooled)/n if n else None,
        "current":{"full_podium_hits":cur_full,"full_podium_rate":pct(cur_full,n),"winner_hits":cur_win,"winner_rate":pct(cur_win,n),"podium_occurrence_hits":cur_ph,"podium_occurrence_recall":pct(cur_ph,truth_p)},
        "confidence_same_k":{"full_podium_hits":con_full,"full_podium_rate":pct(con_full,n),"winner_hits":con_win,"winner_rate":pct(con_win,n),"podium_occurrence_hits":con_ph,"podium_occurrence_recall":pct(con_ph,truth_p)},
        "head_to_head":{"confidence_more_podium_races":cb,"current_more_podium_races":kb,"ties":n-cb-kb,"full_podium_gained":gf,"full_podium_lost":lf,"net_full_podium":gf-lf}
      }
    }
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out["pooled"],ensure_ascii=False))

if __name__=="__main__": main()
