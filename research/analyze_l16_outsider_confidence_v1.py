#!/usr/bin/env python3
import argparse, csv, gzip, json, math
from collections import defaultdict
from pathlib import Path

CANDS=["outsider_daytrend","outsider_raceshape","outsider_gatecourse","outsider_field","outsider_jockey"]
LOCKED_YEAR=2026

def opent(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def load_truth(path):
    out=defaultdict(dict)
    with opent(path) as f:
        for line in f:
            if not line.strip(): continue
            x=json.loads(line)
            rid=str(x.get("race_id") or ""); hid=str(x.get("horse_id") or "")
            if not rid or not hid: continue
            try: pos=int(float((x.get("target") or {}).get("finish_position")))
            except Exception: continue
            out[rid][hid]=pos
    return out

def load_cons(path):
    out={}
    with opent(path) as f:
        for line in f:
            if not line.strip(): continue
            x=json.loads(line)
            out[str(x["race_id"])]={str(h["horse_id"]):int(h["consensus_rank"]) for h in x["horses"]}
    return out

def load_out(path):
    out=defaultdict(dict)
    with open(path,newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            c=r["candidate"]
            if c not in CANDS: continue
            ids=[x for x in str(r["top3_horse_ids"]).split("|") if x]
            out[str(r["race_id"])][c]={hid:i+1 for i,hid in enumerate(ids)}
    return out

def materialize(a):
    if a.year>=LOCKED_YEAR: raise ValueError("2026 locked")
    truth=load_truth(a.snapshot); cons=load_cons(a.consensus); outs=load_out(a.outsider_csv)
    if set(cons)!=set(truth):
        raise ValueError(f"race coverage mismatch cons={len(cons)} truth={len(truth)}")
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    total=used=missing=0
    with gzip.open(p,"wt",encoding="utf-8",newline="") as gz:
        w=csv.writer(gz)
        w.writerow(["year","race_id","horse_id","king_rank","vote1","vote2","vote3","out_votes","is_podium","is_win"])
        for rid,ranks in cons.items():
            if any(c not in outs[rid] for c in CANDS):
                raise ValueError(f"missing formal5 outsider rows race={rid}")
            for hid,krank in ranks.items():
                total+=1
                pos=truth[rid].get(hid)
                if pos is None:
                    missing+=1
                    continue
                votes=[outs[rid][c].get(hid,0) for c in CANDS]
                v1=sum(v==1 for v in votes); v2=sum(v==2 for v in votes); v3=sum(v==3 for v in votes); vo=sum(v==0 for v in votes)
                if v1+v2+v3+vo!=5: raise ValueError("vote accounting mismatch")
                w.writerow([a.year,rid,hid,krank,v1,v2,v3,vo,int(pos<=3),int(pos==1)])
                used+=1
    print(json.dumps({"contract":"L16_OUTSIDER_CONFIDENCE_ROWS_V1","year":a.year,"races":len(cons),"horses":total,"used":used,"missing_finish":missing,"output":str(p)},ensure_ascii=False))

def read_rows(paths):
    rows=[]
    for path in paths:
        with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "year":int(r["year"]),
                    "race_id":r["race_id"],
                    "king_rank":int(r["king_rank"]),
                    "vote1":int(r["vote1"]),
                    "vote2":int(r["vote2"]),
                    "vote3":int(r["vote3"]),
                    "out_votes":int(r["out_votes"]),
                    "y":int(r["is_podium"]),
                })
    return rows

def baseline_rates(train):
    by=defaultdict(lambda:[0,0])
    for r in train:
        z=by[r["king_rank"]]; z[0]+=r["y"]; z[1]+=1
    # light beta smoothing only matters for sparse deep ranks
    return {k:(p+1)/(n+2) for k,(p,n) in by.items()}

def metrics(y,p):
    import numpy as np
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
    yy=np.asarray(y,dtype=int); pp=np.asarray(p,dtype=float)
    return {
      "n":int(len(yy)),
      "podium_rate":float(yy.mean()),
      "log_loss":float(log_loss(yy,pp,labels=[0,1])),
      "brier":float(brier_score_loss(yy,pp)),
      "auc":float(roc_auc_score(yy,pp)),
    }

def delta_buckets(y,base,model):
    cuts=[(-999,-.10,"<=-10pp"),(-.10,-.05,"-10..-5pp"),(-.05,-.02,"-5..-2pp"),(-.02,.02,"-2..+2pp"),(.02,.05,"+2..+5pp"),(.05,.10,"+5..+10pp"),(.10,999,">=+10pp")]
    out=[]
    for lo,hi,label in cuts:
        idx=[i for i,(b,m) in enumerate(zip(base,model)) if (m-b)>=lo and (m-b)<hi]
        if not idx: continue
        yy=[y[i] for i in idx]; bb=[base[i] for i in idx]; mm=[model[i] for i in idx]
        out.append({
          "bucket":label,"n":len(idx),
          "actual_podium_rate":sum(yy)/len(idx),
          "baseline_mean":sum(bb)/len(idx),
          "model_mean":sum(mm)/len(idx),
          "mean_delta":sum(mm[i]-bb[i] for i in range(len(idx)))/len(idx) if False else sum(model[j]-base[j] for j in idx)/len(idx),
        })
    return out

def evaluate(a):
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    rows=read_rows(a.inputs.split(","))
    years=sorted(set(r["year"] for r in rows))
    if years!=[2022,2023,2024,2025]:
        raise ValueError(f"expected 2022-2025 rows, got {years}")

    folds=[]
    coefficient_rows=[]
    for test_year in [2023,2024,2025]:
        train=[r for r in rows if r["year"]<test_year]
        test=[r for r in rows if r["year"]==test_year]
        br=baseline_rates(train)
        base=[br.get(r["king_rank"], sum(x["y"] for x in train)/len(train)) for r in test]

        Xtr=[{"king_rank":str(r["king_rank"]),"vote1":r["vote1"],"vote2":r["vote2"],"vote3":r["vote3"]} for r in train]
        Xte=[{"king_rank":str(r["king_rank"]),"vote1":r["vote1"],"vote2":r["vote2"],"vote3":r["vote3"]} for r in test]
        from sklearn.feature_extraction import DictVectorizer
        vec=DictVectorizer(sparse=True)
        A=vec.fit_transform(Xtr); B=vec.transform(Xte)
        model=LogisticRegression(C=1.0,penalty="l2",solver="liblinear",max_iter=1000)
        model.fit(A,[r["y"] for r in train])
        pred=model.predict_proba(B)[:,1].tolist()
        y=[r["y"] for r in test]
        bm=metrics(y,base); mm=metrics(y,pred)

        names=vec.get_feature_names_out().tolist()
        coef={n:float(c) for n,c in zip(names,model.coef_[0])}
        fold={
          "train_years":sorted(set(r["year"] for r in train)),
          "test_year":test_year,
          "baseline":bm,
          "confidence_model":mm,
          "improvement":{
            "log_loss":bm["log_loss"]-mm["log_loss"],
            "brier":bm["brier"]-mm["brier"],
            "auc":mm["auc"]-bm["auc"],
          },
          "vote_coefficients":{
            "vote1":coef.get("vote1",0.0),
            "vote2":coef.get("vote2",0.0),
            "vote3":coef.get("vote3",0.0),
          },
          "adjustment_buckets":delta_buckets(y,base,pred),
        }
        folds.append(fold)

    # pooled out-of-sample summary weighted by observations
    total_n=sum(f["baseline"]["n"] for f in folds)
    def weighted(metric, side):
        return sum(f[side][metric]*f[side]["n"] for f in folds)/total_n
    pooled={
      "n":total_n,
      "baseline":{"log_loss":weighted("log_loss","baseline"),"brier":weighted("brier","baseline"),"auc":weighted("auc","baseline")},
      "confidence_model":{"log_loss":weighted("log_loss","confidence_model"),"brier":weighted("brier","confidence_model"),"auc":weighted("auc","confidence_model")},
    }
    pooled["improvement"]={
      "log_loss":pooled["baseline"]["log_loss"]-pooled["confidence_model"]["log_loss"],
      "brier":pooled["baseline"]["brier"]-pooled["confidence_model"]["brier"],
      "auc":pooled["confidence_model"]["auc"]-pooled["baseline"]["auc"],
    }

    out={
      "contract":"L16_OUTSIDER_CONFIDENCE_V1",
      "years":years,
      "locked_year":2026,
      "includes_chimera":False,
      "target":"horse finishes in top3",
      "baseline":"train-year empirical podium probability by exact Seven-King consensus rank with beta(1,1) smoothing",
      "model":"L2-regularized logistic regression; exact king rank categorical + vote1 count + vote2 count + vote3 count; no odds/popularity/payout",
      "interpretation":"king rank supplies the base probability; Outsider vote-position counts only adjust confidence up/down",
      "walk_forward":folds,
      "pooled_oos":pooled,
    }
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"contract":out["contract"],"pooled_oos":pooled,"fold_improvements":[{"test_year":f["test_year"],**f["improvement"],"vote_coefficients":f["vote_coefficients"]} for f in folds]},ensure_ascii=False))

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="cmd",required=True)
    m=sub.add_parser("materialize")
    m.add_argument("--year",type=int,required=True); m.add_argument("--consensus",required=True); m.add_argument("--snapshot",required=True); m.add_argument("--outsider-csv",required=True); m.add_argument("--output",required=True)
    e=sub.add_parser("evaluate")
    e.add_argument("--inputs",required=True,help="comma-separated y2022..y2025 confidence rows csv.gz")
    e.add_argument("--output",required=True)
    a=p.parse_args()
    materialize(a) if a.cmd=="materialize" else evaluate(a)

if __name__=="__main__": main()
