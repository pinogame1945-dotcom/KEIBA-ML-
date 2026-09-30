#!/usr/bin/env python3
import argparse,csv,gzip,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

YEARS=(2021,2022,2023,2024,2025)

def opent(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def load_ballots(path):
    pts=defaultdict(lambda:defaultdict(int))
    with opent(path) as fh:
        for row in csv.DictReader(fh):
            rid=str(row["race_id"])
            for k,w in ((1,3),(2,2),(3,1)):
                hid=str(row.get(f"top{k}_horse_id") or "")
                if hid: pts[rid][hid]+=w
    return pts

def load_consensus(path):
    out={}
    with opent(path) as fh:
        for line in fh:
            if not line.strip(): continue
            x=json.loads(line); rid=str(x["race_id"])
            out[rid]={str(h["horse_id"]):int(h["consensus_rank"]) for h in x["horses"]}
    return out

def load_truth(path):
    out=defaultdict(dict)
    with opent(path) as fh:
        for line in fh:
            if not line.strip(): continue
            x=json.loads(line); rid=str(x.get("race_id") or ""); hid=str(x.get("horse_id") or "")
            if not rid or not hid: continue
            try: out[rid][hid]=int(float((x.get("target") or {}).get("finish_position")))
            except (TypeError,ValueError): pass
    return out

def features(rank,score):
    # Rank-specific smooth score correction: base rank intercept + per-rank linear/quadratic score response.
    v=[0.0]*31
    v[0]=score/10.0
    for r in range(1,11):
        if rank==r:
            v[r]=1.0
            v[10+r]=score/10.0
            v[20+r]=(score/10.0)**2
    return v

def build_year(ballots,consensus,snapshot,year):
    pts=load_ballots(ballots); ranks=load_consensus(consensus); truth=load_truth(snapshot)
    rows=[]
    for rid,hs in ranks.items():
        for hid,rank in hs.items():
            if rank>10: continue
            pos=truth.get(rid,{}).get(hid)
            score=int(pts.get(rid,{}).get(hid,0))
            rows.append({"year":year,"race_id":rid,"horse_id":hid,"rank":rank,"score":score,
                         "finish":pos,
                         "y3":(int(pos<=3) if pos is not None else None),
                         "win":(int(pos==1) if pos is not None else None)})
    return rows

def race_metrics(rows,prob_key):
    by=defaultdict(list)
    for x in rows: by[x["race_id"]].append(x)
    m={"races":0,"rank1_top3":0,"winner_at1":0,"winner_at3":0,"winner_at6":0,
       "podium_hits_at3":0,"podium_hits_at6":0,"podium_den":0}
    for rid,rs in by.items():
        # deterministic tie: probability, then original king rank, then horse_id.
        ordered=sorted(rs,key=lambda z:(-z[prob_key],z["rank"],z["horse_id"]))
        base=sorted(rs,key=lambda z:(z["rank"],z["horse_id"]))
        if prob_key=="baseline_score": ordered=base
        winner=next((z["horse_id"] for z in rs if z["win"]==1),None)
        if winner is None: continue
        m["races"]+=1
        m["rank1_top3"]+=ordered[0]["y3"]
        ids=[z["horse_id"] for z in ordered]
        m["winner_at1"]+=int(winner in ids[:1]); m["winner_at3"]+=int(winner in ids[:3]); m["winner_at6"]+=int(winner in ids[:6])
        podium={z["horse_id"] for z in rs if z["y3"]==1}
        m["podium_hits_at3"]+=len(podium.intersection(ids[:3]))
        m["podium_hits_at6"]+=len(podium.intersection(ids[:6]))
        m["podium_den"]+=len(podium)
    d=m["podium_den"] or 1; r=m["races"] or 1
    return {**m,
      "rank1_top3_rate":m["rank1_top3"]/r,
      "winner_capture_at1":m["winner_at1"]/r,
      "winner_capture_at3":m["winner_at3"]/r,
      "winner_capture_at6":m["winner_at6"]/r,
      "podium_recall_at3":m["podium_hits_at3"]/d,
      "podium_recall_at6":m["podium_hits_at6"]/d}

def eval_fold(train,test,year):
    train_labeled=[x for x in train if x["y3"] is not None]
    X=np.asarray([features(x["rank"],x["score"]) for x in train_labeled],dtype=float)
    y=np.asarray([x["y3"] for x in train_labeled],dtype=int)
    model=LogisticRegression(C=0.5,max_iter=1000,solver="lbfgs")
    model.fit(X,y)
    rank_baseline={}
    for r in range(1,11):
        vals=[x["y3"] for x in train_labeled if x["rank"]==r]
        if not vals: raise ValueError(f"missing training rank {r}")
        rank_baseline[r]=sum(vals)/len(vals)
    Xt=np.asarray([features(x["rank"],x["score"]) for x in test],dtype=float)
    p=model.predict_proba(Xt)[:,1]
    scored=[]
    for x,pr in zip(test,p):
        z=dict(x); z["p3"]=float(pr); z["p3_rank_baseline"]=float(rank_baseline[z["rank"]]); z["p3_delta"]=float(pr-rank_baseline[z["rank"]]); z["baseline_score"]=-float(z["rank"]); scored.append(z)
    eval_scored=[z for z in scored if z["y3"] is not None]
    yy=np.asarray([x["y3"] for x in eval_scored],dtype=int)
    pp=np.asarray([x["p3"] for x in eval_scored],dtype=float)
    baseline=race_metrics(eval_scored,"baseline_score")
    ml=race_metrics(eval_scored,"p3")
    by=defaultdict(list)
    for z in eval_scored: by[z["race_id"]].append(z)
    moves=0
    for rs in by.values():
        base_ids=[z["horse_id"] for z in sorted(rs,key=lambda z:(z["rank"],z["horse_id"]))]
        ml_ids=[z["horse_id"] for z in sorted(rs,key=lambda z:(-z["p3"],z["rank"],z["horse_id"]))]
        moves+=int(base_ids!=ml_ids)
    return {
      "eval_year":year,"train_years":sorted({x["year"] for x in train_labeled}),
      "train_rows":len(train_labeled),"test_rows":len(eval_scored),"prediction_rows":len(test),
      "missing_truth_test_rows":len(test)-len(eval_scored),
      "auc":float(roc_auc_score(yy,pp)),"logloss":float(log_loss(yy,pp)),"brier":float(brier_score_loss(yy,pp)),
      "reordered_races":moves,
      "baseline":baseline,"ml":ml,
      "delta":{"rank1_top3_rate":ml["rank1_top3_rate"]-baseline["rank1_top3_rate"],
               "winner_capture_at1":ml["winner_capture_at1"]-baseline["winner_capture_at1"],
               "winner_capture_at3":ml["winner_capture_at3"]-baseline["winner_capture_at3"],
               "winner_capture_at6":ml["winner_capture_at6"]-baseline["winner_capture_at6"],
               "podium_recall_at3":ml["podium_recall_at3"]-baseline["podium_recall_at3"],
               "podium_recall_at6":ml["podium_recall_at6"]-baseline["podium_recall_at6"]},
      "rank_baseline_p3":{str(k):float(v) for k,v in rank_baseline.items()},
      "coefficients":{"intercept":float(model.intercept_[0]),"coef":[float(v) for v in model.coef_[0]]}
    },scored

def aggregate(folds):
    keys=["rank1_top3_rate","winner_capture_at1","winner_capture_at3","winner_capture_at6","podium_recall_at3","podium_recall_at6"]
    rb=sum(f["baseline"]["races"] for f in folds)
    out={"races":rb,"folds":len(folds),"metrics":{}}
    for k in keys:
        b=sum(f["baseline"][k]*f["baseline"]["races"] for f in folds)/rb
        m=sum(f["ml"][k]*f["ml"]["races"] for f in folds)/rb
        out["metrics"][k]={"baseline":b,"ml":m,"delta":m-b}
    out["reordered_races"]=sum(f["reordered_races"] for f in folds)
    return out

def fit_deployment(rows):
    rows=[x for x in rows if x["y3"] is not None]
    X=np.asarray([features(x["rank"],x["score"]) for x in rows],dtype=float)
    y=np.asarray([x["y3"] for x in rows],dtype=int)
    model=LogisticRegression(C=0.5,max_iter=1000,solver="lbfgs")
    model.fit(X,y)
    rb={}
    for r in range(1,11):
        vals=[x["y3"] for x in rows if x["rank"]==r]
        rb[str(r)]=sum(vals)/len(vals)
    return {
      "contract":"L17_KING_OUTSIDER_PODIUM_CALIBRATOR_MODEL_V1",
      "training_years":[2021,2022,2023,2024,2025],
      "feature_spec":"31 features: global score/10; rank1..10 one-hot; rank-specific score/10; rank-specific (score/10)^2",
      "target":"finish_position <= 3",
      "model":"sklearn.linear_model.LogisticRegression",
      "C":0.5,
      "intercept":float(model.intercept_[0]),
      "coef":[float(v) for v in model.coef_[0]],
      "rank_baseline_p3":rb,
      "2026_sealed":True,
      "odds_used":False
    }

def main():
    a=argparse.ArgumentParser()
    for y in YEARS:
        a.add_argument(f"--ballots-{y}",required=True)
        a.add_argument(f"--consensus-{y}",required=True)
        a.add_argument(f"--snapshot-{y}",required=True)
    a.add_argument("--output",required=True)
    args=a.parse_args()
    data={}
    for y in YEARS:
        data[y]=build_year(getattr(args,f"ballots_{y}"),getattr(args,f"consensus_{y}"),getattr(args,f"snapshot_{y}"),y)
    folds=[]; prediction_rows=[]
    for y in (2022,2023,2024,2025):
        train=[x for yy in YEARS if yy<y for x in data[yy]]
        fold,scored=eval_fold(train,data[y],y); folds.append(fold)
        for z in scored:
            prediction_rows.append({k:z[k] for k in ("year","race_id","horse_id","rank","score","finish","y3","p3","p3_rank_baseline","p3_delta")})
    out={"contract":"L17_KING_OUTSIDER_PODIUM_CALIBRATOR_WF_V1",
         "policy":"King ranks 1-10 only; logistic regression with rank-specific linear/quadratic Outsider-score correction; train past years only; rank by predicted podium probability.",
         "target":"finish_position <= 3","years_eval":[2022,2023,2024,2025],"2026_sealed":True,"odds_used":False,
         "folds":folds,"aggregate":aggregate(folds)}
    p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    pred=p.with_name("predictions.csv.gz")
    with gzip.open(pred,"wt",encoding="utf-8",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=["year","race_id","horse_id","rank","score","finish","y3","p3","p3_rank_baseline","p3_delta"])
        w.writeheader(); w.writerows(prediction_rows)
    deploy=fit_deployment([x for y in YEARS for x in data[y]])
    p.with_name("deployment-model.json").write_text(json.dumps(deploy,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L17_PODIUM_CALIBRATOR_OK",json.dumps(out["aggregate"],separators=(",",":")))

if __name__=="__main__":
    main()
