#!/usr/bin/env python3
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from run_l2_core_v1 import (
    FOLDS, BET_TYPES, THRESHOLDS, EPS, load_dataset, group_indices,
    normalized_from_raw, horse_metrics, choose_temperature, ticket_probability,
    CalAgg, max_drawdown, write_csv
)
from build_l2_bet_kings_dataset_v1 import load_day,load_odds_day,decode_odds,payout_map

EDGE_BANDS=(
    ("NEG",-1e9,0.0),
    ("0_10",0.0,0.10),
    ("10_20",0.10,0.20),
    ("20_50",0.20,0.50),
    ("50_100",0.50,1.00),
    ("100_PLUS",1.00,1e9),
)

def parse_args():
    p=argparse.ArgumentParser(description="L2 CORE V1 BASE vs podium-confidence AB.")
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def conf_feat(row):
    return {
        "king_rank":str(int(row["consensus_rank"])),
        "vote1":int(round(float(row["top1_votes"]))),
        "vote2":int(round(float(row["vote2"]))),
        "vote3":int(round(float(row["vote3"]))),
    }

def fit_confidence(df):
    z=df[df["target_top3"].notna()].copy()
    if z.empty:
        raise ValueError("confidence training rows empty")
    v=DictVectorizer(sparse=True)
    X=v.fit_transform([conf_feat(r) for _,r in z.iterrows()]).tocsr()
    X.indices=X.indices.astype(np.int32,copy=False); X.indptr=X.indptr.astype(np.int32,copy=False)
    m=LogisticRegression(C=1.0,solver="liblinear",max_iter=1000)
    m.fit(X,z["target_top3"].astype(int).to_numpy())
    return v,m

def predict_confidence(v,m,df):
    X=v.transform([conf_feat(r) for _,r in df.iterrows()]).tocsr()
    X.indices=X.indices.astype(np.int32,copy=False); X.indptr=X.indptr.astype(np.int32,copy=False)
    return m.predict_proba(X)[:,1]

def logit_array(p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return np.log(p/(1-p))

def meta_features(raw_margin,q,rank_pct):
    q=np.asarray(q,dtype=float); rp=np.asarray(rank_pct,dtype=float); rm=np.asarray(raw_margin,dtype=float)
    return np.column_stack([rm,logit_array(q),rp,q*rp])

def rank_tag_thresholds(calib,q):
    out={}
    tmp=calib.copy().reset_index(drop=True); tmp["q"]=np.asarray(q,dtype=float)
    allq=tmp["q"].to_numpy()
    global_thr=tuple(np.quantile(allq,[1/3,2/3])) if len(allq) else (0.33,0.67)
    for rank in range(1,11):
        vals=tmp.loc[tmp["consensus_rank"]==rank,"q"].to_numpy()
        out[rank]=tuple(np.quantile(vals,[1/3,2/3])) if len(vals)>=30 else global_thr
    return out

def tag_for(rank,q,thr):
    lo,hi=thr.get(int(rank),next(iter(thr.values())))
    if q<lo: return "LOW"
    if q>=hi: return "HIGH"
    return "MID"

def confidence_diagnostics(test,q,thr,test_year):
    tmp=test.copy().reset_index(drop=True)
    tmp["confidence_score"]=np.asarray(q,dtype=float)
    tmp["confidence_tag"]=[tag_for(r,qv,thr) for r,qv in zip(tmp["consensus_rank"],tmp["confidence_score"])]
    rows=[]
    for rank in range(1,11):
        sub=tmp[tmp["consensus_rank"]==rank]
        for tag in ("LOW","MID","HIGH"):
            z=sub[sub["confidence_tag"]==tag]
            if z.empty: continue
            rows.append({
                "test_year":test_year,"rank_scope":str(rank),"confidence_tag":tag,"horses":len(z),
                "mean_confidence":float(z["confidence_score"].mean()),
                "podium_rate_pct":100*float(z["target_top3"].mean()) if z["target_top3"].notna().any() else None,
                "win_rate_pct":100*float(z["target_win"].mean()) if z["target_win"].notna().any() else None,
            })
    sub=tmp[tmp["consensus_rank"].between(2,6)]
    for tag in ("LOW","MID","HIGH"):
        z=sub[sub["confidence_tag"]==tag]
        if z.empty: continue
        rows.append({
            "test_year":test_year,"rank_scope":"2-6","confidence_tag":tag,"horses":len(z),
            "mean_confidence":float(z["confidence_score"].mean()),
            "podium_rate_pct":100*float(z["target_top3"].mean()) if z["target_top3"].notna().any() else None,
            "win_rate_pct":100*float(z["target_win"].mean()) if z["target_win"].notna().any() else None,
        })
    return rows

def edge_band_name(edge):
    for name,lo,hi in EDGE_BANDS:
        if lo <= edge < hi: return name
    return "100_PLUS"

def evaluate_market_mode(mode,test_year,test_df,pwin,backfill_root):
    tmp=test_df.copy().reset_index(drop=True); tmp["p_win"]=np.asarray(pwin,dtype=float)
    race_prob={}; race_date={}
    for rid,sub in tmp.groupby("race_id",sort=False):
        p={int(n):float(v) for n,v in zip(sub["horse_number"],sub["p_win"])}
        s=sum(p.values()); p={k:v/s for k,v in p.items()}
        race_prob[str(rid)]=p; race_date[str(rid)]=str(sub["race_date"].iloc[0])[:10]
    bydate=defaultdict(list)
    for rid,d in race_date.items(): bydate[d].append(rid)
    cal={b:CalAgg() for b in BET_TYPES}
    strat={(b,t):{"tickets":0,"stake":0.0,"ret":0.0,"hit_tickets":0,"races":0,"hit_races":0,"profits":[]} for b in BET_TYPES for t in THRESHOLDS}
    bands={(b,name):{"n":0,"p":0.0,"y":0,"stake":0.0,"ret":0.0} for b in BET_TYPES for name,_,_ in EDGE_BANDS}
    root=Path(backfill_root)
    for date in sorted(bydate):
        wanted=set(bydate[date])
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        oddsday=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in wanted:
            pack=day.get(rid); oddsrec=oddsday.get(rid)
            if pack is None or oddsrec is None: raise SystemExit(f"missing market row y={test_year} race={rid}")
            p=race_prob[rid]; payouts,present=payout_map(pack); odds=decode_odds(oddsrec)
            raceacc={(b,t):{"tickets":0,"stake":0.0,"ret":0.0,"hit":0} for b in BET_TYPES for t in THRESHOLDS}
            for (bet,nums),odd in odds.items():
                if bet not in BET_TYPES or bet not in present: continue
                ph=ticket_probability(bet,nums,p)
                if ph is None: continue
                ph=min(max(float(ph),0.0),1.0)
                ret=float(payouts.get((bet,nums),0.0)); hit=1 if ret>0 else 0
                cal[bet].add(ph,hit)
                edge=ph*float(odd)-1.0
                bn=edge_band_name(edge); z=bands[(bet,bn)]
                z["n"]+=1; z["p"]+=ph; z["y"]+=hit; z["stake"]+=100.0; z["ret"]+=ret
                for t in THRESHOLDS:
                    if edge>=t:
                        r=raceacc[(bet,t)]; r["tickets"]+=1; r["stake"]+=100.0; r["ret"]+=ret; r["hit"]+=hit
            for key,r in raceacc.items():
                if not r["tickets"]: continue
                g=strat[key]; g["tickets"]+=r["tickets"]; g["stake"]+=r["stake"]; g["ret"]+=r["ret"]; g["hit_tickets"]+=r["hit"]; g["races"]+=1; g["hit_races"]+=int(r["hit"]>0); g["profits"].append(r["ret"]-r["stake"])
    calrows=[]
    for bet in BET_TYPES:
        calrows.append({"mode":mode,"test_year":test_year,"bet_type":bet,**cal[bet].result()})
    srows=[]
    for bet in BET_TYPES:
        for t in THRESHOLDS:
            g=strat[(bet,t)]
            srows.append({
                "mode":mode,"test_year":test_year,"bet_type":bet,"edge_threshold":t,"tickets":g["tickets"],"bought_races":g["races"],
                "hit_tickets":g["hit_tickets"],"hit_races":g["hit_races"],"stake_yen":g["stake"],"return_yen":g["ret"],
                "profit_yen":g["ret"]-g["stake"],"roi_pct":100*g["ret"]/g["stake"] if g["stake"] else None,
                "race_hit_rate_pct":100*g["hit_races"]/g["races"] if g["races"] else None,
                "ticket_hit_rate_pct":100*g["hit_tickets"]/g["tickets"] if g["tickets"] else None,
                "max_drawdown_yen":max_drawdown(g["profits"]),
            })
    brows=[]
    for bet in BET_TYPES:
        for name,_,_ in EDGE_BANDS:
            z=bands[(bet,name)]
            brows.append({
                "mode":mode,"test_year":test_year,"bet_type":bet,"edge_band":name,"tickets":z["n"],
                "mean_predicted_hit_pct":100*z["p"]/z["n"] if z["n"] else None,
                "actual_hit_pct":100*z["y"]/z["n"] if z["n"] else None,
                "calibration_gap_pp":100*(z["p"]-z["y"])/z["n"] if z["n"] else None,
                "roi_pct":100*z["ret"]/z["stake"] if z["stake"] else None,
            })
    return calrows,srows,brows

def main():
    a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L2_CORE_V1": raise SystemExit("wrong contract")
    manifest,df=load_dataset(a.dataset_dir)
    for req in ("vote2","vote3","target_top3"):
        if req not in df.columns: raise SystemExit(f"missing confidence field: {req}")
    features=list(manifest["feature_columns"])
    for c in features+["vote2","vote3","top1_votes","consensus_rank","consensus_rank_pct","field_size"]:
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
    df["year"]=df["year"].astype(int); df["race_id"]=df["race_id"].astype(str)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    foldrows=[]; confrank=[]; ticketcal=[]; market=[]; edgebands=[]
    params=dict(objective="binary",n_estimators=260,learning_rate=0.04,num_leaves=31,min_child_samples=50,subsample=0.9,colsample_bytree=0.9,reg_lambda=1.0,random_state=20261001,n_jobs=2,verbosity=-1)
    for test_year,train_years in FOLDS:
        train=df[df["year"].isin(train_years) & (df["train_eligible"]==True) & (df["label_available"]==True)].copy()
        test=df[df["year"]==test_year].copy().reset_index(drop=True)
        races=train[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"])
        cut=max(1,int(len(races)*0.8)); fit_ids=set(races.iloc[:cut]["race_id"].astype(str)); cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
        fit=train[train["race_id"].isin(fit_ids)].copy()
        calib=train[train["race_id"].isin(cal_ids)].copy().reset_index(drop=True)

        m0=lgb.LGBMClassifier(**params)
        m0.fit(fit[features],fit["target_win"].astype(int),sample_weight=1.0/fit["field_size"].astype(float))
        raw_cal=np.asarray(m0.booster_.predict(calib[features],raw_score=True),dtype=float)
        base_temp,_=choose_temperature(calib,raw_cal)

        cv,cm=fit_confidence(fit)
        qcal=predict_confidence(cv,cm,calib)
        metaX=meta_features(raw_cal,qcal,calib["consensus_rank_pct"].to_numpy(dtype=float))
        meta=LogisticRegression(C=1.0,solver="liblinear",max_iter=1000)
        meta.fit(metaX,calib["target_win"].astype(int).to_numpy(),sample_weight=1.0/calib["field_size"].to_numpy(dtype=float))
        conf_raw_cal=meta.decision_function(metaX)
        conf_temp,_=choose_temperature(calib,conf_raw_cal)
        tagthr=rank_tag_thresholds(calib,qcal)

        base_model=lgb.LGBMClassifier(**params)
        base_model.fit(train[features],train["target_win"].astype(int),sample_weight=1.0/train["field_size"].astype(float))
        raw_test=np.asarray(base_model.booster_.predict(test[features],raw_score=True),dtype=float)
        pbase=normalized_from_raw(raw_test,test["race_id"].tolist(),base_temp)

        fv,fm=fit_confidence(train)
        qtest=predict_confidence(fv,fm,test)
        confX=meta_features(raw_test,qtest,test["consensus_rank_pct"].to_numpy(dtype=float))
        conf_raw=meta.decision_function(confX)
        pconf=normalized_from_raw(conf_raw,test["race_id"].tolist(),conf_temp)

        bm=horse_metrics(test,pbase); cmtr=horse_metrics(test,pconf)
        row={"test_year":test_year,"train_years":"|".join(map(str,train_years)),"fit_races":len(fit_ids),"calibration_races":len(cal_ids),
             "base_temperature":base_temp,"conf_temperature":conf_temp}
        for k,v in bm.items(): row["base_"+k]=v
        for k,v in cmtr.items(): row["conf_"+k]=v
        for k in ("win_log_loss","brier_score","calibration_error_10bin","top1_accuracy_pct","winner_in_top3_pct","official_top3_member_recall_pct","official_top3_exact_set_pct"):
            if k in bm and k in cmtr and bm[k] is not None and cmtr[k] is not None:
                row["delta_"+k]=cmtr[k]-bm[k]
        foldrows.append(row)
        confrank.extend(confidence_diagnostics(test,qtest,tagthr,test_year))

        for mode,p in (("BASE",pbase),("CONF",pconf)):
            c,s,b=evaluate_market_mode(mode,test_year,test,p,a.backfill_root)
            ticketcal.extend(c); market.extend(s); edgebands.extend(b)
        print("L2_CORE_CONF_FOLD_DONE "+json.dumps({"test_year":test_year,"base_top1":bm.get("top1_accuracy_pct"),"conf_top1":cmtr.get("top1_accuracy_pct"),"base_top3":bm.get("winner_in_top3_pct"),"conf_top3":cmtr.get("winner_in_top3_pct")},separators=(",",":")),flush=True)

    write_csv(out/"fold-metrics-ab.csv",foldrows)
    write_csv(out/"confidence-rank-tags.csv",confrank)
    write_csv(out/"ticket-calibration-ab.csv",ticketcal)
    write_csv(out/"market-diagnostic-ab.csv",market)
    write_csv(out/"edge-band-calibration-ab.csv",edgebands)
    summary={
        "contract":"L2_CORE_V1_CONFIDENCE_AB_RESULT",
        "confidence_definition":"P(podium | categorical consensus_rank, vote1, vote2, vote3), strict prior-data fit",
        "confidence_used_as_continuous_signal":True,
        "display_tags":"LOW/MID/HIGH by prior calibration-score tertiles within consensus rank; diagnostic only",
        "base_vs_conf":foldrows,
        "2026_locked":True,
        "promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_CORE_V1_CONFIDENCE_AB_READY")
    print(json.dumps({"folds":len(foldrows),"out_dir":str(out)},separators=(",",":")))

if __name__=="__main__":
    main()
