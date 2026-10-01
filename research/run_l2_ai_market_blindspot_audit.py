#!/usr/bin/env python3
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from run_l2_core_v1 import FOLDS,load_dataset,normalized_from_raw
from run_l2_core_v1_ticket_calibration import final_conf_probs
from run_l2_core_v1_confidence_ab import fit_confidence,predict_confidence
from build_l2_bet_kings_dataset_v1 import load_odds_day,final_odds_tuple,finite

DIV_BANDS=(
    ("AI_LE_0.25X",-99.0,-2.0),
    ("0.25_0.50X",-2.0,-1.0),
    ("0.50_0.71X",-1.0,-0.5),
    ("0.71_1.00X",-0.5,0.0),
    ("1.00_1.41X",0.0,0.5),
    ("1.41_2.00X",0.5,1.0),
    ("2.00_4.00X",1.0,2.0),
    ("AI_GE_4.00X",2.0,99.0),
)

def parse_args():
    p=argparse.ArgumentParser(description="Horse-level AI vs market blind-spot audit, strict OOS.")
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def softmax(v):
    v=np.asarray(v,dtype=float); z=np.exp(v-np.max(v)); return z/z.sum()

def win_market_probs(df,root):
    root=Path(root)
    out=np.full(len(df),np.nan,dtype=float)
    odds_out=np.full(len(df),np.nan,dtype=float)
    skipped=[]
    bydate=defaultdict(list)
    for rid,date in df[["race_id","race_date"]].drop_duplicates().itertuples(index=False):
        bydate[str(date)[:10]].append(str(rid))
    row_lookup=defaultdict(dict)
    for i,(rid,no) in enumerate(zip(df["race_id"].astype(str),df["horse_number"].astype(int))):
        row_lookup[rid][int(no)]=i
    for date,rids in bydate.items():
        recs=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",set(rids))
        for rid in rids:
            rec=recs.get(rid)
            if rec is None:
                skipped.append({"race_id":rid,"race_date":date,"reason":"odds_row_missing"})
                continue
            data=((rec.get("odds") or {}).get("1") or {})
            vals={}
            for key,raw in data.items():
                k=str(key)
                if not k.isascii() or not k.isdigit(): continue
                no=int(k)
                tup=final_odds_tuple(raw)
                if not tup: continue
                odd=finite(tup[0])
                if odd is not None and odd>0 and no in row_lookup[rid]:
                    vals[no]=float(odd)
            expected=set(row_lookup[rid])
            if not vals:
                skipped.append({"race_id":rid,"race_date":date,"reason":"win_odds_absent"})
                continue
            if set(vals)!=expected:
                skipped.append({"race_id":rid,"race_date":date,"reason":f"win_odds_incomplete:{len(vals)}/{len(expected)}"})
                continue
            inv={no:1.0/o for no,o in vals.items()}
            total=sum(inv.values())
            if total<=0:
                skipped.append({"race_id":rid,"race_date":date,"reason":"win_odds_invalid_sum"})
                continue
            for no,i in row_lookup[rid].items():
                odds_out[i]=vals[no]
                out[i]=inv[no]/total
    return out,odds_out,skipped

def race_normalize(df,p):
    p=np.asarray(p,dtype=float); out=np.zeros(len(p),dtype=float)
    for _,idx in df.groupby("race_id",sort=False).groups.items():
        ii=np.asarray(list(idx),dtype=int); s=p[ii].sum()
        out[ii]=p[ii]/s if s>0 else 1.0/len(ii)
    return out

def multiclass_race_logloss(df,p):
    eps=1e-15; total=0.0; n=0
    for _,sub in df.groupby("race_id",sort=False):
        y=sub["target_win"].to_numpy(dtype=int)
        if y.sum()!=1: continue
        pp=np.clip(np.asarray(p[sub.index],dtype=float),eps,1.0)
        total+=-math.log(float(pp[np.argmax(y)])); n+=1
    return total/n if n else None

def brier(df,p):
    y=df["target_win"].to_numpy(dtype=float)
    return float(np.mean((np.asarray(p)-y)**2))

def fit_residual(train,pm,pa,q):
    eps=1e-12
    X=np.column_stack([
        np.log(np.clip(pm,eps,1)),
        np.log(np.clip(pa,eps,1)),
        np.log(np.clip(pa,eps,1)/np.clip(pm,eps,1)),
        q,
        train["consensus_rank_pct"].to_numpy(dtype=float),
    ])
    w=1.0/train["field_size"].to_numpy(dtype=float)
    m=LogisticRegression(C=1.0,solver="liblinear",max_iter=1000)
    m.fit(X,train["target_win"].astype(int).to_numpy(),sample_weight=w)
    return m

def predict_model(m,df,pm,pa,q):
    eps=1e-12
    X=np.column_stack([
        np.log(np.clip(pm,eps,1)),
        np.log(np.clip(pa,eps,1)),
        np.log(np.clip(pa,eps,1)/np.clip(pm,eps,1)),
        q,
        df["consensus_rank_pct"].to_numpy(dtype=float),
    ])
    raw=m.decision_function(X)
    return normalized_from_raw(raw,df["race_id"].astype(str).tolist(),1.0)

def band_name(log2r):
    for name,lo,hi in DIV_BANDS:
        if lo<=log2r<hi: return name
    return DIV_BANDS[-1][0]

def aggregate_bands(year,df,pa,pm,q,odds):
    buckets=defaultdict(lambda:{"horses":0,"races":set(),"wins":0,"top3":0,"sum_ai":0.0,"sum_mkt":0.0,"sum_q":0.0,"stake":0.0,"ret":0.0})
    ranks=defaultdict(lambda:{"horses":0,"wins":0,"top3":0,"sum_ai":0.0,"sum_mkt":0.0,"stake":0.0,"ret":0.0})
    for i,r in df.iterrows():
        lr=math.log2(max(pa[i],1e-15)/max(pm[i],1e-15))
        b=band_name(lr)
        z=buckets[b]; z["horses"]+=1; z["races"].add(str(r["race_id"])); z["wins"]+=int(r["target_win"]); z["top3"]+=int(r["target_top3"]) if pd.notna(r["target_top3"]) else 0
        z["sum_ai"]+=pa[i]; z["sum_mkt"]+=pm[i]; z["sum_q"]+=q[i]; z["stake"]+=100.0; z["ret"]+=float(odds[i])*100.0*int(r["target_win"])
        rk=int(r["consensus_rank"]); key=(b,str(rk if rk<=10 else "11+"))
        a=ranks[key]; a["horses"]+=1; a["wins"]+=int(r["target_win"]); a["top3"]+=int(r["target_top3"]) if pd.notna(r["target_top3"]) else 0; a["sum_ai"]+=pa[i]; a["sum_mkt"]+=pm[i]; a["stake"]+=100.0; a["ret"]+=float(odds[i])*100.0*int(r["target_win"])
    rows=[]
    for b,_,_ in DIV_BANDS:
        z=buckets[b]; n=z["horses"]
        rows.append({"test_year":year,"divergence_band":b,"horses":n,"races":len(z["races"]),"mean_ai_win_pct":100*z["sum_ai"]/n if n else None,"mean_market_win_pct":100*z["sum_mkt"]/n if n else None,"actual_win_pct":100*z["wins"]/n if n else None,"actual_top3_pct":100*z["top3"]/n if n else None,"mean_confidence":z["sum_q"]/n if n else None,"actual_over_market_ratio":z["wins"]/z["sum_mkt"] if z["sum_mkt"] else None,"actual_over_ai_ratio":z["wins"]/z["sum_ai"] if z["sum_ai"] else None,"flat_win_roi_pct":100*z["ret"]/z["stake"] if z["stake"] else None})
    rankrows=[]
    for (b,rk),z in sorted(ranks.items()):
        n=z["horses"]
        rankrows.append({"test_year":year,"divergence_band":b,"consensus_rank":rk,"horses":n,"mean_ai_win_pct":100*z["sum_ai"]/n if n else None,"mean_market_win_pct":100*z["sum_mkt"]/n if n else None,"actual_win_pct":100*z["wins"]/n if n else None,"actual_top3_pct":100*z["top3"]/n if n else None,"actual_over_market_ratio":z["wins"]/z["sum_mkt"] if z["sum_mkt"] else None,"flat_win_roi_pct":100*z["ret"]/z["stake"] if z["stake"] else None})
    return rows,rankrows

def write_csv(path,rows):
    if not rows:return
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    a=parse_args()
    manifest,df=load_dataset(a.dataset_dir)
    features=list(manifest["feature_columns"])
    for c in features+["vote2","vote3","top1_votes","consensus_rank","consensus_rank_pct","field_size"]:
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
    df["year"]=df["year"].astype(int); df["race_id"]=df["race_id"].astype(str)
    params=dict(objective="binary",n_estimators=260,learning_rate=0.04,num_leaves=31,min_child_samples=50,subsample=0.9,colsample_bytree=0.9,reg_lambda=1.0,random_state=20261001,n_jobs=2,verbosity=-1)

    yearly={}; bandrows=[]; rankrows=[]; metricrows=[]; skipped_market=[]
    for test_year,train_years in FOLDS:
        train=df[df["year"].isin(train_years) & (df["train_eligible"]==True) & (df["label_available"]==True)].copy()
        test=df[df["year"]==test_year].copy().reset_index(drop=True)
        races=train[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"])
        cut=max(1,int(len(races)*0.8)); fit_ids=set(races.iloc[:cut]["race_id"].astype(str)); cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
        fit=train[train["race_id"].isin(fit_ids)].copy()
        calib=train[train["race_id"].isin(cal_ids)].copy().reset_index(drop=True)
        pa,_=final_conf_probs(train,calib,fit,test,features,params)
        cv,cm=fit_confidence(train); q=predict_confidence(cv,cm,test)
        pm,odds,skipped=win_market_probs(test,a.backfill_root)
        if skipped:
            for row in skipped:
                row["test_year"]=test_year
            skipped_market.extend(skipped)
            bad={row["race_id"] for row in skipped}
            keep=(~test["race_id"].isin(bad)).to_numpy()
            test=test.loc[keep].reset_index(drop=True)
            pa=np.asarray(pa,dtype=float)[keep]
            q=np.asarray(q,dtype=float)[keep]
            pm=np.asarray(pm,dtype=float)[keep]
            odds=np.asarray(odds,dtype=float)[keep]
        if np.any(~np.isfinite(pm)) or np.any(pm<=0) or np.any(~np.isfinite(odds)) or np.any(odds<=0):
            raise SystemExit(f"market filter left invalid rows year={test_year}")
        yearly[test_year]=(test,pa,pm,q,odds)
        br,rr=aggregate_bands(test_year,test,pa,pm,q,odds); bandrows.extend(br); rankrows.extend(rr)
        metricrows.extend([
            {"test_year":test_year,"model":"AI_CONF","race_log_loss":multiclass_race_logloss(test,pa),"brier":brier(test,pa)},
            {"test_year":test_year,"model":"MARKET_FINAL","race_log_loss":multiclass_race_logloss(test,pm),"brier":brier(test,pm)},
        ])
        print("BLINDSPOT_YEAR_READY "+json.dumps({"year":test_year,"horses":len(test)},separators=(",",":")),flush=True)

    # Direct incremental-information test: prior OOS year(s) only.
    residual=[]
    for test_year in (2024,2025):
        prior=[y for y in yearly if y<test_year]
        tr=pd.concat([yearly[y][0] for y in prior],ignore_index=True)
        pa_tr=np.concatenate([yearly[y][1] for y in prior]); pm_tr=np.concatenate([yearly[y][2] for y in prior]); q_tr=np.concatenate([yearly[y][3] for y in prior])
        te,pa,pm,q,_=yearly[test_year]
        market_only=LogisticRegression(C=1.0,solver="liblinear",max_iter=1000)
        Xm=np.log(np.clip(pm_tr,1e-12,1)).reshape(-1,1)
        w=1.0/tr["field_size"].to_numpy(dtype=float)
        market_only.fit(Xm,tr["target_win"].astype(int).to_numpy(),sample_weight=w)
        rawm=market_only.decision_function(np.log(np.clip(pm,1e-12,1)).reshape(-1,1))
        pm_re=normalized_from_raw(rawm,te["race_id"].astype(str).tolist(),1.0)

        aug=fit_residual(tr,pm_tr,pa_tr,q_tr)
        paug=predict_model(aug,te,pm,pa,q)
        residual.extend([
            {"test_year":test_year,"model":"MARKET_RECALIBRATED","race_log_loss":multiclass_race_logloss(te,pm_re),"brier":brier(te,pm_re),"train_years":"|".join(map(str,prior))},
            {"test_year":test_year,"model":"MARKET_PLUS_AI","race_log_loss":multiclass_race_logloss(te,paug),"brier":brier(te,paug),"train_years":"|".join(map(str,prior))},
        ])

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"divergence-bands.csv",bandrows)
    write_csv(out/"divergence-by-rank.csv",rankrows)
    write_csv(out/"raw-model-metrics.csv",metricrows)
    write_csv(out/"incremental-information-test.csv",residual)
    write_csv(out/"skipped-market-races.csv",skipped_market)
    summary={"contract":"L2_AI_VS_MARKET_BLINDSPOT_AUDIT","scope":"all horses in races with complete WIN market coverage; skipped races are explicitly recorded","purpose":"test whether AI contains reproducible information beyond final win-odds market; no buying rule learned","divergence":"log2(AI_CONF_win_probability / normalized_final_win_market_probability)","direct_test":"strict OOS market-recalibrated vs market-plus-AI residual model","market_missing_policy":"exclude whole race from market comparison and write skipped-market-races.csv","skipped_market_races":len(skipped_market),"2026_locked":True,"promotion":False}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_AI_MARKET_BLINDSPOT_AUDIT_READY")

if __name__=="__main__":
    main()
