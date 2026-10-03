#!/usr/bin/env python3
import argparse,gzip,json,math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

YEARS=(2023,2024,2025)
FOLDS=((2024,(2023,)),(2025,(2023,2024)))
FORBIDDEN=("odds","popularity","payout","finish","result","target","return","profit","roi")
RNG_SEED=20261004

def parse_args():
    p=argparse.ArgumentParser(description="Full-field Structural Risk vote beyond Seven-King + market baseline.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--snapshot-year",action="append",required=True,help="YEAR=PATH")
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def safe_num(v):
    if isinstance(v,bool): return float(v)
    if isinstance(v,(int,float,np.integer,np.floating)):
        x=float(v)
        return x if math.isfinite(x) else None
    return None

def is_structural(name):
    s=name.lower()
    return any(x in s for x in (
        "opponent","field","relative","network",
        "lap","pace","time","last_3f"
    ))

def rank_bucket(x):
    x=int(x)
    if x==1:return "R1"
    if x<=3:return "R2_3"
    if x<=6:return "R4_6"
    if x<=10:return "R7_10"
    return "R11_PLUS"

def load_fullfield(path):
    use=["year","race_id","horse_id","horse_number","consensus_rank","market_rank","final_win_odds","target_top3"]
    z=pd.read_csv(path,compression="gzip",usecols=use)
    z["year"]=pd.to_numeric(z["year"],errors="raise").astype(int)
    if 2026 in set(z["year"]): raise SystemExit("2026 sealed")
    z=z[z["year"].isin(YEARS)].copy()
    for c in ("consensus_rank","market_rank","final_win_odds","target_top3"):
        z[c]=pd.to_numeric(z[c],errors="coerce")
    z=z[z["target_top3"].notna()&z["final_win_odds"].notna()&z["market_rank"].notna()&z["consensus_rank"].notna()].copy()
    keys=["year","race_id","horse_id"]
    if z.duplicated(keys).any(): raise SystemExit("duplicate full-field keys")

    counts=z.groupby(["year","race_id"])["horse_id"].transform("size")
    z["field_size"]=counts.astype(float)
    z["market_rank_pct"]=z["market_rank"]/z["field_size"]
    z["consensus_rank_pct"]=z["consensus_rank"]/z["field_size"]
    z["rank_gap"]=z["market_rank"]-z["consensus_rank"]
    z["rank_gap_pct"]=z["rank_gap"]/z["field_size"]
    z["abs_rank_gap_pct"]=z["rank_gap_pct"].abs()
    z["inv_odds"]=1.0/np.clip(z["final_win_odds"].astype(float),1e-9,None)
    denom=z.groupby(["year","race_id"])["inv_odds"].transform("sum")
    z["market_win_probability"]=z["inv_odds"]/denom
    z["log_market_win_probability"]=np.log(np.clip(z["market_win_probability"],1e-12,1.0))
    z["log_final_win_odds"]=np.log(np.clip(z["final_win_odds"].astype(float),1e-12,None))
    z["collapse"]=(z["target_top3"].astype(int)==0).astype(int)
    z["king_bucket"]=[rank_bucket(x) for x in z["consensus_rank"]]
    z["market_bucket"]=[rank_bucket(x) for x in z["market_rank"]]
    return z

def build_features(rows,snaps):
    wanted={(int(r.year),str(r.race_id),str(r.horse_id)):r for r in rows.itertuples(index=False)}
    out=[]; found=set()
    for year,path in sorted(snaps.items()):
        with open_text(path) as f:
            for line in f:
                if not line.strip(): continue
                rec=json.loads(line)
                key=(year,str(rec.get("race_id") or ""),str(rec.get("horse_id") or ""))
                meta=wanted.get(key)
                if meta is None: continue
                row={
                    "year":year,"race_id":key[1],"horse_id":key[2],"horse_number":meta.horse_number,
                    "field_size":float(meta.field_size),
                    "consensus_rank":float(meta.consensus_rank),"market_rank":float(meta.market_rank),
                    "consensus_rank_pct":float(meta.consensus_rank_pct),"market_rank_pct":float(meta.market_rank_pct),
                    "rank_gap_pct":float(meta.rank_gap_pct),"abs_rank_gap_pct":float(meta.abs_rank_gap_pct),
                    "market_win_probability":float(meta.market_win_probability),
                    "log_market_win_probability":float(meta.log_market_win_probability),
                    "final_win_odds":float(meta.final_win_odds),"log_final_win_odds":float(meta.log_final_win_odds),
                    "king_bucket":meta.king_bucket,"market_bucket":meta.market_bucket,
                    "collapse":int(meta.collapse),
                }
                for k,v in (rec.get("features") or {}).items():
                    name=str(k); low=name.lower()
                    if any(t in low for t in FORBIDDEN): continue
                    if not is_structural(name): continue
                    x=safe_num(v)
                    if x is not None: row["f__"+name]=x
                out.append(row); found.add(key)
    coverage=len(found)/len(wanted) if wanted else 0.0
    if coverage<0.95: raise SystemExit(f"snapshot coverage too low {len(found)}/{len(wanted)}")
    return pd.DataFrame(out),coverage

def rank_auc(vals,labels):
    x=np.asarray(vals,dtype=float); y=np.asarray(labels,dtype=int)
    mask=np.isfinite(x); x=x[mask]; y=y[mask]
    n1=int((y==1).sum()); n0=int((y==0).sum())
    if n1<20 or n0<20:return 0.5
    ranks=pd.Series(x).rank(method="average").to_numpy()
    u=float(ranks[y==1].sum()-n1*(n1+1)/2)
    return u/(n1*n0)

def select_features(train,k=30):
    rows=[]
    for c in train.columns:
        if not c.startswith("f__"): continue
        v=pd.to_numeric(train[c],errors="coerce")
        cov=float(v.notna().mean())
        if cov<0.60: continue
        auc=rank_auc(v.to_numpy(dtype=float),train["collapse"].to_numpy(dtype=int))
        rows.append((max(auc,1-auc),c,auc,cov))
    rows.sort(reverse=True)
    return rows[:k]

def model():
    return Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("clf",LogisticRegression(C=0.15,class_weight="balanced",solver="liblinear",max_iter=2500,random_state=RNG_SEED)),
    ])

def metrics(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    return {
        "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
        "average_precision":float(average_precision_score(y,p)) if len(np.unique(y))>1 else None,
        "brier":float(brier_score_loss(y,p)),
        "logloss":float(log_loss(y,np.clip(p,1e-8,1-1e-8),labels=[0,1])),
    }

def bucket_row(year,axis,bucket,q):
    n=len(q)
    if not n:
        return {"test_year":year,"axis":axis,"bucket":bucket,"horses":0}
    high=q[q["structural_vote_high"]==1]
    low=q[q["structural_vote_high"]==0]
    return {
        "test_year":year,"axis":axis,"bucket":bucket,"horses":n,
        "collapse_n":int(q["collapse"].sum()),"collapse_rate_pct":100*float(q["collapse"].mean()),
        "high_n":len(high),"high_collapse_n":int(high["collapse"].sum()),
        "high_collapse_rate_pct":100*float(high["collapse"].mean()) if len(high) else None,
        "low_collapse_rate_pct":100*float(low["collapse"].mean()) if len(low) else None,
        "high_minus_low_pp":100*(float(high["collapse"].mean())-float(low["collapse"].mean())) if len(high) and len(low) else None,
        "mean_vote_delta_pp":100*float(q["structural_vote_delta"].mean()),
        "mean_abs_vote_delta_pp":100*float(q["structural_vote_delta"].abs().mean()),
    }

def main():
    a=parse_args()
    snaps={}
    for spec in a.snapshot_year:
        y,p=spec.split("=",1); snaps[int(y)]=p
    if set(snaps)!=set(YEARS): raise SystemExit(f"snapshot years drift {sorted(snaps)}")

    raw=load_fullfield(a.market_scored)
    df,coverage=build_features(raw,snaps)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    baseline_cols=[
        "field_size","market_rank_pct","consensus_rank_pct",
        "log_market_win_probability","log_final_win_odds",
        "rank_gap_pct","abs_rank_gap_pct",
    ]
    pred_rows=[]; metric_rows=[]; bucket_rows=[]; feature_rows=[]; threshold_rows=[]

    for test_year,train_years in FOLDS:
        train=df[df["year"].isin(train_years)].copy()
        test=df[df["year"]==test_year].copy().reset_index(drop=True)

        selected=select_features(train,k=30)
        struct_cols=[c for _,c,_,_ in selected]
        if len(struct_cols)<10: raise SystemExit(f"too few structural features test={test_year}")
        for rank,(pa,c,raw_auc,cov) in enumerate(selected,1):
            feature_rows.append({
                "test_year":test_year,"train_years":"|".join(map(str,train_years)),
                "rank":rank,"feature":c[3:],"train_predictive_auc":pa,
                "train_auc_larger_means_collapse":raw_auc,"coverage_pct":100*cov,
            })

        b=model(); f=model()
        b.fit(train[baseline_cols],train["collapse"].astype(int))
        f.fit(train[baseline_cols+struct_cols],train["collapse"].astype(int))

        b_train=b.predict_proba(train[baseline_cols])[:,1]
        f_train=f.predict_proba(train[baseline_cols+struct_cols])[:,1]
        train_delta=f_train-b_train
        threshold=float(np.quantile(train_delta,0.80))

        b_test=b.predict_proba(test[baseline_cols])[:,1]
        f_test=f.predict_proba(test[baseline_cols+struct_cols])[:,1]
        delta=f_test-b_test

        test["baseline_collapse_risk"]=b_test
        test["full_collapse_risk"]=f_test
        test["structural_vote_delta"]=delta
        test["structural_vote_high"]=(delta>=threshold).astype(int)
        threshold_rows.append({
            "test_year":test_year,"train_years":"|".join(map(str,train_years)),
            "train_delta_p80":threshold,
            "train_delta_mean":float(np.mean(train_delta)),
            "test_high_rate_pct":100*float(test["structural_vote_high"].mean()),
        })

        y=test["collapse"].astype(int).to_numpy()
        for name,p in (("KING_MARKET_BASELINE",b_test),("KING_MARKET_PLUS_STRUCT",f_test)):
            metric_rows.append({"test_year":test_year,"scope":"ALL_FULLFIELD","model":name,**metrics(y,p)})

        # Broad rank slices only; one model, no per-rank retraining.
        for axis,col in (("KING","king_bucket"),("MARKET","market_bucket")):
            for bucket in ("R1","R2_3","R4_6","R7_10","R11_PLUS"):
                q=test[test[col]==bucket]
                if len(q): bucket_rows.append(bucket_row(test_year,axis,bucket,q))

        # Agreement/disagreement view without creating extra models.
        both_top3=test[(test["consensus_rank"]<=3)&(test["market_rank"]<=3)]
        mixed_top3=test[((test["consensus_rank"]<=3)&(test["market_rank"]>3))|((test["consensus_rank"]>3)&(test["market_rank"]<=3))]
        neither_top3=test[(test["consensus_rank"]>3)&(test["market_rank"]>3)]
        for label,q in (("BOTH_TOP3",both_top3),("ONE_SIDE_TOP3",mixed_top3),("NEITHER_TOP3",neither_top3)):
            if len(q): bucket_rows.append(bucket_row(test_year,"AGREEMENT",label,q))

        for i,r in test.iterrows():
            pred_rows.append({
                "test_year":test_year,"race_id":r["race_id"],"horse_id":r["horse_id"],
                "horse_number":r["horse_number"],"field_size":r["field_size"],
                "consensus_rank":r["consensus_rank"],"market_rank":r["market_rank"],
                "final_win_odds":r["final_win_odds"],
                "baseline_collapse_risk":float(b_test[i]),
                "full_collapse_risk":float(f_test[i]),
                "structural_vote_delta":float(delta[i]),
                "structural_vote_high":int(delta[i]>=threshold),
                "collapse":int(r["collapse"]),
            })

    pd.DataFrame(metric_rows).to_csv(out/"model-metrics.csv",index=False)
    pd.DataFrame(bucket_rows).to_csv(out/"rank-buckets.csv",index=False)
    pd.DataFrame(feature_rows).to_csv(out/"selected-features.csv",index=False)
    pd.DataFrame(threshold_rows).to_csv(out/"thresholds.csv",index=False)
    pd.DataFrame(pred_rows).to_csv(out/"oos-fullfield-votes.csv.gz",index=False,compression="gzip")

    bm=pd.DataFrame(bucket_rows)
    headline=bm[(bm["axis"].isin(["KING","MARKET"]))&(bm["bucket"].isin(["R1","R2_3","R4_6","R7_10"]))].copy()
    headline.to_csv(out/"headline-rank-buckets.csv",index=False)

    mm=pd.DataFrame(metric_rows)
    comparison=[]
    for year in (2024,2025):
        q=mm[mm["test_year"]==year]
        b=q[q["model"]=="KING_MARKET_BASELINE"].iloc[0]
        f=q[q["model"]=="KING_MARKET_PLUS_STRUCT"].iloc[0]
        comparison.append({
            "test_year":year,
            "auc_delta":float(f["roc_auc"]-b["roc_auc"]),
            "brier_delta_full_minus_baseline":float(f["brier"]-b["brier"]),
            "logloss_delta_full_minus_baseline":float(f["logloss"]-b["logloss"]),
            "auc_improves":bool(f["roc_auc"]>b["roc_auc"]),
            "brier_improves":bool(f["brier"]<b["brier"]),
            "logloss_improves":bool(f["logloss"]<b["logloss"]),
        })
    pd.DataFrame(comparison).to_csv(out/"headline-model-comparison.csv",index=False)

    summary={
        "contract":"L1_STRUCTURAL_RISK_FULLFIELD_V1",
        "goal":"Attach one Structural Risk danger vote to every horse without replacing Seven-King or market rankings.",
        "population":"all evaluable horses in 2023-2025 market-scored OOS rows",
        "baseline":"Seven-King rank + market rank/odds/probability + field size",
        "full_model":"same baseline plus pre-race FIELD_OPPONENT + PACE_TIME structural features",
        "vote_definition":"full collapse probability minus baseline collapse probability; positive means structure makes the horse more dangerous than King+market alone imply",
        "high_vote":"training-fold 80th percentile of vote delta, applied unchanged to test year",
        "folds":[{"test_year":y,"train_years":list(t)} for y,t in FOLDS],
        "rank_reporting":"one shared model; broad King and market rank buckets only, no rank-specific models",
        "snapshot_coverage_pct":100*coverage,
        "2026_locked":True,"paid_compute":False,"artifacts_or_cache":False,"promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== MODEL COMPARISON ====="); print((out/"headline-model-comparison.csv").read_text())
    print("===== RANK BUCKETS ====="); print((out/"headline-rank-buckets.csv").read_text())
    print("===== THRESHOLDS ====="); print((out/"thresholds.csv").read_text())
    print("===== SUMMARY ====="); print((out/"summary.json").read_text())
    print("L1_STRUCTURAL_RISK_FULLFIELD_V1_READY")

if __name__=="__main__":
    main()
