#!/usr/bin/env python3
import argparse,csv,gzip,json,math
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
    p=argparse.ArgumentParser(description="All-race structural risk for Seven-King #1 and market #1 anchors.")
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

def odds_band(x):
    x=float(x)
    if x<=1.5:return "LE_1.5"
    if x<=2.0:return "GT_1.5_LE_2.0"
    if x<=3.0:return "GT_2.0_LE_3.0"
    return "GT_3.0"

def market_rank_band(x):
    x=int(x)
    if x==1:return "RANK1"
    if x<=3:return "RANK2_3"
    return "RANK4_PLUS"

def load_anchors(path):
    use=["year","race_id","horse_id","horse_number","consensus_rank","market_rank","final_win_odds","target_top3"]
    z=pd.read_csv(path,compression="gzip",usecols=use)
    z["year"]=pd.to_numeric(z["year"],errors="raise").astype(int)
    if 2026 in set(z["year"]): raise SystemExit("2026 sealed")
    z=z[z["year"].isin(YEARS)].copy()
    for c in ("consensus_rank","market_rank","final_win_odds","target_top3"):
        z[c]=pd.to_numeric(z[c],errors="coerce")
    z=z[z["target_top3"].notna()&z["final_win_odds"].notna()&z["market_rank"].notna()&z["consensus_rank"].notna()].copy()
    z["is_king1"]=(z["consensus_rank"]==1).astype(int)
    z["is_market1"]=(z["market_rank"]==1).astype(int)
    z=z[(z["is_king1"]==1)|(z["is_market1"]==1)].copy()
    z["both_anchor"]=((z["is_king1"]==1)&(z["is_market1"]==1)).astype(int)
    z["anchor_type"]=np.select(
        [z["both_anchor"]==1,z["is_king1"]==1],
        ["BOTH","KING_ONLY"],
        default="MARKET_ONLY"
    )
    z["collapse"]=(z["target_top3"].astype(int)==0).astype(int)
    z["log_final_win_odds"]=np.log(np.clip(z["final_win_odds"].astype(float),1e-12,None))
    z["market_rank_band"]=[market_rank_band(x) for x in z["market_rank"]]
    z["odds_band"]=[odds_band(x) for x in z["final_win_odds"]]
    keys=["year","race_id","horse_id"]
    if z.duplicated(keys).any(): raise SystemExit("duplicate anchor horse rows")
    return z

def build_features(anchors,snaps):
    wanted={(int(r.year),str(r.race_id),str(r.horse_id)):r for r in anchors.itertuples(index=False)}
    rows=[]; found=set()
    for year,path in sorted(snaps.items()):
        with open_text(path) as f:
            for line in f:
                if not line.strip(): continue
                rec=json.loads(line)
                key=(year,str(rec.get("race_id") or ""),str(rec.get("horse_id") or ""))
                meta=wanted.get(key)
                if meta is None: continue
                row={
                    "year":year,"race_id":key[1],"horse_id":key[2],
                    "horse_number":meta.horse_number,
                    "consensus_rank":float(meta.consensus_rank),
                    "market_rank":float(meta.market_rank),
                    "final_win_odds":float(meta.final_win_odds),
                    "log_final_win_odds":float(meta.log_final_win_odds),
                    "is_king1":int(meta.is_king1),
                    "is_market1":int(meta.is_market1),
                    "both_anchor":int(meta.both_anchor),
                    "anchor_type":meta.anchor_type,
                    "market_rank_band":meta.market_rank_band,
                    "odds_band":meta.odds_band,
                    "collapse":int(meta.collapse),
                }
                for k,v in (rec.get("features") or {}).items():
                    name=str(k); low=name.lower()
                    if any(t in low for t in FORBIDDEN): continue
                    if not is_structural(name): continue
                    x=safe_num(v)
                    if x is not None: row["f__"+name]=x
                rows.append(row); found.add(key)
    coverage=len(found)/len(wanted) if wanted else 0.0
    if coverage<0.95: raise SystemExit(f"snapshot coverage too low {len(found)}/{len(wanted)}")
    return pd.DataFrame(rows),coverage

def rank_auc(vals,labels):
    x=np.asarray(vals,dtype=float); y=np.asarray(labels,dtype=int)
    mask=np.isfinite(x); x=x[mask]; y=y[mask]
    n1=int((y==1).sum()); n0=int((y==0).sum())
    if n1<5 or n0<10: return 0.5
    ranks=pd.Series(x).rank(method="average").to_numpy()
    u=float(ranks[y==1].sum()-n1*(n1+1)/2)
    return u/(n1*n0)

def select_features(train,k=30):
    rows=[]
    for c in train.columns:
        if not c.startswith("f__"): continue
        vals=pd.to_numeric(train[c],errors="coerce")
        cov=float(vals.notna().mean())
        if cov<0.60: continue
        auc=rank_auc(vals.to_numpy(dtype=float),train["collapse"].to_numpy(dtype=int))
        rows.append((max(auc,1-auc),c,auc,cov))
    rows.sort(reverse=True)
    return rows[:k]

def structural_pipeline():
    return Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("clf",LogisticRegression(C=0.2,class_weight="balanced",solver="liblinear",max_iter=2000,random_state=RNG_SEED)),
    ])

def market_pipeline():
    return Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("clf",LogisticRegression(C=0.2,class_weight="balanced",solver="liblinear",max_iter=2000,random_state=RNG_SEED+1)),
    ])

def metrics(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    return {
        "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
        "average_precision":float(average_precision_score(y,p)) if len(np.unique(y))>1 else None,
        "brier":float(brier_score_loss(y,p)),
        "logloss":float(log_loss(y,np.clip(p,1e-8,1-1e-8),labels=[0,1])),
    }

def group_row(year,label,q):
    n=len(q); c=int(q["collapse"].sum()) if n else 0
    low=q[q["structural_high"]==0]; high=q[q["structural_high"]==1]
    return {
        "test_year":year,"group":label,"horses":n,"races":q["race_id"].nunique() if n else 0,
        "collapse_n":c,"collapse_rate_pct":100*c/n if n else None,
        "structural_high_n":len(high),
        "structural_high_collapse_n":int(high["collapse"].sum()) if len(high) else 0,
        "structural_high_collapse_rate_pct":100*float(high["collapse"].mean()) if len(high) else None,
        "structural_low_collapse_rate_pct":100*float(low["collapse"].mean()) if len(low) else None,
        "high_minus_low_pp":100*(float(high["collapse"].mean())-float(low["collapse"].mean())) if len(high) and len(low) else None,
        "mean_final_win_odds":float(q["final_win_odds"].mean()) if n else None,
    }

def main():
    a=parse_args()
    snaps={}
    for spec in a.snapshot_year:
        y,p=spec.split("=",1); snaps[int(y)]=p
    if set(snaps)!=set(YEARS): raise SystemExit(f"snapshot years drift {sorted(snaps)}")

    anchors=load_anchors(a.market_scored)
    df,coverage=build_features(anchors,snaps)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    pred_rows=[]; summary_rows=[]; metric_rows=[]; feature_rows=[]; market_rows=[]
    for test_year,train_years in FOLDS:
        train=df[df["year"].isin(train_years)].copy()
        test=df[df["year"]==test_year].copy().reset_index(drop=True)

        selected=select_features(train,k=30)
        features=[c for _,c,_,_ in selected]
        if len(features)<10: raise SystemExit(f"too few structural features test={test_year}")
        for rank,(pa,c,raw,cov) in enumerate(selected,1):
            feature_rows.append({
                "test_year":test_year,"train_years":"|".join(map(str,train_years)),
                "rank":rank,"feature":c[3:],"train_predictive_auc":pa,
                "train_auc_larger_means_collapse":raw,"coverage_pct":100*cov,
            })

        # Structural-only risk.
        sp=structural_pipeline()
        sp.fit(train[features].apply(pd.to_numeric,errors="coerce"),train["collapse"].astype(int))
        p_train=sp.predict_proba(train[features].apply(pd.to_numeric,errors="coerce"))[:,1]
        p_test=sp.predict_proba(test[features].apply(pd.to_numeric,errors="coerce"))[:,1]
        threshold=float(np.quantile(p_train,0.80))
        test["structural_risk"]=p_test
        test["structural_high"]=(test["structural_risk"]>=threshold).astype(int)

        # Market + anchor identity baseline, then the same baseline plus structural features.
        market_cols=["log_final_win_odds","market_rank","is_king1","is_market1","both_anchor"]
        mp=market_pipeline()
        mp.fit(train[market_cols],train["collapse"].astype(int))
        market_prob=mp.predict_proba(test[market_cols])[:,1]

        full_cols=market_cols+features
        fp=market_pipeline()
        fp.fit(train[full_cols].apply(pd.to_numeric,errors="coerce"),train["collapse"].astype(int))
        full_prob=fp.predict_proba(test[full_cols].apply(pd.to_numeric,errors="coerce"))[:,1]

        y=test["collapse"].astype(int).to_numpy()
        for name,p in [
            ("STRUCT_ONLY",p_test),
            ("KING_MARKET_BASELINE",market_prob),
            ("KING_MARKET_PLUS_STRUCT",full_prob),
        ]:
            m=metrics(y,p)
            metric_rows.append({"test_year":test_year,"scope":"ALL_ANCHORS","model":name,**m})

        # Same model comparison by anchor role.
        for label,mask in [
            ("KING1",test["is_king1"]==1),
            ("MARKET1",test["is_market1"]==1),
            ("BOTH",test["both_anchor"]==1),
            ("KING_ONLY",test["anchor_type"]=="KING_ONLY"),
            ("MARKET_ONLY",test["anchor_type"]=="MARKET_ONLY"),
        ]:
            ix=np.asarray(mask)
            if ix.sum()<30: continue
            yy=y[ix]
            for name,p in [
                ("STRUCT_ONLY",p_test[ix]),
                ("KING_MARKET_BASELINE",market_prob[ix]),
                ("KING_MARKET_PLUS_STRUCT",full_prob[ix]),
            ]:
                m=metrics(yy,p)
                metric_rows.append({"test_year":test_year,"scope":label,"model":name,**m})

        groups=[
            ("ALL_ANCHORS",test),
            ("KING1",test[test["is_king1"]==1]),
            ("MARKET1",test[test["is_market1"]==1]),
            ("BOTH",test[test["both_anchor"]==1]),
            ("KING_ONLY",test[test["anchor_type"]=="KING_ONLY"]),
            ("MARKET_ONLY",test[test["anchor_type"]=="MARKET_ONLY"]),
        ]
        for label,g in groups:
            summary_rows.append(group_row(test_year,label,g))

        # Market-aware slices: does structural warning still raise collapse within market strength strata?
        king=test[test["is_king1"]==1]
        for band,g in king.groupby("market_rank_band",sort=False):
            r=group_row(test_year,"KING1_MARKET_"+str(band),g)
            market_rows.append(r)
        for band,g in king.groupby("odds_band",sort=False):
            r=group_row(test_year,"KING1_ODDS_"+str(band),g)
            market_rows.append(r)
        market1=test[test["is_market1"]==1]
        for band,g in market1.groupby("odds_band",sort=False):
            r=group_row(test_year,"MARKET1_ODDS_"+str(band),g)
            market_rows.append(r)

        for i,r in test.iterrows():
            pred_rows.append({
                "test_year":test_year,"race_id":r["race_id"],"horse_id":r["horse_id"],
                "anchor_type":r["anchor_type"],"is_king1":int(r["is_king1"]),"is_market1":int(r["is_market1"]),
                "market_rank":float(r["market_rank"]),"final_win_odds":float(r["final_win_odds"]),
                "structural_risk":float(r["structural_risk"]),"structural_high":int(r["structural_high"]),
                "market_baseline_risk":float(market_prob[i]),"market_plus_struct_risk":float(full_prob[i]),
                "collapse":int(r["collapse"]),
            })

    pd.DataFrame(summary_rows).to_csv(out/"anchor-summary.csv",index=False)
    pd.DataFrame(market_rows).to_csv(out/"market-strata.csv",index=False)
    pd.DataFrame(metric_rows).to_csv(out/"model-metrics.csv",index=False)
    pd.DataFrame(feature_rows).to_csv(out/"selected-features.csv",index=False)
    pd.DataFrame(pred_rows).to_csv(out/"oos-predictions.csv.gz",index=False,compression="gzip")

    met=pd.DataFrame(metric_rows)
    primary=met[(met["scope"].isin(["ALL_ANCHORS","KING1","MARKET1","BOTH"])) & (met["model"].isin(["KING_MARKET_BASELINE","KING_MARKET_PLUS_STRUCT"]))]
    primary.to_csv(out/"headline-model-comparison.csv",index=False)

    summ=pd.DataFrame(summary_rows)
    summ[summ["group"].isin(["KING1","MARKET1","BOTH","KING_ONLY","MARKET_ONLY"])].to_csv(out/"headline-groups.csv",index=False)

    decision={}
    for scope in ("ALL_ANCHORS","KING1","MARKET1","BOTH"):
        sy={}
        consistent=True
        for year in (2024,2025):
            q=met[(met["scope"]==scope)&(met["test_year"]==year)]
            b=q[q["model"]=="KING_MARKET_BASELINE"]
            f=q[q["model"]=="KING_MARKET_PLUS_STRUCT"]
            if b.empty or f.empty:
                sy[str(year)]={"available":False}; consistent=False; continue
            br=float(b.iloc[0]["roc_auc"]); fr=float(f.iloc[0]["roc_auc"])
            bb=float(b.iloc[0]["brier"]); fb=float(f.iloc[0]["brier"])
            sy[str(year)]={
                "auc_delta":fr-br,
                "brier_delta_full_minus_baseline":fb-bb,
                "auc_improves":fr>br,
                "brier_improves":fb<bb,
            }
            consistent=consistent and (fr>br) and (fb<bb)
        decision[scope]={"improves_auc_and_brier_both_years":bool(consistent),"years":sy}

    summary={
        "contract":"L1_STRUCTURAL_RISK_KING_MARKET_V1",
        "goal":"Expand structural contradiction to all races by evaluating both Seven-King #1 and market #1 anchors.",
        "anchor_population":"union of consensus_rank==1 and market_rank==1; same horse deduplicated with BOTH flag",
        "structural_inputs":"pre-race FIELD_OPPONENT + PACE_TIME numeric features only; no odds/popularity/payout",
        "market_use":"final WIN odds and market rank are used in the explicit KING_MARKET_BASELINE and KING_MARKET_PLUS_STRUCT comparison, not in STRUCT_ONLY.",
        "folds":[{"test_year":y,"train_years":list(t)} for y,t in FOLDS],
        "structural_high_threshold":"80th percentile of structural model scores on training rows only, then applied unchanged to test year",
        "feature_selection":"training-fold-only top30 univariate predictive AUC, >=60% observed",
        "snapshot_coverage_pct":100*coverage,
        "decision":decision,
        "2026_locked":True,"paid_compute":False,"artifacts_or_cache":False,
        "promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== HEADLINE GROUPS ====="); print((out/"headline-groups.csv").read_text())
    print("===== MODEL COMPARISON ====="); print((out/"headline-model-comparison.csv").read_text())
    print("===== MARKET STRATA ====="); print((out/"market-strata.csv").read_text())
    print("===== SUMMARY ====="); print((out/"summary.json").read_text())
    print("L1_STRUCTURAL_RISK_KING_MARKET_V1_READY")

if __name__=="__main__":
    main()
