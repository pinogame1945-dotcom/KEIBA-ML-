#!/usr/bin/env python3
import argparse,gzip,json,math,csv
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

YEARS=(2023,2024,2025)
FOLDS=((2024,(2023,)),(2025,(2023,2024)))
FORBIDDEN=("odds","popularity","payout","finish","result","target","return","profit","roi")
RNG_SEED=20261004
BOOTSTRAPS=2000

def parse_args():
    p=argparse.ArgumentParser(description="Consensus Trap V2: strong three-way agreement x structural anomaly.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--outsider-predictions",required=True)
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
    field=("opponent","field","relative","network")
    pace=("lap","pace","time","last_3f")
    return any(x in s for x in field+pace)

def load_cohort(market_path,outsider_path):
    mcols=["year","race_id","horse_id","horse_number","consensus_rank","market_rank","final_win_odds","target_top3"]
    pcols=["year","race_id","horse_id","rank","score","p3","p3_rank_baseline","p3_delta"]
    m=pd.read_csv(market_path,compression="gzip",usecols=mcols)
    p=pd.read_csv(outsider_path,compression="gzip",usecols=pcols)
    m["year"]=pd.to_numeric(m["year"],errors="raise").astype(int)
    p["year"]=pd.to_numeric(p["year"],errors="raise").astype(int)
    if 2026 in set(m["year"]) or 2026 in set(p["year"]): raise SystemExit("2026 sealed")
    m=m[m["year"].isin(YEARS)].copy()
    p=p[p["year"].isin(YEARS)].copy()
    for c in ("consensus_rank","market_rank","final_win_odds","target_top3"):
        m[c]=pd.to_numeric(m[c],errors="coerce")
    for c in ("rank","score","p3","p3_rank_baseline","p3_delta"):
        p[c]=pd.to_numeric(p[c],errors="coerce")
    keys=["year","race_id","horse_id"]
    if m.duplicated(keys).any() or p.duplicated(keys).any(): raise SystemExit("duplicate keys")
    z=m.merge(p,on=keys,how="inner",validate="one_to_one")
    z=z[
        (z["consensus_rank"]==1)&(z["market_rank"]==1)&(z["rank"]==1)&
        (z["p3_delta"]>0)&z["target_top3"].notna()&z["final_win_odds"].notna()
    ].copy()
    z["collapse"]=(z["target_top3"].astype(int)==0).astype(int)

    parts=[]
    for year,g in z.groupby("year",sort=True):
        q=g.copy()
        q["market_strength_pct"]=(-q["final_win_odds"]).rank(method="average",pct=True)
        q["outsider_strength_pct"]=q["p3_delta"].rank(method="average",pct=True)
        q["consensus_pressure"]=0.5*(q["market_strength_pct"]+q["outsider_strength_pct"])
        q["pressure_pct"]=q["consensus_pressure"].rank(method="average",pct=True)
        q["strong_consensus"]=(q["pressure_pct"]>=0.70).astype(int)
        parts.append(q)
    z=pd.concat(parts,ignore_index=True)
    return z

def build_features(cohort,snaps):
    wanted={(int(r.year),str(r.race_id),str(r.horse_id)):r for r in cohort.itertuples(index=False)}
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
                    "final_win_odds":float(meta.final_win_odds),
                    "p3_delta":float(meta.p3_delta),
                    "consensus_pressure":float(meta.consensus_pressure),
                    "pressure_pct":float(meta.pressure_pct),
                    "strong_consensus":int(meta.strong_consensus),
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
    if coverage<0.90: raise SystemExit(f"snapshot coverage too low {len(found)}/{len(wanted)}")
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
        coverage=float(vals.notna().mean())
        if coverage<0.60: continue
        auc=rank_auc(vals.to_numpy(dtype=float),train["collapse"].to_numpy(dtype=int))
        rows.append((max(auc,1-auc),c,auc,coverage))
    rows.sort(reverse=True)
    return rows[:k]

def fit_predict(train,test,features,kind,seed):
    Xtr=train[features].apply(pd.to_numeric,errors="coerce")
    Xte=test[features].apply(pd.to_numeric,errors="coerce")
    y=train["collapse"].astype(int).to_numpy()
    if kind=="LOGISTIC":
        m=Pipeline([
            ("imp",SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("clf",LogisticRegression(C=0.2,class_weight="balanced",solver="liblinear",max_iter=2000,random_state=seed)),
        ])
        m.fit(Xtr,y)
        return m.predict_proba(Xte)[:,1]
    if kind=="LIGHTGBM":
        med=Xtr.median(numeric_only=True)
        Xtr=Xtr.fillna(med).fillna(0.0); Xte=Xte.fillna(med).fillna(0.0)
        m=lgb.LGBMClassifier(
            objective="binary",n_estimators=180,learning_rate=0.035,num_leaves=15,
            min_child_samples=35,subsample=0.9,colsample_bytree=0.8,reg_lambda=2.0,
            random_state=seed,n_jobs=2,verbosity=-1,
        )
        m.fit(Xtr,y)
        return m.predict_proba(Xte)[:,1]
    raise ValueError(kind)

def group_row(year,model,label,q,base_rate=None):
    n=len(q); c=int(q["collapse"].sum()) if n else 0
    rate=float(q["collapse"].mean()) if n else None
    return {
        "test_year":year,"model":model,"group":label,
        "horses":n,"collapse_n":c,
        "collapse_rate_pct":100*rate if rate is not None else None,
        "lift_vs_high_consensus":rate/base_rate if rate is not None and base_rate and base_rate>0 else None,
        "mean_final_win_odds":float(q["final_win_odds"].mean()) if n else None,
        "mean_consensus_pressure":float(q["consensus_pressure"].mean()) if n else None,
        "mean_structural_risk":float(q["structural_risk"].mean()) if n else None,
    }

def exact_odds_matched_effect(q):
    rows=[]
    q=q.copy()
    q["odds_key"]=q["final_win_odds"].round(1)
    for odds,g in q.groupby("odds_key",sort=True):
        a=g[g["structural_high20"]==1]
        b=g[g["structural_high20"]==0]
        if len(a)<2 or len(b)<2: continue
        rows.append((min(len(a),len(b)),float(a["collapse"].mean()-b["collapse"].mean())))
    if not rows: return None,0
    w=np.asarray([x[0] for x in rows],dtype=float)
    d=np.asarray([x[1] for x in rows],dtype=float)
    return float(np.average(d,weights=w)),len(rows)

def bootstrap_exact_odds(q):
    q=q.copy()
    q["race_id"]=q["race_id"].astype(str)
    races=q["race_id"].unique()
    if len(races)<30: return {}
    rng=np.random.default_rng(RNG_SEED)
    vals=[]
    for _ in range(BOOTSTRAPS):
        ids=rng.choice(races,size=len(races),replace=True)
        parts=[]
        for i,rid in enumerate(ids):
            g=q[q["race_id"]==rid].copy()
            if g.empty: continue
            g["race_id"]=g["race_id"]+"#"+str(i)
            parts.append(g)
        if not parts: continue
        b=pd.concat(parts,ignore_index=True)
        eff,_=exact_odds_matched_effect(b)
        if eff is not None and math.isfinite(eff): vals.append(eff)
    if not vals: return {}
    a=np.asarray(vals,dtype=float)
    return {
        "bootstrap_valid":len(a),
        "ci95_low_pp":100*float(np.quantile(a,0.025)),
        "ci95_high_pp":100*float(np.quantile(a,0.975)),
        "p_effect_gt_0":float(np.mean(a>0)),
    }

def main():
    a=parse_args()
    snaps={}
    for spec in a.snapshot_year:
        y,p=spec.split("=",1); snaps[int(y)]=p
    if set(snaps)!=set(YEARS): raise SystemExit(f"snapshot years drift {sorted(snaps)}")

    cohort=load_cohort(a.market_scored,a.outsider_predictions)
    df,coverage=build_features(cohort,snaps)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    summary_rows=[]; feature_rows=[]; predictions=[]; matched=[]
    for fi,(test_year,train_years) in enumerate(FOLDS,1):
        train=df[df["year"].isin(train_years)].copy()
        test=df[df["year"]==test_year].copy().reset_index(drop=True)
        selected=select_features(train,k=30)
        features=[c for _,c,_,_ in selected]
        if len(features)<10: raise SystemExit(f"too few structural features fold={test_year}")
        for rank,(pa,c,raw_auc,cov) in enumerate(selected,1):
            feature_rows.append({
                "test_year":test_year,"train_years":"|".join(map(str,train_years)),
                "selection_rank":rank,"feature":c[3:],
                "train_predictive_auc":pa,"train_auc_larger_means_collapse":raw_auc,
                "coverage_pct":100*cov,
            })

        for kind in ("LOGISTIC","LIGHTGBM"):
            risk=fit_predict(train,test,features,kind,RNG_SEED+fi*100+(1 if kind=="LIGHTGBM" else 0))
            q=test.copy()
            q["structural_risk"]=risk
            q["structural_risk_pct"]=pd.Series(risk).rank(method="average",pct=True).to_numpy()
            q["structural_high20"]=(q["structural_risk_pct"]>=0.80).astype(int)
            q["structural_high30"]=(q["structural_risk_pct"]>=0.70).astype(int)

            try:
                roc=float(roc_auc_score(q["collapse"],risk))
                ap=float(average_precision_score(q["collapse"],risk))
            except ValueError:
                roc=float("nan"); ap=float("nan")

            high=q[q["strong_consensus"]==1].copy()
            high_base=float(high["collapse"].mean()) if len(high) else None
            groups=[
                ("ALL_TRIPLE_AGREE",q),
                ("STRONG_CONSENSUS",high),
                ("STRUCTURAL_HIGH20",q[q["structural_high20"]==1]),
                ("STRONG_X_STRUCT_HIGH20",q[(q["strong_consensus"]==1)&(q["structural_high20"]==1)]),
                ("STRONG_X_STRUCT_NOT_HIGH20",q[(q["strong_consensus"]==1)&(q["structural_high20"]==0)]),
                ("STRONG_X_STRUCT_HIGH30",q[(q["strong_consensus"]==1)&(q["structural_high30"]==1)]),
                ("ODDS_LE_1.5_STRONG_X_STRUCT_HIGH20",q[(q["strong_consensus"]==1)&(q["structural_high20"]==1)&(q["final_win_odds"]<=1.5)]),
            ]
            for label,g in groups:
                r=group_row(test_year,kind,label,g,high_base)
                r["roc_auc"]=roc; r["average_precision"]=ap; r["features"]=len(features)
                summary_rows.append(r)

            hx=q[q["strong_consensus"]==1].copy()
            eff,cells=exact_odds_matched_effect(hx)
            boot=bootstrap_exact_odds(hx)
            matched.append({
                "test_year":test_year,"model":kind,
                "high_consensus_horses":len(hx),
                "matched_cells":cells,
                "exact_odds_struct_high20_minus_rest_pp":100*eff if eff is not None else None,
                **boot,
            })

            for r in q.itertuples(index=False):
                predictions.append({
                    "test_year":test_year,"model":kind,
                    "race_id":r.race_id,"horse_id":r.horse_id,
                    "final_win_odds":r.final_win_odds,"p3_delta":r.p3_delta,
                    "consensus_pressure":r.consensus_pressure,"pressure_pct":r.pressure_pct,
                    "strong_consensus":r.strong_consensus,
                    "structural_risk":r.structural_risk,"structural_risk_pct":r.structural_risk_pct,
                    "structural_high20":r.structural_high20,
                    "collapse":r.collapse,
                })

    pd.DataFrame(summary_rows).to_csv(out/"interaction-summary.csv",index=False)
    pd.DataFrame(feature_rows).to_csv(out/"selected-structural-features.csv",index=False)
    pd.DataFrame(matched).to_csv(out/"exact-odds-matched.csv",index=False)
    pd.DataFrame(predictions).to_csv(out/"oos-predictions.csv.gz",index=False,compression="gzip")

    s=pd.DataFrame(summary_rows)
    headline=s[s["group"].isin(["STRONG_CONSENSUS","STRONG_X_STRUCT_HIGH20","STRONG_X_STRUCT_NOT_HIGH20"])].copy()
    headline.to_csv(out/"headline.csv",index=False)

    decision={}
    for kind in ("LOGISTIC","LIGHTGBM"):
        q=headline[headline["model"]==kind]
        by_year={}
        pass_all=True
        for y in (2024,2025):
            a0=q[(q["test_year"]==y)&(q["group"]=="STRONG_CONSENSUS")]
            a1=q[(q["test_year"]==y)&(q["group"]=="STRONG_X_STRUCT_HIGH20")]
            if a0.empty or a1.empty:
                ok=False; delta=None
            else:
                delta=float(a1.iloc[0]["collapse_rate_pct"]-a0.iloc[0]["collapse_rate_pct"])
                ok=delta>0
            by_year[str(y)]={"delta_pp":delta,"trap_direction_positive":ok}
            pass_all=pass_all and ok
        decision[kind]={"positive_both_oos_years":bool(pass_all),"years":by_year}

    summary={
        "contract":"L1_CONSENSUS_TRAP_V2",
        "hypothesis":"Three-way agreement is normally safe, but a structural anomaly in field/opponent and pace/time space may identify a hidden collapse pocket.",
        "cohort":"Seven-King rank1 + tie-safe market rank1 + frozen Outsider rank1 + p3_delta>0",
        "strong_consensus":"top 30% within-year percentile of 50% market strength + 50% Outsider support strength; outcome-free",
        "structural_model_inputs":"FIELD_OPPONENT and PACE_TIME numeric pre-race snapshot features only",
        "structural_high20":"top 20% OOS structural risk within test year; outcome-free percentile after prediction",
        "folds":[{"test_year":y,"train_years":list(t)} for y,t in FOLDS],
        "feature_selection":"training-fold-only top 30 univariate predictive AUC, >=60% coverage",
        "models":["LOGISTIC","LIGHTGBM"],
        "primary_test":"Within strong-consensus horses, compare structural-high20 vs strong-consensus baseline; require positive direction in both 2024 and 2025.",
        "secondary_test":"Exact one-decimal WIN-odds matched structural-high20 vs rest inside strong-consensus cohort with race-cluster bootstrap.",
        "snapshot_coverage_pct":100*coverage,
        "decision":decision,
        "2026_locked":True,"paid_compute":False,"artifacts_or_cache":False,
        "promotion":False,"causal_claim":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== SUMMARY ====="); print((out/"summary.json").read_text())
    print("===== HEADLINE ====="); print((out/"headline.csv").read_text())
    print("===== EXACT ODDS MATCHED ====="); print((out/"exact-odds-matched.csv").read_text())
    print("L1_CONSENSUS_TRAP_V2_READY")

if __name__=="__main__":
    main()
