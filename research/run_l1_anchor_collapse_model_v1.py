#!/usr/bin/env python3
import argparse
import gzip
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

YEARS=(2023,2024,2025)
FOLDS=((2024,(2023,)),(2025,(2023,2024)))
TOP_FRACS=(0.10,0.20,0.30)
FORBIDDEN=("odds","popularity","payout","finish","result","target","return","profit","roi")

def parse_args():
    p=argparse.ArgumentParser(description="Strict walk-forward unanimous-anchor collapse model.")
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

def category(name):
    s=name.lower()
    if "style" in s or "position" in s or "corner" in s: return "STYLE_POSITION"
    if "lap" in s or "pace" in s or "time" in s or "last_3f" in s: return "PACE_TIME"
    if "distance" in s or "distx" in s: return "DISTANCE"
    if "weight" in s or "body" in s: return "WEIGHT_BODY"
    if "gate" in s or "frame" in s or "draw" in s: return "DRAW"
    if "jockey" in s or "trainer" in s or "actor" in s: return "ACTOR"
    if "opponent" in s or "field" in s or "relative" in s or "network" in s: return "FIELD_OPPONENT"
    if "surface" in s or "track" in s or "weather" in s or "venue" in s or "course" in s: return "RACE_CONDITION"
    if "recent" in s or "history" in s or "prior" in s or "streak" in s or "margin" in s: return "HISTORY_FRAGILITY"
    return "OTHER"

FAMILIES={
    "FIELD_PACE":{"FIELD_OPPONENT","PACE_TIME"},
    "FIELD_PACE_HISTORY":{"FIELD_OPPONENT","PACE_TIME","HISTORY_FRAGILITY"},
    "STRUCT5":{"FIELD_OPPONENT","PACE_TIME","HISTORY_FRAGILITY","RACE_CONDITION","DISTANCE"},
    "ALL_SAFE_NUMERIC":None,
}

def load_anchor_table(market_path, outsider_path):
    mcols=["year","race_id","horse_id","horse_number","consensus_rank","market_rank","final_win_odds","target_top3"]
    pcols=["year","race_id","horse_id","rank","score","p3","p3_rank_baseline","p3_delta"]
    m=pd.read_csv(market_path,compression="gzip",usecols=mcols)
    p=pd.read_csv(outsider_path,compression="gzip",usecols=pcols)
    m["year"]=pd.to_numeric(m["year"],errors="raise").astype(int)
    p["year"]=pd.to_numeric(p["year"],errors="raise").astype(int)
    if 2026 in set(m["year"]) or 2026 in set(p["year"]): raise SystemExit("2026 sealed")
    m=m[m["year"].isin(YEARS)].copy(); p=p[p["year"].isin(YEARS)].copy()
    for c in ("consensus_rank","market_rank","final_win_odds","target_top3"): m[c]=pd.to_numeric(m[c],errors="coerce")
    for c in ("rank","score","p3","p3_rank_baseline","p3_delta"): p[c]=pd.to_numeric(p[c],errors="coerce")
    keys=["year","race_id","horse_id"]
    if m.duplicated(keys).any() or p.duplicated(keys).any(): raise SystemExit("duplicate source keys")
    z=m.merge(p,on=keys,how="inner",validate="one_to_one")
    z=z[(z["consensus_rank"]==1)&(z["market_rank"]==1)&(z["rank"]==1)&(z["p3_delta"]>0)&z["target_top3"].notna()].copy()
    z["collapse"]=(z["target_top3"].astype(int)==0).astype(int)
    z["train_scope"]=(z["final_win_odds"]<=2.0).astype(int)
    z["stress_1p5"]=(z["final_win_odds"]<=1.5).astype(int)
    return z

def build_feature_rows(anchors, snapshot_paths):
    wanted={(int(r.year),str(r.race_id),str(r.horse_id)):r for r in anchors.itertuples(index=False)}
    rows=[]; found=set()
    for year,path in sorted(snapshot_paths.items()):
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
                    "collapse":int(meta.collapse),
                    "train_scope":int(meta.train_scope),
                    "stress_1p5":int(meta.stress_1p5),
                }
                for k,v in (rec.get("features") or {}).items():
                    low=str(k).lower()
                    if any(t in low for t in FORBIDDEN): continue
                    x=safe_num(v)
                    if x is not None: row["f__"+str(k)]=x
                rows.append(row); found.add(key)
    cov=len(found)/len(wanted) if wanted else 0.0
    if cov<0.90: raise SystemExit(f"snapshot coverage too low {len(found)}/{len(wanted)}")
    return pd.DataFrame(rows),cov

def rank_auc(x,y):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=int)
    mask=np.isfinite(x); x=x[mask]; y=y[mask]
    n1=int((y==1).sum()); n0=int((y==0).sum())
    if n1<5 or n0<10: return 0.5
    ranks=pd.Series(x).rank(method="average").to_numpy()
    u=float(ranks[y==1].sum()-n1*(n1+1)/2)
    return u/(n1*n0)

def choose_features(train, family, k=40):
    cols=[]
    allowed=FAMILIES[family]
    for c in train.columns:
        if not c.startswith("f__"): continue
        name=c[3:]
        if allowed is not None and category(name) not in allowed: continue
        obs=pd.to_numeric(train[c],errors="coerce").notna().mean()
        if obs<0.60: continue
        vals=pd.to_numeric(train[c],errors="coerce").to_numpy(dtype=float)
        auc=rank_auc(vals,train["collapse"].to_numpy(dtype=int))
        cols.append((max(auc,1-auc),c,auc))
    cols.sort(reverse=True)
    return cols[:k]

def ranking_rows(test, risk, label, fold, family, model, stress=False):
    q=test.copy()
    q["risk"]=np.asarray(risk,dtype=float)
    q=q.sort_values(["risk","race_id","horse_id"],ascending=[False,True,True]).reset_index(drop=True)
    total_c=int(q["collapse"].sum()); base=float(q["collapse"].mean()) if len(q) else float("nan")
    rows=[]
    for frac in TOP_FRACS:
        n=max(1,int(math.ceil(len(q)*frac)))
        s=q.iloc[:n]
        caught=int(s["collapse"].sum())
        rate=float(s["collapse"].mean())
        rows.append({
            "test_year":fold,"family":family,"model":model,"scope":label,
            "top_fraction":frac,"anchors":len(q),"collapse_n":total_c,
            "flagged":n,"collapse_caught":caught,
            "recall_collapse_pct":100*caught/total_c if total_c else None,
            "flagged_collapse_rate_pct":100*rate,
            "baseline_collapse_rate_pct":100*base,
            "lift_vs_baseline":rate/base if base>0 else None,
            "stress_1p5":bool(stress),
        })
    return rows

def model_fit_predict(train,test,features,kind,seed):
    Xtr=train[features].apply(pd.to_numeric,errors="coerce")
    Xte=test[features].apply(pd.to_numeric,errors="coerce")
    y=train["collapse"].astype(int).to_numpy()
    if kind=="LOGISTIC":
        model=Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("clf",LogisticRegression(C=0.2,class_weight="balanced",solver="liblinear",max_iter=2000,random_state=seed)),
        ])
        model.fit(Xtr,y)
        return model.predict_proba(Xte)[:,1]
    if kind=="LIGHTGBM":
        model=lgb.LGBMClassifier(
            objective="binary",n_estimators=180,learning_rate=0.035,num_leaves=15,
            min_child_samples=35,subsample=0.9,colsample_bytree=0.8,reg_lambda=2.0,
            random_state=seed,n_jobs=2,verbosity=-1,
        )
        Xtr=Xtr.fillna(Xtr.median(numeric_only=True)).fillna(0.0)
        Xte=Xte.fillna(Xtr.median(numeric_only=True)).fillna(0.0)
        model.fit(Xtr,y)
        return model.predict_proba(Xte)[:,1]
    raise ValueError(kind)

def main():
    a=parse_args()
    snaps={}
    for spec in a.snapshot_year:
        y,p=spec.split("=",1); snaps[int(y)]=p
    if set(snaps)!=set(YEARS): raise SystemExit(f"snapshot years drift {sorted(snaps)}")
    anchors=load_anchor_table(a.market_scored,a.outsider_predictions)
    df,coverage=build_feature_rows(anchors,snaps)
    df=df[df["train_scope"]==1].copy()
    if set(df["year"].unique())!=set(YEARS): raise SystemExit("training scope year coverage drift")

    metrics=[]; feature_rows=[]; pred_rows=[]
    for fi,(test_year,train_years) in enumerate(FOLDS,1):
        train=df[df["year"].isin(train_years)].copy()
        test=df[df["year"]==test_year].copy().reset_index(drop=True)
        if train["collapse"].sum()<50 or test["collapse"].sum()<30: raise SystemExit("collapse sample too small")

        # Non-model baselines: used only for comparison, never as model inputs.
        metrics.extend(ranking_rows(test,test["final_win_odds"].to_numpy(),"ODDS_LE_2",test_year,"BASELINE","ODDS_ONLY"))
        metrics.extend(ranking_rows(test,-test["p3_delta"].to_numpy(),"ODDS_LE_2",test_year,"BASELINE","OUTSIDER_ONLY"))

        for family in FAMILIES:
            selected=choose_features(train,family,k=40)
            features=[c for _,c,_ in selected]
            if len(features)<5: raise SystemExit(f"too few features fold={test_year} family={family} n={len(features)}")
            for rank,(pa,c,raw_auc) in enumerate(selected,1):
                feature_rows.append({
                    "test_year":test_year,"train_years":"|".join(map(str,train_years)),
                    "family":family,"selection_rank":rank,"feature":c[3:],
                    "category":category(c[3:]),"train_predictive_auc":pa,
                    "train_auc_larger_means_collapse":raw_auc,
                })
            for kind in ("LOGISTIC","LIGHTGBM"):
                risk=model_fit_predict(train,test,features,kind,20261004+fi*100+(0 if kind=="LOGISTIC" else 1))
                try:
                    roc=float(roc_auc_score(test["collapse"],risk))
                    ap=float(average_precision_score(test["collapse"],risk))
                except ValueError:
                    roc=float("nan"); ap=float("nan")
                rr=ranking_rows(test,risk,"ODDS_LE_2",test_year,family,kind)
                for x in rr: x.update({"roc_auc":roc,"average_precision":ap,"features":len(features)})
                metrics.extend(rr)

                stress_mask=test["stress_1p5"]==1
                stress=test[stress_mask].copy()
                srisk=np.asarray(risk)[stress_mask.to_numpy()]
                if len(stress) and stress["collapse"].sum()>=5:
                    srr=ranking_rows(stress,srisk,"ODDS_LE_1.5_STRESS",test_year,family,kind,stress=True)
                    for x in srr: x.update({"roc_auc":None,"average_precision":None,"features":len(features)})
                    metrics.extend(srr)

                for r,pred in zip(test.itertuples(index=False),risk):
                    pred_rows.append({
                        "test_year":test_year,"family":family,"model":kind,
                        "race_id":r.race_id,"horse_id":r.horse_id,
                        "final_win_odds":r.final_win_odds,"p3_delta":r.p3_delta,
                        "collapse":r.collapse,"stress_1p5":r.stress_1p5,
                        "collapse_risk":float(pred),
                    })

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    mdf=pd.DataFrame(metrics)
    mdf.to_csv(out/"fold-ranking-metrics.csv",index=False)
    pd.DataFrame(feature_rows).to_csv(out/"selected-features.csv",index=False)
    pd.DataFrame(pred_rows).to_csv(out/"oos-predictions.csv.gz",index=False,compression="gzip")

    primary=mdf[(mdf["scope"]=="ODDS_LE_2")&(mdf["top_fraction"]==0.20)].copy()
    primary.to_csv(out/"primary-top20.csv",index=False)
    summary={
        "contract":"L1_ANCHOR_COLLAPSE_MODEL_V1",
        "universe":"Seven-King rank1 + tie-safe market rank1 + frozen Outsider p3_delta>0 + final WIN odds<=2.0",
        "market_usage":"gate and ODDS_ONLY evaluation baseline only; final odds are not model features",
        "outsider_usage":"agreement gate and OUTSIDER_ONLY evaluation baseline only; p3_delta is not a model feature",
        "folds":[{"test_year":y,"train_years":list(t)} for y,t in FOLDS],
        "families":{k:sorted(v) if v is not None else "ALL_SAFE_NUMERIC" for k,v in FAMILIES.items()},
        "feature_selection":"training-fold-only univariate predictive AUC, top 40, >=60% observed",
        "models":["LOGISTIC","LIGHTGBM"],
        "primary_metric":"Recall and lift among top 20% collapse risk, evaluated strict OOS",
        "stress_test":"same OOS models evaluated on final WIN odds<=1.5 subset; never used as training target filter",
        "anchor_rows":int(len(df)),
        "collapse_rows":int(df["collapse"].sum()),
        "snapshot_coverage_pct":100*coverage,
        "2026_locked":True,
        "paid_compute":False,
        "artifacts_or_cache":False,
        "model_persisted":False,
        "promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== SUMMARY ====="); print((out/"summary.json").read_text())
    print("===== PRIMARY TOP20 ====="); print((out/"primary-top20.csv").read_text())
    print("L1_ANCHOR_COLLAPSE_MODEL_V1_READY")

if __name__=="__main__":
    main()
