#!/usr/bin/env python3
import argparse,csv,json,math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,brier_score_loss

from run_l1_market_divergence_audit_v1 import (
    load_dataset,attach_strict_oos_confidence,market_and_finish
)

TEST_FOLDS=((2024,(2023,)),(2025,(2023,2024)))

def parse_args():
    p=argparse.ArgumentParser(description="Strict walk-forward Dissent Gate: market comparison layer outside core L1.")
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def unique(xs):
    out=[]
    for x in xs:
        if x not in out:
            out.append(x)
    return out

def logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return np.log(p/(1-p))

def fit_calibrator(p,y,w):
    m=LogisticRegression(C=100.0,solver="liblinear",max_iter=1000)
    m.fit(logit(p).reshape(-1,1),np.asarray(y,dtype=int),sample_weight=np.asarray(w,dtype=float))
    return m

def apply_calibrator(m,p):
    return m.predict_proba(logit(p).reshape(-1,1))[:,1]

def model_params(seed):
    return dict(
        objective="binary",
        n_estimators=220,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=60,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )

def chronological_fit_cal_split(train):
    races=train[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"])
    cut=max(1,int(len(races)*0.8))
    fit_ids=set(races.iloc[:cut]["race_id"].astype(str))
    cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
    return train[train["race_id"].isin(fit_ids)].copy(),train[train["race_id"].isin(cal_ids)].copy()

def fit_pair(train,market_features,aug_features,seed):
    fit,cal=chronological_fit_cal_split(train)
    if cal.empty or fit.empty:
        raise ValueError("empty fit/cal split")
    wfit=1.0/fit["field_size"].astype(float).to_numpy()
    wcal=1.0/cal["field_size"].astype(float).to_numpy()

    mm0=lgb.LGBMClassifier(**model_params(seed))
    ma0=lgb.LGBMClassifier(**model_params(seed+1))
    mm0.fit(fit[market_features],fit["target_top3"].astype(int),sample_weight=wfit)
    ma0.fit(fit[aug_features],fit["target_top3"].astype(int),sample_weight=wfit)
    pmc=np.asarray(mm0.predict_proba(cal[market_features])[:,1],dtype=float)
    pac=np.asarray(ma0.predict_proba(cal[aug_features])[:,1],dtype=float)
    cm=fit_calibrator(pmc,cal["target_top3"].astype(int),wcal)
    ca=fit_calibrator(pac,cal["target_top3"].astype(int),wcal)
    pmc=apply_calibrator(cm,pmc)
    pac=apply_calibrator(ca,pac)

    mm=lgb.LGBMClassifier(**model_params(seed))
    ma=lgb.LGBMClassifier(**model_params(seed+1))
    wall=1.0/train["field_size"].astype(float).to_numpy()
    mm.fit(train[market_features],train["target_top3"].astype(int),sample_weight=wall)
    ma.fit(train[aug_features],train["target_top3"].astype(int),sample_weight=wall)
    return (mm,cm),(ma,ca),cal,pmc,pac

def predict_pair(pair,df,features):
    m,c=pair
    raw=np.asarray(m.predict_proba(df[features])[:,1],dtype=float)
    return apply_calibrator(c,raw)

def gate_thresholds(cal,pm,pa):
    q=cal.copy().reset_index(drop=True)
    q["delta"]=np.asarray(pa)-np.asarray(pm)
    q["rank_gap"]=q["market_rank"].astype(int)-q["consensus_rank"].astype(int)
    q=q[q["rank_gap"].abs()>=2].copy()
    q["support"]=np.sign(q["rank_gap"].to_numpy(dtype=float))*q["delta"].to_numpy(dtype=float)
    pos=q.loc[q["support"]>0,"support"].to_numpy(dtype=float)
    if len(pos)<50:
        return (0.0,0.0)
    return tuple(np.quantile(pos,[1/3,2/3]))

def gate_tag(score,q1,q2):
    if score<=0: return "REJECT"
    if score<q1: return "LOW"
    if score<q2: return "MID"
    return "HIGH"

def add_market_features(z):
    z=z.copy()
    z["inv_final_win_odds"]=1.0/z["final_win_odds"].astype(float)
    sums=z.groupby("race_id")["inv_final_win_odds"].transform("sum")
    z["market_win_probability"]=z["inv_final_win_odds"]/sums
    z["market_rank_pct"]=z["market_rank"].astype(float)/z["field_size"].astype(float)
    z["log_market_win_probability"]=np.log(np.clip(z["market_win_probability"].to_numpy(dtype=float),1e-12,1.0))
    z["log_final_win_odds"]=np.log(np.clip(z["final_win_odds"].to_numpy(dtype=float),1e-12,None))
    z["rank_gap"]=z["market_rank"].astype(int)-z["consensus_rank"].astype(int)
    z["direction"]=np.where(z["rank_gap"]>=2,"L1_UPGRADE",np.where(z["rank_gap"]<=-2,"L1_DOWNGRADE","NEAR"))
    return z

def add_peer_baseline(test,all_year):
    peers=all_year.groupby("market_rank",dropna=False)["target_top3"].agg(["mean","size"]).reset_index()
    peers=peers.rename(columns={"mean":"market_peer_top3_rate","size":"market_peer_horses"})
    return test.merge(peers,on="market_rank",how="left",validate="many_to_one")

def subset_row(test_year,scope,sub):
    if sub.empty:
        return {"test_year":test_year,"scope":scope,"horses":0,"races":0}
    sign=np.sign(sub["rank_gap"].to_numpy(dtype=float))
    residual=sub["target_top3"].to_numpy(dtype=float)-sub["market_peer_top3_rate"].to_numpy(dtype=float)
    return {
        "test_year":test_year,
        "scope":scope,
        "horses":len(sub),
        "races":sub["race_id"].nunique(),
        "upgrade_horses":int((sub["direction"]=="L1_UPGRADE").sum()),
        "downgrade_horses":int((sub["direction"]=="L1_DOWNGRADE").sum()),
        "mean_support_score":float(sub["dissent_support_score"].mean()) if "dissent_support_score" in sub else None,
        "actual_top3_pct":100*float(sub["target_top3"].mean()),
        "market_peer_top3_pct":100*float(sub["market_peer_top3_rate"].mean()),
        "direction_correct_effect_pp":100*float(np.mean(sign*residual)),
    }

def tier_rows(test_year,test):
    rows=[]
    for direction in ("L1_UPGRADE","L1_DOWNGRADE"):
        d=test[test["direction"]==direction]
        for tag in ("REJECT","LOW","MID","HIGH"):
            z=d[d["dissent_gate_tag"]==tag]
            if z.empty: continue
            r=subset_row(test_year,f"{direction}:{tag}",z)
            r["direction"]=direction
            r["gate_tag"]=tag
            rows.append(r)
    return rows

def metric_rows(test_year,test,pm,pa):
    y=test["target_top3"].astype(int).to_numpy()
    rows=[]
    for name,p in (("MARKET_ONLY",pm),("MARKET_PLUS_L1",pa)):
        rows.append({
            "test_year":test_year,
            "model":name,
            "horses":len(test),
            "log_loss":float(log_loss(y,np.clip(p,1e-12,1-1e-12))),
            "brier":float(brier_score_loss(y,p)),
        })
    return rows

def write_csv(path,rows):
    path=Path(path)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader(); w.writerows(rows)

def main():
    a=parse_args()
    manifest,df=load_dataset(a.dataset_dir)
    if 2026 in set(df["year"]):
        raise SystemExit("2026 sealed")
    aux,skipped=market_and_finish(df,a.backfill_root)
    keep=df.index.intersection(aux.index)
    z=df.loc[keep].copy().join(aux.loc[keep])
    z=z[z["target_top3"].notna()].copy()
    z=attach_strict_oos_confidence(z)
    z=add_market_features(z)

    l1_features=[]
    for c in manifest.get("feature_columns",[]):
        if c in z.columns:
            z[c]=pd.to_numeric(z[c],errors="coerce").fillna(0.0)
            l1_features.append(c)
    z["confidence_score"]=pd.to_numeric(z["confidence_score"],errors="coerce")
    market_features=["field_size","market_rank_pct","log_market_win_probability","log_final_win_odds"]
    aug_features=unique(market_features+l1_features+["confidence_score"])
    for c in market_features:
        z[c]=pd.to_numeric(z[c],errors="coerce").fillna(0.0)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    foldrows=[]; tiers=[]; metrics=[]; importances=[]
    scored_parts=[]

    for fold_i,(test_year,train_years) in enumerate(TEST_FOLDS,1):
        train=z[z["year"].isin(train_years)].copy()
        test=z[z["year"]==test_year].copy().reset_index(drop=True)
        if train["confidence_score"].isna().any() or test["confidence_score"].isna().any():
            raise SystemExit(f"confidence missing fold={test_year}")
        market_pair,aug_pair,cal,pmc,pac=fit_pair(train,market_features,aug_features,20261001+fold_i*10)
        q1,q2=gate_thresholds(cal,pmc,pac)
        pm=predict_pair(market_pair,test,market_features)
        pa=predict_pair(aug_pair,test,aug_features)
        test["market_top3_prob"]=pm
        test["market_plus_l1_top3_prob"]=pa
        test["incremental_top3_prob"]=pa-pm
        sign=np.sign(test["rank_gap"].to_numpy(dtype=float))
        test["dissent_support_score"]=sign*test["incremental_top3_prob"].to_numpy(dtype=float)
        test["dissent_gate_tag"]=[gate_tag(s,q1,q2) if abs(g)>=2 else "NEAR" for s,g in zip(test["dissent_support_score"],test["rank_gap"])]

        year_all=z[z["year"]==test_year].copy()
        test=add_peer_baseline(test,year_all)
        disagree=test[test["rank_gap"].abs()>=2].copy()
        trust=disagree[disagree["dissent_support_score"]>0].copy()
        reject=disagree[disagree["dissent_support_score"]<=0].copy()

        base=subset_row(test_year,"ALL_DISAGREEMENTS",disagree)
        tr=subset_row(test_year,"TRUST",trust)
        rj=subset_row(test_year,"REJECT",reject)
        for rr in (base,tr,rj):
            rr["train_years"]="|".join(map(str,train_years))
            rr["trust_coverage_pct"]=100*len(trust)/len(disagree) if len(disagree) else None
            rr["calibration_low_threshold"]=q1
            rr["calibration_high_threshold"]=q2
            foldrows.append(rr)
        tiers.extend(tier_rows(test_year,disagree))
        metrics.extend(metric_rows(test_year,test,pm,pa))

        booster=aug_pair[0].booster_
        gains=booster.feature_importance(importance_type="gain")
        for name,gain in sorted(zip(aug_features,gains),key=lambda x:-x[1])[:30]:
            importances.append({"test_year":test_year,"feature":name,"gain":float(gain),"is_market_feature":name in market_features})

        keepcols=["year","race_id","horse_id","horse_number","consensus_rank","market_rank","rank_gap","direction","confidence_score","confidence_tag",
                  "final_win_odds","market_top3_prob","market_plus_l1_top3_prob","incremental_top3_prob","dissent_support_score","dissent_gate_tag","target_top3","market_peer_top3_rate"]
        scored_parts.append(test[keepcols])
        print("DISSENT_GATE_FOLD_READY "+json.dumps({
            "test_year":test_year,"train_years":train_years,"disagreements":len(disagree),
            "trust":len(trust),"coverage_pct":100*len(trust)/len(disagree) if len(disagree) else None,
            "all_effect_pp":base.get("direction_correct_effect_pp"),
            "trust_effect_pp":tr.get("direction_correct_effect_pp"),
        },separators=(",",":")),flush=True)

    scored=pd.concat(scored_parts,ignore_index=True)
    write_csv(out/"fold-summary.csv",foldrows)
    write_csv(out/"tier-summary.csv",tiers)
    write_csv(out/"model-metrics.csv",metrics)
    write_csv(out/"feature-importance.csv",importances)
    scored.to_csv(out/"scored-disagreements.csv.gz",index=False,compression="gzip")

    # Pooled headline is evaluation only; the gate itself was fit separately by walk-forward fold.
    pooled=[]
    for year in (2024,2025):
        sy=scored[scored["year"]==year]
        dis=sy[sy["rank_gap"].abs()>=2]
        for scope,sub in (
            ("ALL_DISAGREEMENTS",dis),
            ("TRUST",dis[dis["dissent_support_score"]>0]),
            ("REJECT",dis[dis["dissent_support_score"]<=0]),
        ):
            pooled.append(subset_row(year,scope,sub))
    write_csv(out/"headline.csv",pooled)

    summary={
        "contract":"L1_DISSENT_GATE_V1",
        "architecture":"core L1 remains market-free; Dissent Gate is a post-L1 comparison layer and never reranks L1",
        "folds":[{"test_year":2024,"train_years":[2023]},{"test_year":2025,"train_years":[2023,2024]}],
        "market_model_features":market_features,
        "augmented_model":"market features + frozen L1.7 internal features + strict-OOS podium confidence",
        "gate_score":"sign(market_rank-L1_rank) * (P_top3_market_plus_L1 - P_top3_market_only)",
        "trust_rule":"score > 0; LOW/MID/HIGH are prior-calibration positive-score tertiles, not outcome-optimized thresholds",
        "disagreement_rule":"abs(tie-safe final WIN market rank - L1.7 rank) >= 2",
        "evaluation_only_baseline":"same test-year, same market-rank actual podium rate",
        "skipped_market_races":len(skipped),
        "2026_locked":True,
        "paid_compute":False,
        "promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== FOLD SUMMARY =====")
    print((out/"fold-summary.csv").read_text(encoding="utf-8"))
    print("===== MODEL METRICS =====")
    print((out/"model-metrics.csv").read_text(encoding="utf-8"))
    print("===== TIERS =====")
    print((out/"tier-summary.csv").read_text(encoding="utf-8"))
    print("L1_DISSENT_GATE_V1_READY")

if __name__=="__main__":
    main()
