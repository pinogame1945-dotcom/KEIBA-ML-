#!/usr/bin/env python3
import json, math, time
from pathlib import Path

import numpy as np
import pandas as pd

SEED=20261004
BOOT=3000
LN2=math.log(2.0)

ROUTE_PATH=Path("research-results/l2-trio-engine-router-v2-structrisk/run-37155641216/race-routes.csv.gz")
OUTSIDER_PATH=Path("research-results/l2-trio-split-residual-v4/run-37146919444/race-diagnostics.csv.gz")
THRESH_PATH=Path("research-results/l2-trio-engine-router-v2-structrisk/run-37155641216/struct-risk-thresholds.csv")
OUT=Path("research-results/l2-structrisk-outsider-failure-audit-v1")

SIGNALS={
    "KING1_POS":("sr_king1_delta","positive_delta_p80"),
    "KING1_ABS":("sr_king1_abs_delta","absolute_delta_p80"),
    "MARKET1_POS":("sr_market1_delta","positive_delta_p80"),
    "MARKET1_ABS":("sr_market1_abs_delta","absolute_delta_p80"),
    "KING_TOP3_ABS_MAX":("sr_king_top3_abs_max","absolute_delta_p80"),
    "MARKET_TOP3_ABS_MAX":("sr_market_top3_abs_max","absolute_delta_p80"),
}

def boot_diff(a,b,seed):
    rng=np.random.default_rng(seed)
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    vals=np.empty(BOOT,dtype=float)
    for i in range(BOOT):
        vals[i]=rng.choice(a,len(a),replace=True).mean()-rng.choice(b,len(b),replace=True).mean()
    return float(vals.mean()),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

def boot_prop_diff(a,b,seed):
    return boot_diff(np.asarray(a,dtype=float),np.asarray(b,dtype=float),seed)

def rankdata(x):
    return pd.Series(np.asarray(x,dtype=float)).rank(method="average").to_numpy(dtype=float)

def spearman(x,y):
    rx,ry=rankdata(x),rankdata(y)
    if np.std(rx)==0 or np.std(ry)==0: return float("nan")
    return float(np.corrcoef(rx,ry)[0,1])

def summarize_group(g,flag_col):
    hi=g[g[flag_col]]
    lo=g[~g[flag_col]]
    if len(hi)==0 or len(lo)==0: return None
    d_mean,d_lo,d_hi=boot_diff(
        hi["outsider_logloss_delta_vs_market"],
        lo["outsider_logloss_delta_vs_market"],
        SEED+len(g)+len(hi)
    )
    hbad=(hi["outsider_logloss_delta_vs_market"]>0).astype(float)
    lbad=(lo["outsider_logloss_delta_vs_market"]>0).astype(float)
    p_mean,p_lo,p_hi=boot_prop_diff(hbad,lbad,SEED+100000+len(g)+len(hi))
    hsev=(hi["outsider_logloss_delta_vs_market"]>LN2).astype(float)
    lsev=(lo["outsider_logloss_delta_vs_market"]>LN2).astype(float)
    s_mean,s_lo,s_hi=boot_prop_diff(hsev,lsev,SEED+200000+len(g)+len(hi))
    return {
        "races":int(len(g)),
        "high_races":int(len(hi)),
        "high_share_pct":100*len(hi)/len(g),
        "mean_delta_high":float(hi["outsider_logloss_delta_vs_market"].mean()),
        "mean_delta_low":float(lo["outsider_logloss_delta_vs_market"].mean()),
        "high_minus_low_mean_delta":float(hi["outsider_logloss_delta_vs_market"].mean()-lo["outsider_logloss_delta_vs_market"].mean()),
        "delta_diff_boot_mean":d_mean,"delta_diff_ci_low":d_lo,"delta_diff_ci_high":d_hi,
        "worse_than_market_high_pct":100*float(hbad.mean()),
        "worse_than_market_low_pct":100*float(lbad.mean()),
        "worse_rate_diff_pp":100*float(hbad.mean()-lbad.mean()),
        "worse_rate_diff_ci_low_pp":100*p_lo,"worse_rate_diff_ci_high_pp":100*p_hi,
        "severe_underweight_high_pct":100*float(hsev.mean()),
        "severe_underweight_low_pct":100*float(lsev.mean()),
        "severe_rate_diff_pp":100*float(hsev.mean()-lsev.mean()),
        "severe_rate_diff_ci_low_pp":100*s_lo,"severe_rate_diff_ci_high_pp":100*s_hi,
        "mean_outsider_true_p_high":float(hi["outsider_true_p"].mean()),
        "mean_outsider_true_p_low":float(lo["outsider_true_p"].mean()),
        "mean_market_true_p_high":float(hi["market_true_p"].mean()),
        "mean_market_true_p_low":float(lo["market_true_p"].mean()),
    }

def main():
    t0=time.time()
    for p in [ROUTE_PATH,OUTSIDER_PATH,THRESH_PATH]:
        if not p.exists(): raise SystemExit(f"missing archived input: {p}")
    routes=pd.read_csv(ROUTE_PATH,dtype={"race_id":str})
    outsider=pd.read_csv(OUTSIDER_PATH,dtype={"race_id":str})
    thr=pd.read_csv(THRESH_PATH)
    if 2026 in set(routes["year"]) or 2026 in set(outsider["year"]): raise SystemExit("2026 sealed")
    keep=[
        "year","race_id","race_date",
        "sr_king1_delta","sr_king1_abs_delta","sr_market1_delta","sr_market1_abs_delta",
        "sr_king_top3_abs_max","sr_market_top3_abs_max",
    ]
    routes=routes[keep].copy()
    df=outsider.merge(routes,on=["year","race_id"],how="inner",validate="one_to_one",suffixes=("","_sr"))
    if len(df)<10000: raise SystemExit(f"unexpected join loss rows={len(df)}")
    if df[["outsider_logloss_delta_vs_market","market_true_p","outsider_true_p"]].isna().any().any():
        raise SystemExit("missing outsider diagnostics")
    tmap={int(r.test_year):r for r in thr.itertuples(index=False)}

    rows=[]; corr=[]; deciles=[]
    for name,(col,tcol) in SIGNALS.items():
        flag=f"flag_{name}"
        vals=[]
        for r in df.itertuples(index=False):
            tr=tmap[int(r.year)]
            vals.append(float(getattr(r,col))>=float(getattr(tr,tcol)))
        df[flag]=np.asarray(vals,dtype=bool)
        for year in [2023,2024,2025,"POOLED"]:
            g=df if year=="POOLED" else df[df["year"]==year]
            s=summarize_group(g,flag)
            if s is not None:
                rows.append({"signal":name,"scope":str(year),**s})
            corr.append({
                "signal":name,"scope":str(year),"races":int(len(g)),
                "spearman_risk_vs_outsider_delta":spearman(g[col],g["outsider_logloss_delta_vs_market"])
            })
            # Descriptive deciles only; not a deployable threshold.
            q=pd.qcut(g[col].rank(method="first"),10,labels=False,duplicates="drop")+1
            temp=g.assign(decile=q)
            for d,h in temp.groupby("decile",sort=True):
                deciles.append({
                    "signal":name,"scope":str(year),"decile":int(d),"races":int(len(h)),
                    "mean_risk":float(h[col].mean()),
                    "mean_outsider_delta_vs_market":float(h["outsider_logloss_delta_vs_market"].mean()),
                    "worse_than_market_pct":100*float((h["outsider_logloss_delta_vs_market"]>0).mean()),
                    "severe_underweight_pct":100*float((h["outsider_logloss_delta_vs_market"]>LN2).mean()),
                })

    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT/"high-risk-separation.csv",index=False)
    pd.DataFrame(corr).to_csv(OUT/"correlations.csv",index=False)
    pd.DataFrame(deciles).to_csv(OUT/"deciles.csv",index=False)

    pooled=pd.DataFrame(rows)
    pooled=pooled[pooled["scope"]=="POOLED"].copy()
    pooled["strong_mean_separation"]=(pooled["delta_diff_ci_low"]>0)
    pooled["strong_worse_rate_separation"]=(pooled["worse_rate_diff_ci_low_pp"]>0)
    pooled["strong_severe_rate_separation"]=(pooled["severe_rate_diff_ci_low_pp"]>0)
    pooled["all_three_strong"]=pooled[
        ["strong_mean_separation","strong_worse_rate_separation","strong_severe_rate_separation"]
    ].all(axis=1)
    best=pooled.sort_values(
        ["all_three_strong","high_minus_low_mean_delta","worse_rate_diff_pp","severe_rate_diff_pp"],
        ascending=[False,False,False,False]
    ).iloc[0].to_dict()

    year_consistency=[]
    for name in SIGNALS:
        yy=pd.DataFrame(rows)
        yy=yy[(yy["signal"]==name)&(yy["scope"].isin(["2023","2024","2025"]))]
        year_consistency.append({
            "signal":name,
            "years_high_group_worse_mean":int((yy["high_minus_low_mean_delta"]>0).sum()),
            "years_high_group_more_often_worse_than_market":int((yy["worse_rate_diff_pp"]>0).sum()),
            "years_high_group_more_severe_underweight":int((yy["severe_rate_diff_pp"]>0).sum()),
        })
    pd.DataFrame(year_consistency).to_csv(OUT/"year-consistency.csv",index=False)

    summary={
        "contract":"L2_STRUCTRISK_OUTSIDER_FAILURE_AUDIT_V1_RESULT",
        "question":"Does Structural Risk separate races where the current MARKET+OUTSIDER champion fails badly?",
        "inputs":{
            "structural_risk_race_rows":int(len(routes)),
            "outsider_diagnostic_rows":int(len(outsider)),
            "joined_races":int(len(df)),
            "years":sorted(map(int,df["year"].unique())),
        },
        "failure_definitions":{
            "worse_than_market":"outsider_logloss_delta_vs_market > 0",
            "severe_underweight":"outsider gives true TRIO less than half the pure market probability; equivalent delta > ln(2)",
        },
        "high_risk_definition":"Per-year horse-level training-fold p80 thresholds already frozen by Structural Risk run 37155641216; no outcome-tuned threshold in this audit.",
        "signals":list(SIGNALS),
        "best_pooled_signal":best,
        "decision_rule":"Treat Structural Risk as an L2 failure switch only if it creates a large, year-consistent separation in the current champion's failures. Tiny average LogLoss shifts are not sufficient.",
        "router_used":False,"retraining":False,"new_kaggle_downloads":False,
        "payout_used":False,"roi_used":False,"2026_locked":True,
        "elapsed_seconds":time.time()-t0,
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/"README.md").write_text(
        "# Structural Risk vs current MARKET+OUTSIDER champion failure audit\n\n"
        "Fast race-level audit using only archived OOS outputs. No model retraining or data download. "
        "Tests whether frozen Structural Risk thresholds isolate races where MARKET+OUTSIDER underweights the true TRIO.\n",
        encoding="utf-8"
    )
    print("===== POOLED HIGH-RISK SEPARATION =====")
    print(pooled.to_string(index=False))
    print("===== YEAR CONSISTENCY =====")
    print(pd.DataFrame(year_consistency).to_string(index=False))
    print("===== SUMMARY =====")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print("L2_STRUCTRISK_OUTSIDER_FAILURE_AUDIT_V1_READY")

if __name__=="__main__":
    main()
