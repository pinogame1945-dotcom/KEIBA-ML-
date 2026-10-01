#!/usr/bin/env python3
import argparse,csv,json,math,os,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l2_ticket_lab_v1 import (
    YEARS,FOLDS,EPS,KING_FEATURES,OUTSIDER_FEATURES,L175_FEATURES,
    parse_year_paths,load_market,load_ballots,load_p3,build_tickets,evaluate,write_csv
)

CORRECTION_FEATURES={
    "KING_CORRECTION":KING_FEATURES,
    "KING_OUTSIDER_CORRECTION":KING_FEATURES+OUTSIDER_FEATURES,
    "L175_CORRECTION":KING_FEATURES+OUTSIDER_FEATURES+L175_FEATURES,
}
ALPHAS=(0.0,0.05,0.10,0.20,0.35,0.50,0.75,1.00,1.50,2.00)

def args():
    p=argparse.ArgumentParser(description="L2 Ticket Residual V1: keep quinella market probability fixed as baseline and learn only multiplicative L1.75 corrections.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--ballots-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--outsider-predictions",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def split_fit_cal(train):
    fit_ids=set(); cal_ids=set()
    for year,g in train[["year","race_id"]].drop_duplicates().groupby("year",sort=True):
        ids=sorted(g["race_id"].astype(str))
        if len(ids)<10: raise SystemExit(f"too few races for internal calibration year={year}")
        cut=max(1,min(len(ids)-1,int(len(ids)*0.80)))
        fit_ids.update(ids[:cut]); cal_ids.update(ids[cut:])
    fit=train[train["race_id"].isin(fit_ids)].copy().reset_index(drop=True)
    cal=train[train["race_id"].isin(cal_ids)].copy().reset_index(drop=True)
    if fit.empty or cal.empty: raise SystemExit("empty internal fit/cal split")
    return fit,cal

def race_softmax_from_market(df,correction,alpha):
    base=np.log(np.clip(df["pair_market_prob_norm"].to_numpy(dtype=float),EPS,1.0))
    corr=np.asarray(correction,dtype=float)
    if len(base)!=len(corr): raise ValueError("correction length mismatch")
    out=np.empty(len(df),dtype=float)
    for _,idx in df.groupby("race_id",sort=False).groups.items():
        ii=np.asarray(list(idx),dtype=int)
        z=base[ii]+float(alpha)*corr[ii]
        z=z-np.max(z)
        e=np.exp(np.clip(z,-60,60)); s=float(e.sum())
        out[ii]=e/s if s>0 else 1.0/len(ii)
    return out

def mean_race_logloss(df,p):
    vals=[]
    for _,idx in df.groupby("race_id",sort=False).groups.items():
        ii=np.asarray(list(idx),dtype=int)
        y=df.loc[ii,"hit"].to_numpy(dtype=int)
        w=np.flatnonzero(y==1)
        if len(w)!=1: continue
        vals.append(-math.log(max(float(p[ii[w[0]]]),EPS)))
    return float(np.mean(vals)) if vals else float("inf")

def fit_correction(name,fit,cal,test,seed):
    feats=CORRECTION_FEATURES[name]
    Xfit=fit[feats].astype(float)
    Xcal=cal[feats].astype(float)
    Xtest=test[feats].astype(float)

    # Each race gets equal total weight. The model never sees pair market odds;
    # market probability remains a fixed baseline outside the correction learner.
    nper=fit.groupby("race_id")["race_id"].transform("size").astype(float)
    weights=1.0/nper.to_numpy()

    model=lgb.LGBMClassifier(
        objective="binary",
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=23,
        min_child_samples=120,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=3.0,
        reg_alpha=0.2,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    t0=time.perf_counter()
    model.fit(Xfit,fit["hit"].astype(int),sample_weight=weights)

    # Raw logit is used only as a relative correction score. Per-race constants
    # cancel during softmax; the market baseline itself is never relearned.
    ccal=model.predict(Xcal,raw_score=True)
    ctest=model.predict(Xtest,raw_score=True)

    alpha_rows=[]
    best=None
    for alpha in ALPHAS:
        p=race_softmax_from_market(cal,ccal,alpha)
        ll=mean_race_logloss(cal,p)
        alpha_rows.append({"model":name,"alpha":alpha,"cal_race_log_loss":ll})
        cand=(ll,alpha)
        if best is None or cand<best: best=cand
    best_ll,best_alpha=best

    ptest=race_softmax_from_market(test,ctest,best_alpha)
    elapsed=time.perf_counter()-t0
    imp=model.booster_.feature_importance(importance_type="gain")
    importance=sorted(
        [{"model":name,"feature":f,"gain":float(g)} for f,g in zip(feats,imp)],
        key=lambda x:-x["gain"]
    )
    return {
        "name":name,
        "p":ptest,
        "alpha":float(best_alpha),
        "cal_logloss":float(best_ll),
        "alpha_rows":alpha_rows,
        "importance":importance[:30],
        "elapsed":elapsed,
        "fit_races":fit["race_id"].nunique(),
        "cal_races":cal["race_id"].nunique(),
    }

def binary_metrics(y,p):
    y=np.asarray(y,dtype=float); p=np.clip(np.asarray(p,dtype=float),EPS,1-EPS)
    if len(y)==0: return {"rows":0,"hits":0,"log_loss":None,"brier":None}
    ll=float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
    br=float(np.mean((p-y)**2))
    return {"rows":len(y),"hits":int(y.sum()),"log_loss":ll,"brier":br}

def subset_masks(df):
    return {
        "ALL":np.ones(len(df),dtype=bool),
        "ANY_DISSENT":df["abs_gap_max"].to_numpy(dtype=float)>=2,
        "STRONG_DISSENT_4PLUS":df["abs_gap_max"].to_numpy(dtype=float)>=4,
        "OUTSIDER_ALIGNMENT_POS":df["outsider_alignment_sum"].to_numpy(dtype=float)>0,
        "OUTSIDER_ALIGNMENT_NEG":df["outsider_alignment_sum"].to_numpy(dtype=float)<0,
        "P3_SIGNAL_PRESENT":df["p3_valid_count"].to_numpy(dtype=float)>0,
    }

def main():
    a=args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()

    market=load_market(a.market_scored)
    ballots=load_ballots(parse_year_paths(a.ballots_year))
    p3=load_p3(a.outsider_predictions)
    tickets,build_stats=build_tickets(market,ballots,p3,a.backfill_root)
    tickets=tickets.reset_index(drop=True)

    fold_rows=[]; alpha_rows=[]; importance_rows=[]; timing_rows=[]; subset_rows=[]
    cpu=max(1,os.cpu_count() or 1); workers=min(3,cpu)

    for fi,(test_year,train_years) in enumerate(FOLDS,1):
        train=tickets[tickets["year"].isin(train_years)].copy().reset_index(drop=True)
        test=tickets[tickets["year"]==test_year].copy().reset_index(drop=True)
        fit,cal=split_fit_cal(train)

        pmarket=test["pair_market_prob_norm"].to_numpy(dtype=float)
        base={"test_year":test_year,"train_years":"|".join(map(str,train_years)),"model":"MARKET_RAW","alpha":0.0,
              "fit_races":fit["race_id"].nunique(),"cal_races":cal["race_id"].nunique(),"parallel_workers":workers}
        base.update(evaluate(test,pmarket)); fold_rows.append(base)

        futures={}
        wall=time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for mi,name in enumerate(CORRECTION_FEATURES,1):
                futures[ex.submit(fit_correction,name,fit,cal,test,20261100+fi*100+mi)]=name
            results={}
            for fut in as_completed(futures):
                r=fut.result(); results[r["name"]]=r

        timing_rows.append({"test_year":test_year,"model":"PARALLEL_WALL","seconds":time.perf_counter()-wall})
        for name in CORRECTION_FEATURES:
            r=results[name]; p=r["p"]
            row={"test_year":test_year,"train_years":"|".join(map(str,train_years)),"model":name,
                 "alpha":r["alpha"],"internal_cal_logloss":r["cal_logloss"],
                 "fit_races":r["fit_races"],"cal_races":r["cal_races"],"parallel_workers":workers}
            row.update(evaluate(test,p)); fold_rows.append(row)
            timing_rows.append({"test_year":test_year,"model":name,"seconds":r["elapsed"]})
            for x in r["alpha_rows"]:
                x["test_year"]=test_year; alpha_rows.append(x)
            for x in r["importance"]:
                x["test_year"]=test_year; importance_rows.append(x)

            for subset,mask in subset_masks(test).items():
                y=test.loc[mask,"hit"].to_numpy(dtype=int)
                mb=binary_metrics(y,pmarket[mask])
                mm=binary_metrics(y,p[mask])
                subset_rows.append({
                    "test_year":test_year,"model":name,"subset":subset,
                    "rows":mm["rows"],"hits":mm["hits"],
                    "market_binary_log_loss":mb["log_loss"],
                    "corrected_binary_log_loss":mm["log_loss"],
                    "delta_binary_log_loss":None if mm["log_loss"] is None else mm["log_loss"]-mb["log_loss"],
                    "market_brier":mb["brier"],
                    "corrected_brier":mm["brier"],
                    "delta_brier":None if mm["brier"] is None else mm["brier"]-mb["brier"],
                })
        print("L2_RESIDUAL_FOLD_DONE "+json.dumps({"test_year":test_year,"train_years":train_years,"workers":workers},separators=(",",":")),flush=True)

    fdf=pd.DataFrame(fold_rows)
    delta_rows=[]
    for year in (2024,2025):
        q=fdf[fdf["test_year"]==year].set_index("model")
        for name in CORRECTION_FEATURES:
            delta_rows.append({
                "test_year":year,"model":name,"selected_alpha":float(q.loc[name,"alpha"]),
                "delta_race_log_loss_vs_market_raw":float(q.loc[name,"race_log_loss"]-q.loc["MARKET_RAW","race_log_loss"]),
                "delta_brier_vs_market_raw":float(q.loc[name,"brier"]-q.loc["MARKET_RAW","brier"]),
                "delta_top1_pp_vs_market_raw":float(q.loc[name,"top1_ticket_accuracy_pct"]-q.loc["MARKET_RAW","top1_ticket_accuracy_pct"]),
                "delta_top5_pp_vs_market_raw":float(q.loc[name,"winner_in_top5_pct"]-q.loc["MARKET_RAW","winner_in_top5_pct"]),
                "delta_top10_pp_vs_market_raw":float(q.loc[name,"winner_in_top10_pct"]-q.loc["MARKET_RAW","winner_in_top10_pct"]),
            })

    write_csv(out/"fold-metrics.csv",fold_rows)
    write_csv(out/"delta-vs-market.csv",delta_rows)
    write_csv(out/"alpha-selection.csv",alpha_rows)
    write_csv(out/"subset-diagnostics.csv",subset_rows)
    write_csv(out/"feature-importance.csv",importance_rows)
    write_csv(out/"timing.csv",timing_rows)

    ddf=pd.DataFrame(delta_rows)
    sdf=pd.DataFrame(subset_rows)
    summary={
        "contract":"L2_TICKET_RESIDUAL_V1",
        "bet_type":"QUINELLA",
        "architecture":"p_corrected(ticket|race) proportional to p_market(ticket|race) * exp(alpha * AI_correction_score)",
        "market_baseline":"within-race normalized inverse final quinella odds; frozen historical benchmark",
        "correction_models":{k:v for k,v in CORRECTION_FEATURES.items()},
        "alpha_grid":list(ALPHAS),
        "alpha_selection":"internal prior-year race split only; test-year outcomes never used",
        "fit_policy":"80% of each prior train year fits correction model; remaining 20% selects alpha; no test-year tuning",
        "market_features_in_correction_model":False,
        "market_context_in_l175_features":"Only precomputed King-vs-market rank disagreement fields from L1.75 are allowed in L175_CORRECTION.",
        "scope":"all priced quinella combinations; no TopN filter, no handcrafted ticket rule, no forced BUY/SKIP",
        "build":build_stats,
        "parallel":{"workers":workers,"cpu_count":cpu,"shared_ticket_table":True,"correction_models_run_concurrently":True},
        "result":{
            name:{
                "beats_market_raw_logloss_both_folds":bool(
                    len(ddf[ddf["model"]==name])==2 and
                    (ddf.loc[ddf["model"]==name,"delta_race_log_loss_vs_market_raw"]<0).all()
                ),
                "selected_alphas":ddf.loc[ddf["model"]==name,["test_year","selected_alpha"]].to_dict(orient="records"),
            } for name in CORRECTION_FEATURES
        },
        "subset_note":"Subset metrics are diagnostics only and are not used to choose a production rule.",
        "elapsed_seconds":time.perf_counter()-start,
        "2026_locked":True,
        "promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== FOLD METRICS =====")
    print((out/"fold-metrics.csv").read_text(encoding="utf-8"))
    print("===== DELTA VS MARKET =====")
    print((out/"delta-vs-market.csv").read_text(encoding="utf-8"))
    print("===== ALPHA =====")
    print((out/"alpha-selection.csv").read_text(encoding="utf-8"))
    print("L2_TICKET_RESIDUAL_V1_READY")

if __name__=="__main__":
    main()
