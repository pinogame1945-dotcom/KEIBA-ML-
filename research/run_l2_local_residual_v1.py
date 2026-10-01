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
from run_l2_ticket_residual_v1 import split_fit_cal,binary_metrics

FEATURES={
    "KING_LOCAL":KING_FEATURES,
    "KING_OUTSIDER_LOCAL":KING_FEATURES+OUTSIDER_FEATURES,
    "L175_LOCAL":KING_FEATURES+OUTSIDER_FEATURES+L175_FEATURES,
}
ALPHAS=(0.0,0.10,0.25,0.50,0.75,1.00,1.25,1.50,2.00)
MAIN_GAP=2
DIAG_GAP=4

def args():
    p=argparse.ArgumentParser(description="L2 Local Residual V1: learn only market probability residuals on quinella tickets containing a King-vs-market dissent horse.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--ballots-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--outsider-predictions",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def local_mask(df,gap=MAIN_GAP):
    return df["abs_gap_max"].to_numpy(dtype=float)>=float(gap)

def apply_additive_residual(df,pred,mask,alpha):
    base=df["pair_market_prob_norm"].to_numpy(dtype=float).copy()
    corr=np.zeros(len(df),dtype=float)
    corr[np.asarray(mask,dtype=bool)]=np.asarray(pred,dtype=float)
    raw=np.clip(base+float(alpha)*corr,EPS,None)
    out=np.empty(len(df),dtype=float)
    for _,idx in df.groupby("race_id",sort=False).groups.items():
        ii=np.asarray(list(idx),dtype=int)
        s=float(raw[ii].sum())
        out[ii]=raw[ii]/s if s>0 else 1.0/len(ii)
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

def fit_local(name,fit,cal,test,seed):
    feats=FEATURES[name]
    mfit=local_mask(fit)
    mcal=local_mask(cal)
    mtest=local_mask(test)
    if mfit.sum()<1000 or mcal.sum()<300:
        raise SystemExit(f"too few local rows {name}: fit={int(mfit.sum())} cal={int(mcal.sum())}")

    fit_local=fit.loc[mfit].copy().reset_index(drop=True)
    Xfit=fit_local[feats].astype(float)
    target=(fit_local["hit"].astype(float)-fit_local["pair_market_prob_norm"].astype(float)).to_numpy()
    nper=fit_local.groupby("race_id")["race_id"].transform("size").astype(float)
    weights=1.0/nper.to_numpy()

    model=lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=240,
        learning_rate=0.03,
        num_leaves=23,
        min_child_samples=120,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=4.0,
        reg_alpha=0.3,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    t0=time.perf_counter()
    model.fit(Xfit,target,sample_weight=weights)

    pcal_pred=model.predict(cal.loc[mcal,feats].astype(float))
    ptest_pred=model.predict(test.loc[mtest,feats].astype(float))

    alpha_rows=[]
    best=None
    for alpha in ALPHAS:
        p=apply_additive_residual(cal,pcal_pred,mcal,alpha)
        ll=mean_race_logloss(cal,p)
        alpha_rows.append({"model":name,"alpha":alpha,"cal_race_log_loss":ll})
        cand=(ll,alpha)
        if best is None or cand<best: best=cand
    best_ll,best_alpha=best
    ptest=apply_additive_residual(test,ptest_pred,mtest,best_alpha)

    imp=model.booster_.feature_importance(importance_type="gain")
    importance=sorted(
        [{"model":name,"feature":f,"gain":float(g)} for f,g in zip(feats,imp)],
        key=lambda x:-x["gain"]
    )
    return {
        "name":name,"p":ptest,"alpha":float(best_alpha),"cal_logloss":float(best_ll),
        "alpha_rows":alpha_rows,"importance":importance[:30],
        "elapsed":time.perf_counter()-t0,
        "fit_local_rows":int(mfit.sum()),"cal_local_rows":int(mcal.sum()),"test_local_rows":int(mtest.sum()),
    }

def subset_rows(test,pmarket,p,name,year):
    out=[]
    masks={
        "ALL":np.ones(len(test),dtype=bool),
        "LOCAL_DISSENT_2PLUS":local_mask(test,2),
        "STRONG_DISSENT_4PLUS":local_mask(test,4),
        "NO_DISSENT_LT2":~local_mask(test,2),
        "OUTSIDER_ALIGNMENT_POS":test["outsider_alignment_sum"].to_numpy(dtype=float)>0,
        "OUTSIDER_ALIGNMENT_NEG":test["outsider_alignment_sum"].to_numpy(dtype=float)<0,
    }
    for label,mask in masks.items():
        y=test.loc[mask,"hit"].to_numpy(dtype=int)
        mb=binary_metrics(y,pmarket[mask]); mm=binary_metrics(y,p[mask])
        out.append({
            "test_year":year,"model":name,"subset":label,
            "rows":mm["rows"],"hits":mm["hits"],
            "market_binary_log_loss":mb["log_loss"],"corrected_binary_log_loss":mm["log_loss"],
            "delta_binary_log_loss":None if mm["log_loss"] is None else mm["log_loss"]-mb["log_loss"],
            "market_brier":mb["brier"],"corrected_brier":mm["brier"],
            "delta_brier":None if mm["brier"] is None else mm["brier"]-mb["brier"],
        })
    return out

def main():
    a=args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    market=load_market(a.market_scored)
    ballots=load_ballots(parse_year_paths(a.ballots_year))
    p3=load_p3(a.outsider_predictions)
    tickets,build_stats=build_tickets(market,ballots,p3,a.backfill_root)
    tickets=tickets.reset_index(drop=True)

    cpu=max(1,os.cpu_count() or 1); workers=min(3,cpu)
    fold_rows=[]; delta_rows=[]; alpha_rows=[]; subset_diag=[]; importance_rows=[]; timing_rows=[]

    for fi,(test_year,train_years) in enumerate(FOLDS,1):
        train=tickets[tickets["year"].isin(train_years)].copy().reset_index(drop=True)
        test=tickets[tickets["year"]==test_year].copy().reset_index(drop=True)
        fit,cal=split_fit_cal(train)
        pmarket=test["pair_market_prob_norm"].to_numpy(dtype=float)

        base={"test_year":test_year,"train_years":"|".join(map(str,train_years)),"model":"MARKET_RAW","alpha":0.0,
              "parallel_workers":workers,"local_test_rows":int(local_mask(test).sum())}
        base.update(evaluate(test,pmarket)); fold_rows.append(base)

        futures={}; wall=time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for mi,name in enumerate(FEATURES,1):
                futures[ex.submit(fit_local,name,fit,cal,test,20261200+fi*100+mi)]=name
            results={}
            for fut in as_completed(futures):
                r=fut.result(); results[r["name"]]=r
        timing_rows.append({"test_year":test_year,"model":"PARALLEL_WALL","seconds":time.perf_counter()-wall})

        for name in FEATURES:
            r=results[name]; p=r["p"]
            row={"test_year":test_year,"train_years":"|".join(map(str,train_years)),"model":name,
                 "alpha":r["alpha"],"internal_cal_logloss":r["cal_logloss"],
                 "fit_local_rows":r["fit_local_rows"],"cal_local_rows":r["cal_local_rows"],
                 "test_local_rows":r["test_local_rows"],"parallel_workers":workers}
            row.update(evaluate(test,p)); fold_rows.append(row)
            delta_rows.append({
                "test_year":test_year,"model":name,"selected_alpha":r["alpha"],
                "delta_race_log_loss_vs_market_raw":row["race_log_loss"]-base["race_log_loss"],
                "delta_brier_vs_market_raw":row["brier"]-base["brier"],
                "delta_top1_pp_vs_market_raw":row["top1_ticket_accuracy_pct"]-base["top1_ticket_accuracy_pct"],
                "delta_top5_pp_vs_market_raw":row["winner_in_top5_pct"]-base["winner_in_top5_pct"],
                "delta_top10_pp_vs_market_raw":row["winner_in_top10_pct"]-base["winner_in_top10_pct"],
            })
            subset_diag.extend(subset_rows(test,pmarket,p,name,test_year))
            timing_rows.append({"test_year":test_year,"model":name,"seconds":r["elapsed"]})
            for x in r["alpha_rows"]:
                x["test_year"]=test_year; alpha_rows.append(x)
            for x in r["importance"]:
                x["test_year"]=test_year; importance_rows.append(x)

        print("L2_LOCAL_RESIDUAL_FOLD_DONE "+json.dumps({
            "test_year":test_year,"train_years":train_years,"workers":workers,
            "local_rows":int(local_mask(test).sum()),"all_rows":len(test)
        },separators=(",",":")),flush=True)

    write_csv(out/"fold-metrics.csv",fold_rows)
    write_csv(out/"delta-vs-market.csv",delta_rows)
    write_csv(out/"alpha-selection.csv",alpha_rows)
    write_csv(out/"subset-diagnostics.csv",subset_diag)
    write_csv(out/"feature-importance.csv",importance_rows)
    write_csv(out/"timing.csv",timing_rows)

    ddf=pd.DataFrame(delta_rows)
    summary={
        "contract":"L2_LOCAL_RESIDUAL_V1",
        "bet_type":"QUINELLA",
        "architecture":"MARKET_RAW globally; additive predicted (hit - market_probability) correction only on tickets containing >=1 horse with abs(King rank - market rank) >= 2; renormalize within race",
        "main_local_threshold_abs_rank_gap":MAIN_GAP,
        "strong_dissent_diagnostic_threshold":DIAG_GAP,
        "thresholds_predeclared":True,
        "test_year_threshold_tuning":False,
        "target":"ticket_hit - normalized_market_quinella_probability",
        "correction_models":{k:v for k,v in FEATURES.items()},
        "alpha_grid":list(ALPHAS),
        "alpha_selection":"prior-year internal calibration only",
        "outside_local_region":"correction score forced to zero before within-race normalization",
        "scope":"all priced quinella combinations retained; no TopN filter, no handcrafted ticket selection",
        "build":build_stats,
        "parallel":{"workers":workers,"cpu_count":cpu,"shared_ticket_table":True,"local_models_run_concurrently":True},
        "result":{
            name:{
                "beats_market_raw_logloss_both_folds":bool(
                    len(ddf[ddf["model"]==name])==2 and
                    (ddf.loc[ddf["model"]==name,"delta_race_log_loss_vs_market_raw"]<0).all()
                ),
                "selected_alphas":ddf.loc[ddf["model"]==name,["test_year","selected_alpha"]].to_dict(orient="records"),
            } for name in FEATURES
        },
        "elapsed_seconds":time.perf_counter()-start,
        "2026_locked":True,
        "promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== FOLD METRICS ====="); print((out/"fold-metrics.csv").read_text())
    print("===== DELTA ====="); print((out/"delta-vs-market.csv").read_text())
    print("===== ALPHA ====="); print((out/"alpha-selection.csv").read_text())
    print("L2_LOCAL_RESIDUAL_V1_READY")

if __name__=="__main__":
    main()
