#!/usr/bin/env python3
import argparse,csv,json,math,os,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l2_ticket_lab_v1 import (
    YEARS,FOLDS,EPS,MARKET_FEATURES,KING_FEATURES,OUTSIDER_FEATURES,L175_FEATURES,
    parse_year_paths,load_market,load_ballots,load_p3,build_tickets,write_csv
)
from run_l2_directional_local_v1 import DIRECTIONAL_FEATURES

VALUE_FEATURES={
    "MARKET_VALUE":MARKET_FEATURES,
    "MARKET_KING_VALUE":MARKET_FEATURES+KING_FEATURES,
    "L175_VALUE":MARKET_FEATURES+KING_FEATURES+OUTSIDER_FEATURES+L175_FEATURES+DIRECTIONAL_FEATURES,
}
BLENDS=(0.0,0.02,0.05,0.10,0.20,0.35,0.50,0.75,1.0)
EDGE_THRESHOLDS=(0.0,0.05,0.10,0.20,0.30,0.50)
EDGE_BANDS=(
    ("NEG",-1e99,0.0),
    ("0_5",0.0,0.05),
    ("5_10",0.05,0.10),
    ("10_20",0.10,0.20),
    ("20_50",0.20,0.50),
    ("50_PLUS",0.50,1e99),
)

def args():
    p=argparse.ArgumentParser(description="L2B Quinella Value V1: market-centered ticket hit probability and predeclared edge/ROI evaluation.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--ballots-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--outsider-predictions",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def split_fit_cal(train):
    races=train[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"]).reset_index(drop=True)
    if len(races)<20: raise SystemExit("too few train races")
    cut=max(1,min(len(races)-1,int(len(races)*0.80)))
    fit_ids=set(races.iloc[:cut]["race_id"].astype(str))
    cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
    fit=train[train["race_id"].isin(fit_ids)].copy().reset_index(drop=True)
    cal=train[train["race_id"].isin(cal_ids)].copy().reset_index(drop=True)
    if fit.empty or cal.empty: raise SystemExit("empty fit/cal split")
    return fit,cal

def normalize_by_race(raw,df):
    raw=np.asarray(raw,dtype=float)
    out=np.empty(len(raw),dtype=float)
    for _,idx in df.groupby("race_id",sort=False).groups.items():
        ii=np.asarray(list(idx),dtype=int)
        v=np.clip(raw[ii],EPS,None)
        s=float(v.sum())
        out[ii]=v/s if s>0 else 1.0/len(ii)
    return out

def blend_probs(df,pmodel,lam):
    pm=df["pair_market_prob_norm"].to_numpy(dtype=float)
    p=(1.0-float(lam))*pm+float(lam)*np.asarray(pmodel,dtype=float)
    return normalize_by_race(p,df)

def mean_race_logloss(df,p):
    vals=[]
    for _,idx in df.groupby("race_id",sort=False).groups.items():
        ii=np.asarray(list(idx),dtype=int)
        y=df.loc[ii,"hit"].to_numpy(dtype=int)
        w=np.flatnonzero(y==1)
        if len(w)==1:
            vals.append(-math.log(max(float(p[ii[w[0]]]),EPS)))
    return float(np.mean(vals)) if vals else float("inf")

def fit_value_model(name,fit,cal,test,seed):
    feats=VALUE_FEATURES[name]
    Xfit=fit[feats].astype(float)
    Xcal=cal[feats].astype(float)
    Xtest=test[feats].astype(float)

    nper=fit.groupby("race_id")["race_id"].transform("size").astype(float)
    weights=1.0/nper.to_numpy()

    model=lgb.LGBMClassifier(
        objective="binary",
        n_estimators=260,
        learning_rate=0.035,
        num_leaves=27,
        min_child_samples=100,
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
    pcal0=normalize_by_race(model.predict_proba(Xcal)[:,1],cal)
    ptest0=normalize_by_race(model.predict_proba(Xtest)[:,1],test)

    blend_rows=[]; best=None
    for lam in BLENDS:
        p=blend_probs(cal,pcal0,lam)
        ll=mean_race_logloss(cal,p)
        blend_rows.append({"model":name,"blend":lam,"cal_race_log_loss":ll})
        cand=(ll,lam)
        if best is None or cand<best: best=cand
    best_ll,best_lam=best
    ptest=blend_probs(test,ptest0,best_lam)

    imp=model.booster_.feature_importance(importance_type="gain")
    importance=sorted(
        [{"model":name,"feature":f,"gain":float(g)} for f,g in zip(feats,imp)],
        key=lambda x:-x["gain"]
    )
    return {
        "name":name,"p":ptest,"blend":float(best_lam),"cal_logloss":float(best_ll),
        "blend_rows":blend_rows,"importance":importance[:35],
        "elapsed":time.perf_counter()-t0,
        "fit_races":fit["race_id"].nunique(),"cal_races":cal["race_id"].nunique(),
    }

def max_drawdown(profits):
    equity=0.0; peak=0.0; dd=0.0
    for x in profits:
        equity+=float(x); peak=max(peak,equity); dd=max(dd,peak-equity)
    return dd

def edge_band(x):
    for name,lo,hi in EDGE_BANDS:
        if lo<=x<hi: return name
    return "50_PLUS"

def evaluate_edges(test,p,model,test_year):
    odds=test["pair_market_odds"].to_numpy(dtype=float)
    ret=test["return_yen_per100"].to_numpy(dtype=float)
    hit=test["hit"].to_numpy(dtype=int)
    edge=np.asarray(p,dtype=float)*odds-1.0

    threshold_rows=[]
    for th in EDGE_THRESHOLDS:
        totals={"tickets":0,"stake":0.0,"ret":0.0,"hit_tickets":0,"bought_races":0,"hit_races":0}
        race_profits=[]; ticket_counts=[]
        for rid,idx in test.groupby("race_id",sort=True).groups.items():
            ii=np.asarray(list(idx),dtype=int)
            m=edge[ii]>=th
            n=int(m.sum())
            if n==0: continue
            rret=float(ret[ii][m].sum()); stake=100.0*n
            totals["tickets"]+=n; totals["stake"]+=stake; totals["ret"]+=rret
            totals["hit_tickets"]+=int(hit[ii][m].sum())
            totals["bought_races"]+=1; totals["hit_races"]+=int((hit[ii][m]>0).any())
            race_profits.append(rret-stake); ticket_counts.append(n)
        all_races=test["race_id"].nunique()
        threshold_rows.append({
            "test_year":int(test_year),"model":model,"edge_threshold":th,
            "all_races":int(all_races),"bought_races":totals["bought_races"],
            "bought_race_pct":100.0*totals["bought_races"]/all_races if all_races else None,
            "tickets":totals["tickets"],
            "avg_tickets_per_bought_race":totals["tickets"]/totals["bought_races"] if totals["bought_races"] else None,
            "p95_tickets_per_bought_race":float(np.quantile(ticket_counts,0.95)) if ticket_counts else None,
            "hit_tickets":totals["hit_tickets"],"hit_races":totals["hit_races"],
            "race_hit_rate_pct":100.0*totals["hit_races"]/totals["bought_races"] if totals["bought_races"] else None,
            "stake_yen":totals["stake"],"return_yen":totals["ret"],
            "profit_yen":totals["ret"]-totals["stake"],
            "roi_pct":100.0*totals["ret"]/totals["stake"] if totals["stake"] else None,
            "max_drawdown_yen":max_drawdown(race_profits),
        })

    band_acc=defaultdict(lambda:{"n":0,"p":0.0,"hits":0,"stake":0.0,"ret":0.0,"edge":0.0,"odds":0.0})
    for i,e in enumerate(edge):
        b=edge_band(float(e)); z=band_acc[b]
        z["n"]+=1; z["p"]+=float(p[i]); z["hits"]+=int(hit[i]); z["stake"]+=100.0
        z["ret"]+=float(ret[i]); z["edge"]+=float(e); z["odds"]+=float(odds[i])

    band_rows=[]
    for name,_,_ in EDGE_BANDS:
        z=band_acc[name]
        band_rows.append({
            "test_year":int(test_year),"model":model,"edge_band":name,
            "tickets":z["n"],
            "mean_predicted_hit_pct":100.0*z["p"]/z["n"] if z["n"] else None,
            "actual_hit_pct":100.0*z["hits"]/z["n"] if z["n"] else None,
            "calibration_gap_pp":100.0*(z["p"]-z["hits"])/z["n"] if z["n"] else None,
            "mean_edge_pct":100.0*z["edge"]/z["n"] if z["n"] else None,
            "mean_odds":z["odds"]/z["n"] if z["n"] else None,
            "roi_pct":100.0*z["ret"]/z["stake"] if z["stake"] else None,
        })
    return threshold_rows,band_rows

def model_probability_metrics(test,p,model,test_year,blend):
    race_losses=[]; brier=[]
    for _,idx in test.groupby("race_id",sort=False).groups.items():
        ii=np.asarray(list(idx),dtype=int); y=test.loc[ii,"hit"].to_numpy(dtype=int)
        w=np.flatnonzero(y==1)
        if len(w)!=1: continue
        race_losses.append(-math.log(max(float(p[ii[w[0]]]),EPS)))
        brier.extend((p[ii]-y)**2)
    return {
        "test_year":int(test_year),"model":model,"selected_market_blend":blend,
        "races":test["race_id"].nunique(),"tickets":len(test),
        "race_log_loss":float(np.mean(race_losses)),
        "brier":float(np.mean(brier)),
    }

def main():
    a=args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()

    market=load_market(a.market_scored)
    ballots=load_ballots(parse_year_paths(a.ballots_year))
    p3=load_p3(a.outsider_predictions)
    tickets,build_stats=build_tickets(market,ballots,p3,a.backfill_root)
    tickets=tickets.reset_index(drop=True)

    forbidden={"hit","return_yen_per100","race_date"}
    for name,feats in VALUE_FEATURES.items():
        if forbidden.intersection(feats):
            raise SystemExit(f"evaluation leakage in features {name}: {sorted(forbidden.intersection(feats))}")

    required=["return_yen_per100","race_date"]+DIRECTIONAL_FEATURES
    missing=[x for x in required if x not in tickets.columns]
    if missing: raise SystemExit(f"missing L2B columns: {missing}")

    cpu=max(1,os.cpu_count() or 1); workers=min(3,cpu)
    prob_rows=[]; threshold_rows=[]; band_rows=[]; blend_rows=[]; imp_rows=[]; timing=[]

    for fi,(test_year,train_years) in enumerate(FOLDS,1):
        train=tickets[tickets["year"].isin(train_years)].copy().reset_index(drop=True)
        test=tickets[tickets["year"]==test_year].copy().reset_index(drop=True)
        fit,cal=split_fit_cal(train)
        pmarket=test["pair_market_prob_norm"].to_numpy(dtype=float)

        prob_rows.append(model_probability_metrics(test,pmarket,"MARKET_RAW",test_year,0.0))
        tr,br=evaluate_edges(test,pmarket,"MARKET_RAW",test_year)
        threshold_rows.extend(tr); band_rows.extend(br)

        futures={}; wall=time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for mi,name in enumerate(VALUE_FEATURES,1):
                futures[ex.submit(fit_value_model,name,fit,cal,test,20261800+fi*100+mi)]=name
            result={}
            for fut in as_completed(futures):
                r=fut.result(); result[r["name"]]=r
        timing.append({"test_year":test_year,"model":"PARALLEL_WALL","seconds":time.perf_counter()-wall})

        for name in VALUE_FEATURES:
            r=result[name]
            prob_rows.append(model_probability_metrics(test,r["p"],name,test_year,r["blend"]))
            tr,br=evaluate_edges(test,r["p"],name,test_year)
            threshold_rows.extend(tr); band_rows.extend(br)
            timing.append({"test_year":test_year,"model":name,"seconds":r["elapsed"]})
            for x in r["blend_rows"]:
                x["test_year"]=test_year; blend_rows.append(x)
            for x in r["importance"]:
                x["test_year"]=test_year; imp_rows.append(x)

        print("L2B_VALUE_FOLD_DONE "+json.dumps({
            "test_year":test_year,
            "blends":{name:result[name]["blend"] for name in VALUE_FEATURES},
            "races":test["race_id"].nunique(),"tickets":len(test)
        },separators=(",",":")),flush=True)

    write_csv(out/"probability-metrics.csv",prob_rows)
    write_csv(out/"threshold-policy.csv",threshold_rows)
    write_csv(out/"edge-band-calibration.csv",band_rows)
    write_csv(out/"blend-selection.csv",blend_rows)
    write_csv(out/"feature-importance.csv",imp_rows)
    write_csv(out/"timing.csv",timing)

    tdf=pd.DataFrame(threshold_rows)
    stable=[]
    for model in VALUE_FEATURES:
        for th in EDGE_THRESHOLDS:
            q=tdf[(tdf["model"]==model)&(tdf["edge_threshold"]==th)].copy()
            if len(q)!=2: continue
            stable.append({
                "model":model,"edge_threshold":th,
                "roi_positive_both_years":bool((q["roi_pct"]>100.0).all()),
                "coverage_ge20pct_both_years":bool((q["bought_race_pct"]>=20.0).all()),
                "min_bought_race_pct":float(q["bought_race_pct"].min()),
                "min_roi_pct":float(q["roi_pct"].min()) if q["roi_pct"].notna().all() else None,
                "max_avg_tickets_per_bought_race":float(q["avg_tickets_per_bought_race"].max()) if q["avg_tickets_per_bought_race"].notna().all() else None,
            })
    write_csv(out/"cross-year-screen.csv",stable)

    summary={
        "contract":"L2B_QUINELLA_VALUE_V1",
        "bet_type":"QUINELLA",
        "canonical_probability_baseline":"MARKET_RAW normalized inverse final quinella odds",
        "historical_market_note":"Final odds are a research proxy only; operations must substitute the latest available pre-race odds snapshot.",
        "models":{k:v for k,v in VALUE_FEATURES.items()},
        "probability_policy":"model ticket probabilities are normalized within race then blended back toward MARKET_RAW; blend chosen only by prior-year calibration race log loss",
        "blend_grid":list(BLENDS),
        "edge_definition":"blended_probability * quinella_odds - 1",
        "edge_thresholds_predeclared":list(EDGE_THRESHOLDS),
        "edge_bands_predeclared":[x[0] for x in EDGE_BANDS],
        "training_target":"ticket_hit only",
        "payout_used_for_training":False,
        "payout_use":"ROI evaluation only",
        "no_test_year_threshold_selection":True,
        "cross_year_screen_note":"descriptive only; does not promote or select a production threshold",
        "build":build_stats,
        "parallel":{"workers":workers,"cpu_count":cpu,"shared_ticket_table":True,"models_run_concurrently":True},
        "elapsed_seconds":time.perf_counter()-start,
        "promotion":False,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== SUMMARY ====="); print((out/"summary.json").read_text())
    print("===== PROBABILITY ====="); print((out/"probability-metrics.csv").read_text())
    print("===== THRESHOLDS ====="); print((out/"threshold-policy.csv").read_text())
    print("===== EDGE BANDS ====="); print((out/"edge-band-calibration.csv").read_text())
    print("L2B_QUINELLA_VALUE_V1_READY")

if __name__=="__main__":
    main()
