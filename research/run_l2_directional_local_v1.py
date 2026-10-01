#!/usr/bin/env python3
import argparse,json,math,os,time
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

ROUTES=("UP_ONLY","DOWN_ONLY","MIXED")
ALPHAS=(0.0,0.05,0.10,0.20,0.35,0.50,0.75,1.00,1.50)
DIRECTIONAL_FEATURES=[
    "up_outsider_agree_count","up_outsider_oppose_count",
    "down_outsider_agree_count","down_outsider_oppose_count",
    "up_alignment_sum","down_alignment_sum",
]
FEATURES=KING_FEATURES+OUTSIDER_FEATURES+L175_FEATURES+DIRECTIONAL_FEATURES

def args():
    p=argparse.ArgumentParser(description="L2 Directional Local V1: learn separate market-residual corrections for UP, DOWN and MIXED quinella dissent routes.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--ballots-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--outsider-predictions",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def route_mask(df,route):
    up=df["up_count"].to_numpy(dtype=float)>0
    down=df["down_count"].to_numpy(dtype=float)>0
    if route=="UP_ONLY": return up & ~down
    if route=="DOWN_ONLY": return down & ~up
    if route=="MIXED": return up & down
    raise ValueError(route)

def apply_route_corrections(df,preds_by_route,alphas_by_route):
    base=df["pair_market_prob_norm"].to_numpy(dtype=float).copy()
    corr=np.zeros(len(df),dtype=float)
    for route in ROUTES:
        mask=route_mask(df,route)
        pred=np.asarray(preds_by_route.get(route,[]),dtype=float)
        if len(pred)!=int(mask.sum()):
            raise ValueError(f"prediction size drift route={route} pred={len(pred)} rows={int(mask.sum())}")
        corr[mask]=float(alphas_by_route.get(route,0.0))*pred
    raw=np.clip(base+corr,EPS,None)
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
        if len(w)==1:
            vals.append(-math.log(max(float(p[ii[w[0]]]),EPS)))
    return float(np.mean(vals)) if vals else float("inf")

def fit_route(route,fit,cal,test,seed):
    mf=route_mask(fit,route); mc=route_mask(cal,route); mt=route_mask(test,route)
    if int(mf.sum())<1000 or int(mc.sum())<250 or int(mt.sum())<250:
        raise SystemExit(f"too few route rows route={route} fit={int(mf.sum())} cal={int(mc.sum())} test={int(mt.sum())}")

    f=fit.loc[mf].copy().reset_index(drop=True)
    X=f[FEATURES].astype(float)
    target=(f["hit"].astype(float)-f["pair_market_prob_norm"].astype(float)).to_numpy()

    nper=f.groupby("race_id")["race_id"].transform("size").astype(float)
    weights=1.0/nper.to_numpy()

    model=lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=240,
        learning_rate=0.03,
        num_leaves=23,
        min_child_samples=100,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=4.0,
        reg_alpha=0.3,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    t0=time.perf_counter()
    model.fit(X,target,sample_weight=weights)
    pred_cal=model.predict(cal.loc[mc,FEATURES].astype(float))
    pred_test=model.predict(test.loc[mt,FEATURES].astype(float))

    # Select this route's correction strength using only prior-year calibration.
    zeros_cal={r:np.zeros(int(route_mask(cal,r).sum()),dtype=float) for r in ROUTES}
    zeros_cal[route]=pred_cal
    alpha_rows=[]; best=None
    for alpha in ALPHAS:
        a={r:0.0 for r in ROUTES}; a[route]=alpha
        p=apply_route_corrections(cal,zeros_cal,a)
        ll=mean_race_logloss(cal,p)
        alpha_rows.append({"route":route,"alpha":alpha,"cal_race_log_loss":ll})
        cand=(ll,alpha)
        if best is None or cand<best: best=cand
    best_ll,best_alpha=best

    imp=model.booster_.feature_importance(importance_type="gain")
    importance=sorted(
        [{"route":route,"feature":f,"gain":float(g)} for f,g in zip(FEATURES,imp)],
        key=lambda x:-x["gain"]
    )
    return {
        "route":route,
        "pred_test":pred_test,
        "pred_cal":pred_cal,
        "alpha":float(best_alpha),
        "cal_logloss":float(best_ll),
        "alpha_rows":alpha_rows,
        "importance":importance[:35],
        "elapsed":time.perf_counter()-t0,
        "fit_rows":int(mf.sum()),"cal_rows":int(mc.sum()),"test_rows":int(mt.sum()),
    }

def route_diagnostics(test,pmarket,pmodel,label):
    rows=[]
    up=test["up_count"].to_numpy(dtype=float)>0
    down=test["down_count"].to_numpy(dtype=float)>0
    masks={
        "UP_ONLY":up & ~down,
        "DOWN_ONLY":down & ~up,
        "MIXED":up & down,
        "UP_STRONG4":(up & ~down) & (test["abs_gap_max"].to_numpy(dtype=float)>=4),
        "DOWN_STRONG4":(down & ~up) & (test["abs_gap_max"].to_numpy(dtype=float)>=4),
        "MIXED_STRONG4":(up & down) & (test["abs_gap_max"].to_numpy(dtype=float)>=4),
        "UP_OUTSIDER_AGREE":(up & ~down) & (test["up_outsider_agree_count"].to_numpy(dtype=float)>0),
        "DOWN_OUTSIDER_AGREE":(down & ~up) & (test["down_outsider_agree_count"].to_numpy(dtype=float)>0),
        "UP_OUTSIDER_OPPOSE":(up & ~down) & (test["up_outsider_oppose_count"].to_numpy(dtype=float)>0),
        "DOWN_OUTSIDER_OPPOSE":(down & ~up) & (test["down_outsider_oppose_count"].to_numpy(dtype=float)>0),
    }
    for subset,mask in masks.items():
        y=test.loc[mask,"hit"].to_numpy(dtype=int)
        mb=binary_metrics(y,pmarket[mask]); mm=binary_metrics(y,pmodel[mask])
        rows.append({
            "model":label,"subset":subset,"rows":mm["rows"],"hits":mm["hits"],
            "market_binary_log_loss":mb["log_loss"],"corrected_binary_log_loss":mm["log_loss"],
            "delta_binary_log_loss":None if mm["log_loss"] is None else mm["log_loss"]-mb["log_loss"],
            "market_brier":mb["brier"],"corrected_brier":mm["brier"],
            "delta_brier":None if mm["brier"] is None else mm["brier"]-mb["brier"],
        })
    return rows

def main():
    a=args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()

    market=load_market(a.market_scored)
    ballots=load_ballots(parse_year_paths(a.ballots_year))
    p3=load_p3(a.outsider_predictions)
    tickets,build_stats=build_tickets(market,ballots,p3,a.backfill_root)
    tickets=tickets.reset_index(drop=True)

    missing=[c for c in FEATURES if c not in tickets.columns]
    if missing: raise SystemExit(f"directional feature columns missing: {missing}")

    cpu=max(1,os.cpu_count() or 1); workers=min(3,cpu)
    fold_rows=[]; delta_rows=[]; alpha_rows=[]; diag_rows=[]; importance_rows=[]; timing_rows=[]; route_size_rows=[]

    for fi,(test_year,train_years) in enumerate(FOLDS,1):
        train=tickets[tickets["year"].isin(train_years)].copy().reset_index(drop=True)
        test=tickets[tickets["year"]==test_year].copy().reset_index(drop=True)
        fit,cal=split_fit_cal(train)
        pmarket=test["pair_market_prob_norm"].to_numpy(dtype=float)

        base={"test_year":test_year,"train_years":"|".join(map(str,train_years)),
              "model":"MARKET_RAW","parallel_workers":workers}
        base.update(evaluate(test,pmarket)); fold_rows.append(base)

        futures={}; wall=time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for ri,route in enumerate(ROUTES,1):
                futures[ex.submit(fit_route,route,fit,cal,test,20261300+fi*100+ri)]=route
            result={}
            for fut in as_completed(futures):
                r=fut.result(); result[r["route"]]=r
        timing_rows.append({"test_year":test_year,"model":"PARALLEL_WALL","seconds":time.perf_counter()-wall})

        # Evaluate each route alone.
        for route in ROUTES:
            r=result[route]
            preds={x:np.zeros(int(route_mask(test,x).sum()),dtype=float) for x in ROUTES}
            preds[route]=r["pred_test"]
            alphas={x:0.0 for x in ROUTES}; alphas[route]=r["alpha"]
            p=apply_route_corrections(test,preds,alphas)
            row={"test_year":test_year,"train_years":"|".join(map(str,train_years)),
                 "model":f"{route}_ONLY","selected_alpha":r["alpha"],
                 "internal_cal_logloss":r["cal_logloss"],"parallel_workers":workers}
            row.update(evaluate(test,p)); fold_rows.append(row)
            delta_rows.append({
                "test_year":test_year,"model":f"{route}_ONLY","selected_alpha":r["alpha"],
                "delta_race_log_loss_vs_market_raw":row["race_log_loss"]-base["race_log_loss"],
                "delta_brier_vs_market_raw":row["brier"]-base["brier"],
                "delta_top1_pp_vs_market_raw":row["top1_ticket_accuracy_pct"]-base["top1_ticket_accuracy_pct"],
                "delta_top5_pp_vs_market_raw":row["winner_in_top5_pct"]-base["winner_in_top5_pct"],
                "delta_top10_pp_vs_market_raw":row["winner_in_top10_pct"]-base["winner_in_top10_pct"],
            })
            for d in route_diagnostics(test,pmarket,p,f"{route}_ONLY"):
                d["test_year"]=test_year; diag_rows.append(d)

            timing_rows.append({"test_year":test_year,"model":route,"seconds":r["elapsed"]})
            for x in r["alpha_rows"]:
                x["test_year"]=test_year; alpha_rows.append(x)
            for x in r["importance"]:
                x["test_year"]=test_year; importance_rows.append(x)
            route_size_rows.append({
                "test_year":test_year,"route":route,
                "fit_rows":r["fit_rows"],"cal_rows":r["cal_rows"],"test_rows":r["test_rows"],
                "selected_alpha":r["alpha"]
            })

        # Combine the three independently calibrated directional corrections.
        preds={r:result[r]["pred_test"] for r in ROUTES}
        alphas={r:result[r]["alpha"] for r in ROUTES}
        pcombo=apply_route_corrections(test,preds,alphas)
        combo={"test_year":test_year,"train_years":"|".join(map(str,train_years)),
               "model":"DIRECTIONAL_COMBINED",
               "selected_alpha":"|".join(f"{r}:{alphas[r]}" for r in ROUTES),
               "parallel_workers":workers}
        combo.update(evaluate(test,pcombo)); fold_rows.append(combo)
        delta_rows.append({
            "test_year":test_year,"model":"DIRECTIONAL_COMBINED",
            "selected_alpha":"|".join(f"{r}:{alphas[r]}" for r in ROUTES),
            "delta_race_log_loss_vs_market_raw":combo["race_log_loss"]-base["race_log_loss"],
            "delta_brier_vs_market_raw":combo["brier"]-base["brier"],
            "delta_top1_pp_vs_market_raw":combo["top1_ticket_accuracy_pct"]-base["top1_ticket_accuracy_pct"],
            "delta_top5_pp_vs_market_raw":combo["winner_in_top5_pct"]-base["winner_in_top5_pct"],
            "delta_top10_pp_vs_market_raw":combo["winner_in_top10_pct"]-base["winner_in_top10_pct"],
        })
        for d in route_diagnostics(test,pmarket,pcombo,"DIRECTIONAL_COMBINED"):
            d["test_year"]=test_year; diag_rows.append(d)

        print("L2_DIRECTIONAL_LOCAL_FOLD_DONE "+json.dumps({
            "test_year":test_year,
            "alphas":alphas,
            "route_rows":{r:int(route_mask(test,r).sum()) for r in ROUTES},
        },separators=(",",":")),flush=True)

    write_csv(out/"fold-metrics.csv",fold_rows)
    write_csv(out/"delta-vs-market.csv",delta_rows)
    write_csv(out/"alpha-selection.csv",alpha_rows)
    write_csv(out/"route-diagnostics.csv",diag_rows)
    write_csv(out/"route-sizes.csv",route_size_rows)
    write_csv(out/"feature-importance.csv",importance_rows)
    write_csv(out/"timing.csv",timing_rows)

    ddf=pd.DataFrame(delta_rows)
    combo=ddf[ddf["model"]=="DIRECTIONAL_COMBINED"]
    summary={
        "contract":"L2_DIRECTIONAL_LOCAL_V1",
        "bet_type":"QUINELLA",
        "routes":{
            "UP_ONLY":"pair contains >=1 L1.75 UP horse and no DOWN horse",
            "DOWN_ONLY":"pair contains >=1 L1.75 DOWN horse and no UP horse",
            "MIXED":"pair contains both UP and DOWN horses",
        },
        "near_only_policy":"MARKET_RAW unchanged",
        "target":"ticket_hit - normalized_market_quinella_probability",
        "direction_threshold":"UP if signed_rank_gap>=2; DOWN if <=-2; otherwise NEAR",
        "outsider_directional_context":DIRECTIONAL_FEATURES,
        "alpha_grid":list(ALPHAS),
        "alpha_selection":"each route independently on prior-year internal calibration; no test-year tuning",
        "combine_policy":"apply each route's prior-selected alpha simultaneously, then normalize within race",
        "build":build_stats,
        "parallel":{"workers":workers,"cpu_count":cpu,"shared_ticket_table":True,"route_models_run_concurrently":True},
        "result":{
            "combined_beats_market_raw_logloss_both_folds":bool(
                len(combo)==2 and (combo["delta_race_log_loss_vs_market_raw"]<0).all()
            ),
            "route_selected_alphas":route_size_rows,
        },
        "elapsed_seconds":time.perf_counter()-start,
        "promotion":False,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== FOLD METRICS ====="); print((out/"fold-metrics.csv").read_text())
    print("===== DELTA ====="); print((out/"delta-vs-market.csv").read_text())
    print("===== ROUTE SIZES ====="); print((out/"route-sizes.csv").read_text())
    print("L2_DIRECTIONAL_LOCAL_V1_READY")

if __name__=="__main__":
    main()
