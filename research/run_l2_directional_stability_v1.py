#!/usr/bin/env python3
import argparse,json,math,os,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from run_l2_ticket_lab_v1 import (
    FOLDS,EPS,parse_year_paths,load_market,load_ballots,load_p3,build_tickets,write_csv
)
from run_l2_ticket_residual_v1 import split_fit_cal,binary_metrics
from run_l2_directional_local_v1 import (
    ROUTES,fit_route,route_mask,apply_route_corrections,route_diagnostics
)

BOOTSTRAP_DRAWS=4000
SEGMENT_BOOTSTRAP_DRAWS=1500
MIN_SEGMENT_RACES=80

def args():
    p=argparse.ArgumentParser(description="L2 Directional Stability V1: freeze Directional Local V1 and audit uncertainty/segments without route or threshold search.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--ballots-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--outsider-predictions",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def winner_route(row):
    up=float(row.up_count)>0
    down=float(row.down_count)>0
    if up and not down: return "UP_ONLY"
    if down and not up: return "DOWN_ONLY"
    if up and down: return "MIXED"
    return "NEAR_ONLY"

def race_loss_ledger(test,pmarket,pmodel,model,test_year):
    rows=[]
    pm=np.asarray(pmarket,dtype=float); pp=np.asarray(pmodel,dtype=float)
    for rid,idx in test.groupby("race_id",sort=False).groups.items():
        ii=np.asarray(list(idx),dtype=int)
        sub=test.loc[ii]
        y=sub["hit"].to_numpy(dtype=int)
        pos=np.flatnonzero(y==1)
        if len(pos)!=1: continue
        w=int(pos[0]); wr=sub.iloc[w]
        ml=-math.log(max(float(pm[ii[w]]),EPS))
        pl=-math.log(max(float(pp[ii[w]]),EPS))
        rows.append({
            "test_year":int(test_year),"model":model,"race_id":str(rid),
            "market_loss":ml,"model_loss":pl,"delta_log_loss":pl-ml,
            "winner_market_prob":float(pm[ii[w]]),
            "winner_pair_odds":float(wr["pair_market_odds"]),
            "field_size":int(wr["field_size"]),
            "race_class":str(wr.get("race_class","UNKNOWN") or "UNKNOWN"),
            "grade":str(wr.get("grade","UNKNOWN") or "UNKNOWN"),
            "winner_route":winner_route(wr),
        })
    return rows

def bootstrap_mean(values,draws,seed):
    x=np.asarray(values,dtype=float)
    n=len(x)
    if n<2:
        m=float(np.mean(x)) if n else None
        return {"n":n,"mean":m,"ci_low":m,"ci_high":m,"prob_improve":None}
    rng=np.random.default_rng(seed)
    means=[]
    chunk=200
    for start in range(0,draws,chunk):
        k=min(chunk,draws-start)
        idx=rng.integers(0,n,size=(k,n),endpoint=False)
        means.append(x[idx].mean(axis=1))
    b=np.concatenate(means)
    return {
        "n":n,
        "mean":float(x.mean()),
        "ci_low":float(np.quantile(b,0.025)),
        "ci_high":float(np.quantile(b,0.975)),
        "prob_improve":float(np.mean(b<0.0)),
    }

def field_band(n):
    n=int(n)
    if n<=8: return "LE8"
    if n<=12: return "9_12"
    if n<=16: return "13_16"
    return "17_PLUS"

def odds_band(x):
    x=float(x)
    if x<10: return "LT10"
    if x<30: return "10_30"
    if x<100: return "30_100"
    return "100_PLUS"

def market_prob_band(x):
    x=float(x)
    if x>=0.15: return "GE15PCT"
    if x>=0.08: return "8_15PCT"
    if x>=0.03: return "3_8PCT"
    return "LT3PCT"

def add_segment_columns(df):
    d=df.copy()
    d["field_band"]=[field_band(x) for x in d["field_size"]]
    d["winner_odds_band"]=[odds_band(x) for x in d["winner_pair_odds"]]
    d["winner_market_prob_band"]=[market_prob_band(x) for x in d["winner_market_prob"]]
    return d

def segment_audit(ledger,seed_base):
    rows=[]
    d=add_segment_columns(pd.DataFrame(ledger))
    segment_cols=("race_class","grade","field_band","winner_odds_band","winner_market_prob_band","winner_route")
    for (year,model),g0 in d.groupby(["test_year","model"],sort=True):
        for si,col in enumerate(segment_cols):
            for seg,g in g0.groupby(col,sort=True):
                vals=g["delta_log_loss"].to_numpy(dtype=float)
                if len(vals)<MIN_SEGMENT_RACES: continue
                bs=bootstrap_mean(vals,SEGMENT_BOOTSTRAP_DRAWS,seed_base+int(year)*100+si*13+len(rows))
                rows.append({
                    "test_year":int(year),"model":model,"segment_type":col,"segment":str(seg),
                    "races":bs["n"],"mean_delta_log_loss":bs["mean"],
                    "ci_low":bs["ci_low"],"ci_high":bs["ci_high"],"bootstrap_prob_improve":bs["prob_improve"],
                    "improved_mean":bool(bs["mean"]<0),
                })
    return rows

def ticket_price_audit(test,pmarket,pmodel,model,test_year):
    rows=[]
    odds=test["pair_market_odds"].to_numpy(dtype=float)
    bands=np.asarray([odds_band(x) for x in odds],dtype=object)
    y=test["hit"].to_numpy(dtype=int)
    for band in ("LT10","10_30","30_100","100_PLUS"):
        m=bands==band
        if int(m.sum())<500: continue
        a=binary_metrics(y[m],pmarket[m]); b=binary_metrics(y[m],pmodel[m])
        rows.append({
            "test_year":int(test_year),"model":model,"ticket_odds_band":band,
            "tickets":int(m.sum()),"hits":int(y[m].sum()),
            "market_binary_log_loss":a["log_loss"],"model_binary_log_loss":b["log_loss"],
            "delta_binary_log_loss":b["log_loss"]-a["log_loss"],
            "market_brier":a["brier"],"model_brier":b["brier"],
            "delta_brier":b["brier"]-a["brier"],
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

    cpu=max(1,os.cpu_count() or 1); workers=min(3,cpu)
    ledger=[]; alpha_rows=[]; route_rows=[]; route_diag=[]; ticket_price_rows=[]; timing=[]

    for fi,(test_year,train_years) in enumerate(FOLDS,1):
        train=tickets[tickets["year"].isin(train_years)].copy().reset_index(drop=True)
        test=tickets[tickets["year"]==test_year].copy().reset_index(drop=True)
        fit,cal=split_fit_cal(train)
        pmarket=test["pair_market_prob_norm"].to_numpy(dtype=float)

        futures={}; wall=time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for ri,route in enumerate(ROUTES,1):
                futures[ex.submit(fit_route,route,fit,cal,test,20261400+fi*100+ri)]=route
            result={}
            for fut in as_completed(futures):
                r=fut.result(); result[r["route"]]=r
        timing.append({"test_year":test_year,"model":"PARALLEL_WALL","seconds":time.perf_counter()-wall})

        # Frozen V1 route alphas: chosen only by prior-year calibration exactly as Directional Local V1.
        alphas={r:result[r]["alpha"] for r in ROUTES}
        preds={r:result[r]["pred_test"] for r in ROUTES}
        pcombo=apply_route_corrections(test,preds,alphas)

        ledger.extend(race_loss_ledger(test,pmarket,pcombo,"DIRECTIONAL_COMBINED",test_year))
        ticket_price_rows.extend(ticket_price_audit(test,pmarket,pcombo,"DIRECTIONAL_COMBINED",test_year))
        for d in route_diagnostics(test,pmarket,pcombo,"DIRECTIONAL_COMBINED"):
            d["test_year"]=test_year; route_diag.append(d)

        for route in ROUTES:
            r=result[route]
            zeros={x:np.zeros(int(route_mask(test,x).sum()),dtype=float) for x in ROUTES}
            zeros[route]=r["pred_test"]
            aa={x:0.0 for x in ROUTES}; aa[route]=r["alpha"]
            pr=apply_route_corrections(test,zeros,aa)
            ledger.extend(race_loss_ledger(test,pmarket,pr,f"{route}_ONLY",test_year))
            ticket_price_rows.extend(ticket_price_audit(test,pmarket,pr,f"{route}_ONLY",test_year))
            route_rows.append({
                "test_year":test_year,"route":route,
                "fit_rows":r["fit_rows"],"cal_rows":r["cal_rows"],"test_rows":r["test_rows"],
                "selected_alpha":r["alpha"],"cal_logloss":r["cal_logloss"],
            })
            for ar in r["alpha_rows"]:
                x=dict(ar); x["test_year"]=test_year; alpha_rows.append(x)
            timing.append({"test_year":test_year,"model":route,"seconds":r["elapsed"]})

        print("STABILITY_FOLD_READY "+json.dumps({"test_year":test_year,"alphas":alphas,"races":test["race_id"].nunique()},separators=(",",":")),flush=True)

    # Bootstrap race-level unseen-year deltas. No threshold/route search here.
    ldf=pd.DataFrame(ledger)
    bootstrap_rows=[]
    for model,g in ldf.groupby("model",sort=True):
        for year,g2 in g.groupby("test_year",sort=True):
            bs=bootstrap_mean(g2["delta_log_loss"],BOOTSTRAP_DRAWS,20261500+int(year)+sum(map(ord,model))%997)
            bootstrap_rows.append({"scope":str(year),"model":model,**bs})
        bs=bootstrap_mean(g["delta_log_loss"],BOOTSTRAP_DRAWS,20261600+sum(map(ord,model))%997)
        bootstrap_rows.append({"scope":"POOLED_2024_2025","model":model,**bs})

    segment_rows=segment_audit(ledger,20261700)

    bdf=pd.DataFrame(bootstrap_rows)
    combo=bdf[bdf["model"]=="DIRECTIONAL_COMBINED"].copy()
    year_means={str(r["scope"]):float(r["mean"]) for _,r in combo.iterrows() if r["scope"] in ("2024","2025")}
    pooled=combo[combo["scope"]=="POOLED_2024_2025"].iloc[0]
    strict_stability=bool(
        year_means.get("2024",1)>=0 is False and
        year_means.get("2025",1)>=0 is False and
        float(pooled["ci_high"])<0
    )

    write_csv(out/"race-loss-ledger.csv",ledger)
    write_csv(out/"bootstrap-summary.csv",bootstrap_rows)
    write_csv(out/"segment-stability.csv",segment_rows)
    write_csv(out/"ticket-price-band.csv",ticket_price_rows)
    write_csv(out/"route-diagnostics.csv",route_diag)
    write_csv(out/"route-alphas.csv",route_rows)
    write_csv(out/"alpha-selection.csv",alpha_rows)
    write_csv(out/"timing.csv",timing)

    summary={
        "contract":"L2_DIRECTIONAL_STABILITY_V1",
        "source_architecture":"L2_DIRECTIONAL_LOCAL_V1 frozen routes and thresholds",
        "bet_type":"QUINELLA",
        "no_new_route_search":True,
        "no_threshold_search":True,
        "direction_threshold":"UP >= +2, DOWN <= -2, MIXED both; unchanged from V1",
        "bootstrap":{"draws":BOOTSTRAP_DRAWS,"unit":"race","ci":"percentile_95","seeded":True},
        "segments":["race_class","grade","field_size_band","winning_ticket_odds_band","winning_market_probability_band","winning_ticket_direction_route"],
        "ticket_price_bands":["LT10","10_30","30_100","100_PLUS"],
        "minimum_segment_races":MIN_SEGMENT_RACES,
        "build":build_stats,
        "parallel":{"workers":workers,"cpu_count":cpu,"route_models_run_concurrently":True,"shared_ticket_table":True},
        "stability_gate":{
            "definition":"combined mean delta log loss < 0 in both unseen years AND pooled bootstrap 95% CI upper bound < 0",
            "passed":strict_stability,
            "year_mean_delta_log_loss":year_means,
            "pooled_mean_delta_log_loss":float(pooled["mean"]),
            "pooled_ci_low":float(pooled["ci_low"]),
            "pooled_ci_high":float(pooled["ci_high"]),
            "pooled_bootstrap_prob_improve":float(pooled["prob_improve"]),
        },
        "policy":"No production promotion from this run alone. If strict stability passes, freeze directional probability layer before L2B value evaluation; otherwise retain as research signal.",
        "elapsed_seconds":time.perf_counter()-start,
        "promotion":False,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== SUMMARY ====="); print((out/"summary.json").read_text())
    print("===== BOOTSTRAP ====="); print((out/"bootstrap-summary.csv").read_text())
    print("===== ROUTE ALPHAS ====="); print((out/"route-alphas.csv").read_text())
    print("L2_DIRECTIONAL_STABILITY_V1_READY")

if __name__=="__main__":
    main()
