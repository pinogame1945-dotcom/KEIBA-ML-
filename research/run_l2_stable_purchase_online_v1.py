#!/usr/bin/env python3
"""L2 pre-race-score based online purchase-rate stabilization V1.

Objective: Preserve broad race coverage and improve expected cash return by
deciding whether to skip a FROZEN EV-axis bet, with no result-dependent actions.

Read only previously saved OOS predictions and exact cash settlement. 2024
and 2025 score models were each trained on earlier years. This stage consumes
those saved score predictions; does NOT retrain or recompute original data.

Policies process dates strictly ascending. Every day's thresholds rely ONLY on
scores from prior days of the SAME evaluated year; no day outcomes are touched
to form decisions. Within each day a running skip cap can be enforced using
previous decisions only, without previewing future-day score/labels.

IMPORTANT CAVEAT: original EV anchor + OOS features include FINAL win odds,
so this historical backtest does NOT prove live pre-off executability.
2025 was previously reviewed and is NOT a pristine uninspected year.
No 2026, GPU, paid runner, external datasets or artifact upload.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

PREVIOUS_TAIL_RUN="38682123161"
SOURCE_STAKE_RUN="38680436244"
RULES=(
 "TRIO|AI_PROB|2",
 "WIN|SINGLE|0",
 "WIDE|MARKET|2",
 "EXACTA_FIRST|MARKET|2",
 "TRIFECTA_FIRST|AI_PROB|2",
)
METHODS=("DIRECT_CAP25","DIRECT_CAP100","HURDLE_PAYOFF","TAIL_COMPONENT",
         "AVG_CAP25_CAP100","BLEND_CAP100_TAIL")
SETTINGS=("CUM","ROLL_512","ROLL_1024","CUM_CAP",
          "ROLL_512_CAP","ROLL_1024_CAP")
TARGET_SKIP=(10,15,20,25)
GUARDS=("NO_GUARD","TAIL_TOP10")
EXPERIMENT_YEARS=(2024,2025)
WARMUP=144
DATE_COL="race_date"
Q_LOWER=.0000001

def req(cond,txt):
    if not bool(cond):raise AssertionError(txt)

def load(path):
    return pd.read_csv(path,compression="infer",dtype={"race_id":str},low_memory=False)

def score_universe(base_root,pred_root):
    feat=load(base_root/"race-feature-snapshot.csv.gz")[["year","race_id","race_date"]]
    bets=load(base_root/"all-10-rules-settled-per-race.csv.gz")
    bet=bets[bets.rule.isin(RULES)].merge(feat,on=["year","race_id"],
                                            validate="many_to_one")
    scores=load(pred_root/"new-oos-price-tail-and-payoff-scores.csv.gz")
    req(scores.duplicated(["year","race_id","rule","method"]).sum()==0,
        "duplicate saved predictions")
    req(set(scores.year)==set(EXPERIMENT_YEARS),"saved year mismatch")
    req(set(scores.method)=={
         "DIRECT_CAP5","DIRECT_CAP25","DIRECT_CAP100","DIRECT_RAW",
         "HURDLE_PAYOFF","TAIL_COMPONENT"},"saved model set drift")
    req(scores.expected_return_per_stake.notna().all() and
        scores.payout_tail_probability.notna().all(),
        "non-finite saved payoff scores")
    scores=scores[scores.rule.isin(RULES)&
                  scores.method.isin(("DIRECT_CAP25","DIRECT_CAP100",
                        "HURDLE_PAYOFF","TAIL_COMPONENT"))]
    pred=scores.pivot(index=["year","race_id","rule"],columns="method",
                      values="expected_return_per_stake")
    pred.columns=["s_"+c for c in pred.columns]
    pred=pred.reset_index()
    tails=scores[scores.method.eq("TAIL_COMPONENT")][
         ["year","race_id","rule","payout_tail_probability"]].rename(
         columns={"payout_tail_probability":"tail_risk"})
    result=bet.merge(pred,on=["year","race_id","rule"],how="inner",
                     validate="one_to_one")
    result=result.merge(tails,on=["year","race_id","rule"],how="inner",
                        validate="one_to_one")
    req(len(result)==len(RULES)*3456*2,f"cash-pred coverage {len(result)}")
    req(result.groupby(["year","rule"]).size().eq(3456).all(),"year coverage")
    req(result["return"].ge(0).all() and result.stake.ge(100).all(),"invalid cash")
    req(result[["s_DIRECT_CAP25","s_DIRECT_CAP100",
                "s_HURDLE_PAYOFF","s_TAIL_COMPONENT","tail_risk"]]
        .apply(np.isfinite).all().all(),"nonfinite predictions")
    return result

def rankpct(sorted_historical,test):
    """Score percentile of each new observation versus strictly older scores."""
    older=np.asarray(sorted_historical,dtype=float)
    req(len(older)>0,"missing historical scores")
    return np.searchsorted(older,np.asarray(test,dtype=float),
                           side="right")/len(older)

def compute_signals(hist,today,method):
    """Produce skip risk (larger => worse) and tail protection indicator.

    Each transformation is percentile-ranked relative ONLY to historical
    predictions, avoiding scale drift between different regressors.
    """
    pc={}
    for c in ("s_DIRECT_CAP25","s_DIRECT_CAP100",
              "s_HURDLE_PAYOFF","s_TAIL_COMPONENT","tail_risk"):
        pc[c]=rankpct(np.sort(hist[c].to_numpy(dtype=float)),
                      today[c].to_numpy(dtype=float))
    if method=="AVG_CAP25_CAP100":
        good=.5*pc["s_DIRECT_CAP25"]+.5*pc["s_DIRECT_CAP100"]
    elif method=="BLEND_CAP100_TAIL":
        good=.75*pc["s_DIRECT_CAP100"]+.25*pc["tail_risk"]
    else:good=pc["s_"+method]
    risk=1-good
    return risk,pc["tail_risk"]

def evaluate_route(frame,risk,tail_pct,setting,target,guard):
    """Online; no outcome/odds target is consulted when producing a flag."""
    dates=frame[DATE_COL].to_numpy()
    n=len(frame)
    out=np.zeros(n,dtype=bool)
    seen=0
    accepted_skip=0
    # risk percentiles score relative to historical score distributions
    # are compared to fixed prior-exposure quantile of 0.9/0.85/0.8/0.75.
    cut=1-target/100
    quota="_CAP" in setting
    for i in range(n):
        want=bool(risk[i]>cut)
        if guard=="TAIL_TOP10" and tail_pct[i]>.90:
            want=False
        if quota:
            # Maximum skip share including previously bought warmup days.
            allowed=int(math.floor((i+1)*target/100))
            want=want and accepted_skip<allowed
        out[i]=want
        accepted_skip+=int(want)
        seen+=1
    return out

def daily_scores(frame,method,setting):
    """Date-synchronized sequential scoring: entire current day must be
    scored BEFORE its own scores enter the reference window. No intra-day
    or future-day score is in reference thresholds."""
    x=frame.sort_values([DATE_COL,"race_id"]).reset_index(drop=True)
    n=len(x)
    risk=np.full(n,np.nan)
    tail=np.full(n,np.nan)
    day_groups=x.groupby(DATE_COL,sort=True,observed=True)
    group_indices=[grp.index.to_numpy(dtype=int) for _,grp in day_groups]
    hist_cols=["s_DIRECT_CAP25","s_DIRECT_CAP100",
               "s_HURDLE_PAYOFF","s_TAIL_COMPONENT","tail_risk"]
    vals=x[hist_cols].to_numpy(dtype=float)
    count_history=0
    for ids in group_indices:
        if count_history>=WARMUP:
            start=max(0,count_history-(512 if "ROLL_512" in setting
                                     else 1024 if "ROLL_1024" in setting
                                     else count_history))
            history=x.iloc[start:count_history]
            today=x.iloc[ids]
            risk[ids],tail[ids]=compute_signals(history,today,method)
        count_history+=len(ids)
    return x,risk,tail

def receipt(frame,flag):
    y=frame["return"].to_numpy(float)
    s=frame["stake"].to_numpy(float)
    keep=~flag
    cost=float(s[keep].sum())
    pay=float(y[keep].sum())
    miss=(y==0)&flag
    top=np.sort(y[(y>0)&flag])[::-1]
    return dict(
       race_count=len(frame),
       purchase_races=int(keep.sum()),
       purchase_rate_pct=round(100*float(keep.mean()),6),
       skipped_races=int(flag.sum()),
       skip_rate_pct=round(100*float(flag.mean()),6),
       invested_yen=cost,returned_yen=pay,
       net_profit_yen=round(pay-cost,2),
       roi_pct=round(100*pay/cost,6) if cost else None,
       base_profit_yen=round(float((y-s).sum()),2),
       base_roi_pct=round(100*float(y.sum())/float(s.sum()),6),
       net_gain_from_skips_yen=round(float((s[flag]-y[flag]).sum()),2),
       missed_losers=int(miss.sum()),
       lost_hit_races=int(((y>0)&flag).sum()),
       lost_hit_payout_yen=float(y[flag].sum()),
       lost_top_3_payout_yen=float(top[:3].sum()),
       retained_payout_pct=round(100*pay/float(y.sum()),5),
    )

def payout_calibration(x,flag,method,setting,target,guard):
    val=x["return"].to_numpy(float)/x["stake"].to_numpy(float)
    features=[]
    score=x[method].to_numpy(float) if method in x.columns else None
    if score is None:return []
    # Diagnostics only: statistical binning from current year's features.
    bins=pd.qcut(pd.Series(score).rank(method="first"),5,labels=False)
    for b in range(5):
        idx=(bins.to_numpy()==b)
        cost=float(x.loc[idx,"stake"].sum())
        features.append({
           "bin_of_test_year_forecast_DIAGNOSTIC_ONLY":b,
           "mean_predicted_payout_per_yen":float(np.mean(score[idx])),
           "actual_payout_per_yen":float(x.loc[idx,"return"].sum()/cost),
           "count":int(idx.sum()),"method":method,
           "setting":setting,"guard":guard,"target":target,
        })
    return features

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--base-root",required=True)
    a.add_argument("--predict-root",required=True)
    a.add_argument("--out",required=True)
    args=a.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df=score_universe(Path(args.base_root),Path(args.predict_root))
    req(set(df.year)=={2024,2025},"2026 sealed")
    records=[]
    diag=[]
    examples=[]
    # Cache score ranking per year+rule+method+setting exactly once, reused
    # for all budgets and payout guards; no redundant model fitting.
    for y in EXPERIMENT_YEARS:
      for rule in RULES:
        sample=df[(df.year.eq(y))&(df.rule.eq(rule))]
        req(len(sample)==3456,f"missing race {y} {rule}")
        for setting in SETTINGS:
          for method in METHODS:
            x,risk,tail=daily_scores(sample,method,setting)
            req(np.isnan(risk[:WARMUP]).all() and
                np.isnan(tail[:WARMUP]).all(),"warmup raw risk drift")
            for budget in TARGET_SKIP:
              for guard in GUARDS:
                skip=evaluate_route(x,risk,tail,setting,budget,guard)
                # Warmup mandatory purchase all.
                req(not skip[:WARMUP].any(),"warmup skip")
                rec={"year":y,"rule":rule,"score_method":method,
                     "threshold_mode":setting,"target_skip_pct":budget,
                     "protect_high_tail":guard,**receipt(x,skip)}
                records.append(rec)
                if rule=="TRIO|AI_PROB|2":
                  for i in range(len(x)):
                    if guard=="NO_GUARD" and method in ("DIRECT_CAP25","DIRECT_CAP100") and budget in (15,20):
                      examples.append({"year":y,"race_id":x.at[i,"race_id"],
                         "date":x.at[i,DATE_COL],"method":method,"setting":setting,
                         "target_skip_pct":budget,"skipped":int(skip[i]),
                         "skip_risk_score":risk[i] if np.isfinite(risk[i]) else None,
                         "historical_payout_yen":x.at[i,"return"],
                         "stake_yen":x.at[i,"stake"]})
            # Score calibration by test-year (DIAGNOSTIC ONLY, not decision).
            if method in ("DIRECT_CAP25","DIRECT_CAP100","HURDLE_PAYOFF","TAIL_COMPONENT") and setting=="CUM":
                # model payoffs already frozen, add bin summaries.
                field="s_"+method
                values=x[field].to_numpy(float)
                bins=pd.qcut(pd.Series(values).rank(method="first"),5,labels=False)
                for b in range(5):
                    m=bins.eq(b).to_numpy()
                    cost=float(x.loc[m,"stake"].sum())
                    diag.append({"year":y,"rule":rule,"method":method,"forecast_quintile":b+1,
                       "pred_ratio_mean":float(values[m].mean()),
                       "historical_roi_pct":round(100*float(x.loc[m,"return"].sum())/cost,5),
                       "observations":int(m.sum())})
    allr=pd.DataFrame(records)
    req(len(allr)==len(EXPERIMENT_YEARS)*len(RULES)*len(METHODS)*
         len(SETTINGS)*len(TARGET_SKIP)*len(GUARDS),
         f"missing policy candidates {len(allr)}")
    req(allr.net_profit_yen.notna().all(),"settlement gaps")
    # Selection from 2024 ONLY. Lock exact rule+model+streaming policy and
    # evaluate 2025 with no learned labels/adaptation.
    chosen=[]
    for scope,rv in (("TRIO_PRIMARY","TRIO|AI_PROB|2"),("ALL_BET_FAMILIES",None)):
      for guarantee in (90,85,80,70):
        pool=allr[allr.year.eq(2024)]
        if rv:pool=pool[pool.rule.eq(rv)]
        pool=pool[pool.purchase_rate_pct.ge(guarantee)]
        req(not pool.empty,f"no 2024 candidates {scope}/{guarantee}")
        # Objective is PROFIT (ROI secondary). Deterministic, no 2025 metric.
        sel=pool.sort_values(
           ["net_profit_yen","roi_pct","purchase_races",
            "target_skip_pct","score_method","threshold_mode","protect_high_tail"],
           ascending=[False,False,False,True,True,True,True]).iloc[0]
        follow=allr[(allr.year.eq(2025))&(allr.rule.eq(sel["rule"]))&
           (allr.score_method.eq(sel["score_method"]))&
           (allr.threshold_mode.eq(sel["threshold_mode"]))&
           (allr.target_skip_pct.eq(sel["target_skip_pct"]))&
           (allr.protect_high_tail.eq(sel["protect_high_tail"]))]
        req(len(follow)==1,"2025 locked candidate missing")
        fv=follow.iloc[0]
        chosen.append({"scope":scope,"2024_coverage_floor_pct":guarantee,
          "selected_rule":sel["rule"],"model":sel["score_method"],
          "online_threshold":sel["threshold_mode"],
          "target_skip_pct":int(sel["target_skip_pct"]),
          "tail_guard":sel["protect_high_tail"],
          **{"2024_"+z:sel[z] for z in ("purchase_races","purchase_rate_pct",
               "roi_pct","net_profit_yen","lost_hit_races")},
          **{"2025_"+z:fv[z] for z in ("purchase_races","purchase_rate_pct",
               "roi_pct","net_profit_yen","lost_hit_races")},
        })
    # Only after 2024 policy choices fixed, calculate descriptive year pair
    # stability and pooled profit. Non-promotable exploratory ranking.
    pairs=allr.pivot_table(index=[
       "rule","score_method","threshold_mode","target_skip_pct","protect_high_tail"],
       columns="year",values=["net_profit_yen","roi_pct","purchase_rate_pct",
                              "purchase_races","invested_yen","returned_yen"],
       aggfunc="first")
    pairs.columns=["{}_{}".format(a,b) for a,b in pairs.columns]
    pairs=pairs.reset_index()
    pairs["two_year_profit_yen"]=pairs.net_profit_yen_2024+pairs.net_profit_yen_2025
    pairs["two_year_roi_pct"]=100*(pairs.returned_yen_2024+pairs.returned_yen_2025)/(
       pairs.invested_yen_2024+pairs.invested_yen_2025)
    pairs["max_coverage_gap_pct"]=abs(pairs.purchase_rate_pct_2024-
                                       pairs.purchase_rate_pct_2025)
    pairs.to_csv(out/"diagnostic-two-year-frontier-NOT-selection.csv",index=False)
    allr.to_csv(out/"year-by-year-chronological-policies.csv",index=False)
    pd.DataFrame(chosen).to_csv(out/"2024-selected-locked-2025.csv",index=False)
    pd.DataFrame(diag).to_csv(out/"frozen-payout-score-calibration-diagnostic.csv",index=False)
    pd.DataFrame(examples).to_csv(out/"selected-streaming-decisions-audit.csv.gz",
                                 index=False,compression="gzip")
    best=allr[(allr.year.eq(2024))&
        (allr.rule.eq("TRIO|AI_PROB|2"))&
        (allr.purchase_rate_pct.ge(80))].sort_values(
           ["net_profit_yen","purchase_races"],ascending=[False,False]).head(8)
    summary={
       "contract":"L2_ONLINE_BUY_RATE_STABILITY_V1",
       "scope":"frozen 2023-2025 historical AI expected-value horse",
       "train_years_for_cached_models":"2024 prior 2023; 2025 prior 2023+2024",
       "prediction_source_run":PREVIOUS_TAIL_RUN,
       "exact_cash_source_run":SOURCE_STAKE_RUN,
       "evaluated_years":list(EXPERIMENT_YEARS),
       "n_policy_rows":len(allr),
       "model_types":list(METHODS),
       "threshold_strategies":list(SETTINGS),
       "skip_targets_pct":list(TARGET_SKIP),
       "guard":list(GUARDS),
       "first_day_warmup_races":WARMUP,
       "online_rule":"Never use future dates' scores or ANY evaluated-year outcomes for daily threshold",
       "online_score_source":"Final-odds-based frozen OOS estimates. NOT live-timestamp executable.",
       "no_future_race_outcomes_used_in_skip_flags":True,
       "performance_metric":"Real payout based ROI, yen profit, purchase coverage, forfeited hit payout",
       "2024_only_policy_selections_locked_2025":chosen,
       "2024_primary_top_8_in_sample_search":best.to_dict("records"),
       "2025_not_pristine_holdout":True,
       "multiplicity":"2024 selected across many strategies; fragile + small positive profits cannot justify deployment",
       "no_live_promotion":True,
       "2026_sealed":True,
       "no_paid_runner":True,
       "no_new_data_download":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,
                                               indent=2,default=str)+"\n",encoding="utf-8")
    print("L2_ONLINE_STABLE_PURCHASE_START")
    print(pd.DataFrame(chosen).to_csv(index=False))
    print("TRIO_2024_HIGH_COVERAGE_TOP")
    print(best[["year","rule","score_method","threshold_mode","target_skip_pct",
                "protect_high_tail","purchase_rate_pct","roi_pct",
                "net_profit_yen"]].to_csv(index=False))
    print("L2_ONLINE_STABLE_PURCHASE_COMPLETE",flush=True)

if __name__=="__main__":
    main()
