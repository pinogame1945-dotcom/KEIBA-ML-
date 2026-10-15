#!/usr/bin/env python3
"""L2 high-payout protection: chronological utility/hurdle/tail-aware comparison.

Consumes ONLY saved V1 OOS features/cash settlements/scores. No Kaggle, no
BACKFILL, no 2026, no paid compute. The frozen original AI EV-anchor uses
retrospective FINAL odds; therefore results are research-only, not deployable.
Training and score cutoff calibration strictly precede each evaluation year:
2024=train early 2023, calibrate late 2023, refit 2023;
2025=train 2023+early 2024, calibrate late 2024, refit through 2024.
The 2025 result has been analyzed previously, NOT a pristine blind holdout.
2024 policy search and LOCKED 2025 confirmation explicitly segregated.
"""
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_l2_skip_value_batch_v1 import FEATURES as FEATURE_NAMES

RULES=(
    "TRIO|AI_PROB|2",
    "TRIFECTA_FIRST|AI_PROB|2",
    "WIN|SINGLE|0",
    "WIDE|MARKET|2",
    "EXACTA_FIRST|MARKET|2",
)
YEARS=(2024,2025)
METHODS=("PREV_UTILITY","DIRECT_CAP5","DIRECT_CAP25","DIRECT_CAP100",
         "DIRECT_RAW","HURDLE_PAYOFF","TAIL_COMPONENT")
POLICIES=("NONE","GUARD_TAIL_TOP10","GUARD_TAIL_TOP20",
          "GUARD_EXPECTED_TOP20","GUARD_COMBINED_TOP20")
SKIP_RATE=(0,5,10,15,20,30,40)
HGB_COMMON=dict(max_iter=70,max_leaf_nodes=9,max_depth=3,
    min_samples_leaf=110,learning_rate=.045,l2_regularization=20,
    early_stopping=False,random_state=1945)
SOURCE_RUN="38680436244"

def check(c,m):
    if not bool(c):raise ValueError(m)

def gbr(min_leaf=110):
    return HistGradientBoostingRegressor(**{**HGB_COMMON,"min_samples_leaf":min_leaf})
def gbc(min_leaf=110):
    return HistGradientBoostingClassifier(**{**HGB_COMMON,"min_samples_leaf":min_leaf})
def prep_feats(df,med=None):
    values=df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    values[~np.isfinite(values)]=np.nan
    if med is None:
        med=np.nanmedian(values,axis=0)
        med=np.where(np.isfinite(med),med,0.0)
    return np.where(np.isnan(values),med,values).astype(np.float32),med

def age_split(df,year):
    tr=df[df.year.lt(year)].copy().sort_values(["race_date","race_id"]).reset_index(drop=True)
    te=df[df.year.eq(year)].copy().sort_values(["race_date","race_id"]).reset_index(drop=True)
    check(len(te)==3456 and len(tr)==3456*(year-2023),"year coverage drift")
    # Ex ante calibration: hold latest ~20% calendar races of prior year
    p=tr[tr.year.eq(year-1)]
    cdate=sorted(p.race_date.unique())[int(.78*len(p.race_date.unique()))]
    proper=tr[tr.race_date.lt(cdate)]
    cal=tr[tr.race_date.ge(cdate)]
    check(len(proper)>=2000 and len(cal)>=450 and len(cal)<1500,
          f"not enough prior-only chronological calibration {year}")
    check(proper.race_date.max()<cal.race_date.min() and
          cal.race_date.max()<te.race_date.min(), "time ordering broken")
    return tr,te,proper,cal

def ratio(df):
    return df["return"].to_numpy(dtype=float)/df["stake"].to_numpy(dtype=float)

def model_predictions(tr,targets,kind):
    """Predict expected PAYOUT / stake plus hit and high-tail risk.

    Tail classifier is fitted on all prior races, win payout regressor fitted
    only to historically winning races. Evaluated race outcomes never enter.
    """
    xx,med=prep_feats(tr)
    yy=ratio(tr)
    outs={}
    if kind.startswith("DIRECT"):
        lim={"DIRECT_CAP5":5,"DIRECT_CAP25":25,"DIRECT_CAP100":100,
             "DIRECT_RAW":None}[kind]
        y=yy if lim is None else np.minimum(yy,lim)
        m=gbr()
        m.fit(xx,y)
        for name,z in targets.items():
            zz,_=prep_feats(z,med)
            outs[name]={"expected":np.maximum(0,m.predict(zz)),
                        "tail_risk":np.zeros(len(z))}
        return outs
    check(kind in ("HURDLE_PAYOFF","TAIL_COMPONENT"),kind)
    yhit=(yy>0).astype(int)
    check(yhit.sum()>50 and (len(yhit)-yhit.sum())>50,"insufficient win classes")
    win_model=gbc()
    win_model.fit(xx,yhit)
    # Conditional log payout; modest sample sizes limit depth.
    winmask=yhit==1
    logwin=np.log(np.maximum(yy[winmask],.01))
    p_model=gbr(min_leaf=20)
    p_model.fit(xx[winmask],logwin)
    if kind=="TAIL_COMPONENT":
        # High-payoff protected event: payouts >=8x total wager.
        evt=(yy>=8).astype(int)
        if evt.sum()<15:
            evt=(yy>=5).astype(int)
        check(evt.sum()>=10,f"tail event too scarce: {evt.sum()}")
        # Penalized logistic risk, with scaling trained entirely on history.
        scaler=StandardScaler().fit(xx)
        tm=LogisticRegression(C=.25,max_iter=200,solver="lbfgs")
        tm.fit(scaler.transform(xx),evt)
        normal=yy[(yy>0)&(yy<8)]
        conditional_baseline=float(np.clip(np.mean(normal) if len(normal)>5
                 else np.mean(yy[winmask]),.1,30))
        tail=yy[evt.astype(bool)]
        conditional_tail=float(np.clip(np.mean(tail),conditional_baseline,200))
    # Transform winner log prediction to payout scale, with train-only smearing.
    approx=np.exp(np.clip(p_model.predict(xx[winmask]),-3,8))
    smear=np.mean(np.clip(yy[winmask]/np.maximum(approx,.01),.05,30))
    smear=float(np.clip(smear,.5,4))
    for name,z in targets.items():
        zz,_=prep_feats(z,med)
        ph=win_model.predict_proba(zz)[:,1]
        local=np.maximum(.01,np.exp(np.clip(p_model.predict(zz),-3,8))*smear)
        p_tail=(tm.predict_proba(scaler.transform(zz))[:,1] if kind=="TAIL_COMPONENT"
                else np.zeros(len(z)))
        if kind=="TAIL_COMPONENT":
            # Decompose expectation: non-tail winners + rare large payout events.
            # P_tail is not conditioned on P_hit (both are unconditional).
            expect=np.maximum(ph*conditional_baseline
                         + p_tail*max(0,conditional_tail-conditional_baseline),0)
        else:
            expect=ph*local
        outs[name]={"expected":np.clip(expect,0,500),
                    "tail_risk":p_tail,"p_hit":ph,"conditional":local}
    return outs

def calc(df,skip):
    y=df["return"].to_numpy(float)
    s=df["stake"].to_numpy(float)
    kept=~skip
    net=float(np.sum(y[kept]-s[kept]))
    roi=float(100*y[kept].sum()/s[kept].sum()) if kept.sum() else float("nan")
    lost=y[skip];losers=y[skip][y[skip]>0]
    top=np.sort(losers)[::-1]
    return {
      "races":len(df),"bet_races":int(kept.sum()),"skip_races":int(skip.sum()),
      "skip_pct":round(100*skip.mean(),5),
      "stake_after_yen":float(s[kept].sum()),
      "return_after_yen":float(y[kept].sum()),
      "profit_after_yen":net,"roi_after_pct":round(roi,6),
      "profit_before_yen":float(np.sum(y-s)),
      "roi_before_pct":round(100*y.sum()/s.sum(),6),
      "skip_gain_yen":float(np.sum(s[skip]-y[skip])),
      "miss_races_skipped":int(np.sum(y[skip]==0)),
      "hit_races_lost":int((y[skip]>0).sum()),
      "payout_lost_yen":float(lost.sum()),
      "top1_lost_yen":float(top[:1].sum()),
      "top3_lost_yen":float(top[:3].sum()),
      "top5_lost_yen":float(top[:5].sum()),
      "top10_lost_yen":float(top[:10].sum()),
      "share_gross_payout_kept_pct":round(100*y[kept].sum()/y.sum(),5)
           if y.sum()>0 else 0.0,
    }

def guarded_mask(scores,guard,cal_preds,test_preds,rate):
    # A numerical score threshold is fixed using *strictly historical*
    # calibration scores, no evaluated year's score rankings or labels.
    old=np.asarray(scores,dtype=float)
    tr=np.asarray(cal_preds["expected"],dtype=float)
    te=np.asarray(test_preds["expected"],dtype=float)
    if rate==0:return np.zeros(len(te),dtype=bool)
    cal_score=np.asarray(cal_preds["skip_priority"],dtype=float)
    cut=float(np.quantile(cal_score,1-rate/100))
    mask=old>cut
    if guard=="NONE":return mask
    if guard.startswith("GUARD_TAIL") or guard=="GUARD_COMBINED_TOP20":
        # Hold out high P(high payout) from historical score quantile.
        qp=.90 if guard=="GUARD_TAIL_TOP10" else .80
        h=float(np.quantile(cal_preds["tail_risk"],qp))
        tailprotect=np.asarray(test_preds["tail_risk"])>h
    else:tailprotect=np.zeros(len(te),dtype=bool)
    if guard=="GUARD_EXPECTED_TOP20" or guard=="GUARD_COMBINED_TOP20":
        ec=float(np.quantile(tr,.80))
        prprotect=np.asarray(te)>ec
    else:prprotect=np.zeros(len(te),dtype=bool)
    return mask & ~(tailprotect|prprotect)

def process_one(job,joined,scores):
    rule,year=job
    whole=joined[joined.rule.eq(rule)]
    tr,te,proper,cal=age_split(whole,year)
    te=te.reset_index(drop=True)
    xscore=scores[(scores.rule.eq(rule))&(scores.year.eq(year))&
                  (scores.model.eq("UTILITY_REGRESSOR"))][["race_id","skip_score"]]
    prev=te[["race_id"]].merge(xscore,on="race_id",validate="one_to_one",
                              how="left")["skip_score"].to_numpy(float)
    check(len(prev)==len(te) and np.isfinite(prev).all(),f"previous scores missing {job}")
    # The prior baseline's cutoff cannot be recalibrated from a model lacking
    # its fit state. Reconstruct its exact frozen flags using the previous
    # saved annual report for a *separate reference* (done outside).
    # New model calibration is genuine chronological in-sample held-out data.
    tailcal=model_predictions(proper,{"cal":cal,"test":te},"TAIL_COMPONENT")
    hguard_cal=tailcal["cal"]
    hguard_test=tailcal["test"]
    batch=[]
    estimates=[]
    for kind in METHODS:
        if kind=="PREV_UTILITY":
            # Do not use retrospective test-year percentile. Previous result
            # baseline represented separately and reconstructed via prior report.
            continue
        if kind=="TAIL_COMPONENT":
            calibration=tailcal["cal"]
            test=hguard_test
        else:
            calibration=model_predictions(proper,{"cal":cal},kind)["cal"]
            test=model_predictions(tr,{"test":te},kind)["test"]
        # For tail guard, use same frozen calibration model and test-year model
        # (training vintage may differ), so guard thresholds are historic-only.
        trrisk=hguard_cal["tail_risk"]
        terisk=hguard_test["tail_risk"]
        cc={"expected":calibration["expected"],
            "skip_priority":-calibration["expected"],"tail_risk":trrisk}
        tt={"expected":test["expected"],
            "skip_priority":-test["expected"],"tail_risk":terisk}
        estimates.append(pd.DataFrame({
           "year":year,"race_id":te.race_id,"rule":rule,"method":kind,
           "expected_return_per_stake":tt["expected"],
           "payout_tail_probability":tt["tail_risk"],
           "realized_return_per_stake":ratio(te)
        }))
        for rate in SKIP_RATE:
            for guard in POLICIES:
                if rate==0 and guard!="NONE":continue
                if guard!="NONE" and kind=="TAIL_COMPONENT":
                    # Homogeneous tail model used for both probability / payoff.
                    pass
                mask=guarded_mask(tt["skip_priority"],guard,cc,tt,rate)
                batch.append({
                   "year":year,"rule":rule,"method":kind,
                   "target_skip_pct":rate,"payout_guard":guard,
                   **calc(te,mask),
                })
    return pd.DataFrame(batch),pd.concat(estimates,ignore_index=True)

def payout_bands(joined):
    out=[]
    for year in YEARS:
      for rule in RULES:
        z=joined[(joined.year.eq(year))&(joined.rule.eq(rule))]
        winners=z[z["return"]>0].copy()
        if len(winners)==0:continue
        wins=winners["return"].to_numpy(float)
        wins.sort()
        wins=wins[::-1]
        for n in (1,3,5,10,20,50):
            out.append({
                "year":year,"rule":rule,"highest_hit_count":n,
                "hit_races":len(winners),
                "top_n_hit_payout_yen":float(wins[:n].sum()),
                "all_payout_yen":float(z["return"].sum()),
                "top_n_share_pct":round(100*wins[:n].sum()/z["return"].sum(),5),
            })
    return pd.DataFrame(out)

def old_v1_row(joined,oldreport):
    old=oldreport[
       oldreport.policy.eq("PRIOR_ONLY_CUTOFF")&
       oldreport.model.eq("UTILITY_REGRESSOR")&
       oldreport.rule.isin(RULES)&oldreport.year.isin(YEARS)]
    return old

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--prior-root",required=True)
    pa.add_argument("--output",required=True)
    args=pa.parse_args()
    root=Path(args.prior_root)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(root/"race-feature-snapshot.csv.gz",dtype={"race_id":str})
    p=pd.read_csv(root/"all-10-rules-settled-per-race.csv.gz",dtype={"race_id":str})
    oscore=pd.read_csv(root/"per-race-oos-skip-scores.csv.gz",dtype={"race_id":str})
    old=pd.read_csv(root/"all-model-year-cutoff-and-quota.csv")
    joined=p[p.rule.isin(RULES)].merge(f,on=["year","race_id"],validate="many_to_one")
    check(len(joined)==10368*len(RULES),"payoff join data missing")
    check(f.year.eq(2026).sum()==0 and set(f.year)=={2023,2024,2025},"locked year violation")
    check(joined.groupby(["year","rule"]).size().eq(3456).all(),"year/rule gaps")
    check(f[list(FEATURE_NAMES)].notna().all().all(),"feature NaNs")
    jobs=[(r,y) for r in RULES for y in YEARS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        results=list(ex.map(lambda job:process_one(job,joined,oscore),jobs))
    table=pd.concat([x[0] for x in results],ignore_index=True)
    pred=pd.concat([x[1] for x in results],ignore_index=True)
    check(table.groupby(["year","rule","method","target_skip_pct","payout_guard"]).size().max()==1,
          "duplicate experiment rows")
    check(len(table)>1000,"expected full batch missing")
    top=payout_bands(joined)
    # Independent 2024-only policy selection for different purchase coverage:
    picks=[]
    for name,cover in (("BROAD_90",.90),("BROAD_80",.80),("MODERATE_70",.70)):
        acceptable=table[(table.year.eq(2024))&
                   (table.bet_races.ge(int(math.ceil(3456*cover))))&
                   (table.target_skip_pct.le(30))&
                   (table.target_skip_pct.ge(5))]
        check(len(acceptable)>0,"missing 2024 candidates")
        selected=acceptable.sort_values(
            ["profit_after_yen","roi_after_pct","bet_races"],
            ascending=[False,False,False]).iloc[0]
        confirmed=table[(table.year.eq(2025))&
                   table.rule.eq(selected["rule"])&
                   table.method.eq(selected["method"])&
                   table.target_skip_pct.eq(selected["target_skip_pct"])&
                   table.payout_guard.eq(selected["payout_guard"])]
        check(len(confirmed)==1,"chosen 2024 policy absent in 2025")
        verified=confirmed.iloc[0]
        picks.append({
            "purchase_mandate":name,
            "rule":selected["rule"],
            "method":selected["method"],
            "target_skip_pct":int(selected["target_skip_pct"]),
            "payout_guard":selected["payout_guard"],
            "2024_bet_races":int(selected["bet_races"]),
            "2024_profit_yen":float(selected["profit_after_yen"]),
            "2024_roi_pct":float(selected["roi_after_pct"]),
            "2025_bet_races":int(verified["bet_races"]),
            "2025_profit_yen":float(verified["profit_after_yen"]),
            "2025_roi_pct":float(verified["roi_after_pct"]),
        })
    # Evaluation of pre-existing rather than re-selected rule: TRIO 20 skip
    # from prior frozen run. It is not a pristine holdout either.
    oldcomp=old_v1_row(joined,old)
    byyear=[]
    for yr in YEARS:
        valid=table[(table.year.eq(yr))&(table.target_skip_pct.gt(0)) &
                    (table.bet_races.ge(3456*.7))]
        leads=valid.sort_values(["profit_after_yen","roi_after_pct"],
                         ascending=[False,False]).head(12)
        byyear.extend(leads.to_dict("records"))
    table.to_csv(out/"new-models-risk-utility-hurdle-tail-frontier.csv",index=False)
    pred.to_csv(out/"new-oos-price-tail-and-payoff-scores.csv.gz",index=False,
                compression="gzip")
    top.to_csv(out/"paid-tail-concentration-by-year.csv",index=False)
    oldcomp.to_csv(out/"prior-frozen-utility-model-baselines.csv",index=False)
    pd.DataFrame(picks).to_csv(out/"2024-selected-locked-2025.csv",index=False)
    pd.DataFrame(byyear).to_csv(out/"exploratory-best-per-year.csv",index=False)
    summary={
      "contract":"L2_PAYOFF_TAIL_PROTECTION_V1",
      "source_reused":SOURCE_RUN,
      "total_races_2023_to_2025":10368,
      "years_evaluated":list(YEARS),
      "rules":list(RULES),"methods":list(METHODS),
      "protection_rules":list(POLICIES),
      "skip_rates_pct":list(SKIP_RATE),
      "n_candidate_evaluations":len(table),
      "temporal_cutoff":"Prior-only OOS calibration: last ~22% of previous calendar year, NEVER current-year labels/score percentile",
      "no_test_year_labels_in_training":True,
      "2024_selection_2025_checked":picks,
      "caution":"2025 repeatedly inspected in earlier studies; not a pristine holdout; final odds in original anchor require real-time replay before betting.",
      "current_live_model_promotion":False,
      "2026_sealed":True,
      "highest_2024_2025_exploratory_results":byyear[:8],
      "no_new_downloads":True,
      "free_cpu":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,
                                  default=str,indent=2)+"\n",encoding="utf-8")
    print("L2_PAYOFF_TAIL_PROTECTION_V1_SUMMARY",flush=True)
    print(json.dumps(picks,ensure_ascii=False),flush=True)
    print("TOP_YEARLY_EXPLORATORY",flush=True)
    print(pd.DataFrame(byyear)[["year","rule","method","payout_guard",
          "target_skip_pct","bet_races","roi_after_pct","profit_after_yen"]].to_csv(
              index=False),flush=True)
    print("L2_PAYOFF_TAIL_PROTECTION_V1_COMPLETE",flush=True)

if __name__=="__main__":
    main()
