#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,brier_score_loss,log_loss,roc_auc_score,
)

POLICIES=("BASE","FULL_K2","FULL_K3")
TEST_YEARS=(2023,2024,2025)
STRATEGIES=(
    "WIN_CORE2",
    "QUINELLA_AXIS1",
    "EXACTA_MULTI1",
    "TRIO_AXIS12",
    "TRIO_AXIS1",
    "TRIFECTA_MULTI12",
)
THRESHOLDS={
    "ALL":None,
    "EDGE_GT_0":0.0,
    "EDGE_GT_005":0.05,
    "EDGE_GT_010":0.10,
}
SEGMENTS={
    "ALL_ALERTS":None,
    "TRUE_BLIND":True,
    "FALSE_ALERT":False,
}
CATEGORICAL={
    "strategy","bet_type","venue_code","surface","race_class",
    "discipline","direction","weather","track_condition",
}
EXCLUDE={
    "contract","year","race_id","race_date","policy",
    "selection_key","selection_numbers","selection_horse_ids",
    "blind","hit","return_yen_per100",
}
FORBIDDEN_FEATURE_TOKENS=(
    "blind","hit","return","payout","finish","target","winner",
)


def parse_args():
    p=argparse.ArgumentParser(description="Walk-forward L2 profit test: BASE vs FULL K2/K3 candidate pools.")
    p.add_argument("--tickets",required=True)
    p.add_argument("--coverage",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def load_rows(path):
    rows=[]
    op=gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("empty ticket dataset")
    return pd.DataFrame(rows)


def safe_auc(y,p):
    try:
        return float(roc_auc_score(y,p)) if len(set(map(int,y)))>1 else None
    except ValueError:
        return None


def safe_ap(y,p):
    try:
        return float(average_precision_score(y,p)) if len(set(map(int,y)))>1 else None
    except ValueError:
        return None


def clip_prob(p):
    return np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)


def logit(p):
    p=clip_prob(p)
    return np.log(p/(1-p))


def feature_columns(df):
    cols=[c for c in df.columns if c not in EXCLUDE]
    bad=[c for c in cols if any(tok in c.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise ValueError(f"forbidden outcome-derived feature columns: {bad}")
    return cols


def encode_fit_other(fit,others,cols):
    cats=[c for c in cols if c in CATEGORICAL]
    nums=[c for c in cols if c not in cats]

    fnum=fit[nums].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    fc=pd.get_dummies(
        fit[cats].astype("string").fillna("__MISSING__"),
        columns=cats,dummy_na=False,dtype=float,
    ) if cats else pd.DataFrame(index=fit.index)
    xf=pd.concat([fnum.reset_index(drop=True),fc.reset_index(drop=True)],axis=1)

    out=[]
    for frame in others:
        onum=frame[nums].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(-999.0)
        oc=pd.get_dummies(
            frame[cats].astype("string").fillna("__MISSING__"),
            columns=cats,dummy_na=False,dtype=float,
        ) if cats else pd.DataFrame(index=frame.index)
        xo=pd.concat([onum.reset_index(drop=True),oc.reset_index(drop=True)],axis=1)
        xo=xo.reindex(columns=xf.columns,fill_value=0.0)
        out.append(xo)
    return xf,out


def chronological_split(train):
    races=(
        train[["race_id","race_date"]]
        .drop_duplicates()
        .sort_values(["race_date","race_id"])
        .reset_index(drop=True)
    )
    if len(races)<50:
        raise ValueError(f"not enough training races for calibration split: {len(races)}")
    cut=max(1,min(len(races)-1,int(len(races)*0.80)))
    fit_ids=set(races.iloc[:cut]["race_id"].astype(str))
    cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
    fit=train[train["race_id"].astype(str).isin(fit_ids)].copy()
    cal=train[train["race_id"].astype(str).isin(cal_ids)].copy()
    if fit.empty or cal.empty:
        raise ValueError("empty fit/cal split")
    return fit,cal,len(fit_ids),len(cal_ids)


def make_model(seed):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=180,
        learning_rate=0.03,
        num_leaves=23,
        min_child_samples=100,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=3.0,
        reg_alpha=0.3,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def calibrate_sigmoid(y_cal,p_cal,p_test):
    y=np.asarray(y_cal,dtype=int)
    if len(set(y.tolist()))<2:
        return clip_prob(p_test),{"mode":"identity","reason":"single_class_calibration"}
    x=logit(p_cal).reshape(-1,1)
    model=LogisticRegression(C=1.0,solver="lbfgs",max_iter=200)
    model.fit(x,y)
    out=model.predict_proba(logit(p_test).reshape(-1,1))[:,1]
    return clip_prob(out),{
        "mode":"sigmoid",
        "coef":float(model.coef_[0][0]),
        "intercept":float(model.intercept_[0]),
    }


def model_quality(y,p):
    y=np.asarray(y,dtype=int)
    p=clip_prob(p)
    return {
        "rows":int(len(y)),
        "positives":int(y.sum()),
        "positive_rate":float(y.mean()) if len(y) else None,
        "roc_auc":safe_auc(y,p),
        "pr_auc":safe_ap(y,p),
        "brier":float(brier_score_loss(y,p)),
        "log_loss":float(log_loss(y,p,labels=[0,1])),
    }


def max_drawdown(selected):
    if selected.empty:
        return 0.0,0.0
    race=(
        selected.assign(
            _stake=100.0,
            _ret=pd.to_numeric(selected["return_yen_per100"],errors="coerce").fillna(0.0),
        )
        .groupby(["race_date","race_id"],as_index=False)
        .agg(stake=("_stake","sum"),ret=("_ret","sum"))
        .sort_values(["race_date","race_id"])
    )
    cumulative=0.0
    peak=0.0
    max_dd=0.0
    positive=0
    for row in race.itertuples(index=False):
        profit=float(row.ret-row.stake)
        if profit>0:
            positive+=1
        cumulative+=profit
        peak=max(peak,cumulative)
        max_dd=max(max_dd,peak-cumulative)
    positive_rate=positive/len(race) if len(race) else 0.0
    return max_dd,positive_rate


def segment_frame(frame,segment):
    flag=SEGMENTS[segment]
    if flag is None:
        return frame
    return frame[frame["blind"].astype(bool)==flag]


def metric_row(frame,policy,test_year,segment,strategy,filter_name):
    base=segment_frame(frame,segment)
    if strategy=="PORTFOLIO_DEDUP":
        ordered=base.sort_values(
            ["race_date","race_id","bet_type","selection_key","edge","strategy"],
            ascending=[True,True,True,True,False,True],
        )
        base=ordered.drop_duplicates(
            subset=["race_id","bet_type","selection_key"],keep="first"
        )
    else:
        base=base[base["strategy"]==strategy]
    alert_races=int(base["race_id"].nunique())
    if filter_name=="ALL":
        selected=base
    else:
        threshold=THRESHOLDS[filter_name]
        selected=base[base["edge"]>float(threshold)]
    bought=int(len(selected))
    bought_races=int(selected["race_id"].nunique())
    hit_tickets=int(selected["hit"].astype(bool).sum()) if bought else 0
    stake=100*bought
    ret=float(pd.to_numeric(selected["return_yen_per100"],errors="coerce").fillna(0.0).sum()) if bought else 0.0
    race_hit=0
    if bought:
        race_hit=int(
            selected.groupby("race_id")["hit"].any().sum()
        )
    dd,positive_rate=max_drawdown(selected)
    race_pool=(
        base[["race_id","pool_size","novel_pool_count"]]
        .drop_duplicates("race_id")
    )
    return {
        "policy":policy,
        "test_year":str(test_year),
        "segment":segment,
        "strategy":strategy,
        "filter":filter_name,
        "alert_races":alert_races,
        "priced_tickets":int(len(base)),
        "priced_tickets_per_alert":len(base)/alert_races if alert_races else 0.0,
        "bought_tickets":bought,
        "buy_rate":bought/len(base) if len(base) else 0.0,
        "bought_races":bought_races,
        "skip_races":max(0,alert_races-bought_races),
        "buy_race_rate":bought_races/alert_races if alert_races else 0.0,
        "hit_tickets":hit_tickets,
        "ticket_hit_rate":hit_tickets/bought if bought else 0.0,
        "hit_races":race_hit,
        "race_hit_rate":race_hit/bought_races if bought_races else 0.0,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "avg_ticket_odds":float(selected["odds"].mean()) if bought else None,
        "avg_predicted_probability":float(selected["predicted_probability"].mean()) if bought else None,
        "avg_edge":float(selected["edge"].mean()) if bought else None,
        "avg_pool_size":float(race_pool["pool_size"].mean()) if len(race_pool) else None,
        "avg_novel_pool_count":float(race_pool["novel_pool_count"].mean()) if len(race_pool) else None,
        "avg_bought_tickets_per_alert":bought/alert_races if alert_races else 0.0,
        "max_drawdown_yen":dd,
        "positive_race_rate":positive_rate,
    }


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    a=parse_args()
    coverage=json.loads(Path(a.coverage).read_text(encoding="utf-8"))
    if coverage.get("ability_uses_odds") is not False or coverage.get("locked_years")!=[2026]:
        raise SystemExit("L1 odds/2026 guard broken")
    if coverage.get("place_bets_included") is not False:
        raise SystemExit("PLACE bet prohibition broken")

    df=load_rows(a.tickets)
    if set(df["policy"].unique())!=set(POLICIES):
        raise SystemExit(f"policy coverage mismatch: {sorted(df['policy'].unique())}")
    if any(int(y)>=2026 for y in df["year"].unique()):
        raise SystemExit("2026 leaked into L2 research")
    df["race_month"]=df["race_date"].astype(str).str.slice(5,7).replace("",np.nan)
    df["race_month"]=pd.to_numeric(df["race_month"],errors="coerce").fillna(0).astype(float)
    df["hit"]=df["hit"].astype(bool)
    df["blind"]=df["blind"].astype(bool)
    df["odds"]=pd.to_numeric(df["odds"],errors="coerce")
    df["return_yen_per100"]=pd.to_numeric(df["return_yen_per100"],errors="coerce").fillna(0.0)

    cols=feature_columns(df)
    out_dir=Path(a.out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)

    quality_rows=[]
    all_predictions={p:[] for p in POLICIES}
    fold_meta=[]

    for pidx,policy in enumerate(POLICIES):
        pdf=df[df["policy"]==policy].copy()
        for yidx,test_year in enumerate(TEST_YEARS):
            train=pdf[pdf["year"]<test_year].copy()
            test=pdf[pdf["year"]==test_year].copy()
            if train.empty or test.empty:
                raise ValueError(f"empty train/test policy={policy} test_year={test_year}")
            fit,cal,fit_races,cal_races=chronological_split(train)
            xfit,(xcal,xtest)=encode_fit_other(fit,[cal,test],cols)
            yfit=fit["hit"].astype(int).to_numpy()
            ycal=cal["hit"].astype(int).to_numpy()
            ytest=test["hit"].astype(int).to_numpy()
            if len(set(yfit.tolist()))<2:
                raise ValueError(f"single-class fit policy={policy} year={test_year}")

            model=make_model(31000+pidx*1000+yidx)
            model.fit(xfit,yfit)
            pcal0=model.predict_proba(xcal)[:,1]
            ptest0=model.predict_proba(xtest)[:,1]
            ptest,cal_meta=calibrate_sigmoid(ycal,pcal0,ptest0)

            pred=test.copy()
            pred["predicted_probability"]=ptest
            pred["edge"]=pred["predicted_probability"]*pred["odds"]-1.0
            all_predictions[policy].append(pred)

            q=model_quality(ytest,ptest)
            quality_rows.append({
                "policy":policy,
                "test_year":test_year,
                "train_rows":len(train),
                "fit_rows":len(fit),
                "calibration_rows":len(cal),
                "fit_races":fit_races,
                "calibration_races":cal_races,
                **q,
                "calibration_mode":cal_meta["mode"],
                "calibration_coef":cal_meta.get("coef"),
                "calibration_intercept":cal_meta.get("intercept"),
                "feature_count":int(xfit.shape[1]),
            })
            fold_meta.append({
                "policy":policy,
                "test_year":test_year,
                "train_years":sorted(map(int,train["year"].unique())),
                "fit_races":fit_races,
                "calibration_races":cal_races,
                "test_races":int(test["race_id"].nunique()),
                "calibration":cal_meta,
            })

    strategy_rows=[]
    policy_rows=[]
    for policy in POLICIES:
        yearly=all_predictions[policy]
        for pred in yearly:
            test_year=int(pred["year"].iloc[0])
            for segment in SEGMENTS:
                for strategy in STRATEGIES:
                    for filter_name in THRESHOLDS:
                        strategy_rows.append(metric_row(
                            pred,policy,test_year,segment,strategy,filter_name
                        ))
                for filter_name in THRESHOLDS:
                    policy_rows.append(metric_row(
                        pred,policy,test_year,segment,"PORTFOLIO_DEDUP",filter_name
                    ))
        combined=pd.concat(yearly,ignore_index=True)
        for segment in SEGMENTS:
            for strategy in STRATEGIES:
                for filter_name in THRESHOLDS:
                    strategy_rows.append(metric_row(
                        combined,policy,"ALL",segment,strategy,filter_name
                    ))
            for filter_name in THRESHOLDS:
                policy_rows.append(metric_row(
                    combined,policy,"ALL",segment,"PORTFOLIO_DEDUP",filter_name
                ))

    write_csv(out_dir/"model-quality.csv",quality_rows)
    write_csv(out_dir/"strategy-summary.csv",strategy_rows)
    write_csv(out_dir/"policy-summary.csv",policy_rows)

    # Compact comparison rows for the main BASE/K2/K3 decision.
    comparison=[
        row for row in policy_rows
        if row["test_year"]=="ALL" and row["segment"]=="ALL_ALERTS"
    ]
    write_csv(out_dir/"comparison.csv",comparison)

    summary={
        "contract":"L2_CANDIDATE_PROFIT_V1",
        "policies":list(POLICIES),
        "test_years":list(TEST_YEARS),
        "training_rule":"walk-forward; each test year uses only prior years",
        "calibration_rule":"last 20% of prior training races by date reserved for sigmoid calibration",
        "ticket_stake_yen":100,
        "market_input":"historical final odds",
        "market_caveat":"FINAL_ODDS_PROXY is valid for historical comparison but is not yet a live timestamp execution simulation",
        "bet_types":["WIN","QUINELLA","EXACTA","TRIO","TRIFECTA"],
        "place_bets":False,
        "wide_bets":False,
        "filters":THRESHOLDS,
        "segments":list(SEGMENTS),
        "ability_uses_odds":False,
        "l2_uses_odds":True,
        "locked_years":[2026],
        "folds":fold_meta,
        "coverage":coverage,
    }
    (out_dir/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    lines=[
        "# L2 Candidate Profit V1",
        "",
        "- Comparison: Seven-King BASE vs Seven-King + FULL K2 vs Seven-King + FULL K3",
        "- Scope: Consensus-World Gate alerts only",
        "- Walk-forward test years: 2023, 2024, 2025",
        "- L1/L1.5 never use odds; L2 joins historical final odds.",
        "- Bet types: WIN / QUINELLA / EXACTA / TRIO / TRIFECTA. PLACE is excluded.",
        "- Final odds are a historical execution-price proxy, not a live timestamp simulation.",
        "- Stake: 100 yen per purchased ticket; no L3 bankroll optimization here.",
        "",
        "Main comparison is in comparison.csv. Strategy-level detail is in strategy-summary.csv.",
        "No production policy is promoted automatically from this test.",
        "",
    ]
    (out_dir/"README.md").write_text("\n".join(lines),encoding="utf-8")

    print("L2_CANDIDATE_PROFIT_V1_READY")
    print(json.dumps({
        "rows":len(df),
        "policies":list(POLICIES),
        "test_years":list(TEST_YEARS),
        "comparison_rows":len(comparison),
        "out_dir":str(out_dir),
    },ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
