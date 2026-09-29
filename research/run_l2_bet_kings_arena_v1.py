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
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss,roc_auc_score

TEST_YEARS=(2023,2024,2025)
DEV_YEARS=(2023,2024)
HOLDOUT_YEAR=2025
BET_TYPES=("WIN","QUINELLA","EXACTA","TRIO","TRIFECTA")

IDENTITY_COLUMNS={
    "contract","year","race_id","race_date","bet_type","template",
    "selection_key","selection_numbers","selection_horse_ids",
}
MARKET_OR_TARGET_COLUMNS={
    "hit","return_yen_per100","odds",
}
CATEGORICAL={
    "venue_code","surface","race_class","discipline","direction","weather",
    "track_condition","selected_outsider_1","selected_outsider_2",
}
FORBIDDEN_TOKENS=(
    "odds","implied","payout","return","profit","roi","hit","finish","winner",
    "blind","popularity",
)


def parse_args():
    p=argparse.ArgumentParser(description="L2 Bet Kings Arena V1: odds-free P(hit), then market edge.")
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--contract",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_template(path):
    rows=[]
    with gzip.open(path,"rt",encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty template dataset: {path}")
    return pd.DataFrame(rows)


def clip_prob(values):
    return np.clip(np.asarray(values,dtype=float),1e-6,1-1e-6)


def logit(values):
    p=clip_prob(values)
    return np.log(p/(1-p))


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


def feature_columns(df):
    excluded=IDENTITY_COLUMNS|MARKET_OR_TARGET_COLUMNS
    cols=[c for c in df.columns if c not in excluded]
    bad=[
        c for c in cols
        if any(token in c.lower() for token in FORBIDDEN_TOKENS)
    ]
    if bad:
        raise ValueError(f"FORBIDDEN_MARKET_OR_OUTCOME_FEATURES: {sorted(bad)}")
    if "odds" in cols or "return_yen_per100" in cols or "hit" in cols:
        raise ValueError("market/target feature guard broken")
    return cols


def encode_fit_other(fit,others,cols):
    cats=[c for c in cols if c in CATEGORICAL]
    nums=[c for c in cols if c not in cats]
    fnum=(
        fit[nums].apply(pd.to_numeric,errors="coerce")
        .replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    )
    fc=(
        pd.get_dummies(
            fit[cats].astype("string").fillna("__MISSING__"),
            columns=cats,dummy_na=False,dtype=float,
        )
        if cats else pd.DataFrame(index=fit.index)
    )
    xf=pd.concat([fnum.reset_index(drop=True),fc.reset_index(drop=True)],axis=1)
    out=[]
    for frame in others:
        onum=(
            frame[nums].apply(pd.to_numeric,errors="coerce")
            .replace([np.inf,-np.inf],np.nan).fillna(-999.0)
        )
        oc=(
            pd.get_dummies(
                frame[cats].astype("string").fillna("__MISSING__"),
                columns=cats,dummy_na=False,dtype=float,
            )
            if cats else pd.DataFrame(index=frame.index)
        )
        xo=pd.concat([onum.reset_index(drop=True),oc.reset_index(drop=True)],axis=1)
        out.append(xo.reindex(columns=xf.columns,fill_value=0.0))
    return xf,out


def chronological_split(train):
    races=(
        train[["race_id","race_date"]].drop_duplicates()
        .sort_values(["race_date","race_id"]).reset_index(drop=True)
    )
    if len(races)<30:
        return None
    cut=max(1,min(len(races)-1,int(len(races)*0.80)))
    fit_ids=set(races.iloc[:cut]["race_id"].astype(str))
    cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
    fit=train[train["race_id"].astype(str).isin(fit_ids)].copy()
    cal=train[train["race_id"].astype(str).isin(cal_ids)].copy()
    if fit.empty or cal.empty:
        return None
    return fit,cal,len(fit_ids),len(cal_ids)


def make_model(seed):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=180,
        learning_rate=0.03,
        num_leaves=23,
        min_child_samples=50,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=3.0,
        reg_alpha=0.3,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def calibrate(y_cal,p_cal,p_test):
    y=np.asarray(y_cal,dtype=int)
    if len(set(y.tolist()))<2:
        return clip_prob(p_test),{"mode":"identity","reason":"single_class_calibration"}
    model=LogisticRegression(C=1.0,solver="lbfgs",max_iter=200)
    model.fit(logit(p_cal).reshape(-1,1),y)
    pred=model.predict_proba(logit(p_test).reshape(-1,1))[:,1]
    return clip_prob(pred),{
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
        positive+=int(profit>0)
        cumulative+=profit
        peak=max(peak,cumulative)
        max_dd=max(max_dd,peak-cumulative)
    return max_dd,(positive/len(race) if len(race) else 0.0)


def concentration(selected):
    if selected.empty:
        return {
            "top1_return_share_pct":None,
            "top5_return_share_pct":None,
            "roi_without_top1_return_pct":None,
        }
    race=(
        selected.assign(
            _stake=100.0,
            _ret=pd.to_numeric(selected["return_yen_per100"],errors="coerce").fillna(0.0),
        )
        .groupby("race_id",as_index=False)
        .agg(stake=("_stake","sum"),ret=("_ret","sum"))
    )
    total_ret=float(race["ret"].sum())
    total_stake=float(race["stake"].sum())
    returns=sorted([float(x) for x in race["ret"]],reverse=True)
    top1=returns[0] if returns else 0.0
    top5=sum(returns[:5])
    return {
        "top1_return_share_pct":100*top1/total_ret if total_ret>0 else None,
        "top5_return_share_pct":100*top5/total_ret if total_ret>0 else None,
        "roi_without_top1_return_pct":(
            100*max(0.0,total_ret-top1)/total_stake if total_stake>0 else None
        ),
    }


def metric_row(pred,bet_type,template,test_year,threshold):
    selected=pred[pred["edge"]>float(threshold)].copy()
    source_races=int(pred["race_id"].nunique())
    bought=int(len(selected))
    bought_races=int(selected["race_id"].nunique())
    stake=100*bought
    ret=float(
        pd.to_numeric(selected["return_yen_per100"],errors="coerce").fillna(0.0).sum()
    ) if bought else 0.0
    hit_tickets=int(selected["hit"].astype(bool).sum()) if bought else 0
    hit_races=int(selected.groupby("race_id")["hit"].any().sum()) if bought else 0
    dd,positive=max_drawdown(selected)
    conc=concentration(selected)
    return {
        "bet_type":bet_type,
        "template":template,
        "test_year":test_year,
        "edge_threshold":threshold,
        "source_races":source_races,
        "priced_tickets":int(len(pred)),
        "bought_tickets":bought,
        "bought_races":bought_races,
        "skip_races":max(0,source_races-bought_races),
        "buy_ticket_rate":bought/len(pred) if len(pred) else 0.0,
        "buy_race_rate":bought_races/source_races if source_races else 0.0,
        "hit_tickets":hit_tickets,
        "ticket_hit_rate":hit_tickets/bought if bought else 0.0,
        "hit_races":hit_races,
        "race_hit_rate":hit_races/bought_races if bought_races else 0.0,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "max_drawdown_yen":dd,
        "positive_race_rate":positive,
        "avg_final_odds":float(selected["odds"].mean()) if bought else None,
        "avg_predicted_probability":float(selected["predicted_probability"].mean()) if bought else None,
        "avg_edge":float(selected["edge"].mean()) if bought else None,
        "gate_alert_ticket_share":float(selected["gate_alert"].mean()) if bought else None,
        "novel_ticket_share":float((selected["ticket_novel_count"]>0).mean()) if bought else None,
        **conc,
    }


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=keys,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def candidate_summary(metric_rows,bet_type,templates,thresholds,qualification):
    out=[]
    index={(r["template"],int(r["test_year"]),float(r["edge_threshold"])):r for r in metric_rows}
    for template in templates:
        for threshold in thresholds:
            yearly=[]
            for year in DEV_YEARS:
                row=index.get((template,year,float(threshold)))
                if row is not None:
                    yearly.append(row)
            if len(yearly)!=len(DEV_YEARS):
                out.append({
                    "bet_type":bet_type,"template":template,"edge_threshold":threshold,
                    "qualified":False,"reason":"MISSING_DEVELOPMENT_FOLD",
                })
                continue
            roi_ok=all((r["roi_pct"] is not None and r["roi_pct"]>=qualification["min_roi_pct_each_development_year"]) for r in yearly)
            ticket_ok=all(r["bought_tickets"]>=qualification["min_bought_tickets_each_development_year"] for r in yearly)
            race_ok=all(r["bought_races"]>=qualification["min_bought_races_each_development_year"] for r in yearly)
            qualified=roi_ok and ticket_ok and race_ok
            stake=sum(r["stake_yen"] for r in yearly)
            ret=sum(r["return_yen"] for r in yearly)
            out.append({
                "bet_type":bet_type,
                "template":template,
                "edge_threshold":threshold,
                "qualified":qualified,
                "reason":"QUALIFIED" if qualified else (
                    "ROI_GUARD" if not roi_ok else "MIN_TICKETS" if not ticket_ok else "MIN_RACES"
                ),
                "dev_profit_yen":sum(r["profit_yen"] for r in yearly),
                "dev_stake_yen":stake,
                "dev_return_yen":ret,
                "dev_roi_pct":100*ret/stake if stake else None,
                "min_year_roi_pct":min(float(r["roi_pct"]) for r in yearly if r["roi_pct"] is not None),
                "sum_max_drawdown_yen":sum(r["max_drawdown_yen"] for r in yearly),
                "dev_bought_tickets":sum(r["bought_tickets"] for r in yearly),
                "dev_bought_races":sum(r["bought_races"] for r in yearly),
                "roi_2023":yearly[0]["roi_pct"],
                "profit_2023":yearly[0]["profit_yen"],
                "roi_2024":yearly[1]["roi_pct"],
                "profit_2024":yearly[1]["profit_yen"],
            })
    return out


def choose_candidate(rows):
    complete=[r for r in rows if "dev_profit_yen" in r]
    if not complete:
        return None,"NO_DATA"
    qualified=[r for r in complete if r.get("qualified") is True]
    pool=qualified if qualified else complete
    pool=sorted(
        pool,
        key=lambda r:(
            -float(r.get("dev_profit_yen") or -1e30),
            -float(r.get("min_year_roi_pct") or -1e30),
            float(r.get("sum_max_drawdown_yen") or 1e30),
            int(r.get("dev_bought_tickets") or 10**18),
            str(r.get("template")),
            float(r.get("edge_threshold") or 0.0),
        ),
    )
    return pool[0],("CROWN_CANDIDATE" if qualified else "NO_QUALIFIED_KING")


def main():
    a=parse_args()
    contract=read_json(a.contract)
    manifest=read_json(Path(a.dataset_dir)/"manifest.json")
    if contract.get("contract")!="L2_BET_KINGS_ARENA_V1":
        raise SystemExit("wrong arena contract")
    if manifest.get("source_l15")!="L15_FIXED_V1":
        raise SystemExit("dataset is not based on fixed L1.5")
    if manifest.get("probability_model_uses_odds") is not False:
        raise SystemExit("dataset market separation guard broken")
    if contract["scope"]["locked_years"]!=[2026]:
        raise SystemExit("2026 lock guard broken")
    if contract["walk_forward"]["final_holdout_year"]!=HOLDOUT_YEAR:
        raise SystemExit("holdout contract drift")

    thresholds=[float(x) for x in contract["edge_thresholds"]]
    qualification=contract["king_selection"]["qualification"]
    out_dir=Path(a.out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)

    metric_rows=[]
    quality_rows=[]
    fold_rows=[]
    dev_candidates=[]
    king_rows=[]

    templates_by_bet={bet:[] for bet in BET_TYPES}
    for template,info in manifest["templates"].items():
        templates_by_bet[info["bet_type"]].append(template)

    for bet_index,bet_type in enumerate(BET_TYPES):
        bet_candidate_rows=[]
        for template_index,template in enumerate(sorted(templates_by_bet[bet_type])):
            path=Path(a.dataset_dir)/manifest["templates"][template]["file"]
            df=load_template(path)
            if set(df["template"].astype(str).unique())!={template}:
                raise ValueError(f"template file contamination: {template}")
            if set(df["bet_type"].astype(str).unique())!={bet_type}:
                raise ValueError(f"bet type contamination: {template}")
            if any(int(y)>=2026 for y in df["year"].unique()):
                raise ValueError(f"2026 leakage in {template}")
            df["hit"]=df["hit"].astype(bool)
            df["odds"]=pd.to_numeric(df["odds"],errors="coerce")
            df["return_yen_per100"]=pd.to_numeric(
                df["return_yen_per100"],errors="coerce"
            ).fillna(0.0)
            df["race_month"]=pd.to_numeric(
                df["race_date"].astype(str).str.slice(5,7),errors="coerce"
            ).fillna(0.0)
            cols=feature_columns(df)
            if any("odds" in c.lower() for c in cols):
                raise ValueError(f"ODDS_FEATURE_LEAK template={template}: {cols}")

            predictions={}
            for year_index,test_year in enumerate(TEST_YEARS):
                train=df[df["year"]<test_year].copy()
                test=df[df["year"]==test_year].copy()
                if train.empty or test.empty:
                    fold_rows.append({
                        "bet_type":bet_type,"template":template,"test_year":test_year,
                        "status":"MISSING_TRAIN_OR_TEST",
                    })
                    continue
                split=chronological_split(train)
                if split is None:
                    fold_rows.append({
                        "bet_type":bet_type,"template":template,"test_year":test_year,
                        "status":"INSUFFICIENT_RACES",
                    })
                    continue
                fit,cal,fit_races,cal_races=split
                yfit=fit["hit"].astype(int).to_numpy()
                ycal=cal["hit"].astype(int).to_numpy()
                ytest=test["hit"].astype(int).to_numpy()
                if len(set(yfit.tolist()))<2:
                    fold_rows.append({
                        "bet_type":bet_type,"template":template,"test_year":test_year,
                        "status":"SINGLE_CLASS_FIT","fit_positives":int(yfit.sum()),
                    })
                    continue

                xfit,(xcal,xtest)=encode_fit_other(fit,[cal,test],cols)
                model=make_model(71000+bet_index*10000+template_index*100+year_index)
                model.fit(xfit,yfit)
                pcal0=model.predict_proba(xcal)[:,1]
                ptest0=model.predict_proba(xtest)[:,1]
                ptest,cal_meta=calibrate(ycal,pcal0,ptest0)

                pred=test.copy()
                pred["predicted_probability"]=ptest
                # Critical contract: price enters only here, after odds-free P(hit).
                pred["edge"]=pred["predicted_probability"]*pred["odds"]-1.0
                predictions[test_year]=pred

                quality=model_quality(ytest,ptest)
                quality_rows.append({
                    "bet_type":bet_type,
                    "template":template,
                    "test_year":test_year,
                    "train_rows":len(train),
                    "fit_rows":len(fit),
                    "calibration_rows":len(cal),
                    "fit_races":fit_races,
                    "calibration_races":cal_races,
                    "feature_count":int(xfit.shape[1]),
                    **quality,
                    "calibration_mode":cal_meta["mode"],
                    "calibration_coef":cal_meta.get("coef"),
                    "calibration_intercept":cal_meta.get("intercept"),
                })
                fold_rows.append({
                    "bet_type":bet_type,
                    "template":template,
                    "test_year":test_year,
                    "status":"OK",
                    "train_years":"|".join(map(str,sorted(map(int,train["year"].unique())))),
                    "train_races":int(train["race_id"].nunique()),
                    "test_races":int(test["race_id"].nunique()),
                    "feature_count":int(xfit.shape[1]),
                })
                for threshold in thresholds:
                    metric_rows.append(metric_row(
                        pred,bet_type,template,test_year,threshold
                    ))

            template_metrics=[
                r for r in metric_rows
                if r["bet_type"]==bet_type and r["template"]==template
            ]
            candidates=candidate_summary(
                template_metrics,bet_type,[template],thresholds,qualification
            )
            bet_candidate_rows.extend(candidates)
            dev_candidates.extend(candidates)
            del df
            del predictions

        chosen,status=choose_candidate(bet_candidate_rows)
        if chosen is None:
            king_rows.append({
                "bet_type":bet_type,
                "status":"NO_DATA",
            })
            continue
        holdout=next((
            r for r in metric_rows
            if r["bet_type"]==bet_type
            and r["template"]==chosen["template"]
            and int(r["test_year"])==HOLDOUT_YEAR
            and abs(float(r["edge_threshold"])-float(chosen["edge_threshold"]))<1e-12
        ),None)
        king={
            "bet_type":bet_type,
            "status":status,
            "template":chosen["template"],
            "edge_threshold":chosen["edge_threshold"],
            "qualified_on_2023_2024":chosen.get("qualified",False),
            "dev_profit_yen":chosen.get("dev_profit_yen"),
            "dev_roi_pct":chosen.get("dev_roi_pct"),
            "min_dev_year_roi_pct":chosen.get("min_year_roi_pct"),
            "dev_bought_tickets":chosen.get("dev_bought_tickets"),
            "dev_bought_races":chosen.get("dev_bought_races"),
            "dev_sum_max_drawdown_yen":chosen.get("sum_max_drawdown_yen"),
            "roi_2023":chosen.get("roi_2023"),
            "profit_2023":chosen.get("profit_2023"),
            "roi_2024":chosen.get("roi_2024"),
            "profit_2024":chosen.get("profit_2024"),
            "holdout_used_for_selection":False,
        }
        if holdout:
            for key in (
                "source_races","bought_tickets","bought_races","stake_yen","return_yen",
                "profit_yen","roi_pct","max_drawdown_yen","positive_race_rate",
                "ticket_hit_rate","race_hit_rate","top1_return_share_pct",
                "top5_return_share_pct","roi_without_top1_return_pct",
                "gate_alert_ticket_share","novel_ticket_share",
            ):
                king["holdout_2025_"+key]=holdout.get(key)
        else:
            king["holdout_2025_status"]="UNAVAILABLE"
        king_rows.append(king)

    write_csv(out_dir/"fold-status.csv",fold_rows)
    write_csv(out_dir/"model-quality.csv",quality_rows)
    write_csv(out_dir/"all-metrics.csv",metric_rows)
    write_csv(out_dir/"development-candidates.csv",dev_candidates)
    write_csv(out_dir/"king-selection.csv",king_rows)

    summary={
        "contract":"L2_BET_KINGS_ARENA_RESULT_V1",
        "upstream":"L15_FIXED_V1",
        "probability_model_uses_odds":False,
        "edge_uses_final_odds":True,
        "development_selection_years":list(DEV_YEARS),
        "holdout_year":HOLDOUT_YEAR,
        "holdout_used_for_selection":False,
        "locked_years":[2026],
        "place_bets":False,
        "wide_bets":False,
        "edge_thresholds":thresholds,
        "king_selection":king_rows,
    }
    (out_dir/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    lines=[
        "# L2 Bet Kings Arena V1",
        "",
        "Upstream is frozen L15_FIXED_V1. This arena does not modify L1/L1.5.",
        "",
        "## Leakage guard",
        "",
        "P(hit) models explicitly exclude odds, implied probability, popularity, payout,",
        "return, ROI, finish/result labels and blind labels. Final odds enter only after",
        "prediction when edge = P(hit) * final_odds - 1 is calculated.",
        "",
        "## Selection discipline",
        "",
        "- 2023 and 2024 out-of-sample folds select each bet-type candidate.",
        "- 2025 is a final holdout and is never used to choose template or edge threshold.",
        "- 2026 remains sealed.",
        "- A king candidate must clear ROI >= 100% in both development years and minimum",
        "  ticket/race counts from the contract. Otherwise the bet type is marked",
        "  NO_QUALIFIED_KING even if a research leader is shown.",
        "",
        "The next layer, Bet Router, is intentionally NOT built in this run.",
        "",
    ]
    (out_dir/"README.md").write_text("\n".join(lines),encoding="utf-8")

    print("L2_BET_KINGS_ARENA_V1_READY")
    print(json.dumps({
        "bet_types":len(BET_TYPES),
        "templates":sum(len(v) for v in templates_by_bet.values()),
        "metric_rows":len(metric_rows),
        "king_rows":len(king_rows),
        "out_dir":str(out_dir),
    },ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
