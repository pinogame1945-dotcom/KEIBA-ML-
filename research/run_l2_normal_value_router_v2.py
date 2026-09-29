#!/usr/bin/env python3
import argparse,csv,gzip,json,math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import YEARS,TEMPLATE_TO_BET,load_fixed_ledgers,load_router
from run_l2_bet_kings_arena_v1 import (
    load_template,feature_columns,encode_fit_other,chronological_split,
    make_model,calibrate,model_quality
)

ANALYSIS_YEARS=(2022,2023,2024,2025)
TEST_YEARS=(2023,2024,2025)
DEV_YEARS=(2023,2024)
HOLDOUT_YEAR=2025
BET_TYPES=("QUINELLA","EXACTA","TRIO","TRIFECTA")
TEMPLATES=tuple(t for t,b in TEMPLATE_TO_BET.items() if b in BET_TYPES)
FIXED_BASELINE="QUINELLA_KING_TOP4_BOX"


def parse_args():
    p=argparse.ArgumentParser(description="Normal-race value router V2: odds-free P(hit), market-price EV, forced routing.")
    p.add_argument("--contract",required=True)
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--v1-metrics",required=False)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1)
        out[int(y)]=p
    return out


def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def read_csv_rows(path):
    with open(path,newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite(x,default=0.0):
    try:
        v=float(x)
        return v if math.isfinite(v) else default
    except (TypeError,ValueError):
        return default


def load_normal_universe(router_paths,fixed_dir):
    fixed=load_fixed_ledgers(fixed_dir)
    routers={y:load_router(router_paths[y],y) for y in YEARS}
    universe={}
    meta={}
    for y in ANALYSIS_YEARS:
        alerts=set(fixed[y])
        ids=[]
        for rid,row in routers[y].items():
            if rid in alerts: continue
            ids.append(str(rid))
            meta[(y,str(rid))]=str(row.get("race_date") or "")[:10]
        universe[y]=sorted(ids)
    return universe,meta


def prepare_template_df(path,template,bet_type):
    df=load_template(path)
    if set(df["template"].astype(str).unique())!={template}:
        raise SystemExit(f"template contamination {template}")
    if set(df["bet_type"].astype(str).unique())!={bet_type}:
        raise SystemExit(f"bet type contamination {template}")
    df=df[(df["year"].isin(ANALYSIS_YEARS)) & (df["gate_alert"].astype(int)==0)].copy()
    if df.empty:
        return df
    if "ticket_novel_count" in df.columns and int(pd.to_numeric(df["ticket_novel_count"],errors="coerce").fillna(0).sum())!=0:
        raise SystemExit(f"novel leakage template={template}")
    df["hit"]=df["hit"].astype(bool)
    df["odds"]=pd.to_numeric(df["odds"],errors="coerce")
    df["return_yen_per100"]=pd.to_numeric(df["return_yen_per100"],errors="coerce").fillna(0.0)
    df["race_month"]=pd.to_numeric(df["race_date"].astype(str).str.slice(5,7),errors="coerce").fillna(0.0)
    return df


def train_template_folds(dataset_dir):
    root=Path(dataset_dir)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract")!="L2_BET_KINGS_DATASET_V1":
        raise SystemExit("wrong dataset contract")
    if manifest.get("source_l15")!="L15_FIXED_V1":
        raise SystemExit("wrong L1.5 upstream")
    if manifest.get("probability_model_uses_odds") is not False:
        raise SystemExit("probability market separation drift")
    if manifest.get("locked_years")!=[2026]:
        raise SystemExit("2026 lock drift")

    portfolio={}
    quality=[]
    for ti,template in enumerate(TEMPLATES):
        info=manifest["templates"].get(template)
        if not info:
            raise SystemExit(f"missing template {template}")
        bet=info["bet_type"]
        df=prepare_template_df(root/info["file"],template,bet)
        # Novel-only templates are unavailable in PASS_SEVEN_ONLY races by construction.
        if df.empty:
            continue
        cols=feature_columns(df)
        if any(tok in c.lower() for c in cols for tok in ("odds","payout","return","profit","roi","hit","winner","finish","popularity")):
            raise SystemExit(f"forbidden feature in {template}: {cols}")

        for yi,test_year in enumerate(TEST_YEARS):
            train=df[df["year"]<test_year].copy()
            test=df[df["year"]==test_year].copy()
            if train.empty or test.empty:
                raise SystemExit(f"missing fold template={template} year={test_year}")
            split=chronological_split(train)
            if split is None:
                raise SystemExit(f"chronological split failed template={template} year={test_year}")
            fit,cal,fit_races,cal_races=split
            yfit=fit["hit"].astype(int).to_numpy()
            ycal=cal["hit"].astype(int).to_numpy()
            ytest=test["hit"].astype(int).to_numpy()
            if len(set(yfit.tolist()))<2:
                raise SystemExit(f"single class fit template={template} year={test_year}")

            xfit,(xcal,xtest)=encode_fit_other(fit,[cal,test],cols)
            model=make_model(91000+ti*100+yi)
            model.fit(xfit,yfit)
            pcal0=model.predict_proba(xcal)[:,1]
            ptest0=model.predict_proba(xtest)[:,1]
            ptest,calmeta=calibrate(ycal,pcal0,ptest0)

            pred=test[["year","race_id","race_date","odds","return_yen_per100","hit"]].copy()
            pred["predicted_probability"]=ptest
            # Market price enters only after the odds-free probability prediction.
            pred["expected_return_yen"]=pred["predicted_probability"]*pred["odds"]*100.0
            pred["stake_yen"]=100.0

            for (y,rid,date),g in pred.groupby(["year","race_id","race_date"],sort=False):
                stake=float(g["stake_yen"].sum())
                eret=float(g["expected_return_yen"].sum())
                actual=float(g["return_yen_per100"].sum())
                tickets=int(len(g))
                portfolio[(int(y),str(rid),template)]={
                    "template":template,
                    "bet_type":bet,
                    "race_date":str(date)[:10],
                    "tickets":tickets,
                    "stake_yen":stake,
                    "expected_return_yen":eret,
                    "expected_profit_yen":eret-stake,
                    "expected_roi_pct":100*eret/stake if stake else None,
                    "return_yen":actual,
                    "profit_yen":actual-stake,
                    "hit":int(bool(g["hit"].any())),
                    "avg_predicted_probability":float(g["predicted_probability"].mean()),
                    "avg_final_odds":float(g["odds"].mean()),
                }

            q=model_quality(ytest,ptest)
            quality.append({
                "template":template,"bet_type":bet,"test_year":test_year,
                "train_years":"|".join(map(str,sorted(map(int,train["year"].unique())))),
                "fit_races":fit_races,"calibration_races":cal_races,
                "test_races":int(test["race_id"].nunique()),
                "feature_count":int(xfit.shape[1]),
                **q,
                "calibration_mode":calmeta.get("mode"),
                "calibration_coef":calmeta.get("coef"),
                "calibration_intercept":calmeta.get("intercept"),
            })
        del df
    return manifest,portfolio,quality


def choose_template(y,rid,portfolio,mode):
    rows=[portfolio[(y,rid,t)] for t in TEMPLATES if (y,rid,t) in portfolio]
    if not rows:
        return None
    if mode=="MAX_EXPECTED_ROI":
        return max(rows,key=lambda r:(finite(r["expected_roi_pct"],-1e30),finite(r["expected_profit_yen"],-1e30),-r["tickets"],r["template"]))
    if mode=="MAX_EXPECTED_PROFIT":
        return max(rows,key=lambda r:(finite(r["expected_profit_yen"],-1e30),finite(r["expected_roi_pct"],-1e30),-r["tickets"],r["template"]))
    if mode=="ORACLE_DIAGNOSTIC":
        return max(rows,key=lambda r:(finite(r["profit_yen"],-1e30),-r["stake_yen"],-r["tickets"],r["template"]))
    if mode=="FIXED_BASELINE":
        return portfolio.get((y,rid,FIXED_BASELINE))
    raise ValueError(mode)


def concentration(rows):
    if not rows:
        return {"top1_return_share_pct":None,"top5_return_share_pct":None,"roi_without_top1_pct":None}
    total_ret=sum(x["return_yen"] for x in rows)
    total_stake=sum(x["stake_yen"] for x in rows)
    vals=sorted((x["return_yen"] for x in rows),reverse=True)
    top1=vals[0] if vals else 0.0
    top5=sum(vals[:5])
    return {
        "top1_return_share_pct":100*top1/total_ret if total_ret>0 else None,
        "top5_return_share_pct":100*top5/total_ret if total_ret>0 else None,
        "roi_without_top1_pct":100*max(0.0,total_ret-top1)/total_stake if total_stake else None,
    }


def max_drawdown(rows):
    seq=sorted(rows,key=lambda x:(x["race_date"],x["race_id"]))
    cur=0.0; peak=0.0; dd=0.0
    for x in seq:
        cur+=x["return_yen"]-x["stake_yen"]
        peak=max(peak,cur)
        dd=max(dd,peak-cur)
    return dd


def evaluate(year,universe,meta,portfolio,mode):
    executed=[]
    decisions=[]
    missing=[]
    for rid in universe[year]:
        choice=choose_template(year,rid,portfolio,mode)
        if choice is None:
            missing.append(rid)
            decisions.append({
                "year":year,"race_id":rid,"race_date":meta[(year,rid)],
                "architecture":mode,"execution_status":"DATA_UNAVAILABLE",
                "template":"","bet_type":"","tickets":0,
                "expected_roi_pct":None,"expected_profit_yen":None,
                "realized_profit_yen":None,
            })
            continue
        z=dict(choice)
        z["race_id"]=rid
        z["race_date"]=meta[(year,rid)]
        executed.append(z)
        decisions.append({
            "year":year,"race_id":rid,"race_date":meta[(year,rid)],
            "architecture":mode,"execution_status":"EXECUTED",
            "template":z["template"],"bet_type":z["bet_type"],"tickets":z["tickets"],
            "expected_roi_pct":z["expected_roi_pct"],"expected_profit_yen":z["expected_profit_yen"],
            "realized_profit_yen":z["profit_yen"],
        })

    n=len(universe[year])
    stake=sum(x["stake_yen"] for x in executed)
    ret=sum(x["return_yen"] for x in executed)
    return {
        "architecture":mode,
        "test_year":year,
        "source_races":n,
        "routed_races":n,
        "route_coverage_pct":100.0 if n else 0.0,
        "executed_races":len(executed),
        "execution_coverage_pct":100*len(executed)/n if n else 0.0,
        "data_unavailable_races":len(missing),
        "tickets":sum(x["tickets"] for x in executed),
        "avg_tickets_per_source_race":sum(x["tickets"] for x in executed)/n if n else None,
        "hit_races":sum(x["hit"] for x in executed),
        "race_hit_rate_pct":100*sum(x["hit"] for x in executed)/n if n else None,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(executed),
        "avg_selected_expected_roi_pct":float(np.mean([x["expected_roi_pct"] for x in executed])) if executed else None,
        "avg_selected_expected_profit_yen":float(np.mean([x["expected_profit_yen"] for x in executed])) if executed else None,
        **concentration(executed),
    },decisions


def route_shares(decisions):
    out=[]
    groups=defaultdict(int)
    den=defaultdict(int)
    for r in decisions:
        if r["execution_status"]!="EXECUTED": continue
        key=(int(r["year"]),r["architecture"],r["bet_type"],r["template"])
        groups[key]+=1
        den[(int(r["year"]),r["architecture"])]+=1
    for (y,a,b,t),n in sorted(groups.items()):
        d=den[(y,a)]
        out.append({
            "year":y,"architecture":a,"bet_type":b,"template":t,
            "races":n,"share_pct":100*n/d if d else None,
        })
    return out


def main():
    a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L2_NORMAL_VALUE_ROUTER_V2":
        raise SystemExit("wrong contract")
    if contract["scope"]["battlefield"]!="PASS_SEVEN_ONLY":
        raise SystemExit("battlefield drift")
    if contract["scope"]["race_filtering"] is not False:
        raise SystemExit("race filtering forbidden")
    if contract["scope"]["force_route_every_normal_race"] is not True:
        raise SystemExit("forced route guard broken")
    if contract["scope"]["locked_years"]!=[2026]:
        raise SystemExit("2026 lock drift")
    if contract["probability_model"]["allowed_market_features"] is not False:
        raise SystemExit("P(hit) market leakage contract broken")
    if contract["selection"]["final_holdout_year"]!=2025:
        raise SystemExit("holdout drift")

    paths=parse_paths(a.router_year)
    if set(paths)!=set(YEARS):
        raise SystemExit(f"router years mismatch {sorted(paths)}")
    universe,meta=load_normal_universe(paths,a.fixed_ledger_dir)
    manifest,portfolio,quality=train_template_folds(a.dataset_dir)

    expected={2022:3110,2023:3110,2024:3110,2025:3110}
    if {y:len(universe[y]) for y in ANALYSIS_YEARS}!=expected:
        raise SystemExit(f"normal universe drift {[ (y,len(universe[y])) for y in ANALYSIS_YEARS ]}")

    metrics=[]
    all_decisions=[]
    modes=("MAX_EXPECTED_ROI","MAX_EXPECTED_PROFIT","FIXED_BASELINE","ORACLE_DIAGNOSTIC")
    for y in TEST_YEARS:
        for mode in modes:
            m,d=evaluate(y,universe,meta,portfolio,mode)
            metrics.append(m); all_decisions.extend(d)

    # Development-only selection between the two reachable value routers.
    candidates=[]
    for arch in ("MAX_EXPECTED_ROI","MAX_EXPECTED_PROFIT"):
        rows=[r for r in metrics if r["architecture"]==arch and r["test_year"] in DEV_YEARS]
        stake=sum(r["stake_yen"] for r in rows); ret=sum(r["return_yen"] for r in rows)
        candidates.append({
            "architecture":arch,
            "development_years":"2023|2024",
            "dev_stake_yen":stake,
            "dev_return_yen":ret,
            "dev_profit_yen":ret-stake,
            "dev_roi_pct":100*ret/stake if stake else None,
            "min_dev_year_roi_pct":min(r["roi_pct"] for r in rows),
            "sum_dev_max_drawdown_yen":sum(r["max_drawdown_yen"] for r in rows),
            "dev_tickets":sum(r["tickets"] for r in rows),
            "dev_route_coverage_pct":min(r["route_coverage_pct"] for r in rows),
        })
    candidates.sort(key=lambda r:(-r["dev_profit_yen"],-r["min_dev_year_roi_pct"],r["sum_dev_max_drawdown_yen"],r["dev_tickets"],r["architecture"]))
    chosen=candidates[0]["architecture"]
    for i,r in enumerate(candidates):
        r["selected_on_development"]=int(i==0)

    holdout=next(r for r in metrics if r["architecture"]==chosen and r["test_year"]==HOLDOUT_YEAR)
    fixed_holdout=next(r for r in metrics if r["architecture"]=="FIXED_BASELINE" and r["test_year"]==HOLDOUT_YEAR)

    v1_rows=[]
    if a.v1_metrics and Path(a.v1_metrics).exists():
        for r in read_csv_rows(a.v1_metrics):
            if r.get("architecture")=="DIRECT_TEMPLATE":
                v1_rows.append(r)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"model-quality.csv",quality)
    write_csv(out/"value-router-metrics.csv",metrics)
    write_csv(out/"development-selection.csv",candidates)
    write_csv(out/"route-shares.csv",route_shares(all_decisions))
    write_csv(out/"v1-direct-baseline.csv",v1_rows)
    with gzip.open(out/"value-route-decisions.csv.gz","wt",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(all_decisions[0]))
        w.writeheader(); w.writerows(all_decisions)

    summary={
        "contract":"L2_NORMAL_VALUE_ROUTER_RESULT_V2",
        "scope":"PASS_SEVEN_ONLY forced routing; no skip",
        "normal_races":{str(y):len(universe[y]) for y in ANALYSIS_YEARS},
        "probability_model_uses_market":False,
        "market_stage_uses_final_odds":True,
        "market_price_caveat":"Historical final odds proxy; not live timestamp simulation.",
        "architectures":["MAX_EXPECTED_ROI","MAX_EXPECTED_PROFIT"],
        "development_selection_years":[2023,2024],
        "selected_architecture":chosen,
        "holdout_year":2025,
        "holdout_used_for_selection":False,
        "selected_holdout_2025":holdout,
        "fixed_baseline_holdout_2025":fixed_holdout,
        "v1_direct_baseline_included":bool(v1_rows),
        "race_filtering":False,
        "route_every_normal_race":True,
        "2026_locked":True,
        "production_promotion":False,
        "interpretation_guard":"P(hit) is learned without market inputs. Final odds enter only after prediction to estimate ticket/template EV. Architecture is selected on 2023-2024 only; 2025 is holdout-only. ORACLE is unreachable hindsight diagnostic."
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Normal Value Router V2\n\n"
        "PASS_SEVEN_ONLY only. Every normal race is routed; no learned SKIP class exists. "
        "For each fixed ticket template, P(hit) is learned from pre-race structural features without odds/popularity/payout/results. "
        "Historical final odds enter only after prediction to form expected ticket returns, which are aggregated to template-level expected ROI/profit. "
        "MAX_EXPECTED_ROI and MAX_EXPECTED_PROFIT are selected only on 2023-2024. 2025 is holdout-only. "
        "2026 remains sealed and no production promotion occurs.\n",
        encoding="utf-8",
    )
    print("L2_NORMAL_VALUE_ROUTER_V2_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)


if __name__=="__main__":
    main()
