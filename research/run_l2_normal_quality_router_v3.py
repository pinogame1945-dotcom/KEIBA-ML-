#!/usr/bin/env python3
import argparse,csv,gzip,json,math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import YEARS,TEMPLATE_TO_BET,load_fixed_ledgers,load_router
from run_l2_bet_kings_arena_v1 import (
    load_template,chronological_split,make_model,calibrate,model_quality
)
from run_l2_normal_router_v1 import build_feature_rows,encode_fit_other

ANALYSIS_YEARS=(2022,2023,2024,2025)
TEST_YEARS=(2023,2024,2025)
DEV_YEARS=(2023,2024)
HOLDOUT_YEAR=2025
BET_TYPES=("QUINELLA","EXACTA","TRIO","TRIFECTA")
TEMPLATES=tuple(t for t,b in TEMPLATE_TO_BET.items() if b in BET_TYPES)
FIXED_BASELINE="QUINELLA_KING_TOP4_BOX"
FORBIDDEN_TOKENS=(
    "odds","popularity","payout","return","profit","roi","hit","finish","winner",
    "result","implied","target","label"
)

def parse_args():
    p=argparse.ArgumentParser(description="Normal quality router V3: GOOD/NEUTRAL/BAD template routing.")
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

def build_universe(router_paths,fixed_dir):
    fixed=load_fixed_ledgers(fixed_dir)
    routers={y:load_router(router_paths[y],y) for y in YEARS}
    features=build_feature_rows(routers,fixed)
    expected={y:len(routers[y])-len(fixed[y]) for y in ANALYSIS_YEARS}
    actual=features.groupby("year")["race_id"].nunique().to_dict()
    if any(int(actual.get(y,0))!=int(expected[y]) for y in ANALYSIS_YEARS):
        raise SystemExit(f"normal universe drift actual={actual} expected={expected}")
    feature_cols=[c for c in features.columns if c not in {"year","race_id","race_date"}]
    bad=[c for c in feature_cols if any(tok in c.lower() for tok in FORBIDDEN_TOKENS)]
    if bad:
        raise SystemExit(f"forbidden feature names {sorted(bad)}")
    features["race_id"]=features["race_id"].astype(str)
    features["year"]=features["year"].astype(int)
    return fixed,routers,features,feature_cols,expected

def aggregate_template(path,template,bet_type):
    df=load_template(path)
    if set(df["template"].astype(str).unique())!={template}:
        raise SystemExit(f"template contamination {template}")
    if set(df["bet_type"].astype(str).unique())!={bet_type}:
        raise SystemExit(f"bet type contamination {template}")
    df=df[(df["year"].isin(ANALYSIS_YEARS)) & (df["gate_alert"].astype(int)==0)].copy()
    if df.empty:
        return pd.DataFrame()
    if "ticket_novel_count" in df.columns:
        novel=int(pd.to_numeric(df["ticket_novel_count"],errors="coerce").fillna(0).sum())
        if novel!=0:
            raise SystemExit(f"novel leakage template={template}")
    df["hit"]=df["hit"].astype(bool)
    df["return_yen_per100"]=pd.to_numeric(df["return_yen_per100"],errors="coerce").fillna(0.0)

    rows=[]
    for (y,rid,date),g in df.groupby(["year","race_id","race_date"],sort=False):
        tickets=int(len(g))
        stake=100.0*tickets
        ret=float(g["return_yen_per100"].sum())
        any_hit=int(bool(g["hit"].any()))
        profit=ret-stake
        good=int(profit>0)
        bad=int(not any_hit)
        neutral=int(any_hit and profit<=0)
        if good+bad+neutral!=1:
            raise SystemExit(f"label partition broken template={template} race={rid}")
        rows.append({
            "year":int(y),"race_id":str(rid),"race_date":str(date)[:10],
            "template":template,"bet_type":bet_type,
            "tickets":tickets,"stake_yen":stake,"return_yen":ret,"profit_yen":profit,
            "any_hit":any_hit,"label_good":good,"label_neutral":neutral,"label_bad":bad,
        })
    return pd.DataFrame(rows)

def fit_binary(train,test,feature_cols,target,seed):
    split=chronological_split(train)
    if split is None:
        raise SystemExit(f"chronological split unavailable target={target}")
    fit,cal,fit_races,cal_races=split
    yfit=fit[target].astype(int).to_numpy()
    ycal=cal[target].astype(int).to_numpy()
    ytest=test[target].astype(int).to_numpy()
    if len(set(yfit.tolist()))<2:
        p=float(np.mean(yfit)) if len(yfit) else 0.0
        pred=np.repeat(p,len(test))
        q=model_quality(ytest,pred)
        return pred,{
            "fit_races":fit_races,"calibration_races":cal_races,
            "feature_count":len(feature_cols),"mode":"constant_fit",
            **q
        }

    xfit,(xcal,xtest)=encode_fit_other(fit,[cal,test],feature_cols)
    model=make_model(seed)
    model.fit(xfit,yfit)
    pcal0=model.predict_proba(xcal)[:,1]
    ptest0=model.predict_proba(xtest)[:,1]
    ptest,calmeta=calibrate(ycal,pcal0,ptest0)
    q=model_quality(ytest,ptest)
    return ptest,{
        "fit_races":fit_races,"calibration_races":cal_races,
        "feature_count":int(xfit.shape[1]),
        "mode":calmeta.get("mode"),
        "calibration_coef":calmeta.get("coef"),
        "calibration_intercept":calmeta.get("intercept"),
        **q
    }

def train_all(dataset_dir,features,feature_cols):
    root=Path(dataset_dir)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract")!="L2_BET_KINGS_DATASET_V1":
        raise SystemExit("wrong dataset contract")
    if manifest.get("source_l15")!="L15_FIXED_V1":
        raise SystemExit("wrong upstream")
    if manifest.get("locked_years")!=[2026]:
        raise SystemExit("2026 lock drift")

    portfolio={}
    quality=[]
    label_dist=[]
    available_templates=[]

    for ti,template in enumerate(TEMPLATES):
        info=manifest["templates"].get(template)
        if not info:
            raise SystemExit(f"missing template {template}")
        bet=info["bet_type"]
        lab=aggregate_template(root/info["file"],template,bet)
        if lab.empty:
            continue
        available_templates.append(template)
        joined=lab.merge(features,on=["year","race_id","race_date"],how="left",validate="one_to_one")
        if joined[feature_cols].isna().all(axis=1).any():
            raise SystemExit(f"feature join missing template={template}")

        for y in ANALYSIS_YEARS:
            d=joined[joined["year"]==y]
            if d.empty: continue
            label_dist.append({
                "year":y,"template":template,"bet_type":bet,"races":len(d),
                "good_races":int(d["label_good"].sum()),
                "good_rate_pct":100*float(d["label_good"].mean()),
                "neutral_races":int(d["label_neutral"].sum()),
                "neutral_rate_pct":100*float(d["label_neutral"].mean()),
                "bad_races":int(d["label_bad"].sum()),
                "bad_rate_pct":100*float(d["label_bad"].mean()),
            })

        for yi,test_year in enumerate(TEST_YEARS):
            train=joined[joined["year"]<test_year].copy()
            test=joined[joined["year"]==test_year].copy()
            if train.empty or test.empty:
                continue
            pg,qg=fit_binary(train,test,feature_cols,"label_good",101000+ti*100+yi*2)
            pb,qb=fit_binary(train,test,feature_cols,"label_bad",101001+ti*100+yi*2)

            for rec,p_good,p_bad in zip(test.to_dict("records"),pg,pb):
                portfolio[(test_year,str(rec["race_id"]),template)]={
                    "template":template,"bet_type":bet,"race_date":rec["race_date"],
                    "tickets":int(rec["tickets"]),
                    "stake_yen":float(rec["stake_yen"]),
                    "return_yen":float(rec["return_yen"]),
                    "profit_yen":float(rec["profit_yen"]),
                    "hit":int(rec["any_hit"]),
                    "actual_good":int(rec["label_good"]),
                    "actual_neutral":int(rec["label_neutral"]),
                    "actual_bad":int(rec["label_bad"]),
                    "p_good":float(p_good),
                    "p_bad":float(p_bad),
                    "score_good_minus_bad":float(p_good-p_bad),
                }

            quality.append({
                "template":template,"bet_type":bet,"target":"GOOD","test_year":test_year,
                "train_years":"|".join(map(str,sorted(map(int,train["year"].unique())))),
                **qg
            })
            quality.append({
                "template":template,"bet_type":bet,"target":"BAD","test_year":test_year,
                "train_years":"|".join(map(str,sorted(map(int,train["year"].unique())))),
                **qb
            })

    if not available_templates:
        raise SystemExit("no normal templates")
    return manifest,portfolio,quality,label_dist,tuple(available_templates)

def choose_template(year,rid,portfolio,templates,mode):
    rows=[portfolio[(year,rid,t)] for t in templates if (year,rid,t) in portfolio]
    if not rows:
        return None
    if mode=="MAX_GOOD_PROB":
        return max(rows,key=lambda r:(r["p_good"],-r["p_bad"],-r["tickets"],r["template"]))
    if mode=="GOOD_MINUS_BAD":
        return max(rows,key=lambda r:(r["score_good_minus_bad"],r["p_good"],-r["tickets"],r["template"]))
    if mode=="FIXED_BASELINE":
        return portfolio.get((year,rid,FIXED_BASELINE))
    if mode=="ORACLE_DIAGNOSTIC":
        return max(rows,key=lambda r:(r["profit_yen"],-r["stake_yen"],-r["tickets"],r["template"]))
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
        cur+=x["profit_yen"]
        peak=max(peak,cur)
        dd=max(dd,peak-cur)
    return dd

def evaluate(year,universe,dates,portfolio,templates,mode):
    executed=[]
    decisions=[]
    unavailable=[]
    for rid in universe[year]:
        choice=choose_template(year,rid,portfolio,templates,mode)
        if choice is None:
            unavailable.append(rid)
            decisions.append({
                "year":year,"race_id":rid,"race_date":dates[(year,rid)],
                "architecture":mode,"execution_status":"DATA_UNAVAILABLE",
                "template":"","bet_type":"","tickets":0,
                "p_good":None,"p_bad":None,"score_good_minus_bad":None,
                "realized_profit_yen":None
            })
            continue
        z=dict(choice)
        z["race_id"]=rid
        z["race_date"]=dates[(year,rid)]
        executed.append(z)
        decisions.append({
            "year":year,"race_id":rid,"race_date":dates[(year,rid)],
            "architecture":mode,"execution_status":"EXECUTED",
            "template":z["template"],"bet_type":z["bet_type"],"tickets":z["tickets"],
            "p_good":z["p_good"],"p_bad":z["p_bad"],
            "score_good_minus_bad":z["score_good_minus_bad"],
            "realized_profit_yen":z["profit_yen"]
        })

    n=len(universe[year])
    stake=sum(x["stake_yen"] for x in executed)
    ret=sum(x["return_yen"] for x in executed)
    return {
        "architecture":mode,"test_year":year,
        "source_races":n,"routed_races":n,"route_coverage_pct":100.0 if n else 0.0,
        "executed_races":len(executed),
        "execution_coverage_pct":100*len(executed)/n if n else 0.0,
        "data_unavailable_races":len(unavailable),
        "tickets":sum(x["tickets"] for x in executed),
        "avg_tickets_per_source_race":sum(x["tickets"] for x in executed)/n if n else None,
        "hit_races":sum(x["hit"] for x in executed),
        "race_hit_rate_pct":100*sum(x["hit"] for x in executed)/n if n else None,
        "good_races":sum(x["actual_good"] for x in executed),
        "good_rate_pct":100*sum(x["actual_good"] for x in executed)/n if n else None,
        "neutral_races":sum(x["actual_neutral"] for x in executed),
        "bad_races":sum(x["actual_bad"] for x in executed),
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(executed),
        "avg_selected_p_good":float(np.mean([x["p_good"] for x in executed])) if executed else None,
        "avg_selected_p_bad":float(np.mean([x["p_bad"] for x in executed])) if executed else None,
        **concentration(executed)
    },decisions

def route_shares(decisions):
    groups=defaultdict(int); den=defaultdict(int)
    for r in decisions:
        if r["execution_status"]!="EXECUTED": continue
        key=(int(r["year"]),r["architecture"],r["bet_type"],r["template"])
        groups[key]+=1
        den[(int(r["year"]),r["architecture"])]+=1
    out=[]
    for (y,a,b,t),n in sorted(groups.items()):
        d=den[(y,a)]
        out.append({"year":y,"architecture":a,"bet_type":b,"template":t,"races":n,"share_pct":100*n/d if d else None})
    return out

def main():
    a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L2_NORMAL_QUALITY_ROUTER_V3":
        raise SystemExit("wrong contract")
    if contract["scope"]["battlefield"]!="PASS_SEVEN_ONLY":
        raise SystemExit("battlefield drift")
    if contract["scope"]["race_filtering"] is not False:
        raise SystemExit("race filtering must remain disabled")
    if contract["scope"]["force_route_every_normal_race"] is not True:
        raise SystemExit("forced-route guard broken")
    if contract["scope"]["skip_class"] is not False:
        raise SystemExit("skip class forbidden in V3")
    if contract["scope"]["locked_years"]!=[2026]:
        raise SystemExit("2026 lock drift")
    if contract["models"]["allowed_market_features"] is not False:
        raise SystemExit("market leakage contract broken")
    if contract["selection"]["final_holdout_year"]!=2025:
        raise SystemExit("holdout drift")

    paths=parse_paths(a.router_year)
    if set(paths)!=set(YEARS):
        raise SystemExit(f"router years mismatch {sorted(paths)}")
    _,routers,features,feature_cols,expected=build_universe(paths,a.fixed_ledger_dir)
    dates={(int(r["year"]),str(r["race_id"])):str(r["race_date"]) for r in features[["year","race_id","race_date"]].to_dict("records")}
    universe={y:sorted(features[features["year"]==y]["race_id"].astype(str).tolist()) for y in ANALYSIS_YEARS}

    manifest,portfolio,quality,label_dist,templates=train_all(a.dataset_dir,features,feature_cols)

    metrics=[]; all_decisions=[]
    for y in TEST_YEARS:
        for mode in ("MAX_GOOD_PROB","GOOD_MINUS_BAD","FIXED_BASELINE","ORACLE_DIAGNOSTIC"):
            m,d=evaluate(y,universe,dates,portfolio,templates,mode)
            metrics.append(m); all_decisions.extend(d)

    candidates=[]
    for arch in ("MAX_GOOD_PROB","GOOD_MINUS_BAD"):
        rows=[r for r in metrics if r["architecture"]==arch and r["test_year"] in DEV_YEARS]
        stake=sum(r["stake_yen"] for r in rows); ret=sum(r["return_yen"] for r in rows)
        candidates.append({
            "architecture":arch,"development_years":"2023|2024",
            "dev_stake_yen":stake,"dev_return_yen":ret,"dev_profit_yen":ret-stake,
            "dev_roi_pct":100*ret/stake if stake else None,
            "min_dev_year_roi_pct":min(r["roi_pct"] for r in rows),
            "sum_dev_max_drawdown_yen":sum(r["max_drawdown_yen"] for r in rows),
            "dev_tickets":sum(r["tickets"] for r in rows),
            "dev_route_coverage_pct":min(r["route_coverage_pct"] for r in rows)
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
    write_csv(out/"label-distribution.csv",label_dist)
    write_csv(out/"quality-router-metrics.csv",metrics)
    write_csv(out/"development-selection.csv",candidates)
    write_csv(out/"route-shares.csv",route_shares(all_decisions))
    write_csv(out/"v1-direct-baseline.csv",v1_rows)
    with gzip.open(out/"quality-route-decisions.csv.gz","wt",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(all_decisions[0]))
        w.writeheader(); w.writerows(all_decisions)

    summary={
        "contract":"L2_NORMAL_QUALITY_ROUTER_RESULT_V3",
        "scope":"PASS_SEVEN_ONLY forced routing; no skip",
        "normal_races":{str(y):len(universe[y]) for y in ANALYSIS_YEARS},
        "labels":{
            "GOOD":"template race profit > 0",
            "NEUTRAL":"template hits but race profit <= 0",
            "BAD":"template no hit"
        },
        "probability_models_use_market":False,
        "routing_architectures":["MAX_GOOD_PROB","GOOD_MINUS_BAD"],
        "development_selection_years":[2023,2024],
        "selected_architecture":chosen,
        "holdout_year":2025,
        "holdout_used_for_selection":False,
        "selected_holdout_2025":holdout,
        "fixed_baseline_holdout_2025":fixed_holdout,
        "v1_direct_baseline_included":bool(v1_rows),
        "race_filtering":False,
        "route_every_normal_race":True,
        "skip_class":False,
        "2026_locked":True,
        "production_promotion":False,
        "interpretation_guard":"GOOD/NEUTRAL/BAD are historical supervised labels only. Prediction inputs use pre-race Seven-King/race structure and no odds/payout/results. Architecture selection uses 2023-2024 only; 2025 is holdout-only. ORACLE is unreachable."
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Normal Quality Router V3\n\n"
        "PASS_SEVEN_ONLY only. Every normal race remains in scope; no learned SKIP class exists. "
        "Each available fixed template receives two independently calibrated probabilities from pre-race structural features: P(GOOD) and P(BAD). "
        "GOOD means the template was profitable on that historical race, NEUTRAL means it hit but lost after ticket cost, and BAD means no hit. "
        "MAX_GOOD_PROB and GOOD_MINUS_BAD are compared on 2023-2024 only. 2025 is holdout-only. "
        "Odds, popularity, payout and result-derived values are forbidden as prediction inputs. 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("L2_NORMAL_QUALITY_ROUTER_V3_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
