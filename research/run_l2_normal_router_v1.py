#!/usr/bin/env python3
import argparse,csv,gzip,json,math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import LabelEncoder

from build_l2_bet_kings_dataset_v1 import (
    YEARS,TEMPLATE_TO_BET,load_fixed_ledgers,load_router,seven_stats
)

ANALYSIS_YEARS=(2022,2023,2024,2025)
TEST_YEARS=(2023,2024,2025)
DEV_YEARS=(2023,2024)
HOLDOUT_YEAR=2025
BET_TYPES=("QUINELLA","EXACTA","TRIO","TRIFECTA")
TEMPLATES=tuple(t for t,b in TEMPLATE_TO_BET.items() if b in BET_TYPES)
BASELINE_TEMPLATE="QUINELLA_KING_TOP4_BOX"

CATEGORICAL={
    "venue_code","surface","race_class","discipline","direction","weather","track_condition"
}
FORBIDDEN_TOKENS=(
    "odds","popularity","payout","return","profit","roi","hit","finish","winner","result","implied"
)


def parse_args():
    p=argparse.ArgumentParser(description="L2 normal-race forced router V1.")
    p.add_argument("--contract",required=True)
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True)
    p.add_argument("--dataset-dir",required=True)
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


def finite(x,default=0.0):
    try:
        v=float(x)
        return v if math.isfinite(v) else default
    except (TypeError,ValueError):
        return default


def summary_stats(values,prefix):
    a=np.asarray([finite(x) for x in values],dtype=float)
    if len(a)==0:
        return {
            prefix+"_mean":0.0,prefix+"_std":0.0,prefix+"_min":0.0,
            prefix+"_max":0.0,prefix+"_sum":0.0,
        }
    return {
        prefix+"_mean":float(a.mean()),
        prefix+"_std":float(a.std()),
        prefix+"_min":float(a.min()),
        prefix+"_max":float(a.max()),
        prefix+"_sum":float(a.sum()),
    }


def build_feature_rows(routers,fixed):
    rows=[]
    for year in ANALYSIS_YEARS:
        alerts=set(fixed[year])
        for rid,router in routers[year].items():
            if rid in alerts:
                continue
            order,stats=seven_stats(router)
            race=router.get("race") or {}
            cons=router.get("consensus") or {}
            support=[stats[h]["support"] for h in order]
            borda=[stats[h]["borda"] for h in order]
            top1=[stats[h]["top1_votes"] for h in order]
            best_rank=[stats[h]["best_rank"] for h in order]
            mean_rank=[stats[h]["mean_rank"] for h in order]

            row={
                "year":year,
                "race_id":str(rid),
                "race_date":str(router.get("race_date") or "")[:10],
                "race_month":finite(str(router.get("race_date") or "")[5:7],0.0),
                "venue_code":race.get("venue_code"),
                "surface":race.get("surface"),
                "race_class":race.get("race_class"),
                "discipline":race.get("discipline"),
                "direction":race.get("direction"),
                "weather":race.get("weather"),
                "track_condition":race.get("track_condition"),
                "distance_m":finite(race.get("distance_m")),
                "field_size":finite(race.get("field_size")),
                "seven_union_count":len(order),
                "cw_top1_max_vote_share":finite(cons.get("top1_max_vote_share")),
                "cw_top3_jaccard":finite(cons.get("top3_pairwise_jaccard_mean")),
                "cw_top6_jaccard":finite(cons.get("top6_pairwise_jaccard_mean")),
                "cw_rank_diff_mean":finite(cons.get("pairwise_rank_abs_diff_mean")),
                "cw_rank_std_mean":finite(cons.get("horse_rank_std_mean")),
                "cw_prob_std_mean":finite(cons.get("horse_probability_std_mean")),
                "cw_prob_std_max":finite(cons.get("horse_probability_std_max")),
                "king_support_ge2":sum(x>=2 for x in support),
                "king_support_ge3":sum(x>=3 for x in support),
                "king_support_ge4":sum(x>=4 for x in support),
                "king_support_ge5":sum(x>=5 for x in support),
                "king_support_eq7":sum(x==7 for x in support),
                "king_top1_vote_horses":sum(x>0 for x in top1),
            }
            row.update(summary_stats(support,"king_support"))
            row.update(summary_stats(borda,"king_borda"))
            row.update(summary_stats(top1,"king_top1_votes"))
            row.update(summary_stats(best_rank,"king_best_rank"))
            row.update(summary_stats(mean_rank,"king_mean_rank"))

            # Positional Seven-King features. These are pre-race consensus structure only.
            for pos in range(1,7):
                if pos<=len(order):
                    s=stats[order[pos-1]]
                    row[f"p{pos}_support"]=finite(s.get("support"))
                    row[f"p{pos}_borda"]=finite(s.get("borda"))
                    row[f"p{pos}_best_rank"]=finite(s.get("best_rank"),99.0)
                    row[f"p{pos}_mean_rank"]=finite(s.get("mean_rank"),99.0)
                    row[f"p{pos}_top1_votes"]=finite(s.get("top1_votes"))
                else:
                    row[f"p{pos}_support"]=0.0
                    row[f"p{pos}_borda"]=0.0
                    row[f"p{pos}_best_rank"]=99.0
                    row[f"p{pos}_mean_rank"]=99.0
                    row[f"p{pos}_top1_votes"]=0.0

            row["gap_p1_p2_borda"]=row["p1_borda"]-row["p2_borda"]
            row["gap_p2_p3_borda"]=row["p2_borda"]-row["p3_borda"]
            row["gap_p1_p2_support"]=row["p1_support"]-row["p2_support"]
            row["gap_p2_p3_support"]=row["p2_support"]-row["p3_support"]
            denom=max(1e-9,row["king_borda_sum"])
            row["top2_borda_share"]=(row["p1_borda"]+row["p2_borda"])/denom
            row["top4_borda_share"]=sum(row[f"p{i}_borda"] for i in range(1,5))/denom
            rows.append(row)
    return pd.DataFrame(rows)


def load_performance(dataset_dir):
    root=Path(dataset_dir)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract")!="L2_BET_KINGS_DATASET_V1":
        raise SystemExit("wrong dataset contract")
    if manifest.get("source_l15")!="L15_FIXED_V1":
        raise SystemExit("wrong dataset upstream")
    if 2026 not in manifest.get("locked_years",[]):
        raise SystemExit("2026 dataset lock missing")

    perf=defaultdict(lambda:{"tickets":0,"stake":0.0,"ret":0.0,"hit":0})
    for template in TEMPLATES:
        info=manifest["templates"].get(template)
        if not info:
            raise SystemExit(f"missing template dataset {template}")
        with gzip.open(root/info["file"],"rt",encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                r=json.loads(line)
                y=int(r["year"])
                if y not in ANALYSIS_YEARS: continue
                if int(r.get("gate_alert") or 0)!=0: continue
                if int(r.get("ticket_novel_count") or 0)!=0:
                    raise SystemExit(f"novel leakage race={r['race_id']} template={template}")
                key=(y,str(r["race_id"]),template)
                z=perf[key]
                z["tickets"]+=1
                z["stake"]+=100.0
                z["ret"]+=float(r.get("return_yen_per100") or 0.0)
                z["hit"]=int(z["hit"] or bool(r.get("hit")))
    return manifest,perf


def better(a,b):
    # Higher realized profit, then lower stake/tickets, then lexical template.
    if b is None: return True
    ka=(a["profit"],-a["stake"],-a["tickets"],a["template"])
    kb=(b["profit"],-b["stake"],-b["tickets"],b["template"])
    return ka>kb


def attach_targets(features,perf):
    rows=[]
    zero_perf=[]
    for r in features.to_dict("records"):
        y=int(r["year"]); rid=str(r["race_id"])
        best_all=None
        best_by_bet={}
        available=0
        for template in TEMPLATES:
            z=perf.get((y,rid,template))
            if not z or z["stake"]<=0:
                continue
            available+=1
            item={
                "template":template,
                "bet_type":TEMPLATE_TO_BET[template],
                "tickets":int(z["tickets"]),
                "stake":float(z["stake"]),
                "ret":float(z["ret"]),
                "profit":float(z["ret"]-z["stake"]),
                "hit":int(z["hit"]),
            }
            if better(item,best_all): best_all=item
            bet=item["bet_type"]
            if better(item,best_by_bet.get(bet)): best_by_bet[bet]=item
        z=dict(r)
        z["available_template_count"]=available
        if best_all is None:
            zero_perf.append({"year":y,"race_id":rid,"race_date":r["race_date"]})
            z["target_label_available"]=0
            z["target_template"]=""
            z["target_bet_type"]=""
            z["target_best_profit_yen"]=None
            for bet in BET_TYPES:
                z[f"target_{bet}_template"]=""
                z[f"target_{bet}_best_profit_yen"]=None
        else:
            z["target_label_available"]=1
            z["target_template"]=best_all["template"]
            z["target_bet_type"]=best_all["bet_type"]
            z["target_best_profit_yen"]=best_all["profit"]
            for bet in BET_TYPES:
                b=best_by_bet.get(bet)
                z[f"target_{bet}_template"]=b["template"] if b else ""
                z[f"target_{bet}_best_profit_yen"]=b["profit"] if b else None
        rows.append(z)
    if zero_perf:
        print("DATA_UNAVAILABLE_NORMAL_RACES="+json.dumps(zero_perf,ensure_ascii=False,separators=(",",":")),flush=True)
    return pd.DataFrame(rows)


def feature_columns(df):
    identity={
        "year","race_id","race_date","target_template","target_bet_type",
        "target_best_profit_yen","available_template_count","target_label_available"
    }
    identity|={f"target_{b}_template" for b in BET_TYPES}
    identity|={f"target_{b}_best_profit_yen" for b in BET_TYPES}
    cols=[c for c in df.columns if c not in identity]
    bad=[c for c in cols if any(tok in c.lower() for tok in FORBIDDEN_TOKENS)]
    if bad:
        raise SystemExit(f"forbidden router features {sorted(bad)}")
    return cols


def encode_fit_other(fit,others,cols):
    cats=[c for c in cols if c in CATEGORICAL]
    nums=[c for c in cols if c not in cats]
    fnum=(fit[nums].apply(pd.to_numeric,errors="coerce")
          .replace([np.inf,-np.inf],np.nan).fillna(-999.0))
    fc=(pd.get_dummies(
        fit[cats].astype("string").fillna("__MISSING__"),
        columns=cats,dummy_na=False,dtype=float
    ) if cats else pd.DataFrame(index=fit.index))
    xf=pd.concat([fnum.reset_index(drop=True),fc.reset_index(drop=True)],axis=1)
    outs=[]
    for frame in others:
        onum=(frame[nums].apply(pd.to_numeric,errors="coerce")
              .replace([np.inf,-np.inf],np.nan).fillna(-999.0))
        oc=(pd.get_dummies(
            frame[cats].astype("string").fillna("__MISSING__"),
            columns=cats,dummy_na=False,dtype=float
        ) if cats else pd.DataFrame(index=frame.index))
        xo=pd.concat([onum.reset_index(drop=True),oc.reset_index(drop=True)],axis=1)
        outs.append(xo.reindex(columns=xf.columns,fill_value=0.0))
    return xf,outs


def train_predict(train,test,target,cols,seed):
    usable=train[train[target].astype(str)!=""].copy()
    if usable.empty:
        raise SystemExit(f"no training rows target={target}")
    labels=usable[target].astype(str)
    le=LabelEncoder()
    y=le.fit_transform(labels)
    if len(le.classes_)==1:
        pred=np.repeat(le.classes_[0],len(test))
        return pred,{"classes":1,"feature_count":len(cols)}

    xfit,(xtest,)=encode_fit_other(usable,[test],cols)
    nclasses=len(le.classes_)
    params=dict(
        n_estimators=220,
        learning_rate=0.03,
        num_leaves=23,
        min_child_samples=60,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=3.0,
        reg_alpha=0.3,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    if nclasses==2:
        model=lgb.LGBMClassifier(objective="binary",**params)
    else:
        model=lgb.LGBMClassifier(objective="multiclass",num_class=nclasses,**params)
    model.fit(xfit,y)
    yp=np.asarray(model.predict(xtest),dtype=int)
    pred=le.inverse_transform(yp)
    return pred,{"classes":nclasses,"feature_count":int(xfit.shape[1])}


def choose_available(y,rid,pred_template,perf):
    z=perf.get((y,rid,pred_template))
    if z and z["stake"]>0:
        return pred_template,z,False
    # Data-availability fallback only; deterministic and outcome-blind.
    same_bet=TEMPLATE_TO_BET.get(pred_template)
    ordered=[t for t in TEMPLATES if TEMPLATE_TO_BET[t]==same_bet]
    ordered+=[t for t in TEMPLATES if t not in ordered]
    for t in ordered:
        q=perf.get((y,rid,t))
        if q and q["stake"]>0:
            return t,q,True
    return "",None,True


def concentration(rows):
    if not rows:
        return {"top1_return_share_pct":None,"top5_return_share_pct":None,"roi_without_top1_pct":None}
    total_ret=sum(x["ret"] for x in rows)
    total_stake=sum(x["stake"] for x in rows)
    vals=sorted((x["ret"] for x in rows),reverse=True)
    top1=vals[0] if vals else 0.0
    top5=sum(vals[:5])
    return {
        "top1_return_share_pct":100*top1/total_ret if total_ret>0 else None,
        "top5_return_share_pct":100*top5/total_ret if total_ret>0 else None,
        "roi_without_top1_pct":100*max(0.0,total_ret-top1)/total_stake if total_stake else None,
    }


def max_drawdown(rows):
    rows=sorted(rows,key=lambda x:(x["race_date"],x["race_id"]))
    cur=0.0; peak=0.0; dd=0.0
    for x in rows:
        cur+=x["ret"]-x["stake"]
        peak=max(peak,cur)
        dd=max(dd,peak-cur)
    return dd


def evaluate(test,pred_templates,architecture,perf):
    out=[]
    fallbacks=0
    exact=[]
    bet_ok=[]
    for r,pred in zip(test.to_dict("records"),pred_templates):
        y=int(r["year"]); rid=str(r["race_id"])
        chosen,z,fb=choose_available(y,rid,str(pred),perf)
        fallbacks+=int(fb)
        if z is None:
            stake=ret=0.0; tickets=0; hit=0
        else:
            stake=float(z["stake"]); ret=float(z["ret"]); tickets=int(z["tickets"]); hit=int(z["hit"])
        if int(r.get("target_label_available") or 0)==1:
            exact.append(int(chosen==r["target_template"]))
            bet_ok.append(int(TEMPLATE_TO_BET.get(chosen)==r["target_bet_type"]))
        out.append({
            "race_id":rid,"race_date":r["race_date"],
            "stake":stake,"ret":ret,"tickets":tickets,"hit":hit,
        })
    stake=sum(x["stake"] for x in out)
    ret=sum(x["ret"] for x in out)
    n=len(out)
    executed=sum(x["stake"]>0 for x in out)
    return {
        "architecture":architecture,
        "test_year":int(test["year"].iloc[0]),
        "source_races":n,
        "routed_races":n,
        "route_coverage_pct":100.0 if n else 0.0,
        "executed_races":executed,
        "execution_coverage_pct":100*executed/n if n else 0.0,
        "data_availability_fallback_races":fallbacks,
        "tickets":sum(x["tickets"] for x in out),
        "avg_tickets_per_race":sum(x["tickets"] for x in out)/n if n else None,
        "hit_races":sum(x["hit"] for x in out),
        "race_hit_rate_pct":100*sum(x["hit"] for x in out)/n if n else None,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(out),
        "labeled_races":len(exact),
        "unlabeled_data_unavailable_races":n-len(exact),
        "target_template_accuracy_pct":100*sum(exact)/len(exact) if exact else None,
        "target_bet_type_accuracy_pct":100*sum(bet_ok)/len(bet_ok) if bet_ok else None,
        **concentration(out),
    }


def evaluate_fixed(test,template,name,perf):
    return evaluate(test,[template]*len(test),name,perf)


def evaluate_oracle(test,perf):
    return evaluate(test,test["target_template"].astype(str).to_numpy(),"ORACLE_DIAGNOSTIC",perf)


def multiclass_quality(y_true,y_pred,prefix=""):
    return {
        prefix+"accuracy":float(accuracy_score(y_true,y_pred)),
        prefix+"macro_f1":float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
    }


def main():
    a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L2_NORMAL_ROUTER_V1":
        raise SystemExit("wrong contract")
    if contract["scope"]["locked_years"]!=[2026]:
        raise SystemExit("2026 lock drift")
    if contract["scope"]["race_filtering"] is not False:
        raise SystemExit("race filtering must stay disabled")
    if contract["scope"]["force_route_every_normal_race"] is not True:
        raise SystemExit("forced routing guard broken")

    paths=parse_paths(a.router_year)
    if set(paths)!=set(YEARS):
        raise SystemExit(f"router years mismatch {sorted(paths)}")
    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={y:load_router(paths[y],y) for y in YEARS}

    features=build_feature_rows(routers,fixed)
    expected={y:len(routers[y])-len(fixed[y]) for y in ANALYSIS_YEARS}
    actual=features.groupby("year")["race_id"].nunique().to_dict()
    if any(actual.get(y)!=expected[y] for y in ANALYSIS_YEARS):
        raise SystemExit(f"normal feature universe drift actual={actual} expected={expected}")

    manifest,perf=load_performance(a.dataset_dir)
    data=attach_targets(features,perf)
    after=data.groupby("year")["race_id"].nunique().to_dict()
    if any(after.get(y)!=expected[y] for y in ANALYSIS_YEARS):
        raise SystemExit(f"target attachment silently dropped races actual={after} expected={expected}")

    cols=feature_columns(data)
    quality_rows=[]
    metric_rows=[]
    decision_rows=[]

    for test_year in TEST_YEARS:
        train=data[data["year"]<test_year].copy()
        test=data[data["year"]==test_year].copy().sort_values(["race_date","race_id"]).reset_index(drop=True)

        direct_pred,meta=train_predict(train,test,"target_template",cols,81000+test_year)
        labeled=test["target_label_available"].astype(int)==1
        q=multiclass_quality(test.loc[labeled,"target_template"].astype(str),np.asarray(direct_pred)[labeled.to_numpy()],"template_")
        quality_rows.append({
            "architecture":"DIRECT_TEMPLATE","test_year":test_year,
            "train_years":"|".join(map(str,sorted(train["year"].unique()))),
            "train_races":len(train),"test_races":len(test),
            **meta,**q,
            "labeled_test_races":int(labeled.sum()),
            "unlabeled_data_unavailable_test_races":int((~labeled).sum()),
            **multiclass_quality(
                test.loc[labeled,"target_bet_type"].astype(str),
                np.asarray([TEMPLATE_TO_BET.get(x,"") for x in direct_pred])[labeled.to_numpy()],
                "bet_"
            ),
        })
        metric_rows.append(evaluate(test,direct_pred,"DIRECT_TEMPLATE",perf))

        bet_pred,bet_meta=train_predict(train,test,"target_bet_type",cols,82000+test_year)
        final=[]
        stage2_meta={}
        for bet in BET_TYPES:
            idx=np.where(np.asarray(bet_pred,dtype=str)==bet)[0]
            if len(idx)==0:
                continue
            subtest=test.iloc[idx].copy()
            target=f"target_{bet}_template"
            pred,m=train_predict(train,subtest,target,cols,83000+test_year+BET_TYPES.index(bet)*100)
            for i,p in zip(idx,pred):
                final.append((int(i),str(p)))
            stage2_meta[bet]=m
        final_sorted=[""]*len(test)
        for i,p in final: final_sorted[i]=p
        # If a predicted bet has no stage2 class, use deterministic first template in that bet.
        for i,p in enumerate(final_sorted):
            if not p:
                b=str(bet_pred[i])
                final_sorted[i]=next(t for t in TEMPLATES if TEMPLATE_TO_BET[t]==b)

        hq=multiclass_quality(test.loc[labeled,"target_template"].astype(str),np.asarray(final_sorted)[labeled.to_numpy()],"template_")
        quality_rows.append({
            "architecture":"HIERARCHICAL","test_year":test_year,
            "train_years":"|".join(map(str,sorted(train["year"].unique()))),
            "train_races":len(train),"test_races":len(test),
            "classes":bet_meta["classes"],
            "feature_count":bet_meta["feature_count"],
            **hq,
            "labeled_test_races":int(labeled.sum()),
            "unlabeled_data_unavailable_test_races":int((~labeled).sum()),
            **multiclass_quality(test.loc[labeled,"target_bet_type"].astype(str),np.asarray(bet_pred)[labeled.to_numpy()],"bet_"),
            "stage2_models":"|".join(sorted(stage2_meta)),
        })
        metric_rows.append(evaluate(test,final_sorted,"HIERARCHICAL",perf))

        metric_rows.append(evaluate_fixed(test,BASELINE_TEMPLATE,"FIXED_BASELINE",perf))
        metric_rows.append(evaluate_oracle(test,perf))

        for r,dp,hp,bp in zip(test.to_dict("records"),direct_pred,final_sorted,bet_pred):
            decision_rows.append({
                "year":test_year,"race_id":r["race_id"],"race_date":r["race_date"],
                "target_label_available":int(r.get("target_label_available") or 0),
                "available_template_count":int(r.get("available_template_count") or 0),
                "target_template":r["target_template"],
                "target_bet_type":r["target_bet_type"],
                "direct_template":str(dp),
                "hier_bet_type":str(bp),
                "hier_template":str(hp),
            })

    # Development-only architecture selection.
    candidates=[]
    for arch in ("DIRECT_TEMPLATE","HIERARCHICAL"):
        rows=[r for r in metric_rows if r["architecture"]==arch and int(r["test_year"]) in DEV_YEARS]
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
            "dev_route_coverage_pct":min(r["route_coverage_pct"] for r in rows),
        })
    candidates.sort(key=lambda r:(
        -r["dev_profit_yen"],
        -r["min_dev_year_roi_pct"],
        r["sum_dev_max_drawdown_yen"],
        r["architecture"],
    ))
    chosen=candidates[0]["architecture"]
    for i,r in enumerate(candidates):
        r["selected_on_development"]=int(i==0)

    holdout=next(r for r in metric_rows if r["architecture"]==chosen and int(r["test_year"])==HOLDOUT_YEAR)
    fixed_holdout=next(r for r in metric_rows if r["architecture"]=="FIXED_BASELINE" and int(r["test_year"])==HOLDOUT_YEAR)

    class_rows=[]
    for year in ANALYSIS_YEARS:
        d=data[data["year"]==year]
        labeled_d=d[d["target_label_available"].astype(int)==1]
        for col in ("target_bet_type","target_template"):
            for label,count in labeled_d[col].astype(str).value_counts().items():
                class_rows.append({
                    "year":year,"target":col,"label":label,
                    "races":int(count),"share_pct":100*count/len(labeled_d) if len(labeled_d) else None
                })

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"model-quality.csv",quality_rows)
    write_csv(out/"router-metrics.csv",metric_rows)
    write_csv(out/"development-selection.csv",candidates)
    write_csv(out/"class-distribution.csv",class_rows)
    with gzip.open(out/"route-decisions.csv.gz","wt",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(decision_rows[0]))
        w.writeheader(); w.writerows(decision_rows)

    summary={
        "contract":"L2_NORMAL_ROUTER_RESULT_V1",
        "source_loss_anatomy_run":36526102078,
        "source_structure_run":36524917751,
        "scope":"PASS_SEVEN_ONLY forced routing; no learned skip",
        "normal_races":{str(y):expected[y] for y in ANALYSIS_YEARS},
        "architectures":["DIRECT_TEMPLATE","HIERARCHICAL"],
        "fixed_baseline":BASELINE_TEMPLATE,
        "development_selection_years":[2023,2024],
        "selected_architecture":chosen,
        "holdout_year":2025,
        "holdout_used_for_selection":False,
        "selected_architecture_holdout_2025":holdout,
        "fixed_baseline_holdout_2025":fixed_holdout,
        "feature_market_leakage":False,
        "data_unavailable_races":{str(y):int(((data["year"]==y)&(data["target_label_available"].astype(int)==0)).sum()) for y in ANALYSIS_YEARS},
        "target_uses_realized_profit":True,
        "race_filtering":False,
        "route_every_normal_race":True,
        "2026_locked":True,
        "production_promotion":False,
        "interpretation_guard":"The target is a hindsight best-template label. Router metrics are walk-forward research evidence, not a production strategy. ORACLE_DIAGNOSTIC is an unreachable upper bound."
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Normal Router V1\n\n"
        "PASS_SEVEN_ONLY only. Every normal race is routed; V1 has no learned SKIP class. "
        "Two predeclared architectures are compared: DIRECT_TEMPLATE and HIERARCHICAL (bet type then template). "
        "Features come only from pre-race Seven-King structure and race metadata. Odds, payout, returns and results are not model inputs. "
        "The supervised target is the realized best/least-loss fixed template when historical price data exists. "
        "Races with no priced template are retained in the universe, never silently dropped, are routed normally, and are marked DATA_UNAVAILABLE for evaluation. "
        "Architecture selection uses only 2023-2024 walk-forward results; 2025 is holdout-only. "
        "2026 remains sealed. No production promotion.\n",
        encoding="utf-8",
    )
    print("L2_NORMAL_ROUTER_V1_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)


if __name__=="__main__":
    main()
