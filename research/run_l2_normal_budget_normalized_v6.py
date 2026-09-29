#!/usr/bin/env python3
import argparse,csv,gzip,json
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import YEARS
from run_l2_normal_router_v1 import (
    ANALYSIS_YEARS,DEV_YEARS,HOLDOUT_YEAR,
    load_fixed_ledgers,load_router,build_feature_rows,load_performance,
    attach_targets,feature_columns,choose_available,evaluate
)
from run_l2_normal_pairwise_router_v4 import parse_paths,read_csv_rows
from run_l2_normal_confidence_override_v5 import (
    THRESHOLDS,build_fold,apply_candidate,candidate_name,perf_delta_vs_base
)

def parse_args():
    p=argparse.ArgumentParser(description="Normal Budget-Normalized Router V6.")
    p.add_argument("--contract",required=True)
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--v1-metrics",required=False)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

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

def normalized_drawdown(rows):
    cur=0.0; peak=0.0; dd=0.0
    for r in sorted(rows,key=lambda x:(x["race_date"],x["race_id"])):
        cur+=r["normalized_profit_units"]
        peak=max(peak,cur)
        dd=max(dd,peak-cur)
    return dd

def normalized_concentration(rows):
    if not rows:
        return {
            "normalized_top1_return_share_pct":None,
            "normalized_top5_return_share_pct":None,
            "normalized_roi_without_top1_pct":None,
        }
    vals=sorted((r["normalized_return_units"] for r in rows),reverse=True)
    total=sum(vals)
    stake=float(len(rows))
    top1=vals[0] if vals else 0.0
    top5=sum(vals[:5])
    return {
        "normalized_top1_return_share_pct":100*top1/total if total>0 else None,
        "normalized_top5_return_share_pct":100*top5/total if total>0 else None,
        "normalized_roi_without_top1_pct":100*max(0.0,total-top1)/stake if stake else None,
    }

def normalized_evaluate(test,preds,name,perf):
    rows=[]
    unavailable=0
    for r,pred in zip(test.to_dict("records"),preds):
        y=int(r["year"]); rid=str(r["race_id"])
        chosen,z,_=choose_available(y,rid,str(pred),perf)
        if z is None or float(z["stake"])<=0:
            unavailable+=1
            continue
        raw_stake=float(z["stake"])
        raw_return=float(z["ret"])
        norm_return=raw_return/raw_stake
        rows.append({
            "race_id":rid,
            "race_date":r["race_date"],
            "template":chosen,
            "tickets":int(z["tickets"]),
            "raw_stake_yen":raw_stake,
            "raw_return_yen":raw_return,
            "normalized_stake_units":1.0,
            "normalized_return_units":norm_return,
            "normalized_profit_units":norm_return-1.0,
        })
    n=len(test); executed=len(rows)
    ret=sum(r["normalized_return_units"] for r in rows)
    profit=sum(r["normalized_profit_units"] for r in rows)
    result={
        "candidate":name,
        "test_year":int(test["year"].iloc[0]),
        "source_races":n,
        "executed_races":executed,
        "data_unavailable_races":unavailable,
        "normalized_stake_units":float(executed),
        "normalized_return_units":ret,
        "normalized_profit_units":profit,
        "normalized_roi_pct":100*ret/executed if executed else None,
        "normalized_max_drawdown_units":normalized_drawdown(rows),
        "avg_tickets_per_source_race":sum(r["tickets"] for r in rows)/n if n else None,
        **normalized_concentration(rows),
    }
    return result,rows

def merge_raw(norm,raw):
    out=dict(norm)
    for k,v in raw.items():
        if k in {"architecture","test_year","source_races","executed_races"}:
            continue
        out["raw_"+k]=v
    return out

def main():
    a=parse_args()
    c=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if c.get("contract")!="L2_NORMAL_BUDGET_NORMALIZED_V6": raise SystemExit("wrong contract")
    if c["scope"]["battlefield"]!="PASS_SEVEN_ONLY": raise SystemExit("battlefield drift")
    if c["scope"]["race_filtering"] is not False: raise SystemExit("race filtering forbidden")
    if c["scope"]["force_route_every_normal_race"] is not True: raise SystemExit("forced routing guard broken")
    if c["scope"]["skip_class"] is not False: raise SystemExit("skip forbidden")
    if c["scope"]["locked_years"]!=[2026]: raise SystemExit("2026 lock drift")
    if c["normalization"]["rule"]!="Each executed race receives exactly 1.0 research budget unit regardless of ticket count.":
        raise SystemExit("normalization drift")
    if c["candidates"]["pairwise_allowed_market_features"] is not False: raise SystemExit("market leakage contract broken")
    if c["selection"]["development_years"]!=[2023,2024]: raise SystemExit("dev years drift")
    if c["selection"]["final_holdout_year"]!=2025: raise SystemExit("holdout drift")
    if c["selection"]["raw_profit_not_used_for_selection"] is not True: raise SystemExit("raw-profit selection forbidden")
    if c["selection"]["holdout_used_for_selection"] is not False: raise SystemExit("holdout leak")

    paths=parse_paths(a.router_year)
    if set(paths)!=set(YEARS): raise SystemExit(f"router years mismatch {sorted(paths)}")
    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={y:load_router(paths[y],y) for y in YEARS}
    features=build_feature_rows(routers,fixed)
    expected={y:len(routers[y])-len(fixed[y]) for y in ANALYSIS_YEARS}
    actual=features.groupby("year")["race_id"].nunique().to_dict()
    if any(actual.get(y)!=expected[y] for y in ANALYSIS_YEARS):
        raise SystemExit(f"normal universe drift actual={actual} expected={expected}")

    _,perf=load_performance(a.dataset_dir)
    data=attach_targets(features,perf)
    cols=feature_columns(data)

    refs={}
    if a.v1_metrics and Path(a.v1_metrics).exists():
        for r in read_csv_rows(a.v1_metrics):
            if r.get("architecture")=="DIRECT_TEMPLATE":
                refs[int(r["test_year"])]=r

    dev_folds={}
    fold_rows=[]
    dev_rows=[]

    for y in DEV_YEARS:
        test,base,meta,v1raw,fold=build_fold(y,data,cols,perf,refs)
        dev_folds[y]=(test,base,meta)
        fold_rows.append(fold)

        n0,_=normalized_evaluate(test,base,"V1_NO_OVERRIDE",perf)
        n0.update({"shortlist_k":1,"threshold":None,"override_races":0,"override_rate_pct":0.0})
        dev_rows.append(merge_raw(n0,v1raw))

        for k in (2,3):
            for t in THRESHOLDS:
                preds,_=apply_candidate(base,meta,k,t)
                name=candidate_name(k,t)
                raw=evaluate(test,preds,name,perf)
                norm,_=normalized_evaluate(test,preds,name,perf)
                delta=perf_delta_vs_base(test,preds,base,perf)
                norm.update({
                    "shortlist_k":k,
                    "threshold":t,
                    "override_races":delta["override_races"],
                    "override_rate_pct":delta["override_rate_pct"],
                    "override_profit_delta_vs_v1_yen":delta["override_profit_delta_vs_v1_yen"],
                    "override_improved_races":delta["override_improved_races"],
                    "override_worsened_races":delta["override_worsened_races"],
                })
                dev_rows.append(merge_raw(norm,raw))

    candidate_ids=["V1_NO_OVERRIDE"]+[candidate_name(k,t) for k in (2,3) for t in THRESHOLDS]
    selection=[]
    for cid in candidate_ids:
        rows=[r for r in dev_rows if r["candidate"]==cid]
        selection.append({
            "candidate":cid,
            "shortlist_k":rows[0]["shortlist_k"],
            "threshold":rows[0]["threshold"],
            "development_years":"2023|2024",
            "dev_normalized_stake_units":sum(r["normalized_stake_units"] for r in rows),
            "dev_normalized_return_units":sum(r["normalized_return_units"] for r in rows),
            "dev_normalized_profit_units":sum(r["normalized_profit_units"] for r in rows),
            "dev_normalized_roi_pct":100*sum(r["normalized_return_units"] for r in rows)/sum(r["normalized_stake_units"] for r in rows),
            "min_dev_year_normalized_roi_pct":min(r["normalized_roi_pct"] for r in rows),
            "sum_dev_normalized_max_drawdown_units":sum(r["normalized_max_drawdown_units"] for r in rows),
            "dev_override_races":sum(int(r["override_races"]) for r in rows),
            "dev_raw_tickets":sum(int(r["raw_tickets"]) for r in rows),
            "dev_raw_profit_yen_diagnostic":sum(float(r["raw_profit_yen"]) for r in rows),
        })

    selection.sort(key=lambda r:(
        -r["dev_normalized_profit_units"],
        -r["min_dev_year_normalized_roi_pct"],
        r["sum_dev_normalized_max_drawdown_units"],
        r["dev_override_races"],
        r["dev_raw_tickets"],
        r["candidate"],
    ))
    chosen=selection[0]
    for i,r in enumerate(selection):
        r["selected_on_development"]=int(i==0)

    # Open 2025 only after normalized development selection is fixed.
    test25,base25,meta25,v1raw25,fold25=build_fold(HOLDOUT_YEAR,data,cols,perf,refs)
    fold_rows.append(fold25)

    base_norm25,base_detail25=normalized_evaluate(test25,base25,"V1_NO_OVERRIDE",perf)
    base_norm25.update({"shortlist_k":1,"threshold":None,"override_races":0,"override_rate_pct":0.0})
    base25row=merge_raw(base_norm25,v1raw25)

    if chosen["candidate"]=="V1_NO_OVERRIDE":
        pred25=list(base25)
    else:
        pred25,_=apply_candidate(base25,meta25,int(chosen["shortlist_k"]),float(chosen["threshold"]))
    raw25=evaluate(test25,pred25,chosen["candidate"],perf)
    norm25,detail25=normalized_evaluate(test25,pred25,chosen["candidate"],perf)
    delta25=perf_delta_vs_base(test25,pred25,base25,perf)
    norm25.update({
        "shortlist_k":chosen["shortlist_k"],
        "threshold":chosen["threshold"],
        "override_races":delta25["override_races"],
        "override_rate_pct":delta25["override_rate_pct"],
        "override_profit_delta_vs_v1_yen":delta25["override_profit_delta_vs_v1_yen"],
        "override_improved_races":delta25["override_improved_races"],
        "override_worsened_races":delta25["override_worsened_races"],
    })
    selected25row=merge_raw(norm25,raw25)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"development-normalized-sweep.csv",dev_rows)
    write_csv(out/"development-selection.csv",selection)
    write_csv(out/"holdout-2025.csv",[base25row,selected25row])
    write_csv(out/"fold-status.csv",fold_rows)
    write_csv(out/"holdout-normalized-race-detail-v1.csv",base_detail25)
    write_csv(out/"holdout-normalized-race-detail-selected.csv",detail25)

    summary={
        "contract":"L2_NORMAL_BUDGET_NORMALIZED_RESULT_V6",
        "scope":"PASS_SEVEN_ONLY forced routing; no skip",
        "normalization":"1.0 research budget unit per executed race regardless of ticket count",
        "selection_metric":"development normalized profit units",
        "raw_profit_used_for_selection":False,
        "development_selection_years":[2023,2024],
        "selected_candidate":chosen,
        "holdout_year":2025,
        "holdout_used_for_selection":False,
        "v1_holdout_2025":base25row,
        "selected_holdout_2025":selected25row,
        "race_filtering":False,
        "route_every_normal_race":True,
        "skip_class":False,
        "2026_locked":True,
        "production_promotion":False,
        "interpretation_guard":"Every executed race receives equal research capital. Raw yen/ticket metrics are diagnostics only and do not select the candidate. Threshold/shortlist selection uses 2023-2024 only; 2025 is opened afterward. This is research normalization, not an executable staking plan."
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Normal Budget-Normalized V6\n\n"
        "This experiment removes the V5 bias where fewer tickets automatically reduce total loss. "
        "Each executed race receives exactly one research budget unit, and return is scaled by realized return/stake. "
        "V1_NO_OVERRIDE and Top2/Top3 confidence overrides are selected on 2023-2024 using normalized profit only. "
        "Raw yen and ticket metrics remain diagnostics and are not used for selection. "
        "Only after the development winner is fixed is 2025 evaluated. No SKIP class exists and 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("L2_NORMAL_BUDGET_NORMALIZED_V6_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
