#!/usr/bin/env python3
import argparse,csv,gzip,json
from itertools import combinations
from pathlib import Path

import numpy as np

from build_l2_bet_kings_dataset_v1 import YEARS
from run_l2_normal_router_v1 import (
    ANALYSIS_YEARS,DEV_YEARS,HOLDOUT_YEAR,TEMPLATES,
    load_fixed_ledgers,load_router,build_feature_rows,load_performance,
    attach_targets,feature_columns,choose_available,evaluate
)
from run_l2_normal_pairwise_router_v4 import (
    parse_paths,read_csv_rows,train_v1_shortlist,train_pairwise,pair_vector
)

THRESHOLDS=(0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95)

def parse_args():
    p=argparse.ArgumentParser(description="Normal Confidence Override V5.")
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

def resolved_template(y,rid,pred,perf):
    chosen,_,_=choose_available(y,rid,str(pred),perf)
    return chosen or str(pred)

def pair_prob_a_better(xrace,ta,tb,za,zb,model,templates):
    x,ca,cb=pair_vector(xrace,ta,tb,za,zb,templates)
    pca=float(model.predict_proba(x.reshape(1,-1))[0,1])
    return pca if ca==ta else 1.0-pca

def shortlist_winner(y,rid,candidates,probs,xrace,model,perf,templates):
    avail=[]
    for rank,(t,p) in enumerate(zip(candidates,probs),start=1):
        t=str(t)
        z=perf.get((y,rid,t))
        if z and float(z["stake"])>0:
            avail.append((t,float(p),z,rank))
    if not avail:
        return {
            "winner":"",
            "winner_rank":None,
            "winner_stage1_prob":None,
            "base":"",
            "base_stage1_prob":None,
            "alt_vs_base_confidence":None,
            "pairwise_changed":0,
        }

    base=resolved_template(y,rid,str(candidates[0]),perf)
    bz=perf.get((y,rid,base))
    if not bz:
        base=avail[0][0]; bz=avail[0][2]

    if len(avail)==1:
        return {
            "winner":avail[0][0],
            "winner_rank":avail[0][3],
            "winner_stage1_prob":avail[0][1],
            "base":base,
            "base_stage1_prob":float(probs[0]),
            "alt_vs_base_confidence":None,
            "pairwise_changed":int(avail[0][0]!=base),
        }

    wins={t:0 for t,_,_,_ in avail}
    psum={t:0.0 for t,_,_,_ in avail}
    pstage1={t:p for t,p,_,_ in avail}
    rankmap={t:r for t,_,_,r in avail}
    tickets={t:int(z["tickets"]) for t,_,z,_ in avail}

    for (ta,_,za,_),(tb,_,zb,_) in combinations(avail,2):
        pta=pair_prob_a_better(xrace,ta,tb,za,zb,model,templates)
        ptb=1.0-pta
        psum[ta]+=pta; psum[tb]+=ptb
        if pta>=0.5: wins[ta]+=1
        else: wins[tb]+=1

    winner=sorted(
        wins,
        key=lambda t:(-wins[t],-psum[t],-pstage1[t],tickets[t],t)
    )[0]

    conf=None
    if winner!=base:
        wz=perf.get((y,rid,winner))
        if wz and bz:
            conf=pair_prob_a_better(xrace,winner,base,wz,bz,model,templates)

    return {
        "winner":winner,
        "winner_rank":rankmap[winner],
        "winner_stage1_prob":pstage1[winner],
        "base":base,
        "base_stage1_prob":float(probs[0]),
        "alt_vs_base_confidence":conf,
        "pairwise_changed":int(winner!=base),
    }

def build_fold(test_year,data,cols,perf,refs):
    train=data[data["year"]<test_year].copy().reset_index(drop=True)
    test=data[data["year"]==test_year].copy().sort_values(["race_date","race_id"]).reset_index(drop=True)

    labels,probs,_=train_v1_shortlist(train,test,cols,81000+test_year)
    pair_model,xtest_pair,pmeta=train_pairwise(train,test,cols,perf,TEMPLATES,131000+test_year)

    expected_pair=int(pair_model.n_features_in_)
    actual_pair=int(xtest_pair.shape[1]+2*len(TEMPLATES)+4)
    if expected_pair!=actual_pair:
        raise SystemExit(f"pairwise feature drift year={test_year} expected={expected_pair} actual={actual_pair}")

    base_preds=[]
    meta=[]
    for i,r in enumerate(test.to_dict("records")):
        y=int(r["year"]); rid=str(r["race_id"])
        classes=[str(x) for x in labels[i]]
        pp=[float(x) for x in probs[i]]
        base=resolved_template(y,rid,classes[0],perf)
        top2=shortlist_winner(y,rid,classes[:2],pp[:2],xtest_pair[i],pair_model,perf,TEMPLATES)
        top3=shortlist_winner(y,rid,classes[:3],pp[:3],xtest_pair[i],pair_model,perf,TEMPLATES)
        base_preds.append(base)
        meta.append({
            "year":y,"race_id":rid,"race_date":r["race_date"],
            "v1_top1":base,"v1_top1_raw":classes[0],"v1_top1_prob":pp[0],
            "top2_winner":top2["winner"],
            "top2_winner_rank":top2["winner_rank"],
            "top2_confidence":top2["alt_vs_base_confidence"],
            "top2_changed":top2["pairwise_changed"],
            "top3_winner":top3["winner"],
            "top3_winner_rank":top3["winner_rank"],
            "top3_confidence":top3["alt_vs_base_confidence"],
            "top3_changed":top3["pairwise_changed"],
        })

    m=evaluate(test,base_preds,"V1_NO_OVERRIDE",perf)
    ref=refs.get(test_year)
    if ref:
        drift=abs(float(ref["roi_pct"])-float(m["roi_pct"]))
        if drift>0.02:
            raise SystemExit(f"V1 reproduction drift year={test_year} ref={ref['roi_pct']} now={m['roi_pct']}")
    fold={
        "test_year":test_year,
        "train_years":"|".join(map(str,sorted(train["year"].unique()))),
        "train_races":int(train["race_id"].nunique()),
        "test_races":int(test["race_id"].nunique()),
        "v1_reference_roi_pct":float(ref["roi_pct"]) if ref else None,
        "v1_reproduced_roi_pct":m["roi_pct"],
        **pmeta,
    }
    return test,base_preds,meta,m,fold

def candidate_name(k,threshold):
    return f"TOP{k}_OVERRIDE_T{int(round(threshold*100)):02d}"

def apply_candidate(base_preds,meta,k,threshold):
    out=[]
    overrides=0
    for base,r in zip(base_preds,meta):
        winner=r[f"top{k}_winner"]
        conf=r[f"top{k}_confidence"]
        if winner and winner!=base and conf is not None and float(conf)>=threshold:
            out.append(winner); overrides+=1
        else:
            out.append(base)
    return out,overrides

def perf_delta_vs_base(test,preds,base_preds,perf):
    delta=0.0; changed=0; improved=0; worsened=0
    for r,p,b in zip(test.to_dict("records"),preds,base_preds):
        y=int(r["year"]); rid=str(r["race_id"])
        rp, zp,_=choose_available(y,rid,p,perf)
        rb, zb,_=choose_available(y,rid,b,perf)
        if rp!=rb:
            changed+=1
            pp=(float(zp["ret"])-float(zp["stake"])) if zp else 0.0
            pb=(float(zb["ret"])-float(zb["stake"])) if zb else 0.0
            d=pp-pb
            delta+=d
            improved+=int(d>0); worsened+=int(d<0)
    return {
        "override_races":changed,
        "override_rate_pct":100*changed/len(base_preds) if base_preds else 0.0,
        "override_profit_delta_vs_v1_yen":delta,
        "override_improved_races":improved,
        "override_worsened_races":worsened,
    }

def main():
    a=parse_args()
    c=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if c.get("contract")!="L2_NORMAL_CONFIDENCE_OVERRIDE_V5": raise SystemExit("wrong contract")
    if c["scope"]["battlefield"]!="PASS_SEVEN_ONLY": raise SystemExit("battlefield drift")
    if c["scope"]["race_filtering"] is not False: raise SystemExit("race filtering forbidden")
    if c["scope"]["force_route_every_normal_race"] is not True: raise SystemExit("forced routing guard broken")
    if c["scope"]["skip_class"] is not False: raise SystemExit("skip forbidden")
    if c["scope"]["locked_years"]!=[2026]: raise SystemExit("2026 lock drift")
    if c["override"]["allowed_market_features"] is not False: raise SystemExit("market leakage contract broken")
    if c["selection"]["development_years"]!=[2023,2024]: raise SystemExit("dev years drift")
    if c["selection"]["final_holdout_year"]!=2025: raise SystemExit("holdout drift")
    if c["selection"]["holdout_used_for_selection"] is not False: raise SystemExit("holdout selection leak")

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
    dev_metrics=[]
    all_meta=[]
    for y in DEV_YEARS:
        test,base,meta,v1m,fold=build_fold(y,data,cols,perf,refs)
        dev_folds[y]=(test,base,meta)
        fold_rows.append(fold); all_meta.extend(meta)
        v1m.update({
            "candidate":"V1_NO_OVERRIDE",
            "shortlist_k":1,"threshold":None,
            "override_races":0,"override_rate_pct":0.0,
            "override_profit_delta_vs_v1_yen":0.0,
            "override_improved_races":0,"override_worsened_races":0,
        })
        dev_metrics.append(v1m)
        for k in (2,3):
            for t in THRESHOLDS:
                preds,n=apply_candidate(base,meta,k,t)
                name=candidate_name(k,t)
                m=evaluate(test,preds,name,perf)
                m.update({"candidate":name,"shortlist_k":k,"threshold":t})
                m.update(perf_delta_vs_base(test,preds,base,perf))
                dev_metrics.append(m)

    candidate_ids=["V1_NO_OVERRIDE"]+[candidate_name(k,t) for k in (2,3) for t in THRESHOLDS]
    selection=[]
    for cid in candidate_ids:
        rows=[r for r in dev_metrics if r["candidate"]==cid]
        stake=sum(float(r["stake_yen"]) for r in rows)
        ret=sum(float(r["return_yen"]) for r in rows)
        overrides=sum(int(r["override_races"]) for r in rows)
        tickets=sum(int(r["tickets"]) for r in rows)
        selection.append({
            "candidate":cid,
            "shortlist_k":rows[0]["shortlist_k"],
            "threshold":rows[0]["threshold"],
            "development_years":"2023|2024",
            "dev_stake_yen":stake,"dev_return_yen":ret,"dev_profit_yen":ret-stake,
            "dev_roi_pct":100*ret/stake if stake else None,
            "min_dev_year_roi_pct":min(float(r["roi_pct"]) for r in rows),
            "sum_dev_max_drawdown_yen":sum(float(r["max_drawdown_yen"]) for r in rows),
            "dev_override_races":overrides,
            "dev_tickets":tickets,
        })
    selection.sort(key=lambda r:(
        -r["dev_profit_yen"],
        -r["min_dev_year_roi_pct"],
        r["sum_dev_max_drawdown_yen"],
        r["dev_override_races"],
        r["dev_tickets"],
        r["candidate"],
    ))
    chosen=selection[0]
    for i,r in enumerate(selection): r["selected_on_development"]=int(i==0)

    # Open 2025 only after the development candidate is fixed.
    test25,base25,meta25,v1m25,fold25=build_fold(HOLDOUT_YEAR,data,cols,perf,refs)
    fold_rows.append(fold25)
    if chosen["candidate"]=="V1_NO_OVERRIDE":
        pred25=list(base25)
    else:
        pred25,_=apply_candidate(base25,meta25,int(chosen["shortlist_k"]),float(chosen["threshold"]))
    selected25=evaluate(test25,pred25,chosen["candidate"],perf)
    selected25.update({
        "candidate":chosen["candidate"],
        "shortlist_k":chosen["shortlist_k"],
        "threshold":chosen["threshold"],
    })
    selected25.update(perf_delta_vs_base(test25,pred25,base25,perf))
    v1m25.update({
        "candidate":"V1_NO_OVERRIDE","shortlist_k":1,"threshold":None,
        "override_races":0,"override_rate_pct":0.0,
        "override_profit_delta_vs_v1_yen":0.0,
        "override_improved_races":0,"override_worsened_races":0,
    })

    # Confidence-bin diagnostics use development only.
    bins=[]
    for k in (2,3):
        vals=[]
        for r in all_meta:
            if r[f"top{k}_changed"] and r[f"top{k}_confidence"] is not None:
                vals.append(float(r[f"top{k}_confidence"]))
        for lo,hi in zip(THRESHOLDS[:-1],THRESHOLDS[1:]):
            n=sum(lo<=v<hi for v in vals)
            bins.append({"shortlist_k":k,"confidence_lo":lo,"confidence_hi":hi,"changed_races":n})
        n=sum(v>=THRESHOLDS[-1] for v in vals)
        bins.append({"shortlist_k":k,"confidence_lo":THRESHOLDS[-1],"confidence_hi":1.0,"changed_races":n})

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"development-threshold-sweep.csv",dev_metrics)
    write_csv(out/"development-selection.csv",selection)
    write_csv(out/"fold-status.csv",fold_rows)
    write_csv(out/"confidence-bins-dev.csv",bins)
    write_csv(out/"holdout-2025.csv",[v1m25,selected25])
    with gzip.open(out/"selected-route-decisions-2025.csv.gz","wt",newline="",encoding="utf-8") as f:
        fields=list(meta25[0].keys())+["selected_candidate","selected_template","override_applied"]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for r,b,p in zip(meta25,base25,pred25):
            z=dict(r)
            z["selected_candidate"]=chosen["candidate"]
            z["selected_template"]=p
            z["override_applied"]=int(p!=b)
            w.writerow(z)

    summary={
        "contract":"L2_NORMAL_CONFIDENCE_OVERRIDE_RESULT_V5",
        "scope":"PASS_SEVEN_ONLY forced routing; no skip",
        "normal_races":{str(y):expected[y] for y in ANALYSIS_YEARS},
        "base_router":"V1_TOP1",
        "override_shortlists":[2,3],
        "threshold_grid":list(THRESHOLDS),
        "development_selection_years":[2023,2024],
        "selected_candidate":chosen,
        "holdout_year":2025,
        "holdout_used_for_selection":False,
        "v1_holdout_2025":v1m25,
        "selected_holdout_2025":selected25,
        "pairwise_market_features":False,
        "race_filtering":False,
        "route_every_normal_race":True,
        "skip_class":False,
        "2026_locked":True,
        "production_promotion":False,
        "interpretation_guard":"V1 remains the default route. Pairwise may override only when the development-selected confidence threshold is met. The threshold and shortlist size are selected on 2023-2024 before 2025 is evaluated. Pairwise prediction inputs exclude odds, payout and result-derived inputs."
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Normal Confidence Override V5\n\n"
        "V1_TOP1 is the default for every normal race. Pairwise is only a veto/override layer. "
        "Top2 and Top3 shortlist variants are swept over confidence thresholds 0.50 to 0.95 on 2023-2024. "
        "V1_NO_OVERRIDE is included as a candidate, so the research can explicitly choose to make no changes. "
        "Only after development selection is fixed is 2025 evaluated. No SKIP class exists, every normal race stays in scope, "
        "pairwise prediction inputs exclude market/outcome data, and 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("L2_NORMAL_CONFIDENCE_OVERRIDE_V5_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
