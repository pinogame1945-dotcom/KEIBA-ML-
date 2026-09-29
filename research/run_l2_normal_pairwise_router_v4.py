#!/usr/bin/env python3
import argparse,csv,gzip,json
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

from build_l2_bet_kings_dataset_v1 import YEARS
from run_l2_normal_router_v1 import (
    ANALYSIS_YEARS,TEST_YEARS,DEV_YEARS,HOLDOUT_YEAR,TEMPLATES,
    load_fixed_ledgers,load_router,build_feature_rows,load_performance,
    attach_targets,feature_columns,encode_fit_other,evaluate
)

ARCHS=("V1_TOP1","TOP2_PAIRWISE","TOP3_PAIRWISE")


def parse_args():
    p=argparse.ArgumentParser(description="Normal Pairwise Router V4.")
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


def train_v1_shortlist(train,test,cols,seed):
    usable=train[train["target_template"].astype(str)!=""].copy()
    labels=usable["target_template"].astype(str)
    le=LabelEncoder()
    y=le.fit_transform(labels)
    xfit,(xtest,)=encode_fit_other(usable,[test],cols)
    nclasses=len(le.classes_)
    params=dict(
        n_estimators=220,learning_rate=0.03,num_leaves=23,min_child_samples=60,
        subsample=0.90,colsample_bytree=0.90,reg_lambda=3.0,reg_alpha=0.3,
        random_state=seed,n_jobs=2,deterministic=True,force_col_wise=True,verbosity=-1,
    )
    if nclasses==2:
        model=lgb.LGBMClassifier(objective="binary",**params)
    else:
        model=lgb.LGBMClassifier(objective="multiclass",num_class=nclasses,**params)
    model.fit(xfit,y)
    prob=np.asarray(model.predict_proba(xtest),dtype=float)
    if prob.ndim==1:
        prob=np.column_stack([1-prob,prob])
    order=np.argsort(-prob,axis=1)
    labels_by_rank=np.asarray(le.classes_,dtype=object)[order]
    probs_by_rank=np.take_along_axis(prob,order,axis=1)
    return labels_by_rank,probs_by_rank,np.asarray(xtest,dtype=np.float32)


def perf_key(z,t):
    return (float(z["ret"]-z["stake"]),-float(z["stake"]),-int(z["tickets"]),str(t))


def build_pair_training(train,xtrain,perf,templates):
    tindex={t:i for i,t in enumerate(templates)}
    race_idx=[]; aidx=[]; bidx=[]; atix=[]; btix=[]; labels=[]
    for i,r in enumerate(train.to_dict("records")):
        y=int(r["year"]); rid=str(r["race_id"])
        avail=[]
        for t in templates:
            z=perf.get((y,rid,t))
            if z and float(z["stake"])>0:
                avail.append((t,z))
        for (ta,za),(tb,zb) in combinations(avail,2):
            if ta>tb:
                ta,tb=tb,ta; za,zb=zb,za
            race_idx.append(i)
            aidx.append(tindex[ta]); bidx.append(tindex[tb])
            atix.append(int(za["tickets"])); btix.append(int(zb["tickets"]))
            labels.append(int(perf_key(za,ta)>perf_key(zb,tb)))
    if not labels:
        raise SystemExit("no pairwise training rows")

    ridx=np.asarray(race_idx,dtype=np.int32)
    base=np.asarray(xtrain,dtype=np.float32)[ridx]
    n=len(labels); nt=len(templates)
    extra=np.zeros((n,2*nt+4),dtype=np.float32)
    rr=np.arange(n)
    extra[rr,np.asarray(aidx,dtype=np.int32)]=1.0
    extra[rr,nt+np.asarray(bidx,dtype=np.int32)]=1.0
    at=np.asarray(atix,dtype=np.float32); bt=np.asarray(btix,dtype=np.float32)
    extra[:,2*nt]=at
    extra[:,2*nt+1]=bt
    extra[:,2*nt+2]=at-bt
    extra[:,2*nt+3]=at/np.maximum(bt,1.0)
    x=np.concatenate([base,extra],axis=1)
    y=np.asarray(labels,dtype=np.int8)
    return x,y


def train_pairwise(train,cols,perf,templates,seed):
    xtrain,_=encode_fit_other(train,[],cols)
    xpair,ypair=build_pair_training(train,np.asarray(xtrain,dtype=np.float32),perf,templates)
    model=lgb.LGBMClassifier(
        objective="binary",n_estimators=240,learning_rate=0.03,num_leaves=31,
        min_child_samples=120,subsample=0.90,colsample_bytree=0.85,
        reg_lambda=4.0,reg_alpha=0.5,random_state=seed,n_jobs=2,
        deterministic=True,force_col_wise=True,verbosity=-1,
    )
    model.fit(xpair,ypair)
    return model,{
        "pair_rows":int(len(ypair)),
        "pair_positive_rate":float(ypair.mean()),
        "pair_feature_count":int(xpair.shape[1]),
    }


def pair_vector(xrace,ta,tb,za,zb,templates):
    if ta>tb:
        ta,tb=tb,ta; za,zb=zb,za
    nt=len(templates); idx={t:i for i,t in enumerate(templates)}
    extra=np.zeros(2*nt+4,dtype=np.float32)
    extra[idx[ta]]=1.0
    extra[nt+idx[tb]]=1.0
    at=float(za["tickets"]); bt=float(zb["tickets"])
    extra[2*nt]=at; extra[2*nt+1]=bt
    extra[2*nt+2]=at-bt
    extra[2*nt+3]=at/max(bt,1.0)
    return np.concatenate([np.asarray(xrace,dtype=np.float32),extra]),ta,tb


def rerank(y,rid,candidates,probs,xrace,model,perf,templates):
    avail=[]
    for t,p in zip(candidates,probs):
        t=str(t)
        z=perf.get((y,rid,t))
        if z and float(z["stake"])>0:
            avail.append((t,float(p),z))
    if not avail:
        for t in templates:
            z=perf.get((y,rid,t))
            if z and float(z["stake"])>0:
                return t
        return str(candidates[0])
    if len(avail)==1:
        return avail[0][0]

    wins={t:0 for t,_,_ in avail}
    psum={t:0.0 for t,_,_ in avail}
    pstage1={t:p for t,p,_ in avail}
    tickets={t:int(z["tickets"]) for t,_,z in avail}

    for (ta,_,za),(tb,_,zb) in combinations(avail,2):
        x,ca,cb=pair_vector(xrace,ta,tb,za,zb,templates)
        pca=float(model.predict_proba(x.reshape(1,-1))[0,1])
        pta=pca if ca==ta else 1.0-pca
        ptb=1.0-pta
        psum[ta]+=pta; psum[tb]+=ptb
        if pta>=0.5: wins[ta]+=1
        else: wins[tb]+=1

    return sorted(
        wins,
        key=lambda t:(-wins[t],-psum[t],-pstage1[t],tickets[t],t)
    )[0]


def route_year(test,labels,probs,xtest,pair_model,perf,templates):
    top1=[]; top2=[]; top3=[]; rows=[]
    for i,r in enumerate(test.to_dict("records")):
        y=int(r["year"]); rid=str(r["race_id"])
        classes=[str(x) for x in labels[i]]
        pp=[float(x) for x in probs[i]]
        p1=classes[0]
        p2=rerank(y,rid,classes[:2],pp[:2],xtest[i],pair_model,perf,templates)
        p3=rerank(y,rid,classes[:3],pp[:3],xtest[i],pair_model,perf,templates)
        top1.append(p1); top2.append(p2); top3.append(p3)
        rows.append({
            "year":y,"race_id":rid,"race_date":r["race_date"],
            "v1_top1":p1,"v1_top1_prob":pp[0],
            "v1_top2":classes[1] if len(classes)>1 else "",
            "v1_top2_prob":pp[1] if len(classes)>1 else None,
            "v1_top3":classes[2] if len(classes)>2 else "",
            "v1_top3_prob":pp[2] if len(classes)>2 else None,
            "top2_pairwise":p2,"top3_pairwise":p3,
        })
    return top1,top2,top3,rows


def main():
    a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L2_NORMAL_PAIRWISE_ROUTER_V4":
        raise SystemExit("wrong contract")
    if contract["scope"]["battlefield"]!="PASS_SEVEN_ONLY":
        raise SystemExit("battlefield drift")
    if contract["scope"]["race_filtering"] is not False:
        raise SystemExit("race filtering forbidden")
    if contract["scope"]["force_route_every_normal_race"] is not True:
        raise SystemExit("forced route guard broken")
    if contract["scope"]["skip_class"] is not False:
        raise SystemExit("skip class forbidden")
    if contract["scope"]["locked_years"]!=[2026]:
        raise SystemExit("2026 lock drift")
    if contract["pairwise"]["allowed_market_features"] is not False:
        raise SystemExit("market leakage contract broken")
    if contract["selection"]["final_holdout_year"]!=2025:
        raise SystemExit("holdout drift")

    paths=parse_paths(a.router_year)
    if set(paths)!=set(YEARS):
        raise SystemExit(f"router years mismatch {sorted(paths)}")

    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={y:load_router(paths[y],y) for y in YEARS}
    features=build_feature_rows(routers,fixed)
    expected={y:len(routers[y])-len(fixed[y]) for y in ANALYSIS_YEARS}
    actual=features.groupby("year")["race_id"].nunique().to_dict()
    if any(actual.get(y)!=expected[y] for y in ANALYSIS_YEARS):
        raise SystemExit(f"normal universe drift actual={actual} expected={expected}")

    _,perf=load_performance(a.dataset_dir)
    data=attach_targets(features,perf)
    after=data.groupby("year")["race_id"].nunique().to_dict()
    if any(after.get(y)!=expected[y] for y in ANALYSIS_YEARS):
        raise SystemExit(f"target attachment dropped races actual={after} expected={expected}")
    cols=feature_columns(data)

    refs={}
    if a.v1_metrics and Path(a.v1_metrics).exists():
        for r in read_csv_rows(a.v1_metrics):
            if r.get("architecture")=="DIRECT_TEMPLATE":
                refs[int(r["test_year"])]=r

    metrics=[]; decisions=[]; folds=[]
    for yi,test_year in enumerate(TEST_YEARS):
        train=data[data["year"]<test_year].copy().reset_index(drop=True)
        test=data[data["year"]==test_year].copy().sort_values(["race_date","race_id"]).reset_index(drop=True)

        labels,probs,xtest=train_v1_shortlist(train,test,cols,81000+test_year)
        pair_model,pmeta=train_pairwise(train,cols,perf,TEMPLATES,121000+yi)
        p1,p2,p3,rows=route_year(test,labels,probs,xtest,pair_model,perf,TEMPLATES)
        decisions.extend(rows)

        m1=evaluate(test,p1,"V1_TOP1",perf)
        m2=evaluate(test,p2,"TOP2_PAIRWISE",perf)
        m3=evaluate(test,p3,"TOP3_PAIRWISE",perf)
        metrics.extend([m1,m2,m3])

        ref=refs.get(test_year)
        if ref:
            drift=abs(float(ref["roi_pct"])-float(m1["roi_pct"]))
            if drift>0.02:
                raise SystemExit(f"V1 reproduction drift year={test_year} ref={ref['roi_pct']} now={m1['roi_pct']}")
        folds.append({
            "test_year":test_year,
            "train_years":"|".join(map(str,sorted(train["year"].unique()))),
            "train_races":int(train["race_id"].nunique()),
            "test_races":int(test["race_id"].nunique()),
            "v1_classes":int(labels.shape[1]),
            "v1_reference_roi_pct":float(ref["roi_pct"]) if ref else None,
            "v1_reproduced_roi_pct":m1["roi_pct"],
            **pmeta,
        })

    candidates=[]
    for arch in ARCHS:
        rows=[r for r in metrics if r["architecture"]==arch and int(r["test_year"]) in DEV_YEARS]
        stake=sum(r["stake_yen"] for r in rows); ret=sum(r["return_yen"] for r in rows)
        candidates.append({
            "architecture":arch,"development_years":"2023|2024",
            "dev_stake_yen":stake,"dev_return_yen":ret,
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

    holdout=next(r for r in metrics if r["architecture"]==chosen and int(r["test_year"])==HOLDOUT_YEAR)
    v1_holdout=next(r for r in metrics if r["architecture"]=="V1_TOP1" and int(r["test_year"])==HOLDOUT_YEAR)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"pairwise-router-metrics.csv",metrics)
    write_csv(out/"development-selection.csv",candidates)
    write_csv(out/"fold-status.csv",folds)
    with gzip.open(out/"pairwise-route-decisions.csv.gz","wt",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(decisions[0]))
        w.writeheader(); w.writerows(decisions)

    summary={
        "contract":"L2_NORMAL_PAIRWISE_ROUTER_RESULT_V4",
        "scope":"PASS_SEVEN_ONLY forced routing; no skip",
        "normal_races":{str(y):expected[y] for y in ANALYSIS_YEARS},
        "stage1":"V1 DIRECT_TEMPLATE shortlist",
        "stage2":"pairwise rerank among V1 Top2/Top3 only",
        "pairwise_market_features":False,
        "architectures":list(ARCHS),
        "development_selection_years":[2023,2024],
        "selected_architecture":chosen,
        "holdout_year":2025,
        "holdout_used_for_selection":False,
        "selected_holdout_2025":holdout,
        "v1_top1_holdout_2025":v1_holdout,
        "race_filtering":False,
        "route_every_normal_race":True,
        "skip_class":False,
        "2026_locked":True,
        "production_promotion":False,
        "interpretation_guard":"Stage1 reproduces V1. Pairwise targets use historical template profit comparisons, but prediction features exclude odds/payout/results. Only V1 Top2/Top3 are reranked. Selection uses 2023-2024 only; 2025 is holdout-only."
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Normal Pairwise Router V4\n\n"
        "V1 remains the first-stage router. V4 only reranks V1 Top2 or Top3 candidates. "
        "The pairwise model sees pre-race Seven-King/race structure, template identities, and deterministic ticket counts. "
        "Odds, popularity, payouts and results are forbidden prediction inputs. "
        "V1_TOP1, TOP2_PAIRWISE and TOP3_PAIRWISE are compared on 2023-2024 only; 2025 is holdout-only. "
        "Every normal race remains in scope, no SKIP class exists, and 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("L2_NORMAL_PAIRWISE_ROUTER_V4_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)


if __name__=="__main__":
    main()
