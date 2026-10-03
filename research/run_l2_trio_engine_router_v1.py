#!/usr/bin/env python3
import argparse,gc,json,math,os,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l2_trio_probability_v2 import (
    YEARS,FOLDS,write_csv,load_horse_dataset,attach_win_market,attach_outsider,prepare_l175
)
from run_l2_trio_market_residual_v3 import (
    build_year_matrix,group_layout,offset_softmax,residual_objective,choose_alpha
)
from run_l2_trio_split_residual_v4 import (
    parse_year_paths,load_outsider_ballots,attach_outsider_ballots,
    build_compact_engine_matrices,load_engine_year,train_one_engine,
    race_ids_for_year,true_ticket_prob_per_race,reindex_groups,params_for_threads
)
from run_l2_trio_v3_robustness_audit import bootstrap_mean_ci

EPS=1e-12
SEED=20261004
BOOT_REPS=3000
TARGET_CLIP=0.50

ROUTER_FEATURES=[
    "field_size","combo_count",
    "market_entropy","king_entropy","outsider_entropy",
    "market_top1","market_top3","market_top10",
    "king_top1","king_top3","king_top10",
    "outsider_top1","outsider_top3","outsider_top10",
    "tv_king_market","tv_outsider_market","tv_king_outsider",
    "max_abs_king_market","max_abs_outsider_market","max_abs_king_outsider",
    "kl_king_market","kl_market_king",
    "kl_outsider_market","kl_market_outsider",
    "kl_king_outsider","kl_outsider_king",
    "same_top1_king_market","same_top1_outsider_market","same_top1_king_outsider",
    "top3_overlap_king_market","top3_overlap_outsider_market","top3_overlap_king_outsider",
]

def parse_args():
    p=argparse.ArgumentParser(description="Strict walk-forward router over MARKET / MARKET+KING / MARKET+OUTSIDER.")
    p.add_argument("--contract",required=True)
    p.add_argument("--l175-contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--outsider-predictions",required=True)
    p.add_argument("--ballots-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--matrix-cache",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def entropy_norm(p):
    p=np.clip(np.asarray(p,dtype=np.float64),EPS,None)
    h=-float(np.sum(p*np.log(p)))
    den=math.log(len(p)) if len(p)>1 else 1.0
    return h/den if den>0 else 0.0

def top_mass(p,k):
    p=np.asarray(p,dtype=np.float64)
    k=min(int(k),len(p))
    if k<=0:return 0.0
    if k==len(p):return float(np.sum(p))
    return float(np.partition(p,len(p)-k)[-k:].sum())

def kl(a,b):
    a=np.clip(np.asarray(a,dtype=np.float64),EPS,None)
    b=np.clip(np.asarray(b,dtype=np.float64),EPS,None)
    return float(np.sum(a*np.log(a/b)))

def top_indices(p,k):
    p=np.asarray(p,dtype=np.float64)
    k=min(k,len(p))
    if k<=0:return set()
    return set(np.argpartition(p,len(p)-k)[-k:].tolist())

def router_feature_frame(qm,pk,po,ri,meta,year):
    starts,counts=group_layout(ri)
    meta=meta.sort_values("race_index").reset_index(drop=True)
    if len(meta)!=len(starts):
        raise SystemExit(f"router meta/group mismatch year={year} meta={len(meta)} groups={len(starts)}")
    rows=[]
    for j,(s,c) in enumerate(zip(starts,counts)):
        e=int(s+c)
        m=np.asarray(qm[s:e],dtype=np.float64)
        k=np.asarray(pk[s:e],dtype=np.float64)
        o=np.asarray(po[s:e],dtype=np.float64)
        mt1=int(np.argmax(m)); kt1=int(np.argmax(k)); ot1=int(np.argmax(o))
        mt3=top_indices(m,3); kt3=top_indices(k,3); ot3=top_indices(o,3)
        r=meta.iloc[j]
        rows.append({
            "year":int(year),"race_id":str(r["race_id"]),"race_date":str(r["race_date"]),
            "field_size":float(r["field_size"]),"combo_count":float(c),
            "market_entropy":entropy_norm(m),"king_entropy":entropy_norm(k),"outsider_entropy":entropy_norm(o),
            "market_top1":top_mass(m,1),"market_top3":top_mass(m,3),"market_top10":top_mass(m,10),
            "king_top1":top_mass(k,1),"king_top3":top_mass(k,3),"king_top10":top_mass(k,10),
            "outsider_top1":top_mass(o,1),"outsider_top3":top_mass(o,3),"outsider_top10":top_mass(o,10),
            "tv_king_market":0.5*float(np.abs(k-m).sum()),
            "tv_outsider_market":0.5*float(np.abs(o-m).sum()),
            "tv_king_outsider":0.5*float(np.abs(k-o).sum()),
            "max_abs_king_market":float(np.max(np.abs(k-m))),
            "max_abs_outsider_market":float(np.max(np.abs(o-m))),
            "max_abs_king_outsider":float(np.max(np.abs(k-o))),
            "kl_king_market":kl(k,m),"kl_market_king":kl(m,k),
            "kl_outsider_market":kl(o,m),"kl_market_outsider":kl(m,o),
            "kl_king_outsider":kl(k,o),"kl_outsider_king":kl(o,k),
            "same_top1_king_market":float(kt1==mt1),
            "same_top1_outsider_market":float(ot1==mt1),
            "same_top1_king_outsider":float(kt1==ot1),
            "top3_overlap_king_market":float(len(kt3 & mt3))/3.0,
            "top3_overlap_outsider_market":float(len(ot3 & mt3))/3.0,
            "top3_overlap_king_outsider":float(len(kt3 & ot3))/3.0,
        })
    return pd.DataFrame(rows)

def delta_vectors(y,ri,qm,pk,po):
    tq=true_ticket_prob_per_race(y,qm,ri)
    tk=true_ticket_prob_per_race(y,pk,ri)
    to=true_ticket_prob_per_race(y,po,ri)
    ml=-np.log(np.clip(tq,EPS,None))
    kloss=-np.log(np.clip(tk,EPS,None))
    oloss=-np.log(np.clip(to,EPS,None))
    return tq,tk,to,kloss-ml,oloss-ml

def initial_oof_engine(engine,loaded,threads):
    meta,X,y,ri,q,names=loaded[2022]
    n=int(meta["races"])
    fit_cut=max(1,int(n*0.60))
    cal_cut=max(fit_cut+1,int(n*0.80))
    ria=np.asarray(ri)
    fit=ria<fit_cut
    cal=(ria>=fit_cut)&(ria<cal_cut)
    eva=ria>=cal_cut

    Xfit=np.asarray(X)[fit].astype(np.float32,copy=False)
    yfit=np.asarray(y)[fit].astype(np.uint8,copy=False)
    qfit=np.asarray(q)[fit].astype(np.float32,copy=False)
    rifit=reindex_groups(ria[fit])
    m0=lgb.LGBMRegressor(**params_for_threads(residual_objective(qfit,rifit),threads))
    m0.fit(Xfit,yfit)

    Xcal=np.asarray(X)[cal].astype(np.float32,copy=False)
    ycal=np.asarray(y)[cal].astype(np.uint8,copy=False)
    qcal=np.asarray(q)[cal].astype(np.float32,copy=False)
    rical=reindex_groups(ria[cal])
    scal=np.asarray(m0.predict(Xcal),dtype=np.float64)
    (alpha,_),_=choose_alpha(scal,ycal,qcal,rical)

    train=ria<cal_cut
    Xtrain=np.asarray(X)[train].astype(np.float32,copy=False)
    ytrain=np.asarray(y)[train].astype(np.uint8,copy=False)
    qtrain=np.asarray(q)[train].astype(np.float32,copy=False)
    ritrain=reindex_groups(ria[train])
    model=lgb.LGBMRegressor(**params_for_threads(residual_objective(qtrain,ritrain),threads))
    model.fit(Xtrain,ytrain)

    Xeval=np.asarray(X)[eva].astype(np.float32,copy=False)
    yeval=np.asarray(y)[eva].astype(np.uint8,copy=False)
    qeval=np.asarray(q)[eva].astype(np.float32,copy=False)
    rieval=reindex_groups(ria[eva])
    score=np.asarray(model.predict(Xeval),dtype=np.float64)
    p=offset_softmax(score,qeval,rieval,alpha)
    out={"engine":engine,"alpha":float(alpha),"p":p,"y":yeval,"q":qeval,"ri":rieval,
         "race_start":cal_cut,"race_count":n-cal_cut}
    del Xfit,yfit,qfit,rifit,m0,Xcal,ycal,qcal,rical,scal,Xtrain,ytrain,qtrain,ritrain,model,Xeval
    gc.collect()
    return out

def router_params(seed,threads):
    return dict(
        objective="regression",n_estimators=180,learning_rate=0.035,
        num_leaves=7,max_depth=4,min_child_samples=70,
        subsample=0.9,subsample_freq=1,colsample_bytree=0.9,
        reg_lambda=6.0,reg_alpha=0.5,random_state=seed,
        n_jobs=max(1,int(threads)),verbosity=-1,force_col_wise=True
    )

def train_router(history,test,threads):
    X=history[ROUTER_FEATURES].to_numpy(dtype=np.float32)
    Xt=test[ROUTER_FEATURES].to_numpy(dtype=np.float32)
    yk=np.clip(history["king_delta"].to_numpy(dtype=np.float64),-TARGET_CLIP,TARGET_CLIP)
    yo=np.clip(history["outsider_delta"].to_numpy(dtype=np.float64),-TARGET_CLIP,TARGET_CLIP)
    with ThreadPoolExecutor(max_workers=2) as ex:
        fk=ex.submit(lambda: lgb.LGBMRegressor(**router_params(SEED+11,threads)).fit(X,yk))
        fo=ex.submit(lambda: lgb.LGBMRegressor(**router_params(SEED+29,threads)).fit(X,yo))
        mk=fk.result(); mo=fo.result()
    pk=np.asarray(mk.predict(Xt),dtype=np.float64)
    po=np.asarray(mo.predict(Xt),dtype=np.float64)
    choices=np.argmin(np.column_stack([np.zeros(len(test)),pk,po]),axis=1)
    imp=[]
    for name,model in (("KING_DELTA",mk),("OUTSIDER_DELTA",mo)):
        gains=model.booster_.feature_importance(importance_type="gain")
        for rank,(fn,g) in enumerate(sorted(zip(ROUTER_FEATURES,gains),key=lambda z:-z[1]),1):
            imp.append({"target":name,"rank":rank,"feature":fn,"gain":float(g)})
    return pk,po,choices,imp

def main():
    t0=time.time(); a=parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    c=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    l175=json.loads(Path(a.l175_contract).read_text(encoding="utf-8"))
    if c.get("contract")!="L2_TRIO_ENGINE_ROUTER_V1": raise SystemExit("wrong router contract")
    if l175.get("contract")!="L175_MARKET_CONTEXT_V1": raise SystemExit("wrong L175 contract")
    if c["cost_policy"]["github_standard_cpu_only"] is not True or c["cost_policy"]["gpu"] is not False:
        raise SystemExit("cost guard drift")
    if c["locked_years"]!=[2026]: raise SystemExit("2026 lock drift")

    cpu=max(1,os.cpu_count() or 1)
    engine_threads=max(1,cpu//2)
    router_threads=max(1,cpu//2)
    print(f"L2_TRIO_ROUTER_CPU cpu={cpu} engine_threads={engine_threads} router_threads={router_threads}",flush=True)

    _,df=load_horse_dataset(a.dataset_dir)
    df,skipped=attach_win_market(df,a.backfill_root)
    df=attach_outsider(df,a.outsider_predictions)
    df=prepare_l175(df)
    ballot_stats,race_candidates=load_outsider_ballots(parse_year_paths(a.ballots_year))
    df=attach_outsider_ballots(df,ballot_stats,race_candidates)

    for y in YEARS:
        build_year_matrix(y,df,a.backfill_root,a.matrix_cache)
        build_compact_engine_matrices(y,df,a.matrix_cache)

    loaded={
        "KING":{y:load_engine_year(a.matrix_cache,y,"KING") for y in YEARS},
        "OUTSIDER":{y:load_engine_year(a.matrix_cache,y,"OUTSIDER") for y in YEARS},
    }

    print("L2_TRIO_ROUTER_INITIAL_OOF_START year=2022 split=60/20/20",flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        fk=ex.submit(initial_oof_engine,"KING",loaded["KING"],engine_threads)
        fo=ex.submit(initial_oof_engine,"OUTSIDER",loaded["OUTSIDER"],engine_threads)
        ik=fk.result(); io=fo.result()
    if not np.array_equal(ik["y"],io["y"]) or not np.array_equal(ik["ri"],io["ri"]):
        raise SystemExit("initial OOF alignment drift")
    meta22=race_ids_for_year(a.matrix_cache,2022).iloc[ik["race_start"]:].reset_index(drop=True)
    f22=router_feature_frame(ik["q"],ik["p"],io["p"],ik["ri"],meta22,2022)
    _,_,_,kd22,od22=delta_vectors(ik["y"],ik["ri"],ik["q"],ik["p"],io["p"])
    f22["king_delta"]=kd22; f22["outsider_delta"]=od22
    history=f22.copy()
    print(f"L2_TRIO_ROUTER_INITIAL_OOF_READY races={len(history)} king_mean={kd22.mean():.6f} outsider_mean={od22.mean():.6f}",flush=True)

    fold_rows=[]; route_rows=[]; imp_rows=[]; boot_rows=[]; runtime_rows=[]
    pooled_sel=[]; pooled_k=[]; pooled_o=[]; pooled_sel_minus_o=[]; pooled_oracle=[]

    for fi,(test_year,train_years) in enumerate(FOLDS,1):
        ft=time.time()
        with ThreadPoolExecutor(max_workers=2) as ex:
            fk=ex.submit(train_one_engine,"KING",loaded["KING"],train_years,test_year,engine_threads)
            fo=ex.submit(train_one_engine,"OUTSIDER",loaded["OUTSIDER"],train_years,test_year,engine_threads)
            rk=fk.result(); ro=fo.result()

        _,_,ytest,ritest,qtest,_=loaded["KING"][test_year]
        meta=race_ids_for_year(a.matrix_cache,test_year)
        feat=router_feature_frame(qtest,rk["p"],ro["p"],ritest,meta,test_year)
        tq,tk,to,kd,od=delta_vectors(ytest,ritest,qtest,rk["p"],ro["p"])

        predk,predo,choice,imps=train_router(history,feat,router_threads)
        selected=np.where(choice==1,kd,np.where(choice==2,od,0.0))
        oracle=np.minimum.reduce([np.zeros(len(kd)),kd,od])
        versus_out=selected-od

        counts={"MARKET":int(np.sum(choice==0)),"KING":int(np.sum(choice==1)),"OUTSIDER":int(np.sum(choice==2))}
        fold_rows.append({
            "test_year":test_year,"train_years":"|".join(map(str,train_years)),
            "router_train_races":len(history),
            "market_races":counts["MARKET"],"king_races":counts["KING"],"outsider_races":counts["OUTSIDER"],
            "market_share_pct":100.0*counts["MARKET"]/len(choice),
            "king_share_pct":100.0*counts["KING"]/len(choice),
            "outsider_share_pct":100.0*counts["OUTSIDER"]/len(choice),
            "market_mean_delta":0.0,
            "fixed_king_mean_delta":float(kd.mean()),
            "fixed_outsider_mean_delta":float(od.mean()),
            "router_mean_delta":float(selected.mean()),
            "router_minus_outsider_mean_delta":float(versus_out.mean()),
            "oracle_mean_delta":float(oracle.mean()),
        })
        b=bootstrap_mean_ci(selected,reps=BOOT_REPS,seed=SEED+test_year)
        boot_rows.append({"scope":str(test_year),"comparison":"ROUTER_VS_MARKET",**b})
        bo=bootstrap_mean_ci(versus_out,reps=BOOT_REPS,seed=SEED+1000+test_year)
        boot_rows.append({"scope":str(test_year),"comparison":"ROUTER_MINUS_OUTSIDER",**bo})

        for r in imps:
            imp_rows.append({"test_year":test_year,**r})
        for i,row in feat.iterrows():
            route_rows.append({
                "year":test_year,"race_id":str(row["race_id"]),"race_date":str(row["race_date"]),
                "pred_king_delta":float(predk[i]),"pred_outsider_delta":float(predo[i]),
                "route":("MARKET","KING","OUTSIDER")[int(choice[i])],
                "actual_king_delta":float(kd[i]),"actual_outsider_delta":float(od[i]),
                "actual_router_delta":float(selected[i]),
                "oracle_route":("MARKET" if oracle[i]==0 else ("KING" if kd[i]<=od[i] else "OUTSIDER")),
            })

        pooled_sel.append(selected); pooled_k.append(kd); pooled_o.append(od)
        pooled_sel_minus_o.append(versus_out); pooled_oracle.append(oracle)

        feat["king_delta"]=kd; feat["outsider_delta"]=od
        history=pd.concat([history,feat],ignore_index=True)
        runtime_rows.append({"test_year":test_year,"seconds":time.time()-ft,"router_train_races_before_append":len(history)-len(feat)})
        print("L2_TRIO_ROUTER_FOLD_DONE "+json.dumps({
            "test_year":test_year,"history":len(history)-len(feat),"routes":counts,
            "king":float(kd.mean()),"outsider":float(od.mean()),
            "router":float(selected.mean()),"router_minus_outsider":float(versus_out.mean())
        },separators=(",",":")),flush=True)

    ps=np.concatenate(pooled_sel); pk=np.concatenate(pooled_k); po=np.concatenate(pooled_o)
    pso=np.concatenate(pooled_sel_minus_o); por=np.concatenate(pooled_oracle)
    pooled_router=bootstrap_mean_ci(ps,reps=BOOT_REPS,seed=SEED+5000)
    pooled_vs_out=bootstrap_mean_ci(pso,reps=BOOT_REPS,seed=SEED+6000)
    boot_rows.append({"scope":"POOLED_2023_2025","comparison":"ROUTER_VS_MARKET",**pooled_router})
    boot_rows.append({"scope":"POOLED_2023_2025","comparison":"ROUTER_MINUS_OUTSIDER",**pooled_vs_out})

    write_csv(out/"fold-metrics.csv",fold_rows)
    write_csv(out/"bootstrap.csv",boot_rows)
    write_csv(out/"router-feature-importance.csv",imp_rows)
    write_csv(out/"runtime.csv",runtime_rows)
    pd.DataFrame(route_rows).to_csv(out/"race-routes.csv.gz",index=False,compression="gzip")

    summary={
        "contract":"L2_TRIO_ENGINE_ROUTER_V1_RESULT",
        "architecture":"MARKET_KING_OUTSIDER_THREE_WAY_ROUTER",
        "router_features":ROUTER_FEATURES,
        "initial_2022_oof":{"policy":"chronological_60_fit_20_alpha_20_router_label","races":len(f22)},
        "fold_metrics":fold_rows,
        "pooled":{
            "market_mean_delta":0.0,
            "fixed_king_mean_delta":float(pk.mean()),
            "fixed_outsider_mean_delta":float(po.mean()),
            "router_mean_delta":float(ps.mean()),
            "router_minus_outsider_mean_delta":float(pso.mean()),
            "oracle_mean_delta":float(por.mean()),
            "router_vs_market_bootstrap":pooled_router,
            "router_minus_outsider_bootstrap":pooled_vs_out,
        },
        "success_gate":{
            "beat_fixed_outsider_mean":bool(float(ps.mean())<float(po.mean())),
            "beat_fixed_outsider_ci95":bool(float(pooled_vs_out["ci_high"])<0.0),
            "reference_outsider_target":-0.00614873642986566,
        },
        "market_price_used":True,
        "payout_used":False,"roi_used":False,"staking_used":False,
        "2026_locked":True,"promotion":False,
        "elapsed_seconds":time.time()-t0
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 TRIO Engine Router V1\n\n"
        "Strict walk-forward three-way router over MARKET / MARKET+SEVEN_KING / MARKET+OUTSIDER. "
        "Router targets are prior out-of-sample race logloss deltas. 2022 uses chronological 60/20/20 warm-start OOF; "
        "2023-2025 test folds are never used before their turn. No ROI, payout or staking optimization. 2026 sealed.\n",
        encoding="utf-8"
    )
    write_csv(out/"skipped-win-market-races.csv",skipped)
    print("===== FOLD METRICS ====="); print((out/"fold-metrics.csv").read_text())
    print("===== BOOTSTRAP ====="); print((out/"bootstrap.csv").read_text())
    print("===== SUMMARY ====="); print((out/"summary.json").read_text())
    print("L2_TRIO_ENGINE_ROUTER_V1_READY")

if __name__=="__main__":
    main()
