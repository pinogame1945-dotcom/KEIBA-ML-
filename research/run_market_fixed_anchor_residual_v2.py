#!/usr/bin/env python3
import argparse,concurrent.futures,csv,gc,gzip,json,math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l1_objective_rebuild_v1 import YEARS,TEST_YEARS,LOCKED_YEAR,read_year,select_variant
from run_l1_ceiling_audit_v1 import strip_forbidden
from run_l1_state_transition_v1 import build_state_features,attach_state
from run_l1_residual_regime_full_pairwise_v1 import build_regime_features,fit_direct_rank,top1_metrics
from run_l1_sameday_track_bias_v1 import build_same_day_bias,bias_cols,speed_cols
from run_l1_negative_evidence_penalty_v1 import read_market_year
from run_l1_market_correction_distill_v1 import market_rank_frame,add_ai_context
from run_l1_market_correction_deepdive_v1 import fit_risk_full

CAL_YEARS=(2023,2024,2025)
STRICT_BLEND_YEARS=(2024,2025)
ALPHAS=(0.0,0.25,0.50,0.75,1.0)
EXPECTED_ABILITY={
    2021:59.34606481481482,
    2022:60.90856481481482,
    2023:59.432870370370374,
    2024:62.673611111111114,
    2025:61.660879629629626,
}
META_COLS=("distill_ai_rank01","distill_score_z","distill_score_gap_to_best","distill_field_size","risk")
META_CONSTRAINTS=(1,1,-1,0,-1)

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def logloss(y,p):
    y=np.asarray(y,dtype=float); p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def brier(y,p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    return float(np.mean((y-p)**2))

def auc_rank(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    pos=int(y.sum()); neg=len(y)-pos
    if pos==0 or neg==0: return np.nan
    r=pd.Series(p).rank(method="average").to_numpy()
    return float((r[y==1].sum()-pos*(pos+1)/2)/(pos*neg))

def meta_params(seed):
    return dict(
        n_estimators=260,learning_rate=0.035,num_leaves=23,min_child_samples=160,
        subsample=0.9,colsample_bytree=0.9,reg_alpha=1.0,reg_lambda=10.0,
        random_state=seed,n_jobs=-1,verbosity=-1,
    )

def fit_mono_top3(train,valid,seed):
    xtr=train[list(META_COLS)].apply(pd.to_numeric,errors="coerce").astype("float32")
    xva=valid[list(META_COLS)].apply(pd.to_numeric,errors="coerce").astype("float32")
    y=train["_is_top3"].astype(int).to_numpy()
    m=lgb.LGBMClassifier(
        objective="binary",
        monotone_constraints=list(META_CONSTRAINTS),
        monotone_constraints_method="advanced",
        **meta_params(seed),
    )
    m.fit(xtr,y)
    p=np.asarray(m.predict_proba(xva)[:,1],dtype=float)
    gain=m.booster_.feature_importance(importance_type="gain")
    imp=sorted(zip(META_COLS,gain),key=lambda z:z[1],reverse=True)
    del m,xtr,xva; gc.collect()
    return p,imp

def rank_bin(rank):
    r=int(rank)
    if r<=3: return str(r)
    if r<=6: return "4_6"
    if r<=10: return "7_10"
    return "11P"

def fit_grid(train,valid,bins=5,smooth=80.0):
    tr=train.copy(); va=valid.copy()
    tr["_rb"]=tr["distill_ai_rank"].map(rank_bin)
    va["_rb"]=va["distill_ai_rank"].map(rank_bin)
    out=np.zeros(len(va),dtype=float)
    global_rate=float(tr["_is_top3"].mean())
    for rb in sorted(set(tr["_rb"]).union(set(va["_rb"]))):
        t=tr[tr["_rb"]==rb].copy()
        v=va[va["_rb"]==rb].copy()
        if v.empty: continue
        prior=float(t["_is_top3"].mean()) if len(t) else global_rate
        if len(t)<bins*20:
            out[v.index]=prior
            continue
        # Training-only quantile cutpoints. Duplicates collapse safely.
        qs=np.linspace(0,1,bins+1)[1:-1]
        edges=sorted(set(float(t["risk"].quantile(q)) for q in qs))
        def bucket(s):
            return np.searchsorted(edges,np.asarray(s,dtype=float),side="right")
        t["_qb"]=bucket(t["risk"])
        vb=bucket(v["risk"])
        stats=t.groupby("_qb")["_is_top3"].agg(["sum","count"]).to_dict("index")
        pred=[]
        for b in vb:
            st=stats.get(int(b),{"sum":0.0,"count":0.0})
            pred.append((float(st["sum"])+smooth*prior)/(float(st["count"])+smooth))
        out[v.index]=pred
    return out

def metrics_from_score(df,score_col,variant,test_year):
    x=df.copy()
    x["__score"]=pd.to_numeric(x[score_col],errors="coerce").fillna(-1e30)
    x=x.sort_values(["_race_id","__score","_horse_id"],ascending=[True,False,True])
    x["__rank"]=x.groupby("_race_id",sort=False).cumcount()+1
    top=x[x["__rank"]==1]
    winners=x[x["_finish"]==1]
    return {
        "test_year":test_year,"variant":variant,
        "races":int(x["_race_id"].nunique()),
        "top1_top3_pct":100*float(top["_is_top3"].mean()),
        "top1_win_pct":100*float(top["_is_win"].mean()),
        "winner_top3_capture_pct":100*float((winners["__rank"]<=3).mean()),
        "winner_top6_capture_pct":100*float((winners["__rank"]<=6).mean()),
    },x[["_race_id","_horse_id","distill_ai_rank","__rank"]]

def calibration_metrics(df,pcol,variant,test_year):
    y=df["_is_top3"].astype(int).to_numpy()
    p=df[pcol].to_numpy(dtype=float)
    return {
        "test_year":test_year,"variant":variant,"n":len(df),
        "top3_prevalence_pct":100*float(y.mean()),
        "auc":auc_rank(y,p),"logloss":logloss(y,p),"brier":brier(y,p),
    }

def rank_movement(base_rank,new_rank):
    z=base_rank.merge(new_rank,on=["_race_id","_horse_id"],suffixes=("_base","_new"))
    d=(z["__rank_new"]-z["distill_ai_rank_base"]).abs()
    return {
        "mean_abs_rank_move":float(d.mean()),
        "pct_unchanged":100*float((d==0).mean()),
        "pct_move_3plus":100*float((d>=3).mean()),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--year-file",action="append",required=True,help="YEAR:PATH")
    ap.add_argument("--out-dir",required=True)
    ap.add_argument("--cache-only",action="store_true")
    a=ap.parse_args()
    paths={}
    for spec in a.year_file:
        y,p=spec.split(":",1); paths[int(y)]=p
    if set(paths)!=set(YEARS): raise SystemExit(f"year file mismatch {sorted(paths)}")
    if LOCKED_YEAR in paths: raise SystemExit("2026 sealed")

    print(json.dumps({
        "contract":"L1_CONFIDENCE_CALIBRATED_RANKING_V1_RUNTIME",
        "architecture":"Ability -> strictly-OOS market-correction risk -> top3 probability calibration -> final L1 ranking",
        "market_role":"FINAL market is historical teacher for the risk head only; never an inference input",
        "ranking_rule":"No hard VETO. Confidence modifies calibrated probability; final rank follows calibrated score.",
        "meta_constraints":{
            "ability_rank01":"+","ability_score_z":"+","gap_to_best":"-","risk":"-","field_size":"free"
        },
        "secondary_layer":"year Y calibrator trains only on risk predictions/outcomes from years < Y",
        "2026_locked":True,
    },separators=(",",":")),flush=True)

    bias_by_year,_=build_same_day_bias(paths)
    state_by_year,_,_,_,state_cov=build_state_features(paths)
    regime_by_year,regime_cols=build_regime_features(state_by_year)
    spd_cols=speed_cols(regime_cols)
    fit_cols=bias_cols(bias_by_year[2021],with_fit=True)
    markets={y:market_rank_frame(read_market_year(paths[y],y)) for y in TEST_YEARS}

    base_cache={}
    def get_base(year):
        if year not in base_cache:
            raw=strip_forbidden(read_year(paths[year],year))
            base_cache[year]=select_variant(raw,"core4")
            del raw; gc.collect()
        return base_cache[year]

    # Exact OOS ability frames first. Preload once, then run two folds concurrently.
    oos={}
    ability_checks=[]
    for _y in YEARS:
        get_base(_y)

    def ability_fold(test):
        tr_years=(test-2,test-1)
        train_base=pd.concat([get_base(y) for y in tr_years],ignore_index=True,copy=False)
        valid_base=get_base(test).copy()
        btr=pd.concat([bias_by_year[y] for y in tr_years],ignore_index=True,copy=False)
        bva=bias_by_year[test]
        rtr=pd.concat([regime_by_year[y] for y in tr_years],ignore_index=True,copy=False)
        rva=regime_by_year[test]
        mtr=btr.merge(rtr[["_race_id","_horse_id",*spd_cols]],on=["_race_id","_horse_id"],how="left",validate="one_to_one")
        mva=bva.merge(rva[["_race_id","_horse_id",*spd_cols]],on=["_race_id","_horse_id"],how="left",validate="one_to_one")
        tr=attach_state(train_base,mtr,fit_cols+spd_cols)
        va=attach_state(valid_base,mva,fit_cols+spd_cols)
        pred,_=fit_direct_rank(tr,va,5100000+test*100)
        got=top1_metrics(pred,"_score")["top1_top3_pct"]
        if abs(got-EXPECTED_ABILITY[test])>1e-9:
            raise RuntimeError(f"ABILITY reproduction drift year={test} expected={EXPECTED_ABILITY[test]} got={got}")
        ctx=add_ai_context(pred)
        ctx=ctx.merge(markets[test],on=["_race_id","_horse_id"],how="left",validate="one_to_one")
        if ctx["market_rank"].notna().mean()<0.99:
            raise RuntimeError(f"market coverage too low year={test}")
        ctx["market_rank"]=pd.to_numeric(ctx["market_rank"],errors="coerce")
        ctx["_target_gap"]=ctx["market_rank"]-ctx["distill_ai_rank"]
        ctx["_target_gap_clip"]=ctx["_target_gap"].clip(-6,6)
        ctx["_target_up"]=(ctx["_target_gap"]>0).astype(int)
        ctx["_target_over3"]=(ctx["_target_gap"]>=3).astype(int)
        feat=va.copy().merge(
            ctx[["_race_id","_horse_id","_score","distill_ai_rank","distill_field_size","distill_ai_rank01",
                 "distill_score_z","distill_score_gap_to_best","distill_is_ai_top1","distill_is_ai_top3",
                 "distill_is_ai_top6","market_rank","final_win_odds","_target_gap","_target_gap_clip",
                 "_target_up","_target_over3"]],
            on=["_race_id","_horse_id"],how="left",validate="one_to_one",
        )
        chk={"test_year":test,"races":int(ctx["_race_id"].nunique()),"top1_top3_pct":got}
        del train_base,valid_base,btr,bva,rtr,rva,mtr,mva,tr,va,pred,ctx
        gc.collect()
        return test,feat,chk

    print("ABILITY_PARALLEL_START workers=2 folds=5",flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs=[ex.submit(ability_fold,test) for test in TEST_YEARS]
        for fut in concurrent.futures.as_completed(futs):
            test,feat,chk=fut.result()
            oos[test]=feat
            ability_checks.append(chk)
            print(f"ABILITY_OOS_READY year={test} rows={len(feat)} top1_top3={chk['top1_top3_pct']:.6f}",flush=True)
    ability_checks.sort(key=lambda r:r["test_year"])

    # Strict market-correction risk prediction for 2022-2025, two folds concurrently.
    risk_oos={}
    risk_rows=[]
    risk_imp=[]

    def risk_fold(test):
        teacher_years=tuple(y for y in TEST_YEARS if y<test)
        tr=pd.concat([oos[y] for y in teacher_years],ignore_index=True,copy=False)
        va=oos[test].copy()
        cols=[c for c in va.columns if not c.startswith("neg_")]
        risk,imp=fit_risk_full(tr[cols],va[cols],8500000+test*100)
        va=va.copy(); va["risk"]=risk
        scope=va[va["distill_ai_rank"]<=6]
        row={
            "test_year":test,"teacher_years":"|".join(map(str,teacher_years)),
            "top6_risk_auc":auc_rank(scope["_target_over3"],scope["risk"]),
            "top6_risk_logloss":logloss(scope["_target_over3"],scope["risk"]),
        }
        total=sum(float(g) for _,g in imp) or 1.0
        irows=[{"test_year":test,"rank":rank,"feature":f,"gain":float(g),"share":float(g)/total}
               for rank,(f,g) in enumerate(imp[:80],1)]
        del tr,scope
        gc.collect()
        return test,va,row,irows

    print("RISK_PARALLEL_START workers=2 folds=4",flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs=[ex.submit(risk_fold,test) for test in (2022,2023,2024,2025)]
        for fut in concurrent.futures.as_completed(futs):
            test,va,row,irows=fut.result()
            risk_oos[test]=va
            risk_rows.append(row)
            risk_imp.extend(irows)
            print(f"RISK_OOS_READY year={test} auc={row['top6_risk_auc']:.4f}",flush=True)
    risk_rows.sort(key=lambda r:r["test_year"])

    # V2: Market-fixed anchor residual model.
    # p_i is race-normalized: p_i ∝ q_market_i * exp(delta_i).
    # Market coefficient is fixed at 1. AI features learn only the residual delta.
    from scipy.optimize import minimize

    PRIMARY_YEARS=(2023,2024,2025)
    TRAINABLE_YEARS=(2022,2023,2024,2025)
    EDGE_THRESHOLDS=(0.0,0.05,0.10,0.20,0.30,0.50,0.75,1.00)
    EDGE_BANDS=((0.0,0.05),(0.05,0.10),(0.10,0.20),(0.20,0.30),(0.30,0.50),(0.50,1.00),(1.00,np.inf))
    ZONES=("ALL","AI_TOP3","DISAGREE","AI_UNIQUE_TOP3","OVERLAP2_AI_UNIQUE")
    LAMBDAS=(0.001,0.003,0.01,0.03,0.10,0.30,1.0)
    VARIANTS={
        "ANCHOR_ABILITY":[
            "field_size","distill_ai_rank01","distill_score_z","distill_score_gap_to_best",
            "distill_is_ai_top1","distill_is_ai_top3","distill_is_ai_top6",
        ],
        "ANCHOR_ABILITY_RISK":[
            "field_size","distill_ai_rank01","distill_score_z","distill_score_gap_to_best",
            "distill_is_ai_top1","distill_is_ai_top3","distill_is_ai_top6","risk",
        ],
        "ANCHOR_FULL":[
            "field_size","distill_ai_rank01","distill_score_z","distill_score_gap_to_best",
            "distill_is_ai_top1","distill_is_ai_top3","distill_is_ai_top6","risk",
            "rank_gap_norm","abs_rank_gap_norm","top3_overlap_norm","ai_unique_top3",
            "market_unique_top3","top1_match","risk_x_abs_gap","ability_x_market_gap",
            "ability_x_disagree_level","risk_x_disagree_level","rankgap_x_disagree_level",
        ],
    }

    def prepare_fair_frame(year):
        z=risk_oos[year].copy()
        z["year"]=year
        z["final_win_odds"]=pd.to_numeric(z["final_win_odds"],errors="coerce")
        z["market_rank"]=pd.to_numeric(z["market_rank"],errors="coerce")
        z["distill_field_size"]=pd.to_numeric(z["distill_field_size"],errors="coerce")
        z=z[(z["final_win_odds"]>0)&z["market_rank"].notna()&z["distill_ai_rank"].notna()].copy()
        z["field_size"]=z["distill_field_size"].clip(lower=2)
        z["market_inv"]=1.0/z["final_win_odds"]
        invsum=z.groupby("_race_id",sort=False)["market_inv"].transform("sum")
        z["market_prob_norm"]=z["market_inv"]/invsum
        den=(z["field_size"]-1.0).clip(lower=1.0)
        z["market_rank01"]=(z["market_rank"]-1.0)/den
        z["rank_gap"]=z["market_rank"]-z["distill_ai_rank"]
        z["rank_gap_norm"]=z["rank_gap"]/z["field_size"]
        z["abs_rank_gap_norm"]=z["rank_gap"].abs()/z["field_size"]
        z["is_market_top3"]=(z["market_rank"]<=3).astype(int)
        z["is_ai_top3"]=(z["distill_ai_rank"]<=3).astype(int)
        z["both_top3"]=((z["is_market_top3"]==1)&(z["is_ai_top3"]==1)).astype(int)
        z["top3_overlap"]=z.groupby("_race_id",sort=False)["both_top3"].transform("sum")
        z["top3_overlap_norm"]=z["top3_overlap"]/3.0
        z["ai_unique_top3"]=((z["is_ai_top3"]==1)&(z["is_market_top3"]==0)).astype(int)
        z["market_unique_top3"]=((z["is_ai_top3"]==0)&(z["is_market_top3"]==1)).astype(int)
        ai1=z.loc[z["distill_ai_rank"]==1,["_race_id","_horse_id"]].rename(columns={"_horse_id":"ai1"})
        mk1=z.loc[z["market_rank"]==1,["_race_id","_horse_id"]].rename(columns={"_horse_id":"mk1"})
        tm=ai1.merge(mk1,on="_race_id",how="inner")
        match=dict(zip(tm["_race_id"].astype(str),(tm["ai1"].astype(str)==tm["mk1"].astype(str)).astype(int)))
        z["top1_match"]=z["_race_id"].astype(str).map(match).fillna(0).astype(int)
        z["risk_x_abs_gap"]=z["risk"]*z["abs_rank_gap_norm"]
        z["ability_x_market_gap"]=z["distill_score_z"]*z["rank_gap_norm"]
        z["disagree_level"]=(3.0-z["top3_overlap"].clip(0,3))/3.0
        z["ability_x_disagree_level"]=z["distill_score_z"]*z["disagree_level"]
        z["risk_x_disagree_level"]=z["risk"]*z["disagree_level"]
        z["rankgap_x_disagree_level"]=z["rank_gap_norm"]*z["disagree_level"]
        z["target_win"]=z["_is_win"].astype(int)
        z["target_top3"]=z["_is_top3"].astype(int)
        return z

    fair_by_year={y:prepare_fair_frame(y) for y in TRAINABLE_YEARS}
    for y,z in fair_by_year.items():
        print("FAIR_BASE_READY "+json.dumps({
            "year":y,"races":int(z["_race_id"].nunique()),"horses":len(z),
            "win_rate":float(z["target_win"].mean()),
        },separators=(",",":")),flush=True)

    cache_cols=[
        "year","_race_id","_horse_id","_race_date","target_win","target_top3","final_win_odds",
        "market_prob_norm","market_rank","field_size","distill_ai_rank","distill_ai_rank01",
        "distill_score_z","distill_score_gap_to_best","distill_is_ai_top1","distill_is_ai_top3",
        "distill_is_ai_top6","risk","rank_gap_norm","abs_rank_gap_norm","top3_overlap",
        "top3_overlap_norm","ai_unique_top3","market_unique_top3","top1_match",
        "risk_x_abs_gap","ability_x_market_gap","disagree_level",
        "ability_x_disagree_level","risk_x_disagree_level","rankgap_x_disagree_level",
    ]

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    cache=pd.concat([fair_by_year[y][cache_cols] for y in TRAINABLE_YEARS],ignore_index=True)
    cache_path=out/"fair-base-cache.csv.gz"
    cache.to_csv(cache_path,index=False,compression="gzip")
    if cache_path.stat().st_size>95*1024*1024:
        raise SystemExit(f"compact cache unexpectedly too large: {cache_path.stat().st_size}")
    print(f"FAIR_BASE_CACHE_READY rows={len(cache)} bytes={cache_path.stat().st_size}",flush=True)
    if a.cache_only:
        (out/"cache-summary.json").write_text(json.dumps({
            "contract":"MARKET_FIXED_ANCHOR_RESIDUAL_V2_CACHE",
            "rows":int(len(cache)),
            "years":sorted(int(x) for x in cache["year"].unique()),
            "bytes":int(cache_path.stat().st_size),
            "2026_locked":True
        },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print("MARKET_FIXED_ANCHOR_RESIDUAL_V2_CACHE_ONLY_COMPLETE",flush=True)
        return

    def chronological_split(train):
        races=train[["_race_date","_race_id"]].drop_duplicates().sort_values(["_race_date","_race_id"]).reset_index(drop=True)
        cut=max(1,min(len(races)-1,int(len(races)*0.80)))
        fit_ids=set(races.iloc[:cut]["_race_id"].astype(str))
        val_ids=set(races.iloc[cut:]["_race_id"].astype(str))
        return (
            train[train["_race_id"].astype(str).isin(fit_ids)].copy(),
            train[train["_race_id"].astype(str).isin(val_ids)].copy(),
        )

    def fit_transform_stats(df,features):
        x=df[features].apply(pd.to_numeric,errors="coerce")
        med=x.median(axis=0).fillna(0.0)
        x=x.fillna(med)
        mean=x.mean(axis=0)
        std=x.std(axis=0,ddof=0).replace(0,np.nan).fillna(1.0)
        return med,mean,std

    def transform_x(df,features,stats):
        med,mean,std=stats
        x=df[features].apply(pd.to_numeric,errors="coerce").fillna(med)
        return ((x-mean)/std).to_numpy(dtype=np.float64)

    def anchored_objective_arrays(df,features,stats):
        q=df.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
        x=transform_x(q,features,stats)
        y=q["target_win"].to_numpy(dtype=np.float64)
        base=np.log(q["market_prob_norm"].clip(1e-12,1.0).to_numpy(dtype=np.float64))
        rid=q["_race_id"].astype(str).to_numpy()
        starts=np.r_[0,np.flatnonzero(rid[1:]!=rid[:-1])+1]
        sizes=np.diff(np.r_[starts,len(q)])
        nr=len(starts)
        return q,x,y,base,starts,sizes,nr

    def fit_beta(df,features,lam,stats=None):
        if stats is None:
            stats=fit_transform_stats(df,features)
        q,x,y,base,starts,sizes,nr=anchored_objective_arrays(df,features,stats)
        def fg(beta):
            z=base+x.dot(beta)
            mx=np.maximum.reduceat(z,starts)
            repmx=np.repeat(mx,sizes)
            ez=np.exp(z-repmx)
            sums=np.add.reduceat(ez,starts)
            p=ez/np.repeat(sums,sizes)
            loss=(-float(np.dot(y,z))+float(np.sum(mx+np.log(sums))))/nr + 0.5*lam*float(np.dot(beta,beta))
            grad=x.T.dot(p-y)/nr + lam*beta
            return loss,grad
        res=minimize(lambda b:fg(b),np.zeros(x.shape[1],dtype=np.float64),jac=True,method="L-BFGS-B",
                     options={"maxiter":180,"ftol":1e-10,"gtol":1e-7})
        if not res.success and res.nit>=180:
            print(f"ANCHOR_OPT_WARN lambda={lam} nit={res.nit} msg={res.message}",flush=True)
        return np.asarray(res.x,dtype=float),stats,float(res.fun)

    def predict_anchor(df,features,beta,stats):
        q=df.reset_index(drop=True).copy()
        q["__pos"]=np.arange(len(q),dtype=int)
        q=q.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
        x=transform_x(q,features,stats)
        z=np.log(q["market_prob_norm"].clip(1e-12,1.0).to_numpy(dtype=float))+x.dot(beta)
        rid=q["_race_id"].astype(str).to_numpy()
        starts=np.r_[0,np.flatnonzero(rid[1:]!=rid[:-1])+1]
        sizes=np.diff(np.r_[starts,len(q)])
        mx=np.maximum.reduceat(z,starts)
        ez=np.exp(z-np.repeat(mx,sizes))
        sums=np.add.reduceat(ez,starts)
        p=ez/np.repeat(sums,sizes)
        out=np.empty(len(df),dtype=float)
        out[q["__pos"].to_numpy(dtype=int)]=p
        return out

    def race_nll(df,p):
        z=df[["_race_id","target_win"]].copy()
        z["p"]=np.clip(np.asarray(p,dtype=float),1e-12,1.0)
        winners=z[z["target_win"].eq(1)]
        return float(-np.log(winners["p"]).mean())

    def choose_lambda(train,features):
        fit,val=chronological_split(train)
        stats=fit_transform_stats(fit,features)
        def one(lam):
            beta,_,obj=fit_beta(fit,features,lam,stats)
            pv=predict_anchor(val,features,beta,stats)
            return lam,race_nll(val,pv),obj
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            rows=list(ex.map(one,LAMBDAS))
        rows.sort(key=lambda x:(x[1],x[0]))
        return rows[0][0],rows

    def binary_metrics(y,p):
        return {"logloss":logloss(y,p),"brier":brier(y,p),"auc":auc_rank(y,p)}

    def rank_metrics(z,pcol):
        q=z.sort_values(["_race_id",pcol,"_horse_id"],ascending=[True,False,True]).copy()
        q["__r"]=q.groupby("_race_id",sort=False).cumcount()+1
        top=q[q["__r"]==1]
        return {
            "races":int(q["_race_id"].nunique()),
            "top1_win_pct":100*float(top["target_win"].mean()),
            "top1_top3_pct":100*float(top["target_top3"].mean()),
        }

    def zone_mask(z,zone):
        if zone=="ALL": return pd.Series(True,index=z.index)
        if zone=="AI_TOP3": return pd.to_numeric(z["distill_ai_rank"],errors="coerce").le(3)
        if zone=="DISAGREE": return z["distill_ai_rank"].ne(z["market_rank"])
        if zone=="AI_UNIQUE_TOP3": return z["ai_unique_top3"].eq(1)
        if zone=="OVERLAP2_AI_UNIQUE": return z["ai_unique_top3"].eq(1)&z["top3_overlap"].eq(2)
        raise ValueError(zone)

    def add_value_cols(z,pcol):
        q=z.copy()
        q["_fair_odds"]=1.0/q[pcol].clip(lower=1e-12)
        q["_edge"]=q[pcol]*q["final_win_odds"]-1.0
        return q

    def roi_summary(q):
        if q.empty:
            return {"bets":0,"races":0,"win_rate_pct":np.nan,"avg_final_odds":np.nan,
                    "avg_fair_odds":np.nan,"roi_pct":np.nan,"profit_yen":0.0}
        stake=100.0*len(q)
        ret=float((q["target_win"]*q["final_win_odds"]*100.0).sum())
        return {
            "bets":int(len(q)),"races":int(q["_race_id"].nunique()),
            "win_rate_pct":100*float(q["target_win"].mean()),
            "avg_final_odds":float(q["final_win_odds"].mean()),
            "avg_fair_odds":float(q["_fair_odds"].mean()),
            "roi_pct":100*ret/stake,"profit_yen":ret-stake,
        }

    def roi_rows(z,pcol,variant,test_year):
        q=add_value_cols(z,pcol)
        rows=[]
        bands=[]
        for zone in ZONES:
            base=q[zone_mask(q,zone)].copy()
            for th in EDGE_THRESHOLDS:
                rows.append({"test_year":test_year,"variant":variant,"zone":zone,"edge_threshold":th,
                             **roi_summary(base[base["_edge"]>=th])})
            for lo,hi in EDGE_BANDS:
                sel=base[(base["_edge"]>=lo)&(base["_edge"]<hi)]
                bands.append({"test_year":test_year,"variant":variant,"zone":zone,
                              "edge_lo":lo,"edge_hi":"INF" if not np.isfinite(hi) else hi,
                              **roi_summary(sel)})
        return rows,bands

    model_rows=[]
    lambda_rows=[]
    coef_rows=[]
    value_rows=[]
    band_rows=[]
    pred_frames=[]

    for test_year in PRIMARY_YEARS:
        train=pd.concat([fair_by_year[y] for y in TRAINABLE_YEARS if y<test_year],ignore_index=True,copy=False)
        test=fair_by_year[test_year].copy().reset_index(drop=True)
        test["P__MARKET_DIRECT"]=test["market_prob_norm"].clip(1e-12,1.0)
        base=binary_metrics(test["target_win"],test["P__MARKET_DIRECT"])
        base.update(rank_metrics(test,"P__MARKET_DIRECT"))
        base["race_nll"]=race_nll(test,test["P__MARKET_DIRECT"])
        model_rows.append({"test_year":test_year,"variant":"MARKET_DIRECT","train_years":"","horses":len(test),**base})
        rr,bb=roi_rows(test,"P__MARKET_DIRECT","MARKET_DIRECT",test_year)
        value_rows.extend(rr); band_rows.extend(bb)

        def fit_variant(item):
            variant,features=item
            lam,grid=choose_lambda(train,features)
            beta,stats,obj=fit_beta(train,features,lam)
            pred=predict_anchor(test,features,beta,stats)
            return variant,features,lam,grid,beta,pred,obj

        print(f"ANCHOR_PARALLEL_MODELS year={test_year} variants={len(VARIANTS)} workers=3",flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futs=[ex.submit(fit_variant,item) for item in VARIANTS.items()]
            for fut in concurrent.futures.as_completed(futs):
                variant,features,lam,grid,beta,pred,obj=fut.result()
                pcol=f"P__{variant}"
                test[pcol]=pred
                met=binary_metrics(test["target_win"],pred)
                met.update(rank_metrics(test,pcol))
                met["race_nll"]=race_nll(test,pred)
                model_rows.append({
                    "test_year":test_year,"variant":variant,
                    "train_years":"|".join(map(str,sorted(set(train["year"])))),
                    "horses":len(test),"selected_lambda":lam,**met,
                })
                for l,vnll,vobj in grid:
                    lambda_rows.append({"test_year":test_year,"variant":variant,"lambda":l,"validation_race_nll":vnll,"fit_objective":vobj,"selected":int(l==lam)})
                for f,b in zip(features,beta):
                    coef_rows.append({"test_year":test_year,"variant":variant,"feature":f,"coef":float(b),"selected_lambda":lam})
                rr,bb=roi_rows(test,pcol,variant,test_year)
                value_rows.extend(rr); band_rows.extend(bb)
                print(f"ANCHOR_MODEL_DONE year={test_year} variant={variant} lambda={lam} race_nll={met['race_nll']:.6f} logloss={met['logloss']:.6f}",flush=True)

        keep=[
            "year","_race_id","_horse_id","_race_date","target_win","target_top3","final_win_odds",
            "market_prob_norm","market_rank","distill_ai_rank","risk","top3_overlap","ai_unique_top3",
            "P__MARKET_DIRECT",*[f"P__{v}" for v in VARIANTS],
        ]
        pred_frames.append(test[keep].copy())
        del train,test
        gc.collect()

    preds=pd.concat(pred_frames,ignore_index=True)
    pooled_models=[]
    pooled_values=[]
    pooled_bands=[]
    for variant in ("MARKET_DIRECT",*VARIANTS.keys()):
        pcol="P__MARKET_DIRECT" if variant=="MARKET_DIRECT" else f"P__{variant}"
        met=binary_metrics(preds["target_win"],preds[pcol])
        met.update(rank_metrics(preds,pcol))
        met["race_nll"]=race_nll(preds,preds[pcol])
        pooled_models.append({"test_year":"2023-2025","variant":variant,"horses":len(preds),**met})
        rr,bb=roi_rows(preds,pcol,variant,"2023-2025")
        pooled_values.extend(rr); pooled_bands.extend(bb)

    write_csv(out/"ability-checks.csv",ability_checks)
    write_csv(out/"risk-metrics.csv",risk_rows)
    write_csv(out/"fold-probability-metrics.csv",model_rows)
    write_csv(out/"pooled-probability-metrics.csv",pooled_models)
    write_csv(out/"lambda-selection.csv",lambda_rows)
    write_csv(out/"residual-coefficients.csv",coef_rows)
    write_csv(out/"fixed-edge-roi-by-year.csv",value_rows)
    write_csv(out/"fixed-edge-roi-pooled.csv",pooled_values)
    write_csv(out/"edge-band-roi-by-year.csv",band_rows)
    write_csv(out/"state-coverage.csv",state_cov)
    pred_path=out/"oos-fair-value-predictions.csv.gz"
    preds.to_csv(pred_path,index=False,compression="gzip")

    summary={
        "contract":"MARKET_FIXED_ANCHOR_RESIDUAL_V2",
        "formula":"p_i proportional to q_market_i * exp(delta_i); race probabilities normalize to 1",
        "market_coefficient":"fixed at 1; market is not re-learned as a feature",
        "residual_inputs":["Ability","Correction Risk","market-disagreement geometry"],
        "target":"actual race winner; race-level conditional likelihood",
        "calibration":"No isotonic calibration. Market-normalized probability is the fixed anchor; residual strength is regularized.",
        "lambda_selection":"Prior-data chronological 80/20 only; test year never selects lambda.",
        "roi_policy":"ROI never trains model or selects lambda. Fixed edge thresholds and non-overlapping edge bands are reported.",
        "speed":"Ability OOS folds use two concurrent workers; Risk OOS folds use two concurrent workers; compact fair-base cache is committed for future reuse.",
        "primary_years":[2023,2024,2025],
        "2026_locked":True,
        "promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== POOLED PROBABILITY =====")
    print(pd.DataFrame(pooled_models).sort_values("race_nll").to_csv(index=False))
    print("===== FIXED EDGE ROI DISAGREE =====")
    vv=pd.DataFrame(pooled_values)
    print(vv[vv["zone"].eq("DISAGREE")].sort_values(["variant","edge_threshold"]).to_csv(index=False))
    print("MARKET_FIXED_ANCHOR_RESIDUAL_V2_COMPLETE")

if __name__=="__main__":
    main()
