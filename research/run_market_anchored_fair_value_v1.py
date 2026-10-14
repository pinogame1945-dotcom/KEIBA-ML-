#!/usr/bin/env python3
import argparse,concurrent.futures,csv,gc,json,math
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

    # Exact OOS ability frames first.
    oos={}
    ability_checks=[]
    for test in TEST_YEARS:
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
            raise SystemExit(f"ABILITY reproduction drift year={test} expected={EXPECTED_ABILITY[test]} got={got}")
        ctx=add_ai_context(pred)
        ctx=ctx.merge(markets[test],on=["_race_id","_horse_id"],how="left",validate="one_to_one")
        if ctx["market_rank"].notna().mean()<0.99: raise SystemExit(f"market coverage too low year={test}")
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
        oos[test]=feat
        ability_checks.append({"test_year":test,"races":int(ctx["_race_id"].nunique()),"top1_top3_pct":got})
        print(f"ABILITY_OOS_READY year={test} rows={len(feat)} top1_top3={got:.6f}",flush=True)
        del train_base,valid_base,btr,bva,rtr,rva,mtr,mva,tr,va,pred,ctx; gc.collect()

    # Strict market-correction risk prediction for 2022-2025. Only FULL_PRERACE is needed.
    risk_oos={}
    risk_rows=[]
    risk_imp=[]
    for test in (2022,2023,2024,2025):
        teacher_years=tuple(y for y in TEST_YEARS if y<test)
        tr=pd.concat([oos[y] for y in teacher_years],ignore_index=True,copy=False)
        va=oos[test].copy()
        cols=[c for c in va.columns if not c.startswith("neg_")]
        risk,imp=fit_risk_full(tr[cols],va[cols],8500000+test*100)
        va=va.copy(); va["risk"]=risk
        risk_oos[test]=va
        scope=va[va["distill_ai_rank"]<=6]
        risk_rows.append({
            "test_year":test,"teacher_years":"|".join(map(str,teacher_years)),
            "top6_risk_auc":auc_rank(scope["_target_over3"],scope["risk"]),
            "top6_risk_logloss":logloss(scope["_target_over3"],scope["risk"]),
        })
        total=sum(float(g) for _,g in imp) or 1.0
        for rank,(f,g) in enumerate(imp[:80],1):
            risk_imp.append({"test_year":test,"rank":rank,"feature":f,"gain":float(g),"share":float(g)/total})
        print(f"RISK_OOS_READY year={test} auc={risk_rows[-1]['top6_risk_auc']:.4f}",flush=True)
        del tr,va; gc.collect()



    # Market-Anchored Fair Value Model.
    # FINAL market price is an L2 input. Ability and Correction Risk were generated strictly OOS above.
    # Outcome is the only model target; ROI/payout is never used for model fitting.
    from sklearn.impute import SimpleImputer
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    PRIMARY_YEARS=(2023,2024,2025)
    TRAINABLE_YEARS=(2022,2023,2024,2025)
    EDGE_THRESHOLDS=(0.0,0.05,0.10,0.20,0.30,0.50,0.75,1.00)
    ZONES=("ALL","AI_TOP3","DISAGREE","AI_UNIQUE_TOP3","OVERLAP2_AI_UNIQUE")
    MODEL_FAMILIES=("LOGIT","LGBM")
    VARIANTS={
        "MARKET_ONLY":[
            "market_prob_norm","market_logit","market_rank01","log_final_odds","field_size",
        ],
        "MARKET_ABILITY":[
            "market_prob_norm","market_logit","market_rank01","log_final_odds","field_size",
            "distill_ai_rank01","distill_score_z","distill_score_gap_to_best",
            "distill_is_ai_top1","distill_is_ai_top3","distill_is_ai_top6",
        ],
        "MARKET_ABILITY_RISK":[
            "market_prob_norm","market_logit","market_rank01","log_final_odds","field_size",
            "distill_ai_rank01","distill_score_z","distill_score_gap_to_best",
            "distill_is_ai_top1","distill_is_ai_top3","distill_is_ai_top6","risk",
        ],
        "FULL_DISAGREEMENT":[
            "market_prob_norm","market_logit","market_rank01","log_final_odds","field_size",
            "distill_ai_rank01","distill_score_z","distill_score_gap_to_best",
            "distill_is_ai_top1","distill_is_ai_top3","distill_is_ai_top6","risk",
            "rank_gap_norm","abs_rank_gap_norm","top3_overlap_norm",
            "ai_unique_top3","market_unique_top3","top1_match",
            "risk_x_abs_gap","ability_x_market_gap",
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
        pm=z["market_prob_norm"].clip(1e-7,1-1e-7)
        z["market_logit"]=np.log(pm/(1-pm))
        z["log_final_odds"]=np.log(z["final_win_odds"].clip(lower=1.000001))
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
        z["target_win"]=z["_is_win"].astype(int)
        z["target_top3"]=z["_is_top3"].astype(int)
        if z["market_prob_norm"].isna().any(): raise SystemExit(f"market probability missing y={year}")
        return z

    fair_by_year={y:prepare_fair_frame(y) for y in TRAINABLE_YEARS}
    for y,z in fair_by_year.items():
        print("FAIR_FRAME_READY "+json.dumps({
            "year":y,"races":int(z["_race_id"].nunique()),"horses":len(z),
            "win_rate":float(z["target_win"].mean()),
            "top3_overlap_mean":float(z.groupby("_race_id")["top3_overlap"].first().mean()),
        },separators=(",",":")),flush=True)

    def chronological_fit_cal(train):
        races=train[["_race_date","_race_id"]].drop_duplicates().sort_values(["_race_date","_race_id"]).reset_index(drop=True)
        cut=max(1,min(len(races)-1,int(len(races)*0.80)))
        fit_ids=set(races.iloc[:cut]["_race_id"].astype(str))
        cal_ids=set(races.iloc[cut:]["_race_id"].astype(str))
        fit=train[train["_race_id"].astype(str).isin(fit_ids)].copy()
        cal=train[train["_race_id"].astype(str).isin(cal_ids)].copy()
        if fit["target_win"].sum()<50 or cal["target_win"].sum()<20:
            raise SystemExit("insufficient chronological calibration positives")
        return fit,cal

    def model_task(family,variant,features,fit,cal,test,seed):
        xfit=fit[features]; xcal=cal[features]; xte=test[features]
        yfit=fit["target_win"].astype(int).to_numpy()
        ycal=cal["target_win"].astype(int).to_numpy()
        if family=="LOGIT":
            model=Pipeline([
                ("impute",SimpleImputer(strategy="median")),
                ("scale",StandardScaler()),
                ("model",LogisticRegression(C=0.35,solver="lbfgs",max_iter=1500)),
            ])
            model.fit(xfit,yfit)
            pcal=np.asarray(model.predict_proba(xcal)[:,1],dtype=float)
            pte_raw=np.asarray(model.predict_proba(xte)[:,1],dtype=float)
            coef=model.named_steps["model"].coef_[0]
            importance=[{"feature":f,"importance":float(v)} for f,v in zip(features,coef)]
        else:
            model=lgb.LGBMClassifier(
                objective="binary",n_estimators=320,learning_rate=0.025,num_leaves=15,
                min_child_samples=180,subsample=0.90,colsample_bytree=0.90,
                reg_alpha=1.0,reg_lambda=12.0,random_state=seed,
                n_jobs=1,deterministic=True,force_col_wise=True,verbosity=-1,
            )
            model.fit(xfit,yfit)
            pcal=np.asarray(model.predict_proba(xcal)[:,1],dtype=float)
            pte_raw=np.asarray(model.predict_proba(xte)[:,1],dtype=float)
            importance=[{"feature":f,"importance":float(v)} for f,v in zip(features,model.feature_importances_)]
        iso=IsotonicRegression(increasing=True,out_of_bounds="clip",y_min=1e-5,y_max=1-1e-5)
        iso.fit(pcal,ycal)
        pte=np.asarray(iso.predict(pte_raw),dtype=float)
        return family,variant,pte,importance

    def prob_metrics(y,p):
        y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
        return {
            "logloss":logloss(y,p),
            "brier":brier(y,p),
            "auc":auc_rank(y,p),
            "mean_pred_win_pct":100*float(p.mean()),
            "actual_win_pct":100*float(y.mean()),
        }

    def rank_metrics(z,pcol):
        q=z.sort_values(["_race_id",pcol,"_horse_id"],ascending=[True,False,True]).copy()
        q["__r"]=q.groupby("_race_id",sort=False).cumcount()+1
        top=q[q["__r"]==1]
        return {
            "races":int(q["_race_id"].nunique()),
            "top1_win_pct":100*float(top["target_win"].mean()),
            "top1_top3_pct":100*float(top["target_top3"].mean()),
        }

    def calibration_rows(z,pcol,family,variant,test_year):
        q=z.copy()
        q["_calbin"]=pd.qcut(q[pcol].rank(method="first"),10,labels=False,duplicates="drop")
        rows=[]
        for b,g in q.groupby("_calbin",sort=True):
            rows.append({
                "test_year":test_year,"family":family,"variant":variant,"bin":int(b)+1,
                "horses":len(g),"mean_pred_win_pct":100*float(g[pcol].mean()),
                "actual_win_pct":100*float(g["target_win"].mean()),
                "mean_final_odds":float(g["final_win_odds"].mean()),
            })
        return rows

    def zone_mask(z,zone):
        if zone=="ALL": return pd.Series(True,index=z.index)
        if zone=="AI_TOP3": return z["is_ai_top3"].eq(1)
        if zone=="DISAGREE": return z["distill_ai_rank"].ne(z["market_rank"])
        if zone=="AI_UNIQUE_TOP3": return z["ai_unique_top3"].eq(1)
        if zone=="OVERLAP2_AI_UNIQUE": return z["ai_unique_top3"].eq(1)&z["top3_overlap"].eq(2)
        raise ValueError(zone)

    def roi_rows(z,pcol,family,variant,test_year):
        z=z.copy()
        z["_fair_odds"]=1.0/z[pcol].clip(lower=1e-7)
        z["_edge"]=z[pcol]*z["final_win_odds"]-1.0
        rows=[]
        for zone in ZONES:
            base=z[zone_mask(z,zone)].copy()
            for th in EDGE_THRESHOLDS:
                q=base[base["_edge"]>=th]
                if q.empty:
                    rows.append({
                        "test_year":test_year,"family":family,"variant":variant,"zone":zone,
                        "edge_threshold":th,"bets":0,"races":0,"win_rate_pct":np.nan,
                        "avg_final_odds":np.nan,"avg_fair_odds":np.nan,"roi_pct":np.nan,"profit_yen":0.0
                    })
                    continue
                stake=100.0*len(q)
                ret=float((q["target_win"]*q["final_win_odds"]*100.0).sum())
                rows.append({
                    "test_year":test_year,"family":family,"variant":variant,"zone":zone,
                    "edge_threshold":th,"bets":int(len(q)),"races":int(q["_race_id"].nunique()),
                    "win_rate_pct":100*float(q["target_win"].mean()),
                    "avg_final_odds":float(q["final_win_odds"].mean()),
                    "avg_fair_odds":float(q["_fair_odds"].mean()),
                    "roi_pct":100*ret/stake,"profit_yen":ret-stake,
                })
        return rows

    model_rows=[]; cal_rows=[]; value_rows=[]; importance_rows=[]; increment_rows=[]
    prediction_frames=[]

    for test_year in PRIMARY_YEARS:
        train=pd.concat([fair_by_year[y] for y in TRAINABLE_YEARS if y<test_year],ignore_index=True,copy=False)
        test=fair_by_year[test_year].copy().reset_index(drop=True)
        fit,cal=chronological_fit_cal(train)
        print("FAIR_FOLD_START "+json.dumps({
            "test_year":test_year,"train_years":sorted(set(train["year"])),
            "fit_races":int(fit["_race_id"].nunique()),"cal_races":int(cal["_race_id"].nunique()),
            "test_races":int(test["_race_id"].nunique())
        },separators=(",",":")),flush=True)

        # Direct normalized final market probability is the non-parametric benchmark.
        test["P__MARKET_DIRECT"]=test["market_prob_norm"].clip(1e-7,1-1e-7)
        md=prob_metrics(test["target_win"],test["P__MARKET_DIRECT"])
        md.update(rank_metrics(test,"P__MARKET_DIRECT"))
        model_rows.append({"test_year":test_year,"family":"DIRECT","variant":"MARKET_DIRECT","train_years":"","horses":len(test),**md})
        cal_rows.extend(calibration_rows(test,"P__MARKET_DIRECT","DIRECT","MARKET_DIRECT",test_year))
        value_rows.extend(roi_rows(test,"P__MARKET_DIRECT","DIRECT","MARKET_DIRECT",test_year))

        tasks=[]
        for fi,family in enumerate(MODEL_FAMILIES):
            for vi,(variant,features) in enumerate(VARIANTS.items()):
                tasks.append((family,variant,features,fit,cal,test,9100000+test_year*100+fi*10+vi))
        workers=min(4,len(tasks))
        print(f"FAIR_PARALLEL_MODELS year={test_year} tasks={len(tasks)} workers={workers}",flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs=[ex.submit(model_task,*t) for t in tasks]
            for fut in concurrent.futures.as_completed(futs):
                family,variant,pred,imp=fut.result()
                pcol=f"P__{family}__{variant}"
                test[pcol]=pred
                met=prob_metrics(test["target_win"],pred)
                met.update(rank_metrics(test,pcol))
                model_rows.append({
                    "test_year":test_year,"family":family,"variant":variant,
                    "train_years":"|".join(map(str,sorted(set(train["year"])))),
                    "horses":len(test),**met
                })
                cal_rows.extend(calibration_rows(test,pcol,family,variant,test_year))
                value_rows.extend(roi_rows(test,pcol,family,variant,test_year))
                for rank,r in enumerate(sorted(imp,key=lambda x:abs(x["importance"]),reverse=True),1):
                    importance_rows.append({
                        "test_year":test_year,"family":family,"variant":variant,"rank":rank,**r
                    })
                print(f"FAIR_MODEL_DONE year={test_year} family={family} variant={variant} logloss={met['logloss']:.6f}",flush=True)

        # Paired per-race logloss increments. Negative delta means the richer model is better.
        eps=1e-7
        y=test["target_win"].to_numpy(dtype=float)
        pairs=[
            ("MARKET_DIRECT","P__MARKET_DIRECT"),
            ("LOGIT_MARKET","P__LOGIT__MARKET_ONLY"),
            ("LOGIT_ABILITY","P__LOGIT__MARKET_ABILITY"),
            ("LOGIT_RISK","P__LOGIT__MARKET_ABILITY_RISK"),
            ("LOGIT_FULL","P__LOGIT__FULL_DISAGREEMENT"),
            ("LGBM_MARKET","P__LGBM__MARKET_ONLY"),
            ("LGBM_ABILITY","P__LGBM__MARKET_ABILITY"),
            ("LGBM_RISK","P__LGBM__MARKET_ABILITY_RISK"),
            ("LGBM_FULL","P__LGBM__FULL_DISAGREEMENT"),
        ]
        losses={}
        for name,col in pairs:
            p=np.clip(test[col].to_numpy(float),eps,1-eps)
            losses[name]=-(y*np.log(p)+(1-y)*np.log(1-p))
        for family in MODEL_FAMILIES:
            chain=[
                ("MARKET_DIRECT",f"{family}_MARKET"),
                (f"{family}_MARKET",f"{family}_ABILITY"),
                (f"{family}_ABILITY",f"{family}_RISK"),
                (f"{family}_RISK",f"{family}_FULL"),
            ]
            for base_name,new_name in chain:
                d=losses[new_name]-losses[base_name]
                tmp=pd.DataFrame({"race_id":test["_race_id"].astype(str).to_numpy(),"d":d})
                rd=tmp.groupby("race_id")["d"].mean()
                mean=float(rd.mean()); se=float(rd.std(ddof=1)/math.sqrt(len(rd)))
                increment_rows.append({
                    "test_year":test_year,"family":family,"base":base_name,"new":new_name,
                    "delta_logloss_new_minus_base":mean,
                    "ci95_lo":mean-1.96*se,"ci95_hi":mean+1.96*se,
                    "race_fraction_new_better":float((rd<0).mean()),
                    "races":len(rd),
                })

        keep=[
            "year","_race_id","_horse_id","_race_date","target_win","target_top3","final_win_odds",
            "market_prob_norm","market_rank","distill_ai_rank","risk","top3_overlap","ai_unique_top3",
            *[x[1] for x in pairs],
        ]
        prediction_frames.append(test[keep].copy())
        del train,test,fit,cal; gc.collect()

    models=pd.DataFrame(model_rows)
    values=pd.DataFrame(value_rows)
    preds=pd.concat(prediction_frames,ignore_index=True)

    # Pooled OOS summaries preserve the exact per-year OOS predictions.
    pooled_models=[]
    pcols=[c for c in preds.columns if c.startswith("P__")]
    for col in pcols:
        if col=="P__MARKET_DIRECT":
            family,variant="DIRECT","MARKET_DIRECT"
        else:
            _,family,variant=col.split("__",2)
        met=prob_metrics(preds["target_win"],preds[col])
        met.update(rank_metrics(preds,col))
        pooled_models.append({"family":family,"variant":variant,"test_year":"2023-2025","horses":len(preds),**met})

    pooled_values=[]
    for col in pcols:
        if col=="P__MARKET_DIRECT":
            family,variant="DIRECT","MARKET_DIRECT"
        else:
            _,family,variant=col.split("__",2)
        pooled_values.extend(roi_rows(preds,col,family,variant,"2023-2025"))

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"ability-checks.csv",ability_checks)
    write_csv(out/"risk-metrics.csv",risk_rows)
    write_csv(out/"fold-probability-metrics.csv",model_rows)
    write_csv(out/"pooled-probability-metrics.csv",pooled_models)
    write_csv(out/"paired-logloss-increments.csv",increment_rows)
    write_csv(out/"calibration-deciles.csv",cal_rows)
    write_csv(out/"fixed-edge-roi-by-year.csv",value_rows)
    write_csv(out/"fixed-edge-roi-pooled.csv",pooled_values)
    write_csv(out/"feature-importance.csv",importance_rows)
    write_csv(out/"state-coverage.csv",state_cov)

    summary={
        "contract":"MARKET_ANCHORED_FAIR_VALUE_V1",
        "question":"Can strictly-OOS Ability and Correction Risk improve the market's win probability enough to identify genuinely underpriced horses?",
        "target":"actual win only",
        "market_role":"FINAL odds/rank are L2 pricing inputs; they are not used by the upstream Ability model.",
        "risk_role":"Correction Risk is generated strictly OOS from historical market-teacher labels before the current market is combined.",
        "probability_models":[
            "MARKET_DIRECT",
            "LOGIT: MARKET_ONLY -> +ABILITY -> +RISK -> +DISAGREEMENT",
            "LGBM: MARKET_ONLY -> +ABILITY -> +RISK -> +DISAGREEMENT"
        ],
        "calibration":"Each test year uses a chronological 80/20 fit/calibration split of prior-year data only; isotonic calibration is fit only on the prior calibration slice.",
        "parallelization":"Eight model tasks per fold run with up to four concurrent workers; upstream OOS Ability/Risk is computed once and reused.",
        "fair_odds":"1 / calibrated predicted win probability",
        "value_edge":"calibrated win probability * final win odds - 1",
        "roi_policy":"ROI never trains the probability model. Fixed edge thresholds are reported without test-year threshold optimization.",
        "primary_years":[2023,2024,2025],
        "2026_locked":True,
        "promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== POOLED PROBABILITY =====")
    print(pd.DataFrame(pooled_models).sort_values("logloss").to_csv(index=False))
    print("===== FIXED EDGE ROI (OVERLAP2 AI UNIQUE) =====")
    pv=pd.DataFrame(pooled_values)
    print(pv[pv["zone"].eq("OVERLAP2_AI_UNIQUE")].sort_values(["family","variant","edge_threshold"]).to_csv(index=False))
    print("MARKET_ANCHORED_FAIR_VALUE_V1_COMPLETE")


if __name__=="__main__":
    main()
