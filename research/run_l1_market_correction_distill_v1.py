#!/usr/bin/env python3
import argparse,csv,gc,json,math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l1_objective_rebuild_v1 import (
    YEARS,TEST_YEARS,LOCKED_YEAR,prepare,model_common,read_year,select_variant,
)
from run_l1_ceiling_audit_v1 import strip_forbidden
from run_l1_state_transition_v1 import build_state_features,attach_state
from run_l1_residual_regime_full_pairwise_v1 import (
    build_regime_features,fit_direct_rank,top1_metrics,
)
from run_l1_sameday_track_bias_v1 import build_same_day_bias,bias_cols,speed_cols
from run_l1_negative_evidence_penalty_v1 import build_negative_evidence,read_market_year

CORR_TEST_YEARS=(2022,2023,2024,2025)
PRIMARY_YEARS=(2023,2024,2025)
VARIANTS=("RANK_ONLY","FULL_PRERACE","FULL_PRERACE_NEG")

EXPECTED_ABILITY={
    2021:59.34606481481482,
    2022:60.90856481481482,
    2023:59.432870370370374,
    2024:62.673611111111114,
    2025:61.660879629629626,
}

def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def market_rank_frame(market):
    x=market.copy()
    pop=pd.to_numeric(x["final_popularity"],errors="coerce")
    odds=pd.to_numeric(x["final_win_odds"],errors="coerce")
    odds_rank=odds.groupby(x["_race_id"]).rank(method="min",ascending=True)
    x["market_rank"]=pop.where(pop>=1,odds_rank)
    return x[["_race_id","_horse_id","market_rank","final_win_odds"]]

def add_ai_context(pred):
    x=pred.copy()
    x=x.sort_values(["_race_id","_score","_horse_id"],ascending=[True,False,True]).reset_index(drop=True)
    x["distill_ai_rank"]=x.groupby("_race_id",sort=False).cumcount()+1
    n=x.groupby("_race_id")["_horse_id"].transform("size").astype(float)
    x["distill_field_size"]=n
    x["distill_ai_rank01"]=1.0-(x["distill_ai_rank"]-1.0)/(n-1.0).clip(lower=1.0)
    mean=x.groupby("_race_id")["_score"].transform("mean")
    std=x.groupby("_race_id")["_score"].transform("std").replace(0,np.nan)
    x["distill_score_z"]=((x["_score"]-mean)/std).fillna(0.0)
    best=x.groupby("_race_id")["_score"].transform("max")
    x["distill_score_gap_to_best"]=(best-x["_score"]).clip(lower=0.0)
    x["distill_is_ai_top1"]=(x["distill_ai_rank"]==1).astype(float)
    x["distill_is_ai_top3"]=(x["distill_ai_rank"]<=3).astype(float)
    x["distill_is_ai_top6"]=(x["distill_ai_rank"]<=6).astype(float)
    return x

def auc_rank(y,p):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    pos=int(y.sum()); neg=len(y)-pos
    if pos==0 or neg==0:
        return np.nan
    r=pd.Series(p).rank(method="average").to_numpy()
    return float((r[y==1].sum()-pos*(pos+1)/2)/(pos*neg))

def average_precision(y,p):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    pos=int(y.sum())
    if pos==0:
        return np.nan
    order=np.argsort(-p,kind="mergesort")
    ys=y[order]
    tp=np.cumsum(ys)
    precision=tp/(np.arange(len(ys))+1)
    return float((precision*ys).sum()/pos)

def recall_at_prevalence(y,p):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    pos=int(y.sum())
    if pos==0:
        return np.nan
    k=max(1,pos)
    idx=np.argsort(-p,kind="mergesort")[:k]
    return float(y[idx].sum()/pos)

def top_decile_enrichment(y,p):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    base=float(y.mean())
    if base<=0:
        return np.nan
    k=max(1,int(math.ceil(0.10*len(y))))
    idx=np.argsort(-p,kind="mergesort")[:k]
    return float(y[idx].mean()/base)

def regression_metrics(y,p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    mae=float(np.mean(np.abs(y-p)))
    rmse=float(np.sqrt(np.mean((y-p)**2)))
    if np.std(y)>0 and np.std(p)>0:
        corr=float(np.corrcoef(y,p)[0,1])
    else:
        corr=np.nan
    return mae,rmse,corr

def corr_params(seed):
    p=model_common(seed)
    p["n_estimators"]=260
    p["learning_rate"]=0.035
    p["num_leaves"]=31
    p["min_child_samples"]=120
    p["reg_lambda"]=8.0
    p["reg_alpha"]=1.0
    return p

RANK_COLS=(
    "distill_ai_rank","distill_field_size","distill_ai_rank01","distill_score_z",
    "distill_score_gap_to_best","distill_is_ai_top1","distill_is_ai_top3","distill_is_ai_top6",
)

def fit_rank_only(train,valid,seed):
    xtr=train[list(RANK_COLS)].apply(pd.to_numeric,errors="coerce").astype("float32")
    xva=valid[list(RANK_COLS)].apply(pd.to_numeric,errors="coerce").astype("float32")
    y3=train["_target_over3"].astype(int).to_numpy()
    yup=train["_target_up"].astype(int).to_numpy()
    ygap=train["_target_gap_clip"].astype(float).to_numpy()

    m3=lgb.LGBMClassifier(objective="binary",**corr_params(seed+1))
    m3.fit(xtr,y3)
    p3=np.asarray(m3.predict_proba(xva)[:,1],dtype=float)
    imp3=sorted(zip(xtr.columns,m3.booster_.feature_importance(importance_type="gain")),key=lambda z:z[1],reverse=True)
    del m3; gc.collect()

    mu=lgb.LGBMClassifier(objective="binary",**corr_params(seed+2))
    mu.fit(xtr,yup)
    pup=np.asarray(mu.predict_proba(xva)[:,1],dtype=float)
    del mu; gc.collect()

    mg=lgb.LGBMRegressor(objective="huber",**corr_params(seed+3))
    mg.fit(xtr,ygap)
    pg=np.asarray(mg.predict(xva),dtype=float)
    del mg,xtr,xva; gc.collect()
    return p3,pup,pg,imp3

def fit_full(train,valid,seed):
    drop_targets=["_target_gap","_target_gap_clip","_target_up","_target_over3","market_rank","final_win_odds"]
    tr=train.drop(columns=[c for c in drop_targets if c in train.columns]).copy()
    va=valid.drop(columns=[c for c in drop_targets if c in valid.columns]).copy()
    xtr,xva,cats=prepare(tr,va)
    y3=train["_target_over3"].astype(int).to_numpy()
    yup=train["_target_up"].astype(int).to_numpy()
    ygap=train["_target_gap_clip"].astype(float).to_numpy()

    m3=lgb.LGBMClassifier(objective="binary",**corr_params(seed+1))
    m3.fit(xtr,y3,categorical_feature=cats)
    p3=np.asarray(m3.predict_proba(xva)[:,1],dtype=float)
    imp3=sorted(zip(xtr.columns,m3.booster_.feature_importance(importance_type="gain")),key=lambda z:z[1],reverse=True)
    del m3; gc.collect()

    mu=lgb.LGBMClassifier(objective="binary",**corr_params(seed+2))
    mu.fit(xtr,yup,categorical_feature=cats)
    pup=np.asarray(mu.predict_proba(xva)[:,1],dtype=float)
    del mu; gc.collect()

    mg=lgb.LGBMRegressor(objective="huber",**corr_params(seed+3))
    mg.fit(xtr,ygap,categorical_feature=cats)
    pg=np.asarray(mg.predict(xva),dtype=float)
    del mg,xtr,xva; gc.collect()
    return p3,pup,pg,imp3

def eval_classifier(df,pred_col,target_col,scope):
    x=df if scope=="ALL" else df[df["distill_ai_rank"]<=6]
    y=x[target_col].astype(int).to_numpy()
    p=x[pred_col].to_numpy(dtype=float)
    return {
        "scope":scope,
        "n":len(x),
        "prevalence_pct":100*float(y.mean()),
        "roc_auc":auc_rank(y,p),
        "average_precision":average_precision(y,p),
        "recall_at_prevalence_pct":100*recall_at_prevalence(y,p),
        "top_decile_enrichment":top_decile_enrichment(y,p),
    }

def rerank_top6_metrics(df,score_col):
    x=df.copy()
    x["__score"]=pd.to_numeric(x[score_col],errors="coerce").fillna(-1e30)
    # Preserve the original candidate boundary: only reorder the original AI Top6.
    inside=x["distill_ai_rank"]<=6
    x["__candidate_score"]=np.where(inside,x["__score"],-1e30)
    x=x.sort_values(["_race_id","__candidate_score","_horse_id"],ascending=[True,False,True])
    top=x.groupby("_race_id",sort=False).head(1)
    return {
        "races":int(top["_race_id"].nunique()),
        "top1_top3_pct":100*float(top["_is_top3"].mean()),
        "top1_win_pct":100*float(top["_is_win"].mean()),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--year-file",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--out-dir",required=True)
    a=p.parse_args()
    paths={}
    for spec in a.year_file:
        y,path=spec.split(":",1); paths[int(y)]=path
    if set(paths)!=set(YEARS):
        raise SystemExit(f"year file mismatch {sorted(paths)}")
    if LOCKED_YEAR in paths:
        raise SystemExit("2026 sealed")

    print(json.dumps({
        "contract":"L1_MARKET_CORRECTION_DISTILL_V1_RUNTIME",
        "teacher_market":"FINAL popularity rank used only as historical teacher label",
        "student_market_inputs":False,
        "teacher_gap":"market_rank - exact OOS AI rank",
        "target_over3":"gap >= 3 (AI promoted horse by at least 3 ranks vs market)",
        "correction_evaluation_years":list(CORR_TEST_YEARS),
        "primary_years":list(PRIMARY_YEARS),
        "strict_expanding_teacher":"correction test year Y trains only on OOS teacher rows from years < Y",
        "ability_uses_odds":False,
        "2026_locked":True,
    },separators=(",",":")),flush=True)

    neg_by_year,neg_cols,neg_cov=build_negative_evidence(paths)
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

    oos={}
    ability_checks=[]
    for test in TEST_YEARS:
        train_years=(test-2,test-1)
        train_base=pd.concat([get_base(y) for y in train_years],ignore_index=True,copy=False)
        valid_base=get_base(test).copy()
        btr=pd.concat([bias_by_year[y] for y in train_years],ignore_index=True,copy=False)
        bva=bias_by_year[test]
        rtr=pd.concat([regime_by_year[y] for y in train_years],ignore_index=True,copy=False)
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
        labels=markets[test]
        ctx=ctx.merge(labels,on=["_race_id","_horse_id"],how="left",validate="one_to_one")
        if ctx["market_rank"].notna().mean()<0.99:
            raise SystemExit(f"market coverage too low year={test}")
        ctx["market_rank"]=pd.to_numeric(ctx["market_rank"],errors="coerce")
        ctx["_target_gap"]=ctx["market_rank"]-ctx["distill_ai_rank"]
        ctx["_target_gap_clip"]=ctx["_target_gap"].clip(-6,6)
        ctx["_target_up"]=(ctx["_target_gap"]>0).astype(int)
        ctx["_target_over3"]=(ctx["_target_gap"]>=3).astype(int)

        # Student features are strictly pre-race. Current/final market columns stay label-side only.
        feat=va.copy()
        feat=feat.merge(
            ctx[["_race_id","_horse_id","_score","distill_ai_rank","distill_field_size","distill_ai_rank01",
                 "distill_score_z","distill_score_gap_to_best","distill_is_ai_top1","distill_is_ai_top3",
                 "distill_is_ai_top6","market_rank","final_win_odds","_target_gap","_target_gap_clip",
                 "_target_up","_target_over3"]],
            on=["_race_id","_horse_id"],how="left",validate="one_to_one",
        )
        feat=attach_state(feat,neg_by_year[test][["_race_id","_horse_id",*neg_cols]],neg_cols)
        oos[test]=feat
        ability_checks.append({
            "test_year":test,"races":int(ctx["_race_id"].nunique()),
            "top1_top3_pct":got,
            "market_coverage_pct":100*float(ctx["market_rank"].notna().mean()),
            "top6_over3_prevalence_pct":100*float(ctx.loc[ctx["distill_ai_rank"]<=6,"_target_over3"].mean()),
        })
        print(
            f"TEACHER_OOS_READY year={test} rows={len(feat)} "
            f"ability_top1_top3={got:.6f} top6_over3={ability_checks[-1]['top6_over3_prevalence_pct']:.3f}",
            flush=True,
        )
        del train_base,valid_base,btr,bva,rtr,rva,mtr,mva,tr,va,pred,ctx
        gc.collect()

    metrics=[]
    reg_rows=[]
    rerank_rows=[]
    importances=[]

    for test in CORR_TEST_YEARS:
        train_years=tuple(y for y in TEST_YEARS if y<test)
        train=pd.concat([oos[y] for y in train_years],ignore_index=True,copy=False)
        valid=oos[test].copy()
        for variant in VARIANTS:
            if variant=="RANK_ONLY":
                p3,pup,pg,imp=fit_rank_only(train,valid,7200000+test*100)
            elif variant=="FULL_PRERACE":
                base_cols=[c for c in valid.columns if not c.startswith("neg_")]
                p3,pup,pg,imp=fit_full(train[base_cols],valid[base_cols],7300000+test*100)
            else:
                p3,pup,pg,imp=fit_full(train,valid,7400000+test*100)

            scored=valid[["_race_id","_horse_id","_finish","_is_win","_is_top3","distill_ai_rank",
                          "distill_ai_rank01","market_rank","_target_gap","_target_gap_clip",
                          "_target_up","_target_over3"]].copy()
            scored["p_over3"]=p3
            scored["p_up"]=pup
            scored["pred_gap"]=np.clip(pg,-6,6)

            for scope in ("ALL","AI_TOP6"):
                for target,pcol in (("OVER3","p_over3"),("UP_ANY","p_up")):
                    rec=eval_classifier(
                        scored,pcol,
                        "_target_over3" if target=="OVER3" else "_target_up",
                        scope,
                    )
                    metrics.append({
                        "test_year":test,
                        "train_teacher_years":"|".join(map(str,train_years)),
                        "variant":variant,"target":target,**rec,
                    })

            scope=scored[scored["distill_ai_rank"]<=6]
            mae,rmse,corr=regression_metrics(scope["_target_gap_clip"],scope["pred_gap"])
            reg_rows.append({
                "test_year":test,"train_teacher_years":"|".join(map(str,train_years)),
                "variant":variant,"scope":"AI_TOP6","n":len(scope),
                "gap_mae":mae,"gap_rmse":rmse,"gap_corr":corr,
            })

            # A market-distilled ranking: predict where market would demote/promote the current AI Top6.
            scored["distilled_market_rank"]=scored["distill_ai_rank"]+scored["pred_gap"]
            scored["distilled_score"]=-scored["distilled_market_rank"]
            rr=rerank_top6_metrics(scored,"distilled_score")
            base_top=scored[scored["distill_ai_rank"]==1]
            rerank_rows.append({
                "test_year":test,"variant":variant,
                "base_ai_top1_top3_pct":100*float(base_top["_is_top3"].mean()),
                "distilled_top1_top3_pct":rr["top1_top3_pct"],
                "uplift_pp":rr["top1_top3_pct"]-100*float(base_top["_is_top3"].mean()),
                "base_ai_top1_win_pct":100*float(base_top["_is_win"].mean()),
                "distilled_top1_win_pct":rr["top1_win_pct"],
            })

            for rank,(feature,gain) in enumerate(imp[:100],start=1):
                importances.append({
                    "test_year":test,"variant":variant,"target":"OVER3",
                    "importance_rank":rank,"feature":feature,"gain":float(gain),
                    "is_negative_feature":bool(str(feature).startswith("neg_")),
                })

            print(
                f"DISTILL_FOLD_READY year={test} variant={variant} "
                f"over3_auc_top6={metrics[-2]['roc_auc'] if metrics[-2]['target']=='OVER3' else metrics[-1]['roc_auc']:.4f} "
                f"rerank_uplift={rerank_rows[-1]['uplift_pp']:.4f}",
                flush=True,
            )
        del train,valid
        gc.collect()

    # Aggregate primary 2023-2025 so 2022 one-year teacher training does not dominate the conclusion.
    pooled=[]
    for variant in VARIANTS:
        vals=[r for r in metrics if r["variant"]==variant and r["target"]=="OVER3" and r["scope"]=="AI_TOP6" and r["test_year"] in PRIMARY_YEARS]
        pooled.append({
            "variant":variant,
            "years":"|".join(map(str,PRIMARY_YEARS)),
            "n":int(sum(r["n"] for r in vals)),
            "prevalence_pct":float(np.average([r["prevalence_pct"] for r in vals],weights=[r["n"] for r in vals])),
            "roc_auc":float(np.average([r["roc_auc"] for r in vals],weights=[r["n"] for r in vals])),
            "average_precision":float(np.average([r["average_precision"] for r in vals],weights=[r["n"] for r in vals])),
            "recall_at_prevalence_pct":float(np.average([r["recall_at_prevalence_pct"] for r in vals],weights=[r["n"] for r in vals])),
            "top_decile_enrichment":float(np.average([r["top_decile_enrichment"] for r in vals],weights=[r["n"] for r in vals])),
        })
    rerank_pool=[]
    for variant in VARIANTS:
        vals=[r for r in rerank_rows if r["variant"]==variant and r["test_year"] in PRIMARY_YEARS]
        rerank_pool.append({
            "variant":variant,
            "years":"|".join(map(str,PRIMARY_YEARS)),
            "base_ai_top1_top3_pct":float(np.mean([r["base_ai_top1_top3_pct"] for r in vals])),
            "distilled_top1_top3_pct":float(np.mean([r["distilled_top1_top3_pct"] for r in vals])),
            "uplift_pp":float(np.mean([r["uplift_pp"] for r in vals])),
            "positive_years":int(sum(r["uplift_pp"]>0 for r in vals)),
        })

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"ability-checks.csv",ability_checks)
    write_csv(out/"classifier-metrics.csv",metrics)
    write_csv(out/"regression-metrics.csv",reg_rows)
    write_csv(out/"rerank-metrics.csv",rerank_rows)
    write_csv(out/"pooled-over3-top6.csv",pooled)
    write_csv(out/"pooled-rerank.csv",rerank_pool)
    write_csv(out/"feature-importance.csv",importances)
    write_csv(out/"negative-evidence-coverage.csv",neg_cov)
    write_csv(out/"state-coverage.csv",state_cov)

    best=max(pooled,key=lambda r:r["roc_auc"])
    best_rerank=max(rerank_pool,key=lambda r:r["uplift_pp"])
    (out/"summary.json").write_text(json.dumps({
        "contract":"L1_MARKET_CORRECTION_DISTILL_V1",
        "question":"Can pre-race information predict the direction/magnitude of the final market's correction to the exact OOS AI ranking?",
        "teacher_label":"final market rank is historical supervision only; never a student input",
        "strictness":"Ability teacher predictions are OOS. Correction model for year Y uses only OOS teacher-labeled years < Y.",
        "primary_years":list(PRIMARY_YEARS),
        "variants":{
            "RANK_ONLY":"AI rank/score geometry only",
            "FULL_PRERACE":"current L1 pre-race feature space + AI rank/score geometry",
            "FULL_PRERACE_NEG":"FULL_PRERACE plus strictly-prior negative-evidence summaries",
        },
        "primary_over3_top6":pooled,
        "best_over3_predictor":best,
        "market_distilled_rerank":rerank_pool,
        "best_rerank":best_rerank,
        "interpretation_rule":{
            "auc_near_0_5":"market correction is not predictable from these pre-race features",
            "full_beats_rank_only":"existing horse/race features contain information about future market correction beyond rank geometry",
            "rerank_positive":"predicted market correction recovers some of market's edge without using current market",
        },
        "market_is_teacher_only":True,
        "model_uses_current_market":False,
        "2026_locked":True,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== POOLED OVER3 TOP6 =====")
    print((out/"pooled-over3-top6.csv").read_text())
    print("===== POOLED RERANK =====")
    print((out/"pooled-rerank.csv").read_text())
    print("L1_MARKET_CORRECTION_DISTILL_V1_COMPLETE")

if __name__=="__main__":
    main()
