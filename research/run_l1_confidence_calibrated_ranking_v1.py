#!/usr/bin/env python3
import argparse,csv,gc,json,math
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

    ranking_rows=[]
    calib_rows=[]
    move_rows=[]
    meta_imp=[]
    cal_scored={}
    alpha_search=[]
    strict_blend_rows=[]

    for test in CAL_YEARS:
        hist_years=tuple(y for y in risk_oos if y<test)
        tr=pd.concat([risk_oos[y] for y in hist_years],ignore_index=True,copy=False)
        va=risk_oos[test].copy()

        p_mono,imp=fit_mono_top3(tr,va,9300000+test*100)
        p_grid=fit_grid(tr,va,bins=5,smooth=80.0)
        va["p_mono"]=p_mono
        va["p_grid"]=p_grid
        cal_scored[test]=va

        # Base score preserving exact original order.
        base_score=va["distill_ai_rank01"].astype(float)
        va["score_base"]=base_score
        va["score_mono"]=va["p_mono"]
        va["score_grid"]=va["p_grid"]
        va["score_blend50"]=0.50*base_score+0.50*va["p_mono"]

        variants=[
            ("BASE_ABILITY","score_base",None),
            ("GRID_TOP3","score_grid","p_grid"),
            ("MONO_TOP3","score_mono","p_mono"),
            ("BLEND50","score_blend50","p_mono"),
        ]
        base_rank=None
        for name,scol,pcol in variants:
            met,ranks=metrics_from_score(va,scol,name,test)
            ranking_rows.append(met)
            if name=="BASE_ABILITY":
                base_rank=ranks.rename(columns={"__rank":"__rank_base","distill_ai_rank":"distill_ai_rank_base"})
            else:
                rr=ranks.rename(columns={"__rank":"__rank_new"})
                mv=rank_movement(base_rank,rr)
                move_rows.append({"test_year":test,"variant":name,**mv})
            if pcol:
                calib_rows.append(calibration_metrics(va,pcol,name,test))

        total=sum(float(g) for _,g in imp) or 1.0
        for rank,(f,g) in enumerate(imp,1):
            meta_imp.append({"test_year":test,"rank":rank,"feature":f,"gain":float(g),"share":float(g)/total})

        # Candidate blend scores are evaluated OOS in each calibration year.
        for alpha in ALPHAS:
            va[f"blend_{alpha:.2f}"]=alpha*base_score+(1-alpha)*va["p_mono"]
            met,_=metrics_from_score(va,f"blend_{alpha:.2f}",f"BLEND_{alpha:.2f}",test)
            alpha_search.append({"test_year":test,"alpha":alpha,**{k:v for k,v in met.items() if k not in ("test_year","variant")}})
        print(
            f"CALIBRATION_READY year={test} base={ranking_rows[-4]['top1_top3_pct']:.4f} "
            f"mono={ranking_rows[-2]['top1_top3_pct']:.4f} blend50={ranking_rows[-1]['top1_top3_pct']:.4f}",
            flush=True,
        )
        del tr; gc.collect()

    # Strict blend selection can start in 2024 because 2023 is the first nested-OOS calibrated year.
    for test in STRICT_BLEND_YEARS:
        hist=[r for r in alpha_search if r["test_year"]<test]
        by_alpha={}
        for a in ALPHAS:
            vals=[r for r in hist if abs(r["alpha"]-a)<1e-12]
            by_alpha[a]=(
                float(np.mean([r["top1_top3_pct"] for r in vals])),
                float(np.mean([r["top1_win_pct"] for r in vals])),
            )
        # Prefer better Top3, then win, then larger alpha to preserve original L1 when tied.
        chosen=max(ALPHAS,key=lambda a:(by_alpha[a][0],by_alpha[a][1],a))
        va=cal_scored[test].copy()
        va["strict_score"]=chosen*va["distill_ai_rank01"]+(1-chosen)*va["p_mono"]
        met,_=metrics_from_score(va,"strict_score","STRICT_BLEND",test)
        base=next(r for r in ranking_rows if r["test_year"]==test and r["variant"]=="BASE_ABILITY")
        strict_blend_rows.append({
            "test_year":test,"selected_alpha":chosen,
            "train_selection_years":"|".join(map(str,sorted(set(r["test_year"] for r in hist)))),
            "baseline_top1_top3_pct":base["top1_top3_pct"],
            **{k:v for k,v in met.items() if k not in ("test_year","variant")},
            "uplift_pp":met["top1_top3_pct"]-base["top1_top3_pct"],
        })

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"ability-checks.csv",ability_checks)
    write_csv(out/"risk-metrics.csv",risk_rows)
    write_csv(out/"ranking-metrics.csv",ranking_rows)
    write_csv(out/"calibration-metrics.csv",calib_rows)
    write_csv(out/"rank-movement.csv",move_rows)
    write_csv(out/"alpha-search.csv",alpha_search)
    write_csv(out/"strict-blend.csv",strict_blend_rows)
    write_csv(out/"risk-feature-importance.csv",risk_imp)
    write_csv(out/"meta-feature-importance.csv",meta_imp)
    write_csv(out/"state-coverage.csv",state_cov)

    # Pooled simple variants over 2023-2025.
    pooled=[]
    for variant in ("BASE_ABILITY","GRID_TOP3","MONO_TOP3","BLEND50"):
        vals=[r for r in ranking_rows if r["variant"]==variant]
        pooled.append({
            "variant":variant,
            "years":"|".join(map(str,CAL_YEARS)),
            "top1_top3_pct":float(np.mean([r["top1_top3_pct"] for r in vals])),
            "top1_win_pct":float(np.mean([r["top1_win_pct"] for r in vals])),
            "winner_top3_capture_pct":float(np.mean([r["winner_top3_capture_pct"] for r in vals])),
            "winner_top6_capture_pct":float(np.mean([r["winner_top6_capture_pct"] for r in vals])),
            "positive_top1_top3_years":int(sum(
                r["top1_top3_pct"] >
                next(b["top1_top3_pct"] for b in ranking_rows if b["variant"]=="BASE_ABILITY" and b["test_year"]==r["test_year"])
                for r in vals
            )),
        })
    base=next(r for r in pooled if r["variant"]=="BASE_ABILITY")
    for r in pooled: r["top1_top3_uplift_pp"]=r["top1_top3_pct"]-base["top1_top3_pct"]
    write_csv(out/"pooled-ranking.csv",pooled)

    best=max([r for r in pooled if r["variant"]!="BASE_ABILITY"],key=lambda r:r["top1_top3_pct"])
    summary={
        "contract":"L1_CONFIDENCE_CALIBRATED_RANKING_V1",
        "question":"Can market-correction risk be folded back into the original L1 ranking through probability calibration rather than hard rank swapping?",
        "architecture":"Original Ability remains the backbone; risk is an uncertainty modifier inside a strictly-OOS top3 probability calibrator.",
        "market_is_teacher_only":True,
        "model_uses_current_market":False,
        "pooled_ranking":pooled,
        "best_simple_variant":best,
        "strict_blend":strict_blend_rows,
        "promotion_rule":"Promote only with material pooled improvement, multi-year stability, and no meaningful winner-capture regression.",
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== POOLED RANKING =====")
    print((out/"pooled-ranking.csv").read_text())
    print("===== STRICT BLEND =====")
    print((out/"strict-blend.csv").read_text())
    print("L1_CONFIDENCE_CALIBRATED_RANKING_V1_COMPLETE")

if __name__=="__main__":
    main()
