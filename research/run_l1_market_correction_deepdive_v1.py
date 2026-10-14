#!/usr/bin/env python3
import argparse,csv,gc,json,math
from collections import defaultdict
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
from run_l1_market_correction_distill_v1 import (
    market_rank_frame,add_ai_context,auc_rank,average_precision,
)

CORR_YEARS=(2022,2023,2024,2025)
PRIMARY_YEARS=(2023,2024,2025)
RISK_VARIANTS=("RANK_ONLY","FULL_PRERACE","FULL_PRERACE_NEG")
PRIMARY_VARIANT="FULL_PRERACE"
EXPECTED_ABILITY={
    2021:59.34606481481482,
    2022:60.90856481481482,
    2023:59.432870370370374,
    2024:62.673611111111114,
    2025:61.660879629629626,
}
RANK_COLS=(
    "distill_ai_rank","distill_field_size","distill_ai_rank01","distill_score_z",
    "distill_score_gap_to_best","distill_is_ai_top1","distill_is_ai_top3","distill_is_ai_top6",
)
QGRID=(0.50,0.60,0.70,0.80,0.85,0.90,0.95)
RESCUE_MODES=("NEXT_SAFE","MIN_RISK_TOP3")

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def risk_params(seed):
    p=model_common(seed)
    p["n_estimators"]=260
    p["learning_rate"]=0.035
    p["num_leaves"]=31
    p["min_child_samples"]=120
    p["reg_lambda"]=8.0
    p["reg_alpha"]=1.0
    return p

def fit_risk_rank_only(train,valid,seed):
    xtr=train[list(RANK_COLS)].apply(pd.to_numeric,errors="coerce").astype("float32")
    xva=valid[list(RANK_COLS)].apply(pd.to_numeric,errors="coerce").astype("float32")
    y=train["_target_over3"].astype(int).to_numpy()
    m=lgb.LGBMClassifier(objective="binary",**risk_params(seed))
    m.fit(xtr,y)
    p=np.asarray(m.predict_proba(xva)[:,1],dtype=float)
    imp=sorted(zip(xtr.columns,m.booster_.feature_importance(importance_type="gain")),
               key=lambda z:z[1],reverse=True)
    del m,xtr,xva; gc.collect()
    return p,imp

def fit_risk_full(train,valid,seed):
    drop_targets=["_target_gap","_target_gap_clip","_target_up","_target_over3","market_rank","final_win_odds"]
    tr=train.drop(columns=[c for c in drop_targets if c in train.columns]).copy()
    va=valid.drop(columns=[c for c in drop_targets if c in valid.columns]).copy()
    xtr,xva,cats=prepare(tr,va)
    y=train["_target_over3"].astype(int).to_numpy()
    m=lgb.LGBMClassifier(objective="binary",**risk_params(seed))
    m.fit(xtr,y,categorical_feature=cats)
    p=np.asarray(m.predict_proba(xva)[:,1],dtype=float)
    imp=sorted(zip(xtr.columns,m.booster_.feature_importance(importance_type="gain")),
               key=lambda z:z[1],reverse=True)
    del m,xtr,xva; gc.collect()
    return p,imp

def logloss(y,p):
    y=np.asarray(y,dtype=float); p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def brier(y,p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    return float(np.mean((y-p)**2))

def quantile_bucket(s,q=10):
    r=s.rank(method="first",pct=True)
    return np.minimum(q,np.maximum(1,np.ceil(r*q))).astype(int)

def wilson(success,n,z=1.959963984540054):
    if n<=0: return (np.nan,np.nan)
    ph=success/n
    den=1+z*z/n
    center=(ph+z*z/(2*n))/den
    half=z*math.sqrt((ph*(1-ph)+z*z/(4*n))/n)/den
    return (100*(center-half),100*(center+half))

def feature_domain(c):
    s=str(c)
    rules=(
        ("AI_GEOMETRY",("distill_","_score")),
        ("SPEED_REGIME",("regime_speed","seq_")),
        ("DAY_BIAS",("day_","bias_")),
        ("NETWORK",("network_","opponent_")),
        ("STYLE_LAP",("style_","lap_","timepace_")),
        ("DISTANCE",("distance_","same_distance","auto_distance")),
        ("ACTOR",("actor_","jockey","trainer")),
        ("BACKFILL",("backfill_","venue_code","surface","direction","track_condition")),
        ("NEGATIVE",("neg_",)),
        ("RECENT",("recent_","previous_","prior_")),
    )
    for name,prefixes in rules:
        if any(s.startswith(p) or p in s for p in prefixes): return name
    return "OTHER"

def top1_metrics_from_choice(df,choice):
    chosen=df.loc[choice]
    return {
        "n":len(chosen),
        "top3_pct":100*float(chosen["_is_top3"].mean()),
        "win_pct":100*float(chosen["_is_win"].mean()),
    }

def choose_policy(df,threshold,mode):
    # One selected horse index per race. Baseline top1 unless it is high risk.
    picks=[]
    replaced=0
    eligible=0
    for _,g in df[df["distill_ai_rank"]<=6].groupby("_race_id",sort=False):
        g=g.sort_values(["distill_ai_rank","_horse_id"])
        top=g.iloc[0]
        pick=top
        if float(top["risk"])>=threshold:
            eligible+=1
            if mode=="NEXT_SAFE":
                cand=g[(g["distill_ai_rank"]>1)&(g["risk"]<threshold)]
                if len(cand):
                    pick=cand.sort_values(["distill_ai_rank","_horse_id"]).iloc[0]
            elif mode=="MIN_RISK_TOP3":
                cand=g[g["distill_ai_rank"]<=3]
                pick=cand.sort_values(["risk","distill_ai_rank","_horse_id"]).iloc[0]
            else:
                raise ValueError(mode)
        if int(pick["distill_ai_rank"])!=1: replaced+=1
        picks.append(pick.name)
    return np.asarray(picks,dtype=int),eligible,replaced

def policy_score(df,threshold,mode):
    picks,eligible,replaced=choose_policy(df,threshold,mode)
    met=top1_metrics_from_choice(df,picks)
    met.update({
        "threshold":float(threshold),"mode":mode,
        "flagged_races":int(eligible),"replaced_races":int(replaced),
        "replacement_rate_pct":100*replaced/max(1,met["n"]),
    })
    return met

def fit_meta(train,valid,use_risk,seed):
    cols=["distill_ai_rank","distill_ai_rank01","distill_score_z","distill_score_gap_to_best"]
    if use_risk: cols.append("risk")
    tr=train[train["distill_ai_rank"]<=6].copy()
    va=valid[valid["distill_ai_rank"]<=6].copy()
    xtr=tr[cols].apply(pd.to_numeric,errors="coerce").astype("float32")
    xva=va[cols].apply(pd.to_numeric,errors="coerce").astype("float32")
    y=tr["_is_top3"].astype(int).to_numpy()
    pms=risk_params(seed)
    pms["n_estimators"]=180
    pms["num_leaves"]=15
    m=lgb.LGBMClassifier(objective="binary",**pms)
    m.fit(xtr,y)
    pred=np.asarray(m.predict_proba(xva)[:,1],dtype=float)
    del m,xtr,xva; gc.collect()
    va=va.copy(); va["meta_p_top3"]=pred
    va=va.sort_values(["_race_id","meta_p_top3","distill_ai_rank","_horse_id"],
                      ascending=[True,False,True,True])
    picks=va.groupby("_race_id",sort=False).head(1)
    return {
        "auc":auc_rank(va["_is_top3"].to_numpy(),pred),
        "logloss":logloss(va["_is_top3"].to_numpy(),pred),
        "top1_top3_pct":100*float(picks["_is_top3"].mean()),
        "top1_win_pct":100*float(picks["_is_win"].mean()),
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
        "contract":"L1_MARKET_CORRECTION_DEEPDIVE_V1_RUNTIME",
        "goal":"turn market-correction risk into calibrated diagnostics and strict OOS decision policies",
        "teacher_market":"FINAL popularity is historical teacher only",
        "student_market_inputs":False,
        "heavy_model_optimization":"only the OVER3 risk head is trained; prior UP_ANY/gap heads are intentionally skipped",
        "parallelization":"snapshot downloads/projections are parallel; LightGBM uses standard-runner CPU threads internally",
        "primary_variant":PRIMARY_VARIANT,
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
        feat=attach_state(feat,neg_by_year[test][["_race_id","_horse_id",*neg_cols]],neg_cols)
        oos[test]=feat
        ability_checks.append({
            "test_year":test,"races":int(ctx["_race_id"].nunique()),"top1_top3_pct":got,
            "market_coverage_pct":100*float(ctx["market_rank"].notna().mean()),
        })
        print(f"DEEP_TEACHER_READY year={test} rows={len(feat)} top1_top3={got:.6f}",flush=True)
        del train_base,valid_base,btr,bva,rtr,rva,mtr,mva,tr,va,pred,ctx
        gc.collect()

    # Strict expanding-year correction risk predictions.
    risk_preds={v:{} for v in RISK_VARIANTS}
    imp_rows=[]
    risk_metrics=[]
    for test in CORR_YEARS:
        train_years=tuple(y for y in TEST_YEARS if y<test)
        train=pd.concat([oos[y] for y in train_years],ignore_index=True,copy=False)
        valid=oos[test].copy()
        for variant in RISK_VARIANTS:
            if variant=="RANK_ONLY":
                p,imp=fit_risk_rank_only(train,valid,8100000+test*100)
            elif variant=="FULL_PRERACE":
                cols=[c for c in valid.columns if not c.startswith("neg_")]
                p,imp=fit_risk_full(train[cols],valid[cols],8200000+test*100)
            else:
                p,imp=fit_risk_full(train,valid,8300000+test*100)
            scored=valid[["_race_id","_horse_id","_finish","_is_win","_is_top3","distill_ai_rank",
                          "distill_ai_rank01","distill_score_z","distill_score_gap_to_best",
                          "market_rank","final_win_odds","_target_gap","_target_over3"]].copy()
            scored["risk"]=p
            risk_preds[variant][test]=scored
            scope=scored[scored["distill_ai_rank"]<=6]
            y=scope["_target_over3"].astype(int).to_numpy()
            pp=scope["risk"].to_numpy(dtype=float)
            risk_metrics.append({
                "test_year":test,"train_teacher_years":"|".join(map(str,train_years)),
                "variant":variant,"n":len(scope),"prevalence_pct":100*float(y.mean()),
                "roc_auc":auc_rank(y,pp),"average_precision":average_precision(y,pp),
                "logloss":logloss(y,pp),"brier":brier(y,pp),
            })
            total_gain=sum(float(g) for _,g in imp) or 1.0
            for rank,(feature,gain) in enumerate(imp[:120],start=1):
                imp_rows.append({
                    "test_year":test,"variant":variant,"importance_rank":rank,
                    "feature":feature,"gain":float(gain),"gain_share":float(gain)/total_gain,
                    "domain":feature_domain(feature),
                })
            print(f"RISK_READY year={test} variant={variant} auc={risk_metrics[-1]['roc_auc']:.4f}",flush=True)
        del train,valid; gc.collect()

    # Deep descriptive tables for the primary risk head.
    rank_bucket_rows=[]
    top1_decile_rows=[]
    calibration_rows=[]
    extreme_rows=[]
    for year in CORR_YEARS:
        x=risk_preds[PRIMARY_VARIANT][year].copy()
        top6=x[x["distill_ai_rank"]<=6].copy()
        top6["risk_decile"]=quantile_bucket(top6["risk"],10)
        for rank in range(1,7):
            z=top6[top6["distill_ai_rank"]==rank].copy()
            if z.empty: continue
            z["risk_quintile"]=quantile_bucket(z["risk"],5)
            for q,g in z.groupby("risk_quintile"):
                rank_bucket_rows.append({
                    "test_year":year,"ai_rank":rank,"risk_quintile":int(q),"n":len(g),
                    "mean_risk":float(g["risk"].mean()),
                    "market_over3_pct":100*float(g["_target_over3"].mean()),
                    "actual_top3_pct":100*float(g["_is_top3"].mean()),
                    "actual_win_pct":100*float(g["_is_win"].mean()),
                    "mean_market_rank":float(g["market_rank"].mean()),
                    "mean_market_gap":float(g["_target_gap"].mean()),
                    "mean_final_odds":float(pd.to_numeric(g["final_win_odds"],errors="coerce").mean()),
                })
        t=x[x["distill_ai_rank"]==1].copy()
        t["risk_decile"]=quantile_bucket(t["risk"],10)
        for d,g in t.groupby("risk_decile"):
            lo,hi=wilson(int(g["_is_top3"].sum()),len(g))
            top1_decile_rows.append({
                "test_year":year,"risk_decile":int(d),"n":len(g),
                "mean_risk":float(g["risk"].mean()),
                "market_over3_pct":100*float(g["_target_over3"].mean()),
                "actual_top3_pct":100*float(g["_is_top3"].mean()),"top3_ci95_lo":lo,"top3_ci95_hi":hi,
                "actual_win_pct":100*float(g["_is_win"].mean()),
                "mean_market_rank":float(g["market_rank"].mean()),
                "mean_market_gap":float(g["_target_gap"].mean()),
            })
        for d,g in top6.groupby("risk_decile"):
            calibration_rows.append({
                "test_year":year,"risk_decile":int(d),"n":len(g),
                "mean_predicted_risk":100*float(g["risk"].mean()),
                "actual_over3_pct":100*float(g["_target_over3"].mean()),
                "calibration_error_pp":100*float(g["risk"].mean()-g["_target_over3"].mean()),
            })
        high=t[t["risk"]>=t["risk"].quantile(0.90)]
        low=t[t["risk"]<=t["risk"].quantile(0.10)]
        extreme_rows.append({
            "test_year":year,
            "high_risk_n":len(high),"high_risk_top3_pct":100*float(high["_is_top3"].mean()),
            "high_risk_market_over3_pct":100*float(high["_target_over3"].mean()),
            "low_risk_n":len(low),"low_risk_top3_pct":100*float(low["_is_top3"].mean()),
            "low_risk_market_over3_pct":100*float(low["_target_over3"].mean()),
            "top3_gap_low_minus_high_pp":100*float(low["_is_top3"].mean()-high["_is_top3"].mean()),
        })

    # Strict OOS veto policy search: tune on prior correction-risk years, apply to next year.
    policy_search_rows=[]
    strict_policy_rows=[]
    for test in PRIMARY_YEARS:
        hist_years=[y for y in CORR_YEARS if y<test]
        hist=pd.concat([risk_preds[PRIMARY_VARIANT][y] for y in hist_years],ignore_index=True)
        va=risk_preds[PRIMARY_VARIANT][test].copy()
        baseline=100*float(va.loc[va["distill_ai_rank"]==1,"_is_top3"].mean())
        candidates=[]
        for q in QGRID:
            threshold=float(hist.loc[hist["distill_ai_rank"]==1,"risk"].quantile(q))
            for mode in RESCUE_MODES:
                score=policy_score(hist,threshold,mode)
                row={
                    "test_year":test,"train_policy_years":"|".join(map(str,hist_years)),
                    "quantile":q,**score,
                }
                policy_search_rows.append(row); candidates.append(row)
        # Conservative tie-break: maximize top3, then fewer replacements, then higher quantile.
        best=max(candidates,key=lambda r:(r["top3_pct"],-r["replacement_rate_pct"],r["quantile"]))
        test_score=policy_score(va,best["threshold"],best["mode"])
        strict_policy_rows.append({
            "test_year":test,"train_policy_years":"|".join(map(str,hist_years)),
            "selected_quantile":best["quantile"],"selected_threshold":best["threshold"],
            "selected_mode":best["mode"],"baseline_top1_top3_pct":baseline,
            **{f"test_{k}":v for k,v in test_score.items()},
            "uplift_pp":test_score["top3_pct"]-baseline,
        })

    # Risk tier tagging using thresholds from prior years only.
    trust_rows=[]
    for test in PRIMARY_YEARS:
        hist_years=[y for y in CORR_YEARS if y<test]
        hist=pd.concat([risk_preds[PRIMARY_VARIANT][y] for y in hist_years],ignore_index=True)
        t_hist=hist[hist["distill_ai_rank"]==1]
        q50=float(t_hist["risk"].quantile(0.50))
        q80=float(t_hist["risk"].quantile(0.80))
        va=risk_preds[PRIMARY_VARIANT][test]
        top=va[va["distill_ai_rank"]==1].copy()
        top["trust_tier"]=np.where(top["risk"]<q50,"HIGH",np.where(top["risk"]<q80,"MID","LOW"))
        for tier,g in top.groupby("trust_tier"):
            lo,hi=wilson(int(g["_is_top3"].sum()),len(g))
            trust_rows.append({
                "test_year":test,"train_threshold_years":"|".join(map(str,hist_years)),
                "tier":tier,"n":len(g),"risk_mean":float(g["risk"].mean()),
                "actual_top3_pct":100*float(g["_is_top3"].mean()),"top3_ci95_lo":lo,"top3_ci95_hi":hi,
                "actual_win_pct":100*float(g["_is_win"].mean()),
                "market_over3_pct":100*float(g["_target_over3"].mean()),
                "mean_market_rank":float(g["market_rank"].mean()),
            })

    # Does risk improve an actual-outcome meta model beyond ability geometry?
    meta_rows=[]
    for test in PRIMARY_YEARS:
        hist_years=[y for y in CORR_YEARS if y<test]
        hist=pd.concat([risk_preds[PRIMARY_VARIANT][y] for y in hist_years],ignore_index=True)
        va=risk_preds[PRIMARY_VARIANT][test].copy()
        for use_risk in (False,True):
            met=fit_meta(hist,va,use_risk,9100000+test*100+(1 if use_risk else 0))
            meta_rows.append({
                "test_year":test,"train_years":"|".join(map(str,hist_years)),
                "variant":"ABILITY_GEOM_PLUS_RISK" if use_risk else "ABILITY_GEOM_ONLY",
                **met,
            })

    # Feature stability/domain summary across primary years.
    stability=[]
    domain_rows=[]
    fullimp=[r for r in imp_rows if r["variant"]==PRIMARY_VARIANT and r["test_year"] in PRIMARY_YEARS]
    byf=defaultdict(list)
    byd=defaultdict(list)
    for r in fullimp:
        byf[r["feature"]].append(r)
        byd[(r["test_year"],r["domain"])].append(r)
    for f,vals in byf.items():
        stability.append({
            "feature":f,"domain":feature_domain(f),"years_present_top120":len({v["test_year"] for v in vals}),
            "mean_rank":float(np.mean([v["importance_rank"] for v in vals])),
            "mean_gain_share":float(np.mean([v["gain_share"] for v in vals])),
            "max_gain_share":float(np.max([v["gain_share"] for v in vals])),
        })
    stability.sort(key=lambda r:(-r["years_present_top120"],-r["mean_gain_share"],r["mean_rank"]))
    for (year,domain),vals in byd.items():
        domain_rows.append({
            "test_year":year,"domain":domain,"features_in_top120":len(vals),
            "gain_share_top120_sum":float(sum(v["gain_share"] for v in vals)),
        })

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"ability-checks.csv",ability_checks)
    write_csv(out/"risk-metrics.csv",risk_metrics)
    write_csv(out/"risk-by-ai-rank.csv",rank_bucket_rows)
    write_csv(out/"top1-risk-deciles.csv",top1_decile_rows)
    write_csv(out/"calibration.csv",calibration_rows)
    write_csv(out/"risk-extremes.csv",extreme_rows)
    write_csv(out/"policy-search.csv",policy_search_rows)
    write_csv(out/"strict-veto-policy.csv",strict_policy_rows)
    write_csv(out/"trust-tiers.csv",trust_rows)
    write_csv(out/"outcome-meta.csv",meta_rows)
    write_csv(out/"feature-importance.csv",imp_rows)
    write_csv(out/"feature-stability.csv",stability)
    write_csv(out/"feature-domains.csv",domain_rows)
    write_csv(out/"negative-evidence-coverage.csv",neg_cov)
    write_csv(out/"state-coverage.csv",state_cov)

    primary_metric=[r for r in risk_metrics if r["variant"]==PRIMARY_VARIANT and r["test_year"] in PRIMARY_YEARS]
    primary_auc=float(np.mean([r["roc_auc"] for r in primary_metric]))
    primary_ap=float(np.mean([r["average_precision"] for r in primary_metric]))
    veto_uplifts=[r["uplift_pp"] for r in strict_policy_rows]
    meta_base=[r for r in meta_rows if r["variant"]=="ABILITY_GEOM_ONLY"]
    meta_risk=[r for r in meta_rows if r["variant"]=="ABILITY_GEOM_PLUS_RISK"]
    summary={
        "contract":"L1_MARKET_CORRECTION_DEEPDIVE_V1",
        "primary_variant":PRIMARY_VARIANT,
        "primary_years":list(PRIMARY_YEARS),
        "market_is_teacher_only":True,
        "model_uses_current_market":False,
        "risk_head":{
            "mean_auc_2023_2025":primary_auc,
            "mean_average_precision_2023_2025":primary_ap,
        },
        "risk_extremes":extreme_rows,
        "strict_veto_policy":{
            "rows":strict_policy_rows,
            "mean_uplift_pp":float(np.mean(veto_uplifts)),
            "positive_years":int(sum(x>0 for x in veto_uplifts)),
        },
        "outcome_meta":{
            "ability_geom_mean_top1_top3_pct":float(np.mean([r["top1_top3_pct"] for r in meta_base])),
            "ability_geom_plus_risk_mean_top1_top3_pct":float(np.mean([r["top1_top3_pct"] for r in meta_risk])),
            "uplift_pp":float(np.mean([r["top1_top3_pct"] for r in meta_risk])-np.mean([r["top1_top3_pct"] for r in meta_base])),
            "ability_geom_mean_logloss":float(np.mean([r["logloss"] for r in meta_base])),
            "ability_geom_plus_risk_mean_logloss":float(np.mean([r["logloss"] for r in meta_risk])),
        },
        "top_stable_features":stability[:30],
        "decision_rule":"Risk becomes an L1 confidence head only if tiers are stable OOS and either strict veto or outcome-meta adds reproducible actual-result value.",
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== STRICT VETO =====")
    print((out/"strict-veto-policy.csv").read_text())
    print("===== TRUST TIERS =====")
    print((out/"trust-tiers.csv").read_text())
    print("===== OUTCOME META =====")
    print((out/"outcome-meta.csv").read_text())
    print("L1_MARKET_CORRECTION_DEEPDIVE_V1_COMPLETE")

if __name__=="__main__":
    main()
