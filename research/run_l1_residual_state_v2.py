#!/usr/bin/env python3
import argparse,csv,gc,json,math,os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l1_objective_rebuild_v1 import (
    YEARS,TEST_YEARS,LOCKED_YEAR,THREADS,META,
    read_year,select_variant,prepare,model_common,metrics,
)
from run_l1_ceiling_audit_v1 import strip_forbidden
from run_l1_state_transition_v1 import (
    MAX_LAG,read_compact_state_source,build_state_features,attach_state,fit_heads,
)

VARIANTS=(
    "BASELINE",
    "RESIDUAL_V1",
    "V2_FINISH_RESIDUAL",
    "V2_MULTI_RESIDUAL",
    "HYBRID_V1_V2_MULTI",
)
HEADS=("TOP3_BINARY","RANK_GRADED","BLEND_TOP3_RANK")
V2_METRICS=("v2_finish_resid","v2_speed_resid","v2_last3f_resid")

def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def outcome_targets(paths):
    by_year={}
    compact_by_year={}
    for year in YEARS:
        s=read_compact_state_source(paths[year],year)
        race_n=s.groupby("_race_id")["_horse_id"].transform("size").astype(float)
        denom=(race_n-1.0).clip(lower=1.0)
        s["finish_pct_target"]=(1.0-(s["_finish"].astype(float)-1.0)/denom).clip(0.0,1.0)

        sec=s["finish_time_ms"].astype(float)/1000.0
        s["speed_mps_target"]=np.where(
            np.isfinite(s["distance_m"]) & np.isfinite(sec) & (sec>0),
            s["distance_m"].astype(float)/sec,
            np.nan,
        )
        s["last3f_speed_target"]=np.where(
            np.isfinite(s["last_3f"]) & (s["last_3f"].astype(float)>0),
            600.0/s["last_3f"].astype(float),
            np.nan,
        )
        s["speed_rel_target"]=(
            s["speed_mps_target"]-s.groupby("_race_id")["speed_mps_target"].transform("median")
        )
        s["last3f_rel_target"]=(
            s["last3f_speed_target"]-s.groupby("_race_id")["last3f_speed_target"].transform("median")
        )
        cols=[
            "_race_id","_horse_id","_race_date","_year",
            "finish_pct_target","speed_rel_target","last3f_rel_target",
        ]
        by_year[year]=s[cols].copy()
        compact_by_year[year]=s
        print(
            f"V2_TARGET_READY year={year} rows={len(s)} "
            f"speed_cov={100*s['speed_rel_target'].notna().mean():.2f} "
            f"last3f_cov={100*s['last3f_rel_target'].notna().mean():.2f}",
            flush=True,
        )
    return by_year,compact_by_year

def expected_model_frame(paths):
    cache={}
    def get(year):
        if year not in cache:
            raw=strip_forbidden(read_year(paths[year],year))
            cache[year]=select_variant(raw,"structural")
            del raw
            gc.collect()
        return cache[year]
    return cache,get

def reg_params(seed):
    p=model_common(seed)
    p["n_estimators"]=300
    p["learning_rate"]=0.035
    p["num_leaves"]=31
    p["min_child_samples"]=100
    p["reg_lambda"]=8.0
    p["reg_alpha"]=0.8
    return p

def fit_regression_expectation(train,valid,target,seed):
    model_cols=[c for c in train.columns if c not in {
        "finish_pct_target","speed_rel_target","last3f_rel_target"
    }]
    tr=train[model_cols]
    va=valid[model_cols]
    xtr,xva,cats=prepare(tr,va)
    y=pd.to_numeric(train[target],errors="coerce")
    mask=y.notna()
    if int(mask.sum())<1000:
        raise SystemExit(f"not enough expectation labels target={target} n={int(mask.sum())}")
    model=lgb.LGBMRegressor(objective="regression_l2",**reg_params(seed))
    model.fit(xtr.loc[mask],y.loc[mask].to_numpy(),categorical_feature=cats)
    pred=np.asarray(model.predict(xva),dtype=float)
    del model,xtr,xva
    gc.collect()
    return pred

def expectation_predictions(paths,target_by_year):
    cache,get=expected_model_frame(paths)
    preds={}
    diagnostics=[]
    targets=("finish_pct_target","speed_rel_target","last3f_rel_target")
    pred_names={
        "finish_pct_target":"expected_finish_pct",
        "speed_rel_target":"expected_speed_rel",
        "last3f_rel_target":"expected_last3f_rel",
    }

    # 2019 has no earlier snapshot year, so V2 residual is deliberately unavailable.
    base2019=get(2019)
    preds[2019]=base2019[["_race_id","_horse_id","_race_date"]].copy()
    for name in pred_names.values():
        preds[2019][name]=np.nan

    for year in range(2020,2026):
        train_years=(year-1,) if year==2020 else (year-2,year-1)
        train_x=pd.concat([get(y) for y in train_years],ignore_index=True,copy=False)
        valid_x=get(year).copy()
        train_t=pd.concat([target_by_year[y] for y in train_years],ignore_index=True,copy=False)
        valid_t=target_by_year[year]

        train=train_x.merge(
            train_t[["_race_id","_horse_id",*targets]],
            on=["_race_id","_horse_id"],how="left",validate="one_to_one",
        )
        valid=valid_x.merge(
            valid_t[["_race_id","_horse_id",*targets]],
            on=["_race_id","_horse_id"],how="left",validate="one_to_one",
        )
        out=valid[["_race_id","_horse_id","_race_date"]].copy()

        for ti,target in enumerate(targets):
            p=fit_regression_expectation(train,valid,target,1800000+year*100+ti*10)
            out[pred_names[target]]=p
            y=pd.to_numeric(valid[target],errors="coerce").to_numpy(dtype=float)
            mask=np.isfinite(y)&np.isfinite(p)
            rmse=float(np.sqrt(np.mean((y[mask]-p[mask])**2))) if mask.any() else float("nan")
            corr=float(np.corrcoef(y[mask],p[mask])[0,1]) if mask.sum()>2 else float("nan")
            diagnostics.append({
                "prediction_year":year,
                "train_years":"|".join(map(str,train_years)),
                "target":target,
                "rows":int(mask.sum()),
                "rmse":rmse,
                "corr":corr,
            })
        preds[year]=out
        print(
            f"EXPECTATION_OOS_READY year={year} train_years={'|'.join(map(str,train_years))}",
            flush=True,
        )
        del train_x,valid_x,train_t,valid_t,train,out
        for old in list(cache):
            if old < year-2:
                del cache[old]
        gc.collect()
    return preds,diagnostics

def residual_state_features(compact_by_year,target_by_year,preds):
    rows=[]
    for year in YEARS:
        target=target_by_year[year].copy()
        pred=preds[year]
        x=target.merge(
            pred[["_race_id","_horse_id","expected_finish_pct","expected_speed_rel","expected_last3f_rel"]],
            on=["_race_id","_horse_id"],how="left",validate="one_to_one",
        )
        x["v2_finish_resid"]=(x["finish_pct_target"]-x["expected_finish_pct"]).astype("float32")
        x["v2_speed_resid"]=(x["speed_rel_target"]-x["expected_speed_rel"]).astype("float32")
        x["v2_last3f_resid"]=(x["last3f_rel_target"]-x["expected_last3f_rel"]).astype("float32")
        rows.append(x[[
            "_race_id","_horse_id","_race_date","_year",*V2_METRICS
        ]])
    s=pd.concat(rows,ignore_index=True,copy=False)
    del rows

    # Same horse/date duplicates are never ordered within the day. For V2 expectations,
    # duplicated race contexts may yield slightly different OOS expectations, so use their
    # deterministic mean only for future dates. Nothing from the day is visible to itself.
    event=(
        s.groupby(["_horse_id","_race_date"],as_index=False,sort=False)[list(V2_METRICS)]
         .mean()
         .sort_values(["_horse_id","_race_date"])
         .reset_index(drop=True)
    )
    g=event.groupby("_horse_id",sort=False)
    state=event[["_horse_id","_race_date"]].copy()

    feature_map={}
    for metric in V2_METRICS:
        cols=[]
        for k in range(1,MAX_LAG+1):
            c=f"seq_{metric}_lag{k}"
            state[c]=pd.to_numeric(g[metric].shift(k),errors="coerce").astype("float32")
            cols.append(c)
        l1=state[f"seq_{metric}_lag1"]
        l2=state[f"seq_{metric}_lag2"]
        l3=state[f"seq_{metric}_lag3"]
        d12=f"seq_{metric}_delta12"
        d23=f"seq_{metric}_delta23"
        accel=f"seq_{metric}_accel"
        reversal=f"seq_{metric}_reversal"
        pos3=f"seq_{metric}_positive_count3"
        std3=f"seq_{metric}_std3"
        mean3=f"seq_{metric}_mean3"
        state[d12]=(l1-l2).astype("float32")
        state[d23]=(l2-l3).astype("float32")
        state[accel]=(state[d12]-state[d23]).astype("float32")
        valid=l1.notna()&l2.notna()&l3.notna()
        rev=np.where(valid & ((state[d12]*state[d23])<0),1.0,np.where(valid,0.0,np.nan))
        state[reversal]=pd.Series(rev,index=state.index,dtype="float32")
        trio=pd.concat([l1,l2,l3],axis=1)
        state[pos3]=trio.gt(0).sum(axis=1).where(valid,np.nan).astype("float32")
        state[std3]=trio.std(axis=1,ddof=0).astype("float32")
        state[mean3]=trio.mean(axis=1).astype("float32")
        cols.extend([d12,d23,accel,reversal,pos3,std3,mean3])
        feature_map[metric]=cols

    out=(
        s[["_race_id","_horse_id","_race_date","_year"]]
        .merge(state,on=["_horse_id","_race_date"],how="left",validate="many_to_one")
    )
    by_year={}
    coverage=[]
    all_cols=sorted({c for cols in feature_map.values() for c in cols})
    for year in YEARS:
        y=out[out["_year"]==year][["_race_id","_horse_id",*all_cols]].copy()
        if y.duplicated(["_race_id","_horse_id"]).any():
            raise SystemExit(f"duplicate V2 state key year={year}")
        by_year[year]=y
        coverage.append({
            "year":year,
            "rows":len(y),
            "v2_finish_lag1_coverage_pct":100*float(y["seq_v2_finish_resid_lag1"].notna().mean()),
            "v2_finish_lag3_coverage_pct":100*float(y["seq_v2_finish_resid_lag3"].notna().mean()),
            "v2_finish_lag5_coverage_pct":100*float(y["seq_v2_finish_resid_lag5"].notna().mean()),
            "v2_speed_lag1_coverage_pct":100*float(y["seq_v2_speed_resid_lag1"].notna().mean()),
            "v2_last3f_lag1_coverage_pct":100*float(y["seq_v2_last3f_resid_lag1"].notna().mean()),
        })
    del s,event,state,out
    gc.collect()
    return by_year,feature_map,coverage

def variant_columns(variant,v1_lag,v1_transition,v1_residual,v2_map):
    base_state=sorted(set(v1_lag+v1_transition))
    v1=sorted(set(base_state+v1_residual))
    finish=sorted(set(base_state+v2_map["v2_finish_resid"]))
    multi=sorted(set(base_state+sum((v2_map[m] for m in V2_METRICS),[])))
    hybrid=sorted(set(v1+sum((v2_map[m] for m in V2_METRICS),[])))
    return {
        "BASELINE":[],
        "RESIDUAL_V1":v1,
        "V2_FINISH_RESIDUAL":finish,
        "V2_MULTI_RESIDUAL":multi,
        "HYBRID_V1_V2_MULTI":hybrid,
    }[variant]

def pooled(rows):
    out=[]
    for variant in VARIANTS:
        for head in HEADS:
            vals=[r for r in rows if r["variant"]==variant and r["head"]==head]
            w=np.array([r["races"] for r in vals],dtype=float)
            rec={"variant":variant,"head":head,"folds":len(vals),"races":int(w.sum())}
            for key in (
                "top1_win_pct","top1_top3_pct","winner_top3_capture_pct","winner_top6_capture_pct",
                "top3_podium_precision_pct","top6_podium_precision_pct","mean_winner_rank","winner_mrr",
            ):
                arr=np.array([r[key] for r in vals],dtype=float)
                rec[key]=float(np.average(arr,weights=w))
                rec[key+"_worst"]=float(arr.max() if key=="mean_winner_rank" else arr.min())
                rec[key+"_std"]=float(arr.std(ddof=0))
            out.append(rec)
    return out

def add_uplift(pooled_rows,fold_rows):
    base={r["head"]:r for r in pooled_rows if r["variant"]=="BASELINE"}
    fold_base={(r["test_year"],r["head"]):r for r in fold_rows if r["variant"]=="BASELINE"}
    for r in pooled_rows:
        b=base[r["head"]]
        r["top1_top3_uplift_pp"]=r["top1_top3_pct"]-b["top1_top3_pct"]
        diffs=[]
        for fr in fold_rows:
            if fr["variant"]==r["variant"] and fr["head"]==r["head"]:
                diffs.append(
                    fr["top1_top3_pct"]-fold_base[(fr["test_year"],fr["head"])]["top1_top3_pct"]
                )
        r["top1_top3_uplift_worst_fold_pp"]=float(min(diffs)) if diffs else 0.0
        r["top1_top3_uplift_best_fold_pp"]=float(max(diffs)) if diffs else 0.0
    return pooled_rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--year-file",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--out-dir",required=True)
    a=p.parse_args()

    paths={}
    for spec in a.year_file:
        y,path=spec.split(":",1)
        paths[int(y)]=path
    if set(paths)!=set(YEARS):
        raise SystemExit(f"year file mismatch {sorted(paths)}")
    if LOCKED_YEAR in paths:
        raise SystemExit("2026 sealed")

    print(json.dumps({
        "contract":"L1_RESIDUAL_STATE_V2_RUNTIME",
        "variants":list(VARIANTS),
        "heads":list(HEADS),
        "expectation_horizon":"previous one year for 2020; previous two years for 2021-2025",
        "expectation_features":"STRUCTURAL pre-race L1 only",
        "threads":THREADS,
        "ability_uses_odds":False,
        "2026_locked":True,
    },separators=(",",":")),flush=True)

    target_by_year,compact_by_year=outcome_targets(paths)
    expectation,expectation_diag=expectation_predictions(paths,target_by_year)
    v2_by_year,v2_map,v2_coverage=residual_state_features(compact_by_year,target_by_year,expectation)

    v1_by_year,v1_lag,v1_transition,v1_residual,v1_coverage=build_state_features(paths)

    base_cache={}
    def get_base(year):
        if year not in base_cache:
            raw=strip_forbidden(read_year(paths[year],year))
            base_cache[year]=select_variant(raw,"core4")
            del raw
            gc.collect()
        return base_cache[year]

    rows=[]
    feature_counts=[]
    for test in TEST_YEARS:
        train_years=(test-2,test-1)
        train_base=pd.concat([get_base(y) for y in train_years],ignore_index=True,copy=False)
        valid_base=get_base(test)

        v1_train=pd.concat([v1_by_year[y] for y in train_years],ignore_index=True,copy=False)
        v1_valid=v1_by_year[test]
        v2_train=pd.concat([v2_by_year[y] for y in train_years],ignore_index=True,copy=False)
        v2_valid=v2_by_year[test]

        full_train=v1_train.merge(v2_train,on=["_race_id","_horse_id"],how="left",validate="one_to_one")
        full_valid=v1_valid.merge(v2_valid,on=["_race_id","_horse_id"],how="left",validate="one_to_one")

        for vi,variant in enumerate(VARIANTS):
            state_cols=variant_columns(variant,v1_lag,v1_transition,v1_residual,v2_map)
            train=attach_state(train_base,full_train,state_cols)
            valid=attach_state(valid_base,full_valid,state_cols)
            feature_counts.append({
                "test_year":test,"variant":variant,
                "state_feature_count":len(state_cols),
                "total_model_columns":len([c for c in train.columns if c not in META]),
            })
            pred=fit_heads(train,valid,2100000+test*100+vi*10)
            for head in HEADS:
                rows.append({
                    "variant":variant,
                    "test_year":test,
                    "train_years":"|".join(map(str,train_years)),
                    "head":head,
                    "state_feature_count":len(state_cols),
                    **metrics(pred,head),
                })
            print(
                f"V2_FOLD_READY year={test} variant={variant} state_features={len(state_cols)}",
                flush=True,
            )
            del train,valid,pred
            gc.collect()

        del train_base,valid_base,v1_train,v1_valid,v2_train,v2_valid,full_train,full_valid
        for old in list(base_cache):
            if old < test-1:
                del base_cache[old]
        gc.collect()

    pooled_rows=add_uplift(pooled(rows),rows)
    nonbase=[r for r in pooled_rows if r["variant"]!="BASELINE"]
    best=max(nonbase,key=lambda r:(r["top1_top3_uplift_pp"],r["top1_top3_pct"]))

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"fold-metrics.csv",rows)
    write_csv(out/"pooled-metrics.csv",pooled_rows)
    write_csv(out/"expectation-diagnostics.csv",expectation_diag)
    write_csv(out/"v2-state-coverage.csv",v2_coverage)
    write_csv(out/"v1-state-coverage.csv",v1_coverage)
    write_csv(out/"feature-counts.csv",feature_counts)

    (out/"summary.json").write_text(json.dumps({
        "contract":"L1_RESIDUAL_STATE_V2",
        "question":"Can strictly OOS condition-adjusted expected-performance residual states break the L1 top1 podium ceiling?",
        "variants":{
            "BASELINE":"core4 direct static features",
            "RESIDUAL_V1":"V1 lag/state + simple network expectation residuals",
            "V2_FINISH_RESIDUAL":"V1 raw lag/state + OOS expected finish-percentile residual state",
            "V2_MULTI_RESIDUAL":"V1 raw lag/state + OOS finish/speed/last3f residual states",
            "HYBRID_V1_V2_MULTI":"V1 residual state + all V2 OOS residual states",
        },
        "v2_expectation":{
            "features":"STRUCTURAL pre-race L1 features only; no odds/popularity",
            "targets":["finish percentile","race-relative finish speed","race-relative last3f speed"],
            "training":"For race year Y, expectation models use only Y-1 for 2020 or Y-2,Y-1 for 2021-2025.",
            "2019":"No V2 expectation residual because no earlier snapshot year is available.",
            "same_day_duplicates":"Never ordered within day; duplicate OOS residuals are averaged only for future-date state updates.",
        },
        "strictness":[
            "Every V2 expectation is out-of-sample by calendar year.",
            "Every residual state feature is shifted by at least one horse race date.",
            "Current-race outcome never enters its own features.",
            "Final L1 walk-forward remains two prior years -> next unknown year, 2021-2025.",
            "No odds or popularity are used.",
            "2026 is sealed.",
        ],
        "best_nonbaseline":best,
        "promotion_rule":"Require >=2.0pp pooled top1_top3 uplift and no fold worse than -0.5pp.",
        "promotion":bool(
            best["top1_top3_uplift_pp"]>=2.0
            and best["top1_top3_uplift_worst_fold_pp"]>=-0.5
        ),
        "ability_uses_odds":False,
        "2026_locked":True,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== POOLED =====")
    print((out/"pooled-metrics.csv").read_text())
    print("===== EXPECTATION =====")
    print((out/"expectation-diagnostics.csv").read_text())
    print("===== V2 COVERAGE =====")
    print((out/"v2-state-coverage.csv").read_text())
    print("L1_RESIDUAL_STATE_V2_COMPLETE")

if __name__=="__main__":
    main()
