#!/usr/bin/env python3
import argparse,csv,gc,json,math,os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l1_objective_rebuild_v1 import (
    YEARS,TEST_YEARS,LOCKED_YEAR,THREADS,META,
    read_year,select_variant,prepare,model_common,
)
from run_l1_ceiling_audit_v1 import strip_forbidden
from run_l1_state_transition_v1 import build_state_features,attach_state
from run_l1_residual_regime_full_pairwise_v1 import (
    build_regime_features,fit_direct_rank,top1_metrics,
)

AXES=("perf_resid","speed_resid","last3f_resid")
DIRECT_VARIANTS=(
    "BASELINE_DIRECT",
    "REGIME_ALL",
    "REGIME_PERF_ONLY",
    "REGIME_SPEED_ONLY",
    "REGIME_LAST3F_ONLY",
    "REGIME_PERF_SPEED",
    "REGIME_FIELD_REL",
)
CORRECTION_VARIANTS=(
    "CORRECTION_ONLY",
    "CORRECTION_BLEND_25",
    "CORRECTION_BLEND_50",
    "CORRECTION_BLEND_75",
)
PRIMARY_KEYS=(
    ("REGIME_ALL","RANK_GRADED"),
    ("REGIME_FIELD_REL","RANK_GRADED"),
    ("CORRECTION_BLEND_50","TOP3_CORRECTION"),
)

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

def select_axis_cols(regime_cols,axes):
    prefixes=tuple(f"regime_{axis}_" for axis in axes)
    return [c for c in regime_cols if c.startswith(prefixes)]

def field_relative_source_cols(regime_cols):
    wanted=(
        "_lag1","_mean3","_mean5","_slope3","_slope5","_accel",
        "_shock1","_strength3","_persistent_positive3","_persistent_negative3",
        "_uptrend3","_downtrend3","_rebound","_fade",
    )
    return [c for c in regime_cols if c.endswith(wanted)]

def add_field_relative(regime_by_year,regime_cols):
    source=field_relative_source_cols(regime_cols)
    by_year={}
    added=[]
    for year in YEARS:
        src=regime_by_year[year]
        out=src.copy()
        grp=out.groupby("_race_id",sort=False)
        for c in source:
            v=pd.to_numeric(out[c],errors="coerce")
            mean=grp[c].transform(lambda s: pd.to_numeric(s,errors="coerce").mean())
            std=grp[c].transform(lambda s: pd.to_numeric(s,errors="coerce").std(ddof=0))
            rank=grp[c].rank(method="average",ascending=False)
            n=grp[c].transform("count").astype(float)
            z=f"fieldrel_{c}_z"
            pct=f"fieldrel_{c}_pct"
            gap=f"fieldrel_{c}_gap_mean"
            out[z]=((v-mean)/std.replace(0,np.nan)).astype("float32")
            out[pct]=(1.0-(rank-1.0)/(n-1.0).clip(lower=1.0)).astype("float32")
            out[gap]=(v-mean).astype("float32")
            for name in (z,pct,gap):
                if name not in added:
                    added.append(name)
        by_year[year]=out
        print(
            f"FIELD_REL_READY year={year} rows={len(out)} source={len(source)} added={len(added)}",
            flush=True,
        )
    return by_year,added

def rank_normalize(frame,score):
    x=frame[["_race_id","_horse_id"]].copy()
    x["_base_score"]=np.asarray(score,dtype=float)
    grp=x.groupby("_race_id",sort=False)
    rank=grp["_base_score"].rank(method="average",ascending=False)
    n=grp["_base_score"].transform("size").astype(float)
    mean=grp["_base_score"].transform("mean")
    std=grp["_base_score"].transform(lambda s:s.std(ddof=0))
    mx=grp["_base_score"].transform("max")
    x["base_rank_pct"]=(1.0-(rank-1.0)/(n-1.0).clip(lower=1.0)).astype("float32")
    x["base_rank_norm"]=(rank/n).astype("float32")
    x["base_score_z"]=((x["_base_score"]-mean)/std.replace(0,np.nan)).astype("float32")
    x["base_gap_to_max"]=(x["_base_score"]-mx).astype("float32")
    x["base_gap_to_mean"]=(x["_base_score"]-mean).astype("float32")
    return x

def fit_base_rank_score(train,valid,seed):
    pred,_=fit_direct_rank(train,valid,seed)
    return pred

def correction_params(seed):
    return dict(
        objective="binary",
        n_estimators=280,
        learning_rate=0.03,
        num_leaves=23,
        min_child_samples=80,
        subsample=0.90,
        colsample_bytree=0.80,
        reg_lambda=10.0,
        reg_alpha=1.0,
        random_state=seed,
        n_jobs=THREADS,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )

def build_correction_frame(base_frame,regime_frame,base_pred,feature_cols):
    score=rank_normalize(base_pred,base_pred["_score"].to_numpy())
    cols=["_race_id","_horse_id",*feature_cols]
    out=base_frame[list(META)].merge(
        regime_frame[cols],
        on=["_race_id","_horse_id"],how="left",validate="one_to_one",
    )
    out=out.merge(
        score.drop(columns=["_base_score"]),
        on=["_race_id","_horse_id"],how="left",validate="one_to_one",
    )
    return out

def fit_correction(train_corr,test_corr,seed):
    meta=set(META)|{"_race_date"}
    cols=sorted(c for c in train_corr.columns if c not in meta)
    xtr=train_corr[cols].apply(pd.to_numeric,errors="coerce").astype("float32")
    xva=test_corr[cols].apply(pd.to_numeric,errors="coerce").astype("float32")
    y=train_corr["_is_top3"].astype(int).to_numpy()
    model=lgb.LGBMClassifier(**correction_params(seed))
    model.fit(xtr,y)
    p=np.asarray(model.predict_proba(xva)[:,1],dtype=float)
    gain=model.booster_.feature_importance(importance_type="gain")
    imp=sorted(zip(cols,gain),key=lambda z:z[1],reverse=True)
    out=test_corr[["_race_id","_horse_id","_finish","_is_win","_is_top3"]].copy()
    out["_corr"]=p
    del model,xtr,xva
    gc.collect()
    return out,imp

def blend_scores(base_pred,corr_pred,alpha):
    x=base_pred[["_race_id","_horse_id","_finish","_is_win","_is_top3","_score"]].copy()
    x=x.merge(
        corr_pred[["_race_id","_horse_id","_corr"]],
        on=["_race_id","_horse_id"],how="left",validate="one_to_one",
    )
    gb=x.groupby("_race_id",sort=False)
    br=gb["_score"].rank(method="average",ascending=False)
    cr=gb["_corr"].rank(method="average",ascending=False)
    n=gb["_score"].transform("size").astype(float)
    bs=1.0-(br-1.0)/(n-1.0).clip(lower=1.0)
    cs=1.0-(cr-1.0)/(n-1.0).clip(lower=1.0)
    x["_blend"]=(1-alpha)*bs+alpha*cs
    return x

def pooled(rows):
    out=[]
    keys=sorted({(r["architecture"],r["head"]) for r in rows})
    for arch,head in keys:
        vals=[r for r in rows if r["architecture"]==arch and r["head"]==head]
        w=np.array([r["races"] for r in vals],dtype=float)
        rec={"architecture":arch,"head":head,"folds":len(vals),"races":int(w.sum())}
        for key in (
            "top1_top3_pct","top1_win_pct","winner_top3_capture_pct","winner_top6_capture_pct",
        ):
            arr=np.array([r[key] for r in vals],dtype=float)
            rec[key]=float(np.average(arr,weights=w))
            rec[key+"_worst"]=float(arr.min())
            rec[key+"_std"]=float(arr.std(ddof=0))
        out.append(rec)
    return out

def add_uplift(pooled_rows,fold_rows):
    base=next(r for r in pooled_rows if r["architecture"]=="BASELINE_DIRECT" and r["head"]=="RANK_GRADED")
    fold_base={r["test_year"]:r for r in fold_rows if r["architecture"]=="BASELINE_DIRECT" and r["head"]=="RANK_GRADED"}
    for r in pooled_rows:
        r["top1_top3_uplift_pp"]=r["top1_top3_pct"]-base["top1_top3_pct"]
        diffs=[
            fr["top1_top3_pct"]-fold_base[fr["test_year"]]["top1_top3_pct"]
            for fr in fold_rows
            if fr["architecture"]==r["architecture"] and fr["head"]==r["head"]
        ]
        r["top1_top3_uplift_worst_fold_pp"]=float(min(diffs))
        r["top1_top3_uplift_best_fold_pp"]=float(max(diffs))
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
        "contract":"L1_RESIDUAL_REGIME_V2_RUNTIME",
        "direct_variants":list(DIRECT_VARIANTS),
        "correction_variants":list(CORRECTION_VARIANTS),
        "correction_training":"for test Y, baseline trained on Y-2 predicts Y-1 OOS; correction trains on that Y-1 OOS frame; final baseline trains Y-2,Y-1 and predicts Y",
        "threads":THREADS,
        "ability_uses_odds":False,
        "2026_locked":True,
    },separators=(",",":")),flush=True)

    v1_by_year,_,_,_,coverage=build_state_features(paths)
    regime_by_year,regime_cols=build_regime_features(v1_by_year)
    fieldrel_by_year,fieldrel_cols=add_field_relative(regime_by_year,regime_cols)

    variant_cols={
        "BASELINE_DIRECT":[],
        "REGIME_ALL":regime_cols,
        "REGIME_PERF_ONLY":select_axis_cols(regime_cols,("perf_resid",)),
        "REGIME_SPEED_ONLY":select_axis_cols(regime_cols,("speed_resid",)),
        "REGIME_LAST3F_ONLY":select_axis_cols(regime_cols,("last3f_resid",)),
        "REGIME_PERF_SPEED":select_axis_cols(regime_cols,("perf_resid","speed_resid")),
        "REGIME_FIELD_REL":regime_cols+fieldrel_cols,
    }

    base_cache={}
    def get_base(year):
        if year not in base_cache:
            raw=strip_forbidden(read_year(paths[year],year))
            base_cache[year]=select_variant(raw,"core4")
            del raw
            gc.collect()
        return base_cache[year]

    rows=[]
    importance=[]
    feature_counts=[]

    for test in TEST_YEARS:
        train_years=(test-2,test-1)
        train_base=pd.concat([get_base(y) for y in train_years],ignore_index=True,copy=False)
        valid_base=get_base(test).copy()
        train_reg=pd.concat([fieldrel_by_year[y] for y in train_years],ignore_index=True,copy=False)
        valid_reg=fieldrel_by_year[test]

        for vi,arch in enumerate(DIRECT_VARIANTS):
            cols=variant_cols[arch]
            tr=train_base if not cols else attach_state(train_base,train_reg,cols)
            va=valid_base if not cols else attach_state(valid_base,valid_reg,cols)
            pred,imp=fit_direct_rank(tr,va,3100000+test*100+vi*10)
            rows.append({
                "test_year":test,"train_years":"|".join(map(str,train_years)),
                "architecture":arch,"head":"RANK_GRADED","feature_count":len(cols),
                **top1_metrics(pred,"_score"),
            })
            feature_counts.append({
                "test_year":test,"architecture":arch,
                "added_feature_count":len(cols),
                "total_model_columns":len([c for c in tr.columns if c not in META]),
            })
            for rank,(feature,gain) in enumerate(imp[:60],start=1):
                importance.append({
                    "test_year":test,"architecture":arch,"head":"RANK_GRADED",
                    "importance_rank":rank,"feature":feature,"gain":float(gain),
                })
            if arch=="BASELINE_DIRECT":
                final_base_pred=pred.copy()
            del pred,tr,va
            gc.collect()

        # Strict two-stage correction.
        inner_train_year=test-2
        inner_valid_year=test-1
        inner_train=get_base(inner_train_year)
        inner_valid=get_base(inner_valid_year).copy()
        inner_base_pred=fit_base_rank_score(
            inner_train,inner_valid,3200000+test*100
        )

        corr_cols=regime_cols+fieldrel_cols
        corr_train=build_correction_frame(
            inner_valid,
            fieldrel_by_year[inner_valid_year],
            inner_base_pred,
            corr_cols,
        )
        corr_test=build_correction_frame(
            valid_base,
            fieldrel_by_year[test],
            final_base_pred,
            corr_cols,
        )
        corr_pred,corr_imp=fit_correction(corr_train,corr_test,3300000+test*100)

        xonly=corr_pred.copy()
        xonly["_score"]=xonly["_corr"]
        rows.append({
            "test_year":test,"train_years":str(inner_valid_year),
            "architecture":"CORRECTION_ONLY","head":"TOP3_CORRECTION",
            "feature_count":len(corr_cols)+5,
            **top1_metrics(xonly,"_score"),
        })

        for alpha,label in (
            (0.25,"CORRECTION_BLEND_25"),
            (0.50,"CORRECTION_BLEND_50"),
            (0.75,"CORRECTION_BLEND_75"),
        ):
            xb=blend_scores(final_base_pred,corr_pred,alpha)
            rows.append({
                "test_year":test,"train_years":str(inner_valid_year),
                "architecture":label,"head":"TOP3_CORRECTION",
                "feature_count":len(corr_cols)+5,
                **top1_metrics(xb,"_blend"),
            })

        for rank,(feature,gain) in enumerate(corr_imp[:80],start=1):
            importance.append({
                "test_year":test,"architecture":"CORRECTION_MODEL","head":"TOP3_CORRECTION",
                "importance_rank":rank,"feature":feature,"gain":float(gain),
            })
        feature_counts.append({
            "test_year":test,"architecture":"CORRECTION_MODEL",
            "added_feature_count":len(corr_cols)+5,
            "total_model_columns":len(corr_cols)+5,
        })
        print(
            f"RESIDUAL_V2_FOLD_READY year={test} direct_variants={len(DIRECT_VARIANTS)} "
            f"corr_features={len(corr_cols)+5}",
            flush=True,
        )

        del train_base,valid_base,train_reg,valid_reg,inner_train,inner_valid
        del inner_base_pred,corr_train,corr_test,corr_pred,final_base_pred
        gc.collect()
        for old in list(base_cache):
            if old < test-2:
                del base_cache[old]

    pooled_rows=add_uplift(pooled(rows),rows)
    primary=[r for r in pooled_rows if (r["architecture"],r["head"]) in PRIMARY_KEYS]
    best=max(primary,key=lambda r:(r["top1_top3_uplift_pp"],r["top1_top3_pct"]))

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"fold-metrics.csv",rows)
    write_csv(out/"pooled-metrics.csv",pooled_rows)
    write_csv(out/"feature-importance.csv",importance)
    write_csv(out/"feature-counts.csv",feature_counts)
    write_csv(out/"state-coverage.csv",coverage)

    (out/"summary.json").write_text(json.dumps({
        "contract":"L1_RESIDUAL_REGIME_V2",
        "question":"Can residual-regime field-relative context and a strict OOS correction layer lift the stable +1.35pp signal past +2pp?",
        "direct_variants":{
            "BASELINE_DIRECT":"core4 direct LambdaRank",
            "REGIME_ALL":"all residual-regime features",
            "REGIME_PERF_ONLY":"performance residual regime only",
            "REGIME_SPEED_ONLY":"speed residual regime only",
            "REGIME_LAST3F_ONLY":"last3f residual regime only",
            "REGIME_PERF_SPEED":"performance + speed residual regimes",
            "REGIME_FIELD_REL":"all regime + within-race zscore/percentile/gap-to-mean residual context",
        },
        "correction":{
            "training":"For test Y, an inner baseline trains on Y-2 and predicts Y-1; correction trains only on those Y-1 OOS baseline scores + prior-only regime features. Final baseline trains Y-2,Y-1 and predicts Y.",
            "target":"top3 probability",
            "features":"baseline within-race strength + residual regime + field-relative residual regime",
            "variants":{
                "CORRECTION_ONLY":"correction probability only",
                "CORRECTION_BLEND_25":"75% baseline rank score + 25% correction rank score",
                "CORRECTION_BLEND_50":"50/50 fixed blend; primary correction hypothesis",
                "CORRECTION_BLEND_75":"25% baseline + 75% correction",
            },
        },
        "strictness":[
            "All residual/regime features are shifted by >=1 horse race date.",
            "Field-relative residual context uses only pre-race-known state of current entrants.",
            "Correction training sees only OOS baseline predictions from the immediately prior year.",
            "Final evaluation remains two prior calendar years -> next unknown year, 2021-2025.",
            "No odds or popularity are used.",
            "2026 is sealed.",
        ],
        "primary_keys":[list(x) for x in PRIMARY_KEYS],
        "best_primary":best,
        "promotion_rule":"Require >=2.0pp pooled top1_top3 uplift vs baseline direct and no fold worse than -0.5pp.",
        "promotion":bool(best["top1_top3_uplift_pp"]>=2.0 and best["top1_top3_uplift_worst_fold_pp"]>=-0.5),
        "ability_uses_odds":False,
        "2026_locked":True,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== POOLED =====")
    print((out/"pooled-metrics.csv").read_text())
    print("L1_RESIDUAL_REGIME_V2_COMPLETE")

if __name__=="__main__":
    main()
