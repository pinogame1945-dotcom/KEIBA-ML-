#!/usr/bin/env python3
import argparse,csv,gc,json,math
from pathlib import Path

import numpy as np
import pandas as pd

from run_l1_objective_rebuild_v1 import (
    YEARS,TEST_YEARS,LOCKED_YEAR,META,
    read_year,select_variant,
)
from run_l1_ceiling_audit_v1 import strip_forbidden
from run_l1_state_transition_v1 import read_compact_state_source,build_state_features,attach_state
from run_l1_residual_regime_full_pairwise_v1 import (
    build_regime_features,fit_direct_rank,top1_metrics,
)

VARIANTS=(
    "BASELINE",
    "WEIGHT_CHANGE_ONLY",
    "WEIGHT_OPTIMAL_ZONE",
    "WEIGHT_FULL",
    "SPEED_REGIME",
    "WEIGHT_PLUS_SPEED",
    "WEIGHT_SPEED_INTERACTION",
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

def safe_div(a,b):
    b=pd.to_numeric(b,errors="coerce")
    return pd.to_numeric(a,errors="coerce")/b.replace(0,np.nan)

def prior_group_stats(events,value_col,mask=None,prefix="hist"):
    horse=events["_horse_id"]
    v=pd.to_numeric(events[value_col],errors="coerce")
    valid=v.notna()
    if mask is not None:
        valid=valid & pd.Series(mask,index=events.index).fillna(False)
    x=v.where(valid,0.0).fillna(0.0)
    vv=valid.astype(float)

    csum=x.groupby(horse,sort=False).cumsum()-x
    ccount=vv.groupby(horse,sort=False).cumsum()-vv
    sq=(x*x)
    csq=sq.groupby(horse,sort=False).cumsum()-sq

    mean=csum/ccount.replace(0,np.nan)
    var=(csq/ccount.replace(0,np.nan))-(mean*mean)
    std=np.sqrt(var.clip(lower=0))
    return (
        mean.astype("float32"),
        std.astype("float32"),
        ccount.astype("float32"),
    )

def build_weight_state(paths):
    frames=[]
    for year in YEARS:
        x=read_compact_state_source(paths[year],year)
        frames.append(x)
    s=pd.concat(frames,ignore_index=True,copy=False)
    del frames

    # Same horse/date must not be ordered. Current weight is pre-race known, but
    # result-derived "good weight" history may only update for future dates.
    check_cols=["body_weight","_finish","_is_top3","_is_win"]
    dup=s.duplicated(["_horse_id","_race_date"],keep=False)
    if dup.any():
        conflicts=[]
        for key,grp in s.loc[dup].groupby(["_horse_id","_race_date"],sort=False):
            bad=[]
            for c in check_cols:
                vals=pd.to_numeric(grp[c],errors="coerce").dropna().to_numpy()
                if len(vals)>1 and not np.allclose(vals,vals[0],rtol=1e-7,atol=1e-7):
                    bad.append(c)
            if bad:
                conflicts.append((key,bad,grp["_race_id"].tolist()))
                if len(conflicts)>=10:
                    break
        if conflicts:
            raise SystemExit(f"same-day weight state conflict: {conflicts}")

    events=(
        s.sort_values(["_horse_id","_race_date","_race_id"])
         .drop_duplicates(["_horse_id","_race_date"],keep="first")
         .reset_index(drop=True)
    )
    horse=events["_horse_id"]
    g=events.groupby("_horse_id",sort=False)
    out=events[["_horse_id","_race_date"]].copy()
    w=pd.to_numeric(events["body_weight"],errors="coerce")

    for k in range(1,6):
        out[f"weight_lag{k}"]=pd.to_numeric(g["body_weight"].shift(k),errors="coerce").astype("float32")

    l1=out["weight_lag1"]
    l2=out["weight_lag2"]
    l3=out["weight_lag3"]
    l4=out["weight_lag4"]
    l5=out["weight_lag5"]
    recent3=pd.concat([l1,l2,l3],axis=1)
    recent5=pd.concat([l1,l2,l3,l4,l5],axis=1)

    out["weight_change_kg"]=(w-l1).astype("float32")
    out["weight_change_pct"]=(100.0*safe_div(w-l1,l1)).astype("float32")
    out["weight_recent_mean3"]=recent3.mean(axis=1).astype("float32")
    out["weight_recent_mean5"]=recent5.mean(axis=1).astype("float32")
    out["weight_recent_std3"]=recent3.std(axis=1,ddof=0).astype("float32")
    out["weight_recent_std5"]=recent5.std(axis=1,ddof=0).astype("float32")
    out["weight_vs_mean3"]=(w-out["weight_recent_mean3"]).astype("float32")
    out["weight_vs_mean5"]=(w-out["weight_recent_mean5"]).astype("float32")
    out["weight_abs_vs_mean3"]=out["weight_vs_mean3"].abs().astype("float32")
    out["weight_abs_vs_mean5"]=out["weight_vs_mean5"].abs().astype("float32")
    out["weight_prior_trend3"]=((l1-l3)/2.0).astype("float32")
    out["weight_change_accel"]=((w-l1)-(l1-l2)).astype("float32")
    out["weight_prev_change"]=(l1-l2).astype("float32")
    out["weight_prev2_change"]=(l2-l3).astype("float32")

    hist_mean,hist_std,hist_n=prior_group_stats(events,"body_weight",prefix="all")
    out["weight_hist_mean"]=hist_mean
    out["weight_hist_std"]=hist_std
    out["weight_hist_count"]=hist_n
    out["weight_vs_hist_mean"]=(w-hist_mean).astype("float32")
    out["weight_hist_z"]=safe_div(w-hist_mean,hist_std).clip(-10,10).astype("float32")
    out["weight_hist_abs_z"]=out["weight_hist_z"].abs().astype("float32")

    top3_mean,top3_std,top3_n=prior_group_stats(
        events,"body_weight",events["_is_top3"].astype(bool),prefix="top3"
    )
    win_mean,win_std,win_n=prior_group_stats(
        events,"body_weight",events["_is_win"].astype(bool),prefix="win"
    )
    out["weight_top3_mean"]=top3_mean
    out["weight_top3_std"]=top3_std
    out["weight_top3_count"]=top3_n
    out["weight_vs_top3_mean"]=(w-top3_mean).astype("float32")
    out["weight_abs_vs_top3_mean"]=out["weight_vs_top3_mean"].abs().astype("float32")
    out["weight_top3_z"]=safe_div(w-top3_mean,top3_std).clip(-10,10).astype("float32")
    out["weight_win_mean"]=win_mean
    out["weight_win_count"]=win_n
    out["weight_vs_win_mean"]=(w-win_mean).astype("float32")
    out["weight_abs_vs_win_mean"]=out["weight_vs_win_mean"].abs().astype("float32")

    current_date=pd.to_datetime(events["_race_date"],errors="coerce")
    prior_date=pd.to_datetime(g["_race_date"].shift(1),errors="coerce")
    gap=(current_date-prior_date).dt.days.astype("float32")
    out["weight_days_since_prev"]=gap
    out["weight_change_per_sqrt_gap"]=safe_div(
        out["weight_change_kg"],np.sqrt(gap.clip(lower=1))
    ).astype("float32")
    out["weight_long_layoff_90"]=np.where(
        gap.notna(),(gap>=90).astype(float),np.nan
    ).astype("float32")
    out["weight_long_layoff_change"]=(
        out["weight_change_kg"]*out["weight_long_layoff_90"]
    ).astype("float32")

    # Fan the same current/prior state to all same-horse same-date rows.
    expanded=s[["_race_id","_horse_id","_race_date","_year"]].merge(
        out,on=["_horse_id","_race_date"],how="left",validate="many_to_one"
    )

    # Field-relative *change* context is valid at prediction time because all current
    # body weights are pre-race observations.
    by_year={}
    for year in YEARS:
        y=expanded[expanded["_year"]==year].copy()
        grp=y.groupby("_race_id",sort=False)
        for c in ("weight_change_kg","weight_change_pct","weight_hist_z","weight_vs_top3_mean"):
            v=pd.to_numeric(y[c],errors="coerce")
            mean=grp[c].transform(lambda z:pd.to_numeric(z,errors="coerce").mean())
            std=grp[c].transform(lambda z:pd.to_numeric(z,errors="coerce").std(ddof=0))
            rank=grp[c].rank(method="average",ascending=False)
            n=grp[c].transform("count").astype(float)
            y[f"field_{c}_z"]=((v-mean)/std.replace(0,np.nan)).astype("float32")
            y[f"field_{c}_pct"]=(1.0-(rank-1.0)/(n-1.0).clip(lower=1.0)).astype("float32")
            y[f"field_{c}_gap_mean"]=(v-mean).astype("float32")
        keep=[c for c in y.columns if c not in {"_race_date","_year"}]
        by_year[year]=y[keep].copy()
        print(
            f"WEIGHT_STATE_READY year={year} rows={len(y)} "
            f"current_cov={100*y['weight_change_kg'].notna().mean():.2f} "
            f"top3_zone_cov={100*y['weight_vs_top3_mean'].notna().mean():.2f}",
            flush=True,
        )
    return by_year

def speed_cols(regime_cols):
    return [c for c in regime_cols if c.startswith("regime_speed_resid_")]

def weight_groups(sample):
    all_cols=[c for c in sample.columns if c not in {"_race_id","_horse_id"}]
    change_tokens=(
        "weight_change_","weight_vs_mean","weight_hist_z","weight_hist_abs_z",
        "weight_prior_trend","weight_prev","weight_days_since","weight_long_layoff",
        "field_weight_change","field_weight_hist_z",
    )
    optimal_tokens=(
        "weight_top3_","weight_vs_top3_","weight_abs_vs_top3_",
        "weight_win_","weight_vs_win_","weight_abs_vs_win_",
        "field_weight_vs_top3",
    )
    change=[c for c in all_cols if any(t in c for t in change_tokens)]
    optimal=[c for c in all_cols if any(t in c for t in optimal_tokens)]
    return all_cols,sorted(set(change)),sorted(set(optimal))

def add_interactions(frame,weight_cols,speed_feature_cols):
    x=frame.copy()
    interactions=[]
    wchoices=[
        "weight_change_kg","weight_change_pct","weight_hist_z",
        "weight_vs_top3_mean","weight_abs_vs_top3_mean","weight_change_accel",
    ]
    schoices=[
        "regime_speed_resid_lag1","regime_speed_resid_mean3",
        "regime_speed_resid_mean5","regime_speed_resid_slope3",
        "regime_speed_resid_strength3",
    ]
    for w in wchoices:
        if w not in x.columns:
            continue
        for s in schoices:
            if s not in x.columns:
                continue
            c=f"interact_{w}__x__{s}"
            x[c]=(pd.to_numeric(x[w],errors="coerce")*pd.to_numeric(x[s],errors="coerce")).astype("float32")
            interactions.append(c)
    return x,interactions

def pooled(rows):
    out=[]
    for arch in VARIANTS:
        vals=[r for r in rows if r["architecture"]==arch]
        w=np.array([r["races"] for r in vals],dtype=float)
        rec={"architecture":arch,"head":"RANK_GRADED","folds":len(vals),"races":int(w.sum())}
        for key in ("top1_top3_pct","top1_win_pct","winner_top3_capture_pct","winner_top6_capture_pct"):
            arr=np.array([r[key] for r in vals],dtype=float)
            rec[key]=float(np.average(arr,weights=w))
            rec[key+"_worst"]=float(arr.min())
            rec[key+"_std"]=float(arr.std(ddof=0))
        out.append(rec)
    return out

def add_uplift(pool,fold):
    base=next(r for r in pool if r["architecture"]=="BASELINE")
    fold_base={r["test_year"]:r for r in fold if r["architecture"]=="BASELINE"}
    for r in pool:
        r["top1_top3_uplift_pp"]=r["top1_top3_pct"]-base["top1_top3_pct"]
        diffs=[
            x["top1_top3_pct"]-fold_base[x["test_year"]]["top1_top3_pct"]
            for x in fold if x["architecture"]==r["architecture"]
        ]
        r["top1_top3_uplift_worst_fold_pp"]=float(min(diffs))
        r["top1_top3_uplift_best_fold_pp"]=float(max(diffs))
    return pool

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
        "contract":"L1_BODYWEIGHT_STATE_V1_RUNTIME",
        "variants":list(VARIANTS),
        "same_seed_across_variants":True,
        "current_body_weight":"pre-race known observation",
        "historical_good_weight":"strictly prior dates only",
        "threads":"inherited standard CPU",
        "ability_uses_odds":False,
        "2026_locked":True,
    },separators=(",",":")),flush=True)

    weight_by_year=build_weight_state(paths)
    v1_by_year,_,_,_,coverage=build_state_features(paths)
    regime_by_year,regime_cols=build_regime_features(v1_by_year)
    spd_cols=speed_cols(regime_cols)
    full_weight_cols,change_cols,optimal_cols=weight_groups(weight_by_year[2021])

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
        wt_train=pd.concat([weight_by_year[y] for y in train_years],ignore_index=True,copy=False)
        wt_valid=weight_by_year[test]
        rg_train=pd.concat([regime_by_year[y] for y in train_years],ignore_index=True,copy=False)
        rg_valid=regime_by_year[test]

        train_weight=attach_state(train_base,wt_train,full_weight_cols)
        valid_weight=attach_state(valid_base,wt_valid,full_weight_cols)
        train_speed=attach_state(train_base,rg_train,spd_cols)
        valid_speed=attach_state(valid_base,rg_valid,spd_cols)

        merged_train=wt_train.merge(
            rg_train[["_race_id","_horse_id",*spd_cols]],
            on=["_race_id","_horse_id"],how="left",validate="one_to_one",
        )
        merged_valid=wt_valid.merge(
            rg_valid[["_race_id","_horse_id",*spd_cols]],
            on=["_race_id","_horse_id"],how="left",validate="one_to_one",
        )
        merged_train_i,interaction_cols=add_interactions(merged_train,full_weight_cols,spd_cols)
        merged_valid_i,_=add_interactions(merged_valid,full_weight_cols,spd_cols)

        specs={
            "BASELINE":(train_base,valid_base,[]),
            "WEIGHT_CHANGE_ONLY":(
                attach_state(train_base,wt_train,change_cols),
                attach_state(valid_base,wt_valid,change_cols),
                change_cols,
            ),
            "WEIGHT_OPTIMAL_ZONE":(
                attach_state(train_base,wt_train,optimal_cols),
                attach_state(valid_base,wt_valid,optimal_cols),
                optimal_cols,
            ),
            "WEIGHT_FULL":(train_weight,valid_weight,full_weight_cols),
            "SPEED_REGIME":(train_speed,valid_speed,spd_cols),
            "WEIGHT_PLUS_SPEED":(
                attach_state(train_base,merged_train,full_weight_cols+spd_cols),
                attach_state(valid_base,merged_valid,full_weight_cols+spd_cols),
                full_weight_cols+spd_cols,
            ),
            "WEIGHT_SPEED_INTERACTION":(
                attach_state(train_base,merged_train_i,full_weight_cols+spd_cols+interaction_cols),
                attach_state(valid_base,merged_valid_i,full_weight_cols+spd_cols+interaction_cols),
                full_weight_cols+spd_cols+interaction_cols,
            ),
        }

        seed=4100000+test*100
        for arch in VARIANTS:
            tr,va,added=specs[arch]
            pred,imp=fit_direct_rank(tr,va,seed)
            rows.append({
                "test_year":test,
                "train_years":"|".join(map(str,train_years)),
                "architecture":arch,
                "head":"RANK_GRADED",
                "added_feature_count":len(added),
                **top1_metrics(pred,"_score"),
            })
            feature_counts.append({
                "test_year":test,"architecture":arch,
                "added_feature_count":len(added),
                "total_model_columns":len([c for c in tr.columns if c not in META]),
            })
            for rank,(feature,gain) in enumerate(imp[:80],start=1):
                importance.append({
                    "test_year":test,"architecture":arch,
                    "importance_rank":rank,"feature":feature,"gain":float(gain),
                })
            del pred
            gc.collect()

        print(
            f"BODYWEIGHT_FOLD_READY year={test} weight_features={len(full_weight_cols)} "
            f"speed_features={len(spd_cols)} interactions={len(interaction_cols)}",
            flush=True,
        )
        del train_base,valid_base,wt_train,wt_valid,rg_train,rg_valid
        del train_weight,valid_weight,train_speed,valid_speed,merged_train,merged_valid
        del merged_train_i,merged_valid_i,specs
        gc.collect()
        for old in list(base_cache):
            if old < test-1:
                del base_cache[old]

    pool=add_uplift(pooled(rows),rows)
    nonbase=[r for r in pool if r["architecture"]!="BASELINE"]
    best=max(nonbase,key=lambda r:(r["top1_top3_uplift_pp"],r["top1_top3_pct"]))

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"fold-metrics.csv",rows)
    write_csv(out/"pooled-metrics.csv",pool)
    write_csv(out/"feature-importance.csv",importance)
    write_csv(out/"feature-counts.csv",feature_counts)
    write_csv(out/"state-coverage.csv",coverage)

    (out/"summary.json").write_text(json.dumps({
        "contract":"L1_BODYWEIGHT_STATE_V1",
        "question":"Does horse-relative current bodyweight state add information beyond existing raw body_weight/body_weight_diff, especially together with speed residual form?",
        "variants":{
            "BASELINE":"existing core4, which already contains raw current body_weight/body_weight_diff",
            "WEIGHT_CHANGE_ONLY":"current change, horse-relative deviation, recent weight trajectory, layoff-adjusted change, field-relative change",
            "WEIGHT_OPTIMAL_ZONE":"distance from strictly-prior top3/win bodyweight zones",
            "WEIGHT_FULL":"all engineered horse-relative bodyweight state",
            "SPEED_REGIME":"speed residual regime only",
            "WEIGHT_PLUS_SPEED":"all weight state + speed residual regime",
            "WEIGHT_SPEED_INTERACTION":"weight + speed plus explicit fixed current-state interactions",
        },
        "leakage_guards":[
            "Current body weight is a pre-race observation and may be used for the current race.",
            "Top3/win weight zones use outcomes only from strictly prior horse dates.",
            "Same-horse same-date rows are never ordered within a day.",
            "All variants use the identical model seed within each fold.",
            "No odds or popularity are used.",
            "2026 is sealed.",
        ],
        "best_nonbaseline":best,
        "promotion_rule":"Require >=2.0pp pooled top1_top3 uplift vs baseline and no fold worse than -0.5pp.",
        "promotion":bool(best["top1_top3_uplift_pp"]>=2.0 and best["top1_top3_uplift_worst_fold_pp"]>=-0.5),
        "ability_uses_odds":False,
        "2026_locked":True,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== POOLED =====")
    print((out/"pooled-metrics.csv").read_text())
    print("L1_BODYWEIGHT_STATE_V1_COMPLETE")

if __name__=="__main__":
    main()
