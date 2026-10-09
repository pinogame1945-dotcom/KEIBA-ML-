#!/usr/bin/env python3
import argparse,csv,gc,json,math
from pathlib import Path

import numpy as np
import pandas as pd

from run_l1_objective_rebuild_v1 import (
    YEARS,TEST_YEARS,LOCKED_YEAR,META,open_text,read_year,select_variant,
)
from run_l1_ceiling_audit_v1 import strip_forbidden
from run_l1_state_transition_v1 import build_state_features,attach_state
from run_l1_residual_regime_full_pairwise_v1 import (
    build_regime_features,fit_direct_rank,top1_metrics,
)

VARIANTS=(
    "BASELINE",
    "SPEED_REGIME",
    "DAY_BIAS",
    "DAY_BIAS_STYLE_FIT",
    "SPEED_PLUS_DAY_BIAS",
    "SPEED_DAY_INTERACTION",
)

STYLE_FIELDS=(
    "style_recent_avg_first_ratio",
    "style_recent_avg_first_position",
    "style_previous_first_ratio",
    "style_previous_first_position",
)

def finite(v):
    try:
        x=float(v)
    except (TypeError,ValueError):
        return np.nan
    return x if math.isfinite(x) else np.nan

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

def read_bias_source(path,year):
    rows=[]
    with open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            feat=r.get("features") or {}
            target=r.get("target") or {}
            date=str(feat.get("race_date") or r.get("race_date") or "")[:10]
            if not date.startswith(str(year)+"-"):
                raise SystemExit(f"bias source year drift expected={year} date={date}")
            rid=str(r.get("race_id") or "")
            hid=str(r.get("horse_id") or "")
            finish=finite(target.get("finish_position"))
            if not rid or not hid or not math.isfinite(finish) or finish<1:
                continue
            time_ms=finite(target.get("finish_time_ms"))
            distance=finite(feat.get("distance_m"))
            last3f=finite(target.get("last_3f"))
            speed=np.nan
            if math.isfinite(distance) and math.isfinite(time_ms) and time_ms>0:
                speed=distance/(time_ms/1000.0)
            last3f_speed=np.nan
            if math.isfinite(last3f) and last3f>0:
                last3f_speed=600.0/last3f

            row={
                "_race_id":rid,
                "_horse_id":hid,
                "_race_date":date,
                "_year":year,
                "_finish":finish,
                "_is_win":int(finish==1),
                "_is_top3":int(finish<=3),
                "venue_code":str(feat.get("venue_code") or ""),
                "surface":str(feat.get("surface") or ""),
                "race_no":finite(feat.get("race_no")),
                "distance_m":distance,
                "gate":finite(feat.get("gate")),
                "horse_number":finite(feat.get("horse_number")),
                "field_size":finite(feat.get("field_size")),
                "speed_mps":speed,
                "last3f_speed_mps":last3f_speed,
            }
            for c in STYLE_FIELDS:
                row[c]=finite(feat.get(c))
            rows.append(row)
    out=pd.DataFrame.from_records(rows)
    if out.empty:
        raise SystemExit(f"empty bias source year={year}")
    return out

def infer_front_score(df):
    # Ratio is already field-normalized and is preferred. Smaller first-corner ratio means
    # a more forward historical running style. Fall back to raw first position normalized by field size.
    ratio=pd.to_numeric(df["style_recent_avg_first_ratio"],errors="coerce")
    prev_ratio=pd.to_numeric(df["style_previous_first_ratio"],errors="coerce")
    pos=pd.to_numeric(df["style_recent_avg_first_position"],errors="coerce")
    prev_pos=pd.to_numeric(df["style_previous_first_position"],errors="coerce")
    fs=pd.to_numeric(df["field_size"],errors="coerce").clip(lower=2)
    pos_ratio=(pos-1.0)/(fs-1.0)
    prev_pos_ratio=(prev_pos-1.0)/(fs-1.0)
    used=ratio.copy()
    used=used.where(used.notna(),prev_ratio)
    used=used.where(used.notna(),pos_ratio)
    used=used.where(used.notna(),prev_pos_ratio)
    return (1.0-used).clip(0.0,1.0).astype("float32")

def race_summary(race):
    n=len(race)
    top=race[race["_is_top3"]==1]
    if n<2 or top.empty:
        return None

    horse_no=pd.to_numeric(race["horse_number"],errors="coerce")
    denom=max(1.0,float(n-1))
    inner=(1.0-(horse_no-1.0)/denom).clip(0.0,1.0)
    front=pd.to_numeric(race["front_score"],errors="coerce")
    last3=pd.to_numeric(race["last3f_speed_mps"],errors="coerce")
    speed=pd.to_numeric(race["speed_mps"],errors="coerce")
    finish=pd.to_numeric(race["_finish"],errors="coerce")

    def gap_top(v):
        mask=v.notna()
        t=mask & race["_is_top3"].eq(1)
        if int(mask.sum())<2 or int(t.sum())<1:
            return np.nan
        return float(v[t].mean()-v[mask].mean())

    # Positive front/inner gaps mean those profiles overperformed in the completed race.
    front_gap=gap_top(front)
    inner_gap=gap_top(inner)
    last3_gap=gap_top(last3)

    winner=race.loc[finish.idxmin()] if finish.notna().any() else None
    winner_front=float(winner["front_score"]) if winner is not None and pd.notna(winner["front_score"]) else np.nan
    winner_inner=np.nan
    if winner is not None and pd.notna(winner["horse_number"]):
        winner_inner=float(np.clip(1.0-(float(winner["horse_number"])-1.0)/denom,0.0,1.0))

    speed_med=float(speed.median()) if speed.notna().any() else np.nan
    speed_top=float(speed[race["_is_top3"].eq(1)].mean()) if speed[race["_is_top3"].eq(1)].notna().any() else np.nan
    speed_adv=speed_top-speed_med if math.isfinite(speed_top) and math.isfinite(speed_med) else np.nan

    return {
        "front_gap":front_gap,
        "inner_gap":inner_gap,
        "last3_gap":last3_gap,
        "winner_front":winner_front,
        "winner_inner":winner_inner,
        "speed_adv":speed_adv,
    }

class BiasState:
    def __init__(self):
        self.items=[]
    def snapshot(self):
        vals=self.items
        out={"daybias_prior_races":float(len(vals))}
        names=("front_gap","inner_gap","last3_gap","winner_front","winner_inner","speed_adv")
        for name in names:
            x=np.asarray([v.get(name,np.nan) for v in vals],dtype=float)
            x=x[np.isfinite(x)]
            out[f"daybias_{name}_mean"]=float(x.mean()) if len(x) else np.nan
            if len(x):
                weights=np.power(0.65,np.arange(len(x)-1,-1,-1,dtype=float))
                out[f"daybias_{name}_ewma"]=float(np.average(x,weights=weights))
            else:
                out[f"daybias_{name}_ewma"]=np.nan
        return out
    def update(self,summary):
        if summary is not None:
            self.items.append(summary)

def build_same_day_bias(paths):
    by_year={}
    coverage=[]
    for year in YEARS:
        src=read_bias_source(paths[year],year)
        src["front_score"]=infer_front_score(src)

        # Race ordering is strictly chronological inside each date/venue.
        race_meta=(
            src.groupby("_race_id",as_index=False)
               .agg(
                   _race_date=("_race_date","first"),
                   venue_code=("venue_code","first"),
                   surface=("surface","first"),
                   race_no=("race_no","first"),
               )
        )
        if race_meta.duplicated(["_race_date","venue_code","surface","race_no"],keep=False).any():
            # Same race_no duplicates are processed as a batch below, so no arbitrary ordering leaks.
            pass

        feature_rows=[]
        for (date,venue,surface),meta_grp in race_meta.groupby(
            ["_race_date","venue_code","surface"],sort=True
        ):
            state=BiasState()
            meta_grp=meta_grp.sort_values(["race_no","_race_id"],na_position="last")
            for race_no,batch in meta_grp.groupby("race_no",sort=True,dropna=False):
                # Every race at the same stage sees the same state BEFORE any same-stage result update.
                snap=state.snapshot()
                pending=[]
                for rid in batch["_race_id"].tolist():
                    race=src[src["_race_id"]==rid].copy()
                    row=race[["_race_id","_horse_id"]].copy()
                    for k,v in snap.items():
                        row[k]=v
                    row["horse_front_score"]=pd.to_numeric(race["front_score"],errors="coerce").to_numpy()
                    n=len(race)
                    hno=pd.to_numeric(race["horse_number"],errors="coerce")
                    row["horse_inner_score"]=(1.0-(hno-1.0)/max(1.0,float(n-1))).clip(0.0,1.0).to_numpy()
                    row["fit_front_mean"]=(row["horse_front_score"]*snap["daybias_front_gap_mean"]).astype("float32")
                    row["fit_front_ewma"]=(row["horse_front_score"]*snap["daybias_front_gap_ewma"]).astype("float32")
                    row["fit_inner_mean"]=(row["horse_inner_score"]*snap["daybias_inner_gap_mean"]).astype("float32")
                    row["fit_inner_ewma"]=(row["horse_inner_score"]*snap["daybias_inner_gap_ewma"]).astype("float32")
                    feature_rows.append(row)
                    pending.append(race_summary(race))
                for summary in pending:
                    state.update(summary)

        y=pd.concat(feature_rows,ignore_index=True)
        if y.duplicated(["_race_id","_horse_id"]).any():
            raise SystemExit(f"duplicate day-bias key year={year}")
        by_year[year]=y
        coverage.append({
            "year":year,
            "rows":len(y),
            "races":int(y["_race_id"].nunique()),
            "prior_race_any_pct":100*float((y["daybias_prior_races"]>0).mean()),
            "prior_race_2plus_pct":100*float((y["daybias_prior_races"]>=2).mean()),
            "front_bias_cov_pct":100*float(y["daybias_front_gap_ewma"].notna().mean()),
            "inner_bias_cov_pct":100*float(y["daybias_inner_gap_ewma"].notna().mean()),
            "last3_bias_cov_pct":100*float(y["daybias_last3_gap_ewma"].notna().mean()),
        })
        print(
            f"DAY_BIAS_READY year={year} rows={len(y)} races={y['_race_id'].nunique()} "
            f"prior_any={coverage[-1]['prior_race_any_pct']:.2f} "
            f"front_cov={coverage[-1]['front_bias_cov_pct']:.2f}",
            flush=True,
        )
    return by_year,coverage

def bias_cols(frame,with_fit=False):
    base=[c for c in frame.columns if c.startswith("daybias_")]
    if with_fit:
        base += [c for c in frame.columns if c.startswith("horse_") or c.startswith("fit_")]
    return sorted(set(base))

def speed_cols(regime_cols):
    return [c for c in regime_cols if c.startswith("regime_speed_resid_")]

def add_speed_bias_interactions(frame,spd_cols):
    x=frame.copy()
    interactions=[]
    speed_pick=[
        "regime_speed_resid_lag1",
        "regime_speed_resid_mean3",
        "regime_speed_resid_mean5",
        "regime_speed_resid_strength3",
    ]
    bias_pick=[
        "daybias_front_gap_ewma",
        "daybias_inner_gap_ewma",
        "daybias_last3_gap_ewma",
        "daybias_speed_adv_ewma",
        "fit_front_ewma",
        "fit_inner_ewma",
    ]
    for s in speed_pick:
        if s not in x.columns:
            continue
        for b in bias_pick:
            if b not in x.columns:
                continue
            c=f"interact_{s}__x__{b}"
            x[c]=(pd.to_numeric(x[s],errors="coerce")*pd.to_numeric(x[b],errors="coerce")).astype("float32")
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
        "contract":"L1_SAMEDAY_TRACK_BIAS_V1_RUNTIME",
        "variants":list(VARIANTS),
        "same_seed_across_variants":True,
        "bias_update":"strictly previous race_no within same date+venue+surface; same-stage races batch before update",
        "ability_uses_odds":False,
        "2026_locked":True,
    },separators=(",",":")),flush=True)

    bias_by_year,bias_coverage=build_same_day_bias(paths)
    v1_by_year,_,_,_,state_coverage=build_state_features(paths)
    regime_by_year,regime_cols=build_regime_features(v1_by_year)
    spd_cols=speed_cols(regime_cols)

    sample=bias_by_year[2021]
    day_cols=bias_cols(sample,with_fit=False)
    fit_cols=bias_cols(sample,with_fit=True)

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
        btrain=pd.concat([bias_by_year[y] for y in train_years],ignore_index=True,copy=False)
        bvalid=bias_by_year[test]
        rtrain=pd.concat([regime_by_year[y] for y in train_years],ignore_index=True,copy=False)
        rvalid=regime_by_year[test]

        merged_train=btrain.merge(
            rtrain[["_race_id","_horse_id",*spd_cols]],
            on=["_race_id","_horse_id"],how="left",validate="one_to_one",
        )
        merged_valid=bvalid.merge(
            rvalid[["_race_id","_horse_id",*spd_cols]],
            on=["_race_id","_horse_id"],how="left",validate="one_to_one",
        )
        merged_train_i,interaction_cols=add_speed_bias_interactions(merged_train,spd_cols)
        merged_valid_i,_=add_speed_bias_interactions(merged_valid,spd_cols)

        specs={
            "BASELINE":(train_base,valid_base,[]),
            "SPEED_REGIME":(
                attach_state(train_base,rtrain,spd_cols),
                attach_state(valid_base,rvalid,spd_cols),
                spd_cols,
            ),
            "DAY_BIAS":(
                attach_state(train_base,btrain,day_cols),
                attach_state(valid_base,bvalid,day_cols),
                day_cols,
            ),
            "DAY_BIAS_STYLE_FIT":(
                attach_state(train_base,btrain,fit_cols),
                attach_state(valid_base,bvalid,fit_cols),
                fit_cols,
            ),
            "SPEED_PLUS_DAY_BIAS":(
                attach_state(train_base,merged_train,fit_cols+spd_cols),
                attach_state(valid_base,merged_valid,fit_cols+spd_cols),
                fit_cols+spd_cols,
            ),
            "SPEED_DAY_INTERACTION":(
                attach_state(train_base,merged_train_i,fit_cols+spd_cols+interaction_cols),
                attach_state(valid_base,merged_valid_i,fit_cols+spd_cols+interaction_cols),
                fit_cols+spd_cols+interaction_cols,
            ),
        }

        seed=5100000+test*100
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
            f"SAMEDAY_FOLD_READY year={test} day_features={len(fit_cols)} "
            f"speed_features={len(spd_cols)} interactions={len(interaction_cols)}",
            flush=True,
        )
        del train_base,valid_base,btrain,bvalid,rtrain,rvalid
        del merged_train,merged_valid,merged_train_i,merged_valid_i,specs
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
    write_csv(out/"bias-coverage.csv",bias_coverage)
    write_csv(out/"state-coverage.csv",state_coverage)

    (out/"summary.json").write_text(json.dumps({
        "contract":"L1_SAMEDAY_TRACK_BIAS_V1",
        "question":"Does strictly prior same-day same-venue/surface bias add new current-environment information, alone or combined with speed residual state?",
        "variants":{
            "BASELINE":"existing core4",
            "SPEED_REGIME":"speed residual regime only",
            "DAY_BIAS":"prior-race same-day aggregate front/inner/last3f/speed bias",
            "DAY_BIAS_STYLE_FIT":"day bias + each current horse's pre-race style/inside fit",
            "SPEED_PLUS_DAY_BIAS":"speed residual + day bias/style fit",
            "SPEED_DAY_INTERACTION":"same plus explicit fixed speed-residual x day-bias interactions",
        },
        "bias_signals":{
            "front":"completed prior race top3 horses' pre-race forward-style score vs field mean",
            "inner":"completed prior race top3 horse-number inner score vs field mean",
            "last3f":"completed prior race top3 last3f speed vs field mean",
            "speed_adv":"completed prior race top3 finish speed vs race median",
        },
        "leakage_guards":[
            "For current race R, only lower race_no results from the same date+venue+surface are visible.",
            "If multiple races share the same race_no stage, all are scored before any of those outcomes update bias.",
            "Running-style bias uses each completed prior runner's pre-race STYLE features; no current-race result-derived corner order is fabricated.",
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
    print("===== BIAS COVERAGE =====")
    print((out/"bias-coverage.csv").read_text())
    print("L1_SAMEDAY_TRACK_BIAS_V1_COMPLETE")

if __name__=="__main__":
    main()
