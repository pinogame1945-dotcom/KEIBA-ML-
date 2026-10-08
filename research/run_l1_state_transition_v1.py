#!/usr/bin/env python3
import argparse,csv,gc,json,math,os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l1_objective_rebuild_v1 import (
    YEARS,TEST_YEARS,LOCKED_YEAR,THREADS,META,
    open_text,read_year,select_variant,prepare,model_common,metrics,rank_score,
)
from run_l1_ceiling_audit_v1 import strip_forbidden

VARIANTS=("BASELINE","LAG_SLOTS","STATE_TRANSITION","RESIDUAL_STATE_TRANSITION")
HEADS=("TOP3_BINARY","RANK_GRADED","BLEND_TOP3_RANK")

RAW_LAG_METRICS=(
    "finish_pct","is_win","is_top3","speed_mps","last3f_speed_mps",
    "distance_m","body_weight","carried_weight","field_size","margin_lengths",
)
TRANSITION_METRICS=(
    "finish_pct","speed_mps","last3f_speed_mps",
    "distance_m","body_weight","carried_weight",
)
RESIDUAL_METRICS=("perf_resid","speed_resid","last3f_resid")
MAX_LAG=5

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

def read_compact_state_source(path,year):
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
                raise SystemExit(f"state source year drift expected={year} date={date}")
            rid=str(r.get("race_id") or "")
            hid=str(r.get("horse_id") or "")
            if not rid or not hid:
                continue
            finish=finite(target.get("finish_position"))
            if not math.isfinite(finish) or finish<1:
                continue
            distance=finite(feat.get("distance_m"))
            time_ms=finite(target.get("finish_time_ms"))
            last3f=finite(target.get("last_3f"))
            rows.append({
                "_race_id":rid,
                "_horse_id":hid,
                "_race_date":date,
                "_year":year,
                "_finish":finish,
                "_is_win":int(finish==1),
                "_is_top3":int(finish<=3),
                "distance_m":distance,
                "body_weight":finite(feat.get("body_weight")),
                "carried_weight":finite(feat.get("carried_weight")),
                "declared_field_size":finite(feat.get("field_size")),
                "margin_lengths":finite(target.get("margin_lengths")),
                "network_expected_pairwise_score":finite(feat.get("network_expected_pairwise_score")),
                "finish_time_ms":time_ms,
                "last_3f":last3f,
            })
    df=pd.DataFrame.from_records(rows)
    if df.empty:
        raise SystemExit(f"empty compact state source year={year}")
    return df

def build_state_features(paths):
    compact=[]
    for year in YEARS:
        df=read_compact_state_source(paths[year],year)
        compact.append(df)
        print(f"STATE_SOURCE_READY year={year} rows={len(df)} races={df['_race_id'].nunique()}",flush=True)
    s=pd.concat(compact,ignore_index=True,copy=False)
    del compact

    race_n=s.groupby("_race_id")["_horse_id"].transform("size").astype(float)
    denom=(race_n-1.0).clip(lower=1.0)
    s["field_size"]=race_n.astype("float32")
    s["finish_pct"]=(1.0-(s["_finish"].astype(float)-1.0)/denom).clip(0.0,1.0).astype("float32")

    sec=s["finish_time_ms"].astype(float)/1000.0
    s["speed_mps"]=np.where(
        np.isfinite(s["distance_m"]) & np.isfinite(sec) & (sec>0),
        s["distance_m"].astype(float)/sec,
        np.nan,
    ).astype("float32")
    s["last3f_speed_mps"]=np.where(
        np.isfinite(s["last_3f"]) & (s["last_3f"].astype(float)>0),
        600.0/s["last_3f"].astype(float),
        np.nan,
    ).astype("float32")

    race_speed_med=s.groupby("_race_id")["speed_mps"].transform("median")
    race_last3_med=s.groupby("_race_id")["last3f_speed_mps"].transform("median")
    expected=s["network_expected_pairwise_score"].astype(float)
    s["perf_resid"]=(s["finish_pct"].astype(float)-expected).astype("float32")
    s["speed_resid"]=(s["speed_mps"].astype(float)-race_speed_med.astype(float)).astype("float32")
    s["last3f_resid"]=(s["last3f_speed_mps"].astype(float)-race_last3_med.astype(float)).astype("float32")

    # Strict chronology. Same-horse same-date duplicates would make ordering ambiguous, so stop rather than leak.
    dup=s.duplicated(["_horse_id","_race_date"],keep=False)
    if dup.any():
        sample=s.loc[dup,["_horse_id","_race_date","_race_id"]].head(10).to_dict("records")
        raise SystemExit(f"same-horse same-date duplicate in state source: {sample}")

    s=s.sort_values(["_horse_id","_race_date","_race_id"]).reset_index(drop=True)
    g=s.groupby("_horse_id",sort=False)

    out=s[["_race_id","_horse_id","_race_date","_year"]].copy()
    prior_count=g.cumcount().astype("int16")
    out["seq_history_count"]=prior_count

    # Race-gap state.
    current_date=pd.to_datetime(s["_race_date"],errors="coerce")
    lag_dates={}
    for k in range(1,MAX_LAG+1):
        lag_dates[k]=pd.to_datetime(g["_race_date"].shift(k),errors="coerce")
        out[f"seq_gap_current_lag{k}_days"]=(current_date-lag_dates[k]).dt.days.astype("float32")
    out["seq_gap_lag1_lag2_days"]=(lag_dates[1]-lag_dates[2]).dt.days.astype("float32")
    out["seq_gap_lag2_lag3_days"]=(lag_dates[2]-lag_dates[3]).dt.days.astype("float32")

    lag_cols=[]
    for metric in RAW_LAG_METRICS:
        src={
            "is_win":"_is_win",
            "is_top3":"_is_top3",
        }.get(metric,metric)
        for k in range(1,MAX_LAG+1):
            c=f"seq_lag{k}_{metric}"
            out[c]=pd.to_numeric(g[src].shift(k),errors="coerce").astype("float32")
            lag_cols.append(c)
    lag_cols.extend(
        ["seq_history_count"]
        +[f"seq_gap_current_lag{k}_days" for k in range(1,MAX_LAG+1)]
        +["seq_gap_lag1_lag2_days","seq_gap_lag2_lag3_days"]
    )

    transition_cols=[]
    for metric in TRANSITION_METRICS:
        l1=out[f"seq_lag1_{metric}"]
        l2=out[f"seq_lag2_{metric}"]
        l3=out[f"seq_lag3_{metric}"]
        d12=f"seq_delta12_{metric}"
        d23=f"seq_delta23_{metric}"
        accel=f"seq_accel_{metric}"
        reversal=f"seq_reversal_{metric}"
        std3=f"seq_std3_{metric}"
        range3=f"seq_range3_{metric}"
        out[d12]=(l1-l2).astype("float32")
        out[d23]=(l2-l3).astype("float32")
        out[accel]=(out[d12]-out[d23]).astype("float32")
        valid=l1.notna()&l2.notna()&l3.notna()
        rev=np.where(valid & ((out[d12]*out[d23])<0),1.0,np.where(valid,0.0,np.nan))
        out[reversal]=pd.Series(rev,index=out.index,dtype="float32")
        trio=pd.concat([l1,l2,l3],axis=1)
        out[std3]=trio.std(axis=1,ddof=0).astype("float32")
        out[range3]=(trio.max(axis=1)-trio.min(axis=1)).astype("float32")
        transition_cols.extend([d12,d23,accel,reversal,std3,range3])

    residual_cols=[]
    for metric in RESIDUAL_METRICS:
        for k in range(1,MAX_LAG+1):
            c=f"seq_lag{k}_{metric}"
            out[c]=pd.to_numeric(g[metric].shift(k),errors="coerce").astype("float32")
            residual_cols.append(c)
        l1=out[f"seq_lag1_{metric}"]
        l2=out[f"seq_lag2_{metric}"]
        l3=out[f"seq_lag3_{metric}"]
        d12=f"seq_delta12_{metric}"
        d23=f"seq_delta23_{metric}"
        accel=f"seq_accel_{metric}"
        reversal=f"seq_reversal_{metric}"
        pos3=f"seq_positive_count3_{metric}"
        std3=f"seq_std3_{metric}"
        out[d12]=(l1-l2).astype("float32")
        out[d23]=(l2-l3).astype("float32")
        out[accel]=(out[d12]-out[d23]).astype("float32")
        valid=l1.notna()&l2.notna()&l3.notna()
        rev=np.where(valid & ((out[d12]*out[d23])<0),1.0,np.where(valid,0.0,np.nan))
        out[reversal]=pd.Series(rev,index=out.index,dtype="float32")
        trio=pd.concat([l1,l2,l3],axis=1)
        out[pos3]=trio.gt(0).sum(axis=1).where(valid,np.nan).astype("float32")
        out[std3]=trio.std(axis=1,ddof=0).astype("float32")
        residual_cols.extend(
            [f"seq_lag{k}_{metric}" for k in range(1,MAX_LAG+1)]
            +[d12,d23,accel,reversal,pos3,std3]
        )

    # Expected strength from the prior race is residual context, never current-race market.
    for k in range(1,MAX_LAG+1):
        c=f"seq_lag{k}_network_expected_pairwise_score"
        out[c]=pd.to_numeric(g["network_expected_pairwise_score"].shift(k),errors="coerce").astype("float32")
        residual_cols.append(c)

    if any(c.startswith("seq_") is False for c in lag_cols+transition_cols+residual_cols):
        raise SystemExit("state feature namespace violation")

    by_year={}
    coverage=[]
    keep_base=["_race_id","_horse_id"]
    all_state_cols=sorted(set(lag_cols+transition_cols+residual_cols))
    for year in YEARS:
        y=out[out["_year"]==year][keep_base+all_state_cols].copy()
        if y.duplicated(keep_base).any():
            raise SystemExit(f"duplicate state key year={year}")
        by_year[year]=y
        coverage.append({
            "year":year,
            "rows":len(y),
            "lag1_finish_pct_coverage_pct":100*float(y["seq_lag1_finish_pct"].notna().mean()),
            "lag3_finish_pct_coverage_pct":100*float(y["seq_lag3_finish_pct"].notna().mean()),
            "lag5_finish_pct_coverage_pct":100*float(y["seq_lag5_finish_pct"].notna().mean()),
            "lag1_perf_resid_coverage_pct":100*float(y["seq_lag1_perf_resid"].notna().mean()),
            "lag3_perf_resid_coverage_pct":100*float(y["seq_lag3_perf_resid"].notna().mean()),
        })
    del s,out
    gc.collect()
    return by_year,lag_cols,transition_cols,residual_cols,coverage

def variant_state_cols(variant,lag_cols,transition_cols,residual_cols):
    if variant=="BASELINE":
        return []
    if variant=="LAG_SLOTS":
        return lag_cols
    if variant=="STATE_TRANSITION":
        return sorted(set(lag_cols+transition_cols))
    if variant=="RESIDUAL_STATE_TRANSITION":
        return sorted(set(lag_cols+transition_cols+residual_cols))
    raise ValueError(variant)

def attach_state(base,state,cols):
    if not cols:
        return base.copy()
    right=state[["_race_id","_horse_id",*cols]]
    out=base.merge(right,on=["_race_id","_horse_id"],how="left",validate="one_to_one")
    return out

def fit_heads(train,valid,seed):
    train=train.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    valid=valid.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    xtr,xva,cats=prepare(train,valid)
    print(
        f"STATE_TRAIN rows_train={len(train)} rows_valid={len(valid)} "
        f"features={xtr.shape[1]} categorical={len(cats)} threads={THREADS}",
        flush=True,
    )

    top3=lgb.LGBMClassifier(objective="binary",**model_common(seed+1))
    top3.fit(xtr,train["_is_top3"].to_numpy(),categorical_feature=cats)
    ptop3=np.asarray(top3.predict_proba(xva)[:,1],dtype=float)
    del top3
    gc.collect()

    rel=np.select(
        [
            train["_finish"].eq(1),train["_finish"].eq(2),train["_finish"].eq(3),
            train["_finish"].eq(4),train["_finish"].eq(5),train["_finish"].eq(6),
        ],
        [6,5,4,3,2,1],
        default=0,
    ).astype(int)
    group=train.groupby("_race_id",sort=False).size().tolist()
    rank=lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0,1,3,7,15,31,63],
        **model_common(seed+2),
    )
    rank.fit(xtr,rel,group=group,categorical_feature=cats)
    prank=np.asarray(rank.predict(xva),dtype=float)

    pred=valid[["_race_id","_horse_id","_finish","_is_win","_is_top3"]].copy()
    pred["TOP3_BINARY"]=ptop3
    pred["RANK_GRADED"]=prank
    rt=rank_score(pred,"TOP3_BINARY")
    rr=rank_score(pred,"RANK_GRADED")
    pred["BLEND_TOP3_RANK"]=(rt+rr)/2.0

    del xtr,xva,rank
    gc.collect()
    return pred

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
    base={(r["head"]):r for r in pooled_rows if r["variant"]=="BASELINE"}
    fold_base={(r["test_year"],r["head"]):r for r in fold_rows if r["variant"]=="BASELINE"}
    for r in pooled_rows:
        b=base[r["head"]]
        r["top1_top3_uplift_pp"]=r["top1_top3_pct"]-b["top1_top3_pct"]
        if r["variant"]=="BASELINE":
            r["top1_top3_uplift_worst_fold_pp"]=0.0
        else:
            diffs=[]
            for fr in fold_rows:
                if fr["variant"]==r["variant"] and fr["head"]==r["head"]:
                    diffs.append(fr["top1_top3_pct"]-fold_base[(fr["test_year"],fr["head"])]["top1_top3_pct"])
            r["top1_top3_uplift_worst_fold_pp"]=float(min(diffs))
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
        "contract":"L1_STATE_TRANSITION_V1_RUNTIME",
        "variants":list(VARIANTS),
        "heads":list(HEADS),
        "threads":THREADS,
        "max_lag":MAX_LAG,
        "residual_definition":"prior realized finish percentile minus prior pre-race network_expected_pairwise_score; plus prior race-relative speed/last3f residuals",
        "ability_uses_odds":False,
        "2026_locked":True,
    },separators=(",",":")),flush=True)

    state_by_year,lag_cols,transition_cols,residual_cols,coverage=build_state_features(paths)

    cache={}
    def get_base(year):
        if year not in cache:
            raw=strip_forbidden(read_year(paths[year],year))
            # Freeze the comparison base to the prior best simple direct family.
            cache[year]=select_variant(raw,"core4")
            del raw
            gc.collect()
        return cache[year]

    rows=[]
    feature_counts=[]
    for test in TEST_YEARS:
        train_years=(test-2,test-1)
        train_base=pd.concat([get_base(y) for y in train_years],ignore_index=True,copy=False)
        valid_base=get_base(test)

        train_state=pd.concat([state_by_year[y] for y in train_years],ignore_index=True,copy=False)
        valid_state=state_by_year[test]

        for vi,variant in enumerate(VARIANTS):
            state_cols=variant_state_cols(variant,lag_cols,transition_cols,residual_cols)
            train=attach_state(train_base,train_state,state_cols)
            valid=attach_state(valid_base,valid_state,state_cols)
            seq_count=sum(c.startswith("seq_") for c in train.columns)
            feature_counts.append({
                "test_year":test,"variant":variant,
                "state_feature_count":len(state_cols),
                "total_model_columns":len([c for c in train.columns if c not in META]),
                "seq_model_columns":seq_count,
            })
            pred=fit_heads(train,valid,1500000+test*100+vi*10)
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
                f"STATE_FOLD_READY year={test} variant={variant} state_features={len(state_cols)}",
                flush=True,
            )
            del train,valid,pred
            gc.collect()

        del train_base,valid_base,train_state,valid_state
        for year in list(cache):
            if year < test-1:
                del cache[year]
        gc.collect()

    pooled_rows=add_uplift(pooled(rows),rows)
    nonbase=[r for r in pooled_rows if r["variant"]!="BASELINE"]
    best=max(nonbase,key=lambda r:(r["top1_top3_uplift_pp"],r["top1_top3_pct"]))

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"fold-metrics.csv",rows)
    write_csv(out/"pooled-metrics.csv",pooled_rows)
    write_csv(out/"state-coverage.csv",coverage)
    write_csv(out/"feature-counts.csv",feature_counts)

    (out/"summary.json").write_text(json.dumps({
        "contract":"L1_STATE_TRANSITION_V1",
        "question":"Does preserving exact prior-race state and state transitions materially break the ~60% top1 podium ceiling?",
        "variants":{
            "BASELINE":"core4 direct static features only",
            "LAG_SLOTS":"baseline + exact shifted prior 1..5 outcome/context slots",
            "STATE_TRANSITION":"lag slots + consecutive deltas, acceleration, reversal, volatility",
            "RESIDUAL_STATE_TRANSITION":"state transition + prior expectation-vs-realization residual state",
        },
        "residuals":{
            "perf_resid":"prior finish percentile - prior pre-race network expected pairwise score",
            "speed_resid":"prior speed minus prior-race median speed",
            "last3f_resid":"prior last3f speed minus prior-race median last3f speed",
        },
        "strictness":[
            "All sequence features are grouped by horse and shifted by >=1 race.",
            "Current-race target fields never enter current-race sequence features.",
            "Same-horse same-date duplicates are fatal instead of being arbitrarily ordered.",
            "Walk-forward remains two prior years -> next unknown year for 2021-2025.",
            "No odds or popularity are used.",
            "2026 is sealed."
        ],
        "heads":list(HEADS),
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
    print("===== COVERAGE =====")
    print((out/"state-coverage.csv").read_text())
    print("L1_STATE_TRANSITION_V1_COMPLETE")

if __name__=="__main__":
    main()
