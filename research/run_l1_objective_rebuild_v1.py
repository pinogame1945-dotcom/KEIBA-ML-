#!/usr/bin/env python3
import argparse,csv,gzip,json,gc,math,os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

YEARS=(2019,2020,2021,2022,2023,2024,2025)
TEST_YEARS=(2021,2022,2023,2024,2025)
LOCKED_YEAR=2026
THREADS=max(1,int(os.environ.get("L1_THREADS","0") or 0) or (os.cpu_count() or 2))

BASE_CATEGORICAL={
    "venue_code","discipline","surface","direction","weather","track_condition","sex",
    "backfill_course_layout","backfill_race_class_normalized","backfill_grade",
    "backfill_sex_condition","backfill_weight_rule",
}
META={"_race_id","_horse_id","_race_date","_finish","_is_win","_is_top3"}
FORBIDDEN_KEYS={"actual_start_time","jockey_id","trainer_id","final_win_odds","final_popularity"}

FAMILY_PREFIXES=(
    "opponent_","network_","lap_","style_","distx_","backfill_",
    "auto_","ped_","actor_","timepace_",
)
VARIANTS={
    "core4":{
        "allowed_prefixes":{"opponent_","auto_","actor_","timepace_"},
        "actor_prefixes":None,
        "description":"BASE+OPPONENT+AUTO+ACTOR+TIME_PACE",
    },
    "structural":{
        "allowed_prefixes":{"opponent_","network_","lap_","style_","distx_","backfill_","actor_","timepace_"},
        "actor_prefixes":("actor_jockey_","actor_trainer_"),
        "description":"BASE+OPPONENT+NETWORK+LAP+STYLE+DISTANCE+BACKFILL+ACTOR(jockey,trainer)+TIME_PACE",
    },
}
HEADS=("WIN_BINARY","TOP3_BINARY","RANK_GRADED","BLEND_EQUAL","BLEND_TOP3_RANK","BLEND_WIN_RANK")

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def read_year(path,year):
    rows=[]
    with open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            date=str((r.get("features") or {}).get("race_date") or r.get("race_date") or "")[:10]
            if not date.startswith(str(year)+"-"):
                raise SystemExit(f"year drift expected={year} date={date}")
            target=r.get("target") or {}
            finish=target.get("finish_position")
            try:
                finish=int(float(finish))
            except (TypeError,ValueError):
                continue
            if finish<1:
                continue
            feat=dict(r.get("features") or {})
            feat.pop("race_date",None)
            for key in FORBIDDEN_KEYS:
                feat.pop(key,None)
            dt=pd.to_datetime(date,errors="coerce")
            if pd.isna(dt):
                continue
            feat["race_month"]=int(dt.month)
            feat["race_day_of_year"]=int(dt.dayofyear)
            rows.append({
                "_race_id":str(r.get("race_id") or ""),
                "_horse_id":str(r.get("horse_id") or ""),
                "_race_date":date,
                "_finish":finish,
                "_is_win":int(finish==1),
                "_is_top3":int(finish<=3),
                **feat,
            })
    df=pd.DataFrame.from_records(rows)
    if df.empty:
        raise SystemExit(f"empty year {year}")
    bad=df["_race_id"].eq("")|df["_horse_id"].eq("")
    if bad.any():
        raise SystemExit(f"missing identity y={year}")
    winners=df.groupby("_race_id")["_is_win"].sum()
    good=set(winners[winners>=1].index)
    df=df[df["_race_id"].isin(good)].copy()
    print(f"YEAR_CACHE_READY year={year} rows={len(df)} races={df['_race_id'].nunique()} cols={len(df.columns)}",flush=True)
    return df

def keep_variant_column(column,variant):
    if column in META:
        return True
    matched=next((p for p in FAMILY_PREFIXES if column.startswith(p)),None)
    if matched is None:
        return True
    cfg=VARIANTS[variant]
    if matched not in cfg["allowed_prefixes"]:
        return False
    if matched=="actor_" and cfg["actor_prefixes"]:
        return any(column.startswith(p) for p in cfg["actor_prefixes"])
    return True

def select_variant(df,variant):
    cols=[c for c in df.columns if keep_variant_column(c,variant)]
    out=df.loc[:,cols].copy()
    print(f"VARIANT_FRAME variant={variant} rows={len(out)} cols={len(cols)}",flush=True)
    return out

def prepare(train,valid):
    cols=sorted(c for c in train.columns if c not in META and c not in FORBIDDEN_KEYS)
    xtr=train.reindex(columns=cols).copy()
    xva=valid.reindex(columns=cols).copy()
    cats=[]
    for c in cols:
        if c in BASE_CATEGORICAL:
            tv=xtr[c].astype("string").fillna("__MISSING__")
            levels=sorted(set(tv.tolist()))
            xtr[c]=pd.Categorical(tv,categories=levels)
            vv=xva[c].astype("string").fillna("__MISSING__")
            xva[c]=pd.Categorical(vv,categories=levels)
            cats.append(c)
        else:
            xtr[c]=pd.to_numeric(xtr[c],errors="coerce").astype("float32")
            xva[c]=pd.to_numeric(xva[c],errors="coerce").astype("float32")
    return xtr,xva,cats

def model_common(seed):
    return dict(
        n_estimators=360,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.90,
        colsample_bytree=0.82,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=seed,
        n_jobs=THREADS,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )

def train_heads(train,valid,seed):
    train=train.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    valid=valid.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    xtr,xva,cats=prepare(train,valid)
    print(
        f"TRAIN_HEADS rows_train={len(train)} rows_valid={len(valid)} "
        f"features={xtr.shape[1]} categorical={len(cats)} threads={THREADS}",
        flush=True,
    )

    win=lgb.LGBMClassifier(objective="binary",**model_common(seed+1))
    win.fit(xtr,train["_is_win"].to_numpy(),categorical_feature=cats)
    pwin=np.asarray(win.predict_proba(xva)[:,1],dtype=float)
    del win
    gc.collect()

    top3=lgb.LGBMClassifier(objective="binary",**model_common(seed+2))
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
        **model_common(seed+3),
    )
    rank.fit(xtr,rel,group=group,categorical_feature=cats)
    prank=np.asarray(rank.predict(xva),dtype=float)

    out=valid[["_race_id","_horse_id","_finish","_is_win","_is_top3"]].copy()
    out["WIN_BINARY"]=pwin
    out["TOP3_BINARY"]=ptop3
    out["RANK_GRADED"]=prank
    del xtr,xva,rank
    gc.collect()
    return out

def rank_score(df,col):
    x=df[["_race_id",col]]
    r=x.groupby("_race_id")[col].rank(method="average",ascending=False)
    n=x.groupby("_race_id")[col].transform("size")
    return 1.0-(r-1.0)/(n-1.0).clip(lower=1.0)

def add_blends(pred):
    pred=pred.copy()
    rw=rank_score(pred,"WIN_BINARY")
    rt=rank_score(pred,"TOP3_BINARY")
    rr=rank_score(pred,"RANK_GRADED")
    pred["BLEND_EQUAL"]=(rw+rt+rr)/3.0
    pred["BLEND_TOP3_RANK"]=(rt+rr)/2.0
    pred["BLEND_WIN_RANK"]=(rw+rr)/2.0
    return pred

def metrics(pred,col):
    x=pred[["_race_id","_horse_id","_finish","_is_win","_is_top3",col]].copy()
    x["_score"]=x[col].astype(float)
    x=x.sort_values(["_race_id","_score","_horse_id"],ascending=[True,False,True])
    x["_rank"]=x.groupby("_race_id",sort=False).cumcount()+1
    races=x["_race_id"].nunique()
    top1=x[x["_rank"]==1]
    t3=x[x["_rank"]<=3]
    t6=x[x["_rank"]<=6]
    winrows=x[x["_is_win"]==1]
    per3=t3.groupby("_race_id")["_is_win"].max()
    per6=t6.groupby("_race_id")["_is_win"].max()
    return {
        "races":int(races),
        "top1_win_pct":100*float(top1["_is_win"].mean()),
        "top1_top3_pct":100*float(top1["_is_top3"].mean()),
        "winner_top3_capture_pct":100*float(per3.mean()),
        "winner_top6_capture_pct":100*float(per6.mean()),
        "top3_podium_precision_pct":100*float(t3["_is_top3"].mean()),
        "top6_podium_precision_pct":100*float(t6["_is_top3"].mean()),
        "mean_winner_rank":float(winrows["_rank"].mean()),
        "winner_mrr":float((1.0/winrows["_rank"]).mean()),
    }

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

def pooled_metrics(variant,rows):
    pooled=[]
    for head in HEADS:
        vals=[r for r in rows if r["head"]==head]
        weights=np.array([r["races"] for r in vals],dtype=float)
        rec={"variant":variant,"head":head,"folds":len(vals),"races":int(weights.sum())}
        for key in (
            "top1_win_pct","top1_top3_pct","winner_top3_capture_pct","winner_top6_capture_pct",
            "top3_podium_precision_pct","top6_podium_precision_pct","mean_winner_rank","winner_mrr",
        ):
            arr=np.array([r[key] for r in vals],dtype=float)
            rec[key]=float(np.average(arr,weights=weights))
            rec[key+"_worst"]=float(arr.max() if key=="mean_winner_rank" else arr.min())
            rec[key+"_std"]=float(arr.std(ddof=0))
        pooled.append(rec)
    return pooled

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--variants",default="core4,structural")
    p.add_argument("--year-file",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--out-dir",required=True)
    a=p.parse_args()

    variants=[x.strip() for x in a.variants.split(",") if x.strip()]
    invalid=[x for x in variants if x not in VARIANTS]
    if invalid:
        raise SystemExit(f"invalid variants: {invalid}")
    if len(set(variants))!=len(variants):
        raise SystemExit("duplicate variants")

    paths={}
    for spec in a.year_file:
        y,s=spec.split(":",1)
        paths[int(y)]=s
    if set(paths)!=set(YEARS):
        raise SystemExit(f"year file mismatch {sorted(paths)}")
    if LOCKED_YEAR in paths:
        raise SystemExit("2026 sealed")

    print(json.dumps({
        "contract":"L1_OBJECTIVE_REBUILD_V1_RUNTIME",
        "threads":THREADS,
        "variants":variants,
        "years":list(YEARS),
        "test_years":list(TEST_YEARS),
        "shared_year_cache":True,
        "2026_locked":True,
    },separators=(",",":")),flush=True)

    cache={}
    def get(year):
        if year not in cache:
            cache[year]=read_year(paths[year],year)
        return cache[year]

    fold_rows={variant:[] for variant in variants}

    for test in TEST_YEARS:
        train_years=(test-2,test-1)
        train_union=pd.concat([get(y) for y in train_years],ignore_index=True,copy=False)
        valid_union=get(test)

        for vi,variant in enumerate(variants):
            train=select_variant(train_union,variant)
            valid=select_variant(valid_union,variant)
            seed=81000+test+(vi*1000)
            pred=add_blends(train_heads(train,valid,seed))
            for head in HEADS:
                fold_rows[variant].append({
                    "variant":variant,
                    "test_year":test,
                    "train_years":"|".join(map(str,train_years)),
                    "head":head,
                    **metrics(pred,head),
                })
            del train,valid,pred
            gc.collect()

        del train_union
        for year in list(cache):
            if year < test-1:
                del cache[year]
        gc.collect()

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    comparison=[]
    for variant in variants:
        vdir=out/variant
        rows=fold_rows[variant]
        pooled=pooled_metrics(variant,rows)
        write_csv(vdir/"fold-metrics.csv",rows)
        write_csv(vdir/"pooled-metrics.csv",pooled)
        comparison.extend(pooled)
        (vdir/"summary.json").write_text(json.dumps({
            "contract":"L1_OBJECTIVE_REBUILD_V1_RESULT",
            "variant":variant,
            "feature_description":VARIANTS[variant]["description"],
            "train_window_years":2,
            "test_years":list(TEST_YEARS),
            "heads":list(HEADS),
            "rank_relevance":{"1":6,"2":5,"3":4,"4":3,"5":2,"6":1,"7+":0},
            "blend_policy":"fixed predeclared rank-score blends; no test-year weight search",
            "ability_uses_odds":False,
            "threads":THREADS,
            "shared_year_cache":True,
            "2026_locked":True,
            "promotion":False,
        },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    write_csv(out/"comparison.csv",comparison)
    (out/"summary.json").write_text(json.dumps({
        "contract":"L1_OBJECTIVE_REBUILD_V1_AGGREGATE",
        "question":"Does changing the horse-only L1 learning objective and/or adding structural feature families materially improve strict walk-forward ranking?",
        "variants":{v:VARIANTS[v]["description"] for v in variants},
        "heads":list(HEADS),
        "walk_forward":"2 prior years -> next year, 2021-2025",
        "runtime_optimization":{
            "shared_union_projection":True,
            "shared_year_cache_between_variants":True,
            "lightgbm_threads":THREADS,
            "oos_prediction_archive":False,
        },
        "ability_uses_odds":False,
        "2026_locked":True,
        "promotion":False,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== COMPARISON =====")
    print((out/"comparison.csv").read_text())
    print("L1_OBJECTIVE_REBUILD_V1_READY")

if __name__=="__main__":
    main()
