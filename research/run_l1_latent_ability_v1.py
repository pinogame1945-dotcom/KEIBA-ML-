#!/usr/bin/env python3
import argparse,csv,gzip,json,gc,os
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

YEARS=(2019,2020,2021,2022,2023,2024,2025)
TEST_YEARS=(2021,2022,2023,2024,2025)
LOCKED_YEAR=2026
CPU=max(1,int(os.environ.get("L1_THREADS","0") or 0) or (os.cpu_count() or 2))
COMPONENT_WORKERS=max(1,min(2,CPU))
COMPONENT_THREADS=max(1,CPU//COMPONENT_WORKERS)

META=("_race_id","_horse_id","_race_date","_finish","_is_win","_is_top3")
FORBIDDEN={"actual_start_time","jockey_id","trainer_id","final_win_odds","final_popularity"}
CATEGORICAL={
    "venue_code","discipline","surface","direction","weather","track_condition","sex",
    "backfill_course_layout","backfill_race_class_normalized","backfill_grade",
    "backfill_sex_condition","backfill_weight_rule",
}
FAMILY_PREFIXES=("opponent_","network_","lap_","style_","distx_","backfill_","auto_","ped_","actor_","timepace_")
COMPONENTS={
    "FORM":{"prefixes":{"auto_"},"base":True},
    "OPPONENT":{"prefixes":{"opponent_","network_"},"base":True},
    "SPEED_PACE":{"prefixes":{"lap_","style_","timepace_"},"base":True},
    "DISTANCE":{"prefixes":{"distx_"},"base":True},
    "CONDITION":{"prefixes":{"backfill_"},"base":True},
    "ACTOR":{"prefixes":{"actor_"},"base":True},
    "PEDIGREE":{"prefixes":{"ped_"},"base":True},
}
CONTEXT_COLUMNS={
    "venue_code","discipline","surface","direction","weather","track_condition","sex",
    "distance_m","field_size","gate","horse_number","carried_weight","age",
    "backfill_course_layout","backfill_race_class_normalized","backfill_grade",
    "backfill_sex_condition","backfill_weight_rule","race_month","race_day_of_year",
}
DIRECT_PREFIXES={"opponent_","network_","lap_","style_","distx_","backfill_","auto_","actor_","timepace_","ped_"}

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def is_base_key(key):
    return not any(key.startswith(p) for p in FAMILY_PREFIXES)

def read_year(path,year):
    rows=[]
    with open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            feat=dict(r.get("features") or {})
            date=str(feat.get("race_date") or r.get("race_date") or "")[:10]
            if not date.startswith(str(year)+"-"):
                raise SystemExit(f"year drift expected={year} date={date}")
            target=r.get("target") or {}
            try:
                finish=int(float(target.get("finish_position")))
            except (TypeError,ValueError):
                continue
            if finish<1:
                continue
            feat.pop("race_date",None)
            for k in FORBIDDEN:
                feat.pop(k,None)
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
    if (df["_race_id"].eq("")|df["_horse_id"].eq("")).any():
        raise SystemExit(f"missing identity y={year}")
    winners=df.groupby("_race_id")["_is_win"].sum()
    df=df[df["_race_id"].isin(set(winners[winners>=1].index))].copy()
    for c in df.columns:
        if c in META or c in CATEGORICAL:
            continue
        df[c]=pd.to_numeric(df[c],errors="coerce")
        if df[c].dtype.kind in "fc":
            df[c]=df[c].astype("float32")
    print(f"LATENT_YEAR_READY year={year} rows={len(df)} races={df['_race_id'].nunique()} cols={len(df.columns)}",flush=True)
    return df

def component_columns(df,name):
    spec=COMPONENTS[name]
    cols=[]
    for c in df.columns:
        if c in META or c in FORBIDDEN:
            continue
        matched=next((p for p in FAMILY_PREFIXES if c.startswith(p)),None)
        if matched in spec["prefixes"] or (matched is None and spec["base"]):
            cols.append(c)
    return sorted(cols)

def direct_columns(df):
    out=[]
    for c in df.columns:
        if c in META or c in FORBIDDEN:
            continue
        matched=next((p for p in FAMILY_PREFIXES if c.startswith(p)),None)
        if matched is None or matched in DIRECT_PREFIXES:
            out.append(c)
    return sorted(out)

def prep(train,valid,cols):
    xtr=train.reindex(columns=cols).copy()
    xva=valid.reindex(columns=cols).copy()
    cats=[]
    for c in cols:
        if c in CATEGORICAL:
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

def rank_relevance(finish):
    return np.select(
        [finish.eq(1),finish.eq(2),finish.eq(3),finish.eq(4),finish.eq(5),finish.eq(6)],
        [6,5,4,3,2,1],default=0,
    ).astype(int)

def ranker_params(seed,threads,trees=180):
    return dict(
        objective="lambdarank",metric="ndcg",label_gain=[0,1,3,7,15,31,63],
        n_estimators=trees,learning_rate=0.045,num_leaves=31,min_child_samples=80,
        subsample=0.90,colsample_bytree=0.85,reg_lambda=5.0,reg_alpha=0.5,
        random_state=seed,n_jobs=threads,deterministic=True,force_col_wise=True,verbosity=-1,
    )

def binary_params(seed,threads,trees=220):
    return dict(
        objective="binary",n_estimators=trees,learning_rate=0.04,num_leaves=31,
        min_child_samples=80,subsample=0.90,colsample_bytree=0.85,
        reg_lambda=5.0,reg_alpha=0.5,random_state=seed,n_jobs=threads,
        deterministic=True,force_col_wise=True,verbosity=-1,
    )

def fit_component(name,train,score,seed):
    cols=component_columns(train,name)
    xtr,xsc,cats=prep(train,score,cols)
    order=np.lexsort((train["_horse_id"].astype(str).to_numpy(),train["_race_id"].astype(str).to_numpy()))
    train_sorted=train.iloc[order].reset_index(drop=True)
    xtr=xtr.iloc[order].reset_index(drop=True)
    rel=rank_relevance(train_sorted["_finish"])
    group=train_sorted.groupby("_race_id",sort=False).size().tolist()
    model=lgb.LGBMRanker(**ranker_params(seed,COMPONENT_THREADS))
    model.fit(xtr,rel,group=group,categorical_feature=cats)
    pred=np.asarray(model.predict(xsc),dtype="float32")
    del xtr,xsc,model
    gc.collect()
    return name,pred,len(cols)

def fit_components_parallel(train,score,seed_base):
    out={}
    with ThreadPoolExecutor(max_workers=COMPONENT_WORKERS) as ex:
        futures={
            ex.submit(fit_component,name,train,score,seed_base+i*17):name
            for i,name in enumerate(COMPONENTS)
        }
        for fut in as_completed(futures):
            name,pred,ncols=fut.result()
            out[name]=pred
            print(f"ABILITY_COMPONENT_READY component={name} features={ncols} workers={COMPONENT_WORKERS} threads_each={COMPONENT_THREADS}",flush=True)
    return out

def race_rank_features(base,scores):
    out=base[list(META)].copy()
    rank_cols=[]
    z_cols=[]
    for name in COMPONENTS:
        raw=np.asarray(scores[name],dtype="float32")
        col=f"ability_{name.lower()}"
        temp=pd.DataFrame({"_race_id":base["_race_id"].to_numpy(),"_raw":raw})
        rank=temp.groupby("_race_id")["_raw"].rank(method="average",ascending=False)
        n=temp.groupby("_race_id")["_raw"].transform("size")
        pct=(1.0-(rank-1.0)/(n-1.0).clip(lower=1.0)).astype("float32")
        mean=temp.groupby("_race_id")["_raw"].transform("mean")
        std=temp.groupby("_race_id")["_raw"].transform("std").replace(0,np.nan)
        z=((temp["_raw"]-mean)/std).fillna(0.0).clip(-6,6).astype("float32")
        maxv=temp.groupby("_race_id")["_raw"].transform("max")
        gap=(temp["_raw"]-maxv).astype("float32")
        out[col+"_pct"]=pct.to_numpy()
        out[col+"_z"]=z.to_numpy()
        out[col+"_gapmax"]=gap.to_numpy()
        rank_cols.append(col+"_pct")
        z_cols.append(col+"_z")

    out["ability_consensus_mean"]=out[rank_cols].mean(axis=1).astype("float32")
    out["ability_consensus_std"]=out[rank_cols].std(axis=1).fillna(0).astype("float32")
    out["ability_consensus_min"]=out[rank_cols].min(axis=1).astype("float32")
    out["ability_consensus_max"]=out[rank_cols].max(axis=1).astype("float32")
    out["ability_z_mean"]=out[z_cols].mean(axis=1).astype("float32")
    out["ability_z_std"]=out[z_cols].std(axis=1).fillna(0).astype("float32")

    for c in CONTEXT_COLUMNS:
        if c in base.columns and c not in out.columns:
            out[c]=base[c].to_numpy()

    # Race-level uncertainty/context from the distribution of latent abilities.
    grouped=out.groupby("_race_id",sort=False)
    out["race_consensus_spread"]=grouped["ability_consensus_mean"].transform("std").fillna(0).astype("float32")
    out["race_disagreement_mean"]=grouped["ability_consensus_std"].transform("mean").fillna(0).astype("float32")
    return out

def fit_matcher(train_latent,test_latent,seed):
    feature_cols=[c for c in train_latent.columns if c not in META]
    xtr,xte,cats=prep(train_latent,test_latent,feature_cols)

    top3=lgb.LGBMClassifier(**binary_params(seed+1,CPU,260))
    top3.fit(xtr,train_latent["_is_top3"].to_numpy(),categorical_feature=cats)
    ptop3=np.asarray(top3.predict_proba(xte)[:,1],dtype=float)
    del top3
    gc.collect()

    order=np.lexsort((train_latent["_horse_id"].astype(str).to_numpy(),train_latent["_race_id"].astype(str).to_numpy()))
    tr_sorted=train_latent.iloc[order].reset_index(drop=True)
    xrank=xtr.iloc[order].reset_index(drop=True)
    rel=rank_relevance(tr_sorted["_finish"])
    group=tr_sorted.groupby("_race_id",sort=False).size().tolist()
    rank=lgb.LGBMRanker(**ranker_params(seed+2,CPU,260))
    rank.fit(xrank,rel,group=group,categorical_feature=cats)
    prank=np.asarray(rank.predict(xte),dtype=float)

    out=test_latent[list(META)].copy()
    out["LATENT_TOP3"]=ptop3
    out["LATENT_RANK"]=prank
    del xtr,xte,xrank,rank
    gc.collect()

    out["LATENT_BLEND"]=(within_race_pct(out,"LATENT_TOP3")+within_race_pct(out,"LATENT_RANK"))/2.0
    return out

def fit_direct_baseline(train,test,seed):
    cols=direct_columns(train)
    xtr,xte,cats=prep(train,test,cols)

    top3=lgb.LGBMClassifier(**binary_params(seed+1,CPU,260))
    top3.fit(xtr,train["_is_top3"].to_numpy(),categorical_feature=cats)
    ptop3=np.asarray(top3.predict_proba(xte)[:,1],dtype=float)
    del top3
    gc.collect()

    order=np.lexsort((train["_horse_id"].astype(str).to_numpy(),train["_race_id"].astype(str).to_numpy()))
    tr_sorted=train.iloc[order].reset_index(drop=True)
    xrank=xtr.iloc[order].reset_index(drop=True)
    rel=rank_relevance(tr_sorted["_finish"])
    group=tr_sorted.groupby("_race_id",sort=False).size().tolist()
    rank=lgb.LGBMRanker(**ranker_params(seed+2,CPU,260))
    rank.fit(xrank,rel,group=group,categorical_feature=cats)
    prank=np.asarray(rank.predict(xte),dtype=float)

    out=test[list(META)].copy()
    out["DIRECT_TOP3"]=ptop3
    out["DIRECT_RANK"]=prank
    out["DIRECT_BLEND"]=(within_race_pct(out,"DIRECT_TOP3")+within_race_pct(pd.DataFrame({**{m:out[m] for m in META},"DIRECT_RANK":prank}),"DIRECT_RANK"))/2.0
    del xtr,xte,xrank,rank
    gc.collect()
    return out,len(cols)

def within_race_pct(df,col):
    rank=df.groupby("_race_id")[col].rank(method="average",ascending=False)
    n=df.groupby("_race_id")[col].transform("size")
    return 1.0-(rank-1.0)/(n-1.0).clip(lower=1.0)

def metrics(pred,col):
    x=pred[["_race_id","_horse_id","_finish","_is_win","_is_top3",col]].copy()
    x=x.sort_values(["_race_id",col,"_horse_id"],ascending=[True,False,True])
    x["_rank"]=x.groupby("_race_id",sort=False).cumcount()+1
    top1=x[x["_rank"]==1]
    top3=x[x["_rank"]<=3]
    top6=x[x["_rank"]<=6]
    wins=x[x["_is_win"]==1]
    return {
        "races":int(x["_race_id"].nunique()),
        "top1_win_pct":100*float(top1["_is_win"].mean()),
        "top1_top3_pct":100*float(top1["_is_top3"].mean()),
        "winner_top3_capture_pct":100*float(top3.groupby("_race_id")["_is_win"].max().mean()),
        "winner_top6_capture_pct":100*float(top6.groupby("_race_id")["_is_win"].max().mean()),
        "top3_podium_precision_pct":100*float(top3["_is_top3"].mean()),
        "top6_podium_precision_pct":100*float(top6["_is_top3"].mean()),
        "mean_winner_rank":float(wins["_rank"].mean()),
        "winner_mrr":float((1.0/wins["_rank"]).mean()),
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
        raise SystemExit(f"year file mismatch: {sorted(paths)}")
    if LOCKED_YEAR in paths:
        raise SystemExit("2026 sealed")

    print(json.dumps({
        "contract":"L1_LATENT_ABILITY_V1_RUNTIME",
        "cpu":CPU,"component_workers":COMPONENT_WORKERS,"component_threads":COMPONENT_THREADS,
        "components":list(COMPONENTS),"test_years":list(TEST_YEARS),
        "ability_uses_odds":False,"2026_locked":True,
    },separators=(",",":")),flush=True)

    cache={}
    def get(y):
        if y not in cache:
            cache[y]=read_year(paths[y],y)
        return cache[y]

    rows=[]
    for test_year in TEST_YEARS:
        encoder_year=test_year-2
        matcher_year=test_year-1
        encoder_train=get(encoder_year)
        matcher_base=get(matcher_year)
        test_base=get(test_year)

        # Stage 1A: strict OOF latent vectors for matcher training.
        matcher_scores=fit_components_parallel(encoder_train,matcher_base,100000+test_year*100)

        # Stage 1B: refit encoder on all prior two years, then score unknown test year.
        prior2=pd.concat([encoder_train,matcher_base],ignore_index=True,copy=False)
        test_scores=fit_components_parallel(prior2,test_base,200000+test_year*100)

        matcher_latent=race_rank_features(matcher_base,matcher_scores)
        test_latent=race_rank_features(test_base,test_scores)

        # Stage 2: matcher only sees OOF latent vectors + compact race context.
        latent_pred=fit_matcher(matcher_latent,test_latent,300000+test_year*100)

        # Direct two-year baseline on the same data horizon.
        direct_pred,direct_features=fit_direct_baseline(prior2,test_base,400000+test_year*100)

        for model,pred,heads in (
            ("LATENT_ABILITY",latent_pred,("LATENT_TOP3","LATENT_RANK","LATENT_BLEND")),
            ("DIRECT",direct_pred,("DIRECT_TOP3","DIRECT_RANK","DIRECT_BLEND")),
        ):
            for head in heads:
                rows.append({
                    "test_year":test_year,
                    "encoder_train_year":encoder_year,
                    "matcher_train_year":matcher_year,
                    "model":model,
                    "head":head,
                    "direct_feature_count":direct_features if model=="DIRECT" else "",
                    **metrics(pred,head),
                })

        print(f"FOLD_COMPLETE test_year={test_year}",flush=True)
        del matcher_scores,test_scores,matcher_latent,test_latent,latent_pred,direct_pred,prior2
        for y in list(cache):
            if y < test_year-1:
                del cache[y]
        gc.collect()

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"fold-metrics.csv",rows)

    pooled=[]
    for model in ("LATENT_ABILITY","DIRECT"):
        for head in sorted({r["head"] for r in rows if r["model"]==model}):
            vals=[r for r in rows if r["model"]==model and r["head"]==head]
            weights=np.array([r["races"] for r in vals],dtype=float)
            rec={"model":model,"head":head,"folds":len(vals),"races":int(weights.sum())}
            for key in (
                "top1_win_pct","top1_top3_pct","winner_top3_capture_pct","winner_top6_capture_pct",
                "top3_podium_precision_pct","top6_podium_precision_pct","mean_winner_rank","winner_mrr",
            ):
                arr=np.array([r[key] for r in vals],dtype=float)
                rec[key]=float(np.average(arr,weights=weights))
                rec[key+"_worst"]=float(arr.max() if key=="mean_winner_rank" else arr.min())
                rec[key+"_std"]=float(arr.std(ddof=0))
            pooled.append(rec)
    write_csv(out/"pooled-metrics.csv",pooled)

    (out/"summary.json").write_text(json.dumps({
        "contract":"L1_LATENT_ABILITY_V1",
        "design":"supervised domain ability encoders -> race-relative latent vector -> race matcher",
        "components":list(COMPONENTS),
        "strict_fold":"Y-2 trains encoder; Y-1 OOF latent vectors train matcher; encoder refit on Y-2+Y-1 scores test Y; test labels never used for fitting or selection",
        "direct_baseline":"same Y-2+Y-1 horizon, direct TOP3 and LambdaRank on raw safe features",
        "uncertainty":"component disagreement and race-level latent spread are explicit matcher inputs",
        "primary_metric":"top1_top3_pct",
        "promotion_rule":"material multi-year gain required; sub-1pp improvement is not promotion-worthy",
        "ability_uses_odds":False,
        "2026_locked":True,
        "cpu":CPU,
        "component_parallelism":{"workers":COMPONENT_WORKERS,"threads_each":COMPONENT_THREADS},
        "promotion":False,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== POOLED =====")
    print((out/"pooled-metrics.csv").read_text())
    print("L1_LATENT_ABILITY_V1_COMPLETE")

if __name__=="__main__":
    main()
