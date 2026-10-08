#!/usr/bin/env python3
import argparse,csv,gc,json,math,os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l1_objective_rebuild_v1 import (
    YEARS,TEST_YEARS,LOCKED_YEAR,THREADS,META,BASE_CATEGORICAL,
    read_year,select_variant,prepare,model_common,
)
from run_l1_ceiling_audit_v1 import strip_forbidden
from run_l1_state_transition_v1 import build_state_features,attach_state

DIRECT_VARIANTS=("BASELINE_DIRECT","RESIDUAL_REGIME_DIRECT")
PAIR_VARIANTS=("FULL_PAIRWISE_BASE","FULL_PAIRWISE_REGIME")
PAIR_HEADS=("PAIR_FINISH","PAIR_TOP3","PAIR_BLEND")
RESIDUAL_AXES=("perf_resid","speed_resid","last3f_resid")
MAX_PAIR_BASE_FEATURES=96

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

def top1_metrics(df,score_col):
    x=df[["_race_id","_horse_id","_finish","_is_win","_is_top3",score_col]].copy()
    x["_score"]=pd.to_numeric(x[score_col],errors="coerce").fillna(-1e30)
    x=x.sort_values(
        ["_race_id","_score","_horse_id"],
        ascending=[True,False,True],
    )
    x["_rank"]=x.groupby("_race_id",sort=False).cumcount()+1
    top1=x[x["_rank"]==1]
    top3=x[x["_rank"]<=3]
    top6=x[x["_rank"]<=6]
    return {
        "races":int(top1["_race_id"].nunique()),
        "top1_top3_pct":100*float(top1["_is_top3"].mean()),
        "top1_win_pct":100*float(top1["_is_win"].mean()),
        "winner_top3_capture_pct":100*float(
            top3.groupby("_race_id")["_is_win"].max().mean()
        ),
        "winner_top6_capture_pct":100*float(
            top6.groupby("_race_id")["_is_win"].max().mean()
        ),
    }

def build_regime_features(v1_by_year):
    by_year={}
    cols=[]
    for year in YEARS:
        src=v1_by_year[year]
        out=src[["_race_id","_horse_id"]].copy()
        for axis in RESIDUAL_AXES:
            lags=[]
            for k in range(1,6):
                source=f"seq_lag{k}_{axis}"
                if source not in src.columns:
                    raise SystemExit(f"missing residual lag {source}")
                c=f"regime_{axis}_lag{k}"
                out[c]=pd.to_numeric(src[source],errors="coerce").astype("float32")
                lags.append(c)
                if c not in cols: cols.append(c)

            l1,l2,l3,l4,l5=[out[c] for c in lags]
            trio=pd.concat([l1,l2,l3],axis=1)
            five=pd.concat([l1,l2,l3,l4,l5],axis=1)
            valid3=trio.notna().all(axis=1)
            valid5=five.notna().all(axis=1)

            defs={
                f"regime_{axis}_mean3":trio.mean(axis=1),
                f"regime_{axis}_mean5":five.mean(axis=1),
                f"regime_{axis}_std3":trio.std(axis=1,ddof=0),
                f"regime_{axis}_std5":five.std(axis=1,ddof=0),
                f"regime_{axis}_slope3":(l1-l3)/2.0,
                f"regime_{axis}_slope5":(l1-l5)/4.0,
                f"regime_{axis}_accel":(l1-l2)-(l2-l3),
                f"regime_{axis}_shock1":l1-((l2+l3)/2.0),
                f"regime_{axis}_positive_count3":trio.gt(0).sum(axis=1).where(valid3,np.nan),
                f"regime_{axis}_positive_count5":five.gt(0).sum(axis=1).where(valid5,np.nan),
                f"regime_{axis}_uptrend3":((l1>l2)&(l2>l3)).where(valid3,np.nan).astype("float32"),
                f"regime_{axis}_downtrend3":((l1<l2)&(l2<l3)).where(valid3,np.nan).astype("float32"),
                f"regime_{axis}_persistent_positive3":((l1>0)&(l2>0)&(l3>0)).where(valid3,np.nan).astype("float32"),
                f"regime_{axis}_persistent_negative3":((l1<0)&(l2<0)&(l3<0)).where(valid3,np.nan).astype("float32"),
                f"regime_{axis}_rebound":((l2<l3)&(l1>l2)).where(valid3,np.nan).astype("float32"),
                f"regime_{axis}_fade":((l2>l3)&(l1<l2)).where(valid3,np.nan).astype("float32"),
            }
            for c,v in defs.items():
                out[c]=pd.to_numeric(v,errors="coerce").astype("float32")
                if c not in cols: cols.append(c)

            denom=(out[f"regime_{axis}_std3"].abs()+0.05)
            c=f"regime_{axis}_strength3"
            out[c]=(out[f"regime_{axis}_mean3"]/denom).clip(-10,10).astype("float32")
            if c not in cols: cols.append(c)

        by_year[year]=out
        print(
            f"REGIME_READY year={year} rows={len(out)} features={len(cols)} "
            f"perf_lag1_cov={100*out['regime_perf_resid_lag1'].notna().mean():.2f}",
            flush=True,
        )
    return by_year,cols

def fit_direct_rank(train,valid,seed):
    train=train.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    valid=valid.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    xtr,xva,cats=prepare(train,valid)
    rel=np.select(
        [
            train["_finish"].eq(1),train["_finish"].eq(2),train["_finish"].eq(3),
            train["_finish"].eq(4),train["_finish"].eq(5),train["_finish"].eq(6),
        ],
        [6,5,4,3,2,1],
        default=0,
    ).astype(int)
    group=train.groupby("_race_id",sort=False).size().tolist()
    model=lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0,1,3,7,15,31,63],
        **model_common(seed),
    )
    model.fit(xtr,rel,group=group,categorical_feature=cats)
    pred=np.asarray(model.predict(xva),dtype=float)
    out=valid[["_race_id","_horse_id","_finish","_is_win","_is_top3"]].copy()
    out["_score"]=pred
    gain=model.booster_.feature_importance(importance_type="gain")
    names=list(xtr.columns)
    imp=sorted(zip(names,gain),key=lambda z:z[1],reverse=True)
    del xtr,xva,model
    gc.collect()
    return out,imp

PAIR_NAME_HINTS=(
    "recent_","previous_","prior_","days_since_","distance_change_",
    "same_","opponent_","network_","auto_","actor_","timepace_",
    "history_","body_weight","carried_weight","horse_number","frame_number",
    "draw","age",
)

def select_pair_base_cols(train):
    candidates=[
        c for c in train.columns
        if c not in META
        and c not in BASE_CATEGORICAL
        and not c.startswith("regime_")
        and any(c.startswith(h) or c==h for h in PAIR_NAME_HINTS)
    ]
    scored=[]
    n=len(train)
    for c in candidates:
        v=pd.to_numeric(train[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
        present=int(v.notna().sum())
        if present < max(500,int(0.05*n)):
            continue
        if v.nunique(dropna=True)<=1:
            continue
        scored.append((present/n,c))
    scored.sort(key=lambda x:(-x[0],x[1]))
    cols=[c for _,c in scored[:MAX_PAIR_BASE_FEATURES]]
    if len(cols)<20:
        raise SystemExit(f"too few pairwise base features: {len(cols)}")
    return cols

def clean_numeric(df,cols):
    out=np.empty((len(df),len(cols)),dtype=np.float32)
    for j,c in enumerate(cols):
        out[:,j]=pd.to_numeric(df[c],errors="coerce").to_numpy(dtype=np.float32)
    return out

def make_pairs(df):
    # Sort by horse_id, not source/result row order. Alternate pair direction
    # deterministically so A/B orientation cannot encode finish-table order.
    ia=[]; ib=[]
    for _,grp in df.groupby("_race_id",sort=False):
        ids=grp.sort_values("_horse_id").index.to_numpy(dtype=np.int64)
        m=len(ids)
        z=0
        for i in range(m-1):
            for j in range(i+1,m):
                a,b=int(ids[i]),int(ids[j])
                if z & 1:
                    a,b=b,a
                ia.append(a); ib.append(b); z+=1
    return np.asarray(ia,dtype=np.int64),np.asarray(ib,dtype=np.int64)

def pair_matrix(base,ia,ib):
    return (base[ia]-base[ib]).astype(np.float32,copy=False)

def pair_params(seed):
    return dict(
        objective="binary",
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.90,
        colsample_bytree=0.82,
        reg_lambda=10.0,
        reg_alpha=1.0,
        random_state=seed,
        n_jobs=THREADS,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )

def tournament_scores(n,ia,ib,p):
    score=np.zeros(n,dtype=np.float64)
    games=np.zeros(n,dtype=np.int32)
    np.add.at(score,ia,p)
    np.add.at(score,ib,1.0-p)
    np.add.at(games,ia,1)
    np.add.at(games,ib,1)
    ok=games>0
    score[ok]/=games[ok]
    score[~ok]=0.5
    return score

def fit_full_pairwise(train,test,cols,seed):
    train=train.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    test=test.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    tia,tib=make_pairs(train)
    via,vib=make_pairs(test)
    btr=clean_numeric(train,cols)
    bva=clean_numeric(test,cols)
    xtr=pair_matrix(btr,tia,tib)
    xva=pair_matrix(bva,via,vib)

    fa=train.loc[tia,"_finish"].to_numpy(dtype=np.int16)
    fb=train.loc[tib,"_finish"].to_numpy(dtype=np.int16)
    finish_mask=fa!=fb
    yfinish=(fa[finish_mask]<fb[finish_mask]).astype(np.int8)

    ta=train.loc[tia,"_is_top3"].to_numpy(dtype=np.int8)
    tb=train.loc[tib,"_is_top3"].to_numpy(dtype=np.int8)
    top_mask=ta!=tb
    ytop=(ta[top_mask]>tb[top_mask]).astype(np.int8)

    finish=lgb.LGBMClassifier(**pair_params(seed+1))
    finish.fit(xtr[finish_mask],yfinish)
    pfinish=np.asarray(finish.predict_proba(xva)[:,1],dtype=float)
    finish_gain=finish.booster_.feature_importance(importance_type="gain")
    del finish
    gc.collect()

    top=lgb.LGBMClassifier(**pair_params(seed+2))
    top.fit(xtr[top_mask],ytop)
    ptop=np.asarray(top.predict_proba(xva)[:,1],dtype=float)
    top_gain=top.booster_.feature_importance(importance_type="gain")
    del top,xtr,xva,btr,bva
    gc.collect()

    scores={
        "PAIR_FINISH":tournament_scores(len(test),via,vib,pfinish),
        "PAIR_TOP3":tournament_scores(len(test),via,vib,ptop),
    }
    scores["PAIR_BLEND"]=(scores["PAIR_FINISH"]+scores["PAIR_TOP3"])/2.0
    imp={
        "PAIR_FINISH":sorted(zip(cols,finish_gain),key=lambda z:z[1],reverse=True),
        "PAIR_TOP3":sorted(zip(cols,top_gain),key=lambda z:z[1],reverse=True),
    }
    return test,scores,imp,int(finish_mask.sum()),int(top_mask.sum()),len(tia),len(via)

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
        "contract":"L1_RESIDUAL_REGIME_FULL_PAIRWISE_V1_RUNTIME",
        "architectures":[
            "BASELINE_DIRECT","RESIDUAL_REGIME_DIRECT",
            "FULL_PAIRWISE_BASE","FULL_PAIRWISE_REGIME",
        ],
        "pair_heads":list(PAIR_HEADS),
        "full_field_pairwise":True,
        "pair_orientation":"horse_id sorted + deterministic alternating direction; source order prohibited",
        "threads":THREADS,
        "ability_uses_odds":False,
        "2026_locked":True,
    },separators=(",",":")),flush=True)

    v1_by_year,_,_,_,coverage=build_state_features(paths)
    regime_by_year,regime_cols=build_regime_features(v1_by_year)

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
    feature_rows=[]

    for test in TEST_YEARS:
        train_years=(test-2,test-1)
        train_base=pd.concat([get_base(y) for y in train_years],ignore_index=True,copy=False)
        valid_base=get_base(test).copy()
        train_reg=pd.concat([regime_by_year[y] for y in train_years],ignore_index=True,copy=False)
        valid_reg=regime_by_year[test]

        train_full=attach_state(train_base,train_reg,regime_cols)
        valid_full=attach_state(valid_base,valid_reg,regime_cols)

        # Direct baseline vs explicit residual-regime decomposition.
        for vi,(arch,tr,va) in enumerate((
            ("BASELINE_DIRECT",train_base,valid_base),
            ("RESIDUAL_REGIME_DIRECT",train_full,valid_full),
        )):
            pred,imp=fit_direct_rank(tr,va,2600000+test*100+vi*10)
            rows.append({
                "test_year":test,"train_years":"|".join(map(str,train_years)),
                "architecture":arch,"head":"RANK_GRADED",
                **top1_metrics(pred,"_score"),
            })
            for rank,(feature,gain) in enumerate(imp[:50],start=1):
                importance.append({
                    "test_year":test,"architecture":arch,"head":"RANK_GRADED",
                    "importance_rank":rank,"feature":feature,"gain":float(gain),
                })
            del pred
            gc.collect()

        pair_base_cols=select_pair_base_cols(train_base)
        pair_sets=(
            ("FULL_PAIRWISE_BASE",train_base,valid_base,pair_base_cols),
            ("FULL_PAIRWISE_REGIME",train_full,valid_full,pair_base_cols+regime_cols),
        )
        for pi,(arch,tr,va,cols) in enumerate(pair_sets):
            test_frame,scores,imp,n_finish,n_top,n_train_pairs,n_test_pairs=fit_full_pairwise(
                tr,va,cols,2700000+test*100+pi*10
            )
            feature_rows.append({
                "test_year":test,"architecture":arch,
                "base_feature_count":len(pair_base_cols),
                "regime_feature_count":0 if arch=="FULL_PAIRWISE_BASE" else len(regime_cols),
                "total_feature_count":len(cols),
                "train_pairs":n_train_pairs,"test_pairs":n_test_pairs,
                "train_finish_pairs":n_finish,"train_top3_pairs":n_top,
            })
            for head in PAIR_HEADS:
                x=test_frame[["_race_id","_horse_id","_finish","_is_win","_is_top3"]].copy()
                x["_score"]=scores[head]
                rows.append({
                    "test_year":test,"train_years":"|".join(map(str,train_years)),
                    "architecture":arch,"head":head,
                    **top1_metrics(x,"_score"),
                })
            for head in ("PAIR_FINISH","PAIR_TOP3"):
                for rank,(feature,gain) in enumerate(imp[head][:50],start=1):
                    importance.append({
                        "test_year":test,"architecture":arch,"head":head,
                        "importance_rank":rank,"feature":feature,"gain":float(gain),
                    })
            print(
                f"FULL_PAIR_READY year={test} arch={arch} features={len(cols)} "
                f"train_pairs={n_train_pairs} test_pairs={n_test_pairs}",
                flush=True,
            )
            del test_frame,scores
            gc.collect()

        del train_base,valid_base,train_reg,valid_reg,train_full,valid_full
        for old in list(base_cache):
            if old < test-1:
                del base_cache[old]
        gc.collect()

    pooled_rows=add_uplift(pooled(rows),rows)
    primary_keys=[
        ("RESIDUAL_REGIME_DIRECT","RANK_GRADED"),
        ("FULL_PAIRWISE_BASE","PAIR_BLEND"),
        ("FULL_PAIRWISE_REGIME","PAIR_BLEND"),
    ]
    primary=[r for r in pooled_rows if (r["architecture"],r["head"]) in primary_keys]
    best=max(primary,key=lambda r:(r["top1_top3_uplift_pp"],r["top1_top3_pct"]))

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"fold-metrics.csv",rows)
    write_csv(out/"pooled-metrics.csv",pooled_rows)
    write_csv(out/"feature-importance.csv",importance)
    write_csv(out/"pair-counts.csv",feature_rows)
    write_csv(out/"state-coverage.csv",coverage)

    (out/"summary.json").write_text(json.dumps({
        "contract":"L1_RESIDUAL_REGIME_FULL_PAIRWISE_V1",
        "question":"Do explicit residual-regime decomposition and direct full-field horse-vs-horse comparison jointly break the L1 top1 podium ceiling?",
        "architectures":{
            "BASELINE_DIRECT":"core4 direct LambdaRank",
            "RESIDUAL_REGIME_DIRECT":"core4 + decomposed V1 residual regimes, direct LambdaRank",
            "FULL_PAIRWISE_BASE":"all horses compared pair-by-pair from compact pre-race horse features",
            "FULL_PAIRWISE_REGIME":"all horses compared pair-by-pair with the same residual-regime state added",
        },
        "residual_regime":{
            "axes":list(RESIDUAL_AXES),
            "patterns":["lag1..5","mean3/5","std3/5","slope3/5","acceleration","shock","positive counts","uptrend","downtrend","persistent positive/negative","rebound","fade","standardized strength"],
        },
        "pairwise":{
            "coverage":"all unordered horse pairs in every race; 18 horses = 153 pair comparisons",
            "targets":{
                "PAIR_FINISH":"A finishes ahead of B for all non-dead-heat train pairs",
                "PAIR_TOP3":"A is podium and B is not, trained only across the podium boundary",
                "PAIR_BLEND":"fixed 50/50 average of tournament scores",
            },
            "ranking":"mean predicted pairwise win probability across all opponents",
            "anti_row_leak":"pair orientation comes from horse_id sort and deterministic alternating direction, never result/source row order",
            "base_feature_selection":"train-only, label-free coverage filter; maximum 96 compact horse-level numeric features",
        },
        "strictness":[
            "Every history/regime feature is shifted by >=1 horse race date.",
            "Pairwise A/B orientation is independent of finish-table row order.",
            "Two prior calendar years train each next unknown year, 2021-2025.",
            "No odds or popularity are used.",
            "2026 is sealed.",
        ],
        "primary_comparison_keys":[list(x) for x in primary_keys],
        "best_primary":best,
        "promotion_rule":"Require >=2.0pp pooled top1_top3 uplift vs baseline direct and no fold worse than -0.5pp.",
        "promotion":bool(best["top1_top3_uplift_pp"]>=2.0 and best["top1_top3_uplift_worst_fold_pp"]>=-0.5),
        "ability_uses_odds":False,
        "2026_locked":True,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== POOLED =====")
    print((out/"pooled-metrics.csv").read_text())
    print("===== PAIR COUNTS =====")
    print((out/"pair-counts.csv").read_text())
    print("L1_RESIDUAL_REGIME_FULL_PAIRWISE_V1_COMPLETE")

if __name__=="__main__":
    main()
