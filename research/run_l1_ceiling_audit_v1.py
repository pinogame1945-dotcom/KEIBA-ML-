#!/usr/bin/env python3
import argparse,csv,json,gc,math,os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss

from run_l1_objective_rebuild_v1 import (
    YEARS,TEST_YEARS,LOCKED_YEAR,THREADS,META,
    read_year,select_variant,train_heads,metrics,
)

FULL_FORBIDDEN={
    "actual_start_time","jockey_id","trainer_id",
    "pedigree_sire_id","pedigree_dam_id","pedigree_siresire_id","pedigree_damsire_id",
    "finish_position","target_finish_position","finish_time_ms","target_finish_time_ms",
    "last_3f","target_last_3f","prize_money","target_prize_money",
    "final_win_odds","market_final_win_odds","final_popularity","market_final_popularity",
    "payout","payout_yen","market_payout","race_id","horse_id","owner_id","breeder_id",
}
BASE_CATEGORICAL={
    "venue_code","discipline","surface","direction","weather","track_condition","sex",
    "backfill_course_layout","backfill_race_class_normalized","backfill_grade",
    "backfill_sex_condition","backfill_weight_rule",
}
BASE_HEADS=("WIN_BINARY","TOP3_BINARY","RANK_GRADED")
VARIANTS=("core4","structural")
HEAD_COLS=tuple(f"{v}__{h}" for v in VARIANTS for h in BASE_HEADS)
PRIMARY="core4__TOP3_BINARY"

HISTORY_TOKENS=(
    "previous_","recent_","prior_","days_since","distance_change",
    "trend","ewma","rolling","form_","history_",
)
STRUCTURE_PREFIXES=(
    "opponent_","network_","lap_","style_","distx_","backfill_","timepace_",
    "auto_field_","auto_pair_",
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

def strip_forbidden(df):
    bad=[c for c in df.columns if c in FULL_FORBIDDEN]
    if bad:
        return df.drop(columns=bad)
    return df

def add_rank_features(master):
    x=master.copy()
    for col in HEAD_COLS:
        rank=x.groupby("_race_id")[col].rank(method="average",ascending=False)
        n=x.groupby("_race_id")[col].transform("size")
        x[f"rank__{col}"]=rank.astype("float32")
        x[f"pct__{col}"]=(1.0-(rank-1.0)/(n-1.0).clip(lower=1.0)).astype("float32")
    return x

def top1_votes(master):
    votes=[]
    margins={}
    for col in HEAD_COLS:
        ordered=master.sort_values(["_race_id",col,"_horse_id"],ascending=[True,False,True])
        top=ordered.groupby("_race_id",sort=False).head(1)[["_race_id","_horse_id"]].copy()
        top["head"]=col
        votes.append(top)

        def margin(g):
            vals=g[col].to_numpy(dtype=float)
            if len(vals)<2:
                return 0.0
            idx=np.argsort(-vals,kind="stable")
            return float(vals[idx[0]]-vals[idx[1]])
        margins[col]=ordered.groupby("_race_id",sort=False).apply(margin,include_groups=False)
    vote=pd.concat(votes,ignore_index=True)
    counts=vote.groupby(["_race_id","_horse_id"]).size().rename("vote_count").reset_index()
    unique=counts.groupby("_race_id").size().rename("n_unique_candidates")
    maxvote=counts.groupby("_race_id")["vote_count"].max().rename("max_vote_count")
    return vote,counts,unique,maxvote,margins

def numeric_feature_frame(valid,ids):
    feature_cols=[c for c in valid.columns if c not in META and c not in FULL_FORBIDDEN and c not in BASE_CATEGORICAL]
    out=valid[["_race_id","_horse_id",*feature_cols]].merge(ids,on=["_race_id","_horse_id"],how="inner")
    keep=["_race_id","_horse_id"]
    data={}
    for c in feature_cols:
        v=pd.to_numeric(out[c],errors="coerce")
        if v.notna().any():
            data["f_"+c]=v.astype("float32")
    result=out[["_race_id","_horse_id"]].copy()
    for c,v in data.items():
        result[c]=v.to_numpy()
    return result

def build_meta_rows(year,valid,master):
    master=add_rank_features(master)
    vote,counts,unique,maxvote,margins=top1_votes(master)

    top_by_head={}
    for col in HEAD_COLS:
        ordered=master.sort_values(["_race_id",col,"_horse_id"],ascending=[True,False,True])
        top_by_head[col]=ordered.groupby("_race_id",sort=False).head(1)[["_race_id","_horse_id"]].rename(columns={"_horse_id":f"top__{col}"})

    candidate_ids=counts[["_race_id","_horse_id"]].copy()
    cand=master.merge(candidate_ids,on=["_race_id","_horse_id"],how="inner")
    cand=cand.merge(counts,on=["_race_id","_horse_id"],how="left")
    cand=cand.merge(unique,on="_race_id",how="left")
    cand=cand.merge(maxvote,on="_race_id",how="left")
    cand["vote_fraction"]=(cand["vote_count"]/len(HEAD_COLS)).astype("float32")
    cand["max_vote_fraction"]=(cand["max_vote_count"]/len(HEAD_COLS)).astype("float32")
    cand["all_heads_same"]=(cand["n_unique_candidates"]==1).astype("int8")

    primary=top_by_head[PRIMARY].rename(columns={f"top__{PRIMARY}":"_horse_id"})
    cand["is_primary_candidate"]=(cand.set_index(["_race_id","_horse_id"]).index.isin(
        primary.set_index(["_race_id","_horse_id"]).index
    )).astype("int8")

    # Stable model-only feature names.
    rename={}
    for col in HEAD_COLS:
        rename[col]="m_score__"+col
        rename[f"rank__{col}"]="m_rank__"+col
        rename[f"pct__{col}"]="m_pct__"+col
    cand=cand.rename(columns=rename)
    cand=cand.rename(columns={
        "vote_count":"m_vote_count",
        "vote_fraction":"m_vote_fraction",
        "n_unique_candidates":"m_n_unique_candidates",
        "max_vote_count":"m_max_vote_count",
        "max_vote_fraction":"m_max_vote_fraction",
        "all_heads_same":"m_all_heads_same",
        "is_primary_candidate":"m_is_primary_candidate",
    })

    for col,series in margins.items():
        mp=series.rename("m_margin__"+col).reset_index()
        cand=cand.merge(mp,on="_race_id",how="left")

    fnum=numeric_feature_frame(valid,candidate_ids)
    cand=cand.merge(fnum,on=["_race_id","_horse_id"],how="left")
    cand["audit_year"]=int(year)

    primary_rows=cand[cand["m_is_primary_candidate"]==1].copy()
    if primary_rows["_race_id"].nunique()!=valid["_race_id"].nunique():
        raise SystemExit(f"primary row mismatch year={year}")

    primary_rows["collapse_target"]=(1-primary_rows["_is_top3"]).astype("int8")

    # Race-level complementarity.
    comp=primary_rows[["_race_id","_is_top3","_is_win"]].copy()
    comp=comp.rename(columns={"_is_top3":"baseline_hit","_is_win":"baseline_win"})
    oracle=cand.groupby("_race_id")["_is_top3"].max().rename("oracle_hit").reset_index()
    oracle_win=cand.groupby("_race_id")["_is_win"].max().rename("oracle_win").reset_index()
    comp=comp.merge(oracle,on="_race_id").merge(oracle_win,on="_race_id")
    comp=comp.merge(unique.rename("n_unique_candidates").reset_index(),on="_race_id")
    comp=comp.merge(maxvote.rename("max_vote_count").reset_index(),on="_race_id")
    comp["all_heads_same"]=(comp["n_unique_candidates"]==1).astype("int8")
    comp["unanimous_miss"]=(1-comp["oracle_hit"]).astype("int8")
    comp["test_year"]=int(year)

    return primary_rows,cand,comp

def head_metrics(master,year):
    rows=[]
    for col in HEAD_COLS:
        m=metrics(master,col)
        rows.append({"test_year":year,"head":col,**m})
    return rows

def pooled_head_metrics(rows):
    out=[]
    for head in HEAD_COLS:
        vals=[r for r in rows if r["head"]==head]
        w=np.array([r["races"] for r in vals],dtype=float)
        rec={"head":head,"folds":len(vals),"races":int(w.sum())}
        for key in ("top1_win_pct","top1_top3_pct","winner_top3_capture_pct","winner_top6_capture_pct",
                    "top3_podium_precision_pct","top6_podium_precision_pct","mean_winner_rank","winner_mrr"):
            arr=np.array([r[key] for r in vals],dtype=float)
            rec[key]=float(np.average(arr,weights=w))
            rec[key+"_worst"]=float(arr.max() if key=="mean_winner_rank" else arr.min())
            rec[key+"_std"]=float(arr.std(ddof=0))
        out.append(rec)
    return out

def group_columns(df,group):
    model=[c for c in df.columns if c.startswith("m_")]
    fcols=[c for c in df.columns if c.startswith("f_")]
    if group=="MODEL_ONLY":
        cols=model
    elif group=="HISTORY_AUTO":
        hist=[]
        for c in fcols:
            key=c[2:]
            if key.startswith("auto_") or any(tok in key for tok in HISTORY_TOKENS):
                hist.append(c)
        cols=model+hist
    elif group=="STRUCTURE":
        structural=[c for c in fcols if c[2:].startswith(STRUCTURE_PREFIXES)]
        cols=model+structural
    elif group=="ALL_NUMERIC":
        cols=model+fcols
    else:
        raise ValueError(group)
    return sorted(set(cols))

def clean_matrix(train,test,cols):
    usable=[]
    xtr={}
    xte={}
    for c in cols:
        a=pd.to_numeric(train[c],errors="coerce").replace([np.inf,-np.inf],np.nan).astype("float32")
        if not a.notna().any():
            continue
        b=pd.to_numeric(test[c],errors="coerce").replace([np.inf,-np.inf],np.nan).astype("float32")
        usable.append(c)
        xtr[c]=a
        xte[c]=b
    if not usable:
        raise SystemExit("no usable meta features")
    return pd.DataFrame(xtr),pd.DataFrame(xte),usable

def meta_params(seed):
    return dict(
        objective="binary",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=80,
        subsample=0.90,
        colsample_bytree=0.75,
        reg_lambda=10.0,
        reg_alpha=1.0,
        random_state=seed,
        n_jobs=THREADS,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )

def fit_meta(train,test,cols,target,seed):
    xtr,xte,usable=clean_matrix(train,test,cols)
    model=lgb.LGBMClassifier(**meta_params(seed))
    model.fit(xtr,train[target].to_numpy())
    p=np.asarray(model.predict_proba(xte)[:,1],dtype=float)
    gain=model.booster_.feature_importance(importance_type="gain")
    importance=sorted(zip(usable,gain),key=lambda z:z[1],reverse=True)
    del xtr,xte,model
    gc.collect()
    return p,importance

def safe_auc(y,p):
    y=np.asarray(y,dtype=int)
    return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else float("nan")

def risk_metrics(test,p):
    y=test["collapse_target"].to_numpy(dtype=int)
    n=len(test)
    q=max(1,int(math.ceil(n*0.20)))
    order=np.argsort(-p)
    hi=order[:q]
    lo=order[-q:]
    failures=max(1,int(y.sum()))
    return {
        "races":n,
        "base_collapse_pct":100*float(y.mean()),
        "auc":safe_auc(y,p),
        "brier":float(brier_score_loss(y,p)),
        "high20_collapse_pct":100*float(y[hi].mean()),
        "low20_collapse_pct":100*float(y[lo].mean()),
        "high20_failure_capture_pct":100*float(y[hi].sum()/failures),
    }

def strict_meta_audit(primary_by_year,candidate_by_year,comp_by_year):
    collapse_rows=[]
    selector_rows=[]
    importance_rows=[]
    groups=("MODEL_ONLY","HISTORY_AUTO","STRUCTURE","ALL_NUMERIC")

    for year in TEST_YEARS[1:]:
        prior=[y for y in TEST_YEARS if y<year]
        ptr=pd.concat([primary_by_year[y] for y in prior],ignore_index=True,copy=False)
        pte=primary_by_year[year].copy()
        ctr=pd.concat([candidate_by_year[y] for y in prior],ignore_index=True,copy=False)
        cte=candidate_by_year[year].copy()

        comp=comp_by_year[year]
        base_hit=100*float(comp["baseline_hit"].mean())
        oracle_hit=100*float(comp["oracle_hit"].mean())

        for gi,group in enumerate(groups):
            cols=group_columns(ptr,group)
            risk,imp=fit_meta(ptr,pte,cols,"collapse_target",500000+year*10+gi)
            collapse_rows.append({
                "test_year":year,
                "train_years":"|".join(map(str,prior)),
                "feature_group":group,
                "feature_count":len(cols),
                **risk_metrics(pte,risk),
            })
            for rank,(feature,gain) in enumerate(imp[:30],start=1):
                importance_rows.append({
                    "task":"COLLAPSE",
                    "test_year":year,
                    "feature_group":group,
                    "importance_rank":rank,
                    "feature":feature,
                    "gain":float(gain),
                })

        for gi,group in enumerate(("MODEL_ONLY","HISTORY_AUTO","ALL_NUMERIC")):
            cols=group_columns(ctr,group)
            prob,imp=fit_meta(ctr,cte,cols,"_is_top3",600000+year*10+gi)
            scored=cte[["_race_id","_horse_id","_is_top3"]].copy()
            scored["_p"]=prob
            chosen=scored.sort_values(["_race_id","_p","_horse_id"],ascending=[True,False,True]).groupby("_race_id",sort=False).head(1)
            hit=100*float(chosen["_is_top3"].mean())
            denom=oracle_hit-base_hit
            efficiency=(hit-base_hit)/denom if abs(denom)>1e-12 else 0.0
            selector_rows.append({
                "test_year":year,
                "train_years":"|".join(map(str,prior)),
                "feature_group":group,
                "feature_count":len(cols),
                "baseline_top1_top3_pct":base_hit,
                "selector_top1_top3_pct":hit,
                "selector_uplift_pp":hit-base_hit,
                "oracle_top1_top3_pct":oracle_hit,
                "oracle_uplift_pp":oracle_hit-base_hit,
                "oracle_gap_realized_pct":100*efficiency,
            })
            for rank,(feature,gain) in enumerate(imp[:30],start=1):
                importance_rows.append({
                    "task":"SELECTOR",
                    "test_year":year,
                    "feature_group":group,
                    "importance_rank":rank,
                    "feature":feature,
                    "gain":float(gain),
                })

    return collapse_rows,selector_rows,importance_rows

def complementarity_summary(comp_all):
    rows=[]
    for year,g in comp_all.groupby("test_year",sort=True):
        rows.append({
            "test_year":int(year),
            "races":len(g),
            "baseline_top1_top3_pct":100*float(g["baseline_hit"].mean()),
            "oracle_top1_top3_pct":100*float(g["oracle_hit"].mean()),
            "oracle_uplift_pp":100*float((g["oracle_hit"]-g["baseline_hit"]).mean()),
            "all_six_top1_miss_pct":100*float(g["unanimous_miss"].mean()),
            "all_heads_same_top1_pct":100*float(g["all_heads_same"].mean()),
            "avg_unique_top1_candidates":float(g["n_unique_candidates"].mean()),
            "max_vote_mean":float(g["max_vote_count"].mean()),
            "oracle_win_pct":100*float(g["oracle_win"].mean()),
        })
    pooled={
        "test_year":"POOLED",
        "races":len(comp_all),
        "baseline_top1_top3_pct":100*float(comp_all["baseline_hit"].mean()),
        "oracle_top1_top3_pct":100*float(comp_all["oracle_hit"].mean()),
        "oracle_uplift_pp":100*float((comp_all["oracle_hit"]-comp_all["baseline_hit"]).mean()),
        "all_six_top1_miss_pct":100*float(comp_all["unanimous_miss"].mean()),
        "all_heads_same_top1_pct":100*float(comp_all["all_heads_same"].mean()),
        "avg_unique_top1_candidates":float(comp_all["n_unique_candidates"].mean()),
        "max_vote_mean":float(comp_all["max_vote_count"].mean()),
        "oracle_win_pct":100*float(comp_all["oracle_win"].mean()),
    }
    rows.append(pooled)
    return rows

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
        "contract":"L1_CEILING_AUDIT_V1_RUNTIME",
        "threads":THREADS,
        "primary":PRIMARY,
        "heads":list(HEAD_COLS),
        "test_years":list(TEST_YEARS),
        "2026_locked":True,
        "ability_uses_odds":False,
    },separators=(",",":")),flush=True)

    cache={}
    def get(year):
        if year not in cache:
            cache[year]=strip_forbidden(read_year(paths[year],year))
        return cache[year]

    head_rows=[]
    primary_by_year={}
    candidate_by_year={}
    comp_by_year={}

    for test in TEST_YEARS:
        train_years=(test-2,test-1)
        train_union=pd.concat([get(y) for y in train_years],ignore_index=True,copy=False)
        valid_union=get(test).copy()
        master=valid_union[list(META)].copy()

        for vi,variant in enumerate(VARIANTS):
            train=select_variant(train_union,variant)
            valid=select_variant(valid_union,variant)
            pred=train_heads(train,valid,710000+test+vi*10000)
            keep=pred[["_race_id","_horse_id",*BASE_HEADS]].rename(
                columns={h:f"{variant}__{h}" for h in BASE_HEADS}
            )
            master=master.merge(keep,on=["_race_id","_horse_id"],how="left",validate="one_to_one")
            del train,valid,pred,keep
            gc.collect()

        if master[list(HEAD_COLS)].isna().any().any():
            raise SystemExit(f"missing head scores test={test}")

        head_rows.extend(head_metrics(master,test))
        primary,candidates,comp=build_meta_rows(test,valid_union,master)
        primary_by_year[test]=primary
        candidate_by_year[test]=candidates
        comp_by_year[test]=comp

        print(
            f"CEILING_FOLD_READY year={test} baseline={100*comp['baseline_hit'].mean():.3f} "
            f"oracle={100*comp['oracle_hit'].mean():.3f} unanimous_miss={100*comp['unanimous_miss'].mean():.3f} "
            f"avg_candidates={comp['n_unique_candidates'].mean():.3f}",
            flush=True,
        )

        del train_union,valid_union,master
        for year in list(cache):
            if year < test-1:
                del cache[year]
        gc.collect()

    comp_all=pd.concat([comp_by_year[y] for y in TEST_YEARS],ignore_index=True)
    comp_summary=complementarity_summary(comp_all)
    collapse_rows,selector_rows,importance_rows=strict_meta_audit(primary_by_year,candidate_by_year,comp_by_year)

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"head-fold-metrics.csv",head_rows)
    write_csv(out/"head-pooled-metrics.csv",pooled_head_metrics(head_rows))
    write_csv(out/"complementarity.csv",comp_summary)
    write_csv(out/"collapse-predictability.csv",collapse_rows)
    write_csv(out/"strict-selector.csv",selector_rows)
    write_csv(out/"meta-feature-importance.csv",importance_rows)

    pooled_comp=[r for r in comp_summary if r["test_year"]=="POOLED"][0]
    (out/"summary.json").write_text(json.dumps({
        "contract":"L1_CEILING_AUDIT_V1",
        "question":"Is the ~40% top1 podium miss mostly shared/information-limited, or is there exploitable pre-race complementarity left?",
        "primary_head":PRIMARY,
        "candidate_heads":list(HEAD_COLS),
        "diagnostics":{
            "oracle_among_six_top1s":True,
            "strict_past_oos_collapse_predictor":True,
            "strict_past_oos_candidate_selector":True,
            "feature_signal_groups":["MODEL_ONLY","HISTORY_AUTO","STRUCTURE","ALL_NUMERIC"],
        },
        "decision_thresholds":{
            "oracle_uplift_pp_material":3.0,
            "strict_selector_uplift_pp_material":1.0,
            "collapse_auc_useful":0.60,
            "history_auc_increment_useful":0.02,
        },
        "pooled_complementarity":pooled_comp,
        "notes":[
            "Oracle uses finish results only as a diagnostic upper bound and is not deployable.",
            "Meta collapse/selector tests for 2022-2025 train only on earlier outer-fold OOS years.",
            "No odds or popularity are model inputs.",
            "2026 is sealed.",
            "This audit is diagnostic only and cannot promote a production model."
        ],
        "ability_uses_odds":False,
        "2026_locked":True,
        "promotion":False,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== COMPLEMENTARITY =====")
    print((out/"complementarity.csv").read_text())
    print("===== COLLAPSE =====")
    print((out/"collapse-predictability.csv").read_text())
    print("===== SELECTOR =====")
    print((out/"strict-selector.csv").read_text())
    print("L1_CEILING_AUDIT_V1_COMPLETE")

if __name__=="__main__":
    main()
