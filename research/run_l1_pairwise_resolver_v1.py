#!/usr/bin/env python3
import argparse,csv,gc,itertools,json,math,os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l1_objective_rebuild_v1 import (
    YEARS,TEST_YEARS,LOCKED_YEAR,THREADS,META,
    read_year,select_variant,train_heads,
)
from run_l1_ceiling_audit_v1 import (
    FULL_FORBIDDEN,BASE_HEADS,VARIANTS,HEAD_COLS,PRIMARY,
    strip_forbidden,build_meta_rows,group_columns,
)

PAIR_TEST_YEARS=(2022,2023,2024,2025)
FEATURE_GROUPS=("MODEL_ONLY","HISTORY_AUTO","STRUCTURE","ALL_NUMERIC")
PAIR_HEADS=("PAIR_TOP3","PAIR_UTILITY","PAIR_BLEND")
PRIMARY_SCORE="m_score__"+PRIMARY

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

def clean_candidate_frame(df,cols):
    out={}
    usable=[]
    for c in cols:
        if c not in df.columns:
            continue
        v=pd.to_numeric(df[c],errors="coerce").replace([np.inf,-np.inf],np.nan).astype("float32")
        if v.notna().any():
            out[c]=v
            usable.append(c)
    if not usable:
        raise SystemExit("no usable pairwise candidate features")
    return pd.DataFrame(out,index=df.index),usable

def make_pair_indices(df):
    ia=[]; ib=[]
    for _,idx in df.groupby("_race_id",sort=False).groups.items():
        ids=list(idx)
        if len(ids)<2:
            continue
        for a,b in itertools.combinations(ids,2):
            ia.append(a); ib.append(b)
    return np.asarray(ia,dtype=np.int64),np.asarray(ib,dtype=np.int64)

def pair_matrix(df,cols,ia,ib):
    base,usable=clean_candidate_frame(df,cols)
    a=base.loc[ia,usable].to_numpy(dtype=np.float32,copy=False)
    b=base.loc[ib,usable].to_numpy(dtype=np.float32,copy=False)
    diff=a-b

    # Mean of model outputs preserves absolute confidence while diff encodes A-vs-B.
    model_cols=[c for c in usable if c.startswith("m_")]
    if model_cols:
        ma=base.loc[ia,model_cols].to_numpy(dtype=np.float32,copy=False)
        mb=base.loc[ib,model_cols].to_numpy(dtype=np.float32,copy=False)
        mean=(ma+mb)/2.0
        x=np.concatenate([diff,mean],axis=1)
        names=["d__"+c for c in usable]+["mean__"+c for c in model_cols]
    else:
        x=diff
        names=["d__"+c for c in usable]
    return x,names

def augment_symmetric(x,y,n_diff):
    xr=x.copy()
    xr[:,:n_diff]*=-1.0
    return np.concatenate([x,xr],axis=0),np.concatenate([y,1-y],axis=0)

def pair_params(seed):
    return dict(
        objective="binary",
        n_estimators=240,
        learning_rate=0.035,
        num_leaves=23,
        min_child_samples=60,
        subsample=0.90,
        colsample_bytree=0.78,
        reg_lambda=10.0,
        reg_alpha=1.0,
        random_state=seed,
        n_jobs=THREADS,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )

def fit_pair_models(train,test,cols,seed):
    train=train.reset_index(drop=True)
    test=test.reset_index(drop=True)
    tia,tib=make_pair_indices(train)
    via,vib=make_pair_indices(test)
    if len(tia)==0 or len(via)==0:
        raise SystemExit("pairwise candidates missing")

    xtr,names=pair_matrix(train,cols,tia,tib)
    xva,_=pair_matrix(test,cols,via,vib)
    n_diff=sum(1 for n in names if n.startswith("d__"))

    top3a=train.loc[tia,"_is_top3"].to_numpy(dtype=np.int8)
    top3b=train.loc[tib,"_is_top3"].to_numpy(dtype=np.int8)
    top3mask=top3a!=top3b
    ytop=(top3a[top3mask]>top3b[top3mask]).astype(np.int8)
    xtop=xtr[top3mask]
    xtop,ytop=augment_symmetric(xtop,ytop,n_diff)

    fa=train.loc[tia,"_finish"].to_numpy(dtype=np.int16)
    fb=train.loc[tib,"_finish"].to_numpy(dtype=np.int16)
    # Podium status dominates; finish order is only a tie-break inside the same podium class.
    ua=top3a.astype(np.int16)*100-fa
    ub=top3b.astype(np.int16)*100-fb
    utilmask=ua!=ub
    yutil=(ua[utilmask]>ub[utilmask]).astype(np.int8)
    xutil=xtr[utilmask]
    xutil,yutil=augment_symmetric(xutil,yutil,n_diff)

    if len(np.unique(ytop))<2 or len(np.unique(yutil))<2:
        raise SystemExit("pairwise target collapsed")

    top=lgb.LGBMClassifier(**pair_params(seed+1))
    top.fit(xtop,ytop)
    ptop=np.asarray(top.predict_proba(xva)[:,1],dtype=float)
    top_gain=top.booster_.feature_importance(importance_type="gain")
    del top,xtop,ytop
    gc.collect()

    util=lgb.LGBMClassifier(**pair_params(seed+2))
    util.fit(xutil,yutil)
    putil=np.asarray(util.predict_proba(xva)[:,1],dtype=float)
    util_gain=util.booster_.feature_importance(importance_type="gain")
    del util,xutil,yutil,xtr,xva
    gc.collect()

    imp_top=sorted(zip(names,top_gain),key=lambda z:z[1],reverse=True)
    imp_util=sorted(zip(names,util_gain),key=lambda z:z[1],reverse=True)
    return (via,vib,ptop,putil,names,imp_top,imp_util,int(top3mask.sum()),int(utilmask.sum()))

def tournament_scores(cand,ia,ib,p):
    score=np.zeros(len(cand),dtype=np.float64)
    games=np.zeros(len(cand),dtype=np.int32)
    for a,b,pa in zip(ia,ib,p):
        score[a]+=pa
        score[b]+=1.0-pa
        games[a]+=1
        games[b]+=1
    nonzero=games>0
    score[nonzero]/=games[nonzero]
    score[~nonzero]=0.5
    return score

def choose(cand,score):
    x=cand[["_race_id","_horse_id","_is_top3","_is_win","_finish",PRIMARY_SCORE]].copy()
    x["_resolver"]=score
    chosen=x.sort_values(
        ["_race_id","_resolver",PRIMARY_SCORE,"_horse_id"],
        ascending=[True,False,False,True],
    ).groupby("_race_id",sort=False).head(1)
    return chosen

def baseline_choice(cand):
    x=cand[cand["m_is_primary_candidate"]==1][
        ["_race_id","_horse_id","_is_top3","_is_win","_finish",PRIMARY_SCORE]
    ].copy()
    if x["_race_id"].nunique()!=cand["_race_id"].nunique():
        raise SystemExit("baseline coverage mismatch")
    return x

def vote_choice(cand):
    x=cand[["_race_id","_horse_id","_is_top3","_is_win","_finish",PRIMARY_SCORE,"m_vote_count"]].copy()
    return x.sort_values(
        ["_race_id","m_vote_count",PRIMARY_SCORE,"_horse_id"],
        ascending=[True,False,False,True],
    ).groupby("_race_id",sort=False).head(1)

def choice_metrics(chosen,baseline,cand):
    base=baseline.set_index("_race_id")
    sel=chosen.set_index("_race_id")
    common=base.index.intersection(sel.index)
    base=base.loc[common]
    sel=sel.loc[common]
    changed=base["_horse_id"].ne(sel["_horse_id"])
    b_hit=base["_is_top3"].astype(int)
    s_hit=sel["_is_top3"].astype(int)
    rescue=((b_hit==0)&(s_hit==1))
    damage=((b_hit==1)&(s_hit==0))

    nunique=cand.groupby("_race_id").size()
    disagree=nunique[nunique>1].index
    dcommon=common.intersection(disagree)
    db=base.loc[dcommon]
    ds=sel.loc[dcommon]

    return {
        "races":int(len(common)),
        "top1_top3_pct":100*float(s_hit.mean()),
        "top1_win_pct":100*float(sel["_is_win"].mean()),
        "uplift_pp":100*float((s_hit-b_hit).mean()),
        "changed_pct":100*float(changed.mean()),
        "rescued_baseline_miss_pct":100*float(rescue.sum()/max(1,int((b_hit==0).sum()))),
        "damaged_baseline_hit_pct":100*float(damage.sum()/max(1,int((b_hit==1).sum()))),
        "disagreement_races":int(len(dcommon)),
        "disagreement_top3_pct":100*float(ds["_is_top3"].mean()) if len(ds) else float("nan"),
        "disagreement_baseline_top3_pct":100*float(db["_is_top3"].mean()) if len(db) else float("nan"),
        "disagreement_uplift_pp":100*float((ds["_is_top3"].astype(int)-db["_is_top3"].astype(int)).mean()) if len(ds) else float("nan"),
    }

def oracle_choice(cand):
    x=cand.copy()
    x["_oracle_utility"]=x["_is_top3"].astype(int)*100-x["_finish"].astype(float)
    return x.sort_values(
        ["_race_id","_oracle_utility",PRIMARY_SCORE,"_horse_id"],
        ascending=[True,False,False,True],
    ).groupby("_race_id",sort=False).head(1)

def build_base_oos(paths):
    cache={}
    def get(year):
        if year not in cache:
            cache[year]=strip_forbidden(read_year(paths[year],year))
        return cache[year]

    candidates={}
    complementarity={}
    for test in TEST_YEARS:
        train_years=(test-2,test-1)
        train_union=pd.concat([get(y) for y in train_years],ignore_index=True,copy=False)
        valid_union=get(test).copy()
        master=valid_union[list(META)].copy()

        for vi,variant in enumerate(VARIANTS):
            train=select_variant(train_union,variant)
            valid=select_variant(valid_union,variant)
            pred=train_heads(train,valid,810000+test+vi*10000)
            keep=pred[["_race_id","_horse_id",*BASE_HEADS]].rename(
                columns={h:f"{variant}__{h}" for h in BASE_HEADS}
            )
            master=master.merge(keep,on=["_race_id","_horse_id"],how="left",validate="one_to_one")
            del train,valid,pred,keep
            gc.collect()

        if master[list(HEAD_COLS)].isna().any().any():
            raise SystemExit(f"missing head scores test={test}")

        _,cand,comp=build_meta_rows(test,valid_union,master)
        cand=cand.reset_index(drop=True)
        candidates[test]=cand
        complementarity[test]=comp
        print(
            f"PAIR_BASE_READY year={test} races={cand['_race_id'].nunique()} "
            f"candidate_rows={len(cand)} disagree={(cand.groupby('_race_id').size()>1).sum()}",
            flush=True,
        )

        del train_union,valid_union,master
        for year in list(cache):
            if year < test-1:
                del cache[year]
        gc.collect()
    return candidates,complementarity

def aggregate_strategies(rows):
    out=[]
    keys=sorted({(r["feature_group"],r["resolver"]) for r in rows})
    for group,resolver in keys:
        vals=[r for r in rows if r["feature_group"]==group and r["resolver"]==resolver]
        w=np.array([r["races"] for r in vals],dtype=float)
        rec={"feature_group":group,"resolver":resolver,"folds":len(vals),"races":int(w.sum())}
        for key in (
            "top1_top3_pct","top1_win_pct","uplift_pp","changed_pct",
            "rescued_baseline_miss_pct","damaged_baseline_hit_pct",
            "disagreement_top3_pct","disagreement_baseline_top3_pct","disagreement_uplift_pp",
        ):
            arr=np.array([r[key] for r in vals],dtype=float)
            rec[key]=float(np.average(arr,weights=w))
            rec[key+"_worst"]=float(np.nanmin(arr))
            rec[key+"_std"]=float(np.nanstd(arr))
        out.append(rec)
    return out

def strict_policy(strategy_rows):
    # 2022 has no prior resolver OOS year, so use the predeclared conservative default.
    default=("MODEL_ONLY","PAIR_BLEND")
    rows=[]
    history=[]
    for year in PAIR_TEST_YEARS:
        current=[r for r in strategy_rows if r["test_year"]==year]
        if year==PAIR_TEST_YEARS[0] or not history:
            chosen_key=default
            reason="PREDECLARED_DEFAULT"
        else:
            perf={}
            for r in history:
                key=(r["feature_group"],r["resolver"])
                perf.setdefault(key,[]).append(r["uplift_pp"])
            chosen_key=max(
                perf,
                key=lambda k:(float(np.mean(perf[k])), -float(np.std(perf[k])), k==default, k),
            )
            reason="BEST_PRIOR_OOS_MEAN"
        match=[r for r in current if (r["feature_group"],r["resolver"])==chosen_key]
        if len(match)!=1:
            raise SystemExit(f"strict policy strategy missing year={year} key={chosen_key}")
        rec=dict(match[0])
        rec["selection_reason"]=reason
        rows.append(rec)
        history.extend(current)
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
        "contract":"L1_PAIRWISE_RESOLVER_V1_RUNTIME",
        "threads":THREADS,
        "primary":PRIMARY,
        "candidate_heads":list(HEAD_COLS),
        "feature_groups":list(FEATURE_GROUPS),
        "pair_heads":list(PAIR_HEADS),
        "test_years":list(PAIR_TEST_YEARS),
        "2026_locked":True,
        "ability_uses_odds":False,
    },separators=(",",":")),flush=True)

    candidates,complementarity=build_base_oos(paths)
    strategy_rows=[]
    importance_rows=[]

    for year in PAIR_TEST_YEARS:
        prior=[y for y in TEST_YEARS if y<year]
        train=pd.concat([candidates[y] for y in prior],ignore_index=True,copy=False).reset_index(drop=True)
        test=candidates[year].copy().reset_index(drop=True)
        baseline=baseline_choice(test)
        vote=vote_choice(test)
        oracle=oracle_choice(test)

        strategy_rows.append({
            "test_year":year,"train_years":"|".join(map(str,prior)),
            "feature_group":"BASELINE","resolver":"PRIMARY",
            **choice_metrics(baseline,baseline,test),
        })
        strategy_rows.append({
            "test_year":year,"train_years":"|".join(map(str,prior)),
            "feature_group":"BASELINE","resolver":"VOTE_COUNT",
            **choice_metrics(vote,baseline,test),
        })
        strategy_rows.append({
            "test_year":year,"train_years":"|".join(map(str,prior)),
            "feature_group":"ORACLE","resolver":"ORACLE",
            **choice_metrics(oracle,baseline,test),
        })

        for gi,group in enumerate(FEATURE_GROUPS):
            cols=group_columns(train,group)
            via,vib,ptop,putil,names,imp_top,imp_util,n_top_pairs,n_util_pairs=fit_pair_models(
                train,test,cols,910000+year*100+gi*10
            )
            scores={
                "PAIR_TOP3":tournament_scores(test,via,vib,ptop),
                "PAIR_UTILITY":tournament_scores(test,via,vib,putil),
            }
            scores["PAIR_BLEND"]=(scores["PAIR_TOP3"]+scores["PAIR_UTILITY"])/2.0

            for resolver in PAIR_HEADS:
                chosen=choose(test,scores[resolver])
                strategy_rows.append({
                    "test_year":year,
                    "train_years":"|".join(map(str,prior)),
                    "feature_group":group,
                    "resolver":resolver,
                    "feature_count":len(cols),
                    "train_top3_pairs":n_top_pairs,
                    "train_utility_pairs":n_util_pairs,
                    **choice_metrics(chosen,baseline,test),
                })

            for task,imp in (("PAIR_TOP3",imp_top),("PAIR_UTILITY",imp_util)):
                for rank,(feature,gain) in enumerate(imp[:40],start=1):
                    importance_rows.append({
                        "test_year":year,"feature_group":group,"task":task,
                        "importance_rank":rank,"feature":feature,"gain":float(gain),
                    })
            print(
                f"PAIR_FOLD_READY year={year} group={group} features={len(cols)} "
                f"top3_pairs={n_top_pairs} utility_pairs={n_util_pairs}",
                flush=True,
            )
            gc.collect()

    comparable=[r for r in strategy_rows if r["feature_group"] in FEATURE_GROUPS]
    pooled=aggregate_strategies(comparable)
    policy=strict_policy(comparable)

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"fold-strategies.csv",strategy_rows)
    write_csv(out/"pooled-strategies.csv",pooled)
    write_csv(out/"strict-policy.csv",policy)
    write_csv(out/"pair-feature-importance.csv",importance_rows)

    policy_races=sum(r["races"] for r in policy)
    policy_top3=sum(r["top1_top3_pct"]*r["races"] for r in policy)/policy_races
    policy_uplift=sum(r["uplift_pp"]*r["races"] for r in policy)/policy_races
    policy_disagree=sum(r["disagreement_uplift_pp"]*r["disagreement_races"] for r in policy)/sum(r["disagreement_races"] for r in policy)

    best=max(pooled,key=lambda r:(r["uplift_pp"],r["top1_top3_pct"]))
    comp=pd.concat([complementarity[y] for y in PAIR_TEST_YEARS],ignore_index=True)
    oracle_top3=100*float(comp["oracle_hit"].mean())
    baseline_top3=100*float(comp["baseline_hit"].mean())

    (out/"summary.json").write_text(json.dumps({
        "contract":"L1_PAIRWISE_RESOLVER_V1",
        "question":"Can direct A-vs-B learning resolve disagreement among six strong L1 top1 candidates?",
        "primary_anchor":PRIMARY,
        "test_years":list(PAIR_TEST_YEARS),
        "candidate_heads":list(HEAD_COLS),
        "pairwise_targets":{
            "PAIR_TOP3":"train only pairs where podium status differs",
            "PAIR_UTILITY":"podium status dominates; finish order breaks same-class ties",
            "PAIR_BLEND":"fixed 50/50 tournament score blend",
        },
        "strictness":[
            "Base L1 candidate scores are outer-fold OOS.",
            "Resolver for year Y trains only on earlier outer-fold OOS years.",
            "Strict policy chooses strategy only from prior resolver OOS performance; 2022 uses a predeclared default.",
            "No odds or popularity are inputs.",
            "2026 is sealed."
        ],
        "diagnostic_oracle":{"baseline_top1_top3_pct":baseline_top3,"oracle_top1_top3_pct":oracle_top3},
        "exploratory_best_strategy":best,
        "strict_policy":{
            "races":policy_races,
            "top1_top3_pct":policy_top3,
            "uplift_pp":policy_uplift,
            "disagreement_uplift_pp":policy_disagree,
        },
        "promotion_rule":"Require >=2.0pp strict-policy uplift and no material year collapse before considering a new L1 architecture.",
        "ability_uses_odds":False,
        "2026_locked":True,
        "promotion":bool(policy_uplift>=2.0),
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== POOLED =====")
    print((out/"pooled-strategies.csv").read_text())
    print("===== STRICT POLICY =====")
    print((out/"strict-policy.csv").read_text())
    print("L1_PAIRWISE_RESOLVER_V1_COMPLETE")

if __name__=="__main__":
    main()
