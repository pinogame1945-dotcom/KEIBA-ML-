#!/usr/bin/env python3
import argparse,csv,gc,json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l1_objective_rebuild_v1 import (
    YEARS,TEST_YEARS,LOCKED_YEAR,prepare,model_common,read_year,select_variant,
)
from run_l1_ceiling_audit_v1 import strip_forbidden
from run_l1_state_transition_v1 import build_state_features,attach_state
from run_l1_residual_regime_full_pairwise_v1 import (
    build_regime_features,fit_direct_rank,top1_metrics,
)
from run_l1_sameday_track_bias_v1 import build_same_day_bias,bias_cols,speed_cols
from run_l1_negative_evidence_penalty_v1 import (
    build_negative_evidence,read_market_year,market_diag,
)

VARIANTS=("ABILITY","NEG_DIRECT_FREE","NEG_DIRECT_MONO","NEG_CORE_MONO")

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

def badness_cols(neg_cols):
    out=[]
    for c in neg_cols:
        if c in ("neg_combined_weighted_ewma","neg_combined_max_ewma"):
            out.append(c); continue
        if c.endswith(("_mean","_ewma","_cat_rate","_strong_miss_rate","_recent3")):
            out.append(c)
    return sorted(set(out))

def fit_rank(train,valid,seed,monotone_bad=None):
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

    params=model_common(seed)
    constrained=set(monotone_bad or [])
    constraints=[-1 if c in constrained else 0 for c in xtr.columns]
    if constrained:
        params["monotone_constraints"]=constraints
        params["monotone_constraints_method"]="advanced"

    model=lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0,1,3,7,15,31,63],
        **params,
    )
    model.fit(xtr,rel,group=group,categorical_feature=cats)
    pred=np.asarray(model.predict(xva),dtype=float)
    out=valid[["_race_id","_horse_id","_finish","_is_win","_is_top3"]].copy()
    out["_score"]=pred
    gain=model.booster_.feature_importance(importance_type="gain")
    imp=sorted(zip(xtr.columns,gain),key=lambda z:z[1],reverse=True)
    used_constraints={c:-1 for c in xtr.columns if c in constrained}

    del xtr,xva,model
    gc.collect()
    return out,imp,used_constraints

def pooled(rows):
    out=[]
    for variant in VARIANTS:
        vals=[r for r in rows if r["variant"]==variant]
        w=np.array([r["races"] for r in vals],dtype=float)
        rec={"variant":variant,"folds":len(vals),"races":int(w.sum())}
        for key in ("top1_top3_pct","top1_win_pct","winner_top3_capture_pct","winner_top6_capture_pct"):
            arr=np.array([r[key] for r in vals],dtype=float)
            rec[key]=float(np.average(arr,weights=w))
            rec[key+"_worst"]=float(arr.min())
            rec[key+"_std"]=float(arr.std(ddof=0))
        out.append(rec)

    base=next(r for r in out if r["variant"]=="ABILITY")
    fold_base={r["test_year"]:r for r in rows if r["variant"]=="ABILITY"}
    for r in out:
        r["top1_top3_uplift_vs_ability_pp"]=r["top1_top3_pct"]-base["top1_top3_pct"]
        diffs=[
            x["top1_top3_pct"]-fold_base[x["test_year"]]["top1_top3_pct"]
            for x in rows if x["variant"]==r["variant"]
        ]
        r["top1_top3_uplift_worst_fold_pp"]=float(min(diffs))
        r["top1_top3_uplift_best_fold_pp"]=float(max(diffs))
        r["positive_folds"]=int(sum(d>0 for d in diffs))
    return out

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
        "contract":"L1_NEGATIVE_EVIDENCE_DIRECT_V1_RUNTIME",
        "ability_model":"SPEED_PLUS_DAY_BIAS exact walk-forward reproduction",
        "architecture":{
            "NEG_DIRECT_FREE":"ability features + strictly-prior negative-evidence features, unconstrained",
            "NEG_DIRECT_MONO":"same direct ranker, negative badness columns constrained monotone decreasing",
            "NEG_CORE_MONO":"only badness/severity negative features, constrained monotone decreasing",
        },
        "negative_target_source":"past underperformance relative to prior pre-race expectation",
        "posthoc_penalty":False,
        "market_is_evaluation_only":True,
        "ability_uses_odds":False,
        "2026_locked":True,
    },separators=(",",":")),flush=True)

    neg_by_year,neg_cols,neg_coverage=build_negative_evidence(paths)
    bad_cols=badness_cols(neg_cols)
    if not bad_cols:
        raise SystemExit("no negative badness columns")

    bias_by_year,_=build_same_day_bias(paths)
    state_by_year,_,_,_,state_coverage=build_state_features(paths)
    regime_by_year,regime_cols=build_regime_features(state_by_year)
    spd_cols=speed_cols(regime_cols)
    fit_cols=bias_cols(bias_by_year[2021],with_fit=True)
    market_by_year={y:read_market_year(paths[y],y) for y in TEST_YEARS}

    base_cache={}
    def get_base(year):
        if year not in base_cache:
            raw=strip_forbidden(read_year(paths[year],year))
            base_cache[year]=select_variant(raw,"core4")
            del raw
            gc.collect()
        return base_cache[year]

    rows=[]
    market_rows=[]
    importance=[]
    constraints_rows=[]

    expected_baseline={
        2021:59.34606481481482,
        2022:60.90856481481482,
        2023:59.432870370370374,
        2024:62.673611111111114,
        2025:61.660879629629626,
    }

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
        ability_train=attach_state(train_base,merged_train,fit_cols+spd_cols)
        ability_valid=attach_state(valid_base,merged_valid,fit_cols+spd_cols)

        neg_train=pd.concat([neg_by_year[y] for y in train_years],ignore_index=True,copy=False)
        neg_valid=neg_by_year[test]
        neg_train_x=neg_train[["_race_id","_horse_id",*neg_cols]].copy()
        neg_valid_x=neg_valid[["_race_id","_horse_id",*neg_cols]].copy()

        direct_train=attach_state(ability_train,neg_train_x,neg_cols)
        direct_valid=attach_state(ability_valid,neg_valid_x,neg_cols)
        core_train=attach_state(ability_train,neg_train_x,bad_cols)
        core_valid=attach_state(ability_valid,neg_valid_x,bad_cols)

        seed=5100000+test*100

        pred_base,imp_base=fit_direct_rank(ability_train,ability_valid,seed)
        got=top1_metrics(pred_base,"_score")["top1_top3_pct"]
        if abs(got-expected_baseline[test])>1e-9:
            raise SystemExit(
                f"ABILITY reproduction drift year={test} expected={expected_baseline[test]} got={got}"
            )

        pred_free,imp_free,_=fit_rank(direct_train,direct_valid,seed,monotone_bad=None)
        pred_mono,imp_mono,used_mono=fit_rank(direct_train,direct_valid,seed,monotone_bad=bad_cols)
        pred_core,imp_core,used_core=fit_rank(core_train,core_valid,seed,monotone_bad=bad_cols)

        variants={
            "ABILITY":(pred_base,imp_base,{}),
            "NEG_DIRECT_FREE":(pred_free,imp_free,{}),
            "NEG_DIRECT_MONO":(pred_mono,imp_mono,used_mono),
            "NEG_CORE_MONO":(pred_core,imp_core,used_core),
        }

        for variant,(pred,imp,used) in variants.items():
            met=top1_metrics(pred,"_score")
            rows.append({
                "test_year":test,
                "train_years":"|".join(map(str,train_years)),
                "variant":variant,
                **met,
            })
            market_rows.append({
                "test_year":test,
                **market_diag(pred,market_by_year[test],"_score",variant),
            })
            for rank,(feature,gain) in enumerate(imp[:80],start=1):
                importance.append({
                    "test_year":test,"variant":variant,
                    "importance_rank":rank,"feature":feature,"gain":float(gain),
                    "is_negative_feature":bool(feature.startswith("neg_")),
                    "monotone_constraint":int(used.get(feature,0)),
                })
            if used:
                constraints_rows.append({
                    "test_year":test,"variant":variant,
                    "constrained_features":len(used),
                    "negative_feature_count":len(bad_cols),
                })

        print(
            f"NEG_DIRECT_FOLD_READY year={test} "
            f"ability={rows[-4]['top1_top3_pct']:.4f} "
            f"free={rows[-3]['top1_top3_pct']:.4f} "
            f"mono={rows[-2]['top1_top3_pct']:.4f} "
            f"core_mono={rows[-1]['top1_top3_pct']:.4f}",
            flush=True,
        )

        del train_base,valid_base,btrain,bvalid,rtrain,rvalid,merged_train,merged_valid
        del ability_train,ability_valid,neg_train,neg_valid,neg_train_x,neg_valid_x
        del direct_train,direct_valid,core_train,core_valid
        del pred_base,pred_free,pred_mono,pred_core
        gc.collect()
        for old in list(base_cache):
            if old<test-1:
                del base_cache[old]

    pool=pooled(rows)
    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"fold-metrics.csv",rows)
    write_csv(out/"pooled-metrics.csv",pool)
    write_csv(out/"market-diagnostics.csv",market_rows)
    write_csv(out/"feature-importance.csv",importance)
    write_csv(out/"negative-evidence-coverage.csv",neg_coverage)
    write_csv(out/"state-coverage.csv",state_coverage)
    write_csv(out/"monotone-constraints.csv",constraints_rows)

    best=max([r for r in pool if r["variant"]!="ABILITY"],key=lambda z:z["top1_top3_pct"])
    market_pooled=[]
    for variant in VARIANTS:
        vals=[r for r in market_rows if r["variant"]==variant]
        market_pooled.append({
            "variant":variant,
            "top6_ai_up_share_pct":float(np.average([r["top6_ai_up_share_pct"] for r in vals])),
            "top6_ai_up3_share_pct":float(np.average([r["top6_ai_up3_share_pct"] for r in vals])),
            "top6_ai_up3_miss_pct":float(np.average([r["top6_ai_up3_miss_pct"] for r in vals])),
            "top1_market_disagree_top3_pct":float(np.average([r["top1_market_disagree_top3_pct"] for r in vals])),
            "top1_market_agree_top3_pct":float(np.average([r["top1_market_agree_top3_pct"] for r in vals])),
        })
    write_csv(out/"market-pooled.csv",market_pooled)

    (out/"summary.json").write_text(json.dumps({
        "contract":"L1_NEGATIVE_EVIDENCE_DIRECT_V1",
        "ability_model":"SPEED_PLUS_DAY_BIAS exact walk-forward reproduction",
        "architecture":"Negative evidence is part of the ranking model itself; no post-hoc score subtraction.",
        "negative_features":{
            "all_count":len(neg_cols),
            "badness_count":len(bad_cols),
            "badness_columns":bad_cols,
            "meaning":"higher badness = stronger strictly-prior evidence that this horse underperforms under matching conditions",
        },
        "variants":{
            "NEG_DIRECT_FREE":"all negative evidence direct into ranker, no sign constraint",
            "NEG_DIRECT_MONO":"all negative evidence direct into ranker; badness features monotone decreasing",
            "NEG_CORE_MONO":"badness-only negative evidence direct into ranker; all monotone decreasing",
        },
        "pooled":pool,
        "best":best,
        "market_pooled":market_pooled,
        "promotion_rule":"promote only if >=1.0pp pooled Top1->Top3 uplift, >=4/5 positive folds, and worst fold >=-0.5pp",
        "promotion":bool(
            best["top1_top3_uplift_vs_ability_pp"]>=1.0
            and best["positive_folds"]>=4
            and best["top1_top3_uplift_worst_fold_pp"]>=-0.5
        ),
        "posthoc_penalty":False,
        "market_is_evaluation_only":True,
        "ability_uses_odds":False,
        "2026_locked":True,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== POOLED =====")
    print((out/"pooled-metrics.csv").read_text())
    print("===== MARKET POOLED =====")
    print((out/"market-pooled.csv").read_text())
    print("L1_NEGATIVE_EVIDENCE_DIRECT_V1_COMPLETE")

if __name__=="__main__":
    main()
