#!/usr/bin/env python3
import argparse,csv,gc,json,math
from pathlib import Path

import numpy as np
import pandas as pd

from run_l1_objective_rebuild_v1 import YEARS,TEST_YEARS,LOCKED_YEAR,open_text,read_year,select_variant
from run_l1_ceiling_audit_v1 import strip_forbidden
from run_l1_state_transition_v1 import build_state_features,attach_state
from run_l1_residual_regime_full_pairwise_v1 import build_regime_features,fit_direct_rank
from run_l1_sameday_track_bias_v1 import build_same_day_bias,bias_cols,speed_cols

SCOPES=("ALL","AI_TOP1","AI_TOP3","AI_TOP6")

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
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def read_market_year(path,year):
    rows=[]
    with open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            feat=r.get("features") or {}
            target=r.get("target") or {}
            market=r.get("market_outcome") or {}
            date=str(feat.get("race_date") or r.get("race_date") or "")[:10]
            if not date.startswith(str(year)+"-"):
                raise SystemExit(f"market year drift expected={year} date={date}")
            rid=str(r.get("race_id") or "")
            hid=str(r.get("horse_id") or "")
            finish=finite(target.get("finish_position"))
            if not rid or not hid or not math.isfinite(finish) or finish<1:
                continue
            rows.append({
                "_race_id":rid,
                "_horse_id":hid,
                "_year":year,
                "_finish_market":finish,
                "final_win_odds":finite(market.get("final_win_odds")),
                "final_popularity":finite(market.get("final_popularity")),
            })
    out=pd.DataFrame.from_records(rows)
    if out.empty:
        raise SystemExit(f"empty market rows year={year}")
    if out.duplicated(["_race_id","_horse_id"]).any():
        raise SystemExit(f"duplicate market keys year={year}")
    return out

def label_agreement(ai_rank,market_rank):
    d=market_rank-ai_rank
    return np.where(d>0,"AI_UP",np.where(d<0,"AI_DOWN","SAME"))

def summarize(frame,scope,label):
    x=frame
    if scope=="AI_TOP1":
        x=x[x["ai_rank"]==1]
    elif scope=="AI_TOP3":
        x=x[x["ai_rank"]<=3]
    elif scope=="AI_TOP6":
        x=x[x["ai_rank"]<=6]
    x=x[x["agreement"]==label]
    if x.empty:
        return {
            "scope":scope,"agreement":label,"horses":0,"races":0,
            "top3_rate_pct":np.nan,"miss_rate_pct":np.nan,"win_rate_pct":np.nan,
            "avg_ai_rank":np.nan,"avg_market_rank":np.nan,"avg_final_odds":np.nan,
        }
    return {
        "scope":scope,
        "agreement":label,
        "horses":int(len(x)),
        "races":int(x["_race_id"].nunique()),
        "top3_rate_pct":100*float(x["_is_top3"].mean()),
        "miss_rate_pct":100*float((1-x["_is_top3"]).mean()),
        "win_rate_pct":100*float(x["_is_win"].mean()),
        "avg_ai_rank":float(x["ai_rank"].mean()),
        "avg_market_rank":float(x["market_rank"].mean()),
        "avg_final_odds":float(pd.to_numeric(x["final_win_odds"],errors="coerce").mean()),
    }

def gap_bucket(v):
    if v>=3: return "AI_UP_3PLUS"
    if v==2: return "AI_UP_2"
    if v==1: return "AI_UP_1"
    if v==0: return "SAME"
    if v==-1: return "AI_DOWN_1"
    if v==-2: return "AI_DOWN_2"
    return "AI_DOWN_3PLUS"

def summarize_group(x,keys):
    rows=[]
    for key,grp in x.groupby(keys,dropna=False,sort=True):
        if not isinstance(key,tuple):
            key=(key,)
        rec={k:v for k,v in zip(keys,key)}
        rec.update({
            "horses":int(len(grp)),
            "races":int(grp["_race_id"].nunique()),
            "top3_rate_pct":100*float(grp["_is_top3"].mean()),
            "miss_rate_pct":100*float((1-grp["_is_top3"]).mean()),
            "win_rate_pct":100*float(grp["_is_win"].mean()),
            "avg_final_odds":float(pd.to_numeric(grp["final_win_odds"],errors="coerce").mean()),
            "avg_market_rank":float(grp["market_rank"].mean()),
        })
        rows.append(rec)
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
        "contract":"L1_MARKET_DISAGREEMENT_AUDIT_V1_RUNTIME",
        "model":"SPEED_PLUS_DAY_BIAS from L1_SAMEDAY_TRACK_BIAS_V1",
        "agreement_definition":{
            "AI_UP":"ai_rank < final popularity rank",
            "SAME":"ai_rank == final popularity rank",
            "AI_DOWN":"ai_rank > final popularity rank",
        },
        "market_is_evaluation_only":True,
        "model_uses_market":False,
        "2026_locked":True,
    },separators=(",",":")),flush=True)

    bias_by_year,_=build_same_day_bias(paths)
    state_by_year,_,_,_,_=build_state_features(paths)
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

    all_preds=[]
    fold_checks=[]

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
        tr=attach_state(train_base,merged_train,fit_cols+spd_cols)
        va=attach_state(valid_base,merged_valid,fit_cols+spd_cols)

        seed=5100000+test*100
        pred,_=fit_direct_rank(tr,va,seed)
        pred=pred.sort_values(
            ["_race_id","_score","_horse_id"],
            ascending=[True,False,True],
        ).reset_index(drop=True)
        pred["ai_rank"]=pred.groupby("_race_id",sort=False).cumcount()+1
        pred["_year"]=test
        pred=pred.merge(
            market_by_year[test],
            on=["_race_id","_horse_id","_year"],
            how="left",
            validate="one_to_one",
        )

        # final_popularity is the primary market rank. Use odds-derived ordinal rank only
        # as a diagnostic, never as model input.
        odds=pd.to_numeric(pred["final_win_odds"],errors="coerce")
        pred["odds_rank_derived"]=odds.groupby(pred["_race_id"]).rank(method="min",ascending=True)
        pop=pd.to_numeric(pred["final_popularity"],errors="coerce")
        pred["market_rank"]=pop.where(pop>=1,pred["odds_rank_derived"])
        valid_market=pred["market_rank"].notna()
        pred=pred[valid_market].copy()
        pred["market_rank"]=pred["market_rank"].astype(int)
        pred["rank_gap_market_minus_ai"]=pred["market_rank"]-pred["ai_rank"]
        pred["agreement"]=label_agreement(pred["ai_rank"],pred["market_rank"])
        pred["gap_bucket"]=pred["rank_gap_market_minus_ai"].map(gap_bucket)

        pop_compare=pred[
            pred["final_popularity"].notna() & pred["odds_rank_derived"].notna()
        ]
        exact=float(
            (pop_compare["final_popularity"].astype(float)==pop_compare["odds_rank_derived"].astype(float)).mean()
        ) if len(pop_compare) else np.nan
        top1=pred[pred["ai_rank"]==1]
        fold_checks.append({
            "test_year":test,
            "rows":int(len(pred)),
            "races":int(pred["_race_id"].nunique()),
            "market_coverage_pct":100*float(valid_market.mean()),
            "popularity_vs_odds_rank_exact_pct":100*exact if math.isfinite(exact) else np.nan,
            "model_top1_top3_pct":100*float(top1["_is_top3"].mean()),
            "model_top1_win_pct":100*float(top1["_is_win"].mean()),
        })
        all_preds.append(pred[[
            "_race_id","_horse_id","_year","_finish","_is_win","_is_top3",
            "ai_rank","market_rank","rank_gap_market_minus_ai","agreement","gap_bucket",
            "final_win_odds","final_popularity",
        ]].copy())

        print(
            f"MARKET_AUDIT_FOLD_READY year={test} rows={len(pred)} "
            f"races={pred['_race_id'].nunique()} top1_top3={fold_checks[-1]['model_top1_top3_pct']:.4f}",
            flush=True,
        )

        del train_base,valid_base,btrain,bvalid,rtrain,rvalid,merged_train,merged_valid,tr,va,pred
        gc.collect()
        for old in list(base_cache):
            if old<test-1:
                del base_cache[old]

    x=pd.concat(all_preds,ignore_index=True,copy=False)
    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)

    agreement=[]
    for scope in SCOPES:
        for label in ("AI_UP","SAME","AI_DOWN"):
            agreement.append(summarize(x,scope,label))

    # Add each group's share of all misses inside the same scope.
    for scope in SCOPES:
        scoped=x if scope=="ALL" else x[
            x["ai_rank"]<=({"AI_TOP1":1,"AI_TOP3":3,"AI_TOP6":6}[scope])
        ]
        total_misses=int((1-scoped["_is_top3"]).sum())
        for rec in agreement:
            if rec["scope"]!=scope or not rec["horses"]:
                continue
            grp=scoped[scoped["agreement"]==rec["agreement"]]
            misses=int((1-grp["_is_top3"]).sum())
            rec["misses"]=misses
            rec["share_of_scope_misses_pct"]=100*misses/total_misses if total_misses else np.nan

    exact_rank=summarize_group(x[x["ai_rank"]<=10],["ai_rank","agreement"])
    year_agreement=summarize_group(x[x["ai_rank"]<=6],["_year","agreement"])
    gap_rows=summarize_group(x[x["ai_rank"]<=6],["gap_bucket"])
    top1_market=summarize_group(x[x["ai_rank"]==1],["market_rank"])

    write_csv(out/"agreement-summary.csv",agreement)
    write_csv(out/"ai-rank-by-agreement.csv",exact_rank)
    write_csv(out/"year-top6-agreement.csv",year_agreement)
    write_csv(out/"top6-gap-buckets.csv",gap_rows)
    write_csv(out/"top1-by-market-rank.csv",top1_market)
    write_csv(out/"fold-checks.csv",fold_checks)

    top6=[r for r in agreement if r["scope"]=="AI_TOP6"]
    up=next(r for r in top6 if r["agreement"]=="AI_UP")
    down=next(r for r in top6 if r["agreement"]=="AI_DOWN")
    top1=[r for r in agreement if r["scope"]=="AI_TOP1"]
    top1_up=next(r for r in top1 if r["agreement"]=="AI_UP")
    top1_same=next(r for r in top1 if r["agreement"]=="SAME")

    (out/"summary.json").write_text(json.dumps({
        "contract":"L1_MARKET_DISAGREEMENT_AUDIT_V1",
        "model":"SPEED_PLUS_DAY_BIAS exact walk-forward reproduction",
        "years":list(TEST_YEARS),
        "races":int(x["_race_id"].nunique()),
        "horses_with_market":int(len(x)),
        "definition":{
            "AI_UP":"AI rank is better (smaller) than final popularity rank",
            "SAME":"same rank",
            "AI_DOWN":"AI rank is worse (larger) than final popularity rank",
            "miss":"actual finish outside top3",
        },
        "primary_top6":{
            "ai_up":up,
            "ai_down":down,
            "miss_rate_difference_up_minus_down_pp":float(up["miss_rate_pct"]-down["miss_rate_pct"]),
        },
        "top1_note":{
            "ai_up":top1_up,
            "same_market_rank1":top1_same,
            "why_no_ai_down":"An AI-ranked #1 horse cannot have AI rank worse than market rank #1; for AI top1 only AI_UP or SAME are possible.",
        },
        "interpretation_guard":"Raw ALL-horse miss rates mix very different AI ranks. Use AI-rank-by-agreement.csv for like-for-like rank comparisons.",
        "market_is_evaluation_only":True,
        "model_uses_market":False,
        "2026_locked":True,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== AGREEMENT SUMMARY =====")
    print((out/"agreement-summary.csv").read_text())
    print("L1_MARKET_DISAGREEMENT_AUDIT_V1_COMPLETE")

if __name__=="__main__":
    main()
