#!/usr/bin/env python3
import argparse,csv,gc,json,math
from collections import defaultdict,deque
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l1_objective_rebuild_v1 import (
    YEARS,TEST_YEARS,LOCKED_YEAR,THREADS,open_text,read_year,select_variant,model_common,
)
from run_l1_ceiling_audit_v1 import strip_forbidden
from run_l1_state_transition_v1 import build_state_features,attach_state
from run_l1_residual_regime_full_pairwise_v1 import (
    build_regime_features,fit_direct_rank,top1_metrics,
)
from run_l1_sameday_track_bias_v1 import build_same_day_bias,bias_cols,speed_cols
from run_l1_market_disagreement_audit_v1 import read_market_year

LAMBDAS=(0.25,0.50,0.75)
VARIANTS=("ABILITY","EVIDENCE_25","PENALTY_25","PENALTY_50","PENALTY_75","PENALTY_STRICT")
NEG_EVENT_THRESHOLD=0.20

AXES=(
    "all",
    "surface",
    "venue",
    "distance_band",
    "direction",
    "track_condition",
    "race_class",
    "weight_rule",
    "course_layout",
    "surface_distance",
    "venue_surface",
    "surface_going",
)

def finite(v):
    try:
        x=float(v)
    except (TypeError,ValueError):
        return np.nan
    return x if math.isfinite(x) else np.nan

def clean_token(v):
    if v is None:
        return None
    s=str(v).strip()
    return s if s and s.lower()!="nan" else None

def dist_band(v):
    x=finite(v)
    if not math.isfinite(x):
        return None
    return int(math.floor((x+100.0)/200.0)*200)

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

class NegStats:
    __slots__=("count","sum_sev","ewma","cat","strong_miss","recent")
    def __init__(self):
        self.count=0
        self.sum_sev=0.0
        self.ewma=np.nan
        self.cat=0
        self.strong_miss=0
        self.recent=deque(maxlen=5)
    def snapshot(self):
        if self.count<=0:
            return (0,np.nan,np.nan,np.nan,np.nan,np.nan)
        recent3=list(self.recent)[-3:]
        return (
            self.count,
            self.sum_sev/self.count,
            self.ewma,
            self.cat/self.count,
            self.strong_miss/self.count,
            float(np.mean(recent3)) if recent3 else np.nan,
        )
    def update(self,severity,strong_miss):
        if not math.isfinite(severity):
            return
        self.count+=1
        self.sum_sev+=severity
        self.ewma=severity if not math.isfinite(self.ewma) else (0.55*self.ewma+0.45*severity)
        self.cat+=int(severity>=0.35)
        self.strong_miss+=int(bool(strong_miss))
        self.recent.append(float(severity))

def axis_keys(row):
    surface=clean_token(row.get("surface"))
    venue=clean_token(row.get("venue_code"))
    db=dist_band(row.get("distance_m"))
    direction=clean_token(row.get("direction"))
    going=clean_token(row.get("track_condition"))
    race_class=clean_token(row.get("backfill_race_class_normalized"))
    weight_rule=clean_token(row.get("backfill_weight_rule"))
    layout=clean_token(row.get("backfill_course_layout"))
    return {
        "all":"ALL",
        "surface":surface,
        "venue":venue,
        "distance_band":db,
        "direction":direction,
        "track_condition":going,
        "race_class":race_class,
        "weight_rule":weight_rule,
        "course_layout":layout,
        "surface_distance":(surface,db) if surface is not None and db is not None else None,
        "venue_surface":(venue,surface) if venue is not None and surface is not None else None,
        "surface_going":(surface,going) if surface is not None and going is not None else None,
    }

def read_negative_source(paths):
    frames=[]
    keep=(
        "venue_code","surface","distance_m","direction","track_condition",
        "backfill_race_class_normalized","backfill_weight_rule","backfill_course_layout",
        "network_expected_pairwise_score",
    )
    for year in YEARS:
        rows=[]
        with open_text(paths[year]) as f:
            for line in f:
                if not line.strip():
                    continue
                r=json.loads(line)
                feat=r.get("features") or {}
                target=r.get("target") or {}
                date=str(feat.get("race_date") or r.get("race_date") or "")[:10]
                if not date.startswith(str(year)+"-"):
                    raise SystemExit(f"negative source year drift expected={year} date={date}")
                rid=str(r.get("race_id") or "")
                hid=str(r.get("horse_id") or "")
                finish=finite(target.get("finish_position"))
                if not rid or not hid or not math.isfinite(finish) or finish<1:
                    continue
                row={
                    "_race_id":rid,"_horse_id":hid,"_race_date":date,"_year":year,
                    "_finish":finish,
                }
                for c in keep:
                    row[c]=feat.get(c)
                rows.append(row)
        y=pd.DataFrame.from_records(rows)
        if y.empty:
            raise SystemExit(f"empty negative source year={year}")
        frames.append(y)
        print(f"NEG_SOURCE_READY year={year} rows={len(y)} races={y['_race_id'].nunique()}",flush=True)

    s=pd.concat(frames,ignore_index=True)
    race_n=s.groupby("_race_id")["_horse_id"].transform("size").astype(float)
    denom=(race_n-1.0).clip(lower=1.0)
    s["_finish_pct"]=(1.0-(pd.to_numeric(s["_finish"],errors="coerce")-1.0)/denom).clip(0.0,1.0)
    exp=pd.to_numeric(s["network_expected_pairwise_score"],errors="coerce")
    s["_expected"]=exp
    s["_underperf_target"]=(exp-s["_finish_pct"]).clip(lower=0.0,upper=1.0)
    s["_neg_event"]=np.where(
        s["_underperf_target"].notna(),
        (s["_underperf_target"]>=NEG_EVENT_THRESHOLD).astype(float),
        np.nan,
    )
    s["_strong_miss"]=np.where(
        exp.notna(),
        ((exp>=0.55)&(pd.to_numeric(s["_finish"],errors="coerce")>3)).astype(float),
        np.nan,
    )
    return s

def build_negative_evidence(paths):
    s=read_negative_source(paths)
    feature_rows=[]
    coverage=[]
    by_horse=s.groupby("_horse_id",sort=False)

    for hid,horse in by_horse:
        state={axis:defaultdict(NegStats) for axis in AXES}
        horse=horse.sort_values(["_race_date","_race_id"]).copy()
        for date,day in horse.groupby("_race_date",sort=True):
            pending=[]
            for idx,row in day.iterrows():
                keys=axis_keys(row)
                rec={"_race_id":row["_race_id"],"_horse_id":hid,"_year":int(row["_year"])}
                weighted=[]
                max_ewma=[]
                axes_hist=0
                axes_2plus=0
                for axis in AXES:
                    key=keys[axis]
                    stats=None if key is None else state[axis].get(key)
                    snap=(0,np.nan,np.nan,np.nan,np.nan,np.nan) if stats is None else stats.snapshot()
                    count,mean,ewma,cat_rate,strong_miss_rate,recent3=snap
                    prefix=f"neg_{axis}"
                    rec[f"{prefix}_count"]=float(count)
                    rec[f"{prefix}_mean"]=mean
                    rec[f"{prefix}_ewma"]=ewma
                    rec[f"{prefix}_cat_rate"]=cat_rate
                    rec[f"{prefix}_strong_miss_rate"]=strong_miss_rate
                    rec[f"{prefix}_recent3"]=recent3
                    if count>0 and math.isfinite(ewma):
                        axes_hist+=1
                        weighted.append((math.log1p(count),ewma))
                        max_ewma.append(ewma)
                    if count>=2:
                        axes_2plus+=1
                if weighted:
                    den=sum(w for w,_ in weighted)
                    rec["neg_combined_weighted_ewma"]=sum(w*v for w,v in weighted)/den
                    rec["neg_combined_max_ewma"]=max(max_ewma)
                else:
                    rec["neg_combined_weighted_ewma"]=np.nan
                    rec["neg_combined_max_ewma"]=np.nan
                rec["neg_match_axes_with_history"]=float(axes_hist)
                rec["neg_match_axes_2plus"]=float(axes_2plus)
                rec["_underperf_target"]=finite(row["_underperf_target"])
                rec["_neg_event"]=finite(row["_neg_event"])
                feature_rows.append(rec)
                pending.append((keys,finite(row["_underperf_target"]),finite(row["_strong_miss"])))

            # Same horse/date rows see exactly the same PRIOR state. Updates happen only after
            # every row on that horse/date has been snapshotted, so same-day ordering cannot leak.
            for keys,severity,strong_miss in pending:
                if not math.isfinite(severity):
                    continue
                for axis in AXES:
                    key=keys[axis]
                    if key is None:
                        continue
                    state[axis][key].update(severity,math.isfinite(strong_miss) and strong_miss>0.5)

    out=pd.DataFrame.from_records(feature_rows)
    if out.duplicated(["_race_id","_horse_id"]).any():
        raise SystemExit("duplicate negative evidence key")
    neg_cols=sorted(c for c in out.columns if c.startswith("neg_"))
    by_year={}
    for year in YEARS:
        y=out[out["_year"]==year][["_race_id","_horse_id","_underperf_target","_neg_event",*neg_cols]].copy()
        by_year[year]=y
        coverage.append({
            "year":year,
            "rows":len(y),
            "any_negative_history_pct":100*float((y["neg_match_axes_with_history"]>0).mean()),
            "two_plus_axis_history_pct":100*float((y["neg_match_axes_2plus"]>0).mean()),
            "all_axis_count1_pct":100*float((y["neg_all_count"]>0).mean()),
            "surface_history_pct":100*float((y["neg_surface_count"]>0).mean()),
            "venue_history_pct":100*float((y["neg_venue_count"]>0).mean()),
            "distance_history_pct":100*float((y["neg_distance_band_count"]>0).mean()),
        })
        print(
            f"NEG_EVIDENCE_READY year={year} rows={len(y)} features={len(neg_cols)} "
            f"any_history={coverage[-1]['any_negative_history_pct']:.2f}",
            flush=True,
        )
    del s,out
    gc.collect()
    return by_year,neg_cols,coverage

def penalty_params(seed):
    base=model_common(seed)
    base["n_estimators"]=260
    base["learning_rate"]=0.035
    base["num_leaves"]=25
    base["min_child_samples"]=120
    base["reg_lambda"]=8.0
    base["reg_alpha"]=1.0
    return base

def numeric_matrix(train,valid,cols):
    xtr=train.reindex(columns=cols).apply(pd.to_numeric,errors="coerce").astype("float32")
    xva=valid.reindex(columns=cols).apply(pd.to_numeric,errors="coerce").astype("float32")
    return xtr,xva

def fit_penalty_head(train,valid,neg_cols,seed):
    t=train[train["_underperf_target"].notna() & train["_neg_event"].notna()].copy()
    if len(t)<10000:
        raise SystemExit(f"too few penalty training rows {len(t)}")
    xtr,xva=numeric_matrix(t,valid,neg_cols)
    yevent=t["_neg_event"].astype(int).to_numpy()
    ysev=t["_underperf_target"].astype(float).to_numpy()

    clf=lgb.LGBMClassifier(objective="binary",**penalty_params(seed+1))
    clf.fit(xtr,yevent)
    p=np.asarray(clf.predict_proba(xva)[:,1],dtype=float)
    imp1=clf.booster_.feature_importance(importance_type="gain")
    del clf
    gc.collect()

    reg=lgb.LGBMRegressor(objective="huber",**penalty_params(seed+2))
    reg.fit(xtr,ysev)
    sev=np.clip(np.asarray(reg.predict(xva),dtype=float),0.0,1.0)
    imp2=reg.booster_.feature_importance(importance_type="gain")
    del reg,xtr,xva
    gc.collect()

    score=(0.65*p)+(0.35*sev)
    imp=sorted(
        [(c,float(a+b)) for c,a,b in zip(neg_cols,imp1,imp2)],
        key=lambda z:z[1],reverse=True,
    )
    return score,p,sev,imp

def rank01(df,col):
    r=df.groupby("_race_id")[col].rank(method="average",ascending=False)
    n=df.groupby("_race_id")[col].transform("size").astype(float)
    return 1.0-(r-1.0)/(n-1.0).clip(lower=1.0)

def market_diag(pred,market,score_col,variant):
    x=pred[["_race_id","_horse_id","_finish","_is_win","_is_top3",score_col]].copy()
    x=x.merge(market[["_race_id","_horse_id","final_popularity","final_win_odds"]],
              on=["_race_id","_horse_id"],how="left",validate="one_to_one")
    x=x.sort_values(["_race_id",score_col,"_horse_id"],ascending=[True,False,True])
    x["ai_rank"]=x.groupby("_race_id",sort=False).cumcount()+1
    pop=pd.to_numeric(x["final_popularity"],errors="coerce")
    odds=pd.to_numeric(x["final_win_odds"],errors="coerce")
    odds_rank=odds.groupby(x["_race_id"]).rank(method="min",ascending=True)
    x["market_rank"]=pop.where(pop>=1,odds_rank)
    x=x[x["market_rank"].notna()].copy()
    x["market_rank"]=x["market_rank"].astype(int)
    x["gap"]=x["market_rank"]-x["ai_rank"]
    top6=x[x["ai_rank"]<=6]
    up=top6[top6["gap"]>0]
    up3=top6[top6["gap"]>=3]
    top1=x[x["ai_rank"]==1]
    top1_up=top1[top1["gap"]>0]
    def rate(z,c):
        return 100*float(z[c].mean()) if len(z) else np.nan
    return {
        "variant":variant,
        "top6_horses":int(len(top6)),
        "top6_ai_up_horses":int(len(up)),
        "top6_ai_up_share_pct":100*len(up)/len(top6) if len(top6) else np.nan,
        "top6_ai_up_top3_pct":rate(up,"_is_top3"),
        "top6_ai_up_miss_pct":100-rate(up,"_is_top3") if len(up) else np.nan,
        "top6_ai_up3_horses":int(len(up3)),
        "top6_ai_up3_share_pct":100*len(up3)/len(top6) if len(top6) else np.nan,
        "top6_ai_up3_top3_pct":rate(up3,"_is_top3"),
        "top6_ai_up3_miss_pct":100-rate(up3,"_is_top3") if len(up3) else np.nan,
        "top1_market_disagree_horses":int(len(top1_up)),
        "top1_market_disagree_top3_pct":rate(top1_up,"_is_top3"),
        "top1_market_agree_top3_pct":rate(top1[top1["gap"]==0],"_is_top3"),
    }

def pooled(rows):
    out=[]
    for variant in VARIANTS:
        vals=[r for r in rows if r["variant"]==variant]
        if not vals:
            continue
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
    return out

def choose_strict_lambda(prior_rows):
    if not prior_rows:
        return 0.50
    candidates=[]
    for lam in LAMBDAS:
        name=f"PENALTY_{int(lam*100)}"
        vals=[r for r in prior_rows if r["variant"]==name]
        if not vals:
            continue
        score=float(np.mean([r["top1_top3_pct"] for r in vals]))
        candidates.append((score,-abs(lam-0.50),lam))
    return max(candidates)[2] if candidates else 0.50

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
        "contract":"L1_NEGATIVE_EVIDENCE_PENALTY_V1_RUNTIME",
        "ability_model":"SPEED_PLUS_DAY_BIAS",
        "negative_target":"max(0, prior pre-race network expected score - realized finish percentile)",
        "negative_event_threshold":NEG_EVENT_THRESHOLD,
        "penalty_model_inputs":"strictly prior horse-condition negative-evidence summaries only",
        "final_score":"within-race ability rank score MINUS lambda * learned penalty score",
        "lambdas":list(LAMBDAS),
        "strict_selection":"2021 lambda=0.50; later folds choose lambda using earlier OOS folds only",
        "market_is_evaluation_only":True,
        "ability_uses_odds":False,
        "2026_locked":True,
    },separators=(",",":")),flush=True)

    neg_by_year,neg_cols,neg_coverage=build_negative_evidence(paths)
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
    strict_choices=[]

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

        seed=5100000+test*100
        ability_pred,_=fit_direct_rank(ability_train,ability_valid,seed)
        expected_top1_top3={
            2021:59.34606481481482,
            2022:60.90856481481482,
            2023:59.432870370370374,
            2024:62.673611111111114,
            2025:61.660879629629626,
        }[test]
        reproduced=top1_metrics(ability_pred,"_score")["top1_top3_pct"]
        if abs(reproduced-expected_top1_top3)>1e-9:
            raise SystemExit(
                f"ABILITY reproduction drift year={test} expected={expected_top1_top3} got={reproduced}"
            )
        ability_pred["ability_rank01"]=rank01(ability_pred,"_score")

        neg_train=pd.concat([neg_by_year[y] for y in train_years],ignore_index=True,copy=False)
        neg_valid=neg_by_year[test]
        penalty,prob,sev,imp=fit_penalty_head(neg_train,neg_valid,neg_cols,seed+20)
        pframe=neg_valid[["_race_id","_horse_id"]].copy()
        pframe["penalty_score"]=penalty
        pframe["penalty_prob"]=prob
        pframe["penalty_severity"]=sev
        pframe["evidence_score"]=pd.to_numeric(
            neg_valid["neg_combined_weighted_ewma"],errors="coerce"
        ).fillna(0.0).clip(0.0,1.0)

        pred=ability_pred.merge(pframe,on=["_race_id","_horse_id"],how="left",validate="one_to_one")
        pred["penalty_score"]=pd.to_numeric(pred["penalty_score"],errors="coerce").fillna(0.0)
        pred["evidence_score"]=pd.to_numeric(pred["evidence_score"],errors="coerce").fillna(0.0)

        score_cols={"ABILITY":"ability_rank01"}
        pred["score_evidence25"]=pred["ability_rank01"]-(0.25*pred["evidence_score"])
        score_cols["EVIDENCE_25"]="score_evidence25"
        for lam in LAMBDAS:
            c=f"score_penalty{int(lam*100)}"
            pred[c]=pred["ability_rank01"]-(lam*pred["penalty_score"])
            score_cols[f"PENALTY_{int(lam*100)}"]=c

        strict_lambda=choose_strict_lambda([r for r in rows if r["test_year"]<test])
        strict_choices.append({"test_year":test,"lambda":strict_lambda})
        strict_col=f"score_penalty{int(strict_lambda*100)}"
        pred["score_strict"]=pred[strict_col]
        score_cols["PENALTY_STRICT"]="score_strict"

        for variant in VARIANTS:
            col=score_cols[variant]
            rows.append({
                "test_year":test,
                "train_years":"|".join(map(str,train_years)),
                "variant":variant,
                "strict_lambda":strict_lambda if variant=="PENALTY_STRICT" else "",
                **top1_metrics(pred,col),
            })
            market_rows.append({
                "test_year":test,
                **market_diag(pred,market_by_year[test],col,variant),
            })

        for rank,(feature,gain) in enumerate(imp[:100],start=1):
            importance.append({
                "test_year":test,"importance_rank":rank,
                "feature":feature,"gain":gain,
            })

        print(
            f"NEG_PENALTY_FOLD_READY year={test} strict_lambda={strict_lambda:.2f} "
            f"ability={rows[-len(VARIANTS)]['top1_top3_pct']:.4f} "
            f"strict={rows[-1]['top1_top3_pct']:.4f}",
            flush=True,
        )

        del train_base,valid_base,btrain,bvalid,rtrain,rvalid,merged_train,merged_valid
        del ability_train,ability_valid,ability_pred,neg_train,neg_valid,pframe,pred
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
    write_csv(out/"penalty-feature-importance.csv",importance)
    write_csv(out/"negative-evidence-coverage.csv",neg_coverage)
    write_csv(out/"state-coverage.csv",state_coverage)
    write_csv(out/"strict-lambda-choices.csv",strict_choices)

    strict=next(r for r in pool if r["variant"]=="PENALTY_STRICT")
    best=max([r for r in pool if r["variant"]!="ABILITY"],key=lambda z:z["top1_top3_pct"])
    md={v["variant"]:v for v in market_rows if v["test_year"]==2025}

    (out/"summary.json").write_text(json.dumps({
        "contract":"L1_NEGATIVE_EVIDENCE_PENALTY_V1",
        "ability_model":"SPEED_PLUS_DAY_BIAS exact walk-forward reproduction",
        "negative_evidence":{
            "definition":"Past underperformance severity = max(0, pre-race network expected pairwise score - realized finish percentile).",
            "condition_axes":list(AXES),
            "same_day_guard":"all rows for one horse/date are snapshotted before any same-date outcomes update state",
            "target_is_not_plain_miss":True,
        },
        "penalty_head":{
            "inputs":"negative-evidence summaries only",
            "event_target":f"underperformance >= {NEG_EVENT_THRESHOLD}",
            "severity_target":"continuous underperformance severity",
            "blend":"0.65 event probability + 0.35 predicted severity",
        },
        "final_score":"within-race ability rank score - lambda * penalty score",
        "strict_policy":{
            "selection":"2021 lambda=0.50, later years choose among 0.25/0.50/0.75 using only earlier OOS fold Top1->Top3",
            "result":strict,
            "choices":strict_choices,
        },
        "best_exploratory":best,
        "market_diagnostic_note":"Market is used only after prediction to audit whether penalty reduces unsupported AI promotion.",
        "2025_market_diagnostics":md,
        "promotion_rule":"Exploratory promotion requires >=1.0pp pooled Top1->Top3 uplift vs ABILITY and worst fold >=-0.5pp. Strict policy is reported separately.",
        "promotion":bool(
            best["top1_top3_uplift_vs_ability_pp"]>=1.0
            and best["top1_top3_uplift_worst_fold_pp"]>=-0.5
        ),
        "ability_uses_odds":False,
        "market_is_evaluation_only":True,
        "2026_locked":True,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== POOLED =====")
    print((out/"pooled-metrics.csv").read_text())
    print("===== MARKET DIAGNOSTICS =====")
    print((out/"market-diagnostics.csv").read_text())
    print("L1_NEGATIVE_EVIDENCE_PENALTY_V1_COMPLETE")

if __name__=="__main__":
    main()
