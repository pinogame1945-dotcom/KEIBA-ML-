#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import build_year_frame
from run_l2_ticket_market_gap_v2 import add_market_features, market_feature_columns
from run_l2_market_gap_law_arena_v1 import candidate_group_tables, chosen_from_table, metrics
from run_l2_quinella_human_machine_walkforward_v1 import (
    YEARS, TEST_YEARS, PRED_YEARS, EXPECTED_2025_SEQUENCE,
    predict_human_year, select_human_rules, machine_signal,
)

FOCUS_CANDIDATES=(
    "GAP10P__odds_band=40-60__field_size_band=13-14",
    "GAP0_2__field_size_band=13-14__l17_rank_band=11-20",
    "GAP7_9__field_size_band=15-16__model_rank_band=6-8",
)
PRIMARY_CANDIDATE=FOCUS_CANDIDATES[0]

NUMERIC_FEATURES=(
    "odds","market_q_norm","market_rank","l17_rank_score","model_rank",
    "rank_upgrade","l17_vs_market_upgrade","machine_ratio_to_market",
    "pair_rank_sum","pair_rank_gap","pair_worse_rank",
    "pair_prob_product","pair_prob_gap","pair_top3_support_sum",
    "pair_top6_support_sum","pair_rank_std_max","pair_probability_std_max",
    "distance_m","race_month","field_size",
)
CATEGORICAL_FEATURES=(
    "venue_code","discipline","surface","direction","weather",
    "track_condition","course_layout","race_class_normalized","grade",
    "sex_condition","weight_rule","distance_band","month_band",
)

def parse_args():
    p=argparse.ArgumentParser()
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def add_diag_bands(df):
    z=df.copy()
    d=pd.to_numeric(z["distance_m"],errors="coerce")
    z["distance_band"]=pd.cut(
        d,
        bins=[-np.inf,1399,1599,1799,1999,2199,2499,np.inf],
        labels=["<=1399","1400-1599","1600-1799","1800-1999","2000-2199","2200-2499","2500+"],
    ).astype(str)
    m=pd.to_numeric(z["race_month"],errors="coerce")
    z["month_band"]=m.fillna(-1).astype(int).astype(str)
    return z

def build_selected_rows(frames,cols):
    human_predictions={}
    for y in PRED_YEARS:
        pred,_=predict_human_year(y,frames,cols)
        human_predictions[y]=pred

    rows=[]
    selection_audit=[]
    for target in TEST_YEARS:
        dev_years=[y for y in PRED_YEARS if y<target]
        selected,_=select_human_rules(human_predictions,dev_years)
        ids=[x[0] for x in selected]
        if target==2025 and ids!=EXPECTED_2025_SEQUENCE:
            raise SystemExit(f"2025 zero-base reproduction drift got={ids}")
        machine,_=machine_signal(target,frames,cols)
        mcols=["year","race_id","pair_numbers","p_model","machine_ratio_to_market","machine_agree"]
        pred=human_predictions[target].merge(
            machine[mcols],
            on=["year","race_id","pair_numbers"],
            how="left",
            validate="one_to_one",
        )
        pred=add_diag_bands(pred)
        tables=candidate_group_tables(pred)
        for step,(cid,gap,conds,selrow) in enumerate(selected,1):
            if cid not in FOCUS_CANDIDATES:
                continue
            chosen=chosen_from_table(tables,gap,conds).copy()
            chosen["candidate_id"]=cid
            chosen["test_year"]=target
            chosen["selection_step"]=step
            rows.append(chosen)
            selection_audit.append({
                "candidate_id":cid,
                "test_year":target,
                "dev_years":"|".join(map(str,dev_years)),
                "step":step,
                "tickets":len(chosen),
                "machine_agree_tickets":int(chosen["machine_agree"].sum()),
            })
    if not rows:
        raise SystemExit("no focus candidate rows")
    return pd.concat(rows,ignore_index=True),pd.DataFrame(selection_audit)

def metric_row(g,cid,year,relation):
    m=metrics(g,f"{cid}_{year}_{relation}")
    m.update({
        "candidate_id":cid,
        "test_year":year,
        "relation":relation,
        "executed_races":int(g["race_id"].nunique()) if len(g) else 0,
    })
    return m

def bootstrap_roi_baseline(returns,target_n,observed_roi,seed):
    vals=np.asarray(returns,dtype=float)
    if len(vals)==0 or target_n<=0:
        return {}
    rng=np.random.default_rng(seed)
    sims=[]
    remaining=20000
    while remaining>0:
        b=min(1000,remaining)
        idx=rng.integers(0,len(vals),size=(b,target_n))
        ret=vals[idx].sum(axis=1)
        sims.append(100.0*ret/(100.0*target_n))
        remaining-=b
    x=np.concatenate(sims)
    return {
        "bootstrap_iterations":int(len(x)),
        "bootstrap_roi_p10":float(np.quantile(x,0.10)),
        "bootstrap_roi_median":float(np.quantile(x,0.50)),
        "bootstrap_roi_p90":float(np.quantile(x,0.90)),
        "bootstrap_prob_roi_le_observed_pct":100.0*float(np.mean(x<=observed_roi+1e-12)),
        "bootstrap_prob_roi_zero_pct":100.0*float(np.mean(x==0.0)),
    }

def variance_checks(selected):
    rows=[]
    for cid in FOCUS_CANDIDATES:
        g=selected[(selected["candidate_id"]==cid)&(selected["machine_agree"])].copy()
        years=sorted(g["test_year"].unique().tolist())
        if len(years)<2:
            continue
        target=years[-1]
        base=g[g["test_year"]<target].copy()
        test=g[g["test_year"]==target].copy()
        if base.empty or test.empty:
            continue
        p=float(base["hit"].mean())
        n=len(test)
        observed_hits=int(test["hit"].sum())
        observed_roi=100.0*float(test["return_yen_per100"].sum())/(100.0*n)
        row={
            "candidate_id":cid,
            "baseline_years":"|".join(map(str,sorted(base["test_year"].unique()))),
            "target_year":int(target),
            "baseline_tickets":len(base),
            "baseline_hits":int(base["hit"].sum()),
            "baseline_hit_rate_pct":100.0*p,
            "baseline_roi_pct":100.0*float(base["return_yen_per100"].sum())/(100.0*len(base)),
            "target_tickets":n,
            "target_hits":observed_hits,
            "target_roi_pct":observed_roi,
            "expected_hits_if_baseline_rate":n*p,
            "prob_zero_hits_if_baseline_rate_pct":100.0*((1.0-p)**n),
        }
        row.update(bootstrap_roi_baseline(
            base["return_yen_per100"].to_numpy(),n,observed_roi,77000+target+len(base)
        ))
        rows.append(row)
    return pd.DataFrame(rows)

def numeric_shift(selected):
    rows=[]
    for cid in FOCUS_CANDIDATES:
        g=selected[(selected["candidate_id"]==cid)&(selected["machine_agree"])].copy()
        years=sorted(g["test_year"].unique().tolist())
        if len(years)<2:
            continue
        target=years[-1]
        base=g[g["test_year"]<target]
        test=g[g["test_year"]==target]
        for col in NUMERIC_FEATURES:
            if col not in g.columns:
                continue
            a=pd.to_numeric(base[col],errors="coerce").dropna()
            b=pd.to_numeric(test[col],errors="coerce").dropna()
            if len(a)<5 or len(b)<5:
                continue
            pooled=np.sqrt((float(a.var(ddof=1))+float(b.var(ddof=1)))/2.0)
            smd=(float(b.mean())-float(a.mean()))/pooled if pooled>1e-12 else 0.0
            rows.append({
                "candidate_id":cid,
                "baseline_years":"|".join(map(str,sorted(base["test_year"].unique()))),
                "target_year":int(target),
                "feature":col,
                "baseline_n":len(a),
                "target_n":len(b),
                "baseline_mean":float(a.mean()),
                "target_mean":float(b.mean()),
                "baseline_median":float(a.median()),
                "target_median":float(b.median()),
                "standardized_mean_diff":smd,
                "abs_standardized_mean_diff":abs(smd),
            })
    return pd.DataFrame(rows)

def categorical_shift(selected):
    rows=[]
    for cid in FOCUS_CANDIDATES:
        g=selected[(selected["candidate_id"]==cid)&(selected["machine_agree"])].copy()
        years=sorted(g["test_year"].unique().tolist())
        if len(years)<2:
            continue
        target=years[-1]
        base=g[g["test_year"]<target]
        test=g[g["test_year"]==target]
        for col in CATEGORICAL_FEATURES:
            if col not in g.columns:
                continue
            av=base[col].fillna("__NA__").astype(str).value_counts(normalize=True)
            bv=test[col].fillna("__NA__").astype(str).value_counts(normalize=True)
            cats=sorted(set(av.index)|set(bv.index))
            diffs={c:float(bv.get(c,0.0)-av.get(c,0.0)) for c in cats}
            tv=0.5*sum(abs(v) for v in diffs.values())
            top=max(cats,key=lambda c:abs(diffs[c])) if cats else None
            rows.append({
                "candidate_id":cid,
                "baseline_years":"|".join(map(str,sorted(base["test_year"].unique()))),
                "target_year":int(target),
                "feature":col,
                "total_variation_distance":tv,
                "largest_shift_category":top,
                "largest_shift_pct_points":100.0*diffs[top] if top is not None else None,
                "baseline_share_pct":100.0*float(av.get(top,0.0)) if top is not None else None,
                "target_share_pct":100.0*float(bv.get(top,0.0)) if top is not None else None,
            })
    return pd.DataFrame(rows)

def segment_performance(selected):
    rows=[]
    z=selected[selected["machine_agree"]].copy()
    dims=("surface","discipline","distance_band","venue_code","track_condition",
          "race_class_normalized","grade","month_band")
    for cid in FOCUS_CANDIDATES:
        g=z[z["candidate_id"]==cid]
        for year in sorted(g["test_year"].unique()):
            gy=g[g["test_year"]==year]
            for dim in dims:
                if dim not in gy.columns:
                    continue
                for val,seg in gy.groupby(dim,dropna=False):
                    if len(seg)<3:
                        continue
                    m=metrics(seg,f"{cid}_{year}_{dim}_{val}")
                    rows.append({
                        "candidate_id":cid,"test_year":int(year),
                        "dimension":dim,"segment":str(val),
                        "tickets":m["tickets"],"hits":m["hits"],
                        "roi_pct":m["roi_pct"],
                        "roi_after_remove_largest_win_pct":m["roi_after_remove_largest_win_pct"],
                        "median_odds":m["median_odds"],
                    })
    return pd.DataFrame(rows)

def main():
    a=parse_args()
    paths={y:getattr(a,f"l17_{y}") for y in YEARS}
    l17={y:load_l17(paths[y],y) for y in YEARS}
    frames={y:add_market_features(build_year_frame(y,l17[y],a.backfill_root)) for y in YEARS}
    cols=market_feature_columns(frames[2022])

    selected,audit=build_selected_rows(frames,cols)

    yearly=[]
    for cid in FOCUS_CANDIDATES:
        g=selected[selected["candidate_id"]==cid]
        for y in sorted(g["test_year"].unique()):
            gy=g[g["test_year"]==y]
            yearly.append(metric_row(gy,cid,int(y),"ALL"))
            yearly.append(metric_row(gy[gy["machine_agree"]],cid,int(y),"MACHINE_AGREE"))
            yearly.append(metric_row(gy[~gy["machine_agree"]],cid,int(y),"MACHINE_DISAGREE"))
    yearly=pd.DataFrame(yearly)

    var=variance_checks(selected)
    nshift=numeric_shift(selected)
    cshift=categorical_shift(selected)
    seg=segment_performance(selected)

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    audit.to_csv(out/"selection-audit.csv",index=False)
    yearly.to_csv(out/"focus-rule-year-summary.csv",index=False)
    var.to_csv(out/"variance-checks.csv",index=False)
    nshift.sort_values(["candidate_id","abs_standardized_mean_diff"],ascending=[True,False]).to_csv(
        out/"numeric-regime-shifts.csv",index=False
    )
    cshift.sort_values(["candidate_id","total_variation_distance"],ascending=[True,False]).to_csv(
        out/"categorical-regime-shifts.csv",index=False
    )
    seg.to_csv(out/"segment-performance.csv",index=False)

    pvar=var[var["candidate_id"]==PRIMARY_CANDIDATE].to_dict(orient="records")
    pnum=nshift[nshift["candidate_id"]==PRIMARY_CANDIDATE].sort_values(
        "abs_standardized_mean_diff",ascending=False
    ).head(8).to_dict(orient="records")
    pcat=cshift[cshift["candidate_id"]==PRIMARY_CANDIDATE].sort_values(
        "total_variation_distance",ascending=False
    ).head(8).to_dict(orient="records")
    summary={
        "contract":"L2_QUINELLA_REGIME_DIAGNOSIS_V1",
        "purpose":"separate regime drift from small-sample variance for recurring candidate rules",
        "focus_candidates":list(FOCUS_CANDIDATES),
        "primary_candidate":PRIMARY_CANDIDATE,
        "primary_variance_check":pvar[0] if pvar else None,
        "primary_top_numeric_shifts":pnum,
        "primary_top_categorical_shifts":pcat,
        "diagnostic_only":True,
        "no_new_rule_selected_from_2025":True,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("L2_QUINELLA_REGIME_DIAGNOSIS_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
