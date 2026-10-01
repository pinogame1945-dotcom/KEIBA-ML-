#!/usr/bin/env python3
import argparse,csv,json,hashlib
from pathlib import Path

import numpy as np
import pandas as pd

YEARS=(2023,2024,2025)
DIRECTIONS=("L1_UPGRADE","L1_DOWNGRADE")
MULTIPLIERS=(0.8,1.0,1.2)
BOOTSTRAPS=2000
BATCH=200

def parse_args():
    p=argparse.ArgumentParser(description="Race-cluster bootstrap and threshold-sensitivity audit for V3 STRONG dissent.")
    p.add_argument("--scored",required=True)
    p.add_argument("--thresholds",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def write_csv(path,rows):
    path=Path(path)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader(); w.writerows(rows)

def stable_seed(*parts):
    h=hashlib.sha256("|".join(map(str,parts)).encode()).hexdigest()
    return int(h[:8],16)

def race_aggregates(df,score_col,threshold):
    q=df.copy()
    sign=1.0 if q["direction"].iloc[0]=="L1_UPGRADE" else -1.0
    q["contrib"]=sign*(q["target_top3"].astype(float)-q["market_peer_top3_rate"].astype(float))
    q["selected"]=(q[score_col].astype(float)>=float(threshold)).astype(int)
    q["sel_contrib"]=q["contrib"]*q["selected"]
    g=q.groupby("race_id",sort=False).agg(
        all_sum=("contrib","sum"),
        all_n=("contrib","size"),
        sel_sum=("sel_contrib","sum"),
        sel_n=("selected","sum"),
    ).reset_index()
    return g

def point_stats(g):
    all_n=float(g["all_n"].sum())
    sel_n=float(g["sel_n"].sum())
    all_eff=100*float(g["all_sum"].sum()/all_n) if all_n else np.nan
    sel_eff=100*float(g["sel_sum"].sum()/sel_n) if sel_n else np.nan
    return all_eff,sel_eff,sel_eff-all_eff,all_n,sel_n

def cluster_bootstrap(g,reps,seed):
    a_sum=g["all_sum"].to_numpy(dtype=float)
    a_n=g["all_n"].to_numpy(dtype=float)
    s_sum=g["sel_sum"].to_numpy(dtype=float)
    s_n=g["sel_n"].to_numpy(dtype=float)
    n=len(g)
    rng=np.random.default_rng(seed)
    deltas=[]; selected=[]
    left=reps
    while left>0:
        b=min(BATCH,left)
        idx=rng.integers(0,n,size=(b,n),endpoint=False)
        ba=a_sum[idx].sum(axis=1)/a_n[idx].sum(axis=1)
        sn=s_n[idx].sum(axis=1)
        valid=sn>0
        bs=np.full(b,np.nan,dtype=float)
        bs[valid]=s_sum[idx][valid].sum(axis=1)/sn[valid]
        selected.extend((100*bs[valid]).tolist())
        deltas.extend((100*(bs[valid]-ba[valid])).tolist())
        left-=b
    d=np.asarray(deltas,dtype=float)
    s=np.asarray(selected,dtype=float)
    return {
        "bootstrap_reps_valid":int(len(d)),
        "delta_ci95_low_pp":float(np.quantile(d,0.025)),
        "delta_ci95_high_pp":float(np.quantile(d,0.975)),
        "selected_effect_ci95_low_pp":float(np.quantile(s,0.025)),
        "selected_effect_ci95_high_pp":float(np.quantile(s,0.975)),
        "p_boot_delta_gt_0":float(np.mean(d>0)),
    }

def main():
    a=parse_args()
    scored=pd.read_csv(a.scored,compression="gzip")
    thr=pd.read_csv(a.thresholds)
    if 2026 in set(pd.to_numeric(scored["year"],errors="coerce").dropna().astype(int)):
        raise SystemExit("2026 sealed")

    scored["year"]=pd.to_numeric(scored["year"],errors="raise").astype(int)
    for c in ("target_top3","market_peer_top3_rate","score_up","score_down"):
        scored[c]=pd.to_numeric(scored[c],errors="raise")

    threshold_map={}
    for r in thr.itertuples(index=False):
        threshold_map[(int(r.test_year),str(r.direction))]=float(r.strong_threshold)

    rows=[]
    for year in YEARS:
        for direction in DIRECTIONS:
            d=scored[(scored["year"]==year)&(scored["direction"]==direction)].copy()
            if d.empty: raise SystemExit(f"missing cell {year} {direction}")
            score_col="score_up" if direction=="L1_UPGRADE" else "score_down"
            base_thr=threshold_map[(year,direction)]
            for mult in MULTIPLIERS:
                threshold=base_thr*mult
                g=race_aggregates(d,score_col,threshold)
                all_eff,sel_eff,delta,all_n,sel_n=point_stats(g)
                boot=cluster_bootstrap(g,BOOTSTRAPS,stable_seed(year,direction,mult))
                rows.append({
                    "test_year":year,
                    "direction":direction,
                    "threshold_multiplier":mult,
                    "base_threshold":base_thr,
                    "tested_threshold":threshold,
                    "all_horses":int(all_n),
                    "selected_horses":int(sel_n),
                    "selected_races":int((g["sel_n"]>0).sum()),
                    "coverage_pct":100*sel_n/all_n,
                    "all_effect_pp":all_eff,
                    "selected_effect_pp":sel_eff,
                    "selected_minus_all_pp":delta,
                    **boot,
                })

    write_csv(Path(a.out_dir)/"uncertainty.csv",rows)
    rdf=pd.DataFrame(rows)

    frozen=rdf[rdf["threshold_multiplier"]==1.0].copy()
    sens=[]
    for direction in DIRECTIONS:
        q=rdf[rdf["direction"]==direction].copy()
        for year in YEARS:
            y=q[q["test_year"]==year].sort_values("threshold_multiplier")
            sens.append({
                "direction":direction,
                "test_year":year,
                "all_three_multiplier_point_positive":bool((y["selected_minus_all_pp"]>0).all()),
                "all_three_multiplier_selected_effect_positive":bool((y["selected_effect_pp"]>0).all()),
                "min_point_delta_pp":float(y["selected_minus_all_pp"].min()),
                "max_point_delta_pp":float(y["selected_minus_all_pp"].max()),
                "coverage_min_pct":float(y["coverage_pct"].min()),
                "coverage_max_pct":float(y["coverage_pct"].max()),
            })
    write_csv(Path(a.out_dir)/"sensitivity-summary.csv",sens)

    # Predeclared promotion standard for this audit:
    # 1) frozen threshold must have positive point enrichment in all 3 years,
    # 2) race-cluster 95% CI lower bound > 0 in all 3 years,
    # 3) point enrichment remains positive at 0.8x/1.0x/1.2x in all 3 years.
    decision={}
    for direction in DIRECTIONS:
        f=frozen[frozen["direction"]==direction].sort_values("test_year")
        s=pd.DataFrame([x for x in sens if x["direction"]==direction]).sort_values("test_year")
        pass_point=bool(len(f)==3 and (f["selected_minus_all_pp"]>0).all())
        pass_ci=bool(len(f)==3 and (f["delta_ci95_low_pp"]>0).all())
        pass_sensitivity=bool(len(s)==3 and s["all_three_multiplier_point_positive"].all())
        decision[direction]={
            "point_enrichment_all_years":pass_point,
            "ci95_lower_above_zero_all_years":pass_ci,
            "threshold_sensitivity_all_years":pass_sensitivity,
            "formal_promotion_pass":bool(pass_point and pass_ci and pass_sensitivity),
            "frozen_rows":f.to_dict(orient="records"),
            "sensitivity_rows":s.to_dict(orient="records"),
        }

    # Pooled descriptive numbers are not used to override the per-year decision.
    pooled=[]
    for direction in DIRECTIONS:
        q=frozen[frozen["direction"]==direction]
        pooled.append({
            "direction":direction,
            "mean_yearly_selected_minus_all_pp":float(q["selected_minus_all_pp"].mean()),
            "min_yearly_selected_minus_all_pp":float(q["selected_minus_all_pp"].min()),
            "max_yearly_selected_minus_all_pp":float(q["selected_minus_all_pp"].max()),
            "total_selected_horses":int(q["selected_horses"].sum()),
        })
    write_csv(Path(a.out_dir)/"pooled-descriptive.csv",pooled)

    summary={
        "contract":"L1_STRONG_DISSENT_UNCERTAINTY_V4",
        "source":"Frozen V3 scored outputs only; no model retraining and no Kaggle/network data dependency.",
        "bootstrap":"race-cluster bootstrap within each test-year/direction cell",
        "bootstrap_reps":BOOTSTRAPS,
        "threshold_sensitivity_multipliers":list(MULTIPLIERS),
        "threshold_sensitivity_policy":"Multipliers fixed before reading V4 outcomes; no threshold optimization.",
        "formal_promotion_rule":[
            "At frozen 1.0x threshold, STRONG-minus-ALL point enrichment > 0 in 2023, 2024, 2025.",
            "At frozen 1.0x threshold, race-cluster bootstrap 95% CI lower bound for STRONG-minus-ALL > 0 in 2023, 2024, 2025.",
            "At 0.8x, 1.0x, 1.2x thresholds, point enrichment remains > 0 in every 2023, 2024, 2025 cell."
        ],
        "decision":decision,
        "core_L1_market_free":True,
        "rerank":False,
        "paid_compute":False,
        "2026_locked":True,
        "promotion":False
    }
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== UNCERTAINTY =====")
    print((out/"uncertainty.csv").read_text(encoding="utf-8"))
    print("===== SENSITIVITY =====")
    print((out/"sensitivity-summary.csv").read_text(encoding="utf-8"))
    print("===== SUMMARY =====")
    print((out/"summary.json").read_text(encoding="utf-8"))
    print("L1_STRONG_DISSENT_UNCERTAINTY_V4_READY")

if __name__=="__main__":
    main()
