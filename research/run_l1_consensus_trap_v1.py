#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

YEARS=(2023,2024,2025)
EPS=1e-12
BOOTSTRAPS=2000
RNG_SEED=20261004

def parse_args():
    p=argparse.ArgumentParser(description="Test whether extreme three-way agreement behaves like a consensus trap.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--outsider-predictions",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def write_csv(path, rows):
    path=Path(path)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader(); w.writerows(rows)

def odds_band(x):
    x=float(x)
    if x<=1.5: return "LE_1.5"
    if x<=2.0: return "GT_1.5_LE_2.0"
    if x<=3.0: return "GT_2.0_LE_3.0"
    return "GT_3.0"

def support_label(delta,lo,hi):
    x=float(delta)
    if x<=EPS: return "NO_SUPPORT"
    if x<lo: return "SUPPORT_LOW"
    if x<hi: return "SUPPORT_MID"
    return "SUPPORT_HIGH"

def load_sources(market_path, outsider_path):
    mcols=["year","race_id","horse_id","horse_number","consensus_rank","market_rank","final_win_odds","target_top3"]
    pcols=["year","race_id","horse_id","rank","score","p3","p3_rank_baseline","p3_delta"]
    m=pd.read_csv(market_path,compression="gzip",usecols=mcols)
    p=pd.read_csv(outsider_path,compression="gzip",usecols=pcols)
    m["year"]=pd.to_numeric(m["year"],errors="raise").astype(int)
    p["year"]=pd.to_numeric(p["year"],errors="raise").astype(int)
    if 2026 in set(m["year"]) or 2026 in set(p["year"]): raise SystemExit("2026 sealed")
    m=m[m["year"].isin(YEARS)].copy()
    p=p[p["year"].isin(YEARS)].copy()
    for c in ("consensus_rank","market_rank","final_win_odds","target_top3"):
        m[c]=pd.to_numeric(m[c],errors="coerce")
    for c in ("rank","score","p3","p3_rank_baseline","p3_delta"):
        p[c]=pd.to_numeric(p[c],errors="coerce")
    keys=["year","race_id","horse_id"]
    if m.duplicated(keys).any(): raise SystemExit("duplicate market keys")
    if p.duplicated(keys).any(): raise SystemExit("duplicate outsider keys")
    z=m.merge(p,on=keys,how="inner",validate="one_to_one")
    z=z[
        (z["market_rank"]==1)
        & (z["consensus_rank"]==1)
        & (z["rank"]==1)
        & z["target_top3"].notna()
        & z["final_win_odds"].notna()
        & z["p3_delta"].notna()
    ].copy()
    if z.empty: raise SystemExit("empty market+king cohort")
    z["collapse"]=(z["target_top3"].astype(int)==0).astype(int)
    z["odds_band"]=[odds_band(x) for x in z["final_win_odds"]]
    z["odds_key"]=z["final_win_odds"].round(1)

    # Support thresholds are label-free and learned separately within each year.
    z["support_label"]=""
    threshold_rows=[]
    for year in YEARS:
        mask=z["year"]==year
        pos=z.loc[mask & (z["p3_delta"]>EPS),"p3_delta"].to_numpy(dtype=float)
        if len(pos)<30: raise SystemExit(f"too few positive support rows year={year}")
        lo,hi=np.quantile(pos,[1/3,2/3])
        z.loc[mask,"support_label"]=[
            support_label(x,float(lo),float(hi)) for x in z.loc[mask,"p3_delta"]
        ]
        threshold_rows.append({
            "year":year,"positive_support_rows":len(pos),
            "support_low_mid_threshold":float(lo),
            "support_mid_high_threshold":float(hi),
        })

    # Market-adjusted residual: compare against same year and exact one-decimal WIN odds.
    peer=z.groupby(["year","odds_key"],dropna=False)["collapse"].agg(["mean","size"]).reset_index()
    peer=peer.rename(columns={"mean":"market_peer_collapse_rate","size":"market_peer_n"})
    z=z.merge(peer,on=["year","odds_key"],how="left",validate="many_to_one")
    z["collapse_residual_vs_exact_odds"]=z["collapse"]-z["market_peer_collapse_rate"]

    # Non-outcome percentile labels for visual pressure curve.
    parts=[]
    for year,g in z.groupby("year",sort=True):
        q=g.copy()
        q["market_strength_pct"]=(-q["final_win_odds"]).rank(method="average",pct=True)
        q["outsider_support_pct"]=q["p3_delta"].rank(method="average",pct=True)
        q["consensus_pressure"]=0.5*(q["market_strength_pct"]+q["outsider_support_pct"])
        q["pressure_decile"]=np.ceil(q["consensus_pressure"]*10).clip(1,10).astype(int)
        q["support_decile"]=np.ceil(q["outsider_support_pct"]*10).clip(1,10).astype(int)
        parts.append(q)
    z=pd.concat(parts,ignore_index=True)
    return z,threshold_rows

def summary_row(scope,sub):
    n=len(sub)
    c=int(sub["collapse"].sum()) if n else 0
    return {
        "scope":scope,
        "horses":n,
        "races":sub["race_id"].nunique() if n else 0,
        "collapse_n":c,
        "collapse_rate_pct":100*c/n if n else None,
        "mean_final_win_odds":float(sub["final_win_odds"].mean()) if n else None,
        "median_final_win_odds":float(sub["final_win_odds"].median()) if n else None,
        "mean_p3_delta":float(sub["p3_delta"].mean()) if n else None,
        "odds_adjusted_residual_pp":100*float(sub["collapse_residual_vs_exact_odds"].mean()) if n else None,
    }

def support_summaries(z):
    rows=[]
    order=["NO_SUPPORT","SUPPORT_LOW","SUPPORT_MID","SUPPORT_HIGH"]
    for year in YEARS:
        y=z[z["year"]==year]
        for band in ("ALL","LE_1.5","GT_1.5_LE_2.0","GT_2.0_LE_3.0","GT_3.0"):
            b=y if band=="ALL" else y[y["odds_band"]==band]
            for label in order:
                q=b[b["support_label"]==label]
                r=summary_row(f"{year}:{band}:{label}",q)
                r.update({"year":year,"odds_band":band,"support_label":label})
                rows.append(r)
    for band in ("ALL","LE_1.5","GT_1.5_LE_2.0","GT_2.0_LE_3.0","GT_3.0"):
        b=z if band=="ALL" else z[z["odds_band"]==band]
        for label in order:
            q=b[b["support_label"]==label]
            r=summary_row(f"POOLED:{band}:{label}",q)
            r.update({"year":"POOLED","odds_band":band,"support_label":label})
            rows.append(r)
    return rows

def curve_rows(z,col):
    rows=[]
    for year_label,base in [(str(y),z[z["year"]==y]) for y in YEARS]+[("POOLED",z)]:
        for d,g in base.groupby(col,sort=True):
            rows.append({
                "year":year_label,
                "curve":col,
                "bucket":int(d),
                "horses":len(g),
                "collapse_n":int(g["collapse"].sum()),
                "collapse_rate_pct":100*float(g["collapse"].mean()),
                "mean_final_win_odds":float(g["final_win_odds"].mean()),
                "mean_p3_delta":float(g["p3_delta"].mean()),
                "odds_adjusted_residual_pp":100*float(g["collapse_residual_vs_exact_odds"].mean()),
            })
    return rows

def matched_effect(z, high_label, ref_label):
    cells=[]
    for (year,odds),g in z.groupby(["year","odds_key"],sort=True):
        a=g[g["support_label"]==high_label]
        b=g[g["support_label"]==ref_label]
        if len(a)<2 or len(b)<2: continue
        ra=float(a["collapse"].mean()); rb=float(b["collapse"].mean())
        w=min(len(a),len(b))
        cells.append({
            "year":int(year),"odds":float(odds),"high_n":len(a),"ref_n":len(b),
            "high_rate":ra,"ref_rate":rb,"diff":ra-rb,"weight":w,
        })
    if not cells:
        return None,[]
    w=np.asarray([c["weight"] for c in cells],dtype=float)
    d=np.asarray([c["diff"] for c in cells],dtype=float)
    return float(np.average(d,weights=w)),cells

def bootstrap_matched(z, high_label, ref_label):
    # Exact race-cluster bootstrap, vectorized through per-race contribution matrices.
    q=z[z["support_label"].isin([high_label,ref_label])].copy()
    if q.empty: return {}

    q["cell_key"]=q["year"].astype(str)+"|"+q["odds_key"].astype(str)
    race_ids=sorted(q["race_id"].astype(str).unique())
    cell_keys=sorted(q["cell_key"].astype(str).unique())
    race_ix={x:i for i,x in enumerate(race_ids)}
    cell_ix={x:i for i,x in enumerate(cell_keys)}
    R=len(race_ids); C=len(cell_keys)
    if R<20 or C==0: return {}

    counts=np.zeros((R,C,2),dtype=np.float64)
    sums=np.zeros((R,C,2),dtype=np.float64)
    for r in q.itertuples(index=False):
        ri=race_ix[str(r.race_id)]
        ci=cell_ix[str(r.cell_key)]
        li=1 if str(r.support_label)==high_label else 0
        counts[ri,ci,li]+=1.0
        sums[ri,ci,li]+=float(r.collapse)

    counts2=counts.reshape(R,C*2)
    sums2=sums.reshape(R,C*2)
    probs=np.full(R,1.0/R,dtype=np.float64)
    rng=np.random.default_rng(RNG_SEED+sum(ord(c) for c in high_label+ref_label))
    vals=[]
    batch=100
    left=BOOTSTRAPS
    while left>0:
        b=min(batch,left)
        weights=rng.multinomial(R,probs,size=b).astype(np.float64,copy=False)
        bc=(weights@counts2).reshape(b,C,2)
        bs=(weights@sums2).reshape(b,C,2)
        ref_n=bc[:,:,0]; high_n=bc[:,:,1]
        valid=(ref_n>=2)&(high_n>=2)
        ref_rate=np.divide(bs[:,:,0],ref_n,out=np.zeros_like(ref_n),where=ref_n>0)
        high_rate=np.divide(bs[:,:,1],high_n,out=np.zeros_like(high_n),where=high_n>0)
        w=np.minimum(ref_n,high_n)*valid
        denom=w.sum(axis=1)
        num=((high_rate-ref_rate)*w).sum(axis=1)
        good=denom>0
        vals.extend((num[good]/denom[good]).tolist())
        left-=b

    if not vals: return {}
    a=np.asarray(vals,dtype=float)
    return {
        "bootstrap_valid":len(a),
        "ci95_low_pp":100*float(np.quantile(a,0.025)),
        "ci95_high_pp":100*float(np.quantile(a,0.975)),
        "p_effect_gt_0":float(np.mean(a>0)),
    }

def comparison_rows(z):
    comps=[
        ("SUPPORT_HIGH","NO_SUPPORT"),
        ("SUPPORT_HIGH","SUPPORT_LOW"),
        ("ALL_SUPPORT","NO_SUPPORT"),
    ]
    rows=[]
    for scope_name,base in [
        ("ALL",z),
        ("ODDS_LE_2.0",z[z["final_win_odds"]<=2.0]),
        ("ODDS_LE_1.5",z[z["final_win_odds"]<=1.5]),
    ]:
        for high,ref in comps:
            q=base.copy()
            if high=="ALL_SUPPORT":
                q=q.copy()
                q["support_label2"]=np.where(q["p3_delta"]>EPS,"ALL_SUPPORT","NO_SUPPORT")
                use=q.rename(columns={"support_label":"support_label_orig","support_label2":"support_label"})
                h="ALL_SUPPORT"
            else:
                use=q; h=high
            eff,cells=matched_effect(use,h,ref)
            if eff is None:
                rows.append({"scope":scope_name,"high":h,"reference":ref,"matched_cells":0})
                continue
            boot=bootstrap_matched(use,h,ref)
            rows.append({
                "scope":scope_name,
                "high":h,
                "reference":ref,
                "matched_cells":len(cells),
                "matched_high_horses":sum(c["high_n"] for c in cells),
                "matched_ref_horses":sum(c["ref_n"] for c in cells),
                "odds_matched_collapse_diff_pp":100*eff,
                **boot,
            })
    return rows

def main():
    a=parse_args()
    z,thresholds=load_sources(a.market_scored,a.outsider_predictions)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    support=support_summaries(z)
    pressure=curve_rows(z,"pressure_decile")
    support_curve=curve_rows(z,"support_decile")
    comparisons=comparison_rows(z)

    write_csv(out/"support-summary.csv",support)
    write_csv(out/"pressure-curve.csv",pressure)
    write_csv(out/"support-curve.csv",support_curve)
    write_csv(out/"odds-matched-comparisons.csv",comparisons)
    write_csv(out/"support-thresholds.csv",thresholds)

    compact=[]
    for y in list(YEARS)+["POOLED"]:
        q=z if y=="POOLED" else z[z["year"]==y]
        for scope,sub in [
            ("ALL",q),
            ("ODDS_LE_2.0",q[q["final_win_odds"]<=2.0]),
            ("ODDS_LE_1.5",q[q["final_win_odds"]<=1.5]),
        ]:
            hi=sub[sub["support_label"]=="SUPPORT_HIGH"]
            no=sub[sub["support_label"]=="NO_SUPPORT"]
            compact.append({
                "year":y,"scope":scope,
                "high_n":len(hi),"high_collapse_pct":100*float(hi["collapse"].mean()) if len(hi) else None,
                "no_support_n":len(no),"no_support_collapse_pct":100*float(no["collapse"].mean()) if len(no) else None,
                "high_minus_no_raw_pp":100*(float(hi["collapse"].mean())-float(no["collapse"].mean())) if len(hi) and len(no) else None,
                "high_odds_adjusted_residual_pp":100*float(hi["collapse_residual_vs_exact_odds"].mean()) if len(hi) else None,
                "no_support_odds_adjusted_residual_pp":100*float(no["collapse_residual_vs_exact_odds"].mean()) if len(no) else None,
            })
    write_csv(out/"headline.csv",compact)

    summary={
        "contract":"L1_CONSENSUS_TRAP_V1",
        "hypothesis":"Among horses already agreed upon by market rank1 and Seven-King rank1, stronger Outsider reinforcement may identify a non-monotonic consensus trap rather than additional safety.",
        "cohort":"market_rank==1 AND reconstructed consensus_rank==1 AND frozen historical king rank==1",
        "outsider_agreement":"p3_delta > 0; positive rows split into label-free within-year tertiles",
        "market_control":"exact final WIN odds at one-decimal resolution within test year; outcome used only to estimate peer collapse rate",
        "primary_comparisons":[
            "SUPPORT_HIGH vs NO_SUPPORT",
            "SUPPORT_HIGH vs SUPPORT_LOW",
            "ALL_SUPPORT vs NO_SUPPORT",
        ],
        "bootstrap":"race-cluster bootstrap of exact-odds matched cell differences",
        "bootstrap_reps":BOOTSTRAPS,
        "pressure_curve":"descriptive only: average of within-year market-strength percentile and outsider-support percentile while Seven-King rank is fixed at 1",
        "horses":int(len(z)),
        "races":int(z["race_id"].nunique()),
        "collapse_rate_pct":100*float(z["collapse"].mean()),
        "years":list(YEARS),
        "2026_locked":True,
        "paid_compute":False,
        "artifacts_or_cache":False,
        "promotion":False,
        "causal_claim":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== SUMMARY ====="); print((out/"summary.json").read_text())
    print("===== HEADLINE ====="); print((out/"headline.csv").read_text())
    print("===== ODDS MATCHED ====="); print((out/"odds-matched-comparisons.csv").read_text())
    print("===== PRESSURE CURVE ====="); print((out/"pressure-curve.csv").read_text())
    print("L1_CONSENSUS_TRAP_V1_READY")

if __name__=="__main__":
    main()
