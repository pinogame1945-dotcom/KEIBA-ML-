#!/usr/bin/env python3
import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

VALID_YEARS=(2023,2024,2025)
FORBIDDEN_TOKENS=("odds","popularity","payout","finish","result","target","return","profit","roi")

def parse_args():
    p=argparse.ArgumentParser(description="Feature autopsy inside triple-agreement anchor races.")
    p.add_argument("--year",type=int,required=True)
    p.add_argument("--market-scored",required=True)
    p.add_argument("--outsider-predictions",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def safe_numeric(v):
    if isinstance(v,bool):
        return float(v)
    if isinstance(v,(int,float,np.integer,np.floating)):
        x=float(v)
        return x if math.isfinite(x) else None
    return None

def category(name):
    s=name.lower()
    if "style" in s or "position" in s or "corner" in s: return "STYLE_POSITION"
    if "lap" in s or "pace" in s or "time" in s or "last_3f" in s: return "PACE_TIME"
    if "distance" in s or "distx" in s: return "DISTANCE"
    if "weight" in s or "body" in s: return "WEIGHT_BODY"
    if "gate" in s or "frame" in s or "draw" in s: return "DRAW"
    if "jockey" in s or "trainer" in s or "actor" in s: return "ACTOR"
    if "opponent" in s or "field" in s or "relative" in s: return "FIELD_OPPONENT"
    if "surface" in s or "track" in s or "weather" in s or "venue" in s or "course" in s: return "RACE_CONDITION"
    if "recent" in s or "history" in s or "prior" in s or "streak" in s or "margin" in s: return "HISTORY_FRAGILITY"
    return "OTHER"

def load_anchors(year, market_path, outsider_path):
    mcols=["year","race_id","horse_id","horse_number","consensus_rank","market_rank","final_win_odds","target_top3"]
    pcols=["year","race_id","horse_id","rank","score","p3","p3_rank_baseline","p3_delta"]
    m=pd.read_csv(market_path,compression="gzip",usecols=mcols)
    p=pd.read_csv(outsider_path,compression="gzip",usecols=pcols)
    m["year"]=pd.to_numeric(m["year"],errors="raise").astype(int)
    p["year"]=pd.to_numeric(p["year"],errors="raise").astype(int)
    if 2026 in set(m["year"]) or 2026 in set(p["year"]): raise SystemExit("2026 sealed")
    m=m[m["year"]==year].copy()
    p=p[p["year"]==year].copy()
    for c in ("consensus_rank","market_rank","final_win_odds","target_top3"):
        m[c]=pd.to_numeric(m[c],errors="coerce")
    for c in ("rank","score","p3","p3_rank_baseline","p3_delta"):
        p[c]=pd.to_numeric(p[c],errors="coerce")
    keys=["year","race_id","horse_id"]
    if m.duplicated(keys).any() or p.duplicated(keys).any(): raise SystemExit("duplicate source keys")
    z=m.merge(p,on=keys,how="inner",validate="one_to_one")
    q=z[(z["consensus_rank"]==1)&(z["market_rank"]==1)&(z["rank"]==1)&(z["p3_delta"]>0)&z["target_top3"].notna()].copy()
    q["collapse"]=(q["target_top3"].astype(int)==0).astype(int)
    return q

def rank_auc(values, labels):
    v=np.asarray(values,dtype=float)
    y=np.asarray(labels,dtype=int)
    mask=np.isfinite(v)
    v=v[mask]; y=y[mask]
    n1=int((y==1).sum()); n0=int((y==0).sum())
    if n1<2 or n0<2: return None
    ranks=pd.Series(v).rank(method="average").to_numpy()
    u=float(ranks[y==1].sum()-n1*(n1+1)/2)
    return u/(n1*n0)

def scan_scope(df, scope):
    out=[]
    y=df["collapse"].to_numpy(dtype=int)
    n1=int(y.sum()); n0=int(len(y)-n1)
    if n1<10 or n0<30:
        return out
    feature_cols=[c for c in df.columns if c.startswith("f__")]
    for c in feature_cols:
        vals=pd.to_numeric(df[c],errors="coerce")
        valid=vals.notna()
        if int(valid.sum())<max(80,int(0.50*len(df))): continue
        q=pd.DataFrame({"x":vals[valid],"y":df.loc[valid,"collapse"].astype(int)})
        c1=q[q["y"]==1]["x"]; c0=q[q["y"]==0]["x"]
        if len(c1)<10 or len(c0)<30: continue
        m1=float(c1.mean()); m0=float(c0.mean())
        s1=float(c1.std(ddof=1)) if len(c1)>1 else 0.0
        s0=float(c0.std(ddof=1)) if len(c0)>1 else 0.0
        pooled=math.sqrt(((len(c1)-1)*s1*s1+(len(c0)-1)*s0*s0)/max(1,len(c1)+len(c0)-2))
        smd=(m1-m0)/pooled if pooled>1e-12 else 0.0
        auc=rank_auc(q["x"].to_numpy(),q["y"].to_numpy())
        predictive=max(auc,1-auc) if auc is not None else None
        name=c[3:]
        out.append({
            "scope":scope,
            "feature":name,
            "category":category(name),
            "anchors":len(df),
            "collapse_n":n1,
            "survive_n":n0,
            "observed_n":int(valid.sum()),
            "coverage_pct":100*float(valid.mean()),
            "collapse_mean":m1,
            "survive_mean":m0,
            "collapse_median":float(c1.median()),
            "survive_median":float(c0.median()),
            "standardized_mean_diff":smd,
            "abs_smd":abs(smd),
            "auc_larger_means_collapse":auc,
            "predictive_auc":predictive,
            "direction":"HIGHER_COLLAPSE" if smd>0 else ("LOWER_COLLAPSE" if smd<0 else "FLAT"),
            "collapse_missing_pct":100*float(df.loc[df["collapse"]==1,c].isna().mean()),
            "survive_missing_pct":100*float(df.loc[df["collapse"]==0,c].isna().mean()),
        })
    return out

def main():
    a=parse_args()
    if a.year not in VALID_YEARS or a.year==2026: raise SystemExit("2026 sealed")
    anchors=load_anchors(a.year,a.market_scored,a.outsider_predictions)
    wanted={(str(r.race_id),str(r.horse_id)):r for r in anchors.itertuples(index=False)}
    rows=[]
    found=set()
    snapshot_path=Path(a.snapshot)
    with open_text(snapshot_path) as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            key=(str(r.get("race_id") or ""),str(r.get("horse_id") or ""))
            meta=wanted.get(key)
            if meta is None: continue
            feats=r.get("features") or {}
            row={
                "year":a.year,
                "race_id":key[0],
                "horse_id":key[1],
                "horse_number":getattr(meta,"horse_number"),
                "final_win_odds":float(getattr(meta,"final_win_odds")),
                "p3_delta":float(getattr(meta,"p3_delta")),
                "collapse":int(getattr(meta,"collapse")),
            }
            for k,v in feats.items():
                low=str(k).lower()
                if any(t in low for t in FORBIDDEN_TOKENS): continue
                x=safe_numeric(v)
                if x is not None: row["f__"+str(k)]=x
            rows.append(row); found.add(key)
    if not rows: raise SystemExit("no anchor rows matched snapshot")
    coverage=len(found)/len(wanted)
    if coverage<0.90:
        raise SystemExit(f"snapshot anchor coverage too low {len(found)}/{len(wanted)}")

    df=pd.DataFrame(rows)
    scopes=[
        ("TRIPLE_ALL",df),
        ("MARKET_ODDS_LE_2.0",df[df["final_win_odds"]<=2.0].copy()),
        ("MARKET_ODDS_LE_1.5",df[df["final_win_odds"]<=1.5].copy()),
    ]
    scanned=[]
    scope_summary=[]
    for name,q in scopes:
        n=len(q); c=int(q["collapse"].sum()) if n else 0
        scope_summary.append({
            "scope":name,"anchors":n,"collapse_n":c,"survive_n":n-c,
            "collapse_rate_pct":100*c/n if n else None,
        })
        scanned.extend(scan_scope(q,name))

    sdf=pd.DataFrame(scanned)
    if sdf.empty: raise SystemExit("feature scan empty")
    sdf=sdf.sort_values(["scope","predictive_auc","abs_smd"],ascending=[True,False,False])
    top=sdf[(sdf["predictive_auc"]>=0.56)|(sdf["abs_smd"]>=0.20)].copy()
    top=top.sort_values(["scope","predictive_auc","abs_smd"],ascending=[True,False,False])

    cats=[]
    for (scope,cat),g in sdf.groupby(["scope","category"],sort=True):
        cats.append({
            "scope":scope,"category":cat,"features_tested":len(g),
            "max_predictive_auc":float(g["predictive_auc"].max()),
            "max_abs_smd":float(g["abs_smd"].max()),
            "signals_auc_ge_0.56":int((g["predictive_auc"]>=0.56).sum()),
            "signals_abs_smd_ge_0.20":int((g["abs_smd"]>=0.20).sum()),
        })

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(scope_summary).to_csv(out/"scope-summary.csv",index=False)
    sdf.to_csv(out/"feature-scan.csv",index=False)
    top.head(120).to_csv(out/"top-signals.csv",index=False)
    pd.DataFrame(cats).sort_values(["scope","max_predictive_auc"],ascending=[True,False]).to_csv(out/"category-summary.csv",index=False)

    summary={
        "contract":"L1_ANCHOR_COLLAPSE_FEATURE_AUTOPSY_V1",
        "year":a.year,
        "anchor_definition":"Seven-King rank1 + tie-safe market rank1 + frozen Outsider p3_delta>0",
        "snapshot":snapshot_path.name,
        "anchors_source":int(len(anchors)),
        "anchors_matched_snapshot":int(len(found)),
        "anchor_coverage_pct":100*coverage,
        "numeric_features_tested":int(sdf["feature"].nunique()),
        "scopes":[x[0] for x in scopes],
        "purpose":"Exploratory autopsy only; identify pre-race feature differences inside unanimous anchors before fitting a collapse model.",
        "causal_claim":False,
        "outcome_optimized_thresholds":False,
        "paid_compute":False,
        "2026_locked":True,
        "promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== SUMMARY ====="); print((out/"summary.json").read_text())
    print("===== SCOPES ====="); print((out/"scope-summary.csv").read_text())
    print("===== TOP SIGNALS ====="); print("\n".join((out/"top-signals.csv").read_text().splitlines()[:41]))
    print("===== CATEGORY SUMMARY ====="); print((out/"category-summary.csv").read_text())
    print("L1_ANCHOR_COLLAPSE_FEATURE_AUTOPSY_READY")

if __name__=="__main__":
    main()
