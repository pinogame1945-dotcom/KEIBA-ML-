#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import build_year_frame, write_csv
from run_l2_ticket_market_gap_v2 import (
    add_market_features, market_feature_columns, train_rank_predict, add_ranks,
    apply_gap_policy
)

YEARS=(2022,2023,2024)
TEST_YEARS=(2023,2024)
FROZEN_POLICY=(15,5,1)
BANDS=[("5-6",5,7),("7-9",7,10),("10-14",10,15),("15-24",15,25),("25+",25,10**9)]

def args():
    p=argparse.ArgumentParser()
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def metrics(g,label):
    n=len(g)
    stake=100.0*n
    ret=float(g["return_yen_per100"].sum())
    hits=int(g["hit"].sum())
    winners=sorted([float(x) for x in g.loc[g["return_yen_per100"]>0,"return_yen_per100"]],reverse=True)
    largest=winners[0] if winners else 0.0
    removed1=ret-largest
    return {
        "label":label,
        "tickets":n,
        "hits":hits,
        "ticket_hit_rate_pct":100.0*hits/n if n else 0.0,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "largest_single_return_yen":largest,
        "largest_return_share_pct":100.0*largest/ret if ret>0 else 0.0,
        "roi_after_remove_largest_win_pct":100.0*removed1/stake if stake else None,
        "median_odds":float(g["odds"].median()) if n else None,
        "median_market_rank":float(g["market_rank"].median()) if n else None,
        "median_model_rank":float(g["model_rank"].median()) if n else None,
        "median_l17_rank":float(g["l17_rank_score"].median()) if n else None,
    }

def ticket_bootstrap(g,seed=20260930,samples=10000):
    if g.empty:
        return {"samples":samples,"tickets":0,"roi_median":None,"roi_p2_5":None,"roi_p97_5":None,"prob_roi_gt_100_pct":None}
    rng=np.random.default_rng(seed)
    rets=g["return_yen_per100"].to_numpy(dtype=float)
    n=len(rets)
    rois=np.empty(samples,dtype=float)
    chunk=250
    for start in range(0,samples,chunk):
        m=min(chunk,samples-start)
        idx=rng.integers(0,n,size=(m,n))
        rois[start:start+m]=100.0*rets[idx].sum(axis=1)/(100.0*n)
    return {
        "samples":samples,
        "tickets":n,
        "roi_p2_5":float(np.percentile(rois,2.5)),
        "roi_p25":float(np.percentile(rois,25)),
        "roi_median":float(np.percentile(rois,50)),
        "roi_p75":float(np.percentile(rois,75)),
        "roi_p97_5":float(np.percentile(rois,97.5)),
        "prob_roi_gt_100_pct":100.0*float((rois>100).mean()),
    }

def month_rows(g,year,band):
    if g.empty:
        return []
    z=g.copy()
    z["month"]=z["race_date"].astype(str).str[:7]
    rows=[]
    for m,x in z.groupby("month",sort=True):
        r=metrics(x,f"{year}:{band}:{m}")
        r.update({"year":year,"band":band,"month":m})
        rows.append(r)
    return rows

def main():
    a=args()
    paths={y:getattr(a,f"l17_{y}") for y in YEARS}
    l17={y:load_l17(paths[y],y) for y in YEARS}
    frames={y:add_market_features(build_year_frame(y,l17[y],a.backfill_root)) for y in YEARS}
    cols=market_feature_columns(frames[2022])

    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    summary_rows=[]
    month_all=[]
    bootstrap_rows=[]
    full_selected=[]
    target_selected=[]

    for y in TEST_YEARS:
        train=pd.concat([frames[t] for t in YEARS if t<y],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)
        score,_=train_rank_predict(train,test,cols,93000+y)
        test["market_aware_score"]=score
        test=add_ranks(test)
        selected=apply_gap_policy(test,*FROZEN_POLICY).copy()
        selected["year"]=y
        full_selected.append(selected)

        target_selected.append(selected[(selected["rank_upgrade"]>=15)&(selected["rank_upgrade"]<25)].copy())
        for band,lo,hi in BANDS:
            g=selected[(selected["rank_upgrade"]>=lo)&(selected["rank_upgrade"]<hi)].copy()
            r=metrics(g,band)
            r.update({"year":y,"band":band})
            summary_rows.append(r)
            b=ticket_bootstrap(g,seed=20260930+y+lo)
            b.update({"year":y,"band":band})
            bootstrap_rows.append(b)
            month_all.extend(month_rows(g,y,band))

    # Cross-year pooled band results, diagnostic only.
    pooled=pd.concat(full_selected,ignore_index=True)
    pooled_rows=[]
    for band,lo,hi in BANDS:
        g=pooled[(pooled["rank_upgrade"]>=lo)&(pooled["rank_upgrade"]<hi)].copy()
        r=metrics(g,band)
        r.update({"year":"2023+2024","band":band})
        pooled_rows.append(r)

    write_csv(out/"gap-band-by-year.csv",summary_rows)
    write_csv(out/"gap-band-bootstrap.csv",bootstrap_rows)
    write_csv(out/"gap-band-monthly.csv",month_all)
    write_csv(out/"gap-band-pooled-dev.csv",pooled_rows)

    target_df=pd.concat(target_selected,ignore_index=True)
    target_df["distance_band"]=pd.cut(
        target_df["distance_m"],
        [-1,1400,1800,2200,2600,10**9],
        labels=["<=1400","1401-1800","1801-2200","2201-2600","2601+"],
        right=True
    ).astype(str)
    target_df["field_size_band"]=pd.cut(
        target_df["field_size"],
        [-1,10,12,14,16,10**9],
        labels=["<=10","11-12","13-14","15-16","17+"],
        right=True
    ).astype(str)
    target_df["odds_band"]=pd.cut(
        target_df["odds"],
        [-1,20,40,60,80,100,10**9],
        labels=["<20","20-40","40-60","60-80","80-100","100+"],
        right=False
    ).astype(str)
    target_df["market_rank_band"]=pd.cut(
        target_df["market_rank"],
        [-1,15,20,25,30,40,10**9],
        labels=["<=15","16-20","21-25","26-30","31-40","41+"],
        right=True
    ).astype(str)
    target_df["model_rank_band"]=pd.cut(
        target_df["model_rank"],
        [-1,3,5,8,10,15],
        labels=["1-3","4-5","6-8","9-10","11-15"],
        right=True
    ).astype(str)
    target_df["l17_rank_band"]=pd.cut(
        target_df["l17_rank_score"],
        [-1,10,20,30,40,60,10**9],
        labels=["<=10","11-20","21-30","31-40","41-60","61+"],
        right=True
    ).astype(str)

    context_rows=[]
    for y in TEST_YEARS:
        yy=target_df[target_df["year"]==y]
        for col in ["surface","distance_band","field_size_band","odds_band","market_rank_band","model_rank_band","l17_rank_band"]:
            for val,g in yy.groupby(col,dropna=False,sort=True):
                r=metrics(g,f"{y}:{col}:{val}")
                r.update({"year":y,"dimension":col,"value":str(val)})
                context_rows.append(r)
    write_csv(out/"target-15-24-context.csv",context_rows)
    keep=[
        "year","race_id","race_date","pair_numbers","rank_upgrade","odds",
        "market_rank","model_rank","l17_rank_score","a_consensus_rank","b_consensus_rank",
        "surface","distance_m","field_size","hit","return_yen_per100"
    ]
    target_df[keep].to_csv(out/"target-15-24-tickets.csv",index=False)

    target=[r for r in summary_rows if r["band"]=="15-24"]
    target_month=[r for r in month_all if r["band"]=="15-24"]
    month_stats={}
    for y in TEST_YEARS:
        rows=[r for r in target_month if r["year"]==y]
        month_stats[str(y)]={
            "months_with_tickets":len(rows),
            "positive_roi_months":sum(1 for r in rows if r["roi_pct"] is not None and r["roi_pct"]>100),
            "profitable_yen_months":sum(1 for r in rows if r["profit_yen"]>0),
        }

    summary={
        "contract":"L2_MARKET_GAP_BAND_AUDIT_V1_RESULT",
        "frozen_policy":{"model_top_k":15,"min_market_rank_upgrade":5,"max_tickets_per_race":1},
        "policy_reselected":False,
        "test_years":[2023,2024],
        "target_band":"15-24",
        "target_band_by_year":target,
        "target_band_month_stats":month_stats,
        "interpretation_rule":"Do not promote the 15-24 band from this audit alone; it was identified after inspecting 2025 and then checked retrospectively.",
        "2026_locked":True
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_MARKET_GAP_BAND_AUDIT_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
