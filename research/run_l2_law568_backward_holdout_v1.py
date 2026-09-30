#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import build_year_frame, write_csv
from run_l2_ticket_market_gap_v2 import (
    add_market_features, market_feature_columns, train_rank_predict, add_ranks
)
from run_l2_market_gap_law_arena_v1 import (
    add_bins, metrics, pick_law1, candidate_group_tables, chosen_from_table,
    union_frame
)

YEARS=(2020,2021)
HOLDOUT=2021
BOOTSTRAPS=10000

LAW_SPECS={
    "LAW2":("GAP10P",(("model_rank_band","1-3"),)),
    "LAW5":("GAP7_9",(("field_size_band","15-16"),("model_rank_band","9-10"))),
    "LAW6":("GAP7_9",(("market_rank_band","11-15"),("l17_rank_band","21-30"))),
    "LAW8":("GAP5P",(("field_size_band","<=12"),("l17_rank_band","21-30"))),
}

def parse_args():
    p=argparse.ArgumentParser()
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def stress_metrics(g,label):
    m=metrics(g,label)
    wins=sorted(
        [float(x) for x in g.loc[g["return_yen_per100"]>0,"return_yen_per100"]],
        reverse=True
    )
    stake=100.0*len(g)
    top1=wins[0] if wins else 0.0
    top2=sum(wins[:2]) if wins else 0.0
    total=float(g["return_yen_per100"].sum()) if len(g) else 0.0
    m["roi_after_remove_largest_2_wins_pct"]=100.0*(total-top2)/stake if stake else None
    m["largest_2_return_share_pct"]=100.0*top2/total if total>0 else 0.0
    return m

def bootstrap(g,seed):
    n=len(g)
    if n==0:
        return {
            "tickets":0,"median_roi_pct":None,"p_roi_gt_100":None,
            "p2_5_roi_pct":None,"p97_5_roi_pct":None
        }
    rng=np.random.default_rng(seed)
    returns=g["return_yen_per100"].astype(float).to_numpy()
    idx=rng.integers(0,n,size=(BOOTSTRAPS,n))
    roi=returns[idx].sum(axis=1)/(100.0*n)*100.0
    return {
        "tickets":n,
        "median_roi_pct":float(np.median(roi)),
        "p_roi_gt_100":float(np.mean(roi>100.0)),
        "p2_5_roi_pct":float(np.quantile(roi,0.025)),
        "p97_5_roi_pct":float(np.quantile(roi,0.975)),
    }

def month_rows(g,law_id):
    if g.empty:
        return []
    z=g.copy()
    z["month"]=pd.to_datetime(z["race_date"]).dt.strftime("%Y-%m")
    rows=[]
    for month,gg in z.groupby("month",sort=True):
        m=metrics(gg,law_id)
        m.update({"law_id":law_id,"month":month})
        rows.append(m)
    return rows

def main():
    a=parse_args()
    paths={y:getattr(a,f"l17_{y}") for y in YEARS}
    l17={y:load_l17(paths[y],y) for y in YEARS}
    frames={y:add_market_features(build_year_frame(y,l17[y],a.backfill_root)) for y in YEARS}
    cols=market_feature_columns(frames[2020])

    train=frames[2020].copy().reset_index(drop=True)
    test=frames[2021].copy().reset_index(drop=True)
    score,_=train_rank_predict(train,test,cols,95021)
    test["market_aware_score"]=score
    test=add_bins(add_ranks(test))

    tables=candidate_group_tables(test)
    laws={"LAW1":pick_law1(test)}
    for law_id,(gap,conds) in LAW_SPECS.items():
        laws[law_id]=chosen_from_table(tables,gap,conds)

    rows=[]
    boot=[]
    months=[]
    for i,(law_id,g) in enumerate(laws.items(),start=1):
        m=stress_metrics(g,law_id)
        m["law_id"]=law_id
        rows.append(m)
        b=bootstrap(g,20210930+i)
        b["law_id"]=law_id
        boot.append(b)
        if law_id in {"LAW5","LAW6","LAW8"}:
            months.extend(month_rows(g,law_id))

    baseline=union_frame(laws["LAW1"],laws["LAW2"])
    extended=baseline.copy()
    for lid in ("LAW5","LAW6","LAW8"):
        extended=union_frame(extended,laws[lid])

    portfolios=[]
    for pid,g in [
        ("LAW1_LAW2",baseline),
        ("LAW1_LAW2_LAW5_LAW6_LAW8",extended),
    ]:
        m=stress_metrics(g,pid)
        m["portfolio_id"]=pid
        portfolios.append(m)

    month_df=pd.DataFrame(months)
    month_summary=[]
    for law_id in ("LAW5","LAW6","LAW8"):
        x=month_df[month_df["law_id"]==law_id] if not month_df.empty else pd.DataFrame()
        month_summary.append({
            "law_id":law_id,
            "months_with_tickets":int(len(x)),
            "profitable_months":int((x["roi_pct"]>100.0).sum()) if len(x) else 0,
            "months_roi_ge_90":int((x["roi_pct"]>=90.0).sum()) if len(x) else 0,
            "zero_hit_months":int((x["hits"]==0).sum()) if len(x) else 0,
        })

    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"law-2021-metrics.csv",rows)
    write_csv(out/"law-2021-bootstrap.csv",boot)
    if months:
        month_df.to_csv(out/"law-2021-by-month.csv",index=False)
    write_csv(out/"law-2021-month-summary.csv",month_summary)
    write_csv(out/"portfolio-2021-metrics.csv",portfolios)

    keep=[
        "race_id","race_date","pair_numbers","rank_upgrade","market_rank","model_rank",
        "l17_rank_score","field_size","odds","hit","return_yen_per100"
    ]
    for law_id,g in laws.items():
        g[keep].to_csv(out/f"{law_id.lower()}-2021-tickets.csv",index=False)

    summary={
        "contract":"L2_LAW568_BACKWARD_HOLDOUT_V1_RESULT",
        "holdout_year":2021,
        "l2_training_years":[2020],
        "rules_tuned_on_2021":False,
        "law_metrics":{r["law_id"]:r for r in rows},
        "bootstrap":{r["law_id"]:r for r in boot},
        "month_summary":{r["law_id"]:r for r in month_summary},
        "portfolios":{r["portfolio_id"]:r for r in portfolios},
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_LAW568_BACKWARD_HOLDOUT_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
