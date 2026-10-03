#!/usr/bin/env python3
import json,time
from pathlib import Path
import numpy as np
import pandas as pd

SRC=Path("research-results/l17-outsider-market-divergence-v1/run-36821694998/joined-scored.csv.gz")
OUT=Path("research-results/l2-simple-value-axis-v1")

def score_bucket(x):
    x=float(x)
    if x<=0: return "0"
    if x<=3: return "1-3"
    if x<=6: return "4-6"
    if x<=9: return "7-9"
    return "10+"

def gap_bucket(x):
    x=float(x)
    if x<4: return "2-3"
    if x<6: return "4-5"
    return "6+"

def summarize(z,label):
    if z.empty:
        return {"scope":label,"horses":0}
    return {
        "scope":label,
        "horses":int(len(z)),
        "races":int(z["race_id"].nunique()),
        "actual_top3_pct":100*float(z["target_top3"].mean()),
        "market_peer_top3_pct":100*float(z["market_peer_top3_rate"].mean()),
        "lift_vs_market_peer_pp":100*float((z["target_top3"]-z["market_peer_top3_rate"]).mean()),
        "mean_l175_rank":float(z["king_rank"].mean()),
        "mean_market_rank":float(z["market_rank"].mean()),
        "mean_rank_gap":float(z["rank_gap"].mean()),
        "mean_p3_pct":100*float(z["p3"].mean()),
        "mean_p3_delta_pp":100*float(z["p3_delta"].mean()),
        "mean_outsider_score":float(z["score"].mean()),
    }

def choose_axis(g,mode):
    if mode=="P3_FIRST":
        return g.sort_values(
            ["p3","rank_gap","score","king_rank","market_rank","horse_id"],
            ascending=[False,False,False,True,False,True]
        ).iloc[0]
    if mode=="GAP_FIRST":
        return g.sort_values(
            ["rank_gap","p3","score","king_rank","market_rank","horse_id"],
            ascending=[False,False,False,True,False,True]
        ).iloc[0]
    raise ValueError(mode)

def main():
    t0=time.time()
    if not SRC.exists(): raise SystemExit(f"missing archived source: {SRC}")
    df=pd.read_csv(SRC,compression="gzip",dtype={"race_id":str,"horse_id":str})
    df=df[df["year"].isin([2023,2024,2025])].copy()
    if 2026 in set(df["year"]): raise SystemExit("2026 sealed")
    # Value discovery only: L1.75/Seven-King ranks horse at least 2 places above the WIN market.
    c=df[(df["direction"]=="L1_UPGRADE") & (df["rank_gap"]>=2)].copy()
    if c.empty: raise SystemExit("no value candidates")
    # One axis per race. No new horse model and no global reranking.
    rows=[]
    for mode in ["GAP_FIRST","P3_FIRST"]:
        picks=[]
        for (year,rid),g in c.groupby(["year","race_id"],sort=False):
            r=choose_axis(g,mode).copy()
            r["selection_mode"]=mode
            picks.append(r)
        z=pd.DataFrame(picks).reset_index(drop=True)
        z["outsider_score_bucket"]=z["score"].map(score_bucket)
        z["rank_gap_bucket"]=z["rank_gap"].map(gap_bucket)
        z["outsider_agrees"]=z["p3_delta"]>0
        rows.append(z)
    axes=pd.concat(rows,ignore_index=True)

    OUT.mkdir(parents=True,exist_ok=True)
    axes.to_csv(OUT/"axis-picks.csv.gz",index=False,compression="gzip")

    summary_rows=[]
    for mode,z in axes.groupby("selection_mode"):
        summary_rows.append({"selection_mode":mode,**summarize(z,"POOLED")})
        for y in [2023,2024,2025]:
            summary_rows.append({"selection_mode":mode,**summarize(z[z["year"]==y],str(y))})
    pd.DataFrame(summary_rows).to_csv(OUT/"headline.csv",index=False)

    primary=axes[axes["selection_mode"]=="P3_FIRST"].copy()
    bucket_rows=[]
    for col in ["outsider_score_bucket","rank_gap_bucket","outsider_agrees"]:
        for val,g in primary.groupby(col,dropna=False):
            bucket_rows.append({"dimension":col,"value":str(val),**summarize(g,"POOLED")})
            for y in [2023,2024,2025]:
                yy=g[g["year"]==y]
                if len(yy):
                    bucket_rows.append({"dimension":col,"value":str(val),**summarize(yy,str(y))})
    pd.DataFrame(bucket_rows).to_csv(OUT/"axis-tags.csv",index=False)

    # P3 quintiles are descriptive only; not a BUY/SKIP rule.
    q=primary.copy()
    q["p3_quintile"]=pd.qcut(q["p3"].rank(method="first"),5,labels=["Q1_LOW","Q2","Q3","Q4","Q5_HIGH"])
    qrows=[]
    for val,g in q.groupby("p3_quintile",observed=True):
        qrows.append({"p3_quintile":str(val),**summarize(g,"POOLED")})
        for y in [2023,2024,2025]:
            yy=g[g["year"]==y]
            if len(yy):
                qrows.append({"p3_quintile":str(val),**summarize(yy,str(y))})
    pd.DataFrame(qrows).to_csv(OUT/"p3-quintiles.csv",index=False)

    p3_head=pd.DataFrame(summary_rows)
    p3_pool=p3_head[(p3_head["selection_mode"]=="P3_FIRST")&(p3_head["scope"]=="POOLED")].iloc[0].to_dict()
    gap_pool=p3_head[(p3_head["selection_mode"]=="GAP_FIRST")&(p3_head["scope"]=="POOLED")].iloc[0].to_dict()

    summary={
        "contract":"L2_SIMPLE_VALUE_AXIS_V1_RESULT",
        "architecture":"L175_VS_WIN_MARKET_VALUE_DISCOVERY_THEN_EXISTING_P3_AXIS_SELECTION",
        "rules":{
            "value_candidate":"rank_gap = market_rank - L1.75/Seven-King rank >= 2",
            "primary_axis":"among value candidates in each race, choose highest existing walk-forward p3; ties use larger rank_gap then larger Outsider score",
            "outsider_usage":"existing Outsider score and p3_delta are confirmation/context tags only; no horse reranking model",
            "buy_skip":"not defined yet",
            "ticket_construction":"not defined yet",
        },
        "source":"archived OOS horse-level L1.75-compatible divergence rows + safe Outsider podium calibration",
        "years":[2023,2024,2025],
        "primary_pooled":p3_pool,
        "gap_only_baseline_pooled":gap_pool,
        "no_new_model":True,
        "no_kaggle_downloads":True,
        "payout_used":False,
        "trio_odds_used":False,
        "roi_used":False,
        "2026_locked":True,
        "elapsed_seconds":time.time()-t0,
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/"README.md").write_text(
        "# L2 Simple Value Axis V1\n\n"
        "No new horse model. Find horses rated at least two places higher by L1.75/Seven-King than the WIN market, "
        "then select one axis per race using the already-frozen walk-forward podium probability p3. Outsider support "
        "is retained as a context tag only. This stage tests axis quality; it does not yet construct tickets or optimize ROI.\n",
        encoding="utf-8"
    )
    print("===== HEADLINE =====")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("===== PRIMARY TAGS =====")
    print(pd.DataFrame(bucket_rows).to_string(index=False))
    print("===== SUMMARY =====")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print("L2_SIMPLE_VALUE_AXIS_V1_READY")

if __name__=="__main__":
    main()
