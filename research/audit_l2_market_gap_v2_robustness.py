#!/usr/bin/env python3
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_N=10000
BLOCK_BOOTSTRAP_N=10000
SEED=1945
FROZEN_POLICY={"model_top_k":15,"min_market_rank_upgrade":5,"max_tickets_per_race":1}

def args():
    p=argparse.ArgumentParser(description="Robustness audit from persisted frozen V2 outputs.")
    p.add_argument("--source-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def write_csv(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(obj,pd.DataFrame):
        obj.to_csv(path,index=False)
    else:
        pd.DataFrame(obj).to_csv(path,index=False)

def max_drawdown(df):
    if df.empty: return 0.0
    z=df.sort_values(["race_date","race_id"]).copy()
    pnl=z["return_yen_per100"].astype(float)-100.0
    cum=pnl.cumsum().to_numpy()
    peak=np.maximum.accumulate(np.r_[0.0,cum])[1:]
    return float((peak-cum).max()) if len(cum) else 0.0

def metrics(df,source_races,label):
    n=len(df); ret=float(df["return_yen_per100"].sum()) if n else 0.0
    hits=int((df["return_yen_per100"]>0).sum()) if n else 0
    stake=100.0*n
    return {
        "label":label,
        "source_races":int(source_races),
        "executed_races":int(df["race_id"].nunique()) if n else 0,
        "execution_coverage_pct":100.0*df["race_id"].nunique()/source_races if source_races else 0.0,
        "tickets":n,
        "hit_races":hits,
        "race_hit_rate_pct":100.0*hits/source_races if source_races else 0.0,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(df),
    }

def add_time_bins(df):
    z=df.copy()
    z["race_date"]=z["race_date"].astype(str)
    z["month"]=z["race_date"].str.slice(0,7)
    mm=pd.to_numeric(z["race_date"].str.slice(5,7),errors="coerce").fillna(0).astype(int)
    z["quarter"]="Q"+(((mm-1)//3)+1).clip(1,4).astype(str)
    z["half"]=np.where(mm<=6,"H1","H2")
    z["odds_band"]=pd.cut(
        z["odds"],[-np.inf,5,10,20,50,100,np.inf],
        labels=["<5","5-10","10-20","20-50","50-100","100+"],right=False
    ).astype(str)
    z["upgrade_band"]=pd.cut(
        z["rank_upgrade"],[-np.inf,5,7,10,15,25,np.inf],
        labels=["<5","5-6","7-9","10-14","15-24","25+"],right=False
    ).astype(str)
    z["model_rank_band"]=pd.cut(
        z["model_rank"],[0,3,5,10,15,np.inf],
        labels=["1-3","4-5","6-10","11-15","16+"],right=True
    ).astype(str)
    z["market_rank_band"]=pd.cut(
        z["market_rank"],[0,3,5,10,20,40,np.inf],
        labels=["1-3","4-5","6-10","11-20","21-40","41+"],right=True
    ).astype(str)
    a=pd.to_numeric(z["a_consensus_rank"],errors="coerce").fillna(99).astype(int)
    b=pd.to_numeric(z["b_consensus_rank"],errors="coerce").fillna(99).astype(int)
    z["l17_pair_max_rank"]=np.maximum(a,b)
    z["l17_pair_rank_sum"]=a+b
    z["l17_pair_max_rank_band"]=pd.cut(
        z["l17_pair_max_rank"],[0,3,5,7,10,14,np.inf],
        labels=["<=3","4-5","6-7","8-10","11-14","15+"],right=True
    ).astype(str)
    z["l17_pair_rank_sum_band"]=pd.cut(
        z["l17_pair_rank_sum"],[0,5,8,12,18,26,np.inf],
        labels=["<=5","6-8","9-12","13-18","19-26","27+"],right=True
    ).astype(str)
    return z

def group_metrics(df,col,min_n=1):
    rows=[]
    for key,g in df.groupby(col,dropna=False,sort=True):
        if len(g)<min_n: continue
        m=metrics(g,g["race_id"].nunique(),str(key))
        m[col]=key
        rows.append(m)
    return rows

def cumulative_monthly(df):
    rows=[]; ct=0; cr=0.0
    for month,g in df.groupby("month",sort=True):
        t=len(g); h=int((g["return_yen_per100"]>0).sum()); r=float(g["return_yen_per100"].sum())
        ct+=t;cr+=r
        rows.append({
            "month":month,"month_tickets":t,"month_hits":h,
            "month_return_yen":r,"month_profit_yen":r-100*t,
            "month_roi_pct":100*r/(100*t) if t else None,
            "cum_tickets":ct,"cum_stake_yen":100*ct,
            "cum_return_yen":cr,"cum_profit_yen":cr-100*ct,
            "cum_roi_pct":100*cr/(100*ct) if ct else None,
        })
    return rows

def concentration(df):
    total_ret=float(df["return_yen_per100"].sum())
    total_stake=100.0*len(df)
    wins=df[df["return_yen_per100"]>0].sort_values("return_yen_per100",ascending=False)
    rows=[]
    for n in (0,1,2,3,5,10):
        removed=float(wins.head(n)["return_yen_per100"].sum()) if n else 0.0
        ret=total_ret-removed
        rows.append({
            "test":f"remove_top_{n}_wins",
            "removed_return_yen":removed,
            "removed_share_pct":100*removed/total_ret if total_ret else 0.0,
            "return_yen":ret,"stake_yen":total_stake,
            "profit_yen":ret-total_stake,
            "roi_pct":100*ret/total_stake if total_stake else None,
        })
    for cap in (2000,5000,10000,20000):
        ret=float(df["return_yen_per100"].clip(upper=cap).sum())
        rows.append({
            "test":f"cap_each_win_at_{cap}",
            "removed_return_yen":total_ret-ret,
            "removed_share_pct":100*(total_ret-ret)/total_ret if total_ret else 0.0,
            "return_yen":ret,"stake_yen":total_stake,
            "profit_yen":ret-total_stake,
            "roi_pct":100*ret/total_stake if total_stake else None,
        })
    return rows

def bootstrap(df):
    rng=np.random.default_rng(SEED)
    r=df["return_yen_per100"].to_numpy(dtype=float)
    n=len(r)
    vals=np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        idx=rng.integers(0,n,size=n)
        vals[i]=r[idx].sum()/n
    q=np.quantile(vals,[.025,.05,.25,.5,.75,.95,.975])
    return [{
        "method":"ticket_bootstrap",
        "samples":BOOTSTRAP_N,"tickets":n,
        "observed_roi_pct":100*r.sum()/(100*n),
        "roi_p2_5":q[0],"roi_p5":q[1],"roi_p25":q[2],"roi_median":q[3],
        "roi_p75":q[4],"roi_p95":q[5],"roi_p97_5":q[6],
        "prob_roi_gt_100_pct":100*float((vals>100).mean()),
    }]

def month_block_bootstrap(df):
    rng=np.random.default_rng(SEED+1)
    months=[g["return_yen_per100"].to_numpy(dtype=float) for _,g in df.groupby("month",sort=True)]
    m=len(months)
    vals=np.empty(BLOCK_BOOTSTRAP_N)
    for i in range(BLOCK_BOOTSTRAP_N):
        blocks=[months[j] for j in rng.integers(0,m,size=m)]
        x=np.concatenate(blocks)
        vals[i]=x.sum()/len(x)
    q=np.quantile(vals,[.025,.05,.25,.5,.75,.95,.975])
    return [{
        "method":"month_block_bootstrap",
        "samples":BLOCK_BOOTSTRAP_N,"months":m,
        "observed_roi_pct":100*df["return_yen_per100"].sum()/(100*len(df)),
        "roi_p2_5":q[0],"roi_p5":q[1],"roi_p25":q[2],"roi_median":q[3],
        "roi_p75":q[4],"roi_p95":q[5],"roi_p97_5":q[6],
        "prob_roi_gt_100_pct":100*float((vals>100).mean()),
    }]

def leave_one_month_out(df):
    rows=[]
    for month in sorted(df["month"].unique()):
        z=df[df["month"]!=month]
        m=metrics(z,z["race_id"].nunique(),f"exclude_{month}")
        m["excluded_month"]=month
        rows.append(m)
    return rows

def load_race_context(root,race_ids):
    rows=[]
    wanted=set(map(str,race_ids))
    for p in sorted((Path(root)/"data"/"daily").glob("2025-*.jsonl.gz")):
        with gzip.open(p,"rt",encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                pack=json.loads(line)
                race=pack.get("race") or {}
                rid=str(race.get("race_id") or "")
                if rid not in wanted: continue
                entries=pack.get("entries") or []
                rows.append({
                    "race_id":rid,
                    "venue_code":str(race.get("venue_code") or "UNKNOWN"),
                    "surface":str(race.get("surface") or "UNKNOWN"),
                    "discipline":str(race.get("discipline") or "UNKNOWN"),
                    "direction":str(race.get("direction") or "UNKNOWN"),
                    "track_condition":str(race.get("track_condition") or "UNKNOWN"),
                    "race_class_normalized":str(race.get("race_class_normalized") or "UNKNOWN"),
                    "grade":str(race.get("grade") or "UNKNOWN"),
                    "distance_m":float(race.get("distance_m") or 0),
                    "field_size":len(entries),
                })
    out=pd.DataFrame(rows).drop_duplicates("race_id")
    missing=wanted-set(out["race_id"].astype(str)) if not out.empty else wanted
    if missing:
        raise SystemExit(f"missing race context count={len(missing)} sample={sorted(missing)[:10]}")
    return out

def context_bins(df):
    z=df.copy()
    z["distance_band"]=pd.cut(
        z["distance_m"],[0,1400,1800,2200,2600,np.inf],
        labels=["<=1400","1401-1800","1801-2200","2201-2600","2601+"],right=True
    ).astype(str)
    z["field_size_band"]=pd.cut(
        z["field_size"],[0,10,12,14,16,np.inf],
        labels=["<=10","11-12","13-14","15-16","17+"],right=True
    ).astype(str)
    return z

def policy_key(df):
    return (
        df["model_top_k"].astype(int).astype(str)+"|"+
        df["min_market_rank_upgrade"].astype(int).astype(str)+"|"+
        df["max_tickets_per_race"].astype(int).astype(str)
    )

def policy_stability(dev,hold):
    d=dev.copy(); h=hold.copy()
    d["policy_key"]=policy_key(d);h["policy_key"]=policy_key(h)
    cols=["policy_key","roi_pct","profit_yen","execution_coverage_pct","tickets"]
    m=d[cols].merge(h[cols],on="policy_key",suffixes=("_dev","_hold"))
    for c in [x for x in m.columns if x!="policy_key"]:
        m[c]=pd.to_numeric(m[c],errors="coerce")
    spearman=float(m["roi_pct_dev"].corr(m["roi_pct_hold"],method="spearman"))
    pearson=float(m["roi_pct_dev"].corr(m["roi_pct_hold"],method="pearson"))
    m["roi_rank_dev"]=m["roi_pct_dev"].rank(method="min",ascending=False)
    m["roi_rank_hold"]=m["roi_pct_hold"].rank(method="min",ascending=False)
    fk="15|5|1"
    fr=m[m["policy_key"]==fk].iloc[0].to_dict()
    summary=[{
        "policy_count":len(m),
        "roi_spearman_dev_vs_hold":spearman,
        "roi_pearson_dev_vs_hold":pearson,
        "holdout_policies_roi_gt_100":int((m["roi_pct_hold"]>100).sum()),
        "dev_policies_roi_gt_100":int((m["roi_pct_dev"]>100).sum()),
        "frozen_policy_key":fk,
        "frozen_dev_roi_pct":float(fr["roi_pct_dev"]),
        "frozen_hold_roi_pct":float(fr["roi_pct_hold"]),
        "frozen_dev_roi_rank":int(fr["roi_rank_dev"]),
        "frozen_hold_roi_rank":int(fr["roi_rank_hold"]),
    }]
    return m,summary

def neighbor_policies(dev,hold):
    d=dev.copy();h=hold.copy()
    d["policy_key"]=policy_key(d);h["policy_key"]=policy_key(h)
    m=d.merge(h,on="policy_key",suffixes=("_dev","_hold"))
    mask=(
        m["model_top_k_dev"].astype(int).isin([10,15]) &
        m["min_market_rank_upgrade_dev"].astype(int).isin([3,5,10]) &
        m["max_tickets_per_race_dev"].astype(int).isin([1,2])
    )
    keep=[
        "policy_key","model_top_k_dev","min_market_rank_upgrade_dev","max_tickets_per_race_dev",
        "execution_coverage_pct_dev","roi_pct_dev","profit_yen_dev",
        "execution_coverage_pct_hold","roi_pct_hold","profit_yen_hold"
    ]
    return m.loc[mask,keep].sort_values("policy_key")

def global_controls(source):
    pure=pd.read_csv(source/"pure-ranking-policies.csv")
    return pure[(pure["year"]==2025)&(pure["top_n"].isin([1,2,3,5,10]))].copy()

def main():
    a=args()
    src=Path(a.source_dir);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    summary=json.load(open(src/"summary.json",encoding="utf-8"))
    if summary.get("contract")!="L2_TICKET_MARKET_GAP_V2_RESULT":
        raise SystemExit("bad source contract")
    p=summary["selected_policy"]
    got={k:int(p[k]) for k in FROZEN_POLICY}
    if got!=FROZEN_POLICY:
        raise SystemExit(f"frozen policy drift {got}")

    tickets=pd.read_csv(src/"holdout-selected-tickets.csv.gz",compression="gzip")
    tickets=add_time_bins(tickets)
    old=summary["holdout"]
    m=metrics(tickets,int(old["source_races"]),"FROZEN_V2")
    for k in ("tickets","hit_races","stake_yen","return_yen","profit_yen"):
        if abs(float(m[k])-float(old[k]))>1e-9:
            raise SystemExit(f"reproduction drift {k}: {m[k]} vs {old[k]}")
    if abs(float(m["roi_pct"])-float(old["roi_pct"]))>1e-9:
        raise SystemExit("ROI reproduction drift")

    ctx=load_race_context(a.backfill_root,tickets["race_id"])
    tickets=context_bins(tickets.merge(ctx,on="race_id",how="left",validate="one_to_one"))

    write_csv(out/"holdout-monthly.csv",group_metrics(tickets,"month"))
    write_csv(out/"holdout-quarterly.csv",group_metrics(tickets,"quarter"))
    write_csv(out/"holdout-half.csv",group_metrics(tickets,"half"))
    write_csv(out/"holdout-cumulative-monthly.csv",cumulative_monthly(tickets))
    write_csv(out/"holdout-return-concentration.csv",concentration(tickets))
    write_csv(out/"bootstrap-roi.csv",bootstrap(tickets)+month_block_bootstrap(tickets))
    write_csv(out/"leave-one-month-out.csv",leave_one_month_out(tickets))

    for col,name,min_n in [
        ("odds_band","holdout-by-odds-band.csv",10),
        ("upgrade_band","holdout-by-upgrade-band.csv",10),
        ("model_rank_band","holdout-by-model-rank-band.csv",10),
        ("market_rank_band","holdout-by-market-rank-band.csv",10),
        ("l17_pair_max_rank_band","holdout-by-l17-max-rank-band.csv",10),
        ("l17_pair_rank_sum_band","holdout-by-l17-rank-sum-band.csv",10),
        ("surface","holdout-by-surface.csv",10),
        ("venue_code","holdout-by-venue.csv",10),
        ("distance_band","holdout-by-distance-band.csv",10),
        ("field_size_band","holdout-by-field-size-band.csv",10),
        ("race_class_normalized","holdout-by-race-class.csv",10),
    ]:
        write_csv(out/name,group_metrics(tickets,col,min_n))

    wins=tickets[tickets["return_yen_per100"]>0].sort_values("return_yen_per100",ascending=False)
    keep=[
        "race_id","race_date","pair_numbers","return_yen_per100","odds","market_rank","model_rank",
        "rank_upgrade","a_consensus_rank","b_consensus_rank","surface","venue_code",
        "distance_m","field_size","race_class_normalized"
    ]
    write_csv(out/"holdout-winning-tickets.csv",wins[keep])

    dev=pd.read_csv(src/"dev-market-gap-grid.csv")
    hold=pd.read_csv(src/"holdout-market-gap-matrix-diagnostic.csv")
    surface,surf_summary=policy_stability(dev,hold)
    write_csv(out/"policy-dev-vs-holdout.csv",surface)
    write_csv(out/"policy-stability-summary.csv",surf_summary)
    write_csv(out/"neighbor-policy-stability.csv",neighbor_policies(dev,hold))
    write_csv(out/"global-pure-ranking-controls-2025.csv",global_controls(src))

    # Development aggregate and holdout are intentionally reported without pretending
    # the combined 2023-24 result is a per-year transfer test.
    audit_summary={
        "contract":"L2_MARKET_GAP_V2_PERSISTED_ROBUST_AUDIT_RESULT",
        "source_run_id":36654227618,
        "source_contract":summary["contract"],
        "frozen_policy":FROZEN_POLICY,
        "policy_changed_after_holdout":False,
        "upstream_recomputed":False,
        "kaggle_dependency":False,
        "2025_reproduced_exactly":True,
        "development_2023_2024_combined":summary["selected_policy"],
        "holdout_2025":m,
        "positive_months":sum(float(x["profit_yen"])>0 for x in group_metrics(tickets,"month")),
        "months":tickets["month"].nunique(),
        "winning_tickets":int((tickets["return_yen_per100"]>0).sum()),
        "largest_single_return_yen_per100":float(wins["return_yen_per100"].max()) if len(wins) else 0.0,
        "policy_stability":surf_summary[0],
        "tests":[
            "monthly/quarterly/half/cumulative stability",
            "top-win removal and payout caps",
            "ticket bootstrap and month-block bootstrap",
            "leave-one-month-out",
            "odds/market-gap/model-rank/market-rank/L1.7-rank bands",
            "surface/venue/distance/field-size/race-class splits",
            "development-vs-holdout full policy-surface correlation",
            "neighbor-policy stability",
            "2025 global pure-ranking controls"
        ],
        "limitations":[
            "same-race alternative-ticket controls require full pair matrix, which was intentionally not persisted",
            "2023-only and 2024-only frozen-policy transfer cannot be reconstructed from compact persisted V2 outputs"
        ],
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(audit_summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Market Gap V2 — Persisted Robustness Audit\n\n"
        "This audit treats run 36654227618 as frozen evidence. It does not rebuild L1.7, Seven-King scores, "
        "or the market-gap model, so upstream Kaggle availability cannot alter the audit sample. The frozen "
        "2025 ticket list and the persisted development/holdout policy grids are stress-tested for payout "
        "concentration, temporal stability, bootstrap uncertainty, rank/odds/context splits, and policy-surface "
        "stability. 2025 is diagnostic only and 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("L2_MARKET_GAP_V2_PERSISTED_ROBUST_AUDIT_READY")
    print(json.dumps(audit_summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
