#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import (
    YEARS, TEST_YEARS, parse_paths, build_year_frame, evaluate_chosen, write_csv
)
from run_l2_ticket_market_gap_v2 import (
    MODEL_TOP_K, MIN_UPGRADE, MAX_TICKETS, MIN_DEV_EXECUTION_COVERAGE_PCT,
    add_market_features, market_feature_columns, train_rank_predict, add_ranks,
    apply_gap_policy
)

FROZEN_RUN_ID=36654227618
BOOTSTRAP_N=5000
RANDOM_CONTROL_N=2000
SEED=1945

def args():
    p=argparse.ArgumentParser(description="Robustness audit for frozen L2 market-gap V2.")
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--frozen-summary",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def metrics(df,source_races,label):
    return evaluate_chosen(df,source_races,label)

def grouped_metrics(chosen, group_col, label_prefix):
    rows=[]
    for key,g in chosen.groupby(group_col,dropna=False,sort=True):
        m=metrics(g,g["race_id"].nunique(),f"{label_prefix}_{key}")
        m[group_col]=key
        rows.append(m)
    return rows

def add_bins(df):
    z=df.copy()
    z["month"]=z["race_date"].astype(str).str.slice(0,7)
    m=pd.to_numeric(z["race_date"].astype(str).str.slice(5,7),errors="coerce").fillna(0).astype(int)
    z["quarter"]="Q"+(((m-1)//3)+1).clip(1,4).astype(str)
    z["half"]=np.where(m<=6,"H1","H2")
    z["odds_band"]=pd.cut(
        z["odds"],[-np.inf,5,10,20,50,100,np.inf],
        labels=["<=5","5-10","10-20","20-50","50-100","100+"],
        right=False
    ).astype(str)
    z["upgrade_band"]=pd.cut(
        z["rank_upgrade"],[-np.inf,5,7,10,15,25,np.inf],
        labels=["<5","5-6","7-9","10-14","15-24","25+"],
        right=False
    ).astype(str)
    z["model_rank_band"]=pd.cut(
        z["model_rank"],[0,3,5,10,15,np.inf],
        labels=["1-3","4-5","6-10","11-15","16+"],
        right=True
    ).astype(str)
    z["market_rank_band"]=pd.cut(
        z["market_rank"],[0,3,5,10,20,40,np.inf],
        labels=["1-3","4-5","6-10","11-20","21-40","41+"],
        right=True
    ).astype(str)
    mx=np.maximum(z["a_consensus_rank"].astype(int),z["b_consensus_rank"].astype(int))
    sm=z["a_consensus_rank"].astype(int)+z["b_consensus_rank"].astype(int)
    z["l17_pair_max_rank_band"]=pd.cut(
        mx,[0,3,5,7,10,14,np.inf],
        labels=["<=3","4-5","6-7","8-10","11-14","15+"],
        right=True
    ).astype(str)
    z["l17_pair_rank_sum_band"]=pd.cut(
        sm,[0,5,8,12,18,26,np.inf],
        labels=["<=5","6-8","9-12","13-18","19-26","27+"],
        right=True
    ).astype(str)
    z["distance_band"]=pd.cut(
        pd.to_numeric(z["distance_m"],errors="coerce").fillna(0),
        [0,1400,1800,2200,2600,np.inf],
        labels=["<=1400","1401-1800","1801-2200","2201-2600","2601+"],
        right=True
    ).astype(str)
    z["field_size_band"]=pd.cut(
        pd.to_numeric(z["field_size"],errors="coerce").fillna(0),
        [0,10,12,14,16,np.inf],
        labels=["<=10","11-12","13-14","15-16","17+"],
        right=True
    ).astype(str)
    return z

def concentration(chosen):
    total_stake=100.0*len(chosen)
    total_return=float(chosen["return_yen_per100"].sum())
    wins=chosen[chosen["return_yen_per100"]>0].sort_values("return_yen_per100",ascending=False).copy()
    rows=[]
    for n in (0,1,2,3,5,10):
        removed=float(wins.head(n)["return_yen_per100"].sum()) if n else 0.0
        ret=total_return-removed
        rows.append({
            "remove_top_wins":n,
            "tickets":len(chosen),
            "stake_yen":total_stake,
            "return_yen":ret,
            "removed_return_yen":removed,
            "removed_share_pct":100.0*removed/total_return if total_return else 0.0,
            "roi_pct":100.0*ret/total_stake if total_stake else None,
            "profit_yen":ret-total_stake,
        })
    for cap in (2000,5000,10000,20000):
        ret=float(chosen["return_yen_per100"].clip(upper=cap).sum())
        rows.append({
            "remove_top_wins":f"cap_{cap}",
            "tickets":len(chosen),
            "stake_yen":total_stake,
            "return_yen":ret,
            "removed_return_yen":total_return-ret,
            "removed_share_pct":100.0*(total_return-ret)/total_return if total_return else 0.0,
            "roi_pct":100.0*ret/total_stake if total_stake else None,
            "profit_yen":ret-total_stake,
        })
    return rows

def bootstrap_rows(chosen,year):
    if chosen.empty:
        return []
    rng=np.random.default_rng(SEED+year)
    ret=chosen["return_yen_per100"].to_numpy(dtype=float)
    n=len(ret)
    rois=np.empty(BOOTSTRAP_N,dtype=float)
    for i in range(BOOTSTRAP_N):
        idx=rng.integers(0,n,size=n)
        rois[i]=ret[idx].sum()/n
    q=np.quantile(rois,[0.025,0.05,0.25,0.5,0.75,0.95,0.975])
    return [{
        "year":year,
        "bootstrap_samples":BOOTSTRAP_N,
        "tickets":n,
        "observed_roi_pct":100.0*ret.sum()/(100.0*n),
        "roi_p2_5":float(q[0]),
        "roi_p5":float(q[1]),
        "roi_p25":float(q[2]),
        "roi_median":float(q[3]),
        "roi_p75":float(q[4]),
        "roi_p95":float(q[5]),
        "roi_p97_5":float(q[6]),
        "prob_roi_gt_100_pct":100.0*float((rois>100.0).mean()),
    }]

def cumulative_monthly(chosen):
    rows=[]
    if chosen.empty:
        return rows
    g=chosen.groupby("month",sort=True).agg(
        tickets=("pair_horse_ids","size"),
        hits=("hit","sum"),
        return_yen=("return_yen_per100","sum")
    ).reset_index()
    cum_t=0;cum_r=0.0
    for _,r in g.iterrows():
        cum_t+=int(r["tickets"]); cum_r+=float(r["return_yen"])
        rows.append({
            "month":r["month"],
            "month_tickets":int(r["tickets"]),
            "month_hits":int(r["hits"]),
            "month_return_yen":float(r["return_yen"]),
            "cum_tickets":cum_t,
            "cum_stake_yen":100.0*cum_t,
            "cum_return_yen":cum_r,
            "cum_profit_yen":cum_r-100.0*cum_t,
            "cum_roi_pct":100.0*cum_r/(100.0*cum_t),
        })
    return rows

def evaluate_grid_year(df):
    source=df["race_id"].nunique()
    rows=[]
    for k in MODEL_TOP_K:
        for up in MIN_UPGRADE:
            for mx in MAX_TICKETS:
                if mx>k: continue
                c=apply_gap_policy(df,k,up,mx)
                m=metrics(c,source,f"MODEL_TOP{k}_UP{up}_MAX{mx}")
                m.update({"model_top_k":k,"min_market_rank_upgrade":up,"max_tickets_per_race":mx})
                rows.append(m)
    return rows

def policy_key(r):
    return (int(r["model_top_k"]),int(r["min_market_rank_upgrade"]),int(r["max_tickets_per_race"]))

def choose_champion(rows):
    eligible=[r for r in rows if float(r["execution_coverage_pct"])>=MIN_DEV_EXECUTION_COVERAGE_PCT]
    pool=eligible or rows
    return max(pool,key=lambda r:(
        r["roi_pct"] if r["roi_pct"] is not None else -1e18,
        r["profit_yen"],
        -r["max_drawdown_yen"],
        r["execution_coverage_pct"],
    ))

def policy_stability(by_year,frozen):
    frames={}
    for y,rows in by_year.items():
        d=pd.DataFrame(rows)
        d["key"]=d.apply(lambda r:f'{int(r["model_top_k"])}|{int(r["min_market_rank_upgrade"])}|{int(r["max_tickets_per_race"])}',axis=1)
        frames[y]=d.set_index("key")
    keys=sorted(set.intersection(*[set(x.index) for x in frames.values()]))
    wide=pd.DataFrame(index=keys)
    for y in sorted(frames):
        wide[f"roi_{y}"]=pd.to_numeric(frames[y].loc[keys,"roi_pct"],errors="coerce")
        wide[f"coverage_{y}"]=pd.to_numeric(frames[y].loc[keys,"execution_coverage_pct"],errors="coerce")
    corr=wide[[c for c in wide.columns if c.startswith("roi_")]].corr(method="spearman")
    corr_rows=[]
    for a in corr.index:
        for b in corr.columns:
            if a<b:
                corr_rows.append({"series_a":a,"series_b":b,"spearman":float(corr.loc[a,b])})
    fk=f'{frozen["model_top_k"]}|{frozen["min_market_rank_upgrade"]}|{frozen["max_tickets_per_race"]}'
    rank_rows=[]
    for y in sorted(frames):
        d=frames[y].copy()
        d["roi_rank_desc"]=d["roi_pct"].rank(method="min",ascending=False)
        if fk in d.index:
            r=d.loc[fk]
            rank_rows.append({
                "year":y,"frozen_policy_key":fk,
                "roi_pct":float(r["roi_pct"]),
                "execution_coverage_pct":float(r["execution_coverage_pct"]),
                "roi_rank_desc":int(r["roi_rank_desc"]),
                "policy_count":len(d),
            })
    return wide.reset_index(names="policy_key"),corr_rows,rank_rows

def transfer_rows(by_year,preds):
    out=[]
    for source_year,target_years in [(2023,[2024,2025]),(2024,[2025])]:
        champ=choose_champion(by_year[source_year])
        for ty in target_years:
            chosen=apply_gap_policy(
                preds[ty],
                int(champ["model_top_k"]),
                int(champ["min_market_rank_upgrade"]),
                int(champ["max_tickets_per_race"])
            )
            m=metrics(chosen,preds[ty]["race_id"].nunique(),f"TRAINRULE_{source_year}_TEST_{ty}")
            m.update({
                "rule_selected_on_year":source_year,
                "test_year":ty,
                "model_top_k":int(champ["model_top_k"]),
                "min_market_rank_upgrade":int(champ["min_market_rank_upgrade"]),
                "max_tickets_per_race":int(champ["max_tickets_per_race"]),
                "source_year_roi_pct":champ["roi_pct"],
                "source_year_coverage_pct":champ["execution_coverage_pct"],
            })
            out.append(m)
    return out

def same_race_controls(all_df,selected):
    ids=set(selected["race_id"].astype(str))
    z=all_df[all_df["race_id"].astype(str).isin(ids)].copy()
    source=len(ids)
    rows=[]
    for label,col in [
        ("FROZEN_SELECTED",None),
        ("MARKET_TOP1","market_rank"),
        ("MODEL_TOP1","model_rank"),
        ("L17_TOP1","l17_rank_score"),
    ]:
        c=selected if col is None else z[z[col]==1].copy()
        m=metrics(c,source,label)
        rows.append(m)
    return rows

def random_top15_control(all_df,selected):
    ids=sorted(set(selected["race_id"].astype(str)))
    z=all_df[(all_df["race_id"].astype(str).isin(ids))&(all_df["model_rank"]<=15)].copy()
    groups={rid:g["return_yen_per100"].to_numpy(dtype=float) for rid,g in z.groupby(z["race_id"].astype(str))}
    rng=np.random.default_rng(SEED+2025)
    rois=np.empty(RANDOM_CONTROL_N,dtype=float)
    for i in range(RANDOM_CONTROL_N):
        ret=0.0
        for rid in ids:
            arr=groups[rid]
            ret+=float(arr[rng.integers(0,len(arr))])
        rois[i]=ret/len(ids)
    observed=100.0*selected["return_yen_per100"].sum()/(100.0*len(selected))
    q=np.quantile(rois,[0.025,0.5,0.975])
    return [{
        "simulations":RANDOM_CONTROL_N,
        "races":len(ids),
        "observed_frozen_roi_pct":observed,
        "random_top15_roi_p2_5":float(q[0]),
        "random_top15_roi_median":float(q[1]),
        "random_top15_roi_p97_5":float(q[2]),
        "pct_random_ge_observed":100.0*float((rois>=observed).mean()),
    }]

def context_table(chosen,col):
    rows=[]
    for key,g in chosen.groupby(col,dropna=False,sort=True):
        if len(g)<10: continue
        m=metrics(g,g["race_id"].nunique(),f"{col}_{key}")
        m[col]=key
        rows.append(m)
    return rows

def main():
    a=args()
    lp=parse_paths(a.l17_year)
    if set(lp)!=set(YEARS):
        raise SystemExit("L1.7 year path mismatch")
    if 2026 in lp:
        raise SystemExit("2026 sealed")

    frozen=json.load(open(a.frozen_summary,encoding="utf-8"))
    if frozen.get("contract")!="L2_TICKET_MARKET_GAP_V2_RESULT":
        raise SystemExit("bad frozen V2 summary")
    p=frozen["selected_policy"]
    frozen_policy={
        "model_top_k":int(p["model_top_k"]),
        "min_market_rank_upgrade":int(p["min_market_rank_upgrade"]),
        "max_tickets_per_race":int(p["max_tickets_per_race"]),
    }
    if frozen_policy!={"model_top_k":15,"min_market_rank_upgrade":5,"max_tickets_per_race":1}:
        raise SystemExit(f"frozen policy drift: {frozen_policy}")

    l17={y:load_l17(lp[y],y) for y in YEARS}
    frames={y:add_market_features(build_year_frame(y,l17[y],a.backfill_root)) for y in YEARS}
    cols=market_feature_columns(frames[2022])
    preds={}
    for y in TEST_YEARS:
        train=pd.concat([frames[t] for t in YEARS if t<y],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)
        score,_=train_rank_predict(train,test,cols,93000+y)
        test["market_aware_score"]=score
        preds[y]=add_ranks(test)
        print(f"AUDIT_PREDICTION_READY year={y} pairs={len(test)} races={test['race_id'].nunique()}",flush=True)

    selected={}
    yearly=[]
    for y in TEST_YEARS:
        c=apply_gap_policy(
            preds[y],
            frozen_policy["model_top_k"],
            frozen_policy["min_market_rank_upgrade"],
            frozen_policy["max_tickets_per_race"],
        )
        c=add_bins(c)
        selected[y]=c
        m=metrics(c,preds[y]["race_id"].nunique(),"FROZEN_POLICY")
        m["year"]=y
        yearly.append(m)

    # Exact reproducibility guard for untouched 2025 holdout.
    y25=next(x for x in yearly if x["year"]==2025)
    old=frozen["holdout"]
    for k in ("tickets","hit_races","stake_yen","return_yen","profit_yen"):
        if abs(float(y25[k])-float(old[k]))>1e-9:
            raise SystemExit(f"2025 reproduction drift key={k} audit={y25[k]} frozen={old[k]}")
    if abs(float(y25["roi_pct"])-float(old["roi_pct"]))>1e-9:
        raise SystemExit("2025 ROI reproduction drift")

    hold=selected[2025]
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    monthly=grouped_metrics(hold,"month","MONTH")
    quarterly=grouped_metrics(hold,"quarter","QUARTER")
    half=grouped_metrics(hold,"half","HALF")
    write_csv(out/"frozen-policy-yearly.csv",yearly)
    write_csv(out/"holdout-monthly.csv",monthly)
    write_csv(out/"holdout-quarterly.csv",quarterly)
    write_csv(out/"holdout-half.csv",half)
    write_csv(out/"holdout-cumulative-monthly.csv",cumulative_monthly(hold))
    write_csv(out/"holdout-return-concentration.csv",concentration(hold))

    boot=[]
    for y in TEST_YEARS:
        boot.extend(bootstrap_rows(selected[y],y))
    write_csv(out/"bootstrap-roi.csv",boot)

    for col,name in [
        ("odds_band","holdout-by-odds-band.csv"),
        ("upgrade_band","holdout-by-upgrade-band.csv"),
        ("model_rank_band","holdout-by-model-rank-band.csv"),
        ("market_rank_band","holdout-by-market-rank-band.csv"),
        ("l17_pair_max_rank_band","holdout-by-l17-max-rank-band.csv"),
        ("l17_pair_rank_sum_band","holdout-by-l17-rank-sum-band.csv"),
        ("distance_band","holdout-by-distance-band.csv"),
        ("field_size_band","holdout-by-field-size-band.csv"),
        ("surface","holdout-by-surface.csv"),
        ("venue_code","holdout-by-venue.csv"),
        ("race_class_normalized","holdout-by-race-class.csv"),
    ]:
        write_csv(out/name,context_table(hold,col))

    wins=hold[hold["return_yen_per100"]>0].sort_values("return_yen_per100",ascending=False).head(30)
    keep=[
        "race_id","race_date","pair_numbers","return_yen_per100","odds",
        "market_rank","model_rank","rank_upgrade",
        "a_consensus_rank","b_consensus_rank","surface","venue_code",
        "distance_m","field_size","race_class_normalized"
    ]
    write_csv(out/"holdout-top-winning-tickets.csv",wins[keep])

    write_csv(out/"same-race-controls.csv",same_race_controls(preds[2025],hold))
    write_csv(out/"random-top15-control.csv",random_top15_control(preds[2025],hold))

    by_year={y:evaluate_grid_year(preds[y]) for y in TEST_YEARS}
    all_policy_rows=[]
    for y,rows in by_year.items():
        for r in rows:
            q=dict(r); q["year"]=y; all_policy_rows.append(q)
    write_csv(out/"policy-grid-by-year.csv",all_policy_rows)

    wide,corr,ranks=policy_stability(by_year,frozen_policy)
    write_csv(out/"policy-roi-wide.csv",wide)
    write_csv(out/"policy-roi-spearman.csv",corr)
    write_csv(out/"frozen-policy-rank-by-year.csv",ranks)
    write_csv(out/"single-year-champion-transfer.csv",transfer_rows(by_year,preds))

    neigh=[]
    for r in all_policy_rows:
        if int(r["model_top_k"]) in (10,15) and int(r["min_market_rank_upgrade"]) in (3,5,10) and int(r["max_tickets_per_race"]) in (1,2):
            neigh.append(r)
    write_csv(out/"neighbor-policy-stability.csv",neigh)

    pos_months=sum(1 for r in monthly if float(r["profit_yen"])>0)
    top1_ret=float(hold[hold["return_yen_per100"]>0]["return_yen_per100"].max()) if (hold["return_yen_per100"]>0).any() else 0.0
    summary={
        "contract":"L2_MARKET_GAP_V2_ROBUST_AUDIT_RESULT",
        "source_run_id":FROZEN_RUN_ID,
        "frozen_policy":frozen_policy,
        "policy_changed_after_holdout":False,
        "2025_reproduced_exactly":True,
        "yearly":yearly,
        "holdout_positive_months":pos_months,
        "holdout_months":len(monthly),
        "holdout_largest_single_return_yen_per100":top1_ret,
        "tests":{
            "bootstrap_samples":BOOTSTRAP_N,
            "random_top15_simulations":RANDOM_CONTROL_N,
            "jackpot_dependence":True,
            "time_slices":True,
            "market_gap_bands":True,
            "odds_bands":True,
            "l17_rank_bands":True,
            "race_context":True,
            "policy_grid_year_by_year":True,
            "single_year_champion_transfer":True,
            "same_race_controls":True,
        },
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Market Gap V2 — Robustness Audit\n\n"
        "The frozen V2 policy is not changed after looking at the 2025 holdout. "
        "This audit reproduces the same walk-forward scores and then stress-tests the frozen "
        "Top15 / market-rank-upgrade>=5 / max1 policy for time stability, payout concentration, "
        "bootstrap uncertainty, odds and rank bands, race context, neighboring policy stability, "
        "same-race controls, random Top15 controls, and single-year policy transfer. "
        "All outputs are compact diagnostics; no large prediction matrix is persisted. 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("L2_MARKET_GAP_V2_ROBUST_AUDIT_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
