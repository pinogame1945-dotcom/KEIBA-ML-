#!/usr/bin/env python3
import argparse,json,time,math
from pathlib import Path
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l2_trio_probability_v2 import load_horse_dataset,attach_win_market,attach_outsider,prepare_l175

SEED=20261004
FOLDS=((2023,(2022,)),(2024,(2022,2023)),(2025,(2022,2023,2024)))
METHODS=("KING_ORDER","MARKET","OUTSIDER","L175_P3","ML_COMBINED")
KS=(1,2,3)

ML_FEATURES=(
    "king_rank_pct","king_mean_rank_pct","king_rank_std_pct","king_best_rank_pct","king_worst_rank_pct",
    "king_top1_vote_share","king_top3_support_share","king_top6_support_share",
    "king_probability_mean","king_probability_std",
    "market_rank_pct","log_market_win_odds","signed_rank_gap_pct","abs_rank_gap_pct",
    "outsider_available","outsider_score_scaled","p3_calibrated","p3_delta","outsider_alignment_score",
)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--outsider-predictions",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def prep(df):
    z=df.copy()
    z["target_top3"]=pd.to_numeric(z["target_top3"],errors="coerce")
    z["consensus_rank"]=pd.to_numeric(z["consensus_rank"],errors="coerce")
    z["field_size"]=pd.to_numeric(z["field_size"],errors="coerce")
    z["market_rank"]=pd.to_numeric(z["market_rank"],errors="coerce")
    for c in ML_FEATURES:
        z[c]=pd.to_numeric(z[c],errors="coerce").fillna(0.0)
    return z

def race_frames(df,year):
    y=df[df["year"]==year].copy()
    clean=[]; excluded=defaultdict(int)
    for rid,g in y.groupby("race_id",sort=False):
        g=g.sort_values(["consensus_rank","horse_number","horse_id"]).copy()
        if g["target_top3"].isna().any():
            excluded["missing_target"]+=1; continue
        if int(g["target_top3"].sum())!=3:
            excluded["nonunique_top3"]+=1; continue
        fs=int(g["field_size"].iloc[0])
        if fs<13:
            excluded["field_lt13"]+=1; continue
        if int(g["consensus_rank"].min())!=1:
            excluded["rank_not_start1"]+=1; continue
        clean.append(g)
    return clean,dict(excluded)

def fit_tail_model(df,train_years):
    q=df[(df["year"].isin(train_years))&(df["field_size"]>=13)&(df["consensus_rank"]>=13)].copy()
    q=q[q["target_top3"].notna()]
    # Exclude races without exactly three unique podium horses.
    sums=q[["race_id","year"]].drop_duplicates()
    valid=set()
    for (y,rid),g in df[df["year"].isin(train_years)].groupby(["year","race_id"],sort=False):
        if int(g["field_size"].iloc[0])>=13 and not g["target_top3"].isna().any() and int(g["target_top3"].sum())==3:
            valid.add((int(y),str(rid)))
    q=q[q.apply(lambda r:(int(r["year"]),str(r["race_id"])) in valid,axis=1)].copy()
    if q.empty: raise SystemExit("no tail training rows")
    y=q["target_top3"].astype(int)
    pos=max(1,int(y.sum())); neg=max(1,len(y)-pos)
    spw=float(min(50.0,max(1.0,math.sqrt(neg/pos))))
    m=lgb.LGBMClassifier(
        objective="binary",n_estimators=220,learning_rate=0.04,num_leaves=15,
        min_child_samples=50,subsample=0.9,colsample_bytree=0.9,reg_lambda=2.0,
        scale_pos_weight=spw,random_state=SEED,n_jobs=max(1,__import__("os").cpu_count() or 1),verbosity=-1
    )
    m.fit(q[list(ML_FEATURES)],y)
    return m,len(q),pos,spw

def score_tail(g,method,model=None):
    t=g[g["consensus_rank"]>=13].copy()
    if method=="KING_ORDER":
        t["_score"]=-t["consensus_rank"].astype(float)
    elif method=="MARKET":
        t["_score"]=-t["market_rank"].astype(float)
    elif method=="OUTSIDER":
        t["_score"]=t["outsider_score_scaled"].astype(float)
    elif method=="L175_P3":
        t["_score"]=t["p3_calibrated"].astype(float)
    elif method=="ML_COMBINED":
        if model is None: raise ValueError("model required")
        t["_score"]=model.predict_proba(t[list(ML_FEATURES)])[:,1]
    else:
        raise ValueError(method)
    return t.sort_values(
        ["_score","p3_calibrated","outsider_score_scaled","market_rank","consensus_rank","horse_number"],
        ascending=[False,False,False,True,True,True]
    )

def captured(g,selected_nos):
    winners=set(g.loc[g["target_top3"]==1,"horse_number"].astype(int))
    return winners.issubset(set(int(x) for x in selected_nos))

def analyze_year(df,year,model):
    races,excluded=race_frames(df,year)
    rank_rows=[]
    method_acc={(m,k):defaultdict(int) for m in METHODS for k in KS}
    summary=defaultdict(int)
    summary["challenge_races"]=len(races)

    for g in races:
        winner=g[g["target_top3"]==1].copy()
        top12=set(g.loc[g["consensus_rank"]<=12,"horse_number"].astype(int))
        tail=g[g["consensus_rank"]>=13].copy()
        tail_winner=winner[winner["consensus_rank"]>=13].copy()
        base=captured(g,top12)
        summary["base_capture"]+=int(base)
        tw=len(tail_winner)
        summary[f"tail_intruders_{tw}"]+=1
        if tw>0: summary["tail_miss_races"]+=1

        for _,r in tail.iterrows():
            rank=int(r["consensus_rank"])
            rank_rows.append({
                "year":year,"race_id":str(r["race_id"]),"rank":rank,
                "is_top3":int(r["target_top3"]==1),
                "market_rank":int(r["market_rank"]),
                "outsider_score":float(r["outsider_score_scaled"]),
                "p3":float(r["p3_calibrated"]),
            })

        for method in METHODS:
            ordered=score_tail(g,method,model)
            tail_true=set(tail_winner["horse_number"].astype(int))
            for k in KS:
                chosen=set(ordered.head(k)["horse_number"].astype(int))
                acc=method_acc[(method,k)]
                acc["races"]+=1
                if tw>0:
                    acc["miss_races"]+=1
                    acc["tail_any_hit"]+=int(bool(chosen & tail_true))
                    acc["tail_all_hit"]+=int(tail_true.issubset(chosen))
                add_set=top12|chosen
                addcap=captured(g,add_set)
                acc["add_capture"]+=int(addcap)
                acc["add_rescue"]+=int((not base) and addcap)

                keep_cut=12-k
                swap_base=set(g.loc[g["consensus_rank"]<=keep_cut,"horse_number"].astype(int))
                swap_set=swap_base|chosen
                swapcap=captured(g,swap_set)
                acc["swap_capture"]+=int(swapcap)
                acc["swap_rescue"]+=int((not base) and swapcap)
                acc["swap_lost_base"]+=int(base and not swapcap)

    rank_df=pd.DataFrame(rank_rows)
    rank_summary=[]
    if not rank_df.empty:
        for rank,g in rank_df.groupby("rank"):
            rank_summary.append({
                "year":year,"rank":int(rank),"appearances":int(len(g)),
                "top3_count":int(g["is_top3"].sum()),
                "top3_rate_pct":100*float(g["is_top3"].mean())
            })

    out_methods=[]
    base_pct=100*summary["base_capture"]/summary["challenge_races"] if summary["challenge_races"] else None
    for (method,k),acc in method_acc.items():
        r=acc["races"]; miss=acc["miss_races"]
        out_methods.append({
            "year":year,"method":method,"k":k,
            "challenge_races":r,
            "base_top12_capture_pct":base_pct,
            "tail_miss_races":miss,
            "tail_any_hit_pct_on_miss":100*acc["tail_any_hit"]/miss if miss else None,
            "tail_all_hit_pct_on_miss":100*acc["tail_all_hit"]/miss if miss else None,
            "add_full_capture_pct":100*acc["add_capture"]/r if r else None,
            "add_rescue_count":acc["add_rescue"],
            "add_rescue_pct_all":100*acc["add_rescue"]/r if r else None,
            "swap_full_capture_pct":100*acc["swap_capture"]/r if r else None,
            "swap_delta_vs_base_pp":100*acc["swap_capture"]/r-base_pct if r else None,
            "swap_rescue_count":acc["swap_rescue"],
            "swap_lost_base_count":acc["swap_lost_base"],
        })

    oracle=[]
    total=summary["challenge_races"]
    base=summary["base_capture"]
    for k in KS:
        possible=base+sum(summary.get(f"tail_intruders_{j}",0) for j in range(1,k+1))
        oracle.append({
            "year":year,"k":k,"challenge_races":total,
            "oracle_capture_count":possible,
            "oracle_capture_pct":100*possible/total if total else None,
            "oracle_gain_vs_top12_pp":100*(possible-base)/total if total else None
        })

    year_summary={
        "year":year,
        "challenge_races":summary["challenge_races"],
        "base_top12_capture_count":summary["base_capture"],
        "base_top12_capture_pct":100*summary["base_capture"]/summary["challenge_races"] if summary["challenge_races"] else None,
        "tail_miss_races":summary["tail_miss_races"],
        "tail_intrusion_pct":100*summary["tail_miss_races"]/summary["challenge_races"] if summary["challenge_races"] else None,
        "tail_intruders_0":summary["tail_intruders_0"],
        "tail_intruders_1":summary["tail_intruders_1"],
        "tail_intruders_2":summary["tail_intruders_2"],
        "tail_intruders_3":summary["tail_intruders_3"],
        "excluded":excluded,
    }
    return year_summary,rank_summary,out_methods,oracle,rank_df

def pool_rows(year_summaries,rank_frames,method_rows,oracle_rows):
    total=sum(r["challenge_races"] for r in year_summaries)
    base=sum(r["base_top12_capture_count"] for r in year_summaries)
    miss=sum(r["tail_miss_races"] for r in year_summaries)
    pooled_summary={
        "year":"POOLED","challenge_races":total,
        "base_top12_capture_count":base,
        "base_top12_capture_pct":100*base/total,
        "tail_miss_races":miss,
        "tail_intrusion_pct":100*miss/total,
        "tail_intruders_0":sum(r["tail_intruders_0"] for r in year_summaries),
        "tail_intruders_1":sum(r["tail_intruders_1"] for r in year_summaries),
        "tail_intruders_2":sum(r["tail_intruders_2"] for r in year_summaries),
        "tail_intruders_3":sum(r["tail_intruders_3"] for r in year_summaries),
    }

    rdf=pd.concat(rank_frames,ignore_index=True)
    pooled_rank=[]
    for rank,g in rdf.groupby("rank"):
        pooled_rank.append({
            "year":"POOLED","rank":int(rank),"appearances":int(len(g)),
            "top3_count":int(g["is_top3"].sum()),
            "top3_rate_pct":100*float(g["is_top3"].mean())
        })

    pm=[]
    mdf=pd.DataFrame(method_rows)
    for (method,k),g in mdf.groupby(["method","k"]):
        # Reconstruct counts from yearly percentages/counts.
        races=int(g["challenge_races"].sum())
        missr=int(g["tail_miss_races"].sum())
        add_rescue=int(g["add_rescue_count"].sum())
        swap_rescue=int(g["swap_rescue_count"].sum())
        swap_lost=int(g["swap_lost_base_count"].sum())
        # Weighted percentages by denominators.
        any_hits=sum((x["tail_any_hit_pct_on_miss"]/100)*x["tail_miss_races"] for _,x in g.iterrows())
        all_hits=sum((x["tail_all_hit_pct_on_miss"]/100)*x["tail_miss_races"] for _,x in g.iterrows())
        addcaps=sum((x["add_full_capture_pct"]/100)*x["challenge_races"] for _,x in g.iterrows())
        swapcaps=sum((x["swap_full_capture_pct"]/100)*x["challenge_races"] for _,x in g.iterrows())
        pm.append({
            "year":"POOLED","method":method,"k":int(k),"challenge_races":races,
            "base_top12_capture_pct":100*base/total,
            "tail_miss_races":missr,
            "tail_any_hit_pct_on_miss":100*any_hits/missr if missr else None,
            "tail_all_hit_pct_on_miss":100*all_hits/missr if missr else None,
            "add_full_capture_pct":100*addcaps/races if races else None,
            "add_rescue_count":add_rescue,
            "add_rescue_pct_all":100*add_rescue/races if races else None,
            "swap_full_capture_pct":100*swapcaps/races if races else None,
            "swap_delta_vs_base_pp":100*swapcaps/races-100*base/total if races else None,
            "swap_rescue_count":swap_rescue,
            "swap_lost_base_count":swap_lost,
        })

    po=[]
    for k in KS:
        possible=base+sum(pooled_summary.get(f"tail_intruders_{j}",0) for j in range(1,k+1))
        po.append({
            "year":"POOLED","k":k,"challenge_races":total,
            "oracle_capture_count":possible,
            "oracle_capture_pct":100*possible/total,
            "oracle_gain_vs_top12_pp":100*(possible-base)/total
        })
    return pooled_summary,pooled_rank,pm,po

def main():
    t0=time.time(); a=parse_args()
    c=json.load(open(a.contract,encoding="utf-8"))
    assert c["contract"]=="L17_TAIL_RESCUE_AUDIT_V1"
    assert c["cost_policy"]["github_standard_cpu_only"] is True
    assert c["cost_policy"]["gpu"] is False
    assert c["data_policy"]["2026_locked"] is True

    _,df=load_horse_dataset(a.dataset_dir)
    df,skipped_market=attach_win_market(df,a.backfill_root)
    df=attach_outsider(df,a.outsider_predictions)
    df=prep(prepare_l175(df))
    if 2026 in set(df["year"]): raise SystemExit("2026 sealed")

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    year_summaries=[]; rank_summaries=[]; method_rows=[]; oracle_rows=[]; rank_frames=[]
    model_meta=[]

    for test_year,train_years in FOLDS:
        model,nrows,pos,spw=fit_tail_model(df,train_years)
        model_meta.append({"test_year":test_year,"train_years":"|".join(map(str,train_years)),"train_tail_rows":nrows,"positive_tail_rows":pos,"scale_pos_weight":spw})
        ys,rs,mr,orr,rdf=analyze_year(df,test_year,model)
        year_summaries.append(ys); rank_summaries.extend(rs); method_rows.extend(mr); oracle_rows.extend(orr); rank_frames.append(rdf)
        print("TAIL_RESCUE_YEAR_DONE "+json.dumps({"year":test_year,"base_pct":ys["base_top12_capture_pct"],"tail_intrusion_pct":ys["tail_intrusion_pct"]},separators=(",",":")),flush=True)

    ps,pr,pm,po=pool_rows(year_summaries,rank_frames,method_rows,oracle_rows)
    year_summaries.append(ps); rank_summaries.extend(pr); method_rows.extend(pm); oracle_rows.extend(po)

    pd.DataFrame(year_summaries).to_csv(out/"top12-tail-summary.csv",index=False)
    pd.DataFrame(rank_summaries).to_csv(out/"rank13-18-top3-rates.csv",index=False)
    pd.DataFrame(method_rows).to_csv(out/"selector-rescue-metrics.csv",index=False)
    pd.DataFrame(oracle_rows).to_csv(out/"oracle-ceiling.csv",index=False)
    pd.DataFrame(model_meta).to_csv(out/"ml-fold-meta.csv",index=False)

    summary={
        "contract":"L17_TAIL_RESCUE_AUDIT_V1_RESULT",
        "question":"Can Seven-King ranks 13-18 rescue podium horses missed by horse-level Top12?",
        "pooled":ps,
        "pooled_rank13_18":[r for r in pr],
        "pooled_selector_metrics":[r for r in pm],
        "pooled_oracle":[r for r in po],
        "skipped_market_races":len(skipped_market),
        "selectors":list(METHODS),
        "2026_locked":True,
        "elapsed_seconds":time.time()-t0
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L1.7 Tail Rescue Audit V1\n\n"
        "Horse-level audit only. Base set is Seven-King ranks 1-12. Tail is ranks 13-18. "
        "No betting logic, no value-horse constraint, no payout optimization. "
        "Static selectors and a strict walk-forward tail-only ML selector are compared for adding or swapping 1-3 tail horses.\n",
        encoding="utf-8"
    )
    print("===== POOLED SUMMARY ====="); print(json.dumps(ps,ensure_ascii=False,indent=2))
    print("===== POOLED RANK 13-18 ====="); print(pd.DataFrame(pr).to_string(index=False))
    print("===== POOLED SELECTORS ====="); print(pd.DataFrame(pm).to_string(index=False))
    print("===== POOLED ORACLE ====="); print(pd.DataFrame(po).to_string(index=False))
    print("L17_TAIL_RESCUE_AUDIT_V1_READY")

if __name__=="__main__":
    main()
