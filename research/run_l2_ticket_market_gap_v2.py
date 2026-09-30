#!/usr/bin/env python3
import gzip
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l2_ticket_eval_quinella_v0 import (
    YEARS, TEST_YEARS, DEV_YEARS, HOLDOUT_YEAR,
    parse_args, parse_paths, build_year_frame, feature_columns, encode,
    evaluate_chosen, baseline, write_csv
)
from build_l2_l17_fullfield_dataset_v1 import load_l17

MODEL_TOP_K=(1,3,5,10,15)
MIN_UPGRADE=(0,1,2,3,5,10)
MAX_TICKETS=(1,2,3,5)
MIN_DEV_EXECUTION_COVERAGE_PCT=20.0

def add_market_features(df):
    z=df.copy()
    z["market_inv_odds"]=1.0/z["odds"].clip(lower=1e-9)
    denom=z.groupby(["year","race_id"])["market_inv_odds"].transform("sum")
    z["market_q_norm"]=z["market_inv_odds"]/denom.clip(lower=1e-12)
    z["market_log_q"]=np.log(z["market_q_norm"].clip(lower=1e-12))
    z=z.sort_values(
        ["year","race_id","market_q_norm","pair_rank_sum","pair_numbers"],
        ascending=[True,True,False,True,True]
    ).copy()
    z["market_rank"]=z.groupby(["year","race_id"],sort=False).cumcount()+1
    return z.sort_index()

def market_feature_columns(df):
    pre=feature_columns(df)
    # Horse numbers created spurious importance in V0/V1. Keep race/L1.7 structure,
    # but remove raw number identifiers from the market-relative model.
    pre=[c for c in pre if c not in {
        "a_horse_number","b_horse_number","pair_horse_number_gap"
    }]
    return pre+["market_q_norm","market_log_q","market_rank"]

def train_rank_predict(train,test,cols,seed):
    # Races without an observed winning QUINELLA pair do not define a ranking label.
    good=train.groupby(["year","race_id"])["hit"].transform("sum")>0
    tr=train.loc[good].copy()
    tr=tr.sort_values(["year","race_id","pair_numbers"]).reset_index(drop=True)
    te=test.copy().sort_values(["year","race_id","pair_numbers"])
    te["_orig_index"]=te.index
    te=te.reset_index(drop=True)

    xtr,xte=encode(tr,te,cols)
    y=tr["hit"].astype(int).to_numpy()
    group=tr.groupby(["year","race_id"],sort=False).size().tolist()

    model=lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=360,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=4.0,
        reg_alpha=0.5,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(xtr,y,group=group)
    pred=np.asarray(model.predict(xte),dtype=float)
    s=pd.Series(pred,index=te["_orig_index"].astype(int).to_numpy())
    out=s.reindex(test.index).to_numpy()
    imp=pd.DataFrame({"feature":cols,"gain":model.feature_importances_})
    return out,imp

def add_ranks(df):
    z=df.copy()
    specs=[
        ("market_rank_score","market_q_norm"),
        ("l17_rank_score","pair_prob_product"),
        ("model_rank","market_aware_score"),
    ]
    for rank_col,score_col in specs:
        if rank_col=="market_rank_score":
            z[rank_col]=z["market_rank"].astype(int)
            continue
        ordered=z.sort_values(
            ["year","race_id",score_col,"pair_rank_sum","pair_numbers"],
            ascending=[True,True,False,True,True]
        )
        r=ordered.groupby(["year","race_id"],sort=False).cumcount()+1
        z.loc[ordered.index,rank_col]=r.to_numpy()
        z[rank_col]=z[rank_col].astype(int)
    z["rank_upgrade"]=z["market_rank"]-z["model_rank"]
    z["l17_vs_market_upgrade"]=z["market_rank"]-z["l17_rank_score"]
    return z

def ranking_rows(df,year):
    n=df["race_id"].nunique()
    rows=[]
    for label,rank_col in [
        ("MARKET","market_rank"),
        ("L17_PRODUCT","l17_rank_score"),
        ("MARKET_AWARE_MODEL","model_rank"),
    ]:
        for k in (1,3,5,10,15):
            hit_races=df[(df[rank_col]<=k)&(df["hit"]==1)]["race_id"].nunique()
            rows.append({
                "year":year,"ranking":label,"top_k":k,
                "hit_races":hit_races,"source_races":n,
                "coverage_pct":100.0*hit_races/n if n else 0.0
            })
    return rows

def apply_gap_policy(df,model_top_k,min_upgrade,max_tickets):
    z=df[
        (df["model_rank"]<=model_top_k) &
        (df["rank_upgrade"]>=min_upgrade)
    ].copy()
    if z.empty:
        return z
    z=z.sort_values(
        ["year","race_date","race_id","rank_upgrade","model_rank","market_rank","market_aware_score"],
        ascending=[True,True,True,False,True,False,False]
    )
    return z.groupby(["year","race_id"],sort=False).head(max_tickets).copy()

def pure_rank_policy(df,rank_col,n):
    return df[df[rank_col]<=n].copy()

def selected_gap_distribution(chosen):
    if chosen.empty:
        return pd.DataFrame(columns=["rank_upgrade","tickets","hits","return_yen","stake_yen","roi_pct"])
    g=chosen.groupby("rank_upgrade",as_index=False).agg(
        tickets=("pair_horse_ids","size"),
        hits=("hit","sum"),
        return_yen=("return_yen_per100","sum")
    )
    g["stake_yen"]=g["tickets"]*100.0
    g["roi_pct"]=100.0*g["return_yen"]/g["stake_yen"]
    return g.sort_values("rank_upgrade",ascending=False)

def selected_pair_distribution(chosen):
    if chosen.empty:
        return pd.DataFrame(columns=[
            "a_consensus_rank","b_consensus_rank","tickets","hits","return_yen","stake_yen","roi_pct"
        ])
    g=chosen.groupby(["a_consensus_rank","b_consensus_rank"],as_index=False).agg(
        tickets=("pair_horse_ids","size"),
        hits=("hit","sum"),
        return_yen=("return_yen_per100","sum")
    )
    g["stake_yen"]=g["tickets"]*100.0
    g["roi_pct"]=100.0*g["return_yen"]/g["stake_yen"]
    return g.sort_values(["tickets","a_consensus_rank","b_consensus_rank"],ascending=[False,True,True])

def selected_rank_involvement(chosen):
    rows=[]
    if chosen.empty:
        return pd.DataFrame(columns=["consensus_rank","ticket_mentions","ticket_share_pct","hits_involving_rank"])
    total=len(chosen)
    max_rank=int(max(chosen["a_consensus_rank"].max(),chosen["b_consensus_rank"].max()))
    for r in range(1,max_rank+1):
        mask=(chosen["a_consensus_rank"]==r)|(chosen["b_consensus_rank"]==r)
        rows.append({
            "consensus_rank":r,
            "ticket_mentions":int(mask.sum()),
            "ticket_share_pct":100.0*float(mask.sum())/total,
            "hits_involving_rank":int(chosen.loc[mask,"hit"].sum()),
        })
    return pd.DataFrame(rows)

def main():
    a=parse_args()
    lp=parse_paths(a.l17_year)
    if set(lp)!=set(YEARS):
        raise SystemExit("L1.7 year path mismatch")
    if 2026 in lp:
        raise SystemExit("2026 sealed")

    l17={y:load_l17(lp[y],y) for y in YEARS}
    frames={y:add_market_features(build_year_frame(y,l17[y],a.backfill_root)) for y in YEARS}
    cols=market_feature_columns(frames[2022])

    predictions={}
    fold_rows=[]
    importance=[]
    ranking=[]
    for y in TEST_YEARS:
        train=pd.concat([frames[t] for t in YEARS if t<y],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)
        score,imp=train_rank_predict(train,test,cols,93000+y)
        test["market_aware_score"]=score
        test=add_ranks(test)
        predictions[y]=test
        ranking.extend(ranking_rows(test,y))
        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(str(t) for t in YEARS if t<y),
            "train_pairs":len(train),
            "test_pairs":len(test),
            "source_races":test["race_id"].nunique(),
            "positive_pairs":int(test["hit"].sum()),
            "feature_count":len(cols),
            "mean_abs_rank_upgrade":float(test["rank_upgrade"].abs().mean()),
            "pct_pairs_upgraded":100.0*float((test["rank_upgrade"]>0).mean()),
        })
        imp["test_year"]=y
        importance.append(imp)

    # Pure ranking economics: no value threshold, to see whether the market-aware
    # model actually ranks winners better before selecting "undervalued" gaps.
    pure=[]
    for y in TEST_YEARS:
        df=predictions[y]
        source=df["race_id"].nunique()
        for label,col in [
            ("MARKET","market_rank"),
            ("L17_PRODUCT","l17_rank_score"),
            ("MARKET_AWARE_MODEL","model_rank"),
        ]:
            for n in (1,2,3,5,10):
                m=evaluate_chosen(pure_rank_policy(df,col,n),source,f"{label}_TOP{n}")
                m.update({"year":y,"ranking":label,"top_n":n})
                pure.append(m)

    dev=pd.concat([predictions[y] for y in DEV_YEARS],ignore_index=True)
    dev_source=dev[["year","race_id"]].drop_duplicates().shape[0]
    grid=[]
    for k in MODEL_TOP_K:
        for up in MIN_UPGRADE:
            for mx in MAX_TICKETS:
                if mx>k:
                    continue
                chosen=apply_gap_policy(dev,k,up,mx)
                m=evaluate_chosen(chosen,dev_source,f"MODEL_TOP{k}_UP{up}_MAX{mx}")
                m.update({"model_top_k":k,"min_market_rank_upgrade":up,"max_tickets_per_race":mx})
                grid.append(m)

    eligible=[r for r in grid if r["execution_coverage_pct"]>=MIN_DEV_EXECUTION_COVERAGE_PCT]
    pool=eligible or grid
    champion=max(pool,key=lambda r:(
        r["roi_pct"] if r["roi_pct"] is not None else -1e18,
        r["profit_yen"],
        -r["max_drawdown_yen"],
        r["execution_coverage_pct"],
    ))

    hold=predictions[HOLDOUT_YEAR]
    hold_source=hold["race_id"].nunique()
    hold_chosen=apply_gap_policy(
        hold,
        int(champion["model_top_k"]),
        int(champion["min_market_rank_upgrade"]),
        int(champion["max_tickets_per_race"])
    )
    hold_metrics=evaluate_chosen(hold_chosen,hold_source,"FROZEN_DEV_CHAMPION")
    hold_metrics.update({
        "model_top_k":int(champion["model_top_k"]),
        "min_market_rank_upgrade":int(champion["min_market_rank_upgrade"]),
        "max_tickets_per_race":int(champion["max_tickets_per_race"]),
        "selected_from_dev_years":list(DEV_YEARS),
    })

    # Holdout matrix is diagnostic only.
    hold_matrix=[]
    for k in MODEL_TOP_K:
        for up in MIN_UPGRADE:
            for mx in MAX_TICKETS:
                if mx>k:
                    continue
                chosen=apply_gap_policy(hold,k,up,mx)
                m=evaluate_chosen(chosen,hold_source,f"MODEL_TOP{k}_UP{up}_MAX{mx}")
                m.update({"model_top_k":k,"min_market_rank_upgrade":up,"max_tickets_per_race":mx})
                hold_matrix.append(m)

    baselines=[]
    for y in TEST_YEARS:
        df=predictions[y]
        source=df["race_id"].nunique()
        for name in ("TOP4_BOX","TOP1_TO6","TOP6_BOX"):
            m=evaluate_chosen(baseline(df,name),source,name)
            m["year"]=y
            baselines.append(m)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"model-fold-metrics.csv",fold_rows)
    write_csv(out/"ranking-comparison.csv",ranking)
    write_csv(out/"pure-ranking-policies.csv",pure)
    write_csv(out/"dev-market-gap-grid.csv",grid)
    write_csv(out/"holdout-market-gap-matrix-diagnostic.csv",hold_matrix)
    write_csv(out/"baseline-metrics.csv",baselines)
    write_csv(out/"feature-importance.csv",pd.concat(importance,ignore_index=True))
    write_csv(out/"holdout-selected-gap-distribution.csv",selected_gap_distribution(hold_chosen))
    write_csv(out/"holdout-selected-rank-pairs.csv",selected_pair_distribution(hold_chosen))
    write_csv(out/"holdout-selected-rank-involvement.csv",selected_rank_involvement(hold_chosen))
    with gzip.open(out/"holdout-selected-tickets.csv.gz","wt",newline="",encoding="utf-8") as f:
        keep=[
            "year","race_id","race_date","pair_horse_ids","pair_numbers",
            "market_q_norm","market_rank","pair_prob_product","l17_rank_score",
            "market_aware_score","model_rank","rank_upgrade",
            "hit","return_yen_per100","odds",
            "a_consensus_rank","b_consensus_rank"
        ]
        hold_chosen[keep].to_csv(f,index=False)

    summary={
        "contract":"L2_TICKET_MARKET_GAP_V2_RESULT",
        "architecture":"market_aware_lambdarank_then_rank_upgrade_selection",
        "bet_type":"QUINELLA",
        "all_runners_scored":True,
        "candidate_horse_selection":False,
        "template_router":False,
        "raw_ev_multiplication":False,
        "market_probability":"normalized inverse quinella odds within race",
        "market_features":["market_q_norm","market_log_q","market_rank"],
        "value_signal":"market_rank - market_aware_model_rank",
        "development_years":list(DEV_YEARS),
        "holdout_year":HOLDOUT_YEAR,
        "holdout_used_for_policy_selection":False,
        "selected_policy":champion,
        "holdout":hold_metrics,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Ticket Market Gap — QUINELLA V2\n\n"
        "This version removes raw predicted-probability × odds EV selection. "
        "For each race, inverse quinella odds are normalized into a market probability distribution. "
        "A LightGBM LambdaRank model sees market probability/rank plus L1.7 and race structure, and learns the winning-pair ranking. "
        "The value signal is rank upgrade versus market: market_rank - model_rank. "
        "Tickets are selected only when the model upgrades them relative to the market. "
        "2023-2024 choose the policy; 2025 is a frozen holdout; 2026 stays sealed. "
        "Payout/result fields are evaluation labels only, not model features.\n",
        encoding="utf-8"
    )
    print("L2_TICKET_MARKET_GAP_V2_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
