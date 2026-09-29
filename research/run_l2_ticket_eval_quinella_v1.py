#!/usr/bin/env python3
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from run_l2_ticket_eval_quinella_v0 import (
    YEARS, TEST_YEARS, DEV_YEARS, HOLDOUT_YEAR,
    parse_args, parse_paths, build_year_frame, feature_columns, train_predict,
    evaluate_chosen, baseline, write_csv
)
from build_l2_l17_fullfield_dataset_v1 import load_l17

RANKERS={
    "MODEL":"prediction",
    "L17_PRODUCT":"pair_prob_product",
}
SHORTLIST_K=(1,3,5,10,15)
FINAL_N=(1,2,3,5)
EDGE_GRID=(0.00,0.10,0.20,0.50,1.00)
MIN_DEV_EXECUTION_COVERAGE_PCT=20.0

def add_ranks(df):
    out=df.copy()
    for label,col in RANKERS.items():
        out[f"rank_{label}"]=(
            out.sort_values(["year","race_id",col,"pair_rank_sum","pair_numbers"],
                            ascending=[True,True,False,True,True])
               .groupby(["year","race_id"]).cumcount()
               .add(1)
               .reindex(out.index)
        )
    return out

def pure_rank_policy(df,ranker,n):
    return df[df[f"rank_{ranker}"]<=n].copy()

def gated_market_policy(df,ranker,shortlist_k,min_edge,final_n):
    z=df[
        (df[f"rank_{ranker}"]<=shortlist_k) &
        (df["edge"]>=min_edge)
    ].copy()
    if z.empty:
        return z
    z=z.sort_values(
        ["year","race_date","race_id","edge",f"rank_{ranker}","prediction"],
        ascending=[True,True,True,False,True,False]
    )
    return z.groupby(["year","race_id"],sort=False).head(final_n).copy()

def selected_pair_distribution(chosen):
    if chosen.empty:
        return pd.DataFrame(columns=[
            "a_consensus_rank","b_consensus_rank","tickets","hits",
            "stake_yen","return_yen","roi_pct"
        ])
    g=chosen.groupby(["a_consensus_rank","b_consensus_rank"],as_index=False).agg(
        tickets=("pair_horse_ids","size"),
        hits=("hit","sum"),
        return_yen=("return_yen_per100","sum"),
    )
    g["stake_yen"]=g["tickets"]*100.0
    g["roi_pct"]=100.0*g["return_yen"]/g["stake_yen"]
    return g.sort_values(["tickets","a_consensus_rank","b_consensus_rank"],
                         ascending=[False,True,True])

def rank_involvement(chosen):
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
    frames={y:build_year_frame(y,l17[y],a.backfill_root) for y in YEARS}
    cols=feature_columns(frames[2022])

    fold_rows=[]
    predictions={}
    importances=[]
    for y in TEST_YEARS:
        train=pd.concat([frames[t] for t in YEARS if t<y],ignore_index=True)
        test=frames[y].copy()
        p,imp=train_predict(train,test,cols,92000+y)
        test["prediction"]=p
        test["edge"]=test["prediction"]*test["odds"]-1.0
        test=add_ranks(test)
        predictions[y]=test

        yy=test["hit"].astype(int).to_numpy()
        eps=np.clip(p,1e-9,1-1e-9)
        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(str(t) for t in YEARS if t<y),
            "train_pairs":len(train),
            "test_pairs":len(test),
            "source_races":test["race_id"].nunique(),
            "positive_rate_pct":100.0*float(yy.mean()),
            "roc_auc":float(roc_auc_score(yy,p)),
            "log_loss":float(log_loss(yy,eps)),
            "brier":float(brier_score_loss(yy,p)),
            "feature_count":len(cols),
        })
        imp["test_year"]=y
        importances.append(imp)

    # A-stage only: no market. This answers whether ticket ranking alone helps.
    pure_rows=[]
    for y in TEST_YEARS:
        df=predictions[y]
        source=df["race_id"].nunique()
        for ranker in RANKERS:
            for n in (1,2,3,5,10):
                chosen=pure_rank_policy(df,ranker,n)
                m=evaluate_chosen(chosen,source,f"{ranker}_TOP{n}")
                m.update({"year":y,"ranker":ranker,"top_n":n})
                pure_rows.append(m)

    # B-stage: market may only act inside a pre-race shortlist.
    dev=pd.concat([predictions[y] for y in DEV_YEARS],ignore_index=True)
    dev_source=dev[["year","race_id"]].drop_duplicates().shape[0]
    grid=[]
    for ranker in RANKERS:
        for k in SHORTLIST_K:
            for edge in EDGE_GRID:
                for n in FINAL_N:
                    if n>k:
                        continue
                    chosen=gated_market_policy(dev,ranker,k,edge,n)
                    m=evaluate_chosen(
                        chosen,dev_source,
                        f"{ranker}_TOP{k}_EDGE{edge:.2f}_MAX{n}"
                    )
                    m.update({
                        "ranker":ranker,
                        "shortlist_k":k,
                        "min_edge":edge,
                        "max_tickets_per_race":n,
                    })
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
    hold_chosen=gated_market_policy(
        hold,
        champion["ranker"],
        int(champion["shortlist_k"]),
        float(champion["min_edge"]),
        int(champion["max_tickets_per_race"]),
    )
    hold_metrics=evaluate_chosen(
        hold_chosen,hold_source,
        "FROZEN_DEV_CHAMPION"
    )
    hold_metrics.update({
        "ranker":champion["ranker"],
        "shortlist_k":int(champion["shortlist_k"]),
        "min_edge":float(champion["min_edge"]),
        "max_tickets_per_race":int(champion["max_tickets_per_race"]),
        "selected_from_dev_years":list(DEV_YEARS),
    })

    # Holdout matrix is diagnostic only; the frozen champion above is the promoted comparison.
    hold_matrix=[]
    for ranker in RANKERS:
        for k in SHORTLIST_K:
            for edge in EDGE_GRID:
                for n in FINAL_N:
                    if n>k:
                        continue
                    chosen=gated_market_policy(hold,ranker,k,edge,n)
                    m=evaluate_chosen(chosen,hold_source,f"{ranker}_TOP{k}_EDGE{edge:.2f}_MAX{n}")
                    m.update({"ranker":ranker,"shortlist_k":k,"min_edge":edge,"max_tickets_per_race":n})
                    hold_matrix.append(m)

    baseline_rows=[]
    for y in TEST_YEARS:
        df=predictions[y]
        source=df["race_id"].nunique()
        for name in ("TOP4_BOX","TOP1_TO6","TOP6_BOX"):
            m=evaluate_chosen(baseline(df,name),source,name)
            m["year"]=y
            baseline_rows.append(m)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"model-fold-metrics.csv",fold_rows)
    write_csv(out/"pure-ranking-policies.csv",pure_rows)
    write_csv(out/"dev-shortlist-market-grid.csv",grid)
    write_csv(out/"holdout-shortlist-market-matrix-diagnostic.csv",hold_matrix)
    write_csv(out/"baseline-metrics.csv",baseline_rows)
    write_csv(out/"feature-importance.csv",pd.concat(importances,ignore_index=True))
    write_csv(out/"holdout-selected-rank-pairs.csv",selected_pair_distribution(hold_chosen))
    write_csv(out/"holdout-selected-rank-involvement.csv",rank_involvement(hold_chosen))

    with gzip.open(out/"holdout-selected-tickets.csv.gz","wt",newline="",encoding="utf-8") as f:
        keep=[
            "year","race_id","race_date","pair_horse_ids","pair_numbers",
            "prediction","odds","edge","hit","return_yen_per100",
            "a_consensus_rank","b_consensus_rank","pair_prob_sum","pair_prob_product",
            f'rank_{champion["ranker"]}'
        ]
        hold_chosen[keep].to_csv(f,index=False)

    summary={
        "contract":"L2_TICKET_EVAL_QUINELLA_V1_RESULT",
        "architecture":"pre_market_ticket_ranking_then_market_inside_shortlist",
        "bet_type":"QUINELLA",
        "all_runners_scored":True,
        "candidate_horse_selection":False,
        "template_router":False,
        "model_input_odds":False,
        "market_cannot_promote_outside_shortlist":True,
        "rankers":list(RANKERS),
        "shortlist_k":list(SHORTLIST_K),
        "development_years":list(DEV_YEARS),
        "holdout_year":HOLDOUT_YEAR,
        "holdout_used_for_policy_selection":False,
        "policy_selection_objective":"highest development ROI with >=20% execution coverage; tie-break profit/DD/coverage",
        "selected_policy":champion,
        "holdout":hold_metrics,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Ticket Evaluator — QUINELLA V1\n\n"
        "Stage A ranks every quinella ticket using pre-market L1.7/race information. "
        "Stage B may use final odds only inside a frozen Stage-A shortlist, so a longshot outside the shortlist "
        "cannot jump to the top solely because of price. MODEL and raw L1.7 pair-product ranking are both tested. "
        "Pure ranking policies are reported separately from market-gated policies. Development is 2023-2024; "
        "the selected policy is frozen before 2025. Outsider is unused and 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("L2_TICKET_EVAL_QUINELLA_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
