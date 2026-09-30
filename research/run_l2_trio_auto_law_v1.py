#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_market_gap_trio_v0 import (
    add_market_features,
    add_ranks,
    build_year_frame,
    feature_columns,
    train_rank_predict,
)

INPUT_YEARS=(2021,2022,2023,2024,2025)
TEST_YEARS=(2022,2023,2024,2025)
DEV_YEARS=(2022,2023,2024)
CONFIRM_YEAR=2025

RULE_FEATURES=(
    "rank_upgrade",
    "model_rank",
    "market_rank",
    "odds",
    "field_size",
    "trio_rank_sum",
    "trio_worst_rank",
    "trio_top3_count",
    "trio_top6_count",
    "trio_outside_top6_count",
)

# Tree complexity is selected only from discovery-year leave-one-year-out validation.
# No split threshold/band is hand-written: sklearn learns every boundary from 2022-2024.
MAX_LEAF_CANDIDATES=(4,8,16,32)


def parse_args():
    p=argparse.ArgumentParser(
        description="TRIO Auto LAW V1: learn rule boundaries from discovery data; 2025 confirmation only."
    )
    for y in INPUT_YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--work-dir",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def metrics(g,label):
    n=len(g)
    stake=100.0*n
    ret=float(g["return_yen_per100"].sum()) if n else 0.0
    hits=int(g["hit"].sum()) if n else 0
    wins=sorted(
        [float(x) for x in g.loc[g["return_yen_per100"]>0,"return_yen_per100"]],
        reverse=True,
    )
    def roi_ex(k):
        return 100.0*(ret-sum(wins[:k]))/stake if stake else None
    return {
        "label":label,
        "tickets":int(n),
        "races":int(g["race_id"].nunique()) if n else 0,
        "hits":hits,
        "hit_rate_pct":100.0*hits/n if n else 0.0,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "roi_ex_top1_win_pct":roi_ex(1),
        "roi_ex_top2_wins_pct":roi_ex(2),
        "roi_ex_top3_wins_pct":roi_ex(3),
        "largest_single_return_yen":wins[0] if wins else 0.0,
        "largest3_return_share_pct":100.0*sum(wins[:3])/ret if ret>0 else 0.0,
        "median_odds":float(g["odds"].median()) if n else None,
        "mean_odds":float(g["odds"].mean()) if n else None,
        "median_market_rank":float(g["market_rank"].median()) if n else None,
        "median_model_rank":float(g["model_rank"].median()) if n else None,
        "median_rank_upgrade":float(g["rank_upgrade"].median()) if n else None,
    }


def x_matrix(df):
    x=df[list(RULE_FEATURES)].copy()
    for c in RULE_FEATURES:
        x[c]=pd.to_numeric(x[c],errors="coerce").fillna(0.0).astype("float32")
    return x


def min_leaf_for(n):
    # Data-size-derived regularization, not a ticket-rule boundary.
    # At least 500 rows; otherwise 0.5% of the discovery training sample.
    return max(500,int(round(n*0.005)))


def train_policy_tree(train,max_leaf_nodes):
    x=x_matrix(train)
    # Gross return multiple: 0 for losers, e.g. 23.4 for a 2340-yen return.
    # Poisson is appropriate for non-negative, highly skewed return targets.
    y=(pd.to_numeric(train["return_yen_per100"],errors="coerce").fillna(0.0)/100.0).to_numpy()
    min_leaf=min_leaf_for(len(train))
    tree=DecisionTreeRegressor(
        criterion="poisson",
        splitter="best",
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_leaf,
        random_state=20260930,
    )
    tree.fit(x,y)
    leaf=tree.apply(x)
    tmp=pd.DataFrame({
        "leaf":leaf,
        "return_yen_per100":train["return_yen_per100"].to_numpy(),
    })
    stats=tmp.groupby("leaf",as_index=False).agg(
        train_tickets=("return_yen_per100","size"),
        train_return_yen=("return_yen_per100","sum"),
    )
    stats["train_roi_pct"]=100.0*stats["train_return_yen"]/(100.0*stats["train_tickets"])
    # Break-even is the only leaf acceptance threshold.
    selected=set(stats.loc[stats["train_roi_pct"]>100.0,"leaf"].astype(int))
    return tree,selected,stats,min_leaf


def apply_policy(tree,selected,df):
    leaf=tree.apply(x_matrix(df))
    mask=np.isin(leaf,np.fromiter(selected,dtype=np.int64) if selected else np.array([],dtype=np.int64))
    out=df.loc[mask].copy()
    out["auto_leaf_id"]=leaf[mask]
    return out


def main():
    a=parse_args()
    if 2026 in INPUT_YEARS or CONFIRM_YEAR==2026:
        raise SystemExit("2026 sealed")

    work=Path(a.work_dir); work.mkdir(parents=True,exist_ok=True)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    l17_paths={y:getattr(a,f"l17_{y}") for y in INPUT_YEARS}
    l17={y:load_l17(l17_paths[y],y) for y in INPUT_YEARS}

    # One runner session: base tables built once and reused for all model/rule work.
    base_paths={}
    build_rows=[]
    for y in INPUT_YEARS:
        frame,meta=build_year_frame(y,l17[y],a.backfill_root)
        frame=add_market_features(frame)
        p=work/f"trio-base-{y}.pkl.gz"
        frame.to_pickle(p,compression="gzip")
        base_paths[y]=p
        build_rows.append(meta)
        print(f"TRIO_AUTO_BASE_READY year={y} rows={len(frame)}",flush=True)
        del frame

    pred_paths={}
    fold_rows=[]
    for y in TEST_YEARS:
        train_years=[2021] if y==2022 else [t for t in (2022,2023,2024) if t<y]
        train=pd.concat(
            [pd.read_pickle(base_paths[t],compression="gzip") for t in train_years],
            ignore_index=True,
        )
        test=pd.read_pickle(base_paths[y],compression="gzip").reset_index(drop=True)
        cols=feature_columns(train)
        score,_=train_rank_predict(train,test,cols,96000+y)
        test["market_aware_score"]=score
        test=add_ranks(test)
        keep=[
            "year","race_id","race_date","trio_numbers","hit","return_yen_per100",
            "odds","market_rank","model_rank","rank_upgrade","market_aware_score",
            "field_size","trio_rank_sum","trio_worst_rank","trio_top3_count",
            "trio_top6_count","trio_outside_top6_count",
        ]
        p=work/f"trio-pred-{y}.pkl.gz"
        test[keep].to_pickle(p,compression="gzip")
        pred_paths[y]=p
        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(map(str,train_years)),
            "train_trios":int(len(train)),
            "test_trios":int(len(test)),
            "source_races":int(test["race_id"].nunique()),
            "feature_count":int(len(cols)),
        })
        print(f"TRIO_AUTO_PRED_READY year={y} rows={len(test)}",flush=True)
        del train,test

    preds={y:pd.read_pickle(pred_paths[y],compression="gzip") for y in TEST_YEARS}

    # Select tree complexity by leave-one-discovery-year-out validation.
    cv_rows=[]
    config_rows=[]
    for max_leaf_nodes in MAX_LEAF_CANDIDATES:
        per_holdout=[]
        for holdout in DEV_YEARS:
            train_years=[y for y in DEV_YEARS if y!=holdout]
            tr=pd.concat([preds[y] for y in train_years],ignore_index=True)
            va=preds[holdout]
            tree,selected,leaf_stats,min_leaf=train_policy_tree(tr,max_leaf_nodes)
            chosen=apply_policy(tree,selected,va)
            m=metrics(chosen,f"leaf{max_leaf_nodes}_holdout{holdout}")
            row={
                "max_leaf_nodes":max_leaf_nodes,
                "min_samples_leaf":min_leaf,
                "holdout_year":holdout,
                "selected_train_leaves":len(selected),
                **m,
            }
            cv_rows.append(row)
            per_holdout.append(row)
            del tr,chosen,tree,leaf_stats

        pooled_tickets=sum(x["tickets"] for x in per_holdout)
        pooled_return=sum(x["return_yen"] for x in per_holdout)
        pooled_roi=100.0*pooled_return/(100.0*pooled_tickets) if pooled_tickets else 0.0
        worst_roi=min((x["roi_pct"] or 0.0) for x in per_holdout)
        worst_ex1=min((x["roi_ex_top1_win_pct"] or 0.0) for x in per_holdout)
        worst_ex3=min((x["roi_ex_top3_wins_pct"] or 0.0) for x in per_holdout)
        config_rows.append({
            "max_leaf_nodes":max_leaf_nodes,
            "cv_total_tickets":pooled_tickets,
            "cv_pooled_roi_pct":pooled_roi,
            "cv_worst_year_roi_pct":worst_roi,
            "cv_worst_year_roi_ex_top1_pct":worst_ex1,
            "cv_worst_year_roi_ex_top3_pct":worst_ex3,
        })

    configs=pd.DataFrame(config_rows).sort_values(
        [
            "cv_worst_year_roi_ex_top3_pct",
            "cv_worst_year_roi_ex_top1_pct",
            "cv_worst_year_roi_pct",
            "cv_pooled_roi_pct",
            "cv_total_tickets",
        ],
        ascending=[False,False,False,False,False],
    ).reset_index(drop=True)
    configs["selection_rank"]=range(1,len(configs)+1)
    chosen_leaf_nodes=int(configs.iloc[0]["max_leaf_nodes"])

    dev=pd.concat([preds[y] for y in DEV_YEARS],ignore_index=True)
    final_tree,selected_leaves,leaf_stats,min_leaf=train_policy_tree(dev,chosen_leaf_nodes)
    dev_selected=apply_policy(final_tree,selected_leaves,dev)
    confirm_selected=apply_policy(final_tree,selected_leaves,preds[CONFIRM_YEAR])

    dev_m=metrics(dev_selected,"DISCOVERY_2022_2024")
    conf_m=metrics(confirm_selected,"CONFIRMATION_2025")
    tree_text=export_text(final_tree,feature_names=list(RULE_FEATURES),decimals=3)

    leaf_stats=leaf_stats.sort_values("train_roi_pct",ascending=False).reset_index(drop=True)
    leaf_stats["selected_break_even_plus"]=leaf_stats["leaf"].astype(int).isin(selected_leaves)

    pd.DataFrame(build_rows).to_csv(out/"build-coverage.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(out/"ranking-folds.csv",index=False)
    pd.DataFrame(cv_rows).to_csv(out/"complexity-loyo.csv",index=False)
    configs.to_csv(out/"complexity-selection.csv",index=False)
    leaf_stats.to_csv(out/"final-tree-leaves.csv",index=False)
    dev_selected.to_csv(out/"selected-discovery-tickets.csv.gz",index=False,compression="gzip")
    confirm_selected.to_csv(out/"selected-confirmation-2025-tickets.csv.gz",index=False,compression="gzip")
    (out/"final-tree.txt").write_text(tree_text,encoding="utf-8")

    summary={
        "contract":"L2_TRIO_AUTO_LAW_V1_RESULT",
        "bet_type":"TRIO",
        "manual_ticket_rule_boundaries":False,
        "rule_features":list(RULE_FEATURES),
        "boundary_learning":"DecisionTreeRegressor learns all split thresholds from 2022-2024 only",
        "complexity_selection":"leave-one-discovery-year-out; 2025 excluded",
        "complexity_candidates":{"max_leaf_nodes":list(MAX_LEAF_CANDIDATES)},
        "chosen_max_leaf_nodes":chosen_leaf_nodes,
        "chosen_min_samples_leaf":min_leaf,
        "selected_positive_roi_leaves":sorted(map(int,selected_leaves)),
        "leaf_acceptance":"train leaf ROI > 100% (break-even only)",
        "discovery_years":list(DEV_YEARS),
        "confirmation_year":CONFIRM_YEAR,
        "confirmation_used_for_selection":False,
        "discovery_metrics":dev_m,
        "confirmation_2025_metrics":conf_m,
        "session_reuse":{
            "base_table_built_once_per_year":True,
            "prediction_table_built_once_per_test_year":True,
            "auto_rule_search_reuses_predictions":True,
            "github_artifact_or_cache_used":False,
        },
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_TRIO_AUTO_LAW_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
