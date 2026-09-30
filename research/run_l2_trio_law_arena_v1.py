#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

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
UPGRADE_THRESHOLDS=(5,10,20)
MODEL_RANK_BANDS=(
    ("1-3",1,3),
    ("4-5",4,5),
    ("6-10",6,10),
    ("11-20",11,20),
)


def parse_args():
    p=argparse.ArgumentParser(description="TRIO LAW Arena V1: fixed 12-candidate rank-upgrade x model-rank grid.")
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
    hit_races=int(g.loc[g["hit"]==1,"race_id"].nunique()) if n else 0
    wins=sorted(
        [float(x) for x in g.loc[g["return_yen_per100"]>0,"return_yen_per100"]],
        reverse=True,
    )

    def roi_ex_top(k):
        removed=sum(wins[:k])
        return 100.0*(ret-removed)/stake if stake else None

    return {
        "label":label,
        "tickets":int(n),
        "races":int(g["race_id"].nunique()) if n else 0,
        "hits":hits,
        "hit_races":hit_races,
        "hit_rate_pct":100.0*hits/n if n else 0.0,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "roi_ex_top1_win_pct":roi_ex_top(1),
        "roi_ex_top2_wins_pct":roi_ex_top(2),
        "roi_ex_top3_wins_pct":roi_ex_top(3),
        "largest_single_return_yen":wins[0] if wins else 0.0,
        "largest3_return_share_pct":100.0*sum(wins[:3])/ret if ret>0 else 0.0,
        "median_odds":float(g["odds"].median()) if n else None,
        "mean_odds":float(g["odds"].mean()) if n else None,
        "median_market_rank":float(g["market_rank"].median()) if n else None,
        "median_model_rank":float(g["model_rank"].median()) if n else None,
        "median_rank_upgrade":float(g["rank_upgrade"].median()) if n else None,
    }


def write_csv(path,rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)


def candidate_id(threshold,band):
    return f"UP{threshold}__MODEL_{band}"


def main():
    a=parse_args()
    if 2026 in INPUT_YEARS or CONFIRM_YEAR==2026:
        raise SystemExit("2026 sealed")

    l17_paths={y:getattr(a,f"l17_{y}") for y in INPUT_YEARS}
    l17={y:load_l17(l17_paths[y],y) for y in INPUT_YEARS}
    work=Path(a.work_dir)
    out=Path(a.out_dir)
    work.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)

    # Session reuse: build each full TRIO base table exactly once in this runner.
    base_paths={}
    build_rows=[]
    for y in INPUT_YEARS:
        frame,meta=build_year_frame(y,l17[y],a.backfill_root)
        frame=add_market_features(frame)
        p=work/f"trio-base-{y}.pkl.gz"
        frame.to_pickle(p,compression="gzip")
        base_paths[y]=p
        build_rows.append(meta)
        print(f"TRIO_LAW_BASE_READY year={y} rows={len(frame)} path={p}",flush=True)
        del frame

    # Walk-forward model predictions. Persist runner-local predictions so all 12 LAW
    # candidates reuse the exact same model output without recomputation.
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
        score,_=train_rank_predict(train,test,cols,95000+y)
        test["market_aware_score"]=score
        test=add_ranks(test)
        p=work/f"trio-pred-{y}.pkl.gz"
        keep=[
            "year","race_id","race_date","trio_numbers","hit","return_yen_per100",
            "odds","market_rank","model_rank","rank_upgrade","market_aware_score",
            "field_size","trio_rank_sum","trio_worst_rank","trio_top3_count",
            "trio_top6_count","trio_outside_top6_count",
        ]
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
        print(f"TRIO_LAW_PRED_READY year={y} rows={len(test)} path={p}",flush=True)
        del train,test

    preds={y:pd.read_pickle(pred_paths[y],compression="gzip") for y in TEST_YEARS}
    dev=pd.concat([preds[y] for y in DEV_YEARS],ignore_index=True)
    confirm=preds[CONFIRM_YEAR]

    arena=[]
    by_year=[]
    ticket_samples=[]
    for threshold in UPGRADE_THRESHOLDS:
        for band,lo,hi in MODEL_RANK_BANDS:
            cid=candidate_id(threshold,band)
            d=dev[
                (dev["rank_upgrade"]>=threshold)&
                (dev["model_rank"]>=lo)&
                (dev["model_rank"]<=hi)
            ].copy()
            c=confirm[
                (confirm["rank_upgrade"]>=threshold)&
                (confirm["model_rank"]>=lo)&
                (confirm["model_rank"]<=hi)
            ].copy()

            dm=metrics(d,cid)
            cm=metrics(c,cid)
            yr_metrics=[]
            for y in DEV_YEARS:
                yy=d[d["year"]==y]
                ym=metrics(yy,cid)
                ym.update({"candidate_id":cid,"year":y,"period":"DISCOVERY"})
                by_year.append(ym)
                yr_metrics.append(ym)
            ym=metrics(c,cid)
            ym.update({"candidate_id":cid,"year":CONFIRM_YEAR,"period":"CONFIRMATION_NOT_FOR_SELECTION"})
            by_year.append(ym)

            arena.append({
                "candidate_id":cid,
                "rank_upgrade_min":threshold,
                "model_rank_band":band,
                "model_rank_lo":lo,
                "model_rank_hi":hi,
                "dev_tickets":dm["tickets"],
                "dev_races":dm["races"],
                "dev_hits":dm["hits"],
                "dev_hit_rate_pct":dm["hit_rate_pct"],
                "dev_roi_pct":dm["roi_pct"],
                "dev_profit_yen":dm["profit_yen"],
                "dev_roi_ex_top1_win_pct":dm["roi_ex_top1_win_pct"],
                "dev_roi_ex_top2_wins_pct":dm["roi_ex_top2_wins_pct"],
                "dev_roi_ex_top3_wins_pct":dm["roi_ex_top3_wins_pct"],
                "dev_largest3_return_share_pct":dm["largest3_return_share_pct"],
                "dev_median_odds":dm["median_odds"],
                "dev_mean_odds":dm["mean_odds"],
                "dev_median_market_rank":dm["median_market_rank"],
                "dev_median_model_rank":dm["median_model_rank"],
                "dev_median_rank_upgrade":dm["median_rank_upgrade"],
                "dev_min_year_tickets":min(x["tickets"] for x in yr_metrics),
                "dev_min_year_roi_pct":min((x["roi_pct"] or 0.0) for x in yr_metrics),
                "confirm_2025_tickets":cm["tickets"],
                "confirm_2025_races":cm["races"],
                "confirm_2025_hits":cm["hits"],
                "confirm_2025_hit_rate_pct":cm["hit_rate_pct"],
                "confirm_2025_roi_pct":cm["roi_pct"],
                "confirm_2025_profit_yen":cm["profit_yen"],
                "confirm_2025_roi_ex_top1_win_pct":cm["roi_ex_top1_win_pct"],
                "confirm_2025_roi_ex_top2_wins_pct":cm["roi_ex_top2_wins_pct"],
                "confirm_2025_roi_ex_top3_wins_pct":cm["roi_ex_top3_wins_pct"],
                "confirm_2025_median_odds":cm["median_odds"],
            })

            sample_cols=[
                "year","race_id","race_date","trio_numbers","odds","market_rank",
                "model_rank","rank_upgrade","hit","return_yen_per100",
            ]
            sample=pd.concat([
                d.sort_values(
                    ["rank_upgrade","model_rank","odds"],
                    ascending=[False,True,False],
                ).head(20).assign(candidate_id=cid,period="DISCOVERY"),
                c.sort_values(
                    ["rank_upgrade","model_rank","odds"],
                    ascending=[False,True,False],
                ).head(10).assign(candidate_id=cid,period="CONFIRMATION"),
            ],ignore_index=True)
            ticket_samples.append(sample[["candidate_id","period"]+sample_cols])

    arena_df=pd.DataFrame(arena)
    # Critical: discovery ordering uses only 2022-2024 columns.
    arena_df=arena_df.sort_values(
        [
            "dev_roi_ex_top3_wins_pct",
            "dev_roi_ex_top2_wins_pct",
            "dev_roi_ex_top1_win_pct",
            "dev_roi_pct",
            "dev_min_year_tickets",
            "dev_tickets",
        ],
        ascending=[False,False,False,False,False,False],
        na_position="last",
    ).reset_index(drop=True)
    arena_df["discovery_rank"]=range(1,len(arena_df)+1)

    top=arena_df.head(5).copy()
    write_csv(out/"build-coverage.csv",build_rows)
    write_csv(out/"folds.csv",fold_rows)
    arena_df.to_csv(out/"candidate-arena.csv",index=False)
    pd.DataFrame(by_year).to_csv(out/"candidate-by-year.csv",index=False)
    top.to_csv(out/"top5-discovery-candidates.csv",index=False)
    pd.concat(ticket_samples,ignore_index=True).to_csv(out/"candidate-ticket-samples.csv",index=False)

    summary={
        "contract":"L2_TRIO_LAW_ARENA_V1_RESULT",
        "bet_type":"TRIO",
        "candidate_count":int(len(arena_df)),
        "candidate_grid":{
            "rank_upgrade_min":list(UPGRADE_THRESHOLDS),
            "model_rank_bands":[{"label":b,"lo":lo,"hi":hi} for b,lo,hi in MODEL_RANK_BANDS],
        },
        "discovery_years":list(DEV_YEARS),
        "confirmation_year":CONFIRM_YEAR,
        "confirmation_used_for_selection":False,
        "selection_sort":"dev ROI after removing top3 wins, then top2, top1, raw ROI, ticket support",
        "session_reuse":{
            "base_table_built_once_per_year":True,
            "prediction_table_built_once_per_test_year":True,
            "law_candidates_reuse_prediction_tables":True,
            "github_artifact_or_cache_used":False,
        },
        "top5_discovery_candidates":top.to_dict(orient="records"),
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_TRIO_LAW_ARENA_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
