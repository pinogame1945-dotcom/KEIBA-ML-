#!/usr/bin/env python3
import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import (
    build_year_frame, feature_columns, evaluate_chosen, baseline, write_csv
)
from run_l2_ticket_market_gap_v2 import (
    MODEL_TOP_K, MIN_UPGRADE, MAX_TICKETS,
    add_market_features, market_feature_columns, train_rank_predict, add_ranks,
    ranking_rows, pure_rank_policy, apply_gap_policy,
    selected_gap_distribution, selected_pair_distribution, selected_rank_involvement
)

TRAIN_YEAR=2021
TEST_YEAR=2022
FROZEN_POLICY={"model_top_k":15,"min_market_rank_upgrade":5,"max_tickets_per_race":1}

def args():
    p=argparse.ArgumentParser(description="Past-independent market-gap test: train 2021 -> test 2022.")
    p.add_argument("--l17-2021",required=True)
    p.add_argument("--l17-2022",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def gap_bands(df):
    z=df.copy()
    z["gap_band"]=pd.cut(
        z["rank_upgrade"],
        [-10**9,5,7,10,15,25,10**9],
        labels=["<5","5-6","7-9","10-14","15-24","25+"],
        right=False
    ).astype(str)
    rows=[]
    for band,g in z.groupby("gap_band",sort=False):
        if not len(g): continue
        m=evaluate_chosen(g,g["race_id"].nunique(),str(band))
        m["gap_band"]=band
        rows.append(m)
    return rows

def main():
    a=args()
    l17_2021=load_l17(a.l17_2021,TRAIN_YEAR)
    l17_2022=load_l17(a.l17_2022,TEST_YEAR)

    f21=add_market_features(build_year_frame(TRAIN_YEAR,l17_2021,a.backfill_root))
    f22=add_market_features(build_year_frame(TEST_YEAR,l17_2022,a.backfill_root))
    cols=market_feature_columns(f21)

    score,imp=train_rank_predict(f21,f22.reset_index(drop=True),cols,93000+TEST_YEAR)
    test=f22.reset_index(drop=True).copy()
    test["market_aware_score"]=score
    test=add_ranks(test)

    source=test["race_id"].nunique()
    frozen=apply_gap_policy(
        test,
        FROZEN_POLICY["model_top_k"],
        FROZEN_POLICY["min_market_rank_upgrade"],
        FROZEN_POLICY["max_tickets_per_race"],
    )
    frozen_metrics=evaluate_chosen(frozen,source,"FROZEN_V2_POLICY_ON_2022")
    frozen_metrics.update(FROZEN_POLICY)

    grid=[]
    for k in MODEL_TOP_K:
        for up in MIN_UPGRADE:
            for mx in MAX_TICKETS:
                if mx>k: continue
                c=apply_gap_policy(test,k,up,mx)
                m=evaluate_chosen(c,source,f"MODEL_TOP{k}_UP{up}_MAX{mx}")
                m.update({"model_top_k":k,"min_market_rank_upgrade":up,"max_tickets_per_race":mx})
                grid.append(m)

    pure=[]
    for label,col in [
        ("MARKET","market_rank"),
        ("L17_PRODUCT","l17_rank_score"),
        ("MARKET_AWARE_MODEL","model_rank"),
    ]:
        for n in (1,2,3,5,10):
            m=evaluate_chosen(pure_rank_policy(test,col,n),source,f"{label}_TOP{n}")
            m.update({"year":TEST_YEAR,"ranking":label,"top_n":n})
            pure.append(m)

    baselines=[]
    for name in ("TOP4_BOX","TOP1_TO6","TOP6_BOX"):
        m=evaluate_chosen(baseline(test,name),source,name)
        m["year"]=TEST_YEAR
        baselines.append(m)

    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"ranking-comparison.csv",ranking_rows(test,TEST_YEAR))
    write_csv(out/"pure-ranking-policies.csv",pure)
    write_csv(out/"frozen-policy.csv",[frozen_metrics])
    write_csv(out/"policy-grid-2022-diagnostic.csv",grid)
    write_csv(out/"frozen-gap-bands.csv",gap_bands(frozen))
    write_csv(out/"baseline-metrics.csv",baselines)
    write_csv(out/"feature-importance.csv",imp)
    write_csv(out/"selected-gap-distribution.csv",selected_gap_distribution(frozen))
    write_csv(out/"selected-rank-pairs.csv",selected_pair_distribution(frozen))
    write_csv(out/"selected-rank-involvement.csv",selected_rank_involvement(frozen))
    with gzip.open(out/"selected-tickets.csv.gz","wt",newline="",encoding="utf-8") as fh:
        keep=[
            "year","race_id","race_date","pair_horse_ids","pair_numbers",
            "market_q_norm","market_rank","pair_prob_product","l17_rank_score",
            "market_aware_score","model_rank","rank_upgrade",
            "hit","return_yen_per100","odds",
            "a_consensus_rank","b_consensus_rank"
        ]
        frozen[keep].to_csv(fh,index=False)

    profitable=sum(1 for r in grid if r["roi_pct"] is not None and r["roi_pct"]>100)
    summary={
        "contract":"L2_MARKET_GAP_PAST_INDEPENDENT_2022_RESULT",
        "purpose":"Test the already-frozen 2025-observed market-gap hypothesis on an earlier untouched evaluation year.",
        "train_year":TRAIN_YEAR,
        "test_year":TEST_YEAR,
        "training_data_uses_test_year":False,
        "policy_selected_on_2022":False,
        "frozen_policy":FROZEN_POLICY,
        "frozen_policy_result":frozen_metrics,
        "diagnostic_policy_count":len(grid),
        "diagnostic_profitable_policy_count":profitable,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Market Gap — Past Independent 2022\n\n"
        "Train the market-aware LambdaRank model using 2021 only and evaluate 2022 only. "
        "The V2 frozen policy (model Top15, market-rank upgrade >=5, max 1 ticket) is applied without tuning on 2022. "
        "The full 2022 policy grid is diagnostic only and must not be promoted based on 2022. "
        "This test was conceived after inspecting 2025, so it is a historical independent check of that hypothesis, "
        "not a pristine forward holdout. 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("L2_MARKET_GAP_PAST_INDEPENDENT_2022_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
