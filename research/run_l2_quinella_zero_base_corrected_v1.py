#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import build_year_frame, write_csv
from run_l2_ticket_market_gap_v2 import add_market_features, market_feature_columns, train_rank_predict, add_ranks
from run_l2_market_gap_law_arena_v1 import (
    add_bins, candidate_group_tables, candidate_specs, chosen_from_table,
    metrics, union_frame, unique_vs, key_series,
)

INPUT_YEARS=(2021,2022,2023,2024,2025)
DEV_YEARS=(2022,2023,2024)
CONFIRM_YEAR=2025
MAX_STEPS=8

MIN_UNIQUE_DEV_TICKETS=120
MIN_UNIQUE_EACH_DEV_YEAR=30
MIN_INCREMENTAL_POOLED_ROI=105.0
MIN_INCREMENTAL_EX_MAX_ROI=95.0
MIN_YEARS_ROI_GE_90=2
MIN_WORST_YEAR_ROI=70.0

def parse_args():
    p=argparse.ArgumentParser()
    for y in INPUT_YEARS:
        p.add_argument(f"--l17-{y}", required=True)
    p.add_argument("--backfill-root", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()

def subset_years(g, years):
    return g[g["year"].isin(years)].copy()

def frame_keys(g):
    return set(key_series(g).tolist()) if not g.empty else set()

def predict_year(year, frames, cols):
    train_years=[2021] if year==2022 else [t for t in DEV_YEARS if t<year]
    train=pd.concat([frames[t] for t in train_years], ignore_index=True)
    test=frames[year].copy().reset_index(drop=True)
    score,_=train_rank_predict(train, test, cols, 94000+year)
    test["market_aware_score"]=score
    test=add_bins(add_ranks(test))
    fold={
        "test_year":year,
        "train_years":"|".join(map(str, train_years)),
        "train_pairs":len(train),
        "test_pairs":len(test),
        "source_races":int(test["race_id"].nunique()),
    }
    return test, fold

def incremental_stats(chosen, current):
    inc=unique_vs(chosen, current)
    pooled=metrics(inc, "incremental_dev")
    by_year={}
    for y in DEV_YEARS:
        yy=inc[inc["year"]==y].copy()
        by_year[y]=metrics(yy, f"incremental_{y}")
    tickets_each=[by_year[y]["tickets"] for y in DEV_YEARS]
    rois_each=[
        by_year[y]["roi_pct"] if by_year[y]["roi_pct"] is not None else -1.0
        for y in DEV_YEARS
    ]
    years_ge_90=sum(1 for x in rois_each if x>=90.0)
    eligible=(
        len(inc)>=MIN_UNIQUE_DEV_TICKETS and
        min(tickets_each)>=MIN_UNIQUE_EACH_DEV_YEAR and
        (pooled["roi_pct"] or 0.0)>=MIN_INCREMENTAL_POOLED_ROI and
        (pooled["roi_after_remove_largest_win_pct"] or 0.0)>=MIN_INCREMENTAL_EX_MAX_ROI and
        years_ge_90>=MIN_YEARS_ROI_GE_90 and
        min(rois_each)>=MIN_WORST_YEAR_ROI
    )
    return {
        "eligible":eligible,
        "incremental":inc,
        "pooled":pooled,
        "by_year":by_year,
        "tickets_each":tickets_each,
        "rois_each":rois_each,
        "years_ge_90":years_ge_90,
    }

def main():
    a=parse_args()
    paths={y:getattr(a, f"l17_{y}") for y in INPUT_YEARS}
    l17={y:load_l17(paths[y], y) for y in INPUT_YEARS}
    frames={y:add_market_features(build_year_frame(y, l17[y], a.backfill_root)) for y in INPUT_YEARS}
    cols=market_feature_columns(frames[2022])

    folds=[]
    dev_predictions={}
    for y in DEV_YEARS:
        pred, fold=predict_year(y, frames, cols)
        dev_predictions[y]=pred
        folds.append(fold)

    pred_dev=pd.concat([dev_predictions[y] for y in DEV_YEARS], ignore_index=True)
    dev_tables=candidate_group_tables(pred_dev)
    specs=candidate_specs()

    candidate_cache={}
    for cid,gap,conds in specs:
        chosen=chosen_from_table(dev_tables, gap, conds)
        if chosen.empty:
            continue
        candidate_cache[cid]=(gap,conds,chosen)

    # No frozen seed. Start from a genuinely empty portfolio.
    current_dev=pred_dev.iloc[0:0].copy()
    used=set()
    selected=[]
    selection_rows=[]
    candidate_rows=[]

    for step in range(1, MAX_STEPS+1):
        eligible=[]
        for cid,(gap,conds,chosen) in candidate_cache.items():
            if cid in used:
                continue
            s=incremental_stats(chosen, current_dev)
            p=s["pooled"]
            row={
                "step_considered":step,
                "candidate_id":cid,
                "gap_spec":gap,
                "conditions":";".join(f"{d}={v}" for d,v in conds) if conds else "NONE",
                "eligible":bool(s["eligible"]),
                "incremental_dev_tickets":p["tickets"],
                "incremental_dev_hits":p["hits"],
                "incremental_dev_roi_pct":p["roi_pct"],
                "incremental_dev_profit_yen":p["profit_yen"],
                "incremental_dev_roi_after_remove_largest_win_pct":p["roi_after_remove_largest_win_pct"],
                "incremental_dev_largest_return_share_pct":p["largest_return_share_pct"],
                "incremental_dev_worst_year_roi_pct":min(s["rois_each"]),
                "incremental_dev_years_roi_ge_90":s["years_ge_90"],
                "incremental_dev_min_year_tickets":min(s["tickets_each"]),
            }
            candidate_rows.append(row)
            if s["eligible"]:
                eligible.append((cid,gap,conds,chosen,s,row))

        if not eligible:
            break

        eligible.sort(
            key=lambda x:(
                x[5]["incremental_dev_profit_yen"],
                x[5]["incremental_dev_roi_after_remove_largest_win_pct"],
                x[5]["incremental_dev_worst_year_roi_pct"],
                x[5]["incremental_dev_tickets"],
                x[0],
            ),
            reverse=True,
        )
        cid,gap,conds,chosen,s,row=eligible[0]
        current_dev=union_frame(current_dev, chosen)
        pm=metrics(current_dev, f"PORTFOLIO_STEP_{step}")
        rule_id=f"RULE{step}"
        selection_rows.append({
            "step":step,
            "rule_id":rule_id,
            "candidate_id":cid,
            "selection_basis":"greedy_dev_profit_zero_seed",
            "gap_spec":gap,
            "conditions":";".join(f"{d}={v}" for d,v in conds) if conds else "NONE",
            "incremental_dev_tickets":s["pooled"]["tickets"],
            "incremental_dev_hits":s["pooled"]["hits"],
            "incremental_dev_roi_pct":s["pooled"]["roi_pct"],
            "incremental_dev_profit_yen":s["pooled"]["profit_yen"],
            "incremental_dev_roi_after_remove_largest_win_pct":s["pooled"]["roi_after_remove_largest_win_pct"],
            "incremental_dev_worst_year_roi_pct":min(s["rois_each"]),
            "portfolio_dev_tickets_after":pm["tickets"],
            "portfolio_dev_hits_after":pm["hits"],
            "portfolio_dev_roi_pct_after":pm["roi_pct"],
            "portfolio_dev_profit_yen_after":pm["profit_yen"],
        })
        selected.append((rule_id,cid,gap,conds,chosen))
        used.add(cid)

    # Confirmation is opened only after the full rule sequence is frozen.
    pred_confirm, fold=predict_year(CONFIRM_YEAR, frames, cols)
    folds.append(fold)
    confirm_tables=candidate_group_tables(pred_confirm)

    current_confirm=pred_confirm.iloc[0:0].copy()
    selected_dev_year_rows=[]
    selected_confirm_rows=[]
    for rule_id,cid,gap,conds,chosen_dev in selected:
        for y in DEV_YEARS:
            m=metrics(chosen_dev[chosen_dev["year"]==y].copy(), rule_id)
            m.update({"rule_id":rule_id,"candidate_id":cid,"year":y})
            selected_dev_year_rows.append(m)
        chosen_confirm=chosen_from_table(confirm_tables, gap, conds)
        cm=metrics(chosen_confirm, rule_id)
        cm.update({"rule_id":rule_id,"candidate_id":cid,"year":CONFIRM_YEAR})
        selected_confirm_rows.append(cm)
        current_confirm=union_frame(current_confirm, chosen_confirm)

    final_dev=metrics(current_dev, "ZERO_BASE_FINAL_DEV")
    final_confirm=metrics(current_confirm, "ZERO_BASE_FINAL_2025")

    out=Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"folds.csv", folds)
    pd.DataFrame(selection_rows).to_csv(out/"selected-rule-sequence.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(out/"candidate-step-audit.csv", index=False)
    pd.DataFrame(selected_dev_year_rows).to_csv(out/"selected-rules-dev-by-year.csv", index=False)
    pd.DataFrame(selected_confirm_rows).to_csv(out/"selected-rules-confirm-2025.csv", index=False)

    summary={
        "contract":"L2_QUINELLA_ZERO_BASE_CORRECTED_ODDS_V1",
        "source_variant":"ODDS_COMMA_CORRECTED",
        "odds_decoder_numeric_normalization":"remove thousands separators before float conversion",
        "zero_seed":True,
        "legacy_laws_seeded":False,
        "selection_years":list(DEV_YEARS),
        "confirmation_year":CONFIRM_YEAR,
        "selection_used_2025":False,
        "selected_rule_count":len(selected),
        "candidate_count":len(candidate_cache),
        "selected_sequence":[
            {
                "step":row["step"],
                "rule_id":row["rule_id"],
                "candidate_id":row["candidate_id"],
                "gap_spec":row["gap_spec"],
                "conditions":row["conditions"],
                "incremental_dev_tickets":row["incremental_dev_tickets"],
                "incremental_dev_hits":row["incremental_dev_hits"],
                "incremental_dev_roi_pct":row["incremental_dev_roi_pct"],
                "incremental_dev_profit_yen":row["incremental_dev_profit_yen"],
                "incremental_dev_roi_after_remove_largest_win_pct":row["incremental_dev_roi_after_remove_largest_win_pct"],
                "incremental_dev_worst_year_roi_pct":row["incremental_dev_worst_year_roi_pct"],
            }
            for row in selection_rows
        ],
        "final_discovery":final_dev,
        "final_confirmation_2025":final_confirm,
        "thresholds":{
            "min_unique_dev_tickets":MIN_UNIQUE_DEV_TICKETS,
            "min_unique_each_dev_year":MIN_UNIQUE_EACH_DEV_YEAR,
            "min_incremental_pooled_roi_pct":MIN_INCREMENTAL_POOLED_ROI,
            "min_incremental_ex_max_roi_pct":MIN_INCREMENTAL_EX_MAX_ROI,
            "min_years_roi_ge_90":MIN_YEARS_ROI_GE_90,
            "min_worst_year_roi_pct":MIN_WORST_YEAR_ROI,
            "max_steps":MAX_STEPS,
        },
        "2026_locked":True,
    }
    (out/"summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)+"\n",
        encoding="utf-8",
    )
    print("L2_QUINELLA_ZERO_BASE_CORRECTED_ODDS_V1_READY")
    print(json.dumps(summary, ensure_ascii=False, separators=(",",":")))

if __name__=="__main__":
    main()
