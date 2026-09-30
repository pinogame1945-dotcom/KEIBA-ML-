#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import build_year_frame, write_csv
from run_l2_ticket_market_gap_v2 import add_market_features, market_feature_columns, train_rank_predict, add_ranks
from run_l2_market_gap_law_arena_v1 import (
    INPUT_YEARS, TEST_YEARS, DEV_YEARS, CONFIRM_YEAR,
    add_bins, sort_for_pick, metrics, key_series, union_frame, unique_vs,
    pick_law1, candidate_group_tables, candidate_specs, chosen_from_table, subset_years
)

LAW2_ID="GAP10P__model_rank_band=1-3"
MAX_STEPS=8
MIN_UNIQUE_DISCOVERY_TICKETS=100
MIN_UNIQUE_EACH_YEAR=20
MIN_INCREMENTAL_POOLED_ROI=105.0
MIN_INCREMENTAL_EX_MAX_ROI=95.0
MIN_YEARS_ROI_GE_90=2
MIN_WORST_YEAR_ROI=70.0

def parse_args():
    p=argparse.ArgumentParser()
    for y in INPUT_YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def law2_from_tables(tables):
    return chosen_from_table(
        tables,
        "GAP10P",
        (("model_rank_band","1-3"),)
    )

def frame_keys(g):
    return set(key_series(g).tolist()) if not g.empty else set()

def same_ticket_set(a,b):
    return frame_keys(a)==frame_keys(b)

def year_metrics(g,label):
    rows=[]
    for y in TEST_YEARS:
        m=metrics(g[g["year"]==y].copy(),label)
        m.update({"law_or_portfolio":label,"year":y})
        rows.append(m)
    return rows

def portfolio_period_rows(g,label):
    rows=[]
    for name,years in [
        ("DISCOVERY_2022_2024",DEV_YEARS),
        ("CONFIRM_2025",(CONFIRM_YEAR,)),
        ("ALL_2022_2025",TEST_YEARS),
    ]:
        m=metrics(subset_years(g,years),label)
        m.update({"portfolio":label,"period":name})
        rows.append(m)
    return rows

def candidate_incremental_stats(chosen,current_dev,current_confirm):
    dev=subset_years(chosen,DEV_YEARS)
    conf=subset_years(chosen,(CONFIRM_YEAR,))
    inc_dev=unique_vs(dev,current_dev)
    inc_conf=unique_vs(conf,current_confirm)

    pooled=metrics(inc_dev,"incremental_dev")
    confm=metrics(inc_conf,"incremental_confirm")
    by_year={}
    for y in DEV_YEARS:
        yy=inc_dev[inc_dev["year"]==y]
        by_year[y]=metrics(yy,f"incremental_{y}")

    tickets_each=[by_year[y]["tickets"] for y in DEV_YEARS]
    rois_each=[
        by_year[y]["roi_pct"] if by_year[y]["roi_pct"] is not None else -1.0
        for y in DEV_YEARS
    ]
    years_ge_90=sum(1 for x in rois_each if x>=90.0)
    eligible=(
        len(inc_dev)>=MIN_UNIQUE_DISCOVERY_TICKETS and
        min(tickets_each)>=MIN_UNIQUE_EACH_YEAR and
        (pooled["roi_pct"] or 0.0)>=MIN_INCREMENTAL_POOLED_ROI and
        (pooled["roi_after_remove_largest_win_pct"] or 0.0)>=MIN_INCREMENTAL_EX_MAX_ROI and
        years_ge_90>=MIN_YEARS_ROI_GE_90 and
        min(rois_each)>=MIN_WORST_YEAR_ROI
    )
    return {
        "eligible":eligible,
        "inc_dev":inc_dev,
        "inc_conf":inc_conf,
        "pooled":pooled,
        "confirm":confm,
        "by_year":by_year,
        "tickets_each":tickets_each,
        "rois_each":rois_each,
        "years_ge_90":years_ge_90,
    }

def overlap_row(a_id,a,b_id,b,period):
    aa=subset_years(a,period)
    bb=subset_years(b,period)
    ka=frame_keys(aa);kb=frame_keys(bb)
    inter=len(ka&kb);union=len(ka|kb)
    return {
        "law_a":a_id,
        "law_b":b_id,
        "tickets_a":len(ka),
        "tickets_b":len(kb),
        "overlap_tickets":inter,
        "a_overlap_pct":100.0*inter/len(ka) if ka else 0.0,
        "b_overlap_pct":100.0*inter/len(kb) if kb else 0.0,
        "jaccard_pct":100.0*inter/union if union else 0.0,
    }

def main():
    a=parse_args()
    paths={y:getattr(a,f"l17_{y}") for y in INPUT_YEARS}
    l17={y:load_l17(paths[y],y) for y in INPUT_YEARS}
    frames={y:add_market_features(build_year_frame(y,l17[y],a.backfill_root)) for y in INPUT_YEARS}
    cols=market_feature_columns(frames[2022])

    predictions={}
    folds=[]
    for y in TEST_YEARS:
        train_years=[2021] if y==2022 else [t for t in (2022,2023,2024) if t<y]
        train=pd.concat([frames[t] for t in train_years],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)
        score,_=train_rank_predict(train,test,cols,93000+y)
        test["market_aware_score"]=score
        test=add_bins(add_ranks(test))
        predictions[y]=test
        folds.append({
            "test_year":y,
            "train_years":"|".join(map(str,train_years)),
            "train_pairs":len(train),
            "test_pairs":len(test),
            "source_races":int(test["race_id"].nunique()),
        })

    pred_all=pd.concat([predictions[y] for y in TEST_YEARS],ignore_index=True)
    tables=candidate_group_tables(pred_all)

    law1=pick_law1(pred_all)
    law1_counts={y:int((law1["year"]==y).sum()) for y in TEST_YEARS}
    expected_law1={2022:105,2023:102,2024:84,2025:68}
    if law1_counts!=expected_law1:
        raise SystemExit(f"law1 drift got={law1_counts} expected={expected_law1}")

    law2=law2_from_tables(tables)
    law2_counts={y:int((law2["year"]==y).sum()) for y in TEST_YEARS}
    expected_law2={2022:197,2023:127,2024:103,2025:47}
    if law2_counts!=expected_law2:
        raise SystemExit(f"law2 drift got={law2_counts} expected={expected_law2}")

    selected=[
        {"law_id":"LAW1","candidate_id":"FROZEN_LAW1","chosen":law1},
        {"law_id":"LAW2","candidate_id":LAW2_ID,"chosen":law2},
    ]
    current=union_frame(law1,law2)
    current_dev=subset_years(current,DEV_YEARS)
    current_conf=subset_years(current,(CONFIRM_YEAR,))

    specs=candidate_specs()
    candidate_cache={}
    excluded_ids={LAW2_ID}
    for cid,gap,conds in specs:
        if cid in excluded_ids:
            continue
        chosen=chosen_from_table(tables,gap,conds)
        if chosen.empty:
            continue
        if same_ticket_set(chosen,law1) or same_ticket_set(chosen,law2):
            continue
        candidate_cache[cid]=(gap,conds,chosen)

    selection_rows=[]
    candidate_step_rows=[]
    # Record fixed laws as steps 1-2.
    base1=metrics(subset_years(law1,DEV_YEARS),"LAW1")
    base1c=metrics(subset_years(law1,(CONFIRM_YEAR,)),"LAW1")
    selection_rows.append({
        "step":1,"law_id":"LAW1","candidate_id":"FROZEN_LAW1",
        "selection_basis":"fixed",
        "incremental_dev_tickets":base1["tickets"],
        "incremental_dev_hits":base1["hits"],
        "incremental_dev_roi_pct":base1["roi_pct"],
        "incremental_dev_profit_yen":base1["profit_yen"],
        "incremental_dev_roi_after_remove_largest_win_pct":base1["roi_after_remove_largest_win_pct"],
        "confirm_incremental_tickets":base1c["tickets"],
        "confirm_incremental_hits":base1c["hits"],
        "confirm_incremental_roi_pct":base1c["roi_pct"],
        "confirm_incremental_profit_yen":base1c["profit_yen"],
    })
    law2_inc_dev=unique_vs(subset_years(law2,DEV_YEARS),subset_years(law1,DEV_YEARS))
    law2_inc_conf=unique_vs(subset_years(law2,(CONFIRM_YEAR,)),subset_years(law1,(CONFIRM_YEAR,)))
    base2=metrics(law2_inc_dev,"LAW2_INCREMENTAL")
    base2c=metrics(law2_inc_conf,"LAW2_INCREMENTAL")
    selection_rows.append({
        "step":2,"law_id":"LAW2","candidate_id":LAW2_ID,
        "selection_basis":"fixed",
        "incremental_dev_tickets":base2["tickets"],
        "incremental_dev_hits":base2["hits"],
        "incremental_dev_roi_pct":base2["roi_pct"],
        "incremental_dev_profit_yen":base2["profit_yen"],
        "incremental_dev_roi_after_remove_largest_win_pct":base2["roi_after_remove_largest_win_pct"],
        "confirm_incremental_tickets":base2c["tickets"],
        "confirm_incremental_hits":base2c["hits"],
        "confirm_incremental_roi_pct":base2c["roi_pct"],
        "confirm_incremental_profit_yen":base2c["profit_yen"],
    })

    used={LAW2_ID}
    for step in range(3,MAX_STEPS+1):
        eligible=[]
        for cid,(gap,conds,chosen) in candidate_cache.items():
            if cid in used:
                continue
            s=candidate_incremental_stats(chosen,current_dev,current_conf)
            p=s["pooled"];c=s["confirm"]
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
                "confirm_incremental_tickets":c["tickets"],
                "confirm_incremental_hits":c["hits"],
                "confirm_incremental_roi_pct":c["roi_pct"],
                "confirm_incremental_profit_yen":c["profit_yen"],
            }
            candidate_step_rows.append(row)
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
            reverse=True
        )
        cid,gap,conds,chosen,s,row=eligible[0]
        law_id=f"LAW{step}"
        before_dev=current_dev.copy()
        before_conf=current_conf.copy()
        current=union_frame(current,chosen)
        current_dev=subset_years(current,DEV_YEARS)
        current_conf=subset_years(current,(CONFIRM_YEAR,))
        pm=metrics(current_dev,f"PORTFOLIO_{step}")
        pcm=metrics(current_conf,f"PORTFOLIO_{step}")
        selection_rows.append({
            "step":step,
            "law_id":law_id,
            "candidate_id":cid,
            "selection_basis":"greedy_incremental_discovery_profit",
            "gap_spec":gap,
            "conditions":";".join(f"{d}={v}" for d,v in conds) if conds else "NONE",
            "incremental_dev_tickets":s["pooled"]["tickets"],
            "incremental_dev_hits":s["pooled"]["hits"],
            "incremental_dev_roi_pct":s["pooled"]["roi_pct"],
            "incremental_dev_profit_yen":s["pooled"]["profit_yen"],
            "incremental_dev_roi_after_remove_largest_win_pct":s["pooled"]["roi_after_remove_largest_win_pct"],
            "incremental_dev_worst_year_roi_pct":min(s["rois_each"]),
            "confirm_incremental_tickets":s["confirm"]["tickets"],
            "confirm_incremental_hits":s["confirm"]["hits"],
            "confirm_incremental_roi_pct":s["confirm"]["roi_pct"],
            "confirm_incremental_profit_yen":s["confirm"]["profit_yen"],
            "portfolio_dev_tickets_after":pm["tickets"],
            "portfolio_dev_roi_pct_after":pm["roi_pct"],
            "portfolio_dev_profit_yen_after":pm["profit_yen"],
            "portfolio_2025_tickets_after":pcm["tickets"],
            "portfolio_2025_roi_pct_after":pcm["roi_pct"],
            "portfolio_2025_profit_yen_after":pcm["profit_yen"],
        })
        selected.append({"law_id":law_id,"candidate_id":cid,"chosen":chosen})
        used.add(cid)

    # Final portfolio and selected-law reports.
    final=current
    selected_year_rows=[]
    for x in selected:
        selected_year_rows.extend(year_metrics(x["chosen"],x["law_id"]))
    portfolio_year_rows=year_metrics(final,"FINAL_PORTFOLIO")
    portfolio_period=portfolio_period_rows(final,"FINAL_PORTFOLIO")

    overlap=[]
    for i,a0 in enumerate(selected):
        for j,b0 in enumerate(selected):
            if j<i:
                continue
            r=overlap_row(a0["law_id"],a0["chosen"],b0["law_id"],b0["chosen"],TEST_YEARS)
            overlap.append(r)

    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"folds.csv",folds)
    pd.DataFrame(selection_rows).to_csv(out/"selected-law-sequence.csv",index=False)
    pd.DataFrame(candidate_step_rows).to_csv(out/"candidate-step-audit.csv",index=False)
    pd.DataFrame(selected_year_rows).to_csv(out/"selected-laws-by-year.csv",index=False)
    pd.DataFrame(portfolio_year_rows).to_csv(out/"final-portfolio-by-year.csv",index=False)
    pd.DataFrame(portfolio_period).to_csv(out/"final-portfolio-periods.csv",index=False)
    pd.DataFrame(overlap).to_csv(out/"selected-law-overlap.csv",index=False)

    final_dev=metrics(subset_years(final,DEV_YEARS),"FINAL_PORTFOLIO_DEV")
    final_conf=metrics(subset_years(final,(CONFIRM_YEAR,)),"FINAL_PORTFOLIO_2025")
    summary={
        "contract":"L2_MARKET_GAP_LAW_PORTFOLIO_V1_RESULT",
        "fixed_laws":["LAW1",LAW2_ID],
        "selected_law_count":len(selected),
        "selected_sequence":[
            {k:v for k,v in row.items() if k not in {"selection_basis"}}
            for row in selection_rows
        ],
        "final_discovery":final_dev,
        "final_confirmation_2025":final_conf,
        "selection_used_2025":False,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_MARKET_GAP_LAW_PORTFOLIO_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
