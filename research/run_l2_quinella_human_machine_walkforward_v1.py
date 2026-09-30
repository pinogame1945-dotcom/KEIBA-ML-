#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import build_year_frame
from run_l2_ticket_market_gap_v2 import (
    add_market_features,
    market_feature_columns,
    train_rank_predict,
    add_ranks,
)
from run_l2_market_gap_law_arena_v1 import (
    add_bins,
    candidate_group_tables,
    candidate_specs,
    chosen_from_table,
    metrics,
    union_frame,
    unique_vs,
    key_series,
)
from run_l2_quinella_auto_calibration_v1 import train_base, normalize_race

YEARS=(2021,2022,2023,2024,2025)
TEST_YEARS=(2023,2024,2025)
PRED_YEARS=(2022,2023,2024,2025)
MAX_STEPS=8

MIN_UNIQUE_EACH_DEV_YEAR=30
MIN_INCREMENTAL_POOLED_ROI=105.0
MIN_INCREMENTAL_EX_MAX_ROI=95.0
MIN_WORST_YEAR_ROI=70.0

EXPECTED_2025_SEQUENCE=[
    "GAP7_9__field_size_band=15-16__model_rank_band=6-8",
    "GAP10P__odds_band=40-60__field_size_band=13-14",
    "GAP10P__pair_worse_rank_band=11-14__market_rank_band=21-25",
    "GAP5_6__odds_band=60-80__model_rank_band=11-15",
    "GAP0_2__field_size_band=13-14__l17_rank_band=11-20",
    "GAP3_4__model_rank_band=9-10__l17_rank_band=21-30",
    "GAP3_4__market_rank_band=<=10__l17_rank_band=31-40",
    "GAP5P__market_rank_band=<=10__l17_rank_band=41-60",
]

def parse_args():
    p=argparse.ArgumentParser()
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def scaled_thresholds(dev_years):
    n=len(dev_years)
    return {
        "min_unique_dev_tickets":40*n,
        "min_unique_each_dev_year":MIN_UNIQUE_EACH_DEV_YEAR,
        "min_incremental_pooled_roi_pct":MIN_INCREMENTAL_POOLED_ROI,
        "min_incremental_ex_max_roi_pct":MIN_INCREMENTAL_EX_MAX_ROI,
        "min_years_roi_ge_90":max(1,math.ceil((2*n)/3)),
        "min_worst_year_roi_pct":MIN_WORST_YEAR_ROI,
        "max_steps":MAX_STEPS,
    }

def predict_human_year(year,frames,cols):
    train_years=[2021] if year==2022 else [t for t in (2022,2023,2024) if t<year]
    train=pd.concat([frames[t] for t in train_years],ignore_index=True)
    test=frames[year].copy().reset_index(drop=True)
    score,_=train_rank_predict(train,test,cols,94000+year)
    test["market_aware_score"]=score
    test=add_bins(add_ranks(test))
    return test,train_years

def select_human_rules(predictions,dev_years):
    pred_dev=pd.concat([predictions[y] for y in dev_years],ignore_index=True)
    tables=candidate_group_tables(pred_dev)
    specs=candidate_specs()
    cache={}
    for cid,gap,conds in specs:
        chosen=chosen_from_table(tables,gap,conds)
        if not chosen.empty:
            cache[cid]=(gap,conds,chosen)

    th=scaled_thresholds(dev_years)
    current=pred_dev.iloc[0:0].copy()
    used=set()
    selected=[]

    for step in range(1,MAX_STEPS+1):
        eligible=[]
        for cid,(gap,conds,chosen) in cache.items():
            if cid in used:
                continue
            inc=unique_vs(chosen,current)
            pooled=metrics(inc,"incremental")
            by_year=[]
            for y in dev_years:
                yy=inc[inc["year"]==y].copy()
                by_year.append(metrics(yy,f"incremental_{y}"))
            tickets_each=[m["tickets"] for m in by_year]
            rois_each=[m["roi_pct"] if m["roi_pct"] is not None else -1.0 for m in by_year]
            years_ge90=sum(x>=90.0 for x in rois_each)

            ok=(
                len(inc)>=th["min_unique_dev_tickets"] and
                min(tickets_each)>=th["min_unique_each_dev_year"] and
                (pooled["roi_pct"] or 0.0)>=th["min_incremental_pooled_roi_pct"] and
                (pooled["roi_after_remove_largest_win_pct"] or 0.0)>=th["min_incremental_ex_max_roi_pct"] and
                years_ge90>=th["min_years_roi_ge_90"] and
                min(rois_each)>=th["min_worst_year_roi_pct"]
            )
            if not ok:
                continue
            row={
                "step":step,
                "candidate_id":cid,
                "gap_spec":gap,
                "conditions":";".join(f"{d}={v}" for d,v in conds) if conds else "NONE",
                "incremental_dev_tickets":pooled["tickets"],
                "incremental_dev_hits":pooled["hits"],
                "incremental_dev_roi_pct":pooled["roi_pct"],
                "incremental_dev_profit_yen":pooled["profit_yen"],
                "incremental_dev_ex_max_roi_pct":pooled["roi_after_remove_largest_win_pct"],
                "incremental_dev_worst_year_roi_pct":min(rois_each),
            }
            eligible.append((cid,gap,conds,chosen,row))

        if not eligible:
            break
        eligible.sort(
            key=lambda x:(
                x[4]["incremental_dev_profit_yen"],
                x[4]["incremental_dev_ex_max_roi_pct"],
                x[4]["incremental_dev_worst_year_roi_pct"],
                x[4]["incremental_dev_tickets"],
                x[0],
            ),
            reverse=True,
        )
        cid,gap,conds,chosen,row=eligible[0]
        current=union_frame(current,chosen)
        selected.append((cid,gap,conds,row))
        used.add(cid)

    return selected,th

def apply_selected_rules(pred,selected):
    tables=candidate_group_tables(pred)
    out=pred.iloc[0:0].copy()
    for cid,gap,conds,row in selected:
        chosen=chosen_from_table(tables,gap,conds)
        out=union_frame(out,chosen)
    return out

def machine_signal(year,frames,cols):
    train_years=[t for t in YEARS if t<year]
    train=pd.concat([frames[t] for t in train_years],ignore_index=True)
    test=frames[year].copy().reset_index(drop=True)
    raw=train_base(train,test,cols,98000+year)
    test["raw_model_probability"]=raw
    test=normalize_race(test,"raw_model_probability","p_model")
    test["machine_ratio_to_market"]=(
        test["p_model"]/test["market_q_norm"].clip(lower=1e-12)
    )
    test["machine_agree"]=test["machine_ratio_to_market"]>1.0
    return test,train_years

def mrow(g,label,year):
    m=metrics(g,label)
    m.update({
        "year":year,
        "label":label,
        "executed_races":int(g["race_id"].nunique()) if len(g) else 0,
        "mean_machine_ratio":float(g["machine_ratio_to_market"].mean()) if len(g) else None,
        "median_machine_ratio":float(g["machine_ratio_to_market"].median()) if len(g) else None,
        "machine_agree_share_pct":100.0*float(g["machine_agree"].mean()) if len(g) else None,
    })
    return m

def main():
    a=parse_args()
    paths={y:getattr(a,f"l17_{y}") for y in YEARS}
    l17={y:load_l17(paths[y],y) for y in YEARS}
    frames={y:add_market_features(build_year_frame(y,l17[y],a.backfill_root)) for y in YEARS}
    cols=market_feature_columns(frames[2022])

    human_predictions={}
    human_train_plan={}
    for y in PRED_YEARS:
        pred,tr=predict_human_year(y,frames,cols)
        human_predictions[y]=pred
        human_train_plan[y]=tr

    selected_rows=[]
    cohort_rows=[]
    fold_rows=[]

    for target in TEST_YEARS:
        dev_years=[y for y in PRED_YEARS if y<target]
        selected,th=select_human_rules(human_predictions,dev_years)

        ids=[x[0] for x in selected]
        if target==2025 and ids!=EXPECTED_2025_SEQUENCE:
            raise SystemExit(
                "2025 zero-base reproduction drift "
                f"got={ids} expected={EXPECTED_2025_SEQUENCE}"
            )

        human_pred=human_predictions[target].copy()
        human_sel=apply_selected_rules(human_pred,selected)

        machine,machine_train=machine_signal(target,frames,cols)
        machine_cols=[
            "year","race_id","pair_numbers","p_model",
            "machine_ratio_to_market","machine_agree"
        ]
        z=human_pred.merge(
            machine[machine_cols],
            on=["year","race_id","pair_numbers"],
            how="left",
            validate="one_to_one",
        )
        if z["machine_agree"].isna().any():
            raise SystemExit(f"machine/human universe merge drift target={target}")

        z["_key"]=key_series(z)
        human_keys=set(key_series(human_sel).tolist())
        z["human_rule"]=z["_key"].isin(human_keys)
        z["cohort"]=np.select(
            [
                z["human_rule"] & z["machine_agree"],
                z["human_rule"] & ~z["machine_agree"],
                ~z["human_rule"] & z["machine_agree"],
            ],
            ["BOTH","HUMAN_ONLY","MACHINE_ONLY"],
            default="NEITHER",
        )

        rows={}
        for label,mask in [
            ("HUMAN_ALL",z["human_rule"]),
            ("MACHINE_ALL",z["machine_agree"]),
            ("BOTH",z["cohort"]=="BOTH"),
            ("HUMAN_ONLY",z["cohort"]=="HUMAN_ONLY"),
            ("MACHINE_ONLY",z["cohort"]=="MACHINE_ONLY"),
        ]:
            row=mrow(z[mask].copy(),label,target)
            cohort_rows.append(row)
            rows[label]=row

        for step,(cid,gap,conds,row) in enumerate(selected,1):
            selected_rows.append({
                "test_year":target,
                "dev_years":"|".join(map(str,dev_years)),
                "step":step,
                "candidate_id":cid,
                "gap_spec":gap,
                "conditions":";".join(f"{d}={v}" for d,v in conds) if conds else "NONE",
                **{k:v for k,v in row.items() if k not in {"step","candidate_id","gap_spec","conditions"}},
            })

        fold_rows.append({
            "test_year":target,
            "human_rule_dev_years":"|".join(map(str,dev_years)),
            "human_rank_model_train_years":"|".join(map(str,human_train_plan[target])),
            "machine_train_years":"|".join(map(str,machine_train)),
            "selected_rule_count":len(selected),
            "min_unique_dev_tickets":th["min_unique_dev_tickets"],
            "min_years_roi_ge_90":th["min_years_roi_ge_90"],
            "human_tickets":rows["HUMAN_ALL"]["tickets"],
            "both_tickets":rows["BOTH"]["tickets"],
            "human_only_tickets":rows["HUMAN_ONLY"]["tickets"],
            "human_roi_pct":rows["HUMAN_ALL"]["roi_pct"],
            "both_roi_pct":rows["BOTH"]["roi_pct"],
            "human_only_roi_pct":rows["HUMAN_ONLY"]["roi_pct"],
            "both_minus_human_roi_points":(
                (rows["BOTH"]["roi_pct"] or 0.0)-(rows["HUMAN_ALL"]["roi_pct"] or 0.0)
            ),
            "both_minus_human_only_roi_points":(
                (rows["BOTH"]["roi_pct"] or 0.0)-(rows["HUMAN_ONLY"]["roi_pct"] or 0.0)
            ),
        })

    cdf=pd.DataFrame(cohort_rows)
    pooled_rows=[]
    for label in ("HUMAN_ALL","BOTH","HUMAN_ONLY","MACHINE_ONLY"):
        # Pooled metrics reconstructed from per-year totals to avoid ticket duplication
        x=cdf[cdf["label"]==label]
        tickets=int(x["tickets"].sum())
        hits=int(x["hits"].sum())
        stake=float(x["stake_yen"].sum())
        ret=float(x["return_yen"].sum())
        pooled_rows.append({
            "label":label,
            "years":"2023|2024|2025",
            "tickets":tickets,
            "hits":hits,
            "hit_rate_pct":100.0*hits/tickets if tickets else None,
            "stake_yen":stake,
            "return_yen":ret,
            "profit_yen":ret-stake,
            "roi_pct":100.0*ret/stake if stake else None,
        })

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(out/"walkforward-summary.csv",index=False)
    pd.DataFrame(selected_rows).to_csv(out/"selected-human-rules-by-test-year.csv",index=False)
    cdf.to_csv(out/"cohort-metrics-by-year.csv",index=False)
    pd.DataFrame(pooled_rows).to_csv(out/"pooled-cohort-metrics.csv",index=False)

    fdf=pd.DataFrame(fold_rows)
    summary={
        "contract":"L2_QUINELLA_HUMAN_MACHINE_WALKFORWARD_V1",
        "test_years":list(TEST_YEARS),
        "design":"human rule portfolio rediscovered only from prior OOF years; machine learned only from prior years; intersection tested on next year",
        "human_candidate_space":"same corrected-odds zero-base candidate space",
        "human_threshold_scaling":{
            "min_unique_dev_tickets":"40 * number_of_prior_dev_years",
            "min_unique_each_dev_year":MIN_UNIQUE_EACH_DEV_YEAR,
            "min_incremental_pooled_roi_pct":MIN_INCREMENTAL_POOLED_ROI,
            "min_incremental_ex_max_roi_pct":MIN_INCREMENTAL_EX_MAX_ROI,
            "min_years_roi_ge_90":"ceil(2/3 * number_of_prior_dev_years), minimum 1",
            "min_worst_year_roi_pct":MIN_WORST_YEAR_ROI,
            "max_steps":MAX_STEPS,
        },
        "machine_signal":"within-race normalized p_model > normalized market probability",
        "machine_ratio_threshold":1.0,
        "manual_machine_threshold_tuning":False,
        "2023_note":"low-confidence early fold because human rule discovery has only 2022 as prior OOF year",
        "2025_zero_base_sequence_reproduced":True,
        "year_results":fold_rows,
        "pooled_results":pooled_rows,
        "both_beats_human_all_years":int((fdf["both_minus_human_roi_points"]>0).sum()),
        "both_beats_human_only_years":int((fdf["both_minus_human_only_roi_points"]>0).sum()),
        "2026_locked":True,
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )
    print("L2_QUINELLA_HUMAN_MACHINE_WALKFORWARD_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
