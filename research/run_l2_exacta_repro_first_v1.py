#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_exacta_decomposed_v1 import YEARS, TEST_YEARS, build_pair_year_frame
from run_l2_exacta_direct_profit_v1 import (
    STAKE,
    expand_ordered_tickets,
    feature_columns,
    train_predict,
    metric_row,
    tail_audit,
)

POWER=1.5


def parse_args():
    p=argparse.ArgumentParser(
        description="EXACTA Repro-First V1: require independent prior-period profit models to agree."
    )
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--direct-profit-baseline",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1)
        out[int(y)]=p
    return out


def split_2022_halves(df):
    dates=sorted(df["race_date"].astype(str).unique())
    if len(dates)<20:
        raise SystemExit("too few 2022 dates to make independent halves")
    cut=len(dates)//2
    left=set(dates[:cut])
    right=set(dates[cut:])
    a=df[df["race_date"].astype(str).isin(left)].copy()
    b=df[df["race_date"].astype(str).isin(right)].copy()
    if a.empty or b.empty:
        raise SystemExit("2022 half split empty")
    return [
        ("2022_H1",a),
        ("2022_H2",b),
    ]


def independent_periods(frames,test_year):
    if test_year==2023:
        return split_2022_halves(frames[2022])
    return [(str(y),frames[y].copy()) for y in YEARS if y<test_year]


def attach_prediction(test,pred,label):
    out=test.copy()
    out[f"pred_return_{label}"]=np.asarray(pred,dtype=float)
    return out


def evaluate_with_prediction(test,pred,label,year):
    z=test.copy()
    z["predicted_return_yen"]=np.asarray(pred,dtype=float)
    z["predicted_profit_yen"]=z["predicted_return_yen"]-STAKE
    return metric_row(z,label,year),z


def stability_summary(metrics):
    rows=[]
    m=pd.DataFrame(metrics)
    for label,g in m.groupby("label",sort=False):
        gy=g[g["year"].isin(TEST_YEARS)].copy()
        if gy.empty:
            continue
        rois=gy["roi_pct"].astype(float).tolist()
        profits=gy["profit_yen"].astype(float).tolist()
        rows.append({
            "label":label,
            "years":"2023|2024|2025",
            "mean_roi_pct":float(np.mean(rois)),
            "median_roi_pct":float(np.median(rois)),
            "worst_year_roi_pct":float(np.min(rois)),
            "best_year_roi_pct":float(np.max(rois)),
            "roi_std_pct":float(np.std(rois,ddof=0)),
            "profitable_years":int(sum(x>100.0 for x in rois)),
            "nonlosing_years":int(sum(x>=100.0 for x in rois)),
            "total_profit_yen":float(np.sum(profits)),
            "all_years_same_positive_direction":bool(all(x>100.0 for x in rois)),
        })
    return rows


def main():
    a=parse_args()
    started=time.perf_counter()
    lp=parse_paths(a.l17_year)
    if set(lp)!=set(YEARS) or 2026 in lp:
        raise SystemExit("L1.7 years must be exactly 2022-2025; 2026 sealed")

    frames={}
    for y in YEARS:
        t0=time.perf_counter()
        pair=build_pair_year_frame(y,load_l17(lp[y],y),a.backfill_root)
        t1=time.perf_counter()
        frames[y]=expand_ordered_tickets(pair)
        print("REPRO_YEAR_READY "+json.dumps({
            "year":y,
            "tickets":len(frames[y]),
            "pair_build_seconds":round(t1-t0,3),
            "ordered_expand_seconds":round(time.perf_counter()-t1,3),
        },separators=(",",":")),flush=True)
        del pair

    cols=feature_columns(frames[2022])
    metrics=[]
    folds=[]
    tails=[]
    agreement_rows=[]

    for test_year in TEST_YEARS:
        test=frames[test_year].copy().reset_index(drop=True)

        # Control: one pooled model over all prior years.
        pooled_train=pd.concat(
            [frames[y] for y in YEARS if y<test_year],
            ignore_index=True,
        )
        t0=time.perf_counter()
        pooled_pred,pooled_imp=train_predict(
            pooled_train,test,cols,POWER,120000+test_year
        )
        row,scored=evaluate_with_prediction(
            test,pooled_pred,"POOLED_PRIOR_TWEEDIE_1_5",test_year
        )
        metrics.append(row)
        tails.extend(
            {**x,"year":test_year}
            for x in tail_audit(scored,"POOLED_PRIOR_TWEEDIE_1_5")
        )
        folds.append({
            "test_year":test_year,
            "lane":"POOLED_PRIOR_TWEEDIE_1_5",
            "training_periods":"|".join(str(y) for y in YEARS if y<test_year),
            "models":1,
            "elapsed_seconds":time.perf_counter()-t0,
        })

        # Repro-first: independent period models must all predict >100.
        periods=independent_periods(frames,test_year)
        period_preds=[]
        period_labels=[]
        lane_start=time.perf_counter()
        for idx,(period_label,train_period) in enumerate(periods):
            pred,_=train_predict(
                train_period,test,cols,POWER,121000+test_year*10+idx
            )
            period_preds.append(np.asarray(pred,dtype=float))
            period_labels.append(period_label)

        pred_matrix=np.vstack(period_preds)
        min_pred=pred_matrix.min(axis=0)
        mean_pred=pred_matrix.mean(axis=0)
        positive_votes=(pred_matrix>STAKE).sum(axis=0)
        unanimous=(positive_votes==len(periods))

        # Strict reproducibility gate: worst independent prior-period prediction
        # itself must exceed break-even. No average can rescue a negative period.
        strict_pred=np.where(unanimous,min_pred,0.0)
        row,strict_scored=evaluate_with_prediction(
            test,strict_pred,"REPRO_ALL_PERIODS_TWEEDIE_1_5",test_year
        )
        metrics.append(row)
        tails.extend(
            {**x,"year":test_year}
            for x in tail_audit(strict_scored,"REPRO_ALL_PERIODS_TWEEDIE_1_5")
        )
        folds.append({
            "test_year":test_year,
            "lane":"REPRO_ALL_PERIODS_TWEEDIE_1_5",
            "training_periods":"|".join(period_labels),
            "models":len(periods),
            "elapsed_seconds":time.perf_counter()-lane_start,
        })

        selected=strict_scored[strict_scored["predicted_profit_yen"]>0].copy()
        agreement_rows.append({
            "test_year":test_year,
            "periods":"|".join(period_labels),
            "period_model_count":len(periods),
            "all_tickets":len(test),
            "unanimous_positive_tickets":int(unanimous.sum()),
            "unanimous_positive_pct":100.0*float(unanimous.mean()),
            "mean_positive_votes":float(positive_votes.mean()),
            "selected_mean_worst_pred_return":float(selected["predicted_return_yen"].mean()) if len(selected) else None,
            "selected_median_worst_pred_return":float(selected["predicted_return_yen"].median()) if len(selected) else None,
            "selected_mean_period_pred_return":float(mean_pred[unanimous].mean()) if unanimous.any() else None,
        })

    # Add frozen previous Direct Profit 1.5 as a context baseline only.
    prev=pd.read_csv(a.direct_profit_baseline)
    prev=prev[prev["label"]=="DIRECT_TWEEDIE_1_5"].copy()
    if len(prev)!=1:
        raise SystemExit("frozen DIRECT_TWEEDIE_1_5 baseline missing")

    stability=stability_summary(metrics)
    stability_df=pd.DataFrame(stability)

    by_year=pd.DataFrame(metrics)
    prior_context={
        "label":"FROZEN_PREVIOUS_DIRECT_TWEEDIE_1_5",
        "years":"2023|2024|2025",
        "mean_roi_pct":None,
        "median_roi_pct":None,
        "worst_year_roi_pct":None,
        "best_year_roi_pct":None,
        "roi_std_pct":None,
        "profitable_years":None,
        "nonlosing_years":None,
        "total_profit_yen":float(prev.iloc[0]["profit_yen"]),
        "all_years_same_positive_direction":False,
        "aggregate_roi_pct":float(prev.iloc[0]["roi_pct"]),
        "aggregate_tickets":int(prev.iloc[0]["tickets"]),
    }

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    by_year.to_csv(out/"metrics-by-year.csv",index=False)
    stability_df.to_csv(out/"reproducibility-summary.csv",index=False)
    pd.DataFrame(folds).to_csv(out/"folds.csv",index=False)
    pd.DataFrame(tails).to_csv(out/"tail-audit.csv",index=False)
    pd.DataFrame(agreement_rows).to_csv(out/"agreement-audit.csv",index=False)
    (out/"previous-baseline-context.json").write_text(
        json.dumps(prior_context,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )

    summary={
        "contract":"L2_EXACTA_REPRO_FIRST_V1",
        "bet_type":"EXACTA",
        "objective":"prefer cross-period reproducibility over average ROI",
        "base_model":"LightGBM Tweedie return regression",
        "tweedie_variance_power":POWER,
        "power_choice_reason":"fixed midpoint of the pre-existing 1.2/1.5/1.8 candidate set; no new test-year tuning in this lane",
        "control":"one model trained on all prior years",
        "repro_lane":"independent prior-period models; every model must predict return > 100 yen",
        "repro_score":"minimum predicted return across independent prior-period models",
        "2023_independent_periods":"chronological halves of 2022",
        "2024_independent_periods":["2022","2023"],
        "2025_independent_periods":["2022","2023","2024"],
        "primary_evaluation":[
            "ROI by individual test year",
            "worst_year_roi",
            "profitable_year_count",
            "ROI standard deviation",
            "tail-removal robustness",
        ],
        "average_roi_is_not_primary":True,
        "manual_topk":False,
        "manual_odds_band":False,
        "manual_ticket_cap":False,
        "2026_locked":True,
        "script_total_seconds":time.perf_counter()-started,
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )
    (out/"README.md").write_text(
        "# L2 EXACTA Repro-First V1\n\n"
        "This lane treats year-to-year reproducibility as the primary research objective. "
        "For each unseen test year, separate prior-period Tweedie return models are trained independently. "
        "A ticket passes only when every independent model predicts a return above the 100-yen stake; "
        "the ticket score is the minimum prediction across those models. For 2023, 2022 is split chronologically "
        "into two independent halves. For 2024 the periods are 2022 and 2023; for 2025 they are 2022, 2023, and 2024. "
        "A pooled-prior model with the same Tweedie 1.5 configuration is the control. "
        "The primary readout is per-year ROI, worst-year ROI, profitable-year count, dispersion, and tail robustness, "
        "not aggregate ROI. 2026 remains sealed.\n",
        encoding="utf-8",
    )

    print("L2_EXACTA_REPRO_FIRST_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
    print("===== YEARLY =====")
    print(by_year.to_csv(index=False))
    print("===== REPRODUCIBILITY =====")
    print(stability_df.to_csv(index=False))
    print("===== AGREEMENT =====")
    print(pd.DataFrame(agreement_rows).to_csv(index=False))


if __name__=="__main__":
    main()
