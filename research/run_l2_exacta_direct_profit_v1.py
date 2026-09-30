#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_exacta_decomposed_v1 import (
    YEARS,
    TEST_YEARS,
    build_pair_year_frame,
)
from run_l2_exacta_v0 import encode

STAKE=100.0
POWERS=(1.2,1.5,1.8)

OUTCOME_OR_ID={
    "year","race_id","race_date","pair_horse_ids","pair_numbers",
    "a_horse_id","b_horse_id","a_horse_number","b_horse_number",
    "pair_hit","hit_a_to_b","hit_b_to_a","return_a_to_b","return_b_to_a",
    "direction_label","direction_eligible","strict_direction_eligible",
    "multi_direction_hit","odds_a_to_b","odds_b_to_a",
    "market_q_a_to_b","market_q_b_to_a",
    "dir_l17_prob","dir_market_aware_prob",
}
FORBIDDEN_FEATURE_TOKENS=(
    "hit","return","payout","profit","finish","result","target","winner",
)


def parse_args():
    p=argparse.ArgumentParser(
        description="EXACTA Direct Profit V1: predict ticket monetary return directly, not hit probability."
    )
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--auto-baseline",required=True)
    p.add_argument("--learned-baseline",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1)
        out[int(y)]=p
    return out


def expand_ordered_tickets(frame):
    rows=[]
    safe_cols=[
        c for c in frame.columns
        if c not in OUTCOME_OR_ID
        and not any(tok in c.lower() for tok in FORBIDDEN_FEATURE_TOKENS)
    ]
    for _,r in frame.iterrows():
        for orientation in (1,0):
            if orientation==1:
                odds=float(r["odds_a_to_b"])
                reverse_odds=float(r["odds_b_to_a"])
                market_q=float(r["market_q_a_to_b"])
                hit=int(r["hit_a_to_b"])
                ret=float(r["return_a_to_b"])
                sign=1.0
                ticket=f'{int(r["a_horse_number"])}>{int(r["b_horse_number"])}'
            else:
                odds=float(r["odds_b_to_a"])
                reverse_odds=float(r["odds_a_to_b"])
                market_q=float(r["market_q_b_to_a"])
                hit=int(r["hit_b_to_a"])
                ret=float(r["return_b_to_a"])
                sign=-1.0
                ticket=f'{int(r["b_horse_number"])}>{int(r["a_horse_number"])}'

            row={
                "year":int(r["year"]),
                "race_id":str(r["race_id"]),
                "race_date":str(r["race_date"]),
                "ticket":ticket,
                "hit":hit,
                "return_yen":ret,
                "realized_profit_yen":ret-STAKE,
                "odds":odds,
                "log_odds":math.log(max(odds,1e-12)),
                "market_implied_probability":1.0/max(odds,1e-12),
                "market_q_orientation":market_q,
                "market_log_q_orientation":math.log(max(market_q,1e-12)),
                "market_share_within_pair":market_q/max(float(r["pair_market_q"]),1e-12),
                "log_odds_ratio_to_reverse":math.log(max(odds,1e-12))-math.log(max(reverse_odds,1e-12)),
            }
            for c in safe_cols:
                v=r[c]
                if c.startswith("dir_"):
                    row[c]=sign*float(v)
                else:
                    row[c]=v
            rows.append(row)

    out=pd.DataFrame(rows)
    out["market_rank_orientation"]=out.groupby(
        ["year","race_id"]
    )["market_q_orientation"].rank(method="min",ascending=False).astype(int)
    return out


def feature_columns(df):
    exclude={
        "year","race_id","race_date","ticket","hit","return_yen","realized_profit_yen",
    }
    cols=[c for c in df.columns if c not in exclude]
    bad=[c for c in cols if any(tok in c.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise SystemExit(f"outcome-derived features leaked: {bad}")
    return cols


def make_model(power,seed):
    return lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=float(power),
        n_estimators=320,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=6.0,
        reg_alpha=1.0,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def train_predict(train,test,cols,power,seed):
    xtr,xte=encode(train,test,cols)
    y=pd.to_numeric(train["return_yen"],errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    model=make_model(power,seed)
    model.fit(xtr,y)
    pred=np.asarray(model.predict(xte),dtype=float)
    pred=np.clip(pred,0.0,None)
    imp=pd.DataFrame({
        "feature":cols,
        "importance_split":model.feature_importances_,
        "importance_gain":model.booster_.feature_importance(importance_type="gain"),
    })
    return pred,imp


def metric_row(df,label,year):
    selected=df[df["predicted_profit_yen"]>0.0].copy()
    source_races=int(df["race_id"].nunique())
    executed=int(selected["race_id"].nunique())
    tickets=len(selected)
    stake=STAKE*tickets
    ret=float(selected["return_yen"].sum()) if tickets else 0.0
    hits=int(selected["hit"].sum()) if tickets else 0
    hit_races=int(selected.loc[selected["hit"]==1,"race_id"].nunique()) if tickets else 0
    return {
        "label":label,
        "year":year,
        "source_races":source_races,
        "executed_races":executed,
        "execution_coverage_pct":100.0*executed/source_races if source_races else 0.0,
        "tickets":tickets,
        "tickets_per_source_race":tickets/source_races if source_races else 0.0,
        "tickets_per_executed_race":tickets/executed if executed else 0.0,
        "hit_tickets":hits,
        "hit_races":hit_races,
        "race_hit_rate_pct_all":100.0*hit_races/source_races if source_races else 0.0,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else 0.0,
        "mean_predicted_return_yen":float(selected["predicted_return_yen"].mean()) if tickets else None,
        "median_predicted_return_yen":float(selected["predicted_return_yen"].median()) if tickets else None,
        "mean_predicted_profit_yen":float(selected["predicted_profit_yen"].mean()) if tickets else None,
        "median_predicted_profit_yen":float(selected["predicted_profit_yen"].median()) if tickets else None,
        "median_odds":float(selected["odds"].median()) if tickets else None,
        "mean_odds":float(selected["odds"].mean()) if tickets else None,
    }


def tail_audit(df,label):
    selected=df[df["predicted_profit_yen"]>0.0].copy()
    if selected.empty:
        return []
    stake=STAKE*len(selected)
    wins=selected[selected["return_yen"]>0].sort_values("return_yen",ascending=False)
    out=[]
    for n in (0,1,3,10,25):
        removed=float(wins.head(n)["return_yen"].sum()) if n else 0.0
        ret=float(selected["return_yen"].sum())-removed
        out.append({
            "label":label,
            "removed_top_winning_tickets":n,
            "selected_tickets":len(selected),
            "stake_yen":stake,
            "return_yen_after_removal":ret,
            "profit_yen_after_removal":ret-stake,
            "roi_pct_after_removal":100.0*ret/stake if stake else 0.0,
            "removed_return_yen":removed,
        })
    return out


def decile_audit(df,label):
    z=df.copy()
    try:
        z["predicted_profit_decile"]=pd.qcut(
            z["predicted_profit_yen"].rank(method="first"),
            10,
            labels=False,
        )+1
    except ValueError:
        z["predicted_profit_decile"]=1
    rows=[]
    for dec,g in z.groupby("predicted_profit_decile",sort=True):
        stake=STAKE*len(g)
        ret=float(g["return_yen"].sum())
        rows.append({
            "label":label,
            "predicted_profit_decile":int(dec),
            "tickets":len(g),
            "mean_predicted_profit_yen":float(g["predicted_profit_yen"].mean()),
            "median_predicted_profit_yen":float(g["predicted_profit_yen"].median()),
            "mean_odds":float(g["odds"].mean()),
            "hit_rate_pct":100.0*float(g["hit"].mean()),
            "actual_mean_return_yen":ret/len(g) if len(g) else 0.0,
            "actual_mean_profit_yen":ret/len(g)-STAKE if len(g) else 0.0,
            "roi_pct":100.0*ret/stake if stake else 0.0,
        })
    return rows


def main():
    a=parse_args()
    started=time.perf_counter()
    lp=parse_paths(a.l17_year)
    if set(lp)!=set(YEARS) or 2026 in lp:
        raise SystemExit("L1.7 years must be exactly 2022-2025; 2026 sealed")

    pair_frames={
        y:build_pair_year_frame(y,load_l17(lp[y],y),a.backfill_root)
        for y in YEARS
    }
    frames={y:expand_ordered_tickets(pair_frames[y]) for y in YEARS}
    cols=feature_columns(frames[2022])

    fold_rows=[]
    metric_rows=[]
    importance=[]
    predictions={p:[] for p in POWERS}

    for y in TEST_YEARS:
        train=pd.concat([frames[t] for t in YEARS if t<y],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)
        for idx,power in enumerate(POWERS):
            label=f"DIRECT_TWEEDIE_{str(power).replace('.','_')}"
            t0=time.perf_counter()
            pred,imp=train_predict(train,test,cols,power,111000+y*10+idx)
            scored=test.copy()
            scored["predicted_return_yen"]=pred
            scored["predicted_profit_yen"]=pred-STAKE
            predictions[power].append(scored)

            metric_rows.append(metric_row(scored,label,y))
            fold_rows.append({
                "test_year":y,
                "label":label,
                "train_years":"|".join(str(t) for t in YEARS if t<y),
                "train_tickets":len(train),
                "test_tickets":len(test),
                "feature_count":len(cols),
                "train_positive_return_tickets":int((train["return_yen"]>0).sum()),
                "train_mean_return_yen":float(train["return_yen"].mean()),
                "test_mean_return_yen":float(test["return_yen"].mean()),
                "elapsed_seconds":time.perf_counter()-t0,
            })
            imp["test_year"]=y
            imp["label"]=label
            importance.append(imp)

    aggregate=[]
    tails=[]
    deciles=[]
    for power in POWERS:
        label=f"DIRECT_TWEEDIE_{str(power).replace('.','_')}"
        z=pd.concat(predictions[power],ignore_index=True)
        aggregate.append(metric_row(z,label,"ALL"))
        tails.extend(tail_audit(z,label))
        deciles.extend(decile_audit(z,label))

    auto=pd.read_csv(a.auto_baseline)
    auto=auto[auto["variant"]=="AUTO_MARKET_AWARE_EV"]
    if len(auto)!=1:
        raise SystemExit("AUTO baseline missing")
    learned=json.loads(Path(a.learned_baseline).read_text(encoding="utf-8"))
    learned_overall=learned["overall_2023_2025"]

    comparison=[
        {
            "model":"AUTO_V0_MARKET_AWARE",
            "source_races":int(auto.iloc[0]["source_races"]),
            "executed_races":int(auto.iloc[0]["bet_races"]),
            "tickets":int(auto.iloc[0]["selected_tickets"]),
            "tickets_per_source_race":float(auto.iloc[0]["tickets_per_source_race"]),
            "hit_races":int(auto.iloc[0]["hit_races"]),
            "profit_yen":float(auto.iloc[0]["profit_yen_equal100"]),
            "roi_pct":float(auto.iloc[0]["roi_pct_equal100"]),
        },
        {
            "model":"LEARNED_BOUNDARY_V1",
            "source_races":int(learned_overall["source_races"]),
            "executed_races":int(learned_overall["executed_races"]),
            "tickets":int(learned_overall["tickets"]),
            "tickets_per_source_race":float(learned_overall["tickets_per_source_race"]),
            "hit_races":int(learned_overall["hit_races"]),
            "profit_yen":float(learned_overall["profit_yen"]),
            "roi_pct":float(learned_overall["roi_pct"]),
        },
    ]
    for row in aggregate:
        comparison.append({
            "model":row["label"],
            "source_races":row["source_races"],
            "executed_races":row["executed_races"],
            "tickets":row["tickets"],
            "tickets_per_source_race":row["tickets_per_source_race"],
            "hit_races":row["hit_races"],
            "profit_yen":row["profit_yen"],
            "roi_pct":row["roi_pct"],
        })

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out/"metrics-by-year.csv",index=False)
    pd.DataFrame(aggregate).to_csv(out/"aggregate.csv",index=False)
    pd.DataFrame(comparison).to_csv(out/"baseline-comparison.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(out/"folds.csv",index=False)
    pd.concat(importance,ignore_index=True).to_csv(out/"feature-importance.csv",index=False)
    pd.DataFrame(tails).to_csv(out/"tail-audit.csv",index=False)
    pd.DataFrame(deciles).to_csv(out/"predicted-profit-deciles.csv",index=False)

    summary={
        "contract":"L2_EXACTA_DIRECT_PROFIT_V1",
        "bet_type":"EXACTA",
        "objective":"predict realized monetary return per 100-yen ticket directly",
        "target":"return_yen_per100; predicted_profit = predicted_return - 100",
        "models":[
            {"label":f"DIRECT_TWEEDIE_{str(p).replace('.','_')}","objective":"tweedie","variance_power":p}
            for p in POWERS
        ],
        "selection_rule":"predicted_return_yen > 100",
        "selection_rule_reason":"mathematical positive predicted profit only",
        "manual_topk":False,
        "manual_confidence_floor":False,
        "manual_odds_band":False,
        "manual_ticket_cap":False,
        "manual_skip_rate":False,
        "probability_model_used_for_selection":False,
        "final_odds_feature_used":True,
        "final_odds_caveat":"historical final odds are an execution-price proxy, not live timestamp odds",
        "walk_forward":{"test_years":list(TEST_YEARS),"training":"all prior years only"},
        "tail_robustness":"reported by zeroing top 1/3/10/25 selected winning payouts while retaining stakes",
        "2026_locked":True,
        "script_total_seconds":time.perf_counter()-started,
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    (out/"README.md").write_text(
        "# L2 EXACTA Direct Profit V1\n\n"
        "This experiment stops estimating hit probability for ticket selection. "
        "Each ordered exacta ticket is trained directly against its realized monetary return per 100 yen. "
        "LightGBM Tweedie regressors model the zero-heavy, positive-skew return target. "
        "A ticket is bought only when predicted monetary return exceeds the 100-yen stake. "
        "There is no Top-K, confidence floor, odds band, ticket cap, forced skip rate, or legacy LAW. "
        "Three Tweedie variance powers are reported side-by-side; none is selected using the test year. "
        "Tail-removal diagnostics test jackpot dependence. Final odds are historical execution-price proxies. "
        "2026 remains sealed.\n",
        encoding="utf-8",
    )

    print("L2_EXACTA_DIRECT_PROFIT_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
    print("===== BASELINE COMPARISON =====")
    print(pd.DataFrame(comparison).to_csv(index=False))
    print("===== AGGREGATE =====")
    print(pd.DataFrame(aggregate).to_csv(index=False))
    print("===== TAIL AUDIT =====")
    print(pd.DataFrame(tails).to_csv(index=False))


if __name__=="__main__":
    main()
