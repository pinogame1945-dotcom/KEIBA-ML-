#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import build_year_frame, encode, evaluate_chosen
from run_l2_ticket_market_gap_v2 import add_market_features, market_feature_columns

YEARS=(2021,2022,2023,2024,2025)
OOF_YEARS=(2022,2023,2024,2025)
CALIBRATED_YEARS=(2023,2024,2025)
EPS=1e-12

def parse_args():
    p=argparse.ArgumentParser()
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def normalize_race(df,col,out):
    z=df.copy()
    denom=z.groupby(["year","race_id"])[col].transform("sum").clip(lower=EPS)
    z[out]=z[col]/denom
    return z

def train_base(train,test,cols,seed):
    xtr,xte=encode(train,test,cols)
    y=train["hit"].astype(int).to_numpy()
    model=lgb.LGBMClassifier(
        objective="binary",
        n_estimators=320,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
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
    model.fit(xtr,y)
    p=np.asarray(model.predict_proba(xte)[:,1],dtype=float)
    return p

def clipped_log(x):
    return np.log(np.clip(np.asarray(x,dtype=float),EPS,1.0))

def fit_blend_alpha(cal):
    y=cal["hit"].astype(int).to_numpy()
    pm=cal["p_model"].to_numpy(dtype=float)
    pq=cal["market_q_norm"].to_numpy(dtype=float)

    def objective(alpha):
        p=alpha*pm+(1.0-alpha)*pq
        return float(log_loss(y,np.clip(p,1e-9,1-1e-9),labels=[0,1]))

    res=minimize_scalar(objective,bounds=(0.0,1.0),method="bounded",options={"xatol":1e-5})
    return float(np.clip(res.x,0.0,1.0)),float(res.fun)

def fit_stack(cal):
    x=np.column_stack([
        clipped_log(cal["p_model"]),
        clipped_log(cal["market_q_norm"]),
    ])
    y=cal["hit"].astype(int).to_numpy()
    m=LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=500,
        n_jobs=1,
    )
    m.fit(x,y)
    return m

def apply_stack(model,df):
    x=np.column_stack([
        clipped_log(df["p_model"]),
        clipped_log(df["market_q_norm"]),
    ])
    raw=model.predict_proba(x)[:,1]
    z=df.copy()
    z["p_stack_raw"]=raw
    z=normalize_race(z,"p_stack_raw","p_stack")
    return z

def prob_metrics(df,col):
    y=df["hit"].astype(int).to_numpy()
    p=np.clip(df[col].to_numpy(dtype=float),1e-9,1-1e-9)
    return {
        "log_loss":float(log_loss(y,p,labels=[0,1])),
        "brier":float(brier_score_loss(y,p)),
        "roc_auc":float(roc_auc_score(y,p)),
    }

def betting_metrics(df,pcol,label):
    z=df.copy()
    z["edge"]=z[pcol]*z["odds"]-1.0
    chosen=z[z["edge"]>0.0].copy()
    source=int(z["race_id"].nunique())
    m=evaluate_chosen(chosen,source,label)
    m.update({
        "mean_edge":float(chosen["edge"].mean()) if len(chosen) else None,
        "median_edge":float(chosen["edge"].median()) if len(chosen) else None,
        "median_odds":float(chosen["odds"].median()) if len(chosen) else None,
        "mean_tickets_per_executed_race":(
            float(len(chosen)/m["executed_races"]) if m["executed_races"] else 0.0
        ),
    })
    return m

def main():
    a=parse_args()
    paths={y:getattr(a,f"l17_{y}") for y in YEARS}
    l17={y:load_l17(paths[y],y) for y in YEARS}
    frames={y:add_market_features(build_year_frame(y,l17[y],a.backfill_root)) for y in YEARS}
    cols=market_feature_columns(frames[2022])

    oof={}
    base_rows=[]
    for y in OOF_YEARS:
        train_years=[t for t in YEARS if t<y]
        train=pd.concat([frames[t] for t in train_years],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)
        raw=train_base(train,test,cols,96000+y)
        test["raw_model_probability"]=raw
        test=normalize_race(test,"raw_model_probability","p_model")
        oof[y]=test
        pm=prob_metrics(test,"p_model")
        pq=prob_metrics(test,"market_q_norm")
        base_rows.append({
            "year":y,
            "train_years":"|".join(map(str,train_years)),
            "train_pairs":len(train),
            "test_pairs":len(test),
            "source_races":int(test["race_id"].nunique()),
            "model_log_loss":pm["log_loss"],
            "model_brier":pm["brier"],
            "model_auc":pm["roc_auc"],
            "market_log_loss":pq["log_loss"],
            "market_brier":pq["brier"],
            "market_auc":pq["roc_auc"],
        })

    variant_rows=[]
    parameter_rows=[]
    all_eval=[]

    for y in CALIBRATED_YEARS:
        cal_years=[t for t in OOF_YEARS if t<y]
        cal=pd.concat([oof[t] for t in cal_years],ignore_index=True)
        test=oof[y].copy()

        alpha,alpha_loss=fit_blend_alpha(cal)
        test["p_blend"]=alpha*test["p_model"]+(1.0-alpha)*test["market_q_norm"]

        stack=fit_stack(cal)
        test=apply_stack(stack,test)

        parameter_rows.append({
            "test_year":y,
            "calibration_years":"|".join(map(str,cal_years)),
            "blend_alpha_model":alpha,
            "blend_alpha_market":1.0-alpha,
            "blend_calibration_log_loss":alpha_loss,
            "stack_intercept":float(stack.intercept_[0]),
            "stack_coef_log_model":float(stack.coef_[0][0]),
            "stack_coef_log_market":float(stack.coef_[0][1]),
        })

        for variant,pcol in [
            ("RAW_MODEL","p_model"),
            ("MARKET_ONLY","market_q_norm"),
            ("AUTO_BLEND","p_blend"),
            ("STACKED_CALIBRATION","p_stack"),
        ]:
            pm=prob_metrics(test,pcol)
            bm=betting_metrics(test,pcol,f"{variant}_{y}")
            row={
                "year":y,
                "variant":variant,
                **pm,
                **bm,
            }
            variant_rows.append(row)

        # retain all for pre-2025 variant selection
        all_eval.append(test.assign(eval_year=y))

    # Select calibration family using probability quality only on 2023-2024.
    pre=pd.concat([x for x in all_eval if int(x["year"].iloc[0]) in (2023,2024)],ignore_index=True)
    pre_scores=[]
    for variant,pcol in [
        ("RAW_MODEL","p_model"),
        ("MARKET_ONLY","market_q_norm"),
        ("AUTO_BLEND","p_blend"),
        ("STACKED_CALIBRATION","p_stack"),
    ]:
        pm=prob_metrics(pre,pcol)
        pre_scores.append({"variant":variant,**pm})
    pre_scores=sorted(pre_scores,key=lambda r:(r["log_loss"],r["brier"],-r["roc_auc"],r["variant"]))
    champion=pre_scores[0]["variant"]

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(base_rows).to_csv(out/"base-oof-metrics.csv",index=False)
    pd.DataFrame(parameter_rows).to_csv(out/"calibration-parameters.csv",index=False)
    pd.DataFrame(variant_rows).to_csv(out/"variant-metrics.csv",index=False)
    pd.DataFrame(pre_scores).to_csv(out/"pre2025-calibration-ranking.csv",index=False)

    vdf=pd.DataFrame(variant_rows)
    confirm=vdf[vdf["year"]==2025].copy()
    champion_confirm=confirm[confirm["variant"]==champion].iloc[0].to_dict()

    summary={
        "contract":"L2_QUINELLA_AUTO_CALIBRATION_V1",
        "source_variant":"ODDS_COMMA_CORRECTED",
        "manual_bins_used":False,
        "legacy_laws_seeded":False,
        "manual_rule_grid_used":False,
        "ticket_cap_used":False,
        "buy_rule":"calibrated_probability * final_odds > 1.0",
        "buy_rule_reason":"mathematical break-even only",
        "calibration_methods":[
            "RAW_MODEL",
            "MARKET_ONLY",
            "AUTO_BLEND",
            "STACKED_CALIBRATION",
        ],
        "blend_alpha_learned":True,
        "stack_coefficients_learned":True,
        "calibration_is_strictly_prior_oof":True,
        "pre2025_variant_selection_basis":"minimum probability log-loss on 2023-2024; ROI not used",
        "pre2025_champion":champion,
        "pre2025_probability_ranking":pre_scores,
        "confirmation_2025":confirm.to_dict(orient="records"),
        "champion_confirmation_2025":champion_confirm,
        "feature_count":len(cols),
        "2025_used_for_variant_selection":False,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )
    print("L2_QUINELLA_AUTO_CALIBRATION_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
