#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss, roc_auc_score

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_market_gap_trio_v0 import (
    add_market_features,
    build_year_frame,
    encode,
    feature_columns,
)

YEARS=(2021,2022,2023,2024,2025)
OOF_YEARS=(2022,2023,2024,2025)
CALIBRATED_YEARS=(2023,2024,2025)
EPS=1e-12
MARKET_FEATURES={"market_q_norm","market_log_q","market_rank","market_inv_odds"}


def parse_args():
    p=argparse.ArgumentParser(
        description="TRIO Market Residual V1: market probability baseline + L1.7-only learned correction."
    )
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--work-dir",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def residual_feature_columns(df):
    cols=[c for c in feature_columns(df) if c not in MARKET_FEATURES]
    bad=[c for c in cols if "market" in c.lower() or c=="odds"]
    if bad:
        raise SystemExit(f"market leakage into residual features: {sorted(bad)}")
    return cols


def add_outcome_mass_and_residual(df):
    z=df.copy()
    positives=z.groupby(["year","race_id"])["hit"].transform("sum").clip(lower=1)
    z["outcome_mass"]=(z["hit"].astype(float)/positives.astype(float)).astype("float32")
    z["market_residual_target"]=(z["outcome_mass"]-z["market_q_norm"]).astype("float32")
    race_err=z.groupby(["year","race_id"])["market_residual_target"].sum().abs().max()
    if float(race_err)>1e-5:
        raise SystemExit(f"residual target race-sum drift={race_err}")
    return z


def train_residual_model(train,test,cols,seed):
    tr=add_outcome_mass_and_residual(train)
    xtr,xte=encode(tr,test,cols)
    y=tr["market_residual_target"].astype("float32").to_numpy()
    model=lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=260,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.90,
        colsample_bytree=0.88,
        reg_lambda=6.0,
        reg_alpha=0.5,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(xtr,y)
    raw=np.asarray(model.predict(xte),dtype=float)
    sd=float(np.std(raw))
    if not math.isfinite(sd) or sd<=1e-12:
        raise SystemExit("residual model score variance collapsed")
    score=(raw-float(np.mean(raw)))/sd
    imp=pd.DataFrame({"feature":cols,"gain":model.feature_importances_})
    return score.astype("float32"),imp


def apply_tilt(df,alpha,out_col):
    z=df.copy()
    q=np.clip(z["market_q_norm"].to_numpy(dtype=float),EPS,1.0)
    s=z["residual_score"].to_numpy(dtype=float)
    logw=np.log(q)+float(alpha)*s
    z["_logw"]=logw
    maxlog=z.groupby(["year","race_id"])["_logw"].transform("max").to_numpy(dtype=float)
    w=np.exp(logw-maxlog)
    z["_w"]=w
    denom=z.groupby(["year","race_id"])["_w"].transform("sum").to_numpy(dtype=float)
    z[out_col]=(w/np.clip(denom,EPS,None)).astype("float32")
    return z.drop(columns=["_logw","_w"])


def winner_log_loss(df,col):
    pos=df[df["hit"]==1]
    if pos.empty:
        return None
    p=np.clip(pos[col].to_numpy(dtype=float),1e-12,1.0)
    return float(-np.log(p).mean())


def probability_metrics(df,col):
    y=df["hit"].astype(int).to_numpy()
    p=np.clip(df[col].to_numpy(dtype=float),1e-12,1.0-1e-12)
    sums=df.groupby(["year","race_id"])[col].sum()
    return {
        "winner_log_loss":winner_log_loss(df,col),
        "binary_brier":float(brier_score_loss(y,p)),
        "roc_auc":float(roc_auc_score(y,p)),
        "mean_race_probability_sum":float(sums.mean()),
        "max_abs_race_probability_sum_error":float((sums-1.0).abs().max()),
    }


def alpha_objective(frame,alpha):
    z=apply_tilt(frame,alpha,"p_tmp")
    return winner_log_loss(z,"p_tmp")


def fit_alpha(cal):
    res=minimize_scalar(
        lambda a: alpha_objective(cal,float(a)),
        bounds=(-20.0,20.0),
        method="bounded",
        options={"xatol":1e-4},
    )
    alpha=float(res.x)
    return alpha,float(res.fun),bool(abs(alpha)>=19.9)


def betting_metrics(df,pcol,label):
    z=df.copy()
    z["edge"]=z[pcol]*z["odds"]-1.0
    chosen=z[z["edge"]>0.0].copy()

    n=len(chosen)
    stake=100.0*n
    ret=float(chosen["return_yen_per100"].sum()) if n else 0.0
    hits=int(chosen["hit"].sum()) if n else 0
    wins=sorted(
        [float(x) for x in chosen.loc[chosen["return_yen_per100"]>0,"return_yen_per100"]],
        reverse=True,
    )
    def roi_ex(k):
        return 100.0*(ret-sum(wins[:k]))/stake if stake else None

    executed=int(chosen["race_id"].nunique()) if n else 0
    return {
        "label":label,
        "source_races":int(z["race_id"].nunique()),
        "tickets":int(n),
        "executed_races":executed,
        "mean_tickets_per_executed_race":float(n/executed) if executed else 0.0,
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
        "mean_edge":float(chosen["edge"].mean()) if n else None,
        "median_edge":float(chosen["edge"].median()) if n else None,
        "median_odds":float(chosen["odds"].median()) if n else None,
        "mean_odds":float(chosen["odds"].mean()) if n else None,
    }


def winner_score_diagnostics(df,year):
    pos=df[df["hit"]==1]["residual_score"]
    all_scores=df["residual_score"]
    return {
        "year":year,
        "winning_tickets":int(len(pos)),
        "winner_mean_residual_score":float(pos.mean()) if len(pos) else None,
        "winner_median_residual_score":float(pos.median()) if len(pos) else None,
        "winner_positive_score_pct":100.0*float((pos>0).mean()) if len(pos) else None,
        "all_mean_residual_score":float(all_scores.mean()),
        "all_std_residual_score":float(all_scores.std()),
    }


def main():
    a=parse_args()
    if 2026 in YEARS:
        raise SystemExit("2026 sealed")

    work=Path(a.work_dir)
    out=Path(a.out_dir)
    work.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)

    l17_paths={y:getattr(a,f"l17_{y}") for y in YEARS}
    l17={y:load_l17(l17_paths[y],y) for y in YEARS}

    base_paths={}
    coverage=[]
    for y in YEARS:
        frame,meta=build_year_frame(y,l17[y],a.backfill_root)
        frame=add_market_features(frame)
        p=work/f"trio-base-{y}.pkl.gz"
        frame.to_pickle(p,compression="gzip")
        base_paths[y]=p
        coverage.append(meta)
        print(f"TRIO_RESIDUAL_BASE_READY year={y} rows={len(frame)}",flush=True)
        del frame

    oof_paths={}
    fold_rows=[]
    importances=[]
    score_diag=[]
    for y in OOF_YEARS:
        train_years=[2021] if y==2022 else [t for t in (2022,2023,2024) if t<y]
        train=pd.concat(
            [pd.read_pickle(base_paths[t],compression="gzip") for t in train_years],
            ignore_index=True,
        )
        test=pd.read_pickle(base_paths[y],compression="gzip").reset_index(drop=True)
        cols=residual_feature_columns(train)
        score,imp=train_residual_model(train,test,cols,98000+y)
        test["residual_score"]=score

        keep=[
            "year","race_id","race_date","trio_numbers","hit","return_yen_per100",
            "odds","market_q_norm","residual_score",
        ]
        oof=test[keep].copy()
        p=work/f"trio-residual-oof-{y}.pkl.gz"
        oof.to_pickle(p,compression="gzip")
        oof_paths[y]=p

        market_m=probability_metrics(oof,"market_q_norm")
        score_diag.append(winner_score_diagnostics(oof,y))
        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(map(str,train_years)),
            "train_trios":int(len(train)),
            "test_trios":int(len(test)),
            "source_races":int(test["race_id"].nunique()),
            "positive_trios":int(test["hit"].sum()),
            "feature_count":int(len(cols)),
            "market_winner_log_loss":market_m["winner_log_loss"],
            "market_brier":market_m["binary_brier"],
            "market_auc":market_m["roc_auc"],
        })
        imp["test_year"]=y
        importances.append(imp)
        print(f"TRIO_RESIDUAL_OOF_READY year={y} rows={len(oof)}",flush=True)
        del train,test,oof

    oof={y:pd.read_pickle(oof_paths[y],compression="gzip") for y in OOF_YEARS}

    parameter_rows=[]
    variant_rows=[]
    calibrated={}
    for y in CALIBRATED_YEARS:
        cal_years=[t for t in OOF_YEARS if t<y]
        cal=pd.concat([oof[t] for t in cal_years],ignore_index=True)
        alpha,cal_loss,boundary_hit=fit_alpha(cal)

        test=apply_tilt(oof[y],alpha,"p_residual")
        unit=apply_tilt(oof[y],1.0,"p_unit")

        parameter_rows.append({
            "test_year":y,
            "calibration_years":"|".join(map(str,cal_years)),
            "alpha":alpha,
            "calibration_winner_log_loss":cal_loss,
            "optimizer_boundary_hit":boundary_hit,
        })

        for variant,pcol,frame in [
            ("MARKET_ONLY","market_q_norm",test),
            ("RESIDUAL_TILT_UNIT","p_unit",unit),
            ("RESIDUAL_TILT_CALIBRATED","p_residual",test),
        ]:
            pm=probability_metrics(frame,pcol)
            bm=betting_metrics(frame,pcol,f"{variant}_{y}")
            variant_rows.append({"year":y,"variant":variant,**pm,**bm})

        calibrated[y]=test
        del cal,test,unit

    # 2023-2024 are discovery/evaluation only; ROI never selects alpha or a variant.
    pre=pd.DataFrame(variant_rows)
    pre=pre[pre["year"].isin([2023,2024])].copy()
    pre_summary=pre.groupby("variant",as_index=False).agg(
        mean_winner_log_loss=("winner_log_loss","mean"),
        mean_brier=("binary_brier","mean"),
        mean_auc=("roc_auc","mean"),
        total_tickets=("tickets","sum"),
        total_hits=("hits","sum"),
        total_stake_yen=("stake_yen","sum"),
        total_return_yen=("return_yen","sum"),
    )
    pre_summary["pooled_roi_pct"]=np.where(
        pre_summary["total_stake_yen"]>0,
        100.0*pre_summary["total_return_yen"]/pre_summary["total_stake_yen"],
        np.nan,
    )

    vdf=pd.DataFrame(variant_rows)
    confirm=vdf[vdf["year"]==2025].copy()

    pd.DataFrame(coverage).to_csv(out/"build-coverage.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(out/"residual-oof-folds.csv",index=False)
    pd.DataFrame(score_diag).to_csv(out/"winner-residual-score-diagnostics.csv",index=False)
    pd.DataFrame(parameter_rows).to_csv(out/"alpha-calibration.csv",index=False)
    vdf.to_csv(out/"variant-probability-and-ev-metrics.csv",index=False)
    pre_summary.to_csv(out/"pre2025-summary.csv",index=False)
    pd.concat(importances,ignore_index=True).groupby(
        "feature",as_index=False
    )["gain"].sum().sort_values("gain",ascending=False).to_csv(
        out/"feature-importance.csv",index=False
    )

    market_pre=pre_summary[pre_summary["variant"]=="MARKET_ONLY"].iloc[0]
    residual_pre=pre_summary[pre_summary["variant"]=="RESIDUAL_TILT_CALIBRATED"].iloc[0]
    market_2025=confirm[confirm["variant"]=="MARKET_ONLY"].iloc[0].to_dict()
    residual_2025=confirm[confirm["variant"]=="RESIDUAL_TILT_CALIBRATED"].iloc[0].to_dict()

    summary={
        "contract":"L2_TRIO_MARKET_RESIDUAL_V1_RESULT",
        "bet_type":"TRIO",
        "architecture":"market probability baseline * exp(alpha * L1.7-only residual score), normalized within race",
        "residual_target":"normalized official outcome mass - market probability",
        "residual_model_market_features_used":False,
        "alpha_learning":"strictly prior OOF winner-log-loss minimization",
        "manual_ticket_rule_grid":False,
        "manual_edge_threshold_search":False,
        "ev_formula":"edge = corrected_probability * final_odds - 1",
        "buy_rule_diagnostic":"edge > 0 mathematical break-even only",
        "pre2025_probability_delta":{
            "market_mean_winner_log_loss":float(market_pre["mean_winner_log_loss"]),
            "residual_mean_winner_log_loss":float(residual_pre["mean_winner_log_loss"]),
            "residual_minus_market":float(
                residual_pre["mean_winner_log_loss"]-market_pre["mean_winner_log_loss"]
            ),
        },
        "pre2025_residual_pooled_roi_pct":(
            None if pd.isna(residual_pre["pooled_roi_pct"]) else float(residual_pre["pooled_roi_pct"])
        ),
        "confirmation_2025":{
            "market":market_2025,
            "residual_calibrated":residual_2025,
        },
        "2025_used_for_model_or_alpha_selection":False,
        "session_reuse":{
            "base_table_built_once_per_year":True,
            "residual_oof_built_once_per_test_year":True,
            "all_tilts_and_ev_diagnostics_reuse_oof_tables":True,
            "github_artifact_or_cache_used":False,
        },
        "output_policy":"large base/OOF tables remain runner-local; only compact results committed",
        "2026_locked":True,
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )
    print("L2_TRIO_MARKET_RESIDUAL_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
