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
        description="TRIO Direct EV V1: estimate ticket hit probability directly, then evaluate p*odds."
    )
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--work-dir",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def normalize_race(df,col,out):
    z=df.copy()
    denom=z.groupby(["year","race_id"])[col].transform("sum").clip(lower=EPS)
    z[out]=(z[col]/denom).astype("float32")
    return z


def direct_feature_columns(df):
    cols=[c for c in feature_columns(df) if c not in MARKET_FEATURES]
    bad=[c for c in cols if "market" in c.lower() or c=="odds"]
    if bad:
        raise SystemExit(f"market leakage into direct probability model: {sorted(bad)}")
    return cols


def train_direct_probability(train,test,cols,seed):
    xtr,xte=encode(train,test,cols)
    y=train["hit"].astype(int).to_numpy()
    model=lgb.LGBMClassifier(
        objective="binary",
        n_estimators=260,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.90,
        colsample_bytree=0.88,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(xtr,y)
    raw=np.asarray(model.predict_proba(xte)[:,1],dtype=float)
    imp=pd.DataFrame({"feature":cols,"gain":model.feature_importances_})
    return raw,imp


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


def fit_blend_alpha(cal):
    pos=cal[cal["hit"]==1]
    pm=pos["p_model"].to_numpy(dtype=float)
    pq=pos["market_q_norm"].to_numpy(dtype=float)

    def objective(alpha):
        p=np.clip(alpha*pm+(1.0-alpha)*pq,1e-12,1.0)
        return float(-np.log(p).mean())

    res=minimize_scalar(
        objective,
        bounds=(0.0,1.0),
        method="bounded",
        options={"xatol":1e-5},
    )
    return float(np.clip(res.x,0.0,1.0)),float(res.fun)


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

    # Build each full TRIO base table once in this runner and reuse it.
    base_paths={}
    coverage=[]
    for y in YEARS:
        frame,meta=build_year_frame(y,l17[y],a.backfill_root)
        frame=add_market_features(frame)
        p=work/f"trio-base-{y}.pkl.gz"
        frame.to_pickle(p,compression="gzip")
        base_paths[y]=p
        coverage.append(meta)
        print(f"TRIO_DIRECT_BASE_READY year={y} rows={len(frame)}",flush=True)
        del frame

    oof_paths={}
    fold_rows=[]
    importances=[]
    for y in OOF_YEARS:
        train_years=[2021] if y==2022 else [t for t in (2022,2023,2024) if t<y]
        train=pd.concat(
            [pd.read_pickle(base_paths[t],compression="gzip") for t in train_years],
            ignore_index=True,
        )
        test=pd.read_pickle(base_paths[y],compression="gzip").reset_index(drop=True)
        cols=direct_feature_columns(train)
        raw,imp=train_direct_probability(train,test,cols,97000+y)
        test["raw_model_probability"]=raw
        test=normalize_race(test,"raw_model_probability","p_model")

        keep=[
            "year","race_id","race_date","trio_numbers","hit","return_yen_per100",
            "odds","market_q_norm","p_model",
        ]
        oof=test[keep].copy()
        p=work/f"trio-direct-oof-{y}.pkl.gz"
        oof.to_pickle(p,compression="gzip")
        oof_paths[y]=p

        pm=probability_metrics(oof,"p_model")
        pq=probability_metrics(oof,"market_q_norm")
        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(map(str,train_years)),
            "train_trios":int(len(train)),
            "test_trios":int(len(test)),
            "source_races":int(test["race_id"].nunique()),
            "positive_trios":int(test["hit"].sum()),
            "feature_count":int(len(cols)),
            "model_winner_log_loss":pm["winner_log_loss"],
            "model_binary_brier":pm["binary_brier"],
            "model_auc":pm["roc_auc"],
            "market_winner_log_loss":pq["winner_log_loss"],
            "market_binary_brier":pq["binary_brier"],
            "market_auc":pq["roc_auc"],
        })
        imp["test_year"]=y
        importances.append(imp)
        print(
            f"TRIO_DIRECT_OOF_READY year={y} rows={len(oof)} "
            f"model_wll={pm['winner_log_loss']:.6f} market_wll={pq['winner_log_loss']:.6f}",
            flush=True,
        )
        del train,test,oof

    oof={y:pd.read_pickle(oof_paths[y],compression="gzip") for y in OOF_YEARS}

    parameter_rows=[]
    variant_rows=[]
    eval_frames={}
    for y in CALIBRATED_YEARS:
        cal_years=[t for t in OOF_YEARS if t<y]
        cal=pd.concat([oof[t] for t in cal_years],ignore_index=True)
        test=oof[y].copy()

        alpha,cal_loss=fit_blend_alpha(cal)
        test["p_blend"]=(
            alpha*test["p_model"]+(1.0-alpha)*test["market_q_norm"]
        ).astype("float32")
        # Inputs already sum to one per race, so a convex blend also sums to one.
        parameter_rows.append({
            "test_year":y,
            "calibration_years":"|".join(map(str,cal_years)),
            "blend_alpha_model":alpha,
            "blend_alpha_market":1.0-alpha,
            "calibration_winner_log_loss":cal_loss,
        })

        for variant,pcol in [
            ("DIRECT_MODEL","p_model"),
            ("MARKET_ONLY","market_q_norm"),
            ("AUTO_BLEND","p_blend"),
        ]:
            pm=probability_metrics(test,pcol)
            bm=betting_metrics(test,pcol,f"{variant}_{y}")
            variant_rows.append({"year":y,"variant":variant,**pm,**bm})

        eval_frames[y]=test
        del cal,test

    # Choose the probability family on 2023-2024 only. ROI is never used here.
    pre=pd.concat([eval_frames[2023],eval_frames[2024]],ignore_index=True)
    pre_scores=[]
    for variant,pcol in [
        ("DIRECT_MODEL","p_model"),
        ("MARKET_ONLY","market_q_norm"),
        ("AUTO_BLEND","p_blend"),
    ]:
        pm=probability_metrics(pre,pcol)
        pre_scores.append({"variant":variant,**pm})
    pre_scores=sorted(
        pre_scores,
        key=lambda r:(r["winner_log_loss"],r["binary_brier"],-r["roc_auc"],r["variant"]),
    )
    champion=pre_scores[0]["variant"]

    vdf=pd.DataFrame(variant_rows)
    confirm=vdf[vdf["year"]==2025].copy()
    champion_confirm=confirm[confirm["variant"]==champion].iloc[0].to_dict()

    pd.DataFrame(coverage).to_csv(out/"build-coverage.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(out/"direct-oof-metrics.csv",index=False)
    pd.DataFrame(parameter_rows).to_csv(out/"blend-parameters.csv",index=False)
    vdf.to_csv(out/"variant-probability-and-ev-metrics.csv",index=False)
    pd.DataFrame(pre_scores).to_csv(out/"pre2025-probability-ranking.csv",index=False)
    pd.concat(importances,ignore_index=True).groupby(
        "feature",as_index=False
    )["gain"].sum().sort_values("gain",ascending=False).to_csv(
        out/"feature-importance.csv",index=False
    )

    summary={
        "contract":"L2_TRIO_DIRECT_EV_V1_RESULT",
        "bet_type":"TRIO",
        "architecture":"direct ticket hit probability -> race normalization -> optional prior-OOF market blend -> EV",
        "direct_model_uses_market_features":False,
        "probability_target":"official payout-positive TRIO ticket",
        "race_probability_normalization":"sum direct probabilities to 1.0 within each complete-priced race",
        "market_probability":"normalized inverse final TRIO odds",
        "ev_formula":"edge = probability * final_odds - 1",
        "buy_rule":"edge > 0 only",
        "manual_edge_threshold_search":False,
        "manual_ticket_rule_grid":False,
        "calibration":{
            "method":"convex blend of direct model probability and market probability",
            "alpha_learned_by":"minimum winner log-loss on strictly prior OOF years",
            "2025_used_for_calibration_or_variant_selection":False,
        },
        "pre2025_variant_selection_basis":"minimum winner log-loss on 2023-2024; ROI not used",
        "pre2025_champion":champion,
        "pre2025_probability_ranking":pre_scores,
        "confirmation_2025":confirm.to_dict(orient="records"),
        "champion_confirmation_2025":champion_confirm,
        "session_reuse":{
            "base_table_built_once_per_year":True,
            "direct_oof_probability_built_once_per_test_year":True,
            "all_probability_and_ev_variants_reuse_oof_tables":True,
            "github_artifact_or_cache_used":False,
        },
        "output_policy":"large base/OOF tables remain runner-local; only compact results committed",
        "2026_locked":True,
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )
    print("L2_TRIO_DIRECT_EV_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
