#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_market_gap_trio_v0 import add_market_features, build_year_frame
from run_l2_trio_market_residual_v1 import (
    residual_feature_columns,
    train_residual_model,
    fit_alpha,
    apply_tilt,
    probability_metrics,
    betting_metrics,
)

YEARS=(2021,2022,2023,2024,2025)
OOF_YEARS=(2022,2023,2024,2025)
TUNE_YEARS=(2023,2024)
CONFIRM_YEAR=2025
LEAF_CANDIDATES=(4,8,16,32)
EPS=1e-12


def parse_args():
    p=argparse.ArgumentParser(
        description="TRIO Residual Gate V1: learn where L1.7 residual correction helps the market."
    )
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--work-dir",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def meta_features(df):
    q=np.clip(df["market_q_norm"].to_numpy(dtype=float),EPS,1.0)
    return pd.DataFrame({
        "market_log_q":np.log(q).astype("float32"),
        "residual_score":df["residual_score"].to_numpy(dtype="float32"),
    })


def normalize_race(df,raw,out_col):
    z=df.copy()
    z["_meta_raw"]=np.clip(np.asarray(raw,dtype=float),EPS,None)
    denom=z.groupby(["year","race_id"])["_meta_raw"].transform("sum").clip(lower=EPS)
    z[out_col]=(z["_meta_raw"]/denom).astype("float32")
    return z.drop(columns="_meta_raw")


def train_gate(train,test,num_leaves,seed):
    xtr=meta_features(train)
    xte=meta_features(test)
    y=train["hit"].astype(int).to_numpy()
    model=lgb.LGBMClassifier(
        objective="binary",
        n_estimators=220,
        learning_rate=0.03,
        num_leaves=int(num_leaves),
        min_child_samples=5000,
        subsample=0.90,
        colsample_bytree=1.0,
        reg_lambda=12.0,
        reg_alpha=2.0,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(xtr,y)
    raw=np.asarray(model.predict_proba(xte)[:,1],dtype=float)
    return normalize_race(test,raw,"p_gate"),model


def build_residual_oof(base_paths,work):
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
        cols=residual_feature_columns(train)
        score,imp=train_residual_model(train,test,cols,99000+y)
        test["residual_score"]=score
        keep=[
            "year","race_id","race_date","trio_numbers","hit","return_yen_per100",
            "odds","market_q_norm","residual_score",
        ]
        oof=test[keep].copy()
        p=work/f"trio-residual-oof-{y}.pkl.gz"
        oof.to_pickle(p,compression="gzip")
        oof_paths[y]=p
        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(map(str,train_years)),
            "train_trios":int(len(train)),
            "test_trios":int(len(test)),
            "source_races":int(test["race_id"].nunique()),
            "positive_trios":int(test["hit"].sum()),
            "residual_feature_count":int(len(cols)),
        })
        imp["test_year"]=y
        importances.append(imp)
        print(f"TRIO_GATE_RESIDUAL_OOF_READY year={y} rows={len(oof)}",flush=True)
        del train,test,oof
    return oof_paths,fold_rows,importances


def score_gate_config(oof,num_leaves):
    rows=[]
    for y in TUNE_YEARS:
        prior=[t for t in OOF_YEARS if t<y]
        tr=pd.concat([oof[t] for t in prior],ignore_index=True)
        te=oof[y].copy()
        pred,_=train_gate(tr,te,num_leaves,100000+100*num_leaves+y)
        pm=probability_metrics(pred,"p_gate")
        bm=betting_metrics(pred,"p_gate",f"GATE_L{num_leaves}_{y}")
        rows.append({
            "num_leaves":num_leaves,
            "year":y,
            **pm,
            "tickets":bm["tickets"],
            "hits":bm["hits"],
            "stake_yen":bm["stake_yen"],
            "return_yen":bm["return_yen"],
            "roi_pct":bm["roi_pct"],
        })
        del tr,te,pred

    pooled_wll=float(np.mean([r["winner_log_loss"] for r in rows]))
    pooled_brier=float(np.mean([r["binary_brier"] for r in rows]))
    pooled_auc=float(np.mean([r["roc_auc"] for r in rows]))
    return rows,{
        "num_leaves":num_leaves,
        "mean_winner_log_loss":pooled_wll,
        "mean_brier":pooled_brier,
        "mean_auc":pooled_auc,
        "total_tickets":int(sum(r["tickets"] for r in rows)),
        "total_hits":int(sum(r["hits"] for r in rows)),
        "total_stake_yen":float(sum(r["stake_yen"] for r in rows)),
        "total_return_yen":float(sum(r["return_yen"] for r in rows)),
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

    # One runner session: full TRIO bases are built once and reused.
    base_paths={}
    coverage=[]
    for y in YEARS:
        frame,meta=build_year_frame(y,l17[y],a.backfill_root)
        frame=add_market_features(frame)
        p=work/f"trio-base-{y}.pkl.gz"
        frame.to_pickle(p,compression="gzip")
        base_paths[y]=p
        coverage.append(meta)
        print(f"TRIO_GATE_BASE_READY year={y} rows={len(frame)}",flush=True)
        del frame

    oof_paths,fold_rows,importances=build_residual_oof(base_paths,work)
    oof={y:pd.read_pickle(oof_paths[y],compression="gzip") for y in OOF_YEARS}

    # Hyperparameter selection uses only probability quality on 2023-2024.
    # No ticket boundary or ROI participates in this choice.
    tune_detail=[]
    config_rows=[]
    for leaves in LEAF_CANDIDATES:
        detail,summary=score_gate_config(oof,leaves)
        tune_detail.extend(detail)
        config_rows.append(summary)

    configs=pd.DataFrame(config_rows).sort_values(
        ["mean_winner_log_loss","mean_brier","mean_auc","num_leaves"],
        ascending=[True,True,False,True],
    ).reset_index(drop=True)
    configs["selection_rank"]=range(1,len(configs)+1)
    chosen_leaves=int(configs.iloc[0]["num_leaves"])

    # Strictly prior data for 2025 confirmation.
    train_2025=pd.concat([oof[2022],oof[2023],oof[2024]],ignore_index=True)
    test_2025=oof[2025].copy()
    gate_2025,gate_model=train_gate(
        train_2025,test_2025,chosen_leaves,101000+chosen_leaves
    )

    # Prior-only global alpha baseline, for a fair comparison with the previous experiment.
    alpha,alpha_loss,boundary_hit=fit_alpha(train_2025)
    global_2025=apply_tilt(test_2025,alpha,"p_global")

    variants=[]
    for variant,pcol,frame in [
        ("MARKET_ONLY","market_q_norm",test_2025),
        ("GLOBAL_ALPHA_RESIDUAL","p_global",global_2025),
        ("LOCAL_RESIDUAL_GATE","p_gate",gate_2025),
    ]:
        pm=probability_metrics(frame,pcol)
        bm=betting_metrics(frame,pcol,f"{variant}_2025")
        variants.append({"year":2025,"variant":variant,**pm,**bm})

    market_pm=probability_metrics(test_2025,"market_q_norm")
    gate_pm=probability_metrics(gate_2025,"p_gate")

    # Compact diagnostics showing whether the gate changes the market locally.
    diag=gate_2025.copy()
    diag["ratio_vs_market"]=(
        diag["p_gate"]/diag["market_q_norm"].clip(lower=EPS)
    ).astype("float32")
    diag["log_ratio_vs_market"]=np.log(
        diag["ratio_vs_market"].clip(lower=EPS)
    ).astype("float32")
    winners=diag[diag["hit"]==1]
    gate_diag={
        "all_ratio_median":float(diag["ratio_vs_market"].median()),
        "all_abs_log_ratio_mean":float(diag["log_ratio_vs_market"].abs().mean()),
        "winner_ratio_median":float(winners["ratio_vs_market"].median()),
        "winner_log_ratio_mean":float(winners["log_ratio_vs_market"].mean()),
        "winner_gate_up_pct":100.0*float((winners["ratio_vs_market"]>1.0).mean()),
        "tickets_probability_raised_pct":100.0*float((diag["ratio_vs_market"]>1.0).mean()),
    }

    pd.DataFrame(coverage).to_csv(out/"build-coverage.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(out/"residual-oof-folds.csv",index=False)
    pd.DataFrame(tune_detail).to_csv(out/"gate-complexity-by-year.csv",index=False)
    configs.to_csv(out/"gate-complexity-selection.csv",index=False)
    pd.DataFrame(variants).to_csv(out/"confirmation-2025-variants.csv",index=False)
    pd.concat(importances,ignore_index=True).groupby(
        "feature",as_index=False
    )["gain"].sum().sort_values("gain",ascending=False).to_csv(
        out/"residual-feature-importance.csv",index=False
    )
    pd.DataFrame({
        "feature":["market_log_q","residual_score"],
        "gain":gate_model.feature_importances_,
    }).to_csv(out/"gate-feature-importance.csv",index=False)

    summary={
        "contract":"L2_TRIO_RESIDUAL_GATE_V1_RESULT",
        "bet_type":"TRIO",
        "architecture":"local meta-calibrator learns where market probability should be adjusted by L1.7 residual score",
        "gate_features":["market_log_q","residual_score"],
        "manual_ticket_rule_boundaries":False,
        "gate_split_boundaries_learned":True,
        "complexity_candidates":{"num_leaves":list(LEAF_CANDIDATES)},
        "complexity_selection_years":list(TUNE_YEARS),
        "complexity_selection_basis":"probability winner-log-loss then Brier/AUC; ROI not used",
        "chosen_num_leaves":chosen_leaves,
        "global_alpha_2025_baseline":{
            "alpha":alpha,
            "prior_oof_winner_log_loss":alpha_loss,
            "optimizer_boundary_hit":boundary_hit,
        },
        "confirmation_2025_probability_delta":{
            "market_winner_log_loss":market_pm["winner_log_loss"],
            "gate_winner_log_loss":gate_pm["winner_log_loss"],
            "gate_minus_market":gate_pm["winner_log_loss"]-market_pm["winner_log_loss"],
            "market_auc":market_pm["roc_auc"],
            "gate_auc":gate_pm["roc_auc"],
        },
        "confirmation_2025_variants":variants,
        "confirmation_2025_gate_diagnostics":gate_diag,
        "2025_used_for_gate_or_complexity_selection":False,
        "manual_edge_threshold_search":False,
        "ev_buy_rule_diagnostic":"p * final_odds > 1 only",
        "session_reuse":{
            "base_table_built_once_per_year":True,
            "residual_oof_built_once_per_test_year":True,
            "all_gate_complexities_reuse_same_oof_tables":True,
            "github_artifact_or_cache_used":False,
        },
        "2026_locked":True,
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )
    print("L2_TRIO_RESIDUAL_GATE_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
