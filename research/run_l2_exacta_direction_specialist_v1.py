#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_exacta_decomposed_v1 import (
    YEARS, build_pair_year_frame, pair_feature_columns,
    direction_feature_columns, train_pair_ranker
)
from run_l2_exacta_v0 import encode

TEST_YEARS=(2023,2024,2025)
K_VALUES=tuple(range(1,16))
STAKE=100.0

def parse_args():
    p=argparse.ArgumentParser(description="EXACTA direction specialist V1.")
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1); out[int(y)]=p
    if set(out)!=set(YEARS) or 2026 in out:
        raise SystemExit("years must be exactly 2022-2025; 2026 sealed")
    return out

def train_direction(train,test,cols,seed,value_weighted):
    tr=train[train["strict_direction_eligible"]==1].copy().sort_values(
        ["year","race_id","pair_numbers"]
    ).reset_index(drop=True)
    if tr.empty or tr["direction_label"].nunique()!=2:
        raise SystemExit("direction labels invalid")

    te=test.copy()
    te["_orig_index"]=te.index
    te=te.sort_values(["year","race_id","pair_numbers"]).reset_index(drop=True)
    xtr,xte=encode(tr,te,cols)
    y=tr["direction_label"].astype(int).to_numpy()

    fit_kwargs={}
    weight_info={"weighted":False}
    if value_weighted:
        win_ret=np.where(
            y==1,
            pd.to_numeric(tr["return_a_to_b"],errors="coerce").fillna(0).to_numpy(float),
            pd.to_numeric(tr["return_b_to_a"],errors="coerce").fillna(0).to_numpy(float),
        )
        positive=win_ret[win_ret>0]
        if len(positive)==0:
            raise SystemExit("no positive return for value weights")
        cap=float(np.quantile(positive,0.95))
        clipped=np.minimum(win_ret,cap)
        weights=1.0+np.log1p(clipped/100.0)
        weights=weights/weights.mean()
        fit_kwargs["sample_weight"]=weights
        weight_info={
            "weighted":True,
            "cap_return_yen_q95":cap,
            "weight_min":float(weights.min()),
            "weight_mean":float(weights.mean()),
            "weight_max":float(weights.max()),
        }

    model=lgb.LGBMClassifier(
        objective="binary",
        n_estimators=320,
        learning_rate=0.025,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    t=time.perf_counter()
    model.fit(xtr,y,**fit_kwargs)
    sec=time.perf_counter()-t
    pred=np.asarray(model.predict_proba(xte)[:,1],dtype=float)
    s=pd.Series(pred,index=te["_orig_index"].astype(int).to_numpy())
    out=s.reindex(test.index).to_numpy()
    imp=pd.DataFrame({"feature":cols,"gain":model.feature_importances_})
    return out,imp,sec,len(tr),weight_info

def predicted_direction(df,policy):
    qab=df["market_q_a_to_b"].to_numpy(float)
    qba=df["market_q_b_to_a"].to_numpy(float)
    if policy=="MARKET_ONE":
        return np.where(qab>=qba,1,0)
    if policy=="L17_VOTE_ONE":
        dv=df["dir_expert_vote_margin"].to_numpy(float)
        return np.where(dv>0,1,np.where(dv<0,0,np.where(qab>=qba,1,0)))
    if policy=="DIR_STANDARD":
        return (df["dir_standard_prob"].to_numpy(float)>=0.5).astype(int)
    if policy=="DIR_VALUE":
        return (df["dir_value_prob"].to_numpy(float)>=0.5).astype(int)
    raise ValueError(policy)

def evaluate_policy(df,year,k,policy):
    g=df[df["pair_model_rank"]<=k].copy()
    pred=predicted_direction(g,policy)
    rab=g["return_a_to_b"].to_numpy(float)
    rba=g["return_b_to_a"].to_numpy(float)
    ret=np.where(pred==1,rab,rba)
    stake=len(g)*STAKE
    total=float(ret.sum())
    wins=np.sort(ret[ret>0])[::-1]
    top1=float(wins[0]) if len(wins) else 0.0
    top3=float(wins[:3].sum()) if len(wins) else 0.0

    pos=g[g["strict_direction_eligible"]==1].copy()
    pos_pred=predicted_direction(pos,policy)
    correct=int((pos_pred==pos["direction_label"].to_numpy(int)).sum())

    return {
        "year":year,"top_k":k,"policy":policy,
        "pairs_bet":len(g),
        "races":int(g["race_id"].nunique()),
        "tickets":len(g),
        "captured_direction_pairs":len(pos),
        "direction_correct":correct,
        "direction_accuracy_pct":100.0*correct/len(pos) if len(pos) else 0.0,
        "stake_yen":stake,
        "return_yen":total,
        "profit_yen":total-stake,
        "roi_pct":100.0*total/stake if stake else 0.0,
        "roi_minus_top1_pct":100.0*(total-top1)/stake if stake else 0.0,
        "roi_minus_top3_pct":100.0*(total-top3)/stake if stake else 0.0,
    }

def perfect_ceiling(df,year,k):
    g=df[df["pair_model_rank"]<=k].copy()
    rab=g["return_a_to_b"].to_numpy(float)
    rba=g["return_b_to_a"].to_numpy(float)
    ret=np.maximum(rab,rba)
    stake=len(g)*STAKE
    total=float(ret.sum())
    return {
        "year":year,"top_k":k,
        "roi_pct":100.0*total/stake if stake else 0.0,
    }

def stability(metrics):
    out=[]
    for (k,p),g in metrics.groupby(["top_k","policy"],sort=False):
        rois=g["roi_pct"].astype(float).tolist()
        out.append({
            "top_k":int(k),"policy":p,
            "mean_roi_pct":float(np.mean(rois)),
            "median_roi_pct":float(np.median(rois)),
            "worst_year_roi_pct":float(np.min(rois)),
            "best_year_roi_pct":float(np.max(rois)),
            "roi_std_pct":float(np.std(rois)),
            "profitable_years":int(sum(x>100 for x in rois)),
            "pooled_roi_pct":100.0*float(g["return_yen"].sum())/float(g["stake_yen"].sum()),
            "pooled_profit_yen":float(g["profit_yen"].sum()),
            "worst_year_roi_minus_top1_pct":float(g["roi_minus_top1_pct"].min()),
            "mean_direction_accuracy_pct":float(g["direction_accuracy_pct"].mean()),
        })
    return pd.DataFrame(out)

def main():
    a=parse_args(); started=time.perf_counter()
    lp=parse_paths(a.l17_year)

    frames={}
    for y in YEARS:
        frames[y]=build_pair_year_frame(y,load_l17(lp[y],y),a.backfill_root)

    pair_cols=pair_feature_columns(frames[2022],market=True)
    dir_cols=direction_feature_columns(frames[2022],market=True)
    if any(c.startswith("dir_") for c in pair_cols):
        raise SystemExit("directional leakage into pair ranker")

    metric_rows=[]; fold_rows=[]; weight_rows=[]; importance_rows=[]; ceiling_rows=[]
    policies=("MARKET_ONE","L17_VOTE_ONE","DIR_STANDARD","DIR_VALUE")

    for y in TEST_YEARS:
        train=pd.concat([frames[t] for t in YEARS if t<y],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)

        pair_pred,_,pair_sec=train_pair_ranker(train,test,pair_cols,171000+y)
        test["pair_model_score"]=pair_pred
        test["pair_model_rank"]=test.groupby("race_id")["pair_model_score"].rank(
            method="min",ascending=False
        ).astype(int)

        p1,imp1,sec1,n1,w1=train_direction(train,test,dir_cols,181000+y,False)
        p2,imp2,sec2,n2,w2=train_direction(train,test,dir_cols,191000+y,True)
        test["dir_standard_prob"]=p1
        test["dir_value_prob"]=p2

        for name,imp in (("DIR_STANDARD",imp1),("DIR_VALUE",imp2)):
            z=imp.sort_values("gain",ascending=False).head(40).copy()
            z["year"]=y; z["model"]=name
            importance_rows.extend(z.to_dict("records"))

        weight_rows.append({"test_year":y,**w2})
        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(str(t) for t in YEARS if t<y),
            "train_direction_rows":n1,
            "pair_train_seconds":pair_sec,
            "dir_standard_seconds":sec1,
            "dir_value_seconds":sec2,
        })

        for k in K_VALUES:
            for p in policies:
                metric_rows.append(evaluate_policy(test,y,k,p))
            ceiling_rows.append(perfect_ceiling(test,y,k))

        print("DIRECTION_SPECIALIST_FOLD_READY "+json.dumps({
            "test_year":y,
            "train_years":[t for t in YEARS if t<y],
            "direction_train_rows":n1,
            "value_weight_cap_yen":w2["cap_return_yen_q95"],
            "top10_standard_roi":[x["roi_pct"] for x in metric_rows if x["year"]==y and x["top_k"]==10 and x["policy"]=="DIR_STANDARD"][0],
            "top10_value_roi":[x["roi_pct"] for x in metric_rows if x["year"]==y and x["top_k"]==10 and x["policy"]=="DIR_VALUE"][0],
        },separators=(",",":")),flush=True)

    metrics=pd.DataFrame(metric_rows)
    stable=stability(metrics)
    ceiling=pd.DataFrame(ceiling_rows)

    compare=stable[stable["top_k"].isin([1,3,5,10,15])].copy()
    compare=compare.merge(
        ceiling.groupby("top_k",as_index=False)["roi_pct"].agg(
            perfect_direction_worst_year_roi_pct="min",
            perfect_direction_mean_roi_pct="mean",
        ),
        on="top_k",how="left"
    )

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(out/"metrics-by-year-k.csv",index=False)
    stable.to_csv(out/"stability-summary.csv",index=False)
    compare.to_csv(out/"primary-summary.csv",index=False)
    ceiling.to_csv(out/"perfect-direction-ceiling.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(out/"folds.csv",index=False)
    pd.DataFrame(weight_rows).to_csv(out/"value-weight-audit.csv",index=False)
    pd.DataFrame(importance_rows).to_csv(out/"feature-importance.csv",index=False)

    summary={
        "contract":"L2_EXACTA_DIRECTION_SPECIALIST_V1",
        "question":"Can a dedicated direction model close the gap between stable pair selection and exacta profitability?",
        "pair_candidate_generator":"walk-forward decomposed market-aware unordered-pair rank; frozen within each fold",
        "direction_models":{
            "DIR_STANDARD":"market-aware binary direction classifier; uniform training weights",
            "DIR_VALUE":"same architecture; winning exacta payout weighted with log compression and training-only q95 cap"
        },
        "baselines":["MARKET_ONE","L17_VOTE_ONE"],
        "no_skip":True,
        "one_ticket_per_selected_pair":True,
        "top_k":list(K_VALUES),
        "test_years":list(TEST_YEARS),
        "walk_forward":"all prior years only",
        "test_outcomes_used_for_model_selection":False,
        "2026_locked":True,
        "script_total_seconds":time.perf_counter()-started,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 EXACTA Direction Specialist V1\n\n"
        "The unordered-pair candidate generator is held fixed. Only A>B versus B>A is learned. "
        "DIR_STANDARD is the ordinary market-aware direction classifier. DIR_VALUE uses the same model but weights historical "
        "winning-pair examples by exacta payout with log compression and a training-only 95th-percentile cap, so expensive "
        "correct directions matter more without letting one jackpot dominate. Both always buy exactly one orientation for every "
        "selected pair; there is no confidence gate, no odds filter, no LAW search, and no test-year tuning. 2026 is sealed.\n",
        encoding="utf-8"
    )

    print("L2_EXACTA_DIRECTION_SPECIALIST_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
    print("===== PRIMARY SUMMARY =====")
    print(compare.to_csv(index=False))
    print("===== TOP10 YEAR METRICS =====")
    print(metrics[metrics.top_k==10].to_csv(index=False))

if __name__=="__main__":
    main()
