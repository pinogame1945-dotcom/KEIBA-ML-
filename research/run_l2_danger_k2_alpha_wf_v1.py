#!/usr/bin/env python3
import argparse,csv,json,math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from run_l2_danger_k2_third_selector_v1 import (
    YEARS,TEST_YEARS,LOCKED_YEARS,TICKET_PRICE,
    load_fixed_ledgers,load_router,parse_paths,build_rows,make_model,
    safe_auc,safe_ap,write_csv,max_drawdown,
)

ALPHAS=(0.0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0)
TOPK=2
OOF_FOLDS=5
FIXED_BENCHMARK_ALPHAS=(0.0,0.5,1.0)

def parse_args():
    p=argparse.ArgumentParser(description="Walk-forward alpha selection for K2 value ranking.")
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def score(prob,odd,alpha):
    return float(prob)*(max(float(odd),1e-9)**float(alpha))

def evaluate(scored,race_eval,label,test_year,alpha):
    rows=[]
    for rid,g in scored.groupby("race_id",sort=False):
        ranked=g.sort_values(["value_score","third_prob","horse_id"],ascending=[False,False,True])
        chosen=ranked.head(min(TOPK,len(ranked)))
        info=race_eval[rid]
        a1,a2=info["a1"],info["a2"]
        nums=info["horse_no"]
        ret=0.0
        ids=[]
        for x in chosen.itertuples(index=False):
            hid=str(x.horse_id)
            ids.append(hid)
            hn=nums[hid]
            ret+=float(info["payouts"].get(("TRIFECTA",(nums[a1],nums[a2],hn)),0.0))
            ret+=float(info["payouts"].get(("TRIFECTA",(nums[a2],nums[a1],hn)),0.0))
        tickets=2*len(ids)
        stake=tickets*TICKET_PRICE
        rows.append({
            "policy":label,"test_year":test_year,"alpha":float(alpha),
            "race_id":rid,"race_date":info["race_date"],
            "selected_k2_ids":"|".join(ids),
            "tickets":tickets,"stake_yen":stake,"return_yen":ret,
            "profit_yen":ret-stake,"hit":int(ret>0),
        })
    return pd.DataFrame(rows)

def summarize(df):
    stake=float(df["stake_yen"].sum()) if len(df) else 0.0
    ret=float(df["return_yen"].sum()) if len(df) else 0.0
    returns=sorted([float(x) for x in df["return_yen"]],reverse=True) if len(df) else []
    top1=returns[0] if returns else 0.0
    return {
        "races":len(df),
        "hit_races":int(df["hit"].sum()) if len(df) else 0,
        "tickets":int(df["tickets"].sum()) if len(df) else 0,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(df) if len(df) else 0.0,
        "top1_return_share_pct":100*top1/ret if ret else None,
        "roi_without_top1_pct":100*(ret-top1)/stake if stake else None,
    }

def oof_struct_probs(train):
    model0,features=make_model(False)
    y=train["third_label"].astype(int).to_numpy()
    groups=train["race_id"].astype(str).to_numpy()
    n_groups=len(set(groups.tolist()))
    n_splits=min(OOF_FOLDS,n_groups)
    if n_splits<2:
        raise SystemExit("not enough race groups for OOF")
    probs=np.full(len(train),np.nan,dtype=float)
    fold_rows=[]
    splitter=GroupKFold(n_splits=n_splits)
    for fold,(tr,va) in enumerate(splitter.split(train[features],y,groups),1):
        ytr=y[tr]
        if len(set(ytr.tolist()))<2:
            raise SystemExit(f"OOF fold single class fold={fold}")
        model,_=make_model(False)
        model.fit(train.iloc[tr][features],train.iloc[tr]["third_label"].astype(int))
        probs[va]=model.predict_proba(train.iloc[va][features])[:,1]
        fold_rows.append({
            "fold":fold,"train_rows":len(tr),"valid_rows":len(va),
            "train_races":int(train.iloc[tr]["race_id"].nunique()),
            "valid_races":int(train.iloc[va]["race_id"].nunique()),
            "train_positive":int(train.iloc[tr]["third_label"].sum()),
            "valid_positive":int(train.iloc[va]["third_label"].sum()),
        })
    if np.isnan(probs).any():
        raise SystemExit("OOF probability coverage incomplete")
    return probs,features,fold_rows

def select_alpha(train_scored,race_eval,outer_test_year):
    candidate_rows=[]
    for alpha in ALPHAS:
        s=train_scored.copy()
        s["value_score"]=[score(p,o,alpha) for p,o in zip(s["third_prob"],s["k2_win_odds"])]
        ev=evaluate(s,race_eval,"TRAIN_OOF",outer_test_year,alpha)
        overall=summarize(ev)
        per_year=[]
        for y in sorted(s["year"].unique()):
            e=ev[ev["race_id"].isin(set(s.loc[s["year"]==y,"race_id"].astype(str)))]
            m=summarize(e)
            per_year.append((int(y),m))
        min_year_roi=min((m["roi_pct"] or 0.0) for _,m in per_year)
        min_year_wo=min((m["roi_without_top1_pct"] or 0.0) for _,m in per_year)
        max_top1_share=max((m["top1_return_share_pct"] if m["top1_return_share_pct"] is not None else 100.0) for _,m in per_year)
        candidate_rows.append({
            "outer_test_year":outer_test_year,
            "alpha":float(alpha),
            "train_years":"|".join(str(y) for y,_ in per_year),
            "train_races":overall["races"],
            "train_hits":overall["hit_races"],
            "train_roi_pct":overall["roi_pct"],
            "train_roi_without_top1_pct":overall["roi_without_top1_pct"],
            "train_max_drawdown_yen":overall["max_drawdown_yen"],
            "train_top1_return_share_pct":overall["top1_return_share_pct"],
            "min_train_year_roi_pct":min_year_roi,
            "min_train_year_roi_without_top1_pct":min_year_wo,
            "max_train_year_top1_share_pct":max_top1_share,
        })
    # Robustness first: highest worst-year ROI after removing the largest return,
    # then combined ROI without top1, then worst-year raw ROI, then lower concentration,
    # then lower alpha as the simpler/less price-leveraged tie-break.
    best=max(candidate_rows,key=lambda r:(
        r["min_train_year_roi_without_top1_pct"],
        r["train_roi_without_top1_pct"] if r["train_roi_without_top1_pct"] is not None else -1e9,
        r["min_train_year_roi_pct"],
        -(r["max_train_year_top1_share_pct"] if r["max_train_year_top1_share_pct"] is not None else 100.0),
        -r["alpha"],
    ))
    for r in candidate_rows:
        r["selected"]=int(r["alpha"]==best["alpha"])
    return float(best["alpha"]),candidate_rows

def main():
    a=parse_args()
    paths=parse_paths(a.router_year)
    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={y:load_router(paths[y],y) for y in YEARS}
    for y in YEARS:
        missing=set(fixed[y])-set(routers[y])
        if missing:
            raise SystemExit(f"fixed alerts missing router y={y} sample={sorted(missing)[:5]}")

    df,race_eval,counters=build_rows(fixed,routers,Path(a.backfill_root))
    if any(df["year"]>=2026):
        raise SystemExit("2026 leakage")
    expected_races={2022:344,2023:346,2024:342,2025:346}
    for y,n in expected_races.items():
        got=int(df[df["year"]==y]["race_id"].nunique())
        if got!=n:
            raise SystemExit(f"K2 race universe drift y={y} got={got} expected={n}")

    folds=[]
    alpha_rows=[]
    quality=[]
    test_metrics=[]
    selected_rows=[]
    oof_rows=[]

    for test_year in TEST_YEARS:
        train=df[df["year"]<test_year].copy().reset_index(drop=True)
        test=df[df["year"]==test_year].copy().reset_index(drop=True)

        oof_prob,features,oof_fold_rows=oof_struct_probs(train)
        train_scored=train[["year","race_id","race_date","horse_id","k2_win_odds"]].copy()
        train_scored["third_prob"]=oof_prob
        chosen_alpha,candidates=select_alpha(train_scored,race_eval,test_year)
        alpha_rows.extend(candidates)
        for x in oof_fold_rows:
            oof_rows.append({"outer_test_year":test_year,**x})

        model,_=make_model(False)
        model.fit(train[features],train["third_label"].astype(int))
        te_prob=model.predict_proba(test[features])[:,1]
        quality.append({
            "test_year":test_year,
            "chosen_alpha":chosen_alpha,
            "train_rows":len(train),"train_positive":int(train["third_label"].sum()),
            "test_rows":len(test),"test_positive":int(test["third_label"].sum()),
            "roc_auc":safe_auc(test["third_label"].astype(int),te_prob),
            "pr_auc":safe_ap(test["third_label"].astype(int),te_prob),
            "feature_count_input":len(features),
        })
        folds.append({
            "test_year":test_year,
            "train_years":"|".join(map(str,sorted(train["year"].unique()))),
            "train_races":int(train["race_id"].nunique()),
            "test_races":int(test["race_id"].nunique()),
            "chosen_alpha":chosen_alpha,
        })

        base=test[["year","race_id","race_date","horse_id","k2_win_odds"]].copy()
        base["third_prob"]=te_prob

        for label,alpha in [
            ("WF_SELECTED",chosen_alpha),
            ("FIXED_A0",0.0),
            ("FIXED_A05",0.5),
            ("FIXED_A1",1.0),
        ]:
            s=base.copy()
            s["value_score"]=[score(p,o,alpha) for p,o in zip(s["third_prob"],s["k2_win_odds"])]
            ev=evaluate(s,race_eval,label,test_year,alpha)
            m=summarize(ev)
            test_metrics.append({
                "policy":label,"test_year":test_year,"alpha":float(alpha),**m
            })
            selected_rows.extend(ev.to_dict("records"))

    stability=[]
    for policy in ("WF_SELECTED","FIXED_A0","FIXED_A05","FIXED_A1"):
        rs=[r for r in test_metrics if r["policy"]==policy]
        stake=sum(r["stake_yen"] for r in rs)
        ret=sum(r["return_yen"] for r in rs)
        wo=[r["roi_without_top1_pct"] if r["roi_without_top1_pct"] is not None else 0.0 for r in rs]
        shares=[r["top1_return_share_pct"] if r["top1_return_share_pct"] is not None else 100.0 for r in rs]
        stability.append({
            "policy":policy,
            "alphas":"|".join(str(r["alpha"]) for r in sorted(rs,key=lambda x:x["test_year"])),
            "roi_2023":next(r["roi_pct"] for r in rs if r["test_year"]==2023),
            "roi_2024":next(r["roi_pct"] for r in rs if r["test_year"]==2024),
            "roi_2025":next(r["roi_pct"] for r in rs if r["test_year"]==2025),
            "min_year_roi_pct":min(r["roi_pct"] or 0.0 for r in rs),
            "combined_roi_pct":100*ret/stake if stake else None,
            "combined_profit_yen":ret-stake,
            "min_roi_without_top1_pct":min(wo),
            "max_top1_return_share_pct":max(shares),
            "all_years_roi_100plus":int(all((r["roi_pct"] or 0.0)>=100 for r in rs)),
        })

    summary={
        "contract":"L2_DANGER_K2_ALPHA_WF_V1",
        "analysis_years":list(YEARS),
        "test_years":list(TEST_YEARS),
        "locked_years":list(LOCKED_YEARS),
        "ticket":"Fixed BASE seats A1/A2 both orders, selected K2 Top2 only.",
        "probability_model":"STRUCT-only exact-third logistic model.",
        "alpha_grid":list(ALPHAS),
        "score":"predicted third probability * (K2 WIN odds ** alpha)",
        "alpha_selection":"TRAIN-only race-group OOF probabilities. Choose alpha by highest worst-train-year ROI after removing that year's largest return; tie-break by combined ROI-without-top1, worst-year raw ROI, lower concentration, then lower alpha.",
        "outer_walk_forward":"2022->2023; 2022-23->2024; 2022-24->2025.",
        "market_caveat":"Historical final WIN odds proxy only; production requires pre-bet timestamp contract.",
        "anti_leakage":"No test-year alpha tuning. No trifecta price completeness filter. No payout/result inputs; payouts are train-selection/evaluation targets only within proper split.",
        "folds":folds,
        "quality":quality,
        "alpha_candidates":alpha_rows,
        "test_metrics":test_metrics,
        "stability":stability,
        "counters":counters,
        "production_promotion":False,
    }

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"folds.csv",folds)
    write_csv(out/"oof-folds.csv",oof_rows)
    write_csv(out/"alpha-selection.csv",alpha_rows)
    write_csv(out/"model-quality.csv",quality)
    write_csv(out/"test-metrics.csv",test_metrics)
    write_csv(out/"stability.csv",stability)
    write_csv(out/"selected-races.csv",selected_rows)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Danger K2 Alpha Walk-Forward V1\n\n"
        "Validates whether the prior alpha=1 result generalizes. "
        "Alpha is selected using TRAIN-only race-group OOF predictions with jackpot-robust ROI as the primary criterion, "
        "then carried untouched into the next year. BASE seat structure and K2 Top2 count stay fixed. "
        "2026 remains sealed and this run does not promote a production rule.\n",
        encoding="utf-8",
    )
    print("L2_DANGER_K2_ALPHA_WF_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
