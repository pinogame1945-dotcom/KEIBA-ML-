#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from research.run_l2_bet_kings_arena_v1 import (
    BET_TYPES,
    calibrate,
    chronological_split,
    encode_fit_other,
    feature_columns,
    load_template,
    make_model,
)

OOS_YEARS=(2023,2024,2025)
EVAL_YEARS=(2024,2025)
ROUTER_NAMES=("SIMPLE_EXPECTED_ROI","STRATEGY_UTILITY","DIRECT_BET_TYPE")
TARGET_PREFIXES=("actual_",)
FORBIDDEN_ROUTER_TOKENS=("finish","winner","payout","return_yen","profit_yen","roi_pct","actual_")
ROUTER_CATEGORICAL={"bet_type","template"}
DIRECT_CLASSES=("SKIP","WIN","QUINELLA","EXACTA","TRIO","TRIFECTA")


def parse_args():
    p=argparse.ArgumentParser(description="L2 Router Arena V1")
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--market-gap-race-summary",required=True)
    p.add_argument("--k2-novel-ledger",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--work-dir",required=True)
    return p.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with open(path,newline="",encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def finite(v,default=0.0):
    try:
        x=float(v)
        return x if math.isfinite(x) else default
    except (TypeError,ValueError):
        return default


def as_bool(v):
    return str(v).strip().lower() in {"1","true","yes"}


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def percentile(values,q):
    vals=np.asarray(list(values),dtype=float)
    if len(vals)==0:
        return 0.0
    return float(np.quantile(vals,q))


def load_market_gap(path):
    out={}
    for r in read_csv(path):
        rid=str(r.get("race_id") or "")
        if not rid:
            continue
        out[rid]={
            "market_gap_seven_top1_market_rank":finite(r.get("seven_top1_market_rank"),99.0),
            "market_gap_seven_top1_final_odds":finite(r.get("seven_top1_final_odds"),0.0),
            "market_gap_market_favorite_seven_rank":finite(r.get("market_favorite_seven_rank"),99.0),
            "market_gap_market_favorite_in_seven_union":1.0 if as_bool(r.get("market_favorite_in_seven_union")) else 0.0,
            "market_gap_top1_agreement":1.0 if as_bool(r.get("top1_agreement")) else 0.0,
            "market_gap_top3_overlap_count":finite(r.get("top3_overlap_count"),0.0),
            "market_gap_top3_jaccard":finite(r.get("top3_jaccard"),0.0),
            "market_gap_seven_market_spearman":finite(r.get("seven_market_spearman"),0.0),
            "market_gap_k2_novel_count":finite(r.get("k2_novel_count"),0.0),
        }
    return out


def load_novel_race_features(path):
    grouped=defaultdict(list)
    for r in read_csv(path):
        grouped[str(r["race_id"])].append(r)
    out={}
    for rid,rows in grouped.items():
        odds=[finite(r.get("final_odds"),0.0) for r in rows]
        ranks=[finite(r.get("market_rank"),99.0) for r in rows]
        source_counts=[finite(r.get("source_count"),0.0) for r in rows]
        out[rid]={
            "k2_novel_market_count":float(len(rows)),
            "k2_novel_rank7plus_count":float(sum(x>=7 for x in ranks)),
            "k2_novel_50_100_count":float(sum(50<=x<100 for x in odds)),
            "k2_novel_10_20_count":float(sum(10<=x<20 for x in odds)),
            "k2_novel_100plus_count":float(sum(x>=100 for x in odds)),
            "k2_novel_odds_mean":float(np.mean(odds)) if odds else 0.0,
            "k2_novel_odds_min":float(np.min(odds)) if odds else 0.0,
            "k2_novel_odds_max":float(np.max(odds)) if odds else 0.0,
            "k2_novel_market_rank_mean":float(np.mean(ranks)) if ranks else 0.0,
            "k2_novel_market_rank_max":float(np.max(ranks)) if ranks else 0.0,
            "k2_novel_dual_source_count":float(sum(x>=2 for x in source_counts)),
        }
    return out


def strategy_row(group,threshold,market_gap,novel_gap):
    selected=group[group["edge"]>=float(threshold)].copy()
    if selected.empty:
        return None
    selected=selected.sort_values(["selection_key"])
    n=len(selected)
    stake=100.0*n
    ret=float(pd.to_numeric(selected["return_yen_per100"],errors="coerce").fillna(0.0).sum())
    probs=pd.to_numeric(selected["predicted_probability"],errors="coerce").fillna(0.0)
    odds=pd.to_numeric(selected["odds"],errors="coerce").fillna(0.0)
    edge=pd.to_numeric(selected["edge"],errors="coerce").fillna(-1.0)
    exp_return=float((probs*odds*100.0).sum())
    first=selected.iloc[0]
    novel_mask=pd.to_numeric(selected["ticket_novel_count"],errors="coerce").fillna(0)>0
    row={
        "year":int(first["year"]),
        "race_id":str(first["race_id"]),
        "race_date":str(first["race_date"]),
        "bet_type":str(first["bet_type"]),
        "template":str(first["template"]),
        "edge_threshold":float(threshold),
        "action_key":f"{first['template']}@{float(threshold):.2f}",
        "ticket_count":int(n),
        "predicted_probability_mean":float(probs.mean()),
        "predicted_probability_min":float(probs.min()),
        "predicted_probability_max":float(probs.max()),
        "predicted_probability_sum":float(probs.sum()),
        "final_odds_mean":float(odds.mean()),
        "final_odds_min":float(odds.min()),
        "final_odds_max":float(odds.max()),
        "final_odds_p50":percentile(odds,0.50),
        "edge_mean":float(edge.mean()),
        "edge_min":float(edge.min()),
        "edge_max":float(edge.max()),
        "edge_p50":percentile(edge,0.50),
        "expected_return_yen":exp_return,
        "expected_roi":exp_return/stake if stake else 0.0,
        "expected_profit_per_100_ticketscale":100.0*((exp_return/stake)-1.0) if stake else -100.0,
        "novel_ticket_share":float(novel_mask.mean()),
        "gate_alert":float(pd.to_numeric(selected["gate_alert"],errors="coerce").fillna(0).max()),
        "gate_score":float(pd.to_numeric(selected["gate_score_alert_only"],errors="coerce").fillna(0).max()),
        "candidate_pool_size":float(pd.to_numeric(selected["candidate_pool_size"],errors="coerce").fillna(0).max()),
        "seven_union_count":float(pd.to_numeric(selected["seven_union_count"],errors="coerce").fillna(0).max()),
        "novel_pool_count":float(pd.to_numeric(selected["novel_pool_count"],errors="coerce").fillna(0).max()),
        "cw_top1_max_vote_share":float(pd.to_numeric(selected["cw_top1_max_vote_share"],errors="coerce").fillna(0).max()),
        "cw_top3_jaccard":float(pd.to_numeric(selected["cw_top3_jaccard"],errors="coerce").fillna(0).max()),
        "cw_top6_jaccard":float(pd.to_numeric(selected["cw_top6_jaccard"],errors="coerce").fillna(0).max()),
        "cw_rank_diff_mean":float(pd.to_numeric(selected["cw_rank_diff_mean"],errors="coerce").fillna(0).max()),
        "cw_rank_std_mean":float(pd.to_numeric(selected["cw_rank_std_mean"],errors="coerce").fillna(0).max()),
        "cw_prob_std_mean":float(pd.to_numeric(selected["cw_prob_std_mean"],errors="coerce").fillna(0).max()),
        "cw_prob_std_max":float(pd.to_numeric(selected["cw_prob_std_max"],errors="coerce").fillna(0).max()),
        "actual_return_yen":ret,
        "actual_profit_yen":ret-stake,
        "actual_roi_pct":100.0*ret/stake if stake else 0.0,
        "actual_positive_profit":1 if ret>stake else 0,
        "actual_hit":1 if ret>0 else 0,
        "actual_winning_tickets":int((pd.to_numeric(selected["return_yen_per100"],errors="coerce").fillna(0)>0).sum()),
    }
    for k,v in market_gap.get(str(first["race_id"]),{}).items():
        row[k]=v
    for k,v in novel_gap.get(str(first["race_id"]),{}).items():
        row[k]=v
    for k in (
        "market_gap_seven_top1_market_rank","market_gap_seven_top1_final_odds",
        "market_gap_market_favorite_seven_rank","market_gap_market_favorite_in_seven_union",
        "market_gap_top1_agreement","market_gap_top3_overlap_count","market_gap_top3_jaccard",
        "market_gap_seven_market_spearman","market_gap_k2_novel_count",
        "k2_novel_market_count","k2_novel_rank7plus_count","k2_novel_50_100_count",
        "k2_novel_10_20_count","k2_novel_100plus_count","k2_novel_odds_mean",
        "k2_novel_odds_min","k2_novel_odds_max","k2_novel_market_rank_mean",
        "k2_novel_market_rank_max","k2_novel_dual_source_count",
    ):
        row.setdefault(k,0.0)
    return row


def generate_strategy_ledger(dataset_dir,manifest,thresholds,market_gap,novel_gap,work_dir):
    rows=[]
    quality=[]
    templates=sorted(manifest["templates"])
    for ti,template in enumerate(templates):
        info=manifest["templates"][template]
        path=Path(dataset_dir)/info["file"]
        df=load_template(path)
        df["hit"]=df["hit"].astype(bool)
        df["odds"]=pd.to_numeric(df["odds"],errors="coerce")
        df["return_yen_per100"]=pd.to_numeric(df["return_yen_per100"],errors="coerce").fillna(0.0)
        df["race_month"]=pd.to_numeric(
            df["race_date"].astype(str).str.slice(5,7),errors="coerce"
        ).fillna(0.0)
        cols=feature_columns(df)
        if any("odds" in c.lower() for c in cols):
            raise ValueError(f"P(hit) odds leakage: {template}")
        for yi,test_year in enumerate(OOS_YEARS):
            train=df[df["year"]<test_year].copy()
            test=df[df["year"]==test_year].copy()
            if train.empty or test.empty:
                continue
            split=chronological_split(train)
            if split is None:
                continue
            fit,cal,fit_races,cal_races=split
            yfit=fit["hit"].astype(int).to_numpy()
            ycal=cal["hit"].astype(int).to_numpy()
            if len(set(yfit.tolist()))<2:
                continue
            xfit,(xcal,xtest)=encode_fit_other(fit,[cal,test],cols)
            model=make_model(91000+ti*100+yi)
            model.fit(xfit,yfit)
            pcal=model.predict_proba(xcal)[:,1]
            ptest0=model.predict_proba(xtest)[:,1]
            ptest,cal_meta=calibrate(ycal,pcal,ptest0)
            pred=test.copy()
            pred["predicted_probability"]=ptest
            pred["edge"]=pred["predicted_probability"]*pred["odds"]-1.0
            quality.append({
                "template":template,
                "bet_type":info["bet_type"],
                "test_year":test_year,
                "train_rows":len(train),
                "fit_rows":len(fit),
                "cal_rows":len(cal),
                "fit_races":fit_races,
                "cal_races":cal_races,
                "feature_count":int(xfit.shape[1]),
                "calibration_mode":cal_meta.get("mode"),
            })
            for (_,rid),grp in pred.groupby(["year","race_id"],sort=False):
                for threshold in thresholds:
                    row=strategy_row(grp,threshold,market_gap,novel_gap)
                    if row is not None:
                        rows.append(row)
        del df
    ledger=pd.DataFrame(rows)
    if ledger.empty:
        raise SystemExit("empty strategy ledger")
    if any(int(y)>=2026 for y in ledger["year"].unique()):
        raise SystemExit("2026 leakage into strategy ledger")
    work=Path(work_dir)
    work.mkdir(parents=True,exist_ok=True)
    with gzip.open(work/"strategy-ledger.jsonl.gz","wt",encoding="utf-8") as fh:
        for row in ledger.to_dict("records"):
            fh.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    return ledger,quality


def action_feature_columns(df):
    excluded={
        "year","race_id","race_date","action_key",
        "actual_return_yen","actual_profit_yen","actual_roi_pct",
        "actual_positive_profit","actual_hit","actual_winning_tickets",
        "expected_return_yen",
    }
    cols=[c for c in df.columns if c not in excluded]
    bad=[c for c in cols if any(tok in c.lower() for tok in FORBIDDEN_ROUTER_TOKENS)]
    if bad:
        raise ValueError(f"router action outcome leakage columns: {sorted(bad)}")
    return cols


def encode_router(train,test,cols,categorical):
    cats=[c for c in cols if c in categorical]
    nums=[c for c in cols if c not in cats]
    xtr_num=train[nums].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    xte_num=test[nums].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    if cats:
        tr_cat=pd.get_dummies(train[cats].astype("string").fillna("__MISSING__"),columns=cats,dtype=float)
        te_cat=pd.get_dummies(test[cats].astype("string").fillna("__MISSING__"),columns=cats,dtype=float)
    else:
        tr_cat=pd.DataFrame(index=train.index)
        te_cat=pd.DataFrame(index=test.index)
    xtr=pd.concat([xtr_num.reset_index(drop=True),tr_cat.reset_index(drop=True)],axis=1)
    xte=pd.concat([xte_num.reset_index(drop=True),te_cat.reset_index(drop=True)],axis=1)
    xte=xte.reindex(columns=xtr.columns,fill_value=0.0)
    return xtr,xte


def choose_simple(test):
    chosen=[]
    for rid,grp in test.groupby("race_id",sort=False):
        g=grp[grp["expected_roi"]>1.0].copy()
        if g.empty:
            continue
        g=g.sort_values(
            ["expected_roi","expected_profit_per_100_ticketscale","ticket_count","action_key"],
            ascending=[False,False,True,True],
        )
        chosen.append(g.iloc[0].to_dict())
    return chosen


def make_utility_model(seed):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=160,
        learning_rate=0.035,
        num_leaves=19,
        min_child_samples=80,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=4.0,
        reg_alpha=0.5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def choose_utility(train,test,seed):
    cols=action_feature_columns(train)
    xtr,xte=encode_router(train,test,cols,ROUTER_CATEGORICAL)
    y=train["actual_positive_profit"].astype(int).to_numpy()
    if len(set(y.tolist()))<2:
        return [],{"status":"SINGLE_CLASS","feature_count":len(xtr.columns)}
    model=make_utility_model(seed)
    model.fit(xtr,y)
    prob=model.predict_proba(xte)[:,1]
    scored=test.copy()
    scored["router_probability"]=prob
    scored["router_score"]=scored["router_probability"]*scored["expected_roi"]
    chosen=[]
    for rid,grp in scored.groupby("race_id",sort=False):
        g=grp[grp["router_probability"]>=0.50].copy()
        if g.empty:
            continue
        g=g.sort_values(
            ["router_score","router_probability","expected_roi","ticket_count","action_key"],
            ascending=[False,False,False,True,True],
        )
        chosen.append(g.iloc[0].to_dict())
    return chosen,{"status":"OK","feature_count":len(xtr.columns),"train_rows":len(train),"positive_rate":float(y.mean())}


def race_frame(actions,with_target=True):
    rows=[]
    for rid,grp in actions.groupby("race_id",sort=False):
        first=grp.iloc[0]
        row={
            "year":int(first["year"]),
            "race_id":str(rid),
            "race_date":str(first["race_date"]),
        }
        common=[
            c for c in grp.columns
            if c.startswith("market_gap_") or c.startswith("k2_novel_")
        ]
        for c in common:
            row[c]=finite(first.get(c),0.0)
        for c in (
            "gate_alert","gate_score","candidate_pool_size","seven_union_count",
            "novel_pool_count","cw_top1_max_vote_share","cw_top3_jaccard",
            "cw_top6_jaccard","cw_rank_diff_mean","cw_rank_std_mean",
            "cw_prob_std_mean","cw_prob_std_max"
        ):
            row[c]=finite(first.get(c),0.0)
        for bet in BET_TYPES:
            b=grp[grp["bet_type"]==bet]
            prefix=bet.lower()+"_"
            row[prefix+"action_count"]=float(len(b))
            if len(b):
                row[prefix+"max_expected_roi"]=float(b["expected_roi"].max())
                row[prefix+"max_expected_profit_per100"]=float(b["expected_profit_per_100_ticketscale"].max())
                row[prefix+"min_ticket_count"]=float(b["ticket_count"].min())
                row[prefix+"max_novel_ticket_share"]=float(b["novel_ticket_share"].max())
            else:
                row[prefix+"max_expected_roi"]=0.0
                row[prefix+"max_expected_profit_per100"]=0.0
                row[prefix+"min_ticket_count"]=0.0
                row[prefix+"max_novel_ticket_share"]=0.0
        row["overall_max_expected_roi"]=float(grp["expected_roi"].max())
        row["overall_action_count"]=float(len(grp))
        if with_target:
            profitable=grp[grp["actual_roi_pct"]>100.0].copy()
            if profitable.empty:
                row["direct_target"]="SKIP"
            else:
                profitable=profitable.sort_values(
                    ["actual_roi_pct","actual_profit_yen","ticket_count","action_key"],
                    ascending=[False,False,True,True],
                )
                row["direct_target"]=str(profitable.iloc[0]["bet_type"])
        rows.append(row)
    return pd.DataFrame(rows)


def choose_direct(train_actions,test_actions,seed):
    tr=race_frame(train_actions,with_target=True)
    te=race_frame(test_actions,with_target=False)
    features=[c for c in tr.columns if c not in {"year","race_id","race_date","direct_target"}]
    bad=[c for c in features if any(tok in c.lower() for tok in FORBIDDEN_ROUTER_TOKENS)]
    if bad:
        raise ValueError(f"direct router outcome leakage: {bad}")
    xtr=tr[features].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    xte=te[features].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    y=tr["direct_target"].astype(str)
    classes=sorted(y.unique())
    if len(classes)<2:
        return [],{"status":"SINGLE_CLASS","classes":classes}
    model=lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=180,
        learning_rate=0.035,
        num_leaves=19,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=4.0,
        reg_alpha=0.5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(xtr,y)
    pred=model.predict(xte)
    class_prob=model.predict_proba(xte)
    chosen=[]
    for i,race in te.reset_index(drop=True).iterrows():
        label=str(pred[i])
        if label=="SKIP":
            continue
        rid=str(race["race_id"])
        g=test_actions[(test_actions["race_id"].astype(str)==rid)&(test_actions["bet_type"]==label)].copy()
        if g.empty:
            continue
        g=g.sort_values(
            ["expected_roi","expected_profit_per_100_ticketscale","ticket_count","action_key"],
            ascending=[False,False,True,True],
        )
        row=g.iloc[0].to_dict()
        row["router_probability"]=float(np.max(class_prob[i]))
        row["direct_predicted_bet_type"]=label
        chosen.append(row)
    return chosen,{"status":"OK","feature_count":len(features),"classes":classes,"train_races":len(tr)}


def selections_df(chosen,router_name,test_year):
    rows=[]
    for r in chosen:
        rows.append({
            "router":router_name,
            "test_year":test_year,
            "race_id":r["race_id"],
            "race_date":r["race_date"],
            "bet_type":r["bet_type"],
            "template":r["template"],
            "edge_threshold":r["edge_threshold"],
            "action_key":r["action_key"],
            "ticket_count":r["ticket_count"],
            "expected_roi":r["expected_roi"],
            "novel_ticket_share":r["novel_ticket_share"],
            "gate_alert":r["gate_alert"],
            "gate_score":r["gate_score"],
            "actual_return_yen":r["actual_return_yen"],
            "actual_profit_yen":r["actual_profit_yen"],
            "actual_roi_pct":r["actual_roi_pct"],
            "actual_hit":r["actual_hit"],
            "router_probability":r.get("router_probability"),
        })
    return pd.DataFrame(rows)


def max_drawdown_by_race(df):
    if df.empty:
        return 0.0
    x=df.copy()
    x["stake_yen"]=100.0*pd.to_numeric(x["ticket_count"],errors="coerce").fillna(0.0)
    x=x.sort_values(["race_date","race_id"])
    cumulative=0.0
    peak=0.0
    dd=0.0
    for r in x.itertuples(index=False):
        cumulative+=float(r.actual_return_yen-r.stake_yen)
        peak=max(peak,cumulative)
        dd=max(dd,peak-cumulative)
    return dd


def concentration_metrics(df):
    if df.empty:
        return {
            "top1_return_share_pct":None,"top3_return_share_pct":None,
            "top5_return_share_pct":None,"top10_return_share_pct":None,
            "roi_zero_top1_pct":None,"roi_zero_top3_pct":None,"roi_zero_top5_pct":None,
        }
    stake=float((100.0*pd.to_numeric(df["ticket_count"],errors="coerce").fillna(0.0)).sum())
    returns=sorted([float(x) for x in df["actual_return_yen"] if float(x)>0],reverse=True)
    total=float(sum(returns))
    out={}
    for k in (1,3,5,10):
        share=100.0*sum(returns[:k])/total if total>0 else None
        out[f"top{k}_return_share_pct"]=share
    for k in (1,3,5):
        ret=total-sum(returns[:k])
        out[f"roi_zero_top{k}_pct"]=100.0*ret/stake if stake else None
    return out


def metrics(df,router,year):
    if df.empty:
        return {
            "router":router,"test_year":year,"bought_races":0,"bought_tickets":0,
            "stake_yen":0.0,"return_yen":0.0,"profit_yen":0.0,"roi_pct":None,
            "max_drawdown_yen":0.0,"positive_race_rate":0.0,"race_hit_rate":0.0,
            **concentration_metrics(df),
        }
    tickets=int(pd.to_numeric(df["ticket_count"],errors="coerce").fillna(0).sum())
    stake=100.0*tickets
    ret=float(pd.to_numeric(df["actual_return_yen"],errors="coerce").fillna(0.0).sum())
    positive=float((pd.to_numeric(df["actual_profit_yen"],errors="coerce").fillna(0.0)>0).mean())
    hit=float((pd.to_numeric(df["actual_hit"],errors="coerce").fillna(0)>0).mean())
    counts=df["bet_type"].value_counts().to_dict()
    return {
        "router":router,
        "test_year":year,
        "bought_races":int(df["race_id"].nunique()),
        "bought_tickets":tickets,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown_by_race(df),
        "positive_race_rate":positive,
        "race_hit_rate":hit,
        "gate_race_share":float(pd.to_numeric(df["gate_alert"],errors="coerce").fillna(0).mean()),
        "k2_used_race_share":float((pd.to_numeric(df["novel_ticket_share"],errors="coerce").fillna(0)>0).mean()),
        "win_races":int(counts.get("WIN",0)),
        "quinella_races":int(counts.get("QUINELLA",0)),
        "exacta_races":int(counts.get("EXACTA",0)),
        "trio_races":int(counts.get("TRIO",0)),
        "trifecta_races":int(counts.get("TRIFECTA",0)),
        **concentration_metrics(df),
    }


def segment_metrics(df,router,year):
    out=[]
    segments=[
        ("ALL",pd.Series(True,index=df.index)),
        ("GATE",pd.to_numeric(df["gate_alert"],errors="coerce").fillna(0)>0),
        ("NON_GATE",pd.to_numeric(df["gate_alert"],errors="coerce").fillna(0)<=0),
        ("K2_USED",pd.to_numeric(df["novel_ticket_share"],errors="coerce").fillna(0)>0),
        ("NO_K2",pd.to_numeric(df["novel_ticket_share"],errors="coerce").fillna(0)<=0),
    ]
    for name,mask in segments:
        m=metrics(df[mask].copy(),router,year)
        m["segment"]=name
        out.append(m)
    return out


def choose_research_leader(metric_rows,contract):
    q=contract["research_leader_selection"]["qualification"]
    by=defaultdict(dict)
    for r in metric_rows:
        if r.get("segment")!="ALL":
            continue
        by[r["router"]][int(r["test_year"])]=r
    candidates=[]
    for router in ROUTER_NAMES:
        years=by.get(router,{})
        if any(y not in years for y in EVAL_YEARS):
            continue
        yr=[years[y] for y in EVAL_YEARS]
        qualified=all(
            r["roi_pct"] is not None
            and float(r["roi_pct"])>=float(q["min_roi_pct_each_eval_year"])
            and int(r["bought_races"])>=int(q["min_bought_races_each_eval_year"])
            for r in yr
        )
        stake=sum(float(r["stake_yen"]) for r in yr)
        ret=sum(float(r["return_yen"]) for r in yr)
        candidates.append({
            "router":router,
            "qualified":qualified,
            "profit_2024":years[2024]["profit_yen"],
            "roi_2024":years[2024]["roi_pct"],
            "races_2024":years[2024]["bought_races"],
            "profit_2025":years[2025]["profit_yen"],
            "roi_2025":years[2025]["roi_pct"],
            "races_2025":years[2025]["bought_races"],
            "combined_profit_yen":sum(float(r["profit_yen"]) for r in yr),
            "combined_roi_pct":100.0*ret/stake if stake else None,
            "min_year_roi_pct":min(float(r["roi_pct"]) for r in yr),
            "sum_max_drawdown_yen":sum(float(r["max_drawdown_yen"]) for r in yr),
            "mean_top5_return_share_pct":float(np.mean([
                float(r["top5_return_share_pct"]) for r in yr
                if r["top5_return_share_pct"] is not None
            ])),
            "mean_roi_zero_top1_pct":float(np.mean([
                float(r["roi_zero_top1_pct"]) for r in yr
                if r["roi_zero_top1_pct"] is not None
            ])),
        })
    qualified=[r for r in candidates if r["qualified"]]
    if not qualified:
        return candidates,{"status":"NO_QUALIFIED_ROUTER"}
    ranked=sorted(
        qualified,
        key=lambda r:(
            -r["combined_profit_yen"],
            -r["min_year_roi_pct"],
            -r["mean_roi_zero_top1_pct"],
            r["mean_top5_return_share_pct"],
            r["sum_max_drawdown_yen"],
            r["router"],
        )
    )
    return candidates,{"status":"RESEARCH_LEADER","router":ranked[0]["router"],**ranked[0]}


def main():
    a=parse_args()
    contract=read_json(a.contract)
    manifest=read_json(Path(a.dataset_dir)/"manifest.json")
    if contract.get("contract")!="L2_ROUTER_ARENA_V1":
        raise SystemExit("wrong router contract")
    if contract["scope"]["locked_years"]!=[2026]:
        raise SystemExit("2026 lock drift")
    if manifest.get("source_l15")!="L15_FIXED_V1":
        raise SystemExit("dataset upstream drift")
    if manifest.get("probability_model_uses_odds") is not False:
        raise SystemExit("P(hit) market separation drift")
    thresholds=[float(x) for x in contract["ticket_generation"]["edge_thresholds"]]
    market_gap=load_market_gap(a.market_gap_race_summary)
    novel_gap=load_novel_race_features(a.k2_novel_ledger)

    ledger,quality=generate_strategy_ledger(
        a.dataset_dir,manifest,thresholds,market_gap,novel_gap,a.work_dir
    )
    if set(map(int,ledger["year"].unique()))!={2023,2024,2025}:
        raise SystemExit(f"strategy OOS year regression: {sorted(ledger['year'].unique())}")

    selection_frames=[]
    fold_status=[]
    for test_year in EVAL_YEARS:
        train=ledger[ledger["year"]<test_year].copy()
        test=ledger[ledger["year"]==test_year].copy()
        if test_year==2024:
            train=train[train["year"]==2023]
        if test_year==2025:
            train=train[train["year"].isin([2023,2024])]
        if train.empty or test.empty:
            raise SystemExit(f"router fold empty test_year={test_year}")

        simple=selections_df(choose_simple(test),"SIMPLE_EXPECTED_ROI",test_year)
        selection_frames.append(simple)
        fold_status.append({
            "router":"SIMPLE_EXPECTED_ROI","test_year":test_year,
            "train_years":"NONE_DETERMINISTIC","status":"OK",
            "train_action_rows":0,"test_action_rows":len(test)
        })

        utility,umeta=choose_utility(train,test,120000+test_year)
        udf=selections_df(utility,"STRATEGY_UTILITY",test_year)
        selection_frames.append(udf)
        fold_status.append({
            "router":"STRATEGY_UTILITY","test_year":test_year,
            "train_years":"|".join(map(str,sorted(map(int,train["year"].unique())))),
            "test_action_rows":len(test),**umeta
        })

        direct,dmeta=choose_direct(train,test,130000+test_year)
        ddf=selections_df(direct,"DIRECT_BET_TYPE",test_year)
        selection_frames.append(ddf)
        fold_status.append({
            "router":"DIRECT_BET_TYPE","test_year":test_year,
            "train_years":"|".join(map(str,sorted(map(int,train["year"].unique())))),
            "test_action_rows":len(test),**dmeta
        })

    selections=pd.concat(selection_frames,ignore_index=True) if selection_frames else pd.DataFrame()
    metric_rows=[]
    if not selections.empty:
        for (router,year),df in selections.groupby(["router","test_year"],sort=True):
            metric_rows.extend(segment_metrics(df.copy(),router,int(year)))

    candidates,leader=choose_research_leader(metric_rows,contract)
    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"phit-fold-status.csv",quality)
    write_csv(out/"router-fold-status.csv",fold_status)
    write_csv(out/"router-selections.csv",selections.to_dict("records") if not selections.empty else [])
    write_csv(out/"router-metrics.csv",metric_rows)
    write_csv(out/"router-candidates.csv",candidates)

    summary={
        "contract":"L2_ROUTER_ARENA_RESULT_V1",
        "upstream":"L15_FIXED_V1",
        "templates":len(manifest["templates"]),
        "edge_thresholds":thresholds,
        "strategy_action_rows":int(len(ledger)),
        "strategy_oos_years":[2023,2024,2025],
        "router_eval_years":[2024,2025],
        "2025_is_holdout":False,
        "locked_years":[2026],
        "routers":list(ROUTER_NAMES),
        "research_leader":leader,
        "promotion":"RESEARCH_ONLY_NOT_L2_FIX",
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Router Arena V1\n\n"
        "Upstream L1/L1.5 is frozen. The 23 Bet Kings templates are unchanged.\n\n"
        "P(hit) remains odds-free. Final odds and Market Gap/K2 signals are introduced only "
        "after P(hit), at the strategy/Router layer.\n\n"
        "Router evaluation is walk-forward: train on 2023 -> test 2024, then train on "
        "2023-2024 -> test 2025. 2025 is development evidence, not an untouched holdout. "
        "2026 remains sealed.\n\n"
        "Three routers compete: deterministic SIMPLE_EXPECTED_ROI, action-level "
        "STRATEGY_UTILITY, and race-level DIRECT_BET_TYPE. Each race selects at most one "
        "strategy or SKIP. Flat stake is 100 yen per selected ticket; L3 bankroll sizing "
        "is not part of this run.\n",
        encoding="utf-8",
    )
    print("L2_ROUTER_ARENA_V1_READY")
    print(json.dumps({
        "strategy_action_rows":len(ledger),
        "selection_rows":len(selections),
        "metric_rows":len(metric_rows),
        "leader":leader,
        "out_dir":str(out),
    },ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
