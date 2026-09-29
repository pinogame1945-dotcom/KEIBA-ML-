#!/usr/bin/env python3
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from build_l2_bet_kings_dataset_v1 import (
    decode_odds,finite,horse_number_map,load_day,
    load_fixed_ledgers,load_odds_day,load_router,payout_map,
)

YEARS=(2022,2023,2024,2025)
TEST_YEARS=(2023,2024,2025)
LOCKED_YEARS=(2026,)
FRACTIONS=(0.05,0.10,0.15,0.20,0.30,0.40,0.50)
TICKET_PRICE=100.0

STRUCT_NUMERIC=[
    "gate_score","candidate_pool_size","seven_union_count","novel_pool_count",
    "field_size","distance_m","race_month",
    "cw_top1_max_vote_share","cw_top3_jaccard","cw_top6_jaccard",
    "cw_rank_diff_mean","cw_rank_std_mean","cw_prob_std_mean","cw_prob_std_max",
]
STRUCT_CATEGORICAL=[
    "venue_code","surface","race_class","discipline","direction",
    "selected_outsider_1","selected_outsider_2","selected_outsider_pair",
]
MARKET_NUMERIC=[
    "a1_win_odds","a2_win_odds","anchor_win_odds_mean","anchor_win_odds_max",
    "anchor_implied_sum","a1_market_rank","a2_market_rank",
    "k2_win_odds_min","k2_win_odds_mean","k2_win_odds_median","k2_win_odds_max",
    "k2_log_win_odds_mean","k2_implied_sum","k2_implied_mean",
    "k2_best_market_rank","k2_market_rank_mean",
    "k2_under10_share","k2_under20_share","k2_under50_share",
    "k2_best_vs_a1_ratio","k2_best_vs_anchor_worst_ratio",
    "k2_mean_vs_anchor_mean_ratio",
]

MODEL_SPECS={
    "STRUCT_PROFIT_EVENT":{
        "market":False,
        "value_weight":False,
    },
    "MARKET_PROFIT_EVENT":{
        "market":True,
        "value_weight":False,
    },
    "STRUCT_PROFIT_VALUE":{
        "market":False,
        "value_weight":True,
    },
    "MARKET_PROFIT_VALUE":{
        "market":True,
        "value_weight":True,
    },
}

def parse_args():
    p=argparse.ArgumentParser(description="Direct L2 buy/skip gate for fixed danger A1/A2 -> K2 trifecta rule.")
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_paths(items):
    out={}
    for spec in items:
        year,path=spec.split(":",1)
        out[int(year)]=path
    if set(out)!=set(YEARS):
        raise SystemExit(f"router years mismatch got={sorted(out)} expected={list(YEARS)}")
    return out

def write_csv(path,rows):
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        p.write_text("",encoding="utf-8")
        return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(p,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def safe_auc(y,p):
    y=np.asarray(y,dtype=int)
    if len(set(y.tolist()))<2:
        return None
    return float(roc_auc_score(y,p))

def safe_ap(y,p):
    y=np.asarray(y,dtype=int)
    if len(set(y.tolist()))<2:
        return None
    return float(average_precision_score(y,p))

def build_rows(fixed,routers,root):
    date_pairs=defaultdict(list)
    for y in YEARS:
        for rid in fixed[y]:
            d=str(routers[y][rid].get("race_date") or "")[:10]
            if len(d)!=10:
                raise SystemExit(f"bad race date race={rid}")
            date_pairs[d].append((y,rid))

    rows=[]
    counters=defaultdict(int)
    for di,date in enumerate(sorted(date_pairs),1):
        pairs=date_pairs[date]
        wanted={rid for _,rid in pairs}
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        odds_day=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for year,rid in sorted(pairs):
            fr=fixed[year][rid]
            rr=routers[year][rid]
            pack=day.get(rid)
            odds_rec=odds_day.get(rid)
            if pack is None or odds_rec is None:
                raise SystemExit(f"missing race/odds row race={rid}")

            novel=[str(x) for x in fr.get("novel_horse_ids") or []]
            if not novel:
                counters["no_k2_races"]+=1
                continue
            anchors=[str(x) for x in fr.get("seven_anchor_horse_ids") or []]
            candidates=[str(x) for x in fr.get("candidate_horse_ids") or []]
            if len(anchors)!=2:
                raise SystemExit(f"bad anchors race={rid}")

            horse_no=horse_number_map(pack)
            missing_horses=set(anchors+novel+candidates)-set(horse_no)
            if missing_horses:
                raise SystemExit(f"horse number missing race={rid} sample={sorted(missing_horses)[:5]}")

            odds_map=decode_odds(odds_rec)
            payouts,_=payout_map(pack)

            candidate_odds={}
            for hid in candidates:
                num=horse_no[hid]
                odd=odds_map.get(("WIN",(num,)))
                if odd is None or odd<=0:
                    raise SystemExit(f"WIN odds missing for candidate race={rid} horse={hid}")
                candidate_odds[hid]=float(odd)

            a1_hid,a2_hid=anchors
            a1_no,a2_no=horse_no[a1_hid],horse_no[a2_hid]
            a1_odd,a2_odd=candidate_odds[a1_hid],candidate_odds[a2_hid]
            k2_odds=np.array([candidate_odds[x] for x in novel],dtype=float)

            ordered_market=sorted(candidates,key=lambda h:(candidate_odds[h],h))
            market_rank={hid:i+1 for i,hid in enumerate(ordered_market)}
            k2_ranks=np.array([market_rank[x] for x in novel],dtype=float)

            race_return=0.0
            for hid in novel:
                h=horse_no[hid]
                race_return+=float(payouts.get(("TRIFECTA",(a1_no,a2_no,h)),0.0))
                race_return+=float(payouts.get(("TRIFECTA",(a2_no,a1_no,h)),0.0))
            stake=2*TICKET_PRICE*len(novel)
            profit=race_return-stake

            meta=rr.get("race") or {}
            cons=rr.get("consensus") or {}
            outs=list(fr.get("selected_outsiders") or [])
            anchor_mean=(a1_odd+a2_odd)/2.0
            anchor_worst=max(a1_odd,a2_odd)
            k2_min=float(np.min(k2_odds))
            k2_mean=float(np.mean(k2_odds))
            n=len(k2_odds)

            rows.append({
                "year":year,
                "race_id":rid,
                "race_date":date,
                "stake_yen":stake,
                "return_yen":race_return,
                "profit_yen":profit,
                "return_ratio":race_return/stake if stake else 0.0,
                "profit_label":int(profit>0),
                "hit_label":int(race_return>0),

                "gate_score":finite(fr.get("gate_score")) or 0.0,
                "candidate_pool_size":len(candidates),
                "seven_union_count":len(fr.get("seven_union_horse_ids") or []),
                "novel_pool_count":len(novel),
                "venue_code":meta.get("venue_code"),
                "surface":meta.get("surface"),
                "race_class":meta.get("race_class"),
                "discipline":meta.get("discipline"),
                "direction":meta.get("direction"),
                "distance_m":finite(meta.get("distance_m")) or 0.0,
                "field_size":finite(meta.get("field_size")) or 0.0,
                "race_month":int(date[5:7]),
                "cw_top1_max_vote_share":finite(cons.get("top1_max_vote_share")) or 0.0,
                "cw_top3_jaccard":finite(cons.get("top3_pairwise_jaccard_mean")) or 0.0,
                "cw_top6_jaccard":finite(cons.get("top6_pairwise_jaccard_mean")) or 0.0,
                "cw_rank_diff_mean":finite(cons.get("pairwise_rank_abs_diff_mean")) or 0.0,
                "cw_rank_std_mean":finite(cons.get("horse_rank_std_mean")) or 0.0,
                "cw_prob_std_mean":finite(cons.get("horse_probability_std_mean")) or 0.0,
                "cw_prob_std_max":finite(cons.get("horse_probability_std_max")) or 0.0,
                "selected_outsider_1":outs[0] if len(outs)>0 else "__NONE__",
                "selected_outsider_2":outs[1] if len(outs)>1 else "__NONE__",
                "selected_outsider_pair":"|".join(outs) if outs else "__NONE__",

                "a1_win_odds":a1_odd,
                "a2_win_odds":a2_odd,
                "anchor_win_odds_mean":anchor_mean,
                "anchor_win_odds_max":anchor_worst,
                "anchor_implied_sum":1.0/a1_odd+1.0/a2_odd,
                "a1_market_rank":float(market_rank[a1_hid]),
                "a2_market_rank":float(market_rank[a2_hid]),
                "k2_win_odds_min":k2_min,
                "k2_win_odds_mean":k2_mean,
                "k2_win_odds_median":float(np.median(k2_odds)),
                "k2_win_odds_max":float(np.max(k2_odds)),
                "k2_log_win_odds_mean":float(np.mean(np.log(k2_odds))),
                "k2_implied_sum":float(np.sum(1.0/k2_odds)),
                "k2_implied_mean":float(np.mean(1.0/k2_odds)),
                "k2_best_market_rank":float(np.min(k2_ranks)),
                "k2_market_rank_mean":float(np.mean(k2_ranks)),
                "k2_under10_share":float(np.sum(k2_odds<10.0))/n,
                "k2_under20_share":float(np.sum(k2_odds<20.0))/n,
                "k2_under50_share":float(np.sum(k2_odds<50.0))/n,
                "k2_best_vs_a1_ratio":k2_min/a1_odd,
                "k2_best_vs_anchor_worst_ratio":k2_min/anchor_worst,
                "k2_mean_vs_anchor_mean_ratio":k2_mean/anchor_mean,
            })
        if di%25==0:
            print(f"DIRECT_GATE_BUILD_PROGRESS dates={di}/{len(date_pairs)} races={len(rows)}",flush=True)
    return pd.DataFrame(rows),dict(counters)

def make_model(feature_market):
    nums=STRUCT_NUMERIC+(MARKET_NUMERIC if feature_market else [])
    prep=ColumnTransformer([
        ("num",Pipeline([("scale",StandardScaler())]),nums),
        ("cat",OneHotEncoder(handle_unknown="ignore"),STRUCT_CATEGORICAL),
    ])
    clf=LogisticRegression(
        max_iter=3000,
        C=0.35,
        class_weight="balanced",
        solver="lbfgs",
    )
    return Pipeline([("prep",prep),("clf",clf)]),nums+STRUCT_CATEGORICAL

def training_weights(train,value_weight):
    w=np.ones(len(train),dtype=float)
    if not value_weight:
        return w
    pos=train["profit_label"].astype(int).to_numpy()==1
    rr=pd.to_numeric(train["return_ratio"],errors="coerce").fillna(0.0).to_numpy(dtype=float)
    # Cap jackpot influence: positive examples receive at most 4x extra multiplier.
    bonus=np.minimum(3.0,np.log1p(np.maximum(rr,0.0)))
    w[pos]=1.0+bonus[pos]
    return w

def max_drawdown(sel):
    if sel.empty:
        return 0.0
    cum=0.0
    peak=0.0
    maxdd=0.0
    for r in sel.sort_values(["race_date","race_id"]).itertuples(index=False):
        cum+=float(r.profit_yen)
        peak=max(peak,cum)
        maxdd=max(maxdd,peak-cum)
    return maxdd

def concentration(sel):
    if sel.empty:
        return {
            "largest_race_return_yen":0.0,
            "top1_return_share_pct":None,
            "top5_return_share_pct":None,
            "roi_without_top1_pct":None,
        }
    race_returns=sorted([float(x) for x in sel["return_yen"]],reverse=True)
    total_ret=sum(race_returns)
    total_stake=float(sel["stake_yen"].sum())
    top1=race_returns[0] if race_returns else 0.0
    top5=sum(race_returns[:5])
    return {
        "largest_race_return_yen":top1,
        "top1_return_share_pct":100*top1/total_ret if total_ret else None,
        "top5_return_share_pct":100*top5/total_ret if total_ret else None,
        "roi_without_top1_pct":100*(total_ret-top1)/total_stake if total_stake else None,
    }

def metric_row(sel,source,model,test_year,frac,cutoff):
    stake=float(sel["stake_yen"].sum()) if len(sel) else 0.0
    ret=float(sel["return_yen"].sum()) if len(sel) else 0.0
    return {
        "model":model,
        "test_year":test_year,
        "target_train_top_fraction":frac,
        "score_cutoff":cutoff,
        "source_races":len(source),
        "selected_races":len(sel),
        "selected_race_pct":100*len(sel)/len(source) if len(source) else 0.0,
        "profit_races":int(sel["profit_label"].sum()) if len(sel) else 0,
        "profit_race_rate_pct":100*float(sel["profit_label"].mean()) if len(sel) else 0.0,
        "hit_races":int(sel["hit_label"].sum()) if len(sel) else 0,
        "tickets":int(round(stake/TICKET_PRICE)),
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(sel),
        **concentration(sel),
    }

def main():
    a=parse_args()
    paths=parse_paths(a.router_year)
    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={y:load_router(paths[y],y) for y in YEARS}
    for y in YEARS:
        missing=set(fixed[y])-set(routers[y])
        if missing:
            raise SystemExit(f"fixed alerts missing router y={y} sample={sorted(missing)[:5]}")

    df,counters=build_rows(fixed,routers,Path(a.backfill_root))
    if any(df["year"]>=2026):
        raise SystemExit("2026 leakage")
    expected_counts={2022:344,2023:346,2024:342,2025:346}
    for y,n in expected_counts.items():
        got=int((df["year"]==y).sum())
        if got!=n:
            raise SystemExit(f"K2 race universe drift y={y} got={got} expected={n}")

    metrics=[]
    quality=[]
    folds=[]
    selected_rows=[]
    baseline=[]
    for test_year in TEST_YEARS:
        train=df[df["year"]<test_year].copy().reset_index(drop=True)
        test=df[df["year"]==test_year].copy().reset_index(drop=True)
        ytr=train["profit_label"].astype(int)
        yte=test["profit_label"].astype(int)
        if ytr.nunique()<2:
            raise SystemExit(f"single target class train y={test_year}")

        folds.append({
            "test_year":test_year,
            "train_years":"|".join(map(str,sorted(train["year"].unique()))),
            "train_races":len(train),
            "train_profit_races":int(ytr.sum()),
            "test_races":len(test),
            "test_profit_races":int(yte.sum()),
            "test_profit_rate_pct":100*float(yte.mean()),
        })
        baseline.append(metric_row(test,test,"ALL_DANGER",test_year,1.0,float("-inf")))

        for model_name,spec in MODEL_SPECS.items():
            model,features=make_model(spec["market"])
            weights=training_weights(train,spec["value_weight"])
            model.fit(
                train[features],
                ytr,
                clf__sample_weight=weights,
            )
            tr_score=model.predict_proba(train[features])[:,1]
            te_score=model.predict_proba(test[features])[:,1]
            quality.append({
                "model":model_name,
                "test_year":test_year,
                "train_races":len(train),
                "train_profit_races":int(ytr.sum()),
                "test_races":len(test),
                "test_profit_races":int(yte.sum()),
                "roc_auc":safe_auc(yte,te_score),
                "pr_auc":safe_ap(yte,te_score),
                "feature_count_input":len(features),
            })
            scored=test.copy()
            scored["score"]=te_score

            for frac in FRACTIONS:
                cutoff=float(np.quantile(tr_score,1.0-frac))
                sel=scored[scored["score"]>=cutoff].copy()
                metrics.append(metric_row(sel,test,model_name,test_year,frac,cutoff))
                for row in sel.itertuples(index=False):
                    selected_rows.append({
                        "model":model_name,
                        "test_year":test_year,
                        "target_train_top_fraction":frac,
                        "score_cutoff":cutoff,
                        "score":float(row.score),
                        "race_id":row.race_id,
                        "race_date":row.race_date,
                        "stake_yen":float(row.stake_yen),
                        "return_yen":float(row.return_yen),
                        "profit_yen":float(row.profit_yen),
                        "profit_label":int(row.profit_label),
                    })

    metrics.extend(baseline)

    stability=[]
    for model_name in MODEL_SPECS:
        for frac in FRACTIONS:
            rs=[
                r for r in metrics
                if r["model"]==model_name and
                abs(float(r["target_train_top_fraction"])-frac)<1e-12
            ]
            if len(rs)!=3:
                raise SystemExit(f"metric coverage missing model={model_name} frac={frac}")
            stake=sum(r["stake_yen"] for r in rs)
            ret=sum(r["return_yen"] for r in rs)
            roi_wo=[
                r["roi_without_top1_pct"] if r["roi_without_top1_pct"] is not None else 0.0
                for r in rs
            ]
            shares=[
                r["top1_return_share_pct"] if r["top1_return_share_pct"] is not None else 100.0
                for r in rs
            ]
            stability.append({
                "model":model_name,
                "target_train_top_fraction":frac,
                "roi_2023":next(r["roi_pct"] for r in rs if r["test_year"]==2023),
                "roi_2024":next(r["roi_pct"] for r in rs if r["test_year"]==2024),
                "roi_2025":next(r["roi_pct"] for r in rs if r["test_year"]==2025),
                "min_year_roi_pct":min((r["roi_pct"] or 0.0) for r in rs),
                "combined_roi_pct":100*ret/stake if stake else None,
                "combined_profit_yen":ret-stake,
                "selected_races":sum(r["selected_races"] for r in rs),
                "min_selected_races":min(r["selected_races"] for r in rs),
                "min_roi_without_top1_pct":min(roi_wo),
                "max_top1_return_share_pct":max(shares),
                "all_years_roi_100plus":int(all((r["roi_pct"] or 0.0)>=100 for r in rs)),
                "all_years_roi_wo_top1_100plus":int(all(x>=100 for x in roi_wo)),
            })

    stability.sort(key=lambda r:(
        -r["all_years_roi_100plus"],
        -r["all_years_roi_wo_top1_100plus"],
        -(r["min_year_roi_pct"] or -1e9),
        -(r["combined_profit_yen"] or -1e9),
    ))

    summary={
        "contract":"L2_DANGER_DIRECT_BUY_GATE_V1",
        "analysis_years":list(YEARS),
        "test_years":list(TEST_YEARS),
        "locked_years":list(LOCKED_YEARS),
        "source_universe":"Every danger race with at least one K2 novel; no trifecta/exacta odds completeness filter.",
        "ticket_rule":"Buy A1->A2->every K2 novel and A2->A1->every K2 novel; 100 yen each.",
        "target":"profit_yen > 0 at race level.",
        "market_lane":"Uses final WIN odds only as L2 market features. No trifecta/exacta ticket odds, popularity field, payout, result, or missingness is an input.",
        "market_caveat":"FINAL_WIN_ODDS_PROXY is historical research only; live deployment still needs a pre-bet timestamp policy.",
        "models":MODEL_SPECS,
        "threshold_rule":"For each test year, selection cutoffs are quantiles of prior-year TRAIN scores only.",
        "fractions":list(FRACTIONS),
        "folds":folds,
        "quality":quality,
        "baseline":baseline,
        "stability":stability,
        "counters":counters,
        "production_promotion":False,
    }

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"folds.csv",folds)
    write_csv(out/"model-quality.csv",quality)
    write_csv(out/"selection-metrics.csv",metrics)
    write_csv(out/"stability.csv",stability)
    write_csv(out/"selected-races.csv",selected_rows)
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    (out/"README.md").write_text(
        "# L2 Danger Direct Buy Gate V1\n\n"
        "Directly learns race-level buy/skip for the fixed A1/A2 -> K2 trifecta rule. "
        "The source universe is every K2-eligible danger race; realized payouts are used only for the training target and evaluation. "
        "Two feature scopes are compared: structural-only and structural + final WIN-market features. "
        "Two objectives are compared: profit-event classification and capped value-weighted profit-event classification. "
        "No trifecta/exacta odds-completeness filtering is allowed. 2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_DANGER_DIRECT_BUY_GATE_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
