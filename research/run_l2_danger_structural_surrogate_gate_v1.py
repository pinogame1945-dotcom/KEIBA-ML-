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
FRACTIONS=(0.10,0.20,0.30,0.40,0.50)
TICKET_PRICE=100.0

NUMERIC=[
    "gate_score","candidate_pool_size","seven_union_count","novel_pool_count",
    "field_size","distance_m","race_month",
    "cw_top1_max_vote_share","cw_top3_jaccard","cw_top6_jaccard",
    "cw_rank_diff_mean","cw_rank_std_mean","cw_prob_std_mean","cw_prob_std_max",
]
CATEGORICAL=[
    "venue_code","surface","race_class","discipline","direction",
    "selected_outsider_1","selected_outsider_2","selected_outsider_pair",
]
COHESION_SIGNS={
    "field_size":-1.0,
    "candidate_pool_size":-1.0,
    "seven_union_count":-1.0,
    "novel_pool_count":-1.0,
    "cw_rank_diff_mean":-1.0,
    "cw_rank_std_mean":-1.0,
    "cw_top6_jaccard":1.0,
    "cw_top3_jaccard":1.0,
    "cw_top1_max_vote_share":1.0,
}

EXPECTED_ALL_OLD_ELIGIBLE={
    2023:(65,217.92270531400968),
    2024:(70,142.10300429184548),
    2025:(55,266.6666666666667),
}

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1)
        out[int(y)]=p
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
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(p,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def build_rows(fixed,routers,root):
    date_pairs=defaultdict(list)
    for y in YEARS:
        for rid in fixed[y]:
            d=str(routers[y][rid].get("race_date") or "")[:10]
            if len(d)!=10:
                raise SystemExit(f"bad race date race={rid}")
            date_pairs[d].append((y,rid))

    rows=[]
    for di,date in enumerate(sorted(date_pairs),1):
        pairs=date_pairs[date]
        wanted={rid for _,rid in pairs}
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        odds=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for year,rid in sorted(pairs):
            fr=fixed[year][rid]
            rr=routers[year][rid]
            pack=day.get(rid)
            oddsrec=odds.get(rid)
            if pack is None or oddsrec is None:
                raise SystemExit(f"missing race/odds row race={rid}")
            novel=[str(x) for x in fr.get("novel_horse_ids") or []]
            if not novel:
                continue
            anchors=[str(x) for x in fr.get("seven_anchor_horse_ids") or []]
            candidates=[str(x) for x in fr.get("candidate_horse_ids") or []]
            if len(anchors)!=2:
                raise SystemExit(f"bad anchors race={rid}")
            horse_no=horse_number_map(pack)
            missing=set(anchors+novel+candidates)-set(horse_no)
            if missing:
                raise SystemExit(f"horse number missing race={rid} sample={sorted(missing)[:5]}")
            a1,a2=horse_no[anchors[0]],horse_no[anchors[1]]
            odds_map=decode_odds(oddsrec)
            payouts,present=payout_map(pack)

            old_flags=[]
            race_return=0.0
            for hid in novel:
                h=horse_no[hid]
                win_ok=("WIN",(h,)) in odds_map
                exacta_ok=any(
                    ("EXACTA",(h,horse_no[x])) in odds_map
                    for x in candidates if x!=hid
                )
                tri1=(a1,a2,h)
                tri2=(a2,a1,h)
                trifecta_ok=(
                    ("TRIFECTA",tri1) in odds_map and
                    ("TRIFECTA",tri2) in odds_map
                )
                old_ok=(
                    "EXACTA" in present and
                    "TRIFECTA" in present and
                    win_ok and exacta_ok and trifecta_ok
                )
                old_flags.append(old_ok)
                race_return+=float(payouts.get(("TRIFECTA",tri1),0.0))
                race_return+=float(payouts.get(("TRIFECTA",tri2),0.0))

            meta=rr.get("race") or {}
            cons=rr.get("consensus") or {}
            outs=list(fr.get("selected_outsiders") or [])
            stake=2*TICKET_PRICE*len(novel)
            rows.append({
                "year":year,
                "race_id":rid,
                "race_date":date,
                "old_all_eligible":int(all(old_flags)),
                "old_any_eligible":int(any(old_flags)),
                "stake_yen":stake,
                "return_yen":race_return,
                "profit_yen":race_return-stake,
                "hit":int(race_return>0),
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
            })
        if di%25==0:
            print(f"SURROGATE_BUILD_PROGRESS dates={di}/{len(date_pairs)} races={len(rows)}",flush=True)
    return pd.DataFrame(rows)

def max_drawdown(sel):
    if sel.empty:
        return 0.0
    cumulative=0.0
    peak=0.0
    maxdd=0.0
    for r in sel.sort_values(["race_date","race_id"]).itertuples(index=False):
        cumulative+=float(r.return_yen-r.stake_yen)
        peak=max(peak,cumulative)
        maxdd=max(maxdd,peak-cumulative)
    return maxdd

def concentration(sel):
    if sel.empty:
        return {
            "largest_race_return_yen":0.0,
            "top1_return_share_pct":None,
            "top5_return_share_pct":None,
            "roi_without_top1_pct":None,
        }
    rets=sorted([float(x) for x in sel["return_yen"]],reverse=True)
    total_ret=sum(rets)
    total_stake=float(sel["stake_yen"].sum())
    top1=rets[0] if rets else 0.0
    top5=sum(rets[:5])
    return {
        "largest_race_return_yen":top1,
        "top1_return_share_pct":100*top1/total_ret if total_ret else None,
        "top5_return_share_pct":100*top5/total_ret if total_ret else None,
        "roi_without_top1_pct":100*(total_ret-top1)/total_stake if total_stake else None,
    }

def metric_row(sel,source,model,test_year,target_fraction,cutoff):
    stake=float(sel["stake_yen"].sum()) if len(sel) else 0.0
    ret=float(sel["return_yen"].sum()) if len(sel) else 0.0
    return {
        "model":model,
        "test_year":test_year,
        "target_train_top_fraction":target_fraction,
        "score_cutoff":cutoff,
        "source_races":len(source),
        "selected_races":len(sel),
        "selected_race_pct":100*len(sel)/len(source) if len(source) else 0.0,
        "old_all_eligible_baseline_pct":100*float(source["old_all_eligible"].mean()) if len(source) else None,
        "old_all_eligible_selected_pct":100*float(sel["old_all_eligible"].mean()) if len(sel) else None,
        "hit_races":int(sel["hit"].sum()) if len(sel) else 0,
        "hit_rate_pct":100*float(sel["hit"].mean()) if len(sel) else 0.0,
        "tickets":int(round(stake/TICKET_PRICE)),
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(sel),
        **concentration(sel),
    }

def fit_equal_cohesion(train,test):
    tr_score=np.zeros(len(train),dtype=float)
    te_score=np.zeros(len(test),dtype=float)
    used=0
    for feat,sign in COHESION_SIGNS.items():
        tr=pd.to_numeric(train[feat],errors="coerce").fillna(0.0).to_numpy(dtype=float)
        te=pd.to_numeric(test[feat],errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mu=float(np.mean(tr))
        sd=float(np.std(tr))
        if not math.isfinite(sd) or sd<1e-12:
            continue
        tr_score+=sign*(tr-mu)/sd
        te_score+=sign*(te-mu)/sd
        used+=1
    if used==0:
        raise SystemExit("no cohesion features")
    return tr_score/used,te_score/used

def fit_logit(train,test):
    prep=ColumnTransformer([
        ("num",Pipeline([("scale",StandardScaler())]),NUMERIC),
        ("cat",OneHotEncoder(handle_unknown="ignore"),CATEGORICAL),
    ])
    model=Pipeline([
        ("prep",prep),
        ("clf",LogisticRegression(max_iter=2000,C=0.5,class_weight="balanced")),
    ])
    y=train["old_all_eligible"].astype(int)
    model.fit(train[NUMERIC+CATEGORICAL],y)
    tr=model.predict_proba(train[NUMERIC+CATEGORICAL])[:,1]
    te=model.predict_proba(test[NUMERIC+CATEGORICAL])[:,1]
    return tr,te

def auc_row(model,test_year,test,score):
    y=test["old_all_eligible"].astype(int).to_numpy()
    return {
        "model":model,
        "test_year":test_year,
        "races":len(test),
        "positive_races":int(y.sum()),
        "positive_rate_pct":100*float(y.mean()),
        "roc_auc":float(roc_auc_score(y,score)),
        "pr_auc":float(average_precision_score(y,score)),
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

    df=build_rows(fixed,routers,Path(a.backfill_root))
    if any(df["year"]>=2026):
        raise SystemExit("2026 leakage")

    for y,(expected_races,expected_roi) in EXPECTED_ALL_OLD_ELIGIBLE.items():
        s=df[(df["year"]==y)&(df["old_all_eligible"]==1)]
        roi=100*float(s["return_yen"].sum())/float(s["stake_yen"].sum())
        if len(s)!=expected_races or abs(roi-expected_roi)>1e-9:
            raise SystemExit(f"reconstruction drift y={y} races={len(s)} roi={roi}")

    metrics=[]
    aucs=[]
    folds=[]
    for test_year in TEST_YEARS:
        train=df[df["year"]<test_year].copy().reset_index(drop=True)
        test=df[df["year"]==test_year].copy().reset_index(drop=True)
        if train.empty or test.empty:
            raise SystemExit(f"empty fold y={test_year}")
        if train["old_all_eligible"].nunique()<2:
            raise SystemExit(f"single target train y={test_year}")

        folds.append({
            "test_year":test_year,
            "train_years":"|".join(map(str,sorted(train["year"].unique()))),
            "train_races":len(train),
            "test_races":len(test),
            "train_positive_races":int(train["old_all_eligible"].sum()),
            "test_positive_races":int(test["old_all_eligible"].sum()),
        })

        for model in ("COHESION_EQUAL","SURROGATE_LOGIT"):
            if model=="COHESION_EQUAL":
                tr_score,te_score=fit_equal_cohesion(train,test)
            else:
                tr_score,te_score=fit_logit(train,test)
            aucs.append(auc_row(model,test_year,test,te_score))
            scored=test.copy()
            scored["score"]=te_score

            for frac in FRACTIONS:
                cutoff=float(np.quantile(tr_score,1.0-frac))
                sel=scored[scored["score"]>=cutoff].copy()
                metrics.append(metric_row(sel,test,model,test_year,frac,cutoff))

        metrics.append(metric_row(test,test,"ALL_DANGER",test_year,1.0,float("-inf")))

    stability=[]
    for model in ("COHESION_EQUAL","SURROGATE_LOGIT"):
        for frac in FRACTIONS:
            rs=[
                r for r in metrics
                if r["model"]==model and
                abs(float(r["target_train_top_fraction"])-frac)<1e-12
            ]
            if len(rs)!=3:
                raise SystemExit(f"missing metrics {model} {frac}")
            stake=sum(r["stake_yen"] for r in rs)
            ret=sum(r["return_yen"] for r in rs)
            stability.append({
                "model":model,
                "target_train_top_fraction":frac,
                "roi_2023":next(r["roi_pct"] for r in rs if r["test_year"]==2023),
                "roi_2024":next(r["roi_pct"] for r in rs if r["test_year"]==2024),
                "roi_2025":next(r["roi_pct"] for r in rs if r["test_year"]==2025),
                "min_year_roi_pct":min(r["roi_pct"] for r in rs if r["roi_pct"] is not None),
                "combined_roi_pct":100*ret/stake if stake else None,
                "combined_profit_yen":ret-stake,
                "selected_races":sum(r["selected_races"] for r in rs),
                "min_selected_races":min(r["selected_races"] for r in rs),
                "min_roi_without_top1_pct":min(
                    r["roi_without_top1_pct"] for r in rs
                    if r["roi_without_top1_pct"] is not None
                ),
                "max_top1_return_share_pct":max(
                    r["top1_return_share_pct"] for r in rs
                    if r["top1_return_share_pct"] is not None
                ),
                "all_years_roi_100plus":int(all((r["roi_pct"] or 0)>=100 for r in rs)),
            })

    stability.sort(
        key=lambda r:(
            -r["all_years_roi_100plus"],
            -(r["min_year_roi_pct"] or -1e9),
            -(r["combined_profit_yen"] or -1e9),
        )
    )

    summary={
        "contract":"L2_DANGER_STRUCTURAL_SURROGATE_GATE_V1",
        "analysis_years":list(YEARS),
        "test_years":list(TEST_YEARS),
        "locked_years":list(LOCKED_YEARS),
        "ticket_rule":"For each selected danger race, buy A1->A2->every K2 novel and A2->A1->every K2 novel, 100 yen each.",
        "target":"old_all_eligible is diagnostic training target only; no odds/missingness is an input feature.",
        "models":{
            "COHESION_EQUAL":"Fixed-sign equal-weight standardized structural cohesion score; no target fitting.",
            "SURROGATE_LOGIT":"Logistic regression predicting old_all_eligible from pre-race structural features only."
        },
        "threshold_rule":"Test-year cutoff is computed from TRAIN score quantiles only; no test-year quantile or outcome tuning.",
        "fractions":list(FRACTIONS),
        "folds":folds,
        "membership_quality":aucs,
        "stability":stability,
        "interpretation_guard":"Diagnostic search only. Market-data completeness is not a production betting feature.",
        "production_promotion":False,
    }

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"folds.csv",folds)
    write_csv(out/"membership-quality.csv",aucs)
    write_csv(out/"selection-metrics.csv",metrics)
    write_csv(out/"stability.csv",stability)
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    (out/"README.md").write_text(
        "# L2 Danger Structural Surrogate Gate V1\n\n"
        "Tests whether the strong old all-odds-complete danger-race slice has a reproducible pre-race structural signature. "
        "2022 is training-only for the 2023 fold; 2023-2025 are strict future-year tests. "
        "No odds, popularity, payout, result, hit, horse-number, or missingness feature is used for selection. "
        "Ticket returns are used only after selection for evaluation. 2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_DANGER_STRUCTURAL_SURROGATE_GATE_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
