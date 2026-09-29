#!/usr/bin/env python3
import argparse,csv,itertools,json,math
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
from run_l2_danger_direct_buy_gate_v1 import (
    STRUCT_NUMERIC,STRUCT_CATEGORICAL,MARKET_NUMERIC,
)

YEARS=(2022,2023,2024,2025)
TEST_YEARS=(2023,2024,2025)
LOCKED_YEARS=(2026,)
ACTIONS=("BASE","K2_SECOND")
ACTION_PREFIX={"BASE":"base","K2_SECOND":"second"}
OBJECTIVES=("EVENT","VALUE")
FRACTIONS=(0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50)
TICKET_PRICE=100.0

EXTRA_MARKET_NUMERIC=[
    "a3_win_odds","a3_market_rank",
    "top3_king_win_odds_mean","top3_king_win_odds_max",
    "top3_king_implied_sum",
    "k2_best_vs_top3_worst_ratio","k2_mean_vs_top3_mean_ratio",
]
FEATURE_NUMERIC=STRUCT_NUMERIC+MARKET_NUMERIC+EXTRA_MARKET_NUMERIC
FEATURES=FEATURE_NUMERIC+STRUCT_CATEGORICAL

def parse_args():
    p=argparse.ArgumentParser(description="Walk-forward BASE/K2_SECOND/SKIP router for danger races.")
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1); out[int(y)]=p
    if set(out)!=set(YEARS):
        raise SystemExit(f"router years mismatch got={sorted(out)} expected={list(YEARS)}")
    return out

def write_csv(path,rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        p.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(p,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def safe_auc(y,p):
    vals=np.asarray(y,dtype=int)
    return float(roc_auc_score(vals,p)) if len(set(vals.tolist()))>1 else None

def safe_ap(y,p):
    vals=np.asarray(y,dtype=int)
    return float(average_precision_score(vals,p)) if len(set(vals.tolist()))>1 else None

def percentile_against_train(train_scores,test_scores):
    ref=np.sort(np.asarray(train_scores,dtype=float))
    x=np.asarray(test_scores,dtype=float)
    return np.searchsorted(ref,x,side="right")/max(1,len(ref))

def make_model():
    prep=ColumnTransformer([
        ("num",Pipeline([("scale",StandardScaler())]),FEATURE_NUMERIC),
        ("cat",OneHotEncoder(handle_unknown="ignore"),STRUCT_CATEGORICAL),
    ])
    clf=LogisticRegression(
        max_iter=3000,
        C=0.35,
        class_weight="balanced",
        solver="lbfgs",
    )
    return Pipeline([("prep",prep),("clf",clf)])

def action_prefix(action):
    try:
        return ACTION_PREFIX[action]
    except KeyError as exc:
        raise ValueError(f"unknown action: {action}") from exc

def training_weights(frame,objective,action):
    w=np.ones(len(frame),dtype=float)
    if objective=="EVENT":
        return w
    p=action_prefix(action)
    label=frame[f"{p}_profit_label"].astype(int).to_numpy()==1
    rr=pd.to_numeric(frame[f"{p}_return_ratio"],errors="coerce").fillna(0.0).to_numpy(dtype=float)
    bonus=np.minimum(3.0,np.log1p(np.maximum(rr,0.0)))
    w[label]=1.0+bonus[label]
    return w

def build_rows(fixed,routers,root):
    date_pairs=defaultdict(list)
    for y in YEARS:
        for rid in fixed[y]:
            d=str(routers[y][rid].get("race_date") or "")[:10]
            if len(d)!=10: raise SystemExit(f"bad race date race={rid}")
            date_pairs[d].append((y,rid))

    rows=[]; counters=defaultdict(int)
    for di,date in enumerate(sorted(date_pairs),1):
        pairs=date_pairs[date]; wanted={rid for _,rid in pairs}
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        odds_day=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)

        for year,rid in sorted(pairs):
            fr=fixed[year][rid]; rr=routers[year][rid]
            pack=day.get(rid); odds_rec=odds_day.get(rid)
            if pack is None or odds_rec is None:
                raise SystemExit(f"missing race/odds row race={rid}")

            novel=[str(x) for x in fr.get("novel_horse_ids") or []]
            if not novel:
                counters["no_k2_races"]+=1
                continue
            kings=[str(x) for x in fr.get("seven_consensus_order") or []]
            candidates=[str(x) for x in fr.get("candidate_horse_ids") or []]
            if len(kings)<3: raise SystemExit(f"king order short race={rid}")
            a1,a2,a3=kings[:3]

            horse_no=horse_number_map(pack)
            missing=set(candidates+kings[:3]+novel)-set(horse_no)
            if missing: raise SystemExit(f"horse number missing race={rid} sample={sorted(missing)[:5]}")

            odds_map=decode_odds(odds_rec)
            payouts,_=payout_map(pack)
            candidate_odds={}
            for hid in candidates:
                odd=odds_map.get(("WIN",(horse_no[hid],)))
                if odd is None or odd<=0:
                    raise SystemExit(f"WIN odds missing race={rid} horse={hid}")
                candidate_odds[hid]=float(odd)

            ordered_market=sorted(candidates,key=lambda h:(candidate_odds[h],h))
            market_rank={hid:i+1 for i,hid in enumerate(ordered_market)}

            a1o,a2o,a3o=(candidate_odds[a1],candidate_odds[a2],candidate_odds[a3])
            k2_odds=np.asarray([candidate_odds[h] for h in novel],dtype=float)
            k2_ranks=np.asarray([market_rank[h] for h in novel],dtype=float)
            top3_odds=np.asarray([a1o,a2o,a3o],dtype=float)

            base_return=0.0
            second_return=0.0
            for k in novel:
                kn=horse_no[k]
                base_return+=float(payouts.get(("TRIFECTA",(horse_no[a1],horse_no[a2],kn)),0.0))
                base_return+=float(payouts.get(("TRIFECTA",(horse_no[a2],horse_no[a1],kn)),0.0))
                for first,third in itertools.permutations((a1,a2,a3),2):
                    second_return+=float(payouts.get(("TRIFECTA",(horse_no[first],kn,horse_no[third])),0.0))

            base_stake=2*TICKET_PRICE*len(novel)
            second_stake=6*TICKET_PRICE*len(novel)

            meta=rr.get("race") or {}; cons=rr.get("consensus") or {}
            outs=list(fr.get("selected_outsiders") or [])
            anchor_mean=(a1o+a2o)/2.0
            anchor_worst=max(a1o,a2o)
            k2_min=float(np.min(k2_odds)); k2_mean=float(np.mean(k2_odds)); n=len(k2_odds)
            top3_mean=float(np.mean(top3_odds)); top3_worst=float(np.max(top3_odds))

            rows.append({
                "year":year,"race_id":rid,"race_date":date,
                "base_stake_yen":base_stake,"base_return_yen":base_return,
                "base_profit_yen":base_return-base_stake,
                "base_return_ratio":base_return/base_stake if base_stake else 0.0,
                "base_profit_label":int(base_return>base_stake),
                "base_hit_label":int(base_return>0),
                "second_stake_yen":second_stake,"second_return_yen":second_return,
                "second_profit_yen":second_return-second_stake,
                "second_return_ratio":second_return/second_stake if second_stake else 0.0,
                "second_profit_label":int(second_return>second_stake),
                "second_hit_label":int(second_return>0),

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

                "a1_win_odds":a1o,"a2_win_odds":a2o,
                "anchor_win_odds_mean":anchor_mean,
                "anchor_win_odds_max":anchor_worst,
                "anchor_implied_sum":1.0/a1o+1.0/a2o,
                "a1_market_rank":float(market_rank[a1]),
                "a2_market_rank":float(market_rank[a2]),
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
                "k2_best_vs_a1_ratio":k2_min/a1o,
                "k2_best_vs_anchor_worst_ratio":k2_min/anchor_worst,
                "k2_mean_vs_anchor_mean_ratio":k2_mean/anchor_mean,

                "a3_win_odds":a3o,
                "a3_market_rank":float(market_rank[a3]),
                "top3_king_win_odds_mean":top3_mean,
                "top3_king_win_odds_max":top3_worst,
                "top3_king_implied_sum":float(np.sum(1.0/top3_odds)),
                "k2_best_vs_top3_worst_ratio":k2_min/top3_worst,
                "k2_mean_vs_top3_mean_ratio":k2_mean/top3_mean,
            })
        if di%25==0:
            print(f"SEAT_ROUTER_BUILD_PROGRESS dates={di}/{len(date_pairs)} races={len(rows)}",flush=True)
    return pd.DataFrame(rows),dict(counters)

def action_view(frame,action):
    p=action_prefix(action)
    out=frame.copy()
    out["stake_yen"]=out[f"{p}_stake_yen"]
    out["return_yen"]=out[f"{p}_return_yen"]
    out["profit_yen"]=out[f"{p}_profit_yen"]
    out["profit_label"]=out[f"{p}_profit_label"]
    out["hit_label"]=out[f"{p}_hit_label"]
    return out

def max_drawdown(rows):
    cum=peak=maxdd=0.0
    for r in rows.sort_values(["race_date","race_id"]).itertuples(index=False):
        cum+=float(r.profit_yen); peak=max(peak,cum); maxdd=max(maxdd,peak-cum)
    return maxdd

def concentration(rows):
    vals=sorted([float(x) for x in rows["return_yen"]],reverse=True) if len(rows) else []
    ret=sum(vals); stake=float(rows["stake_yen"].sum()) if len(rows) else 0.0
    top1=vals[0] if vals else 0.0
    return {
        "largest_race_return_yen":top1,
        "top1_return_share_pct":100*top1/ret if ret else None,
        "top5_return_share_pct":100*sum(vals[:5])/ret if ret else None,
        "roi_without_top1_pct":100*(ret-top1)/stake if stake else None,
    }

def metric(policy,objective,test_year,frac,source,chosen):
    stake=float(chosen["stake_yen"].sum()) if len(chosen) else 0.0
    ret=float(chosen["return_yen"].sum()) if len(chosen) else 0.0
    return {
        "policy":policy,"objective":objective,"test_year":test_year,
        "target_train_top_fraction":frac,
        "source_races":len(source),"selected_races":len(chosen),
        "selected_race_pct":100*len(chosen)/len(source) if len(source) else 0.0,
        "base_selected":int((chosen["chosen_action"]=="BASE").sum()) if len(chosen) else 0,
        "second_selected":int((chosen["chosen_action"]=="K2_SECOND").sum()) if len(chosen) else 0,
        "profit_races":int(chosen["profit_label"].sum()) if len(chosen) else 0,
        "hit_races":int(chosen["hit_label"].sum()) if len(chosen) else 0,
        "tickets":int(round(stake/TICKET_PRICE)),
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(chosen) if len(chosen) else 0.0,
        **concentration(chosen),
    }

def choose_policy(test,percentiles,objective,frac,mode):
    threshold=1.0-frac
    rows=[]
    for i,r in test.iterrows():
        bp=float(percentiles[(objective,"BASE")][i])
        sp=float(percentiles[(objective,"K2_SECOND")][i])
        action=None; score=None
        if mode=="ROUTER":
            bq=bp>=threshold; sq=sp>=threshold
            if bq and sq:
                action="BASE" if bp>=sp else "K2_SECOND"
                score=max(bp,sp)
            elif bq:
                action="BASE"; score=bp
            elif sq:
                action="K2_SECOND"; score=sp
        elif mode=="BASE_ONLY":
            if bp>=threshold: action="BASE"; score=bp
        elif mode=="SECOND_ONLY":
            if sp>=threshold: action="K2_SECOND"; score=sp
        else:
            raise ValueError(mode)

        if action is None:
            continue
        p=action_prefix(action)
        rows.append({
            **r.to_dict(),
            "chosen_action":action,
            "action_percentile":score,
            "stake_yen":float(r[f"{p}_stake_yen"]),
            "return_yen":float(r[f"{p}_return_yen"]),
            "profit_yen":float(r[f"{p}_profit_yen"]),
            "profit_label":int(r[f"{p}_profit_label"]),
            "hit_label":int(r[f"{p}_hit_label"]),
        })
    if rows:
        return pd.DataFrame(rows)
    out=test.iloc[0:0].copy()
    out["chosen_action"]=pd.Series(dtype="object")
    out["action_percentile"]=pd.Series(dtype="float64")
    out["stake_yen"]=pd.Series(dtype="float64")
    out["return_yen"]=pd.Series(dtype="float64")
    out["profit_yen"]=pd.Series(dtype="float64")
    out["profit_label"]=pd.Series(dtype="int64")
    out["hit_label"]=pd.Series(dtype="int64")
    return out

def baseline_rows(test,action):
    v=action_view(test,action)
    v["chosen_action"]=action
    return v

def main():
    a=parse_args(); paths=parse_paths(a.router_year)
    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={y:load_router(paths[y],y) for y in YEARS}
    for y in YEARS:
        missing=set(fixed[y])-set(routers[y])
        if missing: raise SystemExit(f"fixed alerts missing router y={y} sample={sorted(missing)[:5]}")

    df,counters=build_rows(fixed,routers,Path(a.backfill_root))
    if any(df["year"]>=2026): raise SystemExit("2026 leakage")
    expected={2022:344,2023:346,2024:342,2025:346}
    for y,n in expected.items():
        got=int((df["year"]==y).sum())
        if got!=n: raise SystemExit(f"K2 universe drift y={y} got={got} expected={n}")

    folds=[]; quality=[]; metrics=[]; selected_rows=[]
    for test_year in TEST_YEARS:
        train=df[df["year"]<test_year].copy().reset_index(drop=True)
        test=df[df["year"]==test_year].copy().reset_index(drop=True)
        folds.append({
            "test_year":test_year,
            "train_years":"|".join(map(str,sorted(train["year"].unique()))),
            "train_races":len(train),"test_races":len(test),
            "train_base_profit_races":int(train["base_profit_label"].sum()),
            "train_second_profit_races":int(train["second_profit_label"].sum()),
            "test_base_profit_races":int(test["base_profit_label"].sum()),
            "test_second_profit_races":int(test["second_profit_label"].sum()),
        })

        percentiles={}
        for objective in OBJECTIVES:
            for action in ACTIONS:
                p=action_prefix(action)
                ytr=train[f"{p}_profit_label"].astype(int)
                yte=test[f"{p}_profit_label"].astype(int)
                if ytr.nunique()<2:
                    raise SystemExit(f"single class action={action} objective={objective} test={test_year}")
                model=make_model()
                model.fit(
                    train[FEATURES],ytr,
                    clf__sample_weight=training_weights(train,objective,action),
                )
                tr_score=model.predict_proba(train[FEATURES])[:,1]
                te_score=model.predict_proba(test[FEATURES])[:,1]
                percentiles[(objective,action)]=percentile_against_train(tr_score,te_score)
                quality.append({
                    "objective":objective,"action":action,"test_year":test_year,
                    "train_races":len(train),"train_profit_races":int(ytr.sum()),
                    "test_races":len(test),"test_profit_races":int(yte.sum()),
                    "roc_auc":safe_auc(yte,te_score),"pr_auc":safe_ap(yte,te_score),
                    "feature_count_input":len(FEATURES),
                })

        for objective in OBJECTIVES:
            for frac in FRACTIONS:
                for mode in ("ROUTER","BASE_ONLY","SECOND_ONLY"):
                    chosen=choose_policy(test,percentiles,objective,frac,mode)
                    policy=f"{mode}_{objective}"
                    metrics.append(metric(policy,objective,test_year,frac,test,chosen))
                    for row in chosen.itertuples(index=False):
                        selected_rows.append({
                            "policy":policy,"objective":objective,"test_year":test_year,
                            "target_train_top_fraction":frac,
                            "race_id":row.race_id,"race_date":row.race_date,
                            "chosen_action":row.chosen_action,
                            "action_percentile":float(row.action_percentile),
                            "stake_yen":float(row.stake_yen),
                            "return_yen":float(row.return_yen),
                            "profit_yen":float(row.profit_yen),
                            "profit_label":int(row.profit_label),
                        })

        for action in ACTIONS:
            base=baseline_rows(test,action)
            metrics.append(metric(f"ALL_{action}","BASELINE",test_year,1.0,test,base))

    stability=[]
    policies=sorted({r["policy"] for r in metrics if not r["policy"].startswith("ALL_")})
    for policy in policies:
        objective=next(r["objective"] for r in metrics if r["policy"]==policy)
        for frac in FRACTIONS:
            rs=[r for r in metrics if r["policy"]==policy and abs(float(r["target_train_top_fraction"])-frac)<1e-12]
            if len(rs)!=3: raise SystemExit(f"missing stability policy={policy} frac={frac}")
            stake=sum(r["stake_yen"] for r in rs); ret=sum(r["return_yen"] for r in rs)
            wo=[r["roi_without_top1_pct"] if r["roi_without_top1_pct"] is not None else 0.0 for r in rs]
            sh=[r["top1_return_share_pct"] if r["top1_return_share_pct"] is not None else 100.0 for r in rs]
            stability.append({
                "policy":policy,"objective":objective,"target_train_top_fraction":frac,
                "roi_2023":next(r["roi_pct"] for r in rs if r["test_year"]==2023),
                "roi_2024":next(r["roi_pct"] for r in rs if r["test_year"]==2024),
                "roi_2025":next(r["roi_pct"] for r in rs if r["test_year"]==2025),
                "min_year_roi_pct":min((r["roi_pct"] or 0.0) for r in rs),
                "combined_roi_pct":100*ret/stake if stake else None,
                "combined_profit_yen":ret-stake,
                "selected_races":sum(r["selected_races"] for r in rs),
                "base_selected":sum(r["base_selected"] for r in rs),
                "second_selected":sum(r["second_selected"] for r in rs),
                "min_selected_races":min(r["selected_races"] for r in rs),
                "min_roi_without_top1_pct":min(wo),
                "max_top1_return_share_pct":max(sh),
                "all_years_roi_100plus":int(all((r["roi_pct"] or 0.0)>=100 for r in rs)),
                "all_years_roi_wo_top1_100plus":int(all(x>=100 for x in wo)),
            })
    stability.sort(key=lambda r:(
        -r["all_years_roi_100plus"],
        -r["all_years_roi_wo_top1_100plus"],
        -(r["min_year_roi_pct"] or -1e9),
        -(r["combined_profit_yen"] or -1e9),
    ))

    summary={
        "contract":"L2_DANGER_SEAT_ROUTER_V1",
        "analysis_years":list(YEARS),"test_years":list(TEST_YEARS),
        "locked_years":list(LOCKED_YEARS),
        "actions":{
            "BASE":"A1/A2 in first-second both orders; every K2 novel third.",
            "K2_SECOND":"Every K2 novel second; ordered pair from Seven consensus Top3 in first/third.",
            "SKIP":"No ticket.",
        },
        "router":"Train separate action-specific profit classifiers. Convert each action score to its own TRAIN-distribution percentile, then choose the higher qualifying action; otherwise SKIP.",
        "objectives":{
            "EVENT":"Balanced logistic profit-event model.",
            "VALUE":"Same event target with capped positive sample weighting by realized return ratio on TRAIN only.",
        },
        "market_input":"Structural features + final WIN odds proxy only. No trifecta odds, payout, result, popularity, or odds missingness is an input feature.",
        "market_caveat":"FINAL_WIN_ODDS_PROXY is historical research only; deployment needs a pre-bet odds timestamp contract.",
        "threshold_rule":"Qualification is based on prior-year TRAIN percentile only. No test-year tuning.",
        "fractions":list(FRACTIONS),
        "folds":folds,"quality":quality,"stability":stability,
        "counters":counters,"production_promotion":False,
    }
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"folds.csv",folds)
    write_csv(out/"model-quality.csv",quality)
    write_csv(out/"selection-metrics.csv",metrics)
    write_csv(out/"stability.csv",stability)
    write_csv(out/"selected-races.csv",selected_rows)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Danger Seat Router V1\n\n"
        "Walk-forward 3-way L2 router over BASE / K2_SECOND / SKIP. "
        "Action models use structural features plus final WIN-market proxy; score scales are aligned by prior-TRAIN percentiles before routing. "
        "No trifecta odds, result fields, payout values, popularity, or missingness are input features. "
        "2026 remains sealed and this run does not promote a production rule.\n",
        encoding="utf-8",
    )
    print("L2_DANGER_SEAT_ROUTER_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
