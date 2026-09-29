#!/usr/bin/env python3
import argparse,csv,json,math
from pathlib import Path

import numpy as np
import pandas as pd

from run_l2_danger_k2_third_selector_v1 import (
    YEARS,TEST_YEARS,LOCKED_YEARS,TICKET_PRICE,
    STRUCT_NUMERIC,STRUCT_CATEGORICAL,MARKET_NUMERIC,
    load_fixed_ledgers,load_router,parse_paths,build_rows,make_model,
    safe_auc,safe_ap,write_csv,max_drawdown,
)

SCORE_RULES=("P_ONLY","P_X_SQRT_ODDS","P_X_ODDS_CAP30","P_X_ODDS")
TOPKS=(1,2)

def parse_args():
    p=argparse.ArgumentParser(description="Rank K2 by exact-third probability x market value proxy.")
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def value_score(prob,odd,rule):
    p=float(prob); o=max(1e-9,float(odd))
    if rule=="P_ONLY": return p
    if rule=="P_X_SQRT_ODDS": return p*math.sqrt(o)
    if rule=="P_X_ODDS_CAP30": return p*min(o,30.0)
    if rule=="P_X_ODDS": return p*o
    raise ValueError(rule)

def eval_selected(scored,race_eval,model_name,score_rule,topk,test_year):
    rows=[]
    for rid,g in scored.groupby("race_id",sort=False):
        ranked=g.sort_values(["value_score","third_prob","horse_id"],ascending=[False,False,True])
        chosen=ranked.head(min(topk,len(ranked)))
        info=race_eval[rid]; a1=info["a1"]; a2=info["a2"]; nums=info["horse_no"]
        ret=0.0; ids=[]; probs=[]; odds=[]; scores=[]
        for x in chosen.itertuples(index=False):
            hid=str(x.horse_id); hn=nums[hid]
            ids.append(hid); probs.append(float(x.third_prob)); odds.append(float(x.k2_win_odds)); scores.append(float(x.value_score))
            ret+=float(info["payouts"].get(("TRIFECTA",(nums[a1],nums[a2],hn)),0.0))
            ret+=float(info["payouts"].get(("TRIFECTA",(nums[a2],nums[a1],hn)),0.0))
        tickets=2*len(ids); stake=tickets*TICKET_PRICE
        rows.append({
            "model":model_name,"score_rule":score_rule,"topk":topk,"test_year":test_year,
            "race_id":rid,"race_date":info["race_date"],
            "selected_k2_count":len(ids),"selected_k2_ids":"|".join(ids),
            "selected_third_probs":"|".join(f"{x:.8f}" for x in probs),
            "selected_win_odds":"|".join(f"{x:.4f}" for x in odds),
            "selected_value_scores":"|".join(f"{x:.8f}" for x in scores),
            "tickets":tickets,"stake_yen":stake,"return_yen":ret,
            "profit_yen":ret-stake,"hit":int(ret>0),
        })
    return pd.DataFrame(rows)

def eval_all(test,race_eval,test_year):
    rows=[]
    for rid,g in test.groupby("race_id",sort=False):
        info=race_eval[rid]; a1=info["a1"]; a2=info["a2"]; nums=info["horse_no"]
        ret=0.0; ids=[str(x) for x in g["horse_id"].tolist()]
        for hid in ids:
            hn=nums[hid]
            ret+=float(info["payouts"].get(("TRIFECTA",(nums[a1],nums[a2],hn)),0.0))
            ret+=float(info["payouts"].get(("TRIFECTA",(nums[a2],nums[a1],hn)),0.0))
        tickets=2*len(ids); stake=tickets*TICKET_PRICE
        rows.append({
            "model":"BASELINE","score_rule":"ALL","topk":0,"test_year":test_year,
            "race_id":rid,"race_date":info["race_date"],
            "selected_k2_count":len(ids),"selected_k2_ids":"|".join(ids),
            "tickets":tickets,"stake_yen":stake,"return_yen":ret,
            "profit_yen":ret-stake,"hit":int(ret>0),
        })
    return pd.DataFrame(rows)

def metrics(df,model,score_rule,topk,year,baseline_hits,baseline_tickets):
    stake=float(df["stake_yen"].sum()); ret=float(df["return_yen"].sum())
    returns=sorted([float(x) for x in df["return_yen"]],reverse=True)
    top1=returns[0] if returns else 0.0
    hits=int(df["hit"].sum()); tickets=int(df["tickets"].sum())
    return {
        "model":model,"score_rule":score_rule,"topk":topk,"test_year":year,
        "races":len(df),"hit_races":hits,
        "hit_retention_pct":100*hits/baseline_hits if baseline_hits else None,
        "tickets":tickets,
        "ticket_reduction_pct":100*(baseline_tickets-tickets)/baseline_tickets if baseline_tickets else None,
        "avg_tickets_per_race":tickets/len(df) if len(df) else 0.0,
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(df),
        "top1_return_share_pct":100*top1/ret if ret else None,
        "roi_without_top1_pct":100*(ret-top1)/stake if stake else None,
    }

def main():
    a=parse_args(); paths=parse_paths(a.router_year)
    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={y:load_router(paths[y],y) for y in YEARS}
    for y in YEARS:
        missing=set(fixed[y])-set(routers[y])
        if missing: raise SystemExit(f"fixed alerts missing router y={y} sample={sorted(missing)[:5]}")

    df,race_eval,counters=build_rows(fixed,routers,Path(a.backfill_root))
    if any(df["year"]>=2026): raise SystemExit("2026 leakage")
    expected_races={2022:344,2023:346,2024:342,2025:346}
    for y,n in expected_races.items():
        got=int(df[df["year"]==y]["race_id"].nunique())
        if got!=n: raise SystemExit(f"K2 race universe drift y={y} got={got} expected={n}")

    folds=[]; quality=[]; year_metrics=[]; selected_rows=[]
    for test_year in TEST_YEARS:
        train=df[df["year"]<test_year].copy().reset_index(drop=True)
        test=df[df["year"]==test_year].copy().reset_index(drop=True)
        ytr=train["third_label"].astype(int); yte=test["third_label"].astype(int)
        if ytr.nunique()<2: raise SystemExit(f"single class train {test_year}")

        base=eval_all(test,race_eval,test_year)
        base_hits=int(base["hit"].sum()); base_tickets=int(base["tickets"].sum())
        year_metrics.append(metrics(base,"BASELINE","ALL",0,test_year,base_hits,base_tickets))
        selected_rows.extend(base.to_dict("records"))

        folds.append({
            "test_year":test_year,
            "train_years":"|".join(map(str,sorted(train["year"].unique()))),
            "train_k2_rows":len(train),"train_third_rows":int(ytr.sum()),
            "test_k2_rows":len(test),"test_third_rows":int(yte.sum()),
            "test_races":int(test["race_id"].nunique()),
            "baseline_hits":base_hits,"baseline_tickets":base_tickets,
        })

        for model_name,market in (("STRUCT_PROB",False),("MARKET_PROB",True)):
            model,features=make_model(market)
            model.fit(train[features],ytr)
            te_prob=model.predict_proba(test[features])[:,1]
            quality.append({
                "model":model_name,"test_year":test_year,
                "train_k2_rows":len(train),"train_third_rows":int(ytr.sum()),
                "test_k2_rows":len(test),"test_third_rows":int(yte.sum()),
                "roc_auc":safe_auc(yte,te_prob),"pr_auc":safe_ap(yte,te_prob),
                "feature_count_input":len(features),
            })
            scored=test[["year","race_id","race_date","horse_id","k2_win_odds"]].copy()
            scored["third_prob"]=te_prob

            for rule in SCORE_RULES:
                scored["value_score"]=[
                    value_score(p,o,rule)
                    for p,o in zip(scored["third_prob"],scored["k2_win_odds"])
                ]
                for topk in TOPKS:
                    ev=eval_selected(scored,race_eval,model_name,rule,topk,test_year)
                    year_metrics.append(metrics(ev,model_name,rule,topk,test_year,base_hits,base_tickets))
                    selected_rows.extend(ev.to_dict("records"))

    stability=[]
    combos=sorted({(r["model"],r["score_rule"],r["topk"]) for r in year_metrics})
    for model,rule,topk in combos:
        rs=[r for r in year_metrics if r["model"]==model and r["score_rule"]==rule and r["topk"]==topk]
        if len(rs)!=3: raise SystemExit(f"metric coverage missing {model} {rule} top{topk}")
        stake=sum(r["stake_yen"] for r in rs); ret=sum(r["return_yen"] for r in rs)
        wo=[r["roi_without_top1_pct"] if r["roi_without_top1_pct"] is not None else 0.0 for r in rs]
        sh=[r["top1_return_share_pct"] if r["top1_return_share_pct"] is not None else 100.0 for r in rs]
        stability.append({
            "model":model,"score_rule":rule,"topk":topk,
            "roi_2023":next(r["roi_pct"] for r in rs if r["test_year"]==2023),
            "roi_2024":next(r["roi_pct"] for r in rs if r["test_year"]==2024),
            "roi_2025":next(r["roi_pct"] for r in rs if r["test_year"]==2025),
            "min_year_roi_pct":min(r["roi_pct"] or 0.0 for r in rs),
            "combined_roi_pct":100*ret/stake if stake else None,
            "combined_profit_yen":ret-stake,
            "combined_tickets":sum(r["tickets"] for r in rs),
            "avg_tickets_per_race":sum(r["tickets"] for r in rs)/sum(r["races"] for r in rs),
            "min_hit_retention_pct":min(r["hit_retention_pct"] or 0.0 for r in rs),
            "min_ticket_reduction_pct":min(r["ticket_reduction_pct"] or 0.0 for r in rs),
            "min_roi_without_top1_pct":min(wo),
            "max_top1_return_share_pct":max(sh),
            "all_years_roi_100plus":int(all((r["roi_pct"] or 0.0)>=100 for r in rs)),
        })
    stability.sort(key=lambda r:(
        -r["all_years_roi_100plus"],
        -(r["min_year_roi_pct"] or -1e9),
        -(r["combined_roi_pct"] or -1e9),
        -(r["min_hit_retention_pct"] or -1e9),
    ))

    summary={
        "contract":"L2_DANGER_K2_VALUE_SELECTOR_V1",
        "analysis_years":list(YEARS),"test_years":list(TEST_YEARS),
        "locked_years":list(LOCKED_YEARS),
        "base_ticket":"A1->A2->selected K2 and A2->A1->selected K2, 100 yen each.",
        "target":"Per-K2 exact third-place probability learned walk-forward.",
        "score_rules":{
            "P_ONLY":"predicted third probability",
            "P_X_SQRT_ODDS":"predicted third probability * sqrt(K2 WIN odds)",
            "P_X_ODDS_CAP30":"predicted third probability * min(K2 WIN odds,30)",
            "P_X_ODDS":"predicted third probability * K2 WIN odds",
        },
        "price_proxy":"K2 WIN odds only; within-race anchor prices are common to every K2 and do not affect K2 ranking.",
        "market_caveat":"Historical final WIN odds proxy. Production use requires a pre-bet timestamp contract.",
        "anti_leakage":"No trifecta price completeness filter. No payout/result fields in features. Trifecta payouts are evaluation only.",
        "selection_rule":"Every danger race remains bought; only K2 Top1/Top2 selection changes.",
        "folds":folds,"quality":quality,"stability":stability,
        "counters":counters,"production_promotion":False,
    }
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"folds.csv",folds)
    write_csv(out/"model-quality.csv",quality)
    write_csv(out/"year-metrics.csv",year_metrics)
    write_csv(out/"stability.csv",stability)
    write_csv(out/"selected-races.csv",selected_rows)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Danger K2 Value Selector V1\n\n"
        "Keeps BASE seats fixed and ranks K2 by walk-forward exact-third probability combined with a WIN-odds value proxy. "
        "Compares probability-only against progressively stronger price weighting. "
        "No trifecta-odds completeness filtering is allowed; payouts/results are evaluation-only. 2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_DANGER_K2_VALUE_SELECTOR_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
