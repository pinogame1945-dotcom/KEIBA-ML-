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
    decode_odds,finite,horse_number_map,load_day,load_fixed_ledgers,
    load_odds_day,load_router,payout_map,seven_stats,
)

YEARS=(2022,2023,2024,2025)
TEST_YEARS=(2023,2024,2025)
LOCKED_YEARS=(2026,)
TICKET_PRICE=100.0

STRUCT_NUMERIC=[
    "gate_score","candidate_pool_size","seven_union_count","novel_pool_count",
    "field_size","distance_m","race_month",
    "cw_top1_max_vote_share","cw_top3_jaccard","cw_top6_jaccard",
    "cw_rank_diff_mean","cw_rank_std_mean","cw_prob_std_mean","cw_prob_std_max",
    "novel_position","candidate_position","horse_number","horse_number_ratio",
]
STRUCT_CATEGORICAL=[
    "venue_code","surface","race_class","discipline","direction",
    "selected_outsider_1","selected_outsider_2","selected_outsider_pair",
]
MARKET_NUMERIC=[
    "k2_win_odds","k2_market_rank","k2_market_rank_ratio",
    "a1_win_odds","a2_win_odds","anchor_win_odds_mean","anchor_win_odds_max",
    "k2_vs_a1_ratio","k2_vs_anchor_mean_ratio","k2_vs_anchor_worst_ratio",
]
SPECS={
    "STRUCT_THIRD":{"market":False},
    "MARKET_THIRD":{"market":True},
}

def parse_args():
    p=argparse.ArgumentParser(description="L2 K2 exact-third selector for fixed BASE trifecta.")
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
    y=np.asarray(y,dtype=int)
    return float(roc_auc_score(y,p)) if len(set(y.tolist()))>1 else None

def safe_ap(y,p):
    y=np.asarray(y,dtype=int)
    return float(average_precision_score(y,p)) if len(set(y.tolist()))>1 else None

def result_targets(pack):
    out={}
    counters=defaultdict(int)
    for row in pack.get("results") or []:
        hid=str(row.get("horse_id") or "")
        status=str(row.get("result_status") or "").upper()
        if not hid:
            continue
        if status in {"SCRATCHED","EXCLUDED"}:
            counters[f"result_status_{status.lower()}"]+=1
            continue
        if status=="FINISHED":
            try:
                pos=int(row.get("official_finish_position"))
            except (TypeError,ValueError):
                raise SystemExit(f"FINISHED result missing official position horse={hid}")
            if pos<=0:
                raise SystemExit(f"FINISHED result invalid official position horse={hid} pos={pos}")
            out[hid]={"third":int(pos==3),"top3":int(pos<=3),"status":status,"position":pos}
        else:
            # A legitimate starter that did not finish cannot occupy the third-place seat.
            counters[f"result_status_{status.lower() or 'unknown'}"]+=1
            out[hid]={"third":0,"top3":0,"status":status or "UNKNOWN","position":None}
    return out,dict(counters)

def make_model(market):
    nums=STRUCT_NUMERIC+(MARKET_NUMERIC if market else [])
    prep=ColumnTransformer([
        ("num",Pipeline([("scale",StandardScaler())]),nums),
        ("cat",OneHotEncoder(handle_unknown="ignore"),STRUCT_CATEGORICAL),
    ])
    clf=LogisticRegression(
        max_iter=3000,C=0.35,class_weight="balanced",solver="lbfgs"
    )
    return Pipeline([("prep",prep),("clf",clf)]),nums+STRUCT_CATEGORICAL

def build_rows(fixed,routers,root):
    date_pairs=defaultdict(list)
    for y in YEARS:
        for rid in fixed[y]:
            date=str(routers[y][rid].get("race_date") or "")[:10]
            if len(date)!=10: raise SystemExit(f"bad race date {rid}")
            date_pairs[date].append((y,rid))

    rows=[]; race_eval={}; counters=defaultdict(int)
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
                counters["no_k2_races"]+=1; continue
            kings=[str(x) for x in fr.get("seven_consensus_order") or []]
            if len(kings)<2: raise SystemExit(f"king order short {rid}")
            a1,a2=kings[:2]
            candidates=[str(x) for x in fr.get("candidate_horse_ids") or []]
            horse_no=horse_number_map(pack)
            missing=set(candidates)-set(horse_no)
            if missing: raise SystemExit(f"horse number missing race={rid} sample={sorted(missing)[:5]}")

            targets,target_counts=result_targets(pack)
            for key,val in target_counts.items():
                counters[key]+=val
            missing_targets=[k for k in novel if k not in targets]
            if missing_targets:
                raise SystemExit(f"eligible result target missing K2 race={rid} sample={missing_targets[:5]}")

            odds=decode_odds(odds_rec)
            candidate_odds={}
            for hid in candidates:
                odd=odds.get(("WIN",(horse_no[hid],)))
                if odd is None or odd<=0:
                    raise SystemExit(f"WIN odds missing race={rid} horse={hid}")
                candidate_odds[hid]=float(odd)

            market_order=sorted(candidates,key=lambda h:(candidate_odds[h],h))
            market_rank={h:i+1 for i,h in enumerate(market_order)}
            candidate_pos={h:i+1 for i,h in enumerate(candidates)}
            payouts,_=payout_map(pack)

            a1o,a2o=candidate_odds[a1],candidate_odds[a2]
            amean=(a1o+a2o)/2.0; aworst=max(a1o,a2o)
            meta=rr.get("race") or {}; cons=rr.get("consensus") or {}
            outs=list(fr.get("selected_outsiders") or [])
            field_size=float(finite(meta.get("field_size")) or len(horse_no) or 1.0)

            race_eval[rid]={
                "year":year,"race_id":rid,"race_date":date,
                "a1":a1,"a2":a2,"novel":novel,"horse_no":horse_no,
                "payouts":payouts,
            }

            for i,hid in enumerate(novel,1):
                odd=candidate_odds[hid]
                mr=float(market_rank[hid])
                rows.append({
                    "year":year,"race_id":rid,"race_date":date,"horse_id":hid,
                    "third_label":int(targets[hid]["third"]),
                    "top3_label":int(targets[hid]["top3"]),
                    "result_status_target":targets[hid]["status"],

                    "gate_score":finite(fr.get("gate_score")) or 0.0,
                    "candidate_pool_size":len(candidates),
                    "seven_union_count":len(fr.get("seven_union_horse_ids") or []),
                    "novel_pool_count":len(novel),
                    "field_size":field_size,
                    "distance_m":finite(meta.get("distance_m")) or 0.0,
                    "race_month":int(date[5:7]),
                    "cw_top1_max_vote_share":finite(cons.get("top1_max_vote_share")) or 0.0,
                    "cw_top3_jaccard":finite(cons.get("top3_pairwise_jaccard_mean")) or 0.0,
                    "cw_top6_jaccard":finite(cons.get("top6_pairwise_jaccard_mean")) or 0.0,
                    "cw_rank_diff_mean":finite(cons.get("pairwise_rank_abs_diff_mean")) or 0.0,
                    "cw_rank_std_mean":finite(cons.get("horse_rank_std_mean")) or 0.0,
                    "cw_prob_std_mean":finite(cons.get("horse_probability_std_mean")) or 0.0,
                    "cw_prob_std_max":finite(cons.get("horse_probability_std_max")) or 0.0,
                    "novel_position":float(i),
                    "candidate_position":float(candidate_pos[hid]),
                    "horse_number":float(horse_no[hid]),
                    "horse_number_ratio":float(horse_no[hid])/max(1.0,field_size),

                    "venue_code":meta.get("venue_code"),
                    "surface":meta.get("surface"),
                    "race_class":meta.get("race_class"),
                    "discipline":meta.get("discipline"),
                    "direction":meta.get("direction"),
                    "selected_outsider_1":outs[0] if len(outs)>0 else "__NONE__",
                    "selected_outsider_2":outs[1] if len(outs)>1 else "__NONE__",
                    "selected_outsider_pair":"|".join(outs) if outs else "__NONE__",

                    "k2_win_odds":odd,
                    "k2_market_rank":mr,
                    "k2_market_rank_ratio":mr/max(1.0,float(len(candidates))),
                    "a1_win_odds":a1o,
                    "a2_win_odds":a2o,
                    "anchor_win_odds_mean":amean,
                    "anchor_win_odds_max":aworst,
                    "k2_vs_a1_ratio":odd/a1o,
                    "k2_vs_anchor_mean_ratio":odd/amean,
                    "k2_vs_anchor_worst_ratio":odd/aworst,
                })
        if di%25==0:
            print(f"K2_SELECTOR_BUILD_PROGRESS dates={di}/{len(date_pairs)} k2_rows={len(rows)}",flush=True)

    return pd.DataFrame(rows),race_eval,dict(counters)

def evaluate_policy(test_scores,race_eval,model_name,policy,test_year):
    selected=[]
    for rid,g in test_scores.groupby("race_id",sort=False):
        ordered=g.sort_values(["score","horse_id"],ascending=[False,True])
        if policy=="TOP1":
            chosen=ordered.head(1)
        elif policy=="TOP2":
            chosen=ordered.head(min(2,len(ordered)))
        elif policy=="ALL":
            chosen=ordered
        elif policy=="NOVEL_ORDER_TOP1":
            chosen=g.sort_values(["novel_position","horse_id"]).head(1)
        elif policy=="NOVEL_ORDER_TOP2":
            chosen=g.sort_values(["novel_position","horse_id"]).head(min(2,len(g)))
        else:
            raise ValueError(policy)

        info=race_eval[rid]; a1=info["a1"]; a2=info["a2"]; nums=info["horse_no"]
        ret=0.0
        ids=[str(x) for x in chosen["horse_id"].tolist()]
        for h in ids:
            hn=nums[h]
            ret+=float(info["payouts"].get(("TRIFECTA",(nums[a1],nums[a2],hn)),0.0))
            ret+=float(info["payouts"].get(("TRIFECTA",(nums[a2],nums[a1],hn)),0.0))
        tickets=2*len(ids); stake=tickets*TICKET_PRICE
        selected.append({
            "model":model_name,"policy":policy,"test_year":test_year,
            "race_id":rid,"race_date":info["race_date"],
            "selected_k2_count":len(ids),"selected_k2_ids":"|".join(ids),
            "tickets":tickets,"stake_yen":stake,"return_yen":ret,
            "profit_yen":ret-stake,"hit":int(ret>0),
        })
    return pd.DataFrame(selected)

def max_drawdown(df):
    cum=peak=maxdd=0.0
    for r in df.sort_values(["race_date","race_id"]).itertuples(index=False):
        cum+=float(r.profit_yen); peak=max(peak,cum); maxdd=max(maxdd,peak-cum)
    return maxdd

def metrics(df,model,policy,year,baseline_hits=None):
    stake=float(df["stake_yen"].sum()); ret=float(df["return_yen"].sum())
    rr=sorted([float(x) for x in df["return_yen"]],reverse=True)
    top1=rr[0] if rr else 0.0
    hits=int(df["hit"].sum())
    return {
        "model":model,"policy":policy,"test_year":year,
        "races":len(df),"hit_races":hits,
        "hit_rate_pct":100*hits/len(df) if len(df) else 0.0,
        "baseline_hit_retention_pct":100*hits/baseline_hits if baseline_hits else None,
        "tickets":int(df["tickets"].sum()),
        "avg_tickets_per_race":float(df["tickets"].mean()) if len(df) else 0.0,
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
        folds.append({
            "test_year":test_year,
            "train_years":"|".join(map(str,sorted(train["year"].unique()))),
            "train_k2_rows":len(train),"train_third_rows":int(ytr.sum()),
            "test_k2_rows":len(test),"test_third_rows":int(yte.sum()),
            "test_races":int(test["race_id"].nunique()),
        })

        # Baseline is identical regardless model.
        all_scores=test[["year","race_id","race_date","horse_id","novel_position"]].copy()
        all_scores["score"]=0.0
        base_eval=evaluate_policy(all_scores,race_eval,"BASELINE","ALL",test_year)
        base_hits=int(base_eval["hit"].sum())
        year_metrics.append(metrics(base_eval,"BASELINE","ALL",test_year,base_hits))
        for r in base_eval.to_dict("records"): selected_rows.append(r)

        for model_name,spec in SPECS.items():
            model,features=make_model(spec["market"])
            model.fit(train[features],ytr)
            tr_score=model.predict_proba(train[features])[:,1]
            te_score=model.predict_proba(test[features])[:,1]
            quality.append({
                "model":model_name,"test_year":test_year,
                "train_k2_rows":len(train),"train_third_rows":int(ytr.sum()),
                "test_k2_rows":len(test),"test_third_rows":int(yte.sum()),
                "roc_auc":safe_auc(yte,te_score),"pr_auc":safe_ap(yte,te_score),
                "feature_count_input":len(features),
            })
            scored=test[["year","race_id","race_date","horse_id","novel_position"]].copy()
            scored["score"]=te_score
            for policy in ("TOP1","TOP2"):
                ev=evaluate_policy(scored,race_eval,model_name,policy,test_year)
                year_metrics.append(metrics(ev,model_name,policy,test_year,base_hits))
                for r in ev.to_dict("records"): selected_rows.append(r)

        for policy in ("NOVEL_ORDER_TOP1","NOVEL_ORDER_TOP2"):
            ev=evaluate_policy(all_scores,race_eval,"ORDER_CONTROL",policy,test_year)
            year_metrics.append(metrics(ev,"ORDER_CONTROL",policy,test_year,base_hits))
            for r in ev.to_dict("records"): selected_rows.append(r)

    stability=[]
    pairs=sorted({(r["model"],r["policy"]) for r in year_metrics})
    for model,policy in pairs:
        rs=[r for r in year_metrics if r["model"]==model and r["policy"]==policy]
        if len(rs)!=3: raise SystemExit(f"metric coverage missing {model} {policy}")
        stake=sum(r["stake_yen"] for r in rs); ret=sum(r["return_yen"] for r in rs)
        wo=[r["roi_without_top1_pct"] if r["roi_without_top1_pct"] is not None else 0.0 for r in rs]
        shares=[r["top1_return_share_pct"] if r["top1_return_share_pct"] is not None else 100.0 for r in rs]
        stability.append({
            "model":model,"policy":policy,
            "roi_2023":next(r["roi_pct"] for r in rs if r["test_year"]==2023),
            "roi_2024":next(r["roi_pct"] for r in rs if r["test_year"]==2024),
            "roi_2025":next(r["roi_pct"] for r in rs if r["test_year"]==2025),
            "min_year_roi_pct":min(r["roi_pct"] or 0.0 for r in rs),
            "combined_roi_pct":100*ret/stake if stake else None,
            "combined_profit_yen":ret-stake,
            "combined_tickets":sum(r["tickets"] for r in rs),
            "avg_tickets_per_race":sum(r["tickets"] for r in rs)/sum(r["races"] for r in rs),
            "min_hit_retention_pct":min(r["baseline_hit_retention_pct"] or 0.0 for r in rs),
            "min_roi_without_top1_pct":min(wo),
            "max_top1_return_share_pct":max(shares),
            "all_years_roi_100plus":int(all((r["roi_pct"] or 0.0)>=100 for r in rs)),
        })
    stability.sort(key=lambda r:(
        -r["all_years_roi_100plus"],
        -(r["min_year_roi_pct"] or -1e9),
        -(r["combined_roi_pct"] or -1e9),
    ))

    summary={
        "contract":"L2_DANGER_K2_THIRD_SELECTOR_V1",
        "analysis_years":list(YEARS),"test_years":list(TEST_YEARS),
        "locked_years":list(LOCKED_YEARS),
        "base_ticket":"A1->A2->selected K2 and A2->A1->selected K2, 100 yen each.",
        "target":"Per-K2 exact official third-place label.",
        "policies":["ALL","TOP1","TOP2","NOVEL_ORDER_TOP1","NOVEL_ORDER_TOP2"],
        "models":SPECS,
        "market_rule":"MARKET_THIRD uses final WIN odds proxy only. No trifecta odds, payouts, result fields, popularity, or odds missingness is an input.",
        "market_caveat":"Historical final WIN odds proxy; live use requires pre-bet timestamp policy.",
        "selection_rule":"Always buy each danger race; only K2 count changes. No buy/skip is mixed into this experiment.",
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
        "# L2 Danger K2 Third Selector V1\n\n"
        "Keeps the BASE trifecta seat structure fixed and tests whether K2 novel horses can be pruned to Top1/Top2 by an exact-third classifier. "
        "The experiment never changes A1/A2 seats and never mixes in buy/skip, so any effect is attributable to K2 point reduction. "
        "Structural-only and structural+WIN-market models are compared against ALL K2 and frozen novel-order controls. 2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_DANGER_K2_THIRD_SELECTOR_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
