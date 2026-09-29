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
FRACTIONS=(0.10,0.15,0.20,0.25,0.30,0.40,0.50)
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
TARGETS=("anchor_top2","anchor_top3","k2_third","k2_top3")

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

def result_positions(pack):
    out={}
    for row in pack.get("results") or []:
        hid=str(row.get("horse_id") or "")
        if not hid or row.get("result_status")!="FINISHED":
            continue
        try:
            pos=int(row.get("official_finish_position"))
        except (TypeError,ValueError):
            continue
        if pos>0:
            out[hid]=pos
    return out

def safe_auc(y,p):
    vals=np.asarray(y,dtype=int)
    return float(roc_auc_score(vals,p)) if len(set(vals.tolist()))>1 else None

def safe_ap(y,p):
    vals=np.asarray(y,dtype=int)
    return float(average_precision_score(vals,p)) if len(set(vals.tolist()))>1 else None

def build_rows(fixed,routers,root):
    date_pairs=defaultdict(list)
    for y in YEARS:
        for rid in fixed[y]:
            date=str(routers[y][rid].get("race_date") or "")[:10]
            if len(date)!=10: raise SystemExit(f"bad race date race={rid}")
            date_pairs[date].append((y,rid))

    rows=[]; counters=defaultdict(int)
    for di,date in enumerate(sorted(date_pairs),1):
        pairs=date_pairs[date]
        wanted={rid for _,rid in pairs}
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
            anchors=[str(x) for x in fr.get("seven_anchor_horse_ids") or []]
            candidates=[str(x) for x in fr.get("candidate_horse_ids") or []]
            if len(anchors)!=2: raise SystemExit(f"bad anchors race={rid}")

            horse_no=horse_number_map(pack)
            missing=set(anchors+novel+candidates)-set(horse_no)
            if missing:
                raise SystemExit(f"horse number missing race={rid} sample={sorted(missing)[:5]}")

            odds_map=decode_odds(odds_rec)
            payouts,_=payout_map(pack)
            candidate_odds={}
            for hid in candidates:
                odd=odds_map.get(("WIN",(horse_no[hid],)))
                if odd is None or odd<=0:
                    raise SystemExit(f"WIN odds missing race={rid} horse={hid}")
                candidate_odds[hid]=float(odd)

            positions=result_positions(pack)
            if not positions:
                raise SystemExit(f"result positions missing race={rid}")

            a1_hid,a2_hid=anchors
            a1_no,a2_no=horse_no[a1_hid],horse_no[a2_hid]
            a1_odd,a2_odd=candidate_odds[a1_hid],candidate_odds[a2_hid]
            k2_odds=np.asarray([candidate_odds[x] for x in novel],dtype=float)
            ordered_market=sorted(candidates,key=lambda h:(candidate_odds[h],h))
            market_rank={hid:i+1 for i,hid in enumerate(ordered_market)}
            k2_ranks=np.asarray([market_rank[x] for x in novel],dtype=float)

            race_return=0.0
            for hid in novel:
                h=horse_no[hid]
                race_return+=float(payouts.get(("TRIFECTA",(a1_no,a2_no,h)),0.0))
                race_return+=float(payouts.get(("TRIFECTA",(a2_no,a1_no,h)),0.0))
            stake=2*TICKET_PRICE*len(novel)
            profit=race_return-stake

            p1=positions.get(a1_hid,99); p2=positions.get(a2_hid,99)
            k2_pos=[positions.get(h,99) for h in novel]
            anchor_top2=int(p1<=2 and p2<=2)
            anchor_top3=int(p1<=3 and p2<=3)
            k2_third=int(any(p==3 for p in k2_pos))
            k2_top3=int(any(p<=3 for p in k2_pos))
            structural_exact=int(anchor_top2 and k2_third)
            direct_hit=int(race_return>0)
            if structural_exact!=direct_hit:
                counters["structural_exact_vs_payout_hit_mismatch"]+=1

            meta=rr.get("race") or {}; cons=rr.get("consensus") or {}
            outs=list(fr.get("selected_outsiders") or [])
            anchor_mean=(a1_odd+a2_odd)/2.0; anchor_worst=max(a1_odd,a2_odd)
            k2_min=float(np.min(k2_odds)); k2_mean=float(np.mean(k2_odds)); n=len(k2_odds)
            price_richness_raw=math.log1p(k2_mean)+0.5*math.log1p(anchor_mean)

            rows.append({
                "year":year,"race_id":rid,"race_date":date,
                "stake_yen":stake,"return_yen":race_return,"profit_yen":profit,
                "return_ratio":race_return/stake if stake else 0.0,
                "direct_hit":direct_hit,
                "profit_label":int(profit>0),
                "anchor_top2":anchor_top2,
                "anchor_top3":anchor_top3,
                "k2_third":k2_third,
                "k2_top3":k2_top3,
                "structural_exact":structural_exact,
                "price_richness_raw":price_richness_raw,

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

                "a1_win_odds":a1_odd,"a2_win_odds":a2_odd,
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
            print(f"DECOMP_BUILD_PROGRESS dates={di}/{len(date_pairs)} races={len(rows)}",flush=True)
    return pd.DataFrame(rows),dict(counters)

def make_model(with_market):
    nums=STRUCT_NUMERIC+(MARKET_NUMERIC if with_market else [])
    features=nums+STRUCT_CATEGORICAL
    prep=ColumnTransformer([
        ("num",Pipeline([("scale",StandardScaler())]),nums),
        ("cat",OneHotEncoder(handle_unknown="ignore"),STRUCT_CATEGORICAL),
    ])
    model=Pipeline([
        ("prep",prep),
        ("clf",LogisticRegression(max_iter=3000,C=0.35,class_weight="balanced")),
    ])
    return model,features

def percentile_against_train(train_vals,vals):
    ref=np.sort(np.asarray(train_vals,dtype=float))
    x=np.asarray(vals,dtype=float)
    return np.searchsorted(ref,x,side="right")/max(1,len(ref))

def max_drawdown(sel):
    cum=peak=maxdd=0.0
    for r in sel.sort_values(["race_date","race_id"]).itertuples(index=False):
        cum+=float(r.profit_yen); peak=max(peak,cum); maxdd=max(maxdd,peak-cum)
    return maxdd

def concentration(sel):
    if sel.empty:
        return {"top1_return_share_pct":None,"top5_return_share_pct":None,"roi_without_top1_pct":None}
    vals=sorted([float(x) for x in sel["return_yen"]],reverse=True)
    total=sum(vals); stake=float(sel["stake_yen"].sum()); top1=vals[0] if vals else 0.0
    return {
        "top1_return_share_pct":100*top1/total if total else None,
        "top5_return_share_pct":100*sum(vals[:5])/total if total else None,
        "roi_without_top1_pct":100*(total-top1)/stake if stake else None,
    }

def metric_row(sel,source,model,test_year,frac,cutoff):
    stake=float(sel["stake_yen"].sum()) if len(sel) else 0.0
    ret=float(sel["return_yen"].sum()) if len(sel) else 0.0
    return {
        "model":model,"test_year":test_year,"target_train_top_fraction":frac,
        "score_cutoff":cutoff,"source_races":len(source),"selected_races":len(sel),
        "selected_race_pct":100*len(sel)/len(source) if len(source) else 0.0,
        "direct_hits":int(sel["direct_hit"].sum()) if len(sel) else 0,
        "direct_hit_rate_pct":100*float(sel["direct_hit"].mean()) if len(sel) else 0.0,
        "tickets":int(round(stake/TICKET_PRICE)),
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(sel),
        **concentration(sel),
    }

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

    label_rates=[]
    for y in YEARS:
        d=df[df["year"]==y]
        row={"year":y,"races":len(d)}
        for t in (*TARGETS,"structural_exact","direct_hit"):
            row[f"{t}_count"]=int(d[t].sum())
            row[f"{t}_rate_pct"]=100*float(d[t].mean())
        label_rates.append(row)

    quality=[]; metrics=[]; selected=[]; folds=[]
    for test_year in TEST_YEARS:
        train=df[df["year"]<test_year].copy().reset_index(drop=True)
        test=df[df["year"]==test_year].copy().reset_index(drop=True)
        folds.append({
            "test_year":test_year,
            "train_years":"|".join(map(str,sorted(train["year"].unique()))),
            "train_races":len(train),"test_races":len(test),
        })

        lane_scores={}
        for scope,with_market in (("STRUCT",False),("MARKET",True)):
            tr_probs={}; te_probs={}
            for target in TARGETS:
                ytr=train[target].astype(int); yte=test[target].astype(int)
                if ytr.nunique()<2:
                    raise SystemExit(f"single target class target={target} test={test_year}")
                model,features=make_model(with_market)
                model.fit(train[features],ytr)
                trp=model.predict_proba(train[features])[:,1]
                tep=model.predict_proba(test[features])[:,1]
                tr_probs[target]=trp; te_probs[target]=tep
                quality.append({
                    "scope":scope,"target":target,"test_year":test_year,
                    "train_races":len(train),"train_positives":int(ytr.sum()),
                    "test_races":len(test),"test_positives":int(yte.sum()),
                    "roc_auc":safe_auc(yte,tep),"pr_auc":safe_ap(yte,tep),
                })

            lane_scores[f"{scope}_EXACT"]=(
                tr_probs["anchor_top2"]*tr_probs["k2_third"],
                te_probs["anchor_top2"]*te_probs["k2_third"],
            )
            lane_scores[f"{scope}_BROAD"]=(
                tr_probs["anchor_top3"]*tr_probs["k2_top3"],
                te_probs["anchor_top3"]*te_probs["k2_top3"],
            )

            if scope=="MARKET":
                tr_price=percentile_against_train(train["price_richness_raw"],train["price_richness_raw"])
                te_price=percentile_against_train(train["price_richness_raw"],test["price_richness_raw"])
                ex_tr,ex_te=lane_scores["MARKET_EXACT"]
                br_tr,br_te=lane_scores["MARKET_BROAD"]
                lane_scores["MARKET_EXACT_PRICE_RICH"]=(ex_tr*tr_price,ex_te*te_price)
                lane_scores["MARKET_BROAD_PRICE_RICH"]=(br_tr*tr_price,br_te*te_price)

        for lane,(tr_score,te_score) in lane_scores.items():
            scored=test.copy(); scored["score"]=te_score
            for frac in FRACTIONS:
                cutoff=float(np.quantile(tr_score,1.0-frac))
                sel=scored[scored["score"]>=cutoff].copy()
                metrics.append(metric_row(sel,test,lane,test_year,frac,cutoff))
                for r in sel.itertuples(index=False):
                    selected.append({
                        "model":lane,"test_year":test_year,
                        "target_train_top_fraction":frac,"score_cutoff":cutoff,
                        "score":float(r.score),"race_id":r.race_id,"race_date":r.race_date,
                        "stake_yen":float(r.stake_yen),"return_yen":float(r.return_yen),
                        "profit_yen":float(r.profit_yen),"direct_hit":int(r.direct_hit),
                    })

        metrics.append(metric_row(test,test,"ALL_DANGER",test_year,1.0,float("-inf")))

    lanes=sorted({r["model"] for r in metrics if r["model"]!="ALL_DANGER"})
    stability=[]
    for lane in lanes:
        for frac in FRACTIONS:
            rs=[r for r in metrics if r["model"]==lane and abs(float(r["target_train_top_fraction"])-frac)<1e-12]
            if len(rs)!=3: raise SystemExit(f"missing lane metrics {lane} frac={frac}")
            stake=sum(r["stake_yen"] for r in rs); ret=sum(r["return_yen"] for r in rs)
            wo=[r["roi_without_top1_pct"] if r["roi_without_top1_pct"] is not None else 0.0 for r in rs]
            shares=[r["top1_return_share_pct"] if r["top1_return_share_pct"] is not None else 100.0 for r in rs]
            stability.append({
                "model":lane,"target_train_top_fraction":frac,
                "roi_2023":next(r["roi_pct"] for r in rs if r["test_year"]==2023),
                "roi_2024":next(r["roi_pct"] for r in rs if r["test_year"]==2024),
                "roi_2025":next(r["roi_pct"] for r in rs if r["test_year"]==2025),
                "min_year_roi_pct":min((r["roi_pct"] or 0.0) for r in rs),
                "combined_roi_pct":100*ret/stake if stake else None,
                "combined_profit_yen":ret-stake,
                "selected_races":sum(r["selected_races"] for r in rs),
                "min_selected_races":min(r["selected_races"] for r in rs),
                "min_roi_without_top1_pct":min(wo),
                "max_top1_return_share_pct":max(shares),
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
        "contract":"L2_DANGER_DECOMPOSED_BUY_GATE_V1",
        "analysis_years":list(YEARS),"test_years":list(TEST_YEARS),
        "locked_years":list(LOCKED_YEARS),
        "source_universe":"Every danger race with at least one K2 novel; no trifecta/exacta odds-completeness filter.",
        "ticket_rule":"A1->A2->every K2 novel and A2->A1->every K2 novel, 100 yen each.",
        "component_targets":{
            "anchor_top2":"A1 and A2 both finish in official Top2.",
            "anchor_top3":"A1 and A2 both finish in official Top3.",
            "k2_third":"At least one K2 novel finishes official 3rd.",
            "k2_top3":"At least one K2 novel finishes official Top3.",
        },
        "composite_scores":{
            "STRUCT_EXACT":"P(anchor_top2)*P(k2_third), structural features only.",
            "STRUCT_BROAD":"P(anchor_top3)*P(k2_top3), structural features only.",
            "MARKET_EXACT":"Same exact decomposition with final WIN-market features added.",
            "MARKET_BROAD":"Same broad decomposition with final WIN-market features added.",
            "MARKET_EXACT_PRICE_RICH":"MARKET_EXACT multiplied by train-relative pre-race WIN-price richness percentile.",
            "MARKET_BROAD_PRICE_RICH":"MARKET_BROAD multiplied by train-relative pre-race WIN-price richness percentile.",
        },
        "price_richness_guard":"Price richness is deterministic from final WIN odds only and is tested as a separate lane; it is not assumed to be true value.",
        "market_caveat":"FINAL_WIN_ODDS_PROXY is historical research only; live deployment needs a pre-bet timestamp policy.",
        "threshold_rule":"All test-year cutoffs come from prior-year TRAIN score quantiles only.",
        "fractions":list(FRACTIONS),
        "folds":folds,"label_rates":label_rates,"component_quality":quality,
        "stability":stability,"counters":counters,
        "production_promotion":False,
    }
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"folds.csv",folds)
    write_csv(out/"label-rates.csv",label_rates)
    write_csv(out/"component-quality.csv",quality)
    write_csv(out/"selection-metrics.csv",metrics)
    write_csv(out/"stability.csv",stability)
    write_csv(out/"selected-races.csv",selected)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Danger Decomposed Buy Gate V1\n\n"
        "Decomposes the fixed danger trifecta hit into anchor-control and K2-intrusion events, then recombines their probabilities. "
        "Structural-only and structural+WIN-market lanes are compared. Price-rich variants are separate diagnostics, not assumed value. "
        "Payouts are used only for final ticket evaluation. 2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_DANGER_DECOMPOSED_BUY_GATE_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
