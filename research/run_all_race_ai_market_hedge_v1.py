#!/usr/bin/env python3
"""Retrospective all-race ROI split and 100-yen favorite hedge.
Reuses exact precommitted ONE-PICK_V1 selections; no fitting, data downloads,
outcome-informed selection, or arbitrary strategy filtering.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

YEARS=(2023,2024,2025)
N_YEAR=3456
N_TOTAL=N_YEAR*len(YEARS)
SOURCE_STRATEGIES=("MARKET_FAVORITE","ANCHOR_FULL")
REQUIRED=["year","_race_id","_horse_id","strategy","target_win","final_win_odds"]
EXPECTED={
    "MARKET_FAVORITE": {"wins":3488,"return_yen":823730.0},
    "ANCHOR_FULL":{"wins":1917,"return_yen":902520.0}
}

def require(condition,message):
    if not bool(condition):
        raise ValueError(message)

def settle(d,col):
    """Historical V1 payout approximation: 100 times final decimal odds on win."""
    return np.where(d[f"target_win_{col}"].to_numpy(dtype=int).eq(1)
                    if hasattr(d[f"target_win_{col}"].to_numpy(dtype=int),"eq")
                    else d[f"target_win_{col}"].to_numpy(dtype=int)==1,
                    d[f"final_win_odds_{col}"].to_numpy(dtype=float)*100.0,0.0)

def metrics(d,prefix,stake,returns):
    n=len(d)
    sum_stake=float(np.sum(stake))
    sum_return=float(np.sum(returns))
    return {
        "bets_or_races":int(n),
        "stake_yen":round(sum_stake,2),
        "return_yen":round(sum_return,2),
        "profit_yen":round(sum_return-sum_stake,2),
        "roi_pct":round(100.0*sum_return/sum_stake,8) if sum_stake else None,
    }

def views(d):
    yield ("2023-2025", d)
    for y in YEARS:
        yield (str(y),d[d["year"].eq(y)].copy())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--picks",required=True)
    ap.add_argument("--out-dir",required=True)
    args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    picks=pd.read_csv(args.picks,compression="infer",usecols=REQUIRED,
                      dtype={"_race_id":str,"_horse_id":str,"strategy":str},
                      low_memory=False)
    require(not picks[REQUIRED].isna().any().any(),"missing picks fields")
    picks["year"]=pd.to_numeric(picks["year"],errors="raise").astype(int)
    require(not picks["year"].eq(2026).any(),"2026 sealed")
    require(set(picks["year"].unique())==set(YEARS),"unexpected years")
    picks["target_win"]=pd.to_numeric(picks["target_win"],errors="raise").astype(int)
    picks["final_win_odds"]=pd.to_numeric(picks["final_win_odds"],errors="raise").astype(float)
    require(picks["target_win"].isin([0,1]).all(),"non-binary winner flags")
    require(np.isfinite(picks["final_win_odds"]).all() and
            picks["final_win_odds"].gt(0).all(),"invalid odds")
    for strategy in SOURCE_STRATEGIES:
        z=picks[picks["strategy"].eq(strategy)]
        require(len(z)==N_TOTAL and z["_race_id"].nunique()==N_TOTAL,
                f"{strategy}: total coverage mismatch")
        for y in YEARS:
            require(z[z["year"].eq(y)]["_race_id"].nunique()==N_YEAR,
                    f"{strategy}/{y}: year coverage mismatch")
    market=picks[picks["strategy"].eq("MARKET_FAVORITE")].drop(columns="strategy")
    ai=picks[picks["strategy"].eq("ANCHOR_FULL")].drop(columns="strategy")
    d=market.merge(ai,on=["year","_race_id"],validate="one_to_one",
                   suffixes=("_market","_ai"))
    require(len(d)==N_TOTAL,"join row count mismatch")
    require(d["_race_id"].nunique()==N_TOTAL,"race unique count mismatch")
    d["same"]=d["_horse_id_market"].eq(d["_horse_id_ai"])
    d["market_win"]=d["target_win_market"].eq(1)
    d["ai_win"]=d["target_win_ai"].eq(1)
    require(not (d["market_win"] & d["ai_win"] & ~d["same"]).any(),
            "different picks both win: check dead heats and five-case classification")
    d["market_return"]=np.where(d["market_win"],d["final_win_odds_market"]*100,0.0)
    d["ai_return"]=np.where(d["ai_win"],d["final_win_odds_ai"]*100,0.0)
    for strategy,col in (("MARKET_FAVORITE","market"),("ANCHOR_FULL","ai")):
        expected=EXPECTED[strategy]
        require(int(d[f"{col}_win"].sum())==expected["wins"],
                f"baseline wins drift {strategy}")
        require(abs(float(d[f"{col}_return"].sum())-expected["return_yen"])<.01,
                f"baseline returns drift {strategy}")
    require(int(d["same"].sum())==2340,"same-selection count drift")
    d["outcome"]=np.select([
        d["same"] & d["market_win"],
        d["same"] & ~d["market_win"],
        ~d["same"] & d["market_win"],
        ~d["same"] & d["ai_win"],
    ],["AGREE_HIT","AGREE_MISS","DISAGREE_MARKET_HIT",
       "DISAGREE_AI_HIT"],default="DISAGREE_BOTH_MISS")
    d["combined_stake_yen"]=np.where(d["same"],100.,200.)
    d["combined_return_yen"]=np.where(d["same"],
                                       d["ai_return"],
                                       d["ai_return"]+d["market_return"])
    d["combined_any_hit"]=(d["market_win"]|d["ai_win"])
    segment_rows=[]
    portfolio_rows=[]
    counts=[]
    for year, group in views(d):
        require(len(group)==(N_TOTAL if year=="2023-2025" else N_YEAR),
                f"{year} missing races")
        for segment,sub in (
            ("ALL",group),
            ("AGREE",group[group["same"]]),
            ("DISAGREE",group[~group["same"]]),
        ):
            n=len(sub)
            require(n>0,f"{year}/{segment} no races")
            market_stake=100.*n
            ret_m=float(sub["market_return"].sum())
            ret_a=float(sub["ai_return"].sum())
            wins_m=int(sub["market_win"].sum())
            wins_a=int(sub["ai_win"].sum())
            segment_rows.append({
                "year":year,"segment":segment,"races":n,
                "pct_of_year":round(100*n/len(group),6),
                "stake_per_selector_yen":round(market_stake,2),
                "market_wins":wins_m,"market_hit_pct":round(100*wins_m/n,6),
                "market_return_yen":round(ret_m,2),
                "market_roi_pct":round(100*ret_m/market_stake,6),
                "ai_wins":wins_a,"ai_hit_pct":round(100*wins_a/n,6),
                "ai_return_yen":round(ret_a,2),
                "ai_roi_pct":round(100*ret_a/market_stake,6),
                "ai_minus_market_return_yen":round(ret_a-ret_m,2),
                "ai_minus_market_roi_pp":round(100*(ret_a-ret_m)/market_stake,6),
                "market_winning_avg_odds":round(ret_m/(100*wins_m),5) if wins_m else None,
                "ai_winning_avg_odds":round(ret_a/(100*wins_a),5) if wins_a else None,
            })
        for strategy in ("MARKET_100_ALL","AI_100_ALL","HEDGE_AI_100_PLUS_MARKET_100_IF_DISAGREE",
                         "TWO_TICKETS_EVERY_RACE_UNNECESSARY_DUPLICATE"):
            if strategy=="MARKET_100_ALL":
                stake=np.full(len(group),100.)
                ret=group["market_return"].to_numpy(dtype=float)
                winning=int(group["market_win"].sum())
                tickets=len(group)
            elif strategy=="AI_100_ALL":
                stake=np.full(len(group),100.)
                ret=group["ai_return"].to_numpy(dtype=float)
                winning=int(group["ai_win"].sum())
                tickets=len(group)
            elif strategy=="HEDGE_AI_100_PLUS_MARKET_100_IF_DISAGREE":
                stake=group["combined_stake_yen"].to_numpy(dtype=float)
                ret=group["combined_return_yen"].to_numpy(dtype=float)
                winning=int(group["combined_any_hit"].sum())
                tickets=int(stake.sum()/100)
            else:
                stake=np.full(len(group),200.)
                ret=group["market_return"].to_numpy(dtype=float)+group["ai_return"].to_numpy(dtype=float)
                winning=int(group["combined_any_hit"].sum())
                tickets=2*len(group)
            stake_yen=float(stake.sum())
            return_yen=float(ret.sum())
            portfolio_rows.append({
                "year":year,"strategy":strategy,"races":len(group),
                "tickets":tickets,"hit_races":winning,"hit_race_pct":round(100*winning/len(group),6),
                "stake_yen":round(stake_yen,2),
                "return_yen":round(return_yen,2),
                "profit_yen":round(return_yen-stake_yen,2),
                "roi_pct":round(100*return_yen/stake_yen,6),
            })
        for outcome,n in group["outcome"].value_counts().items():
            counts.append({"year":year,"outcome":outcome,"races":int(n),
                           "pct_of_races":round(100*n/len(group),6)})
    seg=pd.DataFrame(segment_rows)
    portfolio=pd.DataFrame(portfolio_rows)
    outcounts=pd.DataFrame(counts)
    require(int(outcounts[outcounts["year"].eq("2023-2025")]["races"].sum())==N_TOTAL,
            "outcome partition incomplete")
    pooled=seg[seg["year"].eq("2023-2025")].set_index("segment")
    agreed=pooled.loc["AGREE"]
    disagreed=pooled.loc["DISAGREE"]
    require(abs(float(agreed["ai_minus_market_return_yen"]))<.01,
            "matched horse did not settle identically")
    require(abs(float(disagreed["ai_minus_market_return_yen"])-78790.0)<.01,
            "disagreement mismatch against historical +78790 yen")
    market_disagree_roi=float(disagreed["market_roi_pct"])
    ai_pooled_roi=float(pooled.loc["ALL","ai_roi_pct"])
    hedge=portfolio[(portfolio["year"].eq("2023-2025")) &
                    portfolio["strategy"].eq("HEDGE_AI_100_PLUS_MARKET_100_IF_DISAGREE")].iloc[0]
    ai_base=portfolio[(portfolio["year"].eq("2023-2025")) &
                     portfolio["strategy"].eq("AI_100_ALL")].iloc[0]
    summary={
        "contract":"ALL_RACE_AI_MARKET_HEDGE_V1",
        "source":"existing all-race-one-picks.csv.gz from Run 38668174639",
        "years":list(YEARS),"races":N_TOTAL,
        "agreement_races":int(d["same"].sum()),
        "disagreement_races":int((~d["same"]).sum()),
        "strategy":"fixed 100-yen win stake: always AI; additionally buy favorite for 100 yen only if AI selected a different horse",
        "overall_market_roi_pct":float(pooled.loc["ALL","market_roi_pct"]),
        "overall_ai_roi_pct":ai_pooled_roi,
        "favorite_only_in_disagreement_roi_pct":market_disagree_roi,
        "ai_only_in_disagreement_roi_pct":float(disagreed["ai_roi_pct"]),
        "hedged_overall_roi_pct":float(hedge["roi_pct"]),
        "hedged_vs_ai_roi_pp":round(float(hedge["roi_pct"])-float(ai_base["roi_pct"]),6),
        "hedged_vs_ai_profit_yen":round(float(hedge["profit_yen"])-float(ai_base["profit_yen"]),2),
        "cashflow_notice":"hedged strategy risks extra capital on disagreement; all ROI is stakes weighted",
        "independent_hedge_break_even_roi_pct":100.0,
        "hedge_improves_base_ai_roi_if_added_favorite_roi_exceeds_pct":ai_pooled_roi,
        "payout_approximation":"final_win_odds * 100 yen; JRA exact paid dividend unverified",
        "closing_odds_bias":"final odds may not be observable at betting time",
        "betting_outcome":"retrospective only; no fitted router, no post-outcome switching",
        "2026_locked":True,
    }
    seg.to_csv(out/"agree-vs-disagree-roi.csv",index=False)
    portfolio.to_csv(out/"hedge-strategies.csv",index=False)
    outcounts.to_csv(out/"outcome-breakdown.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("==== AGREE VS DISAGREE ==== ",flush=True)
    print(seg.to_csv(index=False),flush=True)
    print("==== ALL RACE PORTFOLIO ==== ",flush=True)
    print(portfolio.to_csv(index=False),flush=True)
    print("==== DECISION SUMMARY ==== ",flush=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print("ALL_RACE_AI_MARKET_HEDGE_V1_COMPLETE",flush=True)

if __name__=="__main__":
    main()
