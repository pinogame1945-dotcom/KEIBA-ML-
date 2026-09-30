#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_exacta_decomposed_v1 import YEARS, build_pair_year_frame, pair_feature_columns, train_pair_ranker

TEST_YEARS=(2023,2024,2025)
K_VALUES=tuple(range(1,16))
STAKE=100.0

def parse_args():
    p=argparse.ArgumentParser(description="EXACTA bidirectional diagnosis on walk-forward ML unordered-pair ranks.")
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1); out[int(y)]=p
    if set(out)!=set(YEARS) or 2026 in out:
        raise SystemExit("L1.7 years must be exactly 2022-2025; 2026 sealed")
    return out

def policy_arrays(g,policy):
    qab=g["market_q_a_to_b"].to_numpy(float)
    qba=g["market_q_b_to_a"].to_numpy(float)
    rab=g["return_a_to_b"].to_numpy(float)
    rba=g["return_b_to_a"].to_numpy(float)
    dv=g["dir_expert_vote_margin"].to_numpy(float)

    if policy=="MARKET_ONE":
        choose_ab=qab>=qba
        ret=np.where(choose_ab,rab,rba)
        tickets=np.ones(len(g),dtype=float)
    elif policy=="L17_ONE":
        # Positive vote margin means canonical A is preferred by more experts.
        # Tie is broken deterministically toward the market-favored direction.
        choose_ab=np.where(dv>0,True,np.where(dv<0,False,qab>=qba))
        ret=np.where(choose_ab,rab,rba)
        tickets=np.ones(len(g),dtype=float)
    elif policy=="BOTH_DIRECTIONS":
        ret=rab+rba
        tickets=np.full(len(g),2.0,dtype=float)
    elif policy=="PERFECT_DIRECTION_CEILING":
        # Evaluation-only upper bound: one ticket per selected pair, but if the
        # selected pair is the winning first-two pair, credit the winning direction.
        # This is intentionally non-deployable and uses outcomes only to quantify
        # how much of the remaining gap is direction prediction.
        ret=np.maximum(rab,rba)
        tickets=np.ones(len(g),dtype=float)
    else:
        raise ValueError(policy)
    return ret,tickets

def evaluate(df,year,k,policy):
    g=df[df["pair_model_rank"]<=k].copy()
    ret,tickets=policy_arrays(g,policy)
    stake=float(tickets.sum()*STAKE)
    total_return=float(ret.sum())
    races=int(g["race_id"].nunique())
    winning_ticket_mask=ret>0
    hit_races=int(g.loc[winning_ticket_mask,"race_id"].nunique())
    wins=np.sort(ret[ret>0])[::-1]
    top1=float(wins[0]) if len(wins) else 0.0
    top3=float(wins[:3].sum()) if len(wins) else 0.0
    return {
        "year":year,
        "top_k":k,
        "policy":policy,
        "source_races":int(df["race_id"].nunique()),
        "executed_races":races,
        "tickets":int(tickets.sum()),
        "tickets_per_race":float(tickets.sum()/races) if races else 0.0,
        "hit_races":hit_races,
        "race_hit_rate_pct":100.0*hit_races/races if races else 0.0,
        "stake_yen":stake,
        "return_yen":total_return,
        "profit_yen":total_return-stake,
        "roi_pct":100.0*total_return/stake if stake else 0.0,
        "roi_minus_top1_pct":100.0*(total_return-top1)/stake if stake else 0.0,
        "roi_minus_top3_pct":100.0*(total_return-top3)/stake if stake else 0.0,
        "largest_return_yen":top1,
    }

def stability(rows):
    m=pd.DataFrame(rows)
    out=[]
    for (k,policy),g in m.groupby(["top_k","policy"],sort=False):
        rois=g["roi_pct"].astype(float).tolist()
        out.append({
            "top_k":int(k),
            "policy":policy,
            "years":"2023|2024|2025",
            "mean_roi_pct":float(np.mean(rois)),
            "median_roi_pct":float(np.median(rois)),
            "worst_year_roi_pct":float(np.min(rois)),
            "best_year_roi_pct":float(np.max(rois)),
            "roi_std_pct":float(np.std(rois,ddof=0)),
            "profitable_years":int(sum(x>100 for x in rois)),
            "nonlosing_years":int(sum(x>=100 for x in rois)),
            "total_stake_yen":float(g["stake_yen"].sum()),
            "total_return_yen":float(g["return_yen"].sum()),
            "total_profit_yen":float(g["profit_yen"].sum()),
            "pooled_roi_pct":100.0*float(g["return_yen"].sum())/float(g["stake_yen"].sum()),
            "worst_year_roi_minus_top1_pct":float(g["roi_minus_top1_pct"].min()),
        })
    return pd.DataFrame(out)

def main():
    a=parse_args(); started=time.perf_counter()
    lp=parse_paths(a.l17_year)

    frames={}
    for y in YEARS:
        frames[y]=build_pair_year_frame(y,load_l17(lp[y],y),a.backfill_root)

    cols=pair_feature_columns(frames[2022],market=True)
    if any(c.startswith("dir_") for c in cols):
        raise SystemExit("directional leakage into unordered pair ranker")

    all_rows=[]; fold_rows=[]; capture_rows=[]
    policies=("MARKET_ONE","L17_ONE","BOTH_DIRECTIONS","PERFECT_DIRECTION_CEILING")

    for y in TEST_YEARS:
        train=pd.concat([frames[t] for t in YEARS if t<y],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)
        pred,_,secs=train_pair_ranker(train,test,cols,161000+y)
        test["pair_model_score"]=pred
        test["pair_model_rank"]=test.groupby("race_id")["pair_model_score"].rank(
            method="min",ascending=False
        ).astype(int)

        source=int(test["race_id"].nunique())
        for k in K_VALUES:
            cap=int(test.loc[(test["pair_hit"]==1)&(test["pair_model_rank"]<=k),"race_id"].nunique())
            capture_rows.append({
                "year":y,"top_k":k,"source_races":source,
                "captured_races":cap,
                "pair_capture_pct":100.0*cap/source if source else 0.0,
            })
            for p in policies:
                all_rows.append(evaluate(test,y,k,p))

        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(str(t) for t in YEARS if t<y),
            "train_pair_rows":len(train),
            "test_pair_rows":len(test),
            "feature_count":len(cols),
            "train_seconds":secs,
        })
        print("BIDIR_FOLD_READY "+json.dumps({
            "test_year":y,
            "train_years":[t for t in YEARS if t<y],
            "top10_pair_capture_pct":[x for x in capture_rows if x["year"]==y and x["top_k"]==10][0]["pair_capture_pct"],
            "train_seconds":round(secs,3),
        },separators=(",",":")),flush=True)

    metrics=pd.DataFrame(all_rows)
    stable=stability(all_rows)

    # Diagnostic: how much ROI is lost to direction uncertainty at equal selected pairs.
    diag=[]
    for y in TEST_YEARS:
        for k in K_VALUES:
            g=metrics[(metrics.year==y)&(metrics.top_k==k)].set_index("policy")
            both=float(g.loc["BOTH_DIRECTIONS","roi_pct"])
            market=float(g.loc["MARKET_ONE","roi_pct"])
            l17=float(g.loc["L17_ONE","roi_pct"])
            ceiling=float(g.loc["PERFECT_DIRECTION_CEILING","roi_pct"])
            diag.append({
                "year":y,"top_k":k,
                "market_one_roi_pct":market,
                "l17_one_roi_pct":l17,
                "both_directions_roi_pct":both,
                "perfect_direction_ceiling_roi_pct":ceiling,
                "ceiling_minus_market_points":ceiling-market,
                "ceiling_minus_l17_points":ceiling-l17,
                "both_break_even":bool(both>=100.0),
                "perfect_direction_break_even":bool(ceiling>=100.0),
            })
    diag=pd.DataFrame(diag)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(out/"policy-metrics-by-year-k.csv",index=False)
    stable.to_csv(out/"stability-summary.csv",index=False)
    diag.to_csv(out/"direction-diagnosis.csv",index=False)
    pd.DataFrame(capture_rows).to_csv(out/"mlpair-foundation.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(out/"mlpair-folds.csv",index=False)

    primary=stable[stable["top_k"].isin([1,3,5,10,15])].copy()
    primary.to_csv(out/"primary-k-summary.csv",index=False)

    summary={
        "contract":"L2_EXACTA_BIDIRECTIONAL_V1",
        "question":"Is exacta instability primarily direction prediction or the price structure itself?",
        "candidate_generator":"walk-forward decomposed market-aware ML unordered-pair rank",
        "policies":{
            "MARKET_ONE":"one direction per selected pair, market-favored orientation",
            "L17_ONE":"one direction per selected pair, Seven-King vote orientation; market breaks ties",
            "BOTH_DIRECTIONS":"both A>B and B>A, two tickets per selected pair",
            "PERFECT_DIRECTION_CEILING":"evaluation-only one-ticket upper bound using realized winning direction"
        },
        "top_k":list(K_VALUES),
        "flat_stake_yen_per_ticket":100,
        "law_search":False,
        "test_years":list(TEST_YEARS),
        "walk_forward":"all prior years only",
        "primary_metrics":["ROI by year","worst-year ROI","profitable-year count","tail robustness"],
        "2026_locked":True,
        "script_total_seconds":time.perf_counter()-started,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 EXACTA Bidirectional V1\n\n"
        "This diagnosis holds the walk-forward ML unordered-pair candidate set fixed and varies only orientation policy. "
        "MARKET_ONE buys the market-favored direction; L17_ONE buys the Seven-King-vote direction; BOTH_DIRECTIONS buys "
        "both exacta directions. PERFECT_DIRECTION_CEILING is outcome-aware and evaluation-only, representing the maximum "
        "one-direction ROI possible if orientation were predicted perfectly for every selected pair. No LAW search or "
        "test-year tuning is performed. 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("L2_EXACTA_BIDIRECTIONAL_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
    print("===== PRIMARY K SUMMARY =====")
    print(primary.to_csv(index=False))
    print("===== DIRECTION DIAGNOSIS TOP10 =====")
    print(diag[diag.top_k==10].to_csv(index=False))

if __name__=="__main__":
    main()
