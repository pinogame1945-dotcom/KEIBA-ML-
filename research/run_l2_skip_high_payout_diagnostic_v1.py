#!/usr/bin/env python3
"""Diagnose WHY skip models discarded large dividends; safeguard study.

Inputs are saved frozen 2023-25 feature snapshots, exact per-race ticket
settlement and 2024-25 OOS scores from L2_SKIP_VALUE_CHRONOLOGICAL_BATCH_V1.
There is NO refit, no new Backfill fetch, and no 2026 examination.

Score thresholds are reconstructed to match the predecessor's report exactly;
this reconstruction uses ONLY the count of flags already saved in the previous
experiment, and verifies full year-by-year payout/hit parity. It must NOT
be interpreted as a newly deployable model or fresh holdout.

Safeguard feature thresholds are determined exclusively from prior-year
feature distributions. Select safeguard with 2024 results, report fixed
choice on 2025 (known inspected year, NOT pristine blind test).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

RULES=(
 "TRIO|AI_PROB|2",
 "TRIFECTA_FIRST|AI_PROB|2",
 "WIN|SINGLE|0",
 "EXACTA_FIRST|MARKET|2",
 "WIDE|MARKET|2",
)
POLS=(
 ("TRIO|AI_PROB|2","UTILITY_REGRESSOR",10),
 ("TRIO|AI_PROB|2","UTILITY_REGRESSOR",20),
 ("TRIO|AI_PROB|2","UTILITY_REGRESSOR",30),
 ("TRIO|AI_PROB|2","UTILITY_REGRESSOR",40),
 ("TRIO|AI_PROB|2","HIT_CLASSIFIER",10),
 ("TRIO|AI_PROB|2","HIT_CLASSIFIER",20),
 ("TRIO|AI_PROB|2","RISK_ONLY",10),
 ("TRIO|AI_PROB|2","RISK_ONLY",20),
 ("TRIFECTA_FIRST|AI_PROB|2","RISK_ONLY",20),
 ("WIN|SINGLE|0","UTILITY_REGRESSOR",10),
 ("WIN|SINGLE|0","RISK_ONLY",10),
 ("EXACTA_FIRST|MARKET|2","UTILITY_REGRESSOR",10),
 ("WIDE|MARKET|2","UTILITY_REGRESSOR",20),
)
DIAG=(
 ("TRIO|AI_PROB|2","UTILITY_REGRESSOR",10),
 ("TRIO|AI_PROB|2","UTILITY_REGRESSOR",20),
 ("TRIO|AI_PROB|2","UTILITY_REGRESSOR",40),
 ("TRIFECTA_FIRST|AI_PROB|2","RISK_ONLY",20),
 ("WIN|SINGLE|0","RISK_ONLY",10),
)
GUARDS=(
 "NONE",
 "HIGH_ODDS_Q75","HIGH_ODDS_Q90",
 "HIGH_EDGE_Q75","HIGH_AI_PROB_Q75",
 "HIGH_RANK_5PLUS","LOW_RISK_Q25",
 "FAVORITE_RANK_1OR2",
 "HIGH_RELATIVE_ODDS_Q80",
 "CONSENSUS_CLASSIFIER_20",
 "CONSENSUS_UTILITY_20",
 "CONSENSUS_BOTH_20",
)
BASE_SOURCE_RUN="38680436244"
HEADLINES=("TRIO|AI_PROB|2","TRIFECTA_FIRST|AI_PROB|2")
DETAIL_CUTS=(0,500,1000,3000,10000,30000,100000,10000000)

def ensure(ok,message):
    if not bool(ok):raise AssertionError(message)

def input_df(path,ids=True):
    kw={"low_memory":False}
    if ids:kw["dtype"]={"race_id":str}
    return pd.read_csv(path,compression="infer",**kw)

def expected_mask(arr, count):
    """Reconstruct strictly > TRAIN CUTOFF from previous saved COUNT.

    A unique gap between top-n and remainder is necessary to reconstruct exact
    flags with no guessing; checks exact payout totals AND hits afterward.
    """
    scores=np.asarray(arr,dtype=float)
    n=len(scores)
    ensure(np.isfinite(scores).all(),"nonfinite model scores")
    ensure(0<=count<=n,f"wrong saved skip count {count}/{n}")
    if count==0:return np.zeros(n,dtype=bool)
    if count==n:return np.ones(n,dtype=bool)
    ss=np.sort(scores)
    lower=ss[n-count-1];upper=ss[n-count]
    ensure(upper>lower, f"ambiguous saved cutoff at tie, count={count}")
    mask=scores>=upper
    ensure(int(mask.sum())==count,"score reconstruction count mismatch")
    return mask

def profit_report(y, rule, model, rate, guard, frame, original_skip, safeguarded_skip):
    st=frame["stake"].to_numpy(float)
    ret=frame["return"].to_numpy(float)
    restore=original_skip & ~safeguarded_skip
    keep=~safeguarded_skip
    paid=float(ret[keep].sum());cost=float(st[keep].sum())
    original_saved=float((st[original_skip]-ret[original_skip]).sum())
    safeguarded_saved=float((st[safeguarded_skip]-ret[safeguarded_skip]).sum())
    won=ret>0
    return {
       "year":y,"rule":rule,"model":model,"target_skip_pct":rate,"guard":guard,
       "races":len(st),"bet_races":int(keep.sum()),
       "actual_skip_races":int(safeguarded_skip.sum()),
       "actual_skip_pct":round(100*float(safeguarded_skip.mean()),5),
       "all_race_stake_yen":float(st.sum()),
       "all_race_return_yen":float(ret.sum()),
       "all_race_profit_yen":float((ret-st).sum()),
       "stake_yen":cost,"return_yen":paid,"net_profit_yen":paid-cost,
       "roi_pct":round(100*paid/cost,6) if cost else np.nan,
       "lost_paid_races":int(np.sum(won & safeguarded_skip)),
       "lost_payout_yen":float(ret[safeguarded_skip].sum()),
       "saved_historical_hit_races_by_guard":int(np.sum(won & restore)),
       "saved_historical_payout_by_guard_yen":float(ret[restore].sum()),
       "reintroduced_stake_by_guard_yen":float(st[restore].sum()),
       "improvement_over_unguarded_yen":safeguarded_saved-original_saved,
       "gain_from_skip_yen":safeguarded_saved,
    }

def guard_allow(frame, name, history, masks):
    n=len(frame)
    if name=="NONE":return np.zeros(n,dtype=bool)
    def q(column,quant):
        return float(np.quantile(history[column].to_numpy(float),quant))
    def col(name):return frame[name].to_numpy(float)
    if name=="HIGH_ODDS_Q75":return col("selected_odds")>=q("selected_odds",.75)
    if name=="HIGH_ODDS_Q90":return col("selected_odds")>=q("selected_odds",.90)
    if name=="HIGH_EDGE_Q75":return col("selected_edge")>=q("selected_edge",.75)
    if name=="HIGH_AI_PROB_Q75":return col("selected_p_ai")>=q("selected_p_ai",.75)
    if name=="HIGH_RANK_5PLUS":return col("selected_market_rank")>=5
    if name=="LOW_RISK_Q25":return col("selected_risk")<=q("selected_risk",.25)
    if name=="FAVORITE_RANK_1OR2":return col("selected_market_rank")<=2
    if name=="HIGH_RELATIVE_ODDS_Q80":return col("price_gap_favorite")>=q("price_gap_favorite",.80)
    if name=="CONSENSUS_CLASSIFIER_20":return ~masks["HIT_CLASSIFIER"]
    if name=="CONSENSUS_UTILITY_20":return ~masks["UTILITY_REGRESSOR"]
    if name=="CONSENSUS_BOTH_20":
        return ~(masks["HIT_CLASSIFIER"] & masks["UTILITY_REGRESSOR"])
    raise ValueError(name)

def lost_payout_bins(df,mask):
    rows=[]
    pool=df.copy()
    pool["_skipped"]=mask
    pool["_hit"]=pool["return"]>0
    pool["_bin"]=pd.cut(pool["return"],DETAIL_CUTS,right=False)
    for text,sub in pool.groupby("_bin",observed=False):
        d=sub.loc[sub["_skipped"]]
        r=sub.loc[~sub["_skipped"]]
        rows.append({
          "payout_band_yen":str(text),"total_bet_races":len(sub),
          "skipped_races":len(d),
          "lost_hit_races":int(d["_hit"].sum()),
          "lost_payout_yen":float(d["return"].sum()),
          "kept_hit_races":int(r["_hit"].sum()),
          "kept_payout_yen":float(r["return"].sum()),
        })
    return rows

def lost_top_concentration(df,mask):
    prizes=df["return"].to_numpy(float)
    lost=(prizes[mask])
    lost=lost[lost>0]
    lost.sort()
    lost=lost[::-1]
    payout_sum=float(lost.sum())
    all_hit=prizes[prizes>0]
    all_hit.sort();all_hit=all_hit[::-1]
    return {
      "lost_hit_count":len(lost),
      "total_lost_payout_yen":payout_sum,
      "lost_largest_single_yen":float(lost[0]) if len(lost) else 0,
      **{f"lost_top_{k}_yen":float(lost[:k].sum()) for k in (1,3,5,10,20)},
      **{f"lost_top_{k}_share_pct":round(100*lost[:k].sum()/payout_sum,4) if payout_sum else 0
         for k in (1,3,5,10,20)},
      "all_hit_largest_1_yen":float(all_hit[0]) if len(all_hit) else 0,
      "all_hit_largest_10_sum_yen":float(all_hit[:10].sum()),
      "all_hit_top10_fraction_of_gross_pct":round(100*all_hit[:10].sum()/float(prizes.sum()),5),
    }

def read_and_audit(root):
    f=input_df(root/"race-feature-snapshot.csv.gz")
    b=input_df(root/"all-10-rules-settled-per-race.csv.gz")
    s=input_df(root/"per-race-oos-skip-scores.csv.gz")
    report=input_df(root/"all-model-year-cutoff-and-quota.csv",ids=False)
    ensure(len(f)==10368 and f.race_id.is_unique,"feature rows drift")
    ensure(set(f.year)=={2023,2024,2025},"feature years drift")
    ensure(len(b)==103680,"cash contract changed")
    ensure(set(s.year)=={2024,2025},"score years drift")
    ensure(s.duplicated(["year","race_id","rule","model"]).sum()==0,"duplicate risk scores")
    b=b[b.rule.isin(RULES)].copy()
    s=s[s.rule.isin(RULES)].copy()
    ensure(len(s)==3456*2*len(RULES)*5,
           f"missing score models {len(s)}")
    pivot=s.pivot(index=["year","race_id","rule"],
                  columns="model",values="skip_score").reset_index()
    all_=b.merge(pivot,on=["year","race_id","rule"],how="inner",
                 validate="one_to_one")
    all_=all_.merge(f,on=["year","race_id"],how="left",validate="many_to_one")
    ensure(len(all_)==3456*2*len(RULES),"joined cache coverage drift")
    ensure(all_.anchor_finish.notna().all(),"missing official finish")
    ensure(all_["stake"].ge(100).all(),"zero ticket")
    return f,b,all_,report

def score_flags(sub,report,y,rule,model,rate,cache):
    key=(y,rule,model,rate)
    if key in cache:return cache[key]
    rr=report[(report.year.astype(str).eq(str(y))) &
       (report.rule.eq(rule)) & (report.model.eq(model)) &
       (report.policy.eq("PRIOR_ONLY_CUTOFF")) &
       (report.target_skip_pct.eq(rate))]
    ensure(len(rr)==1,f"missing old report {key}")
    prior=rr.iloc[0]
    mask=expected_mask(sub[model].to_numpy(float),int(prior.skip_races))
    st=sub["stake"].to_numpy(float);ret=sub["return"].to_numpy(float)
    ensure(int(mask.sum())==int(prior.skip_races),f"count drift {key}")
    ensure(abs(float(ret[mask].sum())-float(prior.forgone_payout_yen))<.0001,
           f"lost payout parity FAIL {key}")
    ensure(int(np.sum((ret>0)&mask))==int(prior.hit_races_lost),
           f"lost hit count parity FAIL {key}")
    ensure(abs(float((st[mask]-ret[mask]).sum())-float(prior.gain_from_skip_yen))<.0001,
           f"profit parity FAIL {key}")
    cache[key]=mask
    return mask

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prior-results-root",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    hist,bets,joined,prev=read_and_audit(Path(args.prior_results_root))
    masks={}
    frames={}
    for y in (2024,2025):
        for rule in RULES:
            df=joined[(joined.year.eq(y)) & (joined.rule.eq(rule))].sort_values("race_id").reset_index(drop=True)
            ensure(len(df)==3456,f"missing year {y} for {rule}")
            frames[(y,rule)]=df
    # reconstruct every saved target for each method to detect score/tie drift.
    n_checked=0
    for (y,rule),df in frames.items():
        for model in ("HIT_CLASSIFIER","UTILITY_REGRESSOR","RISK_ONLY",
                      "ODDS_ONLY","MARKET_RANK_ONLY"):
            for rate in (5,10,15,20,30,40):
                score_flags(df,prev,y,rule,model,rate,masks)
                n_checked+=1

    diag_rows=[]
    bins=[]
    top_lost=[]
    groupby=[]
    for rule,model,rate in DIAG:
        for y in (2024,2025):
            frame=frames[(y,rule)]
            flag=score_flags(frame,prev,y,rule,model,rate,masks)
            nums=lost_top_concentration(frame,flag)
            st=frame["stake"].to_numpy(float);pay=frame["return"].to_numpy(float)
            total_bets=int((pay>0).sum())
            paid_skipped=frame[flag & frame["return"].gt(0)].sort_values(
                ["return","race_id"],ascending=[False,True])
            diag_rows.append({"year":y,"rule":rule,"model":model,"target_skip_pct":rate,
                 "skip_races":int(flag.sum()),"skip_pct":round(100*flag.mean(),5),
                 "paid_races":total_bets,
                 "paid_races_skipped":int((flag & (pay>0)).sum()),
                 "lost_hit_rate_among_skipped_pct":round(100*(flag & (pay>0)).sum()/max(1,flag.sum()),5),
                 "gross_return_yen":float(pay.sum()),
                 "forgone_gross_pct":round(100*float(pay[flag].sum())/max(1.,float(pay.sum())),5),
                 "foregone_return_per_skipped_hit_yen":round(float(pay[flag].sum())/
                          max(1,int((flag & (pay>0)).sum())),2),
                 "net_skip_gain_yen":float((st[flag]-pay[flag]).sum()),
                 **nums})
            for row in lost_payout_bins(frame,flag):
                bins.append({"year":y,"rule":rule,"model":model,"target_skip_pct":rate,**row})
            for rank,(idx,rr) in enumerate(paid_skipped.head(40).iterrows(),1):
                top_lost.append({
                  "year":y,"rule":rule,"model":model,"target_skip_pct":rate,
                  "lost_rank":rank,"race_id":rr.race_id,"race_date":rr.race_date,
                  "actual_payout_yen":rr["return"],
                  "ticket_cost_yen":rr.stake,
                  "actual_anchor_finish_DIAGNOSTIC_ONLY":rr.anchor_finish,
                  "pre_race_selected_odds":rr.selected_odds,
                  "pre_race_selected_market_rank":rr.selected_market_rank,
                  "pre_race_selected_risk":rr.selected_risk,
                  "pre_race_selected_probability":rr.selected_p_ai,
                  "pre_race_anchor_edge":rr.selected_edge,
                  "skip_score":rr[model],
                })
            # interpret coarse feature movements, not post-race features.
            for name,col in [("selected_odds","selected_odds"),
                             ("market_rank","selected_market_rank"),
                             ("risk","selected_risk"),
                             ("p_ai","selected_p_ai"),
                             ("top3_overlap","top3_overlap")]:
                d=frame[flag]
                k=frame[~flag]
                w=frame[flag & frame["return"].gt(0)]
                groupby.append({
                  "year":y,"rule":rule,"model":model,"target_skip_pct":rate,
                  "feature":name,
                  "skipped_median":float(d[col].median()) if len(d) else np.nan,
                  "kept_median":float(k[col].median()) if len(k) else np.nan,
                  "lost_winner_median":float(w[col].median()) if len(w) else np.nan,
                })

    # Payout protection: ALWAYS keep a ticket if a fixed pre-race guard fires.
    # Guard cutoffs fixed from previous calendar years' features, independent
    # of the 2024/2025 test year's outcomes.
    guards=[]
    for rule,model,rate in POLS:
        for y in (2024,2025):
            g=frames[(y,rule)]
            hist_feat=hist[hist.year.lt(y)]
            ensure(len(hist_feat)==(y-2023)*3456,"training feature chronology violation")
            flag=score_flags(g,prev,y,rule,model,rate,masks)
            peer_masks={
             "UTILITY_REGRESSOR":score_flags(g,prev,y,rule,"UTILITY_REGRESSOR",20,masks),
             "HIT_CLASSIFIER":score_flags(g,prev,y,rule,"HIT_CLASSIFIER",20,masks)
            }
            for guard in GUARDS:
                protected=guard_allow(g,guard,hist_feat,peer_masks)
                newmask=flag & ~protected
                ensure(np.all(newmask<=flag),"guard must never add skips")
                guards.append(profit_report(y,rule,model,rate,guard,g,flag,newmask))
    results=pd.DataFrame(guards)
    ensure(len(results)==len(POLS)*2*len(GUARDS),"guard coverage broken")

    # Select purely from 2024: a broad bet coverage mandate, NO test-2025
    # information may be used to choose family, rate, or guard.
    elected=[]
    for mandate,coverage in (("BROAD_80_PCT",0.8),("MODERATE_70_PCT",0.7)):
        candidates=results[(results.year.eq(2024)) &
             (results.bet_races.ge(int(np.ceil(3456*coverage)))) &
             (results.target_skip_pct.le(30))].copy()
        assert len(candidates)
        # Focus absolute profit first; ROI second; then coverage; simple guard.
        candidates["complexity"]=(candidates["guard"].ne("NONE")).astype(int)
        candidates=candidates.sort_values(
            ["net_profit_yen","roi_pct","bet_races","complexity"],
            ascending=[False,False,False,True])
        sel=candidates.iloc[0]
        confirmed=results[(results.year.eq(2025)) &
              (results.rule.eq(sel["rule"])) & (results.model.eq(sel["model"])) &
              (results.target_skip_pct.eq(sel["target_skip_pct"])) &
              (results.guard.eq(sel["guard"]))]
        ensure(len(confirmed)==1,"locked candidate unavailable")
        v=confirmed.iloc[0]
        elected.append({
           "mandate":mandate,"2024_selected_rule":sel["rule"],
           "selected_model":sel["model"],
           "selected_target_skip_pct":int(sel["target_skip_pct"]),
           "selected_protection_guard":sel["guard"],
           "2024_bet_races":int(sel.bet_races),
           "2024_profit_yen":float(sel.net_profit_yen),
           "2024_roi_pct":float(sel.roi_pct),
           "2025_locked_bet_races":int(v.bet_races),
           "2025_locked_profit_yen":float(v.net_profit_yen),
           "2025_locked_roi_pct":float(v.roi_pct),
           "2025_locked_improvement_over_unguarded_yen":float(v.improvement_over_unguarded_yen),
        })
    # Compare fixed BASELINE risk-seen in 2024, TRIO utility10/20 in both years;
    # and multiple loss-tail concentration, without cherry-picked future.
    baseline=[]
    for rule,model,rate in DIAG:
        for yr in (2024,2025):
            base=results[(results.year.eq(yr)) &
               (results.rule.eq(rule)) & (results.model.eq(model)) &
               (results.target_skip_pct.eq(rate)) & (results.guard.eq("NONE"))]
            ensure(len(base)==1,"baseline guard missing")
            baseline.append(base.iloc[0].to_dict())
    summary={
      "contract":"L2_SKIP_BIG_PAYOUT_SACRIFICE_DIAGNOSTIC_V1",
      "source_precommitted_run":BASE_SOURCE_RUN,
      "years_in_history":[2023,2024,2025],
      "evaluated_years":[2024,2025],
      "total_races_2023_2025":10368,
      "score_cuttoff_reconstructed_flags_parity":f"PASS: {n_checked} year/rule/model/rate combinations hit/count/profit exact",
      "distinct_rules_diagnosed":len(RULES),
      "guard_variants_per_policy":len(GUARDS),
      "policies_tested":len(POLS),
      "guard_candidate_rule":"Subsets of prior skip flags only, so never adds risk via extra skips",
      "guard_feature_threshold":"Prior year feature quantiles; 2024 uses 2023, 2025 uses 2023-2024",
      "feature_price_warning":"Frozen AI value axis and several features use FINAL odds, not a deployable pre-off indicator",
      "baseline_preregistered_comparators":baseline,
      "2024_elections_and_locked_2025_evaluation":elected,
      "target_label_warning":"All 'lost high payout' analyses are post-hoc diagnostics. Labels never used to make per-race flags.",
      "multiple_comparison_warning":"Several 2024 candidate guards selected by profit, 2025 previously inspected and not pristine OOS.",
      "outlook":"Safeguards are candidate diagnostics, not production ML.",
      "promotion":False,"paid_runner":False,"2026_locked":True,
    }
    pd.DataFrame(diag_rows).to_csv(out/"lost-winning-ticket-concentration.csv",index=False)
    pd.DataFrame(bins).to_csv(out/"lost-payout-by-amount-band.csv",index=False)
    pd.DataFrame(top_lost).to_csv(out/"top-40-lost-prizes-with-pre-race-context.csv",index=False)
    pd.DataFrame(groupby).to_csv(out/"skip-and-lost-winner-feature-drift.csv",index=False)
    results.to_csv(out/"2024-2025-feature-guard-safety-evaluation.csv",index=False)
    pd.DataFrame(elected).to_csv(out/"2024-choice-locked-2025-validation.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str)+"\n")
    print("L2_BIG_PAYOUT_DIAGNOSTIC_RESULT",flush=True)
    print(json.dumps(elected,ensure_ascii=False),flush=True)
    print("LOST_TOP_CONCENTRATION")
    print(pd.DataFrame(diag_rows)[["year","rule","model","target_skip_pct","skip_races",
       "paid_races_skipped","total_lost_payout_yen","lost_top_10_yen",
       "lost_top_10_share_pct","net_skip_gain_yen"]].to_csv(index=False),flush=True)
    print("L2_SKIP_BIG_PAYOUT_DIAGNOSTIC_COMPLETE",flush=True)

if __name__=="__main__":
    main()
