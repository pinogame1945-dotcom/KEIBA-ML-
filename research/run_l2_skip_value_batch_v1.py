#!/usr/bin/env python3
"""L2 skip-value batch, strictly chronological backtest, 2023-2025.

Part A: freeze expected-value axis and exact historical cash settlement for all
standard racing bet formats (the 7 physical anchor-containing ticket families).
Part B: compare odds/risk baseline versus classifier and utility regressors.
Part C: fixed skip quotas, both deployable TRAIN-score cutoff and diagnostic
TEST-year score rank, with ORACLE clearly labelled.
Part D: 2024 candidate selection then 2025 policy confirmation.

Caveat: existing anchor selected using FINAL_WIN_ODDS; the outer
2024/25 train/test time split does not cure that operational lookahead.
No 2026, no paid computation, no test-year target fitting.
"""
import argparse
import concurrent.futures
import json
import math
import os
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,get_inputs,assert_ok
from build_l2_market_helpers_v1 import load_day,horse_number_map
from run_l2_value_anchor_bet_frontier_v1 import payouts,pair_templates
from run_value_anchor_official_positions_v1 import positions

BET_RULES=(
 ("WIN","SINGLE",0),
 ("PLACE","SINGLE",0),
 ("TRIO","AI_PROB",2),
 ("WIDE","MARKET",2),
 ("WIDE","MARKET",3),
 ("WIDE","AI_PROB",7),
 ("EXACTA_FIRST","MARKET",2),
 ("EXACTA_FIRST","MARKET",7),
 ("QUINELLA","AI_PROB",2),
 ("TRIFECTA_FIRST","AI_PROB",2),
)
SKIP_PCTS=(0,5,10,15,20,30,40)
STUDY_YEARS=(2024,2025)
MODES=("HIT_CLASSIFIER","UTILITY_REGRESSOR","ODDS_ONLY","MARKET_RANK_ONLY","RISK_ONLY")
FEATURES=(
 "selected_market_rank","selected_ai_rank","selected_odds","selected_log_odds",
 "selected_p_ai","selected_p_market","selected_risk",
 "selected_edge","selected_log_edge","prob_gap","rank_gap",
 "field_size","market_top1_odds","market_top2_odds",
 "ai_top1_p","ai_top2_p","ai_top3_p","market_top1_p",
 "ai_entropy","market_entropy","top3_overlap","max_p_gap",
 "market_rank_percentile","selected_market_prob_share","ai_share_ratio",
 "price_gap_favorite","odds_field_dispersion",
)
HEURISTICS=("ODDS_ONLY","MARKET_RANK_ONLY","RISK_ONLY")
BASELINE_ROIS={
 "WIN|SINGLE|0":86.898148,
 "PLACE|SINGLE|0":83.490548,
 "TRIO|AI_PROB|2":88.888889,
 "WIDE|MARKET|2":85.940876,
 "WIDE|MARKET|3":83.900141,
 "WIDE|AI_PROB|7":80.342394,
 "EXACTA_FIRST|MARKET|2":88.713349,
 "EXACTA_FIRST|MARKET|7":84.77146,
 "QUINELLA|AI_PROB|2":84.415027,
 "TRIFECTA_FIRST|AI_PROB|2":84.552951,
}

def ridrule(fam,src,k):return f"{fam}|{src}|{k}"
def nn(v,default=0.0):
    try:
        f=float(v)
        return f if math.isfinite(f) else default
    except (ValueError,TypeError):
        return default

def race_features(group,axis):
    g=group.copy()
    h=g.set_index("_horse_id",drop=False)
    assert_ok(axis in h.index,f"selected anchor {axis} missing")
    # Race information computed ONLY from predictions/mkt fields.
    for name in ("market_rank","final_win_odds","P__ANCHOR_FULL",
                 "market_prob_norm","P__MARKET_DIRECT","risk",
                 "distill_ai_rank","top3_overlap"):
        g[name]=pd.to_numeric(g[name],errors="coerce")
    a=g[g["_horse_id"].eq(axis)].iloc[0]
    g_m=g.sort_values(["market_rank","final_win_odds","_horse_id"],
                 ascending=[True,True,True])
    g_ai=g.sort_values(["P__ANCHOR_FULL","market_rank","_horse_id"],
                  ascending=[False,True,True])
    size=len(g)
    odds=np.maximum(g["final_win_odds"].to_numpy(dtype=float),1.0)
    p=np.maximum(g["P__ANCHOR_FULL"].to_numpy(dtype=float),1.e-9)
    p=p/p.sum()
    q=np.maximum(g["market_prob_norm"].to_numpy(dtype=float),1.e-9)
    q=q/q.sum()
    rank=float(a["market_rank"])
    ai_rank=float(a["distill_ai_rank"])
    odds_a=float(a["final_win_odds"])
    prob=float(a["P__ANCHOR_FULL"])
    mp=float(a["market_prob_norm"])
    fav_odd=nn(g_m.iloc[0]["final_win_odds"],1.0)
    second_odd=nn(g_m.iloc[min(1,size-1)]["final_win_odds"],fav_odd)
    pa=[nn(g_ai.iloc[min(j,size-1)]["P__ANCHOR_FULL"]) for j in range(3)]
    prob_gap=prob-mp
    result={
      "selected_market_rank":rank,
      "selected_ai_rank":ai_rank,
      "selected_odds":odds_a,
      "selected_log_odds":math.log1p(odds_a),
      "selected_p_ai":prob,
      "selected_p_market":mp,
      "selected_risk":nn(a["risk"]),
      "selected_edge":prob*odds_a-1,
      "selected_log_edge":math.log(max(.01,prob*odds_a)),
      "prob_gap":prob_gap,
      "rank_gap":rank-ai_rank,
      "field_size":size,
      "market_top1_odds":fav_odd,
      "market_top2_odds":second_odd,
      "ai_top1_p":pa[0],
      "ai_top2_p":pa[1],
      "ai_top3_p":pa[2],
      "market_top1_p":nn(g_m.iloc[0]["market_prob_norm"]),
      "ai_entropy":float(-(p*np.log(p)).sum()),
      "market_entropy":float(-(q*np.log(q)).sum()),
      "top3_overlap":nn(g_m.iloc[0]["top3_overlap"]),
      "max_p_gap":nn(g_ai.iloc[0]["P__ANCHOR_FULL"])-nn(g_m.iloc[0]["P__ANCHOR_FULL"]),
      "market_rank_percentile":(rank-1)/max(size-1,1),
      "selected_market_prob_share":mp,
      "ai_share_ratio":prob/max(mp,1.e-6),
      "price_gap_favorite":odds_a/max(fav_odd,1),
      "odds_field_dispersion":float(np.std(np.log1p(odds))),
    }
    assert_ok(all(math.isfinite(float(z)) for z in result.values()),"NaN in raw predictors")
    return result

def collect_year(year,all_data,picks,root):
    data=all_data[all_data.year.eq(year)].copy()
    bydate=data[["_race_id","_race_date"]].drop_duplicates().groupby("_race_date")
    groups=data.groupby("_race_id",sort=False)
    feat_rows=[];out_rows=[]
    for date,subset in bydate:
        ids=subset["_race_id"].astype(str).tolist()
        packs=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",set(ids))
        for rid in ids:
            pack=packs.get(rid)
            assert_ok(pack is not None,f"payout day pack missing {rid}")
            group=groups.get_group(rid)
            axis=picks["ANCHOR_FULL"][rid]
            nos=horse_number_map(pack)
            assert_ok(axis in nos and set(group["_horse_id"]).issubset(nos),
                      f"horse number coverage failed {rid}")
            available=payouts(pack,rid)
            finish=positions(pack).get(axis)
            assert_ok(finish is not None, f"official result missing {rid}")
            a=group[group["_horse_id"].eq(axis)].iloc[0]
            assert_ok(int(a["target_win"])==int(finish==1) and
                      int(a["target_top3"])==int(finish<=3),
                      f"official outcome mismatch {rid}")
            feat_rows.append({"year":year,"race_id":rid,"race_date":date,
                              "anchor_finish":finish,"anchor_win":int(finish==1),
                              **race_features(group,axis)})
            ranked={}
            for source in ("MARKET","AI_PROB"):
                g=(group.sort_values(["market_rank","final_win_odds","_horse_id"],
                       ascending=[True,True,True]) if source=="MARKET"
                   else group.sort_values(["P__ANCHOR_FULL","market_rank","_horse_id"],
                       ascending=[False,True,True]))
                ranked[source]=[nos[x] for x in g["_horse_id"].astype(str) if x!=axis]
            paid_no=nos[axis]
            generated={}
            for source,k in {("MARKET",2),("MARKET",3),("MARKET",7),
                             ("AI_PROB",2),("AI_PROB",7)}:
                generated[(source,k)]=pair_templates(paid_no,ranked[source][:k])
            for family,source,k in BET_RULES:
                if family in ("WIN","PLACE"):
                    tickets={(family,(paid_no,))}
                else:
                    tickets=generated[(source,k)][family]
                n=len(tickets)
                assert_ok(n>0,f"{rid} empty tickets {family},{source},{k}")
                reward=float(sum(available.get(t,0.) for t in tickets))
                out_rows.append({"year":year,"race_id":rid,"rule":ridrule(family,source,k),
                                 "tickets":n,"stake":100*n,"return":reward,
                                 "hit":int(reward>0),"skip_gain":100*n-reward})
    assert_ok(len(feat_rows)==RACES_YEAR and len(out_rows)==RACES_YEAR*len(BET_RULES),
              f"{year} incomplete settlement {len(feat_rows)} {len(out_rows)}")
    return pd.DataFrame(feat_rows),pd.DataFrame(out_rows)

def train_features(x,medians=None):
    a=x[list(FEATURES)].to_numpy(dtype=np.float32)
    a=np.where(np.isfinite(a),a,np.nan)
    if medians is None:
        medians=np.nanmedian(a,axis=0)
        medians=np.where(np.isfinite(medians),medians,0.)
    return np.where(np.isnan(a),medians,a).astype(np.float32),medians

def build_model(kind):
    common=dict(max_iter=85,max_leaf_nodes=11,max_depth=3,
                min_samples_leaf=110,learning_rate=0.045,
                l2_regularization=15.0,early_stopping=False,random_state=1945)
    if kind=="HIT_CLASSIFIER":return HistGradientBoostingClassifier(**common)
    if kind=="UTILITY_REGRESSOR":return HistGradientBoostingRegressor(
        loss="squared_error",**common)
    raise ValueError(kind)

def put_result(rows,y,rule,method,policy,budget,mask,stake,ret,oracle=False):
    assert_ok(len(mask)==len(stake)==len(ret),"bad policy input")
    keep=~mask
    original_cost=float(stake.sum())
    original_return=float(ret.sum())
    after_cost=float(stake[keep].sum())
    after_return=float(ret[keep].sum())
    saving=float((stake[mask]-ret[mask]).sum())
    records=int(len(mask))
    count=int(mask.sum())
    rows.append({
        "year":str(y),"rule":rule,"model":method,
        "policy":policy,"target_skip_pct":budget,
        "races":records,"skip_races":count,"skip_pct":round(100*count/records,5),
        "bet_races":records-count,
        "stake_before_yen":round(original_cost,2),
        "return_before_yen":round(original_return,2),
        "roi_before_pct":round(100*original_return/original_cost,5),
        "profit_before_yen":round(original_return-original_cost,2),
        "stake_after_yen":round(after_cost,2),
        "return_after_yen":round(after_return,2),
        "roi_after_pct":round(100*after_return/after_cost,5) if after_cost else None,
        "profit_after_yen":round(after_return-after_cost,2),
        "gain_from_skip_yen":round(saving,2),
        "forgone_payout_yen":round(float(ret[mask].sum()),2),
        "hit_races_lost":int(np.sum((ret>0)&mask)),
        "true_loss_races_skipped":int(np.sum((ret==0)&mask)),
        "captured_return_pct":round(100*after_return/original_return,5)
             if original_return else None,
        "oracle":bool(oracle),
    })

def select_test_masks(train_scores,test_scores,fraction):
    n=len(test_scores)
    num=int(round(n*fraction/100))
    # Score distributions only. No test-year labels enter any threshold.
    threshold=float(np.quantile(train_scores,1-fraction/100)) if fraction else math.inf
    deployed=np.asarray(test_scores>threshold,dtype=bool)
    # analysis of test scores only (requires full test-year score distribution).
    top=np.zeros(n,dtype=bool)
    if num:
        top[np.argsort(-test_scores,kind="stable")[:num]]=True
    return deployed,top,threshold

def eval_one_rule(rule,features,outcomes):
    def ensure(x):assert_ok(x,"")
    key=outcomes[outcomes.rule.eq(rule)].sort_values(["year","race_id"])
    feat=features.sort_values(["year","race_id"])
    full=feat.merge(key,on=["year","race_id"],validate="one_to_one")
    assert_ok(len(full)==TOTAL_RACES,f"merged races {rule}")
    expected=BASELINE_ROIS[rule]
    current=100*float(full["return"].sum())/float(full["stake"].sum())
    assert_ok(abs(current-expected)<0.00001,
              f"historical payback regressions changed for {rule}: {current} != {expected}")
    rows=[];score_rows=[]
    importance_rows=[]
    for yr in STUDY_YEARS:
        train=full[full.year.lt(yr)].copy().reset_index(drop=True)
        test=full[full.year.eq(yr)].copy().reset_index(drop=True)
        assert_ok(len(train)==(yr-2023)*RACES_YEAR and len(test)==RACES_YEAR,
                  f"split drift {yr}")
        xtr,med=train_features(train)
        xte,_=train_features(test,med)
        stake=test["stake"].to_numpy(float);reward=test["return"].to_numpy(float)
        true_gain=stake-reward
        train_roi=np.divide(train["return"].to_numpy(float),
                            train["stake"].to_numpy(float))
        yhit=train["hit"].to_numpy(int)
        assert_ok(len(np.unique(yhit))==2,f"train labels only one class {rule}")
        methods={}
        for kind in ("HIT_CLASSIFIER","UTILITY_REGRESSOR"):
            model=build_model(kind)
            label=(yhit if kind=="HIT_CLASSIFIER" else np.clip(train_roi,0,25))
            model.fit(xtr,label)
            if kind=="HIT_CLASSIFIER":
                # High score => likely miss.
                trscore=1-model.predict_proba(xtr)[:,1]
                tescore=1-model.predict_proba(xte)[:,1]
            else:
                # High score => low clipped expected payout per staked yen.
                trscore=-model.predict(xtr)
                tescore=-model.predict(xte)
            methods[kind]=(trscore,tescore)
        methods["ODDS_ONLY"]=(-train["selected_odds"].to_numpy(float),
                              -test["selected_odds"].to_numpy(float))
        methods["MARKET_RANK_ONLY"]=(train["selected_market_rank"].to_numpy(float),
                                    test["selected_market_rank"].to_numpy(float))
        methods["RISK_ONLY"]=(train["selected_risk"].to_numpy(float),
                              test["selected_risk"].to_numpy(float))
        for kind,(str_,ste_) in methods.items():
            for budget in SKIP_PCTS:
                if budget==0 and kind not in ("HIT_CLASSIFIER",):
                    continue
                deployed,analytic,threshold=select_test_masks(str_,ste_,budget)
                put_result(rows,yr,rule,kind,"PRIOR_ONLY_CUTOFF",
                           budget,deployed,stake,reward)
                if budget:
                    put_result(rows,yr,rule,kind,"RETROSPECTIVE_SCORE_QUOTA",
                               budget,analytic,stake,reward)
            score_rows.append(pd.DataFrame({
                "year":yr,"race_id":test.race_id,"rule":rule,"model":kind,
                "skip_score":ste_,"stake":stake,"return":reward,
                "realized_skip_gain":true_gain,
            }))
    return rows,pd.concat(score_rows,ignore_index=True)

def describe_univariate(features,payoffs):
    sample=features.merge(payoffs[payoffs.rule.eq("WIN|SINGLE|0")],
                          on=["year","race_id"],validate="one_to_one")
    dims=["selected_market_rank","selected_odds","selected_edge",
          "selected_risk","top3_overlap","rank_gap","field_size"]
    out=[]
    for d in dims:
        x=sample.copy()
        # Train-era interpretations only, descriptive; no thresholds promoted.
        x["bin"]=pd.qcut(x[d].rank(method="first"),q=5,labels=False)
        for y,name in [("2023","2023_TRAIN"),("2024","2024_TEST"),
                       ("2025","2025_TEST")]:
            v=x[x.year.eq(int(y))]
            for b,g in v.groupby("bin"):
                s=g.stake.sum();r=g["return"].sum()
                out.append({"year":y,"dimension":d,"bin":int(b),
                            "n":len(g),"median_feature":float(g[d].median()),
                            "hit_rate_pct":round(100*g.hit.mean(),5),
                            "win_roi_pct":round(100*r/s,5) if s else None,
                            "net_skip_gain_yen":round(s-r,2)})
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--picks",required=True)
    ap.add_argument("--backfill-root",required=True)
    ap.add_argument("--out-dir",required=True)
    args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    frame,picks=get_inputs(args.predictions,args.picks)
    z=pd.read_csv(args.predictions,compression="infer",usecols=[
        "year","_race_id","_horse_id","_race_date",
        "market_prob_norm","P__MARKET_DIRECT","distill_ai_rank",
        "risk","top3_overlap"],
        dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    z["_race_date"]=z["_race_date"].astype(str).str[:10]
    frame=frame.merge(z,on=["year","_race_id","_horse_id"],validate="one_to_one")
    assert_ok(frame["_race_id"].nunique()==TOTAL_RACES,"race count drift")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        batches=list(pool.map(
            lambda y:collect_year(y,frame,picks,Path(args.backfill_root)),YEARS))
    feat=pd.concat([x[0] for x in batches],ignore_index=True)
    plys=pd.concat([x[1] for x in batches],ignore_index=True)
    assert_ok(len(feat)==TOTAL_RACES and feat.race_id.is_unique,"feature dedup")
    assert_ok(len(plys)==TOTAL_RACES*len(BET_RULES),"all bet template scores incomplete")
    assert_ok((feat["anchor_win"].sum()==1917) and
              (feat["anchor_finish"].le(3).sum()==4512),"selection not frozen")
    assert_ok(feat.groupby("year").size().to_dict()=={y:RACES_YEAR for y in YEARS},
              "year count drift")
    # Parallelize independent strategy research after a single receipt pass.
    rules=list(BASELINE_ROIS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(3,os.cpu_count() or 2)) as pool:
        fits=list(pool.map(lambda key:eval_one_rule(key,feat,plys),rules))
    result=pd.DataFrame([r for row,_ in fits for r in row])
    scored=pd.concat([x for _,x in fits],ignore_index=True)
    assert_ok(len(result)>700,"incomplete models")
    # Outcome-independent trained cutoffs are the primary evaluation.
    oos=result[(result.policy.eq("PRIOR_ONLY_CUTOFF")) & (result.target_skip_pct.gt(0))].copy()
    pooled=oos.groupby(["rule","model","policy","target_skip_pct"],as_index=False)[
      ["races","skip_races","bet_races","stake_before_yen","return_before_yen",
       "profit_before_yen","stake_after_yen","return_after_yen",
       "profit_after_yen","gain_from_skip_yen","forgone_payout_yen",
       "hit_races_lost","true_loss_races_skipped"]].sum()
    pooled["skip_pct"]=100*pooled["skip_races"]/pooled["races"]
    pooled["roi_after_pct"]=100*pooled["return_after_yen"]/pooled["stake_after_yen"]
    pooled["captured_return_pct"]=100*pooled["return_after_yen"]/pooled["return_before_yen"]
    # Single simple policy chosen strictly using 2024 results, then observe 2025.
    pool24=oos[(oos.year.eq("2024")) & (oos.skip_pct.le(30.0)) &
               (oos.skip_pct.ge(1.0)) & (oos.bet_races.ge(2400))].copy()
    # Criterion prioritizes achievable positive ROI, then total net profit,
    # with 70%-of-all-races minimum coverage.
    pool24=pool24.sort_values(["roi_after_pct","profit_after_yen","bet_races"],
                              ascending=[False,False,False])
    assert_ok(len(pool24)>0,"no 2024 selection candidates")
    selected=pool24.iloc[0]
    row25=oos[(oos.year.eq("2025")) &
             (oos.rule.eq(selected["rule"])) &
             (oos.model.eq(selected["model"])) &
             (oos.target_skip_pct.eq(selected["target_skip_pct"]))]
    assert_ok(len(row25)==1,"locked candidate missing 2025")
    # Hindsight oracle skips highest *realized* recoverable loss first, per year,
    # clearly separated from train-only cutoff model.
    oracle=[]
    for rule in rules:
        d=plys[plys.rule.eq(rule)]
        for yr in STUDY_YEARS:
            g=d[d.year.eq(yr)].copy()
            stake=g.stake.to_numpy(float);ret=g["return"].to_numpy(float)
            gain=stake-ret
            order=np.argsort(-gain,kind="stable")
            for p in SKIP_PCTS:
                n=int(round(len(g)*p/100))
                mask=np.zeros(len(g),dtype=bool)
                if n:mask[order[:n]]=True
                put_result(oracle,yr,rule,"RESULT_AWARE_ORACLE","POST_RESULT_QUOTA",
                           p,mask,stake,ret,oracle=True)
    oracle=pd.DataFrame(oracle)
    diagnostics=describe_univariate(feat,plys)
    feat.to_csv(out/"race-feature-snapshot.csv.gz",index=False,compression="gzip")
    plys.to_csv(out/"all-10-rules-settled-per-race.csv.gz",index=False,compression="gzip")
    result.to_csv(out/"all-model-year-cutoff-and-quota.csv",index=False)
    pooled.to_csv(out/"pooled-2024-2025-prior-only-cutoff.csv",index=False)
    scored.to_csv(out/"per-race-oos-skip-scores.csv.gz",index=False,compression="gzip")
    diagnostics.to_csv(out/"pre-race-feature-risk-profiles.csv",index=False)
    oracle.to_csv(out/"oracle-upper-bound-not-deployable.csv",index=False)
    main_columns=["rule","model","target_skip_pct","skip_pct",
                  "bet_races","roi_after_pct","profit_after_yen",
                  "gain_from_skip_yen","forgone_payout_yen"]
    top=pooled[pooled.bet_races.ge(4800)].sort_values(
         ["roi_after_pct","profit_after_yen"],ascending=[False,False]).head(18)
    compact24=selected[["rule","model","target_skip_pct",
                        "skip_pct","bet_races","roi_before_pct","roi_after_pct",
                        "profit_after_yen"]].to_dict()
    compact25=row25.iloc[0][["rule","model","target_skip_pct","skip_pct",
                             "bet_races","roi_before_pct","roi_after_pct",
                             "profit_after_yen"]].to_dict()
    summary={
        "contract":"L2_SKIP_VALUE_CHRONOLOGICAL_BATCH_V1",
        "evaluation": "2024 trained 2023; 2025 trained 2023+2024; no 2026",
        "baseline_stake":"100 yen per ticket, previous 82-variant study",
        "features":list(FEATURES),
        "signals":"market price, OOS ability rank/prob, disagreement, correction risk proxy",
        "outsider_vote_note":"Only the already available OOS risk proxy is used; no missing outsiders invented",
        "templates_evaluated":len(BET_RULES),
        "skip_budgets_pct":list(SKIP_PCTS),
        "main_policy":"PRIOR_ONLY_CUTOFF cutoff from train scores, applied unchanged on test year",
        "retrospective_score_quota":"NOT DEPLOYABLE yearly score ranks use whole test-year feature distribution, never labels",
        "oracle":"POST-RACE result-aware only, not trading policy",
        "primary_choice_selected_from":"2024 alone, 70% minimum race buy coverage",
        "selected_in_2024":compact24,"locked_2025_confirmation":compact25,
        "best_2024_2025_averaged_exploratory":top[main_columns].to_dict("records"),
        "risk_label_warning":"Regression training ROI clipped at 25; final evaluation uses uncapped real historical payouts",
        "lookahead_warning":"The frozen axis has been selected with final win odds, not timestamp-executable odds",
        "promotion":False,"paid_storage":False,"2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    print("L2_SKIP_VALUE_BATCH_V1_COMPLETE",flush=True)
    print("LOCKED_2024_SELECTED_2025_RESULT",flush=True)
    print(json.dumps({"selected_2024":compact24,"locked_2025":compact25},
                     ensure_ascii=False,default=str),flush=True)
    print("POOLED_PRIOR_ONLY_CUTOFF_TOP",flush=True)
    print(top[main_columns].to_csv(index=False),flush=True)

if __name__=="__main__":
    main()
