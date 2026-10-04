#!/usr/bin/env python3
import argparse,itertools,json,math,time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from run_l2_trio_probability_v2 import load_horse_dataset,attach_win_market,attach_outsider,prepare_l175
from build_l2_bet_kings_dataset_v1 import load_day,load_odds_day,iter_decoded_odds,payout_map

YEARS=(2022,2023,2024,2025)
FOLDS=((2023,(2022,)),(2024,(2022,2023)),(2025,(2022,2023,2024)))
SEED=20261004
STAKE=100.0
TOPKS=(1,3,5,8,12)
KING_MAXS=(1,2,3)
MARKET_MINS=(2,3,4)
MARKET_MAXS=(5,6,7)
ODDS_FLOORS=(0.0,10.0,20.0,30.0,40.0,50.0)

HFEATS=(
    "king_rank_pct","market_rank_pct","signed_rank_gap_pct",
    "king_top3_support_share","king_probability_mean",
    "p3_calibrated","p3_delta","outsider_score_scaled",
    "outsider_available",
)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--outsider-predictions",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def axis_for_race(g):
    q=g[(g["signed_rank_gap"]>=1)&(g["outsider_available"]>0)].copy()
    if q.empty:return None
    return q.sort_values(
        ["p3_calibrated","signed_rank_gap","outsider_score","consensus_rank","market_rank","horse_id"],
        ascending=[False,False,False,True,False,True]
    ).iloc[0]

def hf(row,prefix):
    return {prefix+c:float(row[c]) for c in HFEATS}

def odds_band(x):
    x=float(x)
    if x<10:return "LT10"
    if x<20:return "10_20"
    if x<50:return "20_50"
    if x<100:return "50_100"
    if x<200:return "100_200"
    if x<500:return "200_500"
    return "500_PLUS"

def build_year(year,df,root):
    root=Path(root)
    ydf=df[df["year"]==year].copy()
    groups={str(rid):g.copy() for rid,g in ydf.groupby("race_id",sort=False)}
    bydate=defaultdict(list)
    for rid,date in ydf[["race_id","race_date"]].drop_duplicates().itertuples(index=False):
        bydate[str(date)[:10]].append(str(rid))
    rows=[]; race_stats=[]; skipped=defaultdict(int)
    for di,(date,rids) in enumerate(sorted(bydate.items()),1):
        wanted=set(rids)
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        oddday=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in rids:
            g=groups[rid].sort_values(["consensus_rank","horse_number","horse_id"]).reset_index(drop=True)
            axis=axis_for_race(g)
            if axis is None:
                skipped["no_value_axis"]+=1; continue
            pack=day.get(rid); orec=oddday.get(rid)
            if pack is None or orec is None:
                skipped["missing_pack_or_odds"]+=1; continue
            payouts,present=payout_map(pack)
            if "TRIO" not in present:
                skipped["no_trio_payout"]+=1; continue
            trio_odds={nums:price for bet,nums,price in iter_decoded_odds(orec) if bet=="TRIO"}
            if not trio_odds:
                skipped["no_trio_odds"]+=1; continue
            axisd=axis.to_dict()
            axis_no=int(axisd["horse_number"])
            field_size_norm=float(g["field_size"].iloc[0])/18.0
            horses=[r for r in g.to_dict("records") if int(r["horse_number"])!=axis_no]
            local=[]
            for a,b in itertools.combinations(horses,2):
                if (float(a["consensus_rank"]),int(a["horse_number"])) > (float(b["consensus_rank"]),int(b["horse_number"])):
                    a,b=b,a
                nums=tuple(sorted((axis_no,int(a["horse_number"]),int(b["horse_number"]))))
                odd=trio_odds.get(nums)
                if odd is None or not math.isfinite(float(odd)) or float(odd)<=0:
                    continue
                feat={}
                feat.update(hf(axisd,"axis_")); feat.update(hf(a,"p1_")); feat.update(hf(b,"p2_"))
                feat.update({
                    "field_size_norm":field_size_norm,
                    "partners_p3_sum":float(a["p3_calibrated"]+b["p3_calibrated"]),
                    "partners_p3_min":float(min(a["p3_calibrated"],b["p3_calibrated"])),
                    "partners_score_sum":float(a["outsider_score_scaled"]+b["outsider_score_scaled"]),
                    "trio_p3_sum":float(axisd["p3_calibrated"]+a["p3_calibrated"]+b["p3_calibrated"]),
                    "trio_king_prob_sum":float(axisd["king_probability_mean"]+a["king_probability_mean"]+b["king_probability_mean"]),
                    "trio_gap_mean":float((axisd["signed_rank_gap_pct"]+a["signed_rank_gap_pct"]+b["signed_rank_gap_pct"])/3.0),
                    "trio_outsider_available":float(axisd["outsider_available"]+a["outsider_available"]+b["outsider_available"]),
                })
                hit=int(("TRIO",nums) in payouts)
                ret=float(payouts.get(("TRIO",nums),0.0))
                local.append({
                    "year":year,"race_id":rid,"race_date":date,
                    "axis_horse_id":str(axisd["horse_id"]),"axis_no":axis_no,
                    "axis_consensus_rank_raw":float(axisd["consensus_rank"]),
                    "axis_market_rank_raw":float(axisd["market_rank"]),
                    "axis_rank_gap_raw":float(axisd["signed_rank_gap"]),
                    "p1_no":int(a["horse_number"]),"p2_no":int(b["horse_number"]),
                    "ticket":"-".join(map(str,nums)),
                    "trio_odds":float(odd),"odds_band":odds_band(odd),
                    "hit":hit,"return_yen":ret,**feat
                })
            if not local:
                skipped["no_priced_axis_ticket"]+=1; continue
            for rr in local:
                rr["race_weight"]=float(1.0/len(local))
            rows.extend(local)
            race_stats.append({
                "year":year,"race_id":rid,"race_date":date,"axis_no":axis_no,
                "axis_king_rank":float(axis["consensus_rank"]),
                "axis_market_rank":float(axis["market_rank"]),
                "axis_rank_gap":float(axis["signed_rank_gap"]),
                "axis_p3":float(axis["p3_calibrated"]),
                "candidate_tickets":len(local),"axis_top3":int(any(x["hit"] for x in local))
            })
        if di%50==0:
            print(f"PROBONLY_BUILD_PROGRESS year={year} dates={di}/{len(bydate)} rows={len(rows)}",flush=True)
    if not rows: raise SystemExit(f"no ticket rows year={year}")
    out=pd.DataFrame(rows); rs=pd.DataFrame(race_stats)
    print("PROBONLY_YEAR_READY "+json.dumps({
        "year":year,"races":int(rs["race_id"].nunique()),"rows":len(out),
        "axis_top3_pct":100*float(rs["axis_top3"].mean()),"skipped":dict(skipped)
    },separators=(",",":")),flush=True)
    return out,rs,dict(skipped)

def metrics(sel,total_races):
    if sel.empty:
        return {"tickets":0,"bought_races":0,"coverage_pct":0.0,"stake_yen":0.0,"return_yen":0.0,"profit_yen":0.0,"roi_pct":None,"race_hit_rate_pct":None,"ticket_hit_rate_pct":None,"avg_tickets_per_bought_race":None}
    br=int(sel["race_id"].nunique())
    stake=STAKE*len(sel); ret=float(sel["return_yen"].sum())
    hr=int(sel.loc[sel["hit"]==1,"race_id"].nunique())
    return {
        "tickets":int(len(sel)),"bought_races":br,
        "coverage_pct":100*br/total_races if total_races else 0.0,
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "race_hit_rate_pct":100*hr/br if br else None,
        "ticket_hit_rate_pct":100*float(sel["hit"].mean()),
        "avg_tickets_per_bought_race":len(sel)/br if br else None,
    }

def freeze_probability_rank(df,maxk=12):
    if df.empty:return df.copy()
    z=df.sort_values(["race_id","p_hat","ticket"],ascending=[True,False,True]).copy()
    z["prob_rank"]=z.groupby("race_id",sort=False).cumcount()+1
    return z[z["prob_rank"]<=int(maxk)].copy()

def select_topk(df,k):
    if "prob_rank" not in df.columns:
        df=freeze_probability_rank(df,max(TOPKS))
    return df[df["prob_rank"]<=int(k)].copy()


def apply_postrank_rule(df,k,king_max,market_min,market_max,odds_floor):
    if df.empty:return df.copy()
    if "prob_rank" not in df.columns:
        df=freeze_probability_rank(df,max(TOPKS))
    return df[
        (df["prob_rank"]<=int(k)) &
        (df["axis_consensus_rank_raw"]<=float(king_max)) &
        (df["axis_market_rank_raw"]>=float(market_min)) &
        (df["axis_market_rank_raw"]<=float(market_max)) &
        (df["trio_odds"]>=float(odds_floor))
    ].copy()

def choose_rule(policy_df,total_races):
    races=policy_df[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"]).reset_index(drop=True)
    cut=max(1,len(races)//2)
    halves=[
        policy_df[policy_df["race_id"].astype(str).isin(set(races.iloc[:cut]["race_id"].astype(str)))],
        policy_df[policy_df["race_id"].astype(str).isin(set(races.iloc[cut:]["race_id"].astype(str)))],
    ]
    grid=[]
    for k in TOPKS:
      for km in KING_MAXS:
       for mn in MARKET_MINS:
        for mx in MARKET_MAXS:
         if mn>mx: continue
         for floor in ODDS_FLOORS:
          ms=[]
          for h in halves:
            sel=apply_postrank_rule(h,k,km,mn,mx,floor)
            ms.append(metrics(sel,int(h["race_id"].nunique())))
          pooled=metrics(apply_postrank_rule(policy_df,k,km,mn,mx,floor),total_races)
          eligible=all((m["coverage_pct"] or 0.0)>=60.0 and m["tickets"]>=100 for m in ms)
          rois=[m["roi_pct"] if m["roi_pct"] is not None else -1e9 for m in ms]
          grid.append({
              "topk":k,"king_max":km,"market_min":mn,"market_max":mx,
              "odds_floor":floor,"eligible":int(eligible),
              "roi_half1":rois[0],"roi_half2":rois[1],
              "robust_roi":min(rois),"pooled_roi":pooled["roi_pct"],
              "pooled_coverage_pct":pooled["coverage_pct"],
              "pooled_tickets":pooled["tickets"],
          })
    gd=pd.DataFrame(grid)
    q=gd[gd["eligible"]==1].copy()
    if q.empty:
        q=gd.sort_values(["pooled_coverage_pct","pooled_tickets"],ascending=False).head(25)
    best=q.sort_values(["robust_roi","pooled_roi","pooled_coverage_pct","topk"],ascending=[False,False,False,True]).iloc[0]
    rule={c:(int(best[c]) if c in ("topk","king_max","market_min","market_max") else float(best[c]))
          for c in ("topk","king_max","market_min","market_max","odds_floor")}
    return rule,gd

def fit_model(fit,cal,features,cpu):
    pos=max(1,int(fit["hit"].sum())); neg=max(1,len(fit)-pos)
    spw=float(min(50.0,max(1.0,math.sqrt(neg/pos))))
    model=lgb.LGBMClassifier(
        objective="binary",n_estimators=220,learning_rate=0.04,num_leaves=31,
        min_child_samples=100,subsample=0.9,colsample_bytree=0.9,reg_lambda=2.0,
        scale_pos_weight=spw,random_state=SEED,n_jobs=max(1,cpu),verbosity=-1
    )
    model.fit(fit[features],fit["hit"].astype(int),sample_weight=fit["race_weight"])
    raw=model.predict_proba(cal[features])[:,1]
    iso=IsotonicRegression(increasing=True,out_of_bounds="clip",y_min=1e-8,y_max=1.0)
    iso.fit(raw,cal["hit"].astype(int).to_numpy(),sample_weight=cal["race_weight"].to_numpy(dtype=float))
    return model,iso,spw

def add_predictions(df,model,iso,features):
    z=df.copy()
    raw=model.predict_proba(z[features])[:,1]
    z["p_hat"]=iso.predict(raw)
    return z

def bootstrap_roi(sel,reps=1000,seed=SEED):
    if sel.empty:return {"ci_low":None,"ci_high":None}
    race=sel.groupby("race_id",as_index=False).agg(stake=("hit","size"),ret=("return_yen","sum"))
    race["stake"]=race["stake"]*STAKE
    rng=np.random.default_rng(seed); n=len(race); vals=[]
    for _ in range(reps):
        idx=rng.integers(0,n,size=n)
        s=float(race["stake"].to_numpy()[idx].sum()); r=float(race["ret"].to_numpy()[idx].sum())
        vals.append(100*r/s if s else 0.0)
    return {"ci_low":float(np.quantile(vals,.025)),"ci_high":float(np.quantile(vals,.975))}

def main():
    t0=time.time(); a=parse_args()
    c=json.load(open(a.contract,encoding="utf-8"))
    assert c["contract"]=="L2_SIMPLE_VALUE_TRIO_FILTER_V4"
    assert c["ticket_policy"]["probability_times_odds"] is False
    assert c["ticket_policy"]["odds_can_promote_ticket"] is False
    assert c["ticket_policy"]["low_odds_ticket_replacement"] is False
    assert c["cost_policy"]["github_standard_cpu_only"] is True and c["cost_policy"]["gpu"] is False

    _,df=load_horse_dataset(a.dataset_dir)
    df,skipped_market=attach_win_market(df,a.backfill_root)
    df=attach_outsider(df,a.outsider_predictions)
    df=prepare_l175(df)
    if 2026 in set(df["year"]): raise SystemExit("2026 sealed")
    cpu=max(1,__import__("os").cpu_count() or 1)

    yearly={}; race_meta={}; skipped={}
    for y in YEARS:
        yearly[y],race_meta[y],skipped[y]=build_year(y,df,a.backfill_root)

    features=[f"{prefix}{c}" for prefix in ("axis_","p1_","p2_") for c in HFEATS]
    features += [
        "field_size_norm","partners_p3_sum","partners_p3_min","partners_score_sum",
        "trio_p3_sum","trio_king_prob_sum","trio_gap_mean","trio_outsider_available",
    ]
    features=list(dict.fromkeys(features))
    forbidden=("trio_odds","log_trio_odds","trio_implied_prob","odds_rank","log_market_win_odds")
    if any(any(x in f for x in forbidden) for f in features):
        raise SystemExit(f"odds feature leaked into probability model: {features}")

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    fold_rows=[]; grid_rows=[]; pooled_sel=[]; baseline_rows=[]; selected_rules=[]
    last_model=None

    for test_year,train_years in FOLDS:
        train=pd.concat([yearly[y] for y in train_years],ignore_index=True)
        races=train[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"]).reset_index(drop=True)
        n=len(races); fit_cut=max(1,int(n*0.70)); cal_cut=max(fit_cut+1,int(n*0.85))
        fit_ids=set(races.iloc[:fit_cut]["race_id"].astype(str))
        cal_ids=set(races.iloc[fit_cut:cal_cut]["race_id"].astype(str))
        policy_ids=set(races.iloc[cal_cut:]["race_id"].astype(str))
        fit=train[train["race_id"].astype(str).isin(fit_ids)].copy()
        cal=train[train["race_id"].astype(str).isin(cal_ids)].copy()
        policy=train[train["race_id"].astype(str).isin(policy_ids)].copy()

        model,iso,spw=fit_model(fit,cal,features,cpu); last_model=model
        policyp=freeze_probability_rank(add_predictions(policy,model,iso,features),max(TOPKS))
        rule,grid=choose_rule(policyp,int(policyp["race_id"].nunique()))
        grid["test_year"]=test_year; grid_rows.append(grid)

        testp=freeze_probability_rank(add_predictions(yearly[test_year],model,iso,features),max(TOPKS))
        total_races=int(testp["race_id"].nunique())
        sel=apply_postrank_rule(testp,rule["topk"],rule["king_max"],rule["market_min"],rule["market_max"],rule["odds_floor"])
        pooled_sel.append(sel.assign(test_year=test_year))
        m=metrics(sel,total_races); ci=bootstrap_roi(sel,seed=SEED+test_year)
        base=metrics(select_topk(testp,rule["topk"]),total_races)
        baseline_rows.append({"test_year":test_year,"topk":rule["topk"],**base})
        selected_rules.append({"test_year":test_year,**rule})
        fold_rows.append({
            "test_year":test_year,"train_years":"|".join(map(str,train_years)),
            "fit_races":len(fit_ids),"calibration_races":len(cal_ids),"policy_races":len(policy_ids),
            "scale_pos_weight":spw,**rule,**m,
            "baseline_same_topk_roi_pct":base["roi_pct"],
            "baseline_same_topk_coverage_pct":base["coverage_pct"],
            "roi_ci_low":ci["ci_low"],"roi_ci_high":ci["ci_high"],
        })
        print("FILTER_V4_FOLD_DONE "+json.dumps(fold_rows[-1],separators=(",",":")),flush=True)

    deployed=pd.concat(pooled_sel,ignore_index=True) if pooled_sel else pd.DataFrame()
    total_test_races=sum(int(yearly[y]["race_id"].nunique()) for y in (2023,2024,2025))
    pooled=metrics(deployed,total_test_races)
    pci=bootstrap_roi(deployed,reps=2000,seed=SEED+999)
    pooled.update({"roi_ci_low":pci["ci_low"],"roi_ci_high":pci["ci_high"]})

    pd.DataFrame(fold_rows).to_csv(out/"fold-selected-metrics.csv",index=False)
    pd.DataFrame(selected_rules).to_csv(out/"selected-rules.csv",index=False)
    pd.DataFrame(baseline_rows).to_csv(out/"baseline-same-topk.csv",index=False)
    pd.concat(grid_rows,ignore_index=True).to_csv(out/"policy-grid.csv",index=False)
    pd.DataFrame([
        {"year":y,"ticket_rows":len(yearly[y]),"axis_races":int(yearly[y]["race_id"].nunique()),**{f"skip_{k}":v for k,v in skipped[y].items()}}
        for y in YEARS
    ]).to_csv(out/"coverage.csv",index=False)

    imp=[]
    for name,val in zip(features,last_model.feature_importances_):
        imp.append({"feature":name,"importance":int(val)})
    pd.DataFrame(sorted(imp,key=lambda r:-r["importance"])).to_csv(out/"feature-importance-last-fold.csv",index=False)

    summary={
        "contract":"L2_SIMPLE_VALUE_TRIO_FILTER_V4_RESULT",
        "architecture":"GAP1_AXIS_PLUS_PROBABILITY_ONLY_RANK_THEN_AXIS_ZONE_AND_ODDS_FLOOR",
        "market_filter_semantics":"DROP_ONLY_AFTER_PROBABILITY_TOPK_FREEZE; NEVER_PROMOTE_OR_REPLACE",
        "probability_times_odds":False,
        "selected_rules":selected_rules,
        "folds":fold_rows,
        "pooled_2023_2025":pooled,
        "coverage_floor_for_policy_selection_pct":60.0,
        "test_year_never_used_for_rule_selection":True,
        "payout_used_only_for_prior_rule_selection_and_test_evaluation":True,
        "2026_locked":True,"promotion":False,
        "elapsed_seconds":time.time()-t0,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Simple Value TRIO Filter V4\n\n"
        "GAP1 axis and probability-only ticket ranking stay intact. Probability TOP-K is frozen first. "
        "Only then may the rule drop races whose chosen axis is outside a transparent King-rank/market-rank zone "
        "and drop tickets below a minimum TRIO odds floor. Odds never promote or replace a ticket. "
        "Each test-year rule is selected strictly from prior chronological data using separate fit/calibration/policy slices.\n",
        encoding="utf-8"
    )
    print("===== SELECTED FOLDS ====="); print(pd.DataFrame(fold_rows).to_string(index=False))
    print("===== POOLED ====="); print(json.dumps(pooled,ensure_ascii=False,indent=2))
    print("L2_SIMPLE_VALUE_TRIO_FILTER_V4_READY")

if __name__=="__main__":
    main()
