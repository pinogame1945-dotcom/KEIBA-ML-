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

def select_topk(df,k):
    out=[]
    for _,g in df.groupby("race_id",sort=False):
        q=g.sort_values(["p_hat","ticket"],ascending=[False,True]).head(k).copy()
        q["prob_rank"]=np.arange(1,len(q)+1)
        out.append(q)
    return pd.concat(out,ignore_index=True) if out else df.iloc[:0].copy()

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

def summarize_true_ranks(testp,year):
    z=testp.sort_values(["race_id","p_hat","ticket"],ascending=[True,False,True]).copy()
    z["prob_rank"]=z.groupby("race_id",sort=False).cumcount()+1
    total_races=int(z["race_id"].nunique())
    hitrows=z[z["hit"]==1].copy()
    if hitrows.empty:
        return {"year":year,"total_races":total_races,"axis_top3_races":0}
    rr=hitrows.groupby("race_id",as_index=False).agg(
        best_true_rank=("prob_rank","min"),
        winning_tickets=("prob_rank","size"),
        candidate_tickets=("prob_rank","max"),
    )
    ranks=rr["best_true_rank"].astype(int)
    cuts=[1,3,5,8,12,20,30,50]
    out={
        "year":year,
        "total_races":total_races,
        "axis_top3_races":int(len(rr)),
        "axis_top3_pct":100*len(rr)/total_races,
        "mean_best_true_rank":float(ranks.mean()),
        "median_best_true_rank":float(ranks.median()),
        "p75_best_true_rank":float(ranks.quantile(.75)),
        "p90_best_true_rank":float(ranks.quantile(.90)),
        "p95_best_true_rank":float(ranks.quantile(.95)),
    }
    for k in cuts:
        out[f"capture_top{k}_count"]=int((ranks<=k).sum())
        out[f"capture_top{k}_pct_given_axis_top3"]=100*float((ranks<=k).mean())
        out[f"capture_top{k}_pct_all_races"]=100*float((ranks<=k).sum())/total_races
    bins=[0,1,3,5,8,12,20,50,10**9]
    labels=["1","2-3","4-5","6-8","9-12","13-20","21-50","51+"]
    rr["rank_bucket"]=pd.cut(rr["best_true_rank"],bins=bins,labels=labels,include_lowest=True)
    buckets=rr.groupby("rank_bucket",observed=False).size().to_dict()
    out["rank_buckets"]={str(k):int(v) for k,v in buckets.items()}
    return out

def main():
    t0=time.time(); a=parse_args()
    c=json.load(open(a.contract,encoding="utf-8"))
    assert c["contract"]=="L2_SIMPLE_VALUE_TRIO_PROBONLY_V3_GAP1"
    assert c["ticket_probability"]["trio_odds_feature"] is False
    assert c["ticket_probability"]["win_odds_feature"] is False
    assert c["ticket_probability"]["probability_times_odds"] is False
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
    rows=[]; rank_detail=[]

    for test_year,train_years in FOLDS:
        train=pd.concat([yearly[y] for y in train_years],ignore_index=True)
        races=train[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"]).reset_index(drop=True)
        cut=max(1,int(len(races)*0.80))
        fit_ids=set(races.iloc[:cut]["race_id"].astype(str)); cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
        fit=train[train["race_id"].astype(str).isin(fit_ids)].copy()
        cal=train[train["race_id"].astype(str).isin(cal_ids)].copy()
        model,iso,spw=fit_model(fit,cal,features,cpu)
        testp=add_predictions(yearly[test_year],model,iso,features)
        z=testp.sort_values(["race_id","p_hat","ticket"],ascending=[True,False,True]).copy()
        z["prob_rank"]=z.groupby("race_id",sort=False).cumcount()+1
        hit=z[z["hit"]==1].copy()
        if not hit.empty:
            best=hit.sort_values(["race_id","prob_rank"]).groupby("race_id",as_index=False).first()
            rank_detail.append(best[["year","race_id","race_date","axis_no","prob_rank","trio_odds","return_yen"]].rename(columns={"prob_rank":"best_true_rank"}))
        sm=summarize_true_ranks(testp,test_year)
        sm.update({"train_years":"|".join(map(str,train_years)),"fit_races":len(fit_ids),"calibration_races":len(cal_ids),"scale_pos_weight":spw})
        rows.append(sm)
        print("TRUE_RANK_YEAR "+json.dumps(sm,separators=(",",":")),flush=True)

    detail=pd.concat(rank_detail,ignore_index=True)
    ranks=detail["best_true_rank"].astype(int)
    total=sum(int(yearly[y]["race_id"].nunique()) for y in (2023,2024,2025))
    pooled={
        "total_races":total,
        "axis_top3_races":int(len(detail)),
        "axis_top3_pct":100*len(detail)/total,
        "mean_best_true_rank":float(ranks.mean()),
        "median_best_true_rank":float(ranks.median()),
        "p75_best_true_rank":float(ranks.quantile(.75)),
        "p90_best_true_rank":float(ranks.quantile(.90)),
        "p95_best_true_rank":float(ranks.quantile(.95)),
    }
    for k in (1,3,5,8,12,20,30,50):
        pooled[f"capture_top{k}_count"]=int((ranks<=k).sum())
        pooled[f"capture_top{k}_pct_given_axis_top3"]=100*float((ranks<=k).mean())
        pooled[f"capture_top{k}_pct_all_races"]=100*float((ranks<=k).sum())/total
    bins=[0,1,3,5,8,12,20,50,10**9]
    labels=["1","2-3","4-5","6-8","9-12","13-20","21-50","51+"]
    detail["rank_bucket"]=pd.cut(detail["best_true_rank"],bins=bins,labels=labels,include_lowest=True)
    pooled["rank_buckets"]={str(k):int(v) for k,v in detail.groupby("rank_bucket",observed=False).size().to_dict().items()}

    pd.DataFrame(rows).to_csv(out/"year-summary.csv",index=False)
    detail.to_csv(out/"axis-hit-true-ranks.csv.gz",index=False,compression="gzip")
    summary={
        "contract":"L2_TRIO_TRUE_RANK_AUDIT_V1",
        "base_contract":"L2_SIMPLE_VALUE_TRIO_PROBONLY_V3_GAP1",
        "question":"When the fixed GAP1 value axis actually finishes top3, where does the winning TRIO rank under the probability-only ticket model?",
        "years":rows,
        "pooled":pooled,
        "probability_times_odds":False,
        "odds_used_for_ranking":False,
        "dead_heat_policy":"use best-ranked winning TRIO when multiple payout combinations exist",
        "2026_locked":True,
        "elapsed_seconds":time.time()-t0,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 TRIO True Rank Audit V1\n\n"
        "Uses the exact GAP1 axis and probability-only model design from V3. "
        "For races where the axis is in the winning TRIO, records the best probability rank of the actual winning TRIO. "
        "No odds multiplication, no market promotion, no test-year rule selection.\n",
        encoding="utf-8"
    )
    print("===== POOLED TRUE RANK =====")
    print(json.dumps(pooled,ensure_ascii=False,indent=2))
    print("L2_TRIO_TRUE_RANK_AUDIT_V1_READY")

if __name__=="__main__":
    main()
