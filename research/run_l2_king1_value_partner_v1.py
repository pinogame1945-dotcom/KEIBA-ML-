#!/usr/bin/env python3
import argparse,json,math,time
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

def choose_anchor_value(g):
    ordered=g.sort_values(
        ["consensus_rank","p3_calibrated","king_probability_mean","horse_number","horse_id"],
        ascending=[True,False,False,True,True]
    )
    anchor=ordered.iloc[0]
    anchor_no=int(anchor["horse_number"])
    q=g[
        (g["horse_number"].astype(int)!=anchor_no) &
        (g["signed_rank_gap"]>=1) &
        (g["outsider_available"]>0)
    ].copy()
    if q.empty:return anchor,None
    value=q.sort_values(
        ["p3_calibrated","signed_rank_gap","outsider_score","consensus_rank","market_rank","horse_id"],
        ascending=[False,False,False,True,False,True]
    ).iloc[0]
    return anchor,value

def build_year(year,df,root):
    root=Path(root)
    ydf=df[df["year"]==year].copy()
    groups={str(rid):g.copy() for rid,g in ydf.groupby("race_id",sort=False)}
    bydate=defaultdict(list)
    for rid,date in ydf[["race_id","race_date"]].drop_duplicates().itertuples(index=False):
        bydate[str(date)[:10]].append(str(rid))

    rows=[]; race_stats=[]; skipped=defaultdict(int)
    source_races=len(groups)

    for di,(date,rids) in enumerate(sorted(bydate.items()),1):
        wanted=set(rids)
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        oddday=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)

        for rid in rids:
            g=groups[rid].sort_values(["consensus_rank","horse_number","horse_id"]).reset_index(drop=True)
            anchor,value=choose_anchor_value(g)
            if value is None:
                skipped["no_distinct_gap1_value"]+=1
                continue

            pack=day.get(rid); orec=oddday.get(rid)
            if pack is None or orec is None:
                skipped["missing_pack_or_odds"]+=1
                continue
            payouts,present=payout_map(pack)
            if "TRIO" not in present:
                skipped["no_trio_payout"]+=1
                continue
            trio_odds={nums:price for bet,nums,price in iter_decoded_odds(orec) if bet=="TRIO"}
            if not trio_odds:
                skipped["no_trio_odds"]+=1
                continue

            ad=anchor.to_dict(); vd=value.to_dict()
            anchor_no=int(ad["horse_number"]); value_no=int(vd["horse_number"])
            field_size_norm=float(g["field_size"].iloc[0])/18.0
            partners=[
                r for r in g.to_dict("records")
                if int(r["horse_number"]) not in (anchor_no,value_no)
            ]
            local=[]
            for p in partners:
                partner_no=int(p["horse_number"])
                nums=tuple(sorted((anchor_no,value_no,partner_no)))
                odd=trio_odds.get(nums)
                if odd is None or not math.isfinite(float(odd)) or float(odd)<=0:
                    continue
                feat={}
                feat.update(hf(ad,"anchor_"))
                feat.update(hf(vd,"value_"))
                feat.update(hf(p,"partner_"))
                feat.update({
                    "field_size_norm":field_size_norm,
                    "anchor_value_p3_sum":float(ad["p3_calibrated"]+vd["p3_calibrated"]),
                    "anchor_value_gap_sum":float(ad["signed_rank_gap_pct"]+vd["signed_rank_gap_pct"]),
                    "anchor_value_outsider_sum":float(ad["outsider_score_scaled"]+vd["outsider_score_scaled"]),
                    "trio_p3_sum":float(ad["p3_calibrated"]+vd["p3_calibrated"]+p["p3_calibrated"]),
                    "trio_king_prob_sum":float(ad["king_probability_mean"]+vd["king_probability_mean"]+p["king_probability_mean"]),
                    "trio_gap_mean":float((ad["signed_rank_gap_pct"]+vd["signed_rank_gap_pct"]+p["signed_rank_gap_pct"])/3.0),
                    "trio_outsider_available":float(ad["outsider_available"]+vd["outsider_available"]+p["outsider_available"]),
                })
                hit=int(("TRIO",nums) in payouts)
                ret=float(payouts.get(("TRIO",nums),0.0))
                local.append({
                    "year":year,"race_id":rid,"race_date":date,
                    "anchor_no":anchor_no,"value_no":value_no,"partner_no":partner_no,
                    "anchor_horse_id":str(ad["horse_id"]),"value_horse_id":str(vd["horse_id"]),
                    "anchor_market_rank":float(ad["market_rank"]),
                    "value_market_rank":float(vd["market_rank"]),
                    "value_king_rank":float(vd["consensus_rank"]),
                    "value_rank_gap":float(vd["signed_rank_gap"]),
                    "ticket":"-".join(map(str,nums)),
                    "trio_odds":float(odd),"odds_band":odds_band(odd),
                    "hit":hit,"return_yen":ret,**feat
                })

            if not local:
                skipped["no_priced_role_ticket"]+=1
                continue

            for rr in local:
                rr["race_weight"]=float(1.0/len(local))
            rows.extend(local)

            winning_trios=[nums for (bet,nums) in payouts if bet=="TRIO"]
            anchor_top3=int(any(anchor_no in nums for nums in winning_trios))
            value_top3=int(any(value_no in nums for nums in winning_trios))
            both_top3=int(any(anchor_no in nums and value_no in nums for nums in winning_trios))
            race_stats.append({
                "year":year,"race_id":rid,"race_date":date,
                "anchor_no":anchor_no,"value_no":value_no,
                "anchor_top3":anchor_top3,"value_top3":value_top3,"both_top3":both_top3,
                "anchor_market_rank":float(ad["market_rank"]),
                "value_king_rank":float(vd["consensus_rank"]),
                "value_market_rank":float(vd["market_rank"]),
                "value_rank_gap":float(vd["signed_rank_gap"]),
                "candidate_tickets":len(local),
            })

        if di%50==0:
            print(f"ROLE_BUILD_PROGRESS year={year} dates={di}/{len(bydate)} rows={len(rows)}",flush=True)

    if not rows:
        raise SystemExit(f"no role ticket rows year={year}")
    out=pd.DataFrame(rows); rs=pd.DataFrame(race_stats)
    print("ROLE_YEAR_READY "+json.dumps({
        "year":year,"source_races":source_races,"eligible_races":int(rs["race_id"].nunique()),
        "coverage_pct":100*float(rs["race_id"].nunique())/source_races if source_races else 0.0,
        "rows":len(out),
        "anchor_top3_pct":100*float(rs["anchor_top3"].mean()),
        "value_top3_pct":100*float(rs["value_top3"].mean()),
        "both_top3_pct":100*float(rs["both_top3"].mean()),
        "skipped":dict(skipped)
    },separators=(",",":")),flush=True)
    return out,rs,dict(skipped),source_races

def metrics(sel,total_races):
    if sel.empty:
        return {
            "tickets":0,"bought_races":0,"coverage_pct":0.0,
            "stake_yen":0.0,"return_yen":0.0,"profit_yen":0.0,
            "roi_pct":None,"race_hit_rate_pct":None,"ticket_hit_rate_pct":None,
            "avg_tickets_per_bought_race":None
        }
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

def freeze_rank(df,maxk=12):
    if df.empty:return df.copy()
    z=df.sort_values(["race_id","p_hat","ticket"],ascending=[True,False,True]).copy()
    z["prob_rank"]=z.groupby("race_id",sort=False).cumcount()+1
    return z[z["prob_rank"]<=int(maxk)].copy()

def fit_model(fit,cal,features,cpu):
    pos=max(1,int(fit["hit"].sum())); neg=max(1,len(fit)-pos)
    spw=float(min(50.0,max(1.0,math.sqrt(neg/pos))))
    model=lgb.LGBMClassifier(
        objective="binary",n_estimators=180,learning_rate=0.05,num_leaves=31,
        min_child_samples=80,subsample=0.9,colsample_bytree=0.9,reg_lambda=2.0,
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

def bootstrap_roi(sel,reps=1200,seed=SEED):
    if sel.empty:return {"ci_low":None,"ci_high":None}
    race=sel.groupby("race_id",as_index=False).agg(stake=("hit","size"),ret=("return_yen","sum"))
    race["stake"]=race["stake"]*STAKE
    rng=np.random.default_rng(seed); n=len(race); vals=[]
    stake_arr=race["stake"].to_numpy(); ret_arr=race["ret"].to_numpy()
    for _ in range(reps):
        idx=rng.integers(0,n,size=n)
        s=float(stake_arr[idx].sum()); r=float(ret_arr[idx].sum())
        vals.append(100*r/s if s else 0.0)
    return {"ci_low":float(np.quantile(vals,.025)),"ci_high":float(np.quantile(vals,.975))}

def main():
    t0=time.time(); a=parse_args()
    c=json.load(open(a.contract,encoding="utf-8"))
    assert c["contract"]=="L2_KING1_VALUE_PARTNER_V1"
    assert c["ticket_probability"]["trio_odds_feature"] is False
    assert c["ticket_probability"]["win_odds_feature"] is False
    assert c["ticket_probability"]["probability_times_odds"] is False
    assert c["ticket_probability"]["market_can_promote_ticket"] is False
    assert c["cost_policy"]["github_standard_cpu_only"] is True and c["cost_policy"]["gpu"] is False

    _,df=load_horse_dataset(a.dataset_dir)
    df,skipped_market=attach_win_market(df,a.backfill_root)
    df=attach_outsider(df,a.outsider_predictions)
    df=prepare_l175(df)
    if 2026 in set(df["year"]): raise SystemExit("2026 sealed")
    cpu=max(1,__import__("os").cpu_count() or 1)

    yearly={}; race_meta={}; skipped={}; source_races={}
    for y in YEARS:
        yearly[y],race_meta[y],skipped[y],source_races[y]=build_year(y,df,a.backfill_root)

    features=[f"{prefix}{col}" for prefix in ("anchor_","value_","partner_") for col in HFEATS]
    features += [
        "field_size_norm","anchor_value_p3_sum","anchor_value_gap_sum","anchor_value_outsider_sum",
        "trio_p3_sum","trio_king_prob_sum","trio_gap_mean","trio_outsider_available",
    ]
    features=list(dict.fromkeys(features))
    forbidden=("trio_odds","log_trio_odds","trio_implied_prob","odds_rank","log_market_win_odds")
    if any(any(x in f for x in forbidden) for f in features):
        raise SystemExit(f"odds feature leaked into probability model: {features}")

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    fold_rows=[]; selected_by_k=defaultdict(list); role_years=[]

    for y in (2023,2024,2025):
        rs=race_meta[y]
        role_years.append({
            "year":y,
            "source_races":source_races[y],
            "eligible_races":int(len(rs)),
            "eligible_coverage_pct":100*len(rs)/source_races[y],
            "anchor_top3_pct":100*float(rs["anchor_top3"].mean()),
            "value_top3_pct":100*float(rs["value_top3"].mean()),
            "both_top3_pct":100*float(rs["both_top3"].mean()),
            "mean_anchor_market_rank":float(rs["anchor_market_rank"].mean()),
            "mean_value_king_rank":float(rs["value_king_rank"].mean()),
            "mean_value_market_rank":float(rs["value_market_rank"].mean()),
            "mean_value_rank_gap":float(rs["value_rank_gap"].mean()),
        })

    for test_year,train_years in FOLDS:
        train=pd.concat([yearly[y] for y in train_years],ignore_index=True)
        races=train[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"]).reset_index(drop=True)
        cut=max(1,int(len(races)*0.80))
        fit_ids=set(races.iloc[:cut]["race_id"].astype(str))
        cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
        fit=train[train["race_id"].astype(str).isin(fit_ids)].copy()
        cal=train[train["race_id"].astype(str).isin(cal_ids)].copy()

        model,iso,spw=fit_model(fit,cal,features,cpu)
        testp=freeze_rank(add_predictions(yearly[test_year],model,iso,features),max(TOPKS))
        total_races=source_races[test_year]

        for k in TOPKS:
            sel=testp[testp["prob_rank"]<=k].copy()
            selected_by_k[k].append(sel.assign(test_year=test_year))
            m=metrics(sel,total_races); ci=bootstrap_roi(sel,seed=SEED+test_year+k)
            fold_rows.append({
                "test_year":test_year,"train_years":"|".join(map(str,train_years)),
                "fit_races":len(fit_ids),"calibration_races":len(cal_ids),
                "scale_pos_weight":spw,"topk":k,**m,
                "roi_ci_low":ci["ci_low"],"roi_ci_high":ci["ci_high"],
            })
        print("ROLE_FOLD_DONE "+json.dumps({"test_year":test_year,"eligible":int(testp["race_id"].nunique())},separators=(",",":")),flush=True)

    total_source=sum(source_races[y] for y in (2023,2024,2025))
    pooled=[]
    for k in TOPKS:
        z=pd.concat(selected_by_k[k],ignore_index=True)
        m=metrics(z,total_source); ci=bootstrap_roi(z,reps=1800,seed=SEED+1000+k)
        pooled.append({"topk":k,**m,"roi_ci_low":ci["ci_low"],"roi_ci_high":ci["ci_high"]})

    role_all=pd.concat([race_meta[y] for y in (2023,2024,2025)],ignore_index=True)
    role_pooled={
        "source_races":total_source,
        "eligible_races":int(len(role_all)),
        "eligible_coverage_pct":100*len(role_all)/total_source,
        "anchor_top3_pct":100*float(role_all["anchor_top3"].mean()),
        "value_top3_pct":100*float(role_all["value_top3"].mean()),
        "both_top3_pct":100*float(role_all["both_top3"].mean()),
        "mean_anchor_market_rank":float(role_all["anchor_market_rank"].mean()),
        "mean_value_king_rank":float(role_all["value_king_rank"].mean()),
        "mean_value_market_rank":float(role_all["value_market_rank"].mean()),
        "mean_value_rank_gap":float(role_all["value_rank_gap"].mean()),
    }

    pd.DataFrame(fold_rows).to_csv(out/"fold-topk-metrics.csv",index=False)
    pd.DataFrame(pooled).to_csv(out/"pooled-topk-metrics.csv",index=False)
    pd.DataFrame(role_years).to_csv(out/"role-year-metrics.csv",index=False)
    pd.DataFrame([
        {"year":y,"source_races":source_races[y],"eligible_races":int(len(race_meta[y])),
         **{f"skip_{kk}":vv for kk,vv in skipped[y].items()}}
        for y in YEARS
    ]).to_csv(out/"coverage.csv",index=False)

    summary={
        "contract":"L2_KING1_VALUE_PARTNER_V1_RESULT",
        "architecture":"KING1_ANCHOR_PLUS_DISTINCT_GAP1_VALUE_PLUS_PROBABILITY_RANKED_THIRD",
        "ranking":"third-horse tickets ranked by p_hat only",
        "probability_times_odds":False,
        "odds_used_for_ranking":False,
        "role_years":role_years,
        "role_pooled":role_pooled,
        "pooled_topk":pooled,
        "stake":"flat 100 yen per selected ticket",
        "2026_locked":True,
        "promotion":False,
        "elapsed_seconds":time.time()-t0,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 King1 + Value + Partner V1\n\n"
        "Role separation test. Seven-King rank1 is the reliability anchor. A distinct GAP1 value horse is fixed as the value leg. "
        "Only the third horse is probability-ranked. No odds multiplication and no market-based ticket promotion.\n",
        encoding="utf-8"
    )
    print("===== ROLE POOLED ====="); print(json.dumps(role_pooled,ensure_ascii=False,indent=2))
    print("===== POOLED TOPK ====="); print(pd.DataFrame(pooled).to_string(index=False))
    print("L2_KING1_VALUE_PARTNER_V1_READY")

if __name__=="__main__":
    main()
