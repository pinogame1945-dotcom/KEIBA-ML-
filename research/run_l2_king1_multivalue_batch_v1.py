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
POOLS=(("TOP1",1),("TOP2",2),("TOP3",3),("ALL",10**9))

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

def choose_anchor_values(g):
    ordered=g.sort_values(
        ["consensus_rank","p3_calibrated","king_probability_mean","horse_number","horse_id"],
        ascending=[True,False,False,True,True]
    )
    anchor=ordered.iloc[0].to_dict()
    anchor_no=int(anchor["horse_number"])
    q=g[
        (g["horse_number"].astype(int)!=anchor_no) &
        (g["signed_rank_gap"]>=1) &
        (g["outsider_available"]>0)
    ].copy()
    if q.empty:
        return anchor,[]
    q=q.sort_values(
        ["p3_calibrated","signed_rank_gap","outsider_score","consensus_rank","market_rank","horse_id"],
        ascending=[False,False,False,True,False,True]
    )
    vals=q.to_dict("records")
    for i,r in enumerate(vals,1):
        r["value_order"]=i
    return anchor,vals

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
            anchor,values=choose_anchor_values(g)
            if not values:
                skipped["no_gap1_value"]+=1
                race_stats.append({
                    "year":year,"race_id":rid,"race_date":date,
                    "candidate_count":0,"anchor_top3":0,
                    **{f"ceiling_{name}":0 for name,_ in POOLS},
                    **{f"rescue_vs_top1_{name}":0 for name,_ in POOLS if name!="TOP1"},
                })
                continue

            pack=day.get(rid); orec=oddday.get(rid)
            if pack is None or orec is None:
                skipped["missing_pack_or_odds"]+=1; continue
            payouts,present=payout_map(pack)
            if "TRIO" not in present:
                skipped["no_trio_payout"]+=1; continue
            trio_odds={nums:price for bet,nums,price in iter_decoded_odds(orec) if bet=="TRIO"}
            if not trio_odds:
                skipped["no_trio_odds"]+=1; continue

            anchor_no=int(anchor["horse_number"])
            value_order={int(v["horse_number"]):int(v["value_order"]) for v in values}
            all_records=[r for r in g.to_dict("records") if int(r["horse_number"])!=anchor_no]
            field_size_norm=float(g["field_size"].iloc[0])/18.0
            local=[]

            for a,b in itertools.combinations(all_records,2):
                ano=int(a["horse_number"]); bno=int(b["horse_number"])
                orders=[value_order[x] for x in (ano,bno) if x in value_order]
                if not orders:
                    continue
                min_value_order=min(orders)
                if (float(a["consensus_rank"]),ano) > (float(b["consensus_rank"]),bno):
                    a,b=b,a; ano,bno=bno,ano
                nums=tuple(sorted((anchor_no,ano,bno)))
                odd=trio_odds.get(nums)
                if odd is None or not math.isfinite(float(odd)) or float(odd)<=0:
                    continue

                feat={}
                feat.update(hf(anchor,"anchor_"))
                feat.update(hf(a,"p1_"))
                feat.update(hf(b,"p2_"))
                feat.update({
                    "field_size_norm":field_size_norm,
                    "partners_p3_sum":float(a["p3_calibrated"]+b["p3_calibrated"]),
                    "partners_p3_min":float(min(a["p3_calibrated"],b["p3_calibrated"])),
                    "partners_outsider_sum":float(a["outsider_score_scaled"]+b["outsider_score_scaled"]),
                    "trio_p3_sum":float(anchor["p3_calibrated"]+a["p3_calibrated"]+b["p3_calibrated"]),
                    "trio_king_prob_sum":float(anchor["king_probability_mean"]+a["king_probability_mean"]+b["king_probability_mean"]),
                    "trio_gap_mean":float((anchor["signed_rank_gap_pct"]+a["signed_rank_gap_pct"]+b["signed_rank_gap_pct"])/3.0),
                    "trio_outsider_available":float(anchor["outsider_available"]+a["outsider_available"]+b["outsider_available"]),
                })
                hit=int(("TRIO",nums) in payouts)
                ret=float(payouts.get(("TRIO",nums),0.0))
                local.append({
                    "year":year,"race_id":rid,"race_date":date,
                    "anchor_no":anchor_no,"p1_no":ano,"p2_no":bno,
                    "min_value_order":int(min_value_order),
                    "candidate_count":len(values),
                    "ticket":"-".join(map(str,nums)),
                    "trio_odds":float(odd),
                    "hit":hit,"return_yen":ret,**feat
                })

            if not local:
                skipped["no_priced_pool_ticket"]+=1
                continue
            for rr in local:
                rr["race_weight"]=float(1.0/len(local))
            rows.extend(local)

            winning_trios=[nums for (bet,nums) in payouts if bet=="TRIO"]
            anchor_top3=int(any(anchor_no in nums for nums in winning_trios))
            stat={
                "year":year,"race_id":rid,"race_date":date,
                "candidate_count":len(values),
                "anchor_top3":anchor_top3,
                "anchor_market_rank":float(anchor["market_rank"]),
            }
            top1_ceiling=None
            for name,lim in POOLS:
                pool_nos={int(v["horse_number"]) for v in values if int(v["value_order"])<=lim}
                ceiling=int(any(anchor_no in nums and bool(pool_nos.intersection(nums)) for nums in winning_trios))
                stat[f"ceiling_{name}"]=ceiling
                if name=="TOP1":
                    top1_ceiling=ceiling
                else:
                    stat[f"rescue_vs_top1_{name}"]=int(ceiling and not top1_ceiling)
            race_stats.append(stat)

        if di%50==0:
            print(f"MULTIVALUE_BUILD_PROGRESS year={year} dates={di}/{len(bydate)} rows={len(rows)}",flush=True)

    if not rows:
        raise SystemExit(f"no rows year={year}")
    out=pd.DataFrame(rows); rs=pd.DataFrame(race_stats)
    print("MULTIVALUE_YEAR_READY "+json.dumps({
        "year":year,"source_races":source_races,"ticket_races":int(out["race_id"].nunique()),
        "rows":len(out),"skipped":dict(skipped)
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
    rng=np.random.default_rng(seed); n=len(race)
    stake_arr=race["stake"].to_numpy(); ret_arr=race["ret"].to_numpy(); vals=[]
    for _ in range(reps):
        idx=rng.integers(0,n,size=n)
        s=float(stake_arr[idx].sum()); r=float(ret_arr[idx].sum())
        vals.append(100*r/s if s else 0.0)
    return {"ci_low":float(np.quantile(vals,.025)),"ci_high":float(np.quantile(vals,.975))}

def summarize_roles(rs,source_races):
    out={
        "source_races":int(source_races),
        "rows_with_role_stats":int(len(rs)),
        "zero_candidate_races":int((rs["candidate_count"]==0).sum()),
        "candidate_count_1":int((rs["candidate_count"]==1).sum()),
        "candidate_count_2":int((rs["candidate_count"]==2).sum()),
        "candidate_count_3":int((rs["candidate_count"]==3).sum()),
        "candidate_count_4plus":int((rs["candidate_count"]>=4).sum()),
    }
    nz=rs[rs["candidate_count"]>0]
    out["mean_candidate_count_when_present"]=float(nz["candidate_count"].mean()) if len(nz) else None
    out["anchor_top3_pct_when_value_present"]=100*float(nz["anchor_top3"].mean()) if len(nz) else None
    for name,_ in POOLS:
        out[f"ceiling_{name}_count"]=int(nz[f"ceiling_{name}"].sum())
        out[f"ceiling_{name}_pct_all_source"]=100*float(nz[f"ceiling_{name}"].sum())/source_races
        out[f"ceiling_{name}_pct_value_present"]=100*float(nz[f"ceiling_{name}"].mean()) if len(nz) else None
        if name!="TOP1":
            out[f"rescue_vs_top1_{name}_count"]=int(nz[f"rescue_vs_top1_{name}"].sum())
            out[f"rescue_vs_top1_{name}_pct_all_source"]=100*float(nz[f"rescue_vs_top1_{name}"].sum())/source_races
    return out

def main():
    t0=time.time(); a=parse_args()
    c=json.load(open(a.contract,encoding="utf-8"))
    assert c["contract"]=="L2_KING1_MULTIVALUE_BATCH_V1"
    assert c["probability_model"]["probability_times_odds"] is False
    assert c["probability_model"]["odds_can_promote_ticket"] is False
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

    features=[f"{prefix}{col}" for prefix in ("anchor_","p1_","p2_") for col in HFEATS]
    features += [
        "field_size_norm","partners_p3_sum","partners_p3_min","partners_outsider_sum",
        "trio_p3_sum","trio_king_prob_sum","trio_gap_mean","trio_outsider_available",
    ]
    features=list(dict.fromkeys(features))
    forbidden=("trio_odds","log_trio_odds","trio_implied_prob","odds_rank","log_market_win_odds","min_value_order","candidate_count")
    if any(any(x in f for x in forbidden) for f in features):
        raise SystemExit(f"forbidden feature leaked: {features}")

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    fold_rows=[]; selected=defaultdict(list); role_years=[]

    for y in (2023,2024,2025):
        role_years.append({"year":y,**summarize_roles(race_meta[y],source_races[y])})

    for test_year,train_years in FOLDS:
        train=pd.concat([yearly[y] for y in train_years],ignore_index=True)
        races=train[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"]).reset_index(drop=True)
        cut=max(1,int(len(races)*0.80))
        fit_ids=set(races.iloc[:cut]["race_id"].astype(str))
        cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
        fit=train[train["race_id"].astype(str).isin(fit_ids)].copy()
        cal=train[train["race_id"].astype(str).isin(cal_ids)].copy()
        model,iso,spw=fit_model(fit,cal,features,cpu)
        testp=add_predictions(yearly[test_year],model,iso,features)
        total_races=source_races[test_year]

        for pool,lim in POOLS:
            pooldf=testp[testp["min_value_order"]<=lim].copy()
            ranked=freeze_rank(pooldf,max(TOPKS))
            for k in TOPKS:
                sel=ranked[ranked["prob_rank"]<=k].copy()
                selected[(pool,k)].append(sel.assign(test_year=test_year))
                m=metrics(sel,total_races); ci=bootstrap_roi(sel,seed=SEED+test_year+k+lim%1000)
                fold_rows.append({
                    "test_year":test_year,"train_years":"|".join(map(str,train_years)),
                    "fit_races":len(fit_ids),"calibration_races":len(cal_ids),
                    "scale_pos_weight":spw,"pool":pool,"topk":k,**m,
                    "roi_ci_low":ci["ci_low"],"roi_ci_high":ci["ci_high"],
                })
        print("MULTIVALUE_FOLD_DONE "+json.dumps({"test_year":test_year},separators=(",",":")),flush=True)

    total_source=sum(source_races[y] for y in (2023,2024,2025))
    pooled=[]
    for pool,_ in POOLS:
        for k in TOPKS:
            z=pd.concat(selected[(pool,k)],ignore_index=True)
            m=metrics(z,total_source); ci=bootstrap_roi(z,reps=1500,seed=SEED+1000+k+len(pool))
            pooled.append({"pool":pool,"topk":k,**m,"roi_ci_low":ci["ci_low"],"roi_ci_high":ci["ci_high"]})

    role_all=pd.concat([race_meta[y] for y in (2023,2024,2025)],ignore_index=True)
    role_pooled=summarize_roles(role_all,total_source)

    pd.DataFrame(fold_rows).to_csv(out/"fold-pool-topk-metrics.csv",index=False)
    pd.DataFrame(pooled).to_csv(out/"pooled-pool-topk-metrics.csv",index=False)
    pd.DataFrame(role_years).to_csv(out/"role-year-summary.csv",index=False)
    pd.DataFrame([role_pooled]).to_csv(out/"role-pooled-summary.csv",index=False)
    pd.DataFrame([
        {"year":y,"source_races":source_races[y],"ticket_races":int(yearly[y]["race_id"].nunique()),
         **{f"skip_{kk}":vv for kk,vv in skipped[y].items()}}
        for y in YEARS
    ]).to_csv(out/"coverage.csv",index=False)

    summary={
        "contract":"L2_KING1_MULTIVALUE_BATCH_V1_RESULT",
        "architecture":"KING1_FIXED_PLUS_VALUE_POOL_TOP1_TOP2_TOP3_ALL_PLUS_PROBABILITY_ONLY_TICKET_RANK",
        "probability_times_odds":False,
        "odds_used_for_ranking":False,
        "shared_probability_model_across_pools":True,
        "role_years":role_years,
        "role_pooled":role_pooled,
        "pooled_pool_topk":pooled,
        "2026_locked":True,
        "promotion":False,
        "elapsed_seconds":time.time()-t0,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 King1 Multi-Value Batch V1\n\n"
        "King1 stays fixed as anchor. GAP1 value horses are retained as TOP1, TOP2, TOP3, or ALL pools. "
        "A ticket is eligible when it contains King1 and at least one value horse from the chosen pool. "
        "The same walk-forward probability-only model ranks tickets for all pools. Odds never promote tickets. "
        "The run also reports candidate-count distribution, oracle role-pair ceilings, and rescues versus TOP1-only.\n",
        encoding="utf-8"
    )
    print("===== ROLE POOLED ====="); print(json.dumps(role_pooled,ensure_ascii=False,indent=2))
    print("===== POOLED POOL x TOPK ====="); print(pd.DataFrame(pooled).to_string(index=False))
    print("L2_KING1_MULTIVALUE_BATCH_V1_READY")

if __name__=="__main__":
    main()
