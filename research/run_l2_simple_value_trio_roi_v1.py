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
POLICY_THRESHOLDS=(-0.20,0.0,0.10,0.20,0.30,0.50,0.75,1.0)
POLICY_TOPK=(1,2,3,5,8,12)
EPS=1e-12

HFEATS=(
    "king_rank_pct","market_rank_pct","signed_rank_gap_pct",
    "king_top3_support_share","king_probability_mean",
    "p3_calibrated","p3_delta","outsider_score_scaled",
    "outsider_available","log_market_win_odds",
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
    q=g[(g["signed_rank_gap"]>=2)&(g["outsider_available"]>0)].copy()
    if q.empty:return None
    return q.sort_values(
        ["p3_calibrated","signed_rank_gap","outsider_score","consensus_rank","market_rank","horse_id"],
        ascending=[False,False,False,True,False,True]
    ).iloc[0]

def hf(row,prefix):
    return {prefix+c:float(row[c]) for c in HFEATS}

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
            axis_no=int(axis["horse_number"])
            others=g[g["horse_number"].astype(int)!=axis_no]
            local=[]
            for i,j in itertools.combinations(range(len(others)),2):
                a=others.iloc[i]; b=others.iloc[j]
                if (float(a["consensus_rank"]),int(a["horse_number"])) > (float(b["consensus_rank"]),int(b["horse_number"])):
                    a,b=b,a
                nums=tuple(sorted((axis_no,int(a["horse_number"]),int(b["horse_number"]))))
                odd=trio_odds.get(nums)
                if odd is None or not math.isfinite(float(odd)) or float(odd)<=0: continue
                feat={}
                feat.update(hf(axis,"axis_")); feat.update(hf(a,"p1_")); feat.update(hf(b,"p2_"))
                feat.update({
                    "field_size_norm":float(g["field_size"].iloc[0])/18.0,
                    "partners_p3_sum":float(a["p3_calibrated"]+b["p3_calibrated"]),
                    "partners_p3_min":float(min(a["p3_calibrated"],b["p3_calibrated"])),
                    "partners_score_sum":float(a["outsider_score_scaled"]+b["outsider_score_scaled"]),
                    "trio_p3_sum":float(axis["p3_calibrated"]+a["p3_calibrated"]+b["p3_calibrated"]),
                    "trio_king_prob_sum":float(axis["king_probability_mean"]+a["king_probability_mean"]+b["king_probability_mean"]),
                    "trio_gap_mean":float((axis["signed_rank_gap_pct"]+a["signed_rank_gap_pct"]+b["signed_rank_gap_pct"])/3.0),
                    "trio_outsider_available":float(axis["outsider_available"]+a["outsider_available"]+b["outsider_available"]),
                    "log_trio_odds":float(math.log(max(float(odd),1e-9))),
                    "trio_implied_prob":float(1.0/max(float(odd),1e-9)),
                })
                hit=int(("TRIO",nums) in payouts)
                ret=float(payouts.get(("TRIO",nums),0.0))
                local.append({
                    "year":year,"race_id":rid,"race_date":date,
                    "axis_horse_id":str(axis["horse_id"]),"axis_no":axis_no,
                    "p1_no":int(a["horse_number"]),"p2_no":int(b["horse_number"]),
                    "ticket":"-".join(map(str,nums)),"trio_odds":float(odd),
                    "hit":hit,"return_yen":ret,**feat
                })
            if not local:
                skipped["no_priced_axis_ticket"]+=1; continue
            odds=np.array([r["trio_odds"] for r in local],dtype=float)
            order=pd.Series(odds).rank(method="min",ascending=True).to_numpy(dtype=float)
            for rr,rank in zip(local,order):
                rr["odds_rank_pct"]=float(rank/len(local))
                rr["race_weight"]=float(1.0/len(local))
            rows.extend(local)
            race_stats.append({
                "year":year,"race_id":rid,"race_date":date,"axis_no":axis_no,
                "candidate_tickets":len(local),"axis_top3":int(any(x["hit"] for x in local))
            })
        if di%50==0:
            print(f"VALUE_TRIO_BUILD_PROGRESS year={year} dates={di}/{len(bydate)} rows={len(rows)}",flush=True)
    if not rows: raise SystemExit(f"no ticket rows year={year}")
    out=pd.DataFrame(rows)
    rs=pd.DataFrame(race_stats)
    print("VALUE_TRIO_YEAR_READY "+json.dumps({
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
        "tickets":int(len(sel)),"bought_races":br,"coverage_pct":100*br/total_races if total_races else 0.0,
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,"roi_pct":100*ret/stake if stake else None,
        "race_hit_rate_pct":100*hr/br if br else None,"ticket_hit_rate_pct":100*float(sel["hit"].mean()),
        "avg_tickets_per_bought_race":len(sel)/br if br else None,
    }

def select_policy_rows(df,threshold,topk):
    out=[]
    for _,g in df.groupby("race_id",sort=False):
        q=g[g["edge"]>=threshold].sort_values(["edge","p_hat","trio_odds"],ascending=[False,False,False]).head(topk)
        if len(q): out.append(q)
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
    z["edge"]=z["p_hat"]*z["trio_odds"]-1.0
    return z

def choose_policy(cal):
    races=cal[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"]).reset_index(drop=True)
    cut=max(1,len(races)//2)
    a_ids=set(races.iloc[:cut]["race_id"].astype(str)); b_ids=set(races.iloc[cut:]["race_id"].astype(str))
    halves=[cal[cal["race_id"].astype(str).isin(a_ids)],cal[cal["race_id"].astype(str).isin(b_ids)]]
    grid=[]
    for th in POLICY_THRESHOLDS:
        for k in POLICY_TOPK:
            ms=[]
            for h in halves:
                sel=select_policy_rows(h,th,k)
                ms.append(metrics(sel,int(h["race_id"].nunique())))
            pooled=metrics(select_policy_rows(cal,th,k),int(cal["race_id"].nunique()))
            eligible=all((m["coverage_pct"] or 0)>=40.0 and m["tickets"]>=100 for m in ms)
            rois=[m["roi_pct"] if m["roi_pct"] is not None else -1e9 for m in ms]
            grid.append({
                "edge_threshold":th,"topk":k,"eligible":int(eligible),
                "roi_half1":rois[0],"roi_half2":rois[1],"robust_roi":min(rois),
                "pooled_roi":pooled["roi_pct"],"pooled_coverage_pct":pooled["coverage_pct"],
                "pooled_tickets":pooled["tickets"],
            })
    gd=pd.DataFrame(grid)
    q=gd[gd["eligible"]==1].copy()
    if q.empty:
        q=gd.sort_values(["pooled_coverage_pct","pooled_tickets"],ascending=False).head(12)
    best=q.sort_values(["robust_roi","pooled_roi","pooled_coverage_pct","topk"],ascending=[False,False,False,True]).iloc[0]
    return float(best["edge_threshold"]),int(best["topk"]),gd

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
    assert c["contract"]=="L2_SIMPLE_VALUE_TRIO_ROI_V1"
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
        "log_trio_odds","trio_implied_prob","odds_rank_pct",
    ]
    features=list(dict.fromkeys(features))

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    fold_rows=[]; policy_rows=[]; fixed_rows=[]; pooled_sel=[]
    for test_year,train_years in FOLDS:
        train=pd.concat([yearly[y] for y in train_years],ignore_index=True)
        races=train[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"]).reset_index(drop=True)
        cut=max(1,int(len(races)*0.80))
        fit_ids=set(races.iloc[:cut]["race_id"].astype(str)); cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
        fit=train[train["race_id"].astype(str).isin(fit_ids)].copy()
        cal=train[train["race_id"].astype(str).isin(cal_ids)].copy()
        model,iso,spw=fit_model(fit,cal,features,cpu)
        calp=add_predictions(cal,model,iso,features)
        th,k,grid=choose_policy(calp)
        grid["test_year"]=test_year; policy_rows.append(grid)
        testp=add_predictions(yearly[test_year],model,iso,features)
        sel=select_policy_rows(testp,th,k)
        pooled_sel.append(sel.assign(test_year=test_year))
        m=metrics(sel,int(testp["race_id"].nunique())); ci=bootstrap_roi(sel,seed=SEED+test_year)
        fold_rows.append({
            "test_year":test_year,"train_years":"|".join(map(str,train_years)),
            "fit_races":len(fit_ids),"calibration_races":len(cal_ids),
            "scale_pos_weight":spw,"selected_edge_threshold":th,"selected_topk":k,
            **m,"roi_ci_low":ci["ci_low"],"roi_ci_high":ci["ci_high"],
        })
        for name,fth,fk in (
            ("TOP1_ANY",-1e9,1),("TOP3_ANY",-1e9,3),("TOP5_ANY",-1e9,5),
            ("TOP3_EDGE0",0.0,3),("TOP5_EDGE0",0.0,5),
        ):
            sm=metrics(select_policy_rows(testp,fth,fk),int(testp["race_id"].nunique()))
            fixed_rows.append({"test_year":test_year,"policy":name,**sm})
        print("VALUE_TRIO_FOLD_DONE "+json.dumps(fold_rows[-1],separators=(",",":")),flush=True)

    deployed=pd.concat(pooled_sel,ignore_index=True)
    pooled=metrics(deployed,int(sum(yearly[y]["race_id"].nunique() for y in (2023,2024,2025))))
    pci=bootstrap_roi(deployed,reps=2000,seed=SEED+999)
    pooled.update({"roi_ci_low":pci["ci_low"],"roi_ci_high":pci["ci_high"]})

    pd.DataFrame(fold_rows).to_csv(out/"fold-metrics.csv",index=False)
    pd.concat(policy_rows,ignore_index=True).to_csv(out/"policy-grid.csv",index=False)
    pd.DataFrame(fixed_rows).to_csv(out/"fixed-policy-metrics.csv",index=False)
    pd.DataFrame([
        {"year":y,"ticket_rows":len(yearly[y]),"axis_races":int(yearly[y]["race_id"].nunique()),**{f"skip_{k}":v for k,v in skipped[y].items()}}
        for y in YEARS
    ]).to_csv(out/"coverage.csv",index=False)

    imp=[]
    for name,val in zip(features,model.feature_importances_):
        imp.append({"feature":name,"importance":int(val)})
    pd.DataFrame(sorted(imp,key=lambda r:-r["importance"])).to_csv(out/"feature-importance-last-fold.csv",index=False)

    summary={
        "contract":"L2_SIMPLE_VALUE_TRIO_ROI_V1_RESULT",
        "architecture":"FROZEN_VALUE_AXIS_THEN_TICKET_MODEL_WITH_TRIO_ODDS",
        "axis_rule":"market_rank-consensus_rank>=2, highest existing OOS p3",
        "ticket_space":"all priced TRIO tickets containing axis",
        "ticket_model":"LightGBM binary ticket hit model; partner L1.75/p3/Outsider context plus final TRIO odds",
        "policy_selection":"prior chronological calibration only; maximize minimum ROI across two halves with >=40% coverage each",
        "stake":"flat 100 yen per selected ticket",
        "folds":fold_rows,"pooled_2023_2025":pooled,
        "horse_reranking":False,"structural_risk_used":False,
        "payout_used_for_prior_policy_selection":True,
        "test_year_payout_used_only_for_evaluation":True,
        "2026_locked":True,"promotion":False,
        "elapsed_seconds":time.time()-t0,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Simple Value TRIO ROI V1\n\n"
        "Frozen value-axis selection first. Every priced TRIO containing that axis becomes a ticket candidate. "
        "A ticket-level model sees partner L1.75/p3/Outsider context and the TRIO market price together. "
        "Policy is selected strictly from prior chronological calibration data and evaluated with flat 100-yen stakes.\n",
        encoding="utf-8"
    )
    print("===== FOLD METRICS ====="); print(pd.DataFrame(fold_rows).to_string(index=False))
    print("===== POOLED ====="); print(json.dumps(pooled,ensure_ascii=False,indent=2))
    print("L2_SIMPLE_VALUE_TRIO_ROI_V1_READY")

if __name__=="__main__":
    main()
