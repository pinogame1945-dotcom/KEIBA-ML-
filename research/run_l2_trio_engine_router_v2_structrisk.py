#!/usr/bin/env python3
import argparse,json,math,os,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import run_l2_trio_engine_router_v1 as v1
import run_l1_structural_risk_fullfield_v1 as sr
from run_l2_trio_probability_v2 import (
    YEARS,FOLDS,write_csv,load_horse_dataset,attach_win_market,attach_outsider,prepare_l175
)
from run_l2_trio_market_residual_v3 import build_year_matrix
from run_l2_trio_split_residual_v4 import (
    parse_year_paths,load_outsider_ballots,attach_outsider_ballots,
    build_compact_engine_matrices,load_engine_year,train_one_engine,race_ids_for_year
)
from run_l2_trio_v3_robustness_audit import bootstrap_mean_ci

SEED=20261004
BOOT_REPS=3000
BASE_ROUTER_FEATURES=list(v1.ROUTER_FEATURES)
STRUCT_ROUTER_FEATURES=[
    "sr_king1_delta","sr_king1_abs_delta","sr_king1_full_risk",
    "sr_market1_delta","sr_market1_abs_delta","sr_market1_full_risk",
    "sr_king_top3_delta_mean","sr_king_top3_delta_max","sr_king_top3_delta_min",
    "sr_king_top3_abs_mean","sr_king_top3_abs_max","sr_king_top3_high_pos_share","sr_king_top3_high_abs_share",
    "sr_market_top3_delta_mean","sr_market_top3_delta_max","sr_market_top3_delta_min",
    "sr_market_top3_abs_mean","sr_market_top3_abs_max","sr_market_top3_high_pos_share","sr_market_top3_high_abs_share",
    "sr_king_top6_abs_mean","sr_king_top6_abs_max","sr_king_top6_high_pos_share","sr_king_top6_high_abs_share",
    "sr_market_top6_abs_mean","sr_market_top6_abs_max","sr_market_top6_high_pos_share","sr_market_top6_high_abs_share",
    "sr_both_top3_abs_mean","sr_both_top3_abs_max","sr_both_top3_high_pos_share","sr_both_top3_high_abs_share",
    "sr_all_abs_mean","sr_all_abs_max",
]
RISK_ROUTER_FEATURES=BASE_ROUTER_FEATURES+STRUCT_ROUTER_FEATURES
STRUCT_BASELINE_COLS=[
    "field_size","market_rank_pct","consensus_rank_pct",
    "log_market_win_probability","log_final_win_odds",
    "rank_gap_pct","abs_rank_gap_pct",
]

def parse_args():
    p=argparse.ArgumentParser(description="Three-way TRIO router V2 with walk-forward Structural Risk context.")
    p.add_argument("--contract",required=True)
    p.add_argument("--l175-contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--outsider-predictions",required=True)
    p.add_argument("--ballots-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--snapshot-year",action="append",required=True,help="YEAR=PATH")
    p.add_argument("--matrix-cache",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_snapshot_paths(items):
    out={}
    for spec in items:
        y,p=spec.split("=",1)
        out[int(y)]=p
    if set(out)!=set(YEARS):
        raise SystemExit(f"snapshot years mismatch: {sorted(out)}")
    return out

def prepare_structural_frame(df,snaps):
    z=df.copy()
    z=z[z["target_top3"].notna()].copy()
    z["final_win_odds"]=pd.to_numeric(z["market_win_odds"],errors="coerce")
    z["field_size"]=pd.to_numeric(z["field_size"],errors="raise").astype(float)
    z["consensus_rank"]=pd.to_numeric(z["consensus_rank"],errors="raise").astype(float)
    z["market_rank"]=pd.to_numeric(z["market_rank"],errors="raise").astype(float)
    z["consensus_rank_pct"]=z["consensus_rank"]/z["field_size"]
    z["market_rank_pct"]=z["market_rank"]/z["field_size"]
    z["rank_gap_pct"]=(z["market_rank"]-z["consensus_rank"])/z["field_size"]
    z["abs_rank_gap_pct"]=z["rank_gap_pct"].abs()
    z["inv_odds"]=1.0/np.clip(z["final_win_odds"].astype(float),1e-9,None)
    denom=z.groupby(["year","race_id"])["inv_odds"].transform("sum")
    z["market_win_probability"]=z["inv_odds"]/denom
    z["log_market_win_probability"]=np.log(np.clip(z["market_win_probability"],1e-12,1.0))
    z["log_final_win_odds"]=np.log(np.clip(z["final_win_odds"],1e-12,None))
    z["collapse"]=(pd.to_numeric(z["target_top3"],errors="raise").astype(int)==0).astype(int)
    z["king_bucket"]=[sr.rank_bucket(x) for x in z["consensus_rank"]]
    z["market_bucket"]=[sr.rank_bucket(x) for x in z["market_rank"]]
    keep=[
        "year","race_id","horse_id","horse_number","field_size",
        "consensus_rank","market_rank","consensus_rank_pct","market_rank_pct",
        "rank_gap_pct","abs_rank_gap_pct","market_win_probability",
        "log_market_win_probability","final_win_odds","log_final_win_odds",
        "king_bucket","market_bucket","collapse",
    ]
    z=z[keep].copy()
    sf,coverage=sr.build_features(z,snaps)
    if 2026 in set(sf["year"].astype(int)):
        raise SystemExit("2026 sealed in Structural Risk frame")
    return sf,coverage

def train_structural_fold(sf,test_year,train_years):
    train=sf[sf["year"].isin(train_years)].copy()
    test=sf[sf["year"]==test_year].copy().reset_index(drop=True)
    selected=sr.select_features(train,k=30)
    struct_cols=[c for _,c,_,_ in selected]
    if len(struct_cols)<10:
        raise SystemExit(f"too few structural features test={test_year}: {len(struct_cols)}")
    b=sr.model(); f=sr.model()
    b.fit(train[STRUCT_BASELINE_COLS],train["collapse"].astype(int))
    f.fit(train[STRUCT_BASELINE_COLS+struct_cols],train["collapse"].astype(int))
    btr=b.predict_proba(train[STRUCT_BASELINE_COLS])[:,1]
    ftr=f.predict_proba(train[STRUCT_BASELINE_COLS+struct_cols])[:,1]
    dtr=ftr-btr
    pos_thr=float(np.quantile(dtr,0.80))
    abs_thr=float(np.quantile(np.abs(dtr),0.80))
    bte=b.predict_proba(test[STRUCT_BASELINE_COLS])[:,1]
    fte=f.predict_proba(test[STRUCT_BASELINE_COLS+struct_cols])[:,1]
    d=fte-bte
    out=test[["year","race_id","horse_id","horse_number","consensus_rank","market_rank"]].copy()
    out["sr_baseline_risk"]=bte
    out["sr_full_risk"]=fte
    out["sr_delta"]=d
    out["sr_abs_delta"]=np.abs(d)
    out["sr_high_pos"]=(d>=pos_thr).astype(float)
    out["sr_high_abs"]=(np.abs(d)>=abs_thr).astype(float)
    feat_rows=[]
    for rank,(pa,c,raw_auc,cov) in enumerate(selected,1):
        feat_rows.append({
            "test_year":test_year,"train_years":"|".join(map(str,train_years)),
            "rank":rank,"feature":c[3:],"train_predictive_auc":float(pa),
            "train_auc_larger_means_collapse":float(raw_auc),"coverage_pct":100.0*float(cov),
        })
    return out,feat_rows,{
        "test_year":test_year,"train_years":"|".join(map(str,train_years)),
        "train_horses":len(train),"test_horses":len(test),
        "positive_delta_p80":pos_thr,"absolute_delta_p80":abs_thr,
        "test_mean_delta":float(np.mean(d)),"test_mean_abs_delta":float(np.mean(np.abs(d))),
    }

def safe_mean(q,col):
    return float(q[col].mean()) if len(q) else 0.0
def safe_max(q,col):
    return float(q[col].max()) if len(q) else 0.0
def safe_min(q,col):
    return float(q[col].min()) if len(q) else 0.0

def aggregate_structural_race(scores):
    rows=[]
    for (year,rid),g in scores.groupby(["year","race_id"],sort=False):
        g=g.copy()
        k1=g.sort_values(["consensus_rank","horse_number","horse_id"]).iloc[0]
        m1=g.sort_values(["market_rank","horse_number","horse_id"]).iloc[0]
        k3=g[g["consensus_rank"]<=3]; m3=g[g["market_rank"]<=3]
        k6=g[g["consensus_rank"]<=6]; m6=g[g["market_rank"]<=6]
        both3=g[(g["consensus_rank"]<=3)&(g["market_rank"]<=3)]
        rows.append({
            "year":int(year),"race_id":str(rid),
            "sr_king1_delta":float(k1["sr_delta"]),
            "sr_king1_abs_delta":float(k1["sr_abs_delta"]),
            "sr_king1_full_risk":float(k1["sr_full_risk"]),
            "sr_market1_delta":float(m1["sr_delta"]),
            "sr_market1_abs_delta":float(m1["sr_abs_delta"]),
            "sr_market1_full_risk":float(m1["sr_full_risk"]),
            "sr_king_top3_delta_mean":safe_mean(k3,"sr_delta"),
            "sr_king_top3_delta_max":safe_max(k3,"sr_delta"),
            "sr_king_top3_delta_min":safe_min(k3,"sr_delta"),
            "sr_king_top3_abs_mean":safe_mean(k3,"sr_abs_delta"),
            "sr_king_top3_abs_max":safe_max(k3,"sr_abs_delta"),
            "sr_king_top3_high_pos_share":safe_mean(k3,"sr_high_pos"),
            "sr_king_top3_high_abs_share":safe_mean(k3,"sr_high_abs"),
            "sr_market_top3_delta_mean":safe_mean(m3,"sr_delta"),
            "sr_market_top3_delta_max":safe_max(m3,"sr_delta"),
            "sr_market_top3_delta_min":safe_min(m3,"sr_delta"),
            "sr_market_top3_abs_mean":safe_mean(m3,"sr_abs_delta"),
            "sr_market_top3_abs_max":safe_max(m3,"sr_abs_delta"),
            "sr_market_top3_high_pos_share":safe_mean(m3,"sr_high_pos"),
            "sr_market_top3_high_abs_share":safe_mean(m3,"sr_high_abs"),
            "sr_king_top6_abs_mean":safe_mean(k6,"sr_abs_delta"),
            "sr_king_top6_abs_max":safe_max(k6,"sr_abs_delta"),
            "sr_king_top6_high_pos_share":safe_mean(k6,"sr_high_pos"),
            "sr_king_top6_high_abs_share":safe_mean(k6,"sr_high_abs"),
            "sr_market_top6_abs_mean":safe_mean(m6,"sr_abs_delta"),
            "sr_market_top6_abs_max":safe_max(m6,"sr_abs_delta"),
            "sr_market_top6_high_pos_share":safe_mean(m6,"sr_high_pos"),
            "sr_market_top6_high_abs_share":safe_mean(m6,"sr_high_abs"),
            "sr_both_top3_abs_mean":safe_mean(both3,"sr_abs_delta"),
            "sr_both_top3_abs_max":safe_max(both3,"sr_abs_delta"),
            "sr_both_top3_high_pos_share":safe_mean(both3,"sr_high_pos"),
            "sr_both_top3_high_abs_share":safe_mean(both3,"sr_high_abs"),
            "sr_all_abs_mean":safe_mean(g,"sr_abs_delta"),
            "sr_all_abs_max":safe_max(g,"sr_abs_delta"),
        })
    return pd.DataFrame(rows)

def merge_structural(feat,risk_race,test_year):
    z=feat.merge(risk_race,on=["year","race_id"],how="left",validate="one_to_one")
    missing=z[STRUCT_ROUTER_FEATURES].isna().any(axis=1)
    if missing.any():
        sample=z.loc[missing,["year","race_id"]].head(5).to_dict("records")
        raise SystemExit(f"Structural Risk race coverage missing test={test_year} n={int(missing.sum())} sample={sample}")
    return z

def run_router(history,test,features,threads):
    old=list(v1.ROUTER_FEATURES)
    try:
        v1.ROUTER_FEATURES=list(features)
        return v1.train_router(history,test,threads)
    finally:
        v1.ROUTER_FEATURES=old

def choice_delta(choice,kd,od):
    return np.where(choice==1,kd,np.where(choice==2,od,0.0))

def route_counts(choice):
    return {
        "MARKET":int(np.sum(choice==0)),
        "KING":int(np.sum(choice==1)),
        "OUTSIDER":int(np.sum(choice==2)),
    }

def main():
    t0=time.time(); a=parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    c=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    l175=json.loads(Path(a.l175_contract).read_text(encoding="utf-8"))
    if c.get("contract")!="L2_TRIO_ENGINE_ROUTER_V2_STRUCTRISK":
        raise SystemExit("wrong V2 contract")
    if l175.get("contract")!="L175_MARKET_CONTEXT_V1":
        raise SystemExit("wrong L175 contract")
    if c["cost_policy"]["github_standard_cpu_only"] is not True or c["cost_policy"]["gpu"] is not False:
        raise SystemExit("cost guard drift")
    if c["locked_years"]!=[2026]:
        raise SystemExit("2026 lock drift")

    cpu=max(1,os.cpu_count() or 1)
    engine_threads=max(1,cpu//2)
    router_threads=max(1,cpu//2)
    print(f"L2_TRIO_ROUTER_V2_CPU cpu={cpu} engine_threads={engine_threads} router_threads={router_threads}",flush=True)

    _,df=load_horse_dataset(a.dataset_dir)
    df,skipped=attach_win_market(df,a.backfill_root)
    df=attach_outsider(df,a.outsider_predictions)
    df=prepare_l175(df)
    ballot_stats,race_candidates=load_outsider_ballots(parse_year_paths(a.ballots_year))
    df=attach_outsider_ballots(df,ballot_stats,race_candidates)

    snaps=parse_snapshot_paths(a.snapshot_year)
    sf,struct_coverage=prepare_structural_frame(df,snaps)
    struct_race={}; struct_feature_rows=[]; struct_threshold_rows=[]
    for test_year,train_years in FOLDS:
        scores,fr,tr=train_structural_fold(sf,test_year,train_years)
        struct_race[test_year]=aggregate_structural_race(scores)
        struct_feature_rows.extend(fr); struct_threshold_rows.append(tr)
        print("STRUCT_RISK_FOLD_READY "+json.dumps({
            "test_year":test_year,"train_years":list(train_years),
            "horses":len(scores),"races":int(scores["race_id"].nunique()),
            "pos_p80":tr["positive_delta_p80"],"abs_p80":tr["absolute_delta_p80"]
        },separators=(",",":")),flush=True)

    for y in YEARS:
        build_year_matrix(y,df,a.backfill_root,a.matrix_cache)
        build_compact_engine_matrices(y,df,a.matrix_cache)

    loaded={
        "KING":{y:load_engine_year(a.matrix_cache,y,"KING") for y in YEARS},
        "OUTSIDER":{y:load_engine_year(a.matrix_cache,y,"OUTSIDER") for y in YEARS},
    }

    print("L2_TRIO_ROUTER_V2_INITIAL_OOF_START year=2022 split=60/20/20",flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        fk=ex.submit(v1.initial_oof_engine,"KING",loaded["KING"],engine_threads)
        fo=ex.submit(v1.initial_oof_engine,"OUTSIDER",loaded["OUTSIDER"],engine_threads)
        ik=fk.result(); io=fo.result()
    if not np.array_equal(ik["y"],io["y"]) or not np.array_equal(ik["ri"],io["ri"]):
        raise SystemExit("initial OOF alignment drift")
    meta22=race_ids_for_year(a.matrix_cache,2022).iloc[ik["race_start"]:].reset_index(drop=True)
    f22=v1.router_feature_frame(ik["q"],ik["p"],io["p"],ik["ri"],meta22,2022)
    _,_,_,kd22,od22=v1.delta_vectors(ik["y"],ik["ri"],ik["q"],ik["p"],io["p"])
    f22["king_delta"]=kd22; f22["outsider_delta"]=od22
    initial_history=f22.copy()
    risk_history=[]
    print(f"L2_TRIO_ROUTER_V2_INITIAL_OOF_READY races={len(initial_history)}",flush=True)

    fold_rows=[]; route_rows=[]; imp_rows=[]; boot_rows=[]; runtime_rows=[]
    pooled_risk=[]; pooled_base=[]; pooled_o=[]; pooled_k=[]; pooled_oracle=[]
    pooled_risk_minus_base=[]; pooled_risk_minus_out=[]

    for test_year,train_years in FOLDS:
        ft=time.time()
        with ThreadPoolExecutor(max_workers=2) as ex:
            fk=ex.submit(train_one_engine,"KING",loaded["KING"],train_years,test_year,engine_threads)
            fo=ex.submit(train_one_engine,"OUTSIDER",loaded["OUTSIDER"],train_years,test_year,engine_threads)
            rk=fk.result(); ro=fo.result()

        _,_,ytest,ritest,qtest,_=loaded["KING"][test_year]
        meta=race_ids_for_year(a.matrix_cache,test_year)
        feat_base=v1.router_feature_frame(qtest,rk["p"],ro["p"],ritest,meta,test_year)
        feat_risk=merge_structural(feat_base,struct_race[test_year],test_year)
        _,_,_,kd,od=v1.delta_vectors(ytest,ritest,qtest,rk["p"],ro["p"])

        if test_year==2023:
            base_history=initial_history
            predbk,predbo,base_choice,base_imp=run_router(base_history,feat_base,BASE_ROUTER_FEATURES,router_threads)
            predk,predo,risk_choice,risk_imp=predbk,predbo,base_choice,base_imp
            risk_training_policy="2022_OOF_BASE_ONLY"
        else:
            base_history=pd.concat(risk_history,ignore_index=True)
            predbk,predbo,base_choice,base_imp=run_router(base_history,feat_base,BASE_ROUTER_FEATURES,router_threads)
            predk,predo,risk_choice,risk_imp=run_router(base_history,feat_risk,RISK_ROUTER_FEATURES,router_threads)
            risk_training_policy="PRIOR_OOS_YEARS_WITH_STRUCTRISK"

        base_selected=choice_delta(base_choice,kd,od)
        risk_selected=choice_delta(risk_choice,kd,od)
        oracle=np.minimum.reduce([np.zeros(len(kd)),kd,od])
        risk_minus_base=risk_selected-base_selected
        risk_minus_out=risk_selected-od
        bc=route_counts(base_choice); rc=route_counts(risk_choice)

        fold_rows.append({
            "test_year":test_year,"train_years":"|".join(map(str,train_years)),
            "router_training_policy":risk_training_policy,
            "router_train_races":len(base_history),
            "fixed_king_mean_delta":float(kd.mean()),
            "fixed_outsider_mean_delta":float(od.mean()),
            "base_router_mean_delta":float(base_selected.mean()),
            "risk_router_mean_delta":float(risk_selected.mean()),
            "risk_minus_base_router_mean_delta":float(risk_minus_base.mean()),
            "risk_minus_outsider_mean_delta":float(risk_minus_out.mean()),
            "oracle_mean_delta":float(oracle.mean()),
            "base_market_share_pct":100.0*bc["MARKET"]/len(base_choice),
            "base_king_share_pct":100.0*bc["KING"]/len(base_choice),
            "base_outsider_share_pct":100.0*bc["OUTSIDER"]/len(base_choice),
            "risk_market_share_pct":100.0*rc["MARKET"]/len(risk_choice),
            "risk_king_share_pct":100.0*rc["KING"]/len(risk_choice),
            "risk_outsider_share_pct":100.0*rc["OUTSIDER"]/len(risk_choice),
        })
        for comparison,arr,seedoff in (
            ("RISK_ROUTER_VS_MARKET",risk_selected,0),
            ("RISK_MINUS_BASE_ROUTER",risk_minus_base,1000),
            ("RISK_MINUS_OUTSIDER",risk_minus_out,2000),
        ):
            b=bootstrap_mean_ci(arr,reps=BOOT_REPS,seed=SEED+seedoff+test_year)
            boot_rows.append({"scope":str(test_year),"comparison":comparison,**b})

        if test_year!=2023:
            for r in risk_imp:
                imp_rows.append({"test_year":test_year,"router":"STRUCT_RISK",**r})
        for r in base_imp:
            imp_rows.append({"test_year":test_year,"router":"BASE_SAME_HISTORY",**r})

        for i,row in feat_risk.iterrows():
            route_rows.append({
                "year":test_year,"race_id":str(row["race_id"]),"race_date":str(row["race_date"]),
                "base_route":("MARKET","KING","OUTSIDER")[int(base_choice[i])],
                "risk_route":("MARKET","KING","OUTSIDER")[int(risk_choice[i])],
                "pred_king_delta":float(predk[i]),"pred_outsider_delta":float(predo[i]),
                "actual_king_delta":float(kd[i]),"actual_outsider_delta":float(od[i]),
                "actual_base_router_delta":float(base_selected[i]),
                "actual_risk_router_delta":float(risk_selected[i]),
                "sr_king1_delta":float(row["sr_king1_delta"]),
                "sr_king1_abs_delta":float(row["sr_king1_abs_delta"]),
                "sr_market1_delta":float(row["sr_market1_delta"]),
                "sr_market1_abs_delta":float(row["sr_market1_abs_delta"]),
                "sr_king_top3_abs_max":float(row["sr_king_top3_abs_max"]),
                "sr_market_top3_abs_max":float(row["sr_market_top3_abs_max"]),
            })

        pooled_risk.append(risk_selected); pooled_base.append(base_selected)
        pooled_o.append(od); pooled_k.append(kd); pooled_oracle.append(oracle)
        pooled_risk_minus_base.append(risk_minus_base); pooled_risk_minus_out.append(risk_minus_out)

        hist=feat_risk.copy()
        hist["king_delta"]=kd; hist["outsider_delta"]=od
        risk_history.append(hist)

        runtime_rows.append({
            "test_year":test_year,"seconds":time.time()-ft,
            "router_train_races":len(base_history),"risk_features_used":int(test_year!=2023),
        })
        print("L2_TRIO_ROUTER_V2_FOLD_DONE "+json.dumps({
            "test_year":test_year,
            "fixed_outsider":float(od.mean()),
            "base_router":float(base_selected.mean()),
            "risk_router":float(risk_selected.mean()),
            "risk_minus_base":float(risk_minus_base.mean()),
            "risk_minus_outsider":float(risk_minus_out.mean()),
            "base_routes":bc,"risk_routes":rc
        },separators=(",",":")),flush=True)

    pr=np.concatenate(pooled_risk); pb=np.concatenate(pooled_base)
    po=np.concatenate(pooled_o); pk=np.concatenate(pooled_k); por=np.concatenate(pooled_oracle)
    prb=np.concatenate(pooled_risk_minus_base); pro=np.concatenate(pooled_risk_minus_out)
    pooled_risk_boot=bootstrap_mean_ci(pr,reps=BOOT_REPS,seed=SEED+5000)
    pooled_base_diff_boot=bootstrap_mean_ci(prb,reps=BOOT_REPS,seed=SEED+6000)
    pooled_out_diff_boot=bootstrap_mean_ci(pro,reps=BOOT_REPS,seed=SEED+7000)
    boot_rows.append({"scope":"POOLED_2023_2025","comparison":"RISK_ROUTER_VS_MARKET",**pooled_risk_boot})
    boot_rows.append({"scope":"POOLED_2023_2025","comparison":"RISK_MINUS_BASE_ROUTER",**pooled_base_diff_boot})
    boot_rows.append({"scope":"POOLED_2023_2025","comparison":"RISK_MINUS_OUTSIDER",**pooled_out_diff_boot})

    # Risk is only learnable by the router from 2024 onward because 2022 has no earlier OOS Structural Risk history.
    mask_2425=np.concatenate([
        np.zeros(len(pooled_risk[0]),dtype=bool),
        np.ones(len(pooled_risk[1]),dtype=bool),
        np.ones(len(pooled_risk[2]),dtype=bool),
    ])
    pr_2425=pr[mask_2425]; pb_2425=pb[mask_2425]; po_2425=po[mask_2425]
    dbase_2425=pr_2425-pb_2425; dout_2425=pr_2425-po_2425
    b2425=bootstrap_mean_ci(dbase_2425,reps=BOOT_REPS,seed=SEED+8000)
    o2425=bootstrap_mean_ci(dout_2425,reps=BOOT_REPS,seed=SEED+9000)
    boot_rows.append({"scope":"POOLED_2024_2025","comparison":"RISK_MINUS_BASE_ROUTER",**b2425})
    boot_rows.append({"scope":"POOLED_2024_2025","comparison":"RISK_MINUS_OUTSIDER",**o2425})

    write_csv(out/"fold-metrics.csv",fold_rows)
    write_csv(out/"bootstrap.csv",boot_rows)
    write_csv(out/"router-feature-importance.csv",imp_rows)
    write_csv(out/"struct-risk-selected-features.csv",struct_feature_rows)
    write_csv(out/"struct-risk-thresholds.csv",struct_threshold_rows)
    write_csv(out/"runtime.csv",runtime_rows)
    write_csv(out/"skipped-win-market-races.csv",skipped)
    pd.DataFrame(route_rows).to_csv(out/"race-routes.csv.gz",index=False,compression="gzip")

    summary={
        "contract":"L2_TRIO_ENGINE_ROUTER_V2_STRUCTRISK_RESULT",
        "architecture":"MARKET_KING_OUTSIDER_ROUTER_PLUS_STRUCTURAL_RISK",
        "base_router_features":BASE_ROUTER_FEATURES,
        "structural_router_features":STRUCT_ROUTER_FEATURES,
        "structural_risk":{
            "source":"runner-local strict walk-forward reconstruction from annual snapshots",
            "snapshot_coverage_pct":100.0*float(struct_coverage),
            "folds":[{"test_year":y,"train_years":list(t)} for y,t in FOLDS],
            "2023_router_note":"Structural Risk exists pre-race for 2023, but there is no prior OOS Structural Risk router-label history before 2023; V2 therefore intentionally matches base Router in 2023.",
            "u_shape_guard":"Both signed delta and absolute delta/high-absolute-risk aggregates are supplied. Rankings are never rewritten.",
        },
        "fold_metrics":fold_rows,
        "pooled_2023_2025":{
            "fixed_king_mean_delta":float(pk.mean()),
            "fixed_outsider_mean_delta":float(po.mean()),
            "base_router_mean_delta":float(pb.mean()),
            "risk_router_mean_delta":float(pr.mean()),
            "risk_minus_base_mean_delta":float(prb.mean()),
            "risk_minus_outsider_mean_delta":float(pro.mean()),
            "oracle_mean_delta":float(por.mean()),
            "risk_vs_market_bootstrap":pooled_risk_boot,
            "risk_minus_base_bootstrap":pooled_base_diff_boot,
            "risk_minus_outsider_bootstrap":pooled_out_diff_boot,
        },
        "pooled_2024_2025_structrisk_effect":{
            "base_router_mean_delta":float(pb_2425.mean()),
            "risk_router_mean_delta":float(pr_2425.mean()),
            "fixed_outsider_mean_delta":float(po_2425.mean()),
            "risk_minus_base_mean_delta":float(dbase_2425.mean()),
            "risk_minus_outsider_mean_delta":float(dout_2425.mean()),
            "risk_minus_base_bootstrap":b2425,
            "risk_minus_outsider_bootstrap":o2425,
        },
        "success_gate":{
            "structural_risk_improves_same_history_router_2024_2025":bool(float(dbase_2425.mean())<0),
            "structural_risk_improves_same_history_router_ci95_2024_2025":bool(float(b2425["ci_high"])<0),
            "beats_fixed_outsider_2024_2025":bool(float(dout_2425.mean())<0),
            "beats_fixed_outsider_ci95_2024_2025":bool(float(o2425["ci_high"])<0),
            "reference_fixed_outsider_2023_2025":-0.00614873642986566,
        },
        "market_price_used":True,"payout_used":False,"roi_used":False,"staking_used":False,
        "rank_rewrite":False,"2026_locked":True,"promotion":False,
        "elapsed_seconds":time.time()-t0,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 TRIO Engine Router V2 + Structural Risk\n\n"
        "Adds leak-safe Structural Risk race context to the existing MARKET / MARKET+KING / MARKET+OUTSIDER router. "
        "The correction engines and horse rankings are unchanged. Structural Risk is reconstructed strict walk-forward "
        "from annual snapshots. For 2024/2025, a base router and a Structural-Risk router are trained on exactly the same prior OOS years "
        "so the incremental value of Structural Risk is measured directly. No ROI, payout or staking optimization. 2026 sealed.\n",
        encoding="utf-8"
    )
    print("===== FOLD METRICS ====="); print((out/"fold-metrics.csv").read_text())
    print("===== BOOTSTRAP ====="); print((out/"bootstrap.csv").read_text())
    print("===== STRUCT RISK THRESHOLDS ====="); print((out/"struct-risk-thresholds.csv").read_text())
    print("===== SUMMARY ====="); print((out/"summary.json").read_text())
    print("L2_TRIO_ENGINE_ROUTER_V2_STRUCTRISK_READY")

if __name__=="__main__":
    main()
