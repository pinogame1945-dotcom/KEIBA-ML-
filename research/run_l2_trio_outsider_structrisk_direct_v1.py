#!/usr/bin/env python3
import argparse,itertools,json,math,os,time
from pathlib import Path

import numpy as np
import pandas as pd

import run_l1_structural_risk_fullfield_v1 as sr
import run_l2_trio_engine_router_v1 as router_v1
import run_l2_trio_engine_router_v2_structrisk as router_v2
from run_l2_trio_probability_v2 import (
    YEARS,FOLDS,write_csv,load_horse_dataset,attach_win_market,attach_outsider,prepare_l175
)
from run_l2_trio_market_residual_v3 import build_year_matrix,offset_softmax
from run_l2_trio_split_residual_v4 import (
    parse_year_paths,load_outsider_ballots,attach_outsider_ballots,
    build_compact_engine_matrices,load_engine_year,train_one_engine,race_ids_for_year
)
from run_l2_trio_v3_robustness_audit import bootstrap_mean_ci,race_vectors

SEED=20261004
BOOT_REPS=5000
SCOPES=("TOP1","TOP3")
MODES=("POS","ABS")
LAMBDAS=(0.0,0.05,0.10,0.15,0.20,0.30,0.40,0.60)
STRUCT_BASELINE_COLS=[
    "field_size","market_rank_pct","consensus_rank_pct",
    "log_market_win_probability","log_final_win_odds",
    "rank_gap_pct","abs_rank_gap_pct",
]

def parse_args():
    p=argparse.ArgumentParser(description="Direct Structural Risk correction on MARKET+OUTSIDER TRIO probabilities.")
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
        y,p=spec.split("=",1); out[int(y)]=p
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
    sf,coverage=sr.build_features(z[keep].copy(),snaps)
    return sf,coverage

def fit_structural(train,test,test_year,train_label):
    selected=sr.select_features(train,k=30)
    struct_cols=[c for _,c,_,_ in selected]
    if len(struct_cols)<10:
        raise SystemExit(f"too few structural features test={test_year}")
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
    out["sr_delta"]=d
    out["sr_abs_delta"]=np.abs(d)
    out["sr_high_pos"]=(d>=pos_thr).astype(np.uint8)
    out["sr_high_abs"]=(np.abs(d)>=abs_thr).astype(np.uint8)
    meta={
        "test_year":int(test_year),"train_label":str(train_label),
        "train_horses":int(len(train)),"test_horses":int(len(test)),
        "positive_delta_p80":pos_thr,"absolute_delta_p80":abs_thr,
        "test_mean_delta":float(np.mean(d)),"test_mean_abs_delta":float(np.mean(np.abs(d))),
        "feature_count":len(struct_cols),
    }
    return out,meta

def structural_scores_for_year(sf,test_year,train_years):
    train=sf[sf["year"].isin(train_years)].copy()
    test=sf[sf["year"]==test_year].copy().reset_index(drop=True)
    return fit_structural(train,test,test_year,"|".join(map(str,train_years)))

def structural_scores_warm_2022(sf,races22):
    races22=races22.sort_values("race_index").reset_index(drop=True)
    n=len(races22)
    fit_cut=max(1,int(n*0.60))
    eval_cut=max(fit_cut+1,int(n*0.80))
    fit_ids=set(races22.iloc[:fit_cut]["race_id"].astype(str))
    eval_ids=set(races22.iloc[eval_cut:]["race_id"].astype(str))
    train=sf[(sf["year"]==2022)&(sf["race_id"].astype(str).isin(fit_ids))].copy()
    test=sf[(sf["year"]==2022)&(sf["race_id"].astype(str).isin(eval_ids))].copy().reset_index(drop=True)
    scores,meta=fit_structural(train,test,2022,"2022_FIRST60")
    meta["warm_policy"]="eligible_races_first60_train_middle20_unused_last20_eval"
    meta["fit_races"]=len(fit_ids); meta["eval_races"]=len(eval_ids)
    return scores,meta,eval_cut

def risk_ticket_vectors(df,scores,matrix_cache,year,race_meta=None):
    ydf=df[df["year"]==year].copy()
    groups={str(rid):g for rid,g in ydf.groupby("race_id",sort=False)}
    smap={}
    for r in scores.itertuples(index=False):
        smap[(str(r.race_id),str(r.horse_id))]={
            "POS":int(r.sr_high_pos),"ABS":int(r.sr_high_abs),
            "consensus_rank":float(r.consensus_rank),"market_rank":float(r.market_rank),
        }
    races=(race_meta.copy() if race_meta is not None else race_ids_for_year(matrix_cache,year)).sort_values("race_index").reset_index(drop=True)
    buffers={(scope,mode):[] for scope in SCOPES for mode in MODES}
    total=0
    for rr in races.itertuples(index=False):
        rid=str(rr.race_id)
        if rid not in groups:
            raise SystemExit(f"risk ticket race missing year={year} race={rid}")
        g=groups[rid].sort_values(["consensus_rank","horse_number","horse_id"]).reset_index(drop=True)
        n=len(g)
        comb=np.asarray(list(itertools.combinations(range(n),3)),dtype=np.int16)
        expected=int(rr.combos) if hasattr(rr,"combos") else len(comb)
        if len(comb)!=expected:
            raise SystemExit(f"risk combo mismatch y={year} race={rid}")
        cr=g["consensus_rank"].to_numpy(dtype=float)
        mr=g["market_rank"].to_numpy(dtype=float)
        hids=g["horse_id"].astype(str).tolist()
        for scope in SCOPES:
            lim=1 if scope=="TOP1" else 3
            eligible=(cr<=lim)|(mr<=lim)
            for mode in MODES:
                flag=np.zeros(n,dtype=np.float64)
                for i,hid in enumerate(hids):
                    d=smap.get((rid,hid))
                    if d is None:
                        raise SystemExit(f"structural horse missing y={year} race={rid} horse={hid}")
                    flag[i]=1.0 if eligible[i] and d[mode]>0 else 0.0
                buffers[(scope,mode)].append(flag[comb].sum(axis=1).astype(np.float32))
        total+=len(comb)
    return {k:np.concatenate(v).astype(np.float32,copy=False) for k,v in buffers.items()},total

def adjust_probability(base_p,ri,risk_count,lam):
    if float(lam)==0.0:
        return np.asarray(base_p,dtype=np.float64)
    return offset_softmax(-np.asarray(risk_count,dtype=np.float64),base_p,ri,float(lam))

def mean_logloss(y,p,ri):
    rv=race_vectors(y,p,p,ri)
    # true_q == true_p here, so use direct log loss from true_p.
    return float(np.mean(-np.log(np.clip(rv["true_p"],1e-12,None))))

def evaluate_candidate(history,scope,mode,lam):
    losses=[]
    races=0
    for ep in history:
        p=adjust_probability(ep["p"],ep["ri"],ep["risk"][(scope,mode)],lam)
        rv=race_vectors(ep["y"],p,ep["q"],ep["ri"])
        loss=-np.log(np.clip(rv["true_p"],1e-12,None))
        losses.append(loss); races+=len(loss)
    return float(np.concatenate(losses).mean()),races

def choose_policy(history):
    rows=[]; best=None
    for scope in SCOPES:
        for mode in MODES:
            for lam in LAMBDAS:
                ll,n=evaluate_candidate(history,scope,mode,lam)
                row={"scope":scope,"mode":mode,"lambda":float(lam),"history_races":n,"history_logloss":ll}
                rows.append(row)
                key=(ll,0 if lam==0 else 1,float(lam),scope,mode)
                if best is None or key<best[0]:
                    best=(key,row)
    return best[1],rows

def main():
    t0=time.time(); a=parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    l175=json.loads(Path(a.l175_contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L2_TRIO_OUTSIDER_STRUCTRISK_DIRECT_V1":
        raise SystemExit("wrong direct correction contract")
    if l175.get("contract")!="L175_MARKET_CONTEXT_V1":
        raise SystemExit("wrong L175 contract")
    if contract["cost_policy"]["github_standard_cpu_only"] is not True or contract["cost_policy"]["gpu"] is not False:
        raise SystemExit("cost guard drift")
    if contract["locked_years"]!=[2026]:
        raise SystemExit("2026 lock drift")

    cpu=max(1,os.cpu_count() or 1)
    engine_threads=max(1,cpu)
    print(f"OUTSIDER_STRUCTRISK_DIRECT_CPU cpu={cpu} outsider_threads={engine_threads}",flush=True)

    _,df=load_horse_dataset(a.dataset_dir)
    df,skipped=attach_win_market(df,a.backfill_root)
    df=attach_outsider(df,a.outsider_predictions)
    df=prepare_l175(df)
    ballot_stats,race_candidates=load_outsider_ballots(parse_year_paths(a.ballots_year))
    df=attach_outsider_ballots(df,ballot_stats,race_candidates)

    snaps=parse_snapshot_paths(a.snapshot_year)
    sf,struct_coverage=prepare_structural_frame(df,snaps)

    for y in YEARS:
        build_year_matrix(y,df,a.backfill_root,a.matrix_cache)
        build_compact_engine_matrices(y,df,a.matrix_cache)
    loaded={y:load_engine_year(a.matrix_cache,y,"OUTSIDER") for y in YEARS}

    # 2022 warm-start: outsider engine and Structural Risk are both OOF on the same last 20% eligible races.
    warm_engine=router_v1.initial_oof_engine("OUTSIDER",loaded,engine_threads)
    races22=race_ids_for_year(a.matrix_cache,2022)
    warm_scores,warm_meta,eval_cut=structural_scores_warm_2022(sf,races22)
    warm_races=races22.iloc[eval_cut:].reset_index(drop=True)
    warm_risk,total=risk_ticket_vectors(df,warm_scores,a.matrix_cache,2022,warm_races)
    if total!=len(warm_engine["p"]):
        raise SystemExit(f"warm ticket alignment mismatch risk={total} engine={len(warm_engine['p'])}")
    history=[{
        "year":2022,"p":warm_engine["p"],"q":warm_engine["q"],
        "y":warm_engine["y"],"ri":warm_engine["ri"],"risk":warm_risk,
    }]
    print("DIRECT_WARM_READY "+json.dumps({
        "races":warm_engine["race_count"],"tickets":len(warm_engine["p"]),
        "struct":warm_meta
    },separators=(",",":")),flush=True)

    fold_rows=[]; grid_rows=[]; boot_rows=[]; risk_meta_rows=[warm_meta]; runtime_rows=[]; diag_rows=[]
    pooled_base=[]; pooled_corr=[]; pooled_diff=[]

    for test_year,train_years in FOLDS:
        ft=time.time()
        policy,grid=choose_policy(history)
        for r in grid:
            grid_rows.append({"test_year":test_year,"selected":int(
                r["scope"]==policy["scope"] and r["mode"]==policy["mode"] and abs(r["lambda"]-policy["lambda"])<1e-12
            ),**r})

        scores,rmeta=structural_scores_for_year(sf,test_year,train_years)
        risk_meta_rows.append(rmeta)
        risk,total=risk_ticket_vectors(df,scores,a.matrix_cache,test_year)

        ro=train_one_engine("OUTSIDER",loaded,train_years,test_year,engine_threads)
        _,_,ytest,ritest,qtest,_=loaded[test_year]
        if total!=len(ro["p"]):
            raise SystemExit(f"ticket alignment mismatch y={test_year} risk={total} engine={len(ro['p'])}")
        pcorr=adjust_probability(ro["p"],ritest,risk[(policy["scope"],policy["mode"])],policy["lambda"])

        rvb=race_vectors(ytest,ro["p"],qtest,ritest)
        rvc=race_vectors(ytest,pcorr,qtest,ritest)
        diff=rvc["logloss_delta"]-rvb["logloss_delta"]
        pooled_base.append(rvb["logloss_delta"]); pooled_corr.append(rvc["logloss_delta"]); pooled_diff.append(diff)
        b=bootstrap_mean_ci(diff,reps=BOOT_REPS,seed=SEED+test_year)
        boot_rows.append({"scope":str(test_year),"comparison":"CORRECTED_MINUS_OUTSIDER_BASE",**b})
        bc=bootstrap_mean_ci(rvc["logloss_delta"],reps=BOOT_REPS,seed=SEED+1000+test_year)
        boot_rows.append({"scope":str(test_year),"comparison":"CORRECTED_VS_MARKET",**bc})

        starts,counts=router_v1.group_layout(ritest)
        selected_risk=risk[(policy["scope"],policy["mode"])]
        risky_ticket_share=float(np.mean(selected_risk>0))
        fold_rows.append({
            "test_year":test_year,"train_years":"|".join(map(str,train_years)),
            "history_oos_races":int(sum(len(router_v1.group_layout(ep["ri"])[0]) for ep in history)),
            "selected_scope":policy["scope"],"selected_mode":policy["mode"],"selected_lambda":policy["lambda"],
            "base_outsider_mean_delta_vs_market":float(rvb["logloss_delta"].mean()),
            "corrected_mean_delta_vs_market":float(rvc["logloss_delta"].mean()),
            "corrected_minus_base_mean_delta":float(diff.mean()),
            "risky_ticket_share_pct":100.0*risky_ticket_share,
            "engine_alpha":float(ro["alpha"]),
            "engine_runtime_seconds":float(ro["runtime_seconds"]),
        })
        meta=race_ids_for_year(a.matrix_cache,test_year)
        for i,rr in enumerate(meta.itertuples(index=False)):
            s=int(starts[i]); c=int(counts[i]); e=s+c
            diag_rows.append({
                "test_year":test_year,"race_id":str(rr.race_id),"race_date":str(rr.race_date),
                "selected_scope":policy["scope"],"selected_mode":policy["mode"],"selected_lambda":policy["lambda"],
                "risky_ticket_share_pct":100.0*float(np.mean(selected_risk[s:e]>0)),
                "base_true_ticket_probability":float(rvb["true_p"][i]),
                "corrected_true_ticket_probability":float(rvc["true_p"][i]),
                "base_logloss_delta_vs_market":float(rvb["logloss_delta"][i]),
                "corrected_logloss_delta_vs_market":float(rvc["logloss_delta"][i]),
            })

        history.append({
            "year":test_year,"p":ro["p"],"q":qtest,"y":ytest,"ri":ritest,"risk":risk
        })
        runtime_rows.append({"test_year":test_year,"seconds":time.time()-ft})
        print("DIRECT_FOLD_DONE "+json.dumps({
            "test_year":test_year,"policy":policy,
            "base":float(rvb["logloss_delta"].mean()),
            "corrected":float(rvc["logloss_delta"].mean()),
            "diff":float(diff.mean()),"ci":[b["ci_low"],b["ci_high"]]
        },separators=(",",":")),flush=True)

    pb=np.concatenate(pooled_base); pc=np.concatenate(pooled_corr); pdiff=np.concatenate(pooled_diff)
    bpool=bootstrap_mean_ci(pdiff,reps=BOOT_REPS,seed=SEED+5000)
    cpool=bootstrap_mean_ci(pc,reps=BOOT_REPS,seed=SEED+6000)
    boot_rows.append({"scope":"POOLED_2023_2025","comparison":"CORRECTED_MINUS_OUTSIDER_BASE",**bpool})
    boot_rows.append({"scope":"POOLED_2023_2025","comparison":"CORRECTED_VS_MARKET",**cpool})

    write_csv(out/"fold-metrics.csv",fold_rows)
    write_csv(out/"policy-grid.csv",grid_rows)
    write_csv(out/"bootstrap.csv",boot_rows)
    write_csv(out/"struct-risk-meta.csv",risk_meta_rows)
    write_csv(out/"runtime.csv",runtime_rows)
    write_csv(out/"skipped-win-market-races.csv",skipped)
    pd.DataFrame(diag_rows).to_csv(out/"race-diagnostics.csv.gz",index=False,compression="gzip")

    summary={
        "contract":"L2_TRIO_OUTSIDER_STRUCTRISK_DIRECT_V1_RESULT",
        "architecture":"MARKET_PLUS_OUTSIDER_THEN_DIRECT_STRUCTURAL_RISK_TICKET_ATTENUATION",
        "correction":{
            "candidate_scopes":list(SCOPES),"candidate_modes":list(MODES),"lambda_grid":list(LAMBDAS),
            "selection":"strict prior OOS mean race logloss only",
            "mechanism":"For tickets containing a warned upper-ranked horse, multiply MARKET+OUTSIDER ticket probability by exp(-lambda * warned_horse_count), then renormalize within race.",
            "rank_rewrite":False,"router_used":False,
        },
        "structural_risk":{
            "snapshot_coverage_pct":100.0*float(struct_coverage),
            "warm_2022":warm_meta,
            "folds":[{"test_year":y,"train_years":list(t)} for y,t in FOLDS],
            "u_shape_guard":"POS and ABS warning definitions compete using prior OOS only.",
        },
        "fold_metrics":fold_rows,
        "pooled_2023_2025":{
            "base_outsider_mean_delta_vs_market":float(pb.mean()),
            "corrected_mean_delta_vs_market":float(pc.mean()),
            "corrected_minus_base_mean_delta":float(pdiff.mean()),
            "corrected_minus_base_bootstrap":bpool,
            "corrected_vs_market_bootstrap":cpool,
        },
        "success_gate":{
            "beats_outsider_mean":bool(float(pdiff.mean())<0),
            "beats_outsider_ci95":bool(float(bpool["ci_high"])<0),
            "beats_reference_minus_0_00615":bool(float(pc.mean()) < -0.00614873642986566),
            "reference_fixed_outsider_mean":-0.00614873642986566,
        },
        "market_price_used":True,"payout_used":False,"roi_used":False,"staking_used":False,
        "2026_locked":True,"promotion":False,"elapsed_seconds":time.time()-t0,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 TRIO Outsider + Direct Structural Risk V1\n\n"
        "Keeps MARKET+OUTSIDER as the probability engine and applies only a small, prior-OOS-selected attenuation to tickets containing upper-ranked horses warned by Structural Risk. "
        "No router, no rank rewrite, no payout/ROI/staking optimization, and 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("===== FOLD METRICS ====="); print((out/"fold-metrics.csv").read_text())
    print("===== BOOTSTRAP ====="); print((out/"bootstrap.csv").read_text())
    print("===== SUMMARY ====="); print((out/"summary.json").read_text())
    print("L2_TRIO_OUTSIDER_STRUCTRISK_DIRECT_V1_READY")

if __name__=="__main__":
    main()
