#!/usr/bin/env python3
import argparse,hashlib,json,math,multiprocessing,os,time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from run_l2_core_v1 import load_dataset,write_csv
from run_l2_pricer_latent_race_v1 import load_outcomes,build_races,event_prob_from_rows

EPS=1e-15
TEST_YEARS=(2023,2024,2025)
BET_TYPES=("WIN","EXACTA","TRIO","TRIFECTA")
DRAWS=1024
SHUFFLES=8
BOOT_REPS=3000
SIGMA0=0.40
LAMBDA=0.50

MODELS=(
    "BASE_PL",
    "NORMAL_HOMO_S040",
    "HETERO_REAL",
    "HETERO_RACE_MEAN",
    "HETERO_RACE_RMS",
    "HETERO_SHUFFLE_8",
    "HETERO_REVERSED",
)

def parse_args():
    p=argparse.ArgumentParser(description="L2 PRICER V5 Seven-King uncertainty causal audit")
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--workers",type=int,default=0)
    return p.parse_args()

def clip_prob(x):
    return min(max(float(x),EPS),1.0)

def seed_for(rid,label):
    raw=f"{rid}|{label}|20261004".encode()
    return int.from_bytes(hashlib.blake2b(raw,digest_size=8).digest(),"little") & 0x7fffffff

def shared_z(race):
    rng=np.random.default_rng(seed_for(race["race_id"],"V5_COMMON_Z"))
    half=(DRAWS+1)//2
    z=rng.standard_normal((half,len(race["base"])),dtype=np.float64)
    return np.concatenate((z,-z),axis=0)[:DRAWS]

def rows_from_sigma(race,z,sigma_vec):
    logw=np.log(np.clip(race["base"],EPS,None))[None,:]+z*np.asarray(sigma_vec,dtype=np.float64)[None,:]
    logw-=np.max(logw,axis=1,keepdims=True)
    w=np.exp(logw)
    w/=np.sum(w,axis=1,keepdims=True)
    return w

def reverse_uncertainty(u):
    u=np.asarray(u,dtype=np.float64)
    order=np.argsort(u,kind="stable")
    out=np.empty_like(u)
    out[order]=u[order[::-1]]
    return out

def shuffle_uncertainty(race,k):
    u=np.asarray(race["uncertainty"],dtype=np.float64)
    rng=np.random.default_rng(seed_for(race["race_id"],f"V5_SHUFFLE_{k}"))
    return u[rng.permutation(len(u))]

def loss_from_rows(race,rows):
    out={}
    for bet in BET_TYPES:
        p=event_prob_from_rows(rows,race["winning_idx"].get(bet),bet)
        out[bet]=None if p is None else -math.log(clip_prob(p))
    return out

def evaluate_race(race):
    z=shared_z(race)
    u=np.asarray(race["uncertainty"],dtype=np.float64)
    real_sigma=SIGMA0*(1.0+LAMBDA*u)
    sigma_mean=np.full(len(u),float(np.mean(real_sigma)))
    sigma_rms=np.full(len(u),float(np.sqrt(np.mean(real_sigma**2))))
    sigma_reverse=SIGMA0*(1.0+LAMBDA*reverse_uncertainty(u))

    model_rows={
        "BASE_PL":np.asarray(race["base"],dtype=np.float64).reshape(1,-1),
        "NORMAL_HOMO_S040":rows_from_sigma(race,z,np.full(len(u),SIGMA0)),
        "HETERO_REAL":rows_from_sigma(race,z,real_sigma),
        "HETERO_RACE_MEAN":rows_from_sigma(race,z,sigma_mean),
        "HETERO_RACE_RMS":rows_from_sigma(race,z,sigma_rms),
        "HETERO_REVERSED":rows_from_sigma(race,z,sigma_reverse),
    }
    shuffled=[]
    for k in range(SHUFFLES):
        su=shuffle_uncertainty(race,k)
        shuffled.append(rows_from_sigma(race,z,SIGMA0*(1.0+LAMBDA*su)))
    model_rows["HETERO_SHUFFLE_8"]=np.concatenate(shuffled,axis=0)

    out={}
    for model,rows in model_rows.items():
        out[model]=loss_from_rows(race,rows)
    return {
        "race_id":race["race_id"],"race_date":race["race_date"],"year":race["year"],
        "subsets":race["subsets"],
        "losses":out,
        "uncertainty_mean":float(np.mean(u)),
        "uncertainty_std":float(np.std(u)),
        "sigma_mean":float(np.mean(real_sigma)),
        "sigma_rms":float(np.sqrt(np.mean(real_sigma**2))),
    }

def _chunk_worker(chunk):
    return [evaluate_race(r) for r in chunk]

def evaluate_all(races,workers):
    if workers<=1:
        return _chunk_worker(races)
    chunks=[races[i::workers] for i in range(workers)]
    ctx=multiprocessing.get_context("fork")
    out=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as ex:
        for part in ex.map(_chunk_worker,[c for c in chunks if c]):
            out.extend(part)
    return out

def metrics_rows(results):
    rows=[]
    for year in TEST_YEARS:
        yr=[r for r in results if r["year"]==year]
        for model in MODELS:
            for subset in ("ALL","TAIL11_TRUE","KING1_MISS_TOP3_TRUE","NEITHER_STRESS"):
                for bet in BET_TYPES:
                    vals=[]
                    for r in yr:
                        active=True if subset=="ALL" else bool(r["subsets"][subset])
                        if not active: continue
                        v=r["losses"][model][bet]
                        if v is not None: vals.append(float(v))
                    if not vals: continue
                    rows.append({
                        "test_year":year,"model":model,"subset":subset,"bet_type":bet,
                        "races":len(vals),"nll":float(np.mean(vals))
                    })
    base={(r["test_year"],r["subset"],r["bet_type"]):r["nll"] for r in rows if r["model"]=="BASE_PL"}
    for r in rows:
        b=base[(r["test_year"],r["subset"],r["bet_type"])]
        r["base_pl_nll"]=b
        r["delta_vs_base"]=r["nll"]-b
    return rows

def pair_diff(results,year,a,b):
    rs=[r for r in results if (year=="POOLED" or r["year"]==year)]
    vals=[]
    years=[]
    for r in rs:
        va=r["losses"][a]["TRIFECTA"]; vb=r["losses"][b]["TRIFECTA"]
        if va is None or vb is None: continue
        vals.append(float(va)-float(vb)); years.append(int(r["year"]))
    return np.asarray(vals,dtype=np.float64),np.asarray(years,dtype=np.int32)

def bootstrap_ci(d,years,label):
    rng=np.random.default_rng(seed_for(label,"BOOT"))
    means=np.empty(BOOT_REPS,dtype=np.float64)
    if len(set(years.tolist()))<=1:
        n=len(d)
        for start in range(0,BOOT_REPS,200):
            k=min(200,BOOT_REPS-start)
            idx=rng.integers(0,n,size=(k,n))
            means[start:start+k]=d[idx].mean(axis=1)
    else:
        uniq=sorted(set(years.tolist()))
        pos={y:np.where(years==y)[0] for y in uniq}
        total=len(d)
        for i in range(BOOT_REPS):
            pieces=[]
            for y in uniq:
                idx=pos[y]
                pieces.append(d[rng.choice(idx,size=len(idx),replace=True)])
            means[i]=np.concatenate(pieces).mean()
    lo,hi=np.quantile(means,[0.025,0.975])
    return float(lo),float(hi)

def pairwise_rows(results):
    pairs=(
        ("HETERO_REAL","HETERO_SHUFFLE_8"),
        ("HETERO_REAL","HETERO_RACE_RMS"),
        ("HETERO_REAL","HETERO_RACE_MEAN"),
        ("HETERO_REAL","NORMAL_HOMO_S040"),
        ("HETERO_REAL","HETERO_REVERSED"),
        ("HETERO_RACE_RMS","NORMAL_HOMO_S040"),
    )
    rows=[]
    for year in (*TEST_YEARS,"POOLED"):
        for a,b in pairs:
            d,ys=pair_diff(results,year,a,b)
            lo,hi=bootstrap_ci(d,ys,f"{year}|{a}|{b}")
            rows.append({
                "test_year":year,"model_a":a,"model_b":b,"metric":"TRIFECTA_NLL",
                "mean_delta_a_minus_b":float(np.mean(d)),
                "ci95_low":lo,"ci95_high":hi,
                "a_better_race_pct":float(100*np.mean(d<0)),
                "races":len(d),"bootstrap_reps":BOOT_REPS
            })
    return rows

def fold_summary(metric_rows):
    out=[]
    for year in TEST_YEARS:
        row={"test_year":year}
        for model in MODELS:
            hit=[r for r in metric_rows if r["test_year"]==year and r["model"]==model and r["subset"]=="ALL" and r["bet_type"]=="TRIFECTA"][0]
            key=model.lower()
            row[f"{key}_trifecta_nll"]=hit["nll"]
            row[f"{key}_delta_vs_base"]=hit["delta_vs_base"]
        out.append(row)
    return out

def uncertainty_summary(results):
    rows=[]
    for year in TEST_YEARS:
        q=[r for r in results if r["year"]==year]
        rows.append({
            "test_year":year,"races":len(q),
            "mean_uncertainty_mean":float(np.mean([r["uncertainty_mean"] for r in q])),
            "mean_uncertainty_std":float(np.mean([r["uncertainty_std"] for r in q])),
            "mean_sigma_mean":float(np.mean([r["sigma_mean"] for r in q])),
            "mean_sigma_rms":float(np.mean([r["sigma_rms"] for r in q])),
        })
    return rows

def verdict(pair_rows):
    def row(year,a,b):
        return next(r for r in pair_rows if r["test_year"]==year and r["model_a"]==a and r["model_b"]==b)
    pooled_shuffle=row("POOLED","HETERO_REAL","HETERO_SHUFFLE_8")
    pooled_rms=row("POOLED","HETERO_REAL","HETERO_RACE_RMS")
    yearly_shuffle=[row(y,"HETERO_REAL","HETERO_SHUFFLE_8") for y in TEST_YEARS]
    yearly_rms=[row(y,"HETERO_REAL","HETERO_RACE_RMS") for y in TEST_YEARS]
    neg_shuffle=sum(r["mean_delta_a_minus_b"]<0 for r in yearly_shuffle)
    neg_rms=sum(r["mean_delta_a_minus_b"]<0 for r in yearly_rms)
    if pooled_shuffle["ci95_high"]<0 and pooled_rms["ci95_high"]<0 and neg_shuffle>=2 and neg_rms>=2:
        return "HORSE_SPECIFIC_DISAGREEMENT_ADDS_VALUE"
    if pooled_shuffle["mean_delta_a_minus_b"]<0 and pooled_rms["mean_delta_a_minus_b"]<0:
        return "HORSE_SPECIFIC_DISAGREEMENT_DIRECTIONAL_NOT_CONCLUSIVE"
    return "NO_CLEAR_HORSE_SPECIFIC_SIGNAL"

def main():
    a=parse_args()
    c=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if c.get("contract")!="L2_PRICER_UNCERTAINTY_CAUSAL_AUDIT_V5": raise SystemExit("wrong contract")
    cp=c["cost_policy"]
    if cp["github_standard_cpu_only"] is not True or cp["paid_runner"] or cp["gpu"] or cp["paid_cloud_compute"] or cp["paid_artifact_or_cache"]:
        raise SystemExit("cost guard drift")
    if c["data_policy"]["locked_years"]!=[2026]: raise SystemExit("2026 lock drift")
    if abs(c["frozen_parameters"]["sigma"]-SIGMA0)>1e-12 or abs(c["frozen_parameters"]["lambda"]-LAMBDA)>1e-12:
        raise SystemExit("frozen parameter drift")

    manifest,df=load_dataset(a.dataset_dir)
    df["year"]=df["year"].astype(int); df["race_id"]=df["race_id"].astype(str)
    for col in ("mean_probability","rank_std_pct","probability_std","rank_range_pct"):
        df[col]=pd.to_numeric(df[col],errors="coerce").fillna(0.0)
    if 2026 in set(df["year"]): raise SystemExit("2026 sealed")

    workers=a.workers if a.workers>0 else max(1,min(4,os.cpu_count() or 2))
    started=time.perf_counter()
    outcomes=load_outcomes(df,a.backfill_root)
    races=build_races(df,outcomes,set(TEST_YEARS))
    print(f"CAUSAL_AUDIT_SETUP races={len(races)} draws={DRAWS} shuffles={SHUFFLES} workers={workers}",flush=True)
    results=evaluate_all(races,workers)
    print("CAUSAL_AUDIT_RACES_DONE",flush=True)

    metrics=metrics_rows(results)
    pairs=pairwise_rows(results)
    folds=fold_summary(metrics)
    us=uncertainty_summary(results)
    v=verdict(pairs)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"nll-metrics.csv",metrics)
    write_csv(out/"paired-bootstrap.csv",pairs)
    write_csv(out/"fold-summary.csv",folds)
    write_csv(out/"uncertainty-summary.csv",us)
    summary={
        "contract":"L2_PRICER_UNCERTAINTY_CAUSAL_AUDIT_V5_RESULT",
        "frozen_parameters":{"sigma":SIGMA0,"lambda":LAMBDA},
        "models":list(MODELS),"draws":DRAWS,"shuffle_replicates":SHUFFLES,
        "bootstrap_reps":BOOT_REPS,"parallel_workers":workers,
        "2026_locked":True,"market_used":False,"roi_optimized":False,
        "fold_summary":folds,"paired_bootstrap":pairs,"uncertainty_summary":us,
        "verdict":v,"elapsed_seconds":time.perf_counter()-started,"promotion":False
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 PRICER Uncertainty Causal Audit V5\n\n"
        "Frozen sigma=0.40/lambda=0.50. The audit keeps the same race-level uncertainty distribution while destroying or neutralizing "
        "its horse assignment: race-mean, race-RMS matched, eight within-race shuffles, and reversed assignment. All latent controls "
        "reuse antithetic common random numbers. Paired race-level bootstrap is reported by year and pooled. No odds/ROI/ticket/staking. 2026 sealed.\n",
        encoding="utf-8"
    )
    print("===== FOLD SUMMARY =====")
    print((out/"fold-summary.csv").read_text())
    print("===== PAIRED BOOTSTRAP =====")
    print((out/"paired-bootstrap.csv").read_text())
    print("VERDICT="+v)
    print("L2_PRICER_UNCERTAINTY_CAUSAL_AUDIT_V5_READY")

if __name__=="__main__":
    main()
