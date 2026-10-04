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
FOLDS=((2023,(2022,)),(2024,(2022,2023)),(2025,(2022,2023,2024)))
BET_TYPES=("WIN","EXACTA","TRIO","TRIFECTA")
CAL_DRAWS=256
TEST_DRAWS=2048
BOOT_REPS=2500

TEMPS=tuple(round(x,2) for x in np.arange(0.85,1.51,0.05))
HOMO_SIGMAS=tuple(round(x,2) for x in np.arange(0.10,0.91,0.10))
HET_SIGMAS=(0.20,0.30,0.40,0.50,0.60)
HET_LAMBDAS=(0.25,0.50,0.75,1.00,1.25)
LOCKED_SIGMA=0.40
LOCKED_LAMBDA=0.50

def parse_args():
    p=argparse.ArgumentParser(description="L2 PRICER V4 temperature-vs-latent audit")
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--workers",type=int,default=0)
    return p.parse_args()

def clip_prob(x):
    return min(max(float(x),EPS),1.0)

def seed_for(rid,phase):
    raw=f"{rid}|NORMAL_SHARED|{phase}|20261004".encode()
    return int.from_bytes(hashlib.blake2b(raw,digest_size=8).digest(),"little") & 0x7fffffff

def shared_z(race,draws,phase):
    rng=np.random.default_rng(seed_for(race["race_id"],phase))
    half=(draws+1)//2
    z=rng.standard_normal((half,len(race["base"])),dtype=np.float64)
    return np.concatenate((z,-z),axis=0)[:draws]

def temperature_rows(race,T):
    p=np.clip(np.asarray(race["base"],dtype=np.float64),EPS,None)
    w=np.power(p,1.0/float(T))
    w/=w.sum()
    return w.reshape(1,-1)

def normal_rows(race,z,sigma,lam):
    scale=float(sigma)*(1.0+float(lam)*np.asarray(race["uncertainty"],dtype=np.float64))
    logw=np.log(np.clip(race["base"],EPS,None))[None,:]+z*scale[None,:]
    logw-=np.max(logw,axis=1,keepdims=True)
    w=np.exp(logw); w/=np.sum(w,axis=1,keepdims=True)
    return w

def loss_from_rows(race,rows):
    out={}
    for bet in BET_TYPES:
        p=event_prob_from_rows(rows,race["winning_idx"].get(bet),bet)
        out[bet]=None if p is None else -math.log(clip_prob(p))
    return out

def eval_cfg_races(cfg,races,draws,phase,per_race=False):
    sums=defaultdict(float); counts=defaultdict(int); details=[]
    for race in races:
        fam=cfg["family"]
        if fam=="BASE_PL":
            rows=np.asarray(race["base"],dtype=np.float64).reshape(1,-1)
        elif fam=="TEMPERATURE_PL":
            rows=temperature_rows(race,cfg["temperature"])
        else:
            z=shared_z(race,draws,phase)
            rows=normal_rows(race,z,cfg["sigma"],cfg["lambda"])
        loss=loss_from_rows(race,rows)
        for bet,v in loss.items():
            if v is not None:
                sums[bet]+=v; counts[bet]+=1
        if per_race:
            details.append({
                "race_id":race["race_id"],"race_date":race["race_date"],"year":race["year"],
                **{f"{b.lower()}_nll":loss[b] for b in BET_TYPES},
                **{f"subset_{k}":bool(v) for k,v in race["subsets"].items()},
            })
    means={b:(sums[b]/counts[b] if counts[b] else None) for b in BET_TYPES}
    return {"cfg":cfg,"means":means,"counts":dict(counts),"details":details}

def cal_races(all_races,years):
    q=sorted([r for r in all_races if r["year"] in set(years)],key=lambda r:(r["race_date"],r["race_id"]))
    cut=max(1,int(len(q)*0.80))
    return q[cut:]

def cfg_grids():
    temp=[{"family":"TEMPERATURE_PL","temperature":T,"sigma":None,"lambda":None} for T in TEMPS]
    homo=[{"family":"NORMAL_HOMO","temperature":None,"sigma":s,"lambda":0.0} for s in HOMO_SIGMAS]
    hetero=[{"family":"NORMAL_HETERO_PRIOR","temperature":None,"sigma":s,"lambda":l} for s in HET_SIGMAS for l in HET_LAMBDAS]
    return {"TEMPERATURE_PL":temp,"NORMAL_HOMO":homo,"NORMAL_HETERO_PRIOR":hetero}

def cfg_sort_key(c):
    return (c.get("temperature") or 0.0,c.get("sigma") or 0.0,c.get("lambda") or 0.0)

def _cal_worker(args):
    cfg,races=args
    return eval_cfg_races(cfg,races,CAL_DRAWS,"CAL",False)

def select_cfgs(cal,workers):
    selected={}; grid=[]
    for fam,configs in cfg_grids().items():
        if workers>1:
            ctx=multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as ex:
                results=list(ex.map(_cal_worker,[(c,cal) for c in configs],chunksize=1))
        else:
            results=[_cal_worker((c,cal)) for c in configs]
        results.sort(key=lambda r:(r["means"]["TRIFECTA"],cfg_sort_key(r["cfg"])))
        best=results[0]
        selected[fam]=best["cfg"]
        for r in results:
            c=r["cfg"]
            grid.append({
                "family":fam,"temperature":c.get("temperature"),"sigma":c.get("sigma"),"lambda":c.get("lambda"),
                "cal_trifecta_nll":r["means"]["TRIFECTA"],"cal_trio_nll":r["means"]["TRIO"],
                "cal_exacta_nll":r["means"]["EXACTA"],"cal_win_nll":r["means"]["WIN"],
                "selected":int(c==best["cfg"])
            })
    return selected,grid

def aggregate_details(test_year,name,details,base_details,cfg):
    bmap={r["race_id"]:r for r in base_details}
    rows=[]
    subsets=("ALL","TAIL11_TRUE","KING1_MISS_TOP3_TRUE","NEITHER_STRESS")
    for subset in subsets:
        for bet in BET_TYPES:
            vals=[]; bases=[]
            key=f"{bet.lower()}_nll"
            for r in details:
                active=True if subset=="ALL" else bool(r[f"subset_{subset}"])
                if not active or r[key] is None: continue
                vals.append(float(r[key])); bases.append(float(bmap[r["race_id"]][key]))
            if not vals: continue
            v=float(np.mean(vals)); b=float(np.mean(bases))
            rows.append({
                "test_year":test_year,"model":name,"subset":subset,"bet_type":bet,"races":len(vals),
                "nll":v,"base_pl_nll":b,"delta_vs_base":v-b,
                "temperature":cfg.get("temperature"),"sigma":cfg.get("sigma"),"lambda":cfg.get("lambda")
            })
    return rows

def paired_bootstrap(year,a_name,a_details,b_name,b_details):
    amap={r["race_id"]:r for r in a_details}; bmap={r["race_id"]:r for r in b_details}
    ids=sorted(set(amap)&set(bmap))
    d=np.asarray([float(amap[i]["trifecta_nll"])-float(bmap[i]["trifecta_nll"]) for i in ids],dtype=np.float64)
    rng=np.random.default_rng(20261004+year+sum(ord(x) for x in a_name+b_name))
    means=np.empty(BOOT_REPS,dtype=np.float64)
    n=len(d)
    for start in range(0,BOOT_REPS,250):
        k=min(250,BOOT_REPS-start)
        idx=rng.integers(0,n,size=(k,n))
        means[start:start+k]=d[idx].mean(axis=1)
    lo,hi=np.quantile(means,[0.025,0.975])
    return {
        "test_year":year,"model_a":a_name,"model_b":b_name,"metric":"TRIFECTA_NLL",
        "mean_delta_a_minus_b":float(d.mean()),"ci95_low":float(lo),"ci95_high":float(hi),
        "a_better_race_pct":float(100*np.mean(d<0)),"races":n,"bootstrap_reps":BOOT_REPS
    }

def main():
    a=parse_args()
    c=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if c.get("contract")!="L2_PRICER_TEMPERATURE_AUDIT_V4": raise SystemExit("wrong contract")
    cp=c["cost_policy"]
    if cp["github_standard_cpu_only"] is not True or cp["paid_runner"] or cp["gpu"] or cp["paid_cloud_compute"] or cp["paid_artifact_or_cache"]:
        raise SystemExit("cost guard drift")
    if c["data_policy"]["locked_years"]!=[2026]: raise SystemExit("2026 lock drift")

    manifest,df=load_dataset(a.dataset_dir)
    df["year"]=df["year"].astype(int); df["race_id"]=df["race_id"].astype(str)
    for col in ("mean_probability","rank_std_pct","probability_std","rank_range_pct"):
        df[col]=pd.to_numeric(df[col],errors="coerce").fillna(0.0)
    if 2026 in set(df["year"]): raise SystemExit("2026 sealed")

    workers=a.workers if a.workers>0 else max(1,min(4,os.cpu_count() or 2))
    started=time.perf_counter()
    outcomes=load_outcomes(df,a.backfill_root)
    races=build_races(df,outcomes,{2022,2023,2024,2025})

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    grid_rows=[]; selected_rows=[]; metric_rows=[]; pair_rows=[]; fold_rows=[]

    for test_year,train_years in FOLDS:
        cal=cal_races(races,train_years)
        test=[r for r in races if r["year"]==test_year]
        print(f"TEMP_AUDIT_FOLD_START year={test_year} cal={len(cal)} test={len(test)} workers={workers}",flush=True)

        selected,grid=select_cfgs(cal,workers)
        for z in grid: z["test_year"]=test_year; grid_rows.append(z)

        locked={"family":"NORMAL_HETERO_V3_LOCKED","temperature":None,"sigma":LOCKED_SIGMA,"lambda":LOCKED_LAMBDA}
        base={"family":"BASE_PL","temperature":None,"sigma":0.0,"lambda":0.0}
        cfgs={
            "BASE_PL":base,
            "TEMPERATURE_PL":selected["TEMPERATURE_PL"],
            "NORMAL_HOMO":selected["NORMAL_HOMO"],
            "NORMAL_HETERO_PRIOR":selected["NORMAL_HETERO_PRIOR"],
            "NORMAL_HETERO_V3_LOCKED":locked,
        }
        for name,cfg in cfgs.items():
            selected_rows.append({"test_year":test_year,"model":name,**cfg})
            print("TEMP_AUDIT_SELECTED "+json.dumps({"test_year":test_year,"model":name,**cfg},separators=(",",":")),flush=True)

        results={}
        base_res=eval_cfg_races(base,test,1,"TEST",True)
        results["BASE_PL"]=base_res
        nonbase=[(n,cfgs[n]) for n in ("TEMPERATURE_PL","NORMAL_HOMO","NORMAL_HETERO_PRIOR","NORMAL_HETERO_V3_LOCKED")]
        if workers>1:
            ctx=multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=min(workers,4),mp_context=ctx) as ex:
                futs={n:ex.submit(eval_cfg_races,cfg,test,TEST_DRAWS,"TEST",True) for n,cfg in nonbase}
                for n,f in futs.items(): results[n]=f.result()
        else:
            for n,cfg in nonbase: results[n]=eval_cfg_races(cfg,test,TEST_DRAWS,"TEST",True)

        for name,res in results.items():
            metric_rows.extend(aggregate_details(test_year,name,res["details"],base_res["details"],cfgs[name]))

        for aa,bb in (
            ("NORMAL_HETERO_PRIOR","TEMPERATURE_PL"),
            ("NORMAL_HETERO_PRIOR","NORMAL_HOMO"),
            ("NORMAL_HETERO_V3_LOCKED","TEMPERATURE_PL"),
            ("NORMAL_HETERO_V3_LOCKED","NORMAL_HOMO"),
        ):
            pair_rows.append(paired_bootstrap(test_year,aa,results[aa]["details"],bb,results[bb]["details"]))

        row={"test_year":test_year}
        for name in cfgs:
            rr=[x for x in metric_rows if x["test_year"]==test_year and x["model"]==name and x["subset"]=="ALL" and x["bet_type"]=="TRIFECTA"][0]
            row[f"{name.lower()}_trifecta_nll"]=rr["nll"]
            row[f"{name.lower()}_delta_vs_base"]=rr["delta_vs_base"]
        fold_rows.append(row)
        print("TEMP_AUDIT_FOLD_DONE "+json.dumps(row,separators=(",",":")),flush=True)

    write_csv(out/"calibration-grid.csv",grid_rows)
    write_csv(out/"selected-parameters.csv",selected_rows)
    write_csv(out/"nll-metrics.csv",metric_rows)
    write_csv(out/"paired-bootstrap.csv",pair_rows)
    write_csv(out/"fold-summary.csv",fold_rows)

    # Cross-year paired verdict uses all fold-level race differences through the yearly bootstrap CIs plus consistency.
    hetero_vs_temp=[r for r in pair_rows if r["model_a"]=="NORMAL_HETERO_V3_LOCKED" and r["model_b"]=="TEMPERATURE_PL"]
    wins=sum(r["mean_delta_a_minus_b"]<0 for r in hetero_vs_temp)
    sig=sum(r["ci95_high"]<0 for r in hetero_vs_temp)
    verdict=(
        "HETERO_ADDS_STABLE_VALUE" if wins==3 and sig>=2 else
        "HETERO_DIRECTIONAL_NOT_CONCLUSIVE" if wins>=2 else
        "TEMPERATURE_EXPLAINS_MOST_OR_ALL"
    )
    summary={
        "contract":"L2_PRICER_TEMPERATURE_AUDIT_V4_RESULT",
        "question":"Does horse-specific uncertainty beat simple temperature flattening?",
        "calibration_draws":CAL_DRAWS,"test_draws":TEST_DRAWS,"bootstrap_reps":BOOT_REPS,
        "parallel_workers":workers,"common_random_numbers":True,"antithetic":True,
        "2026_locked":True,"market_used":False,"roi_optimized":False,
        "fold_summary":fold_rows,"paired_bootstrap":pair_rows,"verdict":verdict,
        "elapsed_seconds":time.perf_counter()-started,"promotion":False
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 PRICER Temperature Audit V4\n\n"
        "Direct walk-forward audit of raw PL, prior-only temperature-scaled PL, homoscedastic Normal latent PL, "
        "prior-selected heteroscedastic Normal latent PL, and the V3-locked heteroscedastic setting sigma=0.40/lambda=0.50. "
        "Normal models share antithetic random draws race-by-race. Paired race-level bootstrap intervals test whether heteroscedasticity "
        "adds value beyond simple flattening. No odds/ROI/ticket selection/staking. 2026 sealed.\n",
        encoding="utf-8"
    )
    print("===== FOLD SUMMARY =====")
    print((out/"fold-summary.csv").read_text())
    print("===== PAIRED BOOTSTRAP =====")
    print((out/"paired-bootstrap.csv").read_text())
    print("VERDICT="+verdict)
    print("L2_PRICER_TEMPERATURE_AUDIT_V4_READY")

if __name__=="__main__":
    main()
