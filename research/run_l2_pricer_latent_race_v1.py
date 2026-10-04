#!/usr/bin/env python3
import argparse,csv,gzip,hashlib,json,math,multiprocessing,os,time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import load_day,payout_map
from run_l2_core_v1 import load_dataset,write_csv

FOLDS=((2023,(2022,)),(2024,(2022,2023)),(2025,(2022,2023,2024)))
BET_TYPES=("WIN","EXACTA","TRIO","TRIFECTA")
EPS=1e-15
CAL_DRAWS=160
TEST_DRAWS=1024

HOMO_SIGMAS=(0.20,0.40,0.60,0.80,1.00)
HET_SIGMAS=(0.20,0.40,0.60,0.80)
HET_LAMBDAS=(0.50,1.00,1.50)
T_SIGMAS=(0.30,0.50,0.70)
T_LAMBDAS=(0.75,1.50)
T_DFS=(3.0,5.0)

G_RACES=None
G_DRAWS=None


def parse_args():
    p=argparse.ArgumentParser(description="Fast latent-race pricer V1: deterministic PL vs latent-strength PL mixtures.")
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--workers",type=int,default=0)
    return p.parse_args()


def clip_prob(x):
    return min(max(float(x),EPS),1.0)


def seed_for(race_id,family,phase):
    raw=f"{race_id}|{family}|{phase}|20261004".encode()
    return int.from_bytes(hashlib.blake2b(raw,digest_size=8).digest(),"little") & 0x7FFFFFFF


def normalize(v):
    x=np.clip(np.asarray(v,dtype=np.float64),EPS,None)
    return x/x.sum()


def uncertainty_vector(sub):
    rank=np.asarray(sub["rank_std_pct"],dtype=float)
    prob=np.asarray(sub["probability_std"],dtype=float)
    rr=np.asarray(sub["rank_range_pct"],dtype=float)
    def robust01(x):
        if len(x)<2: return np.zeros_like(x)
        lo=float(np.quantile(x,0.10)); hi=float(np.quantile(x,0.90))
        if hi<=lo+1e-12: return np.zeros_like(x)
        return np.clip((x-lo)/(hi-lo),0.0,1.0)
    return np.asarray((robust01(rank)+robust01(prob)+robust01(rr))/3.0,dtype=np.float64)


def ordered2_rows(p,a,b):
    den=1.0-p[:,a]
    return p[:,a]*p[:,b]/np.clip(den,EPS,None)


def ordered3_rows(p,a,b,c):
    den1=1.0-p[:,a]
    den2=1.0-p[:,a]-p[:,b]
    return p[:,a]*(p[:,b]/np.clip(den1,EPS,None))*(p[:,c]/np.clip(den2,EPS,None))


def event_prob_from_rows(p,winning_idx,bet):
    if not winning_idx:
        return None
    total=np.zeros(p.shape[0],dtype=np.float64)
    if bet=="WIN":
        for (a,) in winning_idx: total += p[:,a]
    elif bet=="EXACTA":
        for a,b in winning_idx: total += ordered2_rows(p,a,b)
    elif bet=="TRIFECTA":
        for a,b,c in winning_idx: total += ordered3_rows(p,a,b,c)
    elif bet=="TRIO":
        for a,b,c in winning_idx:
            total += (
                ordered3_rows(p,a,b,c)+ordered3_rows(p,a,c,b)+
                ordered3_rows(p,b,a,c)+ordered3_rows(p,b,c,a)+
                ordered3_rows(p,c,a,b)+ordered3_rows(p,c,b,a)
            )
    else:
        raise ValueError(bet)
    return float(np.mean(np.clip(total,0.0,1.0)))


def draw_probability_rows(race,cfg,draws,phase):
    base=race["base"]
    n=len(base)
    family=cfg["family"]
    if family=="BASE_PL":
        return base.reshape(1,-1)
    rng=np.random.default_rng(seed_for(race["race_id"],family,phase))
    half=(draws+1)//2
    if family in ("NORMAL_HOMO","NORMAL_HETERO"):
        z=rng.standard_normal((half,n),dtype=np.float64)
    elif family=="STUDENTT_HETERO":
        df=float(cfg["df"])
        z=rng.standard_t(df,size=(half,n)).astype(np.float64)
        z*=math.sqrt((df-2.0)/df)
    else:
        raise ValueError(family)
    z=np.concatenate((z,-z),axis=0)[:draws]
    sigma=np.full(n,float(cfg["sigma"]),dtype=np.float64)
    if family in ("NORMAL_HETERO","STUDENTT_HETERO"):
        sigma*=1.0+float(cfg["lambda"])*race["uncertainty"]
    logw=np.log(np.clip(base,EPS,None))[None,:]+z*sigma[None,:]
    logw-=np.max(logw,axis=1,keepdims=True)
    w=np.exp(logw)
    w/=np.sum(w,axis=1,keepdims=True)
    return w


def eval_config_local(cfg,races,draws,phase,collect_subsets=False):
    sums=defaultdict(float); counts=defaultdict(int)
    subset_sums=defaultdict(float); subset_counts=defaultdict(int)
    for race in races:
        rows=draw_probability_rows(race,cfg,draws,phase)
        for bet in BET_TYPES:
            prob=event_prob_from_rows(rows,race["winning_idx"].get(bet),bet)
            if prob is None: continue
            nll=-math.log(clip_prob(prob))
            sums[bet]+=nll; counts[bet]+=1
            if collect_subsets:
                for subset,active in race["subsets"].items():
                    if active:
                        subset_sums[(subset,bet)]+=nll
                        subset_counts[(subset,bet)]+=1
    means={bet:(sums[bet]/counts[bet] if counts[bet] else None) for bet in BET_TYPES}
    objective=means["TRIFECTA"] if means["TRIFECTA"] is not None else float("inf")
    out={"config":cfg,"objective":objective,"means":means,"counts":dict(counts)}
    if collect_subsets:
        out["subset_means"]={f"{s}|{b}":subset_sums[(s,b)]/subset_counts[(s,b)] for (s,b) in subset_sums if subset_counts[(s,b)]}
        out["subset_counts"]={f"{s}|{b}":subset_counts[(s,b)] for (s,b) in subset_counts}
    return out


def _worker_eval(cfg):
    return eval_config_local(cfg,G_RACES,G_DRAWS,"CAL",False)


def parallel_eval(configs,races,draws,workers):
    global G_RACES,G_DRAWS
    G_RACES=races; G_DRAWS=draws
    if workers<=1 or len(configs)<=1:
        return [eval_config_local(c,races,draws,"CAL",False) for c in configs]
    ctx=multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as ex:
        return list(ex.map(_worker_eval,configs,chunksize=1))


def config_grid():
    return {
        "NORMAL_HOMO":[{"family":"NORMAL_HOMO","sigma":s,"lambda":0.0,"df":None} for s in HOMO_SIGMAS],
        "NORMAL_HETERO":[{"family":"NORMAL_HETERO","sigma":s,"lambda":l,"df":None} for s in HET_SIGMAS for l in HET_LAMBDAS],
        "STUDENTT_HETERO":[{"family":"STUDENTT_HETERO","sigma":s,"lambda":l,"df":df} for s in T_SIGMAS for l in T_LAMBDAS for df in T_DFS],
    }


def load_outcomes(df,backfill_root):
    bydate=defaultdict(set)
    for r in df[["race_date","race_id"]].drop_duplicates().itertuples(index=False):
        bydate[str(r.race_date)[:10]].add(str(r.race_id))
    root=Path(backfill_root)
    outcomes={}
    for di,date in enumerate(sorted(bydate),1):
        wanted=bydate[date]
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in wanted:
            pack=day.get(rid)
            if pack is None: raise SystemExit(f"missing race pack race={rid} date={date}")
            payouts,present=payout_map(pack)
            rec={bet:[] for bet in BET_TYPES}
            for (bet,nums),ret in payouts.items():
                if bet in rec and ret>0: rec[bet].append(tuple(int(x) for x in nums))
            outcomes[rid]=rec
        if di%80==0:
            print(f"LATENT_OUTCOME_PROGRESS dates={di}/{len(bydate)}",flush=True)
    return outcomes


def build_races(df,outcomes,years):
    races=[]
    z=df[df["year"].isin(years)].copy()
    for (year,rid),sub in z.groupby(["year","race_id"],sort=False):
        sub=sub.sort_values(["consensus_rank","horse_number"],kind="stable").reset_index(drop=True)
        nums=[int(x) for x in sub["horse_number"]]
        idx={n:i for i,n in enumerate(nums)}
        base=normalize(sub["mean_probability"].to_numpy(dtype=float))
        winning_idx={}
        for bet in BET_TYPES:
            vals=[]
            for ticket in outcomes.get(str(rid),{}).get(bet,[]):
                if all(n in idx for n in ticket): vals.append(tuple(idx[n] for n in ticket))
            winning_idx[bet]=vals
        rank_by_num={int(n):int(r) for n,r in zip(sub["horse_number"],sub["consensus_rank"])}
        trio_tickets=outcomes.get(str(rid),{}).get("TRIO",[])
        king1_num=int(sub.loc[sub["consensus_rank"]==1,"horse_number"].iloc[0])
        tail11=any(any(rank_by_num.get(n,99)>=11 for n in t) for t in trio_tickets)
        king1_miss=bool(trio_tickets) and all(king1_num not in t for t in trio_tickets)
        races.append({
            "year":int(year),"race_id":str(rid),"race_date":str(sub["race_date"].iloc[0])[:10],
            "base":base,"uncertainty":uncertainty_vector(sub),"winning_idx":winning_idx,
            "subsets":{"ALL":True,"TAIL11_TRUE":tail11,"KING1_MISS_TOP3_TRUE":king1_miss,"NEITHER_STRESS":(not tail11 and not king1_miss)},
        })
    return races


def chronological_calibration_races(races,train_years):
    q=[r for r in races if r["year"] in train_years]
    q=sorted(q,key=lambda r:(r["race_date"],r["race_id"]))
    cut=max(1,int(len(q)*0.8))
    return q[cut:]


def metrics_rows(test_year,family,result,base_result,selected_cfg):
    rows=[]
    for subset in ("ALL","TAIL11_TRUE","KING1_MISS_TOP3_TRUE","NEITHER_STRESS"):
        for bet in BET_TYPES:
            key=f"{subset}|{bet}"
            val=result.get("subset_means",{}).get(key)
            b=base_result.get("subset_means",{}).get(key)
            n=result.get("subset_counts",{}).get(key,0)
            if val is None or b is None: continue
            rows.append({
                "test_year":test_year,"family":family,"subset":subset,"bet_type":bet,"races":n,
                "nll":val,"base_pl_nll":b,"delta_vs_base":val-b,
                "sigma":selected_cfg.get("sigma"),"lambda":selected_cfg.get("lambda"),"df":selected_cfg.get("df"),
            })
    return rows


def main():
    a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L2_PRICER_LATENT_RACE_V1": raise SystemExit("wrong contract")
    cp=contract["cost_policy"]
    if cp["github_standard_cpu_only"] is not True or cp["gpu"] is not False or cp["paid_artifact_or_cache"] is not False:
        raise SystemExit("cost guard drift")
    if contract["data_policy"]["locked_years"]!=[2026]: raise SystemExit("2026 lock drift")

    manifest,df=load_dataset(a.dataset_dir)
    df["year"]=df["year"].astype(int); df["race_id"]=df["race_id"].astype(str)
    for c in ("mean_probability","rank_std_pct","probability_std","rank_range_pct"):
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
    if 2026 in set(df["year"]): raise SystemExit("2026 sealed")

    started=time.perf_counter()
    outcomes=load_outcomes(df,a.backfill_root)
    all_races=build_races(df,outcomes,{2022,2023,2024,2025})
    workers=a.workers if a.workers>0 else max(1,min(4,os.cpu_count() or 2))
    grids=config_grid()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    selected_rows=[]; grid_rows=[]; metric_rows=[]

    for test_year,train_years in FOLDS:
        cal=chronological_calibration_races(all_races,set(train_years))
        test=[r for r in all_races if r["year"]==test_year]
        print(f"LATENT_FOLD_START year={test_year} cal={len(cal)} test={len(test)} workers={workers}",flush=True)
        selected={}
        for family,configs in grids.items():
            results=parallel_eval(configs,cal,CAL_DRAWS,workers)
            results.sort(key=lambda r:(r["objective"],r["config"]["sigma"],r["config"].get("lambda") or 0.0,r["config"].get("df") or 0.0))
            best=results[0]
            selected[family]=best["config"]
            for rr in results:
                c=rr["config"]
                grid_rows.append({
                    "test_year":test_year,"family":family,"sigma":c.get("sigma"),"lambda":c.get("lambda"),"df":c.get("df"),
                    "cal_draws":CAL_DRAWS,"cal_trifecta_nll":rr["means"].get("TRIFECTA"),
                    "cal_exacta_nll":rr["means"].get("EXACTA"),"cal_trio_nll":rr["means"].get("TRIO"),
                    "selected":int(c==best["config"]),
                })
            selected_rows.append({"test_year":test_year,**best["config"],"cal_trifecta_nll":best["means"].get("TRIFECTA")})
            print("LATENT_SELECTED "+json.dumps({"test_year":test_year,**best["config"],"cal_trifecta_nll":best["means"].get("TRIFECTA")},separators=(",",":")),flush=True)

        base_cfg={"family":"BASE_PL","sigma":0.0,"lambda":0.0,"df":None}
        base_result=eval_config_local(base_cfg,test,1,"TEST",True)
        metric_rows.extend(metrics_rows(test_year,"BASE_PL",base_result,base_result,base_cfg))
        test_cfgs=[selected[f] for f in ("NORMAL_HOMO","NORMAL_HETERO","STUDENTT_HETERO")]
        if workers>1:
            ctx=multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=min(workers,3),mp_context=ctx) as ex:
                futs=[ex.submit(eval_config_local,c,test,TEST_DRAWS,"TEST",True) for c in test_cfgs]
                test_results=[f.result() for f in futs]
        else:
            test_results=[eval_config_local(c,test,TEST_DRAWS,"TEST",True) for c in test_cfgs]
        for cfg,res in zip(test_cfgs,test_results):
            metric_rows.extend(metrics_rows(test_year,cfg["family"],res,base_result,cfg))
        print(f"LATENT_FOLD_DONE year={test_year}",flush=True)

    write_csv(out/"selected-parameters.csv",selected_rows)
    write_csv(out/"calibration-grid.csv",grid_rows)
    write_csv(out/"nll-metrics.csv",metric_rows)
    mdf=pd.DataFrame(metric_rows)
    allrows=mdf[mdf["subset"]=="ALL"].copy()
    piv=[]
    for (year,fam),g in allrows.groupby(["test_year","family"],sort=False):
        row={"test_year":int(year),"family":fam}
        for r in g.itertuples(index=False):
            row[f"{str(r.bet_type).lower()}_nll"]=float(r.nll)
            row[f"{str(r.bet_type).lower()}_delta_vs_base"]=float(r.delta_vs_base)
        piv.append(row)
    write_csv(out/"fold-summary.csv",piv)
    summary={
        "contract":"L2_PRICER_LATENT_RACE_V1_RESULT",
        "architecture":"LATENT_LOG_STRENGTH_MIXTURE_OF_PL",
        "families":["BASE_PL","NORMAL_HOMO","NORMAL_HETERO","STUDENTT_HETERO"],
        "calibration_draws":CAL_DRAWS,"test_draws":TEST_DRAWS,"antithetic":True,
        "selection_metric":"prior-only chronological calibration TRIFECTA NLL",
        "parallel_workers":workers,"2026_locked":True,"market_used":False,"roi_optimized":False,
        "selected_parameters":selected_rows,"fold_summary":piv,
        "elapsed_seconds":time.perf_counter()-started,"promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 PRICER Latent Race V1\n\n"
        "Fast probability-quality A/B only. Frozen L1.7 mean probability is the deterministic PL baseline. "
        "Latent families inject antithetic Normal or Student-t uncertainty into race-level log strengths, average coherent PL ticket probabilities, "
        "and are tuned only on prior chronological calibration races. No odds, ROI, ticket selection, or staking. 2026 sealed.\n",
        encoding="utf-8",
    )
    print("===== FOLD SUMMARY =====")
    print((out/"fold-summary.csv").read_text())
    print("===== SELECTED =====")
    print((out/"selected-parameters.csv").read_text())
    print("L2_PRICER_LATENT_RACE_V1_READY")


if __name__=="__main__":
    main()
