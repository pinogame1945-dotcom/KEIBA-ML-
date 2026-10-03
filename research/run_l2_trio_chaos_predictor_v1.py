#!/usr/bin/env python3
import argparse,csv,gc,gzip,json,math,os,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss

from build_l2_bet_kings_dataset_v1 import load_odds_day,iter_decoded_odds

YEARS=(2023,2024,2025)
FOLDS=((2024,(2023,)),(2025,(2023,2024)))
SEED=20261004
EPS=1e-12

STRUCTURE_FEATURES=(
    "field_size",
    "race_entropy","race_top1_probability","race_top2_probability_sum","race_top3_probability_sum",
    "race_top1_top2_gap","race_rank_std_mean","race_rank_std_max",
    "race_probability_std_mean","race_probability_std_max",
    "king1_mean_rank_pct","king1_rank_std_pct","king1_top1_vote_share",
    "king1_top3_support_share","king1_top6_support_share",
    "king1_mean_probability","king1_probability_std",
    "outs_outside_top5_positive_count","outs_outside_top5_weighted_sum",
    "outs_outside_top5_weighted_max","outs_outside_top5_support_max",
    "outs_outside_top5_v1_sum","outs_outside_top5_v1_max",
)
MARKET_FEATURES=(
    "trio_market_overround",
    "trio_market_entropy_norm",
    "trio_market_hhi",
    "trio_market_effective_fraction",
    "trio_market_top1_mass",
    "trio_market_top3_mass",
    "trio_market_top10_mass",
    "trio_market_q90",
    "trio_market_q50",
    "trio_market_q10",
)

def parse_args():
    p=argparse.ArgumentParser(description="Strict walk-forward predictor for V4 both-fail / CHAOS races.")
    p.add_argument("--contract",required=True)
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--ballots-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--race-dates",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--v4-diagnostics",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_year_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1);out[int(y)]=p
    if set(out)!=set(YEARS):
        raise SystemExit(f"year paths mismatch: {sorted(out)}")
    return out

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def finite(x,default=0.0):
    try:
        v=float(x)
        return v if math.isfinite(v) else default
    except (TypeError,ValueError):
        return default

def load_dates(path):
    z=pd.read_csv(path,dtype={"race_id":str,"race_date":str})
    z["year"]=pd.to_numeric(z["year"],errors="raise").astype(int)
    z=z[z["year"].isin(YEARS)].copy()
    if 2026 in set(z["year"]): raise SystemExit("2026 sealed")
    if z.duplicated(["year","race_id"]).any(): raise SystemExit("duplicate race dates")
    return {(int(r.year),str(r.race_id)):str(r.race_date)[:10] for r in z.itertuples(index=False)}

def load_ballots(paths):
    stats={}
    race_candidates=defaultdict(set)
    for year,path in sorted(paths.items()):
        temp=defaultdict(lambda:defaultdict(lambda:{"weighted":0,"support":0,"v1":0,"v2":0,"v3":0}))
        with gzip.open(path,"rt",encoding="utf-8",newline="") as fh:
            for row in csv.DictReader(fh):
                rid=str(row["race_id"]);cand=str(row["candidate"])
                race_candidates[(year,rid)].add(cand)
                seen=set()
                for pos,w in ((1,3),(2,2),(3,1)):
                    hid=str(row.get(f"top{pos}_horse_id") or "")
                    if not hid: continue
                    if hid in seen: raise SystemExit(f"duplicate horse in ballot y={year} race={rid} candidate={cand}")
                    seen.add(hid)
                    d=temp[rid][hid];d["weighted"]+=w;d["support"]+=1;d[f"v{pos}"]+=1
        for rid,hmap in temp.items():
            if len(race_candidates[(year,rid)])!=13:
                raise SystemExit(f"expected 13 outsider candidates y={year} race={rid}")
            for hid,d in hmap.items():
                stats[(year,str(rid),str(hid))]=dict(d)
    return stats,race_candidates

def entropy_norm(vals):
    p=np.asarray(vals,dtype=np.float64)
    p=np.clip(p,EPS,None);p=p/p.sum()
    if len(p)<=1:return 0.0
    return float(-(p*np.log(p)).sum()/math.log(len(p)))

def load_l17_structure(paths,dates,ballot_stats,race_candidates):
    rows=[]
    for year,path in sorted(paths.items()):
        seen=0
        with open_text(path) as fh:
            for line in fh:
                if not line.strip():continue
                rec=json.loads(line)
                if rec.get("contract")!="L17_SEVEN_KING_FULLFIELD_OUTPUT_V1":
                    raise SystemExit(f"bad L1.7 contract year={year}")
                rid=str(rec.get("race_id") or "")
                key=(year,rid)
                if key not in dates: raise SystemExit(f"missing race date y={year} race={rid}")
                if len(race_candidates.get(key,set()))!=13:
                    raise SystemExit(f"missing outsider ballot coverage y={year} race={rid}")
                horses=sorted(rec.get("horses") or [],key=lambda h:(int(h["consensus_rank"]),str(h["horse_id"])))
                n=int(rec.get("field_size") or 0)
                if n!=len(horses) or n<3: raise SystemExit(f"field drift y={year} race={rid}")
                probs=np.asarray([finite(h.get("mean_probability")) for h in horses],dtype=float)
                probs=np.clip(probs,0,None);probs=(probs/probs.sum()) if probs.sum()>0 else np.full(n,1.0/n)
                rankstd=np.asarray([finite(h.get("rank_std")) for h in horses],dtype=float)
                probstd=np.asarray([finite(h.get("probability_std")) for h in horses],dtype=float)
                k=horses[0]
                outside=[]
                for h in horses:
                    if int(h["consensus_rank"])<=5:continue
                    d=ballot_stats.get((year,rid,str(h["horse_id"])),{"weighted":0,"support":0,"v1":0,"v2":0,"v3":0})
                    if d["weighted"]>0:outside.append(d)
                row={
                    "year":year,"race_id":rid,"race_date":dates[key],
                    "field_size":float(n),
                    "race_entropy":entropy_norm(probs),
                    "race_top1_probability":float(probs[0]),
                    "race_top2_probability_sum":float(probs[:2].sum()),
                    "race_top3_probability_sum":float(probs[:3].sum()),
                    "race_top1_top2_gap":float(probs[0]-probs[1]) if n>1 else float(probs[0]),
                    "race_rank_std_mean":float(rankstd.mean()),
                    "race_rank_std_max":float(rankstd.max()),
                    "race_probability_std_mean":float(probstd.mean()),
                    "race_probability_std_max":float(probstd.max()),
                    "king1_mean_rank_pct":finite(k.get("mean_rank"))/n,
                    "king1_rank_std_pct":finite(k.get("rank_std"))/n,
                    "king1_top1_vote_share":finite(k.get("top1_votes"))/7.0,
                    "king1_top3_support_share":finite(k.get("top3_support"))/7.0,
                    "king1_top6_support_share":finite(k.get("top6_support"))/7.0,
                    "king1_mean_probability":finite(k.get("mean_probability")),
                    "king1_probability_std":finite(k.get("probability_std")),
                    "outs_outside_top5_positive_count":float(len(outside)),
                    "outs_outside_top5_weighted_sum":float(sum(d["weighted"] for d in outside)),
                    "outs_outside_top5_weighted_max":float(max((d["weighted"] for d in outside),default=0)),
                    "outs_outside_top5_support_max":float(max((d["support"] for d in outside),default=0)),
                    "outs_outside_top5_v1_sum":float(sum(d["v1"] for d in outside)),
                    "outs_outside_top5_v1_max":float(max((d["v1"] for d in outside),default=0)),
                }
                rows.append(row);seen+=1
        if seen!=3456: raise SystemExit(f"L1.7 race coverage regression year={year}: {seen}")
    z=pd.DataFrame(rows)
    if len(z)!=3456*3: raise SystemExit(f"structure race total drift {len(z)}")
    return z

def build_market_shape(races,backfill_root):
    root=Path(backfill_root)/"data"/"odds"/"daily"
    rows=[]
    for date,g in races.groupby("race_date",sort=True):
        wanted=set(g["race_id"].astype(str))
        recs=load_odds_day(root/f"{date}.jsonl.gz",wanted)
        for r in g.itertuples(index=False):
            rid=str(r.race_id);n=int(r.field_size);expected=math.comb(n,3)
            rec=recs.get(rid)
            if rec is None: raise SystemExit(f"missing odds record race={rid} date={date}")
            prices={}
            for bet,nums,price in iter_decoded_odds(rec):
                if bet=="TRIO":prices[tuple(nums)]=float(price)
            if len(prices)!=expected:
                raise SystemExit(f"incomplete TRIO market race={rid} got={len(prices)} expected={expected}")
            raw=1.0/np.asarray(list(prices.values()),dtype=np.float64)
            over=float(raw.sum());q=raw/over
            qs=np.sort(q)
            hhi=float(np.square(q).sum())
            desc=qs[::-1]
            rows.append({
                "year":int(r.year),"race_id":rid,
                "trio_market_overround":over,
                "trio_market_entropy_norm":float(-(q*np.log(np.clip(q,EPS,None))).sum()/math.log(len(q))),
                "trio_market_hhi":hhi,
                "trio_market_effective_fraction":float((1.0/hhi)/len(q)),
                "trio_market_top1_mass":float(desc[:1].sum()),
                "trio_market_top3_mass":float(desc[:3].sum()),
                "trio_market_top10_mass":float(desc[:10].sum()),
                "trio_market_q90":float(np.quantile(q,.90)),
                "trio_market_q50":float(np.quantile(q,.50)),
                "trio_market_q10":float(np.quantile(q,.10)),
            })
    return pd.DataFrame(rows)

def load_labels(path):
    rows=[]
    with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            y=int(r["year"])
            if y not in YEARS:continue
            kd=float(r["king_logloss_delta_vs_market"]);od=float(r["outsider_logloss_delta_vs_market"])
            rows.append({"year":y,"race_id":str(r["race_id"]),"chaos_label":int(kd>=0 and od>=0)})
    z=pd.DataFrame(rows)
    if z.duplicated(["year","race_id"]).any():raise SystemExit("duplicate V4 labels")
    return z

def sigmoid(x):
    return 1.0/(1.0+np.exp(-np.clip(np.asarray(x,dtype=float),-60,60)))

def choose_temp(y,raw):
    best=(1.0,float("inf"))
    for t in np.exp(np.linspace(math.log(.35),math.log(3.0),31)):
        p=sigmoid(raw/t)
        ll=log_loss(y,p,labels=[0,1])
        if ll<best[1]:best=(float(t),float(ll))
    return best

def chronological_split(df):
    q=df.sort_values(["race_date","race_id"],kind="stable").reset_index(drop=True)
    cut=max(1,int(len(q)*.8))
    return q.iloc[:cut].copy(),q.iloc[cut:].copy()

def model_params(seed,threads):
    return dict(
        objective="binary",n_estimators=180,learning_rate=.04,num_leaves=15,max_depth=5,
        min_child_samples=100,subsample=.9,colsample_bytree=.85,reg_lambda=4.0,reg_alpha=.25,
        random_state=seed,n_jobs=max(1,int(threads)),verbosity=-1,force_col_wise=True
    )

def fit_predict(name,features,train,test,threads,seed):
    fit,cal=chronological_split(train)
    if fit["chaos_label"].nunique()<2 or cal["chaos_label"].nunique()<2:
        raise SystemExit(f"degenerate calibration labels {name}")
    m0=lgb.LGBMClassifier(**model_params(seed,threads))
    m0.fit(fit[list(features)],fit["chaos_label"].astype(int))
    raw_cal=np.asarray(m0.booster_.predict(cal[list(features)],raw_score=True),dtype=float)
    temp,cal_ll=choose_temp(cal["chaos_label"].astype(int).to_numpy(),raw_cal)
    model=lgb.LGBMClassifier(**model_params(seed+1000,threads))
    model.fit(train[list(features)],train["chaos_label"].astype(int))
    raw=np.asarray(model.booster_.predict(test[list(features)],raw_score=True),dtype=float)
    p=sigmoid(raw/temp)
    gain=model.booster_.feature_importance(importance_type="gain")
    imp=sorted([{"feature":f,"gain":float(g)} for f,g in zip(features,gain)],key=lambda x:-x["gain"])
    return {"name":name,"p":p,"temp":temp,"cal_ll":cal_ll,"importance":imp}

def risk_rows(test,p,feature_set,test_year):
    y=test["chaos_label"].to_numpy(dtype=int);n=len(y);base=float(y.mean());order=np.argsort(-p)
    out=[]
    for frac in (.10,.20,.30):
        k=max(1,int(math.ceil(n*frac)));idx=order[:k]
        rate=float(y[idx].mean());recall=float(y[idx].sum()/max(1,y.sum()))
        out.append({
            "test_year":test_year,"feature_set":feature_set,"risk_fraction":frac,"races":k,
            "test_base_rate":base,"chaos_rate":rate,"lift":rate/base if base>0 else None,
            "chaos_recall":recall
        })
    return out

def metric_row(test,p,name,test_year,train_years,temp,cal_ll):
    y=test["chaos_label"].to_numpy(dtype=int)
    train_base=None
    return {
        "test_year":test_year,"feature_set":name,"train_years":"|".join(map(str,train_years)),
        "races":len(y),"chaos_races":int(y.sum()),"chaos_rate":float(y.mean()),
        "roc_auc":float(roc_auc_score(y,p)),
        "average_precision":float(average_precision_score(y,p)),
        "logloss":float(log_loss(y,p,labels=[0,1])),
        "brier":float(brier_score_loss(y,p)),
        "temperature":float(temp),"internal_cal_logloss":float(cal_ll),
    }

def main():
    t0=time.time();a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L2_TRIO_CHAOS_PREDICTOR_V1":raise SystemExit("wrong contract")
    if contract["cost_policy"]["github_standard_cpu_only"] is not True or contract["cost_policy"]["gpu"] is not False:
        raise SystemExit("cost guard drift")
    paths=parse_year_paths(a.l17_year);ballot_paths=parse_year_paths(a.ballots_year)
    dates=load_dates(a.race_dates)
    stats,candidates=load_ballots(ballot_paths)
    structure=load_l17_structure(paths,dates,stats,candidates)
    market=build_market_shape(structure[["year","race_id","race_date","field_size"]],a.backfill_root)
    labels=load_labels(a.v4_diagnostics)
    data=structure.merge(market,on=["year","race_id"],how="inner",validate="one_to_one").merge(
        labels,on=["year","race_id"],how="inner",validate="one_to_one"
    )
    expected={2023:3443,2024:3438,2025:3444}
    got=data.groupby("year").size().to_dict()
    if got!=expected:raise SystemExit(f"label/market coverage drift got={got} expected={expected}")
    if 2026 in set(data["year"]):raise SystemExit("2026 sealed")

    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    cpu=max(1,os.cpu_count() or 1);threads=max(1,cpu//2)
    feature_sets={
        "STRUCTURE_ONLY":list(STRUCTURE_FEATURES),
        "STRUCTURE_PLUS_TRIO_MARKET_SHAPE":list(STRUCTURE_FEATURES)+list(MARKET_FEATURES),
    }
    fold_metrics=[];risk=[];imps=[];pred_rows=[];pooled={k:[] for k in feature_sets}
    for fi,(test_year,train_years) in enumerate(FOLDS,1):
        train=data[data["year"].isin(train_years)].copy().reset_index(drop=True)
        test=data[data["year"]==test_year].copy().sort_values(["race_date","race_id"]).reset_index(drop=True)
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs={
                name:ex.submit(fit_predict,name,features,train,test,threads,SEED+fi*100+j*10)
                for j,(name,features) in enumerate(feature_sets.items())
            }
            results={name:f.result() for name,f in futs.items()}
        for name,res in results.items():
            p=res["p"];row=metric_row(test,p,name,test_year,train_years,res["temp"],res["cal_ll"])
            prior=float(train["chaos_label"].mean())
            base_p=np.full(len(test),prior,dtype=float)
            y=test["chaos_label"].to_numpy(dtype=int)
            row["prior_rate"]=prior
            row["baseline_logloss"]=float(log_loss(y,base_p,labels=[0,1]))
            row["baseline_brier"]=float(brier_score_loss(y,base_p))
            row["logloss_delta_vs_prior"]=row["logloss"]-row["baseline_logloss"]
            row["brier_delta_vs_prior"]=row["brier"]-row["baseline_brier"]
            fold_metrics.append(row)
            risk.extend(risk_rows(test,p,name,test_year))
            for rank,imp in enumerate(res["importance"][:20],1):
                imps.append({"test_year":test_year,"feature_set":name,"rank":rank,**imp})
            for i,r in test.iterrows():
                pred_rows.append({
                    "test_year":test_year,"year":int(r["year"]),"race_id":str(r["race_id"]),"race_date":str(r["race_date"]),
                    "chaos_label":int(r["chaos_label"]),"feature_set":name,"p_chaos":float(p[i])
                })
            pooled[name].append(pd.DataFrame({"y":y,"p":p}))
        print("CHAOS_PREDICTOR_FOLD_DONE "+json.dumps({
            "test_year":test_year,
            **{name:{
                "auc":next(r["roc_auc"] for r in fold_metrics if r["test_year"]==test_year and r["feature_set"]==name),
                "top10_lift":next(r["lift"] for r in risk if r["test_year"]==test_year and r["feature_set"]==name and abs(r["risk_fraction"]-.10)<1e-9)
            } for name in feature_sets}
        },separators=(",",":")),flush=True)
        gc.collect()

    pooled_rows=[]
    for name,parts in pooled.items():
        z=pd.concat(parts,ignore_index=True);y=z["y"].to_numpy(dtype=int);p=z["p"].to_numpy(dtype=float)
        pooled_rows.append({
            "feature_set":name,"races":len(y),"chaos_rate":float(y.mean()),
            "roc_auc":float(roc_auc_score(y,p)),"average_precision":float(average_precision_score(y,p)),
            "logloss":float(log_loss(y,p,labels=[0,1])),"brier":float(brier_score_loss(y,p))
        })
        order=np.argsort(-p)
        for frac in (.10,.20,.30):
            k=max(1,int(math.ceil(len(y)*frac)));idx=order[:k];base=float(y.mean())
            pooled_rows[-1][f"top{int(frac*100)}_chaos_rate"]=float(y[idx].mean())
            pooled_rows[-1][f"top{int(frac*100)}_lift"]=float(y[idx].mean()/base)
            pooled_rows[-1][f"top{int(frac*100)}_recall"]=float(y[idx].sum()/max(1,y.sum()))

    pd.DataFrame(fold_metrics).to_csv(out/"fold-metrics.csv",index=False)
    pd.DataFrame(risk).to_csv(out/"risk-concentration.csv",index=False)
    pd.DataFrame(imps).to_csv(out/"feature-importance.csv",index=False)
    pd.DataFrame(pred_rows).to_csv(out/"predictions.csv.gz",index=False,compression="gzip")
    pd.DataFrame(pooled_rows).to_csv(out/"pooled-metrics.csv",index=False)
    summary={
        "contract":"L2_TRIO_CHAOS_PREDICTOR_V1_RESULT",
        "strict_walk_forward":{"2024":[2023],"2025":[2023,2024]},
        "feature_sets":list(feature_sets),
        "fold_metrics":fold_metrics,"pooled_metrics":pooled_rows,
        "cpu_count":cpu,"parallel_models":2,"threads_per_model":threads,
        "historical_market_proxy":"FINAL_TRIO_ODDS",
        "operational_market_requirement":"LATEST_TIMESTAMPED_PRE_RACE_TRIO_ODDS",
        "forbidden_features_used":False,"2026_locked":True,
        "runtime_seconds_total":time.time()-t0,
        "promotion":False
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 TRIO CHAOS Predictor V1\n\n"
        "Strict walk-forward test of whether V4 both-fail races can be identified before outcomes. "
        "2024 is predicted from 2023 only; 2025 from 2023-2024. No winning-trio identity, result, payout, ROI, "
        "V4 ex-post probabilities/deltas, or 2026 data are features. Two feature sets run in parallel: "
        "L1.7+safe Outsider race structure, and the same plus full-race TRIO market-shape statistics.\n",
        encoding="utf-8"
    )
    print("L2_TRIO_CHAOS_PREDICTOR_V1_READY")
    print(json.dumps({"pooled":pooled_rows,"runtime_seconds":time.time()-t0},separators=(",",":")))

if __name__=="__main__":
    main()
