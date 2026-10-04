#!/usr/bin/env python3
import argparse,csv,gzip,json,math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l2_core_v1 import load_dataset,write_csv
from build_l2_bet_kings_dataset_v1 import load_odds_day,final_odds_tuple,finite

FOLDS=((2023,(2022,)),(2024,(2022,2023)),(2025,(2022,2023,2024)))
MODELS=("MARKET_ONLY","L1_ONLY","MARKET_CRITIC")
TARGETS=("WIN","TOP3")
BOOT_REPS=2500
EPS=1e-12

def parse_args():
    p=argparse.ArgumentParser(description="L1 Market Critic V1")
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def clip01(x):
    return min(max(float(x),EPS),1.0-EPS)

def add_market(df,root):
    root=Path(root)
    rows=[]
    bydate=defaultdict(list)
    for rid,date in df[["race_id","race_date"]].drop_duplicates().itertuples(index=False):
        bydate[str(date)[:10]].append(str(rid))
    groups={str(rid):g for rid,g in df.groupby("race_id",sort=False)}
    skipped=[]
    for date,rids in sorted(bydate.items()):
        wanted=set(rids)
        oddsday=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in rids:
            sub=groups[rid]
            orec=oddsday.get(rid)
            if orec is None:
                skipped.append({"race_id":rid,"race_date":date,"reason":"odds_missing"}); continue
            raw=((orec.get("odds") or {}).get("1") or {})
            odds={}
            for key,val in raw.items():
                k=str(key)
                if not k.isascii() or not k.isdigit(): continue
                tup=final_odds_tuple(val)
                if not tup: continue
                odd=finite(tup[0])
                if odd is not None and odd>0: odds[int(k)]=float(odd)
            expected=set(int(x) for x in sub["horse_number"].dropna().astype(int))
            if set(odds)!=expected:
                skipped.append({"race_id":rid,"race_date":date,"reason":f"win_odds_incomplete:{len(odds)}/{len(expected)}"}); continue
            inv={n:1.0/o for n,o in odds.items()}
            overround=sum(inv.values())
            p={n:v/overround for n,v in inv.items()}
            vals=np.asarray(list(p.values()),dtype=float)
            ent=-float(np.sum(vals*np.log(np.clip(vals,1e-15,None))))/math.log(len(vals)) if len(vals)>1 else 0.0
            top=sorted(vals,reverse=True)
            uniq=sorted(set(odds.values()))
            rank={o:1+sum(1 for x in odds.values() if x<o) for o in uniq}
            top1=max(vals); top2=sorted(vals,reverse=True)[1] if len(vals)>1 else 0.0
            for idx,r in sub.iterrows():
                no=int(r["horse_number"])
                rows.append({
                    "idx":int(idx),
                    "final_win_odds":odds[no],
                    "log_final_win_odds":math.log(max(odds[no],1.000001)),
                    "market_pwin_raw":p[no],
                    "market_rank":int(rank[odds[no]]),
                    "market_rank_pct":int(rank[odds[no]])/len(vals),
                    "market_overround":overround,
                    "market_entropy":ent,
                    "market_top1_p":top1,
                    "market_top2_p":top2,
                    "market_top1_top2_gap":top1-top2,
                    "market_gap_to_top1":top1-p[no],
                })
    aux=pd.DataFrame(rows).set_index("idx") if rows else pd.DataFrame()
    keep=df.index.intersection(aux.index)
    z=df.loc[keep].copy().join(aux.loc[keep])
    # normalize L1 mean probability within each race so probability-gap features are comparable to market.
    z["l1_pwin_norm"]=0.0
    for rid,idxs in z.groupby("race_id").groups.items():
        ii=list(idxs)
        v=np.clip(pd.to_numeric(z.loc[ii,"mean_probability"],errors="coerce").fillna(0.0).to_numpy(dtype=float),0,None)
        s=v.sum(); v=v/s if s>0 else np.full(len(v),1.0/len(v))
        z.loc[ii,"l1_pwin_norm"]=v
    z["l1_market_prob_gap"]=z["l1_pwin_norm"]-z["market_pwin_raw"]
    z["l1_market_rank_gap_pct"]=z["market_rank_pct"]-z["consensus_rank_pct"]
    return z,skipped

def model_params(seed):
    return dict(
        objective="binary",n_estimators=320,learning_rate=0.035,num_leaves=31,
        min_child_samples=80,subsample=0.9,colsample_bytree=0.9,reg_lambda=2.0,
        random_state=seed,n_jobs=2,verbosity=-1
    )

def fit_model(df,features,target,seed):
    m=lgb.LGBMClassifier(**model_params(seed))
    w=1.0/np.maximum(df["field_size"].to_numpy(dtype=float),1.0)
    m.fit(df[features],df[target].astype(int),sample_weight=w)
    return m

def pred(m,df,features):
    return np.clip(m.predict_proba(df[features])[:,1],EPS,1-EPS)

def binary_metrics(y,p):
    y=np.asarray(y,dtype=float); p=np.clip(np.asarray(p,dtype=float),EPS,1-EPS)
    ll=float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
    br=float(np.mean((p-y)**2))
    return ll,br

def race_normalize(df,p):
    p=np.asarray(p,dtype=float); out=np.zeros(len(p),dtype=float)
    for _,idxs in df.groupby("race_id",sort=False).groups.items():
        ii=np.asarray(list(idxs),dtype=int)
        v=np.clip(p[ii],EPS,None); s=v.sum()
        out[ii]=v/s if s>0 else 1.0/len(ii)
    return out

def winner_nll(df,p):
    pn=race_normalize(df,p)
    vals=[]
    for _,idxs in df.groupby("race_id",sort=False).groups.items():
        ii=np.asarray(list(idxs),dtype=int); sub=df.loc[ii]
        y=sub["target_win"].to_numpy(dtype=int)
        if y.sum()!=1: continue
        j=ii[np.where(y==1)[0][0]]
        vals.append(-math.log(clip01(pn[j])))
    return float(np.mean(vals)) if vals else None

def per_race_loss(df,p,target):
    p=np.clip(np.asarray(p,dtype=float),EPS,1-EPS)
    if target=="WIN":
        pn=race_normalize(df,p)
        out={}
        for rid,idxs in df.groupby("race_id",sort=False).groups.items():
            ii=np.asarray(list(idxs),dtype=int); y=df.loc[ii,"target_win"].to_numpy(dtype=int)
            if y.sum()!=1: continue
            j=ii[np.where(y==1)[0][0]]
            out[str(rid)]=-math.log(clip01(pn[j]))
        return out
    out={}
    for rid,idxs in df.groupby("race_id",sort=False).groups.items():
        ii=np.asarray(list(idxs),dtype=int); y=df.loc[ii,"target_top3"].to_numpy(dtype=float); pp=p[ii]
        mask=np.isfinite(y)
        if not mask.any(): continue
        yy=y[mask]; qq=pp[mask]
        out[str(rid)]=float(np.mean(-(yy*np.log(qq)+(1-yy)*np.log(1-qq))))
    return out

def boot_pair(a,b,year,target):
    keys=sorted(set(a)&set(b)); d=np.asarray([a[k]-b[k] for k in keys],dtype=float)
    rng=np.random.default_rng(20261004+int(year)+(0 if target=="WIN" else 100))
    means=np.empty(BOOT_REPS,dtype=float); n=len(d)
    for s in range(0,BOOT_REPS,250):
        k=min(250,BOOT_REPS-s); idx=rng.integers(0,n,size=(k,n)); means[s:s+k]=d[idx].mean(axis=1)
    lo,hi=np.quantile(means,[0.025,0.975])
    return {
        "test_year":year,"target":target,"model_a":"MARKET_CRITIC","model_b":"MARKET_ONLY",
        "mean_delta_a_minus_b":float(d.mean()),"ci95_low":float(lo),"ci95_high":float(hi),
        "critic_better_race_pct":float(100*np.mean(d<0)),"races":n,"bootstrap_reps":BOOT_REPS
    }

def band_thresholds(delta):
    return tuple(float(x) for x in np.quantile(np.asarray(delta,dtype=float),[.10,.25,.75,.90]))

def band(x,t):
    q10,q25,q75,q90=t
    if x<=q10: return "STRONG_OVERVALUED"
    if x<=q25: return "OVERVALUED"
    if x>=q90: return "STRONG_UNDERVALUED"
    if x>=q75: return "UNDERVALUED"
    return "NEUTRAL"

def band_rows(test,p_market_win,p_critic_win,p_market_top3,p_critic_top3,thr,year):
    x=test.copy().reset_index(drop=True)
    x["p_market_win"]=p_market_win; x["p_critic_win"]=p_critic_win
    x["p_market_top3"]=p_market_top3; x["p_critic_top3"]=p_critic_top3
    x["critic_delta_win"]=x["p_critic_win"]-x["p_market_win"]
    x["critic_band"]=[band(v,thr) for v in x["critic_delta_win"]]
    rows=[]
    order=("STRONG_OVERVALUED","OVERVALUED","NEUTRAL","UNDERVALUED","STRONG_UNDERVALUED")
    for b in order:
        g=x[x["critic_band"]==b]
        if g.empty: continue
        win=g["target_win"].astype(float); top3=g["target_top3"].astype(float)
        valid=top3.notna()
        roi=100*float(np.mean(g["final_win_odds"].to_numpy(dtype=float)*win.to_numpy(dtype=float)))
        rows.append({
            "test_year":year,"critic_band":b,"horses":len(g),"races":g["race_id"].nunique(),
            "mean_final_win_odds":float(g["final_win_odds"].mean()),
            "mean_market_win_pct":100*float(g["p_market_win"].mean()),
            "mean_critic_win_pct":100*float(g["p_critic_win"].mean()),
            "actual_win_pct":100*float(win.mean()),
            "actual_minus_market_win_pp":100*float((win-g["p_market_win"]).mean()),
            "mean_market_top3_pct":100*float(g.loc[valid,"p_market_top3"].mean()) if valid.any() else None,
            "mean_critic_top3_pct":100*float(g.loc[valid,"p_critic_top3"].mean()) if valid.any() else None,
            "actual_top3_pct":100*float(top3[valid].mean()) if valid.any() else None,
            "actual_minus_market_top3_pp":100*float((top3[valid]-g.loc[valid,"p_market_top3"]).mean()) if valid.any() else None,
            "flat_win_roi_final_odds_proxy_pct":roi,
        })
    return rows

def main():
    a=parse_args()
    c=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if c.get("contract")!="L1_MARKET_CRITIC_V1": raise SystemExit("wrong contract")
    cp=c["cost_policy"]
    if cp["github_standard_cpu_only"] is not True or cp["paid_runner"] or cp["gpu"] or cp["paid_cloud_compute"] or cp["paid_artifact_or_cache"]:
        raise SystemExit("cost guard drift")
    if c["data_policy"]["locked_years"]!=[2026]: raise SystemExit("2026 lock drift")

    manifest,df=load_dataset(a.dataset_dir)
    if 2026 in set(df["year"]): raise SystemExit("2026 sealed")
    l1_features=list(manifest["feature_columns"])
    for col in l1_features:
        df[col]=pd.to_numeric(df[col],errors="coerce").fillna(0.0)
    df["year"]=pd.to_numeric(df["year"],errors="raise").astype(int)
    df["race_id"]=df["race_id"].astype(str)
    df["target_win"]=pd.to_numeric(df["target_win"],errors="coerce")
    df["target_top3"]=pd.to_numeric(df["target_top3"],errors="coerce")

    z,skipped=add_market(df,a.backfill_root)
    z=z.reset_index(drop=True)
    market_features=[
        "field_size","final_win_odds","log_final_win_odds","market_pwin_raw","market_rank_pct",
        "market_overround","market_entropy","market_top1_p","market_top2_p",
        "market_top1_top2_gap","market_gap_to_top1"
    ]
    gap_features=["l1_pwin_norm","l1_market_prob_gap","l1_market_rank_gap_pct"]
    critic_features=list(dict.fromkeys(market_features+l1_features+gap_features))

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    metric_rows=[]; pair_rows=[]; bands=[]; importance=[]; threshold_rows=[]

    for test_year,train_years in FOLDS:
        train=z[z["year"].isin(train_years)].copy()
        test=z[z["year"]==test_year].copy().reset_index(drop=True)
        races=train[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"])
        cut=max(1,int(len(races)*0.8))
        fit_ids=set(races.iloc[:cut]["race_id"].astype(str)); cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
        fit=train[train["race_id"].isin(fit_ids)].copy()
        cal=train[train["race_id"].isin(cal_ids)].copy().reset_index(drop=True)

        predictions={}
        cal_delta_win=None
        for target_name,target_col in (("WIN","target_win"),("TOP3","target_top3")):
            if target_name=="WIN":
                fit_t=fit[(fit["train_eligible"]==True)&fit[target_col].notna()].copy()
                train_t=train[(train["train_eligible"]==True)&train[target_col].notna()].copy()
            else:
                fit_t=fit[fit[target_col].notna()].copy()
                train_t=train[train[target_col].notna()].copy()

            feature_map={"MARKET_ONLY":market_features,"L1_ONLY":l1_features,"MARKET_CRITIC":critic_features}
            cal_pred={}
            test_pred={}
            for mi,(model_name,features) in enumerate(feature_map.items()):
                mf=fit_model(fit_t,features,target_col,20261004+test_year*10+mi+(0 if target_name=="WIN" else 1000))
                cal_pred[model_name]=pred(mf,cal,features)
                m=fit_model(train_t,features,target_col,20261104+test_year*10+mi+(0 if target_name=="WIN" else 1000))
                test_pred[model_name]=pred(m,test,features)
                if model_name=="MARKET_CRITIC":
                    gains=m.booster_.feature_importance(importance_type="gain")
                    for name,g in sorted(zip(features,gains),key=lambda x:-x[1])[:40]:
                        importance.append({"test_year":test_year,"target":target_name,"feature":name,"gain":float(g)})
            predictions[target_name]=test_pred

            y=test[target_col].to_numpy(dtype=float)
            valid=np.isfinite(y)
            for model_name in MODELS:
                ll,br=binary_metrics(y[valid],test_pred[model_name][valid])
                metric_rows.append({
                    "test_year":test_year,"target":target_name,"model":model_name,
                    "horses":int(valid.sum()),"races":int(test.loc[valid,"race_id"].nunique()),
                    "binary_logloss":ll,"brier":br,
                    "race_winner_nll":winner_nll(test,test_pred[model_name]) if target_name=="WIN" else None
                })
            a_loss=per_race_loss(test,test_pred["MARKET_CRITIC"],target_name)
            b_loss=per_race_loss(test,test_pred["MARKET_ONLY"],target_name)
            pair_rows.append(boot_pair(a_loss,b_loss,test_year,target_name))

            if target_name=="WIN":
                cal_delta_win=cal_pred["MARKET_CRITIC"]-cal_pred["MARKET_ONLY"]

        thr=band_thresholds(cal_delta_win)
        threshold_rows.append({"test_year":test_year,"q10":thr[0],"q25":thr[1],"q75":thr[2],"q90":thr[3],"source_prior_years":"|".join(map(str,train_years))})
        bands.extend(band_rows(
            test,
            predictions["WIN"]["MARKET_ONLY"],predictions["WIN"]["MARKET_CRITIC"],
            predictions["TOP3"]["MARKET_ONLY"],predictions["TOP3"]["MARKET_CRITIC"],
            thr,test_year
        ))
        print("MARKET_CRITIC_FOLD_DONE "+json.dumps({"test_year":test_year,"fit_races":len(fit_ids),"calibration_races":len(cal_ids),"test_races":test["race_id"].nunique()},separators=(",",":")),flush=True)

    write_csv(out/"fold-metrics.csv",metric_rows)
    write_csv(out/"paired-bootstrap.csv",pair_rows)
    write_csv(out/"critic-bands.csv",bands)
    write_csv(out/"critic-thresholds.csv",threshold_rows)
    write_csv(out/"feature-importance.csv",importance)
    write_csv(out/"skipped-market-races.csv",skipped)

    # pooled headline from yearly proper-score deltas
    headline=[]
    for target in TARGETS:
        rows=[r for r in metric_rows if r["target"]==target]
        for metric in ("binary_logloss","brier","race_winner_nll"):
            vals=[]
            for y in (2023,2024,2025):
                arow=next((r for r in rows if r["test_year"]==y and r["model"]=="MARKET_CRITIC"),None)
                brow=next((r for r in rows if r["test_year"]==y and r["model"]=="MARKET_ONLY"),None)
                if arow and brow and arow.get(metric) is not None and brow.get(metric) is not None:
                    vals.append(float(arow[metric])-float(brow[metric]))
            if vals:
                headline.append({"target":target,"metric":metric,"mean_year_delta_critic_minus_market":float(np.mean(vals)),"years_better":sum(v<0 for v in vals),"years":len(vals)})
    write_csv(out/"headline.csv",headline)

    sig=[r for r in pair_rows if r["ci95_high"]<0]
    win_sig=sum(1 for r in sig if r["target"]=="WIN")
    top3_sig=sum(1 for r in sig if r["target"]=="TOP3")
    if win_sig>=2 or top3_sig>=2:
        verdict="MARKET_CRITIC_ADDS_STABLE_SIGNAL"
    elif any(r["mean_delta_a_minus_b"]<0 for r in pair_rows):
        verdict="MARKET_CRITIC_DIRECTIONAL_NOT_CONCLUSIVE"
    else:
        verdict="MARKET_CRITIC_NO_CLEAR_GAIN"

    summary={
        "contract":"L1_MARKET_CRITIC_V1_RESULT",
        "architecture":"MARKET_ONLY vs L1_ONLY vs MARKET_PLUS_L1_CRITIC",
        "critic_score":"p_market_critic - p_market_only",
        "folds":[2023,2024,2025],
        "market_price":"NETKEIBA_FINAL_WIN_ODDS",
        "closing_market_research_only":True,
        "2026_locked":True,
        "bootstrap_reps":BOOT_REPS,
        "verdict":verdict,
        "promotion":False
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L1 Market Critic V1\n\n"
        "Horse-level critic of the closing WIN market. Negative critic score means the market is judged overvalued; positive means undervalued. "
        "All test years are strict prior-year walk-forward. Final odds are a historical closing-market proxy, not a live deployment guarantee. "
        "2026 is sealed. No staking optimization is performed here.\n",encoding="utf-8"
    )
    print("===== HEADLINE ====="); print((out/"headline.csv").read_text())
    print("===== PAIRED BOOTSTRAP ====="); print((out/"paired-bootstrap.csv").read_text())
    print("===== CRITIC BANDS ====="); print((out/"critic-bands.csv").read_text())
    print("VERDICT="+verdict)
    print("L1_MARKET_CRITIC_V1_READY")

if __name__=="__main__":
    main()
