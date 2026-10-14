#!/usr/bin/env python3
import argparse,csv,json,math
from pathlib import Path
import concurrent.futures
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PRIMARY_YEARS=(2023,2024,2025)
EDGE_THRESHOLDS=(0.0,0.05,0.10,0.20,0.30,0.50,0.75,1.00)
EDGE_BANDS=((0.0,0.05),(0.05,0.10),(0.10,0.20),(0.20,0.30),(0.30,0.50),(0.50,1.00),(1.00,np.inf))
ZONES=("ALL","AI_TOP3","DISAGREE","AI_UNIQUE_TOP3","OVERLAP2_AI_UNIQUE")
LAMBDAS=(0.001,0.003,0.01,0.03,0.10,0.30,1.0)
VARIANTS={
    "ANCHOR_ABILITY":[
        "field_size","distill_ai_rank01","distill_score_z","distill_score_gap_to_best",
        "distill_is_ai_top1","distill_is_ai_top3","distill_is_ai_top6",
    ],
    "ANCHOR_ABILITY_RISK":[
        "field_size","distill_ai_rank01","distill_score_z","distill_score_gap_to_best",
        "distill_is_ai_top1","distill_is_ai_top3","distill_is_ai_top6","risk",
    ],
    "ANCHOR_FULL":[
        "field_size","distill_ai_rank01","distill_score_z","distill_score_gap_to_best",
        "distill_is_ai_top1","distill_is_ai_top3","distill_is_ai_top6","risk",
        "rank_gap_norm","abs_rank_gap_norm","top3_overlap_norm","ai_unique_top3",
        "market_unique_top3","top1_match","risk_x_abs_gap","ability_x_market_gap",
        "ability_x_disagree_level","risk_x_disagree_level","rankgap_x_disagree_level",
    ],
}

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def logloss(y,p):
    y=np.asarray(y,dtype=float); p=np.clip(np.asarray(p,dtype=float),1e-12,1-1e-12)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def brier(y,p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    return float(np.mean((y-p)**2))

def auc_rank(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    pos=int(y.sum()); neg=len(y)-pos
    if pos==0 or neg==0: return np.nan
    r=pd.Series(p).rank(method="average").to_numpy()
    return float((r[y==1].sum()-pos*(pos+1)/2)/(pos*neg))

def chronological_split(train):
    races=train[["_race_date","_race_id"]].drop_duplicates().sort_values(["_race_date","_race_id"]).reset_index(drop=True)
    cut=max(1,min(len(races)-1,int(len(races)*0.80)))
    fit_ids=set(races.iloc[:cut]["_race_id"].astype(str))
    val_ids=set(races.iloc[cut:]["_race_id"].astype(str))
    return (
        train[train["_race_id"].astype(str).isin(fit_ids)].copy(),
        train[train["_race_id"].astype(str).isin(val_ids)].copy(),
    )

def fit_transform_stats(df,features):
    x=df[features].apply(pd.to_numeric,errors="coerce")
    med=x.median(axis=0).fillna(0.0)
    x=x.fillna(med)
    mean=x.mean(axis=0)
    std=x.std(axis=0,ddof=0).replace(0,np.nan).fillna(1.0)
    return med,mean,std

def transform_x(df,features,stats):
    med,mean,std=stats
    x=df[features].apply(pd.to_numeric,errors="coerce").fillna(med)
    return ((x-mean)/std).to_numpy(dtype=np.float64)

def anchored_arrays(df,features,stats):
    q=df.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    x=transform_x(q,features,stats)
    y=q["target_win"].to_numpy(dtype=np.float64)
    base=np.log(q["market_prob_norm"].clip(1e-12,1.0).to_numpy(dtype=np.float64))
    rid=q["_race_id"].astype(str).to_numpy()
    starts=np.r_[0,np.flatnonzero(rid[1:]!=rid[:-1])+1]
    sizes=np.diff(np.r_[starts,len(q)])
    return q,x,y,base,starts,sizes,len(starts)

def fit_beta(df,features,lam,stats=None):
    if stats is None: stats=fit_transform_stats(df,features)
    q,x,y,base,starts,sizes,nr=anchored_arrays(df,features,stats)
    def fg(beta):
        z=base+x.dot(beta)
        mx=np.maximum.reduceat(z,starts)
        ez=np.exp(z-np.repeat(mx,sizes))
        sums=np.add.reduceat(ez,starts)
        p=ez/np.repeat(sums,sizes)
        loss=(-float(np.dot(y,z))+float(np.sum(mx+np.log(sums))))/nr + 0.5*lam*float(np.dot(beta,beta))
        grad=x.T.dot(p-y)/nr + lam*beta
        return loss,grad
    res=minimize(lambda b:fg(b),np.zeros(x.shape[1],dtype=np.float64),jac=True,method="L-BFGS-B",
                 options={"maxiter":180,"ftol":1e-10,"gtol":1e-7})
    return np.asarray(res.x,dtype=float),stats,float(res.fun)

def predict_anchor(df,features,beta,stats):
    q=df.reset_index(drop=True).copy()
    q["__pos"]=np.arange(len(q),dtype=int)
    q=q.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    x=transform_x(q,features,stats)
    z=np.log(q["market_prob_norm"].clip(1e-12,1.0).to_numpy(dtype=float))+x.dot(beta)
    rid=q["_race_id"].astype(str).to_numpy()
    starts=np.r_[0,np.flatnonzero(rid[1:]!=rid[:-1])+1]
    sizes=np.diff(np.r_[starts,len(q)])
    mx=np.maximum.reduceat(z,starts)
    ez=np.exp(z-np.repeat(mx,sizes))
    sums=np.add.reduceat(ez,starts)
    p=ez/np.repeat(sums,sizes)
    out=np.empty(len(df),dtype=float)
    out[q["__pos"].to_numpy(dtype=int)]=p
    return out

def race_nll(df,p):
    q=df[["_race_id","target_win"]].copy()
    q["p"]=np.clip(np.asarray(p,dtype=float),1e-12,1.0)
    winners=q[q["target_win"].eq(1)]
    return float(-np.log(winners["p"]).mean())

def choose_lambda(train,features):
    fit,val=chronological_split(train)
    stats=fit_transform_stats(fit,features)
    def one(lam):
        beta,_,obj=fit_beta(fit,features,lam,stats)
        pv=predict_anchor(val,features,beta,stats)
        return lam,race_nll(val,pv),obj
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        rows=list(ex.map(one,LAMBDAS))
    rows.sort(key=lambda x:(x[1],x[0]))
    return rows[0][0],rows

def metrics(y,p):
    return {"logloss":logloss(y,p),"brier":brier(y,p),"auc":auc_rank(y,p)}

def rank_metrics(z,pcol):
    q=z.sort_values(["_race_id",pcol,"_horse_id"],ascending=[True,False,True]).copy()
    q["__r"]=q.groupby("_race_id",sort=False).cumcount()+1
    top=q[q["__r"]==1]
    return {
        "races":int(q["_race_id"].nunique()),
        "top1_win_pct":100*float(top["target_win"].mean()),
        "top1_top3_pct":100*float(top["target_top3"].mean()),
    }

def zone_mask(z,zone):
    if zone=="ALL": return pd.Series(True,index=z.index)
    if zone=="AI_TOP3": return pd.to_numeric(z["distill_ai_rank"],errors="coerce").le(3)
    if zone=="DISAGREE": return z["distill_ai_rank"].ne(z["market_rank"])
    if zone=="AI_UNIQUE_TOP3": return z["ai_unique_top3"].eq(1)
    if zone=="OVERLAP2_AI_UNIQUE": return z["ai_unique_top3"].eq(1)&z["top3_overlap"].eq(2)
    raise ValueError(zone)

def add_value_cols(z,pcol):
    q=z.copy()
    q["_fair_odds"]=1.0/q[pcol].clip(lower=1e-12)
    q["_edge"]=q[pcol]*q["final_win_odds"]-1.0
    return q

def roi_summary(q):
    if q.empty:
        return {"bets":0,"races":0,"win_rate_pct":np.nan,"avg_final_odds":np.nan,
                "avg_fair_odds":np.nan,"roi_pct":np.nan,"profit_yen":0.0}
    stake=100.0*len(q)
    ret=float((q["target_win"]*q["final_win_odds"]*100.0).sum())
    return {
        "bets":int(len(q)),"races":int(q["_race_id"].nunique()),
        "win_rate_pct":100*float(q["target_win"].mean()),
        "avg_final_odds":float(q["final_win_odds"].mean()),
        "avg_fair_odds":float(q["_fair_odds"].mean()),
        "roi_pct":100*ret/stake,"profit_yen":ret-stake,
    }

def roi_rows(z,pcol,variant,test_year):
    q=add_value_cols(z,pcol)
    rows=[]; bands=[]
    for zone in ZONES:
        base=q[zone_mask(q,zone)].copy()
        for th in EDGE_THRESHOLDS:
            rows.append({"test_year":test_year,"variant":variant,"zone":zone,"edge_threshold":th,
                         **roi_summary(base[base["_edge"]>=th])})
        for lo,hi in EDGE_BANDS:
            sel=base[(base["_edge"]>=lo)&(base["_edge"]<hi)]
            bands.append({"test_year":test_year,"variant":variant,"zone":zone,
                          "edge_lo":lo,"edge_hi":"INF" if not np.isfinite(hi) else hi,
                          **roi_summary(sel)})
    return rows,bands

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",required=True)
    ap.add_argument("--out-dir",required=True)
    a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    z=pd.read_csv(a.cache,compression="infer",low_memory=False)
    z["_race_date"]=z["_race_date"].astype(str)
    for c in ("year","target_win","target_top3","final_win_odds","market_prob_norm","market_rank",
              "distill_ai_rank","top3_overlap","ai_unique_top3","market_unique_top3","top1_match"):
        z[c]=pd.to_numeric(z[c],errors="coerce")
    if 2026 in set(z["year"].dropna().astype(int)): raise SystemExit("2026 sealed")
    # Derived disagreement interactions are cheap and deterministic.
    z["disagree_level"]=(3.0-z["top3_overlap"].clip(0,3))/3.0
    z["ability_x_disagree_level"]=z["distill_score_z"]*z["disagree_level"]
    z["risk_x_disagree_level"]=z["risk"]*z["disagree_level"]
    z["rankgap_x_disagree_level"]=z["rank_gap_norm"]*z["disagree_level"]

    model_rows=[]; lambda_rows=[]; coef_rows=[]; value_rows=[]; band_rows=[]; pred_frames=[]
    for test_year in PRIMARY_YEARS:
        train=z[z["year"]<test_year].copy().reset_index(drop=True)
        test=z[z["year"]==test_year].copy().reset_index(drop=True)
        test["P__MARKET_DIRECT"]=test["market_prob_norm"].clip(1e-12,1.0)
        base=metrics(test["target_win"],test["P__MARKET_DIRECT"])
        base.update(rank_metrics(test,"P__MARKET_DIRECT"))
        base["race_nll"]=race_nll(test,test["P__MARKET_DIRECT"])
        model_rows.append({"test_year":test_year,"variant":"MARKET_DIRECT","train_years":"","horses":len(test),**base})
        rr,bb=roi_rows(test,"P__MARKET_DIRECT","MARKET_DIRECT",test_year)
        value_rows.extend(rr); band_rows.extend(bb)

        def fit_variant(item):
            variant,features=item
            lam,grid=choose_lambda(train,features)
            beta,stats,obj=fit_beta(train,features,lam)
            pred=predict_anchor(test,features,beta,stats)
            return variant,features,lam,grid,beta,pred,obj

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futs=[ex.submit(fit_variant,item) for item in VARIANTS.items()]
            for fut in concurrent.futures.as_completed(futs):
                variant,features,lam,grid,beta,pred,obj=fut.result()
                pcol=f"P__{variant}"
                test[pcol]=pred
                met=metrics(test["target_win"],pred)
                met.update(rank_metrics(test,pcol))
                met["race_nll"]=race_nll(test,pred)
                model_rows.append({
                    "test_year":test_year,"variant":variant,
                    "train_years":"|".join(map(str,sorted(set(train["year"].astype(int))))),
                    "horses":len(test),"selected_lambda":lam,**met,
                })
                for l,vnll,vobj in grid:
                    lambda_rows.append({"test_year":test_year,"variant":variant,"lambda":l,
                                        "validation_race_nll":vnll,"fit_objective":vobj,"selected":int(l==lam)})
                for f,b in zip(features,beta):
                    coef_rows.append({"test_year":test_year,"variant":variant,"feature":f,
                                      "coef":float(b),"selected_lambda":lam})
                rr,bb=roi_rows(test,pcol,variant,test_year)
                value_rows.extend(rr); band_rows.extend(bb)
                print(f"ANCHOR_CACHE_MODEL_DONE year={test_year} variant={variant} lambda={lam} "
                      f"race_nll={met['race_nll']:.6f} logloss={met['logloss']:.6f}",flush=True)

        keep=["year","_race_id","_horse_id","_race_date","target_win","target_top3","final_win_odds",
              "market_prob_norm","market_rank","distill_ai_rank","risk","top3_overlap","ai_unique_top3",
              "P__MARKET_DIRECT",*[f"P__{v}" for v in VARIANTS]]
        pred_frames.append(test[keep].copy())

    preds=pd.concat(pred_frames,ignore_index=True)
    pooled_models=[]; pooled_values=[]; pooled_bands=[]
    for variant in ("MARKET_DIRECT",*VARIANTS.keys()):
        pcol="P__MARKET_DIRECT" if variant=="MARKET_DIRECT" else f"P__{variant}"
        met=metrics(preds["target_win"],preds[pcol]); met.update(rank_metrics(preds,pcol)); met["race_nll"]=race_nll(preds,preds[pcol])
        pooled_models.append({"test_year":"2023-2025","variant":variant,"horses":len(preds),**met})
        rr,bb=roi_rows(preds,pcol,variant,"2023-2025"); pooled_values.extend(rr); pooled_bands.extend(bb)

    write_csv(out/"fold-probability-metrics.csv",model_rows)
    write_csv(out/"pooled-probability-metrics.csv",pooled_models)
    write_csv(out/"lambda-selection.csv",lambda_rows)
    write_csv(out/"residual-coefficients.csv",coef_rows)
    write_csv(out/"fixed-edge-roi-by-year.csv",value_rows)
    write_csv(out/"fixed-edge-roi-pooled.csv",pooled_values)
    write_csv(out/"edge-band-roi-by-year.csv",band_rows)
    preds.to_csv(out/"oos-fair-value-predictions.csv.gz",index=False,compression="gzip")
    (out/"summary.json").write_text(json.dumps({
        "contract":"MARKET_FIXED_ANCHOR_RESIDUAL_V2_FROM_CACHE",
        "cache":str(a.cache),
        "formula":"p_i proportional to q_market_i * exp(delta_i)",
        "market_coefficient":"fixed at 1",
        "calibration":"no isotonic",
        "lambda_selection":"prior-data chronological 80/20 only",
        "roi_target_used_for_training":False,
        "2026_locked":True,
        "promotion":False
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(pd.DataFrame(pooled_models).sort_values("race_nll").to_csv(index=False),flush=True)
    print("MARKET_FIXED_ANCHOR_RESIDUAL_V2_FROM_CACHE_COMPLETE",flush=True)

if __name__=="__main__":
    main()
