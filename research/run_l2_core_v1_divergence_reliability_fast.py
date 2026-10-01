#!/usr/bin/env python3
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l2_core_v1 import FOLDS,THRESHOLDS,load_dataset,max_drawdown
from run_l2_core_v1_ticket_calibration import final_conf_probs
from run_l2_core_v1_market_valuation_fast import (
    BET_TYPES,load_archived_calibrators,race_probabilities,ticket_p
)
from build_l2_bet_kings_dataset_v1 import iter_decoded_odds,load_day,load_odds_day,payout_map

# Diagnostic binning only. The learned correction factor comes from prior OOS outcomes.
LOGP_EDGES=(-99.0,-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,0.1)
LOGR_EDGES=(-99.0,-1.0,-0.32192809489,0.32192809489,1.0,2.0,3.0,99.0)
PRIOR=300.0
FACTOR_MIN=0.05
FACTOR_MAX=5.0
EDGE_BANDS=(("NEG",-1e99,0.0),("0_10",0.0,0.10),("10_20",0.10,0.20),("20_50",0.20,0.50),("50_100",0.50,1.0),("100_PLUS",1.0,1e99))

def parse_args():
    p=argparse.ArgumentParser(description="Fast OOS divergence reliability correction for L2 market valuation.")
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--calibrator-bins",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def bindex(x,edges):
    for i in range(len(edges)-1):
        if edges[i] <= x < edges[i+1]: return i
    return len(edges)-2

def cell_key(bet,pai,pmkt):
    lp=math.log10(max(pai,1e-12))
    lr=math.log2(max(pai,1e-12)/max(pmkt,1e-12))
    return bet,bindex(lp,LOGP_EDGES),bindex(lr,LOGR_EDGES)

def edge_band(x):
    for name,lo,hi in EDGE_BANDS:
        if lo<=x<hi: return name
    return "100_PLUS"

def merged_stats(hist):
    out=defaultdict(lambda:{"n":0,"hits":0,"sum_ai":0.0})
    for yearly in hist:
        for k,z in yearly.items():
            q=out[k]; q["n"]+=z["n"]; q["hits"]+=z["hits"]; q["sum_ai"]+=z["sum_ai"]
    return out

def learn_factors(hist):
    agg=merged_stats(hist); factors={}; rows=[]
    # Hierarchical fallback by bet + divergence, then bet-wide.
    bd=defaultdict(lambda:{"n":0,"hits":0,"sum_ai":0.0})
    bb=defaultdict(lambda:{"n":0,"hits":0,"sum_ai":0.0})
    for (bet,pi,ri),z in agg.items():
        for tgt in (bd[(bet,ri)],bb[bet]):
            tgt["n"]+=z["n"]; tgt["hits"]+=z["hits"]; tgt["sum_ai"]+=z["sum_ai"]

    def factor(z):
        if not z or z["n"]<=0 or z["sum_ai"]<=0: return 1.0
        mean_ai=z["sum_ai"]/z["n"]
        corrected=(z["hits"]+PRIOR*mean_ai)/(z["n"]+PRIOR)
        return min(FACTOR_MAX,max(FACTOR_MIN,corrected/max(mean_ai,1e-15)))

    for key,z in agg.items():
        bet,pi,ri=key
        # Sparse cells back off to divergence-only, then bet-wide.
        source="2D"
        zz=z
        if z["n"]<500:
            zz=bd[(bet,ri)]; source="DIVERGENCE"
        if zz["n"]<500:
            zz=bb[bet]; source="BET"
        f=factor(zz)
        factors[key]=f
        rows.append({
            "bet_type":bet,"logp_bin":pi,"logratio_bin":ri,"cell_tickets":z["n"],"cell_hits":z["hits"],
            "cell_mean_ai_pct":100*z["sum_ai"]/z["n"] if z["n"] else None,
            "source":source,"source_tickets":zz["n"],"reliability_factor":f
        })
    return factors,bd,bb,rows

def lookup_factor(bet,pai,pmkt,factors,bd,bb):
    key=cell_key(bet,pai,pmkt)
    if key in factors: return factors[key]
    ri=key[2]
    def f(z):
        if not z or z["n"]<=0 or z["sum_ai"]<=0: return 1.0
        mean_ai=z["sum_ai"]/z["n"]
        corrected=(z["hits"]+PRIOR*mean_ai)/(z["n"]+PRIOR)
        return min(FACTOR_MAX,max(FACTOR_MIN,corrected/max(mean_ai,1e-15)))
    if bd.get((bet,ri),{}).get("n",0)>=500: return f(bd[(bet,ri)])
    return f(bb.get(bet))

def scan_year(year,test,pwin,calibrators,root,factors_pack=None):
    racep,raced=race_probabilities(test,pwin)
    bydate=defaultdict(list)
    for rid,d in raced.items(): bydate[d].append(rid)
    yearly=defaultdict(lambda:{"n":0,"hits":0,"sum_ai":0.0})
    ev={(mode,b,t):{"tickets":0,"races":0,"hit_tickets":0,"hit_races":0,"stake":0.0,"ret":0.0,"profits":[]} for mode in ("AI_ONLY","DIVERGENCE_CORR") for b in BET_TYPES for t in THRESHOLDS}
    eb={(mode,b,name):{"n":0,"p":0.0,"y":0,"stake":0.0,"ret":0.0} for mode in ("AI_ONLY","DIVERGENCE_CORR") for b in BET_TYPES for name,_,_ in EDGE_BANDS}
    root=Path(root)
    for date in sorted(bydate):
        wanted=set(bydate[date])
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        oddsday=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in wanted:
            pack=day.get(rid); oddsrec=oddsday.get(rid)
            if pack is None or oddsrec is None: raise SystemExit(f"missing market row race={rid}")
            payouts,present=payout_map(pack); p=racep[rid]
            items=defaultdict(list); sumq=defaultdict(float); summ=defaultdict(float)
            for bet,nums,odd in iter_decoded_odds(oddsrec):
                if bet not in present: continue
                raw=ticket_p(bet,nums,p)
                if raw is None: continue
                q=calibrators[bet].predict_one(raw)
                m=1.0/float(odd)
                ret=float(payouts.get((bet,nums),0.0)); hit=int(ret>0)
                items[bet].append([nums,float(odd),q,m,ret,hit])
                sumq[bet]+=q; summ[bet]+=m
            for bet,arr in items.items():
                if not arr or sumq[bet]<=0 or summ[bet]<=0: continue
                base=[]
                for nums,odd,q,m,ret,hit in arr:
                    pai=q/sumq[bet]; pm=m/summ[bet]
                    base.append([odd,pai,pm,ret,hit])
                    z=yearly[cell_key(bet,pai,pm)]
                    z["n"]+=1; z["hits"]+=hit; z["sum_ai"]+=pai
                if factors_pack is None: continue
                factors,bd,bb=factors_pack
                corr_raw=[]
                for odd,pai,pm,ret,hit in base:
                    f=lookup_factor(bet,pai,pm,factors,bd,bb)
                    corr_raw.append(pai*f)
                sc=sum(corr_raw)
                if sc<=0: sc=1.0
                raceacc={(mode,t):{"tickets":0,"hit":0,"stake":0.0,"ret":0.0} for mode in ("AI_ONLY","DIVERGENCE_CORR") for t in THRESHOLDS}
                for (odd,pai,pm,ret,hit),cr in zip(base,corr_raw):
                    pc=cr/sc
                    for mode,pv in (("AI_ONLY",pai),("DIVERGENCE_CORR",pc)):
                        edge=pv*odd-1.0
                        z=eb[(mode,bet,edge_band(edge))]
                        z["n"]+=1; z["p"]+=pv; z["y"]+=hit; z["stake"]+=100.0; z["ret"]+=ret
                        for t in THRESHOLDS:
                            if edge>=t:
                                r=raceacc[(mode,t)]; r["tickets"]+=1; r["hit"]+=hit; r["stake"]+=100.0; r["ret"]+=ret
                for (mode,t),r in raceacc.items():
                    if not r["tickets"]: continue
                    g=ev[(mode,bet,t)]
                    g["tickets"]+=r["tickets"]; g["races"]+=1; g["hit_tickets"]+=r["hit"]; g["hit_races"]+=int(r["hit"]>0)
                    g["stake"]+=r["stake"]; g["ret"]+=r["ret"]; g["profits"].append(r["ret"]-r["stake"])
    return yearly,ev,eb

def output_rows(year,ev,eb):
    market=[]; edges=[]
    for mode in ("AI_ONLY","DIVERGENCE_CORR"):
        for bet in BET_TYPES:
            for t in THRESHOLDS:
                g=ev[(mode,bet,t)]
                market.append({"test_year":year,"mode":mode,"bet_type":bet,"edge_threshold":t,"tickets":g["tickets"],"bought_races":g["races"],"hit_tickets":g["hit_tickets"],"hit_races":g["hit_races"],"stake_yen":g["stake"],"return_yen":g["ret"],"profit_yen":g["ret"]-g["stake"],"roi_pct":100*g["ret"]/g["stake"] if g["stake"] else None,"race_hit_rate_pct":100*g["hit_races"]/g["races"] if g["races"] else None,"ticket_hit_rate_pct":100*g["hit_tickets"]/g["tickets"] if g["tickets"] else None,"max_drawdown_yen":max_drawdown(g["profits"])})
            for name,_,_ in EDGE_BANDS:
                z=eb[(mode,bet,name)]
                edges.append({"test_year":year,"mode":mode,"bet_type":bet,"edge_band":name,"tickets":z["n"],"mean_predicted_hit_pct":100*z["p"]/z["n"] if z["n"] else None,"actual_hit_pct":100*z["y"]/z["n"] if z["n"] else None,"calibration_gap_pp":100*(z["p"]-z["y"])/z["n"] if z["n"] else None,"roi_pct":100*z["ret"]/z["stake"] if z["stake"] else None})
    return market,edges

def write_csv(path,rows):
    if not rows: return
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L2_CORE_V1": raise SystemExit("wrong contract")
    manifest,df=load_dataset(a.dataset_dir)
    features=list(manifest["feature_columns"])
    for c in features+["vote2","vote3","top1_votes","consensus_rank","consensus_rank_pct","field_size"]:
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
    df["year"]=df["year"].astype(int); df["race_id"]=df["race_id"].astype(str)
    archived=load_archived_calibrators(a.calibrator_bins)
    params=dict(objective="binary",n_estimators=260,learning_rate=0.04,num_leaves=31,min_child_samples=50,subsample=0.9,colsample_bytree=0.9,reg_lambda=1.0,random_state=20261001,n_jobs=2,verbosity=-1)
    hist=[]; factor_rows=[]; market_rows=[]; edge_rows=[]
    for test_year,train_years in FOLDS:
        train=df[df["year"].isin(train_years) & (df["train_eligible"]==True) & (df["label_available"]==True)].copy()
        test=df[df["year"]==test_year].copy().reset_index(drop=True)
        races=train[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"])
        cut=max(1,int(len(races)*0.8)); fit_ids=set(races.iloc[:cut]["race_id"].astype(str)); cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
        fit=train[train["race_id"].isin(fit_ids)].copy()
        calib=train[train["race_id"].isin(cal_ids)].copy().reset_index(drop=True)
        ptest,_=final_conf_probs(train,calib,fit,test,features,params)
        pack=None
        if hist:
            factors,bd,bb,rows=learn_factors(hist)
            for r in rows:
                r["test_year"]=test_year; r["trained_on_years"]="|".join(str(y) for y in range(2023,test_year))
            factor_rows.extend(rows); pack=(factors,bd,bb)
        yearly,ev,eb=scan_year(test_year,test,ptest,archived[test_year],a.backfill_root,pack)
        if pack is not None:
            m,e=output_rows(test_year,ev,eb); market_rows.extend(m); edge_rows.extend(e)
        hist.append(yearly)
        print("DIVERGENCE_CORR_YEAR_DONE "+json.dumps({"year":test_year,"evaluated":pack is not None},separators=(",",":")),flush=True)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"learned-reliability-factors.csv",factor_rows)
    write_csv(out/"market-diagnostic.csv",market_rows)
    write_csv(out/"edge-band-calibration.csv",edge_rows)
    summary={"contract":"L2_CORE_V1_DIVERGENCE_RELIABILITY","method":"prior-OOS reliability correction conditioned on calibrated AI probability band and AI/normalized-market divergence; AI probabilities are rescaled then renormalized within race/bet","market_is_not_prediction_target":True,"market_blend":False,"odds_used_in_l1_or_hit_model":False,"odds_used_only_for_divergence_reliability_and_final_valuation":True,"evaluation_years":[2024,2025],"2026_locked":True,"promotion":False}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_DIVERGENCE_RELIABILITY_READY")

if __name__=="__main__":
    main()
