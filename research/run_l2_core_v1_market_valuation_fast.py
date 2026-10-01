#!/usr/bin/env python3
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l2_core_v1 import FOLDS,THRESHOLDS,load_dataset,max_drawdown
from run_l2_core_v1_ticket_calibration import final_conf_probs
from build_l2_bet_kings_dataset_v1 import (
    BET_GROUP,iter_decoded_odds,load_day,load_odds_day,payout_map
)

BET_TYPES=("WIN","QUINELLA","EXACTA","TRIO","TRIFECTA")
ALPHAS=tuple(i/20 for i in range(21))
EDGE_BANDS=(("NEG",-1e99,0.0),("0_10",0.0,0.10),("10_20",0.10,0.20),("20_50",0.20,0.50),("50_100",0.50,1.0),("100_PLUS",1.0,1e99))

def parse_args():
    p=argparse.ArgumentParser(description="Fast L2 market valuation: reuse frozen ticket calibrators and stream odds once.")
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--calibrator-bins",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

class ArchivedCalibrator:
    def __init__(self,xs,ys):
        order=np.argsort(xs)
        self.x=np.asarray(xs,dtype=float)[order]
        self.y=np.asarray(ys,dtype=float)[order]
    def predict_one(self,p):
        x=math.log10(max(float(p),1e-8))
        return float(np.interp(x,self.x,self.y,left=self.y[0],right=self.y[-1]))

def load_archived_calibrators(path):
    rows=defaultdict(lambda:defaultdict(list))
    with open(path,newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y=int(r["test_year"]); b=r["bet_type"]
            p=float(r["mean_raw_p"]); q=float(r["isotonic_hit_rate"])
            rows[y][b].append((math.log10(max(p,1e-8)),q))
    out={}
    for y,bb in rows.items():
        out[y]={}
        for b,pairs in bb.items():
            out[y][b]=ArchivedCalibrator([x for x,_ in pairs],[q for _,q in pairs])
    return out

def ordered2(p,a,b):
    if a==b or a not in p or b not in p: return None
    den=1.0-p[a]
    return p[a]*p[b]/den if den>1e-12 else 0.0

def ordered3(p,a,b,c):
    if len({a,b,c})<3 or any(x not in p for x in (a,b,c)): return None
    den1=1.0-p[a]; den2=1.0-p[a]-p[b]
    if den1<=1e-12 or den2<=1e-12: return 0.0
    return p[a]*(p[b]/den1)*(p[c]/den2)

def ticket_p(bet,nums,p):
    if bet=="WIN":
        return p.get(nums[0])
    if bet=="EXACTA":
        return ordered2(p,*nums)
    if bet=="QUINELLA":
        a=ordered2(p,nums[0],nums[1]); b=ordered2(p,nums[1],nums[0])
        return None if a is None or b is None else a+b
    if bet=="TRIFECTA":
        return ordered3(p,*nums)
    if bet=="TRIO":
        a,b,c=nums
        vals=(ordered3(p,a,b,c),ordered3(p,a,c,b),ordered3(p,b,a,c),ordered3(p,b,c,a),ordered3(p,c,a,b),ordered3(p,c,b,a))
        return None if any(v is None for v in vals) else sum(vals)
    return None

def race_probabilities(df,pwin):
    tmp=df.copy().reset_index(drop=True); tmp["pwin"]=np.asarray(pwin,dtype=float)
    probs={}; dates={}
    for rid,sub in tmp.groupby("race_id",sort=False):
        d={int(n):float(v) for n,v in zip(sub["horse_number"],sub["pwin"])}
        s=sum(d.values())
        probs[str(rid)]={k:v/s for k,v in d.items()}
        dates[str(rid)]=str(sub["race_date"].iloc[0])[:10]
    return probs,dates

def choose_alpha(history):
    out={}
    for bet in BET_TYPES:
        scores={a:0.0 for a in ALPHAS}; n=0
        for yr in history:
            z=yr.get(bet)
            if not z: continue
            n+=z["races"]
            for a in ALPHAS: scores[a]+=z["loss"][a]
        out[bet]=min(ALPHAS,key=lambda a:scores[a]) if n else 1.0
    return out

def edge_band(x):
    for name,lo,hi in EDGE_BANDS:
        if lo<=x<hi: return name
    return "100_PLUS"

def init_eval():
    return {(mode,b,t):{"tickets":0,"races":0,"hit_tickets":0,"hit_races":0,"stake":0.0,"ret":0.0,"profits":[]} for mode in ("AI_ONLY","MARKET_BLEND") for b in BET_TYPES for t in THRESHOLDS}

def init_edge():
    return {(mode,b,name):{"n":0,"p":0.0,"y":0,"stake":0.0,"ret":0.0} for mode in ("AI_ONLY","MARKET_BLEND") for b in BET_TYPES for name,_,_ in EDGE_BANDS}

def scan_year(year,test,pwin,calibrators,backfill_root,alphas=None):
    racep,raced=race_probabilities(test,pwin)
    bydate=defaultdict(list)
    for rid,d in raced.items(): bydate[d].append(rid)
    root=Path(backfill_root)
    trainstat={b:{"races":0,"loss":{a:0.0 for a in ALPHAS}} for b in BET_TYPES}
    ev=init_eval(); eb=init_edge()
    for date in sorted(bydate):
        wanted=set(bydate[date])
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        oddsday=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in wanted:
            pack=day.get(rid); oddsrec=oddsday.get(rid)
            if pack is None or oddsrec is None: raise SystemExit(f"missing market row race={rid}")
            payouts,present=payout_map(pack); p=racep[rid]
            items=defaultdict(list); sum_ai=defaultdict(float); sum_mkt=defaultdict(float)
            for bet,nums,odd in iter_decoded_odds(oddsrec):
                if bet not in present: continue
                raw=ticket_p(bet,nums,p)
                if raw is None: continue
                q=calibrators[bet].predict_one(raw)
                m=1.0/float(odd)
                ret=float(payouts.get((bet,nums),0.0)); hit=int(ret>0)
                items[bet].append((nums,float(odd),q,m,ret,hit))
                sum_ai[bet]+=q; sum_mkt[bet]+=m
            for bet,arr in items.items():
                if not arr or sum_ai[bet]<=0 or sum_mkt[bet]<=0: continue
                normalized=[]
                hits=0
                for nums,odd,q,m,ret,hit in arr:
                    pai=q/sum_ai[bet]; pm=m/sum_mkt[bet]
                    normalized.append((odd,pai,pm,ret,hit))
                    hits+=hit
                if hits==1:
                    win=next(x for x in normalized if x[4])
                    for a in ALPHAS:
                        pb=max(1e-15,a*win[1]+(1-a)*win[2])
                        trainstat[bet]["loss"][a]+=-math.log(pb)
                    trainstat[bet]["races"]+=1
                if alphas is None: continue
                rb={(mode,t):{"tickets":0,"hit":0,"stake":0.0,"ret":0.0} for mode in ("AI_ONLY","MARKET_BLEND") for t in THRESHOLDS}
                alpha=alphas[bet]
                for odd,pai,pm,ret,hit in normalized:
                    for mode,pv in (("AI_ONLY",pai),("MARKET_BLEND",alpha*pai+(1-alpha)*pm)):
                        e=pv*odd-1.0
                        z=eb[(mode,bet,edge_band(e))]
                        z["n"]+=1; z["p"]+=pv; z["y"]+=hit; z["stake"]+=100.0; z["ret"]+=ret
                        for t in THRESHOLDS:
                            if e>=t:
                                r=rb[(mode,t)]; r["tickets"]+=1; r["hit"]+=hit; r["stake"]+=100.0; r["ret"]+=ret
                for (mode,t),r in rb.items():
                    if not r["tickets"]: continue
                    g=ev[(mode,bet,t)]; g["tickets"]+=r["tickets"]; g["races"]+=1; g["hit_tickets"]+=r["hit"]; g["hit_races"]+=int(r["hit"]>0); g["stake"]+=r["stake"]; g["ret"]+=r["ret"]; g["profits"].append(r["ret"]-r["stake"])
    return trainstat,ev,eb

def rows_eval(year,alphas,ev,eb):
    market=[]; edges=[]
    for mode in ("AI_ONLY","MARKET_BLEND"):
        for bet in BET_TYPES:
            for t in THRESHOLDS:
                g=ev[(mode,bet,t)]
                market.append({"test_year":year,"mode":mode,"bet_type":bet,"alpha_market_blend":alphas.get(bet) if mode=="MARKET_BLEND" else 1.0,"edge_threshold":t,"tickets":g["tickets"],"bought_races":g["races"],"hit_tickets":g["hit_tickets"],"hit_races":g["hit_races"],"stake_yen":g["stake"],"return_yen":g["ret"],"profit_yen":g["ret"]-g["stake"],"roi_pct":100*g["ret"]/g["stake"] if g["stake"] else None,"race_hit_rate_pct":100*g["hit_races"]/g["races"] if g["races"] else None,"ticket_hit_rate_pct":100*g["hit_tickets"]/g["tickets"] if g["tickets"] else None,"max_drawdown_yen":max_drawdown(g["profits"])})
            for name,_,_ in EDGE_BANDS:
                z=eb[(mode,bet,name)]
                edges.append({"test_year":year,"mode":mode,"bet_type":bet,"alpha_market_blend":alphas.get(bet) if mode=="MARKET_BLEND" else 1.0,"edge_band":name,"tickets":z["n"],"mean_predicted_hit_pct":100*z["p"]/z["n"] if z["n"] else None,"actual_hit_pct":100*z["y"]/z["n"] if z["n"] else None,"calibration_gap_pp":100*(z["p"]-z["y"])/z["n"] if z["n"] else None,"roi_pct":100*z["ret"]/z["stake"] if z["stake"] else None})
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
    history=[]; alpha_rows=[]; market_rows=[]; edge_rows=[]
    for test_year,train_years in FOLDS:
        train=df[df["year"].isin(train_years) & (df["train_eligible"]==True) & (df["label_available"]==True)].copy()
        test=df[df["year"]==test_year].copy().reset_index(drop=True)
        races=train[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"])
        cut=max(1,int(len(races)*0.8)); fit_ids=set(races.iloc[:cut]["race_id"].astype(str)); cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
        fit=train[train["race_id"].isin(fit_ids)].copy()
        calib=train[train["race_id"].isin(cal_ids)].copy().reset_index(drop=True)
        ptest,temp=final_conf_probs(train,calib,fit,test,features,params)
        cals=archived[test_year]
        alphas=choose_alpha(history) if history else None
        stat,ev,eb=scan_year(test_year,test,ptest,cals,a.backfill_root,alphas)
        if alphas is not None:
            for b in BET_TYPES: alpha_rows.append({"test_year":test_year,"bet_type":b,"alpha_ai":alphas[b],"alpha_market":1-alphas[b],"trained_on_years":"|".join(str(y) for y in range(2023,test_year))})
            m,e=rows_eval(test_year,alphas,ev,eb); market_rows.extend(m); edge_rows.extend(e)
        history.append(stat)
        print("FAST_MARKET_YEAR_DONE "+json.dumps({"year":test_year,"evaluated":alphas is not None},separators=(",",":")),flush=True)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"learned-market-blend.csv",alpha_rows)
    write_csv(out/"market-diagnostic.csv",market_rows)
    write_csv(out/"edge-band-calibration.csv",edge_rows)
    summary={"contract":"L2_CORE_V1_FAST_MARKET_VALUATION","method":"linear probability blend between calibrated AI ticket distribution and normalized inverse-odds market distribution","ticket_calibrators_reused_from":str(a.calibrator_bins),"ticket_calibration_recomputed":False,"odds_used_in_l1_or_hit_model":False,"odds_used_only_in_market_valuation":True,"evaluation_years":[2024,2025],"2023_used_for_market_weight_training":True,"2026_locked":True,"promotion":False}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_FAST_MARKET_VALUATION_READY")

if __name__=="__main__":
    main()
