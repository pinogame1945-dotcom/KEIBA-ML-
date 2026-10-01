#!/usr/bin/env python3
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from run_l2_core_v1 import FOLDS,THRESHOLDS,load_dataset,max_drawdown
from run_l2_core_v1_ticket_calibration import final_conf_probs
from run_l2_core_v1_market_valuation_fast import BET_TYPES,load_archived_calibrators,race_probabilities,ticket_p
from run_l2_core_v1_divergence_reliability_fast import learn_factors,lookup_factor,cell_key
from build_l2_bet_kings_dataset_v1 import iter_decoded_odds,load_day,load_odds_day,payout_map

BET_INDEX={b:i for i,b in enumerate(BET_TYPES)}
EDGE_BANDS=(("NEG",-1e99,0.0),("0_10",0.0,0.10),("10_20",0.10,0.20),("20_50",0.20,0.50),("50_100",0.50,1.0),("100_PLUS",1.0,1e99))

def parse_args():
    p=argparse.ArgumentParser(description="Fast shared final buy-value probability model, strict walk-forward.")
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--calibrator-bins",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def cliplog10(x):
    return min(0.0,max(-8.0,math.log10(max(float(x),1e-12))))/8.0

def cliplog2ratio(a,b):
    return min(8.0,max(-8.0,math.log2(max(a,1e-12)/max(b,1e-12))))/8.0

def feat_row(bet,pai,pcorr,pmkt,odd,rf):
    # One model, but bet-specific blocks allow the same signal to mean different things by ticket type.
    base=np.array([
        1.0,
        cliplog10(pai),
        cliplog10(pcorr),
        cliplog10(pmkt),
        cliplog2ratio(pai,pmkt),
        min(1.0,max(0.0,math.log10(max(float(odd),1.0))/5.0)),
        min(1.0,max(-1.0,math.log2(max(rf,1e-6))/4.0)),
    ],dtype=np.float64)
    out=np.zeros(len(BET_TYPES)*len(base),dtype=np.float64)
    j=BET_INDEX[bet]*len(base)
    out[j:j+len(base)]=base
    return out

def edge_band(x):
    for name,lo,hi in EDGE_BANDS:
        if lo<=x<hi: return name
    return "100_PLUS"

def init_eval():
    return {(mode,b,t):{"tickets":0,"races":0,"hit_tickets":0,"hit_races":0,"stake":0.0,"ret":0.0,"profits":[]} for mode in ("AI_ONLY","DIVERGENCE_CORR","BUY_MODEL") for b in BET_TYPES for t in THRESHOLDS}

def init_edge():
    return {(mode,b,name):{"n":0,"p":0.0,"y":0,"stake":0.0,"ret":0.0} for mode in ("AI_ONLY","DIVERGENCE_CORR","BUY_MODEL") for b in BET_TYPES for name,_,_ in EDGE_BANDS}

def add_eval(ev,eb,bet,mode,vals):
    raceacc={t:{"tickets":0,"hit":0,"stake":0.0,"ret":0.0} for t in THRESHOLDS}
    for odd,pv,ret,hit in vals:
        e=pv*odd-1.0
        z=eb[(mode,bet,edge_band(e))]
        z["n"]+=1; z["p"]+=pv; z["y"]+=hit; z["stake"]+=100.0; z["ret"]+=ret
        for t in THRESHOLDS:
            if e>=t:
                r=raceacc[t]; r["tickets"]+=1; r["hit"]+=hit; r["stake"]+=100.0; r["ret"]+=ret
    for t,r in raceacc.items():
        if not r["tickets"]: continue
        g=ev[(mode,bet,t)]
        g["tickets"]+=r["tickets"]; g["races"]+=1; g["hit_tickets"]+=r["hit"]; g["hit_races"]+=int(r["hit"]>0)
        g["stake"]+=r["stake"]; g["ret"]+=r["ret"]; g["profits"].append(r["ret"]-r["stake"])

def scan_year(year,test,pwin,calibrators,root,div_pack,model,do_eval,do_train):
    racep,raced=race_probabilities(test,pwin)
    bydate=defaultdict(list)
    for rid,d in raced.items(): bydate[d].append(rid)
    yearly=defaultdict(lambda:{"n":0,"hits":0,"sum_ai":0.0})
    ev=init_eval(); eb=init_edge()
    root=Path(root)
    trained_rows=0

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
                q=calibrators[bet].predict_one(raw); m=1.0/float(odd)
                ret=float(payouts.get((bet,nums),0.0)); hit=int(ret>0)
                items[bet].append([float(odd),q,m,ret,hit])
                sumq[bet]+=q; summ[bet]+=m

            for bet,arr in items.items():
                if not arr or sumq[bet]<=0 or summ[bet]<=0: continue
                base=[]
                corr_raw=[]
                factors,bd,bb=div_pack if div_pack is not None else ({},defaultdict(dict),defaultdict(dict))
                for odd,q,m,ret,hit in arr:
                    pai=q/sumq[bet]; pm=m/summ[bet]
                    z=yearly[cell_key(bet,pai,pm)]
                    z["n"]+=1; z["hits"]+=hit; z["sum_ai"]+=pai
                    rf=lookup_factor(bet,pai,pm,factors,bd,bb) if div_pack is not None else 1.0
                    base.append([odd,pai,pm,ret,hit,rf])
                    corr_raw.append(pai*rf)
                sc=sum(corr_raw) or 1.0
                pcorr=[x/sc for x in corr_raw]

                X=np.vstack([feat_row(bet,x[1],pc,x[2],x[0],x[5]) for x,pc in zip(base,pcorr)])
                y=np.asarray([x[4] for x in base],dtype=np.int32)

                if do_eval:
                    pred=model.predict_proba(X)[:,1]
                    sp=float(pred.sum())
                    if sp<=0: pred=np.full(len(pred),1.0/len(pred))
                    else: pred=pred/sp
                    add_eval(ev,eb,bet,"AI_ONLY",[(x[0],x[1],x[3],x[4]) for x in base])
                    add_eval(ev,eb,bet,"DIVERGENCE_CORR",[(x[0],pc,x[3],x[4]) for x,pc in zip(base,pcorr)])
                    add_eval(ev,eb,bet,"BUY_MODEL",[(x[0],float(pp),x[3],x[4]) for x,pp in zip(base,pred)])

                if do_train:
                    if not hasattr(model,"classes_"):
                        model.partial_fit(X,y,classes=np.array([0,1],dtype=np.int32))
                    else:
                        model.partial_fit(X,y)
                    trained_rows+=len(y)

    return yearly,ev,eb,trained_rows

def output_rows(year,ev,eb):
    market=[]; edges=[]
    for mode in ("AI_ONLY","DIVERGENCE_CORR","BUY_MODEL"):
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
    model=SGDClassifier(loss="log_loss",penalty="l2",alpha=1e-6,learning_rate="optimal",average=True,random_state=20261001)
    div_hist=[]; market_rows=[]; edge_rows=[]; train_rows=[]

    for test_year,train_years in FOLDS:
        train=df[df["year"].isin(train_years) & (df["train_eligible"]==True) & (df["label_available"]==True)].copy()
        test=df[df["year"]==test_year].copy().reset_index(drop=True)
        races=train[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"])
        cut=max(1,int(len(races)*0.8)); fit_ids=set(races.iloc[:cut]["race_id"].astype(str)); cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
        fit=train[train["race_id"].isin(fit_ids)].copy()
        calib=train[train["race_id"].isin(cal_ids)].copy().reset_index(drop=True)
        ptest,_=final_conf_probs(train,calib,fit,test,features,params)

        div_pack=None
        if div_hist:
            factors,bd,bb,_=learn_factors(div_hist)
            div_pack=(factors,bd,bb)

        do_eval=hasattr(model,"classes_")
        yearly,ev,eb,ntrain=scan_year(test_year,test,ptest,archived[test_year],a.backfill_root,div_pack,model,do_eval,True)
        if do_eval:
            m,e=output_rows(test_year,ev,eb); market_rows.extend(m); edge_rows.extend(e)
        div_hist.append(yearly)
        train_rows.append({"year":test_year,"tickets_online_trained":ntrain,"evaluated_before_training":do_eval})
        print("BUY_MODEL_YEAR_DONE "+json.dumps({"year":test_year,"evaluated":do_eval,"trained_rows":ntrain},separators=(",",":")),flush=True)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"training-progress.csv",train_rows)
    write_csv(out/"market-diagnostic.csv",market_rows)
    write_csv(out/"edge-band-calibration.csv",edge_rows)
    coef=[]
    if hasattr(model,"coef_"):
        width=7
        names=["bias","log_ai","log_corr","log_market","log_ai_market_ratio","log_odds","log_reliability"]
        for bet in BET_TYPES:
            j=BET_INDEX[bet]*width
            for k,n in enumerate(names):
                coef.append({"bet_type":bet,"feature":n,"coefficient":float(model.coef_[0,j+k])})
    write_csv(out/"model-coefficients.csv",coef)
    summary={"contract":"L2_CORE_V1_SHARED_BUY_VALUE","model":"single online SGD logistic hit-probability model with bet-specific feature blocks","walk_forward":"train 2023 -> test 2024; update with 2024 -> test 2025","inputs":["calibrated_ai_p","divergence_corrected_p","normalized_market_p","ai_market_ratio","odds","reliability_factor","bet_type"],"target":"ticket_hit","profit_not_used_as_training_label":True,"probabilities_renormalized_within_race_bet":True,"odds_used_only_in_final_L2_market_value_layer":True,"2026_locked":True,"promotion":False}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_SHARED_BUY_VALUE_READY")

if __name__=="__main__":
    main()
