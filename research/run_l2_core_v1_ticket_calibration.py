#!/usr/bin/env python3
import argparse,csv,itertools,json,math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from run_l2_core_v1 import (
    FOLDS,BET_TYPES,THRESHOLDS,CalAgg,choose_temperature,group_indices,
    horse_metrics,load_dataset,max_drawdown,normalized_from_raw,ticket_probability,write_csv
)
from run_l2_core_v1_confidence_ab import fit_confidence,predict_confidence,meta_features
from build_l2_bet_kings_dataset_v1 import load_day,load_odds_day,decode_odds,payout_map

LOG_MIN=-8.0
LOG_MAX=0.0
LOG_BINS=160
PRIOR_STRENGTH=200.0
EDGE_BANDS=(
    ("NEG",-1e99,0.0),("0_10",0.0,0.10),("10_20",0.10,0.20),
    ("20_50",0.20,0.50),("50_100",0.50,1.0),("100_PLUS",1.0,1e99),
)
PROB_BANDS=(
    ("LT_0.01",0.0,0.0001),("0.01_0.02",0.0001,0.0002),("0.02_0.05",0.0002,0.0005),
    ("0.05_0.10",0.0005,0.001),("0.10_0.20",0.001,0.002),("0.20_0.50",0.002,0.005),
    ("0.50_1",0.005,0.01),("1_2",0.01,0.02),("2_5",0.02,0.05),("5_10",0.05,0.10),
    ("10_PLUS",0.10,1.01),
)

def parse_args():
    p=argparse.ArgumentParser(description="L2 CORE V1 market-free ticket probability calibration.")
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def fit_meta(df,X):
    m=LogisticRegression(C=1.0,solver="liblinear",max_iter=1000)
    m.fit(X,df["target_win"].astype(int).to_numpy(),sample_weight=1.0/df["field_size"].to_numpy(dtype=float))
    raw=m.decision_function(X)
    t,_=choose_temperature(df,raw)
    return m,float(t)

def crossfit_conf_probs(fit,calib,features,params):
    base=lgb.LGBMClassifier(**params)
    base.fit(fit[features],fit["target_win"].astype(int),sample_weight=1.0/fit["field_size"].astype(float))
    cv,cm=fit_confidence(fit)
    raw=np.asarray(base.booster_.predict(calib[features],raw_score=True),dtype=float)
    q=predict_confidence(cv,cm,calib)
    X=meta_features(raw,q,calib["consensus_rank_pct"].to_numpy(dtype=float))

    races=calib[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"])
    cut=max(1,len(races)//2)
    a_ids=set(races.iloc[:cut]["race_id"].astype(str)); b_ids=set(races.iloc[cut:]["race_id"].astype(str))
    if not a_ids or not b_ids: raise ValueError("crossfit split empty")
    out=np.empty(len(calib),dtype=float)
    for train_ids,test_ids in ((a_ids,b_ids),(b_ids,a_ids)):
        tr=np.asarray([i for i,r in enumerate(calib["race_id"].astype(str)) if r in train_ids],dtype=int)
        te=np.asarray([i for i,r in enumerate(calib["race_id"].astype(str)) if r in test_ids],dtype=int)
        m,t=fit_meta(calib.iloc[tr],X[tr])
        rr=m.decision_function(X[te])
        out[te]=normalized_from_raw(rr,calib.iloc[te]["race_id"].astype(str).tolist(),t)
    return out

def final_conf_probs(train,calib,fit,test,features,params):
    base0=lgb.LGBMClassifier(**params)
    base0.fit(fit[features],fit["target_win"].astype(int),sample_weight=1.0/fit["field_size"].astype(float))
    cv0,cm0=fit_confidence(fit)
    raw_cal=np.asarray(base0.booster_.predict(calib[features],raw_score=True),dtype=float)
    qcal=predict_confidence(cv0,cm0,calib)
    Xcal=meta_features(raw_cal,qcal,calib["consensus_rank_pct"].to_numpy(dtype=float))
    meta,temp=fit_meta(calib,Xcal)

    base=lgb.LGBMClassifier(**params)
    base.fit(train[features],train["target_win"].astype(int),sample_weight=1.0/train["field_size"].astype(float))
    cv,cm=fit_confidence(train)
    raw_test=np.asarray(base.booster_.predict(test[features],raw_score=True),dtype=float)
    qtest=predict_confidence(cv,cm,test)
    Xtest=meta_features(raw_test,qtest,test["consensus_rank_pct"].to_numpy(dtype=float))
    meta_raw=meta.decision_function(Xtest)
    return normalized_from_raw(meta_raw,test["race_id"].astype(str).tolist(),temp),temp

def ticket_space(nums):
    for a in nums: yield "WIN",(a,)
    for x in itertools.combinations(nums,2): yield "QUINELLA",x
    for x in itertools.permutations(nums,2): yield "EXACTA",x
    if len(nums)>=3:
        for x in itertools.combinations(nums,3): yield "TRIO",x
        for x in itertools.permutations(nums,3): yield "TRIFECTA",x

def make_race_prob(df,p):
    tmp=df.copy().reset_index(drop=True); tmp["p"]=np.asarray(p,dtype=float)
    probs={}; dates={}
    for rid,sub in tmp.groupby("race_id",sort=False):
        d={int(n):float(v) for n,v in zip(sub["horse_number"],sub["p"])}
        s=sum(d.values()); probs[str(rid)]={k:v/s for k,v in d.items()}
        dates[str(rid)]=str(sub["race_date"].iloc[0])[:10]
    return probs,dates

def bin_index(p):
    lp=math.log10(max(float(p),10**LOG_MIN))
    x=(min(max(lp,LOG_MIN),LOG_MAX)-LOG_MIN)/(LOG_MAX-LOG_MIN)
    return min(LOG_BINS-1,max(0,int(x*LOG_BINS)))

def aggregate_ticket_bins(df,pwin,root):
    racep,raced=make_race_prob(df,pwin)
    bydate=defaultdict(list)
    for rid,d in raced.items(): bydate[d].append(rid)
    agg={(b,i):{"n":0,"hits":0,"sum_p":0.0} for b in BET_TYPES for i in range(LOG_BINS)}
    root=Path(root)
    for date in sorted(bydate):
        wanted=set(bydate[date])
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in wanted:
            pack=day.get(rid)
            if pack is None: raise SystemExit(f"missing calibration race pack {rid}")
            payouts,present=payout_map(pack); p=racep[rid]; nums=sorted(p)
            for bet,sel in ticket_space(nums):
                if bet not in present: continue
                pr=ticket_probability(bet,sel,p)
                if pr is None: continue
                z=agg[(bet,bin_index(pr))]; z["n"]+=1; z["sum_p"]+=float(pr); z["hits"]+=int((bet,sel) in payouts)
    return agg

def fit_ticket_calibrators(agg):
    models={}; rows=[]
    for bet in BET_TYPES:
        xs=[]; ys=[]; ws=[]; refs=[]
        for i in range(LOG_BINS):
            z=agg[(bet,i)]
            if not z["n"]: continue
            mp=z["sum_p"]/z["n"]
            empirical=z["hits"]/z["n"]
            smooth=(z["hits"]+PRIOR_STRENGTH*mp)/(z["n"]+PRIOR_STRENGTH)
            xs.append(math.log10(max(mp,10**LOG_MIN))); ys.append(smooth); ws.append(z["n"]); refs.append((i,z,mp,empirical,smooth))
        if len(xs)<3: raise SystemExit(f"not enough calibration bins for {bet}")
        iso=IsotonicRegression(increasing=True,out_of_bounds="clip",y_min=1e-12,y_max=1.0)
        iso.fit(np.asarray(xs),np.asarray(ys),sample_weight=np.asarray(ws,dtype=float))
        models[bet]=iso
        for (i,z,mp,emp,smooth),x in zip(refs,xs):
            rows.append({
                "bet_type":bet,"bin_index":i,"tickets":z["n"],"hits":z["hits"],"mean_raw_p":mp,
                "empirical_hit_rate":emp,"smoothed_hit_rate":smooth,"isotonic_hit_rate":float(iso.predict([x])[0]),
            })
    return models,rows

def calibrate_race_market(models,odds,p):
    bybet=defaultdict(list)
    for (bet,sel),odd in odds.items():
        if bet not in BET_TYPES: continue
        raw=ticket_probability(bet,sel,p)
        if raw is None: continue
        q=float(models[bet].predict([math.log10(max(raw,10**LOG_MIN))])[0])
        bybet[bet].append([sel,float(odd),float(raw),q])
    out={}
    for bet,items in bybet.items():
        sq=sum(x[3] for x in items)
        if sq<=0: sq=sum(x[2] for x in items)
        for sel,odd,raw,q in items:
            cal=(q/sq) if sq>0 else raw
            out[(bet,sel)]=(odd,raw,cal)
    return out

def band_name(x,bands):
    for name,lo,hi in bands:
        if lo<=x<hi: return name
    return bands[-1][0]

def evaluate(test_year,df,pwin,models,root):
    racep,raced=make_race_prob(df,pwin)
    bydate=defaultdict(list)
    for rid,d in raced.items(): bydate[d].append(rid)
    cals={(mode,b):CalAgg() for mode in ("RAW_CONF","TICKET_CAL") for b in BET_TYPES}
    strat={(mode,b,t):{"tickets":0,"stake":0.0,"ret":0.0,"hit_tickets":0,"races":0,"hit_races":0,"profits":[]} for mode in ("RAW_CONF","TICKET_CAL") for b in BET_TYPES for t in THRESHOLDS}
    edge={(mode,b,name):{"n":0,"p":0.0,"y":0,"stake":0.0,"ret":0.0} for mode in ("RAW_CONF","TICKET_CAL") for b in BET_TYPES for name,_,_ in EDGE_BANDS}
    prob={(mode,b,name):{"n":0,"p":0.0,"y":0} for mode in ("RAW_CONF","TICKET_CAL") for b in BET_TYPES for name,_,_ in PROB_BANDS}
    root=Path(root)
    for date in sorted(bydate):
        wanted=set(bydate[date])
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        oddsday=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in wanted:
            pack=day.get(rid); oddsrec=oddsday.get(rid)
            if pack is None or oddsrec is None: raise SystemExit(f"missing test market row {rid}")
            payouts,present=payout_map(pack); odds=decode_odds(oddsrec); p=racep[rid]
            mapped=calibrate_race_market(models,odds,p)
            raceacc={(mode,b,t):{"tickets":0,"stake":0.0,"ret":0.0,"hit":0} for mode in ("RAW_CONF","TICKET_CAL") for b in BET_TYPES for t in THRESHOLDS}
            for (bet,sel),(odd,raw,calp) in mapped.items():
                if bet not in present: continue
                hit=int((bet,sel) in payouts); ret=float(payouts.get((bet,sel),0.0))
                for mode,ph in (("RAW_CONF",raw),("TICKET_CAL",calp)):
                    cals[(mode,bet)].add(ph,hit)
                    eb=band_name(ph*odd-1.0,EDGE_BANDS); z=edge[(mode,bet,eb)]
                    z["n"]+=1; z["p"]+=ph; z["y"]+=hit; z["stake"]+=100.0; z["ret"]+=ret
                    pb=band_name(ph,PROB_BANDS); z2=prob[(mode,bet,pb)]
                    z2["n"]+=1; z2["p"]+=ph; z2["y"]+=hit
                    for t in THRESHOLDS:
                        if ph*odd-1.0>=t:
                            a=raceacc[(mode,bet,t)]; a["tickets"]+=1; a["stake"]+=100.0; a["ret"]+=ret; a["hit"]+=hit
            for key,a in raceacc.items():
                if not a["tickets"]: continue
                g=strat[key]; g["tickets"]+=a["tickets"]; g["stake"]+=a["stake"]; g["ret"]+=a["ret"]; g["hit_tickets"]+=a["hit"]; g["races"]+=1; g["hit_races"]+=int(a["hit"]>0); g["profits"].append(a["ret"]-a["stake"])
    calrows=[]; marketrows=[]; edgerows=[]; probrows=[]
    for mode in ("RAW_CONF","TICKET_CAL"):
        for bet in BET_TYPES:
            calrows.append({"mode":mode,"test_year":test_year,"bet_type":bet,**cals[(mode,bet)].result()})
            for t in THRESHOLDS:
                g=strat[(mode,bet,t)]
                marketrows.append({"mode":mode,"test_year":test_year,"bet_type":bet,"edge_threshold":t,"tickets":g["tickets"],"bought_races":g["races"],"hit_tickets":g["hit_tickets"],"hit_races":g["hit_races"],"stake_yen":g["stake"],"return_yen":g["ret"],"profit_yen":g["ret"]-g["stake"],"roi_pct":100*g["ret"]/g["stake"] if g["stake"] else None,"race_hit_rate_pct":100*g["hit_races"]/g["races"] if g["races"] else None,"ticket_hit_rate_pct":100*g["hit_tickets"]/g["tickets"] if g["tickets"] else None,"max_drawdown_yen":max_drawdown(g["profits"])})
            for name,_,_ in EDGE_BANDS:
                z=edge[(mode,bet,name)]
                edgerows.append({"mode":mode,"test_year":test_year,"bet_type":bet,"edge_band":name,"tickets":z["n"],"mean_predicted_hit_pct":100*z["p"]/z["n"] if z["n"] else None,"actual_hit_pct":100*z["y"]/z["n"] if z["n"] else None,"calibration_gap_pp":100*(z["p"]-z["y"])/z["n"] if z["n"] else None,"roi_pct":100*z["ret"]/z["stake"] if z["stake"] else None})
            for name,_,_ in PROB_BANDS:
                z=prob[(mode,bet,name)]
                probrows.append({"mode":mode,"test_year":test_year,"bet_type":bet,"probability_band_pct":name,"tickets":z["n"],"mean_predicted_hit_pct":100*z["p"]/z["n"] if z["n"] else None,"actual_hit_pct":100*z["y"]/z["n"] if z["n"] else None,"calibration_gap_pp":100*(z["p"]-z["y"])/z["n"] if z["n"] else None})
    return calrows,marketrows,edgerows,probrows

def main():
    a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L2_CORE_V1": raise SystemExit("wrong contract")
    if contract["cost_policy"]["github_standard_cpu_only"] is not True or contract["cost_policy"]["gpu"] is not False: raise SystemExit("cost guard drift")
    manifest,df=load_dataset(a.dataset_dir)
    features=list(manifest["feature_columns"])
    for req in ("vote2","vote3","target_top3"):
        if req not in df.columns: raise SystemExit(f"missing field {req}")
    for c in features+["vote2","vote3","top1_votes","consensus_rank","consensus_rank_pct","field_size"]:
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
    df["year"]=df["year"].astype(int); df["race_id"]=df["race_id"].astype(str)
    params=dict(objective="binary",n_estimators=260,learning_rate=0.04,num_leaves=31,min_child_samples=50,subsample=0.9,colsample_bytree=0.9,reg_lambda=1.0,random_state=20261001,n_jobs=2,verbosity=-1)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    folds=[]; caltables=[]; ticketcal=[]; market=[]; edge=[]; prob=[]
    for test_year,train_years in FOLDS:
        train=df[df["year"].isin(train_years) & (df["train_eligible"]==True) & (df["label_available"]==True)].copy()
        test=df[df["year"]==test_year].copy().reset_index(drop=True)
        races=train[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"])
        cut=max(1,int(len(races)*0.8)); fit_ids=set(races.iloc[:cut]["race_id"].astype(str)); cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
        fit=train[train["race_id"].isin(fit_ids)].copy()
        calib=train[train["race_id"].isin(cal_ids)].copy().reset_index(drop=True)

        p_cross=crossfit_conf_probs(fit,calib,features,params)
        agg=aggregate_ticket_bins(calib,p_cross,a.backfill_root)
        models,table=fit_ticket_calibrators(agg)
        for r in table: r["test_year"]=test_year
        caltables.extend(table)

        p_test,temp=final_conf_probs(train,calib,fit,test,features,params)
        hm=horse_metrics(test,p_test)
        folds.append({"test_year":test_year,"train_years":"|".join(map(str,train_years)),"fit_races":len(fit_ids),"ticket_calibration_races":len(cal_ids),"final_conf_temperature":temp,**hm})
        c,m,e,p=evaluate(test_year,test,p_test,models,a.backfill_root)
        ticketcal.extend(c); market.extend(m); edge.extend(e); prob.extend(p)
        print("L2_TICKET_CAL_FOLD_DONE "+json.dumps({"test_year":test_year,"ticket_calibration_races":len(cal_ids),"win_log_loss":hm.get("win_log_loss")},separators=(",",":")),flush=True)

    write_csv(out/"fold-metrics.csv",folds)
    write_csv(out/"calibrator-bins.csv",caltables)
    write_csv(out/"ticket-calibration-raw-vs-cal.csv",ticketcal)
    write_csv(out/"market-diagnostic-raw-vs-cal.csv",market)
    write_csv(out/"edge-band-calibration-raw-vs-cal.csv",edge)
    write_csv(out/"probability-band-calibration-raw-vs-cal.csv",prob)
    summary={"contract":"L2_CORE_V1_TICKET_CALIBRATION_RESULT","method":"market-free per-bet isotonic on cross-fit prior-race ticket probabilities, then within-race renormalization","odds_used_for_calibration":False,"odds_used_only_for_final_valuation":True,"prior_strength":PRIOR_STRENGTH,"2026_locked":True,"folds":folds,"promotion":False}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_CORE_V1_TICKET_CALIBRATION_READY")

if __name__=="__main__":
    main()
