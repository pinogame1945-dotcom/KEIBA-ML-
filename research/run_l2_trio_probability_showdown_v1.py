#!/usr/bin/env python3
import argparse,gc,itertools,json,math,time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import load_day,payout_map
from run_l2_trio_probability_v2 import (
    load_horse_dataset,attach_win_market,attach_outsider,prepare_l175,
    softmax_by_race,choose_temperature
)
from run_l2_trio_market_residual_v3 import (
    build_year_matrix,load_matrix,group_layout
)

YEARS=(2022,2023,2024,2025)
FOLDS=((2023,(2022,)),(2024,(2022,2023)),(2025,(2022,2023,2024)))
EPS=1e-12
STAKE=100.0
TOPKS=(1,4,8,12,20)
ROI_KS=(4,8,12,20)
ALPHA_GRID=(0.50,0.60,0.70,0.80,0.90,1.00)
BETA_GRID=(0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00,1.20,1.50)

# Strictly fundamental Seven-King features. No market, Outsider or existing p3.
POS_FEATURES=(
    "king_rank_pct","king_mean_rank_pct","king_rank_std_pct","king_best_rank_pct","king_worst_rank_pct",
    "king_top1_vote_share","king_top3_support_share","king_top6_support_share",
    "king_probability_mean","king_probability_std",
    "race_entropy","race_top3_probability_sum","race_top1_top2_gap",
)

BANNED_DIRECT_SUBSTRINGS=(
    "market","signed_gap","abs_gap","dissent","outsider","p3","alignment"
)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--outsider-predictions",required=True)
    p.add_argument("--matrix-cache",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)

def ordered_top3_map(df,backfill_root):
    root=Path(backfill_root)
    bydate=defaultdict(list)
    for rid,date in df[["race_id","race_date"]].drop_duplicates().itertuples(index=False):
        bydate[str(date)[:10]].append(str(rid))
    out={}; bad=defaultdict(int)
    for di,(date,rids) in enumerate(sorted(bydate.items()),1):
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",set(rids))
        for rid in rids:
            pack=day.get(rid)
            if pack is None:
                bad["missing_pack"]+=1; continue
            payouts,present=payout_map(pack)
            winners=[tuple(map(int,k[1])) for k,v in payouts.items() if k[0]=="TRIFECTA" and float(v)>0]
            winners=list(dict.fromkeys(winners))
            if len(winners)!=1 or len(winners[0])!=3 or len(set(winners[0]))!=3:
                bad["nonunique_or_missing_trifecta"]+=1; continue
            out[rid]=winners[0]
        if di%100==0:
            print(f"SHOWDOWN_ORDERED_LABEL_PROGRESS dates={di}/{len(bydate)}",flush=True)
    return out,dict(bad)

def add_position_labels(df,ordered):
    z=df.copy()
    z["target_pos1"]=np.nan; z["target_pos2"]=np.nan; z["target_pos3"]=np.nan
    for idx,r in z.iterrows():
        t=ordered.get(str(r["race_id"]))
        if t is None: continue
        no=int(r["horse_number"])
        z.at[idx,"target_pos1"]=int(no==t[0])
        z.at[idx,"target_pos2"]=int(no==t[1])
        z.at[idx,"target_pos3"]=int(no==t[2])
    z["race_weight"]=1.0/z["field_size"].astype(float).clip(lower=1.0)
    return z

def direct_feature_indices(meta):
    names=list(meta["feature_names"])
    keep=[]
    for i,n in enumerate(names):
        low=n.lower()
        if any(b in low for b in BANNED_DIRECT_SUBSTRINGS):
            continue
        keep.append(i)
    if not keep:
        raise SystemExit("no direct fundamental features")
    return keep,[names[i] for i in keep]

def race_norm_from_log(logv,ri):
    ri=np.asarray(ri,dtype=np.int64); logv=np.asarray(logv,dtype=np.float64)
    starts,counts=group_layout(ri)
    mx=np.maximum.reduceat(logv,starts)
    ex=np.exp(np.clip(logv-np.repeat(mx,counts),-80,80))
    sm=np.add.reduceat(ex,starts)
    return ex/np.repeat(sm,counts)

def temperature_scale(p,ri,temp):
    return race_norm_from_log(np.log(np.clip(np.asarray(p,dtype=float),EPS,None))/max(float(temp),1e-6),ri)

def blend_market_ai(q,p,ri,beta):
    q=np.clip(np.asarray(q,dtype=float),EPS,None)
    p=np.clip(np.asarray(p,dtype=float),EPS,None)
    b=float(beta)
    return race_norm_from_log((1.0-b)*np.log(q)+b*np.log(p),ri)

def race_logloss(y,p,ri):
    y=np.asarray(y,dtype=np.uint8); p=np.asarray(p,dtype=float); ri=np.asarray(ri,dtype=np.int64)
    starts,counts=group_layout(ri); vals=[]
    for s,c in zip(starts,counts):
        yy=y[s:s+c]; pos=np.where(yy==1)[0]
        if len(pos)==1: vals.append(-math.log(max(float(p[s+pos[0]]),EPS)))
    return float(np.mean(vals)) if vals else None

def probability_metrics(y,p,ri):
    y=np.asarray(y,dtype=np.uint8); p=np.asarray(p,dtype=float); ri=np.asarray(ri,dtype=np.int64)
    starts,counts=group_layout(ri)
    losses=[]; briers=[]; hits={k:0 for k in TOPKS}; truep=[]; masserr=0.0
    bins=[{"w":0.0,"p":0.0,"y":0.0} for _ in range(15)]
    for s,c in zip(starts,counts):
        e=s+c; yy=y[s:e]; pp=p[s:e]
        pos=np.where(yy==1)[0]
        if len(pos)!=1: continue
        win=int(pos[0]); pw=max(float(pp[win]),EPS)
        losses.append(-math.log(pw)); briers.append(float(np.sum((pp-yy)**2))); truep.append(pw)
        order=np.argsort(-pp,kind="stable")
        for k in TOPKS: hits[k]+=int(win in set(order[:min(k,len(order))]))
        masserr=max(masserr,abs(float(pp.sum())-1.0))
        rw=1.0/len(pp)
        for prob,target in zip(pp,yy):
            bi=min(int(max(float(prob),0.0)*15),14)
            bins[bi]["w"]+=rw; bins[bi]["p"]+=rw*float(prob); bins[bi]["y"]+=rw*float(target)
    races=len(losses); totalw=sum(b["w"] for b in bins); ece=0.0
    for b in bins:
        if b["w"]>0 and totalw>0:
            ece+=(b["w"]/totalw)*abs(b["p"]/b["w"]-b["y"]/b["w"])
    out={
        "races":races,
        "race_log_loss":float(np.mean(losses)) if losses else None,
        "multiclass_brier":float(np.mean(briers)) if briers else None,
        "ticket_ece_15bin":ece,
        "mean_true_ticket_probability":float(np.mean(truep)) if truep else None,
        "probability_mass_max_error":masserr,
    }
    for k in TOPKS: out[f"winner_top{k}_pct"]=100*hits[k]/races if races else None
    return out

def select_beta(y,q,p,ri):
    best=(0.0,float("inf")); rows=[]
    for b in BETA_GRID:
        mix=blend_market_ai(q,p,ri,b)
        ll=race_logloss(y,mix,ri)
        rows.append((b,ll))
        if ll is not None and ll<best[1]: best=(float(b),float(ll))
    return best,rows

def discounted_probs(p,comb,a2,a3):
    q=np.clip(np.asarray(p,dtype=np.float64),EPS,None); q=q/q.sum()
    r=np.power(q,float(a2)); r=r/r.sum()
    s=np.power(q,float(a3)); s=s/s.sum()
    c=np.asarray(comb,dtype=np.int64); a,b,d=c[:,0],c[:,1],c[:,2]
    def ordp(i,j,k):
        den2=np.maximum(1.0-r[i],EPS)
        den3=np.maximum(1.0-s[i]-s[j],EPS)
        return q[i]*(r[j]/den2)*(s[k]/den3)
    v=(ordp(a,b,d)+ordp(a,d,b)+ordp(b,a,d)+ordp(b,d,a)+ordp(d,a,b)+ordp(d,b,a))
    sm=float(v.sum())
    return v/sm if sm>0 else np.full(len(v),1.0/len(v))

def horse_groups(df,year):
    return {str(rid):g.sort_values(["consensus_rank","horse_number","horse_id"]).reset_index(drop=True)
            for rid,g in df[df["year"]==year].groupby("race_id",sort=False)}

def race_meta(cache,year):
    return pd.read_csv(Path(cache)/str(year)/"races.csv").sort_values("race_index").reset_index(drop=True)

def selected_race_mask(ri,indices):
    idx=set(int(x) for x in indices)
    return np.fromiter((int(x) in idx for x in np.asarray(ri,dtype=int)),dtype=bool,count=len(ri))

def fit_discount_alphas(year,df,cache,loaded,fit_race_indices):
    meta,X,y,ri,qm,bk,bw=loaded[year]
    rm=race_meta(cache,year); groups=horse_groups(df,year)
    keep=set(int(x) for x in fit_race_indices)
    grid=[]
    best=(1.0,1.0,float("inf"))
    yarr=np.asarray(y); riarr=np.asarray(ri)
    starts,counts=group_layout(riarr)
    for a2 in ALPHA_GRID:
        for a3 in ALPHA_GRID:
            vals=[]
            for rix,(st,cnt) in enumerate(zip(starts,counts)):
                if rix not in keep: continue
                rid=str(rm.iloc[rix]["race_id"]); g=groups[rid]; n=len(g)
                comb=np.asarray(list(itertools.combinations(range(n),3)),dtype=np.int16)
                p=np.clip(g["king_probability_mean"].to_numpy(dtype=float),0,None)
                if p.sum()<=0:p=np.full(n,1.0/n)
                probs=discounted_probs(p,comb,a2,a3)
                pos=np.where(yarr[st:st+cnt]==1)[0]
                if len(pos)==1: vals.append(-math.log(max(float(probs[int(pos[0])]),EPS)))
            ll=float(np.mean(vals)) if vals else float("inf")
            grid.append({"alpha2":a2,"alpha3":a3,"fit_logloss":ll})
            if ll<best[2]: best=(float(a2),float(a3),ll)
    return best,grid

def discounted_distribution(year,df,cache,loaded,a2,a3):
    meta,X,y,ri,qm,bk,bw=loaded[year]; rm=race_meta(cache,year); groups=horse_groups(df,year)
    parts=[]
    for rix,row in rm.iterrows():
        rid=str(row["race_id"]); g=groups[rid]; n=len(g)
        comb=np.asarray(list(itertools.combinations(range(n),3)),dtype=np.int16)
        p=np.clip(g["king_probability_mean"].to_numpy(dtype=float),0,None)
        if p.sum()<=0:p=np.full(n,1.0/n)
        parts.append(discounted_probs(p,comb,a2,a3).astype(np.float32))
    return np.concatenate(parts)

def fit_position_models(df,train_years,allowed_recent_ids=None,recent_year=None):
    q=df[df["year"].isin(train_years)].copy()
    if allowed_recent_ids is not None and recent_year is not None:
        mask=(q["year"]!=recent_year) | q["race_id"].astype(str).isin(set(map(str,allowed_recent_ids)))
        q=q[mask].copy()
    q=q[q["target_pos1"].notna()].copy()
    if q.empty: raise SystemExit("no position training rows")
    models=[]
    for col in ("target_pos1","target_pos2","target_pos3"):
        yy=q[col].astype(int)
        pos=max(1,int(yy.sum())); neg=max(1,len(yy)-pos)
        spw=float(min(30.0,max(1.0,math.sqrt(neg/pos))))
        m=lgb.LGBMClassifier(
            objective="binary",n_estimators=180,learning_rate=0.04,num_leaves=31,max_bin=63,
            min_child_samples=100,subsample=0.9,subsample_freq=1,colsample_bytree=0.9,
            reg_lambda=2.0,scale_pos_weight=spw,random_state=20261004,n_jobs=-1,
            verbosity=-1,force_col_wise=True
        )
        m.fit(q[list(POS_FEATURES)],yy,sample_weight=q["race_weight"].to_numpy(dtype=float))
        models.append(m)
    return models,len(q)

def position_distribution(year,df,cache,loaded,models,return_horse_rows=False):
    meta,X,y,ri,qm,bk,bw=loaded[year]; rm=race_meta(cache,year); groups=horse_groups(df,year)
    parts=[]; horse_rows=[]
    for rix,row in rm.iterrows():
        rid=str(row["race_id"]); g=groups[rid].copy(); n=len(g)
        probs=[]
        for m in models:
            v=np.asarray(m.predict_proba(g[list(POS_FEATURES)])[:,1],dtype=float)
            sm=float(v.sum()); v=v/sm if sm>0 else np.full(n,1.0/n)
            probs.append(v)
        p1,p2,p3=probs
        comb=np.asarray(list(itertools.combinations(range(n),3)),dtype=np.int16)
        a,b,c=comb[:,0],comb[:,1],comb[:,2]
        v=(p1[a]*p2[b]*p3[c]+p1[a]*p2[c]*p3[b]+
           p1[b]*p2[a]*p3[c]+p1[b]*p2[c]*p3[a]+
           p1[c]*p2[a]*p3[b]+p1[c]*p2[b]*p3[a])
        sm=float(v.sum()); v=v/sm if sm>0 else np.full(len(v),1.0/len(v))
        parts.append(v.astype(np.float32))
        if return_horse_rows:
            ptop=np.clip(p1+p2+p3,0.0,1.0)
            for j,rr in g.reset_index(drop=True).iterrows():
                rank=int(rr["consensus_rank"])
                band="1-6" if rank<=6 else ("7-12" if rank<=12 else "13-18")
                horse_rows.append({
                    "year":year,"race_id":rid,"horse_number":int(rr["horse_number"]),
                    "consensus_rank":rank,"rank_band":band,
                    "p_top3":float(ptop[j]),"actual_top3":int(rr["target_top3"]==1)
                })
    return np.concatenate(parts),horse_rows

def direct_training_parts(loaded,train_years,feature_idx,recent_year=None,recent_fit_max=None):
    Xs=[]; ys=[]; ws=[]
    for y in train_years:
        meta,X,yy,ri,qm,bk,bw=loaded[y]
        ria=np.asarray(ri,dtype=np.int64)
        mask=np.ones(len(ria),dtype=bool)
        if recent_year is not None and y==recent_year and recent_fit_max is not None:
            mask=ria<int(recent_fit_max)
        counts=np.bincount(ria[mask])
        denom=np.maximum(counts[ria[mask]],1)
        Xs.append(np.asarray(X)[mask][:,feature_idx].astype(np.float32,copy=False))
        ys.append(np.asarray(yy)[mask].astype(np.uint8,copy=False))
        ws.append((1.0/denom).astype(np.float32,copy=False))
    return np.concatenate(Xs),np.concatenate(ys),np.concatenate(ws)

def fit_direct(loaded,train_years,feature_idx,recent_year=None,recent_fit_max=None):
    X,y,w=direct_training_parts(loaded,train_years,feature_idx,recent_year,recent_fit_max)
    model=lgb.LGBMClassifier(
        objective="binary",n_estimators=130,learning_rate=0.04,num_leaves=31,max_bin=63,
        min_child_samples=250,subsample=0.9,subsample_freq=1,colsample_bytree=0.85,
        reg_lambda=4.0,reg_alpha=0.2,random_state=20261004,n_jobs=-1,verbosity=-1,force_col_wise=True
    )
    model.fit(X,y,sample_weight=w)
    rows=len(y)
    del X,y,w; gc.collect()
    return model,rows

def direct_distribution(model,loaded,year,feature_idx,temp=1.0):
    meta,X,y,ri,qm,bk,bw=loaded[year]
    raw=np.asarray(model.booster_.predict(np.asarray(X)[:,feature_idx],raw_score=True),dtype=float)
    return softmax_by_race(raw,ri,temp)

def choose_temp_from_probs(y,p,ri):
    best=(1.0,float("inf")); rows=[]
    for t in np.exp(np.linspace(math.log(0.45),math.log(2.2),17)):
        pp=temperature_scale(p,ri,float(t)); ll=race_logloss(y,pp,ri)
        rows.append((float(t),ll))
        if ll is not None and ll<best[1]:best=(float(t),float(ll))
    return best,rows

def payout_vectors(backfill_root,cache,year):
    rm=race_meta(cache,year); root=Path(backfill_root); bydate=defaultdict(list)
    for _,r in rm.iterrows(): bydate[str(r["race_date"])[:10]].append((int(r["race_index"]),str(r["race_id"])))
    out=np.full(len(rm),np.nan,dtype=float)
    for date,items in sorted(bydate.items()):
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",set(rid for _,rid in items))
        for rix,rid in items:
            pack=day.get(rid)
            if pack is None: continue
            payouts,present=payout_map(pack)
            vals=[float(v) for k,v in payouts.items() if k[0]=="TRIO" and float(v)>0]
            vals=list(dict.fromkeys(vals))
            if len(vals)==1: out[rix]=vals[0]
    return out

def roi_topk(y,p,ri,payout_by_race,k):
    y=np.asarray(y,dtype=np.uint8); p=np.asarray(p,dtype=float); ri=np.asarray(ri,dtype=np.int64)
    starts,counts=group_layout(ri)
    races=hits=tickets=0; ret=0.0; payouts_hit=[]
    for rix,(s,c) in enumerate(zip(starts,counts)):
        if rix>=len(payout_by_race) or not np.isfinite(payout_by_race[rix]): continue
        yy=y[s:s+c]; pos=np.where(yy==1)[0]
        if len(pos)!=1: continue
        kk=min(int(k),c); order=np.argsort(-p[s:s+c],kind="stable")[:kk]
        hit=int(int(pos[0]) in set(order))
        races+=1; tickets+=kk; hits+=hit
        if hit:
            ret+=float(payout_by_race[rix]); payouts_hit.append(float(payout_by_race[rix]))
    stake=STAKE*tickets
    return {
        "races":races,"tickets":tickets,"avg_tickets_per_race":tickets/races if races else None,
        "hit_races":hits,"hit_rate_pct":100*hits/races if races else None,
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "avg_payout_when_hit_yen":float(np.mean(payouts_hit)) if payouts_hit else None
    }

def calibration_band(rows):
    z=pd.DataFrame(rows); out=[]
    if z.empty:return out
    for (year,band),g in z.groupby(["year","rank_band"]):
        p=np.clip(g["p_top3"].to_numpy(dtype=float),EPS,1-EPS)
        y=g["actual_top3"].to_numpy(dtype=int)
        out.append({
            "year":year,"rank_band":band,"horses":len(g),"actual_top3_pct":100*float(y.mean()),
            "mean_predicted_top3_pct":100*float(p.mean()),
            "brier":float(np.mean((p-y)**2)),
            "binary_logloss":float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
        })
    for band,g in z.groupby("rank_band"):
        p=np.clip(g["p_top3"].to_numpy(dtype=float),EPS,1-EPS); y=g["actual_top3"].to_numpy(dtype=int)
        out.append({
            "year":"POOLED","rank_band":band,"horses":len(g),"actual_top3_pct":100*float(y.mean()),
            "mean_predicted_top3_pct":100*float(p.mean()),"brier":float(np.mean((p-y)**2)),
            "binary_logloss":float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
        })
    return out

def main():
    t0=time.time(); a=parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    c=json.load(open(a.contract,encoding="utf-8"))
    assert c["contract"]=="L2_TRIO_PROBABILITY_SHOWDOWN_V1"
    assert c["cost_policy"]["github_standard_cpu_only"] is True
    assert c["cost_policy"]["gpu"] is False
    assert c["cost_policy"]["paid_artifact_or_cache"] is False
    assert c["data_policy"]["kaggle_metadata_first"] is True
    assert c["data_policy"]["404_retry"] is False

    _,df=load_horse_dataset(a.dataset_dir)
    df,skipped_market=attach_win_market(df,a.backfill_root)
    df=attach_outsider(df,a.outsider_predictions)
    df=prepare_l175(df)
    if 2026 in set(df["year"]): raise SystemExit("2026 sealed")

    ordered,ordered_bad=ordered_top3_map(df,a.backfill_root)
    df=add_position_labels(df,ordered)

    for y in YEARS:
        build_year_matrix(y,df,a.backfill_root,a.matrix_cache)
    loaded={y:load_matrix(a.matrix_cache,y) for y in YEARS}
    payout_by_year={y:payout_vectors(a.backfill_root,a.matrix_cache,y) for y in (2023,2024,2025)}

    fidx,fnames=direct_feature_indices(loaded[2022][0])
    prob_rows=[]; roi_rows=[]; param_rows=[]; p3_rows=[]; alpha_rows=[]; beta_rows=[]; temp_rows=[]

    # Pure market baseline once per test year.
    for test_year,_ in FOLDS:
        meta,X,y,ri,qm,bk,bw=loaded[test_year]
        mm=probability_metrics(y,qm,ri)
        prob_rows.append({"test_year":test_year,"engine":"PURE_TRIO_MARKET","variant":"MARKET_ONLY",**mm})
        for k in ROI_KS:
            roi_rows.append({"test_year":test_year,"engine":"PURE_TRIO_MARKET","variant":"MARKET_ONLY","k":k,**roi_topk(y,qm,ri,payout_by_year[test_year],k)})

    for test_year,train_years in FOLDS:
        recent=train_years[-1]
        rmeta,Xr,yr,rir,qmr,bkr,bwr=loaded[recent]
        cut=max(1,int(rmeta["races"]*0.80))
        cal_mask=np.asarray(rir)>=cut
        fit_race_indices=list(range(cut))
        cal_ri=np.asarray(rir)[cal_mask]-cut
        ycal=np.asarray(yr)[cal_mask]
        qcal=np.asarray(qmr)[cal_mask]
        rm_recent=race_meta(a.matrix_cache,recent)
        fit_recent_ids=set(rm_recent.iloc[:cut]["race_id"].astype(str))

        # 1) Current Harville / PL.
        pl_cal=np.asarray(bkr)[cal_mask]
        (beta_pl,bll),bgrid=select_beta(ycal,qcal,pl_cal,cal_ri)
        for b,ll in bgrid: beta_rows.append({"test_year":test_year,"engine":"CURRENT_HARVILLE_PL","beta":b,"cal_logloss":ll,"selected":int(abs(b-beta_pl)<1e-12)})
        meta,Xt,yt,rit,qmt,bkt,bwt=loaded[test_year]
        methods={"CURRENT_HARVILLE_PL":np.asarray(bkt,dtype=float)}
        mixes={"CURRENT_HARVILLE_PL":blend_market_ai(qmt,methods["CURRENT_HARVILLE_PL"],rit,beta_pl)}
        param_rows.append({"test_year":test_year,"engine":"CURRENT_HARVILLE_PL","alpha2":"","alpha3":"","temperature":1.0,"market_blend_beta":beta_pl})

        # 2) Discounted Harville.
        (a2,a3,fitll),agrid=fit_discount_alphas(recent,df,a.matrix_cache,loaded,fit_race_indices)
        for rr in agrid: alpha_rows.append({"test_year":test_year,**rr,"selected":int(abs(rr["alpha2"]-a2)<1e-12 and abs(rr["alpha3"]-a3)<1e-12)})
        disc_recent=discounted_distribution(recent,df,a.matrix_cache,loaded,a2,a3)
        disc_cal=disc_recent[cal_mask]
        (beta_disc,_),bgrid=select_beta(ycal,qcal,disc_cal,cal_ri)
        for b,ll in bgrid: beta_rows.append({"test_year":test_year,"engine":"DISCOUNTED_HARVILLE","beta":b,"cal_logloss":ll,"selected":int(abs(b-beta_disc)<1e-12)})
        disc_test=discounted_distribution(test_year,df,a.matrix_cache,loaded,a2,a3)
        methods["DISCOUNTED_HARVILLE"]=disc_test
        mixes["DISCOUNTED_HARVILLE"]=blend_market_ai(qmt,disc_test,rit,beta_disc)
        param_rows.append({"test_year":test_year,"engine":"DISCOUNTED_HARVILLE","alpha2":a2,"alpha3":a3,"temperature":1.0,"market_blend_beta":beta_disc,"fit_logloss":fitll})
        del disc_recent,disc_cal; gc.collect()

        # 3) Position-specific 1/2/3. Calibrate on chrono holdout.
        pos_cal_models,pos_cal_rows=fit_position_models(df,train_years,fit_recent_ids,recent)
        pos_recent,_=position_distribution(recent,df,a.matrix_cache,loaded,pos_cal_models,False)
        pos_cal=pos_recent[cal_mask]
        (tpos,tll),tgrid=choose_temp_from_probs(ycal,pos_cal,cal_ri)
        for t,ll in tgrid: temp_rows.append({"test_year":test_year,"engine":"POSITION_SPECIFIC_1_2_3","temperature":t,"cal_logloss":ll,"selected":int(abs(t-tpos)<1e-12)})
        pos_cal=temperature_scale(pos_cal,cal_ri,tpos)
        (beta_pos,_),bgrid=select_beta(ycal,qcal,pos_cal,cal_ri)
        for b,ll in bgrid: beta_rows.append({"test_year":test_year,"engine":"POSITION_SPECIFIC_1_2_3","beta":b,"cal_logloss":ll,"selected":int(abs(b-beta_pos)<1e-12)})
        del pos_cal_models,pos_recent,pos_cal; gc.collect()

        pos_models,pos_full_rows=fit_position_models(df,train_years)
        pos_test,horse_audit=position_distribution(test_year,df,a.matrix_cache,loaded,pos_models,True)
        pos_test=temperature_scale(pos_test,rit,tpos)
        methods["POSITION_SPECIFIC_1_2_3"]=pos_test
        mixes["POSITION_SPECIFIC_1_2_3"]=blend_market_ai(qmt,pos_test,rit,beta_pos)
        p3_rows.extend(horse_audit)
        param_rows.append({"test_year":test_year,"engine":"POSITION_SPECIFIC_1_2_3","alpha2":"","alpha3":"","temperature":tpos,"market_blend_beta":beta_pos,"cal_train_horse_rows":pos_cal_rows,"full_train_horse_rows":pos_full_rows})
        del pos_models; gc.collect()

        # 4) Direct TRIO fundamental-only combination model.
        dcal_model,dcal_rows=fit_direct(loaded,train_years,fidx,recent,cut)
        d_recent=direct_distribution(dcal_model,loaded,recent,fidx,1.0)
        d_cal=d_recent[cal_mask]
        (tdir,tdll),tgrid=choose_temp_from_probs(ycal,d_cal,cal_ri)
        for t,ll in tgrid: temp_rows.append({"test_year":test_year,"engine":"DIRECT_TRIO","temperature":t,"cal_logloss":ll,"selected":int(abs(t-tdir)<1e-12)})
        d_cal=temperature_scale(d_cal,cal_ri,tdir)
        (beta_dir,_),bgrid=select_beta(ycal,qcal,d_cal,cal_ri)
        for b,ll in bgrid: beta_rows.append({"test_year":test_year,"engine":"DIRECT_TRIO","beta":b,"cal_logloss":ll,"selected":int(abs(b-beta_dir)<1e-12)})
        del dcal_model,d_recent,d_cal; gc.collect()

        direct_model,d_full_rows=fit_direct(loaded,train_years,fidx)
        direct_test=direct_distribution(direct_model,loaded,test_year,fidx,tdir)
        methods["DIRECT_TRIO"]=direct_test
        mixes["DIRECT_TRIO"]=blend_market_ai(qmt,direct_test,rit,beta_dir)
        param_rows.append({"test_year":test_year,"engine":"DIRECT_TRIO","alpha2":"","alpha3":"","temperature":tdir,"market_blend_beta":beta_dir,"cal_train_ticket_rows":dcal_rows,"full_train_ticket_rows":d_full_rows})
        del direct_model; gc.collect()

        # Evaluate all AI-only and market+AI distributions on untouched test year.
        market_mm=probability_metrics(yt,qmt,rit)
        for engine,pai in methods.items():
            mm=probability_metrics(yt,pai,rit)
            prob_rows.append({"test_year":test_year,"engine":engine,"variant":"AI_ONLY",**mm,
                              "logloss_delta_vs_market":mm["race_log_loss"]-market_mm["race_log_loss"]})
            pm=mixes[engine]
            mx=probability_metrics(yt,pm,rit)
            prob_rows.append({"test_year":test_year,"engine":engine,"variant":"MARKET_PLUS_AI",**mx,
                              "logloss_delta_vs_market":mx["race_log_loss"]-market_mm["race_log_loss"]})
            for variant,pp in (("AI_ONLY",pai),("MARKET_PLUS_AI",pm)):
                for k in ROI_KS:
                    roi_rows.append({"test_year":test_year,"engine":engine,"variant":variant,"k":k,
                                     **roi_topk(yt,pp,rit,payout_by_year[test_year],k)})
        print("SHOWDOWN_FOLD_DONE "+json.dumps({
            "test_year":test_year,
            "params":[r for r in param_rows if r["test_year"]==test_year],
            "market_logloss":market_mm["race_log_loss"]
        },separators=(",",":")),flush=True)
        del methods,mixes,pos_test,direct_test,disc_test; gc.collect()

    # Pooled probability summary weighted by race count.
    pdf=pd.DataFrame(prob_rows)
    pooled_prob=[]
    for (engine,variant),g in pdf.groupby(["engine","variant"]):
        weights=g["races"].astype(float).to_numpy()
        row={"test_year":"POOLED","engine":engine,"variant":variant,"races":int(weights.sum())}
        for col in ["race_log_loss","multiclass_brier","ticket_ece_15bin","mean_true_ticket_probability",
                    "winner_top1_pct","winner_top4_pct","winner_top8_pct","winner_top12_pct","winner_top20_pct"]:
            vals=pd.to_numeric(g[col],errors="coerce").to_numpy(dtype=float)
            row[col]=float(np.average(vals,weights=weights))
        if "logloss_delta_vs_market" in g.columns:
            vals=pd.to_numeric(g["logloss_delta_vs_market"],errors="coerce")
            if vals.notna().any(): row["logloss_delta_vs_market"]=float(np.average(vals.fillna(0),weights=weights))
        pooled_prob.append(row)
    prob_rows.extend(pooled_prob)

    # Pooled ROI by summing economics, not averaging ROI.
    rdf=pd.DataFrame(roi_rows); pooled_roi=[]
    for (engine,variant,k),g in rdf.groupby(["engine","variant","k"]):
        races=int(g["races"].sum()); tickets=int(g["tickets"].sum()); hits=int(g["hit_races"].sum())
        stake=float(g["stake_yen"].sum()); ret=float(g["return_yen"].sum())
        pooled_roi.append({
            "test_year":"POOLED","engine":engine,"variant":variant,"k":int(k),
            "races":races,"tickets":tickets,"avg_tickets_per_race":tickets/races if races else None,
            "hit_races":hits,"hit_rate_pct":100*hits/races if races else None,
            "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
            "roi_pct":100*ret/stake if stake else None,
            "avg_payout_when_hit_yen":ret/hits if hits else None
        })
    roi_rows.extend(pooled_roi)

    p3_summary=calibration_band(p3_rows)

    # Decision summary.
    pr=pd.DataFrame(prob_rows)
    decisions=[]
    market_year={int(r["test_year"]):r for r in prob_rows if r["engine"]=="PURE_TRIO_MARKET" and r["test_year"]!="POOLED"}
    for engine in ("CURRENT_HARVILLE_PL","DISCOUNTED_HARVILLE","POSITION_SPECIFIC_1_2_3","DIRECT_TRIO"):
        g=pr[(pr["engine"]==engine)&(pr["variant"]=="MARKET_PLUS_AI")&(pr["test_year"]!="POOLED")].copy()
        deltas=pd.to_numeric(g["logloss_delta_vs_market"],errors="coerce").tolist()
        decisions.append({
            "engine":engine,"years_beating_market":sum(1 for x in deltas if x<0),
            "all_years_beating_market":all(x<0 for x in deltas),
            "mean_delta_vs_market":float(np.mean(deltas)) if deltas else None,
            "continue_candidate":bool(sum(1 for x in deltas if x<0)>=2 and np.mean(deltas)<0)
        })

    write_csv(out/"probability-metrics.csv",prob_rows)
    write_csv(out/"roi-topk.csv",roi_rows)
    write_csv(out/"parameters.csv",param_rows)
    write_csv(out/"discount-alpha-grid.csv",alpha_rows)
    write_csv(out/"market-blend-beta-grid.csv",beta_rows)
    write_csv(out/"temperature-grid.csv",temp_rows)
    write_csv(out/"fullfield-p3-calibration.csv",p3_summary)
    write_csv(out/"decision.csv",decisions)

    summary={
        "contract":"L2_TRIO_PROBABILITY_SHOWDOWN_V1_RESULT",
        "ai_feature_policy":"SEVEN_KING_FUNDAMENTALS_ONLY; NO WIN_MARKET, TRIO_MARKET, OUTSIDER OR STORED_P3 INSIDE AI ENGINES",
        "market_plus_ai_form":"normalize(q_market^(1-beta) * p_ai^beta)",
        "historical_market_proxy":"FINAL_TRIO_ODDS",
        "probability_times_odds_ranking":False,
        "odds_can_promote_ticket":False,
        "payout_used_only_for_roi_evaluation":True,
        "direct_feature_names":fnames,
        "ordered_label_diagnostics":ordered_bad,
        "skipped_win_market_races":len(skipped_market),
        "decisions":decisions,
        "probability_pooled":[r for r in pooled_prob],
        "roi_pooled":[r for r in pooled_roi],
        "p3_pooled":[r for r in p3_summary if r["year"]=="POOLED"],
        "2026_locked":True,
        "promotion":False,
        "elapsed_seconds":time.time()-t0
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 TRIO Probability Showdown V1\n\n"
        "Final decision run: current Harville/PL vs discounted Harville vs position-specific 1/2/3 vs direct TRIO. "
        "AI engines are Seven-King-fundamental only. Final TRIO market odds are used only as a separate market distribution "
        "and in the market+AI blend. Test years are untouched walk-forward years; 2026 is sealed. "
        "ROI is evaluated at flat 100 yen per selected ticket for probability-only top 4/8/12/20.\n",
        encoding="utf-8"
    )
    print("===== DECISION ====="); print(pd.DataFrame(decisions).to_string(index=False))
    print("===== POOLED PROBABILITY ====="); print(pd.DataFrame(pooled_prob).to_string(index=False))
    print("===== POOLED ROI ====="); print(pd.DataFrame(pooled_roi).to_string(index=False))
    print("===== P3 ====="); print(pd.DataFrame([r for r in p3_summary if r["year"]=="POOLED"]).to_string(index=False))
    print("L2_TRIO_PROBABILITY_SHOWDOWN_V1_READY")

if __name__=="__main__":
    main()
