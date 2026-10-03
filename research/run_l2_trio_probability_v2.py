#!/usr/bin/env python3
import argparse,csv,gzip,itertools,json,math,time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import load_odds_day,final_odds_tuple,finite

YEARS=(2022,2023,2024,2025)
FOLDS=((2023,(2022,)),(2024,(2022,2023)),(2025,(2022,2023,2024)))
EPS=1e-12

HORSE_FEATURES=(
    'king_rank_pct','king_mean_rank_pct','king_rank_std_pct','king_best_rank_pct','king_worst_rank_pct',
    'king_top1_vote_share','king_top3_support_share','king_top6_support_share','king_probability_mean','king_probability_std',
    'market_rank_pct','log_market_win_odds','signed_rank_gap_pct','abs_rank_gap_pct','dissent_up','dissent_down',
    'outsider_available','outsider_score_scaled','p3_calibrated','p3_delta','outsider_alignment_score',
)
AGG_FEATURES=(
    'field_size_norm','king_rank_mean_pct','king_rank_span_pct','market_rank_mean_pct','market_rank_span_pct',
    'signed_gap_mean_pct','abs_gap_mean_pct','king_probability_sum','p3_sum','outsider_available_share',
    'outsider_score_mean','alignment_score_mean','race_entropy','race_top3_probability_sum','race_top1_top2_gap',
)

def parse_args():
    p=argparse.ArgumentParser(description='L2 V2 direct TRIO probability engine from frozen L1.75 context.')
    p.add_argument('--contract',required=True)
    p.add_argument('--l175-contract',required=True)
    p.add_argument('--dataset-dir',required=True)
    p.add_argument('--backfill-root',required=True)
    p.add_argument('--outsider-predictions',required=True)
    p.add_argument('--matrix-cache',required=True)
    p.add_argument('--out-dir',required=True)
    return p.parse_args()

def clip01(x): return min(max(float(x),EPS),1.0-EPS)

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text('',encoding='utf-8'); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def load_horse_dataset(root):
    root=Path(root); m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if m.get('contract')!='L2_CORE_DATASET_V1': raise SystemExit('wrong horse dataset contract')
    if m.get('locked_years')!=[2026] or m.get('odds_in_dataset') is not False: raise SystemExit('horse dataset guard drift')
    rows=[]
    with gzip.open(root/m['file'],'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    df=pd.DataFrame(rows)
    df['year']=pd.to_numeric(df['year'],errors='raise').astype(int)
    df['race_id']=df['race_id'].astype(str); df['horse_id']=df['horse_id'].astype(str)
    if 2026 in set(df['year']): raise SystemExit('2026 sealed')
    return m,df

def attach_win_market(df,backfill_root):
    root=Path(backfill_root); groups={str(rid):g for rid,g in df.groupby('race_id',sort=False)}
    bydate=defaultdict(list)
    for rid,date in df[['race_id','race_date']].drop_duplicates().itertuples(index=False):
        bydate[str(date)[:10]].append(str(rid))
    rows=[]; skipped=[]
    for di,(date,rids) in enumerate(sorted(bydate.items()),1):
        wanted=set(rids); oddsday=load_odds_day(root/'data'/'odds'/'daily'/f'{date}.jsonl.gz',wanted)
        for rid in rids:
            sub=groups[rid]; orec=oddsday.get(rid)
            if orec is None:
                skipped.append({'race_id':rid,'race_date':date,'reason':'odds_missing'}); continue
            raw=((orec.get('odds') or {}).get('1') or {})
            odds={}
            for key,val in raw.items():
                k=str(key)
                if not k.isascii() or not k.isdigit(): continue
                tup=final_odds_tuple(val)
                if not tup: continue
                odd=finite(tup[0])
                if odd is not None and odd>0: odds[int(k)]=float(odd)
            expected=set(int(x) for x in sub['horse_number'].dropna().astype(int))
            if set(odds)!=expected:
                skipped.append({'race_id':rid,'race_date':date,'reason':f'win_odds_incomplete:{len(odds)}/{len(expected)}'}); continue
            unique=sorted(set(odds.values()))
            rank_by_odd={o:1+sum(1 for x in odds.values() if x<o) for o in unique}
            for idx,r in sub.iterrows():
                no=int(r['horse_number'])
                rows.append({'idx':int(idx),'market_win_odds':odds[no],'market_rank':int(rank_by_odd[odds[no]])})
        if di%50==0: print(f'L2_TRIO_MARKET_PROGRESS dates={di}/{len(bydate)}',flush=True)
    if not rows: raise SystemExit('no market rows')
    aux=pd.DataFrame(rows).set_index('idx')
    keep=df.index.intersection(aux.index)
    return df.loc[keep].copy().join(aux.loc[keep]),skipped

def attach_outsider(df,path):
    use=['year','race_id','horse_id','rank','score','p3','p3_rank_baseline','p3_delta']
    p=pd.read_csv(path,compression='gzip',usecols=use)
    p['year']=pd.to_numeric(p['year'],errors='raise').astype(int)
    p['race_id']=p['race_id'].astype(str); p['horse_id']=p['horse_id'].astype(str)
    if 2026 in set(p['year']): raise SystemExit('2026 sealed in outsider source')
    for c in ['rank','score','p3','p3_rank_baseline','p3_delta']:
        p[c]=pd.to_numeric(p[c],errors='coerce')
    if p.duplicated(['year','race_id','horse_id']).any(): raise SystemExit('duplicate outsider prediction keys')
    p=p.rename(columns={'rank':'outsider_rank','score':'outsider_score','p3':'p3_calibrated'})
    z=df.merge(p,on=['year','race_id','horse_id'],how='left',validate='one_to_one')
    z['outsider_available']=z['outsider_score'].notna().astype(float)
    z['outsider_rank_mismatch']=((z['outsider_available']>0)&(z['outsider_rank'].fillna(-1).astype(float)!=z['consensus_rank'].astype(float))).astype(int)
    for c in ['outsider_score','p3_calibrated','p3_rank_baseline','p3_delta']:
        z[c]=pd.to_numeric(z[c],errors='coerce').fillna(0.0)
    return z

def prepare_l175(df):
    z=df.copy(); fs=z['field_size'].astype(float).clip(lower=1.0)
    z['king_rank_pct']=z['consensus_rank'].astype(float)/fs
    z['king_mean_rank_pct']=z['mean_rank'].astype(float)/fs
    z['king_rank_std_pct']=z['rank_std'].astype(float)/fs
    z['king_best_rank_pct']=z['best_rank'].astype(float)/fs
    z['king_worst_rank_pct']=z['worst_rank'].astype(float)/fs
    z['king_top1_vote_share']=z['top1_votes'].astype(float)/7.0
    z['king_top3_support_share']=z['top3_support'].astype(float)/7.0
    z['king_top6_support_share']=z['top6_support'].astype(float)/7.0
    z['king_probability_mean']=z['mean_probability'].astype(float)
    z['king_probability_std']=z['probability_std'].astype(float)
    z['market_rank_pct']=z['market_rank'].astype(float)/fs
    z['log_market_win_odds']=np.log(np.clip(z['market_win_odds'].astype(float),1.000001,None))
    z['signed_rank_gap']=z['market_rank'].astype(float)-z['consensus_rank'].astype(float)
    z['signed_rank_gap_pct']=z['signed_rank_gap']/fs
    z['abs_rank_gap_pct']=np.abs(z['signed_rank_gap_pct'])
    z['dissent_up']=(z['signed_rank_gap']>=2).astype(float)
    z['dissent_down']=(z['signed_rank_gap']<=-2).astype(float)
    z['outsider_score_scaled']=z['outsider_score'].astype(float)/39.0
    z['outsider_alignment_score']=np.sign(z['signed_rank_gap'].to_numpy(dtype=float))*z['p3_delta'].to_numpy(dtype=float)
    for c in HORSE_FEATURES:
        z[c]=pd.to_numeric(z[c],errors='coerce').fillna(0.0).astype(np.float32)
    return z

def trio_pl_probs(p,comb):
    p=np.asarray(p,dtype=np.float64); c=np.asarray(comb,dtype=np.int64)
    a,b,d=c[:,0],c[:,1],c[:,2]
    def ord3(i,j,k):
        pi,pj,pk=p[i],p[j],p[k]
        den1=np.maximum(1.0-pi,EPS); den2=np.maximum(1.0-pi-pj,EPS)
        return pi*(pj/den1)*(pk/den2)
    q=ord3(a,b,d)+ord3(a,d,b)+ord3(b,a,d)+ord3(b,d,a)+ord3(d,a,b)+ord3(d,b,a)
    s=float(q.sum())
    return (q/s if s>0 else np.full(len(q),1.0/len(q))).astype(np.float32)

def build_year_matrix(year,df,cache):
    cache=Path(cache)/str(year); cache.mkdir(parents=True,exist_ok=True); done=cache/'meta.json'
    if done.exists():
        meta=json.loads(done.read_text(encoding='utf-8'))
        if meta.get('contract')=='L2_TRIO_MATRIX_V2' and meta.get('feature_count')==len(HORSE_FEATURES)*3+len(AGG_FEATURES):
            print(f'L2_TRIO_MATRIX_REUSE year={year} rows={meta["rows"]}',flush=True); return meta
    parts=[]; ys=[]; rids=[]; base_king=[]; base_market=[]; race_meta=[]
    feature_names=[f'h{i}_{c}' for i in (1,2,3) for c in HORSE_FEATURES]+list(AGG_FEATURES)
    ydf=df[df['year']==year].copy()
    race_order=ydf[['race_id','race_date']].drop_duplicates().sort_values(['race_date','race_id'])['race_id'].astype(str).tolist()
    groups={str(rid):g for rid,g in ydf.groupby('race_id',sort=False)}
    race_index=0; excluded=defaultdict(int)
    for rid in race_order:
        g=groups[rid].sort_values(['consensus_rank','horse_number','horse_id']).reset_index(drop=True)
        top3=g['target_top3']
        if top3.isna().any(): excluded['missing_trio_label']+=1; continue
        podium_idx=np.where(top3.to_numpy(dtype=int)==1)[0]
        if len(podium_idx)!=3: excluded['deadheat_or_nonunique_trio']+=1; continue
        n=len(g)
        if n<3: excluded['field_lt3']+=1; continue
        comb=np.asarray(list(itertools.combinations(range(n),3)),dtype=np.int16)
        ranks=g['consensus_rank'].to_numpy(dtype=np.float32)
        order=np.argsort(ranks[comb],axis=1,kind='stable'); ordered=np.take_along_axis(comb,order,axis=1)
        H=g.loc[:,HORSE_FEATURES].to_numpy(dtype=np.float32,copy=True)
        X3=H[ordered].reshape(len(comb),-1)
        fs=float(n); kr=ranks[comb]/fs; mr=g['market_rank'].to_numpy(dtype=np.float32)[comb]/fs
        sg=g['signed_rank_gap_pct'].to_numpy(dtype=np.float32)[comb]; ag=np.abs(sg)
        kp=g['king_probability_mean'].to_numpy(dtype=np.float32)[comb]
        p3=g['p3_calibrated'].to_numpy(dtype=np.float32)[comb]
        oa=g['outsider_available'].to_numpy(dtype=np.float32)[comb]
        oscore=g['outsider_score_scaled'].to_numpy(dtype=np.float32)[comb]
        align=g['outsider_alignment_score'].to_numpy(dtype=np.float32)[comb]
        agg=np.column_stack([
            np.full(len(comb),fs/18.0,dtype=np.float32),
            kr.mean(1),kr.max(1)-kr.min(1),mr.mean(1),mr.max(1)-mr.min(1),
            sg.mean(1),ag.mean(1),kp.sum(1),p3.sum(1),oa.mean(1),oscore.mean(1),align.mean(1),
            np.full(len(comb),float(g['race_entropy'].iloc[0]),dtype=np.float32),
            np.full(len(comb),float(g['race_top3_probability_sum'].iloc[0]),dtype=np.float32),
            np.full(len(comb),float(g['race_top1_top2_gap'].iloc[0]),dtype=np.float32),
        ]).astype(np.float32,copy=False)
        X=np.concatenate([X3,agg],axis=1).astype(np.float32,copy=False)
        target_set=set(int(x) for x in podium_idx)
        y=np.fromiter((1 if set(map(int,row))==target_set else 0 for row in comb),dtype=np.uint8,count=len(comb))
        if int(y.sum())!=1: raise SystemExit(f'unique trio target invariant failed race={rid} ysum={int(y.sum())}')
        kingp=g['mean_probability'].to_numpy(dtype=np.float64); kingp=np.clip(kingp,0,None)
        kingp=kingp/kingp.sum() if kingp.sum()>0 else np.full(n,1.0/n)
        marketp=1.0/np.clip(g['market_win_odds'].to_numpy(dtype=np.float64),1.000001,None); marketp=marketp/marketp.sum()
        parts.append(X); ys.append(y); rids.append(np.full(len(comb),race_index,dtype=np.int32))
        base_king.append(trio_pl_probs(kingp,comb)); base_market.append(trio_pl_probs(marketp,comb))
        race_meta.append({'race_index':race_index,'race_id':str(rid),'race_date':str(g['race_date'].iloc[0])[:10],'combos':len(comb),'field_size':n})
        race_index+=1
        if race_index%500==0: print(f'L2_TRIO_MATRIX_PROGRESS year={year} races={race_index}',flush=True)
    if not parts: raise SystemExit(f'no eligible races year={year}')
    X=np.concatenate(parts); y=np.concatenate(ys); race_idx=np.concatenate(rids); bk=np.concatenate(base_king); bm=np.concatenate(base_market)
    np.save(cache/'X.npy',X,allow_pickle=False); np.save(cache/'y.npy',y,allow_pickle=False); np.save(cache/'race_idx.npy',race_idx,allow_pickle=False)
    np.save(cache/'baseline_king.npy',bk,allow_pickle=False); np.save(cache/'baseline_market.npy',bm,allow_pickle=False)
    write_csv(cache/'races.csv',race_meta)
    meta={'contract':'L2_TRIO_MATRIX_V2','year':year,'races':race_index,'rows':int(len(y)),'positives':int(y.sum()),'feature_count':int(X.shape[1]),'feature_names':feature_names,'excluded_races':dict(excluded),'float_dtype':'float32'}
    done.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'L2_TRIO_MATRIX_READY year={year} races={race_index} rows={len(y)} features={X.shape[1]}',flush=True)
    return meta

def load_matrix(cache,year):
    root=Path(cache)/str(year); meta=json.loads((root/'meta.json').read_text(encoding='utf-8'))
    return meta,np.load(root/'X.npy',mmap_mode='r'),np.load(root/'y.npy',mmap_mode='r'),np.load(root/'race_idx.npy',mmap_mode='r'),np.load(root/'baseline_king.npy',mmap_mode='r'),np.load(root/'baseline_market.npy',mmap_mode='r')

def softmax_by_race(raw,race_idx,temp):
    raw=np.asarray(raw,dtype=np.float64); ri=np.asarray(race_idx,dtype=np.int64); out=np.empty(len(raw),dtype=np.float64)
    if not len(raw): return out
    starts=np.r_[0,1+np.where(ri[1:]!=ri[:-1])[0]]; ends=np.r_[starts[1:],len(raw)]; t=max(float(temp),1e-6)
    for s,e in zip(starts,ends):
        z=np.clip(raw[s:e]/t,-60,60); z-=np.max(z); ex=np.exp(z); sm=ex.sum()
        out[s:e]=ex/sm if sm>0 else 1.0/(e-s)
    return out

def race_logloss(y,p,race_idx):
    y=np.asarray(y,dtype=np.uint8); p=np.asarray(p,dtype=np.float64); ri=np.asarray(race_idx,dtype=np.int64); vals=[]
    starts=np.r_[0,1+np.where(ri[1:]!=ri[:-1])[0]]; ends=np.r_[starts[1:],len(y)]
    for s,e in zip(starts,ends):
        pos=np.where(y[s:e]==1)[0]
        if len(pos)==1: vals.append(-math.log(clip01(p[s+pos[0]])))
    return float(np.mean(vals)) if vals else None

def choose_temperature(raw,y,race_idx):
    temps=np.exp(np.linspace(math.log(0.35),math.log(2.5),25)); best=(1.0,float('inf'))
    for t in temps:
        p=softmax_by_race(raw,race_idx,float(t)); ll=race_logloss(y,p,race_idx)
        if ll is not None and ll<best[1]: best=(float(t),ll)
    return best

def metrics(y,p,race_idx):
    y=np.asarray(y,dtype=np.uint8); p=np.asarray(p,dtype=np.float64); ri=np.asarray(race_idx,dtype=np.int64)
    starts=np.r_[0,1+np.where(ri[1:]!=ri[:-1])[0]]; ends=np.r_[starts[1:],len(y)]
    ll=[]; brier=[]; hit1=hit3=hit10=0; truep=[]; masserr=0.0
    bins=[{'w':0.0,'p':0.0,'y':0.0} for _ in range(15)]
    for s,e in zip(starts,ends):
        yy=y[s:e]; pp=p[s:e]; pos=np.where(yy==1)[0]
        if len(pos)!=1: continue
        win=int(pos[0]); ll.append(-math.log(clip01(pp[win]))); brier.append(float(np.sum((pp-yy)**2))); truep.append(float(pp[win]))
        order=np.argsort(-pp,kind='stable'); hit1+=int(win==order[0]); hit3+=int(win in set(order[:3])); hit10+=int(win in set(order[:10]))
        masserr=max(masserr,abs(float(pp.sum())-1.0)); rw=1.0/len(pp)
        for prob,target in zip(pp,yy):
            b=min(int(max(prob,0.0)*15),14); bins[b]['w']+=rw; bins[b]['p']+=rw*float(prob); bins[b]['y']+=rw*float(target)
    races=len(ll); totalw=sum(x['w'] for x in bins); ece=0.0
    for z in bins:
        if z['w']>0: ece+=(z['w']/totalw)*abs(z['p']/z['w']-z['y']/z['w'])
    return {'races':races,'race_log_loss':float(np.mean(ll)) if ll else None,'multiclass_brier':float(np.mean(brier)) if brier else None,'ticket_ece_15bin':ece,'winner_top1_pct':100*hit1/races if races else None,'winner_top3_pct':100*hit3/races if races else None,'winner_top10_pct':100*hit10/races if races else None,'mean_true_ticket_probability':float(np.mean(truep)) if truep else None,'probability_mass_max_error':masserr}

def concat_arrays(items): return np.concatenate([np.asarray(x) for x in items],axis=0)

def main():
    t0=time.time(); a=parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    contract=json.loads(Path(a.contract).read_text(encoding='utf-8')); l175=json.loads(Path(a.l175_contract).read_text(encoding='utf-8'))
    if contract.get('contract')!='L2_TRIO_PROBABILITY_V2': raise SystemExit('wrong V2 contract')
    if l175.get('contract')!='L175_MARKET_CONTEXT_V1' or l175.get('ranking_policy')!='SEVEN_KING_UNCHANGED': raise SystemExit('wrong L1.75 contract')
    if contract['cost_policy']['github_standard_cpu_only'] is not True or contract['cost_policy']['gpu'] is not False: raise SystemExit('cost guard drift')
    manifest,df=load_horse_dataset(a.dataset_dir)
    df,skipped=attach_win_market(df,a.backfill_root); df=attach_outsider(df,a.outsider_predictions); df=prepare_l175(df)
    coverage=[]
    for y in YEARS:
        q=df[df['year']==y]
        coverage.append({'year':y,'races':q['race_id'].nunique(),'horses':len(q),'outsider_available_horses':int(q['outsider_available'].sum()),'outsider_rank_mismatch_horses':int(q['outsider_rank_mismatch'].sum())})
        build_year_matrix(y,df,a.matrix_cache)
    write_csv(out/'coverage.csv',coverage); write_csv(out/'skipped-market-races.csv',skipped)
    matrix_meta={y:load_matrix(a.matrix_cache,y)[0] for y in YEARS}
    fold_rows=[]; importance=[]; runtime_rows=[]
    for test_year,train_years in FOLDS:
        ft=time.time(); recent=train_years[-1]
        loaded={y:load_matrix(a.matrix_cache,y) for y in set(train_years+(test_year,))}
        rmeta,Xr,yr,rir,bkr,bmr=loaded[recent]
        cut=max(1,int(rmeta['races']*0.8)); fit_recent=np.asarray(rir)<cut; cal_recent=~fit_recent
        Xfit_parts=[]; yfit_parts=[]; wfit_parts=[]
        for y in train_years[:-1]:
            meta,X,yy,ri,_,_=loaded[y]; ria=np.asarray(ri); counts=np.bincount(ria,minlength=meta['races'])
            Xfit_parts.append(X); yfit_parts.append(yy); wfit_parts.append(1.0/counts[ria])
        ria=np.asarray(rir)
        Xfit_parts.append(np.asarray(Xr)[fit_recent]); yfit_parts.append(np.asarray(yr)[fit_recent])
        counts=np.bincount(ria[fit_recent],minlength=cut); wfit_parts.append(1.0/np.maximum(counts[ria[fit_recent]],1))
        Xfit=concat_arrays(Xfit_parts).astype(np.float32,copy=False); yfit=concat_arrays(yfit_parts).astype(np.uint8,copy=False); wfit=concat_arrays(wfit_parts).astype(np.float32,copy=False)
        Xcal=np.asarray(Xr)[cal_recent].astype(np.float32,copy=False); ycal=np.asarray(yr)[cal_recent].astype(np.uint8,copy=False)
        rical=ria[cal_recent].astype(np.int32,copy=False); rical=rical-rical.min()
        params=dict(objective='binary',n_estimators=180,learning_rate=0.05,num_leaves=31,max_bin=63,min_child_samples=100,subsample=0.9,subsample_freq=1,colsample_bytree=0.85,reg_lambda=1.0,random_state=20261004,n_jobs=-1,verbosity=-1,force_col_wise=True)
        m0=lgb.LGBMClassifier(**params); m0.fit(Xfit,yfit,sample_weight=wfit)
        rawcal=np.asarray(m0.booster_.predict(Xcal,raw_score=True),dtype=float); temp,cal_ll=choose_temperature(rawcal,ycal,rical)
        Xtrain_parts=[]; ytrain_parts=[]; wtrain_parts=[]
        for y in train_years:
            meta,X,yy,ri,_,_=loaded[y]; ria2=np.asarray(ri); counts2=np.bincount(ria2,minlength=meta['races'])
            Xtrain_parts.append(X); ytrain_parts.append(yy); wtrain_parts.append(1.0/counts2[ria2])
        Xtrain=concat_arrays(Xtrain_parts).astype(np.float32,copy=False); ytrain=concat_arrays(ytrain_parts).astype(np.uint8,copy=False); wtrain=concat_arrays(wtrain_parts).astype(np.float32,copy=False)
        model=lgb.LGBMClassifier(**params); model.fit(Xtrain,ytrain,sample_weight=wtrain)
        meta,Xtest,ytest,ritest,bk,bm=loaded[test_year]
        raw=np.asarray(model.booster_.predict(Xtest,raw_score=True),dtype=float); p=softmax_by_race(raw,ritest,temp)
        row={'test_year':test_year,'train_years':'|'.join(map(str,train_years)),'temperature':temp,'calibration_race_log_loss':cal_ll,'train_rows':int(len(ytrain)),'test_rows':int(len(ytest))}
        for name,prob in [('DIRECT_L175',p),('BASE_KING_PL',bk),('BASE_MARKET_PL',bm)]:
            mm=metrics(ytest,prob,ritest)
            for k,v in mm.items(): row[f'{name.lower()}_{k}']=v
        row['logloss_delta_vs_king']=row['direct_l175_race_log_loss']-row['base_king_pl_race_log_loss']
        row['logloss_delta_vs_market']=row['direct_l175_race_log_loss']-row['base_market_pl_race_log_loss']
        fold_rows.append(row)
        gains=model.booster_.feature_importance(importance_type='gain'); names=matrix_meta[test_year]['feature_names']
        for n,g in sorted(zip(names,gains),key=lambda x:-x[1]): importance.append({'test_year':test_year,'feature':n,'gain':float(g)})
        runtime_rows.append({'test_year':test_year,'seconds':time.time()-ft,'train_rows':int(len(ytrain)),'test_rows':int(len(ytest))})
        print('L2_TRIO_FOLD_DONE '+json.dumps({'test_year':test_year,'temp':temp,'direct_logloss':row['direct_l175_race_log_loss'],'king_logloss':row['base_king_pl_race_log_loss'],'market_logloss':row['base_market_pl_race_log_loss'],'direct_top10':row['direct_l175_winner_top10_pct']},separators=(',',':')),flush=True)
        del Xfit,yfit,wfit,Xcal,ycal,Xtrain,ytrain,wtrain,model,m0,raw,p
    write_csv(out/'fold-metrics.csv',fold_rows); write_csv(out/'feature-importance.csv',importance); write_csv(out/'runtime.csv',runtime_rows)
    summary={'contract':'L2_TRIO_PROBABILITY_V2_RESULT','architecture':'DIRECT_TRIO_COMBINATION_SCORE_SOFTMAX','upstream':'L175_MARKET_CONTEXT_V1','ranking_policy':'SEVEN_KING_UNCHANGED','outsider_policy':'SIGNAL_ONLY_NO_RERANK','trio_market_odds_used_as_feature':False,'win_market_context_used':True,'2026_locked':True,'matrix_meta':matrix_meta,'coverage':coverage,'skipped_market_races':len(skipped),'folds':fold_rows,'runtime_seconds_total':time.time()-t0,'promotion':False}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text('# L2 TRIO Probability V2\n\nDirect unordered TRIO probability model from frozen L1.75 context. No BUY/SKIP, ROI target, TRIO odds feature, bankroll, or stake sizing. Standard CPU only; 2026 sealed.\n',encoding='utf-8')
    maxerr=max(float(r.get('direct_l175_probability_mass_max_error') or 0.0) for r in fold_rows)
    if maxerr>1e-8: raise SystemExit(f'probability mass invariant failed max_error={maxerr}')
    print('L2_TRIO_PROBABILITY_V2_READY')
    print(json.dumps({'folds':len(fold_rows),'runtime_seconds':time.time()-t0,'max_mass_error':maxerr,'out_dir':str(out)},separators=(',',':')))

if __name__=='__main__': main()
