#!/usr/bin/env python3
import argparse,csv,itertools,json,math,time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import load_odds_day,final_odds_tuple,finite,parse_odds_key
from run_l2_trio_probability_v2 import (
    YEARS,FOLDS,HORSE_FEATURES,AGG_FEATURES,write_csv,load_horse_dataset,
    attach_win_market,attach_outsider,prepare_l175,trio_pl_probs,metrics
)

EPS=1e-12
TRIO_CODE_SIZE=200000

def parse_args():
    p=argparse.ArgumentParser(description='L2 V3 market-offset residual TRIO probability engine.')
    p.add_argument('--contract',required=True)
    p.add_argument('--l175-contract',required=True)
    p.add_argument('--dataset-dir',required=True)
    p.add_argument('--backfill-root',required=True)
    p.add_argument('--outsider-predictions',required=True)
    p.add_argument('--matrix-cache',required=True)
    p.add_argument('--out-dir',required=True)
    return p.parse_args()

def combo_features(g,comb):
    n=len(g); fs=float(n)
    ranks=g['consensus_rank'].to_numpy(dtype=np.float32)
    order=np.argsort(ranks[comb],axis=1,kind='stable')
    ordered=np.take_along_axis(comb,order,axis=1)
    H=g.loc[:,HORSE_FEATURES].to_numpy(dtype=np.float32,copy=True)
    X3=H[ordered].reshape(len(comb),-1)
    kr=ranks[comb]/fs
    mr=g['market_rank'].to_numpy(dtype=np.float32)[comb]/fs
    sg=g['signed_rank_gap_pct'].to_numpy(dtype=np.float32)[comb]
    ag=np.abs(sg)
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
    return np.concatenate([X3,agg],axis=1).astype(np.float32,copy=False)

def trio_market_distribution(orec,horse_numbers,comb):
    raw=((orec.get('odds') or {}).get('7') or {}) if orec is not None else {}
    lut=np.full(TRIO_CODE_SIZE,np.nan,dtype=np.float64)
    observed=0
    for key,val in raw.items():
        nums=parse_odds_key('TRIO',key)
        if nums is None: continue
        tup=final_odds_tuple(val)
        if not tup: continue
        odd=finite(tup[0])
        if odd is None or odd<=0: continue
        a,b,c=nums
        code=int(a)*10000+int(b)*100+int(c)
        if 0<=code<TRIO_CODE_SIZE:
            lut[code]=float(odd); observed+=1
    nums=np.sort(np.asarray(horse_numbers,dtype=np.int16)[comb],axis=1)
    codes=nums[:,0].astype(np.int32)*10000+nums[:,1].astype(np.int32)*100+nums[:,2].astype(np.int32)
    prices=lut[codes]
    valid=np.isfinite(prices)&(prices>0)
    if not valid.all():
        return None,{'expected':len(comb),'available':int(valid.sum()),'raw_market_rows':observed}
    rawp=1.0/prices
    overround=float(rawp.sum())
    if not np.isfinite(overround) or overround<=0:
        return None,{'expected':len(comb),'available':int(valid.sum()),'raw_market_rows':observed}
    q=(rawp/overround).astype(np.float32)
    return q,{'expected':len(comb),'available':len(comb),'raw_market_rows':observed,'overround':overround}

def build_year_matrix(year,df,backfill_root,cache):
    cache=Path(cache)/str(year); cache.mkdir(parents=True,exist_ok=True)
    done=cache/'meta.json'
    feature_names=[f'h{i}_{c}' for i in (1,2,3) for c in HORSE_FEATURES]+list(AGG_FEATURES)
    if done.exists():
        meta=json.loads(done.read_text(encoding='utf-8'))
        if meta.get('contract')=='L2_TRIO_MARKET_RESIDUAL_MATRIX_V3' and meta.get('feature_count')==len(feature_names):
            print(f'L2_TRIO_V3_MATRIX_REUSE year={year} rows={meta["rows"]}',flush=True)
            return meta

    ydf=df[df['year']==year].copy()
    groups={str(rid):g for rid,g in ydf.groupby('race_id',sort=False)}
    bydate=defaultdict(list)
    for rid,date in ydf[['race_id','race_date']].drop_duplicates().itertuples(index=False):
        bydate[str(date)[:10]].append(str(rid))

    Xs=[]; ys=[]; ris=[]; qmarkets=[]; bking=[]; bwin=[]; race_meta=[]
    excluded=defaultdict(int); missing_market_combos=0; race_index=0
    root=Path(backfill_root)

    for di,(date,rids) in enumerate(sorted(bydate.items()),1):
        oddsday=load_odds_day(root/'data'/'odds'/'daily'/f'{date}.jsonl.gz',set(rids))
        for rid in sorted(rids):
            g=groups[rid].sort_values(['consensus_rank','horse_number','horse_id']).reset_index(drop=True)
            top3=g['target_top3']
            if top3.isna().any():
                excluded['missing_trio_label']+=1; continue
            podium_idx=np.where(top3.to_numpy(dtype=int)==1)[0]
            if len(podium_idx)!=3:
                excluded['deadheat_or_nonunique_trio']+=1; continue
            n=len(g)
            if n<3:
                excluded['field_lt3']+=1; continue
            comb=np.asarray(list(itertools.combinations(range(n),3)),dtype=np.int16)
            q,mdiag=trio_market_distribution(oddsday.get(rid),g['horse_number'].to_numpy(dtype=np.int16),comb)
            if q is None:
                excluded['incomplete_trio_market']+=1
                missing_market_combos += int(mdiag['expected'])-int(mdiag['available'])
                continue

            X=combo_features(g,comb)
            target_set=set(int(x) for x in podium_idx)
            y=np.fromiter((1 if set(map(int,row))==target_set else 0 for row in comb),dtype=np.uint8,count=len(comb))
            if int(y.sum())!=1:
                raise SystemExit(f'unique trio target invariant failed race={rid} ysum={int(y.sum())}')

            kingp=g['mean_probability'].to_numpy(dtype=np.float64); kingp=np.clip(kingp,0,None)
            kingp=kingp/kingp.sum() if kingp.sum()>0 else np.full(n,1.0/n)
            winp=1.0/np.clip(g['market_win_odds'].to_numpy(dtype=np.float64),1.000001,None); winp=winp/winp.sum()

            Xs.append(X); ys.append(y); ris.append(np.full(len(comb),race_index,dtype=np.int32))
            qmarkets.append(q); bking.append(trio_pl_probs(kingp,comb)); bwin.append(trio_pl_probs(winp,comb))
            race_meta.append({
                'race_index':race_index,'race_id':rid,'race_date':date,'field_size':n,'combos':len(comb),
                'trio_market_overround':mdiag['overround']
            })
            race_index+=1
        if di%50==0:
            print(f'L2_TRIO_V3_MATRIX_PROGRESS year={year} dates={di}/{len(bydate)} races={race_index}',flush=True)

    if not Xs: raise SystemExit(f'no V3 eligible races year={year}')
    X=np.concatenate(Xs).astype(np.float32,copy=False)
    y=np.concatenate(ys); ri=np.concatenate(ris)
    qm=np.concatenate(qmarkets).astype(np.float32,copy=False)
    bk=np.concatenate(bking).astype(np.float32,copy=False)
    bw=np.concatenate(bwin).astype(np.float32,copy=False)

    np.save(cache/'X.npy',X,allow_pickle=False)
    np.save(cache/'y.npy',y,allow_pickle=False)
    np.save(cache/'race_idx.npy',ri,allow_pickle=False)
    np.save(cache/'base_trio_market.npy',qm,allow_pickle=False)
    np.save(cache/'baseline_king.npy',bk,allow_pickle=False)
    np.save(cache/'baseline_win_market.npy',bw,allow_pickle=False)
    write_csv(cache/'races.csv',race_meta)

    meta={
        'contract':'L2_TRIO_MARKET_RESIDUAL_MATRIX_V3','year':year,'races':race_index,'rows':int(len(y)),
        'positives':int(y.sum()),'feature_count':int(X.shape[1]),'feature_names':feature_names,
        'excluded_races':dict(excluded),'missing_trio_market_combos':int(missing_market_combos),
        'mean_trio_market_overround':float(np.mean([r['trio_market_overround'] for r in race_meta])),
        'float_dtype':'float32'
    }
    done.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'L2_TRIO_V3_MATRIX_READY year={year} races={race_index} rows={len(y)} excluded={dict(excluded)}',flush=True)
    return meta

def load_matrix(cache,year):
    root=Path(cache)/str(year)
    meta=json.loads((root/'meta.json').read_text(encoding='utf-8'))
    return (
        meta,
        np.load(root/'X.npy',mmap_mode='r'),
        np.load(root/'y.npy',mmap_mode='r'),
        np.load(root/'race_idx.npy',mmap_mode='r'),
        np.load(root/'base_trio_market.npy',mmap_mode='r'),
        np.load(root/'baseline_king.npy',mmap_mode='r'),
        np.load(root/'baseline_win_market.npy',mmap_mode='r'),
    )

def group_layout(race_idx):
    ri=np.asarray(race_idx,dtype=np.int64)
    if len(ri)==0: return np.array([],dtype=np.int64),np.array([],dtype=np.int64)
    starts=np.r_[0,1+np.where(ri[1:]!=ri[:-1])[0]].astype(np.int64)
    counts=np.diff(np.r_[starts,len(ri)]).astype(np.int64)
    return starts,counts

def offset_softmax(scores,base_q,race_idx,alpha=1.0):
    s=np.asarray(scores,dtype=np.float64)
    q=np.clip(np.asarray(base_q,dtype=np.float64),EPS,None)
    starts,counts=group_layout(race_idx)
    z=np.log(q)+float(alpha)*s
    mx=np.maximum.reduceat(z,starts)
    ex=np.exp(z-np.repeat(mx,counts))
    sm=np.add.reduceat(ex,starts)
    p=ex/np.repeat(sm,counts)
    return p

def residual_objective(base_q,race_idx):
    q=np.clip(np.asarray(base_q,dtype=np.float64),EPS,None)
    starts,counts=group_layout(race_idx)
    logq=np.log(q)
    rep_counts=counts
    def obj(y_true,y_pred):
        z=logq+np.asarray(y_pred,dtype=np.float64)
        mx=np.maximum.reduceat(z,starts)
        ex=np.exp(z-np.repeat(mx,rep_counts))
        sm=np.add.reduceat(ex,starts)
        p=ex/np.repeat(sm,rep_counts)
        y=np.asarray(y_true,dtype=np.float64)
        grad=p-y
        hess=np.maximum(p*(1.0-p),1e-6)
        return grad,hess
    return obj

def race_logloss(y,p,race_idx):
    y=np.asarray(y,dtype=np.uint8); p=np.asarray(p,dtype=np.float64); ri=np.asarray(race_idx,dtype=np.int64)
    starts,counts=group_layout(ri); vals=[]
    for s,c in zip(starts,counts):
        e=s+c; pos=np.where(y[s:e]==1)[0]
        if len(pos)==1: vals.append(-math.log(max(float(p[s+pos[0]]),EPS)))
    return float(np.mean(vals)) if vals else None

def choose_alpha(scores,y,base_q,race_idx):
    grid=np.r_[0.0,np.linspace(0.10,1.50,15),2.0]
    best=(0.0,float('inf'))
    rows=[]
    for a in grid:
        p=offset_softmax(scores,base_q,race_idx,float(a))
        ll=race_logloss(y,p,race_idx)
        rows.append((float(a),ll))
        if ll is not None and ll<best[1]: best=(float(a),float(ll))
    return best,rows

def reindex_groups(ri):
    ri=np.asarray(ri,dtype=np.int64)
    starts,counts=group_layout(ri)
    return np.repeat(np.arange(len(starts),dtype=np.int32),counts)

def concat_train(parts):
    X=np.concatenate([np.asarray(p[0]) for p in parts],axis=0).astype(np.float32,copy=False)
    y=np.concatenate([np.asarray(p[1]) for p in parts],axis=0).astype(np.uint8,copy=False)
    q=np.concatenate([np.asarray(p[2]) for p in parts],axis=0).astype(np.float32,copy=False)
    group_counts=[]
    for _,_,_,ri in parts:
        _,counts=group_layout(ri); group_counts.extend(counts.tolist())
    ri=np.repeat(np.arange(len(group_counts),dtype=np.int32),np.asarray(group_counts,dtype=np.int64))
    return X,y,q,ri

def model_params(objective):
    return dict(
        objective=objective,n_estimators=130,learning_rate=0.04,num_leaves=15,max_depth=5,max_bin=63,
        min_child_samples=400,subsample=0.9,subsample_freq=1,colsample_bytree=0.80,
        reg_lambda=5.0,reg_alpha=0.25,random_state=20261004,n_jobs=-1,verbosity=-1,force_col_wise=True
    )

def main():
    t0=time.time(); a=parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    contract=json.loads(Path(a.contract).read_text(encoding='utf-8'))
    l175=json.loads(Path(a.l175_contract).read_text(encoding='utf-8'))
    if contract.get('contract')!='L2_TRIO_MARKET_RESIDUAL_V3': raise SystemExit('wrong V3 contract')
    if l175.get('contract')!='L175_MARKET_CONTEXT_V1' or l175.get('ranking_policy')!='SEVEN_KING_UNCHANGED':
        raise SystemExit('wrong L1.75 contract')
    if contract['cost_policy']['github_standard_cpu_only'] is not True or contract['cost_policy']['gpu'] is not False:
        raise SystemExit('cost guard drift')

    _,df=load_horse_dataset(a.dataset_dir)
    df,skipped=attach_win_market(df,a.backfill_root)
    df=attach_outsider(df,a.outsider_predictions)
    df=prepare_l175(df)

    coverage=[]
    for y in YEARS:
        q=df[df['year']==y]
        coverage.append({
            'year':y,'input_races':q['race_id'].nunique(),'horses':len(q),
            'outsider_available_horses':int(q['outsider_available'].sum()),
            'outsider_rank_mismatch_horses':int(q['outsider_rank_mismatch'].sum())
        })
        build_year_matrix(y,df,a.backfill_root,a.matrix_cache)
    write_csv(out/'coverage.csv',coverage); write_csv(out/'skipped-win-market-races.csv',skipped)

    loaded={y:load_matrix(a.matrix_cache,y) for y in YEARS}
    fold_rows=[]; imp_rows=[]; alpha_rows=[]; runtime_rows=[]

    for test_year,train_years in FOLDS:
        ft=time.time()
        recent=train_years[-1]
        rmeta,Xr,yr,rir,qmr,_,_=loaded[recent]
        cut=max(1,int(rmeta['races']*0.80))
        ria=np.asarray(rir)
        fit_mask=ria<cut; cal_mask=~fit_mask

        fit_parts=[]
        for y in train_years[:-1]:
            _,X,yy,ri,qm,_,_=loaded[y]
            fit_parts.append((X,yy,qm,ri))
        fit_parts.append((
            np.asarray(Xr)[fit_mask],np.asarray(yr)[fit_mask],
            np.asarray(qmr)[fit_mask],reindex_groups(ria[fit_mask])
        ))
        Xfit,yfit,qfit,rifit=concat_train(fit_parts)
        Xcal=np.asarray(Xr)[cal_mask].astype(np.float32,copy=False)
        ycal=np.asarray(yr)[cal_mask].astype(np.uint8,copy=False)
        qcal=np.asarray(qmr)[cal_mask].astype(np.float32,copy=False)
        rical=reindex_groups(ria[cal_mask])

        m0=lgb.LGBMRegressor(**model_params(residual_objective(qfit,rifit)))
        m0.fit(Xfit,yfit)
        scal=np.asarray(m0.predict(Xcal),dtype=np.float64)
        (alpha,cal_ll),agrid=choose_alpha(scal,ycal,qcal,rical)
        for av,ll in agrid:
            alpha_rows.append({'test_year':test_year,'alpha':av,'calibration_race_log_loss':ll,'selected':int(abs(av-alpha)<1e-12)})

        full_parts=[]
        for y in train_years:
            _,X,yy,ri,qm,_,_=loaded[y]
            full_parts.append((X,yy,qm,ri))
        Xtrain,ytrain,qtrain,ritrain=concat_train(full_parts)
        model=lgb.LGBMRegressor(**model_params(residual_objective(qtrain,ritrain)))
        model.fit(Xtrain,ytrain)

        meta,Xtest,ytest,ritest,qtest,bk,bw=loaded[test_year]
        score=np.asarray(model.predict(Xtest),dtype=np.float64)
        pcorr=offset_softmax(score,qtest,ritest,alpha)
        row={
            'test_year':test_year,'train_years':'|'.join(map(str,train_years)),
            'selected_alpha':alpha,'calibration_race_log_loss':cal_ll,
            'train_rows':int(len(ytrain)),'test_rows':int(len(ytest)),
            'score_std':float(np.std(score)),'mean_abs_log_multiplier':float(np.mean(np.abs(alpha*score)))
        }
        for name,prob in [
            ('CORRECTED_TRIO_MARKET',pcorr),
            ('PURE_TRIO_MARKET',qtest),
            ('BASE_KING_PL',bk),
            ('BASE_WIN_MARKET_PL',bw),
        ]:
            mm=metrics(ytest,prob,ritest)
            for k,v in mm.items(): row[f'{name.lower()}_{k}']=v
        row['logloss_delta_vs_pure_trio_market']=row['corrected_trio_market_race_log_loss']-row['pure_trio_market_race_log_loss']
        row['brier_delta_vs_pure_trio_market']=row['corrected_trio_market_multiclass_brier']-row['pure_trio_market_multiclass_brier']
        fold_rows.append(row)

        gains=model.booster_.feature_importance(importance_type='gain')
        names=meta['feature_names']
        for n,gain in sorted(zip(names,gains),key=lambda x:-x[1]):
            imp_rows.append({'test_year':test_year,'feature':n,'gain':float(gain)})

        runtime_rows.append({'test_year':test_year,'seconds':time.time()-ft,'train_rows':int(len(ytrain)),'test_rows':int(len(ytest))})
        print('L2_TRIO_V3_FOLD_DONE '+json.dumps({
            'test_year':test_year,'alpha':alpha,
            'corrected_logloss':row['corrected_trio_market_race_log_loss'],
            'pure_market_logloss':row['pure_trio_market_race_log_loss'],
            'delta':row['logloss_delta_vs_pure_trio_market'],
            'corrected_top10':row['corrected_trio_market_winner_top10_pct'],
            'pure_market_top10':row['pure_trio_market_winner_top10_pct']
        },separators=(',',':')),flush=True)

        del Xfit,yfit,qfit,rifit,Xcal,ycal,qcal,m0,scal,Xtrain,ytrain,qtrain,ritrain,model,score,pcorr

    write_csv(out/'fold-metrics.csv',fold_rows)
    write_csv(out/'feature-importance.csv',imp_rows)
    write_csv(out/'alpha-calibration.csv',alpha_rows)
    write_csv(out/'runtime.csv',runtime_rows)

    matrix_meta={str(y):loaded[y][0] for y in YEARS}
    all_deltas=[float(r['logloss_delta_vs_pure_trio_market']) for r in fold_rows]
    summary={
        'contract':'L2_TRIO_MARKET_RESIDUAL_V3_RESULT',
        'architecture':'TRIO_MARKET_OFFSET_PLUS_L175_MULTIPLICATIVE_RESIDUAL',
        'historical_market_proxy':'FINAL_TRIO_ODDS',
        'operational_market_requirement':'LATEST_TIMESTAMPED_PRE_RACE_TRIO_ODDS',
        'trio_market_probability_used_as_offset':True,
        'trio_market_probability_used_as_feature':False,
        'payout_used':False,'roi_used':False,'2026_locked':True,
        'matrix_meta':matrix_meta,'coverage':coverage,'skipped_win_market_races':len(skipped),
        'folds':fold_rows,'runtime_seconds_total':time.time()-t0,
        'beats_pure_trio_market_all_years':all(x<0 for x in all_deltas),
        'promotion':False
    }
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text(
        '# L2 TRIO Market Residual V3\n\n'
        'Normalized TRIO market probability is the base distribution. L1.75 learns only a multiplicative residual correction. '
        'Historical final TRIO odds are a proxy; operational use requires latest timestamped pre-race TRIO odds. '
        'No payout, ROI, BUY/SKIP, budget or stakes are learned. 2026 remains sealed.\n',
        encoding='utf-8'
    )
    maxerr=max(float(r.get('corrected_trio_market_probability_mass_max_error') or 0.0) for r in fold_rows)
    if maxerr>1e-8: raise SystemExit(f'V3 probability invariant failed max_error={maxerr}')
    print('L2_TRIO_MARKET_RESIDUAL_V3_READY')
    print(json.dumps({'folds':len(fold_rows),'runtime_seconds':time.time()-t0,'max_mass_error':maxerr,'out_dir':str(out)},separators=(',',':')))

if __name__=='__main__':
    main()
