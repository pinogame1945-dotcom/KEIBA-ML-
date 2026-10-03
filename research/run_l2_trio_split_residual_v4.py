#!/usr/bin/env python3
import argparse,gc,itertools,json,math,os,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l2_trio_probability_v2 import (
    YEARS,FOLDS,write_csv,load_horse_dataset,attach_win_market,attach_outsider,prepare_l175
)
from run_l2_trio_market_residual_v3 import (
    build_year_matrix,load_matrix,group_layout,offset_softmax,residual_objective,choose_alpha,model_params
)
from run_l2_trio_v3_robustness_audit import bootstrap_mean_ci,race_vectors,feature_keep

EPS=1e-12
BOOT_REPS=5000
SEED=20261004

OUTSIDER_FEATURE_NAMES=[
    'h1_market_rank_pct','h1_log_market_win_odds','h1_outsider_available','h1_outsider_score_scaled',
    'h2_market_rank_pct','h2_log_market_win_odds','h2_outsider_available','h2_outsider_score_scaled',
    'h3_market_rank_pct','h3_log_market_win_odds','h3_outsider_available','h3_outsider_score_scaled',
    'field_size_norm','market_rank_mean_pct','market_rank_span_pct',
    'outsider_available_share','outsider_score_mean','outsider_score_max','outsider_score_span'
]

def parse_args():
    p=argparse.ArgumentParser(description='Parallel split residual TRIO V4.')
    p.add_argument('--contract',required=True)
    p.add_argument('--l175-contract',required=True)
    p.add_argument('--dataset-dir',required=True)
    p.add_argument('--backfill-root',required=True)
    p.add_argument('--outsider-predictions',required=True)
    p.add_argument('--matrix-cache',required=True)
    p.add_argument('--out-dir',required=True)
    return p.parse_args()

def reindex_groups(ri):
    ri=np.asarray(ri,dtype=np.int64)
    starts,counts=group_layout(ri)
    return np.repeat(np.arange(len(starts),dtype=np.int32),counts)

def copy_columns_chunked(src_path,dst_path,keep,rows,chunk=200000):
    src=np.load(src_path,mmap_mode='r')
    if src.shape[0]!=rows:
        raise SystemExit(f'row mismatch source={src.shape[0]} expected={rows}')
    dst=np.lib.format.open_memmap(dst_path,mode='w+',dtype=np.float32,shape=(rows,len(keep)))
    for s in range(0,rows,chunk):
        e=min(rows,s+chunk)
        dst[s:e]=np.asarray(src[s:e,keep],dtype=np.float32)
    dst.flush()
    del dst,src

def outsider_combo_features(g,comb):
    n=len(g); fs=float(n)
    market_rank=g['market_rank'].to_numpy(dtype=np.float32)
    order=np.argsort(market_rank[comb],axis=1,kind='stable')
    ordered=np.take_along_axis(comb,order,axis=1)
    mr=(market_rank/fs).astype(np.float32)
    lo=g['log_market_win_odds'].to_numpy(dtype=np.float32)
    oa=g['outsider_available'].to_numpy(dtype=np.float32)
    oscore=g['outsider_score_scaled'].to_numpy(dtype=np.float32)
    per=np.stack([mr,lo,oa,oscore],axis=1)
    X3=per[ordered].reshape(len(comb),-1).astype(np.float32,copy=False)
    mr3=mr[comb]; oa3=oa[comb]; os3=oscore[comb]
    agg=np.column_stack([
        np.full(len(comb),fs/18.0,dtype=np.float32),
        mr3.mean(1),mr3.max(1)-mr3.min(1),
        oa3.mean(1),os3.mean(1),os3.max(1),os3.max(1)-os3.min(1)
    ]).astype(np.float32,copy=False)
    X=np.concatenate([X3,agg],axis=1).astype(np.float32,copy=False)
    if X.shape[1]!=len(OUTSIDER_FEATURE_NAMES):
        raise SystemExit(f'outsider feature count mismatch {X.shape[1]}')
    return X

def build_compact_engine_matrices(year,df,matrix_cache):
    root=Path(matrix_cache)/str(year)
    meta=json.loads((root/'meta.json').read_text(encoding='utf-8'))
    rows=int(meta['rows'])
    engine_meta_path=root/'engine-v4-meta.json'
    king_path=root/'X_king_v4.npy'
    outsider_path=root/'X_outsider_v4.npy'
    if engine_meta_path.exists() and king_path.exists() and outsider_path.exists():
        em=json.loads(engine_meta_path.read_text(encoding='utf-8'))
        if em.get('contract')=='L2_TRIO_SPLIT_RESIDUAL_MATRIX_V4' and em.get('rows')==rows:
            print(f'L2_TRIO_V4_ENGINE_MATRIX_REUSE year={year} rows={rows}',flush=True)
            return em

    full_names=list(meta['feature_names'])
    king_keep=feature_keep(full_names,'NO_OUTSIDER')
    king_names=[full_names[i] for i in king_keep]
    copy_columns_chunked(root/'X.npy',king_path,king_keep,rows)

    races=pd.read_csv(root/'races.csv',dtype={'race_id':str,'race_date':str}).sort_values('race_index')
    ydf=df[df['year']==year].copy()
    groups={str(rid):g for rid,g in ydf.groupby('race_id',sort=False)}
    out_mm=np.lib.format.open_memmap(outsider_path,mode='w+',dtype=np.float32,shape=(rows,len(OUTSIDER_FEATURE_NAMES)))
    pos=0
    for k,r in enumerate(races.itertuples(index=False),1):
        rid=str(r.race_id)
        g=groups[rid].sort_values(['consensus_rank','horse_number','horse_id']).reset_index(drop=True)
        n=len(g)
        comb=np.asarray(list(itertools.combinations(range(n),3)),dtype=np.int16)
        expected=int(r.combos)
        if len(comb)!=expected:
            raise SystemExit(f'combo count mismatch year={year} race={rid} got={len(comb)} expected={expected}')
        Xo=outsider_combo_features(g,comb)
        out_mm[pos:pos+expected]=Xo
        pos+=expected
        if k%500==0:
            print(f'L2_TRIO_V4_OUTSIDER_MATRIX_PROGRESS year={year} races={k}/{len(races)} rows={pos}',flush=True)
    if pos!=rows:
        raise SystemExit(f'outsider row total mismatch year={year} got={pos} expected={rows}')
    out_mm.flush(); del out_mm

    em={
        'contract':'L2_TRIO_SPLIT_RESIDUAL_MATRIX_V4','year':year,'rows':rows,
        'king_feature_count':len(king_names),'king_feature_names':king_names,
        'outsider_feature_count':len(OUTSIDER_FEATURE_NAMES),'outsider_feature_names':OUTSIDER_FEATURE_NAMES,
        'outsider_canonical_order':'MARKET_RANK_ASC','king_canonical_order':'SEVEN_KING_RANK_ASC'
    }
    engine_meta_path.write_text(json.dumps(em,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'L2_TRIO_V4_ENGINE_MATRIX_READY year={year} king={len(king_names)} outsider={len(OUTSIDER_FEATURE_NAMES)} rows={rows}',flush=True)
    return em

def load_engine_year(matrix_cache,year,engine):
    root=Path(matrix_cache)/str(year)
    base=load_matrix(matrix_cache,year)
    meta,y,ri,q = base[0],base[2],base[3],base[4]
    em=json.loads((root/'engine-v4-meta.json').read_text(encoding='utf-8'))
    if engine=='KING':
        X=np.load(root/'X_king_v4.npy',mmap_mode='r')
        names=em['king_feature_names']
    elif engine=='OUTSIDER':
        X=np.load(root/'X_outsider_v4.npy',mmap_mode='r')
        names=em['outsider_feature_names']
    else:
        raise ValueError(engine)
    return meta,X,y,ri,q,names

def concat_parts(parts):
    X=np.concatenate([np.asarray(p[0]) for p in parts],axis=0).astype(np.float32,copy=False)
    y=np.concatenate([np.asarray(p[1]) for p in parts],axis=0).astype(np.uint8,copy=False)
    q=np.concatenate([np.asarray(p[2]) for p in parts],axis=0).astype(np.float32,copy=False)
    counts=[]
    for _,_,_,ri in parts:
        _,c=group_layout(ri); counts.extend(c.tolist())
    ri=np.repeat(np.arange(len(counts),dtype=np.int32),np.asarray(counts,dtype=np.int64))
    return X,y,q,ri

def prepare_engine_fold(engine_loaded,train_years):
    recent=train_years[-1]
    rmeta,Xr,yr,rir,qmr,names=engine_loaded[recent]
    cut=max(1,int(rmeta['races']*0.80))
    ria=np.asarray(rir)
    fit_mask=ria<cut; cal_mask=~fit_mask
    fit_parts=[]
    for y in train_years[:-1]:
        _,X,yy,ri,qm,_=engine_loaded[y]
        fit_parts.append((X,yy,qm,ri))
    fit_parts.append((
        np.asarray(Xr)[fit_mask],np.asarray(yr)[fit_mask],
        np.asarray(qmr)[fit_mask],reindex_groups(ria[fit_mask])
    ))
    cal=(
        np.asarray(Xr)[cal_mask].astype(np.float32,copy=False),
        np.asarray(yr)[cal_mask].astype(np.uint8,copy=False),
        np.asarray(qmr)[cal_mask].astype(np.float32,copy=False),
        reindex_groups(ria[cal_mask])
    )
    full_parts=[]
    for y in train_years:
        _,X,yy,ri,qm,_=engine_loaded[y]
        full_parts.append((X,yy,qm,ri))
    return fit_parts,cal,full_parts,names

def params_for_threads(objective,threads):
    p=model_params(objective)
    p['n_jobs']=int(max(1,threads))
    return p

def train_one_engine(engine,engine_loaded,train_years,test_year,threads):
    t0=time.time()
    fit_parts,cal,full_parts,names=prepare_engine_fold(engine_loaded,train_years)
    Xfit,yfit,qfit,rifit=concat_parts(fit_parts)
    Xcal,ycal,qcal,rical=cal
    m0=lgb.LGBMRegressor(**params_for_threads(residual_objective(qfit,rifit),threads))
    m0.fit(Xfit,yfit)
    scal=np.asarray(m0.predict(Xcal),dtype=np.float64)
    (alpha,cal_ll),alpha_grid=choose_alpha(scal,ycal,qcal,rical)
    del Xfit,yfit,qfit,rifit,Xcal,ycal,qcal,rical,m0,scal,fit_parts,cal
    gc.collect()

    Xtrain,ytrain,qtrain,ritrain=concat_parts(full_parts)
    model=lgb.LGBMRegressor(**params_for_threads(residual_objective(qtrain,ritrain),threads))
    model.fit(Xtrain,ytrain)
    meta,Xtest,ytest,ritest,qtest,_=engine_loaded[test_year]
    score=np.asarray(model.predict(Xtest),dtype=np.float64)
    p=offset_softmax(score,qtest,ritest,alpha)
    gain=model.booster_.feature_importance(importance_type='gain')
    importance=sorted(
        [{'feature':n,'gain':float(g)} for n,g in zip(names,gain)],
        key=lambda r:-r['gain']
    )
    result={
        'engine':engine,'test_year':test_year,'train_years':list(train_years),
        'alpha':float(alpha),'calibration_logloss':float(cal_ll),
        'feature_count':len(names),'threads':threads,
        'score':score,'p':p,'importance':importance,'alpha_grid':alpha_grid,
        'runtime_seconds':time.time()-t0
    }
    del Xtrain,ytrain,qtrain,ritrain,model,full_parts
    gc.collect()
    return result

def race_ids_for_year(matrix_cache,year):
    z=pd.read_csv(Path(matrix_cache)/str(year)/'races.csv',dtype={'race_id':str,'race_date':str})
    z=z.sort_values('race_index').reset_index(drop=True)
    return z[['race_index','race_id','race_date','field_size']]

def true_ticket_prob_per_race(y,p,ri):
    y=np.asarray(y,dtype=np.uint8); p=np.asarray(p,dtype=np.float64)
    starts,counts=group_layout(ri)
    out=np.empty(len(starts),dtype=np.float64)
    for k,(s,c) in enumerate(zip(starts,counts)):
        pos=np.where(y[s:s+c]==1)[0]
        if len(pos)!=1: raise SystemExit('nonunique target in diagnostics')
        out[k]=p[s+int(pos[0])]
    return out

def main():
    t0=time.time(); a=parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    contract=json.loads(Path(a.contract).read_text(encoding='utf-8'))
    l175=json.loads(Path(a.l175_contract).read_text(encoding='utf-8'))
    if contract.get('contract')!='L2_TRIO_SPLIT_RESIDUAL_V4': raise SystemExit('wrong V4 contract')
    if l175.get('contract')!='L175_MARKET_CONTEXT_V1': raise SystemExit('wrong L1.75 contract')
    if contract['cost_policy']['github_standard_cpu_only'] is not True or contract['cost_policy']['gpu'] is not False:
        raise SystemExit('cost guard drift')

    cpu=max(1,os.cpu_count() or 1)
    engine_threads=max(1,cpu//2)
    print(f'L2_TRIO_V4_CPU cpu={cpu} parallel_engines=2 threads_per_engine={engine_threads}',flush=True)

    _,df=load_horse_dataset(a.dataset_dir)
    df,skipped=attach_win_market(df,a.backfill_root)
    df=attach_outsider(df,a.outsider_predictions)
    df=prepare_l175(df)

    engine_meta=[]
    for y in YEARS:
        build_year_matrix(y,df,a.backfill_root,a.matrix_cache)
        engine_meta.append(build_compact_engine_matrices(y,df,a.matrix_cache))

    loaded={
        'KING':{y:load_engine_year(a.matrix_cache,y,'KING') for y in YEARS},
        'OUTSIDER':{y:load_engine_year(a.matrix_cache,y,'OUTSIDER') for y in YEARS}
    }

    fold_rows=[]; boot_rows=[]; h2h_rows=[]; imp_rows=[]; alpha_rows=[]; race_diag=[]; runtime_rows=[]
    pooled={'KING':[],'OUTSIDER':[]}
    pooled_h2h=[]

    for test_year,train_years in FOLDS:
        ft=time.time()
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_k=ex.submit(train_one_engine,'KING',loaded['KING'],train_years,test_year,engine_threads)
            fut_o=ex.submit(train_one_engine,'OUTSIDER',loaded['OUTSIDER'],train_years,test_year,engine_threads)
            rk=fut_k.result(); ro=fut_o.result()

        _,_,ytest,ritest,qtest,_=loaded['KING'][test_year]
        rvk=race_vectors(ytest,rk['p'],qtest,ritest)
        rvo=race_vectors(ytest,ro['p'],qtest,ritest)
        pooled['KING'].append(rvk['logloss_delta']); pooled['OUTSIDER'].append(rvo['logloss_delta'])

        for r in [rk,ro]:
            rv=rvk if r['engine']=='KING' else rvo
            fold_rows.append({
                'test_year':test_year,'engine':r['engine'],'train_years':'|'.join(map(str,train_years)),
                'selected_alpha':r['alpha'],'calibration_logloss':r['calibration_logloss'],
                'feature_count':r['feature_count'],'threads':r['threads'],
                'mean_logloss_delta_vs_market':float(rv['logloss_delta'].mean()),
                'mean_brier_delta_vs_market':float(rv['brier_delta'].mean()),
                'mean_true_ticket_probability':float(rv['true_p'].mean()),
                'market_mean_true_ticket_probability':float(rv['true_q'].mean()),
                'runtime_seconds':r['runtime_seconds']
            })
            b=bootstrap_mean_ci(rv['logloss_delta'],reps=BOOT_REPS,seed=SEED+test_year+(0 if r['engine']=='KING' else 1000))
            boot_rows.append({'scope':str(test_year),'engine':r['engine'],'metric':'race_logloss_delta',**b})
            for rank,imp in enumerate(r['importance'][:25],1):
                imp_rows.append({'test_year':test_year,'engine':r['engine'],'rank':rank,**imp})
            for av,ll in r['alpha_grid']:
                alpha_rows.append({
                    'test_year':test_year,'engine':r['engine'],'alpha':float(av),
                    'calibration_race_log_loss':float(ll),'selected':int(abs(float(av)-r['alpha'])<1e-12)
                })

        meta=race_ids_for_year(a.matrix_cache,test_year)
        tq=true_ticket_prob_per_race(ytest,qtest,ritest)
        tk=true_ticket_prob_per_race(ytest,rk['p'],ritest)
        to=true_ticket_prob_per_race(ytest,ro['p'],ritest)
        k_loss=-np.log(np.clip(tk,EPS,None)); o_loss=-np.log(np.clip(to,EPS,None)); m_loss=-np.log(np.clip(tq,EPS,None))
        k_delta=k_loss-m_loss; o_delta=o_loss-m_loss
        both=(k_delta<0)&(o_delta<0); onlyk=(k_delta<0)&~(o_delta<0); onlyo=~(k_delta<0)&(o_delta<0); neither=~(k_delta<0)&~(o_delta<0)
        best=np.where((k_loss<m_loss)&(k_loss<=o_loss),'KING',
             np.where((o_loss<m_loss)&(o_loss<k_loss),'OUTSIDER','MARKET'))
        oracle_loss=np.minimum.reduce([k_loss,o_loss,m_loss])
        oracle_delta=oracle_loss-m_loss
        pooled_h2h.append(oracle_delta)
        h2h_rows.append({
            'test_year':test_year,'races':len(k_delta),
            'king_better_than_market_pct':float(np.mean(k_delta<0)*100),
            'outsider_better_than_market_pct':float(np.mean(o_delta<0)*100),
            'both_improve_pct':float(np.mean(both)*100),
            'only_king_improves_pct':float(np.mean(onlyk)*100),
            'only_outsider_improves_pct':float(np.mean(onlyo)*100),
            'neither_improves_pct':float(np.mean(neither)*100),
            'king_beats_outsider_pct':float(np.mean(k_loss<o_loss)*100),
            'outsider_beats_king_pct':float(np.mean(o_loss<k_loss)*100),
            'oracle_best_of_market_king_outsider_mean_logloss_delta':float(oracle_delta.mean())
        })
        for i,row in meta.iterrows():
            race_diag.append({
                'year':test_year,'race_id':str(row['race_id']),'race_date':str(row['race_date']),
                'field_size':int(row['field_size']),
                'market_true_p':float(tq[i]),'king_true_p':float(tk[i]),'outsider_true_p':float(to[i]),
                'king_logloss_delta_vs_market':float(k_delta[i]),
                'outsider_logloss_delta_vs_market':float(o_delta[i]),
                'best_ex_post':str(best[i])
            })

        runtime_rows.append({
            'test_year':test_year,'wall_seconds':time.time()-ft,
            'king_model_seconds':rk['runtime_seconds'],'outsider_model_seconds':ro['runtime_seconds'],
            'parallel_threads_each':engine_threads,'cpu_count':cpu
        })
        print('L2_TRIO_V4_FOLD_DONE '+json.dumps({
            'year':test_year,
            'king_delta':float(rvk['logloss_delta'].mean()),
            'outsider_delta':float(rvo['logloss_delta'].mean()),
            'king_alpha':rk['alpha'],'outsider_alpha':ro['alpha'],
            'both_improve_pct':h2h_rows[-1]['both_improve_pct'],
            'only_king_pct':h2h_rows[-1]['only_king_improves_pct'],
            'only_outsider_pct':h2h_rows[-1]['only_outsider_improves_pct'],
            'wall_seconds':runtime_rows[-1]['wall_seconds']
        },separators=(',',':')),flush=True)
        del rk,ro,rvk,rvo,tk,to,tq,k_loss,o_loss,m_loss,k_delta,o_delta,oracle_loss,oracle_delta
        gc.collect()

    for engine in ['KING','OUTSIDER']:
        v=np.concatenate(pooled[engine])
        b=bootstrap_mean_ci(v,reps=BOOT_REPS,seed=SEED+(2000 if engine=='KING' else 3000))
        boot_rows.append({'scope':'POOLED_2023_2025','engine':engine,'metric':'race_logloss_delta',**b})

    oracle=np.concatenate(pooled_h2h)
    oracle_b=bootstrap_mean_ci(oracle,reps=BOOT_REPS,seed=SEED+4000)

    write_csv(out/'fold-metrics.csv',fold_rows)
    write_csv(out/'bootstrap.csv',boot_rows)
    write_csv(out/'head-to-head.csv',h2h_rows)
    write_csv(out/'feature-importance.csv',imp_rows)
    write_csv(out/'alpha-calibration.csv',alpha_rows)
    write_csv(out/'runtime.csv',runtime_rows)
    write_csv(out/'skipped-win-market-races.csv',skipped)
    pd.DataFrame(race_diag).to_csv(out/'race-diagnostics.csv.gz',index=False,compression='gzip')

    pooled_boot={(r['engine']):r for r in boot_rows if r['scope']=='POOLED_2023_2025'}
    summary={
        'contract':'L2_TRIO_SPLIT_RESIDUAL_V4_RESULT',
        'engines':['MARKET_PLUS_SEVEN_KING','MARKET_PLUS_OUTSIDER','MARKET_ONLY'],
        'cpu_count':cpu,'parallel_engines':2,'threads_per_engine':engine_threads,
        'matrix_meta':engine_meta,
        'folds':fold_rows,'head_to_head':h2h_rows,
        'pooled_bootstrap':pooled_boot,
        'oracle_best_of_three_upper_bound':oracle_b,
        'oracle_note':'Ex-post oracle is a non-deployable ceiling only. Future World Router must be trained/evaluated strict walk-forward and may not use these labels from its test year.',
        'future_router_ready':True,
        'operational_market_requirement':'LATEST_TIMESTAMPED_PRE_RACE_TRIO_ODDS',
        'payout_used':False,'roi_used':False,'2026_locked':True,
        'runtime_seconds_total':time.time()-t0,
        'promotion':False
    }
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text(
        '# L2 TRIO Split Residual V4\n\n'
        'Two clean market-residual engines are trained in parallel from shared preprocessing: '
        'MARKET+SEVEN_KING and MARKET+OUTSIDER. The Outsider engine is canonically ordered by market rank, '
        'not Seven-King rank, to remove hidden King dependence. Results include race-level strict walk-forward '
        'diagnostics for later World Router integration. No ROI, payout optimization, BUY/SKIP tuning, or 2026 data.\n',
        encoding='utf-8'
    )
    print('L2_TRIO_SPLIT_RESIDUAL_V4_READY')
    print(json.dumps({
        'runtime_seconds':time.time()-t0,'parallel_engines':2,'threads_each':engine_threads,
        'king_pooled_ci_high':pooled_boot['KING']['ci_high'],
        'outsider_pooled_ci_high':pooled_boot['OUTSIDER']['ci_high'],
        'oracle_mean_delta':oracle_b['mean']
    },separators=(',',':')))

if __name__=='__main__':
    main()
