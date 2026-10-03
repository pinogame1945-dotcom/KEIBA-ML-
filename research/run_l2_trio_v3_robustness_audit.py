#!/usr/bin/env python3
import argparse,gc,json,math,time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l2_trio_probability_v2 import YEARS,FOLDS,write_csv,load_horse_dataset,attach_win_market,attach_outsider,prepare_l175
from run_l2_trio_market_residual_v3 import (
    build_year_matrix,load_matrix,group_layout,offset_softmax,residual_objective,
    choose_alpha,concat_train,model_params
)

EPS=1e-12
BOOT_REPS=5000
SEED=20261004
ALPHA_GRID=np.round(np.arange(0.0,1.5001,0.1),10)

def parse_args():
    p=argparse.ArgumentParser(description='Robustness audit for L2 TRIO Market Residual V3.')
    p.add_argument('--contract',required=True)
    p.add_argument('--l175-contract',required=True)
    p.add_argument('--dataset-dir',required=True)
    p.add_argument('--backfill-root',required=True)
    p.add_argument('--outsider-predictions',required=True)
    p.add_argument('--matrix-cache',required=True)
    p.add_argument('--out-dir',required=True)
    return p.parse_args()

def feature_keep(names,variant):
    names=list(names)
    if variant=='FULL_V3':
        return np.arange(len(names),dtype=int)
    keep=[]
    for i,n in enumerate(names):
        low=n.lower()
        if variant=='NO_OUTSIDER':
            drop=(
                'outsider_' in low or 'p3_' in low or low=='p3_sum' or
                'alignment_' in low or low=='alignment_score_mean'
            )
        elif variant=='NO_SEVEN_KING':
            drop=(
                'king_' in low or 'signed_gap' in low or 'abs_gap' in low or
                'dissent_' in low or 'p3_' in low or low=='p3_sum' or
                'alignment_' in low or low in {
                    'race_entropy','race_top3_probability_sum','race_top1_top2_gap'
                }
            )
        else:
            raise ValueError(variant)
        if not drop: keep.append(i)
    if not keep: raise ValueError(f'empty feature set variant={variant}')
    return np.asarray(keep,dtype=int)

def subset_part(part,keep):
    X,y,q,ri=part
    return np.asarray(X)[:,keep],y,q,ri

def reindex_groups(ri):
    ri=np.asarray(ri,dtype=np.int64)
    starts,counts=group_layout(ri)
    return np.repeat(np.arange(len(starts),dtype=np.int32),counts)

def prepare_fold_parts(loaded,train_years):
    recent=train_years[-1]
    rmeta,Xr,yr,rir,qmr,_,_=loaded[recent]
    cut=max(1,int(rmeta['races']*0.80))
    ria=np.asarray(rir)
    fit_mask=ria<cut
    cal_mask=~fit_mask
    fit_parts=[]
    for y in train_years[:-1]:
        _,X,yy,ri,qm,_,_=loaded[y]
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
        _,X,yy,ri,qm,_,_=loaded[y]
        full_parts.append((X,yy,qm,ri))
    return fit_parts,cal,full_parts

def fit_variant(variant,names,fit_parts,cal,full_parts,test_tuple):
    keep=feature_keep(names,variant)
    fit_sub=[subset_part(p,keep) for p in fit_parts]
    Xfit,yfit,qfit,rifit=concat_train(fit_sub)
    Xcal,ycal,qcal,rical=cal
    Xcal=np.asarray(Xcal)[:,keep].astype(np.float32,copy=False)

    m0=lgb.LGBMRegressor(**model_params(residual_objective(qfit,rifit)))
    m0.fit(Xfit,yfit)
    scal=np.asarray(m0.predict(Xcal),dtype=np.float64)
    (alpha,cal_ll),_=choose_alpha(scal,ycal,qcal,rical)
    del Xfit,yfit,qfit,rifit,Xcal,m0,scal,fit_sub
    gc.collect()

    full_sub=[subset_part(p,keep) for p in full_parts]
    Xtrain,ytrain,qtrain,ritrain=concat_train(full_sub)
    model=lgb.LGBMRegressor(**model_params(residual_objective(qtrain,ritrain)))
    model.fit(Xtrain,ytrain)

    meta,Xtest,ytest,ritest,qtest,bk,bw=test_tuple
    score=np.asarray(model.predict(np.asarray(Xtest)[:,keep]),dtype=np.float64)
    p=offset_softmax(score,qtest,ritest,alpha)
    out={
        'variant':variant,'alpha':float(alpha),'calibration_logloss':float(cal_ll),
        'feature_count':int(len(keep)),'score':score,'p':p
    }
    del Xtrain,ytrain,qtrain,ritrain,model,full_sub
    gc.collect()
    return out

def race_vectors(y,p,q,race_idx):
    y=np.asarray(y,dtype=np.uint8); p=np.asarray(p,dtype=np.float64); q=np.asarray(q,dtype=np.float64)
    starts,counts=group_layout(race_idx)
    ll_delta=[]; br_delta=[]; true_p=[]; true_q=[]
    for s,c in zip(starts,counts):
        e=s+c; yy=y[s:e]; pos=np.where(yy==1)[0]
        if len(pos)!=1: continue
        w=s+int(pos[0])
        pp=np.asarray(p[s:e],dtype=np.float64); qq=np.asarray(q[s:e],dtype=np.float64)
        tp=max(float(p[w]),EPS); tq=max(float(q[w]),EPS)
        ll_delta.append(-math.log(tp)+math.log(tq))
        br_delta.append(float(np.sum((pp-yy)**2)-np.sum((qq-yy)**2)))
        true_p.append(tp); true_q.append(tq)
    return {
        'logloss_delta':np.asarray(ll_delta,dtype=np.float64),
        'brier_delta':np.asarray(br_delta,dtype=np.float64),
        'true_p':np.asarray(true_p,dtype=np.float64),
        'true_q':np.asarray(true_q,dtype=np.float64),
    }

def bootstrap_mean_ci(values,reps=BOOT_REPS,seed=SEED):
    v=np.asarray(values,dtype=np.float64)
    n=len(v)
    if n<2: return {'n':n,'mean':float(v.mean()) if n else None,'ci_low':None,'ci_high':None,'p_improve':None}
    rng=np.random.default_rng(seed)
    means=np.empty(reps,dtype=np.float64)
    chunk=100
    pos=0
    while pos<reps:
        k=min(chunk,reps-pos)
        idx=rng.integers(0,n,size=(k,n),endpoint=False)
        means[pos:pos+k]=v[idx].mean(axis=1)
        pos+=k
    lo,hi=np.quantile(means,[0.025,0.975])
    return {'n':n,'mean':float(v.mean()),'ci_low':float(lo),'ci_high':float(hi),'p_improve':float(np.mean(means<0))}

def race_meta(cache,year):
    p=Path(cache)/str(year)/'races.csv'
    z=pd.read_csv(p)
    z['race_id']=z['race_id'].astype(str)
    z['race_date']=z['race_date'].astype(str)
    z['month']=z['race_date'].str.slice(5,7)
    z['course_code']=z['race_id'].str.slice(4,6)
    fs=pd.to_numeric(z['field_size'],errors='raise').astype(int)
    z['field_size_bucket']=np.select(
        [fs<=10,fs<=13,fs<=16,fs<=18],
        ['<=10','11-13','14-16','17-18'],
        default='19+'
    )
    return z.sort_values('race_index').reset_index(drop=True)

def slice_rows(year,rv,meta,min_n=100):
    if len(meta)!=len(rv['logloss_delta']):
        raise SystemExit(f'race meta length mismatch year={year} meta={len(meta)} vec={len(rv["logloss_delta"])}')
    z=meta.copy()
    z['ll_delta']=rv['logloss_delta']; z['brier_delta']=rv['brier_delta']
    rows=[]
    for dim in ['month','course_code','field_size_bucket']:
        for val,g in z.groupby(dim,sort=True):
            n=len(g)
            rows.append({
                'test_year':year,'dimension':dim,'value':str(val),'races':n,
                'reportable':int(n>=min_n),
                'mean_logloss_delta':float(g['ll_delta'].mean()),
                'mean_brier_delta':float(g['brier_delta'].mean()),
                'improves_logloss':int(float(g['ll_delta'].mean())<0),
            })
    return rows

def alpha_sensitivity(score,y,q,ri,year):
    rows=[]
    for a in ALPHA_GRID:
        p=offset_softmax(score,q,ri,float(a))
        rv=race_vectors(y,p,q,ri)
        rows.append({
            'test_year':year,'alpha':float(a),
            'mean_logloss_delta':float(rv['logloss_delta'].mean()),
            'mean_brier_delta':float(rv['brier_delta'].mean()),
            'improves_logloss':int(float(rv['logloss_delta'].mean())<0)
        })
    return rows

def main():
    t0=time.time(); a=parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    contract=json.loads(Path(a.contract).read_text(encoding='utf-8'))
    l175=json.loads(Path(a.l175_contract).read_text(encoding='utf-8'))
    if contract.get('contract')!='L2_TRIO_V3_ROBUSTNESS_AUDIT': raise SystemExit('wrong audit contract')
    if l175.get('contract')!='L175_MARKET_CONTEXT_V1': raise SystemExit('wrong L1.75 contract')
    if contract['bootstrap']['replicates']!=BOOT_REPS or contract['bootstrap']['seed']!=SEED: raise SystemExit('bootstrap guard drift')
    if contract['cost_policy']['github_standard_cpu_only'] is not True or contract['cost_policy']['gpu'] is not False: raise SystemExit('cost guard drift')

    _,df=load_horse_dataset(a.dataset_dir)
    df,skipped=attach_win_market(df,a.backfill_root)
    df=attach_outsider(df,a.outsider_predictions)
    df=prepare_l175(df)
    for y in YEARS:
        build_year_matrix(y,df,a.backfill_root,a.matrix_cache)

    loaded={y:load_matrix(a.matrix_cache,y) for y in YEARS}
    bootstrap_rows=[]; slice_out=[]; alpha_out=[]; ablation_rows=[]; runtime_rows=[]
    pooled_ll=[]; pooled_br=[]; yearly_primary={}

    for test_year,train_years in FOLDS:
        ft=time.time()
        fit_parts,cal,full_parts=prepare_fold_parts(loaded,train_years)
        test_tuple=loaded[test_year]
        meta,Xtest,ytest,ritest,qtest,bk,bw=test_tuple
        names=meta['feature_names']

        variants={}
        for variant in ['FULL_V3','NO_OUTSIDER','NO_SEVEN_KING']:
            variants[variant]=fit_variant(variant,names,fit_parts,cal,full_parts,test_tuple)
            rv=race_vectors(ytest,variants[variant]['p'],qtest,ritest)
            ablation_rows.append({
                'test_year':test_year,'variant':variant,'selected_alpha':variants[variant]['alpha'],
                'feature_count':variants[variant]['feature_count'],
                'mean_logloss_delta_vs_market':float(rv['logloss_delta'].mean()),
                'mean_brier_delta_vs_market':float(rv['brier_delta'].mean()),
                'races':len(rv['logloss_delta'])
            })

        market_rv=race_vectors(ytest,qtest,qtest,ritest)
        ablation_rows.append({
            'test_year':test_year,'variant':'MARKET_ONLY','selected_alpha':0.0,'feature_count':0,
            'mean_logloss_delta_vs_market':0.0,'mean_brier_delta_vs_market':0.0,
            'races':len(market_rv['logloss_delta'])
        })

        full=variants['FULL_V3']
        rv=race_vectors(ytest,full['p'],qtest,ritest)
        yearly_primary[test_year]=rv
        pooled_ll.append(rv['logloss_delta']); pooled_br.append(rv['brier_delta'])

        for metric,key in [('race_logloss_delta','logloss_delta'),('race_multiclass_brier_delta','brier_delta')]:
            b=bootstrap_mean_ci(rv[key],seed=SEED+test_year+(0 if key=='logloss_delta' else 10000))
            bootstrap_rows.append({'scope':str(test_year),'metric':metric,**b})

        m=race_meta(a.matrix_cache,test_year)
        slice_out.extend(slice_rows(test_year,rv,m,min_n=int(contract['stability_slices']['minimum_races_to_report'])))
        alpha_out.extend(alpha_sensitivity(full['score'],ytest,qtest,ritest,test_year))

        runtime_rows.append({'test_year':test_year,'seconds':time.time()-ft})
        print('L2_TRIO_V3_AUDIT_FOLD_DONE '+json.dumps({
            'year':test_year,'alpha':full['alpha'],
            'll_delta':float(rv['logloss_delta'].mean()),
            'brier_delta':float(rv['brier_delta'].mean()),
            'no_outsider_ll_delta':ablation_rows[-4]['mean_logloss_delta_vs_market'],
            'no_king_ll_delta':ablation_rows[-3]['mean_logloss_delta_vs_market']
        },separators=(',',':')),flush=True)

        del variants,fit_parts,cal,full_parts
        gc.collect()

    pll=np.concatenate(pooled_ll); pbr=np.concatenate(pooled_br)
    for metric,v,seed in [
        ('race_logloss_delta',pll,SEED+999),
        ('race_multiclass_brier_delta',pbr,SEED+1999)
    ]:
        b=bootstrap_mean_ci(v,seed=seed)
        bootstrap_rows.append({'scope':'POOLED_2023_2025','metric':metric,**b})

    log_boot={r['scope']:r for r in bootstrap_rows if r['metric']=='race_logloss_delta'}
    all_point=all(float(yearly_primary[y]['logloss_delta'].mean())<0 for y in [2023,2024,2025])
    pooled_ci=log_boot['POOLED_2023_2025']['ci_high']<0
    yearly_ci_count=sum(log_boot[str(y)]['ci_high']<0 for y in [2023,2024,2025])
    primary_pass=bool(all_point and pooled_ci and yearly_ci_count>=2)

    reportable=[r for r in slice_out if r['reportable']==1]
    seg_improve=sum(r['improves_logloss'] for r in reportable)
    seg_share=seg_improve/len(reportable) if reportable else None

    alpha_summary=[]
    for y in [2023,2024,2025]:
        rows=[r for r in alpha_out if r['test_year']==y]
        improving=[r['alpha'] for r in rows if r['improves_logloss']==1]
        alpha_summary.append({
            'test_year':y,'improving_grid_points':len(improving),
            'min_improving_alpha':min(improving) if improving else None,
            'max_improving_alpha':max(improving) if improving else None
        })

    write_csv(out/'bootstrap.csv',bootstrap_rows)
    write_csv(out/'stability-slices.csv',slice_out)
    write_csv(out/'alpha-sensitivity.csv',alpha_out)
    write_csv(out/'ablations.csv',ablation_rows)
    write_csv(out/'runtime.csv',runtime_rows)
    write_csv(out/'alpha-summary.csv',alpha_summary)
    write_csv(out/'skipped-win-market-races.csv',skipped)

    summary={
        'contract':'L2_TRIO_V3_ROBUSTNESS_AUDIT_RESULT',
        'subject':'L2_TRIO_MARKET_RESIDUAL_V3',
        'primary_rule_predeclared':contract['primary_rule'],
        'primary_pass':primary_pass,
        'all_three_year_point_estimates_improve':all_point,
        'pooled_logloss_bootstrap_ci_upper_below_zero':pooled_ci,
        'yearly_logloss_ci_upper_below_zero_count':yearly_ci_count,
        'bootstrap':bootstrap_rows,
        'reportable_segments':len(reportable),
        'reportable_segments_improving':seg_improve,
        'reportable_segment_improvement_share':seg_share,
        'alpha_summary':alpha_summary,
        'operational_caveat':contract['operational_caveat'],
        '2026_locked':True,
        'runtime_seconds_total':time.time()-t0,
        'promotion':False
    }
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text(
        '# L2 TRIO V3 Robustness Audit\n\n'
        'Predeclared stress test of the frozen V3 market-residual probability improvement. '
        'Race-cluster bootstrap, month/course/field-size stability, alpha sensitivity, and feature-family ablations. '
        'No ROI optimization, payout target, BUY/SKIP tuning, or 2026 data.\n',
        encoding='utf-8'
    )
    print('L2_TRIO_V3_ROBUSTNESS_AUDIT_READY')
    print(json.dumps({
        'primary_pass':primary_pass,'yearly_ci_passes':yearly_ci_count,
        'pooled_ci_high':log_boot['POOLED_2023_2025']['ci_high'],
        'reportable_segment_improvement_share':seg_share,
        'runtime_seconds':time.time()-t0
    },separators=(',',':')))

if __name__=='__main__':
    main()
