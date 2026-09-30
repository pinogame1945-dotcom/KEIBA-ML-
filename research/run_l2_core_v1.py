#!/usr/bin/env python3
import argparse,csv,gzip,heapq,itertools,json,math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import load_day,load_odds_day,decode_odds,payout_map

FOLDS=((2023,(2022,)),(2024,(2022,2023)),(2025,(2022,2023,2024)))
BET_TYPES=('WIN','QUINELLA','EXACTA','TRIO','TRIFECTA')
THRESHOLDS=(0.0,0.05,0.10,0.20)
TOPK=3
EPS=1e-12

def parse_args():
    p=argparse.ArgumentParser(description='Train and evaluate L2 CORE V1 shared finish-order model.')
    p.add_argument('--contract',required=True)
    p.add_argument('--dataset-dir',required=True)
    p.add_argument('--backfill-root',required=True)
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

def load_dataset(root):
    root=Path(root); m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if m.get('contract')!='L2_CORE_DATASET_V1': raise SystemExit('wrong L2 core dataset contract')
    if m.get('locked_years')!=[2026] or m.get('odds_in_dataset') is not False: raise SystemExit('dataset guard drift')
    rows=[]
    with gzip.open(root/m['file'],'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return m,pd.DataFrame(rows)

def group_indices(race_ids):
    out=defaultdict(list)
    for i,rid in enumerate(race_ids): out[str(rid)].append(i)
    return out

def normalized_from_raw(raw,race_ids,temp):
    raw=np.asarray(raw,dtype=float); out=np.empty(len(raw),dtype=float)
    for idxs in group_indices(race_ids).values():
        z=np.clip(raw[idxs]/float(temp),-60,60); z=z-np.max(z); e=np.exp(z); s=e.sum()
        out[idxs]=e/s if s>0 else 1.0/len(idxs)
    return out

def normalize_existing(vals,race_ids):
    vals=np.asarray(vals,dtype=float); out=np.empty(len(vals),dtype=float)
    for idxs in group_indices(race_ids).values():
        v=np.clip(vals[idxs],0,None); s=v.sum(); out[idxs]=v/s if s>0 else 1.0/len(idxs)
    return out

def unique_winner_logloss(df,p):
    vals=[]
    for idxs in group_indices(df['race_id'].astype(str).tolist()).values():
        sub=df.iloc[idxs]
        if int(sub['winner_count'].iloc[0])!=1: continue
        pos=np.where(sub['target_win'].to_numpy(dtype=int)==1)[0]
        if len(pos)!=1: continue
        vals.append(-math.log(clip01(p[idxs[pos[0]]])))
    return float(np.mean(vals)) if vals else None

def horse_metrics(df,p):
    mask=(df['label_available']==True)&(df['winner_count']==1)
    orig_idx=df.index[mask].to_numpy()
    uniq=df.loc[mask].copy().reset_index(drop=True)
    if uniq.empty: return {}
    pu=np.asarray(p)[orig_idx]
    y=uniq['target_win'].to_numpy(dtype=float)
    logloss=[]; top1=0; top3=0; races=0; podium_hits=0; podium_total=0; podium_exact=0; podium_races=0
    bins=[{'n':0,'psum':0.0,'ysum':0.0} for _ in range(10)]
    for rid,idx_labels in uniq.groupby('race_id').groups.items():
        idxs=np.asarray(list(idx_labels),dtype=int); sub=uniq.loc[idxs]; pp=pu[idxs]
        winner=np.where(sub['target_win'].to_numpy(dtype=int)==1)[0]
        if len(winner)!=1: continue
        w=winner[0]; logloss.append(-math.log(clip01(pp[w]))); races+=1
        order=np.argsort(-pp,kind='stable')
        top1+=int(order[0]==w); top3+=int(w in set(order[:3]))
        t3=sub['target_top3'].to_numpy()
        if not pd.isna(t3).all():
            actual={i for i,v in enumerate(t3) if not pd.isna(v) and int(v)==1}
            if actual:
                pred=set(order[:3]); podium_hits+=len(actual & pred); podium_total+=len(actual)
                podium_races+=1; podium_exact+=int(len(actual)==3 and pred==actual)
    for prob,target in zip(pu,y):
        b=min(int(max(prob,0.0)*10),9); bins[b]['n']+=1; bins[b]['psum']+=float(prob); bins[b]['ysum']+=float(target)
    total=sum(x['n'] for x in bins); ece=0.0
    for x in bins:
        if x['n']:
            ece += (x['n']/total)*abs(x['psum']/x['n']-x['ysum']/x['n'])
    return {
        'labeled_unique_winner_races':races,'win_log_loss':float(np.mean(logloss)) if logloss else None,
        'brier_score':float(np.mean((pu-y)**2)),'calibration_error_10bin':ece,
        'top1_accuracy_pct':100*top1/races if races else None,'winner_in_top3_pct':100*top3/races if races else None,
        'official_top3_member_recall_pct':100*podium_hits/podium_total if podium_total else None,
        'official_top3_exact_set_pct':100*podium_exact/podium_races if podium_races else None,
    }

def choose_temperature(df,raw):
    temps=np.exp(np.linspace(math.log(0.30),math.log(3.00),41)); best=(None,float('inf'))
    for t in temps:
        p=normalized_from_raw(raw,df['race_id'].astype(str).tolist(),float(t))
        ll=unique_winner_logloss(df,p)
        if ll is not None and ll<best[1]: best=(float(t),ll)
    return (1.0,None) if best[0] is None else best

def ordered2(p,a,b):
    if a==b: return 0.0
    den=1.0-p[a]
    return p[a]*p[b]/den if den>EPS else 0.0

def ordered3(p,a,b,c):
    if len({a,b,c})<3: return 0.0
    den1=1.0-p[a]; den2=1.0-p[a]-p[b]
    if den1<=EPS or den2<=EPS: return 0.0
    return p[a]*(p[b]/den1)*(p[c]/den2)

def ticket_probability(bet,nums,p):
    if any(n not in p for n in nums): return None
    if bet=='WIN': return float(p[nums[0]])
    if bet=='EXACTA': return float(ordered2(p,nums[0],nums[1]))
    if bet=='QUINELLA': return float(ordered2(p,nums[0],nums[1])+ordered2(p,nums[1],nums[0]))
    if bet=='TRIFECTA': return float(ordered3(p,nums[0],nums[1],nums[2]))
    if bet=='TRIO': return float(sum(ordered3(p,*perm) for perm in itertools.permutations(nums,3)))
    return None

def probability_invariant(p):
    nums=list(p)
    win=sum(ticket_probability('WIN',(a,),p) for a in nums)
    exacta=sum(ticket_probability('EXACTA',(a,b),p) for a in nums for b in nums if a!=b)
    quin=sum(ticket_probability('QUINELLA',x,p) for x in itertools.combinations(nums,2))
    if len(nums)>=3:
        trif=sum(ticket_probability('TRIFECTA',x,p) for x in itertools.permutations(nums,3))
        trio=sum(ticket_probability('TRIO',x,p) for x in itertools.combinations(nums,3))
    else: trif=trio=1.0
    return max(abs(win-1),abs(exacta-1),abs(quin-1),abs(trif-1),abs(trio-1))

class CalAgg:
    def __init__(self,bins=20):
        self.n=0; self.brier=0.0; self.logloss=0.0; self.bins=[{'n':0,'p':0.0,'y':0.0} for _ in range(bins)]
    def add(self,p,y):
        p=clip01(p); y=float(y); self.n+=1; self.brier+=(p-y)**2; self.logloss+=-(y*math.log(p)+(1-y)*math.log(1-p))
        b=min(int(p*len(self.bins)),len(self.bins)-1); z=self.bins[b]; z['n']+=1; z['p']+=p; z['y']+=y
    def result(self):
        if not self.n: return {'tickets':0,'brier':None,'log_loss':None,'ece':None}
        ece=0.0
        for z in self.bins:
            if z['n']: ece+=(z['n']/self.n)*abs(z['p']/z['n']-z['y']/z['n'])
        return {'tickets':self.n,'brier':self.brier/self.n,'log_loss':self.logloss/self.n,'ece':ece}

def max_drawdown(profits):
    equity=0.0; peak=0.0; dd=0.0
    for x in profits:
        equity+=x; peak=max(peak,equity); dd=max(dd,peak-equity)
    return dd

def market_evaluate(test_year,test_df,pwin,backfill_root,candidate_writer):
    tmp=test_df.copy().reset_index(drop=True); tmp['p_win']=np.asarray(pwin,dtype=float)
    race_prob={}; race_date={}
    for rid,sub in tmp.groupby('race_id',sort=False):
        p={int(n):float(v) for n,v in zip(sub['horse_number'],sub['p_win'])}
        s=sum(p.values()); p={k:v/s for k,v in p.items()}
        race_prob[str(rid)]=p; race_date[str(rid)]=str(sub['race_date'].iloc[0])[:10]
    date_to=defaultdict(list)
    for rid,d in race_date.items(): date_to[d].append(rid)
    cal={b:CalAgg() for b in BET_TYPES}
    strategy={(b,t):{'tickets':0,'stake':0.0,'ret':0.0,'hit_tickets':0,'races':0,'hit_races':0,'profits':[]} for b in BET_TYPES for t in THRESHOLDS}
    skip_missing_number=defaultdict(int); market_rows=defaultdict(int); invariant_max=0.0
    root=Path(backfill_root)
    for di,date in enumerate(sorted(date_to),1):
        wanted=set(date_to[date]); day=load_day(root/'data'/'daily'/f'{date}.jsonl.gz',wanted); oddsday=load_odds_day(root/'data'/'odds'/f'daily/{date}.jsonl.gz',wanted)
        for rid in sorted(wanted):
            pack=day.get(rid); oddsrec=oddsday.get(rid)
            if pack is None or oddsrec is None: raise SystemExit(f'missing market/race row y={test_year} race={rid}')
            p=race_prob[rid]; invariant_max=max(invariant_max,probability_invariant(p))
            payouts,present=payout_map(pack); odds=decode_odds(oddsrec)
            race_acc={(b,t):{'tickets':0,'stake':0.0,'ret':0.0,'hit':0} for b in BET_TYPES for t in THRESHOLDS}
            heaps_p={b:[] for b in BET_TYPES}; heaps_e={b:[] for b in BET_TYPES}
            for (bet,nums),odd in odds.items():
                if bet not in BET_TYPES or bet not in present: continue
                ph=ticket_probability(bet,nums,p)
                if ph is None:
                    skip_missing_number[bet]+=1; continue
                ph=min(max(ph,0.0),1.0); ret=float(payouts.get((bet,nums),0.0)); hit=1 if ret>0 else 0
                cal[bet].add(ph,hit); market_rows[bet]+=1
                edge=ph*float(odd)-1.0; sel='-'.join(map(str,nums))
                item_p=(ph,sel,float(odd),edge,hit,ret); item_e=(edge,sel,float(odd),ph,hit,ret)
                if len(heaps_p[bet])<TOPK: heapq.heappush(heaps_p[bet],item_p)
                elif item_p>heaps_p[bet][0]: heapq.heapreplace(heaps_p[bet],item_p)
                if len(heaps_e[bet])<TOPK: heapq.heappush(heaps_e[bet],item_e)
                elif item_e>heaps_e[bet][0]: heapq.heapreplace(heaps_e[bet],item_e)
                for t in THRESHOLDS:
                    if edge>=t:
                        z=race_acc[(bet,t)]; z['tickets']+=1; z['stake']+=100.0; z['ret']+=ret; z['hit']+=hit
            for bet in BET_TYPES:
                for rank,item in enumerate(sorted(heaps_p[bet],reverse=True),1):
                    ph,sel,odd,edge,hit,ret=item
                    candidate_writer.writerow([test_year,date,rid,bet,'P_HIT',rank,sel,ph,odd,edge,hit,ret])
                for rank,item in enumerate(sorted(heaps_e[bet],reverse=True),1):
                    edge,sel,odd,ph,hit,ret=item
                    candidate_writer.writerow([test_year,date,rid,bet,'EDGE_FINAL_ODDS_PROXY',rank,sel,ph,odd,edge,hit,ret])
            for key,z in race_acc.items():
                if not z['tickets']: continue
                g=strategy[key]; g['tickets']+=z['tickets']; g['stake']+=z['stake']; g['ret']+=z['ret']; g['hit_tickets']+=z['hit']; g['races']+=1; g['hit_races']+=int(z['hit']>0); g['profits'].append(z['ret']-z['stake'])
        if di%25==0: print(f'L2_CORE_MARKET_PROGRESS year={test_year} dates={di}/{len(date_to)}',flush=True)
    cal_rows=[]
    for bet in BET_TYPES:
        r=cal[bet].result(); cal_rows.append({'test_year':test_year,'bet_type':bet,**r,'missing_number_tickets':skip_missing_number[bet]})
    strat_rows=[]
    for bet in BET_TYPES:
        for t in THRESHOLDS:
            g=strategy[(bet,t)]; stake=g['stake']; ret=g['ret']
            strat_rows.append({'test_year':test_year,'bet_type':bet,'edge_threshold':t,'tickets':g['tickets'],'bought_races':g['races'],'hit_tickets':g['hit_tickets'],'hit_races':g['hit_races'],'stake_yen':stake,'return_yen':ret,'profit_yen':ret-stake,'roi_pct':100*ret/stake if stake else None,'race_hit_rate_pct':100*g['hit_races']/g['races'] if g['races'] else None,'ticket_hit_rate_pct':100*g['hit_tickets']/g['tickets'] if g['tickets'] else None,'max_drawdown_yen':max_drawdown(g['profits'])})
    return cal_rows,strat_rows,invariant_max,dict(market_rows)

def main():
    a=parse_args(); contract=json.loads(Path(a.contract).read_text(encoding='utf-8'))
    if contract.get('contract')!='L2_CORE_V1' or contract.get('status')!='DESIGN_FROZEN': raise SystemExit('wrong L2 core contract')
    if contract['cost_policy']['github_standard_cpu_only'] is not True or contract['cost_policy']['gpu'] is not False: raise SystemExit('cost guard drift')
    manifest,df=load_dataset(a.dataset_dir); features=list(manifest['feature_columns'])
    for c in features: df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0.0)
    df['year']=df['year'].astype(int); df['race_id']=df['race_id'].astype(str)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    fold_rows=[]; calibration_rows=[]; market_rows=[]; importance_rows=[]; max_invariant=0.0; market_ticket_counts={}
    horse_path=out/'horse-probabilities.csv.gz'; cand_path=out/'ticket-top-candidates.csv.gz'
    with gzip.open(horse_path,'wt',newline='',encoding='utf-8') as hf, gzip.open(cand_path,'wt',newline='',encoding='utf-8') as cf:
        hw=csv.writer(hf); hw.writerow(['test_year','race_date','race_id','horse_id','horse_number','p_win','model_rank','target_win','target_top3','winner_count'])
        cw=csv.writer(cf); cw.writerow(['test_year','race_date','race_id','bet_type','rank_basis','rank','selection','p_hit','market_odds','edge','hit','return_yen_per100'])
        for test_year,train_years in FOLDS:
            train=df[df['year'].isin(train_years) & (df['train_eligible']==True) & (df['label_available']==True)].copy()
            test=df[df['year']==test_year].copy().reset_index(drop=True)
            races=train[['race_date','race_id']].drop_duplicates().sort_values(['race_date','race_id'])
            cut=max(1,int(len(races)*0.8)); fit_races=set(races.iloc[:cut]['race_id'].astype(str)); calib_races=set(races.iloc[cut:]['race_id'].astype(str))
            fit=train[train['race_id'].isin(fit_races)].copy(); calib=train[train['race_id'].isin(calib_races)].copy().reset_index(drop=True)
            params=dict(objective='binary',n_estimators=260,learning_rate=0.04,num_leaves=31,min_child_samples=50,subsample=0.9,colsample_bytree=0.9,reg_lambda=1.0,random_state=20261001,n_jobs=2,verbosity=-1)
            m0=lgb.LGBMClassifier(**params)
            m0.fit(fit[features],fit['target_win'].astype(int),sample_weight=1.0/fit['field_size'].astype(float))
            raw_cal=np.asarray(m0.booster_.predict(calib[features],raw_score=True),dtype=float)
            temp,cal_ll=choose_temperature(calib,raw_cal)
            model=lgb.LGBMClassifier(**params)
            model.fit(train[features],train['target_win'].astype(int),sample_weight=1.0/train['field_size'].astype(float))
            raw=np.asarray(model.booster_.predict(test[features],raw_score=True),dtype=float)
            p=normalized_from_raw(raw,test['race_id'].tolist(),temp)
            base=normalize_existing(test['mean_probability'].to_numpy(dtype=float),test['race_id'].tolist())
            metrics=horse_metrics(test,p); base_metrics=horse_metrics(test,base)
            row={'test_year':test_year,'train_years':'|'.join(map(str,train_years)),'fit_races':len(fit_races),'calibration_races':len(calib_races),'temperature':temp,'calibration_win_log_loss':cal_ll,**metrics}
            for k,v in base_metrics.items(): row['baseline_'+k]=v
            fold_rows.append(row)
            gains=model.booster_.feature_importance(importance_type='gain')
            for name,g in sorted(zip(features,gains),key=lambda x:-x[1]): importance_rows.append({'test_year':test_year,'feature':name,'gain':float(g)})
            ranks=np.empty(len(test),dtype=int)
            for idxs in group_indices(test['race_id'].tolist()).values():
                order=np.argsort(-p[idxs],kind='stable')
                for rank,pos in enumerate(order,1): ranks[idxs[pos]]=rank
            for i,r in test.iterrows():
                hw.writerow([test_year,r['race_date'],r['race_id'],r['horse_id'],int(r['horse_number']),float(p[i]),int(ranks[i]),r['target_win'],r['target_top3'],int(r['winner_count'])])
            crows,mrows,inv,counts=market_evaluate(test_year,test,p,a.backfill_root,cw)
            calibration_rows.extend(crows); market_rows.extend(mrows); max_invariant=max(max_invariant,inv); market_ticket_counts[str(test_year)]=counts
            print('L2_CORE_FOLD_DONE '+json.dumps({'test_year':test_year,'temperature':temp,'win_log_loss':metrics.get('win_log_loss'),'top1_accuracy_pct':metrics.get('top1_accuracy_pct'),'winner_in_top3_pct':metrics.get('winner_in_top3_pct'),'probability_invariant_max_error':inv},separators=(',',':')),flush=True)
    write_csv(out/'fold-metrics.csv',fold_rows); write_csv(out/'ticket-calibration.csv',calibration_rows); write_csv(out/'market-diagnostic.csv',market_rows); write_csv(out/'feature-importance.csv',importance_rows)
    summary={'contract':'L2_CORE_V1_RESULT','architecture':'PLACKETT_LUCE_STYLE_SHARED_STRENGTH','folds':[x[0] for x in FOLDS],'2026_locked':True,'odds_as_probability_model_feature':False,'market_price':'NETKEIBA_FINAL_HISTORICAL_PROXY','probability_invariant_max_error':max_invariant,'market_ticket_counts':market_ticket_counts,'horse_metrics':fold_rows,'ticket_calibration':calibration_rows,'market_diagnostic':market_rows,'promotion':False}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text('# L2 CORE V1 run\n\nShared horse-strength -> Plackett-Luce-style finish-order probabilities -> ticket probabilities -> final-odds historical valuation proxy. Odds are not probability-model features. Stake sizing remains L3. 2026 is sealed.\n',encoding='utf-8')
    if max_invariant>1e-8: raise SystemExit(f'probability invariant failed max_error={max_invariant}')
    print('L2_CORE_V1_READY')
    print(json.dumps({'folds':len(fold_rows),'probability_invariant_max_error':max_invariant,'out_dir':str(out)},separators=(',',':')))

if __name__=='__main__': main()
