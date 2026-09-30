#!/usr/bin/env python3
import argparse,csv,gzip,json,math,re
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import load_day,payout_map,horse_number_map
from build_l2_l17_fullfield_dataset_v1 import load_l17

YEARS=(2022,2023,2024,2025)
EXPECTED_PER_YEAR=3456

def parse_args():
    p=argparse.ArgumentParser(description='Build leak-safe horse-level dataset for L2 CORE V1 from L1.7 plus separately joined outcomes.')
    p.add_argument('--l17-year',action='append',required=True,help='YEAR:PATH')
    p.add_argument('--race-dates',required=True)
    p.add_argument('--backfill-root',required=True)
    p.add_argument('--out-dir',required=True)
    return p.parse_args()

def parse_year_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(':',1); out[int(y)]=p
    return out

def finite(v,default=0.0):
    try:
        if isinstance(v,str): v=v.replace(',','').strip()
        x=float(v)
        return x if math.isfinite(x) else default
    except (TypeError,ValueError): return default

def safe_name(x):
    return re.sub(r'[^A-Za-z0-9_]+','_',str(x)).strip('_').lower()

def entropy_norm(ps):
    vals=[max(finite(x),1e-15) for x in ps]
    s=sum(vals)
    if s<=0 or len(vals)<=1: return 0.0
    vals=[x/s for x in vals]
    h=-sum(x*math.log(x) for x in vals)
    return h/math.log(len(vals))

def load_dates(path):
    out={}
    with open(path,newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            y=int(r['year']); rid=str(r['race_id']); d=str(r['race_date'])[:10]
            if y==2026: raise SystemExit('2026 sealed')
            if rid in out: raise SystemExit(f'duplicate race date: {rid}')
            out[rid]=(y,d)
    return out

def main():
    a=parse_args(); paths=parse_year_paths(a.l17_year)
    if set(paths)!=set(YEARS): raise SystemExit(f'L1.7 years mismatch: {sorted(paths)}')
    l17={y:load_l17(paths[y],y) for y in YEARS}
    for y in YEARS:
        if len(l17[y])!=EXPECTED_PER_YEAR:
            raise SystemExit(f'L1.7 race coverage regression year={y}: {len(l17[y])} != {EXPECTED_PER_YEAR}')
    dates=load_dates(a.race_dates)
    all_rids={rid for y in YEARS for rid in l17[y]}
    if set(dates)!=all_rids:
        miss=sorted(all_rids-set(dates))[:10]; extra=sorted(set(dates)-all_rids)[:10]
        raise SystemExit(f'race date coverage mismatch missing={miss} extra={extra}')

    expert_names=sorted({name for y in YEARS for rec in l17[y].values() for h in rec['horses'] for name in (h.get('experts') or {})})
    if len(expert_names)!=7: raise SystemExit(f'expected seven experts got={expert_names}')
    expert_safe={e:safe_name(e) for e in expert_names}
    feature_columns=[
        'field_size','consensus_rank','consensus_rank_pct','mean_rank','mean_rank_pct','rank_std','rank_std_pct',
        'best_rank','best_rank_pct','worst_rank','worst_rank_pct','rank_range_pct','top1_votes','top1_vote_share',
        'top3_support','top3_support_share','top6_support','top6_support_share','mean_probability','probability_std',
        'race_entropy','race_top1_probability','race_top2_probability_sum','race_top3_probability_sum',
        'race_top1_top2_gap','race_rank_std_mean','race_rank_std_max','race_probability_std_mean','race_probability_std_max'
    ]
    for e in expert_names:
        s=expert_safe[e]; feature_columns += [f'expert_{s}_rank_pct',f'expert_{s}_probability']

    race_by_date=defaultdict(list)
    for rid,(y,d) in dates.items(): race_by_date[d].append((y,rid))
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    horse_path=out/'horse-rows.jsonl.gz'
    counts=defaultdict(int); year_races=defaultdict(int); dead_heat_races=defaultdict(int); unlabeled=defaultdict(int); top3_labeled=defaultdict(int)
    root=Path(a.backfill_root)
    with gzip.open(horse_path,'wt',encoding='utf-8') as fh:
        for di,date in enumerate(sorted(race_by_date),1):
            wanted={rid for _,rid in race_by_date[date]}
            day=load_day(root/'data'/'daily'/f'{date}.jsonl.gz',wanted)
            for year,rid in sorted(race_by_date[date]):
                pack=day.get(rid)
                if pack is None: raise SystemExit(f'missing race pack race={rid} date={date}')
                rec=l17[year][rid]; horses=list(rec['horses']); field_size=int(rec['field_size'])
                if field_size!=len(horses): raise SystemExit(f'field drift race={rid}')
                hno=horse_number_map(pack)
                if set(str(h['horse_id']) for h in horses)-set(hno):
                    raise SystemExit(f'L1.7 horse missing from race pack race={rid}')
                payouts,present=payout_map(pack)
                win_nums=sorted({key[1][0] for key in payouts if key[0]=='WIN'})
                trio_sets=[set(key[1]) for key in payouts if key[0]=='TRIO']
                podium=set().union(*trio_sets) if trio_sets else set()
                labeled=len(win_nums)>0
                train_eligible=len(win_nums)==1
                if len(win_nums)>1: dead_heat_races[year]+=1
                if not labeled: unlabeled[year]+=1
                if podium: top3_labeled[year]+=1
                year_races[year]+=1

                ordered=sorted(horses,key=lambda h:int(h['consensus_rank']))
                probs=[finite(h.get('mean_probability')) for h in ordered]
                rankstd=[finite(h.get('rank_std')) for h in ordered]
                probstd=[finite(h.get('probability_std')) for h in ordered]
                pnorm=[max(x,0.0) for x in probs]; ps=sum(pnorm)
                if ps>0: pnorm=[x/ps for x in pnorm]
                else: pnorm=[1.0/field_size]*field_size
                race_feats={
                    'race_entropy':entropy_norm(pnorm),
                    'race_top1_probability':pnorm[0] if pnorm else 0.0,
                    'race_top2_probability_sum':sum(pnorm[:2]),
                    'race_top3_probability_sum':sum(pnorm[:3]),
                    'race_top1_top2_gap':(pnorm[0]-pnorm[1]) if len(pnorm)>1 else 0.0,
                    'race_rank_std_mean':sum(rankstd)/len(rankstd) if rankstd else 0.0,
                    'race_rank_std_max':max(rankstd) if rankstd else 0.0,
                    'race_probability_std_mean':sum(probstd)/len(probstd) if probstd else 0.0,
                    'race_probability_std_max':max(probstd) if probstd else 0.0,
                }
                for h in horses:
                    hid=str(h['horse_id']); no=int(hno[hid]); cr=int(h['consensus_rank'])
                    row={
                        'contract':'L2_CORE_HORSE_ROW_V1','year':year,'race_id':rid,'race_date':date,
                        'horse_id':hid,'horse_number':no,'label_available':bool(labeled),'train_eligible':bool(train_eligible),
                        'winner_count':len(win_nums),'target_win':int(no in win_nums) if labeled else None,
                        'target_top3':int(no in podium) if podium else None,'field_size':field_size,
                        'consensus_rank':cr,'consensus_rank_pct':cr/field_size,
                        'mean_rank':finite(h.get('mean_rank'),99.0),'mean_rank_pct':finite(h.get('mean_rank'),99.0)/field_size,
                        'rank_std':finite(h.get('rank_std')),'rank_std_pct':finite(h.get('rank_std'))/field_size,
                        'best_rank':finite(h.get('best_rank'),99.0),'best_rank_pct':finite(h.get('best_rank'),99.0)/field_size,
                        'worst_rank':finite(h.get('worst_rank'),99.0),'worst_rank_pct':finite(h.get('worst_rank'),99.0)/field_size,
                        'rank_range_pct':(finite(h.get('worst_rank'),99.0)-finite(h.get('best_rank'),99.0))/field_size,
                        'top1_votes':finite(h.get('top1_votes')),'top1_vote_share':finite(h.get('top1_votes'))/7.0,
                        'top3_support':finite(h.get('top3_support')),'top3_support_share':finite(h.get('top3_support'))/7.0,
                        'top6_support':finite(h.get('top6_support')),'top6_support_share':finite(h.get('top6_support'))/7.0,
                        'mean_probability':finite(h.get('mean_probability')),'probability_std':finite(h.get('probability_std')),
                        **race_feats,
                    }
                    views=h.get('experts') or {}
                    if set(views)!=set(expert_names): raise SystemExit(f'expert coverage drift race={rid} horse={hid}')
                    for e in expert_names:
                        v=views[e]; s=expert_safe[e]
                        row[f'expert_{s}_rank_pct']=finite(v.get('rank'),99.0)/field_size
                        row[f'expert_{s}_probability']=finite(v.get('probability'))
                    fh.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                    counts[year]+=1
            if di%50==0: print(f'L2_CORE_DATASET_PROGRESS dates={di}/{len(race_by_date)}',flush=True)

    manifest={
        'contract':'L2_CORE_DATASET_V1','source':'L17_SEVEN_KING_FULLFIELD_OUTPUT_V1','years':list(YEARS),
        'races':{str(y):year_races[y] for y in YEARS},'horse_rows':{str(y):counts[y] for y in YEARS},
        'dead_heat_races':{str(y):dead_heat_races[y] for y in YEARS},'unlabeled_races':{str(y):unlabeled[y] for y in YEARS},
        'top3_labeled_races':{str(y):top3_labeled[y] for y in YEARS},'expert_names':expert_names,
        'feature_columns':feature_columns,'target':'target_win','training_excludes_multiwinner_dead_heats':True,
        'outcomes_joined_separately':True,'odds_in_dataset':False,'locked_years':[2026],'file':horse_path.name,
    }
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('L2_CORE_DATASET_V1_READY')
    print(json.dumps({'races':sum(year_races.values()),'rows':sum(counts.values()),'dead_heat_races':sum(dead_heat_races.values()),'features':len(feature_columns)},separators=(',',':')))

if __name__=='__main__': main()
