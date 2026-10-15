#!/usr/bin/env python3
"""Independent comparison using pinned own KEIBA-BACKFILL race packs, no scraping.
Official popularity is the saved result.popularity. 5-horse TRIO box: 10 x 1000 JPY.
No ML training, no 2026 model-validation unlock, no race selection tuning.
"""
import argparse, csv, gzip, json, re, collections
from pathlib import Path

VENUES={'札幌':'01','函館':'02','福島':'03','新潟':'04','東京':'05',
        '中山':'06','中京':'07','京都':'08','阪神':'09','小倉':'10'}
COST=10000

def load_targets(path):
    rows=[]
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        date,venue,*specs=line.split()
        date="-".join(f"{int(x):0{4 if i==0 else 2}d}" for i,x in enumerate(date.split('/')))
        assert venue in VENUES,(date,venue)
        for s in specs:
            num,paid=map(int,s.split('='))
            rows.append(dict(date=date,venue=venue,venue_code=VENUES[venue],
                             race_no=num,site_paid=paid))
    assert len(rows)==len(set((r['date'],r['venue'],r['race_no']) for r in rows)),"duplicate race"
    return rows

def make_paths(rows,manifest_path,output):
    manifest=json.loads(manifest_path.read_text())
    days=manifest.get('days',{})
    paths=[];missing={}
    for day in sorted(set(r['date'] for r in rows)):
        v=days.get(day)
        if v is None or v.get('status')!='SUCCESS':
            missing[day]=v.get('status','ABSENT') if v else 'ABSENT'
            continue
        path=v.get('file')
        assert path and path.startswith('data/daily/') and path.endswith('.jsonl.gz'),(day,path)
        assert '..' not in path and '\\' not in path,(day,path)
        paths.append('/'+path)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(['/data/manifest.json']+paths)+'\n')
    print('METADATA_PATHS',len(paths),'MISSING_DAYS',json.dumps(missing,ensure_ascii=False),flush=True)
    return missing

def valid_rank(r):
    try:
        rank=int(r.get('popularity'))
        return rank if rank>0 else None
    except (TypeError,ValueError):return None

def analyze(pack,r):
    race=pack.get('race') or {}
    rid=str(race.get('race_id') or '')
    assert len(rid)==12 and rid.isdigit(),rid
    assert rid[4:6]==r['venue_code'] and int(rid[-2:])==r['race_no'],(rid,r)
    assert race.get('actual_date')==r['date'],(rid,race.get('actual_date'),r['date'])
    numbers={}
    for entry in pack.get('entries') or []:
        h=str(entry.get('horse_id') or '')
        n=entry.get('horse_number')
        if h and n is not None:
            n=int(n)
            assert h not in numbers or numbers[h]==n
            numbers[h]=n
    pop={}
    finishing={}
    for result in pack.get('results') or []:
        h=str(result.get('horse_id') or '')
        p=valid_rank(result)
        if p is not None:
            assert h in numbers,(rid,h)
            assert h not in pop,(rid,h)
            pop[h]=p
        pos=result.get('official_finish_position')
        if pos is not None and str(pos).isdigit() and int(pos) in (1,2,3):
            finishing[int(pos)]=numbers[h]
    chosen={numbers[h] for h,p in pop.items() if p<=5}
    assert len(chosen)==5,(rid,'popular_count',len(chosen),sorted(pop.values()))
    assert set(finishing)=={1,2,3},(rid,'finishing',finishing)
    payouts=[]
    for p in pack.get('payouts') or []:
        if p.get('bet_type')!='TRIO':continue
        nums=tuple(sorted(map(int,re.findall(r'\d+',str(p.get('combination') or '')))))
        assert len(nums)==3 and len(set(nums))==3,(rid,p)
        amount=p.get('payout_yen')
        assert amount is not None,(rid,p)
        amount=int(float(str(amount).replace(',','')))
        assert amount>=0,(rid,p)
        payouts.append((nums,amount))
    assert payouts,(rid,'missing trio')
    finishing_numbers=tuple(sorted(finishing.values()))
    assert finishing_numbers in {n for n,v in payouts},(rid,'trio/finish mismatch',finishing_numbers,payouts)
    paid=sum(1000//100*pay for combo,pay in payouts if set(combo).issubset(chosen))
    return dict(race_id=rid,final_pop5='-'.join(map(str,sorted(chosen))),
                official_123='-'.join(map(str,finishing_numbers)),
                official_123_pop='-'.join(str(pop_horse_for_num(pop,numbers,finishing[i])) for i in (1,2,3)),
                pop5_paid=paid,pop5_hit=int(paid>0),
                site_hit=int(r['site_paid']>0),
                site_minus_pop5=r['site_paid']-paid)

def pop_horse_for_num(pop,numbers,n):
    found=[p for h,p in pop.items() if numbers[h]==n]
    assert len(found)==1,(n,found)
    return found[0]

def stats(label,rows):
    if not rows:return {}
    n=len(rows);s=sum(r['site_paid'] for r in rows);p=sum(r['pop5_paid'] for r in rows)
    d=dict(group=label,races=n,invested=n*COST,
           site_paid=s,site_hit=sum(r['site_hit'] for r in rows),
           site_roi_pct=round(s/(n*COST)*100,4),
           popularity5_paid=p,popularity5_hit=sum(r['pop5_hit'] for r in rows),
           popularity5_roi_pct=round(p/(n*COST)*100,4),
           popularity5_profit=p-n*COST,
           both_hit=sum(r['site_hit'] and r['pop5_hit'] for r in rows),
           only_popularity_hit=sum((not r['site_hit']) and r['pop5_hit'] for r in rows))
    print('STATS',json.dumps(d,ensure_ascii=False),flush=True)
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--targets',type=Path,required=True)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('research/affiliate_top5/results'))
    ap.add_argument('--prepare-paths',type=Path)
    args=ap.parse_args()
    rows=load_targets(args.targets)
    manifest=args.root/'data/manifest.json'
    if args.prepare_paths:
        make_paths(rows,manifest,args.prepare_paths)
        return
    by_key={(r['date'],r['venue_code'],r['race_no']):r for r in rows}
    assert len(by_key)==len(rows)
    days=json.loads(manifest.read_text())['days']
    by_date=collections.defaultdict(list)
    for r in rows:by_date[r['date']].append(r)
    results=[]; missing=[];errors=[];visited=set()
    for date,ts in sorted(by_date.items()):
        info=days.get(date)
        if not info or info.get('status')!='SUCCESS':
            missing.extend(dict(r,status='HISTORICAL_PACK_NOT_AVAILABLE') for r in ts)
            continue
        p=args.root/info['file']
        assert p.is_file() and p.stat().st_size>0,(date,'metadata source missing/404',p)
        found=set()
        with gzip.open(p,'rt') as f:
            for line in f:
                if not line.strip():continue
                pack=json.loads(line)
                race=pack.get('race') or {}
                rid=str(race.get('race_id') or '')
                assert len(rid)==12 and rid.isdigit(),(date,rid)
                key=(date,rid[4:6],int(rid[-2:]))
                row=by_key.get(key)
                if row is None:continue
                if key in visited:errors.append(('duplicated race',key));continue
                visited.add(key);found.add(key)
                try:results.append(dict(row,**analyze(pack,row),status='OK'))
                except Exception as e:errors.append((key,str(e)))
        for r in ts:
            key=(date,r['venue_code'],r['race_no'])
            if key not in found:errors.append(('race not in manifest-selected day',key))
    results.sort(key=lambda r:(r['date'],r['venue_code'],r['race_no']))
    missing.sort(key=lambda r:(r['date'],r['venue_code'],r['race_no']))
    args.output.mkdir(parents=True,exist_ok=True)
    cols=['date','venue','venue_code','race_no','race_id','final_pop5','official_123',
          'official_123_pop','site_paid','pop5_paid','site_minus_pop5',
          'site_hit','pop5_hit','status']
    with (args.output/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(results)
    with (args.output/'missing.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['date','venue','race_no','site_paid','status'],extrasaction='ignore')
        w.writeheader();w.writerows(missing)
    groups=[stats('MATCHED_ALL',results)]+[stats(str(y),[r for r in results if r['date'].startswith(str(y))]) for y in (2024,2025,2026)]
    site_total=sum(r['site_paid'] for r in rows)
    summary={'total_pasted_races':len(rows),'processed':len(results),'missing':len(missing),'errors':errors[:30],
             'all_pasted_site_roi_pct':round(site_total/(len(rows)*COST)*100,4),
             'source':'KEIBA-BACKFILL pinned Git SHA, manifest success race packs',
             'comparison':'Final-result popularity 1-5 TRIO 10 tickets x 1000JPY',
             'date_coverage_note':'Most recent historical packs after 2026-09-20 may not yet exist',
             'groups':[g for g in groups if g]}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    for r in results: print('RACE',r['date'],r['venue'],str(r['race_no'])+'R',
                           'SITE',r['site_paid'],'TOP5',r['pop5_paid'],
                           'POPS123',r['official_123_pop'],flush=True)
    print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)
    print('COMPLETED',len(results),'/',len(rows), 'MISSING',len(missing),'ERRORS',len(errors),flush=True)
    if errors:raise SystemExit('DATA_INTEGRITY_ERRORS='+str(len(errors)))

if __name__=='__main__':main()
