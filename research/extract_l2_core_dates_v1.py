#!/usr/bin/env python3
import argparse,csv,gzip,json
from pathlib import Path

EXPECTED_PER_YEAR=3456

def open_text(path):
    return gzip.open(path,'rt',encoding='utf-8') if str(path).endswith('.gz') else open(path,'rt',encoding='utf-8')

def parse_args():
    p=argparse.ArgumentParser(description='Extract race dates for L2 CORE V1 from one persisted L1 expert score table per year.')
    p.add_argument('--score-year',action='append',required=True,help='YEAR:PATH')
    p.add_argument('--dates-out',required=True)
    p.add_argument('--race-dates-out',required=True)
    return p.parse_args()

def main():
    a=parse_args(); seen={}; rows=[]
    for spec in a.score_year:
        ys,path=spec.split(':',1); year=int(ys)
        if year==2026: raise SystemExit('2026 sealed')
        count=0
        with open_text(path) as f:
            for line in f:
                if not line.strip(): continue
                r=json.loads(line)
                if r.get('contract')!='L1_TO_L2_OUTPUT_CONTRACT_V1':
                    raise SystemExit(f'bad score contract year={year}')
                rid=str(r.get('race_id') or '')
                date=str(r.get('race_date') or '')[:10]
                if not rid or len(date)!=10:
                    raise SystemExit(f'missing race/date year={year} rid={rid}')
                if not date.startswith(str(year)+'-'):
                    raise SystemExit(f'year/date drift year={year} race={rid} date={date}')
                old=seen.get(rid)
                if old and old!=(year,date):
                    raise SystemExit(f'duplicate race drift race={rid} old={old} new={(year,date)}')
                if not old:
                    seen[rid]=(year,date); rows.append((year,rid,date)); count+=1
        if count!=EXPECTED_PER_YEAR:
            raise SystemExit(f'race coverage regression year={year}: {count} != {EXPECTED_PER_YEAR}')
    dates=sorted({d for _,_,d in rows})
    Path(a.dates_out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.dates_out).write_text('\n'.join(dates)+'\n',encoding='utf-8')
    with open(a.race_dates_out,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['year','race_id','race_date']); w.writerows(sorted(rows))
    print('L2_CORE_DATES_READY')
    print(json.dumps({'races':len(rows),'dates':len(dates),'years':sorted({y for y,_,_ in rows})},separators=(',',':')))

if __name__=='__main__': main()
