#!/usr/bin/env python3
"""Independent fixed-picklist analysis. Read-only public netkeiba results. Python stdlib+bs4+requests."""
import concurrent.futures as cf
import csv, json, re, sys, time
from pathlib import Path
from datetime import datetime
import os
import requests
from bs4 import BeautifulSoup

HOME=Path(__file__).parent
CODES={1:'札幌',2:'函館',3:'福島',4:'新潟',5:'東京',6:'中山',7:'中京',8:'京都',9:'阪神',10:'小倉'}
s=requests.Session()
s.headers.update({'User-Agent':'Mozilla/5.0 (compatible; independent historical horse-race results check)', 'Accept-Language':'ja,en;q=0.8'})
def fetch(url):
    for i in range(4):
        try:
            r=s.get(url,timeout=24)
            if r.status_code == 200 and len(r.content)>800:
                if not r.encoding: r.encoding=r.apparent_encoding
                return BeautifulSoup(r.text,'html.parser')
            if r.status_code in (403,404): raise RuntimeError(f'{r.status_code} on {url}')
            if r.status_code not in (429,502,503,504) and r.status_code<500: raise RuntimeError(f'{r.status_code} on {url}')
            cause=str(r.status_code)
        except requests.RequestException as e: cause=str(e)
        if i<3:time.sleep(2**i)
    raise RuntimeError(f'failed {url}: {cause}')

def clean(x):
    return re.sub(r'\s+','',x.get_text(' ',strip=True)).replace(',','').replace('，','')

def targets():
    z=[]
    for l in (HOME/'targets.txt').read_text().splitlines():
        if not l.strip() or l.startswith('#'):continue
        date,venue,*pairs=l.split()
        for pair in pairs:
            n,p=map(int,pair.split('='))
            z.append({'date':date,'venue':venue,'number':n,'site':p})
    assert len(z)==len({(r['date'],r['venue'],r['number']) for r in z})
    return z

def day(d):
    url='https://db.netkeiba.com/race/list/'+datetime.strptime(d,'%Y/%m/%d').strftime('%Y%m%d')+'/'
    soup=fetch(url)
    out={}
    for a in soup.select('a[href]'):
        m=re.search(r'/race/(20\d{10})/?(?:\?.*)?$',a.get('href',''))
        if not m:continue
        rid=m.group(1);venue=CODES.get(int(rid[4:6]));no=int(rid[-2:])
        if venue:
            key=(venue,no)
            if key in out and out[key]!=rid:raise ValueError(f'duplicate day links {d} {key}')
            out[key]=rid
    if not out:raise ValueError(f'no race links: {url}')
    return out

def race(rid):
    url='https://db.netkeiba.com/race/'+rid+'/'
    soup=fetch(url)
    tab=soup.select_one('table.race_table_01')
    if tab is None:raise ValueError('no race results table '+rid)
    ranks={};chosen=set();pops={};horses=0
    for tr in tab.select('tr'):
        td=tr.find_all('td',recursive=False)
        if len(td)<18:continue
        a=[clean(x) for x in td]
        if not(a[0].isdigit() and a[2].isdigit()):continue
        horses+=1
        no=int(a[2])
        if a[17].isdigit():
            p=int(a[17]);pops[no]=p
            if p<=5:chosen.add(no)
        if a[0] in ('1','2','3'):ranks[a[0]]=no
    if len(ranks)!=3 or len(chosen)!=5:raise ValueError(f'bad horses {rid} n={horses}, finish={ranks}, top5={chosen}')
    combos=[]
    for tab in soup.select('table.pay_table_01'):
        for tr in tab.select('tr'):
            th=tr.find_all('th',recursive=False)
            td=tr.find_all('td',recursive=False)
            if not th or '三連複' not in clean(th[0]):continue
            if len(td)<2:continue
            bets=re.findall(r'\d{1,2}[-－]\d{1,2}[-－]\d{1,2}',clean(td[0]))
            pays=re.findall(r'\d+',clean(td[1]))
            if len(bets)!=len(pays):raise ValueError(f'bad payout shape {rid} bets={bets} pays={pays}')
            for b,p in zip(bets,pays):
                combos.append((tuple(sorted(map(int,re.findall(r'\d+',b)))),int(p)))
    if not combos:raise ValueError(f'no trio payout {rid}')
    finish=tuple(sorted(ranks.values()))
    if finish not in [b for b,p in combos]:raise ValueError(f'trio mismatch {rid} finish={finish} payouts={combos}')
    pay=sum(y*10 for b,y in combos if set(b)<=chosen)
    return {'rid':rid,'finish':'-'.join(map(str,finish)),
            'finish_pop':'-'.join(str(pops.get(n,'?')) for n in finish),
            'chosen':'-'.join(map(str,sorted(chosen))),
            'trio100':next(y for b,y in combos if b==finish),
            'popular5':pay,'pop_hit':int(pay>0),'url':url}

def main():
    if os.getenv('PROBE_ONLY'):
        q=fetch('https://db.netkeiba.com/race/202605040305/')
        print('PROBE title',q.title.get_text(' ',strip=True) if q.title else 'no title')
        print('PROBE tables',[(t.get('class'),len(t.select('tr')),len(t.find_all('td'))) for t in q.find_all('table')][:10])
        for t in q.select('table.race_table_01'):
            print('PROBE head',[(len(tr.find_all('td')),len(tr.find_all('th')),tr.get_text(' ',strip=True)[:200]) for tr in t.select('tr')[:5]])
        print('PROBE pay',[(t.get('class'),t.get_text(' ',strip=True)[:300]) for t in q.find_all('table') if '三連複' in t.get_text(' ',strip=True)][:3]);return
    inp=targets();dmap={};fails=[]
    print('REQUESTED',len(inp),'DISTINCT_DAYS',len(set(r['date'] for r in inp)),flush=True)
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futures={ex.submit(day,d):d for d in {r['date'] for r in inp}}
        for future in cf.as_completed(futures):
            d=futures[future]
            try:dmap[d]=future.result()
            except Exception as e:fails.append(('day',d,str(e)))
    work=[]
    for r in inp:
        rid=dmap.get(r['date'],{}).get((r['venue'],r['number']))
        if not rid:fails.append(('missing race',r))
        else:work.append((r,rid))
    print('RESOLVED',len(work),'FAILS',len(fails),flush=True)
    out=[]
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futures={ex.submit(race,rid):r for r,rid in work}
        for future in cf.as_completed(futures):
            r=futures[future]
            try:out.append(dict(r,**future.result()))
            except Exception as e:fails.append(('race',r,str(e)))
    out.sort(key=lambda r:(r['date'],r['venue'],r['number']))
    if out:
        with (HOME/'result.csv').open('w') as f:
            w=csv.DictWriter(f,out[0].keys());w.writeheader();w.writerows(out)
    (HOME/'errors.json').write_text(json.dumps(fails,ensure_ascii=False,indent=2))
    def report(label,a):
        if not a:return
        site=sum(x['site'] for x in a);pop=sum(x['popular5'] for x in a);n=len(a)
        print('SUMMARY',label,'N',n,'INVEST',n*10000,'SITE_PAYOUT',site,'SITE_ROI%',round(site/n/100,2),
              'SITE_HITS',sum(x['site']>0 for x in a),'POP_PAYOUT',pop,'POP_ROI%',round(pop/n/100,2),
              'POP_HITS',sum(x['pop_hit'] for x in a),flush=True)
    report('ALL',out)
    for y in ('2024','2025','2026'):report(y,[x for x in out if x['date'].startswith(y)])
    print('BOTH_HIT',sum(x['site']>0 and x['pop_hit'] for x in out),
          'POPULAR_ONLY',sum(x['site']==0 and x['pop_hit'] for x in out),flush=True)
    for v in out:
        print('ROW',v['date'],v['venue'],str(v['number'])+'R','SITE',v['site'],'POP',v['popular5'],
              'FINISH_POP',v['finish_pop'],'RACE_ID',v['rid'],flush=True)
    print('FAILED',len(fails),json.dumps(fails[:25],ensure_ascii=False),flush=True)
    print('COMPLETED',len(out),'/',len(inp),flush=True)
    if fails:sys.exit(2)

if __name__=='__main__':main()
