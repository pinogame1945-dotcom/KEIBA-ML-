#!/usr/bin/env python3
"""Diagnostics for 2026-10-17 Tokyo R5 official runner row. Never mutate source data."""
import json,sys,re
from pathlib import Path
from bs4 import BeautifulSoup
sys.path.insert(0,'scripts')
from all_jra_20261017_collect import src_fetch,clean
a=json.loads(Path('audit-results/live-20261017/202605040605.json').read_text('utf-8'))
html,meta=src_fetch(a['source_discovery']['jra_card_url'])
soup=BeautifulSoup(html,'html.parser')
results=[]
for tr in soup.select('tr'):
 if 'ジェットシェヴロン' not in clean(tr.get_text(' ',strip=True)):continue
 cells=tr.find_all(['th','td'],recursive=False)
 results.append({'cells':[{'text':clean(c.get_text(' ',strip=True))[:120],'html':str(c)[:450]} for c in cells[:5]],'row':str(tr)[:1500]})
print('TOKYO_5_OFFICIAL_ROW_DIAG '+json.dumps({'records':results,'metadata':meta},ensure_ascii=False))
if len(results)!=1:raise ValueError('EXPECTED_ONE_HORSE_ROW:'+str(len(results)))


# Inspect only public official Tokyo R5 odds layout, without claiming a numeric quote.
from all_jra_20261017_collect import official_fetch,JRA_O,action_of,by_text
odds_s,odds_meta=official_fetch(JRA_O,a['source_discovery']['jra_odds_cname'])
odds_dom=BeautifulSoup(odds_s,'html.parser')
print('TOKYO_5_ODDS_WIN_PLACE_DIAG '+json.dumps({
 'tables':len(odds_dom.select('table.tanpuku')),
 'rows':[{'num':clean(r.select_one('td.num').get_text(' ',strip=True)) if r.select_one('td.num') else None,
          'horse':clean(r.select_one('td.horse').get_text(' ',strip=True))[:35] if r.select_one('td.horse') else None,
          'win':clean(r.select_one('td.odds_tan').get_text(' ',strip=True))[:50] if r.select_one('td.odds_tan') else None}
  for r in odds_dom.select('table.tanpuku tr') if r.select_one('td.num')],
 'links':[{'bet':label,'codes':[action_of(x,'/JRADB/accessO.html') for x in by_text(odds_dom,label)]}
  for label in ('枠連','馬連','ワイド','馬単','3連複','3連単')]
},ensure_ascii=False))
for label,cls in [('馬連','umaren'),('3連単','tan3')]:
 item=by_text(odds_dom,label)
 target=next((action_of(anchor,'/JRADB/accessO.html') for anchor in item if action_of(anchor,'/JRADB/accessO.html')),None)
 if not target:continue
 full,meta=official_fetch(JRA_O,target)
 s=BeautifulSoup(full,'html.parser')
 tabs=s.select('table.'+cls)
 print('TOKYO_5_ODDS_MATRIX_DIAG '+json.dumps({
  'bet':label,'tables':len(tabs),
  'sizes':[(len(t.select('tr'))) for t in tabs[:15]],
  'head_rows':[
    [{'cells':[clean(c.get_text(' ',strip=True))[:45] for c in row.find_all(['td','th'],recursive=False)]} for row in t.select('tr')[:8]]
    for t in tabs[:3]]
  },ensure_ascii=False))
