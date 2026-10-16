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
