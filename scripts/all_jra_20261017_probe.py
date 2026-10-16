#!/usr/bin/env python3
"""One-shot bounded JRA official day/race navigation discovery; never guess CNAME."""
import json,re,sys
from bs4 import BeautifulSoup
sys.path.insert(0,"scripts")
from fuji_onsale_jra_deep_probe_20261016 import fetch,CARD,ODDS
def links(html):
 s=BeautifulSoup(html,"html.parser")
 items=[]
 for a in s.select("a[onclick],a[href]"):
  t=re.sub(r"\\s+","",a.get_text(" ",strip=True))
  onclick=a.get("onclick","")
  href=a.get("href","")
  if not (("accessD" in onclick or "accessO" in onclick or "accessD" in href or "accessO" in href) and (t in ("1","2","3","4","5","6","7","8","9","10","11","12") or "東京" in t or "京都" in t or "新潟" in t or "レース" in t or "出馬" in t or "オッズ" in t)):continue
  items.append({"text":t[:45],"onclick":onclick[:260],"href":href[:250]})
 return items
html,meta=fetch(CARD)
print("OFFICIAL_CARD_NAV "+json.dumps({"sha256":meta["sha256"],"links":links(html)[:120]},ensure_ascii=False))
