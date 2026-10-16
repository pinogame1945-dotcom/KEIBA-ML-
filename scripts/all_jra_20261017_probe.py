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

# Discover all three venue race-selection pages by following published JRA links.
from urllib.parse import urljoin
from fuji_onsale_jra_deep_probe_20261016 import CARD as ROOT_CARD
root=BeautifulSoup(html,"html.parser")
for place in ("4回東京6日","4回京都6日","4回新潟1日"):
 anchor=next((a for a in root.select("a[href]") if re.sub(r"\\s+","",a.get_text(" ",strip=True))==place),None)
 if anchor is None: raise ValueError("MEETING_LINK_MISSING:"+place)
 meeting,mm=fetch(urljoin("https://www.jra.go.jp",anchor["href"]))
 ms=BeautifulSoup(meeting,"html.parser")
 picks=[a.get("onclick","") for a in ms.select("a[onclick]") if "pw01drl" in a.get("onclick","")]
 if len(set(picks))!=1: raise ValueError("RACE_SELECTION_LINK_NOT_UNIQUE:"+place+":"+str(picks))
 m=re.search(r"pw01drl[^'\\\"]+",picks[0])
 if not m:raise ValueError("SELECTION_CNAME_MISSING")
 page,pm=fetch("https://www.jra.go.jp/JRADB/accessD.html",m.group(0))
 print("VENUE_RACE_SELECTION "+json.dumps({"venue":place,"cname":m.group(0),"page_sha256":pm["sha256"],"links":links(page)[:65]},ensure_ascii=False))
