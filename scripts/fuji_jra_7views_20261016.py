#!/usr/bin/env python3
"""Read-only bounded real 2026 Fuji JRA betting odds page structure validation."""
import importlib.util,hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path
from bs4 import BeautifulSoup
spec=importlib.util.spec_from_file_location("jra",Path("scripts/fuji_jra_post_protocol_20261016.py"))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root,_=m.fetch(m.CARD_URL)
s=BeautifulSoup(root,"html.parser")
race=[x for x in m.links(s) if "20261017" in x["cname"] and x["label"]=="オッズ"]
if len(set(x["cname"] for x in race))!=1:raise ValueError("CANNOT_RESOLVE_FUJI_ODDS_FROM_LIVE_OFFICIAL_CARD")
w,_=m.fetch(m.POST_URL,race[0]["cname"])
s=BeautifulSoup(w,"html.parser")
nav=[]
for a in s.select("a[onclick]"):
 label=re.sub(r"\s+","",a.get_text(" ",strip=True))
 onclick=a.get("onclick","")
 z=re.search(r"doAction\(\s*['\"](/JRADB/accessO\.html)['\"]\s*,\s*['\"]([^'\"]+)['\"]",onclick)
 if z and label in {"単勝・複勝","枠連","馬連","ワイド","馬単","3連複","3連単"}:
  if "20261017" not in z.group(2) or not re.fullmatch(r"[A-Za-z0-9]+/[A-Fa-f0-9]{2}",z.group(2)):raise ValueError("UNVERIFIED_BET_LINK")
  if label not in {x["label"] for x in nav}:nav.append({"label":label,"cname":z.group(2)})
if len(nav)!=7:raise ValueError("NOT_ALL_SEVEN_JRA_ODDS_VIEWS_FOUND:"+str(nav))
for r in nav:
 txt,meta=m.fetch(m.POST_URL,r["cname"])
 soup=BeautifulSoup(txt,"html.parser")
 full=re.sub(r"\s+"," ",soup.get_text(" ",strip=True))
 if not ("2026年10月17日" in full and "4回東京6日" in full and "11レース" in full and "富士ステークス" in full):
  raise ValueError("WRONG_RACE_PAGE:"+r["label"])
 if "パラメータエラー" in full:raise ValueError("JRA_ERROR013_"+r["label"])
 tables=[]
 for t in soup.select("table"):
  trs=t.select("tr")
  compact=[]
  for tr in trs[:5]:
   compact.append([re.sub(r"\s+"," ",c.get_text(" ",strip=True))[:130] for c in tr.find_all(["td","th"],recursive=False)[:15]])
  tables.append({"class":t.get("class"),"id":t.get("id"),"rows":len(trs),
    "head":compact,"tail":[[re.sub(r"\s+"," ",c.get_text(" ",strip=True))[:100] for c in tr.find_all(["td","th"],recursive=False)[:15]] for tr in trs[-2:]]})
 maintext=full.split("ここから本文です",1)[-1]
 marker=re.search(r"\d{1,2}時\d{2}分現在オッズ",maintext)
 print("JRA_BET_PAGE "+json.dumps({"bet":r["label"],"live_cname":r["cname"],
  "observed_utc":datetime.now(timezone.utc).isoformat(),"html_sha256":meta["sha256"],
  "published_odds_label":marker.group(0) if marker else None,
  "tables":tables[:8],"table_count":len(tables),"main_excerpt":maintext[:1700],
  "main_tail":maintext[-550:]},ensure_ascii=False))
