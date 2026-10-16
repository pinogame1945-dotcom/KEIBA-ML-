#!/usr/bin/env python3
"""Small read-only check for Friday 18:30 Sunday GI Shuka Sho odds availability.
All POST payloads are obtained from official public HTML, never guessed.
"""
import importlib.util,json,re
from datetime import datetime,timezone
from pathlib import Path
from bs4 import BeautifulSoup
s=importlib.util.spec_from_file_location("jra",Path("scripts/fuji_jra_post_protocol_20261016.py"))
jra=importlib.util.module_from_spec(s);s.loader.exec_module(jra)
h,_=jra.fetch(jra.CARD_URL)
card=BeautifulSoup(h,"html.parser")
entries=[x for x in jra.links(card) if x["label"]=="オッズ" and "20261017" in x["cname"]]
if len({x["cname"] for x in entries})!=1:raise ValueError("FUJI_ODDS_NOT_FOUND")
p,_=jra.fetch(jra.POST_URL,entries[0]["cname"])
page=BeautifulSoup(p,"html.parser")
def clean(s):return re.sub(r"\s+","",s or "")
candidates=[]
for a in page.select("a[onclick]"):
 label=clean(a.get_text(" ",strip=True))
 code=a.get("onclick","")
 m=re.search(r"doAction\(\s*['\"](/JRADB/accessO\.html)['\"]\s*,\s*['\"]([A-Za-z0-9]+/[A-Fa-f0-9]{2})['\"]",code)
 if m and ("4回京都7日" in label or "10月18日" in label or "20261018" in m.group(2)):
  candidates.append({"label":label[:75],"cname":m.group(2)})
print("SHUKA_FRIDAY_LINKS "+json.dumps({"observed_at_utc":datetime.now(timezone.utc).isoformat(),"links":candidates[:15]},ensure_ascii=False))
routes={x["cname"]:x for x in candidates if "京都" in x["label"] or "20261018" in x["cname"]}
for cname,r in list(routes.items())[:2]:
 try:
  html,meta=jra.fetch(jra.POST_URL,cname)
  soap=BeautifulSoup(html,"html.parser")
  text=clean(soap.get_text(" ",strip=True))
  links=jra.links(soap)
  print("SHUKA_FRIDAY_ROUTING "+json.dumps({"cname":cname,"label":r["label"],
    "title":soap.title.get_text(" ",strip=True) if soap.title else None,
    "has_shuka":"秋華賞" in text,"has_kyoto":"京都" in text,
    "tail_excerpt":text[-450:],"odds_links":links[:8]},ensure_ascii=False))
 except Exception as e:print("SHUKA_FRIDAY_BLOCKED "+json.dumps({"cname":cname,"reason":type(e).__name__+":"+str(e)},ensure_ascii=False))
