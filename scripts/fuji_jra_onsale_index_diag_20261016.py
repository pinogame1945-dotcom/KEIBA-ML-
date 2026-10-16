#!/usr/bin/env python3
"""Read-only inspection of official JRA odds landing after 2026-10-16 18:30 JST.
No fabricated CNAME, no final odds, no purchase, no storage artifact.
"""
import json,re,sys
from bs4 import BeautifulSoup
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urljoin
import importlib.util

spec=importlib.util.spec_from_file_location("original_post",Path("scripts/fuji_jra_post_protocol_20261016.py"))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
card,cmeta=m.fetch(m.CARD_URL)
s=BeautifulSoup(card,"html.parser")
assert all(x in re.sub(r"\s+","",s.get_text(" ",strip=True)) for x in ("富士ステークス","2026年10月17日","4回東京6日"))
links=m.links(s)
assert len({x["cname"] for x in links if x["label"]=="オッズ"})==1 and next(x["cname"] for x in links if x["label"]=="オッズ")==m.START_CNAME
html,meta=m.fetch(m.POST_URL,m.START_CNAME)
soup=BeautifulSoup(html,"html.parser")
def clean(x):return re.sub(r"\s+"," ",x or "").strip()
text=clean(soup.get_text(" ",strip=True))
print("DIAG_SNAPSHOT "+json.dumps({"utc":datetime.now(timezone.utc).isoformat(),"race_id":"202605040611","card_sha":cmeta["sha256"],"index_sha":meta["sha256"],"index_title":clean(soup.title.get_text(" ",strip=True)) if soup.title else None,"has_parameter_error":"パラメータエラー" in text,"visible_text_last_4500":text[-4500:]},ensure_ascii=False))
visible=[]
for tag in soup.find_all(["a","button","input","select","form","option"]):
  attrs={k:str(v)[:450] for k,v in tag.attrs.items() if k in ("id","class","type","name","value","href","action","method","onclick","onchange","data-action","data-url","data-cname")}
  label=clean(tag.get_text(" ",strip=True))[:120]
  if tag.name=="input" and tag.get("type") in ("hidden","password"):continue
  if label or attrs.get("onclick") or attrs.get("href") or tag.name=="form":
    visible.append({"tag":tag.name,"text":label,"attrs":attrs})
print("DIAG_RELEVANT_CONTROLS "+json.dumps(visible[-105:],ensure_ascii=False))
js=[]
for sc in soup.select("script"):
 t=sc.get_text(" ",strip=True)
 if not t:continue
 if any(x in t for x in ("accessO.html","doAction","CNAME","odds","oli00")):
  js.append({"source":sc.get("src"),"snippet":t[:4200]})
print("DIAG_INLINE_JS "+json.dumps(js[:8],ensure_ascii=False))
