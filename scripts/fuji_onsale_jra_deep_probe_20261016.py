#!/usr/bin/env python3
"""At-sale real source probe: JRA official POST odds navigation.
No fabricated URLs/odds, no purchases, no model execution.
"""
import json,re,hashlib
from urllib.request import Request,urlopen
from urllib.parse import urlparse,urlencode
from datetime import datetime,timezone
from bs4 import BeautifulSoup
CARD="https://www.jra.go.jp/JRADB/accessD.html?CNAME=pw01dde0105202604061120261017%2F22"
ODDS="https://www.jra.go.jp/JRADB/accessO.html"
def fetch(url,cname=None):
 kwargs={}
 if cname is not None:
  kwargs={"data":urlencode({"cname":cname}).encode("ascii"),"method":"POST"}
 headers={"User-Agent":"Mozilla/5.0","Accept-Language":"ja","Referer":CARD}
 if cname:
  headers.update({"Content-Type":"application/x-www-form-urlencoded","Origin":"https://www.jra.go.jp"})
 with urlopen(Request(url,headers=headers,**kwargs),timeout=22) as r:
  assert urlparse(r.geturl()).hostname=="www.jra.go.jp"
  d=r.read(5000000)
  assert len(d)<5000000
  h=r.headers
  last=r.geturl()
 enc=(re.search(r"charset=([\w-]+)",h.get("Content-Type",""),re.I) or [])
 codes=([enc.group(1)] if enc else [])+["utf-8","cp932","euc_jp"]
 content=None
 for cd in dict.fromkeys(codes):
  try:
   v=d.decode(cd)
   if "オッズ" in v:content=v;break
  except (UnicodeError,LookupError):continue
 if not content:raise ValueError("UNREADABLE_JRA_HTML")
 return content,{"url":last,"sha256":hashlib.sha256(d).hexdigest(),"bytes":len(d),"http_date":h.get("Date")}
def inspect(html):
 soup=BeautifulSoup(html,"html.parser")
 txt=re.sub(r"\s+"," ",soup.get_text(" ",strip=True))
 forms=[]
 for f in soup.select("form"):
  forms.append({"method":f.get("method"),"action":f.get("action"),
    "controls":[{"tag":x.name,"name":x.get("name"),"id":x.get("id"),"type":x.get("type"),
      "value":str(x.get("value",""))[:100],
      "options":[{"value":o.get("value"),"text":o.get_text(" ",strip=True)[:80]} for o in x.select("option")][:30]} for x in f.select("input,select,button")][:36]})
 anchors=[]
 for a in soup.select("a[href],a[onclick],button[onclick],input[onclick],img[onclick]"):
  body=a.get_text(" ",strip=True) if a.name not in ("img","input") else a.get("alt","")
  attr={k:a.get(k) for k in ("href","onclick","type","value","id","data-race","data-day")}
  if any(v in str(body)+str(attr) for v in ("オッズ","10月17","東京","pw15","開催","race","accessO","2026","doAction","10/17")):
   anchors.append({"tag":a.name,"text":body[:90],"attrs":{k:str(v)[:240] for k,v in attr.items() if v is not None}})
 scripts=[]
 for x in soup.find_all("script"):
  text=x.get_text() or ""
  for line in text.splitlines():
   if any(v in line for v in ("function doAction","function submit","pw15oli","accessO.html","cname","raceList")):
    scripts.append(line[:450])
 return {"title":soup.title.get_text(" ",strip=True) if soup.title else None,
  "headings":[x.get_text(" ",strip=True) for x in soup.select("h1,h2,h3,h4")[:20]],
  "forms":forms[:7],"anchor_count":len(anchors),"anchors":anchors[:95],
  "script_hits":scripts[:22],"page_start":txt[:650],
  "contains_oct17":("10月17日" in txt or "10/17" in txt),"contains_tokyo":"東京" in txt,
  "error013":("エラー013" in txt or "パラメータエラー" in txt),
  "text_mentions_oct17": [txt[max(0,m.start()-60):m.end()+100] for m in list(re.finditer("10月17日|10/17|10月18日|10/18",txt))[:15]]}
def main():
 card,meta=fetch(CARD)
 so=BeautifulSoup(card,"html.parser")
 hits=[]
 for a in so.select("a[onclick]"):
  if re.sub(r"\s+","",a.get_text(" ",strip=True))=="オッズ":
   m=re.search(r"doAction\(\s*['\"](/JRADB/accessO\.html)['\"]\s*,\s*['\"]([^'\"]+)['\"]",a["onclick"])
   if m:hits.append(m.group(2))
 specific=list(set(x for x in hits if "202604061120261017" in x))
 if len(specific)!=1:
  print("JRA_CARD_ACTION_DIAG "+json.dumps({"hits":hits,"anchors":[{"text":a.get_text(" ",strip=True)[:50],"onclick":a.get("onclick","")[:140]} for a in so.select("a[onclick]") if "accessO.html" in a.get("onclick","")][:20],"card_title":so.title.get_text(" ",strip=True) if so.title else None},ensure_ascii=False))
  raise ValueError("RACE_SPECIFIC_OFFICIAL_ODDS_LINK_NOT_UNIQUE")
 page,info=fetch(ODDS,specific[0])
 report={"checked_utc":datetime.now(timezone.utc).isoformat(),"race_id":"202605040611",
  "card":meta,"odds_index":info,"observed_cname":specific[0],"global_links":sorted(set(hits)),"observations":inspect(page)}
 print("JRA_ONSALE_DIAG "+json.dumps(report,ensure_ascii=False,separators=(",",":")))
if __name__=="__main__":main()
