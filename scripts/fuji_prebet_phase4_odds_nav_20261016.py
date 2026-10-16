#!/usr/bin/env python3
"""P4 diagnostic, no odds assumed. Follow ONLY the observed official JRA odds link.
Screenshots/API calls are not faked, prices may not be on-sale yet.
"""
import json,re,hashlib,sys
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urljoin,urlparse,quote,parse_qs
from bs4 import BeautifulSoup
RID="202605040611"
URL="https://www.jra.go.jp/JRADB/accessD.html?CNAME=pw01dde0105202604061120261017%2F22"
def read(url):
 if urlparse(url).hostname!="www.jra.go.jp" or urlparse(url).scheme!="https":raise ValueError("JRA_HOST_ONLY")
 with urlopen(Request(url,headers={"User-Agent":"Mozilla/5.0 (compatible; FujiOddsAudit/1.0)","Accept-Language":"ja"}),timeout=25) as r:
  if urlparse(r.geturl()).hostname!="www.jra.go.jp":raise ValueError("REDIRECT")
  data=r.read(7000000);typ=r.headers.get("Content-Type","");last_modified=r.headers.get("Last-Modified");server_date=r.headers.get("Date")
 if len(data)>=7000000:raise ValueError("SIZE_LIMIT")
 txt=None
 enc=re.search(r"charset\s*=\s*([\w-]+)",typ,re.I)
 for cs in dict.fromkeys([enc.group(1) if enc else None,"utf-8","cp932","euc_jp","shift_jis"]):
  if not cs:continue
  try:
   t=data.decode(cs)
   if "オッズ" in t:txt=t;break
  except (UnicodeError,LookupError):pass
 if txt is None:raise ValueError("DECODE_INCOMPLETE")
 return txt,{"source":url,"http_date":server_date,"last_modified":last_modified,"sha256":hashlib.sha256(data).hexdigest()}
def find_links(html):
 soup=BeautifulSoup(html,"html.parser")
 options=[]
 for a in soup.select("a[onclick]"):
  label=re.sub(r"\s+","",a.get_text(" ",strip=True))
  onclick=a.get("onclick","")
  if "オッズ" not in label and "accessO.html" not in onclick:continue
  m=re.search(r"doAction\(\s*['\"]([^'\"]*accessO\.html)['\"]\s*,\s*['\"]([^'\"]+)['\"]",onclick)
  options.append({"label":label[:60],"onclick":onclick[:240],"cname":m.group(2) if m else None,"path":m.group(1) if m else None})
 return options
def odds_url(o):
 path=o.get("path");cname=o.get("cname")
 if not path or not cname or not path.endswith("accessO.html") or not re.fullmatch(r"[A-Za-z0-9/]+",cname):return None
 absolute=urljoin("https://www.jra.go.jp/JRADB/",path)
 if urlparse(absolute).hostname!="www.jra.go.jp" or urlparse(absolute).path!="/JRADB/accessO.html":return None
 return absolute+"?CNAME="+quote(cname,safe="/")
def main():
 html,meta=read(URL)
 txt=re.sub(r"\s+","",BeautifulSoup(html,"html.parser").get_text(" ",strip=True))
 if "第29回富士" not in txt or "2026年10月17日" not in txt or "4回東京6日" not in txt:raise ValueError("WRONG_JRA_CARD")
 menus=find_links(html)
 out={"contract":"FUJI_PREBET_P4_JRA_ODDS_NAVIGATION_V1", "race_id":RID,"official_card":meta,"odds_links":menus,
      "odds_page":None,"on_sale_confirmed":False,"has_prices":False}
 valid=[odds_url(o) for o in menus if "オッズ" in o["label"]]
 valid=[v for v in valid if v]
 if len(set(valid))==1:
  target=valid[0]
  try:
   data,md=read(target)
   soup=BeautifulSoup(data,"html.parser")
   norm=re.sub(r"\s+","",soup.get_text(" ",strip=True))
   heads=[re.sub(r"\s+","",x.get_text(" ",strip=True))[:70] for x in soup.select("h1,h2,h3,h4,table th")[:50]]
   oddsMenu=find_links(data)
   anchors=[{"text":re.sub(r"\s+","",a.get_text(" ",strip=True))[:40],"href":a.get("href"),"onclick":a.get("onclick","")[:190]} for a in soup.select("a[onclick]") if any(k in (a.get_text(" ",strip=True)+a.get("onclick","")) for k in ("単勝","複勝","馬連","馬単","ワイド","三連","3連","枠連"))][:30]
   out["odds_page"]={"status":"OFFICIAL_ODDS_NAV_REACHED","url":target,"metadata":md,
     "race_in_content":("富士" in norm and ("2026年10月17日" in norm or "10月17日" in norm)),
     "body_excerpt":norm[-950:],
     "headings":heads,"bet_type_nav":anchors,"table_count":len(soup.select("table")),
     "has_no_price_placeholders":("---" in norm or "発売前" in norm or "発売しておりません" in norm)}
   # Do NOT assert confirmed on-sale until verified typed horse-number to odds pairs.
  except Exception as e:out["odds_page"]={"status":"ODDS_LINK_FETCH_BLOCKED","reason":type(e).__name__+":"+str(e),"url":target}
 else:out["odds_page"]={"status":"ODDS_NAV_NOT_DETERMINED","valid_link_count":len(set(valid))}
 print("PHASE4_NAVIGATION "+json.dumps(out,ensure_ascii=False))
 return 0 if out["odds_page"]["status"]=="OFFICIAL_ODDS_NAV_REACHED" else 2
if __name__=="__main__":sys.exit(main())
