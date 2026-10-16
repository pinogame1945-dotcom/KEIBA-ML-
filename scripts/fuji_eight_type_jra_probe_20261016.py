#!/usr/bin/env python3
"""JRA eight wager-type pages using only CNAMEs taken from LIVE official Fuji.
Single refresh snapshot audit. The result does not claim all market combinations
are complete unless an explicit numeric/combination coverage check can prove it.
"""
import re,json,time
from datetime import datetime,timezone
from bs4 import BeautifulSoup
from fuji_onsale_jra_deep_probe_20261016 import fetch,CARD,ODDS
bet_labels=["単勝・複勝","枠連","馬連","ワイド","馬単","3連複","3連単"]
def text(s):return re.sub(r"\s+"," ",s or "").strip()
def links(soup):
 out={}
 for a in soup.select("a[onclick]"):
  label=text(a.get_text(" ",strip=True))
  if label not in bet_labels:continue
  m=re.search(r"doAction\(\s*['\"](/JRADB/accessO\.html)['\"]\s*,\s*['\"]([^'\"]+)['\"]",a.get("onclick",""))
  if m and "202604061120261017" in m.group(2):
   out.setdefault(label,set()).add(m.group(2))
 return out
def race_reference(html):
 soup=BeautifulSoup(html,"html.parser")
 heads=" ".join(x.get_text(" ",strip=True) for x in soup.select("h1,h2,h3,h4"))
 good=("富士ステークス" in heads and "2026年10月17日" in heads and "4回東京6日" in heads and "11レース" in heads)
 if not good:raise ValueError("OFFICIAL_RACE_HEADER_MISMATCH:"+heads[:300])
 txt=text(soup.get_text(" ",strip=True))
 stamp=re.findall(r"(\d{1,2}時\d{2}分現在オッズ)",txt)
 return soup,stamp[-1] if stamp else None,heads[:330]
def main():
 html,_=fetch(CARD)
 soup=BeautifulSoup(html,"html.parser")
 race_options=[]
 for a in soup.select("a[onclick]"):
  m=re.search(r"doAction\(\s*['\"](/JRADB/accessO\.html)['\"]\s*,\s*['\"]([^'\"]+)['\"]",a.get("onclick",""))
  if m and "202604061120261017" in m.group(2):race_options.append(m.group(2))
 if len(set(race_options))!=1:raise ValueError("FUJI_ODDS_LINK_NOT_UNIQUE")
 initial,info=fetch(ODDS,race_options[0])
 race_reference(initial)
 all_links=links(BeautifulSoup(initial,"html.parser"))
 if set(all_links)!=set(bet_labels) or any(len(v)!=1 for v in all_links.values()):
  raise ValueError("OFFICIAL_EIGHT_TYPE_ODDS_LINKS_INCOMPLETE:"+str(all_links))
 records=[]
 for name in bet_labels:
  cname=next(iter(all_links[name]))
  if name=="単勝・複勝":html,meta=initial,info
  else:
   time.sleep(.3)
   html,meta=fetch(ODDS,cname)
  page,updated,head=race_reference(html)
  page_tables=[]
  for i,t in enumerate(page.select("table")):
   sample=[]
   for tr in t.select("tr")[:4]:
    sample.append([text(z.get_text(" ",strip=True))[:110] for z in tr.find_all(["th","td"],recursive=False)][:12])
   page_tables.append({"index":i,"id":t.get("id"),"class":t.get("class"),
      "rows":len(t.select("tr")),"sample":sample,
      "snippet":text(t.get_text(" ",strip=True))[:500]})
  page_text=text(page.get_text(" ",strip=True))
  panel_links=[]
  for a in page.select("a[onclick]"):
   l=text(a.get_text(" ",strip=True))
   m=re.search(r"doAction\(\s*['\"](/JRADB/accessO\.html[^'\"]*)['\"]\s*,\s*['\"]([^'\"]+)['\"]",a.get("onclick",""))
   if m and len(panel_links)<15 and l and (l in ("1着","2着","3着","全組み合わせ","全て","1番","2番","3番","次へ") or "オッズ" in l):
    panel_links.append({"label":l[:65],"cname":m.group(2)})
  records.append({"type":name,"cname":cname,"updated_label":updated,"fetch_http_date":meta["http_date"],
   "sha256":meta["sha256"],"bytes":meta["bytes"],"title":head,"table_count":len(page_tables),
   "tables":page_tables[:6],"navigation":panel_links,
   "pagination_detected":any(z in page_text for z in ("次のページ","1着馬を選択","1頭軸","頭目"))})
 print("JRA_EIGHT_TYPE_ONSALE "+json.dumps({"race_id":"202605040611","checked_utc":datetime.now(timezone.utc).isoformat(),
 "all_types_fetched":len(records),"recording_policy":"INSPECT_ONLY_NEVER_TREAT_PARTIAL_PAGE_AS_ALL_COMBINATIONS",
 "markets":records},ensure_ascii=False,separators=(",",":")))
if __name__=="__main__":main()
