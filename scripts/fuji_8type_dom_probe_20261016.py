#!/usr/bin/env python3
"""Diagnose each of 7 *actual* JRA odds pages, no guessed URLs/cname, no writes.
2026 Fuji Stakes Tokyo 11R. The WIN/PLACE page has already been verified.
"""
import json,re
from bs4 import BeautifulSoup
from datetime import datetime,timezone
from fuji_onsale_jra_deep_probe_20261016 import fetch,CARD,ODDS

LABELS={"単勝・複勝":"WIN_PLACE","枠連":"BRACKET_QUINELLA","馬連":"QUINELLA",
        "ワイド":"WIDE","馬単":"EXACTA","3連複":"TRIO","3連単":"TRIFECTA"}
PATTERN=re.compile(r"doAction\(\s*['\"](/JRADB/accessO\.html)['\"]\s*,\s*['\"]([^'\"]+)['\"]")
def norm(s):return re.sub(r"\s+"," ",s or "").strip()
def actions(soup):
 result=[]
 for x in soup.select("a[onclick],button[onclick]"):
  m=PATTERN.search(x.get("onclick",""))
  if m:
   label=norm(x.get_text(" ",strip=True))
   result.append({"label":label[:50],"code":m.group(2),"html":str(x)[:300]})
 return result
def main():
 card,_=fetch(CARD)
 cardlinks=actions(BeautifulSoup(card,"html.parser"))
 specific=sorted({a["code"] for a in cardlinks if "202604061120261017" in a["code"]})
 if len(specific)!=1:raise ValueError("RACE_SPECIFIC_ODDS_LINK_NOT_UNIQUE")
 base,_=fetch(ODDS,specific[0])
 base_soup=BeautifulSoup(base,"html.parser")
 navigation=actions(base_soup)
 picks={}
 for label,typ in LABELS.items():
  matches={x["code"] for x in navigation if x["label"]==label}
  if label=="単勝・複勝":matches={specific[0]}
  if len(matches)!=1:raise ValueError(f"NO_UNIQUE_OFFICIAL_{typ}_LINK:{matches}")
  picks[typ]=matches.pop()
 out={"checked_utc":datetime.now(timezone.utc).isoformat(),"race_id":"202605040611","pages":{}}
 for typ,code in picks.items():
  html,meta=fetch(ODDS,code)
  soup=BeautifulSoup(html,"html.parser")
  headers=[norm(x.get_text(" ",strip=True)) for x in soup.select("h1,h2,h3,h4")]
  txt=norm(soup.get_text(" ",strip=True))
  if "富士" not in " ".join(headers) or "2026年10月17日" not in " ".join(headers) or "4回東京6日" not in " ".join(headers):
   raise ValueError("WRONG_RACE_PAGE:"+typ)
  ts=re.search(r"(\d{1,2})時(\d{2})分現在",txt)
  tables=[]
  for i,t in enumerate(soup.select("table")):
   rows=[]
   for tr in list(t.select("tr"))[:7]:
    cells=tr.find_all(["th","td"],recursive=False)
    rows.append({"values":[norm(c.get_text(" ",strip=True)) for c in cells][:14],
                 "cell_classes":[c.get("class") for c in cells][:14],
                 "row_classes":tr.get("class"),"row_html":str(tr)[:900]})
   tables.append({"index":i,"class":t.get("class"),"id":t.get("id"),"rows":len(t.select("tr")),
    "row_sample":rows,"text_end":norm(t.get_text(" ",strip=True))[-210:]})
  nav=actions(soup)
  out["pages"][typ]={"cname":code,"http":meta,"title":soup.title.get_text(" ",strip=True) if soup.title else "",
    "page_time":ts.group(0) if ts else None,"tables":tables[:8],"table_count":len(tables),
    "other_actions":[{"label":x["label"],"code":x["code"]} for x in nav if x["code"] not in set(picks.values())][:60],
    "text_tail":txt[-490:],
    "possible_pagination":len([x for x in nav if any(z in x["label"] for z in ("次","前","表示","全","100","人気順","選択","順"))]),
    "headings":headers[:8]}
 print("JRA_8TYPE_DOM_DIAG "+json.dumps(out,ensure_ascii=False,separators=(",",":")))
if __name__=="__main__":main()
