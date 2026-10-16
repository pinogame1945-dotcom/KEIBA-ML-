#!/usr/bin/env python3
"""Read-only official Fuji odds exact tables and bet-type navigation.
All data sourced from actual observed JRA race-specific POST CNAME.
"""
from bs4 import BeautifulSoup
import json,re
from datetime import datetime,timezone
from fuji_onsale_jra_deep_probe_20261016 import fetch,CARD,ODDS
def clean(s):return re.sub(r"\s+"," ",s or "").strip()
def main():
 card,meta=fetch(CARD)
 sop=BeautifulSoup(card,"html.parser")
 links=[]
 for a in sop.select("a[onclick]"):
  match=re.search(r"doAction\(\s*['\"](/JRADB/accessO\.html)['\"]\s*,\s*['\"]([^'\"]+)['\"]",a["onclick"])
  if match and "202604061120261017" in match.group(2):links.append(match.group(2))
 if len(set(links))!=1:raise ValueError("RACE_SPECIFIC_CNAME_ABSENT")
 html,resp=fetch(ODDS,links[0])
 soup=BeautifulSoup(html,"html.parser")
 heads=[clean(x.get_text(" ",strip=True)) for x in soup.select("h1,h2,h3,h4")]
 if "富士" not in " ".join(heads) or "2026年10月17日" not in " ".join(heads):raise ValueError("WRONG_RACE")
 tables=[]
 for i,table in enumerate(soup.select("table")):
  rows=[]
  for tr in table.select("tr")[:7]:
   cells=tr.find_all(["th","td"],recursive=False)
   rows.append({"text":[clean(c.get_text(" ",strip=True))[:180] for c in cells],
                "html":[str(c)[:400] for c in cells[:2]]})
  tables.append({"index":i,"class":table.get("class"),"id":table.get("id"),
                 "rows":len(table.select("tr")),"excerpt":clean(table.get_text(" ",strip=True))[:350],
                 "sample":rows})
 controls=[]
 for a in soup.select("a[onclick]"):
  label=clean(a.get_text(" ",strip=True))
  handler=a.get("onclick","")
  if "accessO.html" in handler and (len(handler)>80 or "2026" in handler or any(x in label for x in ("馬単","ワイド","枠連","3連","三連","単勝","複勝","馬連"))):
   controls.append({"text":label[:50],"on":handler[:300]})
 paragraph=clean(soup.get_text(" ",strip=True))
 markers={}
 for needle in ("オッズ", "更新", "発表", "票数", "時分", "単勝","複勝","発売","馬番順"):
  matches=list(re.finditer(needle,paragraph))
  markers[needle]=[paragraph[max(0,m.start()-50):m.end()+110] for m in matches[-6:]]
 report={"checked_at_utc":datetime.now(timezone.utc).isoformat(),"target":"202605040611",
  "cname":links[0],"meta":resp,"heads":heads,"table_count":len(tables),"tables":tables[:12],
  "official_odds_handlers":controls[:90],"status_markers":markers,
  "html_structure_links":len(soup.select("a[onclick]"))}
 print("JRA_LIVE_TABLE_PROBE "+json.dumps(report,ensure_ascii=False,separators=(",",":")))
if __name__=="__main__":main()
