#!/usr/bin/env python3
"""Read-only extraction of JRA *live* (NOT final) odds for actual 2026 Fuji S.
Uses ONLY CNAME values present in freshly fetched JRA HTML. Does not guess
URLs/cnames, substitute final odds, select tickets, or place bets.
Saves one runner-local JSON snapshot; no Actions artifact/cache uploads.
"""
from __future__ import annotations
import importlib.util,hashlib,json,re,sys
from datetime import datetime,timezone
from pathlib import Path
from itertools import combinations,permutations
from bs4 import BeautifulSoup

spec=importlib.util.spec_from_file_location("jra_post",Path("scripts/fuji_jra_post_protocol_20261016.py"))
jra=importlib.util.module_from_spec(spec);spec.loader.exec_module(jra)
RACE="202605040611";DAY="2026-10-17"
JRA_TYPES={"単勝・複勝":("WIN","PLACE"),"枠連":("BRACKET_QUINELLA",),"馬連":("QUINELLA",),
 "ワイド":("WIDE",),"馬単":("EXACTA",),"3連複":("TRIO",),"3連単":("TRIFECTA",)}
expected={r["horse_number"]:r for r in json.loads(
 Path("audit-results/fuji-20261017-149-asof-20261016.json").read_text("utf-8"))["runners"]}
if len(expected)!=14 or sorted(expected)!=list(range(1,15)):raise ValueError("WRONG_AUDIT_FIELD_SIZE")
frames={i:sum(r["gate"]==i for r in expected.values()) for i in range(1,9)}
def compact(s):return re.sub(r"\s+"," ",str(s or "")).strip()
def no_space(s):return re.sub(r"\s+","",str(s or ""))
def rowcols(tr):
 return [compact(td.get_text(" ",strip=True)) for td in tr.find_all(["td","th"],recursive=False)]
def quote(v,ranged=False):
 v=compact(v)
 if v in ("票数なし","発売前","---","-"):
  return {"status":"NO_PUBLISHED_PRICE","odds":None,"odds_min":None,"odds_max":None,"raw":v}
 if ranged:
  z=re.fullmatch(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)",v)
  if not z:raise ValueError("ODDS_RANGE_FORMAT_UNVERIFIED:"+v)
  lo,hi=map(float,z.groups())
  if lo<=0 or hi<lo:raise ValueError("ODDS_RANGE_INVALID")
  return {"status":"QUOTED","odds":None,"odds_min":lo,"odds_max":hi,"raw":v}
 if not re.fullmatch(r"\d+(?:\.\d+)?",v):raise ValueError("ODDS_FORMAT_UNVERIFIED:"+v)
 p=float(v)
 if p<=0:raise ValueError("ODDS_NONPOSITIVE")
 return {"status":"QUOTED","odds":p,"odds_min":None,"odds_max":None,"raw":v}
def item(kind,numbers,value,ranged=False):
 return {"bet_type":kind,"combination":list(numbers),**quote(value,ranged)}
def parse_page(kind,html):
 soup=BeautifulSoup(html,"html.parser")
 text=no_space(soup.get_text(" ",strip=True))
 for exact in ("2026年10月17日","4回東京6日","11レース","富士ステークス"):
  if exact not in text:raise ValueError("WRONG_RACE_FOR_"+kind+":"+exact)
 if "パラメータエラー" in text:raise ValueError("JRA_ERROR013")
 minute=re.search(r"(\d{1,2}時\d{2}分現在オッズ)",text)
 if not minute:raise ValueError("OFFICIAL_PUBLISHER_MINUTE_MISSING")
 tables={
  "WIN":soup.select("table.tanpuku"),
  "BRACKET_QUINELLA":soup.select("table.waku"),
  "QUINELLA":soup.select("table.umaren"),
  "WIDE":soup.select("table.wide"),
  "EXACTA":soup.select("table.umatan"),
  "TRIO":soup.select("table.fuku3"),
  "TRIFECTA":soup.select("table.tan3"),
 }
 primary=tables[kind]
 all_rows=[]
 if kind=="WIN":
  if len(primary)!=1:raise ValueError("WIN_TABLE_COUNT")
  trs=primary[0].select("tr")
  if len(trs)!=15:raise ValueError("WIN_HORSE_ROWS")
  for tr in trs[1:]:
   cols=rowcols(tr)
   names=[(i,x["horse_name"]) for i,x in expected.items() if x["horse_name"] in no_space(" ".join(cols))]
   if len(names)!=1:raise ValueError("WIN_ROW_NO_UNIQUE_HORSE:"+str(cols))
   no,name=names[0]
   idx=next((k for k,v in enumerate(cols) if no_space(v)==no_space(name)),None)
   if idx is None or idx+2>=len(cols):raise ValueError("WIN_COLUMNS_NOT_FOUND")
   # number must immediately precede horse name, independent of rowspan gate
   if idx==0 or cols[idx-1]!=str(no):raise ValueError("JRA_HORSE_NUMBER_CONFLICT:"+name)
   all_rows.extend([item("WIN",(no,),cols[idx+1]),item("PLACE",(no,),cols[idx+2],True)])
 elif kind=="BRACKET_QUINELLA":
  if len(primary)!=8:raise ValueError("BRACKET_TABLE_COUNT")
  for fst,t in enumerate(primary,1):
   trs=t.select("tr")
   if len(trs)!=9-fst:raise ValueError("BRACKET_ROW_COUNT")
   for tr,sec in zip(trs,range(fst,9)):
    c=rowcols(tr)
    if len(c)<2 or c[0]!=str(sec):raise ValueError("BRACKET_ROW_ORDER")
    if fst==sec and frames[fst]<2:
     if c[1] not in ("","票数なし"):raise ValueError("IMPOSSIBLE_SAME_FRAME_PUBLISHED")
     continue
    if not c[1]:raise ValueError("BRACKET_EMPTY_PUBLISHABLE")
    all_rows.append(item("BRACKET_QUINELLA",(fst,sec),c[1]))
 elif kind in ("QUINELLA","WIDE"):
  if len(primary)!=13:raise ValueError("PAIR_TABLE_COUNT_"+kind)
  for fst,t in enumerate(primary,1):
   trs=t.select("tr")
   if len(trs)!=14-fst:raise ValueError("PAIR_ROW_COUNT_"+kind)
   for tr,sec in zip(trs,range(fst+1,15)):
    c=rowcols(tr)
    if len(c)<2 or c[0]!=str(sec):raise ValueError("PAIR_ROW_ORDER_"+kind)
    all_rows.append(item(kind,(fst,sec),c[1],kind=="WIDE"))
 elif kind=="EXACTA":
  if len(primary)!=14:raise ValueError("EXACTA_TABLE_COUNT")
  for fst,t in enumerate(primary,1):
   trs=t.select("tr")
   if len(trs)!=14:raise ValueError("EXACTA_ROW_COUNT")
   for tr,sec in zip(trs,range(1,15)):
    c=rowcols(tr)
    if len(c)<2 or c[0]!=str(sec):raise ValueError("EXACTA_ROW_ORDER")
    if fst==sec:
     if c[1]:raise ValueError("EXACTA_DIAGONAL_NOT_BLANK")
     continue
    all_rows.append(item(kind,(fst,sec),c[1]))
 elif kind=="TRIO":
  pairs=list(combinations(range(1,14),2))
  if len(primary)!=len(pairs):raise ValueError("TRIO_TABLE_COUNT")
  for (fst,sec),t in zip(pairs,primary):
   trs=t.select("tr")
   if len(trs)!=14-sec:raise ValueError("TRIO_ROW_COUNT")
   for tr,third in zip(trs,range(sec+1,15)):
    c=rowcols(tr)
    if len(c)<2 or c[0]!=str(third):raise ValueError("TRIO_ROW_ORDER")
    all_rows.append(item(kind,(fst,sec,third),c[1]))
 elif kind=="TRIFECTA":
  pairs=list(permutations(range(1,15),2))
  if len(primary)!=len(pairs):raise ValueError("TRIFECTA_TABLE_COUNT")
  for (fst,sec),t in zip(pairs,primary):
   trs=t.select("tr")
   if len(trs)!=13:raise ValueError("TRIFECTA_ROW_COUNT")
   third_list=[i for i in range(1,15) if i!=fst]
   for tr,third in zip(trs,third_list):
    c=rowcols(tr)
    if len(c)<2 or c[0]!=str(third):raise ValueError("TRIFECTA_ROW_ORDER")
    if third==sec:
     if c[1]:raise ValueError("TRIFECTA_DIAGONAL_NOT_BLANK")
     continue
    all_rows.append(item(kind,(fst,sec,third),c[1]))
 return {"publisher_minute":minute.group(1),"records":all_rows,"page_type":kind}
def main():
 src,_=jra.fetch(jra.CARD_URL)
 card=BeautifulSoup(src,"html.parser")
 candidates=[x for x in jra.links(card) if x["label"]=="オッズ" and "20261017" in x["cname"]]
 cnames={x["cname"] for x in candidates}
 if len(cnames)!=1:raise ValueError("FUJI_LIVE_CNAME_NOT_PRESENT_IN_JRA_CARD")
 win,_=jra.fetch(jra.POST_URL,next(iter(cnames)))
 page=BeautifulSoup(win,"html.parser")
 nav={}
 for a in page.select("a[onclick]"):
  label=no_space(a.get_text(" ",strip=True))
  if label not in JRA_TYPES:continue
  x=re.search(r"doAction\(\s*['\"](/JRADB/accessO\.html)['\"]\s*,\s*['\"]([A-Za-z0-9]+/[A-Fa-f0-9]{2})['\"]",a["onclick"])
  if not x or "20261017" not in x.group(2):continue
  if label in nav and nav[label]!=x.group(2):raise ValueError("AMBIGUOUS_JRA_ODDS_TAB")
  nav[label]=x.group(2)
 if set(nav)!=set(JRA_TYPES):raise ValueError("SEVEN_TABS_NOT_COMPLETE")
 extracted=[];pages=[];mins=set();seen=set()
 for label,types in JRA_TYPES.items():
  h,meta=jra.fetch(jra.POST_URL,nav[label])
  primary=types[0]
  p=parse_page(primary,h)
  minlabel=p["publisher_minute"]
  mins.add(minlabel)
  pages.append({"page_label":label,"JRA_html_sha256":meta["sha256"],
    "JRA_published_minute":minlabel,"observed_at_utc":datetime.now(timezone.utc).isoformat()})
  for itemx in p["records"]:
   key=(itemx["bet_type"],tuple(itemx["combination"]))
   if key in seen:raise ValueError("DUPLICATE_BET_KEY")
   seen.add(key);extracted.append(itemx)
 counts={k:sum(x["bet_type"]==k for x in extracted) for k in (
 "WIN","PLACE","BRACKET_QUINELLA","QUINELLA","WIDE","EXACTA","TRIO","TRIFECTA")}
 expected_counts={"WIN":14,"PLACE":14,"BRACKET_QUINELLA":34,"QUINELLA":91,
  "WIDE":91,"EXACTA":182,"TRIO":364,"TRIFECTA":2184}
 if counts!=expected_counts:raise ValueError("JRA_COMBINATION_COUNT_INCOMPLETE:"+str(counts))
 if len(mins)!=1:raise ValueError("MULTIPLE_JRA_PUBLISHED_MINUTES:"+str(mins))
 quoted={k:sum(x["bet_type"]==k and x["status"]=="QUOTED" for x in extracted) for k in counts}
 output={"contract":"FUJI_JRA_EIGHT_LIVE_ODDS_SNAPSHOT_V1","race_id":RACE,
   "race_date":DAY,"source_kind":"JRA_PRE_RACE_PUBLISHED_ODDS_NOT_FINAL",
   "observed_at_utc":datetime.now(timezone.utc).isoformat(),
   "publisher_minute_label":next(iter(mins)),
   "publisher_minute_not_server_timestamp":True,
   "all_pages_same_published_minute":True,
   "pages":pages,"n_horses":14,"number_to_horse_id":{str(i):r["horse_id"] for i,r in expected.items()},
   "ticket_type_counts":counts,"published_quote_counts":quoted,
   "no_votes_counts":{k:counts[k]-quoted[k] for k in counts},
   "quotes":extracted,"actual_bets_placed":False}
 path=Path("/tmp/fuji-jra-8-live-odds.json")
 path.write_text(json.dumps(output,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
 print("FUJI_8_REAL_ODDS_PASS "+json.dumps({
   "race_id":RACE,"published_minute":output["publisher_minute_label"],
   "counts":counts,"quoted_counts":quoted,"no_votes_counts":output["no_votes_counts"],
   "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
   "samples":[x for x in extracted if x["status"]=="QUOTED" and (
     x["combination"]==[4] or x["combination"]==[4,7] or x["combination"]==[4,5,7] or x["combination"]==[4,5,7])][:12],
   "runner_local_json_only":str(path)},ensure_ascii=False))
if __name__=="__main__":
 try:main()
 except Exception as exc:
  print("FUJI_8_REAL_ODDS_BLOCKED "+json.dumps({"reason":type(exc).__name__+":"+str(exc)},ensure_ascii=False))
  sys.exit(2)
