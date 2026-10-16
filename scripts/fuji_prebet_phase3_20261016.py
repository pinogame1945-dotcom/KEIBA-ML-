#!/usr/bin/env python3
"""Phase 3 read-only: stage-updatable Fuji body weight, weather & TURF going.
Distinguish actual page values, day-of values and prepublication. No stale carryover.
"""
import hashlib,json,re,sys,unicodedata
from datetime import datetime,timezone
from urllib.parse import urlparse
from urllib.request import Request,urlopen
from bs4 import BeautifulSoup
from pathlib import Path
RID="202605040611"
URL=f"https://race.netkeiba.com/race/shutuba.html?race_id={RID}"
JRA_BABA="https://www.jra.go.jp/keiba/baba/"
DAY="2026-10-17"
RACES=Path("audit-results/fuji-20261017-149-asof-20261016.json")
expected=json.loads(RACES.read_text(encoding="utf-8"))
IDs={x["horse_id"] for x in expected["runners"]}
assert len(IDs)==14 and expected["race_id"]==RID
def clean(v):return unicodedata.normalize("NFKC",str(v or "")).strip()
def fetch(url,hostname):
 with urlopen(Request(url,headers={"User-Agent":"Mozilla/5.0 (compatible; FujiPrebetP3/1.0)","Accept-Language":"ja"}),timeout=25) as h:
  if urlparse(h.geturl()).hostname!=hostname:raise ValueError("UNTRUSTED_REDIRECT")
  data=h.read(6_000_000)
  if len(data)>=6_000_000:raise ValueError("SIZE_LIMIT")
  cs=h.headers.get_content_charset() or "utf-8"
  html=None
  for encoding in dict.fromkeys((cs,"utf-8","cp932","shift_jis","euc_jp")):
   try:
    candidate=data.decode(encoding)
    if "馬場" in candidate or "富士" in candidate or "東京" in candidate:
     html=candidate
     break
   except (UnicodeError,LookupError):pass
  if html is None:raise UnicodeError("JAPANESE_DOCUMENT_ENCODING_UNKNOWN")
  if any(w in html for w in ["アクセス制限","不正なアクセス"]):raise ValueError("SOURCE_DENIED")
  return html,hashlib.sha256(data).hexdigest()
def parse_card(html):
 soup=BeautifulSoup(html,"html.parser")
 name=soup.select_one(".RaceName")
 if not name or "富士" not in clean(name.get_text(" ",strip=True)):raise ValueError("WRONG_RACE")
 table=soup.select_one("table.Shutuba_Table")
 if not table:raise ValueError("SHUTUBA_TABLE_MISSING")
 header=table.select_one("tr")
 headers=[clean(x.get_text(" ",strip=True)).replace(" ","") for x in header.find_all("th")]
 idx=next((k for k,x in enumerate(headers) if "馬体重" in x),None)
 if idx is None:raise ValueError("BODY_WEIGHT_COLUMN_MISSING")
 rows=[]
 for tr in table.select("tr"):
  td=tr.find_all("td",recursive=False)
  if not td:continue
  m=re.search(r"/horse/(\d+)",str(tr))
  if not m:continue
  horse_id=m.group(1)
  if horse_id not in IDs:raise ValueError("UNKNOWN_HORSE:"+horse_id)
  if idx>=len(td):raise ValueError("BODY_WEIGHT_CELL_MISSING:"+horse_id)
  raw=clean(td[idx].get_text(" ",strip=True))
  weight=re.fullmatch(r"\s*(\d{3})\s*\(\s*([+-]?\d+)\s*\)\s*",raw)
  number=int(weight.group(1)) if weight else None
  change=int(weight.group(2)) if weight else None
  if number is not None and not 200<=number<=800:raise ValueError("BODY_WEIGHT_RANGE")
  rows.append({"horse_id":horse_id,"raw":raw,"body_weight":number,
               "body_weight_diff":change,
               "status":"PUBLISHED" if weight else "UNRELEASED_OR_UNPARSEABLE"})
 if len(rows)!=14 or len(set(x["horse_id"] for x in rows))!=14:raise ValueError("HORSES_14_MISMATCH")
 raw_course=clean(soup.select_one(".RaceData01").get_text(" ",strip=True)) if soup.select_one(".RaceData01") else ""
 weather_match=re.search(r"天候\s*[:：]\s*(晴|曇|雨|小雨|雪|小雪)",raw_course)
 # Match the TURF condition, never DIRT going from another race.
 going_match=re.search(r"(?:芝|馬場)\s*[:：]\s*(良|稍重|重|不良)",raw_course)
 return {"race_data01":raw_course,"weather":weather_match.group(1) if weather_match else None,
         "turf_going":going_match.group(1) if going_match else None,
         "runners":rows,
         "weight_published_count":sum(x["status"]=="PUBLISHED" for x in rows)}
def parse_jra_baba(html):
 soup=BeautifulSoup(html,"html.parser")
 text=clean(soup.get_text(" ",strip=True))
 # JRA's front-page field information is often stale pre-race, so require an
 # explicit DAY-OF observation instead of previous Friday's current value.
 date_verifiable=bool(re.search(r"10\s*月\s*17\s*日.{0,55}(?:現在|時点)",text))
 date_header=("2026年10月17日" in text or "2026年 10月 17日" in text)
 if not date_verifiable or not date_header:
  return {"status":"STALE_OR_UNDATED","as_of_race_day":False,"weather":None,"turf_going":None,"snippet":text[:220]}
 m=re.search(r"天候\s*[:：]\s*(晴|曇|雨|小雨|雪|小雪)",text)
 t=re.search(r"芝\s*(?:コース)?\s*[:：]?\s*(良|稍重|重|不良)",text)
 return {"status":"DAY_OF_JRA_OBSERVATION" if m and t else "DAY_OF_PARSE_INCOMPLETE",
         "as_of_race_day":True,"weather":m.group(1) if m else None,"turf_going":t.group(1) if t else None,
         "snippet":text[:220]}
def main():
 out={"contract":"FUJI_PREBET_PHASE3_SOURCE_V1","race_id":RID,"race_date":DAY,
      "observed_at_utc":datetime.now(timezone.utc).isoformat(),
      "netkeiba_source":URL,"jra_turf_source":JRA_BABA,
      "netkeiba":None,"jra_track":None,
      "source_errors":[],"confirmed_current_race_data":{}}
 try:
  html,sha=fetch(URL,"race.netkeiba.com")
  out["netkeiba"]=parse_card(html)
  out["netkeiba"]["html_sha256"]=sha
 except Exception as e:out["source_errors"].append({"source":"netkeiba","error":type(e).__name__+":"+str(e)})
 try:
  html,sha=fetch(JRA_BABA,"www.jra.go.jp")
  out["jra_track"]=parse_jra_baba(html)
  out["jra_track"]["html_sha256"]=sha
 except Exception as e:out["source_errors"].append({"source":"jra_baba","error":type(e).__name__+":"+str(e)})
 if out["netkeiba"]:
  out["confirmed_current_race_data"]={
    "race.weather":out["netkeiba"]["weather"],
    "race.track_condition":out["netkeiba"]["turf_going"],
    "entry.body_weight":out["netkeiba"]["weight_published_count"],
    "entry.body_weight_diff":out["netkeiba"]["weight_published_count"]}
  if out["jra_track"] and out["jra_track"]["as_of_race_day"]:
   out["jra_netkeiba_weather_agreement"]=(out["jra_track"]["weather"]==out["netkeiba"]["weather"]
     if out["netkeiba"]["weather"] and out["jra_track"]["weather"] else None)
   out["jra_netkeiba_turf_agreement"]=(out["jra_track"]["turf_going"]==out["netkeiba"]["turf_going"]
     if out["netkeiba"]["turf_going"] and out["jra_track"]["turf_going"] else None)
 print("PHASE3_SUMMARY "+json.dumps({k:v for k,v in out.items() if k not in ("netkeiba","jra_track")},ensure_ascii=False))
 if out["netkeiba"]:print("NETKEIBA_4_ITEM_STATUS "+json.dumps({"weather":out["netkeiba"]["weather"],"going":out["netkeiba"]["turf_going"],"weight_published":out["netkeiba"]["weight_published_count"],"runner_count":len(out["netkeiba"]["runners"])},ensure_ascii=False))
 if out["jra_track"]:print("JRA_BABA_STATUS "+json.dumps(out["jra_track"],ensure_ascii=False))
 print("PHASE3_JSON "+json.dumps(out,ensure_ascii=False))
 return 0 if out["netkeiba"] and out["jra_track"] else 2
if __name__=="__main__":sys.exit(main())
