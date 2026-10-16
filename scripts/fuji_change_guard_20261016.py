#!/usr/bin/env python3
"""2026 Fuji: fail-closed pre-bet field-change detector (no prediction/auto-recompute).

Accepts current netkeiba public racecard and reference previous verified snapshot.
Does not look up historical results, change any trained model, or alter BACKFILL.
"""
import argparse,hashlib,json,re,sys,unicodedata
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request,urlopen
from bs4 import BeautifulSoup

RID="202605040611"
DATE="2026-10-17"
URL=f"https://race.netkeiba.com/race/shutuba.html?race_id={RID}"
BASE=Path("audit-results/fuji-20261017-149-asof-20261016.json")
IMPACT_FIELDS=("horse_name","gate","horse_number","sex","age","carried_weight","jockey_id","trainer_id","entry_status","body_weight","body_weight_diff")
HORSE_RE=re.compile(r"/horse/(\d{8,12})(?:/|[?#]|$)")
JOCKEY_RE=re.compile(r"/jockey/(?:result/recent/)?(\d{4,6})(?:/|[?#]|$)")
TRAINER_RE=re.compile(r"/trainer/(?:result/recent/)?(\d{4,6})(?:/|[?#]|$)")

def clean(v):return re.sub(r"\s+","",unicodedata.normalize("NFKC",str(v or "")))
def iso_time(s):
 d=datetime.fromisoformat(s.replace("Z","+00:00"))
 if d.tzinfo is None or d.utcoffset() is None:raise ValueError("TIMEZONE_AWARE_REQUIRED")
 return d.astimezone(timezone.utc)
def fetch():
 with urlopen(Request(URL,headers={"User-Agent":"Mozilla/5.0 (compatible; FujiFieldChangeAudit/1.0)","Accept-Language":"ja"}),timeout=22) as r:
  if r.status!=200 or urlparse(r.geturl()).hostname!="race.netkeiba.com":raise ValueError("WRONG_SOURCE")
  data=r.read(6_000_001)
  if len(data)>6_000_000:raise ValueError("HTML_TOO_LARGE")
  charset=r.headers.get_content_charset() or "utf-8"
  date=r.headers.get("Date")
 html=None
 for enc in dict.fromkeys((charset,"utf-8","cp932","euc_jp")):
  try:
   candidate=data.decode(enc)
   if "富士" in candidate and "馬名" in candidate:html=candidate;break
  except (UnicodeError,LookupError):pass
 if html is None:raise ValueError("JAPANESE_HTML_DECODE_FAILED")
 return html,hashlib.sha256(data).hexdigest(),date

def row_link_id(tr,pattern,cell=None):
 nodes=(cell or tr).select("a[href]")
 vals={m.group(1) for a in nodes if (m:=pattern.search(a["href"]))}
 if len(vals)>1:raise ValueError("AMBIGUOUS_RELATED_ID")
 return next(iter(vals),None)
def parse(html):
 soup=BeautifulSoup(html,"html.parser")
 title=clean(soup.select_one(".RaceName").get_text(" ",strip=True)) if soup.select_one(".RaceName") else ""
 race_condition=clean(soup.select_one(".RaceData02").get_text(" ",strip=True)) if soup.select_one(".RaceData02") else ""
 course=clean(soup.select_one(".RaceData01").get_text(" ",strip=True)) if soup.select_one(".RaceData01") else ""
 if "富士" not in title or "4回東京6日" not in race_condition or not ("芝" in course and "1600m" in course):
  raise ValueError("RACE_IDENTITY_OR_COURSE_NOT_VERIFIABLE")
 table=soup.select_one("table.Shutuba_Table")
 if table is None:raise ValueError("SHUTUBA_TABLE_NOT_FOUND")
 head=[clean(th.get_text(" ",strip=True)) for th in table.select("tr th")][:35]
 def column(key):
  return next((i for i,v in enumerate(head) if key in v),None)
 col={x:column(x) for x in ("枠","馬番","馬名","性齢","斤量","騎手","厩舎","馬体重")}
 if col["厩舎"] is None:col["厩舎"]=column("調教師")
 if any(col[x] is None for x in ("枠","馬番","馬名","性齢","斤量","騎手","厩舎")):
  raise ValueError("REQUIRED_COLS_UNAVAILABLE")
 rows=[]
 for tr in table.select("tr"):
  cells=tr.find_all("td",recursive=False)
  if not cells:continue
  horse_id=row_link_id(tr,HORSE_RE)
  if horse_id is None:
   if re.search(r"出走取消|出走除外|競走除外",clean(tr.get_text(" ",strip=True))):
    raise ValueError("CANCELLED_ROW_WITHOUT_HORSE_ID")
   continue
  def text(key):
   i=col[key]
   return clean(cells[i].get_text(" ",strip=True)) if i is not None and i<len(cells) else ""
  def id_for(key,pattern):
   i=col[key]
   return row_link_id(tr,pattern,cells[i]) if i is not None and i<len(cells) else None
  name_cell=cells[col["馬名"]]
  name_link=next((a for a in name_cell.select("a[href]") if HORSE_RE.search(a["href"])),None)
  horse_name=clean(name_link.get_text(" ",strip=True)) if name_link else None
  if not horse_name:raise ValueError("HORSE_NAME_NOT_FOUND")
  stxt=clean(tr.get_text(" ",strip=True))
  status="SCRATCHED" if ("出走取消" in stxt or re.search(r"(?<!出)取消",stxt)) else "EXCLUDED" if ("競走除外" in stxt or "出走除外" in stxt) else "LISTED"
  sexage=re.fullmatch(r"(牡|牝|セ)(\d+)",text("性齢"))
  number=lambda x: int(x) if re.fullmatch(r"\d+",x) else None
  kilograms=text("斤量")
  weight= float(kilograms) if re.fullmatch(r"\d+(?:\.\d+)?",kilograms) else None
  body=re.fullmatch(r"(\d{3})\(([+-]?\d+)\)",text("馬体重"))
  row={"horse_id":horse_id,"horse_name":horse_name,
       "horse_number":number(text("馬番")),"gate":number(text("枠")),
       "sex":sexage.group(1) if sexage else None,"age":int(sexage.group(2)) if sexage else None,
       "carried_weight":weight,"jockey_id":id_for("騎手",JOCKEY_RE),
       "trainer_id":id_for("厩舎",TRAINER_RE),"entry_status":status,
       "body_weight":int(body.group(1)) if body else None,
       "body_weight_diff":int(body.group(2)) if body else None}
  if status=="LISTED" and (row["horse_number"] is None or row["jockey_id"] is None or row["trainer_id"] is None or row["sex"] is None or row["carried_weight"] is None):
   raise ValueError("ACTIVE_FIELD_REQUIRED_ID_OR_NUMBER_MISSING:"+horse_id)
  rows.append(row)
 if not rows:raise ValueError("NO_RUNNERS_PARSED")
 ids=[x["horse_id"] for x in rows]
 if len(ids)!=len(set(ids)):raise ValueError("HORSE_ID_DUPLICATE")
 horse_numbers=[x["horse_number"] for x in rows if x["horse_number"] is not None]
 if len(horse_numbers)!=len(set(horse_numbers)):raise ValueError("HORSE_NUMBER_DUPLICATE")
 return {"race_id":RID,"race_date":DATE,"race_name":title,"course":course,
         "condition":race_condition,"horses":sorted(rows,key=lambda x:x["horse_id"])}

def compare(baseline,current):
 if baseline.get("race_id")!=RID or current.get("race_id")!=RID:raise ValueError("RACE_ID_DRIFT")
 old=baseline.get("runners")
 now=current.get("horses")
 if not isinstance(old,list) or not isinstance(now,list) or not old or not now:raise ValueError("NO_RUNNERS")
 past={str(h["horse_id"]):h for h in old}
 present={str(h["horse_id"]):h for h in now}
 if len(past)!=len(old) or len(present)!=len(now):raise ValueError("DUPLICATE_HORSE_ID")
 events=[]
 for hid in sorted(past.keys()-present.keys()):events.append({"horse_id":hid,"event":"REMOVED_FROM_CARD","needs_recalculation":True})
 for hid in sorted(present.keys()-past.keys()):events.append({"horse_id":hid,"event":"ADDED_TO_CARD","needs_recalculation":True})
 for hid in sorted(past.keys()&present.keys()):
  p,n=past[hid],present[hid]
  for key in IMPACT_FIELDS:
   oldval=p.get(key,"LISTED" if key=="entry_status" else None)
   newval=n.get(key,None)
   if oldval!=newval:events.append({"horse_id":hid,"horse_name":n.get("horse_name"),
     "event":"FIELD_CHANGED","field":key,"previous":oldval,"current":newval,"needs_recalculation":True})
 # Classification does NOT equate cancellation with race-day DNF.
 canceled=[h["horse_id"] for h in now if h.get("entry_status") in ("SCRATCHED","EXCLUDED")]
 return {"contract":"FUJI_PREBET_ENTRY_CHANGE_GUARD_V1","race_id":RID,
  "baseline_runner_count":len(old),"current_row_count":len(now),"current_scratched_or_excluded":canceled,
  "changes":events,"changed_count":len(events),
  "status":"INVALIDATE_RECOMPUTE_REQUIRED" if events else "UNCHANGED_NO_RECOMPUTE",
  "prediction_invalidated":bool(events),
  "actual_prediction_rerun":False,"actual_prediction_written":False}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--reference",default=str(BASE))
 p.add_argument("--html",help="offline tested fixture; skip actual fetch")
 a=p.parse_args()
 baseline=json.loads(Path(a.reference).read_text("utf-8"))
 observation=datetime.now(timezone.utc).isoformat()
 if a.html:html=Path(a.html).read_text("utf-8");sha=hashlib.sha256(html.encode()).hexdigest();source="FIXTURE"
 else:html,sha,http_date=fetch();source=URL
 report=compare(baseline,parse(html))
 report.update(current_source=source,current_observed_at_utc=observation,current_html_sha256=sha,
               reference_observed_at_utc=baseline.get("observed_at_client_utc"),
               reference_html_sha256=baseline.get("source_sha256"))
 print("CHANGE_GUARD_RESULT "+json.dumps(report,ensure_ascii=False))
if __name__=="__main__":
 try:main()
 except Exception as e:
  print("CHANGE_GUARD_BLOCKED "+json.dumps({"status":"BLOCKED_SOURCE_UNVERIFIED","reason":type(e).__name__+":"+str(e)},ensure_ascii=False))
  sys.exit(2)
