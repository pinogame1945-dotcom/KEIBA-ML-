#!/usr/bin/env python3
"""Fuji same-race / timestamp integrity gate. Validation only; never places bets.
JRA live card and netkeiba current card are checked together; client fetch time
is not falsely represented as the publisher's last data update.
"""
import argparse,hashlib,importlib.util,json,re,sys,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
from urllib.parse import urlparse
from urllib.request import Request,urlopen
from bs4 import BeautifulSoup

RID="202605040611"
RACE_DATE="2026-10-17"
OFFICIAL="https://www.jra.go.jp/JRADB/accessD.html?CNAME=pw01dde0105202604061120261017%2F22"
ORIGINAL_CARD="https://race.netkeiba.com/race/shutuba.html?race_id=202605040611"
spec=importlib.util.spec_from_file_location("fuji_change_guard",Path("scripts/fuji_change_guard_20261016.py"))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
BASE=json.loads(g.BASE.read_text(encoding="utf-8"))
def clean(v):return re.sub(r"\s+","",str(v or ""))
def strict_time(v):
 if not isinstance(v,str):raise ValueError("SOURCE_TIMESTAMP_ABSENT")
 try:dt=datetime.fromisoformat(v.replace("Z","+00:00"))
 except ValueError:raise ValueError("SOURCE_TIMESTAMP_INVALID") from None
 if dt.tzinfo is None:raise ValueError("SOURCE_TIMESTAMP_MISSING_TZ")
 return dt.astimezone(timezone.utc)
def snapshot_hash(text):return hashlib.sha256(text.encode("utf-8")).hexdigest()
def fetch_jra():
 u=urlparse(OFFICIAL)
 if u.hostname!="www.jra.go.jp" or u.scheme!="https":raise ValueError("OFFICIAL_CARD_URL_DRIFT")
 with urlopen(Request(OFFICIAL,headers={"User-Agent":"Mozilla/5.0 (compatible; FujiIntegrityAudit/1.0)","Accept-Language":"ja"}),timeout=24) as r:
  if urlparse(r.geturl()).hostname!="www.jra.go.jp":raise ValueError("JRA_REDIRECT")
  raw=r.read(6_000_001);server_date=r.headers.get("Date")
  if len(raw)>6_000_000:raise ValueError("OFFICIAL_HTML_TOO_LARGE")
  cs=r.headers.get_content_charset()
 html=None
 for enc in dict.fromkeys([cs,"utf-8","cp932","euc_jp"]):
  if not enc:continue
  try:
   v=raw.decode(enc)
   if "出馬表" in v and "富士" in v:html=v;break
  except (UnicodeError,LookupError):pass
 if html is None:raise ValueError("JRA_HTML_DECODE_FAILED")
 return html,hashlib.sha256(raw).hexdigest(),server_date

def parse_jra_official(html,candidate_names):
 soup=BeautifulSoup(html,"html.parser")
 text=clean(soup.get_text(" ",strip=True))
 if not all(x in text for x in ("富士ステークス","2026年10月17日","4回東京6日","11レース")):
  raise ValueError("OFFICIAL_RACE_HEADER_INCOMPATIBLE")
 entries=[]
 for table in soup.select("table"):
  for tr in table.select("tr"):
   cells=tr.find_all(["th","td"],recursive=False)
   if len(cells)<3:continue
   name_text=clean(cells[2].get_text(" ",strip=True))
   matched=[name for name in candidate_names if name in name_text]
   if not matched:continue
   if len(matched)>1:raise ValueError("OFFICIAL_AMBIGUOUS_HORSE_NAME")
   name=matched[0]
   num=clean(cells[1].get_text(" ",strip=True))
   number=int(num) if re.fullmatch(r"(?:\d{1,2})",num) else None
   gats=[]
   for img in cells[0].select("img[alt]"):
    m=re.search(r"枠([1-8])",img.get("alt",""))
    if m:gats.append(int(m.group(1)))
   gate_text=clean(cells[0].get_text(" ",strip=True))
   gate=gats[0] if len(gats)==1 else int(gate_text) if re.fullmatch(r"[1-8]",gate_text) else None
   raw=clean(tr.get_text(" ",strip=True))
   status="SCRATCHED" if "出走取消" in raw else "EXCLUDED" if "競走除外" in raw or "出走除外" in raw else "LISTED"
   entries.append({"horse_name":name,"horse_number":number,"gate":gate,"entry_status":status})
 if not entries or len({x["horse_name"] for x in entries})!=len(entries):
  raise ValueError("JRA_RUNNER_LIST_INCOMPLETE_OR_DUPLICATE")
 return entries

def check_join(current,official):
 actual=current["horses"]
 official_by_name={x["horse_name"]:x for x in official}
 if set(official_by_name)!={x["horse_name"] for x in actual}:
  return {"status":"BLOCKED_RUNNER_SET_DIVERGENCE","missing_official":sorted({x["horse_name"] for x in actual}-set(official_by_name)),
          "unmatched_official":sorted(set(official_by_name)-{x["horse_name"] for x in actual})}
 changed=[]
 for h in actual:
  x=official_by_name[h["horse_name"]]
  for field in ("horse_number","gate","entry_status"):
   if h.get(field)!=x.get(field):
    changed.append({"horse_id":h["horse_id"],"horse_name":h["horse_name"],"field":field,
                    "netkeiba":h.get(field),"jra":x.get(field)})
 return {"status":"CONFIRMED" if not changed else "BLOCKED_SOURCE_DISAGREEMENT",
         "disagreements":changed,"official_verified_runners":len(official)}

def integrity(current,official,change,net_meta,jra_meta,*,now_utc,max_age_seconds=None,max_skew_seconds=None,odds=None):
 errors=[]
 if current["race_id"]!=RID:errors.append("NETKEIBA_WRONG_RACE")
 if net_meta.get("race_id")!=RID or jra_meta.get("race_id")!=RID:errors.append("SOURCE_RACE_ID_MISMATCH")
 for key,host in (("net","race.netkeiba.com"),("jra","www.jra.go.jp")):
  rec=net_meta if key=="net" else jra_meta
  url=urlparse(rec.get("url",""))
  if url.scheme!="https" or url.hostname!=host:errors.append(key.upper()+"_UNTRUSTED_SOURCE_URL")
  if not re.fullmatch(r"[0-9a-f]{64}",str(rec.get("sha256",""))):errors.append(key.upper()+"_SOURCE_HASH_MISSING")
 now=strict_time(now_utc)
 moments={}
 for key,rec in (("net",net_meta),("jra",jra_meta)):
  try:
   d=strict_time(rec.get("fetched_at_utc"));moments[key]=d
   if d>now+timedelta(seconds=5):errors.append(key.upper()+"_SOURCE_CLOCK_IN_FUTURE")
   if max_age_seconds is None:errors.append("FRESHNESS_LIMIT_NOT_SUPPLIED")
   elif (now-d).total_seconds()>max_age_seconds:errors.append(key.upper()+"_SOURCE_STALE")
  except ValueError as e:errors.append(key.upper()+"_"+str(e))
 if max_skew_seconds is None:errors.append("CROSS_SOURCE_SKEW_LIMIT_NOT_SUPPLIED")
 elif len(moments)==2 and abs((moments["net"]-moments["jra"]).total_seconds())>max_skew_seconds:
  errors.append("SOURCE_TIMESTAMPS_TOO_FAR_APART")
 source_join=check_join(current,official)
 if source_join["status"]!="CONFIRMED":errors.append(source_join["status"])
 if change["status"]=="INVALIDATE_RECOMPUTE_REQUIRED":errors.append("INPUT_CHANGED_OLD_PREDICTION_INVALID")
 if odds is not None:
  try:
   spec2=importlib.util.spec_from_file_location("fuji_jra_odds_validate",Path("scripts/fuji_prebet_phase4_odds_validator_20261016.py"))
   validator=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(validator)
   validated=validator.validate_quote_batch(odds)
   for rec in validated["quotes"]:
    if rec["bet_type"]!="BRACKET_QUINELLA" and any(no not in [r["horse_number"] for r in current["horses"] if r["entry_status"]=="LISTED"] for no in rec["combination"]):
     errors.append("ODDS_INCLUDE_WITHDRAWN_OR_UNKNOWN_HORSE_NUMBER")
   if strict_time(odds["fetched_at_utc"])>now+timedelta(seconds=5):
    errors.append("ODDS_FETCH_TIME_IN_FUTURE")
   elif max_age_seconds is not None and (now-strict_time(odds["fetched_at_utc"])).total_seconds()>max_age_seconds:
    errors.append("ODDS_SOURCE_STALE")
   if len(moments)==2 and max_skew_seconds is not None and abs((strict_time(odds["fetched_at_utc"])-moments["net"]).total_seconds())>max_skew_seconds:
    errors.append("ODDS_VS_CARD_SKEW")
   # Publisher update timestamps are optional, but cannot be silently invented
  except (ValueError,KeyError) as e:errors.append("ODDS_CONTRACT_"+str(e))
 # No live prices / track conditions today: prevent "bet ready" from being claimed.
 if errors:status="BLOCKED_INTEGRITY"
 elif odds is None:status="CARD_CROSSCHECKED_WAITING_FOR_LIVE_ODDS"
 else:status="SOURCE_INTEGRITY_CHECKED_NOT_MODEL_CERTIFIED"
 return {"contract":"FUJI_PREBET_SOURCE_INTEGRITY_V1","race_id":RID,
  "status":status,"reasons":sorted(set(errors)),"race_header_verified":True,
  "entry_changed":change["prediction_invalidated"],
  "recalculation_needed":bool(change["prediction_invalidated"]),
  "cross_source":source_join,"quote_available":odds is not None,
  "max_age_seconds":max_age_seconds,"max_skew_seconds":max_skew_seconds,
  "clock_basis":"CLIENT_FETCH_OBSERVED_TIME_NOT_PUBLISHER_UPDATE_TIMESTAMP",
  "no_final_odds_substitution":True,"actual_prediction_rerun":False,
  "actual_bets_placed":False}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--max-age-seconds",type=int,required=True,help="Caller-supplied freshness policy for this test only")
 p.add_argument("--max-skew-seconds",type=int,required=True,help="Caller-supplied cross-source policy for this test only")
 a=p.parse_args()
 if not 1<=a.max_age_seconds<=86400 or not 1<=a.max_skew_seconds<=86400:
  raise ValueError("INVALID_FRESHNESS_POLICY")
 nhtml,nsha,nserver=g.fetch();net_at=datetime.now(timezone.utc).isoformat()
 current=g.parse(nhtml)
 jhtml,jsha,jserver=fetch_jra();jra_at=datetime.now(timezone.utc).isoformat()
 official=parse_jra_official(jhtml,{x["horse_name"] for x in current["horses"]})
 change=g.compare(BASE,current)
 net_meta={"race_id":RID,"url":ORIGINAL_CARD,"sha256":nsha,"fetched_at_utc":net_at,"http_date":nserver}
 jra_meta={"race_id":RID,"url":OFFICIAL,"sha256":jsha,"fetched_at_utc":jra_at,"http_date":jserver}
 report=integrity(current,official,change,net_meta,jra_meta,now_utc=datetime.now(timezone.utc).isoformat(),
                  max_age_seconds=a.max_age_seconds,max_skew_seconds=a.max_skew_seconds)
 report.update(netkeiba=net_meta,jra=jra_meta,changes=change["changes"],current_horses=len(current["horses"]))
 print("FUJI_INTEGRITY_RESULT "+json.dumps(report,ensure_ascii=False))
 if report["status"]=="BLOCKED_INTEGRITY":return 2
 return 0
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as e:
  print("FUJI_INTEGRITY_BLOCKED "+json.dumps({"status":"BLOCKED_SOURCE_UNVERIFIED","reason":type(e).__name__+":"+str(e)},ensure_ascii=False))
  sys.exit(2)
