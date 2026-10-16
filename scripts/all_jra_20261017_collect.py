#!/usr/bin/env python3
"""2026-10-17 JRA all 36 races: pre-race source audit + published 8 bet-type odds.
Discovery follows only CNAMEs/links actually emitted by official JRA HTML.
No model/L2/L3 changes, no post-race data, no guessed prices, no periodic polling.
"""
import argparse
import hashlib
import itertools
import json
import math
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

sys.path.insert(0,"scripts")
import fuji_onsale_jra_deep_probe_20261016 as transport
import fuji_8type_live_odds_20261016 as base_odds
import fuji_integrity_guard_20261016 as official_guard
import audit_fuji_149_asof_20261016 as catalog_audit

DAY="2026-10-17"
TOKEN="20261017"
JRA_D="https://www.jra.go.jp/JRADB/accessD.html"
JRA_O="https://www.jra.go.jp/JRADB/accessO.html"
MEETINGS={"05":("東京",4,6),"08":("京都",4,6),"04":("新潟",4,1)}
BET_TABLE={"BRACKET_QUINELLA":"waku","QUINELLA":"umaren","WIDE":"wide",
           "EXACTA":"umatan","TRIO":"fuku3","TRIFECTA":"tan3"}
BET_LABELS=base_odds.TYPES
HORSE=re.compile(r"/horse/([0-9A-Za-z]+)(?:/|[?#]|$)")
JOCKEY=re.compile(r"/jockey/(?:result/recent/)?(\d+)(?:/|[?#]|$)")
TRAINER=re.compile(r"/trainer/(?:result/recent/)?(\d+)(?:/|[?#]|$)")
CARD_CNAME=re.compile(r"pw01dde01(?P<venue>\d{2})2026(?P<meet>\d{2})(?P<day>\d{2})(?P<no>\d{2})20261017/[0-9A-Fa-f]{2}")
ACTION=re.compile(r"doAction\(\s*['\"](/JRADB/access[DO]\.html)['\"]\s*,\s*['\"]([^'\"]+)['\"]")
def utc():return datetime.now(timezone.utc).isoformat()
def clean(v):return re.sub(r"\s+","",unicodedata.normalize("NFKC",str(v or ""))).strip()
def src_fetch(url):
 from urllib.error import HTTPError,URLError
 for attempt in range(3):
  try:return src_fetch_once(url)
  except (URLError,TimeoutError) as exc:
   retryable=(isinstance(exc,HTTPError) and exc.code in (502,503,504)) or "timed out" in str(exc).lower()
   if not retryable or attempt==2:raise
   time.sleep(1.5*(attempt+1))
 raise RuntimeError("SOURCE_FETCH_RETRIES_EXHAUSTED")
def src_fetch_once(url):
 if urlparse(url).hostname not in ("www.jra.go.jp","race.netkeiba.com"):raise ValueError("UNTRUSTED_SOURCE")
 with urlopen(Request(url,headers={"User-Agent":"Mozilla/5.0","Accept-Language":"ja"}),timeout=24) as r:
  if urlparse(r.geturl()).hostname!=urlparse(url).hostname:raise ValueError("UNTRUSTED_REDIRECT")
  raw=r.read(6_000_001)
  if len(raw)>6_000_000:raise ValueError("HTML_TOO_LARGE")
  declared=r.headers.get_content_charset()
  html=None
  for enc in dict.fromkeys([declared,"utf-8","cp932","euc_jp"]):
   if not enc:continue
   try:
    candidate=raw.decode(enc)
    if "馬" in candidate or "出馬" in candidate or "レース" in candidate:
     html=candidate;break
   except (UnicodeError,LookupError):continue
  if html is None:raise ValueError("UNREADABLE_SOURCE_HTML")
  if "アクセス制限" in html or "不正なアクセス" in html:raise ValueError("SOURCE_ACCESS_DENIED")
  return html,{"sha256":hashlib.sha256(raw).hexdigest(),"http_date":r.headers.get("Date"),"fetched_at_utc":utc()}
def action_of(anchor,path):
 m=ACTION.search(anchor.get("onclick",""))
 return m.group(2) if m and m.group(1)==path else None
def by_text(soup,name):
 return [a for a in soup.select("a") if clean(a.get_text(" ",strip=True))==name]
def official_fetch(url,cname=None):
 from urllib.error import HTTPError,URLError
 for attempt in range(3):
  try:return transport.fetch(url,cname)
  except (URLError,TimeoutError) as exc:
   retryable=(isinstance(exc,HTTPError) and exc.code in (502,503,504)) or "timed out" in str(exc).lower()
   if not retryable or attempt==2:raise
   time.sleep(1.5*(attempt+1))
 raise RuntimeError("UNREACHABLE_FETCH_RETRY")

def discover():
 root,root_meta=official_fetch(transport.CARD)
 parsed=BeautifulSoup(root,"html.parser")
 choices=[action_of(a,"/JRADB/accessD.html") for a in by_text(parsed,"レース選択へ戻る")]
 choices={x for x in choices if x and "20261017" in x}
 if len(choices)!=1:raise ValueError("OFFICIAL_CURRENT_RACE_SELECTION_UNAVAILABLE")
 index,meta=official_fetch(JRA_D,choices.pop())
 idx=BeautifulSoup(index,"html.parser")
 links={}
 for venue,(name,meet,day) in MEETINGS.items():
  label=f"{meet}回{name}{day}日"
  vals={action_of(a,"/JRADB/accessD.html") for a in by_text(idx,label)}
  vals.discard(None)
  vals={x for x in vals if TOKEN in x}
  if len(vals)!=1:raise ValueError("VENUE_SELECTION_NOT_UNIQUE:"+label)
  links[venue]=vals.pop()
 records=[]
 observations={}
 for venue,cname in links.items():
  html,venue_meta=official_fetch(JRA_D,cname)
  venue_meta.update(venue_code=venue,selection_cname=cname)
  observations[venue]=venue_meta
  soup=BeautifulSoup(html,"html.parser")
  selections=[]
  for a in soup.select("a[href]"):
   href=a.get("href","")
   m=CARD_CNAME.search(href)
   if not m:continue
   if m.group("venue")!=venue or (int(m.group("meet")),int(m.group("day")))!=(MEETINGS[venue][1],MEETINGS[venue][2]):
    raise ValueError("RACE_LINK_CROSS_VENUE")
   if clean(a.get_text(" ",strip=True))!="出馬表":continue  # Other JRA navigation links may point at a racecard
   next_link=a.find_next("a")
   if next_link is None or clean(next_link.get_text(" ",strip=True))!="オッズ":
    raise ValueError("CARD_ODDS_PAIR_NOT_ADJACENT")
   odds_code=action_of(next_link,"/JRADB/accessO.html")
   if not odds_code or "20261017" not in odds_code:raise ValueError("ODDS_LINK_INVALID")
   rid=f"2026{venue}{m.group('meet')}{m.group('day')}{m.group('no')}"
   selections.append({"race_id":rid,"venue_code":venue,"venue":MEETINGS[venue][0],
        "meeting":MEETINGS[venue][1],"meeting_day":MEETINGS[venue][2],
        "race_no":int(m.group("no")),"jra_card_url":urljoin("https://www.jra.go.jp",href),
        "jra_odds_cname":odds_code,"race_date":DAY})
  if len(selections)!=12 or {x["race_no"] for x in selections}!=set(range(1,13)):
   raise ValueError("NOT_TWELVE_OFFICIAL_RACE_LINKS:"+venue+":"+str(len(selections)))
  records.extend(selections)
 if len(records)!=36 or len({r["race_id"] for r in records})!=36:
  raise ValueError("NOT_36_UNIQUE_OFFICIAL_RACES")
 return sorted(records,key=lambda x:(x["venue_code"],x["race_no"])),{"root":root_meta,"index":meta,"venues":observations}
def link_id(cell,regex):
 if cell:
  for a in cell.select("a[href]"):
   m=regex.search(a.get("href",""))
   if m:return m.group(1)
 return None
def net_card(race):
 rid=race["race_id"]
 url=f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
 html,meta=src_fetch(url)
 soup=BeautifulSoup(html,"html.parser")
 tab=soup.select_one("table.Shutuba_Table")
 if tab is None:raise ValueError("NETKEIBA_SHUTUBA_TABLE_ABSENT")
 heads=[clean(x.get_text(" ",strip=True)) for x in tab.select("tr th")][:30]
 def col(*name):return next((i for i,x in enumerate(heads) if any(n in x for n in name)),None)
 cols={k:col(*items) for k,items in {
  "gate":("枠",),"horse_number":("馬番",),"horse_name":("馬名",),
  "sexage":("性齢",),"carried_weight":("斤量",),"jockey":("騎手",),
  "trainer":("厩舎","調教師"),"body_weight":("馬体重",)}.items()}
 def value(cells,key):
  i=cols[key]
  return clean(cells[i].get_text(" ",strip=True)) if i is not None and i<len(cells) else ""
 runners=[]
 for row in tab.select("tr"):
  cells=row.find_all("td",recursive=False)
  if not cells:continue
  horse_links=[a for a in row.select("a[href]") if HORSE.search(a.get("href",""))]
  if not horse_links:continue
  horse_ids={HORSE.search(a["href"]).group(1) for a in horse_links}
  if len(horse_ids)!=1:raise ValueError("HORSE_ID_AMBIGUOUS")
  def number(s):return int(s) if re.fullmatch(r"\d+",s) else None
  def decimal(s):return float(s) if re.fullmatch(r"\d+(?:\.\d+)?",s) else None
  sexage=re.search(r"(牡|牝|セ)(\d+)",value(cells,"sexage"))
  rawbody=value(cells,"body_weight")
  body=re.fullmatch(r"(\d{3})\(([+-]?\d+)\)",rawbody)
  runners.append({"horse_id":horse_ids.pop(),
   "horse_name":clean(next((a.get_text(" ",strip=True) for a in horse_links if clean(a.get_text(" ",strip=True))),"")),
   "gate":number(value(cells,"gate")),"horse_number":number(value(cells,"horse_number")),
   "sex":sexage.group(1) if sexage else None,"age":int(sexage.group(2)) if sexage else None,
   "carried_weight":decimal(value(cells,"carried_weight")),
   "jockey_id":link_id(cells[cols["jockey"]] if cols["jockey"] is not None and cols["jockey"]<len(cells) else row,JOCKEY),
   "trainer_id":link_id(cells[cols["trainer"]] if cols["trainer"] is not None and cols["trainer"]<len(cells) else row,TRAINER),
   "body_weight":int(body.group(1)) if body else None,
   "body_weight_diff":int(body.group(2)) if body else None,
   "source_cells":{k:value(cells,k) for k in cols}})
 n=len(runners)
 if not 3<=n<=18:raise ValueError("UNEXPECTED_RUNNER_COUNT:"+str(n))
 nums=[x["horse_number"] for x in runners]
 if sorted(nums)!=list(range(1,n+1)) or len({x["horse_id"] for x in runners})!=n or len({x["horse_name"] for x in runners})!=n:
  raise ValueError("NETKEIBA_HORSE_IDS_OR_NUMBERS_INVALID")
 raw_course=clean(soup.select_one(".RaceData01").get_text(" ",strip=True)) if soup.select_one(".RaceData01") else ""
 raw_condition=clean(soup.select_one(".RaceData02").get_text(" ",strip=True)) if soup.select_one(".RaceData02") else ""
 if race["venue"] not in raw_condition or f'{race["meeting"]}回' not in raw_condition or f'{race["meeting_day"]}日' not in raw_condition:
  raise ValueError("NETKEIBA_RACE_HEADER_MISMATCH")
 distance=re.search(r"(\d{3,4})m",raw_course)
 if not distance:raise ValueError("DISTANCE_NOT_PRESENT")
 surface="TURF" if "芝" in raw_course and "ダ" not in raw_course else "DIRT" if "ダ" in raw_course and "芝" not in raw_course else None
 classraw=next((x for x in ("新馬","未勝利","1勝クラス","2勝クラス","3勝クラス","オープン") if x in raw_condition),None)
 age=re.search(r"([234])歳(以上)?",raw_condition)
 grade=re.search(r"\(G([123])\)",clean(soup.title.get_text(" ",strip=True)),re.I) if soup.title else None
 raw={
  "race.venue_code":race["venue_code"],"race.meeting_no":race["meeting"],
  "race.meeting_day":race["meeting_day"],"race.race_no":race["race_no"],
  "race.race_id":rid,"race.discipline":"FLAT" if surface and "障害" not in raw_course else None,
  "race.surface":surface,"race.distance_m":int(distance.group(1)),
  "race.direction":"LEFT" if "左" in raw_course else "RIGHT" if "右" in raw_course else None,
  "race.weather":re.search(r"天候[:：]?([^/]+)",raw_course).group(1) if re.search(r"天候[:：]?([^/]+)",raw_course) else None,
  "race.track_condition":re.search(r"馬場[:：]?([^/]+)",raw_course).group(1) if re.search(r"馬場[:：]?([^/]+)",raw_course) else None,
  "race.actual_start_time":None,
  "race.course_layout":"OUTER" if "外" in raw_course else "INNER" if "内" in raw_course else None,
  "race.course_laps":int(re.search(r"(\d+)周",raw_course).group(1)) if re.search(r"(\d+)周",raw_course) else None,
  "race.race_class_raw":classraw,
  "race.race_class_normalized":{"新馬":"NEWCOMER","未勝利":"MAIDEN","1勝クラス":"ONE_WIN","2勝クラス":"TWO_WIN","3勝クラス":"THREE_WIN","オープン":"OPEN"}.get(classraw),
  "race.grade":"G"+grade.group(1) if grade else None,
  "race.age_condition_raw":age.group(0) if age else None,
  "race.age_min":int(age.group(1)) if age else None,
  "race.age_max":int(age.group(1)) if age and not age.group(2) else None,
  "race.sex_condition":"FEMALE_ONLY" if "牝馬限定" in raw_condition else "ANY",
  "race.weight_rule":next((x for k,x in (("ハンデ","HANDICAP"),("別定","SPECIAL_WEIGHT"),("定量","SET_WEIGHT"),("馬齢","WEIGHT_FOR_AGE")) if k in raw_condition),None),
  "race.mixed":"(混)" in raw_condition or "[混]" in raw_condition,
  "race.international":"(国際)" in raw_condition,
  "race.special_designated":"(特指)" in raw_condition,
  "race.designated":"(指)" in raw_condition or "[指]" in raw_condition,
  "race.course_meta_raw":raw_course,"race.race_condition_raw":raw_condition,"race.field_size":n}
 cat=json.loads(Path("contracts/l1-feature-catalog-v1.json").read_text("utf-8"))
 if len(cat.get("features",[]))!=149:raise ValueError("CATALOG_149_DRIFT")
 catalog_audit.EXPECTED=n
 catalog_audit.URL=url
 ledger=catalog_audit.audit(cat,runners,raw)
 for item in ledger:item["checked_from"]="ACTUAL_2026_CURRENT_RACE_PRE_RACE_SOURCE"
 return {"source_url":url,"meta":meta,"runner_count":n,"runners":runners,
   "race_name":clean(soup.select_one(".RaceName").get_text(" ",strip=True)) if soup.select_one(".RaceName") else None,
   "raw_course":raw_course,"raw_condition":raw_condition,
   "features_149":ledger,"feature_status_counts":{s:sum(f["status"]==s for f in ledger) for s in set(f["status"] for f in ledger)}}
def parse_live_jra_entries(html,names):
 soup=BeautifulSoup(html,"html.parser")
 records=[]
 for table in soup.select("table"):
  for tr in table.select("tr"):
   cells=tr.find_all(["th","td"],recursive=False)
   if len(cells)<3:continue
   name_text=clean(cells[2].get_text(" ",strip=True))
   matched=[n for n in names if n in name_text]
   if not matched:continue
   if len(matched)!=1:raise ValueError("AMBIGUOUS_JRA_HORSE_NAME")
   name=matched[0]
   num_text=clean(cells[1].get_text(" ",strip=True))
   num=int(num_text) if re.fullmatch(r"\d{1,2}",num_text) else None
   frame=[]
   for img in cells[0].select("img[alt]"):
    m=re.search(r"枠([1-8])",clean(img.get("alt","")))
    if m:frame.append(int(m.group(1)))
   raw_frame=clean(cells[0].get_text(" ",strip=True))
   frame_no=frame[0] if len(frame)==1 else int(raw_frame) if re.fullmatch(r"[1-8]",raw_frame) else None
   raw=clean(tr.get_text(" ",strip=True))
   status="SCRATCHED" if num_text in ("取消","出走取消") or "出走取消" in raw else "EXCLUDED" if num_text in ("除外","競走除外","出走除外") or "競走除外" in raw or "出走除外" in raw else "LISTED"
   records.append({"horse_name":name,"horse_number":num,"gate":frame_no,"entry_status":status})
 if not records or len({x["horse_name"] for x in records})!=len(records):
  raise ValueError("JRA_HORSE_LIST_INCOMPLETE_OR_DUPLICATE")
 return records

def official_card(race,net):
 html,meta=src_fetch(race["jra_card_url"])
 soup=BeautifulSoup(html,"html.parser")
 page=clean(soup.get_text(" ",strip=True))
 target=f'{race["meeting"]}回{race["venue"]}{race["meeting_day"]}日'
 if TOKEN not in race["jra_card_url"] or "2026年10月17日" not in page or target not in page or f'{race["race_no"]}レース' not in page:
  raise ValueError("JRA_CARD_RACE_IDENTITY_MISMATCH")
 expected_names={x["horse_name"] for x in net["runners"]}
 verified=parse_live_jra_entries(html,expected_names)
 parsed={x["horse_name"]:x for x in verified}
 if set(parsed)!=expected_names:raise ValueError("JRA_NETKEIBA_NAME_LIST_DIVERGENCE")
 differs=[]
 for runner in net["runners"]:
  x=parsed[runner["horse_name"]]
  for k in ("horse_number","gate"):
   if k=="horse_number" and x["entry_status"] in ("SCRATCHED","EXCLUDED") and x[k] is None:continue
   if x[k]!=runner[k]:differs.append({"horse":runner["horse_name"],"field":k,"jra":x[k],"netkeiba":runner[k]})
 if differs:raise ValueError("JRA_NETKEIBA_NUMBER_GATE_DIFFERENCE:"+json.dumps(differs,ensure_ascii=False)[:500])
 return {"meta":meta,"official_horse_set":verified,"identity_verified":True,
         "changed_or_scratch":[x for x in verified if x["entry_status"]!="LISTED"]}
def theoretical(n,gates):
 horses=list(range(1,n+1))
 frames={k:sum(v==k for v in gates.values()) for k in range(1,9)}
 valid_frame={tuple((i,j)) for i in range(1,9) for j in range(i,9) if (i!=j and frames[i] and frames[j]) or (i==j and frames[i]>=2)}
 return {"WIN":{(i,) for i in horses},"PLACE":{(i,) for i in horses},
         "BRACKET_QUINELLA":valid_frame,
         "QUINELLA":set(itertools.combinations(horses,2)),
         "WIDE":set(itertools.combinations(horses,2)),
         "EXACTA":set(itertools.permutations(horses,2)),
         "TRIO":set(itertools.combinations(horses,3)),
         "TRIFECTA":set(itertools.permutations(horses,3))}
def parse_matrix(soup,typ,n,gates,retired_numbers=frozenset()):
 if typ=="WIN_PLACE":
  return base_odds.parse_winplace(soup)
 cls=BET_TABLE[typ]
 tabs=soup.select("table."+cls)
 expected_tables={"BRACKET_QUINELLA":8,"QUINELLA":n-1,"WIDE":n-1,
                  "EXACTA":n,"TRIO":math.comb(n-1,2),"TRIFECTA":n*(n-1)}
 if len(tabs)!=expected_tables[typ]:raise ValueError("JRA_TABLE_COUNT_MISMATCH:"+typ+f':{len(tabs)}/{expected_tables[typ]}')
 parsed=[]
 for index,tab in enumerate(tabs):
  if typ=="BRACKET_QUINELLA":
   prefix=(index+1,);seconds=list(range(prefix[0],9))
  elif typ in ("QUINELLA","WIDE"):
   prefix=(index+1,);seconds=list(range(prefix[0]+1,n+1))
  elif typ=="EXACTA":
   prefix=(index+1,);seconds=list(range(1,n+1))
  elif typ=="TRIO":
   prefix=list(itertools.combinations(range(1,n),2))[index];seconds=list(range(prefix[-1]+1,n+1))
  else:
   prefix=list(itertools.permutations(range(1,n+1),2))[index]
   seconds=[x for x in range(1,n+1) if x!=prefix[0]]
  rows=tab.select("tr")
  if len(rows)!=len(seconds):raise ValueError("JRA_TABLE_ROW_COUNT_MISMATCH:"+typ)
  for row,b in zip(rows,seconds):
   cells=row.find_all(["td","th"],recursive=False)
   if len(cells)!=2:raise ValueError("JRA_MATRIX_TWO_CELL_RULE:"+typ)
   shown=clean(cells[0].get_text(" ",strip=True))
   if shown!=str(b):raise ValueError("JRA_MATRIX_ROW_ORDER:"+typ+":"+shown+"/"+str(b))
   quote=clean(cells[1].get_text(" ",strip=True))
   invalid=(typ=="EXACTA" and b==prefix[0]) or (typ=="TRIFECTA" and b==prefix[1])
   invalid=invalid or (typ=="BRACKET_QUINELLA" and b==prefix[0] and sum(v==b for v in gates.values())<2)
   invalid=invalid or (typ=="BRACKET_QUINELLA" and (b not in gates.values() or prefix[0] not in gates.values()))
   if invalid:
    if quote not in ("","-","－") and not (quote in ("取消","除外") and (b in retired_numbers or any(x in retired_numbers for x in prefix))):
     raise ValueError("IMPOSSIBLE_COMBINATION_HAS_PRICE:"+typ)
    continue
   if quote in ("","-","－"):raise ValueError("VALID_COMBINATION_HAS_EMPTY_QUOTE:"+typ)
   parsed.append((tuple(prefix)+(b,),quote))
 return {typ:parsed}
def parse_official_odds(race,net,official):
 runners=net["runners"];n=len(runners)
 gates={x["horse_number"]:x["gate"] for x in runners}
 retired_names={x["horse_name"]:x["entry_status"] for x in official["changed_or_scratch"]}
 retired_numbers={x["horse_number"] for x in runners if x["horse_name"] in retired_names}
 live_frames={g for no,g in gates.items() if no not in retired_numbers}
 if any(x not in range(1,9) for x in gates.values()):raise ValueError("NO_VALID_GATE_FRAME")
 # Only race-specific observed link from official venue race-selection page.
 html,header_meta=official_fetch(JRA_O,race["jra_odds_cname"])
 soup=BeautifulSoup(html,"html.parser")
 links={}
 for label,bet in BET_LABELS.items():
  if bet=="WIN_PLACE":continue
  candidates={action_of(a,"/JRADB/accessO.html") for a in by_text(soup,label)}
  candidates.discard(None)
  if len(candidates)==1:links[bet]=candidates.pop()
  elif not candidates and bet=="BRACKET_QUINELLA" and n<=8:pass
  else:raise ValueError("JRA_BET_LINK_MISSING_OR_DUPLICATE:"+bet)
 pages={"WIN_PLACE":(html,header_meta,race["jra_odds_cname"])}
 for typ,cname in links.items():
  markup,meta=official_fetch(JRA_O,cname)
  pages[typ]=(markup,meta,cname)
 # Existing Fuji parsers remain unchanged: supply only the current race context.
 base_odds.N=n
 base_odds.RUNNERS={x["horse_number"]:x for x in runners}
 expected=theoretical(n,gates)
 seen={x:set() for x in expected}
 quotes=[];no_votes=[];scratched=[];page_meta={}
 update_times=set()
 for typ,(markup,meta,cname) in pages.items():
  dom=BeautifulSoup(markup,"html.parser")
  heads=clean(" ".join(h.get_text(" ",strip=True) for h in dom.select("h1,h2,h3,h4")))
  expected_venue=f'{race["meeting"]}回{race["venue"]}{race["meeting_day"]}日'
  if "2026年10月17日" not in heads or expected_venue not in heads or f'{race["race_no"]}レース' not in heads:
   raise ValueError("ODDS_PAGE_WRONG_RACE:"+race["race_id"]+":"+typ)
  displayed=re.search(r"(\d{1,2})時(\d{2})分現在",clean(dom.get_text(" ",strip=True)))
  if not displayed:raise ValueError("JRA_SOURCE_TIME_MISSING")
  update_times.add(displayed.group(0))
  page_meta[typ]={"cname":cname,"sha256":meta["sha256"],"http_date":meta.get("http_date"),
                  "official_update_display":displayed.group(0),"fetched_at_utc":utc()}
  partial=parse_matrix(dom,typ,n,gates,retired_numbers)
  for bet,values in partial.items():
   for combo,raw in values:
    if combo in seen[bet]:raise ValueError("ODDS_DUPLICATE_COMBINATION:"+bet)
    seen[bet].add(combo)
    item={"bet_type":bet,"combination":list(combo)}
    # Official '取消' / '除外' is not the same as '票数なし'.
    # For 枠連, a frame remains valid when another active horse is still in it.
    invalidated=(any(fr not in live_frames for fr in combo) if bet=="BRACKET_QUINELLA"
                 else any(no in retired_numbers for no in combo))
    if clean(raw) in ("取消","除外"):
     if not invalidated:raise ValueError("OFFICIAL_CANCELLATION_WITH_ACTIVE_SELECTION:"+bet+":"+str(combo))
     scratched.append({**item,"status":"SCRATCHED_OR_EXCLUDED","odds":None})
     continue
    if invalidated:raise ValueError("OFFICIAL_QUOTE_FOR_SCRATCHED_SELECTION:"+bet+":"+str(combo)+":"+str(raw))
    p=base_odds.parse_one_odd(raw,bet in ("PLACE","WIDE"))
    if p["status"]=="QUOTED":
     p.pop("status");quotes.append({**item,**p})
    else:no_votes.append({**item,**p})
 # Do not combine prices from mismatched publisher updates.
 if len(update_times)!=1:raise ValueError("MIXED_JRA_SOURCE_UPDATE_TIMES:"+json.dumps({k:v["official_update_display"] for k,v in page_meta.items()},ensure_ascii=False,sort_keys=True))
 absent=[]
 if n<=8 and "BRACKET_QUINELLA" not in links:
  absent=["BRACKET_QUINELLA"]
 for bet,exp in expected.items():
  if bet in absent:continue
  if seen[bet]!=exp:raise ValueError("COMBINATION_MISSING_OR_EXTRA:"+bet+":"+str(len(seen[bet]))+"/"+str(len(exp)))
 summary={bet:{"expected":len(combos),"listed":len(seen[bet]),
                "quoted":sum(q["bet_type"]==bet for q in quotes),
                "no_votes":sum(q["bet_type"]==bet for q in no_votes),
                "scratched":sum(q["bet_type"]==bet for q in scratched),
                "status":"NOT_OFFERED" if bet in absent else "COLLECTED"} for bet,combos in expected.items()}
 return {"source":"JRA_OFFICIAL_POST","snapshot_type":"PRE_RACE_LIVE_ODDS","publisher_update_display_consistent":True,
         "publisher_update_day_not_inferred":True,"page_observations":page_meta,
         "by_bet_type":summary,"expected_combination_count":sum(len(x) for k,x in expected.items() if k not in absent),
         "quoted_combination_count":len(quotes),"no_votes_combination_count":len(no_votes),
         "scratched_combination_count":len(scratched),
         "combination_coverage_complete":True,
         "quotes":quotes,"no_votes":no_votes,"scratched_combinations":scratched,
         "not_offered_bet_types":absent}
def atomic_official_odds(race,ncard,official):
 for attempt in range(3):
  try:return parse_official_odds(race,ncard,official)
  except ValueError as exc:
   if "MIXED_JRA_SOURCE_UPDATE_TIMES" not in str(exc) or attempt==2:raise
   time.sleep(2*(attempt+1))
 raise RuntimeError("ODDS_ATOMIC_RETRY_EXHAUSTED")

def reusable_verified(path,race):
 """Reuse only an already verified, explicitly PRE-RACE snapshot for the exact race."""
 if not path.exists():return None
 try:
  item=json.loads(path.read_text("utf-8"))
  odds=item.get("jra_odds",{})
  if item.get("contract")!="JRA_20261017_ALL_RACES_PRE_RACE_AUDIT_V1":return None
  if item.get("race_id")!=race["race_id"] or item.get("race_date")!=DAY:return None
  if item.get("source_discovery")!=race:return None
  if item.get("status")!="SOURCE_AND_ODDS_VERIFIED":return None
  if item.get("includes_results") is not False or item.get("includes_final_odds") is not False:return None
  if item.get("models_executed") is not False or item.get("backfill_history_features_computed") is not False:return None
  stamp=datetime.fromisoformat(item["captured_at_utc"])
  if stamp.tzinfo is None or stamp.astimezone(timezone.utc)>=datetime(2026,10,16,15,0,tzinfo=timezone.utc):return None
  if item.get("jra_card",{}).get("identity_verified") is not True:return None
  if odds.get("combination_coverage_complete") is not True or odds.get("publisher_update_display_consistent") is not True:return None
  if odds.get("snapshot_type")!="PRE_RACE_LIVE_ODDS":return None
  if not all(isinstance(odds.get(k),list) for k in ("quotes","no_votes","scratched_combinations")):return None
  if odds["expected_combination_count"]!=len(odds["quotes"])+len(odds["no_votes"])+len(odds["scratched_combinations"]):return None
  return item
 except (ValueError,KeyError,TypeError,OverflowError,AttributeError,UnicodeError):return None
def single(race):
 rid=race["race_id"]
 out={"contract":"JRA_20261017_ALL_RACES_PRE_RACE_AUDIT_V1",
      "race_id":rid,"race_date":DAY,"source_discovery":race,
      "captured_at_utc":utc(),"includes_results":False,"includes_final_odds":False,
      "models_executed":False,"backfill_history_features_computed":False,
      "status":"BLOCKED","errors":[]}
 try:
  ncard=net_card(race)
  out["netkeiba"]=ncard
  official=official_card(race,ncard)
  out["jra_card"]=official
  odds=atomic_official_odds(race,ncard,official)
  out["jra_odds"]=odds
  out["status"]="SOURCE_AND_ODDS_VERIFIED"
 except Exception as exc:
  out["errors"].append(type(exc).__name__+":"+str(exc))
  out["status"]="BLOCKED_SOURCE_OR_INTEGRITY"
 return out
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--out",type=Path,default=Path("audit-results/live-20261017"))
 p.add_argument("--limit",type=int,default=36)
 p.add_argument("--resume-verified",action="store_true",help="Keep already validated pre-race race snapshots; retry blocked races only")
 args=p.parse_args()
 if not 1<=args.limit<=36:raise ValueError("LIMIT")
 races,discovery=discover()
 output=args.out
 output.mkdir(parents=True,exist_ok=True)
 manifest={"contract":"JRA_20261017_ALL_RACES_SOURCE_MANIFEST_V1","day":DAY,
           "generated_at_utc":utc(),"race_count_expected":36,
           "discovery_verified":True,"actual_races_processed":min(args.limit,len(races)),
           "discovery_meta":discovery,"races":[]}
 for i,race in enumerate(races[:args.limit]):
  dest=output/(race["race_id"]+".json")
  cached=reusable_verified(dest,race) if args.resume_verified else None
  item=cached if cached is not None else single(race)
  if cached is None:dest.write_text(json.dumps(item,ensure_ascii=False,indent=2)+"\n","utf-8")
  rec={"race_id":race["race_id"],"venue":race["venue"],"race_no":race["race_no"],
       "status":item["status"],"error":item["errors"],"reused_verified_snapshot":cached is not None,
       "runner_count":item.get("netkeiba",{}).get("runner_count"),
       "odds_quote_count":item.get("jra_odds",{}).get("quoted_combination_count"),
       "no_votes":item.get("jra_odds",{}).get("no_votes_combination_count"),
       "scratched":item.get("jra_odds",{}).get("scratched_combination_count"),
       "file":str(dest)}
  manifest["races"].append(rec)
  print("ALL_RACES_PROGRESS "+json.dumps(rec,ensure_ascii=False),flush=True)
  if i+1<len(races[:args.limit]):time.sleep(0.7)
 manifest["reused_verified_count"]=sum(x["reused_verified_snapshot"] for x in manifest["races"])
 manifest["full_source_and_odds_verified"]=sum(x["status"]=="SOURCE_AND_ODDS_VERIFIED" for x in manifest["races"])
 manifest["blocked"]=sum(x["status"]!="SOURCE_AND_ODDS_VERIFIED" for x in manifest["races"])
 (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n","utf-8")
 print("ALL_RACES_MANIFEST "+json.dumps({"expected":36,"processed":manifest["actual_races_processed"],
  "verified":manifest["full_source_and_odds_verified"],"blocked":manifest["blocked"]},ensure_ascii=False),flush=True)
 if args.limit==36 and manifest["blocked"]:return 2
 return 0
if __name__=="__main__":
 try:sys.exit(main())
 except Exception as e:
  print("ALL_RACES_COLLECTOR_FATAL "+type(e).__name__+":"+str(e),flush=True);sys.exit(3)
