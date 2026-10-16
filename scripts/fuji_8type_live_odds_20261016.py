#!/usr/bin/env python3
"""Official real JRA 8-ticket Fuji pre-race odds -> 14-runner race JSON.
Reads all 7 official bet-form pages (WIN+PLACE share one).
- no guessed CNAME; follow JRA live card's race-specific POST & form links
- no results/final odds, no forecasts, no betting, no modifications of frozen L1/L2/L3
- NO_VOTES is explicitly unavailable, never a positive estimated number
- validates exactly the combinatorial set of listed selections; rejects missing/dupes
- quotes from different JRA refreshes retain per-page fetch timestamp and page update text
"""
import argparse,json,re,itertools,sys,unicodedata
from pathlib import Path
from datetime import datetime,timezone
from bs4 import BeautifulSoup
from fuji_onsale_jra_deep_probe_20261016 import fetch,CARD,ODDS
from fuji_prebet_phase4_odds_validator_20261016 import validate_quote_batch
RACE="202605040611";DATE="2026-10-17";N=14
TYPES={"単勝・複勝":"WIN_PLACE","枠連":"BRACKET_QUINELLA","馬連":"QUINELLA",
       "ワイド":"WIDE","馬単":"EXACTA","3連複":"TRIO","3連単":"TRIFECTA"}
ACTION=re.compile(r"doAction\(\s*['\"](/JRADB/accessO\.html)['\"]\s*,\s*['\"]([^'\"]+)['\"]")
POSITIVE=re.compile(r"(?:\d+)(?:\.\d+)?")
RANGE=re.compile(r"(\d+(?:\.\d+)?)\s*[-－ー〜~]\s*(\d+(?:\.\d+)?)")
BASE=json.loads(Path("audit-results/fuji-20261017-149-asof-20261016.json").read_text("utf-8"))
RUNNERS={r["horse_number"]:r for r in BASE["runners"]}
if len(RUNNERS)!=N or BASE["race_id"]!=RACE:raise ValueError("BASELINE_14_RUNNER_CONTRACT_FAILURE")
def clean(t):return re.sub(r"\s+"," ",unicodedata.normalize("NFKC",t or "")).strip()
def obs():return datetime.now(timezone.utc).isoformat()
def actions(soup):
 arr=[]
 for a in soup.select("a[onclick]"):
  m=ACTION.search(a["onclick"])
  if m:arr.append((clean(a.get_text(" ",strip=True)),m.group(2)))
 return arr
def expected():
 frame={i:0 for i in range(1,9)}
 for h in RUNNERS.values():frame[h["gate"]]+=1
 bracket=[(a,b) for a in range(1,9) for b in range(a,9) if a!=b or frame[a]>=2]
 horses=list(range(1,N+1))
 return {
  "WIN":{(i,) for i in horses},"PLACE":{(i,) for i in horses},
  "BRACKET_QUINELLA":set(bracket),
  "QUINELLA":set(itertools.combinations(horses,2)),
  "WIDE":set(itertools.combinations(horses,2)),
  "EXACTA":set(itertools.permutations(horses,2)),
  "TRIO":set(itertools.combinations(horses,3)),
  "TRIFECTA":set(itertools.permutations(horses,3))}
def parse_one_odd(value,ranged=False):
 text=clean(value)
 if "票数なし" in text or text in ("－","-","発売なし","発売前"):
  return {"status":"NO_VOTES","odds":None} if not ranged else {"status":"NO_VOTES","odds_min":None,"odds_max":None}
 if ranged:
  m=RANGE.fullmatch(text)
  if not m:raise ValueError("NOT_NUMERIC_JRA_ODDS_RANGE:"+text[:80])
  low,high=map(float,m.groups())
  if not (0<low<=high):raise ValueError("INVALID_RANGE")
  return {"status":"QUOTED","odds_min":low,"odds_max":high}
 if not POSITIVE.fullmatch(text):raise ValueError("NOT_NUMERIC_JRA_ODDS:"+text[:80])
 value=float(text)
 if value<=0:raise ValueError("INVALID_ODDS")
 return {"status":"QUOTED","odds":value}
def parse_winplace(soup):
 tab=soup.select("table.tanpuku")
 if len(tab)!=1:raise ValueError("WIN_PLACE_TABLE_MISSING")
 seen=[]
 for tr in tab[0].select("tr"):
  no=tr.select_one("td.num");name=tr.select_one("td.horse")
  if not no:continue
  i=int(clean(no.get_text(" ",strip=True)))
  if i not in RUNNERS:raise ValueError("UNKNOWN_HORSE")
  nm=clean(name.get_text(" ",strip=True)) if name else ""
  if nm!=clean(RUNNERS[i]["horse_name"]):raise ValueError("HORSE_NAME_MISMATCH:"+str(i)+":"+nm)
  w=tr.select_one("td.odds_tan")
  p=tr.select_one("td.odds_fuku")
  if w is None or p is None:raise ValueError("WIN_PLACE_QUOTE_CELL_ABSENT")
  seen.append((i,clean(w.get_text(" ",strip=True)),clean(p.get_text(" ",strip=True))))
 if len(seen)!=N or {x[0] for x in seen}!=set(range(1,N+1)):raise ValueError("INVALID_WIN_PLACE_RUNNER_SET")
 return {"WIN":[((i,),w) for i,w,_ in seen],"PLACE":[((i,),p) for i,_,p in seen]}
def parse_bet(soup,typ):
 if typ=="WIN_PLACE":return parse_winplace(soup)
 classes={"BRACKET_QUINELLA":"waku","QUINELLA":"umaren","WIDE":"wide",
          "EXACTA":"umatan","TRIO":"fuku3","TRIFECTA":"tan3"}
 tabs=soup.select("table."+classes[typ])
 expected_tables={"BRACKET_QUINELLA":8,"QUINELLA":13,"WIDE":13,"EXACTA":14,"TRIO":78,"TRIFECTA":182}
 if len(tabs)!=expected_tables[typ]:raise ValueError("JRA_TABLE_COUNT_MISMATCH:"+typ+":"+str(len(tabs)))
 result=[]
 for index,tab in enumerate(tabs):
  rows=tab.select("tr")
  if typ=="BRACKET_QUINELLA":
   a=index+1
   expected_second=list(range(a,9))
   prefix=(a,)
  elif typ in ("QUINELLA","WIDE"):
   a=index+1
   expected_second=list(range(a+1,N+1))
   prefix=(a,)
  elif typ=="EXACTA":
   a=index+1
   expected_second=list(range(1,N+1))
   prefix=(a,)
  elif typ=="TRIO":
   a,c=list(itertools.combinations(range(1,N),2))[index]
   expected_second=list(range(c+1,N+1))
   prefix=(a,c)
  else:
   a,c=list(itertools.permutations(range(1,N+1),2))[index]
   expected_second=[x for x in range(1,N+1) if x!=a]
   prefix=(a,c)
  if len(rows)!=len(expected_second):raise ValueError("JRA_ROW_COUNT_MISMATCH:"+typ+":"+str(index))
  for row,b in zip(rows,expected_second):
   cells=row.find_all(["th","td"],recursive=False)
   if len(cells)!=2:raise ValueError("MATRIX_ROW_COL_COUNT:"+typ)
   number=clean(cells[0].get_text(" ",strip=True))
   if number!=str(b):raise ValueError("MATRIX_ROW_NUMBER_ORDER:"+typ+":"+str(prefix)+":"+number+"/"+str(b))
   odds=clean(cells[1].get_text(" ",strip=True))
   combination=prefix+(b,)
   # JRA includes intentional blank diagonal (same horse). Only that diagonal may be empty.
   invalid=(typ=="EXACTA" and a==b) or (typ=="TRIFECTA" and c==b)
   if invalid:
    if odds not in ("","-","－"):raise ValueError("NONEMPTY_IMPOSSIBLE_HORSE_REPEAT")
    continue
   if typ=="BRACKET_QUINELLA" and a==b and sum(h["gate"]==a for h in RUNNERS.values())<2:
    if odds not in ("","-","－"):raise ValueError("NONEMPTY_IMPOSSIBLE_FRAME_REPEAT")
    continue
   if odds in ("","-","－"):raise ValueError("UNEXPLAINED_EMPTY_JRA_ODDS:"+typ+":"+str(combination))
   result.append((combination,odds))
 return {typ:result}
def verify_page(soup,typ):
 heads=" ".join(clean(x.get_text(" ",strip=True)) for x in soup.select("h1,h2,h3,h4"))
 if not all(x in heads for x in ("富士","2026年10月17日","4回東京6日","11レース")):
  raise ValueError("OFFICIAL_RACE_HEADER_MISMATCH:"+typ)
 page=clean(soup.get_text(" ",strip=True))
 m=re.search(r"(\d{1,2})時(\d{2})分現在",page)
 if not m:raise ValueError("PUBLISHER_UPDATE_DISPLAY_ABSENT:"+typ)
 hour,minute=int(m.group(1)),int(m.group(2))
 if hour>23 or minute>59:raise ValueError("PUBLISHER_UPDATE_CLOCK_INVALID")
 return m.group(0)
def extract_all():
 card,cardinfo=fetch(CARD)
 targets=sorted(set(code for label,code in actions(BeautifulSoup(card,"html.parser")) if "202604061120261017" in code))
 if len(targets)!=1:raise ValueError("FUJI_RACE_SPECIFIC_CARD_LINK_NOT_UNIQUE")
 base,basemeta=fetch(ODDS,targets[0])
 soup=BeautifulSoup(base,"html.parser")
 verify_page(soup,"WIN_PLACE")
 navigation=actions(soup)
 picks={"WIN_PLACE":targets[0]}
 for label,t in TYPES.items():
  if t=="WIN_PLACE":continue
  links={code for name,code in navigation if name==label}
  if len(links)!=1:raise ValueError("OFFICIAL_BET_TYPE_LINK_NOT_UNIQUE:"+t)
  picks[t]=links.pop()
 expected_keys=expected()
 pages={}; quotes=[]; unavailable=[]; seen={t:set() for t in expected_keys}
 for typ,cname in picks.items():
  if typ=="WIN_PLACE":html,meta=base,basemeta
  else:html,meta=fetch(ODDS,cname)
  fetched_at=obs()
  dom=BeautifulSoup(html,"html.parser")
  publisher_time=verify_page(dom,typ)
  raw_bets=parse_bet(dom,typ)
  pages[typ]={"url":ODDS,"cname":cname,"sha256":meta["sha256"],
              "http_date":meta.get("http_date"),"fetched_at_utc":fetched_at,
              "official_update_display":publisher_time,
              "publisher_update_day_not_inferred":True}
  for bet,records in raw_bets.items():
   for combo,value in records:
    if combo in seen[bet]:raise ValueError("DUPLICATE_JRA_COMBINATION:"+bet)
    seen[bet].add(combo)
    parsed=parse_one_odd(value,bet in ("PLACE","WIDE"))
    record={"bet_type":bet,"combination":list(combo),**parsed}
    if parsed["status"]=="QUOTED":
     record.pop("status")
     quotes.append(record)
    else:unavailable.append(record)
 # An atomic market snapshot is not claimed if the seven official pages
 # displayed different publisher refresh minutes. Retry later rather than mix.
 displayed_times={v["official_update_display"] for v in pages.values()}
 if len(displayed_times)!=1:raise ValueError("MIXED_JRA_PUBLISHER_UPDATE_TIMES:"+",".join(sorted(displayed_times)))
 summary={}
 for bet,expect in expected_keys.items():
  missing=expect-seen[bet];extra=seen[bet]-expect
  if missing or extra:raise ValueError("COMBINATION_COVERAGE_INVALID:"+bet+":missing="+str(len(missing))+":extra="+str(len(extra)))
  count=sum(q["bet_type"]==bet for q in quotes)
  unavailable_count=sum(q["bet_type"]==bet for q in unavailable)
  summary[bet]={"expected":len(expect),"listed":len(seen[bet]),"quoted":count,"no_votes":unavailable_count}
  if count+unavailable_count!=len(expect):raise ValueError("QUOTE_COVERAGE_SUM_INVALID")
 batch={"race_id":RACE,"source_url":ODDS,"race_header_verified":True,
        "snapshot_type":"PRE_RACE_LIVE_ODDS","source_error":None,
        "fetched_at_utc":obs(),"page_updated_at":None,"quotes":quotes}
 result=validate_quote_batch(batch)
 if result["records"]!=len(quotes):raise ValueError("PREEXISTING_VALIDATOR_COUNT_DRIFT")
 return {"contract":"FUJI_JRA_8TYPE_LIVE_ODDS_V1","race_id":RACE,"race_date":DATE,
         "source":"JRA_OFFICIAL_POST","snapshot_type":"PRE_RACE_LIVE_ODDS",
         "captured_at_utc":obs(),"source_card_sha256":cardinfo["sha256"],
         "page_observations":pages,"by_bet_type":summary,
         "publisher_update_display_consistent":True,
         "expected_combination_count":sum(x["expected"] for x in summary.values()),
         "quoted_combination_count":len(quotes),"no_votes_combination_count":len(unavailable),
         "combination_coverage_complete":True,"all_quotes_numerically_available":not unavailable,
         "status":"COMPLETE_COMBINATIONS_WITH_NO_VOTES" if unavailable else "COMPLETE_ALL_ODDS_AVAILABLE",
         "quotes":quotes,"no_votes":unavailable,
         "audit_note":"Official displayed update time has no verified calendar day. Per-page client fetch and SHA-256 retained. No final odds."}
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("audit-results/fuji-20261017-8type-live-odds.json"))
 p.add_argument("--summary-only",action="store_true")
 args=p.parse_args()
 d=extract_all()
 if not args.summary_only:
  args.output.parent.mkdir(parents=True,exist_ok=True)
  args.output.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print("FUJI_8TYPE_SUMMARY "+json.dumps({k:v for k,v in d.items() if k not in ("quotes","no_votes","page_observations")},ensure_ascii=False,separators=(",",":")))
 print("FUJI_8TYPE_PAGE_TIMES "+json.dumps({t:{"official_update_display":v["official_update_display"],"fetched_at_utc":v["fetched_at_utc"],"sha256":v["sha256"]} for t,v in d["page_observations"].items()},ensure_ascii=False,separators=(",",":")))
 if not args.summary_only:print("FUJI_8TYPE_JSON_WRITTEN "+str(args.output))
if __name__=="__main__":
 try:main()
 except Exception as e:
  print("FUJI_8TYPE_BLOCKED "+json.dumps({"status":"BLOCKED_SOURCE_UNVERIFIED","reason":type(e).__name__+":"+str(e)},ensure_ascii=False));sys.exit(2)
