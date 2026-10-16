#!/usr/bin/env python3
"""Read-only P1: BACKFILL horse IDs from netkeiba pre-race card vs official JRA numbering.
No ML, no betting, no netkeiba odds. Prices are explicitly NOT validated by this probe.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup

VENUES={"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京",
        "06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}
NET_HOSTS={"race.netkeiba.com","race.sp.netkeiba.com"}
JRA_HOSTS={"www.jra.go.jp","sp.jra.jp"}

def clean(s):
    return re.sub(r"[\s\u3000]+","",str(s or "")).strip()

def decode_html(data, content_type):
    match=re.search(r"charset\s*=\s*['\"]?([\w-]+)",content_type,re.I)
    for encoding in dict.fromkeys(([match.group(1)] if match else [])+["utf-8","euc-jp","cp932"]):
        try: return data.decode(encoding)
        except (UnicodeDecodeError,LookupError): pass
    raise ValueError("SOURCE_ENCODING_UNSUPPORTED")

def fetch_public_html(url,hosts):
    u=urlparse(url)
    if u.scheme!="https" or u.hostname not in hosts: raise ValueError("SOURCE_HOST_NOT_ALLOWED")
    last="UNKNOWN_FETCH_ERROR"
    for attempt in range(3):
        if attempt: time.sleep(1.0*(2**attempt))
        try:
            req=Request(url,headers={"User-Agent":"KEIBA-LIVE-PREBET-PROBE/1.0","Accept-Language":"ja"})
            with urlopen(req,timeout=18) as res:
                if urlparse(res.geturl()).hostname not in hosts: raise ValueError("SOURCE_REDIRECT_HOST_NOT_ALLOWED")
                data=res.read(5_000_001)
                if len(data)>5_000_000: raise ValueError("SOURCE_BODY_TOO_LARGE")
                html=decode_html(data,res.headers.get("content-type",""))
                if re.search(r"アクセス制限|不正なアクセス|通信制限",html):
                    raise ValueError("SOURCE_BLOCKED")
                return html
        except HTTPError as exc:
            if exc.code not in {502,503,504}: raise ValueError("SOURCE_HTTP_"+str(exc.code)) from exc
            last="SOURCE_HTTP_"+str(exc.code)
        except (URLError,TimeoutError) as exc: last="SOURCE_NETWORK_TEMPORARY:"+str(exc)
    raise ValueError(last)

def official_jra_identity(url):
    u=urlparse(url)
    if u.scheme!="https" or u.hostname not in JRA_HOSTS or u.path!="/JRADB/accessD.html":
        raise ValueError("JRA_OFFICIAL_CARD_URL_REQUIRED")
    cname=parse_qs(u.query).get("CNAME",[""])[0].split("/")[0]
    m=re.fullmatch(r"pw01dde\d{2}(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})",cname)
    if not m or m.group(1) not in VENUES: raise ValueError("JRA_CARD_IDENTITY_INVALID")
    venue,year,meeting,day,rno,date=m.groups()
    datetime.strptime(date,"%Y%m%d")
    if date[:4]!=year or not 1<=int(rno)<=12 or int(day)<1 or int(meeting)<1:
        raise ValueError("JRA_CARD_IDENTITY_INCONSISTENT")
    return {"race_id":year+venue+meeting+day+rno,"venue":VENUES[venue],
            "race_date":date[:4]+"-"+date[4:6]+"-"+date[6:8],
            "meeting_no":int(meeting),"meeting_day":int(day),"race_no":int(rno)}

def netkeiba_race_id(url):
    u=urlparse(url)
    if u.scheme!="https" or u.hostname not in NET_HOSTS or u.path!="/race/shutuba.html":
        raise ValueError("NETKEIBA_SHUTUBA_URL_REQUIRED")
    rid=parse_qs(u.query).get("race_id",[""])[0]
    if not re.fullmatch(r"\d{12}",rid): raise ValueError("NETKEIBA_RACE_ID_INVALID")
    return rid

def cell_number(cell):
    if cell is None: return None
    value=clean(cell.get_text(" ",strip=True))
    m=re.fullmatch(r"\D*(\d{1,2})(?:番)?",value)
    n=int(m.group(1)) if m else 0
    return n if 1<=n<=18 else None

def parse_netkeiba_entries(html):
    soup=BeautifulSoup(html,"html.parser")
    tables=[x for x in soup.find_all("table") if any(
        "shutuba" in str(t).lower() or "racecard" in str(t).lower()
        for t in list(x.get("class",[]))+[x.get("id","")])]
    if not tables: raise ValueError("NETKEIBA_RUNNER_TABLE_NOT_FOUND")
    rows=[]
    for table in tables:
        for tr in table.find_all("tr"):
            horse_cell=tr.select_one("td.HorseInfo, td[class*='HorseInfo']")
            if horse_cell is None: horse_cell=tr
            links=[]
            for a in horse_cell.find_all("a",href=True):
                m=re.search(r"(?:^|/)horse/([0-9A-Za-z]+)(?:/|[?#]|$)",a["href"])
                if m and clean(a.get_text(" ",strip=True)):
                    links.append((m.group(1),clean(a.get_text(" ",strip=True))))
            ids={item[0] for item in links}
            if not ids: continue
            if len(ids)!=1: raise ValueError("NETKEIBA_MULTIPLE_HORSE_IDS_PER_ROW")
            hid=next(iter(ids))
            name=next(n for i,n in links if i==hid)
            rows.append({"horse_id":hid,"horse_name":name,
                         "netkeiba_horse_no":cell_number(tr.select_one("td.Umaban, td[class*='Umaban']")),
                         "netkeiba_gate":cell_number(tr.select_one("td.Waku, td[class*='Waku']"))})
    if not rows: raise ValueError("NETKEIBA_ENTRIES_NOT_FOUND")
    if len({r["horse_id"] for r in rows})!=len(rows): raise ValueError("NETKEIBA_HORSE_ID_DUPLICATE")
    if len({r["horse_name"] for r in rows})!=len(rows): raise ValueError("NETKEIBA_HORSE_NAME_DUPLICATE")
    known=[r["netkeiba_horse_no"] for r in rows if r["netkeiba_horse_no"] is not None]
    if len(set(known))!=len(known): raise ValueError("NETKEIBA_HORSE_NO_DUPLICATE")
    return rows

def verify_jra_card_header(html,identity):
    txt=clean(BeautifulSoup(html,"html.parser").get_text(" ",strip=True))
    body=txt.split("ここから本文です",1)[-1][:900]
    yyyy,mm,dd=map(int,identity["race_date"].split("-"))
    needed=[f"{yyyy}年{mm}月{dd}日",
            f"{identity['meeting_no']}回{identity['venue']}{identity['meeting_day']}日",
            f"{identity['race_no']}レース"]
    if not all(v in body for v in needed):
        raise ValueError("JRA_OFFICIAL_HEADER_IDENTITY_MISMATCH")

def parse_official_jra_entries(html,names):
    soup=BeautifulSoup(html,"html.parser")
    rows=[]
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells=tr.find_all(["th","td"],recursive=False)
            if len(cells)<3: continue
            horse_text=clean(cells[2].get_text(" ",strip=True))
            found=[name for name in names if name in horse_text]
            if len(found)!=1: continue
            no=cell_number(cells[1])
            gate=cell_number(cells[0])
            if gate is None:
                for img in cells[0].find_all("img"):
                    m=re.search(r"枠([1-8])",clean(img.get("alt","")))
                    if m: gate=int(m.group(1)); break
            rows.append({"horse_name":found[0],"horse_no":no,"gate":gate})
    if not rows: raise ValueError("JRA_RUNNER_ROWS_MISSING")
    if len({r["horse_name"] for r in rows})!=len(rows):
        raise ValueError("JRA_RUNNER_ROW_AMBIGUOUS")
    return rows

def validate_crosswalk(race_id,jra_url,net_rows,jra_rows,expected=None):
    ident=official_jra_identity(jra_url)
    if race_id!=ident["race_id"]: raise ValueError("JRA_NETKEIBA_RACE_ID_MISMATCH")
    if expected is not None and (len(net_rows)!=expected or len(jra_rows)!=expected):
        raise ValueError("EXPECTED_RUNNER_COUNT_MISMATCH")
    by_name={r["horse_name"]:r for r in jra_rows}
    if {r["horse_name"] for r in net_rows}!=set(by_name):
        raise ValueError("JRA_NETKEIBA_RUNNER_SET_MISMATCH")
    if any(r["horse_no"] is None or r["gate"] is None for r in jra_rows):
        return {"state":"WAIT_OFFICIAL_NUMBERING","identity":ident,"number_of_horses":len(net_rows),"entries":[]}
    numbers=[r["horse_no"] for r in jra_rows]
    if len(set(numbers))!=len(numbers) or sorted(numbers)!=list(range(1,len(jra_rows)+1)):
        raise ValueError("JRA_HORSE_NO_INVALID_OR_MISSING")
    if any(not 1<=r["gate"]<=8 for r in jra_rows): raise ValueError("JRA_GATE_INVALID")
    entries=[]
    for net in net_rows:
        jra=by_name[net["horse_name"]]
        if (net["netkeiba_horse_no"] is not None and net["netkeiba_horse_no"]!=jra["horse_no"]) or (
            net["netkeiba_gate"] is not None and net["netkeiba_gate"]!=jra["gate"]):
            raise ValueError("NETKEIBA_JRA_NUMBERING_CONFLICT")
        entries.append({"horse_id":net["horse_id"],"horse_name":net["horse_name"],
                        "horse_no":jra["horse_no"],"gate":jra["gate"]})
    return {"state":"ID_CROSSWALK_VALIDATED","identity":ident,
            "number_of_horses":len(entries),"entries":sorted(entries,key=lambda x:x["horse_no"])}

def detect_jra_odds_navigation(html):
    soup=BeautifulSoup(html,"html.parser")
    options=[]
    for a in soup.select("a[onclick]"):
        if clean(a.get_text(" ",strip=True))!="オッズ": continue
        m=re.search(r"doAction\(\s*['\"]([^'\"]*accessO\.html)['\"]\s*,\s*['\"]([^'\"]+)['\"]",a["onclick"])
        if m: options.append((m.group(1),m.group(2)))
    return {"state":"OFFICIAL_ODDS_NAVIGATION_DISCOVERED" if len(options)==1 else "OFFICIAL_ODDS_NAVIGATION_UNVERIFIED",
            "option_count":len(options),"prices_collected":False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--race-id",required=True)
    p.add_argument("--jra-card-url",required=True)
    p.add_argument("--expected",type=int,required=True,help="Official field size 1-18; mandatory")
    p.add_argument("--netkeiba-html",type=Path,help="Offline fixture instead of HTTP")
    p.add_argument("--jra-html",type=Path,help="Offline fixture instead of HTTP")
    p.add_argument("--output",type=Path,help="Runner-local JSON only; no artifact upload")
    a=p.parse_args()
    url=f"https://race.netkeiba.com/race/shutuba.html?race_id={a.race_id}"
    result={"schema_version":1,"race_id":a.race_id,"netkeiba_url":url,
            "jra_card_url":a.jra_card_url,"checked_at":datetime.now(timezone.utc).isoformat(),
            "prebet_odds_complete":False,"source_kind":"PRE_RACE_PUBLIC_HTML",
            "model_inference":False,"betting":False}
    try:
        if netkeiba_race_id(url)!=a.race_id or not 1<=a.expected<=18:
            raise ValueError("INVALID_INPUT")
        ident=official_jra_identity(a.jra_card_url)
        if ident["race_id"]!=a.race_id: raise ValueError("JRA_NETKEIBA_RACE_ID_MISMATCH")
        n=a.netkeiba_html.read_text(encoding="utf-8") if a.netkeiba_html else fetch_public_html(url,NET_HOSTS)
        j=a.jra_html.read_text(encoding="utf-8") if a.jra_html else fetch_public_html(a.jra_card_url,JRA_HOSTS)
        verify_jra_card_header(j,ident)
        net=parse_netkeiba_entries(n)
        official=parse_official_jra_entries(j,{r["horse_name"] for r in net})
        result["crosswalk"]=validate_crosswalk(a.race_id,a.jra_card_url,net,official,a.expected)
        result["jra_odds"]=detect_jra_odds_navigation(j)
        result["status"]=result["crosswalk"]["state"]
    except Exception as exc:
        result["status"]="BLOCKED"
        result["error"]=str(exc)
    serialized=json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n"
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(serialized,encoding="utf-8")
    print(serialized,end="")
    return 0 if result["status"]=="ID_CROSSWALK_VALIDATED" else 2

if __name__=="__main__": sys.exit(main())
