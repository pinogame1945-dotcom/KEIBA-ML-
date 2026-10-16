#!/usr/bin/env python3
"""Read-only parity audit: historical netkeiba shutuba vs KEIBA-BACKFILL race pack.
Historical shutuba is a post-event live page NOW, not a stored pre-race snapshot.
Never report its current values as observed before the race. No odds collected.
"""
import gzip, json, re, sys, time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse

DATE="2025-10-18"
IDS=("202505040611","202505040605") # Fuji Stakes 14, Tokyo 5R maiden debut 16
PACK="https://raw.githubusercontent.com/pinogame1945-dotcom/KEIBA-BACKFILL/main/data/daily/2025-10-18.jsonl.gz"
NET="https://race.netkeiba.com/race/shutuba.html?race_id="
HORSE_RE=re.compile(r"/horse/([0-9A-Za-z]+)(?:/|$|[?#])")
ID_RE={"jockey":re.compile(r"/jockey/(?:result/recent/)?(\d+)"),
       "trainer":re.compile(r"/trainer/(?:result/recent/)?(\d+)")}

class FetchError(Exception): pass

def fetch(url,binary=False):
    host=urlparse(url).hostname
    if host not in ("raw.githubusercontent.com","race.netkeiba.com") or not url.startswith("https://"):
        raise FetchError("HOST_NOT_ALLOWED")
    for attempt in range(3):
        if attempt: time.sleep(2**attempt)
        try:
            req=Request(url,headers={"User-Agent":"KEIBA-MODEL-COMPATIBILITY-AUDIT/1.0",
                                     "Accept-Language":"ja,en;q=0.5"})
            with urlopen(req,timeout=25) as response:
                if urlparse(response.geturl()).hostname!=host: raise FetchError("REDIRECT_DENIED")
                raw=response.read(15000000)
                if len(raw)>=15000000: raise FetchError("RESPONSE_TOO_LARGE")
                if binary: return raw
                enc=re.search(r"charset=([\w-]+)",response.headers.get("content-type",""),re.I)
                for codec in ([enc.group(1)] if enc else [])+["utf-8","euc-jp","cp932"]:
                    try: return raw.decode(codec)
                    except (UnicodeError, LookupError): continue
                raise FetchError("UNDECODABLE_HTML")
        except HTTPError as e:
            if e.code not in (502,503,504): raise FetchError("HTTP_"+str(e.code))
            last="HTTP_"+str(e.code)
        except (URLError,TimeoutError) as e: last="NETWORK_"+type(e).__name__
    raise FetchError(last)

def norm(x): return re.sub(r"\s+","",str(x or "")).replace("\u3000","").strip()
def num(x):
    text=norm(x)
    m=re.fullmatch(r"\d+(?:\.\d+)?",text)
    return float(text) if m else None

def extract_id(node,kind):
    for a in node.select("a[href]"):
        match=ID_RE[kind].search(a.get("href",""))
        if match: return match.group(1)
    return None

def parse_netkeiba(html):
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(html,"html.parser")
    table=soup.select_one("table.Shutuba_Table")
    if not table: raise ValueError("NO_OFFICIAL_SHUTUBA_TABLE")
    rows=[]
    for tr in table.select("tr"):
        horse=tr.select_one("td.HorseInfo")
        if horse is None: continue
        links=[]
        for a in horse.select("a[href]"):
            found=HORSE_RE.search(a.get("href",""))
            if found: links.append((found.group(1),norm(a.get_text(" ",strip=True))))
        links=list(dict.fromkeys(links))
        if not links: continue
        if len({id for id,name in links})!=1: raise ValueError("AMBIGUOUS_HORSE_ID")
        hid=links[0][0]
        name=next((name for id,name in links if name),None)
        def field(css):
            el=tr.select_one(css)
            return norm(el.get_text(" ",strip=True)) if el else None
        def intval(css):
            txt=field(css)
            return int(txt) if txt and txt.isdigit() else None
        sexage=field("td.Barei")
        match=re.search(r"(牡|牝|セ)(\d+)",sexage or "")
        rows.append({"horse_id":hid,"horse_name":name,
                     "horse_number":intval("td.Umaban"),"gate":intval("td.Waku"),
                     "sex":match.group(1) if match else None,
                     "age":int(match.group(2)) if match else None,
                     "carried_weight":num(field("td.Kinryo")),
                     "jockey_id":extract_id(tr.select_one("td.Jockey") or tr,"jockey"),
                     "trainer_id":extract_id(tr.select_one("td.Trainer") or tr,"trainer")})
    if not rows: raise ValueError("NO_HORSE_ID_IN_CARD")
    ids=[r["horse_id"] for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError("DUPLICATE_HORSE_ID")
    header=soup.select_one(".RaceData02")
    title=soup.select_one(".RaceName")
    race_info=soup.select_one(".RaceData01")
    return rows,{
        "condition":norm(header.get_text(" ",strip=True)) if header else "",
        "race_name":norm(title.get_text(" ",strip=True)) if title else "",
        "course":norm(race_info.get_text(" ",strip=True)) if race_info else "",
        "full_header":norm((soup.select_one(".RaceList_NameBox") or soup).get_text(" ",strip=True))[:700]}

def compare(base,public):
    pairs=[("horse_name","horse_name"),("horse_number","horse_number"),("gate","gate"),
           ("sex","sex"),("age","age"),("carried_weight","carried_weight"),
           ("jockey_id","jockey_id"),("trainer_id","trainer_id")]
    diffs=[]
    by_id={x["horse_id"]:x for x in base}
    net_by_id={x["horse_id"]:x for x in public}
    missing_in_public=sorted(set(by_id)-set(net_by_id))
    missing_in_backfill=sorted(set(net_by_id)-set(by_id))
    for hid in sorted(set(by_id)&set(net_by_id)):
        a,b=by_id[hid],net_by_id[hid]
        for stored,live in pairs:
            x,y=a.get(stored),b.get(live)
            # No guessing / missing values are not silently counted as matches.
            if x is None or y is None:
                diffs.append({"horse_id":hid,"field":stored,"type":"MISSING","backfill":x,"card":y})
            elif str(x)!=str(y) and not (
                stored=="carried_weight" and float(x)==float(y)):
                diffs.append({"horse_id":hid,"field":stored,"type":"DIFFERENT","backfill":x,"card":y})
    return {"backfill_entries":len(base),"public_card_entries":len(public),
            "missing_in_public":missing_in_public,"missing_in_backfill":missing_in_backfill,
            "field_differences":diffs,
            "checks":len(set(by_id)&set(net_by_id))*len(pairs),
            "strict_value_match":not(diffs or missing_in_public or missing_in_backfill)}

def main():
    summary={"contract":"KEIBA_PREBET_BACKFILL_PARITY_AUDIT_V1","audit_fetched_at":datetime.now(timezone.utc).isoformat(),
             "historical_card_is_post_event_mutable":True,
             "pre_race_timestamped_snapshot_available":False,
             "no_model_changes":True,
             "netkeiba_odds_used":False,"races":[]}
    try:
        blob=fetch(PACK,binary=True)
        text=gzip.decompress(blob).decode("utf-8")
        by_race={str(row["race"]["race_id"]):row for row in map(json.loads,filter(None,text.splitlines()))}
        for idx,race_id in enumerate(IDS):
            if idx: time.sleep(2)
            if race_id not in by_race: raise ValueError("BACKFILL_RACE_ABSENT_"+race_id)
            row=by_race[race_id]
            html=fetch(NET+race_id)
            public,meta=parse_netkeiba(html)
            comp=compare(row["entries"],public)
            r=row["race"]
            info={"race_id":race_id,"backfill_race_name":r.get("race_name"),
                  "backfill_class":r.get("race_class_normalized"),
                  "backfill_grade":r.get("grade"),
                  "backfill_age_min":r.get("age_min"),"backfill_weight_rule":r.get("weight_rule"),
                  "backfill_course_layout":r.get("course_layout"),
                  "backfill_course_laps":r.get("course_laps"),
                  "backfill_designated":r.get("designated"),
                  "backfill_international":r.get("international"),
                  "backfill_mixed":r.get("mixed"),
                  "backfill_source_condition":r.get("race_condition_raw"),
                  "public_header":meta,
                  "comparison":comp,
                  "strict_match":comp["strict_value_match"]}
            if race_id=="202505040605":
                info["new_horse_note"]="NEWCOMER: no prior starts is expected; past-record availability must be checked separately"
            if "(指)" in meta["condition"] and "[指]" not in meta["condition"]:
                info["designated_parser_issue"]="BACKFILL race-meta.mjs recognizes [指] but not (指)"
            summary["races"].append(info)
        summary["status"]="COMPLETED_WITH_WARNINGS" if any(
           not item["strict_match"] or item.get("designated_parser_issue") for item in summary["races"]
           ) else "PARITY_OK_HISTORICAL_PRESENT_VIEW_ONLY"
    except Exception as e:
        summary["status"]="BLOCKED";summary["blocked_reason"]=type(e).__name__+":"+str(e)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    # If network or format gate fails, do not claim parity.
    return 2 if summary["status"]=="BLOCKED" else 0

if __name__=="__main__":sys.exit(main())
