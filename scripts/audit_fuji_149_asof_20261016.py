#!/usr/bin/env python3
"""149-feature AS-OF audit for *unrun* 2026 Fuji Stakes (not a historical-race parity audit).
Read-only. Source: netkeiba pre-race shutuba. No final odds, results or forecasting.
All 149 catalog rows emitted. 'Requires history' is not counted as obtained.
"""
import hashlib
import json
import re
import sys
import time
import unicodedata
from datetime import datetime,timezone
from urllib.parse import urlparse
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
from pathlib import Path
from bs4 import BeautifulSoup

RID="202605040611"
URL=f"https://race.netkeiba.com/race/shutuba.html?race_id={RID}"
EXPECTED=14
CATALOG=Path(__file__).resolve().parents[1]/"contracts/l1-feature-catalog-v1.json"
RACE_SOURCES=["netkeiba/shutuba", "jra/accessD"]
HORSE_PATTERN=re.compile(r"/horse/([0-9A-Za-z]+)(?:/|[?#]|$)")
JOCKEY_PATTERN=re.compile(r"/jockey/(?:result/recent/)?(\d+)(?:/|[?#]|$)")
TRAINER_PATTERN=re.compile(r"/trainer/(?:result/recent/)?(\d+)(?:/|[?#]|$)")
PROHIBITED="FUTURE_NOT_RELEASED"

def clean(v):
    return re.sub(r"\s+","",unicodedata.normalize("NFKC",str(v or ""))).strip()

def fetch():
    last=None
    for attempt in range(3):
        if attempt:time.sleep(2**attempt)
        try:
            with urlopen(Request(URL,headers={"User-Agent":"Mozilla/5.0 (compatible; KeibaDataAudit/1.0)","Accept-Language":"ja"}),timeout=22) as r:
                if urlparse(r.geturl()).hostname!="race.netkeiba.com":raise ValueError("SOURCE_REDIRECT_HOST")
                raw=r.read(6000000)
                if len(raw)>=6000000:raise ValueError("RESPONSE_TOO_LARGE")
                encoding=r.headers.get_content_charset() or "utf-8"
                try:html=raw.decode(encoding)
                except (UnicodeError,LookupError):html=raw.decode("euc-jp")
                if "アクセス制限" in html or "不正なアクセス" in html:raise ValueError("SOURCE_ACCESS_RESTRICTED")
                return html,hashlib.sha256(raw).hexdigest(),r.headers.get("Date")
        except HTTPError as e:
            if e.code not in (502,503,504):raise RuntimeError(f"HTTP_{e.code}_NOT_RETRIED") from e
            last=f"HTTP_{e.code}"
        except (URLError,TimeoutError) as e:last=f"TEMPORARY_NETWORK_{type(e).__name__}"
    raise RuntimeError(last or "FETCH_FAILED")

def link_id(cell,regex):
    if cell is None:return None
    for a in cell.select("a[href]"):
        m=regex.search(a.get("href",""))
        if m:return m.group(1)
    return None

def number(s):
    s=clean(s)
    return int(s) if re.fullmatch(r"\d+",s) else None

def real(s):
    s=clean(s)
    return float(s) if re.fullmatch(r"\d+(?:\.\d+)?",s) else None

def get_text(cell):
    return clean(cell.get_text(" ",strip=True)) if cell else ""

def parse(html):
    soup=BeautifulSoup(html,"html.parser")
    table=soup.select_one("table.Shutuba_Table")
    if table is None:raise ValueError("NETKEIBA_SHUTUBA_TABLE_ABSENT")
    header_cells=table.select("tr th")
    head=[clean(h.get_text(" ",strip=True)) for h in header_cells][:30]
    # Race card's final columns vary by subscription; use header-based lookup.
    def col(*names):
        return next((i for i,x in enumerate(head) if any(n in x for n in names)),None)
    idx={k:col(*names) for k,names in {
        "gate":("枠",),"horse_number":("馬番",),"horse_name":("馬名",),
        "sexage":("性齢",),"carried_weight":("斤量",),
        "jockey":("騎手",),"trainer":("厩舎","調教師"),
        "body_weight":("馬体重",)}.items()}
    rows=[]
    for tr in table.select("tr"):
        cells=tr.find_all("td",recursive=False)
        if not cells:continue
        links=[a for a in tr.select("a[href]") if HORSE_PATTERN.search(a.get("href",""))]
        if not links:continue
        ids={HORSE_PATTERN.search(a["href"]).group(1) for a in links}
        if len(ids)!=1:raise ValueError("AMBIGUOUS_HORSE_ID")
        def td(k):
            i=idx[k]
            return cells[i] if i is not None and i<len(cells) else None
        def text(k):return get_text(td(k))
        horse_id=next(iter(ids))
        horse_name=clean(next((a.get_text(" ",strip=True) for a in links if clean(a.get_text(" ",strip=True))),""))
        sexage=re.search(r"(牡|牝|セ)\s*(\d+)",text("sexage"))
        body=re.search(r"(\d{3})(?:\s*\(([+-]?\d+)\))?",text("body_weight"))
        rows.append({"horse_id":horse_id,"horse_name":horse_name,
          "gate":number(text("gate")),"horse_number":number(text("horse_number")),
          "sex":sexage.group(1) if sexage else None,
          "age":int(sexage.group(2)) if sexage else None,
          "carried_weight":real(text("carried_weight")),
          "jockey_id":link_id(td("jockey") or tr,JOCKEY_PATTERN),
          "trainer_id":link_id(td("trainer") or tr,TRAINER_PATTERN),
          "body_weight":int(body.group(1)) if body else None,
          "body_weight_diff":int(body.group(2)) if body and body.group(2) is not None else None,
          "source_cells":{k:text(k) for k in idx}})
    if len(rows)!=EXPECTED:raise ValueError(f"RUNNER_COUNT:{len(rows)}/{EXPECTED} (headers={head!r})")
    if len({r["horse_id"] for r in rows})!=len(rows):raise ValueError("HORSE_ID_DUPLICATE")
    if len({r["horse_name"] for r in rows})!=len(rows):raise ValueError("HORSE_NAME_DUPLICATE")
    box=soup.select_one(".RaceList_NameBox")
    raw_course=get_text(soup.select_one(".RaceData01"))
    raw_condition=get_text(soup.select_one(".RaceData02"))
    name=get_text(soup.select_one(".RaceName"))
    title=get_text(soup.title)
    if "富士" not in name and "富士" not in title:raise ValueError("RACE_NAME_MISMATCH")
    if "東京" not in raw_condition or "6日" not in raw_condition or "4回" not in raw_condition:
        raise ValueError("MEETING_HEADER_MISMATCH")
    distance=re.search(r"([0-9]{3,4})m",raw_course)
    count=re.search(r"(\d+)頭",raw_condition)
    if not count or int(count.group(1))!=EXPECTED:raise ValueError("FIELD_COUNT_IN_HEADER_MISMATCH")
    if not distance:raise ValueError("COURSE_DISTANCE_MISSING")
    if "芝" in raw_course and "ダート" in raw_course:surface="MIXED"
    elif "芝" in raw_course:surface="TURF"
    elif "ダ" in raw_course:surface="DIRT"
    else:surface=None
    mclass=next((name for name in ("新馬","未勝利","1勝クラス","2勝クラス","3勝クラス","オープン") if name in raw_condition),None)
    klass={"新馬":"NEWCOMER","未勝利":"MAIDEN","1勝クラス":"ONE_WIN","2勝クラス":"TWO_WIN","3勝クラス":"THREE_WIN","オープン":"OPEN"}.get(mclass)
    grade=re.search(r"\(G([123])\)",title,re.I)
    age=re.search(r"([234])歳(以上)?",raw_condition)
    weight_rule=next((v for label,v in (("ハンデ","HANDICAP"),("別定","SPECIAL_WEIGHT"),("定量","SET_WEIGHT"),("馬齢","WEIGHT_FOR_AGE")) if label in raw_condition),None)
    raw={"race.venue_code":"05","race.meeting_no":4,"race.meeting_day":6,"race.race_no":11,
         "race.race_id":RID,"race.discipline":"FLAT" if surface and "障害" not in raw_course else None,
         "race.surface":surface,"race.distance_m":int(distance.group(1)),
         "race.direction":"LEFT" if "左" in raw_course else "RIGHT" if "右" in raw_course else None,
         "race.weather":(re.search(r"天候[:：]?([^/]+)",raw_course).group(1) if re.search(r"天候[:：]?([^/]+)",raw_course) else None),
         "race.track_condition":(re.search(r"馬場[:：]?([^/]+)",raw_course).group(1) if re.search(r"馬場[:：]?([^/]+)",raw_course) else None),
         "race.actual_start_time":None,
         "race.course_layout":"OUTER" if "外" in raw_course else "INNER" if "内" in raw_course else None,
         "race.course_laps":int(re.search(r"(\d+)周",raw_course).group(1)) if re.search(r"(\d+)周",raw_course) else None,
         "race.race_class_raw":mclass,"race.race_class_normalized":klass,
         "race.grade":"G"+grade.group(1) if grade else None,
         "race.age_condition_raw":age.group(0) if age else None,
         "race.age_min":int(age.group(1)) if age else None,
         "race.age_max":int(age.group(1)) if age and not age.group(2) else None,
         "race.sex_condition":"FEMALE_ONLY" if "牝馬限定" in raw_condition else "ANY",
         "race.weight_rule":weight_rule,
         "race.mixed":"(混)" in raw_condition or "[混]" in raw_condition,
         "race.international":"(国際)" in raw_condition,
         "race.special_designated":"(特指)" in raw_condition,
         "race.designated":"(指)" in raw_condition or "[指]" in raw_condition,
         "race.course_meta_raw":raw_course,
         "race.race_condition_raw":raw_condition,
         "race.field_size":len(rows)}
    # 'NORMAL' is not blindly inferred from absence of '内/外':
    # this value would need a backed, track-specific course map.
    return rows,raw,{"title":title,"race_name":name,"raw_course":raw_course,"raw_condition":raw_condition,
                     "table_headings":head,
                     "no_inner_outer_information_in_page":raw["race.course_layout"] is None,
                     "designation_legacy_regex_conflict":("(指)" in raw_condition and "[指]" not in raw_condition)}

def audit(catalog,rows,raw):
    assert len(catalog["features"])==149
    entry_mapping={
      "entry.horse_id":"horse_id","entry.horse_number":"horse_number","entry.gate":"gate",
      "entry.sex":"sex","entry.age":"age","entry.carried_weight":"carried_weight",
      "entry.jockey_id":"jockey_id","entry.trainer_id":"trainer_id",
      "entry.body_weight":"body_weight","entry.body_weight_diff":"body_weight_diff"}
    report=[]
    for f in catalog["features"]:
        fid=f["feature_id"]
        boundary=f["history_boundary"]
        item={"feature_id":fid,"catalog_status":f["status"],"domain":f["domain"],
              "available_at_contract":f["available_at"],"source_field":f["source_field"],
              "history_boundary":boundary,"source_checked":URL,
              "checked_from":"ACTUAL_2026_FUJI_NETKEIBA_PRE_RACE_PAGE",
              "status":None,"observed_value":None,"coverage":None,"reason":None}
        if boundary=="CURRENT_RACE_RESULT":
            item.update(status="NOT_YET_PUBLISHED",reason="Result only after actual race. No pseudo-result.")
        elif boundary=="CURRENT_RACE_MARKET":
            item.update(status="POST_RACE_FINAL_ONLY",reason="Final odds/popularity/payout unavailable before race; live odds source is JRA, not netkeiba.")
        elif boundary=="STRICT_PRIOR_DATE_ONLY":
            item.update(status="BACKFILL_HISTORY_REQUIRED",reason="Not a raw feature on current card. Historical source/point-in-time feature computation not executed by page probe.")
        elif boundary=="STATIC_FACT":
            item.update(status="HORSE_PROFILE_REQUIRED",reason="Horse or pedigree profile pack is a separate source, not the live race table. Presence for all 14 not yet checked.")
        elif boundary=="CURRENT_RACE_KNOWN_FACT":
            if fid=="race.actual_start_time":
                item.update(status="NOT_YET_PUBLISHED",reason="Actual start not known until race started; scheduled post time is not actual start.")
            elif fid in entry_mapping:
                key=entry_mapping[fid];vals=[r.get(key) for r in rows];found=sum(v is not None and v!="" for v in vals)
                item["coverage"]={"present":found,"expected":EXPECTED}
                if found==EXPECTED:
                    item["status"]="OBSERVED_NETKEIBA_PAGE"
                    item["observed_value"]=[{"horse_id":r["horse_id"],"value":r[key]} for r in rows]
                elif found==0 and key in ("body_weight","body_weight_diff"):
                    item.update(status="NOT_YET_PUBLISHED",reason="Body weight not published in fetched pre-race page")
                else:
                    item.update(status="PARTIAL_OR_NOT_EXTRACTED",reason=f"Only {found}/{EXPECTED} values parsed; cannot claim source absence")
            else:
                val=raw.get(fid)
                if val is not None:
                    item["status"]="OBSERVED_OR_DERIVED_FROM_PAGE"
                    item["observed_value"]=val
                    if fid=="race.designated" and val:
                        item["reason"]="NOTICE: legacy BACKFILL normalizer only recognizes [指]; current netkeiba has (指). This is a normalization difference, not a page absence."
                elif fid in ("race.weather","race.track_condition"):
                    item.update(status="NOT_YET_PUBLISHED_OR_NOT_VISIBLE",reason="No verified page value at observation time; check JRA current conditions separately.")
                else:item.update(status="NOT_VISIBLE_OR_UNPROVEN",reason="Not observed in current HTML with a safe source mapping; not assumed false or zero.")
        else:
            item.update(status="UNCLASSIFIED_BLOCKED",reason="Unknown catalog boundary; stop on novel category")
        report.append(item)
    assert len(report)==len({x["feature_id"] for x in report})==149
    return report

def main():
    at=datetime.now(timezone.utc).isoformat()
    cat=json.loads(CATALOG.read_text(encoding="utf-8"))
    if len(cat.get("features",[]))!=149:
        raise ValueError("CATALOG_DRIFT_EXPECTED_149")
    result={"contract":"FUJI_20261017_149_ASOF_AUDIT_V1","race_id":RID,"race_date":"2026-10-17",
            "is_pre_race":True,"source_url":URL,"observed_at_client_utc":at,
            "catalog_sha256":hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
            "excludes_current_race_results":True,"netkeiba_odds_used":False,
            "backfill_actual_data_retrieved":False,"ledger":[]}
    try:
        html,sha,http_date=fetch()
        rows,raw,meta=parse(html)
        result.update(source_sha256=sha,source_http_date=http_date,raw_page=meta,
                      runners=rows,catalog_feature_count=149)
        result["ledger"]=audit(cat,rows,raw)
        statuses={}
        for r in result["ledger"]:statuses[r["status"]]=statuses.get(r["status"],0)+1
        result["status_counts"]=statuses
        result["status"]="COMPLETED_WITH_UNVERIFIED_ITEMS"
        essential=("entry.horse_id","entry.horse_number","entry.gate","entry.carried_weight",
                  "entry.jockey_id","entry.trainer_id","entry.sex","entry.age")
        blocking=[i for i in result["ledger"] if i["feature_id"] in essential and i["status"]!="OBSERVED_NETKEIBA_PAGE"]
        result["core_entry_complete"]=not blocking
        result["core_entry_blocked_fields"]=[i["feature_id"] for i in blocking]
    except Exception as exc:
        result.update(status="BLOCKED",error=type(exc).__name__+":"+str(exc))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 2 if result["status"]=="BLOCKED" else 0

if __name__=="__main__":sys.exit(main())
