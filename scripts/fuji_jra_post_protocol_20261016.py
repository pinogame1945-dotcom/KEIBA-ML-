#!/usr/bin/env python3
"""Real JRA POST protocol investigation, read-only and bounded.
Only uses cname strings extracted from official HTML, never synthesizes CNAME.
Checks error013 and verifies Fuji identity before declaring quotes usable.
"""
import json,re,hashlib,sys
from urllib.request import Request,urlopen
from urllib.parse import urlparse,urlencode
from datetime import datetime,timezone
from bs4 import BeautifulSoup
from pathlib import Path

CARD_URL="https://www.jra.go.jp/JRADB/accessD.html?CNAME=pw01dde0105202604061120261017%2F22"
POST_URL="https://www.jra.go.jp/JRADB/accessO.html"
START_CNAME="pw15oli00/6D"  # official CARD doAction on observation, verify before use
BAD=("パラメータエラー","エラー013","ご指定のページに接続できませんでした")
def fetch(url, cname=None):
    if urlparse(url).hostname!="www.jra.go.jp" or not url.startswith("https://"):
        raise ValueError("NON_JRA_SOURCE")
    headers={"User-Agent":"Mozilla/5.0 (compatible; FujiPrebetP4Post/1.0)","Accept-Language":"ja,en;q=0.5"}
    if cname is not None:
        if not re.fullmatch(r"[A-Za-z0-9]+/[A-Fa-f0-9]{2}",cname):raise ValueError("UNVERIFIED_CNAME_FORMAT")
        headers.update({"Content-Type":"application/x-www-form-urlencoded","Referer":CARD_URL,
                        "Origin":"https://www.jra.go.jp"})
        req=Request(url,data=urlencode({"cname":cname}).encode("ascii"),headers=headers,method="POST")
    else:req=Request(url,headers=headers)
    with urlopen(req,timeout=22) as response:
        if urlparse(response.geturl()).hostname!="www.jra.go.jp":raise ValueError("CROSS_SITE_REDIRECT")
        data=response.read(6_000_000)
        if len(data)==6_000_000:raise ValueError("MAX_HTML_BYTES")
        ct=response.headers.get("Content-Type","")
        m=re.search(r"charset\s*=\s*([\w-]+)",ct,re.I)
        header_date=response.headers.get("Date")
    content=None
    for enc in dict.fromkeys(([m.group(1)] if m else [])+["cp932","utf-8","euc_jp"]):
        try:
            txt=data.decode(enc)
            if "オッズ" in txt or "パラメータエラー" in txt or "富士" in txt:
                content=txt
                break
        except (UnicodeError,LookupError):continue
    if content is None:raise ValueError("NO_VALID_JAPANESE_DECODE")
    return content,{"sha256":hashlib.sha256(data).hexdigest(),"http_date":header_date,"bytes":len(data),"method":"POST" if cname else "GET"}
def links(soup):
    candidates=[]
    for a in soup.select("a[onclick]"):
        code=a.get("onclick","")
        m=re.search(r"doAction\(\s*['\"](/JRADB/accessO\.html)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)",code)
        if not m:continue
        name=re.sub(r"\s+","",a.get_text(" ",strip=True)) or (a.select_one("img[alt]") or {}).get("alt","")
        if not re.fullmatch(r"[A-Za-z0-9]+/[A-Fa-f0-9]{2}",m.group(2)):continue
        candidates.append({"label":name[:46],"cname":m.group(2)})
    return candidates
def snapshot(html,meta,source_cname=None):
    soup=BeautifulSoup(html,"html.parser")
    txt=re.sub(r"\s+","",soup.get_text(" ",strip=True))
    bad=any(z in txt for z in BAD)
    races=["富士" in txt,"2026年10月17日" in txt, "4回東京6日" in txt]
    headings=[re.sub(r"\s+","",h.get_text(" ",strip=True))[:80] for h in soup.select("h1,h2,h3,table caption")[:12]]
    nav=links(soup)
    return {"cname":source_cname,"meta":meta,"error013":bad,"header":headings,
      "fuji_race_identifiers":races,"links_total":len(nav),
      "links":nav[:35],
      "table_count":len(soup.select("table")),
      "excerpt":txt[:420],
      "odds_values_verified":False}
def main():
    out={"contract":"FUJI_JRA_P4_POST_PROTOCOL_20261016_V1","race_id":"202605040611",
         "checked_utc":datetime.now(timezone.utc).isoformat(),"steps":[],"status":"UNVERIFIED"}
    try:
        card,cm=fetch(CARD_URL);cs=snapshot(card,cm)
        out["steps"].append({"step":"official_card_get",**cs})
        vals=[x["cname"] for x in cs["links"] if x["label"]=="オッズ"]
        if len(set(vals))!=1 or vals[0]!=START_CNAME:raise ValueError("OFFICIAL_ODDS_LINK_CHANGED")
        if not all(cs["fuji_race_identifiers"]):raise ValueError("CARD_NOT_FUJI_2026")
        index,im=fetch(POST_URL,START_CNAME)
        page=snapshot(index,im,START_CNAME)
        out["steps"].append({"step":"odds_index_post",**page})
        if page["error013"]:out["status"]="JRA_POST_ERROR013"
        elif page["links_total"]==0:out["status"]="JRA_POST_INDEX_NO_NAV"
        else:
            out["status"]="JRA_POST_INDEX_NAV_SUCCESS"
            # Limit to ONE subpage found from official POST response. Prefer Fuji
            # or Tokyo day, to identify actual routing while limiting requests.
            candidate=next((x for x in page["links"] if "東京" in x["label"] or "10月17" in x["label"]),None)
            if candidate:
                follow,meta=fetch(POST_URL,candidate["cname"])
                step=snapshot(follow,meta,candidate["cname"])
                out["steps"].append({"step":"first_verified_index_child_post",**step})
                if step["error013"]:out["status"]="JRA_CHILD_POST_ERROR013"
        print("JRA_POST_SUMMARY "+json.dumps({"status":out["status"],
          "steps":[{"step":s["step"],"cname":s["cname"],"error013":s["error013"],"links_total":s["links_total"],
                    "table_count":s["table_count"],"headings":s["header"],"excerpt":s["excerpt"][:120]} for s in out["steps"]]},ensure_ascii=False))
    except Exception as exc:out["status"]="BLOCKED";out["reason"]=type(exc).__name__+":"+str(exc)
    print("JRA_POST_JSON "+json.dumps(out,ensure_ascii=False))
    return 0 if out["status"]=="JRA_POST_INDEX_NAV_SUCCESS" else 2
if __name__=="__main__":sys.exit(main())
