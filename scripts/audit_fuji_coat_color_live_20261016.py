#!/usr/bin/env python3
"""Read-only check: netkeiba horse profiles' displayed coat colours of 14 Fuji horses.
This does NOT repair or modify existing BACKFILL storage nor edit frozen feature specifications.
"""
import json,re,sys,time,unicodedata
from datetime import datetime,timezone
from pathlib import Path
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen
from bs4 import BeautifulSoup
SRC=Path("audit-results/fuji-20261017-149-asof-20261016.json")
j=json.loads(SRC.read_text(encoding="utf-8"))
assert len(j["runners"])==14
COATS=("黒鹿毛","青鹿毛","栃栗毛","鹿毛","栗毛","青毛","芦毛","白毛")
P=re.compile(r"(?:牡|牝|セ)\s*\d{1,2}\s*歳?\s*("+("|".join(COATS))+r")")
out={"contract":"FUJI_20261017_PROFILE_COAT_COLOR_DIRECT_READ_ONLY_V1",
     "observed_at_utc":datetime.now(timezone.utc).isoformat(),
     "source":"db.netkeiba.com/horse/<horse_id>/",
     "date_scope":"2026-10-16 before Fuji",
     "horses":[]}
for i,h in enumerate(j["runners"]):
    id=h["horse_id"];url=f"https://db.netkeiba.com/horse/{id}/"
    e={"horse_id":id,"horse_name":h["horse_name"],"url":url,
       "coat_color":None,"status":None,"source_snippet":None}
    try:
      if i:time.sleep(1.6)
      req=Request(url,headers={"User-Agent":"Mozilla/5.0 (compatible; KeibaProfileAudit/1.0)",
                                "Accept-Language":"ja,en;q=0.5"})
      with urlopen(req,timeout=20) as res:
        if res.status!=200:raise RuntimeError(f"HTTP {res.status}")
        raw=res.read(5000000)
      html=None
      for enc in ("euc_jp","cp932","utf-8"):
        try:
          candidate=raw.decode(enc)
          if "競走馬" in candidate or h["horse_name"] in candidate:
            html=candidate;break
        except UnicodeError:pass
      if not html:raise RuntimeError("DECODE_FAILED")
      soup=BeautifulSoup(html,"html.parser")
      # Coat text appears alongside status/sex/age, *not* profile td-th rows.
      candidates=[]
      for selector in (".horse_title",".horse_title_2","div.HorseTitle",".HorseName"):
        for node in soup.select(selector)[:3]:candidates.append(node.get_text(" ",strip=True))
      candidates.append(soup.get_text(" ",strip=True)[:3500])
      candidate=" ".join(unicodedata.normalize("NFKC",x) for x in candidates)
      m=P.search(candidate)
      if m and h["horse_name"] in candidate:
        e.update(coat_color=m.group(1),status="LIVE_PROFILE_CONFIRMED",
                 source_snippet=candidate[max(0,m.start()-38):m.end()+38])
      else:e.update(status="COAT_PATTERN_NOT_FOUND",source_snippet=candidate[:140])
    except HTTPError as x:e.update(status=f"HTTP_{x.code}")
    except (URLError,TimeoutError) as x:e.update(status="TEMPORARY_NETWORK_ERROR",reason=type(x).__name__)
    except Exception as x:e.update(status="SOURCE_READ_FAILED",reason=str(x))
    out["horses"].append(e)
    print("HORSE_COAT",json.dumps(e,ensure_ascii=False))
out["confirmed_count"]=sum(x["status"]=="LIVE_PROFILE_CONFIRMED" for x in out["horses"])
out["complete"]=out["confirmed_count"]==14
print("COAT_SUMMARY",json.dumps({"confirmed":out["confirmed_count"],"total":14,"statuses":{v:sum(x["status"]==v for x in out["horses"]) for v in sorted(set(x["status"] for x in out["horses"]))}},ensure_ascii=False))
print("COAT_JSON",json.dumps(out,ensure_ascii=False))
sys.exit(0 if out["complete"] else 2)
