#!/usr/bin/env python3
import argparse,csv,json
from collections import defaultdict
from pathlib import Path

def ap():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--years",default="2021,2022,2023,2024,2025")
    p.add_argument("--output-support",required=True)
    p.add_argument("--output-council",required=True)
    p.add_argument("--output-csv",required=True)
    return p.parse_args()

def merge_stat(dst,src):
    dst["n"]+=int(src.get("n",0)); dst["win"]+=int(src.get("win",0)); dst["podium"]+=int(src.get("podium",0))

def fin(s):
    n=s["n"]
    return {**s,"win_rate":s["win"]/n if n else None,"podium_rate":s["podium"]/n if n else None}

def merge_scope(scopes):
    dims=("by_support_count","by_support_score","by_weighted_wait")
    merged={d:defaultdict(lambda:{"n":0,"win":0,"podium":0}) for d in dims}
    for s in scopes:
        for d in dims:
            for k,v in s[d].items():
                merge_stat(merged[d][str(k)],v)
    out={}
    for d in dims:
        out[d]={k:fin(v) for k,v in sorted(merged[d].items(),key=lambda z:int(z[0]))}
    out["n"]=sum(int(s.get("n",0)) for s in scopes)
    return out

def main():
    a=ap(); root=Path(a.root)
    years=[int(x) for x in a.years.split(",") if x.strip()]
    supports={}
    councils={}
    for y in years:
        yp=root/f"y{y}"
        supports[y]=json.loads((yp/"support-summary.json").read_text(encoding="utf-8"))
        councils[y]=json.loads((yp/"council.json").read_text(encoding="utf-8"))

    first=supports[years[0]]
    support={
      "contract":"L16_OUTSIDER_SUPPORT_SAFE_ALL_YEARS_V1",
      "years":years,
      "races":sum(supports[y]["races"] for y in years),
      "candidate_count":first["candidate_count"],
      "candidates":first["candidates"],
      "expected_king_top3_rows":sum(supports[y]["expected_king_top3_rows"] for y in years),
      "evaluable_king_top3_rows":sum(supports[y]["evaluable_king_top3_rows"] for y in years),
      "missing_finish_rows":sum(supports[y]["missing_finish_rows"] for y in years),
      "definition":first["definition"],
      "overall":{
        "king_top3":merge_scope([supports[y]["overall"]["king_top3"] for y in years]),
        "by_king_rank":{
          str(k):merge_scope([supports[y]["overall"]["by_king_rank"][str(k)] for y in years])
          for k in (1,2,3)
        },
      },
      "by_year":{str(y):supports[y]["by_year"][str(y)] for y in years},
      "2026_sealed":True,
      "odds_used":False,
    }

    def agg_metric(path):
        hits=races=0
        for y in years:
            x=councils[y]
            for key in path:x=x[key]
            hits+=int(x["hits"]); races+=int(x["races"])
        return {"hits":hits,"races":races,"rate":hits/races if races else None}

    council={
      "contract":"L16_OUTSIDER_COUNCIL_SAFE_ALL_YEARS_V1",
      "years":years,
      "races":sum(councils[y]["races"] for y in years),
      "candidate_count":13,
      "metrics":{},
      "seven_top3_blind":{"count":sum(councils[y]["seven_top3_blind"]["count"] for y in years),"council":{}},
      "seven_top6_blind":{"count":sum(councils[y]["seven_top6_blind"]["count"] for y in years),"council":{}},
      "by_year":{str(y):councils[y] for y in years},
      "2026_sealed":True,
      "odds_used":False,
    }
    for scheme in ("weighted","count"):
        council["metrics"][scheme]={str(k):agg_metric(["metrics",scheme,str(k)]) for k in (1,3,6)}
        council["seven_top3_blind"]["council"][scheme]={str(k):agg_metric(["seven_top3_blind","council",scheme,str(k)]) for k in (1,3,6)}
        council["seven_top6_blind"]["council"][scheme]={str(k):agg_metric(["seven_top6_blind","council",scheme,str(k)]) for k in (1,3,6)}

    Path(a.output_support).write_text(json.dumps(support,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    Path(a.output_council).write_text(json.dumps(council,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    fields=["scope","king_rank","dimension","value","n","win","podium","win_rate","podium_rate"]
    with open(a.output_csv,"w",newline="",encoding="utf-8-sig") as fh:
        w=csv.DictWriter(fh,fieldnames=fields);w.writeheader()
        def emit(scope,kr,obj):
            for dim,key in (("support_count","by_support_count"),("support_score","by_support_score"),("weighted_wait","by_weighted_wait")):
                for value,s in obj[key].items():
                    w.writerow({"scope":scope,"king_rank":kr,"dimension":dim,"value":value,**s})
        emit("overall_top3","1-3",support["overall"]["king_top3"])
        for k in ("1","2","3"):emit("overall_rank",k,support["overall"]["by_king_rank"][k])

    print("L16_OUTSIDER_SUPPORT_FAST_AGG_OK")
    print(json.dumps({
      "races":support["races"],
      "missing_finish_rows":support["missing_finish_rows"],
      "weighted_council":council["metrics"]["weighted"],
      "count_council":council["metrics"]["count"],
    },ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
