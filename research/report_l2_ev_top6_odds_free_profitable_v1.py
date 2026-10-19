#!/usr/bin/env python3
"""Sort-only report of *existing* 2023-2025 all 6870 x 7 L2 non-odds gate outcomes.

No bet recalculation, no new filtering strategy, no model scoring.
All in-sample profitable post-prune combinations retained, including tiny samples.
"""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

YRS=("2023","2024","2025")
NAMES={"WIN":"単勝","PLACE":"複勝","QUINELLA":"馬連","WIDE":"ワイド",
       "EXACTA_FIRST":"馬単・軸1着","EXACTA_SECOND":"馬単・軸2着",
       "TRIO":"三連複","TRIFECTA_FIRST":"三連単・軸1着",
       "TRIFECTA_SECOND":"三連単・軸2着","TRIFECTA_THIRD":"三連単・軸3着"}
GATES=("RAW_TOP6","RAW_TOP10","CALIBRATED_TOP6","CALIBRATED_TOP10",
       "ABILITY_Z_GE_M1","ABILITY_Z_GE_0","MODEL_WIN_P_GE_003")

def load(path):
    with gzip.open(path,"rt",encoding="utf-8",newline="") as fd:
        return list(csv.DictReader(fd))

def intv(d,name):return int(d[name])
def key(r):return (int(r["ev_axis_rank"]),r["family"],
                  int(r["raw_ai_rank_1"]),int(r["raw_ai_rank_2"]),r["gate"])

def write(path,rows):
    assert rows
    with path.open("w",newline="",encoding="utf-8-sig") as fd:
        writer=csv.DictWriter(fd,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pooled",required=True)
    ap.add_argument("--yearly",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    pooled=load(a.pooled)
    yearly=load(a.yearly)
    assert len(pooled)==6870*8,len(pooled)
    assert len(yearly)==6870*8*3,len(yearly)
    yearly_index={(r["year"],*key(r)):r for r in yearly}
    assert len(yearly_index)==len(yearly)
    eligible=[r for r in pooled if r["gate"] in GATES and
              intv(r,"after_profit_yen")>0 and intv(r,"after_tickets")>0]
    positive3=[]
    for r in eligible:
        yr=[yearly_index[(year,*key(r))] for year in YRS]
        tickets=intv(r,"after_tickets")
        hits=intv(r,"after_hits")
        refund=intv(r,"after_return_yen")
        assert tickets==sum(intv(q,"after_tickets") for q in yr)
        assert hits==sum(intv(q,"after_hits") for q in yr)
        assert refund==sum(intv(q,"after_return_yen") for q in yr)
        assert refund-tickets*100==intv(r,"after_profit_yen")
        assert tickets>0 and hits>0
        # The exact first & second partners are ordered for TRIFECTA.
        j,k=int(r["raw_ai_rank_1"]),int(r["raw_ai_rank_2"])
        partner=f"{j}位" + (f" → {k}位" if k else "")
        days=sum(intv(q,"after_profit_yen")>0 and
                 intv(q,"after_tickets")>0 for q in yr)
        positive3.append({
            "期待値AI軸順位":intv(r,"ev_axis_rank"),
            "券種と軸位置":NAMES[r["family"]],
            "純粋AI相手順位":partner,
            "削減条件":r["gate"],
            "購入点数_3年":tickets,
            "的中点数_3年":hits,
            "的中率_3年_pct":round(hits/tickets*100,6),
            "投資金額_3年_円":tickets*100,
            "払戻金額_3年_円":refund,
            "利益_3年_円":refund-tickets*100,
            "回収率_3年_pct":round(refund/tickets,6),
            "削減前購入点数_3年":intv(r,"before_tickets"),
            "削減前的中点数_3年":intv(r,"before_hits"),
            "削減前利益_3年_円":intv(r,"before_profit_yen"),
            "削減前回収率_3年_pct":r["before_roi_pct"],
            "削減後黒字年度数":days,
            **{f"{q['year']}_購入点数":intv(q,"after_tickets") for q in yr},
            **{f"{q['year']}_的中点数":intv(q,"after_hits") for q in yr},
            **{f"{q['year']}_利益_円":intv(q,"after_profit_yen") for q in yr},
            **{f"{q['year']}_回収率_pct":round(intv(q,"after_return_yen")/
                                             intv(q,"after_tickets"),6)
               if intv(q,"after_tickets") else "" for q in yr},
            "_family":r["family"],"_ranks":(j,k),
            "_gate":r["gate"],"_key":key(r),
        })
    assert positive3
    key_sort = {
       "hit_rate":lambda r:(-r["的中率_3年_pct"],-r["的中点数_3年"],-r["購入点数_3年"]),
       "volume":lambda r:(-r["購入点数_3年"],-r["的中率_3年_pct"],-r["利益_3年_円"]),
       "profit":lambda r:(-r["利益_3年_円"],-r["購入点数_3年"]),
       "roi":lambda r:(-r["回収率_3年_pct"],-r["的中点数_3年"]),
       "hit_count":lambda r:(-r["的中点数_3年"],-r["的中率_3年_pct"]),
    }
    keys_display=list(positive3[0].keys())[:-3]
    keys_display.remove("_family")
    def strip(rows):
        return [{k:r[k] for k in keys_display if not k.startswith("_")} for r in rows]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    # One full detailed CSV, without exclusion or sampling, sorted primarily
    # by hit count (not by hindsight ROIs that reward one large jackpot).
    write(out/"all-post-prune-three-year-profitable-detailed.csv",
          strip(sorted(positive3,key=key_sort["hit_count"])))
    by_sort={}
    for name,sortfun in key_sort.items():
        ordered=sorted(positive3,key=sortfun)
        write(out/f"top-100-by-{name}.csv",strip(ordered[:100]))
        by_sort[name]=ordered[:18]
    stable=sorted([x for x in positive3 if x["削減後黒字年度数"]==3],
                  key=key_sort["profit"])
    write(out/"all-three-individual-years-profitable.csv",
          strip(stable)) if stable else None
    group_by={"削減条件":Counter(),"券種と軸位置":Counter(),"期待値AI軸順位":Counter(),
              "黒字年度数":Counter(),"的中回数帯":Counter()}
    for x in positive3:
        group_by["削減条件"][x["削減条件"]]+=1
        group_by["券種と軸位置"][x["券種と軸位置"]]+=1
        group_by["期待値AI軸順位"][x["期待値AI軸順位"]]+=1
        group_by["黒字年度数"][x["削減後黒字年度数"]]+=1
        hits=x["的中点数_3年"]
        key="1" if hits==1 else ("2-4" if hits<5 else
              ("5-9" if hits<10 else "10+"))
        group_by["的中回数帯"][key]+=1
    subset=[x for x in positive3 if x["購入点数_3年"]>=1000 and
            x["的中点数_3年"]>=10]
    summary={
      "source":"2023-2025 saved precomputed exact-rank ticket/gate/settlement aggregates",
      "source_recalculated":False,
      "count_positive_settings":len(positive3),
      "count_positive_across_all_three_individual_years":len(stable),
      "sample_filter":None,
      "summary":{k:dict(v) for k,v in group_by.items()},
      "top_by":{k:strip(rows) for k,rows in by_sort.items()},
      "top_hit_rate_min_1000_tickets_10_hits":
        strip(sorted(subset,key=key_sort["hit_rate"])[:18]),
      "top_profit_three_individual_years":
        strip(stable[:18]),
      "warning":"A single EV/rank physical ticket can appear across multiple gates and strategies. Do not sum profits. Historical EV axis already used final win odds. No new forward validation."
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("BLACK_SETTINGS",len(positive3),"THREE_YEARS_BLACK",len(stable),
          "HIT_RATE_WITH_MIN_SAMPLE",len(subset),flush=True)
    print("REPORT_COMPLETE",flush=True)

if __name__=="__main__":main()
