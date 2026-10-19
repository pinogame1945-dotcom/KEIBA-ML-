#!/usr/bin/env python3
"""Report-only: identify 2023/24/25 consistently losing, well-sampled EV1-6
raw-AI exact-rank purchase configurations, both unpruned and each of seven
pre-existing odds-free L2 pruning conditions. No rescore, no change to L2.1.
Historical sample is same three years and 2026 stays sealed.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

YEARS=("2023","2024","2025")
GATES=("ALL","RAW_TOP6","RAW_TOP10","CALIBRATED_TOP6","CALIBRATED_TOP10",
       "ABILITY_Z_GE_M1","ABILITY_Z_GE_0","MODEL_WIN_P_GE_003")
NAMES={"WIN":"単勝","PLACE":"複勝","QUINELLA":"馬連","WIDE":"ワイド",
       "EXACTA_FIRST":"馬単・軸1着","EXACTA_SECOND":"馬単・軸2着",
       "TRIO":"三連複","TRIFECTA_FIRST":"三連単・軸1着",
       "TRIFECTA_SECOND":"三連単・軸2着","TRIFECTA_THIRD":"三連単・軸3着"}
MIN_PER_YEAR=1000
THRESHOLDS=(50,70,80,90,100)

def read(path):
    with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))

def write(path,records):
    assert records, path
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)

def key(r):
    return (int(r["ev_axis_rank"]),r["family"],
            int(r["raw_ai_rank_1"]),int(r["raw_ai_rank_2"]),r["gate"])

def stats(r):
    n=int(r["after_tickets"])
    h=int(r["after_hits"])
    v=int(r["after_return_yen"])
    assert 0<=h<=n and v>=0
    return (n,h,v)

def profit(m):return m[2]-100*m[0]
def roi(m):return 100*m[2]/(100*m[0]) if m[0] else None

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-yearly",required=True)
    p.add_argument("--source-pooled",required=True)
    p.add_argument("--out-dir",required=True)
    a=p.parse_args()
    annual=read(a.source_yearly)
    pooled=read(a.source_pooled)
    assert len(pooled)==6870*8 and len(annual)==6870*8*3
    lookup={}
    for row in annual:
        yr=row["year"]; assert yr in YEARS
        ky=(yr,key(row))
        assert ky not in lookup,ky
        lookup[ky]=row
    assert len(lookup)==len(annual)
    all_ranked=[]
    all_year_red=Counter()
    sampled=Counter()
    by_threshold=Counter()
    for p_row in pooled:
        ky=key(p_row);gate=ky[-1]
        assert gate in GATES
        yrows=[lookup[(y,ky)] for y in YEARS]
        mm=[stats(r) for r in yrows]
        pool=stats(p_row)
        assert pool==tuple(sum(c[i] for c in mm) for i in range(3)), ("POOLED_PARITY_DRIFT",ky)
        if min(x[0] for x in mm) < MIN_PER_YEAR: continue
        sampled[gate]+=1
        if all(profit(x)<0 for x in mm):
            all_year_red[gate]+=1
        else:continue
        for t in THRESHOLDS:
            if all(roi(x)<t for x in mm):
                by_threshold[(gate,t)]+=1
        if not all(roi(x)<80 for x in mm):continue
        ev,fam,j,k,gate=ky
        partner=("なし" if j==0 else str(j)+"位" + ("→"+str(k)+"位" if k else ""))
        row={
            "期待値AI軸":ev,
            "券種":NAMES[fam],
            "純粋AI相手順位":partner,
            "削減条件":gate,
            "3年購入点数":pool[0],
            "3年的中点数":pool[1],
            "3年的中率_pct":round(100*pool[1]/pool[0],6),
            "3年投資円":pool[0]*100,
            "3年払戻円":pool[2],
            "3年損益円":profit(pool),
            "3年ROI_pct":round(roi(pool),6),
            "各年ROI最高_pct":round(max(roi(x) for x in mm),6),
            "各年ROI最低_pct":round(min(roi(x) for x in mm),6),
            **{f"{year}購入点数":m[0] for year,m in zip(YEARS,mm)},
            **{f"{year}的中点数":m[1] for year,m in zip(YEARS,mm)},
            **{f"{year}損益円":profit(m) for year,m in zip(YEARS,mm)},
            **{f"{year}ROI_pct":round(roi(m),6) for year,m in zip(YEARS,mm)},
        }
        assert all(row[f"{year}損益円"]<0 and row[f"{year}ROI_pct"]<80 for year in YEARS)
        all_ranked.append(row)
    assert sampled["ALL"] and all_year_red["ALL"] and all_ranked
    # Sorting-only, no filter tuning or new cuts.
    by_group=defaultdict(list)
    for row in all_ranked:by_group[row["削減条件"]].append(row)
    row_counts=[]
    for gate in GATES:
        for t in THRESHOLDS:
            row_counts.append({
                "削減条件":gate,
                "各年購入1000点以上":sampled[gate],
                "そのうち3年連続赤字":all_year_red[gate],
                "各年ROI上限_pct_未満":t,
                "条件一致設定数":by_threshold[(gate,t)]
            })
    full=sorted(all_ranked,
        key=lambda x:(x["各年ROI最高_pct"],x["3年ROI_pct"],-x["3年購入点数"]))
    byvolume=sorted(all_ranked,
        key=lambda x:(-x["3年購入点数"],x["3年ROI_pct"]))
    byroi=sorted(all_ranked,
        key=lambda x:(x["3年ROI_pct"],-x["3年購入点数"]))
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    write(out/"all-consistent-low-roi-losses-3yr.csv",full)
    write(out/"counts-by-gate-and-roi-ceiling.csv",row_counts)
    write(out/"top-100-all-by-volume.csv",byvolume[:100])
    write(out/"top-100-all-by-3yr-roi-ascending.csv",byroi[:100])
    write(out/"top-100-all-by-worst-annual-ceiling.csv",full[:100])
    for gate in GATES:
        arr=by_group[gate]
        if arr:
            write(out/f"top-100-{gate}-by-3yr-roi.csv",
                sorted(arr,key=lambda x:(x["3年ROI_pct"],-x["3年購入点数"]))[:100])
            write(out/f"top-100-{gate}-by-volume.csv",
                sorted(arr,key=lambda x:(-x["3年購入点数"],x["3年ROI_pct"]))[:100])
    summary={
        "contract":"L2_EV1_6_CONSISTENT_THREE_YEAR_NEGATIVE_HIGH_VOLUME_V1",
        "criterion":"at least 1000 post-prune 100-yen tickets in EACH 2023/24/25, each year profit <0 and ROI <80%",
        "threshold_purchases_each_year":MIN_PER_YEAR,
        "counts_by_gate_and_rois":row_counts,
        "selected_base_unpruned_all":len(by_group["ALL"]),
        "selected_odds_free_pruned_sum_not_unique":sum(len(by_group[g]) for g in GATES if g!="ALL"),
        "all_matching_rows":len(all_ranked),
        "top_baseline_most_consistently_low_roi":sorted(by_group["ALL"],
             key=lambda x:(x["各年ROI最高_pct"],x["3年ROI_pct"],-x["3年購入点数"]))[:25],
        "top_baseline_lowest_3yr_roi":sorted(by_group["ALL"],
             key=lambda x:(x["3年ROI_pct"],-x["3年購入点数"]))[:25],
        "top_baseline_highest_volume":sorted(by_group["ALL"],
             key=lambda x:(-x["3年購入点数"],x["3年ROI_pct"]))[:25],
        "top_pruned_most_consistently_low_roi":sorted(
             [x for x in all_ranked if x["削減条件"]!="ALL"],
             key=lambda x:(x["各年ROI最高_pct"],x["3年ROI_pct"],-x["3年購入点数"]))[:25],
        "top_pruned_highest_volume":sorted(
             [x for x in all_ranked if x["削減条件"]!="ALL"],
             key=lambda x:(-x["3年購入点数"],x["3年ROI_pct"]))[:25],
        "example_by_bet_family":{
          family:dict(Counter(x["券種"] for x in by_group[gate]))
          for family,gate in (("baseline","ALL"),("postprune_ai_prob","MODEL_WIN_P_GE_003"))
        },
        "source":"saved Run 39115499898 yearly and 3-year pooled exact-rank no-odds pruning results",
        "no_predictions_or_tickets_recomputed":True,
        "no_logic_or_frozen_L21_changed":True,
        "2026_sealed":True,
        "not_proof_of_future_loss":"All filters selected and evaluated retrospectively on the same 2023-25 years. EV axis uses historical final win odds. A named config under different gates is not a different physical ticket by definition.",
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("THREE_YEAR_HIGH_VOLUME_LOSSES_DONE",json.dumps({
        "n_baseline":len(by_group["ALL"]),
        "n_pruned":summary["selected_odds_free_pruned_sum_not_unique"],
        "by_gate":[{"gate":g,"n":len(by_group[g])} for g in GATES],
        "top5_base":summary["top_baseline_most_consistently_low_roi"][:5],
    },ensure_ascii=False),flush=True)

if __name__=="__main__":main()
