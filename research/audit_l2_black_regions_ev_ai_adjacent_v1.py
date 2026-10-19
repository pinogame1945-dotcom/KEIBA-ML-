#!/usr/bin/env python3
"""Audit 'black regions' around the pre-existing 134 three-year profitable L2 strategies.

Geometry: within same family and same *odds-free* gate, change exactly ONE
coordinate by +-1: EV axis rank, pure AI first partner rank, or pure AI second
partner rank (only for 3-horse bets). Valid existing configurations only.
Stable black means positive P&L with >=1 ticket in EACH of 2023/24/25.
A connected component of 2+ stable-black configs is a contiguous black region.

No model/horse/payout recalculation, no hindsight new profit optimization,
no changes to frozen L2.1, 2026 sealed.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import re
from collections import defaultdict, Counter, deque
from pathlib import Path

NAMES={"WIN":"単勝","PLACE":"複勝","QUINELLA":"馬連","WIDE":"ワイド",
       "EXACTA_FIRST":"馬単・軸1着","EXACTA_SECOND":"馬単・軸2着",
       "TRIO":"三連複","TRIFECTA_FIRST":"三連単・軸1着",
       "TRIFECTA_SECOND":"三連単・軸2着","TRIFECTA_THIRD":"三連単・軸3着"}
REVERSE={v:k for k,v in NAMES.items()}
DOUBLE={"QUINELLA","WIDE","EXACTA_FIRST","EXACTA_SECOND"}
TRIPLE={"TRIO","TRIFECTA_FIRST","TRIFECTA_SECOND","TRIFECTA_THIRD"}
YEARS=("2023","2024","2025")

def fail(condition,message):
    if not condition:raise AssertionError(message)
def load_gz_csv(path):
    with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))
def key(r):
    return (int(r["ev_axis_rank"]),r["family"],
            int(r["raw_ai_rank_1"]),int(r["raw_ai_rank_2"]),r["gate"])
def clean_rows(rows):
    return list(rows)
def save_csv(path,rows):
    rows=list(rows)
    if not rows:return
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
def n(r):return int(r["after_tickets"])
def hits(r):return int(r["after_hits"])
def pay(r):return int(r["after_return_yen"])
def profit(r):return pay(r)-n(r)*100
def roipct(r):return round(pay(r)/n(r),6) if n(r) else None
def human(k):
    ev,fam,j,t,gate=k
    s="-" if not j else str(j) + ("→"+str(t) if t else "")
    return {"EV軸順位":ev,"券種":NAMES[fam],"純AI相手順位":s,"削減条件":gate}
def valid(k):
    e,f,j,t,_=k
    if not 1<=e<=6:return False
    if f in ("WIN","PLACE"):return j==t==0
    if f in DOUBLE:return 1<=j<=18 and t==0
    if f=="TRIO":return 1<=j<t<=18
    if f in TRIPLE:return 1<=j<=18 and 1<=t<=18 and j!=t
    return False

def offsets(k):
    ev,f,j,t,gate=k
    possible=[]
    for di in (-1,1):
        x=(ev+di,f,j,t,gate)
        if valid(x):possible.append(("EV軸"+("−1" if di<0 else "＋1"),x))
    if f not in ("WIN","PLACE"):
        for di in (-1,1):
            x=(ev,f,j+di,t,gate)
            if valid(x):possible.append(("純AI相手1"+("−1" if di<0 else "＋1"),x))
    if f in TRIPLE:
        for di in (-1,1):
            x=(ev,f,j,t+di,gate)
            if valid(x):possible.append(("純AI相手2"+("−1" if di<0 else "＋1"),x))
    return possible

def main():
    p=argparse.ArgumentParser()
    for a in ("selected-134","pruned-yearly","pruned-pooled","out-dir"):
        p.add_argument("--"+a,required=True)
    args=p.parse_args()
    yearly=load_gz_csv(args.pruned_yearly)
    pooled=load_gz_csv(args.pruned_pooled)
    fail(len(yearly)==6870*8*3 and len(pooled)==6870*8,
         "unmodified reference full universe count drift")
    ymap={}
    for r in yearly:
        lookup=(r["year"],key(r))
        fail(lookup not in ymap,"duplicate annual index")
        ymap[lookup]=r
    pmap={}
    for r in pooled:
        k=key(r);fail(k not in pmap,"duplicate pooled key")
        pmap[k]=r
    fail(set(ymap)=={(yr,k) for k in pmap for yr in YEARS},
         "annual/pool missing unmatched configurations")
    for k,row in pmap.items():
        ar=[ymap[(yr,k)] for yr in YEARS]
        fail(n(row)==sum(n(r) for r in ar) and
             hits(row)==sum(hits(r) for r in ar) and
             pay(row)==sum(pay(r) for r in ar),
             "annual vs 3-year base figures fail parity "+str(k))
    with open(args.selected_134,encoding="utf-8-sig",newline="") as f:
        original=list(csv.DictReader(f))
    fail(len(original)==134,"selected 134 CSV count drift")
    stable=set()
    for row in original:
        ev=int(row["期待値AI軸順位"])
        j,knum=map(int,re.findall(r"(\d+)位",row["純粋AI相手順位"])) if (
            "→" in row["純粋AI相手順位"]) else (
            (int(re.findall(r"(\d+)位",row["純粋AI相手順位"])[0]),0)
            if re.search(r"\d+位",row["純粋AI相手順位"]) else (0,0))
        ky=(ev,REVERSE[row["券種と軸位置"]],j,knum,row["削減条件"])
        fail(ky in pmap and valid(ky),"134 CSV includes unknown settings "+str(ky))
        fail(ky not in stable,"duplicate 134 selection")
        fail(profit(pmap[ky])>0 and all(profit(ymap[(yr,ky)])>0 and
             n(ymap[(yr,ky)])>0 for yr in YEARS),
             "historical 134 selection is no longer annually black "+str(ky))
        stable.add(ky)
    allstable={k for k,r in pmap.items() if k[-1]!="ALL" and
             profit(r)>0 and all(profit(ymap[(yr,k)])>0 and
             n(ymap[(yr,k)])>0 for yr in YEARS)}
    fail(stable==allstable,"all stable-3-year odd-free configs differ from pinned 134")
    selected_summary=[]
    neighbor_rows=[]
    linked=defaultdict(set)
    for k in sorted(stable):
        refs=offsets(k)
        have=[]; ev_have=[]
        for change,adj in refs:
            if adj not in pmap:continue
            qr=pmap[adj]
            yy=[ymap[(y,adj)] for y in YEARS]
            black=adj in stable
            status=("BLACK_ALL_THREE_YEARS" if black else
                    "BLACK_3YR_TOTAL_ONLY" if profit(qr)>0 else
                    "NO_BETS" if n(qr)==0 else
                    "ZERO_PROFIT" if profit(qr)==0 else "RED_3YR_TOTAL")
            rr={"変更方向":change,
                **{("中心_"+a):b for a,b in human(k).items()},
                **{("隣接_"+a):b for a,b in human(adj).items()},
                "中心3年ROI_pct":roipct(pmap[k]),
                "隣接3年ROI_pct":roipct(qr),
                "中心3年購入":n(pmap[k]),
                "隣接3年購入":n(qr),
                "中心3年的中":hits(pmap[k]),
                "隣接3年的中":hits(qr),
                "隣接3年利益円":profit(qr),
                "隣接判定":status,
                **{y+"隣接ROI_pct":roipct(a) for y,a in zip(YEARS,yy)},
                **{y+"隣接購入":n(a) for y,a in zip(YEARS,yy)},
                **{y+"隣接的中":hits(a) for y,a in zip(YEARS,yy)},
                **{y+"隣接損益円":profit(a) for y,a in zip(YEARS,yy)}}
            neighbor_rows.append(rr)
            have.append((change,adj,status))
            if change.startswith("EV軸"):ev_have.append((change,adj,status))
            if black:
                linked[k].add(adj)
                linked[adj].add(k)
        yy=[ymap[(y,k)] for y in YEARS]
        black_n=sum(status=="BLACK_ALL_THREE_YEARS" for _,_,status in have)
        black_ev=sum(status=="BLACK_ALL_THREE_YEARS" for _,_,status in ev_have)
        pooled_black=sum(status in ("BLACK_ALL_THREE_YEARS","BLACK_3YR_TOTAL_ONLY")
                         for _,_,status in have)
        red_n=sum(status=="RED_3YR_TOTAL" for _,_,status in have)
        selected_summary.append({
            **human(k),"3年ROI_pct":roipct(pmap[k]),
            "3年利益円":profit(pmap[k]),
            "3年購入":n(pmap[k]),
            "3年的中":hits(pmap[k]),
            "3年的中率_pct":round(100*hits(pmap[k])/n(pmap[k]),6),
            **{y+"ROI_pct":roipct(r) for y,r in zip(YEARS,yy)},
            **{y+"購入":n(r) for y,r in zip(YEARS,yy)},
            **{y+"的中":hits(r) for y,r in zip(YEARS,yy)},
            **{y+"利益円":profit(r) for y,r in zip(YEARS,yy)},
            "3年毎年的中>=1":int(all(hits(r)>=1 for r in yy)),
            "3年毎年的中>=2":int(all(hits(r)>=2 for r in yy)),
            "3年毎年的中>=3":int(all(hits(r)>=3 for r in yy)),
            "隣接可能数":len(have),
            "隣接3年連続黒字数":black_n,
            "隣接EV軸3年連続黒字数":black_ev,
            "隣接3年合算黒字数":pooled_black,
            "隣接3年合算赤字数":red_n,
            "周辺3年連続黒字率_pct":round(100*black_n/len(have),6) if have else None,
            "_key":k
        })
    # Connected components within each same family and gate. Each component
    # of size >=2 is a contiguous region; 1 is an isolated historical black.
    remaining=set(stable)
    members=[];regions=[]
    components=[]
    while remaining:
        seed=min(remaining);queue=deque([seed]);remaining.remove(seed)
        component=[]
        while queue:
            cur=queue.popleft()
            component.append(cur)
            for adjacent in linked[cur]:
                if adjacent in remaining:
                    remaining.remove(adjacent);queue.append(adjacent)
        components.append(sorted(component))
    components.sort(key=lambda xs:(-len(xs),xs[0]))
    for rid,comp in enumerate(components,1):
        allstats=[x for x in selected_summary if x["_key"] in comp]
        primary=comp[0]
        regions.append({
            "領域ID":rid,"連続黒字設定数":len(comp),
            "券種":NAMES[primary[1]],"削減条件":primary[-1],
            "EV軸順位範囲":str(min(x[0] for x in comp))+"-"+str(max(x[0] for x in comp)),
            "純AI相手1順位範囲":str(min(x[2] for x in comp))+"-"+str(max(x[2] for x in comp)),
            "純AI相手2順位範囲":str(min(x[3] for x in comp))+"-"+str(max(x[3] for x in comp)),
            "3年の計算上ROI最小_pct":min(x["3年ROI_pct"] for x in allstats),
            "3年の計算上ROI最大_pct":max(x["3年ROI_pct"] for x in allstats),
            "各年1回以上的中設定数":sum(x["3年毎年的中>=1"] for x in allstats),
            "各年2回以上的中設定数":sum(x["3年毎年的中>=2"] for x in allstats),
            "各年3回以上的中設定数":sum(x["3年毎年的中>=3"] for x in allstats),
            "設定ごとの3年購入単純合計_重複未除去":sum(x["3年購入"] for x in allstats),
            "設定ごとの3年利益単純合計_重複未除去円":sum(x["3年利益円"] for x in allstats),
        })
        for k in comp:
            v=next(x for x in allstats if x["_key"]==k)
            members.append({
                "領域ID":rid,"領域サイズ":len(comp),
                **{a:b for a,b in v.items() if not a.startswith("_")}})
    agg=Counter()
    for x in selected_summary:
        v=x["隣接3年連続黒字数"]
        agg["isolated" if v==0 else "1_stable_neighbor" if v==1 else "2plus_stable_neighbors"]+=1
        agg["EV_axis_neighbor_stable" if x["隣接EV軸3年連続黒字数"] else "EV_axis_neighbor_not_stable"]+=1
        for i in (1,2,3):
            if x[f"3年毎年的中>={i}"]:agg[f"hits_at_least_{i}_each_year"]+=1
        if v>=2 and x["3年毎年的中>=2"]:
            agg["at_least_2_black_neighbors_and_2_hits_each_year"]+=1
    components_size=Counter(len(xs) for xs in components)
    best=sorted(selected_summary,key=lambda x:(
          -x["隣接3年連続黒字数"],-x["隣接EV軸3年連続黒字数"],
          -x["3年毎年的中>=2"],-x["3年的中"],-x["3年購入"]))
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    clean=lambda rows:[{a:b for a,b in r.items() if not a.startswith("_")} for r in rows]
    save_csv(out/"all-134-black-area-neighbor-metrics.csv",clean(best))
    save_csv(out/"all-134-adjacent-rank-comparisons.csv",neighbor_rows)
    save_csv(out/"black-area-connected-components.csv",regions)
    save_csv(out/"black-area-component-members.csv",members)
    save_csv(out/"top-100-by-neighbor-stability.csv",clean(best[:100]))
    robust=[x for x in best if x["隣接3年連続黒字数"]>=2 and
            x["3年毎年的中>=2"]]
    if robust:save_csv(out/"multi-neighbor-black-and-annual-hit-stable.csv",clean(robust))
    summary={
      "contract":"L2_EV1_6_ODDS_FREE_BLACK_REGIONS_3YEAR_STABLE_V1",
      "selection_source":"exact previously extracted 134 all-3-year-black configurations",
      "neighbor_definition":"exactly one of EV axis rank, AI partner rank1, AI partner rank2 changes by +/-1, same family and same gate; unordered TRIO j<k, trifecta j!=k",
      "black_region_definition":"connected graph of two or more 3-year individually profitable configs",
      "stable_134_count":len(stable),
      "settings_classification":dict(agg),
      "connected_component_sizes":dict(sorted(components_size.items())),
      "number_black_regions_size_2plus":sum(len(c)>1 for c in components),
      "settings_in_black_regions_size_2plus":sum(len(c) for c in components if len(c)>1),
      "max_component_size":max(map(len,components)),
      "region_top_25":regions[:25],
      "top_25_by_black_neighbors":clean(best[:25]),
      "multi_black_neighbors_and_annual_hits_ge_2_count":len(robust),
      "multi_black_neighbors_and_annual_hits_ge_2_top":clean(robust[:25]),
      "annual_years":list(YEARS),
      "source_data_recalculated":False,
      "no_exotic_odds_pruning":True,
      "no_frozen_L21_change":True,
      "2026_sealed":True,
      "risk":"Retrospective in-sample black-spot selection; neighboring EV ranks are different horses, not small perturbations in model scores; profits are per-setting, not deduplicated portfolio."
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("BLACK_AREAS_134_COMPLETE",json.dumps({
          k:summary[k] for k in ("stable_134_count","settings_classification",
          "connected_component_sizes","number_black_regions_size_2plus",
          "max_component_size","multi_black_neighbors_and_annual_hits_ge_2_count")},
          ensure_ascii=False),flush=True)

if __name__=="__main__":main()
