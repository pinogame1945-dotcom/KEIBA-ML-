#!/usr/bin/env python3
"""Attribute actual ticket-level profit to disjoint ticket-odds bands from frozen
2023-25 7-kind 638-template per-floor grid. No downloads/training/resettlement.
ROI denominator includes all losing purchased tickets within each odds band.

Price = archived quoted FINAL odds (lower end for PLACE/WIDE), independent of
winning payout; official normalized payouts determine cash return.
"""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

SRC=Path("research-results/l2-all-ticket-price-floor-v1/run-38685292226/all-bet-ticket-odds-floor-grid.csv.gz")
OUT=Path("research-results/l2-ticket-odds-band-profit-v1")
YEARS=("2023","2024","2025","2023-2025")
FLOORS=(0,1.5,2,3,5,7,10,15,20,30,50,75,100,150,200,300,500,1000,2000,5000)
BROADS=((0,2),(2,3),(3,5),(5,10),(10,20),(20,50),(50,100),
        (100,200),(200,500),(500,1000),(1000,2000),(2000,5000),(5000,float("inf")))
BENCH=(
    ("EV","SINGLE","0","WIN"),
    ("EV","SINGLE","0","PLACE"),
    ("EV","AI_PROB","2","QUINELLA"),
    ("EV","AI_PROB","2","WIDE"),
    ("EV","AI_PROB","2","EXACTA_FIRST"),
    ("EV","AI_PROB","2","EXACTA_SECOND"),
    ("EV","AI_PROB","2","EXACTA_BOTH"),
    ("EV","AI_PROB","2","TRIO"),
    ("EV","AI_PROB","2","TRIFECTA_FIRST"),
    ("EV","AI_PROB","2","TRIFECTA_SECOND"),
    ("EV","AI_PROB","2","TRIFECTA_THIRD"),
    ("EV","AI_PROB","2","TRIFECTA_ANY"),
    ("EV","MARKET","2","QUINELLA"),
    ("EV","MARKET","2","WIDE"),
    ("EV","MARKET","2","EXACTA_FIRST"),
    ("EV","MARKET","2","EXACTA_SECOND"),
    ("EV","MARKET","2","TRIO"),
    ("EV","MARKET","2","TRIFECTA_FIRST"),
    ("EV","AI_PROB","5","TRIO"),
    ("EV","AI_PROB","5","TRIFECTA_FIRST"),
    ("EV","AI_RAW","5","TRIO"),
    ("EV","AI_RAW","5","TRIFECTA_SECOND"),
)
FIX=("tickets","hit_tickets","stake_yen","return_yen")
def robust_float(v):
    return round(float(v),5)
def fmtband(lo,hi):
    return f"{lo:g}-{hi:g}" if hi!=float("inf") else f"{lo:g}+"
def row_out(key,lo,hi,n,hit,return_yen):
    yr,axis,order,k,fam=key
    cost=n*100
    return {"year":yr,"axis":axis,"order":order,"k":k,"family":fam,
            "odds_low":lo,"odds_high":None if hi==float("inf") else hi,
            "odds_band":fmtband(lo,hi),
            "tickets":n,"hit_tickets":hit,"stake_yen":cost,
            "return_yen":return_yen,"profit_yen":return_yen-cost,
            "roi_pct":round(return_yen/n,5) if n else None,
            "ticket_hit_pct":round(hit*100/n,5) if n else None}
def write_csv(path,rows,cols):
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore")
        w.writeheader();w.writerows(rows)
def main():
    assert SRC.exists(),"saved grid missing"
    by=defaultdict(dict)
    with gzip.open(SRC,"rt",encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            k=(r["year"],r["axis"],r["order"],r["k"],r["family"])
            x=float(r["price_floor"])
            assert x in FLOORS,(k,x)
            assert x not in by[k],("duplicate floor",k,x)
            by[k][x]=r
    assert len({k[1:] for k in by if k[0]=="2023-2025"})>=600,"not all templates loaded"
    OUT.mkdir(exist_ok=True,parents=True)
    fine=[]
    broad=[]
    differences=0
    for key,d in sorted(by.items()):
        def snap(threshold):
            r=d.get(threshold)
            return tuple(int(round(float(r[col]))) if r else 0 for col in FIX)
        old=snap(0)
        for i,a in enumerate(FLOORS):
            b=FLOORS[i+1] if i+1<len(FLOORS) else float("inf")
            curr=snap(a)
            nxt=snap(b) if b!=float("inf") else (0,0,0,0)
            n,hit,stake,ret=(a1-a2 for a1,a2 in zip(curr,nxt))
            assert n>=0 and hit>=0 and stake==n*100 and ret>=0,(key,a,b,curr,nxt)
            assert hit<=n,(key,a,b,hit,n)
            fine.append(row_out(key,a,b,n,hit,ret))
            differences+=n
        for a,b in BROADS:
            curr=snap(a)
            nxt=snap(b) if b!=float("inf") else (0,0,0,0)
            n,hit,stake,ret=(a1-a2 for a1,a2 in zip(curr,nxt))
            assert n>=0 and hit>=0 and stake==n*100 and ret>=0,(key,a,b,curr,nxt)
            broad.append(row_out(key,a,b,n,hit,ret))
        keyfine=[x for x in fine[-len(FLOORS):]]
        assert sum(v["tickets"] for v in keyfine)==old[0],"sum tickets drift"
        assert sum(v["hit_tickets"] for v in keyfine)==old[1],"sum hits drift"
        assert sum(v["return_yen"] for v in keyfine)==old[3],"sum returns drift"
        assert sum(v["tickets"] for v in broad[-len(BROADS):])==old[0],"broad tickets drift"
        assert sum(v["return_yen"] for v in broad[-len(BROADS):])==old[3],"broad cash drift"
    fields=list(broad[0].keys())
    evb=[x for x in broad if x["axis"]=="EV"]
    evf=[x for x in fine if x["axis"]=="EV"]
    bench=[x for x in evb if (x["axis"],x["order"],x["k"],x["family"]) in BENCH]
    write_csv(OUT/"ev-every-ticket-strategy-broad-odds-profit.csv",evb,fields)
    write_csv(OUT/"ev-every-ticket-strategy-fine-odds-profit.csv.gz.csv",evf,fields) if False else None
    with gzip.open(OUT/"ev-every-ticket-strategy-fine-odds-profit.csv.gz","wt",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader();w.writerows(evf)
    write_csv(OUT/"ev-representative-strategies-odds-profit.csv",bench,fields)
    # One ranking row per strategy/broad price band. Yearly consistency is
    # descriptive, NOT proof of profitability after selecting on those years.
    idx={(x["year"],x["axis"],x["order"],x["k"],x["family"],x["odds_band"]):x for x in evb}
    consistent=[]
    for x in evb:
        if x["year"]!="2023-2025" or x["tickets"]<500:continue
        z=[idx[(y,x["axis"],x["order"],x["k"],x["family"],x["odds_band"])] for y in YEARS[:3]]
        good=sum(v["roi_pct"] is not None and v["roi_pct"]>100 for v in z)
        r=dict(x,positive_years=good,
               roi_2023=z[0]["roi_pct"],roi_2024=z[1]["roi_pct"],
               roi_2025=z[2]["roi_pct"],
               wins_2023=z[0]["hit_tickets"],wins_2024=z[1]["hit_tickets"],wins_2025=z[2]["hit_tickets"])
        consistent.append(r)
    consistent.sort(key=lambda r:(-r["positive_years"],-r["profit_yen"],-r["tickets"]))
    write_csv(OUT/"ev-candidate-profitable-odds-bands-500-tickets-min.csv",consistent,list(consistent[0].keys()))
    # Pre-defined, not cherry-picked, ticket strategies. Classify odds bin
    # based on physical combination price, never selected horse win odds.
    for tpl in BENCH:
        arr=[r for r in bench if r["year"]=="2023-2025" and
             (r["axis"],r["order"],r["k"],r["family"])==tpl and r["tickets"]>0]
        if not arr:continue
        allcost=sum(r["stake_yen"] for r in arr)
        allret=sum(r["return_yen"] for r in arr)
        # Summarize all odds bands containing tickets; include losing bands.
        print("TICKET_PRICE_PROFIT_BENCH",json.dumps({
          "template":tpl[1:],"total_tickets":sum(r["tickets"] for r in arr),
          "total_hit_tickets":sum(r["hit_tickets"] for r in arr),
          "overall_roi_pct":round(100*allret/allcost,4),
          "bands":[{"band":r["odds_band"],"tickets":r["tickets"],
                    "hits":r["hit_tickets"],"profit":r["profit_yen"],
                    "roi":r["roi_pct"]} for r in arr]},
          ensure_ascii=False),flush=True)
    shortlist=[r for r in consistent if r["positive_years"]==3 and r["hit_tickets"]>=15
              and r["profit_yen"]>0 and r["tickets"]>=500]
    shortlist.sort(key=lambda r:-r["profit_yen"])
    print("EV_3YEAR_PROFIT_BANDS",json.dumps(shortlist[:35],ensure_ascii=False),flush=True)
    print("EV_TICKET_ODDS_BAND_COMPLETE",json.dumps({
        "fine_rows":len(evf),"broad_rows":len(evb),
        "total_strategies_all_axes":len({k[1:] for k in by if k[0]=="2023-2025"}),
        "total_ev_strategy_bands":len([k for k in by if k[0]=="2023-2025" and k[1]=="EV"]),
        "representative_templates":len(BENCH),"qualified_positive_3year":len(shortlist),
        "source":str(SRC),"2026_locked":True},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
