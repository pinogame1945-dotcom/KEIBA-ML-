#!/usr/bin/env python3
import argparse,csv,gzip,json,math,time
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import (
    load_day,load_odds_day,iter_decoded_odds,payout_map,canonical_numbers
)

YEARS=(2023,2024,2025)
DEV_YEARS=(2023,2024)
HOLDOUT_YEAR=2025
CAP=30.0

BASE_PAIRS=((1,2),(1,3),(1,4),(2,3),(1,5))
KEEP_PAIRS=((1,2),(1,3),(1,4),(2,3))
ROUTES=("BASE","OUTSIDER_SWAP","L175_SWAP","AGREE_SWAP")

def parse_args():
    p=argparse.ArgumentParser(description="One-swap external candidate benchmark on frozen KING five-point quinella structure.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--ballots-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_year_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1)
        out[int(y)]=p
    if set(out)!=set(YEARS):
        raise SystemExit(f"ballot years mismatch: {sorted(out)}")
    return out

def finite(v,default=0.0):
    try:
        x=float(v)
        return x if math.isfinite(x) else default
    except (TypeError,ValueError):
        return default

def load_market(path):
    required={
        "year","race_id","horse_id","horse_number","consensus_rank",
        "market_rank","rank_gap","direction","asymmetric_dissent_score"
    }
    rows={}
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf-8",newline="") as fh:
        r=csv.DictReader(fh)
        fields=set(r.fieldnames or [])
        miss=sorted(required-fields)
        if miss: raise SystemExit(f"market source missing columns: {miss}")
        for row in r:
            y=int(float(row["year"]))
            if y not in YEARS: continue
            rid=str(row["race_id"])
            rank=int(float(row["consensus_rank"]))
            item={
                "horse_id":str(row["horse_id"]),
                "num":int(float(row["horse_number"])),
                "rank":rank,
                "market_rank":int(float(row["market_rank"])),
                "rank_gap":finite(row["rank_gap"]),
                "direction":str(row["direction"]),
                "dissent":finite(row["asymmetric_dissent_score"],0.0),
            }
            d=rows.setdefault((y,rid),{})
            if rank in d: raise SystemExit(f"duplicate KING rank y={y} race={rid} rank={rank}")
            d[rank]=item
    return rows

def load_outsider_votes(paths,market):
    # Exact safe 13-Outsider weighted council:
    # top1=3, top2=2, top3=1; support count and vote-position counts retained.
    market_hids={}
    for key,rankmap in market.items():
        market_hids[key]={x["horse_id"]:x["num"] for x in rankmap.values()}
    votes={}
    candidates=defaultdict(set)
    for year,path in sorted(paths.items()):
        temp=defaultdict(lambda:defaultdict(lambda:{"weighted":0,"support":0,"v1":0,"v2":0,"v3":0}))
        with gzip.open(path,"rt",encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rid=str(row["race_id"]); cand=str(row["candidate"])
                candidates[(year,rid)].add(cand)
                seen=set()
                for k,w in ((1,3),(2,2),(3,1)):
                    hid=str(row.get(f"top{k}_horse_id") or "")
                    if not hid: continue
                    if hid in seen: raise SystemExit(f"duplicate candidate horse y={year} race={rid} cand={cand}")
                    seen.add(hid)
                    d=temp[rid][hid]
                    d["weighted"]+=w; d["support"]+=1; d[f"v{k}"]+=1
        for (yy,rid),hmap in market_hids.items():
            if yy!=year: continue
            if len(candidates[(year,rid)])!=13:
                raise SystemExit(f"expected 13 outsider candidates y={year} race={rid}")
            out={}
            for hid,num in hmap.items():
                d=temp[rid].get(hid,{"weighted":0,"support":0,"v1":0,"v2":0,"v3":0})
                out[num]={
                    "horse_id":hid,"num":num,
                    "weighted":int(d["weighted"]),"support":int(d["support"]),
                    "v1":int(d["v1"]),"v2":int(d["v2"]),"v3":int(d["v3"]),
                }
            votes[(year,rid)]=out
    return votes

def outsider_candidate(rankmap,ovotes):
    excluded={rankmap[r]["num"] for r in range(1,6) if r in rankmap}
    cand=[x for x in ovotes.values() if x["num"] not in excluded and x["weighted"]>0]
    if not cand: return None
    cand.sort(key=lambda x:(-x["weighted"],-x["v1"],-x["support"],-x["v2"],-x["v3"],x["num"],x["horse_id"]))
    return cand[0]["num"]

def l175_candidate(rankmap):
    cand=[]
    for rank,x in rankmap.items():
        if rank<=5: continue
        if x["direction"]!="L1_UPGRADE" or x["rank_gap"]<2 or x["dissent"]<=0: continue
        cand.append(x)
    if not cand: return None
    cand.sort(key=lambda x:(-x["dissent"],-x["rank_gap"],x["rank"],x["num"],x["horse_id"]))
    return cand[0]["num"]

def agree_candidate(rankmap,ovotes):
    excluded={rankmap[r]["num"] for r in range(1,6) if r in rankmap}
    bynum={x["num"]:x for x in rankmap.values()}
    cand=[]
    for num,o in ovotes.items():
        x=bynum.get(num)
        if x is None or num in excluded: continue
        if o["weighted"]<=0: continue
        if x["direction"]!="L1_UPGRADE" or x["rank_gap"]<2 or x["dissent"]<=0: continue
        cand.append((o,x))
    if not cand: return None
    cand.sort(key=lambda z:(
        -z[0]["weighted"],-z[0]["v1"],-z[0]["support"],
        -z[1]["dissent"],-z[1]["rank_gap"],z[1]["rank"],z[1]["num"],z[1]["horse_id"]
    ))
    return cand[0][1]["num"]

def structural_tickets(route,rankmap,ovotes):
    for r in range(1,6):
        if r not in rankmap: return None,None,True
    keep=[canonical_numbers("QUINELLA",(rankmap[a]["num"],rankmap[b]["num"])) for a,b in KEEP_PAIRS]
    original=canonical_numbers("QUINELLA",(rankmap[1]["num"],rankmap[5]["num"]))
    fallback=False
    alt_num=None
    if route=="BASE":
        fifth=original
    else:
        if route=="OUTSIDER_SWAP": alt_num=outsider_candidate(rankmap,ovotes)
        elif route=="L175_SWAP": alt_num=l175_candidate(rankmap)
        elif route=="AGREE_SWAP": alt_num=agree_candidate(rankmap,ovotes)
        else: raise ValueError(route)
        if alt_num is None or alt_num==rankmap[1]["num"]:
            fifth=original; fallback=True
        else:
            fifth=canonical_numbers("QUINELLA",(rankmap[1]["num"],alt_num))
            if fifth in keep:
                fifth=original; fallback=True
    ts=keep+[fifth]
    if len(ts)!=5 or len(set(ts))!=5:
        raise SystemExit(f"ticket uniqueness drift route={route}")
    return ts,alt_num,fallback

def max_drawdown(profits):
    eq=peak=mdd=0.0
    for p in profits:
        eq+=float(p); peak=max(peak,eq); mdd=max(mdd,peak-eq)
    return mdd

def write_csv(path,rows):
    if not rows:
        Path(path).write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def aggregate(rows,years,period,route):
    q=[x for x in rows if x["year"] in years and x["route"]==route]
    if not q: return None
    stake=sum(x["stake_yen"] for x in q); ret=sum(x["return_yen"] for x in q)
    base=[x for x in rows if x["year"] in years and x["route"]=="BASE"]
    b_stake=sum(x["stake_yen"] for x in base); b_ret=sum(x["return_yen"] for x in base)
    profits=[x["profit_yen"] for x in sorted(q,key=lambda z:(z["year"],z["race_date"],z["race_id"]))]
    return {
        "period":period,"route":route,"races":len(q),
        "active_races":sum(x["tickets"]>0 for x in q),
        "active_race_rate_pct":100.0*sum(x["tickets"]>0 for x in q)/len(q),
        "tickets":sum(x["tickets"] for x in q),
        "avg_points":sum(x["tickets"] for x in q)/len(q),
        "hit_races":sum(x["hit"] for x in q),
        "hit_rate_pct":100.0*sum(x["hit"] for x in q)/len(q),
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(profits),
        "fallback_races":sum(x["fallback"] for x in q),
        "fallback_rate_pct":100.0*sum(x["fallback"] for x in q)/len(q),
        "new_rescue_races":sum(x["new_rescue"] for x in q),
        "broken_hit_races":sum(x["broken_hit"] for x in q),
        "both_hit_races":sum(x["both_hit"] for x in q),
        "return_delta_vs_base_yen":ret-b_ret,
        "stake_delta_vs_base_yen":stake-b_stake,
        "roi_delta_vs_base_pp":(100.0*ret/stake-100.0*b_ret/b_stake) if stake and b_stake else None,
    }

def main():
    a=parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    market=load_market(a.market_scored)
    ovotes=load_outsider_votes(parse_year_paths(a.ballots_year),market)
    if set(market)!=set(ovotes): raise SystemExit("market/outsider coverage mismatch")
    wanted={y:{rid for yy,rid in market if yy==y} for y in YEARS}
    root=Path(a.backfill_root)
    rows=[]; audit=[]; counters=defaultdict(int); seen=defaultdict(set)

    for year in YEARS:
        for day_path in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
            date=day_path.name[:10]
            odds_path=root/"data"/"odds"/"daily"/f"{date}.jsonl.gz"
            if not odds_path.exists():
                counters[f"missing_odds_day_{year}"]+=1; continue
            day=load_day(day_path,wanted[year])
            if not day: continue
            oddsday=load_odds_day(odds_path,set(day))
            for rid,pack in day.items():
                rid=str(rid); key=(year,rid)
                rankmap=market.get(key); vote=ovotes.get(key)
                if rankmap is None or vote is None: continue
                oddsrec=oddsday.get(rid)
                if oddsrec is None:
                    counters["missing_race_odds_record"]+=1; continue
                payouts,present=payout_map(pack)
                if "QUINELLA" not in present:
                    counters["missing_quinella_payout"]+=1; continue
                qodds={}
                for bet,nums,price in iter_decoded_odds(oddsrec):
                    if bet=="QUINELLA": qodds[tuple(sorted(nums))]=float(price)

                built={}
                complete=True
                for route in ROUTES:
                    ts,alt_num,fallback=structural_tickets(route,rankmap,vote)
                    if ts is None:
                        complete=False; break
                    if any(t not in qodds for t in ts):
                        complete=False; break
                    chosen=[t for t in ts if qodds[t]<=CAP]
                    ret=sum(float(payouts.get(("QUINELLA",t),0.0)) for t in chosen)
                    built[route]={
                        "structural":ts,"chosen":chosen,"ret":ret,
                        "alt_num":alt_num,"fallback":fallback,
                    }
                if not complete:
                    counters["incomplete_route_ticket_odds"]+=1; continue

                seen[year].add(rid)
                base_hit=int(built["BASE"]["ret"]>0)
                for route in ROUTES:
                    b=built[route]
                    hit=int(b["ret"]>0); tickets=len(b["chosen"]); stake=100.0*tickets
                    new_rescue=int(route!="BASE" and base_hit==0 and hit==1)
                    broken=int(route!="BASE" and base_hit==1 and hit==0)
                    both=int(route!="BASE" and base_hit==1 and hit==1)
                    rows.append({
                        "year":year,"race_date":date,"race_id":rid,"route":route,
                        "tickets":tickets,"stake_yen":stake,"return_yen":b["ret"],
                        "profit_yen":b["ret"]-stake,"hit":hit,
                        "fallback":int(b["fallback"]),
                        "new_rescue":new_rescue,"broken_hit":broken,"both_hit":both,
                    })
                    audit.append({
                        "year":year,"race_date":date,"race_id":rid,"route":route,
                        "alt_horse_number":b["alt_num"] if b["alt_num"] is not None else "",
                        "fallback":int(b["fallback"]),
                        "structural_tickets":"|".join(f"{a:02d}-{bb:02d}" for a,bb in b["structural"]),
                        "bought_tickets":"|".join(f"{a:02d}-{bb:02d}" for a,bb in b["chosen"]),
                        "bought_odds":"|".join(f"{qodds[t]:.3f}" for t in b["chosen"]),
                        "return_yen":b["ret"],"hit":hit,
                        "base_hit":base_hit,"new_rescue":new_rescue,"broken_hit":broken,
                    })
        print("ONE_SWAP_YEAR_DONE "+json.dumps({"year":year,"races":len(seen[year])},separators=(",",":")),flush=True)

    summary=[]
    for y in YEARS:
        for route in ROUTES:
            summary.append(aggregate(rows,(y,),str(y),route))
    for route in ROUTES:
        summary.append(aggregate(rows,DEV_YEARS,"2023|2024_DEV",route))
        summary.append(aggregate(rows,(HOLDOUT_YEAR,),"2025_HOLDOUT",route))

    dev=[x for x in summary if x and x["period"]=="2023|2024_DEV"]
    selected=sorted(dev,key=lambda x:(-x["roi_pct"],-x["hit_rate_pct"],x["route"]))[0]
    holdout=next(x for x in summary if x and x["period"]=="2025_HOLDOUT" and x["route"]==selected["route"])
    base_dev=next(x for x in summary if x and x["period"]=="2023|2024_DEV" and x["route"]=="BASE")
    base_hold=next(x for x in summary if x and x["period"]=="2025_HOLDOUT" and x["route"]=="BASE")
    decision={
        "selected_route":selected["route"],
        "selection_period":"2023|2024_DEV",
        "selection_rule":"highest ROI among four predeclared routes; tie by higher hit rate then route name",
        "dev":{
            "base_roi_pct":base_dev["roi_pct"],"selected_roi_pct":selected["roi_pct"],
            "roi_delta_pp":selected["roi_pct"]-base_dev["roi_pct"],
            "new_rescue_races":selected["new_rescue_races"],
            "broken_hit_races":selected["broken_hit_races"],
            "return_delta_vs_base_yen":selected["return_delta_vs_base_yen"],
            "fallback_rate_pct":selected["fallback_rate_pct"],
        },
        "holdout_2025":{
            "base_roi_pct":base_hold["roi_pct"],"selected_roi_pct":holdout["roi_pct"],
            "roi_delta_pp":holdout["roi_pct"]-base_hold["roi_pct"],
            "base_hit_rate_pct":base_hold["hit_rate_pct"],
            "selected_hit_rate_pct":holdout["hit_rate_pct"],
            "new_rescue_races":holdout["new_rescue_races"],
            "broken_hit_races":holdout["broken_hit_races"],
            "return_delta_vs_base_yen":holdout["return_delta_vs_base_yen"],
            "stake_delta_vs_base_yen":holdout["stake_delta_vs_base_yen"],
            "fallback_rate_pct":holdout["fallback_rate_pct"],
        }
    }

    write_csv(out/"summary.csv",[x for x in summary if x])
    with gzip.open(out/"audit.csv.gz","wt",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(audit[0].keys()))
        w.writeheader(); w.writerows(audit)
    (out/"decision.json").write_text(json.dumps(decision,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    meta={
        "contract":"L2_ONE_SWAP_EXTERNAL_V1",
        "baseline_pairs":[list(x) for x in BASE_PAIRS],
        "kept_pairs":[list(x) for x in KEEP_PAIRS],
        "replaced_pair":[1,5],
        "routes":{
            "BASE":"original 1-5 fifth ticket",
            "OUTSIDER_SWAP":"KING1 x highest safe Outsider weighted-council horse outside KING Top5",
            "L175_SWAP":"KING1 x strongest positive L1_UPGRADE asymmetric_dissent horse outside KING Top5",
            "AGREE_SWAP":"KING1 x horse outside KING Top5 with both positive Outsider support and positive L1_UPGRADE dissent; prioritize Outsider support then dissent",
        },
        "fallback_policy":"If external candidate is unavailable or duplicates a kept ticket, retain original KING1-KING5 ticket.",
        "odds_gate":"final quinella odds <= 30.0 applied identically after each five-ticket structural route is built",
        "odds_source":"final odds; feasibility only, not production-time executable",
        "selection_uses_years":list(DEV_YEARS),
        "holdout_year":HOLDOUT_YEAR,
        "payout_use":"evaluation only; never used for candidate choice",
        "threshold_search":False,
        "ml_training":False,
        "counters":dict(counters),
        "usable_races":{str(y):len(seen[y]) for y in YEARS},
        "elapsed_seconds":time.perf_counter()-started,
        "promotion":False,
        "2026_locked":True,
    }
    (out/"meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== META ====="); print((out/"meta.json").read_text())
    print("===== SUMMARY ====="); print((out/"summary.csv").read_text())
    print("===== DECISION ====="); print((out/"decision.json").read_text())
    print("L2_ONE_SWAP_EXTERNAL_V1_READY")

if __name__=="__main__":
    main()
