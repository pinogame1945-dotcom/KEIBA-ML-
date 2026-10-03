#!/usr/bin/env python3
import argparse,csv,gzip,hashlib,json,time
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import load_day,load_odds_day,iter_decoded_odds,payout_map,canonical_numbers

YEARS=(2023,2024,2025)
PRIMARY=(2024,2025)
CAP=30.0
TARGETS=(3.0,3.25,3.5)
METHODS=("KING","DISSENT_STRENGTH","CHEAP_ODDS")
BASE_PAIRS=((1,2),(1,3),(1,4),(2,3),(1,5))

def parse_args():
    p=argparse.ArgumentParser(description="Prune frozen five-point KING quinella set after <=30x cap toward 3.0/3.25/3.5 avg points.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def stable_bucket(year,rid):
    raw=f"{year}:{rid}".encode()
    n=int.from_bytes(hashlib.sha256(raw).digest()[:8],"big")
    return n/float(2**64)

def quota_for(target,year,rid):
    if target==3.0:
        return 3
    frac=target-3.0
    return 4 if stable_bucket(year,rid)<frac else 3

def load_market(path):
    rows={}
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf-8",newline="") as fh:
        r=csv.DictReader(fh)
        fields=set(r.fieldnames or [])
        need={"year","race_id","horse_number","consensus_rank","asymmetric_dissent_score"}
        missing=sorted(need-fields)
        if missing:
            raise SystemExit(f"market source missing required columns: {missing}")
        for row in r:
            y=int(float(row["year"]))
            if y not in YEARS: continue
            rid=str(row["race_id"])
            rank=int(float(row["consensus_rank"]))
            num=int(float(row["horse_number"]))
            raw=row.get("asymmetric_dissent_score")
            try:
                dissent=float(raw) if raw not in (None,"") else 0.0
            except ValueError:
                dissent=0.0
            d=rows.setdefault((y,rid),{})
            if rank in d:
                raise SystemExit(f"duplicate rank y={y} race={rid} rank={rank}")
            d[rank]={"num":num,"dissent":dissent}
    return rows

def method_key(method,cand):
    # cand: {pair, odds, payout, dissent_strength}
    p=cand["pair"]
    king=(p[0]+p[1],p[0],p[1])
    if method=="KING":
        return king
    if method=="DISSENT_STRENGTH":
        return (-cand["dissent_strength"],)+king
    if method=="CHEAP_ODDS":
        return (cand["odds"],)+king
    raise ValueError(method)

def max_drawdown(profits):
    eq=peak=mdd=0.0
    for p in profits:
        eq+=float(p)
        peak=max(peak,eq)
        mdd=max(mdd,peak-eq)
    return mdd

def write_csv(path,rows):
    if not rows:
        Path(path).write_text("",encoding="utf-8"); return
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def aggregate(rows,baseline,years,label,method,target):
    q=[x for x in rows if x["year"] in years and x["method"]==method and x["target"]==target]
    b=[x for x in baseline if x["year"] in years]
    if not q or not b: return None
    stake=sum(x["stake_yen"] for x in q); ret=sum(x["return_yen"] for x in q)
    b_stake=sum(x["stake_yen"] for x in b); b_ret=sum(x["return_yen"] for x in b)
    active=sum(x["tickets"]>0 for x in q); hits=sum(x["hit"] for x in q)
    b_hits=sum(x["hit"] for x in b)
    actual_retention=100.0*ret/b_ret if b_ret else None
    break_even_needed=100.0*stake/b_ret if b_ret else None
    profits=[x["profit_yen"] for x in sorted(q,key=lambda z:(z["year"],z["race_date"],z["race_id"]))]
    return {
        "period":label,"method":method,"target_avg_points":target,
        "races":len(q),"active_races":active,
        "active_race_rate_pct":100.0*active/len(q),
        "tickets":sum(x["tickets"] for x in q),
        "avg_points":sum(x["tickets"] for x in q)/len(q),
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "hit_races":hits,"hit_rate_pct":100.0*hits/len(q),
        "baseline_30cap_stake_yen":b_stake,
        "baseline_30cap_return_yen":b_ret,
        "baseline_30cap_roi_pct":100.0*b_ret/b_stake if b_stake else None,
        "baseline_30cap_hit_races":b_hits,
        "hit_retention_pct":100.0*hits/b_hits if b_hits else None,
        "return_retention_pct":actual_retention,
        "return_retention_needed_for_100roi_pct":break_even_needed,
        "retention_gap_to_100roi_pp":actual_retention-break_even_needed if actual_retention is not None else None,
        "extra_return_needed_yen":max(0.0,stake-ret),
        "max_drawdown_yen":max_drawdown(profits),
    }

def main():
    a=parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    market=load_market(a.market_scored)
    wanted={y:{rid for yy,rid in market if yy==y} for y in YEARS}
    root=Path(a.backfill_root)
    baseline=[]; rows=[]; audit=[]; counters=defaultdict(int)
    usable=defaultdict(set)

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
                rid=str(rid); mk=market.get((year,rid))
                if mk is None: continue
                if any(r not in mk for p in BASE_PAIRS for r in p):
                    counters["missing_required_rank"]+=1; continue
                orec=oddsday.get(rid)
                if orec is None:
                    counters["missing_race_odds_record"]+=1; continue

                qodds={}
                for bet,nums,price in iter_decoded_odds(orec):
                    if bet=="QUINELLA":
                        qodds[tuple(sorted(nums))]=float(price)
                payouts,present=payout_map(pack)
                if "QUINELLA" not in present:
                    counters["missing_quinella_payout"]+=1; continue

                five=[]
                complete=True
                for rp in BASE_PAIRS:
                    a1,b1=rp
                    ticket=canonical_numbers("QUINELLA",(mk[a1]["num"],mk[b1]["num"]))
                    odds=qodds.get(ticket)
                    if odds is None:
                        complete=False; break
                    five.append({
                        "pair":rp,
                        "ticket":ticket,
                        "odds":float(odds),
                        "payout":float(payouts.get(("QUINELLA",ticket),0.0)),
                        "dissent_strength":(abs(mk[a1]["dissent"])+abs(mk[b1]["dissent"]))/2.0,
                    })
                if not complete:
                    counters["incomplete_five_candidate_odds"]+=1; continue

                capped=[x for x in five if x["odds"]<=CAP]
                usable[year].add(rid)
                b_stake=100.0*len(capped); b_ret=sum(x["payout"] for x in capped)
                baseline.append({
                    "year":year,"race_date":date,"race_id":rid,
                    "tickets":len(capped),"stake_yen":b_stake,
                    "return_yen":b_ret,"profit_yen":b_ret-b_stake,"hit":int(b_ret>0),
                })

                for method in METHODS:
                    ordered=sorted(capped,key=lambda x:method_key(method,x))
                    for target in TARGETS:
                        quota=quota_for(target,year,rid)
                        chosen=ordered[:quota]
                        stake=100.0*len(chosen); ret=sum(x["payout"] for x in chosen)
                        rows.append({
                            "year":year,"race_date":date,"race_id":rid,
                            "method":method,"target":target,
                            "quota":quota,"tickets":len(chosen),
                            "stake_yen":stake,"return_yen":ret,
                            "profit_yen":ret-stake,"hit":int(ret>0),
                        })
                        audit.append({
                            "year":year,"race_id":rid,"method":method,"target":target,
                            "quota":quota,"capped_candidates":len(capped),
                            "chosen_pairs":"|".join(f"{x['pair'][0]}-{x['pair'][1]}" for x in chosen),
                            "chosen_odds":"|".join(f"{x['odds']:.3f}" for x in chosen),
                        })

        print("PRUNE_YEAR_DONE "+json.dumps({"year":year,"usable_races":len(usable[year])},separators=(",",":")),flush=True)

    summary=[]
    for y in YEARS:
        for method in METHODS:
            for target in TARGETS:
                summary.append(aggregate(rows,baseline,(y,),str(y),method,target))
    for method in METHODS:
        for target in TARGETS:
            summary.append(aggregate(rows,baseline,PRIMARY,"2024|2025",method,target))

    write_csv(out/"summary.csv",[x for x in summary if x])
    with gzip.open(out/"audit.csv.gz","wt",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(audit[0].keys()))
        w.writeheader(); w.writerows(audit)

    meta={
        "contract":"L2_30CAP_PRUNING_TARGETS_V1",
        "base_pairs":[list(x) for x in BASE_PAIRS],
        "base_ticket_gate":"final quinella odds <= 30.0",
        "methods":{
            "KING":"rank-sum ascending, then better rank",
            "DISSENT_STRENGTH":"pair mean absolute L1.75 asymmetric_dissent_score descending, then KING order",
            "CHEAP_ODDS":"lower final quinella odds first, then KING order",
        },
        "targets":list(TARGETS),
        "quota_policy":"3 tickets always; for 3.25/3.5, a stable pre-result SHA256(year:race_id) bucket gives a 4th ticket to ~25%/~50% of races. Same race quota assignment for every method.",
        "purpose":"Measure how much 30x-cap return can be retained while reducing average points, and compare actual return retention with retention required for ROI 100%.",
        "odds_source":"final odds; feasibility only, not production-time executable",
        "result_use":"payout only for evaluation; never for ticket ordering",
        "evaluation_years":list(YEARS),
        "primary_years":list(PRIMARY),
        "counters":dict(counters),
        "usable_races":{str(y):len(usable[y]) for y in YEARS},
        "elapsed_seconds":time.perf_counter()-started,
        "promotion":False,
        "2026_locked":True,
    }
    (out/"meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== META ====="); print((out/"meta.json").read_text())
    print("===== SUMMARY ====="); print((out/"summary.csv").read_text())
    print("L2_30CAP_PRUNING_TARGETS_V1_READY")

if __name__=="__main__":
    main()
