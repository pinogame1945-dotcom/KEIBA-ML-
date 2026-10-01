#!/usr/bin/env python3
import argparse,csv,gzip,json,time
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import load_day,load_odds_day,iter_decoded_odds,payout_map,canonical_numbers

YEARS=(2023,2024,2025)
DEV_YEARS=(2023,2024)
HOLDOUT_YEAR=2025
THRESHOLDS=(1.0,3.0,5.0,7.0,10.0)

# Frozen 5-point structure from prior compression study.
BASE_PAIRS=((1,2),(1,3),(1,4),(2,3),(1,5))
MIN_ACTIVE_RACE_RATE_PCT=90.0
MIN_AVG_TICKETS=2.0

def parse_args():
    p=argparse.ArgumentParser(description="Final-odds feasibility test: drop cheap quinella tickets from frozen KING 5-point set.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def load_king(path):
    out={}
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf-8",newline="") as fh:
        r=csv.DictReader(fh)
        for row in r:
            y=int(float(row["year"]))
            if y not in YEARS: continue
            rid=str(row["race_id"])
            rank=int(float(row["consensus_rank"]))
            num=int(float(row["horse_number"]))
            d=out.setdefault((y,rid),{})
            if rank in d and d[rank]!=num:
                raise SystemExit(f"duplicate rank y={y} race={rid} rank={rank}")
            d[rank]=num
    return out

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

def aggregate(rows,years,label,threshold):
    q=[x for x in rows if x["year"] in years and x["threshold"]==threshold]
    if not q: return None
    active=sum(x["tickets"]>0 for x in q)
    stake=sum(x["stake_yen"] for x in q)
    ret=sum(x["return_yen"] for x in q)
    profits=[x["profit_yen"] for x in sorted(q,key=lambda z:(z["year"],z["race_date"],z["race_id"]))]
    hits=sum(x["hit"] for x in q)
    total_possible=5*len(q)
    return {
        "period":label,
        "threshold":threshold,
        "races":len(q),
        "active_races":active,
        "active_race_rate_pct":100.0*active/len(q),
        "tickets":sum(x["tickets"] for x in q),
        "avg_tickets":sum(x["tickets"] for x in q)/len(q),
        "ticket_retention_pct":100.0*sum(x["tickets"] for x in q)/total_possible,
        "zero_ticket_races":len(q)-active,
        "hit_races":hits,
        "hit_rate_all_races_pct":100.0*hits/len(q),
        "hit_rate_active_pct":100.0*hits/active if active else None,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(profits),
    }

def main():
    a=parse_args()
    outdir=Path(a.out_dir); outdir.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    king=load_king(a.market_scored)
    wanted={y:{rid for yy,rid in king if yy==y} for y in YEARS}
    root=Path(a.backfill_root)
    rows=[]; ticket_rows=[]; counters=defaultdict(int)
    usable_races=defaultdict(set)

    for year in YEARS:
        for day_path in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
            date=day_path.name[:10]
            odds_path=root/"data"/"odds"/"daily"/f"{date}.jsonl.gz"
            if not odds_path.exists():
                counters[f"missing_odds_day_{year}"]+=1
                continue
            day=load_day(day_path,wanted[year])
            if not day: continue
            oddsday=load_odds_day(odds_path,set(day))
            for rid,pack in day.items():
                rid=str(rid); key=(year,rid)
                ranks=king.get(key)
                if not ranks: continue
                if any(r not in ranks for p in BASE_PAIRS for r in p):
                    counters["missing_required_rank"]+=1
                    continue
                orec=oddsday.get(rid)
                if orec is None:
                    counters["missing_race_odds_record"]+=1
                    continue
                qodds={}
                for bet,nums,price in iter_decoded_odds(orec):
                    if bet=="QUINELLA":
                        qodds[tuple(sorted(nums))]=float(price)
                payouts,present=payout_map(pack)
                if "QUINELLA" not in present:
                    counters["missing_quinella_payout"]+=1
                    continue

                candidates=[]
                complete=True
                for rp in BASE_PAIRS:
                    ticket=canonical_numbers("QUINELLA",(ranks[rp[0]],ranks[rp[1]]))
                    odds=qodds.get(ticket)
                    if odds is None:
                        complete=False
                        break
                    payout=float(payouts.get(("QUINELLA",ticket),0.0))
                    candidates.append((rp,ticket,float(odds),payout))
                if not complete:
                    counters["incomplete_five_candidate_odds"]+=1
                    continue

                usable_races[year].add(rid)
                for rp,ticket,odds,payout in candidates:
                    ticket_rows.append({
                        "year":year,"race_date":date,"race_id":rid,
                        "rank_pair":f"{rp[0]}-{rp[1]}",
                        "final_quinella_odds":odds,
                        "return_yen_per100":payout,
                        "winner":int(payout>0),
                    })

                for threshold in THRESHOLDS:
                    chosen=[x for x in candidates if x[2]>=threshold]
                    stake=100.0*len(chosen)
                    ret=sum(x[3] for x in chosen)
                    rows.append({
                        "year":year,"race_date":date,"race_id":rid,
                        "threshold":threshold,
                        "tickets":len(chosen),
                        "stake_yen":stake,
                        "return_yen":ret,
                        "profit_yen":ret-stake,
                        "hit":int(ret>0),
                    })
        print("LOWODDS_YEAR_DONE "+json.dumps({
            "year":year,"usable_races":len(usable_races[year])
        },separators=(",",":")),flush=True)

    summaries=[]
    for threshold in THRESHOLDS:
        for y in YEARS:
            summaries.append(aggregate(rows,(y,),str(y),threshold))
        summaries.append(aggregate(rows,DEV_YEARS,"2023|2024_DEV",threshold))
        summaries.append(aggregate(rows,(HOLDOUT_YEAR,),"2025_HOLDOUT",threshold))

    dev=[x for x in summaries if x and x["period"]=="2023|2024_DEV"]
    eligible=[
        x for x in dev
        if x["active_race_rate_pct"]>=MIN_ACTIVE_RACE_RATE_PCT
        and x["avg_tickets"]>=MIN_AVG_TICKETS
    ]
    if not eligible:
        raise SystemExit("no threshold satisfies coverage guard")
    selected=sorted(
        eligible,
        key=lambda x:(-x["roi_pct"],-x["active_race_rate_pct"],x["threshold"])
    )[0]
    selected_threshold=float(selected["threshold"])
    holdout=next(x for x in summaries if x and x["period"]=="2025_HOLDOUT" and float(x["threshold"])==selected_threshold)
    baseline_dev=next(x for x in summaries if x and x["period"]=="2023|2024_DEV" and float(x["threshold"])==1.0)
    baseline_holdout=next(x for x in summaries if x and x["period"]=="2025_HOLDOUT" and float(x["threshold"])==1.0)

    selection={
        "selected_threshold":selected_threshold,
        "selection_period":"2023|2024_DEV",
        "selection_rule":"highest dev ROI among predeclared thresholds with active_race_rate>=90% and avg_tickets>=2.0; ties prefer higher race coverage then lower threshold",
        "predeclared_thresholds":list(THRESHOLDS),
        "coverage_guard":{
            "min_active_race_rate_pct":MIN_ACTIVE_RACE_RATE_PCT,
            "min_avg_tickets":MIN_AVG_TICKETS,
        },
        "dev":{
            "baseline_roi_pct":baseline_dev["roi_pct"],
            "selected_roi_pct":selected["roi_pct"],
            "baseline_hit_rate_pct":baseline_dev["hit_rate_all_races_pct"],
            "selected_hit_rate_pct":selected["hit_rate_all_races_pct"],
            "selected_active_race_rate_pct":selected["active_race_rate_pct"],
            "selected_avg_tickets":selected["avg_tickets"],
        },
        "holdout_2025":{
            "baseline_roi_pct":baseline_holdout["roi_pct"],
            "selected_roi_pct":holdout["roi_pct"],
            "baseline_hit_rate_pct":baseline_holdout["hit_rate_all_races_pct"],
            "selected_hit_rate_pct":holdout["hit_rate_all_races_pct"],
            "selected_active_race_rate_pct":holdout["active_race_rate_pct"],
            "selected_avg_tickets":holdout["avg_tickets"],
            "roi_delta_pp":holdout["roi_pct"]-baseline_holdout["roi_pct"],
        },
    }

    write_csv(outdir/"threshold-summary.csv",[x for x in summaries if x])
    write_csv(outdir/"ticket-detail.csv",ticket_rows)
    (outdir/"selection.json").write_text(json.dumps(selection,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    meta={
        "contract":"L2_FIVEPOINT_LOWODDS_GATE_V1",
        "base_pairs":[list(x) for x in BASE_PAIRS],
        "odds_source":"final quinella odds from BACKFILL odds daily; feasibility only, not production-time executable",
        "thresholds":list(THRESHOLDS),
        "selection_uses_years":list(DEV_YEARS),
        "holdout_year":HOLDOUT_YEAR,
        "payout_use":"evaluation only",
        "race_gate":False,
        "ticket_gate":"buy candidate ticket iff final_quinella_odds >= selected threshold",
        "counters":dict(counters),
        "usable_races":{str(y):len(usable_races[y]) for y in YEARS},
        "elapsed_seconds":time.perf_counter()-start,
        "promotion":False,
        "2026_locked":True,
    }
    (outdir/"meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== META ====="); print((outdir/"meta.json").read_text())
    print("===== THRESHOLDS ====="); print((outdir/"threshold-summary.csv").read_text())
    print("===== SELECTION ====="); print((outdir/"selection.json").read_text())
    print("L2_FIVEPOINT_LOWODDS_GATE_V1_READY")

if __name__=="__main__":
    main()
