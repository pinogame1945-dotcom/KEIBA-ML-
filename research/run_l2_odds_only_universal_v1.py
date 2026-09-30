#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_l2_bet_kings_dataset_v1 import decode_odds, payout_map, horse_number_map

YEARS=(2021,2022,2023,2024,2025)
TEST_YEARS=(2022,2023,2024,2025)
BET_TYPES=("WIN","QUINELLA","EXACTA","TRIO","TRIFECTA")
GAMMA_GRID=(0.50,0.65,0.80,0.90,1.00,1.10,1.20,1.35,1.55,1.80,2.10,2.50,3.00)
VARIANTS=("RAW_MARKET","POWER_CALIBRATED_MARKET")
STAKE=100.0
EPS=1e-15

def parse_args():
    p=argparse.ArgumentParser(description="Odds-only universal five-bet market baseline.")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def top3_numbers(pack):
    hid_to_no=horse_number_map(pack)
    by_pos=defaultdict(list)
    for r in pack.get("results") or []:
        if str(r.get("result_status") or "").upper()!="FINISHED":
            continue
        hid=str(r.get("horse_id") or "")
        if hid not in hid_to_no:
            continue
        try:
            pos=int(r.get("official_finish_position"))
        except (TypeError,ValueError):
            continue
        if pos in (1,2,3):
            by_pos[pos].append(int(hid_to_no[hid]))
    if any(len(by_pos[p])!=1 for p in (1,2,3)):
        return None
    return by_pos[1][0],by_pos[2][0],by_pos[3][0]

def winner_key(bet,top3):
    a,b,c=top3
    if bet=="WIN":
        return (a,)
    if bet=="QUINELLA":
        return tuple(sorted((a,b)))
    if bet=="EXACTA":
        return (a,b)
    if bet=="TRIO":
        return tuple(sorted((a,b,c)))
    if bet=="TRIFECTA":
        return (a,b,c)
    raise ValueError(bet)

def read_year_pairs(root,year):
    root=Path(root)
    daily=root/"data"/"daily"
    odds=root/"data"/"odds"/"daily"
    files=sorted(daily.glob(f"{year}-*.jsonl.gz"))
    if not files:
        raise SystemExit(f"no race packs year={year}")
    for day_path in files:
        date=day_path.name[:10]
        odds_path=odds/day_path.name
        if not odds_path.exists():
            raise SystemExit(f"missing odds day: {odds_path}")
        day={}
        with gzip.open(day_path,"rt",encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                pack=json.loads(line)
                rid=str((pack.get("race") or {}).get("race_id") or "")
                if rid:
                    day[rid]=pack
        market={}
        with gzip.open(odds_path,"rt",encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row=json.loads(line)
                rid=str(row.get("race_id") or "")
                if rid in day:
                    market[rid]=row
        for rid,pack in day.items():
            row=market.get(rid)
            if row is None:
                continue
            yield date,rid,pack,row

def market_distribution(tickets):
    raw=np.fromiter((1.0/max(EPS,float(odd)) for _,odd in tickets),dtype=float,count=len(tickets))
    s=float(raw.sum())
    if not math.isfinite(s) or s<=0:
        raise ValueError("invalid market inverse-odds sum")
    return raw/s,s

def logsumexp_scaled(logps,gamma):
    vals=gamma*logps
    m=float(np.max(vals))
    return m+math.log(float(np.exp(vals-m).sum()))

def collect_calibration_stats(root):
    stats={
        y:{bet:{g:{"nll":0.0,"races":0} for g in GAMMA_GRID} for bet in BET_TYPES}
        for y in YEARS
    }
    counts={y:{"races_seen":0,"usable_top3":0,"excluded_top3":0} for y in YEARS}
    for y in YEARS:
        for date,rid,pack,odds_record in read_year_pairs(root,y):
            counts[y]["races_seen"]+=1
            top3=top3_numbers(pack)
            if top3 is None:
                counts[y]["excluded_top3"]+=1
                continue
            counts[y]["usable_top3"]+=1
            odds_map=decode_odds(odds_record)
            payouts,present=payout_map(pack)
            per_bet=defaultdict(list)
            for (bet,key),odd in odds_map.items():
                if bet in BET_TYPES:
                    per_bet[bet].append((key,float(odd)))
            for bet in BET_TYPES:
                if bet not in present:
                    continue
                tickets=per_bet.get(bet) or []
                if not tickets:
                    continue
                probs,_=market_distribution(tickets)
                idx={key:i for i,(key,_) in enumerate(tickets)}
                wk=winner_key(bet,top3)
                if wk not in idx:
                    continue
                pw=max(EPS,probs[idx[wk]])
                logps=np.log(np.clip(probs,EPS,None))
                lpw=math.log(pw)
                for g in GAMMA_GRID:
                    logz=logsumexp_scaled(logps,g)
                    rec=stats[y][bet][g]
                    rec["nll"]+=(-g*lpw+logz)
                    rec["races"]+=1
    return stats,counts

def fit_gammas(stats,test_year):
    train_years=[y for y in YEARS if y<test_year]
    chosen={}
    rows=[]
    for bet in BET_TYPES:
        scored=[]
        for g in GAMMA_GRID:
            nll=0.0
            races=0
            for y in train_years:
                nll+=stats[y][bet][g]["nll"]
                races+=stats[y][bet][g]["races"]
            mean=nll/races if races else float("inf")
            rows.append({
                "test_year":test_year,
                "train_years":"|".join(map(str,train_years)),
                "bet_type":bet,
                "gamma":g,
                "races":races,
                "mean_winner_nll":mean,
            })
            scored.append((mean,abs(g-1.0),g))
        scored.sort()
        chosen[bet]=float(scored[0][2])
    for r in rows:
        r["selected_gamma"]=int(float(r["gamma"])==chosen[r["bet_type"]])
    return chosen,rows

def empty_metric():
    return {
        "source_races":0,
        "priced_tickets":0,
        "probability_mass_priced":0.0,
        "selected_tickets":0,
        "selected_races":0,
        "hit_tickets":0,
        "hit_races":0,
        "stake_yen":0.0,
        "actual_return_yen":0.0,
        "predicted_return_yen":0.0,
        "edge_sum":0.0,
        "overround_sum":0.0,
        "overround_min":float("inf"),
        "overround_max":0.0,
        "winner_nll_sum":0.0,
        "winner_nll_races":0,
    }

def add_metric(dst,src):
    for k,v in src.items():
        if k=="overround_min":
            dst[k]=min(dst[k],v)
        elif k=="overround_max":
            dst[k]=max(dst[k],v)
        else:
            dst[k]+=v

def metric_row(year,bet,variant,m):
    source=m["source_races"]
    selected=m["selected_tickets"]
    stake=m["stake_yen"]
    return {
        "year":year,
        "bet_type":bet,
        "variant":variant,
        "source_races":source,
        "priced_tickets":m["priced_tickets"],
        "mean_probability_mass_priced":m["probability_mass_priced"]/source if source else 0.0,
        "selected_tickets":selected,
        "selected_races":m["selected_races"],
        "execution_coverage_pct":100.0*m["selected_races"]/source if source else 0.0,
        "tickets_per_source_race":selected/source if source else 0.0,
        "tickets_per_executed_race":selected/m["selected_races"] if m["selected_races"] else 0.0,
        "hit_tickets":m["hit_tickets"],
        "hit_races":m["hit_races"],
        "race_hit_rate_pct_all":100.0*m["hit_races"]/source if source else 0.0,
        "stake_yen":stake,
        "actual_return_yen":m["actual_return_yen"],
        "profit_yen":m["actual_return_yen"]-stake,
        "roi_pct":100.0*m["actual_return_yen"]/stake if stake else None,
        "predicted_return_yen":m["predicted_return_yen"],
        "predicted_profit_yen":m["predicted_return_yen"]-stake,
        "mean_selected_edge":m["edge_sum"]/selected if selected else None,
        "mean_market_overround":m["overround_sum"]/source if source else None,
        "min_market_overround":m["overround_min"] if source else None,
        "max_market_overround":m["overround_max"] if source else None,
        "winning_ticket_nll":m["winner_nll_sum"]/m["winner_nll_races"] if m["winner_nll_races"] else None,
    }

def evaluate_year(root,year,gammas,out_race):
    metrics={(variant,bet):empty_metric() for variant in VARIANTS for bet in BET_TYPES}
    with gzip.open(out_race,"wt",newline="",encoding="utf-8") as fh:
        fields=[
            "year","race_id","race_date","bet_type","variant","gamma",
            "market_overround","priced_tickets","selected_tickets",
            "stake_yen","predicted_return_yen","actual_return_yen","profit_yen","hit"
        ]
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()
        for date,rid,pack,odds_record in read_year_pairs(root,year):
            odds_map=decode_odds(odds_record)
            payouts,present=payout_map(pack)
            top3=top3_numbers(pack)
            per_bet=defaultdict(list)
            for (bet,key),odd in odds_map.items():
                if bet in BET_TYPES:
                    per_bet[bet].append((key,float(odd)))
            for bet in BET_TYPES:
                if bet not in present:
                    continue
                tickets=per_bet.get(bet) or []
                if not tickets:
                    continue
                probs,overround=market_distribution(tickets)
                gamma=float(gammas[bet])
                powered=np.power(probs,gamma)
                z=float(powered.sum())
                cal=powered/z

                for variant,ps,g in (
                    ("RAW_MARKET",probs,1.0),
                    ("POWER_CALIBRATED_MARKET",cal,gamma),
                ):
                    m=metrics[(variant,bet)]
                    m["source_races"]+=1
                    m["priced_tickets"]+=len(tickets)
                    m["probability_mass_priced"]+=float(np.sum(ps))
                    m["overround_sum"]+=overround
                    m["overround_min"]=min(m["overround_min"],overround)
                    m["overround_max"]=max(m["overround_max"],overround)
                    selected=0
                    race_return=0.0
                    race_pred=0.0
                    race_hit=0
                    for (key,odd),p in zip(tickets,ps):
                        ev=p*odd
                        edge=ev-1.0
                        if edge<=0.0:
                            continue
                        ret=float(payouts.get((bet,key),0.0))
                        hit=int(ret>0)
                        selected+=1
                        race_return+=ret
                        race_pred+=STAKE*ev
                        race_hit=max(race_hit,hit)
                        m["selected_tickets"]+=1
                        m["hit_tickets"]+=hit
                        m["stake_yen"]+=STAKE
                        m["actual_return_yen"]+=ret
                        m["predicted_return_yen"]+=STAKE*ev
                        m["edge_sum"]+=edge
                    if selected:
                        m["selected_races"]+=1
                        m["hit_races"]+=race_hit
                    if top3 is not None:
                        wk=winner_key(bet,top3)
                        idx={key:i for i,(key,_) in enumerate(tickets)}
                        if wk in idx:
                            pwin=max(EPS,ps[idx[wk]])
                            m["winner_nll_sum"]-=math.log(pwin)
                            m["winner_nll_races"]+=1
                    w.writerow({
                        "year":year,"race_id":rid,"race_date":date,"bet_type":bet,
                        "variant":variant,"gamma":g,"market_overround":overround,
                        "priced_tickets":len(tickets),"selected_tickets":selected,
                        "stake_yen":selected*STAKE,"predicted_return_yen":race_pred,
                        "actual_return_yen":race_return,
                        "profit_yen":race_return-selected*STAKE,"hit":race_hit,
                    })
    return metrics

def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def main():
    a=parse_args()
    stats,counts=collect_calibration_stats(a.backfill_root)
    fold_rows=[]
    grid_rows=[]
    gammas={}
    for y in TEST_YEARS:
        g,rows=fit_gammas(stats,y)
        gammas[y]=g
        grid_rows.extend(rows)
        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(str(t) for t in YEARS if t<y),
            "win_gamma":g["WIN"],
            "quinella_gamma":g["QUINELLA"],
            "exacta_gamma":g["EXACTA"],
            "trio_gamma":g["TRIO"],
            "trifecta_gamma":g["TRIFECTA"],
        })

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    yearly=[]
    combined={(variant,bet):empty_metric() for variant in VARIANTS for bet in BET_TYPES}
    race_files=[]
    for y in TEST_YEARS:
        race_file=out/f"race-market-summary-{y}.csv.gz"
        race_files.append(race_file.name)
        metrics=evaluate_year(a.backfill_root,y,gammas[y],race_file)
        for variant in VARIANTS:
            for bet in BET_TYPES:
                yearly.append(metric_row(y,bet,variant,metrics[(variant,bet)]))
                add_metric(combined[(variant,bet)],metrics[(variant,bet)])

    combined_rows=[
        metric_row("ALL",bet,variant,combined[(variant,bet)])
        for variant in VARIANTS for bet in BET_TYPES
    ]
    comparison=[]
    by={(r["variant"],r["bet_type"]):r for r in combined_rows}
    for bet in BET_TYPES:
        raw=by[("RAW_MARKET",bet)]
        cal=by[("POWER_CALIBRATED_MARKET",bet)]
        comparison.append({
            "bet_type":bet,
            "raw_market_roi_pct":raw["roi_pct"],
            "calibrated_market_roi_pct":cal["roi_pct"],
            "raw_selected_tickets":raw["selected_tickets"],
            "calibrated_selected_tickets":cal["selected_tickets"],
            "raw_execution_coverage_pct":raw["execution_coverage_pct"],
            "calibrated_execution_coverage_pct":cal["execution_coverage_pct"],
            "raw_mean_overround":raw["mean_market_overround"],
            "raw_winner_nll":raw["winning_ticket_nll"],
            "calibrated_winner_nll":cal["winning_ticket_nll"],
        })

    write_csv(out/"calibration-folds.csv",fold_rows)
    write_csv(out/"gamma-grid.csv",grid_rows)
    write_csv(out/"economics-yearly.csv",yearly)
    write_csv(out/"economics-combined.csv",combined_rows)
    write_csv(out/"raw-vs-calibrated-market.csv",comparison)

    summary={
        "contract":"L2_ODDS_ONLY_UNIVERSAL_V1",
        "bet_types":list(BET_TYPES),
        "horse_model_used":False,
        "l17_used":False,
        "seven_king_used":False,
        "raw_market_probability":"(1/odds) normalized across all priced tickets within race and bet type",
        "calibration":{
            "method":"p_cal = p_market^gamma / sum(p_market^gamma)",
            "gamma_grid":list(GAMMA_GRID),
            "selection_metric":"minimum strictly-prior-year winning-ticket negative log likelihood",
            "odds_used_as_input":True,
            "roi_used_for_gamma_selection":False,
            "test_year_used_for_gamma_selection":False,
        },
        "buy_rule":"probability * same historical final odds > 1.0",
        "manual_edge_threshold":False,
        "manual_top_k":False,
        "manual_ticket_cap":False,
        "manual_odds_band":False,
        "manual_race_skip_rule":False,
        "market_price_stage":"historical FINAL odds",
        "important_interpretation":"RAW_MARKET is a self-consistency sanity check; with a complete pari-mutuel market, normalized inverse odds should ordinarily imply no positive EV after overround/takeout.",
        "calibration_counts":counts,
        "folds":fold_rows,
        "race_summary_files":race_files,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Odds-Only Universal V1\n\n"
        "Seven-King/L1.7 information is completely removed. For each race and bet type, historical final "
        "odds are converted to normalized inverse-odds market probabilities. RAW_MARKET tests pure market "
        "self-consistency. POWER_CALIBRATED_MARKET applies a prior-year-only power calibration selected by "
        "winning-ticket negative log likelihood, never by ROI. Both variants use exactly probability * the "
        "same final odds > 1.0 as the buy rule. No edge floor, Top-K, ticket cap, odds band or race skip rule "
        "is used. 2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_ODDS_ONLY_UNIVERSAL_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
    print("===== RAW VS CALIBRATED MARKET =====")
    with open(out/"raw-vs-calibrated-market.csv",encoding="utf-8") as fh:
        print(fh.read())

if __name__=="__main__":
    main()
