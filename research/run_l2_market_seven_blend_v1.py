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
from build_l2_l17_fullfield_dataset_v1 import load_l17
import run_l2_universal_ticket_ev_v1 as v1

YEARS=(2021,2022,2023,2024,2025)
TEST_YEARS=(2022,2023,2024,2025)
BET_TYPES=("WIN","QUINELLA","EXACTA","TRIO","TRIFECTA")
VARIANTS=("MARKET_ONLY","SEVEN_ONLY","MARKET_SEVEN_BLEND")
ALPHA_GRID=(0.0,0.05,0.10,0.20,0.35,0.50,0.70,1.00,1.30)
STAKE=100.0
EPS=1e-15

def parse_args():
    p=argparse.ArgumentParser(description="Market baseline plus Seven-King residual tilt for all five bet types.")
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def read_year_pairs(root,year,l17):
    root=Path(root)
    daily=root/"data"/"daily"
    odds=root/"data"/"odds"/"daily"
    seen=set()
    for day_path in sorted(daily.glob(f"{year}-*.jsonl.gz")):
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
                if rid in l17:
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
                raise SystemExit(f"missing odds row y={year} race={rid}")
            seen.add(rid)
            yield date,rid,pack,row,l17[rid]
    if len(seen)!=len(l17):
        missing=sorted(set(l17)-seen)[:10]
        raise SystemExit(f"L1.7/daily coverage drift y={year} seen={len(seen)} l17={len(l17)} sample={missing}")

def per_bet_tickets(odds_record):
    out=defaultdict(list)
    for (bet,key),odd in decode_odds(odds_record).items():
        if bet in BET_TYPES:
            out[bet].append((key,float(odd)))
    return out

def market_probs(tickets):
    odds=np.fromiter((odd for _,odd in tickets),dtype=float,count=len(tickets))
    raw=1.0/np.clip(odds,EPS,None)
    overround=float(raw.sum())
    if not math.isfinite(overround) or overround<=0:
        raise ValueError("bad market overround")
    return raw/overround,odds,overround

def seven_probs(tickets,q,bet):
    keys=np.asarray([key for key,_ in tickets],dtype=int)
    max_no=max(max(q),int(keys.max()))
    qv=np.zeros(max_no+1,dtype=float)
    for no,p in q.items():
        qv[int(no)]=float(p)

    if bet=="WIN":
        p=qv[keys[:,0]]
    elif bet=="EXACTA":
        a,b=keys[:,0],keys[:,1]
        qa,qb=qv[a],qv[b]
        p=qa*(qb/np.clip(1.0-qa,EPS,None))
    elif bet=="QUINELLA":
        a,b=keys[:,0],keys[:,1]
        qa,qb=qv[a],qv[b]
        p=qa*(qb/np.clip(1.0-qa,EPS,None))
        p+=qb*(qa/np.clip(1.0-qb,EPS,None))
    elif bet=="TRIFECTA":
        a,b,c=keys[:,0],keys[:,1],keys[:,2]
        qa,qb,qc=qv[a],qv[b],qv[c]
        p=qa*(qb/np.clip(1.0-qa,EPS,None))*(qc/np.clip(1.0-qa-qb,EPS,None))
    elif bet=="TRIO":
        a,b,c=keys[:,0],keys[:,1],keys[:,2]
        qa,qb,qc=qv[a],qv[b],qv[c]
        def ord3(x,y,z):
            return x*(y/np.clip(1.0-x,EPS,None))*(z/np.clip(1.0-x-y,EPS,None))
        p=(
            ord3(qa,qb,qc)+ord3(qa,qc,qb)+
            ord3(qb,qa,qc)+ord3(qb,qc,qa)+
            ord3(qc,qa,qb)+ord3(qc,qb,qa)
        )
    else:
        raise ValueError(bet)

    p=np.asarray(p,dtype=float)
    if np.any(~np.isfinite(p)) or np.any(p<0):
        raise ValueError(f"bad seven probability bet={bet}")
    mass=float(p.sum())
    if mass<=0:
        raise ValueError(f"zero seven probability mass bet={bet}")
    return p/mass,mass

def blend_probs(pm,ps,alpha):
    logp=(1.0-alpha)*np.log(np.clip(pm,EPS,None))+alpha*np.log(np.clip(ps,EPS,None))
    m=float(np.max(logp))
    w=np.exp(logp-m)
    return w/float(w.sum())

def top3_numbers(pack,rec):
    horses=rec.get("horses") or []
    allowed={str(h.get("horse_id") or "") for h in horses}
    top3=v1.top3_from_pack(pack,allowed)
    if top3 is None:
        return None
    hid_to_no={str(h["horse_id"]):int(h["horse_number"]) for h in horses}
    try:
        return tuple(hid_to_no[x] for x in top3)
    except KeyError:
        return None

def validate_numbers(pack,rec,year,rid):
    pmap=horse_number_map(pack)
    for h in rec.get("horses") or []:
        hid=str(h.get("horse_id") or "")
        no=int(h.get("horse_number"))
        if hid not in pmap or int(pmap[hid])!=no:
            raise SystemExit(f"horse number drift y={year} race={rid} horse={hid}")

def collect_alpha_stats(root,l17):
    stats={
        y:{bet:{a:{"nll":0.0,"races":0} for a in ALPHA_GRID} for bet in BET_TYPES}
        for y in YEARS
    }
    counts={y:{"races_seen":0,"usable_top3":0,"excluded_top3":0} for y in YEARS}

    for y in YEARS:
        for date,rid,pack,odds_record,rec in read_year_pairs(root,y,l17[y]):
            counts[y]["races_seen"]+=1
            validate_numbers(pack,rec,y,rid)
            top3=top3_numbers(pack,rec)
            if top3 is None:
                counts[y]["excluded_top3"]+=1
                continue
            counts[y]["usable_top3"]+=1
            payouts,present=payout_map(pack)
            tickets_by_bet=per_bet_tickets(odds_record)
            q=v1.race_q(rec,1.0)

            for bet in BET_TYPES:
                if bet not in present:
                    continue
                tickets=tickets_by_bet.get(bet) or []
                if not tickets:
                    continue
                idx={key:i for i,(key,_) in enumerate(tickets)}
                wk=v1.winner_key(bet,top3)
                wi=idx.get(wk)
                if wi is None:
                    continue
                pm,_,_=market_probs(tickets)
                ps,_=seven_probs(tickets,q,bet)
                lpm=np.log(np.clip(pm,EPS,None))
                lps=np.log(np.clip(ps,EPS,None))
                for a in ALPHA_GRID:
                    logits=(1.0-a)*lpm+a*lps
                    m=float(np.max(logits))
                    logz=m+math.log(float(np.exp(logits-m).sum()))
                    recs=stats[y][bet][a]
                    recs["nll"]+=(-float(logits[wi])+logz)
                    recs["races"]+=1
    return stats,counts

def choose_alphas(stats,test_year):
    train_years=[y for y in YEARS if y<test_year]
    chosen={}
    rows=[]
    for bet in BET_TYPES:
        scored=[]
        for a in ALPHA_GRID:
            nll=0.0
            races=0
            for y in train_years:
                nll+=stats[y][bet][a]["nll"]
                races+=stats[y][bet][a]["races"]
            mean=nll/races if races else float("inf")
            rows.append({
                "test_year":test_year,
                "train_years":"|".join(map(str,train_years)),
                "bet_type":bet,
                "alpha_seven":a,
                "alpha_market":1.0-a,
                "races":races,
                "mean_winner_nll":mean,
            })
            scored.append((mean,abs(a),a))
        scored.sort()
        chosen[bet]=float(scored[0][2])
    for row in rows:
        row["selected_alpha"]=int(float(row["alpha_seven"])==chosen[row["bet_type"]])
    return chosen,rows

def empty_metric():
    return {
        "source_races":0,
        "priced_tickets":0,
        "selected_tickets":0,
        "selected_races":0,
        "hit_tickets":0,
        "hit_races":0,
        "stake_yen":0.0,
        "actual_return_yen":0.0,
        "predicted_return_yen":0.0,
        "edge_sum":0.0,
        "winner_nll_sum":0.0,
        "winner_nll_races":0,
        "seven_priced_mass_sum":0.0,
    }

def add_metric(dst,src):
    for k,v in src.items():
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
        "winning_ticket_nll":m["winner_nll_sum"]/m["winner_nll_races"] if m["winner_nll_races"] else None,
        "mean_seven_probability_mass_on_priced_universe":m["seven_priced_mass_sum"]/source if source else None,
    }

def evaluate_year(root,year,l17,alphas,out_race):
    metrics={(variant,bet):empty_metric() for variant in VARIANTS for bet in BET_TYPES}
    with gzip.open(out_race,"wt",newline="",encoding="utf-8") as fh:
        fields=[
            "year","race_id","race_date","bet_type","variant","alpha_seven",
            "priced_tickets","selected_tickets","stake_yen","predicted_return_yen",
            "actual_return_yen","profit_yen","hit","winner_probability"
        ]
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()

        for date,rid,pack,odds_record,rec in read_year_pairs(root,year,l17):
            validate_numbers(pack,rec,year,rid)
            payouts,present=payout_map(pack)
            top3=top3_numbers(pack,rec)
            tickets_by_bet=per_bet_tickets(odds_record)
            q=v1.race_q(rec,1.0)

            for bet in BET_TYPES:
                if bet not in present:
                    continue
                tickets=tickets_by_bet.get(bet) or []
                if not tickets:
                    continue
                pm,odds,_=market_probs(tickets)
                ps,seven_mass=seven_probs(tickets,q,bet)
                alpha=float(alphas[bet])
                pb=blend_probs(pm,ps,alpha)
                idx={key:i for i,(key,_) in enumerate(tickets)}
                winner_idx=None
                if top3 is not None:
                    winner_idx=idx.get(v1.winner_key(bet,top3))

                for variant,p,va in (
                    ("MARKET_ONLY",pm,0.0),
                    ("SEVEN_ONLY",ps,1.0),
                    ("MARKET_SEVEN_BLEND",pb,alpha),
                ):
                    m=metrics[(variant,bet)]
                    m["source_races"]+=1
                    m["priced_tickets"]+=len(tickets)
                    m["seven_priced_mass_sum"]+=seven_mass

                    ev=p*odds
                    mask=ev>1.0
                    selected=int(np.count_nonzero(mask))
                    pred=STAKE*float(ev[mask].sum()) if selected else 0.0
                    edge_sum=float((ev[mask]-1.0).sum()) if selected else 0.0

                    actual=0.0
                    hit_tickets=0
                    for (pbet,key),ret in payouts.items():
                        if pbet!=bet:
                            continue
                        i=idx.get(key)
                        if i is not None and mask[i]:
                            actual+=float(ret)
                            hit_tickets+=1
                    race_hit=int(hit_tickets>0)

                    m["selected_tickets"]+=selected
                    m["hit_tickets"]+=hit_tickets
                    m["stake_yen"]+=selected*STAKE
                    m["actual_return_yen"]+=actual
                    m["predicted_return_yen"]+=pred
                    m["edge_sum"]+=edge_sum
                    if selected:
                        m["selected_races"]+=1
                        m["hit_races"]+=race_hit
                    winner_probability=None
                    if winner_idx is not None:
                        winner_probability=float(p[winner_idx])
                        m["winner_nll_sum"]-=math.log(max(EPS,winner_probability))
                        m["winner_nll_races"]+=1

                    w.writerow({
                        "year":year,"race_id":rid,"race_date":date,"bet_type":bet,
                        "variant":variant,"alpha_seven":va,"priced_tickets":len(tickets),
                        "selected_tickets":selected,"stake_yen":selected*STAKE,
                        "predicted_return_yen":pred,"actual_return_yen":actual,
                        "profit_yen":actual-selected*STAKE,"hit":race_hit,
                        "winner_probability":winner_probability,
                    })
    return metrics

def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def main():
    a=parse_args()
    paths={y:Path(getattr(a,f"l17_{y}")) for y in YEARS}
    l17={y:load_l17(paths[y],y) for y in YEARS}

    stats,counts=collect_alpha_stats(a.backfill_root,l17)
    alphas={}
    fold_rows=[]
    grid_rows=[]
    for y in TEST_YEARS:
        chosen,rows=choose_alphas(stats,y)
        alphas[y]=chosen
        grid_rows.extend(rows)
        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(str(t) for t in YEARS if t<y),
            "win_alpha_seven":chosen["WIN"],
            "quinella_alpha_seven":chosen["QUINELLA"],
            "exacta_alpha_seven":chosen["EXACTA"],
            "trio_alpha_seven":chosen["TRIO"],
            "trifecta_alpha_seven":chosen["TRIFECTA"],
        })

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    yearly=[]
    combined={(variant,bet):empty_metric() for variant in VARIANTS for bet in BET_TYPES}
    race_files=[]
    for y in TEST_YEARS:
        race_file=out/f"race-blend-summary-{y}.csv.gz"
        race_files.append(race_file.name)
        metrics=evaluate_year(a.backfill_root,y,l17[y],alphas[y],race_file)
        for variant in VARIANTS:
            for bet in BET_TYPES:
                yearly.append(metric_row(y,bet,variant,metrics[(variant,bet)]))
                add_metric(combined[(variant,bet)],metrics[(variant,bet)])

    combined_rows=[
        metric_row("ALL",bet,variant,combined[(variant,bet)])
        for variant in VARIANTS for bet in BET_TYPES
    ]
    by={(r["variant"],r["bet_type"]):r for r in combined_rows}
    comparison=[]
    for bet in BET_TYPES:
        m=by[("MARKET_ONLY",bet)]
        s=by[("SEVEN_ONLY",bet)]
        b=by[("MARKET_SEVEN_BLEND",bet)]
        comparison.append({
            "bet_type":bet,
            "market_winner_nll":m["winning_ticket_nll"],
            "seven_winner_nll":s["winning_ticket_nll"],
            "blend_winner_nll":b["winning_ticket_nll"],
            "blend_nll_delta_vs_market":b["winning_ticket_nll"]-m["winning_ticket_nll"],
            "market_selected_tickets":m["selected_tickets"],
            "seven_selected_tickets":s["selected_tickets"],
            "blend_selected_tickets":b["selected_tickets"],
            "seven_roi_pct":s["roi_pct"],
            "blend_roi_pct":b["roi_pct"],
            "blend_profit_yen":b["profit_yen"],
            "blend_execution_coverage_pct":b["execution_coverage_pct"],
        })

    write_csv(out/"alpha-folds.csv",fold_rows)
    write_csv(out/"alpha-grid.csv",grid_rows)
    write_csv(out/"economics-yearly.csv",yearly)
    write_csv(out/"economics-combined.csv",combined_rows)
    write_csv(out/"market-vs-seven-vs-blend.csv",comparison)

    summary={
        "contract":"L2_MARKET_SEVEN_BLEND_V1",
        "bet_types":list(BET_TYPES),
        "market_probability":"normalized inverse final odds within race and bet type",
        "seven_probability":"L1.7 Seven-King mean_probability -> Plackett-Luce ticket probability -> renormalized on priced ticket universe",
        "blend_probability":"p_blend proportional to p_market^(1-alpha) * p_seven^alpha",
        "alpha_grid":list(ALPHA_GRID),
        "alpha_selection":"minimum strictly-prior-year winning-ticket negative log likelihood",
        "roi_used_for_alpha_selection":False,
        "test_year_used_for_alpha_selection":False,
        "payout_used_for_alpha_selection":False,
        "buy_rule":"variant_probability * historical final odds > 1.0",
        "manual_edge_threshold":False,
        "manual_top_k":False,
        "manual_ticket_cap":False,
        "manual_odds_band":False,
        "manual_race_skip_rule":False,
        "folds":fold_rows,
        "calibration_counts":counts,
        "race_summary_files":race_files,
        "market_price_caveat":"historical FINAL odds are a research execution-price proxy, not timestamped production execution prices",
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Market + Seven Blend V1\n\n"
        "This is the direct three-way control: market-only, Seven-only and market+Seven. "
        "The market distribution is normalized inverse final odds. The Seven distribution is the L1.7 "
        "full-field mean-probability world decomposed into each ticket type and renormalized over the same "
        "priced ticket universe. The combined distribution is a geometric opinion pool: "
        "p ∝ market^(1-alpha) * seven^alpha. Alpha is chosen independently for each bet type using only "
        "strictly prior-year winning-ticket negative log likelihood; ROI, payout and the test year are never "
        "used to choose alpha. All three variants use the same EV>1 rule. 2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_MARKET_SEVEN_BLEND_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
    print("===== MARKET VS SEVEN VS BLEND =====")
    with open(out/"market-vs-seven-vs-blend.csv",encoding="utf-8") as fh:
        print(fh.read())

if __name__=="__main__":
    main()
