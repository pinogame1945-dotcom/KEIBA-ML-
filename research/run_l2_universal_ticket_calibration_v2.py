#!/usr/bin/env python3
import argparse
import csv
import gzip
import heapq
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from build_l2_bet_kings_dataset_v1 import decode_odds, payout_map, horse_number_map
from build_l2_l17_fullfield_dataset_v1 import load_l17
import run_l2_universal_ticket_ev_v1 as v1

YEARS=v1.YEARS
TEST_YEARS=v1.TEST_YEARS
BET_TYPES=v1.BET_TYPES
STAKE=v1.STAKE
EPS=v1.EPS
GAMMA_GRID=(0.60,0.75,0.90,1.00,1.15,1.35,1.60,2.00,2.50,3.00)
VARIANTS=("BASE_GAMMA1","POWER_CALIBRATED")
TOP_SAMPLE_PER_BET_YEAR=30


def parse_args():
    p=argparse.ArgumentParser(description="Universal ticket EV V2 with market-free ticket-distribution power calibration.")
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


@lru_cache(maxsize=None)
def combo3_indices(n):
    return np.asarray(list(itertools.combinations(range(n),3)),dtype=int)


def ticket_prob_array(q,bet_type):
    nums=sorted(q)
    a=np.asarray([q[n] for n in nums],dtype=float)
    n=len(a)
    if bet_type=="WIN":
        out=a
    else:
        qi=a[:,None]
        qj=a[None,:]
        p2=qi*(qj/np.clip(1.0-qi,EPS,None))
        mask2=~np.eye(n,dtype=bool)
        p2=np.where(mask2,p2,0.0)
        if bet_type=="EXACTA":
            out=p2[mask2]
        elif bet_type=="QUINELLA":
            iu=np.triu_indices(n,1)
            out=(p2+p2.T)[iu]
        else:
            qi3=a[:,None,None]
            qj3=a[None,:,None]
            qk3=a[None,None,:]
            denom1=np.clip(1.0-qi3,EPS,None)
            denom2=np.clip(1.0-qi3-qj3,EPS,None)
            p3=qi3*(qj3/denom1)*(qk3/denom2)
            idx=np.arange(n)
            distinct=(
                (idx[:,None,None]!=idx[None,:,None]) &
                (idx[:,None,None]!=idx[None,None,:]) &
                (idx[None,:,None]!=idx[None,None,:])
            )
            p3=np.where(distinct,p3,0.0)
            if bet_type=="TRIFECTA":
                out=p3[distinct]
            elif bet_type=="TRIO":
                c=combo3_indices(n)
                i,j,k=c[:,0],c[:,1],c[:,2]
                out=(
                    p3[i,j,k]+p3[i,k,j]+
                    p3[j,i,k]+p3[j,k,i]+
                    p3[k,i,j]+p3[k,j,i]
                )
            else:
                raise ValueError(bet_type)
    out=np.asarray(out,dtype=float)
    if out.size==0:
        return out
    if not np.all(np.isfinite(out)) or np.any(out<0):
        raise ValueError(f"invalid ticket probability array bet={bet_type}")
    s=float(out.sum())
    if abs(s-1.0)>1e-8:
        raise ValueError(f"ticket probability mass drift bet={bet_type} sum={s}")
    return out


def fit_ticket_gammas(rows,temperature):
    losses={bet:{g:0.0 for g in GAMMA_GRID} for bet in BET_TYPES}
    counts={bet:0 for bet in BET_TYPES}
    for base,top3 in rows:
        q=v1.q_from_base(base,temperature)
        for bet in BET_TYPES:
            probs=ticket_prob_array(q,bet)
            if probs.size==0:
                continue
            wk=v1.winner_key(bet,top3)
            pw=max(EPS,v1.ticket_probability(bet,wk,q))
            logp=np.log(np.clip(probs,EPS,None))
            lpw=math.log(pw)
            for g in GAMMA_GRID:
                # -log(p_w^g / sum_j p_j^g)
                m=float(np.max(g*logp))
                logz=m+math.log(float(np.exp(g*logp-m).sum()))
                losses[bet][g]+=(-g*lpw+logz)
            counts[bet]+=1

    chosen={}
    grid_rows=[]
    for bet in BET_TYPES:
        if not counts[bet]:
            raise SystemExit(f"no gamma calibration races bet={bet}")
        scored=[]
        for g in GAMMA_GRID:
            mean=losses[bet][g]/counts[bet]
            row={"bet_type":bet,"gamma":g,"races":counts[bet],"mean_winner_nll":mean}
            grid_rows.append(row)
            scored.append((mean,abs(g-1.0),g))
        scored.sort()
        chosen[bet]=float(scored[0][2])
    return chosen,grid_rows


def empty_metric():
    return v1.empty_metric()


def update_selected(m,ret,predicted,edge):
    hit=int(ret>0)
    m["selected_tickets"]+=1
    m["hit_tickets"]+=hit
    m["stake_yen"]+=STAKE
    m["actual_return_yen"]+=ret
    m["predicted_return_yen"]+=predicted
    m["edge_sum"]+=edge
    return hit


def metric_row(year,bet,variant,m):
    r=v1.metric_row(year,bet,m)
    r["variant"]=variant
    return r


def push_sample(heap,serial,rec):
    item=(float(rec["edge"]),serial,rec)
    if len(heap)<TOP_SAMPLE_PER_BET_YEAR:
        heapq.heappush(heap,item)
    elif item[0]>heap[0][0]:
        heapq.heapreplace(heap,item)


def evaluate_year(year,temperature,gammas,l17,backfill_root,out_race_gz):
    root=Path(backfill_root)
    metrics={(variant,bet):empty_metric() for variant in VARIANTS for bet in BET_TYPES}
    samples={bet:[] for bet in BET_TYPES}
    serial=0
    seen=set()

    with gzip.open(out_race_gz,"wt",newline="",encoding="utf-8") as race_fh:
        fields=[
            "year","race_id","race_date","bet_type","variant","gamma","field_size",
            "theoretical_tickets","priced_tickets","probability_mass_priced",
            "selected_tickets","predicted_return_yen","stake_yen",
            "actual_return_yen","profit_yen","hit","market_overround"
        ]
        w=csv.DictWriter(race_fh,fieldnames=fields)
        w.writeheader()

        for day_path in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
            date=day_path.name[:10]
            odds_path=root/"data"/"odds"/"daily"/day_path.name
            if not odds_path.exists():
                raise SystemExit(f"missing odds day: {odds_path}")
            day_rows={}
            with gzip.open(day_path,"rt",encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    pack=json.loads(line)
                    rid=str((pack.get("race") or {}).get("race_id") or "")
                    if rid in l17:
                        day_rows[rid]=pack
            odds_rows={}
            with gzip.open(odds_path,"rt",encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r=json.loads(line)
                    rid=str(r.get("race_id") or "")
                    if rid in day_rows:
                        odds_rows[rid]=r

            for rid,pack in day_rows.items():
                rec=l17[rid]
                odds_record=odds_rows.get(rid)
                if odds_record is None:
                    raise SystemExit(f"missing odds row y={year} race={rid}")
                seen.add(rid)
                horses=rec.get("horses") or []
                by_no={int(h["horse_number"]):h for h in horses}
                hid_to_no={str(h["horse_id"]):int(h["horse_number"]) for h in horses}
                pack_horse_no=horse_number_map(pack)
                for hid,no in hid_to_no.items():
                    if hid not in pack_horse_no or int(pack_horse_no[hid])!=no:
                        raise SystemExit(f"horse number drift y={year} race={rid} horse={hid}")

                q=v1.race_q(rec,temperature)
                odds_map=decode_odds(odds_record)
                payouts,present=payout_map(pack)
                allowed=set(hid_to_no)
                top3=v1.top3_from_pack(pack,allowed)
                top3_nums=tuple(hid_to_no[x] for x in top3) if top3 else None
                per_bet={bet:[] for bet in BET_TYPES}
                for (bet,key),odd in odds_map.items():
                    if bet in per_bet:
                        per_bet[bet].append((key,float(odd)))

                for bet in BET_TYPES:
                    if bet not in present:
                        continue
                    tickets=per_bet[bet]
                    if not tickets:
                        continue
                    gamma=float(gammas[bet])
                    probs_all=ticket_prob_array(q,bet)
                    logp=np.log(np.clip(probs_all,EPS,None))
                    mmax=float(np.max(gamma*logp))
                    z=math.exp(mmax)*float(np.exp(gamma*logp-mmax).sum())
                    if not math.isfinite(z) or z<=0:
                        raise SystemExit(f"bad power normalization y={year} race={rid} bet={bet} z={z}")

                    theo=v1.theoretical_ticket_count(bet,len(horses))
                    if probs_all.size!=theo:
                        raise SystemExit(f"theoretical ticket count drift y={year} race={rid} bet={bet} array={probs_all.size} theo={theo}")

                    race_stats={
                        "BASE_GAMMA1":{"mass":0.0,"selected":0,"pred":0.0,"actual":0.0,"hit":0},
                        "POWER_CALIBRATED":{"mass":0.0,"selected":0,"pred":0.0,"actual":0.0,"hit":0},
                    }
                    market_q_sum=0.0
                    for key,odd in tickets:
                        p0=max(0.0,v1.ticket_probability(bet,key,q))
                        pc=(p0**gamma)/z
                        market_q_sum+=1.0/max(EPS,odd)
                        for variant,p in (("BASE_GAMMA1",p0),("POWER_CALIBRATED",pc)):
                            m=metrics[(variant,bet)]
                            m["priced_tickets"]+=1
                            race_stats[variant]["mass"]+=p
                            ev=p*odd
                            edge=ev-1.0
                            if edge>0.0:
                                ret=float(payouts.get((bet,key),0.0))
                                predicted=STAKE*ev
                                hit=update_selected(m,ret,predicted,edge)
                                rs=race_stats[variant]
                                rs["selected"]+=1
                                rs["pred"]+=predicted
                                rs["actual"]+=ret
                                rs["hit"]=max(rs["hit"],hit)
                                if variant=="POWER_CALIBRATED":
                                    serial+=1
                                    push_sample(samples[bet],serial,{
                                        "year":year,
                                        "race_id":rid,
                                        "race_date":date,
                                        "bet_type":bet,
                                        "gamma":gamma,
                                        "ticket_numbers":"-".join(map(str,key)),
                                        "base_probability":p0,
                                        "calibrated_probability":pc,
                                        "odds":odd,
                                        "ev":ev,
                                        "edge":edge,
                                        "predicted_return_yen":predicted,
                                        "actual_return_yen":ret,
                                        "hit":hit,
                                        "agreement_confidence":v1.agreement_confidence(key,by_no),
                                    })

                    for variant in VARIANTS:
                        m=metrics[(variant,bet)]
                        m["source_races"]+=1
                        m["theoretical_tickets"]+=theo
                        m["probability_mass_priced"]+=race_stats[variant]["mass"]
                        rs=race_stats[variant]
                        if rs["selected"]:
                            m["selected_races"]+=1
                            m["hit_races"]+=rs["hit"]
                        if top3_nums:
                            wk=v1.winner_key(bet,top3_nums)
                            pwin=max(EPS,v1.ticket_probability(bet,wk,q))
                            if variant=="POWER_CALIBRATED":
                                pwin=max(EPS,(pwin**gamma)/z)
                            m["winning_ticket_nll_sum"]-=math.log(pwin)
                            m["winning_ticket_nll_races"]+=1
                        w.writerow({
                            "year":year,
                            "race_id":rid,
                            "race_date":date,
                            "bet_type":bet,
                            "variant":variant,
                            "gamma":1.0 if variant=="BASE_GAMMA1" else gamma,
                            "field_size":len(horses),
                            "theoretical_tickets":theo,
                            "priced_tickets":len(tickets),
                            "probability_mass_priced":rs["mass"],
                            "selected_tickets":rs["selected"],
                            "predicted_return_yen":rs["pred"],
                            "stake_yen":rs["selected"]*STAKE,
                            "actual_return_yen":rs["actual"],
                            "profit_yen":rs["actual"]-rs["selected"]*STAKE,
                            "hit":rs["hit"],
                            "market_overround":market_q_sum,
                        })

    if len(seen)!=len(l17):
        missing=sorted(set(l17)-seen)[:10]
        raise SystemExit(f"evaluation coverage drift y={year} seen={len(seen)} l17={len(l17)} sample={missing}")
    return metrics,samples


def main():
    a=parse_args()
    paths={y:Path(getattr(a,f"l17_{y}")) for y in YEARS}
    l17={y:load_l17(paths[y],y) for y in YEARS}
    calibration_rows,calibration_counts=v1.collect_calibration_rows(a.backfill_root,l17)

    fold_rows=[]
    gamma_grid_rows=[]
    temperatures={}
    gammas_by_year={}
    for y in TEST_YEARS:
        train_years=[t for t in YEARS if t<y]
        train=[]
        for t in train_years:
            train.extend(calibration_rows[t])
        tf=v1.fit_temperature(train)
        temperature=tf["temperature"]
        temperatures[y]=temperature
        gammas,grid=fit_ticket_gammas(train,temperature)
        gammas_by_year[y]=gammas
        for r in grid:
            r["test_year"]=y
            r["train_years"]="|".join(map(str,train_years))
            r["selected_gamma"]=int(float(r["gamma"])==float(gammas[r["bet_type"]]))
            gamma_grid_rows.append(r)
        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(map(str,train_years)),
            "temperature":temperature,
            "temperature_train_races":tf["train_races"],
            "win_gamma":gammas["WIN"],
            "quinella_gamma":gammas["QUINELLA"],
            "exacta_gamma":gammas["EXACTA"],
            "trio_gamma":gammas["TRIO"],
            "trifecta_gamma":gammas["TRIFECTA"],
        })

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    yearly=[]
    combined={(variant,bet):empty_metric() for variant in VARIANTS for bet in BET_TYPES}
    samples=[]
    race_files=[]

    for y in TEST_YEARS:
        race_file=out/f"race-ticket-summary-{y}.csv.gz"
        metrics,heaps=evaluate_year(
            y,temperatures[y],gammas_by_year[y],l17[y],a.backfill_root,race_file
        )
        race_files.append(race_file.name)
        for variant in VARIANTS:
            for bet in BET_TYPES:
                yearly.append(metric_row(y,bet,variant,metrics[(variant,bet)]))
                v1.add_metric(combined[(variant,bet)],metrics[(variant,bet)])
        for bet in BET_TYPES:
            top=sorted(heaps[bet],key=lambda x:(x[0],x[1]),reverse=True)
            samples.extend([x[2] for x in top])

    combined_rows=[
        metric_row("ALL",bet,variant,combined[(variant,bet)])
        for variant in VARIANTS for bet in BET_TYPES
    ]

    comparison=[]
    by={(r["variant"],r["bet_type"]):r for r in combined_rows}
    for bet in BET_TYPES:
        b=by[("BASE_GAMMA1",bet)]
        c=by[("POWER_CALIBRATED",bet)]
        comparison.append({
            "bet_type":bet,
            "base_roi_pct":b["roi_pct"],
            "calibrated_roi_pct":c["roi_pct"],
            "roi_delta_pctpt":c["roi_pct"]-b["roi_pct"],
            "base_selected_tickets":b["selected_tickets"],
            "calibrated_selected_tickets":c["selected_tickets"],
            "selected_ticket_reduction_pct":100.0*(1.0-c["selected_tickets"]/b["selected_tickets"]) if b["selected_tickets"] else 0.0,
            "base_mean_selected_edge":b["mean_selected_edge"],
            "calibrated_mean_selected_edge":c["mean_selected_edge"],
            "base_winning_ticket_nll":b["winning_ticket_nll"],
            "calibrated_winning_ticket_nll":c["winning_ticket_nll"],
            "winning_ticket_nll_delta":c["winning_ticket_nll"]-b["winning_ticket_nll"],
        })

    v1.write_csv(out/"calibration-folds.csv",fold_rows)
    v1.write_csv(out/"gamma-grid.csv",gamma_grid_rows)
    v1.write_csv(out/"economics-yearly.csv",yearly)
    v1.write_csv(out/"economics-combined.csv",combined_rows)
    v1.write_csv(out/"base-vs-calibrated.csv",comparison)
    v1.write_csv(out/"top-positive-ev-tickets.csv",samples)

    summary={
        "contract":"L2_UNIVERSAL_TICKET_CALIBRATION_V2",
        "parent":"L2_UNIVERSAL_TICKET_EV_V1",
        "diagnosis":"V1 produces severe positive-EV hallucination; V2 recalibrates each ticket distribution without using market prices.",
        "bet_types":list(BET_TYPES),
        "calibration":{
            "horse_level":"same strictly-prior-year temperature calibration as V1",
            "ticket_level":"per-bet power calibration p' = p^gamma / sum(p^gamma) within each race",
            "gamma_candidates":list(GAMMA_GRID),
            "gamma_selection_metric":"minimum prior-year winning-ticket negative log likelihood",
            "roi_used_for_gamma_selection":False,
            "odds_used_for_gamma_selection":False,
            "market_used_for_gamma_selection":False,
            "test_year_used_for_gamma_selection":False,
        },
        "buy_rule":"calibrated_ticket_probability * odds > 1.0",
        "manual_edge_threshold":False,
        "manual_top_k":False,
        "manual_ticket_cap":False,
        "manual_odds_band":False,
        "manual_race_skip_rule":False,
        "stake_policy":"flat 100 yen research stake; L3 remains separate",
        "market_price_stage":"historical FINAL odds proxy",
        "market_price_caveat":"FINAL odds are not live timestamp execution prices",
        "folds":fold_rows,
        "calibration_data_counts":calibration_counts,
        "race_summary_files":race_files,
        "large_ticket_matrices_persisted":False,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Universal Ticket Calibration V2\n\n"
        "V1 showed severe positive-EV hallucination: ticket probabilities were too generous in the tails. "
        "V2 keeps the same market-free L1.7 probability world, then calibrates each bet-type ticket distribution "
        "with p' = p^gamma / sum(p^gamma) inside each race. Gamma is selected only by strictly-prior-year "
        "winning-ticket negative log likelihood. Odds, ROI, payout and the test year are not used to choose gamma. "
        "The betting rule remains exactly p' * odds > 1.0 with no human edge floor, Top-K, ticket cap, odds band, "
        "or race skip rule. 2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_UNIVERSAL_TICKET_CALIBRATION_V2_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
    print("===== BASE VS CALIBRATED =====")
    with open(out/"base-vs-calibrated.csv",encoding="utf-8") as fh:
        print(fh.read())


if __name__=="__main__":
    main()
