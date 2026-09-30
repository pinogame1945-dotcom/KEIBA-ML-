#!/usr/bin/env python3
import argparse
import csv
import gzip
import heapq
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.optimize import minimize_scalar

from build_l2_bet_kings_dataset_v1 import decode_odds, payout_map, horse_number_map
from build_l2_l17_fullfield_dataset_v1 import load_l17

YEARS=(2021,2022,2023,2024,2025)
TEST_YEARS=(2022,2023,2024,2025)
BET_TYPES=("WIN","QUINELLA","EXACTA","TRIO","TRIFECTA")
EPS=1e-12
STAKE=100.0
TOP_SAMPLE_PER_BET_YEAR=50


def parse_args():
    p=argparse.ArgumentParser(description="Universal all-ticket EV engine over L1.7 full-field consensus.")
    for y in YEARS:
        p.add_argument(f"--l17-{y}", required=True)
    p.add_argument("--backfill-root", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def finite(v, default=None):
    try:
        if isinstance(v, str):
            v=v.replace(",","").strip()
        x=float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def race_q(rec, temperature):
    horses=rec.get("horses") or []
    if not horses:
        raise ValueError(f"empty L1.7 race={rec.get('race_id')}")
    raw={}
    max_log=-1e300
    inv_t=1.0/max(float(temperature), 1e-9)
    for h in horses:
        no=int(h["horse_number"])
        p=max(EPS, finite(h.get("mean_probability"), EPS))
        z=math.log(p)*inv_t
        raw[no]=z
        max_log=max(max_log,z)
    weights={no:math.exp(z-max_log) for no,z in raw.items()}
    denom=sum(weights.values())
    if denom<=0:
        raise ValueError(f"invalid probability denominator race={rec.get('race_id')}")
    return {no:w/denom for no,w in weights.items()}


def exacta_prob(a,b,q):
    if a==b or a not in q or b not in q:
        return 0.0
    qa=q[a]
    return max(0.0, qa*(q[b]/max(EPS,1.0-qa)))


def trifecta_prob(a,b,c,q):
    if len({a,b,c})!=3 or any(x not in q for x in (a,b,c)):
        return 0.0
    qa,qb,qc=q[a],q[b],q[c]
    return max(0.0, qa*(qb/max(EPS,1.0-qa))*(qc/max(EPS,1.0-qa-qb)))


def ticket_probability(bet_type, nums, q):
    nums=tuple(int(x) for x in nums)
    if bet_type=="WIN":
        return q.get(nums[0],0.0) if len(nums)==1 else 0.0
    if bet_type=="EXACTA":
        return exacta_prob(nums[0],nums[1],q) if len(nums)==2 else 0.0
    if bet_type=="QUINELLA":
        if len(nums)!=2:
            return 0.0
        a,b=nums
        return exacta_prob(a,b,q)+exacta_prob(b,a,q)
    if bet_type=="TRIFECTA":
        return trifecta_prob(nums[0],nums[1],nums[2],q) if len(nums)==3 else 0.0
    if bet_type=="TRIO":
        if len(nums)!=3 or len(set(nums))!=3:
            return 0.0
        a,b,c=nums
        return (
            trifecta_prob(a,b,c,q)+trifecta_prob(a,c,b,q)+
            trifecta_prob(b,a,c,q)+trifecta_prob(b,c,a,q)+
            trifecta_prob(c,a,b,q)+trifecta_prob(c,b,a,q)
        )
    raise ValueError(bet_type)


def theoretical_ticket_count(bet_type,n):
    if bet_type=="WIN":
        return n
    if bet_type=="QUINELLA":
        return n*(n-1)//2
    if bet_type=="EXACTA":
        return n*(n-1)
    if bet_type=="TRIO":
        return n*(n-1)*(n-2)//6 if n>=3 else 0
    if bet_type=="TRIFECTA":
        return n*(n-1)*(n-2) if n>=3 else 0
    raise ValueError(bet_type)


def top3_from_pack(pack, allowed_hids):
    by_pos=defaultdict(list)
    for r in pack.get("results") or []:
        if str(r.get("result_status") or "").upper()!="FINISHED":
            continue
        hid=str(r.get("horse_id") or "")
        if hid not in allowed_hids:
            continue
        try:
            pos=int(r.get("official_finish_position"))
        except (TypeError,ValueError):
            continue
        if pos in (1,2,3):
            by_pos[pos].append(hid)
    if any(len(by_pos[p])!=1 for p in (1,2,3)):
        return None
    return (by_pos[1][0],by_pos[2][0],by_pos[3][0])


def iter_year_packs(backfill_root, year):
    root=Path(backfill_root)
    daily=root/"data"/"daily"
    files=sorted(daily.glob(f"{year}-*.jsonl.gz"))
    if not files:
        raise SystemExit(f"no daily packs for year={year}")
    for path in files:
        date=path.name[:10]
        with gzip.open(path,"rt",encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                yield date,json.loads(line)


def collect_calibration_rows(backfill_root, l17_by_year):
    rows={y:[] for y in YEARS}
    counts={y:{"seen":0,"usable_top3":0,"excluded_ties_or_missing":0} for y in YEARS}
    for y in YEARS:
        l17=l17_by_year[y]
        for date,pack in iter_year_packs(backfill_root,y):
            rid=str((pack.get("race") or {}).get("race_id") or "")
            rec=l17.get(rid)
            if rec is None:
                continue
            counts[y]["seen"]+=1
            horses=rec.get("horses") or []
            allowed={str(h.get("horse_id") or "") for h in horses}
            top3=top3_from_pack(pack,allowed)
            if top3 is None:
                counts[y]["excluded_ties_or_missing"]+=1
                continue
            hid_to_no={str(h["horse_id"]):int(h["horse_number"]) for h in horses}
            try:
                top3_nums=tuple(hid_to_no[h] for h in top3)
            except KeyError:
                counts[y]["excluded_ties_or_missing"]+=1
                continue
            base={int(h["horse_number"]):max(EPS,finite(h.get("mean_probability"),EPS)) for h in horses}
            rows[y].append((base,top3_nums))
            counts[y]["usable_top3"]+=1
        if counts[y]["seen"]!=len(l17):
            raise SystemExit(f"L1.7/daily coverage drift y={y} seen={counts[y]['seen']} l17={len(l17)}")
    return rows,counts


def q_from_base(base, temperature):
    inv_t=1.0/max(float(temperature),1e-9)
    logs={k:math.log(max(EPS,v))*inv_t for k,v in base.items()}
    m=max(logs.values())
    w={k:math.exp(v-m) for k,v in logs.items()}
    s=sum(w.values())
    return {k:v/s for k,v in w.items()}


def top3_nll_for_temperature(rows, temperature):
    total=0.0
    used=0
    for base,top3 in rows:
        q=q_from_base(base,temperature)
        a,b,c=top3
        p=max(EPS,trifecta_prob(a,b,c,q))
        total-=math.log(p)
        used+=1
    return total/max(1,used)


def fit_temperature(rows):
    if not rows:
        raise SystemExit("no prior calibration races")
    baseline=top3_nll_for_temperature(rows,1.0)
    res=minimize_scalar(
        lambda t: top3_nll_for_temperature(rows,t),
        bounds=(0.20,5.00),
        method="bounded",
        options={"xatol":1e-4,"maxiter":100},
    )
    if not res.success:
        raise SystemExit(f"temperature optimization failed: {res}")
    t=float(res.x)
    return {
        "temperature":t,
        "train_races":len(rows),
        "train_nll_t1":baseline,
        "train_nll_calibrated":top3_nll_for_temperature(rows,t),
    }


def winner_key(bet_type, top3_nums):
    a,b,c=top3_nums
    if bet_type=="WIN":
        return (a,)
    if bet_type=="QUINELLA":
        return tuple(sorted((a,b)))
    if bet_type=="EXACTA":
        return (a,b)
    if bet_type=="TRIO":
        return tuple(sorted((a,b,c)))
    if bet_type=="TRIFECTA":
        return (a,b,c)
    raise ValueError(bet_type)


def agreement_confidence(nums, by_no):
    vals=[]
    for no in nums:
        h=by_no.get(int(no))
        if not h:
            continue
        mean=max(EPS,finite(h.get("mean_probability"),EPS))
        std=max(0.0,finite(h.get("probability_std"),0.0))
        vals.append(1.0/(1.0+std/mean))
    return sum(vals)/len(vals) if vals else 0.0


def empty_metric():
    return {
        "source_races":0,
        "priced_tickets":0,
        "theoretical_tickets":0,
        "probability_mass_priced":0.0,
        "selected_tickets":0,
        "selected_races":0,
        "hit_tickets":0,
        "hit_races":0,
        "stake_yen":0.0,
        "actual_return_yen":0.0,
        "predicted_return_yen":0.0,
        "edge_sum":0.0,
        "winning_ticket_nll_sum":0.0,
        "winning_ticket_nll_races":0,
    }


def add_metric(dst, src):
    for k,v in src.items():
        dst[k]+=v


def metric_row(year,bet,m):
    selected=m["selected_tickets"]
    source=m["source_races"]
    stake=m["stake_yen"]
    return {
        "year":year,
        "bet_type":bet,
        "source_races":source,
        "priced_tickets":m["priced_tickets"],
        "theoretical_tickets":m["theoretical_tickets"],
        "priced_ticket_coverage_pct":100.0*m["priced_tickets"]/m["theoretical_tickets"] if m["theoretical_tickets"] else 0.0,
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
        "roi_pct":100.0*m["actual_return_yen"]/stake if stake else 0.0,
        "predicted_return_yen":m["predicted_return_yen"],
        "predicted_profit_yen":m["predicted_return_yen"]-stake,
        "mean_selected_edge":m["edge_sum"]/selected if selected else None,
        "winning_ticket_nll":m["winning_ticket_nll_sum"]/m["winning_ticket_nll_races"] if m["winning_ticket_nll_races"] else None,
    }


def push_sample(heap, serial, rec):
    item=(float(rec["edge"]),serial,rec)
    if len(heap)<TOP_SAMPLE_PER_BET_YEAR:
        heapq.heappush(heap,item)
    elif item[0]>heap[0][0]:
        heapq.heapreplace(heap,item)


def evaluate_year(year, temperature, l17, backfill_root, out_race_gz):
    root=Path(backfill_root)
    metrics={bet:empty_metric() for bet in BET_TYPES}
    samples={bet:[] for bet in BET_TYPES}
    serial=0
    seen=set()

    with gzip.open(out_race_gz,"wt",newline="",encoding="utf-8") as race_fh:
        fields=[
            "year","race_id","race_date","bet_type","field_size",
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

                q=race_q(rec,temperature)
                odds_map=decode_odds(odds_record)
                payouts,present=payout_map(pack)
                allowed=set(hid_to_no)
                top3=top3_from_pack(pack,allowed)
                top3_nums=tuple(hid_to_no[x] for x in top3) if top3 else None

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
                    m=metrics[bet]
                    m["source_races"]+=1
                    theo=theoretical_ticket_count(bet,len(horses))
                    m["theoretical_tickets"]+=theo

                    probability_mass=0.0
                    market_q_sum=0.0
                    selected=0
                    race_pred=0.0
                    race_actual=0.0
                    race_hit=0
                    for key,odd in tickets:
                        p=ticket_probability(bet,key,q)
                        if p<0 or not math.isfinite(p):
                            raise SystemExit(f"bad probability y={year} race={rid} bet={bet} key={key} p={p}")
                        probability_mass+=p
                        market_q_sum+=1.0/max(EPS,odd)
                        m["priced_tickets"]+=1
                        ev=p*odd
                        edge=ev-1.0
                        if edge>0.0:
                            ret=float(payouts.get((bet,key),0.0))
                            selected+=1
                            serial+=1
                            predicted=STAKE*ev
                            actual=ret
                            hit=int(ret>0)
                            race_pred+=predicted
                            race_actual+=actual
                            race_hit=max(race_hit,hit)
                            m["selected_tickets"]+=1
                            m["hit_tickets"]+=hit
                            m["stake_yen"]+=STAKE
                            m["actual_return_yen"]+=actual
                            m["predicted_return_yen"]+=predicted
                            m["edge_sum"]+=edge
                            sample={
                                "year":year,
                                "race_id":rid,
                                "race_date":date,
                                "bet_type":bet,
                                "ticket_numbers":"-".join(map(str,key)),
                                "probability":p,
                                "odds":odd,
                                "ev":ev,
                                "edge":edge,
                                "predicted_return_yen":predicted,
                                "actual_return_yen":actual,
                                "hit":hit,
                                "agreement_confidence":agreement_confidence(key,by_no),
                            }
                            push_sample(samples[bet],serial,sample)
                    m["probability_mass_priced"]+=probability_mass
                    if selected:
                        m["selected_races"]+=1
                        m["hit_races"]+=race_hit
                    if top3_nums:
                        wk=winner_key(bet,top3_nums)
                        wp=max(EPS,ticket_probability(bet,wk,q))
                        m["winning_ticket_nll_sum"]-=math.log(wp)
                        m["winning_ticket_nll_races"]+=1

                    w.writerow({
                        "year":year,
                        "race_id":rid,
                        "race_date":date,
                        "bet_type":bet,
                        "field_size":len(horses),
                        "theoretical_tickets":theo,
                        "priced_tickets":len(tickets),
                        "probability_mass_priced":probability_mass,
                        "selected_tickets":selected,
                        "predicted_return_yen":race_pred,
                        "stake_yen":selected*STAKE,
                        "actual_return_yen":race_actual,
                        "profit_yen":race_actual-selected*STAKE,
                        "hit":race_hit,
                        "market_overround":market_q_sum,
                    })

    if len(seen)!=len(l17):
        missing=sorted(set(l17)-seen)[:10]
        raise SystemExit(f"evaluation coverage drift y={year} seen={len(seen)} l17={len(l17)} sample={missing}")
    return metrics,samples


def write_csv(path, rows):
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


def probability_self_test():
    q={1:0.40,2:0.30,3:0.20,4:0.10}
    checks={}
    checks["WIN"]=sum(ticket_probability("WIN",(a,),q) for a in q)
    checks["QUINELLA"]=sum(ticket_probability("QUINELLA",x,q) for x in itertools.combinations(q,2))
    checks["EXACTA"]=sum(ticket_probability("EXACTA",x,q) for x in itertools.permutations(q,2))
    checks["TRIO"]=sum(ticket_probability("TRIO",x,q) for x in itertools.combinations(q,3))
    checks["TRIFECTA"]=sum(ticket_probability("TRIFECTA",x,q) for x in itertools.permutations(q,3))
    for k,v in checks.items():
        if abs(v-1.0)>1e-9:
            raise AssertionError((k,v))
    return checks


def main():
    a=parse_args()
    paths={y:Path(getattr(a,f"l17_{y}")) for y in YEARS}
    l17={y:load_l17(paths[y],y) for y in YEARS}
    if any(y==2026 for y in l17):
        raise SystemExit("2026 sealed")

    self_test=probability_self_test()
    calibration_rows,calibration_counts=collect_calibration_rows(a.backfill_root,l17)

    folds=[]
    temperatures={}
    for y in TEST_YEARS:
        train_years=[t for t in YEARS if t<y]
        rows=[]
        for t in train_years:
            rows.extend(calibration_rows[t])
        fit=fit_temperature(rows)
        temperatures[y]=fit["temperature"]
        test_nll=top3_nll_for_temperature(calibration_rows[y],fit["temperature"])
        folds.append({
            "test_year":y,
            "train_years":"|".join(map(str,train_years)),
            **fit,
            "test_usable_top3_races":len(calibration_rows[y]),
            "test_top3_nll":test_nll,
        })

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)

    yearly_rows=[]
    combined={bet:empty_metric() for bet in BET_TYPES}
    sample_rows=[]
    race_files=[]
    for y in TEST_YEARS:
        race_file=out/f"race-ticket-summary-{y}.csv.gz"
        metrics,samples=evaluate_year(y,temperatures[y],l17[y],a.backfill_root,race_file)
        race_files.append(race_file.name)
        for bet in BET_TYPES:
            yearly_rows.append(metric_row(y,bet,metrics[bet]))
            add_metric(combined[bet],metrics[bet])
            top=sorted(samples[bet],key=lambda x:(x[0],x[1]),reverse=True)
            sample_rows.extend([x[2] for x in top])

    combined_rows=[metric_row("ALL",bet,combined[bet]) for bet in BET_TYPES]
    write_csv(out/"calibration-folds.csv",folds)
    write_csv(out/"economics-yearly.csv",yearly_rows)
    write_csv(out/"economics-combined.csv",combined_rows)
    write_csv(out/"top-positive-ev-tickets.csv",sample_rows)

    summary={
        "contract":"L2_UNIVERSAL_TICKET_EV_V1",
        "candidate_universe":"all priced tickets present in historical odds for every L1.7 full-field runner",
        "bet_types":list(BET_TYPES),
        "probability_model":{
            "family":"temperature-calibrated Plackett-Luce decomposition",
            "horse_strength_source":"L1.7 seven-king mean_probability only",
            "odds_used_in_probability_model":False,
            "market_used_in_probability_model":False,
            "payout_used_in_probability_model":False,
            "finish_results_used_for_temperature_training":"strictly prior years only",
            "temperature_bounds":[0.20,5.00],
        },
        "buy_rule":"ticket_probability * odds > 1.0",
        "manual_edge_threshold":False,
        "manual_top_k":False,
        "manual_ticket_cap":False,
        "manual_odds_band":False,
        "manual_race_skip_rule":False,
        "stake_policy":"research evaluation uses flat 100 yen per EV-positive ticket; L3 stake optimization remains separate",
        "market_price_stage":"historical FINAL odds proxy",
        "market_price_caveat":"FINAL odds are not a live timestamp execution price and must not be treated as production-realistic backtest pricing",
        "walk_forward":{
            "calibration_test_years":list(TEST_YEARS),
            "each_test_year_temperature_uses":"all prior years only",
        },
        "calibration_data_counts":calibration_counts,
        "temperature_by_test_year":{str(y):temperatures[y] for y in TEST_YEARS},
        "probability_self_test":self_test,
        "race_summary_files":race_files,
        "large_ticket_matrices_persisted":False,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Universal Ticket EV V1\n\n"
        "All market-priced WIN / QUINELLA / EXACTA / TRIO / TRIFECTA tickets are evaluated from the "
        "same L1.7 full-field probability world. L1.7 mean horse probabilities are temperature-calibrated "
        "with strictly prior-year official top-3 results, then decomposed with a Plackett-Luce model into "
        "ordered and unordered ticket probabilities. Odds do not enter the probability model. The only buy "
        "rule is probability × odds > 1.0. There is no human edge threshold, Top-K, ticket cap, odds band, "
        "or race-skip rule. Historical final odds are only a research execution-price proxy. Flat 100-yen "
        "stakes are used for L2 economics; stake optimization belongs to L3. Full ticket matrices stay "
        "runner-ephemeral; compact race-level summaries and metrics are committed. 2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_UNIVERSAL_TICKET_EV_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
    print("===== ECONOMICS COMBINED =====")
    with open(out/"economics-combined.csv",encoding="utf-8") as fh:
        print(fh.read())


if __name__=="__main__":
    main()
