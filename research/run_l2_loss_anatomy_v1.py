#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import (
    TEMPLATE_TO_BET,
    canonical_numbers,
    generate_templates,
    horse_number_map,
    payout_map,
    seven_stats,
)

YEARS=(2024,2025)
EXPECTED_PER_YEAR=3456
ROUTERS=("SIMPLE_EXPECTED_ROI","STRATEGY_UTILITY","DIRECT_BET_TYPE")
BET_REQUIRED_FINISH={
    "WIN":1,
    "QUINELLA":2,
    "EXACTA":2,
    "TRIO":3,
    "TRIFECTA":3,
}


def parse_args():
    p=argparse.ArgumentParser(description="Diagnose why selected L2 Router bets lose without retraining Router/P(hit).")
    p.add_argument("--selections",required=True)
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")


def parse_year_paths(items):
    out={}
    for spec in items:
        year,path=spec.split(":",1)
        out[int(year)]=path
    return out


def load_csv(path):
    with open(path,newline="",encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def finite(v,default=0.0):
    try:
        return float(v)
    except (TypeError,ValueError):
        return default


def load_routers(paths):
    out={}
    for year in YEARS:
        rows={}
        with open_text(paths[year]) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row=json.loads(line)
                rid=str(row.get("race_id") or "")
                if not rid or rid in rows:
                    raise SystemExit(f"bad router race_id y{year}: {rid}")
                rows[rid]=row
        if len(rows)!=EXPECTED_PER_YEAR:
            raise SystemExit(f"router count regression y{year}: {len(rows)} != {EXPECTED_PER_YEAR}")
        out[year]=rows
    return out


def load_fixed(root):
    out={}
    root=Path(root)
    for year in YEARS:
        rows={}
        path=root/f"y{year}.jsonl"
        with open(path,encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row=json.loads(line)
                rid=str(row.get("race_id") or "")
                if rid:
                    rows[rid]=row
        out[year]=rows
    return out


def load_selected(selections):
    rows=[]
    wanted=defaultdict(set)
    for r in load_csv(selections):
        router=str(r.get("router") or "")
        year=int(float(r.get("test_year") or 0))
        rid=str(r.get("race_id") or "")
        if router not in ROUTERS or year not in YEARS or not rid:
            continue
        r["router"]=router
        r["test_year"]=year
        r["ticket_count"]=int(float(r.get("ticket_count") or 0))
        r["actual_return_yen"]=finite(r.get("actual_return_yen"))
        r["actual_profit_yen"]=finite(r.get("actual_profit_yen"))
        r["actual_roi_pct"]=finite(r.get("actual_roi_pct"))
        r["actual_hit"]=1 if str(r.get("actual_hit") or "").strip().lower() in {"1","true"} else 0
        rows.append(r)
        wanted[str(r.get("race_date") or "")[:10]].add(rid)
    return rows,wanted


def load_race_packs(backfill_root,wanted):
    root=Path(backfill_root)/"data"/"daily"
    out={}
    missing_dates=[]
    for date,rids in sorted(wanted.items()):
        path=root/f"{date}.jsonl.gz"
        if not path.exists():
            missing_dates.append(date)
            continue
        with gzip.open(path,"rt",encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row=json.loads(line)
                rid=str((row.get("race") or {}).get("race_id") or "")
                if rid in rids:
                    out[rid]=row
    if missing_dates:
        raise SystemExit(f"missing daily packs: {missing_dates[:10]}")
    missing=sorted({rid for ids in wanted.values() for rid in ids}-set(out))
    if missing:
        raise SystemExit(f"missing selected race packs: {missing[:10]}")
    return out


def finish_maps(race_pack):
    pos_to_ids=defaultdict(list)
    hid_to_pos={}
    for r in race_pack.get("results") or []:
        status=str(r.get("result_status") or "").upper()
        try:
            pos=int(r.get("official_finish_position"))
        except (TypeError,ValueError):
            continue
        hid=str(r.get("horse_id") or "")
        if status!="FINISHED" or pos<=0 or not hid:
            continue
        pos_to_ids[pos].append(hid)
        hid_to_pos[hid]=pos
    return dict(pos_to_ids),hid_to_pos


def candidate_pool(router_row,fixed_row):
    seven_order,seven_map=seven_stats(router_row)
    if fixed_row is not None:
        fixed_order=[str(x) for x in fixed_row.get("seven_consensus_order") or []]
        if fixed_order and fixed_order!=seven_order:
            raise ValueError(f"seven consensus drift race={router_row.get('race_id')}")
        candidates=[str(x) for x in fixed_row.get("candidate_horse_ids") or []]
        novel=[str(x) for x in fixed_row.get("novel_horse_ids") or []]
        gate=True
    else:
        candidates=list(seven_order)
        novel=[]
        gate=False
    return seven_order,candidates,novel,gate


def winning_combos(race_pack,bet_type):
    payouts,_=payout_map(race_pack)
    return {
        nums for (bt,nums),value in payouts.items()
        if bt==bet_type and float(value)>0
    }


def raw_template_combos(template,bet_type,seven_order,novel,candidates,horse_no):
    generated=generate_templates(seven_order,novel,candidates)
    rows=generated.get(template) or []
    out=set()
    for hids in rows:
        try:
            nums=[horse_no[x] for x in hids]
        except KeyError:
            continue
        out.add(canonical_numbers(bet_type,nums))
    return out


def podium_coverage(pos_to_ids,candidates):
    cand=set(candidates)
    podium=[]
    for pos in (1,2,3):
        podium.extend(pos_to_ids.get(pos,[]))
    covered=[hid for hid in podium if hid in cand]
    return podium,covered


def classify(row,race_pack,router_row,fixed_row):
    bet_type=str(row["bet_type"])
    template=str(row["template"])
    if TEMPLATE_TO_BET.get(template)!=bet_type:
        return {"cause":"DATA_OR_CONTRACT_GAP","detail":"template_bet_type_mismatch"}

    seven_order,candidates,novel,gate=candidate_pool(router_row,fixed_row)
    horse_no=horse_number_map(race_pack)
    pos_to_ids,_=finish_maps(race_pack)
    podium,podium_covered=podium_coverage(pos_to_ids,candidates)
    wins=winning_combos(race_pack,bet_type)

    if not wins:
        return {
            "cause":"DATA_OR_CONTRACT_GAP",
            "detail":"no_winning_payout_combo",
            "gate_alert":int(gate),
            "candidate_pool_size":len(candidates),
            "novel_count":len(novel),
            "podium_total":len(podium),
            "podium_covered":len(podium_covered),
        }

    candidate_numbers={horse_no[h] for h in candidates if h in horse_no}
    candidate_can_hit=any(set(combo).issubset(candidate_numbers) for combo in wins)

    raw=raw_template_combos(template,bet_type,seven_order,novel,candidates,horse_no)
    template_can_hit=bool(raw & wins)

    if row["actual_profit_yen"]>=0:
        cause="NON_LOSS"
        detail="profitable_or_break_even"
    elif row["actual_hit"]==1 or row["actual_return_yen"]>0:
        cause="HIT_BUT_NEGATIVE"
        detail="winning_action_but_stake_exceeded_return"
    elif not candidate_can_hit:
        cause="HORSE_SELECTION_MISS"
        detail="candidate_pool_cannot_form_any_winning_combo"
    elif not template_can_hit:
        cause="TICKET_TEMPLATE_MISS"
        detail="candidate_pool_can_hit_but_raw_template_cannot"
    else:
        cause="EDGE_OR_PRICE_FILTER_MISS"
        detail="raw_template_contains_winner_but_purchased_edge_filtered_action_missed"

    required=BET_REQUIRED_FINISH.get(bet_type,0)
    required_finish_ids=[]
    for pos in range(1,required+1):
        required_finish_ids.extend(pos_to_ids.get(pos,[]))
    required_covered=sum(hid in set(candidates) for hid in required_finish_ids)

    return {
        "cause":cause,
        "detail":detail,
        "gate_alert":int(gate),
        "candidate_pool_size":len(candidates),
        "seven_union_size":len(seven_order),
        "novel_count":len(novel),
        "podium_total":len(podium),
        "podium_covered":len(podium_covered),
        "podium_coverage_rate":len(podium_covered)/len(podium) if podium else None,
        "required_finish_horses":len(required_finish_ids),
        "required_finish_covered":required_covered,
        "candidate_can_hit":int(candidate_can_hit),
        "template_can_hit":int(template_can_hit),
        "winning_combo_count":len(wins),
        "raw_template_combo_count":len(raw),
    }


def write_csv(path,rows,gzip_output=False):
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
    if gzip_output:
        with gzip.open(path,"wt",newline="",encoding="utf-8") as fh:
            w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    else:
        with open(path,"w",newline="",encoding="utf-8") as fh:
            w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore")
            w.writeheader(); w.writerows(rows)


def summarize(diag):
    groups=defaultdict(list)
    for r in diag:
        if r["actual_profit_yen"]<0:
            groups[(r["router"],r["test_year"],r["cause"])].append(r)
    rows=[]
    for (router,year,cause),items in sorted(groups.items()):
        stake=sum(100.0*r["ticket_count"] for r in items)
        ret=sum(r["actual_return_yen"] for r in items)
        loss=-sum(r["actual_profit_yen"] for r in items)
        rows.append({
            "router":router,
            "test_year":year,
            "cause":cause,
            "losing_races":len(items),
            "stake_yen":stake,
            "return_yen":ret,
            "loss_yen":loss,
            "avg_podium_coverage":sum(finite(r.get("podium_coverage_rate")) for r in items)/len(items),
            "gate_share":sum(int(r.get("gate_alert",0)) for r in items)/len(items),
            "k2_candidate_share":sum(int(r.get("novel_count",0)>0) for r in items)/len(items),
        })
    totals=defaultdict(lambda:{"races":0,"loss":0.0})
    for r in rows:
        k=(r["router"],r["test_year"])
        totals[k]["races"]+=r["losing_races"]
        totals[k]["loss"]+=r["loss_yen"]
    for r in rows:
        t=totals[(r["router"],r["test_year"])]
        r["share_of_losing_races_pct"]=100.0*r["losing_races"]/t["races"] if t["races"] else None
        r["share_of_loss_yen_pct"]=100.0*r["loss_yen"]/t["loss"] if t["loss"] else None
    return rows


def summarize_bet_type(diag):
    groups=defaultdict(list)
    for r in diag:
        if r["actual_profit_yen"]<0:
            groups[(r["router"],r["test_year"],r["bet_type"],r["cause"])].append(r)
    rows=[]
    for (router,year,bet,cause),items in sorted(groups.items()):
        rows.append({
            "router":router,
            "test_year":year,
            "bet_type":bet,
            "cause":cause,
            "losing_races":len(items),
            "loss_yen":-sum(r["actual_profit_yen"] for r in items),
            "avg_podium_coverage":sum(finite(r.get("podium_coverage_rate")) for r in items)/len(items),
        })
    return rows


def coverage_summary(diag):
    groups=defaultdict(list)
    for r in diag:
        groups[(r["router"],r["test_year"],"LOSS" if r["actual_profit_yen"]<0 else "NON_LOSS")].append(r)
    rows=[]
    for (router,year,status),items in sorted(groups.items()):
        dist=Counter(int(r.get("podium_covered") or 0) for r in items)
        total=len(items)
        rows.append({
            "router":router,
            "test_year":year,
            "status":status,
            "races":total,
            "avg_podium_coverage":sum(finite(r.get("podium_coverage_rate")) for r in items)/total if total else None,
            "podium_covered_0":dist.get(0,0),
            "podium_covered_1":dist.get(1,0),
            "podium_covered_2":dist.get(2,0),
            "podium_covered_3plus":sum(v for k,v in dist.items() if k>=3),
        })
    return rows


def main():
    a=parse_args()
    paths=parse_year_paths(a.router_year)
    if set(paths)!=set(YEARS):
        raise SystemExit(f"need router years {YEARS}, got {sorted(paths)}")

    selected,wanted=load_selected(a.selections)
    routers=load_routers(paths)
    fixed=load_fixed(a.fixed_ledger_dir)
    packs=load_race_packs(a.backfill_root,wanted)

    diag=[]
    for i,row in enumerate(selected,1):
        year=row["test_year"]
        rid=str(row["race_id"])
        router_row=routers[year].get(rid)
        if router_row is None:
            raise SystemExit(f"router row missing y{year} race={rid}")
        d=classify(row,packs[rid],router_row,fixed[year].get(rid))
        diag.append({
            "router":row["router"],
            "test_year":year,
            "race_id":rid,
            "race_date":row["race_date"],
            "bet_type":row["bet_type"],
            "template":row["template"],
            "edge_threshold":row["edge_threshold"],
            "ticket_count":row["ticket_count"],
            "expected_roi":row.get("expected_roi"),
            "actual_return_yen":row["actual_return_yen"],
            "actual_profit_yen":row["actual_profit_yen"],
            "actual_roi_pct":row["actual_roi_pct"],
            "actual_hit":row["actual_hit"],
            "novel_ticket_share":row.get("novel_ticket_share"),
            **d,
        })
        if i%1000==0:
            print(f"LOSS_ANATOMY_PROGRESS {i}/{len(selected)}",flush=True)

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    cause_rows=summarize(diag)
    bet_rows=summarize_bet_type(diag)
    cov_rows=coverage_summary(diag)

    write_csv(out/"loss-causes.csv",cause_rows)
    write_csv(out/"loss-causes-by-bet-type.csv",bet_rows)
    write_csv(out/"candidate-podium-coverage.csv",cov_rows)
    write_csv(out/"race-diagnostics.csv.gz",diag,gzip_output=True)

    direct={}
    for year in YEARS:
        rows=[r for r in cause_rows if r["router"]=="DIRECT_BET_TYPE" and r["test_year"]==year]
        direct[str(year)]={
            r["cause"]:{
                "losing_races":r["losing_races"],
                "share_of_losing_races_pct":r["share_of_losing_races_pct"],
                "loss_yen":r["loss_yen"],
                "share_of_loss_yen_pct":r["share_of_loss_yen_pct"],
                "avg_podium_coverage":r["avg_podium_coverage"],
            }
            for r in rows
        }
    summary={
        "contract":"L2_LOSS_ANATOMY_RESULT_V1",
        "source_router_run":36511322402,
        "source_router_commit":"8bd55e406685372c471a179e38c5b17418987fdf",
        "analysis_years":[2024,2025],
        "selected_actions":len(selected),
        "retrained_phit":False,
        "rerun_router":False,
        "2026_locked":True,
        "direct_bet_type_loss_causes":direct,
        "cause_definitions":{
            "HORSE_SELECTION_MISS":"Candidate pool cannot form any official winning combination for the selected bet type.",
            "TICKET_TEMPLATE_MISS":"Candidate pool can form a winner, but the raw selected template does not contain an official winning combination.",
            "EDGE_OR_PRICE_FILTER_MISS":"Raw template contains an official winner, but the purchased edge-filtered action missed it.",
            "HIT_BUT_NEGATIVE":"Purchased action hit but total return was below total stake.",
        },
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Loss Anatomy V1\n\n"
        "This audit reuses archived Router selections from run 36511322402. "
        "It does not retrain P(hit) and does not rerun any Router.\n\n"
        "Losses are split into horse-selection miss, ticket-template miss, "
        "edge/price-filter miss, and hit-but-negative. Candidate podium coverage "
        "is reported separately. 2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_LOSS_ANATOMY_V1_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)


if __name__=="__main__":
    main()
