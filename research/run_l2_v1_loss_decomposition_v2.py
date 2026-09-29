#!/usr/bin/env python3
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import (
    YEARS,load_fixed_ledgers,load_router,seven_stats,load_day,payout_map,horse_number_map
)
from run_l2_normal_router_v1 import (
    ANALYSIS_YEARS,TEMPLATES,TEMPLATE_TO_BET,
    build_feature_rows,load_performance,attach_targets,feature_columns,
    train_predict,choose_available,evaluate
)
from run_l2_normal_loss_anatomy_v1 import winning_combos,any_fully_covered

TEST_YEARS=(2023,2024,2025)
ROOT_ORDER=(
    "V1_PROFITABLE",
    "L1_CANDIDATE_IMPOSSIBLE",
    "L2_ROUTING_MISS_SAME_BET",
    "L2_ROUTING_MISS_CROSS_BET",
    "L2_ROLE_TEMPLATE_MISS",
    "L2_HIT_BUT_PRICE_POINTS_LOSS",
    "PAYOUT_INCOMPLETE",
    "DATA_UNAVAILABLE",
)

def parse_args():
    p=argparse.ArgumentParser(description="V1 loss decomposition V2.")
    p.add_argument("--contract",required=True)
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--v1-metrics",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1)
        out[int(y)]=p
    return out

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def load_reference_metrics(path):
    out={}
    with open(path,newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("architecture")=="DIRECT_TEMPLATE":
                out[int(r["test_year"])]=r
    return out

def best_item(items):
    if not items: return None
    return max(items,key=lambda x:(x["profit_yen"],x["return_yen"],-x["stake_yen"],x["template"]))

def summarize_period(label,rows):
    n=len(rows)
    executed=[r for r in rows if r["root_cause"]!="DATA_UNAVAILABLE"]
    losses=[r for r in rows if r["root_cause"] not in ("V1_PROFITABLE","DATA_UNAVAILABLE")]
    stake=sum(float(r["selected_stake_yen"]) for r in executed)
    ret=sum(float(r["selected_return_yen"]) for r in executed)
    recoverable=[r for r in losses if r["root_cause"] in ("L2_ROUTING_MISS_SAME_BET","L2_ROUTING_MISS_CROSS_BET")]
    return {
        "period":label,
        "races":n,
        "executed_races":len(executed),
        "v1_profitable_races":sum(r["root_cause"]=="V1_PROFITABLE" for r in rows),
        "v1_nonprofitable_races":len(losses),
        "v1_stake_yen":stake,
        "v1_return_yen":ret,
        "v1_profit_yen":ret-stake,
        "v1_roi_pct":100*ret/stake if stake else None,
        "l1_candidate_impossible_races":sum(r["root_cause"]=="L1_CANDIDATE_IMPOSSIBLE" for r in rows),
        "l1_candidate_impossible_share_of_nonprofit_pct":100*sum(r["root_cause"]=="L1_CANDIDATE_IMPOSSIBLE" for r in rows)/len(losses) if losses else None,
        "l2_same_bet_routing_miss_races":sum(r["root_cause"]=="L2_ROUTING_MISS_SAME_BET" for r in rows),
        "l2_cross_bet_routing_miss_races":sum(r["root_cause"]=="L2_ROUTING_MISS_CROSS_BET" for r in rows),
        "l2_role_template_miss_races":sum(r["root_cause"]=="L2_ROLE_TEMPLATE_MISS" for r in rows),
        "l2_hit_but_price_points_loss_races":sum(r["root_cause"]=="L2_HIT_BUT_PRICE_POINTS_LOSS" for r in rows),
        "routing_recoverable_races":len(recoverable),
        "routing_recoverable_share_of_nonprofit_pct":100*len(recoverable)/len(losses) if losses else None,
        "routing_hindsight_gain_yen":sum(float(r["best_all_delta_vs_selected_yen"]) for r in recoverable),
        "data_unavailable_races":sum(r["root_cause"]=="DATA_UNAVAILABLE" for r in rows),
        "payout_incomplete_races":sum(r["root_cause"]=="PAYOUT_INCOMPLETE" for r in rows),
    }

def period_sets(rows):
    out=[]
    for y in TEST_YEARS:
        yy=[r for r in rows if int(r["year"])==y]
        out.append((str(y),yy))
    out.append(("2024-2025",[r for r in rows if int(r["year"]) in (2024,2025)]))
    out.append(("2023-2025",list(rows)))
    return out

def main():
    a=parse_args()
    c=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if c.get("contract")!="L2_V1_LOSS_DECOMPOSITION_V2": raise SystemExit("wrong contract")
    if c["scope"]["battlefield"]!="PASS_SEVEN_ONLY": raise SystemExit("battlefield drift")
    if c["scope"]["analysis_years"]!=[2023,2024,2025]: raise SystemExit("analysis years drift")
    if c["scope"]["locked_years"]!=[2026]: raise SystemExit("2026 lock drift")
    if c["scope"]["race_filtering"] is not False: raise SystemExit("race filtering forbidden")
    if c["scope"]["all_normal_races_used"] is not True: raise SystemExit("all-race guard broken")
    if c["scope"]["selected_architecture"]!="DIRECT_TEMPLATE": raise SystemExit("architecture drift")
    if c["hindsight"]["permitted_as_prediction_features"] is not False: raise SystemExit("hindsight leakage contract broken")
    if c["guards"]["reproduce_v1_roi"] is not True: raise SystemExit("V1 reproduction guard missing")
    if c["promotion"] is not False: raise SystemExit("diagnostic must not promote production")

    paths=parse_paths(a.router_year)
    if set(paths)!=set(YEARS): raise SystemExit(f"router years mismatch {sorted(paths)}")

    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={y:load_router(paths[y],y) for y in YEARS}
    features=build_feature_rows(routers,fixed)
    expected={y:len(routers[y])-len(fixed[y]) for y in ANALYSIS_YEARS}
    actual=features.groupby("year")["race_id"].nunique().to_dict()
    if any(actual.get(y)!=expected[y] for y in ANALYSIS_YEARS):
        raise SystemExit(f"normal universe drift actual={actual} expected={expected}")

    _,perf=load_performance(a.dataset_dir)
    data=attach_targets(features,perf)
    cols=feature_columns(data)
    refs=load_reference_metrics(a.v1_metrics)

    predictions={}
    reproduction=[]
    for y in TEST_YEARS:
        train=data[data["year"]<y].copy()
        test=data[data["year"]==y].copy().sort_values(["race_date","race_id"]).reset_index(drop=True)
        pred,_=train_predict(train,test,"target_template",cols,81000+y)
        m=evaluate(test,pred,"DIRECT_TEMPLATE",perf)
        ref=refs.get(y)
        if not ref: raise SystemExit(f"missing V1 reference year={y}")
        if abs(float(m["roi_pct"])-float(ref["roi_pct"]))>0.02:
            raise SystemExit(f"V1 ROI reproduction drift y={y} now={m['roi_pct']} ref={ref['roi_pct']}")
        if int(m["tickets"])!=int(float(ref["tickets"])):
            raise SystemExit(f"V1 ticket reproduction drift y={y} now={m['tickets']} ref={ref['tickets']}")
        reproduction.append({
            "year":y,
            "reproduced_roi_pct":m["roi_pct"],
            "reference_roi_pct":float(ref["roi_pct"]),
            "reproduced_profit_yen":m["profit_yen"],
            "reference_profit_yen":float(ref["profit_yen"]),
            "reproduced_tickets":m["tickets"],
            "reference_tickets":int(float(ref["tickets"])),
        })
        for r,p in zip(test.to_dict("records"),pred):
            predictions[(y,str(r["race_id"]))]={
                "race_date":str(r["race_date"]),
                "predicted_template":str(p),
            }

    date_to_races=defaultdict(list)
    race_owner={}
    for (y,rid),p in predictions.items():
        date=str(p["race_date"])[:10]
        date_to_races[date].append(rid)
        race_owner[rid]=y

    root=Path(a.backfill_root)
    rows=[]
    for di,date in enumerate(sorted(date_to_races),1):
        wanted=set(date_to_races[date])
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in sorted(wanted):
            y=race_owner[rid]
            pack=day.get(rid)
            if pack is None: raise SystemExit(f"missing race pack {rid}")
            router=routers[y][rid]
            seven_order,_=seven_stats(router)
            horse_no=horse_number_map(pack)
            if any(h not in horse_no for h in seven_order):
                raise SystemExit(f"seven horse number missing race={rid}")
            candidate_nums=[horse_no[h] for h in seven_order]

            payouts,_=payout_map(pack)
            wins={bet:winning_combos(payouts,bet) for bet in ("QUINELLA","EXACTA","TRIO","TRIFECTA")}

            pred=predictions[(y,rid)]["predicted_template"]
            selected_template,z,fallback=choose_available(y,rid,pred,perf)
            if not selected_template: selected_template=pred
            selected_bet=TEMPLATE_TO_BET.get(selected_template) or TEMPLATE_TO_BET.get(pred) or ""

            if z is None or float(z["stake"])<=0:
                rows.append({
                    "year":y,"race_id":rid,"race_date":date,
                    "predicted_template":pred,"selected_template":selected_template,
                    "selected_bet_type":selected_bet,
                    "data_availability_fallback":int(bool(fallback)),
                    "selected_tickets":0,"selected_stake_yen":0.0,"selected_return_yen":0.0,"selected_profit_yen":0.0,
                    "selected_hit":0,"selected_profitable":0,
                    "candidate_feasible_for_selected_bet":None,
                    "same_bet_any_template_hit":0,"same_bet_any_profitable_template":0,
                    "all_bets_any_template_hit":0,"all_bets_any_profitable_template":0,
                    "best_same_bet_template":"","best_same_bet_profit_yen":None,
                    "best_all_template":"","best_all_bet_type":"","best_all_profit_yen":None,
                    "best_all_delta_vs_selected_yen":None,
                    "root_cause":"DATA_UNAVAILABLE",
                })
                continue

            selected_stake=float(z["stake"])
            selected_ret=float(z["ret"])
            selected_profit=selected_ret-selected_stake
            selected_hit=int(selected_ret>0)

            template_items=[]
            for t in TEMPLATES:
                q=perf.get((y,rid,t))
                if not q or float(q["stake"])<=0: continue
                template_items.append({
                    "template":t,
                    "bet_type":TEMPLATE_TO_BET[t],
                    "tickets":int(q["tickets"]),
                    "stake_yen":float(q["stake"]),
                    "return_yen":float(q["ret"]),
                    "profit_yen":float(q["ret"]-q["stake"]),
                    "hit":int(float(q["ret"])>0),
                })
            same=[x for x in template_items if x["bet_type"]==selected_bet]
            best_same=best_item(same)
            best_all=best_item(template_items)

            same_hit=any(x["hit"] for x in same)
            same_profit=any(x["profit_yen"]>0 for x in same)
            all_hit=any(x["hit"] for x in template_items)
            all_profit=any(x["profit_yen"]>0 for x in template_items)

            selected_wins=wins.get(selected_bet,[])
            payout_ok=bool(selected_wins)
            candidate_feasible=any_fully_covered(selected_wins,candidate_nums) if payout_ok else None

            if selected_profit>0:
                root_cause="V1_PROFITABLE"
            elif not payout_ok:
                root_cause="PAYOUT_INCOMPLETE"
            elif not candidate_feasible:
                root_cause="L1_CANDIDATE_IMPOSSIBLE"
            elif same_profit:
                root_cause="L2_ROUTING_MISS_SAME_BET"
            elif all_profit:
                root_cause="L2_ROUTING_MISS_CROSS_BET"
            elif same_hit:
                root_cause="L2_HIT_BUT_PRICE_POINTS_LOSS"
            else:
                root_cause="L2_ROLE_TEMPLATE_MISS"

            rows.append({
                "year":y,"race_id":rid,"race_date":date,
                "predicted_template":pred,"selected_template":selected_template,
                "selected_bet_type":selected_bet,
                "data_availability_fallback":int(bool(fallback)),
                "selected_tickets":int(z["tickets"]),
                "selected_stake_yen":selected_stake,
                "selected_return_yen":selected_ret,
                "selected_profit_yen":selected_profit,
                "selected_hit":selected_hit,
                "selected_profitable":int(selected_profit>0),
                "candidate_feasible_for_selected_bet":int(bool(candidate_feasible)) if candidate_feasible is not None else None,
                "same_bet_any_template_hit":int(same_hit),
                "same_bet_any_profitable_template":int(same_profit),
                "all_bets_any_template_hit":int(all_hit),
                "all_bets_any_profitable_template":int(all_profit),
                "best_same_bet_template":best_same["template"] if best_same else "",
                "best_same_bet_profit_yen":best_same["profit_yen"] if best_same else None,
                "best_all_template":best_all["template"] if best_all else "",
                "best_all_bet_type":best_all["bet_type"] if best_all else "",
                "best_all_profit_yen":best_all["profit_yen"] if best_all else None,
                "best_all_delta_vs_selected_yen":(best_all["profit_yen"]-selected_profit) if best_all else None,
                "root_cause":root_cause,
            })

        if di%50==0:
            print(f"V1_LOSS_DECOMP_PROGRESS dates={di}/{len(date_to_races)} races={len(rows)}",flush=True)

    expected_total=sum(expected[y] for y in TEST_YEARS)
    if len(rows)!=expected_total:
        raise SystemExit(f"decomposition race count drift actual={len(rows)} expected={expected_total}")

    period_summary=[]
    root_summary=[]
    bet_summary=[]
    transitions=[]

    for label,subset in period_sets(rows):
        period_summary.append(summarize_period(label,subset))
        losses=[r for r in subset if r["root_cause"] not in ("V1_PROFITABLE","DATA_UNAVAILABLE")]
        for cat in ROOT_ORDER:
            rr=[r for r in subset if r["root_cause"]==cat]
            if not rr and cat not in ("V1_PROFITABLE","DATA_UNAVAILABLE"): 
                pass
            selected_profit=sum(float(r["selected_profit_yen"]) for r in rr)
            root_summary.append({
                "period":label,
                "root_cause":cat,
                "races":len(rr),
                "share_of_all_pct":100*len(rr)/len(subset) if subset else None,
                "share_of_nonprofitable_pct":100*len(rr)/len(losses) if losses and cat not in ("V1_PROFITABLE","DATA_UNAVAILABLE") else None,
                "selected_profit_sum_yen":selected_profit,
                "avg_selected_profit_yen":selected_profit/len(rr) if rr else None,
                "hindsight_gain_sum_yen":sum(float(r["best_all_delta_vs_selected_yen"] or 0.0) for r in rr),
            })

        for bet in ("QUINELLA","EXACTA","TRIO","TRIFECTA"):
            bb=[r for r in subset if r["selected_bet_type"]==bet and r["root_cause"]!="DATA_UNAVAILABLE"]
            if not bb: continue
            stake=sum(float(r["selected_stake_yen"]) for r in bb)
            ret=sum(float(r["selected_return_yen"]) for r in bb)
            lossbb=[r for r in bb if r["root_cause"]!="V1_PROFITABLE"]
            bet_summary.append({
                "period":label,"selected_bet_type":bet,"races":len(bb),
                "share_of_routes_pct":100*len(bb)/sum(1 for r in subset if r["root_cause"]!="DATA_UNAVAILABLE"),
                "tickets":sum(int(r["selected_tickets"]) for r in bb),
                "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
                "roi_pct":100*ret/stake if stake else None,
                "hit_rate_pct":100*sum(int(r["selected_hit"]) for r in bb)/len(bb),
                "profitable_race_rate_pct":100*sum(r["root_cause"]=="V1_PROFITABLE" for r in bb)/len(bb),
                "candidate_impossible_share_of_nonprofit_pct":100*sum(r["root_cause"]=="L1_CANDIDATE_IMPOSSIBLE" for r in bb)/len(lossbb) if lossbb else None,
                "routing_recoverable_share_of_nonprofit_pct":100*sum(r["root_cause"] in ("L2_ROUTING_MISS_SAME_BET","L2_ROUTING_MISS_CROSS_BET") for r in bb)/len(lossbb) if lossbb else None,
            })

        trans=defaultdict(lambda:{"races":0,"gain":0.0})
        for r in subset:
            if r["root_cause"] not in ("L2_ROUTING_MISS_SAME_BET","L2_ROUTING_MISS_CROSS_BET"): continue
            key=(r["selected_template"],r["best_all_template"],r["root_cause"])
            trans[key]["races"]+=1
            trans[key]["gain"]+=float(r["best_all_delta_vs_selected_yen"] or 0.0)
        for (src,dst,cat),v in sorted(trans.items(),key=lambda kv:(-kv[1]["gain"],-kv[1]["races"],kv[0])):
            transitions.append({
                "period":label,"root_cause":cat,
                "selected_template":src,"best_hindsight_template":dst,
                "races":v["races"],"hindsight_gain_sum_yen":v["gain"],
                "avg_hindsight_gain_yen":v["gain"]/v["races"] if v["races"] else None,
            })

    priority=[]
    recent=[r for r in rows if int(r["year"]) in (2024,2025)]
    recent_losses=[r for r in recent if r["root_cause"] not in ("V1_PROFITABLE","DATA_UNAVAILABLE")]
    for cat in ROOT_ORDER:
        if cat in ("V1_PROFITABLE","DATA_UNAVAILABLE"): continue
        rr=[r for r in recent if r["root_cause"]==cat]
        if not rr: continue
        loss=-sum(min(float(r["selected_profit_yen"]),0.0) for r in rr)
        priority.append({
            "period":"2024-2025","root_cause":cat,"races":len(rr),
            "share_of_nonprofitable_pct":100*len(rr)/len(recent_losses) if recent_losses else None,
            "selected_loss_magnitude_yen":loss,
            "share_of_total_selected_loss_magnitude_pct":None,
            "hindsight_gain_sum_yen":sum(float(r["best_all_delta_vs_selected_yen"] or 0.0) for r in rr),
        })
    total_loss=sum(r["selected_loss_magnitude_yen"] for r in priority)
    for r in priority:
        r["share_of_total_selected_loss_magnitude_pct"]=100*r["selected_loss_magnitude_yen"]/total_loss if total_loss else None
    priority.sort(key=lambda r:(-r["selected_loss_magnitude_yen"],-r["races"],r["root_cause"]))

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"v1-reproduction.csv",reproduction)
    write_csv(out/"period-summary.csv",period_summary)
    write_csv(out/"root-cause-summary.csv",root_summary)
    write_csv(out/"bet-type-summary.csv",bet_summary)
    write_csv(out/"routing-transitions.csv",transitions)
    write_csv(out/"priority-2024-2025.csv",priority)
    with gzip.open(out/"race-decomposition.csv.gz","wt",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    summary={
        "contract":"L2_V1_LOSS_DECOMPOSITION_RESULT_V2",
        "scope":"PASS_SEVEN_ONLY / V1 DIRECT_TEMPLATE",
        "analysis_years":list(TEST_YEARS),
        "normal_races":{str(y):expected[y] for y in TEST_YEARS},
        "v1_reproduction_passed":True,
        "root_categories":list(ROOT_ORDER),
        "candidate_feasibility_is_bet_specific":True,
        "recent_priority_period":"2024-2025",
        "race_filtering":False,
        "all_normal_races_used":True,
        "hindsight_guard":"Best-alternative and gain fields are diagnostic upper bounds only and are never prediction features.",
        "2026_locked":True,
        "production_promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# V1 Loss Decomposition V2\n\n"
        "This diagnostic reproduces the frozen Normal Router V1 DIRECT_TEMPLATE walk-forward outputs and then assigns each race a responsibility layer. "
        "Candidate impossibility is evaluated against the bet actually selected by V1: exact/quinella need the winning top-two horses inside Seven-King; trio/trifecta need the podium trio inside Seven-King. "
        "Only losses that remain feasible upstream are assigned to same-bet routing miss, cross-bet routing miss, role/template miss, or hit-but-price/points loss. "
        "Hindsight alternatives are diagnostic upper bounds only. No race filtering, no production promotion, and 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("L2_V1_LOSS_DECOMPOSITION_V2_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
