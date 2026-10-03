#!/usr/bin/env python3
import argparse,csv,gzip,itertools,json,math,time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import load_day,payout_map
from run_l2_core_v1 import BET_TYPES,CalAgg,clip01,load_dataset,probability_invariant,ticket_probability,write_csv
from run_l2_race_distribution_mixture_v2 import (
    FOLDS,COLLAPSE_FACTORS,OUTSIDER_BOOSTS,
    load_outsider_ballots,attach_outsider,build_race_frame,
    fit_gate_pair,gate_predict,gate_metric_rows,
    scenario_probabilities,actual_top3_nums,actual_winner_num,make_race_lookup
)

MARGINS=(0.05,0.10,0.15,0.20)

def parse_args():
    p=argparse.ArgumentParser(description="Sparse world router V3.")
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--ballots-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_year_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1); out[int(y)]=p
    if set(out)!={2022,2023,2024,2025}:
        raise SystemExit(f"ballot years mismatch: {sorted(out)}")
    return out

def route_world(qc,qo,base_c,base_o,margin_c,margin_o):
    ac=float(qc)>=min(float(base_c)+float(margin_c),0.999999)
    ao=float(qo)>=min(float(base_o)+float(margin_o),0.999999)
    if ac and ao: return "BOTH"
    if ac: return "COLLAPSE"
    if ao: return "OUTSIDER"
    return "NORMAL"

def router_parameter_loss(cal_races,cal_df,qc,qo,base_c,base_o,margin_c,margin_o,cf,ob):
    lookup=make_race_lookup(cal_df,set(cal_races["year"].astype(int)))
    losses=[]; routed=0
    for r,qcv,qov in zip(cal_races.itertuples(index=False),qc,qo):
        key=(int(r.year),str(r.race_id))
        sub=lookup.get(key)
        if sub is None: continue
        actual=actual_top3_nums(sub)
        if actual is None: continue
        worlds=scenario_probabilities(sub,cf,ob)
        route=route_world(qcv,qov,base_c,base_o,margin_c,margin_o)
        routed+=int(route!="NORMAL")
        p=ticket_probability("TRIO",actual,worlds[route])
        losses.append(-math.log(clip01(p)))
    return (float(np.mean(losses)) if losses else float("inf"),len(losses),routed)

def select_router(cal_races,cal_df,qc,qo,base_c,base_o):
    rows=[]
    for mc in MARGINS:
        for mo in MARGINS:
            for cf in COLLAPSE_FACTORS:
                for ob in OUTSIDER_BOOSTS:
                    loss,n,routed=router_parameter_loss(cal_races,cal_df,qc,qo,base_c,base_o,mc,mo,cf,ob)
                    rows.append({
                        "collapse_margin":mc,"outsider_margin":mo,
                        "collapse_factor":cf,"outsider_max_boost":ob,
                        "calibration_trio_nll":loss,"races":n,
                        "routed_races":routed,
                        "routed_rate_pct":100.0*routed/n if n else 0.0,
                    })
    rows.sort(key=lambda x:(
        x["calibration_trio_nll"],
        x["routed_races"],
        abs(1.0-x["collapse_factor"]),
        x["outsider_max_boost"],
        x["collapse_margin"],x["outsider_margin"],
    ))
    return rows[0],rows

def enumerate_tickets(bet,nums):
    if bet=="WIN": return ((n,) for n in nums)
    if bet=="QUINELLA": return itertools.combinations(nums,2)
    if bet=="EXACTA": return itertools.permutations(nums,2)
    if bet=="TRIO": return itertools.combinations(nums,3)
    if bet=="TRIFECTA": return itertools.permutations(nums,3)
    raise ValueError(bet)

def evaluate_fold(test_year,test_df,test_races,qc,qo,base_c,base_o,selected,backfill_root):
    lookup=make_race_lookup(test_df,{test_year})
    race_models={}; race_date={}; per_race=[]; route_counts=defaultdict(int)
    max_inv=0.0
    mc=selected["collapse_margin"]; mo=selected["outsider_margin"]
    cf=selected["collapse_factor"]; ob=selected["outsider_max_boost"]

    for ri,(r,qcv,qov) in enumerate(zip(test_races.itertuples(index=False),qc,qo)):
        rid=str(r.race_id); sub=lookup[(test_year,rid)]
        worlds=scenario_probabilities(sub,cf,ob)
        route=route_world(qcv,qov,base_c,base_o,mc,mo)
        route_counts[route]+=1
        pbase=worlds["NORMAL"]; prouter=worlds[route]
        max_inv=max(max_inv,abs(sum(pbase.values())-1.0),abs(sum(prouter.values())-1.0))
        if ri<3:
            max_inv=max(max_inv,probability_invariant(pbase),probability_invariant(prouter))
        aw=actual_winner_num(sub); at=actual_top3_nums(sub)
        wn_base=wn_router=tn_base=tn_router=None
        if aw is not None:
            wn_base=-math.log(clip01(pbase[aw]))
            wn_router=-math.log(clip01(prouter[aw]))
        if at is not None:
            tn_base=-math.log(clip01(ticket_probability("TRIO",at,pbase)))
            tn_router=-math.log(clip01(ticket_probability("TRIO",at,prouter)))
        per_race.append({
            "test_year":test_year,"race_id":rid,"race_date":str(r.race_date)[:10],
            "q_collapse":float(qcv),"q_outsider":float(qov),"route":route,
            "collapse_label":int(r.collapse_label),"outsider_label":int(r.outsider_label),
            "winner_nll_base":wn_base,"winner_nll_router":wn_router,
            "trio_nll_base":tn_base,"trio_nll_router":tn_router,
        })
        race_models[rid]=(pbase,prouter)
        race_date[rid]=str(r.race_date)[:10]

    cal={(model,bet):CalAgg() for model in ("BASE","ROUTER") for bet in BET_TYPES}
    root=Path(backfill_root); bydate=defaultdict(list)
    for rid,d in race_date.items(): bydate[d].append(rid)

    for di,date in enumerate(sorted(bydate),1):
        wanted=set(bydate[date])
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in wanted:
            pack=day.get(rid)
            if pack is None: raise SystemExit(f"missing result pack year={test_year} race={rid}")
            payouts,present=payout_map(pack)
            pbase,prouter=race_models[rid]
            nums=tuple(sorted(pbase))
            for bet in BET_TYPES:
                if bet not in present: continue
                for ticket in enumerate_tickets(bet,nums):
                    ticket=tuple(ticket)
                    y=1 if float(payouts.get((bet,ticket),0.0))>0 else 0
                    cal[("BASE",bet)].add(ticket_probability(bet,ticket,pbase),y)
                    cal[("ROUTER",bet)].add(ticket_probability(bet,ticket,prouter),y)
        if di%25==0:
            print(f"WORLD_ROUTER_TICKET_PROGRESS year={test_year} dates={di}/{len(bydate)}",flush=True)

    ticket_rows=[]
    for model in ("BASE","ROUTER"):
        for bet in BET_TYPES:
            ticket_rows.append({"test_year":test_year,"model":model,"bet_type":bet,**cal[(model,bet)].result()})

    dr=pd.DataFrame(per_race)
    win=dr.dropna(subset=["winner_nll_base","winner_nll_router"])
    trio=dr.dropna(subset=["trio_nll_base","trio_nll_router"])
    headline={
        "test_year":test_year,
        "winner_races":len(win),
        "winner_log_loss_base":float(win["winner_nll_base"].mean()),
        "winner_log_loss_router":float(win["winner_nll_router"].mean()),
        "winner_delta_router_minus_base":float((win["winner_nll_router"]-win["winner_nll_base"]).mean()),
        "trio_races":len(trio),
        "actual_trio_nll_base":float(trio["trio_nll_base"].mean()),
        "actual_trio_nll_router":float(trio["trio_nll_router"].mean()),
        "actual_trio_nll_delta_router_minus_base":float((trio["trio_nll_router"]-trio["trio_nll_base"]).mean()),
        "routed_races":sum(v for k,v in route_counts.items() if k!="NORMAL"),
        "routed_rate_pct":100.0*sum(v for k,v in route_counts.items() if k!="NORMAL")/len(test_races),
        "route_normal":route_counts["NORMAL"],"route_collapse":route_counts["COLLAPSE"],
        "route_outsider":route_counts["OUTSIDER"],"route_both":route_counts["BOTH"],
        "probability_invariant_max_error":max_inv,
    }

    diagnostics=[]
    for route in ("NORMAL","COLLAPSE","OUTSIDER","BOTH"):
        q=dr[dr["route"]==route].dropna(subset=["trio_nll_base","trio_nll_router"])
        diagnostics.append({
            "test_year":test_year,"slice_type":"ROUTE","slice":route,"races":len(q),
            "collapse_actual_rate_pct":100.0*q["collapse_label"].mean() if len(q) else None,
            "outsider_actual_rate_pct":100.0*q["outsider_label"].mean() if len(q) else None,
            "trio_nll_base":float(q["trio_nll_base"].mean()) if len(q) else None,
            "trio_nll_router":float(q["trio_nll_router"].mean()) if len(q) else None,
            "delta_router_minus_base":float((q["trio_nll_router"]-q["trio_nll_base"]).mean()) if len(q) else None,
        })
    for name,mask in (
        ("KING1_COLLAPSE_TRUE",dr["collapse_label"]==1),
        ("OUTSIDER_RISE_TRUE",dr["outsider_label"]==1),
        ("BOTH_TRUE",(dr["collapse_label"]==1)&(dr["outsider_label"]==1)),
        ("NEITHER_TRUE",(dr["collapse_label"]==0)&(dr["outsider_label"]==0)),
    ):
        q=dr[mask].dropna(subset=["trio_nll_base","trio_nll_router"])
        diagnostics.append({
            "test_year":test_year,"slice_type":"ACTUAL","slice":name,"races":len(q),
            "collapse_actual_rate_pct":100.0*q["collapse_label"].mean() if len(q) else None,
            "outsider_actual_rate_pct":100.0*q["outsider_label"].mean() if len(q) else None,
            "trio_nll_base":float(q["trio_nll_base"].mean()) if len(q) else None,
            "trio_nll_router":float(q["trio_nll_router"].mean()) if len(q) else None,
            "delta_router_minus_base":float((q["trio_nll_router"]-q["trio_nll_base"]).mean()) if len(q) else None,
        })
    return headline,ticket_rows,diagnostics,per_race

def main():
    a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L2_WORLD_ROUTER_V3": raise SystemExit("wrong contract")
    if contract["cost_policy"]["github_standard_cpu_only"] is not True or contract["cost_policy"]["gpu"] is not False:
        raise SystemExit("cost guard drift")
    if contract["data_policy"]["locked_years"]!=[2026]: raise SystemExit("2026 lock drift")

    manifest,df=load_dataset(a.dataset_dir)
    df["year"]=df["year"].astype(int); df["race_id"]=df["race_id"].astype(str)
    for c in manifest["feature_columns"]:
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
    stats,candidates=load_outsider_ballots(parse_year_paths(a.ballots_year))
    df=attach_outsider(df,stats,candidates)
    races=build_race_frame(df)
    if 2026 in set(races["year"]): raise SystemExit("2026 sealed")

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    fold_rows=[]; gate_rows=[]; grid_rows=[]; ticket_rows=[]; diagnostic_rows=[]; per_race_all=[]

    for fi,(test_year,train_years) in enumerate(FOLDS,1):
        train_r=races[races["year"].isin(train_years)].dropna(subset=["collapse_label","outsider_label"]).copy()
        test_r=races[races["year"]==test_year].dropna(subset=["collapse_label","outsider_label"]).copy().reset_index(drop=True)
        train_df=df[df["year"].isin(train_years)].copy(); test_df=df[df["year"]==test_year].copy()

        mc,tc,cal_c,qc_cal,cal_loss_c=fit_gate_pair(train_r,"collapse_label",20265000+fi)
        mo,to,cal_o,qo_cal,cal_loss_o=fit_gate_pair(train_r,"outsider_label",20266000+fi)
        if list(cal_c["race_id"])!=list(cal_o["race_id"]): raise SystemExit("calibration race drift")
        cal_r=cal_c.reset_index(drop=True)
        cal_df=train_df[train_df["race_id"].isin(set(cal_r["race_id"].astype(str)))].copy()
        base_c=float(train_r["collapse_label"].mean()); base_o=float(train_r["outsider_label"].mean())

        selected,grid=select_router(cal_r,cal_df,qc_cal,qo_cal,base_c,base_o)
        for row in grid:
            grid_rows.append({"test_year":test_year,**row,"selected":int(all(row[k]==selected[k] for k in (
                "collapse_margin","outsider_margin","collapse_factor","outsider_max_boost"
            )))})
        qc=gate_predict(mc,tc,test_r); qo=gate_predict(mo,to,test_r)
        gate_rows.extend(gate_metric_rows(test_year,test_r,qc,qo))

        headline,tickets,diagnostics,per_race=evaluate_fold(
            test_year,test_df,test_r,qc,qo,base_c,base_o,selected,a.backfill_root
        )
        fold_rows.append({
            **headline,"train_years":"|".join(map(str,train_years)),
            "train_collapse_rate_pct":100.0*base_c,"train_outsider_rate_pct":100.0*base_o,
            "collapse_gate_temperature":tc,"outsider_gate_temperature":to,
            "collapse_gate_internal_cal_logloss":cal_loss_c,
            "outsider_gate_internal_cal_logloss":cal_loss_o,
            "selected_collapse_margin":selected["collapse_margin"],
            "selected_outsider_margin":selected["outsider_margin"],
            "selected_collapse_factor":selected["collapse_factor"],
            "selected_outsider_max_boost":selected["outsider_max_boost"],
            "selected_internal_trio_nll":selected["calibration_trio_nll"],
            "selected_internal_routed_rate_pct":selected["routed_rate_pct"],
        })
        ticket_rows.extend(tickets); diagnostic_rows.extend(diagnostics); per_race_all.extend(per_race)
        print("L2_WORLD_ROUTER_FOLD_DONE "+json.dumps({
            "test_year":test_year,
            "collapse_margin":selected["collapse_margin"],"outsider_margin":selected["outsider_margin"],
            "collapse_factor":selected["collapse_factor"],"outsider_boost":selected["outsider_max_boost"],
            "routed_rate_pct":headline["routed_rate_pct"],
            "trio_delta":headline["actual_trio_nll_delta_router_minus_base"],
        },separators=(",",":")),flush=True)

    write_csv(out/"fold-metrics.csv",fold_rows)
    write_csv(out/"gate-metrics.csv",gate_rows)
    write_csv(out/"router-grid.csv",grid_rows)
    write_csv(out/"ticket-calibration-ab.csv",ticket_rows)
    write_csv(out/"router-diagnostics.csv",diagnostic_rows)
    with gzip.open(out/"race-router.csv.gz","wt",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=list(per_race_all[0].keys()))
        w.writeheader(); w.writerows(per_race_all)

    summary={
        "contract":"L2_WORLD_ROUTER_V3_RESULT",
        "architecture":"SPARSE_WORLD_ROUTER",
        "folds":[x[0] for x in FOLDS],
        "fold_metrics":fold_rows,"gate_metrics":gate_rows,
        "ticket_calibration":ticket_rows,"diagnostics":diagnostic_rows,
        "roi_optimized":False,"market_price_used":False,
        "full_ticket_space_from_runners":True,"2026_locked":True,
        "elapsed_seconds":time.perf_counter()-started,"promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 World Router V3\n\n"
        "NORMAL is preserved unless a calibrated collapse/Outsider gate exceeds its prior event rate by a broad predeclared margin. "
        "No odds, ROI optimization, ticket selection, or staking. 2026 sealed.\n",encoding="utf-8"
    )
    max_inv=max(float(x["probability_invariant_max_error"]) for x in fold_rows)
    if max_inv>1e-8: raise SystemExit(f"router invariant failed max={max_inv}")
    print("===== FOLD METRICS ====="); print((out/"fold-metrics.csv").read_text())
    print("===== TICKET CALIBRATION ====="); print((out/"ticket-calibration-ab.csv").read_text())
    print("===== ROUTER DIAGNOSTICS ====="); print((out/"router-diagnostics.csv").read_text())
    print("L2_WORLD_ROUTER_V3_READY")

if __name__=="__main__":
    main()
