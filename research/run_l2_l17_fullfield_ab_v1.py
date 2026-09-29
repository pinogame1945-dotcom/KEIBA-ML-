#!/usr/bin/env python3
import argparse,csv,gzip,json,math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import (
    YEARS,TEMPLATE_TO_BET,load_fixed_ledgers,load_router,seven_stats,
    load_day,payout_map,horse_number_map
)
from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_normal_loss_anatomy_v1 import winning_combos,any_fully_covered
from run_l2_normal_router_v1 import (
    ANALYSIS_YEARS,TEST_YEARS,TEMPLATES,
    build_feature_rows,load_performance,attach_targets,feature_columns,
    train_predict,evaluate
)

ARMS=("OLD_V1","L17_INFO_ONLY","L17_FULLFIELD")

def parse_args():
    p=argparse.ArgumentParser(description="A/B/C compare old normal L2 vs L1.7 full-field inputs.")
    p.add_argument("--contract",required=True)
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True)
    p.add_argument("--l17-year",action="append",required=True)
    p.add_argument("--old-dataset-dir",required=True)
    p.add_argument("--fullfield-dataset-dir",required=True)
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

def finite(x,default=0.0):
    try:
        v=float(x)
        return v if math.isfinite(v) else default
    except (TypeError,ValueError):
        return default

def stat(vals,kind):
    a=np.asarray([finite(x) for x in vals],dtype=float)
    if not len(a): return 0.0
    if kind=="mean": return float(a.mean())
    if kind=="std": return float(a.std())
    if kind=="min": return float(a.min())
    if kind=="max": return float(a.max())
    if kind=="sum": return float(a.sum())
    raise ValueError(kind)

def build_l17_extra(l17):
    max_field=max(int(r["field_size"]) for rows in l17.values() for r in rows.values())
    expert_names=sorted({
        name
        for rows in l17.values()
        for r in rows.values()
        for h in r.get("horses",[])
        for name in (h.get("experts") or {})
    })
    rows=[]
    for year in ANALYSIS_YEARS:
        for rid,r in l17[year].items():
            horses=sorted(r["horses"],key=lambda x:int(x["consensus_rank"]))
            probs=[finite(h.get("mean_probability")) for h in horses]
            rankstd=[finite(h.get("rank_std")) for h in horses]
            probstd=[finite(h.get("probability_std")) for h in horses]
            top6support=[finite(h.get("top6_support")) for h in horses]
            z={
                "year":year,
                "race_id":str(rid),
                "l17_field_size":int(r["field_size"]),
                "l17_rank_std_mean":stat(rankstd,"mean"),
                "l17_rank_std_max":stat(rankstd,"max"),
                "l17_probability_std_mean":stat(probstd,"mean"),
                "l17_probability_std_max":stat(probstd,"max"),
                "l17_top6_support_mean":stat(top6support,"mean"),
                "l17_top6_support_sum":stat(top6support,"sum"),
                "l17_top1_probability":sum(probs[:1]),
                "l17_top2_probability_sum":sum(probs[:2]),
                "l17_top3_probability_sum":sum(probs[:3]),
                "l17_top6_probability_sum":sum(probs[:6]),
                "l17_tail_after6_probability_sum":sum(probs[6:]),
                "l17_p1_p2_probability_gap":(probs[0]-probs[1]) if len(probs)>1 else 0.0,
                "l17_p2_p3_probability_gap":(probs[1]-probs[2]) if len(probs)>2 else 0.0,
                "l17_p6_p7_probability_gap":(probs[5]-probs[6]) if len(probs)>6 else 0.0,
            }
            for pos in range(1,max_field+1):
                if pos<=len(horses):
                    h=horses[pos-1]
                    prefix=f"l17_p{pos}_"
                    z[prefix+"mean_rank"]=finite(h.get("mean_rank"),99.0)
                    z[prefix+"rank_std"]=finite(h.get("rank_std"))
                    z[prefix+"best_rank"]=finite(h.get("best_rank"),99.0)
                    z[prefix+"worst_rank"]=finite(h.get("worst_rank"),99.0)
                    z[prefix+"top1_votes"]=finite(h.get("top1_votes"))
                    z[prefix+"top3_support"]=finite(h.get("top3_support"))
                    z[prefix+"top6_support"]=finite(h.get("top6_support"))
                    z[prefix+"mean_probability"]=finite(h.get("mean_probability"))
                    z[prefix+"probability_std"]=finite(h.get("probability_std"))
                    views=h.get("experts") or {}
                    for e in expert_names:
                        v=views.get(e) or {}
                        z[prefix+f"expert_{e}_rank"]=finite(v.get("rank"),99.0)
                        z[prefix+f"expert_{e}_probability"]=finite(v.get("probability"))
                else:
                    prefix=f"l17_p{pos}_"
                    z[prefix+"mean_rank"]=99.0
                    z[prefix+"rank_std"]=0.0
                    z[prefix+"best_rank"]=99.0
                    z[prefix+"worst_rank"]=99.0
                    z[prefix+"top1_votes"]=0.0
                    z[prefix+"top3_support"]=0.0
                    z[prefix+"top6_support"]=0.0
                    z[prefix+"mean_probability"]=0.0
                    z[prefix+"probability_std"]=0.0
                    for e in expert_names:
                        z[prefix+f"expert_{e}_rank"]=99.0
                        z[prefix+f"expert_{e}_probability"]=0.0
            rows.append(z)
    return pd.DataFrame(rows),max_field,expert_names

def load_fullfield_performance(root):
    root=Path(root)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract")!="L2_L17_FULLFIELD_DATASET_V1":
        raise SystemExit("wrong fullfield dataset contract")
    if manifest.get("source_l17")!="L17_SEVEN_KING_FULLFIELD_OUTPUT_V1":
        raise SystemExit("wrong fullfield source")
    if manifest.get("locked_years")!=[2026]:
        raise SystemExit("fullfield 2026 lock drift")
    perf=defaultdict(lambda:{"tickets":0,"stake":0.0,"ret":0.0,"hit":0})
    for t in TEMPLATES:
        info=manifest["templates"].get(t)
        if not info: raise SystemExit(f"missing fullfield template={t}")
        with gzip.open(root/info["file"],"rt",encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                r=json.loads(line)
                y=int(r["year"]); rid=str(r["race_id"])
                k=(y,rid,t)
                z=perf[k]
                z["tickets"]+=1
                z["stake"]+=100.0
                z["ret"]+=float(r.get("return_yen_per100") or 0.0)
                z["hit"]=int(z["hit"] or bool(r.get("hit")))
    return manifest,perf

def read_reference(path):
    out={}
    with open(path,newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("architecture")=="DIRECT_TEMPLATE":
                out[int(r["test_year"])]=r
    return out

def coverage_diagnostic(routers,fixed,l17,backfill_root):
    date_to=defaultdict(list); owner={}
    for y in TEST_YEARS:
        alerts=set(fixed[y])
        for rid,row in routers[y].items():
            if rid in alerts: continue
            d=str(row.get("race_date") or "")[:10]
            date_to[d].append(rid); owner[rid]=y
    acc=defaultdict(lambda:{
        "races":0,"winner_old":0,"top2_old":0,"top3_old":0,
        "winner_full":0,"top2_full":0,"top3_full":0
    })
    root=Path(backfill_root)
    for date in sorted(date_to):
        wanted=set(date_to[date])
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in wanted:
            y=owner[rid]; pack=day.get(rid)
            if pack is None: raise SystemExit(f"coverage pack missing race={rid}")
            old_order,_=seven_stats(routers[y][rid])
            full_order=[str(x) for x in l17[y][rid]["consensus_order"]]
            hno=horse_number_map(pack)
            old_nums=[hno[h] for h in old_order]
            full_nums=[hno[h] for h in full_order]
            payouts,_=payout_map(pack)
            wins={
                "WIN":winning_combos(payouts,"WIN"),
                "QUINELLA":winning_combos(payouts,"QUINELLA"),
                "TRIO":winning_combos(payouts,"TRIO"),
            }
            if not all(wins.values()): continue
            a=acc[y]; a["races"]+=1
            a["winner_old"]+=int(any_fully_covered(wins["WIN"],old_nums))
            a["top2_old"]+=int(any_fully_covered(wins["QUINELLA"],old_nums))
            a["top3_old"]+=int(any_fully_covered(wins["TRIO"],old_nums))
            a["winner_full"]+=int(any_fully_covered(wins["WIN"],full_nums))
            a["top2_full"]+=int(any_fully_covered(wins["QUINELLA"],full_nums))
            a["top3_full"]+=int(any_fully_covered(wins["TRIO"],full_nums))
    out=[]
    for y in TEST_YEARS:
        a=acc[y]; n=a["races"]
        out.append({
            "year":y,"races":n,
            "old_winner_in_pool_pct":100*a["winner_old"]/n if n else None,
            "old_top2_all_in_pool_pct":100*a["top2_old"]/n if n else None,
            "old_top3_all_in_pool_pct":100*a["top3_old"]/n if n else None,
            "full_winner_in_pool_pct":100*a["winner_full"]/n if n else None,
            "full_top2_all_in_pool_pct":100*a["top2_full"]/n if n else None,
            "full_top3_all_in_pool_pct":100*a["top3_full"]/n if n else None,
        })
    return out

def main():
    a=parse_args()
    c=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if c.get("contract")!="L2_L17_FULLFIELD_AB_V1": raise SystemExit("wrong contract")
    inv=c["invariants"]
    for k in ("odds_as_model_feature","popularity_as_model_feature","payout_as_model_feature","result_as_model_feature","outsider_used","gate_used_in_l17","race_filtering","skip_class"):
        if inv[k] is not False: raise SystemExit(f"invariant broken: {k}")
    if inv["force_route_every_normal_race"] is not True: raise SystemExit("forced route drift")
    if c["locked_years"]!=[2026]: raise SystemExit("2026 lock drift")

    rp=parse_paths(a.router_year); lp=parse_paths(a.l17_year)
    if set(rp)!=set(YEARS) or set(lp)!=set(YEARS): raise SystemExit("year path mismatch")
    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={y:load_router(rp[y],y) for y in YEARS}
    l17={y:load_l17(lp[y],y) for y in YEARS}

    old_features=build_feature_rows(routers,fixed)
    extra,max_field,experts=build_l17_extra(l17)
    enriched=old_features.merge(extra,on=["year","race_id"],how="left",validate="one_to_one")
    if enriched.isna().all(axis=1).any(): raise SystemExit("L1.7 merge failed")
    expected=old_features.groupby("year")["race_id"].nunique().to_dict()
    actual=enriched.groupby("year")["race_id"].nunique().to_dict()
    if actual!=expected: raise SystemExit(f"feature universe drift old={expected} new={actual}")

    old_manifest,old_perf=load_performance(a.old_dataset_dir)
    full_manifest,full_perf=load_fullfield_performance(a.fullfield_dataset_dir)
    old_data=attach_targets(old_features,old_perf)
    info_data=attach_targets(enriched,old_perf)
    full_data=attach_targets(enriched,full_perf)
    datasets={"OLD_V1":(old_data,old_perf),"L17_INFO_ONLY":(info_data,old_perf),"L17_FULLFIELD":(full_data,full_perf)}

    refs=read_reference(a.v1_metrics)
    metrics=[]; fold_rows=[]; decision_rows=[]
    for arm in ARMS:
        data,perf=datasets[arm]
        cols=feature_columns(data)
        for y in TEST_YEARS:
            train=data[data["year"]<y].copy()
            test=data[data["year"]==y].copy().sort_values(["race_date","race_id"]).reset_index(drop=True)
            pred,meta=train_predict(train,test,"target_template",cols,81000+y)
            m=evaluate(test,pred,arm,perf)
            metrics.append(m)
            fold_rows.append({
                "arm":arm,"test_year":y,
                "train_years":"|".join(map(str,sorted(train["year"].unique()))),
                "train_races":int(train["race_id"].nunique()),
                "test_races":int(test["race_id"].nunique()),
                "feature_count":meta["feature_count"],
                "classes":meta["classes"],
            })
            for r,p in zip(test.to_dict("records"),pred):
                decision_rows.append({
                    "arm":arm,"year":y,"race_id":r["race_id"],"race_date":r["race_date"],
                    "predicted_template":str(p),
                    "target_template":str(r["target_template"]),
                    "target_bet_type":str(r["target_bet_type"]),
                    "target_label_available":int(r["target_label_available"]),
                })
            if arm=="OLD_V1":
                ref=refs.get(y)
                if not ref: raise SystemExit(f"missing reference y={y}")
                if abs(float(m["roi_pct"])-float(ref["roi_pct"]))>0.02:
                    raise SystemExit(f"OLD V1 ROI reproduction drift y={y} now={m['roi_pct']} ref={ref['roi_pct']}")
                if int(m["tickets"])!=int(float(ref["tickets"])):
                    raise SystemExit(f"OLD V1 ticket reproduction drift y={y}")

    compare=[]
    for y in TEST_YEARS:
        base=next(x for x in metrics if x["architecture"]=="OLD_V1" and int(x["test_year"])==y)
        for arm in ("L17_INFO_ONLY","L17_FULLFIELD"):
            z=next(x for x in metrics if x["architecture"]==arm and int(x["test_year"])==y)
            compare.append({
                "year":y,"arm":arm,
                "roi_pct":z["roi_pct"],"old_roi_pct":base["roi_pct"],"roi_delta_pp":z["roi_pct"]-base["roi_pct"],
                "profit_yen":z["profit_yen"],"old_profit_yen":base["profit_yen"],"profit_delta_yen":z["profit_yen"]-base["profit_yen"],
                "hit_rate_pct":z["race_hit_rate_pct"],"old_hit_rate_pct":base["race_hit_rate_pct"],"hit_delta_pp":z["race_hit_rate_pct"]-base["race_hit_rate_pct"],
                "avg_tickets_per_race":z["avg_tickets_per_race"],"old_avg_tickets_per_race":base["avg_tickets_per_race"],
                "max_drawdown_yen":z["max_drawdown_yen"],"old_max_drawdown_yen":base["max_drawdown_yen"],
            })

    coverage=coverage_diagnostic(routers,fixed,l17,a.backfill_root)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"metrics.csv",metrics)
    write_csv(out/"comparison-vs-old.csv",compare)
    write_csv(out/"fold-status.csv",fold_rows)
    write_csv(out/"candidate-coverage.csv",coverage)
    with gzip.open(out/"route-decisions.csv.gz","wt",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(decision_rows[0]))
        w.writeheader(); w.writerows(decision_rows)

    holdout={
        arm:next(x for x in metrics if x["architecture"]==arm and int(x["test_year"])==2025)
        for arm in ARMS
    }
    summary={
        "contract":"L2_L17_FULLFIELD_AB_RESULT_V1",
        "arms":list(ARMS),
        "same_direct_template_hyperparameters":True,
        "old_v1_reproduction_passed":True,
        "l17_max_field_size":max_field,
        "l17_experts":experts,
        "development_years":[2023,2024],
        "holdout_year":2025,
        "holdout_used_for_selection":False,
        "holdout_2025":holdout,
        "candidate_coverage":coverage,
        "old_ticket_dataset_contract":old_manifest.get("contract"),
        "fullfield_ticket_dataset_contract":full_manifest.get("contract"),
        "odds_as_model_feature":False,
        "outsider_used":False,
        "race_filtering":False,
        "skip_class":False,
        "2026_locked":True,
        "production_promotion":False,
        "interpretation_guard":"INFO_ONLY isolates added L1.7 information while keeping the old ticket universe. FULLFIELD adds both L1.7 information and all-runner ticket candidacy. Final odds/payouts are evaluation/target data only, never model features."
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 × L1.7 Full-Field A/B V1\n\n"
        "Three-arm comparison: OLD_V1, L17_INFO_ONLY, and L17_FULLFIELD. "
        "All arms use the same DIRECT_TEMPLATE LightGBM hyperparameters and the same 14 normal template family. "
        "INFO_ONLY adds full-field Seven-King ranks/probabilities but preserves the old ticket universe. "
        "FULLFIELD also applies the same template rules to the complete L1.7 consensus field, allowing ranks below the old candidate pool to enter tickets. "
        "No Outsider, no model-input odds/popularity/payout/results, no race filtering, no SKIP, and 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("L2_L17_FULLFIELD_AB_V1_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
