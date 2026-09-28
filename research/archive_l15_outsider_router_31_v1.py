#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

YEARS=(2022,2023,2024,2025)
FEATURE_SETS=("COMPACT","FULL")
POLICIES=("INDIVIDUAL_TOP1","INDIVIDUAL_TOP2","INDIVIDUAL_TOP3","COMBO_K1","COMBO_K2","COMBO_K3","COMBO_K4","COMBO_K5")
FIXED=("FIXED_K1","FIXED_K2","FIXED_K3","FIXED_K4","FIXED_K5")
ALL_BLIND=1366
GATE_ALERTS=1384
GATE_BLIND=277
FIVE_ORACLE=244
ALL11_ORACLE=250

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--run-id",required=True,type=int)
    p.add_argument("--root",default="research-results/l15-outsider-router-31-v1")
    return p.parse_args()

def add_metrics(rows):
    alerts=sum(int(x["alerts"]) for x in rows)
    blind=sum(int(x["gate_caught_blind"]) for x in rows)
    rescued=sum(int(x["rescued"]) for x in rows)
    calls=sum(int(x["outsider_calls"]) for x in rows)
    false_calls=sum(int(x["false_alert_calls"]) for x in rows)
    return {
        "alerts":alerts,
        "gate_caught_blind":blind,
        "rescued":rescued,
        "rescue_rate_within_gate_blind":rescued/blind if blind else None,
        "end_to_end_rescue_rate_all_blind":rescued/ALL_BLIND,
        "outsider_calls":calls,
        "avg_outsiders_per_alert":calls/alerts if alerts else None,
        "rescue_per_100_outsider_calls":100*rescued/calls if calls else 0.0,
        "rescue_per_100_gate_alerts":100*rescued/alerts if alerts else 0.0,
        "false_alert_calls":false_calls,
        "oracle_capture_vs_five_union":rescued/FIVE_ORACLE,
        "oracle_gap_to_five_union":FIVE_ORACLE-rescued,
        "oracle_gap_to_all11":ALL11_ORACLE-rescued,
    }

def main():
    a=parse_args()
    run_dir=Path(a.root)/f"run-{a.run_id}"
    folds=[]
    for y in YEARS:
        p=run_dir/f"y{y}"/"result.json"
        if not p.exists():
            raise SystemExit(f"missing fold: {p}")
        x=json.loads(p.read_text(encoding="utf-8"))
        if x.get("contract")!="L15_OUTSIDER_ROUTER_31_V1_FOLD":
            raise SystemExit(f"bad fold contract {p}")
        folds.append(x)

    if sum(x["alerts"] for x in folds)!=GATE_ALERTS:
        raise SystemExit("gate alert total mismatch")
    if sum(x["gate_caught_blind"] for x in folds)!=GATE_BLIND:
        raise SystemExit("gate blind total mismatch")
    if sum(x["five_outsider_oracle_on_gate_blind"] for x in folds)!=FIVE_ORACLE:
        raise SystemExit("five oracle mismatch")

    fixed={}
    for name in FIXED:
        fixed[name]=add_metrics([x["fixed_baselines"][name]["test"] for x in folds])
        fixed[name]["yearly"]=[
            {
                "year":x["test_year"],
                "combo":x["fixed_baselines"][name]["combo"],
                "rescued":x["fixed_baselines"][name]["test"]["rescued"],
                "gate_caught_blind":x["gate_caught_blind"],
            } for x in folds
        ]

    feature_sets={}
    for fs in FEATURE_SETS:
        pol={}
        for name in POLICIES:
            agg=add_metrics([x["feature_sets"][fs]["policies"][name]["test"] for x in folds])
            if name.startswith("COMBO_K"):
                k=int(name.split("K")[1])
            else:
                k=int(name.replace("INDIVIDUAL_TOP",""))
            baseline=fixed[f"FIXED_K{k}"]
            agg["delta_rescued_vs_fixed"]=agg["rescued"]-baseline["rescued"]
            agg["delta_recall_pp_vs_fixed"]=100*((agg["rescue_rate_within_gate_blind"] or 0)-(baseline["rescue_rate_within_gate_blind"] or 0))
            agg["yearly"]=[
                {
                    "year":x["test_year"],
                    "rescued":x["feature_sets"][fs]["policies"][name]["test"]["rescued"],
                    "gate_caught_blind":x["gate_caught_blind"],
                    "delta_vs_fixed":x["feature_sets"][fs]["policies"][name]["delta_rescued_vs_fixed"],
                } for x in folds
            ]
            pol[name]=agg
        feature_sets[fs]={
            "fold_feature_counts":[{"year":x["test_year"],"count":x["feature_sets"][fs]["feature_count"]} for x in folds],
            "policies":pol,
        }

    # Empirical summary only; no hidden test tuning is used to rerun models.
    candidate_rows=[]
    for fs in FEATURE_SETS:
        for name,m in feature_sets[fs]["policies"].items():
            candidate_rows.append((m["rescued"],-m["outsider_calls"],fs,name,m))
    candidate_rows.sort(reverse=True,key=lambda x:(x[0],x[1],x[2],x[3]))
    top_by_rescue=[
        {
            "feature_set":fs,
            "policy":name,
            "rescued":m["rescued"],
            "calls":m["outsider_calls"],
            "gate_blind_recall":m["rescue_rate_within_gate_blind"],
            "end_to_end":m["end_to_end_rescue_rate_all_blind"],
            "oracle_capture_vs_five_union":m["oracle_capture_vs_five_union"],
        }
        for _,_,fs,name,m in candidate_rows[:10]
    ]

    summary={
        "contract":"L15_OUTSIDER_ROUTER_31_V1_SUMMARY",
        "run_id":a.run_id,
        "folds":folds,
        "aggregate":{
            "fixed_baselines":fixed,
            "feature_sets":feature_sets,
            "top_by_rescue":top_by_rescue,
        },
        "reference":{
            "gate_alerts":GATE_ALERTS,
            "gate_caught_blind":GATE_BLIND,
            "all_test_blind":ALL_BLIND,
            "five_outsider_hindsight_union":FIVE_ORACLE,
            "all11_hindsight_union":ALL11_ORACLE,
            "warning":"oracle figures are hindsight ceilings, not deployable router performance",
        },
        "interpretation_guardrails":[
            "Do not choose a production policy solely by aggregate rescued count; call-count/candidate-inflation cost still needs L2 integration.",
            "COMBO_K models choose only among combinations of the same size K, so probability calibration across different K is not required for this comparison.",
            "2026 is sealed and odds are unused.",
        ],
    }
    (run_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    def pct(v):
        return "" if v is None else f"{100*float(v):.2f}%"

    lines=[
        f"# L1.5 Outsider Router 31 V1 — run {a.run_id}",
        "",
        f"- Gate alerts: {GATE_ALERTS:,}",
        f"- True blind inside Gate alerts: {GATE_BLIND}",
        f"- Five-outsider hindsight union: {FIVE_ORACLE} / {GATE_BLIND} = {100*FIVE_ORACLE/GATE_BLIND:.2f}%",
        f"- All-11 hindsight union: {ALL11_ORACLE} / {GATE_BLIND} = {100*ALL11_ORACLE/GATE_BLIND:.2f}%",
        "- 2026 sealed / odds NO",
        "",
        "## Train-only fixed baselines",
        "",
        "| Policy | Rescued | /277 | End-to-end /1366 | Calls | Rescue/100 calls |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in FIXED:
        x=fixed[name]
        lines.append(
            f"| {name} | {x['rescued']} | {pct(x['rescue_rate_within_gate_blind'])} | "
            f"{pct(x['end_to_end_rescue_rate_all_blind'])} | {x['outsider_calls']} | {x['rescue_per_100_outsider_calls']:.2f} |"
        )

    for fs in FEATURE_SETS:
        lines += [
            "",
            f"## {fs} learned Router",
            "",
            "| Policy | Rescued | /277 | Δ vs fixed | End-to-end /1366 | Calls | Rescue/100 calls | Oracle capture |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name in POLICIES:
            x=feature_sets[fs]["policies"][name]
            lines.append(
                f"| {name} | {x['rescued']} | {pct(x['rescue_rate_within_gate_blind'])} | "
                f"{x['delta_rescued_vs_fixed']:+d} | {pct(x['end_to_end_rescue_rate_all_blind'])} | "
                f"{x['outsider_calls']} | {x['rescue_per_100_outsider_calls']:.2f} | "
                f"{pct(x['oracle_capture_vs_five_union'])} |"
            )

    lines += [
        "",
        "Race-level decisions for all 1,384 Gate alerts are stored per year in yYYYY/decisions.csv.",
        "No adaptive K is promoted yet because candidate-inflation/L2 cost is not measured.",
        "",
    ]
    (run_dir/"README.md").write_text("\n".join(lines),encoding="utf-8")

    with open(run_dir/"policy-comparison.csv","w",newline="",encoding="utf-8") as fh:
        fields=["family","feature_set","policy","rescued","gate_blind_recall","end_to_end","calls","rescue_per_100_calls","delta_vs_fixed","oracle_capture"]
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()
        for name,x in fixed.items():
            w.writerow({
                "family":"fixed","feature_set":"","policy":name,"rescued":x["rescued"],
                "gate_blind_recall":x["rescue_rate_within_gate_blind"],
                "end_to_end":x["end_to_end_rescue_rate_all_blind"],"calls":x["outsider_calls"],
                "rescue_per_100_calls":x["rescue_per_100_outsider_calls"],"delta_vs_fixed":0,
                "oracle_capture":x["oracle_capture_vs_five_union"],
            })
        for fs in FEATURE_SETS:
            for name,x in feature_sets[fs]["policies"].items():
                w.writerow({
                    "family":"learned","feature_set":fs,"policy":name,"rescued":x["rescued"],
                    "gate_blind_recall":x["rescue_rate_within_gate_blind"],
                    "end_to_end":x["end_to_end_rescue_rate_all_blind"],"calls":x["outsider_calls"],
                    "rescue_per_100_calls":x["rescue_per_100_outsider_calls"],
                    "delta_vs_fixed":x["delta_rescued_vs_fixed"],
                    "oracle_capture":x["oracle_capture_vs_five_union"],
                })

    print("L15_OUTSIDER_ROUTER_31_SUMMARY_READY")
    print(json.dumps({
        "run_id":a.run_id,
        "top_by_rescue":top_by_rescue[:5],
        "path":str(run_dir),
    },ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
