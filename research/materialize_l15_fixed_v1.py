#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

CONTRACT_NAME="L15_FIXED_V1"
OUTPUT_CONTRACT="L15_FIXED_OUTPUT_V1"
EXPECTED_YEARS={2022,2023,2024,2025}
EXPECTED_RACES_PER_YEAR=3456
EXPECTED_ALERTS_PER_YEAR=346
FORBIDDEN_KEYS={
    "odds","win_odds","final_win_odds","popularity","final_popularity",
    "payout","payouts","return_yen","profit_yen","roi",
}


def parse_args():
    p=argparse.ArgumentParser(description="Materialize frozen L1.5 FIX V1 output from pinned historical inputs.")
    p.add_argument("--year",required=True,type=int)
    p.add_argument("--router-seven",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--contract",default="contracts/l15-fixed-v1.json")
    p.add_argument("--candidate-root",default=None)
    p.add_argument("--alerts-only",action="store_true")
    return p.parse_args()


def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")


def pipe_list(value):
    return [x for x in str(value or "").split("|") if x]


def finite(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None


def read_csv(path):
    with open(path,newline="",encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def read_contract(path):
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("contract")!=CONTRACT_NAME or obj.get("status")!="FROZEN":
        raise ValueError("L15 FIX V1 contract is missing or not frozen")
    if obj.get("immutability",{}).get("mutate_v1") is not False:
        raise ValueError("L15 FIX V1 immutability guard is broken")
    return obj


def load_router(path):
    out={}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            if not rid:
                raise ValueError("router row missing race_id")
            if rid in out:
                raise ValueError(f"duplicate router race_id: {rid}")
            out[rid]=row
    return out


def seven_order(router_row):
    experts=router_row.get("experts") or {}
    if len(experts)!=7:
        raise ValueError(f"expected 7 experts race_id={router_row.get('race_id')} got={len(experts)}")
    stats=defaultdict(lambda:{
        "support":0,
        "borda":0.0,
        "top1_votes":0,
        "best_rank":99,
        "rank_sum":0.0,
    })
    for expert in experts.values():
        ids=[str(x) for x in (expert.get("top6_horse_ids") or []) if str(x)]
        if not ids:
            raise ValueError(f"expert missing Top6 race_id={router_row.get('race_id')}")
        for rank,hid in enumerate(ids,1):
            s=stats[hid]
            s["support"]+=1
            s["borda"]+=float(7-rank)
            s["top1_votes"]+=int(rank==1)
            s["best_rank"]=min(s["best_rank"],rank)
            s["rank_sum"]+=rank
    for s in stats.values():
        s["mean_rank"]=s["rank_sum"]/s["support"]
    ordered=sorted(
        stats,
        key=lambda hid:(
            -stats[hid]["borda"],
            -stats[hid]["support"],
            -stats[hid]["top1_votes"],
            stats[hid]["best_rank"],
            stats[hid]["mean_rank"],
            hid,
        ),
    )
    return ordered


def load_outsider_top6(path):
    by_race=defaultdict(dict)
    for row in read_csv(path):
        rid=str(row["race_id"])
        label=str(row["label_ja"])
        ids=pipe_list(row.get("top6_horse_ids"))
        probs=pipe_list(row.get("top6_probabilities"))
        items=[]
        for idx,hid in enumerate(ids):
            items.append({
                "horse_id":hid,
                "rank":idx+1,
                "probability":finite(probs[idx]) if idx<len(probs) else None,
            })
        by_race[rid][label]=items
    return by_race


def outsider_novel_order(label_rows,selected_labels,seven_union):
    stats=defaultdict(lambda:{
        "support":0,
        "borda":0.0,
        "best_rank":99,
        "probability_max":0.0,
        "probability_sum":0.0,
        "probability_n":0,
    })
    for label in selected_labels:
        rows=label_rows.get(label)
        if rows is None:
            raise ValueError(f"selected outsider missing Top6 rows: {label}")
        for item in rows:
            hid=item["horse_id"]
            if hid in seven_union:
                continue
            rank=int(item["rank"])
            prob=finite(item.get("probability"))
            s=stats[hid]
            s["support"]+=1
            s["borda"]+=float(7-rank)
            s["best_rank"]=min(s["best_rank"],rank)
            if prob is not None:
                s["probability_max"]=max(s["probability_max"],prob)
                s["probability_sum"]+=prob
                s["probability_n"]+=1
    for s in stats.values():
        s["probability_mean"]=s["probability_sum"]/s["probability_n"] if s["probability_n"] else 0.0
    ordered=sorted(
        stats,
        key=lambda hid:(
            -stats[hid]["probability_max"],
            -stats[hid]["borda"],
            -stats[hid]["support"],
            stats[hid]["best_rank"],
            hid,
        ),
    )
    return ordered


def assert_no_forbidden(obj,path="root"):
    if isinstance(obj,dict):
        for k,v in obj.items():
            key=str(k).lower()
            if key in FORBIDDEN_KEYS or any(token in key for token in ("payout","popularity")):
                raise ValueError(f"forbidden L1.5 field {k} at {path}")
            assert_no_forbidden(v,path+"."+str(k))
    elif isinstance(obj,list):
        for i,v in enumerate(obj):
            assert_no_forbidden(v,path+f"[{i}]")


def main():
    a=parse_args()
    contract=read_contract(a.contract)
    if a.year not in EXPECTED_YEARS:
        raise SystemExit(f"L15 FIX V1 historical scope is 2022-2025; requested {a.year}")
    if a.year in set(contract.get("historical_scope",{}).get("locked_years") or []):
        raise SystemExit(f"year is locked by contract: {a.year}")

    root=Path(a.candidate_root or contract["frozen_sources"]["candidate_root"])
    ydir=root/f"y{a.year}"
    decisions_path=ydir/"router-decisions.csv"
    outsider_path=ydir/"outsider-top6.csv"
    seven_path=ydir/"seven-union.csv"
    for p in (decisions_path,outsider_path,seven_path):
        if not p.exists():
            raise SystemExit(f"fixed source missing: {p}")

    decisions={str(r["race_id"]):r for r in read_csv(decisions_path)}
    declared_union={
        str(r["race_id"]):set(pipe_list(r["seven_union_horse_ids"]))
        for r in read_csv(seven_path)
    }
    outsiders=load_outsider_top6(outsider_path)
    router=load_router(a.router_seven)

    if len(router)!=EXPECTED_RACES_PER_YEAR:
        raise SystemExit(f"router race count regression year={a.year}: {len(router)} != {EXPECTED_RACES_PER_YEAR}")
    if len(decisions)!=EXPECTED_ALERTS_PER_YEAR:
        raise SystemExit(f"Gate alert count regression year={a.year}: {len(decisions)} != {EXPECTED_ALERTS_PER_YEAR}")
    if set(decisions)!=set(declared_union):
        raise SystemExit("decision/seven-union alert coverage mismatch")
    if not set(decisions).issubset(router):
        raise SystemExit("Gate alert contains race missing from seven-king router")

    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    writer=gzip.open if str(out).endswith(".gz") else open
    rows_written=0
    alert_rows=0
    novel_total=0

    with writer(out,"wt",encoding="utf-8") as fh:
        for rid in sorted(router):
            is_alert=rid in decisions
            if a.alerts_only and not is_alert:
                continue
            base_order=seven_order(router[rid])
            seven_union=set(base_order)
            if is_alert and seven_union!=declared_union[rid]:
                raise ValueError(
                    f"seven union regression race_id={rid}: router={len(seven_union)} "
                    f"fixed={len(declared_union[rid])}"
                )

            selected=[]
            novel=[]
            gate_score=None
            action="PASS_SEVEN_ONLY"
            if is_alert:
                decision=decisions[rid]
                selected=pipe_list(decision.get("FULL__COMBO_K2"))
                if len(selected)!=2:
                    raise ValueError(f"FULL K2 must select exactly 2 outsiders race_id={rid}: {selected}")
                novel=outsider_novel_order(outsiders.get(rid) or {},selected,seven_union)
                gate_score=finite(decision.get("gate_score"))
                action="INTERVENE_FULL_K2"
                alert_rows+=1
                novel_total+=len(novel)

            record={
                "contract":OUTPUT_CONTRACT,
                "l15_version":"1.0.0",
                "year":a.year,
                "race_id":rid,
                "gate_alert":is_alert,
                "gate_score":gate_score,
                "gate_action":action,
                "seven_consensus_order":base_order,
                "seven_anchor_horse_ids":base_order[:2],
                "seven_union_horse_ids":sorted(seven_union),
                "selected_outsiders":selected,
                "novel_horse_ids":novel,
                "candidate_horse_ids":base_order+novel,
            }
            assert_no_forbidden(record)
            fh.write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n")
            rows_written+=1

    expected_rows=EXPECTED_ALERTS_PER_YEAR if a.alerts_only else EXPECTED_RACES_PER_YEAR
    if rows_written!=expected_rows:
        raise SystemExit(f"output row count regression: {rows_written} != {expected_rows}")
    if alert_rows!=EXPECTED_ALERTS_PER_YEAR:
        raise SystemExit(f"alert output regression: {alert_rows} != {EXPECTED_ALERTS_PER_YEAR}")

    print("L15_FIXED_V1_MATERIALIZED")
    print(json.dumps({
        "year":a.year,
        "rows":rows_written,
        "alerts":alert_rows,
        "novel_horses":novel_total,
        "alerts_only":a.alerts_only,
        "output":str(out),
    },ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
