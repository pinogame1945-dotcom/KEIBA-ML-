#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

FORBIDDEN={
    "actual_is_win","actual_finish_position","target","finish_position",
    "finish_time_ms","last_3f","prize_money","final_win_odds",
    "final_popularity","payout","payouts",
}

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs", required=True, help="Comma-separated L2 output JSONL(.gz)")
    p.add_argument("--arena-summary", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()

def read_rows(path):
    op=gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

def main():
    a=args()
    paths=[Path(x) for x in a.inputs.split(",") if x.strip()]
    if len(paths)<2:
        raise ValueError("need at least two expert outputs")
    expert_rows={}
    all_keys=None
    top1=defaultdict(dict)
    for path in paths:
        rows=read_rows(path)
        if not rows:
            raise ValueError(f"empty: {path}")
        expert=str(rows[0]["expert_id"])
        if expert in expert_rows:
            raise ValueError("duplicate expert_id")
        keys=set()
        for row in rows:
            if row.get("contract")!="L1_TO_L2_OUTPUT_CONTRACT_V1":
                raise ValueError("contract mismatch")
            leaked=sorted(FORBIDDEN.intersection(row))
            if leaked:
                raise ValueError("forbidden fields: "+",".join(leaked))
            share_sum=sum(float(v) for v in (row.get("family_abs_share") or {}).values())
            if (row.get("family_abs_contribution") or {}) and abs(share_sum-1.0)>1e-6:
                raise ValueError("family_abs_share sum mismatch")
            key=(str(row["race_id"]),str(row["horse_id"]))
            keys.add(key)
            if int(row["predicted_rank"])==1:
                top1[str(row["race_id"])][expert]=str(row["horse_id"])
        if all_keys is None:
            all_keys=keys
        elif keys!=all_keys:
            raise ValueError("expert race/horse coverage mismatch")
        expert_rows[expert]=rows

    races=sorted({race for race,_horse in all_keys})
    experts=list(expert_rows)
    agreed=0
    for race in races:
        picks=[top1[race].get(e) for e in experts]
        if len(set(picks))==1:
            agreed+=1

    arena=json.loads(Path(a.arena_summary).read_text(encoding="utf-8"))
    resources={}
    for row in arena.get("results") or []:
        resources[row["name"]]={
            "feature_count":row.get("feature_count"),
            "resource_usage":row.get("resource_usage"),
            "l2_output":row.get("l2_output"),
        }

    summary={
        "contract":"L2_MULTI_EXPERT_PROTOTYPE_V1",
        "experts":len(experts),
        "expert_ids":experts,
        "races":len(races),
        "rows_per_expert":len(all_keys),
        "top1_full_agreement_races":agreed,
        "top1_full_agreement_rate":agreed/len(races) if races else None,
        "coverage_identical":True,
        "forbidden_field_check":"PASS",
        "resources":resources,
    }
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_MULTI_EXPERT_PROTOTYPE_OK")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
