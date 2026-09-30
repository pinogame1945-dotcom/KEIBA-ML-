#!/usr/bin/env python3
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import build_year_frame
from run_l2_ticket_market_gap_v2 import (
    add_market_features,
    market_feature_columns,
    train_rank_predict,
    add_ranks,
)
from run_l2_market_gap_law_arena_v1 import (
    add_bins,
    candidate_group_tables,
    chosen_from_table,
    metrics,
    key_series,
)
from run_l2_quinella_auto_calibration_v1 import train_base, normalize_race

YEARS=(2021,2022,2023,2024,2025)
CONFIRM_YEAR=2025

# Frozen exactly from the corrected-odds zero-base discovery run.
FROZEN_RULES={
    "RULE5":{
        "gap":"GAP0_2",
        "conditions":(("field_size_band","13-14"),("l17_rank_band","11-20")),
        "expected_2025_tickets":585,
    },
    "RULE7":{
        "gap":"GAP3_4",
        "conditions":(("market_rank_band","<=10"),("l17_rank_band","31-40")),
        "expected_2025_tickets":68,
    },
}

def parse_args():
    p=argparse.ArgumentParser()
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def key_col(df):
    return key_series(df)

def add_machine_signal(frames, cols):
    train=pd.concat([frames[y] for y in (2021,2022,2023,2024)],ignore_index=True)
    test=frames[2025].copy().reset_index(drop=True)
    raw=train_base(train,test,cols,98025)
    test["raw_model_probability"]=raw
    test=normalize_race(test,"raw_model_probability","p_model")
    test["machine_ratio_to_market"]=(
        test["p_model"] / test["market_q_norm"].clip(lower=1e-12)
    )
    # Natural zero-crossing only: the learned model values the ticket above market.
    test["machine_agree"]=test["machine_ratio_to_market"]>1.0
    return test

def add_human_rules(frames, cols):
    train=pd.concat([frames[y] for y in (2022,2023,2024)],ignore_index=True)
    test=frames[2025].copy().reset_index(drop=True)
    score,_=train_rank_predict(train,test,cols,94000+2025)
    test["market_aware_score"]=score
    test=add_bins(add_ranks(test))
    tables=candidate_group_tables(test)

    selected={}
    for rid,spec in FROZEN_RULES.items():
        g=chosen_from_table(tables,spec["gap"],spec["conditions"])
        got=len(g)
        exp=spec["expected_2025_tickets"]
        if got!=exp:
            raise SystemExit(f"{rid} frozen reproduction drift got={got} expected={exp}")
        selected[rid]=g.copy()
    return test,selected

def mrow(g,label):
    m=metrics(g,label)
    m["label"]=label
    m["executed_races"]=int(g["race_id"].nunique()) if len(g) else 0
    m["mean_machine_ratio"]=float(g["machine_ratio_to_market"].mean()) if len(g) else None
    m["median_machine_ratio"]=float(g["machine_ratio_to_market"].median()) if len(g) else None
    m["machine_agree_share_pct"]=100.0*float(g["machine_agree"].mean()) if len(g) else None
    return m

def main():
    a=parse_args()
    paths={y:getattr(a,f"l17_{y}") for y in YEARS}
    l17={y:load_l17(paths[y],y) for y in YEARS}
    frames={y:add_market_features(build_year_frame(y,l17[y],a.backfill_root)) for y in YEARS}
    cols=market_feature_columns(frames[2022])

    machine=add_machine_signal(frames,cols)
    human_frame,rules=add_human_rules(frames,cols)

    machine_cols=[
        "year","race_id","pair_numbers","p_model","market_q_norm",
        "machine_ratio_to_market","machine_agree"
    ]
    z=human_frame.merge(
        machine[machine_cols],
        on=["year","race_id","pair_numbers"],
        how="left",
        validate="one_to_one",
    )
    if z["machine_agree"].isna().any():
        raise SystemExit("machine/human ticket universe merge drift")

    z["_key"]=key_col(z)
    rule_keys={}
    for rid,g in rules.items():
        ks=set(key_col(g).tolist())
        rule_keys[rid]=ks
        z[rid.lower()]=z["_key"].isin(ks)

    z["human_rule"]=z["rule5"]|z["rule7"]
    z["cohort"]=np.select(
        [
            z["human_rule"] & z["machine_agree"],
            z["human_rule"] & ~z["machine_agree"],
            ~z["human_rule"] & z["machine_agree"],
        ],
        ["BOTH","HUMAN_ONLY","MACHINE_ONLY"],
        default="NEITHER",
    )

    rows=[]
    for label,mask in [
        ("HUMAN_ALL",z["human_rule"]),
        ("MACHINE_ALL",z["machine_agree"]),
        ("BOTH",z["cohort"]=="BOTH"),
        ("HUMAN_ONLY",z["cohort"]=="HUMAN_ONLY"),
        ("MACHINE_ONLY",z["cohort"]=="MACHINE_ONLY"),
        ("NEITHER",z["cohort"]=="NEITHER"),
    ]:
        rows.append(mrow(z[mask].copy(),label))

    rule_rows=[]
    for rid in ("RULE5","RULE7"):
        base=z[z[rid.lower()]].copy()
        for suffix,mask in [
            ("ALL",pd.Series(True,index=base.index)),
            ("MACHINE_AGREE",base["machine_agree"]),
            ("MACHINE_DISAGREE",~base["machine_agree"]),
        ]:
            g=base[mask].copy()
            r=mrow(g,f"{rid}_{suffix}")
            r["rule_id"]=rid
            r["machine_relation"]=suffix
            rule_rows.append(r)

    human=z[z["human_rule"]].copy()
    human["rule_membership"]=np.select(
        [
            human["rule5"] & human["rule7"],
            human["rule5"],
            human["rule7"],
        ],
        ["RULE5|RULE7","RULE5","RULE7"],
        default="NONE",
    )
    human_export=human[[
        "race_date","race_id","pair_numbers","odds","hit","return_yen_per100",
        "rule_membership","machine_agree","machine_ratio_to_market",
        "p_model","market_q_norm","market_rank","model_rank","l17_rank_score",
        "rank_upgrade","field_size"
    ]].sort_values(["race_date","race_id","rule_membership","machine_ratio_to_market"],
                  ascending=[True,True,True,False])

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"cohort-metrics-2025.csv",index=False)
    pd.DataFrame(rule_rows).to_csv(out/"rule-machine-split-2025.csv",index=False)
    human_export.to_csv(
        out/"human-rule-tickets-2025.csv.gz",
        index=False,
        compression="gzip",
    )

    rowmap={r["label"]:r for r in rows}
    rmap={r["label"]:r for r in rule_rows}
    summary={
        "contract":"L2_QUINELLA_HUMAN_MACHINE_CROSS_V1",
        "confirmation_year":2025,
        "source_variant":"ODDS_COMMA_CORRECTED",
        "human_rules_frozen_from_zero_base_run":{
            "run_id":36684420309,
            "rules":{
                rid:{
                    "gap":spec["gap"],
                    "conditions":[list(x) for x in spec["conditions"]],
                    "expected_2025_tickets":spec["expected_2025_tickets"],
                }
                for rid,spec in FROZEN_RULES.items()
            },
        },
        "machine_model":{
            "type":"learned-boundary LightGBM classifier",
            "train_years":[2021,2022,2023,2024],
            "signal":"within-race normalized p_model > normalized market probability",
            "manual_machine_threshold_tuning":False,
            "ratio_threshold":1.0,
            "threshold_reason":"natural market-relative zero crossing",
        },
        "primary_2025":{
            k:rowmap[k] for k in ("HUMAN_ALL","MACHINE_ALL","BOTH","HUMAN_ONLY","MACHINE_ONLY")
        },
        "rule_splits_2025":{
            k:rmap[k] for k in (
                "RULE5_ALL","RULE5_MACHINE_AGREE","RULE5_MACHINE_DISAGREE",
                "RULE7_ALL","RULE7_MACHINE_AGREE","RULE7_MACHINE_DISAGREE"
            )
        },
        "2025_used_to_tune_human_rules":False,
        "2025_used_to_tune_machine_threshold":False,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )
    print("L2_QUINELLA_HUMAN_MACHINE_CROSS_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
