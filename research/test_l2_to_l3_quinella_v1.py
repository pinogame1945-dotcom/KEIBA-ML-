#!/usr/bin/env python3
import csv
import gzip
import json
import subprocess
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EMITTER=ROOT/"research"/"emit_l2_to_l3_quinella_v1.py"
CATALOG=ROOT/"contracts"/"l2-quinella-law-catalog-v1.json"

FIELDS=[
    "race_id","race_date","pair_horse_ids","pair_numbers","law_id",
    "market_snapshot_id","market_snapshot_kind","market_observed_at_utc",
    "odds","market_rank","model_rank","rank_upgrade","l17_rank_score",
    "field_size","l2_model_version","market_q_norm","market_aware_score",
    "pair_prob_product","surface","distance_m","venue","race_number"
]

def write_csv(path,rows):
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

def base(law):
    return {
        "race_id":"202509010101",
        "race_date":"2025-01-05",
        "pair_horse_ids":"h002|h001",
        "pair_numbers":"7-3",
        "law_id":law,
        "market_snapshot_id":"final-202509010101",
        "market_snapshot_kind":"HISTORICAL_FINAL",
        "market_observed_at_utc":"",
        "odds":"28.4",
        "market_rank":"14",
        "model_rank":"3",
        "rank_upgrade":"11",
        "l17_rank_score":"9",
        "field_size":"16",
        "l2_model_version":"market-gap-v2",
        "market_q_norm":"0.0123",
        "market_aware_score":"1.25",
        "pair_prob_product":"0.021",
        "surface":"TURF",
        "distance_m":"1600",
        "venue":"TEST",
        "race_number":"1",
    }

with tempfile.TemporaryDirectory() as td:
    td=Path(td)
    inp=td/"in.csv"
    out=td/"out.jsonl.gz"
    write_csv(inp,[base("LAW1"),base("LAW2")])
    subprocess.run([
        "python",str(EMITTER),
        "--input",str(inp),
        "--law-catalog",str(CATALOG),
        "--output",str(out),
        "--generated-at-utc","2026-09-30T05:00:00Z",
    ],check=True,cwd=ROOT)
    with gzip.open(out,"rt",encoding="utf-8") as f:
        rows=[json.loads(x) for x in f if x.strip()]
    assert len(rows)==1,rows
    r=rows[0]
    assert r["contract"]=="L2_TO_L3_QUINELLA_V1"
    assert r["ticket_key"]=="QUINELLA:202509010101:h001:h002"
    assert r["horses"]==[
        {"horse_id":"h001","horse_number":3},
        {"horse_id":"h002","horse_number":7},
    ],r["horses"]
    assert r["selected_by_laws"]==["LAW1","LAW2"],r
    assert r["law_count"]==2
    assert "stake_yen" not in r
    assert "hit" not in r

    disabled=td/"disabled.csv"
    write_csv(disabled,[base("LAW5")])
    p=subprocess.run([
        "python",str(EMITTER),
        "--input",str(disabled),
        "--law-catalog",str(CATALOG),
        "--output",str(td/"bad.jsonl"),
        "--generated-at-utc","2026-09-30T05:00:00Z",
    ],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode!=0,p.stdout
    assert "not enabled for handoff" in (p.stderr+p.stdout),p.stderr

    live=base("LAW1")
    live["market_snapshot_kind"]="LIVE"
    live["market_observed_at_utc"]=""
    live_path=td/"live.csv"
    write_csv(live_path,[live])
    p=subprocess.run([
        "python",str(EMITTER),
        "--input",str(live_path),
        "--law-catalog",str(CATALOG),
        "--output",str(td/"live.jsonl"),
        "--generated-at-utc","2026-09-30T05:00:00Z",
    ],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode!=0
    assert "requires market_observed_at_utc" in (p.stderr+p.stdout)

print("L2_TO_L3_QUINELLA_V1_SMOKE_OK")
