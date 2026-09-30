#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import (
    BET_GROUP, ARITY, canonical_numbers, final_odds_tuple, decode_odds, finite
)

YEARS=(2022,2023,2024,2025)

def parse_args():
    p=argparse.ArgumentParser(description="Audit shared odds decoder across all supported bet types.")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def read_jsonl_gz(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def parse_key(bet_type,key):
    n=ARITY[bet_type]
    key=str(key)
    if n==1:
        if not re.fullmatch(r"\d{1,2}",key):
            return None
        nums=[int(key)]
    else:
        if not re.fullmatch(r"\d{%d}"%(2*n),key):
            return None
        nums=[int(key[i*2:i*2+2]) for i in range(n)]
    if any(x<=0 for x in nums) or (n>1 and len(set(nums))!=n):
        return None
    return canonical_numbers(bet_type,nums)

def old_float(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None

def main():
    a=parse_args()
    root=Path(a.backfill_root)/"data"/"odds"/"daily"
    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)

    stats=defaultdict(lambda:defaultdict(int))
    examples=defaultdict(list)
    files_seen=defaultdict(int)
    records_seen=defaultdict(int)

    for year in YEARS:
        files=sorted(root.glob(f"{year}-*.jsonl.gz"))
        if not files:
            raise SystemExit(f"no odds files for {year}")
        files_seen[year]=len(files)
        for path in files:
            for rec in read_jsonl_gz(path):
                records_seen[year]+=1
                decoded=decode_odds(rec)
                root_odds=rec.get("odds") or {}
                for bet_type,group in BET_GROUP.items():
                    data=root_odds.get(group)
                    if not isinstance(data,dict):
                        continue

                    raw_valid=set()
                    old_valid=set()
                    comma_valid=set()
                    for key,raw in data.items():
                        nums=parse_key(bet_type,key)
                        if nums is None:
                            continue
                        tup=final_odds_tuple(raw)
                        if not tup:
                            continue
                        raw_price=tup[0]
                        price=finite(raw_price)
                        if price is None or price<=0:
                            continue
                        raw_valid.add(nums)

                        op=old_float(raw_price)
                        if op is not None and op>0:
                            old_valid.add(nums)

                        if isinstance(raw_price,str) and "," in raw_price:
                            comma_valid.add(nums)
                            if len(examples[(year,bet_type)])<5:
                                examples[(year,bet_type)].append(raw_price)

                    decoded_keys={nums for (bt,nums),price in decoded.items() if bt==bet_type and price>0}
                    s=stats[(year,bet_type)]
                    s["raw_valid_unique"]+=len(raw_valid)
                    s["old_decoder_unique"]+=len(old_valid)
                    s["comma_price_unique"]+=len(comma_valid)
                    s["new_decoder_unique"]+=len(decoded_keys)
                    s["new_missing_unique"]+=len(raw_valid-decoded_keys)
                    s["new_extra_unique"]+=len(decoded_keys-raw_valid)
                    s["old_missing_unique"]+=len(raw_valid-old_valid)

    rows=[]
    failures=[]
    for year in YEARS:
        for bet_type in BET_GROUP:
            s=stats[(year,bet_type)]
            row={
                "year":year,
                "bet_type":bet_type,
                "odds_files":files_seen[year],
                "odds_records":records_seen[year],
                **s,
                "recovered_vs_old":s["new_decoder_unique"]-s["old_decoder_unique"],
                "comma_examples":"|".join(examples[(year,bet_type)]),
            }
            rows.append(row)
            if s["new_missing_unique"]!=0 or s["new_extra_unique"]!=0:
                failures.append(row)

    with open(out/"all-bet-types.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    totals=[]
    for bet_type in BET_GROUP:
        group=[r for r in rows if r["bet_type"]==bet_type]
        totals.append({
            "bet_type":bet_type,
            "raw_valid_unique":sum(r["raw_valid_unique"] for r in group),
            "old_decoder_unique":sum(r["old_decoder_unique"] for r in group),
            "comma_price_unique":sum(r["comma_price_unique"] for r in group),
            "new_decoder_unique":sum(r["new_decoder_unique"] for r in group),
            "recovered_vs_old":sum(r["recovered_vs_old"] for r in group),
            "new_missing_unique":sum(r["new_missing_unique"] for r in group),
            "new_extra_unique":sum(r["new_extra_unique"] for r in group),
        })

    with open(out/"totals-by-bet-type.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(totals[0]))
        w.writeheader()
        w.writerows(totals)

    summary={
        "contract":"ODDS_DECODER_ALL_BET_TYPES_AUDIT_V1",
        "years":list(YEARS),
        "bet_types":list(BET_GROUP),
        "normalization":"strip thousands separators before float conversion",
        "assertion":"every valid positive raw final-odds key must appear exactly once in decoded market map",
        "status":"PASS" if not failures else "FAIL",
        "failures":failures,
        "totals":totals,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ODDS_DECODER_ALL_BET_TYPES_AUDIT "+summary["status"])
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
    if failures:
        raise SystemExit("shared odds decoder coverage mismatch")

if __name__=="__main__":
    main()
