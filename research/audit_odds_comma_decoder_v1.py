#!/usr/bin/env python3
import argparse, gzip, json, math, re
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import BET_GROUP, ARITY, UNORDERED, canonical_numbers, final_odds_tuple, decode_odds

YEARS=(2022,2023,2024,2025)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def old_parse(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None

def new_parse(v):
    try:
        if isinstance(v,str):
            v=v.replace(",","").strip()
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None

def rows(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    a=parse_args()
    root=Path(a.backfill_root)/"data"/"odds"/"daily"
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    audit=[]
    examples=defaultdict(list)
    for y in YEARS:
        acc={b:defaultdict(int) for b in BET_GROUP}
        for p in sorted(root.glob(f"{y}-*.jsonl.gz")):
            for rec in rows(p):
                odds=rec.get("odds") or {}
                for bet_type,group in BET_GROUP.items():
                    data=odds.get(group)
                    if not isinstance(data,dict):
                        continue
                    n=ARITY[bet_type]
                    for key,raw in data.items():
                        key=str(key)
                        if n==1:
                            if not re.fullmatch(r"\d{1,2}",key): continue
                            nums=[int(key)]
                        else:
                            if not re.fullmatch(r"\d{%d}"%(2*n),key): continue
                            nums=[int(key[i*2:i*2+2]) for i in range(n)]
                        if any(x<=0 for x in nums) or (n>1 and len(set(nums))!=n):
                            continue
                        tup=final_odds_tuple(raw)
                        if not tup:
                            continue
                        v=tup[0]
                        acc[bet_type]["stored_valid_shape"]+=1
                        if isinstance(v,str) and "," in v:
                            acc[bet_type]["comma_values"]+=1
                            if len(examples[(y,bet_type)])<5:
                                examples[(y,bet_type)].append(v)
                        o=old_parse(v)
                        nn=new_parse(v)
                        if o is not None and o>0:
                            acc[bet_type]["old_decoder_prices"]+=1
                        if nn is not None and nn>0:
                            acc[bet_type]["new_decoder_prices"]+=1
                        if (o is None or o<=0) and (nn is not None and nn>0):
                            acc[bet_type]["recovered_by_comma_fix"]+=1
        for bet_type in BET_GROUP:
            d=acc[bet_type]
            audit.append({
                "year":y,
                "bet_type":bet_type,
                "stored_valid_shape":d["stored_valid_shape"],
                "comma_values":d["comma_values"],
                "old_decoder_prices":d["old_decoder_prices"],
                "new_decoder_prices":d["new_decoder_prices"],
                "recovered_by_comma_fix":d["recovered_by_comma_fix"],
                "recovery_pct_of_new":100.0*d["recovered_by_comma_fix"]/d["new_decoder_prices"] if d["new_decoder_prices"] else 0.0,
                "examples":"|".join(examples[(y,bet_type)])
            })

    import csv
    with open(out/"odds-decoder-audit.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0].keys()))
        w.writeheader();w.writerows(audit)

    summary={"contract":"ODDS_COMMA_DECODER_AUDIT_V1","years":list(YEARS),"rows":audit}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ODDS_COMMA_DECODER_AUDIT_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
