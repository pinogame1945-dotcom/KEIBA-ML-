#!/usr/bin/env python3
import argparse
import gzip
import json
from pathlib import Path


def args():
    p=argparse.ArgumentParser(description="Deterministically sample complete races from a yearly snapshot.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-races", type=int, required=True)
    return p.parse_args()


def open_text(path, mode):
    return gzip.open(path, mode, encoding="utf-8") if str(path).endswith(".gz") else open(path, mode, encoding="utf-8")


def main():
    a=args()
    if a.max_races < 1:
        raise ValueError("--max-races must be >= 1")
    selected=[]
    selected_ids=[]
    seen=set()
    with open_text(a.input,"rt") as fh:
        for line in fh:
            if not line.strip():
                continue
            row=json.loads(line)
            race_id=str(row.get("race_id") or "")
            if not race_id:
                continue
            if race_id not in seen:
                if len(selected_ids) >= a.max_races:
                    break
                seen.add(race_id)
                selected_ids.append(race_id)
            selected.append(row)
    out=Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open_text(out,"wt") as fh:
        for row in selected:
            fh.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    print("SNAPSHOT_RACE_SAMPLE_READY")
    print(json.dumps({"input":a.input,"output":str(out),"races":len(selected_ids),"rows":len(selected)},ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
