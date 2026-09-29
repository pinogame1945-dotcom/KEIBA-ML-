#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
from pathlib import Path


def parse_args():
    p=argparse.ArgumentParser(description="Extract Gate alert race dates from router snapshots.")
    p.add_argument("--candidate-root", required=True)
    p.add_argument("--router-year", action="append", required=True, help="YEAR:PATH")
    p.add_argument("--dates-out", required=True)
    p.add_argument("--map-out", required=True)
    return p.parse_args()


def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")


def parse_year_paths(items):
    out={}
    for spec in items:
        y,path=spec.split(":",1)
        out[int(y)]=path
    return out


def main():
    a=parse_args()
    root=Path(a.candidate_root)
    paths=parse_year_paths(a.router_year)
    years=(2022,2023,2024,2025)
    if set(paths)!=set(years):
        raise SystemExit(f"router years mismatch: {sorted(paths)}")

    mapping=[]
    dates=set()
    for year in years:
        dpath=root/f"y{year}"/"router-decisions.csv"
        if not dpath.exists():
            raise SystemExit(f"missing decisions: {dpath}")
        with open(dpath,newline="",encoding="utf-8-sig") as fh:
            wanted={str(r["race_id"]) for r in csv.DictReader(fh)}
        found={}
        with open_text(paths[year]) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row=json.loads(line)
                rid=str(row.get("race_id") or "")
                if rid not in wanted:
                    continue
                date=str(row.get("race_date") or "")[:10]
                if len(date)!=10:
                    raise SystemExit(f"router row missing race_date year={year} race_id={rid}")
                found[rid]=date
                dates.add(date)
        missing=sorted(wanted-set(found))
        if missing:
            raise SystemExit(f"router date coverage missing year={year} count={len(missing)} sample={missing[:10]}")
        mapping.extend({"year":year,"race_id":rid,"race_date":date} for rid,date in sorted(found.items()))

    if len(mapping)!=1384:
        raise SystemExit(f"Gate alert count regression: {len(mapping)} != 1384")

    dates_path=Path(a.dates_out)
    dates_path.parent.mkdir(parents=True,exist_ok=True)
    dates_path.write_text("\n".join(sorted(dates))+"\n",encoding="utf-8")

    map_path=Path(a.map_out)
    map_path.parent.mkdir(parents=True,exist_ok=True)
    with open(map_path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=["year","race_id","race_date"])
        w.writeheader()
        w.writerows(mapping)

    print("L2_GATE_DATES_READY")
    print(json.dumps({
        "gate_alerts":len(mapping),
        "unique_dates":len(dates),
        "dates_out":str(dates_path),
        "map_out":str(map_path),
    },ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
