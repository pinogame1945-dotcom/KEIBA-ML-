#!/usr/bin/env python3
import argparse
import gzip
import json
from pathlib import Path

YEARS=(2022,2023,2024,2025)
EXPECTED_PER_YEAR=3456


def parse_args():
    p=argparse.ArgumentParser(description="Extract all race dates needed by L2 Bet Kings Arena V1.")
    p.add_argument("--router-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--dates-out",required=True)
    p.add_argument("--race-dates-out",required=True)
    return p.parse_args()


def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")


def parse_year_paths(items):
    out={}
    for spec in items:
        year,path=spec.split(":",1)
        out[int(year)]=path
    return out


def main():
    a=parse_args()
    paths=parse_year_paths(a.router_year)
    if set(paths)!=set(YEARS):
        raise SystemExit(f"router years mismatch: {sorted(paths)}")

    dates=set()
    race_rows=[]
    seen=set()
    for year in YEARS:
        count=0
        with open_text(paths[year]) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row=json.loads(line)
                rid=str(row.get("race_id") or "")
                date=str(row.get("race_date") or "")[:10]
                if not rid or len(date)!=10:
                    raise SystemExit(f"router missing race_id/date year={year}")
                if rid in seen:
                    raise SystemExit(f"duplicate race_id across years: {rid}")
                seen.add(rid)
                dates.add(date)
                race_rows.append((year,rid,date))
                count+=1
        if count!=EXPECTED_PER_YEAR:
            raise SystemExit(f"router count regression year={year}: {count} != {EXPECTED_PER_YEAR}")

    dates_path=Path(a.dates_out)
    dates_path.parent.mkdir(parents=True,exist_ok=True)
    dates_path.write_text("\n".join(sorted(dates))+"\n",encoding="utf-8")

    rows_path=Path(a.race_dates_out)
    rows_path.parent.mkdir(parents=True,exist_ok=True)
    rows_path.write_text(
        "year,race_id,race_date\n"+
        "".join(f"{y},{rid},{date}\n" for y,rid,date in sorted(race_rows)),
        encoding="utf-8",
    )

    print("L2_BET_KINGS_DATES_READY")
    print(json.dumps({
        "races":len(race_rows),
        "unique_dates":len(dates),
        "dates_out":str(dates_path),
        "race_dates_out":str(rows_path),
    },ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
