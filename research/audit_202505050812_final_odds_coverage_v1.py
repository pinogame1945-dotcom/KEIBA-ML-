#!/usr/bin/env python3
"""Inspect actual 2025-11-30 archived NETKEIBA_FINAL odds for blind race.

This is an explicitly historical-only coverage investigation, never an
assertion that prices were observable before betting closed.
"""
import argparse
import gzip
import json
from pathlib import Path

RID="202505050812"

def inspect(path):
    selected=[]
    with gzip.open(path,"rt",encoding="utf-8") as file:
        for line in file:
            if not line.strip():continue
            row=json.loads(line)
            if str(row.get("race_id"))==RID:selected.append(row)
    if len(selected)!=1:raise ValueError(f"EXACT_ONE_USER_SELECTED_RACE_REQUIRED_FOUND_{len(selected)}")
    row=selected[0]
    if (row.get("source")!="NETKEIBA_FINAL" or
        row.get("odds_kind")!="FINAL" or row.get("source_status")!="result"):
        raise ValueError("NOT_AUTHENTIC_HISTORICAL_FINAL_ODDS")
    groups=row.get("odds") or {}
    summary={}
    for key,entries in groups.items():
        if not isinstance(entries,dict):raise ValueError("BAD_FINAL_ODDS_GROUP_"+key)
        samples=list(entries.items())[:2]
        summary[key]={"count":len(entries),
                      "sample_keys":[str(k) for k,_ in samples],
                      "sample_row_shapes":[len(v) if isinstance(v,list) else None for _,v in samples],
                      "status":(row.get("group_summary") or {}).get(key,{}).get("status")}
    out={
        "race_id":RID,"actual_date":row.get("actual_date"),
        "source":row["source"],"price_phase":"FINAL_RESEARCH_ONLY",
        "source_status":row["source_status"],
        "group_summary":summary,
        "saved_prediction_or_tickets_used":False,
        "prebet_usable":False,
        "2026_results_used":False,
    }
    print("USER_SELECTED_FINAL_COMBINATION_ODDS_PROBE",json.dumps(out,sort_keys=True))
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    x=p.parse_args()
    inspect(x.source)

if __name__=="__main__":main()
