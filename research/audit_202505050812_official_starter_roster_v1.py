#!/usr/bin/env python3
"""Cross-check full 202505050812 historical card and FINAL WIN quote roster.

Outputs ONLY horse_number/id/name and entry_status; no outcome is accessed.
The canonical horse roster comes from immutable existing BACKFILL day pack.
"""
from __future__ import annotations
import argparse,gzip,json

RID="202505050812"

def find_race(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        matches=[]
        for line in f:
            if not line.strip():continue
            r=json.loads(line)
            if str((r.get("race") or {}).get("race_id"))==RID:
                matches.append(r)
    if len(matches)!=1:raise ValueError("CANONICAL_DAY_CARD_NOT_SINGLE_MATCH")
    return matches[0]

def inspect(path):
    r=find_race(path)
    horses=r.get("entries")
    if not isinstance(horses,list) or not horses:raise ValueError("CANONICAL_DAY_CARD_MISSING_ENTRIES")
    indexed={}
    for h in horses:
        n=h.get("horse_number")
        if type(n) is not int or n<1 or n>24 or n in indexed:
            raise ValueError("CANONICAL_DAY_CARD_HORSE_NUMBERS_NOT_UNIQUE")
        indexed[n]={"horse_number":n,"horse_id":h.get("horse_id"),
                    "horse_name":h.get("horse_name"),
                    "entry_status":h.get("entry_status")}
    status={}
    for h in indexed.values():
        status[h["entry_status"]]=status.get(h["entry_status"],0)+1
    print("BLIND_202505050812_BACKFILL_CANONICAL_FIELD",
          json.dumps({"race_id":RID,"entry_count":len(indexed),"status_counts":status,
          "horses":sorted(indexed.values(),key=lambda x:x["horse_number"]),
          "no_finish_results_read":True},ensure_ascii=False,sort_keys=True),flush=True)
    return indexed

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    a=p.parse_args()
    inspect(a.source)

if __name__=="__main__":main()
