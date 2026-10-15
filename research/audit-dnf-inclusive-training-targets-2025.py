#!/usr/bin/env python3
"""DNF-inclusive 2025 research-label coverage audit, NEVER inference features.

The original L1 objective read_year() filtered null finish before feature
building, thus the 5 DNF/started horses disappeared. This research-only
target generator retains STARTED DNF with win=0/top3=0. It is NEVER called
by PREBET prediction and outputs no betting signals or ROI.
"""
from __future__ import annotations
import argparse,csv,gzip,json,math,re
from pathlib import Path

CONTRACT="KEIBA_ALL_STARTED_DNF_INCLUSIVE_TRAINING_TARGET_AUDIT_V1"
def rows(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)

def load_saved(r):
    obs={}
    with gzip.open(r,"rt",encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            if row.get("year")!="2025":continue
            rid=row["_race_id"];hid=row["_horse_id"]
            try:win=int(float(row["target_win"]));top3=int(float(row["target_top3"]))
            except (TypeError,ValueError) as exc:
                raise ValueError("INVALID_ARCHIVED_OOS_TRAINING_LABEL") from exc
            if win not in (0,1) or top3 not in (0,1) or win>top3:
                raise ValueError("ARCHIVED_OOS_WIN_TOP3_LABEL_DRIFT")
            panel=obs.setdefault(rid,{})
            if hid in panel:raise ValueError("DUPLICATE_ARCHIVED_OOS_LABEL")
            panel[hid]={"target_win":win,"target_top3":top3}
    return obs

def generate(pack,predictions):
    saved=load_saved(predictions)
    cases=[]; labels=[]; missing=set(); dnfs=set()
    for rec in rows(pack):
        race=rec.get("race") or {}
        rid=str(race.get("race_id") or "")
        if not re.fullmatch(r"2025\d{8}",rid):
            raise ValueError("ONLY_FROZEN_2025_24_RACES_AUDITED")
        entries=rec.get("entries") or []
        numbered={str(e["horse_id"]):int(e["horse_number"]) for e in entries
                  if e.get("entry_status")=="STARTED"
                  and e.get("horse_id") is not None and e.get("horse_number") is not None}
        if len(numbered)<3:raise ValueError("MISSING_STARTED_HORSES_"+rid)
        result_by_horse={str(e["horse_id"]):e for e in (rec.get("results") or [])
                         if e.get("horse_id") is not None}
        results_by_no={int(e["horse_number"]):e for e in (rec.get("results") or [])
                       if e.get("horse_number") is not None}
        actual_dnf=[]
        for hid,no in numbered.items():
            r=result_by_horse.get(hid) or results_by_no.get(no)
            if not isinstance(r,dict):
                raise ValueError("MISSING_RESULT_STATUS_FOR_STARTED_"+rid+"_"+hid)
            status=r.get("result_status")
            finish=r.get("finish_position")
            if status=="DNF":
                if finish is not None:raise ValueError("DNF_BUT_NUMERIC_FINISH_"+rid)
                win=top3=0
                dnfs.add((rid,hid))
                actual_dnf.append(hid)
            else:
                # BACKFILL result records do not necessarily expose a
                # numeric finish_position under that exact key. For
                # *historical training targets only*, use the original
                # already-archived labeled OOS winner/podium flags.
                historic=saved.get(rid,{}).get(hid)
                if historic is None:
                    raise ValueError("FINISHER_MISSING_FROM_ORIGINAL_OOS_"+rid+"_"+hid)
                win=historic["target_win"];top3=historic["target_top3"]
            labels.append({"race_id":rid,"horse_id":hid,"horse_number":no,
                           "target_win":win,"target_top3":top3,
                           "training_only_derived_from_result":True})
        missing_now=set(numbered)-set(saved.get(rid,{}))
        missing.update((rid,hid) for hid in missing_now)
        if missing_now!=set(actual_dnf):
            raise ValueError("FINISH_FILTER_NOT_FULL_CAUSE_OF_ORIGINAL_OOS_HORSE_EXCLUSIONS_"+
                             rid+":"+str(sorted(missing_now^set(actual_dnf))))
        cases.append({"race_id":rid,"started":len(numbered),
                      "source_oos_horse_count":len(saved.get(rid,set())),
                      "dnf_started_count":len(actual_dnf),
                      "dnf_excluded_horse_ids":sorted(actual_dnf)})
    if len(cases)!=24 or len(labels)<200 or len(dnfs)!=5 or missing!=dnfs:
        raise ValueError("ORIGINAL_DNF_POPULATION_DISCREPANCY")
    return {"contract":CONTRACT,"date":"2025-01-06",
            "race_count":len(cases),"all_started_horse_labels":len(labels),
            "previously_excluded_dnf_horses":len(dnfs),
            "oos_missing_set_equals_actual_DNF":True,
            "both_DNF_win_and_top3_labels_are_zero":True,
            "training_targets_only":True,
            "NOT_USABLE_FOR_PREBET_INFERENCE":True,
            "NOT_A_RETRAINED_OR_EVALUATED_MODEL":True,
            "races":cases,"labels":labels}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--race-pack",required=True)
    p.add_argument("--predictions",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    z=generate(a.race_pack,a.predictions)
    dst=Path(a.out);dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(z,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ALL_STARTED_INCLUDING_DNF_TRAINING_TARGET_AUDIT_PASS",
          z["race_count"],z["all_started_horse_labels"],
          "original_hindsight_excluded_DNF",z["previously_excluded_dnf_horses"],
          "NOT_INFERENCE",flush=True)

if __name__=="__main__":main()
