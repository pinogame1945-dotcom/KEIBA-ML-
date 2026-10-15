#!/usr/bin/env python3
"""Replay stored L1.7 podium OOF probabilities only; never use truth labels.

Source is the existing 2021-25 expanding-fold predictions.csv.gz, NOT the
2021-25 deployment model (which would leak when scoring a 2025 race).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

CONTRACT="KEIBA_ARCHIVED_L17_PODIUM_OOF_V1"
SUMMARY_CONTRACT="L17_KING_OUTSIDER_PODIUM_CALIBRATOR_WF_V1"

def require_prob(row,key):
    v=float(row[key])
    if not math.isfinite(v) or not 0<=v<=1:
        raise ValueError(f"invalid {key}")
    return v

def validate_fold(summary,year):
    if summary.get("contract")!=SUMMARY_CONTRACT or not summary.get("2026_sealed"):
        raise ValueError("L17 summary contract/holdout drift")
    matches=[row for row in summary.get("folds",[])
             if int(row["eval_year"])==year]
    if len(matches)!=1:
        raise ValueError("L17 stored OOS fold not found")
    train=matches[0].get("train_years",[])
    if not train or any(int(y)>=year or int(y)<2021 for y in train):
        raise ValueError("L17 fold trains/validates on target or future")
    return sorted(int(y) for y in train)

def replay(race, csvfile, summary_file):
    race_id=str(race["race_id"])
    year=int(race_id[:4])
    if year not in range(2022,2026):
        raise ValueError("L17 valid only 2022..2025; 2026 sealed")
    summary=json.loads(Path(summary_file).read_text(encoding="utf-8"))
    years=validate_fold(summary,year)
    horses=race.get("horses",[])
    if len(horses)<3 or len({x["horse_id"] for x in horses})!=len(horses):
        raise ValueError("historical race horse IDs missing or duplicated")
    expected={x["horse_id"] for x in horses}
    score_by_horse={h["horse_id"]:h.get("saved_outsider_points",h.get("outsider_points")) for h in horses}
    all_rows={}
    opener=gzip.open if str(csvfile).endswith(".gz") else open
    with opener(csvfile,"rt",encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("race_id"))!=race_id:continue
            if int(row.get("year") or -1)!=year:
                raise ValueError("historical L17 year mismatch")
            hid=str(row.get("horse_id") or "")
            if hid not in expected or hid in all_rows:
                raise ValueError("unexpected/duplicated L17 horse "+hid)
            rank=int(row["rank"])
            score=int(row["score"])
            if not (1<=rank<=10 and 0<=score<=39):
                raise ValueError("invalid L17 rank/outsider score")
            if score_by_horse[hid] is not None and score_by_horse[hid]!=score:
                raise ValueError("L17_OUTSIDER_POINTS_DRIFT "+hid)
            p3=require_prob(row,"p3")
            baseline=require_prob(row,"p3_rank_baseline")
            delta=float(row["p3_delta"])
            if not math.isfinite(delta) or abs(p3-baseline-delta)>1e-8:
                raise ValueError("L17 delta mismatch")
            # Explicit whitelist, discard finish, y3, win, or any other labels.
            all_rows[hid]={"horse_id":hid,"l17_consensus_rank":rank,
                           "outsider_score":score,"p3_oos":p3,
                           "p3_rank_baseline":baseline,"p3_delta":delta,
                           "calibration_status":"OOS_WALK_FORWARD"}
    if len(all_rows)!=min(10,len(horses)):
        raise ValueError("L17 Top10 coverage drift")
    ranks=sorted(x["l17_consensus_rank"] for x in all_rows.values())
    if ranks!=list(range(1,len(ranks)+1)):
        raise ValueError("L17 expected complete consensus ranks 1..10")
    return {"contract":CONTRACT,"race_id":race_id,"eval_year":year,
            "prior_train_years":years,"output_scope":"L17_ORIGINAL_CONSENSUS_TOP10_ONLY",
            "prediction_origin":"HISTORICAL_OOS_SAVED_WALK_FORWARD_NOT_NEW_INFERENCE",
            "ranking_policy":"NO_RERANK_KING_UNCHANGED",
            "model_name":"L1.7_OUTSIDER_PODIUM_CALIBRATOR_NOT_L1.75",
            "model_scores":all_rows,
            "important_note":"Predictions are copied from saved fold OOS scores. The source CSV has outcomes, but this replay ignores those columns entirely. Not a proof of timestamped live availability."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--race-report",required=True)
    p.add_argument("--predictions",required=True)
    p.add_argument("--summary",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    report=replay(json.loads(Path(a.race_report).read_text(encoding="utf-8")),
                  a.predictions,a.summary)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ARCHIVED_L17_TOP10_OOS_READY",report["race_id"],len(report["model_scores"]))

if __name__=="__main__":
    main()
