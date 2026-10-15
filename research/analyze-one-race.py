#!/usr/bin/env python3
"""One command to inspect a historical race, existing models, and saved kings.

No training, no race result labels, and no full-year hindsight odds are
converted to live features. Saved scores are separate archival replay.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def one_race(a):
    folder=Path(a.output_dir).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    feature_file=folder/"features.json"
    model_file=folder/"model-rankings.json"
    argv=["node",str(ROOT/"research"/"predict-one-race.mjs"),
          "--race",a.race,"--source-root",str(Path(a.source_root).resolve()),
          "--history-from",a.history_from,"--phase","EARLY",
          "--registry",str(Path(a.registry).resolve()),
          "--features-out",str(feature_file),
          "--results-out",str(model_file)]
    run=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True)
    if run.returncode:
        raise RuntimeError("One-race features/scorer failed (no rankings accepted): "+
                           (run.stderr or run.stdout)[-1200:])
    feature=json.loads(feature_file.read_text(encoding="utf-8"))
    models=json.loads(model_file.read_text(encoding="utf-8"))
    if feature["race_id"]!=models["race_id"]:
        raise RuntimeError("RACE_ID_DRIFT: feature and model score IDs differ")
    horse_matrix={
        h["horse_id"]:{
            "horse_id":h["horse_id"],"horse_number":h.get("horse_number"),
            "horse_name":h.get("horse_name"),
            "feature_count":len(h["features"]),
            "feature_non_null":sum(value is not None for value in h["features"].values()),
            "model_ranks":{}
        }
        for h in feature["features_by_horse"]
    }
    for row in models["model_results"]:
        if row["status"]!="SCORED":continue
        for ranked in row["ranking"]:
            hid=ranked["horse_id"]
            if hid not in horse_matrix:raise RuntimeError("MODEL_HORSE_ID_DRIFT "+hid)
            horse_matrix[hid]["model_ranks"][row["model"]]={
                "rank":ranked["rank"],"score":ranked["score"],
                "origin":row.get("origin")
            }
    report={
        "contract":"KEIBA_ONE_RACE_COMBINED_ANALYSIS_V1",
        "race_id":feature["race_id"],
        "race_date":feature["race_date"],
        "mode":"HISTORICAL_EARLY_AS_OF_REPLAY",
        "source_pack":feature["source_pack"],
        "history":feature["history"],
        "starters":len(feature["features_by_horse"]),
        "feature_schema_version":feature["feature_schema_version"],
        "models":models["model_results"],
        "horses":[],
        "saved_seven_kings":{
            "status":"NOT_REQUESTED",
            "note":"Use --seven-kings to load existing saved horse-level scores from Kaggle metadata."
        },
        "files":{
            "features":str(feature_file),"model_rankings":str(model_file)
        },
        "disclaimer":"This is a historical race reconstruction, not verified timestamped live odds or pre-race racecard. Historical OOF replays are not new model inference."
    }
    if a.seven_kings:
        if not os.environ.get("KAGGLE_API_TOKEN"):
            report["saved_seven_kings"]={
                "status":"BLOCKED","reason":"KAGGLE_API_TOKEN_MISSING"
            }
        else:
            seven_file=folder/"seven-king-saved-ranks.json"
            script=ROOT/"research"/"replay-seven-kings-kaggle.sh"
            call=subprocess.run(
                ["bash",str(script),feature["race_id"],str(seven_file)],
                cwd=ROOT,capture_output=True,text=True)
            if call.returncode:
                report["saved_seven_kings"]={
                    "status":"BLOCKED",
                    "reason":"ARCHIVE_SCORE_FETCH_OR_VALIDATION_FAILED",
                    "detail":(call.stderr or call.stdout)[-1000:]
                }
            else:
                seven=json.loads(seven_file.read_text(encoding="utf-8"))
                expected={x["horse_id"] for x in feature["features_by_horse"]}
                actual={x["horse_id"] for x in seven["horses"]}
                if len(seven["horses"])!=len(expected) or actual!=expected:
                    report["saved_seven_kings"]={
                        "status":"BLOCKED",
                        "reason":"HORSE_ID_COVERAGE_DRIFT",
                        "expected_count":len(expected),
                        "saved_count":len(seven["horses"])
                    }
                else:
                    for z in seven["horses"]:
                        item=horse_matrix[z["horse_id"]]
                        item["saved_seven_king_ranks"]=z["king_ranks"]
                        item["saved_seven_king_probabilities"]=z["king_raw_probabilities"]
                        item["archived_rank_sum_comparator_rank"]=z["rank_sum_comparator_rank"]
                    report["saved_seven_kings"]={
                        "status":"ARCHIVED_OOF_REPLAY",
                        "source_contract":seven["contract"],
                        "provenance_note":seven["verification_note"],
                        "rank_sum_is_official_router":False,
                        "horses":seven["horses"]
                    }
    report["horses"]=sorted(horse_matrix.values(),
        key=lambda h:(h["horse_number"] is None,
                      h["horse_number"] if h["horse_number"] is not None else 999,
                      h["horse_id"]))
    output=folder/"analysis.json"
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "race_id":report["race_id"],"runners":report["starters"],
        "models_ready":models["scored_models"],
        "models_missing_or_blocked":models["blocked_or_missing_models"],
        "seven_kings":report["saved_seven_kings"]["status"],
        "report":str(output)
    },ensure_ascii=False,indent=2))
    if a.require_scored and models["scored_models"]==0 and report["saved_seven_kings"]["status"]!="ARCHIVED_OOF_REPLAY":
        raise SystemExit("No verified model score available; refusing to label feature-only results as predictions")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--race",required=True,help="Netkeiba race URL or canonical 12-digit ID")
    p.add_argument("--source-root",required=True,help="Existing KEIBA-BACKFILL checkout")
    p.add_argument("--history-from",default="2021-01-01")
    p.add_argument("--registry",default=str(ROOT/"models"/"one-race-registry.json"))
    p.add_argument("--output-dir",default="out/one-race-analysis")
    p.add_argument("--seven-kings",action="store_true",
                   help="Restore existing saved seven king score files, strictly from Kaggle dataset listing")
    p.add_argument("--require-scored",action="store_true")
    one_race(p.parse_args())

if __name__=="__main__":main()
