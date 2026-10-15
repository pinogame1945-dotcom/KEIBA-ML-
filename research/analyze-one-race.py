#!/usr/bin/env python3
"""One command to inspect a historical race, existing models, and saved kings.

No training, no race result labels, and no full-year hindsight odds are
converted to live features. Saved scores are separate archival replay.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from market_context_v1 import attach_win_market

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
        "prior_year_new_model_inference":{
            "status":"NOT_REQUESTED",
            "note":"Only prior-year model selection can be valid for original target-year evaluation; historical FINAL race features still need timestamp verification."
        },
        "archived_final_market_fair_value":{
            "status":"NOT_REQUESTED","not_prebet":True,
            "note":"Separate unpromoted FINAL-odds research model. NEVER treat as live L1.75 price."
        },
        "saved_l17_podium":{
            "status":"NOT_REQUESTED",
            "note":"L1.7 saved fold OOS podium probability, not L1.75 and not a 2025-trained deployment model"
        },
        "saved_seven_kings":{
            "status":"NOT_REQUESTED",
            "note":"Use --seven-kings to load existing saved horse-level scores from Kaggle metadata."
        },
        "files":{
            "features":str(feature_file),"model_rankings":str(model_file)
        },
        "disclaimer":"This is a historical race reconstruction, not verified timestamped live odds or pre-race racecard. Historical OOF replays are not new model inference."
    }
    if a.github_saved:
        if a.council_archive or a.outsider_ballots or a.seven_kings:
            raise ValueError("--github-saved cannot be combined with other archived-score sources")
        year=feature["race_id"][:4]
        cache=folder/"_saved_council_session"
        gather=subprocess.run(
            ["bash",str(ROOT/"scripts"/"fetch-saved-council-github.sh"),
             year,str(cache)],cwd=ROOT,capture_output=True,text=True)
        if gather.returncode:
            report["saved_seven_kings"]={
                "status":"BLOCKED",
                "reason":"GITHUB_SAVED_ARCHIVE_UNAVAILABLE",
                "detail":(gather.stderr or gather.stdout)[-1000:]
            }
        else:
            a.council_archive=str(cache/f"council-y{year}.jsonl.gz")
            a.outsider_ballots=str(cache/f"outsider-ballots-y{year}.csv.gz")
            report["files"]["saved_archive_session"]=str(cache)
    if a.outsider_ballots and not a.council_archive:
        raise ValueError("--outsider-ballots requires --council-archive")
    if a.council_archive:
        if a.seven_kings:
            raise ValueError("Choose one saved-score source: --council-archive OR --seven-kings")
        archived_file=folder/"council-saved-ranks.json"
        args=[sys.executable,str(ROOT/"research"/"replay-archived-council.py"),
              "--race",feature["race_id"],
              "--council-file",str(Path(a.council_archive).resolve()),
              "--output",str(archived_file)]
        if a.outsider_ballots:
            args.extend(["--ballots-file",str(Path(a.outsider_ballots).resolve())])
        execute=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
        if execute.returncode:
            report["saved_seven_kings"]={
                "status":"BLOCKED",
                "reason":"GITHUB_ARCHIVE_REPLAY_FAILED",
                "detail":(execute.stderr or execute.stdout)[-1000:]
            }
        else:
            archive=json.loads(archived_file.read_text(encoding="utf-8"))
            expected=set(horse_matrix)
            actual={h["horse_id"] for h in archive["horses"]}
            if expected!=actual or len(archive["horses"])!=len(expected):
                report["saved_seven_kings"]={
                    "status":"BLOCKED","reason":"HORSE_ID_COVERAGE_DRIFT",
                    "expected_count":len(expected),"archived_count":len(archive["horses"])
                }
            else:
                for item in archive["horses"]:
                    horse=horse_matrix[item["horse_id"]]
                    horse["saved_seven_king_ranks"]=item["king_ranks"]
                    horse["archived_rank_sum_comparator_rank"]=item["pure_7king_rank_sum_comparator_rank"]
                    for king,rank in item["king_ranks"].items():
                        horse["model_ranks"]["KING:"+king]={
                            "rank":rank,"score":None,
                            "origin":"HISTORICAL_SAVED_FINAL_OOF_REPLAY"
                        }
                    if archive["outsider_status"]=="VERIFIED_SAVED_BALLOTS":
                        horse["saved_outsider_points"]=item["outsider_points"]
                        horse["saved_outsider_votes"]=item["outsider_votes"]
                        for candidate,rank in item["outsider_top3_from_each"].items():
                            if rank is None:continue
                            horse["model_ranks"]["OUTSIDER:"+candidate]={
                                "rank":rank,"score":None,
                                "origin":"HISTORICAL_SAVED_FINAL_TOP3_BALLOT_ONLY"
                            }
                report["saved_seven_kings"]={
                    "status":"GITHUB_ARCHIVED_7KING_REPLAY",
                    "source_contract":archive["contract"],
                    "source_prediction_phase":archive["source_prediction_phase"],
                    "outsider_status":archive["outsider_status"],
                    "seven_king_count":len(archive["expert_order"]),
                    "outsider_model_count":len(archive.get("outsider_ballots") or {}),
                    "rank_sum_is_official_router":False,
                    "note":"Archived FINAL-phase research scores (not a live re-prediction). Results/final market odds not used."
                }
                report["files"]["archived_council"]=str(archived_file)
    if a.github_saved and a.council_archive and a.outsider_ballots:
        if feature["race_id"][:4]=="2021":
            report["saved_l17_podium"]={"status":"NO_PRIOR_TRAINING_YEAR"}
        elif report["saved_seven_kings"]["status"]=="GITHUB_ARCHIVED_7KING_REPLAY":
            cache=folder/"_saved_council_session"
            out=folder/"l17-saved-podium.json"
            executed=subprocess.run([
                sys.executable,str(ROOT/"research"/"replay-saved-l17.py"),
                "--race-report",str(folder/"council-saved-ranks.json"),
                "--predictions",str(cache/"l17-oof-all.csv.gz"),
                "--summary",str(cache/"l17-fold-summary.json"),
                "--output",str(out)
            ],cwd=ROOT,capture_output=True,text=True)
            if executed.returncode:
                report["saved_l17_podium"]={
                    "status":"BLOCKED","reason":"L17_OOF_PROVENANCE_OR_SCORE_DRIFT",
                    "detail":(executed.stderr or executed.stdout)[-900:]
                }
            else:
                l17=json.loads(out.read_text(encoding="utf-8"))
                scores=l17["model_scores"]
                for hid,record in scores.items():
                    h=horse_matrix[hid]
                    h["l17_saved_p3_oos"]=record["p3_oos"]
                    h["l17_saved_p3_baseline"]=record["p3_rank_baseline"]
                    h["l17_saved_p3_delta"]=record["p3_delta"]
                    h["l17_original_consensus_rank"]=record["l17_consensus_rank"]
                    h["model_ranks"]["L17:OUTSIDER_PODIUM_P3"]={
                        "rank":record["l17_consensus_rank"],
                        "score":record["p3_oos"],
                        "origin":l17["prediction_origin"],
                        "metric":"PODIUM_PROBABILITY_NOT_ABILITY_RANK"
                    }
                report["saved_l17_podium"]={
                    "status":"ARCHIVED_OOS_P3",
                    "trained_on_prior_years":l17["prior_train_years"],
                    "score_rows":len(scores),
                    "no_rerank":True,
                    "source":l17["model_name"],
                    "not_l175":True,
                    "note":"Score replay only; excludes source CSV finish and y3 labels"
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
    if a.prior_year_models:
        saved=json.loads(Path(a.prior_year_models).read_text(encoding="utf-8"))
        if (saved.get("contract")!="KEIBA_PRIOR_YEAR_MODEL_SINGLE_RACE_V1"
            or saved.get("race_id")!=report["race_id"]
            or saved.get("model_count")!=7
            or saved.get("model_parameters_selected_before_target") is not True
            or saved.get("input_outcome_labels_used") is not False):
            raise ValueError("PRIOR_YEAR_FROZEN_MODEL_PROVENANCE_CONTRACT_DRIFT")
        if saved.get("snapshot_prediction_phase")!="FINAL_ARCHIVED_TIMESTAMP_NOT_VERIFIED":
            raise ValueError("PRIOR_YEAR_SNAPSHOT_PHASE_DRIFT")
        model_names={m.get("alias") for m in saved["models"]}
        if model_names!={"core4","pedlegacy","condition","full","light","pedv1","condrc"}:
            raise ValueError("PRIOR_YEAR_7KING_MODEL_SET_DRIFT")
        if any(str(m.get("model_selected_no_later_than") or "")>=report["race_date"]
               for m in saved["models"]):
            raise ValueError("PRIOR_YEAR_MODEL_SELECTION_NOT_PRIOR")
        if len(saved["horses"])!=len(horse_matrix) or (
            {h["horse_id"] for h in saved["horses"]}!=set(horse_matrix)):
            raise ValueError("PRIOR_YEAR_PRED_HORSE_ID_COVERAGE_DRIFT")
        for item in saved["horses"]:
            hid=item["horse_id"]
            row=horse_matrix[hid]
            ranks=item["king_ranks"]
            probs=item["king_raw_probabilities"]
            if set(ranks)!=model_names or set(probs)!=model_names:
                raise ValueError("PRIOR_YEAR_7KING_PER_HORSE_COVERAGE_DRIFT")
            row["prior_year_king_rank_sum"]=item["prior_year_7king_rank_sum_comparator_rank"]
            row["prior_year_king_ranks"]=ranks
            row["prior_year_king_raw_probabilities"]=probs
            for model_name in sorted(model_names):
                row["model_ranks"]["PRIOR_YEAR_LGBM:"+model_name]={
                    "rank":ranks[model_name],
                    "score":probs[model_name],
                    "origin":"ACTUAL_MODEL_INFERENCE_PRIOR_YEAR_PARAMETERS_ON_HISTORICAL_FINAL",
                    "source_final_timing_verified":False
                }
        report["prior_year_new_model_inference"]={
            "status":"ORIGINAL_MODELS_SCORED_NOT_VERIFIED_PREBET",
            "model_count":7,
            "model_selected_before_race":True,
            "final_snapshot_prebet_timestamp_verified":False,
            "not_official_frozen_council":True,
            "not_l175":True,
        }
    report["horses"]=sorted(horse_matrix.values(),
        key=lambda h:(h["horse_number"] is None,
                      h["horse_number"] if h["horse_number"] is not None else 999,
                      h["horse_id"]))
    if a.github_fair_value:
        if a.market_win_snapshot:
            raise ValueError("MUST_NOT_MIX_ARCHIVED_FINAL_MARKET_MODEL_WITH_PRESTART_QUOTES")
        year=int(report["race_id"][:4])
        if year<2023 or year>2025:
            raise ValueError("FAIR_OOS_VALID_ONLY_2023_TO_2025_2026_SEALED")
        src="research-results/market-fixed-anchor-residual-v2/run-38659856085"
        dest=folder/"_saved_market_fair_session"
        dest.mkdir(exist_ok=True)
        files=(
            ("oos-fair-value-predictions.csv.gz",11000000),
            ("summary.json",150000),
            ("lambda-selection.csv",150000)
        )
        success=True
        for name,max_bytes in files:
            p=dest/name
            grab=subprocess.run([
                sys.executable,str(ROOT/"scripts"/"github-readonly-archive.py"),
                "--repo","pinogame1945-dotcom/KEIBA-ML-",
                "--ref","freeze/l2.1-standard50-to-l3-20261015",
                "--path",src+"/"+name,
                "--dest",str(p),"--max-bytes",str(max_bytes)
            ],cwd=ROOT,capture_output=True,text=True)
            if grab.returncode:
                report["archived_final_market_fair_value"]={
                    "status":"BLOCKED","reason":"GITHUB_FAIR_OOS_SOURCE_UNAVAILABLE",
                    "detail":(grab.stderr or grab.stdout)[-500:],
                    "not_prebet":True
                }
                success=False
                break
        if success:
            from importlib.util import spec_from_file_location,module_from_spec
            spec=spec_from_file_location("fair_oos",
                ROOT/"research"/"replay-final-market-oos-v2.py")
            module=module_from_spec(spec)
            spec.loader.exec_module(module)
            try:
                saved=module.replay(report,
                    dest/"oos-fair-value-predictions.csv.gz",
                    dest/"summary.json",dest/"lambda-selection.csv")
            except ValueError as exc:
                report["archived_final_market_fair_value"]={
                    "status":"BLOCKED",
                    "reason":"FAIR_OOS_RACE_FIELD_OR_FOLD_DRIFT",
                    "detail":str(exc),
                    "not_prebet":True
                }
            else:
                for hid,vals in saved["horse_results"].items():
                    horse_matrix[hid]["archived_final_market_oos_research"]=vals
                report["archived_final_market_fair_value"]={
                    "status":saved["status"],
                    "source_research_contract":saved["source_research_contract"],
                    "models":list(vals["oos_win_probabilities"]),
                    "horse_count":saved["horse_count"],
                    "trained_on_prior_years_only":True,
                    "source_final_odds":True,
                    "not_prebet":True,
                    "not_production_l175":True,
                    "do_not_use_for_live_betting":True,
                }
    if a.market_win_snapshot:
        quote=json.loads(Path(a.market_win_snapshot).read_text(encoding="utf-8"))
        report=attach_win_market(report,quote)
    else:
        report["market_context"]={
            "status":"NOT_PROVIDED",
            "not_l175":True,
            "not_bet_roi_or_ev":True,
            "reason":"No independently timestamped original JRA pre-start win odds snapshot supplied; historical FINAL odds must not be substituted."
        }
    output=folder/"analysis.json"
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # The one-race horse analysis may be used as a read-only identity gate
    # for an independently supplied ORIGINAL ANCHOR_FULL and full live JRA
    # market snapshot. Do NOT replace the frozen axis with seven kings.
    if a.mobile_l21_capture or a.original_l21_features or a.l3_limits:
        from importlib.util import spec_from_file_location,module_from_spec
        loader=spec_from_file_location("l21_l3_runner",
            ROOT/"research"/"run-one-race-paper-pipeline.py")
        runner=module_from_spec(loader)
        sys.modules[loader.name]=runner
        loader.loader.exec_module(runner)
        handoff=runner.run(
            report["race_id"],folder/"l21-l3",
            a.mobile_l21_capture,a.original_l21_features,a.l3_limits,
            output)
        report["l21_l3_paper_handoff"]={
            "status":handoff["status"],
            "stages":handoff["stages"],
            "required_inputs_missing":handoff["required_inputs_missing"],
            "path":str(folder/"l21-l3"/"pipeline-status.json"),
            "no_live_bets_executed":True,
        }
    else:
        report["l21_l3_paper_handoff"]={
            "status":"BLOCKED_SOURCE_NOT_PROVIDED",
            "required_inputs_missing":["original_ANCHOR_FULL_asof",
                "same_JRA_WIN_QUINELLA_TRIO_TRIFECTA_prebet_quote",
                "L3_budget_and_limits"],
            "no_live_bets_executed":True,
        }
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "race_id":report["race_id"],"runners":report["starters"],
        "models_ready":models["scored_models"],
        "models_missing_or_blocked":models["blocked_or_missing_models"],
        "seven_kings":report["saved_seven_kings"]["status"],
        "prior_year_new_inference":report["prior_year_new_model_inference"]["status"],
        "l17_p3":report["saved_l17_podium"]["status"],
        "market_context":report["market_context"]["status"],
        "historical_final_fair_oos":report["archived_final_market_fair_value"]["status"],
        "frozen_l21_to_l3":report["l21_l3_paper_handoff"]["status"],
        "archive_outsider_model_count":report["saved_seven_kings"].get("outsider_model_count",0),
        "report":str(output)
    },ensure_ascii=False,indent=2))
    if a.require_scored and models["scored_models"]==0 and report["saved_seven_kings"]["status"] not in ("ARCHIVED_OOF_REPLAY","GITHUB_ARCHIVED_7KING_REPLAY") and report["prior_year_new_model_inference"]["status"]!="ORIGINAL_MODELS_SCORED_NOT_VERIFIED_PREBET":
        raise SystemExit("No verified model score available; refusing to label feature-only results as predictions")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--race",required=True,help="Netkeiba race URL or canonical 12-digit ID")
    p.add_argument("--source-root",required=True,help="Existing KEIBA-BACKFILL checkout")
    p.add_argument("--history-from",default="2021-01-01")
    p.add_argument("--registry",default=str(ROOT/"models"/"one-race-registry.json"))
    p.add_argument("--output-dir",default="out/one-race-analysis")
    p.add_argument("--github-saved",action="store_true",
                   help="Restore already-committed 7 king and 13 Outsider score archive from public GitHub once")
    p.add_argument("--council-archive",
                   help="Existing Github research archive races/yYYYY.jsonl.gz (historical ranks)")
    p.add_argument("--outsider-ballots",
                   help="Existing Github research archive yYYYY/ballots.csv.gz (13 candidates)")
    p.add_argument("--seven-kings",action="store_true",
                   help="Restore existing saved seven king score files, strictly from Kaggle dataset listing")
    p.add_argument("--prior-year-models",
                   help="Verified JSON produced by seven prior-year trained model scorer; no target-year model selection")
    p.add_argument("--github-fair-value",action="store_true",
                   help="RESEARCH ONLY: read nonpromoted OOS final-odds fair-value predictions; never combine with current pre-race market")
    p.add_argument("--market-win-snapshot",
                   help="Original timestamped PRE_RACE JRA WIN OddsRow snapshot as KEIBA_JRA_WIN_ODDS_ASOF_V1; FINAL odds forbidden")
    p.add_argument("--mobile-l21-capture",
                   help="Original 4-product JRA full PRE_RACE KEIBA-MOBILE export; optional fail-closed L2+L3 integration")
    p.add_argument("--original-l21-features",
                   help="Exact timestamped ORIGINAL ANCHOR_FULL full-race horse features; prior-year seven king is NOT an acceptable substitute")
    p.add_argument("--l3-limits",
                   help="Flat100 money budget/max-ticket-count JSON; L3 blocks instead of trimming frozen L2.1 tickets")
    p.add_argument("--require-scored",action="store_true")
    one_race(p.parse_args())

if __name__=="__main__":main()
