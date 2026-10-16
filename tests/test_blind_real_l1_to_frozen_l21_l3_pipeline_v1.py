#!/usr/bin/env python3
"""Strict real L1+Risk join; independently synthetic L2.1->L3 V3 wire test.

Do not publish synthetic quotes/p3 or treat tests as 2025 betting advice.
"""
import copy
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
from bridge_blind_original_l1_to_l175_v1 import bridge, CONTRACT
import run_github_race_ids_v1 as runner
from l3_v3_operational import POLICY

RACE="202505050812"
DAY="2025-11-30"
QUOTED="2025-11-30T15:30:00+09:00"
CUTOFF="2025-11-30T16:00:00+09:00"
START="2025-11-30T16:05:00+09:00"
KEY="SYNTHETIC_TEST_ONLY:"+RACE
TEST_URL="https://www.jra.go.jp/test-only"
IDENTITIES=[f"TEST_ONLY_NOT_ACTUAL_HORSE_{i}" for i in range(1,18)]

def original_just_for_bridge():
    target={
        "contract":"KEIBA_UNSCORED_RACE_FEATURE_EXTRACT_V1",
        "race_id":RACE,
        "source_prediction_phase":"FINAL_PRESTART_PROJECTED_FROM_HISTORICAL_SNAPSHOT",
        "source_manifest_count_and_date_range_verified":True,
        "actual_source_capture_time_independently_verified":False,
        "used_target_labels":False,"used_results_payouts_final_odds":False,
        "used_historical_oos_target_predictions":False,
        "rows":[{"_race_id":RACE,"_horse_id":hid,
                 "_race_date":DAY,"horse_number":i}
                for i,hid in enumerate(IDENTITIES,1)]
    }
    ability={
        "contract":"KEIBA_202505050812_REAL_BLIND_ORIGINAL_ABILITY_V1",
        "race_id":RACE,"training_years":[2023,2024],
        "selected_target_outcomes_or_final_odds_used":False,
        "target_saved_historical_predictions_read":False,
        "genuine_asof_source_timestamps_verified":False,
        "horse_scores":[{"horse_id":hid,"ai_raw_rank":i,
                         "ability_score":float(18-i),
                         "ability_score_z":(9-i)/5,
                         "ability_gap_to_best":i-1.}
                        for i,hid in enumerate(IDENTITIES,1)]
    }
    risk={
        "contract":"KEIBA_BLIND_202505050812_FROZEN_CORRECTION_RISK_V1",
        "race_id":RACE,"risk_prior_teacher_years":[2021,2022,2023,2024],
        "risk_seed":8702500,"selected_2025_outcome_or_final_odds_used":False,
        "selected_2025_saved_prediction_used":False,
        "genuine_target_prestart_capture_time_independently_verified":False,
        "original_oos_metric_reproduction":[
            {"oos_test_year":yr,"metric_reference_exact":True}
            for yr in (2021,2022,2023,2024)],
        "horse_risk":[{"horse_id":hid,"ai_raw_rank":i,
                       "correction_risk":0.07+i/100}
                      for i,hid in enumerate(IDENTITIES,1)]
    }
    return ability,risk,target

def write_synthetic_downstream(tmp,joined):
    # Entirely fabricated inputs solely for a deterministic wiring unit test.
    root=Path(tmp);race_dir=root/RACE;race_dir.mkdir()
    card={"contract":"KEIBA_GITHUB_JRA_RACE_CARD_V1",
        "race":{"raceKey":KEY,"canonicalRaceId":RACE,
                "raceDate":DAY,"startTime":"16:05","raceStatus":"SCHEDULED",
                "scheduleStatus":"ACTIVE","status":"OFFICIAL",
                "sourceUrl":TEST_URL},
        "entries":[{"raceKey":KEY,"horseNo":x["horse_number"],
                    "canonicalHorseId":x["horse_id"],
                    "entryStatus":"ACTIVE"}
                   for x in joined["horses"]],
        "capture":{"source_system":"KEIBA_GITHUB_JRA",
                   "final_probe_checked":True,"is_final":False,
                   "source_status":"LIVE"}}
    weights=[.4]+[.6/16]*16
    model={
      "contract":"KEIBA_L21_ORIGINAL_ANCHOR_FEATURES_PREBET_V1",
      "race_id":RACE,"canonical_race_key":KEY,
      "anchor_source":"EXACT_EXISTING_ANCHOR_FULL",
      "anchor_horse_id":IDENTITIES[0],
      "feature_source":"ORIGINAL_L21_HORSE_FEATURES",
      "input_phase":"PRE_RACE","features_asof_asserted":True,
      "features_observed_at":"2025-11-30T15:00:00+09:00",
      "win_odds_observed_at":QUOTED,
      "anchor_computed_at":"2025-11-30T15:35:00+09:00",
      "horses":[{
          "horse_id":row["horse_id"],"horse_number":row["horse_number"],
          "P__ANCHOR_FULL":weights[row["horse_number"]-1],
          "risk":row["correction_risk"],
          "distill_score_z":row["distill_score_z"],
          "distill_ai_rank":row["ai_raw_rank"],
          "market_rank":row["horse_number"],
          "market_prob_norm":1/17,
          "win_odds_at_cutoff":float(row["horse_number"]+2)}
          for row in joined["horses"]]}
    def oddsrow(bt,ns,price):
        return {"raceKey":KEY,"betType":bt,
          "selection1":ns[0],"selection2":ns[1] if len(ns)>1 else None,
          "selection3":ns[2] if len(ns)>2 else None,
          "odds":price,"oddsMin":None,"oddsMax":None,
          "observedAt":QUOTED,"sourceUrl":TEST_URL}
    h=list(range(1,18))
    rows=[oddsrow("QUINELLA",(a,b),16.)
          for a,b in itertools.combinations(h,2)]
    rows += [oddsrow("TRIO",(a,b,c),35.)
             for a,b,c in itertools.combinations(h,3)]
    rows += [oddsrow("TRIFECTA",(a,b,c),240.)
             for a,b,c in itertools.permutations(h,3)]
    assert len(rows)==136+680+4080==4896
    quotes={"contract":"KEIBA_JRA_COMBINATION_ODDS_ASOF_V1",
        "canonical_race_id":RACE,"raceKey":KEY,
        "source_system":"KEIBA_GITHUB_JRA","source_phase":"PRE_RACE",
        "is_final":False,"source_status":"LIVE",
        "source_confirmed_prestart":True,
        "observed_at":QUOTED,"bet_cutoff_at":CUTOFF,
        "scheduled_start_at":START,"rows":rows,
        "win_odds_rows":[oddsrow("WIN",(n,),float(n+2)) for n in h]}
    p3={"contract":"KEIBA_L17_KING_OUTSIDER_P3_ASOF_V1",
        "race_id":RACE,"source":"L17_KING_OUTSIDER_PAST_ONLY_WF_P3",
        "model_version":"TEST_FIXTURE_NO_REAL_MODEL",
        "coverage_scope":"TOP10_OTHER_UNKNOWN",
        "observed_before_bet":True,"source_independently_verified":False,
        "observed_at":QUOTED,
        "horses":[{"horse_number":x["horse_number"],
                   "horse_id":x["horse_id"],"p3":0.06}
                  for x in joined["horses"]]}
    sources={
        "race_card":card,"original_l21_features":model,
        "jra_prebet_quotes":quotes,"original_l175_horse_p3":p3}
    manifest={
        "contract":runner.MANIFEST,
        "races":{RACE:{}},
        "budget":{"race_budget_yen":{RACE:100000},
            "max_race_exposure_yen":100000,
            "available_cash_by_date_yen":{DAY:100000},
            "already_reserved_same_day_yen":{DAY:0},
            "max_day_exposure_yen":100000}}
    for key,body in sources.items():
        p=race_dir/(key+".json")
        p.write_text(json.dumps(body),encoding="utf8")
        manifest["races"][RACE][key]=f"{RACE}/{key}.json"
    return manifest

class BlindBridge(unittest.TestCase):
    def test_original_l1_risk_can_join_all_seventeen_without_asof_promotion(self):
        output=bridge(*original_just_for_bridge())
        self.assertEqual(output["contract"],CONTRACT)
        self.assertEqual(output["horse_count"],17)
        self.assertEqual(len(output["horses"]),17)
        self.assertTrue(output["l1_original_ability_and_risk_bridge_connected"])
        self.assertFalse(output["frozen_l21_ready"])
        self.assertFalse(output["official_l3_v3_ready"])
        self.assertFalse(output["purchase_ready"])
        self.assertEqual(len(output["blocked_reasons"]),5)
        self.assertTrue(all(x["correction_risk"]<1 for x in output["horses"]))

    def test_mismatched_horses_or_different_target_predictions_rejected(self):
        ability,risk,target=original_just_for_bridge()
        risk["horse_risk"][0]["horse_id"]="NOT_SAME_HORSE"
        with self.assertRaisesRegex(ValueError,"HORSE_SET_MISMATCH"):
            bridge(ability,risk,target)
        ability,risk,target=original_just_for_bridge()
        target["rows"][0]["final_win_odds"]=3
        # Reject any outcome field even if other research flags are false.
        with self.assertRaisesRegex(ValueError,"BLIND_TARGET_RESULT_FIELD_FORBIDDEN"):
            bridge(ability,risk,target)

    def test_synthetic_17_horse_full_race_id_to_original_l21_to_l3_v3(self):
        joined=bridge(*original_just_for_bridge())
        with tempfile.TemporaryDirectory() as tmp:
            manifest=write_synthetic_downstream(tmp,joined)
            result=runner.run([RACE],manifest,tmp)
            rec=result["races"][0]
            self.assertEqual(rec["status"],"PAPER_ONLY_COMPLETE",rec.get("blocked_reason"))
            self.assertEqual(len(rec["rankings"]["ai_raw"]),17)
            self.assertEqual(rec["l2_1"]["frozen_version"],"2.1.0")
            self.assertEqual(rec["l2_1"]["policy"],"STANDARD")
            self.assertEqual(rec["l2_1"]["ticket_quote_missing_unique_count"],0)
            self.assertGreater(rec["l2_1"]["physical_ticket_count"],0)
            self.assertEqual(rec["l3_v3"]["required_ticket_count"],
                             rec["l2_1"]["physical_ticket_count"])
            self.assertEqual(
                {(x["bet_type"],x["canonical_selection"]) for x in rec["l2_1"]["tickets"]},
                {(x["bet_type"],x["canonical_selection"]) for x in rec["l3_v3"]["tickets"]})
            self.assertTrue(all(x["stake_yen"] in (100,200,300)
                                for x in rec["l3_v3"]["tickets"]))
            self.assertEqual(result["l3_policy"],POLICY)
            self.assertFalse(result["actual_bets_placed"])
            self.assertFalse(result["verified_collector_timestamp_provenance"])
            self.assertTrue(joined["research_postrace_snapshot_is_not_verified_prebet"])
            print("SYNTHETIC_ONLY_FULL_17_HORSE_PIPELINE_L21_L3_PASS",
                  "horse_count",17,"tickets",rec["l2_1"]["physical_ticket_count"],
                  "total_paper_yen",rec["l3_v3"]["portfolio_total_yen"],
                  "actual_prebet_quotes_or_p3_present",False,flush=True)

if __name__=="__main__":
    unittest.main()
