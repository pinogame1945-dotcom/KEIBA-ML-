#!/usr/bin/env python3
"""Original frozen ANCHOR_FULL integration, with synthetic as-of inputs ONLY.

Actual 2025 archived coefficients + 19-feature parity are independently checked
by original-anchor-2025-parity-github.yml. Here no future results, no network.
Only test code injects a historical clock; production CLI never does.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
sys.path.insert(0,str(ROOT/"tests"))
import run_github_race_ids_v1 as entry
from test_github_race_id_integration_v1 import IDS, TIME, CUTOFF, DAY, fixture, write_fixture

FAKE_TIME=dt.datetime.fromisoformat("2025-01-06T13:34:00+09:00")

def parameters():
    features=tuple(entry.ANCHOR.MODEL.FEATURES)
    return {
      "contract":"KEIBA_ORIGINAL_ANCHOR_FULL_PRIOR_TRAINED_PARAMS_V1",
      "trained_only_on_years":[2022,2023,2024],
      "variant":"ANCHOR_FULL",
      "features_in_original_order":list(features),
      "not_promoted_or_authenticated_for_live_betting":True,
      "coefficients":{name:(.4 if name=="distill_score_z" else 0.) for name in features},
      "standardizer":{name:{"mean":0.,"std":1.,"median":0.} for name in features},
      "source":"SYNTHETIC_COEFFICIENTS_TEST_ONLY"
    }

def companions(rid):
    card,old,_,_=fixture(rid)
    return {
      "contract":"KEIBA_L1_ORIGINAL_ABILITY_RISK_ASOF_V1",
      "race_id":rid,
      "canonical_race_key":card["race"]["raceKey"],
      "feature_source":"ORIGINAL_PREBET_ABILITY_AND_RISK",
      "source_observed_before_bet":True,
      "observed_at":"2025-01-06T13:00:00+09:00",
      "horses":[{
         "horse_id":h["horse_id"],
         "horse_number":h["horse_number"],
         "distill_ai_rank":h["distill_ai_rank"],
         "distill_score_z":h["distill_score_z"],
         "distill_score_gap_to_best":float(h["distill_ai_rank"]-1),
         "risk":h["risk"],
      } for h in old["horses"]]
    }

class OriginalAnchorGithub(unittest.TestCase):
    def test_exact_original_feature_shape_and_no_market_proxy(self):
        card,_,quotes,_=fixture(IDS[0])
        model=entry.ANCHOR.calc(parameters(),card,quotes,companions(IDS[0]),now=FAKE_TIME)
        self.assertTrue(model["not_official_live_bet_prediction"])
        self.assertTrue(model["original_companion_provenance_not_independently_verified"])
        horses=model["model_full_probs"]["horses"]
        self.assertEqual(len(horses),8)
        self.assertAlmostEqual(sum(x["P__ANCHOR_FULL"] for x in horses),1.)
        self.assertEqual(
            sorted(h["distill_ai_rank"] for h in horses),list(range(1,9)))
        self.assertEqual(len(entry.ANCHOR.MODEL.FEATURES),19)
        self.assertEqual(model["source_win_market_capture_at"],TIME)
        self.assertEqual(model["model_full_probs"]["anchor_computed_at"],
                         FAKE_TIME.isoformat())
        l2=entry.L21.execute(card,model["model_full_probs"],quotes)
        self.assertEqual(l2["frozen_policy"],"KEIBA_L2_1_FIXED_PURCHASE_V1")
        self.assertEqual(l2["status"],"PAPER_ONLY_PREBET_QUOTE_VALIDATED")
        self.assertGreater(l2["physical_ticket_count"],0)

    def test_two_races_original_model_to_frozen_l21_to_v3(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=write_fixture(root,IDS)
            for rid in IDS:
                (root/rid/"original_l21_features.json").unlink()
                source["races"][rid].pop("original_l21_features")
                for key,document in (
                    ("original_anchor_parameters",parameters()),
                    ("original_ability_risk",companions(rid))):
                    p=root/rid/(key+".json")
                    p.write_text(json.dumps(document),encoding="utf-8")
                    source["races"][rid][key]=rid+"/"+p.name
            true_calc=entry.ANCHOR.calc
            # Only synthetic tests can inject historical clock. Production
            # execution always uses real wall-clock UTC and rejects stale odds.
            with mock.patch.object(entry.ANCHOR,"calc",
                side_effect=lambda a,c,q,cmp:true_calc(
                    a,c,q,cmp,now=FAKE_TIME)):
                result=entry.run(list(IDS),source,root)
            self.assertEqual(len(result["races"]),2)
            for x in result["races"]:
                self.assertEqual(x["status"],"PAPER_ONLY_COMPLETE",x.get("blocked_reason"))
                self.assertEqual(x["original_anchor_source"],
                                "ORIGINAL_FROZEN_2025_PRIOR_MODEL_GITHUB_INFERENCE")
                self.assertEqual(len(x["rankings"]["ai_raw"]),8)
                self.assertEqual(len(x["rankings"]["expected_value"]),8)
                self.assertGreater(x["l2_1"]["physical_ticket_count"],0)
                self.assertEqual(
                    {(t["bet_type"],t["canonical_selection"])
                     for t in x["l2_1"]["tickets"]},
                    {(t["bet_type"],t["canonical_selection"])
                     for t in x["l3_v3"]["tickets"]})
                self.assertTrue(all(t["stake_yen"] in (100,200,300)
                                    for t in x["l3_v3"]["tickets"]))
            self.assertFalse(result["actual_bets_placed"])

    def test_original_model_missing_full_field_risk_blocks(self):
        card,_,quotes,_=fixture(IDS[0])
        a=companions(IDS[0])
        a["horses"].pop()
        with self.assertRaisesRegex(ValueError,"ORIGINAL_ABILITY_RISK_FIELD_COVERAGE"):
            entry.ANCHOR.calc(parameters(),card,quotes,a,now=FAKE_TIME)

    def test_forbid_results_in_original_companions(self):
        card,_,quotes,_=fixture(IDS[0])
        a=companions(IDS[0])
        a["horses"][0]["finish_position"]=1
        with self.assertRaisesRegex(ValueError,"POSTRACE_OR_OUTCOME_COLUMNS_FORBIDDEN"):
            entry.ANCHOR.calc(parameters(),card,quotes,a,now=FAKE_TIME)

    def test_forbid_proxy_ranks_and_fake_coefficients_year(self):
        card,_,quotes,_=fixture(IDS[0])
        a=companions(IDS[0])
        a["feature_source"]="SEVEN_KING_PROXY"
        with self.assertRaisesRegex(ValueError,"SEVEN_KING_OR_ARCHIVED_POSTRACE_NOT_ALLOWED"):
            entry.ANCHOR.calc(parameters(),card,quotes,a,now=FAKE_TIME)
        p=parameters();p["trained_only_on_years"]=[2023,2024,2025]
        with self.assertRaisesRegex(ValueError,"ANCHOR_TRAINING_YEAR"):
            entry.ANCHOR.calc(p,card,quotes,companions(IDS[0]),now=FAKE_TIME)

    def test_old_historical_clock_never_passes_production_entry(self):
        card,_,quotes,_=fixture(IDS[0])
        with self.assertRaisesRegex(ValueError,"REAL_ANCHOR_INFERENCE_TIME"):
            entry.ANCHOR.calc(parameters(),card,quotes,companions(IDS[0]))

    def test_mixing_saved_probs_and_prior_coefficients_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            manifest=write_fixture(root,IDS[:1])
            rid=IDS[0]
            manifest["races"][rid]["original_anchor_parameters"]="anything"
            result=entry.run([rid],manifest,root)
            self.assertEqual(result["races"][0]["status"],"BLOCKED")
            self.assertEqual(result["races"][0]["blocked_reason"],
              "CANNOT_MIX_PRIOR_MODEL_AND_EXTERNALLY_SCORED_ANCHOR")

if __name__=="__main__":
    unittest.main()
