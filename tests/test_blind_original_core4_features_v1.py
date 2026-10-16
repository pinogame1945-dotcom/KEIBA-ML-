#!/usr/bin/env python3
import copy
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
from audit_blind_original_core4_features_v1 import audit

def payload():
    rows=[]
    for i in range(17):
        rows.append({"_race_id":"202505050812","_horse_id":str(2020000001+i),
        "_race_date":"2025-11-30", "race_month":11,"race_day_of_year":334,
        "venue_code":"TOKYO", "opponent_pre_rating":i+1,
        "network_test":i, "style_test":1,
        "actor_jockey_form_30d":i, "auto_recent_lag":i})
    return {"contract":"KEIBA_UNSCORED_RACE_FEATURE_EXTRACT_V1",
    "race_id":"202505050812",
    "source_manifest_count_and_date_range_verified":True,
    "actual_source_capture_time_independently_verified":False,
    "used_target_labels":False,"used_results_payouts_final_odds":False,
    "used_historical_oos_target_predictions":False,
    "rows":rows}

class TestRealCore4SchemaGate(unittest.TestCase):
    def test_original_core4_exact_feature_families_and_identity(self):
        d=audit(payload())
        self.assertEqual(d["core4_feature_count"],7)
        self.assertEqual(d["excluded_non_core4_count"],2)
        self.assertEqual(d["core4_group_counts"],
                         {"BASE":3,"opponent_":1,"auto_":1,"actor_":1})
        self.assertEqual(d["horses"],17)
        self.assertFalse(d["year_2025_saved_AI_targets_used"])
        self.assertEqual(len(d["core4_feature_name_sha256"]),64)

    def test_fail_closed_for_forbidden_postrace_or_saved_signal(self):
        for field in ["finish_position","target_win","P__ANCHOR_FULL",
                      "distill_ai_rank","market_rank","payout_yen",
                      "actual_start_time","market_final_win_odds"]:
            x=payload()
            x["rows"][0][field]=123
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError,"ORIGINAL_CORE4_TARGET_LEAK"):
                    audit(x)

    def test_race_id_count_source_capture_cannot_be_assumed(self):
        x=payload()
        x["rows"].pop()
        with self.assertRaisesRegex(ValueError,"17_HORSES_EXPECTED"):
            audit(x)
        x=payload()
        x["source_manifest_count_and_date_range_verified"]=False
        with self.assertRaisesRegex(ValueError,"SOURCE_MANIFEST"):
            audit(x)
        x=payload()
        x["actual_source_capture_time_independently_verified"]=True
        with self.assertRaisesRegex(ValueError,"SOURCE_CAPTURE"):
            audit(x)

if __name__=="__main__":
    unittest.main()
