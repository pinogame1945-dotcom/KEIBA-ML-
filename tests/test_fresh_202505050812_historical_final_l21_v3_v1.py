#!/usr/bin/env python3
"""17 real-starter historical 202505050812 frozen formula contract tests."""
import json
import math
import sys
import unittest
from pathlib import Path
from itertools import combinations,permutations

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
from run_fresh_202505050812_historical_final_l21_v3_v1 import (
  historical_full_frozen,final_price_table,RID,CONTRACT)

STARTERS=list(range(1,17))+[18]
SCRATCHED=17

def mock_input():
    field=[]
    for i,n in enumerate(STARTERS):
        hid=f"SYNTHETIC_{n:02}"
        field.append({"horse_number":n,"horse_id":hid,
                      "horse_name":f"synthetic_{n}","entry_status":"STARTED"})
    field.append({"horse_number":SCRATCHED,"horse_id":"SYNTHETIC_SCRATCHED",
                  "horse_name":"synthetic_scratch","entry_status":"SCRATCHED"})
    card={"race":{"race_id":RID},"entries":field}
    horse=[{"horse_id":f"SYNTHETIC_{n:02}","horse_number":n,
            "ai_raw_rank":i+1,"distill_score_z":(8-i)/5,
            "distill_score_gap_to_best":i/5,
            "correction_risk":0.03+i*0.01}
            for i,n in enumerate(STARTERS)]
    bridge={
      "contract":"KEIBA_BLIND_ORIGINAL_L1_TO_FROZEN_L175_BRIDGE_V1",
      "race_id":RID,
      "l1_original_ability_and_risk_bridge_connected":True,
      "target_result_or_saved_prediction_read":False,
      "horse_count":17,
      "horses":horse
    }
    odds={"race_id":RID,"actual_date":"2025-11-30",
          "source":"NETKEIBA_FINAL","odds_kind":"FINAL",
          "source_status":"result","odds":{}}
    odds["odds"]["1"]={str(n).zfill(2):[2.+n,0,0] for n in range(1,19)}
    odds["odds"]["4"]={
       "".join(str(n).zfill(2) for n in combo):[35.,0,0]
       for combo in combinations(range(1,19),2)}
    odds["odds"]["7"]={
       "".join(str(n).zfill(2) for n in combo):[35.,0,0]
       for combo in combinations(range(1,19),3)}
    odds["odds"]["8"]={
       "".join(str(n).zfill(2) for n in combo):[240.,0,0]
       for combo in permutations(range(1,19),3)}
    from score_github_original_anchor_asof import MODEL
    names=list(MODEL.FEATURES)
    params={
      "contract":"KEIBA_ORIGINAL_ANCHOR_FULL_PRIOR_TRAINED_PARAMS_V1",
      "trained_only_on_years":[2022,2023,2024],
      "variant":"ANCHOR_FULL",
      "features_in_original_order":names,
      "coefficients":{k:0.0 for k in names},
      "standardizer":{k:{"mean":0.0,"std":1.0,"median":0.0} for k in names}
    }
    return bridge,card,odds,params

class GenuineHistoricalFormula(unittest.TestCase):
    def test_all_frozen_rules_and_exact_v3_unknown_p3(self):
        bridge,card,odds,param=mock_input()
        data=historical_full_frozen(bridge,card,odds,param)
        self.assertEqual(data["contract"],CONTRACT)
        self.assertEqual(data["started_horse_count"],17)
        self.assertEqual(data["race_entry_count"],18)
        self.assertEqual(data["original_l2_1_standard_line_gate_count"],50)
        self.assertEqual(data["original_l3_v3_policy"],"REVERSE_P3_RACE_MAX300")
        self.assertEqual(len(data["horses"]),17)
        self.assertEqual(len(data["rankings"]["ai_raw"]),17)
        self.assertEqual(len(data["rankings"]["expected_value"]),17)
        self.assertFalse(data["historical_saved_l21_tickets_used"])
        self.assertFalse(data["historical_saved_2025_oos_scores_used"])
        self.assertFalse(data["is_live_or_purchase_ready"])
        self.assertEqual(data["l3_v3_paper_stake_yen"],
                         sum(x["stake_yen"] for x in data["tickets"]))
        self.assertTrue(all(x["min_p3"] is None for x in data["tickets"]))
        self.assertTrue(all(x["stake_yen"] in (100,200,300)
                            for x in data["tickets"]))
        self.assertEqual(len({(x["bet_type"],x["canonical_selection"]) for x in data["tickets"]}),
                         len(data["tickets"]))
        self.assertAlmostEqual(sum(x["original_anchor_win_probability"]
                                   for x in data["horses"]),1.0)
        print("SYNTHETIC_FIXTURE_ORIGINAL_FINAL_L2_L3_PASS",
              "tickets",data["bet_count"],"yen",data["l3_v3_paper_stake_yen"])

    def test_18_registered_one_scratched_and_canonical_identity(self):
        bridge,card,odds,param=mock_input()
        bridge["horses"][0]["horse_id"]="WRONG_HORSE"
        with self.assertRaisesRegex(ValueError,"BLIND_SCORE_HORSE_ID_VS_BACKFILL_CANONICAL_DRIFT"):
            historical_full_frozen(bridge,card,odds,param)

    def test_missing_final_quote_blocks_no_fake_zero(self):
        bridge,card,odds,param=mock_input()
        odds["odds"]["8"].pop("010203")
        with self.assertRaisesRegex(ValueError,"FINAL_COMBINATION_QUOTE_MISSING"):
            historical_full_frozen(bridge,card,odds,param)

    def test_no_final_as_live_and_no_2026_hindsight(self):
        bridge,card,odds,param=mock_input()
        odds["odds_kind"]="PRE_RACE"
        with self.assertRaisesRegex(ValueError,"NOT_VERIFIED_BACKFILL_HISTORICAL_FINAL_SOURCE"):
            historical_full_frozen(bridge,card,odds,param)

if __name__=="__main__":
    unittest.main()
