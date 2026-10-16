#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import run_l2_ev_rank2to6_medium_strict_v1 as analysis
import run_l2_raw_market25_stability_v1 as original

class FrozenStabilityTests(unittest.TestCase):
    def test_frozen_gates_and_stability_thresholds(self):
        self.assertEqual(len(analysis.GATES),17)
        self.assertEqual(analysis.GATES,original.GATES)
        self.assertEqual(analysis.RULES,original.RULES)
        self.assertEqual(analysis.official.CONFIG["STABLE_MEDIUM"],
            dict(n=500,hits=30,per_year_hits=8,max1=.40,max3=.70,minus3_roi=90))
        self.assertEqual(analysis.official.CONFIG["STABLE_STRICT"],
            dict(n=800,hits=50,per_year_hits=12,max1=.30,max3=.55,minus3_roi=100))

    def test_line_key_and_2023_same_evaluation_window(self):
        tid=analysis.ev.LINE_KEYS[0]
        self.assertEqual(analysis.key("|".join(map(str,tid))),tid)
        self.assertFalse(analysis.ev.pruning_eval_eligible(2023,"2023-01-29"))
        self.assertTrue(analysis.ev.pruning_eval_eligible(2023,"2023-02-04"))
        self.assertTrue(analysis.ev.pruning_eval_eligible(2024,"2024-01-06"))

    def test_official_decisions_preserve_strict_subset_medium(self):
        tid=analysis.ev.LINE_KEYS[0]
        selected={tid:(1000,100,200000)}
        years={y:{} for y in analysis.YEARS}
        for y in analysis.YEARS:
            counter=original.new_counter()
            counter["n"]=400
            counter["hits"]=60
            counter["payout"]=60000
            counter["top3"]=[400,500,600]
            years[y][tid,"ALL"]=counter
        chosen,decisions=original.select_stable(years,selected)
        self.assertIn((tid,"ALL"),chosen["STABLE_MEDIUM"])
        self.assertIn((tid,"ALL"),chosen["STABLE_STRICT"])
        self.assertEqual(len(decisions),17)

if __name__=="__main__":unittest.main()
