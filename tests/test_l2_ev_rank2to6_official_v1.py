#!/usr/bin/env python3
"""Regression checks for read-only EV-ranks 2-6 official L2.1 research."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))

import pandas as pd
import run_l2_ev_rank2to6_official_v1 as study
import run_l2_raw_market25_stability_v1 as official

class RankTests(unittest.TestCase):
    def test_ties_are_deterministic_and_score_descends(self):
        g=pd.DataFrame([
            {"_horse_id":"10","P__ANCHOR_FULL":.20,"final_win_odds":5,"market_rank":3},
            {"_horse_id":"02","P__ANCHOR_FULL":.25,"final_win_odds":4,"market_rank":2},
            {"_horse_id":"04","P__ANCHOR_FULL":.10,"final_win_odds":9,"market_rank":1},
            {"_horse_id":"03","P__ANCHOR_FULL":.05,"final_win_odds":4,"market_rank":5},
            {"_horse_id":"05","P__ANCHOR_FULL":.04,"final_win_odds":4,"market_rank":4},
            {"_horse_id":"06","P__ANCHOR_FULL":.03,"final_win_odds":4,"market_rank":6}])
        self.assertEqual(study.order_ev(g),["02","10","04","03","05","06"])
    def test_small_field_rank6_is_unavailable_not_failure(self):
        # Rank 6 does not exist in five-horse races; ranks 2-5 remain valid.
        g=pd.DataFrame([
            {"_horse_id":str(n),"P__ANCHOR_FULL":1.0/n,
             "final_win_odds":2.0,"market_rank":n}
            for n in range(1,6)])
        seq=study.order_ev(g)
        self.assertEqual(seq,["1","2","3","4","5"])
        self.assertEqual([rank for rank in study.RANKS if rank<=len(seq)],
                         [2,3,4,5])
    def test_top_three_rewards_and_merging(self):
        a=study.counter();b=study.counter()
        for r in [0,100,800,1200]:official.update_counter(a,r)
        for r in [900,0,20]:official.update_counter(b,r)
        study.add(a,b)
        self.assertEqual((a["n"],a["hits"],a["payout"]),(7,5,3020))
        self.assertEqual(sorted(a["top3"],reverse=True),[1200,900,800])
    def test_frozen_stability_values_and_scope(self):
        self.assertEqual(study.RANKS,(2,3,4,5,6))
        self.assertEqual(study.POLICIES,official.RULES)
        self.assertEqual(study.GATES,official.GATES)
        self.assertEqual(official.CONFIG["STABLE_MEDIUM"],
                         dict(n=500,hits=30,per_year_hits=8,max1=.4,max3=.7,minus3_roi=90))
        self.assertEqual(official.CONFIG["STABLE_STRICT"],
                         dict(n=800,hits=50,per_year_hits=12,max1=.3,max3=.55,minus3_roi=100))
    def test_empty_portfolios_are_zero_not_fake_roi(self):
        selected={rank:{p:set() for p in study.POLICIES} for rank in study.RANKS}
        rows=study.portfolio_rows({},selected)
        self.assertEqual(len(rows),72)
        self.assertTrue(all(x["tickets"]==0 and x["roi_pct"] is None for x in rows))
if __name__=="__main__":unittest.main()
