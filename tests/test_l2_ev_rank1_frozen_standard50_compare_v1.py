#!/usr/bin/env python3
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import run_l2_ev_rank1_frozen_standard50_compare_v1 as target

class FrozenRank1Compare(unittest.TestCase):
    def test_exact_frozen_l21_standard_50_and_no_new_method(self):
        obj,selected=target.frozen_standard("contracts/KEIBA_L2_1_FIXED_PURCHASE_V1.json")
        self.assertEqual(obj["primary_policy"],"STANDARD")
        self.assertEqual(len(selected),9)
        self.assertEqual(sum(map(len,selected.values())),50)
        self.assertEqual(obj["research_baseline"]["STANDARD_ALL"]["tickets"],13479)

    def test_common_period_does_not_take_jan2023(self):
        self.assertFalse(target.ev.pruning_eval_eligible(2023,"2023-01-29"))
        self.assertTrue(target.ev.pruning_eval_eligible(2023,"2023-02-04"))
        self.assertTrue(target.ev.pruning_eval_eligible(2024,"2024-01-06"))

    def test_dedup_tickets_once(self):
        v=target.blank()
        target.count(v,{("TRIO",(1,2,3)):500,("QUINELLA",(1,2)):0})
        self.assertEqual(v,dict(tickets=2,hits=1,payout_yen=500,races_bought=1))

if __name__=="__main__":unittest.main()
