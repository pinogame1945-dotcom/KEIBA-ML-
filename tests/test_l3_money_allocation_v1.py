#!/usr/bin/env python3
"""Tiny no-network guard suite for frozen-ticket L3 money allocation."""
import importlib.util
import unittest
from pathlib import Path

path=Path(__file__).resolve().parents[1]/"research/run_l3_money_allocation_v1.py"
spec=importlib.util.spec_from_file_location("l3",path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def ticket(i, p3, support, odds):
    return dict(year=2023,race_id="race1",bet_type="TRIO",
                selection=str(i),race_date="2023-01-01",
                hit=0,payout_yen_per100=0,l175_complete=int(p3 is not None),
                min_p3=p3,support_min=support,mean_p3_delta=.01,
                ticket_odds=odds)

class Stakes(unittest.TestCase):
    def setUp(self):
        self.ts=[ticket(1,.8,2,20),ticket(2,.2,9,120),ticket(3,None,None,None)]

    def test_budget_and_fixed_membership(self):
        for n in (1,2,3):
            arr=self.ts[:n]
            expect={m.key(t) for t in arr}
            for policy in m.POLICIES:
                budget=(100 if policy=="FLAT100" else 200)*n
                a=m.stakes(policy,arr,budget)
                self.assertEqual(set(a),expect)
                self.assertEqual(sum(a.values()),budget)
                self.assertTrue(all(x>=100 and x%100==0 and x<=300 for x in a.values()))

    def test_prohibit_insufficient_budget(self):
        with self.assertRaisesRegex(ValueError,"INSUFFICIENT_BUDGET"):
            m.stakes("MIN_P3", self.ts,200)

    def test_no_label_leak(self):
        a=m.stakes("MIN_P3",self.ts,600)
        altered=[dict(t,hit=1,payout_yen_per100=1000000) for t in self.ts]
        self.assertEqual(a,m.stakes("MIN_P3",altered,600))

    def test_missing_kept_and_flat(self):
        self.assertEqual(m.stakes("FLAT200",self.ts,600),
                         {m.key(t):200 for t in self.ts})
        self.assertEqual(m.stakes("MIN_P3",self.ts,600)[m.key(self.ts[2])],100)

    def test_frozen_contract(self):
        self.assertEqual(m.ROOT.joinpath("contracts/KEIBA_L2_1_FIXED_PURCHASE_V1.json").is_file(),True)
        self.assertTrue(m.FINAL_ODDS_ONLY.issubset(m.POLICIES))

if __name__=="__main__":
    unittest.main()
