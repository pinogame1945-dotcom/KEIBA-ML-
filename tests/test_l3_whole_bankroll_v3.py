#!/usr/bin/env python3
"""Zero-external-service checks for L3 bankroll accounting and no silent skips."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import run_l3_whole_bankroll_v3 as m

class Bankroll(unittest.TestCase):
    def test_prefund_exact_requirement(self):
        days={"2023-01-01":[100,0,1],
              "2023-01-02":[300,1000,2],
              "2023-01-03":[500,0,4]}
        need,dd=m.minimum_initial_cash(days)
        self.assertEqual(need,400)
        self.assertEqual(dd,500)
        self.assertEqual(m.first_shortage(days,399)["status"],"INSUFFICIENT_BUDGET")
        x=m.first_shortage(days,400)
        self.assertEqual(x["status"],"COMPLETE")
        self.assertEqual(x["final_cash_yen"],500)

    def test_initial_smaller_than_later_big_stake(self):
        days={"2024-01-01":[100,1000,1],
              "2024-01-02":[500,0,2]}
        need,dd=m.minimum_initial_cash(days)
        self.assertEqual(need,100)
        self.assertEqual(m.first_shortage(days,100)["status"],"COMPLETE")

    def test_no_overnight_credit_or_partial_purchase(self):
        days={"2025-03-01":[100,1000,1],
              "2025-03-02":[1200,0,10]}
        b=m.first_shortage(days,100)
        self.assertEqual(b["status"],"INSUFFICIENT_BUDGET")
        self.assertEqual(b["first_unfundable_date"],"2025-03-02")
        self.assertEqual(b["tickets_completed"],1)
        self.assertEqual(b["shortage_yen"],200)
        self.assertEqual(m.minimum_initial_cash(days)[0],300)

    def test_needs_one_full_day_at_start(self):
        days={"2023-01-01":[1500,4000,10]}
        self.assertEqual(m.minimum_initial_cash(days)[0],1500)
        self.assertEqual(m.first_shortage(days,1400)["days_completed"],0)

    def test_empty_dates_and_order(self):
        days={"2023-01-03":[200,100,1],
              "2023-01-01":[100,100,1]}
        self.assertEqual(m.minimum_initial_cash(days)[0],200)
        self.assertEqual(m.first_shortage(days,200)["status"],"COMPLETE")

if __name__=="__main__":
    unittest.main()
