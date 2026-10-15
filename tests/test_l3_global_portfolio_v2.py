#!/usr/bin/env python3
"""V2 invariant tests: frozen picks, exact global budget, no outcome leakage."""
import importlib.util
import sys
import unittest
from pathlib import Path

RESEARCH=Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(RESEARCH))
import run_l3_global_portfolio_v2 as x

def row(n, p3, out, bet="TRIFECTA", race="r1", day="2023-01-01"):
    return {
        "year":2023, "race_id":race, "race_date":day,
        "bet_type":bet, "selection":str(n),
        "min_p3":p3, "support_min":out,
        "mean_p3_delta":0.2,"ticket_odds":999.0,
        "hit":int(n%2==0), "payout_yen_per100":1000+n,
        "l175_complete":int(p3 is not None)
    }

class GlobalPortfolio(unittest.TestCase):
    def setUp(self):
        self.ts=[row(1,.12,2),row(2,.3,5,bet="TRIO"),row(3,None,None),row(4,.5,9,bet="QUINELLA")]

    def test_all_policies_full_set_and_budget(self):
        for strategy in x.CORE+x.POSTHOC+x.DIAGNOSTICS:
            for cap in x.CAPS:
                allocations=x.allocate(self.ts,strategy,cap,200*len(self.ts))
                self.assertEqual(set(allocations),set(map(x.ticket_key,self.ts)))
                self.assertEqual(sum(allocations.values()),200*len(self.ts))
                self.assertTrue(all(s%100==0 and 100<=s<=cap for s in allocations.values()))

    def test_day_pool_recovers_single_ticket_races(self):
        two=[row(1,.1,0,race="A"),row(2,.8,7,race="B")]
        a=x.allocate(two,"REVERSE_P3",300,400)
        self.assertEqual(sorted(a.values()),[100,300])

    def test_missing_always_bought(self):
        a=x.allocate(self.ts,"REVERSE_P3",300,800)
        self.assertGreaterEqual(a[x.ticket_key(self.ts[2])],100)

    def test_outcome_and_final_odds_never_enter_policy(self):
        a=x.allocate(self.ts,"P3_OUTSIDER",500,800)
        changed=[dict(t,hit=1,payout_yen_per100=9_999_999,ticket_odds=0.5)
                 for t in self.ts]
        self.assertEqual(a,x.allocate(changed,"P3_OUTSIDER",500,800))

    def test_reject_capital_mismatch(self):
        with self.assertRaisesRegex(ValueError,"capital mismatch"):
            x.allocate(self.ts,"REVERSE_P3",300,700)

    def test_frozen_untouched(self):
        self.assertTrue((x.frozen.ROOT/"contracts/KEIBA_L2_1_FIXED_PURCHASE_V1.json").is_file())

if __name__=="__main__":
    unittest.main()
