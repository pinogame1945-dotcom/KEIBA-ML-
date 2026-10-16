#!/usr/bin/env python3
"""Pure aggregation regression tests; no BACKFILL or historical outcomes needed."""
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"research"))
import run_l2_raw_top_odds_boundary_v1 as mod


class TestRawTopOddsBorder(unittest.TestCase):
    def test_original_152_by_20(self):
        self.assertEqual(len(mod.TEMPLATES), 152)
        self.assertEqual(len(mod.BINS), 20)
        self.assertEqual(set(mod.K_VALUES), {2, 3, 5, 7, 10})
        self.assertEqual(len(set(mod.TEMPLATES)), 152)
        self.assertEqual(len(mod.FAMILIES), 10)

    def test_zero_and_positive_negative(self):
        self.assertEqual(mod.calc(Counter())["color"], "NO_TICKETS")
        self.assertEqual(mod.calc(Counter(tickets=1,payout_yen=100))["color"], "BREAK_EVEN")
        self.assertEqual(mod.calc(Counter(tickets=2,payout_yen=500))["color"], "BLACK")
        self.assertEqual(mod.calc(Counter(tickets=2,payout_yen=100))["color"], "RED")

    def test_cumulative_floor_and_reduced_ticket_border(self):
        lines = defaultdict(Counter)
        raw = ("AI_RAW", "TRIO")
        bands = [mod.fixed.bin_name(v) for v in mod.BINS]
        # Keep k10 at 0-1.5 as red; reducing to k2 turns the SAME
        # disjoint price band black (fixture, not historical empirical data).
        for year in (2023, 2024, 2025):
            mod.combine(lines,year,(raw[0],10,raw[1],bands[0]),
                        Counter(tickets=10,hits=1,payout_yen=500,races=1))
            mod.combine(lines,year,(raw[0],2,raw[1],bands[0]),
                        Counter(tickets=2,hits=1,payout_yen=500,races=1))
        dedup=defaultdict(Counter)
        for year in (2023,2024,2025):
            mod.combine(dedup,year,("ALL","ALL"),
                        Counter(tickets=10,hits=1,payout_yen=500,races=1))
            mod.combine(dedup,year,("TRIO",bands[0]),
                        Counter(tickets=10,hits=1,payout_yen=500,races=1))
        b, f, moves, u = mod.outputs(lines,dedup)
        self.assertEqual(len(b), 12160)
        self.assertEqual(len(f), 12160)
        self.assertEqual(len(moves), 150)
        probe = next(x for x in moves if x["partner_order"]=="AI_RAW"
                     and x["family"]=="TRIO" and x["other_horses_k"]==2)
        self.assertEqual(probe["red_to_black_vs_k10"],1)
        self.assertEqual(probe["k_lowest_black_disjoint_band"], 0)
        self.assertIsNone(probe["k10_lowest_black_disjoint_band"])
        self.assertEqual(probe["all_years_black_bands"], 1)
        self.assertEqual(mod.calc(Counter(tickets=10,payout_yen=500))["roi_pct"],50)
        self.assertEqual(mod.calc(Counter(tickets=2,payout_yen=500))["roi_pct"],250)


if __name__ == "__main__":
    unittest.main()
