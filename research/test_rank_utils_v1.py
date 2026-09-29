#!/usr/bin/env python3
import random
import unittest

from rank_utils import RANK_TIE_POLICY, deterministic_ranks, tie_diagnostics


def rank_map(scores, horse_ids):
    ranks=deterministic_ranks(scores,horse_ids)
    return {str(h):int(r) for h,r in zip(horse_ids,ranks)}


class TieSafeRankingTests(unittest.TestCase):
    def test_exact_tie_breaks_by_horse_id(self):
        scores=[0.7,0.7,0.6,0.6]
        horses=["H20","H10","H40","H30"]
        self.assertEqual(
            rank_map(scores,horses),
            {"H20":2,"H10":1,"H40":4,"H30":3},
        )

    def test_row_shuffle_invariance(self):
        base=[
            ("H01",0.91),
            ("H02",0.80),
            ("H03",0.80),
            ("H04",0.72),
            ("H05",0.72),
            ("H06",0.72),
            ("H07",0.51),
            ("H08",0.51),
        ]
        expected=rank_map([s for _,s in base],[h for h,_ in base])
        rng=random.Random(1945)
        for _ in range(100):
            rows=base[:]
            rng.shuffle(rows)
            got=rank_map([s for _,s in rows],[h for h,_ in rows])
            self.assertEqual(got,expected)

    def test_top6_boundary_tie_detected(self):
        horses=[f"H{i:02d}" for i in range(1,9)]
        scores=[0.9,0.8,0.7,0.6,0.5,0.4,0.4,0.2]
        d=tie_diagnostics(scores,horses)
        self.assertEqual(d["policy"],RANK_TIE_POLICY)
        self.assertTrue(d["boundary_tie"]["6"])
        self.assertFalse(d["boundary_tie"]["1"])
        self.assertFalse(d["boundary_tie"]["3"])

    def test_no_tie(self):
        horses=["A","B","C"]
        scores=[0.3,0.2,0.1]
        d=tie_diagnostics(scores,horses)
        self.assertFalse(d["any_tie"])
        self.assertEqual(d["tie_group_count"],0)
        self.assertEqual(d["max_tie_group_size"],1)

    def test_duplicate_horse_rejected(self):
        with self.assertRaises(ValueError):
            deterministic_ranks([0.5,0.5],["A","A"])


if __name__=="__main__":
    unittest.main()
