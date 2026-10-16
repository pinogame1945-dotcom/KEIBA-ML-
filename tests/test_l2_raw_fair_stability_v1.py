#!/usr/bin/env python3
"""Parity with frozen policy quantile/tie semantics, and strict subset."""
import sys
import unittest
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import run_l2_fixed_ticket_pruning_3040_v1 as fixed
import run_l2_raw_selected_stability_fair_v1 as mod
import run_l2_raw_market25_stability_v1 as official

class TestFairScore(unittest.TestCase):
    def test_threshold_and_mask_bit_exact_for_every_mode_and_keep(self):
        history=[]
        stat=defaultdict(lambda:[0,0,0.])
        for fam in ("WIN","EXACTA","TRIO","TRIFECTA"):
            for bz in (0,3,5,10,19):
                # Exercise both >=100 and <100 grouping threshold paths.
                for i in range(124 if bz%2==0 else 58):
                    c=(float(2+i%17), int(i%29==0)*100,
                       (fam,bz,i%3,i%5,i%4),-(i%59)/11,
                       -(i%61)/17, (i%7)/9)
                    history.append(c)
                    fixed.add_learning(stat,c)
        reference=fixed.create_thresholds(history,stat)
        for mode in fixed.MODES:
            for keep in fixed.KEEP:
                thresholds=mod.selected_thresholds(history,stat,mode,keep)
                for card in history[::7]:
                    a=fixed.mask(card,stat,reference)[f"{mode}_{keep}"]
                    b=mod.mask_one(card,stat,thresholds,mode)
                    self.assertEqual(a,b,(mode,keep,card[2]))
    def test_official_two_thresholds(self):
        self.assertEqual(mod.CONFIG,official.CONFIG)
        self.assertEqual(mod.GATES,official.GATES)
        self.assertTrue(mod.CONFIG["STABLE_STRICT"]["max3"] <
                        mod.CONFIG["STABLE_MEDIUM"]["max3"])
        self.assertEqual(mod.START,"2023-02-01")
if __name__=="__main__": unittest.main()
