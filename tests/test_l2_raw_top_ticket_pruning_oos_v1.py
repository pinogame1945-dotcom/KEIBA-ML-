#!/usr/bin/env python3
"""No historical data: regression checks for original scoring and ROI boundaries."""
import sys
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import run_l2_raw_top_ticket_pruning_oos_v1 as t


class TestPrune(unittest.TestCase):
    def test_same_threshold_as_original(self):
        history=[]
        stat=defaultdict(lambda:[0,0,0.])
        for kind in ("WIN","QUINELLA","TRIO","TRIFECTA"):
            for band in (0,3,11):
                for i in range(121):
                    feature=(kind,band,i%4,i%5,(i*3)%5)
                    card=(float(2+i%13),100*(i%13==0),feature,
                          -.1-i/1000, -.2-(i%33)/100,((i%7)-3)/8)
                    history.append(card)
                    t.fixed.add_learning(stat,card)
        a=t.fixed.create_thresholds(history,stat)
        b=t.fast_prior_thresholds(history,stat)
        self.assertEqual(a,b)
        self.assertEqual(len(t.POLICIES),21)
        self.assertEqual(len(t.TEMPLATES),152)

    def test_roi_contract(self):
        self.assertEqual(t.calc(0,0,0)["color"],"NO_TICKETS")
        self.assertEqual(t.calc(100,5,9999)["color"],"RED")
        self.assertEqual(t.calc(100,5,10000)["color"],"BREAK_EVEN")
        self.assertEqual(t.calc(100,5,15000)["color"],"BLACK")
        self.assertEqual(t.calc(100,5,15000)["roi_pct"],150)

    def test_policy_boundaries(self):
        import numpy as np
        old={}
        for year in ("2024","2025"):
            for order,k,family in t.TEMPLATES:
                for band in t.BANDS:
                    old[(year,order,k,family,band)]=(0,0,0)
        sample=("AI_RAW",2,"WIDE",t.BANDS[0])
        for year in ("2024","2025"):
            old[(year,*sample)]=(10,1,500)  # 50% baseline, red
        results={}
        for year in (2024,2025):
            arr=np.zeros((21,3),dtype=np.int64)
            arr[0]=[10,1,500]
            for z in range(1,21):
                arr[z]=[2,1,500]  # 250% after pruning
            line=defaultdict(lambda:np.zeros((21,3),dtype=np.int64))
            line[sample]=arr
            results[year]=(line,arr,None,0,0)
        bands,pooled,borders,policies=t.analyze_baseline_and_pruning(results,old)
        self.assertEqual(len(bands),2*21*3040)
        self.assertEqual(len(pooled),21*3040)
        self.assertEqual(len(borders),21*152)
        self.assertEqual(len(policies),21)
        base=next(p for p in policies if p["policy"]=="BASE")
        adjusted=next(p for p in policies if p["policy"]=="PROB_50")
        self.assertEqual(base["baseline_black_lines"],0)
        self.assertEqual(adjusted["red_to_black_lines"],1)
        self.assertEqual(adjusted["physical_tickets"],4)
        self.assertEqual(base["physical_tickets"],20)
        raw=next(r for r in borders if r["policy"]=="PROB_50"
                 and r["partner_order"]=="AI_RAW" and
                 r["other_horses_k"]==2 and r["family"]=="WIDE")
        self.assertEqual(raw["after_first_black_band"],0)
        self.assertIsNone(raw["before_first_black_band"])


if __name__=="__main__":
    unittest.main()
