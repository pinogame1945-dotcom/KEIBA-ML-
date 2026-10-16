#!/usr/bin/env python3
"""Free CPU regression tests for all 20 frozen EV-rank pruning masks."""
import sys
import tempfile
import unittest
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import run_l2_ev_rank1_pruning20_same_stage_v1 as study
import run_l2_fixed_ticket_pruning_3040_v1 as frozen
import run_l2_raw_top_ticket_pruning_oos_v1 as prior

class PruningTests(unittest.TestCase):
    def test_exact_frozen_methods_and_3040_original_lines(self):
        self.assertEqual(len(study.POLICIES),21)
        self.assertEqual(study.POLICIES,frozen.POLICIES)
        self.assertEqual(len(study.POLICIES[1:]),20)
        self.assertEqual(len(study.LINE_KEYS),3040)
        self.assertEqual(len(set(study.LINE_KEYS)),3040)
        self.assertEqual(study.RANKS,(1,))

    def test_original_ev_rank1_audited_and_short_fields_skipped(self):
        g=pd.DataFrame([dict(_horse_id=str(i),P__ANCHOR_FULL=1/i,
                            final_win_odds=2,market_rank=i) for i in range(1,6)])
        info=pd.Series(dict(VALUE_EV_horse="1"))
        picks={"ANCHOR_FULL":{"r":"1"}}
        self.assertEqual(study.choose_axis("r",g,info,picks,1),"1")
        self.assertEqual(study.choose_axis("r",g,info,picks,1),picks["ANCHOR_FULL"]["r"])

    def test_original_ev1_3040_historical_baseline(self):
        src="research-results/l2-ticket-odds-band-profit-v1/EV軸_全152買い方_オッズ帯別_年度推移_詳細20区間.csv"
        archive=study.baseline_from_saved(src,1)
        self.assertEqual(len(archive),3040)
        self.assertEqual(set(archive),set(study.LINE_KEYS))
        self.assertTrue(all(set(x)==set(study.YEARS) for x in archive.values()))

    def test_pooled_black_from_three_year_total_not_every_year(self):
        tid=study.LINE_KEYS[0]
        out={}
        for yr,(n,hit,pay) in zip(study.YEARS,[(10,1,200),(10,0,0),(10,1,3100)]):
            row=np.zeros((21,3),dtype=np.int64)
            row[:,0]=n;row[:,1]=hit;row[:,2]=pay
            out[yr]={tid:row}
        selected,originals=study.black_lines(out)
        self.assertTrue(all(tid in selected[k] for k in study.POLICIES))
        self.assertEqual(tuple(originals[tid][0]),(30,2,3300))
        self.assertEqual(sum(originals[tid][0,:]),3332)

    def test_2023_january_training_excluded_for_base_and_all_masks(self):
        self.assertEqual(study.EVAL_START_2023,"2023-02-01")
        self.assertFalse(study.pruning_eval_eligible(2023,"2023-01-28"))
        self.assertFalse(study.pruning_eval_eligible(2023,"2023-01-29"))
        self.assertTrue(study.pruning_eval_eligible(2023,"2023-02-04"))
        self.assertTrue(study.pruning_eval_eligible(2024,"2024-01-06"))
        self.assertTrue(study.pruning_eval_eligible(2025,"2025-01-05"))

    def test_fast_thresholds_equal_frozen_semantics(self):
        card=(15.0,0,("TRIO",7,0,0,1),-.9,-1.2,0.1)
        history=[card]*110
        stats=defaultdict(lambda:[0,0,0.])
        for c in history:frozen.add_learning(stats,c)
        a=prior.fast_prior_thresholds(history,stats)
        b=frozen.create_thresholds(history,stats)
        self.assertEqual(a,b)
        self.assertEqual(frozen.mask(card,stats,a),frozen.mask(card,stats,b))
if __name__=="__main__":unittest.main()
