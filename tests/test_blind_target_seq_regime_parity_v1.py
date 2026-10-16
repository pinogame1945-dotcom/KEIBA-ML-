#!/usr/bin/env python3
"""Numerical parity against the frozen ORIGINAL state + regime code.

Target outcomes appear only in this synthetic comparison fixture passed
to the reference function; production tested function receives *no*
target results and only source races before the selected target date.
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
from run_l1_state_transition_v1 import read_compact_state_source,build_state_features
from run_l1_residual_regime_full_pairwise_v1 import build_regime_features
from compute_blind_target_seq_regime_v1 import compute,prior_events

TARGET="202505050812"
DAY="2025-11-30"

def hrow(yr,day,no,horse,finish,rid=None):
    rid=rid or f"{yr}0505{no:04d}"
    return {
      "race_id":rid,"horse_id":f"2020000{horse:03d}",
      "race_date":day,
      "features":{"race_date":day,"distance_m":1600,"body_weight":470+horse,
                  "carried_weight":56,"field_size":17 if rid==TARGET else 3,
                  "network_expected_pairwise_score":(4-horse)/4,
                  "horse_number":horse},
      "target":{"finish_position":finish,
                "finish_time_ms":96000+1000*horse+no*2,
                "last_3f":35+horse*.2,"margin_lengths":horse*.15},
    }

def generate(tmp):
    root=Path(tmp)
    paths={}
    all_prior=[]
    for year in range(2019,2026):
        records=[]
        for no,day in ((1,f"{year}-01-10"),(2,f"{year}-06-12")):
            for horse in range(1,4):
                records.append(hrow(year,day,no,horse,horse))
        if year==2025:
            for horse in range(1,18):
                records.append(hrow(year,DAY,812,horse,horse,rid=TARGET))
        path=root/f"{year}.jsonl"
        path.write_text("".join(json.dumps(row)+"\n" for row in records))
        paths[year]=path
        all_prior.extend(records[:-17] if year==2025 else records)
    frames=[read_compact_state_source(p,y)
            for y,p in paths.items()]
    prior=pd.concat(frames,ignore_index=True)
    prior=prior.loc[(prior["_race_id"]!=TARGET)&(prior["_race_date"]<DAY)]
    target={"contract":"KEIBA_UNSCORED_RACE_FEATURE_EXTRACT_V1",
            "race_id":TARGET,
            "used_target_labels":False,"used_results_payouts_final_odds":False,
            "used_historical_oos_target_predictions":False,
            "rows":[{"_race_id":TARGET,"_horse_id":f"2020000{horse:03d}",
                    "_race_date":DAY} for horse in range(1,18)]}
    return paths,prior,target


class OriginalSeqRegimeParity(unittest.TestCase):
    def test_all_source_lag_regime_formulas_identical_on_unseen_target(self):
        with tempfile.TemporaryDirectory() as temp:
            paths,prior,target=generate(temp)
            actual=compute(prior,target)
            exact,_,_,_,_=build_state_features(paths)
            regimes,_=build_regime_features(exact)
            original=exact[2025].merge(regimes[2025],
                on=["_race_id","_horse_id"],validate="one_to_one")
            original=original[original["_race_id"]==TARGET].set_index("_horse_id")
            self.assertEqual(actual["target_horses"],17)
            self.assertEqual(actual["regime_feature_count"],66)
            for horse in actual["horse_features"]:
                before=original.loc[horse["horse_id"]]
                for key,val in horse["features"].items():
                    with self.subTest(horse=horse["horse_id"],key=key):
                        expected=before[key]
                        if val is None:
                            self.assertTrue(pd.isna(expected),
                                            f"{key}: expected {expected}")
                        else:
                            self.assertTrue(np.isclose(float(val),float(expected),
                                atol=1e-4,rtol=1e-5,equal_nan=True),
                                f"{key}: got {val}, source {expected}")
            self.assertFalse(actual["target_outcome_used"])
            self.assertTrue(actual["strict_prior_date_only"])

    def test_selected_or_later_race_leak_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,prior,target=generate(tmp)
            future=prior.iloc[0].copy()
            future["_race_date"]=DAY
            with self.assertRaisesRegex(ValueError,"HISTORY_NOT_STRICT_PRIOR"):
                compute(pd.concat([prior,pd.DataFrame([future])]),target)
            target["used_target_labels"]=True
            with self.assertRaisesRegex(ValueError,"TARGET_RESULTS"):
                compute(prior,target)

    def test_duplicate_same_horse_same_day_conflict_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,prior,_=generate(tmp)
            copy=prior.iloc[0].copy()
            copy["_race_id"]="202005050899"
            copy["_finish"]=9
            with self.assertRaisesRegex(ValueError,"SAME_DAY_STATE_ROWS_CONFLICT"):
                prior_events(pd.concat([prior,pd.DataFrame([copy])]),DAY)


if __name__=="__main__":
    unittest.main()
