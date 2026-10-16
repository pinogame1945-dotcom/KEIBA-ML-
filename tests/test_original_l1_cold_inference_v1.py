#!/usr/bin/env python3
"""New-race original L1 cold scoring: no lookup of historical target results."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
from run_original_l1_cold_inference_v1 import score_one, validate, CONTRACT


def original_ability_train():
    output=[]
    for yr in (2023,2024):
        for race in range(1,13):
            rid=f"{yr}0101{race:04d}"
            for h in range(1,9):
                rank=(h+race)%8+1
                output.append({
                    "_race_id":rid,
                    "_horse_id":f"h{yr}-{race}-{h}",
                    "_race_date":f"{yr}-01-01",
                    "_finish":rank,
                    "_is_win":int(rank==1),
                    "_is_top3":int(rank<=3),
                    "recent_form":float(h+race%3),
                    "speed_index":float((h+race)%6),
                    "venue_code":"TOKYO" if race%2 else "KYOTO",
                })
    return pd.DataFrame(output)


def original_prior_teacher():
    output=[]
    for yr in (2021,2022,2023,2024):
        for race in range(1,10):
            rid=f"{yr}0101{race:04d}"
            for h in range(1,9):
                v=(h+race)%8
                output.append({
                    "_race_id":rid,
                    "_horse_id":f"h{yr}-{race}-{h}",
                    "_race_date":f"{yr}-01-01",
                    "_score":float(v)/10,
                    "distill_ai_rank":h,
                    "distill_field_size":8.,
                    "distill_ai_rank01":1-(h-1)/7,
                    "distill_score_z":float(h-4)/2,
                    "distill_score_gap_to_best":float(h-1)/10,
                    "distill_is_ai_top1":int(h==1),
                    "distill_is_ai_top3":int(h<=3),
                    "distill_is_ai_top6":int(h<=6),
                    "_target_over3":int(v>3),
                    "recent_form":float(h+race%3),
                    "speed_index":float(v%6),
                    "venue_code":"TOKYO" if race%2 else "KYOTO",
                })
    return pd.DataFrame(output)


def target():
    return pd.DataFrame([{
        "_race_id":"202501010999",
        "_horse_id":f"unscored-horse-{h}",
        "_race_date":"2025-01-03",
        "recent_form":float(h+1),
        "speed_index":float(h%6),
        "venue_code":"TOKYO",
    } for h in range(1,11)])


class ColdInference(unittest.TestCase):
    def test_training_original_models_not_reading_target_predictions(self):
        out=score_one(target(),original_ability_train(),original_prior_teacher(),2025)
        self.assertEqual(out["contract"],CONTRACT)
        self.assertEqual(out["ability_prior_years"],[2023,2024])
        self.assertEqual(out["risk_prior_oos_teacher_years"],[2021,2022,2023,2024])
        self.assertFalse(out["historical_oos_predictions_used_as_target_input"])
        self.assertFalse(out["target_finish_result_or_final_odds_used"])
        self.assertFalse(out["l2_1_ready"])
        self.assertFalse(out["l3_v3_ready"])
        self.assertFalse(out["prediction_before_bet_verified"])
        self.assertEqual(len(out["horse_scores"]),10)
        self.assertEqual(sorted(r["ai_raw_rank"] for r in out["horse_scores"]),list(range(1,11)))
        self.assertEqual(len({r["horse_id"] for r in out["horse_scores"]}),10)
        self.assertTrue(all(np.isfinite(r["distill_score_z"]) for r in out["horse_scores"]))
        self.assertTrue(all(0<=r["correction_risk"]<=1 for r in out["horse_scores"]))

    def test_target_never_contains_results_cached_predictions_or_final_prices(self):
        for name,value in (
            ("finish_position",3),("target_win",1),("final_win_odds",3.1),
            ("distill_ai_rank",1),("risk",.5),("P__ANCHOR_FULL",.11),
            ("payout_yen",1000),("market_rank",2),("_score",3.),
        ):
            x=target()
            x[name]=value
            with self.subTest(field=name):
                with self.assertRaisesRegex(ValueError,"TARGET_HAS_RESULT_MARKET_OR_CACHED_PREDICTION"):
                    validate(x,original_ability_train(),original_prior_teacher(),2025)

    def test_future_year_label_and_wrong_train_year_refused(self):
        train=original_ability_train()
        bad=train.copy()
        bad["_race_id"]=bad["_race_id"].str.replace("2023","2025",regex=False)
        bad["_race_date"]=bad["_race_date"].str.replace("2023","2025",regex=False)
        with self.assertRaisesRegex(ValueError,"ABILITY_MUST_TRAIN_TWO_PRIOR_YEARS"):
            validate(target(),bad,original_prior_teacher(),2025)
        teacher=original_prior_teacher()
        teacher=teacher.loc[teacher["_race_id"].str[:4].astype(int)>2021]
        with self.assertRaisesRegex(ValueError,"CORRECTION_RISK_MUST_USE_ALL_PRIOR"):
            validate(target(),train,teacher,2025)

    def test_missing_starter_duplicate_horse_and_outcome_absent(self):
        t=target()
        t.loc[2,"_horse_id"]=t.loc[1,"_horse_id"]
        with self.assertRaisesRegex(ValueError,"DUPLICATE_HORSE"):
            validate(t,original_ability_train(),original_prior_teacher(),2025)
        t=target().iloc[:2].copy()
        with self.assertRaisesRegex(ValueError,"INVALID_FULL_STARTER_COUNT"):
            validate(t,original_ability_train(),original_prior_teacher(),2025)

    def test_model_input_feature_schema_drift_rejected(self):
        t=target()
        t["unknown_future_feature"]=0.7
        with self.assertRaisesRegex(ValueError,"ORIGINAL_ABILITY_FEATURE_SCHEMA_DRIFT"):
            score_one(t,original_ability_train(),original_prior_teacher(),2025)

    def test_fake_model_rank_feature_rejected(self):
        t=target()
        t["market_ev_signal"]=0.7
        with self.assertRaisesRegex(ValueError,"TARGET_MODEL_OR_MARKET_FEATURE_LEAK"):
            score_one(t,original_ability_train(),original_prior_teacher(),2025)


if __name__=="__main__":
    unittest.main()
