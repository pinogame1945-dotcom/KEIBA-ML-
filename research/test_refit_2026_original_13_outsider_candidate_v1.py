#!/usr/bin/env python3
"""Synthetic safety tests ONLY; do not fit a 2026 Outsider on fake outcomes."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from refit_2026_original_13_outsider_prior_only_candidate_v1 import (
    config,prior_meta_gate,
)
from refit_2026_seven_king_prior_only_v1 import frozen_trainer,frozen_params

class ThirteenOutsiderRefitSafety(unittest.TestCase):
    def make(self,d,iteration=125):
        meta=Path(d)/"old.json"
        schema=Path(d)/"old-schema.json"
        values={
          "feature_count":20,"best_iteration":iteration,
          "prediction_phase":"FINAL","feature_selection":{
              "contract":"L1_TRAIN_ONLY_FEATURE_SELECTION_V1",
              "mode":"none",
              "selected_features":[f"nh_test_{i}" for i in range(20)]},
          "split":{
            "train_start":"2023-01-01","train_end":"2024-12-31",
            "valid_start":"2025-01-01","valid_end":"2025-12-31",
          }}
        meta.write_text(json.dumps(values))
        schema.write_text(json.dumps({
          "prediction_phase":"FINAL","feature_order":[f"nh_test_{i}" for i in range(20)]
        }))
        return meta,schema
    def test_original_13_are_exact_and_2026_sealed(self):
        entries=config()
        self.assertEqual(len(entries),13)
        self.assertEqual(len(set(entries)),13)
        self.assertEqual(set(entries.values(),) if False else
            {k for k in entries if "outsider_" in k},set(entries))
        for v in entries.values():
            self.assertEqual(v["feature_sets"],["BASE"])
    def test_prior_best_iteration_and_schema_accepted_only_with_original_fold(self):
        with TemporaryDirectory() as tmp:
            m,s=self.make(tmp)
            rounds,order,_,_=prior_meta_gate("outsider_daytrend",m,s)
            self.assertEqual(rounds,125)
            self.assertEqual(len(order),20)
    def test_original_selection_report_must_be_none_and_exact(self):
        with TemporaryDirectory() as tmp:
            m,s=self.make(tmp)
            obj=json.loads(m.read_text())
            obj["feature_selection"]["mode"]="train_v1"
            m.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,"ORIGINAL_2025_MODEL_MODE_DRIFT"):
                prior_meta_gate("outsider_daytrend",m,s)
            obj["feature_selection"]["mode"]="none"
            obj["feature_selection"]["selected_features"]=["unknown"]
            m.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,"ORIGINAL_2025_MODEL_MODE_DRIFT"):
                prior_meta_gate("outsider_daytrend",m,s)

    def test_wrong_fold_rejected(self):
        with TemporaryDirectory() as tmp:
            m,s=self.make(tmp)
            a=json.loads(m.read_text())
            a["split"]["valid_end"]="2026-12-31"
            m.write_text(json.dumps(a))
            with self.assertRaisesRegex(ValueError,"2025_OOS_FOLD"):
                prior_meta_gate("outsider_daytrend",m,s)
    def test_invalid_rounds_rejected(self):
        with TemporaryDirectory() as tmp:
            m,s=self.make(tmp,iteration=601)
            with self.assertRaisesRegex(ValueError,"ROUNDS_OR_SCHEMA"):
                prior_meta_gate("outsider_daytrend",m,s)
    def test_original_trainer_params_only(self):
        cfg=frozen_params(frozen_trainer())
        self.assertEqual(cfg["n_estimators"],600)
        self.assertEqual(cfg["learning_rate"],0.03)
        self.assertEqual(cfg["random_state"],42)
        self.assertEqual(cfg["histogram_pool_size"],1024)

if __name__=="__main__":
    unittest.main()
