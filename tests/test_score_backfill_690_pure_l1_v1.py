"""Pretrained model recipe must never be modified: import/fixture input only."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import unittest
import tempfile
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research"))
import score_backfill_690_pure_l1_v1 as score
import phase3_20261010_original_l1_24_cold_v1 as frozen

class TestOriginalPureAbility(unittest.TestCase):
    def test_original_model_seed_and_fit_years_locked(self):
        self.assertEqual(score.TRAIN_MODEL_SEED,frozen.TRAIN_MODEL_SEED)
        self.assertEqual(score.FIT_YEARS,(2024,2025))
        self.assertEqual(score.N_FEATURES,690)

    def test_original_frozen_context_used(self):
        self.assertIs(score.prepare,frozen.prepare)
        self.assertIs(score.model_common,frozen.model_common)
        self.assertIs(score.add_ai_context,frozen.add_ai_context)
        self.assertIs(score.bias_cols,frozen.bias_cols)

    def test_no_source_data_means_no_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,"DELIVERED_DATE_NOT_RECEIVED"):
                score.load_delivered(folder)

if __name__=="__main__":
    unittest.main()
