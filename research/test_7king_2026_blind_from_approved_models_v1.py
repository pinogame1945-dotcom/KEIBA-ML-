#!/usr/bin/env python3
"""Safety tests for the 2026 seven-king entry. Never train actual king models."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from score_7king_2026_blind_from_approved_models_v1 import (
    CANDIDATES, model_gate, original_unscored_dataset, run,
)

class ApprovedModelGateTest(unittest.TestCase):
    def make_release(self, temp: Path, *, year=2026, status="APPROVED_2026_BLIND",
                     outcomes=False, fake_model_hash=False):
        folder=temp/"core4"
        folder.mkdir()
        model=folder/"model.txt"
        model.write_text("LIGHTGBM_FAKE_NO_TRAINING",encoding="utf-8")
        schema={
            "contract":"L1_FEATURE_SCHEMA_V1",
            "feature_order":[f"x{i}" for i in range(649)],
            "category_levels":{},
            "prediction_phase":"FINAL",
            "ability_uses_odds":False,
            "ml_dataset_version":3,
            "feature_schema_version":8,
            "leakage_policy":"STRICT_PRIOR_DATE_ONLY",
            "training_years":[2024,2025],
            "model_version":"FAKE_TEST_ONLY",
        }
        schema_file=folder/"schema.json"
        schema_file.write_text(json.dumps(schema),encoding="utf-8")
        meta={
            "contract":"KEIBA_7KING_2026_BLIND_MODEL_RELEASE_V1",
            "deployment_status":status,
            "candidate_name":CANDIDATES["core4"][0],
            "prediction_year":year,
            "training_years":[2024,2025],
            "train_start":"2024-01-01",
            "train_end":"2025-12-31",
            "snapshot_generation":"eb13d4096519167b",
            "2026_outcomes_used":outcomes,
            "ability_uses_odds":False,
            "feature_selection":"none",
            "prediction_phase":"FINAL",
            "training_rule_id":"TEST_DUMMY_NOT_A_POLICY",
            "training_rule_approval_ref":"SYNTHETIC_UNIT_TEST_ONLY",
            "model_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),
            "schema_sha256":hashlib.sha256(schema_file.read_bytes()).hexdigest(),
            "best_iteration":1,
            "model_version":"FAKE_TEST_ONLY",
        }
        if fake_model_hash:
            meta["model_sha256"]="0"*64
        (folder/"model.json").write_text(json.dumps(meta),encoding="utf-8")
        return folder

    def test_2025_released_model_rejected_even_if_status_forged(self):
        with TemporaryDirectory() as t:
            directory=self.make_release(Path(t),year=2025)
            with self.assertRaisesRegex(ValueError,
                                        "PROVENANCE_MISMATCH"):
                model_gate("core4",directory)

    def test_outcome_labelled_2026_model_rejected(self):
        with TemporaryDirectory() as t:
            directory=self.make_release(Path(t),outcomes=True)
            with self.assertRaisesRegex(ValueError,
                                        "RESULTS_OR_MODEL_SPEC_VIOLATION"):
                model_gate("core4",directory)

    def test_unapproved_model_rejected(self):
        with TemporaryDirectory() as t:
            directory=self.make_release(Path(t),status="RESEARCH_ONLY")
            with self.assertRaisesRegex(ValueError,
                                        "NOT_AN_APPROVED"):
                model_gate("core4",directory)

    def test_sha_mismatch_rejected(self):
        with TemporaryDirectory() as t:
            directory=self.make_release(Path(t),fake_model_hash=True)
            with self.assertRaisesRegex(ValueError,
                                        "MODEL_SHA256_MISMATCH"):
                model_gate("core4",directory)

    def test_missing_seven_approved_models_rejected_before_input(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            self.make_release(root)
            with self.assertRaisesRegex(ValueError,
                                        "RELEASE_FILE_MISSING"):
                run(root/"missing-target",root,root/"not-written")
            self.assertFalse((root/"not-written").exists())

    def test_target_outcomes_rejected(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            (root/"2026-10-10").mkdir()
            doc={
                "contract":"KEIBA_BACKFILL_690_PLUS_ORIGINAL_V4_238_UNSCORED_V1",
                "race_id":"202605040301",
                "race_date":"2026-10-10",
                "feature_count":928,
                "readiness":"ASOF_UNVERIFIED_DO_NOT_BET",
                "used_target_results_or_final_odds":False,
                "2026_results_read_or_evaluated":False,
                "starters":1,
                "rows":[{
                    "_race_id":"202605040301",
                    "_horse_id":"2020012345",
                    "_race_date":"2026-10-10",
                    **{f"example_{i}":None for i in range(927)},
                    "target":1,
                }]
            }
            (root/"2026-10-10/202605040301.json").write_text(
                json.dumps(doc),encoding="utf-8")
            with patch.dict("score_7king_2026_blind_from_approved_models_v1.DAY_EXPECTATIONS",
                            {"2026-10-10":(1,1)},clear=True):
                with self.assertRaisesRegex(ValueError,
                                            "TARGET_RESULT_FINAL_ODDS_OR_SCHEMA_LEAK"):
                    original_unscored_dataset(root,root/"flat.jsonl")


if __name__=="__main__":
    unittest.main()
