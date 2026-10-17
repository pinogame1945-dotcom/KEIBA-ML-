#!/usr/bin/env python3
"""No real prior or 2026 races. Pure local fail-closed 2026 refit checks."""
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from refit_2026_seven_king_prior_only_v1 import (
    CANDIDATES, frozen_trainer, frozen_params, train_one
)

class Frozen2026DeploymentRefitGateTest(unittest.TestCase):
    def test_seven_original_king_architectures(self):
        self.assertEqual(len(CANDIDATES),7)
        self.assertEqual(sorted(c[1] for c in CANDIDATES.values()),
                         [164,266,423,457,470,504,649])

    def test_original_params_are_extracted_not_rewritten(self):
        params=frozen_params(frozen_trainer())
        self.assertEqual(params["objective"],"binary")
        self.assertEqual(params["n_estimators"],600)
        self.assertEqual(params["learning_rate"],0.03)
        self.assertEqual(params["num_leaves"],31)
        self.assertEqual(params["min_child_samples"],30)
        self.assertEqual(params["subsample"],0.9)
        self.assertEqual(params["subsample_freq"],1)
        self.assertEqual(params["colsample_bytree"],0.9)
        self.assertEqual(params["reg_lambda"],1.0)
        self.assertEqual(params["random_state"],42)
        self.assertTrue(params["deterministic"])
        self.assertTrue(params["force_col_wise"])

    def test_rejects_2026_as_training_data(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            for y in (2025,2026):
                (root/f"snapshot-{y}.jsonl.gz").write_bytes(b"test")
            with self.assertRaisesRegex(ValueError,
                "EXACTLY_2024_AND_2025_PRIOR_SNAPSHOTS_REQUIRED"):
                train_one("core4",root/"meta",root/"schema",
                          [root/"snapshot-2025.jsonl.gz",
                           root/"snapshot-2026.jsonl.gz"],
                          root/"output","user-proceed-2026")

    def test_rejects_unapproved_training_rule(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            for y in (2024,2025):
                (root/f"snapshot-{y}.jsonl.gz").write_bytes(b"test")
            with self.assertRaisesRegex(ValueError,
                "REFIT_PROTOCOL_REFERENCE_MISSING"):
                train_one("core4",root/"meta",root/"schema",
                          [root/"snapshot-2024.jsonl.gz",
                           root/"snapshot-2025.jsonl.gz"],
                          root/"output","")

    def test_original_2025_fold_required_before_training(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            for y in (2024,2025):
                (root/f"snapshot-{y}.jsonl.gz").write_bytes(b"test")
            meta={"split":{"train_start":"2024-01-01",
                           "train_end":"2025-12-31",
                           "valid_start":"2026-01-01",
                           "valid_end":"2026-12-31"},
                  "feature_count":649,"best_iteration":399}
            (root/"meta.json").write_text(json.dumps(meta))
            (root/"schema.json").write_text(json.dumps(
                {"feature_order":[f"x{i}" for i in range(649)]}))
            with self.assertRaisesRegex(ValueError,
                "NOT_THE_FROZEN_2025_OUT_OF_SAMPLE_KING"):
                train_one("core4",root/"meta.json",root/"schema.json",
                          [root/"snapshot-2024.jsonl.gz",
                           root/"snapshot-2025.jsonl.gz"],
                          root/"output","user-proceed-2026")


    def make_prior_test_fixtures(self, root, *, wrong_year=False, duplicate=False):
        from refit_2026_seven_king_prior_only_v1 import frozen_trainer
        trainer=frozen_trainer()
        history=trainer.DEFAULT_FEATURE_CONTRACT["history_windows"]
        small=trainer.normalize_small_sample_policy(
            None, json.loads(trainer.DEFAULT_SMALL_SAMPLE_CONTRACT_PATH.read_text()),
            history)
        for year in (2024,2025):
            row={
                "ml_dataset_version":3,
                "feature_schema_version":8,
                "leakage_policy":"STRICT_PRIOR_DATE_ONLY",
                "prediction_phase":"FINAL",
                "feature_sets":["BASE","OPPONENT","AUTO","ACTOR","TIME_PACE"],
                "history_windows":history,
                "small_sample_policy":small,
                "race_id":str(year)+"05050101",
                "horse_id":"2019010001",
                "features":{"race_date":str(2025 if (year==2024 and wrong_year) else year)+"-05-05"},
                "target":{"is_win":True},
            }
            with gzip.open(root/f"snapshot-{year}.jsonl.gz","wt",encoding="utf-8") as out:
                out.write(json.dumps(row)+"\n")
                if year==2024 and duplicate:
                    out.write(json.dumps(row)+"\n")
        meta={"split":{"train_start":"2023-01-01",
                       "train_end":"2024-12-31",
                       "valid_start":"2025-01-01",
                       "valid_end":"2025-12-31"},
              "feature_count":649,"best_iteration":399}
        (root/"meta.json").write_text(json.dumps(meta))
        (root/"schema.json").write_text(json.dumps(
            {"feature_order":[f"x{i}" for i in range(649)]}))
        return [root/"snapshot-2024.jsonl.gz",
                root/"snapshot-2025.jsonl.gz"]

    def test_rejects_2025_rows_masquerading_as_2024_snapshot(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            files=self.make_prior_test_fixtures(root,wrong_year=True)
            with self.assertRaisesRegex(ValueError,
                "TRAINING_ROW_YEAR_DOES_NOT_MATCH_SNAPSHOT"):
                train_one("core4",root/"meta.json",root/"schema.json",
                          files,root/"output","candidate-protocol-2026")
            self.assertFalse((root/"output").exists())

    def test_rejects_duplicate_horses_within_historical_snapshots(self):
        with TemporaryDirectory() as t:
            root=Path(t)
            files=self.make_prior_test_fixtures(root,duplicate=True)
            with self.assertRaisesRegex(ValueError,
                "PRIOR_TRAINING_HORSE_DUPLICATED"):
                train_one("core4",root/"meta.json",root/"schema.json",
                          files,root/"output","candidate-protocol-2026")
            self.assertFalse((root/"output").exists())

    def test_refit_does_not_relax_research_2026_lock(self):
        c=json.loads(Path("research/l1-raceclass-absorption-walkforward-v1.json").read_text())
        self.assertEqual(c["locked_years"],[2026])
        self.assertEqual(c["validation_years"],[2021,2022,2023,2024,2025])
        self.assertEqual(c["training_window_years"],2)

if __name__=="__main__":
    unittest.main()
