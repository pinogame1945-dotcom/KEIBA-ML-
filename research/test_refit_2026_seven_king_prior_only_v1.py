#!/usr/bin/env python3
"""No real prior or 2026 races. Pure local fail-closed 2026 refit checks."""
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
                "DEPLOYMENT_REFIT_APPROVAL_REFERENCE_MISSING"):
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

    def test_refit_does_not_relax_research_2026_lock(self):
        c=json.loads(Path("research/l1-raceclass-absorption-walkforward-v1.json").read_text())
        self.assertEqual(c["locked_years"],[2026])
        self.assertEqual(c["validation_years"],[2021,2022,2023,2024,2025])
        self.assertEqual(c["training_window_years"],2)

if __name__=="__main__":
    unittest.main()
