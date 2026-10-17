"""Tests only the shape of BACKFILL feature receipt, NOT feature numerical parity."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

P = Path(__file__).resolve().parents[1] / "research" / "receive_backfill_690_v1.py"
spec = importlib.util.spec_from_file_location("receiver", P)
receiver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receiver)
RID = "202605040301"

def panel():
    bias = ["daybias_" + str(i) for i in range(13)]
    bias += ["horse_front_score", "horse_inner_score",
             "fit_front_mean", "fit_front_ewma", "fit_inner_mean", "fit_inner_ewma"]
    speed = ["regime_speed_resid_" + str(i) for i in range(22)]
    features = {k: None for k in (
        ["synthetic_core_" + str(i) for i in range(649)] + bias + speed)}
    row = {"_race_id": RID, "_horse_id": "2024101234", "_race_date": "2026-10-10",
           **features}
    return {"contract": "KEIBA_BACKFILL_ORIGINAL_L1_690_INPUT_V1",
            "race_id": RID, "race_date": "2026-10-10", "starters": 1,
            "feature_count": 690, "used_target_results_or_final_odds": False,
            "readiness": "ASOF_UNVERIFIED_DO_NOT_BET", "rows": [row]}


class TestReceipt(unittest.TestCase):
    def setUp(self):
        self.fixture = panel()
        self.patcher = patch.object(receiver, "CORE4_SHA", receiver.schema_sha(
            [k for k in self.fixture["rows"][0]
             if k.startswith("synthetic_core_")]))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_published_delivery_contract_is_accepted(self):
        self.fixture["contract"] = "KEIBA_BACKFILL_TO_ML_690_UNSCORED_V1"
        rid, date, count, columns = receiver.check_document(
            self.fixture, RID + ".json"
        )
        self.assertEqual((rid, date, count, len(columns)),
                         (RID, "2026-10-10", 1, 690))

    def test_good_single_race_stays_research_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "raw"; src.mkdir()
            (src / (RID + ".json")).write_text(json.dumps(self.fixture))
            rep = receiver.receive(src, root / "received", 1, 1)
            self.assertEqual((rep["races"], rep["horses"]), (1, 1))
            self.assertFalse(rep["numeric_original_parity_checked"])
            self.assertFalse(rep["model_executed"])
            self.assertTrue((root / "received" / (RID + ".json")).is_file())

    def test_unknown_or_missing_real_source_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "SOURCE_DIRECTORY_NOT_FOUND"):
                receiver.receive(Path(tmp) / "absent", Path(tmp) / "received", None, None)

    def test_no_target_results(self):
        self.fixture["used_target_results_or_final_odds"] = True
        with self.assertRaisesRegex(ValueError, "OUTCOME_LEAK_FLAG"):
            receiver.check_document(self.fixture, RID + ".json")

    def test_malformed_horse_columns(self):
        self.fixture["rows"][0].pop("synthetic_core_0")
        with self.assertRaisesRegex(ValueError, "NOT_690"):
            receiver.check_document(self.fixture, RID + ".json")

    def test_missing_attestation_is_blocked(self):
        self.fixture.pop("used_target_results_or_final_odds")
        with self.assertRaisesRegex(ValueError, "ATTESTATION"):
            receiver.check_document(self.fixture, RID + ".json")

    def test_nan_is_blocked(self):
        self.fixture["rows"][0]["synthetic_core_1"] = float("nan")
        with self.assertRaisesRegex(ValueError, "NON_FINITE"):
            receiver.check_document(self.fixture, RID + ".json")


if __name__ == "__main__":
    unittest.main()
