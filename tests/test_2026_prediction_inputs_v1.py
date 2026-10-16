#!/usr/bin/env python3
"""Unit tests for the read-only 2026 source/panel guard."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from audit_2026_prediction_inputs_v1 import summarize_history, summarize_panel


def sample_day(rid="202605040611", when="2026-10-10"):
    a = {"horse_id": "2021105898", "horse_number": 1, "entry_status": "STARTED",
         "sex": "牝", "age": 5, "carried_weight": 55.0}
    return {"race": {"race_id": rid, "actual_date": when, "venue_code": "05",
                     "surface": "TURF", "distance_m": 1600, "discipline": "FLAT"},
            "entries": [a],
            "results": [{"horse_id": a["horse_id"], "result_status": "FINISHED",
                         "official_finish_position": 1, "last_3f": 33.4}],
            "laps": [{"lap_no": 1, "time_seconds": 12.1}]}


class PhaseOneAuditTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.base = Path(self.dir.name)

    def history(self, row):
        path = self.base / "history.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def panel(self, row):
        path = self.base / "panel.json"
        path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        return path

    def sample_panel(self):
        return {
            "contract": "KEIBA_UNSCORED_RACE_FEATURE_EXTRACT_V1",
            "race_id": "202605040611", "race_date": "2026-10-17",
            "used_target_labels": False,
            "used_results_payouts_final_odds": False,
            "used_historical_oos_target_predictions": False,
            "actual_source_capture_time_independently_verified": False,
            "rows": [{"_race_id": "202605040611", "_race_date": "2026-10-17",
                      "_horse_id": "2021105898", "race_month": 10}]
        }

    def test_history_valid_but_not_feature_ready(self):
        result = summarize_history([self.history(sample_day())], "2026-10-17")
        self.assertTrue(result["raw_history_valid"])
        self.assertFalse(result["ai_prediction_ready"])
        self.assertFalse(result["core4_649_feature_schema_verified"])
        self.assertEqual(result["counts"]["races"], 1)
        self.assertEqual(result["coverage"]["race.venue_code"]["pct"], 100)

    def test_same_day_results_never_enter_history(self):
        result = summarize_history([self.history(sample_day())], "2026-10-10")
        self.assertFalse(result["raw_history_valid"])
        self.assertTrue(any("STRICTLY_PRIOR" in e for e in result["errors"]))

    def test_future_results_never_enter_history(self):
        result = summarize_history([self.history(sample_day())], "2026-10-09")
        self.assertGreater(result["total_errors"], 0)

    def test_source_horse_mismatch(self):
        row = sample_day()
        row["results"][0]["horse_id"] = "2021105000"
        result = summarize_history([self.history(row)], "2026-10-17")
        self.assertTrue(any("MISMATCH" in e for e in result["errors"]))

    def test_source_duplicate_horse_numbers(self):
        row = sample_day()
        row["entries"].append({**row["entries"][0], "horse_id": "2021105000"})
        row["results"].append({**row["results"][0], "horse_id": "2021105000"})
        result = summarize_history([self.history(row)], "2026-10-17")
        self.assertTrue(any("HORSE_NUMBER" in e for e in result["errors"]))

    def test_safe_unscored_panel_not_incorrectly_called_ready(self):
        doc = self.sample_panel()
        result = summarize_panel(self.panel(doc), "202605040611", "2026-10-17")
        self.assertEqual(result["total_errors"], 0)
        self.assertEqual(result["status"], "PANEL_AUDIT_INCOMPLETE")
        self.assertFalse(result["ai_prediction_ready"])

    def test_postrace_prediction_feature_rejected(self):
        doc = self.sample_panel()
        doc["rows"][0]["final_win_odds"] = 2.1
        result = summarize_panel(self.panel(doc), "202605040611", "2026-10-17")
        self.assertTrue(any("LEAK" in e for e in result["errors"]))

    def test_labels_and_saved_predictions_rejected(self):
        doc = self.sample_panel()
        doc["used_target_labels"] = True
        doc["used_historical_oos_target_predictions"] = True
        result = summarize_panel(self.panel(doc), "202605040611", "2026-10-17")
        self.assertEqual(result["total_errors"], 2)

    def test_fake_649_column_reference_rejected(self):
        doc = self.sample_panel()
        features = [f"f_{i:03d}" for i in range(649)]
        keys = self.base / "fake-core4.json"
        keys.write_text(json.dumps(features))
        with self.assertRaisesRegex(ValueError, "SHA256_MISMATCH"):
            summarize_panel(self.panel(doc), "202605040611", "2026-10-17", keys)


if __name__ == "__main__":
    unittest.main()
