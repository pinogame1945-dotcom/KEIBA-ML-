"""Single package must reproduce the existing frozen runner bit-for-bit."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "tests"))
import run_github_race_ids_v1 as frozen
from test_github_race_id_integration_v1 import IDS, write_fixture
from keiba_runtime import analyze, analyze_to_json


class UnifiedRaceRuntimeTests(unittest.TestCase):
    def test_two_races_identical_to_existing_frozen_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = write_fixture(temp, IDS)
            result = analyze(list(IDS), Path(temp) / "manifest.json")
            self.assertEqual(result, frozen.run(list(IDS), manifest, temp))
            self.assertEqual(result["l3_policy"], "REVERSE_P3_RACE_MAX300")
            self.assertTrue(all(x["status"] == "PAPER_ONLY_COMPLETE"
                                for x in result["races"]))

    def test_one_command_creates_existing_app_json(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = write_fixture(temp, IDS[:1])
            output = Path(temp) / "complete.json"
            command = [sys.executable, "-m", "keiba_runtime",
                       "--race", IDS[0], "--manifest", str(Path(temp) / "manifest.json"),
                       "--output", str(output)]
            result = subprocess.run(command, cwd=ROOT, text=True,
                                    capture_output=True, check=True)
            self.assertIn("KEIBA_ONE_SHOT", result.stdout)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")),
                             frozen.run([IDS[0]], manifest, temp))

    def test_missing_upstream_is_blocked_without_fake_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            write_fixture(temp, IDS[:1])
            report = analyze_to_json(list(IDS), Path(temp) / "manifest.json",
                                     Path(temp) / "out.json")
            self.assertEqual(report["races"][1]["status"], "BLOCKED")
            self.assertIsNone(report["races"][1]["rankings"])
            self.assertEqual(report["races"][0]["status"],
                             "PAPER_ONLY_PARTIAL_REQUESTED_BATCH")
            self.assertFalse(report["actual_bets_placed"])

    def test_missing_manifest_never_forges_model_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output.json"
            with self.assertRaises(FileNotFoundError):
                analyze_to_json([IDS[0]], Path(temp) / "absent.json", output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
