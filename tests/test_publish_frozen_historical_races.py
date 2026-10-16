#!/usr/bin/env python3
"""Offline tests for repository-only historical JSON publisher."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "historical_publisher", ROOT / "scripts/publish_frozen_historical_races.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def example():
    horses = ["20220001", "20220002", "20220003"]
    return {
        "contract": publisher.INPUT_CONTRACT,
        "status": "HISTORICAL_FINAL_RESEARCH_ONLY",
        "source_phase": "FINAL_RESEARCH_ARCHIVED_NOT_AT_BET_CUTOFF",
        "genuinely_prebet_validated": False,
        "historical_2023_2025_L2_selection_was_hindsight": True,
        "original_models_retrained": False,
        "year_2026_results_read": False,
        "real_money_bets_placed": False,
        "complete_same_day_JRA_schedule_certified": False,
        "rankings_source": "ORIGINAL_FROZEN_OOS_SCORES",
        "tickets_source": "ORIGINAL_FROZEN_L21_13479_PHYSICAL_TICKETS",
        "races": [{
            "race_id": "202507010212", "race_date": "2025-01-06",
            "status": "HISTORICAL_FINAL_ONLY_UNAVAILABLE_AS_PREBET",
            "original_anchor_horse_id": horses[1],
            "l3_v3_policy": "REVERSE_P3_RACE_MAX300",
            "rankings": {
                "ai_raw": [{"rank": i, "horse_id": h,
                            "distill_ai_rank": i} for i, h in enumerate(horses, 1)],
                "ai_prob": [{"rank": i, "horse_id": h, "original_pwin": .2}
                            for i, h in enumerate(horses, 1)],
                "value_ev": [{"rank": i, "horse_id": h, "historical_final_value": 2.2}
                             for i, h in enumerate([horses[1], horses[0], horses[2]], 1)]},
            "original_risk_by_horse_id": dict.fromkeys(horses, .12),
            "original_ability_z_by_horse_id": dict.fromkeys(horses, 1.3),
            "missing_l175_ticket_p3": 1,
            "frozen_physical_ticket_count": 2,
            "l3_v3_total_yen": 400,
            "tickets": [
                {"bet_type": "QUINELLA", "canonical_selection": "1-2",
                 "stake_yen": 300, "min_p3": .10,
                 "p3_missing": False, "historical_price_phase": "FINAL_RESEARCH_ONLY"},
                {"bet_type": "TRIO", "canonical_selection": "1-2-3",
                 "stake_yen": 100, "min_p3": None,
                 "p3_missing": True, "historical_price_phase": "FINAL_RESEARCH_ONLY"}],
        }]
    }


class FakeClient:
    def __init__(self, size=0):
        self.size = size
        self.writes = []
        self.found = {}

    def tree_size(self):
        return self.size

    def existing(self, path):
        return self.found.get(path)

    def put(self, path, blob):
        self.writes.append((path, blob))


class PublishTest(unittest.TestCase):
    def test_replay_publisher_outputs_stable_fixed_url_and_correct_policy(self):
        records = publisher.validate_replay(example())
        self.assertEqual(len(records), 1)
        key, content = records[0]
        self.assertEqual(key, "historical/2025/202507010212.json")
        decoded = json.loads(content)
        self.assertEqual(decoded["scope"], "HISTORICAL_RESEARCH_REPLAY_ONLY")
        self.assertFalse(decoded["purchase_ready"])
        self.assertEqual(decoded["l3_v3"]["total_paper_stake_yen"], 400)
        self.assertEqual(decoded["l3_v3"]["policy"], "REVERSE_P3_RACE_MAX300")
        self.assertEqual(decoded["l2_1"]["frozen_commit_sha"], publisher.FREEZE_SHA)
        self.assertEqual(decoded["l3_v3"]["missing_l17_p3_ticket_count"], 1)
        self.assertEqual(len(decoded["rankings"]["ai_raw"]), 3)
        fake = FakeClient()
        links = publisher.publish(records, fake)
        self.assertEqual(fake.writes, records)
        self.assertIn("202507010212.json", links[0])

    def test_no_rewrite_even_if_same_size_but_different_blob(self):
        records = publisher.validate_replay(example())
        path, raw = records[0]
        sha = hashlib.sha1(("blob " + str(len(raw)) + "\0").encode() + raw).hexdigest()
        fake = FakeClient(len(raw))
        fake.found[path] = {"type": "file", "size": len(raw), "sha": sha}
        publisher.publish(records, fake)
        self.assertEqual(fake.writes, [])
        fake.found[path]["sha"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "IMMUTABLE_SHA_DRIFT"):
            publisher.publish(records, fake)
        self.assertEqual(fake.writes, [])

    def test_budget_and_total_free_storage_fail_before_writing(self):
        records = publisher.validate_replay(example())
        fake = FakeClient(publisher.MAX_PUBLISHED_HISTORY_BYTES)
        with self.assertRaisesRegex(ValueError, "FREE_GIT_STORAGE_GUARD"):
            publisher.publish(records, fake)
        self.assertFalse(fake.writes)

    def test_no_live_predicted_status_or_retro_results(self):
        sample = example()
        sample["source_phase"] = "PRE_RACE"
        with self.assertRaisesRegex(ValueError, "UNTRUSTED_REPLAY_source_phase"):
            publisher.validate_replay(sample)
        sample = example()
        sample["year_2026_results_read"] = True
        with self.assertRaisesRegex(ValueError, "UNTRUSTED_REPLAY_year_2026_results_read"):
            publisher.validate_replay(sample)
        sample = example()
        sample["races"][0]["race_id"] = "202607010212"
        with self.assertRaisesRegex(ValueError, "SEALED_2026"):
            publisher.validate_replay(sample)

    def test_frozen_ticket_invariants_and_unsafe_stakes(self):
        sample = example()
        sample["races"][0]["tickets"][0]["stake_yen"] = 500
        with self.assertRaisesRegex(ValueError, "INVALID_V3_STAKE_UNIT"):
            publisher.validate_replay(sample)
        sample = example()
        sample["races"][0]["l3_v3_total_yen"] = 900
        with self.assertRaisesRegex(ValueError, "V3_STAKE_CONSERVATION_FAIL"):
            publisher.validate_replay(sample)
        sample = example()
        sample["races"][0]["status"] = "HISTORICAL_V3_SAME_DAY_NOT_FUNDED"
        with self.assertRaisesRegex(ValueError, "INSUFFICIENT_BUDGET_MUST_BLOCK_ALL"):
            publisher.validate_replay(sample)
        sample = example()
        sample["races"][0]["tickets"].append(copy.deepcopy(sample["races"][0]["tickets"][0]))
        with self.assertRaisesRegex(ValueError, "DUPLICATE_FROZEN_PHYSICAL_TICKET"):
            publisher.validate_replay(sample)

    def test_multi_race_batch_one_record_per_unique_race(self):
        sample = example()
        other = copy.deepcopy(sample["races"][0])
        other["race_id"] = "202506010102"
        sample["races"].append(other)
        output = publisher.validate_replay(sample)
        self.assertEqual(len(output), 2)
        self.assertNotEqual(output[0][0], output[1][0])
        sample["races"].append(copy.deepcopy(other))
        with self.assertRaisesRegex(ValueError, "DUPLICATE_PUBLISHED_RACE"):
            publisher.validate_replay(sample)


if __name__ == "__main__":
    unittest.main()
