#!/usr/bin/env python3
"""Offline regression of operational V3 against historical V2 stake algorithm."""
import copy
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
v3 = importlib.import_module("l3_v3_operational")
v2 = importlib.import_module("run_l3_global_portfolio_v2")

RACE = "202507010212"
DAY = "2025-01-06"
QUOTE = "2025-01-06T13:00:00+09:00"
CUTOFF = "2025-01-06T14:25:00+09:00"


def make_race(race_id=RACE, missing=True, verified=False):
    lines = [
        ("QUINELLA", [1, 2], 17.2),
        ("TRIO", [2, 3, 5], 18.3),
        ("TRIFECTA", [4, 3, 5], 240.8),
        ("TRIO", [3, 5, 7], 33.2),
        ("QUINELLA", [5, 6], 22.6),
    ]
    tickets = [
        {"race_id": race_id, "bet_type": bt,
         "canonical_selection": "-".join(map(str, ns)),
         "selection_numbers": ns,
         "observed_at": QUOTE,
         "decimal_odds": price,
         "baseline_stake_yen": 100,
         "matched_line_gates": [{"line": "AI_RAW|3|QUINELLA|15-20", "gate": "ALL"}]}
        for bt, ns, price in lines
    ]
    horses = [
        {"horse_number": n, "horse_id": "horse" + str(n),
         "p3": ({1:.10, 2:.25, 3:.60, 4:.10, 5:.90, 6:.15, 7:.60}[n]
                if n != 6 or not missing else None)}
        for n in range(1, 8)
    ]
    l2 = {"contract": v3.L21, "frozen_policy": "KEIBA_L2_1_FIXED_PURCHASE_V1",
          "frozen_version": "2.1.0", "policy": "STANDARD", "race_id": race_id,
          "status": ("PREBET_QUOTE_VALIDATED_VERIFIED" if verified
                     else "PAPER_ONLY_PREBET_QUOTE_VALIDATED"),
          "source_authenticity_independently_verified": verified,
          "historical_final_odds_used": False,
          "quote_observed_at": QUOTE, "bet_cutoff_at": CUTOFF,
          "physical_ticket_count": len(tickets),
          "baseline_total_stake_yen": 100 * len(tickets), "tickets": tickets}
    signal = {"contract": v3.SIGNALS, "race_id": race_id,
              "source": "L17_KING_OUTSIDER_PAST_ONLY_WF_P3", "model_version": "synthetic-test-1",
              "coverage_scope": "TOP10_OTHER_UNKNOWN",
              "observed_before_bet": True, "source_independently_verified": verified,
              "observed_at": QUOTE, "horses": horses}
    return {"race_date": DAY, "l2_1": l2, "horse_p3": signal}


def make_input(*races, per_race_budget=1000, cash=5000):
    return {"contract": v3.CONFIG, "policy": v3.POLICY,
            "races": list(races), "budget": {
                "race_budget_yen": {r["l2_1"]["race_id"]: per_race_budget
                                    for r in races},
                "max_race_exposure_yen": 10000,
                "available_cash_by_date_yen": {DAY: cash},
                "already_reserved_same_day_yen": {DAY: 0},
                "max_day_exposure_yen": 10000}}


class L3Operational(unittest.TestCase):
    def test_exact_v2_reverse_p3_race_300_stakes(self):
        src = make_race()
        result = v3.run(make_input(src))
        self.assertEqual(result["status"], "BLOCKED_OR_PAPER_ONLY")
        self.assertEqual(result["races"][0]["status"], "PAPER_ONLY_SOURCE_OR_P3_UNVERIFIED")
        self.assertEqual(result["races"][0]["portfolio_total_yen"], 1000)
        got = {(t["bet_type"], t["canonical_selection"]): t["stake_yen"]
               for t in result["races"][0]["tickets"]}
        rows = [
            {"year": 2025, "race_id": RACE, "bet_type": t["bet_type"],
             "selection": t["canonical_selection"], "min_p3": min_p3,
             "support_min": None, "mean_p3_delta": None, "ticket_odds": None}
            for t, min_p3 in v3._validate_race(src)[2]
        ]
        expected = v2.allocate(rows, "REVERSE_P3", 300, 200 * len(rows))
        expect_by_type = {(k[2], k[3]): amount for k, amount in expected.items()}
        self.assertEqual(got, expect_by_type)
        self.assertEqual(len(got), 5)
        self.assertEqual(got[("QUINELLA", "5-6")], 100)
        self.assertTrue(all(n in (100, 200, 300) for n in got.values()))

    def test_budget_caps_and_conservation(self):
        src = make_race(missing=False)
        result = v3.run(make_input(src, per_race_budget=700))
        obj = result["races"][0]
        self.assertEqual(obj["portfolio_total_yen"], 700)
        self.assertEqual(obj["required_ticket_count"], 5)
        self.assertEqual(obj["ticket_count"], 5)
        self.assertFalse(result["actual_bets_placed"])
        self.assertEqual(result["daily_bankroll"][0]["same_day_payouts_credited_yen"], 0)
        self.assertEqual(
            result["daily_bankroll"][0]["cash_after_reservation_yen"], 4300)
        result = v3.run(make_input(src, per_race_budget=3000))
        self.assertEqual(result["races"][0]["portfolio_total_yen"], 1000)

    def test_race_budget_insufficient_never_prunes(self):
        result = v3.run(make_input(make_race(), per_race_budget=400))
        self.assertEqual(result["daily_bankroll"][0]["status"], "INSUFFICIENT_RACE_BUDGET")
        self.assertEqual(result["races"][0]["status"], "INSUFFICIENT_RACE_BUDGET")
        self.assertEqual(result["races"][0]["tickets"], [])

    def test_day_prefunding_all_or_none_across_multiple_races(self):
        r1 = make_race()
        r2 = make_race("202507010211", missing=False)
        result = v3.run(make_input(r1, r2, cash=1900))
        self.assertEqual(result["daily_bankroll"][0]["status"], "INSUFFICIENT_BUDGET")
        self.assertEqual(result["daily_bankroll"][0]["shortage_yen"], 100)
        self.assertTrue(all(r["tickets"] == [] for r in result["races"]))
        ok = v3.run(make_input(r1, r2, cash=2000))
        self.assertEqual(ok["daily_bankroll"][0]["status"], "FUNDED")
        self.assertEqual(ok["daily_bankroll"][0]["cash_after_reservation_yen"], 0)
        self.assertTrue(all(r["ticket_count"] == 5 for r in ok["races"]))

    def test_same_day_reserved_cash_and_day_exposure(self):
        src = make_race()
        x = make_input(src, cash=1200)
        x["budget"]["already_reserved_same_day_yen"][DAY] = 300
        self.assertEqual(v3.run(x)["daily_bankroll"][0]["status"], "INSUFFICIENT_BUDGET")
        x["budget"]["already_reserved_same_day_yen"][DAY] = 0
        x["budget"]["max_day_exposure_yen"] = 900
        self.assertEqual(v3.run(x)["daily_bankroll"][0]["status"], "DAY_EXPOSURE_CAP_EXCEEDED")

    def test_historical_replay_never_claims_live_even_with_boolean_flags(self):
        ok = v3.run(make_input(make_race(missing=False, verified=True)))
        self.assertEqual(ok["status"], "BLOCKED_OR_PAPER_ONLY")
        self.assertEqual(ok["races"][0]["status"], "PAPER_ONLY_SOURCE_OR_P3_UNVERIFIED")
        self.assertEqual(ok["races"][0]["portfolio_total_yen"], 1000)
        missing = v3.run(make_input(make_race(missing=True, verified=True)))
        self.assertEqual(missing["races"][0]["status"], "PAPER_ONLY_SOURCE_OR_P3_UNVERIFIED")

    def test_drift_guards(self):
        baseline = make_input(make_race())
        scenarios = [
            (lambda p: p["races"][0]["l2_1"].update(historical_final_odds_used=True),
             "HISTORICAL_FINAL_ODDS"),
            (lambda p: p["races"][0]["l2_1"]["tickets"].append(
                copy.deepcopy(p["races"][0]["l2_1"]["tickets"][0])),
             "L2_1_TICKET_COUNT_DRIFT"),
            (lambda p: p["races"][0]["horse_p3"]["horses"][0].update(p3=1.1),
             "INVALID_L175_HORSE_P3"),
            (lambda p: p["races"][0]["horse_p3"].update(
                observed_at="2025-01-06T14:26:00+09:00"),
             "L175_P3_NOT_BEFORE_BET_CUTOFF"),
            (lambda p: p["races"][0]["l2_1"]["tickets"][0].update(
                canonical_selection="2-1"), "SELECTION_ID_DRIFT"),
            (lambda p: p["races"][0]["horse_p3"]["horses"].pop(),
             "L175_P3_TICKET_HORSE_COVERAGE_MISSING"),
            (lambda p: p["budget"]["race_budget_yen"].update({RACE: 155}),
             "INVALID_RACE_BUDGET"),
            (lambda p: p["races"][0].update(race_date="2025-01-07"),
             "BET_CUTOFF_RACE_DATE_DRIFT"),
        ]
        for mutate, expected in scenarios:
            p = copy.deepcopy(baseline)
            mutate(p)
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    v3.run(p)

    def test_randomized_exact_v2_differential_parity(self):
        import itertools
        import random
        random_generator = random.Random(20261016)
        combinations = list(itertools.permutations(range(1, 19), 3))
        for n in (1, 2, 3, 5, 8, 16, 30, 60, 120):
            samples = []
            for i in range(n):
                nums = combinations[i]
                row = {"bet_type": "TRIFECTA", "canonical_selection":
                       "-".join(map(str, nums))}
                p3 = random_generator.choice([None, .05, .05, .13, .2, .4, .4, .8])
                samples.append((row, p3))
            got = v3.reverse_p3_stakes(samples, 200 * n)
            legacy = [
                {"year": 2025, "race_id": RACE, "bet_type": row["bet_type"],
                 "selection": row["canonical_selection"], "min_p3": p3,
                 "support_min": None, "mean_p3_delta": None, "ticket_odds": None}
                for row, p3 in samples
            ]
            before = v2.allocate(legacy, "REVERSE_P3", 300, 200 * n)
            reference = {(key[2], key[3]): amount for key, amount in before.items()}
            with self.subTest(n=n):
                self.assertEqual(got, reference)

    def test_command_line_generates_json(self):
        import json
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            input_path = path / "github-produced.json"
            output_path = path / "l3-v3.json"
            input_path.write_text(json.dumps(make_input(make_race())), encoding="utf-8")
            done = subprocess.run(
                [sys.executable, str(ROOT / "research/l3_v3_operational.py"),
                 "--input", str(input_path), "--output", str(output_path)],
                capture_output=True, text=True, check=False)
            self.assertEqual(done.returncode, 0, done.stderr)
            value = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(value["contract"], v3.CONTRACT)
            self.assertEqual(value["policy"], "REVERSE_P3_RACE_MAX300")
            self.assertEqual(value["races"][0]["portfolio_total_yen"], 1000)
            self.assertFalse(value["actual_bets_placed"])

    def test_duplicate_races_not_double_bought(self):
        race = make_race()
        with self.assertRaisesRegex(ValueError, "DUPLICATE_RACE_ID"):
            v3.run(make_input(race, copy.deepcopy(race)))


if __name__ == "__main__":
    unittest.main()
