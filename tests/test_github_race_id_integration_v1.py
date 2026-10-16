#!/usr/bin/env python3
"""No-network GitHub-only race-ID batch: original L2 rules and V3 money."""
import copy
import importlib.util
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import run_github_race_ids_v1 as orchestrator
from l3_v3_operational import POLICY

DAY = "2025-01-06"
TIME = "2025-01-06T13:32:00+09:00"
CUTOFF = "2025-01-06T14:25:00+09:00"
IDS = ("202507010212", "202507010211")
SOURCE = "https://www.jra.go.jp/test"


def fixture(rid):
    key = "GITHUB_JRA:" + rid
    card = {
        "contract": "KEIBA_GITHUB_JRA_RACE_CARD_V1",
        "race": {"raceKey": key, "canonicalRaceId": rid, "raceDate": DAY,
                 "startTime": "14:30", "raceStatus": "SCHEDULED",
                 "scheduleStatus": "ACTIVE", "status": "OFFICIAL",
                 "sourceUrl": SOURCE},
        "entries": [
            {"raceKey": key, "horseNo": i, "canonicalHorseId": f"horse{i}",
             "horseName": f"horse-{i}", "entryStatus": "ACTIVE"}
            for i in range(1, 9)
        ],
        "capture": {
            "source_system": "KEIBA_GITHUB_JRA", "final_probe_checked": True,
            "is_final": False, "source_status": "LIVE",
            "final_probe_checked_at": "2025-01-06T13:33:00+09:00",
            "bet_cutoff_at": CUTOFF
        }
    }
    original = {
        "contract": "KEIBA_L21_ORIGINAL_ANCHOR_FEATURES_PREBET_V1",
        "race_id": rid, "canonical_race_key": key,
        "anchor_source": "EXACT_EXISTING_ANCHOR_FULL",
        "anchor_horse_id": "horse1",
        "feature_source": "ORIGINAL_L21_HORSE_FEATURES",
        "input_phase": "PRE_RACE", "features_asof_asserted": True,
        "features_observed_at": "2025-01-06T13:00:00+09:00",
        "win_odds_observed_at": TIME,
        "anchor_computed_at": "2025-01-06T13:34:00+09:00",
        "horses": [{
            "horse_id": f"horse{n}", "horse_number": n,
            "P__ANCHOR_FULL": 0.31 if n == 1 else 0.14 - n * .005,
            "risk": .09, "distill_score_z": .45,
            "distill_ai_rank": {2: 3, 3: 4, 4: 5, 5: 2}.get(n,n),
            "market_rank": n, "market_prob_norm": .15,
            "win_odds_at_cutoff": float(n + 2)
        } for n in range(1, 9)]
    }
    def row(bt, ns, odds):
        return {"raceKey": key, "betType": bt,
                "selection1": ns[0], "selection2": ns[1] if len(ns) > 1 else None,
                "selection3": ns[2] if len(ns) > 2 else None,
                "odds": odds, "oddsMin": None, "oddsMax": None,
                "observedAt": TIME, "sourceUrl": SOURCE}
    odds = [row("WIN", (n,), n + 2.) for n in range(1, 9)]
    for a,b in itertools.combinations(range(1,9),2):
        odds.append(row("QUINELLA", (a,b), 16.))
    for a,b,c in itertools.combinations(range(1,9),3):
        odds.append(row("TRIO", (a,b,c), 16. if (b+c)%2 else 35.))
    for a,b,c in itertools.permutations(range(1,9),3):
        odds.append(row("TRIFECTA", (a,b,c),
                        240. if a==1 else 25. if c==1 else 500.))
    assert len(odds) == 428
    quotes = {
        "contract": "KEIBA_JRA_COMBINATION_ODDS_ASOF_V1",
        "canonical_race_id": rid, "raceKey": key,
        "source_system": "KEIBA_GITHUB_JRA", "source_phase": "PRE_RACE",
        "is_final": False, "source_status": "LIVE",
        "source_confirmed_prestart": True, "observed_at": TIME,
        "bet_cutoff_at": CUTOFF, "scheduled_start_at":
            DAY + "T14:30:00+09:00",
        "rows": [x for x in odds if x["betType"]!="WIN"],
        "win_odds_rows": [x for x in odds if x["betType"]=="WIN"],
    }
    p3 = {"contract": "KEIBA_L175_HORSE_P3_ASOF_V1",
          "race_id": rid, "source": "ORIGINAL_L175_PREBET_P3",
          "model_version": "SYNTHETIC_ONLY_NOT_REAL",
          "observed_before_bet": True,
          "source_independently_verified": False,
          "observed_at": TIME,
          "horses": [{"horse_number": n, "horse_id": f"horse{n}",
                      "p3": (None if n == 8 else .06 + n * .08)}
                     for n in range(1,9)]}
    return card, original, quotes, p3


def write_fixture(tmp, races=IDS):
    root = Path(tmp)
    entries = {}
    for rid in races:
        folder = root / rid
        folder.mkdir()
        card, original, quotes, p3 = fixture(rid)
        entries[rid] = {}
        for k,obj in zip(("race_card","original_l21_features",
                          "jra_prebet_quotes","original_l175_horse_p3"),
                         (card,original,quotes,p3)):
            name = folder / (k + ".json")
            name.write_text(json.dumps(obj),encoding="utf-8")
            entries[rid][k] = f"{rid}/{k}.json"
    manifest = {"contract": orchestrator.MANIFEST, "races": entries,
        "budget": {
            "race_budget_yen": {rid: 10000 for rid in races},
            "max_race_exposure_yen": 10000,
            "available_cash_by_date_yen": {DAY: 20000},
            "already_reserved_same_day_yen": {DAY: 0},
            "max_day_exposure_yen": 20000}}
    (root/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    return manifest


class GithubRaceIntegration(unittest.TestCase):
    def test_exact_frozen_l2_and_v3_two_races(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=write_fixture(tmp)
            result=orchestrator.run(list(IDS),manifest,tmp)
            self.assertEqual(result["status"],"PAPER_ONLY_OR_BLOCKED")
            self.assertEqual(len(result["races"]),2)
            self.assertEqual(len(result["daily_bankroll"]),1)
            all_cost=0
            for i,r in enumerate(result["races"]):
                self.assertEqual(r["race_id"],IDS[i])
                self.assertEqual(r["status"],"PAPER_ONLY_COMPLETE",r.get("blocked_reason"))
                self.assertEqual(len(r["rankings"]["ai_raw"]),8)
                self.assertEqual(len(r["rankings"]["expected_value"]),8)
                self.assertEqual(r["rankings"]["ai_raw"][0]["horse_number"],1)
                self.assertEqual(r["rankings"]["ai_raw"][1]["horse_number"],5)
                self.assertEqual(r["l2_1"]["frozen_version"],"2.1.0")
                self.assertEqual(r["l2_1"]["ticket_quote_missing_unique_count"],0)
                self.assertGreater(r["l2_1"]["physical_ticket_count"],0)
                self.assertEqual(r["l3_v3"]["required_ticket_count"],
                                 r["l2_1"]["physical_ticket_count"])
                self.assertEqual(
                    {(x["bet_type"],x["canonical_selection"])
                     for x in r["l2_1"]["tickets"]},
                    {(x["bet_type"],x["canonical_selection"])
                     for x in r["l3_v3"]["tickets"]})
                self.assertTrue(all(x["stake_yen"] in (100,200,300)
                                    for x in r["l3_v3"]["tickets"]))
                all_cost+=r["l3_v3"]["portfolio_total_yen"]
            self.assertEqual(all_cost,result["daily_bankroll"][0]["requested_new_stakes_yen"])
            self.assertFalse(result["actual_bets_placed"])
            self.assertEqual(result["mobile_role"],"READ_ONLY_DISPLAY")

    def test_missing_race_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=write_fixture(tmp,IDS[:1])
            res=orchestrator.run(list(IDS),manifest,tmp)
            self.assertEqual(res["races"][1]["status"],"BLOCKED")
            self.assertEqual(res["races"][1]["blocked_reason"],
                             "RACE_ID_NOT_IN_GITHUB_SOURCE_MANIFEST")
            self.assertEqual(res["races"][1]["rankings"],None)

    def test_invalid_or_final_quote_blocks_only_affected_race(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=write_fixture(tmp)
            p=Path(tmp)/IDS[0]/"jra_prebet_quotes.json"
            content=json.loads(p.read_text())
            content["is_final"]=True
            p.write_text(json.dumps(content))
            result=orchestrator.run(list(IDS),manifest,tmp)
            self.assertEqual(result["races"][0]["status"],"BLOCKED")
            self.assertEqual(result["races"][1]["status"],"PAPER_ONLY_COMPLETE")

    def test_missing_original_p3_blocks_v3_and_does_not_forge_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=write_fixture(tmp,IDS[:1])
            p=Path(tmp)/IDS[0]/"original_l175_horse_p3.json"
            value=json.loads(p.read_text());value["source"]="SEVEN_KING_PROXY"
            p.write_text(json.dumps(value))
            result=orchestrator.run(list(IDS[:1]),manifest,tmp)
            self.assertEqual(result["races"][0]["status"],"BLOCKED_L3_V3")
            self.assertIn("ORIGINAL_L175_P3_SOURCE_REQUIRED",
                          result["races"][0]["blocked_reason"])
            self.assertEqual(result["races"][0]["l3_v3"],None)

    def test_prebet_only_guard_and_manifest_security(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=write_fixture(tmp,IDS[:1])
            manifest["races"][IDS[0]]["race_card"]="../file.json"
            r=orchestrator.run(list(IDS[:1]),manifest,tmp)
            self.assertEqual(r["races"][0]["blocked_reason"],"PATH_TRAVERSAL_FORBIDDEN")
            with self.assertRaisesRegex(ValueError,"DUPLICATE_REQUESTED_RACE"):
                orchestrator.run([IDS[0],IDS[0]],manifest,tmp)
            with self.assertRaisesRegex(ValueError,"FUTURE_MODEL_FOLD_NOT_PROVEN"):
                orchestrator.run(["202707010212"],manifest,tmp)

    def test_cli_writes_complete_display_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp)
            target=Path(tmp)/"result.json"
            cmd=[sys.executable,str(ROOT/"research/run_github_race_ids_v1.py"),
                 "--race",IDS[0],"--race",IDS[1],
                 "--manifest",str(Path(tmp)/"manifest.json"),
                 "--output",str(target)]
            p=subprocess.run(cmd,capture_output=True,text=True,check=False)
            self.assertEqual(p.returncode,0,p.stderr)
            result=json.loads(target.read_text())
            self.assertEqual(result["l3_policy"],POLICY)
            self.assertEqual(len(result["races"]),2)
            self.assertTrue(all(x["l3_v3"] is not None for x in result["races"]))


if __name__=="__main__":
    unittest.main()
