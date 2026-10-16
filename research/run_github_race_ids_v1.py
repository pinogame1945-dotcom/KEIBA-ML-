#!/usr/bin/env python3
"""Race ID(s) -> original L2.1 -> adopted L3 V3, entirely in GitHub.

The caller supplies a GitHub-produced, source-attested manifest. No Android
app dependency, no proxy model, no historical FINAL odds, no live network,
no payout/result data. Missing inputs produce per-race BLOCKED diagnostics.
"""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "KEIBA_GITHUB_RACE_OUTPUT_BATCH_V1"
MANIFEST = "KEIBA_GITHUB_RACE_INPUT_MANIFEST_V1"
_ID = re.compile(r"^20[0-9]{10}$")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "research" / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


L21 = load_module("l21_github_frozen_prebet.py", "keiba_github_l21")
L3 = load_module("l3_v3_operational.py", "keiba_official_l3_v3")
ANCHOR = load_module("score_github_original_anchor_asof.py", "keiba_original_anchor_github")


def race_id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("NETKEIBA_12_DIGIT_RACE_ID_REQUIRED")
    if int(value[:4]) < 2022:
        raise ValueError("FROZEN_L21_RESEARCH_INPUT_YEAR_UNSUPPORTED")
    if int(value[:4]) >= 2027:
        raise ValueError("FUTURE_MODEL_FOLD_NOT_PROVEN")
    return value


def source_file(root, reference):
    if not isinstance(reference, str) or not reference or "\\0" in reference:
        raise ValueError("INVALID_GITHUB_SOURCE_PATH")
    name = Path(reference)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("PATH_TRAVERSAL_FORBIDDEN")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError("GITHUB_SOURCE_FILE_MISSING")
    if target.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("SOURCE_FILE_TOO_LARGE")
    return json.loads(target.read_text(encoding="utf-8"))


def ranking(model, card):
    horses = model.get("horses")
    if not isinstance(horses, list):
        raise ValueError("ORIGINAL_L21_HORSE_PANEL_MISSING")
    race = card.get("race") or {}
    entries = card.get("entries") or []
    name_map = {}
    for entry in entries:
        if entry.get("entryStatus") != "ACTIVE":
            continue
        no = entry.get("horseNo")
        if type(no) is not int or no in name_map:
            raise ValueError("CARD_HORSE_NUMBER_INVALID")
        name_map[no] = entry
    if len(name_map) != len(horses):
        raise ValueError("ORIGINAL_L21_RANKING_FIELD_DRIFT")
    raw, ev = [], []
    seen_id = set()
    for h in horses:
        n = h["horse_number"]
        if n not in name_map or h["horse_id"] != name_map[n].get("canonicalHorseId"):
            raise ValueError("RANKING_HORSE_ID_DRIFT")
        if h["horse_id"] in seen_id:
            raise ValueError("RANKING_DUPLICATE_HORSE")
        seen_id.add(h["horse_id"])
        prob = L21.number(h.get("P__ANCHOR_FULL"), "P__ANCHOR_FULL")
        price = L21.number(h.get("win_odds_at_cutoff"), "original_win_quote")
        if not 0 <= prob <= 1 or price < 1:
            raise ValueError("INVALID_ORIGINAL_ANCHOR_PROBABILITY_PRICE")
        mr = h.get("market_rank")
        ai_rank = h.get("distill_ai_rank")
        if type(mr) is not int or type(ai_rank) is not int:
            raise ValueError("ORIGINAL_AI_RANK_INVALID")
        row = {
            "horse_id": h["horse_id"],
            "horse_number": n,
            "horse_name": name_map[n].get("horseName"),
            "ai_raw_rank": ai_rank,
            "P__ANCHOR_FULL": prob,
            "win_odds_at_quote": price,
            "expected_return_index": prob * price,
            "expected_net_return_per_yen": prob * price - 1.0,
            "model_market_rank": mr,
        }
        raw.append(row)
        ev.append(row)
    if sorted(h["ai_raw_rank"] for h in raw) != list(range(1, len(raw) + 1)):
        raise ValueError("ORIGINAL_AI_RAW_RANK_NOT_COMPLETE")
    raw.sort(key=lambda x: (x["ai_raw_rank"], x["horse_id"]))
    ev.sort(key=lambda x: (-x["expected_return_index"], -x["P__ANCHOR_FULL"],
                           x["model_market_rank"], x["horse_id"]))
    return {
        "ai_raw": [{
            "rank": i, "horse_id": h["horse_id"],
            "horse_number": h["horse_number"], "horse_name": h["horse_name"],
            "original_raw_rank": h["ai_raw_rank"]
        } for i, h in enumerate(raw, 1)],
        "expected_value": [{
            "rank": i, "horse_id": h["horse_id"],
            "horse_number": h["horse_number"], "horse_name": h["horse_name"],
            "original_ANCHOR_FULL_probability": h["P__ANCHOR_FULL"],
            "quote_win_odds": h["win_odds_at_quote"],
            "expected_return_index": h["expected_return_index"],
            "expected_net_return_per_yen": h["expected_net_return_per_yen"]
        } for i, h in enumerate(ev, 1)],
        "ranking_definitions": {
            "ai_raw": "ORIGINAL_DISTILL_AI_RANK_UNADJUSTED_BY_MARKET",
            "expected_value": "ORIGINAL_ANCHOR_FULL_P_X_SAME_SNAPSHOT_WIN_ODDS",
            "not_independent_expected_value_model": True,
            "odds_snapshot_as_of": model.get("win_odds_observed_at"),
        }
    }


def run(requested, manifest, manifest_root=None):
    if manifest.get("contract") != MANIFEST:
        raise ValueError("GITHUB_MANIFEST_CONTRACT_INVALID")
    entries = manifest.get("races")
    budget = manifest.get("budget")
    if not isinstance(entries, dict) or not isinstance(budget, dict):
        raise ValueError("GITHUB_MANIFEST_RACES_AND_BUDGET_REQUIRED")
    if not isinstance(requested, list) or not requested:
        raise ValueError("NO_REQUESTED_RACE_IDS")
    if len(set(requested)) != len(requested):
        raise ValueError("DUPLICATE_REQUESTED_RACE_ID")
    requested = [race_id(x) for x in requested]
    root = (Path(manifest_root) if manifest_root else ROOT).resolve()
    results = {}
    compatible = []
    l2s = {}
    for rid in requested:
        fail = lambda reason: {"race_id": rid, "status": "BLOCKED",
                               "blocked_reason": reason,
                               "rankings": None, "l2_1": None, "l3_v3": None}
        entry = entries.get(rid)
        if not isinstance(entry, dict):
            results[rid] = fail("RACE_ID_NOT_IN_GITHUB_SOURCE_MANIFEST")
            continue
        try:
            required = ("race_card", "jra_prebet_quotes", "original_l175_horse_p3")
            files = {key: source_file(root, entry.get(key)) for key in required}
            card, quotes, p3 = (files[x] for x in required)
            use_original_model = ("original_anchor_parameters" in entry or
                                  "original_ability_risk" in entry)
            if use_original_model:
                if "original_l21_features" in entry:
                    raise ValueError("CANNOT_MIX_PRIOR_MODEL_AND_EXTERNALLY_SCORED_ANCHOR")
                params = source_file(root, entry.get("original_anchor_parameters"))
                companions = source_file(root, entry.get("original_ability_risk"))
                # Real current UTC clock is mandatory for executable CLI.
                # Historical synthetic tests may call the pure scorer directly,
                # but cannot inject a fabricated clock into this entrypoint.
                derived = ANCHOR.calc(params, card, quotes, companions)
                model = derived["model_full_probs"]
            else:
                model = source_file(root, entry.get("original_l21_features"))
            source = (card.get("capture") or {})
            if (card.get("contract") != "KEIBA_GITHUB_JRA_RACE_CARD_V1" or
                source.get("source_system") != "KEIBA_GITHUB_JRA" or
                source.get("final_probe_checked") is not True or
                source.get("is_final") is not False or
                source.get("source_status") != "LIVE"):
                raise ValueError("GITHUB_OFFICIAL_PREBET_RACECARD_PROVENANCE_REQUIRED")
            if (quotes.get("source_system") != "KEIBA_GITHUB_JRA" or
                quotes.get("canonical_race_id") != rid or
                (card.get("race") or {}).get("canonicalRaceId") != rid):
                raise ValueError("GITHUB_L2_QUOTE_RACE_DRIFT")
            l2 = L21.execute(card, model, quotes, policy="STANDARD")
            if l2["status"] != "PAPER_ONLY_PREBET_QUOTE_VALIDATED":
                raise ValueError(l2["status"])
            ranks = ranking(model, card)
            l2s[rid] = l2
            results[rid] = {"race_id": rid,
                            "status": "L2_READY_L3_PENDING",
                            "blocked_reason": None,
                            "rankings": ranks,
                            "original_anchor_source": (
                                "ORIGINAL_FROZEN_2025_PRIOR_MODEL_GITHUB_INFERENCE"
                                if use_original_model else
                                "PRECOMPUTED_ORIGINAL_HORSE_INPUT_ASOF_NOT_INDEPENDENTLY_VERIFIED"),
                            "l2_1": l2,
                            "l3_v3": None}
            compatible.append({"race_date": card["race"]["raceDate"],
                               "l2_1": l2, "horse_p3": p3})
        except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
            results[rid] = fail(str(exc))
    incomplete_requested_batch = len(compatible) != len(requested)
    if compatible:
        try:
            request = {"contract": L3.CONFIG, "policy": L3.POLICY,
                       "races": compatible, "budget": budget}
            allocated = L3.run(request)
            for block in allocated["races"]:
                rid = block["race_id"]
                results[rid]["l3_v3"] = block
                results[rid]["status"] = (
                    "PAPER_ONLY_COMPLETE"
                    if block["status"] == "PAPER_ONLY_SOURCE_OR_P3_UNVERIFIED"
                    else block["status"])
                if not block["tickets"] and block["status"] not in (
                    "READY_VERIFIED_SOURCE", "PAPER_ONLY_SOURCE_OR_P3_UNVERIFIED"):
                    results[rid]["blocked_reason"] = block["status"]
                elif incomplete_requested_batch:
                    # Cannot certify a same-day bankroll if one requested race
                    # has missing frozen inputs. Preserve diagnostics, never READY.
                    results[rid]["status"] = "PAPER_ONLY_PARTIAL_REQUESTED_BATCH"
                    results[rid]["blocked_reason"] = "REQUESTED_RACES_HAVE_MISSING_UPSTREAM_INPUTS"
            daily = allocated["daily_bankroll"]
        except (ValueError, KeyError, TypeError, AssertionError) as exc:
            for record in compatible:
                rid = record["l2_1"]["race_id"]
                results[rid]["status"] = "BLOCKED_L3_V3"
                results[rid]["blocked_reason"] = str(exc)
            daily = []
    else:
        daily = []
    ordered = [results[rid] for rid in requested]
    return {
        "contract": CONTRACT,
        "requested_race_ids": requested,
        "status": ("PAPER_ONLY_OR_BLOCKED" if any(
            r["status"] != "READY_VERIFIED_SOURCE" for r in ordered)
            else "READY_VERIFIED_SOURCE"),
        "processing_owner": "KEIBA_ML_GITHUB",
        "mobile_role": "READ_ONLY_DISPLAY",
        "l2_policy": "KEIBA_L2_1_FIXED_PURCHASE_V1:STANDARD",
        "l3_policy": "REVERSE_P3_RACE_MAX300",
        "races": ordered,
        "daily_bankroll": daily,
        "daily_bankroll_scope": "VALID_REQUESTED_RACES_ONLY_NOT_ATTESTED_ALL_JRA_RACES",
        "all_requested_races_have_original_inputs": not incomplete_requested_batch,
        "verified_collector_timestamp_provenance": False,
        "historical_final_odds_prohibited": True,
        "2026_historical_outcomes_sealed": True,
        "actual_bets_placed": False,
        "unknown_year_is_not_evaluated_for_roi": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", action="append", required=True,
                    help="netkeiba 12-digit race ID; repeat for multiple races")
    ap.add_argument("--manifest", required=True,
                    help="GitHub-generated mapping of race IDs to prebet files")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    target = Path(a.output)
    manifest_file = Path(a.manifest).resolve()
    report = run(a.race, json.loads(manifest_file.read_text(encoding="utf-8")),
                 manifest_file.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print("KEIBA_GITHUB_RACE_BATCH", report["status"], len(report["races"]))
    for item in report["races"]:
        print(item["race_id"], item["status"],
              item.get("blocked_reason") or "")


if __name__ == "__main__":
    main()
