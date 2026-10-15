#!/usr/bin/env python3
"""Operational L3 V3 allocator for frozen L2.1 tickets.

GitHub-side, no data collection, no placing bets. Exact V2 REVERSE_P3_RACE_MAX300
racewise priority at 200*N, minimum 100, maximum 300. Budget shortage is
all-or-none. Daily funding reserves all same-day stakes before any return.
Inputs must be already-generated original frozen L2.1 tickets and independently
derived, timestamped horse-level prebet p3 (never substitute win probability).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path

CONTRACT = "KEIBA_L3_V3_OPERATIONAL_V1"
CONFIG = "KEIBA_L3_V3_RUNTIME_CONFIG_V1"
SIGNALS = "KEIBA_L175_HORSE_P3_ASOF_V1"
L21 = "KEIBA_L21_PREBET_QUOTED_TICKETS_V1"
POLICY = "REVERSE_P3_RACE_MAX300"
ALLOWED_BETS = {"QUINELLA": 2, "TRIO": 3, "TRIFECTA": 3}


def yen(value, name):
    if type(value) is not int or value < 0 or value > 100_000_000 or value % 100:
        raise ValueError("INVALID_" + name)
    return value


def iso(value, name):
    if not isinstance(value, str):
        raise ValueError("INVALID_" + name)
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("INVALID_" + name) from exc
    if parsed.tzinfo is None:
        raise ValueError("TIMEZONE_REQUIRED_" + name)
    return parsed.astimezone(dt.timezone.utc)


def ticket_key(row):
    return (row["bet_type"], row["canonical_selection"])


def ticket_id(race_id, row):
    return hashlib.sha256(
        (race_id + "|" + row["bet_type"] + "|" + row["canonical_selection"]).encode("ascii")
    ).hexdigest()[:20]


def _validate_race(bundle):
    if not isinstance(bundle, dict):
        raise ValueError("BAD_RACE_BUNDLE")
    race_date = bundle.get("race_date")
    if not isinstance(race_date, str) or not re.fullmatch(r"20\d\d-\d\d-\d\d", race_date):
        raise ValueError("RACE_DATE_REQUIRED")
    try:
        dt.date.fromisoformat(race_date)
    except ValueError as exc:
        raise ValueError("INVALID_RACE_DATE") from exc
    l2 = bundle.get("l2_1")
    if not isinstance(l2, dict):
        raise ValueError("FROZEN_L2_1_REQUIRED")
    race_id = l2.get("race_id")
    if not isinstance(race_id, str) or not re.fullmatch(r"\d{12}", race_id) or race_id[:4] != race_date[:4]:
        raise ValueError("RACE_ID_OR_YEAR_DRIFT")
    if (l2.get("contract") != L21 or
            l2.get("frozen_policy") != "KEIBA_L2_1_FIXED_PURCHASE_V1" or
            l2.get("frozen_version") != "2.1.0" or l2.get("policy") != "STANDARD"):
        raise ValueError("NOT_FROZEN_STANDARD_L21")
    if l2.get("status") not in ("PAPER_ONLY_PREBET_QUOTE_VALIDATED",
                                "PREBET_QUOTE_VALIDATED_VERIFIED"):
        raise ValueError("L2_1_NOT_PREBET_VALIDATED")
    if l2.get("historical_final_odds_used") is not False:
        raise ValueError("HISTORICAL_FINAL_ODDS_NOT_ALLOWED")
    observed = iso(l2.get("quote_observed_at"), "ODDS_OBSERVED")
    cutoff = iso(l2.get("bet_cutoff_at"), "BET_CUTOFF")
    if not observed < cutoff:
        raise ValueError("ODDS_NOT_BEFORE_CUTOFF")
    if race_id.startswith("2026"):
        now = dt.datetime.now(dt.timezone.utc)
        if not observed <= now < cutoff:
            raise ValueError("2026_HISTORICAL_OR_FUTURE_FAKE_PREBET_FORBIDDEN")
    tickets = l2.get("tickets")
    if not isinstance(tickets, list) or len(tickets) > 1000:
        raise ValueError("INVALID_FROZEN_TICKETS")
    if len(tickets) != l2.get("physical_ticket_count") or (
        l2.get("baseline_total_stake_yen") != 100 * len(tickets)
    ):
        raise ValueError("L2_1_TICKET_COUNT_DRIFT")
    seen = set()
    for row in tickets:
        if not isinstance(row, dict) or row.get("race_id") != race_id:
            raise ValueError("L2_1_TICKET_RACE_DRIFT")
        bet = row.get("bet_type")
        if bet not in ALLOWED_BETS:
            raise ValueError("FROZEN_BET_TYPE_DRIFT")
        ns = row.get("selection_numbers")
        if (not isinstance(ns, list) or len(ns) != ALLOWED_BETS[bet] or
                any(type(n) is not int or not 1 <= n <= 24 for n in ns) or
                len(ns) != len(set(ns)) or
                (bet != "TRIFECTA" and ns != sorted(ns))):
            raise ValueError("INVALID_CANONICAL_SELECTION")
        if row.get("canonical_selection") != "-".join(map(str, ns)):
            raise ValueError("SELECTION_ID_DRIFT")
        k = ticket_key(row)
        if k in seen:
            raise ValueError("DUPLICATE_PHYSICAL_TICKET")
        seen.add(k)
        if row.get("baseline_stake_yen") != 100:
            raise ValueError("L2_1_STAKE_BASELINE_DRIFT")
        price = row.get("decimal_odds")
        if type(price) not in (int, float) or not math.isfinite(price) or not 1 <= price <= 1_000_000:
            raise ValueError("INVALID_PREBET_ODDS")
        if iso(row.get("observed_at"), "TICKET_ODDS") != observed:
            raise ValueError("TICKET_ODDS_TIME_DRIFT")
        gates = row.get("matched_line_gates")
        if not isinstance(gates, list) or not gates or any(
            not isinstance(g, dict) or not isinstance(g.get("line"), str)
            or not isinstance(g.get("gate"), str) or not g["line"] or not g["gate"]
            for g in gates
        ):
            raise ValueError("FROZEN_LINE_GATE_EVIDENCE_MISSING")

    signals = bundle.get("horse_p3")
    if not isinstance(signals, dict) or signals.get("contract") != SIGNALS or (
        signals.get("race_id") != race_id
    ) or signals.get("source") != "ORIGINAL_L175_PREBET_P3":
        raise ValueError("ORIGINAL_L175_P3_SOURCE_REQUIRED")
    if signals.get("observed_before_bet") is not True:
        raise ValueError("L175_ASOF_ATTESTATION_REQUIRED")
    if not isinstance(signals.get("model_version"), str) or not signals["model_version"]:
        raise ValueError("L175_MODEL_VERSION_REQUIRED")
    signal_at = iso(signals.get("observed_at"), "L175_P3_ASOF")
    if signal_at >= cutoff:
        raise ValueError("L175_P3_NOT_BEFORE_BET_CUTOFF")
    if race_id.startswith("2026") and signal_at > dt.datetime.now(dt.timezone.utc):
        raise ValueError("2026_FUTURE_L175_P3_NOT_YET_OBSERVED")
    horses = signals.get("horses")
    if not isinstance(horses, list) or not horses:
        raise ValueError("L175_HORSE_P3_EMPTY")
    by_no = {}
    ids = set()
    for h in horses:
        if not isinstance(h, dict) or set(h) != {"horse_number", "horse_id", "p3"}:
            raise ValueError("L175_HORSE_SIGNAL_FIELDS_INVALID")
        n = h.get("horse_number")
        horse_id = h.get("horse_id")
        p3 = h.get("p3")
        if type(n) is not int or not 1 <= n <= 24 or n in by_no or (
            not isinstance(horse_id, str) or not horse_id or horse_id in ids
        ):
            raise ValueError("L175_HORSE_ID_OR_NUMBER_DRIFT")
        if p3 is not None and (type(p3) not in (float, int) or
                               not math.isfinite(p3) or not 0 <= p3 <= 1):
            raise ValueError("INVALID_L175_HORSE_P3")
        by_no[n] = p3
        ids.add(horse_id)
    covered = {n for t in tickets for n in t["selection_numbers"]}
    if not covered.issubset(by_no):
        raise ValueError("L175_P3_TICKET_HORSE_COVERAGE_MISSING")
    if (set(signals) - {"contract", "race_id", "source", "observed_before_bet",
                        "model_version", "observed_at", "horses", "source_independently_verified"}):
        raise ValueError("UNEXPECTED_L175_SIGNAL_FIELDS")
    source_verified = (signals.get("source_independently_verified") is True and
        l2.get("source_authenticity_independently_verified") is True and
        l2["status"] == "PREBET_QUOTE_VALIDATED_VERIFIED")
    enriched = []
    for row in tickets:
        values = [by_no[n] for n in row["selection_numbers"]]
        enriched.append((row, min(values) if all(v is not None for v in values) else None))
    return race_date, race_id, enriched, source_verified


def reverse_p3_stakes(enriched, budget_yen):
    """V2 score ordering, including tied p3 and missing p3; no payout in input."""
    n = len(enriched)
    if n == 0:
        return {}
    budget = yen(budget_yen, "ALLOCATION_BUDGET")
    if budget < 100 * n or budget > 300 * n:
        raise ValueError("INSUFFICIENT_OR_EXCESS_ALLOCATION_BUDGET")
    keys = [ticket_key(row) for row, _ in enriched]
    if len(keys) != len(set(keys)):
        raise ValueError("DUPLICATE_L2_TICKET")
    distinct = sorted({p for _, p in enriched if p is not None})
    position = {p: (.5 if len(distinct) == 1 else
                    1 - i / (len(distinct) - 1)) for i, p in enumerate(distinct)}
    ordered = sorted(enriched, key=lambda item: (
        item[1] is None,
        -(position[item[1]] if item[1] is not None else 0),
        ticket_key(item[0])))
    assigned = {k: 100 for k in keys}
    units = (budget - 100 * n) // 100
    for row, _ in ordered:
        extra = min(2, units)
        assigned[ticket_key(row)] += 100 * extra
        units -= extra
        if units == 0:
            break
    if units or sum(assigned.values()) != budget or any(
        amount not in (100, 200, 300) for amount in assigned.values()
    ):
        raise AssertionError("V3_ALLOCATION_CONSERVATION_FAILURE")
    return assigned


def run(payload, now=None):
    if not isinstance(payload, dict) or payload.get("contract") != CONFIG or (
        payload.get("policy") != POLICY
    ):
        raise ValueError("NOT_OFFICIAL_L3_V3_CONFIG")
    records = payload.get("races")
    if not isinstance(records, list) or not records:
        raise ValueError("NO_RACES")
    config = payload.get("budget")
    if not isinstance(config, dict):
        raise ValueError("L3_V3_BUDGET_CONFIG_REQUIRED")
    race_limits = config.get("race_budget_yen")
    available = config.get("available_cash_by_date_yen")
    reserved = config.get("already_reserved_same_day_yen", {})
    if not isinstance(race_limits, dict) or not isinstance(available, dict) or not isinstance(reserved, dict):
        raise ValueError("INVALID_L3_V3_BUDGET_MAPS")
    max_race = yen(config.get("max_race_exposure_yen"), "MAX_RACE_EXPOSURE")
    day_cap = config.get("max_day_exposure_yen")
    if day_cap is not None:
        day_cap = yen(day_cap, "MAX_DAY_EXPOSURE")
    parsed = []
    seen_races = set()
    for raw in records:
        day, race_id, enriched, verified = _validate_race(raw)
        if race_id in seen_races:
            raise ValueError("DUPLICATE_RACE_ID")
        seen_races.add(race_id)
        if race_id not in race_limits:
            raise ValueError("RACE_BUDGET_MISSING_" + race_id)
        cap = min(yen(race_limits[race_id], "RACE_BUDGET"), max_race)
        n = len(enriched)
        required_min = 100 * n
        # V2 budget policy is 200 yen per ticket on average.
        # A lower user cap reduces increments; a higher cap never forces extra spend.
        target = min(cap, 200 * n)
        parsed.append(dict(day=day, race_id=race_id, enriched=enriched,
                           verified=verified, minimum=required_min,
                           requested_budget=cap, target=target,
                           blocking=("INSUFFICIENT_RACE_BUDGET" if cap < required_min else None)))

    by_day = {}
    for record in parsed:
        by_day.setdefault(record["day"], []).append(record)
    days = []
    output = []
    for day in sorted(by_day):
        group = by_day[day]
        if day not in available:
            raise ValueError("DATE_BANKROLL_REQUIRED_" + day)
        cash = yen(available[day], "AVAILABLE_BANKROLL")
        held = yen(reserved.get(day, 0), "ALREADY_RESERVED_DAY")
        if held > cash:
            raise ValueError("EXISTING_RESERVATION_EXCEEDS_CASH")
        spend = sum(r["target"] for r in group if not r["blocking"])
        effective = cash - held
        failed = next((r["blocking"] for r in group if r["blocking"]), None)
        if failed:
            state = failed
        elif day_cap is not None and held + spend > day_cap:
            state = "DAY_EXPOSURE_CAP_EXCEEDED"
        elif spend > effective:
            state = "INSUFFICIENT_BUDGET"
        else:
            state = "FUNDED"
        days.append({"race_date": day, "status": state,
                     "available_cash_yen": cash, "already_reserved_yen": held,
                     "requested_new_stakes_yen": spend,
                     "total_reserved_if_funded_yen": held + spend,
                     "cash_after_reservation_yen": effective - spend if state == "FUNDED" else None,
                     "shortage_yen": max(0, spend - effective) if state == "INSUFFICIENT_BUDGET" else 0,
                     "same_day_payouts_credited_yen": 0,
                     "bankroll_rule": "RESERVE_ALL_SAME_DAY_BEFORE_PAYOUT"})
        for r in group:
            if state != "FUNDED":
                status = r["blocking"] or ("BLOCKED_" + state)
                tickets_out = []
            else:
                stakes = reverse_p3_stakes(r["enriched"], r["target"])
                tickets_out = []
                for ticket, min_p3 in r["enriched"]:
                    tickets_out.append({
                        "L2_1_ticket_id": ticket_id(r["race_id"], ticket),
                        "race_id": r["race_id"],
                        "bet_type": ticket["bet_type"],
                        "canonical_selection": ticket["canonical_selection"],
                        "selection_numbers": list(ticket["selection_numbers"]),
                        "decimal_odds": ticket["decimal_odds"],
                        "odds_timestamp": ticket["observed_at"],
                        "source_line_gates": list(ticket["matched_line_gates"]),
                        "stake_yen": stakes[ticket_key(ticket)],
                        "min_p3": min_p3,
                        "p3_missing": min_p3 is None,
                    })
                status = ("READY_VERIFIED_SOURCE" if r["verified"] and
                          all(t["min_p3"] is not None for t in tickets_out)
                          else "PAPER_ONLY_SOURCE_OR_P3_UNVERIFIED")
            output.append({
                "race_id": r["race_id"], "race_date": day, "status": status,
                "required_minimum_yen": r["minimum"],
                "approved_race_budget_yen": r["requested_budget"],
                "portfolio_total_yen": sum(t["stake_yen"] for t in tickets_out),
                "required_ticket_count": len(r["enriched"]),
                "ticket_count": len(tickets_out),
                "missing_p3_tickets": sum(p is None for _, p in r["enriched"]),
                "l2_1_bet_set_unchanged": True,
                "tickets": tickets_out,
            })

    return {"contract": CONTRACT, "policy": POLICY, "l2_1_contract": "KEIBA_L2_1_FIXED_PURCHASE_V1",
            "l3_version": "V3", "stake_policy": POLICY,
            "status": ("READY_VERIFIED_SOURCE" if all(
                r["status"] == "READY_VERIFIED_SOURCE" for r in output)
                else "BLOCKED_OR_PAPER_ONLY"),
            "races": output, "daily_bankroll": days,
            "actual_bets_placed": False,
            "historical_roi_is_not_future_guarantee": True,
            "2026_historical_result_validation_sealed": True}


def main():
    parser = argparse.ArgumentParser(description="GitHub-side frozen L2.1 to official L3 V3")
    parser.add_argument("--input", required=True, help="L2.1 tickets + original time-safe L1.75 p3 + budget")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    try:
        result = run(input_data)
    except (ValueError, TypeError, KeyError, AssertionError) as exc:
        result = {"contract": CONTRACT, "status": "BLOCKED_INVALID_INPUT",
                  "reason": str(exc), "races": [], "actual_bets_placed": False}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("L3_V3_OPERATIONAL", result["status"], args.output)
    if result["status"] == "BLOCKED_INVALID_INPUT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
