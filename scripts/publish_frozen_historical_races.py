#!/usr/bin/env python3
"""Publish strictly historical frozen L2.1+V3 replay JSON to a public Git branch.

No artifacts, cache, paid storage, cloud compute or betting; publication only
via standard GitHub REST Contents API with the existing Actions GITHUB_TOKEN.
Creates at most 24 small, immutable race records per run. Same race may only
be written once with identical bytes. Fails closed on changes or storage cap.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "pinogame1945-dotcom/KEIBA-ML-"
DEST_BRANCH = "data/historical-l21-v3-v1"
FREEZE_SHA = "9655c717333636b5cc931f672e12dbc377e396bc"
INPUT_CONTRACT = "KEIBA_L21_FROZEN_FINAL_ARCHIVE_V3_REPLAY_V1"
OUTPUT_CONTRACT = "KEIBA_FROZEN_L21_V3_HISTORICAL_DISPLAY_V1"
MAX_RACES_PER_RUN = 24
MAX_BYTES_PER_RACE = 32768
MAX_NEW_BYTES_PER_RUN = 262144
MAX_PUBLISHED_HISTORY_BYTES = 4 * 1024 * 1024
ALLOWED_STATUS = {
    "HISTORICAL_FINAL_ONLY_UNAVAILABLE_AS_PREBET",
    "HISTORICAL_V3_SAME_DAY_NOT_FUNDED",
}
RID = re.compile(r"202[345][0-9]{8}\Z")


def check(condition, code):
    if not condition:
        raise ValueError(code)


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def validate_replay(data):
    check(isinstance(data, dict) and data.get("contract") == INPUT_CONTRACT,
          "NOT_FROZEN_REPLAY_INPUT")
    required = {
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
    }
    for field, expected in required.items():
        check(data.get(field) == expected, "UNTRUSTED_REPLAY_" + field)
    races = data.get("races")
    check(isinstance(races, list) and 1 <= len(races) <= MAX_RACES_PER_RUN,
          "RACE_COUNT_OUT_OF_RANGE")
    check(len({r.get("race_id") for r in races if isinstance(r, dict)}) == len(races),
          "DUPLICATE_PUBLISHED_RACE")
    output = []
    for r in races:
        check(isinstance(r, dict), "INVALID_RACE_ROW")
        rid = r.get("race_id")
        check(isinstance(rid, str) and RID.fullmatch(rid) is not None,
              "INVALID_OR_SEALED_2026_RACE_ID")
        check(r.get("race_date", "").startswith(rid[:4]),
              "RACE_ID_YEAR_MISMATCH")
        check(r.get("status") in ALLOWED_STATUS, "NOT_HISTORICAL_REPLAY_STATUS")
        check(r.get("l3_v3_policy") == "REVERSE_P3_RACE_MAX300",
              "WRONG_V3_STAKE_POLICY")
        rank = r.get("rankings")
        check(isinstance(rank, dict) and
              set(("ai_raw", "ai_prob", "value_ev")).issubset(rank),
              "ORIGINAL_RANKINGS_MISSING")
        rows = [rank["ai_raw"], rank["ai_prob"], rank["value_ev"]]
        check(all(isinstance(row, list) and row for row in rows),
              "EMPTY_ORIGINAL_RANKING")
        horses = [{entry["horse_id"] for entry in row} for row in rows]
        check(horses[0] == horses[1] == horses[2] and len(horses[0]) == len(rows[0]),
              "ORIGINAL_RANKED_HORSE_SET_DRIFT")
        for row in rows:
            check([e["rank"] for e in row] == list(range(1, len(row) + 1)),
                  "ORIGINAL_RANK_NUMBER_DRIFT")
        check(rank["value_ev"][0]["horse_id"] == r.get("original_anchor_horse_id"),
              "FROZEN_ANCHOR_CHANGED")
        tickets = r.get("tickets")
        check(isinstance(tickets, list), "INVALID_FROZEN_TICKETS")
        seen = set()
        paid = 0
        for ticket in tickets:
            key = (ticket["bet_type"], ticket["canonical_selection"])
            check(key not in seen, "DUPLICATE_FROZEN_PHYSICAL_TICKET")
            seen.add(key)
            check(ticket["bet_type"] in ("QUINELLA", "TRIO", "TRIFECTA"),
                  "NOT_FROZEN_L21_BET_TYPE")
            check(ticket.get("historical_price_phase") == "FINAL_RESEARCH_ONLY",
                  "NOT_FINAL_RESEARCH_TICKET_PROVENANCE")
            stake = ticket.get("stake_yen")
            check(type(stake) is int and stake in (100, 200, 300),
                  "INVALID_V3_STAKE_UNIT")
            paid += stake
        check(paid == r.get("l3_v3_total_yen"), "V3_STAKE_CONSERVATION_FAIL")
        if r["status"] == "HISTORICAL_FINAL_ONLY_UNAVAILABLE_AS_PREBET":
            check(len(tickets) == r.get("frozen_physical_ticket_count"),
                  "FROZEN_TICKET_SET_CHANGED")
        else:
            check(not tickets and paid == 0, "INSUFFICIENT_BUDGET_MUST_BLOCK_ALL")
        body = {
            "contract": OUTPUT_CONTRACT,
            "race_id": rid,
            "race_date": r["race_date"],
            "scope": "HISTORICAL_RESEARCH_REPLAY_ONLY",
            "purchase_ready": False,
            "prediction_as_of_before_start_verified": False,
            "historical_final_odds_used": True,
            "l2_1": {"version": "2.1.0", "policy": "STANDARD",
                     "frozen_commit_sha": FREEZE_SHA,
                     "tickets_unchanged": True,
                     "frozen_physical_ticket_count": r["frozen_physical_ticket_count"]},
            "l3_v3": {"version": "V3", "policy": "REVERSE_P3_RACE_MAX300",
                      "total_paper_stake_yen": paid,
                      "status": r["status"],
                      "available_tickets": tickets,
                      "missing_l17_p3_ticket_count": r["missing_l175_ticket_p3"]},
            "rankings": rank,
            "anchor_horse_id": r["original_anchor_horse_id"],
            "correction_risk_by_horse_id": r["original_risk_by_horse_id"],
            "ability_score_z_by_horse_id": r["original_ability_z_by_horse_id"],
            "horse_names_and_numbers_for_rankings_attached": False,
            "source_disclosures": {
                "original_oos_scores": True,
                "frozen_l21_physical_tickets": True,
                "l17_top10_p3_else_unknown": True,
                "hindsight_selected_2023_2025": True,
                "2026_unknown_year_sealed": True,
                "historical_return_is_not_future_profit_guarantee": True,
            },
        }
        raw = encode(body)
        check(len(raw) <= MAX_BYTES_PER_RACE, "RACE_OUTPUT_SIZE_GUARD")
        output.append((f"historical/{rid[:4]}/{rid}.json", raw))
    check(sum(len(blob) for _, blob in output) <= MAX_NEW_BYTES_PER_RUN,
          "RUN_OUTPUT_SIZE_GUARD")
    return output


class GithubContents:
    def __init__(self, token):
        check(bool(token), "GITHUB_TOKEN_REQUIRED")
        self.token = token

    def request(self, method, uri, payload=None, accept_missing=False):
        path = "/repos/" + REPO + "/" + uri.lstrip("/")
        url = "https://api.github.com" + path
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + self.token,
            "User-Agent": "KEIBA-ML-Historical-JSON-Publisher",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        data = None if payload is None else encode(payload)
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, method=method, headers=headers, data=data)
        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                return json.loads(res.read(2_000_000))
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and accept_missing:
                return None
            raise ValueError("GITHUB_PUBLISH_HTTP_" + str(exc.code) +
                             "_STOP_NO_FALLBACK") from exc

    def existing(self, path):
        uri = "contents/" + urllib.parse.quote(path, safe="/") + "?ref=" + urllib.parse.quote(DEST_BRANCH, safe="")
        item = self.request("GET", uri, accept_missing=True)
        if item is None:
            return None
        check(isinstance(item, dict) and item.get("type") == "file",
              "PUBLISHED_FILE_METADATA_DRIFT")
        return item

    def tree_size(self):
        ref = self.request("GET", "git/ref/heads/" + DEST_BRANCH)
        check(isinstance(ref, dict) and ref.get("object", {}).get("type") == "commit",
              "OUTPUT_BRANCH_MISSING")
        tree = self.request("GET", "git/trees/" + ref["object"]["sha"] + "?recursive=1")
        check(tree.get("truncated") is False, "OUTPUT_TREE_TRUNCATED_STOP")
        return sum(x.get("size", 0) for x in tree.get("tree", [])
                   if x.get("type") == "blob" and
                   x.get("path", "").startswith("historical/"))

    def put(self, path, data):
        payload = {"message": "data(historical): freeze original L2.1 V3 replay " +
                   path.split("/")[-1].split(".")[0],
                   "branch": DEST_BRANCH,
                   "content": base64.b64encode(data).decode("ascii")}
        return self.request("PUT", "contents/" + urllib.parse.quote(path, safe="/"), payload)


def publish(outputs, client, dry_run=False):
    check(isinstance(outputs, list) and 1 <= len(outputs) <= MAX_RACES_PER_RUN,
          "OUTPUT_COUNT_INVALID")
    existing_size = client.tree_size() if not dry_run else 0
    pending = []
    links = []
    for path, raw in outputs:
        check(path.startswith("historical/") and path.endswith(".json"),
              "UNSAFE_PUBLICATION_PATH")
        item = client.existing(path) if not dry_run else None
        if item:
            check(item.get("size") == len(raw), "PUBLISHED_REPLAY_IMMUTABLE_SIZE_DRIFT")
            expected = hashlib.sha1(("blob " + str(len(raw)) + "\0").encode() + raw).hexdigest()
            check(item.get("sha") == expected, "PUBLISHED_REPLAY_IMMUTABLE_SHA_DRIFT")
        else:
            pending.append((path, raw))
        links.append("https://github.com/" + REPO + "/blob/" + DEST_BRANCH + "/" + path)
    extra = sum(len(raw) for _, raw in pending)
    check(extra <= MAX_NEW_BYTES_PER_RUN, "NEW_PUBLICATION_EXCEEDS_RUN_CAP")
    check(existing_size + extra <= MAX_PUBLISHED_HISTORY_BYTES,
          "FREE_GIT_STORAGE_GUARD_STOP_BEFORE_WRITE")
    if not dry_run:
        for path, raw in pending:
            client.put(path, raw)
            print("PUBLISHED_IMMUTABLE_HISTORICAL_JSON", path,
                  len(raw), hashlib.sha256(raw).hexdigest(), flush=True)
    for link in links:
        print("HISTORICAL_RESULT_OPEN_URL", link, flush=True)
    return links


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--input", required=True)
    cli.add_argument("--dry-run", action="store_true")
    args = cli.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = validate_replay(data)
    if args.dry_run:
        for path, blob in rows:
            print("HISTORICAL_RESULT_DRY_RUN", path, len(blob), flush=True)
    else:
        publish(rows, GithubContents(os.environ.get("GITHUB_TOKEN")))


if __name__ == "__main__":
    main()
