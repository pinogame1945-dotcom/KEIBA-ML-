#!/usr/bin/env python3
"""One pass over an authenticated yearly Kaggle snapshot for MANY race IDs.

Only source features, never result labels / final odds. Each output is the
existing single-race scorer's JSONL contract; errors fail closed.
"""
from __future__ import annotations
import argparse
import gzip
import json
import re
from pathlib import Path

PATTERN = re.compile(r"202[2-5][0-9]{8}")

def extract_many(source: Path, race_ids: list[str], out_root: Path) -> dict[str, int]:
    if not race_ids or len(race_ids) > 100 or len(set(race_ids)) != len(race_ids):
        raise ValueError("INVALID_OR_DUPLICATE_BATCH_IDS")
    if any(PATTERN.fullmatch(rid) is None for rid in race_ids):
        raise ValueError("2022_TO_2025_ONLY_2026_SEALED")
    if len({rid[:4] for rid in race_ids}) != 1:
        raise ValueError("BATCH_EXTRACTOR_REQUIRES_ONE_YEAR")
    if not source.is_file():
        raise ValueError("MISSING_METADATA_RESOLVED_SNAPSHOT")
    selected: dict[str, dict[str, dict]] = {rid: {} for rid in race_ids}
    op = gzip.open if source.suffix == ".gz" else open
    with op(source, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get("race_id") or "")
            if rid not in selected:
                continue
            hid = str(row.get("horse_id") or "")
            if not hid or hid in selected[rid]:
                raise ValueError("DUPLICATE_OR_MISSING_HORSE_ID_" + rid)
            if (row.get("feature_schema_version") != 8 or
                    row.get("leakage_policy") != "STRICT_PRIOR_DATE_ONLY"):
                raise ValueError("SNAPSHOT_SCHEMA_OR_LEAKAGE_DRIFT_" + rid)
            features = row.get("features")
            if not isinstance(features, dict) or not features:
                raise ValueError("MISSING_PRIOR_ONLY_FEATURES_" + rid)
            selected[rid][hid] = {
                "race_id": rid,
                "horse_id": hid,
                "feature_schema_version": 8,
                "leakage_policy": "STRICT_PRIOR_DATE_ONLY",
                "features": features,
            }
    counts: dict[str, int] = {}
    # Write nothing unless *every* requested race has valid complete rows.
    for rid, horses in selected.items():
        if not 3 <= len(horses) <= 24:
            raise ValueError("RACE_NOT_FOUND_OR_INVALID_STARTERS_" + rid)
        counts[rid] = len(horses)
    for rid, horses in selected.items():
        dest = out_root / rid / "features.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as out:
            for hid in sorted(horses):
                out.write(json.dumps(horses[hid], ensure_ascii=False, separators=(",", ":")) + "\n")
        tmp.replace(dest)
    return counts

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--ids-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    ids = [x.strip() for x in Path(args.ids_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    result = extract_many(Path(args.input), ids, Path(args.output_dir))
    print("BATCH_YEAR_SNAPSHOT_EXTRACTED", json.dumps(result, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
