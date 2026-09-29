#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKED_YEAR = 2026
EXPECTED_GENERATION = "eb13d4096519167b"


def parse_args():
    p = argparse.ArgumentParser(description="Run all L1 outsiders for one validation year using one local data restore.")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--inputs", required=True, help="Comma-separated rolling three-year snapshot paths")
    p.add_argument("--validation-snapshot", required=True)
    p.add_argument("--router-seven", required=True)
    p.add_argument("--horse-config", default="research/l1-outsider-arena-v1.json")
    p.add_argument("--nonhorse-config", default="research/l1-outsider-nonhorse-v1.json")
    p.add_argument("--transition-config", default="research/l1-outsider-transition-v1.json")
    p.add_argument("--out-root", default="out/outsider-year-lane-v2")
    p.add_argument("--ml-source-sha", default="")
    p.add_argument("--only-candidates", default="", help="Comma-separated candidate names; empty means all")
    p.add_argument("--gate-race-ids", default="", help="CSV containing race_id for compact Top6 export")
    p.add_argument("--top6-output", default="", help="Append compact outsider Top6 rows here")
    p.add_argument("--seven-union-output", default="", help="Write seven-king Top6 union rows here")
    return p.parse_args()


def run(cmd):
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def open_first_row(path):
    p = Path(path)
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"empty snapshot: {p}")


def validate_contract(args, horse_cfg, nonhorse_cfg, transition_cfg):
    year = args.year
    if year >= LOCKED_YEAR or any(year not in cfg["validation_years"] for cfg in (horse_cfg, nonhorse_cfg, transition_cfg)):
        raise RuntimeError(f"invalid/locked validation year: {year}")
    for cfg in (horse_cfg, nonhorse_cfg, transition_cfg):
        if cfg["snapshot_generation"] != EXPECTED_GENERATION:
            raise RuntimeError("snapshot generation mismatch")
        if cfg["training_window_years"] != 2:
            raise RuntimeError("training window must stay at 2 years")
        if cfg["ability_uses_odds"] is not False:
            raise RuntimeError("L1 odds use is forbidden")
        if LOCKED_YEAR not in cfg["locked_years"]:
            raise RuntimeError("2026 lock missing")

    inputs = [Path(x).resolve() for x in args.inputs.split(",") if x.strip()]
    if len(inputs) != 3 or any(not p.is_file() for p in inputs):
        raise RuntimeError(f"expected exactly three snapshot files, got {inputs}")
    valid_snapshot = Path(args.validation_snapshot).resolve()
    router = Path(args.router_seven).resolve()
    if not valid_snapshot.is_file() or not router.is_file():
        raise RuntimeError("validation snapshot or seven-king router missing")

    row = open_first_row(valid_snapshot)
    if int(str((row.get("features") or {}).get("race_date") or row.get("race_date") or "0")[:4]) != year:
        raise RuntimeError("validation snapshot year mismatch")
    return inputs, valid_snapshot, router



def load_gate_race_ids(path):
    if not path:
        return set()
    import csv
    out = set()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            rid = str(row.get("race_id") or "").strip()
            if rid:
                out.add(rid)
    if not out:
        raise RuntimeError(f"empty gate race-id set: {path}")
    return out


def append_candidate_top6(score_path, gate_ids, candidate, label_ja, year, output_path):
    import csv
    rows = {}
    opener = gzip.open if str(score_path).endswith(".gz") else open
    with opener(score_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = str(rec.get("race_id") or "")
            if rid not in gate_ids:
                continue
            rank = int(rec.get("predicted_rank") or 999)
            if rank > 6:
                continue
            rows.setdefault(rid, []).append({
                "rank": rank,
                "horse_id": str(rec.get("horse_id") or ""),
                "horse_number": rec.get("horse_number"),
                "prob": rec.get("race_normalized_win_probability"),
            })
    missing = sorted(gate_ids - set(rows))
    if missing:
        raise RuntimeError(f"candidate Top6 missing gate races candidate={candidate} missing={missing[:10]} count={len(missing)}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists() and out.stat().st_size > 0
    fields = [
        "year","race_id","candidate","label_ja","top6_horse_ids","top6_horse_numbers",
        "top6_probabilities","top6_count",
    ]
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            w.writeheader()
        for rid in sorted(rows):
            z = sorted(rows[rid], key=lambda x: (x["rank"], x["horse_id"]))
            w.writerow({
                "year": year,
                "race_id": rid,
                "candidate": candidate,
                "label_ja": label_ja,
                "top6_horse_ids": "|".join(x["horse_id"] for x in z),
                "top6_horse_numbers": "|".join("" if x["horse_number"] is None else str(x["horse_number"]) for x in z),
                "top6_probabilities": "|".join("" if x["prob"] is None else f'{float(x["prob"]):.12g}' for x in z),
                "top6_count": len(z),
            })


def write_seven_union(router_path, gate_ids, year, output_path):
    import csv
    rows = {}
    opener = gzip.open if str(router_path).endswith(".gz") else open
    with opener(router_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = str(rec.get("race_id") or "")
            if rid not in gate_ids:
                continue
            experts = rec.get("experts") or {}
            if len(experts) != 7:
                raise RuntimeError(f"expected seven experts race_id={rid}, got={len(experts)}")
            union = set()
            for expert in experts.values():
                union.update(str(x) for x in (expert.get("top6_horse_ids") or []) if str(x))
            if not union:
                raise RuntimeError(f"empty seven-king Top6 union race_id={rid}")
            rows[rid] = sorted(union)
    missing = sorted(gate_ids - set(rows))
    if missing:
        raise RuntimeError(f"seven-union missing gate races: {missing[:10]} count={len(missing)}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fields = ["year","race_id","seven_union_horse_ids","seven_union_count"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for rid in sorted(rows):
            ids = rows[rid]
            w.writerow({
                "year": year,
                "race_id": rid,
                "seven_union_horse_ids": "|".join(ids),
                "seven_union_count": len(ids),
            })


def emit_result(candidate, family, label_ja, mode, year, train_start, train_end, out_dir):
    summary = json.loads((out_dir / "feature-arena-summary.json").read_text(encoding="utf-8"))
    result = summary["results"][0]
    rescue = json.loads((out_dir / "rescue.json").read_text(encoding="utf-8"))
    payload = {
        "candidate": candidate,
        "family": family,
        "label_ja": label_ja,
        "mode": mode,
        "validation_year": year,
        "train_start": train_start,
        "train_end": train_end,
        "feature_sets": result["feature_sets"],
        "feature_count": result["feature_count"],
        "metrics": result["metrics"],
        "resource_usage": result.get("resource_usage"),
        "rescue": rescue,
    }
    print("L1_OUTSIDER_RESULT", flush=True)
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def run_candidate(args, row, family, inputs, valid_snapshot, router, gate_ids):
    year = args.year
    candidate = row["name"]
    label_ja = row.get("label_ja") or candidate
    mode = row.get("mode")
    train_start = f"{year-2}-01-01"
    train_end = f"{year-1}-12-31"
    valid_start = f"{year}-01-01"
    valid_end = f"{year}-12-31"

    lane_root = Path(args.out_root).resolve() / str(year)
    work = lane_root / candidate
    custom = lane_root / "derived" / f"{candidate}.jsonl.gz"
    cand_json = lane_root / "candidate-json" / f"{candidate}.json"
    cand_json.parent.mkdir(parents=True, exist_ok=True)

    if family == "horse":
        feature_sets = row["feature_sets"]
        arena_inputs = ",".join(str(p) for p in inputs)
    else:
        feature_sets = ["BASE"]
        custom.parent.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, "research/build_l1_nonhorse_dataset_v1.py",
            "--inputs", ",".join(str(p) for p in inputs),
            "--mode", mode,
            "--output", custom,
        ])
        if not custom.is_file() or custom.stat().st_size == 0:
            raise RuntimeError(f"nonhorse dataset missing: {custom}")
        arena_inputs = str(custom)

    cand_json.write_text(
        json.dumps([{"name": candidate, "feature_sets": feature_sets}], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"=== OUTSIDER START year={year} family={family} candidate={candidate} ===", flush=True)
    run([
        sys.executable, "research/run_feature_arena.py",
        "--inputs", arena_inputs,
        "--out-dir", work,
        "--train-start", train_start,
        "--train-end", train_end,
        "--valid-start", valid_start,
        "--valid-end", valid_end,
        "--prediction-phase", "FINAL",
        "--feature-selection", "none",
        "--candidates-json", cand_json,
        "--source-ref", "main",
        "--ml-source-sha", args.ml_source_sha or os.environ.get("GITHUB_SHA", ""),
        "--keep-projections",
    ])

    projection = next((p for p in (work / "projections").glob("*.jsonl.gz") if p.is_file() and p.stat().st_size), None)
    model = next((p for p in (work / "models").glob("*.txt") if p.is_file() and p.stat().st_size), None)
    meta = next((p for p in (work / "models").glob("*.json") if p.is_file() and p.stat().st_size), None)
    schema = next((p for p in (work / "schemas").glob("*-schema.json") if p.is_file() and p.stat().st_size), None)
    if not all((projection, model, meta, schema)):
        raise RuntimeError(f"candidate outputs incomplete: {candidate}")

    score = work / "score.jsonl.gz"
    run([
        sys.executable, "research/emit_l1_to_l2.py",
        "--dataset", projection,
        "--model", model,
        "--schema", schema,
        "--meta", meta,
        "--output", score,
        "--candidate-name", candidate,
        "--feature-sets-json", json.dumps(feature_sets, separators=(",", ":")),
        "--valid-start", valid_start,
        "--valid-end", valid_end,
        "--chunk-size", "64",
    ])

    if gate_ids and args.top6_output:
        append_candidate_top6(
            score, gate_ids, candidate, label_ja, year, args.top6_output
        )

    run([
        sys.executable, "research/analyze_l1_outsider_rescue_v1.py",
        "--candidate-score", score,
        "--router-seven", router,
        "--snapshot", valid_snapshot,
        "--candidate-name", candidate,
        "--year", str(year),
        "--output", work / "rescue.json",
    ])

    emit_result(candidate, family, label_ja, mode, year, train_start, train_end, work)
    print(f"=== OUTSIDER COMPLETE year={year} family={family} candidate={candidate} ===", flush=True)

    shutil.rmtree(work, ignore_errors=True)
    if custom.exists():
        custom.unlink()
    cand_json.unlink(missing_ok=True)


def main():
    args = parse_args()
    horse_cfg = json.loads((ROOT / args.horse_config).read_text(encoding="utf-8"))
    nonhorse_cfg = json.loads((ROOT / args.nonhorse_config).read_text(encoding="utf-8"))
    transition_cfg = json.loads((ROOT / args.transition_config).read_text(encoding="utf-8"))
    inputs, valid_snapshot, router = validate_contract(args, horse_cfg, nonhorse_cfg, transition_cfg)

    gate_ids = load_gate_race_ids(args.gate_race_ids) if args.gate_race_ids else set()
    if bool(args.top6_output) != bool(gate_ids):
        raise RuntimeError("--top6-output and --gate-race-ids must be used together")
    if args.seven_union_output:
        if not gate_ids:
            raise RuntimeError("--seven-union-output requires --gate-race-ids")
        write_seven_union(router, gate_ids, args.year, args.seven_union_output)

    candidates = (
        [("horse", x) for x in horse_cfg["candidates"]]
        + [("nonhorse", x) for x in nonhorse_cfg["candidates"]]
        + [("transition", x) for x in transition_cfg["candidates"]]
    )
    only = {x.strip() for x in args.only_candidates.split(",") if x.strip()}
    if only:
        candidates = [(family, row) for family, row in candidates if row.get("name") in only]
        found = {row.get("name") for _, row in candidates}
        missing = sorted(only - found)
        if missing:
            raise RuntimeError(f"unknown --only-candidates: {missing}")
    failures = []
    for family, row in candidates:
        try:
            run_candidate(args, row, family, inputs, valid_snapshot, router, gate_ids)
        except Exception as exc:
            candidate = row.get("name") or "unknown"
            failure = {
                "candidate": candidate,
                "family": family,
                "validation_year": args.year,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print("L1_OUTSIDER_FAILURE", flush=True)
            print(json.dumps(failure, ensure_ascii=False, separators=(",", ":")), flush=True)
            traceback.print_exc()
            lane_root = Path(args.out_root).resolve() / str(args.year)
            shutil.rmtree(lane_root / candidate, ignore_errors=True)
            custom = lane_root / "derived" / f"{candidate}.jsonl.gz"
            if custom.exists():
                custom.unlink()

    print("L1_OUTSIDER_YEAR_LANE_DONE", flush=True)
    print(json.dumps({
        "validation_year": args.year,
        "candidate_count": len(candidates),
        "failure_count": len(failures),
        "failures": failures,
        "race_id_policy": "preserve rescue and four-way race IDs",
    }, ensure_ascii=False, separators=(",", ":")), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
