#!/usr/bin/env python3
import argparse
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURE_CONTRACT_PATH = ROOT / "contracts" / "l1-feature-set-contract-v1.json"
LOCKED_RESEARCH_YEAR = 2026
EXPECTED_DATASET_VERSION = 3
EXPECTED_FEATURE_SCHEMA_VERSION = 8
EXPECTED_LEAKAGE_POLICY = "STRICT_PRIOR_DATE_ONLY"

DEFAULT_CANDIDATE_NAMES = [
    ("base", ["BASE"]),
    ("opponent", ["BASE", "OPPONENT"]),
    ("network", ["BASE", "NETWORK"]),
    ("lap", ["BASE", "LAP"]),
    ("style", ["BASE", "STYLE"]),
    ("distance", ["BASE", "DISTANCE"]),
    ("backfill", ["BASE", "BACKFILL"]),
    ("auto", ["BASE", "AUTO"]),
    ("pedigree", ["BASE", "PEDIGREE"]),
    ("actor", ["BASE", "ACTOR"]),
    ("time_pace", ["BASE", "TIME_PACE"]),
]

METRIC_DIRECTIONS = {
    "top1_winner_capture": "desc",
    "top3_winner_capture": "desc",
    "top6_winner_capture": "desc",
    "mean_winner_rank": "asc",
    "mean_reciprocal_winner_rank": "desc",
    "race_normalized_nll": "asc",
    "raw_logloss": "asc",
    "raw_brier": "asc",
    "raw_roc_auc": "desc",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the L1 WIN Feature Arena from schema-8 yearly superset snapshots."
    )
    p.add_argument("--inputs", required=True, help="Comma-separated snapshot-YYYY.jsonl.gz files")
    p.add_argument("--out-dir", default="out/feature-arena")
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--valid-start", required=True)
    p.add_argument("--valid-end", required=True)
    p.add_argument("--train-race-class", default="ALL")
    p.add_argument("--valid-race-class", default="ALL")
    p.add_argument("--train-surface", choices=["ALL","TURF","DIRT"], default="ALL")
    p.add_argument("--valid-surface", choices=["ALL","TURF","DIRT"], default="ALL")
    p.add_argument("--prediction-phase", choices=["EARLY", "FINAL"], default="FINAL")
    p.add_argument(
        "--candidates-json",
        help='Optional JSON file: [{"name":"base","feature_sets":["BASE"]}, ...]',
    )
    p.add_argument("--feature-selection", choices=["none", "train_v1"], default="none")
    p.add_argument("--fs-max-missing-rate", type=float, default=0.98)
    p.add_argument("--fs-max-correlation", type=float, default=0.995)
    p.add_argument("--fs-min-inner-gain-fraction", type=float, default=0.0)
    p.add_argument("--fs-inner-valid-fraction", type=float, default=0.20)
    p.add_argument("--source-repo", default="pinogame1945-dotcom/KEIBA-BACKFILL")
    p.add_argument("--source-ref", default="main")
    p.add_argument("--source-sha")
    p.add_argument("--ml-source-sha")
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--keep-projections", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    return p.parse_args()


def _proc_tree_rss_kib(root_pid):
    root_pid = int(root_pid)
    rows = {}
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return 0
    for child in proc_root.iterdir():
        if not child.name.isdigit():
            continue
        try:
            status = (child / "status").read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            continue
        ppid = None
        rss = 0
        for line in status.splitlines():
            if line.startswith("PPid:"):
                ppid = int(line.split()[1])
            elif line.startswith("VmRSS:"):
                rss = int(line.split()[1])
        if ppid is not None:
            rows[int(child.name)] = (ppid, rss)

    children = {}
    for pid, (ppid, _rss) in rows.items():
        children.setdefault(ppid, []).append(pid)

    total = 0
    stack = [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        row = rows.get(pid)
        if row:
            total += row[1]
        stack.extend(children.get(pid, []))
    return total


def run(cmd, label):
    command = [str(x) for x in cmd]
    print("+", " ".join(command), flush=True)
    started = time.monotonic()
    proc = subprocess.Popen(command, cwd=ROOT)
    peak_kib = 0
    last_log = -15.0
    while True:
        elapsed = time.monotonic() - started
        rss_kib = _proc_tree_rss_kib(proc.pid)
        peak_kib = max(peak_kib, rss_kib)
        if elapsed - last_log >= 15.0:
            print(
                "FEATURE_ARENA_RESOURCE_SAMPLE "
                + json.dumps(
                    {
                        "label": label,
                        "elapsed_seconds": round(elapsed, 1),
                        "rss_mib": round(rss_kib / 1024.0, 1),
                        "peak_rss_mib": round(peak_kib / 1024.0, 1),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            last_log = elapsed
        code = proc.poll()
        if code is not None:
            break
        time.sleep(1.0)

    elapsed = time.monotonic() - started
    usage = {
        "label": label,
        "elapsed_seconds": round(elapsed, 3),
        "peak_rss_mib": round(peak_kib / 1024.0, 3),
        "exit_code": int(code),
    }
    print(
        "FEATURE_ARENA_RESOURCE_USAGE "
        + json.dumps(usage, separators=(",", ":")),
        flush=True,
    )
    if code != 0:
        raise subprocess.CalledProcessError(code, command)
    return usage


def load_contract():
    return json.loads(FEATURE_CONTRACT_PATH.read_text(encoding="utf-8"))


def normalize_sets(raw, contract):
    canonical = list(contract["feature_sets"])
    requested = {str(x).strip().upper() for x in raw if str(x).strip()}
    requested.add("BASE")
    invalid = sorted(requested - set(canonical))
    if invalid:
        raise ValueError("invalid feature set(s): " + ", ".join(invalid))
    return [name for name in canonical if name in requested]


def default_candidates(contract):
    rows = [
        {"name": name, "feature_sets": normalize_sets(sets, contract)}
        for name, sets in DEFAULT_CANDIDATE_NAMES
    ]
    rows.append({"name": "all", "feature_sets": list(contract["feature_sets"])})
    return rows


def load_candidates(path, contract):
    if not path:
        rows = default_candidates(contract)
    else:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("--candidates-json must contain a non-empty JSON array")
        normalized = []
        for item in rows:
            if not isinstance(item, dict):
                raise ValueError("candidate entries must be JSON objects")
            name = str(item.get("name") or "").strip()
            sets = item.get("feature_sets")
            if not name or not isinstance(sets, list) or not sets:
                raise ValueError("candidate requires name and non-empty feature_sets")
            actor_prefixes = item.get("actor_prefixes") or []
            if not isinstance(actor_prefixes, list):
                raise ValueError("candidate actor_prefixes must be a JSON array")
            allowed_actor_prefixes = {"actor_jockey_", "actor_trainer_", "actor_horse_jockey_"}
            actor_prefixes = [str(x).strip() for x in actor_prefixes if str(x).strip()]
            invalid_actor_prefixes = sorted(set(actor_prefixes) - allowed_actor_prefixes)
            if invalid_actor_prefixes:
                raise ValueError("invalid actor_prefixes: " + ", ".join(invalid_actor_prefixes))
            auto_slices = item.get("auto_slices") or []
            if not isinstance(auto_slices, list):
                raise ValueError("candidate auto_slices must be a JSON array")
            allowed_auto_slices = {"ROLLING", "CONDITION", "FIELD", "PAIR"}
            auto_slices = [str(x).strip().upper() for x in auto_slices if str(x).strip()]
            invalid_auto_slices = sorted(set(auto_slices) - allowed_auto_slices)
            if invalid_auto_slices:
                raise ValueError("invalid auto_slices: " + ", ".join(invalid_auto_slices))
            pedigree_slices = item.get("pedigree_slices") or []
            if not isinstance(pedigree_slices, list):
                raise ValueError("candidate pedigree_slices must be a JSON array")
            allowed_pedigree_slices = {"LEGACY", "RACE_CLASS"}
            pedigree_slices = [str(x).strip().upper() for x in pedigree_slices if str(x).strip()]
            invalid_pedigree_slices = sorted(set(pedigree_slices) - allowed_pedigree_slices)
            if invalid_pedigree_slices:
                raise ValueError("invalid pedigree_slices: " + ", ".join(invalid_pedigree_slices))
            feature_sets = normalize_sets(sets, contract)
            if actor_prefixes and "ACTOR" not in feature_sets:
                raise ValueError("actor_prefixes requires ACTOR feature set")
            if auto_slices and "AUTO" not in feature_sets:
                raise ValueError("auto_slices requires AUTO feature set")
            if pedigree_slices and "PEDIGREE" not in feature_sets:
                raise ValueError("pedigree_slices requires PEDIGREE feature set")
            normalized.append({
                "name": name,
                "feature_sets": feature_sets,
                "actor_prefixes": actor_prefixes,
                "auto_slices": auto_slices,
                "pedigree_slices": pedigree_slices,
            })
        rows = normalized

    seen_names = set()
    seen_sets = set()
    out = []
    for row in rows:
        name = row["name"]
        if name in seen_names:
            raise ValueError("duplicate candidate name: " + name)
        signature = (
            tuple(row["feature_sets"]),
            tuple(sorted(row.get("actor_prefixes") or [])),
            tuple(sorted(row.get("auto_slices") or [])),
            tuple(sorted(row.get("pedigree_slices") or [])),
        )
        if signature in seen_sets:
            raise ValueError("duplicate candidate feature set: " + ",".join(signature))
        seen_names.add(name)
        seen_sets.add(signature)
        row.setdefault("actor_prefixes", [])
        row.setdefault("auto_slices", [])
        row.setdefault("pedigree_slices", [])
        out.append(row)
    return out


def parse_inputs(raw):
    paths = [Path(x.strip()).resolve() for x in str(raw).split(",") if x.strip()]
    if not paths:
        raise ValueError("--inputs is empty")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def first_row(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"empty snapshot input: {path}")


def source_contract(inputs):
    rows = [first_row(path) for path in inputs]
    first = rows[0]
    keys = [
        "ml_dataset_version",
        "feature_schema_version",
        "leakage_policy",
        "prediction_phase",
        "feature_sets",
        "history_windows",
        "small_sample_policy",
    ]
    for path, row in zip(inputs[1:], rows[1:]):
        for key in keys:
            if row.get(key) != first.get(key):
                raise ValueError(f"snapshot contract mismatch for {key}: {path}")

    if int(first.get("ml_dataset_version", 0)) != EXPECTED_DATASET_VERSION:
        raise ValueError("snapshot ml_dataset_version mismatch")
    if int(first.get("feature_schema_version", 0)) != EXPECTED_FEATURE_SCHEMA_VERSION:
        raise ValueError("snapshot feature_schema_version mismatch")
    if first.get("leakage_policy") != EXPECTED_LEAKAGE_POLICY:
        raise ValueError("snapshot leakage_policy mismatch")
    source_phase = str(first.get("prediction_phase") or "").upper()
    if source_phase not in {"EARLY", "FINAL"}:
        raise ValueError("snapshot prediction_phase is missing or invalid")
    source_sets = [str(x).upper() for x in (first.get("feature_sets") or [])]
    if "BASE" not in source_sets:
        raise ValueError("snapshot feature_sets must include BASE")
    history = first.get("history_windows")
    small_sample = first.get("small_sample_policy")
    if not isinstance(history, dict) or not isinstance(small_sample, dict):
        raise ValueError("snapshot history/small-sample contract missing")
    return {
        "ml_dataset_version": EXPECTED_DATASET_VERSION,
        "feature_schema_version": EXPECTED_FEATURE_SCHEMA_VERSION,
        "leakage_policy": EXPECTED_LEAKAGE_POLICY,
        "prediction_phase": source_phase,
        "feature_sets": source_sets,
        "history_windows": history,
        "small_sample_policy": small_sample,
    }


def check_dates(a):
    from datetime import date
    train_start = date.fromisoformat(a.train_start)
    train_end = date.fromisoformat(a.train_end)
    valid_start = date.fromisoformat(a.valid_start)
    valid_end = date.fromisoformat(a.valid_end)
    if train_start > train_end:
        raise ValueError("train_start must be <= train_end")
    if valid_start > valid_end:
        raise ValueError("valid_start must be <= valid_end")
    if train_end >= valid_start:
        raise ValueError("train_end must be before valid_start")
    for label, value in [
        ("train_start", train_start),
        ("train_end", train_end),
        ("valid_start", valid_start),
        ("valid_end", valid_end),
    ]:
        if value.year >= LOCKED_RESEARCH_YEAR:
            raise ValueError(f"2026 research lock: {label}={value.isoformat()} is forbidden")


def validate_candidates(candidates, source):
    available = set(source["feature_sets"])
    for candidate in candidates:
        missing = set(candidate["feature_sets"]) - available
        if missing:
            raise ValueError(
                f"candidate {candidate['name']} needs feature sets absent from snapshot: "
                + ", ".join(sorted(missing))
            )


def metric_views(results):
    views = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        available = [
            {
                "candidate": row["name"],
                "feature_sets": row["feature_sets"],
                "value": row["metrics"].get(metric),
            }
            for row in results
            if row["metrics"].get(metric) is not None
        ]
        available.sort(
            key=lambda row: row["value"],
            reverse=(direction == "desc"),
        )
        views[metric] = {"direction": direction, "rows": available}
    return views


def deltas_vs_base(results):
    base = next((row for row in results if row["feature_sets"] == ["BASE"]), None)
    if base is None:
        return None
    out = {}
    for row in results:
        delta = {}
        for metric in METRIC_DIRECTIONS:
            current = row["metrics"].get(metric)
            baseline = base["metrics"].get(metric)
            if current is not None and baseline is not None:
                delta[metric] = float(current) - float(baseline)
        out[row["name"]] = delta
    return {"baseline_candidate": base["name"], "rows": out}


def safe_slug(value):
    out = []
    for ch in str(value).lower():
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    if not slug:
        raise ValueError("candidate name produced an empty slug")
    return slug


def main():
    a = parse_args()
    check_dates(a)
    contract = load_contract()
    inputs = parse_inputs(a.inputs)
    source = source_contract(inputs)
    candidates = load_candidates(a.candidates_json, contract)
    validate_candidates(candidates, source)

    requested_phase = a.prediction_phase.upper()
    if source["prediction_phase"] == "EARLY" and requested_phase == "FINAL":
        raise ValueError("cannot promote an EARLY snapshot to FINAL availability")
    if a.feature_selection == "train_v1":
        if not (0 <= a.fs_max_missing_rate <= 1):
            raise ValueError("--fs-max-missing-rate must be from 0 to 1")
        if not (0 <= a.fs_max_correlation <= 1):
            raise ValueError("--fs-max-correlation must be from 0 to 1")
        if a.fs_min_inner_gain_fraction < 0:
            raise ValueError("--fs-min-inner-gain-fraction must be >= 0")
        if not (0.05 <= a.fs_inner_valid_fraction <= 0.50):
            raise ValueError("--fs-inner-valid-fraction must be from 0.05 to 0.50")

    plan = {
        "contract": "L1_FEATURE_ARENA_WIN_V1",
        "model_family": "WIN_BINARY_LIGHTGBM",
        "ability_uses_odds": False,
        "source_snapshot_contract": source,
        "prediction_phase": requested_phase,
        "train": {
            "start": a.train_start,
            "end": a.train_end,
            "race_class": str(a.train_race_class).strip().upper(),
        },
        "validation": {
            "start": a.valid_start,
            "end": a.valid_end,
            "race_class": str(a.valid_race_class).strip().upper(),
        },
        "feature_selection": a.feature_selection,
        "candidates": candidates,
        "combined_winner_score": None,
        "note": "Arena reports per-metric views; it does not invent a single composite winner.",
    }
    print("L1_FEATURE_ARENA_PLAN")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if a.plan_only:
        return

    out = Path(a.out_dir).resolve()
    projections = out / "projections"
    models = out / "models"
    schemas = out / "schemas"
    oof = out / "oof"
    diagnostics = out / "diagnostics"
    for path in (projections, models, schemas, oof, diagnostics):
        path.mkdir(parents=True, exist_ok=True)

    results = []
    input_arg = ",".join(str(path) for path in inputs)
    for candidate in candidates:
        name = candidate["name"]
        slug = safe_slug(name)
        sets = candidate["feature_sets"]
        actor_prefixes = candidate.get("actor_prefixes") or []
        auto_slices = candidate.get("auto_slices") or []
        pedigree_slices = candidate.get("pedigree_slices") or []
        sets_arg = ",".join(sets)
        projected = projections / f"{slug}.jsonl.gz"
        model = models / f"{slug}.txt"
        meta = models / f"{slug}.json"
        schema = schemas / f"{slug}-schema.json"
        pred = oof / f"{slug}.jsonl.gz"

        candidate_started = time.monotonic()
        projection_cmd = [
            "node", "research/project-yearly-snapshots.mjs",
            "--inputs", input_arg,
            "--output", projected,
            "--feature-sets", sets_arg,
            "--prediction-phase", requested_phase,
        ]
        if actor_prefixes:
            projection_cmd.extend(["--actor-prefixes", ",".join(actor_prefixes)])
        if auto_slices:
            projection_cmd.extend(["--auto-slices", ",".join(auto_slices)])
        if pedigree_slices:
            projection_cmd.extend(["--pedigree-slices", ",".join(pedigree_slices)])
        projection_usage = run(projection_cmd, label=f"{name}:projection")

        cmd = [
            sys.executable, "research/train-staged-lightgbm.py",
            "--dataset", projected,
            "--feature-sets", sets_arg,
            "--prediction-phase", requested_phase,
            "--history-windows-json", json.dumps(source["history_windows"], separators=(",", ":")),
            "--small-sample-policy-json", json.dumps(source["small_sample_policy"], separators=(",", ":")),
            "--feature-selection", a.feature_selection,
            "--fs-max-missing-rate", a.fs_max_missing_rate,
            "--fs-max-correlation", a.fs_max_correlation,
            "--fs-min-inner-gain-fraction", a.fs_min_inner_gain_fraction,
            "--fs-inner-valid-fraction", a.fs_inner_valid_fraction,
            "--train-start", a.train_start,
            "--train-end", a.train_end,
            "--valid-start", a.valid_start,
            "--valid-end", a.valid_end,
            "--train-race-class", a.train_race_class,
            "--valid-race-class", a.valid_race_class,
            "--train-surface", a.train_surface,
            "--valid-surface", a.valid_surface,
            "--model-out", model,
            "--meta-out", meta,
            "--schema-out", schema,
            "--predictions-out", pred,
            "--model-version", f"L1_WIN_ARENA_{slug.upper()}_{requested_phase}",
            "--source-repo", a.source_repo,
            "--source-ref", a.source_ref,
        ]
        if a.source_sha:
            cmd.extend(["--source-sha", a.source_sha])
        if a.ml_source_sha:
            cmd.extend(["--ml-source-sha", a.ml_source_sha])
        if a.diagnostics:
            cmd.extend([
                "--diagnostics-out", diagnostics / f"{slug}.json.gz",
                "--contributions-out", diagnostics / f"{slug}-contributions.jsonl.gz",
            ])
        training_usage = run(cmd, label=f"{name}:training")

        metadata = json.loads(meta.read_text(encoding="utf-8"))
        results.append({
            "name": name,
            "feature_sets": sets,
            "actor_prefixes": actor_prefixes,
            "auto_slices": auto_slices,
            "pedigree_slices": pedigree_slices,
            "race_scope": {
                "train": str(a.train_race_class).strip().upper(),
                "valid": str(a.valid_race_class).strip().upper(),
            },
            "surface_scope": {
                "train": str(a.train_surface).strip().upper(),
                "valid": str(a.valid_surface).strip().upper(),
            },
            "feature_count": metadata["feature_count"],
            "metrics": metadata["metrics"],
            "best_iteration": metadata["best_iteration"],
            "model": str(model),
            "metadata": str(meta),
            "schema": str(schema),
            "oof": str(pred),
            "resource_usage": {
                "projection": projection_usage,
                "training": training_usage,
                "candidate_elapsed_seconds": round(time.monotonic() - candidate_started, 3),
                "candidate_peak_rss_mib": max(
                    projection_usage["peak_rss_mib"],
                    training_usage["peak_rss_mib"],
                ),
            },
        })
        if not a.keep_projections:
            projected.unlink(missing_ok=True)

    summary = {
        **plan,
        "results": results,
        "metric_views": metric_views(results),
        "delta_vs_base": deltas_vs_base(results),
    }
    summary_path = out / "feature-arena-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("L1_FEATURE_ARENA_DONE")
    print(json.dumps({
        "summary": str(summary_path),
        "candidates": len(results),
        "metric_views": summary["metric_views"],
        "delta_vs_base": summary["delta_vs_base"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
