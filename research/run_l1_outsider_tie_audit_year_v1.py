#!/usr/bin/env python3
import argparse, csv, gzip, hashlib, json, math, random, shutil, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPNS = (1, 3, 6)
SHUFFLE_TRIALS = 5
EXPECTED_GENERATION = "eb13d4096519167b"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--validation-snapshot", required=True)
    p.add_argument("--router-seven", required=True)
    p.add_argument("--nonhorse-config", default="research/l1-outsider-nonhorse-v1.json")
    p.add_argument("--transition-config", default="research/l1-outsider-transition-v1.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--work-root", default="out/outsider-tie-audit-v1")
    p.add_argument("--ml-source-sha", default="")
    return p.parse_args()

def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "rt", encoding="utf-8")

def run(cmd):
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

def finite(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def load_truth(path):
    winners = defaultdict(set)
    seen = defaultdict(set)
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rid = str(r.get("race_id") or "")
            hid = str(r.get("horse_id") or "")
            if not rid or not hid:
                continue
            seen[rid].add(hid)
            t = r.get("target") or {}
            is_win = t.get("is_win")
            if is_win is None:
                try:
                    is_win = int(float(t.get("finish_position"))) == 1
                except (TypeError, ValueError):
                    is_win = False
            if is_win:
                winners[rid].add(hid)
    out = {rid: winners[rid] for rid in seen if winners[rid]}
    if not out:
        raise RuntimeError("winner truth empty")
    return out

def load_router(path):
    out = {}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("contract") != "ROUTER_FEATURE_SNAPSHOT_V1":
                raise RuntimeError("bad router contract")
            out[str(r["race_id"])] = r
    return out

def expert_set(view, n):
    if n == 1:
        x = view.get("top1_horse_id")
        return {str(x)} if x else set()
    return {str(x) for x in (view.get(f"top{n}_horse_ids") or []) if str(x)}

def horse_number_key(v):
    if v is None:
        return (1, 999999)
    try:
        return (0, int(float(v)))
    except (TypeError, ValueError):
        return (1, 999999)

def stable_shuffle_order(rows, candidate, year, rid, trial):
    z = list(rows)
    seed_raw = f"{candidate}|{year}|{rid}|{trial}".encode()
    seed = int(hashlib.sha256(seed_raw).hexdigest()[:16], 16)
    random.Random(seed).shuffle(z)
    z.sort(key=lambda r: -r["score"])  # stable: preserves shuffled order inside ties
    return z

def top_set(rows, n, rank_key):
    return {r["horse_id"] for r in rows if int(r[rank_key]) <= n}

def audit_score(score_path, truth, router, candidate, year):
    by = defaultdict(list)
    with open_text(score_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rid = str(r.get("race_id") or "")
            hid = str(r.get("horse_id") or "")
            score = finite(r.get("raw_margin_logit"))
            if score is None:
                score = finite(r.get("raw_win_probability"))
            rank = r.get("predicted_rank")
            if not rid or not hid or score is None or rank is None:
                raise RuntimeError(f"bad score row candidate={candidate}")
            by[rid].append({
                "horse_id": hid,
                "horse_number": r.get("horse_number"),
                "score": score,
                "old_rank": int(rank),
            })

    races = sorted(by)
    common = set(races) & set(truth) & set(router)
    if common != set(races):
        raise RuntimeError(f"coverage mismatch candidate={candidate} score={len(races)} common={len(common)}")

    out = {
        "contract": "L1_OUTSIDER_TIE_AUDIT_V1",
        "candidate": candidate,
        "year": year,
        "races": len(races),
        "rows": sum(len(v) for v in by.values()),
        "any_tie_races": 0,
        "any_rank_changed_races": 0,
        "any_tie_rate": 0.0,
        "any_rank_changed_rate": 0.0,
        "shuffle_trials": SHUFFLE_TRIALS,
        "safe_tie_breaker": "raw_margin_logit desc, horse_number asc, horse_id asc",
        "old_tie_breaker": "stable score order / original row order",
        "topn": {},
        "changed_examples": [],
    }
    stats = {}
    for n in TOPNS:
        stats[n] = {
            "boundary_tie_races": 0,
            "topn_set_changed_races": 0,
            "old_capture": 0,
            "safe_capture": 0,
            "old_only_hits": 0,
            "safe_only_hits": 0,
            "old_rescues": 0,
            "safe_rescues": 0,
            "shuffle_sensitive_races": 0,
        }

    for rid in races:
        rows = by[rid]
        old = sorted(rows, key=lambda r: r["old_rank"])
        expected = list(range(1, len(old) + 1))
        got = [r["old_rank"] for r in old]
        if got != expected:
            raise RuntimeError(f"non-contiguous old ranks candidate={candidate} race={rid}")

        counts = Counter(r["score"] for r in rows)
        has_tie = any(v > 1 for v in counts.values())
        out["any_tie_races"] += int(has_tie)

        safe = sorted(rows, key=lambda r: (-r["score"], horse_number_key(r["horse_number"]), r["horse_id"]))
        for i, r in enumerate(safe, start=1):
            r["safe_rank"] = i
        if [r["horse_id"] for r in old] != [r["horse_id"] for r in safe]:
            out["any_rank_changed_races"] += 1

        winners = truth[rid]
        views = (router[rid].get("experts") or {})
        if len(views) != 7:
            raise RuntimeError(f"expected 7 experts race={rid}")

        for n in TOPNS:
            st = stats[n]
            if len(old) >= n:
                boundary = old[n - 1]["score"]
                tied = [r for r in rows if r["score"] == boundary]
                if len(tied) > 1:
                    tied_old = sorted(r["old_rank"] for r in tied)
                    if min(tied_old) <= n < max(tied_old):
                        st["boundary_tie_races"] += 1

            old_set = {r["horse_id"] for r in old[:n]}
            safe_set = {r["horse_id"] for r in safe[:n]}
            old_hit = bool(winners & old_set)
            safe_hit = bool(winners & safe_set)
            st["old_capture"] += int(old_hit)
            st["safe_capture"] += int(safe_hit)
            st["old_only_hits"] += int(old_hit and not safe_hit)
            st["safe_only_hits"] += int(safe_hit and not old_hit)
            changed = old_set != safe_set
            st["topn_set_changed_races"] += int(changed)

            seven_union = set().union(*(expert_set(v, n) for v in views.values()))
            seven_blind = not bool(winners & seven_union)
            st["old_rescues"] += int(seven_blind and old_hit)
            st["safe_rescues"] += int(seven_blind and safe_hit)

            shuffle_changed = False
            for trial in range(SHUFFLE_TRIALS):
                zz = stable_shuffle_order(rows, candidate, year, rid, trial)
                zset = {r["horse_id"] for r in zz[:n]}
                if zset != old_set:
                    shuffle_changed = True
                    break
            st["shuffle_sensitive_races"] += int(shuffle_changed)

            if changed and len(out["changed_examples"]) < 50:
                out["changed_examples"].append({
                    "race_id": rid,
                    "topn": n,
                    "winner": sorted(winners),
                    "old_top": sorted(old_set),
                    "safe_top": sorted(safe_set),
                })

    total = len(races)
    out["any_tie_rate"] = out["any_tie_races"] / total
    out["any_rank_changed_rate"] = out["any_rank_changed_races"] / total
    for n in TOPNS:
        st = stats[n]
        for key in (
            "boundary_tie_races","topn_set_changed_races","old_capture","safe_capture",
            "old_only_hits","safe_only_hits","old_rescues","safe_rescues","shuffle_sensitive_races"
        ):
            st[key + "_rate"] = st[key] / total
        st["capture_delta"] = st["safe_capture"] - st["old_capture"]
        st["rescue_delta"] = st["safe_rescues"] - st["old_rescues"]
        out["topn"][str(n)] = st
    return out

def main():
    a = parse_args()
    if a.year >= 2026:
        raise SystemExit("2026 sealed")

    cfgs = []
    for p in (a.nonhorse_config, a.transition_config):
        cfg = json.loads((ROOT / p).read_text(encoding="utf-8"))
        if cfg["snapshot_generation"] != EXPECTED_GENERATION:
            raise RuntimeError("snapshot generation mismatch")
        if cfg["training_window_years"] != 2 or cfg["ability_uses_odds"] is not False:
            raise RuntimeError("config contract mismatch")
        if a.year not in cfg["validation_years"] or 2026 not in cfg["locked_years"]:
            raise RuntimeError("year policy mismatch")
        cfgs.append(cfg)

    candidates = []
    for cfg in cfgs:
        candidates.extend(cfg["candidates"])
    names = [x["name"] for x in candidates]
    if len(names) != 13 or len(set(names)) != 13:
        raise RuntimeError(f"expected 13 unique outsiders, got {names}")

    inputs = [Path(x).resolve() for x in a.inputs.split(",") if x.strip()]
    if len(inputs) != 3 or any(not p.is_file() for p in inputs):
        raise RuntimeError("expected three local snapshot inputs")
    valid_snapshot = Path(a.validation_snapshot).resolve()
    router_path = Path(a.router_seven).resolve()
    truth = load_truth(valid_snapshot)
    router = load_router(router_path)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(a.work_root).resolve() / str(a.year)
    work_root.mkdir(parents=True, exist_ok=True)
    results = []

    train_start = f"{a.year - 2}-01-01"
    train_end = f"{a.year - 1}-12-31"
    valid_start = f"{a.year}-01-01"
    valid_end = f"{a.year}-12-31"

    for row in candidates:
        candidate = row["name"]
        mode = row["mode"]
        print(f"=== TIE AUDIT START year={a.year} candidate={candidate} mode={mode} ===", flush=True)
        custom = work_root / f"{candidate}.jsonl.gz"
        work = work_root / candidate
        cand_json = work_root / f"{candidate}.candidate.json"

        run([
            sys.executable, "research/build_l1_nonhorse_dataset_v1.py",
            "--inputs", ",".join(str(p) for p in inputs),
            "--mode", mode,
            "--output", custom,
        ])
        cand_json.write_text(
            json.dumps([{"name": candidate, "feature_sets": ["BASE"]}], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        run([
            sys.executable, "research/run_feature_arena.py",
            "--inputs", custom,
            "--out-dir", work,
            "--train-start", train_start,
            "--train-end", train_end,
            "--valid-start", valid_start,
            "--valid-end", valid_end,
            "--prediction-phase", "FINAL",
            "--feature-selection", "none",
            "--candidates-json", cand_json,
            "--source-ref", "research/l1-outsider-tie-audit-v1-20260929",
            "--ml-source-sha", a.ml_source_sha,
            "--keep-projections",
        ])

        projection = next((p for p in (work / "projections").glob("*.jsonl.gz") if p.stat().st_size), None)
        model = next((p for p in (work / "models").glob("*.txt") if p.stat().st_size), None)
        meta = next((p for p in (work / "models").glob("*.json") if p.stat().st_size), None)
        schema = next((p for p in (work / "schemas").glob("*-schema.json") if p.stat().st_size), None)
        if not all((projection, model, meta, schema)):
            raise RuntimeError(f"missing model outputs candidate={candidate}")

        score = work / "score.jsonl.gz"
        run([
            sys.executable, "research/emit_l1_to_l2.py",
            "--dataset", projection,
            "--model", model,
            "--schema", schema,
            "--meta", meta,
            "--output", score,
            "--candidate-name", candidate,
            "--feature-sets-json", '["BASE"]',
            "--valid-start", valid_start,
            "--valid-end", valid_end,
            "--chunk-size", "64",
        ])

        audit = audit_score(score, truth, router, candidate, a.year)
        audit["label_ja"] = row.get("label_ja") or candidate
        audit["mode"] = mode
        p = out_dir / f"{candidate}.json"
        p.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        results.append(audit)
        print("L1_OUTSIDER_TIE_AUDIT_RESULT " + json.dumps({
            "candidate": candidate,
            "year": a.year,
            "races": audit["races"],
            "any_tie_races": audit["any_tie_races"],
            "rank_changed_races": audit["any_rank_changed_races"],
            "top1_delta": audit["topn"]["1"]["capture_delta"],
            "top6_rescue_delta": audit["topn"]["6"]["rescue_delta"],
        }, ensure_ascii=False, separators=(",", ":")), flush=True)

        shutil.rmtree(work, ignore_errors=True)
        custom.unlink(missing_ok=True)
        cand_json.unlink(missing_ok=True)

    summary = {
        "contract": "L1_OUTSIDER_TIE_AUDIT_YEAR_V1",
        "year": a.year,
        "candidate_count": len(results),
        "candidates": [x["candidate"] for x in results],
        "races_per_candidate": sorted({x["races"] for x in results}),
        "safe_tie_breaker": "raw_margin_logit desc, horse_number asc, horse_id asc",
        "shuffle_trials": SHUFFLE_TRIALS,
        "2026_sealed": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("L1_OUTSIDER_TIE_AUDIT_YEAR_OK " + json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)

if __name__ == "__main__":
    main()
