#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

EXPERTS = (
    "core4_no_pedigree",
    "no_auto_full_pedigree_legacy",
    "jockey_trainer_condition",
    "no_auto_full",
    "no_auto_jockey",
)
FEATURE_CONTRACT = "L15_ROLE_CANDIDATE_FEATURES_V2"
DECISION_CONTRACT = "L15_COUNCIL_ROUTER_V3_DECISION"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True, choices=(2023, 2024, 2025))
    p.add_argument("--decision", action="append", required=True, help="ROLE:PATH")
    p.add_argument("--year-feature", required=True)
    p.add_argument("--year-pred", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()

def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "rt", encoding="utf-8")

def finite(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def flatten(prefix, obj, out, keep_strings=True):
    if not isinstance(obj, dict):
        return
    for k, v in obj.items():
        key = f"{prefix}_{k}"
        if isinstance(v, bool):
            out[key] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            x = finite(v)
            if x is not None:
                out[key] = x
        elif isinstance(v, str) and keep_strings:
            out[key] = v
        elif isinstance(v, dict):
            flatten(key, v, out, keep_strings=keep_strings)

def jaccard(a, b):
    a, b = set(a), set(b)
    u = a | b
    return len(a & b) / len(u) if u else 1.0

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def stdev(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0

def classify(d):
    b = int(d["baseline_hit"])
    c = int(d["chosen_hit"])
    o = int(d["oracle_hit"])
    if b == 0 and c == 1:
        return "RESCUE"
    if b == 1 and c == 0:
        return "REGRESSION"
    if b == 1 and c == 1:
        return "BOTH_HIT"
    if o == 1:
        return "MISSED_RESCUE"
    return "DEAD_ZONE"

def read_decisions(year, specs):
    out = {}
    for spec in specs:
        role, path = spec.split(":", 1)
        with open_text(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("contract") != DECISION_CONTRACT:
                    raise ValueError(f"unexpected decision contract: {path}")
                if int(d["test_year"]) != year or str(d["role"]) != role:
                    continue
                key = (str(d["race_id"]), role, int(d["top_n"]))
                if key in out:
                    raise ValueError(f"duplicate decision: {key}")
                d["classification"] = classify(d)
                d["intervened"] = int(str(d["chosen_strategy"]) != "router_hard")
                out[key] = d
    return out

def read_predictions(path):
    out = defaultdict(dict)
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            expert = str(r["expert_name"])
            if expert not in EXPERTS:
                continue
            key = (str(r["race_id"]), str(r["role"]), int(float(r["top_n"])))
            out[key][expert] = float(r["predicted_role_hit_probability"])
    return out

def iter_feature_groups(path):
    current = None
    groups = defaultdict(dict)
    seen = set()
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("contract") != FEATURE_CONTRACT:
                raise ValueError(f"unexpected feature contract in {path}")
            expert = str(row["expert_name"])
            if expert not in EXPERTS:
                continue
            rid = str(row["race_id"])
            if current is None:
                current = rid
            if rid != current:
                if rid in seen:
                    raise ValueError(f"race_id reappeared after flush: {rid}")
                seen.add(current)
                yield current, groups
                current = rid
                groups = defaultdict(dict)
            cell = (str(row["role"]), int(row["top_n"]))
            groups[cell][expert] = row
    if current is not None:
        yield current, groups

def aggregate_nested(prefix, rows, field, out):
    by_key = defaultdict(list)
    for expert, row in rows.items():
        tmp = {}
        flatten("", row.get(field) or {}, tmp, keep_strings=False)
        for k, v in tmp.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                by_key[k.lstrip("_")].append(float(v))
    for k, vals in by_key.items():
        out[f"{prefix}_{k}_mean"] = mean(vals)
        out[f"{prefix}_{k}_std"] = stdev(vals)
        out[f"{prefix}_{k}_min"] = min(vals)
        out[f"{prefix}_{k}_max"] = max(vals)

def state_features(rows, pred):
    if set(rows) != set(EXPERTS):
        raise ValueError(f"expert coverage mismatch: {sorted(rows)}")
    first = rows[EXPERTS[0]]
    out = {}
    flatten("race", first.get("race") or {}, out, keep_strings=True)
    flatten("coverage", first.get("data_coverage") or {}, out, keep_strings=False)
    flatten("consensus", first.get("consensus") or {}, out, keep_strings=False)

    lists = {e: tuple(map(str, rows[e].get("candidate_horse_ids") or [])) for e in EXPERTS}
    pairs = []
    for i, a in enumerate(EXPERTS):
        for b in EXPERTS[i + 1:]:
            pairs.append(jaccard(lists[a], lists[b]))
    union = set().union(*(set(v) for v in lists.values()))
    supports = Counter()
    for horses in lists.values():
        supports.update(set(horses))
    out.update({
        "council_pair_jaccard_mean": mean(pairs) or 0.0,
        "council_pair_jaccard_min": min(pairs) if pairs else 1.0,
        "council_pair_jaccard_max": max(pairs) if pairs else 1.0,
        "council_union_size": len(union),
        "council_max_horse_support": max(supports.values()) if supports else 0,
        "council_mean_horse_support": mean(list(supports.values())) or 0.0,
        "council_unanimous_horse_count": sum(1 for x in supports.values() if x == len(EXPERTS)),
    })

    if set(pred) != set(EXPERTS):
        raise ValueError(f"prediction coverage mismatch: {sorted(pred)}")
    vals = sorted((float(v) for v in pred.values()), reverse=True)
    total = sum(max(v, 1e-12) for v in vals)
    q = [max(v, 1e-12) / total for v in vals]
    out.update({
        "router_p_max": vals[0],
        "router_p_second": vals[1],
        "router_p_gap": vals[0] - vals[1],
        "router_p_mean": mean(vals),
        "router_p_std": stdev(vals),
        "router_p_entropy": -sum(x * math.log(x) for x in q) / math.log(len(q)),
    })
    aggregate_nested("candidate", rows, "candidate_summary", out)
    aggregate_nested("expert", rows, "expert_summary", out)
    return out

def numeric_feature_summary(rows):
    all_vals = defaultdict(list)
    by_class = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cls = r["classification"]
        for k, v in r["features"].items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            x = finite(v)
            if x is None:
                continue
            all_vals[k].append(x)
            by_class[cls][k].append(x)

    result = {}
    for cls in ("RESCUE", "REGRESSION", "MISSED_RESCUE", "DEAD_ZONE"):
        items = []
        for k, vals in by_class[cls].items():
            if len(vals) < 20 or len(all_vals[k]) < 50:
                continue
            mu = mean(all_vals[k])
            sd = stdev(all_vals[k])
            cm = mean(vals)
            if sd <= 1e-12:
                continue
            items.append({
                "feature": k,
                "n": len(vals),
                "class_mean": cm,
                "overall_mean": mu,
                "z_shift": (cm - mu) / sd,
            })
        items.sort(key=lambda x: (-abs(x["z_shift"]), x["feature"]))
        result[cls] = items[:30]
    return result

def categorical_feature_summary(rows):
    candidate_fields = (
        "race_venue_code", "race_surface", "race_race_class", "race_discipline",
        "race_direction", "race_weather", "race_track_condition",
    )
    total_n = len(rows)
    overall = {f: Counter() for f in candidate_fields}
    by_class = defaultdict(lambda: {f: Counter() for f in candidate_fields})
    class_n = Counter(r["classification"] for r in rows)
    for r in rows:
        cls = r["classification"]
        feats = r["features"]
        for f in candidate_fields:
            v = feats.get(f)
            if v is None or str(v) == "":
                continue
            v = str(v)
            overall[f][v] += 1
            by_class[cls][f][v] += 1

    result = {}
    for cls in ("RESCUE", "REGRESSION", "MISSED_RESCUE", "DEAD_ZONE"):
        items = []
        cn = class_n[cls]
        if not cn:
            result[cls] = []
            continue
        for f in candidate_fields:
            for value, n in by_class[cls][f].items():
                if n < 10 or overall[f][value] < 20:
                    continue
                p_cls = n / cn
                p_all = overall[f][value] / total_n
                items.append({
                    "feature": f,
                    "value": value,
                    "class_n": n,
                    "class_share": p_cls,
                    "overall_share": p_all,
                    "lift": p_cls / p_all if p_all else None,
                })
        items.sort(key=lambda x: (-x["lift"], -x["class_n"], x["feature"], x["value"]))
        result[cls] = items[:30]
    return result

def summarize_group(rows):
    n = len(rows)
    cls = Counter(r["classification"] for r in rows)
    interventions = sum(r["intervened"] for r in rows)
    baseline_miss = sum(1 for r in rows if int(r["baseline_hit"]) == 0)
    baseline_hit = n - baseline_miss
    strategies = defaultdict(Counter)
    for r in rows:
        s = str(r["chosen_strategy"])
        strategies[s]["races"] += 1
        strategies[s][r["classification"]] += 1
    strategy_out = {}
    for s, c in sorted(strategies.items()):
        strategy_out[s] = {
            "races": c["races"],
            "rescue": c["RESCUE"],
            "regression": c["REGRESSION"],
            "net_rescue": c["RESCUE"] - c["REGRESSION"],
            "missed_rescue": c["MISSED_RESCUE"],
            "dead_zone": c["DEAD_ZONE"],
        }
    return {
        "races": n,
        "interventions": interventions,
        "intervention_rate": interventions / n if n else None,
        "rescue": cls["RESCUE"],
        "regression": cls["REGRESSION"],
        "net_rescue": cls["RESCUE"] - cls["REGRESSION"],
        "both_hit": cls["BOTH_HIT"],
        "missed_rescue": cls["MISSED_RESCUE"],
        "dead_zone": cls["DEAD_ZONE"],
        "dead_zone_rate": cls["DEAD_ZONE"] / n if n else None,
        "rescue_rate_among_baseline_misses": cls["RESCUE"] / baseline_miss if baseline_miss else None,
        "regression_rate_among_baseline_hits": cls["REGRESSION"] / baseline_hit if baseline_hit else None,
        "strategies": strategy_out,
    }

def main():
    a = parse_args()
    decisions = read_decisions(a.year, a.decision)
    preds = read_predictions(a.year_pred)
    rows = []
    seen = set()

    for rid, groups in iter_feature_groups(a.year_feature):
        for (role, top_n), expert_rows in groups.items():
            key = (rid, role, top_n)
            d = decisions.get(key)
            if d is None:
                continue
            feats = state_features(expert_rows, preds.get(key, {}))
            rows.append({
                "contract": "L15_COUNCIL_RESCUE_AUDIT_V1_RACE",
                "year": a.year,
                "race_id": rid,
                "role": role,
                "top_n": top_n,
                "classification": d["classification"],
                "intervened": int(d["intervened"]),
                "chosen_strategy": d["chosen_strategy"],
                "baseline_hit": int(d["baseline_hit"]),
                "chosen_hit": int(d["chosen_hit"]),
                "oracle_hit": int(d["oracle_hit"]),
                "predicted_hit_probability": d.get("predicted_hit_probability"),
                "strategy_margin": d.get("strategy_margin"),
                "candidate_horse_ids": d.get("candidate_horse_ids") or [],
                "features": feats,
            })
            seen.add(key)

    missing = sorted(set(decisions) - seen)
    if missing:
        raise ValueError(f"missing feature groups for {len(missing)} decisions, first={missing[:3]}")

    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with gzip.open(outdir / "audit-races.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")

    by_cell = defaultdict(list)
    by_role = defaultdict(list)
    for r in rows:
        by_cell[f'{r["role"]}|{r["top_n"]}'].append(r)
        by_role[r["role"]].append(r)

    summary = {
        "contract": "L15_COUNCIL_RESCUE_AUDIT_V1",
        "year": a.year,
        "business_objective": "profit_maximization",
        "definitions": {
            "RESCUE": "baseline miss, Council V3 hit",
            "REGRESSION": "baseline hit, Council V3 miss",
            "BOTH_HIT": "baseline and Council V3 both hit",
            "MISSED_RESCUE": "baseline and V3 miss, but another permitted council strategy hits",
            "DEAD_ZONE": "all permitted council strategies miss (oracle_hit=0)",
        },
        "guardrails": {
            "odds_used": False,
            "locked_years": [2026],
            "snapshot_reparse": False,
            "note": "DEAD_ZONE is Council V3 candidate-pool dead zone, not seven-king Top6 dead zone.",
        },
        "all": summarize_group(rows),
        "by_role": {k: summarize_group(v) for k, v in sorted(by_role.items())},
        "by_cell": {k: summarize_group(v) for k, v in sorted(by_cell.items())},
        "numeric_characteristics": numeric_feature_summary(rows),
        "categorical_characteristics": categorical_feature_summary(rows),
        "next_stage": {
            "rescue_gate_v4": "Keep hard router by default; switch only when predicted rescue benefit exceeds regression risk.",
            "outsider_join": "Join DEAD_ZONE rows to outsider results by year/race_id.",
        },
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        "COUNCIL_RESCUE_AUDIT_OK",
        "year", a.year,
        "races", len(rows),
        "rescue", summary["all"]["rescue"],
        "regression", summary["all"]["regression"],
        "net", summary["all"]["net_rescue"],
        "missed_rescue", summary["all"]["missed_rescue"],
        "dead_zone", summary["all"]["dead_zone"],
        flush=True,
    )
    for cell, s in summary["by_cell"].items():
        print(
            "AUDIT_CELL", cell,
            "races", s["races"],
            "rescue", s["rescue"],
            "regression", s["regression"],
            "net", s["net_rescue"],
            "missed", s["missed_rescue"],
            "dead", s["dead_zone"],
            "intervention_pct", round((s["intervention_rate"] or 0.0) * 100, 2),
            flush=True,
        )

if __name__ == "__main__":
    main()
