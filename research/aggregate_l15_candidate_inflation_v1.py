#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

YEARS = (2022, 2023, 2024, 2025)
LABEL_TO_CANDIDATE = {
    "当日傾向型": "outsider_daytrend",
    "レース構造型": "outsider_raceshape",
    "枠・コース型": "outsider_gatecourse",
    "メンバー構成型": "outsider_field",
    "騎手型": "outsider_jockey",
}
POLICIES = (
    "FULL_COMBO_K2",
    "FULL_COMBO_K3",
    "COMPACT_INDIVIDUAL_TOP3",
    "FIXED_K2",
    "FIXED_K3",
    "FIXED_K5",
)
EXPECTED_RESCUE = {
    "FULL_COMBO_K2": 192,
    "FULL_COMBO_K3": 216,
    "COMPACT_INDIVIDUAL_TOP3": 216,
    "FIXED_K2": 180,
    "FIXED_K3": 210,
    "FIXED_K5": 244,
}
GATE_ALERTS = 1384
GATE_BLIND = 277


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True, type=int)
    p.add_argument("--root", default="research-results/l15-candidate-inflation-v1")
    return p.parse_args()


def pipe_set(value):
    return {x for x in str(value or "").split("|") if x}


def truthy(value):
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def percentile(values, q):
    if not values:
        return None
    z = sorted(values)
    if len(z) == 1:
        return float(z[0])
    pos = (len(z) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(z[lo])
    w = pos - lo
    return float(z[lo] * (1 - w) + z[hi] * w)


def learned_selection(row, policy):
    if policy == "FULL_COMBO_K2":
        return pipe_set(row.get("FULL__COMBO_K2"))
    if policy == "FULL_COMBO_K3":
        return pipe_set(row.get("FULL__COMBO_K3"))
    if policy == "COMPACT_INDIVIDUAL_TOP3":
        return pipe_set(row.get("COMPACT__INDIVIDUAL_TOP3"))
    raise KeyError(policy)


def main():
    a = parse_args()
    run_dir = Path(a.root) / f"run-{a.run_id}"
    race_rows = []
    metric = {
        p: {
            "alerts": 0,
            "blind_alerts": 0,
            "false_alerts": 0,
            "rescued": 0,
            "selected_outsider_calls": 0,
            "novel_total": 0,
            "novel_blind_total": 0,
            "novel_false_total": 0,
            "novel_counts": [],
            "novel_blind_counts": [],
            "novel_false_counts": [],
            "zero_novel_alerts": 0,
        }
        for p in POLICIES
    }

    for year in YEARS:
        ydir = run_dir / f"y{year}"
        top6_path = ydir / "outsider-top6.csv"
        seven_path = ydir / "seven-union.csv"
        decisions_path = ydir / "router-decisions.csv"
        result_path = ydir / "router-result.json"
        for p in (top6_path, seven_path, decisions_path, result_path):
            if not p.exists():
                raise SystemExit(f"missing input: {p}")

        top6 = {}
        for row in load_csv(top6_path):
            rid = str(row["race_id"])
            label = str(row["label_ja"])
            top6[(rid, label)] = pipe_set(row["top6_horse_ids"])

        seven = {
            str(r["race_id"]): pipe_set(r["seven_union_horse_ids"])
            for r in load_csv(seven_path)
        }
        decisions = load_csv(decisions_path)
        fold = json.loads(result_path.read_text(encoding="utf-8"))
        fixed = {
            p: set(fold["fixed_baselines"][p]["combo"])
            for p in ("FIXED_K2", "FIXED_K3", "FIXED_K5")
        }

        decision_ids = {str(r["race_id"]) for r in decisions}
        if decision_ids != set(seven):
            raise SystemExit(
                f"seven/decision race coverage mismatch year={year} "
                f"decisions={len(decision_ids)} seven={len(seven)}"
            )

        expected_top6 = len(decisions) * len(LABEL_TO_CANDIDATE)
        if len(top6) != expected_top6:
            raise SystemExit(
                f"outsider Top6 coverage mismatch year={year} "
                f"got={len(top6)} expected={expected_top6}"
            )

        for row in decisions:
            rid = str(row["race_id"])
            blind = truthy(row["blind"])
            actual = pipe_set(row.get("actual_rescuers"))
            seven_set = seven[rid]

            for policy in POLICIES:
                selected = fixed[policy] if policy.startswith("FIXED_") else learned_selection(row, policy)
                if not selected:
                    raise SystemExit(f"empty policy selection year={year} race_id={rid} policy={policy}")
                unknown = sorted(selected - set(LABEL_TO_CANDIDATE))
                if unknown:
                    raise SystemExit(f"unknown outsider labels policy={policy}: {unknown}")

                by_label = {}
                outsider_union = set()
                for label in sorted(selected):
                    ids = top6.get((rid, label))
                    if ids is None:
                        raise SystemExit(
                            f"missing candidate Top6 year={year} race_id={rid} label={label}"
                        )
                    by_label[label] = sorted(ids)
                    outsider_union.update(ids)

                novel = outsider_union - seven_set
                rescued = bool(blind and (selected & actual))
                m = metric[policy]
                m["alerts"] += 1
                m["selected_outsider_calls"] += len(selected)
                m["novel_total"] += len(novel)
                m["novel_counts"].append(len(novel))
                if not novel:
                    m["zero_novel_alerts"] += 1
                if blind:
                    m["blind_alerts"] += 1
                    m["novel_blind_total"] += len(novel)
                    m["novel_blind_counts"].append(len(novel))
                    if rescued:
                        m["rescued"] += 1
                else:
                    m["false_alerts"] += 1
                    m["novel_false_total"] += len(novel)
                    m["novel_false_counts"].append(len(novel))

                race_rows.append({
                    "year": year,
                    "race_id": rid,
                    "gate_score": row.get("gate_score", ""),
                    "blind": blind,
                    "policy": policy,
                    "selected_outsiders": "|".join(sorted(selected)),
                    "selected_outsider_count": len(selected),
                    "seven_union_horse_ids": "|".join(sorted(seven_set)),
                    "seven_union_count": len(seven_set),
                    "selected_outsider_top6": json.dumps(by_label, ensure_ascii=False, separators=(",", ":")),
                    "selected_outsider_top6_union": "|".join(sorted(outsider_union)),
                    "selected_outsider_top6_union_count": len(outsider_union),
                    "novel_horses": "|".join(sorted(novel)),
                    "novel_count": len(novel),
                    "actual_rescuers": "|".join(sorted(actual)),
                    "rescued": rescued,
                })

    if any(metric[p]["alerts"] != GATE_ALERTS for p in POLICIES):
        raise SystemExit("aggregate gate alert total mismatch")
    if any(metric[p]["blind_alerts"] != GATE_BLIND for p in POLICIES):
        raise SystemExit("aggregate gate blind total mismatch")
    for policy, expected in EXPECTED_RESCUE.items():
        got = metric[policy]["rescued"]
        if got != expected:
            raise SystemExit(f"rescue regression policy={policy} got={got} expected={expected}")

    summary_rows = []
    for policy in POLICIES:
        m = metric[policy]
        alerts = m["alerts"]
        blind = m["blind_alerts"]
        false = m["false_alerts"]
        novel_total = m["novel_total"]
        row = {
            "policy": policy,
            "alerts": alerts,
            "blind_alerts": blind,
            "false_alerts": false,
            "rescued": m["rescued"],
            "selected_outsider_calls": m["selected_outsider_calls"],
            "avg_selected_outsiders_per_alert": m["selected_outsider_calls"] / alerts,
            "novel_horses_total": novel_total,
            "avg_novel_horses_per_alert": novel_total / alerts,
            "avg_novel_horses_per_true_blind_alert": m["novel_blind_total"] / blind,
            "avg_novel_horses_per_false_alert": m["novel_false_total"] / false,
            "rescued_blind_per_100_novel_horses": (
                100 * m["rescued"] / novel_total if novel_total else 0.0
            ),
            "zero_novel_alerts": m["zero_novel_alerts"],
            "zero_novel_alert_rate": m["zero_novel_alerts"] / alerts,
            "novel_p50": percentile(m["novel_counts"], 0.50),
            "novel_p90": percentile(m["novel_counts"], 0.90),
            "novel_p95": percentile(m["novel_counts"], 0.95),
            "novel_max": max(m["novel_counts"]) if m["novel_counts"] else None,
        }
        summary_rows.append(row)

    run_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "policy","alerts","blind_alerts","false_alerts","rescued",
        "selected_outsider_calls","avg_selected_outsiders_per_alert",
        "novel_horses_total","avg_novel_horses_per_alert",
        "avg_novel_horses_per_true_blind_alert","avg_novel_horses_per_false_alert",
        "rescued_blind_per_100_novel_horses","zero_novel_alerts",
        "zero_novel_alert_rate","novel_p50","novel_p90","novel_p95","novel_max",
    ]
    with open(run_dir / "policy-comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    race_fields = [
        "year","race_id","gate_score","blind","policy","selected_outsiders",
        "selected_outsider_count","seven_union_horse_ids","seven_union_count",
        "selected_outsider_top6","selected_outsider_top6_union",
        "selected_outsider_top6_union_count","novel_horses","novel_count",
        "actual_rescuers","rescued",
    ]
    with open(run_dir / "race-level-policy.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=race_fields)
        w.writeheader()
        for row in race_rows:
            w.writerow(row)

    summary = {
        "contract": "L15_CANDIDATE_INFLATION_V1",
        "run_id": a.run_id,
        "years": list(YEARS),
        "gate_alerts": GATE_ALERTS,
        "gate_caught_blind": GATE_BLIND,
        "ability_uses_odds": False,
        "locked_years": [2026],
        "policies": {row["policy"]: row for row in summary_rows},
        "guardrails": [
            "Candidate inflation measures novel Top6 horse IDs added beyond the seven-king Top6 union.",
            "Rescue counts must exactly reproduce Router31 V1 before inflation metrics are accepted.",
            "No production policy is promoted by this experiment alone; L2 ROI/EV remains the downstream decision.",
            "2026 is sealed and odds/popularity/payout are not used.",
        ],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# L1.5 Candidate Inflation V1 — run {a.run_id}",
        "",
        f"- Gate alerts: {GATE_ALERTS:,}",
        f"- True blind alerts: {GATE_BLIND}",
        "- Seven-king baseline: union of all seven experts' Top6 horse IDs",
        "- Outsider contribution: only horse IDs absent from that seven-king union count as novel",
        "- 2026 sealed / odds NO / standard CPU only",
        "",
        "| Policy | Rescue | Novel total | Novel/alert | Novel/true blind | Novel/false alert | Rescue/100 novel |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['policy']} | {row['rescued']} | {row['novel_horses_total']} | "
            f"{row['avg_novel_horses_per_alert']:.3f} | "
            f"{row['avg_novel_horses_per_true_blind_alert']:.3f} | "
            f"{row['avg_novel_horses_per_false_alert']:.3f} | "
            f"{row['rescued_blind_per_100_novel_horses']:.3f} |"
        )
    lines += [
        "",
        "Race-level policy rows preserve race_id, selected outsiders, every selected outsider Top6 set, seven-king union, novel horse IDs, and rescue flag.",
        "No winner is promoted here; L2 ticket inflation / edge / ROI / profit is the next gate.",
        "",
    ]
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print("L15_CANDIDATE_INFLATION_READY")
    print(json.dumps({
        "run_id": a.run_id,
        "policies": {r["policy"]: {
            "rescued": r["rescued"],
            "novel_total": r["novel_horses_total"],
            "avg_novel": r["avg_novel_horses_per_alert"],
            "avg_false": r["avg_novel_horses_per_false_alert"],
            "rescue_per_100_novel": r["rescued_blind_per_100_novel_horses"],
        } for r in summary_rows},
        "path": str(run_dir),
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
