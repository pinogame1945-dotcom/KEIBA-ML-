#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import math
import itertools
from collections import defaultdict
from pathlib import Path

YEARS = (2022, 2023, 2024, 2025)
TOP_NS = (3, 4, 5, 6, 7, 8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-root", required=True)
    p.add_argument("--router-root", required=True)
    p.add_argument("--fixed-root", default="research-results/l15-fixed-v1")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "rt", encoding="utf-8")


def find_one(root, patterns):
    root = Path(root)
    for pat in patterns:
        hits = sorted(root.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no file under {root} matching {patterns}")


def ordered_outcomes(groups, slots=3):
    pieces = [()]
    for rank in sorted(groups):
        if rank > slots:
            continue
        horses = list(groups[rank])
        occupied = min(len(horses), slots - rank + 1)
        if occupied <= 0:
            continue
        perms = list(itertools.permutations(horses, occupied))
        pieces = [a + b for a in pieces for b in perms]
    return sorted(set(x for x in pieces if len(x) == slots))


def read_truth(path):
    groups = defaultdict(lambda: defaultdict(list))
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get("race_id") or "")
            hid = str(row.get("horse_id") or "")
            target = row.get("target") or {}
            try:
                finish = int(float(target.get("finish_position")))
            except (TypeError, ValueError):
                continue
            if rid and hid and finish <= 3:
                groups[rid][finish].append(hid)
    out = {}
    dead = 0
    for rid, g in groups.items():
        outcomes = ordered_outcomes(g, 3)
        if outcomes:
            out[rid] = outcomes
            if any(len(v) > 1 for v in g.values()):
                dead += 1
    return out, dead


def load_router(path):
    out = {}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get("race_id") or "")
            if not rid:
                raise ValueError(f"router row missing race_id in {path}")
            if rid in out:
                raise ValueError(f"duplicate router race_id {rid}")
            out[rid] = row
    return out


def seven_order(router_row):
    experts = router_row.get("experts") or {}
    if len(experts) != 7:
        raise ValueError(f"expected 7 experts race_id={router_row.get('race_id')} got={len(experts)}")
    stats = defaultdict(lambda: {
        "support": 0,
        "borda": 0.0,
        "top1_votes": 0,
        "best_rank": 99,
        "rank_sum": 0.0,
    })
    for expert in experts.values():
        ids = [str(x) for x in (expert.get("top6_horse_ids") or []) if str(x)]
        if not ids:
            raise ValueError(f"expert missing Top6 race_id={router_row.get('race_id')}")
        for rank, hid in enumerate(ids, 1):
            s = stats[hid]
            s["support"] += 1
            s["borda"] += float(7 - rank)
            s["top1_votes"] += int(rank == 1)
            s["best_rank"] = min(s["best_rank"], rank)
            s["rank_sum"] += rank
    for s in stats.values():
        s["mean_rank"] = s["rank_sum"] / s["support"]
    return sorted(
        stats,
        key=lambda hid: (
            -stats[hid]["borda"],
            -stats[hid]["support"],
            -stats[hid]["top1_votes"],
            stats[hid]["best_rank"],
            stats[hid]["mean_rank"],
            hid,
        ),
    )


def load_fixed_alerts(path):
    out = {}
    with open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get("race_id") or "")
            if not rid:
                continue
            out[rid] = [str(x) for x in (row.get("candidate_horse_ids") or []) if str(x)]
    return out


def captures(candidate_ids, outcomes):
    s = set(candidate_ids)
    return any(set(o).issubset(s) for o in outcomes)


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


def init_acc():
    return {
        "races": 0,
        "alerts": 0,
        "dead_heat_races": 0,
        "top_hits": {n: 0 for n in TOP_NS},
        "union_hits": 0,
        "l15_hits": 0,
        "alert_union_hits": 0,
        "alert_l15_hits": 0,
        "alert_podium_rescues": 0,
        "union_sizes": [],
        "l15_sizes": [],
        "min_n_when_union_captures": [],
    }


def add(acc, row):
    acc["races"] += 1
    acc["alerts"] += int(row["alert"])
    acc["dead_heat_races"] += int(row["dead_heat"])
    for n in TOP_NS:
        acc["top_hits"][n] += int(row[f"top{n}"])
    acc["union_hits"] += int(row["union"])
    acc["l15_hits"] += int(row["l15"])
    if row["alert"]:
        acc["alert_union_hits"] += int(row["union"])
        acc["alert_l15_hits"] += int(row["l15"])
        acc["alert_podium_rescues"] += int(row["l15"] and not row["union"])
    acc["union_sizes"].append(row["union_size"])
    acc["l15_sizes"].append(row["l15_size"])
    if row["min_n"] is not None:
        acc["min_n_when_union_captures"].append(row["min_n"])


def finalize(acc):
    races = acc["races"]
    alerts = acc["alerts"]
    out = {
        "races": races,
        "alerts": alerts,
        "dead_heat_races": acc["dead_heat_races"],
        "consensus_podium_capture": {
            f"top{n}": {
                "hits": acc["top_hits"][n],
                "rate": acc["top_hits"][n] / races if races else None,
                "trifecta_box_points": n * (n - 1) * (n - 2),
            }
            for n in TOP_NS
        },
        "seven_union_podium_capture": {
            "hits": acc["union_hits"],
            "rate": acc["union_hits"] / races if races else None,
            "avg_candidate_count": sum(acc["union_sizes"]) / races if races else None,
            "candidate_count_p50": percentile(acc["union_sizes"], 0.50),
            "candidate_count_p90": percentile(acc["union_sizes"], 0.90),
        },
        "l15_fixed_candidate_pool_podium_capture": {
            "hits": acc["l15_hits"],
            "rate": acc["l15_hits"] / races if races else None,
            "avg_candidate_count": sum(acc["l15_sizes"]) / races if races else None,
            "candidate_count_p50": percentile(acc["l15_sizes"], 0.50),
            "candidate_count_p90": percentile(acc["l15_sizes"], 0.90),
        },
        "alert_only": {
            "races": alerts,
            "seven_union_hits": acc["alert_union_hits"],
            "seven_union_rate": acc["alert_union_hits"] / alerts if alerts else None,
            "l15_fixed_hits": acc["alert_l15_hits"],
            "l15_fixed_rate": acc["alert_l15_hits"] / alerts if alerts else None,
            "podium_rescues_added_by_outsiders": acc["alert_podium_rescues"],
            "podium_rescue_rate_per_alert": acc["alert_podium_rescues"] / alerts if alerts else None,
        },
        "minimum_consensus_cutoff_when_union_can_capture": {
            "capturable_races": len(acc["min_n_when_union_captures"]),
            "p50": percentile(acc["min_n_when_union_captures"], 0.50),
            "p75": percentile(acc["min_n_when_union_captures"], 0.75),
            "p90": percentile(acc["min_n_when_union_captures"], 0.90),
            "mean": (
                sum(acc["min_n_when_union_captures"]) / len(acc["min_n_when_union_captures"])
                if acc["min_n_when_union_captures"] else None
            ),
        },
    }
    return out


def main():
    a = parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = init_acc()
    by_year = {}
    validation = {}

    for year in YEARS:
        snapshot = find_one(a.snapshot_root, [f"snapshot-{year}.jsonl.gz", f"snapshot-{year}.jsonl"])
        router = find_one(Path(a.router_root) / str(year), ["router-7k.jsonl.gz", "router-7k.jsonl"])
        fixed_path = Path(a.fixed_root) / f"y{year}.jsonl"

        truth, dead = read_truth(snapshot)
        routers = load_router(router)
        alerts = load_fixed_alerts(fixed_path)

        if len(routers) != 3456:
            raise SystemExit(f"router race count regression year={year}: {len(routers)} != 3456")
        if len(alerts) != 346:
            raise SystemExit(f"fixed alert count regression year={year}: {len(alerts)} != 346")

        missing_truth = sorted(set(routers) - set(truth))
        if missing_truth:
            raise SystemExit(f"truth missing for router races year={year}: {len(missing_truth)}")

        acc = init_acc()
        for rid in sorted(routers):
            outcomes = truth[rid]
            order = seven_order(routers[rid])
            is_alert = rid in alerts
            l15 = alerts[rid] if is_alert else order

            top = {}
            for n in TOP_NS:
                top[n] = captures(order[:n], outcomes)
            union_hit = captures(order, outcomes)
            l15_hit = captures(l15, outcomes)
            min_n = None
            if union_hit:
                for n in range(3, len(order) + 1):
                    if captures(order[:n], outcomes):
                        min_n = n
                        break

            row = {
                "alert": is_alert,
                "dead_heat": len(outcomes) > 1,
                "union": union_hit,
                "l15": l15_hit,
                "union_size": len(order),
                "l15_size": len(l15),
                "min_n": min_n,
            }
            for n in TOP_NS:
                row[f"top{n}"] = top[n]
            add(acc, row)
            add(overall, row)

        by_year[str(year)] = finalize(acc)
        validation[str(year)] = {
            "router_races": len(routers),
            "truth_races": len(truth),
            "fixed_alerts": len(alerts),
            "dead_heat_truth_races": dead,
            "snapshot": snapshot.name,
            "router": router.name,
        }

    summary = {
        "contract": "L15_PODIUM_CONTAINMENT_V1",
        "years": list(YEARS),
        "locked_years": [2026],
        "ability_uses_odds": False,
        "method": {
            "ranking": "Frozen L15 seven_order Borda/support consensus reconstructed from each seven expert Top6.",
            "truth": "Official finish positions from same-generation yearly snapshot; dead heats count as success if any valid ordered 1-2-3 outcome is fully contained.",
            "l15_pool": "Non-alert races use seven consensus union; alert races use frozen L15_FIXED_V1 candidate_horse_ids including selected outsider novel horses.",
            "note": "This measures podium containment only, not ROI/EV or trifecta order accuracy.",
        },
        "overall": finalize(overall),
        "by_year": by_year,
        "validation": validation,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = []
    for scope, metrics in [("OVERALL", summary["overall"])] + [(y, by_year[y]) for y in map(str, YEARS)]:
        base = {"scope": scope, "races": metrics["races"], "alerts": metrics["alerts"]}
        for n in TOP_NS:
            base[f"top{n}_hits"] = metrics["consensus_podium_capture"][f"top{n}"]["hits"]
            base[f"top{n}_rate"] = metrics["consensus_podium_capture"][f"top{n}"]["rate"]
        base["union_hits"] = metrics["seven_union_podium_capture"]["hits"]
        base["union_rate"] = metrics["seven_union_podium_capture"]["rate"]
        base["union_avg_count"] = metrics["seven_union_podium_capture"]["avg_candidate_count"]
        base["l15_hits"] = metrics["l15_fixed_candidate_pool_podium_capture"]["hits"]
        base["l15_rate"] = metrics["l15_fixed_candidate_pool_podium_capture"]["rate"]
        base["l15_avg_count"] = metrics["l15_fixed_candidate_pool_podium_capture"]["avg_candidate_count"]
        base["alert_podium_rescues"] = metrics["alert_only"]["podium_rescues_added_by_outsiders"]
        rows.append(base)

    fields = list(rows[0])
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("L15_PODIUM_CONTAINMENT_READY")
    print(json.dumps(summary["overall"], ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
