#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rank_utils import RANK_TIE_POLICY, deterministic_ranks, tie_diagnostics

ROOT = Path(__file__).resolve().parents[1]
FEATURE_CONTRACT = json.loads((ROOT / "contracts" / "l1-feature-set-contract-v1.json").read_text(encoding="utf-8"))
L2_CONTRACT = json.loads((ROOT / "contracts" / "l1-to-l2-output-contract-v1.json").read_text(encoding="utf-8"))

BASE_CATEGORICAL = [
    "venue_code", "discipline", "surface", "direction", "weather",
    "track_condition", "sex",
    "backfill_course_layout", "backfill_race_class_normalized", "backfill_grade",
    "backfill_sex_condition", "backfill_weight_rule",
]


def args():
    p = argparse.ArgumentParser(description="Emit leakage-safe L1_TO_L2_OUTPUT_CONTRACT_V1 rows from a trained L1 model.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--candidate-name", required=True)
    p.add_argument("--feature-sets-json", required=True)
    p.add_argument("--actor-prefixes-json", default="[]")
    p.add_argument("--auto-slices-json", default="[]")
    p.add_argument("--pedigree-slices-json", default="[]")
    p.add_argument("--valid-start", required=True)
    p.add_argument("--valid-end", required=True)
    p.add_argument("--chunk-size", type=int, default=128)
    return p.parse_args()


def opener(path, mode):
    return gzip.open if str(path).endswith(".gz") else open


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def proc_memory():
    rss_kib = 0
    hwm_kib = 0
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm_kib = int(line.split()[1])
    except OSError:
        pass
    avail_kib = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MemAvailable:"):
                avail_kib = int(line.split()[1])
                break
    except OSError:
        pass
    return {
        "rss_mib": round(rss_kib / 1024, 3),
        "peak_rss_mib": round(hwm_kib / 1024, 3),
        "system_mem_available_mib": round(avail_kib / 1024, 3),
    }


def mem_sample(stage, **extra):
    payload = {"stage": stage, **proc_memory(), **extra}
    print("L2_OUTPUT_MEMORY_SAMPLE " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
    return payload


def family_for_feature(name):
    for family, spec in FEATURE_CONTRACT["feature_sets"].items():
        for prefix in spec.get("prefixes") or []:
            if str(name).startswith(str(prefix)):
                return family
    return "BASE"


def iter_rows(path):
    with opener(path, "rt")(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_validation_frame(dataset, valid_start, valid_end, feature_order, category_levels):
    records = []
    ids = []
    for row in iter_rows(dataset):
        features = dict(row.get("features") or {})
        race_date = str(features.pop("race_date", "") or "")[:10]
        if not race_date or race_date < valid_start or race_date > valid_end:
            continue
        features.pop("actual_start_time", None)
        features.pop("jockey_id", None)
        features.pop("trainer_id", None)
        dt = pd.to_datetime(race_date, errors="coerce")
        features["race_month"] = int(dt.month) if not pd.isna(dt) else np.nan
        features["race_day_of_year"] = int(dt.dayofyear) if not pd.isna(dt) else np.nan
        record = {name: features.get(name) for name in feature_order}
        records.append(record)
        horse_number = features.get("horse_number")
        if horse_number is not None:
            try:
                horse_number = int(float(horse_number))
            except (TypeError, ValueError):
                horse_number = None
        ids.append({
            "race_date": race_date,
            "race_id": str(row.get("race_id") or ""),
            "horse_id": str(row.get("horse_id") or ""),
            "horse_number": horse_number,
        })
    if not records:
        raise ValueError("no validation rows found")

    frame = pd.DataFrame.from_records(records, columns=feature_order)
    for col in feature_order:
        if col in category_levels:
            vals = frame[col].astype("string").fillna("__MISSING__")
            frame[col] = pd.Categorical(vals, categories=[str(x) for x in category_levels[col]])
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame, ids


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def main():
    a = args()
    if a.chunk_size < 1 or a.chunk_size > 1000:
        raise ValueError("--chunk-size must be 1..1000")
    if a.valid_start[:4] == "2026" or a.valid_end[:4] == "2026":
        raise ValueError("2026 remains locked")

    mem_sample("start")
    schema = json.loads(Path(a.schema).read_text(encoding="utf-8"))
    meta = json.loads(Path(a.meta).read_text(encoding="utf-8"))
    feature_order = list(schema.get("feature_order") or [])
    category_levels = dict(schema.get("category_levels") or {})
    if not feature_order:
        raise ValueError("schema feature_order is empty")

    feature_sets = json.loads(a.feature_sets_json)
    actor_prefixes = json.loads(a.actor_prefixes_json)
    auto_slices = json.loads(a.auto_slices_json)
    pedigree_slices = json.loads(a.pedigree_slices_json)
    expert_config = {
        "candidate_name": a.candidate_name,
        "prediction_phase": str(meta.get("prediction_phase") or "FINAL"),
        "feature_sets": feature_sets,
        "actor_prefixes": actor_prefixes,
        "auto_slices": auto_slices,
        "pedigree_slices": pedigree_slices,
        "feature_selection_mode": (meta.get("feature_selection") or {}).get("mode", "none"),
    }
    expert_id = sha256_json(expert_config)[:16]

    booster = lgb.Booster(model_file=a.model)
    mem_sample("model_loaded", feature_count=len(feature_order), expert_id=expert_id)

    frame, ids = load_validation_frame(
        a.dataset, a.valid_start, a.valid_end, feature_order, category_levels
    )
    mem_sample("validation_loaded", rows=len(frame), races=len({x["race_id"] for x in ids}))

    probabilities = np.empty(len(frame), dtype=float)
    raw_margins = np.empty(len(frame), dtype=float)
    contributions = [None] * len(frame)
    biases = np.empty(len(frame), dtype=float)
    family_names = list(FEATURE_CONTRACT["feature_sets"])
    feature_families = [family_for_feature(name) for name in feature_order]
    best_iteration = int(meta.get("best_iteration") or booster.best_iteration or -1)

    for start in range(0, len(frame), a.chunk_size):
        stop = min(len(frame), start + a.chunk_size)
        part = frame.iloc[start:stop]
        probabilities[start:stop] = booster.predict(part, num_iteration=best_iteration)
        raw_margins[start:stop] = booster.predict(part, raw_score=True, num_iteration=best_iteration)
        matrix = np.asarray(
            booster.predict(part, pred_contrib=True, num_iteration=best_iteration),
            dtype=float,
        )
        if matrix.ndim != 2 or matrix.shape[1] != len(feature_order) + 1:
            raise ValueError(f"unexpected pred_contrib shape: {matrix.shape}")
        for local, values in enumerate(matrix[:, :len(feature_order)]):
            signed = {name: 0.0 for name in family_names}
            absolute = {name: 0.0 for name in family_names}
            for idx, value in enumerate(values):
                family = feature_families[idx]
                signed[family] += float(value)
                absolute[family] += abs(float(value))
            total_abs = float(sum(absolute.values()))
            shares = {
                name: (float(absolute[name] / total_abs) if total_abs > 0 else 0.0)
                for name in family_names
            }
            global_idx = start + local
            bias = float(matrix[local, len(feature_order)])
            reconstructed = bias + float(sum(signed.values()))
            if abs(reconstructed - float(raw_margins[global_idx])) > 1e-5:
                raise ValueError("TreeSHAP raw-margin reconstruction mismatch")
            p_from_margin = sigmoid(float(raw_margins[global_idx]))
            if abs(p_from_margin - float(probabilities[global_idx])) > 1e-6:
                raise ValueError("raw/probability mismatch")
            biases[global_idx] = bias
            contributions[global_idx] = (signed, absolute, shares)
        mem_sample("chunk", start=start, stop=stop)

    by_race = {}
    for idx, ident in enumerate(ids):
        by_race.setdefault(ident["race_id"], []).append(idx)

    normalized = np.empty(len(frame), dtype=float)
    ranks = np.empty(len(frame), dtype=int)
    race_summaries = {}
    for race_id, indices in by_race.items():
        probs = np.clip(probabilities[indices], 1e-15, 1.0)
        total = float(probs.sum())
        pn = probs / total if total > 0 else np.full(len(indices), 1.0 / len(indices))
        local_rank = np.asarray(
            deterministic_ranks(
                probabilities[indices].tolist(),
                [ids[global_idx]["horse_id"] for global_idx in indices],
            ),
            dtype=int,
        )
        for pos, global_idx in enumerate(indices):
            normalized[global_idx] = float(pn[pos])
            ranks[global_idx] = int(local_rank[pos])

        ordered = np.sort(pn)[::-1]
        field_size = len(ordered)
        entropy = float(-np.sum(np.clip(pn, 1e-15, 1.0) * np.log(np.clip(pn, 1e-15, 1.0))))
        tie_info = tie_diagnostics(
            probabilities[indices].tolist(),
            [ids[global_idx]["horse_id"] for global_idx in indices],
        )
        race_summaries[race_id] = {
            "field_size": field_size,
            "rank_tie_policy": RANK_TIE_POLICY,
            "rank_tie_group_count": tie_info["tie_group_count"],
            "rank_tied_horse_count": tie_info["tied_horse_count"],
            "rank_max_tie_group_size": tie_info["max_tie_group_size"],
            "rank_top1_boundary_tie": tie_info["boundary_tie"]["1"],
            "rank_top3_boundary_tie": tie_info["boundary_tie"]["3"],
            "rank_top6_boundary_tie": tie_info["boundary_tie"]["6"],
            "top1_probability": float(ordered[0]),
            "top2_probability": float(ordered[1]) if field_size > 1 else None,
            "top1_top2_gap": float(ordered[0] - ordered[1]) if field_size > 1 else None,
            "top3_probability_mass": float(np.sum(ordered[:3])),
            "normalized_entropy": float(entropy / np.log(field_size)) if field_size > 1 else 0.0,
        }

    repro = meta.get("reproducibility") or {}
    source = meta.get("source") or {}
    model_sha = repro.get("model_sha256") or sha256_file(a.model)
    forbidden = set(L2_CONTRACT.get("forbidden_payload_fields") or [])
    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write = gzip.open if str(out_path).endswith(".gz") else open
    with write(out_path, "wt", encoding="utf-8") as fh:
        for idx, ident in enumerate(ids):
            signed, absolute, shares = contributions[idx]
            record = {
                "contract": "L1_TO_L2_OUTPUT_CONTRACT_V1",
                **ident,
                "model_version": str(meta.get("model_version") or ""),
                "expert_id": expert_id,
                "prediction_phase": str(meta.get("prediction_phase") or "FINAL"),
                "feature_sets": feature_sets,
                "ml_dataset_version": int((meta.get("dataset_contract") or {}).get("ml_dataset_version", 0)),
                "feature_schema_version": int((meta.get("dataset_contract") or {}).get("feature_schema_version", 0)),
                "leakage_policy": str((meta.get("dataset_contract") or {}).get("leakage_policy") or ""),
                "raw_win_probability": float(probabilities[idx]),
                "race_normalized_win_probability": float(normalized[idx]),
                "predicted_rank": int(ranks[idx]),
                "raw_margin_logit": float(raw_margins[idx]),
                "bias_logit": float(biases[idx]),
                "family_contribution_logit": signed,
                "family_abs_contribution": absolute,
                "family_abs_share": shares,
                "race_summary": race_summaries[ident["race_id"]],
                "model_sha256": model_sha,
                "training_config_sha256": repro.get("training_config_sha256"),
                "feature_catalog_sha256": repro.get("feature_catalog_sha256"),
                "feature_contract_sha256": repro.get("feature_contract_sha256"),
                "source_backfill_sha": source.get("sha"),
                "ml_source_sha": source.get("ml_source_sha"),
            }
            leaked = sorted(forbidden.intersection(record))
            if leaked:
                raise ValueError("forbidden L2 output fields: " + ", ".join(leaked))
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    final_mem = mem_sample("done", rows=len(ids), races=len(by_race), expert_id=expert_id)
    print("L1_TO_L2_OUTPUT_READY")
    print(json.dumps({
        "candidate": a.candidate_name,
        "expert_id": expert_id,
        "rows": len(ids),
        "races": len(by_race),
        "feature_count": len(feature_order),
        "chunk_size": a.chunk_size,
        "output": str(out_path),
        "l2_peak_rss_mib": final_mem["peak_rss_mib"],
        "system_mem_available_mib_at_end": final_mem["system_mem_available_mib"],
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
