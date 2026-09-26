#!/usr/bin/env python3
import argparse
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run bounded-memory L1 walk-forward folds.")
    p.add_argument("--source-root", required=True)
    p.add_argument("--out-dir", default="out/walk-forward")
    p.add_argument("--first-holdout", type=int, required=True)
    p.add_argument("--last-holdout", type=int, required=True)
    p.add_argument("--train-years", type=int, default=3)
    p.add_argument("--warmup-years", type=int, default=1)
    p.add_argument("--history-limit", type=int, default=5)
    p.add_argument("--stage", default="style",
                   choices=["base","opponent_v1","opponent_both","lap","style","backfill_v1","auto_v1","auto_backfill_v1","pedigree","distance"])
    p.add_argument("--source-repo", default="pinogame1945-dotcom/KEIBA-BACKFILL")
    p.add_argument("--source-ref", default="main")
    p.add_argument("--source-sha")
    p.add_argument("--keep-datasets", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    return p.parse_args()


def fold_spec(year, train_years, warmup_years):
    train_first = year - train_years
    return {
        "holdout_year": year,
        "source_start": f"{train_first - warmup_years}-01-01",
        "source_end": f"{year}-12-31",
        "emit_start": f"{train_first}-01-01",
        "emit_end": f"{year}-12-31",
        "train_start": f"{train_first}-01-01",
        "train_end": f"{year - 1}-12-31",
        "valid_start": f"{year}-01-01",
        "valid_end": f"{year}-12-31",
    }


def run(cmd):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def main():
    a = parse_args()
    if a.first_holdout > a.last_holdout:
        raise ValueError("first-holdout must be <= last-holdout")
    if a.train_years < 1 or a.warmup_years < 0:
        raise ValueError("invalid train/warmup years")

    folds = [
        fold_spec(y, a.train_years, a.warmup_years)
        for y in range(a.first_holdout, a.last_holdout + 1)
    ]
    print("L1_WALK_FORWARD_PLAN")
    print(json.dumps({
        "stage": a.stage,
        "train_years": a.train_years,
        "warmup_years": a.warmup_years,
        "folds": folds,
    }, indent=2))
    if a.plan_only:
        return

    root = Path(a.out_dir)
    datasets = root / "tmp-datasets"
    models = root / "models"
    oof_dir = root / "oof"
    schemas = root / "schemas"
    for p in (datasets, models, oof_dir, schemas):
        p.mkdir(parents=True, exist_ok=True)

    summaries = []
    oof_paths = []

    for fold in folds:
        year = fold["holdout_year"]
        dataset = datasets / f"fold-{year}.jsonl.gz"
        model = models / f"l1-{a.stage}-{year}.txt"
        meta = models / f"l1-{a.stage}-{year}.json"
        schema = schemas / f"l1-{a.stage}-{year}-schema.json"
        pred = oof_dir / f"oof-{year}.jsonl.gz"

        run([
            "node", "research/build-staged-dataset.mjs",
            "--source-root", a.source_root,
            "--output", dataset,
            "--source-start", fold["source_start"],
            "--source-end", fold["source_end"],
            "--emit-start", fold["emit_start"],
            "--emit-end", fold["emit_end"],
            "--history-limit", a.history_limit,
        ])

        cmd = [
            sys.executable, "research/train-staged-lightgbm.py",
            "--dataset", dataset,
            "--stage", a.stage,
            "--train-start", fold["train_start"],
            "--train-end", fold["train_end"],
            "--valid-start", fold["valid_start"],
            "--valid-end", fold["valid_end"],
            "--model-out", model,
            "--meta-out", meta,
            "--schema-out", schema,
            "--predictions-out", pred,
            "--model-version", f"L1_{a.stage.upper()}_WF_{year}",
            "--source-repo", a.source_repo,
            "--source-ref", a.source_ref,
        ]
        if a.source_sha:
            cmd.extend(["--source-sha", a.source_sha])
        run(cmd)

        metadata = json.loads(meta.read_text(encoding="utf-8"))
        summaries.append({
            "holdout_year": year,
            "model_version": metadata["model_version"],
            "split": metadata["split"],
            "metrics": metadata["metrics"],
            "feature_count": metadata["feature_count"],
        })
        oof_paths.append(pred)

        if not a.keep_datasets:
            dataset.unlink(missing_ok=True)

    combined = oof_dir / "l1-oof-all.jsonl.gz"
    with gzip.open(combined, "wt", encoding="utf-8") as dst:
        for p in oof_paths:
            with gzip.open(p, "rt", encoding="utf-8") as src:
                shutil.copyfileobj(src, dst)

    summary = {
        "contract": "L1_WALK_FORWARD_V1",
        "stage": a.stage,
        "source": {
            "repository": a.source_repo,
            "ref": a.source_ref,
            "sha": a.source_sha,
        },
        "train_years": a.train_years,
        "warmup_years": a.warmup_years,
        "history_limit": a.history_limit,
        "folds": summaries,
        "combined_oof": str(combined),
    }
    (root / "walk-forward-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("L1_WALK_FORWARD_DONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
