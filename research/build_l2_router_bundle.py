#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def parse_args():
    p=argparse.ArgumentParser(description="Build immutable manifest-backed L2 king-router experiment bundle.")
    p.add_argument("--router-version", required=True)
    p.add_argument("--snapshot-generation", required=True)
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--test-start", required=True)
    p.add_argument("--test-end", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--decisions", required=True)
    p.add_argument("--train-l2-dataset-ref", required=True)
    p.add_argument("--test-l2-dataset-ref", required=True)
    p.add_argument("--expert", action="append", default=[], help="name=expert_id")
    p.add_argument("--code-sha", required=True)
    p.add_argument("--out-root", default="out/l2-router-bundles")
    return p.parse_args()


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as fh:
        for chunk in iter(lambda: fh.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value):
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_jsonl(path):
    op=gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main():
    a=parse_args()
    summary_path=Path(a.summary)
    decisions_path=Path(a.decisions)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not decisions_path.is_file():
        raise FileNotFoundError(decisions_path)

    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    expected=f"L2_KING_ROUTER_{a.router_version.upper()}"
    actual=str(summary.get("contract") or "")
    if actual != expected:
        raise ValueError(f"router summary contract mismatch: expected={expected} actual={actual}")

    experts={}
    for spec in a.expert:
        if "=" not in spec:
            raise ValueError("--expert must be name=expert_id")
        name,expert_id=spec.split("=",1)
        name=name.strip()
        expert_id=expert_id.strip()
        if not name or not expert_id:
            raise ValueError("--expert name and expert_id must be non-empty")
        experts[name]=expert_id

    rows=0
    races=set()
    for row in read_jsonl(decisions_path):
        rows+=1
        race_id=str(row.get("race_id") or "")
        if not race_id:
            raise ValueError("router decision row missing race_id")
        races.add(race_id)
    if rows == 0:
        raise ValueError("router decisions are empty")

    base={
        "contract":"L2_KING_ROUTER_BUNDLE_V1",
        "router_version":a.router_version.upper(),
        "router_summary_contract":expected,
        "snapshot_generation":a.snapshot_generation,
        "train_period":{"start":a.train_start,"end":a.train_end},
        "test_period":{"start":a.test_start,"end":a.test_end},
        "train_l2_dataset_ref":a.train_l2_dataset_ref,
        "test_l2_dataset_ref":a.test_l2_dataset_ref,
        "experts":experts,
        "code_sha":a.code_sha,
        "decision_rows":rows,
        "decision_races":len(races),
    }
    bundle_id=sha256_json(base)[:16]
    out_dir=Path(a.out_root)/bundle_id
    out_dir.mkdir(parents=True,exist_ok=True)

    summary_dst=out_dir/"router-summary.json"
    shutil.copyfile(summary_path,summary_dst)

    decisions_dst=out_dir/"router-decisions.jsonl.gz"
    if str(decisions_path).endswith(".gz"):
        shutil.copyfile(decisions_path,decisions_dst)
    else:
        with open(decisions_path,"rt",encoding="utf-8") as src, gzip.open(decisions_dst,"wt",encoding="utf-8") as dst:
            shutil.copyfileobj(src,dst)

    files=[]
    total_bytes=0
    for path in (summary_dst,decisions_dst):
        entry={"file":path.name,"bytes":path.stat().st_size,"sha256":sha256_file(path)}
        files.append(entry)
        total_bytes+=entry["bytes"]

    manifest={
        **base,
        "bundle_id":bundle_id,
        "files":files,
        "total_bytes":total_bytes,
        "persistent_upload":False,
        "storage_class":"KAGGLE_PRIVATE_ROUTER_DATASET",
    }
    manifest_path=out_dir/"manifest.json"
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("L2_KING_ROUTER_BUNDLE_READY")
    print(json.dumps({
        "bundle_id":bundle_id,
        "router_version":a.router_version.upper(),
        "decision_rows":rows,
        "decision_races":len(races),
        "total_bytes":total_bytes,
        "manifest":str(manifest_path),
    },ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
