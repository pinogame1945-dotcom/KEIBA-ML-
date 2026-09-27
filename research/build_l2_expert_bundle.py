#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def args():
    p=argparse.ArgumentParser(description="Build immutable manifest-backed L2 expert output bundle.")
    p.add_argument("--snapshot-generation", required=True)
    p.add_argument("--validation-start", required=True)
    p.add_argument("--validation-end", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--expert", action="append", required=True, help="candidate_name=path")
    p.add_argument("--out-root", default="out/l2-bundles")
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


def read_rows(path):
    op=gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main():
    a=args()
    summary_path=Path(a.summary)
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("contract")!="L2_MULTI_EXPERT_PROTOTYPE_V1":
        raise ValueError("unexpected L2 summary contract")

    parsed=[]
    for spec in a.expert:
        if "=" not in spec:
            raise ValueError("--expert must be candidate=path")
        candidate,raw_path=spec.split("=",1)
        path=Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows=0
        races=set()
        expert_ids=set()
        model_versions=set()
        contracts=set()
        for row in read_rows(path):
            rows+=1
            races.add(str(row["race_id"]))
            expert_ids.add(str(row["expert_id"]))
            model_versions.add(str(row.get("model_version") or ""))
            contracts.add(str(row.get("contract") or ""))
        if rows==0:
            raise ValueError(f"empty expert output: {path}")
        if contracts!={"L1_TO_L2_OUTPUT_CONTRACT_V1"}:
            raise ValueError(f"contract mismatch: {candidate} {contracts}")
        if len(expert_ids)!=1:
            raise ValueError(f"expert_id mismatch: {candidate} {expert_ids}")
        if len(model_versions)!=1:
            raise ValueError(f"model_version mismatch: {candidate}")
        parsed.append({
            "candidate":candidate,
            "source_path":str(path),
            "file":f"{candidate}.jsonl.gz",
            "expert_id":next(iter(expert_ids)),
            "model_version":next(iter(model_versions)),
            "rows":rows,
            "races":len(races),
            "bytes":path.stat().st_size,
            "sha256":sha256_file(path),
        })

    base={
        "contract":"L2_EXPERT_BUNDLE_V1",
        "snapshot_generation":a.snapshot_generation,
        "validation_start":a.validation_start,
        "validation_end":a.validation_end,
        "l1_output_contract":"L1_TO_L2_OUTPUT_CONTRACT_V1",
        "expert_outputs":[
            {k:v for k,v in row.items() if k!="source_path"}
            for row in parsed
        ],
        "comparison_summary":{
            "experts":summary.get("experts"),
            "races":summary.get("races"),
            "rows_per_expert":summary.get("rows_per_expert"),
            "top1_full_agreement_races":summary.get("top1_full_agreement_races"),
            "top1_full_agreement_rate":summary.get("top1_full_agreement_rate"),
            "coverage_identical":summary.get("coverage_identical"),
            "forbidden_field_check":summary.get("forbidden_field_check"),
        },
    }
    bundle_id=sha256_json(base)[:16]
    out_dir=Path(a.out_root)/bundle_id
    out_dir.mkdir(parents=True,exist_ok=True)

    total_bytes=0
    for row in parsed:
        src=Path(row["source_path"])
        dst=out_dir/row["file"]
        shutil.copyfile(src,dst)
        total_bytes+=dst.stat().st_size

    summary_dst=out_dir/"comparison-summary.json"
    shutil.copyfile(summary_path,summary_dst)
    summary_entry={
        "file":"comparison-summary.json",
        "bytes":summary_dst.stat().st_size,
        "sha256":sha256_file(summary_dst),
    }
    total_bytes+=summary_entry["bytes"]

    manifest={
        **base,
        "bundle_id":bundle_id,
        "summary":summary_entry,
        "total_bytes":total_bytes,
        "persistent_upload":False,
        "storage_class":"KAGGLE_PRIVATE_COMPANION_DATASET",
    }
    manifest_path=out_dir/"manifest.json"
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("L2_EXPERT_BUNDLE_READY")
    print(json.dumps({
        "bundle_id":bundle_id,
        "snapshot_generation":a.snapshot_generation,
        "experts":[{"candidate":x["candidate"],"expert_id":x["expert_id"],"rows":x["rows"],"races":x["races"]} for x in parsed],
        "total_bytes":total_bytes,
        "manifest":str(manifest_path),
    },ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
