#!/usr/bin/env python3
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def parse_args():
    p=argparse.ArgumentParser(description="Build immutable manifest-backed KEIBA research bundle.")
    p.add_argument("--kind", required=True)
    p.add_argument("--snapshot-generation", required=True)
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--test-start", required=True)
    p.add_argument("--test-end", required=True)
    p.add_argument("--code-sha", required=True)
    p.add_argument("--file", action="append", required=True, help="archive_name=source_path")
    p.add_argument("--out-root", default="out/research-bundles")
    return p.parse_args()


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value):
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main():
    a=parse_args()
    entries=[]
    names=set()
    for spec in a.file:
        if "=" not in spec:
            raise ValueError("--file must be archive_name=source_path")
        name,raw=spec.split("=",1)
        name=name.strip()
        src=Path(raw)
        if not name or "/" in name or "\\" in name:
            raise ValueError("archive_name must be a simple filename")
        if name in names:
            raise ValueError(f"duplicate archive name: {name}")
        if not src.is_file():
            raise FileNotFoundError(src)
        names.add(name)
        entries.append({
            "file":name,
            "source_path":str(src),
            "bytes":src.stat().st_size,
            "sha256":sha256_file(src),
        })

    base={
        "contract":"KEIBA_RESEARCH_BUNDLE_V1",
        "kind":a.kind,
        "snapshot_generation":a.snapshot_generation,
        "train_period":{"start":a.train_start,"end":a.train_end},
        "test_period":{"start":a.test_start,"end":a.test_end},
        "code_sha":a.code_sha,
        "files":[{k:v for k,v in e.items() if k!="source_path"} for e in entries],
    }
    bundle_id=sha256_json(base)[:16]
    out=Path(a.out_root)/bundle_id
    out.mkdir(parents=True,exist_ok=True)

    total=0
    for e in entries:
        dst=out/e["file"]
        shutil.copy2(e["source_path"],dst)
        total += dst.stat().st_size

    manifest={
        **base,
        "bundle_id":bundle_id,
        "total_bytes":total,
        "persistent_upload":True,
        "storage_class":"KAGGLE_PRIVATE_RESEARCH_DATASET",
        "storage_preflight_policy":"CHECK_ONCE_BEFORE_COMPUTE_180_GIB_NO_PREUPLOAD_RECHECK",
    }
    mp=out/"manifest.json"
    mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("KEIBA_RESEARCH_BUNDLE_V1_READY")
    print(json.dumps({"bundle_id":bundle_id,"files":len(entries),"total_bytes":total,"manifest":str(mp)},ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
