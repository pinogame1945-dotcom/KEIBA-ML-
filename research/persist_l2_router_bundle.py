#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

DEFAULT_SAFETY_LIMIT=193273528320


def parse_args():
    p=argparse.ArgumentParser(description="Persist an L2 router bundle to a guarded private Kaggle dataset.")
    p.add_argument("bundle_root", nargs="?", default="out/l2-router-bundles")
    p.add_argument("--ephemeral", action="store_true")
    return p.parse_args()


def run(*args, capture=False, check=True):
    result=subprocess.run(
        [str(x) for x in args],
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )
    if check and result.returncode != 0:
        msg=(result.stderr or result.stdout or "").strip() if capture else ""
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(map(str,args))} {msg}")
    return result


def sha256sum(path):
    import hashlib
    h=hashlib.sha256()
    with open(path,"rb") as fh:
        for chunk in iter(lambda: fh.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_exists(ref):
    return run("kaggle","datasets","status",ref,capture=True,check=False).returncode == 0


def owned_bytes():
    rows=[]
    for page in range(1,1001):
        r=run("kaggle","datasets","list","--mine","--page",str(page),"--format","json(ref,totalBytes)",capture=True)
        raw=(r.stdout or "").strip()
        if not raw or raw.lower().startswith("no datasets found"):
            batch=[]
        else:
            batch=json.loads(raw)
            if not isinstance(batch,list):
                raise RuntimeError("unexpected Kaggle datasets list payload")
        rows.extend(batch)
        if not batch:
            break
    else:
        raise RuntimeError("Kaggle dataset pagination safety stop")
    return sum(int(x.get("totalBytes") or 0) for x in rows)


def main():
    a=parse_args()
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN is required")

    manifests=sorted(Path(a.bundle_root).glob("*/manifest.json"))
    if not manifests:
        raise SystemExit(f"router bundle manifest not found under {a.bundle_root}")
    manifest_path=manifests[-1]
    bundle_dir=manifest_path.parent
    m=json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("contract") != "L2_KING_ROUTER_BUNDLE_V1":
        raise SystemExit("unexpected router bundle contract")

    for entry in m["files"]:
        path=bundle_dir/entry["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(entry["bytes"]):
            raise SystemExit(f"size mismatch: {path.name}")
        if sha256sum(path) != entry["sha256"]:
            raise SystemExit(f"sha256 mismatch: {path.name}")

    bundle_id=str(m["bundle_id"])
    router_version=str(m["router_version"])
    snapshot_generation=str(m["snapshot_generation"])
    bundle_bytes=int(m["total_bytes"])
    slug=f"keiba-ml-router-{bundle_id}"
    dataset_ref=f"pino1945/{slug}"

    print(f"Router bundle: {bundle_id}")
    print(f"Router version: {router_version}")
    print(f"Snapshot generation: {snapshot_generation}")
    print(f"Bundle bytes: {bundle_bytes}")

    safety_limit=int(os.environ.get("KAGGLE_ROUTER_SAFETY_LIMIT_BYTES",DEFAULT_SAFETY_LIMIT))
    verify_attempts=int(os.environ.get("KAGGLE_ROUTER_VERIFY_ATTEMPTS","60"))
    verify_sleep=float(os.environ.get("KAGGLE_ROUTER_VERIFY_SLEEP_SECONDS","10"))

    created_here=False
    with tempfile.TemporaryDirectory() as tmp:
        stage=Path(tmp)
        for entry in m["files"]:
            shutil.copy2(bundle_dir/entry["file"],stage/entry["file"])
        shutil.copy2(manifest_path,stage/"manifest.json")

        lines=[]
        for name in ["router-summary.json","router-decisions.jsonl.gz","manifest.json"]:
            lines.append(f"{sha256sum(stage/name)}  {name}")
        (stage/"SHA256SUMS").write_text("\n".join(lines)+"\n",encoding="utf-8")

        metadata={
            "id":dataset_ref,
            "title":f"KEIBA ML Router {router_version} {bundle_id}"[:50],
            "licenses":[{"name":"other"}],
            "description":(
                "Private KEIBA-ML L2 king-router experiment bundle. "
                f"Router {router_version}; bundle {bundle_id}; snapshot generation {snapshot_generation}. "
                "Contains frozen test decisions and summary for reproducible router comparisons."
            ),
        }
        (stage/"dataset-metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

        if dataset_exists(dataset_ref):
            print("Dataset already exists; verifying remote manifest instead of duplicating.")
        else:
            current=owned_bytes()
            projected=current+bundle_bytes
            print(f"Owned Kaggle dataset bytes: {current}")
            print(f"Projected bytes after router upload: {projected}")
            print(f"Safety ceiling: {safety_limit}")
            if projected > safety_limit:
                raise SystemExit("Projected Kaggle storage exceeds the 180 GiB safety ceiling; refusing upload.")
            run("kaggle","datasets","create","-p",str(stage),"--quiet","--keep-tabular","--dir-mode","skip")
            created_here=True

        verified=False
        with tempfile.TemporaryDirectory() as verify_tmp:
            verify_dir=Path(verify_tmp)
            for attempt in range(1,verify_attempts+1):
                remote=verify_dir/"manifest.json"
                if remote.exists():
                    remote.unlink()
                r=run("kaggle","datasets","download",dataset_ref,"-f","manifest.json","-p",str(verify_dir),"--unzip","--quiet","--force",capture=True,check=False)
                if r.returncode == 0 and remote.is_file():
                    if remote.read_bytes() != (stage/"manifest.json").read_bytes():
                        raise SystemExit("Remote router manifest did not match local manifest.")
                    print(f"Kaggle router manifest verification succeeded on attempt {attempt}.")
                    verified=True
                    break
                if attempt < verify_attempts:
                    time.sleep(verify_sleep)
        if not verified:
            raise SystemExit("Kaggle router dataset did not become readable within the verification window.")

        if a.ephemeral:
            if not created_here:
                raise SystemExit("Ephemeral test unexpectedly reused an existing dataset; refusing delete.")
            run("kaggle","datasets","delete",dataset_ref,"--yes")
            created_here=False
            print("KAGGLE_ROUTER_EPHEMERAL_DELETE_OK")

    print("KAGGLE_ROUTER_PERSIST_OK")
    print(f"dataset_ref={dataset_ref}")
    print(f"bundle_id={bundle_id}")
    print(f"router_version={router_version}")
    print(f"snapshot_generation={snapshot_generation}")


if __name__=="__main__":
    main()
