#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def parse_args():
    p=argparse.ArgumentParser(description="Persist a preflight-approved KEIBA research bundle to private Kaggle.")
    p.add_argument("bundle_root", nargs="?", default="out/research-bundles")
    return p.parse_args()


def run(*args,capture=False,check=True):
    r=subprocess.run([str(x) for x in args],text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,check=False)
    if check and r.returncode != 0:
        msg=(r.stderr or r.stdout or "").strip() if capture else ""
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str,args))} {msg}")
    return r


def sha256(path):
    import hashlib
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def exists(ref):
    return run("kaggle","datasets","status",ref,capture=True,check=False).returncode == 0


def main():
    a=parse_args()
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN is required")
    if os.environ.get("KAGGLE_STORAGE_PREFLIGHT_OK") != "1":
        raise SystemExit("Kaggle storage preflight marker missing; capacity must be checked before compute.")

    manifests=sorted(Path(a.bundle_root).glob("*/manifest.json"))
    if not manifests:
        raise SystemExit("research bundle manifest not found")
    mp=manifests[-1]
    d=mp.parent
    m=json.loads(mp.read_text(encoding="utf-8"))
    if m.get("contract")!="KEIBA_RESEARCH_BUNDLE_V1":
        raise SystemExit("unexpected research bundle contract")

    for e in m["files"]:
        p=d/e["file"]
        if not p.is_file() or p.stat().st_size != int(e["bytes"]) or sha256(p)!=e["sha256"]:
            raise SystemExit(f"bundle integrity failure: {e['file']}")

    bid=str(m["bundle_id"])
    kind=str(m["kind"]).lower().replace("_","-")
    slug=f"keiba-ml-research-{kind}-{bid}"
    ref=f"pino1945/{slug}"

    with tempfile.TemporaryDirectory() as td:
        stage=Path(td)
        for e in m["files"]:
            shutil.copy2(d/e["file"],stage/e["file"])
        shutil.copy2(mp,stage/"manifest.json")
        sums=[f"{sha256(stage/e['file'])}  {e['file']}" for e in m["files"]]
        sums.append(f"{sha256(stage/'manifest.json')}  manifest.json")
        (stage/"SHA256SUMS").write_text("\n".join(sums)+"\n",encoding="utf-8")
        meta={
            "id":ref,
            "title":f"KEIBA ML Research {kind} {bid}"[:50],
            "licenses":[{"name":"other"}],
            "description":(
                f"Private KEIBA-ML research bundle {kind}; {bid}; "
                f"snapshot generation {m['snapshot_generation']}. "
                "Capacity was checked once before compute under the 180 GiB policy."
            )
        }
        (stage/"dataset-metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

        if exists(ref):
            print("Dataset already exists; verifying remote manifest.")
        else:
            # Intentionally no capacity re-check here. Preflight happened before compute.
            run("kaggle","datasets","create","-p",str(stage),"--quiet","--keep-tabular","--dir-mode","skip")

        ok=False
        with tempfile.TemporaryDirectory() as vd:
            verify=Path(vd)
            for attempt in range(1,61):
                remote=verify/"manifest.json"
                if remote.exists(): remote.unlink()
                r=run("kaggle","datasets","download",ref,"-f","manifest.json","-p",str(verify),
                      "--unzip","--quiet","--force",capture=True,check=False)
                if r.returncode==0 and remote.is_file():
                    if remote.read_bytes()!=(stage/"manifest.json").read_bytes():
                        raise SystemExit("remote manifest mismatch")
                    ok=True
                    print(f"Kaggle research manifest verification succeeded on attempt {attempt}.")
                    break
                if attempt<60: time.sleep(10)
        if not ok:
            raise SystemExit("research dataset did not become readable within verification window")

    print("KAGGLE_RESEARCH_PERSIST_OK")
    print(f"dataset_ref={ref}")
    print(f"bundle_id={bid}")


if __name__=="__main__":
    main()
