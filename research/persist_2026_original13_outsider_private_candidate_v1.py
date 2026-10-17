#!/usr/bin/env python3
"""Persist unapproved ORIGINAL 13 Outsider models privately to existing Kaggle.

Requires external GitHub Actions free-storage preflight. Never publishes to
GitHub and never sets a model to approved. Refuses existing Kaggle dataset ID,
missing any of 13 models, unexpected labels, or inconsistent 2025 OOS source.
Explicitly preserves original model binaries/metadata, not their label rows.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from refit_2026_original_13_outsider_prior_only_candidate_v1 import config

REF="pino1945/keiba-ml-2026-original13-outsider-candidate-v1"
LIMIT=193273528320  # established 180-GiB conservative research ceiling
CAP=128*1024*1024
OOS="2025-oos"
BLIND="2026-candidate"
CONTRACT="KEIBA_ORIGINAL_13_OUTSIDER_2026_CANDIDATE_PRIVATE_HANDOFF_V1"

def require(x,msg):
    if not x:raise ValueError(msg)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for z in iter(lambda:f.read(1024*1024),b""):h.update(z)
    return h.hexdigest()

def run(cmd,check=True):
    p=subprocess.run([str(v) for v in cmd],text=True,capture_output=True)
    if check and p.returncode:
        raise RuntimeError("KAGGLE_PRIVATE_HANDOFF_COMMAND_FAILED:"+
                           (p.stderr or p.stdout or "")[-2500:])
    return p

def save(original_models,candidate_models,stage):
    require(os.environ.get("KAGGLE_STORAGE_PREFLIGHT_OK")=="1",
            "FREE_PRIVATE_KAGGLE_STORAGE_PREFLIGHT_MISSING")
    used=int(os.environ["KAGGLE_STORAGE_PREFLIGHT_BYTES"])
    require(used<LIMIT,"FREE_PRIVATE_STORAGE_ALREADY_AT_RESEARCH_CEILING")
    require(not stage.exists(),"REFUSE_OVERWRITE_STAGED_MODELS")
    candidates=config()
    stage.mkdir(parents=True)
    items={}
    for name in sorted(candidates):
        src=original_models/name
        dst=candidate_models/name
        a=(src/"model.txt",src/"model.json",src/"schema.json")
        b=(dst/"model.txt",dst/"model.json",dst/"schema.json")
        require(all(z.is_file() and z.stat().st_size>0 for z in (*a,*b)),
                "INCOMPLETE_ORIGINAL_OUTSIDER_MODEL_SET:"+name)
        old=json.loads(a[1].read_text())
        old_schema=json.loads(a[2].read_text())
        new=json.loads(b[1].read_text())
        check_split=old.get("split") or {}
        require(
            check_split.get("train_start")=="2023-01-01" and
            check_split.get("train_end")=="2024-12-31" and
            check_split.get("valid_start")=="2025-01-01" and
            check_split.get("valid_end")=="2025-12-31" and
            old.get("best_iteration")==new.get("source_original_2025_oos_best_iteration")
            and new.get("source_original_2025_oos_meta_sha256")==sha(a[1])
            and new.get("source_original_2025_oos_schema_sha256")==sha(a[2])
            and new.get("model_sha256")==sha(b[0])
            and new.get("schema_sha256")==sha(b[2])
            and new.get("candidate_name")==name
            and new.get("deployment_status")=="CANDIDATE_2026_BLIND_UNRELEASED"
            and new.get("training_years")==[2024,2025]
            and new.get("source_schema_years")==[2023,2024,2025]
            and new.get("2026_outcomes_used") is False
            and new.get("2026_validation_used") is False
            and new.get("training_rule_approval_ref") is None
            and old_schema.get("feature_order")==json.loads(b[2].read_text())["feature_order"],
            "ORIGINAL_13_OUTSIDER_2025_OOS_TO_2026_CANDIDATE_PROVENANCE_DRIFT:"+name)
        names={}
        for label,paths in ((OOS,a),(BLIND,b)):
            for suffix,file in zip(("model.txt","model.json","schema.json"),paths):
                dest=f"{name}__{label}__{suffix}"
                shutil.copyfile(file,stage/dest)
                names[dest]={"sha256":sha(stage/dest),"bytes":(stage/dest).stat().st_size}
        items[name]={
            "original_2025_oos_best_iteration":old["best_iteration"],
            "2025_validation_rows":old.get("split",{}).get("valid_rows"),
            "2026_training_rows":new["training_rows"],
            "original_feature_count":len(old_schema["feature_order"]),
            "files":names,
        }
    require(len(items)==13,"NOT_ORIGINAL_13")
    payload={
        "contract":CONTRACT,"dataset_ref":REF,
        "source_snapshot_generation":"eb13d4096519167b",
        "2025_oos_training_years":[2023,2024],
        "2025_oos_validation_year":2025,
        "candidate_2026_training_years":[2024,2025],
        "target_2026_results_read":False,
        "2026_outcomes_used_to_train_or_validate":False,
        "2026_candidate_approval_status":"UNRELEASED_RESEARCH_ONLY",
        "ready_for_betting":False,
        "outsider_candidate_count":13,
        "kings_retrained":0,
        "candidates":items,
    }
    (stage/"index.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    total=sum(f.stat().st_size for f in stage.iterdir() if f.is_file())
    require(total>0 and total<=CAP and used+total<LIMIT,
            "PRIVATE_KAGGLE_STORAGE_CAPACITY_NOT_VERIFIED")
    print("ORIGINAL_13_OUTSIDER_PRIVATE_SAVE_PREFLIGHT_OK "+
          json.dumps({"candidate_models":13,"stage_bytes":total,
                      "private_storage_approved":True,
                      "research_only":True},sort_keys=True),flush=True)
    return payload

def persist(stage):
    require(os.environ.get("KAGGLE_STORAGE_PREFLIGHT_OK")=="1",
            "PRIVATE_STORAGE_PREFLIGHT_NOT_APPROVED")
    pre=run(["kaggle","datasets","status",REF],check=False)
    require(pre.returncode!=0,"REFUSE_OVERWRITE_EXISTING_KAGGLE_OUTSIDER_MODEL_DATASET")
    with tempfile.TemporaryDirectory() as t:
        d=Path(t)
        for p in stage.iterdir():
            if p.is_file():shutil.copy2(p,d/p.name)
        (d/"dataset-metadata.json").write_text(json.dumps({
            "id":REF,
            "title":"KEIBA 2026 Original 13 Outsider Unreleased",
            "licenses":[{"name":"other"}],
            "description":"PRIVATE research-only original outsider 2025 OOS reproduced weights and 2026 prior-only refit, no 2026 validation.",
        },ensure_ascii=False,indent=2)+"\n")
        # Kaggle datasets CREATE defaults to private. No --public flag, ever.
        run(["kaggle","datasets","create","-p",str(d),"--quiet",
             "--keep-tabular","--dir-mode","skip"])
        with tempfile.TemporaryDirectory() as v:
            target=Path(v)
            run(["kaggle","datasets","download",REF,"-p",str(target),"--unzip",
                 "--quiet","--force"])
            expected={p.name:sha(p) for p in stage.iterdir() if p.is_file()}
            found={p.name:sha(p) for p in target.iterdir() if p.is_file()
                   and p.name in expected}
            require(expected==found,
                    "KAGGLE_13_OUTSIDER_PRIVATE_REMOTE_BYTES_NOT_EXACTLY_VERIFIED")
    print("ORIGINAL_13_OUTSIDER_UNRELEASED_MODEL_BUNDLE_PRIVATE_KAGGLE_VERIFIED "+
          json.dumps({"dataset_ref":REF,"files":len(expected),
                      "candidate_models":13,"2026_evaluation":False},sort_keys=True),
          flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original-oos-root",required=True,type=Path)
    p.add_argument("--2026-candidate-root",required=True,type=Path,dest="candidate_root")
    p.add_argument("--stage",required=True,type=Path)
    a=p.parse_args()
    save(a.original_oos_root,a.candidate_root,a.stage)
    persist(a.stage)
