#!/usr/bin/env python3
"""Once-only free-CPU 2024+2025 refit for all seven original candidate kings.

Reads the previously verified Kaggle snapshots. The original projection and
trainer functions are invoked unchanged; outputs are CANDIDATE only, never
APPROVED or deployable. Neither snapshots nor features are generated anew.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
from refit_2026_seven_king_prior_only_v1 import CANDIDATES,train_one

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"research/l1-raceclass-absorption-walkforward-v1.json"
ALIASES=("core4","pedlegacy","condition","full","light","pedv1","condrc")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot-2024",type=Path,required=True)
    p.add_argument("--snapshot-2025",type=Path,required=True)
    p.add_argument("--prior-meta-root",type=Path,required=True)
    p.add_argument("--models-out",type=Path,required=True)
    p.add_argument("--work-root",type=Path,required=True)
    p.add_argument("--protocol-reference",required=True)
    a=p.parse_args()
    frozen=json.loads(FREEZE.read_text(encoding="utf-8"))
    assert frozen["snapshot_generation"]=="eb13d4096519167b"
    assert frozen["training_window_years"]==2
    assert frozen["locked_years"]==[2026]
    assert frozen["feature_selection"]=="none"
    assert frozen["prediction_phase"]=="FINAL"
    assert frozen["ability_uses_odds"] is False
    assert len(frozen["candidates"])==7
    assert not a.models_out.exists()
    a.models_out.mkdir(parents=True)
    a.work_root.mkdir(parents=True,exist_ok=True)
    source={2024:a.snapshot_2024,2025:a.snapshot_2025}
    for year,path in source.items():
        if path.name not in {f"snapshot-{year}.jsonl",f"snapshot-{year}.jsonl.gz"} or not path.is_file():
            raise ValueError("VERIFIED_HISTORIC_INPUT_NOT_FOUND:"+str(year))
    status=[]
    for alias in ALIASES:
        name,count=CANDIDATES[alias]
        entry=next(z for z in frozen["candidates"] if z["name"]==name)
        projections=[]
        for year in (2024,2025):
            dest=a.work_root/(f"snapshot-{year}.jsonl.gz")
            args=["node",str(ROOT/"research/project-yearly-snapshots.mjs"),
                  "--inputs",str(source[year]),"--output",str(dest),
                  "--feature-sets",",".join(entry["feature_sets"]),
                  "--prediction-phase","FINAL"]
            for flag,key in (("--actor-prefixes","actor_prefixes"),
                             ("--auto-slices","auto_slices"),
                             ("--pedigree-slices","pedigree_slices")):
                if entry.get(key): args.extend((flag,",".join(entry[key])))
            subprocess.run(args,cwd=ROOT,check=True)
            projections.append(dest)
        train_one(alias,a.prior_meta_root/alias/"model.json",
                  a.prior_meta_root/alias/"schema.json",projections,
                  a.models_out/alias,a.protocol_reference)
        meta=json.loads((a.models_out/alias/"model.json").read_text(encoding="utf-8"))
        assert meta["deployment_status"]=="CANDIDATE_2026_BLIND_UNRELEASED"
        assert meta["training_rule_approval_ref"] is None
        status.append({"alias":alias,"features":count,
                       "best_iteration":meta["best_iteration"],
                       "training_rows":meta["training_rows"],
                       "model_sha256":meta["model_sha256"]})
        for file in projections: file.unlink()
        gc.collect()
        print("ORIGINAL_2026_BLIND_RESEARCH_REFIT_ONE_KING "+
              json.dumps(status[-1],sort_keys=True),flush=True)
    assert len(status)==7
    record={"contract":"KEIBA_ORIGINAL_7KING_2026_UNRELEASED_REFIT_V1",
            "kings":status,"models":len(status),
            "training_years":[2024,2025],
            "source_snapshot_generation":"eb13d4096519167b",
            "method":"2025 OOS best_iteration fixed for prior-only refit",
            "2026_outcomes_used":False,
            "2026_validation_used":False,
            "released":False,"ready_for_betting":False}
    (a.models_out/"refit-index.json").write_text(json.dumps(record,ensure_ascii=False,
                                           indent=2)+"\n",encoding="utf-8")
    print("ALL_7_ORIGINAL_2026_UNRELEASED_REFITS_TRAINED "+
          json.dumps({"kings":7,"2026_outcomes_used":False,
                      "released":False},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
