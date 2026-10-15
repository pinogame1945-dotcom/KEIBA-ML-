#!/usr/bin/env python3
"""Extract one 2021-2025 race's feature vectors from verified Kaggle
superset snapshot in ONE streaming pass. Never store target/outcome fields.
The resulting tiny JSONL is reused across all model families on one runner.
"""
import argparse,gzip,json,re
from pathlib import Path

def extract(input_path,race_id,output_path):
    if not re.fullmatch(r"202[1-5]\d{8}",race_id):
        raise ValueError("only known historical 2021..2025 canonical races; 2026 sealed")
    source=Path(input_path)
    reader=gzip.open if source.suffix==".gz" else open
    selected={}
    with reader(source,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            if str(row.get("race_id") or "")!=race_id:continue
            hid=str(row.get("horse_id") or "")
            if not hid or hid in selected:
                raise ValueError("missing/duplicated horse in original Kaggle snapshot")
            if row.get("feature_schema_version")!=8 or row.get("leakage_policy")!="STRICT_PRIOR_DATE_ONLY":
                raise ValueError("wrong snapshot feature/leakage contract")
            features=row.get("features")
            if not isinstance(features,dict) or not features:
                raise ValueError("missing horse features")
            # Do not forward the target, finish, payout, final odds, or
            # other original archive content, even though source contains it.
            selected[hid]={
                "race_id":race_id,"horse_id":hid,
                "feature_schema_version":8,
                "leakage_policy":"STRICT_PRIOR_DATE_ONLY",
                "features":features,
            }
    if len(selected)<3 or len(selected)>24:
        raise ValueError("target race not found or field size outside 3..24")
    d=Path(output_path)
    d.parent.mkdir(parents=True,exist_ok=True)
    with d.open("w",encoding="utf-8") as f:
        for h in sorted(selected):
            f.write(json.dumps(selected[h],ensure_ascii=False,separators=(",",":"))+"\n")
    print("ONE_RACE_SUPERSET_FEATURES_EXTRACTED",race_id,len(selected),
          "all prediction model families reuse",d,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--race",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    extract(a.input,a.race,a.output)
