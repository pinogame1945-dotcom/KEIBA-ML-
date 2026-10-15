#!/usr/bin/env python3
"""Execute the seven ORIGINAL trained models from *an earlier year* on one
historical superset snapshot race. Model selection is prior-year only.
Not a verified pre-bet prediction: source features are archived FINAL phase.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,math
from datetime import date
from pathlib import Path

EXPERTS=("core4","pedlegacy","condition","full","light","pedv1","condrc")

def lines(path):
    op=gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",encoding="utf-8") as stream:
        for line in stream:
            if line.strip():yield json.loads(line)

def run(snapshot,bundles,race_id):
    import lightgbm as lgb
    import pandas as pd
    if len(race_id)!=12 or not race_id.isascii() or not race_id.isdigit():
        raise ValueError("invalid canonical race ID")
    year=int(race_id[:4])
    if not 2022<=year<=2025:raise ValueError("Only 2022..2025 historical years, 2026 sealed")
    rows=[]
    for source in lines(snapshot):
        if str(source.get("race_id") or "")!=race_id:continue
        if source.get("feature_schema_version")!=8 or source.get("leakage_policy")!="STRICT_PRIOR_DATE_ONLY":
            raise ValueError("TARGET_SNAPSHOT_FEATURE_CONTRACT_DRIFT")
        # Never use outcomes, final odds, labels, margins or entry order.
        rows.append({"horse_id":str(source.get("horse_id") or ""),
                     "features":dict(source.get("features") or {})})
    horses={row["horse_id"] for row in rows}
    if len(rows)<3 or len(horses)!=len(rows):
        raise ValueError("target race absent or duplicate horse IDs")
    date_field={str(row["features"].get("race_date") or "")[:10] for row in rows}
    if len(date_field)!=1:raise ValueError("inconsistent target race date")
    race_date=date.fromisoformat(next(iter(date_field)))
    if race_date.year!=year:raise ValueError("race year mismatch")
    rank_maps={}
    probabilities={}
    provenance=[]
    for alias in EXPERTS:
        folder=Path(bundles)/alias
        meta=json.loads((folder/"model.json").read_text(encoding="utf-8"))
        schema=json.loads((folder/"schema.json").read_text(encoding="utf-8"))
        model=folder/"model.txt"
        checksum=hashlib.sha256(model.read_bytes()).hexdigest()
        if checksum!=meta.get("reproducibility",{}).get("model_sha256") or checksum!=schema.get("reproducibility",{}).get("model_sha256"):
            raise ValueError("artifact SHA mismatch "+alias)
        split=meta.get("split") or {}
        valid_end=date.fromisoformat(str(split.get("valid_end")))
        train_end=date.fromisoformat(str(split.get("train_end")))
        if not(train_end<valid_end<race_date):
            raise ValueError("MODEL_SELECTION_OR_VALIDATION_SEES_TARGET_DATE "+alias)
        if valid_end.year!=year-1:
            raise ValueError("Expect immediately previous-year frozen model "+alias)
        if meta.get("prediction_phase")!="FINAL" or schema.get("prediction_phase")!="FINAL":
            raise ValueError("FINAL model contract required "+alias)
        if meta.get("dataset_contract",{}).get("feature_schema_version")!=8:
            raise ValueError("feature schema drift "+alias)
        features=schema["feature_order"]
        if features!=meta.get("features"):raise ValueError("feature order mismatch "+alias)
        levels=schema["category_levels"]
        raw=[]
        for row in rows:
            vec={k:v for k,v in row["features"].items() if k not in ("race_date","actual_start_time","jockey_id","trainer_id")}
            vec["race_month"]=race_date.month
            vec["race_day_of_year"]=race_date.timetuple().tm_yday
            raw.append({k:vec.get(k) for k in features})
        frame=pd.DataFrame(raw,columns=features)
        for key in features:
            if key in levels:
                values=frame[key].astype("string").fillna("__MISSING__")
                frame[key]=pd.Categorical(values,categories=levels[key])
            else:
                frame[key]=pd.to_numeric(frame[key],errors="coerce")
        booster=lgb.Booster(model_file=str(model))
        if booster.feature_name()!=features:raise ValueError("LightGBM feature order mismatch "+alias)
        raw_probs=booster.predict(frame,num_iteration=int(meta["best_iteration"]))
        score={row["horse_id"]:float(p) for row,p in zip(rows,raw_probs)}
        if set(score)!=horses or any(not math.isfinite(v) or not 0<=v<=1 for v in score.values()):
            raise ValueError("invalid single race model scores "+alias)
        order=sorted(horses,key=lambda h:(-score[h],h))
        rank_maps[alias]={hid:i+1 for i,hid in enumerate(order)}
        probabilities[alias]=score
        provenance.append({"alias":alias,"train_end":train_end.isoformat(),
                           "model_selected_no_later_than":valid_end.isoformat(),
                           "validation_precedes_target":True,"sha256":checksum,
                           "features":len(features)})
    result=[]
    for hid in horses:
        ranks={name:rank_maps[name][hid] for name in EXPERTS}
        arr=list(ranks.values())
        result.append({
            "horse_id":hid,
            "king_ranks":ranks,
            "king_raw_probabilities":{name:probabilities[name][hid] for name in EXPERTS},
            "rank_sum":sum(arr),
            "top6_king_votes":sum(x<=6 for x in arr),
            "top3_king_votes":sum(x<=3 for x in arr),
            "top1_king_votes":sum(x==1 for x in arr),
        })
    result.sort(key=lambda row:(row["rank_sum"],-row["top6_king_votes"],
        -row["top3_king_votes"],-row["top1_king_votes"],row["horse_id"]))
    for index,row in enumerate(result,start=1):
        row["prior_year_7king_rank_sum_comparator_rank"]=index
    return {
        "contract":"KEIBA_PRIOR_YEAR_MODEL_SINGLE_RACE_V1",
        "race_id":race_id,"race_date":race_date.isoformat(),"model_count":7,
        "model_parameters_selected_before_target":True,
        "snapshot_prediction_phase":"FINAL_ARCHIVED_TIMESTAMP_NOT_VERIFIED",
        "historically_verified_live_bet":False,
        "ensemble_note":"Seven independent original prior-year trained models; rank sum comparator M127_S0 for inspection only, not frozen L2.1 router",
        "input_outcome_labels_used":False,
        "horse_count":len(rows),"models":provenance,"horses":result,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot",required=True)
    p.add_argument("--bundles",required=True)
    p.add_argument("--race",required=True)
    p.add_argument("--output")
    args=p.parse_args()
    result=run(args.snapshot,args.bundles,args.race)
    if args.output:
        f=Path(args.output);f.parent.mkdir(parents=True,exist_ok=True)
        f.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("PRIOR_YEAR_SEVEN_KING_ORIGINAL_MODELS_SCORED",result["race_id"],result["horse_count"],flush=True)
    for row in result["horses"][:6]:
        print("PRIOR_YEAR_MODEL_RANK",row["prior_year_7king_rank_sum_comparator_rank"],
              row["horse_id"],row["top3_king_votes"],row["rank_sum"],flush=True)

if __name__=="__main__":main()
