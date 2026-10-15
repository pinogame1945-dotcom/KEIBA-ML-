#!/usr/bin/env python3
"""Score all registered models for one race. No result/odds or auto-training."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math,sys
from pathlib import Path

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def read_score_rows(path):
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf-8",newline="") as f:
        if str(path).endswith((".jsonl",".jsonl.gz")):
            for line in f:
                if line.strip():yield json.loads(line)
        else:
            yield from csv.DictReader(f)

def ranked(horses,values):
    if set(values)!=set(h["horse_id"] for h in horses):
        raise ValueError("model horse coverage differs from target starters")
    if any(not math.isfinite(float(v)) for v in values.values()):
        raise ValueError("NaN or infinite model score")
    # Stable by canonical horse ID; never use source/finish position row order.
    ordered=sorted(values, key=lambda h:(-float(values[h]),h))
    return [{"rank":i+1,"horse_id":h,"horse_name":next(x.get("horse_name") for x in horses if x["horse_id"]==h),
             "horse_number":next(x.get("horse_number") for x in horses if x["horse_id"]==h),
             "score":float(values[h])} for i,h in enumerate(ordered)]

def guard_metadata(meta,race):
    if meta.get("feature_schema_version") not in (None,race["feature_schema_version"]):
        raise ValueError("feature schema mismatch")
    if meta.get("prediction_phase")!=race["phase"]:
        raise ValueError("prediction phase mismatch")
    if meta.get("feature_sets")!=race["features_by_horse"][0]["feature_sets"]:
        raise ValueError("feature sets mismatch")
    if meta.get("history_windows")!=race["features_by_horse"][0]["history_windows"]:
        raise ValueError("history windows mismatch")
    if meta.get("small_sample_policy")!=race["features_by_horse"][0]["small_sample_policy"]:
        raise ValueError("small sample policy mismatch")
    split=meta.get("split") or {}
    cutoff=max(str(split.get("train_end") or ""),str(split.get("valid_end") or ""))
    if not cutoff:
        raise ValueError("model metadata lacks train/validation end")
    if cutoff>=race["race_date"]:
        raise ValueError("AS_OF_LEAKAGE: model trained/selected with target or later results")

def score_lightgbm(spec,race,root):
    import pandas as pd
    import lightgbm as lgb
    model=(root/spec["model_path"]).resolve()
    meta_path=(root/spec["metadata_path"]).resolve()
    if not model.is_file() or not meta_path.is_file():
        raise FileNotFoundError("model and metadata must exist")
    meta=read_json(meta_path)
    guard_metadata(meta,race)
    expected=meta.get("reproducibility",{}).get("model_sha256")
    if not expected or sha256(model)!=expected:raise ValueError("model SHA mismatch")
    features=meta.get("features")
    cats=meta.get("categorical_features",[])
    levels=meta.get("category_levels",{})
    if not isinstance(features,list) or not features or len(set(features))!=len(features):
        raise ValueError("missing/nonunique ordered feature schema")
    df=pd.DataFrame([r["features"] for r in race["features_by_horse"]])
    missing=[f for f in features if f not in df.columns]
    if missing:raise ValueError("model features absent: "+",".join(missing[:12]))
    df=df[features].copy()
    for f in features:
        if f in cats:
            if f not in levels:raise ValueError("missing categorical levels: "+f)
            text=df[f].astype("string").fillna("__MISSING__")
            df[f]=pd.Categorical(text,categories=levels[f])
        else:
            df[f]=pd.to_numeric(df[f],errors="coerce")
    booster=lgb.Booster(model_file=str(model))
    if booster.feature_name()!=features:
        raise ValueError("LightGBM model feature name/order drift")
    raw=booster.predict(df,num_iteration=int(meta.get("best_iteration") or 0) or None)
    return {row["horse_id"]:float(p) for row,p in zip(race["features_by_horse"],raw)}

def score_archived(spec,race,root):
    # These values are *historical saved OOF predictions*, not a new invocation.
    path=(root/spec["score_path"]).resolve()
    meta=read_json((root/spec["metadata_path"]).resolve())
    if not meta.get("historical_oof",False):
        raise ValueError("archived model must declare historical_oof=true")
    guard_metadata(meta,race)
    field=spec["score_field"]
    selected={}
    orientation=spec.get("score_direction","desc")
    if orientation not in ("asc","desc"):raise ValueError("invalid score_direction")
    for row in read_score_rows(path):
        if str(row.get(spec.get("race_field","race_id")))==race["race_id"]:
            if spec.get("required_contract") and row.get("contract")!=spec["required_contract"]:
                raise ValueError("archived score contract mismatch")
            hid=str(row.get(spec.get("horse_field","horse_id")) or "")
            if hid in selected:raise ValueError("duplicate archived horse: "+hid)
            raw=float(row[field])
            selected[hid]=raw if orientation=="desc" else -raw
    if not selected:raise ValueError("race missing from score archive")
    return selected

def score_ensemble(spec,scored,horses):
    members=spec.get("members") or []
    weights=spec.get("weights") or [1]*len(members)
    if not members or len(weights)!=len(members) or any(float(w)<=0 for w in weights):
        raise ValueError("invalid ensemble members/weights")
    if any(m not in scored for m in members):
        raise ValueError("ensemble member missing: "+",".join(m for m in members if m not in scored))
    kind=spec.get("method","rank_average")
    byhorse={h["horse_id"]:0.0 for h in horses}
    for member,weight in zip(members,weights):
        ranks={x["horse_id"]:x["rank"] for x in scored[member]["ranking"]}
        for hid in byhorse:
            if kind=="rank_average":
                byhorse[hid]-=float(weight)*ranks[hid]
            elif kind=="top3_vote":
                byhorse[hid]+=float(weight)*max(0,4-ranks[hid])
            else:raise ValueError("unsupported ensemble method "+kind)
    return byhorse

def evaluate(race,registry,root):
    if race.get("contract")!="KEIBA_ONE_RACE_ASOF_V1":raise ValueError("unknown input contract")
    horses=race.get("features_by_horse") or []
    if not horses:raise ValueError("no starters")
    ids=[str(x["horse_id"]) for x in horses]
    if len(set(ids))!=len(ids):raise ValueError("duplicate input horse ID")
    if any(x.get("target") is not None or x.get("market_outcome") is not None for x in horses):
        raise ValueError("target outcomes are forbidden in inference input")
    report={"contract":"KEIBA_ALL_MODELS_ONE_RACE_V1","race_id":race["race_id"],
            "race_date":race["race_date"],"prediction_phase":race["phase"],
            "data_status":race["status"],"model_results":[]}
    scored={}
    names=set()
    for spec in registry.get("models",[]):
        name=spec["name"]
        if name in names:raise ValueError("duplicate model name: "+name)
        names.add(name)
        record={"model":name,"kind":spec.get("kind"),"status":"UNAVAILABLE"}
        try:
            if spec.get("kind")=="unconfigured":
                record["reason"]="MISSING_PROMOTED_MODEL_OR_OOF_SCORE"
            elif spec.get("kind")=="lightgbm":
                values=score_lightgbm(spec,race,root)
                record.update(status="SCORED",origin="NEW_INFERENCE",ranking=ranked(horses,values))
            elif spec.get("kind")=="archived_oof":
                values=score_archived(spec,race,root)
                record.update(status="SCORED",origin="HISTORICAL_SCORE_REPLAY",ranking=ranked(horses,values))
            else:raise ValueError("unknown model kind")
        except (ValueError,KeyError,FileNotFoundError,ImportError) as exc:
            record.update(status="BLOCKED",reason=str(exc))
        report["model_results"].append(record)
        if record["status"]=="SCORED":scored[name]=record
    for spec in registry.get("ensembles",[]):
        name=spec["name"]
        if name in names:raise ValueError("duplicate model/ensemble name: "+name)
        names.add(name)
        record={"model":name,"kind":"ensemble","status":"BLOCKED"}
        try:
            values=score_ensemble(spec,scored,horses)
            record.update(status="SCORED",origin="DETERMINISTIC_REGISTERED_ENSEMBLE",ranking=ranked(horses,values))
        except (ValueError,KeyError) as exc:record["reason"]=str(exc)
        report["model_results"].append(record)
    report["scored_models"]=sum(x["status"]=="SCORED" for x in report["model_results"])
    report["blocked_or_missing_models"]=len(report["model_results"])-report["scored_models"]
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--registry",default="models/one-race-registry.json")
    ap.add_argument("--output")
    ap.add_argument("--require-scored",action="store_true")
    args=ap.parse_args()
    registry_path=Path(args.registry).resolve()
    report=evaluate(read_json(args.input),read_json(registry_path),registry_path.parent)
    result=json.dumps(report,ensure_ascii=False,indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(result+"\n",encoding="utf-8")
    print(result)
    if args.require_scored and report["scored_models"]==0:
        raise SystemExit("No promoted as-of model/archived OOF scores available; no fake rankings produced")

if __name__=="__main__":main()
