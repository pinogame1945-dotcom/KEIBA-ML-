#!/usr/bin/env python3
"""Frozen ML speed-regime 22 numerical parity using existing daily result packs.

2019..2021 snapshots are not fetched: original 471 target horses have exactly
their stored prior_starts in the strictly-prior 2022..2026-10-16 BACKFILL packs.
For each historical race needed by the 471 horses, take ALL finished peers and
reconstruct the original 2025 speed_mps -> race median -> speed_resid float32.
Re-execute immutable ORIGINAL ML horse_features via its original AST nodes,
not a newly invented speed/regime formula. No feature/model full rebuild,
Kaggle dataset, outcomes of 2026-10-17, payout, or actual betting.
"""
from __future__ import annotations
import argparse,ast,gzip,hashlib,json,math,re
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_DAY="2026-10-17"
FIRST_DAY="2022-01-01"
ML_HORSE_CODE_SHA="e048b4af2a64b2ddc2f99a3fff3b1e7354090482"
ML_STATE_SHA="f02cfa39bd378445a69ca11e588abf1010b9acb1"
ML_REGIME_SHA="1d236a07dcf0ca1722b8448a269f28c7232f3d0f"
SPEED_PREFIX="regime_speed_resid_"
FORBIDDEN_STATUSES={"SCRATCHED","EXCLUDED"}

def need(check,reason):
    if not check:raise ValueError(reason)

def gitblob(path):
    data=Path(path).read_bytes()
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\x00"+data).hexdigest()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def literal_assign(tree,name):
    options=[]
    for stmt in tree.body:
        if isinstance(stmt,ast.Assign):
            if any(isinstance(t,ast.Name) and t.id==name for t in stmt.targets):
                options.append(ast.literal_eval(stmt.value))
    need(len(options)==1,"FROZEN_ML_ORIGINAL_CONSTANT_NOT_FOUND:"+name)
    return options[0]

def original_functions(horse_source,state_source,regime_source):
    for path,digest in ((horse_source,ML_HORSE_CODE_SHA),
                        (state_source,ML_STATE_SHA),
                        (regime_source,ML_REGIME_SHA)):
        need(gitblob(path)==digest,"ORIGINAL_ML_SOURCE_NOT_FROZEN:"+str(path))
    funcs=ast.parse(horse_source.read_text("utf8"))
    state=ast.parse(state_source.read_text("utf8"))
    regime=ast.parse(regime_source.read_text("utf8"))
    names={"tofloat","std0","mean","horse_features"}
    selected=[stmt for stmt in funcs.body
              if isinstance(stmt,ast.FunctionDef) and stmt.name in names]
    need({z.name for z in selected}==names,"ORIGINAL_ML_HORSE_FUNCTIONS_MISSING")
    cfg={k:literal_assign(state,k) for k in
         ("RAW_LAG_METRICS","TRANSITION_METRICS","RESIDUAL_METRICS","MAX_LAG")}
    cfg["RESIDUAL_AXES"]=literal_assign(regime,"RESIDUAL_AXES")
    need(cfg["MAX_LAG"]==5 and
         "speed_resid" in cfg["RESIDUAL_METRICS"] and
         "speed_resid" in cfg["RESIDUAL_AXES"],
         "ORIGINAL_ML_SPEED_MODEL_CONTRACT_CHANGED")
    namespace={"np":np,"pd":pd,"math":math,**cfg}
    exec(compile(ast.Module(body=selected,type_ignores=[]),
                 str(horse_source),"exec"),namespace)
    return namespace["horse_features"]

def numeric(v):
    try:
        if v is None or v=="":return float("nan")
        x=float(v)
        return x if math.isfinite(x) else float("nan")
    except (TypeError,ValueError):
        return float("nan")

def load_reference(root):
    folder=root/"ml_features/delivery/690"
    idx=json.loads((folder/"index.json").read_text(encoding="utf8"))
    need(idx.get("contract")=="KEIBA_BACKFILL_TO_ML_690_INDEX_V1"
         and idx.get("total_races")==60 and idx.get("total_horses")==786
         and idx.get("feature_count")==690
         and idx.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET"
         and idx.get("original_ml_full_numerical_parity_verified") is False,
         "CACHED_690_INDEX_INVALID")
    selected={rid:v for rid,v in idx["races"].items() if v["date"]==TARGET_DAY}
    need(len(selected)==36 and sum(v["starters"] for v in selected.values())==471,
         "EXPECTED_36_RACES_471_HORSES")
    targets={};counts={};speed_names=None
    for rid,item in sorted(selected.items()):
        p=root/item["path"]
        need(p.is_file() and sha(p)==item["sha256"],
             "SAVED_TARGET_690_RACE_SHA_MISMATCH:"+rid)
        doc=json.loads(p.read_text(encoding="utf8"))
        need(doc.get("contract")=="KEIBA_BACKFILL_TO_ML_690_UNSCORED_V1" and
             doc.get("race_date")==TARGET_DAY and doc.get("race_id")==rid and
             doc.get("starters")==item["starters"] and
             doc.get("used_target_results_or_final_odds") is False and
             doc.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET",
             "UNSAFE_OR_CHANGED_SAVED_690_RACE:"+rid)
        for row in doc["rows"]:
            hid=str(row["_horse_id"])
            k=(rid,hid)
            need(k not in targets and
                 row["_race_date"]==TARGET_DAY and row["_race_id"]==rid and
                 type(row.get("prior_starts")) in (int,float) and
                 row["prior_starts"]>=0 and
                 int(row["prior_starts"])==row["prior_starts"],
                 "INVALID_TARGET_HORSE_OR_HISTORY_COUNT:"+rid+":"+hid)
            keys={x for x in row if x.startswith(SPEED_PREFIX)}
            if speed_names is None:speed_names=keys
            need(len(keys)==22 and keys==speed_names,
                 "NOT_22_FROZEN_ML_SPEED_FIELDS:"+rid+":"+hid)
            targets[k]={s:row[s] for s in keys}
            counts[hid]=int(row["prior_starts"])
    need(len(targets)==471 and len(counts)==471,
         "REAL_471_TARGETS_NOT_UNIQUE")
    return targets,counts,speed_names

def collect_prior(raw_daily,ids):
    source=Path(raw_daily)
    names=sorted(f for f in source.glob("*.jsonl.gz")
                 if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.jsonl\.gz",f.name)
                 and FIRST_DAY<=f.name[:10]<TARGET_DAY)
    need(len(names)==517 and
         sum(x.name.startswith("2026-") for x in names)==87 and
         names[0].name.startswith("2022-") and names[-1].name.startswith("2026-10-16"),
         "ORIGINAL_2022_TO_TARGET_PRIOR_SOURCE_DATES_NOT_517")
    historic=defaultdict(list);counts=Counter()
    races_scanned=0; race_has_target=0; peer_values=0
    for filename in names:
        date=filename.name[:10]
        with gzip.open(filename,"rt",encoding="utf8") as f:
            for line in f:
                if not line.strip():continue
                row=json.loads(line)
                race=row.get("race") or {}
                rid=str(race.get("race_id") or "")
                day=str(race.get("actual_date") or race.get("scheduled_date") or "")[:10]
                need(day==date and rid.isdigit() and len(rid)==12 and day<TARGET_DAY,
                     "PRIOR_RACE_DATE_OR_ID_NOT_AUTHENTIC:"+filename.name)
                races_scanned+=1
                entries={str(e.get("horse_id")):e for e in row.get("entries",[])}
                results={str(r.get("horse_id")):r for r in row.get("results",[])}
                needed=[]
                for hid in ids:
                    if hid not in entries:continue
                    e=entries[hid];r=results.get(hid)
                    if r is None:continue
                    if str(e.get("entry_status") or "").upper() in FORBIDDEN_STATUSES or str(r.get("result_status") or "").upper() in FORBIDDEN_STATUSES:
                        continue
                    counts[hid]+=1
                    needed.append(hid)
                if not needed:continue
                race_has_target+=1
                distance=numeric(race.get("distance_m"))
                peers=[]
                for hid,r in results.items():
                    e=entries.get(hid)
                    if e is None:continue
                    if str(e.get("entry_status") or "").upper() in FORBIDDEN_STATUSES or str(r.get("result_status") or "").upper() in FORBIDDEN_STATUSES:
                        continue
                    finish=numeric(r.get("official_finish_position"))
                    if not(math.isfinite(finish) and finish>=1):continue
                    time_ms=numeric(r.get("finish_time_ms"))
                    # ORIGINAL ML state builder uses float32 before race median.
                    sec=time_ms/1000.
                    speed=np.float32(distance/sec) if (math.isfinite(distance) and
                          math.isfinite(sec) and sec>0) else np.float32(np.nan)
                    peers.append((hid,float(speed)))
                # Reproduce original Pandas median() and residual float32 stages.
                if peers:
                    frame=pd.DataFrame(peers,columns=["_horse_id","speed_mps"])
                    frame["speed_mps"]=frame["speed_mps"].astype("float32")
                    med=frame["speed_mps"].median()
                    resid=(frame["speed_mps"].astype(float)-float(med)).astype("float32")
                    found=dict(zip(frame["_horse_id"],resid))
                    for hid in needed:
                        if hid in found:
                            historic[hid].append({"_race_id":rid,"_race_date":day,
                                                   "speed_resid":float(found[hid])})
                            peer_values+=1
    need(race_has_target>2500 and peer_values>3000,
         "SOURCE_FINISHED_PRIOR_PEERS_MISSING")
    return historic,counts,{"historical_files":len(names),
        "historical_races_scanned":races_scanned,
        "historic_races_for_target_horses":race_has_target,
        "finished_target_horse_events_with_peer_median":peer_values}

def compare(targets,counts,speed_names,historic,observed,functions,output,stats):
    need(all(int(observed[h])==want for h,want in counts.items()),
         "BACKFILL_ORIGINAL_PRIOR_START_COUNT_DISAGREES_WITH_690")
    per_horse={};categories=Counter();numeric_drift=Counter();examples=[]
    total=0;races=set();non_null=0
    for (rid,hid),expected in sorted(targets.items()):
        events=historic.get(hid,[])
        frame=pd.DataFrame.from_records(events,columns=["_race_id","_race_date","speed_resid"])
        values=functions(frame,TARGET_DAY)
        found={k:v for k,v in values.items() if k.startswith(SPEED_PREFIX)}
        need(set(found)==speed_names,
             "FROZEN_ORIGINAL_ML_SPEED22_SCHEMA_NOT_THE_SAME:"+hid)
        for col in sorted(speed_names):
            value=found[col]
            actual=None if not math.isfinite(float(value)) else float(value)
            cached=expected[col]
            if cached is None and actual is None:category="equal_null"
            elif type(cached) in (int,float) and actual is not None and cached==actual:
                category="exact"
                non_null+=1
            elif cached is not None and actual is not None and math.isclose(
                    float(cached),actual,rel_tol=1e-5,abs_tol=1e-5):
                category="approx_only"
            else:
                category="mismatch"
            categories[category]+=1
            total+=1
            if category in ("approx_only","mismatch"):
                numeric_drift[col]+=1
                if len(examples)<25:
                    examples.append({"race_id":rid,"horse_id":hid,"field":col,
                                     "cached":cached,"original_ml":actual,"kind":category})
        per_horse[(rid,hid)] = len(events)
        races.add(rid)
    need(total==471*22 and len(races)==36,
         "NOT_36_RACES_471_HORSES_22_VALUES")
    passed=not(categories["approx_only"] or categories["mismatch"])
    report={
        "contract":"KEIBA_2026_1017_ORIGINAL_ML_SPEED_REGIME_22_NUMERIC_PARITY_V1",
        "races":len(races),"horses":len(targets),"columns":22,"cells":total,
        "categories":dict(categories),"not_null_numeric_exact":non_null,
        "original_frozen_ml_formula_directly_invoked":True,
        "source_year_first":2022,"prior_dated_max":"2026-10-16",
        "prior_starts_all_471_equal_cached_690":True,
        "real_historical_source":stats,
        "per_column_differences":dict(numeric_drift),
        "first_differences":examples,
        "all_22_original_ml_numerical_exact":passed,
        "original_other_19_daybias_success_runs":[39105758197,39106692871],
        "original_649_full_numeric_parity_certified":False,
        "entire_690_numeric_parity_certified":False,
        "target_day_outcomes_used":False,"model_fit":False,
        "l21_tickets_generated":False,"l3_money_generated":False,
        "ready_for_betting":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf8")
    print("ORIGINAL_ML_SPEED_REGIME_22_PARITY "
          +json.dumps({"races":len(races),"horses":len(targets),"cells":total,
                        "categories":dict(categories),"exact":passed,
                        "betting":False},sort_keys=True),flush=True)
    need(passed,"ORIGINAL_ML_SPEED_REGIME_22_NOT_EXACT_PARITY")

def main():
    ap=argparse.ArgumentParser()
    for key in ("daily-source","saved-690","original-ml-seq",
                "original-ml-state","original-ml-regime","output"):
        ap.add_argument("--"+key,type=Path,required=True)
    a=ap.parse_args()
    horsefn=original_functions(a.original_ml_seq,a.original_ml_state,a.original_ml_regime)
    targets,counts,names=load_reference(a.saved_690)
    historic,observed,stats=collect_prior(a.daily_source,set(counts))
    compare(targets,counts,names,historic,observed,horsefn,a.output,stats)

if __name__=="__main__":main()
