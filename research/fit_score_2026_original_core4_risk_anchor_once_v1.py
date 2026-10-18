#!/usr/bin/env python3
"""2026 blind original LambdaRank + past-OOS Correction Risk + original ANCHOR_FULL.

Only frozen 2019-25 originals, 2026 target 690 prestart candidate and JRA WIN
market. No 2026 outcomes, results, payout, FINAL prices, or L2.1 ticket engines.
Reuses exact original formula modules pinned to feature/unified-race-runtime-20261016.
Models and scores are unapproved research candidates pending as-of certification.
"""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,importlib.util,json,math
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from run_l1_objective_rebuild_v1 import META,model_common,prepare,BASE_CATEGORICAL
from run_l1_market_correction_distill_v1 import add_ai_context
from run_l1_residual_regime_full_pairwise_v1 import build_regime_features
from run_l1_sameday_track_bias_v1 import bias_cols,speed_cols
import run_l1_state_transition_v1 as state
import rebuild_blind_original_2025_risk_from_prior_oos_v1 as prior_risk
from run_l1_market_correction_deepdive_v1 import risk_params

PRIOR_CONTRACT="KEIBA_ORIGINAL_2026_PRIOR_2019_2025_ONCE_V1"
L175_INDEX="KEIBA_L175_BACKFILL_2026_1017_RESEARCH_PRIVATE_V1"
L175_RACE="KEIBA_2026_1017_L175_MARKET_JOIN_RESEARCH_V1"
TARGET_INDEX="KEIBA_BACKFILL_TO_ML_690_INDEX_V1"
ORIGINAL_BASE="eb3392b725560e77d46b28aef036b6188814125ed8f7a3bfde0c53522dbadf58"
ORIGINAL_ANCHOR="KEIBA_ORIGINAL_ANCHOR_FULL_PRIOR_TRAINED_PARAMS_V1"
RACE_DAY="2026-10-17"
FIT_YEARS=(2024,2025)
TEACHER_YEARS=(2022,2023,2024,2025)
RACE_COUNT=36
HORSE_COUNT=471

def need(x,msg):
    if not x:raise ValueError(msg)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for ch in iter(lambda:f.read(1024*1024),b""):h.update(ch)
    return h.hexdigest()

def get(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def read_frame(p):
    return pd.read_csv(p,compression="infer",low_memory=False,
       dtype={"_race_id":"string","_horse_id":"string","_race_date":"string"})

def keys(frame):
    return set(map(tuple,frame[["_race_id","_horse_id"]].to_numpy()))

def from_prior_csv(pair):
    y,p=pair
    f=read_frame(p)
    need(f["_year"].eq(y).all() and
       f["_race_id"].astype(str).str[:4].eq(str(y)).all() and
       f["_race_date"].astype(str).lt(RACE_DAY).all(),"ORIGINAL_STATE_DATE_LEAK")
    return y,f

def original_regimes(derived):
    # Same exact state+regime operators used by the original 2025 replay.
    compacts={y:str(derived/f"compact-strict-prior-{y}.csv.gz")
              for y in range(2019,2026)}
    old_reader=state._read_state_year_item
    old_pool=state.concurrent.futures.ProcessPoolExecutor
    try:
        state._read_state_year_item=from_prior_csv
        state.concurrent.futures.ProcessPoolExecutor=concurrent.futures.ThreadPoolExecutor
        historical,*others=state.build_state_features(compacts)
    finally:
        state._read_state_year_item=old_reader
        state.concurrent.futures.ProcessPoolExecutor=old_pool
    regimes,columns=build_regime_features(historical)
    return historical,regimes,columns

def verify_prior(root):
    idx=get(root/"index.json")
    need(idx.get("contract")==PRIOR_CONTRACT and
         idx.get("only_prior_years")==list(range(2019,2026)) and
         idx.get("original_core4_fit_years")==list(FIT_YEARS) and
         idx.get("risk_teacher_years")==list(TEACHER_YEARS) and
         idx.get("2026_target_outcomes_used") is False and
         idx.get("model_fit_performed") is False,"PRIOR_MODEL_INPUT_SOURCE_OR_SCOPE_DRIFT")
    files=idx["files"]
    need(len(files)==23,"HISTORICAL_PRIOR_SOURCE_FILES_NOT_23")
    for name,item in files.items():
        target=root/name
        need(target.is_file() and target.stat().st_size==item["bytes"] and
             sha(target)==item["sha256"],"ORIGINAL_PAST_YEAR_PRIOR_BYTES_DRIFT:"+name)
    return idx

def verify_target(backfill,market_index,market_file):
    i=get(backfill/"ml_features/delivery/690/index.json")
    m=get(market_index)
    need(i.get("contract")==TARGET_INDEX and i.get("feature_count")==690 and
         m.get("contract")==L175_INDEX and m.get("races")==36 and
         m.get("horses")==471 and
         m.get("market_asof_independently_verified") is False and
         m.get("ready_for_betting") is False and
         m["files"].get(market_file.name)==sha(market_file),
         "TARGET_MARKET_PROVENANCE_OR_SHA_NOT_AUTHENTIC")
    races={rid:v for rid,v in i["races"].items() if v["date"]==RACE_DAY}
    need(len(races)==36,"2026_1017_TARGET_RACE_SCOPE_DRIFT")
    expected_names=None;allrows=[];ids={}
    for rid,item in sorted(races.items()):
        docpath=backfill/item["path"]
        need(docpath.is_file() and sha(docpath)==item["sha256"],"TARGET_690_SHA_DRIFT:"+rid)
        doc=get(docpath)
        need(doc.get("contract")=="KEIBA_BACKFILL_TO_ML_690_UNSCORED_V1" and
          doc.get("race_id")==rid and doc.get("race_date")==RACE_DAY and
          doc.get("feature_count")==690 and
          doc.get("used_target_results_or_final_odds") is False and
          doc.get("source_capture_asof_independently_verified") is False,
          "TARGET_690_NOT_SEALED:"+rid)
        rows=doc.get("rows") or []
        need(len(rows)==item["starters"],"TARGET_690_HORSE_COUNT_DRIFT:"+rid)
        for h in rows:
            names=set(h)-{"_race_id","_horse_id","_race_date"}
            if expected_names is None:expected_names=names
            need(names==expected_names and len(names)==690 and
                 not {"_finish","_is_win","_is_top3","target","payout","result",
                      "final_win_odds","P__ANCHOR_FULL","risk"}.intersection(h),
                 "TARGET_690_FORBIDDEN_OR_SCHEMA_DRIFT:"+rid)
            hid=str(h["_horse_id"]);num=h.get("horse_number")
            need((rid,hid) not in ids and type(num) is int and num>0,
                 "TARGET_690_DUPLICATE_OR_INVALID_HORSE_NUMBER:"+rid)
            ids[(rid,hid)]=num
            allrows.append(h)
    f=pd.DataFrame.from_records(allrows)
    base=sorted(x for x in expected_names if not(x.startswith(("daybias_","horse_front_score","horse_inner_score","fit_","regime_speed_"))))
    need(len(base)==649 and
         hashlib.sha256(json.dumps(base,separators=(",",":")).encode()).hexdigest()==ORIGINAL_BASE and
         len(f)==HORSE_COUNT and f["_race_id"].nunique()==RACE_COUNT and
         not f.duplicated(["_race_id","_horse_id"]).any(),
         "2026_TARGET_ORIGINAL_649_NAME_SHA_OR_POPULATION_DRIFT")
    mk={}
    for line in market_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():continue
        race=json.loads(line)
        rid=race.get("race_id")
        need(rid not in mk and rid in races and
            race.get("contract")==L175_RACE and
            race.get("ready_for_betting") is False and
            race.get("market_asof_independently_verified") is False and
            race.get("l21_l3_invoked") is False and
            race.get("2026_target_results_used") is False and
            race.get("win_quote_full_field") is True and
            race.get("horse_count")==len(race.get("horses") or []),
            "2026_MARKET_RESEARCH_SOURCE_DRIFT")
        horses={}
        for h in race["horses"]:
            hid=str(h["horse_id"])
            num=h["horse_number"]
            odds=h["market_win_odds"]
            need((rid,hid) in ids and ids[(rid,hid)]==num and hid not in horses and
                 type(odds) in (int,float) and math.isfinite(odds) and odds>=1,
                 "2026_MARKET_HORSE_OR_PRICE_OR_IDENTITY_DRIFT:"+rid)
            horses[hid]=h
        mk[rid]=horses
    need(len(mk)==36 and
         sum(len(horses) for horses in mk.values())==HORSE_COUNT,
         "2026_ORIGINAL_36R_471H_MARKET_PROVENANCE_FAIL")
    return f,mk,ids

def validate_params(p,anchor_code):
    params=get(p)
    need(params.get("contract")==ORIGINAL_ANCHOR and
         params.get("trained_only_on_years")==[2022,2023,2024] and
         params.get("variant")=="ANCHOR_FULL" and
         params.get("features_in_original_order")==list(anchor_code.FEATURES) and
         params.get("market_quote_asof_required") is True and
         params.get("must_refuse_missing_prior_asof_companion_features") is True,
         "EXACT_ORIGINAL_2025_ANCHOR_PRIOR_MODEL_DRIFT")
    need(set(params["coefficients"])==set(params["standardizer"])==set(anchor_code.FEATURES),
         "ORIGINAL_ANCHOR_COEFF_NORMALIZER_SCHEMA_DRIFT")
    return params

def run(prior,backfill,market_index,market_file,anchor_params,
        original_anchor_py,model_dest):
    need(not model_dest.exists(),"REFUSE_EXISTING_MODEL_DEST_OVERWRITE")
    verify_prior(prior)
    # For reading historic projected data, workflow copies immutable prior bytes
    # (verified above) to files named exactly as original 2025 operators expect.
    originals={y:prior for y in range(2019,2026)}
    oldspec=importlib.util.spec_from_file_location("original_anchor_frozen_2025",original_anchor_py)
    oldmodule=importlib.util.module_from_spec(oldspec)
    oldspec.loader.exec_module(oldmodule)
    params=validate_params(anchor_params,oldmodule)
    valid,market,source_numbers=verify_target(backfill,market_index,market_file)
    states,regimes,regime_columns=original_regimes(prior)
    speed=speed_cols(regime_columns)
    bias_example=read_frame(prior/"original-2024-daybias.csv.gz")
    fit=bias_cols(bias_example,with_fit=True)
    need(len(speed)==22 and len(fit)==19,"ORIGINAL_19_BIAS_22_REGIME_SPEED_DRIFT")
    train_parts=[]
    for year in FIT_YEARS:
        base=read_frame(prior/f"original-{year}-core4-base.csv.gz")
        bias=read_frame(prior/f"original-{year}-daybias.csv.gz")
        need(len(base)==len(bias) and keys(base)==keys(bias),
             "ORIGINAL_CORE4_TRAIN_HORSE_SET_DRIFT")
        merged=bias.merge(regimes[year][["_race_id","_horse_id",*speed]],
                          on=["_race_id","_horse_id"],how="left",validate="one_to_one")
        joined=state.attach_state(base,merged,fit+speed)
        need(len(joined)==len(base),"ORIGINAL_CORE4_TRAIN_JOIN_LOSS")
        train_parts.append(joined)
    train=pd.concat(train_parts,ignore_index=True)
    need(set(train.columns)-META==set(valid.columns)-{"_race_id","_horse_id","_race_date"} and
         set(train["_race_id"].astype(str).str[:4].astype(int))==set(FIT_YEARS),
         "2026_ORIGINAL_CORE4_TRAIN_VS_690_INPUT_SCHEMA_NOT_IDENTICAL")
    train=train.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    valid=valid.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    xtr,xva,cats=prepare(train,valid)
    need(xtr.shape[1]==690 and list(xtr.columns)==list(xva.columns) and
         list(xtr.columns)==sorted(xtr.columns),"ORIGINAL_690_FULL_MODEL_COLUMNS_CHANGED")
    feature_order=list(xtr.columns)
    categories={x:list(xtr[x].cat.categories) for x in cats}
    relevance=np.select([train["_finish"].eq(j) for j in range(1,7)],
                        [6,5,4,3,2,1],default=0).astype(int)
    group=train.groupby("_race_id",sort=False).size().tolist()
    need(sum(group)==len(train),"2026_LAMBDARANK_HISTORICAL_TRAIN_GROUP_DRIFT")
    seed=5100000+2026*100
    ability=lgb.LGBMRanker(objective="lambdarank",metric="ndcg",
      label_gain=[0,1,3,7,15,31,63],**model_common(seed))
    ability.fit(xtr,relevance,group=group,categorical_feature=cats)
    z=np.asarray(ability.predict(xva),dtype=float)
    need(len(z)==HORSE_COUNT and np.isfinite(z).all(),"2026_ORIGINAL_ABILITY_SCORE_INVALID")
    model_dest.mkdir(parents=True,exist_ok=False)
    ability_path=model_dest/"original-core4-2026-lambdarank-model.txt"
    ability.booster_.save_model(str(ability_path))
    pred=valid[["_race_id","_horse_id"]].copy()
    pred["_score"]=z
    context=add_ai_context(pred)
    need(len(context)==HORSE_COUNT and
         all(sorted(x["distill_ai_rank"].astype(int).tolist())==list(range(1,len(x)+1))
             for _,x in context.groupby("_race_id")),"2026_ORIGINAL_AI_RANK_GAP")
    print("ORIGINAL_2026_PURE_LAMBDARANK_REAL_SCORE 36R_471H historical_fit=2024_2025",flush=True)
    # The original 2025 Risk teacher builds each year in historical walk-forward:
    # 2022 fits 2020/21, ... 2025 fits 2023/24. No 2026 labels.
    prior_risk.TEACHER_YEARS=TEACHER_YEARS
    teacher,checks=prior_risk.teacher_frames(prior,regimes,regime_columns)
    need(len(checks)==4 and
         [x["oos_test_year"] for x in checks]==list(TEACHER_YEARS) and
         all(x["metric_reference_exact"] for x in checks) and
         set(teacher["_race_id"].astype(str).str[:4].astype(int))==set(TEACHER_YEARS),
         "ORIGINAL_RISK_WALK_FORWARD_TEACHER_DRIFT")
    required=["_race_id","_horse_id","_score","distill_ai_rank",
       "distill_field_size","distill_ai_rank01","distill_score_z",
       "distill_score_gap_to_best","distill_is_ai_top1","distill_is_ai_top3",
       "distill_is_ai_top6"]
    source=valid.merge(context[required],on=["_race_id","_horse_id"],how="left",validate="one_to_one")
    need(len(source)==HORSE_COUNT and source["distill_ai_rank"].notna().all() and
         not {"_finish","_is_win","_is_top3","final_win_odds",
             "market_rank","risk","P__ANCHOR_FULL"}.intersection(source.columns),
         "ORIGINAL_2026_RISK_TARGET_POSTRACE_LEAK")
    source=source[[c for c in source if not c.startswith("neg_")]].copy()
    removed={"_finish","_is_win","_is_top3","_target_gap",
             "_target_gap_clip","_target_up","_target_over3","market_rank","final_win_odds"}
    need(set(teacher.columns)-removed==set(source.columns),
         "2026_ORIGINAL_OOS_RISK_TRAIN_TARGET_FEATURES_NOT_IDENTICAL")
    te=teacher[[*source.columns,"_target_over3"]].copy()
    rxtr,rxva,rcats=prepare(te,source)
    risk_seed=8500000+2026*100
    risk=lgb.LGBMClassifier(objective="binary",**risk_params(risk_seed))
    risk.fit(rxtr,te["_target_over3"].astype(int).to_numpy(),categorical_feature=rcats)
    pr=np.asarray(risk.predict_proba(rxva)[:,1],dtype=float)
    need(len(pr)==HORSE_COUNT and np.isfinite(pr).all() and
         np.all((pr>=0)&(pr<=1)),"ORIGINAL_2026_CORRECTION_RISK_INVALID")
    risk_path=model_dest/"original-correction-risk-2026-model.txt"
    risk.booster_.save_model(str(risk_path))
    risk_columns=list(rxtr.columns)
    risk_categories={x:list(rxtr[x].cat.categories) for x in rcats}
    scores=context.copy()
    scores["risk"]=pr
    need(len(scores)==HORSE_COUNT,"ORIGINAL_2026_NEW_ABILITY_RISK_471_HORSES_DRIFT")
    print("ORIGINAL_2026_CORRECTION_RISK_PAST_OOS_REAL_SCORE 36R_471H 2022_2025_teachers",flush=True)
    # Frozen original ANCHOR_FULL 2025 coefficients, NOT a newly approved
    # 2026 refit, applied to original 2026 ability+risk with same-snapshot WIN
    # prices already stored in pinned BACKFILL L1.75 research output.
    geom_spec=importlib.util.spec_from_file_location("original_anchor_geometry_2025",
              original_anchor_py.parent/"anchor-original-feature-geometry.py")
    geometry=importlib.util.module_from_spec(geom_spec)
    geom_spec.loader.exec_module(geometry)
    need(tuple(params["features_in_original_order"])==tuple(geometry.FIELDS),
         "ORIGINAL_ANCHOR_19_FEATURE_GEOMETRY_DRIFT")
    full=[]
    anchor_count=0;blocked=[]
    for rid,frame in scores.groupby("_race_id",sort=True):
        prior_panel={}
        win_prices={}
        for _,h in frame.iterrows():
            hid=str(h["_horse_id"]);num=source_numbers[(rid,hid)]
            need(hid in market[rid] and market[rid][hid]["horse_number"]==num,
                 "2026_ANCHOR_MARKET_HORSE_KEY_DRIFT:"+rid)
            prior_panel[num]={"horse_id":hid,"distill_ai_rank":int(h["distill_ai_rank"]),
              "distill_score_z":float(h["distill_score_z"]),
              "distill_score_gap_to_best":float(h["distill_score_gap_to_best"]),
              "risk":float(h["risk"])}
            win_prices[num]=float(market[rid][hid]["market_win_odds"])
        need(len(prior_panel)==len(frame),"ANCHOR_SOURCE_FIELD_MISMATCH")
        try:geometry_result=geometry.build_geometry(prior_panel,win_prices)
        except ValueError as err:
            # Never invent a market top favorite tie-breaker or replace model.
            if str(err)!="ORIGINAL_GEOMETRY_MARKET_FAVORITE_TIE_NO_POPULARITY":
                raise
            blocked.append(rid)
            geometry_result=None
        logits={}
        if geometry_result:
            for number,info in geometry_result.items():
                logit=math.log(info["market_prob_norm"])
                for feat in oldmodule.FEATURES:
                    unit=params["standardizer"][feat]
                    sd=float(unit["std"])
                    need(sd>0 and math.isfinite(sd),"ANCHOR_ORIGINAL_BAD_STANDARDIZER_SD")
                    logit+=(info["features"][feat]-float(unit["mean"]))/sd*float(params["coefficients"][feat])
                logits[number]=logit
            mx=max(logits.values())
            den=math.fsum(math.exp(v-mx) for v in logits.values())
            need(den>0 and math.isfinite(den),"ANCHOR_PROBABILITY_NORMALIZATION_INVALID")
            probs={number:math.exp(z-mx)/den for number,z in logits.items()}
            need(abs(sum(probs.values())-1)<1e-9,"ANCHOR_NORMALIZER_DRIFT")
            axis=max(probs,key=lambda n:(probs[n]*win_prices[n],probs[n],-geometry_result[n]["market_rank"],-n))
            anchor_count+=1
        else:
            probs={};axis=None
        horse_rows=[]
        for _,h in frame.iterrows():
            hid=str(h["_horse_id"]);num=source_numbers[(rid,hid)]
            row={"horse_id":hid,"horse_number":num,
              "distill_ai_rank":int(h["distill_ai_rank"]),
              "distill_score_z":float(h["distill_score_z"]),
              "distill_score_gap_to_best":float(h["distill_score_gap_to_best"]),
              "risk":float(h["risk"]),"original_prior_market_win_odds":win_prices[num],
              "market_rank":market[rid][hid]["market_rank"],
              "P__ANCHOR_FULL":probs.get(num),
              "original_anchor":num==axis if axis is not None else None}
            horse_rows.append(row)
        need(sorted(x["distill_ai_rank"] for x in horse_rows)==list(range(1,len(horse_rows)+1)),
             "ORIGINAL_L1_PURE_RANKS_CHANGED")
        full.append({"race_id":rid,"race_date":RACE_DAY,
          "model_source":"ORIGINAL_TRUE_LAMBDARANK_2024_2025_PLUS_PAST_OOS_RISK_2022_2025",
          "frozen_original_anchor_source":"ORIGINAL_2025_PRIOR_2022_2024_PARAMS_REUSE_RESEARCH",
          "anchor_status":"RESEARCH_SCORED_ASOF_UNVERIFIED" if axis is not None else
                          "BLOCKED_ORIGINAL_TOP_MARKET_FAVORITE_TIE",
          "anchor_horse_number":axis,
          "horses":sorted(horse_rows,key=lambda x:x["distill_ai_rank"]),
          "ready_for_betting":False,"2026_target_results_used":False,
          "asof_independently_verified":False,
          "l21_l3_invoked":False})
    need(len(full)==36 and sum(len(x["horses"]) for x in full)==HORSE_COUNT,
         "2026_ORIGINAL_MODEL_SCORED_SCOPE_DRIFT")
    # No frozen L2.1 invocation: 2026 date is historical and true cutoff
    # provenance has not been independently established.
    scored=model_dest/"original-2026-1017-36r-471h-ability-risk-anchor.jsonl"
    scored.write_text("".join(json.dumps(x,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n" for x in full),
                      encoding="utf8")
    # Persist the verified original 2025 prior-only ANCHOR parameters once,
    # so later inference does NOT rebuild them or redownload old 2025 OOS.
    anchor_copy=model_dest/"original-anchor-full-2025-prior-params.json"
    anchor_copy.write_bytes(anchor_params.read_bytes())
    need(sha(anchor_copy)==sha(anchor_params),"ORIGINAL_ANCHOR_PRIOR_PARAMS_SAVE_DRIFT")
    saved={
      "contract":"KEIBA_2026_ORIGINAL_PURE_AI_RISK_ANCHOR_RESEARCH_CANDIDATE_V1",
      "2026_prediction_date":RACE_DAY,
      "races":len(full),"horses":HORSE_COUNT,
      "original_ability_fit_years":list(FIT_YEARS),
      "original_ability_seed":seed,
      "original_ability_feature_count":len(feature_order),
      "original_ability_columns":feature_order,
      "original_ability_categorical_levels":categories,
      "original_risk_oos_teacher_years":list(TEACHER_YEARS),
      "original_risk_seed":risk_seed,
      "original_risk_feature_count":len(risk_columns),
      "original_risk_feature_order":risk_columns,
      "original_risk_categorical_levels":risk_categories,
      "risk_original_oos_reference_checks":checks,
      "anchor_reuse_original_2025_train_years":[2022,2023,2024],
      "anchor_model_retrained_for_2026":False,
      "anchor_races_scored":anchor_count,
      "anchor_races_blocked_favorite_tie":blocked,
      "2026_target_outcomes_used":False,
      "target_input_base649_names_sha256":ORIGINAL_BASE,
      "source_asof_independently_verified":False,
      "original_model_release_approved":False,
      "frozen_l21_l3_invoked":False,
      "ready_for_betting":False,
      "files":{p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in (ability_path,risk_path,anchor_copy,scored)}
    }
    (model_dest/"index.json").write_text(json.dumps(saved,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("ORIGINAL_2026_REAL_LAMBDARANK_RISK_AND_ANCHOR_ONCE "+
          json.dumps({"races":36,"horses":471,"ability_training_years":list(FIT_YEARS),
          "past_oos_teacher_years":list(TEACHER_YEARS),
          "anchor_scored_races":anchor_count,"anchor_blocked_tie_races":blocked,
          "model_weights":2,"model_release":False,"2026_results_used":False},sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser()
    for name in ("saved-priors","backfill","l175-market-index","l175-market-file",
                 "anchor-2025-params","original-anchor-code","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    run(a.saved_priors,a.backfill,a.l175_market_index,a.l175_market_file,
        a.anchor_2025_params,a.original_anchor_code,a.output)
if __name__=="__main__":main()
