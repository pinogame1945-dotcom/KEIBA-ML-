#!/usr/bin/env python3
"""2026-10-10 historical L1 original-model -> frozen L2.1 -> V3 research replay.

All data inputs are separately pinned: 24x315 saved BACKFILL original 690,
2026 L1 ability/risk LightGBM models fitted on 2024-25/past-OOS 2022-25,
original 2025 ANCHOR_FULL prior coefficients, frozen original 50 rule-gates,
official L3 V3 reverse_p3_stakes, and post-race NETKEIBA_FINAL quotes.

Historic FINAL odds are NOT a bettable prediction snapshot. No 2026 target
outcome is used to fit/rank/choose ticket/stake. Separately compare frozen
tickets to historical official outcomes only AFTER all stake decisions.
10/10 original 7King/Outsider L1.75 per-horse P3 signals unavailable:
genuine V3 missing-P3 ordering, NO proxy/made-up P3.
No frozen policy rewrite, dataset upload, betting, paid resources, or artifacts.
"""
from __future__ import annotations
import argparse,ast,gzip,hashlib,importlib.util,json,math,sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

DAY="2026-10-10"
BF="KEIBA_BACKFILL_TO_ML_690_INDEX_V1"
ODDS="KEIBA_BACKFILL_TO_ML_ODDS_INDEX_V1"
PURE="KEIBA_2026_ORIGINAL_PURE_AI_RISK_ANCHOR_RESEARCH_CANDIDATE_V1"
ORIGINAL_P3_NOTE="NOT_AVAILABLE_FOR_2026_1010_USE_FROZEN_MISSING_P3_POLICY"
FROZEN_L3_BLOB="71ea06e8a4d1a9694fc3a113c3c376a6b0bb4289"
SENTINEL_BLOCK="HISTORICAL_POSTRACE_FINAL_ODDS_DO_NOT_BET"

def need(v,msg):
    if not v:raise ValueError(msg)

def data(p):return json.loads(Path(p).read_text(encoding="utf8"))
def checksum(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()

def gitblob(p):
    b=Path(p).read_bytes()
    return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\x00"+b).hexdigest()

def original_predictors(original_index,model_dir,geometry_py):
    idx=data(original_index)
    need(idx.get("contract")==PURE and idx.get("2026_target_outcomes_used") is False
         and idx.get("original_model_release_approved") is False
         and idx.get("original_ability_fit_years")==[2024,2025]
         and idx.get("original_risk_oos_teacher_years")==[2022,2023,2024,2025]
         and idx.get("anchor_reuse_original_2025_train_years")==[2022,2023,2024],
         "SOURCE_ORIGINAL_TRAIN_YEAR_MODEL_SPEC_CHANGED")
    def named(name):
        p=model_dir/name
        record=idx["files"].get(name)
        need(p.is_file() and record is not None and checksum(p)==record["sha256"]
             and p.stat().st_size==record["bytes"],"SAVED_MODEL_BYTE_SHA_DRIFT:"+name)
        return p
    a=named("original-core4-2026-lambdarank-model.txt")
    r=named("original-correction-risk-2026-model.txt")
    anchor=named("original-anchor-full-2025-prior-params.json")
    ability=lgb.Booster(model_file=str(a))
    risk=lgb.Booster(model_file=str(r))
    a_cols=idx["original_ability_columns"]
    r_cols=idx["original_risk_feature_order"]
    need(len(a_cols)==690 and len(a_cols)==idx["original_ability_feature_count"]
         and a_cols==ability.feature_name()
         and len(r_cols)==idx["original_risk_feature_count"]
         and r_cols==risk.feature_name(),
         "AUTHENTIC_LGBM_FEATURE_ORDER_OR_COUNT_DRIFT")
    spec=importlib.util.spec_from_file_location("frozen_2025_original_anchor_geometry",geometry_py)
    geom=importlib.util.module_from_spec(spec);spec.loader.exec_module(geom)
    params=data(anchor)
    need(params["contract"]=="KEIBA_ORIGINAL_ANCHOR_FULL_PRIOR_TRAINED_PARAMS_V1"
         and params["trained_only_on_years"]==[2022,2023,2024]
         and params["variant"]=="ANCHOR_FULL"
         and params["features_in_original_order"]==list(geom.FIELDS)
         and set(params["coefficients"])==set(params["standardizer"])==set(geom.FIELDS),
         "FROZEN_ORIGINAL_2025_ANCHOR_MODEL_DRIFT")
    return idx,ability,risk,params,geom,a_cols,r_cols

def frozen_math(files):
    sys.path.insert(0,str(files["replay"].parent))
    import replay_2026_1017_original_frozen_l21_ticket_candidates_v1 as prior
    original=prior.load_frozen_math({
        "fixed":files["fixed"],"gates":files["gates"],
        "lab":files["lab"],"pair":files["pair"],"geometry":files["geom"]})
    need(gitblob(files["v3"])==FROZEN_L3_BLOB,"ORIGINAL_ADOPTED_L3_V3_BLOB_DRIFT")
    tree=ast.parse(files["v3"].read_text("utf8"),filename=str(files["v3"]))
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef)
           and x.name in ("yen","ticket_key","reverse_p3_stakes")]
    need(len(nodes)==3,"ADOPTED_L3_V3_REVERSE_P3_ENGINE_NOT_AVAILABLE")
    scope={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(files["v3"]),"exec"),scope)
    return original[0],original[1],scope["reverse_p3_stakes"]

def category_x(frame,cols,cats):
    need(len(cols)==len(set(cols)) and set(cols).issubset(frame.columns),
         "ORIGINAL_SAVED_LGBM_FEATURE_SCHEMA_UNAVAILABLE")
    result=frame.reindex(columns=cols).copy()
    for col in cols:
        if col in cats:
            value=result[col].astype("string").fillna("__MISSING__")
            result[col]=pd.Categorical(value,categories=cats[col])
        else:
            result[col]=pd.to_numeric(result[col],errors="coerce").astype("float32")
    return result

def exact_original_context(frame):
    # Original frozen run_l1_market_correction_distill_v1.py:add_ai_context
    x=frame.sort_values(["_race_id","_score","_horse_id"],
        ascending=[True,False,True]).reset_index(drop=True).copy()
    x["distill_ai_rank"]=x.groupby("_race_id",sort=False).cumcount()+1
    n=x.groupby("_race_id")["_horse_id"].transform("size").astype(float)
    x["distill_field_size"]=n
    x["distill_ai_rank01"]=1.0-(x["distill_ai_rank"]-1.0)/(n-1.0).clip(lower=1.0)
    m=x.groupby("_race_id")["_score"].transform("mean")
    std=x.groupby("_race_id")["_score"].transform("std").replace(0,np.nan)
    x["distill_score_z"]=((x["_score"]-m)/std).fillna(0.0)
    best=x.groupby("_race_id")["_score"].transform("max")
    x["distill_score_gap_to_best"]=(best-x["_score"]).clip(lower=0.0)
    x["distill_is_ai_top1"]=(x["distill_ai_rank"]==1).astype(float)
    x["distill_is_ai_top3"]=(x["distill_ai_rank"]<=3).astype(float)
    x["distill_is_ai_top6"]=(x["distill_ai_rank"]<=6).astype(float)
    return x

def source_inputs(bf_root):
    path=bf_root/"ml_features/delivery"
    ix=data(path/"690/index.json")
    odd=data(path/"odds/index.json")
    need(ix["contract"]==BF and odd["contract"]==ODDS and
         ix["feature_count"]==690 and
         ix.get("original_ml_full_numerical_parity_verified") is False and
         ix.get("readiness")=="ASOF_UNVERIFIED_DO_NOT_BET",
         "FROZEN_690_OR_ODDS_INDEX_CONTRACT_CHANGED")
    r={k:v for k,v in ix["races"].items() if v["date"]==DAY}
    q={k:v for k,v in odd["races"].items() if v["date"]==DAY}
    need(len(r)==len(q)==24 and set(r)==set(q) and
         sum(v["starters"] for v in r.values())==315,
         "ORIGINAL_2026_1010_NOT_24R315H")
    rows=[];sources={};rid_to_hid={}
    for rid,meta in sorted(r.items()):
        rp=bf_root/meta["path"]; op=bf_root/q[rid]["path"]
        need(rp.is_file() and op.is_file() and
             checksum(rp)==meta["sha256"] and checksum(op)==q[rid]["sha256"],
             "SAVED_2026_1010_SHA_FAIL:"+rid)
        packet=data(rp);market=data(op)
        need(packet["contract"]=="KEIBA_BACKFILL_TO_ML_690_UNSCORED_V1"
             and packet["race_id"]==rid and packet["race_date"]==DAY
             and packet["feature_count"]==690 and packet["starters"]==meta["starters"]
             and packet["used_target_results_or_final_odds"] is False
             and len(packet["rows"])==meta["starters"]
             and market["race_id"]==rid and market["race_date"]==DAY
             and market["contract"]=="KEIBA_BACKFILL_TO_ML_ODDS_V1"
             and market["odds"]["source"]=="NETKEIBA_FINAL"
             and market["odds"]["snapshot_type"]=="POST_RACE_FINAL"
             and market["odds"]["readiness"]=="FINAL_ODDS_NOT_PRE_RACE_ASOF"
             and q[rid]["source"]=="NETKEIBA_FINAL" and
             q[rid]["snapshot_type"]=="POST_RACE_FINAL"
             and set(("WIN","QUINELLA","TRIO","TRIFECTA")).issubset(
                 set(market["odds"]["present_bet_types"]))
             and market["feature_sha256"]==meta["sha256"],
             "NO_VALID_HISTORICAL_FINAL_REPLAY_INPUT:"+rid)
        quotes=market["odds"]["raw_group_odds"]
        need(all(z in quotes for z in ("1","4","7","8")),
             "MISSING_HISTORICAL_2026_1010_MARKET:"+rid)
        by_no={}
        for num,arr in quotes["1"].items():
            try:price=float(str(arr[0]).replace(",",""))
            except (ValueError,TypeError):continue
            if math.isfinite(price) and price>0:
                by_no[int(num)]=price
        rs={};unique=set()
        for row in packet["rows"]:
            hid=str(row["_horse_id"]);num=row["horse_number"]
            need(hid and (rid,hid) not in unique and type(num)is int
                 and 1<=num<=18 and num not in rs and
                 row["_race_id"]==rid and row["_race_date"]==DAY,
                 "DUPLICATE_HORSE_IN_REAL_1010_RACE:"+rid)
            unique.add((rid,hid))
            rs[num]=hid
            rows.append(row)
        need(len(rs)==len(packet["rows"]) and set(rs).issubset(by_no),
             "MISSING_WIN_ODDS_FOR_REAL_HORSE:"+rid)
        sources[rid]={"raw_group_odds":quotes,"win":by_no,
                      "horse_num_to_id":rs,"horse_id_to_num":{v:k for k,v in rs.items()},
                      "horse_count":len(rs)}
    df=pd.DataFrame(rows)
    need(len(df)==315 and
         len(df.columns)==693 and
         df[["_race_id","_horse_id"]].duplicated().sum()==0,
         "2026_1010_690_SAVED_SOURCE_NOT_FULL")
    return df,sources

def real_original_model_score(frame,sources,idx,ability,risk,params,geometry,a_cols,r_cols):
    frame=frame.sort_values(["_race_id","_horse_id"]).reset_index(drop=True)
    xva=category_x(frame,a_cols,idx["original_ability_categorical_levels"])
    z=np.asarray(ability.predict(xva),dtype=float)
    need(len(z)==315 and np.isfinite(z).all(),
         "ORIGINAL_L1_SAVED_RANK_MODEL_SCORE_FAILED")
    context=exact_original_context(pd.DataFrame({
        "_race_id":frame["_race_id"].astype(str),
        "_horse_id":frame["_horse_id"].astype(str),"_score":z}))
    required=["_race_id","_horse_id","_score","distill_ai_rank",
       "distill_field_size","distill_ai_rank01","distill_score_z",
       "distill_score_gap_to_best","distill_is_ai_top1","distill_is_ai_top3",
       "distill_is_ai_top6"]
    src=frame.merge(context[required],on=["_race_id","_horse_id"],
                    how="left",validate="one_to_one")
    src=src[[c for c in src.columns if not c.startswith("neg_")]].copy()
    # Reproduce the frozen 10/17 scorer's EXACT prepare(train,valid):
    # the historical risk teacher retains "_target_over3" as a training
    # feature, while original 2026 inference reindexes it to missing/NaN.
    # Absolutely never fill from 2026 true outcomes.
    missing=set(r_cols)-set(src.columns)
    need(missing=={"_target_over3"},
         "ORIGINAL_SAVED_RISK_REQUIRED_FEATURE_DRIFT:"+str(sorted(missing)))
    src["_target_over3"]=float("nan")
    rx=category_x(src,r_cols,idx["original_risk_categorical_levels"])
    probs=np.asarray(risk.predict(rx),dtype=float)
    need(len(probs)==315 and np.isfinite(probs).all() and
         np.all((probs>=0)&(probs<=1)),
         "ORIGINAL_RISK_SAVED_MODEL_SCORE_FAILED")
    scores=context.copy()
    scores["risk"]=probs
    grouped={}
    blocked=[]
    for rid,g in scores.groupby("_race_id",sort=True):
        quotes=sources[rid];panel={};w={}
        for _,r in g.iterrows():
            hid=str(r["_horse_id"]);num=quotes["horse_id_to_num"][hid]
            panel[num]={"horse_id":hid,"distill_ai_rank":int(r["distill_ai_rank"]),
                        "distill_score_z":float(r["distill_score_z"]),
                        "distill_score_gap_to_best":float(r["distill_score_gap_to_best"]),
                        "risk":float(r["risk"])}
            w[num]=quotes["win"][num]
        try:geom=geometry.build_geometry(panel,w)
        except ValueError as exc:
            if str(exc)=="ORIGINAL_GEOMETRY_MARKET_FAVORITE_TIE_NO_POPULARITY":
                blocked.append(rid)
                grouped[rid]=None
                continue
            raise
        logits={}
        for num,v in geom.items():
            a=math.log(v["market_prob_norm"])
            for feat in params["features_in_original_order"]:
                unit=params["standardizer"][feat]
                sd=float(unit["std"])
                need(sd>0,"UNFROZEN_ANCHOR_PRIOR_STD")
                a+=(v["features"][feat]-float(unit["mean"]))/sd*float(params["coefficients"][feat])
            logits[num]=a
        m=max(logits.values());s=math.fsum(math.exp(x-m) for x in logits.values())
        p={num:math.exp(v-m)/s for num,v in logits.items()}
        need(abs(math.fsum(p.values())-1)<1e-9,
             "ORIGINAL_ANCHOR_PROB_SUM_DRIFT")
        axis=max(p,key=lambda n:(p[n]*w[n],p[n],-geom[n]["market_rank"],-n))
        horse_panel={};gf=[]
        for _,r in g.iterrows():
            hid=str(r["_horse_id"]);num=quotes["horse_id_to_num"][hid]
            gf.append({"_horse_id":hid,"P__ANCHOR_FULL":p[num],
                "market_rank":geom[num]["market_rank"],
                "final_win_odds":w[num],
                "distill_ai_rank":int(r["distill_ai_rank"])})
            horse_panel[num]={**panel[num],
                "P__ANCHOR_FULL":p[num],
                "market_rank":geom[num]["market_rank"],
                "market_prob_norm":geom[num]["market_prob_norm"]}
        rr=pd.DataFrame(gf)
        rank=rr.sort_values(["P__ANCHOR_FULL","market_rank","_horse_id"],
          ascending=[False,True,True],kind="mergesort")
        p_rank={quotes["horse_id_to_num"][str(hid)]:i for i,hid in
                enumerate(rank["_horse_id"],1)}
        for num,v in horse_panel.items():v["P_rank"]=p_rank[num]
        grouped[rid]={"group":rr,"panel":horse_panel,"axis":axis,
             "ranked_top":int(g.sort_values(["distill_ai_rank"]).iloc[0]["distill_ai_rank"]),
             "ai_raw_top_hid":rr.sort_values(["distill_ai_rank","market_rank","_horse_id"],
                         ascending=[True,True,True],kind="mergesort").iloc[0]["_horse_id"]}
    print("ORIGINAL_2026_1010_L1_SAVED_WEIGHTS_INFERENCE "+
          json.dumps({"races":24,"horses":315,"blocked_favorite_ties":blocked,
                      "after_race_market":True,"fit_or_retrain":False},sort_keys=True),flush=True)
    return grouped,blocked

def original_stakes(file):
    need(gitblob(file)==FROZEN_L3_BLOB,"FROZEN_L3_V3_BLOB_NOT_REAL")
    tree=ast.parse(file.read_text("utf8"))
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and
           x.name in ("yen","ticket_key","reverse_p3_stakes")]
    need(len(nodes)==3,"FROZEN_L3_REVERSE_P3_NOT_FOUND")
    n={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),"exec"),n)
    return n["reverse_p3_stakes"]

def official_hist_quotes(d):
    # original frozen parse_price accepts three-slot NETKEIBA_FINAL raw arrays.
    return {"odds":d["raw_group_odds"]}

def select_and_stake(sources,scored,l21_math,l3_alloc):
    create_cards,flags,_=l21_math
    output=[];n_tickets=0;counts=Counter()
    for rid in sorted(sources):
        item=scored.get(rid)
        if item is None:
            output.append({"race_id":rid,"status":"BLOCKED_ORIGINAL_ANCHOR_FAVORITE_TIE",
                           "tickets":[],"stake_total_yen":0});continue
        axis=item["axis"]
        companion=sources[rid]
        grp=item["group"];hp=item["panel"]
        nmap=companion["horse_num_to_id"]
        pack={"entries":[{"horse_id":hid,"horse_number":no}
                         for no,hid in nmap.items()],
              "results":[],"payouts":[]}
        info={"AI_RAW_TOP_horse":str(item["ai_raw_top_hid"]),
              "VALUE_EV_horse":str(hp[axis]["horse_id"])}
        lines=(
          ("AI_PROB|2|TRIO|15-20",("ABILITY_Z_GE_0","ABILITY_Z_GE_M1","AGREEMENT_TOP6","ALL","CALIBRATED_TOP10","CALIBRATED_TOP6","MARKET_TOP10","MARKET_TOP6","MODEL_WIN_P_GE_003","RAW_TOP10","RAW_TOP6","RISK_LT040","RISK_LT060")),
        )
        # Never use local/prior rules as source of truth: passed frozen contract
        # 50 STANDARD line-gates is the only selection contract.
        accepted={}
        accepted_groups={}
        for tid,ticket,card in create_cards(
                grp,pack,official_hist_quotes(companion),rid,
                hp[axis]["horse_id"],info):
            line="|".join(map(str,tid))
            need(line not in ("","None"),"INVALID_ORIGINAL_L21_TEMPLATE_LINE")
            requirements=FROZEN_RULES.get(line)
            if not requirements:continue
            gates=flags(ticket,hp,axis)
            passed=sorted(requirements.intersection(k for k,v in gates.items() if v))
            if not passed:continue
            kind,seq=ticket
            need(kind in ("QUINELLA","TRIO","TRIFECTA") and
                 axis in seq and card[1]==0,
                 "RESULT_OR_UNFROZEN_SELECTION_CONTAMINATION")
            key=(kind,tuple(seq))
            record=accepted.get(key)
            if record is None:
                record={"bet_type":kind,"selection":list(seq),"decimal_odds":card[0],
                        "matched_line_gates":[],"strict_diagnostic":False}
                accepted[key]=record
            else:need(record["decimal_odds"]==card[0],"DUPLICATE_TICKET_ODDS_DRIFT")
            record["matched_line_gates"].extend(
                {"line":line,"gate":gate} for gate in passed)
            record["strict_diagnostic"] |= any(
                (line,gate) in FROZEN_STRICT for gate in passed)
        tix=sorted(accepted.values(),key=lambda x:(x["bet_type"],x["selection"]))
        n=len(tix)
        cap=300
        if n>3:
            output.append({"race_id":rid,"status":"INSUFFICIENT_RACE_BUDGET",
                "original_axis":axis,"L21_tickets":n,
                "required_minimum_yen":100*n,"max_race_budget_yen":300,
                "stake_total_yen":0,
                "candidate_tickets":tix,"tickets":[]})
            counts["budget_blocked_races"]+=1;continue
        alloc_input=[({"bet_type":t["bet_type"],
                       "canonical_selection":"-".join(map(str,t["selection"]))},None)
                     for t in tix]
        budget=min(cap,200*n)
        stakes=l3_alloc(alloc_input,budget) if n else {}
        for t in tix:
            k=(t["bet_type"],"-".join(map(str,t["selection"])))
            t["stake_yen"]=stakes[k]
            t["min_original_p3"]=None
            t["p3_missing"]=True
            t["research_final_odds_only"]=True
            t["hit"]=None;t["payout_yen"]=None
            counts[t["bet_type"]]+=1
        need(sum(t["stake_yen"] for t in tix)==budget and
             all(t["stake_yen"] in (100,200,300) for t in tix),
             "ORIGINAL_V3_STAKE_CONSERVATION_FAILED")
        output.append({"race_id":rid,"status":"HISTORICAL_POSTRACE_PAPER_L3_ONLY",
            "original_axis":axis,"L21_tickets":n,"stake_total_yen":budget,
            "p3_data":ORIGINAL_P3_NOTE,"tickets":tix})
        n_tickets+=n
        print("L3_1010_PAPER_RACE "+json.dumps({"race_id":rid,"n":n,
             "total_yen":budget,"tickets":[{"kind":x["bet_type"],"nums":x["selection"],
             "odds":x["decimal_odds"],"yen":x["stake_yen"]} for x in tix]},
             ensure_ascii=False,sort_keys=True),flush=True)
    return output,n_tickets,counts

def official_results_for_display(hist_source):
    # This file is opened ONLY AFTER full original L1->frozen L2.1->V3 lock.
    # Actual payout rows are the source of truth when dead heats are present.
    import re
    target=hist_source/"data/daily/2026-10-10.jsonl.gz"
    need(target.is_file(),"MISSING_SEPARATE_HISTORIC_RESULT_PACK")
    rows={}
    arity={"QUINELLA":2,"TRIO":3,"TRIFECTA":3}
    with gzip.open(target,"rt",encoding="utf8") as f:
        for line in f:
            if not line.strip():continue
            x=json.loads(line);race=x.get("race",{})
            rid=str(race.get("race_id") or "")
            need(rid not in rows and str(race.get("actual_date") or
                    race.get("scheduled_date") or "")[:10]==DAY,
                 "RACE_RESULTS_WRONG_DAY")
            starters={str(e["horse_id"]):int(e["horse_number"])
                      for e in x.get("entries",[])
                      if e.get("horse_id") and e.get("horse_number")}
            placements={}
            for r in x.get("results",[]):
                hid=str(r.get("horse_id") or "")
                value=r.get("official_finish_position")
                if hid not in starters or value is None:continue
                try:rank=int(value)
                except (ValueError,TypeError):continue
                if rank>=1:placements.setdefault(rank,[]).append(starters[hid])
            need(len(placements)>=3 or
                 (len(placements)>=2 and any(len(v)>1 for v in placements.values())),
                 "OFFICIAL_FINISH_RANKS_INCOMPLETE:"+rid)
            placement_view={str(p):sorted(v) for p,v in sorted(placements.items())
                            if p<=3}
            # In a dead heat, ranks may skip a value (e.g. 2,2,4).
            # NEVER break ties or fabricate unique top3.
            paid={};paid_types=set()
            for p in x.get("payouts") or []:
                bet=str(p.get("bet_type") or "")
                if bet not in arity:continue
                paid_types.add(bet)
                ns=tuple(int(s) for s in re.findall(r"\\d+",str(p.get("combination") or "")))
                if len(ns)!=arity[bet] or len(set(ns))!=len(ns):continue
                if bet!="TRIFECTA":ns=tuple(sorted(ns))
                try:yen=float(str(p.get("payout_yen") or "").replace(",",""))
                except (ValueError,TypeError):continue
                if not math.isfinite(yen) or yen<=0:continue
                need((bet,ns) not in paid,"DUPLICATE_ACTUAL_OFFICIAL_PAID_COMBINATION:"+rid)
                paid[(bet,ns)]=round(yen)
            rows[rid]={"actual_placements":placement_view,
                       "payouts_per100":paid,
                       "available_payout_categories":paid_types,
                       "dead_heat":any(len(z)>1 for z in placements.values())}
    need(len(rows)==24,"NOT_ALL_24_RESULT_PACKS_AVAILABLE")
    return rows

def settle_only_after_lock(records,hist_source):
    # Original L2.1/V3 choices are already fixed; no results feed back into
    # score, anchor, gates, prices, P3 ordering or stake selection.
    result=official_results_for_display(hist_source)
    total=Counter();races_with_tickets=0;dead_heats=0
    for race in records:
        rid=race["race_id"];truth=result[rid]
        race["actual_1_2_3"]=truth["actual_placements"]
        race["race_has_dead_heat"]=truth["dead_heat"]
        dead_heats+=truth["dead_heat"]
        race_paid=0
        for t in race["tickets"]:
            typ=t["bet_type"]
            need(typ in truth["available_payout_categories"],
                 "OFFICIAL_REAL_PAYOUT_FAMILY_NOT_AVAILABLE:"+rid+":"+typ)
            ns=tuple(t["selection"])
            if typ!="TRIFECTA":ns=tuple(sorted(ns))
            paid=int(truth["payouts_per100"].get((typ,ns),0))
            t["hit"]=paid>0
            t["payout_yen"]=paid*t["stake_yen"]//100
            t["payout_per100_yen"]=paid
            if t["hit"]:total["hits"]+=1
            total["stake_yen"]+=t["stake_yen"]
            total["payout_yen"]+=t["payout_yen"]
            total["tickets"]+=1
            race_paid+=t["payout_yen"]
        race["matched_tickets"]=sum(x["hit"] is True for x in race["tickets"])
        race["payout_yen"]=race_paid
        race["profit_yen"]=race_paid-race["stake_total_yen"]
        if race["tickets"]:
            races_with_tickets+=1
            print("L3_1010_RESULT_OBSERVATION "+json.dumps({"race_id":rid,
                  "placements":truth["actual_placements"],
                  "dead_heat":truth["dead_heat"],
                  "hits":race["matched_tickets"],
                  "stake_yen":race["stake_total_yen"],
                  "payout_yen":race_paid,
                  "tickets":[{"kind":t["bet_type"],"nums":t["selection"],
                  "yen":t["stake_yen"],"hit":t["hit"],
                  "payout_yen":t["payout_yen"]} for t in race["tickets"]]},
                  ensure_ascii=False,sort_keys=True),flush=True)
    need(total["tickets"]==sum(len(r["tickets"]) for r in records),
         "RESULT_OBSERVATION_CHANGED_L21_L3_TICKET_COUNT")
    print("RESULT_SETTLEMENT_OFFICIAL_PAID_TICKET_COMBINATIONS "+
          json.dumps({"tickets":total["tickets"],"hits":total["hits"],
           "stake_yen":total["stake_yen"],
           "payout_yen":total["payout_yen"],
           "dead_heat_races":dead_heats},sort_keys=True),flush=True)
    return total,races_with_tickets

def main():
    global FROZEN_RULES,FROZEN_STRICT
    ap=argparse.ArgumentParser()
    for n in ("backfill-root","model-dir","model-index","original-geometry",
              "original-create-cards","original-gates","original-odds-lab",
              "original-templates","original-replay-helper",
              "frozen-l21-contract","original-v3-source",
              "historical-results-root","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args()
    need(not a.output.exists(),"NO_OVERWRITE_EXISTING_HISTORIC_RESULT")
    c=data(a.frozen_l21_contract)
    need(c["contract"]=="KEIBA_L2_1_FIXED_PURCHASE_V1" and
         c["version"]=="2.1.0" and c["primary_policy"]=="STANDARD"
         and c["rule_counts"]["STANDARD"]==50 and
         c["rule_counts"]["STRICT_SUBSET"]==26,
         "FROZEN_L21_POLICY_NOT_50_26")
    rules=c["frozen_rules"]["STANDARD"]
    FROZEN_RULES={}
    for x in rules:FROZEN_RULES.setdefault(x["line"],set()).add(x["gate"])
    FROZEN_STRICT={(x["line"],x["gate"]) for x in
                   c["frozen_rules"]["STRICT_SUBSET"]}
    need(len(FROZEN_RULES)==9 and len(FROZEN_STRICT)==26,
         "FROZEN_L21_RULES_DRIFT")
    frame,sources=source_inputs(a.backfill_root)
    idx,ability,risk,params,geometry,acols,rcols=original_predictors(
        a.model_index,a.model_dir,a.original_geometry)
    scored,blocked=real_original_model_score(frame,sources,idx,ability,
         risk,params,geometry,acols,rcols)
    create,gate,geometry2=load_math_unused if False else frozen_math({
        "fixed":a.original_create_cards,"gates":a.original_gates,
        "lab":a.original_odds_lab,"pair":a.original_templates,
        "geom":a.original_geometry,"replay":a.original_replay_helper,
        "v3":a.original_v3_source})
    v3=original_stakes(a.original_v3_source)
    records,n,counts=select_and_stake(sources,scored,(create,gate,geometry2),v3)
    need(len(records)==24 and len(blocked)<=24,"1010_L21_24_RACES_FAILED")
    # IMPORTANT: all 24 race tickets and V3 stakes fixed before touching any results.
    settled,with_tickets=settle_only_after_lock(records,a.historical_results_root)
    sum_yen=sum(x["stake_total_yen"] for x in records)
    need(sum_yen==settled["stake_yen"] and
         n==settled["tickets"] and sum(r["matched_tickets"] for r in records)
         ==settled["hits"],"POST_HOC_SETTLEMENT_AFFECTED_L3_MONEY")
    report={"contract":"KEIBA_2026_1010_24R_ORIGINAL_L21_L3_V3_FINAL_ODDS_REFERENCE_ONLY_V1",
        "date":DAY,"races":24,"horses":315,"tickets":n,
        "matched_tickets":settled["hits"],
        "fixed_l21_rules":50,"l21_original_model_weights_reused":True,
        "l3_original_reverse_p3_engine_used":True,
        "l3_max_race_budget_yen":300,
        "total_paper_stake_yen":sum_yen,
        "total_official_observed_payout_yen":settled["payout_yen"],
        "total_paper_profit_yen":settled["payout_yen"]-sum_yen,
        "tickets_by_type":dict(counts),"blocked_original_favorite_ties":blocked,
        "races_with_tickets":with_tickets,
        "market_source":"NETKEIBA_FINAL",
        "market_snapshot_type":"POST_RACE_FINAL",
        "genuine_1010_L175_p3_available":False,
        "missing_p3_alternative":"OFFICIAL_V3_MISSING_P3_SORT_NO_PROXY",
        "result_observation_performed_after_ticket_lock":True,
        "results_not_used_for_training_or_reselection":True,
        "payout_yen_verified":True,
        "real_prebet_provenance_verified":False,
        "actual_betting_not_performed":True,"ready_for_betting":False,
        "per_race":records}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print("ORIGINAL_2026_1010_L3_V3_OBSERVATION_SUMMARY "+
      json.dumps({"races":24,"horses":315,"tickets":n,"hits":settled["hits"],
       "stake_yen":sum_yen,"payout_yen":settled["payout_yen"],
       "profit_yen":settled["payout_yen"]-sum_yen,
       "races_with_tickets":with_tickets,"model_refit":False,
       "quote":"POST_RACE_FINAL_NOT_PREBET","p3_missing":True,
       "actual_bets":False},ensure_ascii=False,sort_keys=True),flush=True)

if __name__=="__main__":main()
