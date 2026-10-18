#!/usr/bin/env python3
"""Original frozen L2.1 50-rule input-feature availability audit (NOT betting).

Read saved, SHA-verified genuine 2026 candidate original ability/Risk/ANCHOR and
frozen L1.75 same-quote WIN market. Reuse original 2025 ANCHOR geometry and the
original frozen L2 gate_flags code without reimplementing either calculation.
NEVER compute actual purchase tickets, payouts, ROI, or future 2026 labels.
"""
from __future__ import annotations
import argparse, ast, collections, hashlib, importlib.util, json, math
from pathlib import Path

DAY="2026-10-17"
SOURCE="KEIBA_2026_ORIGINAL_PURE_AI_RISK_ANCHOR_RESEARCH_CANDIDATE_V1"
MARKET="KEIBA_L175_BACKFILL_2026_1017_RESEARCH_PRIVATE_V1"
MARKET_RACE="KEIBA_2026_1017_L175_MARKET_JOIN_RESEARCH_V1"
FREEZE="KEIBA_L2_1_FIXED_PURCHASE_V1"
NEEDED=("P__ANCHOR_FULL","market_prob_norm","risk","distill_score_z",
        "distill_ai_rank","market_rank","P_rank")
REQS=("AI_PROB","AI_RAW","MARKET")
SCOPE=("QUINELLA","TRIO","TRIFECTA")

def demand(ok,reason):
    if not ok:raise ValueError(reason)
def load(p):return json.loads(p.read_text(encoding="utf8"))
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as fd:
        for chunk in iter(lambda:fd.read(1048576),b""):h.update(chunk)
    return h.hexdigest()
def lines(p):
    with p.open("r",encoding="utf8") as fd:
        for line in fd:
            if line.strip():yield json.loads(line)

def original_functions(code,geom):
    # Execute the pinned, unmodified original gate_flags function only, never
    # import the historical training/ticket/payout engine or its side effects.
    tree=ast.parse(code.read_text(encoding="utf8"),filename=str(code))
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="gate_flags"]
    demand(len(nodes)==1 and len(nodes[0].args.args)==3,
           "ORIGINAL_FROZEN_L21_GATE_FLAGS_FUNCTION_ABSENT")
    namespace={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(code),"exec"),namespace)
    path=importlib.util.spec_from_file_location("pinned_original_anchor_geometry",geom)
    module=importlib.util.module_from_spec(path)
    path.loader.exec_module(module)
    demand(hasattr(module,"build_geometry"),"ORIGINAL_2025_MARKET_GEOMETRY_NOT_FOUND")
    return namespace["gate_flags"],module.build_geometry

def run(pure_index,pure_scores,market_index,market_scores,freeze_file,
        original_gate_code,original_geometry_code,report):
    demand(not report.exists(),"REFUSE_SOURCE_AUDIT_OVERWRITE")
    p=load(pure_index);m=load(market_index);f=load(freeze_file)
    demand(p.get("contract")==SOURCE and p.get("races")==36 and
           p.get("horses")==471 and p.get("anchor_races_scored")==36 and
           p.get("anchor_races_blocked_favorite_tie")==[] and
           p.get("original_model_release_approved") is False and
           p.get("source_asof_independently_verified") is False and
           p.get("ready_for_betting") is False and
           p.get("2026_target_outcomes_used") is False and
           p.get("files",{}).get(pure_scores.name,{}).get("sha256")==digest(pure_scores),
           "ORIGINAL_PURE_RESEARCH_SOURCE_SHA_OR_RELEASE_DRIFT")
    demand(m.get("contract")==MARKET and m.get("races")==36 and
           m.get("horses")==471 and m.get("market_asof_independently_verified") is False and
           m.get("2026_target_results_used") is False and
           m.get("ready_for_betting") is False and
           m.get("files",{}).get(market_scores.name)==digest(market_scores),
           "FROZEN_L175_SAVED_JRA_WIN_PROVENANCE_INVALID")
    demand(f.get("contract")==FREEZE and f.get("version")=="2.1.0" and
           f.get("primary_policy")=="STANDARD" and
           f.get("rule_counts",{}).get("STANDARD")==50 and
           f.get("rule_counts",{}).get("STRICT_SUBSET")==26 and
           len(f.get("frozen_rules",{}).get("STANDARD",[]))==50,
           "L21_FROZEN_CONTRACT_NOT_EXACT_50_RULES")
    gates={rec["gate"] for rec in f["frozen_rules"]["STANDARD"]}
    demand(set(rec["line"].split("|")[0] for rec in f["frozen_rules"]["STANDARD"])==set(REQS) and
           set(rec["line"].split("|")[2] for rec in f["frozen_rules"]["STANDARD"]).issubset(
               {"QUINELLA","TRIO","TRIFECTA_FIRST","TRIFECTA_THIRD"}) and
           len(gates)==14,"L21_FROZEN_GATE_OR_LINE_SCHEMA_CHANGED")
    flag_fn,geometry_fn=original_functions(original_gate_code,original_geometry_code)
    market={}
    for r in lines(market_scores):
        rid=str(r.get("race_id") or "")
        hs=r.get("horses") or []
        demand(rid not in market and r.get("contract")==MARKET_RACE and
               r.get("race_date",DAY)==DAY and
               r.get("horse_count")==len(hs) and
               r.get("market_asof_independently_verified") is False and
               r.get("ready_for_betting") is False and
               r.get("2026_target_results_used") is False and
               r.get("win_market_source")=="JRA_OFFICIAL_POST",
               "SAVED_JRA_WIN_MARKET_RACE_INVALID:"+rid)
        market[rid]={str(h["horse_id"]):h for h in hs}
        demand(len(market[rid])==len(hs),"SAVED_MARKET_HORSE_IDS_NOT_UNIQUE:"+rid)
    demand(len(market)==36 and sum(map(len,market.values()))==471,
           "MARKET_NOT_REAL_36R_471H")
    checked=[];seen=set();horses=0;probes=0
    for r in lines(pure_scores):
        rid=str(r.get("race_id") or "")
        hs=r.get("horses") or []
        demand(rid in market and rid not in seen and r.get("race_date")==DAY and
               r.get("anchor_status")=="RESEARCH_SCORED_ASOF_UNVERIFIED" and
               r.get("2026_target_results_used") is False and
               r.get("asof_independently_verified") is False and
               r.get("ready_for_betting") is False and
               r.get("l21_l3_invoked") is False,
               "PURE_MODEL_HISTORICAL_SOURCE_DRIFT:"+rid)
        current={str(h["horse_id"]):h for h in hs}
        demand(len(current)==len(market[rid]) and
               set(current)==set(market[rid]),"ORIGINAL_L21_PURE_MARKET_ID_DRIFT:"+rid)
        companions={};win={};old_ranks={}
        for hid,h in current.items():
            mh=market[rid][hid];num=h["horse_number"]
            demand(type(num)is int and num>0 and type(h["market_rank"])is int and
                   num==mh["horse_number"] and h["market_rank"]==mh["market_rank"] and
                   h["original_prior_market_win_odds"]==mh["market_win_odds"] and
                   type(h["distill_ai_rank"])is int and
                   all(type(h[z])in (float,int) and math.isfinite(h[z]) for z in
                       ("distill_score_z","distill_score_gap_to_best","risk","P__ANCHOR_FULL")),
                   "FROZEN_L21_SCORE_PRICE_AND_HORSE_MISMATCH:"+rid+":"+hid)
            demand(num not in companions,"DUPLICATE_ORIGINAL_HORSE_NUMBER:"+rid)
            companions[num]={k:h[k] for k in ("distill_ai_rank",
                  "distill_score_z","distill_score_gap_to_best","risk")}
            companions[num]["horse_id"]=hid
            win[num]=float(mh["market_win_odds"])
            old_ranks[num]=h["market_rank"]
        geom=geometry_fn(companions,win)
        demand(set(geom)==set(companions) and
               all(geom[n]["market_rank"]==old_ranks[n] for n in geom),
               "ORIGINAL_L21_MARKET_COMPETITION_RANK_DRIFT:"+rid)
        # Same immutable original AI_PROB tie-break as lab.sort_horses.
        prob_ranked=sorted(current.values(),
            key=lambda h:(-h["P__ANCHOR_FULL"],h["market_rank"],str(h["horse_id"])))
        prob_rank={h["horse_number"]:ix for ix,h in enumerate(prob_ranked,1)}
        # Exactly the 7 values consumed by original frozen gate_flags.
        panel={}
        for hid,h in current.items():
            n=h["horse_number"];g=geom[n]
            panel[n]={"P__ANCHOR_FULL":h["P__ANCHOR_FULL"],
                      "market_prob_norm":g["market_prob_norm"],
                      "risk":h["risk"],"distill_score_z":h["distill_score_z"],
                      "distill_ai_rank":h["distill_ai_rank"],
                      "market_rank":h["market_rank"],"P_rank":prob_rank[n]}
            demand(set(panel[n])==set(NEEDED) and
                   all(type(v)in (int,float) and math.isfinite(v)
                       for v in panel[n].values()) and
                   0<g["market_prob_norm"]<1 and
                   0<=h["P__ANCHOR_FULL"]<=1,
                   "ORIGINAL_L21_GATE_FEATURE_MISSING_OR_INVALID:"+rid+":"+hid)
        demand(abs(math.fsum(q["market_prob_norm"] for q in panel.values())-1)<1e-9 and
               abs(math.fsum(q["P__ANCHOR_FULL"] for q in panel.values())-1)<1e-9 and
               set(prob_rank.values())==set(range(1,len(panel)+1)),
               "ORIGINAL_L21_MARKET_NORM_OR_PROBABILITY_RANK_DRIFT:"+rid)
        axis=r["anchor_horse_number"]
        demand(axis in panel and sum(h["original_anchor"] for h in current.values())==1 and
               next(h["horse_number"] for h in current.values() if h["original_anchor"])==axis,
               "ORIGINAL_L21_ANCHOR_AXIS_DRIFT:"+rid)
        # Probe the EXACT original frozen function using non-purchase pairs.
        # This is feature-schema coverage only; NOT ticket selection.
        for n in panel:
            if n==axis:continue
            flags=flag_fn(("QUINELLA",tuple(sorted((axis,n)))),panel,axis)
            demand(gates.issubset(flags) and
                   all(type(flags[g])is bool for g in gates),
                   "ORIGINAL_FROZEN_GATE_PROBE_FAILED:"+rid+":"+str(n))
            probes+=1
        checked.append({"race_id":rid,"horses":len(hs),
                        "original_gate_input_features":list(NEEDED),
                        "missing_original_gate_fields":[],
                        "original_gate_probe_pairs":len(panel)-1,
                        "source_asof_independently_verified":False,
                        "l21_ticket_selection_run":False,
                        "ready_for_betting":False})
        horses+=len(hs);seen.add(rid)
    demand(len(seen)==36 and seen==set(market) and horses==471 and probes==435,
           "ORIGINAL_FROZEN_50_GATE_FEATURE_AUDIT_SCOPE_DRIFT")
    result={
        "contract":"KEIBA_2026_1017_ORIGINAL_FROZEN_L21_GATE_PANEL_READONLY_AUDIT_V1",
        "races":36,"horses":471,"frozen_standard_rules":50,
        "frozen_gate_names":sorted(gates),"frozen_gate_names_covered":len(gates),
        "original_gate_feature_fields":list(NEEDED),
        "original_gate_feature_values_verified":horses*len(NEEDED),
        "original_gate_probe_pairs_no_bets":probes,
        "original_2025_anchor_market_probability_formula_reused":True,
        "original_2025_gate_flags_function_reused":True,
        "original_prob_rank_tie_policy_from_frozen_lab":True,
        "missing_gate_fields":[],
        "independent_original_asof_certification":False,
        "model_release_approval":False,
        "bet_cutoff_ticket_odds_certification":False,
        "frozen_l21_ticket_engine_invoked":False,
        "l3_v3_invoked":False,"ready_for_betting":False,
        "races_audited":sorted(checked,key=lambda x:x["race_id"])
    }
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("ORIGINAL_FROZEN_L21_REAL_GATE_PANEL_VERIFIED "+
          json.dumps({"races":36,"horses":471,"gate_names":len(gates),
            "values":horses*len(NEEDED),"safe_probes":probes,
            "l21_ticket_engine":False,"model_approved":False},sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("pure-index","pure-scores","market-index","market-scores",
        "frozen-contract","original-gate-code","original-geometry-code","output"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    run(a.pure_index,a.pure_scores,a.market_index,a.market_scores,
        a.frozen_contract,a.original_gate_code,a.original_geometry_code,a.output)
