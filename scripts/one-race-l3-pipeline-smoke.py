#!/usr/bin/env python3
"""Synth-only end-to-end exact JRA 4-product -> frozen L2.1 -> baseline L3.
No historical result, remote services, place betting or 2026 scoring.
"""
import importlib.util,itertools,json,tempfile
from pathlib import Path
script=Path("research/run-one-race-paper-pipeline.py").resolve()
spec=importlib.util.spec_from_file_location("keiba_orchestrator",script)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

rid="202507010212"
key="SYNTHETIC_JRA_RACE_CARD";url="https://www.jra.go.jp/test"
observed="2025-01-06T13:32:00+09:00"
def row(bt,seq,price):
    return {"raceKey":key,"betType":bt,"selection1":seq[0],
            "selection2":seq[1] if len(seq)>1 else None,
            "selection3":seq[2] if len(seq)>2 else None,
            "odds":price,"oddsMin":None,"oddsMax":None,
            "observedAt":observed,"sourceUrl":url}
entries=[{"raceKey":key,"horseNo":n,"canonicalHorseId":f"horse{n}",
          "entryStatus":"ACTIVE"} for n in range(1,9)]
odds=[row("WIN",(n,),float(n+2)) for n in range(1,9)]
for a,b in itertools.combinations(range(1,9),2):
    odds.append(row("QUINELLA",(a,b),16.0))
for a,b,c in itertools.combinations(range(1,9),3):
    odds.append(row("TRIO",(a,b,c),16.0 if (b+c)%2 else 35.0))
for a,b,c in itertools.permutations(range(1,9),3):
    odds.append(row("TRIFECTA",(a,b,c),
                    240.0 if a==1 else 25.0 if c==1 else 500.0))
assert len(odds)==428

card={"race":{"raceKey":key,"canonicalRaceId":rid,"raceDate":"2025-01-06",
              "startTime":"14:30","raceStatus":"SCHEDULED",
              "scheduleStatus":"ACTIVE","status":"OFFICIAL",
              "sourceUrl":url},
      "entries":entries,
      "capture":{"source_system":"KEIBA_MOBILE_JRA",
                 "final_probe_checked":True,"is_final":False,
                 "source_status":"LIVE",
                 "final_probe_checked_at":"2025-01-06T13:33:00+09:00",
                 "bet_cutoff_at":"2025-01-06T14:25:00+09:00"},
      "odds_rows":odds}
model={"contract":"KEIBA_L21_ORIGINAL_ANCHOR_FEATURES_PREBET_V1",
       "race_id":rid,"canonical_race_key":key,
       "anchor_source":"EXACT_EXISTING_ANCHOR_FULL","anchor_horse_id":"horse1",
       "feature_source":"ORIGINAL_L21_HORSE_FEATURES","input_phase":"PRE_RACE",
       "features_asof_asserted":True,
       "features_observed_at":"2025-01-06T13:00:00+09:00",
       "win_odds_observed_at":observed,
       "anchor_computed_at":"2025-01-06T13:34:00+09:00",
       "horses":[{"horse_id":f"horse{n}","horse_number":n,
         "P__ANCHOR_FULL":0.31 if n==1 else 0.14-.005*n,
         "risk":0.09,"distill_score_z":0.45,
         "distill_ai_rank":{2:3,3:4,4:5,5:2}.get(n,n),
         "market_rank":n,"market_prob_norm":0.15,
         "win_odds_at_cutoff":float(n+2)} for n in range(1,9)]}
limits={"contract":"KEIBA_L3_FIXED100_LIMITS_V1",
        "race_budget_yen":10000,"max_race_exposure_yen":10000,
        "max_ticket_count":1000,
        "stake_policy":"BASELINE_FLAT_100_ALL_OR_NONE"}
analysis={"contract":"KEIBA_ONE_RACE_COMBINED_ANALYSIS_V1",
          "race_id":rid,
          "horses":[{"horse_id":h["horse_id"],"horse_number":h["horse_number"]}
                    for h in model["horses"]]}
with tempfile.TemporaryDirectory() as d:
    d=Path(d)
    def put(name,x):
        p=d/name;p.write_text(json.dumps(x,ensure_ascii=False),encoding="utf-8")
        return str(p)
    m=put("mobile.json",card)
    orig=put("original.json",model)
    cash=put("limits.json",limits)
    l1=put("l1.json",analysis)
    missing=module.run(rid,d/"missing")
    assert missing["status"]=="BLOCKED_MISSING_INPUTS"
    assert len(missing["required_inputs_missing"])==3
    report=module.run(rid,d/"all",m,orig,cash,l1)
    assert report["status"]=="PAPER_ONLY_LIMITS_CHECKED",report
    assert report["stages"]["L1_75_PRODUCTION"]=="BLOCKED_NO_PROMOTED_ASOF_MARKET_CALIBRATOR"
    tickets=json.loads((d/"all/L2_1-quoted-tickets.json").read_text())
    portfolio=json.loads((d/"all/L3-fixed100-paper-portfolio.json").read_text())
    assert tickets["physical_ticket_count"]>0
    assert tickets["status"]=="PAPER_ONLY_PREBET_QUOTE_VALIDATED"
    assert portfolio["status"]=="PAPER_ONLY_BASELINE_READY"
    assert portfolio["required_ticket_count"]==tickets["physical_ticket_count"]
    assert portfolio["portfolio_total_yen"]==tickets["baseline_total_stake_yen"]
    assert len(portfolio["tickets"])==len(tickets["tickets"])
    assert all(item["stake_yen"]==100 for item in portfolio["tickets"])
    assert (d/"all/L2_1-preview.md").exists()
    assert not portfolio["any_purchase_executed"]
    under=put("under.json",{**limits,"race_budget_yen":100})
    shortage=module.run(rid,d/"budget",m,orig,under,l1)
    assert shortage["status"]=="BLOCKED_UPSTREAM_OR_BUDGET"
    shortage_obj=json.loads((d/"budget/L3-fixed100-paper-portfolio.json").read_text())
    assert shortage_obj["status"]=="INSUFFICIENT_BUDGET_DO_NOT_DROP_TICKETS"
    assert shortage_obj["tickets"]==[]
    broken={**card,"odds_rows":[r for r in odds if not(
        r["betType"]=="QUINELLA" and r["selection1"]==1 and r["selection2"]==5)]}
    bad=put("partial.json",broken)
    subset=module.run(rid,d/"partial",bad,orig,cash,l1)
    assert subset["status"]=="BLOCKED_UPSTREAM_OR_BUDGET",subset
    assert subset["result_portfolio_status"]=="BLOCKED_UPSTREAM_PRICE_COVERAGE"
    assert not json.loads((d/"partial/L3-fixed100-paper-portfolio.json").read_text())["tickets"]
    fake={**model,"win_odds_observed_at":"2025-01-06T13:31:00+09:00"}
    mismatch=module.run(rid,d/"asof",m,put("asof.json",fake),cash,l1)
    assert mismatch["status"]=="BLOCKED_CONTRACT_OR_PROVENANCE_DRIFT"
    assert "SAME_MARKET_SNAPSHOT" in mismatch["failure_reason"]
    # The original (non-promoted) 2025 ANCHOR_FULL can also be scored
    # from authentic feature math + prior-only scaler and original asof
    # upstream companion fields. This is synthetic, NOT an actual past bet.
    import datetime
    import importlib.util
    adapter_spec=importlib.util.spec_from_file_location(
        "original_anchor_model",Path("research/score-original-anchor-asof.py").resolve())
    adapter=importlib.util.module_from_spec(adapter_spec)
    adapter_spec.loader.exec_module(adapter)
    params={
       "contract":"KEIBA_ORIGINAL_ANCHOR_FULL_PRIOR_TRAINED_PARAMS_V1",
       "variant":"ANCHOR_FULL","trained_only_on_years":[2022,2023,2024],
       "features_in_original_order":list(adapter.MODEL.FEATURES),
       "not_promoted_or_authenticated_for_live_betting":True,
       "coefficients":{k:(1.0 if k=="distill_score_z" else 0.0)
                       for k in adapter.MODEL.FEATURES},
       "standardizer":{k:{"mean":0,"std":1,"median":0}
                       for k in adapter.MODEL.FEATURES}
    }
    companion={
       "contract":"KEIBA_L1_ORIGINAL_ABILITY_RISK_ASOF_V1",
       "race_id":rid,"canonical_race_key":key,
       "feature_source":"ORIGINAL_PREBET_ABILITY_AND_RISK",
       "source_observed_before_bet":True,
       "observed_at":"2025-01-06T13:00:00+09:00",
       "horses":[{"horse_id":h["horse_id"],"horse_number":h["horse_number"],
                  "distill_ai_rank":h["distill_ai_rank"],
                  "distill_score_z":float(10-h["horse_number"]),
                  "distill_score_gap_to_best":float(h["horse_number"]-1),
                  "risk":h["risk"]} for h in model["horses"]],
    }
    synthetic_time=datetime.datetime.fromisoformat("2025-01-06T13:34:00+09:00")
    scored=module.run(rid,d/"reconstructed",m,None,cash,l1,
       put("prior-params.json",params),put("true-companions.json",companion),
       _synthetic_now=synthetic_time)
    assert scored["status"]=="PAPER_ONLY_LIMITS_CHECKED",scored
    assert scored["stages"]["ORIGINAL_ANCHOR_FULL"].startswith("RECONSTRUCTED_PRIOR_MODEL")
    assert scored["stages"]["L1_75_PRODUCTION"].startswith("NOT_PROMOTED")
    rebuilt=json.loads((d/"reconstructed/L2_1-original-asof-features.json").read_text())
    assert rebuilt["anchor_source"]=="EXACT_EXISTING_ANCHOR_FULL"
    assert len(rebuilt["horses"])==8
    assert abs(sum(h["P__ANCHOR_FULL"] for h in rebuilt["horses"])-1)<1e-10
    assert json.loads((d/"reconstructed/L3-fixed100-paper-portfolio.json").read_text())["any_purchase_executed"] is False
    print("PRIOR_ORIGINAL_ANCHOR_FULL_MODEL_TO_FROZEN_L21_TO_FLAT100_L3_SYNTHETIC_E2E_PASS",
          rebuilt["anchor_horse_id"],scored.get("result_ticket_count"))
    print("ONE_RACE_L1_ATTACH_TO_L2_FIXED_50_GATES_TO_L3_FIXED100_PAPER_E2E_PASS",
          "physical",tickets["physical_ticket_count"],
          "budget",portfolio["portfolio_total_yen"],
          "only_original_anchor","full_jra_4_products",len(odds))
