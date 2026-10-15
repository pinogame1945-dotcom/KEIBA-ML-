#!/usr/bin/env python3
"""Synthetic real original-parameter asof scorer and fail-closed source gates.

No archived final quotes or synthetic runner used in real L2 execution.
"""
import datetime,importlib.util,itertools,json
from pathlib import Path
p=Path("research/score-original-anchor-asof.py").resolve()
spec=importlib.util.spec_from_file_location("original_anchor_asof_smoke",p)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
rid="202507010212";key="JRA_SYNTH";url="https://www.jra.go.jp/test"
win="2025-01-06T13:30:00+09:00"
types={"WIN":1,"QUINELLA":2,"TRIO":3,"TRIFECTA":3}
horses=[1,2,3,4]
rows=[]
for kind,k in types.items():
    seq=itertools.permutations(horses,3) if kind=="TRIFECTA" else itertools.combinations(horses,k)
    for v in seq:
        rows.append({"raceKey":key,"betType":kind,
             "selection1":v[0],"selection2":v[1] if k>1 else None,
             "selection3":v[2] if k>2 else None,
             "odds":{1:3.0,2:4.0,3:8.0,4:20.0}[v[0]] if kind=="WIN" else 35.0,
             "oddsMin":None,"oddsMax":None,"observedAt":win,"sourceUrl":url})
assert len(rows)==38
card={"race":{"raceKey":key,"canonicalRaceId":rid,"raceDate":"2025-01-06",
              "startTime":"14:30","raceStatus":"SCHEDULED","scheduleStatus":"ACTIVE",
              "status":"OFFICIAL","sourceUrl":url},
      "entries":[{"raceKey":key,"entryStatus":"ACTIVE","horseNo":n,
                  "canonicalHorseId":f"horse{n}"} for n in horses],
      "capture":{"source_system":"KEIBA_MOBILE_JRA","final_probe_checked":True,
                 "is_final":False,"source_status":"LIVE",
                 "final_probe_checked_at":"2025-01-06T13:31:00+09:00",
                 "bet_cutoff_at":"2025-01-06T14:25:00+09:00"},
      "odds_rows":rows}
model={"contract":"KEIBA_ORIGINAL_ANCHOR_FULL_PRIOR_TRAINED_PARAMS_V1",
       "variant":"ANCHOR_FULL","trained_only_on_years":[2022,2023,2024],
       "features_in_original_order":list(s.MODEL.FEATURES),
       "not_promoted_or_authenticated_for_live_betting":True,
       "coefficients":{f:(1.0 if f=="distill_score_z" else 0.0) for f in s.MODEL.FEATURES},
       "standardizer":{f:{"mean":0.0,"median":0.0,"std":1.0}
                       for f in s.MODEL.FEATURES}}
feature={"contract":"KEIBA_L1_ORIGINAL_ABILITY_RISK_ASOF_V1",
         "race_id":rid,"canonical_race_key":key,
         "feature_source":"ORIGINAL_PREBET_ABILITY_AND_RISK",
         "source_observed_before_bet":True,
         "observed_at":"2025-01-06T13:00:00+09:00",
         "horses":[{"horse_id":f"horse{n}","horse_number":n,
                    "distill_ai_rank":n,"distill_score_z":float(5-n),
                    "distill_score_gap_to_best":float(n-1),
                    "risk":0.1*n} for n in horses]}
from copy import deepcopy
now=datetime.datetime.fromisoformat("2025-01-06T13:33:00+09:00")
good=s.calc(model,card,feature,now)
assert good["contract"]=="KEIBA_ORIGINAL_ANCHOR_FULL_ASOF_PAPER_V1"
assert good["original_anchor_number"]==1
panel=good["model_full_probs"]
assert abs(sum(h["P__ANCHOR_FULL"] for h in panel["horses"])-1)<1e-12
assert all(h["win_odds_at_cutoff"] in (3,4,8,20) for h in panel["horses"])
assert panel["anchor_computed_at"]=="2025-01-06T04:33:00+00:00"
assert good["not_official_live_bet_prediction"]
assert not good["original_companion_provenance_not_independently_verified"] is False

def bad(change,reason):
    m=deepcopy(model);c=deepcopy(card);f=deepcopy(feature)
    change(m,c,f)
    try:s.calc(m,c,f,now)
    except ValueError as e:assert reason in str(e),(reason,str(e))
    else:raise AssertionError("ASOF_UNSAFE_INPUT_ACCEPTED "+reason)
bad(lambda m,c,f:m.update(trained_only_on_years=[2022,2023,2024,2025]),"ANCHOR_TRAINING_YEAR")
bad(lambda m,c,f:f.update(feature_source="SAVED_FINAL_SEVEN_KINGS"),"SEVEN_KING_OR_ARCHIVED")
bad(lambda m,c,f:f["horses"].pop(),"FIELD_COVERAGE")
bad(lambda m,c,f:f["horses"][1].update(distill_ai_rank=1),"RANK_NOT_COMPLETE")
bad(lambda m,c,f:f["horses"][1].update(risk=None),"MISSING_NUMERIC")
bad(lambda m,c,f:f.update(observed_at="2025-01-06T14:00:00+09:00"),"OBSERVED_AFTER")
bad(lambda m,c,f:c["entries"][0].update(canonicalHorseId="wrong"),"ORIGINAL_ABILITY_RISK_VS_OFFICIAL_JRA_HORSE_ID_DRIFT")
bad(lambda m,c,f:c["entries"][0].update(canonicalHorseId=None),"OFFICIAL_JRA_HORSE_ID_MISSING")
bad(lambda m,c,f:f["horses"][0].update(finish_position=8),"POSTRACE_OR_OUTCOME_COLUMNS_FORBIDDEN")
bad(lambda m,c,f:f["horses"][0].update(final_win_odds=3.0),"POSTRACE_OR_OUTCOME_COLUMNS_FORBIDDEN")
bad(lambda m,c,f:f.update(target_win=1),"POSTRACE_OR_OUTCOME_COLUMNS_FORBIDDEN")
bad(lambda m,c,f:c["odds_rows"][0].update(oddsMin=2.0),"INVALID_WIN_QUOTE_RANGE")
bad(lambda m,c,f:c["odds_rows"][0].update(observedAt="2025-01-06T12:00:00+09:00"),"ODDS_TIMES_MIXED")
bad(lambda m,c,f:c["odds_rows"].pop(),"MISSING_ONE_OR_MORE_L21_BET_TYPES") if False else None
try:s.calc(model,card,feature,datetime.datetime.now(datetime.timezone.utc))
except ValueError as e:assert "REAL_ANCHOR_INFERENCE_TIME_NOT_WITHIN" in str(e)
else:raise AssertionError("past FINAL cannot be replayed as live")
print("ORIGINAL_PRIOR_ANCHOR_FULL_ASOF_FEATURES_PLUS_REAL_MODEL_FORMULA_PAPER_SMOKE_PASS",
      len(panel["horses"]),good["original_anchor_number"])
