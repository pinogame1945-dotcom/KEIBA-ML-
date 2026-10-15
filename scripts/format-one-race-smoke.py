#!/usr/bin/env python3
"""One-race table never confuses FINAL research and live quote."""
import importlib.util
from pathlib import Path
p=Path("research/format-one-race.py").resolve()
spec=importlib.util.spec_from_file_location("format_race",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sample={"contract":"KEIBA_ONE_RACE_COMBINED_ANALYSIS_V1",
        "race_id":"202507010212",
        "prior_year_new_model_inference":{"status":"ORIGINAL_MODELS_SCORED_NOT_VERIFIED_PREBET"},
        "saved_seven_kings":{"status":"GITHUB_ARCHIVED_7KING_REPLAY"},
        "saved_l17_podium":{"status":"ARCHIVED_OOS_P3"},
        "market_context":{"status":"NOT_PROVIDED"},
        "archived_final_market_fair_value":{"status":"HISTORICAL_FINAL_ODDS_OOS_RESEARCH_REPLAY"},
        "horses":[{"horse_id":str(i),"horse_number":i,
                   "horse_name":f"架空馬{i}","prior_year_king_rank_sum":i,
                   "archived_rank_sum_comparator_rank":i,
                   "saved_outsider_points":i,"l17_saved_p3_oos":0.1*i,
                   "archived_final_market_oos_research":{
                     "oos_win_probabilities":{
                       "MARKET_DIRECT":0.2*i,
                       "ANCHOR_FULL":0.25*i
                     }
                   }} for i in (1,2,3)]}
out=m.build(sample)
assert "レースID" in out
assert "架空馬1" in out
assert "研究時の最終市場勝率" in out
assert "正式L1.75ではない" in out
assert "L2.1馬券生成" in out
sample["market_context"]={"status":"TIMESTAMP_CHECKED_SOURCE_AUTHENTICITY_NOT_INDEPENDENTLY_ATTESTED"}
try:m.build(sample)
except ValueError as ex:assert "DO_NOT_SHOW_FINAL" in str(ex)
else:raise AssertionError("final research shown alongside prestart quote")
print("ONE_RACE_HUMAN_REPORT_STRICT_PROVENANCE_PASS")
