#!/usr/bin/env python3
"""Historical market fair OOS is non-promoted and cannot leak outcomes."""
import csv,gzip,importlib.util,json,tempfile
from pathlib import Path
p=Path("research/replay-final-market-oos-v2.py").resolve()
sp=importlib.util.spec_from_file_location("fair_v2",p)
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
with tempfile.TemporaryDirectory() as root:
    d=Path(root)
    analysis={"contract":"KEIBA_ONE_RACE_COMBINED_ANALYSIS_V1",
              "race_id":"202507010212",
              "horses":[{"horse_id":str(x)} for x in (111,222,333)]}
    meta={"contract":"MARKET_FIXED_ANCHOR_RESIDUAL_V2_FROM_CACHE",
          "promotion":False,"2026_locked":True,
          "roi_target_used_for_training":False,
          "lambda_selection":"prior-data chronological 80/20 only"}
    (d/"meta.json").write_text(json.dumps(meta))
    with open(d/"lambda.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["test_year","variant","lambda","selected"]);w.writeheader()
        for v in ("ANCHOR_ABILITY","ANCHOR_ABILITY_RISK","ANCHOR_FULL"):
            w.writerow({"test_year":2025,"variant":v,"lambda":"0.01","selected":1})
    rows=[]
    for i,h in enumerate(analysis["horses"]):
        p=[0.6,0.3,0.1][i]
        rows.append({"year":2025,"_race_id":analysis["race_id"],
                     "_horse_id":h["horse_id"],"_race_date":"2025-01-06",
                     "market_prob_norm":p,"P__MARKET_DIRECT":p,
                     "P__ANCHOR_ABILITY":p,"P__ANCHOR_ABILITY_RISK":p,
                     "P__ANCHOR_FULL":p,"risk":0.3,
                     "target_win":1 if i==0 else 0,"target_top3":i<3,
                     "final_win_odds":10.0*i+2})
    def write():
        with gzip.open(d/"fair.csv.gz","wt",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write()
    def run():
        return m.replay(analysis,d/"fair.csv.gz",d/"meta.json",d/"lambda.csv")
    x=run()
    assert x["horse_count"]==3 and x["model_count"]==4
    assert x["promotion"] is False and x["is_prestart_market"] is False
    assert all("target_win" not in h and "final_win_odds" not in h
               for h in x["horse_results"].values())
    for row in rows:
        row["target_win"]=0
        row["target_top3"]=False
        row["final_win_odds"]=999999
    write()
    assert run()==x, "outcomes/final price must not change stored forecast extraction"
    meta["promotion"]=True
    (d/"meta.json").write_text(json.dumps(meta))
    try:run()
    except ValueError as e:assert "PROMOTION" in str(e)
    else:raise AssertionError("Promoted without proven production")
    meta["promotion"]=False
    (d/"meta.json").write_text(json.dumps(meta))
    rows[0]["P__ANCHOR_FULL"]=0.999
    write()
    try:run()
    except ValueError as e:assert "NORMALIZATION" in str(e)
    else:raise AssertionError("partial probabilities accepted")
print("FAIR_FINAL_OOS_NOT_PRODUCTION_AND_NO_OUTCOME_LEAK_PASS")
