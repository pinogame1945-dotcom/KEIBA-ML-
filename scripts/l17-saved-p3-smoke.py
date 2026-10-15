#!/usr/bin/env python3
"""No-network, outcome-proof saved L1.7 walk-forward probability smoke."""
import csv,gzip,importlib.util,json,tempfile
from pathlib import Path
spec=importlib.util.spec_from_file_location("l17","research/replay-saved-l17.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix="keiba-l17-") as tmp:
    root=Path(tmp)
    source={"race_id":"202507010212","horses":[
        {"horse_id":"1001","outsider_points":6},
        {"horse_id":"1002","outsider_points":2},
        {"horse_id":"1003","outsider_points":0}]}
    summary={"contract":"L17_KING_OUTSIDER_PODIUM_CALIBRATOR_WF_V1",
             "2026_sealed":True,"folds":[
                {"eval_year":2025,"train_years":[2021,2022,2023,2024]}]}
    (root/"summary.json").write_text(json.dumps(summary))
    rows=[]
    for i,horse in enumerate(source["horses"],start=1):
        p3=[0.7,0.4,0.25][i-1]
        baseline=[0.6,0.3,0.2][i-1]
        rows.append({
            "year":2025,"race_id":"202507010212","horse_id":horse["horse_id"],
            "rank":i,"score":horse["outsider_points"],"p3":p3,
            "p3_rank_baseline":baseline,"p3_delta":p3-baseline,
            "finish":111+i,"y3":9000+i
        })
    path=root/"predictions.csv.gz"
    with gzip.open(path,"wt",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=module.replay(source,str(path),str(root/"summary.json"))
    assert len(report["model_scores"])==3
    assert report["prior_train_years"]==[2021,2022,2023,2024]
    assert report["model_scores"]["1001"]["p3_oos"]==0.7
    assert all("finish" not in x and "y3" not in x for x in report["model_scores"].values())
    # Changing outcome labels must not change any model output
    for row in rows:row["finish"]=999999;row["y3"]=1234
    with gzip.open(path,"wt",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert module.replay(source,str(path),str(root/"summary.json"))==report
    summary["folds"][0]["train_years"].append(2025)
    (root/"summary.json").write_text(json.dumps(summary))
    try:module.replay(source,str(path),str(root/"summary.json"))
    except ValueError as ex:assert "target or future" in str(ex)
    else:raise AssertionError("lookahead fold allowed")
print("SAVED_L17_P3_HISTORICAL_OOS_SMOKE_PASS")
