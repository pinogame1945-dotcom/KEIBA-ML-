#!/usr/bin/env python3
"""Synthetic score-file fixtures: no real racing outcome, no network."""
import gzip,importlib.util,json,tempfile
from pathlib import Path

spec=importlib.util.spec_from_file_location("replay7","research/replay-seven-kings.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix="seven-king-test-") as root:
    paths={}
    for i,name in enumerate(module.EXPERTS):
        p=Path(root,f"{name}.jsonl.gz")
        rows=[]
        for rank,hid in enumerate(("2222222222","1111111111") if i%2 else
                                  ("1111111111","2222222222"),start=1):
            rows.append({"contract":module.CONTRACT,"race_id":"202505010111",
                "horse_id":hid,"predicted_rank":rank,"horse_number":1 if hid=="1111111111" else 2,
                "race_date":"2025-02-01","prediction_phase":"FINAL",
                "raw_margin_logit":1.0 if rank==1 else -1.0,
                "raw_win_probability":0.8 if rank==1 else 0.2})
        with gzip.open(p,"wt",encoding="utf-8") as f:
            for row in rows:f.write(json.dumps(row)+"\n")
        paths[name]=str(p)
    result=module.predict("202505010111",paths)
    assert result["model_count"]==7
    assert len(result["horses"])==2
    assert result["horses"][0]["horse_id"]=="1111111111"
    assert result["horses"][0]["rank_sum_comparator_rank"]==1
    assert result["horses"][0]["king_raw_probabilities"]["core4"]==0.8
    first=Path(paths["core4"])
    with gzip.open(first,"rt",encoding="utf-8") as f:rows=[json.loads(l) for l in f]
    rows[1]["horse_id"]=rows[0]["horse_id"]
    with gzip.open(first,"wt",encoding="utf-8") as f:
        for row in rows:f.write(json.dumps(row)+"\n")
    try:module.predict("202505010111",paths)
    except ValueError as e:assert "bad/doubled horse score" in str(e)
    else:raise AssertionError("duplicate did not fail")
print("SEVEN_KING_SAVED_SCORE_REPLAY_SMOKE_PASS")
