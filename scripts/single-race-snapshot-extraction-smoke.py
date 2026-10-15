#!/usr/bin/env python3
"""Proof that archived one-race feature projection excludes targets."""
import importlib.util,json,tempfile
from pathlib import Path
spec=importlib.util.spec_from_file_location("extract_race",Path("research/extract-race-snapshot.py").resolve())
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    inp=root/"all.jsonl"
    rid="202507010212"
    rows=[]
    for hid in ("11","22","33"):
        rows.append({"race_id":rid,"horse_id":hid,
          "feature_schema_version":8,"leakage_policy":"STRICT_PRIOR_DATE_ONLY",
          "features":{"race_date":"2025-01-06","historic_form":2.5},
          "target":{"is_win":hid=="11","finish_position":1 if hid=="11" else 9},
          "result":{"finish":1},"payouts":[10]})
    rows.append({"race_id":"202507010211","horse_id":"44",
                 "feature_schema_version":8,"leakage_policy":"STRICT_PRIOR_DATE_ONLY",
                 "features":{"race_date":"2025-01-06"}})
    inp.write_text("\n".join(json.dumps(x) for x in rows),encoding="utf-8")
    out=root/"race.jsonl"
    module.extract(inp,rid,out)
    result=[json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(result)==3
    assert sorted(r["horse_id"] for r in result)==["11","22","33"]
    assert all(not set(r).intersection({"target","result","payouts"}) for r in result)
    assert all(r["features"]["historic_form"]==2.5 for r in result)
    try:module.extract(inp,"202607010212",root/"future.jsonl")
    except ValueError as ex:assert "2026 sealed" in str(ex)
    else:raise AssertionError("sealed year accepted")
print("ONE_RACE_SNAPSHOT_SINGLE_PASS_LABEL_REDACTION_SAFE")
