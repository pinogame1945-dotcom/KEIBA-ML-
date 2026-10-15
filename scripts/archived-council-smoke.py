#!/usr/bin/env python3
"""Prove archived council replay never uses outcome/final market fields."""
import csv,gzip,importlib.util,json,tempfile
from pathlib import Path
s=importlib.util.spec_from_file_location("archive","research/replay-archived-council.py")
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
with tempfile.TemporaryDirectory(prefix="keiba-council-fixture-") as root:
    root=Path(root)
    ids=["1111111111","2222222222","3333333333"]
    ballots={}
    for index,name in enumerate(m.OUTSIDERS):
        ballots[name]=ids[index%3:]+ids[:index%3]
    support=m.aggregate_ballots(ballots)
    archive={
        "year":2025,"race_date":"2025-01-05","race_id":"202505010101",
        "horses":ids,
        "rank_matrix":[[1,2,3] if index%2==0 else [3,2,1]
                       for index in range(len(m.EXPERTS))],
        "outsider_support":[[hid,z["points"],z["votes"],z["best"]] for hid,z in support.items()],
        "winner":[ids[2]],"podium":ids,
        "market_order":[ids[2],ids[1],ids[0]],
        "market_ranks":[3,2,1],
        "actual_postrace_label":"MUST_NEVER_LEAK"
    }
    file=root/"races.jsonl.gz"
    with gzip.open(file,"wt",encoding="utf-8") as f:
        f.write(json.dumps(archive)+"\n")
    bpath=root/"ballots.csv.gz"
    with gzip.open(bpath,"wt",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["race_id","candidate",
                                       "top1_horse_id","top2_horse_id","top3_horse_id"])
        w.writeheader()
        for name,ranks in ballots.items():
            w.writerow({"race_id":"202505010101","candidate":name,
                **{f"top{i}_horse_id":hid for i,hid in enumerate(ranks,1)}})
    report=m.replay("202505010101",str(file),str(bpath))
    assert report["outsider_status"]=="VERIFIED_SAVED_BALLOTS"
    assert len(report["horses"])==3
    assert len(report["outsider_ballots"])==13
    assert report["horses"][0]["pure_7king_rank_sum_comparator_rank"]==1
    assert report["horses"][0]["horse_id"]==ids[0]
    assert len(report["horses"][0]["king_ranks"])==7
    assert len(report["horses"][0]["outsider_top3_from_each"])==13
    text=json.dumps(report)
    assert "MUST_NEVER_LEAK" not in text
    assert '"winner"' not in text
    assert '"podium"' not in text
    assert '"market_order"' not in text
    q=importlib.util.spec_from_file_location("id_only","research/replay-one-race-id.py")
    id_only=importlib.util.module_from_spec(q)
    q.loader.exec_module(id_only)
    assert id_only.parse_id("202505010101")=="202505010101"
    assert id_only.parse_id("https://race.netkeiba.com/race/shutuba.html?race_id=202505010101")=="202505010101"
    try:id_only.parse_id("https://evil.invalid/?race_id=202505010101")
    except ValueError:pass
    else:raise AssertionError("untrusted race URL host accepted")
    id_report=id_only.render("202505010101",file,bpath)
    assert id_report["model_families"]=={
        "seven_king":7,"outsider_top3_ballot":13,
        "scope":"archived individual ranks and Top3 votes only"
    }
    assert id_report["runners"]==3
    assert id_report["not_live_prediction"] is True
    assert "MUST_NEVER_LEAK" not in json.dumps(id_report)
    assert all(len(h["historical_13outsider_top3_votes"])==13 for h in id_report["horses"])
    archive["outsider_support"][0][1]+=1
    with gzip.open(file,"wt",encoding="utf-8") as f:
        f.write(json.dumps(archive)+"\n")
    try:m.replay("202505010101",str(file),str(bpath))
    except ValueError as e:assert "OUTSIDER_VOTE_DRIFT" in str(e)
    else:raise AssertionError("corrupt ballot support was accepted")
print("ARCHIVED_COUNCIL_7KING_13OUTSIDER_SMOKE_PASS")
