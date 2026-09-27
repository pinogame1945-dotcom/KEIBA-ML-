#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

CONTRACT="ROUTER_ROLE_LABELS_V1"


def parse_args():
    p=argparse.ArgumentParser(description="Join post-race truth to a pre-race Router Feature Snapshot as a separate label file.")
    p.add_argument("--features",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--output",required=True)
    return p.parse_args()


def open_text(path,mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path,mode,encoding="utf-8")
    return open(path,mode,encoding="utf-8")


def read_features(path):
    rows={}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            if row.get("contract")!="ROUTER_FEATURE_SNAPSHOT_V1":
                raise ValueError("unexpected router feature contract")
            rid=str(row["race_id"])
            rows[rid]=row
    return rows


def read_winners(path,wanted):
    winners=defaultdict(set)
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            if rid not in wanted: continue
            target=row.get("target") or {}
            if target.get("is_win") is True:
                winners[rid].add(str(row.get("horse_id") or ""))
    missing=[rid for rid in wanted if not winners[rid]]
    if missing:
        raise ValueError(f"{len(missing)} races have no winner label")
    return winners


def first_winner_rank(expert,winners):
    for idx,hid in enumerate(expert.get("top6_horse_ids") or [],start=1):
        if hid in winners:
            return idx
    return None


def main():
    a=parse_args()
    features=read_features(a.features)
    winners=read_winners(a.snapshot,set(features))
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    writer=gzip.open if str(out).endswith(".gz") else open
    count=0
    with writer(out,"wt",encoding="utf-8") as fh:
        for rid in sorted(features):
            f=features[rid]
            labels={}
            for name,expert in f["experts"].items():
                ws=winners[rid]
                labels[name]={
                    "top1_hit":expert["top1_horse_id"] in ws,
                    "top3_hit":bool(ws.intersection(expert.get("top3_horse_ids") or [])),
                    "top6_hit":bool(ws.intersection(expert.get("top6_horse_ids") or [])),
                    "winner_rank_within_top6":first_winner_rank(expert,ws),
                }
            row={
                "contract":CONTRACT,
                "race_id":rid,
                "race_date":f.get("race_date"),
                "expert_labels":labels,
            }
            fh.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
            count+=1
    print("ROUTER_ROLE_LABELS_V1_OK")
    print(json.dumps({"races":count,"output":str(out)},ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
