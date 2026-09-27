#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

FEATURE_CONTRACT="L15_COLUMN_CANDIDATE_FEATURES_V1"
LABEL_CONTRACT="L15_COLUMN_CANDIDATE_LABELS_V1"


def parse_args():
    p=argparse.ArgumentParser(description="Baseline arena for L1.5 column-router candidate sets.")
    p.add_argument("--features",required=True)
    p.add_argument("--labels",required=True)
    p.add_argument("--output",required=True)
    return p.parse_args()


def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")


def read(path,contract):
    rows={}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            if row.get("contract")!=contract:
                raise ValueError(f"unexpected contract in {path}")
            cid=str(row["candidate_id"])
            if cid in rows:
                raise ValueError(f"duplicate candidate_id {cid}")
            rows[cid]=row
    return rows


def main():
    a=parse_args()
    features=read(a.features,FEATURE_CONTRACT)
    labels=read(a.labels,LABEL_CONTRACT)
    if set(features)!=set(labels):
        raise ValueError("feature/label candidate coverage differs")

    grouped=defaultdict(lambda:{"races":set(),"hits":0,"rows":0})
    race_column_n=defaultdict(lambda:defaultdict(list))
    race_column_any=defaultdict(list)

    for cid,f in features.items():
        l=labels[cid]
        key=(f["column"],f["expert_name"],int(f["top_n"]))
        g=grouped[key]
        g["races"].add(str(f["race_id"]))
        g["hits"]+=int(bool(l["slot_hit"]))
        g["rows"]+=1
        rc=(str(f["race_id"]),f["column"])
        race_column_n[rc][int(f["top_n"])].append(bool(l["slot_hit"]))
        race_column_any[rc].append(bool(l["slot_hit"]))

    fixed=defaultdict(lambda:defaultdict(dict))
    for (column,expert,n),g in sorted(grouped.items()):
        races=len(g["races"])
        hit_rate=g["hits"]/races if races else None
        fixed[column][expert][str(n)]={
            "races":races,
            "hits":g["hits"],
            "hit_rate":hit_rate,
            "candidate_size":n,
            "capture_per_candidate_horse":hit_rate/n if hit_rate is not None else None,
        }

    oracle_same_size=defaultdict(dict)
    for column in sorted({f["column"] for f in features.values()}):
        ns=sorted({int(f["top_n"]) for f in features.values() if f["column"]==column})
        for n in ns:
            races=0; hits=0
            for (rid,c),by_n in race_column_n.items():
                if c!=column or n not in by_n: continue
                races+=1
                hits+=int(any(by_n[n]))
            oracle_same_size[column][str(n)]={
                "races":races,
                "hits":hits,
                "hit_rate":hits/races if races else None,
                "hindsight_only":True,
            }

    oracle_any={}
    for column in sorted({f["column"] for f in features.values()}):
        rows=[(rc,hits) for rc,hits in race_column_any.items() if rc[1]==column]
        races=len(rows)
        hits=sum(int(any(v)) for _,v in rows)
        oracle_any[column]={
            "races":races,
            "hits":hits,
            "hit_rate":hits/races if races else None,
            "hindsight_only":True,
            "warning":"Diagnostic upper bound only; larger candidate sets dominate raw capture."
        }

    out={
        "contract":"L15_COLUMN_BASELINE_ARENA_V1",
        "candidate_rows":len(features),
        "fixed_candidates":fixed,
        "oracle_same_top_n":oracle_same_size,
        "oracle_any_candidate":oracle_any,
        "notes":[
            "No odds are used.",
            "Raw hit rate must be interpreted together with candidate size.",
            "oracle_same_top_n measures expert-routing headroom without changing candidate size.",
            "No buy/skip or stake decisions are made here."
        ]
    }
    p=Path(a.output)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L15_COLUMN_BASELINE_ARENA_V1_OK")
    print(json.dumps(out,ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
