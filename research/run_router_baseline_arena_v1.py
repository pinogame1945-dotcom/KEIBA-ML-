#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

ROLES={"TOP1":"top1_hit","TOP3":"top3_hit","TOP6":"top6_hit"}


def parse_args():
    p=argparse.ArgumentParser(description="Baseline Router Arena for TOP1/TOP3/TOP6 expert routing.")
    p.add_argument("--features",required=True)
    p.add_argument("--labels",required=True)
    p.add_argument("--output",required=True)
    return p.parse_args()


def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")


def read_rows(path,contract):
    rows={}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            if row.get("contract")!=contract:
                raise ValueError(f"unexpected contract in {path}")
            rows[str(row["race_id"])]=row
    return rows


def choose_max(experts,key,reverse=True):
    def val(name):
        v=experts[name].get(key)
        if v is None:
            return float("-inf") if reverse else float("inf")
        return float(v)
    return sorted(experts,key=lambda n:(val(n),n),reverse=reverse)[0]


def majority_choice(experts):
    votes=Counter(e["top1_horse_id"] for e in experts.values())
    max_vote=max(votes.values())
    horses={h for h,v in votes.items() if v==max_vote}
    eligible=[n for n,e in experts.items() if e["top1_horse_id"] in horses]
    return sorted(eligible,key=lambda n:(float(experts[n].get("top1_top2_gap") or 0.0),n),reverse=True)[0]


def selector_metrics(features,labels,selector,role):
    key=ROLES[role]
    hits=0
    choices=Counter()
    for rid,f in features.items():
        name=selector(f["experts"])
        choices[name]+=1
        hits+=int(labels[rid]["expert_labels"][name][key])
    return {
        "races":len(features),
        "hits":hits,
        "hit_rate":hits/len(features) if features else None,
        "choice_counts":dict(choices),
    }


def fixed_metrics(features,labels,name,role):
    key=ROLES[role]
    hits=sum(int(labels[rid]["expert_labels"][name][key]) for rid in features)
    return {"races":len(features),"hits":hits,"hit_rate":hits/len(features) if features else None}


def oracle(features,labels,role):
    key=ROLES[role]
    hits=sum(any(x[key] for x in labels[rid]["expert_labels"].values()) for rid in features)
    return {"races":len(features),"hits":hits,"hit_rate":hits/len(features) if features else None,"hindsight_only":True}


def main():
    a=parse_args()
    features=read_rows(a.features,"ROUTER_FEATURE_SNAPSHOT_V1")
    labels=read_rows(a.labels,"ROUTER_ROLE_LABELS_V1")
    if set(features)!=set(labels):
        raise ValueError("feature/label race coverage differs")
    if not features:
        raise ValueError("no races")
    expert_names=sorted(next(iter(features.values()))["experts"])
    for f in features.values():
        if sorted(f["experts"])!=expert_names:
            raise ValueError("expert set differs by race")

    selectors={
        "max_top1_top2_gap":lambda e:choose_max(e,"top1_top2_gap",True),
        "max_top1_probability":lambda e:choose_max(e,"top1_probability",True),
        "min_entropy":lambda e:choose_max(e,"normalized_entropy",False),
        "majority_top1_then_gap":majority_choice,
    }
    roles={}
    for role in ROLES:
        roles[role]={
            "fixed_experts":{name:fixed_metrics(features,labels,name,role) for name in expert_names},
            "selectors":{name:selector_metrics(features,labels,fn,role) for name,fn in selectors.items()},
            "oracle_any_expert":oracle(features,labels,role),
        }

    summary={
        "contract":"ROUTER_BASELINE_ARENA_V1",
        "races":len(features),
        "experts":expert_names,
        "roles":roles,
        "notes":[
            "No odds are used.",
            "Selectors choose an expert; they do not make buy/skip decisions.",
            "Oracle is hindsight-only and is an upper-bound diagnostic, never a deployable strategy."
        ],
    }
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ROUTER_BASELINE_ARENA_V1_OK")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
