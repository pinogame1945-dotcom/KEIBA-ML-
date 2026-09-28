#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

FEATURE_CONTRACT="L15_COLUMN_CANDIDATE_FEATURES_V1"
LABEL_CONTRACT="L15_COLUMN_CANDIDATE_LABELS_V1"
FORBIDDEN={
    "winner_horse_ids","actual_is_win","actual_finish_position","finish_position","target",
    "payout","payouts","final_win_odds","final_popularity","slot_hit"
}
SLOT_INDEX={"COL1":0,"COL2":1,"COL3":2}


def parse_args():
    p=argparse.ArgumentParser(description="Build L1.5 column-router candidate features and separate labels.")
    p.add_argument("--config",required=True)
    p.add_argument("--router-features",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--expert",action="append",required=True,help="name=path")
    p.add_argument("--features-out",required=True)
    p.add_argument("--labels-out",required=True)
    return p.parse_args()


def open_text(path,mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path,mode,encoding="utf-8")
    return open(path,mode,encoding="utf-8")


def finite(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except (TypeError,ValueError):
        return None


def mean(xs):
    vals=[float(x) for x in xs if x is not None]
    return sum(vals)/len(vals) if vals else None


def jaccard(a,b):
    a=set(a); b=set(b)
    u=a|b
    return len(a&b)/len(u) if u else 1.0


def candidate_id(rid,column,expert,top_n):
    raw=f"{rid}|{column}|{expert}|{top_n}".encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def assert_no_forbidden(value,path="root"):
    if isinstance(value,dict):
        for k,v in value.items():
            if k in FORBIDDEN:
                raise ValueError(f"forbidden feature key {k} at {path}")
            assert_no_forbidden(v,path+"."+str(k))
    elif isinstance(value,list):
        for i,v in enumerate(value):
            assert_no_forbidden(v,path+f"[{i}]")


def read_router_features(path):
    out={}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            if row.get("contract")!="ROUTER_FEATURE_SNAPSHOT_V1":
                raise ValueError("unexpected router feature contract")
            out[str(row["race_id"])]=row
    return out


def read_expert(path):
    races=defaultdict(list)
    meta={"expert_ids":set(),"model_versions":set()}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            if row.get("contract")!="L1_TO_L2_OUTPUT_CONTRACT_V1":
                raise ValueError(f"unexpected expert contract in {path}")
            rid=str(row.get("race_id") or "")
            hid=str(row.get("horse_id") or "")
            if not rid or not hid:
                raise ValueError("missing race_id/horse_id")
            races[rid].append(row)
            meta["expert_ids"].add(str(row.get("expert_id") or ""))
            meta["model_versions"].add(str(row.get("model_version") or ""))
    if len(meta["expert_ids"])!=1 or "" in meta["expert_ids"]:
        raise ValueError(f"expert_id mismatch in {path}")
    if len(meta["model_versions"])!=1:
        raise ValueError(f"model_version mismatch in {path}")
    for rid,rows in races.items():
        rows.sort(key=lambda r:int(r["predicted_rank"]))
        ranks=[int(x["predicted_rank"]) for x in rows]
        if ranks!=list(range(1,len(rows)+1)):
            raise ValueError(f"{rid}: non-contiguous expert ranks")
    return races,{
        "expert_id":next(iter(meta["expert_ids"])),
        "model_version":next(iter(meta["model_versions"])),
    }


def ordered_outcomes(groups,slots):
    pieces=[()]
    for rank in sorted(groups):
        if rank>slots: continue
        horses=list(groups[rank])
        occupied=min(len(horses),slots-rank+1)
        if occupied<=0: continue
        perms=list(itertools.permutations(horses,occupied))
        pieces=[a+b for a in pieces for b in perms]
    return sorted(set(x for x in pieces if len(x)==slots))


def read_truth(path,wanted):
    groups=defaultdict(lambda:defaultdict(list))
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            if rid not in wanted: continue
            target=row.get("target") or {}
            try:
                finish=int(float(target.get("finish_position")))
            except (TypeError,ValueError):
                continue
            if finish<=3:
                groups[rid][finish].append(str(row.get("horse_id") or ""))
    out={}
    bad=[]
    dead=0
    for rid in wanted:
        trio=ordered_outcomes(groups[rid],3)
        if not trio:
            bad.append(rid); continue
        if any(len(v)>1 for v in groups[rid].values()):
            dead+=1
        out[rid]=trio
    if bad:
        raise ValueError(f"{len(bad)} races cannot form ordered podium outcomes")
    return out,dead


def expert_candidate_summary(expert_rows,all_experts,expert_name,top_n):
    rows=expert_rows[:top_n]
    probs=[finite(r.get("race_normalized_win_probability")) for r in rows]
    horse_ids=[str(r["horse_id"]) for r in rows]
    overlaps=[]
    for other_name,other_rows in all_experts.items():
        if other_name==expert_name: continue
        overlaps.append(jaccard(horse_ids,[str(r["horse_id"]) for r in other_rows[:top_n]]))
    return {
        "probability_mass":sum(x for x in probs if x is not None),
        "probability_mean":mean(probs),
        "last_included_probability":probs[-1] if probs else None,
        "same_top_n_jaccard_mean":mean(overlaps),
        "candidate_count":len(horse_ids),
    }


def main():
    a=parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8"))
    if cfg.get("contract")!="L15_COLUMN_ROUTER_EXPERIMENT_V1":
        raise ValueError("unexpected L1.5 config contract")
    if 2026 not in cfg.get("locked_years",[]):
        raise ValueError("2026 lock missing")
    if cfg.get("ability_uses_odds") is not False:
        raise ValueError("L1.5 must not use odds")

    router=read_router_features(a.router_features)
    experts={}
    metas={}
    for spec in a.expert:
        if "=" not in spec:
            raise ValueError("--expert must be name=path")
        name,path=spec.split("=",1)
        experts[name],metas[name]=read_expert(path)

    expected=sorted(cfg.get("expected_experts") or [])
    if expected and sorted(experts)!=expected:
        raise ValueError(f"expert set mismatch: expected={expected} actual={sorted(experts)}")
    race_sets=[set(x) for x in experts.values()]
    if any(s!=race_sets[0] for s in race_sets[1:]):
        raise ValueError("expert race coverage differs")
    wanted=race_sets[0]
    if set(router)!=wanted:
        raise ValueError("router-feature race coverage differs from expert scores")

    per_race=sum(len(v["top_n"]) for v in cfg["candidate_space"].values())*len(experts)
    if per_race>int(cfg["max_candidates_per_race"]):
        raise ValueError(f"candidate space {per_race} exceeds cap {cfg['max_candidates_per_race']}")

    truth,dead=read_truth(a.snapshot,wanted)
    fpath=Path(a.features_out); lpath=Path(a.labels_out)
    fpath.parent.mkdir(parents=True,exist_ok=True)
    lpath.parent.mkdir(parents=True,exist_ok=True)
    fw=gzip.open if str(fpath).endswith(".gz") else open
    lw=gzip.open if str(lpath).endswith(".gz") else open
    feature_rows=0
    label_rows=0

    with fw(fpath,"wt",encoding="utf-8") as f_out, lw(lpath,"wt",encoding="utf-8") as l_out:
        for rid in sorted(wanted):
            base=router[rid]
            per_expert={n:experts[n][rid] for n in experts}
            outcomes=truth[rid]
            for column,space in cfg["candidate_space"].items():
                slot=SLOT_INDEX[column]
                valid_slot_horses=sorted({outcome[slot] for outcome in outcomes})
                for name in sorted(experts):
                    erows=per_expert[name]
                    esummary=base["experts"][name]
                    for top_n in space["top_n"]:
                        n=int(top_n)
                        if n<1 or n>len(erows):
                            continue
                        horses=[str(r["horse_id"]) for r in erows[:n]]
                        cid=candidate_id(rid,column,name,n)
                        feature={
                            "contract":FEATURE_CONTRACT,
                            "race_id":rid,
                            "race_date":base.get("race_date"),
                            "candidate_id":cid,
                            "column":column,
                            "expert_name":name,
                            "expert_id":metas[name]["expert_id"],
                            "model_version":metas[name]["model_version"],
                            "top_n":n,
                            "candidate_horse_ids":horses,
                            "candidate_size_cost":n,
                            "race":base.get("race") or {},
                            "data_coverage":base.get("data_coverage") or {},
                            "consensus":base.get("consensus") or {},
                            "expert_summary":esummary,
                            "candidate_summary":expert_candidate_summary(erows,per_expert,name,n),
                        }
                        assert_no_forbidden(feature)
                        label={
                            "contract":LABEL_CONTRACT,
                            "race_id":rid,
                            "race_date":base.get("race_date"),
                            "candidate_id":cid,
                            "column":column,
                            "expert_name":name,
                            "top_n":n,
                            "slot_hit":bool(set(horses).intersection(valid_slot_horses)),
                            "dead_heat_affected":len(valid_slot_horses)>1,
                        }
                        f_out.write(json.dumps(feature,ensure_ascii=False,separators=(",",":"))+"\n")
                        l_out.write(json.dumps(label,ensure_ascii=False,separators=(",",":"))+"\n")
                        feature_rows+=1
                        label_rows+=1

    print("L15_COLUMN_DATASET_V1_OK")
    print(json.dumps({
        "races":len(wanted),
        "experts":sorted(experts),
        "candidates_per_race":per_race,
        "feature_rows":feature_rows,
        "label_rows":label_rows,
        "dead_heat_races":dead,
        "features_out":str(fpath),
        "labels_out":str(lpath),
    },ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
