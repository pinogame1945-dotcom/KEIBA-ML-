#!/usr/bin/env python3
import argparse,gzip,hashlib,itertools,json
from collections import defaultdict
from pathlib import Path

SRC_CONTRACT="L15_COLUMN_CANDIDATE_FEATURES_V1"
FEATURE_CONTRACT="L15_ROLE_CANDIDATE_FEATURES_V2"
LABEL_CONTRACT="L15_ROLE_CANDIDATE_LABELS_V2"

ROLE_SOURCE={"ANCHOR":"COL1","MAINLINE":"COL2","COVER":"COL3"}

def open_text(path,mode="rt"):
    return gzip.open(path,mode,encoding="utf-8") if str(path).endswith(".gz") else open(path,mode,encoding="utf-8")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--source-features",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--features-out",required=True)
    p.add_argument("--labels-out",required=True)
    return p.parse_args()

def ordered_outcomes(groups,slots=3):
    pieces=[()]
    for rank in sorted(groups):
        if rank>slots: continue
        horses=list(groups[rank])
        occupied=min(len(horses),slots-rank+1)
        if occupied<=0: continue
        perms=list(itertools.permutations(horses,occupied))
        pieces=[a+b for a in pieces for b in perms]
    return sorted(set(x for x in pieces if len(x)==slots))

def read_truth(path):
    groups=defaultdict(lambda:defaultdict(list))
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            hid=str(row.get("horse_id") or "")
            target=row.get("target") or {}
            try:
                finish=int(float(target.get("finish_position")))
            except (TypeError,ValueError):
                continue
            if rid and hid and finish<=3:
                groups[rid][finish].append(hid)
    out={}
    dead=0
    for rid,g in groups.items():
        outcomes=ordered_outcomes(g,3)
        if outcomes:
            out[rid]=outcomes
            if any(len(v)>1 for v in g.values()):
                dead+=1
    return out,dead

def cid(rid,role,expert,top_n):
    return hashlib.sha256(f"{rid}|{role}|{expert}|{top_n}".encode()).hexdigest()[:20]

def main():
    a=parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8"))
    if cfg.get("contract")!="L15_ROLE_ROUTER_EXPERIMENT_V2": raise ValueError("bad config")
    if 2026 not in cfg.get("locked_years",[]): raise ValueError("2026 lock missing")
    truth,dead=read_truth(a.snapshot)
    fpath=Path(a.features_out); lpath=Path(a.labels_out)
    fpath.parent.mkdir(parents=True,exist_ok=True); lpath.parent.mkdir(parents=True,exist_ok=True)
    fw=gzip.open if str(fpath).endswith(".gz") else open
    lw=gzip.open if str(lpath).endswith(".gz") else open
    frows=lrows=missing=0
    seen=set()
    with open_text(a.source_features) as src, fw(fpath,"wt",encoding="utf-8") as fo, lw(lpath,"wt",encoding="utf-8") as lo:
        for line in src:
            if not line.strip(): continue
            row=json.loads(line)
            if row.get("contract")!=SRC_CONTRACT: raise ValueError("bad source contract")
            source_col=row.get("column")
            role=next((r for r,c in ROLE_SOURCE.items() if c==source_col),None)
            if not role: continue
            n=int(row["top_n"])
            if n not in cfg["roles"][role]["top_n"]: continue
            rid=str(row["race_id"])
            if rid not in truth:
                missing+=1; continue
            new_id=cid(rid,role,row["expert_name"],n)
            if new_id in seen: raise ValueError(f"duplicate {new_id}")
            seen.add(new_id)
            cand=set(map(str,row.get("candidate_horse_ids") or []))
            outcomes=truth[rid]
            anchor=any(o[0] in cand for o in outcomes)
            any_opp=any(bool(cand.intersection(o[1:3])) for o in outcomes)
            both_opp=any(set(o[1:3]).issubset(cand) for o in outcomes)
            recall=max((len(cand.intersection(o[1:3]))/2.0 for o in outcomes),default=0.0)
            target={"ANCHOR":anchor,"MAINLINE":any_opp,"COVER":both_opp}[role]

            feat=dict(row)
            feat["contract"]=FEATURE_CONTRACT
            feat.pop("column",None)
            feat["role"]=role
            feat["candidate_id"]=new_id
            lab={
                "contract":LABEL_CONTRACT,
                "race_id":rid,
                "race_date":row.get("race_date"),
                "candidate_id":new_id,
                "role":role,
                "expert_name":row["expert_name"],
                "top_n":n,
                "role_hit":bool(target),
                "winner_capture":bool(anchor),
                "any_podium_opponent_capture":bool(any_opp),
                "both_podium_opponents_capture":bool(both_opp),
                "podium_opponent_recall":recall,
            }
            fo.write(json.dumps(feat,ensure_ascii=False,separators=(",",":"))+"\n")
            lo.write(json.dumps(lab,ensure_ascii=False,separators=(",",":"))+"\n")
            frows+=1; lrows+=1
    print("L15_ROLE_DATASET_V2_OK")
    print(json.dumps({"feature_rows":frows,"label_rows":lrows,"missing_truth_rows":missing,"dead_heat_races":dead},separators=(",",":")))

if __name__=="__main__":
    main()
