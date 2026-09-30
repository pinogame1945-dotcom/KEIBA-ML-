#!/usr/bin/env python3
import argparse, gzip, json, math, statistics
from collections import defaultdict
from pathlib import Path

CONTRACT="L16_FULL_RANK_CONSENSUS_V1"
EXPERT_INPUT_CONTRACT="L1_TO_L2_OUTPUT_CONTRACT_V1"
FORBIDDEN_TOKENS=("odds","popularity","payout","return_yen","profit_yen","roi")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--expert",action="append",required=True,help="alias=score.jsonl[.gz]")
    p.add_argument("--output",required=True)
    return p.parse_args()

def open_text(path,mode="rt"):
    return gzip.open(path,mode,encoding="utf-8") if str(path).endswith(".gz") else open(path,mode,encoding="utf-8")

def finite(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None

def read_expert(path):
    races=defaultdict(list)
    with open_text(path) as f:
        for line in f:
            if not line.strip(): continue
            x=json.loads(line)
            if x.get("contract")!=EXPERT_INPUT_CONTRACT:
                raise ValueError(f"bad input contract: {path}")
            rid=str(x.get("race_id") or ""); hid=str(x.get("horse_id") or "")
            if not rid or not hid: raise ValueError(f"missing id: {path}")
            row={
                "horse_id":hid,
                "predicted_rank":int(x["predicted_rank"]),
                "probability":finite(x.get("race_normalized_win_probability")),
            }
            races[rid].append(row)
    for rid,rows in races.items():
        rows.sort(key=lambda z:z["predicted_rank"])
        ranks=[x["predicted_rank"] for x in rows]
        if ranks!=list(range(1,len(rows)+1)):
            raise ValueError(f"{path} race={rid}: ranks must be 1..N")
        if len({x["horse_id"] for x in rows})!=len(rows):
            raise ValueError(f"{path} race={rid}: duplicate horse")
    return races

def assert_no_forbidden(obj,path="root"):
    if isinstance(obj,dict):
        for k,v in obj.items():
            low=str(k).lower()
            if any(tok in low for tok in FORBIDDEN_TOKENS):
                raise ValueError(f"forbidden field {k} at {path}")
            assert_no_forbidden(v,path+"."+str(k))
    elif isinstance(obj,list):
        for i,v in enumerate(obj): assert_no_forbidden(v,path+f"[{i}]")

def main():
    a=parse_args()
    specs=[]
    for s in a.expert:
        if "=" not in s: raise ValueError("--expert must be alias=path")
        alias,path=s.split("=",1)
        specs.append((alias.strip(),path))
    if len(specs)!=7: raise ValueError(f"L1.6 requires seven experts, got {len(specs)}")
    if len({x[0] for x in specs})!=7: raise ValueError("duplicate expert alias")

    data={alias:read_expert(path) for alias,path in specs}
    race_sets=[set(x) for x in data.values()]
    if any(s!=race_sets[0] for s in race_sets[1:]):
        raise ValueError("expert race coverage differs")

    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    writer=gzip.open if str(out).endswith(".gz") else open
    race_count=horse_count=0

    with writer(out,"wt",encoding="utf-8") as fh:
        for rid in sorted(race_sets[0]):
            per_alias={alias:data[alias][rid] for alias,_ in specs}
            horse_sets=[{x["horse_id"] for x in rows} for rows in per_alias.values()]
            if any(s!=horse_sets[0] for s in horse_sets[1:]):
                raise ValueError(f"{rid}: cross-expert horse coverage differs")
            horses=sorted(horse_sets[0]); n=len(horses)
            if n<1: raise ValueError(f"{rid}: empty field")
            by_alias={
                alias:{x["horse_id"]:x for x in rows}
                for alias,rows in per_alias.items()
            }
            scored=[]
            for hid in horses:
                ranks={alias:int(by_alias[alias][hid]["predicted_rank"]) for alias,_ in specs}
                probs=[by_alias[alias][hid]["probability"] for alias,_ in specs]
                probs=[x for x in probs if x is not None]
                rank_vals=list(ranks.values())
                borda=sum(n+1-r for r in rank_vals)
                rec={
                    "horse_id":hid,
                    "borda_score":float(borda),
                    "normalized_borda":float(borda)/(7.0*n),
                    "mean_rank":sum(rank_vals)/7.0,
                    "rank_std":statistics.pstdev(rank_vals),
                    "best_rank":min(rank_vals),
                    "worst_rank":max(rank_vals),
                    "top1_votes":sum(r==1 for r in rank_vals),
                    "probability_mean":sum(probs)/len(probs) if probs else None,
                    "probability_std":statistics.pstdev(probs) if len(probs)>=2 else 0.0 if len(probs)==1 else None,
                    "expert_ranks":ranks,
                }
                scored.append(rec)
            scored.sort(key=lambda x:(
                -x["borda_score"],
                -x["top1_votes"],
                x["best_rank"],
                x["mean_rank"],
                -(x["probability_mean"] if x["probability_mean"] is not None else -1.0),
                x["horse_id"],
            ))
            for i,x in enumerate(scored,1): x["consensus_rank"]=i
            if [x["consensus_rank"] for x in scored]!=list(range(1,n+1)):
                raise ValueError(f"{rid}: consensus ranks not contiguous")
            record={
                "contract":CONTRACT,
                "version":"1.0.0",
                "race_id":rid,
                "field_size":n,
                "expert_count":7,
                "candidate_policy":"ALL_HORSES",
                "horses":scored,
            }
            assert_no_forbidden(record)
            fh.write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n")
            race_count+=1; horse_count+=n
    print("L16_FULL_RANK_CONSENSUS_READY")
    print(json.dumps({"races":race_count,"horses":horse_count,"output":str(out)},separators=(",",":")))

if __name__=="__main__": main()
