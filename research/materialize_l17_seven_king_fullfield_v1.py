#!/usr/bin/env python3
import argparse,gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path

EXPECTED_EXPERTS=7
FORBIDDEN=("odds","popularity","payout","roi","return_yen","profit_yen","finish_position","target")

def open_text(path,mode="rt"):
    return gzip.open(path,mode,encoding="utf-8") if str(path).endswith(".gz") else open(path,mode,encoding="utf-8")

def finite(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None

def horse_number_key(v):
    try: return (0,int(v))
    except (TypeError,ValueError): return (1,10**9)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--year",required=True,type=int)
    p.add_argument("--expert",action="append",required=True,help="name=score_path")
    p.add_argument("--output",required=True)
    return p.parse_args()

def load_expert(name,path):
    races=defaultdict(list)
    with open_text(path) as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            if r.get("contract")!="L1_TO_L2_OUTPUT_CONTRACT_V1":
                raise ValueError(f"{name}: bad input contract")
            rid=str(r.get("race_id") or "")
            hid=str(r.get("horse_id") or "")
            if not rid or not hid: raise ValueError(f"{name}: missing race_id/horse_id")
            score=finite(r.get("raw_margin_logit"))
            if score is None:
                p=finite(r.get("raw_win_probability"))
                if p is None: raise ValueError(f"{name}: missing score")
                score=p
            prob=finite(r.get("race_normalized_win_probability"))
            if prob is None: raise ValueError(f"{name}: missing normalized probability")
            races[rid].append({
                "horse_id":hid,
                "horse_number":r.get("horse_number"),
                "score":score,
                "probability":prob
            })
    for rid,rows in races.items():
        if len({x["horse_id"] for x in rows})!=len(rows):
            raise ValueError(f"{name} {rid}: duplicate horse")
        rows.sort(key=lambda x:(-x["score"],horse_number_key(x["horse_number"]),x["horse_id"]))
        for i,x in enumerate(rows,1): x["rank"]=i
    return races

def stdev(xs):
    return statistics.pstdev(xs) if len(xs)>1 else 0.0

def main():
    a=parse_args()
    if a.year==2026: raise SystemExit("2026 sealed")
    specs={}
    for spec in a.expert:
        name,path=spec.split("=",1)
        specs[name]=path
    if len(specs)!=EXPECTED_EXPERTS:
        raise SystemExit(f"expected {EXPECTED_EXPERTS} experts, got {len(specs)}")
    expert_races={name:load_expert(name,path) for name,path in specs.items()}
    race_sets=[set(v) for v in expert_races.values()]
    if any(s!=race_sets[0] for s in race_sets[1:]):
        raise SystemExit("expert race coverage mismatch")

    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    write=gzip.open if str(out).endswith(".gz") else open
    rows_written=0
    with write(out,"wt",encoding="utf-8") as fh:
        for rid in sorted(race_sets[0]):
            per_horse=defaultdict(dict)
            horse_nums={}
            field=None
            for name,races in expert_races.items():
                rows=races[rid]
                ids={x["horse_id"] for x in rows}
                if field is None: field=ids
                elif ids!=field: raise ValueError(f"{rid}: cross-expert horse coverage mismatch")
                for x in rows:
                    hid=x["horse_id"]
                    horse_nums.setdefault(hid,x.get("horse_number"))
                    per_horse[hid][name]=x
            horse_rows=[]
            for hid in sorted(field):
                views=per_horse[hid]
                ranks=[int(x["rank"]) for x in views.values()]
                probs=[float(x["probability"]) for x in views.values()]
                row={
                    "horse_id":hid,
                    "horse_number":horse_nums.get(hid),
                    "mean_rank":sum(ranks)/len(ranks),
                    "rank_std":stdev(ranks),
                    "best_rank":min(ranks),
                    "worst_rank":max(ranks),
                    "top1_votes":sum(r==1 for r in ranks),
                    "top3_support":sum(r<=3 for r in ranks),
                    "top6_support":sum(r<=6 for r in ranks),
                    "mean_probability":sum(probs)/len(probs),
                    "probability_std":stdev(probs),
                    "experts":{
                        name:{
                            "rank":int(views[name]["rank"]),
                            "probability":float(views[name]["probability"])
                        } for name in sorted(views)
                    }
                }
                horse_rows.append(row)
            horse_rows.sort(key=lambda x:(
                x["mean_rank"],
                -x["top6_support"],
                -x["top3_support"],
                -x["top1_votes"],
                -x["mean_probability"],
                horse_number_key(x["horse_number"]),
                x["horse_id"],
            ))
            for i,x in enumerate(horse_rows,1): x["consensus_rank"]=i
            rec={
                "contract":"L17_SEVEN_KING_FULLFIELD_OUTPUT_V1",
                "l17_version":"1.0.0",
                "year":a.year,
                "race_id":rid,
                "field_size":len(horse_rows),
                "expert_count":EXPECTED_EXPERTS,
                "consensus_order":[x["horse_id"] for x in horse_rows],
                "horses":horse_rows,
            }
            raw=json.dumps(rec,ensure_ascii=False,separators=(",",":"))
            low=raw.lower()
            for token in FORBIDDEN:
                if f'"{token}"' in low:
                    raise ValueError(f"forbidden output token: {token}")
            fh.write(raw+"\n")
            rows_written+=1
    print("L17_SEVEN_KING_FULLFIELD_READY")
    print(json.dumps({"year":a.year,"races":rows_written,"experts":sorted(specs),"output":str(out)},ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
