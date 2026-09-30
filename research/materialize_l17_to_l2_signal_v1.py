#!/usr/bin/env python3
import argparse,csv,gzip,json,math
from collections import defaultdict
from pathlib import Path

CONTRACT="L17_TO_L2_SIGNAL_V1"
FORBIDDEN_KEYS=("odds","popularity","payout","return_yen","profit_yen","roi","finish","target","y3")

def opent(path,mode="rt"):
    return gzip.open(path,mode,encoding="utf-8") if str(path).endswith(".gz") else open(path,mode,encoding="utf-8")

def finite_or_none(v):
    if v in (None,""): return None
    x=float(v)
    if not math.isfinite(x): raise ValueError(f"non-finite value: {v}")
    return x

def load_consensus(path):
    out={}
    with opent(path) as fh:
        for line in fh:
            if not line.strip(): continue
            x=json.loads(line)
            rid=str(x["race_id"])
            if rid in out: raise ValueError(f"duplicate consensus race {rid}")
            rows=sorted(x["horses"],key=lambda z:int(z["consensus_rank"]))
            ranks=[int(z["consensus_rank"]) for z in rows]
            if ranks!=list(range(1,len(rows)+1)): raise ValueError(f"{rid}: non-contiguous king ranks")
            out[rid]=rows
    return out

def load_ballots(path):
    sig=defaultdict(lambda:defaultdict(lambda:{"score":0,"count":0,"v1":0,"v2":0,"v3":0}))
    candidates=defaultdict(set)
    with opent(path) as fh:
        for row in csv.DictReader(fh):
            rid=str(row["race_id"]); cand=str(row["candidate"])
            if cand in candidates[rid]: raise ValueError(f"{rid}: duplicate outsider candidate {cand}")
            candidates[rid].add(cand)
            for k,w in ((1,3),(2,2),(3,1)):
                hid=str(row.get(f"top{k}_horse_id") or "")
                if not hid: continue
                d=sig[rid][hid]; d["score"]+=w; d["count"]+=1; d[f"v{k}"]+=1
    for rid,c in candidates.items():
        if len(c)!=13: raise ValueError(f"{rid}: expected 13 outsider candidates, got {len(c)}")
    return sig,candidates

def load_predictions(path,year):
    out={}
    if not path: return out
    with opent(path) as fh:
        for row in csv.DictReader(fh):
            if int(row["year"])!=year: continue
            key=(str(row["race_id"]),str(row["horse_id"]))
            if key in out: raise ValueError(f"duplicate prediction {key}")
            out[key]={
              "rank":int(row["rank"]),
              "score":int(row["score"]),
              "p3":finite_or_none(row["p3"]),
              "base":finite_or_none(row["p3_rank_baseline"]),
              "delta":finite_or_none(row["p3_delta"]),
            }
    return out

def assert_no_forbidden_keys(obj,path="root"):
    if isinstance(obj,dict):
        for k,v in obj.items():
            low=str(k).lower()
            if any(tok in low for tok in FORBIDDEN_KEYS):
                raise ValueError(f"forbidden key {k} at {path}")
            assert_no_forbidden_keys(v,path+"."+str(k))
    elif isinstance(obj,list):
        for i,v in enumerate(obj): assert_no_forbidden_keys(v,path+f"[{i}]")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--year",type=int,required=True)
    p.add_argument("--consensus",required=True)
    p.add_argument("--ballots",required=True)
    p.add_argument("--predictions")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    if a.year>=2026: raise ValueError("2026 sealed")

    consensus=load_consensus(a.consensus)
    sig,candidates=load_ballots(a.ballots)
    pred=load_predictions(a.predictions,a.year)
    if set(consensus)!=set(candidates): raise ValueError("consensus/ballot race coverage mismatch")

    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    races=horses=calibrated=0
    with opent(out,"wt") as fh:
        for rid in sorted(consensus):
            rows=consensus[rid]
            known={str(x["horse_id"]) for x in rows}
            extra=set(sig[rid])-known
            if extra: raise ValueError(f"{rid}: outsider horses absent from Seven consensus: {sorted(extra)[:3]}")
            hs=[]
            for x in rows:
                hid=str(x["horse_id"]); kr=int(x["consensus_rank"])
                s=sig[rid].get(hid,{"score":0,"count":0,"v1":0,"v2":0,"v3":0})
                if not (0<=s["score"]<=39 and 0<=s["count"]<=13):
                    raise ValueError(f"{rid}/{hid}: invalid outsider signal")
                if s["score"]!=3*s["v1"]+2*s["v2"]+s["v3"] or s["count"]!=s["v1"]+s["v2"]+s["v3"]:
                    raise ValueError(f"{rid}/{hid}: outsider vote arithmetic mismatch")
                pr=pred.get((rid,hid))
                if a.year>=2022 and kr<=10:
                    if pr is None: raise ValueError(f"{rid}/{hid}: missing OOS p3 prediction")
                    if pr["rank"]!=kr or pr["score"]!=s["score"]:
                        raise ValueError(f"{rid}/{hid}: prediction input mismatch")
                    if not (0<=pr["p3"]<=1 and 0<=pr["base"]<=1 and -1<=pr["delta"]<=1):
                        raise ValueError(f"{rid}/{hid}: invalid calibrated probability")
                    if abs((pr["p3"]-pr["base"])-pr["delta"])>1e-9:
                        raise ValueError(f"{rid}/{hid}: p3 delta mismatch")
                    p3,base,delta,status=pr["p3"],pr["base"],pr["delta"],"OOS_WALK_FORWARD"
                    calibrated+=1
                elif kr<=10:
                    p3=base=delta=None; status="NO_PRIOR_TRAINING_YEAR"
                else:
                    if pr is not None: raise ValueError(f"{rid}/{hid}: prediction exists outside top10 scope")
                    p3=base=delta=None; status="OUTSIDE_CALIBRATION_SCOPE"
                rec={
                  "horse_id":hid,
                  "king_rank":kr,
                  "king_borda_score":float(x["borda_score"]),
                  "king_normalized_borda":float(x["normalized_borda"]),
                  "king_mean_rank":float(x["mean_rank"]),
                  "king_rank_std":float(x["rank_std"]),
                  "king_best_rank":int(x["best_rank"]),
                  "king_worst_rank":int(x["worst_rank"]),
                  "king_top1_votes":int(x["top1_votes"]),
                  "king_probability_mean":finite_or_none(x.get("probability_mean")),
                  "king_probability_std":finite_or_none(x.get("probability_std")),
                  "outsider_score":int(s["score"]),
                  "outsider_support_count":int(s["count"]),
                  "outsider_top1_votes":int(s["v1"]),
                  "outsider_top2_votes":int(s["v2"]),
                  "outsider_top3_votes":int(s["v3"]),
                  "p3_calibrated":p3,
                  "p3_rank_baseline":base,
                  "p3_delta":delta,
                  "calibration_status":status,
                }
                hs.append(rec)
            if [z["king_rank"] for z in hs]!=list(range(1,len(hs)+1)):
                raise ValueError(f"{rid}: king ranking mutated")
            record={
              "contract":CONTRACT,
              "version":"1.0.0",
              "year":a.year,
              "race_id":rid,
              "field_size":len(hs),
              "ranking_policy":"SEVEN_KING_UNCHANGED",
              "outsider_policy":"SIGNAL_ONLY_NO_RERANK",
              "calibration_scope":"KING_RANK_1_TO_10",
              "horses":hs,
            }
            assert_no_forbidden_keys(record)
            fh.write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n")
            races+=1; horses+=len(hs)
    # No unused top10 predictions are allowed.
    expected=sum(1 for rid,rows in consensus.items() for x in rows if int(x["consensus_rank"])<=10) if a.year>=2022 else 0
    if len(pred)!=expected:
        raise ValueError(f"prediction coverage mismatch: got={len(pred)} expected={expected}")
    print("L17_TO_L2_SIGNAL_READY")
    print(json.dumps({"year":a.year,"races":races,"horses":horses,"calibrated_horses":calibrated,
                      "ranking_policy":"SEVEN_KING_UNCHANGED","output":str(out)},separators=(",",":")))

if __name__=="__main__":
    main()
