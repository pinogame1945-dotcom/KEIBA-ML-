#!/usr/bin/env python3
import argparse,csv,gzip,itertools,json
from collections import defaultdict
from pathlib import Path

CANDS=[
    "outsider_daytrend","outsider_raceshape","outsider_gatecourse",
    "outsider_field","outsider_jockey"
]
BUCKETS=("4-6","7-10","11+","unranked")

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--consensus",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--outsider-csv",required=True)
    p.add_argument("--year",type=int,required=True)
    p.add_argument("--output",required=True)
    return p.parse_args()

def open_text(p):
    return gzip.open(p,"rt",encoding="utf-8") if str(p).endswith(".gz") else open(p,"rt",encoding="utf-8")

def load_truth(path):
    finish=defaultdict(lambda:defaultdict(list))
    with open_text(path) as f:
        for line in f:
            if not line.strip(): continue
            x=json.loads(line)
            rid=str(x.get("race_id") or ""); hid=str(x.get("horse_id") or "")
            if not rid or not hid: continue
            try: pos=int(float((x.get("target") or {}).get("finish_position")))
            except (TypeError,ValueError): continue
            if pos<=3: finish[rid][pos].append(hid)
    truth={}
    for rid,g in finish.items():
        pieces=[()]
        for rank in sorted(g):
            hs=list(g[rank]); occupied=min(len(hs),3-rank+1)
            perms=list(itertools.permutations(hs,occupied))
            pieces=[a+b for a in pieces for b in perms]
        outcomes=sorted(set(x for x in pieces if len(x)==3))
        if outcomes:
            truth[rid]={
                "outcomes":outcomes,
                "podium_horses":set(h for hs in g.values() for h in hs)
            }
    return truth

def load_consensus(path):
    out={}
    with open_text(path) as f:
        for line in f:
            if not line.strip(): continue
            x=json.loads(line)
            if x.get("contract")!="L16_FULL_RANK_CONSENSUS_V1":
                raise ValueError("bad L1.6 contract")
            rid=str(x["race_id"])
            rows=sorted(x["horses"],key=lambda z:int(z["consensus_rank"]))
            order=[str(z["horse_id"]) for z in rows]
            ranks={str(z["horse_id"]):int(z["consensus_rank"]) for z in rows}
            out[rid]={"order":order,"ranks":ranks}
    return out

def load_outsiders(path):
    out=defaultdict(dict)
    with open(path,newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rid=str(r["race_id"]); c=r["candidate"]
            out[rid][c]={
                "top3":set(x for x in r["top3_horse_ids"].split("|") if x),
                "top6":set(x for x in r["top6_horse_ids"].split("|") if x),
            }
    return out

def full_hit(sel,outcomes):
    return any(set(o).issubset(sel) for o in outcomes)

def bucket(rank):
    if rank is None: return "unranked"
    if 4<=rank<=6: return "4-6"
    if 7<=rank<=10: return "7-10"
    if rank>=11: return "11+"
    return None

def blank_stats():
    return {
      "top3_rescued_occurrences":0,
      "top6_rescued_occurrences":0,
      "novel_top3_selections":0,
      "novel_top3_true_podium":0,
      "novel_top6_selections":0,
      "novel_top6_true_podium":0,
      "race_full_rescues_top3":0,
      "race_full_rescues_top6":0,
      "full_hits_after_top3":0,
      "full_hits_after_top6":0,
      "by_l16_rank_bucket":{b:{"top3_rescued":0,"top6_rescued":0} for b in BUCKETS},
    }

def main():
    a=args()
    if a.year==2026: raise ValueError("2026 locked")
    t=load_truth(a.snapshot); c=load_consensus(a.consensus); o=load_outsiders(a.outsider_csv)
    if set(c)!=set(t):
        raise ValueError(f"truth/consensus coverage mismatch consensus={len(c)} truth={len(t)}")
    if len(c)!=3456: raise ValueError(f"expected 3456 races, got {len(c)}")
    for rid in c:
        if set(o.get(rid,{}))!=set(CANDS):
            raise ValueError(f"{rid}: outsider coverage mismatch {set(o.get(rid,{}))}")

    denom_buckets={b:0 for b in BUCKETS}
    missed_occ=0
    races_missed=0
    l16_top3_hits=0
    stats={cand:blank_stats() for cand in CANDS}
    stats["ALL5_UNION"]=blank_stats()
    all5_top3_size_total=0
    all5_top6_size_total=0
    l16_plus_all5_top3_size_total=0
    l16_plus_all5_top6_size_total=0

    for rid in sorted(c):
        order=c[rid]["order"]; ranks=c[rid]["ranks"]
        top3=set(order[:3])
        outcomes=t[rid]["outcomes"]; podium=t[rid]["podium_horses"]
        base_hit=full_hit(top3,outcomes)
        l16_top3_hits+=int(base_hit)
        if not base_hit: races_missed+=1

        missed=podium-top3
        missed_occ+=len(missed)
        for h in missed:
            b=bucket(ranks.get(h))
            if b: denom_buckets[b]+=1

        union3=set(); union6=set()
        for cand in CANDS:
            union3 |= o[rid][cand]["top3"]
            union6 |= o[rid][cand]["top6"]

        all5_top3_size_total += len(union3)
        all5_top6_size_total += len(union6)
        l16_plus_all5_top3_size_total += len(top3|union3)
        l16_plus_all5_top6_size_total += len(top3|union6)

        candidates={cand:(o[rid][cand]["top3"],o[rid][cand]["top6"]) for cand in CANDS}
        candidates["ALL5_UNION"]=(union3,union6)
        for cand,(s3,s6) in candidates.items():
            st=stats[cand]
            r3=missed&s3; r6=missed&s6
            st["top3_rescued_occurrences"]+=len(r3)
            st["top6_rescued_occurrences"]+=len(r6)
            n3=s3-top3; n6=s6-top3
            st["novel_top3_selections"]+=len(n3)
            st["novel_top3_true_podium"]+=len(n3&podium)
            st["novel_top6_selections"]+=len(n6)
            st["novel_top6_true_podium"]+=len(n6&podium)
            h3=full_hit(top3|s3,outcomes)
            h6=full_hit(top3|s6,outcomes)
            st["full_hits_after_top3"]+=int(h3)
            st["full_hits_after_top6"]+=int(h6)
            if not base_hit:
                st["race_full_rescues_top3"]+=int(h3)
                st["race_full_rescues_top6"]+=int(h6)
            for h in r3:
                b=bucket(ranks.get(h))
                if b: st["by_l16_rank_bucket"][b]["top3_rescued"]+=1
            for h in r6:
                b=bucket(ranks.get(h))
                if b: st["by_l16_rank_bucket"][b]["top6_rescued"]+=1

    for cand,st in stats.items():
        st["top3_rescue_rate_on_l16_missed_podium_occurrences"]=st["top3_rescued_occurrences"]/missed_occ if missed_occ else None
        st["top6_rescue_rate_on_l16_missed_podium_occurrences"]=st["top6_rescued_occurrences"]/missed_occ if missed_occ else None
        st["novel_top3_precision_for_actual_podium"]=st["novel_top3_true_podium"]/st["novel_top3_selections"] if st["novel_top3_selections"] else None
        st["novel_top6_precision_for_actual_podium"]=st["novel_top6_true_podium"]/st["novel_top6_selections"] if st["novel_top6_selections"] else None
        st["race_full_rescue_rate_on_l16_top3_failures_top3"]=st["race_full_rescues_top3"]/races_missed if races_missed else None
        st["race_full_rescue_rate_on_l16_top3_failures_top6"]=st["race_full_rescues_top6"]/races_missed if races_missed else None
        st["full_podium_rate_after_l16_top3_plus_outsider_top3"]=st["full_hits_after_top3"]/len(c)
        st["full_podium_rate_after_l16_top3_plus_outsider_top6"]=st["full_hits_after_top6"]/len(c)
        for b,d in st["by_l16_rank_bucket"].items():
            d["denominator"]=denom_buckets[b]
            d["top3_rescue_rate"]=d["top3_rescued"]/denom_buckets[b] if denom_buckets[b] else None
            d["top6_rescue_rate"]=d["top6_rescued"]/denom_buckets[b] if denom_buckets[b] else None

    result={
      "contract":"L16_OUTSIDER_RESCUE_MATRIX_V1",
      "year":a.year,
      "locked_years":[2026],
      "races":len(c),
      "l16_top3_full_hits":l16_top3_hits,
      "l16_top3_full_rate":l16_top3_hits/len(c),
      "l16_top3_failure_races":races_missed,
      "l16_top3_missed_podium_horse_occurrences":missed_occ,
      "missed_podium_l16_rank_buckets":denom_buckets,
      "candidates":stats,
      "selection_size":{
        "all5_top3_union_avg":all5_top3_size_total/len(c),
        "all5_top6_union_avg":all5_top6_size_total/len(c),
        "l16_top3_plus_all5_top3_avg":l16_plus_all5_top3_size_total/len(c),
        "l16_top3_plus_all5_top6_avg":l16_plus_all5_top6_size_total/len(c),
      }
    }
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__": main()
