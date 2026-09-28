#!/usr/bin/env python3
import argparse, gzip, json, math, statistics
from collections import defaultdict
from pathlib import Path

TOP_NS=(1,3,6)

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--candidate-score",required=True)
    p.add_argument("--router-seven",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--candidate-name",required=True)
    p.add_argument("--year",required=True,type=int)
    p.add_argument("--output",required=True)
    return p.parse_args()

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def load_truth(path):
    wins=defaultdict(set)
    seen=defaultdict(set)
    with open_text(path) as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            rid=str(r.get("race_id") or "")
            hid=str(r.get("horse_id") or "")
            if not rid or not hid: continue
            seen[rid].add(hid)
            t=r.get("target") or {}
            is_win=t.get("is_win")
            finish=t.get("finish_position")
            win=bool(is_win)
            if is_win is None:
                try: win=int(float(finish))==1
                except (TypeError,ValueError): win=False
            if win: wins[rid].add(hid)
    return {rid:wins[rid] for rid in seen if wins[rid]}

def load_candidate(path):
    races=defaultdict(list)
    with open_text(path) as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            if r.get("contract")!="L1_TO_L2_OUTPUT_CONTRACT_V1":
                raise ValueError("bad candidate score contract")
            races[str(r["race_id"])].append(r)
    for rid,rows in races.items():
        rows.sort(key=lambda x:int(x["predicted_rank"]))
        ranks=[int(x["predicted_rank"]) for x in rows]
        if ranks!=list(range(1,len(rows)+1)):
            raise ValueError(f"{rid}: candidate ranks not contiguous")
    return races

def load_router(path):
    out={}
    with open_text(path) as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            if r.get("contract")!="ROUTER_FEATURE_SNAPSHOT_V1":
                raise ValueError("bad router contract")
            out[str(r["race_id"])]=r
    return out

def hit(winners,horses):
    return bool(set(winners).intersection(horses))

def jac(a,b):
    a=set(a); b=set(b); u=a|b
    return len(a&b)/len(u) if u else 1.0

def expert_set(view,n):
    if n==1:
        x=view.get("top1_horse_id")
        return {str(x)} if x else set()
    return set(map(str,view.get(f"top{n}_horse_ids") or []))

def mean(xs):
    return sum(xs)/len(xs) if xs else None

def main():
    a=args()
    if a.year>=2026: raise SystemExit("2026 is sealed")
    truth=load_truth(a.snapshot)
    cand=load_candidate(a.candidate_score)
    router=load_router(a.router_seven)
    common=set(truth)&set(cand)&set(router)
    if not common:
        raise SystemExit("no common races")
    if common!=set(cand) or common!=set(router):
        raise SystemExit(f"race coverage mismatch truth={len(truth)} cand={len(cand)} router={len(router)} common={len(common)}")

    counts={n:defaultdict(float) for n in TOP_NS}
    pair=defaultdict(lambda:{n:defaultdict(float) for n in TOP_NS})
    rescue_race_ids={n:[] for n in TOP_NS}
    blind_top6_ranks=[]
    top1_novel=top1_novel_hits=0
    experts=None

    for rid in sorted(common):
        winners=truth[rid]
        rows=cand[rid]
        rank_by_horse={str(r["horse_id"]):int(r["predicted_rank"]) for r in rows}
        rec=router[rid]
        views=rec.get("experts") or {}
        if experts is None: experts=sorted(views)
        if sorted(views)!=experts:
            raise SystemExit(f"{rid}: expert set changed")
        csets={n:set(str(r["horse_id"]) for r in rows[:n]) for n in TOP_NS}
        unions={}
        for n in TOP_NS:
            unions[n]=set().union(*(expert_set(views[e],n) for e in experts))
            d=counts[n]
            d["races"]+=1
            ch=int(hit(winners,csets[n])); uh=int(hit(winners,unions[n]))
            d["candidate_hits"]+=ch
            d["seven_union_hits"]+=uh
            d["seven_blind_spots"]+=int(not uh)
            rescued=bool((not uh) and ch)
            d["rescues"]+=int(rescued)
            if rescued:
                rescue_race_ids[n].append(rid)
            d["duplicate_hits"]+=int(uh and ch)
            d["candidate_union_jaccard_sum"]+=jac(csets[n],unions[n])
            for e in experts:
                eset=expert_set(views[e],n)
                q=pair[e][n]
                q["races"]+=1
                q["jaccard_sum"]+=jac(csets[n],eset)
                eh=int(hit(winners,eset))
                q["candidate_only_hits"]+=int(ch and not eh)
                q["expert_only_hits"]+=int(eh and not ch)
                if n==1:
                    q["top1_agree"]+=int(csets[n]==eset)
        seven_top1={str(views[e].get("top1_horse_id")) for e in experts if views[e].get("top1_horse_id")}
        ctop1=next(iter(csets[1])) if csets[1] else None
        novel=int(ctop1 is not None and ctop1 not in seven_top1)
        top1_novel+=novel
        top1_novel_hits+=int(novel and hit(winners,csets[1]))
        if not hit(winners,unions[6]):
            wranks=[rank_by_horse[w] for w in winners if w in rank_by_horse]
            if wranks: blind_top6_ranks.append(min(wranks))

    topn={}
    for n in TOP_NS:
        d=counts[n]; races=int(d["races"]); blind=int(d["seven_blind_spots"]); hits=int(d["candidate_hits"])
        topn[str(n)]={
            "races":races,
            "candidate_hit_count":hits,
            "candidate_hit_rate":hits/races,
            "seven_union_hit_count":int(d["seven_union_hits"]),
            "seven_union_hit_rate":d["seven_union_hits"]/races,
            "seven_blind_spot_count":blind,
            "rescue_count":int(d["rescues"]),
            "rescue_rate_on_seven_blind_spots":(d["rescues"]/blind) if blind else None,
            "rescue_share_of_candidate_hits":(d["rescues"]/hits) if hits else None,
            "duplicate_hit_count":int(d["duplicate_hits"]),
            "mean_jaccard_vs_seven_union":d["candidate_union_jaccard_sum"]/races,
        }

    pairwise={}
    for e in experts or []:
        pairwise[e]={}
        for n in TOP_NS:
            q=pair[e][n]; races=int(q["races"])
            pairwise[e][str(n)]={
                "races":races,
                "mean_jaccard":q["jaccard_sum"]/races if races else None,
                "candidate_only_hit_count":int(q["candidate_only_hits"]),
                "expert_only_hit_count":int(q["expert_only_hits"]),
                "top1_agreement_rate":(q["top1_agree"]/races) if n==1 and races else None,
            }

    out={
        "contract":"L1_OUTSIDER_RESCUE_V1",
        "candidate":a.candidate_name,
        "validation_year":a.year,
        "races":len(common),
        "expert_count":len(experts or []),
        "topn":topn,
        "top1_novel_selection_count":top1_novel,
        "top1_novel_selection_rate":top1_novel/len(common),
        "top1_novel_selection_hit_rate":top1_novel_hits/top1_novel if top1_novel else None,
        "seven_top6_blind_spot_candidate_winner_rank":{
            "count":len(blind_top6_ranks),
            "mean":mean(blind_top6_ranks),
            "median":statistics.median(blind_top6_ranks) if blind_top6_ranks else None,
            "top10_rate":sum(x<=10 for x in blind_top6_ranks)/len(blind_top6_ranks) if blind_top6_ranks else None,
        },
        "pairwise":pairwise,
        "rescue_race_ids_by_topn":{
            str(n):sorted(rescue_race_ids[n]) for n in TOP_NS
        },
        "notes":[
            "Ability model uses no odds.",
            "Rescue means all seven existing kings miss the winner at the same TopN while the outsider captures it.",
            "2026 is not read."
        ]
    }
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L1_OUTSIDER_RESCUE_OK")
    print(json.dumps(out,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
