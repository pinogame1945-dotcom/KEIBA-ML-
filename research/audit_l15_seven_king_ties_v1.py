#!/usr/bin/env python3
import argparse,csv,gzip,json,math
from collections import Counter,defaultdict
from pathlib import Path

TOPNS=(1,3,6)

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def finite(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None

def read_winners(path):
    winners=defaultdict(set)
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            hid=str(row.get("horse_id") or "")
            t=row.get("target") or {}
            win=t.get("is_win")
            if win is None:
                try: win=int(float(t.get("finish_position")))==1
                except (TypeError,ValueError): win=False
            if win and rid and hid:
                winners[rid].add(hid)
    if not winners: raise ValueError("snapshot winners empty")
    return dict(winners)

def read_scores(path):
    by=defaultdict(list)
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            hid=str(row.get("horse_id") or "")
            score=finite(row.get("raw_margin_logit"))
            if score is None: score=finite(row.get("raw_win_probability"))
            rank=row.get("predicted_rank")
            if not rid or not hid or score is None or rank is None:
                raise ValueError(f"bad score row in {path}")
            by[rid].append({"horse_id":hid,"score":score,"old_rank":int(rank)})
    for rid,rows in by.items():
        ranks=sorted(r["old_rank"] for r in rows)
        if ranks!=list(range(1,len(rows)+1)):
            raise ValueError(f"{path} {rid}: non-contiguous old ranks")
        if len({r["horse_id"] for r in rows})!=len(rows):
            raise ValueError(f"{path} {rid}: duplicate horse")
    return by

def safe_rows(rows):
    ordered=sorted(rows,key=lambda r:(-r["score"],r["horse_id"]))
    return [{**r,"safe_rank":i+1} for i,r in enumerate(ordered)]

def tie_info(rows,n):
    if len(rows)<n: return False
    old=sorted(rows,key=lambda r:r["old_rank"])
    boundary=old[n-1]["score"]
    same=[r for r in rows if r["score"]==boundary]
    old_ranks=sorted(r["old_rank"] for r in same)
    return len(same)>1 and min(old_ranks)<=n<max(old_ranks)

def top_set(rows,n,key):
    return {r["horse_id"] for r in rows if r[key]<=n}

def consensus_order(per_expert,key):
    stats=defaultdict(lambda:{"support":0,"borda":0.0,"top1_votes":0,"best_rank":99,"rank_sum":0.0})
    for rows in per_expert.values():
        for r in rows:
            rank=int(r[key])
            if rank>6: continue
            s=stats[r["horse_id"]]
            s["support"]+=1
            s["borda"]+=float(7-rank)
            s["top1_votes"]+=int(rank==1)
            s["best_rank"]=min(s["best_rank"],rank)
            s["rank_sum"]+=rank
    for s in stats.values():
        s["mean_rank"]=s["rank_sum"]/s["support"]
    return sorted(stats,key=lambda hid:(-stats[hid]["borda"],-stats[hid]["support"],-stats[hid]["top1_votes"],stats[hid]["best_rank"],stats[hid]["mean_rank"],hid))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--year",required=True,type=int)
    ap.add_argument("--snapshot",required=True)
    ap.add_argument("--score",action="append",required=True,help="alias=path")
    ap.add_argument("--out-dir",required=True)
    a=ap.parse_args()
    if a.year==2026: raise SystemExit("2026 sealed")

    score_paths={}
    for spec in a.score:
        alias,path=spec.split("=",1)
        score_paths[alias]=path
    if len(score_paths)!=7: raise SystemExit(f"expected 7 kings, got {len(score_paths)}")

    winners=read_winners(a.snapshot)
    data={alias:read_scores(path) for alias,path in score_paths.items()}
    race_sets=[set(v) for v in data.values()]
    if any(s!=race_sets[0] for s in race_sets[1:]): raise SystemExit("king race coverage mismatch")
    races=sorted(race_sets[0])
    missing=[r for r in races if r not in winners]
    if missing: raise SystemExit(f"winner missing for {len(missing)} races")

    summary={"contract":"L15_SEVEN_KING_TIE_AUDIT_V1","year":a.year,"races":len(races),"kings":{},"seven":{}}
    changed=[]
    safe_by_alias={alias:{} for alias in data}

    for alias,by in data.items():
        st={"races":len(races),"any_tie_races":0,"topn":{}}
        for n in TOPNS:
            st["topn"][str(n)]={"boundary_tie_races":0,"topn_set_changed_races":0,"old_capture":0,"safe_capture":0,"old_only_hits":0,"safe_only_hits":0}
        for rid in races:
            rows=by[rid]
            sr=safe_rows(rows)
            safe_by_alias[alias][rid]=sr
            counts=Counter(r["score"] for r in rows)
            if any(v>1 for v in counts.values()): st["any_tie_races"]+=1
            winner=winners[rid]
            for n in TOPNS:
                x=st["topn"][str(n)]
                bt=tie_info(rows,n)
                if bt: x["boundary_tie_races"]+=1
                old=top_set(rows,n,"old_rank"); new=top_set(sr,n,"safe_rank")
                oh=bool(winner & old); nh=bool(winner & new)
                x["old_capture"]+=int(oh); x["safe_capture"]+=int(nh)
                if old!=new:
                    x["topn_set_changed_races"]+=1
                    changed.append({"year":a.year,"race_id":rid,"scope":alias,"topn":n,"winner":"|".join(sorted(winner)),
                                    "old_top":"|".join(sorted(old)),"safe_top":"|".join(sorted(new)),
                                    "old_hit":int(oh),"safe_hit":int(nh),"boundary_tie":int(bt)})
                x["old_only_hits"]+=int(oh and not nh)
                x["safe_only_hits"]+=int(nh and not oh)
        for n in TOPNS:
            x=st["topn"][str(n)]
            for k in ("boundary_tie_races","topn_set_changed_races","old_capture","safe_capture","old_only_hits","safe_only_hits"):
                x[k+"_rate"]=x[k]/len(races)
            x["capture_delta"]=x["safe_capture"]-x["old_capture"]
        st["any_tie_rate"]=st["any_tie_races"]/len(races)
        summary["kings"][alias]=st

    seven={"races":len(races),"top6_union_changed_races":0,"old_blind":0,"safe_blind":0,
           "old_blind_to_safe_hit":0,"old_hit_to_safe_blind":0,
           "consensus_top1_changed_races":0,"consensus_anchor_top2_changed_races":0}
    for rid in races:
        old_rows={a:data[a][rid] for a in data}
        safe_rows_map={a:safe_by_alias[a][rid] for a in data}
        old_union=set().union(*(top_set(rows,6,"old_rank") for rows in old_rows.values()))
        safe_union=set().union(*(top_set(rows,6,"safe_rank") for rows in safe_rows_map.values()))
        winner=winners[rid]
        ob=not bool(winner & old_union); sb=not bool(winner & safe_union)
        seven["old_blind"]+=int(ob); seven["safe_blind"]+=int(sb)
        seven["old_blind_to_safe_hit"]+=int(ob and not sb)
        seven["old_hit_to_safe_blind"]+=int((not ob) and sb)
        old_cons=consensus_order(old_rows,"old_rank")
        safe_cons=consensus_order(safe_rows_map,"safe_rank")
        top1_changed=(old_cons[:1]!=safe_cons[:1])
        anchor_changed=(set(old_cons[:2])!=set(safe_cons[:2]))
        seven["consensus_top1_changed_races"]+=int(top1_changed)
        seven["consensus_anchor_top2_changed_races"]+=int(anchor_changed)
        if old_union!=safe_union:
            seven["top6_union_changed_races"]+=1
            changed.append({"year":a.year,"race_id":rid,"scope":"SEVEN_UNION","topn":6,"winner":"|".join(sorted(winner)),
                            "old_top":"|".join(sorted(old_union)),"safe_top":"|".join(sorted(safe_union)),
                            "old_hit":int(not ob),"safe_hit":int(not sb),"boundary_tie":1})
    for k in list(seven):
        if k!="races":
            seven[k+"_rate"]=seven[k]/len(races)
    seven["blind_delta"]=seven["safe_blind"]-seven["old_blind"]
    summary["seven"]=seven

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    fields=["year","race_id","scope","topn","winner","old_top","safe_top","old_hit","safe_hit","boundary_tie"]
    with open(out/"changed-races.csv","w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(changed)
    print("L15_SEVEN_KING_TIE_AUDIT_OK")
    print(json.dumps({"year":a.year,"races":len(races),"changed_rows":len(changed),"seven":seven},ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
