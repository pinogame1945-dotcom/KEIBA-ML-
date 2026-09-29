#!/usr/bin/env python3
import argparse, gzip, itertools, json, csv
from collections import defaultdict
from pathlib import Path

YEARS=(2022,2023,2024,2025)
KING_ALIASES=("core4","pedlegacy","condition","full","light","pedv1","condrc")

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot-root",required=True)
    p.add_argument("--scores-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def open_text(p):
    return gzip.open(p,"rt",encoding="utf-8") if str(p).endswith(".gz") else open(p,"rt",encoding="utf-8")

def find_one(root,names):
    root=Path(root)
    for n in names:
        p=root/n
        if p.exists() and p.stat().st_size>0:return p
    raise FileNotFoundError((root,names))

def read_truth(path):
    groups=defaultdict(lambda:defaultdict(list))
    fields=defaultdict(set)
    with open_text(path) as f:
        for line in f:
            if not line.strip():continue
            x=json.loads(line); rid=str(x.get("race_id") or ""); hid=str(x.get("horse_id") or "")
            if not rid or not hid:continue
            fields[rid].add(hid)
            try: pos=int(float((x.get("target") or {}).get("finish_position")))
            except (TypeError,ValueError):continue
            if pos<=3:groups[rid][pos].append(hid)
    out={}
    for rid,g in groups.items():
        pieces=[()]
        for rank in sorted(g):
            if rank>3:continue
            hs=list(g[rank]); occupied=min(len(hs),3-rank+1)
            perms=list(itertools.permutations(hs,occupied))
            pieces=[a+b for a in pieces for b in perms]
        outcomes=sorted(set(x for x in pieces if len(x)==3))
        if outcomes:out[rid]={"outcomes":outcomes,"field_size":len(fields[rid])}
    return out

def read_scores(path):
    races=defaultdict(list)
    with open_text(path) as f:
        for line in f:
            if not line.strip():continue
            x=json.loads(line); rid=str(x["race_id"]); hid=str(x["horse_id"])
            races[rid].append((int(x["predicted_rank"]),hid))
    for rid in races:
        races[rid].sort()
        ranks=[r for r,_ in races[rid]]
        if ranks!=list(range(1,len(ranks)+1)):raise ValueError(f"{path} {rid}: noncontiguous ranks")
    return {rid:[h for _,h in rows] for rid,rows in races.items()}

def captures(ids,outcomes):
    s=set(ids)
    return any(set(o).issubset(s) for o in outcomes)

def consensus(orders,source_n,final_n):
    stats=defaultdict(lambda:{"support":0,"borda":0.0,"top1":0,"best":999,"sumrank":0.0})
    for order in orders.values():
        for rank,hid in enumerate(order[:source_n],1):
            s=stats[hid]; s["support"]+=1; s["borda"]+=float(source_n+1-rank); s["top1"]+=int(rank==1); s["best"]=min(s["best"],rank); s["sumrank"]+=rank
    for s in stats.values():s["mean"]=s["sumrank"]/s["support"]
    ranked=sorted(stats,key=lambda h:(-stats[h]["borda"],-stats[h]["support"],-stats[h]["top1"],stats[h]["best"],stats[h]["mean"],h))
    return ranked[:final_n], ranked

def pct(n,d): return n/d if d else None

def main():
    a=args(); outdir=Path(a.out_dir); outdir.mkdir(parents=True,exist_ok=True)
    overall={
      "races":0,
      "old_top6_consensus_top10_hits":0,
      "old_top6_union_hits":0,
      "new_top10_consensus_top10_hits":0,
      "new_top10_union_hits":0,
      "new_top10_union_candidates_total":0,
      "old_top6_union_candidates_total":0,
      "king_top10":{k:{"hits":0} for k in KING_ALIASES}
    }
    by_year={}
    by_field=defaultdict(lambda:{"races":0,"old_hits":0,"new_hits":0,"union_hits":0})
    for y in YEARS:
        snap=find_one(a.snapshot_root,[f"snapshot-{y}.jsonl.gz",f"snapshot-{y}.jsonl"])
        truth=read_truth(snap)
        scores={}
        for k in KING_ALIASES:
            d=Path(a.scores_root)/str(y)/k
            scores[k]=read_scores(find_one(d,["score.jsonl.gz","score.jsonl"]))
        race_sets=[set(v) for v in scores.values()]
        if any(s!=race_sets[0] for s in race_sets[1:]):raise SystemExit(f"{y}: king race coverage mismatch")
        if len(race_sets[0])!=3456:raise SystemExit(f"{y}: races {len(race_sets[0])} != 3456")
        yy={"races":0,"old_hits":0,"new_hits":0,"old_union_hits":0,"new_union_hits":0,
            "old_union_candidates_total":0,"new_union_candidates_total":0,
            "king_top10":{k:{"hits":0} for k in KING_ALIASES}}
        for rid in sorted(race_sets[0]):
            outcomes=truth[rid]["outcomes"]; fs=truth[rid]["field_size"]
            orders={k:scores[k][rid] for k in KING_ALIASES}
            old_final,old_all=consensus(orders,6,10)
            new_final,new_all=consensus(orders,10,10)
            old_union=set(h for o in orders.values() for h in o[:6])
            new_union=set(h for o in orders.values() for h in o[:10])
            oldhit=captures(old_final,outcomes); newhit=captures(new_final,outcomes)
            oldu=captures(old_union,outcomes); newu=captures(new_union,outcomes)
            overall["races"]+=1; yy["races"]+=1; by_field[fs]["races"]+=1
            overall["old_top6_consensus_top10_hits"]+=int(oldhit)
            overall["new_top10_consensus_top10_hits"]+=int(newhit)
            overall["old_top6_union_hits"]+=int(oldu)
            overall["new_top10_union_hits"]+=int(newu)
            overall["old_top6_union_candidates_total"]+=len(old_union)
            overall["new_top10_union_candidates_total"]+=len(new_union)
            yy["old_hits"]+=int(oldhit); yy["new_hits"]+=int(newhit); yy["old_union_hits"]+=int(oldu); yy["new_union_hits"]+=int(newu)
            yy["old_union_candidates_total"]+=len(old_union); yy["new_union_candidates_total"]+=len(new_union)
            by_field[fs]["old_hits"]+=int(oldhit); by_field[fs]["new_hits"]+=int(newhit); by_field[fs]["union_hits"]+=int(newu)
            for k in KING_ALIASES:
                h=captures(orders[k][:10],outcomes)
                overall["king_top10"][k]["hits"]+=int(h); yy["king_top10"][k]["hits"]+=int(h)
        for k,v in yy["king_top10"].items():v["rate"]=pct(v["hits"],yy["races"])
        yy["old_rate"]=pct(yy["old_hits"],yy["races"]); yy["new_rate"]=pct(yy["new_hits"],yy["races"])
        yy["old_union_rate"]=pct(yy["old_union_hits"],yy["races"]); yy["new_union_rate"]=pct(yy["new_union_hits"],yy["races"])
        yy["old_union_avg_candidates"]=yy["old_union_candidates_total"]/yy["races"]
        yy["new_union_avg_candidates"]=yy["new_union_candidates_total"]/yy["races"]
        by_year[str(y)]=yy
    n=overall["races"]
    for k,v in overall["king_top10"].items():v["rate"]=pct(v["hits"],n)
    overall["old_top6_consensus_top10_rate"]=pct(overall["old_top6_consensus_top10_hits"],n)
    overall["new_top10_consensus_top10_rate"]=pct(overall["new_top10_consensus_top10_hits"],n)
    overall["old_top6_union_rate"]=pct(overall["old_top6_union_hits"],n)
    overall["new_top10_union_rate"]=pct(overall["new_top10_union_hits"],n)
    overall["old_top6_union_avg_candidates"]=overall["old_top6_union_candidates_total"]/n
    overall["new_top10_union_avg_candidates"]=overall["new_top10_union_candidates_total"]/n
    overall["consensus_top10_gain_pp"]=(overall["new_top10_consensus_top10_rate"]-overall["old_top6_consensus_top10_rate"])*100
    fsout={}
    for fs,d in sorted(by_field.items()):
        fsout[str(fs)]={**d,"old_rate":pct(d["old_hits"],d["races"]),"new_rate":pct(d["new_hits"],d["races"]),"union_rate":pct(d["union_hits"],d["races"])}
    summary={
      "contract":"L15_7K_TOP10_EXPANSION_V1","years":list(YEARS),"locked_years":[2026],
      "ability_uses_odds":False,
      "method":{
        "old":"Each king contributes Top6; Borda points=7-rank; final consensus Top10.",
        "new":"Each king contributes Top10; Borda points=11-rank; final consensus Top10.",
        "union":"Union contains every horse appearing in any king TopN; not a final 10-horse betting set.",
        "truth":"Official 1st-3rd from same-generation snapshot; dead heats accepted if any valid podium outcome is contained."
      },
      "overall":overall,"by_year":by_year,"by_field_size":fsout
    }
    (outdir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with open(outdir/"king-top10.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["king","hits","races","rate"])
        for k,v in overall["king_top10"].items():w.writerow([k,v["hits"],n,v["rate"]])
    print("L15_7K_TOP10_EXPANSION_READY")
    print(json.dumps(overall,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":main()
