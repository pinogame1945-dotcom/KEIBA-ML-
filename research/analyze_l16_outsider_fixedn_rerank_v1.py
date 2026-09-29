#!/usr/bin/env python3
import argparse,csv,gzip,json,itertools
from pathlib import Path
from collections import defaultdict

CANDS=["outsider_daytrend","outsider_raceshape","outsider_gatecourse","outsider_field","outsider_jockey"]
NS=[3,4,5,6]
A3=[0.0,0.5,1.0,1.5,2.0,3.0]
B6=[0.0,0.25,0.5,1.0,1.5]
PROMOTE_MIN=[0,2,3]

def ap():
    p=argparse.ArgumentParser()
    p.add_argument("--consensus",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--outsider-csv",required=True)
    p.add_argument("--year",type=int,required=True)
    p.add_argument("--output",required=True)
    return p.parse_args()

def opent(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def load_truth(path):
    groups=defaultdict(lambda:defaultdict(list))
    positions=defaultdict(dict)
    with opent(path) as f:
        for line in f:
            if not line.strip(): continue
            x=json.loads(line)
            rid=str(x.get("race_id") or ""); hid=str(x.get("horse_id") or "")
            if not rid or not hid: continue
            try: pos=int(float((x.get("target") or {}).get("finish_position")))
            except Exception: continue
            positions[rid][hid]=pos
            if pos<=3:
                groups[rid][pos].append(hid)
    out={}
    for rid,posmap in positions.items():
        g=groups[rid]
        pieces=[()]
        for rank in sorted(g):
            if rank>3: continue
            hs=list(g[rank]); occupied=min(len(hs),3-rank+1)
            perms=list(itertools.permutations(hs,occupied))
            pieces=[a+b for a in pieces for b in perms]
        outcomes=sorted(set(x for x in pieces if len(x)==3))
        if not outcomes:
            raise ValueError(f"truth outcomes missing {rid}")
        out[rid]={"positions":posmap,"outcomes":outcomes}
    return out

def load_cons(path):
    out={}
    with opent(path) as f:
        for line in f:
            if not line.strip(): continue
            x=json.loads(line)
            rows=sorted(x["horses"],key=lambda z:int(z["consensus_rank"]))
            out[str(x["race_id"])]={str(z["horse_id"]):int(z["consensus_rank"]) for z in rows}
    return out

def load_out(path):
    out=defaultdict(dict)
    with open(path,newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            c=r["candidate"]
            if c not in CANDS: continue
            rid=str(r["race_id"])
            out[rid][c]={
                3:set(x for x in r["top3_horse_ids"].split("|") if x),
                6:set(x for x in r["top6_horse_ids"].split("|") if x),
            }
    return out

def metric():
    return {"races":0,"full_podium_hits":0,"winner_hits":0,"podium_occ_hits":0,"podium_occ_total":0,"selected_total":0}

def upd(m,sel,truth):
    m["races"]+=1
    m["selected_total"]+=len(sel)
    positions=truth["positions"]
    podium={h for h,p in positions.items() if p<=3}
    winner={h for h,p in positions.items() if p==1}
    selected=set(sel)
    m["podium_occ_total"]+=len(podium)
    m["podium_occ_hits"]+=len(podium & selected)
    if any(set(o).issubset(selected) for o in truth["outcomes"]): m["full_podium_hits"]+=1
    if winner & selected: m["winner_hits"]+=1

def fin(m):
    r=m["races"]
    return {**m,
      "full_podium_rate":m["full_podium_hits"]/r if r else None,
      "winner_capture_rate":m["winner_hits"]/r if r else None,
      "podium_occ_recall":m["podium_occ_hits"]/m["podium_occ_total"] if m["podium_occ_total"] else None,
      "avg_selected":m["selected_total"]/r if r else None,
    }

def votes_for(rid,h,outs):
    v3=sum(h in outs[rid][c][3] for c in CANDS)
    v6=sum(h in outs[rid][c][6] for c in CANDS)
    return v3,v6

def choose_baseline(ranks,n):
    return [h for h,r in sorted(ranks.items(),key=lambda z:(z[1],z[0])) if r<=n]

def choose_internal(ranks,outs,rid,n,mode):
    top6=[h for h,r in ranks.items() if r<=6]
    def key(h):
        v3,v6=votes_for(rid,h,outs)
        v=v3 if mode=="v3" else v6
        return (-v,ranks[h],h)
    return sorted(top6,key=key)[:min(n,len(top6))]

def policy_name(a,b,t):
    return f"weighted_a3={a:g}_b6extra={b:g}_minv3={t}"

def choose_weighted(ranks,outs,rid,n,a,b,minv3):
    base=set(choose_baseline(ranks,n))
    scored=[]
    for h,r in ranks.items():
        v3,v6=votes_for(rid,h,outs)
        extra=v6-v3
        bonus=a*v3+b*extra
        # if outside baseline and threshold requested, suppress outsider bonus unless enough Top3 votes
        if h not in base and minv3 and v3<minv3:
            bonus=0.0
        score=-float(r)+bonus
        scored.append((score,v3,v6,-r,h))
    scored.sort(reverse=True)
    return [x[-1] for x in scored[:min(n,len(scored))]]

def main():
    a=ap()
    if a.year==2026: raise ValueError("2026 locked")
    truth=load_truth(a.snapshot); ranks=load_cons(a.consensus); outs=load_out(a.outsider_csv)
    if set(ranks)!=set(truth): raise ValueError(f"coverage mismatch ranks={len(ranks)} truth={len(truth)}")
    for rid in ranks:
        miss=[c for c in CANDS if c not in outs[rid]]
        if miss: raise ValueError(f"missing outsider rows {rid} {miss}")

    metrics={}
    for n in NS:
        metrics[(f"king_baseline",n)]=metric()
        metrics[(f"internal_v3",n)]=metric()
        metrics[(f"internal_v6",n)]=metric()
        for a3,b6,t in itertools.product(A3,B6,PROMOTE_MIN):
            if a3==0 and b6==0 and t!=0: continue
            metrics[(policy_name(a3,b6,t),n)]=metric()

    for rid,rr in ranks.items():
        tr=truth[rid]
        for n in NS:
            upd(metrics[("king_baseline",n)],choose_baseline(rr,n),tr)
            upd(metrics[("internal_v3",n)],choose_internal(rr,outs,rid,n,"v3"),tr)
            upd(metrics[("internal_v6",n)],choose_internal(rr,outs,rid,n,"v6"),tr)
            for a3,b6,t in itertools.product(A3,B6,PROMOTE_MIN):
                if a3==0 and b6==0 and t!=0: continue
                nm=policy_name(a3,b6,t)
                upd(metrics[(nm,n)],choose_weighted(rr,outs,rid,n,a3,b6,t),tr)

    out={"contract":"L16_OUTSIDER_FIXEDN_RERANK_V1","year":a.year,"races":len(ranks),"includes_chimera":False,"truth_semantics":"official_l16_dead_heat_outcomes","n_values":NS,"policies":{}}
    for (p,n),m in metrics.items():
        out["policies"].setdefault(p,{})[str(n)]=fin(m)

    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L16_OUTSIDER_FIXEDN_RERANK_OK",a.year,len(out["policies"]))

if __name__=="__main__": main()
