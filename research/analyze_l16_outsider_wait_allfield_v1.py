#!/usr/bin/env python3
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path

CANDS=["outsider_daytrend","outsider_raceshape","outsider_gatecourse","outsider_field","outsider_jockey"]

def ap():
    p=argparse.ArgumentParser()
    p.add_argument("--consensus",required=True)
    p.add_argument("--snapshot",required=True)
    p.add_argument("--outsider-csv",required=True)
    p.add_argument("--year",type=int,required=True)
    p.add_argument("--output-summary",required=True)
    p.add_argument("--output-horses",required=True)
    return p.parse_args()

def opent(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def load_truth(path):
    out=defaultdict(dict)
    with opent(path) as f:
        for line in f:
            if not line.strip(): continue
            x=json.loads(line)
            rid=str(x.get("race_id") or ""); hid=str(x.get("horse_id") or "")
            if not rid or not hid: continue
            try: pos=int(float((x.get("target") or {}).get("finish_position")))
            except Exception: continue
            out[rid][hid]=pos
    return out

def load_cons(path):
    out={}
    with opent(path) as f:
        for line in f:
            if not line.strip(): continue
            x=json.loads(line)
            rows=sorted(x["horses"],key=lambda h:int(h["consensus_rank"]))
            out[str(x["race_id"])]=[(str(h["horse_id"]),int(h["consensus_rank"])) for h in rows]
    return out

def load_out(path):
    out=defaultdict(dict)
    with open(path,newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            c=r["candidate"]
            if c not in CANDS: continue
            out[str(r["race_id"])][c]=set(x for x in r["top3_horse_ids"].split("|") if x)
    return out

def blank():
    return {"n":0,"win":0,"podium":0}

def add(s,pos):
    s["n"]+=1
    if pos==1: s["win"]+=1
    if pos<=3: s["podium"]+=1

def finish(s):
    n=s["n"]
    return {**s,
      "win_rate":s["win"]/n if n else None,
      "podium_rate":s["podium"]/n if n else None
    }

def main():
    a=ap()
    if a.year==2026: raise ValueError("2026 locked")
    truth=load_truth(a.snapshot); cons=load_cons(a.consensus); outs=load_out(a.outsider_csv)
    if set(cons)!=set(truth): raise ValueError(f"coverage mismatch cons={len(cons)} truth={len(truth)}")

    by_wait={str(w):blank() for w in range(6)}
    by_rank_wait=defaultdict(lambda:{str(w):blank() for w in range(6)})
    total_horses=0
    evaluated_horses=0
    missing_finish_horses=0

    op=Path(a.output_horses); op.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(op,"wt",encoding="utf-8",newline="") as gz:
        wr=csv.writer(gz)
        wr.writerow(["year","race_id","horse_id","king_rank","outsider_top3_support","wait_count","finish_position","has_finish_position","is_win","is_podium"])
        for rid,rows in cons.items():
            if any(c not in outs[rid] for c in CANDS):
                raise ValueError(f"missing outsider rows {rid}")
            for hid,rank in rows:
                pos=truth[rid].get(hid)
                support=sum(hid in outs[rid][c] for c in CANDS)
                wait=5-support
                total_horses+=1
                if pos is None:
                    missing_finish_horses+=1
                    wr.writerow([a.year,rid,hid,rank,support,wait,"",0,"",""])
                    continue
                evaluated_horses+=1
                wr.writerow([a.year,rid,hid,rank,support,wait,pos,1,int(pos==1),int(pos<=3)])
                add(by_wait[str(wait)],pos)
                add(by_rank_wait[str(rank)][str(wait)],pos)

    summary={
      "contract":"L16_OUTSIDER_WAIT_ALLFIELD_V1",
      "year":a.year,
      "races":len(cons),
      "horses":total_horses,
      "evaluated_horses":evaluated_horses,
      "missing_finish_horses":missing_finish_horses,
      "includes_chimera":False,
      "definition":"For every horse in every race: wait_count = 5 - number of formal Outsiders placing that horse in their own Top3.",
      "by_wait":{w:finish(s) for w,s in by_wait.items()},
      "by_king_rank_wait":{r:{w:finish(s) for w,s in ws.items()} for r,ws in sorted(by_rank_wait.items(),key=lambda z:int(z[0]))}
    }
    sp=Path(a.output_summary); sp.parent.mkdir(parents=True,exist_ok=True)
    sp.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L16_OUTSIDER_WAIT_ALLFIELD_OK",a.year,len(cons),total_horses,evaluated_horses,missing_finish_horses)

if __name__=="__main__": main()
