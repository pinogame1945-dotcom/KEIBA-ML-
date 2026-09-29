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
    p.add_argument("--output",required=True)
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
            out[str(x["race_id"])]={str(h["horse_id"]):int(h["consensus_rank"]) for h in x["horses"]}
    return out

def load_out(path):
    out=defaultdict(dict)
    with open(path,newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            c=r["candidate"]
            if c not in CANDS: continue
            ids=[x for x in r["top3_horse_ids"].split("|") if x]
            out[str(r["race_id"])][c]={hid:i+1 for i,hid in enumerate(ids)}
    return out

def blank():
    return {"n":0,"win":0,"podium":0}

def add(s,pos):
    s["n"]+=1
    if pos==1: s["win"]+=1
    if pos<=3: s["podium"]+=1

def fin(s):
    n=s["n"]
    return {**s,"win_rate":s["win"]/n if n else None,"podium_rate":s["podium"]/n if n else None}

def key(v1,v2,v3,vout):
    return f"1st={v1}|2nd={v2}|3rd={v3}|out={vout}"

def main():
    a=ap()
    if a.year==2026: raise ValueError("2026 locked")
    truth=load_truth(a.snapshot); cons=load_cons(a.consensus); outs=load_out(a.outsider_csv)
    if set(cons)!=set(truth): raise ValueError("coverage mismatch")

    by_pattern=defaultdict(blank)
    by_rank_pattern=defaultdict(lambda:defaultdict(blank))
    by_wait_pattern=defaultdict(lambda:defaultdict(blank))

    for rid,ranks in cons.items():
        if any(c not in outs[rid] for c in CANDS):
            raise ValueError(f"missing outsider rows {rid}")
        for hid,krank in ranks.items():
            pos=truth[rid].get(hid)
            if pos is None: continue
            votes=[outs[rid][c].get(hid,0) for c in CANDS]
            v1=sum(v==1 for v in votes); v2=sum(v==2 for v in votes); v3=sum(v==3 for v in votes); vout=sum(v==0 for v in votes)
            if v1+v2+v3+vout!=5: raise ValueError("vote accounting")
            k=key(v1,v2,v3,vout)
            add(by_pattern[k],pos)
            add(by_rank_pattern[str(krank)][k],pos)
            add(by_wait_pattern[str(vout)][k],pos)

    out={
      "contract":"L16_OUTSIDER_WAIT_BREAKDOWN_V1",
      "year":a.year,
      "races":len(cons),
      "includes_chimera":False,
      "definition":"For each horse across formal5 Outsiders, count how many rank it 1st, 2nd, 3rd, or outside Top3.",
      "by_pattern":{k:fin(v) for k,v in by_pattern.items()},
      "by_king_rank_pattern":{r:{k:fin(v) for k,v in d.items()} for r,d in by_rank_pattern.items()},
      "by_wait_count_pattern":{w:{k:fin(v) for k,v in d.items()} for w,d in by_wait_pattern.items()}
    }
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L16_OUTSIDER_WAIT_BREAKDOWN_OK",a.year)

if __name__=="__main__": main()
