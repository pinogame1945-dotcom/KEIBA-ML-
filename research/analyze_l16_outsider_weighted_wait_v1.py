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
    if pos==1:s["win"]+=1
    if pos<=3:s["podium"]+=1

def fin(s):
    n=s["n"]
    return {**s,"win_rate":s["win"]/n if n else None,"podium_rate":s["podium"]/n if n else None}

def main():
    a=ap()
    if a.year==2026: raise ValueError("2026 locked")
    truth=load_truth(a.snapshot); cons=load_cons(a.consensus); outs=load_out(a.outsider_csv)
    if set(cons)!=set(truth): raise ValueError("coverage mismatch")

    # wait_count: Top3外のOutsider人数 (0..5)
    # weighted_wait: outsider rank1=0, rank2=1, rank3=2, Top3外=3, summed across 5 (0..15)
    by_wait_weight=defaultdict(lambda:defaultdict(blank))
    by_rank_wait_weight=defaultdict(lambda:defaultdict(lambda:defaultdict(blank)))
    by_weight=defaultdict(blank)

    for rid,ranks in cons.items():
        if any(c not in outs[rid] for c in CANDS):
            raise ValueError(f"missing outsider {rid}")
        for hid,krank in ranks.items():
            pos=truth[rid].get(hid)
            if pos is None: continue
            wc=0; ww=0
            for c in CANDS:
                r=outs[rid][c].get(hid)
                if r is None:
                    wc+=1; ww+=3
                elif r==1:
                    ww+=0
                elif r==2:
                    ww+=1
                elif r==3:
                    ww+=2
                else:
                    raise ValueError("bad outsider rank")
            add(by_wait_weight[str(wc)][str(ww)],pos)
            add(by_rank_wait_weight[str(krank)][str(wc)][str(ww)],pos)
            add(by_weight[str(ww)],pos)

    out={
      "contract":"L16_OUTSIDER_WEIGHTED_WAIT_V1",
      "year":a.year,
      "races":len(cons),
      "includes_chimera":False,
      "definition":{
        "wait_count":"number of formal5 outsiders not placing horse in Top3",
        "weighted_wait":"sum over formal5: outsider rank1=0, rank2=1, rank3=2, outside Top3=3; range 0-15"
      },
      "by_wait_count_and_weighted_wait":{
        wc:{ww:fin(s) for ww,s in sorted(d.items(),key=lambda z:int(z[0]))}
        for wc,d in sorted(by_wait_weight.items(),key=lambda z:int(z[0]))
      },
      "by_king_rank_wait_count_and_weighted_wait":{
        kr:{
          wc:{ww:fin(s) for ww,s in sorted(d2.items(),key=lambda z:int(z[0]))}
          for wc,d2 in sorted(d1.items(),key=lambda z:int(z[0]))
        }
        for kr,d1 in sorted(by_rank_wait_weight.items(),key=lambda z:int(z[0]))
      },
      "by_weighted_wait":{ww:fin(s) for ww,s in sorted(by_weight.items(),key=lambda z:int(z[0]))}
    }
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L16_OUTSIDER_WEIGHTED_WAIT_OK",a.year)

if __name__=="__main__": main()
