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
            out[str(r["race_id"])][c]=set(x for x in r["top3_horse_ids"].split("|") if x)
    return out

def blank():
    return {"n":0,"win":0,"podium":0}

def add(s,pos):
    s["n"]+=1
    if pos==1:s["win"]+=1
    if pos<=3:s["podium"]+=1

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
    if set(cons)!=set(truth): raise ValueError(f"coverage mismatch {len(cons)} {len(truth)}")

    exact={str(rank):{str(wait):blank() for wait in range(6)} for rank in [1,2,3,4,5,6]}
    rank_all={str(rank):blank() for rank in [1,2,3,4,5,6]}
    wait_all={str(wait):blank() for wait in range(6)}

    for rid,ranks in cons.items():
        if any(c not in outs[rid] for c in CANDS):
            raise ValueError(f"missing outsider {rid}")
        for h,rank in ranks.items():
            if rank not in (1,2,3,4,5,6): continue
            pos=truth[rid].get(h)
            if pos is None: continue
            support=sum(h in outs[rid][c] for c in CANDS)
            wait=5-support
            add(exact[str(rank)][str(wait)],pos)
            add(rank_all[str(rank)],pos)
            add(wait_all[str(wait)],pos)

    exact_f={r:{w:finish(s) for w,s in ws.items()} for r,ws in exact.items()}
    rank_f={r:finish(s) for r,s in rank_all.items()}
    wait_f={w:finish(s) for w,s in wait_all.items()}

    # Expected podium rate for each wait bucket from rank composition only.
    expected={}
    for w in map(str,range(6)):
        n=sum(exact_f[r][w]["n"] for r in ["1","2","3","4","5","6"])
        exp_num=sum(exact_f[r][w]["n"]*rank_f[r]["podium_rate"] for r in ["1","2","3","4","5","6"])
        actual=wait_f[w]["podium_rate"]
        exp=exp_num/n if n else None
        expected[w]={
          "n":n,
          "actual_podium_rate":actual,
          "rank_only_expected_podium_rate":exp,
          "delta_pp_vs_rank_only":(actual-exp)*100 if exp is not None and actual is not None else None
        }

    out={
      "contract":"L16_OUTSIDER_WAIT_SIGNAL_V1",
      "year":a.year,
      "races":len(cons),
      "includes_chimera":False,
      "definition":"wait_count = 5 - number of outsiders placing the same King rank 1-6 horse in their own Top3",
      "exact_king_rank_by_wait":exact_f,
      "king_rank_baseline":rank_f,
      "wait_all_king_top6_ranks_by_outsider_top3":wait_f,
      "wait_vs_rank_only_expected":expected
    }
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L16_OUTSIDER_WAIT_SIGNAL_OK",a.year)

if __name__=="__main__": main()
