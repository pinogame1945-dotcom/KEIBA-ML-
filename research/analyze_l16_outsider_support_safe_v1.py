#!/usr/bin/env python3
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path

YEARS=(2021,2022,2023,2024,2025)
EXPECTED_CANDIDATES=13

def ap():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True,help="Root containing yYYYY/{consensus.jsonl.gz,snapshot.jsonl[.gz],outsider-top3/*.csv}")
    p.add_argument("--output-json",required=True)
    p.add_argument("--output-csv",required=True)
    return p.parse_args()

def opent(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def load_truth(path):
    out=defaultdict(dict)
    with opent(path) as fh:
        for line in fh:
            if not line.strip(): continue
            x=json.loads(line)
            rid=str(x.get("race_id") or ""); hid=str(x.get("horse_id") or "")
            if not rid or not hid: continue
            try:
                pos=int(float((x.get("target") or {}).get("finish_position")))
            except (TypeError,ValueError):
                continue
            out[rid][hid]=pos
    return out

def load_consensus(path):
    out={}
    with opent(path) as fh:
        for line in fh:
            if not line.strip(): continue
            x=json.loads(line)
            rows=sorted(x["horses"],key=lambda z:int(z["consensus_rank"]))
            out[str(x["race_id"])]={str(z["horse_id"]):int(z["consensus_rank"]) for z in rows}
    return out

def load_outsider_top3(path):
    out=defaultdict(dict)
    files=sorted(Path(path).glob("*.csv"))
    candidates=set()
    for p in files:
        with p.open(newline="",encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                rid=str(r["race_id"]); cand=str(r["candidate"])
                candidates.add(cand)
                ranks={}
                for k in (1,2,3):
                    hid=str(r.get(f"top{k}_horse_id") or "")
                    if hid: ranks[hid]=k
                out[rid][cand]=ranks
    if len(candidates)!=EXPECTED_CANDIDATES:
        raise ValueError(f"expected {EXPECTED_CANDIDATES} candidates, got {sorted(candidates)}")
    return out,sorted(candidates)

def blank():
    return {"n":0,"win":0,"podium":0}

def add(s,pos):
    s["n"]+=1
    if pos==1:s["win"]+=1
    if pos<=3:s["podium"]+=1

def finish(s):
    n=s["n"]
    return {**s,"win_rate":s["win"]/n if n else None,"podium_rate":s["podium"]/n if n else None}

def finalize_map(d):
    return {str(k):finish(v) for k,v in sorted(d.items(),key=lambda z:int(z[0]))}

def monotonic_report(d):
    rows=[(int(k),finish(v)) for k,v in sorted(d.items(),key=lambda z:int(z[0])) if v["n"]>0]
    pairs=0; nondec=0
    for (_,a),(_,b) in zip(rows,rows[1:]):
        if a["podium_rate"] is None or b["podium_rate"] is None: continue
        pairs+=1
        nondec+=int(b["podium_rate"]>=a["podium_rate"])
    return {
        "occupied_bins":len(rows),
        "adjacent_pairs":pairs,
        "podium_non_decreasing_pairs":nondec,
        "podium_non_decreasing_share":nondec/pairs if pairs else None,
    }

def build_scope(rows):
    by_count=defaultdict(blank)
    by_score=defaultdict(blank)
    by_wait=defaultdict(blank)
    cross=defaultdict(lambda:defaultdict(blank))
    for r in rows:
        add(by_count[r["support_count"]],r["finish_position"])
        add(by_score[r["support_score"]],r["finish_position"])
        add(by_wait[r["weighted_wait"]],r["finish_position"])
        add(cross[r["support_count"]][r["support_score"]],r["finish_position"])
    return {
        "n":len(rows),
        "by_support_count":finalize_map(by_count),
        "by_support_score":finalize_map(by_score),
        "by_weighted_wait":finalize_map(by_wait),
        "by_support_count_and_score":{
            str(c):finalize_map(d) for c,d in sorted(cross.items(),key=lambda z:int(z[0]))
        },
        "monotonicity":{
            "support_count":monotonic_report(by_count),
            "support_score":monotonic_report(by_score),
        },
    }

def main():
    a=ap(); root=Path(a.root)
    all_rows=[]; by_year={}; candidates_ref=None

    for year in YEARS:
        yroot=root/f"y{year}"
        cons_path=yroot/"consensus.jsonl.gz"
        snap_gz=yroot/"snapshot.jsonl.gz"
        snap_plain=yroot/"snapshot.jsonl"
        snap_path=snap_gz if snap_gz.exists() else snap_plain
        top3_dir=yroot/"outsider-top3"
        if not cons_path.exists() or not snap_path.exists() or not top3_dir.exists():
            raise ValueError(f"missing year inputs y{year}")

        truth=load_truth(snap_path)
        cons=load_consensus(cons_path)
        outs,candidates=load_outsider_top3(top3_dir)
        if candidates_ref is None: candidates_ref=candidates
        elif candidates!=candidates_ref: raise ValueError("candidate set differs by year")
        if set(cons)!=set(truth):
            raise ValueError(f"coverage mismatch y{year}: consensus={len(cons)} truth={len(truth)}")
        for rid in cons:
            missing=[c for c in candidates if c not in outs[rid]]
            if missing: raise ValueError(f"missing outsider top3 y{year} race={rid}: {missing}")

        rows=[]
        for rid,ranks in cons.items():
            for hid,krank in ranks.items():
                if krank>3: continue
                pos=truth[rid].get(hid)
                if pos is None: continue
                support_count=0; support_score=0
                for cand in candidates:
                    orank=outs[rid][cand].get(hid)
                    if orank is not None:
                        support_count+=1
                        support_score+=(4-orank)  # 1st=3, 2nd=2, 3rd=1
                rows.append({
                    "year":year,
                    "race_id":rid,
                    "horse_id":hid,
                    "king_rank":krank,
                    "finish_position":pos,
                    "support_count":support_count,
                    "support_score":support_score,
                    "weighted_wait":3*len(candidates)-support_score,
                })
        expected=len(cons)*3
        if len(rows)!=expected:
            raise ValueError(f"expected exactly 3 king horses per race y{year}: rows={len(rows)} expected={expected}")
        all_rows.extend(rows)
        by_year[str(year)]={
            "races":len(cons),
            "king_top3":build_scope(rows),
            "by_king_rank":{str(k):build_scope([r for r in rows if r["king_rank"]==k]) for k in (1,2,3)},
        }

    out={
        "contract":"L16_OUTSIDER_SUPPORT_SAFE_V1",
        "years":list(YEARS),
        "races":sum(by_year[str(y)]["races"] for y in YEARS),
        "candidate_count":len(candidates_ref),
        "candidates":candidates_ref,
        "definition":{
            "support_count":"number of tie-safe outsiders placing horse in Top3",
            "support_score":"sum across outsiders: rank1=3, rank2=2, rank3=1, outside Top3=0",
            "weighted_wait":"39-support_score for 13 outsiders",
            "evaluation_population":"Seven-King full-rank consensus positions 1-3 only",
        },
        "overall":{
            "king_top3":build_scope(all_rows),
            "by_king_rank":{str(k):build_scope([r for r in all_rows if r["king_rank"]==k]) for k in (1,2,3)},
        },
        "by_year":by_year,
        "2026_sealed":True,
        "odds_used":False,
    }

    p=Path(a.output_json); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    fields=["scope","year","king_rank","dimension","value","n","win","podium","win_rate","podium_rate"]
    with open(a.output_csv,"w",newline="",encoding="utf-8-sig") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
        def emit(scope,year,king_rank,obj):
            for dim,key in (("support_count","by_support_count"),("support_score","by_support_score"),("weighted_wait","by_weighted_wait")):
                for value,s in obj[key].items():
                    w.writerow({"scope":scope,"year":year,"king_rank":king_rank,"dimension":dim,"value":value,**s})
        emit("overall_top3","ALL","1-3",out["overall"]["king_top3"])
        for k in ("1","2","3"): emit("overall_rank","ALL",k,out["overall"]["by_king_rank"][k])
        for y in map(str,YEARS):
            emit("year_top3",y,"1-3",out["by_year"][y]["king_top3"])
            for k in ("1","2","3"): emit("year_rank",y,k,out["by_year"][y]["by_king_rank"][k])

    print("L16_OUTSIDER_SUPPORT_SAFE_OK")
    print(json.dumps({
        "races":out["races"],
        "candidate_count":out["candidate_count"],
        "top3_support_count_monotonicity":out["overall"]["king_top3"]["monotonicity"]["support_count"],
        "rank1_support_count_monotonicity":out["overall"]["by_king_rank"]["1"]["monotonicity"]["support_count"],
        "rank2_support_count_monotonicity":out["overall"]["by_king_rank"]["2"]["monotonicity"]["support_count"],
        "rank3_support_count_monotonicity":out["overall"]["by_king_rank"]["3"]["monotonicity"]["support_count"],
    },ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
