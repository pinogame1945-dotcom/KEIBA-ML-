#!/usr/bin/env python3
"""Candidate-only 2026 seven king + 13 original outsider Top3 input adapter.

Reuses the frozen L1.6 consensus. The 13 expert names are strictly the
original six nonhorse plus seven transition candidates documented in
l16-outsider-support-safe-v1.yml; no substitute, ranks/payout/2026
results/p3 calibration. Never approves any 2026 model or bets.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

KING_CONTRACT="L16_FULL_RANK_CONSENSUS_V1"
CONTRACT="KEIBA_2026_7KING_13_OUTSIDER_RESEARCH_ONLY_SIGNAL_V1"
KING_ORDER=("core4","pedlegacy","condition","full","light","pedv1","condrc")
CONFIGS=("research/l1-outsider-nonhorse-v1.json",
         "research/l1-outsider-transition-v1.json")
EXPECTED_DAYS={"2026-10-10":(24,315),"2026-10-17":(36,471)}
EXPECTED_TOTAL=(60,786)
SCORE_WEIGHTS=(3,2,1)
BANNED_KEYS={"target","finish_position","payout","actual_is_win",
             "win_odds","final_win_odds","final_popularity"}

def demand(ok,msg):
    if not ok:raise ValueError(msg)

def canonical_candidates():
    names=[]
    for path in CONFIGS:
        x=json.loads(Path(path).read_text(encoding="utf-8"))
        demand(x["locked_years"]==[2026] and
               x["snapshot_generation"]=="eb13d4096519167b" and
               x["ability_uses_odds"] is False,
               "FROZEN_13_OUTSIDER_RESEARCH_LOCK_DRIFT")
        names.extend(c["name"] for c in x["candidates"])
    demand(len(names)==13 and len(set(names))==13,"NOT_ORIGINAL_13_OUTSIDERS")
    return tuple(names)

def load_consensus(path):
    seen={}
    days={d:[0,0] for d in EXPECTED_DAYS}
    with (gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")) as f:
        for line in f:
            if not line.strip():continue
            x=json.loads(line)
            rid=str(x.get("race_id") or "")
            day="2026-10-10" if rid in set(
                f"2026{v}0403{n:02d}" for v in ("05","08") for n in range(1,13)
            ) else "2026-10-17" if rid in set(
                f"2026{v}04{k}{n:02d}" for v,k in (("04","01"),("05","06"),("08","06")) for n in range(1,13)
            ) else ""
            demand(x.get("contract")==KING_CONTRACT and
                   x.get("expert_count")==7 and x.get("candidate_policy")=="ALL_HORSES"
                   and rid not in seen and day in days,
                   "CANDIDATE_SEVEN_KING_FROZEN_CONSENSUS_DRIFT:"+rid)
            rows=x.get("horses") or []
            demand(len(rows)==x.get("field_size") and
                   sorted(int(z["consensus_rank"]) for z in rows)==list(range(1,len(rows)+1)) and
                   len({str(z["horse_id"]) for z in rows})==len(rows),
                   "SEVEN_KING_RANK_POPULATION_DRIFT:"+rid)
            demand(all(set(z.get("expert_ranks") or {})==set(KING_ORDER) for z in rows),
                   "SEVEN_KING_EXPERT_SET_DRIFT:"+rid)
            seen[rid]=x
            days[day][0]+=1
            days[day][1]+=len(rows)
    demand(len(seen)==60 and sum(len(z["horses"]) for z in seen.values())==786 and
           all(tuple(v)==EXPECTED_DAYS[d] for d,v in days.items()),
           "2026_60_RACES_786_HORSES_OR_DAY_COVERAGE_DRIFT")
    return seen

def load_top3_dir(folder,candidates,kings):
    files=sorted(folder.glob("*.csv"))
    demand({f.stem for f in files}==set(candidates),"MISSING_OR_EXTRA_13_OUTSIDER_BALLOT_FILE")
    votes=defaultdict(dict)
    for f in files:
        with f.open(encoding="utf-8-sig",newline="") as stream:
            reader=csv.DictReader(stream)
            demand(set(reader.fieldnames or [])=={
                "race_id","candidate","top1_horse_id","top2_horse_id","top3_horse_id"},
                "OUTSIDER_TOP3_FROZEN_CSV_FORMAT_DRIFT:"+f.name)
            rows=0
            for x in reader:
                rid=str(x["race_id"])
                candidate=str(x["candidate"])
                demand(candidate==f.stem and rid in kings and
                       candidate not in votes[rid],
                       "OUTSIDER_UNKNOWN_CANDIDATE_OR_RACE_DUPLICATE:"+f.name)
                top3=[str(x[f"top{k}_horse_id"] or "") for k in (1,2,3)]
                demand(all(top3) and len(set(top3))==3,
                       "OUTSIDER_TOP3_INCOMPLETE_OR_DUPLICATED:"+rid+":"+candidate)
                allowed={str(h["horse_id"]) for h in kings[rid]["horses"]}
                demand(set(top3).issubset(allowed),
                       "OUTSIDER_VOTE_HORSE_NOT_IN_FROZEN_KING_FIELD:"+rid+":"+candidate)
                votes[rid][candidate]=top3
                rows+=1
            demand(rows==60,"OUTSIDER_BALLOT_FILE_NOT_60_RACES:"+f.name)
    demand(len(votes)==60 and all(set(v)==set(candidates) for v in votes.values()),
           "OUTSIDER_13X60_BALLOTS_NOT_COMPLETE")
    return votes

def run(council_path,top3_dir,output):
    require_dir=top3_dir.exists() and top3_dir.is_dir()
    demand(require_dir,"ORIGINAL_13_OUTSIDER_REAL_TOP3_NOT_AVAILABLE_NO_FALLBACK")
    candidates=canonical_candidates()
    kings=load_consensus(council_path)
    votes=load_top3_dir(top3_dir,candidates,kings)
    demand(not output.exists(),"RESEARCH_ONLY_SIGNAL_REFUSE_OVERWRITE")
    output.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if str(output).endswith(".gz") else open
    n=0
    with opener(output,"wt",encoding="utf-8") as dest:
        for rid in sorted(kings):
            origin=kings[rid]
            hs=[]
            for z in sorted(origin["horses"],key=lambda x:x["consensus_rank"]):
                hid=str(z["horse_id"])
                count=top1=top2=top3=score=0
                for candidate in candidates:
                    ranking=votes[rid][candidate]
                    if hid in ranking:
                        rank=ranking.index(hid)+1
                        count+=1;score+=4-rank
                        if rank==1:top1+=1
                        if rank==2:top2+=1
                        if rank==3:top3+=1
                demand(0<=count<=13 and score==3*top1+2*top2+top3,
                       "FROZEN_OUTSIDER_TOP3_ARITHMETIC_DRIFT")
                hs.append({
                    "horse_id":hid,
                    "king_rank":int(z["consensus_rank"]),
                    "king_borda_score":z["borda_score"],
                    "king_probability_mean":z["probability_mean"],
                    "outsider_support_count":count,
                    "outsider_score":score,
                    "outsider_top1_votes":top1,
                    "outsider_top2_votes":top2,
                    "outsider_top3_votes":top3,
                    "p3_calibrated":None,
                    "calibration_status":"NOT_TRAINED_FOR_2026",
                })
            demand([h["king_rank"] for h in hs]==list(range(1,len(hs)+1)),
                   "KING_CONSENSUS_RERANK_FORBIDDEN")
            doc={
                "contract":CONTRACT,"race_id":rid,"prediction_year":2026,
                "horse_count":len(hs),"candidate_count":13,
                "ranking_policy":"SEVEN_KING_UNCHANGED",
                "outsider_policy":"SIGNAL_ONLY_NO_RERANK",
                "calibration_available":False,"model_release_authorized":False,
                "ready_for_betting":False,"l175_l21_l3_invoked":False,
                "2026_results_used":False,"horses":hs
            }
            dest.write(json.dumps(doc,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")
            n+=len(hs)
    demand(n==786,"UNEXPECTED_OUTSIDER_HORSE_ROWS")
    print("2026_SEVENKING_13_OUTSIDER_RESEARCH_ONLY_SIGNAL_PASS "+
          json.dumps({"races":60,"horses":n,"outsiders":13,
                      "production_approved":False,"input_ranks_unchanged":True},
                     sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--consensus",type=Path,required=True)
    p.add_argument("--ballots-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.consensus,a.ballots_dir,a.output)
