#!/usr/bin/env python3
"""Replay 7 king tie-safe ranks + 13 Outsider ballots from EXISTING GitHub archives.

Saved archives are distinct from a live re-prediction. Explicitly discard
winner, podium, market_order and market_ranks from historic council records.
"""
from __future__ import annotations
import argparse,csv,gzip,json,math
from collections import defaultdict
from pathlib import Path

EXPERTS=("core4","pedlegacy","condition","full","light","pedv1","condrc")
OUTSIDERS=(
    "outsider_bodychange","outsider_chimera","outsider_classmove",
    "outsider_coursetransition","outsider_daytrend","outsider_disttransition",
    "outsider_field","outsider_gatecourse","outsider_jockey",
    "outsider_opponentgap","outsider_raceshape","outsider_rotation",
    "outsider_weightchange"
)
COUNCIL_CONTRACT="L1_COUNCIL_REBUILD_YEAR_V1"

def text_open(path):
    return gzip.open(path,"rt",encoding="utf-8-sig") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8-sig")

def read_council(path,race_id):
    row=None
    with text_open(path) as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line)
            if str(r.get("race_id") or "")!=race_id:continue
            if row is not None:raise ValueError("duplicated race in saved council archive")
            row=r
    if row is None:raise ValueError("race not found in archived council matrix")
    if int(row.get("year",0))!=int(race_id[:4]):raise ValueError("year mismatch")
    horse_ids=[str(h) for h in row.get("horses") or []]
    if len(horse_ids)<3 or len(horse_ids)!=len(set(horse_ids)) or any(not h.isdigit() for h in horse_ids):
        raise ValueError("invalid archived horse IDs")
    matrix=row.get("rank_matrix")
    if not isinstance(matrix,list) or len(matrix)!=len(EXPERTS):
        raise ValueError("archived matrix must contain exactly 7 experts")
    for ranks in matrix:
        if not isinstance(ranks,list) or len(ranks)!=len(horse_ids) or sorted(ranks)!=list(range(1,len(horse_ids)+1)):
            raise ValueError("invalid expert rank permutation")
    # Strictly project only pre-race horse scores/support. Archived outcome,
    # market_order and market_ranks are forbidden output/input here.
    return {"race_id":race_id,"race_date":str(row.get("race_date") or ""),
            "horses":horse_ids,"rank_matrix":matrix,
            "outsider_support":row.get("outsider_support") or []}

def read_ballots(path,race_id,horse_ids):
    ballots={}
    with text_open(path) as f:
        for row in csv.DictReader(f):
            if str(row.get("race_id") or "")!=race_id:continue
            name=str(row.get("candidate") or "")
            if name in ballots:raise ValueError("duplicate outsider candidate "+name)
            if name not in OUTSIDERS:raise ValueError("unknown outsider candidate "+name)
            ids=[str(row.get(f"top{pos}_horse_id") or "").strip() for pos in (1,2,3)]
            ids=[hid or None for hid in ids]
            if len([h for h in ids if h])!=len(set(h for h in ids if h)):
                raise ValueError("outsider candidate has repeated horse in Top3")
            if any(h and h not in horse_ids for h in ids):
                raise ValueError("outsider ballot horse absent from king field "+name)
            ballots[name]=ids
    if set(ballots)!=set(OUTSIDERS):
        raise ValueError("missing 13 Outsider candidate ballots for race: "+
                         ",".join(sorted(set(OUTSIDERS)-set(ballots))))
    return ballots

def aggregate_ballots(ballots):
    points=defaultdict(lambda:{"points":0,"votes":0,"best":99})
    for ids in ballots.values():
        for index,hid in enumerate(ids,1):
            if not hid:continue
            p=points[hid]
            p["points"]+=4-index;p["votes"]+=1;p["best"]=min(p["best"],index)
    return points

def replay(race_id,council_file,ballots_file):
    rec=read_council(council_file,race_id)
    horse_ids=rec["horses"]
    per_horse={}
    for j,hid in enumerate(horse_ids):
        rs={EXPERTS[i]:int(rec["rank_matrix"][i][j]) for i in range(7)}
        arr=list(rs.values())
        per_horse[hid]={
            "horse_id":hid,"king_ranks":rs,"king_rank_sum":sum(arr),
            "king_top6_votes":sum(r<=6 for r in arr),
            "king_top3_votes":sum(r<=3 for r in arr),
            "king_top1_votes":sum(r==1 for r in arr),
        }
    ordered=sorted(horse_ids,key=lambda h:(
        per_horse[h]["king_rank_sum"],-per_horse[h]["king_top6_votes"],
        -per_horse[h]["king_top3_votes"],-per_horse[h]["king_top1_votes"],h))
    for place,hid in enumerate(ordered,1):
        per_horse[hid]["pure_7king_rank_sum_comparator_rank"]=place
    report={
        "contract":"KEIBA_GITHUB_ARCHIVED_7KING_13OUTSIDER_V1",
        "race_id":race_id,"race_date":rec["race_date"],
        "source":"PRIOR_RESEARCH_COMMITTED_GITHUB_ARCHIVES",
        "prediction_origin":"HISTORICAL_SAVED_OOF_REPLAY_NOT_NEW_INFERENCE",
        "source_prediction_phase":"FINAL_RESEARCH_NO_PRESERVED_PREBET_SNAPSHOTS",
        "winner_or_payouts_used":False,
        "final_market_odds_used":False,
        "expert_order":list(EXPERTS),
        "council_method":"M127_S0 rank sum with Top6/Top3/Top1 tie breaks, horse ID. This is NOT frozen L2.1 router.",
        "horses":sorted(per_horse.values(),key=lambda z:z["pure_7king_rank_sum_comparator_rank"]),
        "outsider_status":"UNAVAILABLE_NO_BALLOT_FILE",
        "outsider_ballots":None,
    }
    if ballots_file:
        ballots=read_ballots(ballots_file,race_id,set(horse_ids))
        points=aggregate_ballots(ballots)
        saved={}
        for row in rec["outsider_support"]:
            if not isinstance(row,list) or len(row)!=4:
                raise ValueError("archived outsider support matrix corrupt")
            hid,p,v,b=row
            hid=str(hid)
            if hid in saved or hid not in per_horse:
                raise ValueError("duplicate/unknown archived outsider support horse "+hid)
            saved[hid]={"points":int(p),"votes":int(v),"best":int(b)}
        if dict(points)!=saved:
            raise ValueError("OUTSIDER_VOTE_DRIFT: ballots disagree with original council archive")
        report["outsider_status"]="VERIFIED_SAVED_BALLOTS"
        report["outsider_ballots"]={
            name:ids for name,ids in sorted(ballots.items())
        }
        for row in report["horses"]:
            hid=row["horse_id"]
            row["outsider_points"]=points.get(hid,{}).get("points",0)
            row["outsider_votes"]=points.get(hid,{}).get("votes",0)
            row["outsider_best_vote_position"]=points.get(hid,{}).get("best",None)
            row["outsider_top3_from_each"]={
                name:(ids.index(hid)+1 if hid in ids else None)
                for name,ids in ballots.items()
            }
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--race",required=True)
    p.add_argument("--council-file",required=True)
    p.add_argument("--ballots-file")
    p.add_argument("--output")
    a=p.parse_args()
    if len(a.race)!=12 or not a.race.isascii() or not a.race.isdigit():
        raise ValueError("canonical 12-digit race id required")
    if int(a.race[:4])>=2026:
        raise ValueError("2026 sealed from historical research replay")
    result=replay(a.race,a.council_file,a.ballots_file)
    txt=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    if a.output:
        dst=Path(a.output);dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_text(txt,encoding="utf-8")
    print(txt)

if __name__=="__main__":main()
