#!/usr/bin/env python3
"""Read existing 7 king horse-level historical score snapshots; no new training."""
import argparse,gzip,json,sys,math
from pathlib import Path

EXPERTS=("core4","pedlegacy","condition","full","light","pedv1","condrc")
CONTRACT="L1_TO_L2_OUTPUT_CONTRACT_V1"

def read(path):
    with (gzip.open(path,"rt",encoding="utf-8") if path.endswith(".gz") else open(path,encoding="utf-8")) as f:
        for line in f:
            if line.strip():yield json.loads(line)

def finite(value):
    try:
        result=float(value)
        return result if math.isfinite(result) else None
    except (ValueError,TypeError):
        return None

def load_one(path,race_id):
    rows={}
    for row in read(path):
        if str(row.get("race_id") or "")!=race_id:continue
        if row.get("contract")!=CONTRACT:
            raise ValueError("unexpected score contract "+str(row.get("contract")))
        hid=str(row.get("horse_id") or "")
        if not hid or hid in rows:
            raise ValueError("bad/doubled horse score "+hid)
        if row.get("race_date") and str(row["race_date"])[:4]!=race_id[:4]:
            raise ValueError("score date/year drift")
        phase=str(row.get("prediction_phase") or "")
        if phase!="FINAL":
            raise ValueError("archived seven-king source must declare FINAL research phase")
        if row.get("target") is not None or row.get("market_outcome") is not None:
            raise ValueError("model scoring must not read labels")
        # Apply exact 2026-10-04 rerank_l1_score_safe_v1.py rule, never
        # trust predicted_rank tie order from the archive producer.
        margin=finite(row.get("raw_margin_logit"))
        prob=finite(row.get("raw_win_probability"))
        score=margin if margin is not None else prob
        if score is None:raise ValueError("raw score unavailable for tie-safe re-ranking")
        rows[hid]={
            "score":score,"model_raw_margin_logit":margin,
            "model_raw_win_probability":prob,
            "race_normalized_win_probability":finite(row.get("race_normalized_win_probability")),
            "horse_no":row.get("horse_number"),
            "model_sha256":str(row.get("model_sha256") or ""),
            "contribution_family":row.get("family_contribution_logit"),
            "historical_unsafely_tied_rank":row.get("predicted_rank"),
        }
    if not rows:raise ValueError("no saved score for target race "+race_id+" in "+path)
    ordered=sorted(rows,key=lambda hid:(-rows[hid]["score"],hid))
    for index,hid in enumerate(ordered,start=1):
        rows[hid]["rank"]=index
    return rows

def predict(race_id,paths):
    data={name:load_one(paths[name],race_id) for name in EXPERTS}
    horse_sets=[set(rows) for rows in data.values()]
    if any(s!=horse_sets[0] for s in horse_sets[1:]):
        raise ValueError("cross-king horse coverage mismatch: cannot compare or build council")
    horse_ids=sorted(horse_sets[0])
    rankings=[]
    for hid in horse_ids:
        ranks={e:data[e][hid]["rank"] for e in EXPERTS}
        horse_nums={data[e][hid]["horse_no"] for e in EXPERTS if data[e][hid]["horse_no"] is not None}
        if len(horse_nums)>1:raise ValueError("cross-king horse number mismatch "+hid)
        arr=list(ranks.values())
        # Pure rank-sum 7-way comparator from run_l1_council_rebuild_v1.py
        # with mask=127, market_share=0. NOT claimed to be the frozen 7-king router.
        key=(sum(arr),-sum(r<=6 for r in arr),-sum(r<=3 for r in arr),
             -sum(r==1 for r in arr),hid)
        rankings.append({"horse_id":hid,"horse_number":next(iter(horse_nums),None),
                         "king_ranks":ranks,"rank_sum":sum(arr),
                         "king_raw_probabilities":{e:data[e][hid]["model_raw_win_probability"] for e in EXPERTS},
                         "king_normalized_probabilities":{e:data[e][hid]["race_normalized_win_probability"] for e in EXPERTS},
                         "king_score_logit":{e:data[e][hid]["model_raw_margin_logit"] for e in EXPERTS},
                         "_tie_key":key})
    rankings.sort(key=lambda row:row["_tie_key"])
    for i,row in enumerate(rankings,start=1):
        row["rank_sum_comparator_rank"]=i
        del row["_tie_key"]
    return {"contract":"KEIBA_ARCHIVED_7KING_ONE_RACE_V1","race_id":race_id,
            "origin":"HISTORICAL_SAVED_OOF_REPLAY_NOT_NEW_INFERENCE",
            "prediction_phase":"FINAL_ARCHIVED_UNVERIFIED_TIMESTAMPS",
            "tie_safe_rule":"raw_margin_logit (fallback raw_win_probability) descending, horse_id ascending",
            "expert_names":list(EXPERTS),"model_count":len(EXPERTS),
            "horses":rankings,"rank_sum_comparator":
              "M127_S0 from experimental council rebuild: rank sum, top6/top3/top1 tie-breaks, horse ID. This is NOT official 7-king router or L2.1 ANCHOR_FULL.",
            "verification_note":"Source 7-king workflow trains on two earlier calendar years and scores the held-out year. However FINAL phase snapshots derive from completed archives and input publication times were not independently proven; not confirmed live-bet deployable."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--race",required=True)
    p.add_argument("--expert",action="append",required=True,help="alias=existing local score.jsonl[.gz]")
    p.add_argument("--output")
    a=p.parse_args()
    if not (len(a.race)==12 and a.race.isascii() and a.race.isdigit()):
        raise ValueError("12 digit race id required")
    if a.race[:4]=="2026":raise ValueError("2026 unknown-year evaluation sealed")
    paths={}
    for raw in a.expert:
        name,sep,path=raw.partition("=")
        if not sep or name in paths:raise ValueError("malformed/double --expert "+raw)
        paths[name]=path
    if set(paths)!=set(EXPERTS):raise ValueError("seven king source completeness violation")
    for path in paths.values():
        if not Path(path).is_file():raise FileNotFoundError(path)
    report=predict(a.race,paths)
    result=json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    if a.output:
        p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(result,encoding="utf-8")
    print(result)

if __name__=="__main__":main()
