#!/usr/bin/env python3
"""Read existing 7 king horse-level historical score snapshots; no new training."""
import argparse,gzip,json,sys
from pathlib import Path

EXPERTS=("core4","pedlegacy","condition","full","light","pedv1","condrc")
CONTRACT="L1_TO_L2_OUTPUT_CONTRACT_V1"

def read(path):
    with (gzip.open(path,"rt",encoding="utf-8") if path.endswith(".gz") else open(path,encoding="utf-8")) as f:
        for line in f:
            if line.strip():yield json.loads(line)

def load_one(path,race_id):
    result={}
    for row in read(path):
        if str(row.get("race_id") or "")!=race_id:continue
        if row.get("contract")!=CONTRACT:
            raise ValueError("unexpected score contract "+str(row.get("contract")))
        hid=str(row.get("horse_id") or "")
        rank=int(row.get("predicted_rank") or 0)
        if not hid or rank<1 or hid in result:raise ValueError("bad/doubled horse score "+hid)
        if row.get("race_date") and str(row["race_date"])[:4]!=race_id[:4]:
            raise ValueError("score date/year drift")
        result[hid]={"rank":rank,"horse_no":row.get("horse_number")}
    if not result:raise ValueError("no saved score for target race "+race_id+" in "+path)
    if sorted(item["rank"] for item in result.values())!=list(range(1,len(result)+1)):
        raise ValueError("non-contiguous rank score file: "+path)
    return result

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
                         "_tie_key":key})
    rankings.sort(key=lambda row:row["_tie_key"])
    for i,row in enumerate(rankings,start=1):
        row["rank_sum_comparator_rank"]=i
        del row["_tie_key"]
    return {"contract":"KEIBA_ARCHIVED_7KING_ONE_RACE_V1","race_id":race_id,
            "origin":"HISTORICAL_SAVED_OOF_REPLAY_NOT_NEW_INFERENCE",
            "expert_names":list(EXPERTS),"model_count":len(EXPERTS),
            "horses":rankings,"rank_sum_comparator":
              "M127_S0 from experimental council rebuild: rank sum, top6/top3/top1 tie-breaks, horse ID. This is NOT official 7-king router or L2.1 ANCHOR_FULL.",
            "verification_note":"The individual scores were previously computed for this historical race. Model fold provenance must be audited before interpreting as true as-of forecasts."}

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
