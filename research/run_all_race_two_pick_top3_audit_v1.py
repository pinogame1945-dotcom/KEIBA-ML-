#!/usr/bin/env python3
"""Audit top-3 finish of market favorite vs ANCHOR_FULL value pick.
Only reuse persisted OOS data; no betting strategy training or optimization.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import numpy as np
import pandas as pd

YEARS=(2023,2024,2025)
PER_YEAR=3456
TOTAL=10368
BOTH_WIN_MISS=5951
CATS=("BOTH_TOP3","MARKET_ONLY_TOP3","AI_ONLY_TOP3","NEITHER_TOP3")

def require(x,msg):
    if not bool(x):
        raise ValueError(msg)

def read_data(a):
    # Single local read of each persisted data source; no Kaggle downloads.
    picks=pd.read_csv(a.picks,compression="infer",usecols=[
        "year","_race_id","_horse_id","strategy","target_win"],
        dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    predictions=pd.read_csv(a.predictions,compression="infer",usecols=[
        "year","_race_id","_horse_id","target_top3","target_win"],
        dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    for label,df in (("picks",picks),("predictions",predictions)):
        df["year"]=pd.to_numeric(df["year"],errors="raise").astype(int)
        require(not df["year"].eq(2026).any(),f"{label}: 2026 sealed")
        require(set(df["year"].unique())==set(YEARS),f"{label}: bad years")
        require(not df[["year","_race_id","_horse_id"]].isna().any().any(),
                f"{label}: null IDs")
    for col in ("target_top3","target_win"):
        predictions[col]=pd.to_numeric(predictions[col],errors="raise")
        require(predictions[col].isin((0,1)).all(),f"{col}: bad labels")
    require(predictions[["year","_race_id","_horse_id"]].duplicated().sum()==0,
            "duplicate predictions")
    require(not ((predictions["target_win"]==1) &
                  (predictions["target_top3"]!=1)).any(),"winner not top3")
    picks=picks[picks["strategy"].isin(("MARKET_FAVORITE","ANCHOR_FULL"))].copy()
    require(len(picks)==TOTAL*2,"selected picks not full coverage")
    require(not picks[["year","_race_id","strategy"]].duplicated().any(),
            "duplicate picks")
    joined=picks.merge(predictions,on=["year","_race_id","_horse_id"],
                       how="left",validate="many_to_one",
                       suffixes=("_pick","_oos"),indicator=True)
    require(joined["_merge"].eq("both").all(),"missing pick in OOS source")
    require(joined["target_win_pick"].eq(joined["target_win_oos"]).all(),
            "picked winner target drift")
    for y in YEARS:
        for s in ("MARKET_FAVORITE","ANCHOR_FULL"):
            n=joined.loc[joined.year.eq(y)&joined.strategy.eq(s),"_race_id"].nunique()
            require(n==PER_YEAR,f"missing {s} picks for {y}")
    return joined

def process_year(y,joined):
    d=joined[joined.year.eq(y)]
    m=d[d.strategy.eq("MARKET_FAVORITE")][["_race_id","_horse_id","target_win_pick","target_top3"]]
    a=d[d.strategy.eq("ANCHOR_FULL")][["_race_id","_horse_id","target_win_pick","target_top3"]]
    m=m.rename(columns={"_horse_id":"horse_market","target_win_pick":"market_win",
                        "target_top3":"market_top3"})
    a=a.rename(columns={"_horse_id":"horse_ai","target_win_pick":"ai_win",
                        "target_top3":"ai_top3"})
    result=m.merge(a,on="_race_id",how="inner",validate="one_to_one")
    require(len(result)==PER_YEAR,f"{y}: race count")
    result["year"]=y
    result["same_horse"]=result.horse_market.eq(result.horse_ai)
    result["both_win_miss"]=result.market_win.eq(0)&result.ai_win.eq(0)
    result["either_top3"]=result.market_top3.eq(1)|result.ai_top3.eq(1)
    result["both_top3"]=result.market_top3.eq(1)&result.ai_top3.eq(1)
    result["class_top3"]=np.select([
        result.both_top3,
        result.market_top3.eq(1)&result.ai_top3.eq(0),
        result.market_top3.eq(0)&result.ai_top3.eq(1)],
        CATS[:3],default=CATS[3])
    require(result.loc[result.same_horse,"market_top3"].eq(
        result.loc[result.same_horse,"ai_top3"]).all(),"same horse label drift")
    return result

def calc_group(yr,scope,agree,data):
    n=len(data)
    if n==0:return None
    row={"year":yr,"scope":scope,"selection_relation":agree,"races":n,
         "market_top3_races":int(data.market_top3.sum()),
         "ai_top3_races":int(data.ai_top3.sum()),
         "either_top3_races":int(data.either_top3.sum()),
         "either_top3_pct":round(100*data.either_top3.mean(),6),
         "both_top3_races":int(data.both_top3.sum()),
         "both_top3_pct":round(100*data.both_top3.mean(),6),
         "neither_top3_races":int((~data.either_top3).sum()),
         "neither_top3_pct":round(100*(~data.either_top3).mean(),6),
         "same_horse_races":int(data.same_horse.sum())}
    counts=data.class_top3.value_counts()
    for cat in CATS:
        row[cat+"_races"]=int(counts.get(cat,0))
        row[cat+"_pct"]=round(100*counts.get(cat,0)/n,6)
    require(sum(row[c+"_races"] for c in CATS)==n,"class partition mismatch")
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--picks",required=True)
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--out-dir",required=True)
    opts=ap.parse_args()
    out=Path(opts.out_dir);out.mkdir(parents=True,exist_ok=True)
    joined=read_data(opts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        chunks=list(pool.map(lambda y:process_year(y,joined),YEARS))
    races=pd.concat(chunks,ignore_index=True)
    require(len(races)==TOTAL and races["_race_id"].nunique()==TOTAL,"race coverage")
    require(int(races.both_win_miss.sum())==BOTH_WIN_MISS,"5951 miss baseline drift")
    require(int(races.market_win.sum())==3488,"market win baseline drift")
    require(int(races.ai_win.sum())==1917,"AI win baseline drift")
    require(int(races.same_horse.sum())==2340,"same horse baseline drift")
    report=[]
    for label,z in [("2023-2025",races)]+[(str(y),races[races.year.eq(y)]) for y in YEARS]:
        for scope,s in [("ALL",z),("BOTH_WIN_MISS",z[z.both_win_miss])]:
            for agreement,sub in [("ALL",s),("SAME",s[s.same_horse]),
                                  ("DIFFERENT",s[~s.same_horse])]:
                row=calc_group(label,scope,agreement,sub)
                if row:report.append(row)
    result=pd.DataFrame(report)
    result.to_csv(out/"top3-miss-capture-by-year.csv",index=False)
    races.to_csv(out/"all-races-top3-two-picks.csv.gz",index=False,compression="gzip")
    headline=result[(result.year=="2023-2025") &
                    (result.scope=="BOTH_WIN_MISS")].to_dict("records")
    summary={
      "contract":"ALL_RACE_FAVORITE_AI_VALUE_TOP3_CAPTURE_V1",
      "run_scope":"2023-2025 all races, 2026 sealed",
      "source_picks":"ALL_RACE_ONE_PICK_V1 38668174639",
      "source_finish_labels":"MARKET_FIXED_ANCHOR_RESIDUAL_V2 38659856085",
      "selected_win_misses":BOTH_WIN_MISS,
      "classification":"selected market favorite and selected AI expected-value top1 both failed to win",
      "top3_true":"OOS target_top3=1",
      "interpretation":"On both-win-miss races a top3 horse finished 2nd/3rd; this is a necessary not sufficient condition for trio-anchor.",
      "quinella_warning":"Top3 binary does not distinguish 2nd versus 3rd; exact quinella-anchor potential requires original finish positions.",
      "trio_warning":"Covering at least one horse in top3 does not prove that a ticket using a limited number of partner horses hits, nor profitability.",
      "headlines":headline,
      "model_training":False,"2026_locked":True}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("TOP3_CAPTURE_RESULTS_START",flush=True)
    print(result.to_csv(index=False),flush=True)
    print("TOP3_CAPTURE_RESULTS_END",flush=True)
    print("ALL_RACE_FAVORITE_AI_TOP3_CAPTURE_COMPLETE",flush=True)

if __name__=="__main__":main()
