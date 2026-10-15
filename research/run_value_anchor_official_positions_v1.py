#!/usr/bin/env python3
"""Historical official finishing position of the previously frozen OOS selections.
Only reports post-race outcomes. No model fit, no decision rule selected from results.
All 2023-2025 races are included; 2026 sealed.
"""
import argparse
import concurrent.futures
import json
from collections import Counter
from pathlib import Path
import pandas as pd

from run_all_race_flow_ticket_audit_v1 import YEARS, RACES_YEAR, TOTAL_RACES, get_inputs, assert_ok
from build_l2_market_helpers_v1 import load_day

PACK_SOURCE_SHA="0d4af461b77f0060dd9c789dd2da646026fada47"
RANK_BUCKETS=("1着","2着","3着","4着以下","着順なし")
SCOPE_NAMES=("ALL","AXIS_NOT_FIRST","BOTH_WIN_MISS")
SELECTIONS=("AI_VALUE","MARKET_FAVORITE")

def bucket(position):
    if position is None:
        return "着順なし"
    if position==1:
        return "1着"
    if position==2:
        return "2着"
    if position==3:
        return "3着"
    if position>=4:
        return "4着以下"
    raise ValueError("invalid official finish position "+str(position))

def positions(pack):
    out={}
    for row in pack.get("results") or []:
        h=str(row.get("horse_id") or "").strip()
        if not h:
            continue
        if h in out:
            raise ValueError("duplicate result horse_id "+h)
        raw=row.get("official_finish_position")
        try:
            pos=int(raw) if raw is not None and str(raw).strip() else None
        except (TypeError,ValueError):
            pos=None
        if pos is not None and pos<=0:
            pos=None
        out[h]=pos
    return out

def parse_year(y, df, picks, root):
    d=df[df.year.eq(y)]
    race_groups=d.groupby("_race_id",sort=False)
    dates=d[["_race_id","_race_date"]].drop_duplicates().copy()
    assert_ok(not dates.duplicated("_race_id").any(),f"{y} inconsistent race date")
    days=dates.groupby("_race_date",sort=True)["_race_id"].apply(list)
    rows=[]
    missing=[]
    for date,rids in days.items():
        path=root/"data"/"daily"/f"{date}.jsonl.gz"
        day=load_day(path,set(rids))
        for rid in rids:
            pack=day.get(rid)
            if pack is None:
                missing.append((rid,date))
                continue
            g=race_groups.get_group(rid)
            hlabels=g.set_index("_horse_id")[["target_win","target_top3"]]
            pos=positions(pack)
            ids={
              "AI_VALUE":picks["ANCHOR_FULL"][rid],
              "MARKET_FAVORITE":picks["MARKET_FAVORITE"][rid]
            }
            row={
              "year":y,"race_id":rid,"race_date":date,
              "same_selection":int(ids["AI_VALUE"]==ids["MARKET_FAVORITE"])
            }
            for selector in SELECTIONS:
                hid=ids[selector]
                assert_ok(hid in hlabels.index and hid in pos,
                          f"{rid} selected horse missing {selector} horse={hid}")
                finish=pos[hid]
                win=int(hlabels.loc[hid,"target_win"])
                top3=int(hlabels.loc[hid,"target_top3"])
                assert_ok(win==int(finish==1),f"{rid} {selector} win mismatch "
                          f"pred={win} official={finish}")
                assert_ok(top3==int(finish is not None and finish<=3),
                          f"{rid} {selector} top3 mismatch "
                          f"pred={top3} official={finish}")
                row[selector+"_id"]=hid
                row[selector+"_position"]=finish
                row[selector+"_bucket"]=bucket(finish)
                row[selector+"_is_win"]=win
            row["both_win_miss"]=int(not(row["AI_VALUE_is_win"] or row["MARKET_FAVORITE_is_win"]))
            rows.append(row)
    assert_ok(not missing, f"{y} missing daily packs: {missing[:10]}")
    assert_ok(len(rows)==RACES_YEAR, f"{y}: bad race count {len(rows)}")
    return rows

def sum_segment(group,year,scope,selection):
    n=len(group)
    if not n:
        return None
    counts=group[selection+"_bucket"].value_counts()
    row={"year":year,"scope":scope,"selector":selection,"races":n}
    for value in RANK_BUCKETS:
        row[value+"_races"]=int(counts.get(value,0))
        row[value+"_pct"]=round(100*int(counts.get(value,0))/n,6)
    row["top2_races"]=row["1着_races"]+row["2着_races"]
    row["top2_pct"]=round(100*row["top2_races"]/n,6)
    row["top3_races"]=row["top2_races"]+row["3着_races"]
    row["top3_pct"]=round(100*row["top3_races"]/n,6)
    row["fourth_or_nonfinish_races"]=row["4着以下_races"]+row["着順なし_races"]
    assert_ok(sum(row[b+"_races"] for b in RANK_BUCKETS)==n,"bucket total wrong")
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--picks",required=True)
    ap.add_argument("--backfill-root",required=True)
    ap.add_argument("--out-dir",required=True)
    args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    pred,picks=get_inputs(args.predictions,args.picks)
    dates=pd.read_csv(args.predictions,compression="infer",
        usecols=["year","_race_id","_horse_id","_race_date"],
        dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    assert_ok(not dates["_race_date"].isna().any(),"missing OOS race date")
    dates["_race_date"]=dates["_race_date"].astype(str).str[:10]
    assert_ok(dates["_race_date"].str.match(r"^\d{4}-\d\d-\d\d$").all(),
              "bad OOS date")
    pred=pred.merge(dates,on=["year","_race_id","_horse_id"],validate="one_to_one")
    assert_ok(pred["_race_id"].nunique()==TOTAL_RACES,"OOS race count changed")
    root=Path(args.backfill_root)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        chunks=list(executor.map(lambda y:parse_year(y,pred,picks,root),YEARS))
    races=pd.DataFrame([row for chunk in chunks for row in chunk])
    assert_ok(len(races)==TOTAL_RACES and races["race_id"].is_unique,"race coverage")
    assert_ok(races.groupby("year").size().to_dict()=={x:RACES_YEAR for x in YEARS},
              "yearly coverage wrong")
    assert_ok(races["AI_VALUE_is_win"].sum()==1917,"AI winner baseline drift")
    assert_ok(races["MARKET_FAVORITE_is_win"].sum()==3488,"market winner baseline drift")
    assert_ok(races["both_win_miss"].sum()==5951,"both-win-miss baseline drift")
    assert_ok(races["same_selection"].sum()==2340,"same selection baseline drift")
    assert_ok(races["AI_VALUE_bucket"].eq("1着").sum()==1917,
              "AI 1st count mismatch")
    output=[]
    for year,period in [("2023-2025",races)]+[(str(y),races[races.year.eq(y)]) for y in YEARS]:
        for name,segment in [
            ("ALL",period),
            ("AXIS_NOT_FIRST",period[period["AI_VALUE_is_win"].eq(0)]),
            ("BOTH_WIN_MISS",period[period["both_win_miss"].eq(1)])
        ]:
            for selection in SELECTIONS:
                output.append(sum_segment(segment,year,name,selection))
    table=pd.DataFrame(output)
    table.to_csv(out/"official-position-by-year-and-failure-scope.csv",
                 index=False,encoding="utf-8")
    races.to_csv(out/"selected-horse-official-positions.csv.gz",
                 index=False,compression="gzip")
    overall=table[(table["year"]=="2023-2025") &
                  (table["selector"]=="AI_VALUE")].to_dict("records")
    summary={
      "contract":"VALUE_ANCHOR_OFFICIAL_POSITION_V1",
      "historical_scope":"2023-2025, all 10,368 OOS races, same frozen AI EV selections as exacta tests",
      "results_source_sha":PACK_SOURCE_SHA,
      "actual_position_source":"BACKFILL race.results[].official_finish_position",
      "predictions_source":"market-fixed-anchor-residual-v2 run 38659856085",
      "picks_source":"all-race-one-pick-v1 run 38668174639",
      "race_count":TOTAL_RACES,"ai_value_1st":1917,
      "ai_value_not_first":8451,
      "market_favorite_1st":3488,"both_win_miss":5951,
      "rank_buckets":list(RANK_BUCKETS),
      "headline_ai_value":overall,
      "other_nonfinish_note":"No official finish is reported as 着順なし, not silently assigned rank 4+.",
      "selection_leakage_note":"Selection uses final win odds retrospectively. Finish labels used only after selection.",
      "2026_locked":True,"model_retraining":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("VALUE_ANCHOR_POSITION_BREAKDOWN_START",flush=True)
    print(table.to_csv(index=False),flush=True)
    print("VALUE_ANCHOR_POSITION_BREAKDOWN_END",flush=True)
    print("VALUE_ANCHOR_POSITION_AUDIT_COMPLETE",flush=True)

if __name__=="__main__":
    main()
