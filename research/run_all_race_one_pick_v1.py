#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import numpy as np
import pandas as pd

YEARS=(2023,2024,2025)
EXPECTED_RACES=3456
PRIMARY="ANCHOR_ABILITY_RISK"
MODEL_COLS={
    "ANCHOR_ABILITY":"P__ANCHOR_ABILITY",
    "ANCHOR_ABILITY_RISK":"P__ANCHOR_ABILITY_RISK",
    "ANCHOR_FULL":"P__ANCHOR_FULL",
}

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def pick_favorite(df):
    q=df.copy()
    q["__market_rank"]=pd.to_numeric(q["market_rank"],errors="coerce")
    q["__odds"]=pd.to_numeric(q["final_win_odds"],errors="coerce")
    q=q.sort_values(["_race_id","__market_rank","__odds","_horse_id"],
                    ascending=[True,True,True,True])
    return q.groupby("_race_id",sort=False).head(1).copy()

def pick_value(df,pcol):
    q=df.copy()
    q["__p"]=pd.to_numeric(q[pcol],errors="coerce")
    q["__odds"]=pd.to_numeric(q["final_win_odds"],errors="coerce")
    q["__value"]=q["__p"]*q["__odds"]
    q["__market_rank"]=pd.to_numeric(q["market_rank"],errors="coerce")
    if q[["__p","__odds","__market_rank"]].isna().any().any():
        raise SystemExit(f"missing prediction/odds/rank for {pcol}")
    q=q.sort_values(
        ["_race_id","__value","__p","__market_rank","_horse_id"],
        ascending=[True,False,False,True,True],
    )
    return q.groupby("_race_id",sort=False).head(1).copy()

def summarize(picks,fav_ids,strategy,year):
    n=int(len(picks))
    if n==0: raise SystemExit(f"empty picks {strategy} {year}")
    win=pd.to_numeric(picks["target_win"],errors="coerce").fillna(0).astype(int)
    odds=pd.to_numeric(picks["final_win_odds"],errors="coerce").astype(float)
    mr=pd.to_numeric(picks["market_rank"],errors="coerce").astype(float)
    stake=100.0*n
    ret=float((win*odds*100.0).sum())
    ids=picks.set_index("_race_id")["_horse_id"].astype(str)
    fav=fav_ids.reindex(ids.index).astype(str)
    switched=(ids!=fav)
    edge=None
    if "__value" in picks.columns:
        edge=(pd.to_numeric(picks["__value"],errors="coerce")-1.0)*100.0
    return {
        "year":year,
        "strategy":strategy,
        "primary":int(strategy==PRIMARY),
        "races":n,
        "bets":n,
        "wins":int(win.sum()),
        "hit_pct":100.0*float(win.mean()),
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake,
        "avg_final_odds":float(odds.mean()),
        "median_final_odds":float(odds.median()),
        "avg_market_rank":float(mr.mean()),
        "median_market_rank":float(mr.median()),
        "favorite_selected_pct":100.0*float(mr.eq(1).mean()),
        "market_top3_selected_pct":100.0*float(mr.le(3).mean()),
        "switch_from_favorite_pct":100.0*float(switched.mean()),
        "avg_selected_edge_pct":float(edge.mean()) if edge is not None else np.nan,
        "median_selected_edge_pct":float(edge.median()) if edge is not None else np.nan,
    }

def rank_bucket(r):
    r=int(r)
    if r==1:return "1"
    if r==2:return "2"
    if r==3:return "3"
    if r<=6:return "4-6"
    if r<=10:return "7-10"
    return "11+"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--out-dir",required=True)
    a=ap.parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(a.predictions,compression="infer",low_memory=False)
    required={"year","_race_id","_horse_id","target_win","final_win_odds","market_rank",*MODEL_COLS.values()}
    missing=required-set(df.columns)
    if missing: raise SystemExit(f"missing columns: {sorted(missing)}")
    if 2026 in set(pd.to_numeric(df["year"],errors="coerce").dropna().astype(int)):
        raise SystemExit("2026 sealed")
    df["year"]=pd.to_numeric(df["year"],errors="raise").astype(int)
    df=df[df["year"].isin(YEARS)].copy()
    if int(df["_race_id"].nunique()) != EXPECTED_RACES*len(YEARS):
        raise SystemExit(f"race total mismatch: {df['_race_id'].nunique()}")

    rows=[]
    rank_rows=[]
    chosen=[]
    for year in YEARS:
        y=df[df["year"].eq(year)].copy()
        races=int(y["_race_id"].nunique())
        if races!=EXPECTED_RACES:
            raise SystemExit(f"year={year} race mismatch expected={EXPECTED_RACES} got={races}")
        fav=pick_favorite(y)
        if len(fav)!=EXPECTED_RACES:
            raise SystemExit(f"favorite pick coverage drift y={year}")
        fav_ids=fav.set_index("_race_id")["_horse_id"].astype(str)
        fav["strategy"]="MARKET_FAVORITE"
        fav["selected_value"]=np.nan
        chosen.append(fav[["year","_race_id","_horse_id","strategy","target_win","final_win_odds","market_rank","selected_value"]])
        rows.append(summarize(fav,fav_ids,"MARKET_FAVORITE",year))
        for strategy,pcol in MODEL_COLS.items():
            p=pick_value(y,pcol)
            if len(p)!=EXPECTED_RACES:
                raise SystemExit(f"AI pick coverage drift y={year} strategy={strategy}")
            p["strategy"]=strategy
            p["selected_value"]=p["__value"]
            chosen.append(p[["year","_race_id","_horse_id","strategy","target_win","final_win_odds","market_rank","selected_value"]])
            rows.append(summarize(p,fav_ids,strategy,year))
            b=p.assign(rank_bucket=pd.to_numeric(p["market_rank"],errors="raise").astype(int).map(rank_bucket))
            counts=b["rank_bucket"].value_counts()
            for bucket in ("1","2","3","4-6","7-10","11+"):
                rank_rows.append({
                    "year":year,"strategy":strategy,"market_rank_bucket":bucket,
                    "picks":int(counts.get(bucket,0)),
                    "pick_pct":100.0*float(counts.get(bucket,0))/EXPECTED_RACES,
                })

    # Pooled metrics from all 10,368 mandatory bets.
    chosen_df=pd.concat(chosen,ignore_index=True)
    for strategy in ("MARKET_FAVORITE",*MODEL_COLS.keys()):
        p=chosen_df[chosen_df["strategy"].eq(strategy)].copy()
        fav_all=chosen_df[chosen_df["strategy"].eq("MARKET_FAVORITE")].set_index("_race_id")["_horse_id"].astype(str)
        rows.append(summarize(p,fav_all,strategy,"2023-2025"))

    result=pd.DataFrame(rows)
    base=result[result["strategy"].eq("MARKET_FAVORITE")][["year","roi_pct","hit_pct"]].rename(
        columns={"roi_pct":"market_favorite_roi_pct","hit_pct":"market_favorite_hit_pct"}
    )
    comp=result.merge(base,on="year",how="left")
    comp["roi_uplift_pp"]=comp["roi_pct"]-comp["market_favorite_roi_pct"]
    comp["hit_uplift_pp"]=comp["hit_pct"]-comp["market_favorite_hit_pct"]

    write_csv(out/"all-race-one-pick-summary.csv",comp.to_dict("records"))
    write_csv(out/"market-rank-selection-distribution.csv",rank_rows)
    chosen_df.to_csv(out/"all-race-one-picks.csv.gz",index=False,compression="gzip")

    pooled=comp[comp["year"].astype(str).eq("2023-2025")].sort_values("roi_pct",ascending=False)
    primary=pooled[pooled["strategy"].eq(PRIMARY)].iloc[0].to_dict()
    baseline=pooled[pooled["strategy"].eq("MARKET_FAVORITE")].iloc[0].to_dict()
    summary={
        "contract":"ALL_RACE_ONE_PICK_V1",
        "objective":"Beat market favorite ROI while betting exactly one 100-yen win ticket in every race.",
        "no_skip":True,
        "no_edge_threshold":True,
        "no_odds_filter":True,
        "years":[2023,2024,2025],
        "races_per_year":EXPECTED_RACES,
        "pooled_races":EXPECTED_RACES*len(YEARS),
        "stake_per_race_yen":100,
        "selection_rule":"For each anchor model, choose horse maximizing predicted win probability * final win odds.",
        "primary_strategy":PRIMARY,
        "primary_reason":"Chosen before ROI comparison because ANCHOR_ABILITY_RISK had the best pooled probability logloss among anchor residual variants.",
        "market_favorite_pooled_roi_pct":baseline["roi_pct"],
        "primary_pooled_roi_pct":primary["roi_pct"],
        "primary_roi_uplift_pp":primary["roi_uplift_pp"],
        "payout_note":"Return uses final decimal win odds * 100 on winners, consistent with the existing Fair Value study; exact payout audit is separate.",
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== ALL-RACE ONE-PICK =====")
    print(comp.sort_values(["year","strategy"]).to_csv(index=False),flush=True)
    print("ALL_RACE_ONE_PICK_V1_COMPLETE",flush=True)

if __name__=="__main__":
    main()
