#!/usr/bin/env python3
import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import build_year_frame, write_csv
from run_l2_ticket_market_gap_v2 import (
    add_market_features, market_feature_columns, train_rank_predict, add_ranks
)

INPUT_YEARS=(2021,2022,2023,2024,2025)
TEST_YEARS=(2022,2023,2024,2025)
DEV_YEARS=(2022,2023,2024)
CONFIRM_YEAR=2025
MODEL_TOP_K=15

GAP_SPECS={
    "GAP0_2":(0,3),
    "GAP3_4":(3,5),
    "GAP5_6":(5,7),
    "GAP7_9":(7,10),
    "GAP10_14":(10,15),
    "GAP15_24":(15,25),
    "GAP25P":(25,10**9),
    "GAP5P":(5,10**9),
    "GAP10P":(10,10**9),
}

DIM_VALUES={
    "field_size_band":["<=12","13-14","15-16","17+"],
    "surface":["TURF","DIRT"],
    "distance_band":["<=1400","1401-1800","1801-2200","2201+"],
    "market_rank_band":["<=10","11-15","16-20","21-25","26-30","31-40","41+"],
    "model_rank_band":["1-3","4-5","6-8","9-10","11-15"],
    "l17_rank_band":["<=10","11-20","21-30","31-40","41-60","61+"],
    "odds_band":["<20","20-40","40-60","60-80","80-100","100+"],
    "pair_worse_rank_band":["<=6","7-10","11-14","15+"],
}
PAIR_DIMS=[
    ("field_size_band","surface"),
    ("field_size_band","distance_band"),
    ("field_size_band","market_rank_band"),
    ("field_size_band","model_rank_band"),
    ("field_size_band","l17_rank_band"),
    ("surface","distance_band"),
    ("market_rank_band","model_rank_band"),
    ("market_rank_band","l17_rank_band"),
    ("model_rank_band","l17_rank_band"),
    ("odds_band","field_size_band"),
    ("odds_band","market_rank_band"),
    ("odds_band","model_rank_band"),
    ("pair_worse_rank_band","field_size_band"),
    ("pair_worse_rank_band","market_rank_band"),
]

QUAL={
    "min_tickets_each_dev_year":30,
    "min_total_dev_tickets":120,
    "min_unique_tickets_vs_law1":60,
    "max_candidate_overlap_share_with_law1_pct":60.0,
    "min_pooled_dev_roi_pct":100.0,
    "min_pooled_dev_roi_after_remove_largest_win_pct":95.0,
    "min_unique_dev_roi_pct":95.0,
}

def parse_args():
    p=argparse.ArgumentParser()
    for y in INPUT_YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def add_bins(df):
    z=df.copy()
    z["field_size_band"]=pd.cut(
        z["field_size"],[-1,12,14,16,10**9],
        labels=["<=12","13-14","15-16","17+"],right=True
    ).astype(str)
    z["distance_band"]=pd.cut(
        z["distance_m"],[-1,1400,1800,2200,10**9],
        labels=["<=1400","1401-1800","1801-2200","2201+"],right=True
    ).astype(str)
    z["market_rank_band"]=pd.cut(
        z["market_rank"],[-1,10,15,20,25,30,40,10**9],
        labels=["<=10","11-15","16-20","21-25","26-30","31-40","41+"],right=True
    ).astype(str)
    z["model_rank_band"]=pd.cut(
        z["model_rank"],[-1,3,5,8,10,15],
        labels=["1-3","4-5","6-8","9-10","11-15"],right=True
    ).astype(str)
    z["l17_rank_band"]=pd.cut(
        z["l17_rank_score"],[-1,10,20,30,40,60,10**9],
        labels=["<=10","11-20","21-30","31-40","41-60","61+"],right=True
    ).astype(str)
    z["odds_band"]=pd.cut(
        z["odds"],[-1,20,40,60,80,100,10**9],
        labels=["<20","20-40","40-60","60-80","80-100","100+"],right=False
    ).astype(str)
    z["pair_worse_rank_band"]=pd.cut(
        z["pair_worse_rank"],[-1,6,10,14,10**9],
        labels=["<=6","7-10","11-14","15+"],right=True
    ).astype(str)
    return z

def sort_for_pick(df):
    return df.sort_values(
        ["year","race_date","race_id","rank_upgrade","model_rank","market_rank","market_aware_score","pair_numbers"],
        ascending=[True,True,True,False,True,False,False,True]
    )

def metrics(g,label):
    n=len(g); stake=100.0*n
    ret=float(g["return_yen_per100"].sum()) if n else 0.0
    hits=int(g["hit"].sum()) if n else 0
    wins=sorted([float(x) for x in g.loc[g["return_yen_per100"]>0,"return_yen_per100"]],reverse=True)
    largest=wins[0] if wins else 0.0
    return {
        "label":label,
        "tickets":n,
        "hits":hits,
        "hit_rate_pct":100.0*hits/n if n else 0.0,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "largest_single_return_yen":largest,
        "largest_return_share_pct":100.0*largest/ret if ret>0 else 0.0,
        "roi_after_remove_largest_win_pct":100.0*(ret-largest)/stake if stake else None,
        "median_odds":float(g["odds"].median()) if n else None,
    }

def key_series(g):
    return g["year"].astype(str)+"|"+g["race_id"].astype(str)+"|"+g["pair_numbers"].astype(str)

def pick_law1(pred):
    # Reproduce the previously audited frozen V2 policy exactly:
    # first select at most one ticket/race from MODEL_TOP15 + UPGRADE>=5,
    # then classify that frozen ticket into law1 (GAP15-24 x field 15-16).
    base=pred[
        (pred["model_rank"]<=15)&
        (pred["rank_upgrade"]>=5)
    ].copy()
    base=sort_for_pick(base).groupby(["year","race_id"],sort=False).head(1).copy()
    return base[
        (base["rank_upgrade"]>=15)&(base["rank_upgrade"]<25)&
        (base["field_size"]>=15)&(base["field_size"]<=16)
    ].copy()

def candidate_group_tables(pred):
    tables={}
    for gap,(lo,hi) in GAP_SPECS.items():
        base=pred[
            (pred["model_rank"]<=MODEL_TOP_K)&
            (pred["rank_upgrade"]>=lo)&(pred["rank_upgrade"]<hi)
        ].copy()
        if base.empty:
            continue
        base=sort_for_pick(base)
        tables[(gap,())]=base.groupby(["year","race_id"],sort=False).head(1).copy()
        for dim in DIM_VALUES:
            tables[(gap,(dim,))]=base.groupby(
                ["year","race_id",dim],sort=False,dropna=False
            ).head(1).copy()
        for d1,d2 in PAIR_DIMS:
            tables[(gap,(d1,d2))]=base.groupby(
                ["year","race_id",d1,d2],sort=False,dropna=False
            ).head(1).copy()
    return tables

def candidate_specs():
    specs=[]
    for gap in GAP_SPECS:
        specs.append((f"{gap}",gap,()))
        for d,vals in DIM_VALUES.items():
            for v in vals:
                specs.append((f"{gap}__{d}={v}",gap,((d,v),)))
        for d1,d2 in PAIR_DIMS:
            for v1 in DIM_VALUES[d1]:
                for v2 in DIM_VALUES[d2]:
                    specs.append((
                        f"{gap}__{d1}={v1}__{d2}={v2}",
                        gap,((d1,v1),(d2,v2))
                    ))
    return specs

def chosen_from_table(tables,gap,conds):
    dims=tuple(d for d,_ in conds)
    t=tables.get((gap,dims))
    if t is None or t.empty:
        return pd.DataFrame()
    z=t
    for d,v in conds:
        z=z[z[d].astype(str)==str(v)]
        if z.empty:
            break
    return z.copy()

def subset_years(g,years):
    return g[g["year"].isin(years)].copy()

def unique_vs(g,base):
    if g.empty:
        return g.copy()
    b=set(key_series(base).tolist())
    return g[~key_series(g).isin(b)].copy()

def intersection_count(a,b):
    if a.empty or b.empty:
        return 0
    return len(set(key_series(a)).intersection(set(key_series(b))))

def union_frame(a,b):
    if a.empty:
        return b.copy()
    if b.empty:
        return a.copy()
    z=pd.concat([a,b],ignore_index=True)
    z["_k"]=key_series(z)
    z=z.drop_duplicates("_k",keep="first").drop(columns="_k")
    return z

def main():
    a=parse_args()
    paths={y:getattr(a,f"l17_{y}") for y in INPUT_YEARS}
    l17={y:load_l17(paths[y],y) for y in INPUT_YEARS}
    frames={y:add_market_features(build_year_frame(y,l17[y],a.backfill_root)) for y in INPUT_YEARS}
    cols=market_feature_columns(frames[2022])

    predictions={}
    folds=[]
    for y in TEST_YEARS:
        train_years=[2021] if y==2022 else [t for t in (2022,2023,2024) if t<y]
        train=pd.concat([frames[t] for t in train_years],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)
        score,_=train_rank_predict(train,test,cols,93000+y)
        test["market_aware_score"]=score
        test=add_bins(add_ranks(test))
        predictions[y]=test
        folds.append({
            "test_year":y,
            "train_years":"|".join(map(str,train_years)),
            "train_pairs":len(train),
            "test_pairs":len(test),
            "source_races":int(test["race_id"].nunique()),
        })

    pred_all=pd.concat([predictions[y] for y in TEST_YEARS],ignore_index=True)
    law1=pick_law1(pred_all)
    expected_law1_counts={2022:105,2023:102,2024:84,2025:68}
    got_law1_counts={y:int((law1["year"]==y).sum()) for y in TEST_YEARS}
    if got_law1_counts!=expected_law1_counts:
        print(f"ODDS_COMMA_EXPECTED_LAW1_DRIFT got={got_law1_counts} old={expected_law1_counts}",flush=True)
    law1_dev=subset_years(law1,DEV_YEARS)
    law1_confirm=subset_years(law1,[CONFIRM_YEAR])

    tables=candidate_group_tables(pred_all)
    specs=candidate_specs()
    rows=[]; year_rows=[]
    for cid,gap,conds in specs:
        chosen=chosen_from_table(tables,gap,conds)
        if chosen.empty:
            continue
        dev=subset_years(chosen,DEV_YEARS)
        conf=subset_years(chosen,[CONFIRM_YEAR])
        by_year={}
        for y in DEV_YEARS:
            yy=chosen[chosen["year"]==y]
            m=metrics(yy,cid);m.update({"candidate_id":cid,"year":y})
            year_rows.append(m);by_year[y]=m

        pooled=metrics(dev,cid)
        confm=metrics(conf,cid)
        uniq=unique_vs(dev,law1_dev)
        uniqm=metrics(uniq,cid)
        uniq_conf=unique_vs(conf,law1_confirm)
        uniq_conf_m=metrics(uniq_conf,cid)
        overlap=intersection_count(dev,law1_dev)
        candidate_overlap_share=100.0*overlap/len(dev) if len(dev) else 0.0
        law1_overlap_share=100.0*overlap/len(law1_dev) if len(law1_dev) else 0.0
        union_dev=union_frame(dev,law1_dev)
        union_conf=union_frame(conf,law1_confirm)
        union_dev_m=metrics(union_dev,cid)
        union_conf_m=metrics(union_conf,cid)

        dev_tickets_each=[by_year[y]["tickets"] for y in DEV_YEARS]
        dev_rois_each=[by_year[y]["roi_pct"] if by_year[y]["roi_pct"] is not None else -1.0 for y in DEV_YEARS]
        qualifies=(
            min(dev_tickets_each)>=QUAL["min_tickets_each_dev_year"] and
            len(dev)>=QUAL["min_total_dev_tickets"] and
            len(uniq)>=QUAL["min_unique_tickets_vs_law1"] and
            candidate_overlap_share<=QUAL["max_candidate_overlap_share_with_law1_pct"] and
            (pooled["roi_pct"] or 0.0)>=QUAL["min_pooled_dev_roi_pct"] and
            (pooled["roi_after_remove_largest_win_pct"] or 0.0)>=QUAL["min_pooled_dev_roi_after_remove_largest_win_pct"] and
            (uniqm["roi_pct"] or 0.0)>=QUAL["min_unique_dev_roi_pct"]
        )
        rows.append({
            "candidate_id":cid,
            "gap_spec":gap,
            "conditions":";".join(f"{d}={v}" for d,v in conds) if conds else "NONE",
            "qualified":bool(qualifies),
            "dev_tickets":len(dev),
            "dev_hits":pooled["hits"],
            "dev_roi_pct":pooled["roi_pct"],
            "dev_profit_yen":pooled["profit_yen"],
            "dev_roi_after_remove_largest_win_pct":pooled["roi_after_remove_largest_win_pct"],
            "dev_largest_return_share_pct":pooled["largest_return_share_pct"],
            "dev_worst_year_roi_pct":min(dev_rois_each),
            "dev_best_year_roi_pct":max(dev_rois_each),
            "dev_min_year_tickets":min(dev_tickets_each),
            "overlap_tickets_with_law1":overlap,
            "candidate_overlap_share_with_law1_pct":candidate_overlap_share,
            "law1_overlap_share_with_candidate_pct":law1_overlap_share,
            "unique_dev_tickets_vs_law1":len(uniq),
            "unique_dev_hits_vs_law1":uniqm["hits"],
            "unique_dev_roi_pct_vs_law1":uniqm["roi_pct"],
            "unique_dev_profit_yen_vs_law1":uniqm["profit_yen"],
            "law1_union_dev_tickets":union_dev_m["tickets"],
            "law1_union_dev_roi_pct":union_dev_m["roi_pct"],
            "law1_union_dev_profit_yen":union_dev_m["profit_yen"],
            "confirm_2025_tickets":len(conf),
            "confirm_2025_hits":confm["hits"],
            "confirm_2025_roi_pct":confm["roi_pct"],
            "confirm_2025_profit_yen":confm["profit_yen"],
            "confirm_2025_unique_tickets_vs_law1":len(uniq_conf),
            "confirm_2025_unique_roi_pct_vs_law1":uniq_conf_m["roi_pct"],
            "law1_union_2025_tickets":union_conf_m["tickets"],
            "law1_union_2025_roi_pct":union_conf_m["roi_pct"],
            "law1_union_2025_profit_yen":union_conf_m["profit_yen"],
        })

    arena=pd.DataFrame(rows)
    if arena.empty:
        raise SystemExit("no candidates generated")
    arena=arena.sort_values(
        [
            "qualified","dev_worst_year_roi_pct","unique_dev_roi_pct_vs_law1",
            "dev_roi_after_remove_largest_win_pct","unique_dev_tickets_vs_law1",
            "candidate_overlap_share_with_law1_pct"
        ],
        ascending=[False,False,False,False,False,True],
        na_position="last"
    ).reset_index(drop=True)
    arena["discovery_rank"]=range(1,len(arena)+1)

    qualified=arena[arena["qualified"]].copy()
    top=qualified.head(100).copy() if not qualified.empty else arena.head(100).copy()
    top_ids=set(top["candidate_id"])
    detail=pd.DataFrame(year_rows)
    detail=detail[detail["candidate_id"].isin(top_ids)].copy()

    law1_rows=[]
    for label,g in [
        ("LAW1_DEV_2022_2024",law1_dev),
        ("LAW1_CONFIRM_2025",law1_confirm),
        ("LAW1_ALL_2022_2025",law1),
    ]:
        m=metrics(g,label);m["period"]=label;law1_rows.append(m)

    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"folds.csv",folds)
    arena.to_csv(out/"candidate-arena.csv",index=False)
    top.to_csv(out/"top-candidates.csv",index=False)
    detail.to_csv(out/"top-candidates-by-dev-year.csv",index=False)
    write_csv(out/"law1-baseline.csv",law1_rows)

    summary={
        "contract":"L2_MARKET_GAP_LAW_ARENA_V1_RESULT",
        "source_variant":"ODDS_COMMA_CORRECTED",
        "odds_decoder_numeric_normalization":"remove thousands separators before float conversion",
        "candidate_count":int(len(arena)),
        "qualified_candidate_count":int(len(qualified)),
        "discovery_years":list(DEV_YEARS),
        "confirmation_year":CONFIRM_YEAR,
        "confirmation_used_for_selection":False,
        "law1_dev":law1_rows[0],
        "law1_confirmation_2025":law1_rows[1],
        "top_discovery_candidates":top.head(20).to_dict(orient="records"),
        "qualification":QUAL,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("L2_MARKET_GAP_LAW_ARENA_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
