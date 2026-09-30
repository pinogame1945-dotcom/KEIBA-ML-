#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import build_year_frame, write_csv
from run_l2_ticket_market_gap_v2 import add_market_features, market_feature_columns, train_rank_predict, add_ranks
from run_l2_market_gap_law_arena_v1 import (
    INPUT_YEARS, TEST_YEARS, DEV_YEARS, CONFIRM_YEAR,
    add_bins, metrics, pick_law1, candidate_group_tables, chosen_from_table,
    subset_years, union_frame, unique_vs, key_series
)

LAW_DEFS={
    "LAW2":("GAP10P",(("model_rank_band","1-3"),)),
    "LAW3":("GAP5_6",(("model_rank_band","11-15"),("l17_rank_band","21-30"))),
    "LAW4":("GAP10P",(("pair_worse_rank_band","11-14"),)),
    "LAW5":("GAP7_9",(("field_size_band","15-16"),("model_rank_band","9-10"))),
    "LAW6":("GAP7_9",(("market_rank_band","11-15"),("l17_rank_band","21-30"))),
    "LAW7":("GAP3_4",(("odds_band","40-60"),("field_size_band","13-14"))),
    "LAW8":("GAP5P",(("field_size_band","<=12"),("l17_rank_band","21-30"))),
}
OLD_INCREMENTAL={
    "LAW1": {"dev_tickets":291,"dev_roi":147.35395189003435,"confirm_tickets":68,"confirm_roi":95.88235294117646},
    "LAW2": {"dev_tickets":385,"dev_roi":133.27272727272728,"confirm_tickets":42,"confirm_roi":151.42857142857142},
    "LAW3": {"dev_tickets":777,"dev_roi":134.73616473616474,"confirm_tickets":142,"confirm_roi":36.54929577464789},
    "LAW4": {"dev_tickets":417,"dev_roi":161.3189448441247,"confirm_tickets":120,"confirm_roi":0.0},
    "LAW5": {"dev_tickets":285,"dev_roi":181.47368421052633,"confirm_tickets":24,"confirm_roi":172.08333333333334},
    "LAW6": {"dev_tickets":306,"dev_roi":167.45098039215685,"confirm_tickets":24,"confirm_roi":119.16666666666667},
    "LAW7": {"dev_tickets":622,"dev_roi":125.33762057877813,"confirm_tickets":201,"confirm_roi":97.46268656716418},
    "LAW8": {"dev_tickets":324,"dev_roi":143.9506172839506,"confirm_tickets":23,"confirm_roi":116.52173913043478},
}

def parse_args():
    p=argparse.ArgumentParser()
    for y in INPUT_YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def stress(g,label):
    m=metrics(g,label)
    wins=sorted([float(x) for x in g.loc[g["return_yen_per100"]>0,"return_yen_per100"]],reverse=True)
    ret=float(g["return_yen_per100"].sum()) if len(g) else 0.0
    stake=100.0*len(g)
    top2=sum(wins[:2])
    m["roi_after_remove_largest_2_wins_pct"]=100.0*(ret-top2)/stake if stake else None
    return m

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
        score,_=train_rank_predict(train,test,cols,94000+y)
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

    pred=pd.concat([predictions[y] for y in TEST_YEARS],ignore_index=True)
    tables=candidate_group_tables(pred)
    laws={"LAW1":pick_law1(pred)}
    for law,(gap,conds) in LAW_DEFS.items():
        laws[law]=chosen_from_table(tables,gap,conds)

    by_year=[]
    for law,g in laws.items():
        for y in TEST_YEARS:
            m=stress(g[g["year"]==y].copy(),law)
            m.update({"law_id":law,"year":y})
            by_year.append(m)

    seq=[]
    current=pd.DataFrame()
    for law in [f"LAW{i}" for i in range(1,9)]:
        g=laws[law]
        dev=subset_years(g,DEV_YEARS)
        conf=subset_years(g,(CONFIRM_YEAR,))
        base_dev=subset_years(current,DEV_YEARS) if not current.empty else pd.DataFrame()
        base_conf=subset_years(current,(CONFIRM_YEAR,)) if not current.empty else pd.DataFrame()
        inc_dev=unique_vs(dev,base_dev) if not base_dev.empty else dev.copy()
        inc_conf=unique_vs(conf,base_conf) if not base_conf.empty else conf.copy()
        dm=stress(inc_dev,law)
        cm=stress(inc_conf,law)
        old=OLD_INCREMENTAL[law]
        seq.append({
            "law_id":law,
            "incremental_dev_tickets":dm["tickets"],
            "incremental_dev_hits":dm["hits"],
            "incremental_dev_roi_pct":dm["roi_pct"],
            "incremental_dev_roi_after_remove_largest_win_pct":dm["roi_after_remove_largest_win_pct"],
            "incremental_dev_roi_after_remove_largest_2_wins_pct":dm["roi_after_remove_largest_2_wins_pct"],
            "incremental_confirm_tickets":cm["tickets"],
            "incremental_confirm_hits":cm["hits"],
            "incremental_confirm_roi_pct":cm["roi_pct"],
            "old_incremental_dev_tickets":old["dev_tickets"],
            "old_incremental_dev_roi_pct":old["dev_roi"],
            "old_incremental_confirm_tickets":old["confirm_tickets"],
            "old_incremental_confirm_roi_pct":old["confirm_roi"],
            "delta_dev_tickets":dm["tickets"]-old["dev_tickets"],
            "delta_confirm_tickets":cm["tickets"]-old["confirm_tickets"],
        })
        current=union_frame(current,g) if not current.empty else g.copy()

    base=union_frame(laws["LAW1"],laws["LAW2"])
    additions=[]
    for law in ("LAW5","LAW6","LAW8"):
        dev=unique_vs(subset_years(laws[law],DEV_YEARS),subset_years(base,DEV_YEARS))
        conf=unique_vs(subset_years(laws[law],(CONFIRM_YEAR,)),subset_years(base,(CONFIRM_YEAR,)))
        dm=stress(dev,law); cm=stress(conf,law)
        u=union_frame(base,laws[law])
        umd=stress(subset_years(u,DEV_YEARS),f"LAW1+2+{law}")
        umc=stress(subset_years(u,(CONFIRM_YEAR,)),f"LAW1+2+{law}")
        additions.append({
            "candidate_law":law,
            "unique_dev_tickets":dm["tickets"],"unique_dev_hits":dm["hits"],"unique_dev_roi_pct":dm["roi_pct"],
            "unique_confirm_tickets":cm["tickets"],"unique_confirm_hits":cm["hits"],"unique_confirm_roi_pct":cm["roi_pct"],
            "union_dev_tickets":umd["tickets"],"union_dev_roi_pct":umd["roi_pct"],
            "union_confirm_tickets":umc["tickets"],"union_confirm_roi_pct":umc["roi_pct"],
        })

    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"folds.csv",folds)
    pd.DataFrame(by_year).to_csv(out/"fixed-laws-by-year.csv",index=False)
    pd.DataFrame(seq).to_csv(out/"fixed-law-sequence-old-vs-corrected.csv",index=False)
    pd.DataFrame(additions).to_csv(out/"law568-vs-base12.csv",index=False)

    base_dev=stress(subset_years(base,DEV_YEARS),"LAW1+LAW2_DEV")
    base_conf=stress(subset_years(base,(CONFIRM_YEAR,)),"LAW1+LAW2_2025")
    all568=base.copy()
    for law in ("LAW5","LAW6","LAW8"):
        all568=union_frame(all568,laws[law])
    summary={
        "contract":"L2_QUINELLA_FIXED_LAWS_COMMA_REVALIDATION_V1",
        "decoder_fix":"comma-formatted numeric odds are normalized before float conversion",
        "fixed_law_conditions_changed":False,
        "law_counts":{law:{str(y):int((g["year"]==y).sum()) for y in TEST_YEARS} for law,g in laws.items()},
        "base_law1_law2":{"discovery":base_dev,"confirm_2025":base_conf},
        "base_plus_law568":{
            "discovery":stress(subset_years(all568,DEV_YEARS),"LAW1+2+5+6+8_DEV"),
            "confirm_2025":stress(subset_years(all568,(CONFIRM_YEAR,)),"LAW1+2+5+6+8_2025"),
        },
        "2026_locked":True
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_QUINELLA_FIXED_LAWS_COMMA_REVALIDATION_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
