#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import build_year_frame
from run_l2_ticket_market_gap_v2 import add_market_features, market_feature_columns
from run_l2_market_gap_law_arena_v1 import candidate_group_tables, chosen_from_table, metrics
from run_l2_quinella_human_machine_walkforward_v1 import (
    YEARS, TEST_YEARS, PRED_YEARS, EXPECTED_2025_SEQUENCE,
    predict_human_year, select_human_rules, machine_signal
)

def parse_args():
    p=argparse.ArgumentParser()
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def enrich_metrics(g,label,test_year,candidate_id,relation):
    m=metrics(g,label)
    m.update({
        "test_year":test_year,
        "candidate_id":candidate_id,
        "relation":relation,
        "executed_races":int(g["race_id"].nunique()) if len(g) else 0,
        "mean_machine_ratio":float(g["machine_ratio_to_market"].mean()) if len(g) else None,
        "median_machine_ratio":float(g["machine_ratio_to_market"].median()) if len(g) else None,
    })
    return m

def main():
    a=parse_args()
    paths={y:getattr(a,f"l17_{y}") for y in YEARS}
    l17={y:load_l17(paths[y],y) for y in YEARS}
    frames={y:add_market_features(build_year_frame(y,l17[y],a.backfill_root)) for y in YEARS}
    cols=market_feature_columns(frames[2022])

    human_predictions={}
    for y in PRED_YEARS:
        pred,_=predict_human_year(y,frames,cols)
        human_predictions[y]=pred

    by_year_rows=[]
    selection_rows=[]
    pooled_frames=defaultdict(lambda: defaultdict(list))
    selected_years=defaultdict(list)

    for target in TEST_YEARS:
        dev_years=[y for y in PRED_YEARS if y<target]
        selected,_=select_human_rules(human_predictions,dev_years)
        ids=[x[0] for x in selected]
        if target==2025 and ids!=EXPECTED_2025_SEQUENCE:
            raise SystemExit(
                "2025 zero-base reproduction drift "
                f"got={ids} expected={EXPECTED_2025_SEQUENCE}"
            )

        machine,_=machine_signal(target,frames,cols)
        machine_cols=[
            "year","race_id","pair_numbers","p_model",
            "machine_ratio_to_market","machine_agree"
        ]
        pred=human_predictions[target].merge(
            machine[machine_cols],
            on=["year","race_id","pair_numbers"],
            how="left",
            validate="one_to_one",
        )
        if pred["machine_agree"].isna().any():
            raise SystemExit(f"machine/human universe merge drift target={target}")
        tables=candidate_group_tables(pred)

        for step,(cid,gap,conds,row) in enumerate(selected,1):
            chosen=chosen_from_table(tables,gap,conds).copy()
            selected_years[cid].append(target)
            selection_rows.append({
                "test_year":target,
                "dev_years":"|".join(map(str,dev_years)),
                "step":step,
                "candidate_id":cid,
                "gap_spec":gap,
                "conditions":";".join(f"{d}={v}" for d,v in conds) if conds else "NONE",
                "test_tickets":len(chosen),
            })
            parts={
                "ALL":chosen,
                "MACHINE_AGREE":chosen[chosen["machine_agree"]].copy(),
                "MACHINE_DISAGREE":chosen[~chosen["machine_agree"]].copy(),
            }
            for rel,g in parts.items():
                r=enrich_metrics(g,f"{cid}_{rel}_{target}",target,cid,rel)
                by_year_rows.append(r)
                pooled_frames[cid][rel].append(g)

    by_year=pd.DataFrame(by_year_rows)
    sel_df=pd.DataFrame(selection_rows)

    pooled_rows=[]
    for cid,years in sorted(selected_years.items()):
        uniq_years=sorted(set(years))
        for rel in ("ALL","MACHINE_AGREE","MACHINE_DISAGREE"):
            parts=pooled_frames[cid][rel]
            g=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
            if g.empty:
                base={
                    "label":f"{cid}_{rel}_POOLED","tickets":0,"hits":0,
                    "hit_rate_pct":0.0,"stake_yen":0.0,"return_yen":0.0,
                    "profit_yen":0.0,"roi_pct":None,
                    "largest_single_return_yen":0.0,
                    "largest_return_share_pct":0.0,
                    "roi_after_remove_largest_win_pct":None,
                    "median_odds":None,
                }
            else:
                base=metrics(g,f"{cid}_{rel}_POOLED")
            base.update({
                "candidate_id":cid,
                "relation":rel,
                "selected_year_count":len(uniq_years),
                "selected_years":"|".join(map(str,uniq_years)),
            })
            pooled_rows.append(base)

    pooled=pd.DataFrame(pooled_rows)

    repeat_rows=[]
    for cid,years in sorted(selected_years.items()):
        uniq_years=sorted(set(years))
        if len(uniq_years)<2:
            continue
        yy=by_year[by_year["candidate_id"]==cid]
        p=pooled[pooled["candidate_id"]==cid]
        pmap={r["relation"]:r for r in p.to_dict(orient="records")}

        agree_beats_all=0
        agree_beats_disagree=0
        agree_profitable_years=0
        all_profitable_years=0
        valid_years=0
        for y in uniq_years:
            q=yy[yy["test_year"]==y]
            qm={r["relation"]:r for r in q.to_dict(orient="records")}
            if "ALL" not in qm or "MACHINE_AGREE" not in qm or "MACHINE_DISAGREE" not in qm:
                continue
            valid_years+=1
            ra=qm["MACHINE_AGREE"]["roi_pct"]
            rall=qm["ALL"]["roi_pct"]
            rd=qm["MACHINE_DISAGREE"]["roi_pct"]
            if ra is not None and rall is not None and ra>rall:
                agree_beats_all+=1
            if ra is not None and rd is not None and ra>rd:
                agree_beats_disagree+=1
            if ra is not None and ra>100:
                agree_profitable_years+=1
            if rall is not None and rall>100:
                all_profitable_years+=1

        repeat_rows.append({
            "candidate_id":cid,
            "selected_year_count":len(uniq_years),
            "selected_years":"|".join(map(str,uniq_years)),
            "valid_comparison_years":valid_years,
            "agree_beats_all_years":agree_beats_all,
            "agree_beats_disagree_years":agree_beats_disagree,
            "agree_profitable_years":agree_profitable_years,
            "all_profitable_years":all_profitable_years,
            "all_tickets":pmap["ALL"]["tickets"],
            "all_roi_pct":pmap["ALL"]["roi_pct"],
            "all_ex_max_roi_pct":pmap["ALL"]["roi_after_remove_largest_win_pct"],
            "agree_tickets":pmap["MACHINE_AGREE"]["tickets"],
            "agree_roi_pct":pmap["MACHINE_AGREE"]["roi_pct"],
            "agree_ex_max_roi_pct":pmap["MACHINE_AGREE"]["roi_after_remove_largest_win_pct"],
            "disagree_tickets":pmap["MACHINE_DISAGREE"]["tickets"],
            "disagree_roi_pct":pmap["MACHINE_DISAGREE"]["roi_pct"],
            "disagree_ex_max_roi_pct":pmap["MACHINE_DISAGREE"]["roi_after_remove_largest_win_pct"],
            "agree_minus_all_roi_points":(
                None if pmap["MACHINE_AGREE"]["roi_pct"] is None or pmap["ALL"]["roi_pct"] is None
                else pmap["MACHINE_AGREE"]["roi_pct"]-pmap["ALL"]["roi_pct"]
            ),
            "agree_minus_disagree_roi_points":(
                None if pmap["MACHINE_AGREE"]["roi_pct"] is None or pmap["MACHINE_DISAGREE"]["roi_pct"] is None
                else pmap["MACHINE_AGREE"]["roi_pct"]-pmap["MACHINE_DISAGREE"]["roi_pct"]
            ),
        })

    repeat=pd.DataFrame(repeat_rows)
    if not repeat.empty:
        repeat=repeat.sort_values(
            [
                "selected_year_count",
                "agree_beats_disagree_years",
                "agree_beats_all_years",
                "agree_ex_max_roi_pct",
                "agree_tickets",
            ],
            ascending=[False,False,False,False,False],
            na_position="last",
        )

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    sel_df.to_csv(out/"selected-rules-by-year.csv",index=False)
    by_year.to_csv(out/"rule-machine-by-year.csv",index=False)
    pooled.to_csv(out/"rule-machine-pooled.csv",index=False)
    repeat.to_csv(out/"repeat-rule-stability.csv",index=False)

    top_repeat=repeat.head(20).to_dict(orient="records") if not repeat.empty else []
    summary={
        "contract":"L2_QUINELLA_RULE_MACHINE_STABILITY_V1",
        "source_variant":"ODDS_COMMA_CORRECTED",
        "test_years":list(TEST_YEARS),
        "selection_contract":"human rule portfolio rediscovered using prior OOF years only",
        "machine_signal":"within-race normalized p_model > normalized market probability",
        "machine_ratio_threshold":1.0,
        "manual_machine_threshold_tuning":False,
        "repeat_definition":"same exact candidate_id selected in at least 2 test-year discovery portfolios",
        "repeat_candidate_count":int(len(repeat)),
        "three_year_repeat_count":int((repeat["selected_year_count"]==3).sum()) if not repeat.empty else 0,
        "top_repeat_candidates":top_repeat,
        "interpretation_guard":"recurrence summary is descriptive across completed walk-forward folds; do not treat cross-fold recurrence selection itself as a new untouched holdout result",
        "2025_zero_base_sequence_reproduced":True,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )
    print("L2_QUINELLA_RULE_MACHINE_STABILITY_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
