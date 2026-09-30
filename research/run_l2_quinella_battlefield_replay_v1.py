#!/usr/bin/env python3
import argparse
import itertools
import json
import math
from pathlib import Path

import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import build_year_frame
from run_l2_ticket_market_gap_v2 import add_market_features, market_feature_columns
from run_l2_market_gap_law_arena_v1 import metrics
from run_l2_quinella_regime_diagnosis_v1 import build_selected_rows, PRIMARY_CANDIDATE
from run_l2_quinella_human_machine_walkforward_v1 import YEARS

SEGMENT_DIMS=(
    "surface","distance_band","race_class_normalized","weight_rule",
    "venue_code","track_condition","month_band","direction",
)
TARGET_YEARS=(2024,2025)
TOP_K=5

def parse_args():
    p=argparse.ArgumentParser()
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def wilson_lower(hits,n,z=1.2815515655446004):
    if n<=0:
        return 0.0
    ph=hits/n
    z2=z*z
    den=1.0+z2/n
    ctr=ph+z2/(2*n)
    rad=z*math.sqrt((ph*(1-ph)/n)+(z2/(4*n*n)))
    return max(0.0,(ctr-rad)/den)

def roi_metrics(g,label):
    if g.empty:
        return {
            "label":label,"tickets":0,"hits":0,"hit_rate_pct":0.0,
            "stake_yen":0.0,"return_yen":0.0,"profit_yen":0.0,
            "roi_pct":None,"largest_single_return_yen":0.0,
            "largest_return_share_pct":0.0,
            "roi_after_remove_largest_win_pct":None,"median_odds":None,
        }
    return metrics(g,label)

def segment_mask(df,definition):
    mask=pd.Series(True,index=df.index)
    for col,val in definition:
        mask &= df[col].fillna("__NA__").astype(str).eq(val)
    return mask

def definition_id(definition):
    return "__".join(f"{c}={v}" for c,v in definition)

def candidate_segments(prior):
    n_all=len(prior)
    min_support=max(8,int(math.ceil(0.10*n_all)))
    rows=[]
    for width in (1,2):
        for dims in itertools.combinations(SEGMENT_DIMS,width):
            vals=prior[list(dims)].fillna("__NA__").astype(str)
            counts=vals.value_counts(dropna=False)
            for key,n in counts.items():
                if n<min_support:
                    continue
                if not isinstance(key, tuple):
                    key=(key,)
                definition=tuple(zip(dims,key))
                g=prior[segment_mask(prior,definition)].copy()
                hits=int(g["hit"].sum())
                if hits<1:
                    continue
                med=float(g["odds"].median())
                low=wilson_lower(hits,len(g))
                m=roi_metrics(g,"PRIOR")
                per_year=[]
                for y in sorted(g["test_year"].unique()):
                    gy=g[g["test_year"]==y]
                    my=roi_metrics(gy,f"PRIOR_{y}")
                    per_year.append((int(y),len(gy),my["roi_pct"]))
                rows.append({
                    "segment_id":definition_id(definition),
                    "segment_width":width,
                    "definition":definition,
                    "prior_tickets":len(g),
                    "prior_hits":hits,
                    "prior_hit_rate_pct":100.0*hits/len(g),
                    "prior_roi_pct":m["roi_pct"],
                    "prior_ex_max_roi_pct":m["roi_after_remove_largest_win_pct"],
                    "prior_median_odds":med,
                    "wilson80_hit_low_pct":100.0*low,
                    "conservative_roi_proxy_pct":100.0*low*med,
                    "prior_year_detail":"|".join(
                        f"{y}:{nn}:{'' if rr is None else round(float(rr),3)}"
                        for y,nn,rr in per_year
                    ),
                })
    rows.sort(
        key=lambda r:(r["conservative_roi_proxy_pct"],r["prior_tickets"],r["prior_roi_pct"] or -1e9),
        reverse=True,
    )
    return rows,min_support

def overlap_aware_top(rows,prior,k=TOP_K):
    chosen=[]
    covered=set()
    for r in rows:
        idx=set(prior.index[segment_mask(prior,r["definition"])])
        if not idx:
            continue
        novelty=len(idx-covered)/len(idx)
        # Avoid five near-duplicates of exactly the same tickets.
        if chosen and novelty<0.25:
            continue
        rr=dict(r)
        rr["prior_novelty_share_pct"]=100.0*novelty
        chosen.append(rr)
        covered |= idx
        if len(chosen)>=k:
            break
    return chosen

def evaluate_definition(prior,target,row,target_year,rank):
    definition=row["definition"]
    pg=prior[segment_mask(prior,definition)].copy()
    tg=target[segment_mask(target,definition)].copy()
    pm=roi_metrics(pg,"PRIOR")
    tm=roi_metrics(tg,"TARGET")
    out={k:v for k,v in row.items() if k!="definition"}
    out.update({
        "target_year":target_year,
        "pretarget_rank":rank,
        "target_tickets":tm["tickets"],
        "target_hits":tm["hits"],
        "target_roi_pct":tm["roi_pct"],
        "target_ex_max_roi_pct":tm["roi_after_remove_largest_win_pct"],
        "target_profit_yen":tm["profit_yen"],
        "target_median_odds":tm["median_odds"],
        "target_retention_pct":100.0*len(tg)/len(target) if len(target) else 0.0,
    })
    return out,tg

def main():
    a=parse_args()
    paths={y:getattr(a,f"l17_{y}") for y in YEARS}
    l17={y:load_l17(paths[y],y) for y in YEARS}
    frames={y:add_market_features(build_year_frame(y,l17[y],a.backfill_root)) for y in YEARS}
    cols=market_feature_columns(frames[2022])

    selected,audit=build_selected_rows(frames,cols)
    z=selected[
        (selected["candidate_id"]==PRIMARY_CANDIDATE) &
        (selected["machine_agree"])
    ].copy()
    if sorted(z["test_year"].unique().tolist())!=[2023,2024,2025]:
        raise SystemExit("primary candidate walk-forward coverage drift")

    all_rows=[]
    union_rows=[]
    dim_rows=[]
    selections={}

    for target_year in TARGET_YEARS:
        prior=z[z["test_year"]<target_year].copy()
        target=z[z["test_year"]==target_year].copy()
        ranked,min_support=candidate_segments(prior)
        if not ranked:
            raise SystemExit(f"no pretarget segments target={target_year}")

        chosen=overlap_aware_top(ranked,prior,TOP_K)
        selections[target_year]=[r["segment_id"] for r in chosen]

        for rank,row in enumerate(chosen,1):
            ev,_=evaluate_definition(prior,target,row,target_year,rank)
            ev["selection_scope"]="OVERLAP_AWARE_TOP"
            ev["pretarget_min_support"]=min_support
            all_rows.append(ev)

        # Also preserve the best single-dimensional segment in each fixed dimension.
        for dim in SEGMENT_DIMS:
            candidates=[r for r in ranked if r["segment_width"]==1 and r["definition"][0][0]==dim]
            if not candidates:
                continue
            row=candidates[0]
            ev,_=evaluate_definition(prior,target,row,target_year,1)
            ev["selection_scope"]="BEST_WITHIN_DIMENSION"
            ev["dimension"]=dim
            ev["pretarget_min_support"]=min_support
            dim_rows.append(ev)

        target_union=pd.Series(False,index=target.index)
        prior_union=pd.Series(False,index=prior.index)
        for row in chosen:
            target_union |= segment_mask(target,row["definition"])
            prior_union |= segment_mask(prior,row["definition"])
        pg=prior[prior_union].copy()
        tg=target[target_union].copy()
        pm=roi_metrics(pg,f"UNION_PRIOR_{target_year}")
        tm=roi_metrics(tg,f"UNION_TARGET_{target_year}")
        bm=roi_metrics(target,f"ALL_TARGET_{target_year}")
        union_rows.append({
            "target_year":target_year,
            "prior_years":"|".join(map(str,sorted(prior["test_year"].unique()))),
            "selected_segments":"|".join(r["segment_id"] for r in chosen),
            "prior_tickets":pm["tickets"],
            "prior_hits":pm["hits"],
            "prior_roi_pct":pm["roi_pct"],
            "prior_ex_max_roi_pct":pm["roi_after_remove_largest_win_pct"],
            "baseline_target_tickets":bm["tickets"],
            "baseline_target_hits":bm["hits"],
            "baseline_target_roi_pct":bm["roi_pct"],
            "target_tickets":tm["tickets"],
            "target_hits":tm["hits"],
            "target_roi_pct":tm["roi_pct"],
            "target_ex_max_roi_pct":tm["roi_after_remove_largest_win_pct"],
            "target_profit_yen":tm["profit_yen"],
            "target_retention_pct":100.0*len(tg)/len(target) if len(target) else 0.0,
        })

    recurrence=sorted(set(selections[2024]) & set(selections[2025]))
    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out/"pretarget-selected-segments.csv",index=False)
    pd.DataFrame(dim_rows).to_csv(out/"best-segment-by-dimension.csv",index=False)
    pd.DataFrame(union_rows).to_csv(out/"pretarget-union-replay.csv",index=False)
    audit.to_csv(out/"selection-audit.csv",index=False)

    summary={
        "contract":"L2_QUINELLA_BATTLEFIELD_REPLAY_V1",
        "primary_candidate":PRIMARY_CANDIDATE,
        "machine_filter":"machine_agree only",
        "selection_protocol":"for each target year, rank fixed single/pair categorical segments using prior years only; Wilson-80 lower hit bound x prior median odds; then choose up to 5 with >=25% novelty",
        "segment_dimensions":list(SEGMENT_DIMS),
        "target_years":list(TARGET_YEARS),
        "top_k":TOP_K,
        "selected_2024":selections[2024],
        "selected_2025":selections[2025],
        "reselected_in_both_targets":recurrence,
        "union_replay":union_rows,
        "diagnostic_exploratory":True,
        "target_year_never_used_for_segment_selection":True,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_QUINELLA_BATTLEFIELD_REPLAY_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
