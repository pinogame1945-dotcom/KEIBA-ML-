#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from build_l2_bet_kings_dataset_v1 import (
    canonical_numbers,
    decode_odds,
    finite,
    horse_number_map,
    load_day,
    load_fixed_ledgers,
    load_odds_day,
    load_router,
    payout_map,
)

YEARS=(2023,2024,2025)
LOCKED_YEARS=(2026,)
EXPECTED_OLD={
    2023:{"instances":694,"roi":153.64553314121036},
    2024:{"instances":737,"roi":186.58073270013568},
    2025:{"instances":743,"roi":160.3028263795424},
}

NUMERIC_FEATURES=[
    "gate_score","candidate_pool_size","seven_union_count","novel_pool_count",
    "field_size","distance_m","cw_top1_max_vote_share","cw_top3_jaccard",
    "cw_top6_jaccard","cw_rank_diff_mean","cw_rank_std_mean",
    "cw_prob_std_mean","cw_prob_std_max","novel_position",
    "k2_horse_number","k2_number_frac",
]
CATEGORICAL_FEATURES=[
    "venue_code","surface","race_class","discipline","direction",
    "selected_outsider_1","selected_outsider_2","selected_outsider_pair",
]

def parse_args():
    p=argparse.ArgumentParser(description="Audit the hidden market-completeness prefilter behind old K2 A12_THIRD results.")
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1)
        out[int(y)]=p
    if set(out)!=set(YEARS):
        raise SystemExit(f"router years mismatch {sorted(out)} expected={list(YEARS)}")
    return out

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def mean(vals):
    vals=[float(x) for x in vals if x is not None and math.isfinite(float(x))]
    return sum(vals)/len(vals) if vals else None

def stdev(vals):
    vals=[float(x) for x in vals if x is not None and math.isfinite(float(x))]
    if len(vals)<2: return 0.0
    m=sum(vals)/len(vals)
    return math.sqrt(sum((x-m)**2 for x in vals)/(len(vals)-1))

def smd(a,b):
    a=[float(x) for x in a if x is not None and math.isfinite(float(x))]
    b=[float(x) for x in b if x is not None and math.isfinite(float(x))]
    if not a or not b: return None
    ma,mb=mean(a),mean(b); sa,sb=stdev(a),stdev(b)
    pooled=math.sqrt((sa*sa+sb*sb)/2.0)
    return (ma-mb)/pooled if pooled>0 else 0.0

def summarize_partition(rows,year,label):
    part=[r for r in rows if r["year"]==year and r["old_eligible"]==label]
    stake=200.0*len(part)
    ret=sum(r["a12_third_return_yen"] for r in part)
    return {
        "year":year,
        "old_eligible":int(label),
        "instances":len(part),
        "races":len({r["race_id"] for r in part}),
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "hit_instances":sum(r["a12_third_return_yen"]>0 for r in part),
        "hit_rate_pct":100*sum(r["a12_third_return_yen"]>0 for r in part)/len(part) if part else 0.0,
    }

def summarize_reason(rows,year,reason):
    part=[r for r in rows if r["year"]==year and reason in r["missing_reasons"].split("|")]
    stake=200.0*len(part); ret=sum(r["a12_third_return_yen"] for r in part)
    return {
        "year":year,"missing_reason":reason,"instances":len(part),
        "races":len({r["race_id"] for r in part}),
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "hit_instances":sum(r["a12_third_return_yen"]>0 for r in part),
    }

def structural_diff(rows):
    out=[]
    for year in list(YEARS)+["ALL"]:
        subset=rows if year=="ALL" else [r for r in rows if r["year"]==year]
        keep=[r for r in subset if r["old_eligible"]]
        drop=[r for r in subset if not r["old_eligible"]]
        for feat in NUMERIC_FEATURES:
            av=[r.get(feat) for r in keep]; bv=[r.get(feat) for r in drop]
            out.append({
                "year":year,"feature":feat,
                "kept_mean":mean(av),"dropped_mean":mean(bv),
                "standardized_mean_diff_kept_minus_dropped":smd(av,bv),
                "kept_n":sum(x is not None for x in av),
                "dropped_n":sum(x is not None for x in bv),
            })
    out.sort(key=lambda r:(str(r["year"]),-(abs(r["standardized_mean_diff_kept_minus_dropped"] or 0.0)),r["feature"]))
    return out

def categorical_diff(rows):
    out=[]
    for feat in CATEGORICAL_FEATURES:
        vals=sorted({str(r.get(feat) or "__MISSING__") for r in rows})
        for val in vals:
            keep=[r for r in rows if r["old_eligible"]]
            drop=[r for r in rows if not r["old_eligible"]]
            kr=sum(str(r.get(feat) or "__MISSING__")==val for r in keep)/len(keep) if keep else 0.0
            dr=sum(str(r.get(feat) or "__MISSING__")==val for r in drop)/len(drop) if drop else 0.0
            out.append({
                "feature":feat,"value":val,
                "kept_rate_pct":100*kr,"dropped_rate_pct":100*dr,
                "rate_diff_pp":100*(kr-dr),
                "kept_count":sum(str(r.get(feat) or "__MISSING__")==val for r in keep),
                "dropped_count":sum(str(r.get(feat) or "__MISSING__")==val for r in drop),
            })
    out.sort(key=lambda r:-abs(r["rate_diff_pp"]))
    return out

def structural_predictability(rows):
    df=pd.DataFrame(rows)
    out=[]
    for test_year in (2024,2025):
        train=df[df["year"]<test_year].copy()
        test=df[df["year"]==test_year].copy()
        ytr=train["old_eligible"].astype(int)
        yte=test["old_eligible"].astype(int)
        num=[c for c in NUMERIC_FEATURES if c in df.columns]
        cat=[c for c in CATEGORICAL_FEATURES if c in df.columns]
        prep=ColumnTransformer([
            ("num",Pipeline([("scale",StandardScaler())]),num),
            ("cat",OneHotEncoder(handle_unknown="ignore"),cat),
        ])
        model=Pipeline([
            ("prep",prep),
            ("clf",LogisticRegression(max_iter=1000,C=0.5,class_weight="balanced")),
        ])
        model.fit(train[num+cat],ytr)
        p=model.predict_proba(test[num+cat])[:,1]
        out.append({
            "test_year":test_year,
            "train_years":"|".join(map(str,sorted(train["year"].unique()))),
            "train_instances":len(train),"test_instances":len(test),
            "positive_rate_test_pct":100*float(yte.mean()),
            "roc_auc":float(roc_auc_score(yte,p)),
            "pr_auc":float(average_precision_score(yte,p)),
        })
    return out

def main():
    a=parse_args(); paths=parse_paths(a.router_year)
    if 2026 not in LOCKED_YEARS:
        raise SystemExit("2026 lock missing")
    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={y:load_router(paths[y],y) for y in YEARS}

    date_pairs=defaultdict(list)
    for y in YEARS:
        missing=set(fixed[y])-set(routers[y])
        if missing: raise SystemExit(f"fixed alerts missing router y={y} sample={sorted(missing)[:5]}")
        for rid in fixed[y]:
            d=str(routers[y][rid].get("race_date") or "")[:10]
            date_pairs[d].append((y,rid))

    root=Path(a.backfill_root)
    rows=[]
    race_class=defaultdict(dict)
    processed=0
    for di,date in enumerate(sorted(date_pairs),1):
        pairs=date_pairs[date]; wanted={rid for _,rid in pairs}
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        odds=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for year,rid in sorted(pairs):
            fr=fixed[year][rid]; rr=routers[year][rid]
            pack=day.get(rid); oddsrec=odds.get(rid)
            if pack is None or oddsrec is None:
                raise SystemExit(f"missing daily/odds row race={rid}")
            horse_no=horse_number_map(pack)
            anchors=[str(x) for x in fr.get("seven_anchor_horse_ids") or []]
            novel=[str(x) for x in fr.get("novel_horse_ids") or []]
            candidates=[str(x) for x in fr.get("candidate_horse_ids") or []]
            if len(anchors)!=2: raise SystemExit(f"anchor cardinality race={rid}")
            if not novel:
                race_class[year][rid]={"category":"NO_K2","eligible_instances":0,"total_instances":0}
                processed+=1; continue
            missing=set(anchors+novel+candidates)-set(horse_no)
            if missing: raise SystemExit(f"horse number missing race={rid} sample={sorted(missing)[:5]}")
            a1,a2=(horse_no[anchors[0]],horse_no[anchors[1]])
            odds_map=decode_odds(oddsrec)
            payouts,present=payout_map(pack)
            race_meta=rr.get("race") or {}
            consensus=rr.get("consensus") or {}
            outs=list(fr.get("selected_outsiders") or [])
            eligible_count=0
            for pos,hid in enumerate(novel,1):
                h=horse_no[hid]
                win_present=("WIN",(h,)) in odds_map
                exacta_count=sum(
                    ("EXACTA",(h,horse_no[x])) in odds_map
                    for x in candidates if x!=hid
                )
                tri1=(a1,a2,h); tri2=(a2,a1,h)
                tri1_present=("TRIFECTA",tri1) in odds_map
                tri2_present=("TRIFECTA",tri2) in odds_map
                reasons=[]
                if "EXACTA" not in present: reasons.append("PAYOUT_EXACTA_MISSING")
                if "TRIFECTA" not in present: reasons.append("PAYOUT_TRIFECTA_MISSING")
                if not win_present: reasons.append("WIN_ODDS_MISSING")
                if exacta_count<=0: reasons.append("EXACTA_FIRST_ODDS_MISSING")
                if not tri1_present or not tri2_present: reasons.append("A12_THIRD_ODDS_INCOMPLETE")
                old_eligible=not reasons
                eligible_count+=int(old_eligible)
                ret=float(payouts.get(("TRIFECTA",tri1),0.0))+float(payouts.get(("TRIFECTA",tri2),0.0))
                fs=finite(race_meta.get("field_size")) or 0.0
                row={
                    "year":year,"race_id":rid,"race_date":date,
                    "old_eligible":bool(old_eligible),
                    "missing_reasons":"|".join(reasons) if reasons else "__NONE__",
                    "a12_third_return_yen":ret,
                    "a12_third_hit":int(ret>0),
                    "gate_score":finite(fr.get("gate_score")) or 0.0,
                    "candidate_pool_size":len(candidates),
                    "seven_union_count":len(fr.get("seven_union_horse_ids") or []),
                    "novel_pool_count":len(novel),
                    "novel_position":pos,
                    "k2_horse_number":h,
                    "k2_number_frac":h/fs if fs else 0.0,
                    "venue_code":race_meta.get("venue_code"),
                    "surface":race_meta.get("surface"),
                    "race_class":race_meta.get("race_class"),
                    "discipline":race_meta.get("discipline"),
                    "direction":race_meta.get("direction"),
                    "distance_m":finite(race_meta.get("distance_m")) or 0.0,
                    "field_size":fs,
                    "cw_top1_max_vote_share":finite(consensus.get("top1_max_vote_share")) or 0.0,
                    "cw_top3_jaccard":finite(consensus.get("top3_pairwise_jaccard_mean")) or 0.0,
                    "cw_top6_jaccard":finite(consensus.get("top6_pairwise_jaccard_mean")) or 0.0,
                    "cw_rank_diff_mean":finite(consensus.get("pairwise_rank_abs_diff_mean")) or 0.0,
                    "cw_rank_std_mean":finite(consensus.get("horse_rank_std_mean")) or 0.0,
                    "cw_prob_std_mean":finite(consensus.get("horse_probability_std_mean")) or 0.0,
                    "cw_prob_std_max":finite(consensus.get("horse_probability_std_max")) or 0.0,
                    "selected_outsider_1":outs[0] if len(outs)>0 else "__NONE__",
                    "selected_outsider_2":outs[1] if len(outs)>1 else "__NONE__",
                    "selected_outsider_pair":"|".join(outs) if outs else "__NONE__",
                    "win_odds_present":int(win_present),
                    "exacta_first_priced_count":exacta_count,
                    "tri_a1a2k_priced":int(tri1_present),
                    "tri_a2a1k_priced":int(tri2_present),
                }
                rows.append(row)
            category="ALL_OLD_ELIGIBLE" if eligible_count==len(novel) else ("NONE_OLD_ELIGIBLE" if eligible_count==0 else "PARTIAL_OLD_ELIGIBLE")
            race_class[year][rid]={"category":category,"eligible_instances":eligible_count,"total_instances":len(novel)}
            processed+=1
        if di%25==0:
            print(f"OLD_PREFILTER_AUDIT_PROGRESS dates={di}/{len(date_pairs)} races={processed} instances={len(rows)}",flush=True)

    # Exact reconstruction guard for the old misleading "100%" subset.
    partition_rows=[]
    for y in YEARS:
        kept=summarize_partition(rows,y,True)
        dropped=summarize_partition(rows,y,False)
        partition_rows.extend([kept,dropped])
        exp=EXPECTED_OLD[y]
        if kept["instances"]!=exp["instances"]:
            raise SystemExit(f"old eligible instance reconstruction mismatch y={y} got={kept['instances']} expected={exp['instances']}")
        if abs(kept["roi_pct"]-exp["roi"])>1e-9:
            raise SystemExit(f"old ROI reconstruction mismatch y={y} got={kept['roi_pct']} expected={exp['roi']}")

    reasons=sorted({z for r in rows for z in r["missing_reasons"].split("|") if z!="__NONE__"})
    reason_rows=[summarize_reason(rows,y,reason) for y in YEARS for reason in reasons]

    race_rows=[]
    for y in YEARS:
        for category in ("ALL_OLD_ELIGIBLE","PARTIAL_OLD_ELIGIBLE","NONE_OLD_ELIGIBLE","NO_K2"):
            rids=[rid for rid,v in race_class[y].items() if v["category"]==category]
            inst=[r for r in rows if r["year"]==y and r["race_id"] in set(rids)]
            stake=200.0*len(inst); ret=sum(r["a12_third_return_yen"] for r in inst)
            race_rows.append({
                "year":y,"race_category":category,"races":len(rids),"instances":len(inst),
                "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
                "roi_pct":100*ret/stake if stake else None,
                "hit_instances":sum(r["a12_third_return_yen"]>0 for r in inst),
            })

    numdiff=structural_diff(rows)
    catdiff=categorical_diff(rows)
    predict=structural_predictability(rows)

    kept_all=[r for r in rows if r["old_eligible"]]
    drop_all=[r for r in rows if not r["old_eligible"]]
    stake_all=200.0*len(rows); ret_all=sum(r["a12_third_return_yen"] for r in rows)
    top_numeric=sorted(
        [r for r in numdiff if r["year"]=="ALL"],
        key=lambda r:-abs(r["standardized_mean_diff_kept_minus_dropped"] or 0.0)
    )[:8]
    top_cat=catdiff[:12]
    summary={
        "contract":"L2_DANGER_OLD_PREFILTER_AUDIT_V1",
        "analysis_years":list(YEARS),
        "locked_years":list(LOCKED_YEARS),
        "old_prefilter_definition":{
            "unit":"K2 novel instance",
            "requires_win_odds":True,
            "requires_any_exacta_k2_first_price":True,
            "requires_both_a1_a2_k2_and_a2_a1_k2_trifecta_prices":True,
            "requires_exacta_and_trifecta_payout_types":True,
        },
        "interpretation_guard":"This reconstructs and diagnoses the old market-completeness prefilter. Missingness itself is NOT promoted as a betting rule.",
        "old_reconstruction":{
            str(y):next(r for r in partition_rows if r["year"]==y and r["old_eligible"]==1)
            for y in YEARS
        },
        "excluded_partition":{
            str(y):next(r for r in partition_rows if r["year"]==y and r["old_eligible"]==0)
            for y in YEARS
        },
        "full_population":{
            "instances":len(rows),"stake_yen":stake_all,"return_yen":ret_all,
            "profit_yen":ret_all-stake_all,"roi_pct":100*ret_all/stake_all if stake_all else None,
        },
        "old_kept_share_pct":100*len(kept_all)/len(rows),
        "old_dropped_share_pct":100*len(drop_all)/len(rows),
        "structural_predictability":predict,
        "largest_numeric_structural_differences":top_numeric,
        "largest_categorical_structural_differences":top_cat,
        "next_decision":"If structural-only predictability is weak, treat the old ROI lift as market-data-selection bias. If it is materially predictive and stable, investigate those pre-race structural variables directly in a fresh walk-forward gate without using odds-missingness as a feature.",
        "production_promotion":False,
    }

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"instance-partition.csv",partition_rows)
    write_csv(out/"missingness-reasons.csv",reason_rows)
    write_csv(out/"race-partition.csv",race_rows)
    write_csv(out/"numeric-structural-diff.csv",numdiff)
    write_csv(out/"categorical-structural-diff.csv",catdiff)
    write_csv(out/"structural-predictability.csv",predict)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Danger Old Prefilter Audit V1\n\n"
        "This audit exactly reconstructs the hidden K2-instance market-completeness filter behind the old A12_THIRD result. "
        "It then measures the payout-only ROI of kept versus excluded K2 instances, decomposes exclusion reasons, "
        "compares pre-race race/consensus structure, and tests whether the old keep/drop label is predictable from structural features alone. "
        "The market-missingness label is diagnostic only and is not a production betting feature. 2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_DANGER_OLD_PREFILTER_AUDIT_V1_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
