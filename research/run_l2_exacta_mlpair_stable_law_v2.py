#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_exacta_decomposed_v1 import YEARS, build_pair_year_frame, pair_feature_columns, train_pair_ranker
from run_l2_exacta_stable_law_v1 import (
    STAKE, K_VALUES, MIN_TICKETS_PER_PERIOD, MIN_RACES_PER_PERIOD,
    SHORTLIST_PER_FOLD, atom_templates, atom_mask, stats, expand_oriented,
)

SCORED_YEARS=(2023,2024,2025)
TEST_YEARS=(2024,2025)
BEAM_SINGLE=40


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1); out[int(y)]=p
    if set(out)!=set(YEARS) or 2026 in out:
        raise SystemExit("L1.7 years must be exactly 2022-2025; 2026 sealed")
    return out


def split_halves(df,label):
    dates=sorted(df["race_date"].astype(str).unique())
    cut=len(dates)//2
    a=set(dates[:cut]); b=set(dates[cut:])
    return [
        (f"{label}_H1",df[df["race_date"].astype(str).isin(a)].copy()),
        (f"{label}_H2",df[df["race_date"].astype(str).isin(b)].copy()),
    ]


def prior_periods(scored,test_year):
    if test_year==2024:
        return split_halves(scored[2023],"2023")
    if test_year==2025:
        return [("2023",scored[2023].copy()),("2024",scored[2024].copy())]
    raise ValueError(test_year)


def eval_rule(periods,k,atoms):
    per=[]
    sig=" & ".join([f"pair_model_rank<={k}"]+[a["signature"] for a in atoms])
    for name,df in periods:
        mask=df["pair_model_rank"].to_numpy()<=k
        for a in atoms:
            mask &= atom_mask(df,a)
        s=stats(df,mask); s["period"]=name; per.append(s)
    valid=all(x["tickets"]>=MIN_TICKETS_PER_PERIOD and x["races"]>=MIN_RACES_PER_PERIOD for x in per)
    stable=valid and all(x["roi_pct"]>100.0 for x in per)
    robust=stable and all(x["roi_minus_top1_pct"]>100.0 for x in per)
    return {
        "signature":sig,"k":k,"atoms":atoms,"period_stats":per,
        "valid_support":valid,"stable_prior":stable,"robust_prior":robust,
        "worst_prior_roi":min((x["roi_pct"] for x in per),default=0.0),
        "median_prior_roi":float(np.median([x["roi_pct"] for x in per])) if per else 0.0,
        "worst_prior_roi_minus_top1":min((x["roi_minus_top1_pct"] for x in per),default=0.0),
        "min_prior_tickets":min((x["tickets"] for x in per),default=0),
        "min_prior_races":min((x["races"] for x in per),default=0),
    }


def apply_rule(df,rule):
    mask=df["pair_model_rank"].to_numpy()<=rule["k"]
    for a in rule["atoms"]:
        mask &= atom_mask(df,a)
    return stats(df,mask)


def main():
    a=parse_args(); started=time.perf_counter()
    lp=parse_paths(a.l17_year)

    raw={}
    for y in YEARS:
        raw[y]=build_pair_year_frame(y,load_l17(lp[y],y),a.backfill_root)

    pair_cols=pair_feature_columns(raw[2022],market=True)
    if any(c.startswith("dir_") for c in pair_cols):
        raise SystemExit("directional leakage into unordered pair ranker")

    scored_pair={}
    foundation=[]
    folds=[]
    for y in SCORED_YEARS:
        train=pd.concat([raw[t] for t in YEARS if t<y],ignore_index=True)
        test=raw[y].copy().reset_index(drop=True)
        pred,_,secs=train_pair_ranker(train,test,pair_cols,151000+y)
        test["pair_model_score"]=pred
        test["pair_model_rank"]=test.groupby("race_id")["pair_model_score"].rank(
            method="min",ascending=False
        ).astype(int)

        source=int(test["race_id"].nunique())
        for k in (1,3,5,10,15):
            captured=int(test.loc[
                (test["pair_hit"]==1)&(test["pair_model_rank"]<=k),"race_id"
            ].nunique())
            foundation.append({
                "year":y,"top_k":k,"source_races":source,
                "captured_races":captured,
                "pair_capture_pct":100.0*captured/source if source else 0.0,
            })

        # Reuse the proven V1 orientation builder, but feed the OOT ML pair rank
        # into its candidate-rank slot; rename immediately afterward.
        test["market_pair_rank_original"]=test["pair_market_rank"]
        test["pair_market_rank"]=test["pair_model_rank"]
        oriented=expand_oriented(test)
        oriented["pair_model_rank"]=oriented["pair_market_rank"]
        scored_pair[y]=oriented

        folds.append({
            "test_year":y,
            "train_years":"|".join(str(t) for t in YEARS if t<y),
            "train_pair_rows":len(train),"test_pair_rows":len(test),
            "feature_count":len(pair_cols),"train_seconds":secs,
        })
        print("MLPAIR_OOT_READY "+json.dumps({
            "test_year":y,
            "train_years":[t for t in YEARS if t<y],
            "top10_pair_capture_pct":[x for x in foundation if x["year"]==y and x["top_k"]==10][0]["pair_capture_pct"],
            "seconds":round(secs,3),
        },separators=(",",":")),flush=True)

    threshold_rows=[]; prior_rows=[]; unseen_rows=[]

    for test_year in TEST_YEARS:
        periods=prior_periods(scored_pair,test_year)
        train_all=pd.concat([x[1] for x in periods],ignore_index=True)
        atoms=atom_templates(train_all)
        for at in atoms:
            threshold_rows.append({
                "test_year":test_year,"feature":at["feature"],"op":at["op"],
                "quantile":at["q"],"threshold":at["threshold"],"signature":at["signature"],
            })

        singles=[]
        for k in K_VALUES:
            singles.append(eval_rule(periods,k,[]))
            for at in atoms:
                singles.append(eval_rule(periods,k,[at]))

        valid=[x for x in singles if x["valid_support"]]
        valid.sort(key=lambda x:(
            x["worst_prior_roi"],x["worst_prior_roi_minus_top1"],
            x["min_prior_races"],x["min_prior_tickets"]
        ),reverse=True)
        seeds=valid[:BEAM_SINGLE]

        combos=[]; seen_combo=set()
        for i in range(len(seeds)):
            for j in range(i+1,len(seeds)):
                merged=[]; seen=set()
                for at in seeds[i]["atoms"]+seeds[j]["atoms"]:
                    if at["signature"] not in seen:
                        merged.append(at); seen.add(at["signature"])
                if len(merged)!=2:
                    continue
                k=min(seeds[i]["k"],seeds[j]["k"])
                key=(k,tuple(sorted(x["signature"] for x in merged)))
                if key in seen_combo:
                    continue
                seen_combo.add(key)
                combos.append(eval_rule(periods,k,merged))

        passing=[x for x in singles+combos if x["stable_prior"]]
        dedup={}
        for x in passing:
            old=dedup.get(x["signature"])
            score=(x["robust_prior"],x["worst_prior_roi_minus_top1"],x["worst_prior_roi"],x["min_prior_races"])
            if old is None:
                dedup[x["signature"]]=x
            else:
                oldscore=(old["robust_prior"],old["worst_prior_roi_minus_top1"],old["worst_prior_roi"],old["min_prior_races"])
                if score>oldscore: dedup[x["signature"]]=x
        selected=list(dedup.values())
        selected.sort(key=lambda x:(
            x["robust_prior"],x["worst_prior_roi_minus_top1"],
            x["worst_prior_roi"],x["min_prior_races"],x["min_prior_tickets"]
        ),reverse=True)
        selected=selected[:SHORTLIST_PER_FOLD]

        print("MLPAIR_STABLE_LAW_FOLD "+json.dumps({
            "test_year":test_year,
            "prior_periods":[x[0] for x in periods],
            "atoms":len(atoms),"single_rules":len(singles),
            "valid_single_rules":len(valid),"combo_rules":len(combos),
            "stable_candidates":len(dedup),"shortlisted":len(selected),
            "robust_shortlisted":sum(1 for x in selected if x["robust_prior"]),
        },separators=(",",":")),flush=True)

        for rank,rule in enumerate(selected,1):
            ts=apply_rule(scored_pair[test_year],rule)
            unseen_rows.append({
                "test_year":test_year,"prior_rank":rank,"signature":rule["signature"],
                "robust_prior":rule["robust_prior"],
                "worst_prior_roi":rule["worst_prior_roi"],
                "worst_prior_roi_minus_top1":rule["worst_prior_roi_minus_top1"],
                "median_prior_roi":rule["median_prior_roi"],
                "min_prior_tickets":rule["min_prior_tickets"],
                "min_prior_races":rule["min_prior_races"],
                **{f"test_{k}":v for k,v in ts.items()},
            })
            for ps in rule["period_stats"]:
                prior_rows.append({
                    "test_year":test_year,"prior_rank":rank,
                    "signature":rule["signature"],"robust_prior":rule["robust_prior"],**ps
                })

    unseen=pd.DataFrame(unseen_rows)
    recurrence=[]
    if not unseen.empty:
        for sig,g in unseen.groupby("signature",sort=False):
            rois=g["test_roi_pct"].astype(float).tolist()
            recurrence.append({
                "signature":sig,
                "selected_folds":len(g),
                "selected_test_years":"|".join(str(x) for x in sorted(g["test_year"].astype(int))),
                "profitable_unseen_years":int(sum(x>100.0 for x in rois)),
                "worst_unseen_roi":float(min(rois)),
                "median_unseen_roi":float(np.median(rois)),
                "mean_unseen_roi":float(np.mean(rois)),
                "total_test_tickets":int(g["test_tickets"].sum()),
                "total_test_races":int(g["test_races"].sum()),
                "total_test_profit_yen":float(g["test_profit_yen"].sum()),
                "all_selected_unseen_profitable":bool(all(x>100.0 for x in rois)),
                "all_prior_robust_when_selected":bool(g["robust_prior"].all()),
            })
    rec=pd.DataFrame(recurrence)
    if not rec.empty:
        rec=rec.sort_values(
            ["selected_folds","profitable_unseen_years","worst_unseen_roi","total_test_races"],
            ascending=[False,False,False,False]
        )
        repeated=rec[(rec["selected_folds"]==2)&(rec["profitable_unseen_years"]==2)]
    else:
        repeated=rec

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(foundation).to_csv(out/"mlpair-foundation.csv",index=False)
    pd.DataFrame(folds).to_csv(out/"mlpair-folds.csv",index=False)
    unseen.to_csv(out/"unseen-year-results.csv",index=False)
    pd.DataFrame(prior_rows).to_csv(out/"prior-period-audit.csv",index=False)
    pd.DataFrame(threshold_rows).to_csv(out/"learned-thresholds.csv",index=False)
    rec.to_csv(out/"cross-year-recurrence.csv",index=False)

    summary={
        "contract":"L2_EXACTA_MLPAIR_STABLE_LAW_V2",
        "foundation":"decomposed market-aware ML unordered pair rank",
        "scored_years":{
            "2023":"trained on 2022 only",
            "2024":"trained on 2022-2023",
            "2025":"trained on 2022-2024"
        },
        "law_test_years":[2024,2025],
        "why_2023_not_law_test":"no leakage-safe pre-2022 ML-pair population exists in the four-year dataset",
        "2024_prior_periods":["2023_H1","2023_H2"],
        "2025_prior_periods":["2023","2024"],
        "search":"ML pair rank 1..15 plus prior-derived simple conditions and beam two-condition intersections",
        "prior_gate":f"ROI>100 in every independent prior period with >= {MIN_TICKETS_PER_PERIOD} tickets and >= {MIN_RACES_PER_PERIOD} races",
        "robust_flag":"ROI>100 even after removing the largest winning payout in every prior period",
        "same_signature_selected_2024_and_2025_and_profitable_both":int(len(repeated)),
        "test_year_never_used_for_selection":True,
        "2026_locked":True,
        "script_total_seconds":time.perf_counter()-started,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 EXACTA ML-Pair Stable LAW V2\n\n"
        "The candidate universe is the true decomposed market-aware ML unordered-pair rank generated walk-forward. "
        "2023 is scored by 2022 training, 2024 by 2022-2023, and 2025 by 2022-2024. "
        "LAW discovery uses only already-out-of-time scored prior data. "
        "2024 laws must reproduce in both halves of 2023; 2025 laws must reproduce in 2023 and 2024. "
        "The unseen year never selects the law. 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("L2_EXACTA_MLPAIR_STABLE_LAW_V2_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
    print("===== MLPAIR FOUNDATION =====")
    print(pd.DataFrame(foundation).to_csv(index=False))
    print("===== RECURRENCE TOP =====")
    print(rec.head(50).to_csv(index=False) if not rec.empty else "NONE")
    print("===== UNSEEN TOP =====")
    print(unseen.head(80).to_csv(index=False) if not unseen.empty else "NONE")


if __name__=="__main__":
    main()
