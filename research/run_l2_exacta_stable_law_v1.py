#!/usr/bin/env python3
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd

from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_exacta_decomposed_v1 import YEARS, TEST_YEARS, build_pair_year_frame

STAKE=100.0
K_VALUES=tuple(range(1,16))
QUANTILES=(0.20,0.40,0.60,0.80)
MIN_TICKETS_PER_PERIOD=50
MIN_RACES_PER_PERIOD=30
SHORTLIST_PER_FOLD=100

NUMERIC_FEATURES=(
    "market_direction_share",
    "market_direction_strength",
    "odds",
    "ordered_market_rank",
    "pair_consensus_rank_sum",
    "pair_consensus_rank_abs_gap",
    "pair_top1_votes_sum",
    "pair_top3_support_sum",
    "pair_top6_support_sum",
    "pair_mean_probability_sum",
    "pair_expert_direction_agreement",
    "first_consensus_advantage",
    "first_mean_rank_advantage",
    "first_top1_vote_advantage",
    "first_top3_support_advantage",
    "first_top6_support_advantage",
    "first_mean_probability_advantage",
    "expert_vote_for_first",
    "field_size",
    "distance_m",
)
BOOLEAN_FEATURES=(
    "ticket_supported_by_market",
    "ticket_supported_by_l17",
    "market_l17_direction_agree",
    "both_support_ticket",
)


def parse_args():
    p=argparse.ArgumentParser(description="EXACTA stable-law discovery over decomposed market-pair candidates.")
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def parse_paths(items):
    out={}
    for x in items:
        y,p=x.split(":",1); out[int(y)]=p
    return out


def expand_oriented(pair):
    safe_pair=[
        c for c in (
            "pair_consensus_rank_sum","pair_consensus_rank_abs_gap",
            "pair_top1_votes_sum","pair_top3_support_sum","pair_top6_support_sum",
            "pair_mean_probability_sum","pair_expert_direction_agreement",
            "field_size","distance_m","surface","discipline","venue_code","race_class_normalized",
            "dir_consensus_rank_diff","dir_mean_rank_diff","dir_top1_votes_diff",
            "dir_top3_support_diff","dir_top6_support_diff","dir_mean_probability_diff",
            "dir_expert_vote_margin",
        ) if c in pair.columns
    ]
    keep=list(dict.fromkeys([
        "year","race_id","race_date","a_horse_number","b_horse_number",
        "hit_a_to_b","hit_b_to_a","return_a_to_b","return_b_to_a",
        "odds_a_to_b","odds_b_to_a","market_q_a_to_b","market_q_b_to_a",
        "pair_market_q","pair_market_rank",*safe_pair
    ]))
    base=pair[keep].copy()

    def side(ab):
        z=base.copy()
        sign=1.0 if ab else -1.0
        if ab:
            z["ticket"]=z["a_horse_number"].astype(int).astype(str)+">"+z["b_horse_number"].astype(int).astype(str)
            z["hit"]=z["hit_a_to_b"].astype("int8")
            z["return_yen"]=pd.to_numeric(z["return_a_to_b"],errors="coerce").fillna(0.0)
            z["odds"]=pd.to_numeric(z["odds_a_to_b"],errors="coerce")
            z["market_q_orientation"]=pd.to_numeric(z["market_q_a_to_b"],errors="coerce")
        else:
            z["ticket"]=z["b_horse_number"].astype(int).astype(str)+">"+z["a_horse_number"].astype(int).astype(str)
            z["hit"]=z["hit_b_to_a"].astype("int8")
            z["return_yen"]=pd.to_numeric(z["return_b_to_a"],errors="coerce").fillna(0.0)
            z["odds"]=pd.to_numeric(z["odds_b_to_a"],errors="coerce")
            z["market_q_orientation"]=pd.to_numeric(z["market_q_b_to_a"],errors="coerce")

        pairq=pd.to_numeric(z["pair_market_q"],errors="coerce").fillna(0.0).clip(lower=1e-15)
        z["market_direction_share"]=z["market_q_orientation"]/pairq
        z["market_direction_strength"]=(z["market_direction_share"]-0.5).abs()*2.0
        z["first_consensus_advantage"]=-sign*pd.to_numeric(z.get("dir_consensus_rank_diff",0.0),errors="coerce").fillna(0.0)
        z["first_mean_rank_advantage"]=-sign*pd.to_numeric(z.get("dir_mean_rank_diff",0.0),errors="coerce").fillna(0.0)
        z["first_top1_vote_advantage"]=sign*pd.to_numeric(z.get("dir_top1_votes_diff",0.0),errors="coerce").fillna(0.0)
        z["first_top3_support_advantage"]=sign*pd.to_numeric(z.get("dir_top3_support_diff",0.0),errors="coerce").fillna(0.0)
        z["first_top6_support_advantage"]=sign*pd.to_numeric(z.get("dir_top6_support_diff",0.0),errors="coerce").fillna(0.0)
        z["first_mean_probability_advantage"]=sign*pd.to_numeric(z.get("dir_mean_probability_diff",0.0),errors="coerce").fillna(0.0)
        z["expert_vote_for_first"]=sign*pd.to_numeric(z.get("dir_expert_vote_margin",0.0),errors="coerce").fillna(0.0)
        z["ticket_supported_by_market"]=(z["market_direction_share"]>=0.5).astype("int8")
        z["ticket_supported_by_l17"]=(z["expert_vote_for_first"]>0).astype("int8")
        z["market_l17_direction_agree"]=(
            ((z["market_direction_share"]-0.5)*z["expert_vote_for_first"])>0
        ).astype("int8")
        z["both_support_ticket"]=(
            (z["ticket_supported_by_market"]==1)&(z["ticket_supported_by_l17"]==1)
        ).astype("int8")
        cols=["year","race_id","race_date","ticket","hit","return_yen","odds",
              "market_q_orientation","pair_market_rank","market_direction_share",
              "market_direction_strength","first_consensus_advantage",
              "first_mean_rank_advantage","first_top1_vote_advantage",
              "first_top3_support_advantage","first_top6_support_advantage",
              "first_mean_probability_advantage","expert_vote_for_first",
              "ticket_supported_by_market","ticket_supported_by_l17",
              "market_l17_direction_agree","both_support_ticket",
              *[c for c in safe_pair if not c.startswith("dir_")]]
        return z[list(dict.fromkeys(cols))]

    out=pd.concat([side(True),side(False)],ignore_index=True,copy=False)
    out["ordered_market_rank"]=out.groupby(["year","race_id"])["market_q_orientation"].rank(
        method="min",ascending=False
    ).astype("int16")
    return out[out["pair_market_rank"]<=max(K_VALUES)].reset_index(drop=True)


def split_2022_halves(df):
    dates=sorted(df["race_date"].astype(str).unique())
    cut=len(dates)//2
    left=set(dates[:cut]); right=set(dates[cut:])
    return [("2022_H1",df[df["race_date"].astype(str).isin(left)].copy()),
            ("2022_H2",df[df["race_date"].astype(str).isin(right)].copy())]


def train_periods(frames,test_year):
    if test_year==2023:
        return split_2022_halves(frames[2022])
    return [(str(y),frames[y].copy()) for y in YEARS if y<test_year]


def atom_templates(train_all):
    atoms=[]
    for feat in NUMERIC_FEATURES:
        if feat not in train_all.columns:
            continue
        vals=pd.to_numeric(train_all[feat],errors="coerce").dropna()
        if vals.empty or vals.nunique()<4:
            continue
        for q in QUANTILES:
            thr=float(vals.quantile(q))
            qname=f"q{int(q*100):02d}"
            atoms.append({"feature":feat,"op":">=","q":qname,"threshold":thr,
                          "signature":f"{feat}>={qname}"})
            atoms.append({"feature":feat,"op":"<=","q":qname,"threshold":thr,
                          "signature":f"{feat}<={qname}"})
    for feat in BOOLEAN_FEATURES:
        if feat in train_all.columns:
            atoms.append({"feature":feat,"op":"==","q":"1","threshold":1.0,
                          "signature":f"{feat}==1"})
    # Broad categorical conditions only; support guards below reject tiny niches.
    for feat in ("surface","discipline","race_class_normalized"):
        if feat not in train_all.columns:
            continue
        vc=train_all[feat].fillna("__NA__").astype(str).value_counts(normalize=True)
        for val,share in vc.items():
            if share>=0.08:
                atoms.append({"feature":feat,"op":"==str","q":str(val),"threshold":str(val),
                              "signature":f"{feat}=={val}"})
    return atoms


def atom_mask(df,a):
    f=a["feature"]; op=a["op"]; thr=a["threshold"]
    if op=="==str":
        return df[f].fillna("__NA__").astype(str).to_numpy()==str(thr)
    x=pd.to_numeric(df[f],errors="coerce").fillna(0.0).to_numpy()
    if op==">=": return x>=float(thr)
    if op=="<=": return x<=float(thr)
    return x==float(thr)


def stats(df,mask):
    g=df.loc[mask]
    tickets=len(g)
    races=int(g["race_id"].nunique()) if tickets else 0
    stake=STAKE*tickets
    ret=float(g["return_yen"].sum()) if tickets else 0.0
    wins=g[g["return_yen"]>0].sort_values("return_yen",ascending=False)
    ret_minus_top1=ret-(float(wins.iloc[0]["return_yen"]) if len(wins) else 0.0)
    return {
        "tickets":tickets,"races":races,"return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else 0.0,
        "roi_minus_top1_pct":100.0*ret_minus_top1/stake if stake else 0.0,
        "hits":int(g["hit"].sum()) if tickets else 0,
    }


def evaluate_rule(periods,k,atoms):
    per=[]
    sig=" & ".join([f"pair_market_rank<={k}"]+[a["signature"] for a in atoms])
    for pname,pdf in periods:
        mask=(pdf["pair_market_rank"].to_numpy()<=k)
        for a in atoms:
            # thresholds are fold-level training thresholds and apply unchanged to every period.
            mask &= atom_mask(pdf,a)
        s=stats(pdf,mask); s["period"]=pname; per.append(s)
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
    mask=(df["pair_market_rank"].to_numpy()<=rule["k"])
    for a in rule["atoms"]:
        mask &= atom_mask(df,a)
    return stats(df,mask)


def main():
    a=parse_args(); t0=time.perf_counter()
    lp=parse_paths(a.l17_year)
    if set(lp)!=set(YEARS) or 2026 in lp:
        raise SystemExit("years must be exactly 2022-2025; 2026 sealed")

    frames={}
    for y in YEARS:
        t=time.perf_counter()
        pair=build_pair_year_frame(y,load_l17(lp[y],y),a.backfill_root)
        frames[y]=expand_oriented(pair)
        print("STABLE_LAW_YEAR_READY "+json.dumps({
            "year":y,"candidate_tickets_top15":len(frames[y]),
            "races":int(frames[y]["race_id"].nunique()),
            "seconds":round(time.perf_counter()-t,3)
        },separators=(",",":")),flush=True)
        del pair

    fold_selected=[]; test_rows=[]; threshold_rows=[]; discovery_rows=[]

    for test_year in TEST_YEARS:
        periods=train_periods(frames,test_year)
        train_all=pd.concat([x[1] for x in periods],ignore_index=True)
        atoms=atom_templates(train_all)
        for at in atoms:
            threshold_rows.append({
                "test_year":test_year,"feature":at["feature"],"op":at["op"],
                "quantile":at["q"],"threshold":at["threshold"],"signature":at["signature"]
            })

        singles=[]
        for k in K_VALUES:
            # base pair-rank law
            singles.append(evaluate_rule(periods,k,[]))
            for at in atoms:
                singles.append(evaluate_rule(periods,k,[at]))

        passing=[x for x in singles if x["stable_prior"]]
        passing.sort(key=lambda x:(
            x["robust_prior"],x["worst_prior_roi_minus_top1"],
            x["worst_prior_roi"],x["min_prior_races"],x["min_prior_tickets"]
        ),reverse=True)

        # Discover a limited second condition only from the strongest prior-stable singles.
        seeds=passing[:30]
        combos=[]
        for i in range(len(seeds)):
            for j in range(i+1,len(seeds)):
                a=seeds[i]; b=seeds[j]
                k=min(a["k"],b["k"])
                merged=[]
                seen=set()
                for at in a["atoms"]+b["atoms"]:
                    if at["signature"] not in seen:
                        merged.append(at); seen.add(at["signature"])
                if len(merged)!=2:
                    continue
                combos.append(evaluate_rule(periods,k,merged))

        candidates=passing+[x for x in combos if x["stable_prior"]]
        dedup={}
        for x in candidates:
            old=dedup.get(x["signature"])
            if old is None or (
                x["robust_prior"],x["worst_prior_roi_minus_top1"],x["worst_prior_roi"]
            )>(
                old["robust_prior"],old["worst_prior_roi_minus_top1"],old["worst_prior_roi"]
            ):
                dedup[x["signature"]]=x
        candidates=list(dedup.values())
        candidates.sort(key=lambda x:(
            x["robust_prior"],x["worst_prior_roi_minus_top1"],
            x["worst_prior_roi"],x["min_prior_races"],x["min_prior_tickets"]
        ),reverse=True)
        selected=candidates[:SHORTLIST_PER_FOLD]

        print("STABLE_LAW_FOLD "+json.dumps({
            "test_year":test_year,
            "atoms":len(atoms),
            "single_rules":len(singles),
            "prior_stable_singles":len(passing),
            "stable_candidates_with_combos":len(candidates),
            "shortlisted":len(selected),
            "robust_shortlisted":sum(1 for x in selected if x["robust_prior"]),
        },separators=(",",":")),flush=True)

        for rank,rule in enumerate(selected,1):
            ts=apply_rule(frames[test_year],rule)
            row={
                "test_year":test_year,"prior_rank":rank,"signature":rule["signature"],
                "robust_prior":rule["robust_prior"],
                "worst_prior_roi":rule["worst_prior_roi"],
                "worst_prior_roi_minus_top1":rule["worst_prior_roi_minus_top1"],
                "median_prior_roi":rule["median_prior_roi"],
                "min_prior_tickets":rule["min_prior_tickets"],
                "min_prior_races":rule["min_prior_races"],
                **{f"test_{k}":v for k,v in ts.items()},
            }
            test_rows.append(row)
            fold_selected.append((test_year,rule))
            for ps in rule["period_stats"]:
                discovery_rows.append({
                    "test_year":test_year,"prior_rank":rank,"signature":rule["signature"],
                    "robust_prior":rule["robust_prior"],**ps
                })

    test_df=pd.DataFrame(test_rows)
    recurrence=[]
    if not test_df.empty:
        for sig,g in test_df.groupby("signature",sort=False):
            rois=g["test_roi_pct"].astype(float).tolist()
            recurrence.append({
                "signature":sig,
                "selected_folds":len(g),
                "selected_test_years":"|".join(str(x) for x in sorted(g["test_year"].astype(int).tolist())),
                "profitable_unseen_years":int(sum(x>100.0 for x in rois)),
                "nonlosing_unseen_years":int(sum(x>=100.0 for x in rois)),
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

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    test_df.to_csv(out/"unseen-year-results.csv",index=False)
    pd.DataFrame(discovery_rows).to_csv(out/"prior-period-audit.csv",index=False)
    pd.DataFrame(threshold_rows).to_csv(out/"learned-thresholds.csv",index=False)
    rec.to_csv(out/"cross-year-recurrence.csv",index=False)

    stable3=rec[(rec["selected_folds"]==3)&(rec["profitable_unseen_years"]==3)] if not rec.empty else rec
    summary={
        "contract":"L2_EXACTA_STABLE_LAW_V1",
        "foundation":"decomposed unordered pair market rank; top15 candidate universe",
        "why_this_foundation":"pair capture was stable across 2023-2025; law search focuses on how to bet those pairs",
        "search":"pair rank 1..15 plus data-derived q20/q40/q60/q80 thresholds on market/L1.7/race features; stable singles then two-condition intersections",
        "prior_gate":f"ROI>100 in every independent prior period with >= {MIN_TICKETS_PER_PERIOD} tickets and >= {MIN_RACES_PER_PERIOD} races per period",
        "robust_flag":"also ROI>100 after removing largest winning payout in every prior period",
        "test_policy":"shortlist determined only from prior periods; unseen year never used for law selection",
        "2023_prior_periods":["2022_H1","2022_H2"],
        "2024_prior_periods":["2022","2023"],
        "2025_prior_periods":["2022","2023","2024"],
        "shortlist_per_fold":SHORTLIST_PER_FOLD,
        "same_signature_selected_all_3_and_profitable_all_3":int(len(stable3)),
        "2026_locked":True,
        "script_total_seconds":time.perf_counter()-t0,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 EXACTA Stable LAW V1\n\n"
        "This search does not optimize on the unseen test year. It starts from the stable unordered-pair market ranking "
        "and searches simple interpretable betting laws that combine pair rank with market direction, exacta price, "
        "L1.7 pair quality, L1.7 directional advantages, and broad race context. Numeric thresholds are derived from "
        "prior-data quantiles rather than fixed payout/odds cutoffs. A law must be profitable in every independent "
        "prior period and meet support guards before it can be tested on the next year. A robust flag additionally "
        "requires profitability after removing the single largest winning payout in every prior period. "
        "The final report emphasizes recurring law signatures and unseen-year reproducibility. 2026 is sealed.\n",
        encoding="utf-8"
    )
    print("L2_EXACTA_STABLE_LAW_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
    print("===== CROSS YEAR RECURRENCE TOP =====")
    print(rec.head(40).to_csv(index=False) if not rec.empty else "NONE")
    print("===== UNSEEN YEAR TOP =====")
    print(test_df.head(40).to_csv(index=False) if not test_df.empty else "NONE")


if __name__=="__main__":
    main()
