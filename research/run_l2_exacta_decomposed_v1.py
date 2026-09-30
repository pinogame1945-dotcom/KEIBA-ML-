#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import decode_odds, payout_map, horse_number_map
from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_exacta_v0 import (
    EXPECTED_RACES_PER_YEAR,
    TOP_K,
    finite,
    read_jsonl_gz,
    race_features,
    pair_premarket_features,
    encode,
)

YEARS=(2022,2023,2024,2025)
TEST_YEARS=(2023,2024,2025)

ID_COLS={
    "year","race_id","race_date","pair_horse_ids","pair_numbers",
    "a_horse_id","b_horse_id","a_horse_number","b_horse_number",
    "pair_hit","hit_a_to_b","hit_b_to_a","return_a_to_b","return_b_to_a","direction_label","direction_eligible",
    "strict_direction_eligible","multi_direction_hit",
    "odds_a_to_b","odds_b_to_a","market_q_a_to_b","market_q_b_to_a",
    "pair_market_q","pair_market_log_q","pair_market_rank",
    "pair_l17_score","pair_l17_rank","pair_market_aware_score","pair_model_rank",
    "dir_l17_prob","dir_market_aware_prob",
}
FORBIDDEN=("odds","payout","return","profit","roi","hit","finish","result","popularity","target")


def parse_args():
    p=argparse.ArgumentParser(description="EXACTA V1 decomposed model: unordered pair ranker + direction classifier.")
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--baseline-direction",required=True)
    p.add_argument("--baseline-coverage",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1)
        out[int(y)]=p
    return out


def build_pair_year_frame(year,l17_rows,backfill_root):
    t0=time.perf_counter()
    if len(l17_rows)!=EXPECTED_RACES_PER_YEAR:
        raise SystemExit(f"L1.7 race coverage drift y={year}: {len(l17_rows)}")

    expert_names=sorted({
        n for r in l17_rows.values()
        for h in r.get("horses",[])
        for n in (h.get("experts") or {})
    })
    if len(expert_names)!=7:
        raise SystemExit(f"expected seven experts y={year}: {expert_names}")

    root=Path(backfill_root)
    rows=[]
    priced_races=0
    missing_reverse_pairs=0
    direction_eligible_races=0
    multi_direction_races=0

    for day_path in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
        date=day_path.name[:10]
        odds_path=root/"data"/"odds"/"daily"/day_path.name
        if not odds_path.exists():
            raise SystemExit(f"missing odds day {date}")
        odds_by_race={str(x.get("race_id") or ""):x for x in read_jsonl_gz(odds_path)}

        for pack in read_jsonl_gz(day_path):
            race=pack.get("race") or {}
            rid=str(race.get("race_id") or "")
            rec=l17_rows.get(rid)
            if rec is None:
                continue
            odds_rec=odds_by_race.get(rid)
            if odds_rec is None:
                continue

            hno=horse_number_map(pack)
            horses=list(rec["horses"])
            if len(horses)!=int(rec["field_size"]):
                raise SystemExit(f"L1.7 field drift race={rid}")
            horses=sorted(horses,key=lambda h:(int(hno[str(h["horse_id"])]),str(h["horse_id"])))

            odds_map=decode_odds(odds_rec)
            payouts,present=payout_map(pack)
            if "EXACTA" not in present:
                continue

            exacta_odds={
                nums:float(price)
                for (bt,nums),price in odds_map.items()
                if bt=="EXACTA" and price and price>0
            }
            if not exacta_odds:
                continue
            inv={k:1.0/v for k,v in exacta_odds.items()}
            denom=sum(inv.values())
            if denom<=0:
                continue

            base=race_features(pack,rec,date)
            race_rows=[]
            race_unique_direction=0
            race_multi_direction=0

            for i in range(len(horses)):
                for j in range(i+1,len(horses)):
                    ha=horses[i]; hb=horses[j]
                    aid=str(ha["horse_id"]); bid=str(hb["horse_id"])
                    an=int(hno[aid]); bn=int(hno[bid])
                    if not an<bn:
                        raise SystemExit(f"canonical pair order drift race={rid} {an},{bn}")

                    oab=exacta_odds.get((an,bn))
                    oba=exacta_odds.get((bn,an))
                    if oab is None or oba is None:
                        missing_reverse_pairs+=1
                        continue

                    qab=inv[(an,bn)]/denom
                    qba=inv[(bn,an)]/denom
                    rab=float(payouts.get(("EXACTA",(an,bn)),0.0))
                    rba=float(payouts.get(("EXACTA",(bn,an)),0.0))
                    hab=int(rab>0)
                    hba=int(rba>0)
                    pair_hit=int(hab or hba)
                    direction_eligible=int(hab+hba==1)
                    multi_direction_hit=int(hab+hba>1)
                    if direction_eligible:
                        label=int(hab==1)  # 1 = canonical low-number A -> B, 0 = B -> A
                        race_unique_direction+=1
                    else:
                        label=-1
                    race_multi_direction += multi_direction_hit

                    feats=pair_premarket_features(ha,hb,expert_names)
                    feats["dir_market_q_diff"]=qab-qba
                    feats["dir_market_log_ratio"]=math.log(max(qab,1e-15))-math.log(max(qba,1e-15))

                    row={
                        "year":year,
                        "race_id":rid,
                        "race_date":date,
                        "pair_horse_ids":f"{aid}|{bid}",
                        "pair_numbers":f"{an}-{bn}",
                        "a_horse_id":aid,
                        "b_horse_id":bid,
                        "a_horse_number":an,
                        "b_horse_number":bn,
                        "pair_hit":pair_hit,
                        "hit_a_to_b":hab,
                        "hit_b_to_a":hba,
                        "return_a_to_b":rab,
                        "return_b_to_a":rba,
                        "direction_label":label,
                        "direction_eligible":direction_eligible,
                        "strict_direction_eligible":0,
                        "multi_direction_hit":multi_direction_hit,
                        "odds_a_to_b":oab,
                        "odds_b_to_a":oba,
                        "market_q_a_to_b":qab,
                        "market_q_b_to_a":qba,
                        "pair_market_q":qab+qba,
                        "pair_market_log_q":math.log(max(qab+qba,1e-15)),
                        **base,
                        **feats,
                    }
                    race_rows.append(row)

            if race_rows:
                priced_races+=1
                direction_eligible_races += int(race_unique_direction==1)
                multi_direction_races += int(race_multi_direction>0)
                rows.extend(race_rows)

    if not rows:
        raise SystemExit(f"no pair rows y={year}")

    df=pd.DataFrame(rows)
    df["pair_market_rank"]=df.groupby(["year","race_id"])["pair_market_q"].rank(
        method="min",ascending=False
    ).astype(int)
    race_hits=df.groupby(["year","race_id"])["pair_hit"].transform("sum")
    df["strict_direction_eligible"]=(
        (race_hits==1) & (df["direction_eligible"]==1) & (df["pair_hit"]==1)
    ).astype(int)
    strict_races=int(df.loc[df["strict_direction_eligible"]==1,"race_id"].nunique())

    print("EXACTA_PAIR_YEAR_READY "+json.dumps({
        "year":year,
        "priced_races":priced_races,
        "pair_rows":len(df),
        "positive_pairs":int(df["pair_hit"].sum()),
        "direction_eligible_races":strict_races,
        "multi_direction_races":multi_direction_races,
        "missing_reverse_pairs":missing_reverse_pairs,
        "seconds":round(time.perf_counter()-t0,3),
    },separators=(",",":")),flush=True)
    return df


def pair_feature_columns(df,market=False):
    cols=[]
    for c in df.columns:
        if c in ID_COLS or c.startswith("dir_"):
            continue
        if any(tok in c.lower() for tok in FORBIDDEN):
            continue
        if c in {"a_horse_number","b_horse_number"}:
            continue
        cols.append(c)
    if market:
        cols += ["pair_market_q","pair_market_log_q","pair_market_rank"]
    return list(dict.fromkeys(cols))


def direction_feature_columns(df,market=False):
    # Directional signed features plus symmetric context. Horse numbers are never used.
    cols=[]
    for c in df.columns:
        if c in ID_COLS:
            continue
        if any(tok in c.lower() for tok in FORBIDDEN):
            continue
        if c.startswith("dir_market_"):
            if market:
                cols.append(c)
            continue
        if c.startswith("dir_"):
            cols.append(c)
            continue
        if c.startswith(("pair_","expert_")) or c in {
            "race_month","venue_code","discipline","surface","direction","weather",
            "track_condition","course_layout","race_class_normalized","grade",
            "sex_condition","weight_rule","distance_m","field_size",
        }:
            cols.append(c)
    return list(dict.fromkeys(cols))


def train_pair_ranker(train,test,cols,seed):
    t0=time.perf_counter()
    good=train.groupby(["year","race_id"])["pair_hit"].transform("sum")>0
    tr=train.loc[good].copy().sort_values(["year","race_id","pair_numbers"]).reset_index(drop=True)
    te=test.copy().sort_values(["year","race_id","pair_numbers"])
    te["_orig_index"]=te.index
    te=te.reset_index(drop=True)
    xtr,xte=encode(tr,te,cols)
    y=tr["pair_hit"].astype(int).to_numpy()
    group=tr.groupby(["year","race_id"],sort=False).size().tolist()

    model=lgb.LGBMRanker(
        objective="lambdarank",metric="ndcg",
        n_estimators=280,learning_rate=0.035,num_leaves=31,
        min_child_samples=80,subsample=0.90,colsample_bytree=0.90,
        reg_lambda=4.0,reg_alpha=0.5,random_state=seed,
        n_jobs=2,deterministic=True,force_col_wise=True,verbosity=-1,
    )
    model.fit(xtr,y,group=group)
    pred=np.asarray(model.predict(xte),dtype=float)
    s=pd.Series(pred,index=te["_orig_index"].astype(int).to_numpy())
    out=s.reindex(test.index).to_numpy()
    imp=pd.DataFrame({"feature":cols,"gain":model.feature_importances_})
    return out,imp,time.perf_counter()-t0


def train_direction_classifier(train,test,cols,seed):
    t0=time.perf_counter()
    tr=train[train["strict_direction_eligible"]==1].copy().sort_values(
        ["year","race_id","pair_numbers"]
    ).reset_index(drop=True)
    if tr.empty or tr["direction_label"].nunique()!=2:
        raise SystemExit("direction training labels invalid")
    te=test.copy()
    te["_orig_index"]=te.index
    te=te.sort_values(["year","race_id","pair_numbers"]).reset_index(drop=True)
    xtr,xte=encode(tr,te,cols)
    y=tr["direction_label"].astype(int).to_numpy()

    model=lgb.LGBMClassifier(
        objective="binary",n_estimators=220,learning_rate=0.035,num_leaves=15,
        min_child_samples=60,subsample=0.90,colsample_bytree=0.90,
        reg_lambda=4.0,reg_alpha=0.5,random_state=seed,
        n_jobs=2,deterministic=True,force_col_wise=True,verbosity=-1,
    )
    model.fit(xtr,y)
    pred=np.asarray(model.predict_proba(xte)[:,1],dtype=float)
    s=pd.Series(pred,index=te["_orig_index"].astype(int).to_numpy())
    out=s.reindex(test.index).to_numpy()
    imp=pd.DataFrame({"feature":cols,"gain":model.feature_importances_})
    return out,imp,time.perf_counter()-t0,len(tr)


def _direction_prediction(frame,dir_mode):
    if dir_mode=="market":
        return np.where(
            frame["market_q_a_to_b"]>frame["market_q_b_to_a"],1,
            np.where(frame["market_q_a_to_b"]<frame["market_q_b_to_a"],0,-1)
        )
    if dir_mode=="l17":
        return (frame["dir_l17_prob"].to_numpy()>=0.5).astype(int)
    if dir_mode=="market_aware":
        return (frame["dir_market_aware_prob"].to_numpy()>=0.5).astype(int)
    raise ValueError(dir_mode)


def evaluate_variant(df,year,label,pair_rank_col,dir_mode):
    # Direction accuracy: strict unique exacta winner only, matching V0.
    positive=df[df["strict_direction_eligible"]==1].copy()
    if positive.empty:
        return []
    if positive["race_id"].duplicated().any():
        raise SystemExit(f"multiple strict direction pairs y={year}")

    positive["_dir_pred"]=_direction_prediction(positive,dir_mode)
    positive["_dir_correct"]=positive["_dir_pred"]==positive["direction_label"]
    positive["_dir_tie"]=positive["_dir_pred"]<0
    source_direction=len(positive)

    # End-to-end exact hit coverage: all priced races, including dead-heat multi-payout races.
    all_pairs=df.copy()
    all_pairs["_dir_pred"]=_direction_prediction(all_pairs,dir_mode)
    all_pairs["_pred_hit"]=(
        ((all_pairs["_dir_pred"]==1) & (all_pairs["hit_a_to_b"]==1)) |
        ((all_pairs["_dir_pred"]==0) & (all_pairs["hit_b_to_a"]==1))
    )
    source_all=all_pairs["race_id"].nunique()

    rows=[]
    for k in TOP_K:
        cap=positive[positive[pair_rank_col]<=k]
        correct=int(cap["_dir_correct"].sum())
        ties=int(cap["_dir_tie"].sum())
        reverse=len(cap)-correct-ties

        chosen=all_pairs[all_pairs[pair_rank_col]<=k]
        exact_hit_races=int(chosen.loc[chosen["_pred_hit"],"race_id"].nunique())
        rows.append({
            "year":year,
            "model":label,
            "top_k":k,
            "source_direction_races":source_direction,
            "pair_captured_races":len(cap),
            "pair_capture_pct":100.0*len(cap)/source_direction if source_direction else 0.0,
            "direction_correct_races":correct,
            "direction_reverse_races":reverse,
            "direction_tie_races":ties,
            "direction_correct_pct_of_captured":100.0*correct/len(cap) if len(cap) else 0.0,
            "direction_correct_pct_excluding_ties":100.0*correct/(correct+reverse) if correct+reverse else 0.0,
            "exact_hit_races_top_k":exact_hit_races,
            "exact_hit_coverage_pct":100.0*exact_hit_races/source_all if source_all else 0.0,
            "ticket_budget_semantics":"one chosen orientation per unordered pair",
        })
    return rows


def add_direct_baseline(rows,direction_path,coverage_path):
    d=pd.read_csv(direction_path)
    c=pd.read_csv(coverage_path)
    labels={
        "MARKET":"DIRECT_MARKET",
        "L17_ONLY_MODEL":"DIRECT_L17_ONLY",
        "MARKET_AWARE_MODEL":"DIRECT_MARKET_AWARE",
    }
    out=[]
    for year in TEST_YEARS:
        for old,new in labels.items():
            for k in TOP_K:
                dr=d[(d.year==year)&(d.ranking==old)&(d.top_k==k)]
                cr=c[(c.year==year)&(c.ranking==old)&(c.top_k==k)]
                if len(dr)!=1 or len(cr)!=1:
                    raise SystemExit(f"baseline missing y={year} {old} k={k}")
                x=dr.iloc[0]; y=cr.iloc[0]
                out.append({
                    "year":year,"model":new,"top_k":k,
                    "source_direction_races":int(x.direction_eligible_races),
                    "pair_captured_races":int(x.pair_captured_races),
                    "pair_capture_pct":float(x.pair_capture_pct),
                    "direction_correct_races":int(x.direction_correct_races),
                    "direction_reverse_races":int(x.direction_reverse_races),
                    "direction_tie_races":int(x.direction_tie_races),
                    "direction_correct_pct_of_captured":float(x.direction_correct_pct_of_captured),
                    "direction_correct_pct_excluding_ties":float(x.direction_correct_pct_excluding_ties),
                    "exact_hit_races_top_k":int(y.hit_races),
                    "exact_hit_coverage_pct":float(y.exact_hit_coverage_pct),
                    "ticket_budget_semantics":"top K ordered exacta tickets",
                })
    return out+rows


def main():
    a=parse_args()
    t_all=time.perf_counter()
    lp=parse_paths(a.l17_year)
    if set(lp)!=set(YEARS) or 2026 in lp:
        raise SystemExit("L1.7 years must be exactly 2022-2025; 2026 sealed")

    build_seconds=0.0
    frames={}
    for y in YEARS:
        t=time.perf_counter()
        frames[y]=build_pair_year_frame(y,load_l17(lp[y],y),a.backfill_root)
        build_seconds+=time.perf_counter()-t

    pair_l17_cols=pair_feature_columns(frames[2022],market=False)
    pair_market_cols=pair_feature_columns(frames[2022],market=True)
    dir_l17_cols=direction_feature_columns(frames[2022],market=False)
    dir_market_cols=direction_feature_columns(frames[2022],market=True)

    # Signed reversal guard: canonical A/B reversal is already audited in V0.
    # Here explicitly prevent directional market features from entering pair rankers.
    if any(c.startswith("dir_") for c in pair_l17_cols+pair_market_cols):
        raise SystemExit("directional leakage into unordered pair ranker")
    if any("market" in c.lower() for c in pair_l17_cols+dir_l17_cols):
        raise SystemExit("market leakage into L1.7-only decomposed path")

    all_eval=[]
    fold_rows=[]
    imps=[]
    training_seconds=0.0

    for y in TEST_YEARS:
        train=pd.concat([frames[t] for t in YEARS if t<y],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)

        s1,i1,sec1=train_pair_ranker(train,test,pair_l17_cols,98100+y)
        s2,i2,sec2=train_pair_ranker(train,test,pair_market_cols,98200+y)
        d1,j1,sec3,n_dir=train_direction_classifier(train,test,dir_l17_cols,98300+y)
        d2,j2,sec4,_=train_direction_classifier(train,test,dir_market_cols,98400+y)
        training_seconds += sec1+sec2+sec3+sec4

        test["pair_l17_score"]=s1
        test["pair_market_aware_score"]=s2
        test["pair_l17_rank"]=test.groupby("race_id")["pair_l17_score"].rank(method="min",ascending=False).astype(int)
        test["pair_model_rank"]=test.groupby("race_id")["pair_market_aware_score"].rank(method="min",ascending=False).astype(int)
        test["dir_l17_prob"]=d1
        test["dir_market_aware_prob"]=d2

        all_eval += evaluate_variant(test,y,"DECOMP_MARKET","pair_market_rank","market")
        all_eval += evaluate_variant(test,y,"DECOMP_L17_ONLY","pair_l17_rank","l17")
        all_eval += evaluate_variant(test,y,"DECOMP_MARKET_AWARE","pair_model_rank","market_aware")

        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(str(t) for t in YEARS if t<y),
            "train_pair_rows":len(train),
            "test_pair_rows":len(test),
            "train_direction_rows":n_dir,
            "pair_l17_features":len(pair_l17_cols),
            "pair_market_features":len(pair_market_cols),
            "direction_l17_features":len(dir_l17_cols),
            "direction_market_features":len(dir_market_cols),
            "pair_l17_train_seconds":sec1,
            "pair_market_train_seconds":sec2,
            "direction_l17_train_seconds":sec3,
            "direction_market_train_seconds":sec4,
        })
        for imp,name in [(i1,"PAIR_L17"),(i2,"PAIR_MARKET"),(j1,"DIR_L17"),(j2,"DIR_MARKET")]:
            imp["test_year"]=y; imp["model"]=name; imps.append(imp)

    comparison=add_direct_baseline(all_eval,a.baseline_direction,a.baseline_coverage)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(comparison).to_csv(out/"direct-vs-decomposed.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(out/"fold-metrics.csv",index=False)
    pd.concat(imps,ignore_index=True).to_csv(out/"feature-importance.csv",index=False)

    # Compact primary view: top10 only, all six comparable variants.
    top10=pd.DataFrame(comparison)
    top10=top10[top10.top_k==10].copy()
    top10.to_csv(out/"top10-primary.csv",index=False)

    summary={
        "contract":"L2_EXACTA_DECOMPOSED_V1_RESULT",
        "bet_type":"EXACTA",
        "design":"unordered pair ranker + one orientation classifier",
        "speed_design":{
            "ordered_pairs_not_retrained":True,
            "prior_direct_v0_reused_as_frozen_baseline":True,
            "all_bet_decoder_audit_not_repeated":True,
            "unordered_pair_rows":"N*(N-1)/2",
            "direction_training_rows":"only unique winning first-two pairs",
            "pair_models_per_fold":2,
            "direction_models_per_fold":2,
            "build_seconds":build_seconds,
            "model_training_seconds":training_seconds,
            "script_total_seconds":time.perf_counter()-t_all,
        },
        "variants":[
            "DIRECT_MARKET","DIRECT_L17_ONLY","DIRECT_MARKET_AWARE",
            "DECOMP_MARKET","DECOMP_L17_ONLY","DECOMP_MARKET_AWARE",
        ],
        "primary_metrics":[
            "pair_capture_pct",
            "direction_correct_pct_of_captured",
            "exact_hit_coverage_pct",
        ],
        "walk_forward":{"test_years":list(TEST_YEARS),"training":"all prior years only"},
        "law_search":False,
        "roi_policy_search":False,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 EXACTA V1 — Direct vs Decomposed\n\n"
        "The frozen corrected V0 direct model is reused as the baseline and is not retrained. "
        "The new path halves the candidate matrix to unordered first-two pairs, ranks those pairs, "
        "then chooses exactly one orientation with a separate direction classifier. "
        "This compares equal ticket budgets at Top-K: direct uses K ordered tickets; decomposed uses "
        "K unordered pairs with one chosen orientation each. No LAW/ROI search is performed. "
        "2026 outcomes remain sealed.\n",
        encoding="utf-8"
    )

    print("EXACTA_DECOMPOSED_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
    print("===== TOP10 PRIMARY =====")
    print(top10.to_csv(index=False))


if __name__=="__main__":
    main()
