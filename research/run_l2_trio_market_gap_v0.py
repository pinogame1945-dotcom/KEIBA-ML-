#!/usr/bin/env python3
import argparse
import gzip
import itertools
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import decode_odds, payout_map, horse_number_map, canonical_numbers
from build_l2_l17_fullfield_dataset_v1 import load_l17

YEARS=(2021,2022,2023,2024,2025)
TEST_YEARS=(2022,2023,2024,2025)
EXPECTED_RACES_PER_YEAR=3456
TOP_KS=(1,3,5,10,20)

IDENTITY={
    "year","race_id","race_date","trio_horse_ids","trio_numbers",
    "hit","return_yen_per100","odds"
}
FORBIDDEN_MODEL_TOKENS=(
    "payout","return","profit","roi","hit","finish","result","popularity","target","horse_number"
)

def parse_args():
    p=argparse.ArgumentParser(description="TRIO V0: all combinations, L1.7/market/model rank comparison only.")
    for y in YEARS:
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--work-dir",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def finite(v,default=0.0):
    try:
        x=float(v)
        return x if math.isfinite(x) else default
    except (TypeError,ValueError):
        return default

def read_jsonl_gz(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def race_features(pack,date,field_size):
    race=pack.get("race") or {}
    return {
        "race_month":finite(date[5:7]),
        "venue_code":str(race.get("venue_code") or "UNKNOWN"),
        "discipline":str(race.get("discipline") or "UNKNOWN"),
        "surface":str(race.get("surface") or "UNKNOWN"),
        "direction":str(race.get("direction") or "UNKNOWN"),
        "weather":str(race.get("weather") or "UNKNOWN"),
        "track_condition":str(race.get("track_condition") or "UNKNOWN"),
        "course_layout":str(race.get("course_layout") or "UNKNOWN"),
        "race_class_normalized":str(race.get("race_class_normalized") or "UNKNOWN"),
        "grade":str(race.get("grade") or "UNKNOWN"),
        "sex_condition":str(race.get("sex_condition") or "UNKNOWN"),
        "weight_rule":str(race.get("weight_rule") or "UNKNOWN"),
        "distance_m":finite(race.get("distance_m")),
        "field_size":int(field_size),
    }

def horse_features(prefix,h,row):
    row[prefix+"consensus_rank"]=int(h["consensus_rank"])
    row[prefix+"mean_rank"]=finite(h.get("mean_rank"),99.0)
    row[prefix+"rank_std"]=finite(h.get("rank_std"))
    row[prefix+"top1_votes"]=finite(h.get("top1_votes"))
    row[prefix+"top3_support"]=finite(h.get("top3_support"))
    row[prefix+"top6_support"]=finite(h.get("top6_support"))
    row[prefix+"mean_probability"]=finite(h.get("mean_probability"))
    row[prefix+"probability_std"]=finite(h.get("probability_std"))

def trio_features(year,rid,date,pack,rec,hs,odd,ret):
    hs=sorted(hs,key=lambda h:int(h["consensus_rank"]))
    ranks=[int(h["consensus_rank"]) for h in hs]
    probs=[finite(h.get("mean_probability")) for h in hs]
    top3=[finite(h.get("top3_support")) for h in hs]
    top6=[finite(h.get("top6_support")) for h in hs]
    rstd=[finite(h.get("rank_std")) for h in hs]
    pstd=[finite(h.get("probability_std")) for h in hs]

    row={
        "year":year,
        "race_id":rid,
        "race_date":date,
        "trio_horse_ids":"|".join(str(h["horse_id"]) for h in hs),
        "trio_numbers":"-".join(str(int(h["horse_number"])) for h in hs),
        "hit":int(ret>0),
        "return_yen_per100":float(ret),
        "odds":float(odd),
    }
    row.update(race_features(pack,date,rec["field_size"]))
    for prefix,h in zip(("a_","b_","c_"),hs):
        horse_features(prefix,h,row)

    row.update({
        "trio_rank_sum":sum(ranks),
        "trio_best_rank":min(ranks),
        "trio_middle_rank":sorted(ranks)[1],
        "trio_worst_rank":max(ranks),
        "trio_rank_spread":max(ranks)-min(ranks),
        "trio_top3_count":sum(r<=3 for r in ranks),
        "trio_top6_count":sum(r<=6 for r in ranks),
        "trio_top10_count":sum(r<=10 for r in ranks),
        "trio_outside_top6_count":sum(r>6 for r in ranks),
        "trio_includes_consensus1":int(1 in ranks),
        "trio_prob_sum":sum(probs),
        "trio_prob_product":float(np.prod(probs)),
        "trio_prob_min":min(probs),
        "trio_prob_max":max(probs),
        "trio_prob_spread":max(probs)-min(probs),
        "trio_top3_support_sum":sum(top3),
        "trio_top3_support_min":min(top3),
        "trio_top6_support_sum":sum(top6),
        "trio_top6_support_min":min(top6),
        "trio_rank_std_mean":float(np.mean(rstd)),
        "trio_rank_std_max":max(rstd),
        "trio_probability_std_mean":float(np.mean(pstd)),
        "trio_probability_std_max":max(pstd),
    })

    expert_names=sorted(set.intersection(*[
        set((h.get("experts") or {}).keys()) for h in hs
    ]))
    if len(expert_names)!=7:
        raise ValueError(f"race={rid}: trio expert coverage expected=7 got={expert_names}")

    exp_rank_sums=[]
    exp_rank_max=[]
    exp_rank_spreads=[]
    exp_top3_counts=[]
    exp_top6_counts=[]
    exp_prob_products=[]
    all_top6=0
    all_top10=0
    for name in expert_names:
        er=[finite((h.get("experts") or {}).get(name,{}).get("rank"),99.0) for h in hs]
        ep=[finite((h.get("experts") or {}).get(name,{}).get("probability")) for h in hs]
        exp_rank_sums.append(sum(er))
        exp_rank_max.append(max(er))
        exp_rank_spreads.append(max(er)-min(er))
        exp_top3_counts.append(sum(r<=3 for r in er))
        exp_top6_counts.append(sum(r<=6 for r in er))
        exp_prob_products.append(float(np.prod(ep)))
        all_top6+=int(max(er)<=6)
        all_top10+=int(max(er)<=10)

    row.update({
        "experts_rank_sum_mean":float(np.mean(exp_rank_sums)),
        "experts_rank_sum_std":float(np.std(exp_rank_sums)),
        "experts_rank_sum_min":min(exp_rank_sums),
        "experts_rank_sum_max":max(exp_rank_sums),
        "experts_worst_rank_mean":float(np.mean(exp_rank_max)),
        "experts_worst_rank_std":float(np.std(exp_rank_max)),
        "experts_worst_rank_min":min(exp_rank_max),
        "experts_worst_rank_max":max(exp_rank_max),
        "experts_rank_spread_mean":float(np.mean(exp_rank_spreads)),
        "experts_rank_spread_max":max(exp_rank_spreads),
        "experts_top3_members_mean":float(np.mean(exp_top3_counts)),
        "experts_top3_members_std":float(np.std(exp_top3_counts)),
        "experts_top6_members_mean":float(np.mean(exp_top6_counts)),
        "experts_top6_members_std":float(np.std(exp_top6_counts)),
        "experts_all_three_top6_count":all_top6,
        "experts_all_three_top10_count":all_top10,
        "experts_trio_prob_product_mean":float(np.mean(exp_prob_products)),
        "experts_trio_prob_product_std":float(np.std(exp_prob_products)),
        "experts_trio_prob_product_min":min(exp_prob_products),
        "experts_trio_prob_product_max":max(exp_prob_products),
    })
    return row

def build_year_frame(year,l17_rows,backfill_root):
    if len(l17_rows)!=EXPECTED_RACES_PER_YEAR:
        raise SystemExit(f"L1.7 race coverage drift y={year}: {len(l17_rows)}")

    root=Path(backfill_root)
    rows=[]
    priced_races=0
    incomplete_market_races=0
    no_positive_races=0
    multi_positive_races=0
    missing_odds_days=[]

    for day_path in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
        date=day_path.name[:10]
        odds_path=root/"data"/"odds"/"daily"/day_path.name
        if not odds_path.exists():
            missing_odds_days.append(date)
            continue
        odds_by_race={str(x.get("race_id") or ""):x for x in read_jsonl_gz(odds_path)}
        for pack in read_jsonl_gz(day_path):
            race=pack.get("race") or {}
            rid=str(race.get("race_id") or "")
            if rid not in l17_rows:
                continue
            rec=l17_rows[rid]
            odds_rec=odds_by_race.get(rid)
            if odds_rec is None:
                continue

            hno=horse_number_map(pack)
            horses=sorted(rec["horses"],key=lambda x:int(x["consensus_rank"]))
            if len(horses)!=int(rec["field_size"]):
                raise SystemExit(f"L1.7 field drift race={rid}")
            if any(str(h["horse_id"]) not in hno for h in horses):
                raise SystemExit(f"horse missing from race pack race={rid}")
            for h in horses:
                h["horse_number"]=hno[str(h["horse_id"])]

            odds_map=decode_odds(odds_rec)
            payouts,present=payout_map(pack)
            if "TRIO" not in present:
                continue

            combos=list(itertools.combinations(horses,3))
            priced=[]
            missing=0
            for hs in combos:
                nums=[hno[str(h["horse_id"])] for h in hs]
                key=("TRIO",canonical_numbers("TRIO",nums))
                odd=odds_map.get(key)
                if odd is None:
                    missing+=1
                    continue
                ret=float(payouts.get(key,0.0))
                priced.append((hs,odd,ret))

            if missing or len(priced)!=len(combos):
                incomplete_market_races+=1
                continue
            positives=sum(ret>0 for _,_,ret in priced)
            if positives==0:
                no_positive_races+=1
                continue
            if positives>1:
                multi_positive_races+=1

            for hs,odd,ret in priced:
                rows.append(trio_features(year,rid,date,pack,rec,hs,odd,ret))
            priced_races+=1

    if not rows:
        raise SystemExit(f"no complete TRIO rows y={year}")
    df=pd.DataFrame(rows)
    for c in [
        "venue_code","discipline","surface","direction","weather","track_condition",
        "course_layout","race_class_normalized","grade","sex_condition","weight_rule"
    ]:
        df[c]=df[c].astype("category")
    for c in df.select_dtypes(include=["float64"]).columns:
        if c not in {"return_yen_per100","odds"}:
            df[c]=pd.to_numeric(df[c],downcast="float")
    for c in df.select_dtypes(include=["int64"]).columns:
        if c not in {"year"}:
            df[c]=pd.to_numeric(df[c],downcast="integer")

    summary={
        "year":year,
        "l17_races":len(l17_rows),
        "complete_priced_races":priced_races,
        "priced_trios":len(df),
        "positive_trios":int(df["hit"].sum()),
        "incomplete_market_races_skipped":incomplete_market_races,
        "no_positive_races_skipped":no_positive_races,
        "multi_positive_deadheat_races":multi_positive_races,
        "missing_odds_days":missing_odds_days,
    }
    print("TRIO_V0_YEAR_READY "+json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)
    return df,summary

def add_market_features(df):
    z=df.copy()
    z["market_inv_odds"]=(1.0/z["odds"].clip(lower=1e-9)).astype("float32")
    denom=z.groupby(["year","race_id"])["market_inv_odds"].transform("sum")
    z["market_q_norm"]=(z["market_inv_odds"]/denom.clip(lower=1e-12)).astype("float32")
    z["market_log_q"]=np.log(z["market_q_norm"].clip(lower=1e-12)).astype("float32")
    ordered=z.sort_values(
        ["year","race_id","market_q_norm","trio_rank_sum","trio_numbers"],
        ascending=[True,True,False,True,True]
    )
    ranks=ordered.groupby(["year","race_id"],sort=False).cumcount()+1
    z.loc[ordered.index,"market_rank"]=ranks.to_numpy()
    z["market_rank"]=pd.to_numeric(z["market_rank"],downcast="integer")
    return z

def feature_columns(df):
    excluded=set(IDENTITY)|{"market_inv_odds"}
    cols=[c for c in df.columns if c not in excluded]
    bad=[
        c for c in cols
        if any(tok in c.lower() for tok in FORBIDDEN_MODEL_TOKENS)
        and c not in {"market_q_norm","market_log_q","market_rank"}
    ]
    if bad:
        raise SystemExit(f"forbidden model features={sorted(bad)}")
    return cols

def encode(train,test,cols):
    xtr=train[cols].copy()
    xte=test[cols].copy()
    for c in cols:
        if str(xtr[c].dtype)=="category" or str(xte[c].dtype)=="category" or xtr[c].dtype=="object" or xte[c].dtype=="object":
            tr=xtr[c].astype("string").fillna("__NA__")
            te=xte[c].astype("string").fillna("__NA__")
            values=sorted(tr.unique())
            mp={v:i for i,v in enumerate(values)}
            xtr[c]=tr.map(mp).fillna(-1).astype("int16")
            xte[c]=te.map(mp).fillna(-1).astype("int16")
        else:
            xtr[c]=pd.to_numeric(xtr[c],errors="coerce").fillna(0.0).astype("float32")
            xte[c]=pd.to_numeric(xte[c],errors="coerce").fillna(0.0).astype("float32")
    return xtr,xte

def train_rank_predict(train,test,cols,seed):
    good=train.groupby(["year","race_id"])["hit"].transform("sum")>0
    tr=train.loc[good].copy()
    tr=tr.sort_values(["year","race_id","trio_numbers"]).reset_index(drop=True)
    te=test.copy().sort_values(["year","race_id","trio_numbers"])
    te["_orig_index"]=te.index
    te=te.reset_index(drop=True)

    xtr,xte=encode(tr,te,cols)
    y=tr["hit"].astype(int).to_numpy()
    group=tr.groupby(["year","race_id"],sort=False).size().tolist()
    model=lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=240,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(xtr,y,group=group)
    pred=np.asarray(model.predict(xte),dtype=float)
    s=pd.Series(pred,index=te["_orig_index"].astype(int).to_numpy())
    out=s.reindex(test.index).to_numpy()
    imp=pd.DataFrame({"feature":cols,"gain":model.feature_importances_})
    return out,imp

def add_ranks(df):
    z=df.copy()
    specs=[
        ("l17_product_rank","trio_prob_product"),
        ("l17_sum_rank","trio_prob_sum"),
        ("model_rank","market_aware_score"),
    ]
    for rank_col,score_col in specs:
        ordered=z.sort_values(
            ["year","race_id",score_col,"trio_rank_sum","trio_numbers"],
            ascending=[True,True,False,True,True]
        )
        r=ordered.groupby(["year","race_id"],sort=False).cumcount()+1
        z.loc[ordered.index,rank_col]=r.to_numpy()
        z[rank_col]=pd.to_numeric(z[rank_col],downcast="integer")
    z["rank_upgrade"]=(z["market_rank"]-z["model_rank"]).astype("int32")
    z["l17_product_upgrade_vs_market"]=(z["market_rank"]-z["l17_product_rank"]).astype("int32")
    return z

def ranking_rows(df,year):
    n=df["race_id"].nunique()
    rows=[]
    for label,col in [
        ("MARKET","market_rank"),
        ("L17_PRODUCT","l17_product_rank"),
        ("L17_SUM","l17_sum_rank"),
        ("MARKET_AWARE_MODEL","model_rank"),
    ]:
        for k in TOP_KS:
            hits=df[(df[col]<=k)&(df["hit"]==1)]["race_id"].nunique()
            rows.append({
                "year":year,
                "ranking":label,
                "top_k":k,
                "hit_races":int(hits),
                "source_races":int(n),
                "coverage_pct":100.0*hits/n if n else 0.0,
            })
    return rows

def rescue_rows(df,year):
    rows=[]
    n=df["race_id"].nunique()
    race_hits={}
    for k in TOP_KS:
        market=set(df[(df["market_rank"]<=k)&(df["hit"]==1)]["race_id"].astype(str))
        model=set(df[(df["model_rank"]<=k)&(df["hit"]==1)]["race_id"].astype(str))
        both=len(market&model)
        model_only=len(model-market)
        market_only=len(market-model)
        neither=n-len(market|model)
        rows.append({
            "year":year,
            "top_k":k,
            "source_races":int(n),
            "both_hit":both,
            "model_only_hit":model_only,
            "market_only_hit":market_only,
            "neither_hit":neither,
            "net_model_minus_market_hit_races":model_only-market_only,
        })
        race_hits[k]=(market,model)
    return rows

def winner_rows(df):
    keep=[
        "year","race_id","race_date","trio_horse_ids","trio_numbers","odds",
        "market_q_norm","market_rank","l17_product_rank","l17_sum_rank",
        "market_aware_score","model_rank","rank_upgrade",
        "a_consensus_rank","b_consensus_rank","c_consensus_rank",
        "trio_rank_sum","trio_worst_rank","trio_rank_spread",
        "trio_prob_sum","trio_prob_product",
        "trio_top3_count","trio_top6_count","trio_outside_top6_count",
        "experts_all_three_top6_count","experts_all_three_top10_count",
        "experts_rank_sum_std","experts_worst_rank_std"
    ]
    return df.loc[df["hit"]==1,keep].copy()

def upgrade_bin_rows(winners):
    bins=[-10**9,-20,-10,0,1,5,10,20,40,10**9]
    labels=["<=-21","-20..-11","-10..-1","0","1..4","5..9","10..19","20..39","40+"]
    z=winners.copy()
    z["upgrade_band"]=pd.cut(z["rank_upgrade"],bins=bins,labels=labels,right=False)
    out=z.groupby(["year","upgrade_band"],observed=True).agg(
        winning_tickets=("race_id","size"),
        races=("race_id","nunique"),
        median_market_rank=("market_rank","median"),
        median_model_rank=("model_rank","median"),
        median_odds=("odds","median"),
    ).reset_index()
    return out

def shape_rows(winners):
    z=winners.copy()
    z["shape"]=(
        "T3="+z["trio_top3_count"].astype(str)+
        "|T6="+z["trio_top6_count"].astype(str)+
        "|OUT6="+z["trio_outside_top6_count"].astype(str)
    )
    return z.groupby(["year","shape"],as_index=False).agg(
        winning_tickets=("race_id","size"),
        races=("race_id","nunique"),
        median_market_rank=("market_rank","median"),
        median_model_rank=("model_rank","median"),
        mean_rank_upgrade=("rank_upgrade","mean"),
    ).sort_values(["year","winning_tickets"],ascending=[True,False])

def write_csv(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(obj,pd.DataFrame):
        obj.to_csv(path,index=False)
    else:
        pd.DataFrame(obj).to_csv(path,index=False)

def main():
    a=parse_args()
    if 2026 in YEARS:
        raise SystemExit("2026 sealed")

    l17_paths={y:getattr(a,f"l17_{y}") for y in YEARS}
    l17={y:load_l17(l17_paths[y],y) for y in YEARS}
    work=Path(a.work_dir); work.mkdir(parents=True,exist_ok=True)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    build_rows=[]
    cache={}
    for y in YEARS:
        frame,meta=build_year_frame(y,l17[y],a.backfill_root)
        frame=add_market_features(frame)
        p=work/f"trio-{y}.pkl.gz"
        frame.to_pickle(p,compression="gzip")
        cache[y]=p
        build_rows.append(meta)
        del frame

    fold_rows=[]
    ranking=[]
    rescue=[]
    importances=[]
    winners=[]
    for y in TEST_YEARS:
        train_years=[2021] if y==2022 else [t for t in (2022,2023,2024) if t<y]
        train=pd.concat([pd.read_pickle(cache[t],compression="gzip") for t in train_years],ignore_index=True)
        test=pd.read_pickle(cache[y],compression="gzip").reset_index(drop=True)
        cols=feature_columns(train)
        score,imp=train_rank_predict(train,test,cols,94000+y)
        test["market_aware_score"]=score
        test=add_ranks(test)

        ranking.extend(ranking_rows(test,y))
        rescue.extend(rescue_rows(test,y))
        w=winner_rows(test)
        winners.append(w)
        imp["test_year"]=y
        importances.append(imp)

        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(map(str,train_years)),
            "train_trios":len(train),
            "test_trios":len(test),
            "source_races":int(test["race_id"].nunique()),
            "positive_trios":int(test["hit"].sum()),
            "feature_count":len(cols),
            "mean_abs_rank_upgrade_all_tickets":float(test["rank_upgrade"].abs().mean()),
            "mean_rank_upgrade_winning_tickets":float(test.loc[test["hit"]==1,"rank_upgrade"].mean()),
            "median_rank_upgrade_winning_tickets":float(test.loc[test["hit"]==1,"rank_upgrade"].median()),
        })
        del train,test

    winners_df=pd.concat(winners,ignore_index=True)
    ranking_df=pd.DataFrame(ranking)
    rescue_df=pd.DataFrame(rescue)
    fold_df=pd.DataFrame(fold_rows)
    build_df=pd.DataFrame(build_rows)
    imp_df=pd.concat(importances,ignore_index=True)
    imp_summary=imp_df.groupby("feature",as_index=False)["gain"].sum().sort_values("gain",ascending=False)

    write_csv(out/"build-coverage.csv",build_df)
    write_csv(out/"folds.csv",fold_df)
    write_csv(out/"ranking-comparison.csv",ranking_df)
    write_csv(out/"market-vs-model-rescue.csv",rescue_df)
    write_csv(out/"winning-ticket-ranks.csv",winners_df)
    write_csv(out/"winning-rank-upgrade-bands.csv",upgrade_bin_rows(winners_df))
    write_csv(out/"winning-l17-shapes.csv",shape_rows(winners_df))
    write_csv(out/"feature-importance.csv",imp_summary)

    pooled=[]
    for label in ["MARKET","L17_PRODUCT","L17_SUM","MARKET_AWARE_MODEL"]:
        for k in TOP_KS:
            g=ranking_df[(ranking_df["ranking"]==label)&(ranking_df["top_k"]==k)]
            hits=int(g["hit_races"].sum()); races=int(g["source_races"].sum())
            pooled.append({
                "ranking":label,"top_k":k,"hit_races":hits,"source_races":races,
                "coverage_pct":100.0*hits/races if races else 0.0
            })
    pooled_df=pd.DataFrame(pooled)
    write_csv(out/"ranking-comparison-pooled.csv",pooled_df)

    summary={
        "contract":"L2_TRIO_MARKET_GAP_V0_RESULT",
        "bet_type":"TRIO",
        "architecture":"all_trio_combinations_market_aware_lambdarank_diagnostic_only",
        "all_runners_considered":True,
        "top6_prefilter":False,
        "law_search_enabled":False,
        "ticket_selection_enabled":False,
        "raw_ev_multiplication":False,
        "market_probability":"normalized inverse final TRIO odds within each complete-priced race",
        "value_signal_diagnostic":"market_rank - model_rank",
        "rankings_compared":["MARKET","L17_PRODUCT","L17_SUM","MARKET_AWARE_MODEL"],
        "top_k":[1,3,5,10,20],
        "folds":fold_rows,
        "pooled_ranking":pooled,
        "winning_ticket_rank_upgrade":{
            "mean":float(winners_df["rank_upgrade"].mean()),
            "median":float(winners_df["rank_upgrade"].median()),
            "pct_positive":100.0*float((winners_df["rank_upgrade"]>0).mean()),
            "pct_ge_5":100.0*float((winners_df["rank_upgrade"]>=5).mean()),
            "pct_ge_10":100.0*float((winners_df["rank_upgrade"]>=10).mean()),
            "pct_ge_20":100.0*float((winners_df["rank_upgrade"]>=20).mean()),
        },
        "dead_heat_policy":"official payout-positive TRIO tickets are positives; a dead-heat race may have multiple positives",
        "incomplete_market_policy":"skip entire race if any TRIO combination lacks a final market price",
        "roi_used_for_selection":False,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 TRIO Market Gap V0\n\n"
        "Diagnostic stage only. Generates every priced three-horse TRIO combination from full-field L1.7, "
        "builds three-horse consensus/expert structure features, normalizes inverse final TRIO odds into a within-race market distribution, "
        "trains a walk-forward LightGBM LambdaRank model, and compares MARKET vs L1.7-product vs L1.7-sum vs market-aware model ranks. "
        "No LAW discovery, no ticket selection, no stake allocation, and no 2026 data. "
        "The primary outputs are Top1/3/5/10/20 winner capture and model-only versus market-only rescue counts.\n",
        encoding="utf-8"
    )
    print("L2_TRIO_MARKET_GAP_V0_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
