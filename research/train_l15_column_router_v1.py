#!/usr/bin/env python3
import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

FEATURE_CONTRACT="L15_COLUMN_CANDIDATE_FEATURES_V1"
LABEL_CONTRACT="L15_COLUMN_CANDIDATE_LABELS_V1"
COLUMNS=("COL1","COL2","COL3")


def parse_args():
    p=argparse.ArgumentParser(description="Train fixed-hyperparameter L1.5 column routers and evaluate expert routing at fixed TopN cost.")
    p.add_argument("--config",required=True)
    p.add_argument("--train-features",required=True)
    p.add_argument("--train-labels",required=True)
    p.add_argument("--test-features",required=True)
    p.add_argument("--test-labels",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--summary-out",required=True)
    return p.parse_args()


def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")


def read_rows(path,contract):
    rows={}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            row=json.loads(line)
            if row.get("contract")!=contract:
                raise ValueError(f"unexpected contract in {path}")
            cid=str(row["candidate_id"])
            if cid in rows:
                raise ValueError(f"duplicate candidate_id {cid}")
            rows[cid]=row
    return rows


def num(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else np.nan
    except (TypeError,ValueError):
        return np.nan


def flatten_feature(row):
    race=row.get("race") or {}
    cov=row.get("data_coverage") or {}
    con=row.get("consensus") or {}
    exp=row.get("expert_summary") or {}
    cand=row.get("candidate_summary") or {}
    fam=(cov.get("family_horse_coverage") or {})
    rec={
        "candidate_id":row["candidate_id"],
        "race_id":str(row["race_id"]),
        "race_date":str(row.get("race_date") or "")[:10],
        "column":row["column"],
        "expert_name":row["expert_name"],
        "top_n":num(row.get("top_n")),
        "candidate_size_cost":num(row.get("candidate_size_cost")),
        "venue_code":str(race.get("venue_code") or "UNKNOWN"),
        "surface":str(race.get("surface") or "UNKNOWN"),
        "race_class":str(race.get("race_class") or "UNKNOWN"),
        "discipline":str(race.get("discipline") or "UNKNOWN"),
        "direction":str(race.get("direction") or "UNKNOWN"),
        "weather":str(race.get("weather") or "UNKNOWN"),
        "track_condition":str(race.get("track_condition") or "UNKNOWN"),
        "distance_m":num(race.get("distance_m")),
        "field_size":num(race.get("field_size")),
        "prior_starts_mean":num(cov.get("prior_starts_mean")),
        "prior_starts_zero_rate":num(cov.get("prior_starts_zero_rate")),
        "prior_starts_le1_rate":num(cov.get("prior_starts_le1_rate")),
        "prior_starts_le2_rate":num(cov.get("prior_starts_le2_rate")),
        "recent_window_starts_mean":num(cov.get("recent_window_starts_mean")),
        "history_known_rate":num(cov.get("history_known_rate")),
        "expert_count":num(con.get("expert_count")),
        "top1_unique_horses":num(con.get("top1_unique_horses")),
        "top1_max_vote":num(con.get("top1_max_vote")),
        "top1_max_vote_share":num(con.get("top1_max_vote_share")),
        "top1_full_agreement":1.0 if con.get("top1_full_agreement") is True else 0.0,
        "top3_pairwise_jaccard_mean":num(con.get("top3_pairwise_jaccard_mean")),
        "top6_pairwise_jaccard_mean":num(con.get("top6_pairwise_jaccard_mean")),
        "pairwise_rank_abs_diff_mean":num(con.get("pairwise_rank_abs_diff_mean")),
        "pairwise_rank_abs_diff_max":num(con.get("pairwise_rank_abs_diff_max")),
        "horse_rank_std_mean":num(con.get("horse_rank_std_mean")),
        "horse_rank_std_max":num(con.get("horse_rank_std_max")),
        "horse_probability_std_mean":num(con.get("horse_probability_std_mean")),
        "horse_probability_std_max":num(con.get("horse_probability_std_max")),
        "top1_probability":num(exp.get("top1_probability")),
        "top2_probability":num(exp.get("top2_probability")),
        "top3_probability":num(exp.get("top3_probability")),
        "top1_top2_gap":num(exp.get("top1_top2_gap")),
        "top1_top3_gap":num(exp.get("top1_top3_gap")),
        "top3_probability_mass":num(exp.get("top3_probability_mass")),
        "top6_probability_mass":num(exp.get("top6_probability_mass")),
        "normalized_entropy":num(exp.get("normalized_entropy")),
        "candidate_probability_mass":num(cand.get("probability_mass")),
        "candidate_probability_mean":num(cand.get("probability_mean")),
        "candidate_last_included_probability":num(cand.get("last_included_probability")),
        "candidate_same_top_n_jaccard_mean":num(cand.get("same_top_n_jaccard_mean")),
    }
    for family,value in sorted(fam.items()):
        rec["coverage_"+str(family).lower()]=num(value)
    return rec


def joined_frame(feature_path,label_path):
    features=read_rows(feature_path,FEATURE_CONTRACT)
    labels=read_rows(label_path,LABEL_CONTRACT)
    if set(features)!=set(labels):
        raise ValueError("feature/label candidate coverage differs")
    rows=[]
    for cid,f in features.items():
        r=flatten_feature(f)
        r["slot_hit"]=int(bool(labels[cid]["slot_hit"]))
        rows.append(r)
    return pd.DataFrame(rows)


def year_of(date):
    try: return int(str(date)[:4])
    except Exception: return None


def encode(train,test):
    drop={"candidate_id","race_id","race_date","column","slot_hit"}
    cats=["expert_name","venue_code","surface","race_class","discipline","direction","weather","track_condition"]
    tr=train.drop(columns=[x for x in drop if x in train],errors="ignore").copy()
    te=test.drop(columns=[x for x in drop if x in test],errors="ignore").copy()
    both=pd.concat([tr,te],axis=0,ignore_index=True)
    both=pd.get_dummies(both,columns=[c for c in cats if c in both],dummy_na=False,dtype=float)
    tr_enc=both.iloc[:len(tr)].copy()
    te_enc=both.iloc[len(tr):].copy()
    tr_enc=tr_enc.replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    te_enc=te_enc.replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    return tr_enc,te_enc


def safe_auc(y,p):
    return roc_auc_score(y,p) if len(set(map(int,y)))>1 else None


def routing_at_fixed_cost(test):
    out=defaultdict(dict)
    for column in COLUMNS:
        cdf=test[test["column"]==column]
        for n in sorted(cdf["top_n"].dropna().astype(int).unique()):
            sub=cdf[cdf["top_n"].astype(int)==n]
            races=0; hits=0
            choices=defaultdict(int)
            for rid,g in sub.groupby("race_id"):
                chosen=g.sort_values(["predicted_slot_hit_probability","expert_name"],ascending=[False,True]).iloc[0]
                races+=1
                hits+=int(chosen["slot_hit"])
                choices[str(chosen["expert_name"])]+=1
            out[column][str(n)]={
                "races":races,
                "hits":hits,
                "hit_rate":hits/races if races else None,
                "choice_counts":dict(choices),
                "candidate_size":n,
            }
    return out


def fixed_expert_at_cost(test):
    out=defaultdict(lambda:defaultdict(dict))
    for (column,expert,n),g in test.groupby(["column","expert_name","top_n"]):
        n=int(n)
        out[column][expert][str(n)]={
            "races":int(g["race_id"].nunique()),
            "hits":int(g["slot_hit"].sum()),
            "hit_rate":float(g["slot_hit"].mean()),
            "candidate_size":n,
        }
    return out


def oracle_at_fixed_cost(test):
    out=defaultdict(dict)
    for column in COLUMNS:
        cdf=test[test["column"]==column]
        for n in sorted(cdf["top_n"].dropna().astype(int).unique()):
            sub=cdf[cdf["top_n"].astype(int)==n]
            vals=sub.groupby("race_id")["slot_hit"].max()
            out[column][str(n)]={
                "races":int(len(vals)),
                "hits":int(vals.sum()),
                "hit_rate":float(vals.mean()) if len(vals) else None,
                "candidate_size":n,
                "hindsight_only":True,
            }
    return out


def main():
    a=parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8"))
    if cfg.get("contract")!="L15_COLUMN_ROUTER_EXPERIMENT_V1":
        raise ValueError("unexpected config")
    locked=set(int(x) for x in cfg.get("locked_years",[]))

    train=joined_frame(a.train_features,a.train_labels)
    test=joined_frame(a.test_features,a.test_labels)
    train_years={year_of(x) for x in train["race_date"]}; train_years.discard(None)
    test_years={year_of(x) for x in test["race_date"]}; test_years.discard(None)
    if locked.intersection(train_years|test_years):
        raise ValueError(f"locked year present in L1.5 training/evaluation: {sorted(locked.intersection(train_years|test_years))}")
    if train["race_date"].max() >= test["race_date"].min():
        raise ValueError("train/test chronology violation")

    out_dir=Path(a.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    predictions=[]
    model_metrics={}

    for column in COLUMNS:
        tr=train[train["column"]==column].copy()
        te=test[test["column"]==column].copy()
        if tr.empty or te.empty:
            raise ValueError(f"missing rows for {column}")
        Xtr,Xte=encode(tr,te)
        ytr=tr["slot_hit"].astype(int)
        yte=te["slot_hit"].astype(int)
        model=lgb.LGBMClassifier(
            objective="binary",
            n_estimators=250,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=50,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            random_state=1945,
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(Xtr,ytr)
        p=model.predict_proba(Xte)[:,1]
        pred=te[["candidate_id","race_id","race_date","column","expert_name","top_n","slot_hit"]].copy()
        pred["predicted_slot_hit_probability"]=p
        predictions.append(pred)
        model.booster_.save_model(str(out_dir/f"{column.lower()}-router.txt"))
        model_metrics[column]={
            "train_rows":int(len(tr)),
            "test_rows":int(len(te)),
            "test_log_loss":float(log_loss(yte,p,labels=[0,1])),
            "test_auc":float(safe_auc(yte,p)) if safe_auc(yte,p) is not None else None,
            "feature_count":int(Xtr.shape[1]),
        }

    pred=pd.concat(predictions,ignore_index=True)
    pred_path=out_dir/"predictions.jsonl.gz"
    with gzip.open(pred_path,"wt",encoding="utf-8") as fh:
        for row in pred.to_dict(orient="records"):
            fh.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")

    summary={
        "contract":"L15_COLUMN_ROUTER_MODEL_V1",
        "train_years":sorted(train_years),
        "test_years":sorted(test_years),
        "models":model_metrics,
        "routing_at_fixed_top_n":routing_at_fixed_cost(pred),
        "fixed_expert_at_fixed_top_n":fixed_expert_at_cost(pred),
        "oracle_any_expert_at_fixed_top_n":oracle_at_fixed_cost(pred),
        "predictions":str(pred_path),
        "notes":[
            "No odds are used.",
            "Router comparison is made at fixed TopN cost so larger candidate sets cannot win merely by being larger.",
            "Hyperparameters are fixed; no search/tuning is performed in this script.",
            "2026 is rejected by the locked-year guard."
        ]
    }
    sp=Path(a.summary_out); sp.parent.mkdir(parents=True,exist_ok=True)
    sp.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L15_COLUMN_ROUTER_MODEL_V1_OK")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
