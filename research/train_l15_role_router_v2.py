#!/usr/bin/env python3
import argparse,gc,gzip,itertools,json,math,resource
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,roc_auc_score

FEATURE_CONTRACT="L15_ROLE_CANDIDATE_FEATURES_V2"
LABEL_CONTRACT="L15_ROLE_CANDIDATE_LABELS_V2"
ROLES=("ANCHOR","MAINLINE","COVER")

def json_safe(v):
    if isinstance(v,dict): return {str(k):json_safe(x) for k,x in v.items()}
    if isinstance(v,list): return [json_safe(x) for x in v]
    if isinstance(v,tuple): return [json_safe(x) for x in v]
    if isinstance(v,np.integer): return int(v)
    if isinstance(v,np.floating): return float(v)
    if isinstance(v,np.bool_): return bool(v)
    return v

def parse_args():
    p=argparse.ArgumentParser()
    for name in ["config","train-features","train-labels","test-features","test-labels","out-dir","summary-out"]:
        p.add_argument("--"+name,required=True)
    return p.parse_args()

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def read_rows(path,contract):
    out={}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip(): continue
            r=json.loads(line)
            if r.get("contract")!=contract: raise ValueError(f"unexpected contract in {path}")
            cid=str(r["candidate_id"])
            if cid in out: raise ValueError(f"duplicate candidate_id {cid}")
            out[cid]=r
    return out

def num(v):
    try:
        x=float(v); return x if math.isfinite(x) else np.nan
    except (TypeError,ValueError): return np.nan

def flatten(r):
    race=r.get("race") or {}; cov=r.get("data_coverage") or {}
    con=r.get("consensus") or {}; exp=r.get("expert_summary") or {}
    cand=r.get("candidate_summary") or {}; fam=cov.get("family_horse_coverage") or {}
    x={
      "candidate_id":r["candidate_id"],"race_id":str(r["race_id"]),
      "race_date":str(r.get("race_date") or "")[:10],"role":r["role"],
      "expert_name":r["expert_name"],"top_n":num(r.get("top_n")),
      "candidate_size_cost":num(r.get("candidate_size_cost")),
      "venue_code":str(race.get("venue_code") or "UNKNOWN"),
      "surface":str(race.get("surface") or "UNKNOWN"),
      "race_class":str(race.get("race_class") or "UNKNOWN"),
      "discipline":str(race.get("discipline") or "UNKNOWN"),
      "direction":str(race.get("direction") or "UNKNOWN"),
      "weather":str(race.get("weather") or "UNKNOWN"),
      "track_condition":str(race.get("track_condition") or "UNKNOWN"),
      "distance_m":num(race.get("distance_m")),"field_size":num(race.get("field_size")),
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
    for k,v in sorted(fam.items()): x["coverage_"+str(k).lower()]=num(v)
    return x

def frame_role(fp,lp,role):
    rows=[]; seen=set()
    with open_text(fp) as ff, open_text(lp) as lf:
        fit=(line for line in ff if line.strip())
        lit=(line for line in lf if line.strip())
        for i,(fline,lline) in enumerate(itertools.zip_longest(fit,lit),1):
            if fline is None or lline is None:
                raise ValueError(f"feature/label row count differs near row {i}")
            f=json.loads(fline); l=json.loads(lline)
            if f.get("contract")!=FEATURE_CONTRACT:
                raise ValueError(f"unexpected contract in {fp}")
            if l.get("contract")!=LABEL_CONTRACT:
                raise ValueError(f"unexpected contract in {lp}")
            fcid=str(f["candidate_id"]); lcid=str(l["candidate_id"])
            if fcid!=lcid:
                raise ValueError(f"feature/label order differs near row {i}")
            if str(f.get("role"))!=str(l.get("role")):
                raise ValueError(f"feature/label role differs for {fcid}")
            if str(f.get("role"))!=role:
                continue
            if fcid in seen:
                raise ValueError(f"duplicate candidate_id {fcid}")
            seen.add(fcid)
            x=flatten(f); x["role_hit"]=int(bool(l["role_hit"])); rows.append(x)
    return pd.DataFrame.from_records(rows)

def peak_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

def year_of(d):
    try:return int(str(d)[:4])
    except:return None

def encode(train,test):
    drop={"candidate_id","race_id","race_date","role","role_hit"}
    cats=["expert_name","venue_code","surface","race_class","discipline","direction","weather","track_condition"]
    tr=train.drop(columns=[x for x in drop if x in train],errors="ignore").copy()
    te=test.drop(columns=[x for x in drop if x in test],errors="ignore").copy()
    both=pd.concat([tr,te],ignore_index=True)
    both=pd.get_dummies(both,columns=[c for c in cats if c in both],dummy_na=False,dtype=float)
    tr=both.iloc[:len(tr)].replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    te=both.iloc[len(tr):].replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    return tr,te

def routing(test):
    out=defaultdict(dict)
    for role in ROLES:
        rdf=test[test.role==role]
        for n in sorted(rdf.top_n.astype(int).unique()):
            sub=rdf[rdf.top_n.astype(int)==n]; hits=0; races=0; choices=defaultdict(int)
            for rid,g in sub.groupby("race_id"):
                z=g.sort_values(["predicted_role_hit_probability","expert_name"],ascending=[False,True]).iloc[0]
                races+=1; hits+=int(z.role_hit); choices[str(z.expert_name)]+=1
            out[role][str(n)]={"races":races,"hits":hits,"hit_rate":hits/races if races else None,
                               "choice_counts":dict(choices),"candidate_size":n}
    return out

def fixed(test):
    out=defaultdict(lambda:defaultdict(dict))
    for (role,expert,n),g in test.groupby(["role","expert_name","top_n"]):
        out[role][expert][str(int(n))]={"races":int(g.race_id.nunique()),"hits":int(g.role_hit.sum()),
            "hit_rate":float(g.role_hit.mean()),"candidate_size":int(n)}
    return out

def oracle(test):
    out=defaultdict(dict)
    for role in ROLES:
        rdf=test[test.role==role]
        for n in sorted(rdf.top_n.astype(int).unique()):
            vals=rdf[rdf.top_n.astype(int)==n].groupby("race_id").role_hit.max()
            out[role][str(n)]={"races":int(len(vals)),"hits":int(vals.sum()),
                "hit_rate":float(vals.mean()),"candidate_size":n,"hindsight_only":True}
    return out

def main():
    a=parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8"))
    if cfg.get("contract")!="L15_ROLE_ROUTER_EXPERIMENT_V2": raise ValueError("bad config")
    locked=set(map(int,cfg.get("locked_years",[])))
    outdir=Path(a.out_dir); outdir.mkdir(parents=True,exist_ok=True)
    preds=[]; metrics={}; ty=vy=None
    for role in ROLES:
        tr=frame_role(a.train_features,a.train_labels,role)
        te=frame_role(a.test_features,a.test_labels,role)
        if tr.empty or te.empty:
            raise ValueError(f"empty role frame: {role}")
        rty={year_of(x) for x in tr.race_date}; rty.discard(None)
        rvy={year_of(x) for x in te.race_date}; rvy.discard(None)
        if locked&(rty|rvy): raise ValueError("locked year present")
        if tr.race_date.max()>=te.race_date.min(): raise ValueError("chronology violation")
        if ty is None:
            ty,vy=rty,rvy
        elif rty!=ty or rvy!=vy:
            raise ValueError(f"role year coverage differs: {role}")
        print(f"L15_ROLE_MEMORY role={role} stage=loaded peak_rss_mib={peak_rss_mib():.1f}",flush=True)

        Xtr,Xte=encode(tr,te); ytr=tr.role_hit.astype(int); yte=te.role_hit.astype(int)
        print(f"L15_ROLE_MEMORY role={role} stage=encoded peak_rss_mib={peak_rss_mib():.1f}",flush=True)
        model=lgb.LGBMClassifier(objective="binary",n_estimators=250,learning_rate=0.03,num_leaves=15,
            min_child_samples=50,subsample=1.0,colsample_bytree=1.0,reg_lambda=1.0,
            random_state=1945,n_jobs=2,verbosity=-1)
        model.fit(Xtr,ytr); p=model.predict_proba(Xte)[:,1]
        pr=te[["candidate_id","race_id","race_date","role","expert_name","top_n","role_hit"]].copy()
        pr["predicted_role_hit_probability"]=p; preds.append(pr)
        model.booster_.save_model(str(outdir/f"{role.lower()}-router.txt"))
        auc=roc_auc_score(yte,p) if len(set(map(int,yte)))>1 else None
        metrics[role]={"train_rows":int(len(tr)),"test_rows":int(len(te)),
            "test_log_loss":float(log_loss(yte,p,labels=[0,1])),
            "test_auc":float(auc) if auc is not None else None,"feature_count":int(Xtr.shape[1])}
        print(f"L15_ROLE_MEMORY role={role} stage=trained peak_rss_mib={peak_rss_mib():.1f}",flush=True)
        del tr,te,Xtr,Xte,ytr,yte,model,p,pr
        gc.collect()

    pred=pd.concat(preds,ignore_index=True)
    pp=outdir/"predictions.jsonl.gz"
    with gzip.open(pp,"wt",encoding="utf-8") as fh:
        for row in pred.to_dict(orient="records"):
            fh.write(json.dumps(json_safe(row),ensure_ascii=False,separators=(",",":"))+"\n")
    summary=json_safe({"contract":"L15_ROLE_ROUTER_MODEL_V2","train_years":sorted(ty),"test_years":sorted(vy),
        "models":metrics,"routing_at_fixed_top_n":routing(pred),
        "fixed_expert_at_fixed_top_n":fixed(pred),"oracle_any_expert_at_fixed_top_n":oracle(pred),
        "predictions":str(pp),
        "notes":["No odds are used.","Same model/hyperparameters as V1; only role labels change.",
                 "Comparisons are made at fixed TopN cost.","2026 is locked."]})
    Path(a.summary_out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.summary_out).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L15_ROLE_ROUTER_MODEL_V2_OK")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__": main()
