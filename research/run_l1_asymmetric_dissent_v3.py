#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,brier_score_loss

from run_l1_market_divergence_audit_v1 import load_dataset,market_and_finish
from run_l1_dissent_gate_v1 import (
    model_params,chronological_fit_cal_split,fit_calibrator,apply_calibrator
)

FOLDS=((2023,(2022,)),(2024,(2022,2023)),(2025,(2022,2023,2024)))
DIRECTIONS=("L1_UPGRADE","L1_DOWNGRADE")

def parse_args():
    p=argparse.ArgumentParser(description="Frozen asymmetric STRONG_DISSENT V3.")
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def add_features(z):
    z=z.copy()
    z["inv_final_win_odds"]=1.0/z["final_win_odds"].astype(float)
    sums=z.groupby("race_id")["inv_final_win_odds"].transform("sum")
    z["market_win_probability"]=z["inv_final_win_odds"]/sums
    z["market_rank_pct"]=z["market_rank"].astype(float)/z["field_size"].astype(float)
    z["consensus_rank_pct"]=z["consensus_rank"].astype(float)/z["field_size"].astype(float)
    z["log_market_win_probability"]=np.log(np.clip(z["market_win_probability"].to_numpy(dtype=float),1e-12,1.0))
    z["log_final_win_odds"]=np.log(np.clip(z["final_win_odds"].to_numpy(dtype=float),1e-12,None))
    z["rank_gap"]=z["market_rank"].astype(float)-z["consensus_rank"].astype(float)
    z["abs_rank_gap"]=z["rank_gap"].abs()
    z["rank_gap_pct"]=z["rank_gap"]/z["field_size"].astype(float)
    z["abs_rank_gap_pct"]=z["rank_gap_pct"].abs()
    z["rank_product_pct"]=z["market_rank_pct"]*z["consensus_rank_pct"]
    z["direction"]=np.where(z["rank_gap"]>=2,"L1_UPGRADE",np.where(z["rank_gap"]<=-2,"L1_DOWNGRADE","NEAR"))
    return z

def fit_model(train,features,seed):
    fit,cal=chronological_fit_cal_split(train)
    wfit=1.0/fit["field_size"].astype(float).to_numpy()
    wcal=1.0/cal["field_size"].astype(float).to_numpy()

    m0=lgb.LGBMClassifier(**model_params(seed))
    m0.fit(fit[features],fit["target_top3"].astype(int),sample_weight=wfit)
    raw_cal=np.asarray(m0.predict_proba(cal[features])[:,1],dtype=float)
    calibrator=fit_calibrator(raw_cal,cal["target_top3"].astype(int),wcal)
    pcal=apply_calibrator(calibrator,raw_cal)

    mall=lgb.LGBMClassifier(**model_params(seed))
    wall=1.0/train["field_size"].astype(float).to_numpy()
    mall.fit(train[features],train["target_top3"].astype(int),sample_weight=wall)
    return (mall,calibrator),cal,pcal

def predict(pair,df,features):
    m,c=pair
    raw=np.asarray(m.predict_proba(df[features])[:,1],dtype=float)
    return apply_calibrator(c,raw)

def high_threshold(cal,score,direction):
    q=cal.copy().reset_index(drop=True)
    q["score"]=np.asarray(score,dtype=float)
    q=q[(q["direction"]==direction) & (q["rank_gap"].abs()>=2) & (q["score"]>0)]
    vals=q["score"].to_numpy(dtype=float)
    if len(vals)<100:
        return float("inf")
    return float(np.quantile(vals,2/3))

def peer_baseline(test,all_year):
    peers=all_year.groupby("market_rank",dropna=False)["target_top3"].agg(["mean","size"]).reset_index()
    peers=peers.rename(columns={"mean":"market_peer_top3_rate","size":"market_peer_horses"})
    return test.merge(peers,on="market_rank",how="left",validate="many_to_one")

def effect(sub):
    if sub.empty: return None
    sign=np.sign(sub["rank_gap"].to_numpy(dtype=float))
    residual=sub["target_top3"].to_numpy(dtype=float)-sub["market_peer_top3_rate"].to_numpy(dtype=float)
    return 100*float(np.mean(sign*residual))

def summarize(year,direction,tag,sub,all_direction,threshold):
    return {
        "test_year":year,
        "direction":direction,
        "tag":tag,
        "horses":len(sub),
        "races":sub["race_id"].nunique() if len(sub) else 0,
        "coverage_within_direction_pct":100*len(sub)/len(all_direction) if len(all_direction) else None,
        "threshold":threshold,
        "actual_top3_pct":100*float(sub["target_top3"].mean()) if len(sub) else None,
        "market_peer_top3_pct":100*float(sub["market_peer_top3_rate"].mean()) if len(sub) else None,
        "direction_correct_effect_pp":effect(sub),
        "mean_score":float(sub["asymmetric_dissent_score"].mean()) if len(sub) and "asymmetric_dissent_score" in sub else None,
    }

def write_csv(path,rows):
    path=Path(path)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def main():
    a=parse_args()
    manifest,df=load_dataset(a.dataset_dir)
    if 2026 in set(df["year"]): raise SystemExit("2026 sealed")
    aux,skipped=market_and_finish(df,a.backfill_root)
    keep=df.index.intersection(aux.index)
    z=df.loc[keep].copy().join(aux.loc[keep])
    z=z[z["target_top3"].notna()].copy()
    z=add_features(z)

    numeric=["field_size","market_rank_pct","consensus_rank_pct","log_market_win_probability",
             "log_final_win_odds","rank_gap","abs_rank_gap","rank_gap_pct","abs_rank_gap_pct","rank_product_pct"]
    for c in numeric:
        z[c]=pd.to_numeric(z[c],errors="coerce").fillna(0.0)

    market_features=["field_size","market_rank_pct","log_market_win_probability","log_final_win_odds"]
    up_features=market_features+["consensus_rank_pct","rank_gap","abs_rank_gap","rank_gap_pct","rank_product_pct"]
    down_features=market_features+["rank_gap","abs_rank_gap","rank_gap_pct","abs_rank_gap_pct"]

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]; metrics=[]; thresholds=[]; scored=[]; fold_summary=[]

    for fi,(test_year,train_years) in enumerate(FOLDS,1):
        train=z[z["year"].isin(train_years)].copy()
        test=z[z["year"]==test_year].copy().reset_index(drop=True)

        market_pair,cal,pm_cal=fit_model(train,market_features,20262000+fi)
        up_pair,cal_up,pup_cal=fit_model(train,up_features,20262100+fi)
        down_pair,cal_down,pdown_cal=fit_model(train,down_features,20262200+fi)

        # chronological split is deterministic, so all calibration frames must align
        if list(cal["race_id"])!=list(cal_up["race_id"]) or list(cal["race_id"])!=list(cal_down["race_id"]):
            raise SystemExit("calibration frame drift")

        pm=predict(market_pair,test,market_features)
        pup=predict(up_pair,test,up_features)
        pdown=predict(down_pair,test,down_features)

        sign_cal=np.sign(cal["rank_gap"].to_numpy(dtype=float))
        up_cal_score=sign_cal*(pup_cal-pm_cal)
        down_cal_score=sign_cal*(pdown_cal-pm_cal)
        up_thr=high_threshold(cal,up_cal_score,"L1_UPGRADE")
        down_thr=high_threshold(cal,down_cal_score,"L1_DOWNGRADE")

        sign_test=np.sign(test["rank_gap"].to_numpy(dtype=float))
        test["score_up"]=sign_test*(pup-pm)
        test["score_down"]=sign_test*(pdown-pm)
        test["asymmetric_dissent_score"]=np.where(
            test["direction"]=="L1_UPGRADE",test["score_up"],
            np.where(test["direction"]=="L1_DOWNGRADE",test["score_down"],np.nan)
        )
        test["asymmetric_dissent_tag"]="NEAR"
        upmask=test["direction"]=="L1_UPGRADE"
        downmask=test["direction"]=="L1_DOWNGRADE"
        test.loc[upmask & (test["score_up"]<=0),"asymmetric_dissent_tag"]="CONTRA_UP"
        test.loc[upmask & (test["score_up"]>0) & (test["score_up"]<up_thr),"asymmetric_dissent_tag"]="NORMAL_UP"
        test.loc[upmask & (test["score_up"]>=up_thr),"asymmetric_dissent_tag"]="STRONG_UP"
        test.loc[downmask & (test["score_down"]<=0),"asymmetric_dissent_tag"]="CONTRA_DOWN"
        test.loc[downmask & (test["score_down"]>0) & (test["score_down"]<down_thr),"asymmetric_dissent_tag"]="NORMAL_DOWN"
        test.loc[downmask & (test["score_down"]>=down_thr),"asymmetric_dissent_tag"]="STRONG_DOWN"

        test=peer_baseline(test,z[z["year"]==test_year])
        thresholds.extend([
            {"test_year":test_year,"direction":"L1_UPGRADE","model":"RANK_CROSS","strong_threshold":up_thr},
            {"test_year":test_year,"direction":"L1_DOWNGRADE","model":"GAP_ONLY","strong_threshold":down_thr},
        ])

        y=test["target_top3"].astype(int).to_numpy()
        for name,p in (("MARKET_ONLY",pm),("RANK_CROSS_UP_MODEL",pup),("GAP_ONLY_DOWN_MODEL",pdown)):
            metrics.append({
                "test_year":test_year,"model":name,"horses":len(test),
                "log_loss":float(log_loss(y,np.clip(p,1e-12,1-1e-12))),
                "brier":float(brier_score_loss(y,p)),
            })

        for direction,strong_tag,normal_tag,contra_tag,thr in (
            ("L1_UPGRADE","STRONG_UP","NORMAL_UP","CONTRA_UP",up_thr),
            ("L1_DOWNGRADE","STRONG_DOWN","NORMAL_DOWN","CONTRA_DOWN",down_thr),
        ):
            d=test[test["direction"]==direction].copy()
            allrow=summarize(test_year,direction,"ALL_DISAGREEMENTS",d,d,thr)
            rows.append(allrow)
            for tag in (strong_tag,normal_tag,contra_tag):
                rows.append(summarize(test_year,direction,tag,d[d["asymmetric_dissent_tag"]==tag],d,thr))

            strong=d[d["asymmetric_dissent_tag"]==strong_tag]
            fold_summary.append({
                "test_year":test_year,
                "train_years":"|".join(map(str,train_years)),
                "direction":direction,
                "all_horses":len(d),
                "all_effect_pp":effect(d),
                "strong_horses":len(strong),
                "strong_races":strong["race_id"].nunique(),
                "strong_coverage_pct":100*len(strong)/len(d) if len(d) else None,
                "strong_effect_pp":effect(strong),
                "strong_minus_all_pp":effect(strong)-effect(d) if len(strong) else None,
                "strong_positive":bool((effect(strong) or 0)>0),
                "strong_n_ge_500":bool(len(strong)>=500),
                "strong_beats_all":bool(len(strong)>=500 and effect(strong)>effect(d)),
            })

        keepcols=["year","race_id","horse_id","horse_number","consensus_rank","market_rank","rank_gap","direction",
                  "final_win_odds","target_top3","market_peer_top3_rate","score_up","score_down",
                  "asymmetric_dissent_score","asymmetric_dissent_tag"]
        scored.append(test[keepcols])
        print("ASYMMETRIC_DISSENT_V3_FOLD_READY "+json.dumps({"test_year":test_year,"train_years":train_years},separators=(",",":")),flush=True)

    scored_df=pd.concat(scored,ignore_index=True)
    write_csv(out/"tag-summary.csv",rows)
    write_csv(out/"fold-summary.csv",fold_summary)
    write_csv(out/"model-metrics.csv",metrics)
    write_csv(out/"thresholds.csv",thresholds)
    scored_df.to_csv(out/"scored.csv.gz",index=False,compression="gzip")

    fs=pd.DataFrame(fold_summary)
    decision={}
    for direction in DIRECTIONS:
        q=fs[fs["direction"]==direction].copy()
        decision[direction]={
            "pass_all_three_folds":bool(len(q)==3 and q["strong_beats_all"].all() and q["strong_positive"].all() and q["strong_n_ge_500"].all()),
            "folds":q.to_dict(orient="records"),
        }
    summary={
        "contract":"L1_ASYMMETRIC_DISSENT_V3",
        "frozen_rule":{
            "UP":"RANK_CROSS only",
            "DOWN":"GAP_ONLY only",
            "STRONG":"score >= direction-specific 2/3 quantile of positive prior-calibration support scores",
            "NORMAL":"0 < score < STRONG threshold",
            "CONTRA":"score <= 0",
        },
        "folds":[{"test_year":y,"train_years":list(t)} for y,t in FOLDS],
        "new_stress_test":"2023 trained only on 2022; V2 selected methods using 2024/2025 performance, so this is the new third OOS check",
        "promotion_rule":"Per direction: STRONG >=500 horses, positive direction-correct effect, and stronger than ALL disagreements in every 2023/2024/2025 fold.",
        "decision":decision,
        "core_L1_market_free":True,
        "rerank":False,
        "paid_compute":False,
        "2026_locked":True,
        "promotion":False,
        "skipped_market_races":len(skipped),
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== FOLD SUMMARY =====")
    print((out/"fold-summary.csv").read_text(encoding="utf-8"))
    print("===== TAG SUMMARY =====")
    print((out/"tag-summary.csv").read_text(encoding="utf-8"))
    print("L1_ASYMMETRIC_DISSENT_V3_READY")

if __name__=="__main__":
    main()
