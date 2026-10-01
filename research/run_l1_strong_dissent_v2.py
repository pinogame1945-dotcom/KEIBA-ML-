#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,brier_score_loss

from run_l1_market_divergence_audit_v1 import load_dataset,attach_strict_oos_confidence,market_and_finish
from run_l1_dissent_gate_v1 import model_params,chronological_fit_cal_split,fit_calibrator,apply_calibrator

TEST_FOLDS=((2024,(2023,)),(2025,(2023,2024)))
METHODS=("GAP_ONLY","CONF_ONLY","RANK_CROSS","FULL")

def parse_args():
    p=argparse.ArgumentParser(description="Strict walk-forward benchmark for STRONG_DISSENT V2.")
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def unique(xs):
    out=[]
    for x in xs:
        if x not in out: out.append(x)
    return out

def add_market_features(z):
    z=z.copy()
    z["inv_final_win_odds"]=1.0/z["final_win_odds"].astype(float)
    sums=z.groupby("race_id")["inv_final_win_odds"].transform("sum")
    z["market_win_probability"]=z["inv_final_win_odds"]/sums
    z["market_rank_pct"]=z["market_rank"].astype(float)/z["field_size"].astype(float)
    z["log_market_win_probability"]=np.log(np.clip(z["market_win_probability"].to_numpy(dtype=float),1e-12,1.0))
    z["log_final_win_odds"]=np.log(np.clip(z["final_win_odds"].to_numpy(dtype=float),1e-12,None))
    z["rank_gap"]=z["market_rank"].astype(float)-z["consensus_rank"].astype(float)
    z["abs_rank_gap"]=z["rank_gap"].abs()
    z["rank_gap_pct"]=z["rank_gap"]/z["field_size"].astype(float)
    z["abs_rank_gap_pct"]=z["rank_gap_pct"].abs()
    z["rank_product_pct"]=z["market_rank_pct"]*z["consensus_rank_pct"].astype(float)
    z["direction"]=np.where(z["rank_gap"]>=2,"L1_UPGRADE",np.where(z["rank_gap"]<=-2,"L1_DOWNGRADE","NEAR"))
    return z

def fit_model(fit,cal,features,seed):
    wfit=1.0/fit["field_size"].astype(float).to_numpy()
    wcal=1.0/cal["field_size"].astype(float).to_numpy()
    m0=lgb.LGBMClassifier(**model_params(seed))
    m0.fit(fit[features],fit["target_top3"].astype(int),sample_weight=wfit)
    pcal0=np.asarray(m0.predict_proba(cal[features])[:,1],dtype=float)
    calibrator=fit_calibrator(pcal0,cal["target_top3"].astype(int),wcal)
    pcal=apply_calibrator(calibrator,pcal0)
    mall=lgb.LGBMClassifier(**model_params(seed))
    wall=1.0/pd.concat([fit,cal])["field_size"].astype(float).to_numpy()
    train=pd.concat([fit,cal],ignore_index=True)
    mall.fit(train[features],train["target_top3"].astype(int),sample_weight=wall)
    return (mall,calibrator),pcal

def predict(pair,df,features):
    m,c=pair
    raw=np.asarray(m.predict_proba(df[features])[:,1],dtype=float)
    return apply_calibrator(c,raw)

def high_threshold(cal,score):
    tmp=cal.copy().reset_index(drop=True)
    tmp["support_score"]=np.asarray(score,dtype=float)
    tmp=tmp[tmp["rank_gap"].abs()>=2]
    pos=tmp.loc[tmp["support_score"]>0,"support_score"].to_numpy(dtype=float)
    if len(pos)<50:
        return float("inf")
    return float(np.quantile(pos,2/3))

def peer_baseline(test,all_year):
    peers=all_year.groupby("market_rank",dropna=False)["target_top3"].agg(["mean","size"]).reset_index()
    peers=peers.rename(columns={"mean":"market_peer_top3_rate","size":"market_peer_horses"})
    return test.merge(peers,on="market_rank",how="left",validate="many_to_one")

def effect(sub):
    if sub.empty: return None
    sign=np.sign(sub["rank_gap"].to_numpy(dtype=float))
    residual=sub["target_top3"].to_numpy(dtype=float)-sub["market_peer_top3_rate"].to_numpy(dtype=float)
    return 100*float(np.mean(sign*residual))

def summarize(test_year,method,mode,direction,sub,threshold=None):
    return {
        "test_year":test_year,
        "method":method,
        "selection_mode":mode,
        "direction":direction,
        "horses":len(sub),
        "races":sub["race_id"].nunique() if len(sub) else 0,
        "coverage_within_direction_pct":None,
        "mean_support_score":float(sub[f"score_{method}"].mean()) if len(sub) else None,
        "threshold":threshold,
        "actual_top3_pct":100*float(sub["target_top3"].mean()) if len(sub) else None,
        "market_peer_top3_pct":100*float(sub["market_peer_top3_rate"].mean()) if len(sub) else None,
        "direction_correct_effect_pp":effect(sub),
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
    z=attach_strict_oos_confidence(z)
    z=add_market_features(z)

    l1_features=[]
    for c in manifest.get("feature_columns",[]):
        if c in z.columns:
            z[c]=pd.to_numeric(z[c],errors="coerce").fillna(0.0)
            l1_features.append(c)
    z["confidence_score"]=pd.to_numeric(z["confidence_score"],errors="coerce")
    numeric_extra=["field_size","market_rank_pct","log_market_win_probability","log_final_win_odds",
                   "rank_gap","abs_rank_gap","rank_gap_pct","abs_rank_gap_pct","rank_product_pct",
                   "consensus_rank_pct","confidence_score"]
    for c in numeric_extra:
        z[c]=pd.to_numeric(z[c],errors="coerce")

    market_features=["field_size","market_rank_pct","log_market_win_probability","log_final_win_odds"]
    method_features={
        "GAP_ONLY":unique(market_features+["rank_gap","abs_rank_gap","rank_gap_pct","abs_rank_gap_pct"]),
        "CONF_ONLY":unique(market_features+["confidence_score"]),
        "RANK_CROSS":unique(market_features+["consensus_rank_pct","rank_gap","abs_rank_gap","rank_gap_pct","rank_product_pct"]),
        "FULL":unique(market_features+l1_features+["confidence_score","rank_gap","abs_rank_gap","rank_gap_pct","rank_product_pct"]),
    }

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]; metricrows=[]; thresholds=[]; importance=[]; scored_parts=[]

    for fold_i,(test_year,train_years) in enumerate(TEST_FOLDS,1):
        train=z[z["year"].isin(train_years)].copy()
        test=z[z["year"]==test_year].copy().reset_index(drop=True)
        if train["confidence_score"].isna().any() or test["confidence_score"].isna().any():
            raise SystemExit(f"confidence missing fold={test_year}")

        fit,cal=chronological_fit_cal_split(train)
        market_pair,pm_cal=fit_model(fit,cal,market_features,20261100+fold_i)
        pm_test=predict(market_pair,test,market_features)
        method_pairs={}
        cal_scores={}
        test_scores={}
        high_thr={}

        for mi,method in enumerate(METHODS,1):
            feats=method_features[method]
            pair,pcal=fit_model(fit,cal,feats,20261200+fold_i*20+mi)
            ptest=predict(pair,test,feats)
            method_pairs[method]=pair
            sign_cal=np.sign(cal["rank_gap"].to_numpy(dtype=float))
            sign_test=np.sign(test["rank_gap"].to_numpy(dtype=float))
            sc_cal=sign_cal*(pcal-pm_cal)
            sc_test=sign_test*(ptest-pm_test)
            cal_scores[method]=sc_cal
            test_scores[method]=sc_test
            high_thr[method]=high_threshold(cal,sc_cal)
            test[f"score_{method}"]=sc_test
            thresholds.append({"test_year":test_year,"method":method,"high_threshold":high_thr[method]})

            y=test["target_top3"].astype(int).to_numpy()
            metricrows.append({
                "test_year":test_year,"method":method,"horses":len(test),
                "log_loss":float(log_loss(y,np.clip(ptest,1e-12,1-1e-12))),
                "brier":float(brier_score_loss(y,ptest)),
                "delta_log_loss_vs_market":float(log_loss(y,np.clip(ptest,1e-12,1-1e-12))-log_loss(y,np.clip(pm_test,1e-12,1-1e-12))),
                "delta_brier_vs_market":float(brier_score_loss(y,ptest)-brier_score_loss(y,pm_test)),
            })
            booster=pair[0].booster_
            gains=booster.feature_importance(importance_type="gain")
            for name,gain in sorted(zip(feats,gains),key=lambda x:-x[1])[:20]:
                importance.append({"test_year":test_year,"method":method,"feature":name,"gain":float(gain),"is_market_feature":name in market_features})

        y=test["target_top3"].astype(int).to_numpy()
        metricrows.append({
            "test_year":test_year,"method":"MARKET_ONLY","horses":len(test),
            "log_loss":float(log_loss(y,np.clip(pm_test,1e-12,1-1e-12))),
            "brier":float(brier_score_loss(y,pm_test)),
            "delta_log_loss_vs_market":0.0,"delta_brier_vs_market":0.0,
        })

        test=peer_baseline(test,z[z["year"]==test_year])
        disagreements=test[test["rank_gap"].abs()>=2].copy()

        for direction in ("L1_UPGRADE","L1_DOWNGRADE"):
            d=disagreements[disagreements["direction"]==direction].copy()
            base={
                "test_year":test_year,"method":"ALL","selection_mode":"ALL_DISAGREEMENTS","direction":direction,
                "horses":len(d),"races":d["race_id"].nunique(),"coverage_within_direction_pct":100.0,
                "mean_support_score":None,"threshold":None,
                "actual_top3_pct":100*float(d["target_top3"].mean()),
                "market_peer_top3_pct":100*float(d["market_peer_top3_rate"].mean()),
                "direction_correct_effect_pp":effect(d),
            }
            rows.append(base)
            full_high=d[d["score_FULL"]>=high_thr["FULL"]].copy()
            full_n=len(full_high)
            for method in METHODS:
                h=d[d[f"score_{method}"]>=high_thr[method]].copy()
                rr=summarize(test_year,method,"CALIB_HIGH",direction,h,high_thr[method])
                rr["coverage_within_direction_pct"]=100*len(h)/len(d) if len(d) else None
                rows.append(rr)

                # Fair diagnostic: same number selected as FULL, based only on score ordering.
                n=min(full_n,len(d))
                mh=d.nlargest(n,f"score_{method}").copy() if n else d.iloc[:0].copy()
                rr=summarize(test_year,method,"MATCHED_FULL_COUNT",direction,mh,None)
                rr["coverage_within_direction_pct"]=100*len(mh)/len(d) if len(d) else None
                rows.append(rr)

        keepcols=["year","race_id","horse_id","horse_number","consensus_rank","market_rank","rank_gap","direction",
                  "confidence_score","confidence_tag","final_win_odds","target_top3","market_peer_top3_rate"]+[f"score_{m}" for m in METHODS]
        scored_parts.append(test[keepcols])
        print("STRONG_DISSENT_V2_FOLD_READY "+json.dumps({"test_year":test_year,"train_years":train_years},separators=(",",":")),flush=True)

    scored=pd.concat(scored_parts,ignore_index=True)
    write_csv(out/"benchmark.csv",rows)
    write_csv(out/"model-metrics.csv",metricrows)
    write_csv(out/"thresholds.csv",thresholds)
    write_csv(out/"feature-importance.csv",importance)
    scored.to_csv(out/"scored.csv.gz",index=False,compression="gzip")

    bench=pd.DataFrame(rows)
    decision_rows=[]
    for direction in ("L1_UPGRADE","L1_DOWNGRADE"):
        for mode in ("CALIB_HIGH","MATCHED_FULL_COUNT"):
            q=bench[(bench["direction"]==direction)&(bench["selection_mode"]==mode)&(bench["method"].isin(METHODS))].copy()
            for year in (2024,2025):
                yq=q[q["test_year"]==year]
                full=float(yq.loc[yq["method"]=="FULL","direction_correct_effect_pp"].iloc[0])
                best_simple=float(yq.loc[yq["method"]!="FULL","direction_correct_effect_pp"].max())
                best_method=str(yq.loc[yq["method"]!="FULL"].sort_values("direction_correct_effect_pp",ascending=False).iloc[0]["method"])
                decision_rows.append({
                    "direction":direction,"selection_mode":mode,"test_year":year,
                    "full_effect_pp":full,"best_simple_method":best_method,"best_simple_effect_pp":best_simple,
                    "full_minus_best_simple_pp":full-best_simple,"full_wins":bool(full>best_simple),
                })
    write_csv(out/"dominance.csv",decision_rows)

    strict={}
    for direction in ("L1_UPGRADE","L1_DOWNGRADE"):
        zq=[r for r in decision_rows if r["direction"]==direction and r["selection_mode"]=="MATCHED_FULL_COUNT"]
        strict[direction]={
            "full_wins_both_years":all(r["full_wins"] for r in zq),
            "details":zq,
        }
    summary={
        "contract":"L1_STRONG_DISSENT_V2",
        "purpose":"test whether FULL dissent adds reproducible information beyond simpler disagreement heuristics",
        "folds":[{"test_year":2024,"train_years":[2023]},{"test_year":2025,"train_years":[2023,2024]}],
        "methods":{
            "GAP_ONLY":"market model plus signed/absolute L1-market rank gap only",
            "CONF_ONLY":"market model plus strict-OOS L1 podium confidence only",
            "RANK_CROSS":"market model plus L1 rank and L1-market rank interaction only",
            "FULL":"market model plus frozen L1.7 internal features, confidence, and rank-gap features",
        },
        "high_rule":"top prior-calibration tertile among positive direction-support scores; no test outcome used",
        "matched_diagnostic":"within each test-year direction, each method selects the same count as FULL by score only",
        "strict_uniqueness_rule":"FULL must beat every simple baseline in MATCHED_FULL_COUNT effect in BOTH 2024 and 2025 for that direction",
        "strict_result":strict,
        "skipped_market_races":len(skipped),
        "core_L1_market_free":True,
        "rerank":False,
        "2026_locked":True,
        "promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== DOMINANCE =====")
    print((out/"dominance.csv").read_text(encoding="utf-8"))
    print("===== BENCHMARK =====")
    print((out/"benchmark.csv").read_text(encoding="utf-8"))
    print("L1_STRONG_DISSENT_V2_READY")

if __name__=="__main__":
    main()
