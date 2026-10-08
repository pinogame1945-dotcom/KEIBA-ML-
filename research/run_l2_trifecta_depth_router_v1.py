#!/usr/bin/env python3
import argparse,json,math,os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
 accuracy_score,balanced_accuracy_score,f1_score,log_loss,roc_auc_score,
 brier_score_loss,mean_absolute_error,confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from build_l2_bet_kings_dataset_v1 import decode_odds,payout_map
from run_l2_trifecta_odds_rank_reverse_v1 import read_gz,market_shape,race_meta

YEARS=(2021,2022,2023,2024,2025)
BANDS=("P00_02","P02_10","P10_25","P25_100")
WIDTHS=np.asarray([.02,.08,.15,.75],dtype=float)
TEST_YEARS=(2022,2023,2024,2025)
SELECTION_YEARS=(2022,2023,2024)
RANDOM_STATE=20261008

TRI_MARKET=[
 "top1_odds","top10_odds","top30_odds","top100_odds","median_odds","p90_odds",
 "top10_to_top1","top30_to_top1","median_to_top1",
 "top1_share","top10_share","top30_share","hhi","entropy_norm","effective_tickets"
]
WIN_MARKET=["win_top1_odds","win_top2_odds","win_top3_odds","win_top2_to_top1","win_top1_share","win_top3_share"]
RACE_NUM=["field_size","ticket_count","distance_m","race_month"]
RACE_CAT=["surface","discipline","venue_code","track_condition","weather","race_class","grade","distance_bin","field_size_bin"]

FEATURE_SETS={
 "FIELD_ONLY":{"num":["field_size","ticket_count","distance_m"],"cat":[]},
 "TRI_MARKET":{"num":TRI_MARKET,"cat":[]},
 "ALL_MARKET":{"num":TRI_MARKET+WIN_MARKET,"cat":[]},
 "MARKET_RACE":{"num":TRI_MARKET+WIN_MARKET+RACE_NUM,"cat":RACE_CAT},
}
MODELS=("LOGIT","EXTRA_TREES")

def parse_args():
 p=argparse.ArgumentParser()
 p.add_argument("--backfill-root",required=True)
 p.add_argument("--out-dir",required=True)
 p.add_argument("--workers",type=int,default=0)
 return p.parse_args()

def band_idx(p):
 if p<=.02:return 0
 if p<=.10:return 1
 if p<=.25:return 2
 return 3

def build_year(root,year):
 root=Path(root); rows=[]; c=Counter()
 files=sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz"))
 if not files: raise RuntimeError(f"no daily files year={year}")
 for dp in files:
  date=dp.name[:10]; op=root/"data"/"odds"/"daily"/dp.name
  if not op.exists(): c["missing_odds_day"]+=1; continue
  odds_rows={str(x.get("race_id") or ""):x for x in read_gz(op) if str(x.get("race_id") or "")}
  for pack in read_gz(dp):
   c["race_seen"]+=1
   rid=str((pack.get("race") or {}).get("race_id") or "")
   rec=odds_rows.get(rid)
   if not rid or rec is None: c["missing_odds_row"]+=1; continue
   omap=decode_odds(rec); sh=market_shape(omap)
   if sh is None: c["no_trifecta_odds"]+=1; continue
   if sh["ticket_count"]!=sh["expected_tickets"]: c["incomplete_universe"]+=1; continue
   payouts,present=payout_map(pack)
   if "TRIFECTA" not in present: c["missing_trifecta_payout"]+=1; continue
   wins=[tuple(k[1]) for k,v in payouts.items() if k[0]=="TRIFECTA" and float(v)>0]
   if len(wins)!=1: c["non_single_winner"]+=1; continue
   winner=wins[0]; rank=None; last=None; comp=0
   for pos,(nums,odd) in enumerate(sh["tri"],1):
    if last is None or odd!=last: comp=pos; last=odd
    if nums==winner: rank=comp; break
   if rank is None: c["winner_odds_missing"]+=1; continue
   row=race_meta(pack,year,date,rid,sh); row.update({k:v for k,v in sh.items() if k!="tri"})
   pct=(rank-1)/max(1,sh["ticket_count"]-1)
   row["winner_market_rank"]=int(rank); row["winner_rank_pct"]=float(pct); row["target"]=band_idx(pct)
   rows.append(row); c["ready"]+=1
 return pd.DataFrame(rows),dict(c)

def build_parallel(root,workers):
 maxw=max(1,min(workers or max(2,(os.cpu_count() or 2)-1),4,len(YEARS)))
 with ProcessPoolExecutor(max_workers=maxw) as ex:
  res=list(ex.map(build_year,[root]*len(YEARS),YEARS))
 frames=[x[0] for x in res]; counters={str(y):res[i][1] for i,y in enumerate(YEARS)}
 df=pd.concat(frames,ignore_index=True).sort_values(["year","race_date","race_id"]).reset_index(drop=True)
 return df,counters,maxw

def make_preprocessor(spec,scale=False):
 num=spec["num"]; cat=spec["cat"]; transformers=[]
 if num:
  nsteps=[("impute",SimpleImputer(strategy="median"))]
  if scale:nsteps.append(("scale",StandardScaler()))
  transformers.append(("num",Pipeline(nsteps),num))
 if cat:
  transformers.append(("cat",Pipeline([
   ("impute",SimpleImputer(strategy="most_frequent")),
   ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=True))
  ]),cat))
 return ColumnTransformer(transformers,remainder="drop",sparse_threshold=0.3)

def make_classifier(model_name,spec):
 if model_name=="LOGIT":
  model=LogisticRegression(max_iter=800,class_weight="balanced",solver="lbfgs",random_state=RANDOM_STATE)
  return Pipeline([("prep",make_preprocessor(spec,scale=True)),("model",model)])
 model=ExtraTreesClassifier(
  n_estimators=450,min_samples_leaf=8,max_features="sqrt",
  class_weight="balanced",random_state=RANDOM_STATE,n_jobs=1
 )
 return Pipeline([("prep",make_preprocessor(spec,scale=False)),("model",model)])

def metric_row(y,proba,pred):
 y=np.asarray(y,dtype=int); proba=np.asarray(proba,dtype=float); pred=np.asarray(pred,dtype=int)
 row={
  "accuracy":float(accuracy_score(y,pred)),
  "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
  "macro_f1":float(f1_score(y,pred,average="macro")),
  "log_loss":float(log_loss(y,proba,labels=[0,1,2,3])),
  "ordinal_mae":float(np.mean(np.abs(pred-y))),
 }
 for idx,name in ((0,"shallow2"),(3,"deep25")):
  yy=(y==idx).astype(int); pp=proba[:,idx]
  row[name+"_auc"]=float(roc_auc_score(yy,pp)) if len(np.unique(yy))>1 else None
  row[name+"_brier"]=float(brier_score_loss(yy,pp))
 return row

def prior_predict(train_y,n):
 counts=np.bincount(np.asarray(train_y,dtype=int),minlength=4).astype(float)+1e-9
 p=counts/counts.sum()
 return np.tile(p,(n,1))

def fit_task(df,test_year,feature_set,model_name):
 train=df[df.year<test_year]; test=df[df.year==test_year]
 ytr=train.target.astype(int).to_numpy(); yte=test.target.astype(int).to_numpy()
 if model_name=="PRIOR":
  proba=prior_predict(ytr,len(test)); pred=proba.argmax(axis=1); est=None
 else:
  spec=FEATURE_SETS[feature_set]; cols=spec["num"]+spec["cat"]
  est=make_classifier(model_name,spec); est.fit(train[cols],ytr)
  raw=est.predict_proba(test[cols]); classes=est.named_steps["model"].classes_.astype(int)
  proba=np.zeros((len(test),4),dtype=float)
  for j,c in enumerate(classes): proba[:,c]=raw[:,j]
  pred=proba.argmax(axis=1)
 row={"test_year":test_year,"feature_set":feature_set,"model":model_name,
      "train_races":len(train),"test_races":len(test)}
 row.update(metric_row(yte,proba,pred))
 return {"row":row,"y":yte,"proba":proba,"pred":pred,
         "ticket_count":test.ticket_count.to_numpy(dtype=int),
         "rank_pct":test.winner_rank_pct.to_numpy(dtype=float)}

def regression_task(df,test_year):
 train=df[df.year<test_year]; test=df[df.year==test_year]
 spec=FEATURE_SETS["MARKET_RACE"]; cols=spec["num"]+spec["cat"]
 est=Pipeline([
  ("prep",make_preprocessor(spec,scale=False)),
  ("model",ExtraTreesRegressor(n_estimators=450,min_samples_leaf=8,max_features=.8,random_state=RANDOM_STATE,n_jobs=1))
 ])
 est.fit(train[cols],train.winner_rank_pct.to_numpy(float))
 pred=np.clip(est.predict(test[cols]),0,1); y=test.winner_rank_pct.to_numpy(float)
 corr=float(pd.Series(pred).rank().corr(pd.Series(y).rank()))
 return {"test_year":test_year,"spearman":corr,"mae":float(mean_absolute_error(y,pred)),
         "median_abs_error":float(np.median(np.abs(y-pred))),"train_races":len(train),"test_races":len(test)}

def band_counts(n):
 r2=int(math.floor(.02*(n-1))+1);r10=int(math.floor(.10*(n-1))+1);r25=int(math.floor(.25*(n-1))+1)
 return np.asarray([r2,max(0,r10-r2),max(0,r25-r10),max(0,n-r25)],dtype=int)

def utility_tables(y,proba,tickets):
 rows=[]
 for k in (1,2,3):
  hits=[];cost=[]
  for yy,pp,n in zip(y,proba,tickets):
   chosen=np.argsort(-pp)[:k]; bc=band_counts(int(n))
   hits.append(int(int(yy) in set(chosen.tolist())));cost.append(int(bc[chosen].sum()))
  rows.append({"policy":f"TOP{k}_PREDICTED_BANDS","coverage_pct":100*float(np.mean(hits)),
               "avg_tickets":float(np.mean(cost)),"median_tickets":float(np.median(cost)),
               "avg_board_pct":100*float(np.mean(np.asarray(cost)/tickets))})
 for alpha in (.60,.70,.80,.90,.95):
  hits=[];cost=[]
  for yy,pp,n in zip(y,proba,tickets):
   cum=np.cumsum(pp); idx=int(np.searchsorted(cum,alpha,side="left")); idx=min(idx,3)
   bc=band_counts(int(n)); c=int(bc[:idx+1].sum())
   hits.append(int(int(yy)<=idx));cost.append(c)
  rows.append({"policy":f"CUMPROB_{int(alpha*100)}","coverage_pct":100*float(np.mean(hits)),
               "avg_tickets":float(np.mean(cost)),"median_tickets":float(np.median(cost)),
               "avg_board_pct":100*float(np.mean(np.asarray(cost)/tickets))})
 for idx,label in ((0,"FIXED_TOP2PCT"),(1,"FIXED_TOP10PCT"),(2,"FIXED_TOP25PCT"),(3,"FIXED_ALL")):
  hits=[];cost=[]
  for yy,n in zip(y,tickets):
   bc=band_counts(int(n));c=int(bc[:idx+1].sum())
   hits.append(int(int(yy)<=idx));cost.append(c)
  rows.append({"policy":label,"coverage_pct":100*float(np.mean(hits)),
               "avg_tickets":float(np.mean(cost)),"median_tickets":float(np.median(cost)),
               "avg_board_pct":100*float(np.mean(np.asarray(cost)/tickets))})
 return pd.DataFrame(rows)

def calibration(y,proba):
 out=[]
 for cls,name in ((0,"P00_02"),(3,"P25_100")):
  p=proba[:,cls]; yy=(y==cls).astype(int)
  bins=pd.qcut(pd.Series(p),q=10,duplicates="drop")
  tmp=pd.DataFrame({"p":p,"y":yy,"bin":bins})
  for b,g in tmp.groupby("bin",observed=True):
   out.append({"target":name,"bin":str(b),"races":len(g),
               "predicted_pct":100*float(g.p.mean()),"observed_pct":100*float(g.y.mean())})
 return pd.DataFrame(out)

def target_stability(df):
 out=[]
 for y,g in df.groupby("year"):
  r={"year":int(y),"races":len(g),"median_rank_pct":float(g.winner_rank_pct.median())}
  for i,b in enumerate(BANDS):r[b+"_pct"]=100*float((g.target==i).mean())
  out.append(r)
 return pd.DataFrame(out)

def feature_importance(df,feature_set,model_name):
 train=df[df.year<=2024]; spec=FEATURE_SETS[feature_set]; cols=spec["num"]+spec["cat"]
 est=make_classifier(model_name,spec);est.fit(train[cols],train.target.to_numpy(int))
 prep=est.named_steps["prep"];names=np.asarray(prep.get_feature_names_out(),dtype=str);m=est.named_steps["model"]
 if hasattr(m,"feature_importances_"): imp=np.asarray(m.feature_importances_,dtype=float)
 else: imp=np.mean(np.abs(np.asarray(m.coef_,dtype=float)),axis=0)
 order=np.argsort(-imp)[:30]
 return pd.DataFrame({"feature":names[order],"importance":imp[order]})

def main():
 a=parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
 df,counters,extract_workers=build_parallel(a.backfill_root,a.workers)
 if set(df.year.unique())!=set(YEARS):raise SystemExit("year coverage guard")
 stability=target_stability(df);stability.to_csv(out/"target-stability.csv",index=False)

 tasks=[(y,"PRIOR","PRIOR") for y in TEST_YEARS]
 tasks += [(y,fs,m) for y in TEST_YEARS for fs in FEATURE_SETS for m in MODELS]
 workers=max(1,min(a.workers or max(2,(os.cpu_count() or 2)-1),4))
 results=Parallel(n_jobs=workers,backend="loky")(
  delayed(fit_task)(df,y,fs,m) for y,fs,m in tasks
 )
 metrics=pd.DataFrame([r["row"] for r in results])
 metrics.to_csv(out/"walkforward-metrics.csv",index=False)

 candidates=metrics[(metrics.test_year.isin(SELECTION_YEARS)) & (metrics.model!="PRIOR")]
 sel=(candidates.groupby(["feature_set","model"],as_index=False)
      .agg(selection_log_loss=("log_loss","mean"),selection_macro_f1=("macro_f1","mean"),
           selection_ordinal_mae=("ordinal_mae","mean"),selection_deep_auc=("deep25_auc","mean"),
           selection_shallow_auc=("shallow2_auc","mean"))
      .sort_values(["selection_log_loss","selection_ordinal_mae"],ascending=[True,True]))
 sel.to_csv(out/"model-selection-2022-2024.csv",index=False)
 best=sel.iloc[0];best_fs=str(best.feature_set);best_model=str(best.model)
 chosen=next(r for r in results if r["row"]["test_year"]==2025 and r["row"]["feature_set"]==best_fs and r["row"]["model"]==best_model)
 prior=next(r for r in results if r["row"]["test_year"]==2025 and r["row"]["model"]=="PRIOR")

 util=utility_tables(chosen["y"],chosen["proba"],chosen["ticket_count"]);util.to_csv(out/"2025-router-utility.csv",index=False)
 cal=calibration(chosen["y"],chosen["proba"]);cal.to_csv(out/"2025-calibration.csv",index=False)
 cm=confusion_matrix(chosen["y"],chosen["pred"],labels=[0,1,2,3])
 pd.DataFrame(cm,index=BANDS,columns=BANDS).to_csv(out/"2025-confusion.csv")
 imp=feature_importance(df,best_fs,best_model);imp.to_csv(out/"selected-feature-importance.csv",index=False)

 reg=Parallel(n_jobs=min(workers,len(TEST_YEARS)),backend="loky")(delayed(regression_task)(df,y) for y in TEST_YEARS)
 pd.DataFrame(reg).to_csv(out/"continuous-rank-regression.csv",index=False)

 hold=chosen["row"];prior25=prior["row"]
 ll_gain=100*(prior25["log_loss"]-hold["log_loss"])/prior25["log_loss"]
 verdict="PROMOTE_TO_L2_EXPERIMENT" if (ll_gain>=2 and hold["deep25_auc"]>=.62 and hold["shallow2_auc"]>=.62) else "RESEARCH_ONLY"
 summary={
  "contract":"L2_TRIFECTA_DEPTH_ROUTER_V1_RESULT","races":len(df),"years":list(YEARS),
  "target_bands":{"P00_02":"top 2%","P02_10":"2-10%","P10_25":"10-25%","P25_100":"25-100%"},
  "selection_rule":"choose lowest mean multiclass log loss on expanding walk-forward tests 2022-2024; 2025 untouched until final evaluation",
  "selected":{"feature_set":best_fs,"model":best_model,
              "selection_2022_2024":best.replace({np.nan:None}).to_dict(),
              "holdout_2025":hold,
              "prior_2025":prior25,
              "logloss_improvement_vs_prior_pct":ll_gain},
  "verdict":verdict,
  "promotion_gate":{"logloss_improvement_vs_prior_pct_min":2.0,"deep25_auc_min":.62,"shallow2_auc_min":.62},
  "parallelism":{"feature_extraction_processes":extract_workers,"model_fold_processes":workers},
  "input_counters_by_year":counters,"2026_locked":True
 }
 (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 (out/"README.md").write_text(
  "# L2 Trifecta Depth Router V1\n\n"
  "One-pass research bundle: percentile target stability, expanding walk-forward model comparison, 2025 frozen holdout, "
  "continuous depth regression, calibration, confusion, feature importance, and dynamic search-budget utility. "
  "Feature extraction occurs once from a pinned BACKFILL checkout and yearly extraction/model folds are CPU-parallelized. "
  "2026 remains sealed. Raw race rows are ephemeral; only compact summaries are committed.\n",encoding="utf-8")
 print("TRIFECTA_DEPTH_ROUTER_V1_READY")
 print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
 print("===== SELECTION =====");print(sel.to_csv(index=False))
 print("===== 2025 UTILITY =====");print(util.to_csv(index=False))
 print("===== STABILITY =====");print(stability.to_csv(index=False))
 print("===== REGRESSION =====");print(pd.DataFrame(reg).to_csv(index=False))
 print("===== IMPORTANCE =====");print(imp.head(20).to_csv(index=False))

if __name__=="__main__":main()
