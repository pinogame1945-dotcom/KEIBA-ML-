#!/usr/bin/env python3
import argparse,json,math,os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,balanced_accuracy_score,brier_score_loss,
    log_loss,roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from run_l2_trifecta_depth_router_v1 import build_parallel,FEATURE_SETS,YEARS,RANDOM_STATE

BOUNDARIES=("TOP10","DEEP25")
TEST_YEARS=(2023,2024)
CAL_METHODS=("NONE","SIGMOID","ISOTONIC")
MODEL_NAMES=("LOGIT","EXTRA_TREES")
FEATURE_NAMES=("FIELD_ONLY","TRI_MARKET","ALL_MARKET","MARKET_RACE")
BUDGETS=(15,20,25,30,40,50)
COVERAGE_TARGETS=(85,90,95)

def parse_args():
 p=argparse.ArgumentParser()
 p.add_argument("--backfill-root",required=True)
 p.add_argument("--out-dir",required=True)
 p.add_argument("--workers",type=int,default=0)
 return p.parse_args()

def target(df,boundary):
 if boundary=="TOP10": return (df.winner_rank_pct<=.10).astype(int).to_numpy()
 if boundary=="DEEP25": return (df.winner_rank_pct>.25).astype(int).to_numpy()
 raise ValueError(boundary)

def make_preprocessor(spec,scale):
 tr=[]
 if spec["num"]:
  steps=[("impute",SimpleImputer(strategy="median"))]
  if scale:steps.append(("scale",StandardScaler()))
  tr.append(("num",Pipeline(steps),spec["num"]))
 if spec["cat"]:
  tr.append(("cat",Pipeline([
   ("impute",SimpleImputer(strategy="most_frequent")),
   ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=True))
  ]),spec["cat"]))
 return ColumnTransformer(tr,remainder="drop",sparse_threshold=.3)

def make_model(model_name,spec):
 if model_name=="LOGIT":
  m=LogisticRegression(max_iter=700,class_weight="balanced",solver="lbfgs",random_state=RANDOM_STATE)
  return Pipeline([("prep",make_preprocessor(spec,True)),("model",m)])
 m=ExtraTreesClassifier(
  n_estimators=240,min_samples_leaf=8,max_features="sqrt",
  class_weight="balanced",random_state=RANDOM_STATE,n_jobs=1
 )
 return Pipeline([("prep",make_preprocessor(spec,False)),("model",m)])

def pos_prob(est,X):
 classes=est.named_steps["model"].classes_.astype(int)
 raw=est.predict_proba(X)
 j=int(np.where(classes==1)[0][0])
 return np.asarray(raw[:,j],dtype=float)

def base_task(df,boundary,test_year,feature_name,model_name):
 base=df[df.year<test_year-1]
 cal=df[df.year==test_year-1]
 test=df[df.year==test_year]
 spec=FEATURE_SETS[feature_name];cols=spec["num"]+spec["cat"]
 yb=target(base,boundary);yc=target(cal,boundary);yt=target(test,boundary)
 est=make_model(model_name,spec);est.fit(base[cols],yb)
 return {
  "boundary":boundary,"test_year":test_year,"feature_set":feature_name,"model":model_name,
  "base_n":len(base),"cal_n":len(cal),"test_n":len(test),
  "cal_idx":cal.index.to_numpy(dtype=int),"test_idx":test.index.to_numpy(dtype=int),
  "y_cal":yc,"y_test":yt,
  "p_cal_raw":pos_prob(est,cal[cols]),"p_test_raw":pos_prob(est,test[cols])
 }

def clipped_logit(p):
 p=np.clip(np.asarray(p,dtype=float),1e-5,1-1e-5)
 return np.log(p/(1-p))

def calibrate(method,pcal,ycal,ptest):
 if method=="NONE":return np.clip(ptest,1e-6,1-1e-6)
 if method=="SIGMOID":
  lr=LogisticRegression(C=1e6,solver="lbfgs",max_iter=300)
  lr.fit(clipped_logit(pcal).reshape(-1,1),ycal)
  return np.clip(lr.predict_proba(clipped_logit(ptest).reshape(-1,1))[:,1],1e-6,1-1e-6)
 iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds="clip")
 iso.fit(np.asarray(pcal,float),np.asarray(ycal,int))
 return np.clip(iso.predict(np.asarray(ptest,float)),1e-6,1-1e-6)

def ece(y,p,bins=10):
 y=np.asarray(y,int);p=np.asarray(p,float);edges=np.linspace(0,1,bins+1);v=0.0
 for i in range(bins):
  mask=(p>=edges[i]) & ((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))
  if mask.any():v+=mask.mean()*abs(float(p[mask].mean())-float(y[mask].mean()))
 return float(v)

def metrics(y,p):
 y=np.asarray(y,int);p=np.asarray(p,float);pred=(p>=.5).astype(int)
 return {
  "auc":float(roc_auc_score(y,p)),
  "average_precision":float(average_precision_score(y,p)),
  "brier":float(brier_score_loss(y,p)),
  "log_loss":float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1])),
  "balanced_accuracy_05":float(balanced_accuracy_score(y,pred)),
  "prevalence":float(y.mean()),"mean_pred":float(p.mean()),"ece10":ece(y,p)
 }

def selection_rows(base_results):
 out=[]
 for r in base_results:
  for cm in CAL_METHODS:
   p=calibrate(cm,r["p_cal_raw"],r["y_cal"],r["p_test_raw"])
   row={k:r[k] for k in ("boundary","test_year","feature_set","model","base_n","cal_n","test_n")}
   row["calibration"]=cm;row.update(metrics(r["y_test"],p));out.append(row)
 return pd.DataFrame(out)

def select_configs(rows):
 agg=(rows.groupby(["boundary","feature_set","model","calibration"],as_index=False)
      .agg(selection_brier=("brier","mean"),selection_log_loss=("log_loss","mean"),
           selection_auc=("auc","mean"),selection_ap=("average_precision","mean"),
           selection_ece=("ece10","mean")))
 agg=agg.sort_values(["boundary","selection_brier","selection_log_loss","selection_ece"],
                     ascending=[True,True,True,True])
 winners=agg.groupby("boundary",as_index=False).first()
 return agg,winners

def result_lookup(results,boundary,test_year,fs,model):
 return next(r for r in results if r["boundary"]==boundary and r["test_year"]==test_year
             and r["feature_set"]==fs and r["model"]==model)

def selected_oof(df,results,winners):
 frames=[]
 for boundary in BOUNDARIES:
  w=winners[winners.boundary==boundary].iloc[0]
  parts=[]
  for y in TEST_YEARS:
   r=result_lookup(results,boundary,y,str(w.feature_set),str(w.model))
   p=calibrate(str(w.calibration),r["p_cal_raw"],r["y_cal"],r["p_test_raw"])
   parts.append(pd.DataFrame({"idx":r["test_idx"],"p":p}))
  q=pd.concat(parts,ignore_index=True).rename(columns={"p":"p_top10" if boundary=="TOP10" else "p_deep25"})
  frames.append(q)
 z=frames[0].merge(frames[1],on="idx",how="inner")
 base=df.loc[z.idx.to_numpy()].copy()
 base=base[["winner_rank_pct","ticket_count","year"]].reset_index(drop=True)
 z=z.reset_index(drop=True)
 return pd.concat([base,z[["p_top10","p_deep25"]]],axis=1)

def final_boundary(df,boundary,winner):
 base=df[df.year<=2023];cal=df[df.year==2024];test=df[df.year==2025]
 fs=str(winner.feature_set);model=str(winner.model);cm=str(winner.calibration)
 spec=FEATURE_SETS[fs];cols=spec["num"]+spec["cat"]
 est=make_model(model,spec);est.fit(base[cols],target(base,boundary))
 pcal=pos_prob(est,cal[cols]);ptest=pos_prob(est,test[cols])
 p=calibrate(cm,pcal,target(cal,boundary),ptest)
 prior=float(target(df[df.year<=2024],boundary).mean())
 return {
  "boundary":boundary,"feature_set":fs,"model":model,"calibration":cm,
  "base_n":len(base),"cal_n":len(cal),"test_n":len(test),
  "test_idx":test.index.to_numpy(dtype=int),"y":target(test,boundary),"p":p,
  "prior":prior,"metrics":metrics(target(test,boundary),p),
  "prior_metrics":metrics(target(test,boundary),np.full(len(test),prior,dtype=float))
 }

def cutoff_policy(ptop,pdeep,t10,tdeep):
 ptop=np.asarray(ptop,float);pdeep=np.asarray(pdeep,float)
 cut=np.full(len(ptop),.25,dtype=float)
 cut[pdeep>=tdeep]=1.0
 cut[(pdeep<tdeep)&(ptop>=t10)]=.10
 return cut

def policy_metrics(rank_pct,tickets,cut):
 rank_pct=np.asarray(rank_pct,float);tickets=np.asarray(tickets,int);cut=np.asarray(cut,float)
 hit=(rank_pct<=cut+1e-12)
 searched=np.ceil(cut*tickets).astype(int)
 return {
  "coverage_pct":100*float(hit.mean()),
  "avg_board_pct":100*float(cut.mean()),
  "avg_tickets":float(searched.mean()),"median_tickets":float(np.median(searched)),
  "top10_route_pct":100*float((cut==.10).mean()),
  "top25_route_pct":100*float((cut==.25).mean()),
  "full_route_pct":100*float((cut==1.0).mean())
 }

def grid_policies(frame):
 rows=[]
 for t10 in np.round(np.arange(.40,.901,.025),3):
  for td in np.round(np.arange(.05,.401,.015),3):
   cut=cutoff_policy(frame.p_top10,frame.p_deep25,t10,td)
   r={"t_top10":float(t10),"t_deep25":float(td)};r.update(policy_metrics(frame.winner_rank_pct,frame.ticket_count,cut));rows.append(r)
 return pd.DataFrame(rows)

def select_budget_policies(grid):
 out=[]
 for b in BUDGETS:
  q=grid[grid.avg_board_pct<=b].copy()
  if q.empty:continue
  q=q.sort_values(["coverage_pct","avg_board_pct"],ascending=[False,True])
  r=q.iloc[0].to_dict();r["selection_type"]="BUDGET";r["selection_target"]=float(b);out.append(r)
 for cov in COVERAGE_TARGETS:
  q=grid[grid.coverage_pct>=cov].copy()
  if q.empty:continue
  q=q.sort_values(["avg_board_pct","coverage_pct"],ascending=[True,False])
  r=q.iloc[0].to_dict();r["selection_type"]="COVERAGE";r["selection_target"]=float(cov);out.append(r)
 return pd.DataFrame(out)

def apply_selected_policies(sel,frame):
 out=[]
 for _,s in sel.iterrows():
  cut=cutoff_policy(frame.p_top10,frame.p_deep25,float(s.t_top10),float(s.t_deep25))
  r={"selection_type":s.selection_type,"selection_target":float(s.selection_target),
     "t_top10":float(s.t_top10),"t_deep25":float(s.t_deep25)}
  r.update(policy_metrics(frame.winner_rank_pct,frame.ticket_count,cut));out.append(r)
 return pd.DataFrame(out)

def fixed_curve(frame):
 out=[]
 for pct in (2,5,10,15,20,25,30,40,50,75,100):
  cut=np.full(len(frame),pct/100,dtype=float)
  r={"fixed_pct":pct};r.update(policy_metrics(frame.winner_rank_pct,frame.ticket_count,cut));out.append(r)
 return pd.DataFrame(out)

def calibration_table(y,p,boundary):
 q=pd.DataFrame({"y":np.asarray(y,int),"p":np.asarray(p,float)})
 q["bin"]=pd.qcut(q.p,q=10,duplicates="drop")
 out=[]
 for b,g in q.groupby("bin",observed=True):
  out.append({"boundary":boundary,"bin":str(b),"races":len(g),
              "predicted_pct":100*float(g.p.mean()),"observed_pct":100*float(g.y.mean())})
 return pd.DataFrame(out)

def feature_importance(df,boundary,winner):
 base=df[df.year<=2023];fs=str(winner.feature_set);mn=str(winner.model)
 spec=FEATURE_SETS[fs];cols=spec["num"]+spec["cat"];est=make_model(mn,spec);est.fit(base[cols],target(base,boundary))
 prep=est.named_steps["prep"];names=np.asarray(prep.get_feature_names_out(),dtype=str);m=est.named_steps["model"]
 if hasattr(m,"feature_importances_"):imp=np.asarray(m.feature_importances_,float)
 else:imp=np.abs(np.asarray(m.coef_,float)).reshape(-1)
 order=np.argsort(-imp)[:25]
 return pd.DataFrame({"boundary":boundary,"feature":names[order],"importance":imp[order]})

def main():
 a=parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
 workers=max(1,min(a.workers or max(2,(os.cpu_count() or 2)-1),3))
 df,counters,extract_workers=build_parallel(a.backfill_root,workers)

 tasks=[(b,y,fs,m) for b in BOUNDARIES for y in TEST_YEARS for fs in FEATURE_NAMES for m in MODEL_NAMES]
 results=Parallel(n_jobs=workers,backend="loky")(delayed(base_task)(df,*t) for t in tasks)
 rows=selection_rows(results);agg,winners=select_configs(rows)
 rows.to_csv(out/"nested-walkforward.csv",index=False);agg.to_csv(out/"selection-summary.csv",index=False);winners.to_csv(out/"selected-configs.csv",index=False)

 oof=selected_oof(df,results,winners);grid=grid_policies(oof);sel=select_budget_policies(grid)
 grid.to_csv(out/"validation-policy-grid.csv",index=False);sel.to_csv(out/"validation-selected-policies.csv",index=False)

 finals=[]
 for b in BOUNDARIES:
  w=winners[winners.boundary==b].iloc[0];finals.append(final_boundary(df,b,w))
 hold_rows=[]
 for r in finals:
  row={k:r[k] for k in ("boundary","feature_set","model","calibration","base_n","cal_n","test_n","prior")}
  row.update({"model_"+k:v for k,v in r["metrics"].items()})
  row.update({"prior_"+k:v for k,v in r["prior_metrics"].items()});hold_rows.append(row)
 pd.DataFrame(hold_rows).to_csv(out/"2025-boundary-metrics.csv",index=False)

 top=next(r for r in finals if r["boundary"]=="TOP10");deep=next(r for r in finals if r["boundary"]=="DEEP25")
 if not np.array_equal(top["test_idx"],deep["test_idx"]):raise SystemExit("holdout index mismatch")
 test=df.loc[top["test_idx"],["winner_rank_pct","ticket_count","year"]].reset_index(drop=True)
 test["p_top10"]=top["p"];test["p_deep25"]=deep["p"]
 applied=apply_selected_policies(sel,test);fixed=fixed_curve(test)
 applied.to_csv(out/"2025-dynamic-policies.csv",index=False);fixed.to_csv(out/"2025-fixed-curve.csv",index=False)

 cal=pd.concat([calibration_table(r["y"],r["p"],r["boundary"]) for r in finals],ignore_index=True)
 cal.to_csv(out/"2025-calibration.csv",index=False)
 fi=pd.concat([feature_importance(df,b,winners[winners.boundary==b].iloc[0]) for b in BOUNDARIES],ignore_index=True)
 fi.to_csv(out/"feature-importance.csv",index=False)

 fixed25=fixed[fixed.fixed_pct==25].iloc[0]
 b25=applied[(applied.selection_type=="BUDGET")&(applied.selection_target==25)]
 if len(b25):
  dyn=b25.iloc[0]
  frontier_gain=float(dyn.coverage_pct-fixed25.coverage_pct)
  board_saving=float(fixed25.avg_board_pct-dyn.avg_board_pct)
 else:
  frontier_gain=None;board_saving=None

 auc_top=float(top["metrics"]["auc"]);auc_deep=float(deep["metrics"]["auc"])
 brier_gain_top=100*(top["prior_metrics"]["brier"]-top["metrics"]["brier"])/top["prior_metrics"]["brier"]
 brier_gain_deep=100*(deep["prior_metrics"]["brier"]-deep["metrics"]["brier"])/deep["prior_metrics"]["brier"]
 promote=(
  auc_top>=.65 and auc_deep>=.68 and brier_gain_top>0 and brier_gain_deep>0 and
  b25.shape[0]>0 and (
   (frontier_gain is not None and frontier_gain>=.5 and float(b25.iloc[0].avg_board_pct)<=25.0) or
   (frontier_gain is not None and frontier_gain>=-.5 and board_saving is not None and board_saving>=2.5)
  )
 )
 summary={
  "contract":"L2_TRIFECTA_BOUNDARY_ROUTER_V1_RESULT","races":len(df),"years":list(YEARS),
  "boundaries":{"TOP10":"winner in top 10% of complete trifecta odds board","DEEP25":"winner deeper than 25%"},
  "selection":"nested time order: base train through y-2, calibrate on y-1, test y; select on 2023-2024 only; final base through 2023, calibrate 2024, test 2025",
  "selected_configs":winners.replace({np.nan:None}).to_dict("records"),
  "holdout_2025":hold_rows,
  "fixed25":{"coverage_pct":float(fixed25.coverage_pct),"avg_board_pct":float(fixed25.avg_board_pct),"avg_tickets":float(fixed25.avg_tickets)},
  "dynamic_budget25":None if b25.empty else b25.iloc[0].replace({np.nan:None}).to_dict(),
  "dynamic_minus_fixed25_coverage_pp":frontier_gain,
  "fixed25_minus_dynamic_board_pp":board_saving,
  "brier_improvement_vs_prior_pct":{"TOP10":brier_gain_top,"DEEP25":brier_gain_deep},
  "verdict":"PROMOTE_BOUNDARY_ROUTER" if promote else "RESEARCH_ONLY",
  "promotion_gate":"both holdout AUC gates, both Brier gains positive, and dynamic 25%-budget frontier materially improves coverage or compression",
  "parallelism":{"feature_extraction_processes":extract_workers,"model_processes":workers},
  "input_counters_by_year":counters,"2026_locked":True
 }
 (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 (out/"README.md").write_text(
  "# L2 Trifecta Boundary Router V1\n\n"
  "Consolidated binary-boundary study. Separate calibrated models predict whether the winning trifecta lies within the top 10% or beyond 25% of the complete final-odds board. "
  "Model/calibration choice and routing thresholds are selected only on pre-2025 walk-forward predictions. 2025 is a frozen final holdout; 2026 remains sealed. "
  "Feature extraction happens once and model tasks are CPU-parallelized. No paid storage or artifacts.\n",encoding="utf-8")
 print("TRIFECTA_BOUNDARY_ROUTER_V1_READY");print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
 print("===== SELECTED =====");print(winners.to_csv(index=False))
 print("===== HOLDOUT =====");print(pd.DataFrame(hold_rows).to_csv(index=False))
 print("===== DYNAMIC =====");print(applied.to_csv(index=False))
 print("===== FIXED =====");print(fixed.to_csv(index=False))

if __name__=="__main__":main()
