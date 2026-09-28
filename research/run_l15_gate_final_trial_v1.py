#!/usr/bin/env python3
import argparse,json,math
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from train_l15_king_anxiety_gate_v1 import (
    uncertainty_features, labels, op, encode, make_model, router_features, BUDGETS
)

p=argparse.ArgumentParser()
p.add_argument("--router-year",action="append",required=True)
p.add_argument("--scorecard",required=True)
p.add_argument("--out",required=True)
a=p.parse_args()

CONSENSUS_KEEP={
 "top3_jaccard","top6_jaccard","top3_union","top6_union",
 "rank_diff_mean","rank_std_mean","prob_std_mean","prob_std_max",
 "top3_mass_mean","top6_mass_mean","entropy_mean","entropy_std",
 "vote_pattern","top1_unique","top1_max_vote","top1_max_vote_share"
}

def consensus_world(rec):
 f=uncertainty_features(rec,True)
 return {k:v for k,v in f.items() if k in CONSENSUS_KEEP}

def encode_rows(train,test,key):
 return encode([r[key] for r in train],[r[key] for r in test])

def fit_predict(train,test,key):
 Xtr,Xte=encode_rows(train,test,key)
 ytr=np.array([r["blind"] for r in train],dtype=int)
 m=make_model(seed=1945+len(train)+len(test))
 m.fit(Xtr,ytr)
 pred=m.predict_proba(Xte)[:,1]
 gains=m.booster_.feature_importance(importance_type="gain")
 names=m.booster_.feature_name()
 imp=sorted(({"feature":n,"gain":float(g)} for n,g in zip(names,gains)),key=lambda x:x["gain"],reverse=True)[:20]
 return pred,imp,len(Xtr.columns)

def budget_one_year(y,p,b):
 order=np.argsort(-p); k=max(1,int(math.ceil(len(y)*b))); idx=order[:k]
 hit=int(y[idx].sum()); precision=hit/k; recall=hit/int(y.sum())
 return {"selected":k,"blind":hit,"recall":float(recall),"precision":float(precision),
         "lift":float(precision/(float(y.mean()) or 1))}

def aggregate_budget(vals,b):
 y=np.array([x[2] for x in vals],dtype=int)
 chosen=[]; hit=0
 for yy in (2022,2023,2024,2025):
  vv=sorted([x for x in vals if x[0]==yy],key=lambda x:x[3],reverse=True)
  k=max(1,int(math.ceil(len(vv)*b))); z=vv[:k]
  chosen.extend(z); hit+=sum(x[2] for x in z)
 precision=hit/len(chosen); recall=hit/int(y.sum())
 return {"selected":len(chosen),"blind":int(hit),"recall":float(recall),
         "precision":float(precision),"lift":float(precision/(float(y.mean()) or 1))}

paths={}
for s in a.router_year:
 y,path=s.split(":",1); paths[int(y)]=path
need={2021,2022,2023,2024,2025}
if set(paths)!=need: raise SystemExit(f"need years {sorted(need)} got {sorted(paths)}")

score=json.load(open(a.scorecard,encoding="utf-8")); lab=labels(score)
if not need.issubset(lab): raise SystemExit(f"labels missing {sorted(need-set(lab))}")

rows=[]
for y,path in sorted(paths.items()):
 with op(path) as fh:
  for line in fh:
   if not line.strip(): continue
   rec=json.loads(line); rid=str(rec.get("race_id") or "")
   if rid not in lab[y]: raise ValueError(f"label missing {y} {rid}")
   rows.append({
     "year":y,"race_id":rid,"blind":lab[y][rid],
     "current_gate":router_features(rec),
     "anxiety_gate":uncertainty_features(rec,True),
     "consensus_world":consensus_world(rec)
   })

MODELS=("current_gate","anxiety_gate","consensus_world")
pooled={k:[] for k in MODELS}; folds=[]
for test_year in (2022,2023,2024,2025):
 train=[r for r in rows if r["year"]<test_year]
 test=[r for r in rows if r["year"]==test_year]
 y=np.array([r["blind"] for r in test],dtype=int)
 fold={"test_year":test_year,"races":len(test),"blind":int(y.sum()),"models":{}}
 for key in MODELS:
  pred,imp,fc=fit_predict(train,test,key)
  mm={"roc_auc":float(roc_auc_score(y,pred)),
      "pr_auc":float(average_precision_score(y,pred)),
      "feature_count":fc,"top_features":imp,"budgets":{}}
  for b in BUDGETS:
   mm["budgets"][str(int(b*100))]=budget_one_year(y,pred,b)
  fold["models"][key]=mm
  for r,pp in zip(test,pred):
   pooled[key].append((test_year,r["race_id"],r["blind"],float(pp)))
 folds.append(fold)

aggregate={}
for key,vals in pooled.items():
 y=np.array([x[2] for x in vals],dtype=int); pred=np.array([x[3] for x in vals],dtype=float)
 mm={"roc_auc":float(roc_auc_score(y,pred)),
     "pr_auc":float(average_precision_score(y,pred)),
     "prevalence":float(y.mean()),"budgets":{}}
 for b in BUDGETS:
  mm["budgets"][str(int(b*100))]=aggregate_budget(vals,b)
 aggregate[key]=mm

# direct pairwise deltas vs current gate
comparison=[]
base=aggregate["current_gate"]
for key in MODELS:
 m=aggregate[key]
 comparison.append({
  "model":key,
  "roc_auc":m["roc_auc"],"delta_roc_vs_current":m["roc_auc"]-base["roc_auc"],
  "pr_auc":m["pr_auc"],"delta_pr_vs_current":m["pr_auc"]-base["pr_auc"],
  "b10_blind":m["budgets"]["10"]["blind"],
  "b10_recall":m["budgets"]["10"]["recall"],
  "delta_b10_blind_vs_current":m["budgets"]["10"]["blind"]-base["budgets"]["10"]["blind"],
  "b20_blind":m["budgets"]["20"]["blind"],
  "b20_recall":m["budgets"]["20"]["recall"],
  "delta_b20_blind_vs_current":m["budgets"]["20"]["blind"]-base["budgets"]["20"]["blind"]
 })

out={
 "contract":"L15_GATE_FINAL_TRIAL_V1",
 "models":MODELS,
 "consensus_keep":sorted(CONSENSUS_KEEP),
 "folds":folds,"aggregate":aggregate,"comparison":comparison,
 "odds_used":False,"locked_years":[2026],
 "note":"Fixed-seed final research trial. No 2026 labels used. Same folds and budgets for all models."
}
open(a.out,"w",encoding="utf-8").write(json.dumps(out,ensure_ascii=False,indent=2))
print("L15_GATE_FINAL_TRIAL_RESULT")
print(json.dumps({"comparison":comparison},ensure_ascii=False,separators=(",",":")))
