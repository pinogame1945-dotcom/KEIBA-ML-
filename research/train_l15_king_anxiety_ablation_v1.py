#!/usr/bin/env python3
import argparse,gzip,json,math
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from train_l15_king_anxiety_gate_v1 import (
    uncertainty_features, labels, op, encode, make_model, BUDGETS
)

p=argparse.ArgumentParser()
p.add_argument("--router-year",action="append",required=True)
p.add_argument("--scorecard",required=True)
p.add_argument("--out",required=True)
a=p.parse_args()

VARIANTS={
 "full":{
   "drop":set()
 },
 "no_vote_pattern":{
   "drop":{"vote_pattern","top1_unique","top1_max_vote","top1_max_vote_share"}
 },
 "no_top6_mass":{
   "drop":{"top6_mass_mean"}
 },
 "no_probability_level":{
   "drop":{"p1_mean","p1_min","p1_max","top3_mass_mean","top6_mass_mean"}
 },
 "no_margin":{
   "drop":{"g12_mean","g12_std","g12_min","g12_max","g13_mean","g13_std"}
 },
 "no_entropy":{
   "drop":{"entropy_mean","entropy_std"}
 },
 "no_overlap_structure":{
   "drop":{"top3_jaccard","top6_jaccard","top3_union","top6_union","rank_diff_mean","rank_std_mean"}
 },
 "no_probability_disagreement":{
   "drop":{"p1_std","prob_std_mean","prob_std_max"}
 },
 "top1_core_only":{
   "keep":{"p1_mean","p1_std","p1_min","p1_max","g12_mean","g12_std","g12_min","g12_max","g13_mean","g13_std","entropy_mean","entropy_std","vote_pattern","top1_unique","top1_max_vote","top1_max_vote_share"}
 },
 "prob_margin_only":{
   "keep":{"p1_mean","p1_std","p1_min","p1_max","g12_mean","g12_std","g12_min","g12_max","g13_mean","g13_std","top3_mass_mean","top6_mass_mean","prob_std_mean","prob_std_max"}
 },
 "structure_only":{
   "keep":{"vote_pattern","top1_unique","top1_max_vote","top1_max_vote_share","top3_jaccard","top6_jaccard","top3_union","top6_union","rank_diff_mean","rank_std_mean"}
 }
}

def project(feats,spec):
 if "keep" in spec:
  return {k:v for k,v in feats.items() if k in spec["keep"]}
 drop=spec.get("drop",set())
 return {k:v for k,v in feats.items() if k not in drop}

def budgets_by_year(vals):
 y=np.array([x[2] for x in vals],dtype=int)
 out={}
 for b in BUDGETS:
  selected=[]; hit=0
  for yy in (2022,2023,2024,2025):
   vv=sorted([x for x in vals if x[0]==yy],key=lambda x:x[3],reverse=True)
   k=max(1,int(math.ceil(len(vv)*b)))
   z=vv[:k]; selected.extend(z); hit+=sum(x[2] for x in z)
  precision=hit/len(selected); recall=hit/int(y.sum())
  out[str(int(b*100))]={
    "selected":len(selected),"blind":int(hit),"recall":float(recall),
    "precision":float(precision),"lift":float(precision/(float(y.mean()) or 1))
  }
 return out

def fit_predict(train,test,variant):
 Xtr,Xte=encode(
   [project(r["features"],VARIANTS[variant]) for r in train],
   [project(r["features"],VARIANTS[variant]) for r in test]
 )
 ytr=np.array([r["blind"] for r in train],dtype=int)
 m=make_model(seed=2600+len(train)+len(test)+len(variant))
 m.fit(Xtr,ytr)
 pred=m.predict_proba(Xte)[:,1]
 gains=m.booster_.feature_importance(importance_type="gain")
 names=m.booster_.feature_name()
 imp=sorted(
   ({"feature":n,"gain":float(g)} for n,g in zip(names,gains)),
   key=lambda x:x["gain"],reverse=True
 )[:15]
 return pred,imp,list(Xtr.columns)

paths={}
for s in a.router_year:
 y,path=s.split(":",1); paths[int(y)]=path
need={2021,2022,2023,2024,2025}
if set(paths)!=need: raise SystemExit(f"need years {sorted(need)} got {sorted(paths)}")

score=json.load(open(a.scorecard,encoding="utf-8"))
lab=labels(score)
if not need.issubset(lab): raise SystemExit(f"labels missing {sorted(need-set(lab))}")

rows=[]
for y,path in sorted(paths.items()):
 with op(path) as fh:
  for line in fh:
   if not line.strip():continue
   rec=json.loads(line); rid=str(rec.get("race_id") or "")
   if rid not in lab[y]: raise ValueError(f"label missing {y} {rid}")
   rows.append({
     "year":y,"race_id":rid,"blind":lab[y][rid],
     "features":uncertainty_features(rec,True)
   })

pooled={k:[] for k in VARIANTS}
folds=[]
for test_year in (2022,2023,2024,2025):
 train=[r for r in rows if r["year"]<test_year]
 test=[r for r in rows if r["year"]==test_year]
 y=np.array([r["blind"] for r in test],dtype=int)
 fold={"test_year":test_year,"train_years":sorted({r["year"] for r in train}),"models":{}}
 for variant in VARIANTS:
  pred,imp,cols=fit_predict(train,test,variant)
  roc=float(roc_auc_score(y,pred)); pr=float(average_precision_score(y,pred))
  vv=[]
  for r,pp in zip(test,pred):
   pooled[variant].append((test_year,r["race_id"],r["blind"],float(pp)))
   vv.append((test_year,r["race_id"],r["blind"],float(pp)))
  fold["models"][variant]={
    "roc_auc":roc,"pr_auc":pr,
    "budget10":budgets_by_year(vv)["10"],
    "top_features":imp,"feature_count":len(cols)
  }
 folds.append(fold)

aggregate={}
for variant,vals in pooled.items():
 y=np.array([x[2] for x in vals],dtype=int)
 pr=np.array([x[3] for x in vals],dtype=float)
 aggregate[variant]={
  "roc_auc":float(roc_auc_score(y,pr)),
  "pr_auc":float(average_precision_score(y,pr)),
  "budgets":budgets_by_year(vals)
 }

base=aggregate["full"]
comparison=[]
for v,m in aggregate.items():
 comparison.append({
  "variant":v,
  "roc_auc":m["roc_auc"],
  "delta_roc_vs_full":m["roc_auc"]-base["roc_auc"],
  "pr_auc":m["pr_auc"],
  "delta_pr_vs_full":m["pr_auc"]-base["pr_auc"],
  "b10_recall":m["budgets"]["10"]["recall"],
  "b10_blind":m["budgets"]["10"]["blind"],
  "delta_b10_blind_vs_full":m["budgets"]["10"]["blind"]-base["budgets"]["10"]["blind"],
  "b20_recall":m["budgets"]["20"]["recall"],
  "b20_blind":m["budgets"]["20"]["blind"],
  "delta_b20_blind_vs_full":m["budgets"]["20"]["blind"]-base["budgets"]["20"]["blind"]
 })
comparison=sorted(comparison,key=lambda x:(x["b10_blind"],x["roc_auc"]),reverse=True)

out={
 "contract":"L15_KING_ANXIETY_ABLATION_V1",
 "variants":{k:{kk:sorted(vv) if isinstance(vv,set) else vv for kk,vv in spec.items()} for k,spec in VARIANTS.items()},
 "aggregate":aggregate,
 "comparison":comparison,
 "folds":folds,
 "odds_used":False,
 "locked_years":[2026],
 "note":"Exploratory ablation. Positive delta when removing a group means that group was not helping this fixed model/config; confirm before production."
}
open(a.out,"w",encoding="utf-8").write(json.dumps(out,ensure_ascii=False,indent=2))
print("KING_ANXIETY_ABLATION_RESULT")
print(json.dumps({"comparison":comparison},ensure_ascii=False,separators=(",",":")))
