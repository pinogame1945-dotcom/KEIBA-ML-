#!/usr/bin/env python3
import argparse,gzip,json,math
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

# Baseline is V1 with all margin features removed.
BASE_DROP={"g12_mean","g12_std","g12_min","g12_max","g13_mean","g13_std"}

VARIANTS={
 "baseline_no_margin":{"drop":set()},
 "drop_top3_jaccard":{"drop":{"top3_jaccard"}},
 "drop_top6_jaccard":{"drop":{"top6_jaccard"}},
 "drop_both_jaccard":{"drop":{"top3_jaccard","top6_jaccard"}},
 "drop_union_sizes":{"drop":{"top3_union","top6_union"}},
 "drop_rank_disagreement":{"drop":{"rank_diff_mean","rank_std_mean"}},
 "drop_prob_disagreement":{"drop":{"p1_std","prob_std_mean","prob_std_max"}},
 "drop_top3_mass":{"drop":{"top3_mass_mean"}},
 "drop_top6_mass":{"drop":{"top6_mass_mean"}},
 "drop_both_mass":{"drop":{"top3_mass_mean","top6_mass_mean"}},
 "drop_p1_level":{"drop":{"p1_mean","p1_min","p1_max"}},
 "drop_entropy":{"drop":{"entropy_mean","entropy_std"}},
 "drop_vote_pattern":{"drop":{"vote_pattern","top1_unique","top1_max_vote","top1_max_vote_share"}},
 "drop_top3_world":{"drop":{"top3_jaccard","top3_union","top3_mass_mean"}},
 "drop_top6_world":{"drop":{"top6_jaccard","top6_union","top6_mass_mean"}},
 "consensus_world_only":{"keep":{
    "top3_jaccard","top6_jaccard","top3_union","top6_union",
    "rank_diff_mean","rank_std_mean","prob_std_mean","prob_std_max",
    "top3_mass_mean","top6_mass_mean","entropy_mean","entropy_std",
    "vote_pattern","top1_unique","top1_max_vote","top1_max_vote_share"
 }},
 "probability_world_only":{"keep":{
    "p1_mean","p1_std","p1_min","p1_max",
    "top3_mass_mean","top6_mass_mean","prob_std_mean","prob_std_max",
    "entropy_mean","entropy_std",
    "vote_pattern","top1_unique","top1_max_vote","top1_max_vote_share"
 }}
}

def base_project(feats):
 return {k:v for k,v in feats.items() if k not in BASE_DROP}

def project(feats,spec):
 f=base_project(feats)
 if "keep" in spec:
  return {k:v for k,v in f.items() if k in spec["keep"]}
 return {k:v for k,v in f.items() if k not in spec.get("drop",set())}

def budget_by_year(vals,b):
 y=np.array([x[2] for x in vals],dtype=int)
 chosen=[]; hit=0
 for yy in (2022,2023,2024,2025):
  vv=sorted([x for x in vals if x[0]==yy],key=lambda x:x[3],reverse=True)
  k=max(1,int(math.ceil(len(vv)*b))); z=vv[:k]
  chosen.extend(z); hit+=sum(x[2] for x in z)
 precision=hit/len(chosen); recall=hit/int(y.sum())
 return {"selected":len(chosen),"blind":int(hit),"recall":float(recall),
         "precision":float(precision),"lift":float(precision/(float(y.mean()) or 1))}

def fit_predict(train,test,variant):
 Xtr,Xte=encode(
   [project(r["features"],VARIANTS[variant]) for r in train],
   [project(r["features"],VARIANTS[variant]) for r in test]
 )
 ytr=np.array([r["blind"] for r in train],dtype=int)
 m=make_model(seed=1945+len(train)+len(test))
 m.fit(Xtr,ytr)
 pred=m.predict_proba(Xte)[:,1]
 gains=m.booster_.feature_importance(importance_type="gain")
 names=m.booster_.feature_name()
 imp=sorted(({"feature":n,"gain":float(g)} for n,g in zip(names,gains)),
            key=lambda x:x["gain"],reverse=True)[:20]
 return pred,imp,len(Xtr.columns)

paths={}
for s in a.router_year:
 y,path=s.split(":",1);paths[int(y)]=path
need={2021,2022,2023,2024,2025}
if set(paths)!=need: raise SystemExit(f"need years {sorted(need)} got {sorted(paths)}")

score=json.load(open(a.scorecard,encoding="utf-8")); lab=labels(score)
if not need.issubset(lab): raise SystemExit(f"labels missing {sorted(need-set(lab))}")

rows=[]
for y,path in sorted(paths.items()):
 with op(path) as fh:
  for line in fh:
   if not line.strip():continue
   rec=json.loads(line); rid=str(rec.get("race_id") or "")
   if rid not in lab[y]: raise ValueError(f"label missing {y} {rid}")
   rows.append({"year":y,"race_id":rid,"blind":lab[y][rid],
                "features":uncertainty_features(rec,True)})

pooled={k:[] for k in VARIANTS}; folds=[]
for test_year in (2022,2023,2024,2025):
 train=[r for r in rows if r["year"]<test_year]
 test=[r for r in rows if r["year"]==test_year]
 y=np.array([r["blind"] for r in test],dtype=int)
 fold={"test_year":test_year,"models":{}}
 for variant in VARIANTS:
  pred,imp,fc=fit_predict(train,test,variant)
  vv=[]
  for r,pp in zip(test,pred):
   tup=(test_year,r["race_id"],r["blind"],float(pp))
   pooled[variant].append(tup); vv.append(tup)
  fold["models"][variant]={
    "roc_auc":float(roc_auc_score(y,pred)),
    "pr_auc":float(average_precision_score(y,pred)),
    "b10":budget_by_year(vv,0.10),
    "b20":budget_by_year(vv,0.20),
    "feature_count":fc,
    "top_features":imp
  }
 folds.append(fold)

aggregate={}
for variant,vals in pooled.items():
 y=np.array([x[2] for x in vals],dtype=int)
 pred=np.array([x[3] for x in vals],dtype=float)
 aggregate[variant]={
   "roc_auc":float(roc_auc_score(y,pred)),
   "pr_auc":float(average_precision_score(y,pred)),
   "b05":budget_by_year(vals,0.05),
   "b10":budget_by_year(vals,0.10),
   "b15":budget_by_year(vals,0.15),
   "b20":budget_by_year(vals,0.20),
   "b30":budget_by_year(vals,0.30)
 }

base=aggregate["baseline_no_margin"]
comparison=[]
for v,m in aggregate.items():
 comparison.append({
   "variant":v,
   "roc_auc":m["roc_auc"],"d_roc":m["roc_auc"]-base["roc_auc"],
   "pr_auc":m["pr_auc"],"d_pr":m["pr_auc"]-base["pr_auc"],
   "b10_blind":m["b10"]["blind"],"d_b10":m["b10"]["blind"]-base["b10"]["blind"],
   "b20_blind":m["b20"]["blind"],"d_b20":m["b20"]["blind"]-base["b20"]["blind"]
 })
comparison=sorted(comparison,key=lambda x:(x["b10_blind"],x["roc_auc"]),reverse=True)

out={
 "contract":"L15_KING_ANXIETY_ABLATION_V2",
 "baseline":"baseline_no_margin",
 "base_drop":sorted(BASE_DROP),
 "variants":{k:{kk:sorted(vv) if isinstance(vv,set) else vv for kk,vv in spec.items()} for k,spec in VARIANTS.items()},
 "aggregate":aggregate,"comparison":comparison,"folds":folds,
 "odds_used":False,"locked_years":[2026],
 "note":"Fixed-seed walk-forward fine ablation around no-margin baseline."
}
open(a.out,"w",encoding="utf-8").write(json.dumps(out,ensure_ascii=False,indent=2))
print("KING_ANXIETY_ABLATION_V2_RESULT")
print(json.dumps({"comparison":comparison},ensure_ascii=False,separators=(",",":")))
