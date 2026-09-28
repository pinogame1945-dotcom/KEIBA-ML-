#!/usr/bin/env python3
import argparse,gzip,json,math,statistics
from collections import Counter,defaultdict
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from train_l15_outsider_rescue_router_v1 import router_features,make_model

BUDGETS=(0.05,0.10,0.15,0.20,0.30,0.40,0.50)
p=argparse.ArgumentParser()
p.add_argument("--router-year",action="append",required=True)
p.add_argument("--scorecard",required=True)
p.add_argument("--out",required=True)
a=p.parse_args()

def op(x): return gzip.open(x,"rt",encoding="utf-8") if x.endswith(".gz") else open(x,"rt",encoding="utf-8")
def finite(x):
 try:
  v=float(x); return v if math.isfinite(v) else None
 except: return None
def mean(v):
 z=[x for x in v if x is not None]; return sum(z)/len(z) if z else None
def pstdev(v):
 z=[x for x in v if x is not None]
 return statistics.pstdev(z) if len(z)>1 else 0.0 if z else None
def cls(r):
 d=((r.get("rescue") or {}).get("four_way_race_ids_by_topn") or {}).get("6") or {}
 return {k:set(map(str,d.get(k) or [])) for k in ("both_hit","kings_only","outsider_only","both_miss")}
def labels(score):
 out={}
 for r in score.get("fold_results") or []:
  if r.get("status")!="success": continue
  y=int(r.get("validation_year") or 0)
  if y not in (2021,2022,2023,2024,2025) or y in out: continue
  d=cls(r); u=set().union(*d.values()); out[y]={rid:int(rid not in (d["both_hit"]|d["kings_only"])) for rid in u}
 return out
def uncertainty_features(rec,with_pattern):
 ex=rec.get("experts") or {}; co=rec.get("consensus") or {}
 if len(ex)!=7: raise ValueError("expected 7 experts")
 p1=[];p2=[];p3=[];g12=[];g13=[];ent=[];m3=[];m6=[];t3=[];t6=[];votes=Counter()
 for v in ex.values():
  p1.append(finite(v.get("top1_probability")));p2.append(finite(v.get("top2_probability")));p3.append(finite(v.get("top3_probability")))
  g12.append(finite(v.get("top1_top2_gap")));g13.append(finite(v.get("top1_top3_gap")));ent.append(finite(v.get("normalized_entropy")))
  m3.append(finite(v.get("top3_probability_mass")));m6.append(finite(v.get("top6_probability_mass")))
  t3.append(set(map(str,v.get("top3_horse_ids") or [])));t6.append(set(map(str,v.get("top6_horse_ids") or [])))
  votes[str(v.get("top1_horse_id"))]+=1
 u3=set().union(*t3);u6=set().union(*t6)
 f={
  "p1_mean":mean(p1),"p1_std":pstdev(p1),"p1_min":min(x for x in p1 if x is not None),"p1_max":max(x for x in p1 if x is not None),
  "g12_mean":mean(g12),"g12_std":pstdev(g12),"g12_min":min(x for x in g12 if x is not None),"g12_max":max(x for x in g12 if x is not None),
  "g13_mean":mean(g13),"g13_std":pstdev(g13),
  "entropy_mean":mean(ent),"entropy_std":pstdev(ent),
  "top3_mass_mean":mean(m3),"top6_mass_mean":mean(m6),
  "top3_jaccard":finite(co.get("top3_pairwise_jaccard_mean")),
  "top6_jaccard":finite(co.get("top6_pairwise_jaccard_mean")),
  "rank_diff_mean":finite(co.get("pairwise_rank_abs_diff_mean")),
  "rank_std_mean":finite(co.get("horse_rank_std_mean")),
  "prob_std_mean":finite(co.get("horse_probability_std_mean")),
  "prob_std_max":finite(co.get("horse_probability_std_max")),
  "top3_union":float(len(u3)),"top6_union":float(len(u6))
 }
 if with_pattern:
  f["vote_pattern"]="-".join(map(str,sorted(votes.values(),reverse=True)))
  f["top1_unique"]=float(len(votes))
  f["top1_max_vote"]=float(max(votes.values()))
  f["top1_max_vote_share"]=float(max(votes.values())/7.0)
 return f
def encode(train,test):
 tr=pd.DataFrame(train); te=pd.DataFrame(test)
 cats=[c for c in tr.columns if not pd.api.types.is_numeric_dtype(tr[c])]
 tr=pd.get_dummies(tr,columns=cats,dummy_na=True,dtype=float)
 cats2=[c for c in te.columns if not pd.api.types.is_numeric_dtype(te[c])]
 te=pd.get_dummies(te,columns=cats2,dummy_na=True,dtype=float)
 te=te.reindex(columns=tr.columns,fill_value=0.0)
 return tr.replace([np.inf,-np.inf],np.nan).fillna(-999.0),te.replace([np.inf,-np.inf],np.nan).fillna(-999.0)
def budgets(y,p):
 order=np.argsort(-p); total=int(y.sum()); out={}
 for b in BUDGETS:
  k=max(1,int(math.ceil(len(y)*b))); idx=order[:k]; hit=int(y[idx].sum())
  precision=hit/k; recall=hit/total if total else None
  out[str(int(b*100))]={"selected":k,"blind":hit,"recall":recall,"precision":precision,"lift":precision/(float(y.mean()) or 1)}
 return out
def metrics(y,p):
 return {"roc_auc":float(roc_auc_score(y,p)),"pr_auc":float(average_precision_score(y,p)),"prevalence":float(y.mean()),"budgets":budgets(y,p)}
def fit(train,test,key):
 Xtr,Xte=encode([r[key] for r in train],[r[key] for r in test]); ytr=np.array([r["blind"] for r in train],dtype=int)
 m=make_model(seed=1945+len(train)+len(test));m.fit(Xtr,ytr);pred=m.predict_proba(Xte)[:,1]
 gains=m.booster_.feature_importance(importance_type="gain"); names=m.booster_.feature_name()
 imp=sorted(({"feature":n,"gain":float(g)} for n,g in zip(names,gains)),key=lambda x:x["gain"],reverse=True)[:20]
 return pred,imp

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
    "uncertainty":uncertainty_features(rec,False),
    "uncertainty_pattern":uncertainty_features(rec,True),
    "full_gate":router_features(rec)})

folds=[]; pooled={k:[] for k in ("uncertainty","uncertainty_pattern","full_gate")}
for test_year in (2022,2023,2024,2025):
 train=[r for r in rows if r["year"]<test_year]; test=[r for r in rows if r["year"]==test_year]
 fold={"test_year":test_year,"train_years":sorted({r["year"] for r in train}),"races":len(test),"blind":sum(r["blind"] for r in test),"models":{}}
 y=np.array([r["blind"] for r in test],dtype=int)
 for key in ("uncertainty","uncertainty_pattern","full_gate"):
  pred,imp=fit(train,test,key)
  fold["models"][key]={"metrics":metrics(y,pred),"top_features":imp}
  for r,pp in zip(test,pred): pooled[key].append((test_year,r["race_id"],r["blind"],float(pp)))
 folds.append(fold)

agg={}
for key,vals in pooled.items():
 y=np.array([x[2] for x in vals],dtype=int); p=np.array([x[3] for x in vals],dtype=float)
 per_budget={}
 for b in BUDGETS:
  sel=[];hit=0;totblind=int(y.sum())
  for yy in (2022,2023,2024,2025):
   vv=[x for x in vals if x[0]==yy]; vv=sorted(vv,key=lambda x:x[3],reverse=True)
   k=max(1,int(math.ceil(len(vv)*b))); z=vv[:k];sel+=z;hit+=sum(x[2] for x in z)
  precision=hit/len(sel); recall=hit/totblind
  per_budget[str(int(b*100))]={"selected":len(sel),"blind":hit,"recall":recall,"precision":precision,"lift":precision/(float(y.mean()) or 1)}
 agg[key]={"roc_auc":float(roc_auc_score(y,p)),"pr_auc":float(average_precision_score(y,p)),"prevalence":float(y.mean()),"budgets":per_budget}

out={"contract":"L15_KING_ANXIETY_GATE_V1","folds":folds,"aggregate":agg,
 "score_definition":"continuous LightGBM blind-spot risk score from pre-race seven-king outputs; raw score is ranking-oriented, not calibrated probability",
 "odds_used":False,"locked_years":[2026]}
open(a.out,"w",encoding="utf-8").write(json.dumps(out,ensure_ascii=False,indent=2))
print("KING_ANXIETY_GATE_RESULT")
print(json.dumps(out,ensure_ascii=False,separators=(",",":")))
