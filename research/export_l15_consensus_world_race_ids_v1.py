#!/usr/bin/env python3
import argparse,json,math
import numpy as np
from train_l15_king_anxiety_gate_v1 import (
    uncertainty_features, labels, op, encode, make_model
)

p=argparse.ArgumentParser()
p.add_argument("--router-year",action="append",required=True)
p.add_argument("--scorecard",required=True)
p.add_argument("--out",required=True)
a=p.parse_args()

BUDGETS=(0.05,0.10,0.15,0.20,0.30)
CONSENSUS_KEEP={
 "top3_jaccard","top6_jaccard","top3_union","top6_union",
 "rank_diff_mean","rank_std_mean","prob_std_mean","prob_std_max",
 "top3_mass_mean","top6_mass_mean","entropy_mean","entropy_std",
 "vote_pattern","top1_unique","top1_max_vote","top1_max_vote_share"
}
def consensus_world(rec):
 f=uncertainty_features(rec,True)
 return {k:v for k,v in f.items() if k in CONSENSUS_KEEP}

def fit_predict(train,test):
 Xtr,Xte=encode([r["features"] for r in train],[r["features"] for r in test])
 ytr=np.array([r["blind"] for r in train],dtype=int)
 m=make_model(seed=1945+len(train)+len(test))
 m.fit(Xtr,ytr)
 return m.predict_proba(Xte)[:,1]

paths={}
for s in a.router_year:
 y,path=s.split(":",1); paths[int(y)]=path
need={2021,2022,2023,2024,2025}
if set(paths)!=need: raise SystemExit(f"need years {sorted(need)} got {sorted(paths)}")
score=json.load(open(a.scorecard,encoding="utf-8")); lab=labels(score)

rows=[]
for y,path in sorted(paths.items()):
 with op(path) as fh:
  for line in fh:
   if not line.strip(): continue
   rec=json.loads(line); rid=str(rec.get("race_id") or "")
   if rid not in lab[y]: raise ValueError(f"label missing {y} {rid}")
   rows.append({"year":y,"race_id":rid,"blind":lab[y][rid],"features":consensus_world(rec)})

out={"contract":"L15_CONSENSUS_WORLD_RACE_IDS_V1","budgets":{},"all_scored":[]}
for test_year in (2022,2023,2024,2025):
 train=[r for r in rows if r["year"]<test_year]
 test=[r for r in rows if r["year"]==test_year]
 pred=fit_predict(train,test)
 scored=[{"race_id":r["race_id"],"year":test_year,"blind":bool(r["blind"]),"score":float(pp)}
         for r,pp in zip(test,pred)]
 scored.sort(key=lambda x:x["score"],reverse=True)
 out["all_scored"].extend(scored)
 for b in BUDGETS:
  k=max(1,int(math.ceil(len(scored)*b)))
  z=scored[:k]
  key=str(int(b*100))
  out["budgets"].setdefault(key,{"selected":[],"caught_blind":[]})
  out["budgets"][key]["selected"].extend(z)
  out["budgets"][key]["caught_blind"].extend([x for x in z if x["blind"]])

for key,v in out["budgets"].items():
 v["selected_count"]=len(v["selected"])
 v["caught_blind_count"]=len(v["caught_blind"])
 v["selected_race_ids"]=[x["race_id"] for x in v["selected"]]
 v["caught_blind_race_ids"]=[x["race_id"] for x in v["caught_blind"]]

open(a.out,"w",encoding="utf-8").write(json.dumps(out,ensure_ascii=False,indent=2))
print("CONSENSUS_WORLD_RACE_IDS_RESULT")
print(json.dumps({k:{
 "selected_count":v["selected_count"],
 "caught_blind_count":v["caught_blind_count"]
} for k,v in out["budgets"].items()},ensure_ascii=False,separators=(",",":")))
