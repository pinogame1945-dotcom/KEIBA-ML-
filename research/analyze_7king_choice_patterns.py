#!/usr/bin/env python3
import argparse,gzip,json,statistics
from collections import Counter
p=argparse.ArgumentParser();p.add_argument("--router-year",action="append",required=True);p.add_argument("--scorecard",required=True);a=p.parse_args()
def op(x): return gzip.open(x,"rt",encoding="utf-8") if x.endswith(".gz") else open(x,encoding="utf-8")
def cls(r):
 d=((r.get("rescue")or{}).get("four_way_race_ids_by_topn")or{}).get("6")or{}
 return {k:set(map(str,d.get(k)or[])) for k in("both_hit","kings_only","outsider_only","both_miss")}
s=json.load(open(a.scorecard,encoding="utf-8"));blind={}
for r in s.get("fold_results")or[]:
 if r.get("status")!="success":continue
 y=int(r.get("validation_year")or 0)
 if y in (2022,2023,2024,2025) and y not in blind:
  d=cls(r);u=set().union(*d.values());blind[y]=u-(d["both_hit"]|d["kings_only"])
rows=[]
for spec in a.router_year:
 ys,path=spec.split(":",1);y=int(ys)
 for line in op(path):
  if not line.strip():continue
  r=json.loads(line);e=r["experts"];co=r["consensus"];vc=Counter(str(v["top1_horse_id"]) for v in e.values());t3=set();t6=set()
  for v in e.values():t3|=set(map(str,v["top3_horse_ids"]));t6|=set(map(str,v["top6_horse_ids"]))
  rows.append(dict(year=y,race_id=str(r["race_id"]),blind=str(r["race_id"]) in blind[y],pattern="-".join(map(str,sorted(vc.values(),reverse=True))),u1=len(vc),mv=max(vc.values()),u3=len(t3),u6=len(t6),j3=float(co["top3_pairwise_jaccard_mean"]),j6=float(co["top6_pairwise_jaccard_mean"])))
base=sum(x["blind"] for x in rows)/len(rows);B=[x for x in rows if x["blind"]];N=[x for x in rows if not x["blind"]]
def sm(k):
 def one(z):
  v=[x[k] for x in z];return{"mean":sum(v)/len(v),"median":statistics.median(v)}
 return{"blind":one(B),"normal":one(N)}
def rr(name,pred):
 z=[x for x in rows if pred(x)];b=sum(x["blind"] for x in z);q=b/len(z) if z else None
 return{"rule":name,"races":len(z),"blind":b,"blind_rate":q,"lift":q/base if q is not None else None}
rules=[rr("mv>=7",lambda x:x["mv"]>=7),rr("mv>=6",lambda x:x["mv"]>=6),rr("mv>=5",lambda x:x["mv"]>=5),rr("u1<=2",lambda x:x["u1"]<=2),rr("u1>=5",lambda x:x["u1"]>=5),rr("u3<=5",lambda x:x["u3"]<=5),rr("u3<=6",lambda x:x["u3"]<=6),rr("u6<=7",lambda x:x["u6"]<=7),rr("u6<=8",lambda x:x["u6"]<=8),rr("u6<=9",lambda x:x["u6"]<=9),rr("j6>=.70",lambda x:x["j6"]>=.70),rr("j6>=.80",lambda x:x["j6"]>=.80),rr("j6<=.45",lambda x:x["j6"]<=.45)]
pat=[]
for k,n in Counter(x["pattern"] for x in rows).most_common():
 z=[x for x in rows if x["pattern"]==k];b=sum(x["blind"] for x in z);q=b/len(z);pat.append({"pattern":k,"races":len(z),"blind":b,"blind_rate":q,"lift":q/base})
years={}
for y in sorted(blind):
 z=[x for x in rows if x["year"]==y];b=[x for x in z if x["blind"]];n=[x for x in z if not x["blind"]]
 years[str(y)]={"races":len(z),"blind":len(b),"blind_rate":len(b)/len(z),"blind_mv":sum(x["mv"] for x in b)/len(b),"normal_mv":sum(x["mv"] for x in n)/len(n),"blind_u6":sum(x["u6"] for x in b)/len(b),"normal_u6":sum(x["u6"] for x in n)/len(n),"blind_j6":sum(x["j6"] for x in b)/len(b),"normal_j6":sum(x["j6"] for x in n)/len(n)}
out={"races":len(rows),"blind":len(B),"blind_rate":base,"metrics":{k:sm(k) for k in("u1","mv","u3","u6","j3","j6")},"patterns":pat,"rules":rules,"years":years}
print("CHOICE_PATTERN_RESULT");print(json.dumps(out,ensure_ascii=False,separators=(",",":")))
