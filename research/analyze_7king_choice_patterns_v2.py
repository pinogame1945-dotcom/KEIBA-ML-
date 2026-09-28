#!/usr/bin/env python3
import argparse,gzip,json,math,statistics
from collections import Counter,defaultdict

p=argparse.ArgumentParser()
p.add_argument("--router-year",action="append",required=True)
p.add_argument("--scorecard",required=True)
a=p.parse_args()

def op(x): return gzip.open(x,"rt",encoding="utf-8") if x.endswith(".gz") else open(x,"rt",encoding="utf-8")
def cls(r):
 d=((r.get("rescue") or {}).get("four_way_race_ids_by_topn") or {}).get("6") or {}
 return {k:set(map(str,d.get(k) or [])) for k in ("both_hit","kings_only","outsider_only","both_miss")}
def finite(x):
 try:
  v=float(x); return v if math.isfinite(v) else None
 except: return None
def mean(v):
 z=[x for x in v if x is not None]; return sum(z)/len(z) if z else None
def pstdev(v):
 z=[x for x in v if x is not None]; return statistics.pstdev(z) if len(z)>1 else 0.0 if z else None
def quantile(v,q):
 z=sorted(x for x in v if x is not None)
 if not z:return None
 i=(len(z)-1)*q; lo=int(i); hi=min(lo+1,len(z)-1); f=i-lo
 return z[lo]*(1-f)+z[hi]*f
def qlabel(x,cuts):
 if x is None:return "NA"
 if x<=cuts[0]:return "Q1"
 if x<=cuts[1]:return "Q2"
 if x<=cuts[2]:return "Q3"
 return "Q4"

s=json.load(open(a.scorecard,encoding="utf-8"))
blind={}
for r in s.get("fold_results") or []:
 if r.get("status")!="success": continue
 y=int(r.get("validation_year") or 0)
 if y in (2022,2023,2024,2025) and y not in blind:
  d=cls(r); u=set().union(*d.values()); blind[y]=u-(d["both_hit"]|d["kings_only"])

rows=[]
for spec in a.router_year:
 ys,path=spec.split(":",1); y=int(ys)
 with op(path) as fh:
  for line in fh:
   if not line.strip(): continue
   r=json.loads(line); ex=r["experts"]; co=r["consensus"]
   votes=Counter(str(v["top1_horse_id"]) for v in ex.values())
   mv=max(votes.values()); leaders={h for h,n in votes.items() if n==mv}
   t3=set(); t6=set(); p1=[]; g12=[]; g13=[]; ent=[]; lp=[]; lg=[]; mp=[]; mg=[]
   for v in ex.values():
    h=str(v["top1_horse_id"]); t3|=set(map(str,v.get("top3_horse_ids") or [])); t6|=set(map(str,v.get("top6_horse_ids") or []))
    vp=finite(v.get("top1_probability")); vg=finite(v.get("top1_top2_gap")); vg13=finite(v.get("top1_top3_gap")); ve=finite(v.get("normalized_entropy"))
    p1.append(vp); g12.append(vg); g13.append(vg13); ent.append(ve)
    if h in leaders: lp.append(vp); lg.append(vg)
    else: mp.append(vp); mg.append(vg)
   rows.append({
    "year":y,"race_id":str(r["race_id"]),"blind":str(r["race_id"]) in blind[y],
    "pattern":"-".join(map(str,sorted(votes.values(),reverse=True))),
    "u1":len(votes),"mv":mv,"u3":len(t3),"u6":len(t6),
    "j3":finite(co.get("top3_pairwise_jaccard_mean")),"j6":finite(co.get("top6_pairwise_jaccard_mean")),
    "p1_mean":mean(p1),"p1_std":pstdev(p1),"g12_mean":mean(g12),"g12_std":pstdev(g12),
    "g13_mean":mean(g13),"ent_mean":mean(ent),"ent_std":pstdev(ent),
    "leader_p1":mean(lp),"leader_g12":mean(lg),"minority_p1":mean(mp),"minority_g12":mean(mg),
   })

base=sum(r["blind"] for r in rows)/len(rows)
year_base={y:sum(r["blind"] for r in rows if r["year"]==y)/sum(1 for r in rows if r["year"]==y) for y in sorted(blind)}

cuts={}
for k in ("p1_mean","p1_std","g12_mean","g12_std","g13_mean","ent_mean","ent_std","leader_g12"):
 vals=[r[k] for r in rows if r[k] is not None]; cuts[k]=[quantile(vals,.25),quantile(vals,.50),quantile(vals,.75)]
for r in rows:
 r["u3_bucket"]="u3<=4" if r["u3"]<=4 else "u3=5" if r["u3"]==5 else "u3>=6"
 r["j3_bucket"]="j3<.65" if r["j3"]<.65 else "j3<.75" if r["j3"]<.75 else "j3>=.75"
 for k in cuts:r[k+"_q"]=qlabel(r[k],cuts[k])

def stat(name,z):
 if not z:return None
 b=sum(r["blind"] for r in z); rate=b/len(z)
 yr={}
 pos=0; covered=0
 for y in sorted(blind):
  yz=[r for r in z if r["year"]==y]
  if yz:
   br=sum(r["blind"] for r in yz)/len(yz); yr[str(y)]={"n":len(yz),"blind_rate":br}
   if len(yz)>=10:
    covered+=1; pos+=int(br>year_base[y])
 return {"name":name,"races":len(z),"blind":b,"blind_rate":rate,"lift":rate/base,"blind_capture":b/sum(r["blind"] for r in rows),"positive_years":pos,"covered_years":covered,"years":yr}

individual=[]
features=("pattern","u3_bucket","j3_bucket","p1_mean_q","p1_std_q","g12_mean_q","g12_std_q","ent_mean_q","ent_std_q","leader_g12_q")
for f in features:
 for v in sorted({r[f] for r in rows}):
  individual.append(stat(f+"="+str(v),[r for r in rows if r[f]==v]))

combos=[]
common_patterns={p for p,n in Counter(r["pattern"] for r in rows).items() if n>=100}
for pat in sorted(common_patterns):
 for f in ("u3_bucket","j3_bucket","p1_mean_q","g12_mean_q","g12_std_q","ent_mean_q","ent_std_q","leader_g12_q"):
  for v in sorted({r[f] for r in rows if r["pattern"]==pat}):
   z=[r for r in rows if r["pattern"]==pat and r[f]==v]
   if len(z)>=50: combos.append(stat("pattern="+pat+" & "+f+"="+str(v),z))
 for ub in sorted({r["u3_bucket"] for r in rows if r["pattern"]==pat}):
  for gq in ("Q1","Q2","Q3","Q4"):
   z=[r for r in rows if r["pattern"]==pat and r["u3_bucket"]==ub and r["g12_mean_q"]==gq]
   if len(z)>=40: combos.append(stat("pattern="+pat+" & "+ub+" & g12="+gq,z))

spot={}
for pat in ("3-2-2","4-2-1","3-3-1","3-2-1-1","7","5-2"):
 pr=[r for r in rows if r["pattern"]==pat]
 if not pr:continue
 cells=[]
 for ub in sorted({r["u3_bucket"] for r in pr}):
  for gq in ("Q1","Q2","Q3","Q4"):
   for eq in ("Q1","Q2","Q3","Q4"):
    z=[r for r in pr if r["u3_bucket"]==ub and r["g12_mean_q"]==gq and r["ent_mean_q"]==eq]
    if len(z)>=15:cells.append(stat(ub+" & g12="+gq+" & ent="+eq,z))
 cells=sorted(cells,key=lambda x:(x["lift"],x["races"]),reverse=True)
 spot[pat]={"overall":stat("pattern="+pat,pr),"top_cells":cells[:12]}

stable=[x for x in combos if x and x["races"]>=80 and x["covered_years"]>=3 and x["positive_years"]>=max(2,x["covered_years"]-1)]
stable=sorted(stable,key=lambda x:(x["lift"],x["races"]),reverse=True)[:30]
high_support=sorted([x for x in combos if x and x["races"]>=150],key=lambda x:(x["lift"],x["races"]),reverse=True)[:30]

out={
 "contract":"L15_7KING_CHOICE_PATTERN_AUDIT_V2",
 "races":len(rows),"blind":sum(r["blind"] for r in rows),"blind_rate":base,
 "year_baseline":year_base,
 "quartile_cuts":cuts,
 "individual":individual,
 "stable_high_lift_combos":stable,
 "high_support_combos":high_support,
 "pattern_spotlight":spot
}
print("CHOICE_PATTERN_V2_RESULT")
print(json.dumps(out,ensure_ascii=False,separators=(",",":")))
