#!/usr/bin/env python3
import argparse,gzip,json,math
from collections import Counter,defaultdict

TARGET_PATTERNS=("4-2-1","5-1-1","3-3-1","3-2-2","3-2-1-1","4-1-1-1","2-2-2-1")
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
def quantile(v,q):
 z=sorted(x for x in v if x is not None)
 if not z:return None
 i=(len(z)-1)*q; lo=int(i); hi=min(lo+1,len(z)-1); f=i-lo
 return z[lo]*(1-f)+z[hi]*f

score=json.load(open(a.scorecard,encoding="utf-8"))
blind={}
for r in score.get("fold_results") or []:
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
   pattern="-".join(map(str,sorted(votes.values(),reverse=True)))
   p1=[]; g12=[]; ent=[]; leader_g=[]; t3=set()
   mv=max(votes.values()); leaders={h for h,n in votes.items() if n==mv}
   for v in ex.values():
    h=str(v["top1_horse_id"])
    p1.append(finite(v.get("top1_probability")))
    g12.append(finite(v.get("top1_top2_gap")))
    ent.append(finite(v.get("normalized_entropy")))
    if h in leaders: leader_g.append(finite(v.get("top1_top2_gap")))
    t3.update(map(str,v.get("top3_horse_ids") or []))
   rows.append({
     "year":y,"race_id":str(r["race_id"]),"blind":str(r["race_id"]) in blind[y],
     "pattern":pattern,
     "p1_mean":mean(p1),"g12_mean":mean(g12),"ent_mean":mean(ent),"leader_g12":mean(leader_g),
     "j3":finite(co.get("top3_pairwise_jaccard_mean")),"u3":len(t3)
   })

base=sum(r["blind"] for r in rows)/len(rows)
year_base={y:sum(r["blind"] for r in rows if r["year"]==y)/sum(1 for r in rows if r["year"]==y) for y in sorted(blind)}
cuts={
 "p1_q25":quantile([r["p1_mean"] for r in rows],.25),
 "g12_q25":quantile([r["g12_mean"] for r in rows],.25),
 "ent_q75":quantile([r["ent_mean"] for r in rows],.75),
 "leader_g12_q25":quantile([r["leader_g12"] for r in rows],.25),
}
for r in rows:
 flags={
  "low_p1": r["p1_mean"] is not None and r["p1_mean"]<=cuts["p1_q25"],
  "small_gap": r["g12_mean"] is not None and r["g12_mean"]<=cuts["g12_q25"],
  "high_entropy": r["ent_mean"] is not None and r["ent_mean"]>=cuts["ent_q75"],
  "low_top3_agreement": r["j3"] is not None and r["j3"]<0.65,
  "leader_small_gap": r["leader_g12"] is not None and r["leader_g12"]<=cuts["leader_g12_q25"],
 }
 r.update(flags)
 r["core_flag_count"]=sum(flags[k] for k in ("low_p1","small_gap","high_entropy","low_top3_agreement"))
 r["all_flag_count"]=sum(flags.values())

def stat(name,z):
 if not z:return None
 b=sum(r["blind"] for r in z); rate=b/len(z)
 yr={}
 pos=0; cov=0
 for y in sorted(blind):
  yz=[r for r in z if r["year"]==y]
  if yz:
   br=sum(r["blind"] for r in yz)/len(yz); yr[str(y)]={"n":len(yz),"blind_rate":br}
   if len(yz)>=10:
    cov+=1; pos+=int(br>year_base[y])
 return {
  "name":name,"races":len(z),"blind":b,"blind_rate":rate,"lift":rate/base,
  "blind_capture":b/sum(r["blind"] for r in rows),
  "positive_years":pos,"covered_years":cov,"years":yr
 }

selected=[r for r in rows if r["pattern"] in TARGET_PATTERNS]
pattern_out={}
for pat in TARGET_PATTERNS:
 pr=[r for r in selected if r["pattern"]==pat]
 detail={"overall":stat("pattern="+pat,pr),"signals":{},"core_flag_count":[]}
 for sig in ("low_p1","small_gap","high_entropy","low_top3_agreement","leader_small_gap"):
  detail["signals"][sig]={
   "on":stat(sig+"=1",[r for r in pr if r[sig]]),
   "off":stat(sig+"=0",[r for r in pr if not r[sig]])
  }
 for n in range(5):
  z=[r for r in pr if r["core_flag_count"]==n]
  if z: detail["core_flag_count"].append(stat(f"core_flags={n}",z))
 detail["at_least"]=[]
 for n in range(1,5):
  z=[r for r in pr if r["core_flag_count"]>=n]
  if z: detail["at_least"].append(stat(f"core_flags>={n}",z))
 pattern_out[pat]=detail

agg={
 "overall":stat("selected_7_patterns",selected),
 "signals":{},
 "core_flag_count":[],
 "at_least":[]
}
for sig in ("low_p1","small_gap","high_entropy","low_top3_agreement","leader_small_gap"):
 agg["signals"][sig]={
   "on":stat(sig+"=1",[r for r in selected if r[sig]]),
   "off":stat(sig+"=0",[r for r in selected if not r[sig]])
 }
for n in range(5):
 z=[r for r in selected if r["core_flag_count"]==n]
 if z: agg["core_flag_count"].append(stat(f"core_flags={n}",z))
for n in range(1,5):
 z=[r for r in selected if r["core_flag_count"]>=n]
 if z: agg["at_least"].append(stat(f"core_flags>={n}",z))

cross=[]
for sig in ("low_p1","small_gap","high_entropy","low_top3_agreement"):
 for pat in TARGET_PATTERNS:
  z=[r for r in selected if r["pattern"]==pat and r[sig]]
  if len(z)>=20: cross.append(stat(f"{pat} & {sig}",z))
cross=sorted(cross,key=lambda x:(x["lift"],x["races"]),reverse=True)

out={
 "contract":"L15_7KING_CHOICE_PATTERN_AUDIT_V3",
 "races":len(rows),"blind":sum(r["blind"] for r in rows),"blind_rate":base,
 "target_patterns":TARGET_PATTERNS,
 "cuts":cuts,
 "aggregate":agg,
 "patterns":pattern_out,
 "cross_ranked":cross
}
print("CHOICE_PATTERN_V3_RESULT")
print(json.dumps(out,ensure_ascii=False,separators=(",",":")))
