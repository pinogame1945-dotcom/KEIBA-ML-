#!/usr/bin/env python3
import argparse,gzip,json,math
from collections import Counter

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
   p1=[]; g12=[]; ent=[]; t3=set()
   for v in ex.values():
    p1.append(finite(v.get("top1_probability")))
    g12.append(finite(v.get("top1_top2_gap")))
    ent.append(finite(v.get("normalized_entropy")))
    t3.update(map(str,v.get("top3_horse_ids") or []))
   rows.append({
    "year":y,"race_id":str(r["race_id"]),"blind":str(r["race_id"]) in blind[y],
    "pattern":"-".join(map(str,sorted(votes.values(),reverse=True))),
    "p1_mean":mean(p1),"g12_mean":mean(g12),"ent_mean":mean(ent),
    "j3":finite(co.get("top3_pairwise_jaccard_mean"))
   })

base=sum(r["blind"] for r in rows)/len(rows)
cuts={
 "p1_q25":quantile([r["p1_mean"] for r in rows],.25),
 "g12_q25":quantile([r["g12_mean"] for r in rows],.25),
 "ent_q75":quantile([r["ent_mean"] for r in rows],.75)
}
for r in rows:
 r["low_p1"]=r["p1_mean"] is not None and r["p1_mean"]<=cuts["p1_q25"]
 r["small_gap"]=r["g12_mean"] is not None and r["g12_mean"]<=cuts["g12_q25"]
 r["high_entropy"]=r["ent_mean"] is not None and r["ent_mean"]>=cuts["ent_q75"]
 r["low_top3_agreement"]=r["j3"] is not None and r["j3"]<0.65
 r["uncertainty_count"]=sum(r[k] for k in ("low_p1","small_gap","high_entropy","low_top3_agreement"))
 r["uncertain3"]=r["uncertainty_count"]>=3
 r["uncertain_any"]=r["uncertainty_count"]>=1

year_base={y:sum(r["blind"] for r in rows if r["year"]==y)/sum(1 for r in rows if r["year"]==y) for y in sorted(blind)}
def stat(z):
 if not z:return None
 b=sum(r["blind"] for r in z); rate=b/len(z)
 yrs={}
 for y in sorted(blind):
  yz=[r for r in z if r["year"]==y]
  if yz:
   yrs[str(y)]={"n":len(yz),"blind_rate":sum(r["blind"] for r in yz)/len(yz)}
 return {"races":len(z),"blind":b,"blind_rate":rate,"lift":rate/base,"years":yrs}

patterns=[]
for pat,n in Counter(r["pattern"] for r in rows).most_common():
 pr=[r for r in rows if r["pattern"]==pat]
 rec={"pattern":pat,"overall":stat(pr),"signals":{},"uncertainty_count":[]}
 for sig in ("low_p1","small_gap","high_entropy","low_top3_agreement","uncertain3"):
  on=[r for r in pr if r[sig]]; off=[r for r in pr if not r[sig]]
  rec["signals"][sig]={"on":stat(on),"off":stat(off)}
 for nflag in range(5):
  z=[r for r in pr if r["uncertainty_count"]==nflag]
  if z: rec["uncertainty_count"].append({"count":nflag,**stat(z)})
 patterns.append(rec)

# compact comparison table with uncertainty effect
compact=[]
for rec in patterns:
 o=rec["overall"]; u=rec["signals"]["uncertain3"]["on"]; c=rec["signals"]["uncertain3"]["off"]
 compact.append({
  "pattern":rec["pattern"],"races":o["races"],"overall_rate":o["blind_rate"],
  "uncertain3_races":u["races"] if u else 0,"uncertain3_rate":u["blind_rate"] if u else None,
  "calm_races":c["races"] if c else 0,"calm_rate":c["blind_rate"] if c else None,
  "delta":((u["blind_rate"]-c["blind_rate"]) if u and c else None),
  "low_p1_rate":rec["signals"]["low_p1"]["on"]["blind_rate"] if rec["signals"]["low_p1"]["on"] else None,
  "small_gap_rate":rec["signals"]["small_gap"]["on"]["blind_rate"] if rec["signals"]["small_gap"]["on"] else None,
  "high_entropy_rate":rec["signals"]["high_entropy"]["on"]["blind_rate"] if rec["signals"]["high_entropy"]["on"] else None,
  "low_top3_rate":rec["signals"]["low_top3_agreement"]["on"]["blind_rate"] if rec["signals"]["low_top3_agreement"]["on"] else None
 })
out={
 "contract":"L15_7KING_CHOICE_PATTERN_AUDIT_V4",
 "races":len(rows),"blind":sum(r["blind"] for r in rows),"blind_rate":base,
 "cuts":cuts,"patterns":patterns,"compact":compact
}
print("CHOICE_PATTERN_V4_RESULT")
print(json.dumps(out,ensure_ascii=False,separators=(",",":")))
