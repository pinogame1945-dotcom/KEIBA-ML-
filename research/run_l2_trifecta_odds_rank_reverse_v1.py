#!/usr/bin/env python3
import argparse,csv,gzip,json,math
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from build_l2_bet_kings_dataset_v1 import decode_odds,payout_map

YEARS=(2021,2022,2023,2024,2025)
LAYERS=("SHORT","MIDDLE","DEEP")
NUMERIC=(
 "field_size","distance_m","ticket_count","top1_odds","top10_odds","top30_odds","top100_odds",
 "median_odds","p90_odds","top10_to_top1","top30_to_top1","median_to_top1",
 "top1_share","top10_share","top30_share","hhi","entropy_norm","effective_tickets",
 "win_top1_odds","win_top2_odds","win_top3_odds","win_top2_to_top1","win_top1_share","win_top3_share"
)
CATEGORICAL=("year","surface","discipline","venue_code","track_condition","weather","race_class","grade",
             "distance_bin","field_size_bin","race_month")

def parse_args():
 p=argparse.ArgumentParser()
 p.add_argument("--backfill-root",required=True)
 p.add_argument("--out-dir",required=True)
 return p.parse_args()

def read_gz(path):
 with gzip.open(path,"rt",encoding="utf-8") as f:
  for line in f:
   if line.strip(): yield json.loads(line)

def finite(v,default=None):
 try:
  if isinstance(v,str): v=v.replace(",","").strip()
  x=float(v)
  return x if math.isfinite(x) else default
 except (TypeError,ValueError): return default

def iint(v,default=0):
 x=finite(v)
 return int(x) if x is not None else default

def kth(vals,k):
 if not vals:return None
 return float(vals[min(k,len(vals))-1])

def qtile(vals,q):
 return float(np.quantile(np.asarray(vals,dtype=float),q)) if vals else None

def layer(rank):
 if rank<=30:return "SHORT"
 if rank<=120:return "MIDDLE"
 return "DEEP"

def distance_bin(x):
 x=iint(x)
 if x<=0:return "UNKNOWN"
 if x<=1200:return "D_LE1200"
 if x<=1600:return "D_1300_1600"
 if x<=2000:return "D_1700_2000"
 if x<=2400:return "D_2100_2400"
 return "D_2500P"

def field_bin(x):
 x=iint(x)
 if x<=0:return "UNKNOWN"
 if x<=10:return "F_LE10"
 if x<=13:return "F_11_13"
 if x<=16:return "F_14_16"
 return "F_17_18"

def market_shape(odds_map):
 tri=sorted([(tuple(k[1]),float(v)) for k,v in odds_map.items() if k[0]=="TRIFECTA" and float(v)>0],
            key=lambda x:(x[1],x[0]))
 if not tri:return None
 horses=sorted({n for nums,_ in tri for n in nums})
 n_h=len(horses); expected=n_h*(n_h-1)*(n_h-2)
 if expected<=0:return None
 vals=[o for _,o in tri]
 inv=1/np.asarray(vals,dtype=float)
 probs=inv/inv.sum()
 hhi=float((probs*probs).sum())
 ent=float(-(probs*np.log(np.maximum(probs,1e-300))).sum())
 win=sorted(float(v) for k,v in odds_map.items() if k[0]=="WIN" and float(v)>0)
 win_inv=np.asarray([1/x for x in win],dtype=float) if win else np.asarray([],dtype=float)
 win_q=win_inv/win_inv.sum() if len(win_inv) and win_inv.sum()>0 else np.asarray([],dtype=float)
 return {
  "tri":tri,"field_size":n_h,"ticket_count":len(tri),"expected_tickets":expected,
  "completeness":len(tri)/expected,
  "top1_odds":kth(vals,1),"top10_odds":kth(vals,10),"top30_odds":kth(vals,30),
  "top100_odds":kth(vals,100),"median_odds":qtile(vals,.5),"p90_odds":qtile(vals,.9),
  "top10_to_top1":kth(vals,10)/vals[0],"top30_to_top1":kth(vals,30)/vals[0],
  "median_to_top1":qtile(vals,.5)/vals[0],
  "top1_share":float(probs[:1].sum()),"top10_share":float(probs[:min(10,len(probs))].sum()),
  "top30_share":float(probs[:min(30,len(probs))].sum()),"hhi":hhi,
  "entropy_norm":ent/math.log(len(probs)) if len(probs)>1 else 0.0,
  "effective_tickets":1/hhi if hhi>0 else None,
  "win_top1_odds":kth(win,1),"win_top2_odds":kth(win,2),"win_top3_odds":kth(win,3),
  "win_top2_to_top1":kth(win,2)/win[0] if len(win)>=2 else None,
  "win_top1_share":float(win_q[0]) if len(win_q) else None,
  "win_top3_share":float(win_q[:min(3,len(win_q))].sum()) if len(win_q) else None
 }

def race_meta(pack,year,date,rid,shape):
 r=pack.get("race") or {}
 rc=r.get("race_class_normalized") or r.get("race_class") or r.get("class") or "UNKNOWN"
 dist=finite(r.get("distance_m"),0.0)
 return {
  "year":year,"race_id":rid,"race_date":date,"race_month":iint(date[5:7]),
  "surface":str(r.get("surface") or "UNKNOWN"),"discipline":str(r.get("discipline") or "UNKNOWN"),
  "venue_code":str(r.get("venue_code") or "UNKNOWN"),"track_condition":str(r.get("track_condition") or "UNKNOWN"),
  "weather":str(r.get("weather") or "UNKNOWN"),"race_class":str(rc),"grade":str(r.get("grade") or "UNKNOWN"),
  "distance_m":dist,"distance_bin":distance_bin(dist),
  "field_size":shape["field_size"],"field_size_bin":field_bin(shape["field_size"])
 }

def build(root):
 root=Path(root); rows=[]; counters=Counter(); yearly=Counter()
 for year in YEARS:
  files=sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz"))
  if not files: raise SystemExit(f"no daily files year={year}")
  for dp in files:
   date=dp.name[:10]; op=root/"data"/"odds"/"daily"/dp.name
   if not op.exists():
    counters["missing_odds_day"]+=1; yearly[(year,"missing_odds_day")]+=1; continue
   odds_rows={str(x.get("race_id") or ""):x for x in read_gz(op) if str(x.get("race_id") or "")}
   for pack in read_gz(dp):
    counters["race_seen"]+=1; yearly[(year,"race_seen")]+=1
    rid=str((pack.get("race") or {}).get("race_id") or "")
    rec=odds_rows.get(rid)
    if not rid or rec is None:
     counters["missing_odds_row"]+=1; yearly[(year,"missing_odds_row")]+=1; continue
    odds_map=decode_odds(rec); shape=market_shape(odds_map)
    if shape is None:
     counters["no_trifecta_odds"]+=1; yearly[(year,"no_trifecta_odds")]+=1; continue
    if shape["ticket_count"]!=shape["expected_tickets"]:
     counters["incomplete_universe"]+=1; yearly[(year,"incomplete_universe")]+=1; continue
    payouts,present=payout_map(pack)
    if "TRIFECTA" not in present:
     counters["missing_trifecta_payout"]+=1; yearly[(year,"missing_trifecta_payout")]+=1; continue
    wins=[(tuple(k[1]),float(v)) for k,v in payouts.items() if k[0]=="TRIFECTA" and float(v)>0]
    if len(wins)!=1:
     counters["non_single_winner"]+=1; yearly[(year,"non_single_winner")]+=1; continue
    winner=wins[0][0]
    rank=None; last=None; competition_rank=0
    for pos,(nums,odd) in enumerate(shape["tri"],1):
     if last is None or odd!=last: competition_rank=pos; last=odd
     if nums==winner:
      rank=competition_rank; break
    if rank is None:
     counters["winner_odds_missing"]+=1; yearly[(year,"winner_odds_missing")]+=1; continue
    row=race_meta(pack,year,date,rid,shape)
    row.update({k:v for k,v in shape.items() if k!="tri"})
    row["winner_market_rank"]=int(rank)
    row["winner_rank_pct"]=(rank-1)/max(1,shape["ticket_count"]-1)
    row["layer"]=layer(rank)
    rows.append(row); counters["ready"]+=1; yearly[(year,"ready")]+=1
 if not rows: raise SystemExit("no usable races")
 return pd.DataFrame(rows),counters,yearly

def layer_summary(df):
 out=[]
 overall=len(df)
 for name in LAYERS:
  g=df[df.layer==name]
  r={"layer":name,"races":len(g),"share_pct":100*len(g)/overall,
     "median_market_rank":float(g.winner_market_rank.median()) if len(g) else None,
     "median_rank_pct":float(g.winner_rank_pct.median()) if len(g) else None}
  for col in NUMERIC:
   s=pd.to_numeric(g[col],errors="coerce").dropna()
   r[col+"_mean"]=float(s.mean()) if len(s) else None
   r[col+"_median"]=float(s.median()) if len(s) else None
  out.append(r)
 return pd.DataFrame(out)

def yearly_summary(df,yearly):
 out=[]
 for y in YEARS:
  g=df[df.year==y]; total=len(g)
  r={"year":y,"ready_races":total}
  for name in LAYERS:
   n=int((g.layer==name).sum()); r[name.lower()+"_races"]=n; r[name.lower()+"_pct"]=100*n/total if total else None
  for key in ("race_seen","incomplete_universe","non_single_winner","winner_odds_missing"):
   r[key]=int(yearly.get((y,key),0))
  out.append(r)
 return pd.DataFrame(out)

def numeric_effects(df):
 out=[]; short=df[df.layer=="SHORT"]; deep=df[df.layer=="DEEP"]
 for col in NUMERIC:
  s=pd.to_numeric(df[col],errors="coerce"); ok=s.notna()
  corr=float(s[ok].rank().corr(df.loc[ok,"winner_market_rank"].rank())) if ok.sum()>=50 and s[ok].nunique()>1 else None
  a=pd.to_numeric(short[col],errors="coerce").dropna(); b=pd.to_numeric(deep[col],errors="coerce").dropna()
  smd=None
  if len(a)>1 and len(b)>1:
   pooled=math.sqrt(max(0,(float(a.var())+float(b.var()))/2))
   if pooled>0:smd=(float(b.mean())-float(a.mean()))/pooled
  out.append({"feature":col,"spearman_vs_winner_rank":corr,
              "short_mean":float(a.mean()) if len(a) else None,
              "deep_mean":float(b.mean()) if len(b) else None,
              "deep_minus_short_smd":smd})
 z=pd.DataFrame(out)
 z["abs_spearman"]=z.spearman_vs_winner_rank.abs()
 z["abs_smd"]=z.deep_minus_short_smd.abs()
 return z.sort_values(["abs_spearman","abs_smd"],ascending=False)

def categorical_profile(df):
 out=[]; base_deep=float((df.layer=="DEEP").mean())
 for dim in CATEGORICAL:
  for value,g in df.groupby(dim,dropna=False):
   if len(g)<50: continue
   short=float((g.layer=="SHORT").mean()); mid=float((g.layer=="MIDDLE").mean()); deep=float((g.layer=="DEEP").mean())
   out.append({"dimension":dim,"value":str(value),"races":len(g),
               "short_pct":100*short,"middle_pct":100*mid,"deep_pct":100*deep,
               "deep_lift_vs_all":deep/base_deep if base_deep>0 else None,
               "median_winner_rank":float(g.winner_market_rank.median())})
 z=pd.DataFrame(out)
 if len(z):
  z["abs_deep_lift_delta"]=(z.deep_lift_vs_all-1).abs()
  z=z.sort_values(["abs_deep_lift_delta","races"],ascending=[False,False])
 return z

def main():
 a=parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
 df,counters,yearly_c=build(a.backfill_root)
 ls=layer_summary(df); ys=yearly_summary(df,yearly_c); ne=numeric_effects(df); cp=categorical_profile(df)
 overall={
  "races":len(df),"years":list(YEARS),
  "short_pct":100*float((df.layer=="SHORT").mean()),
  "middle_pct":100*float((df.layer=="MIDDLE").mean()),
  "deep_pct":100*float((df.layer=="DEEP").mean()),
  "median_winner_rank":float(df.winner_market_rank.median()),
  "p90_winner_rank":float(df.winner_market_rank.quantile(.9)),
  "median_ticket_count":float(df.ticket_count.median())
 }
 summary={"contract":"L2_TRIFECTA_ODDS_RANK_REVERSE_V1_RESULT",
          "definition":{"SHORT":"winning trifecta final-odds rank 1-30",
                        "MIDDLE":"rank 31-120","DEEP":"rank 121+",
                        "ranking":"tie-safe competition rank, shortest final trifecta odds first",
                        "universe":"strictly complete priced trifecta boards only",
                        "multi_winner_policy":"exclude races with multiple positive trifecta payouts from primary feature analysis"},
          "features":"pre-race race context and whole-market shape only; no winner-dependent horse features",
          "overall":overall,
          "top_numeric_associations":ne.head(12)[["feature","spearman_vs_winner_rank","deep_minus_short_smd"]].replace({np.nan:None}).to_dict("records"),
          "top_categorical_deep_lifts":cp.head(20)[["dimension","value","races","deep_pct","deep_lift_vs_all"]].replace({np.nan:None}).to_dict("records") if len(cp) else [],
          "counters":dict(counters),"2026_locked":True}
 ys.to_csv(out/"yearly.csv",index=False)
 ls.to_csv(out/"layer-summary.csv",index=False)
 ne.to_csv(out/"numeric-effects.csv",index=False)
 cp.to_csv(out/"categorical-profile.csv",index=False)
 (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 with open(out/"README.md","w",encoding="utf-8") as f:
  f.write("# L2 Trifecta Odds-Rank Reverse V1\n\n")
  f.write("Reverse analysis of where the actual winning trifecta sits in the complete final-odds board. ")
  f.write("Only features available before the result are compared across SHORT (1-30), MIDDLE (31-120), and DEEP (121+). ")
  f.write("2026 is sealed. Raw per-race rows stay runner-local; only compact summaries are committed.\n")
 print("TRIFECTA_ODDS_RANK_REVERSE_V1_READY")
 print(json.dumps(overall,ensure_ascii=False,separators=(",",":")))
 print("===== YEARLY ====="); print(ys.to_csv(index=False))
 print("===== LAYERS ====="); print(ls.to_csv(index=False))
 print("===== NUMERIC TOP12 ====="); print(ne.head(12).to_csv(index=False))
 print("===== CATEGORICAL TOP20 ====="); print(cp.head(20).to_csv(index=False) if len(cp) else "none")

if __name__=="__main__": main()
