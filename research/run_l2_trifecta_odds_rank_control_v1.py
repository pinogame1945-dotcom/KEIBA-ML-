#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from run_l2_trifecta_odds_rank_reverse_v1 import build,NUMERIC,YEARS

PBANDS=("P00_02","P02_10","P10_25","P25_100")

def parse_args():
 p=argparse.ArgumentParser()
 p.add_argument("--backfill-root",required=True)
 p.add_argument("--out-dir",required=True)
 return p.parse_args()

def pband(p):
 if p<=.02:return "P00_02"
 if p<=.10:return "P02_10"
 if p<=.25:return "P10_25"
 return "P25_100"

def percentile_summary(df):
 z=df.copy(); z["pband"]=z.winner_rank_pct.map(pband); out=[]
 for b in PBANDS:
  g=z[z.pband==b]
  r={"band":b,"races":len(g),"share_pct":100*len(g)/len(z),
     "median_abs_rank":float(g.winner_market_rank.median()) if len(g) else None,
     "median_rank_pct":float(g.winner_rank_pct.median()) if len(g) else None,
     "field_size_mean":float(g.field_size.mean()) if len(g) else None}
  for col in NUMERIC:
   s=pd.to_numeric(g[col],errors="coerce").dropna()
   r[col+"_mean"]=float(s.mean()) if len(s) else None
  out.append(r)
 return pd.DataFrame(out)

def controlled_effects(df):
 out=[]; shallow=df[df.winner_rank_pct<=.02]; deep=df[df.winner_rank_pct>.25]
 for col in NUMERIC:
  x=pd.to_numeric(df[col],errors="coerce")
  ok=x.notna() & df.winner_rank_pct.notna()
  raw=float(x[ok].rank().corr(df.loc[ok,"winner_rank_pct"].rank())) if ok.sum()>=50 and x[ok].nunique()>1 else None

  xs=[];ys=[]
  for fs,g in df.loc[ok].assign(_x=x[ok]).groupby("field_size"):
   if len(g)<50 or g._x.nunique()<2:continue
   xr=g._x.rank(method="average",pct=True).to_numpy(dtype=float)
   yr=g.winner_rank_pct.rank(method="average",pct=True).to_numpy(dtype=float)
   xs.extend((xr-xr.mean()).tolist());ys.extend((yr-yr.mean()).tolist())
  controlled=float(np.corrcoef(xs,ys)[0,1]) if len(xs)>=100 else None

  a=pd.to_numeric(shallow[col],errors="coerce").dropna()
  b=pd.to_numeric(deep[col],errors="coerce").dropna()
  smd=None
  if len(a)>1 and len(b)>1:
   pooled=math.sqrt(max(0,(float(a.var())+float(b.var()))/2))
   if pooled>0:smd=(float(b.mean())-float(a.mean()))/pooled

  out.append({"feature":col,"spearman_vs_rank_pct":raw,
              "fieldsize_controlled_rank_corr":controlled,
              "top2pct_mean":float(a.mean()) if len(a) else None,
              "bottom75pct_mean":float(b.mean()) if len(b) else None,
              "bottom75_minus_top2_smd":smd})
 z=pd.DataFrame(out)
 z["abs_controlled"]=z.fieldsize_controlled_rank_corr.abs()
 z["abs_raw"]=z.spearman_vs_rank_pct.abs()
 return z.sort_values(["abs_controlled","abs_raw"],ascending=False)

def fieldsize_summary(df):
 out=[]
 for fs,g in df.groupby("field_size"):
  if len(g)<50:continue
  out.append({"field_size":int(fs),"races":len(g),
              "median_abs_rank":float(g.winner_market_rank.median()),
              "median_rank_pct":float(g.winner_rank_pct.median()),
              "top2pct_rate":100*float((g.winner_rank_pct<=.02).mean()),
              "top10pct_rate":100*float((g.winner_rank_pct<=.10).mean()),
              "deep25pct_rate":100*float((g.winner_rank_pct>.25).mean()),
              "effective_tickets_mean":float(g.effective_tickets.mean()),
              "top30_share_mean":float(g.top30_share.mean()),
              "top10_share_mean":float(g.top10_share.mean())})
 return pd.DataFrame(out)

def yearly_percentiles(df):
 out=[]
 for y in YEARS:
  g=df[df.year==y]
  out.append({"year":y,"races":len(g),
              "median_rank_pct":float(g.winner_rank_pct.median()),
              "top2pct_rate":100*float((g.winner_rank_pct<=.02).mean()),
              "top10pct_rate":100*float((g.winner_rank_pct<=.10).mean()),
              "deep25pct_rate":100*float((g.winner_rank_pct>.25).mean())})
 return pd.DataFrame(out)

def main():
 a=parse_args(); out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
 df,counters,_=build(a.backfill_root)
 ps=percentile_summary(df); ce=controlled_effects(df); fs=fieldsize_summary(df); ys=yearly_percentiles(df)
 summary={"contract":"L2_TRIFECTA_ODDS_RANK_CONTROL_V1_RESULT","races":len(df),
          "target":"winning trifecta rank percentile within complete final-odds board",
          "purpose":"separate real market-shape signal from the mechanical effect of field size / ticket count",
          "top_controlled_signals":ce.head(12)[["feature","spearman_vs_rank_pct","fieldsize_controlled_rank_corr","bottom75_minus_top2_smd"]].replace({np.nan:None}).to_dict("records"),
          "counters":dict(counters),"2026_locked":True}
 ps.to_csv(out/"percentile-bands.csv",index=False)
 ce.to_csv(out/"controlled-effects.csv",index=False)
 fs.to_csv(out/"fieldsize-summary.csv",index=False)
 ys.to_csv(out/"yearly-percentiles.csv",index=False)
 (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 (out/"README.md").write_text("# Trifecta odds-rank control audit\n\nControls the reverse analysis for field size using rank percentile and within-field-size rank correlation. 2026 sealed.\n",encoding="utf-8")
 print("TRIFECTA_ODDS_RANK_CONTROL_V1_READY")
 print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))
 print("===== PERCENTILE BANDS =====");print(ps.to_csv(index=False))
 print("===== CONTROLLED EFFECTS =====");print(ce.head(15).to_csv(index=False))
 print("===== FIELD SIZE =====");print(fs.to_csv(index=False))
 print("===== YEARLY =====");print(ys.to_csv(index=False))

if __name__=="__main__":main()
