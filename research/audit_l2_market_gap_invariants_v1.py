#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from build_l2_l17_fullfield_dataset_v1 import load_l17
from run_l2_ticket_eval_quinella_v0 import build_year_frame,write_csv
from run_l2_ticket_market_gap_v2 import add_market_features,market_feature_columns,train_rank_predict,add_ranks,apply_gap_policy

INPUT_YEARS=(2021,2022,2023,2024,2025)
TEST_YEARS=(2022,2023,2024,2025)
FROZEN_POLICY=(15,5,1)

CANDIDATES={
 "GAP15_24": lambda z:(z.rank_upgrade>=15)&(z.rank_upgrade<25),
 "GAP15_24_FIELD15_16": lambda z:(z.rank_upgrade>=15)&(z.rank_upgrade<25)&(z.field_size.between(15,16)),
 "GAP15_24_MARKET31_40": lambda z:(z.rank_upgrade>=15)&(z.rank_upgrade<25)&(z.market_rank.between(31,40)),
 "GAP15_24_MODEL9_10": lambda z:(z.rank_upgrade>=15)&(z.rank_upgrade<25)&(z.model_rank.between(9,10)),
 "GAP15_24_L17_41_60": lambda z:(z.rank_upgrade>=15)&(z.rank_upgrade<25)&(z.l17_rank_score.between(41,60)),
}

def args():
 p=argparse.ArgumentParser()
 for y in INPUT_YEARS:p.add_argument(f"--l17-{y}",required=True)
 p.add_argument("--backfill-root",required=True);p.add_argument("--out-dir",required=True)
 return p.parse_args()

def metrics(g,label):
 n=len(g); stake=100.0*n; ret=float(g.return_yen_per100.sum()); hits=int(g.hit.sum())
 wins=sorted(g.loc[g.return_yen_per100>0,"return_yen_per100"].astype(float),reverse=True)
 largest=wins[0] if wins else 0.0
 return {
  "label":label,"tickets":n,"hits":hits,
  "hit_rate_pct":100*hits/n if n else 0.0,
  "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
  "roi_pct":100*ret/stake if stake else None,
  "largest_single_return_yen":largest,
  "largest_return_share_pct":100*largest/ret if ret else 0.0,
  "roi_after_remove_largest_win_pct":100*(ret-largest)/stake if stake else None,
  "median_odds":float(g.odds.median()) if n else None,
  "median_market_rank":float(g.market_rank.median()) if n else None,
  "median_model_rank":float(g.model_rank.median()) if n else None,
  "median_l17_rank":float(g.l17_rank_score.median()) if n else None,
 }

def bootstrap(g,seed,samples=10000):
 if g.empty:return {"samples":samples,"prob_roi_gt_100_pct":None,"roi_median":None,"roi_p2_5":None,"roi_p97_5":None}
 a=g.return_yen_per100.to_numpy(float);n=len(a);rng=np.random.default_rng(seed);out=np.empty(samples)
 for i in range(0,samples,250):
  m=min(250,samples-i); idx=rng.integers(0,n,size=(m,n));out[i:i+m]=a[idx].sum(axis=1)/n
 return {"samples":samples,"prob_roi_gt_100_pct":100*float((out>100).mean()),
         "roi_p2_5":float(np.percentile(out,2.5)),"roi_median":float(np.percentile(out,50)),
         "roi_p97_5":float(np.percentile(out,97.5))}

def main():
 a=args()
 l17={y:load_l17(getattr(a,f"l17_{y}"),y) for y in INPUT_YEARS}
 frames={y:add_market_features(build_year_frame(y,l17[y],a.backfill_root)) for y in INPUT_YEARS}
 cols=market_feature_columns(frames[2022])
 preds={}
 for y in TEST_YEARS:
  train_years=[2021] if y==2022 else [t for t in (2022,2023,2024) if t<y]
  train=pd.concat([frames[t] for t in train_years],ignore_index=True)
  test=frames[y].copy().reset_index(drop=True)
  score,_=train_rank_predict(train,test,cols,93000+y);test["market_aware_score"]=score;test=add_ranks(test)
  preds[y]=apply_gap_policy(test,*FROZEN_POLICY).copy()

 rows=[];boots=[]
 for name,fn in CANDIDATES.items():
  for y in TEST_YEARS:
   g=preds[y][fn(preds[y])].copy()
   r=metrics(g,name);r.update({"candidate":name,"year":y});rows.append(r)
   b=bootstrap(g,20260930+y+len(name));b.update({"candidate":name,"year":y,"tickets":len(g)});boots.append(b)
  pooled=pd.concat([preds[y][fn(preds[y])].copy() for y in TEST_YEARS],ignore_index=True)
  r=metrics(pooled,name);r.update({"candidate":name,"year":"2022-2025"});rows.append(r)

 out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
 write_csv(out/"candidate-by-year.csv",rows);write_csv(out/"candidate-bootstrap.csv",boots)
 summary={"contract":"L2_MARKET_GAP_INVARIANT_AUDIT_V1_RESULT",
          "frozen_policy":{"model_top_k":15,"min_market_rank_upgrade":5,"max_tickets_per_race":1},
          "candidate_definitions":list(CANDIDATES),"test_years":list(TEST_YEARS),
          "policy_reselected":False,"2026_locked":True}
 (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print("L2_MARKET_GAP_INVARIANT_AUDIT_V1_READY");print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":main()
