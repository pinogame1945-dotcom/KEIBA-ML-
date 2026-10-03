#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def ap():
    p=argparse.ArgumentParser()
    p.add_argument("--pred",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def main():
    a=ap()
    z=pd.read_csv(a.pred,compression="gzip")
    if 2026 in set(z["test_year"].astype(int)): raise SystemExit("2026 sealed")
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]; corr=[]
    for year in (2024,2025):
        y=z[z["test_year"]==year].copy()
        for scope,mask in [
            ("KING_TOP6",y["consensus_rank"]<=6),
            ("MARKET_TOP6",y["market_rank"]<=6),
            ("BOTH_TOP6",(y["consensus_rank"]<=6)&(y["market_rank"]<=6)),
            ("KING_TOP3",y["consensus_rank"]<=3),
            ("MARKET_TOP3",y["market_rank"]<=3),
        ]:
            q=y[mask].copy()
            if len(q)<100: continue
            q["vote_decile"]=pd.qcut(q["structural_vote_delta"].rank(method="first"),10,labels=False)+1
            for d,g in q.groupby("vote_decile",sort=True):
                rows.append({
                    "test_year":year,"scope":scope,"decile":int(d),"horses":len(g),
                    "collapse_n":int(g["collapse"].sum()),
                    "collapse_rate_pct":100*float(g["collapse"].mean()),
                    "mean_vote_delta_pp":100*float(g["structural_vote_delta"].mean()),
                    "min_vote_delta_pp":100*float(g["structural_vote_delta"].min()),
                    "max_vote_delta_pp":100*float(g["structural_vote_delta"].max()),
                })
            rho=float(pd.Series(q["structural_vote_delta"]).corr(pd.Series(q["collapse"]),method="spearman"))
            rates=pd.DataFrame([r for r in rows if r["test_year"]==year and r["scope"]==scope]).sort_values("decile")
            dif=np.diff(rates["collapse_rate_pct"].to_numpy(dtype=float))
            corr.append({
                "test_year":year,"scope":scope,"horses":len(q),
                "spearman_vote_vs_collapse":rho,
                "adjacent_non_decreasing_pairs":int((dif>=0).sum()),
                "adjacent_pairs":int(len(dif)),
                "decile10_minus_decile1_pp":float(rates.iloc[-1]["collapse_rate_pct"]-rates.iloc[0]["collapse_rate_pct"]),
            })
    pd.DataFrame(rows).to_csv(out/"deciles.csv",index=False)
    pd.DataFrame(corr).to_csv(out/"monotonicity.csv",index=False)
    summary={
        "contract":"L1_STRUCTURAL_RISK_FULLFIELD_MONOTONICITY_V1",
        "question":"Does a higher Structural Risk vote monotonically imply more top3 misses among upper-ranked horses?",
        "method":"Within each OOS year and scope, split vote delta into equal-count deciles and compare top3-miss rate.",
        "2026_locked":True,"paid_compute":False,"kaggle_access":False
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== MONOTONICITY =====")
    print((out/"monotonicity.csv").read_text())
    print("===== DECILES =====")
    print((out/"deciles.csv").read_text())
    print("STRUCTURAL_RISK_MONOTONICITY_READY")
if __name__=="__main__": main()
