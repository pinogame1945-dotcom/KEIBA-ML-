#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path

import numpy as np
import pandas as pd

YEARS=(2023,2024,2025)
DIRECTIONS=("L1_UPGRADE","L1_DOWNGRADE")
EPS=1e-12

def parse_args():
    p=argparse.ArgumentParser(description="Join frozen L1.7 market-divergence rows with frozen safe Outsider OOS signals.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--outsider-predictions",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def write_csv(path,rows):
    path=Path(path)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def direction_effect(df):
    if df.empty: return None
    sign=np.sign(df["rank_gap"].to_numpy(dtype=float))
    resid=df["target_top3"].to_numpy(dtype=float)-df["market_peer_top3_rate"].to_numpy(dtype=float)
    return 100*float(np.mean(sign*resid))

def alignment_label(x):
    if x>EPS: return "OUTSIDER_AGREES"
    if x<-EPS: return "OUTSIDER_OPPOSES"
    return "OUTSIDER_NEUTRAL"

def main():
    a=parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    m=pd.read_csv(a.market_scored,compression="gzip")
    p=pd.read_csv(a.outsider_predictions,compression="gzip",
                  usecols=["year","race_id","horse_id","rank","score","p3","p3_rank_baseline","p3_delta"])

    m["year"]=pd.to_numeric(m["year"],errors="raise").astype(int)
    p["year"]=pd.to_numeric(p["year"],errors="raise").astype(int)
    if 2026 in set(m["year"]) or 2026 in set(p["year"]): raise SystemExit("2026 sealed")

    for c in ("consensus_rank","market_rank","rank_gap","target_top3","market_peer_top3_rate"):
        m[c]=pd.to_numeric(m[c],errors="raise")
    for c in ("rank","score","p3","p3_rank_baseline","p3_delta"):
        p[c]=pd.to_numeric(p[c],errors="raise")

    m=m[m["year"].isin(YEARS)].copy()
    p=p[p["year"].isin(YEARS)].copy()

    # The Outsider predictions file also contains outcome columns historically,
    # but only the explicitly selected safe signal columns above are loaded.
    keys=["year","race_id","horse_id"]
    if p.duplicated(keys).any(): raise SystemExit("duplicate outsider prediction keys")
    z=m.merge(p,on=keys,how="inner",validate="one_to_one")
    if z.empty: raise SystemExit("empty join")
    if not np.array_equal(z["consensus_rank"].astype(int).to_numpy(),z["rank"].astype(int).to_numpy()):
        raise SystemExit("king rank mismatch after join")

    # OOS calibrator scope is King ranks 1..10 only.
    if int(z["consensus_rank"].max())>10: raise SystemExit("outsider predictions outside rank<=10")
    z=z[z["direction"].isin(DIRECTIONS)].copy()
    z["outsider_alignment_score"]=np.sign(z["rank_gap"].to_numpy(dtype=float))*z["p3_delta"].to_numpy(dtype=float)
    z["outsider_alignment"]=[alignment_label(x) for x in z["outsider_alignment_score"]]

    rows=[]
    yearly=[]
    for year in YEARS:
        for direction in DIRECTIONS:
            d=z[(z["year"]==year)&(z["direction"]==direction)].copy()
            if d.empty: continue
            base=direction_effect(d)
            for label in ("ALL","OUTSIDER_AGREES","OUTSIDER_OPPOSES","OUTSIDER_NEUTRAL"):
                q=d if label=="ALL" else d[d["outsider_alignment"]==label]
                if q.empty: continue
                rows.append({
                    "test_year":year,
                    "direction":direction,
                    "outsider_alignment":label,
                    "horses":len(q),
                    "races":q["race_id"].nunique(),
                    "coverage_within_direction_pct":100*len(q)/len(d),
                    "actual_top3_pct":100*float(q["target_top3"].mean()),
                    "market_peer_top3_pct":100*float(q["market_peer_top3_rate"].mean()),
                    "direction_correct_effect_pp":direction_effect(q),
                    "mean_outsider_score":float(q["score"].mean()),
                    "mean_p3_delta_pp":100*float(q["p3_delta"].mean()),
                    "mean_alignment_score_pp":100*float(q["outsider_alignment_score"].mean()),
                    "effect_minus_all_pp":direction_effect(q)-base,
                })
            agree=d[d["outsider_alignment"]=="OUTSIDER_AGREES"]
            oppose=d[d["outsider_alignment"]=="OUTSIDER_OPPOSES"]
            yearly.append({
                "test_year":year,
                "direction":direction,
                "all_horses":len(d),
                "all_effect_pp":base,
                "agree_horses":len(agree),
                "agree_effect_pp":direction_effect(agree),
                "oppose_horses":len(oppose),
                "oppose_effect_pp":direction_effect(oppose),
                "agree_minus_all_pp":direction_effect(agree)-base if len(agree) else None,
                "agree_minus_oppose_pp":direction_effect(agree)-direction_effect(oppose) if len(agree) and len(oppose) else None,
                "agree_beats_all":bool(len(agree) and direction_effect(agree)>base),
                "agree_beats_oppose":bool(len(agree) and len(oppose) and direction_effect(agree)>direction_effect(oppose)),
            })

    # Raw support score buckets: descriptive only, no threshold optimization.
    bucket_rows=[]
    bins=[-0.1,0,3,6,9,39]
    labels=["0","1-3","4-6","7-9","10+"]
    z["outsider_score_bucket"]=pd.cut(z["score"],bins=bins,labels=labels,include_lowest=True,right=True)
    for year in YEARS:
        for direction in DIRECTIONS:
            d=z[(z["year"]==year)&(z["direction"]==direction)]
            for b in labels:
                q=d[d["outsider_score_bucket"].astype(str)==b]
                if q.empty: continue
                bucket_rows.append({
                    "test_year":year,"direction":direction,"outsider_score_bucket":b,
                    "horses":len(q),"races":q["race_id"].nunique(),
                    "direction_correct_effect_pp":direction_effect(q),
                    "actual_top3_pct":100*float(q["target_top3"].mean()),
                    "market_peer_top3_pct":100*float(q["market_peer_top3_rate"].mean()),
                })

    write_csv(out/"alignment-summary.csv",rows)
    write_csv(out/"yearly-headline.csv",yearly)
    write_csv(out/"score-buckets.csv",bucket_rows)
    keep=["year","race_id","horse_id","consensus_rank","market_rank","rank_gap","direction",
          "score","p3","p3_rank_baseline","p3_delta","outsider_alignment_score","outsider_alignment",
          "target_top3","market_peer_top3_rate"]
    z[keep].to_csv(out/"joined-scored.csv.gz",index=False,compression="gzip")

    ydf=pd.DataFrame(yearly)
    decision={}
    for direction in DIRECTIONS:
        q=ydf[ydf["direction"]==direction].sort_values("test_year")
        decision[direction]={
            "outsider_agreement_enriches_vs_all_all_years":bool(len(q)==3 and q["agree_beats_all"].all()),
            "outsider_agreement_beats_opposition_all_years":bool(len(q)==3 and q["agree_beats_oppose"].all()),
            "rows":q.to_dict(orient="records"),
        }

    summary={
        "contract":"L17_OUTSIDER_MARKET_DIVERGENCE_V1",
        "purpose":"Measure whether frozen safe Outsider opinion reinforces or contradicts Seven-King market dissent.",
        "market_source":"Frozen V3 market-divergence horse rows (2023-2025).",
        "outsider_source":"L1.7 King Outsider Podium Calibrator WF V1 predictions run 36747215581.",
        "outsider_ballots_source_run":36724124925,
        "join_scope":"King ranks 1..10 only, because p3_delta calibration scope is ranks 1..10.",
        "alignment_rule":"sign(market_rank - king_rank) * p3_delta; >0 agrees with Seven-King dissent, <0 opposes.",
        "ranking_policy":"SEVEN_KING_UNCHANGED",
        "outsider_policy":"SIGNAL_ONLY_NO_RERANK",
        "target_or_market_used_to_build_outsider_signal":False,
        "odds_used_in_outsider_signal":False,
        "2026_locked":True,
        "decision":decision,
        "promotion":False
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== YEARLY HEADLINE =====")
    print((out/"yearly-headline.csv").read_text(encoding="utf-8"))
    print("===== SUMMARY =====")
    print((out/"summary.json").read_text(encoding="utf-8"))
    print("L17_OUTSIDER_MARKET_DIVERGENCE_V1_READY")

if __name__=="__main__":
    main()
