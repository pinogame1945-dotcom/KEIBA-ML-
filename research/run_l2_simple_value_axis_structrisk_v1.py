#!/usr/bin/env python3
import json,time
from pathlib import Path
import pandas as pd

AXIS_SRC=Path("research-results/l17-outsider-market-divergence-v1/run-36821694998/joined-scored.csv.gz")
STRUCT_SRC=Path("research-results/l1-structural-risk-fullfield-v1/run-37154323510/oos-fullfield-votes.csv.gz")
OUT=Path("research-results/l2-simple-value-axis-structrisk-v1")
YEARS=[2024,2025]

def choose(g):
    return g.sort_values(
        ["p3","rank_gap","score","king_rank","market_rank","horse_id"],
        ascending=[False,False,False,True,False,True]
    ).iloc[0]

def summarize(z,label,mode,total_races):
    if z.empty:
        return {"mode":mode,"scope":label,"picks":0,"coverage_pct":0.0}
    return {
        "mode":mode,
        "scope":label,
        "picks":int(len(z)),
        "coverage_pct":100*len(z)/total_races if total_races else 0.0,
        "actual_top3_pct":100*float(z["target_top3"].mean()),
        "market_peer_top3_pct":100*float(z["market_peer_top3_rate"].mean()),
        "lift_vs_market_peer_pp":100*float((z["target_top3"]-z["market_peer_top3_rate"]).mean()),
        "mean_l175_rank":float(z["king_rank"].mean()),
        "mean_market_rank":float(z["market_rank"].mean()),
        "mean_rank_gap":float(z["rank_gap"].mean()),
        "mean_p3_pct":100*float(z["p3"].mean()),
        "mean_outsider_score":float(z["score"].mean()),
        "struct_warning_share_pct":100*float(z["structural_vote_high"].mean()),
    }

def main():
    t0=time.time()
    if not AXIS_SRC.exists() or not STRUCT_SRC.exists():
        raise SystemExit("missing archived inputs")

    a=pd.read_csv(AXIS_SRC,compression="gzip",dtype={"race_id":str,"horse_id":str})
    s=pd.read_csv(STRUCT_SRC,compression="gzip",dtype={"race_id":str,"horse_id":str})
    s=s.rename(columns={"test_year":"year"})
    if 2026 in set(a["year"]) or 2026 in set(s["year"]):
        raise SystemExit("2026 sealed")

    a=a[a["year"].isin(YEARS)].copy()
    a=a[(a["direction"]=="L1_UPGRADE") & (a["rank_gap"]>=2)].copy()
    s=s[s["year"].isin(YEARS)][[
        "year","race_id","horse_id","structural_vote_delta","structural_vote_high",
        "baseline_collapse_risk","full_collapse_risk"
    ]].copy()

    z=a.merge(s,on=["year","race_id","horse_id"],how="inner",validate="one_to_one")
    if len(z)<10000:
        raise SystemExit(f"unexpected low candidate join rows={len(z)}")

    modes={"BASE":[],"STRUCT_VETO_REPLACE":[],"STRUCT_VETO_SKIP":[]}
    audit=[]
    for (year,rid),g in z.groupby(["year","race_id"],sort=False):
        base=choose(g)
        modes["BASE"].append(base)

        safe=g[g["structural_vote_high"]==0]
        if len(safe):
            repl=choose(safe)
            modes["STRUCT_VETO_REPLACE"].append(repl)
        else:
            repl=base
            modes["STRUCT_VETO_REPLACE"].append(repl)

        if int(base["structural_vote_high"])==0:
            modes["STRUCT_VETO_SKIP"].append(base)

        audit.append({
            "year":int(year),"race_id":rid,
            "base_horse_id":str(base["horse_id"]),
            "base_struct_warning":int(base["structural_vote_high"]),
            "base_top3":int(base["target_top3"]),
            "base_lift":float(base["target_top3"]-base["market_peer_top3_rate"]),
            "replacement_horse_id":str(repl["horse_id"]),
            "replacement_changed":int(str(repl["horse_id"])!=str(base["horse_id"])),
            "replacement_struct_warning":int(repl["structural_vote_high"]),
            "replacement_top3":int(repl["target_top3"]),
            "replacement_lift":float(repl["target_top3"]-repl["market_peer_top3_rate"]),
        })

    frames={k:pd.DataFrame(v).reset_index(drop=True) for k,v in modes.items()}
    total={y:int(z[z["year"]==y]["race_id"].nunique()) for y in YEARS}
    total["POOLED"]=int(z[["year","race_id"]].drop_duplicates().shape[0])

    rows=[]
    for mode,df in frames.items():
        rows.append(summarize(df,"POOLED",mode,total["POOLED"]))
        for y in YEARS:
            yy=df[df["year"]==y]
            rows.append(summarize(yy,str(y),mode,total[y]))

    h=pd.DataFrame(rows)
    base_pool=h[(h["mode"]=="BASE")&(h["scope"]=="POOLED")].iloc[0]
    rep_pool=h[(h["mode"]=="STRUCT_VETO_REPLACE")&(h["scope"]=="POOLED")].iloc[0]
    skip_pool=h[(h["mode"]=="STRUCT_VETO_SKIP")&(h["scope"]=="POOLED")].iloc[0]

    aa=pd.DataFrame(audit)
    changed=aa[aa["replacement_changed"]==1].copy()
    warning=aa[aa["base_struct_warning"]==1].copy()
    diagnostics={
        "total_races":int(len(aa)),
        "base_warning_races":int(len(warning)),
        "base_warning_share_pct":100*len(warning)/len(aa),
        "replacement_changed_races":int(len(changed)),
        "replacement_changed_share_pct":100*len(changed)/len(aa),
        "base_warning_top3_pct":100*float(warning["base_top3"].mean()) if len(warning) else None,
        "base_safe_top3_pct":100*float(aa[aa["base_struct_warning"]==0]["base_top3"].mean()),
        "changed_base_top3_pct":100*float(changed["base_top3"].mean()) if len(changed) else None,
        "changed_replacement_top3_pct":100*float(changed["replacement_top3"].mean()) if len(changed) else None,
        "changed_top3_delta_pp":100*float((changed["replacement_top3"]-changed["base_top3"]).mean()) if len(changed) else None,
        "changed_lift_delta_pp":100*float((changed["replacement_lift"]-changed["base_lift"]).mean()) if len(changed) else None,
    }

    OUT.mkdir(parents=True,exist_ok=True)
    h.to_csv(OUT/"headline.csv",index=False)
    aa.to_csv(OUT/"race-audit.csv.gz",index=False,compression="gzip")

    summary={
        "contract":"L2_SIMPLE_VALUE_AXIS_STRUCTRISK_V1_RESULT",
        "question":"Does frozen Structural Risk improve the simple L1.75-vs-market value-axis strategy?",
        "base_rule":"rank_gap>=2; choose highest existing walk-forward p3",
        "structural_usage":"veto only; no reranking model, no probability correction",
        "modes":{
            "BASE":"current simple value axis",
            "STRUCT_VETO_REPLACE":"if best axis is Structural-high, replace with best non-high candidate in same race when available",
            "STRUCT_VETO_SKIP":"if best axis is Structural-high, skip the race",
        },
        "years":YEARS,
        "base_pooled":base_pool.to_dict(),
        "replace_pooled":rep_pool.to_dict(),
        "skip_pooled":skip_pool.to_dict(),
        "replace_minus_base_top3_pp":float(rep_pool["actual_top3_pct"]-base_pool["actual_top3_pct"]),
        "replace_minus_base_lift_pp":float(rep_pool["lift_vs_market_peer_pp"]-base_pool["lift_vs_market_peer_pp"]),
        "skip_minus_base_top3_pp":float(skip_pool["actual_top3_pct"]-base_pool["actual_top3_pct"]),
        "skip_minus_base_lift_pp":float(skip_pool["lift_vs_market_peer_pp"]-base_pool["lift_vs_market_peer_pp"]),
        "diagnostics":diagnostics,
        "no_new_model":True,"no_kaggle_downloads":True,
        "payout_used":False,"trio_odds_used":False,"roi_used":False,
        "2026_locked":True,"elapsed_seconds":time.time()-t0,
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/"README.md").write_text(
        "# Simple Value Axis + Structural Risk V1\n\n"
        "Tests Structural Risk only as a veto/confirmation tag on the already-defined value-axis strategy. "
        "No new horse model, no rank rewrite, no ticket optimization. 2024-2025 strict OOS Structural votes only.\n",
        encoding="utf-8"
    )
    print("===== HEADLINE =====")
    print(h.to_string(index=False))
    print("===== DIAGNOSTICS =====")
    print(json.dumps(diagnostics,ensure_ascii=False,indent=2))
    print("===== SUMMARY =====")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print("L2_SIMPLE_VALUE_AXIS_STRUCTRISK_V1_READY")

if __name__=="__main__":
    main()
