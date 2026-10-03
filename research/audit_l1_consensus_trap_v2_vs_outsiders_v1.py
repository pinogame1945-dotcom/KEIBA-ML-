#!/usr/bin/env python3
import argparse,csv,gzip,json,math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

YEARS=(2024,2025)
STRUCTURAL4=("outsider_raceshape","outsider_field","outsider_opponentgap","outsider_chimera")

def parse_args():
    p=argparse.ArgumentParser(description="Audit whether Consensus Trap V2 is merely existing Outsider dissent.")
    p.add_argument("--v2-predictions",required=True)
    p.add_argument("--ballots-year",action="append",required=True,help="YEAR=PATH")
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def load_ballots(path):
    by_race=defaultdict(dict)
    candidates=set()
    with gzip.open(path,"rt",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            rid=str(r["race_id"]); cand=str(r["candidate"]); candidates.add(cand)
            ranks={}
            for k in (1,2,3):
                h=str(r.get(f"top{k}_horse_id") or "")
                if h: ranks[h]=k
            by_race[rid][cand]=ranks
    return by_race,sorted(candidates)

def corr(a,b):
    if len(a)<3: return None
    x=pd.Series(a,dtype=float); y=pd.Series(b,dtype=float)
    return float(x.rank(method="average").corr(y.rank(method="average")))

def auc_safe(y,s):
    y=np.asarray(y,dtype=int); s=np.asarray(s,dtype=float)
    if len(np.unique(y))<2: return None
    return float(roc_auc_score(y,s))

def metric_row(year,model,label,q,base_rate):
    n=len(q); c=int(q["collapse"].sum()) if n else 0
    rate=float(q["collapse"].mean()) if n else None
    return {
        "year":year,"model":model,"group":label,"horses":n,"collapse_n":c,
        "collapse_rate_pct":100*rate if rate is not None else None,
        "lift_vs_strong_consensus":rate/base_rate if rate is not None and base_rate else None,
        "v2_flag_rate_pct":100*float(q["structural_high20"].mean()) if n else None,
        "mean_struct4_top3_support":float(q["struct4_top3_support"].mean()) if n else None,
        "mean_all13_top3_support":float(q["all13_top3_support"].mean()) if n else None,
    }

def main():
    a=parse_args()
    ballots={}
    candidate_ref=None
    for spec in a.ballots_year:
        y,p=spec.split("=",1); y=int(y)
        b,c=load_ballots(p)
        ballots[y]=b
        if candidate_ref is None: candidate_ref=c
        elif c!=candidate_ref: raise SystemExit("candidate set drift")
    if set(ballots)!=set(YEARS): raise SystemExit(f"need ballots {YEARS}")
    missing=[x for x in STRUCTURAL4 if x not in candidate_ref]
    if missing: raise SystemExit(f"missing structural outsiders {missing}")

    pred=pd.read_csv(a.v2_predictions,compression="gzip")
    pred=pred[pred["test_year"].isin(YEARS)].copy()
    if 2026 in set(pred["test_year"].astype(int)): raise SystemExit("2026 sealed")

    enriched=[]
    for r in pred.itertuples(index=False):
        year=int(r.test_year); rid=str(r.race_id); hid=str(r.horse_id)
        views=ballots[year].get(rid)
        if not views: raise SystemExit(f"ballot race missing {year} {rid}")
        if any(c not in views for c in candidate_ref):
            raise SystemExit(f"candidate ballot missing {year} {rid}")
        support={c:int(hid in views[c]) for c in candidate_ref}
        top1={c:int(views[c].get(hid)==1) for c in candidate_ref}
        row=r._asdict()
        row["all13_top3_support"]=sum(support.values())
        row["all13_top1_support"]=sum(top1.values())
        row["struct4_top3_support"]=sum(support[c] for c in STRUCTURAL4)
        row["struct4_top1_support"]=sum(top1[c] for c in STRUCTURAL4)
        # Fixed, outcome-independent "strong dissent": at least 3 of 4 structural Outsiders omit anchor from Top3.
        row["struct4_strong_dissent"]=int(row["struct4_top3_support"]<=1)
        for c in STRUCTURAL4:
            row[f"{c}_top3_support"]=support[c]
            row[f"{c}_dissent"]=1-support[c]
        enriched.append(row)
    z=pd.DataFrame(enriched)

    group_rows=[]; candidate_rows=[]; overlap_rows=[]; bucket_rows=[]; correlation_rows=[]
    for year in YEARS:
        for model in sorted(z["model"].unique()):
            q=z[(z["test_year"]==year)&(z["model"]==model)&(z["strong_consensus"]==1)].copy()
            if q.empty: continue
            base=float(q["collapse"].mean())
            v2=q["structural_high20"]==1
            dissent=q["struct4_strong_dissent"]==1
            groups=[
                ("STRONG_CONSENSUS",q),
                ("V2_HIGH20",q[v2]),
                ("STRUCT4_STRONG_DISSENT",q[dissent]),
                ("BOTH",q[v2 & dissent]),
                ("V2_ONLY_NOT_STRONG_DISSENT",q[v2 & ~dissent]),
                ("DISSENT_ONLY_NOT_V2",q[~v2 & dissent]),
                ("NEITHER",q[~v2 & ~dissent]),
                ("V2_WITH_STRUCT4_SUPPORT_3PLUS",q[v2 & (q["struct4_top3_support"]>=3)]),
                ("V2_WITH_STRUCT4_SUPPORT_4OF4",q[v2 & (q["struct4_top3_support"]==4)]),
            ]
            for label,g in groups:
                group_rows.append(metric_row(year,model,label,g,base))

            for k,g in q.groupby("struct4_top3_support",sort=True):
                bucket_rows.append({
                    "year":year,"model":model,"struct4_top3_support":int(k),
                    "horses":len(g),"collapse_n":int(g["collapse"].sum()),
                    "collapse_rate_pct":100*float(g["collapse"].mean()),
                    "v2_high20_n":int(g["structural_high20"].sum()),
                    "v2_high20_rate_pct":100*float(g["structural_high20"].mean()),
                })

            correlation_rows.append({
                "year":year,"model":model,"horses":len(q),
                "spearman_v2risk_vs_struct4_support":corr(q["structural_risk"],q["struct4_top3_support"]),
                "spearman_v2risk_vs_all13_support":corr(q["structural_risk"],q["all13_top3_support"]),
                "auc_collapse_v2_structural_risk":auc_safe(q["collapse"],q["structural_risk"]),
                "auc_collapse_struct4_dissent_count":auc_safe(q["collapse"],4-q["struct4_top3_support"]),
                "auc_collapse_all13_dissent_count":auc_safe(q["collapse"],13-q["all13_top3_support"]),
            })

            flagged=q[v2]
            v2_collapses=set(zip(flagged.loc[flagged["collapse"]==1,"race_id"].astype(str),flagged.loc[flagged["collapse"]==1,"horse_id"].astype(str)))
            for c in STRUCTURAL4:
                d=q[q[f"{c}_dissent"]==1]
                s=q[q[f"{c}_dissent"]==0]
                dset=set(zip(d["race_id"].astype(str),d["horse_id"].astype(str)))
                fset=set(zip(flagged["race_id"].astype(str),flagged["horse_id"].astype(str)))
                inter=fset&dset; union=fset|dset
                dcoll=set(zip(d.loc[d["collapse"]==1,"race_id"].astype(str),d.loc[d["collapse"]==1,"horse_id"].astype(str)))
                candidate_rows.append({
                    "year":year,"model":model,"candidate":c,
                    "dissent_n":len(d),"support_n":len(s),
                    "dissent_collapse_rate_pct":100*float(d["collapse"].mean()) if len(d) else None,
                    "support_collapse_rate_pct":100*float(s["collapse"].mean()) if len(s) else None,
                    "dissent_minus_support_pp":100*(float(d["collapse"].mean())-float(s["collapse"].mean())) if len(d) and len(s) else None,
                    "v2_flagged_n":len(flagged),"v2_dissent_overlap_n":len(inter),
                    "v2_dissent_jaccard":len(inter)/len(union) if union else None,
                    "v2_flagged_collapses":len(v2_collapses),
                    "candidate_dissent_collapses":len(dcoll),
                    "shared_flagged_collapse_n":len(v2_collapses&dcoll),
                    "v2_flagged_collapse_not_candidate_dissent_n":len(v2_collapses-dcoll),
                })

            dset=set(zip(q.loc[dissent,"race_id"].astype(str),q.loc[dissent,"horse_id"].astype(str)))
            fset=set(zip(flagged["race_id"].astype(str),flagged["horse_id"].astype(str)))
            v2c=set(zip(flagged.loc[flagged["collapse"]==1,"race_id"].astype(str),flagged.loc[flagged["collapse"]==1,"horse_id"].astype(str)))
            dc=set(zip(q.loc[dissent & (q["collapse"]==1),"race_id"].astype(str),q.loc[dissent & (q["collapse"]==1),"horse_id"].astype(str)))
            overlap_rows.append({
                "year":year,"model":model,
                "v2_flagged_n":len(fset),"struct4_strong_dissent_n":len(dset),
                "overlap_n":len(fset&dset),"jaccard":len(fset&dset)/len(fset|dset) if fset|dset else None,
                "v2_flagged_collapses":len(v2c),"struct4_dissent_collapses":len(dc),
                "shared_collapse_n":len(v2c&dc),
                "v2_only_collapse_n":len(v2c-dc),
                "struct4_dissent_only_collapse_n":len(dc-v2c),
                "v2_collapse_share_not_explained_by_struct4_dissent":len(v2c-dc)/len(v2c) if v2c else None,
            })

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(group_rows).to_csv(out/"groups.csv",index=False)
    pd.DataFrame(candidate_rows).to_csv(out/"candidate-dissent.csv",index=False)
    pd.DataFrame(overlap_rows).to_csv(out/"v2-vs-struct4-overlap.csv",index=False)
    pd.DataFrame(bucket_rows).to_csv(out/"struct4-support-buckets.csv",index=False)
    pd.DataFrame(correlation_rows).to_csv(out/"correlations.csv",index=False)

    # Concise audit decision: duplicate only if strong existing dissent covers >=80% of V2 collapses in both years/models.
    ov=pd.DataFrame(overlap_rows)
    ov["covered_share"]=1-ov["v2_collapse_share_not_explained_by_struct4_dissent"]
    duplicate=bool((ov["covered_share"]>=0.80).all()) if len(ov) else False
    summary={
        "contract":"L1_CONSENSUS_TRAP_V2_OUTSIDER_OVERLAP_AUDIT_V1",
        "question":"Is V2 structural warning merely a repackaging of existing safe Outsider dissent?",
        "existing_structural_outsiders":list(STRUCTURAL4),
        "strong_dissent_definition":"anchor is Top3-supported by at most 1 of 4 structural Outsiders (>=3 of 4 dissent); fixed before outcome review",
        "duplicate_rule":"Only call duplicate if structural4 strong dissent covers >=80% of V2 flagged collapses for both OOS years and both V2 models.",
        "duplicate_by_rule":duplicate,
        "safe_outsider_candidate_count":len(candidate_ref),
        "safe_outsider_candidates":candidate_ref,
        "old_arena_note":"Historical outsider_lap/outsider_network/outsider_style are not in this current tie-safe 13-ballot pack and are not used in this audit.",
        "2026_locked":True,"paid_compute":False,"kaggle_access":False,"artifacts_or_cache":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== SUMMARY ====="); print((out/"summary.json").read_text())
    print("===== OVERLAP ====="); print((out/"v2-vs-struct4-overlap.csv").read_text())
    print("===== GROUPS ====="); print((out/"groups.csv").read_text())
    print("===== CANDIDATE DISSENT ====="); print((out/"candidate-dissent.csv").read_text())
    print("===== CORRELATIONS ====="); print((out/"correlations.csv").read_text())
    print("L1_CONSENSUS_TRAP_V2_OUTSIDER_AUDIT_READY")

if __name__=="__main__":
    main()
