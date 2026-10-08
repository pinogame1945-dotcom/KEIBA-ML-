#!/usr/bin/env python3
import argparse,csv,gc,json,math,os
from pathlib import Path

import numpy as np
import pandas as pd

from run_l1_ceiling_audit_v1 import (
    TEST_YEARS,THREADS,PRIMARY,group_columns,fit_meta,
)
from run_l1_pairwise_resolver_v1 import (
    YEARS,LOCKED_YEAR,PAIR_TEST_YEARS,FEATURE_GROUPS,PAIR_HEADS,
    PRIMARY_SCORE,build_base_oos,fit_pair_models,tournament_scores,
    choose,baseline_choice,
)

GATE_GROUPS=("MODEL_ONLY","HISTORY_AUTO","STRUCTURE","ALL_NUMERIC")
GATE_FRACTIONS=(0.10,0.20,0.30,0.40,0.50)
PAIR_MARGIN_MINS=(0.00,0.03,0.06,0.10)
DEFAULT_POLICY=("MODEL_ONLY","MODEL_ONLY","PAIR_BLEND",0.20,0.00)

def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def primary_rows(cand):
    x=cand[cand["m_is_primary_candidate"]==1].copy().reset_index(drop=True)
    if x["_race_id"].nunique()!=cand["_race_id"].nunique():
        raise SystemExit("primary row coverage mismatch")
    x["collapse_target"]=(1-x["_is_top3"].astype(int)).astype("int8")
    return x

def risk_by_year(candidates):
    out={}
    importance=[]
    for year in PAIR_TEST_YEARS:
        prior=[y for y in TEST_YEARS if y<year]
        tr=pd.concat([primary_rows(candidates[y]) for y in prior],ignore_index=True,copy=False)
        te=primary_rows(candidates[year])
        out[year]={}
        for gi,group in enumerate(GATE_GROUPS):
            cols=group_columns(tr,group)
            p,imp=fit_meta(tr,te,cols,"collapse_target",1200000+year*100+gi)
            score=te[["_race_id","_horse_id","_is_top3","_is_win"]].copy()
            score["collapse_risk"]=p
            out[year][group]=score
            for rank,(feature,gain) in enumerate(imp[:30],start=1):
                importance.append({
                    "test_year":year,"gate_group":group,
                    "importance_rank":rank,"feature":feature,"gain":float(gain),
                })
            print(
                f"GATE_RISK_READY year={year} group={group} train_years={'|'.join(map(str,prior))} "
                f"features={len(cols)}",
                flush=True,
            )
    return out,importance

def pair_choices_by_year(candidates):
    choices={}
    importance=[]
    for year in PAIR_TEST_YEARS:
        prior=[y for y in TEST_YEARS if y<year]
        train=pd.concat([candidates[y] for y in prior],ignore_index=True,copy=False).reset_index(drop=True)
        test=candidates[year].copy().reset_index(drop=True)
        base=baseline_choice(test)[["_race_id","_horse_id","_is_top3","_is_win",PRIMARY_SCORE]].copy()
        choices[year]={"BASELINE":base}
        for gi,group in enumerate(FEATURE_GROUPS):
            cols=group_columns(train,group)
            via,vib,ptop,putil,names,imp_top,imp_util,n_top,n_util=fit_pair_models(
                train,test,cols,1300000+year*100+gi*10
            )
            score_map={
                "PAIR_TOP3":tournament_scores(test,via,vib,ptop),
                "PAIR_UTILITY":tournament_scores(test,via,vib,putil),
            }
            score_map["PAIR_BLEND"]=(score_map["PAIR_TOP3"]+score_map["PAIR_UTILITY"])/2.0
            for resolver,score in score_map.items():
                picked=choose(test,score)[["_race_id","_horse_id","_is_top3","_is_win",PRIMARY_SCORE,"_resolver"]].copy()
                # Compute resolver advantage over the baseline horse inside the same race.
                tmp=test[["_race_id","_horse_id"]].copy()
                tmp["_resolver"]=score
                base_ids=base[["_race_id","_horse_id"]].rename(columns={"_horse_id":"_base_horse"})
                tmp=tmp.merge(base_ids,on="_race_id",how="inner")
                base_scores=tmp[tmp["_horse_id"]==tmp["_base_horse"]][["_race_id","_resolver"]].rename(
                    columns={"_resolver":"_base_resolver"}
                )
                picked=picked.merge(base_scores,on="_race_id",how="left")
                picked["pair_margin"]=(picked["_resolver"]-picked["_base_resolver"]).fillna(0.0)
                choices[year][(group,resolver)]=picked
            for task,imp in (("PAIR_TOP3",imp_top),("PAIR_UTILITY",imp_util)):
                for rank,(feature,gain) in enumerate(imp[:30],start=1):
                    importance.append({
                        "test_year":year,"pair_group":group,"task":task,
                        "importance_rank":rank,"feature":feature,"gain":float(gain),
                    })
            print(
                f"GATE_PAIR_READY year={year} group={group} train_years={'|'.join(map(str,prior))} "
                f"features={len(cols)} top3_pairs={n_top} utility_pairs={n_util}",
                flush=True,
            )
            gc.collect()
    return choices,importance

def risk_gate_set(risk_df,fraction):
    n=len(risk_df)
    k=max(1,int(math.ceil(n*float(fraction))))
    ranked=risk_df.sort_values(["collapse_risk","_race_id"],ascending=[False,True])
    return set(ranked.head(k)["_race_id"].astype(str))

def gated_choice(baseline,pair,risk_df,fraction,margin_min):
    base=baseline.set_index("_race_id").copy()
    alt=pair.set_index("_race_id").copy()
    risky=risk_gate_set(risk_df,fraction)
    common=base.index.intersection(alt.index)
    out=base.loc[common].copy()
    pair_aligned=alt.loc[common]
    eligible=np.array([
        (rid in risky)
        and (float(pair_aligned.loc[rid,"pair_margin"])>=float(margin_min))
        and (str(pair_aligned.loc[rid,"_horse_id"])!=str(out.loc[rid,"_horse_id"]))
        for rid in common
    ],dtype=bool)
    for col in ("_horse_id","_is_top3","_is_win",PRIMARY_SCORE):
        out.loc[eligible,col]=pair_aligned.loc[eligible,col].to_numpy()
    out["intervened"]=eligible.astype("int8")
    out["collapse_risk"]=risk_df.set_index("_race_id").loc[common,"collapse_risk"].to_numpy()
    return out.reset_index()

def metrics(chosen,baseline,cand):
    base=baseline.set_index("_race_id")
    sel=chosen.set_index("_race_id")
    common=base.index.intersection(sel.index)
    base=base.loc[common]
    sel=sel.loc[common]
    b=base["_is_top3"].astype(int)
    s=sel["_is_top3"].astype(int)
    changed=base["_horse_id"].astype(str).ne(sel["_horse_id"].astype(str))
    rescue=((b==0)&(s==1))
    damage=((b==1)&(s==0))
    intervention=sel["intervened"].astype(int) if "intervened" in sel.columns else changed.astype(int)

    nunique=cand.groupby("_race_id").size()
    disagree=nunique[nunique>1].index
    dcommon=common.intersection(disagree)
    db=base.loc[dcommon]
    ds=sel.loc[dcommon]

    return {
        "races":int(len(common)),
        "top1_top3_pct":100*float(s.mean()),
        "top1_win_pct":100*float(sel["_is_win"].mean()),
        "uplift_pp":100*float((s-b).mean()),
        "changed_pct":100*float(changed.mean()),
        "intervention_pct":100*float(intervention.mean()),
        "rescued_baseline_miss_pct":100*float(rescue.sum()/max(1,int((b==0).sum()))),
        "damaged_baseline_hit_pct":100*float(damage.sum()/max(1,int((b==1).sum()))),
        "net_rescues":int(rescue.sum()-damage.sum()),
        "disagreement_races":int(len(dcommon)),
        "disagreement_top3_pct":100*float(ds["_is_top3"].mean()) if len(ds) else float("nan"),
        "disagreement_baseline_top3_pct":100*float(db["_is_top3"].mean()) if len(db) else float("nan"),
        "disagreement_uplift_pp":100*float(
            (ds["_is_top3"].astype(int)-db["_is_top3"].astype(int)).mean()
        ) if len(ds) else float("nan"),
    }

def enumerate_strategies(candidates,risks,pairs):
    rows=[]
    for year in PAIR_TEST_YEARS:
        baseline=pairs[year]["BASELINE"]
        for gate_group in GATE_GROUPS:
            risk=risks[year][gate_group]
            for pair_group in FEATURE_GROUPS:
                for resolver in PAIR_HEADS:
                    alt=pairs[year][(pair_group,resolver)]
                    for frac in GATE_FRACTIONS:
                        for margin in PAIR_MARGIN_MINS:
                            picked=gated_choice(baseline,alt,risk,frac,margin)
                            rows.append({
                                "test_year":year,
                                "gate_group":gate_group,
                                "pair_group":pair_group,
                                "resolver":resolver,
                                "gate_fraction":frac,
                                "pair_margin_min":margin,
                                **metrics(picked,baseline,candidates[year]),
                            })
        print(f"GATED_GRID_READY year={year}",flush=True)
    return rows

def key_of(r):
    return (
        r["gate_group"],r["pair_group"],r["resolver"],
        float(r["gate_fraction"]),float(r["pair_margin_min"])
    )

def strict_policy(rows):
    selected=[]
    history=[]
    for year in PAIR_TEST_YEARS:
        current=[r for r in rows if r["test_year"]==year]
        if not history:
            chosen_key=DEFAULT_POLICY
            reason="PREDECLARED_DEFAULT"
        else:
            perf={}
            for r in history:
                perf.setdefault(key_of(r),[]).append(float(r["uplift_pp"]))
            chosen_key=max(
                perf,
                key=lambda k:(
                    float(np.mean(perf[k])),
                    -float(np.std(perf[k])),
                    -k[3],       # prefer narrower gate on ties
                    k==DEFAULT_POLICY,
                    k,
                )
            )
            reason="BEST_PRIOR_OOS_MEAN"
        match=[r for r in current if key_of(r)==chosen_key]
        if len(match)!=1:
            raise SystemExit(f"strict strategy missing year={year} key={chosen_key}")
        rec=dict(match[0])
        rec["selection_reason"]=reason
        selected.append(rec)
        history.extend(current)
    return selected

def aggregate(rows):
    out=[]
    keys=sorted({key_of(r) for r in rows})
    for key in keys:
        vals=[r for r in rows if key_of(r)==key]
        w=np.array([r["races"] for r in vals],dtype=float)
        rec={
            "gate_group":key[0],"pair_group":key[1],"resolver":key[2],
            "gate_fraction":key[3],"pair_margin_min":key[4],
            "folds":len(vals),"races":int(w.sum()),
        }
        for metric in (
            "top1_top3_pct","top1_win_pct","uplift_pp","changed_pct","intervention_pct",
            "rescued_baseline_miss_pct","damaged_baseline_hit_pct",
            "disagreement_top3_pct","disagreement_baseline_top3_pct","disagreement_uplift_pp",
        ):
            arr=np.array([r[metric] for r in vals],dtype=float)
            rec[metric]=float(np.average(arr,weights=w))
            rec[metric+"_worst"]=float(np.nanmin(arr))
            rec[metric+"_std"]=float(np.nanstd(arr))
        rec["net_rescues"]=int(sum(r["net_rescues"] for r in vals))
        out.append(rec)
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--year-file",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--out-dir",required=True)
    a=p.parse_args()

    paths={}
    for spec in a.year_file:
        y,path=spec.split(":",1)
        paths[int(y)]=path
    if set(paths)!=set(YEARS):
        raise SystemExit(f"year file mismatch {sorted(paths)}")
    if LOCKED_YEAR in paths:
        raise SystemExit("2026 sealed")

    print(json.dumps({
        "contract":"L1_GATED_PAIRWISE_V1_RUNTIME",
        "threads":THREADS,
        "primary":PRIMARY,
        "gate_groups":list(GATE_GROUPS),
        "pair_groups":list(FEATURE_GROUPS),
        "resolvers":list(PAIR_HEADS),
        "gate_fractions":list(GATE_FRACTIONS),
        "pair_margin_mins":list(PAIR_MARGIN_MINS),
        "default_policy":list(DEFAULT_POLICY),
        "2026_locked":True,
        "ability_uses_odds":False,
    },separators=(",",":")),flush=True)

    candidates,_=build_base_oos(paths)
    risks,risk_importance=risk_by_year(candidates)
    pairs,pair_importance=pair_choices_by_year(candidates)
    grid=enumerate_strategies(candidates,risks,pairs)
    pooled=aggregate(grid)
    policy=strict_policy(grid)

    total_races=sum(r["races"] for r in policy)
    strict_top3=sum(r["top1_top3_pct"]*r["races"] for r in policy)/total_races
    strict_uplift=sum(r["uplift_pp"]*r["races"] for r in policy)/total_races
    strict_changed=sum(r["changed_pct"]*r["races"] for r in policy)/total_races
    strict_rescue=sum(r["net_rescues"] for r in policy)
    best=max(pooled,key=lambda r:(r["uplift_pp"],r["top1_top3_pct"]))

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"gated-grid.csv",grid)
    write_csv(out/"pooled-gated-strategies.csv",pooled)
    write_csv(out/"strict-policy.csv",policy)
    write_csv(out/"gate-feature-importance.csv",risk_importance)
    write_csv(out/"pair-feature-importance.csv",pair_importance)

    (out/"summary.json").write_text(json.dumps({
        "contract":"L1_GATED_PAIRWISE_V1",
        "question":"Can collapse risk gate pairwise intervention tightly enough to preserve correct anchors and rescue only dangerous disagreements?",
        "strictness":[
            "All six candidate L1 heads are outer-fold OOS.",
            "Collapse model for year Y trains only on earlier outer-fold OOS years.",
            "Pairwise model for year Y trains only on earlier outer-fold OOS years.",
            "Gate/pair strategy for year Y is selected only from prior resolver OOS years; 2022 uses a predeclared default.",
            "Gate fraction uses only the current year's risk-score rank, never current labels.",
            "No odds or popularity are inputs.",
            "2026 is sealed."
        ],
        "grid":{
            "gate_groups":list(GATE_GROUPS),
            "pair_groups":list(FEATURE_GROUPS),
            "resolvers":list(PAIR_HEADS),
            "gate_fractions":list(GATE_FRACTIONS),
            "pair_margin_mins":list(PAIR_MARGIN_MINS),
        },
        "exploratory_best":best,
        "strict_policy":{
            "races":total_races,
            "top1_top3_pct":strict_top3,
            "uplift_pp":strict_uplift,
            "changed_pct":strict_changed,
            "net_rescues":strict_rescue,
        },
        "promotion_rule":"Require >=2.0pp strict-policy uplift and no material fold collapse.",
        "promotion":bool(strict_uplift>=2.0 and min(r["uplift_pp"] for r in policy)>-0.5),
        "ability_uses_odds":False,
        "2026_locked":True,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("===== STRICT POLICY =====")
    print((out/"strict-policy.csv").read_text())
    print("===== TOP POOLED =====")
    for row in sorted(pooled,key=lambda r:r["uplift_pp"],reverse=True)[:20]:
        print(json.dumps(row,separators=(",",":")))
    print("L1_GATED_PAIRWISE_V1_COMPLETE")

if __name__=="__main__":
    main()
