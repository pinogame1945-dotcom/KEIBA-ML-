#!/usr/bin/env python3
import argparse,gzip,json,math,statistics
from collections import defaultdict,Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import analyze_l15_5k_council_arena_v1 as arena

BASELINE="router_hard"
ROLES=arena.ROLES
FEATURE_CONTRACT="L15_ROLE_CANDIDATE_FEATURES_V2"
TEST_YEARS=(2023,2024,2025)
MAX_SURVIVORS=4

ALIASES={
    "core4_no_pedigree":"core4",
    "no_auto_full_pedigree_legacy":"pedlegacy",
    "jockey_trainer_condition":"condition",
    "no_auto_full":"full",
    "no_auto_jockey":"light",
}

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--arena",required=True)
    p.add_argument("--role",required=True,choices=ROLES)
    p.add_argument("--year-feature",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--year-snapshot",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--year-pred",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--out-dir",required=True)
    p.add_argument("--summary-out",required=True)
    return p.parse_args()

def parse_map(items):
    out={}
    for spec in items:
        y,path=spec.split(":",1)
        out[int(y)]=path
    return out

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def finite(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None

def scalarize(prefix,obj,out,strings=True):
    if not isinstance(obj,dict):
        return
    for k,v in obj.items():
        key=f"{prefix}_{k}"
        if isinstance(v,bool):
            out[key]=1.0 if v else 0.0
        elif isinstance(v,(int,float)):
            x=finite(v)
            if x is not None:
                out[key]=x
        elif isinstance(v,str) and strings:
            out[key]=v
        elif isinstance(v,dict):
            scalarize(key,v,out,strings=strings)

def jaccard(a,b):
    a=set(a); b=set(b); u=a|b
    return len(a&b)/len(u) if u else 1.0

def iter_races_state(path):
    current=None
    groups=defaultdict(dict)
    seen=set()
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row=json.loads(line)
            if row.get("contract")!=FEATURE_CONTRACT:
                raise ValueError(f"unexpected feature contract in {path}")
            expert=str(row["expert_name"])
            if expert not in arena.EXPERTS:
                continue
            rid=str(row["race_id"])
            if current is None:
                current=rid
            if rid!=current:
                if rid in seen:
                    raise ValueError(f"race_id reappeared after flush: {rid}")
                seen.add(current)
                yield current,groups
                current=rid
                groups=defaultdict(dict)
            cell=(str(row["role"]),int(row["top_n"]))
            if expert in groups[cell]:
                raise ValueError(f"duplicate expert row {rid} {cell} {expert}")
            horses=tuple(map(str,row.get("candidate_horse_ids") or []))
            if len(horses)!=int(row["top_n"]):
                raise ValueError(f"candidate size mismatch {rid} {cell} {expert}")
            groups[cell][expert]={"horses":horses,"row":row}
    if current is not None:
        yield current,groups

def arena_by_year(arena_json):
    out={}
    for fold in arena_json["folds"]:
        out[int(fold["test_year"])]=fold["strategies"]
    return out

def pool_for(test_year,cell,fold_metrics):
    years=list(range(2022,test_year))
    if not years:
        return [BASELINE]
    need=max(1,math.ceil(len(years)/2))
    candidates=[]
    for strategy in arena.STRATEGIES:
        if strategy==BASELINE:
            continue
        vals=[]
        for y in years:
            vals.append(float(fold_metrics[y][strategy][cell]["delta_vs_router_pp"]))
        wins=sum(1 for x in vals if x>1e-12)
        avg=sum(vals)/len(vals)
        worst=min(vals)
        sd=statistics.pstdev(vals) if len(vals)>1 else 0.0
        if wins>=need and avg>0:
            candidates.append((strategy,wins,avg,worst,sd))
    candidates.sort(key=lambda x:(-x[1],-x[2],-x[3],x[4],x[0]))
    return [BASELINE]+[x[0] for x in candidates[:MAX_SURVIVORS]]

def hist_rate(expert_stats,cell,expert):
    st=expert_stats[cell][expert]
    return st["hits"]/st["races"] if st["races"] else 0.0

def candidate_support_features(candidate,lists,hard_candidate):
    supports=Counter()
    for horses in lists.values():
        supports.update(set(horses))
    vals=[supports[h] for h in candidate]
    return {
        "strategy_candidate_support_mean":sum(vals)/len(vals) if vals else 0.0,
        "strategy_candidate_support_min":min(vals) if vals else 0.0,
        "strategy_candidate_support_max":max(vals) if vals else 0.0,
        "strategy_candidate_unanimous_count":sum(1 for x in vals if x==len(lists)),
        "strategy_candidate_hard_jaccard":jaccard(candidate,hard_candidate),
        "strategy_candidate_unique_count":len(set(candidate)),
    }

def common_state(rows,lists,preds,expert_stats,cell):
    first=rows[sorted(rows)[0]]["row"]
    out={}
    scalarize("race",first.get("race") or {},out,strings=True)
    scalarize("coverage",first.get("data_coverage") or {},out,strings=False)
    scalarize("consensus",first.get("consensus") or {},out,strings=False)

    pair=[]
    experts=list(arena.EXPERTS)
    for i,a in enumerate(experts):
        for b in experts[i+1:]:
            pair.append(jaccard(lists[a],lists[b]))
    union=set().union(*(set(x) for x in lists.values()))
    supports=Counter()
    for horses in lists.values():
        supports.update(set(horses))
    out.update({
        "council_pair_jaccard_mean":sum(pair)/len(pair) if pair else 1.0,
        "council_pair_jaccard_min":min(pair) if pair else 1.0,
        "council_pair_jaccard_max":max(pair) if pair else 1.0,
        "council_union_size":len(union),
        "council_max_horse_support":max(supports.values()) if supports else 0,
        "council_mean_horse_support":sum(supports.values())/len(supports) if supports else 0.0,
        "council_unanimous_horse_count":sum(1 for x in supports.values() if x==len(experts)),
    })

    ordered=sorted(experts,key=lambda e:(-preds[e],e))
    pvals=[preds[e] for e in ordered]
    out["hard_expert"]=ordered[0]
    out["router_p_max"]=pvals[0]
    out["router_p_second"]=pvals[1]
    out["router_p_gap"]=pvals[0]-pvals[1]
    out["router_p_mean"]=sum(pvals)/len(pvals)
    out["router_p_std"]=statistics.pstdev(pvals) if len(pvals)>1 else 0.0
    s=sum(max(x,1e-12) for x in pvals)
    q=[max(x,1e-12)/s for x in pvals]
    out["router_p_entropy"]=-sum(x*math.log(x) for x in q)/math.log(len(q))

    for expert in experts:
        alias=ALIASES[expert]
        out[f"router_p_{alias}"]=preds[expert]
        out[f"hist_hit_rate_{alias}"]=hist_rate(expert_stats,cell,expert)
        row=rows[expert]["row"]
        scalarize(f"expert_summary_{alias}",row.get("expert_summary") or {},out,strings=False)
        scalarize(f"candidate_summary_{alias}",row.get("candidate_summary") or {},out,strings=False)
    return out,ordered[0]

def strategy_row(year,rid,cell,strategy,candidate,meta,hit,common,lists,preds,expert_stats,hard_expert):
    role,top_n=cell
    out=dict(common)
    out.update({
        "year":year,
        "race_id":rid,
        "role":role,
        "top_n":top_n,
        "strategy":strategy,
        "role_hit":int(hit),
        "candidate_horse_ids":list(candidate),
    })
    selected=meta.get("experts") or []
    out["strategy_expert_count"]=len(selected)
    out["strategy_selected_router_p_mean"]=sum(preds[e] for e in selected)/len(selected) if selected else 0.0
    out["strategy_selected_router_p_max"]=max((preds[e] for e in selected),default=0.0)
    out["strategy_selected_hist_rate_mean"]=sum(hist_rate(expert_stats,cell,e) for e in selected)/len(selected) if selected else 0.0
    out["strategy_selected_hist_rate_max"]=max((hist_rate(expert_stats,cell,e) for e in selected),default=0.0)
    out.update(candidate_support_features(candidate,lists,lists[hard_expert]))
    return out

def build_records(features,snapshots,preds_map,role_filter):
    expert_stats=defaultdict(lambda:defaultdict(lambda:{"hits":0,"races":0}))
    fusion_stats=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:{"hits":0,"races":0})))
    records=defaultdict(lambda:defaultdict(list))

    for year in range(2021,2026):
        truth=arena.read_truth(snapshots[year])
        predictions=arena.read_predictions(preds_map[year]) if year>=2022 else None
        races=0
        for rid,groups in iter_races_state(features[year]):
            races+=1
            outcomes=truth.get(rid)
            if not outcomes:
                raise ValueError(f"truth missing y{year} race={rid}")
            for cell,rows in groups.items():
                role,top_n=cell
                if role != role_filter:
                    continue
                if set(rows)!=set(arena.EXPERTS):
                    raise ValueError(f"expert coverage mismatch y{year} race={rid} cell={cell}")
                lists={e:rows[e]["horses"] for e in arena.EXPERTS}

                if year>=2022:
                    pp=predictions.get((rid,role,top_n))
                    if pp is None or set(pp)!=set(arena.EXPERTS):
                        raise ValueError(f"prediction coverage mismatch y{year} {(rid,role,top_n)}")
                    common,hard_expert=common_state(rows,lists,pp,expert_stats,cell)
                    for strategy in arena.STRATEGIES:
                        cand,meta=arena.strategy_candidate(
                            strategy,lists,top_n,pp,expert_stats,cell,fusion_stats
                        )
                        hit=arena.hit_for(role,cand,outcomes)
                        records[year][cell].append(
                            strategy_row(year,rid,cell,strategy,cand,meta,hit,common,lists,pp,expert_stats,hard_expert)
                        )

                for expert in arena.EXPERTS:
                    h=arena.hit_for(role,lists[expert],outcomes)
                    arena.add_stat(expert_stats[cell][expert],h)
                for kind in arena.FUSIONS:
                    for subset in arena.SUBSETS:
                        cand=arena.fuse(lists,subset,top_n,kind)
                        h=arena.hit_for(role,cand,outcomes)
                        arena.add_stat(fusion_stats[cell][kind][arena.subset_key(subset)],h)
        print(f"COUNCIL_V3_RECORDS_YEAR_OK year={year} races={races}",flush=True)
    return records

def encode(train,test):
    drop={"year","race_id","role","top_n","role_hit","candidate_horse_ids"}
    tr=pd.DataFrame(train)
    te=pd.DataFrame(test)
    ytr=tr["role_hit"].astype(int).to_numpy()
    yte=te["role_hit"].astype(int).to_numpy()

    trX=tr.drop(columns=[c for c in drop if c in tr],errors="ignore").copy()
    teX=te.drop(columns=[c for c in drop if c in te],errors="ignore").copy()

    cat_cols=[]
    for c in trX.columns:
        if trX[c].dtype=="object":
            cat_cols.append(c)
    both=pd.concat([trX,teX],ignore_index=True)
    both=pd.get_dummies(both,columns=cat_cols,dummy_na=False,dtype=float)
    both=both.replace([np.inf,-np.inf],np.nan).fillna(-999.0)
    Xtr=both.iloc[:len(trX)].reset_index(drop=True)
    Xte=both.iloc[len(trX):].reset_index(drop=True)
    return tr,te,Xtr,Xte,ytr,yte

def fallback_scores(train_rows,test_rows):
    by=defaultdict(list)
    for r in train_rows:
        by[r["strategy"]].append(r["role_hit"])
    rates={s:(sum(v)/len(v) if v else 0.0) for s,v in by.items()}
    return np.array([rates.get(r["strategy"],0.0) for r in test_rows],dtype=float),rates

def fit_predict(train_rows,test_rows,model_path):
    tr,te,Xtr,Xte,ytr,yte=encode(train_rows,test_rows)
    if len(set(map(int,ytr)))<2:
        p,rates=fallback_scores(train_rows,test_rows)
        return tr,te,p,{"mode":"fallback","strategy_rates":rates,"feature_importance":[]}

    model=lgb.LGBMClassifier(
        objective="binary",
        n_estimators=220,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=100,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=1945,
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(Xtr,ytr)
    p=model.predict_proba(Xte)[:,1]
    model.booster_.save_model(str(model_path))
    gains=model.booster_.feature_importance(importance_type="gain")
    names=model.booster_.feature_name()
    top=sorted(zip(names,gains),key=lambda x:-x[1])[:25]
    return tr,te,p,{
        "mode":"lightgbm",
        "feature_count":int(Xtr.shape[1]),
        "feature_importance":[{"feature":n,"gain":float(g)} for n,g in top if g>0],
    }

def evaluate_cell(test_year,cell,pool,train_rows,test_rows,outdir,decision_fh):
    train=[r for r in train_rows if r["strategy"] in pool]
    test=[r for r in test_rows if r["strategy"] in pool]
    role,top_n=cell

    if pool==[BASELINE]:
        grouped=defaultdict(list)
        for r in test:
            grouped[r["race_id"]].append(r)
        decisions=[]
        for rid,rows in grouped.items():
            r=rows[0]
            decisions.append({
                "race_id":rid,"chosen_strategy":BASELINE,"predicted_hit_probability":None,
                "strategy_margin":None,"chosen_hit":r["role_hit"],"baseline_hit":r["role_hit"],
                "oracle_hit":r["role_hit"],"candidate_horse_ids":r["candidate_horse_ids"],
            })
        model_info={"mode":"baseline_only","feature_importance":[]}
    else:
        safe_role=role.lower()
        model_path=outdir/f"y{test_year}-{safe_role}-top{top_n}.txt"
        trdf,tedf,p,model_info=fit_predict(train,test,model_path)
        work=tedf.copy()
        work["predicted_hit_probability"]=p
        decisions=[]
        for rid,g in work.groupby("race_id",sort=True):
            g=g.sort_values(["predicted_hit_probability","strategy"],ascending=[False,True])
            chosen=g.iloc[0]
            second=float(g.iloc[1]["predicted_hit_probability"]) if len(g)>1 else None
            base=g[g["strategy"]==BASELINE]
            if len(base)!=1:
                raise ValueError(f"baseline missing/duplicate {test_year} {cell} {rid}")
            oracle=int(g["role_hit"].max())
            decisions.append({
                "race_id":str(rid),
                "chosen_strategy":str(chosen["strategy"]),
                "predicted_hit_probability":float(chosen["predicted_hit_probability"]),
                "strategy_margin":float(chosen["predicted_hit_probability"]-second) if second is not None else None,
                "chosen_hit":int(chosen["role_hit"]),
                "baseline_hit":int(base.iloc[0]["role_hit"]),
                "oracle_hit":oracle,
                "candidate_horse_ids":list(chosen["candidate_horse_ids"]),
            })

    for d in decisions:
        decision_fh.write(json.dumps({
            "contract":"L15_COUNCIL_ROUTER_V3_DECISION",
            "test_year":test_year,
            "role":role,
            "top_n":top_n,
            "candidate_pool":pool,
            **d,
        },ensure_ascii=False,separators=(",",":"))+"\n")

    races=len(decisions)
    chosen_hits=sum(x["chosen_hit"] for x in decisions)
    base_hits=sum(x["baseline_hit"] for x in decisions)
    oracle_hits=sum(x["oracle_hit"] for x in decisions)
    choices=Counter(x["chosen_strategy"] for x in decisions)

    fixed={}
    for s in pool:
        rows=[r for r in test if r["strategy"]==s]
        fixed[s]={
            "hits":sum(r["role_hit"] for r in rows),
            "races":len(rows),
            "hit_rate":sum(r["role_hit"] for r in rows)/len(rows) if rows else None,
        }
    best_fixed=max(fixed.items(),key=lambda kv:(kv[1]["hit_rate"],kv[0]))

    return {
        "test_year":test_year,
        "train_years":list(range(2022,test_year)),
        "role":role,
        "top_n":top_n,
        "candidate_pool":pool,
        "races":races,
        "router_v3_hits":chosen_hits,
        "router_v3_hit_rate":chosen_hits/races if races else None,
        "baseline_hits":base_hits,
        "baseline_hit_rate":base_hits/races if races else None,
        "delta_vs_baseline_pp":(chosen_hits-base_hits)*100/races if races else None,
        "oracle_hits":oracle_hits,
        "oracle_hit_rate":oracle_hits/races if races else None,
        "oracle_gap_pp":(oracle_hits-chosen_hits)*100/races if races else None,
        "choice_counts":dict(sorted(choices.items())),
        "best_fixed_hindsight":{
            "strategy":best_fixed[0],
            **best_fixed[1],
            "delta_v3_minus_best_fixed_pp":(chosen_hits-best_fixed[1]["hits"])*100/races if races else None,
        },
        "model":model_info,
    }

def main():
    a=parse_args()
    features=parse_map(a.year_feature)
    snapshots=parse_map(a.year_snapshot)
    preds_map=parse_map(a.year_pred)
    if set(features)!={2021,2022,2023,2024,2025}:
        raise ValueError(f"feature years mismatch: {sorted(features)}")
    if set(snapshots)!={2021,2022,2023,2024,2025}:
        raise ValueError(f"snapshot years mismatch: {sorted(snapshots)}")
    if set(preds_map)!={2022,2023,2024,2025}:
        raise ValueError(f"prediction years mismatch: {sorted(preds_map)}")

    arena_json=json.loads(Path(a.arena).read_text(encoding="utf-8"))
    if arena_json.get("contract")!="L15_5K_COUNCIL_ARENA_V1":
        raise ValueError("unexpected arena contract")
    fold_metrics=arena_by_year(arena_json)

    outdir=Path(a.out_dir); outdir.mkdir(parents=True,exist_ok=True)
    records=build_records(features,snapshots,preds_map,a.role)

    decisions_path=outdir/"decisions.jsonl.gz"
    folds=[]
    aggregate=defaultdict(lambda:{"races":0,"v3":0,"base":0,"oracle":0})
    with gzip.open(decisions_path,"wt",encoding="utf-8") as dfh:
        cells=sorted(records[2022])
        for test_year in TEST_YEARS:
            for cell in cells:
                cell_name=f"{cell[0]}|{cell[1]}"
                pool=pool_for(test_year,cell_name,fold_metrics)
                train_rows=[]
                for y in range(2022,test_year):
                    train_rows.extend(records[y][cell])
                test_rows=records[test_year][cell]
                result=evaluate_cell(test_year,cell,pool,train_rows,test_rows,outdir,dfh)
                folds.append(result)
                arow=aggregate[cell_name]
                arow["races"]+=result["races"]
                arow["v3"]+=result["router_v3_hits"]
                arow["base"]+=result["baseline_hits"]
                arow["oracle"]+=result["oracle_hits"]
                print(
                    "COUNCIL_V3_FOLD",
                    test_year,cell_name,
                    "pool",",".join(pool),
                    "v3",round(result["router_v3_hit_rate"]*100,3),
                    "base",round(result["baseline_hit_rate"]*100,3),
                    "delta_pp",round(result["delta_vs_baseline_pp"],3),
                    "oracle_gap_pp",round(result["oracle_gap_pp"],3),
                    "choices",json.dumps(result["choice_counts"],separators=(",",":")),
                    flush=True,
                )

    agg_out={}
    total={"races":0,"v3":0,"base":0,"oracle":0}
    for cell_name,d in sorted(aggregate.items()):
        races=d["races"]
        agg_out[cell_name]={
            "races":races,
            "router_v3_hit_rate":d["v3"]/races,
            "baseline_hit_rate":d["base"]/races,
            "delta_vs_baseline_pp":(d["v3"]-d["base"])*100/races,
            "oracle_hit_rate":d["oracle"]/races,
            "oracle_gap_pp":(d["oracle"]-d["v3"])*100/races,
        }
        for k in total:
            total[k]+=d[k]

    summary={
        "contract":"L15_COUNCIL_ROUTER_V3",
        "business_objective":{
            "primary":"profit_maximization",
            "note":"Council Router V3 improves candidate construction only. Final adoption requires downstream L2/L3 EV/ROI validation.",
        },
        "role":a.role,
        "protocol":{
            "test_years":list(TEST_YEARS),
            "parallel_lane":"one role per standard CPU runner",
            "candidate_pool_selection":"For each test year, select up to four non-baseline council strategies using only earlier OOS arena years; router_hard is always included.",
            "pool_rule":"positive mean delta and positive in at least ceil(prior_year_count/2) prior years; rank by wins, mean, worst-year, volatility.",
            "model":"separate LightGBM binary hit-probability router per role/top_n/test fold",
            "locked_years":[2026],
            "odds_used":False,
        },
        "folds":folds,
        "aggregate_by_cell":agg_out,
        "aggregate_all_cells":{
            "races":total["races"],
            "router_v3_hit_rate":total["v3"]/total["races"],
            "baseline_hit_rate":total["base"]/total["races"],
            "delta_vs_baseline_pp":(total["v3"]-total["base"])*100/total["races"],
            "oracle_hit_rate":total["oracle"]/total["races"],
            "oracle_gap_pp":(total["oracle"]-total["v3"])*100/total["races"],
        },
        "decisions":str(decisions_path),
        "next_stage":{
            "if_stable":"Integrate Council Router V3 output contract, then re-run RC rescue after Council output and pass confidence/strategy metadata to L2.",
            "profit_gate":"L2/L3 must evaluate odds, edge, EV, ticket construction, bankroll and realized ROI before production promotion.",
        },
        "notes":[
            "2026 is never read.",
            "No odds are used.",
            "Strategy candidate pools are selected without future-year lookahead.",
            "Oracle is hindsight-only and used only to measure remaining routing headroom.",
        ],
    }
    Path(a.summary_out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.summary_out).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("L15_COUNCIL_ROUTER_V3_OK",a.role)
    print("AGGREGATE_ALL",
          "v3",round(summary["aggregate_all_cells"]["router_v3_hit_rate"]*100,3),
          "base",round(summary["aggregate_all_cells"]["baseline_hit_rate"]*100,3),
          "delta_pp",round(summary["aggregate_all_cells"]["delta_vs_baseline_pp"],3),
          "oracle_gap_pp",round(summary["aggregate_all_cells"]["oracle_gap_pp"],3))
    for cell,v in agg_out.items():
        print("AGGREGATE_CELL",cell,
              "v3",round(v["router_v3_hit_rate"]*100,3),
              "base",round(v["baseline_hit_rate"]*100,3),
              "delta_pp",round(v["delta_vs_baseline_pp"],3),
              "oracle_gap_pp",round(v["oracle_gap_pp"],3))

if __name__=="__main__":
    main()
