#!/usr/bin/env python3
import argparse,gzip,itertools,json,math
from collections import defaultdict
from pathlib import Path

EXPERTS=(
    "core4_no_pedigree",
    "no_auto_full_pedigree_legacy",
    "jockey_trainer_condition",
    "no_auto_full",
    "no_auto_jockey",
)
ROLES=("ANCHOR","MAINLINE","COVER")
FEATURE_CONTRACT="L15_ROLE_CANDIDATE_FEATURES_V2"
PRED_CONTRACT="L15_ROLE_ROUTER_MODEL_V2"
FUSIONS=("vote","borda","rrf")
SUBSETS=tuple(
    tuple(c)
    for k in range(1,len(EXPERTS)+1)
    for c in itertools.combinations(EXPERTS,k)
)
STRATEGIES=(
    "router_hard",
    "train_best_single",
    "all5_vote",
    "all5_borda",
    "all5_rrf",
    "best_subset_vote",
    "best_subset_borda",
    "best_subset_rrf",
    "hist_weighted_vote",
    "hist_weighted_borda",
    "hist_weighted_rrf",
    "router_weighted_vote",
    "router_weighted_borda",
    "router_weighted_rrf",
    "router_top2_borda",
    "router_top3_borda",
    "hybrid_hist_router_borda",
    "hybrid_hist_router_rrf",
)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--year-feature",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--year-snapshot",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--year-pred",action="append",default=[],help="YEAR:PATH (2022-2025)")
    p.add_argument("--output",required=True)
    return p.parse_args()

def parse_map(items):
    out={}
    for spec in items:
        y,path=spec.split(":",1)
        out[int(y)]=path
    return out

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def ordered_outcomes(groups,slots=3):
    pieces=[()]
    for rank in sorted(groups):
        if rank>slots:
            continue
        horses=list(groups[rank])
        occupied=min(len(horses),slots-rank+1)
        if occupied<=0:
            continue
        perms=list(itertools.permutations(horses,occupied))
        pieces=[a+b for a in pieces for b in perms]
    return sorted(set(x for x in pieces if len(x)==slots))

def read_truth(path):
    groups=defaultdict(lambda:defaultdict(list))
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            hid=str(row.get("horse_id") or "")
            target=row.get("target") or {}
            try:
                finish=int(float(target.get("finish_position")))
            except (TypeError,ValueError):
                continue
            if rid and hid and finish<=3:
                groups[rid][finish].append(hid)
    out={}
    for rid,g in groups.items():
        outcomes=ordered_outcomes(g,3)
        if outcomes:
            out[rid]=outcomes
    return out

def hit_for(role,cand,outcomes):
    cand=set(map(str,cand))
    if role=="ANCHOR":
        return int(any(o[0] in cand for o in outcomes))
    if role=="MAINLINE":
        return int(any(bool(cand.intersection(o[1:3])) for o in outcomes))
    if role=="COVER":
        return int(any(set(o[1:3]).issubset(cand) for o in outcomes))
    raise ValueError(role)

def iter_races(path):
    current=None
    groups=defaultdict(dict)
    seen=set()
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row=json.loads(line)
            if row.get("contract")!=FEATURE_CONTRACT:
                raise ValueError(f"bad feature contract in {path}")
            expert=str(row["expert_name"])
            if expert not in EXPERTS:
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
            key=(str(row["role"]),int(row["top_n"]))
            if expert in groups[key]:
                raise ValueError(f"duplicate expert row {rid} {key} {expert}")
            horses=tuple(map(str,row.get("candidate_horse_ids") or []))
            if len(horses)!=int(row["top_n"]):
                raise ValueError(f"candidate size mismatch {rid} {key} {expert}")
            groups[key][expert]=horses
    if current is not None:
        yield current,groups

def read_predictions(path):
    out=defaultdict(dict)
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r=json.loads(line)
            expert=str(r["expert_name"])
            if expert not in EXPERTS:
                continue
            key=(str(r["race_id"]),str(r["role"]),int(float(r["top_n"])))
            if expert in out[key]:
                raise ValueError(f"duplicate prediction {key} {expert}")
            out[key][expert]=float(r["predicted_role_hit_probability"])
    return out

def fuse(lists,selected,top_n,kind,weights=None):
    scores=defaultdict(float)
    votes=defaultdict(float)
    rrfs=defaultdict(float)
    counts=defaultdict(int)
    best_rank={}
    for expert in selected:
        horses=lists[expert]
        w=1.0 if weights is None else float(weights.get(expert,0.0))
        for i,hid in enumerate(horses,1):
            counts[hid]+=1
            votes[hid]+=w
            rrfs[hid]+=1.0/i
            best_rank[hid]=min(best_rank.get(hid,10**9),i)
            if kind=="vote":
                scores[hid]+=w
            elif kind=="borda":
                scores[hid]+=w*(top_n-i+1)
            elif kind=="rrf":
                scores[hid]+=w/i
            else:
                raise ValueError(kind)
    ranked=sorted(
        scores,
        key=lambda h:(-scores[h],-votes[h],-counts[h],-rrfs[h],best_rank[h],h)
    )
    if len(ranked)<top_n:
        raise ValueError(f"fusion produced {len(ranked)} < {top_n}")
    return tuple(ranked[:top_n])

def subset_key(subset):
    return "|".join(subset)

def train_rate(stat):
    return stat["hits"]/stat["races"] if stat["races"] else 0.0

def choose_best_single(expert_stats,cell):
    ranked=[]
    for expert in EXPERTS:
        st=expert_stats[cell][expert]
        ranked.append((train_rate(st),expert))
    ranked.sort(key=lambda x:(-x[0],x[1]))
    return ranked[0][1]

def choose_best_subset(fusion_stats,cell,kind):
    ranked=[]
    for subset in SUBSETS:
        st=fusion_stats[cell][kind][subset_key(subset)]
        rate=train_rate(st)
        ranked.append((rate,len(subset),subset_key(subset),subset))
    ranked.sort(key=lambda x:(-x[0],x[1],x[2]))
    return ranked[0][3]

def hist_weights(expert_stats,cell):
    rates={e:train_rate(expert_stats[cell][e]) for e in EXPERTS}
    mean=sum(rates.values())/len(rates)
    # Small shrinkage prevents one expert from receiving near-zero weight.
    return {e:0.25*mean+0.75*rates[e] for e in EXPERTS}

def add_stat(d,hit):
    d["races"]+=1
    d["hits"]+=int(hit)

def strategy_candidate(name,lists,top_n,preds,expert_stats,cell,fusion_stats):
    all5=EXPERTS
    hw=hist_weights(expert_stats,cell)

    if name=="router_hard":
        e=sorted(EXPERTS,key=lambda x:(-preds[x],x))[0]
        return lists[e],{"experts":[e]}
    if name=="train_best_single":
        e=choose_best_single(expert_stats,cell)
        return lists[e],{"experts":[e]}
    if name.startswith("all5_"):
        kind=name.split("_",1)[1]
        return fuse(lists,all5,top_n,kind),{"experts":list(all5)}
    if name.startswith("best_subset_"):
        kind=name.rsplit("_",1)[1]
        sub=choose_best_subset(fusion_stats,cell,kind)
        return fuse(lists,sub,top_n,kind),{"experts":list(sub)}
    if name.startswith("hist_weighted_"):
        kind=name.rsplit("_",1)[1]
        return fuse(lists,all5,top_n,kind,hw),{"experts":list(all5)}
    if name.startswith("router_weighted_"):
        kind=name.rsplit("_",1)[1]
        return fuse(lists,all5,top_n,kind,preds),{"experts":list(all5)}
    if name=="router_top2_borda":
        sub=tuple(sorted(EXPERTS,key=lambda x:(-preds[x],x))[:2])
        return fuse(lists,sub,top_n,"borda"),{"experts":list(sub)}
    if name=="router_top3_borda":
        sub=tuple(sorted(EXPERTS,key=lambda x:(-preds[x],x))[:3])
        return fuse(lists,sub,top_n,"borda"),{"experts":list(sub)}
    if name=="hybrid_hist_router_borda":
        w={e:hw[e]*preds[e] for e in EXPERTS}
        return fuse(lists,all5,top_n,"borda",w),{"experts":list(all5)}
    if name=="hybrid_hist_router_rrf":
        w={e:hw[e]*preds[e] for e in EXPERTS}
        return fuse(lists,all5,top_n,"rrf",w),{"experts":list(all5)}
    raise ValueError(name)

def main():
    a=parse_args()
    features=parse_map(a.year_feature)
    snapshots=parse_map(a.year_snapshot)
    preds_map=parse_map(a.year_pred)
    expected={2021,2022,2023,2024,2025}
    if set(features)!=expected or set(snapshots)!=expected:
        raise SystemExit(f"need 2021-2025 features/snapshots: features={sorted(features)} snapshots={sorted(snapshots)}")
    if set(preds_map)!={2022,2023,2024,2025}:
        raise SystemExit(f"need 2022-2025 predictions: got={sorted(preds_map)}")

    expert_stats=defaultdict(lambda:defaultdict(lambda:{"hits":0,"races":0}))
    fusion_stats=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:{"hits":0,"races":0})))
    folds=[]
    aggregate=defaultdict(lambda:defaultdict(lambda:{"hits":0,"races":0}))
    strategy_meta=defaultdict(lambda:defaultdict(lambda:defaultdict(int)))

    for year in range(2021,2026):
        truth=read_truth(snapshots[year])
        predictions=read_predictions(preds_map[year]) if year>=2022 else None
        fold_stats=defaultdict(lambda:defaultdict(lambda:{"hits":0,"races":0}))
        fold_meta=defaultdict(lambda:defaultdict(lambda:defaultdict(int)))

        if year>=2022:
            train_years=list(range(2021,year))
            policy_snapshot={}
            cells=sorted(expert_stats)
            for cell in cells:
                policy_snapshot[f"{cell[0]}|{cell[1]}"]={
                    "best_single":choose_best_single(expert_stats,cell),
                    "best_subset_vote":list(choose_best_subset(fusion_stats,cell,"vote")),
                    "best_subset_borda":list(choose_best_subset(fusion_stats,cell,"borda")),
                    "best_subset_rrf":list(choose_best_subset(fusion_stats,cell,"rrf")),
                    "historical_weights":hist_weights(expert_stats,cell),
                }
        else:
            train_years=[]
            policy_snapshot={}

        race_count=0
        group_count=0
        for rid,groups in iter_races(features[year]):
            race_count+=1
            outcomes=truth.get(rid)
            if not outcomes:
                raise ValueError(f"truth missing y{year} race={rid}")
            for cell,lists in groups.items():
                role,top_n=cell
                group_count+=1
                if set(lists)!=set(EXPERTS):
                    raise ValueError(f"expert coverage mismatch y{year} race={rid} cell={cell}: {sorted(lists)}")

                # Test current year before adding it to training accumulators.
                if year>=2022:
                    pkey=(rid,role,top_n)
                    pp=predictions.get(pkey)
                    if pp is None or set(pp)!=set(EXPERTS):
                        raise ValueError(f"prediction coverage mismatch y{year} {pkey}")
                    for strategy in STRATEGIES:
                        cand,meta=strategy_candidate(
                            strategy,lists,top_n,pp,expert_stats,cell,fusion_stats
                        )
                        hit=hit_for(role,cand,outcomes)
                        add_stat(fold_stats[strategy][cell],hit)
                        add_stat(aggregate[strategy][cell],hit)
                        for e in meta.get("experts",[]):
                            fold_meta[strategy][cell][e]+=1
                            strategy_meta[strategy][cell][e]+=1

                # Update training statistics after test evaluation.
                for expert in EXPERTS:
                    h=hit_for(role,lists[expert],outcomes)
                    add_stat(expert_stats[cell][expert],h)
                for kind in FUSIONS:
                    for subset in SUBSETS:
                        cand=fuse(lists,subset,top_n,kind)
                        h=hit_for(role,cand,outcomes)
                        add_stat(fusion_stats[cell][kind][subset_key(subset)],h)

        if year>=2022:
            cells_out={}
            for strategy in STRATEGIES:
                per={}
                for cell,st in sorted(fold_stats[strategy].items()):
                    role,n=cell
                    rate=st["hits"]/st["races"] if st["races"] else None
                    base=fold_stats["router_hard"][cell]
                    base_rate=base["hits"]/base["races"] if base["races"] else None
                    per[f"{role}|{n}"]={
                        "role":role,"top_n":n,"races":st["races"],"hits":st["hits"],
                        "hit_rate":rate,
                        "delta_vs_router_pp":(rate-base_rate)*100 if rate is not None else None,
                        "expert_use_counts":dict(sorted(fold_meta[strategy][cell].items())),
                    }
                cells_out[strategy]=per
            folds.append({
                "test_year":year,
                "train_years":train_years,
                "race_count":race_count,
                "group_count":group_count,
                "policies":policy_snapshot,
                "strategies":cells_out,
            })
        print(f"ARENA_YEAR_OK year={year} races={race_count} groups={group_count}",flush=True)

    aggregate_out={}
    stability={}
    for strategy in STRATEGIES:
        aggregate_out[strategy]={}
        stability[strategy]={}
        deltas=[]
        pos=neg=tie=0
        for cell,st in sorted(aggregate[strategy].items()):
            role,n=cell
            rate=st["hits"]/st["races"]
            b=aggregate["router_hard"][cell]
            br=b["hits"]/b["races"]
            delta=(rate-br)*100
            fold_signs=[]
            for fold in folds:
                cur=fold["strategies"][strategy][f"{role}|{n}"]
                d=cur["delta_vs_router_pp"]
                fold_signs.append(d)
            wins=sum(1 for d in fold_signs if d>1e-12)
            losses=sum(1 for d in fold_signs if d<-1e-12)
            ties=len(fold_signs)-wins-losses
            aggregate_out[strategy][f"{role}|{n}"]={
                "role":role,"top_n":n,"races":st["races"],"hits":st["hits"],
                "hit_rate":rate,"delta_vs_router_pp":delta,
                "fold_wins":wins,"fold_ties":ties,"fold_losses":losses,
                "expert_use_counts":dict(sorted(strategy_meta[strategy][cell].items())),
            }
            stability[strategy][f"{role}|{n}"]={"wins":wins,"ties":ties,"losses":losses}
            deltas.append(delta)
            if delta>1e-12: pos+=1
            elif delta<-1e-12: neg+=1
            else: tie+=1
        aggregate_out[strategy]["__summary__"]={
            "mean_cell_delta_vs_router_pp":sum(deltas)/len(deltas),
            "positive_cells":pos,"tie_cells":tie,"negative_cells":neg,
        }

    cell_winners={}
    cells=sorted(aggregate["router_hard"])
    for cell in cells:
        role,n=cell
        ranked=[]
        for strategy in STRATEGIES:
            st=aggregate[strategy][cell]
            rate=st["hits"]/st["races"]
            ranked.append((rate,strategy))
        ranked.sort(key=lambda x:(-x[0],x[1]))
        best=ranked[0][0]
        cell_winners[f"{role}|{n}"]={
            "best_hit_rate":best,
            "strategies":[s for r,s in ranked if abs(r-best)<1e-15],
            "router_hit_rate":dict(ranked)["router_hard"] if False else aggregate["router_hard"][cell]["hits"]/aggregate["router_hard"][cell]["races"],
        }

    out={
        "contract":"L15_5K_COUNCIL_ARENA_V1",
        "years":{
            "training_start":2021,
            "walk_forward":[
                {"train":[2021],"test":2022},
                {"train":[2021,2022],"test":2023},
                {"train":[2021,2022,2023],"test":2024},
                {"train":[2021,2022,2023,2024],"test":2025},
            ],
            "locked":[2026],
        },
        "experts":list(EXPERTS),
        "strategy_count":len(STRATEGIES),
        "strategies":list(STRATEGIES),
        "subset_search":{
            "expert_subsets":len(SUBSETS),
            "fusion_rules":list(FUSIONS),
            "candidate_policies":len(SUBSETS)*len(FUSIONS),
            "selection":"training years only, separately for each role/top_n and fold",
        },
        "folds":folds,
        "aggregate":aggregate_out,
        "cell_winners":cell_winners,
        "notes":[
            "No odds used.",
            "2026 never read.",
            "All fused candidate sets keep the same TopN cost as the compared baseline.",
            "candidate_horse_ids order is treated as the expert ranking within TopN.",
            "best_subset_* searches all 31 non-empty subsets of the five experts using only prior years.",
            "Aggregate 2022-2025 results are exploratory; fold stability is required before promotion.",
        ],
    }
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("L15_5K_COUNCIL_ARENA_V1_OK")
    for strategy in STRATEGIES:
        s=aggregate_out[strategy]["__summary__"]
        print("STRATEGY",strategy,
              "mean_delta_pp",round(s["mean_cell_delta_vs_router_pp"],3),
              "cells",f'{s["positive_cells"]}+/{s["tie_cells"]}=/{s["negative_cells"]}-')
    for cell,w in sorted(cell_winners.items()):
        print("CELL_WINNER",cell,
              "rate",round(w["best_hit_rate"]*100,3),
              "strategies",",".join(w["strategies"]),
              "router",round(w["router_hit_rate"]*100,3))

if __name__=="__main__":
    main()
