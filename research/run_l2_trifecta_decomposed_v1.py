#!/usr/bin/env python3
import argparse,csv,hashlib,itertools,json,math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_l2_trifecta_fullfield_ability_v2 import (
    YEARS, TRAIN_YEARS, TEST_YEARS, DEV_YEARS, HOLDOUT_YEAR,
    EXPECTED_RACES_PER_YEAR, HORSE_FEATURES,
    parse_paths, load_l17, iter_races, horse_vector, feature_vector, ordered_l17_score
)

TOPN_GRID=(1,2,3,5,10,20,30,60,120)
SET_L17_TOP=60
SET_MARKET_TOP=60
SET_HASH_NEG=60

SET_FEATURES=(
    tuple(f"h{i}_{name}" for i in (1,2,3) for name in HORSE_FEATURES)
    + (
        "field_size","rank_sum","rank_mean","rank_max","rank_span",
        "prob_sum","prob_mean","prob_product","prob_min","prob_max",
        "rank_std_mean","rank_std_max","top1_votes_sum","top3_support_sum","top6_support_sum",
    )
)
ORDER_FEATURES=(
    tuple(f"s{s}_{name}" for s in (1,2,3) for name in HORSE_FEATURES)
    + (
        "field_size","rank_sum","rank_max","rank_min","rank_span","rank_order_12","rank_order_23",
        "prob_sum","prob_product","prob_min","prob_max","prob_order_12","prob_order_23",
        "rank_std_max","rank_std_mean","top3_support_sum","top6_support_sum",
        "l17_order_score",
    )
)

def parse_args():
    p=argparse.ArgumentParser(description="Two-stage full-field trifecta: podium set then conditional order.")
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def set_key(nums):
    return tuple(sorted(int(x) for x in nums))

def canonical_horses(hs):
    return sorted(
        hs,
        key=lambda h:(int(h.get("consensus_rank",999)),int(h.get("horse_number",999)))
    )

def set_feature_vector(hs,field_size):
    a,b,c=canonical_horses(hs)
    va,vb,vc=horse_vector(a),horse_vector(b),horse_vector(c)
    ranks=[va[0],vb[0],vc[0]]
    probs=[va[8],vb[8],vc[8]]
    rstd=[va[2],vb[2],vc[2]]
    t1=[va[5],vb[5],vc[5]]
    t3=[va[6],vb[6],vc[6]]
    t6=[va[7],vb[7],vc[7]]
    return np.asarray(
        list(va)+list(vb)+list(vc)+[
            float(field_size),
            sum(ranks),sum(ranks)/3.0,max(ranks),max(ranks)-min(ranks),
            sum(probs),sum(probs)/3.0,probs[0]*probs[1]*probs[2],min(probs),max(probs),
            sum(rstd)/3.0,max(rstd),sum(t1),sum(t3),sum(t6),
        ],dtype=np.float32
    )

def build_unordered(race):
    horses=race["horses"]
    sets=[]
    by_key={}
    for hs in itertools.combinations(horses,3):
        nums=set_key([h["horse_number"] for h in hs])
        i=len(sets)
        sets.append((tuple(hs),nums))
        by_key[nums]=i
    positive_keys={set_key(race["combos"][int(i)][3]) for i in race["positive"]}
    positive_idx=sorted(by_key[k] for k in positive_keys if k in by_key)
    if len(positive_idx)!=len(positive_keys):
        raise RuntimeError(f"positive set mapping failed race={race['race_id']}")
    return sets,by_key,np.asarray(positive_idx,dtype=np.int64),positive_keys

def set_l17_score(hs):
    ps=[max(1e-9,min(.999999,float(h.get("mean_probability") or 0.0))) for h in hs]
    return ps[0]*ps[1]*ps[2]

def hash_negatives(race,sets,limit,excluded):
    vals=[]
    rid=race["race_id"]
    for i,(_,nums) in enumerate(sets):
        if i in excluded: continue
        h=int.from_bytes(hashlib.blake2b(f"{rid}:{nums}".encode(),digest_size=8).digest(),"big")
        vals.append((h,i))
    vals.sort()
    return [i for _,i in vals[:limit]]

def market_top_sets(race,by_key,limit):
    seen=set(); out=[]
    for ci in race["market_order"]:
        nums=set_key(race["combos"][int(ci)][3])
        si=by_key.get(nums)
        if si is None or si in seen: continue
        seen.add(si); out.append(si)
        if len(out)>=limit: break
    return out

def sample_set_year(year,l17_rows,root):
    xs=[]; ys=[]; counters=defaultdict(int)
    for race in iter_races(year,l17_rows,root,with_odds=True):
        if race["status"]!="ok":
            counters[race["status"]]+=1; continue
        sets,by_key,pos_idx,_=build_unordered(race)
        selected=set(int(i) for i in pos_idx)

        l17_order=sorted(
            range(len(sets)),
            key=lambda i:(-set_l17_score(sets[i][0]),sets[i][1])
        )
        selected.update(l17_order[:SET_L17_TOP])
        selected.update(market_top_sets(race,by_key,SET_MARKET_TOP))
        selected.update(hash_negatives(race,sets,SET_HASH_NEG,selected))

        idx=np.asarray(sorted(selected),dtype=np.int64)
        x=np.vstack([set_feature_vector(sets[int(i)][0],len(race["horses"])) for i in idx])
        y=np.isin(idx,pos_idx).astype(np.int8)
        if int(y.sum())<=0: raise RuntimeError(f"set positive lost race={race['race_id']}")
        xs.append(x); ys.append(y)
        counters["ok_races"]+=1; counters["rows"]+=len(idx); counters["positives"]+=int(y.sum())
        if len(race["market_order"])<len(race["combos"]):
            counters["races_partial_positive_odds"]+=1
    x=np.vstack(xs).astype(np.float32,copy=False)
    y=np.concatenate(ys).astype(np.int8,copy=False)
    print("DECOMP_SET_SAMPLE "+json.dumps({"year":year,"rows":len(y),"counters":dict(counters)},separators=(",",":")),flush=True)
    return {"x":x,"y":y,"meta":dict(counters)}

def sample_order_year(year,l17_rows,root):
    xs=[]; ys=[]; groups=[]; counters=defaultdict(int)
    for race in iter_races(year,l17_rows,root,with_odds=False):
        if race["status"]!="ok":
            counters[race["status"]]+=1; continue
        positive_exact={tuple(race["combos"][int(i)][3]) for i in race["positive"]}
        positive_sets=sorted({set_key(x) for x in positive_exact})
        by_no={int(h["horse_number"]):h for h in race["horses"]}
        for sk in positive_sets:
            rows=[]; labels=[]
            for nums in itertools.permutations(sk,3):
                a,b,c=(by_no[int(n)] for n in nums)
                rows.append(feature_vector(a,b,c,len(race["horses"]),ordered_l17_score(a,b,c)))
                labels.append(int(tuple(nums) in positive_exact))
            if not any(labels):
                raise RuntimeError(f"order positive lost race={race['race_id']} set={sk}")
            xs.extend(rows); ys.extend(labels); groups.append(6)
            counters["positive_sets"]+=1
        counters["ok_races"]+=1
    x=np.vstack(xs).astype(np.float32,copy=False)
    y=np.asarray(ys,dtype=np.int8)
    print("DECOMP_ORDER_SAMPLE "+json.dumps({"year":year,"rows":len(y),"groups":len(groups),"counters":dict(counters)},separators=(",",":")),flush=True)
    return {"x":x,"y":y,"groups":groups,"meta":dict(counters)}

def fit_set(samples,seed):
    x=np.vstack([s["x"] for s in samples]).astype(np.float32,copy=False)
    y=np.concatenate([s["y"] for s in samples]).astype(np.int8,copy=False)
    model=lgb.LGBMClassifier(
        objective="binary",n_estimators=260,learning_rate=.035,num_leaves=31,
        min_child_samples=100,subsample=.90,colsample_bytree=.90,
        reg_lambda=4.0,reg_alpha=.5,random_state=seed,n_jobs=2,
        deterministic=True,force_col_wise=True,verbosity=-1
    )
    model.fit(x,y)
    imp=pd.DataFrame({"stage":"SET","feature":SET_FEATURES,"gain":model.feature_importances_})
    return model,imp,len(y),int(y.sum())

def fit_order(samples,seed):
    x=np.vstack([s["x"] for s in samples]).astype(np.float32,copy=False)
    y=np.concatenate([s["y"] for s in samples]).astype(np.int8,copy=False)
    model=lgb.LGBMClassifier(
        objective="binary",n_estimators=220,learning_rate=.035,num_leaves=31,
        min_child_samples=60,subsample=.90,colsample_bytree=.90,
        reg_lambda=3.0,reg_alpha=.25,random_state=seed,n_jobs=2,
        deterministic=True,force_col_wise=True,verbosity=-1
    )
    model.fit(x,y)
    imp=pd.DataFrame({"stage":"ORDER","feature":ORDER_FEATURES,"gain":model.feature_importances_})
    return model,imp,len(y),int(y.sum())

def blank():
    return {"source_races":0,"tickets":0,"hit_races":0,"return_yen":0.0}

def update(stat,order,returns,n):
    k=min(int(n),len(order))
    chosen=np.asarray(order[:k],dtype=np.int64)
    stat["source_races"]+=1; stat["tickets"]+=k
    stat["hit_races"]+=int(np.any(returns[chosen]>0))
    stat["return_yen"]+=float(returns[chosen].sum())

def finish(year,n,s):
    stake=100.0*s["tickets"]
    return {
        "year":year,"top_n":n,"source_races":s["source_races"],"tickets":s["tickets"],
        "avg_tickets_per_race":s["tickets"]/s["source_races"] if s["source_races"] else None,
        "hit_races":s["hit_races"],
        "race_hit_rate_pct":100.0*s["hit_races"]/s["source_races"] if s["source_races"] else None,
        "stake_yen":stake,"return_yen":s["return_yen"],"profit_yen":s["return_yen"]-stake,
        "roi_pct":100.0*s["return_yen"]/stake if stake else None
    }

def score_year(year,l17_rows,root,set_model,order_model):
    stats={n:blank() for n in TOPN_GRID}
    set_reach=Counter(); order_cond=Counter(); counters=defaultdict(int)
    candidate_counts=[]; set_counts=[]
    for race in iter_races(year,l17_rows,root,with_odds=False):
        if race["status"]!="ok":
            counters[race["status"]]+=1; continue
        sets,by_key,pos_set_idx,pos_keys=build_unordered(race)
        sx=np.vstack([set_feature_vector(hs,len(race["horses"])) for hs,_ in sets])
        pset=np.asarray(set_model.predict_proba(sx)[:,1],dtype=np.float64)
        # Normalization is monotonic for set ranking and gives a clean pseudo-joint probability.
        ssum=float(pset.sum())
        if ssum>0: pset=pset/ssum

        set_order=np.argsort(-pset,kind="mergesort")
        set_rank=np.empty(len(set_order),dtype=np.int32)
        set_rank[set_order]=np.arange(1,len(set_order)+1,dtype=np.int32)
        best_positive_set_rank=min(int(set_rank[int(i)]) for i in pos_set_idx)
        for k in (1,3,5,10,20,50,100,200):
            if best_positive_set_rank<=k: set_reach[k]+=1

        by_no={int(h["horse_number"]):h for h in race["horses"]}
        ticket_scores=np.empty(len(race["combos"]),dtype=np.float64)
        number_to_idx={tuple(c[3]):i for i,c in enumerate(race["combos"])}
        positive_exact={tuple(race["combos"][int(i)][3]) for i in race["positive"]}

        best_cond_rank=99
        for si,(hs,sk) in enumerate(sets):
            perms=list(itertools.permutations(sk,3))
            ox=[]
            for nums in perms:
                a,b,c=(by_no[int(n)] for n in nums)
                ox.append(feature_vector(a,b,c,len(race["horses"]),ordered_l17_score(a,b,c)))
            ox=np.vstack(ox)
            po=np.asarray(order_model.predict_proba(ox)[:,1],dtype=np.float64)
            osum=float(po.sum())
            if osum>0: po=po/osum
            else: po=np.full(len(po),1.0/len(po))
            for j,nums in enumerate(perms):
                ti=number_to_idx[tuple(nums)]
                ticket_scores[ti]=pset[si]*po[j]
            if sk in pos_keys:
                oorder=np.argsort(-po,kind="mergesort")
                orank=np.empty(len(oorder),dtype=np.int32)
                orank[oorder]=np.arange(1,len(oorder)+1,dtype=np.int32)
                for j,nums in enumerate(perms):
                    if tuple(nums) in positive_exact:
                        best_cond_rank=min(best_cond_rank,int(orank[j]))

        if best_cond_rank==99:
            raise RuntimeError(f"conditional order rank missing race={race['race_id']}")
        order_cond[best_cond_rank]+=1

        order=np.argsort(-ticket_scores,kind="mergesort")
        for n in TOPN_GRID: update(stats[n],order,race["returns"],n)
        counters["ok_races"]+=1
        candidate_counts.append(len(order)); set_counts.append(len(sets))

    rows=[finish(year,n,stats[n]) for n in TOPN_GRID]
    total=counters["ok_races"]
    diagnostics={
        "year":year,"races":total,
        "mean_unordered_sets":float(np.mean(set_counts)) if set_counts else None,
        "mean_ordered_tickets":float(np.mean(candidate_counts)) if candidate_counts else None,
        "set_reach":{str(k):{"races":set_reach[k],"rate_pct":100.0*set_reach[k]/total} for k in (1,3,5,10,20,50,100,200)},
        "conditional_order_rank":{str(k):{"races":order_cond[k],"rate_pct":100.0*order_cond[k]/total} for k in range(1,7)},
        "counters":dict(counters),
    }
    print("DECOMP_SCORE "+json.dumps(diagnostics,separators=(",",":")),flush=True)
    return rows,diagnostics

def aggregate(rows,years):
    out=[]
    for n in TOPN_GRID:
        z=[r for r in rows if r["year"] in years and int(r["top_n"])==n]
        races=sum(r["source_races"] for r in z); tickets=sum(r["tickets"] for r in z)
        hits=sum(r["hit_races"] for r in z); ret=sum(r["return_yen"] for r in z)
        stake=100.0*tickets
        out.append({
            "top_n":n,"years":"|".join(map(str,years)),"source_races":races,"tickets":tickets,
            "hit_races":hits,"race_hit_rate_pct":100.0*hits/races if races else None,
            "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
            "roi_pct":100.0*ret/stake if stake else None
        })
    return out

def choose(rows):
    return max(rows,key=lambda r:(r["roi_pct"],r["profit_yen"],-int(r["top_n"])))

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if hasattr(rows,"to_csv"): rows.to_csv(path,index=False); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def main():
    a=parse_args(); paths=parse_paths(a.l17_year)
    l17={y:load_l17(paths[y],y) for y in YEARS}

    set_samples={y:sample_set_year(y,l17[y],a.backfill_root) for y in TRAIN_YEARS}
    order_samples={y:sample_order_year(y,l17[y],a.backfill_root) for y in TRAIN_YEARS}

    yearly=[]; folds=[]; imps=[]; diag={}
    for fold_idx,test_year in enumerate(TEST_YEARS):
        train_years=[y for y in TRAIN_YEARS if y<test_year]
        sm,si,srows,spos=fit_set([set_samples[y] for y in train_years],221000+fold_idx)
        om,oi,orows,opos=fit_order([order_samples[y] for y in train_years],331000+fold_idx)
        si["test_year"]=test_year; oi["test_year"]=test_year
        imps.extend([si,oi])
        rows,d=score_year(test_year,l17[test_year],a.backfill_root,sm,om)
        yearly.extend(rows); diag[str(test_year)]=d
        folds.append({
            "test_year":test_year,"train_years":"|".join(map(str,train_years)),
            "set_train_rows":srows,"set_positive_rows":spos,
            "order_train_rows":orows,"order_positive_rows":opos,
            "set_feature_count":len(SET_FEATURES),"order_feature_count":len(ORDER_FEATURES)
        })

    dev=aggregate(yearly,DEV_YEARS)
    champ=choose(dev)
    holdout=next(r for r in yearly if r["year"]==HOLDOUT_YEAR and int(r["top_n"])==int(champ["top_n"]))
    combined=aggregate(yearly,TEST_YEARS)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"model-folds.csv",folds)
    write_csv(out/"economics-yearly.csv",yearly)
    write_csv(out/"dev-topn.csv",dev)
    write_csv(out/"combined.csv",combined)
    write_csv(out/"feature-importance.csv",pd.concat(imps,ignore_index=True))

    summary={
        "contract":"L2_TRIFECTA_DECOMPOSED_V1_RESULT",
        "architecture":{
            "stage1":"binary podium-set model over every unordered 3-horse set",
            "stage2":"binary conditional-order model trained on all 6 permutations of each true podium set",
            "final_score":"normalized P(set) * normalized P(order | set)",
            "market_model_features":False,
            "market_usage":"hard-negative sampling for stage1 only",
            "manual_seat_rules":False,"manual_gap_rules":False,"race_skip_rules":False
        },
        "development_years":list(DEV_YEARS),"holdout_year":HOLDOUT_YEAR,
        "holdout_used_for_policy_selection":False,
        "selected_dev_policy":champ,"holdout":holdout,
        "diagnostics":diag,
        "comparison_reference_2025_one_stage_compat":{"top_n":5,"roi_pct":96.22337962962963,"hit_races":78,"source_races":3456},
        "2026_locked":True,"production_promotion":False
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Trifecta Decomposed V1\n\n"
        "Two-stage experiment: rank unordered podium trios, then rank the six finishing orders conditional on a trio. "
        "Final ticket score is normalized P(set) times normalized P(order|set). Market data is not a model feature and is used only for stage-1 hard-negative sampling, matching the COMPAT philosophy. "
        "2023-2024 select one global Top-N; 2025 is frozen holdout; 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("L2_TRIFECTA_DECOMPOSED_V1_READY")
    print(json.dumps(summary,separators=(",",":")))

if __name__=="__main__":
    main()
