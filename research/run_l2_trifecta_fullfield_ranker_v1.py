#!/usr/bin/env python3
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import decode_odds, payout_map, horse_number_map

YEARS=(2022,2023,2024,2025)
TEST_YEARS=(2023,2024,2025)
DEV_YEARS=(2023,2024)
HOLDOUT_YEAR=2025
EXPECTED_RACES_PER_YEAR=3456
TOPN_GRID=(1,2,3,5,10,20,30,60,120)
SAMPLE_MARKET_TOP=40
SAMPLE_L17_TOP=40
SAMPLE_HASH_NEG=40

HORSE_FEATURES=(
    "consensus_rank","mean_rank","rank_std","best_rank","worst_rank",
    "top1_votes","top3_support","top6_support",
    "mean_probability","probability_std",
)
BASE_FEATURES=(
    tuple(f"s{s}_{name}" for s in (1,2,3) for name in HORSE_FEATURES)
    + (
        "field_size",
        "rank_sum","rank_max","rank_min","rank_span","rank_order_12","rank_order_23",
        "prob_sum","prob_product","prob_min","prob_max","prob_order_12","prob_order_23",
        "rank_std_max","rank_std_mean",
        "top3_support_sum","top6_support_sum",
        "l17_order_score",
    )
)
MARKET_FEATURES=("market_q_norm","market_log_q","market_rank","market_rank_pct")
ALL_FEATURES=BASE_FEATURES+MARKET_FEATURES


def parse_args():
    p=argparse.ArgumentParser(description="Full-field TRIFECTA ticket ranker; score every priced ordered ticket.")
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1)
        out[int(y)]=Path(p)
    if set(out)!=set(YEARS):
        raise SystemExit(f"L1.7 year mismatch got={sorted(out)}")
    if 2026 in out:
        raise SystemExit("2026 sealed")
    return out


def read_jsonl_gz(path):
    with gzip.open(path,"rt",encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_l17(path,year):
    out={}
    for rec in read_jsonl_gz(path):
        if rec.get("contract")!="L17_SEVEN_KING_FULLFIELD_OUTPUT_V1":
            raise ValueError(f"bad L1.7 contract y={year}")
        if int(rec.get("year"))!=year:
            raise ValueError(f"L1.7 year drift y={year}")
        rid=str(rec.get("race_id") or "")
        if not rid or rid in out:
            raise ValueError(f"bad/duplicate race_id y={year} rid={rid}")
        out[rid]=rec
    if len(out)!=EXPECTED_RACES_PER_YEAR:
        raise SystemExit(f"L1.7 race coverage drift y={year}: {len(out)}")
    return out


def finite(v,default=0.0):
    try:
        if isinstance(v,str):
            v=v.replace(",","").strip()
        x=float(v)
        return x if math.isfinite(x) else default
    except (TypeError,ValueError):
        return default


def ordered_l17_score(a,b,c):
    pa=max(1e-9,min(0.999999,finite(a.get("mean_probability"))))
    pb=max(1e-9,min(0.999999,finite(b.get("mean_probability"))))
    pc=max(1e-9,min(0.999999,finite(c.get("mean_probability"))))
    return pa*(pb/max(1e-9,1.0-pa))*(pc/max(1e-9,1.0-pa-pb))


def horse_vector(h):
    return (
        finite(h.get("consensus_rank"),99.0),
        finite(h.get("mean_rank"),99.0),
        finite(h.get("rank_std")),
        finite(h.get("best_rank"),99.0),
        finite(h.get("worst_rank"),99.0),
        finite(h.get("top1_votes")),
        finite(h.get("top3_support")),
        finite(h.get("top6_support")),
        finite(h.get("mean_probability")),
        finite(h.get("probability_std")),
    )


def base_vector(a,b,c,field_size,l17_score):
    va=horse_vector(a); vb=horse_vector(b); vc=horse_vector(c)
    ranks=[va[0],vb[0],vc[0]]
    probs=[va[8],vb[8],vc[8]]
    rank_stds=[va[2],vb[2],vc[2]]
    top3=[va[6],vb[6],vc[6]]
    top6=[va[7],vb[7],vc[7]]
    return np.asarray(
        list(va)+list(vb)+list(vc)+[
            float(field_size),
            sum(ranks),max(ranks),min(ranks),max(ranks)-min(ranks),
            ranks[0]-ranks[1],ranks[1]-ranks[2],
            sum(probs),probs[0]*probs[1]*probs[2],min(probs),max(probs),
            probs[0]-probs[1],probs[1]-probs[2],
            max(rank_stds),sum(rank_stds)/3.0,
            sum(top3),sum(top6),
            float(l17_score),
        ],
        dtype=np.float32,
    )


def iter_full_races(year,l17_rows,backfill_root):
    root=Path(backfill_root)
    for day_path in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
        date=day_path.name[:10]
        odds_path=root/"data"/"odds"/"daily"/day_path.name
        if not odds_path.exists():
            yield {"status":"missing_odds_day","date":date}
            continue
        odds_by_race={str(x.get("race_id") or ""):x for x in read_jsonl_gz(odds_path)}
        for pack in read_jsonl_gz(day_path):
            rid=str((pack.get("race") or {}).get("race_id") or "")
            rec=l17_rows.get(rid)
            if rec is None:
                continue
            odds_rec=odds_by_race.get(rid)
            if odds_rec is None:
                yield {"status":"missing_odds_row","race_id":rid,"date":date}
                continue

            odds_all=decode_odds(odds_rec)
            tri_odds={tuple(key[1]):float(val) for key,val in odds_all.items() if key[0]=="TRIFECTA" and val and val>0}
            if not tri_odds:
                yield {"status":"no_trifecta_odds","race_id":rid,"date":date}
                continue

            priced_numbers={n for combo in tri_odds for n in combo}
            hno=horse_number_map(pack)
            horses=[]
            for h in sorted(rec["horses"],key=lambda x:int(x["consensus_rank"])):
                hid=str(h["horse_id"])
                no=hno.get(hid)
                if no is None or int(no) not in priced_numbers:
                    continue
                z=dict(h)
                z["horse_number"]=int(no)
                horses.append(z)

            if len(horses)<3:
                yield {"status":"too_few_active","race_id":rid,"date":date}
                continue
            mapped_numbers={int(h["horse_number"]) for h in horses}
            if mapped_numbers!=priced_numbers:
                yield {
                    "status":"l17_number_mismatch","race_id":rid,"date":date,
                    "priced":len(priced_numbers),"mapped":len(mapped_numbers)
                }
                continue

            expected=len(horses)*(len(horses)-1)*(len(horses)-2)
            if len(tri_odds)!=expected:
                yield {
                    "status":"incomplete_odds_universe","race_id":rid,"date":date,
                    "expected":expected,"priced":len(tri_odds)
                }
                continue

            payouts,present=payout_map(pack)
            if "TRIFECTA" not in present:
                yield {"status":"missing_payout","race_id":rid,"date":date}
                continue

            combos=[]
            odds=[]
            returns=[]
            l17_scores=[]
            horse_by_no={int(h["horse_number"]):h for h in horses}
            for nums in sorted(tri_odds):
                a,b,c=(horse_by_no[int(nums[0])],horse_by_no[int(nums[1])],horse_by_no[int(nums[2])])
                odd=float(tri_odds[nums])
                ret=float(payouts.get(("TRIFECTA",tuple(nums)),0.0))
                score=ordered_l17_score(a,b,c)
                combos.append((a,b,c,tuple(nums)))
                odds.append(odd)
                returns.append(ret)
                l17_scores.append(score)

            odds_arr=np.asarray(odds,dtype=np.float64)
            ret_arr=np.asarray(returns,dtype=np.float64)
            l17_arr=np.asarray(l17_scores,dtype=np.float64)
            inv=1.0/np.clip(odds_arr,1e-12,None)
            q=inv/max(float(inv.sum()),1e-12)
            market_order=np.argsort(-q,kind="mergesort")
            market_rank=np.empty(len(combos),dtype=np.int32)
            market_rank[market_order]=np.arange(1,len(combos)+1,dtype=np.int32)
            l17_order=np.argsort(-l17_arr,kind="mergesort")
            l17_rank=np.empty(len(combos),dtype=np.int32)
            l17_rank[l17_order]=np.arange(1,len(combos)+1,dtype=np.int32)

            positive=np.flatnonzero(ret_arr>0)
            if len(positive)==0:
                yield {"status":"winner_not_in_universe","race_id":rid,"date":date}
                continue

            yield {
                "status":"ok",
                "year":year,
                "race_id":rid,
                "race_date":date,
                "horses":horses,
                "combos":combos,
                "odds":odds_arr,
                "returns":ret_arr,
                "market_q":q.astype(np.float32),
                "market_rank":market_rank,
                "market_order":market_order,
                "l17_scores":l17_arr.astype(np.float32),
                "l17_rank":l17_rank,
                "l17_order":l17_order,
                "positive":positive,
            }


def feature_matrix(race,indices):
    idx=np.asarray(indices,dtype=np.int64)
    out=np.empty((len(idx),len(BASE_FEATURES)),dtype=np.float32)
    for j,i in enumerate(idx):
        a,b,c,_=race["combos"][int(i)]
        out[j,:]=base_vector(a,b,c,len(race["horses"]),race["l17_scores"][int(i)])
    return out


def market_extra_matrix(race,indices):
    idx=np.asarray(indices,dtype=np.int64)
    out=np.empty((len(idx),len(MARKET_FEATURES)),dtype=np.float32)
    total=float(len(race["combos"]))
    for j,i in enumerate(idx):
        q=float(race["market_q"][int(i)])
        mr=float(race["market_rank"][int(i)])
        out[j,:]=(q,math.log(max(q,1e-12)),mr,mr/total)
    return out


def stable_hash_indices(race,limit,excluded):
    vals=[]
    rid=race["race_id"]
    for i,(_,_,_,nums) in enumerate(race["combos"]):
        if i in excluded:
            continue
        token=f"{rid}:{nums[0]}-{nums[1]}-{nums[2]}".encode()
        h=int.from_bytes(hashlib.blake2b(token,digest_size=8).digest(),"big")
        vals.append((h,i))
    vals.sort()
    return [i for _,i in vals[:limit]]


def build_training_sample(year,l17_rows,backfill_root):
    x_base=[]
    x_market=[]
    ys=[]
    groups=[]
    counters=defaultdict(int)

    for race in iter_full_races(year,l17_rows,backfill_root):
        if race["status"]!="ok":
            counters[race["status"]]+=1
            continue
        counters["ok_races"]+=1
        selected=set(int(x) for x in race["positive"])
        selected.update(int(x) for x in race["market_order"][:SAMPLE_MARKET_TOP])
        selected.update(int(x) for x in race["l17_order"][:SAMPLE_L17_TOP])
        selected.update(stable_hash_indices(race,SAMPLE_HASH_NEG,selected))
        idx=np.asarray(sorted(selected),dtype=np.int64)
        y=(race["returns"][idx]>0).astype(np.int8)
        if int(y.sum())<=0:
            raise RuntimeError(f"sample lost positive race={race['race_id']}")
        x_base.append(feature_matrix(race,idx))
        x_market.append(market_extra_matrix(race,idx))
        ys.append(y)
        groups.append(len(idx))
        counters["rows"]+=len(idx)
        counters["positives"]+=int(y.sum())

    if not x_base:
        raise SystemExit(f"no training sample y={year}")
    xb=np.vstack(x_base).astype(np.float32,copy=False)
    xm=np.vstack(x_market).astype(np.float32,copy=False)
    y=np.concatenate(ys).astype(np.int8,copy=False)
    print("FULLFIELD_SAMPLE_READY "+json.dumps({
        "year":year,"rows":len(y),"groups":len(groups),
        "positives":int(y.sum()),"counters":dict(counters)
    },ensure_ascii=False,separators=(",",":")),flush=True)
    return {"base":xb,"market_extra":xm,"y":y,"groups":groups,"meta":dict(counters)}


def train_ranker(samples,market_aware,seed):
    xb=np.vstack([s["base"] for s in samples]).astype(np.float32,copy=False)
    if market_aware:
        xm=np.vstack([s["market_extra"] for s in samples]).astype(np.float32,copy=False)
        x=np.hstack([xb,xm]).astype(np.float32,copy=False)
    else:
        x=xb
    y=np.concatenate([s["y"] for s in samples]).astype(np.int8,copy=False)
    groups=[g for s in samples for g in s["groups"]]
    if sum(groups)!=len(y):
        raise RuntimeError("group length mismatch")
    model=lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=240,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=4.0,
        reg_alpha=0.5,
        random_state=seed,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(x,y,group=groups)
    names=ALL_FEATURES if market_aware else BASE_FEATURES
    imp=pd.DataFrame({"feature":names,"gain":model.feature_importances_})
    return model,imp,len(y),len(groups)


def new_metric():
    return {
        "source_races":0,
        "tickets":0,
        "hit_races":0,
        "return_yen":0.0,
        "candidate_sum":0,
        "field_sum":0,
    }


def update_metric(stat,order,returns,n):
    k=min(int(n),len(order))
    chosen=np.asarray(order[:k],dtype=np.int64)
    ret=float(returns[chosen].sum())
    hit=bool(np.any(returns[chosen]>0))
    stat["source_races"]+=1
    stat["tickets"]+=k
    stat["hit_races"]+=int(hit)
    stat["return_yen"]+=ret


def finalize_metric(year,ranking,n,stat):
    races=stat["source_races"]
    tickets=stat["tickets"]
    stake=100.0*tickets
    return {
        "year":year,
        "ranking":ranking,
        "top_n":n,
        "source_races":races,
        "tickets":tickets,
        "avg_tickets_per_race":tickets/races if races else None,
        "hit_races":stat["hit_races"],
        "race_hit_rate_pct":100.0*stat["hit_races"]/races if races else None,
        "stake_yen":stake,
        "return_yen":stat["return_yen"],
        "profit_yen":stat["return_yen"]-stake,
        "roi_pct":100.0*stat["return_yen"]/stake if stake else None,
    }


def score_year(year,l17_rows,backfill_root,ability_model,market_model):
    stats={(ranking,n):new_metric() for ranking in ("MARKET","L17_ORDER","ABILITY_MODEL","MARKET_AWARE_MODEL") for n in TOPN_GRID}
    counters=defaultdict(int)
    holdout_top={}
    candidate_counts=[]
    field_sizes=[]

    for race in iter_full_races(year,l17_rows,backfill_root):
        if race["status"]!="ok":
            counters[race["status"]]+=1
            continue
        counters["ok_races"]+=1
        all_idx=np.arange(len(race["combos"]),dtype=np.int64)
        xb=feature_matrix(race,all_idx)
        xm_extra=market_extra_matrix(race,all_idx)
        ability_score=np.asarray(ability_model.predict(xb),dtype=np.float64)
        market_score=np.asarray(market_model.predict(np.hstack([xb,xm_extra])),dtype=np.float64)
        ability_order=np.argsort(-ability_score,kind="mergesort")
        market_model_order=np.argsort(-market_score,kind="mergesort")

        orders={
            "MARKET":race["market_order"],
            "L17_ORDER":race["l17_order"],
            "ABILITY_MODEL":ability_order,
            "MARKET_AWARE_MODEL":market_model_order,
        }
        for ranking,order in orders.items():
            for n in TOPN_GRID:
                update_metric(stats[(ranking,n)],order,race["returns"],n)

        candidate_counts.append(len(race["combos"]))
        field_sizes.append(len(race["horses"]))

        if year==HOLDOUT_YEAR:
            # Save only the first 10 from each model for compact diagnosis.
            rows=[]
            for label,order,scores in (
                ("ABILITY_MODEL",ability_order,ability_score),
                ("MARKET_AWARE_MODEL",market_model_order,market_score),
            ):
                for pos,i in enumerate(order[:10],1):
                    a,b,c,nums=race["combos"][int(i)]
                    rows.append({
                        "ranking":label,
                        "rank":pos,
                        "ticket_numbers":"-".join(map(str,nums)),
                        "ticket_horse_ids":"|".join([str(a["horse_id"]),str(b["horse_id"]),str(c["horse_id"])]),
                        "score":float(scores[int(i)]),
                        "market_rank":int(race["market_rank"][int(i)]),
                        "l17_rank":int(race["l17_rank"][int(i)]),
                        "odds":float(race["odds"][int(i)]),
                        "hit":int(race["returns"][int(i)]>0),
                        "return_yen_per100":float(race["returns"][int(i)]),
                    })
            holdout_top[race["race_id"]]={
                "race_date":race["race_date"],
                "field_size":len(race["horses"]),
                "candidate_count":len(race["combos"]),
                "rows":rows,
            }

    rows=[finalize_metric(year,ranking,n,stats[(ranking,n)]) for ranking in ("MARKET","L17_ORDER","ABILITY_MODEL","MARKET_AWARE_MODEL") for n in TOPN_GRID]
    meta={
        "year":year,
        "races":counters["ok_races"],
        "mean_candidates":float(np.mean(candidate_counts)) if candidate_counts else None,
        "median_candidates":float(np.median(candidate_counts)) if candidate_counts else None,
        "max_candidates":int(max(candidate_counts)) if candidate_counts else None,
        "mean_field_size":float(np.mean(field_sizes)) if field_sizes else None,
        "counters":dict(counters),
    }
    print("FULLFIELD_SCORE_READY "+json.dumps(meta,ensure_ascii=False,separators=(",",":")),flush=True)
    return rows,meta,holdout_top


def combine_rows(yearly):
    out=[]
    for ranking in ("MARKET","L17_ORDER","ABILITY_MODEL","MARKET_AWARE_MODEL"):
        for n in TOPN_GRID:
            sub=[r for r in yearly if r["ranking"]==ranking and int(r["top_n"])==n]
            races=sum(r["source_races"] for r in sub)
            tickets=sum(r["tickets"] for r in sub)
            hits=sum(r["hit_races"] for r in sub)
            ret=sum(r["return_yen"] for r in sub)
            stake=100.0*tickets
            out.append({
                "ranking":ranking,
                "top_n":n,
                "test_years":"2023-2025",
                "source_races":races,
                "tickets":tickets,
                "avg_tickets_per_race":tickets/races if races else None,
                "hit_races":hits,
                "race_hit_rate_pct":100.0*hits/races if races else None,
                "stake_yen":stake,
                "return_yen":ret,
                "profit_yen":ret-stake,
                "roi_pct":100.0*ret/stake if stake else None,
            })
    return out


def dev_rows(yearly,ranking):
    out=[]
    for n in TOPN_GRID:
        sub=[r for r in yearly if r["year"] in DEV_YEARS and r["ranking"]==ranking and int(r["top_n"])==n]
        races=sum(r["source_races"] for r in sub)
        tickets=sum(r["tickets"] for r in sub)
        hits=sum(r["hit_races"] for r in sub)
        ret=sum(r["return_yen"] for r in sub)
        stake=100.0*tickets
        out.append({
            "ranking":ranking,
            "top_n":n,
            "development_years":"2023-2024",
            "source_races":races,
            "tickets":tickets,
            "hit_races":hits,
            "race_hit_rate_pct":100.0*hits/races if races else None,
            "stake_yen":stake,
            "return_yen":ret,
            "profit_yen":ret-stake,
            "roi_pct":100.0*ret/stake if stake else None,
        })
    return out


def choose_dev(rows):
    return max(rows,key=lambda r:(
        r["roi_pct"] if r["roi_pct"] is not None else -1e18,
        r["profit_yen"],
        -int(r["top_n"]),
    ))


def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(rows,pd.DataFrame):
        rows.to_csv(path,index=False); return
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main():
    a=parse_args()
    paths=parse_paths(a.l17_year)
    l17={y:load_l17(paths[y],y) for y in YEARS}

    # Training is sampled only to fit on a free standard CPU runner.
    # Evaluation below scores every full-field candidate.
    samples={y:build_training_sample(y,l17[y],a.backfill_root) for y in (2022,2023,2024)}

    yearly=[]
    fold_rows=[]
    importance=[]
    year_meta={}
    holdout_top={}

    for yidx,test_year in enumerate(TEST_YEARS):
        train_years=[y for y in (2022,2023,2024) if y<test_year]
        train_samples=[samples[y] for y in train_years]

        ability,imp_a,train_rows,train_groups=train_ranker(
            train_samples,False,101000+yidx
        )
        market,imp_m,train_rows_m,train_groups_m=train_ranker(
            train_samples,True,102000+yidx
        )
        if train_rows!=train_rows_m or train_groups!=train_groups_m:
            raise RuntimeError("model training coverage mismatch")

        rows,meta,top=score_year(
            test_year,l17[test_year],a.backfill_root,ability,market
        )
        yearly.extend(rows)
        year_meta[str(test_year)]=meta
        if top:
            holdout_top=top

        fold_rows.append({
            "test_year":test_year,
            "train_years":"|".join(map(str,train_years)),
            "sampled_training_rows":train_rows,
            "training_races":train_groups,
            "ability_feature_count":len(BASE_FEATURES),
            "market_aware_feature_count":len(ALL_FEATURES),
            "test_fullfield_races":meta["races"],
            "mean_test_candidates_per_race":meta["mean_candidates"],
            "median_test_candidates_per_race":meta["median_candidates"],
            "max_test_candidates_per_race":meta["max_candidates"],
        })
        imp_a["model"]="ABILITY_MODEL"; imp_a["test_year"]=test_year
        imp_m["model"]="MARKET_AWARE_MODEL"; imp_m["test_year"]=test_year
        importance.extend([imp_a,imp_m])

    combined=combine_rows(yearly)
    dev_ability=dev_rows(yearly,"ABILITY_MODEL")
    dev_market=dev_rows(yearly,"MARKET_AWARE_MODEL")
    ability_champion=choose_dev(dev_ability)
    market_champion=choose_dev(dev_market)

    def holdout_row(ranking,n):
        return next(
            r for r in yearly
            if r["year"]==HOLDOUT_YEAR and r["ranking"]==ranking and int(r["top_n"])==int(n)
        )

    ability_holdout=holdout_row("ABILITY_MODEL",ability_champion["top_n"])
    market_holdout=holdout_row("MARKET_AWARE_MODEL",market_champion["top_n"])
    ability_market_baseline=holdout_row("MARKET",ability_champion["top_n"])
    market_market_baseline=holdout_row("MARKET",market_champion["top_n"])

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"model-folds.csv",fold_rows)
    write_csv(out/"economics-yearly.csv",yearly)
    write_csv(out/"economics-combined.csv",combined)
    write_csv(out/"dev-ability-topn-grid.csv",dev_ability)
    write_csv(out/"dev-market-aware-topn-grid.csv",dev_market)
    write_csv(out/"feature-importance.csv",pd.concat(importance,ignore_index=True))

    # Keep holdout diagnostic compact: top10 only, not the full candidate matrix.
    with gzip.open(out/"holdout-top10-diagnostic.jsonl.gz","wt",encoding="utf-8") as fh:
        for rid in sorted(holdout_top):
            rec={"race_id":rid,**holdout_top[rid]}
            fh.write(json.dumps(rec,ensure_ascii=False,separators=(",",":"))+"\n")

    summary={
        "contract":"L2_TRIFECTA_FULLFIELD_RANKER_V1_RESULT",
        "candidate_universe":"all priced ordered trifecta tickets among all active L1.7 horses",
        "top6_candidate_limit":False,
        "all_candidates_scored_at_evaluation":True,
        "training_negative_sampling":{
            "reason":"fit within free standard CPU runner memory",
            "market_top":SAMPLE_MARKET_TOP,
            "l17_top":SAMPLE_L17_TOP,
            "deterministic_hash_negatives":SAMPLE_HASH_NEG,
            "all_positive_tickets_included":True,
        },
        "models":{
            "ABILITY_MODEL":"L1.7 seat/order features only; no market inputs",
            "MARKET_AWARE_MODEL":"same plus normalized within-race trifecta market probability/rank",
        },
        "raw_odds_as_model_feature":False,
        "manual_seat_rules":False,
        "manual_gap_rules":False,
        "race_skip_rules":False,
        "topn_grid":list(TOPN_GRID),
        "development_years":list(DEV_YEARS),
        "holdout_year":HOLDOUT_YEAR,
        "holdout_used_for_policy_selection":False,
        "ability_selected_dev_policy":ability_champion,
        "ability_holdout":ability_holdout,
        "ability_holdout_market_baseline_same_n":ability_market_baseline,
        "market_aware_selected_dev_policy":market_champion,
        "market_aware_holdout":market_holdout,
        "market_aware_holdout_market_baseline_same_n":market_market_baseline,
        "year_meta":year_meta,
        "2026_locked":True,
        "production_promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Trifecta Full-Field Ranker V1\n\n"
        "This experiment removes the Seven-King Top6 candidate-universe limit. "
        "At evaluation time, every priced ordered trifecta among all active L1.7 horses is scored. "
        "To remain on the free standard GitHub CPU runner, training uses deterministic negative sampling "
        "(market top40 + L1.7 top40 + 40 hash negatives + every positive ticket), while no candidate is sampled away at evaluation. "
        "Two single models are compared: ability-only and market-aware. "
        "There are no hand-authored seat rules, gap buckets, or race skip rules. "
        "2023-2024 select a global Top-N independently for each model; 2025 is frozen holdout; 2026 stays sealed.\n",
        encoding="utf-8",
    )
    print("L2_TRIFECTA_FULLFIELD_RANKER_V1_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)


if __name__=="__main__":
    main()
