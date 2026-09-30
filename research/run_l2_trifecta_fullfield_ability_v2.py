#!/usr/bin/env python3
import argparse,csv,gzip,hashlib,itertools,json,math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import decode_odds,payout_map,horse_number_map

YEARS=(2022,2023,2024,2025)
TRAIN_YEARS=(2022,2023,2024)
TEST_YEARS=(2023,2024,2025)
DEV_YEARS=(2023,2024)
HOLDOUT_YEAR=2025
EXPECTED_RACES_PER_YEAR=3456
TOPN_GRID=(1,2,3,5,10,20,30,60,120)
COMPAT_MARKET_TOP=40
COMPAT_L17_TOP=40
COMPAT_HASH_NEG=40
PURE_L17_TOP=60
PURE_HASH_NEG=60

HORSE_FEATURES=(
    "consensus_rank","mean_rank","rank_std","best_rank","worst_rank",
    "top1_votes","top3_support","top6_support","mean_probability","probability_std",
)
FEATURES=(
    tuple(f"s{s}_{name}" for s in (1,2,3) for name in HORSE_FEATURES)
    + (
        "field_size",
        "rank_sum","rank_max","rank_min","rank_span","rank_order_12","rank_order_23",
        "prob_sum","prob_product","prob_min","prob_max","prob_order_12","prob_order_23",
        "rank_std_max","rank_std_mean","top3_support_sum","top6_support_sum",
        "l17_order_score",
    )
)

def parse_args():
    p=argparse.ArgumentParser(description="Corrected full-field TRIFECTA ability-only reevaluation.")
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1); out[int(y)]=Path(p)
    if set(out)!=set(YEARS): raise SystemExit(f"L1.7 years mismatch: {sorted(out)}")
    if 2026 in out: raise SystemExit("2026 sealed")
    return out

def read_jsonl_gz(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def load_l17(path,year):
    rows={}
    for rec in read_jsonl_gz(path):
        if rec.get("contract")!="L17_SEVEN_KING_FULLFIELD_OUTPUT_V1": raise ValueError("bad L1.7 contract")
        if int(rec.get("year"))!=year: raise ValueError("L1.7 year drift")
        rid=str(rec.get("race_id") or "")
        if not rid or rid in rows: raise ValueError(f"bad L1.7 race id {rid}")
        rows[rid]=rec
    if len(rows)!=EXPECTED_RACES_PER_YEAR:
        raise SystemExit(f"L1.7 race coverage drift y={year}: {len(rows)}")
    return rows

def finite(v,default=0.0):
    try:
        if isinstance(v,str): v=v.replace(",","").strip()
        x=float(v); return x if math.isfinite(x) else default
    except (TypeError,ValueError): return default

def horse_vector(h):
    return (
        finite(h.get("consensus_rank"),99),finite(h.get("mean_rank"),99),finite(h.get("rank_std")),
        finite(h.get("best_rank"),99),finite(h.get("worst_rank"),99),finite(h.get("top1_votes")),
        finite(h.get("top3_support")),finite(h.get("top6_support")),
        finite(h.get("mean_probability")),finite(h.get("probability_std")),
    )

def ordered_l17_score(a,b,c):
    pa=max(1e-9,min(.999999,finite(a.get("mean_probability"))))
    pb=max(1e-9,min(.999999,finite(b.get("mean_probability"))))
    pc=max(1e-9,min(.999999,finite(c.get("mean_probability"))))
    return pa*(pb/max(1e-9,1-pa))*(pc/max(1e-9,1-pa-pb))

def feature_vector(a,b,c,field_size,l17_score):
    va,vb,vc=horse_vector(a),horse_vector(b),horse_vector(c)
    ranks=[va[0],vb[0],vc[0]]; probs=[va[8],vb[8],vc[8]]
    rstd=[va[2],vb[2],vc[2]]; t3=[va[6],vb[6],vc[6]]; t6=[va[7],vb[7],vc[7]]
    return np.asarray(
        list(va)+list(vb)+list(vc)+[
            float(field_size),sum(ranks),max(ranks),min(ranks),max(ranks)-min(ranks),
            ranks[0]-ranks[1],ranks[1]-ranks[2],
            sum(probs),probs[0]*probs[1]*probs[2],min(probs),max(probs),
            probs[0]-probs[1],probs[1]-probs[2],
            max(rstd),sum(rstd)/3.0,sum(t3),sum(t6),float(l17_score),
        ],dtype=np.float32
    )

def load_odds_day(path):
    if not path.exists(): return {}
    return {str(x.get("race_id") or ""):x for x in read_jsonl_gz(path)}

def iter_races(year,l17_rows,backfill_root,with_odds):
    root=Path(backfill_root)
    for day in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
        date=day.name[:10]
        odds_by=load_odds_day(root/"data"/"odds"/"daily"/day.name) if with_odds else {}
        for pack in read_jsonl_gz(day):
            rid=str((pack.get("race") or {}).get("race_id") or "")
            rec=l17_rows.get(rid)
            if rec is None: continue
            entries=pack.get("entries") or []
            started_nos={
                int(e["horse_number"]) for e in entries
                if str(e.get("entry_status") or "STARTED")=="STARTED" and e.get("horse_number")
            }
            hno=horse_number_map(pack)
            by_no={}
            for h in rec.get("horses") or []:
                hid=str(h.get("horse_id") or "")
                no=hno.get(hid)
                if no is None or int(no) not in started_nos: continue
                z=dict(h); z["horse_number"]=int(no); by_no[int(no)]=z
            if set(by_no)!=started_nos:
                yield {"status":"l17_started_mismatch","race_id":rid,"date":date,
                       "started":len(started_nos),"mapped":len(by_no)}
                continue
            if len(by_no)<3:
                yield {"status":"too_few_started","race_id":rid,"date":date}; continue
            horses=sorted(by_no.values(),key=lambda x:int(x["consensus_rank"]))
            combos=[]
            l17_scores=[]
            for a,b,c in itertools.permutations(horses,3):
                nums=(int(a["horse_number"]),int(b["horse_number"]),int(c["horse_number"]))
                combos.append((a,b,c,nums))
                l17_scores.append(ordered_l17_score(a,b,c))
            payouts,present=payout_map(pack)
            if "TRIFECTA" not in present:
                yield {"status":"missing_trifecta_payout","race_id":rid,"date":date}; continue
            win_nums={tuple(key[1]) for key,val in payouts.items() if key[0]=="TRIFECTA" and val>0}
            if not win_nums:
                yield {"status":"no_positive_trifecta_payout","race_id":rid,"date":date}; continue
            number_to_idx={x[3]:i for i,x in enumerate(combos)}
            positive=[number_to_idx[n] for n in win_nums if n in number_to_idx]
            if len(positive)!=len(win_nums):
                yield {"status":"winner_outside_started_l17","race_id":rid,"date":date}; continue
            returns=np.zeros(len(combos),dtype=np.float64)
            for n in win_nums:
                returns[number_to_idx[n]]=float(payouts[("TRIFECTA",n)])
            l17_arr=np.asarray(l17_scores,dtype=np.float64)
            l17_order=np.argsort(-l17_arr,kind="mergesort")
            market_order=np.asarray([],dtype=np.int64)
            market_priced=0
            if with_odds:
                orec=odds_by.get(rid)
                if orec is not None:
                    om=decode_odds(orec)
                    priced=[]
                    for i,(_,_,_,nums) in enumerate(combos):
                        odd=om.get(("TRIFECTA",nums))
                        if odd is not None and odd>0: priced.append((float(odd),i))
                    priced.sort(key=lambda x:(x[0],x[1]))
                    market_order=np.asarray([i for _,i in priced],dtype=np.int64)
                    market_priced=len(priced)
            yield {
                "status":"ok","year":year,"race_id":rid,"race_date":date,
                "horses":horses,"combos":combos,"l17_scores":l17_arr,"l17_order":l17_order,
                "positive":np.asarray(sorted(positive),dtype=np.int64),"returns":returns,
                "market_order":market_order,"market_priced":market_priced,
            }

def matrix_for(race,indices):
    idx=np.asarray(indices,dtype=np.int64)
    x=np.empty((len(idx),len(FEATURES)),dtype=np.float32)
    for j,i in enumerate(idx):
        a,b,c,_=race["combos"][int(i)]
        x[j]=feature_vector(a,b,c,len(race["horses"]),race["l17_scores"][int(i)])
    return x

def hash_negatives(race,limit,excluded):
    vals=[]; rid=race["race_id"]
    for i,(_,_,_,nums) in enumerate(race["combos"]):
        if i in excluded: continue
        h=int.from_bytes(hashlib.blake2b(f"{rid}:{nums}".encode(),digest_size=8).digest(),"big")
        vals.append((h,i))
    vals.sort()
    return [i for _,i in vals[:limit]]

def sample_year(year,l17_rows,root,mode):
    xs=[]; ys=[]; groups=[]; counters=defaultdict(int)
    for race in iter_races(year,l17_rows,root,with_odds=(mode=="COMPAT")):
        if race["status"]!="ok":
            counters[race["status"]]+=1; continue
        selected=set(int(i) for i in race["positive"])
        if mode=="COMPAT":
            selected.update(int(i) for i in race["market_order"][:COMPAT_MARKET_TOP])
            selected.update(int(i) for i in race["l17_order"][:COMPAT_L17_TOP])
            selected.update(hash_negatives(race,COMPAT_HASH_NEG,selected))
            if race["market_priced"]<len(race["combos"]): counters["races_partial_positive_odds"]+=1
            if race["market_priced"]==0: counters["races_no_positive_odds"]+=1
        else:
            selected.update(int(i) for i in race["l17_order"][:PURE_L17_TOP])
            selected.update(hash_negatives(race,PURE_HASH_NEG,selected))
        idx=np.asarray(sorted(selected),dtype=np.int64)
        y=np.isin(idx,race["positive"]).astype(np.int8)
        if int(y.sum())<=0: raise RuntimeError(f"positive lost race={race['race_id']}")
        xs.append(matrix_for(race,idx)); ys.append(y); groups.append(len(idx))
        counters["ok_races"]+=1; counters["rows"]+=len(idx); counters["positives"]+=int(y.sum())
    x=np.vstack(xs).astype(np.float32,copy=False)
    y=np.concatenate(ys).astype(np.int8,copy=False)
    print("ABILITY_V2_SAMPLE "+json.dumps({"year":year,"mode":mode,"rows":len(y),"groups":len(groups),"counters":dict(counters)},ensure_ascii=False,separators=(",",":")),flush=True)
    return {"x":x,"y":y,"groups":groups,"meta":dict(counters)}

def train(samples,seed):
    x=np.vstack([s["x"] for s in samples]).astype(np.float32,copy=False)
    y=np.concatenate([s["y"] for s in samples]).astype(np.int8,copy=False)
    groups=[g for s in samples for g in s["groups"]]
    if sum(groups)!=len(y): raise RuntimeError("group mismatch")
    model=lgb.LGBMRanker(
        objective="lambdarank",metric="ndcg",n_estimators=240,learning_rate=.035,
        num_leaves=31,min_child_samples=100,subsample=.90,colsample_bytree=.90,
        reg_lambda=4.0,reg_alpha=.5,random_state=seed,n_jobs=2,
        deterministic=True,force_col_wise=True,verbosity=-1,
    )
    model.fit(x,y,group=groups)
    imp=pd.DataFrame({"feature":FEATURES,"gain":model.feature_importances_})
    return model,imp,len(y),len(groups)

def blank():
    return {"source_races":0,"tickets":0,"hit_races":0,"return_yen":0.0}

def update(stat,order,returns,n):
    k=min(int(n),len(order))
    chosen=np.asarray(order[:k],dtype=np.int64)
    stat["source_races"]+=1; stat["tickets"]+=k
    stat["hit_races"]+=int(np.any(returns[chosen]>0))
    stat["return_yen"]+=float(returns[chosen].sum())

def finish(year,mode,n,s):
    stake=100.0*s["tickets"]
    return {
        "year":year,"mode":mode,"top_n":n,"source_races":s["source_races"],
        "tickets":s["tickets"],"avg_tickets_per_race":s["tickets"]/s["source_races"] if s["source_races"] else None,
        "hit_races":s["hit_races"],"race_hit_rate_pct":100.0*s["hit_races"]/s["source_races"] if s["source_races"] else None,
        "stake_yen":stake,"return_yen":s["return_yen"],"profit_yen":s["return_yen"]-stake,
        "roi_pct":100.0*s["return_yen"]/stake if stake else None,
    }

def score_year(year,l17_rows,root,models):
    stats={(mode,n):blank() for mode in models for n in TOPN_GRID}
    counters=defaultdict(int); cand=[]; fields=[]
    for race in iter_races(year,l17_rows,root,with_odds=False):
        if race["status"]!="ok":
            counters[race["status"]]+=1; continue
        idx=np.arange(len(race["combos"]),dtype=np.int64)
        x=matrix_for(race,idx)
        for mode,model in models.items():
            pred=np.asarray(model.predict(x),dtype=np.float64)
            order=np.argsort(-pred,kind="mergesort")
            for n in TOPN_GRID: update(stats[(mode,n)],order,race["returns"],n)
        counters["ok_races"]+=1; cand.append(len(idx)); fields.append(len(race["horses"]))
    rows=[finish(year,mode,n,stats[(mode,n)]) for mode in models for n in TOPN_GRID]
    meta={
        "year":year,"races":counters["ok_races"],
        "mean_candidates":float(np.mean(cand)) if cand else None,
        "median_candidates":float(np.median(cand)) if cand else None,
        "max_candidates":int(max(cand)) if cand else None,
        "mean_field_size":float(np.mean(fields)) if fields else None,
        "counters":dict(counters),
    }
    print("ABILITY_V2_SCORE "+json.dumps(meta,ensure_ascii=False,separators=(",",":")),flush=True)
    return rows,meta

def aggregate(rows,years,mode):
    out=[]
    for n in TOPN_GRID:
        z=[r for r in rows if r["mode"]==mode and r["year"] in years and int(r["top_n"])==n]
        races=sum(r["source_races"] for r in z); tickets=sum(r["tickets"] for r in z)
        hits=sum(r["hit_races"] for r in z); ret=sum(r["return_yen"] for r in z); stake=100.0*tickets
        out.append({
            "mode":mode,"top_n":n,"years":"|".join(map(str,years)),"source_races":races,
            "tickets":tickets,"hit_races":hits,"race_hit_rate_pct":100.0*hits/races if races else None,
            "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
            "roi_pct":100.0*ret/stake if stake else None,
        })
    return out

def choose(rows):
    return max(rows,key=lambda r:(r["roi_pct"] if r["roi_pct"] is not None else -1e18,r["profit_yen"],-int(r["top_n"])))

def find_year(rows,year,mode,n):
    return next(r for r in rows if r["year"]==year and r["mode"]==mode and int(r["top_n"])==int(n))

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(rows,pd.DataFrame): rows.to_csv(path,index=False); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def main():
    a=parse_args(); paths=parse_paths(a.l17_year)
    l17={y:load_l17(paths[y],y) for y in YEARS}
    samples={
        mode:{y:sample_year(y,l17[y],a.backfill_root,mode) for y in TRAIN_YEARS}
        for mode in ("COMPAT","PURE")
    }
    yearly=[]; folds=[]; importance=[]; year_meta={}
    for fold_idx,test_year in enumerate(TEST_YEARS):
        train_years=[y for y in TRAIN_YEARS if y<test_year]
        models={}
        for midx,mode in enumerate(("COMPAT","PURE")):
            model,imp,nrows,nraces=train([samples[mode][y] for y in train_years],111000+fold_idx*10+midx)
            models[mode]=model
            imp["mode"]=mode; imp["test_year"]=test_year; importance.append(imp)
            folds.append({
                "test_year":test_year,"mode":mode,"train_years":"|".join(map(str,train_years)),
                "sampled_training_rows":nrows,"training_races":nraces,"feature_count":len(FEATURES)
            })
        rows,meta=score_year(test_year,l17[test_year],a.backfill_root,models)
        yearly.extend(rows); year_meta[str(test_year)]=meta

    dev={mode:aggregate(yearly,DEV_YEARS,mode) for mode in ("COMPAT","PURE")}
    champions={mode:choose(dev[mode]) for mode in ("COMPAT","PURE")}
    holdouts={mode:find_year(yearly,HOLDOUT_YEAR,mode,champions[mode]["top_n"]) for mode in champions}
    fixed_top3={mode:find_year(yearly,HOLDOUT_YEAR,mode,3) for mode in champions}
    combined={mode:aggregate(yearly,TEST_YEARS,mode) for mode in champions}

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"model-folds.csv",folds)
    write_csv(out/"economics-yearly.csv",yearly)
    write_csv(out/"dev-compat-topn.csv",dev["COMPAT"])
    write_csv(out/"dev-pure-topn.csv",dev["PURE"])
    write_csv(out/"combined-compat.csv",combined["COMPAT"])
    write_csv(out/"combined-pure.csv",combined["PURE"])
    write_csv(out/"feature-importance.csv",pd.concat(importance,ignore_index=True))
    summary={
        "contract":"L2_TRIFECTA_FULLFIELD_ABILITY_V2_RESULT",
        "fix":"Do not require every trifecta combination to have a positive decoded odds value. Candidate universe comes from STARTED L1.7 horses, labels/returns from payout.",
        "evaluation_odds_required":False,
        "candidate_universe":"all ordered trifecta permutations of all STARTED horses represented by L1.7",
        "all_candidates_scored":True,
        "manual_seat_rules":False,"manual_gap_rules":False,"race_skip_rules":False,
        "modes":{
            "COMPAT":"No market model features, but retains V1-style market-top negative sampling for isolation of the coverage fix.",
            "PURE":"No market model features and no market-dependent negative sampling."
        },
        "development_years":list(DEV_YEARS),"holdout_year":HOLDOUT_YEAR,
        "holdout_used_for_policy_selection":False,"topn_grid":list(TOPN_GRID),
        "selected_dev_policy":champions,"holdout":holdouts,
        "holdout_fixed_old_top3_reference":fixed_top3,
        "year_meta":year_meta,"2026_locked":True,"production_promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Trifecta Full-Field Ability V2\n\n"
        "Coverage-corrected reevaluation. It removes the V1 requirement that every trifecta ticket in a race have a positive decoded final odds value. "
        "Evaluation candidates are all ordered triples of STARTED L1.7 horses; payout is used only as label/evaluation. "
        "COMPAT retains V1-style market-top negative sampling only to isolate the coverage fix. PURE removes market dependence from sampling too. "
        "2023-2024 select one global Top-N per mode; 2025 is frozen holdout; 2026 remains sealed.\n",encoding="utf-8")
    print("L2_TRIFECTA_FULLFIELD_ABILITY_V2_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__": main()
