#!/usr/bin/env python3
import argparse,csv,gzip,itertools,json,math,os,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import load_day,load_odds_day,iter_decoded_odds,payout_map,canonical_numbers

YEARS=(2023,2024,2025)
FOLDS=((2024,(2023,)),(2025,(2023,2024)))
EPS=1e-12

MARKET_FEATURES=[
    "field_size","pair_market_odds","log_pair_market_odds","pair_market_prob_norm",
    "win_odds_min","win_odds_max","log_win_odds_sum",
    "market_rank_min","market_rank_max","market_rank_sum","market_rank_diff",
]
KING_FEATURES=[
    "king_rank_min","king_rank_max","king_rank_sum","king_rank_diff",
    "confidence_mean","confidence_min","confidence_max",
    "king_top3_count","king_top6_count",
]
OUTSIDER_FEATURES=[
    "outsider_score_sum","outsider_score_max","outsider_score_min",
    "outsider_support_sum","outsider_support_max",
    "outsider_top1_votes_sum","outsider_top2_votes_sum","outsider_top3_votes_sum",
]
L175_FEATURES=[
    "signed_gap_sum","signed_gap_min","signed_gap_max",
    "abs_gap_sum","abs_gap_max","up_count","down_count","near_count",
    "p3_delta_sum","p3_delta_min","p3_delta_max","p3_valid_count",
    "outsider_alignment_sum","outsider_alignment_abs_sum",
]

MODEL_FEATURES={
    "MARKET_ONLY":MARKET_FEATURES,
    "MARKET_PLUS_KING":MARKET_FEATURES+KING_FEATURES,
    "MARKET_PLUS_KING_OUTSIDER":MARKET_FEATURES+KING_FEATURES+OUTSIDER_FEATURES,
    "L175_FULL":MARKET_FEATURES+KING_FEATURES+OUTSIDER_FEATURES+L175_FEATURES,
}

def args():
    p=argparse.ArgumentParser(description="L2 Ticket Lab V1: all-quinella market-conditional benchmark with frozen L1.75 context.")
    p.add_argument("--market-scored",required=True)
    p.add_argument("--ballots-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--outsider-predictions",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_year_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1); out[int(y)]=p
    if set(out)!=set(YEARS): raise SystemExit(f"ballot years mismatch {sorted(out)}")
    return out

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def load_market(path):
    required_cols=[
        "year","race_id","horse_id","horse_number","consensus_rank","market_rank","final_win_odds"
    ]
    header=pd.read_csv(path,compression="gzip",nrows=0)
    present=set(header.columns)
    missing=[x for x in required_cols if x not in present]
    if missing: raise SystemExit(f"market source missing required columns: {missing}")
    optional=[x for x in ("confidence_score","confidence_tag") if x in present]
    df=pd.read_csv(path,compression="gzip",usecols=required_cols+optional,dtype={"race_id":str,"horse_id":str})
    if "confidence_score" not in df.columns: df["confidence_score"]=0.0
    if "confidence_tag" not in df.columns: df["confidence_tag"]="NA_NOT_IN_FROZEN_SOURCE"
    df["year"]=pd.to_numeric(df["year"],errors="raise").astype(int)
    df=df[df["year"].isin(YEARS)].copy()
    if 2026 in set(df["year"]): raise SystemExit("2026 sealed")
    for c in ("horse_number","consensus_rank","market_rank","confidence_score","final_win_odds"):
        df[c]=pd.to_numeric(df[c],errors="coerce")
    if df[["horse_number","consensus_rank","market_rank","final_win_odds"]].isna().any().any():
        raise SystemExit("market source has required numeric nulls")
    df["horse_number"]=df["horse_number"].astype(int)
    df["consensus_rank"]=df["consensus_rank"].astype(int)
    df["market_rank"]=df["market_rank"].astype(int)
    df["signed_gap"]=df["market_rank"]-df["consensus_rank"]
    df["abs_gap"]=df["signed_gap"].abs()
    return df

def load_ballots(paths):
    sig={}
    race_candidates=defaultdict(set)
    for year,path in sorted(paths.items()):
        temp=defaultdict(lambda:defaultdict(lambda:{"score":0,"support":0,"v1":0,"v2":0,"v3":0}))
        with gzip.open(path,"rt",encoding="utf-8") as f:
            r=csv.DictReader(f)
            for row in r:
                rid=str(row["race_id"]); cand=str(row["candidate"])
                key=(year,rid)
                race_candidates[key].add(cand)
                for k,w in ((1,3),(2,2),(3,1)):
                    hid=str(row.get(f"top{k}_horse_id") or "")
                    if not hid: continue
                    d=temp[rid][hid]; d["score"]+=w; d["support"]+=1; d[f"v{k}"]+=1
        for rid,hs in temp.items():
            if len(race_candidates[(year,rid)])!=13:
                raise SystemExit(f"expected 13 outsider candidates year={year} race={rid}")
            for hid,d in hs.items():
                sig[(year,rid,hid)]=d
    return sig

def load_p3(path):
    cols=["year","race_id","horse_id","rank","score","p3_delta"]
    p=pd.read_csv(path,compression="gzip",usecols=cols,dtype={"race_id":str,"horse_id":str})
    p["year"]=pd.to_numeric(p["year"],errors="raise").astype(int)
    p=p[p["year"].isin(YEARS)].copy()
    out={}
    for r in p.itertuples(index=False):
        out[(int(r.year),str(r.race_id),str(r.horse_id))]={
            "rank":int(r.rank),"score":int(r.score),"p3_delta":float(r.p3_delta)
        }
    return out

def horse_context(row,outsider,p3):
    key=(int(row.year),str(row.race_id),str(row.horse_id))
    o=outsider.get(key,{"score":0,"support":0,"v1":0,"v2":0,"v3":0})
    pr=p3.get(key)
    valid=int(pr is not None and int(pr["rank"])==int(row.consensus_rank) and int(pr["score"])==int(o["score"]))
    delta=float(pr["p3_delta"]) if valid else 0.0
    gap=int(row.signed_gap)
    align=(1 if gap>0 else -1 if gap<0 else 0)*delta if valid else 0.0
    return {
        "horse_id":str(row.horse_id),"horse_number":int(row.horse_number),
        "king_rank":int(row.consensus_rank),"market_rank":int(row.market_rank),
        "confidence":float(row.confidence_score) if pd.notna(row.confidence_score) else 0.0,
        "win_odds":float(row.final_win_odds),"signed_gap":gap,"abs_gap":abs(gap),
        "outsider_score":int(o["score"]),"outsider_support":int(o["support"]),
        "outsider_v1":int(o["v1"]),"outsider_v2":int(o["v2"]),"outsider_v3":int(o["v3"]),
        "p3_delta":delta,"p3_valid":valid,"alignment":align,
    }

def pair_features(year,rid,a,b,pair_odds,pair_prob_norm,field_size,race_date="",race_class="UNKNOWN",grade="UNKNOWN"):
    vals=[a,b]
    def arr(k): return [float(x[k]) for x in vals]
    kr=arr("king_rank"); mr=arr("market_rank"); conf=arr("confidence"); wo=arr("win_odds")
    oscore=arr("outsider_score"); osup=arr("outsider_support")
    gap=arr("signed_gap"); ag=[abs(x) for x in gap]
    p3=arr("p3_delta"); al=arr("alignment")
    return {
        "year":int(year),"race_id":str(rid),"race_date":str(race_date),
        "selection_numbers":f"{min(a['horse_number'],b['horse_number']):02d}-{max(a['horse_number'],b['horse_number']):02d}",
        "field_size":int(field_size),
        "race_class":str(race_class or "UNKNOWN"),
        "grade":str(grade or "UNKNOWN"),
        "pair_market_odds":float(pair_odds),
        "log_pair_market_odds":math.log(max(float(pair_odds),EPS)),
        "pair_market_prob_norm":float(pair_prob_norm),
        "win_odds_min":min(wo),"win_odds_max":max(wo),
        "log_win_odds_sum":math.log(max(wo[0],EPS))+math.log(max(wo[1],EPS)),
        "market_rank_min":min(mr),"market_rank_max":max(mr),"market_rank_sum":sum(mr),"market_rank_diff":abs(mr[0]-mr[1]),
        "king_rank_min":min(kr),"king_rank_max":max(kr),"king_rank_sum":sum(kr),"king_rank_diff":abs(kr[0]-kr[1]),
        "confidence_mean":sum(conf)/2,"confidence_min":min(conf),"confidence_max":max(conf),
        "king_top3_count":sum(x<=3 for x in kr),"king_top6_count":sum(x<=6 for x in kr),
        "outsider_score_sum":sum(oscore),"outsider_score_max":max(oscore),"outsider_score_min":min(oscore),
        "outsider_support_sum":sum(osup),"outsider_support_max":max(osup),
        "outsider_top1_votes_sum":a["outsider_v1"]+b["outsider_v1"],
        "outsider_top2_votes_sum":a["outsider_v2"]+b["outsider_v2"],
        "outsider_top3_votes_sum":a["outsider_v3"]+b["outsider_v3"],
        "signed_gap_sum":sum(gap),"signed_gap_min":min(gap),"signed_gap_max":max(gap),
        "abs_gap_sum":sum(ag),"abs_gap_max":max(ag),
        "up_count":sum(x>=2 for x in gap),"down_count":sum(x<=-2 for x in gap),"near_count":sum(-1<=x<=1 for x in gap),
        "p3_delta_sum":sum(p3),"p3_delta_min":min(p3),"p3_delta_max":max(p3),
        "p3_valid_count":a["p3_valid"]+b["p3_valid"],
        "outsider_alignment_sum":sum(al),"outsider_alignment_abs_sum":sum(abs(x) for x in al),
        "up_outsider_agree_count":sum(x["signed_gap"]>=2 and x["alignment"]>0 for x in vals),
        "up_outsider_oppose_count":sum(x["signed_gap"]>=2 and x["alignment"]<0 for x in vals),
        "down_outsider_agree_count":sum(x["signed_gap"]<=-2 and x["alignment"]>0 for x in vals),
        "down_outsider_oppose_count":sum(x["signed_gap"]<=-2 and x["alignment"]<0 for x in vals),
        "up_alignment_sum":sum(float(x["alignment"]) for x in vals if x["signed_gap"]>=2),
        "down_alignment_sum":sum(float(x["alignment"]) for x in vals if x["signed_gap"]<=-2),
    }

def build_tickets(market,outsider,p3,backfill_root):
    t0=time.perf_counter()
    market_by_race={(int(y),str(rid)):g.copy() for (y,rid),g in market.groupby(["year","race_id"],sort=False)}
    wanted_by_year={y:set(market.loc[market["year"]==y,"race_id"].astype(str)) for y in YEARS}
    rows=[]; stats=defaultdict(int); year_races=defaultdict(set)
    root=Path(backfill_root)

    for year in YEARS:
        daily=sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz"))
        if not daily: raise SystemExit(f"no BACKFILL daily files for {year}")
        wanted=wanted_by_year[year]
        for day_path in daily:
            date=day_path.name[:10]
            odds_path=root/"data"/"odds"/"daily"/f"{date}.jsonl.gz"
            if not odds_path.exists(): continue
            # Read only races present in the frozen market source.
            day=load_day(day_path,wanted)
            if not day: continue
            oddsday=load_odds_day(odds_path,set(day))
            for rid,pack in day.items():
                key=(year,str(rid)); g=market_by_race.get(key); orec=oddsday.get(str(rid))
                if g is None or orec is None: stats["missing_market_or_odds"]+=1; continue
                horses=[horse_context(r,outsider,p3) for r in g.itertuples(index=False)]
                nums={h["horse_number"] for h in horses}
                if len(nums)!=len(horses) or len(horses)<2: stats["bad_field"]+=1; continue
                expected={tuple(sorted(x)) for x in itertools.combinations(sorted(nums),2)}
                qodds={}
                for bet,ns,price in iter_decoded_odds(orec):
                    if bet=="QUINELLA": qodds[tuple(sorted(ns))]=float(price)
                if set(qodds)!=expected:
                    stats["incomplete_quinella_odds"]+=1; continue
                payouts,present=payout_map(pack)
                if "QUINELLA" not in present: stats["quinella_payout_missing"]+=1; continue
                winners={ns for (bet,ns),pay in payouts.items() if bet=="QUINELLA" and float(pay)>0}
                if len(winners)!=1:
                    stats["deadheat_or_multi_winner_quinella"]+=1; continue
                winner=next(iter(winners))
                if winner not in qodds:
                    stats["winner_pair_not_priced"]+=1; continue

                inv={ns:1.0/max(price,EPS) for ns,price in qodds.items()}
                invsum=sum(inv.values())
                bynum={h["horse_number"]:h for h in horses}
                race_meta=pack.get("race") or {}
                race_class=str(race_meta.get("race_class_normalized") or race_meta.get("race_class") or "UNKNOWN")
                grade=str(race_meta.get("grade") or "UNKNOWN")
                for ns in sorted(expected):
                    a,b=bynum[ns[0]],bynum[ns[1]]
                    row=pair_features(year,rid,a,b,qodds[ns],inv[ns]/invsum,len(horses),date,race_class,grade)
                    row["hit"]=int(ns==winner)
                    row["return_yen_per100"]=float(payouts.get(("QUINELLA",ns),0.0))
                    rows.append(row)
                stats["races_kept"]+=1; stats[f"races_kept_{year}"]+=1
                stats["tickets"]+=len(expected); year_races[year].add(str(rid))
        print("TICKET_BUILD_YEAR_DONE "+json.dumps({"year":year,"races":len(year_races[year]),"tickets":sum(1 for r in rows if r["year"]==year)},separators=(",",":")),flush=True)

    df=pd.DataFrame(rows)
    if df.empty: raise SystemExit("empty ticket table")
    # Exactly one winning quinella per kept race.
    chk=df.groupby(["year","race_id"])["hit"].sum()
    if not (chk==1).all(): raise SystemExit("winner cardinality drift")
    stats["build_seconds"]=time.perf_counter()-t0
    stats["feature_columns"]=len(MODEL_FEATURES["L175_FULL"])
    stats["rows"]=len(df)
    stats["memory_mib"]=float(df.memory_usage(deep=True).sum()/1024/1024)
    return df,dict(stats)

def normalize_by_race(raw,df):
    raw=np.asarray(raw,dtype=float)
    out=np.empty(len(raw),dtype=float)
    for _,idx in df.groupby("race_id",sort=False).groups.items():
        ii=np.asarray(list(idx),dtype=int)
        v=np.clip(raw[ii],EPS,None); s=float(v.sum())
        out[ii]=v/s if s>0 else 1.0/len(ii)
    return out

def evaluate(test,p):
    race_losses=[]; brier=[]; top1=top3=top5=top10=0; races=0
    winner_probs=[]; market_winner_probs=[]
    for rid,idx in test.groupby("race_id",sort=False).groups.items():
        ii=np.asarray(list(idx),dtype=int); sub=test.loc[ii]
        yy=sub["hit"].to_numpy(dtype=int); pos=np.flatnonzero(yy==1)
        if len(pos)!=1: continue
        probs=p[ii]; w=pos[0]
        race_losses.append(-math.log(max(float(probs[w]),EPS)))
        brier.extend((probs-yy)**2)
        order=np.argsort(-probs,kind="stable")
        top1+=int(order[0]==w); top3+=int(w in order[:3]); top5+=int(w in order[:5]); top10+=int(w in order[:10])
        winner_probs.append(float(probs[w]))
        market_winner_probs.append(float(sub["pair_market_prob_norm"].to_numpy()[w]))
        races+=1
    return {
        "races":races,"tickets":len(test),
        "race_log_loss":float(np.mean(race_losses)) if race_losses else None,
        "brier":float(np.mean(brier)) if brier else None,
        "top1_ticket_accuracy_pct":100*top1/races if races else None,
        "winner_in_top3_pct":100*top3/races if races else None,
        "winner_in_top5_pct":100*top5/races if races else None,
        "winner_in_top10_pct":100*top10/races if races else None,
        "mean_winner_probability":float(np.mean(winner_probs)) if winner_probs else None,
        "mean_market_winner_probability":float(np.mean(market_winner_probs)) if market_winner_probs else None,
    }

def fit_predict(name,train,test,seed):
    feats=MODEL_FEATURES[name]
    Xtr=train[feats].astype(float)
    Xte=test[feats].astype(float)
    pair_counts=train.groupby("race_id")["race_id"].transform("size").astype(float)
    weights=1.0/pair_counts.to_numpy()
    model=lgb.LGBMClassifier(
        objective="binary",n_estimators=260,learning_rate=0.04,num_leaves=31,
        min_child_samples=80,subsample=0.9,colsample_bytree=0.9,
        reg_lambda=2.0,reg_alpha=0.1,random_state=seed,n_jobs=1,verbosity=-1,
    )
    t0=time.perf_counter()
    model.fit(Xtr,train["hit"].astype(int),sample_weight=weights)
    raw=model.predict_proba(Xte)[:,1]
    p=normalize_by_race(raw,test.reset_index(drop=True))
    elapsed=time.perf_counter()-t0
    imp=model.booster_.feature_importance(importance_type="gain")
    importance=sorted(
        [{"model":name,"feature":f,"gain":float(g)} for f,g in zip(feats,imp)],
        key=lambda x:-x["gain"]
    )
    return name,p,elapsed,importance[:25]

def main():
    a=args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    market=load_market(a.market_scored)
    ballots=load_ballots(parse_year_paths(a.ballots_year))
    p3=load_p3(a.outsider_predictions)
    tickets,build_stats=build_tickets(market,ballots,p3,a.backfill_root)
    tickets=tickets.reset_index(drop=True)

    fold_rows=[]; importance_rows=[]; timing_rows=[]
    cpu=max(1,os.cpu_count() or 1); workers=min(4,cpu)
    for fi,(test_year,train_years) in enumerate(FOLDS,1):
        train=tickets[tickets["year"].isin(train_years)].copy().reset_index(drop=True)
        test=tickets[tickets["year"]==test_year].copy().reset_index(drop=True)
        if train.empty or test.empty: raise SystemExit(f"empty fold {test_year}")

        # Raw market baseline: normalized inverse quinella odds.
        pm=test["pair_market_prob_norm"].to_numpy(dtype=float)
        row={"test_year":test_year,"train_years":"|".join(map(str,train_years)),"model":"MARKET_RAW","train_tickets":len(train),"parallel_workers":workers}
        row.update(evaluate(test,pm)); fold_rows.append(row)

        futures={}
        t0=time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for mi,name in enumerate(MODEL_FEATURES,1):
                futures[ex.submit(fit_predict,name,train,test,20261000+fi*100+mi)]=name
            for fut in as_completed(futures):
                name,p,elapsed,imp=fut.result()
                metrics=evaluate(test,p)
                row={"test_year":test_year,"train_years":"|".join(map(str,train_years)),"model":name,"train_tickets":len(train),"parallel_workers":workers}
                row.update(metrics); fold_rows.append(row)
                for x in imp:
                    x["test_year"]=test_year; importance_rows.append(x)
                timing_rows.append({"test_year":test_year,"model":name,"fit_predict_seconds":elapsed})
        timing_rows.append({"test_year":test_year,"model":"PARALLEL_WALL","fit_predict_seconds":time.perf_counter()-t0})
        print("L2_TICKET_LAB_FOLD_DONE "+json.dumps({"test_year":test_year,"train_years":train_years,"workers":workers},separators=(",",":")),flush=True)

    fdf=pd.DataFrame(fold_rows)
    # Compare every learned variant with MARKET_ONLY and MARKET_RAW on the same test fold.
    delta_rows=[]
    for year in (2024,2025):
        q=fdf[fdf["test_year"]==year].set_index("model")
        for name in MODEL_FEATURES:
            if name not in q.index: continue
            delta_rows.append({
                "test_year":year,"model":name,
                "delta_race_log_loss_vs_market_raw":float(q.loc[name,"race_log_loss"]-q.loc["MARKET_RAW","race_log_loss"]),
                "delta_brier_vs_market_raw":float(q.loc[name,"brier"]-q.loc["MARKET_RAW","brier"]),
                "delta_top5_pp_vs_market_raw":float(q.loc[name,"winner_in_top5_pct"]-q.loc["MARKET_RAW","winner_in_top5_pct"]),
                "delta_race_log_loss_vs_market_only":float(q.loc[name,"race_log_loss"]-q.loc["MARKET_ONLY","race_log_loss"]),
                "delta_brier_vs_market_only":float(q.loc[name,"brier"]-q.loc["MARKET_ONLY","brier"]),
                "delta_top5_pp_vs_market_only":float(q.loc[name,"winner_in_top5_pct"]-q.loc["MARKET_ONLY","winner_in_top5_pct"]),
            })

    write_csv(out/"fold-metrics.csv",fold_rows)
    write_csv(out/"delta-vs-market.csv",delta_rows)
    write_csv(out/"feature-importance.csv",importance_rows)
    write_csv(out/"timing.csv",timing_rows)

    # Promotion is intentionally disabled. This first run only answers whether L1.75
    # adds ticket-hit information beyond the market benchmark.
    ddf=pd.DataFrame(delta_rows)
    summary={
        "contract":"L2_TICKET_LAB_V1",
        "bet_type":"QUINELLA",
        "scope":"all priced quinella combinations; no TopN or handcrafted ticket filtering",
        "years":list(YEARS),
        "folds":[{"test_year":y,"train_years":list(t)} for y,t in FOLDS],
        "models":{k:v for k,v in MODEL_FEATURES.items()},
        "market_raw":"within-race normalized inverse final quinella odds; historical benchmark only",
        "probability_note":"This is a market-conditional benchmark lab, not a replacement for the market-free L2 CORE horse-strength engine.",
        "outsider_source":"safe 13-Outsider ballots; OOS p3_delta used only when its stored King rank and score match the frozen market-source row",
        "build":build_stats,
        "parallel":{"workers":workers,"cpu_count":cpu,"shared_ticket_table":True,"models_run_concurrently":True},
        "result":{
            "l175_full_beats_market_only_logloss_both_folds":bool(
                len(ddf[ddf["model"]=="L175_FULL"])==2 and
                (ddf.loc[ddf["model"]=="L175_FULL","delta_race_log_loss_vs_market_only"]<0).all()
            ),
            "l175_full_beats_market_raw_logloss_both_folds":bool(
                len(ddf[ddf["model"]=="L175_FULL"])==2 and
                (ddf.loc[ddf["model"]=="L175_FULL","delta_race_log_loss_vs_market_raw"]<0).all()
            )
        },
        "elapsed_seconds":time.perf_counter()-start,
        "2026_locked":True,
        "promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== FOLD METRICS =====")
    print((out/"fold-metrics.csv").read_text(encoding="utf-8"))
    print("===== DELTA =====")
    print((out/"delta-vs-market.csv").read_text(encoding="utf-8"))
    print("===== TIMING =====")
    print((out/"timing.csv").read_text(encoding="utf-8"))
    print("L2_TICKET_LAB_V1_READY")

if __name__=="__main__":
    main()
