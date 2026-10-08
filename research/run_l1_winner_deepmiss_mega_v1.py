#!/usr/bin/env python3
import argparse,csv,json,gc,os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import run_l1_winner_reason_mining_v1 as wr

YEARS=wr.YEARS
EVAL_YEARS=wr.TEST_YEARS
RESCUE_YEARS=(2022,2023,2024,2025)
CPU=wr.CPU
RESCUE_WORKERS=max(1,min(2,CPU))
RESCUE_THREADS=max(1,CPU//RESCUE_WORKERS)

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def rank_bin(rank):
    rank=int(rank)
    if rank<=3: return "1-3"
    if rank<=6: return "4-6"
    if rank<=10: return "7-10"
    return "11+"

def build_oos_frame(test_year,get):
    train=pd.concat([get(test_year-2),get(test_year-1)],ignore_index=True,copy=False)
    test=get(test_year).copy()
    cols=wr.feature_columns(train)
    xtr,xte,cats=wr.prep(train,test,cols)
    model=lgb.LGBMClassifier(**wr.model_params(810000+test_year))
    model.fit(xtr,train["_is_win"].to_numpy(),sample_weight=wr.race_balanced_weights(train),categorical_feature=cats)
    test["model_score"]=model.predict_proba(xte)[:,1].astype("float32")

    ranked=test[["_race_id","_horse_id","model_score"]].sort_values(
        ["_race_id","model_score","_horse_id"],ascending=[True,False,True],kind="mergesort"
    )
    ranked["model_rank"]=ranked.groupby("_race_id",sort=False).cumcount()+1
    rank_map={(r,h):int(k) for r,h,k in ranked[["_race_id","_horse_id","model_rank"]].itertuples(index=False,name=None)}
    test["model_rank"]=[rank_map[(r,h)] for r,h in test[["_race_id","_horse_id"]].itertuples(index=False,name=None)]
    test["field_size"]=test.groupby("_race_id")["_horse_id"].transform("size").astype("int16")
    test["model_rank_norm"]=(test["model_rank"]/test["field_size"]).astype("float32")
    test["model_score_pct"]=wr.pct_in_race(test["model_score"].to_numpy(),test["_race_id"].to_numpy())
    score_mean=test.groupby("_race_id")["model_score"].transform("mean")
    score_std=test.groupby("_race_id")["model_score"].transform("std").replace(0,np.nan)
    test["model_score_z"]=((test["model_score"]-score_mean)/score_std).fillna(0).clip(-6,6).astype("float32")

    fam_scores,top_family,second_family,fam1,fam2,top_feature,top_feature_contrib,_=wr.explain(model,xte,test,cols)
    test["reason_family_1"]=top_family
    test["reason_family_2"]=second_family
    test["reason_family_1_score"]=fam1
    test["reason_family_2_score"]=fam2
    test["reason_family_gap"]=(fam1-fam2).astype("float32")
    test["reason_strength"]=wr.pct_in_race(fam1,test["_race_id"].to_numpy())
    test["top_reason_feature"]=top_feature
    test["top_reason_feature_contrib"]=top_feature_contrib
    test["top_feature_strength"]=wr.pct_in_race(top_feature_contrib,test["_race_id"].to_numpy())
    for family in wr.FAMILIES:
        key=family.lower()
        arr=np.asarray(fam_scores[family],dtype="float32")
        test[f"fam_{key}_score"]=arr
        test[f"fam_{key}_pct"]=wr.pct_in_race(arr,test["_race_id"].to_numpy())

    compact_cols=[
        "_race_id","_horse_id","_race_date","_finish","_is_win","_is_top3",
        "model_score","model_rank","field_size","model_rank_norm","model_score_pct","model_score_z",
        "reason_family_1","reason_family_2","reason_family_1_score","reason_family_2_score",
        "reason_family_gap","reason_strength","top_reason_feature","top_reason_feature_contrib","top_feature_strength",
    ]
    for family in wr.FAMILIES:
        key=family.lower()
        compact_cols += [f"fam_{key}_score",f"fam_{key}_pct"]
    out=test[compact_cols].copy()
    print(
        f"DEEPMISS_OOS_READY year={test_year} runners={len(out)} races={out['_race_id'].nunique()} "
        f"deep_winners={int(((out['_is_win']==1)&(out['model_rank']>=7)).sum())}",flush=True
    )
    del train,test,xtr,xte,model,fam_scores,ranked
    gc.collect()
    return out

def aggregate_rank_bins(frames):
    rows=[]
    for label,df in [(str(y),frames[y]) for y in EVAL_YEARS]+[("ALL",pd.concat([frames[y] for y in EVAL_YEARS],ignore_index=True))]:
        w=df[df["_is_win"]==1].copy()
        w["rank_bin"]=w["model_rank"].map(rank_bin)
        total=len(w)
        for b in ("1-3","4-6","7-10","11+"):
            x=w[w["rank_bin"]==b]
            rows.append({
                "test_year":label,"rank_bin":b,"winners":len(x),"winner_share_pct":100*len(x)/total if total else 0,
                "mean_model_rank":float(x["model_rank"].mean()) if len(x) else np.nan,
                "mean_reason_strength":float(x["reason_strength"].mean()) if len(x) else np.nan,
                "reason_strength_ge_090_pct":100*float((x["reason_strength"]>=0.90).mean()) if len(x) else np.nan,
                "mean_top_feature_strength":float(x["top_feature_strength"].mean()) if len(x) else np.nan,
            })
    return rows

def family_enrichment(frames):
    rows=[]
    for label,df in [(str(y),frames[y]) for y in EVAL_YEARS]+[("ALL",pd.concat([frames[y] for y in EVAL_YEARS],ignore_index=True))]:
        w=df[df["_is_win"]==1].copy()
        w["rank_bin"]=w["model_rank"].map(rank_bin)
        overall=w["reason_family_1"].value_counts(normalize=True).to_dict()
        bins=("1-3","4-6","7-10","11+","7+")
        for b in bins:
            x=w[w["model_rank"]>=7] if b=="7+" else w[w["rank_bin"]==b]
            n=len(x)
            counts=x["reason_family_1"].value_counts().to_dict()
            for family,count in counts.items():
                share=count/n if n else 0
                base=overall.get(family,0)
                rows.append({
                    "test_year":label,"rank_bin":b,"family":family,"winners":count,
                    "winner_share_pct":100*share,"all_winner_family_share_pct":100*base,
                    "enrichment":share/base if base>0 else np.nan,
                })
    return rows

def tail_group_lift(frames,column,min_year=30,min_all=150):
    rows=[]
    pairs=[(str(y),frames[y]) for y in EVAL_YEARS]+[("ALL",pd.concat([frames[y] for y in EVAL_YEARS],ignore_index=True))]
    for label,df in pairs:
        tail=df[df["model_rank"]>=7]
        baseline=float(tail["_is_win"].mean()) if len(tail) else 0
        min_n=min_all if label=="ALL" else min_year
        for key,x in tail.groupby(column,dropna=False):
            if len(x)<min_n: continue
            wrate=float(x["_is_win"].mean())
            rows.append({
                "test_year":label,"group_column":column,"reason":str(key),"runners":len(x),
                "winners":int(x["_is_win"].sum()),"win_rate_pct":100*wrate,
                "tail_baseline_win_rate_pct":100*baseline,"win_lift":wrate/baseline if baseline>0 else np.nan,
                "winner_share_of_tail_pct":100*float(x["_is_win"].sum())/max(1,int(tail["_is_win"].sum())),
            })
    return rows

def tail_strength_lift(frames):
    rows=[]
    for label,df in [(str(y),frames[y]) for y in EVAL_YEARS]+[("ALL",pd.concat([frames[y] for y in EVAL_YEARS],ignore_index=True))]:
        tail=df[df["model_rank"]>=7]
        base=float(tail["_is_win"].mean())
        total_w=int(tail["_is_win"].sum())
        for col in ("reason_strength","top_feature_strength"):
            for th in (0.50,0.75,0.90,0.95):
                x=tail[tail[col]>=th]
                wrate=float(x["_is_win"].mean()) if len(x) else 0
                rows.append({
                    "test_year":label,"signal":col,"threshold":th,"runners":len(x),"winners":int(x["_is_win"].sum()),
                    "win_rate_pct":100*wrate,"tail_baseline_win_rate_pct":100*base,
                    "win_lift":wrate/base if base>0 else np.nan,
                    "tail_winner_capture_pct":100*int(x["_is_win"].sum())/max(1,total_w),
                })
    return rows

BASE_RESCUE=["model_rank_norm","model_score_pct","model_score_z"]
ALL_REASON=BASE_RESCUE+["reason_strength","top_feature_strength","reason_family_gap"]
for _f in wr.FAMILIES:
    _k=_f.lower()
    ALL_REASON += [f"fam_{_k}_score",f"fam_{_k}_pct"]

def rescue_params(seed,threads):
    return dict(
        objective="binary",n_estimators=180,learning_rate=0.04,num_leaves=15,min_child_samples=80,
        subsample=0.9,colsample_bytree=0.9,reg_lambda=6.0,reg_alpha=0.8,
        random_state=seed,n_jobs=threads,deterministic=True,force_col_wise=True,verbosity=-1,
    )

def fit_rescue_variant(name,features,train_df,test_df,seed):
    tr=train_df[train_df["model_rank"]>=6].copy()
    te=test_df[test_df["model_rank"]>=6].copy()
    xtr=tr[features].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32")
    xte=te[features].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32")
    y=tr["_is_win"].to_numpy()
    pos=max(1,int(y.sum())); neg=max(1,len(y)-pos)
    m=lgb.LGBMClassifier(**rescue_params(seed,RESCUE_THREADS),scale_pos_weight=neg/pos)
    m.fit(xtr,y)
    score=m.predict_proba(xte)[:,1]
    out=te[["_race_id","_horse_id","_is_win","model_rank"]].copy()
    out["rescue_score"]=score
    return name,out

def choose_candidate(df,score_col):
    x=df[df["model_rank"]>=6][["_race_id","_horse_id","model_rank","_is_win",score_col]].copy()
    x=x.sort_values(["_race_id",score_col,"_horse_id"],ascending=[True,False,True],kind="mergesort")
    return x.groupby("_race_id",sort=False).head(1).set_index("_race_id")["_horse_id"].to_dict()

def eval_swap(frame,candidate_map,variant,test_year):
    winners=frame[frame["_is_win"]==1]
    total=len(winners)
    base_cap=int((winners["model_rank"]<=6).sum())
    final_cap=0; rescued=0; lost_rank6=0; deep_total=0; deep_pick=0
    for r,h,rank in winners[["_race_id","_horse_id","model_rank"]].itertuples(index=False,name=None):
        chosen=candidate_map.get(r)
        if rank<=5:
            final_cap+=1
        elif chosen==h:
            final_cap+=1
        if rank>=7:
            deep_total+=1
            if chosen==h:
                rescued+=1; deep_pick+=1
        elif rank==6 and chosen!=h:
            lost_rank6+=1
    return {
        "test_year":test_year,"variant":variant,"winners":total,
        "baseline_top6_captured":base_cap,"final_top6_captured":final_cap,
        "baseline_top6_capture_pct":100*base_cap/total if total else 0,
        "final_top6_capture_pct":100*final_cap/total if total else 0,
        "delta_pp":100*(final_cap-base_cap)/total if total else 0,
        "deep_winners":deep_total,"deep_winners_rescued":rescued,
        "deep_pick_rate_pct":100*deep_pick/deep_total if deep_total else 0,
        "rank6_winners_lost":lost_rank6,"net_rescues":rescued-lost_rank6,
    }

def rescue_study(frames):
    variants={"MODEL_ONLY":BASE_RESCUE,"ALL_REASON":ALL_REASON}
    for family in wr.FAMILIES:
        k=family.lower()
        variants[f"FAMILY_{family}"]=BASE_RESCUE+[f"fam_{k}_score",f"fam_{k}_pct","reason_strength"]
    rows=[]
    for y in RESCUE_YEARS:
        train_df=frames[y-1]; test_df=frames[y]
        # Non-ML heuristics first.
        for name,col in [
            ("HEUR_REASON_STRENGTH","reason_strength"),
            ("HEUR_FEATURE_STRENGTH","top_feature_strength"),
            ("HEUR_FORM","fam_form_pct"),
            ("HEUR_ACTOR","fam_actor_pct"),
            ("HEUR_OPPONENT","fam_opponent_pct"),
            ("HEUR_SPEED_PACE","fam_speed_pace_pct"),
        ]:
            rows.append(eval_swap(test_df,choose_candidate(test_df,col),name,y))

        with ThreadPoolExecutor(max_workers=RESCUE_WORKERS) as ex:
            futs={
                ex.submit(fit_rescue_variant,name,features,train_df,test_df,920000+y*100+i):name
                for i,(name,features) in enumerate(variants.items())
            }
            for fut in as_completed(futs):
                name,pred=fut.result()
                cmap=choose_candidate(pred,"rescue_score")
                rows.append(eval_swap(test_df,cmap,name,y))
                print(f"RESCUE_VARIANT_READY year={y} variant={name}",flush=True)
    # pooled weighted by winner counts
    pooled=[]
    for variant in sorted({r["variant"] for r in rows}):
        xs=[r for r in rows if r["variant"]==variant]
        total=sum(r["winners"] for r in xs)
        base=sum(r["baseline_top6_captured"] for r in xs)
        final=sum(r["final_top6_captured"] for r in xs)
        deep=sum(r["deep_winners"] for r in xs)
        rescued=sum(r["deep_winners_rescued"] for r in xs)
        lost=sum(r["rank6_winners_lost"] for r in xs)
        pooled.append({
            "test_year":"ALL_2022_2025","variant":variant,"winners":total,
            "baseline_top6_captured":base,"final_top6_captured":final,
            "baseline_top6_capture_pct":100*base/total if total else 0,
            "final_top6_capture_pct":100*final/total if total else 0,
            "delta_pp":100*(final-base)/total if total else 0,
            "deep_winners":deep,"deep_winners_rescued":rescued,
            "deep_pick_rate_pct":100*rescued/deep if deep else 0,
            "rank6_winners_lost":lost,"net_rescues":rescued-lost,
        })
    return rows+pooled

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--year-file",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--out-dir",required=True)
    a=p.parse_args()
    paths={}
    for spec in a.year_file:
        y,path=spec.split(":",1); paths[int(y)]=path
    if set(paths)!=set(YEARS): raise SystemExit(f"year file mismatch: {sorted(paths)}")
    if 2026 in paths: raise SystemExit("2026 sealed")

    print(json.dumps({
        "contract":"L1_WINNER_DEEPMISS_MEGA_V1_RUNTIME",
        "oos_years":list(EVAL_YEARS),"rescue_years":list(RESCUE_YEARS),
        "cpu":CPU,"rescue_workers":RESCUE_WORKERS,"rescue_threads":RESCUE_THREADS,
        "design":"OOS winner-reason anatomy + tail matched lift + strict previous-year rescue models",
        "2026_locked":True,"uses_odds":False,
    },separators=(",",":")),flush=True)

    cache={}
    def get(y):
        if y not in cache: cache[y]=wr.read_year(paths[y],y)
        return cache[y]

    frames={}
    for y in EVAL_YEARS:
        frames[y]=build_oos_frame(y,get)
        for old in list(cache):
            if old<y-1: del cache[old]
        gc.collect()

    rank_rows=aggregate_rank_bins(frames)
    fam_enrich=family_enrichment(frames)
    tail_family=tail_group_lift(frames,"reason_family_1",30,150)
    tail_feature=tail_group_lift(frames,"top_reason_feature",30,200)
    strength=tail_strength_lift(frames)
    rescue=rescue_study(frames)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"winner-rank-bins.csv",rank_rows)
    write_csv(out/"deepmiss-family-enrichment.csv",fam_enrich)
    write_csv(out/"deepmiss-family-lift.csv",tail_family)
    write_csv(out/"deepmiss-feature-lift.csv",tail_feature)
    write_csv(out/"deepmiss-strength.csv",strength)
    write_csv(out/"rescue-metrics.csv",rescue)

    pooled_bins=[r for r in rank_rows if r["test_year"]=="ALL"]
    pooled_rescue=[r for r in rescue if r["test_year"]=="ALL_2022_2025"]
    best_rescue=max(pooled_rescue,key=lambda r:r["delta_pp"],default=None)
    deep7=sum(r["winners"] for r in pooled_bins if r["rank_bin"] in ("7-10","11+"))
    allw=sum(r["winners"] for r in pooled_bins)
    pooled_tail_family=[r for r in tail_family if r["test_year"]=="ALL"]
    best_tail_family=max(pooled_tail_family,key=lambda r:r["win_lift"],default=None)
    pooled_tail_feature=[r for r in tail_feature if r["test_year"]=="ALL" and r["winners"]>=20]
    best_tail_feature=max(pooled_tail_feature,key=lambda r:r["win_lift"],default=None)
    pooled_strength=[r for r in strength if r["test_year"]=="ALL" and r["signal"]=="reason_strength"]
    s90=next((r for r in pooled_strength if float(r["threshold"])==0.90),None)

    summary={
        "contract":"L1_WINNER_DEEPMISS_MEGA_V1",
        "question":"Why do actual winners missed below baseline top6 win, and can their pre-race reasons rescue them without adding horses?",
        "strictness":"all baseline reason frames are OOS; rescue model for Y trains only on OOS reason frame Y-1; 2021 anatomy only, rescue evaluated 2022-2025",
        "deepmiss_definition":"baseline model rank >= 7",
        "deepmiss_winners":deep7,"all_winners":allw,"deepmiss_share_pct":100*deep7/allw if allw else 0,
        "best_tail_family":best_tail_family,
        "best_tail_feature_min20wins":best_tail_feature,
        "tail_reason_strength_ge_090":s90,
        "best_fixed_top6_rescue":best_rescue,
        "rescue_variants":[r["variant"] for r in pooled_rescue],
        "parallelism":{"snapshot_download":"existing runner parallel download","rescue_workers":RESCUE_WORKERS,"threads_each":RESCUE_THREADS},
        "uses_odds":False,"2026_locked":True,"promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== DEEPMISS MEGA SUMMARY =====",flush=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print("L1_WINNER_DEEPMISS_MEGA_V1_COMPLETE",flush=True)

if __name__=="__main__":
    main()
