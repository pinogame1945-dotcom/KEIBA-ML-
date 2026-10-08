#!/usr/bin/env python3
import argparse,csv,json,gc
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import run_l1_winner_reason_mining_v1 as wr
import run_l1_winner_deepmiss_mega_v1 as dm

YEARS=wr.YEARS
FRAME_YEARS=wr.TEST_YEARS
EVAL_YEARS=(2022,2023,2024,2025)
CPU=wr.CPU
WORKERS=max(1,min(2,CPU))
THREADS=max(1,CPU//WORKERS)

BASE=["model_rank_norm","model_score_pct","model_score_z"]
ALL_REASON=BASE+["reason_strength","top_feature_strength","reason_family_gap"]
for _f in wr.FAMILIES:
    _k=_f.lower()
    ALL_REASON += [f"fam_{_k}_score",f"fam_{_k}_pct"]

FEATURE_SETS={"MODEL_ONLY":BASE,"ALL_REASON":ALL_REASON}
for _f in wr.FAMILIES:
    _k=_f.lower()
    FEATURE_SETS[f"FAMILY_{_f}"]=BASE+[f"fam_{_k}_score",f"fam_{_k}_pct","reason_strength","top_feature_strength"]

HEURISTICS={
    "RANK7":"__RANK7__",
    "HEUR_REASON_STRENGTH":"reason_strength",
    "HEUR_FEATURE_STRENGTH":"top_feature_strength",
    "HEUR_FORM":"fam_form_pct",
    "HEUR_BASE":"fam_base_pct",
    "HEUR_OPPONENT":"fam_opponent_pct",
    "HEUR_SPEED_PACE":"fam_speed_pace_pct",
    "HEUR_ACTOR":"fam_actor_pct",
}
GATES=(("ALWAYS",None),("FIRE50",0.50),("FIRE25",0.75),("FIRE10",0.90))

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

def model_params(seed):
    return dict(
        objective="binary",n_estimators=180,learning_rate=0.04,num_leaves=15,min_child_samples=70,
        subsample=0.9,colsample_bytree=0.9,reg_lambda=6.0,reg_alpha=0.8,
        random_state=seed,n_jobs=THREADS,deterministic=True,force_col_wise=True,verbosity=-1,
    )

def select_candidates(df,score_col,score_values=None):
    tail=df[df["model_rank"]>=7].copy()
    if score_values is not None:
        tail["_warning_score"]=np.asarray(score_values,dtype="float64")
    elif score_col=="__RANK7__":
        tail["_warning_score"]=tail["model_score"].astype("float64")
        tail=tail.sort_values(["_race_id","model_rank","_horse_id"],ascending=[True,True,True],kind="mergesort")
        return tail.groupby("_race_id",sort=False).head(1)
    else:
        tail["_warning_score"]=pd.to_numeric(tail[score_col],errors="coerce").fillna(-1e30).astype("float64")
    tail=tail.sort_values(["_race_id","_warning_score","_horse_id"],ascending=[True,False,True],kind="mergesort")
    return tail.groupby("_race_id",sort=False).head(1)

def fit_ml_variant(name,features,target,train_df,test_df,seed):
    tr=train_df[train_df["model_rank"]>=7].copy()
    te=test_df[test_df["model_rank"]>=7].copy()
    xtr=tr[features].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32")
    xte=te[features].replace([np.inf,-np.inf],np.nan).fillna(0).astype("float32")
    y=tr[target].astype("int8").to_numpy()
    pos=max(1,int(y.sum())); neg=max(1,len(y)-pos)
    m=lgb.LGBMClassifier(**model_params(seed),scale_pos_weight=neg/pos)
    m.fit(xtr,y)
    train_score=m.predict_proba(xtr)[:,1]
    test_score=m.predict_proba(xte)[:,1]
    train_c=select_candidates(tr,None,train_score)
    test_c=select_candidates(te,None,test_score)
    return name,train_c,test_c

def eval_candidates(frame,candidates,variant,gate,test_year,threshold=None):
    races=int(frame["_race_id"].nunique())
    winners=frame[frame["_is_win"]==1]
    podium=frame[frame["_is_top3"]==1]
    base_w=int((winners["model_rank"]<=6).sum())
    base_p=int((podium["model_rank"]<=6).sum())
    deep_w=int((winners["model_rank"]>=7).sum())
    deep_p=int((podium["model_rank"]>=7).sum())

    c=candidates
    if threshold is not None:
        c=c[c["_warning_score"]>=threshold]
    selected=len(c)
    hit_w=int(c["_is_win"].sum()) if selected else 0
    hit_p=int(c["_is_top3"].sum()) if selected else 0
    total_w=len(winners); total_p=len(podium)
    avg_rank=float(c["model_rank"].mean()) if selected else np.nan
    return {
        "test_year":test_year,"variant":variant,"gate":gate,
        "races":races,"warnings":selected,"fire_rate_pct":100*selected/races if races else 0,
        "warning_win_hits":hit_w,"warning_top3_hits":hit_p,
        "warning_win_rate_pct":100*hit_w/selected if selected else 0,
        "warning_top3_rate_pct":100*hit_p/selected if selected else 0,
        "avg_warning_model_rank":avg_rank,
        "deep_winners":deep_w,"deep_top3_horses":deep_p,
        "deep_winner_capture_pct":100*hit_w/deep_w if deep_w else 0,
        "deep_top3_capture_pct":100*hit_p/deep_p if deep_p else 0,
        "baseline_top6_win_capture_pct":100*base_w/total_w if total_w else 0,
        "top6_plus_warning_win_capture_pct":100*(base_w+hit_w)/total_w if total_w else 0,
        "win_capture_delta_pp":100*hit_w/total_w if total_w else 0,
        "baseline_top6_top3_capture_pct":100*base_p/total_p if total_p else 0,
        "top6_plus_warning_top3_capture_pct":100*(base_p+hit_p)/total_p if total_p else 0,
        "top3_capture_delta_pp":100*hit_p/total_p if total_p else 0,
        "wins_per_1000_warnings":1000*hit_w/selected if selected else 0,
        "top3_per_1000_warnings":1000*hit_p/selected if selected else 0,
        "gate_threshold":threshold if threshold is not None else "",
    }

def candidate_rank_rows(candidates,variant,test_year):
    rows=[]
    c=candidates.copy()
    c["rank_bin"]=np.where(c["model_rank"]==7,"7",np.where(c["model_rank"]<=10,"8-10","11+"))
    for b,x in c.groupby("rank_bin"):
        rows.append({
            "test_year":test_year,"variant":variant,"rank_bin":b,"warnings":len(x),
            "win_hits":int(x["_is_win"].sum()),"top3_hits":int(x["_is_top3"].sum()),
            "win_rate_pct":100*float(x["_is_win"].mean()) if len(x) else 0,
            "top3_rate_pct":100*float(x["_is_top3"].mean()) if len(x) else 0,
        })
    return rows

def quantile_threshold(train_candidates,q):
    vals=pd.to_numeric(train_candidates["_warning_score"],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
    return float(vals.quantile(q)) if len(vals) else np.inf

def aggregate(rows,frames):
    pooled=[]
    totalw=sum(int(frames[y]["_is_win"].sum()) for y in EVAL_YEARS)
    totalp=sum(int(frames[y]["_is_top3"].sum()) for y in EVAL_YEARS)
    basew=sum(int(((frames[y]["_is_win"]==1)&(frames[y]["model_rank"]<=6)).sum()) for y in EVAL_YEARS)
    basep=sum(int(((frames[y]["_is_top3"]==1)&(frames[y]["model_rank"]<=6)).sum()) for y in EVAL_YEARS)
    keys=sorted({(r["variant"],r["gate"]) for r in rows})
    for variant,gate in keys:
        xs=[r for r in rows if r["variant"]==variant and r["gate"]==gate]
        races=sum(int(r["races"]) for r in xs)
        warnings=sum(int(r["warnings"]) for r in xs)
        wh=sum(int(r["warning_win_hits"]) for r in xs)
        ph=sum(int(r["warning_top3_hits"]) for r in xs)
        deepw=sum(int(r["deep_winners"]) for r in xs)
        deepp=sum(int(r["deep_top3_horses"]) for r in xs)
        weighted_rank=sum(float(r["avg_warning_model_rank"])*int(r["warnings"]) for r in xs if int(r["warnings"])>0)
        pooled.append({
            "test_year":"ALL_2022_2025","variant":variant,"gate":gate,
            "races":races,"warnings":warnings,"fire_rate_pct":100*warnings/races if races else 0,
            "warning_win_hits":wh,"warning_top3_hits":ph,
            "warning_win_rate_pct":100*wh/warnings if warnings else 0,
            "warning_top3_rate_pct":100*ph/warnings if warnings else 0,
            "avg_warning_model_rank":weighted_rank/warnings if warnings else np.nan,
            "deep_winners":deepw,"deep_top3_horses":deepp,
            "deep_winner_capture_pct":100*wh/deepw if deepw else 0,
            "deep_top3_capture_pct":100*ph/deepp if deepp else 0,
            "baseline_top6_win_capture_pct":100*basew/totalw if totalw else 0,
            "top6_plus_warning_win_capture_pct":100*(basew+wh)/totalw if totalw else 0,
            "win_capture_delta_pp":100*wh/totalw if totalw else 0,
            "baseline_top6_top3_capture_pct":100*basep/totalp if totalp else 0,
            "top6_plus_warning_top3_capture_pct":100*(basep+ph)/totalp if totalp else 0,
            "top3_capture_delta_pp":100*ph/totalp if totalp else 0,
            "wins_per_1000_warnings":1000*wh/warnings if warnings else 0,
            "top3_per_1000_warnings":1000*ph/warnings if warnings else 0,
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
        "contract":"L1_WARNING_SEAT_V1_RUNTIME","frame_years":list(FRAME_YEARS),"eval_years":list(EVAL_YEARS),
        "cpu":CPU,"workers":WORKERS,"threads_each":THREADS,
        "design":"Top6 frozen + one outside warning seat; prior-year-only warning learning and gates",
        "uses_odds":False,"2026_locked":True,
    },separators=(",",":")),flush=True)

    cache={}
    def get(y):
        if y not in cache: cache[y]=wr.read_year(paths[y],y)
        return cache[y]

    frames={}
    for y in FRAME_YEARS:
        frames[y]=dm.build_oos_frame(y,get)
        for old in list(cache):
            if old<y-1: del cache[old]
        gc.collect()

    rows=[]; rank_rows=[]
    for y in EVAL_YEARS:
        train_df=frames[y-1]
        test_df=frames[y]

        # Simple warning rules.
        for name,col in HEURISTICS.items():
            train_c=select_candidates(train_df,col)
            test_c=select_candidates(test_df,col)
            rank_rows += candidate_rank_rows(test_c,name,y)
            for gate,q in GATES:
                th=None if q is None else quantile_threshold(train_c,q)
                rows.append(eval_candidates(test_df,test_c,name,gate,y,th))

        # ML warning rules: train only on previous OOS year.
        tasks=[]
        for target,label in [("_is_win","WIN"),("_is_top3","TOP3")]:
            for i,(base_name,features) in enumerate(FEATURE_SETS.items()):
                name=f"{label}_{base_name}"
                tasks.append((name,features,target,930000+y*100+i+(0 if label=="WIN" else 50)))

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs=[ex.submit(fit_ml_variant,name,features,target,train_df,test_df,seed) for name,features,target,seed in tasks]
            for fut in as_completed(futs):
                name,train_c,test_c=fut.result()
                rank_rows += candidate_rank_rows(test_c,name,y)
                for gate,q in GATES:
                    th=None if q is None else quantile_threshold(train_c,q)
                    rows.append(eval_candidates(test_df,test_c,name,gate,y,th))
                print(f"WARNING_VARIANT_READY year={y} variant={name}",flush=True)

    metrics=aggregate(rows,frames)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"warning-metrics.csv",metrics)
    write_csv(out/"warning-rank-bins.csv",rank_rows)

    pooled=[r for r in metrics if r["test_year"]=="ALL_2022_2025"]
    always=[r for r in pooled if r["gate"]=="ALWAYS"]
    gated=[r for r in pooled if r["gate"]!="ALWAYS" and r["warnings"]>0]
    rank7=next((r for r in always if r["variant"]=="RANK7"),None)
    best_win=max(always,key=lambda r:r["win_capture_delta_pp"],default=None)
    best_top3=max(always,key=lambda r:r["top3_capture_delta_pp"],default=None)
    best_eff_win=max(gated,key=lambda r:r["wins_per_1000_warnings"],default=None)
    best_eff_top3=max(gated,key=lambda r:r["top3_per_1000_warnings"],default=None)

    # Stability: count years where candidate beats RANK7 at same gate.
    stability=[]
    for variant,gate in sorted({(r["variant"],r["gate"]) for r in rows if r["variant"]!="RANK7"}):
        wins=0; top3s=0; years=0
        for y in EVAL_YEARS:
            arow=next((r for r in rows if r["test_year"]==y and r["variant"]==variant and r["gate"]==gate),None)
            brow=next((r for r in rows if r["test_year"]==y and r["variant"]=="RANK7" and r["gate"]==gate),None)
            if not arow or not brow: continue
            years+=1
            if arow["warning_win_rate_pct"]>brow["warning_win_rate_pct"]: wins+=1
            if arow["warning_top3_rate_pct"]>brow["warning_top3_rate_pct"]: top3s+=1
        stability.append({"variant":variant,"gate":gate,"years":years,"beat_rank7_win_rate_years":wins,"beat_rank7_top3_rate_years":top3s})
    write_csv(out/"warning-stability.csv",stability)

    summary={
        "contract":"L1_WARNING_SEAT_V1",
        "question":"Can a seventh, non-destructive warning horse outside frozen Top6 add useful winner/top3 coverage, and beat simply taking rank7?",
        "strictness":"baseline reason frames OOS; warning ML and gate thresholds for Y use only OOS frame Y-1; 2026 sealed",
        "rank7_control":rank7,
        "best_always_win_capture":best_win,
        "best_always_top3_capture":best_top3,
        "best_gated_win_efficiency":best_eff_win,
        "best_gated_top3_efficiency":best_eff_top3,
        "variants":sorted({r["variant"] for r in pooled}),
        "gates":["ALWAYS","FIRE50","FIRE25","FIRE10"],
        "parallelism":{"workers":WORKERS,"threads_each":THREADS,"snapshot_download":"existing metadata-first parallel runner"},
        "uses_odds":False,"2026_locked":True,"promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== WARNING SEAT SUMMARY =====",flush=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print("L1_WARNING_SEAT_V1_COMPLETE",flush=True)

if __name__=="__main__":
    main()
