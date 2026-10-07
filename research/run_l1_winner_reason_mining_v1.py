#!/usr/bin/env python3
import argparse,csv,gzip,json,gc,os
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

YEARS=(2019,2020,2021,2022,2023,2024,2025)
TEST_YEARS=(2021,2022,2023,2024,2025)
LOCKED_YEAR=2026
CPU=max(1,int(os.environ.get("L1_THREADS","0") or 0) or (os.cpu_count() or 2))
SHAP_CHUNK=max(512,int(os.environ.get("WINNER_REASON_SHAP_CHUNK","4096")))
TOPK_PER_FAMILY=3
META=("_race_id","_horse_id","_race_date","_finish","_is_win","_is_top3")
FORBIDDEN={
    "actual_start_time","jockey_id","trainer_id",
    "pedigree_sire_id","pedigree_dam_id","pedigree_siresire_id","pedigree_damsire_id",
    "finish_position","target_finish_position","finish_time_ms","target_finish_time_ms",
    "last_3f","target_last_3f","prize_money","target_prize_money",
    "final_win_odds","market_final_win_odds","final_popularity","market_final_popularity",
    "payout","payout_yen","market_payout","race_id","horse_id","owner_id","breeder_id",
}
CATEGORICAL={
    "venue_code","discipline","surface","direction","weather","track_condition","sex",
    "backfill_course_layout","backfill_race_class_normalized","backfill_grade",
    "backfill_sex_condition","backfill_weight_rule",
}
FAMILY_PREFIXES={
    "OPPONENT":("opponent_","network_"),
    "SPEED_PACE":("lap_","style_","timepace_"),
    "DISTANCE":("distx_",),
    "CONDITION":("backfill_",),
    "FORM":("auto_",),
    "ACTOR":("actor_",),
    "PEDIGREE":("ped_",),
}
FAMILIES=("BASE","FORM","OPPONENT","SPEED_PACE","DISTANCE","CONDITION","ACTOR","PEDIGREE")


def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")


def feature_family(name):
    for family,prefixes in FAMILY_PREFIXES.items():
        if any(name.startswith(p) for p in prefixes):
            return family
    return "BASE"


def read_year(path,year):
    rows=[]
    with open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            feat=dict(r.get("features") or {})
            date=str(feat.get("race_date") or r.get("race_date") or "")[:10]
            if not date.startswith(str(year)+"-"):
                raise SystemExit(f"year drift expected={year} date={date}")
            target=r.get("target") or {}
            try:
                finish=int(float(target.get("finish_position")))
            except (TypeError,ValueError):
                continue
            if finish<1:
                continue
            feat.pop("race_date",None)
            for k in FORBIDDEN:
                feat.pop(k,None)
            dt=pd.to_datetime(date,errors="coerce")
            if pd.isna(dt):
                continue
            feat["race_month"]=int(dt.month)
            feat["race_day_of_year"]=int(dt.dayofyear)
            rows.append({
                "_race_id":str(r.get("race_id") or ""),
                "_horse_id":str(r.get("horse_id") or ""),
                "_race_date":date,
                "_finish":finish,
                "_is_win":int(finish==1),
                "_is_top3":int(finish<=3),
                **feat,
            })
    df=pd.DataFrame.from_records(rows)
    if df.empty:
        raise SystemExit(f"empty year {year}")
    if (df["_race_id"].eq("")|df["_horse_id"].eq("")).any():
        raise SystemExit(f"missing identity y={year}")
    race_stats=df.groupby("_race_id")["_is_win"].agg(["sum","count"])
    good=race_stats[(race_stats["sum"]>=1)&(race_stats["count"]>race_stats["sum"])].index
    df=df[df["_race_id"].isin(set(good))].copy()
    for c in df.columns:
        if c in META or c in CATEGORICAL:
            continue
        df[c]=pd.to_numeric(df[c],errors="coerce")
        if df[c].dtype.kind in "fc":
            df[c]=df[c].astype("float32")
    print(f"WINNER_REASON_YEAR_READY year={year} rows={len(df)} races={df['_race_id'].nunique()} cols={len(df.columns)}",flush=True)
    return df


def feature_columns(df):
    return sorted(c for c in df.columns if c not in META and c not in FORBIDDEN)


def prep(train,test,cols):
    xtr=train.reindex(columns=cols).copy()
    xte=test.reindex(columns=cols).copy()
    cats=[]
    for c in cols:
        if c in CATEGORICAL:
            tv=xtr[c].astype("string").fillna("__MISSING__")
            levels=sorted(set(tv.tolist()))
            xtr[c]=pd.Categorical(tv,categories=levels)
            vv=xte[c].astype("string").fillna("__MISSING__")
            xte[c]=pd.Categorical(vv,categories=levels)
            cats.append(c)
        else:
            xtr[c]=pd.to_numeric(xtr[c],errors="coerce").astype("float32")
            xte[c]=pd.to_numeric(xte[c],errors="coerce").astype("float32")
    return xtr,xte,cats


def race_balanced_weights(df):
    stats=df.groupby("_race_id")["_is_win"].agg(["sum","count"])
    pos=df["_race_id"].map(stats["sum"]).astype(float)
    neg=df["_race_id"].map(stats["count"]-stats["sum"]).astype(float)
    y=df["_is_win"].to_numpy()
    w=np.where(y==1,0.5/pos.to_numpy(),0.5/neg.to_numpy())
    return w.astype("float32")


def model_params(seed):
    return dict(
        objective="binary",n_estimators=320,learning_rate=0.035,num_leaves=31,
        min_child_samples=70,subsample=0.90,colsample_bytree=0.82,
        reg_lambda=6.0,reg_alpha=0.8,random_state=seed,n_jobs=CPU,
        deterministic=True,force_col_wise=True,verbosity=-1,
    )


def rank_metrics(df,score_col="model_score"):
    x=df[["_race_id","_horse_id","_finish","_is_win","_is_top3",score_col]].copy()
    x=x.sort_values(["_race_id",score_col,"_horse_id"],ascending=[True,False,True],kind="mergesort")
    x["model_rank"]=x.groupby("_race_id",sort=False).cumcount()+1
    top1=x[x["model_rank"]==1]
    winners=x[x["_is_win"]==1]
    return x["model_rank"].to_numpy(),{
        "races":int(x["_race_id"].nunique()),
        "winners":int(x["_is_win"].sum()),
        "top1_win_pct":100*float(top1["_is_win"].mean()),
        "top1_top3_pct":100*float(top1["_is_top3"].mean()),
        "winner_top3_capture_pct":100*float((winners["model_rank"]<=3).mean()),
        "winner_top6_capture_pct":100*float((winners["model_rank"]<=6).mean()),
        "mean_winner_rank":float(winners["model_rank"].mean()),
    }


def topk_positive_sum(arr,k):
    if arr.shape[1]==0:
        return np.zeros(arr.shape[0],dtype="float32")
    pos=np.maximum(arr,0.0)
    if pos.shape[1]<=k:
        return pos.sum(axis=1,dtype="float32")
    idx=pos.shape[1]-k
    return np.partition(pos,idx,axis=1)[:,idx:].sum(axis=1,dtype="float32")


def explain(model,xte,raw_test,cols):
    n=len(xte)
    family_scores={f:np.zeros(n,dtype="float32") for f in FAMILIES}
    top_feature=np.empty(n,dtype=object)
    top_feature_contrib=np.zeros(n,dtype="float32")
    winner_details={}
    family_indices={f:np.array([i for i,c in enumerate(cols) if feature_family(c)==f],dtype=int) for f in FAMILIES}
    booster=model.booster_
    for start in range(0,n,SHAP_CHUNK):
        end=min(n,start+SHAP_CHUNK)
        contrib=np.asarray(booster.predict(xte.iloc[start:end],pred_contrib=True),dtype="float32")[:,:-1]
        local_argmax=np.argmax(contrib,axis=1)
        local_max=contrib[np.arange(len(contrib)),local_argmax]
        top_feature[start:end]=[cols[i] for i in local_argmax]
        top_feature_contrib[start:end]=local_max
        for family,idxs in family_indices.items():
            family_scores[family][start:end]=topk_positive_sum(contrib[:,idxs],TOPK_PER_FAMILY)
        win_local=np.flatnonzero(raw_test.iloc[start:end]["_is_win"].to_numpy()==1)
        for j in win_local:
            global_i=start+j
            order=np.argsort(contrib[j])[::-1][:3]
            items=[]
            for i in order:
                value=raw_test.iloc[global_i][cols[i]] if cols[i] in raw_test.columns else None
                if pd.isna(value):
                    value=""
                elif isinstance(value,(np.floating,np.integer)):
                    value=float(value)
                else:
                    value=str(value)
                items.append((cols[i],float(contrib[j,i]),value,feature_family(cols[i])))
            winner_details[global_i]=items
        del contrib
    fam_matrix=np.column_stack([family_scores[f] for f in FAMILIES])
    order=np.argsort(fam_matrix,axis=1)[:,::-1]
    top_family=np.array(FAMILIES,dtype=object)[order[:,0]]
    second_family=np.array(FAMILIES,dtype=object)[order[:,1]]
    family1_score=fam_matrix[np.arange(n),order[:,0]]
    family2_score=fam_matrix[np.arange(n),order[:,1]]
    no_positive=family1_score<=0
    top_family[no_positive]="NONE"
    second_family[no_positive]="NONE"
    top_feature[top_feature_contrib<=0]="NONE"
    return family_scores,top_family,second_family,family1_score,family2_score,top_feature,top_feature_contrib,winner_details


def pct_in_race(values,race_ids):
    s=pd.Series(values)
    g=pd.Series(race_ids)
    rank=s.groupby(g,sort=False).rank(method="average",ascending=True)
    n=s.groupby(g,sort=False).transform("size")
    return ((rank-1.0)/(n-1.0).clip(lower=1.0)).astype("float32").to_numpy()


def add_counter(counter,key,is_win,is_top3,expected_win):
    rec=counter[key]
    rec[0]+=1
    rec[1]+=int(is_win)
    rec[2]+=int(is_top3)
    rec[3]+=float(expected_win)


def counter_rows(counter,scope,test_year,total_runners,total_winners,min_runners=1):
    rows=[]
    for key,(runners,winners,top3,expected_wins) in counter.items():
        if runners<min_runners:
            continue
        wr=winners/runners
        expected_rate=(expected_wins/runners) if runners else 0.0
        rows.append({
            "scope":scope,"test_year":test_year,"reason":key,
            "runners":runners,"winners":winners,"top3":top3,
            "win_rate_pct":100*wr,"top3_rate_pct":100*(top3/runners),
            "same_race_expected_wins":expected_wins,
            "same_race_baseline_win_rate_pct":100*expected_rate,
            "win_lift":winners/expected_wins if expected_wins>0 else np.nan,
            "winner_share_pct":100*(winners/total_winners) if total_winners else 0.0,
        })
    return rows


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def merge_counter(dst,src):
    for k,v in src.items():
        d=dst[k]
        d[0]+=v[0]; d[1]+=v[1]; d[2]+=v[2]; d[3]+=v[3]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--year-file",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--out-dir",required=True)
    a=p.parse_args()
    paths={}
    for spec in a.year_file:
        y,path=spec.split(":",1); paths[int(y)]=path
    if set(paths)!=set(YEARS):
        raise SystemExit(f"year file mismatch: {sorted(paths)}")
    if LOCKED_YEAR in paths:
        raise SystemExit("2026 sealed")

    print(json.dumps({
        "contract":"L1_WINNER_REASON_MINING_V1_RUNTIME",
        "years":list(YEARS),"test_years":list(TEST_YEARS),"cpu":CPU,
        "method":"past-only win classifier + OOS SHAP contrast",
        "family_score":"top-3 positive SHAP sum per family",
        "uses_odds":False,"causal_claim":False,"2026_locked":True,
    },separators=(",",":")),flush=True)

    cache={}
    def get(y):
        if y not in cache:
            cache[y]=read_year(paths[y],y)
        return cache[y]

    fold_metrics=[]
    winner_rows=[]
    family_rows=[]; signature_rows=[]; feature_rows=[]; strength_rows=[]; importance_rows=[]
    pooled_family=defaultdict(lambda:[0,0,0,0.0])
    pooled_signature=defaultdict(lambda:[0,0,0,0.0])
    pooled_feature=defaultdict(lambda:[0,0,0,0.0])
    pooled_strength=defaultdict(lambda:[0,0,0,0.0])
    pooled_total_runners=0; pooled_total_winners=0

    for test_year in TEST_YEARS:
        train=pd.concat([get(test_year-2),get(test_year-1)],ignore_index=True,copy=False)
        test=get(test_year).copy()
        cols=feature_columns(train)
        xtr,xte,cats=prep(train,test,cols)
        weights=race_balanced_weights(train)
        model=lgb.LGBMClassifier(**model_params(700000+test_year))
        model.fit(xtr,train["_is_win"].to_numpy(),sample_weight=weights,categorical_feature=cats)
        test["model_score"]=model.predict_proba(xte)[:,1].astype("float32")

        ranked=test[["_race_id","_horse_id","_finish","_is_win","_is_top3","model_score"]].copy()
        ranked=ranked.sort_values(["_race_id","model_score","_horse_id"],ascending=[True,False,True],kind="mergesort")
        ranked["model_rank"]=ranked.groupby("_race_id",sort=False).cumcount()+1
        rank_map={(r,h):int(k) for r,h,k in ranked[["_race_id","_horse_id","model_rank"]].itertuples(index=False,name=None)}
        test["model_rank"]=[rank_map[(r,h)] for r,h in test[["_race_id","_horse_id"]].itertuples(index=False,name=None)]
        top1=ranked[ranked["model_rank"]==1]
        winners_ranked=ranked[ranked["_is_win"]==1]
        fold_metrics.append({
            "test_year":test_year,"train_years":f"{test_year-2},{test_year-1}",
            "races":int(test["_race_id"].nunique()),"runners":len(test),"winners":int(test["_is_win"].sum()),
            "features":len(cols),
            "top1_win_pct":100*float(top1["_is_win"].mean()),
            "top1_top3_pct":100*float(top1["_is_top3"].mean()),
            "winner_top3_capture_pct":100*float((winners_ranked["model_rank"]<=3).mean()),
            "winner_top6_capture_pct":100*float((winners_ranked["model_rank"]<=6).mean()),
            "mean_winner_rank":float(winners_ranked["model_rank"].mean()),
        })

        (family_scores,top_family,second_family,fam1,fam2,top_feature,top_feature_contrib,winner_details)=explain(model,xte,test,cols)
        test["reason_family_1"]=top_family
        test["reason_family_2"]=second_family
        test["reason_family_1_score"]=fam1
        test["reason_family_2_score"]=fam2
        test["reason_family_1_pct"]=pct_in_race(fam1,test["_race_id"].to_numpy())
        test["top_reason_feature"]=top_feature
        test["top_reason_feature_contrib"]=top_feature_contrib
        test["top_reason_feature_pct"]=pct_in_race(top_feature_contrib,test["_race_id"].to_numpy())

        fam_counter=defaultdict(lambda:[0,0,0,0.0])
        sig_counter=defaultdict(lambda:[0,0,0,0.0])
        feat_counter=defaultdict(lambda:[0,0,0,0.0])
        strength_counter=defaultdict(lambda:[0,0,0,0.0])
        race_counts=test.groupby("_race_id")["_is_win"].agg(["sum","count"])
        expected_by_race=(race_counts["sum"]/race_counts["count"]).to_dict()
        for i,row in test.iterrows():
            win=int(row["_is_win"]); top3=int(row["_is_top3"])
            expected=float(expected_by_race[row["_race_id"]])
            fam=str(row["reason_family_1"])
            sig=f"{row['reason_family_1']}>{row['reason_family_2']}"
            feat=str(row["top_reason_feature"])
            add_counter(fam_counter,fam,win,top3,expected)
            add_counter(sig_counter,sig,win,top3,expected)
            add_counter(feat_counter,feat,win,top3,expected)
            for threshold in (0.50,0.75,0.90):
                if float(row["reason_family_1_pct"])>=threshold:
                    add_counter(strength_counter,f"family_pct>={threshold:.2f}",win,top3,expected)
                if float(row["top_reason_feature_pct"])>=threshold:
                    add_counter(strength_counter,f"feature_pct>={threshold:.2f}",win,top3,expected)

        total_runners=len(test); total_winners=int(test["_is_win"].sum())
        family_rows.extend(counter_rows(fam_counter,"FAMILY",test_year,total_runners,total_winners,25))
        signature_rows.extend(counter_rows(sig_counter,"SIGNATURE",test_year,total_runners,total_winners,25))
        feature_rows.extend(counter_rows(feat_counter,"FEATURE",test_year,total_runners,total_winners,50))
        strength_rows.extend(counter_rows(strength_counter,"STRENGTH",test_year,total_runners,total_winners,25))
        merge_counter(pooled_family,fam_counter); merge_counter(pooled_signature,sig_counter)
        merge_counter(pooled_feature,feat_counter); merge_counter(pooled_strength,strength_counter)
        pooled_total_runners+=total_runners; pooled_total_winners+=total_winners

        gain=model.booster_.feature_importance(importance_type="gain")
        total_gain=float(gain.sum()) or 1.0
        for c,g in sorted(zip(cols,gain),key=lambda x:x[1],reverse=True)[:80]:
            importance_rows.append({
                "test_year":test_year,"feature":c,"family":feature_family(c),
                "gain":float(g),"gain_share_pct":100*float(g)/total_gain,
            })

        for pos in np.flatnonzero(test["_is_win"].to_numpy()==1):
            row=test.iloc[pos]
            details=winner_details[pos]
            rec={
                "test_year":test_year,
                "race_id":row["_race_id"],"horse_id":row["_horse_id"],
                "race_date":row["_race_date"],"finish":int(row["_finish"]),
                "model_score":float(row["model_score"]),"model_rank":int(row["model_rank"]),
                "reason_family_1":row["reason_family_1"],"reason_family_2":row["reason_family_2"],
                "reason_family_1_score":float(row["reason_family_1_score"]),
                "reason_family_1_pct":float(row["reason_family_1_pct"]),
            }
            for j,(feat,contrib,value,family) in enumerate(details,1):
                rec[f"reason_feature_{j}"]=feat
                rec[f"reason_feature_{j}_family"]=family
                rec[f"reason_feature_{j}_contrib"]=contrib
                rec[f"reason_feature_{j}_value"]=value
            winner_rows.append(rec)

        print(
            f"WINNER_REASON_FOLD_COMPLETE test_year={test_year} races={test['_race_id'].nunique()} "
            f"winners={total_winners} top1_win={fold_metrics[-1]['top1_win_pct']:.3f}",flush=True
        )
        del xtr,xte,model,train,test,ranked,winners_ranked,top1,family_scores,winner_details
        for y in list(cache):
            if y<test_year-1:
                del cache[y]
        gc.collect()

    family_rows.extend(counter_rows(pooled_family,"FAMILY","ALL",pooled_total_runners,pooled_total_winners,100))
    signature_rows.extend(counter_rows(pooled_signature,"SIGNATURE","ALL",pooled_total_runners,pooled_total_winners,200))
    feature_rows.extend(counter_rows(pooled_feature,"FEATURE","ALL",pooled_total_runners,pooled_total_winners,250))
    strength_rows.extend(counter_rows(pooled_strength,"STRENGTH","ALL",pooled_total_runners,pooled_total_winners,100))

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"fold-metrics.csv",fold_metrics)
    write_csv(out/"winner-reasons.csv",winner_rows)
    write_csv(out/"family-lift.csv",sorted(family_rows,key=lambda r:(str(r["test_year"]),-float(r["win_lift"]))))
    write_csv(out/"signature-lift.csv",sorted(signature_rows,key=lambda r:(str(r["test_year"]),-float(r["win_lift"]))))
    write_csv(out/"feature-lift.csv",sorted(feature_rows,key=lambda r:(str(r["test_year"]),-float(r["win_lift"]))))
    write_csv(out/"strength-lift.csv",strength_rows)
    write_csv(out/"global-feature-importance.csv",importance_rows)

    pooled_family_rows=[r for r in family_rows if str(r["test_year"])=="ALL"]
    pooled_sig_rows=[r for r in signature_rows if str(r["test_year"])=="ALL"]
    pooled_feat_rows=[r for r in feature_rows if str(r["test_year"])=="ALL"]
    best_family=max(pooled_family_rows,key=lambda r:r["win_lift"],default=None)
    best_signature=max(pooled_sig_rows,key=lambda r:r["win_lift"],default=None)
    best_feature=max(pooled_feat_rows,key=lambda r:r["win_lift"],default=None)
    winner_family_counts={str(k):int(v) for k,v in pd.Series([r["reason_family_1"] for r in winner_rows]).value_counts().to_dict().items()}
    summary={
        "contract":"L1_WINNER_REASON_MINING_V1",
        "question":"What pre-race information repeatedly distinguished actual winners from same-race losers?",
        "interpretation":"associational reason mining, not causal proof",
        "strict_fold":"holdout Y explained only by a model trained on Y-2 and Y-1; Y labels never fit the model",
        "reason_definition":"LightGBM SHAP positive contribution; family score is top-3 positive contributions within each family",
        "winner_reason_output":"one row per actual first-place horse, including top families and top three feature contributions",
        "contrast":"every reason is also counted on same-race losers; lift uses race-normalized expected wins within the same holdout races",
        "dead_heat_safe":True,
        "tie_safe_ranking":"score desc then horse_id; never source row order",
        "uses_odds":False,
        "2026_locked":True,
        "cpu":CPU,
        "shap_chunk":SHAP_CHUNK,
        "pooled_runners":pooled_total_runners,
        "pooled_winners":pooled_total_winners,
        "winner_top_family_counts":winner_family_counts,
        "best_pooled_family":best_family,
        "best_pooled_signature":best_signature,
        "best_pooled_feature":best_feature,
        "promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== WINNER REASON SUMMARY =====",flush=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print("L1_WINNER_REASON_MINING_V1_COMPLETE",flush=True)

if __name__=="__main__":
    main()
