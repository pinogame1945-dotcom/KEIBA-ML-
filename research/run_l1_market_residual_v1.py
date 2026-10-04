#!/usr/bin/env python3
import argparse,csv,gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path

EXPERTS=("core4","pedlegacy","condition","full","light","pedv1","condrc")
YEARS=(2021,2022,2023,2024,2025)
EXPECTED_RACES=3456

def finite(v):
    try:
        if isinstance(v,str): v=v.replace(",","").strip()
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError): return None

def final_tuple(raw):
    if not isinstance(raw,list) or len(raw)<3:return None
    return raw[3:6] if len(raw)>=6 else raw[:3]

def load_day(path,wanted):
    out={}
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line); rid=str((r.get("race") or {}).get("race_id") or "")
            if rid in wanted: out[rid]=r
    return out

def load_odds(path,wanted):
    out={}
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line); rid=str(r.get("race_id") or "")
            if rid in wanted: out[rid]=r
    return out

def load_state(path):
    rows=[]
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    if len(rows)!=EXPECTED_RACES: raise SystemExit(f"state coverage {path}: {len(rows)}")
    return rows

def write_gz_csv(path,rows,fields):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,"wt",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def cmd_year(a):
    year=int(a.year)
    if year==2026: raise SystemExit("2026 sealed")
    state=load_state(a.state)
    bydate=defaultdict(list)
    for rec in state:
        bydate[str(rec["race_date"])].append(rec)
    root=Path(a.backfill_root)
    out=[]; skipped=[]
    for date,recs in sorted(bydate.items()):
        wanted={str(x["race_id"]) for x in recs}
        dpath=root/"data"/"daily"/f"{date}.jsonl.gz"
        opath=root/"data"/"odds"/"daily"/f"{date}.jsonl.gz"
        if not dpath.exists() or not opath.exists(): raise SystemExit(f"missing files {date}")
        day=load_day(dpath,wanted); oddsday=load_odds(opath,wanted)
        for rec in recs:
            rid=str(rec["race_id"]); pack=day.get(rid); orec=oddsday.get(rid)
            if pack is None or orec is None:
                skipped.append((rid,date,"missing_pack_or_odds")); continue
            hno={}
            for e in pack.get("entries") or []:
                hid=str(e.get("horse_id") or "")
                try:no=int(e.get("horse_number"))
                except (TypeError,ValueError): continue
                if hid and no>0:hno[hid]=no
            raw=((orec.get("odds") or {}).get("1") or {})
            prices={}
            for k,v in raw.items():
                ks=str(k)
                if not ks.isascii() or not ks.isdigit():continue
                tup=final_tuple(v)
                if not tup:continue
                odd=finite(tup[0])
                if odd is not None and odd>0:prices[int(ks)]=odd
            horses=[str(h) for h in rec["horses"]]
            if any(h not in hno for h in horses):
                skipped.append((rid,date,"horse_number_missing")); continue
            if set(prices)!=set(hno[h] for h in horses):
                skipped.append((rid,date,f"win_odds_incomplete:{len(prices)}/{len(horses)}")); continue
            invp={h:1.0/prices[hno[h]] for h in horses}
            s=sum(invp.values())
            if s<=0:
                skipped.append((rid,date,"implied_sum_zero")); continue
            pnorm={h:invp[h]/s for h in horses}
            odds_rank={}
            vals=[prices[hno[h]] for h in horses]
            for h in horses: odds_rank[h]=1+sum(1 for x in vals if x<prices[hno[h]])
            winner=set(map(str,rec["winner"])); podium=set(map(str,rec["podium"]))
            n=len(horses)
            rank_matrix=rec["rank_matrix"]
            if len(rank_matrix)!=7 or any(len(x)!=n for x in rank_matrix):
                raise SystemExit(f"rank matrix drift {rid}")
            for j,h in enumerate(horses):
                ranks=[float(rank_matrix[i][j]) for i in range(7)]
                meanr=sum(ranks)/7.0
                stdr=statistics.pstdev(ranks)
                row={
                    "year":year,"race_id":rid,"horse_id":h,"field_size":n,
                    "target_top3":int(h in podium),"target_win":int(h in winner),
                    "final_win_odds":prices[hno[h]],"market_implied_norm":pnorm[h],
                    "market_rank":odds_rank[h],"market_rank_pct":odds_rank[h]/n,
                    "mean_rank_pct":meanr/n,"rank_std_pct":stdr/n,
                    "best_rank_pct":min(ranks)/n,"worst_rank_pct":max(ranks)/n,
                    "top1_support":sum(r==1 for r in ranks)/7.0,
                    "top3_support":sum(r<=3 for r in ranks)/7.0,
                    "top6_support":sum(r<=6 for r in ranks)/7.0,
                }
                for i,e in enumerate(EXPERTS): row[f"{e}_rank_pct"]=ranks[i]/n
                out.append(row)
    fields=[
        "year","race_id","horse_id","field_size","target_top3","target_win",
        "final_win_odds","market_implied_norm","market_rank","market_rank_pct",
        "mean_rank_pct","rank_std_pct","best_rank_pct","worst_rank_pct",
        "top1_support","top3_support","top6_support",
        *[f"{e}_rank_pct" for e in EXPERTS],
    ]
    write_gz_csv(a.output,out,fields)
    Path(a.skipped).parent.mkdir(parents=True,exist_ok=True)
    with open(a.skipped,"w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["race_id","race_date","reason"]); w.writerows(skipped)
    races=len({r["race_id"] for r in out})
    summary={"year":year,"races":races,"horses":len(out),"skipped_races":len(skipped),"2026_locked":True}
    Path(a.summary).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L1_MARKET_RESIDUAL_YEAR_READY",json.dumps(summary,separators=(",",":")))

def cmd_final(a):
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    frames=[]
    for p in a.input:
        frames.append(pd.read_csv(p,compression="gzip"))
    df=pd.concat(frames,ignore_index=True)
    df["race_id"]=df["race_id"].astype(str)
    market_feats=["market_logit","market_rank_pct"]
    l1_feats=[
        "mean_rank_pct","rank_std_pct","best_rank_pct","worst_rank_pct",
        "top1_support","top3_support","top6_support",
        *[f"{e}_rank_pct" for e in EXPERTS],
    ]
    eps=1e-9
    p=df["market_implied_norm"].clip(eps,1-eps)
    df["market_logit"]=np.log(p/(1-p))

    def fit_predict(train,test,features):
        pipe=Pipeline([
            ("scale",StandardScaler()),
            ("model",LogisticRegression(C=1.0,solver="lbfgs",max_iter=1000)),
        ])
        pipe.fit(train[features],train["target_top3"].astype(int))
        pred=pipe.predict_proba(test[features])[:,1]
        coef=pipe.named_steps["model"].coef_[0]
        return pred,dict(zip(features,map(float,coef)))

    def rank_metrics(z,score_col):
        x=z.sort_values(["race_id",score_col,"horse_id"],ascending=[True,False,True]).copy()
        x["pred_rank"]=x.groupby("race_id").cumcount()+1
        top1=x[x["pred_rank"]==1]
        top3=x[x["pred_rank"]<=3]
        top6=x[x["pred_rank"]<=6]
        per3=top3.groupby("race_id").agg(win=("target_win","max"),pod=("target_top3","sum"))
        per6=top6.groupby("race_id").agg(win=("target_win","max"),pod=("target_top3","sum"))
        return {
            "races":int(x["race_id"].nunique()),
            "top1_top3_pct":100*float(top1["target_top3"].mean()),
            "winner_top3_capture_pct":100*float(per3["win"].mean()),
            "winner_top6_capture_pct":100*float(per6["win"].mean()),
            "top3_podium_precision_pct":100*float(top3["target_top3"].mean()),
            "top6_podium_precision_pct":100*float(top6["target_top3"].mean()),
        }

    folds=[]; coefs=[]; preds=[]
    for test_year in (2022,2023,2024,2025):
        train=df[df["year"]<test_year].copy()
        test=df[df["year"]==test_year].copy()
        if train.empty or test.empty: raise SystemExit(f"fold missing {test_year}")
        fold=test[["year","race_id","horse_id","target_top3","target_win","final_win_odds","market_implied_norm"]].copy()
        fold["market_direct_score"]=-test["final_win_odds"].to_numpy()
        specs={
            "MARKET_CAL":market_feats,
            "L1_ONLY":l1_feats,
            "MARKET_PLUS_L1":market_feats+l1_feats,
        }
        for name,features in specs.items():
            pr,cf=fit_predict(train,test,features)
            fold[name]=pr
            y=test["target_top3"].astype(int).to_numpy()
            m={
                "test_year":test_year,"model":name,
                "train_years":"|".join(map(str,sorted(train["year"].unique()))),
                "horses":len(test),"races":test["race_id"].nunique(),
                "log_loss":float(log_loss(y,pr,labels=[0,1])),
                "brier":float(brier_score_loss(y,pr)),
                "auc":float(roc_auc_score(y,pr)),
                **rank_metrics(pd.concat([test[["race_id","horse_id","target_top3","target_win"]].reset_index(drop=True),pd.Series(pr,name="score")],axis=1),"score")
            }
            folds.append(m)
            for k,v in cf.items():coefs.append({"test_year":test_year,"model":name,"feature":k,"standardized_coef":v})
        md=rank_metrics(pd.concat([
            test[["race_id","horse_id","target_top3","target_win"]].reset_index(drop=True),
            pd.Series(-test["final_win_odds"].to_numpy(),name="score")
        ],axis=1),"score")
        folds.append({"test_year":test_year,"model":"MARKET_DIRECT","train_years":"","horses":len(test),"races":test["race_id"].nunique(),"log_loss":"","brier":"","auc":"","races_rank":md["races"],**{k:v for k,v in md.items() if k!="races"}})

        # Paired per-race log-loss delta: combined - market, negative is better.
        y=test["target_top3"].astype(int).to_numpy()
        pm=fold["MARKET_CAL"].to_numpy(); pc=fold["MARKET_PLUS_L1"].to_numpy()
        llm=-(y*np.log(np.clip(pm,eps,1-eps))+(1-y)*np.log(np.clip(1-pm,eps,1-eps)))
        llc=-(y*np.log(np.clip(pc,eps,1-eps))+(1-y)*np.log(np.clip(1-pc,eps,1-eps)))
        tmp=pd.DataFrame({"race_id":test["race_id"].to_numpy(),"d":llc-llm})
        race_d=tmp.groupby("race_id")["d"].mean()
        mean=float(race_d.mean()); se=float(race_d.std(ddof=1)/math.sqrt(len(race_d)))
        folds.append({
            "test_year":test_year,"model":"INCREMENT_COMBINED_MINUS_MARKET",
            "train_years":"|".join(map(str,sorted(train["year"].unique()))),
            "horses":len(test),"races":test["race_id"].nunique(),
            "log_loss":mean,"brier":"","auc":"",
            "delta_logloss_ci95_lo":mean-1.96*se,"delta_logloss_ci95_hi":mean+1.96*se,
            "race_fraction_combined_better":float((race_d<0).mean()),
        })
        preds.append(fold)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(folds).to_csv(out/"fold-metrics.csv",index=False)
    pd.DataFrame(coefs).to_csv(out/"coefficients.csv",index=False)
    oos=pd.concat(preds,ignore_index=True)
    pooled=[]
    y=oos["target_top3"].astype(int).to_numpy()
    for name in ("MARKET_CAL","L1_ONLY","MARKET_PLUS_L1"):
        pr=oos[name].to_numpy()
        base={
            "model":name,"horses":len(oos),"races":oos["race_id"].nunique(),
            "log_loss":float(log_loss(y,pr,labels=[0,1])),
            "brier":float(brier_score_loss(y,pr)),
            "auc":float(roc_auc_score(y,pr)),
        }
        base.update(rank_metrics(pd.concat([oos[["race_id","horse_id","target_top3","target_win"]].reset_index(drop=True),pd.Series(pr,name="score")],axis=1),"score"))
        pooled.append(base)
    md=rank_metrics(pd.concat([oos[["race_id","horse_id","target_top3","target_win"]].reset_index(drop=True),pd.Series(-oos["final_win_odds"].to_numpy(),name="score")],axis=1),"score")
    pooled.append({"model":"MARKET_DIRECT","horses":len(oos),"races":oos["race_id"].nunique(),"log_loss":"","brier":"","auc":"","races_rank":md["races"],**{k:v for k,v in md.items() if k!="races"}})
    pm=next(x for x in pooled if x["model"]=="MARKET_CAL")
    pc=next(x for x in pooled if x["model"]=="MARKET_PLUS_L1")
    pooled.append({
        "model":"INCREMENT_COMBINED_MINUS_MARKET","horses":len(oos),"races":oos["race_id"].nunique(),
        "log_loss":pc["log_loss"]-pm["log_loss"],"brier":pc["brier"]-pm["brier"],"auc":pc["auc"]-pm["auc"],
        "top1_top3_pct":pc["top1_top3_pct"]-pm["top1_top3_pct"],
        "winner_top3_capture_pct":pc["winner_top3_capture_pct"]-pm["winner_top3_capture_pct"],
        "winner_top6_capture_pct":pc["winner_top6_capture_pct"]-pm["winner_top6_capture_pct"],
        "top3_podium_precision_pct":pc["top3_podium_precision_pct"]-pm["top3_podium_precision_pct"],
        "top6_podium_precision_pct":pc["top6_podium_precision_pct"]-pm["top6_podium_precision_pct"],
    })
    pd.DataFrame(pooled).to_csv(out/"pooled-oos.csv",index=False)
    summary={
        "contract":"L1_FINAL_MARKET_RESIDUAL_V1",
        "question":"Does Seven-King rank information add strict-OOS predictive information beyond calibrated final WIN market?",
        "market_note":"FINAL_POSTHOC benchmark only; not same-time actionable market.",
        "models":{
            "MARKET_CAL":"fixed logistic calibration on normalized final WIN implied probability + market rank percentile",
            "L1_ONLY":"fixed logistic on all seven rank percentiles + consensus/disagreement features",
            "MARKET_PLUS_L1":"union of MARKET_CAL and L1_ONLY features",
        },
        "walk_forward":"train prior years, test next year; pooled OOS 2022-2025",
        "2026_locked":True,"promotion":False
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== POOLED OOS ====="); print((out/"pooled-oos.csv").read_text())
    print("===== FOLDS ====="); print((out/"fold-metrics.csv").read_text())
    print("L1_FINAL_MARKET_RESIDUAL_V1_READY")

def main():
    p=argparse.ArgumentParser(); sp=p.add_subparsers(dest="cmd",required=True)
    y=sp.add_parser("year")
    y.add_argument("--year",type=int,required=True); y.add_argument("--state",required=True); y.add_argument("--backfill-root",required=True)
    y.add_argument("--output",required=True); y.add_argument("--skipped",required=True); y.add_argument("--summary",required=True)
    f=sp.add_parser("final")
    f.add_argument("--input",action="append",required=True); f.add_argument("--out-dir",required=True)
    a=p.parse_args()
    cmd_year(a) if a.cmd=="year" else cmd_final(a)

if __name__=="__main__": main()
