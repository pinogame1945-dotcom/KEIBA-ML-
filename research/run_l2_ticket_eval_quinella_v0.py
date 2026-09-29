#!/usr/bin/env python3
import argparse
import csv
import gzip
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from build_l2_bet_kings_dataset_v1 import (
    decode_odds, payout_map, horse_number_map, canonical_numbers
)
from build_l2_l17_fullfield_dataset_v1 import load_l17

YEARS=(2022,2023,2024,2025)
TEST_YEARS=(2023,2024,2025)
DEV_YEARS=(2023,2024)
HOLDOUT_YEAR=2025
EXPECTED_RACES_PER_YEAR=3456
EDGE_GRID=(0.00,0.05,0.10,0.20,0.30,0.50,0.75,1.00)
MAX_TICKETS_GRID=(1,2,3,5)
MIN_DEV_EXECUTION_COVERAGE_PCT=20.0

IDENTITY={
    "year","race_id","race_date","pair_horse_ids","pair_numbers",
    "hit","return_yen_per100","odds","edge","prediction"
}
FORBIDDEN_TOKENS=("odds","payout","return","profit","roi","hit","finish","result","popularity","target")

def parse_args():
    p=argparse.ArgumentParser(description="L2 ticket-level evaluator V0: all-runner QUINELLA pairs.")
    p.add_argument("--l17-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1)
        out[int(y)]=p
    return out

def finite(v,default=0.0):
    try:
        x=float(v)
        return x if math.isfinite(x) else default
    except (TypeError,ValueError):
        return default

def read_jsonl_gz(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def horse_row_features(prefix,h,row):
    row[prefix+"consensus_rank"]=int(h["consensus_rank"])
    row[prefix+"mean_rank"]=finite(h.get("mean_rank"),99.0)
    row[prefix+"rank_std"]=finite(h.get("rank_std"))
    row[prefix+"best_rank"]=finite(h.get("best_rank"),99.0)
    row[prefix+"worst_rank"]=finite(h.get("worst_rank"),99.0)
    row[prefix+"top1_votes"]=finite(h.get("top1_votes"))
    row[prefix+"top3_support"]=finite(h.get("top3_support"))
    row[prefix+"top6_support"]=finite(h.get("top6_support"))
    row[prefix+"mean_probability"]=finite(h.get("mean_probability"))
    row[prefix+"probability_std"]=finite(h.get("probability_std"))
    row[prefix+"horse_number"]=finite(h.get("horse_number"))

def pair_features(year,rid,date,pack,rec,ha,hb,odd,ret,hit,expert_names):
    if int(ha["consensus_rank"])>int(hb["consensus_rank"]):
        ha,hb=hb,ha
    race=pack.get("race") or {}
    row={
        "year":year,
        "race_id":rid,
        "race_date":date,
        "pair_horse_ids":f'{ha["horse_id"]}|{hb["horse_id"]}',
        "pair_numbers":f'{ha["horse_number"]}-{hb["horse_number"]}',
        "hit":int(hit),
        "return_yen_per100":float(ret),
        "odds":float(odd),
        "race_month":finite(date[5:7]),
        "venue_code":str(race.get("venue_code") or "UNKNOWN"),
        "discipline":str(race.get("discipline") or "UNKNOWN"),
        "surface":str(race.get("surface") or "UNKNOWN"),
        "direction":str(race.get("direction") or "UNKNOWN"),
        "weather":str(race.get("weather") or "UNKNOWN"),
        "track_condition":str(race.get("track_condition") or "UNKNOWN"),
        "course_layout":str(race.get("course_layout") or "UNKNOWN"),
        "race_class_normalized":str(race.get("race_class_normalized") or "UNKNOWN"),
        "grade":str(race.get("grade") or "UNKNOWN"),
        "sex_condition":str(race.get("sex_condition") or "UNKNOWN"),
        "weight_rule":str(race.get("weight_rule") or "UNKNOWN"),
        "distance_m":finite(race.get("distance_m")),
        "field_size":int(rec["field_size"]),
    }
    horse_row_features("a_",ha,row)
    horse_row_features("b_",hb,row)

    ar=int(ha["consensus_rank"]); br=int(hb["consensus_rank"])
    ap=finite(ha.get("mean_probability")); bp=finite(hb.get("mean_probability"))
    row.update({
        "pair_rank_sum":ar+br,
        "pair_rank_gap":br-ar,
        "pair_worse_rank":max(ar,br),
        "pair_prob_sum":ap+bp,
        "pair_prob_product":ap*bp,
        "pair_prob_gap":abs(ap-bp),
        "pair_top3_support_sum":finite(ha.get("top3_support"))+finite(hb.get("top3_support")),
        "pair_top6_support_sum":finite(ha.get("top6_support"))+finite(hb.get("top6_support")),
        "pair_rank_std_max":max(finite(ha.get("rank_std")),finite(hb.get("rank_std"))),
        "pair_probability_std_max":max(finite(ha.get("probability_std")),finite(hb.get("probability_std"))),
        "pair_horse_number_gap":abs(finite(ha.get("horse_number"))-finite(hb.get("horse_number"))),
    })
    va=ha.get("experts") or {}
    vb=hb.get("experts") or {}
    for name in expert_names:
        xa=va.get(name) or {}; xb=vb.get(name) or {}
        ra=finite(xa.get("rank"),99.0); rb=finite(xb.get("rank"),99.0)
        pa=finite(xa.get("probability")); pb=finite(xb.get("probability"))
        row[f"expert_{name}_rank_sum"]=ra+rb
        row[f"expert_{name}_rank_max"]=max(ra,rb)
        row[f"expert_{name}_prob_sum"]=pa+pb
        row[f"expert_{name}_prob_min"]=min(pa,pb)
        row[f"expert_{name}_prob_gap"]=abs(pa-pb)
    return row

def build_year_frame(year,l17_rows,backfill_root):
    if len(l17_rows)!=EXPECTED_RACES_PER_YEAR:
        raise SystemExit(f"L1.7 race coverage drift y={year}: {len(l17_rows)}")
    expert_names=sorted({
        n for r in l17_rows.values() for h in r.get("horses",[])
        for n in (h.get("experts") or {})
    })
    if len(expert_names)!=7:
        raise SystemExit(f"expected seven experts y={year} got={expert_names}")

    root=Path(backfill_root)
    rows=[]
    processed=set()
    missing_odds_days=[]
    priced_races=0
    for day_path in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
        date=day_path.name[:10]
        odds_path=root/"data"/"odds"/"daily"/day_path.name
        if not odds_path.exists():
            missing_odds_days.append(date)
            continue
        odds_by_race={str(x.get("race_id") or ""):x for x in read_jsonl_gz(odds_path)}
        for pack in read_jsonl_gz(day_path):
            race=pack.get("race") or {}
            rid=str(race.get("race_id") or "")
            if rid not in l17_rows:
                continue
            rec=l17_rows[rid]
            odds_rec=odds_by_race.get(rid)
            if odds_rec is None:
                continue
            hno=horse_number_map(pack)
            horses=sorted(rec["horses"],key=lambda x:int(x["consensus_rank"]))
            if len(horses)!=int(rec["field_size"]):
                raise SystemExit(f"L1.7 field drift race={rid}")
            if any(str(h["horse_id"]) not in hno for h in horses):
                raise SystemExit(f"horse missing from pack race={rid}")
            odds_map=decode_odds(odds_rec)
            payouts,present=payout_map(pack)
            if "QUINELLA" not in present:
                continue
            this_rows=0
            for ha,hb in itertools.combinations(horses,2):
                nums=[hno[str(ha["horse_id"])],hno[str(hb["horse_id"])]]
                key=("QUINELLA",canonical_numbers("QUINELLA",nums))
                odd=odds_map.get(key)
                if odd is None:
                    continue
                ret=float(payouts.get(key,0.0))
                row=pair_features(
                    year,rid,date,pack,rec,ha,hb,odd,ret,ret>0,expert_names
                )
                rows.append(row)
                this_rows+=1
            if this_rows:
                priced_races+=1
                processed.add(rid)
    if not rows:
        raise SystemExit(f"no QUINELLA rows y={year}")
    df=pd.DataFrame(rows)
    print("TICKET_EVAL_YEAR_READY "+json.dumps({
        "year":year,
        "l17_races":len(l17_rows),
        "priced_races":priced_races,
        "priced_pairs":len(df),
        "positive_pairs":int(df["hit"].sum()),
        "missing_odds_days":missing_odds_days,
    },ensure_ascii=False,separators=(",",":")),flush=True)
    return df

def feature_columns(df):
    cols=[c for c in df.columns if c not in IDENTITY]
    bad=[c for c in cols if any(tok in c.lower() for tok in FORBIDDEN_TOKENS)]
    if bad:
        raise SystemExit(f"forbidden model features={sorted(bad)}")
    return cols

def encode(train,test,cols):
    xtr=train[cols].copy()
    xte=test[cols].copy()
    for c in cols:
        if xtr[c].dtype=="object" or xte[c].dtype=="object":
            tr=xtr[c].fillna("__NA__").astype(str)
            te=xte[c].fillna("__NA__").astype(str)
            values=sorted(tr.unique())
            mp={v:i for i,v in enumerate(values)}
            xtr[c]=tr.map(mp).fillna(-1).astype("int32")
            xte[c]=te.map(mp).fillna(-1).astype("int32")
        else:
            xtr[c]=pd.to_numeric(xtr[c],errors="coerce").fillna(0.0).astype("float32")
            xte[c]=pd.to_numeric(xte[c],errors="coerce").fillna(0.0).astype("float32")
    return xtr,xte

def train_predict(train,test,cols,seed):
    xtr,xte=encode(train,test,cols)
    y=train["hit"].astype(int).to_numpy()
    model=lgb.LGBMClassifier(
        objective="binary",
        n_estimators=320,
        learning_rate=0.03,
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
    model.fit(xtr,y)
    p=np.asarray(model.predict_proba(xte)[:,1],dtype=float)
    importance=pd.DataFrame({"feature":cols,"gain":model.feature_importances_})
    return p,importance

def max_drawdown_from_tickets(chosen):
    if chosen.empty:
        return 0.0
    race=chosen.groupby(["year","race_date","race_id"],as_index=False).agg(
        stake=("pair_horse_ids",lambda s:100.0*len(s)),
        ret=("return_yen_per100","sum")
    ).sort_values(["year","race_date","race_id"])
    cum=(race["ret"]-race["stake"]).cumsum().to_numpy()
    peak=np.maximum.accumulate(np.r_[0.0,cum])
    dd=peak[1:]-cum
    return float(dd.max()) if len(dd) else 0.0

def evaluate_chosen(chosen,source_races,label):
    if chosen.empty:
        return {
            "label":label,"source_races":source_races,"executed_races":0,
            "execution_coverage_pct":0.0,"tickets":0,"hit_races":0,
            "race_hit_rate_pct":0.0,"stake_yen":0.0,"return_yen":0.0,
            "profit_yen":0.0,"roi_pct":None,"max_drawdown_yen":0.0
        }
    executed=chosen[["year","race_id"]].drop_duplicates().shape[0]
    hit_races=chosen.loc[chosen["hit"]==1,["year","race_id"]].drop_duplicates().shape[0]
    tickets=len(chosen)
    stake=100.0*tickets
    ret=float(chosen["return_yen_per100"].sum())
    return {
        "label":label,
        "source_races":source_races,
        "executed_races":executed,
        "execution_coverage_pct":100.0*executed/source_races if source_races else 0.0,
        "tickets":tickets,
        "hit_races":hit_races,
        "race_hit_rate_pct":100.0*hit_races/source_races if source_races else 0.0,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown_from_tickets(chosen),
    }

def apply_policy(df,min_edge,max_tickets):
    z=df[df["edge"]>=min_edge].copy()
    if z.empty:
        return z
    z=z.sort_values(
        ["year","race_date","race_id","edge","prediction","pair_prob_product"],
        ascending=[True,True,True,False,False,False]
    )
    return z.groupby(["year","race_id"],sort=False).head(max_tickets).copy()

def baseline(df,name):
    if name=="TOP4_BOX":
        return df[(df["a_consensus_rank"]<=4)&(df["b_consensus_rank"]<=4)].copy()
    if name=="TOP1_TO6":
        return df[(df["a_consensus_rank"]==1)&(df["b_consensus_rank"]<=6)].copy()
    if name=="TOP6_BOX":
        return df[(df["a_consensus_rank"]<=6)&(df["b_consensus_rank"]<=6)].copy()
    raise ValueError(name)

def ranking_metrics(df,year):
    rows=[]
    n=df[["year","race_id"]].drop_duplicates().shape[0]
    for score,label in [("prediction","MODEL"),("pair_prob_product","L17_PRODUCT")]:
        z=df.sort_values(["race_id",score],ascending=[True,False]).copy()
        z["rank_in_race"]=z.groupby("race_id").cumcount()+1
        for k in (1,3,5,10):
            hit_races=z[(z["rank_in_race"]<=k)&(z["hit"]==1)]["race_id"].nunique()
            rows.append({"year":year,"ranking":label,"top_k":k,"hit_races":hit_races,
                         "source_races":n,"coverage_pct":100.0*hit_races/n if n else 0.0})
    return rows

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(rows,pd.DataFrame):
        rows.to_csv(path,index=False)
        return
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def main():
    a=parse_args()
    lp=parse_paths(a.l17_year)
    if set(lp)!=set(YEARS):
        raise SystemExit("L1.7 year path mismatch")
    if 2026 in lp:
        raise SystemExit("2026 sealed")

    l17={y:load_l17(lp[y],y) for y in YEARS}
    frames={y:build_year_frame(y,l17[y],a.backfill_root) for y in YEARS}
    cols=feature_columns(frames[2022])

    fold_rows=[]
    ranking_rows=[]
    importances=[]
    predictions={}
    for y in TEST_YEARS:
        train=pd.concat([frames[t] for t in YEARS if t<y],ignore_index=True)
        test=frames[y].copy()
        p,imp=train_predict(train,test,cols,91000+y)
        test["prediction"]=p
        test["edge"]=test["prediction"]*test["odds"]-1.0
        predictions[y]=test

        yy=test["hit"].astype(int).to_numpy()
        eps=np.clip(p,1e-9,1-1e-9)
        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(str(t) for t in YEARS if t<y),
            "train_pairs":len(train),
            "test_pairs":len(test),
            "source_races":test["race_id"].nunique(),
            "positive_rate_pct":100.0*float(yy.mean()),
            "roc_auc":float(roc_auc_score(yy,p)),
            "log_loss":float(log_loss(yy,eps)),
            "brier":float(brier_score_loss(yy,p)),
            "feature_count":len(cols),
        })
        ranking_rows.extend(ranking_metrics(test,y))
        imp["test_year"]=y
        importances.append(imp)
        del train

    dev=pd.concat([predictions[y] for y in DEV_YEARS],ignore_index=True)
    dev_source=dev[["year","race_id"]].drop_duplicates().shape[0]
    grid=[]
    for edge in EDGE_GRID:
        for max_t in MAX_TICKETS_GRID:
            chosen=apply_policy(dev,edge,max_t)
            m=evaluate_chosen(chosen,dev_source,f"EDGE_{edge:.2f}_MAX{max_t}")
            m["min_edge"]=edge
            m["max_tickets_per_race"]=max_t
            grid.append(m)

    eligible=[r for r in grid if r["execution_coverage_pct"]>=MIN_DEV_EXECUTION_COVERAGE_PCT]
    pool=eligible or grid
    champion=max(pool,key=lambda r:(
        r["profit_yen"],
        r["roi_pct"] if r["roi_pct"] is not None else -1e18,
        -r["max_drawdown_yen"],
        -r["tickets"],
    ))
    selected_edge=float(champion["min_edge"])
    selected_max=int(champion["max_tickets_per_race"])

    hold=predictions[HOLDOUT_YEAR]
    hold_source=hold["race_id"].nunique()
    hold_chosen=apply_policy(hold,selected_edge,selected_max)
    hold_metrics=evaluate_chosen(
        hold_chosen,hold_source,
        f"SELECTED_EDGE_{selected_edge:.2f}_MAX{selected_max}"
    )
    hold_metrics["selected_from_dev_years"]=list(DEV_YEARS)
    hold_metrics["min_edge"]=selected_edge
    hold_metrics["max_tickets_per_race"]=selected_max

    baseline_rows=[]
    for y in TEST_YEARS:
        df=predictions[y]
        source=df["race_id"].nunique()
        for name in ("TOP4_BOX","TOP1_TO6","TOP6_BOX"):
            m=evaluate_chosen(baseline(df,name),source,name)
            m["year"]=y
            baseline_rows.append(m)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"model-fold-metrics.csv",fold_rows)
    write_csv(out/"ranking-coverage.csv",ranking_rows)
    write_csv(out/"dev-policy-grid.csv",grid)
    write_csv(out/"baseline-metrics.csv",baseline_rows)
    write_csv(out/"feature-importance.csv",pd.concat(importances,ignore_index=True))
    with gzip.open(out/"holdout-selected-tickets.csv.gz","wt",newline="",encoding="utf-8") as f:
        keep=["year","race_id","race_date","pair_horse_ids","pair_numbers","prediction","odds","edge","hit","return_yen_per100",
              "a_consensus_rank","b_consensus_rank","pair_prob_sum","pair_prob_product"]
        hold_chosen[keep].to_csv(f,index=False)

    summary={
        "contract":"L2_TICKET_EVAL_QUINELLA_V0_RESULT",
        "architecture":"ticket_level_binary_hit_probability_then_market_edge",
        "bet_type":"QUINELLA",
        "all_runners_considered":True,
        "candidate_horse_selection":False,
        "template_router":False,
        "model_input_odds":False,
        "odds_stage":"post_prediction_edge_only",
        "stake_per_ticket_yen":100,
        "development_years":list(DEV_YEARS),
        "holdout_year":HOLDOUT_YEAR,
        "holdout_used_for_policy_selection":False,
        "policy_selection_objective":"maximize development profit subject to minimum execution coverage",
        "minimum_dev_execution_coverage_pct":MIN_DEV_EXECUTION_COVERAGE_PCT,
        "selected_policy":champion,
        "holdout":hold_metrics,
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Ticket Evaluator — QUINELLA V0\n\n"
        "Rebuild from scratch at ticket level. Every priced quinella pair in the L1.7 full field is scored independently. "
        "The model predicts hit probability from pre-race race/L1.7 features only. Final market odds are not model inputs; "
        "they are applied after prediction as edge = p * odds - 1. Development policy is selected on 2023-2024 only, "
        "then frozen for the 2025 holdout. No candidate-horse truncation, no template classifier, no Outsider, and 2026 remains sealed.\n",
        encoding="utf-8"
    )
    print("L2_TICKET_EVAL_QUINELLA_V0_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
