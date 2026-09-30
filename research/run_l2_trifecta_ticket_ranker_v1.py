#!/usr/bin/env python3
import argparse
import csv
import gzip
import itertools
import json
import math
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
TOP6=6
EXPECTED_TICKETS=120
TOPN_GRID=(1,2,3,5,10,20,30,60,120)

IDENTITY={
    "year","race_id","race_date","ticket_horse_ids","ticket_numbers",
    "hit","return_yen_per100","odds","market_q_norm","market_log_q",
    "market_rank","model_score","model_rank","l17_order_score","l17_rank"
}
FORBIDDEN_TOKENS=("payout","return","profit","roi","hit","finish","result","target")


def parse_args():
    p=argparse.ArgumentParser(description="All-candidate Seven-King Top6 TRIFECTA ticket ranker.")
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
        raise SystemExit(f"L1.7 year path mismatch got={sorted(out)} expected={list(YEARS)}")
    if 2026 in out:
        raise SystemExit("2026 sealed")
    return out


def read_jsonl_gz(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
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
            raise ValueError(f"bad/duplicate L1.7 race_id y={year} rid={rid}")
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


def horse_features(prefix,h,row,expert_names):
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
    views=h.get("experts") or {}
    for name in expert_names:
        x=views.get(name) or {}
        row[f"{prefix}expert_{name}_rank"]=finite(x.get("rank"),99.0)
        row[f"{prefix}expert_{name}_probability"]=finite(x.get("probability"))


def ordered_l17_score(a,b,c):
    pa=max(1e-9,min(0.999999,finite(a.get("mean_probability"))))
    pb=max(1e-9,min(0.999999,finite(b.get("mean_probability"))))
    pc=max(1e-9,min(0.999999,finite(c.get("mean_probability"))))
    d2=max(1e-9,1.0-pa)
    d3=max(1e-9,1.0-pa-pb)
    return pa*(pb/d2)*(pc/d3)


def ticket_row(year,rid,date,pack,rec,a,b,c,odd,ret,expert_names):
    race=pack.get("race") or {}
    row={
        "year":year,
        "race_id":rid,
        "race_date":date,
        "ticket_horse_ids":"|".join([str(a["horse_id"]),str(b["horse_id"]),str(c["horse_id"])]),
        "ticket_numbers":f'{a["horse_number"]}-{b["horse_number"]}-{c["horse_number"]}',
        "hit":int(ret>0),
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
    horse_features("s1_",a,row,expert_names)
    horse_features("s2_",b,row,expert_names)
    horse_features("s3_",c,row,expert_names)

    ranks=[int(a["consensus_rank"]),int(b["consensus_rank"]),int(c["consensus_rank"])]
    probs=[finite(a.get("mean_probability")),finite(b.get("mean_probability")),finite(c.get("mean_probability"))]
    rank_stds=[finite(a.get("rank_std")),finite(b.get("rank_std")),finite(c.get("rank_std"))]
    top6s=[finite(a.get("top6_support")),finite(b.get("top6_support")),finite(c.get("top6_support"))]
    row.update({
        "rank_sum":sum(ranks),
        "rank_max":max(ranks),
        "rank_min":min(ranks),
        "rank_span":max(ranks)-min(ranks),
        "rank_order_12":ranks[0]-ranks[1],
        "rank_order_23":ranks[1]-ranks[2],
        "prob_sum":sum(probs),
        "prob_product":probs[0]*probs[1]*probs[2],
        "prob_min":min(probs),
        "prob_max":max(probs),
        "prob_order_12":probs[0]-probs[1],
        "prob_order_23":probs[1]-probs[2],
        "rank_std_max":max(rank_stds),
        "rank_std_mean":sum(rank_stds)/3.0,
        "top6_support_sum":sum(top6s),
        "l17_order_score":ordered_l17_score(a,b,c),
    })
    return row


def build_year_frame(year,l17_rows,backfill_root):
    expert_names=sorted({
        name
        for rec in l17_rows.values()
        for h in rec.get("horses",[])
        for name in (h.get("experts") or {}).keys()
    })
    if len(expert_names)!=7:
        raise SystemExit(f"expected seven experts y={year} got={expert_names}")

    root=Path(backfill_root)
    rows=[]
    counters={
        "daily_files":0,
        "odds_files":0,
        "l17_races_seen":0,
        "full120_races":0,
        "missing_odds_day":0,
        "missing_odds_row":0,
        "missing_trifecta_payout":0,
        "incomplete_top6":0,
        "incomplete_120_odds":0,
    }

    for day_path in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
        counters["daily_files"]+=1
        date=day_path.name[:10]
        odds_path=root/"data"/"odds"/"daily"/day_path.name
        if not odds_path.exists():
            counters["missing_odds_day"]+=1
            continue
        counters["odds_files"]+=1
        odds_by_race={str(x.get("race_id") or ""):x for x in read_jsonl_gz(odds_path)}

        for pack in read_jsonl_gz(day_path):
            race=pack.get("race") or {}
            rid=str(race.get("race_id") or "")
            rec=l17_rows.get(rid)
            if rec is None:
                continue
            counters["l17_races_seen"]+=1
            odds_rec=odds_by_race.get(rid)
            if odds_rec is None:
                counters["missing_odds_row"]+=1
                continue

            hno=horse_number_map(pack)
            horses=sorted(rec["horses"],key=lambda x:int(x["consensus_rank"]))
            if len(horses)!=int(rec["field_size"]):
                raise SystemExit(f"L1.7 field drift race={rid}")
            top6=horses[:TOP6]
            if len(top6)<TOP6 or any(str(h["horse_id"]) not in hno for h in top6):
                counters["incomplete_top6"]+=1
                continue
            for h in top6:
                h["horse_number"]=int(hno[str(h["horse_id"])])

            odds_map=decode_odds(odds_rec)
            payouts,present=payout_map(pack)
            if "TRIFECTA" not in present:
                counters["missing_trifecta_payout"]+=1
                continue

            temp=[]
            for a,b,c in itertools.permutations(top6,3):
                nums=(int(a["horse_number"]),int(b["horse_number"]),int(c["horse_number"]))
                key=("TRIFECTA",nums)
                odd=odds_map.get(key)
                if odd is None or odd<=0:
                    continue
                ret=float(payouts.get(key,0.0))
                temp.append(ticket_row(
                    year,rid,date,pack,rec,a,b,c,odd,ret,expert_names
                ))
            if len(temp)!=EXPECTED_TICKETS:
                counters["incomplete_120_odds"]+=1
                continue

            inv=np.asarray([1.0/max(1e-12,float(x["odds"])) for x in temp],dtype=float)
            total=float(inv.sum())
            q=inv/max(total,1e-12)
            market_order=np.argsort(-q,kind="mergesort")
            market_rank=np.empty(len(temp),dtype=int)
            market_rank[market_order]=np.arange(1,len(temp)+1)

            l17_score=np.asarray([float(x["l17_order_score"]) for x in temp],dtype=float)
            l17_order=np.argsort(-l17_score,kind="mergesort")
            l17_rank=np.empty(len(temp),dtype=int)
            l17_rank[l17_order]=np.arange(1,len(temp)+1)

            for i,x in enumerate(temp):
                x["market_q_norm"]=float(q[i])
                x["market_log_q"]=float(math.log(max(q[i],1e-12)))
                x["market_rank"]=int(market_rank[i])
                x["l17_rank"]=int(l17_rank[i])
            rows.extend(temp)
            counters["full120_races"]+=1

    if not rows:
        raise SystemExit(f"no TRIFECTA rows y={year}")
    df=pd.DataFrame(rows)
    full_races=int(df["race_id"].nunique())
    if len(df)!=full_races*EXPECTED_TICKETS:
        raise SystemExit(f"120-ticket invariant broken y={year} rows={len(df)} races={full_races}")

    print("TRIFECTA_TICKET_YEAR_READY "+json.dumps({
        "year":year,
        "rows":len(df),
        "races":full_races,
        "positive_tickets":int(df["hit"].sum()),
        "coverable_races":int(df.loc[df["hit"]==1,"race_id"].nunique()),
        "top6_capture_pct":100.0*df.loc[df["hit"]==1,"race_id"].nunique()/full_races,
        "counters":counters,
    },ensure_ascii=False,separators=(",",":")),flush=True)
    return df,counters


def feature_columns(df):
    # Market-aware model: normalized market probability/rank are valid L2 features.
    cols=[c for c in df.columns if c not in IDENTITY]
    cols += ["market_q_norm","market_log_q","market_rank"]
    cols=list(dict.fromkeys(cols))
    bad=[c for c in cols if any(tok in c.lower() for tok in FORBIDDEN_TOKENS)]
    if bad:
        raise SystemExit(f"forbidden model features={sorted(bad)}")
    # Raw odds are intentionally excluded; only normalized within-race market view is used.
    if "odds" in cols:
        raise SystemExit("raw odds leaked into features")
    return cols


def encode(train,test,cols):
    xtr=train[cols].copy()
    xte=test[cols].copy()
    for c in cols:
        if xtr[c].dtype=="object" or xte[c].dtype=="object":
            tr=xtr[c].fillna("__NA__").astype(str)
            te=xte[c].fillna("__NA__").astype(str)
            vals=sorted(tr.unique())
            mp={v:i for i,v in enumerate(vals)}
            xtr[c]=tr.map(mp).fillna(-1).astype("int32")
            xte[c]=te.map(mp).fillna(-1).astype("int32")
        else:
            xtr[c]=pd.to_numeric(xtr[c],errors="coerce").fillna(0.0).astype("float32")
            xte[c]=pd.to_numeric(xte[c],errors="coerce").fillna(0.0).astype("float32")
    return xtr,xte


def train_rank_predict(train,test,cols,seed):
    # LambdaRank requires a relevant item. Races whose winning trifecta is
    # outside Seven-King Top6 carry no ordering label for this 120-ticket universe.
    good=train.groupby(["year","race_id"])["hit"].transform("sum")>0
    tr=train.loc[good].copy()
    tr=tr.sort_values(["year","race_id","ticket_numbers"]).reset_index(drop=True)
    te=test.copy().sort_values(["year","race_id","ticket_numbers"])
    te["_orig_index"]=te.index
    te=te.reset_index(drop=True)

    xtr,xte=encode(tr,te,cols)
    y=tr["hit"].astype(int).to_numpy()
    groups=tr.groupby(["year","race_id"],sort=False).size().tolist()
    if any(g!=EXPECTED_TICKETS for g in groups):
        raise SystemExit("training group size drift")

    model=lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=360,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=120,
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
    model.fit(xtr,y,group=groups)
    pred=np.asarray(model.predict(xte),dtype=float)
    s=pd.Series(pred,index=te["_orig_index"].astype(int).to_numpy())
    out=s.reindex(test.index).to_numpy()
    imp=pd.DataFrame({"feature":cols,"gain":model.feature_importances_})
    return out,imp,int(tr["race_id"].nunique()),len(tr)


def add_model_rank(df):
    z=df.copy()
    ordered=z.sort_values(
        ["year","race_id","model_score","market_rank","l17_rank","ticket_numbers"],
        ascending=[True,True,False,True,True,True]
    )
    r=ordered.groupby(["year","race_id"],sort=False).cumcount()+1
    z.loc[ordered.index,"model_rank"]=r.to_numpy()
    z["model_rank"]=z["model_rank"].astype(int)
    return z


def max_drawdown(chosen):
    if chosen.empty:
        return 0.0
    g=chosen.groupby(["year","race_date","race_id"],as_index=False).agg(
        tickets=("ticket_numbers","size"),
        ret=("return_yen_per100","sum"),
    ).sort_values(["year","race_date","race_id"])
    stake=g["tickets"].astype(float)*100.0
    pnl=g["ret"].astype(float)-stake
    cum=pnl.cumsum().to_numpy()
    peaks=np.maximum.accumulate(np.r_[0.0,cum])
    dd=peaks[1:]-cum
    return float(dd.max()) if len(dd) else 0.0


def evaluate(chosen,source_races,label):
    if chosen.empty:
        return {
            "label":label,"source_races":source_races,"executed_races":0,
            "tickets":0,"avg_tickets_per_race":0.0,"hit_races":0,
            "race_hit_rate_pct":0.0,"stake_yen":0.0,"return_yen":0.0,
            "profit_yen":0.0,"roi_pct":None,"avg_return_per_hit_yen":None,
            "max_drawdown_yen":0.0,
        }
    executed=int(chosen["race_id"].nunique())
    hits=int(chosen.loc[chosen["hit"]==1,"race_id"].nunique())
    tickets=len(chosen)
    stake=100.0*tickets
    ret=float(chosen["return_yen_per100"].sum())
    return {
        "label":label,
        "source_races":source_races,
        "executed_races":executed,
        "tickets":tickets,
        "avg_tickets_per_race":tickets/executed if executed else 0.0,
        "hit_races":hits,
        "race_hit_rate_pct":100.0*hits/source_races if source_races else 0.0,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "avg_return_per_hit_yen":ret/hits if hits else None,
        "max_drawdown_yen":max_drawdown(chosen),
    }


def topn(df,rank_col,n):
    return df[df[rank_col]<=n].copy()


def ranking_coverage(df,year):
    source=int(df["race_id"].nunique())
    rows=[]
    for label,col in [("MARKET","market_rank"),("L17_ORDER","l17_rank"),("MODEL","model_rank")]:
        for k in TOPN_GRID:
            hits=int(df[(df[col]<=k)&(df["hit"]==1)]["race_id"].nunique())
            rows.append({
                "year":year,
                "ranking":label,
                "top_n":k,
                "source_races":source,
                "hit_races":hits,
                "hit_rate_pct":100.0*hits/source if source else 0.0,
            })
    return rows


def policy_rows(df,year):
    source=int(df["race_id"].nunique())
    rows=[]
    for label,col in [("MARKET","market_rank"),("L17_ORDER","l17_rank"),("MODEL","model_rank")]:
        for n in TOPN_GRID:
            m=evaluate(topn(df,col,n),source,f"{label}_TOP{n}")
            m.update({"year":year,"ranking":label,"top_n":n})
            rows.append(m)
    return rows


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(rows,pd.DataFrame):
        rows.to_csv(path,index=False)
        return
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    a=parse_args()
    paths=parse_paths(a.l17_year)
    l17={y:load_l17(paths[y],y) for y in YEARS}

    frames={}
    coverage_meta={}
    for y in YEARS:
        frames[y],coverage_meta[y]=build_year_frame(y,l17[y],a.backfill_root)

    cols=feature_columns(frames[2022])
    predictions={}
    fold_rows=[]
    coverage_rows=[]
    economics_rows=[]
    importances=[]

    for y in TEST_YEARS:
        train=pd.concat([frames[t] for t in YEARS if t<y],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)
        score,imp,train_coverable_races,train_rows=train_rank_predict(train,test,cols,97000+y)
        test["model_score"]=score
        test=add_model_rank(test)
        predictions[y]=test

        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(str(t) for t in YEARS if t<y),
            "train_all_rows":len(train),
            "train_coverable_rows":train_rows,
            "train_coverable_races":train_coverable_races,
            "test_rows":len(test),
            "test_races":int(test["race_id"].nunique()),
            "test_coverable_races":int(test.loc[test["hit"]==1,"race_id"].nunique()),
            "feature_count":len(cols),
        })
        coverage_rows.extend(ranking_coverage(test,y))
        economics_rows.extend(policy_rows(test,y))
        imp["test_year"]=y
        importances.append(imp)
        del train

    # Select a single global ticket count on 2023-2024 only. No race filters,
    # no hand-authored rank/seat rules: every race buys exactly N model-ranked tickets.
    dev=pd.concat([predictions[y] for y in DEV_YEARS],ignore_index=True)
    dev_source=int(dev[["year","race_id"]].drop_duplicates().shape[0])
    dev_grid=[]
    for n in TOPN_GRID:
        m=evaluate(topn(dev,"model_rank",n),dev_source,f"MODEL_TOP{n}")
        m["top_n"]=n
        dev_grid.append(m)
    champion=max(dev_grid,key=lambda r:(
        r["roi_pct"] if r["roi_pct"] is not None else -1e18,
        r["profit_yen"],
        -r["max_drawdown_yen"],
        -r["top_n"],
    ))

    hold=predictions[HOLDOUT_YEAR]
    hold_source=int(hold["race_id"].nunique())
    hold_chosen=topn(hold,"model_rank",int(champion["top_n"]))
    hold_metrics=evaluate(
        hold_chosen,hold_source,f"FROZEN_MODEL_TOP{int(champion['top_n'])}"
    )
    hold_metrics["selected_from_dev_years"]=list(DEV_YEARS)
    hold_metrics["top_n"]=int(champion["top_n"])

    combined=pd.concat([predictions[y] for y in TEST_YEARS],ignore_index=True)
    combined_source=int(combined[["year","race_id"]].drop_duplicates().shape[0])
    combined_rows=[]
    for label,col in [("MARKET","market_rank"),("L17_ORDER","l17_rank"),("MODEL","model_rank")]:
        for n in TOPN_GRID:
            m=evaluate(topn(combined,col,n),combined_source,f"{label}_TOP{n}")
            m.update({"ranking":label,"top_n":n,"test_years":"2023-2025"})
            combined_rows.append(m)

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"model-folds.csv",fold_rows)
    write_csv(out/"ranking-coverage.csv",coverage_rows)
    write_csv(out/"economics-yearly.csv",economics_rows)
    write_csv(out/"economics-combined.csv",combined_rows)
    write_csv(out/"dev-model-topn-grid.csv",dev_grid)
    write_csv(out/"feature-importance.csv",pd.concat(importances,ignore_index=True))

    # Compact holdout selections only; candidate matrix remains ephemeral.
    keep=[
        "year","race_id","race_date","ticket_horse_ids","ticket_numbers",
        "model_score","model_rank","market_q_norm","market_rank",
        "l17_order_score","l17_rank","hit","return_yen_per100","odds",
        "s1_consensus_rank","s2_consensus_rank","s3_consensus_rank"
    ]
    with gzip.open(out/"holdout-selected-tickets.csv.gz","wt",encoding="utf-8",newline="") as f:
        hold_chosen[keep].to_csv(f,index=False)

    summary={
        "contract":"L2_TRIFECTA_TICKET_RANKER_V1_RESULT",
        "architecture":"single market-aware LambdaRank over every Seven-King Top6 trifecta candidate",
        "candidate_universe":"all ordered permutations of Seven-King Top6 = 120 tickets per eligible race",
        "manual_seat_rules":False,
        "manual_market_gap_rules":False,
        "race_skip_rules":False,
        "bet_type":"TRIFECTA",
        "model":"LightGBM LambdaRank standard CPU",
        "model_features":"L1.7 seat-specific Seven-King features + race context + normalized final trifecta market probability/rank",
        "raw_odds_as_model_feature":False,
        "market_representation":"normalized inverse odds within the 120-ticket race universe",
        "training_rule":"strict walk-forward; each test year uses only prior years; un-coverable training races omitted from ranking loss",
        "development_years":list(DEV_YEARS),
        "holdout_year":HOLDOUT_YEAR,
        "holdout_used_for_policy_selection":False,
        "policy":"buy the same global top-N model-ranked tickets in every eligible race",
        "topn_grid":list(TOPN_GRID),
        "selected_dev_policy":champion,
        "holdout":hold_metrics,
        "year_coverage":coverage_meta,
        "2026_locked":True,
        "production_promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Trifecta Ticket Ranker V1\n\n"
        "The unit of learning is one trifecta ticket, not a hand-authored race rule. "
        "For every eligible race, all 120 ordered permutations of the Seven-King Top6 are scored by one model. "
        "There are no fixed seat rules, no market-gap buckets used as buy/skip rules, and no race skipping. "
        "2023-2024 select one global Top-N ticket count; 2025 is frozen holdout; 2026 remains sealed. "
        "Final trifecta odds are represented only as normalized within-race market probability/rank, not raw odds. "
        "Payout/result fields are labels/evaluation only.\n",
        encoding="utf-8",
    )
    print("L2_TRIFECTA_TICKET_RANKER_V1_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)


if __name__=="__main__":
    main()
