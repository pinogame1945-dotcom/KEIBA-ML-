#!/usr/bin/env python3
import argparse
import gzip
import itertools
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import decode_odds, payout_map, horse_number_map
from build_l2_l17_fullfield_dataset_v1 import load_l17

YEARS=(2022,2023,2024,2025)
TEST_YEARS=(2023,2024,2025)
EXPECTED_RACES_PER_YEAR=3456
TOP_K=(1,3,5,10,15)

IDENTITY={
    "year","race_id","race_date","ticket_horse_ids","ticket_numbers",
    "unordered_horse_ids","unordered_numbers","a_horse_id","b_horse_id",
    "a_horse_number","b_horse_number","hit","return_yen_per100","odds",
    "market_inv_odds","market_q","market_log_q","market_rank",
    "l17_only_score","l17_rank","market_aware_score","model_rank","rank_upgrade",
}
FORBIDDEN_TOKENS=("odds","payout","return","profit","roi","hit","finish","result","popularity","target")

def parse_args():
    p=argparse.ArgumentParser(description="L2 EXACTA V0: all ordered pairs, market rank, L1.7-only rank, market-aware rank, and direction audit.")
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

def race_features(pack,rec,date):
    race=pack.get("race") or {}
    return {
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

def pair_premarket_features(ha,hb,expert_names):
    ar=finite(ha.get("consensus_rank"),99.0); br=finite(hb.get("consensus_rank"),99.0)
    am=finite(ha.get("mean_rank"),99.0); bm=finite(hb.get("mean_rank"),99.0)
    ars=finite(ha.get("rank_std")); brs=finite(hb.get("rank_std"))
    abest=finite(ha.get("best_rank"),99.0); bbest=finite(hb.get("best_rank"),99.0)
    aworst=finite(ha.get("worst_rank"),99.0); bworst=finite(hb.get("worst_rank"),99.0)
    at1=finite(ha.get("top1_votes")); bt1=finite(hb.get("top1_votes"))
    at3=finite(ha.get("top3_support")); bt3=finite(hb.get("top3_support"))
    at6=finite(ha.get("top6_support")); bt6=finite(hb.get("top6_support"))
    ap=finite(ha.get("mean_probability")); bp=finite(hb.get("mean_probability"))
    aps=finite(ha.get("probability_std")); bps=finite(hb.get("probability_std"))

    row={
        # Symmetric pair information: identical for A->B and B->A.
        "pair_consensus_rank_sum":ar+br,
        "pair_consensus_rank_abs_gap":abs(ar-br),
        "pair_mean_rank_sum":am+bm,
        "pair_mean_rank_abs_gap":abs(am-bm),
        "pair_rank_std_sum":ars+brs,
        "pair_rank_std_abs_gap":abs(ars-brs),
        "pair_best_rank_sum":abest+bbest,
        "pair_worst_rank_sum":aworst+bworst,
        "pair_top1_votes_sum":at1+bt1,
        "pair_top3_support_sum":at3+bt3,
        "pair_top6_support_sum":at6+bt6,
        "pair_mean_probability_sum":ap+bp,
        "pair_mean_probability_product":ap*bp,
        "pair_mean_probability_abs_gap":abs(ap-bp),
        "pair_probability_std_sum":aps+bps,
        "pair_probability_std_abs_gap":abs(aps-bps),

        # Directional information: every field below must flip sign on reversal.
        "dir_consensus_rank_diff":ar-br,
        "dir_mean_rank_diff":am-bm,
        "dir_rank_std_diff":ars-brs,
        "dir_best_rank_diff":abest-bbest,
        "dir_worst_rank_diff":aworst-bworst,
        "dir_top1_votes_diff":at1-bt1,
        "dir_top3_support_diff":at3-bt3,
        "dir_top6_support_diff":at6-bt6,
        "dir_mean_probability_diff":ap-bp,
        "dir_probability_std_diff":aps-bps,
    }

    va=ha.get("experts") or {}
    vb=hb.get("experts") or {}
    vote_margin=0
    for name in expert_names:
        xa=va.get(name) or {}; xb=vb.get(name) or {}
        ra=finite(xa.get("rank"),99.0); rb=finite(xb.get("rank"),99.0)
        pa=finite(xa.get("probability")); pb=finite(xb.get("probability"))
        row[f"expert_{name}_rank_sum"]=ra+rb
        row[f"expert_{name}_rank_abs_gap"]=abs(ra-rb)
        row[f"expert_{name}_prob_sum"]=pa+pb
        row[f"expert_{name}_prob_product"]=pa*pb
        row[f"expert_{name}_prob_abs_gap"]=abs(pa-pb)
        row[f"dir_expert_{name}_rank_diff"]=ra-rb
        row[f"dir_expert_{name}_prob_diff"]=pa-pb
        vote_margin += 1 if ra<rb else (-1 if ra>rb else 0)
    row["dir_expert_vote_margin"]=float(vote_margin)
    row["pair_expert_direction_agreement"]=abs(float(vote_margin))/max(1.0,float(len(expert_names)))
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
    missing_odds_days=[]
    priced_races=0
    multi_hit_races=0

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
            rec=l17_rows.get(rid)
            if rec is None:
                continue
            odds_rec=odds_by_race.get(rid)
            if odds_rec is None:
                continue

            hno=horse_number_map(pack)
            horses=list(rec["horses"])
            if len(horses)!=int(rec["field_size"]):
                raise SystemExit(f"L1.7 field drift race={rid}")
            if any(str(h["horse_id"]) not in hno for h in horses):
                raise SystemExit(f"horse missing from pack race={rid}")

            odds_map=decode_odds(odds_rec)
            payouts,present=payout_map(pack)
            if "EXACTA" not in present:
                continue

            base=race_features(pack,rec,date)
            race_rows=[]
            for ha,hb in itertools.permutations(horses,2):
                aid=str(ha["horse_id"]); bid=str(hb["horse_id"])
                an=int(hno[aid]); bn=int(hno[bid])
                key=("EXACTA",(an,bn))
                odd=odds_map.get(key)
                if odd is None:
                    continue
                ret=float(payouts.get(key,0.0))
                unordered_hids="|".join(sorted((aid,bid)))
                unordered_nums="-".join(str(x) for x in sorted((an,bn)))
                row={
                    "year":year,
                    "race_id":rid,
                    "race_date":date,
                    "ticket_horse_ids":f"{aid}>{bid}",
                    "ticket_numbers":f"{an}>{bn}",
                    "unordered_horse_ids":unordered_hids,
                    "unordered_numbers":unordered_nums,
                    "a_horse_id":aid,
                    "b_horse_id":bid,
                    "a_horse_number":an,
                    "b_horse_number":bn,
                    "hit":int(ret>0),
                    "return_yen_per100":ret,
                    "odds":float(odd),
                    **base,
                    **pair_premarket_features(ha,hb,expert_names),
                }
                race_rows.append(row)

            if race_rows:
                priced_races+=1
                hit_count=sum(x["hit"] for x in race_rows)
                multi_hit_races += int(hit_count>1)
                rows.extend(race_rows)

    if not rows:
        raise SystemExit(f"no EXACTA rows y={year}")

    df=pd.DataFrame(rows)
    print("EXACTA_YEAR_READY "+json.dumps({
        "year":year,
        "l17_races":len(l17_rows),
        "priced_races":priced_races,
        "priced_tickets":len(df),
        "positive_tickets":int(df["hit"].sum()),
        "multi_hit_races":multi_hit_races,
        "missing_odds_days":missing_odds_days,
    },ensure_ascii=False,separators=(",",":")),flush=True)
    return df

def add_market_features(df):
    z=df.copy()
    z["market_inv_odds"]=1.0/z["odds"].clip(lower=1e-12)
    denom=z.groupby(["year","race_id"])["market_inv_odds"].transform("sum")
    z["market_q"]=z["market_inv_odds"]/denom.clip(lower=1e-12)
    z["market_log_q"]=np.log(z["market_q"].clip(lower=1e-12))
    z["market_rank"]=z.groupby(["year","race_id"])["market_q"].rank(method="min",ascending=False).astype(int)
    return z

def base_feature_columns(df):
    cols=[c for c in df.columns if c not in IDENTITY and c not in {"market_log_q"}]
    bad=[c for c in cols if any(tok in c.lower() for tok in FORBIDDEN_TOKENS)]
    if bad:
        raise SystemExit(f"forbidden premarket model features={sorted(bad)}")
    # Raw horse numbers are identifiers, not ability.
    cols=[c for c in cols if c not in {"a_horse_number","b_horse_number"}]
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

def train_ranker(train,test,cols,seed):
    good=train.groupby(["year","race_id"])["hit"].transform("sum")>0
    tr=train.loc[good].copy().sort_values(["year","race_id","ticket_numbers"]).reset_index(drop=True)
    te=test.copy().sort_values(["year","race_id","ticket_numbers"])
    te["_orig_index"]=te.index
    te=te.reset_index(drop=True)

    xtr,xte=encode(tr,te,cols)
    y=tr["hit"].astype(int).to_numpy()
    group=tr.groupby(["year","race_id"],sort=False).size().tolist()

    model=lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=360,
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
    model.fit(xtr,y,group=group)
    pred=np.asarray(model.predict(xte),dtype=float)
    s=pd.Series(pred,index=te["_orig_index"].astype(int).to_numpy())
    out=s.reindex(test.index).to_numpy()
    imp=pd.DataFrame({"feature":cols,"gain":model.feature_importances_})
    return out,imp

def add_model_ranks(df):
    z=df.copy()
    z["l17_rank"]=z.groupby(["year","race_id"])["l17_only_score"].rank(method="min",ascending=False).astype(int)
    z["model_rank"]=z.groupby(["year","race_id"])["market_aware_score"].rank(method="min",ascending=False).astype(int)
    z["rank_upgrade"]=z["market_rank"]-z["model_rank"]
    return z

def reversal_invariant_audit(df,base_cols):
    dir_cols=[c for c in base_cols if c.startswith("dir_")]
    symmetric_numeric=[
        c for c in base_cols
        if c not in dir_cols
        and c.startswith(("pair_","expert_"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    checked=0
    for (_,rid,ukey),g in df.groupby(["year","race_id","unordered_horse_ids"],sort=False):
        if len(g)!=2:
            continue
        a,b=g.iloc[0],g.iloc[1]
        if not (str(a["a_horse_id"])==str(b["b_horse_id"]) and str(a["b_horse_id"])==str(b["a_horse_id"])):
            raise SystemExit(f"reversal identity mismatch race={rid} pair={ukey}")
        for c in symmetric_numeric:
            if not np.isclose(float(a[c]),float(b[c]),rtol=0,atol=1e-9):
                raise SystemExit(f"reversal symmetric drift race={rid} pair={ukey} col={c}")
        for c in dir_cols:
            if not np.isclose(float(a[c]),-float(b[c]),rtol=0,atol=1e-9):
                raise SystemExit(f"reversal sign drift race={rid} pair={ukey} col={c}")
        checked+=1
    if checked<1000:
        raise SystemExit(f"too few reversal pairs checked: {checked}")
    return checked,len(symmetric_numeric),len(dir_cols)

def ranking_coverage(df,year):
    rows=[]
    source=df["race_id"].nunique()
    for label,col in [("MARKET","market_rank"),("L17_ONLY_MODEL","l17_rank"),("MARKET_AWARE_MODEL","model_rank")]:
        hit_rows=df[df["hit"]==1]
        for k in TOP_K:
            hit_races=hit_rows[hit_rows[col]<=k]["race_id"].nunique()
            rows.append({
                "year":year,
                "ranking":label,
                "top_k":k,
                "hit_races":hit_races,
                "source_races":source,
                "exact_hit_coverage_pct":100.0*hit_races/source if source else 0.0,
            })
    return rows

def unique_hit_pairs(df):
    counts=df.groupby(["year","race_id"])["hit"].sum()
    eligible=set(counts[counts==1].index.tolist())
    return df.set_index(["year","race_id"]).loc[list(eligible)].reset_index() if eligible else df.iloc[0:0].copy()

def direction_audit(df,year):
    rows=[]
    unique=unique_hit_pairs(df)
    rankers=[("MARKET","market_rank"),("L17_ONLY_MODEL","l17_rank"),("MARKET_AWARE_MODEL","model_rank")]
    for label,col in rankers:
        per=[]
        for rid,g in unique.groupby("race_id",sort=False):
            hit=g[g["hit"]==1]
            if len(hit)!=1:
                continue
            h=hit.iloc[0]
            rev=g[(g["a_horse_id"].astype(str)==str(h["b_horse_id"])) & (g["b_horse_id"].astype(str)==str(h["a_horse_id"]))]
            if len(rev)!=1:
                continue
            rr=rev.iloc[0]
            hr=int(h[col]); vr=int(rr[col])
            status="CORRECT" if hr<vr else ("REVERSE" if hr>vr else "TIE")
            per.append((rid,hr,vr,status))
        for k in TOP_K:
            cap=[x for x in per if min(x[1],x[2])<=k]
            correct=sum(x[3]=="CORRECT" for x in cap)
            reverse=sum(x[3]=="REVERSE" for x in cap)
            ties=sum(x[3]=="TIE" for x in cap)
            rows.append({
                "year":year,
                "ranking":label,
                "top_k":k,
                "direction_eligible_races":len(per),
                "pair_captured_races":len(cap),
                "pair_capture_pct":100.0*len(cap)/len(per) if per else 0.0,
                "direction_correct_races":correct,
                "direction_reverse_races":reverse,
                "direction_tie_races":ties,
                "direction_correct_pct_of_captured":100.0*correct/len(cap) if cap else 0.0,
                "direction_correct_pct_excluding_ties":100.0*correct/(correct+reverse) if (correct+reverse) else 0.0,
            })
    return rows

def top1_miss_decomposition(df,year):
    rows=[]
    unique=unique_hit_pairs(df)
    for label,col in [("MARKET","market_rank"),("L17_ONLY_MODEL","l17_rank"),("MARKET_AWARE_MODEL","model_rank")]:
        counts={"HIT":0,"REVERSE_MISS":0,"PAIR_MISS":0}
        excluded_top_rank_tie=0
        eligible=0
        for rid,g in unique.groupby("race_id",sort=False):
            hit=g[g["hit"]==1]
            if len(hit)!=1:
                continue
            h=hit.iloc[0]
            best_rank=int(g[col].min())
            top=g[g[col]==best_rank]
            if len(top)!=1:
                excluded_top_rank_tie+=1
                continue
            eligible+=1
            t=top.iloc[0]
            if int(t["hit"])==1:
                counts["HIT"]+=1
            elif str(t["unordered_horse_ids"])==str(h["unordered_horse_ids"]):
                counts["REVERSE_MISS"]+=1
            else:
                counts["PAIR_MISS"]+=1
        for cat,n in counts.items():
            rows.append({
                "year":year,
                "ranking":label,
                "category":cat,
                "races":n,
                "eligible_unique_top_races":eligible,
                "share_pct":100.0*n/eligible if eligible else 0.0,
                "excluded_top_rank_tie_races":excluded_top_rank_tie,
            })
    return rows

def winner_diagnostics(df):
    unique=unique_hit_pairs(df)
    if unique.empty:
        return unique
    hit=unique[unique["hit"]==1].copy()
    rev_map={}
    for (y,rid),g in unique.groupby(["year","race_id"],sort=False):
        h=g[g["hit"]==1]
        if len(h)!=1:
            continue
        h=h.iloc[0]
        rev=g[(g["a_horse_id"].astype(str)==str(h["b_horse_id"])) & (g["b_horse_id"].astype(str)==str(h["a_horse_id"]))]
        if len(rev)==1:
            rev_map[(y,rid)]=rev.iloc[0]
    rows=[]
    for _,h in hit.iterrows():
        key=(h["year"],h["race_id"])
        rr=rev_map.get(key)
        rows.append({
            "year":int(h["year"]),
            "race_id":h["race_id"],
            "race_date":h["race_date"],
            "winning_ticket":h["ticket_numbers"],
            "reverse_ticket":None if rr is None else rr["ticket_numbers"],
            "odds":float(h["odds"]),
            "market_q":float(h["market_q"]),
            "market_rank":int(h["market_rank"]),
            "l17_rank":int(h["l17_rank"]),
            "model_rank":int(h["model_rank"]),
            "rank_upgrade":int(h["rank_upgrade"]),
            "l17_direction_correct":None if rr is None else bool(int(h["l17_rank"])<int(rr["l17_rank"])),
            "model_direction_correct":None if rr is None else bool(int(h["model_rank"])<int(rr["model_rank"])),
            "market_direction_correct":None if rr is None else bool(int(h["market_rank"])<int(rr["market_rank"])),
        })
    return pd.DataFrame(rows)

def write_csv(path,rows):
    if isinstance(rows,pd.DataFrame):
        rows.to_csv(path,index=False)
    else:
        pd.DataFrame(rows).to_csv(path,index=False)

def main():
    a=parse_args()
    lp=parse_paths(a.l17_year)
    if set(lp)!=set(YEARS):
        raise SystemExit("L1.7 year path mismatch")
    if 2026 in lp:
        raise SystemExit("2026 sealed")

    l17={y:load_l17(lp[y],y) for y in YEARS}
    frames={y:add_market_features(build_year_frame(y,l17[y],a.backfill_root)) for y in YEARS}
    base_cols=base_feature_columns(frames[2022])
    market_cols=base_cols+["market_q","market_log_q","market_rank"]

    reversal={}
    for y in YEARS:
        checked,sym_n,dir_n=reversal_invariant_audit(frames[y],base_cols)
        reversal[y]={"checked_pairs":checked,"symmetric_numeric_features":sym_n,"directional_signed_features":dir_n}

    predictions={}
    fold_rows=[]
    imp_l17=[]
    imp_market=[]
    coverage=[]
    direction=[]
    miss=[]
    for y in TEST_YEARS:
        train=pd.concat([frames[t] for t in YEARS if t<y],ignore_index=True)
        test=frames[y].copy().reset_index(drop=True)

        l17_score,i1=train_ranker(train,test,base_cols,96000+y)
        market_score,i2=train_ranker(train,test,market_cols,97000+y)
        test["l17_only_score"]=l17_score
        test["market_aware_score"]=market_score
        test=add_model_ranks(test)
        predictions[y]=test

        coverage.extend(ranking_coverage(test,y))
        direction.extend(direction_audit(test,y))
        miss.extend(top1_miss_decomposition(test,y))
        fold_rows.append({
            "test_year":y,
            "train_years":"|".join(str(t) for t in YEARS if t<y),
            "train_tickets":len(train),
            "test_tickets":len(test),
            "source_races":test["race_id"].nunique(),
            "positive_tickets":int(test["hit"].sum()),
            "unique_direction_races":int((test.groupby("race_id")["hit"].sum()==1).sum()),
            "multi_hit_direction_ambiguous_races":int((test.groupby("race_id")["hit"].sum()>1).sum()),
            "premarket_feature_count":len(base_cols),
            "market_aware_feature_count":len(market_cols),
            "mean_abs_rank_upgrade":float(test["rank_upgrade"].abs().mean()),
            "pct_tickets_upgraded_vs_market":100.0*float((test["rank_upgrade"]>0).mean()),
        })
        i1["test_year"]=y; i2["test_year"]=y
        imp_l17.append(i1); imp_market.append(i2)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"fold-metrics.csv",fold_rows)
    write_csv(out/"ranking-coverage.csv",coverage)
    write_csv(out/"direction-audit.csv",direction)
    write_csv(out/"top1-miss-decomposition.csv",miss)
    write_csv(out/"feature-importance-l17-only.csv",pd.concat(imp_l17,ignore_index=True))
    write_csv(out/"feature-importance-market-aware.csv",pd.concat(imp_market,ignore_index=True))
    winners=pd.concat([winner_diagnostics(predictions[y]) for y in TEST_YEARS],ignore_index=True)
    with gzip.open(out/"winner-ticket-diagnostics.csv.gz","wt",newline="",encoding="utf-8") as f:
        winners.to_csv(f,index=False)

    summary={
        "contract":"L2_EXACTA_V0_RESULT",
        "bet_type":"EXACTA",
        "stage":"V0_AND_DIRECTION_AUDIT",
        "all_ordered_pairs":True,
        "ticket_count_formula":"N*(N-1)",
        "market_probability":"(1/EXACTA_odds)/sum(1/EXACTA_odds) within race",
        "rankers":["MARKET","L17_ONLY_MODEL","MARKET_AWARE_MODEL"],
        "rank_upgrade":"market_rank - market_aware_model_rank",
        "law_search":False,
        "roi_policy_search":False,
        "direct_vs_decomposed_comparison":False,
        "direction_audit":{
            "strict_unique_winning_ticket_only":True,
            "multi_payout_dead_heat_races_excluded_from_direction_judgment":True,
            "primary_metric":"direction_correct_pct_of_captured",
            "miss_classes":["PAIR_MISS","REVERSE_MISS","HIT"],
        },
        "reversal_invariant":reversal,
        "walk_forward":{"years":list(TEST_YEARS),"training_rule":"all prior years only"},
        "2026_locked":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 EXACTA V0 — Ordered Pair / Direction Audit\n\n"
        "Every priced ordered pair A->B is scored. No LAW or ROI policy is searched in V0. "
        "The comparison is MARKET vs L17_ONLY_MODEL vs MARKET_AWARE_MODEL. "
        "The primary diagnostic is whether, after the actual first-two horses are captured, "
        "the correct order is ranked above the reverse order. "
        "Premarket reversal invariants require symmetric pair features to remain identical and "
        "all directional features to change sign under A->B / B->A reversal. "
        "Races with multiple winning EXACTA payouts (for example dead-heat ambiguity) remain in ranking coverage "
        "but are excluded from strict direction judgment. 2026 is sealed.\n",
        encoding="utf-8"
    )
    print("L2_EXACTA_V0_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
