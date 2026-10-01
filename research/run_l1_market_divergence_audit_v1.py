#!/usr/bin/env python3
import argparse,csv,gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from build_l2_bet_kings_dataset_v1 import load_day,load_odds_day,final_odds_tuple,finite

YEARS=(2022,2023,2024,2025)
CONF_YEARS=(2023,2024,2025)

def parse_args():
    p=argparse.ArgumentParser(description="L1.7 vs final WIN market divergence audit.")
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def load_dataset(path):
    path=Path(path)
    manifest=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    rows=[]
    with gzip.open(path/manifest["file"],"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    df=pd.DataFrame(rows)
    df["race_id"]=df["race_id"].astype(str)
    df["horse_id"]=df["horse_id"].astype(str)
    df["year"]=pd.to_numeric(df["year"],errors="raise").astype(int)
    for c in ("horse_number","consensus_rank","field_size","top1_votes","vote2","vote3","target_win","target_top3"):
        df[c]=pd.to_numeric(df[c],errors="coerce")
    return manifest,df

def conf_feat(row):
    return {
        "king_rank":str(int(row["consensus_rank"])),
        "vote1":int(round(float(row["top1_votes"]))),
        "vote2":int(round(float(row["vote2"]))),
        "vote3":int(round(float(row["vote3"]))),
    }

def fit_confidence(df):
    z=df[df["target_top3"].notna()].copy()
    if z.empty:
        raise ValueError("confidence training rows empty")
    v=DictVectorizer(sparse=True)
    X=v.fit_transform([conf_feat(r) for _,r in z.iterrows()]).tocsr()
    X.indices=X.indices.astype(np.int32,copy=False)
    X.indptr=X.indptr.astype(np.int32,copy=False)
    m=LogisticRegression(C=1.0,solver="liblinear",max_iter=1000)
    m.fit(X,z["target_top3"].astype(int).to_numpy())
    return v,m

def predict_confidence(v,m,df):
    X=v.transform([conf_feat(r) for _,r in df.iterrows()]).tocsr()
    X.indices=X.indices.astype(np.int32,copy=False)
    X.indptr=X.indptr.astype(np.int32,copy=False)
    return m.predict_proba(X)[:,1]

def rank_thresholds(calib,q):
    tmp=calib.copy().reset_index(drop=True)
    tmp["q"]=np.asarray(q,dtype=float)
    allq=tmp["q"].to_numpy()
    global_thr=tuple(np.quantile(allq,[1/3,2/3])) if len(allq) else (0.33,0.67)
    out={}
    for rank in range(1,21):
        vals=tmp.loc[tmp["consensus_rank"]==rank,"q"].to_numpy()
        out[rank]=tuple(np.quantile(vals,[1/3,2/3])) if len(vals)>=30 else global_thr
    return out

def tag_for(rank,q,thr):
    lo,hi=thr.get(int(rank),next(iter(thr.values())))
    if q<lo: return "LOW"
    if q>=hi: return "HIGH"
    return "MID"

def attach_strict_oos_confidence(df):
    out=df.copy()
    out["confidence_score"]=np.nan
    out["confidence_tag"]="NA_NO_PRIOR"
    for test_year in CONF_YEARS:
        train=out[(out["year"]<test_year) & out["target_top3"].notna()].copy()
        test_idx=out.index[out["year"]==test_year]
        races=train[["race_date","race_id"]].drop_duplicates().sort_values(["race_date","race_id"])
        cut=max(1,int(len(races)*0.8))
        fit_ids=set(races.iloc[:cut]["race_id"].astype(str))
        cal_ids=set(races.iloc[cut:]["race_id"].astype(str))
        fit=train[train["race_id"].isin(fit_ids)].copy()
        calib=train[train["race_id"].isin(cal_ids)].copy().reset_index(drop=True)
        v0,m0=fit_confidence(fit)
        qcal=predict_confidence(v0,m0,calib)
        thr=rank_thresholds(calib,qcal)
        v,m=fit_confidence(train)
        q=predict_confidence(v,m,out.loc[test_idx])
        out.loc[test_idx,"confidence_score"]=q
        out.loc[test_idx,"confidence_tag"]=[tag_for(r,qv,thr) for r,qv in zip(out.loc[test_idx,"consensus_rank"],q)]
    return out

def market_and_finish(df,root):
    root=Path(root)
    rows=[]
    skipped=[]
    bydate=defaultdict(list)
    for rid,date in df[["race_id","race_date"]].drop_duplicates().itertuples(index=False):
        bydate[str(date)[:10]].append(str(rid))
    for date,rids in sorted(bydate.items()):
        wanted=set(rids)
        packs=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        oddsday=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in rids:
            sub=df[df["race_id"]==rid]
            pack=packs.get(rid)
            orec=oddsday.get(rid)
            if pack is None or orec is None:
                skipped.append({"race_id":rid,"race_date":date,"reason":"daily_or_odds_missing"})
                continue
            raw=((orec.get("odds") or {}).get("1") or {})
            odds={}
            for key,val in raw.items():
                k=str(key)
                if not k.isascii() or not k.isdigit(): continue
                tup=final_odds_tuple(val)
                if not tup: continue
                odd=finite(tup[0])
                if odd is not None and odd>0:
                    odds[int(k)]=float(odd)
            expected=set(int(x) for x in sub["horse_number"].dropna().astype(int))
            if set(odds)!=expected:
                skipped.append({"race_id":rid,"race_date":date,"reason":f"win_odds_incomplete:{len(odds)}/{len(expected)}"})
                continue
            unique=sorted(set(odds.values()))
            market_rank={o:1+sum(1 for x in odds.values() if x<o) for o in unique}
            finish={}
            for r in pack.get("results") or []:
                if r.get("result_status")!="FINISHED": continue
                hid=str(r.get("horse_id") or "")
                try: pos=int(r.get("official_finish_position"))
                except (TypeError,ValueError): continue
                if hid: finish[hid]=pos
            for idx,r in sub.iterrows():
                no=int(r["horse_number"])
                hid=str(r["horse_id"])
                rows.append({
                    "idx":int(idx),
                    "final_win_odds":odds[no],
                    "market_rank":int(market_rank[odds[no]]),
                    "finish_position":finish.get(hid),
                })
    aux=pd.DataFrame(rows).set_index("idx") if rows else pd.DataFrame()
    return aux,skipped

def gap_band(g):
    if g>=5: return "UP_5_PLUS"
    if g>=2: return "UP_2_4"
    if g<=-5: return "DOWN_5_PLUS"
    if g<=-2: return "DOWN_2_4"
    return "NEAR_-1_1"

def direction(g):
    if g>=2: return "L1_UPGRADE"
    if g<=-2: return "L1_DOWNGRADE"
    return "NEAR"

def rank_bucket(x):
    x=int(x)
    return str(x) if x<=10 else "11+"

def aggregate(z,keys):
    out=[]
    for vals,g in z.groupby(keys,dropna=False,sort=True):
        if not isinstance(vals,tuple): vals=(vals,)
        row={k:v for k,v in zip(keys,vals)}
        n=len(g)
        row.update({
            "horses":n,
            "races":g["race_id"].nunique(),
            "mean_l1_rank":float(g["consensus_rank"].mean()),
            "mean_market_rank":float(g["market_rank"].mean()),
            "mean_gap":float(g["rank_gap"].mean()),
            "mean_final_win_odds":float(g["final_win_odds"].mean()),
            "mean_finish_position":float(g["finish_position"].dropna().mean()) if g["finish_position"].notna().any() else None,
            "actual_win_pct":100*float(g["actual_win"].mean()),
            "actual_top3_pct":100*float(g["actual_top3"].mean()),
            "market_peer_win_pct":100*float(g["market_peer_win_rate"].mean()),
            "market_peer_top3_pct":100*float(g["market_peer_top3_rate"].mean()),
            "win_lift_vs_market_peer_pp":100*float((g["actual_win"]-g["market_peer_win_rate"]).mean()),
            "top3_lift_vs_market_peer_pp":100*float((g["actual_top3"]-g["market_peer_top3_rate"]).mean()),
            "mean_confidence":float(g["confidence_score"].dropna().mean()) if g["confidence_score"].notna().any() else None,
        })
        if "direction" in row:
            if row["direction"]=="L1_UPGRADE":
                row["direction_correct_top3_effect_pp"]=row["top3_lift_vs_market_peer_pp"]
            elif row["direction"]=="L1_DOWNGRADE":
                row["direction_correct_top3_effect_pp"]=-row["top3_lift_vs_market_peer_pp"]
            else:
                row["direction_correct_top3_effect_pp"]=None
        out.append(row)
    return out

def write_csv(path,rows):
    path=Path(path)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def stable_cells(z):
    rows=[]
    q=z[(z["year"].isin(CONF_YEARS)) & (z["direction"]!="NEAR")].copy()
    q=q[(q["consensus_rank"]<=10) & (q["market_rank"]<=10)]
    keys=["l1_rank_bucket","market_rank_bucket","confidence_tag","direction"]
    for vals,g in q.groupby(keys,sort=True):
        per=[]
        for y,gy in g.groupby("year"):
            if len(gy)<30: continue
            lift=100*float((gy["actual_top3"]-gy["market_peer_top3_rate"]).mean())
            correct=lift>0 if vals[3]=="L1_UPGRADE" else lift<0
            per.append((int(y),len(gy),lift,bool(correct)))
        if len(per)<2 or len(g)<100: continue
        pooled=100*float((g["actual_top3"]-g["market_peer_top3_rate"]).mean())
        pooled_correct=pooled>0 if vals[3]=="L1_UPGRADE" else pooled<0
        rows.append({
            "l1_rank":vals[0],"market_rank":vals[1],"confidence_tag":vals[2],"direction":vals[3],
            "horses_total":len(g),"years_with_n_ge_30":len(per),"years_correct":sum(x[3] for x in per),
            "pooled_top3_lift_vs_market_peer_pp":pooled,
            "pooled_direction_correct_effect_pp":pooled if vals[3]=="L1_UPGRADE" else -pooled,
            "pooled_correct":pooled_correct,
            "year_details":"|".join(f"{y}:n={n}:lift={lift:.3f}:ok={int(ok)}" for y,n,lift,ok in per),
        })
    rows.sort(key=lambda r:(-r["years_correct"],-r["years_with_n_ge_30"],-r["pooled_direction_correct_effect_pp"],-r["horses_total"]))
    return rows

def main():
    a=parse_args()
    manifest,df=load_dataset(a.dataset_dir)
    if set(df["year"].unique())!=set(YEARS):
        raise SystemExit(f"year coverage drift: {sorted(df['year'].unique())}")
    if 2026 in set(df["year"]): raise SystemExit("2026 sealed")
    aux,skipped=market_and_finish(df,a.backfill_root)
    keep=df.index.intersection(aux.index)
    z=df.loc[keep].copy()
    z=z.join(aux.loc[keep])
    z["actual_win"]=z["finish_position"].eq(1).astype(float)
    z["actual_top3"]=z["finish_position"].le(3).astype(float)
    z["rank_gap"]=z["market_rank"].astype(int)-z["consensus_rank"].astype(int)
    z["gap_band"]=[gap_band(x) for x in z["rank_gap"]]
    z["direction"]=[direction(x) for x in z["rank_gap"]]
    z["l1_rank_bucket"]=[rank_bucket(x) for x in z["consensus_rank"]]
    z["market_rank_bucket"]=[rank_bucket(x) for x in z["market_rank"]]
    z=attach_strict_oos_confidence(z)

    peer=z.groupby(["year","market_rank"],dropna=False).agg(
        market_peer_win_rate=("actual_win","mean"),
        market_peer_top3_rate=("actual_top3","mean"),
        market_peer_horses=("actual_top3","size"),
    ).reset_index()
    z=z.merge(peer,on=["year","market_rank"],how="left",validate="many_to_one")

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    year_direction=aggregate(z,["year","direction"])
    gap_conf=aggregate(z,["year","gap_band","confidence_tag","direction"])
    cells=aggregate(z,["year","market_rank_bucket","l1_rank_bucket","confidence_tag","direction"])
    pooled=aggregate(z[z["year"].isin(CONF_YEARS)],["market_rank_bucket","l1_rank_bucket","confidence_tag","direction"])
    stable=stable_cells(z)

    write_csv(out/"year-direction.csv",year_direction)
    write_csv(out/"gap-confidence.csv",gap_conf)
    write_csv(out/"market-l1-confidence-cells.csv",cells)
    write_csv(out/"pooled-confidence-cells.csv",pooled)
    write_csv(out/"stable-disagreements.csv",stable)
    write_csv(out/"skipped-market-races.csv",skipped)

    compact=[]
    for direction_name in ("L1_UPGRADE","L1_DOWNGRADE"):
        g=z[(z["year"].isin(CONF_YEARS)) & (z["direction"]==direction_name)]
        for tag in ("LOW","MID","HIGH"):
            x=g[g["confidence_tag"]==tag]
            if x.empty: continue
            lift=100*float((x["actual_top3"]-x["market_peer_top3_rate"]).mean())
            compact.append({
                "direction":direction_name,"confidence_tag":tag,"horses":len(x),"races":x["race_id"].nunique(),
                "actual_top3_pct":100*float(x["actual_top3"].mean()),
                "market_peer_top3_pct":100*float(x["market_peer_top3_rate"].mean()),
                "top3_lift_vs_market_peer_pp":lift,
                "direction_correct_effect_pp":lift if direction_name=="L1_UPGRADE" else -lift,
            })
    write_csv(out/"headline.csv",compact)

    summary={
        "contract":"L1_MARKET_DIVERGENCE_AUDIT_V1",
        "definition":"rank_gap = tie-safe final WIN market rank - L1.7 consensus rank; + means L1 rates horse higher than market",
        "market_rank_ties":"competition rank from final odds only; equal odds share rank; never row-order first ranking",
        "confidence":"strict prior-data P(podium | consensus_rank,vote1,vote2,vote3); LOW/MID/HIGH thresholds learned from prior calibration races within rank",
        "confidence_years":[2023,2024,2025],
        "year_2022_confidence":"NA_NO_PRIOR because no earlier seven-king year is available in this lane",
        "market_peer_baseline":"same test_year and same tie-safe final WIN market rank",
        "correctness":"upgrade is supported when top3 lift vs market-rank peers >0; downgrade is supported when lift <0",
        "races_market_covered":int(z["race_id"].nunique()),
        "horses_market_covered":int(len(z)),
        "skipped_market_races":len(skipped),
        "stable_cell_rule":"exact L1 rank<=10 x market rank<=10 x confidence, disagreement >=2; >=30 horses/year, >=2 years, >=100 pooled",
        "stable_cells":len(stable),
        "2026_locked":True,
        "l1_uses_market":False,
        "promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== HEADLINE =====")
    print((out/"headline.csv").read_text(encoding="utf-8"))
    print("===== STABLE DISAGREEMENTS TOP 30 =====")
    if stable:
        print("\n".join((out/"stable-disagreements.csv").read_text(encoding="utf-8").splitlines()[:31]))
    print("L1_MARKET_DIVERGENCE_AUDIT_V1_READY")

if __name__=="__main__":
    main()
