#!/usr/bin/env python3
import argparse,csv,gzip,itertools,json,math,time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from build_l2_bet_kings_dataset_v1 import (
    load_day,load_odds_day,iter_decoded_odds,payout_map
)
from run_l2_core_v1 import (
    BET_TYPES,CalAgg,clip01,load_dataset,probability_invariant,ticket_probability,write_csv
)

FOLDS=((2023,(2022,)),(2024,(2022,2023)),(2025,(2022,2023,2024)))
COLLAPSE_FACTORS=(0.10,0.25,0.50,0.75,1.00)
OUTSIDER_BOOSTS=(0.0,0.5,1.0,2.0,4.0)
EPS=1e-12

def parse_args():
    p=argparse.ArgumentParser(description="Mixture-of-worlds coherent race distribution V2.")
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--ballots-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_year_paths(items):
    out={}
    for spec in items:
        y,path=spec.split(":",1)
        out[int(y)]=path
    if set(out)!={2022,2023,2024,2025}:
        raise SystemExit(f"ballot years mismatch: {sorted(out)}")
    return out

def load_outsider_ballots(paths):
    stats={}
    race_candidates=defaultdict(set)
    for year,path in sorted(paths.items()):
        temp=defaultdict(lambda:defaultdict(lambda:{"weighted":0,"support":0,"v1":0,"v2":0,"v3":0}))
        with gzip.open(path,"rt",encoding="utf-8",newline="") as fh:
            for row in csv.DictReader(fh):
                rid=str(row["race_id"])
                cand=str(row["candidate"])
                race_candidates[(year,rid)].add(cand)
                seen=set()
                for pos,w in ((1,3),(2,2),(3,1)):
                    hid=str(row.get(f"top{pos}_horse_id") or "")
                    if not hid:
                        continue
                    if hid in seen:
                        raise SystemExit(f"duplicate horse inside outsider ballot y={year} race={rid} candidate={cand}")
                    seen.add(hid)
                    d=temp[rid][hid]
                    d["weighted"]+=w
                    d["support"]+=1
                    d[f"v{pos}"]+=1
        for rid,hmap in temp.items():
            if len(race_candidates[(year,rid)])!=13:
                raise SystemExit(f"expected 13 outsider candidates y={year} race={rid}, got={len(race_candidates[(year,rid)])}")
            for hid,d in hmap.items():
                stats[(year,str(rid),str(hid))]=dict(d)
    return stats,race_candidates

def attach_outsider(df,stats,race_candidates):
    z=df.copy()
    vals=[]
    for r in z.itertuples(index=False):
        key=(int(r.year),str(r.race_id),str(r.horse_id))
        d=stats.get(key,{"weighted":0,"support":0,"v1":0,"v2":0,"v3":0})
        vals.append((d["weighted"],d["support"],d["v1"],d["v2"],d["v3"]))
    arr=np.asarray(vals,dtype=float)
    z["outs_weighted"]=arr[:,0]
    z["outs_support"]=arr[:,1]
    z["outs_v1"]=arr[:,2]
    z["outs_v2"]=arr[:,3]
    z["outs_v3"]=arr[:,4]
    races=set((int(y),str(rid)) for y,rid in zip(z["year"],z["race_id"]))
    missing=[k for k in races if len(race_candidates.get(k,set()))!=13]
    if missing:
        raise SystemExit(f"outsider race coverage missing count={len(missing)} sample={missing[:3]}")
    return z

RACE_FEATURES=(
    "field_size",
    "race_entropy","race_top1_probability","race_top2_probability_sum","race_top3_probability_sum",
    "race_top1_top2_gap","race_rank_std_mean","race_rank_std_max",
    "race_probability_std_mean","race_probability_std_max",
    "king1_mean_rank_pct","king1_rank_std_pct","king1_top1_vote_share",
    "king1_top3_support_share","king1_top6_support_share",
    "king1_mean_probability","king1_probability_std",
    "outs_outside_top5_positive_count","outs_outside_top5_weighted_sum",
    "outs_outside_top5_weighted_max","outs_outside_top5_support_max",
    "outs_outside_top5_v1_sum","outs_outside_top5_v1_max",
)

def build_race_frame(df):
    rows=[]
    for (year,rid),g in df.groupby(["year","race_id"],sort=False):
        g=g.sort_values(["consensus_rank","horse_number"],kind="stable")
        one=g[g["consensus_rank"]==1]
        if len(one)!=1:
            raise SystemExit(f"rank1 coverage drift y={year} race={rid} n={len(one)}")
        k=one.iloc[0]
        outside=g[(g["consensus_rank"]>5)&(g["outs_weighted"]>0)]
        top3_available=g["target_top3"].notna().any()
        collapse_label=None
        outsider_label=None
        if top3_available:
            collapse_label=int(float(k["target_top3"])<0.5)
            outsider_label=int(((outside["target_top3"].fillna(0).astype(float)>0.5)).any())
        row={
            "year":int(year),"race_id":str(rid),"race_date":str(k["race_date"])[:10],
            "collapse_label":collapse_label,"outsider_label":outsider_label,
            "field_size":float(k["field_size"]),
            "race_entropy":float(k["race_entropy"]),
            "race_top1_probability":float(k["race_top1_probability"]),
            "race_top2_probability_sum":float(k["race_top2_probability_sum"]),
            "race_top3_probability_sum":float(k["race_top3_probability_sum"]),
            "race_top1_top2_gap":float(k["race_top1_top2_gap"]),
            "race_rank_std_mean":float(k["race_rank_std_mean"]),
            "race_rank_std_max":float(k["race_rank_std_max"]),
            "race_probability_std_mean":float(k["race_probability_std_mean"]),
            "race_probability_std_max":float(k["race_probability_std_max"]),
            "king1_mean_rank_pct":float(k["mean_rank_pct"]),
            "king1_rank_std_pct":float(k["rank_std_pct"]),
            "king1_top1_vote_share":float(k["top1_vote_share"]),
            "king1_top3_support_share":float(k["top3_support_share"]),
            "king1_top6_support_share":float(k["top6_support_share"]),
            "king1_mean_probability":float(k["mean_probability"]),
            "king1_probability_std":float(k["probability_std"]),
            "outs_outside_top5_positive_count":float(len(outside)),
            "outs_outside_top5_weighted_sum":float(outside["outs_weighted"].sum()) if len(outside) else 0.0,
            "outs_outside_top5_weighted_max":float(outside["outs_weighted"].max()) if len(outside) else 0.0,
            "outs_outside_top5_support_max":float(outside["outs_support"].max()) if len(outside) else 0.0,
            "outs_outside_top5_v1_sum":float(outside["outs_v1"].sum()) if len(outside) else 0.0,
            "outs_outside_top5_v1_max":float(outside["outs_v1"].max()) if len(outside) else 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)

def sigmoid(x):
    x=np.clip(np.asarray(x,dtype=float),-60,60)
    return 1.0/(1.0+np.exp(-x))

def binary_logloss(y,p):
    yy=np.asarray(y,dtype=float)
    pp=np.clip(np.asarray(p,dtype=float),1e-9,1-1e-9)
    return float(np.mean(-(yy*np.log(pp)+(1-yy)*np.log(1-pp))))

def binary_brier(y,p):
    yy=np.asarray(y,dtype=float)
    pp=np.asarray(p,dtype=float)
    return float(np.mean((pp-yy)**2))

def choose_binary_temperature(y,raw):
    best=(1.0,float("inf"))
    for t in np.exp(np.linspace(math.log(0.35),math.log(3.0),31)):
        loss=binary_logloss(y,sigmoid(np.asarray(raw)/float(t)))
        if loss<best[1]:
            best=(float(t),loss)
    return best

def gate_params(seed):
    return dict(
        objective="binary",n_estimators=180,learning_rate=0.04,num_leaves=15,
        min_child_samples=80,subsample=0.9,colsample_bytree=0.9,
        reg_lambda=2.0,random_state=seed,n_jobs=2,verbosity=-1,
    )

def chronological_split(races):
    q=races.sort_values(["race_date","race_id"],kind="stable").reset_index(drop=True)
    cut=max(1,int(len(q)*0.8))
    return q.iloc[:cut].copy(),q.iloc[cut:].copy()

def fit_gate_pair(train_races,label,seed):
    fit,cal=chronological_split(train_races)
    if fit[label].nunique()<2 or cal[label].nunique()<2:
        raise SystemExit(f"gate label degenerate: {label}")
    mfit=lgb.LGBMClassifier(**gate_params(seed))
    mfit.fit(fit[list(RACE_FEATURES)],fit[label].astype(int))
    raw_cal=np.asarray(mfit.booster_.predict(cal[list(RACE_FEATURES)],raw_score=True),dtype=float)
    temp,cal_loss=choose_binary_temperature(cal[label].astype(int).to_numpy(),raw_cal)
    q_cal=sigmoid(raw_cal/temp)
    mall=lgb.LGBMClassifier(**gate_params(seed+1000))
    mall.fit(train_races[list(RACE_FEATURES)],train_races[label].astype(int))
    return mall,temp,cal,q_cal,cal_loss

def gate_predict(model,temp,races):
    raw=np.asarray(model.booster_.predict(races[list(RACE_FEATURES)],raw_score=True),dtype=float)
    return sigmoid(raw/float(temp))

def normalize(d):
    vals={k:max(float(v),EPS) for k,v in d.items()}
    s=sum(vals.values())
    return {k:v/s for k,v in vals.items()}

def base_probability(sub):
    return normalize({int(r.horse_number):float(r.mean_probability) for r in sub.itertuples(index=False)})

def scenario_probabilities(sub,collapse_factor,outsider_boost):
    p0=base_probability(sub)
    top1=int(sub.loc[sub["consensus_rank"]==1,"horse_number"].iloc[0])
    pc=dict(p0)
    pc[top1]*=float(collapse_factor)
    pc=normalize(pc)

    eligible=sub[(sub["consensus_rank"]>5)&(sub["outs_weighted"]>0)]
    po=dict(p0)
    if len(eligible):
        mx=max(float(eligible["outs_weighted"].max()),1.0)
        for r in eligible.itertuples(index=False):
            num=int(r.horse_number)
            factor=1.0+float(outsider_boost)*(float(r.outs_weighted)/mx)
            po[num]*=factor
    po=normalize(po)

    pb=dict(pc)
    if len(eligible):
        mx=max(float(eligible["outs_weighted"].max()),1.0)
        for r in eligible.itertuples(index=False):
            num=int(r.horse_number)
            factor=1.0+float(outsider_boost)*(float(r.outs_weighted)/mx)
            pb[num]*=factor
    pb=normalize(pb)
    return {"NORMAL":p0,"COLLAPSE":pc,"OUTSIDER":po,"BOTH":pb}

def weights(qc,qo):
    qc=min(max(float(qc),0.0),1.0)
    qo=min(max(float(qo),0.0),1.0)
    return {
        "NORMAL":(1-qc)*(1-qo),
        "COLLAPSE":qc*(1-qo),
        "OUTSIDER":(1-qc)*qo,
        "BOTH":qc*qo,
    }

def mix_ticket_probability(bet,nums,worlds,w):
    return float(sum(float(w[name])*ticket_probability(bet,nums,p) for name,p in worlds.items()))

def actual_top3_nums(sub):
    q=sub[sub["target_top3"].fillna(0).astype(float)>0.5]
    nums=tuple(sorted(int(x) for x in q["horse_number"]))
    return nums if len(nums)==3 else None

def actual_winner_num(sub):
    q=sub[sub["target_win"].fillna(0).astype(float)>0.5]
    return int(q["horse_number"].iloc[0]) if len(q)==1 else None

def make_race_lookup(df,years):
    return {
        (int(y),str(rid)):g.copy().reset_index(drop=True)
        for (y,rid),g in df[df["year"].isin(years)].groupby(["year","race_id"],sort=False)
    }

def parameter_loss(cal_races,cal_df,qc_map,qo_map,collapse_factor,outsider_boost):
    lookup=make_race_lookup(cal_df,set(cal_races["year"].astype(int)))
    losses=[]
    for r in cal_races.itertuples(index=False):
        key=(int(r.year),str(r.race_id))
        sub=lookup.get(key)
        if sub is None: continue
        actual=actual_top3_nums(sub)
        if actual is None: continue
        worlds=scenario_probabilities(sub,collapse_factor,outsider_boost)
        w=weights(qc_map[str(r.race_id)],qo_map[str(r.race_id)])
        p=mix_ticket_probability("TRIO",actual,worlds,w)
        losses.append(-math.log(clip01(p)))
    return float(np.mean(losses)) if losses else float("inf"),len(losses)

def select_scenario_params(cal_races,cal_df,qc,qo):
    qc_map={str(rid):float(v) for rid,v in zip(cal_races["race_id"],qc)}
    qo_map={str(rid):float(v) for rid,v in zip(cal_races["race_id"],qo)}
    rows=[]
    for cf in COLLAPSE_FACTORS:
        for ob in OUTSIDER_BOOSTS:
            loss,n=parameter_loss(cal_races,cal_df,qc_map,qo_map,cf,ob)
            rows.append({"collapse_factor":cf,"outsider_max_boost":ob,"calibration_trio_nll":loss,"races":n})
    rows.sort(key=lambda x:(x["calibration_trio_nll"],x["collapse_factor"],x["outsider_max_boost"]))
    return rows[0],rows

def auc_or_none(y,p):
    yy=np.asarray(y,dtype=int)
    if len(np.unique(yy))<2: return None
    return float(roc_auc_score(yy,p))

def gate_metric_rows(test_year,races,qc,qo):
    out=[]
    for name,label,p in (
        ("KING1_COLLAPSE_GATE","collapse_label",qc),
        ("OUTSIDER_RISE_GATE","outsider_label",qo),
    ):
        y=races[label].astype(int).to_numpy()
        out.append({
            "test_year":test_year,"gate":name,"races":len(races),
            "actual_rate_pct":100.0*float(np.mean(y)),
            "mean_predicted_pct":100.0*float(np.mean(p)),
            "auc":auc_or_none(y,p),"brier":binary_brier(y,p),"log_loss":binary_logloss(y,p),
        })
    return out

def evaluate_fold(test_year,test_df,test_races,qc,qo,collapse_factor,outsider_boost,backfill_root):
    qc_map={str(rid):float(v) for rid,v in zip(test_races["race_id"],qc)}
    qo_map={str(rid):float(v) for rid,v in zip(test_races["race_id"],qo)}
    lookup=make_race_lookup(test_df,{test_year})
    race_models={}
    race_date={}
    distribution_rows=[]
    max_world_invariant=0.0
    max_weight_error=0.0

    for r in test_races.itertuples(index=False):
        rid=str(r.race_id); key=(test_year,rid)
        sub=lookup[key]
        worlds=scenario_probabilities(sub,collapse_factor,outsider_boost)
        w=weights(qc_map[rid],qo_map[rid])
        max_weight_error=max(max_weight_error,abs(sum(w.values())-1.0))
        for p in worlds.values():
            max_world_invariant=max(max_world_invariant,probability_invariant(p))
        pbase=worlds["NORMAL"]
        pmix={n:sum(w[name]*worlds[name][n] for name in worlds) for n in pbase}
        actual_win=actual_winner_num(sub)
        actual_trio=actual_top3_nums(sub)
        win_nll_base=-math.log(clip01(pbase[actual_win])) if actual_win is not None else None
        win_nll_mix=-math.log(clip01(pmix[actual_win])) if actual_win is not None else None
        trio_nll_base=None; trio_nll_mix=None
        if actual_trio is not None:
            pb=ticket_probability("TRIO",actual_trio,pbase)
            pm=mix_ticket_probability("TRIO",actual_trio,worlds,w)
            trio_nll_base=-math.log(clip01(pb))
            trio_nll_mix=-math.log(clip01(pm))
        distribution_rows.append({
            "test_year":test_year,"race_id":rid,"race_date":str(r.race_date)[:10],
            "collapse_label":int(r.collapse_label),"outsider_label":int(r.outsider_label),
            "q_collapse":qc_map[rid],"q_outsider":qo_map[rid],
            "w_normal":w["NORMAL"],"w_collapse":w["COLLAPSE"],"w_outsider":w["OUTSIDER"],"w_both":w["BOTH"],
            "winner_nll_base":win_nll_base,"winner_nll_mix":win_nll_mix,
            "trio_nll_base":trio_nll_base,"trio_nll_mix":trio_nll_mix,
        })
        race_models[rid]=(pbase,worlds,w)
        race_date[rid]=str(r.race_date)[:10]

    cal={(model,bet):CalAgg() for model in ("BASE","MIXTURE") for bet in BET_TYPES}
    root=Path(backfill_root)
    bydate=defaultdict(list)
    for rid,d in race_date.items(): bydate[d].append(rid)
    for di,date in enumerate(sorted(bydate),1):
        wanted=set(bydate[date])
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        oddsday=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in wanted:
            pack=day.get(rid); oddsrec=oddsday.get(rid)
            if pack is None or oddsrec is None:
                raise SystemExit(f"missing test market pack year={test_year} race={rid}")
            payouts,present=payout_map(pack)
            pbase,worlds,w=race_models[rid]
            for bet,nums,price in iter_decoded_odds(oddsrec):
                if bet not in BET_TYPES or bet not in present: continue
                y=1 if float(payouts.get((bet,nums),0.0))>0 else 0
                pb=ticket_probability(bet,nums,pbase)
                pm=mix_ticket_probability(bet,nums,worlds,w)
                cal[("BASE",bet)].add(pb,y)
                cal[("MIXTURE",bet)].add(pm,y)
        if di%25==0:
            print(f"MIXTURE_TICKET_PROGRESS year={test_year} dates={di}/{len(bydate)}",flush=True)

    ticket_rows=[]
    for model in ("BASE","MIXTURE"):
        for bet in BET_TYPES:
            ticket_rows.append({"test_year":test_year,"model":model,"bet_type":bet,**cal[(model,bet)].result()})

    dr=pd.DataFrame(distribution_rows)
    win=dr.dropna(subset=["winner_nll_base","winner_nll_mix"])
    trio=dr.dropna(subset=["trio_nll_base","trio_nll_mix"])
    headline={
        "test_year":test_year,
        "winner_races":len(win),
        "winner_log_loss_base":float(win["winner_nll_base"].mean()) if len(win) else None,
        "winner_log_loss_mix":float(win["winner_nll_mix"].mean()) if len(win) else None,
        "winner_log_loss_delta_mix_minus_base":float((win["winner_nll_mix"]-win["winner_nll_base"]).mean()) if len(win) else None,
        "trio_races":len(trio),
        "actual_trio_nll_base":float(trio["trio_nll_base"].mean()) if len(trio) else None,
        "actual_trio_nll_mix":float(trio["trio_nll_mix"].mean()) if len(trio) else None,
        "actual_trio_nll_delta_mix_minus_base":float((trio["trio_nll_mix"]-trio["trio_nll_base"]).mean()) if len(trio) else None,
        "probability_invariant_max_error":max(max_world_invariant,max_weight_error),
    }

    diagnostics=[]
    masks={
        "ALL":np.ones(len(dr),dtype=bool),
        "KING1_COLLAPSE_TRUE":dr["collapse_label"].to_numpy(dtype=int)==1,
        "OUTSIDER_RISE_TRUE":dr["outsider_label"].to_numpy(dtype=int)==1,
        "BOTH_TRUE":(dr["collapse_label"].to_numpy(dtype=int)==1)&(dr["outsider_label"].to_numpy(dtype=int)==1),
        "NEITHER_TRUE":(dr["collapse_label"].to_numpy(dtype=int)==0)&(dr["outsider_label"].to_numpy(dtype=int)==0),
    }
    for name,mask in masks.items():
        q=dr.loc[mask].dropna(subset=["trio_nll_base","trio_nll_mix"])
        diagnostics.append({
            "test_year":test_year,"subset":name,"races":len(q),
            "trio_nll_base":float(q["trio_nll_base"].mean()) if len(q) else None,
            "trio_nll_mix":float(q["trio_nll_mix"].mean()) if len(q) else None,
            "delta_mix_minus_base":float((q["trio_nll_mix"]-q["trio_nll_base"]).mean()) if len(q) else None,
        })
    return headline,ticket_rows,diagnostics,distribution_rows

def main():
    a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L2_RACE_DISTRIBUTION_MIXTURE_V2":
        raise SystemExit("wrong contract")
    if contract["cost_policy"]["github_standard_cpu_only"] is not True or contract["cost_policy"]["gpu"] is not False:
        raise SystemExit("cost guard drift")
    if contract["data_policy"]["locked_years"]!=[2026]:
        raise SystemExit("2026 lock drift")

    manifest,df=load_dataset(a.dataset_dir)
    df["year"]=df["year"].astype(int)
    df["race_id"]=df["race_id"].astype(str)
    for c in manifest["feature_columns"]:
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
    stats,race_candidates=load_outsider_ballots(parse_year_paths(a.ballots_year))
    df=attach_outsider(df,stats,race_candidates)
    races=build_race_frame(df)
    if 2026 in set(races["year"]): raise SystemExit("2026 sealed")

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    fold_rows=[]; gate_rows=[]; param_rows=[]; ticket_rows=[]; diagnostic_rows=[]
    all_distribution=[]

    for fi,(test_year,train_years) in enumerate(FOLDS,1):
        train_r=races[races["year"].isin(train_years)].dropna(subset=["collapse_label","outsider_label"]).copy()
        test_r=races[races["year"]==test_year].dropna(subset=["collapse_label","outsider_label"]).copy().reset_index(drop=True)
        train_df=df[df["year"].isin(train_years)].copy()
        test_df=df[df["year"]==test_year].copy()

        mc,tc,cal_c,qc_cal,cal_loss_c=fit_gate_pair(train_r,"collapse_label",20263000+fi)
        mo,to,cal_o,qo_cal,cal_loss_o=fit_gate_pair(train_r,"outsider_label",20264000+fi)
        if list(cal_c["race_id"])!=list(cal_o["race_id"]):
            raise SystemExit("internal calibration race drift")
        cal_r=cal_c.reset_index(drop=True)
        cal_df=train_df[train_df["race_id"].isin(set(cal_r["race_id"].astype(str)))].copy()

        selected,grid=select_scenario_params(cal_r,cal_df,qc_cal,qo_cal)
        for row in grid:
            param_rows.append({"test_year":test_year,**row,"selected":int(
                row["collapse_factor"]==selected["collapse_factor"] and row["outsider_max_boost"]==selected["outsider_max_boost"]
            )})

        qc=gate_predict(mc,tc,test_r)
        qo=gate_predict(mo,to,test_r)
        gate_rows.extend(gate_metric_rows(test_year,test_r,qc,qo))

        headline,tickets,diagnostics,distribution=evaluate_fold(
            test_year,test_df,test_r,qc,qo,
            selected["collapse_factor"],selected["outsider_max_boost"],a.backfill_root
        )
        fold_rows.append({
            **headline,
            "train_years":"|".join(map(str,train_years)),
            "collapse_gate_temperature":tc,
            "outsider_gate_temperature":to,
            "collapse_gate_internal_cal_logloss":cal_loss_c,
            "outsider_gate_internal_cal_logloss":cal_loss_o,
            "selected_collapse_factor":selected["collapse_factor"],
            "selected_outsider_max_boost":selected["outsider_max_boost"],
            "selected_internal_trio_nll":selected["calibration_trio_nll"],
        })
        ticket_rows.extend(tickets)
        diagnostic_rows.extend(diagnostics)
        all_distribution.extend(distribution)
        print("L2_RACE_DISTRIBUTION_MIXTURE_FOLD_DONE "+json.dumps({
            "test_year":test_year,
            "collapse_factor":selected["collapse_factor"],
            "outsider_boost":selected["outsider_max_boost"],
            "winner_delta":headline["winner_log_loss_delta_mix_minus_base"],
            "trio_delta":headline["actual_trio_nll_delta_mix_minus_base"],
        },separators=(",",":")),flush=True)

    write_csv(out/"fold-metrics.csv",fold_rows)
    write_csv(out/"gate-metrics.csv",gate_rows)
    write_csv(out/"scenario-parameter-grid.csv",param_rows)
    write_csv(out/"ticket-calibration-ab.csv",ticket_rows)
    write_csv(out/"scenario-diagnostics.csv",diagnostic_rows)

    # Keep only compact per-race scenario weights, not ticket-level probability tables.
    with gzip.open(out/"race-world-weights.csv.gz","wt",newline="",encoding="utf-8") as fh:
        fields=[
            "test_year","race_id","race_date","collapse_label","outsider_label",
            "q_collapse","q_outsider","w_normal","w_collapse","w_outsider","w_both",
            "winner_nll_base","winner_nll_mix","trio_nll_base","trio_nll_mix"
        ]
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()
        for row in all_distribution: w.writerow(row)

    summary={
        "contract":"L2_RACE_DISTRIBUTION_MIXTURE_V2_RESULT",
        "architecture":"MIXTURE_OF_PLACKETT_LUCE_WORLDS",
        "normal_world_source":"Frozen L1.7 mean_probability, race-normalized; no learned replacement horse-strength model.",
        "worlds":["NORMAL","KING1_COLLAPSE","OUTSIDER_RISE","BOTH"],
        "folds":[x[0] for x in FOLDS],
        "fold_metrics":fold_rows,
        "gate_metrics":gate_rows,
        "ticket_calibration":ticket_rows,
        "diagnostics":diagnostic_rows,
        "selection_metric":"Prior-only internal calibration actual-TRIO negative log likelihood.",
        "roi_optimized":False,
        "market_price_used":False,
        "odds_value_ignored":True,
        "payout_use":"ticket hit labels only",
        "2026_locked":True,
        "elapsed_seconds":time.perf_counter()-started,
        "promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Race Distribution Mixture V2\n\n"
        "NORMAL = frozen L1.7 mean-probability PL world. V2 mixes KING1-collapse, Outsider-rise, and BOTH worlds using walk-forward pre-race gates. "
        "No betting rules, ROI optimization, or staking are performed. 2026 is sealed.\n",
        encoding="utf-8"
    )
    max_inv=max((float(x["probability_invariant_max_error"]) for x in fold_rows),default=0.0)
    if max_inv>1e-8:
        raise SystemExit(f"mixture probability invariant failed max={max_inv}")
    print("===== FOLD METRICS ====="); print((out/"fold-metrics.csv").read_text())
    print("===== GATE METRICS ====="); print((out/"gate-metrics.csv").read_text())
    print("===== TICKET CALIBRATION ====="); print((out/"ticket-calibration-ab.csv").read_text())
    print("===== SCENARIO DIAGNOSTICS ====="); print((out/"scenario-diagnostics.csv").read_text())
    print("L2_RACE_DISTRIBUTION_MIXTURE_V2_READY")

if __name__=="__main__":
    main()
