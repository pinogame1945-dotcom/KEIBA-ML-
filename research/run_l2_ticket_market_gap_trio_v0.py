#!/usr/bin/env python3
import argparse,gzip,itertools,json,math
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from build_l2_bet_kings_dataset_v1 import decode_odds,payout_map,horse_number_map,canonical_numbers
from build_l2_l17_fullfield_dataset_v1 import load_l17

YEARS=(2021,2022,2023,2024,2025); TEST=(2022,2023,2024,2025); TOPKS=(1,3,5,10,20,50,100)
ID_COLS={"year","race_id","race_date","trio_horse_ids","trio_numbers","hit","return_yen_per100","odds","market_aware_score"}

def args():
    p=argparse.ArgumentParser()
    for y in YEARS:p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True);p.add_argument("--out-dir",required=True)
    return p.parse_args()

def f(v,d=0.0):
    try:
        x=float(v);return x if math.isfinite(x) else d
    except (TypeError,ValueError):return d

def readgz(p):
    with gzip.open(p,"rt",encoding="utf-8") as h:
        for line in h:
            if line.strip():yield json.loads(line)

def add_h(prefix,h,r):
    for k,d in [("consensus_rank",99),("mean_rank",99),("rank_std",0),("top3_support",0),("top6_support",0),("mean_probability",0),("probability_std",0)]:
        r[prefix+k]=f(h.get(k),d)

def make_row(y,rid,date,pack,rec,hs,odd,ret,experts):
    hs=sorted(hs,key=lambda h:(int(h["consensus_rank"]),str(h["horse_id"])))
    ranks=[int(h["consensus_rank"]) for h in hs]; probs=[f(h.get("mean_probability")) for h in hs]
    race=pack.get("race") or {}; nums=sorted(int(h["horse_number"]) for h in hs)
    r={"year":y,"race_id":rid,"race_date":date,"trio_horse_ids":"|".join(sorted(str(h["horse_id"]) for h in hs)),
       "trio_numbers":"-".join(map(str,nums)),"hit":int(ret>0),"return_yen_per100":float(ret),"odds":float(odd),
       "race_month":f(date[5:7]),"distance_m":f(race.get("distance_m")),"field_size":int(rec["field_size"]),
       "surface_turf":int(str(race.get("surface") or "").upper()=="TURF"),"surface_dirt":int(str(race.get("surface") or "").upper()=="DIRT")}
    for p,h in zip(("a_","b_","c_"),hs):add_h(p,h,r)
    r.update({"trio_rank_sum":sum(ranks),"trio_rank_spread":max(ranks)-min(ranks),"trio_best_rank":min(ranks),"trio_mid_rank":ranks[1],"trio_worst_rank":max(ranks),
              "trio_top3_count":sum(x<=3 for x in ranks),"trio_top6_count":sum(x<=6 for x in ranks),"trio_top10_count":sum(x<=10 for x in ranks),
              "trio_prob_sum":sum(probs),"trio_prob_product":probs[0]*probs[1]*probs[2],"trio_prob_min":min(probs),"trio_prob_spread":max(probs)-min(probs)})
    sums=[];worsts=[];all6=two6=all10=0
    for name in experts:
        er=[f(((h.get("experts") or {}).get(name) or {}).get("rank"),99) for h in hs]
        sums.append(sum(er));worsts.append(max(er));all6+=int(all(x<=6 for x in er));two6+=int(sum(x<=6 for x in er)>=2);all10+=int(all(x<=10 for x in er))
    r.update({"expert_rank_sum_mean":float(np.mean(sums)),"expert_rank_sum_std":float(np.std(sums)),"expert_rank_sum_max":max(sums),
              "expert_worst_mean":float(np.mean(worsts)),"expert_worst_std":float(np.std(worsts)),"expert_worst_max":max(worsts),
              "expert_all_top6_count":all6,"expert_two_top6_count":two6,"expert_all_top10_count":all10})
    return r

def build(y,l17,root):
    if len(l17)!=3456:raise SystemExit(f"L1.7 coverage drift y={y} got={len(l17)}")
    experts=sorted({n for q in l17.values() for h in q["horses"] for n in (h.get("experts") or {})})
    if len(experts)!=7:raise SystemExit(f"expected seven experts y={y} got={experts}")
    rows=[];priced=0;root=Path(root)
    for dp in sorted((root/"data/daily").glob(f"{y}-*.jsonl.gz")):
        date=dp.name[:10];op=root/"data/odds/daily"/dp.name
        if not op.exists():continue
        odb={str(x.get("race_id") or ""):x for x in readgz(op)}
        for pack in readgz(dp):
            rid=str((pack.get("race") or {}).get("race_id") or "")
            if rid not in l17 or rid not in odb:continue
            rec=l17[rid]; hno=horse_number_map(pack); hs=sorted(rec["horses"],key=lambda x:int(x["consensus_rank"]))
            if len(hs)!=int(rec["field_size"]):raise SystemExit(f"field drift race={rid}")
            if any(str(h["horse_id"]) not in hno for h in hs):raise SystemExit(f"horse missing race={rid}")
            for h in hs:h["horse_number"]=hno[str(h["horse_id"])]
            om=decode_odds(odb[rid]);pm,present=payout_map(pack)
            if "TRIO" not in present:continue
            n=0
            for tri in itertools.combinations(hs,3):
                key=("TRIO",canonical_numbers("TRIO",[hno[str(h["horse_id"])] for h in tri]));odd=om.get(key)
                if odd is None:continue
                ret=float(pm.get(key,0.0));rows.append(make_row(y,rid,date,pack,rec,tri,odd,ret,experts));n+=1
            priced+=int(n>0)
    if not rows:raise SystemExit(f"no TRIO rows y={y}")
    d=pd.DataFrame(rows)
    for c in d.columns:
        if c in ID_COLS:continue
        if d[c].dtype.kind=="f":d[c]=d[c].astype("float32")
        elif d[c].dtype.kind in "iu":d[c]=pd.to_numeric(d[c],downcast="integer")
    print("TRIO_YEAR_READY "+json.dumps({"year":y,"priced_races":priced,"priced_trios":len(d),"positive_trios":int(d.hit.sum())},separators=(",",":")),flush=True)
    return d

def market(d):
    z=d.copy();z["market_q_norm"]=(1/z.odds.clip(lower=1e-9)).astype("float32");z["market_q_norm"]/=z.groupby(["year","race_id"]).market_q_norm.transform("sum")
    z["market_log_q"]=np.log(z.market_q_norm.clip(lower=1e-12)).astype("float32")
    o=z.sort_values(["year","race_id","market_q_norm","trio_rank_sum","trio_numbers"],ascending=[1,1,0,1,1]);z.loc[o.index,"market_rank"]=(o.groupby(["year","race_id"]).cumcount()+1).to_numpy();z["market_rank"]=z.market_rank.astype("int16")
    return z

def features(d):
    ban={"year","race_id","race_date","trio_horse_ids","trio_numbers","hit","return_yen_per100","odds","market_aware_score"}
    return [c for c in d.columns if c not in ban]

def train_predict(tr,te,cols,seed):
    good=tr.groupby(["year","race_id"]).hit.transform("sum")>0;tr=tr.loc[good].sort_values(["year","race_id","trio_numbers"]).reset_index(drop=True)
    t=te.sort_values(["year","race_id","trio_numbers"]).copy();t["_i"]=t.index;t=t.reset_index(drop=True)
    model=lgb.LGBMRanker(objective="lambdarank",metric="ndcg",n_estimators=260,learning_rate=.03,num_leaves=31,min_child_samples=120,
                         subsample=.9,colsample_bytree=.85,reg_lambda=5.,reg_alpha=.5,random_state=seed,n_jobs=2,deterministic=True,force_col_wise=True,verbosity=-1)
    model.fit(tr[cols].astype("float32"),tr.hit.astype("int8").to_numpy(),group=tr.groupby(["year","race_id"],sort=False).size().tolist())
    p=np.asarray(model.predict(t[cols].astype("float32")),dtype="float32");s=pd.Series(p,index=t._i.astype(int));return s.reindex(te.index).to_numpy(),pd.DataFrame({"feature":cols,"gain":model.feature_importances_})

def ranks(d):
    z=d.copy()
    for rc,sc in [("l17_rank_score","trio_prob_product"),("model_rank","market_aware_score")]:
        o=z.sort_values(["year","race_id",sc,"trio_rank_sum","trio_numbers"],ascending=[1,1,0,1,1]);z.loc[o.index,rc]=(o.groupby(["year","race_id"]).cumcount()+1).to_numpy();z[rc]=z[rc].astype("int16")
    z["rank_upgrade"]=(z.market_rank.astype("int32")-z.model_rank.astype("int32")).astype("int16");return z

def coverage(d,y):
    n=d.race_id.nunique();out=[]
    for name,c in [("MARKET","market_rank"),("L17_TRIO_PRODUCT","l17_rank_score"),("MARKET_AWARE_MODEL","model_rank")]:
        for k in TOPKS:
            h=d[(d[c]<=k)&(d.hit==1)].race_id.nunique();out.append({"year":y,"ranking":name,"top_k":k,"hit_races":h,"source_races":n,"coverage_pct":100*h/n})
    return out

def band(v):
    v=int(v)
    for hi,n in [(3,"1-3"),(5,"4-5"),(10,"6-10"),(20,"11-20"),(40,"21-40"),(80,"41-80")]:
        if v<=hi:return n
    return "81+"

def diagnostics(allp):
    w=allp[allp.hit==1].copy();w["market_rank_band"]=w.market_rank.map(band);w["model_rank_band"]=w.model_rank.map(band)
    delta=[];dis=[]
    for y in TEST:
        q=w[w.year==y];delta.append({"year":y,"winning_tickets":len(q),"hit_races":q.race_id.nunique(),"market_rank_median":float(q.market_rank.median()),"model_rank_median":float(q.model_rank.median()),"l17_rank_median":float(q.l17_rank_score.median()),"rank_upgrade_median":float(q.rank_upgrade.median()),"model_better_pct":100*float((q.model_rank<q.market_rank).mean())})
        z=allp[allp.year==y];n=z.race_id.nunique()
        for k in (5,10,20,50):
            m=set(z[(z.market_rank<=k)&(z.hit==1)].race_id);a=set(z[(z.model_rank<=k)&(z.hit==1)].race_id);l=set(z[(z.l17_rank_score<=k)&(z.hit==1)].race_id)
            dis.append({"year":y,"top_k":k,"source_races":n,"market_hit_races":len(m),"model_hit_races":len(a),"l17_hit_races":len(l),"model_only_vs_market":len(a-m),"market_only_vs_model":len(m-a),"model_and_market":len(a&m)})
    keep=["year","race_id","race_date","trio_horse_ids","trio_numbers","odds","return_yen_per100","market_rank","l17_rank_score","model_rank","rank_upgrade","market_q_norm","market_aware_score","trio_best_rank","trio_mid_rank","trio_worst_rank","trio_top3_count","trio_top6_count","trio_top10_count","trio_prob_product","expert_all_top6_count","expert_two_top6_count","field_size","distance_m","market_rank_band","model_rank_band"]
    return w[keep],delta,dis

def main():
    a=args();paths={y:getattr(a,f"l17_{y}") for y in YEARS};l17={y:load_l17(paths[y],y) for y in YEARS};frames={y:market(build(y,l17[y],a.backfill_root)) for y in YEARS};cols=features(frames[2021])
    preds={};folds=[];cov=[];imps=[]
    for y in TEST:
        tys=[t for t in YEARS if t<y];keep=["year","race_id","trio_numbers","hit"]+cols;tr=pd.concat([frames[t][keep] for t in tys],ignore_index=True);te=frames[y].copy().reset_index(drop=True)
        p,imp=train_predict(tr,te,cols,97000+y);te["market_aware_score"]=p;te=ranks(te);preds[y]=te;cov+=coverage(te,y);imp["test_year"]=y;imps.append(imp)
        folds.append({"test_year":y,"train_years":"|".join(map(str,tys)),"train_trios":len(tr),"test_trios":len(te),"source_races":te.race_id.nunique(),"positive_trios":int(te.hit.sum()),"feature_count":len(cols)})
    allp=pd.concat([preds[y] for y in TEST],ignore_index=True);w,delta,dis=diagnostics(allp);n=allp.race_id.nunique()
    for name,c in [("MARKET","market_rank"),("L17_TRIO_PRODUCT","l17_rank_score"),("MARKET_AWARE_MODEL","model_rank")]:
        for k in TOPKS:
            h=allp[(allp[c]<=k)&(allp.hit==1)].race_id.nunique();cov.append({"year":"ALL","ranking":name,"top_k":k,"hit_races":h,"source_races":n,"coverage_pct":100*h/n})
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);pd.DataFrame(folds).to_csv(out/"folds.csv",index=False);pd.DataFrame(cov).to_csv(out/"ranking-coverage.csv",index=False);pd.DataFrame(delta).to_csv(out/"winner-rank-delta-summary.csv",index=False);pd.DataFrame(dis).to_csv(out/"market-model-disagreement.csv",index=False);pd.concat(imps).to_csv(out/"feature-importance.csv",index=False)
    w.groupby(["year","market_rank_band","model_rank_band"]).size().reset_index(name="winning_tickets").to_csv(out/"winner-market-model-matrix.csv",index=False)
    w.groupby(["year","trio_top3_count","trio_top6_count","trio_top10_count"]).size().reset_index(name="winning_tickets").to_csv(out/"winner-l17-structure.csv",index=False)
    with gzip.open(out/"winning-trios.csv.gz","wt",encoding="utf-8",newline="") as h:w.to_csv(h,index=False)
    summary={"contract":"L2_TRIO_MARKET_GAP_V0_RESULT","bet_type":"TRIO","all_runners_considered":True,"law_search":False,"ticket_selection_policy":False,"l3_handoff":False,"raw_ev_multiplication":False,"market_probability":"normalized inverse TRIO odds within race","value_diagnostic":"market_rank - model_rank","walk_forward":True,"test_years":list(TEST),"2026_locked":True,"ranking_coverage":cov,"winner_delta_summary":delta,"disagreement":dis}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print("L2_TRIO_MARKET_GAP_V0_READY");print(json.dumps({"folds":folds,"winner_delta_summary":delta,"law_search":False},ensure_ascii=False,separators=(",",":")))
if __name__=="__main__":main()
