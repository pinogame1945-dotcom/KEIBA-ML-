#!/usr/bin/env python3
import argparse,csv,gzip,json,math
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import load_day,load_odds_day,iter_decoded_odds,payout_map,horse_number_map
from build_l2_l17_fullfield_dataset_v1 import load_l17

YEARS=(2023,2024,2025)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--l17-year",action="append",required=True)
    p.add_argument("--race-dates",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--v4-diagnostics",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_year_paths(items):
    out={}
    for s in items:
        y,p=s.split(":",1);out[int(y)]=p
    if set(out)!=set(YEARS):raise SystemExit(f"year paths mismatch: {sorted(out)}")
    return out

def load_labels(path):
    rows=[]
    with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            y=int(r["year"])
            if y not in YEARS:continue
            kd=float(r["king_logloss_delta_vs_market"]);od=float(r["outsider_logloss_delta_vs_market"])
            rows.append({
                "year":y,"race_id":str(r["race_id"]),
                "both_fail":int(kd>=0 and od>=0)
            })
    z=pd.DataFrame(rows)
    if z.duplicated(["year","race_id"]).any():raise SystemExit("duplicate labels")
    return z

def competition_ranks(items):
    # items: [(horse_no, odds)] lower is better.
    q=sorted(items,key=lambda x:(x[1],x[0]))
    out={}
    last=None;rank=0
    for i,(no,odds) in enumerate(q,1):
        if last is None or odds!=last:rank=i
        out[int(no)]=rank;last=odds
    return out

def summarize(df,name):
    x=df[df["subset"]==name].copy()
    n=len(x)
    def pct(mask): return float(mask.mean()*100.0) if n else None
    return {
        "subset":name,"races":n,
        "market_fav1_in_podium_pct":pct(x["market_top1_in_podium"]==1),
        "market_top3_zero_in_podium_pct":pct(x["market_top3_count"]==0),
        "market_top3_count_mean":float(x["market_top3_count"].mean()),
        "market_top5_count_mean":float(x["market_top5_count"].mean()),
        "market_top5_all3_pct":pct(x["market_top5_count"]==3),
        "market_top5_zero_pct":pct(x["market_top5_count"]==0),
        "market_two_top5_plus_8plus_pct":pct((x["market_top5_count"]>=2)&(x["market_max_rank"]>=8)),
        "market_max_rank_ge8_pct":pct(x["market_max_rank"]>=8),
        "market_max_rank_ge10_pct":pct(x["market_max_rank"]>=10),
        "market_max_rank_ge12_pct":pct(x["market_max_rank"]>=12),
        "market_actual_rank_mean":float(x[["market_r1","market_r2","market_r3"]].to_numpy().mean()),
        "king1_in_podium_pct":pct(x["king_top1_in_podium"]==1),
        "king_top3_count_mean":float(x["king_top3_count"].mean()),
        "king_top6_count_mean":float(x["king_top6_count"].mean()),
        "king_top6_all3_pct":pct(x["king_top6_count"]==3),
        "king_top6_zero_pct":pct(x["king_top6_count"]==0),
        "king_max_rank_ge8_pct":pct(x["king_max_rank"]>=8),
        "king_max_rank_ge10_pct":pct(x["king_max_rank"]>=10),
        "market_fav1_and_king1_both_miss_pct":pct((x["market_top1_in_podium"]==0)&(x["king_top1_in_podium"]==0)),
    }

def main():
    a=parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    paths=parse_year_paths(a.l17_year)
    l17={y:load_l17(paths[y],y) for y in YEARS}
    dates=pd.read_csv(a.race_dates,dtype={"race_id":str,"race_date":str})
    dates["year"]=pd.to_numeric(dates["year"],errors="raise").astype(int)
    dates=dates[dates["year"].isin(YEARS)]
    dmap={(int(r.year),str(r.race_id)):str(r.race_date)[:10] for r in dates.itertuples(index=False)}
    labels=load_labels(a.v4_diagnostics)
    label_map={(int(r.year),str(r.race_id)):int(r.both_fail) for r in labels.itertuples(index=False)}
    root=Path(a.backfill_root)
    rows=[]
    by_date=defaultdict(list)
    for key in label_map:
        by_date[dmap[key]].append(key)

    for di,date in enumerate(sorted(by_date),1):
        keys=by_date[date];wanted={rid for _,rid in keys}
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        oddsday=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for year,rid in keys:
            pack=day.get(rid);orec=oddsday.get(rid)
            if pack is None or orec is None:raise SystemExit(f"missing pack/odds {rid}")
            pay,present=payout_map(pack)
            trios=[nums for (bet,nums),v in pay.items() if bet=="TRIO"]
            if len(trios)!=1:raise SystemExit(f"expected one trio y={year} race={rid} got={len(trios)}")
            actual=set(map(int,trios[0]))
            win_items=[]
            for bet,nums,price in iter_decoded_odds(orec):
                if bet=="WIN":win_items.append((int(nums[0]),float(price)))
            if len(win_items)<3:raise SystemExit(f"incomplete WIN market {rid}")
            mrank=competition_ranks(win_items)
            if not actual.issubset(mrank):raise SystemExit(f"actual horse missing win market {rid}")

            rec=l17[year][rid]
            kh={int(h["horse_number"]):int(h["consensus_rank"]) for h in rec["horses"]}
            if not actual.issubset(kh):raise SystemExit(f"actual horse missing L17 {rid}")
            mrs=sorted(mrank[n] for n in actual);krs=sorted(kh[n] for n in actual)
            market_top3=sum(r<=3 for r in mrs);market_top5=sum(r<=5 for r in mrs)
            king_top3=sum(r<=3 for r in krs);king_top6=sum(r<=6 for r in krs)
            rows.append({
                "year":year,"race_id":rid,"race_date":date,
                "subset":"BOTH_FAIL" if label_map[(year,rid)] else "NOT_BOTH_FAIL",
                "market_r1":mrs[0],"market_r2":mrs[1],"market_r3":mrs[2],
                "market_max_rank":max(mrs),
                "market_top1_in_podium":int(any(r==1 for r in mrs)),
                "market_top3_count":market_top3,"market_top5_count":market_top5,
                "king_r1":krs[0],"king_r2":krs[1],"king_r3":krs[2],
                "king_max_rank":max(krs),
                "king_top1_in_podium":int(any(r==1 for r in krs)),
                "king_top3_count":king_top3,"king_top6_count":king_top6,
            })
        if di%100==0:print(f"CHAOS_QUALITY_PROGRESS {di}/{len(by_date)}",flush=True)

    df=pd.DataFrame(rows)
    both=df[df["subset"]=="BOTH_FAIL"];other=df[df["subset"]=="NOT_BOTH_FAIL"]
    allsum=summarize(pd.concat([
        df.assign(subset="ALL"),df
    ],ignore_index=True),"ALL")
    bsum=summarize(df,"BOTH_FAIL");osum=summarize(df,"NOT_BOTH_FAIL")
    summary={
        "contract":"L2_TRIO_CHAOS_QUALITY_AUDIT",
        "races":len(df),"both_fail_races":len(both),
        "summary":{"ALL":allsum,"BOTH_FAIL":bsum,"NOT_BOTH_FAIL":osum},
        "interpretation_helpers":{
            "fav1_drop_lift":bsum["market_fav1_in_podium_pct"]/allsum["market_fav1_in_podium_pct"] if allsum["market_fav1_in_podium_pct"] else None,
            "top3_total_collapse_lift":bsum["market_top3_zero_in_podium_pct"]/allsum["market_top3_zero_in_podium_pct"] if allsum["market_top3_zero_in_podium_pct"] else None,
            "one_or_more_8plus_lift":bsum["market_max_rank_ge8_pct"]/allsum["market_max_rank_ge8_pct"] if allsum["market_max_rank_ge8_pct"] else None,
            "two_top5_plus_bomb_lift":bsum["market_two_top5_plus_8plus_pct"]/allsum["market_two_top5_plus_8plus_pct"] if allsum["market_two_top5_plus_8plus_pct"] else None,
            "king_top6_zero_lift":bsum["king_top6_zero_pct"]/allsum["king_top6_zero_pct"] if allsum["king_top6_zero_pct"] else None
        },
        "note":"Post-outcome descriptive audit only. These quantities must not be used as pre-race router labels/features without a new strict walk-forward design."
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    df.to_csv(out/"race-patterns.csv.gz",index=False,compression="gzip")
    pd.DataFrame([allsum,bsum,osum]).to_csv(out/"summary-table.csv",index=False)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":
    main()
