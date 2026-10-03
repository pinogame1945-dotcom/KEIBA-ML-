#!/usr/bin/env python3
import argparse,csv,gzip,json,math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from build_l2_bet_kings_dataset_v1 import load_day,load_odds_day,iter_decoded_odds,payout_map

YEARS=(2023,2024,2025)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--v4-diagnostics",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def norm(v):
    return "" if v is None else str(v).strip()

def load_labels(path):
    rows=[]
    with gzip.open(path,"rt",encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            y=int(r["year"])
            if y not in YEARS: continue
            kd=float(r["king_logloss_delta_vs_market"])
            od=float(r["outsider_logloss_delta_vs_market"])
            rows.append({
                "year":y,
                "race_id":str(r["race_id"]),
                "race_date":str(r["race_date"])[:10],
                "both_fail":int(kd>=0 and od>=0),
            })
    z=pd.DataFrame(rows)
    if z.duplicated(["year","race_id"]).any():
        raise SystemExit("duplicate V4 diagnostics")
    return z

def distance_bucket(m):
    try: x=int(float(m))
    except Exception: return "UNKNOWN"
    if x<=1200:return "<=1200"
    if x<=1600:return "1300-1600"
    if x<=2000:return "1700-2000"
    if x<=2400:return "2100-2400"
    return ">=2500"

def field_bucket(n):
    n=int(n)
    if n<=10:return "<=10"
    if n<=13:return "11-13"
    if n<=16:return "14-16"
    return "17-18"

def is_bad_going(s):
    s=norm(s)
    return int(any(tok in s for tok in ("稍重","重","不良","HEAVY","SOFT","YIELDING","MUDDY","SLOPPY")))

def is_wet_weather(s):
    s=norm(s)
    return int(any(tok in s for tok in ("雨","雪","RAIN","SNOW")))

def rate(mask):
    if len(mask)==0:return None
    return float(mask.mean()*100.0)

def condition_rows(df,col,min_n=50):
    out=[]
    base=float(df["both_fail"].mean())
    for val,g in df.groupby(col,dropna=False):
        if len(g)<min_n: continue
        r=float(g["both_fail"].mean())
        out.append({
            "dimension":col,
            "value":str(val),
            "races":len(g),
            "both_fail_races":int(g["both_fail"].sum()),
            "both_fail_pct":100*r,
            "lift_vs_all":r/base if base else None,
            "fav_miss_pct":rate(g["fav_miss"]==1),
            "dominant_fav_pct":rate(g["dominant_fav"]==1),
        })
    return out

def main():
    a=parse_args()
    labels=load_labels(a.v4_diagnostics)
    root=Path(a.backfill_root)
    by_date=defaultdict(list)
    for r in labels.itertuples(index=False):
        by_date[str(r.race_date)].append((int(r.year),str(r.race_id),int(r.both_fail)))

    rows=[]
    for di,date in enumerate(sorted(by_date),1):
        wanted={rid for _,rid,_ in by_date[date]}
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        odd=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for year,rid,both_fail in by_date[date]:
            pack=day.get(rid); orec=odd.get(rid)
            if pack is None or orec is None:
                raise SystemExit(f"missing pack/odds race={rid} date={date}")
            payouts,_=payout_map(pack)
            trios=[nums for (bet,nums),v in payouts.items() if bet=="TRIO"]
            if len(trios)!=1:
                raise SystemExit(f"expected one TRIO result race={rid} got={len(trios)}")
            actual=set(map(int,trios[0]))
            wins=[]
            for bet,nums,price in iter_decoded_odds(orec):
                if bet=="WIN":
                    wins.append((int(nums[0]),float(price)))
            if len(wins)<3:
                raise SystemExit(f"incomplete WIN odds race={rid}")
            wins.sort(key=lambda x:(x[1],x[0]))
            fav_no,fav_odds=wins[0]
            second_odds=wins[1][1]
            gap_ratio=second_odds/max(fav_odds,1e-12)
            fav_miss=int(fav_no not in actual)
            race=pack.get("race") or {}
            fs=len(race.get("entries") or []) if isinstance(race.get("entries"),list) else 0
            if not fs:
                fs=len(pack.get("entries") or [])
            if not fs:
                fs=len(wins)
            going=norm(race.get("track_condition"))
            weather=norm(race.get("weather"))
            surface=norm(race.get("surface"))
            venue=norm(race.get("venue_code") or race.get("venue") or rid[4:6])
            distance=race.get("distance_m")
            dominant=int(fav_odds<=2.0 and gap_ratio>=1.5)
            superdominant=int(fav_odds<=1.5)
            rows.append({
                "year":year,"race_id":rid,"race_date":date,"both_fail":both_fail,
                "field_size":int(fs),"field_bucket":field_bucket(fs),
                "surface":surface or "UNKNOWN","distance_m":distance,
                "distance_bucket":distance_bucket(distance),
                "weather":weather or "UNKNOWN","wet_weather":is_wet_weather(weather),
                "track_condition":going or "UNKNOWN","bad_going":is_bad_going(going),
                "venue":venue or "UNKNOWN",
                "fav_no":fav_no,"fav_odds":fav_odds,"second_odds":second_odds,
                "fav_gap_ratio":gap_ratio,"fav_miss":fav_miss,
                "fav_le_1_5":int(fav_odds<=1.5),
                "fav_le_2_0":int(fav_odds<=2.0),
                "fav_le_2_5":int(fav_odds<=2.5),
                "dominant_fav":dominant,"superdominant_fav":superdominant,
                "dominant_fav_miss":int(dominant and fav_miss),
                "superdominant_fav_miss":int(superdominant and fav_miss),
            })
        if di%100==0:
            print(f"CHAOS_CONTEXT_PROGRESS {di}/{len(by_date)}",flush=True)

    df=pd.DataFrame(rows)
    if len(df)!=len(labels):
        raise SystemExit(f"coverage drift rows={len(df)} labels={len(labels)}")
    base=float(df["both_fail"].mean())
    bf=df[df["both_fail"]==1]
    ok=df[df["both_fail"]==0]

    def subset_stats(g):
        return {
            "races":len(g),
            "fav_odds_mean":float(g["fav_odds"].mean()),
            "fav_odds_median":float(g["fav_odds"].median()),
            "fav_gap_ratio_mean":float(g["fav_gap_ratio"].mean()),
            "fav_miss_pct":rate(g["fav_miss"]==1),
            "fav_le_1_5_pct":rate(g["fav_le_1_5"]==1),
            "fav_le_2_0_pct":rate(g["fav_le_2_0"]==1),
            "dominant_fav_pct":rate(g["dominant_fav"]==1),
            "dominant_fav_miss_pct":rate(g["dominant_fav_miss"]==1),
            "superdominant_fav_miss_pct":rate(g["superdominant_fav_miss"]==1),
            "bad_going_pct":rate(g["bad_going"]==1),
            "wet_weather_pct":rate(g["wet_weather"]==1),
        }

    favorite_bins=[]
    for label,mask in [
        ("fav<=1.5",df["fav_odds"]<=1.5),
        ("1.5<fav<=2.0",(df["fav_odds"]>1.5)&(df["fav_odds"]<=2.0)),
        ("2.0<fav<=2.5",(df["fav_odds"]>2.0)&(df["fav_odds"]<=2.5)),
        ("fav>2.5",df["fav_odds"]>2.5),
        ("dominant: fav<=2.0 & gap>=1.5",df["dominant_fav"]==1),
        ("dominant miss",df["dominant_fav_miss"]==1),
        ("superdominant miss: fav<=1.5",df["superdominant_fav_miss"]==1),
    ]:
        g=df[mask]
        if len(g):
            rr=float(g["both_fail"].mean())
            favorite_bins.append({
                "bucket":label,"races":len(g),"both_fail_races":int(g["both_fail"].sum()),
                "both_fail_pct":100*rr,"lift_vs_all":rr/base,
                "fav_miss_pct":rate(g["fav_miss"]==1)
            })

    conditions=[]
    for col in ("track_condition","bad_going","weather","wet_weather","surface","distance_bucket","field_bucket","venue","year"):
        conditions.extend(condition_rows(df,col))

    conditions_sorted=sorted(conditions,key=lambda x:(-x["lift_vs_all"],-x["races"]))
    summary={
        "contract":"L2_TRIO_CHAOS_CONTEXT_AUDIT",
        "races":len(df),"both_fail_races":int(df["both_fail"].sum()),
        "both_fail_pct":100*base,
        "subsets":{
            "ALL":subset_stats(df),
            "BOTH_FAIL":subset_stats(bf),
            "NOT_BOTH_FAIL":subset_stats(ok),
        },
        "favorite_bins":favorite_bins,
        "top_context_lifts":conditions_sorted[:15],
        "bottom_context_lifts":sorted(conditions,key=lambda x:(x["lift_vs_all"],-x["races"]))[:15],
        "key_posthoc_rates":{
            "both_fail_given_fav_miss_pct":100*float(df.loc[df["fav_miss"]==1,"both_fail"].mean()),
            "both_fail_given_fav_hit_pct":100*float(df.loc[df["fav_miss"]==0,"both_fail"].mean()),
            "both_fail_given_bad_going_pct":100*float(df.loc[df["bad_going"]==1,"both_fail"].mean()) if (df["bad_going"]==1).any() else None,
            "both_fail_given_good_going_pct":100*float(df.loc[df["bad_going"]==0,"both_fail"].mean()) if (df["bad_going"]==0).any() else None,
            "both_fail_given_dominant_fav_miss_pct":100*float(df.loc[df["dominant_fav_miss"]==1,"both_fail"].mean()) if (df["dominant_fav_miss"]==1).any() else None,
            "both_fail_given_superdominant_fav_miss_pct":100*float(df.loc[df["superdominant_fav_miss"]==1,"both_fail"].mean()) if (df["superdominant_fav_miss"]==1).any() else None,
        },
        "note":"Post-outcome descriptive audit. Favorite miss is outcome-conditioned and cannot be a pre-race feature. Track/weather/dominance can be tested pre-race only in a separate strict walk-forward model."
    }
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    pd.DataFrame(conditions).to_csv(out/"condition-rates.csv",index=False)
    pd.DataFrame(favorite_bins).to_csv(out/"favorite-bins.csv",index=False)
    df.to_csv(out/"race-context.csv.gz",index=False,compression="gzip")
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":
    main()
