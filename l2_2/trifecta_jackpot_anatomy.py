#!/usr/bin/env python3
"""L2.2 JACKPOT-ANATOMY V1: describe actual >=JPY10,000 trifecta races, NO betting.

Pinned official 2023-2025 race packs for settled payouts; archived OOS
per-horse final win-market ranks for winners' popularity. 2026 untouched.
This run NEVER selects a bet or learns a model. Outcome-derived strata
(e.g., first-place horse popularity) are labelled post-race-only.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from build_l2_market_helpers_v1 import load_day, horse_number_map
from run_l2_value_anchor_bet_frontier_v1 import payouts
from run_value_anchor_official_positions_v1 import positions

YEARS=(2023,2024,2025)
EXPECTED_PER_YEAR=3456
EXPECTED_TOTAL=10368
PINNED_SHA="0d4af461b77f0060dd9c789dd2da646026fada47"
JACKPOT_MIN=10000

def check(ok, reason):
    if not ok:
        raise ValueError(reason)

def amount_band(yen):
    if yen < 10000: return "UNDER_10000"
    if yen < 30000: return "10000_29999"
    if yen < 100000: return "30000_99999"
    if yen < 1000000: return "100000_999999"
    return "1000000_PLUS"

def popularity_tier(rank):
    rank=int(rank)
    if rank==1:return "1"
    if rank<=3:return "2_3"
    if rank<=5:return "4_5"
    if rank<=9:return "6_9"
    return "10_PLUS"

def finish_tier(rank):
    if rank is None:return "NO_OFFICIAL_POSITION"
    if rank==1:return "FIRST"
    if rank==2:return "SECOND"
    if rank==3:return "THIRD"
    return "FOURTH_PLUS"

def field_band(n):
    if n<=10:return "UP_TO_10"
    if n<=14:return "11_14"
    if n<=16:return "15_16"
    return "17_PLUS"

def distance_band(meters):
    try:
        n=int(float(meters))
    except (TypeError,ValueError):
        return "UNKNOWN"
    if n<=0:return "UNKNOWN"
    if n<1400:return "UNDER_1400"
    if n<1800:return "1400_1799"
    if n<2200:return "1800_2199"
    return "2200_PLUS"

def surface_group(race):
    for k in ("surface","surface_type","track_type","track_surface","course_type"):
        v=str(race.get(k) or "").strip()
        if not v:continue
        upper=v.upper()
        if "芝" in v or "TURF" in upper:return "TURF"
        if "ダ" in v or "DIRT" in upper:return "DIRT"
        if "障" in v or "JUMP" in upper:return "JUMP"
    return "UNKNOWN"

def official_ranks(group,pack,race_id):
    num_map=horse_number_map(pack)
    group_hids=set(group["_horse_id"].astype(str))
    check(group_hids.issubset(num_map),f"{race_id}: runner missing horse number")
    check(len({num_map[hid] for hid in group_hids})==len(group_hids),
          f"{race_id}: duplicated horse number")
    m={}
    for horse_id,raw in group[["_horse_id","market_rank"]].itertuples(index=False,name=None):
        hid=str(horse_id)
        rank=int(raw)
        # Archived market ranks may retain gaps / pre-scratch positions.
        # Do not fabricate renumbered popularity after withdrawals.
        check(1<=rank<=30,f"{race_id}: invalid market rank {rank}")
        m[num_map[hid]]=rank
    return m,num_map

def manifest_files(root,dates):
    payload=json.loads((root/"data/manifest.json").read_text(encoding="utf-8"))
    manifest=payload["days"]
    paths={}
    for day in dates:
        item=manifest.get(day)
        check(item and item.get("status")=="SUCCESS",
              f"no SUCCESS manifest record {day}")
        name=item.get("file")
        check(isinstance(name,str) and name.startswith("data/daily/")
              and name.endswith(".jsonl.gz") and ".." not in name,
              f"metadata file path drift {day} {name}")
        paths[day]=name
    return paths

def prepare(args):
    df=pd.read_csv(args.predictions,compression="infer",usecols=["year","_race_date"])
    check(set(map(int,df.year.unique()))==set(YEARS),"historic years changed or 2026 not sealed")
    dates=sorted(set(df._race_date.astype(str).str[:10]))
    check(300<=len(dates)<=350,"date count outside OOS contract")
    files={"data/manifest.json"}
    files.update(manifest_files(Path(args.backfill_root),dates).values())
    Path(args.sparse_paths).write_text(
        "".join("/"+f+"\n" for f in sorted(files)),encoding="utf-8")
    print("JACKPOT_MANIFEST_RESOLVED",len(dates),"days",len(files),"files",flush=True)

def analyze_year(year,frame,root_text):
    root=Path(root_text)
    group=frame.groupby("_race_id",sort=False)
    meta=frame[["_race_id","_race_date"]].drop_duplicates()
    check(len(meta)==EXPECTED_PER_YEAR,f"{year}: race count changed")
    day_races=meta.assign(day=meta._race_date.astype(str).str[:10]).groupby("day")["_race_id"].apply(list)
    paths=manifest_files(root,list(day_races.index))
    rows=[]
    audit=Counter()
    for di,(day,race_ids) in enumerate(day_races.items(),1):
        all_races=load_day(root/paths[day],set(race_ids))
        for rid in sorted(race_ids):
            pack=all_races.get(rid)
            check(pack is not None, f"missing pack {rid}")
            ranks,hid_to_num=official_ranks(group.get_group(rid),pack,rid)
            no_to_hid={n:h for h,n in hid_to_num.items()}
            race=pack.get("race") or {}
            # Reuse exact payout parser with fail-closed seven-way coverage audit.
            refunded=payouts(pack,rid)
            paid=[(tuple(nums),int(round(value))) for (kind,nums),value in refunded.items()
                  if kind=="TRIFECTA"]
            check(bool(paid),f"{rid}: no valid official TRIFECTA refund")
            check(len({x[0] for x in paid})==len(paid),f"{rid}: repeated paid physical trifecta")
            chosen,payout=max(paid,key=lambda x:(x[1],x[0]))
            check(len(chosen)==3 and len(set(chosen))==3,f"{rid}: invalid paid combination")
            check(all(n in ranks for n in chosen),f"{rid}: payout horse missing from market {chosen}")
            winner_ranks=[ranks[n] for n in chosen]
            official=positions(pack)
            favs=[no_to_hid[n] for n,rank in ranks.items() if rank==1]
            # With scratched/withdrawn runners the archived rank-1 horse may
            # not be in active horse-level predictions. Audit, never re-rank.
            if not favs:audit["archived_rank1_absent"]+=1
            fav_finish=min((official.get(h) for h in favs if official.get(h) is not None),
                           default=None)
            if len(favs)>1:audit["tied_favorites"]+=1
            count_top3=sum(x<=3 for x in winner_ranks)
            count_6plus=sum(x>=6 for x in winner_ranks)
            count_10plus=sum(x>=10 for x in winner_ranks)
            check(count_top3 in (0,1,2,3),"top3 count invalid")
            check(count_6plus in (0,1,2,3),"outsider count invalid")
            if len(paid)>1:audit["multi_payout_races"]+=1
            if len(paid)>1 and len({val>=JACKPOT_MIN for _,val in paid})>1:
                audit["mixed_jackpot_refund_races"]+=1
            if any(x not in (1,2,3) for x in
                   [official.get(no_to_hid.get(n,"")) for n in chosen]):
                audit["paid_finish_not_1_2_3"]+=1
            field=len(group.get_group(rid))
            dist=race.get("distance_m")
            r=dict(year=year,race_id=rid,race_date=day,
                max_trifecta_refund_yen=payout,
                payout_band=amount_band(payout),
                jackpot=int(payout>=JACKPOT_MIN),
                above_30000=int(payout>=30000),
                above_100000=int(payout>=100000),
                above_1000000=int(payout>=1000000),
                paid_trifecta_outcomes=len(paid),
                horse_numbers="-".join(map(str,chosen)),
                winner_popularity_ranks="-".join(map(str,winner_ranks)),
                winner_rank_1=winner_ranks[0],
                winner_rank_2=winner_ranks[1],
                winner_rank_3=winner_ranks[2],
                winner_rank_1_tier=popularity_tier(winner_ranks[0]),
                winner_rank_2_tier=popularity_tier(winner_ranks[1]),
                winner_rank_3_tier=popularity_tier(winner_ranks[2]),
                num_market_top3=count_top3,
                num_market_6plus=count_6plus,
                num_market_10plus=count_10plus,
                market_favorite_in_trifecta=int(1 in winner_ranks),
                market_favorite_finish=(finish_tier(fav_finish) if favs
                                        else "ARCHIVED_RANK1_ABSENT"),
                winner_contains_top3=int(count_top3>0),
                field_size=field,
                field_size_band=field_band(field),
                surface=surface_group(race),
                distance_m=dist if dist is not None else "",
                distance_band=distance_band(dist))
            rows.append(r)
            audit["races"]+=1
            audit["jackpot_races"]+=r["jackpot"]
        if di%55==0:
            print("JACKPOT_YEAR_SCAN",year,"days",di,"races",audit["races"],flush=True)
    check(len(rows)==EXPECTED_PER_YEAR,f"{year}: race rows missing")
    check(len({row["race_id"] for row in rows})==EXPECTED_PER_YEAR,f"{year}: duplicated race")
    return rows,dict(audit)

def grouped_rows(df,feature,segments=("ALL","JACKPOT","NOT_JACKPOT","100000_PLUS")):
    answer=[]
    for (period,block) in [("2023-2025",df)]+[(str(y),df[df.year==y]) for y in YEARS]:
        for scope in segments:
            if scope=="ALL":s=block
            elif scope=="JACKPOT":s=block[block.jackpot==1]
            elif scope=="NOT_JACKPOT":s=block[block.jackpot==0]
            else:s=block[block.above_100000==1]
            total=len(s)
            for name,count in s[feature].value_counts(dropna=False).items():
                answer.append(dict(period=period,scope=scope,
                    feature=feature,value=str(name),n=int(count),
                    scope_n=total,scope_pct=round(count*100/total,6) if total else None))
    return answer

def predictive_rates(df):
    # These features do not depend on the event's official finishing position.
    features=("field_size_band","distance_band","surface")
    answers=[]
    for period,group in [("2023-2025",df)]+[(str(y),df[df.year==y]) for y in YEARS]:
        total=len(group)
        baseline=group.jackpot.sum()/total
        for feature in features:
            for val,block in group.groupby(feature,dropna=False):
                n=len(block);h=int(block.jackpot.sum())
                answers.append(dict(period=period,feature=feature,value=val,
                  races=n,jackpot_races=h,jackpot_rate_pct=100*h/n,
                  overall_rate_pct=100*baseline,rate_ratio=(h/n)/baseline if baseline else None))
    return answers

def run(args):
    df=pd.read_csv(args.predictions,compression="infer",
       usecols=["year","_race_id","_race_date","_horse_id","market_rank","final_win_odds"],
       dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    df.year=df.year.astype(int)
    check(set(df.year.unique())==set(YEARS),"must never load 2026")
    check(df._race_id.nunique()==EXPECTED_TOTAL and
          not df.duplicated(["_race_id","_horse_id"]).any(),"horse-level OOS prediction drift")
    check(df.market_rank.notna().all(),"market ranks absent")
    source=str(args.backfill_root)
    frames={y:df[df.year.eq(y)].copy() for y in YEARS}
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        futs={y:pool.submit(analyze_year,y,frames[y],source) for y in YEARS}
        chunks={y:futs[y].result() for y in YEARS}
    all_rows=[r for y in YEARS for r in chunks[y][0]]
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    races=pd.DataFrame(all_rows)
    check(len(races)==EXPECTED_TOTAL and races.race_id.is_unique,"race output drift")
    races.to_csv(out/"all_races_outcome_anatomy.csv.gz",index=False,compression="gzip")
    features=("payout_band","winner_rank_1_tier","winner_rank_2_tier",
              "winner_rank_3_tier","num_market_top3","num_market_6plus",
              "num_market_10plus","market_favorite_in_trifecta",
              "market_favorite_finish","winner_popularity_ranks")
    distribution=[item for feature in features for item in grouped_rows(races,feature)]
    pd.DataFrame(distribution).to_csv(out/"outcome_anatomy_frequency.csv",index=False)
    pd.DataFrame(predictive_rates(races)).to_csv(out/"pre_race_context_jackpot_rates.csv",index=False)
    sample=races.sort_values(["max_trifecta_refund_yen","race_id"],
                            ascending=[False,True]).head(40)
    sample.to_csv(out/"largest_trifecta_example_races.csv",index=False)
    summary=dict(contract="KEIBA_L2_2_TRIFECTA_JACKPOT_ANATOMY_V1",
      archive_sha=PINNED_SHA,
      subject="Actual official winning TRIFECTA payout >= JPY 10,000 per 100-yen ticket",
      years=list(YEARS),year_2026_sealed=True,
      all_races=EXPECTED_TOTAL,one_representative_payout_per_race="HIGHEST_AMOUNT",
      non_unique_payouts="Multiple dead-heat combos flagged; highest payout used as representative",
      conclusion_type="DESCRIPTIVE ONLY, NO MODEL OR BUYING POLICY",
      caveat="Popularity from archived FINAL win odds, not guaranteed visible before cutoff",
      scope_year={str(y):chunks[y][1] for y in YEARS},
      jackpot_by_year={str(y):dict(races=EXPECTED_PER_YEAR,
               hits=int((races.year.eq(y)&races.jackpot.eq(1)).sum()),
               payout_100k_plus=int((races.year.eq(y)&races.above_100000.eq(1)).sum()))
               for y in YEARS},
      jackpot_total=int(races.jackpot.sum()),
      above_100000_total=int(races.above_100000.sum()),
      above_1000000_total=int(races.above_1000000.sum()),
      caveat_postrace_features="rank of actual 1st/2nd/3rd horses is postrace and cannot screen a race in advance")
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print("JACKPOT_STRUCTURAL_COUNTS",json.dumps(summary["jackpot_by_year"]),flush=True)
    print("JACKPOT_TOTAL",summary["jackpot_total"],"OF",EXPECTED_TOTAL,flush=True)

def selftest():
    check(amount_band(9999)=="UNDER_10000","boundary")
    check(amount_band(10000)=="10000_29999","100x inclusive")
    check(amount_band(100000)=="100000_999999","100k inclusive")
    check(field_band(18)=="17_PLUS","runner band")
    check(finish_tier(None)=="NO_OFFICIAL_POSITION","no outcome")
    check(distance_band(1200)=="UNDER_1400","distance bin")
    check(popularity_tier(10)=="10_PLUS","tier")
    check(surface_group({"surface":"芝"})=="TURF","surface")
    print("JACKPOT_STRUCTURE_SELFTEST_OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--prepare",action="store_true")
    p.add_argument("--selftest",action="store_true")
    for name in ("predictions","backfill-root","sparse-paths","out-dir"):
        p.add_argument("--"+name)
    a=p.parse_args()
    if a.selftest:selftest()
    elif a.prepare:prepare(a)
    else:run(a)
