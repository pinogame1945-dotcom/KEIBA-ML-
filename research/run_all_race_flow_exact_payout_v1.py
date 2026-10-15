#!/usr/bin/env python3
"""Exact BACKFILL payout settlement for precommitted OOS flow tickets.

All 2023-2025 races are staked ex ante, independent of results.
Races with missing matching historical payout packs fail loudly; no silent skips.
100 JPY tickets priced by JRA race-pack payout_yen, NOT win odds as a substitute.
"""
import argparse
import concurrent.futures
import json
from collections import Counter,defaultdict
from pathlib import Path
import pandas as pd

from run_all_race_flow_ticket_audit_v1 import (
    YEARS, TOTAL_RACES, RACES_YEAR, SOURCES, AXES, PARTNERS,
    get_inputs, flow_tickets, assert_ok
)
from build_l2_market_helpers_v1 import load_day,payout_map,horse_number_map

BACKFILL_SHA="0d4af461b77f0060dd9c789dd2da646026fada47"

def make_year(y,data,picks,root):
    frame=data[data["year"].eq(y)]
    frame=frame.copy()
    frame["_race_date"]=frame["_race_date"].astype(str).str[:10]
    by_date=frame[["_race_id","_race_date"]].drop_duplicates().groupby("_race_date")
    keyed=frame.groupby("_race_id",sort=False)
    measures=defaultdict(Counter)
    misspack=[]
    missing_payout=[]
    race_count=0
    for date,sub in by_date:
        rids=list(sub["_race_id"])
        day_file=root/"data"/"daily"/f"{date}.jsonl.gz"
        packs=load_day(day_file,set(rids))
        for rid in rids:
            pack=packs.get(rid)
            if pack is None:
                misspack.append(rid)
                continue
            payout,present=payout_map(pack)
            if not {"QUINELLA","TRIO"}.issubset(present):
                missing_payout.append((rid,sorted(present)))
                continue
            numbers=horse_number_map(pack)
            group=keyed.get_group(rid)
            horse_ids=set(group["_horse_id"].astype(str))
            assert_ok(horse_ids.issubset(numbers.keys()),
                      f"horse_number incomplete race {rid}")
            fav=picks["MARKET_FAVORITE"][rid]
            ai=picks["ANCHOR_FULL"][rid]
            m=group.sort_values(["market_rank","final_win_odds","_horse_id"],
                                ascending=[True,True,True])["_horse_id"].astype(str).tolist()
            a=group.sort_values(["P__ANCHOR_FULL","market_rank","_horse_id"],
                                ascending=[False,True,True])["_horse_id"].astype(str).tolist()
            winner=set(group.loc[group["target_win"].eq(1),"_horse_id"].astype(str))
            both_win_miss=not(bool(winner.intersection({fav,ai})))
            scopes=("ALL","BOTH_WIN_MISS") if both_win_miss else ("ALL",)
            race_count+=1
            for kind in ("QUINELLA","TRIO"):
                # All winning ticket combos returned by authoritative BACKFILL.
                price={k[1]:v for k,v in payout.items() if k[0]==kind}
                assert_ok(bool(price),f"no {kind} winning payout for {rid}")
                for ordering in SOURCES:
                    order=m if ordering=="MARKET" else a
                    for axis in AXES:
                        for k in PARTNERS:
                            id_tickets=flow_tickets(kind,axis,fav,ai,order,k)
                            no_tickets={tuple(sorted(numbers[h] for h in ticket))
                                        for ticket in id_tickets}
                            assert_ok(len(id_tickets)==len(no_tickets),
                                      f"{rid} duplicated horse-number combo")
                            hits=[price.get(t,0.) for t in no_tickets if t in price]
                            ret=sum(hits)
                            for scope in scopes:
                                ct=measures[(str(y),scope,kind,ordering,axis,k)]
                                ct["races"]+=1
                                ct["tickets"]+=len(no_tickets)
                                ct["stake_yen"]+=100*len(no_tickets)
                                ct["return_yen"]+=ret
                                ct["hit_races"]+=int(bool(hits))
                                ct["hit_tickets"]+=len(hits)
                                ct["same_horse_races"]+=int(fav==ai)
    assert_ok(race_count==RACES_YEAR,
              f"{y}: only {race_count} / {RACES_YEAR} exact payout races. "
              f"pack_missing={misspack[:10]} payout_missing={missing_payout[:10]}")
    return measures

def report_rows(data):
    totals=defaultdict(Counter)
    for dat in data:
        for k,c in dat.items():
            totals[k].update(c)
            totals[("2023-2025",)+k[1:]].update(c)
    rows=[]
    for (year,scope,kind,ordering,axis,k),v in totals.items():
        n=v["races"];s=v["stake_yen"];r=v["return_yen"];w=v["hit_races"]
        assert_ok(n==(TOTAL_RACES if scope=="ALL" else 5951)
                  if year=="2023-2025" else n==(
                      RACES_YEAR if scope=="ALL" else
                      {2023:1962,2024:1959,2025:2030}[int(year)]),
                  f"denominator drift {year} {scope}")
        rows.append({
          "year":year,"scope":scope,"bet_type":kind,"partner_order":ordering,
          "anchor_rule":axis,"other_horses_k":k,
          "races":n,"tickets":v["tickets"],
          "tickets_per_race":round(v["tickets"]/n,5),
          "stake_yen":s,"return_yen":round(r,2),
          "profit_yen":round(r-s,2),"roi_pct":round(100*r/s,6),
          "hit_races":w,"hit_race_pct":round(100*w/n,6),
          "winning_tickets":v["hit_tickets"],
          "same_horse_races":v["same_horse_races"],
          "source_sha":BACKFILL_SHA,
        })
    return pd.DataFrame(rows).sort_values(
        ["year","scope","bet_type","partner_order","anchor_rule","other_horses_k"]
    ).reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--picks",required=True)
    ap.add_argument("--backfill-root",required=True)
    ap.add_argument("--out-dir",required=True)
    args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    frame,picks=get_inputs(args.predictions,args.picks)
    dates=pd.read_csv(args.predictions,compression="infer",usecols=[
        "year","_race_id","_horse_id","_race_date"],
        dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    assert_ok(not dates["_race_date"].isna().any(),"missing race_date")
    frame=frame.merge(dates,on=["year","_race_id","_horse_id"],
                      validate="one_to_one")
    assert_ok(frame["_race_id"].nunique()==TOTAL_RACES,"missing races")
    root=Path(args.backfill_root)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        results=list(ex.map(lambda y:make_year(y,frame,picks,root),YEARS))
    all_rows=report_rows(results)
    all_rows.to_csv(out/"exact-quinella-trio-flow-roi.csv",index=False)
    # Combine each bet schedule with always-on AI win ticket, same races,
    # using the original all-race V1 final-win-odds proxy (not exact WIN payout).
    comps=all_rows[all_rows["scope"].eq("ALL")].copy()
    ai_return={2023:307190,2024:285430,2025:309900,"2023-2025":902520}
    ai_stake={2023:345600,2024:345600,2025:345600,"2023-2025":1036800}
    def ky(x):return x if x=="2023-2025" else int(x)
    comps["ai_win_return_proxy_yen"]=comps["year"].map(lambda y:ai_return[ky(y)])
    comps["ai_win_stake_yen"]=comps["year"].map(lambda y:ai_stake[ky(y)])
    comps["combined_stake_yen"]=comps["stake_yen"]+comps["ai_win_stake_yen"]
    comps["combined_return_yen"]=comps["return_yen"]+comps["ai_win_return_proxy_yen"]
    comps["combined_roi_pct"]=(100*comps["combined_return_yen"]/comps["combined_stake_yen"]).round(6)
    comps.to_csv(out/"ai-win-plus-flow-combined-roi.csv",index=False)
    preferred=all_rows[(all_rows.year=="2023-2025") & (all_rows.scope=="ALL") &
                       (all_rows.partner_order=="MARKET") &
                       (all_rows.anchor_rule=="BOTH_AXES_UNION")]
    print("EXACT_PAYOUT_PREFERRED_MKT_PARTNERS_BOTH_AXES_UNION_ALL_RACES",flush=True)
    print(preferred.to_csv(index=False),flush=True)
    print("EXACT_PAYOUT_BOTH_WIN_MISS_DIAGNOSTIC_NOT_PURCHASABLE_POST_HOC",flush=True)
    print(all_rows[(all_rows.year=="2023-2025") &
                   (all_rows.scope=="BOTH_WIN_MISS") &
                   (all_rows.partner_order=="MARKET") &
                   (all_rows.anchor_rule=="BOTH_AXES_UNION")].to_csv(index=False),flush=True)
    pooled=all_rows[(all_rows.year=="2023-2025") & (all_rows.scope=="ALL")]
    positive=pooled[pooled.roi_pct.gt(100)]
    summary={
      "contract":"ALL_RACE_FLOW_EXACT_DIVIDEND_V1",
      "backfill_source_sha":BACKFILL_SHA,
      "historical_scope":"2023-2025, all 10368 races, no hindsight selection",
      "fixed_templates":len(pooled),
      "pooled_templates_gt_100pct":len(positive),
      "pooled_best_roi_pct":float(pooled.roi_pct.max()),
      "pooled_best_variant":pooled.sort_values("roi_pct",ascending=False).iloc[0][[
          "bet_type","partner_order","anchor_rule","other_horses_k","roi_pct",
          "hit_race_pct","stake_yen","profit_yen"]].to_dict(),
      "source":"BACKFILL pinned full race pack payouts, 100 JPY actual payout",
      "no_price_filter":True,"no_skip":True,"2026_locked":True,
      "lookahead_warning":"Final odds used for ANCHOR_FULL value bet; this is retrospective not bet-time replay.",
      "hindsight_warning":"BOTH_WIN_MISS analysis not actionable because actual winner is unknown pre-race.",
      "multiple_tests_warning":"Many fixed templates compared; best ROI is exploratory not out-of-sample strategy validation."
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("EXACT_DIVIDEND_COMPLETE",flush=True)

if __name__=="__main__":main()
