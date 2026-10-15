#!/usr/bin/env python3
"""OOS 2023-25 ALL-race EXACTA vs historic QUINELLA: same anchors, counterpart lists,
fixed 100-yen tickets, exact payouts from the pinned BACKFILL source.

Directions: FORWARD fixes each chosen axis horse in 1st; REVERSE fixes it in 2nd;
BOTH covers both. No result-dependent direction choices.
"""
import argparse
import concurrent.futures
import json
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
from run_all_race_flow_ticket_audit_v1 import (
    YEARS, TOTAL_RACES, RACES_YEAR, SOURCES, AXES, PARTNERS,
    get_inputs, flow_tickets, assert_ok
)
from build_l2_market_helpers_v1 import load_day, payout_map, horse_number_map
BACKFILL_SHA="0d4af461b77f0060dd9c789dd2da646026fada47"
DIRECTIONS=("FORWARD_FIRST_FIXED","REVERSE_SECOND_FIXED","BOTH_DIRECTIONS")
PAYOUT_KINDS=("EXACTA","QUINELLA")
SCOPES=("ALL","BOTH_WIN_MISS")

def anchored_pair_list(axis,market,value,ordered,k):
    """Retain directed anchor roles before forming/examining the two orientations."""
    def from_axis(anchor):
        return [(anchor,h) for h in ordered if h!=anchor][:k]
    if axis=="FAVORITE_AXIS":
        return from_axis(market)
    if axis=="VALUE_AXIS":
        return from_axis(value)
    if axis=="BOTH_AXES_UNION":
        return from_axis(market)+from_axis(value)
    if axis=="TWO_AXES_WITH_SINGLE_FALLBACK":
        if market!=value:
            return [(market,value)]
        return from_axis(market)
    raise ValueError(f"unexpected axis {axis}")

def exacta_tickets(pairs,direction):
    if direction=="FORWARD_FIRST_FIXED":
        return set(pairs)
    if direction=="REVERSE_SECOND_FIXED":
        return {(b,a) for (a,b) in pairs}
    if direction=="BOTH_DIRECTIONS":
        return set(pairs)|{(b,a) for (a,b) in pairs}
    raise ValueError(direction)

def settle_year(year, all_df, pick_ids, root):
    sub=all_df[all_df.year.eq(year)].copy()
    sub["_race_date"]=sub["_race_date"].astype(str).str[:10]
    by_date=sub[["_race_id","_race_date"]].drop_duplicates().groupby("_race_date")
    group=sub.groupby("_race_id",sort=False)
    metrics=defaultdict(Counter)
    race_count=0
    skipped_packs=[]
    missing_payout=[]
    for date, dates in by_date:
        rids=list(dates["_race_id"])
        packs=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",set(rids))
        for rid in rids:
            pack=packs.get(rid)
            if pack is None:
                skipped_packs.append(rid)
                continue
            payout,present=payout_map(pack)
            if not set(PAYOUT_KINDS).issubset(present):
                missing_payout.append((rid,sorted(present)))
                continue
            horse_to_no=horse_number_map(pack)
            g=group.get_group(rid)
            horses=set(g["_horse_id"].astype(str))
            assert_ok(horses.issubset(horse_to_no.keys()),
                      f"{year} {rid}: missing horse numbers")
            market=pick_ids["MARKET_FAVORITE"][rid]
            value=pick_ids["ANCHOR_FULL"][rid]
            assert_ok(market in horses and value in horses,
                      f"missing favorite/value {rid}")
            order_market=g.sort_values(
                ["market_rank","final_win_odds","_horse_id"],
                ascending=[True,True,True])["_horse_id"].astype(str).tolist()
            order_ai=g.sort_values(
                ["P__ANCHOR_FULL","market_rank","_horse_id"],
                ascending=[False,True,True])["_horse_id"].astype(str).tolist()
            win_ids=set(g.loc[g["target_win"].eq(1),"_horse_id"].astype(str))
            both_miss=not bool(win_ids.intersection({market,value}))
            scopes=("ALL","BOTH_WIN_MISS") if both_miss else ("ALL",)
            race_count+=1
            payout_exacta={bet:v for (kind,bet),v in payout.items() if kind=="EXACTA"}
            payout_quinella={bet:v for (kind,bet),v in payout.items() if kind=="QUINELLA"}
            assert_ok(bool(payout_exacta) and bool(payout_quinella),
                      f"{year} {rid}: missing true payout for exacta/quinella")
            for source in SOURCES:
                ranked=order_market if source=="MARKET" else order_ai
                for axis in AXES:
                    for k in PARTNERS:
                        anchored=anchored_pair_list(axis,market,value,ranked,k)
                        expected=flow_tickets("QUINELLA",axis,market,value,ranked,k)
                        actual={tuple(sorted(ab)) for ab in anchored}
                        assert_ok(expected==actual,
                                  f"{rid}: anchored-pair orientation changed quinella legs")
                        # Same tickets and stakes as the prior QUINELLA run, audited once per race.
                        pair_numbers={tuple(sorted(horse_to_no[h] for h in ticket)) for ticket in expected}
                        assert_ok(len(pair_numbers)==len(expected),
                                  f"{rid}: number mapping collision")
                        q_hits=[payout_quinella[t] for t in pair_numbers if t in payout_quinella]
                        for sc in scopes:
                            c=metrics[(str(year),sc,source,axis,k,"QUINELLA_BASELINE")]
                            c["races"]+=1
                            c["tickets"]+=len(pair_numbers)
                            c["stake_yen"]+=100*len(pair_numbers)
                            c["return_yen"]+=sum(q_hits)
                            c["hit_races"]+=int(bool(q_hits))
                            c["hit_tickets"]+=len(q_hits)
                        for direction in DIRECTIONS:
                            candidates=exacta_tickets(anchored,direction)
                            if direction=="BOTH_DIRECTIONS":
                                assert_ok(len(candidates)==2*len(expected),
                                          f"{rid}: both orientations not complete")
                            assert_ok(all(a!=b for a,b in candidates),
                                      f"{rid}: self pair")
                            no_tickets={(horse_to_no[a],horse_to_no[b]) for a,b in candidates}
                            assert_ok(len(no_tickets)==len(candidates),
                                      f"{rid}: duplicate horse number")
                            e_hits=[payout_exacta[t] for t in no_tickets if t in payout_exacta]
                            for sc in scopes:
                                c=metrics[(str(year),sc,source,axis,k,direction)]
                                c["races"]+=1
                                c["tickets"]+=len(no_tickets)
                                c["stake_yen"]+=100*len(no_tickets)
                                c["return_yen"]+=sum(e_hits)
                                c["hit_races"]+=int(bool(e_hits))
                                c["hit_tickets"]+=len(e_hits)
    assert_ok(race_count==RACES_YEAR,
        f"wrong {year} race coverage {race_count} missing_packs={skipped_packs[:10]} "
        f"missing_payout={missing_payout[:10]}")
    return metrics

def consolidate(per_year):
    total=defaultdict(Counter)
    for m in per_year:
        for key,ct in m.items():
            total[key].update(ct)
            total[("2023-2025",)+key[1:]].update(ct)
    records=[]
    for (year,scope,source,axis,k,direction),stats in total.items():
        n=stats["races"]
        expect=(TOTAL_RACES if scope=="ALL" else 5951) if year=="2023-2025" else (
            RACES_YEAR if scope=="ALL" else {2023:1962,2024:1959,2025:2030}[int(year)])
        assert_ok(n==expect,f"coverage {year}/{scope}/{direction}: {n} expected={expect}")
        s=stats["stake_yen"]
        ret=stats["return_yen"]
        records.append({
          "year":year,"scope":scope,
          "bet_type":"QUINELLA" if direction=="QUINELLA_BASELINE" else "EXACTA",
          "direction":direction,"partner_order":source,"anchor_rule":axis,
          "other_horses_k":k,
          "races":n,"tickets":stats["tickets"],
          "tickets_per_race":round(stats["tickets"]/n,6),
          "stake_yen":s,"return_yen":round(ret,2),
          "profit_yen":round(ret-s,2),
          "roi_pct":round(100*ret/s,6) if s else None,
          "hit_races":stats["hit_races"],
          "hit_race_pct":round(100*stats["hit_races"]/n,6),
          "winning_tickets":stats["hit_tickets"],
        })
    return pd.DataFrame(records).sort_values(
        ["year","scope","partner_order","anchor_rule","other_horses_k","bet_type","direction"]
    ).reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--picks",required=True)
    ap.add_argument("--backfill-root",required=True)
    ap.add_argument("--quinella-baseline",required=True)
    ap.add_argument("--out-dir",required=True)
    args=ap.parse_args()
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    df,picks=get_inputs(args.predictions,args.picks)
    dates=pd.read_csv(args.predictions,compression="infer",
                      usecols=["year","_race_id","_horse_id","_race_date"],
                      dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    assert_ok(not dates["_race_date"].isna().any(),"missing race dates")
    df=df.merge(dates,on=["year","_race_id","_horse_id"],validate="one_to_one")
    assert_ok(df["_race_id"].nunique()==TOTAL_RACES,"missing test races")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        chunks=list(pool.map(
            lambda y:settle_year(y,df,picks,Path(args.backfill_root)), YEARS))
    report=consolidate(chunks)
    base=pd.read_csv(args.quinella_baseline)
    comparison=report[report["direction"].eq("QUINELLA_BASELINE")].merge(
        base[base["bet_type"].eq("QUINELLA")],
        on=["year","scope","partner_order","anchor_rule","other_horses_k"],
        how="left",validate="one_to_one",suffixes=("_recomputed","_previous"),
        indicator=True)
    assert_ok(comparison["_merge"].eq("both").all(),"old quinella comparator missing")
    for col in ("stake_yen","return_yen","hit_races","tickets"):
        assert_ok(comparison[col+"_recomputed"].eq(comparison[col+"_previous"]).all(),
                  f"QUINELLA BASELINE DRIFT in {col}")
    exact=report[report["bet_type"].eq("EXACTA")].copy()
    keys=["year","scope","partner_order","anchor_rule","other_horses_k"]
    q=report[report["direction"].eq("QUINELLA_BASELINE")][
       keys+["roi_pct","hit_races","stake_yen","return_yen"]].rename(
       columns={"roi_pct":"matched_quinella_roi_pct",
                "hit_races":"matched_quinella_hits",
                "stake_yen":"matched_quinella_stake_yen",
                "return_yen":"matched_quinella_return_yen"})
    exact=exact.merge(q,on=keys,how="left",validate="many_to_one")
    exact["exacta_minus_quinella_roi_pp"]=(exact["roi_pct"]-
        exact["matched_quinella_roi_pct"]).round(6)
    exact.to_csv(out/"exacta-all-race-vs-matched-quinella.csv",index=False)
    report.to_csv(out/"all-exacta-directions-and-quinella.csv",index=False)
    # Exacta + original one AI-value single bet per race, never result-gated.
    both=exact[exact["scope"].eq("ALL")].copy()
    ai_ret={2023:307190,2024:285430,2025:309900,"2023-2025":902520}
    ai_stake={2023:345600,2024:345600,2025:345600,"2023-2025":1036800}
    both["ai_win_return_proxy_yen"]=both["year"].map(
        lambda x:ai_ret[x if x=="2023-2025" else int(x)])
    both["ai_win_stake_yen"]=both["year"].map(
        lambda x:ai_stake[x if x=="2023-2025" else int(x)])
    both["combined_stake_yen"]=both["stake_yen"]+both["ai_win_stake_yen"]
    both["combined_return_yen"]=both["return_yen"]+both["ai_win_return_proxy_yen"]
    both["combined_roi_pct"]=(100*both["combined_return_yen"]/
                              both["combined_stake_yen"]).round(6)
    both.to_csv(out/"ai-win-plus-exacta-roi.csv",index=False)
    pooled=exact[(exact["year"].eq("2023-2025")) &
                 (exact["scope"].eq("ALL"))].copy()
    top=pooled.sort_values("roi_pct",ascending=False)
    summary={
      "contract":"ALL_RACE_EXACTA_VS_QUINELLA_V1",
      "historical_years":list(YEARS),"races":TOTAL_RACES,
      "pinned_backfill_sha":BACKFILL_SHA,
      "matched_quinella_regression":"PASSED: full 128 all/slice yearly baselines identical to prior run",
      "templates_excluding_control":int(len(pooled)),
      "pooled_profitable_template_count":int(pooled["roi_pct"].gt(100).sum()),
      "pooled_best":top.iloc[0][[
          "direction","partner_order","anchor_rule","other_horses_k",
          "roi_pct","hit_race_pct","stake_yen","profit_yen"]].to_dict(),
      "directions":{
         "FORWARD_FIRST_FIXED":"chosen axis first, each of its k partners second",
         "REVERSE_SECOND_FIXED":"chosen axis second, each of its k partners first",
         "BOTH_DIRECTIONS":"both ordered bets, deduplicated",
         "QUINELLA_BASELINE":"the previous undirected bet on identical horse pairs"
      },
      "same_selection_rules":True,"no_skip":True,"2026_locked":True,
      "payout_source":"pinned full race packs exact settled 100 yen JRA dividends, no odds proxy for exacta",
      "ai_win_hedge":"combined analysis includes old final-odds WIN payout proxy",
      "both_win_miss_slice":"retrospective diagnostic, not observable at bet time",
      "final_odds_selection":"ANCHOR_FULL based on closing odds, so no live operability proven",
      "multiple_testing":"Best of 96 templates is exploratory until walk-forward bet-rule validation"
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("EXACTA_V1_POOLED_RECAP_START",flush=True)
    for axis in ("TWO_AXES_WITH_SINGLE_FALLBACK","BOTH_AXES_UNION",
                 "VALUE_AXIS","FAVORITE_AXIS"):
        subset=pooled[(pooled["partner_order"].eq("MARKET")) &
                      (pooled["anchor_rule"].eq(axis)) &
                      (pooled["other_horses_k"].eq(3))]
        print(subset.to_csv(index=False),flush=True)
    print("EXACTA_V1_POOLED_RECAP_END",flush=True)
    print("ALL_RACE_EXACTA_V1_COMPLETE",flush=True)

if __name__=="__main__":
    main()
