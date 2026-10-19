#!/usr/bin/env python3
"""Combine exactly 134 preselected 2023,2024,2025 individually profitable
EV-axis/raw-rank/gate strategies. Physical tickets are deduplicated by
(race_id, bet-kind, canonical horse-number combination) before staking/refunds.

No prediction/model training. Same odds-free gate predicates & ranks as
apply_odds_free_l2_gates_ev_top6_6870_v1.py. Pinned settlement packs only.
Run must exactly reproduce each of the 134 x 3 saved annual ticket/hit/refund
triples or it FAILS. This is retrospective in-sample, NOT an OOS result.
"""
import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from apply_odds_free_l2_gates_ev_top6_6870_v1 import flags_per_horse, GATE_INDEX
from run_l2_ai_market_rank_betfrontier_v1 import load_predictions
from run_l2_value_anchor_bet_frontier_v1 import payouts
from build_l2_market_helpers_v1 import load_day, horse_number_map
from run_all_race_flow_ticket_audit_v1 import YEARS, RACES_YEAR, TOTAL_RACES, assert_ok

FAMILY_NAMES={
    "単勝":"WIN","複勝":"PLACE","馬連":"QUINELLA","ワイド":"WIDE",
    "馬単・軸1着":"EXACTA_FIRST","馬単・軸2着":"EXACTA_SECOND",
    "三連複":"TRIO","三連単・軸1着":"TRIFECTA_FIRST",
    "三連単・軸2着":"TRIFECTA_SECOND",
    "三連単・軸3着":"TRIFECTA_THIRD",
}
PHYS_KIND={
    "EXACTA_FIRST":"EXACTA","EXACTA_SECOND":"EXACTA",
    "TRIFECTA_FIRST":"TRIFECTA",
    "TRIFECTA_SECOND":"TRIFECTA","TRIFECTA_THIRD":"TRIFECTA",
}
YEARS_TEXT=tuple(map(str,YEARS))


def load_selected(path):
    with open(path,"r",encoding="utf-8-sig",newline="") as fd:
        rows=list(csv.DictReader(fd))
    assert_ok(len(rows)==134,f"expected 134 selected settings, found {len(rows)}")
    strategies=[]
    for row in rows:
        family=FAMILY_NAMES[row["券種と軸位置"]]
        rank_numbers=list(map(int,re.findall(r"(\d+)位",row["純粋AI相手順位"])))
        if family in ("WIN","PLACE"):
            assert_ok(not rank_numbers,"bad single ranks")
            j=k=0
        elif family in ("QUINELLA","WIDE","EXACTA_FIRST","EXACTA_SECOND"):
            assert_ok(len(rank_numbers)==1,"bad double ranks")
            j=rank_numbers[0];k=0
        else:
            assert_ok(len(rank_numbers)==2 and rank_numbers[0]!=rank_numbers[1],
                      "bad triple ranks")
            j,k=rank_numbers
            if family=="TRIO":
                assert_ok(j<k,"non-canonical trio ranks")
        gate=row["削減条件"]
        assert_ok(gate in GATE_INDEX and gate!="ALL",f"not permitted gate {gate}")
        ev=int(row["期待値AI軸順位"])
        assert_ok(1<=ev<=6 and 0<=j<=18 and 0<=k<=18,"rank out of range")
        assert_ok(int(row["削減後黒字年度数"])==3,
                  "selection contains a non-stable setting")
        strategies.append((ev,family,j,k,gate))
    assert_ok(len(strategies)==len(set(strategies))==134,"duplicate selected setting")
    return sorted(strategies)


def historical(path, selected):
    required=set(selected)
    result={}
    with gzip.open(path,"rt",encoding="utf-8",newline="") as fd:
        for r in csv.DictReader(fd):
            if r["year"] not in YEARS_TEXT:continue
            key=(int(r["ev_axis_rank"]),r["family"],
                 int(r["raw_ai_rank_1"]),int(r["raw_ai_rank_2"]),r["gate"])
            if key not in required:continue
            combined=(r["year"],key)
            assert_ok(combined not in result,"duplicate reference strategy")
            n=int(r["after_tickets"]);h=int(r["after_hits"])
            ret=round(float(r["after_return_yen"]))
            assert_ok(ret>n*100 and n>0 and h>0,
                      f"selected setting not profitable in {r['year']}, {key}")
            result[combined]=(n,h,ret)
    assert_ok(len(result)==134*3,
              f"missing 134x3 reference stats, found {len(result)}")
    return result


def physical_ticket(family,anchor,raw,j,k):
    if family in ("WIN","PLACE"):
        return family,(anchor,)
    if j>len(raw):return None
    x=raw[j-1]
    if x==anchor:return None
    if family in ("QUINELLA","WIDE"):
        return family,tuple(sorted((anchor,x)))
    if family=="EXACTA_FIRST":
        return "EXACTA",(anchor,x)
    if family=="EXACTA_SECOND":
        return "EXACTA",(x,anchor)
    if k>len(raw):return None
    y=raw[k-1]
    if y==anchor or y==x:return None
    if family=="TRIO":
        return "TRIO",tuple(sorted((anchor,x,y)))
    if family=="TRIFECTA_FIRST":
        return "TRIFECTA",(anchor,x,y)
    if family=="TRIFECTA_SECOND":
        return "TRIFECTA",(x,anchor,y)
    if family=="TRIFECTA_THIRD":
        return "TRIFECTA",(x,y,anchor)
    raise ValueError("unknown family "+family)


def cell(c, available_races):
    n=c["tickets"];h=c["hit_tickets"];ret=c["return_yen"]
    return {
        "available_races":available_races,
        "bet_races":c["bet_races"],
        "hit_races":c["hit_races"],
        "purchased_tickets":n,
        "winning_tickets":h,
        "hit_rate_pct":round(100*h/n,6) if n else None,
        "stake_yen":100*n,
        "return_yen":ret,
        "profit_yen":ret-100*n,
        "roi_pct":round(ret/n,6) if n else None,
        "races_covered_pct":round(100*c["bet_races"]/available_races,6)
                             if available_races else None,
        "avg_tickets_per_bought_race":round(n/c["bet_races"],5)
                                      if c["bet_races"] else None,
    }


def run_year(year, data, root, selected, originals):
    yr=data[data.year.eq(year)]
    groups=yr.groupby("_race_id",sort=False)
    dates=yr[["_race_id","_race_date"]].drop_duplicates()
    assert_ok(len(dates)==RACES_YEAR and dates["_race_id"].is_unique,
              f"year data race count {year}")
    days=dates.groupby("_race_date")["_race_id"].apply(list)
    strategies_by_rank=defaultdict(list)
    for key in selected: strategies_by_rank[key[0]].append(key)
    results={k:Counter() for k in selected}
    union=Counter()
    by_type=defaultdict(Counter)
    appearances=Counter()
    source_gate=Counter()
    refund_order=[]
    selected_by_race=[]
    for day,rids in days.items():
        packs=load_day(root/"data"/"daily"/(str(day)+".jsonl.gz"),set(rids))
        for rid in rids:
            pack=packs.get(rid)
            assert_ok(pack is not None,f"missing settled race {rid}")
            g=groups.get_group(rid)
            horse_nums=horse_number_map(pack)
            all_ids=set(g["_horse_id"].astype(str))
            assert_ok(all_ids.issubset(horse_nums) and len(all_ids)==len(g),
                      f"missing/duplicated horse {rid}")
            raw=g.sort_values(["distill_ai_rank","market_rank","_horse_id"],
                    ascending=[True,True,True],kind="mergesort")[
                        "_horse_id"].astype(str).tolist()
            raw_nums=[horse_nums[h] for h in raw]
            assert_ok(len(raw_nums)==len(set(raw_nums)),"duplicate raw horse numbers")
            ev=g.assign(__v=g["P__ANCHOR_FULL"]*g["final_win_odds"])
            ev_order=ev.sort_values(["__v","P__ANCHOR_FULL","market_rank","_horse_id"],
                    ascending=[False,False,True,True],kind="mergesort")[
                        "_horse_id"].astype(str).tolist()
            assert_ok(set(ev_order)==set(raw),"EV horse set drift")
            flags=flags_per_horse(g,horse_nums)
            paid=payouts(pack,rid)
            all_ticket_sources=defaultdict(list)
            for er,axis_id in enumerate(ev_order[:6],1):
                if er not in strategies_by_rank:continue
                axis=horse_nums[axis_id]
                for setting in strategies_by_rank[er]:
                    _,fam,j,k,gate=setting
                    ticket=physical_ticket(fam,axis,raw_nums,j,k)
                    if ticket is None:continue
                    gi=GATE_INDEX[gate]
                    # Identical to original study: only companions
                    # are evaluated by gates; for singletons, the axis.
                    selected_partners=[n for n in ticket[1] if n!=axis]
                    keep=(bool(flags[axis][gi])
                          if fam in ("WIN","PLACE")
                          else all(bool(flags[n][gi]) for n in selected_partners))
                    if not keep:continue
                    amount=int(round(paid.get(ticket,0)))
                    s=results[setting]
                    s["tickets"]+=1
                    s["hit_tickets"]+=int(amount>0)
                    s["return_yen"]+=amount
                    all_ticket_sources[ticket].append(setting)
                    appearances[year]+=1
                    source_gate[(year,gate)]+=1
            race_hits=0
            for ticket,source in all_ticket_sources.items():
                amount=int(round(paid.get(ticket,0)))
                c=union
                c["tickets"]+=1
                c["hit_tickets"]+=int(amount>0)
                c["return_yen"]+=amount
                fam=ticket[0]
                f=by_type[fam]
                f["tickets"]+=1
                f["hit_tickets"]+=int(amount>0)
                f["return_yen"]+=amount
                refund_order.append(amount) if amount>0 else None
                race_hits+=int(amount>0)
            if all_ticket_sources:
                union["bet_races"]+=1
                for fam in set(t[0] for t in all_ticket_sources):
                    by_type[fam]["bet_races"]+=1
            if race_hits:
                union["hit_races"]+=1
                for fam in set(t[0] for t in all_ticket_sources if paid.get(t,0)>0):
                    by_type[fam]["hit_races"]+=1
            if all_ticket_sources:
                selected_by_race.append({
                    "year":year,"race_id":rid,
                    "unique_ticket_count":len(all_ticket_sources),
                    "before_dedup_ticket_count":sum(map(len,all_ticket_sources.values())),
                    "winning_tickets":race_hits,
                    "stake_yen":len(all_ticket_sources)*100,
                    "return_yen":sum(int(round(paid.get(t,0))) for t in all_ticket_sources),
                })
    assert_ok(len(results)==134,"chosen 134 are not all indexed")
    for key,c in results.items():
        actual=(c["tickets"],c["hit_tickets"],c["return_yen"])
        expected=originals[(str(year),key)]
        assert_ok(actual==expected,
              f"PRIOR_STUDY_PARITY_FAILURE {year} {key} actual={actual} expected={expected}")
    union_appeared=sum(c["tickets"] for c in results.values())
    assert_ok(union_appeared==appearances[year],"ticket appearance drift")
    assert_ok(union["tickets"]<=union_appeared,"dedup creates tickets")
    assert_ok(union["return_yen"]<=sum(c["return_yen"] for c in results.values()),
              "dedup creates winning payout")
    assert_ok(sum(x["tickets"] for x in by_type.values())==union["tickets"],
              "group type count mismatch")
    assert_ok(sum(x["return_yen"] for x in by_type.values())==union["return_yen"],
              "group type refund mismatch")
    assert_ok(sum(x["tickets"] for x in results.values())==union_appeared,
              "source ticket count mismatch")
    overlap=Counter()
    # Record per-race level duplicated purchases without saving individual picks.
    for rec in selected_by_race:
        overlap["duplicate_removed"] += (
            rec["before_dedup_ticket_count"]-rec["unique_ticket_count"])
    assert_ok(overlap["duplicate_removed"]==union_appeared-union["tickets"],
              "same physical ticket overlap differs")
    print("YEAR_134_DEDUP_COMPLETE",year,"unique",union["tickets"],
          "hits",union["hit_tickets"],"net",union["return_yen"]-union["tickets"]*100,
          "orig_strategy_134x_parity",True,flush=True)
    return {
        "union":union,"by_type":by_type,"strategies":results,
        "selected_by_race":selected_by_race,
        "winning_ticket_amounts":refund_order,
        "before_dedup_count":union_appeared,
    }


def main():
    ap=argparse.ArgumentParser()
    for k in ("selected-134","reference-yearly","predictions","ability-cache",
              "backfill-root","out-dir"):ap.add_argument("--"+k,required=True)
    a=ap.parse_args()
    selected=load_selected(a.selected_134)
    originals=historical(a.reference_yearly,selected)
    data,_=load_predictions(a.predictions,
         "research-results/all-race-one-pick-v1/run-38668174639/all-race-one-picks.csv.gz")
    assert_ok(set(data.year.astype(int))==set(YEARS),"2026 SEALED")
    z=pd.read_csv(a.ability_cache,compression="infer",
          usecols=["_race_id","_horse_id","distill_score_z"],
          dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    assert_ok(not z.duplicated(["_race_id","_horse_id"]).any(),
              "duplicate ability cache keys")
    n=len(data)
    data=data.merge(z,on=["_race_id","_horse_id"],how="left",validate="one_to_one")
    assert_ok(len(data)==n and data["distill_score_z"].notna().all(),
              "missing original frozen ability")
    years={}
    yearly_rows=[]
    family_totals=defaultdict(Counter)
    portfolio=Counter()
    appearances=0
    winner_amounts=[]
    racewise=[]
    for year in YEARS:
        result=run_year(year,data,Path(a.backfill_root),selected,originals)
        ys=result["union"]
        years[str(year)]=cell(ys,RACES_YEAR)
        yearly_rows.append({"year":str(year),**years[str(year)],
                  "before_dedup_tickets":result["before_dedup_count"],
                  "duplicates_removed":result["before_dedup_count"]-ys["tickets"]})
        appearances+=result["before_dedup_count"]
        for key in ("tickets","hit_tickets","return_yen",
                    "bet_races","hit_races"):portfolio[key]+=ys[key]
        for fam,c in result["by_type"].items():
            for k,v in c.items():family_totals[fam][k]+=v
        winner_amounts.extend(result["winning_ticket_amounts"])
        racewise.extend(result["selected_by_race"])
    three=cell(portfolio,TOTAL_RACES)
    three["before_dedup_tickets"]=appearances
    three["duplicates_removed"]=appearances-portfolio["tickets"]
    yearly_rows.append({"year":"2023-2025",**three})
    assert_ok(three["duplicates_removed"]>=0,"negative duplicates")
    assert_ok(three["return_yen"]==sum(y["return_yen"] for y in yearly_rows[:3]),
              "yearly pooling drift")
    assert_ok(three["stake_yen"]==sum(y["stake_yen"] for y in yearly_rows[:3]),
              "yearly stake pooling drift")
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    def write_csv(path,rows):
        assert rows
        with path.open("w",encoding="utf-8",newline="") as fd:
            wr=csv.DictWriter(fd,fieldnames=list(rows[0]))
            wr.writeheader();wr.writerows(rows)
    write_csv(out/"deduped-134-annual-and-total.csv",yearly_rows)
    write_csv(out/"deduped-134-by-bet-kind-3yr.csv",
        [{"bet_kind":f,**cell(c,TOTAL_RACES)} for f,c in sorted(family_totals.items())])
    write_csv(out/"deduped-134-by-race.csv",racewise)
    summary={
        "contract":"L2_EV_TOP6_134_THREE_YEAR_STABLE_ODDS_FREE_UNION_DEDUP_V1",
        "selected_strategy_count":134,
        "selection":"the exact 134 settings profitable separately in EACH year 2023,2024,2025 from Run 39229430682",
        "portfolio":"buy ALL qualifying 134 settings in ALL 2023-25 races; pay each physical bet only once per race",
        "ticket_unit_yen":100,
        "ticket_dedup_key":"(race_id, bet_kind, canonical horse numbers; order retained in exacta/trifecta)",
        "years":years,
        "combined_3_years":three,
        "by_bet_kind":{kind:cell(c,TOTAL_RACES) for kind,c in sorted(family_totals.items())},
        "top_3_winning_ticket_returns_yen":sorted(winner_amounts,reverse=True)[:3],
        "total_winning_refund_yen":sum(winner_amounts),
        "roi_after_removing_top3_winning_refunds_pct":(
             round((three["return_yen"]-
                sum(sorted(winner_amounts,reverse=True)[:3]))/
                portfolio["tickets"],6)
             if portfolio["tickets"] else None),
        "strategy_reproduction":"ALL 134x3 annual ticket counts, hits, and payouts exactly match previously saved post-prune results",
        "model_recalculated":False,
        "2026_sealed":True,
        "frozen_L2_1_unchanged":True,
        "input_historical_final_win_odds_for_EV_ranking":True,
        "per_ticket_pruning_odds_used":False,
        "future_reproducibility":"not tested: strategy set selected on same years",
        "no_extra_experiments":True
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ALL_134_UNION_DONE",json.dumps({
        "combined":three,"years":years,
        "top3":summary["top_3_winning_ticket_returns_yen"],
        "roi_after_removing_top3":summary["roi_after_removing_top3_winning_refunds_pct"]
    },ensure_ascii=False),flush=True)


if __name__=="__main__":main()
