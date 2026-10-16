#!/usr/bin/env python3
"""Rank1 replay of frozen L2.1 STANDARD alongside fixed rank2-6 results."""

from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import pandas as pd
import run_l2_ev_rank2to6_pruning20_best_roi_v1 as ev
import run_l2_all_fixed_odds_band_lenses_v1 as lens
import run_l2_fixed_ticket_pruning_3040_v1 as fixed
import run_l2_hindsight_stable_black_union_v1 as frozen
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,assert_ok

def parse_line(line):
    v=line.split("|")
    assert_ok(len(v)==4,("invalid frozen rule",line))
    return v[0],int(v[1]),v[2],v[3]

def frozen_standard(path):
    o=json.loads(Path(path).read_text(encoding="utf-8"))
    assert_ok(o["contract"]=="KEIBA_L2_1_FIXED_PURCHASE_V1"
        and o["version"]=="2.1.0" and o["primary_policy"]=="STANDARD",
        "L2.1 contract/version mismatch")
    assert_ok(o["ticket_policy"]["anchor"].startswith("EXACT_EXISTING_ANCHOR_FULL"),
              "frozen rank1 anchor mismatch")
    rows=o["frozen_rules"]["STANDARD"]
    assert_ok(len(rows)==o["rule_counts"]["STANDARD"]==50,"frozen 50 rules changed")
    selected=defaultdict(set)
    for item in rows:
        assert_ok(item["gate"] in lens.GATES,"frozen gate not found")
        selected[parse_line(item["line"])].add(item["gate"])
    assert_ok(len(selected)==9 and sum(map(len,selected.values()))==50,
              "frozen line/gate signature changed")
    return o,dict(selected)

def blank():
    return dict(tickets=0,hits=0,payout_yen=0,races_bought=0)

def count(counter,tickets):
    if tickets:
        counter["races_bought"]+=1
        for payout in tickets.values():
            counter["tickets"]+=1
            counter["hits"]+=int(payout>0)
            counter["payout_yen"]+=int(payout)

def result(year,method,st,available):
    n=st["tickets"];ret=st["payout_yen"]
    return dict(rank=1,method=method,period=str(year),
        available_races=available,races_bought=st["races_bought"],
        tickets=n,winning_tickets=st["hits"],stake_yen=n*100,
        return_yen=ret,profit_yen=ret-n*100,
        roi_pct=100*ret/(n*100) if n else None)

def replay(data,picks,races,root,selected):
    original={y:blank() for y in YEARS}
    common={y:blank() for y in YEARS}
    eligible_counts={y:0 for y in YEARS}
    scored=0
    for year in YEARS:
        year_races=races[races.year.eq(year)]
        assert_ok(len(year_races)==RACES_YEAR,("year race count",year))
        groups=data[data.year.eq(year)].groupby("_race_id",sort=False)
        done=0
        for rid,pack,odds,g,info in ev.iterate_races(year,groups,year_races,root):
            done+=1
            axis=ev.choose_axis(rid,g,info,picks,1)
            assert_ok(axis==picks["ANCHOR_FULL"][rid],("EV1 axis mismatch",rid))
            shim=info.copy()
            shim["VALUE_EV_horse"]=axis
            source=defaultdict(dict)
            for tid,ticket,card in fixed.create_cards(g,pack,odds,rid,axis,shim):
                if tid in selected:
                    assert_ok(ticket not in source[tid],"source duplicate")
                    source[tid][ticket]=int(card[1])
            purchased={}
            if source:
                horse,horse_numbers=frozen.make_horse(g,pack)
                anchor=horse_numbers[axis]
                gates_cache={}
                for tid,by_ticket in source.items():
                    required=selected[tid]
                    for ticket,payout in by_ticket.items():
                        if ticket not in gates_cache:
                            flags=lens.gate_flags(ticket,horse,anchor)
                            assert_ok(set(flags)==set(lens.GATES),"gate feature signature drift")
                            gates_cache[ticket]=flags
                        if any(gates_cache[ticket][key] for key in required):
                            if ticket in purchased:
                                assert_ok(purchased[ticket]==payout,"duplicate payout drift")
                            purchased[ticket]=payout
            count(original[year],purchased)
            if ev.pruning_eval_eligible(year,info["race_date"]):
                eligible_counts[year]+=1
                count(common[year],purchased)
            if done%1200==0:print("FROZEN_RANK1_YEAR_PROGRESS",year,done,flush=True)
        assert_ok(done==RACES_YEAR,("rank1 races lost",year,done))
        print("FROZEN_RANK1_YEAR_COMPLETE",year,
              original[year]["tickets"],common[year]["tickets"],flush=True)
    return original,common,eligible_counts

def validate_exact_full(original,contract):
    full=contract["research_baseline"]["STANDARD_ALL"]
    mapping={"tickets":"tickets","hits":"winning_tickets",
             "payout_yen":"return_yen","races_bought":"races_bought"}
    for year in YEARS:
        ref=contract["research_baseline"]["STANDARD_BY_YEAR"][str(year)]
        for ours,theirs in mapping.items():
            assert_ok(original[year][ours]==ref[theirs],
                      ("FROZEN_L2_1_REPLAY_MISMATCH",year,ours,original[year][ours],ref[theirs]))
    for ours,theirs in mapping.items():
        total=sum(original[y][ours] for y in YEARS)
        assert_ok(total==full[theirs],("FROZEN_L2_1_POOLED_MISMATCH",ours,total,full[theirs]))
    print("L21_ORIGINAL_FROZEN_50_EXACT_2023_2024_2025_MATCH",flush=True)

def pooled(stats_by_year):
    return {k:sum(stats_by_year[y][k] for y in YEARS) for k in stats_by_year[2023]}

def write_csv(path,rows):
    assert_ok(rows,"empty rows")
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    for key in ("predictions","picks","racewise","backfill-root",
                "contract","rank2to6-summary","out-dir"):
        p.add_argument("--"+key,required=True)
    a=p.parse_args()
    contract,selected=frozen_standard(a.contract)
    data,picks=lens.frame(a.predictions,a.picks)
    races=pd.read_csv(a.racewise,compression="infer",dtype={
        "race_id":str,"VALUE_EV_horse":str,"AI_RAW_TOP_horse":str})
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert_ok(len(races)==TOTAL_RACES and races.race_id.is_unique,
              "source race coverage drift")
    assert_ok(set(races.year)==set(YEARS),"2026 still sealed")
    original,common,eligible=replay(data,picks,races,Path(a.backfill_root),selected)
    validate_exact_full(original,contract)
    assert_ok(eligible=={2023:3144,2024:3456,2025:3456},
              ("2023-Jan common window drift",eligible))
    rows=[result(y,"FROZEN_L2_1_STANDARD_50",original[y],RACES_YEAR) for y in YEARS]
    rows.append(result("2023-2025","FROZEN_L2_1_STANDARD_50",pooled(original),TOTAL_RACES))
    common_rows=[result(y,"FROZEN_L2_1_STANDARD_50",common[y],eligible[y]) for y in YEARS]
    common_rows.append(result("2023FEB-2025","FROZEN_L2_1_STANDARD_50",
                              pooled(common),sum(eligible.values())))
    source=json.loads(Path(a.__dict__["rank2to6_summary"]).read_text(encoding="utf-8"))
    assert_ok(source["january_2023_excluded_from_all_policies"] is True
              and source["evaluation_start_2023"]=="2023-02-01",
              "2-6 comparison window mismatch")
    assert_ok(len(source["selected_rows"])==5,"missing EV 2-6 winners")
    comparison=[dict(rank=1,source="FROZEN_L2.1",
                     selection="STANDARD_50_GATES",
                     black_lines=9,conditions=50,
                     **{k:common_rows[-1][k] for k in
                         ("tickets","winning_tickets","stake_yen","return_yen","profit_yen","roi_pct")})]
    for row in source["selected_rows"]:
        rank=int(row["ev_rank"])
        assert_ok(rank in (2,3,4,5,6),"unexpected ranked winner")
        comparison.append(dict(rank=rank,source="PRUNED_20_BEST_3YR_ROI",
                selection=row["adopted_method"],black_lines=row["black_odds_lines"],
                conditions=None,tickets=row["purchased_tickets"],
                winning_tickets=row["winning_tickets"],stake_yen=row["stake_yen"],
                return_yen=row["return_yen"],profit_yen=row["profit_yen"],
                roi_pct=row["roi_pct"]))
    assert_ok([r["rank"] for r in comparison]==[1,2,3,4,5,6],
              "rank alignment drift")
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"rank1_frozen_standard50_full_years.csv",rows)
    write_csv(out/"rank1_frozen_standard50_common_window.csv",common_rows)
    write_csv(out/"ev_ranks_1_to_6_comparison_common_window.csv",comparison)
    summary=dict(contract="EV_RANK1_FROZEN_L21_JOIN_EXISTING_RANKS2_TO_6_V1",
                 rank1_frozen_contract="KEIBA_L2_1_FIXED_PURCHASE_V1",
                 rank1_frozen_replay_exactly_matches_all_2023_2025=True,
                 rank1_method_not_reselected=True,rank1_standard_50_conditions=True,
                 rank1_full_years=rows,rank1_common_window=common_rows,
                 common_window_2023_02_through_2025_12=True,
                 ranks2to6_unchanged_source_run=38891800300,
                 comparisons=comparison,
                 counts_per_rank_not_cross_rank_deduplicated=True,
                 different_rank1_and_rank2to6_reduction_stages=True,
                 historical_FINAL_quotes_not_betting_time=True,
                 selected_conditions_evaluated_same_period=True,year2026_sealed=True)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print("RANK1_FROZEN_AND_ALL_SIX_RANKS_COMPARE",json.dumps(comparison,ensure_ascii=False))

if __name__=="__main__":main()
