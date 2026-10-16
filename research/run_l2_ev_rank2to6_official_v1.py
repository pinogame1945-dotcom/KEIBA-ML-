#!/usr/bin/env python3
"""EV ranks 2-6, exact original 152 x 20 price-band tickets, official L2.1 gates.

Descriptive retrospective 2023-25 selection, NOT a forward validation:
ALL_YEAR_BLACK requires positive per-year return for each (rank,line,gate);
STABLE_MEDIUM and STABLE_STRICT use the unchanged official L2.1 thresholds.
The source EV win-odds score uses historical FINAL prices; betting-time
availability is not proved. 2026 is sealed. Frozen helpers are imported
read-only. Every generated line is evaluated (including red lines).
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import run_l2_all_fixed_odds_band_lenses_v1 as lens
import run_l2_fixed_ticket_pruning_3040_v1 as fixed
import run_l2_raw_market25_stability_v1 as official
import run_l2_hindsight_stable_black_union_v1 as stable
from build_l2_market_helpers_v1 import load_day,load_odds_day
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,assert_ok

RANKS=(2,3,4,5,6)
POLICIES=("ALL_YEAR_BLACK","STABLE_MEDIUM","STABLE_STRICT")
GATES=tuple(lens.GATES)
assert GATES==official.GATES and POLICIES==official.RULES
assert len(GATES)==17
assert len(fixed.FAMILIES)==10

def order_ev(g):
    """Fixed EV score, same score as official rank-1 pick; deterministic ties."""
    df=g.copy()
    df["_ev_score"]=df["P__ANCHOR_FULL"].astype(float)*df["final_win_odds"].astype(float)
    assert_ok(df["_ev_score"].notna().all() and df["_ev_score"].ge(0).all(),
              "invalid EV score")
    seq=df.sort_values(["_ev_score","market_rank","_horse_id"],
                       ascending=[False,True,True],kind="mergesort")["_horse_id"].astype(str).tolist()
    assert_ok(len(seq)==len(set(seq)) and len(seq)>=6,"EV rank coverage")
    return seq

def counter():
    return official.new_counter()

def add(dst,src):
    """Merge exact unit-ticket counts, payout and top-three winning receipts."""
    import heapq
    dst["n"]+=src["n"]
    dst["hits"]+=src["hits"]
    dst["payout"]+=src["payout"]
    for v in src["top3"]:
        top=dst["top3"]
        if len(top)<3:heapq.heappush(top,v)
        elif v>top[0]:heapq.heapreplace(top,v)

def csv_gz(path,rows,headers):
    with gzip.open(path,"wt",encoding="utf-8",newline="",compresslevel=6) as f:
        w=csv.DictWriter(f,fieldnames=headers,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def yearly_pass(data,picks,races,root,selected=None):
    """One BACKFILL read per day; one fixed ticket generator per EV rank/race.

    selected=None collects exact all-line x 17-gate stats. selected otherwise
    settles only accepted policies, ORs accepted gates, and physically dedups
    tickets within each race, across all lines and across EV ranks.
    """
    first=selected is None
    stats=defaultdict(counter) if first else None
    portfolios=defaultdict(lambda:{"counter":counter(),"races":0,"hit_races":0})
    ranks=defaultdict(lambda:{"races":0,"wins":0,"top3":0,"market_rank_sum":0.0,
                               "odds_sum":0.0})
    masks={}
    if not first:
        for rank in RANKS:
            for policy in POLICIES:
                mm=defaultdict(int)
                for tid,gate in selected[rank][policy]:
                    mm[tid]|=1<<GATES.index(gate)
                masks[rank,policy]=dict(mm)
    for year in YEARS:
        yr=races[races.year.eq(year)]
        assert_ok(len(yr)==RACES_YEAR,("historical race count",year))
        grouped=data[data.year.eq(year)].groupby("_race_id",sort=False)
        by_rid=yr.set_index("race_id")
        dates=yr.groupby("race_date")["race_id"].apply(list)
        completed=0
        for date,ids in dates.items():
            day=str(date)[:10]
            packs=load_day(root/"data"/"daily"/(day+".jsonl.gz"),set(ids))
            odds=load_odds_day(root/"data"/"odds"/"daily"/(day+".jsonl.gz"),set(ids))
            for rid in sorted(ids):
                pack=packs.get(rid);price=odds.get(rid)
                assert_ok(pack is not None and price is not None,("missing real race",rid))
                assert_ok(price.get("odds_kind")=="FINAL" and
                          price.get("source_status")=="result",("historical price state",rid))
                g=grouped.get_group(rid)
                info=by_rid.loc[rid].copy()
                ev=order_ev(g)
                assert_ok(ev[0]==picks["ANCHOR_FULL"][rid],
                          ("official VALUE_EV rank-1 drift",rid,ev[0],picks["ANCHOR_FULL"][rid]))
                assert_ok(str(info["VALUE_EV_horse"])==ev[0],
                          ("official source racewise drift",rid))
                horse,no=stable.make_horse(g,pack)
                g_by_id=g.set_index("_horse_id")
                within=defaultdict(dict)
                for rank in RANKS:
                    axis=ev[rank-1]
                    if first:
                        st=ranks[rank]
                        st["races"]+=1
                        st["wins"]+=int(g_by_id.loc[axis,"target_win"])
                        st["top3"]+=int(g_by_id.loc[axis,"target_top3"])
                        st["market_rank_sum"]+=float(g_by_id.loc[axis,"market_rank"])
                        st["odds_sum"]+=float(g_by_id.loc[axis,"final_win_odds"])
                    if not first and not any(masks[rank,p] for p in POLICIES):
                        continue
                    shim=info.copy()
                    shim["VALUE_EV_horse"]=axis
                    cache={}
                    axis_no=no[axis]
                    # Historical bet templates, bands, prices and payouts are unchanged.
                    for tid,ticket,card in fixed.create_cards(g,pack,price,rid,axis,shim):
                        if not first and not any(tid in masks[rank,p] for p in POLICIES):
                            continue
                        ret=int(card[1])
                        if ticket not in cache:
                            flags=lens.gate_flags(ticket,horse,axis_no)
                            assert_ok(set(flags)==set(GATES),"official gate keys drift")
                            cache[ticket]=sum(1<<i for i,k in enumerate(GATES) if flags[k])
                            assert_ok(cache[ticket]&1,"ALL gate missing")
                        bm=cache[ticket]
                        if first:
                            for i,gate in enumerate(GATES):
                                if bm&(1<<i):
                                    official.update_counter(stats[rank,year,tid,gate],ret)
                        else:
                            for policy in POLICIES:
                                if bm&masks[rank,policy].get(tid,0):
                                    # Ticket uniqueness is per-race, per policy,
                                    # regardless of overlapping methods and EV ranks.
                                    local=within[rank,policy]
                                    union=within["EV_RANK_2_TO_6_UNION",policy]
                                    assert_ok(ticket not in local or local[ticket]==ret,
                                              ("rank physical payout drift",rid,rank,ticket))
                                    assert_ok(ticket not in union or union[ticket]==ret,
                                              ("cross-rank physical payout drift",rid,ticket))
                                    local[ticket]=ret
                                    union[ticket]=ret
                if not first:
                    for (label,policy),tickets in within.items():
                        if not tickets:continue
                        st=portfolios[label,policy,year]
                        st["races"]+=1
                        st["hit_races"]+=int(any(v>0 for v in tickets.values()))
                        for ret in tickets.values():
                            official.update_counter(st["counter"],ret)
                completed+=1
                if completed%1200==0:
                    print("EV_RANK2_TO6_YEAR_PROGRESS",year,completed,
                          "pass","ledger" if first else "physical_union",flush=True)
        assert_ok(completed==RACES_YEAR,("year count",year,completed))
        print("EV_RANK2_TO6_YEAR_DONE",year,
              "ledger_groups",len(stats) if first else "selected",flush=True)
    return (stats,ranks) if first else portfolios

def evaluate(stats,ranks,original,out):
    selected={}
    audit_fields=None
    decision_count={}
    baseline=[]
    selected_lines=[]
    detailed_path=out/"all_rank_line_gate_official_decisions.csv.gz"
    with gzip.open(detailed_path,"wt",encoding="utf-8",newline="",compresslevel=6) as f:
        w=None
        for rank in RANKS:
            by_year={y:{} for y in YEARS}
            for (r,y,tid,gate),st in stats.items():
                if r==rank:by_year[y][tid,gate]=st
            kept,decisions=official.select_stable(by_year,original)
            selected[rank]=kept
            decision_count[str(rank)]={p:len(kept[p]) for p in POLICIES}
            for d in decisions:
                tid=(d["partner_order"],int(d["other_horses_k"]),
                     d["family"],d["odds_band"])
                orig_profit=int(original[tid]["2023-2025_損益円"])
                dd={"ev_rank":rank,"ev_rank1_original_profit_yen":orig_profit,
                    "ev_rank1_original_black":orig_profit>0,
                    "red_to_black_from_ev_rank1":orig_profit<=0 and d["each_year_black"],
                    **d}
                if w is None:
                    audit_fields=list(dd)
                    w=csv.DictWriter(f,fieldnames=audit_fields)
                    w.writeheader()
                w.writerow(dd)
                if d["medium_decision"]=="ACCEPT" or d["strict_decision"]=="ACCEPT":
                    selected_lines.append(dd)
            for tid,row in original.items():
                vals=[stats.get((rank,y,tid,"ALL"),counter()) for y in YEARS]
                pooled=counter()
                for v in vals:add(pooled,v)
                pm=official.metric(pooled)
                prev=int(row["2023-2025_損益円"])
                base=dict(ev_rank=rank,line="|".join(map(str,tid)),
                     partner_order=tid[0],other_horses_k=tid[1],
                     family=tid[2],odds_band=tid[3],
                     ev_rank1_profit_yen=prev,ev_rank1_black=prev>0,
                     rank_profit_yen=pm["profit_yen"],
                     rank_roi_pct=pm["roi_pct"],
                     rank_tickets=pm["tickets"],rank_hit_tickets=pm["hit_tickets"],
                     rank_hit_rate_pct=pm["hit_rate_pct"],
                     pooled_black=pm["profit_yen"]>0,
                     each_year_black=all(s["n"]>0 and s["payout"]>100*s["n"] for s in vals),
                     red_to_pooled_black=prev<=0 and pm["profit_yen"]>0)
                for y,s in zip(YEARS,vals):
                    base[f"{y}_tickets"]=s["n"]
                    base[f"{y}_hits"]=s["hits"]
                    base[f"{y}_profit_yen"]=s["payout"]-100*s["n"]
                    base[f"{y}_roi_pct"]=100*s["payout"]/(100*s["n"]) if s["n"] else None
                baseline.append(base)
            print("EV_RANK2_TO6_SELECTED",rank,decision_count[str(rank)],flush=True)
    assert_ok(len(baseline)==len(RANKS)*3040,"fixed line count")
    fields=list(baseline[0])
    csv_gz(out/"all_152x20_odds_rank2to6_red_black.csv.gz",baseline,fields)
    selected_headers=audit_fields or ["ev_rank","line","gate"]
    csv_gz(out/"accepted_medium_strict_lines.csv.gz",selected_lines,selected_headers)
    return selected,decision_count,baseline

def portfolio_rows(portfolios,selected):
    rows=[]
    labels=(*RANKS,"EV_RANK_2_TO_6_UNION")
    for label in labels:
        for policy in POLICIES:
            for period in (*YEARS,"2023-2025"):
                total=counter();races=0;wins=0
                yy=YEARS if period=="2023-2025" else (period,)
                for year in yy:
                    s=portfolios.get((label,policy,year))
                    if s:
                        add(total,s["counter"]);races+=s["races"];wins+=s["hit_races"]
                if label in RANKS:
                    count=len(selected[label][policy])
                    uniq=len({tid for tid,gate in selected[label][policy]})
                else:
                    count=sum(len(selected[rank][policy]) for rank in RANKS)
                    uniq=sum(len({tid for tid,gate in selected[rank][policy]}) for rank in RANKS)
                rows.append(dict(ev_rank=label,policy=policy,period=period,
                                 line_gate_pairs=count,unique_rank_lines=uniq,
                                 bought_races=races,hit_races=wins,
                                 **official.metric(total)))
    assert_ok(len(rows)==6*3*4,"portfolio row count")
    return rows

def main():
    p=argparse.ArgumentParser()
    for name in ("predictions","picks","racewise","bands","backfill-root","out-dir"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    original,_=fixed.load_lines(a.bands)
    assert_ok(len(original)==3040,"exact original 152 x 20 baseline")
    data,picks=lens.frame(a.predictions,a.picks)
    races=pd.read_csv(a.racewise,compression="infer",
                      dtype={"race_id":str,"VALUE_EV_horse":str,"AI_RAW_TOP_horse":str})
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert_ok(len(races)==TOTAL_RACES and races.race_id.is_unique,"historical 10368 races")
    assert_ok(set(races.year)==set(YEARS),"2026 sealed")
    root=Path(a.backfill_root)
    stats,ranks=yearly_pass(data,picks,races,root)
    selected,counts,baseline=evaluate(stats,ranks,original,out)
    del stats
    portfolios=yearly_pass(data,picks,races,root,selected)
    rows=portfolio_rows(portfolios,selected)
    official.write(out/"dedup_portfolios_by_rank_year_and_all_rank_union.csv",rows)
    pooled=[x for x in rows if x["period"]=="2023-2025"]
    summary=dict(contract="EV_RANK2_TO6_EXACT_ORIGINAL_152x20_OFFICIAL_L21_V1",
                 years=list(YEARS),ev_ranks=list(RANKS),races=TOTAL_RACES,
                 ticket_templates=152,odds_bands=20,source_lines_per_rank=3040,
                 original_l2_gates=list(GATES),
                 official_stability=official.CONFIG,
                 each_year_positive_requirement=True,
                 black_policy="Each year 2023, 2024, 2025 individually positive; no rank-1 black-line preselection",
                 ev_score="P__ANCHOR_FULL * final_win_odds",
                 rank1_selection_audited_against_frozen_picks=True,
                 selected_counts=counts,
                 baseline_counts={str(rank):{
                      "three_year_pooled_black_lines":sum(r["pooled_black"] for r in baseline if r["ev_rank"]==rank),
                      "each_year_black_lines":sum(r["each_year_black"] for r in baseline if r["ev_rank"]==rank),
                      "rank1_red_to_pooled_black_lines":sum(r["red_to_pooled_black"] for r in baseline if r["ev_rank"]==rank)
                 } for rank in RANKS},
                 portfolios=pooled,
                 all_ranks_physical_dedup=True,
                 per_rank_physical_dedup=True,
                 same_period_hindsight_selection_bias=True,
                 prices="FINAL_POST_CLOSE_NOT_PREBET",
                 sealed_2026=True,unchanged_frozen_L21=True,
                 git_branch_only=True,free_standard_cpu_no_paid_storage=True)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("EV_RANK2_TO6_OFFICIAL_L21_DONE",
          json.dumps({"lines":summary["baseline_counts"],"selected":counts,
                      "portfolios":pooled},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
