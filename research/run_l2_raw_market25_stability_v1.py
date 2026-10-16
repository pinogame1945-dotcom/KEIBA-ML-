#!/usr/bin/env python3
"""Pure AI RAW top1 + frozen MARKET_25, followed by official L2.1 stability gates.

Starting candidates are EXACTLY the 235 original lines black in 2023-25
under MARKET_25 (2023 BASE unpruned; 2024 prior-2023 thresholds; 2025
prior-2023+24 thresholds). Preserve original 17 L2.1 gate semantics,
stable-medium / stable-strict numerical cutoffs and per-year-positive test.
All accepted tickets deduplicated by race/kind/canonical combination.

A one-pass implementation: original bookmaker data is read once; scores from
earlier years determine next-year MARKET_25 without test-year outcomes; selected
ticket events (only 235 source lines) are held in memory for settlement.
No frozen L2.1 edits, no 2026 outcomes, no original 2022 data inference.
"""
from __future__ import annotations
import argparse,csv,gzip,heapq,json,math
from collections import Counter,defaultdict
from pathlib import Path

import pandas as pd
import run_l2_fixed_ticket_pruning_3040_v1 as fixed
import run_l2_hindsight_stable_black_union_v1 as stable
import run_l2_raw_top_odds_boundary_v1 as raw
import run_l2_raw_top_ticket_pruning_oos_v1 as pruning
import run_l2_all_fixed_odds_band_lenses_v1 as lens
from build_l2_market_helpers_v1 import load_day,load_odds_day
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,assert_ok

GATES=tuple(lens.GATES)
POLICY="MARKET_25"
CONFIG={n:dict(stable.POLICIES[n]) for n in ("STABLE_MEDIUM","STABLE_STRICT")}
RULES=("ALL_YEAR_BLACK","STABLE_MEDIUM","STABLE_STRICT")
EXPECTED_SOURCE_BLACK=235
EXPECTED_SOURCE_TICKETS=102150
EXPECTED_SOURCE_HITS=1977
EXPECTED_SOURCE_PAYOUT=18422210
# These benchmarks are results of the prior completed 'MARKET_25' run.
EXPECTED_FULL_TWO_YEARS=(760071,14016,51881230)
EXPECTED_FULL_2023=(1327558,14237,94978800)
assert len(GATES)==17 and len(CONFIG)==2
assert CONFIG["STABLE_MEDIUM"]==dict(n=500,hits=30,per_year_hits=8,
                                    max1=.40,max3=.70,minus3_roi=90)
assert CONFIG["STABLE_STRICT"]==dict(n=800,hits=50,per_year_hits=12,
                                    max1=.30,max3=.55,minus3_roi=100)


def rows(path):
    with (gzip.open(path,"rt",encoding="utf-8-sig",newline="") if str(path).endswith(".gz")
          else open(path,encoding="utf-8-sig",newline="")) as f:
        yield from csv.DictReader(f)


def load_selected(path):
    selected={}
    for r in rows(path):
        if r["method"]!=POLICY:
            continue
        tid=(r["partner_order"],int(r["other_horses_k"]),r["family"],r["odds_band"])
        assert tid not in selected,("duplicate source line",tid)
        selected[tid]=dict(n=int(r["tickets"]),h=int(r["hit_tickets"]),
                           payout=int(r["payout_yen"]),profit=int(r["profit_yen"]))
        assert selected[tid]["profit"]>0
    assert len(selected)==EXPECTED_SOURCE_BLACK,len(selected)
    assert sum(s["n"] for s in selected.values())==EXPECTED_SOURCE_TICKETS
    assert sum(s["h"] for s in selected.values())==EXPECTED_SOURCE_HITS
    assert sum(s["payout"] for s in selected.values())==EXPECTED_SOURCE_PAYOUT
    return selected


def load_line_baselines(original_path,pruned_path,selected):
    expected={}
    for r in rows(original_path):
        year=r["period"]
        if year!="2023": continue
        tid=(r["partner_order"],int(r["other_horses_k"]),r["family"],r["odds_band"])
        if tid in selected:
            expected[(2023,tid)]=(int(r["tickets"]),int(r["hit_tickets"]),int(r["payout_yen"]))
    for r in rows(pruned_path):
        if r["policy"]!=POLICY:continue
        yr=int(r["period"])
        assert yr in (2024,2025)
        tid=(r["partner_order"],int(r["other_horses_k"]),r["family"],r["odds_band"])
        if tid in selected:
            expected[(yr,tid)]=(int(r["after_tickets"]),int(r["after_hits"]),
                                int(r["after_payout_yen"]))
    assert len(expected)==3*len(selected),len(expected)
    for tid,s in selected.items():
        observed=tuple(sum(expected[(year,tid)][i] for year in YEARS) for i in range(3))
        assert observed==(s["n"],s["h"],s["payout"]),("source 235 audit mismatch",tid,observed,s)
    return expected


def prior_market25_thresholds(group_scores,kind_scores):
    """Same thresholds as frozen.create_thresholds in MARKET, KEEP=25 slice.

    When an exact (family, price-bin) group has <100 training physical tickets,
    original falls back to whole-family distribution. Keep quantile/tie
    behaviour frozen, including lower bound and >= threshold.
    """
    out={}
    for kind,arr in kind_scores.items():
        vals=sorted(arr)
        out[kind]=fixed.threshold(vals,25)
    for group,arr in group_scores.items():
        if len(arr)<100:
            out[group]=out[group[0]]
        else:
            out[group]=fixed.threshold(sorted(arr),25)
    return out


def market25_keep(card,year,limits):
    if year==2023:return True
    kind,bz,_,_,_=card[2]
    score=card[4]
    threshold=limits.get((kind,bz),limits.get(kind,float("-inf")))
    return score>=threshold


def update_counter(st,receipt):
    st["n"]+=1
    if receipt>0:
        st["hits"]+=1
        st["payout"]+=receipt
        top=st["top3"]
        if len(top)<3:
            heapq.heappush(top,receipt)
        elif receipt>top[0]:
            heapq.heapreplace(top,receipt)


def new_counter():
    return dict(n=0,hits=0,payout=0,top3=[])


def metric(st):
    n=st["n"];h=st["hits"];pay=st["payout"]
    top=sorted(st["top3"],reverse=True)
    top1=top[0] if top else 0
    top3=sum(top)
    return dict(tickets=n,hit_tickets=h,hit_rate_pct=100*h/n if n else None,
                stake_yen=n*100,payout_yen=pay,profit_yen=pay-100*n,
                roi_pct=100*pay/(100*n) if n else None,
                biggest1_payout=top1,biggest3_payout_sum=top3,
                biggest1_return_share_pct=100*top1/pay if pay else None,
                biggest3_return_share_pct=100*top3/pay if pay else None,
                roi_after_remove_biggest3=100*(pay-top3)/(100*n) if n else None)


def main_pass(data,picks,races,root,selected):
    # Store only numeric MARKET score, not all other model feature score arrays.
    history_group=defaultdict(list)
    history_kind=defaultdict(list)
    yearly_hist=defaultdict(lambda:defaultdict(new_counter))
    # each event: (year, race id, tid, ticket, exact payout, gate bitmap)
    accepted_source_events=[]
    physical_year={}
    counted_lines=defaultdict(Counter)
    yearly_full_market25={}
    for year in YEARS:
        limits=prior_market25_thresholds(history_group,history_kind) if year>2023 else {}
        df=races[races.year.eq(year)]
        groups=data[data.year.eq(year)].groupby("_race_id",sort=False)
        by_race=df.set_index("race_id")
        dates=df.groupby("race_date")["race_id"].apply(list)
        full=new_counter()
        full_race_count=0
        for date,ids in dates.items():
            day=str(date)[:10]
            packs=load_day(root/"data"/"daily"/(day+".jsonl.gz"),set(ids))
            odds=load_odds_day(root/"data"/"odds"/"daily"/(day+".jsonl.gz"),set(ids))
            for rid in sorted(ids):
                pack=packs.get(rid);market=odds.get(rid)
                assert_ok(pack is not None and market is not None,("missing race",rid))
                assert_ok(market.get("odds_kind")=="FINAL"
                          and market.get("source_status")=="result",("odds state",rid))
                g=groups.get_group(rid)
                info=by_race.loc[rid]
                axis=str(info["AI_RAW_TOP_horse"])
                assert_ok(axis==str(g.sort_values(
                    ["distill_ai_rank","market_rank","_horse_id"],
                    ascending=[True,True,True],kind="mergesort").iloc[0]["_horse_id"]),
                    "RAW rank drift")
                assert_ok(str(info["VALUE_EV_horse"])==picks["ANCHOR_FULL"][rid],
                          "frozen EV pick drift")
                shim=info.copy()
                shim["VALUE_EV_horse"]=axis

                all_cards={}
                selected_by_line=defaultdict(dict)
                for tid,ticket,card in fixed.create_cards(g,pack,market,rid,axis,shim):
                    if ticket in all_cards:
                        assert all_cards[ticket]==card,("physical card mismatch",rid,ticket)
                    else:
                        all_cards[ticket]=card
                    if tid in selected:
                        assert ticket not in selected_by_line[tid],("duplicate within line",rid,tid)
                        selected_by_line[tid][ticket]=card

                kept={ticket for ticket,card in all_cards.items()
                      if market25_keep(card,year,limits)}
                if kept:full_race_count+=1
                for ticket in kept:
                    update_counter(full,int(all_cards[ticket][1]))

                # This year's labels never influence its own threshold.
                if year<2025:
                    for card in all_cards.values():
                        group=(card[2][0],card[2][1])
                        v=card[4]
                        history_group[group].append(v)
                        history_kind[group[0]].append(v)

                # The L2.1 source is restricted to the 235 lines in pooled black.
                source_selected={tid:{ticket:card for ticket,card in items.items() if ticket in kept}
                                 for tid,items in selected_by_line.items()}
                source_selected={tid:items for tid,items in source_selected.items() if items}
                if source_selected:
                    horse,no=stable.make_horse(g,pack)
                    axis_num=no[axis]
                    gate_cache={}
                    for tid,items in source_selected.items():
                        for ticket,card in items.items():
                            ret=int(card[1])
                            counted_lines[(year,tid)]["n"]+=1
                            counted_lines[(year,tid)]["h"]+=int(ret>0)
                            counted_lines[(year,tid)]["payout"]+=ret
                            if ticket not in gate_cache:
                                flags=lens.gate_flags(ticket,horse,axis_num)
                                assert set(flags)==set(GATES),("gate signature drift",rid)
                                gate_cache[ticket]=sum(1<<i for i,key in enumerate(GATES)
                                                       if flags[key])
                            mask=gate_cache[ticket]
                            assert mask & 1,("ALL gate must always be set",rid)
                            accepted_source_events.append((year,rid,tid,ticket,ret,mask))
                            for i,key in enumerate(GATES):
                                if mask & (1<<i):
                                    update_counter(yearly_hist[year][(tid,key)],ret)

                physical_year[year]=True
                if sum(x["n"] for x in (full,)) and full["n"]%250000==0:
                    print("MARKET25_PROGRESS",year,full["n"],flush=True)
        assert len(df)==RACES_YEAR,("race count",year)
        yearly_full_market25[year]=dict(full)
        print("MARKET25_YEAR_DONE",year,"physical",full["n"],
              "source_event_count",len(accepted_source_events),
              "source_line_keys",sum(y==year for y,tid in counted_lines),
              "gate_observed",len(yearly_hist[year]),flush=True)
    return yearly_hist,accepted_source_events,counted_lines,yearly_full_market25


def audit_source(counted,expected,full):
    for (year,tid),vals in expected.items():
        obs=counted.get((year,tid),Counter())
        got=(obs["n"],obs["h"],obs["payout"])
        assert got==vals,("MARKET25 source per-line mismatch",year,tid,vals,got)
    assert (full[2023]["n"],full[2023]["hits"],full[2023]["payout"])==EXPECTED_FULL_2023
    got=tuple(sum(full[y][k] for y in (2024,2025)) for k in ("n","hits","payout"))
    assert got==EXPECTED_FULL_TWO_YEARS,("MARKET25 physical total drift",got)
    print("MARKET25_ORIGINAL_235_LINES_AND_EXACT_2024_2025_PHYSICAL_RECONCILED",flush=True)


def select_stable(yearly_hist,selected):
    decisions=[]
    kept={p:set() for p in RULES}
    target=sorted({(tid,g) for tid in selected for g in GATES})
    for tid,gate in target:
        yearly=[yearly_hist[y].get((tid,gate),new_counter()) for y in YEARS]
        totals=new_counter()
        for st in yearly:
            totals["n"]+=st["n"]
            totals["hits"]+=st["hits"]
            totals["payout"]+=st["payout"]
            for val in st["top3"]:
                top=totals["top3"]
                if len(top)<3:heapq.heappush(top,val)
                elif val>top[0]:heapq.heapreplace(top,val)
        m=metric(totals)
        each_year_positive=all(st["n"]>0 and st["payout"]>100*st["n"]
                               for st in yearly)
        year_min_hits=min(st["hits"] for st in yearly)
        share1=m["biggest1_return_share_pct"]
        share3=m["biggest3_return_share_pct"]
        minus3=m["roi_after_remove_biggest3"]
        checks_by_policy={}
        for name in ("STABLE_MEDIUM","STABLE_STRICT"):
            cfg=CONFIG[name]
            checks=dict(
                each_year_black=each_year_positive,
                min_n=m["tickets"]>=cfg["n"],
                min_hits=m["hit_tickets"]>=cfg["hits"],
                min_each_year_hits=year_min_hits>=cfg["per_year_hits"],
                max_single_payout_share=(share1 is not None and share1<=100*cfg["max1"]),
                max_top3_payout_share=(share3 is not None and share3<=100*cfg["max3"]),
                roi_after_top3=(minus3 is not None and minus3>=cfg["minus3_roi"]),
            )
            checks_by_policy[name]=checks
            if all(checks.values()):
                kept[name].add((tid,gate))
        if each_year_positive:
            kept["ALL_YEAR_BLACK"].add((tid,gate))
        decisions.append(dict(
            line="|".join(map(str,tid)),partner_order=tid[0],
            other_horses_k=tid[1],family=tid[2],odds_band=tid[3],
            gate=gate,each_year_black=each_year_positive,
            tickets=m["tickets"],hits=m["hit_tickets"],
            stake_yen=m["stake_yen"],payout_yen=m["payout_yen"],
            profit_yen=m["profit_yen"],roi_pct=m["roi_pct"],
            year2023_tickets=yearly[0]["n"],year2023_hits=yearly[0]["hits"],
            year2023_profit_yen=yearly[0]["payout"]-100*yearly[0]["n"],
            year2024_tickets=yearly[1]["n"],year2024_hits=yearly[1]["hits"],
            year2024_profit_yen=yearly[1]["payout"]-100*yearly[1]["n"],
            year2025_tickets=yearly[2]["n"],year2025_hits=yearly[2]["hits"],
            year2025_profit_yen=yearly[2]["payout"]-100*yearly[2]["n"],
            biggest1_payout_yen=m["biggest1_payout"],
            biggest3_payout_yen=m["biggest3_payout_sum"],
            biggest1_return_share_pct=share1,biggest3_return_share_pct=share3,
            roi_after_top3=minus3,
            medium_decision=("ACCEPT" if (tid,gate) in kept["STABLE_MEDIUM"] else "REJECT"),
            medium_reasons=",".join(n for n,ok in checks_by_policy["STABLE_MEDIUM"].items() if not ok),
            strict_decision=("ACCEPT" if (tid,gate) in kept["STABLE_STRICT"] else "REJECT"),
            strict_reasons=",".join(n for n,ok in checks_by_policy["STABLE_STRICT"].items() if not ok),
        ))
    assert kept["STABLE_STRICT"].issubset(kept["STABLE_MEDIUM"])
    assert kept["STABLE_MEDIUM"].issubset(kept["ALL_YEAR_BLACK"])
    return kept,decisions


def dedup_portfolios(events,kept):
    # accept "OR" over qualified gates for the same ticket; physical ticket
    # is counted once per race no matter how many source lines/gates match.
    accepted_masks=defaultdict(dict)
    for policy in RULES:
        for tid,gate in kept[policy]:
            i=GATES.index(gate)
            accepted_masks[policy][tid]=accepted_masks[policy].get(tid,0)|(1<<i)
    physical={p:{} for p in RULES}
    for y,rid,tid,ticket,ret,mask in events:
        for policy in RULES:
            if mask & accepted_masks[policy].get(tid,0):
                key=(y,rid,ticket)
                if key in physical[policy]:
                    assert physical[policy][key]==ret,("physical payout disagreement",key)
                else:
                    physical[policy][key]=ret
    assert set(physical["STABLE_STRICT"]).issubset(set(physical["STABLE_MEDIUM"]))
    assert set(physical["STABLE_MEDIUM"]).issubset(set(physical["ALL_YEAR_BLACK"]))
    annual=[]
    for policy in RULES:
        for yearname in (*YEARS,"2023-2025"):
            entries=[(key,v) for key,v in physical[policy].items()
                     if yearname=="2023-2025" or key[0]==yearname]
            total=new_counter()
            bought=set()
            hit_races=set()
            for (y,rid,ticket),ret in entries:
                bought.add((y,rid))
                if ret>0:hit_races.add((y,rid))
                update_counter(total,ret)
            annual.append(dict(
                policy=policy,period=str(yearname),
                line_gate_pairs=len(kept[policy]),
                unique_original_lines=len({t for t,g in kept[policy]}),
                bought_races=len(bought),hit_races=len(hit_races),
                **metric(total)))
    return annual


def write(path,rows):
    assert rows
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    for name in ("predictions","picks","racewise","backfill-root",
                 "market25-black-lines","baseline-raw","market25-pruned",
                 "out-dir"):
        p.add_argument("--"+name,required=True)
    a=p.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    selected=load_selected(Path(a.market25_black_lines))
    expected=load_line_baselines(Path(a.baseline_raw),Path(a.market25_pruned),selected)
    data,picks=lens.frame(a.predictions,a.picks)
    races=pd.read_csv(a.racewise,compression="infer",dtype={
        "race_id":str,"VALUE_EV_horse":str,"AI_RAW_TOP_horse":str})
    races["year"]=pd.to_numeric(races["year"],errors="raise").astype(int)
    assert_ok(len(races)==TOTAL_RACES and races.race_id.is_unique)
    assert set(races.year)==set(YEARS)
    observed,events,counts,full=main_pass(data,picks,races,Path(a.backfill_root),selected)
    audit_source(counts,expected,full)
    kept,audits=select_stable(observed,selected)
    summary=dedup_portfolios(events,kept)
    assert len(summary)==12
    write(out/"medium_strict_dedup_portfolios_by_year.csv",summary)
    write(out/"all_market25_black_235_lines_x_17_gates_stability_decisions.csv",audits)
    survivors=[r for r in audits if r["medium_decision"]=="ACCEPT" or r["strict_decision"]=="ACCEPT"]
    # A 'zero retained' scenario is a valid result, with explanations in audit.
    if survivors:write(out/"accepted_stable_lines_and_gates.csv",survivors)
    else:(out/"accepted_stable_lines_and_gates.csv").write_text(
        "line,gate,medium_decision,strict_decision\n",encoding="utf-8")
    comparison=[r for r in summary if r["period"]=="2023-2025"]
    comparison.sort(key=lambda r: RULES.index(r["policy"]))
    output=dict(
        contract="RAW_TOP1_MARKET25_OFFICIAL_L21_STABILITY_V1",
        base_policy="MARKET_25",
        raw_rank1_axis=True,
        source_lines_three_year_black=len(selected),
        year2023_method="BASE_UNPRUNED_TRAINING_YEAR",
        year2024_method="MARKET_25_PRIOR_2023",
        year2025_method="MARKET_25_PRIOR_2023_2024",
        official_gate_rules=GATES,
        official_stability=CONFIG,
        each_year_positive_requirement=True,
        selected_line_gate_counts={p:len(kept[p]) for p in RULES},
        unique_original_line_counts={p:len({t for t,g in kept[p]}) for p in RULES},
        summary=comparison,
        exact_market25_physical_2024_2025_verified=True,
        exact_235_original_line_yearly_verified=True,
        physical_dedup=True,
        same_period_hindsight_selection_bias=True,
        pricing="FINAL_POST_CLOSE_NOT_PREBET",
        year2026_sealed=True,
        no_frozen_L21_or_main_mutation=True,
        stable_strict_subset_of_medium=True,
        historical_input_read_once=True,
        temporary_only_selected_ticket_events=True)
    (out/"summary.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("RAW_MARKET25_OFFICIAL_STABILITY_COMPLETE",json.dumps({
        "candidate_black_lines":len(selected),
        "selected_line_gates":output["selected_line_gate_counts"],
        "results":comparison,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":main()
