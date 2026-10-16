#!/usr/bin/env python3
"""L2.1 STABLE_MEDIUM/STRICT on the best of 20 fair RAW-rank-1 prune policies.

2023 Jan is common prior-only calibration and EXCLUDED from scored races.
2023 Feb-Dec, 2024, 2025 have non-leaking ticket reduction; the same-year
all-year-black source-line and stability selection remains post-hoc descriptive.
Use literal original 17 gate definitions, L2.1 stability numbers, physical dedup.
No change to the frozen EV-axis L2.1 policy and no 2026 results.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import run_l2_fixed_ticket_pruning_3040_v1 as fixed
import run_l2_raw_top_odds_boundary_v1 as raw
import run_l2_all_fixed_odds_band_lenses_v1 as lens
import run_l2_raw_market25_stability_v1 as official
from build_l2_market_helpers_v1 import load_day,load_odds_day
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,assert_ok

START="2023-02-01"
GATES=official.GATES
RULES=official.RULES
CONFIG=official.CONFIG
assert len(GATES)==17 and len(CONFIG)==2

def csvrows(path):
    with (gzip.open(path,"rt",encoding="utf-8-sig",newline="") if str(path).endswith(".gz")
          else open(path,"r",encoding="utf-8-sig",newline="")) as f:
        yield from csv.DictReader(f)

def write_csv(path,rows):
    assert rows
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def choose_source(fair_root,old24_25_path):
    meta=json.loads((fair_root/"summary.json").read_text(encoding="utf-8"))
    policy=meta["best_method"]
    assert policy in fixed.POLICIES and policy!="BASE"
    rank=next(i for i,r in enumerate(meta["policy_ranks"],1)
              if r["method"]==policy)
    assert rank==1 or (rank==2 and meta["policy_ranks"][0]["method"]=="BASE")
    selected={}
    for r in csvrows(fair_root/"all_black_lines_by_method_fair.csv.gz"):
        if r["method"]!=policy: continue
        tid=(r["partner_order"],int(r["other_horses_k"]),r["family"],r["odds_band"])
        assert tid not in selected
        selected[tid]=dict(n=int(r["tickets"]),hits=int(r["hits"]),
                           payout=int(r["payout_yen"]),profit=int(r["profit_yen"]))
        assert selected[tid]["profit"]>0
    assert selected
    # Original source-by-year must match the selected 3-year black line table.
    expected={}
    for r in csvrows(fair_root/"2023_feb_dec_all3040_by21_policies.csv.gz"):
        if r["method"]!=policy:continue
        tid=(r["partner_order"],int(r["other_horses_k"]),r["family"],r["odds_band"])
        if tid in selected: expected[(2023,tid)]=(int(r["tickets"]),int(r["hits"]),int(r["payout_yen"]))
    for r in csvrows(old24_25_path):
        if r["policy"]!=policy:continue
        year=int(r["period"]);assert year in (2024,2025)
        tid=(r["partner_order"],int(r["other_horses_k"]),r["family"],r["odds_band"])
        if tid in selected:
            expected[(year,tid)]=tuple(int(r[x]) for x in
                    ("after_tickets","after_hits","after_payout_yen"))
    assert len(expected)==len(selected)*3
    for tid,s in selected.items():
        actual=tuple(sum(expected[(y,tid)][j] for y in YEARS) for j in range(3))
        assert actual==(s["n"],s["hits"],s["payout"]),("source 3year drift",tid)
    return policy,selected,expected,rank

def selected_thresholds(history,stats,mode,keep):
    """Frozen create_thresholds calculation, only the selected mode/retention.

    Prior years ONLY. The grouping, <100 family fallback and threshold tie
    semantics are identical to run_l2_fixed_ticket_pruning_3040_v1.py.
    """
    by_group=defaultdict(list)
    by_kind=defaultdict(list)
    for card in history:
        kind,bz=card[2][:2]
        score=fixed.calc_scores(card,stats)[mode]
        by_group[(kind,bz)].append(score)
        by_kind[kind].append(score)
    fam={kind:fixed.threshold(x,keep) for kind,x in by_kind.items()}
    out=dict(fam)
    for group,x in by_group.items():
        out[group]=fixed.threshold(x if len(x)>=100 else by_kind[group[0]],keep)
    return out

def mask_one(card,stats,thresholds,mode):
    kind,bz=card[2][:2]
    score=fixed.calc_scores(card,stats)[mode]
    return score>=thresholds.get((kind,bz),
                                  thresholds.get(kind,float("-inf")))

def historical_pass(pred,picks,races,root,selected,policy):
    mode,retention=policy.rsplit("_",1)
    keep=int(retention)
    assert mode in fixed.MODES and keep in fixed.KEEP
    full_history=[]
    full_stats=defaultdict(lambda:[0,0,0.])
    year_stats=defaultdict(lambda:[0,0,0.])
    year_history=[]
    observed=defaultdict(lambda:defaultdict(official.new_counter))
    events=[]
    counted=defaultdict(Counter)
    scored_races=Counter()
    warmup_races=0

    for year in YEARS:
        rs=races[races.year.eq(year)]
        assert_ok(len(rs)==RACES_YEAR,"race coverage drift "+str(year))
        groups=pred[pred.year.eq(year)].groupby("_race_id",sort=False)
        by_race=rs.set_index("race_id")
        days=rs.groupby("race_date")["race_id"].apply(list)
        year_history=[]
        year_stats=defaultdict(lambda:[0,0,0.])
        if year>2023:
            assert full_history
            limits=selected_thresholds(full_history,full_stats,mode,keep)
        else:
            limits={}
        jan_thresholds=None
        jan_stat_snapshot=None
        for date,ids in days.items():
            ds=str(date)[:10]
            if year==2023 and ds>=START and jan_thresholds is None:
                assert year_history and warmup_races>0
                jan_stat_snapshot=year_stats
                jan_thresholds=selected_thresholds(year_history,jan_stat_snapshot,
                                                    mode,keep)
                # Future-year training must incorporate ALL of 2023, but
                # January-only mask must remain frozen throughout Feb-Dec.
                year_stats=defaultdict(lambda:[0,0,0.])
                print("FAIR_STABILITY_2023_JAN_PRIOR_SCORES_READY",
                      "warmup",warmup_races,"cards",len(year_history),flush=True)
            packs=load_day(root/"data"/"daily"/(ds+".jsonl.gz"),set(ids))
            odds=load_odds_day(root/"data"/"odds"/"daily"/(ds+".jsonl.gz"),set(ids))
            for rid in sorted(ids):
                pack=packs.get(rid);market=odds.get(rid)
                assert_ok(pack is not None and market is not None,"race pack or odds missing "+rid)
                assert_ok(market.get("odds_kind")=="FINAL" and
                          market.get("source_status")=="result","not FINAL "+rid)
                gf=groups.get_group(rid)
                info=by_race.loc[rid]
                axis=str(info["AI_RAW_TOP_horse"])
                assert_ok(axis==str(gf.sort_values(
                    ["distill_ai_rank","market_rank","_horse_id"],
                    ascending=[True,True,True],kind="mergesort").iloc[0]["_horse_id"]),
                    "RAW axis disagreement "+rid)
                assert_ok(str(info["VALUE_EV_horse"])==picks["ANCHOR_FULL"][rid],
                          "EV predecessor pick drift "+rid)
                shim=info.copy()
                shim["VALUE_EV_horse"]=axis
                physical={}
                sources=defaultdict(dict)
                for tid,ticket,card in fixed.create_cards(gf,pack,market,rid,axis,shim):
                    if ticket in physical:
                        assert physical[ticket]==card,"physical repeated card mismatch "+rid
                    else:
                        physical[ticket]=card
                    if tid in selected:
                        assert ticket not in sources[tid],("same physical in original line",rid,tid)
                        sources[tid][ticket]=card

                # 2023 Jan is a calibration-only window for ALL 21 policies.
                eligible=(year>2023 or ds>=START)
                if eligible:
                    scored_races[year]+=1
                    mstat=jan_stat_snapshot if year==2023 else full_stats
                    mlim=jan_thresholds if year==2023 else limits
                    assert mstat is not None and mlim is not None
                    chosen={t for t,c in physical.items() if mask_one(c,mstat,mlim,mode)}
                    qualified={tid:{t:c for t,c in items.items() if t in chosen}
                               for tid,items in sources.items()}
                    qualified={k:v for k,v in qualified.items() if v}
                    if qualified:
                        horse,num=official.stable.make_horse(gf,pack)
                        anchor_no=num[axis]
                        cache={}
                        for tid,items in qualified.items():
                            for ticket,card in items.items():
                                payout=int(card[1])
                                counted[(year,tid)]["n"]+=1
                                counted[(year,tid)]["hits"]+=int(payout>0)
                                counted[(year,tid)]["payout"]+=payout
                                if ticket not in cache:
                                    flags=lens.gate_flags(ticket,horse,anchor_no)
                                    assert set(flags)==set(GATES),"L2.1 17 gate drift"
                                    cache[ticket]=sum(1<<i for i,g in enumerate(GATES)
                                                      if flags[g])
                                mask=cache[ticket]
                                assert mask&1,"L2.1 ALL gate missing"
                                events.append((year,rid,tid,ticket,payout,mask))
                                for i,g in enumerate(GATES):
                                    if mask & (1<<i):
                                        official.update_counter(observed[year][(tid,g)],payout)
                else:
                    warmup_races+=1

                # Store distinct physical candidate cards for future years.
                if year<2025:
                    for card in physical.values():
                        fixed.add_learning(year_stats,card)
                        year_history.append(card)
            if scored_races[year] and scored_races[year]%1200==0:
                print("FAIR_STABILITY_PROGRESS",year,scored_races[year],
                      "source_events",len(events),flush=True)

        if year==2023:
            assert warmup_races>0 and scored_races[year]+warmup_races==RACES_YEAR
            # Combine original January stats and Feb-Dec stats for training 2024.
            for k,val in jan_stat_snapshot.items():
                target=year_stats[k]
                for i in range(3):target[i]+=val[i]
        else:
            assert scored_races[year]==RACES_YEAR
        if year<2025:
            full_history.extend(year_history)
            fixed.merge(full_stats,year_stats)
        print("FAIR_STABILITY_YEAR_DONE",year,"evaluated_races",scored_races[year],
              "selected_line_keys",sum(y==year for y,t in counted),
              "yearly_events",sum(y==year for y,*_ in events),flush=True)
    return observed,events,counted,dict(scored_races),warmup_races

def verify_exact(counted,expected):
    for (year,tid),v in expected.items():
        r=counted.get((year,tid),Counter())
        got=(r["n"],r["hits"],r["payout"])
        assert got==v,("exact earlier fair source line disagreement",year,tid,v,got)
    print("FAIR_2023_2025_SELECTED_LINE_PAYOUT_AND_COUNTS_FULLY_MATCH",len(expected),flush=True)

def main():
    ap=argparse.ArgumentParser()
    for p in ("predictions","picks","racewise","historical",
              "fair-root","pruned24-25","out-dir"):
        ap.add_argument("--"+p,required=True)
    a=ap.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    policy,selected,expected,rank=choose_source(
        Path(a.__dict__["fair_root"]),Path(a.pruned24_25))
    pred,picks=lens.frame(a.predictions,a.picks)
    races=pd.read_csv(a.racewise,compression="infer",
                       dtype={"race_id":str,"VALUE_EV_horse":str,"AI_RAW_TOP_horse":str})
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert_ok(len(races)==TOTAL_RACES and races.race_id.is_unique,
              "full predecessor source race coverage")
    audit,events,counted,counts,warmups=historical_pass(
        pred,picks,races,Path(a.historical),selected,policy)
    verify_exact(counted,expected)
    kept,decisions=official.select_stable(audit,selected)
    portfolio=official.dedup_portfolios(events,kept)
    assert len(portfolio)==12
    # Do not count strict on top of medium; strict is a subset.
    assert kept["STABLE_STRICT"].issubset(kept["STABLE_MEDIUM"])
    write_csv(out/"medium_strict_dedup_portfolios_by_year.csv",portfolio)
    write_csv(out/"selected_best_policy_17_gates_stability_decisions.csv",decisions)
    survivors=[x for x in decisions if x["medium_decision"]=="ACCEPT"]
    if survivors:
        write_csv(out/"accepted_stable_lines_and_gates.csv",survivors)
    else:
        (out/"accepted_stable_lines_and_gates.csv").write_text(
          "line,gate,medium_decision,strict_decision\n",encoding="utf-8")
    rolled=[x for x in portfolio if x["period"]=="2023-2025"]
    summary=dict(contract="RAW_TOP1_FAIR20_RANK_FIRST_THEN_OFFICIAL_STABILITY_V1",
        chosen_method=policy,rank_among_fair_policy_result=rank,
        fair_black_source_lines=len(selected),
        source_reconciliation_all_years=3*len(selected),
        evaluated_races=counts,
        January_2023_excluded_from_evaluation=warmups,
        prior_only_pruning=True,
        official_stability=CONFIG,
        official_17_gates=GATES,
        selected_line_gate_counts={p:len(kept[p]) for p in RULES},
        selected_original_line_counts={p:len(set(x[0] for x in kept[p])) for p in RULES},
        portfolio=rolled,
        same_sample_roi_selected_black_bias=True,
        year_2026_unseen=True,
        historical_FINAL_not_prebet=True,
        no_frozen_L21_or_main_changes=True,
        standards="STABLE_STRICT subset of STABLE_MEDIUM",
    )
    (out/"stability_summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FAIR_SELECTED_POLICY_OFFICIAL_STABILITY_COMPLETE",
          json.dumps({"policy":policy,"black_lines":len(selected),
                      "kept":summary["selected_line_gate_counts"],
                      "portfolios":rolled},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
