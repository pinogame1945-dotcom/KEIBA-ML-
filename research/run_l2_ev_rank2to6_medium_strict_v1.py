#!/usr/bin/env python3
"""Frozen Medium/Strict evaluation for EV ranks 2-6 after winner pruning."""

from __future__ import annotations
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path
import pandas as pd
import run_l2_fixed_ticket_pruning_3040_v1 as fixed
import run_l2_raw_market25_stability_v1 as official
import run_l2_raw_top_ticket_pruning_oos_v1 as prior
import run_l2_ev_rank2to6_pruning20_best_roi_v1 as ev
import run_l2_all_fixed_odds_band_lenses_v1 as lens
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,assert_ok

GATES=tuple(lens.GATES)
RULES=tuple(official.RULES)
assert GATES==official.GATES and len(GATES)==17
assert RULES==("ALL_YEAR_BLACK","STABLE_MEDIUM","STABLE_STRICT")
assert official.CONFIG["STABLE_MEDIUM"]==dict(n=500,hits=30,per_year_hits=8,max1=.40,max3=.70,minus3_roi=90)
assert official.CONFIG["STABLE_STRICT"]==dict(n=800,hits=50,per_year_hits=12,max1=.30,max3=.55,minus3_roi=100)

def source_rows(path):
    f=gzip.open if str(path).endswith(".gz") else open
    with f(path,"rt",encoding="utf-8-sig",newline="") as handle:
        yield from csv.DictReader(handle)

def key(s):
    a=s.split("|")
    assert_ok(len(a)==4,"unexpected source line "+s)
    return a[0],int(a[1]),a[2],a[3]

def write_csv(path,rows,headers=None):
    rows=list(rows)
    if headers is None:
        assert_ok(rows,"empty csv "+str(path))
        headers=list(rows[0])
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=headers,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)

def winners(root,rank):
    src=root/("rank-"+str(rank))
    meta=json.loads((src/"summary.json").read_text())
    policy=meta["adopted_policy"]
    assert_ok(meta["ev_rank"]==rank and policy in fixed.POLICIES[1:],"rank/policy drift")
    assert_ok(meta["evaluation_2023_start"]==ev.EVAL_START_2023,"Jan warmup date drift")
    selected={}
    for r in source_rows(src/"adopted_method_black_odds_lines.csv"):
        assert_ok(r["policy"]==policy and int(r["ev_rank"])==rank,"policy mismatch")
        tid=key(r["line"])
        assert_ok(tid not in selected and tid in ev.LINE_KEYS,"duplicate/invalid line")
        n,h,profit=map(int,(r["tickets"],r["hits"],r["profit_yen"]))
        assert_ok(n>0 and profit>0,"not a pooled profitable line")
        selected[tid]=(n,h,profit+100*n)
    assert_ok(len(selected)==meta["adopted_portfolio"]["selected_black_lines"],
              "number of source lines changed")
    expected={}
    for r in source_rows(src/"all_20_policies_3040_lines_3years.csv.gz"):
        if r["policy"]!=policy or r["year"] not in ("2023","2024","2025"):
            continue
        tid=key(r["line"])
        if tid in selected:
            k=(int(r["year"]),tid)
            assert_ok(k not in expected,"duplicate source-year key")
            expected[k]=tuple(int(r[f]) for f in ("tickets","hits","payout_yen"))
    assert_ok(len(expected)==3*len(selected),"missing source-year rows")
    for tid,x in selected.items():
        observed=tuple(sum(expected[y,tid][i] for y in YEARS) for i in range(3))
        assert_ok(observed==x,("source line pooled mismatch",tid))
    return meta,policy,selected,expected

def replay(data,picks,races,root,rank,policy,selected):
    """One real-data pass; learn thresholds only from the prior period."""
    history=[]
    stat=defaultdict(lambda:[0,0,0.])
    hist={y:defaultdict(official.new_counter) for y in YEARS}
    events=[]
    counted=defaultdict(official.new_counter)
    before={y:official.new_counter() for y in YEARS}
    coverage={}
    for year in YEARS:
        yr=races[races.year.eq(year)]
        assert_ok(len(yr)==RACES_YEAR,"year race count mismatch")
        group=data[data.year.eq(year)].groupby("_race_id",sort=False)
        prior_limits=prior.fast_prior_thresholds(history,stat) if year>2023 else None
        prior_stat=stat if year>2023 else None
        train_history=[]
        train_stats=defaultdict(lambda:[0,0,0.])
        jan_history=[]
        jan_stats=defaultdict(lambda:[0,0,0.])
        jan=0;eval_races=0;accepted_events=0
        for rid,pack,odds,g,info in ev.iterate_races(year,group,yr,root):
            eligible=ev.pruning_eval_eligible(year,info["race_date"])
            if eligible and year==2023 and prior_limits is None:
                assert_ok(jan>0 and len(jan_history)>1000,"missing January source training")
                prior_limits=prior.fast_prior_thresholds(jan_history,jan_stats)
                prior_stat={k:tuple(v) for k,v in jan_stats.items()}
                print("STABILITY_2023_THRESHOLDS",rank,jan,len(jan_history),flush=True)
            jan+=int(not eligible)
            eval_races+=int(eligible)
            axis=ev.choose_axis(rid,g,info,picks,rank)
            if axis is None:continue
            shim=info.copy();shim["VALUE_EV_horse"]=axis
            unique={}
            chosen=defaultdict(dict)
            for tid,ticket,card in fixed.create_cards(g,pack,odds,rid,axis,shim):
                old=unique.get(ticket)
                if old is None:unique[ticket]=card
                else:assert_ok(old==card,("ticket card drift",rid,ticket))
                if eligible and tid in selected:
                    assert_ok(ticket not in chosen[tid],"same line duplicate ticket")
                    chosen[tid][ticket]=card
            if eligible and chosen:
                passed={}
                sourced={}
                for tid,cards in chosen.items():
                    for ticket,card in cards.items():
                        if ticket not in passed:
                            passed[ticket]=bool(fixed.mask(card,prior_stat,prior_limits)[policy])
                        if not passed[ticket]:continue
                        ret=int(card[1])
                        if ticket in sourced:
                            assert_ok(sourced[ticket]==ret,"source repeated payout mismatch")
                        else:sourced[ticket]=ret
                        official.update_counter(counted[year,tid],ret)
                if sourced:
                    for ret in sourced.values():
                        official.update_counter(before[year],ret)
                    horse,numbers=official.stable.make_horse(g,pack)
                    axis_number=numbers[axis]
                    votes={}
                    for tid,cards in chosen.items():
                        for ticket,card in cards.items():
                            if not passed.get(ticket):continue
                            ret=int(card[1])
                            if ticket not in votes:
                                flags=lens.gate_flags(ticket,horse,axis_number)
                                assert_ok(set(flags)==set(GATES),"17 gate drift")
                                votes[ticket]=sum(1<<i for i,k in enumerate(GATES) if flags[k])
                                assert_ok(votes[ticket]&1,"ALL gate missing")
                            bm=votes[ticket]
                            events.append((year,rid,tid,ticket,ret,bm))
                            accepted_events+=1
                            for i,k in enumerate(GATES):
                                if bm&(1<<i):
                                    official.update_counter(hist[year][tid,k],ret)
            if year<2025:
                for card in unique.values():
                    fixed.add_learning(train_stats,card)
                    train_history.append(card)
                    if not eligible:
                        fixed.add_learning(jan_stats,card)
                        jan_history.append(card)
        assert_ok(eval_races+jan==RACES_YEAR,"evaluated day coverage")
        if year==2023:assert_ok(jan==312 and prior_limits is not None,"Jan training incomplete")
        else:assert_ok(jan==0,"unexpected later-year exclusion")
        if year<2025:
            history.extend(train_history)
            fixed.merge(stat,train_stats)
        coverage[str(year)]=dict(eval_races=eval_races,jan_training_only=jan,events=accepted_events)
        print("STABILITY_RANK_YEAR_READY",rank,year,"source_events",accepted_events,flush=True)
    return hist,events,counted,before,coverage

def validate_source(counted,expected,before,reference,rank):
    for k,v in expected.items():
        got=counted.get(k,official.new_counter())
        observed=tuple(got[x] for x in ("n","hits","payout"))
        assert_ok(observed==v,("line source receipt drift",rank,k,observed,v))
    observed=tuple(sum(x[k] for x in before.values()) for k in ("n","hits","payout"))
    old=reference["adopted_portfolio"]
    expected_union=tuple(old[k] for k in ("n","hits","payout"))
    assert_ok(observed==expected_union,("prior pruned union changed",rank,observed,expected_union))

def analyze(rank,source_meta,policy,black_lines,hist,events,before,coverage,out):
    chosen,decisions=official.select_stable(hist,black_lines)
    assert_ok(chosen["STABLE_STRICT"].issubset(chosen["STABLE_MEDIUM"]),
              "frozen stability relation drift")
    filtered=official.dedup_portfolios(events,chosen)
    assert_ok(len(filtered)==12,"all years and policies missing")
    before_rows=[]
    for year in (*YEARS,"2023-2025"):
        if year=="2023-2025":
            st=official.new_counter()
            for prior_st in before.values():
                st["n"]+=prior_st["n"]
                st["hits"]+=prior_st["hits"]
                st["payout"]+=prior_st["payout"]
        else:st=before[year]
        before_rows.append(dict(ev_rank=rank,policy="PRUNED_BLACK_SOURCE",
                                period=str(year),line_gate_pairs=0,
                                unique_original_lines=len(black_lines),
                                bought_races=None,hit_races=None,**official.metric(st)))
    all_rows=before_rows+[dict(ev_rank=rank,**x) for x in filtered]
    write_csv(out/"source_all_year_black_medium_strict_by_year.csv",all_rows)
    audit=[dict(ev_rank=rank,fixed_pruner=policy,**d) for d in decisions]
    write_csv(out/"full_17gate_stability_decisions.csv",audit)
    accepted=[d for d in audit if d["medium_decision"]=="ACCEPT"]
    write_csv(out/"accepted_medium_strict_line_gates.csv",accepted,
              list(audit[0]) if audit else ["ev_rank","fixed_pruner","line","gate"])
    totals={r["policy"]:r for r in all_rows if r["period"]=="2023-2025"}
    result=dict(contract="EV_RANK2TO6_AFTER_PRUNE_STABLE_MEDIUM_STRICT_V1",
        ev_rank=rank,adopted_pruner=policy,
        source_run_id=38891800300,source_original_black_lines=len(black_lines),
        pruning_source_counts_and_payouts_exactly_reconciled=True,
        frozen_gates=list(GATES),frozen_rules=official.CONFIG,
        accepted_line_gate_pairs={p:len(chosen[p]) for p in RULES},
        accepted_original_lines={p:len({tid for tid,g in chosen[p]}) for p in RULES},
        years=coverage,three_year_results=totals,
        limits="Jan 2023 excluded common training-only; Feb-Dec 2023 + full 2024/25 scored",
        pruning_policy_not_reselected=True,
        same_sample_post_hoc_profit_selection=True,
        final_historical_odds_not_bet_time_quotes=True,
        frozen_L21_unchanged=True,year_2026_sealed=True)
    (out/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("EV_RANK_PRUNED_MEDIUM_STRICT_COMPLETE",
          json.dumps({"rank":rank,"policy":policy,"accepted":result["accepted_line_gate_pairs"],
                      "totals":totals},ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    for a in ("rank","predictions","picks","racewise","backfill-root","winner-root","out-dir"):
        p.add_argument("--"+a,required=True)
    args=p.parse_args()
    rank=int(args.rank)
    assert_ok(rank in ev.RANKS,"rank must be 2..6")
    meta,policy,black,expected=winners(Path(args.winner_root),rank)
    output=Path(args.out_dir);output.mkdir(parents=True,exist_ok=True)
    data,picks=lens.frame(args.predictions,args.picks)
    races=pd.read_csv(args.racewise,compression="infer",
                      dtype={"race_id":str,"VALUE_EV_horse":str,"AI_RAW_TOP_horse":str})
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert_ok(len(races)==TOTAL_RACES and races.race_id.is_unique,"race source coverage")
    assert_ok(set(races.year)==set(YEARS),"2026 sealed")
    hist,events,counted,before,coverage=replay(
        data,picks,races,Path(args.backfill_root),rank,policy,black)
    validate_source(counted,expected,before,meta,rank)
    analyze(rank,meta,policy,black,hist,events,before,coverage,output)

if __name__=="__main__":main()
