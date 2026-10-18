#!/usr/bin/env python3
"""L2.1 research gate reuse without ANY ticket odds or market gate.
Exact EV axes 1..6 x pure-AI numbered 6,870 bets, all 2023..25.
Existing named gate definitions are reproduced; no model retraining.
Original EV axes were calculated using FINAL historical win odds (existing
research prerequisite) and are UNCHANGED. This experiment only removes
all odds/market-based PRUNING GATES. Stable frozen L2.1 contract unchanged.
"""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_l2_ai_market_rank_betfrontier_v1 import load_predictions
from run_l2_value_anchor_bet_frontier_v1 import payouts
from build_l2_market_helpers_v1 import load_day, horse_number_map
from run_all_race_flow_ticket_audit_v1 import YEARS, RACES_YEAR, TOTAL_RACES, assert_ok
from aggregate_ev_top6_pure_ai_rank_tickets_v1 import winner_to_template, FAMILIES

# Exactly the existing historical *non-price, non-market* per-ticket gate names
# from research/run_l2_all_fixed_odds_band_lenses_v1.py:gate_flags.
# Deliberately exclude MARKET_*, MARKET_UNDERVALUE_*, AGREEMENT_*,
# FAIR_RATIO_*, RISK_* (market-correction risk), and any ticket-odds gate.
GATES=("ALL", "RAW_TOP6", "RAW_TOP10", "CALIBRATED_TOP6",
       "CALIBRATED_TOP10", "ABILITY_Z_GE_M1", "ABILITY_Z_GE_0",
       "MODEL_WIN_P_GE_003")
YEARS_STR=("2023","2024","2025")
MAX_AI_RANK=18
FAMILY_INDEX={f:i for i,f in enumerate(FAMILIES)}
GATE_INDEX={v:i for i,v in enumerate(GATES)}
FAMILY_DOUBLES=("QUINELLA","WIDE","EXACTA_FIRST","EXACTA_SECOND")
FAMILY_TRIPLES=("TRIO","TRIFECTA_FIRST","TRIFECTA_SECOND","TRIFECTA_THIRD")


def reference(path):
    annual={}
    with open(path,encoding="utf-8",newline="") as fd:
        for row in csv.DictReader(fd):
            key=(row["year"],int(row["ev_axis_rank"]),row["family"],
                 int(row["raw_ai_rank_1"]),int(row["raw_ai_rank_2"]))
            assert_ok(key not in annual,f"reference duplicate {key}")
            annual[key]=(int(row["purchased_tickets"]),
                         int(row["hit_tickets"]),round(float(row["return_yen"])))
    assert_ok(sum(k[0]=="2023-2025" for k in annual)==6870,
              "original 6,870 settings missing")
    return annual


def emit(path, rows):
    assert rows
    opener=(gzip.open if str(path).endswith(".gz") else open)
    with opener(path,"wt",newline="",encoding="utf-8") as fd:
        w=csv.DictWriter(fd,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def flags_per_horse(g, no):
    """No market rank, betting-ticket price, or race outcome is read here."""
    raw=dict(zip(g["_horse_id"].astype(str),g["distill_ai_rank"].astype(int)))
    ability=dict(zip(g["_horse_id"].astype(str),g["distill_score_z"].astype(float)))
    p=dict(zip(g["_horse_id"].astype(str),g["P__ANCHOR_FULL"].astype(float)))
    # Same descending P and stable horse ID tie semantics; the reference
    # market-rank secondary tie is intentionally excluded to avoid market gates.
    psort=sorted(p,key=lambda h:(-p[h],h))
    pr={h:i for i,h in enumerate(psort,1)}
    result={}
    for h,n in no.items():
        assert_ok(h in raw and h in ability and h in pr,"missing AI predictor")
        result[n]=np.asarray((
            True,
            raw[h]<=6,
            raw[h]<=10,
            pr[h]<=6,
            pr[h]<=10,
            ability[h]>=-1.,
            ability[h]>=0.,
            p[h]>=.03),dtype=np.bool_)
    return result


def run_year(yi, data, root, counts, wins, returns, metrics):
    y=YEARS[yi]
    frame=data[data.year.eq(y)]
    grps=frame.groupby("_race_id",sort=False)
    dates=frame[["_race_id","_race_date"]].drop_duplicates()
    assert_ok(dates["_race_id"].is_unique and len(dates)==RACES_YEAR,
              f"race count drift {y}")
    byday=dates.groupby("_race_date")["_race_id"].apply(list)
    for day, rids in byday.items():
        packs=load_day(root/"data"/"daily"/(str(day)+".jsonl.gz"),set(rids))
        for rid in rids:
            pack=packs.get(rid)
            assert_ok(pack is not None,f"missing settled pack {rid}")
            g=grps.get_group(rid)
            no=horse_number_map(pack)
            horses=set(g["_horse_id"].astype(str))
            assert_ok(horses.issubset(no),"unmapped frozen horses")
            assert_ok(len(horses)==len(g),"duplicated prediction horse")
            rank=g.sort_values(["distill_ai_rank","market_rank","_horse_id"],
                     ascending=[True,True,True],kind="mergesort")[
                         "_horse_id"].astype(str).tolist()
            numbers=[no[h] for h in rank]
            assert_ok(len(numbers)==len(set(numbers)),"horse number collision")
            rank_of={n:i for i,n in enumerate(numbers,1)}
            ef=g.assign(__v=g["P__ANCHOR_FULL"]*g["final_win_odds"])
            eids=ef.sort_values(["__v","P__ANCHOR_FULL","market_rank","_horse_id"],
                        ascending=[False,False,True,True],kind="mergesort")[
                        "_horse_id"].astype(str).tolist()
            assert_ok(set(eids)==set(rank),"EV/pure AI drift")
            horse_flags=flags_per_horse(g,no)
            pay=payouts(pack,rid) # settlement is used ONLY in wins/returns, not gates
            for er,axis in enumerate(eids[:6],1):
                ai=num_axis= no[axis]
                er_idx=er-1
                gate_vector=np.zeros((len(GATES),MAX_AI_RANK+1),dtype=np.bool_)
                for i,n in enumerate(numbers,1):
                    if n != num_axis:
                        gate_vector[:,i]=horse_flags[n]
                axis_vector=horse_flags[num_axis]
                for gi in range(len(GATES)):
                    f=gate_vector[gi].copy()
                    ixs=np.flatnonzero(f)
                    single=int(axis_vector[gi])
                    for kind in ("WIN","PLACE"):
                        counts[yi,er_idx,gi,FAMILY_INDEX[kind],0,0]+=single
                    for kind in FAMILY_DOUBLES:
                        counts[yi,er_idx,gi,FAMILY_INDEX[kind],ixs,0]+=1
                    matrix=np.logical_and(f[:,None],f[None,:]).astype(np.int32)
                    np.fill_diagonal(matrix,0)
                    counts[yi,er_idx,gi,FAMILY_INDEX["TRIO"]] += np.triu(matrix,1)
                    for kind in ("TRIFECTA_FIRST","TRIFECTA_SECOND","TRIFECTA_THIRD"):
                        counts[yi,er_idx,gi,FAMILY_INDEX[kind]] += matrix
                for (bet,combo),amt in pay.items():
                    tpl=winner_to_template(num_axis,no,rank_of,bet,combo)
                    if tpl is None:
                        continue
                    fam,j,k=tpl
                    fi=FAMILY_INDEX.get(fam)
                    assert_ok(fi is not None,f"unknown paid family {fam}")
                    partner_numbers=([combo[0]] if fam in FAMILY_DOUBLES else
                        [n for n in combo if n!=num_axis])
                    for gi in range(len(GATES)):
                        ok=bool(axis_vector[gi]) if fam in ("WIN","PLACE") else all(
                            horse_flags[n][gi] for n in partner_numbers)
                        if ok:
                            wins[yi,er_idx,gi,fi,j,k]+=1
                            returns[yi,er_idx,gi,fi,j,k]+=int(round(amt))
                metrics["axis_evaluations"]+=1
            metrics["races"]+=1
    assert_ok(metrics["races"]==RACES_YEAR,f"year incomplete {y} {metrics['races']}")
    print("NO_ODDS_GATES_YEAR_COMPLETE",y,"races",metrics["races"],flush=True)


def summarize(source,counts,wins,returns):
    detail=[]; pooled=[]
    for (year, er, fam, j, k), base in source.items():
        if year=="2023-2025":
            continue
        yi=YEARS_STR.index(year)
        fidx=FAMILY_INDEX[fam]
        for gi,gate in enumerate(GATES):
            n=int(counts[yi,er-1,gi,fidx,j,k])
            h=int(wins[yi,er-1,gi,fidx,j,k])
            ret=int(returns[yi,er-1,gi,fidx,j,k])
            assert_ok(0<=h<=n<=base[0],f"invalid subset {year} {er} {fam} {j} {k} {gate}")
            assert_ok(0<=ret<=base[2],f"payout not subset {year} {er} {fam} {j} {k} {gate}")
            if gi==0:
                assert_ok((n,h,ret)==base,
                     f"EXACT_PREVIOUS_6870_BASELINE_DRIFT {year} {er} {fam} {j} {k}: {(n,h,ret)} != {base}")
            bN,bH,bR=base
            before_profit=bR-bN*100
            after_profit=ret-n*100
            detail.append({
                "year":year,"ev_axis_rank":er,"family":fam,
                "raw_ai_rank_1":j,"raw_ai_rank_2":k,"gate":gate,
                "before_tickets":bN,"before_hits":bH,
                "before_hit_pct":round(100*bH/bN,6) if bN else None,
                "before_stake_yen":bN*100,"before_return_yen":bR,
                "before_profit_yen":before_profit,
                "before_roi_pct":round(100*bR/bN,6) if bN else None,
                "after_tickets":n,"after_hits":h,
                "after_hit_pct":round(100*h/n,6) if n else None,
                "after_stake_yen":n*100,"after_return_yen":ret,
                "after_profit_yen":after_profit,
                "after_roi_pct":round(100*ret/n,6) if n else None,
                "profit_delta_yen":after_profit-before_profit,
                "red_to_black":int(before_profit<0 and after_profit>0),
                "black_to_red":int(before_profit>0 and after_profit<0),
                "zero_tickets_after":int(n==0),
            })
    indexed=defaultdict(dict)
    for row in detail:
        key=(row["ev_axis_rank"],row["family"],row["raw_ai_rank_1"],
             row["raw_ai_rank_2"],row["gate"])
        indexed[key][row["year"]]=row
    for (year,er,fam,j,k),base in source.items():
        if year!="2023-2025":
            continue
        for gate in GATES:
            rows=indexed[(er,fam,j,k,gate)]
            # 2023 is included in pooled metrics using same fixed gate,
            # separate from walk-forward training-only price pruning.
            assert_ok(set(rows)==set(YEARS_STR),f"incomplete annual group {er} {fam} {j} {k}")
            bN,bH,bR=base
            n=sum(r["after_tickets"] for r in rows.values())
            h=sum(r["after_hits"] for r in rows.values())
            ret=sum(r["after_return_yen"] for r in rows.values())
            assert_ok(sum(r["before_tickets"] for r in rows.values())==bN
                      and sum(r["before_hits"] for r in rows.values())==bH
                      and sum(r["before_return_yen"] for r in rows.values())==bR,
                      f"pooled baseline drift {er} {fam} {j} {k}")
            bProfit=bR-100*bN
            aProfit=ret-100*n
            pooled.append({
                "year":"2023-2025","ev_axis_rank":er,"family":fam,
                "raw_ai_rank_1":j,"raw_ai_rank_2":k,"gate":gate,
                "before_tickets":bN,"before_hits":bH,
                "before_hit_pct":round(100*bH/bN,6) if bN else None,
                "before_stake_yen":bN*100,"before_return_yen":bR,
                "before_profit_yen":bProfit,
                "before_roi_pct":round(100*bR/bN,6) if bN else None,
                "after_tickets":n,"after_hits":h,
                "after_hit_pct":round(100*h/n,6) if n else None,
                "after_stake_yen":n*100,"after_return_yen":ret,
                "after_profit_yen":aProfit,
                "after_roi_pct":round(100*ret/n,6) if n else None,
                "profit_delta_yen":aProfit-bProfit,
                "red_to_black":int(bProfit<0 and aProfit>0),
                "black_to_red":int(bProfit>0 and aProfit<0),
                "zero_tickets_after":int(n==0),
            })
    stats=[]
    for period, table in (
        ("2023",detail),("2024",detail),("2025",detail),
        ("2023-2025",pooled)):
        for gate in GATES:
            x=[r for r in table if r["year"]==period and r["gate"]==gate]
            stats.append({
                "period":period,"gate":gate,
                "settings":len(x),
                "before_black":sum(r["before_profit_yen"]>0 for r in x),
                "before_red":sum(r["before_profit_yen"]<0 for r in x),
                "before_zero":sum(r["before_profit_yen"]==0 for r in x),
                "after_black":sum(r["after_profit_yen"]>0 for r in x),
                "after_red":sum(r["after_profit_yen"]<0 for r in x),
                "after_zero":sum(r["after_profit_yen"]==0 for r in x),
                "red_to_black":sum(r["red_to_black"] for r in x),
                "black_to_red":sum(r["black_to_red"] for r in x),
                "no_bet_after":sum(r["zero_tickets_after"] for r in x),
                "before_tickets_not_dedup":sum(r["before_tickets"] for r in x),
                "after_tickets_not_dedup":sum(r["after_tickets"] for r in x),
                "lost_hits_not_dedup":sum(r["before_hits"]-r["after_hits"] for r in x),
                "saved_losers_not_dedup":sum(
                     (r["before_tickets"]-r["before_hits"])-
                     (r["after_tickets"]-r["after_hits"]) for r in x),
                "profit_delta_non_dedup_yen":sum(r["profit_delta_yen"] for r in x)
            })
    return detail,pooled,stats


def main():
    p=argparse.ArgumentParser()
    for a in ("predictions","backfill-root","ability-cache","atomic-reference","out-dir"):
        p.add_argument("--"+a,required=True)
    args=p.parse_args()
    pred,_=load_predictions(args.predictions,
             "research-results/all-race-one-pick-v1/run-38668174639/all-race-one-picks.csv.gz")
    assert_ok(set(pred.year)==set(YEARS),"unexpected year (2026 sealed)")
    ability=pd.read_csv(args.ability_cache,compression="infer",
          usecols=["_race_id","_horse_id","distill_score_z"],
          dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    assert_ok(not ability.duplicated(["_race_id","_horse_id"]).any(),"duplicate ability")
    before=len(pred)
    pred=pred.merge(ability,on=["_race_id","_horse_id"],
                    how="left",validate="one_to_one")
    assert_ok(len(pred)==before and pred["distill_score_z"].notna().all(),
              "missing original ability features")
    original=reference(args.atomic_reference)
    shape=(3,6,len(GATES),len(FAMILIES),19,19)
    counts=np.zeros(shape,dtype=np.int32)
    hits=np.zeros(shape,dtype=np.int32)
    returns=np.zeros(shape,dtype=np.int64)
    meta={}
    for yi in range(3):
        m=Counter()
        run_year(yi,pred,Path(args.backfill_root),counts,hits,returns,m)
        meta[YEARS_STR[yi]]=dict(m)
    detail,pooled,stats=summarize(original,counts,hits,returns)
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    emit(out/"all-6870-odds-free-gates-yearly.csv.gz",detail)
    emit(out/"all-6870-odds-free-gates-pooled.csv.gz",pooled)
    emit(out/"red-black-transition-by-gate-and-year.csv",stats)
    (out/"summary.json").write_text(json.dumps({
        "contract":"L2_EV_TOP6_ALL_6870_ODDS_FREE_GATE_EVALUATION_V1",
        "gate_names":list(GATES),
        "gate_origin":"research/run_l2_all_fixed_odds_band_lenses_v1.py:gate_flags",
        "gate_exclusions":"market, fair ratio, risk, ticket-odds bands, any learned price feature",
        "historical_EV_axis_warning":"EV anchor still uses historical FINAL win odds; no change",
        "gate_scoring_odds_free":True,
        "ticket_odds_data_loaded":False,
        "payouts_for_scoring_only":True,
        "2026_sealed":True,
        "L2_1_frozen_unchanged":True,
        "all_6870_baselines_exact_parity":True,
        "model_ranking_reused":True,
        "calibrated_probability_tie_break":"horse id, not market popularity",
        "years":["2023","2024","2025"],
        "metrics":meta,
        "transition_table":stats,
        "warning":"Selecting highest ROI gates based on 2023-25 is retrospective in-sample, not OOS; no physical-bet dedup across 6870 strategies."
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("NO_ODDS_PRUNE_COMPLETE",
          json.dumps([r for r in stats if r["period"]=="2023-2025"],
          ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
