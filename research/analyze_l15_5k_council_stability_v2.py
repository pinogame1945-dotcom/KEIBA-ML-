#!/usr/bin/env python3
import argparse,json,statistics
from collections import defaultdict
from pathlib import Path

SELECTION_YEARS=(2022,2023,2024)
CONFIRM_YEAR=2025
BASELINE="router_hard"
MAX_SURVIVORS=4

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--arena",required=True)
    p.add_argument("--output",required=True)
    return p.parse_args()

def mean(xs):
    return sum(xs)/len(xs) if xs else 0.0

def stdev(xs):
    return statistics.pstdev(xs) if len(xs)>1 else 0.0

def load_cells(arena):
    by_year={}
    for fold in arena["folds"]:
        y=int(fold["test_year"])
        by_year[y]=fold["strategies"]
    return by_year

def all_cells(by_year):
    cells=set()
    for y in by_year:
        for strategy,cmap in by_year[y].items():
            if strategy==BASELINE:
                cells.update(cmap)
    return sorted(cells)

def metrics_for(by_year,strategy,cell,years):
    vals=[]
    hit_gain=0
    races=0
    for y in years:
        row=by_year[y][strategy][cell]
        base=by_year[y][BASELINE][cell]
        vals.append(float(row["delta_vs_router_pp"]))
        hit_gain += int(row["hits"])-int(base["hits"])
        races += int(row["races"])
    wins=sum(1 for x in vals if x>1e-12)
    losses=sum(1 for x in vals if x<-1e-12)
    ties=len(vals)-wins-losses
    return {
        "years":list(years),
        "delta_pp_by_year":{str(y):float(by_year[y][strategy][cell]["delta_vs_router_pp"]) for y in years},
        "mean_delta_pp":mean(vals),
        "median_delta_pp":statistics.median(vals),
        "min_delta_pp":min(vals),
        "max_delta_pp":max(vals),
        "stdev_delta_pp":stdev(vals),
        "wins":wins,
        "ties":ties,
        "losses":losses,
        "net_hit_gain":hit_gain,
        "races":races,
    }

def rank_key(item):
    s,m=item
    # Favour repeated wins first, then average gain, then downside stability.
    return (-m["wins"], -m["mean_delta_pp"], -m["min_delta_pp"], m["stdev_delta_pp"], s)

def main():
    a=parse_args()
    arena=json.loads(Path(a.arena).read_text(encoding="utf-8"))
    if arena.get("contract")!="L15_5K_COUNCIL_ARENA_V1":
        raise ValueError("unexpected arena contract")

    by_year=load_cells(arena)
    needed=set(SELECTION_YEARS+(CONFIRM_YEAR,))
    if not needed.issubset(by_year):
        raise ValueError(f"missing arena years: need={sorted(needed)} got={sorted(by_year)}")

    strategies=list(arena["strategies"])
    cells=all_cells(by_year)
    out_cells={}
    overall=defaultdict(lambda:{"selected_cells":0,"confirmed_positive":0,"confirmed_nonnegative":0,"confirmed_negative":0})

    for cell in cells:
        selection={}
        for strategy in strategies:
            selection[strategy]=metrics_for(by_year,strategy,cell,SELECTION_YEARS)

        eligible=[]
        for strategy,m in selection.items():
            if strategy==BASELINE:
                continue
            # Fixed before looking at 2025: majority-positive and positive mean over 2022-2024.
            if m["wins"]>=2 and m["mean_delta_pp"]>0:
                eligible.append((strategy,m))
        eligible.sort(key=rank_key)
        survivors=[s for s,_ in eligible[:MAX_SURVIVORS]]

        confirm={}
        for strategy in [BASELINE]+survivors:
            row=by_year[CONFIRM_YEAR][strategy][cell]
            confirm[strategy]={
                "delta_vs_router_pp":float(row["delta_vs_router_pp"]),
                "hit_rate":float(row["hit_rate"]),
                "hits":int(row["hits"]),
                "races":int(row["races"]),
                "confirmed_positive":float(row["delta_vs_router_pp"])>1e-12,
                "confirmed_nonnegative":float(row["delta_vs_router_pp"])>=-1e-12,
            }
            if strategy!=BASELINE:
                overall[strategy]["selected_cells"]+=1
                d=float(row["delta_vs_router_pp"])
                if d>1e-12:
                    overall[strategy]["confirmed_positive"]+=1
                elif d>=-1e-12:
                    overall[strategy]["confirmed_nonnegative"]+=1
                else:
                    overall[strategy]["confirmed_negative"]+=1

        promoted=[
            s for s in survivors
            if confirm[s]["confirmed_nonnegative"]
        ]
        strict_promoted=[
            s for s in survivors
            if confirm[s]["confirmed_positive"]
        ]

        out_cells[cell]={
            "selection_years":list(SELECTION_YEARS),
            "confirm_year":CONFIRM_YEAR,
            "selection_metrics":selection,
            "eligible_before_confirm":[s for s,_ in eligible],
            "survivors_before_2025":survivors,
            "confirm_2025":confirm,
            "promoted_nonnegative_2025":promoted,
            "promoted_positive_2025":strict_promoted,
            "baseline_only_if_none":len(promoted)==0,
        }

    out={
        "contract":"L15_5K_COUNCIL_STABILITY_V2",
        "source_contract":arena["contract"],
        "business_objective":{
            "primary":"profit_maximization",
            "interpretation":"L1.5 hit-rate/candidate-capture improvements are intermediate signals only.",
            "promotion_rule":"No Council strategy is production-final from L1.5 metrics alone. Survivors must later pass downstream L2/L3 edge/EV/ROI evaluation under bankroll constraints.",
            "odds_scope":"No odds are used in this L1.5 stability study; odds remain an L2/L3 input.",
        },
        "selection_protocol":{
            "selection_years":list(SELECTION_YEARS),
            "confirm_year":CONFIRM_YEAR,
            "eligibility":"2022-2024: positive delta vs router in at least 2 of 3 years AND positive mean delta.",
            "max_survivors_per_cell":MAX_SURVIVORS,
            "ranking":"wins desc, mean delta desc, worst-year delta desc, volatility asc, strategy name",
            "2025_policy":"2025 is not used to select survivors; it is confirmation only.",
            "production_promotion":"not allowed at this stage",
            "locked_years":[2026],
        },
        "cells":out_cells,
        "strategy_confirmation_summary":dict(sorted(overall.items())),
        "next_stage":{
            "name":"Council Router V3",
            "input":"Only per-cell survivors from this V2 plus router_hard baseline.",
            "goal":"Learn which council method to use for each race from disagreement/consensus/race-context features.",
            "final_goal":"Pass improved candidate sets to L2/L3 and optimize realized ROI/EV, not L1.5 accuracy for its own sake.",
        },
        "notes":[
            "2026 is never read.",
            "No odds used.",
            "The 2025 confirmation is protected from V2 survivor selection.",
            "A positive L1.5 delta does not imply betting profitability.",
        ],
    }

    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("L15_5K_COUNCIL_STABILITY_V2_OK")
    for cell,c in sorted(out_cells.items()):
        surv=",".join(c["survivors_before_2025"]) or "NONE"
        pos=",".join(c["promoted_positive_2025"]) or "NONE"
        nonneg=",".join(c["promoted_nonnegative_2025"]) or "NONE"
        print("CELL",cell,
              "survivors",surv,
              "positive2025",pos,
              "nonnegative2025",nonneg)
        for s in c["survivors_before_2025"]:
            sm=c["selection_metrics"][s]
            cm=c["confirm_2025"][s]
            print("SURVIVOR",cell,s,
                  "sel_mean_pp",round(sm["mean_delta_pp"],3),
                  "sel_wtl",f'{sm["wins"]}/{sm["ties"]}/{sm["losses"]}',
                  "sel_worst_pp",round(sm["min_delta_pp"],3),
                  "confirm2025_pp",round(cm["delta_vs_router_pp"],3))

if __name__=="__main__":
    main()
