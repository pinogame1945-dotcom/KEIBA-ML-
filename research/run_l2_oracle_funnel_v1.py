#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROUTERS=("SIMPLE_EXPECTED_ROI","STRATEGY_UTILITY","DIRECT_BET_TYPE")
YEARS=(2024,2025)
BET_TYPES=("WIN","QUINELLA","EXACTA","TRIO","TRIFECTA")


def parse_args():
    p=argparse.ArgumentParser(description="Aggregate L2 diagnostic oracle funnel without retraining.")
    p.add_argument("--diagnostics",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def as_int(v):
    try:
        return int(float(v))
    except (TypeError,ValueError):
        return 0


def as_float(v):
    try:
        return float(v)
    except (TypeError,ValueError):
        return 0.0


def read_gz_csv(path):
    with gzip.open(path,"rt",newline="",encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def segment_name(row,kind):
    gate=as_int(row.get("gate_alert"))>0
    k2=as_float(row.get("novel_ticket_share"))>0
    if kind=="ALL":
        return True
    if kind=="GATE":
        return gate
    if kind=="NON_GATE":
        return not gate
    if kind=="K2_USED":
        return k2
    if kind=="NO_K2":
        return not k2
    return False


def summarize(items,router,year,bet_type,segment):
    n=len(items)
    cand=sum(as_int(r.get("candidate_can_hit")) for r in items)
    templ=sum(as_int(r.get("template_can_hit")) for r in items)
    hit=sum(as_int(r.get("actual_hit")) for r in items)
    pos=sum(1 for r in items if as_float(r.get("actual_profit_yen"))>0)

    # Funnel must be monotonic for valid rows.
    if not (0 <= pos <= hit <= templ <= cand <= n):
        raise SystemExit(
            f"oracle funnel monotonicity broken router={router} year={year} "
            f"bet={bet_type} segment={segment} n={n} cand={cand} templ={templ} hit={hit} pos={pos}"
        )

    candidate_miss=n-cand
    template_miss=cand-templ
    edge_miss=templ-hit
    hit_nonprofit=hit-pos

    return {
        "router":router,
        "test_year":year,
        "bet_type":bet_type,
        "segment":segment,
        "selected_races":n,
        "candidate_oracle_hits":cand,
        "candidate_oracle_rate_pct":100.0*cand/n if n else None,
        "raw_template_oracle_hits":templ,
        "raw_template_rate_pct":100.0*templ/n if n else None,
        "template_retention_given_candidate_pct":100.0*templ/cand if cand else None,
        "purchased_action_hits":hit,
        "purchased_action_hit_rate_pct":100.0*hit/n if n else None,
        "edge_retention_given_template_pct":100.0*hit/templ if templ else None,
        "positive_profit_races":pos,
        "positive_profit_rate_pct":100.0*pos/n if n else None,
        "profit_retention_given_hit_pct":100.0*pos/hit if hit else None,
        "horse_selection_miss_races":candidate_miss,
        "horse_selection_miss_pct":100.0*candidate_miss/n if n else None,
        "template_miss_races":template_miss,
        "template_miss_pct":100.0*template_miss/n if n else None,
        "edge_filter_miss_races":edge_miss,
        "edge_filter_miss_pct":100.0*edge_miss/n if n else None,
        "hit_but_not_positive_races":hit_nonprofit,
        "hit_but_not_positive_pct":100.0*hit_nonprofit/n if n else None,
        "l2_recoverable_miss_races":template_miss+edge_miss,
        "l2_recoverable_miss_pct":100.0*(template_miss+edge_miss)/n if n else None,
    }


def by_template(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r["router"],int(r["test_year"]),r["bet_type"],r["template"])].append(r)
    out=[]
    for (router,year,bet,template),items in sorted(groups.items()):
        base=summarize(items,router,year,bet,"ALL")
        base["template"]=template
        out.append(base)
    return out


def main():
    a=parse_args()
    rows=read_gz_csv(a.diagnostics)
    rows=[
        r for r in rows
        if r.get("router") in ROUTERS
        and as_int(r.get("test_year")) in YEARS
        and r.get("bet_type") in BET_TYPES
        and r.get("cause")!="DATA_OR_CONTRACT_GAP"
    ]
    if not rows:
        raise SystemExit("no valid diagnostics")

    funnel=[]
    segments=("ALL","GATE","NON_GATE","K2_USED","NO_K2")
    for router in ROUTERS:
        for year in YEARS:
            rr=[r for r in rows if r["router"]==router and as_int(r["test_year"])==year]
            for bet in ("ALL",)+BET_TYPES:
                rb=rr if bet=="ALL" else [r for r in rr if r["bet_type"]==bet]
                for segment in segments:
                    items=[r for r in rb if segment_name(r,segment)]
                    if items:
                        funnel.append(summarize(items,router,year,bet,segment))

    template_rows=by_template(rows)

    # Primary DIRECT recovery table: where L2 can improve without changing L1/L1.5 candidate pool.
    recovery=[]
    for r in funnel:
        if r["router"]!="DIRECT_BET_TYPE" or r["segment"]!="ALL":
            continue
        if r["bet_type"]=="ALL":
            continue
        recovery.append({
            "test_year":r["test_year"],
            "bet_type":r["bet_type"],
            "selected_races":r["selected_races"],
            "candidate_oracle_rate_pct":r["candidate_oracle_rate_pct"],
            "raw_template_rate_pct":r["raw_template_rate_pct"],
            "purchased_action_hit_rate_pct":r["purchased_action_hit_rate_pct"],
            "horse_selection_miss_races":r["horse_selection_miss_races"],
            "template_miss_races":r["template_miss_races"],
            "edge_filter_miss_races":r["edge_filter_miss_races"],
            "l2_recoverable_miss_races":r["l2_recoverable_miss_races"],
            "l2_recoverable_miss_pct":r["l2_recoverable_miss_pct"],
        })

    # Combined 2024-2025 DIRECT, useful for prioritization.
    combined=[]
    direct=[r for r in rows if r["router"]=="DIRECT_BET_TYPE"]
    for bet in ("ALL",)+BET_TYPES:
        items=direct if bet=="ALL" else [r for r in direct if r["bet_type"]==bet]
        if not items:
            continue
        s=summarize(items,"DIRECT_BET_TYPE","2024-2025",bet,"ALL")
        combined.append(s)

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"oracle-funnel.csv",funnel)
    write_csv(out/"oracle-funnel-by-template.csv",template_rows)
    write_csv(out/"direct-recovery-potential.csv",recovery)
    write_csv(out/"direct-combined.csv",combined)

    primary=next(r for r in combined if r["bet_type"]=="ALL")
    by_bet={r["bet_type"]:r for r in combined if r["bet_type"]!="ALL"}
    priority=sorted(
        by_bet.values(),
        key=lambda r:(-r["l2_recoverable_miss_races"],r["bet_type"])
    )
    summary={
        "contract":"L2_ORACLE_FUNNEL_RESULT_V1",
        "source_loss_anatomy_run":36518374493,
        "source_loss_anatomy_commit":"8132d30d9175850441dea0ebc8a8ab32ac339e1d",
        "analysis_years":[2024,2025],
        "retrained_phit":False,
        "rerun_router":False,
        "restored_external_data":False,
        "2026_locked":True,
        "direct_combined_all":primary,
        "direct_bet_type_priority":[
            {
                "bet_type":r["bet_type"],
                "selected_races":r["selected_races"],
                "candidate_oracle_rate_pct":r["candidate_oracle_rate_pct"],
                "raw_template_rate_pct":r["raw_template_rate_pct"],
                "purchased_action_hit_rate_pct":r["purchased_action_hit_rate_pct"],
                "horse_selection_miss_races":r["horse_selection_miss_races"],
                "template_miss_races":r["template_miss_races"],
                "edge_filter_miss_races":r["edge_filter_miss_races"],
                "l2_recoverable_miss_races":r["l2_recoverable_miss_races"],
            }
            for r in priority
        ],
        "interpretation_guard":"Oracle rates are hindsight feasibility diagnostics, not deployable hit rates or ROI estimates."
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    (out/"README.md").write_text(
        "# L2 Oracle Funnel V1\n\n"
        "This is a hindsight diagnostic funnel over already-selected Router actions. "
        "It does not retrain P(hit), rerun a Router, restore Kaggle/BACKFILL data, or touch 2026.\n\n"
        "Stages are: candidate pool can form the official winning combination -> raw template "
        "contains a winning combination -> edge-filtered purchased action actually hits -> "
        "the race finishes with positive profit.\n\n"
        "Candidate/template oracle rates are feasibility ceilings for diagnosis only; they are "
        "not production predictions and must not be interpreted as achievable ROI.\n",
        encoding="utf-8"
    )

    print("L2_ORACLE_FUNNEL_V1_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)


if __name__=="__main__":
    main()
