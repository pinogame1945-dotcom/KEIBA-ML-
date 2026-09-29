#!/usr/bin/env python3
import argparse,csv,itertools,json
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import (
    horse_number_map,load_day,load_fixed_ledgers,load_router,payout_map
)

YEARS=(2022,2023,2024,2025)
EVAL_YEARS=(2023,2024,2025)
LOCKED_YEARS=(2026,)
TICKET_PRICE=100.0

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1); out[int(y)]=p
    if set(out)!=set(YEARS):
        raise SystemExit(f"router years mismatch {sorted(out)}")
    return out

def write_csv(path,rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        p.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(p,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def uniq(seq):
    seen=set(); out=[]
    for x in seq:
        t=tuple(x)
        if len(set(t))!=len(t): continue
        if t in seen: continue
        seen.add(t); out.append(t)
    return out

def formation_templates(kings,k2s):
    if len(kings)<4: return {}
    a1,a2,a3,a4=kings[:4]
    top2=[a1,a2]; top3=[a1,a2,a3]; top4=[a1,a2,a3,a4]
    out=defaultdict(list)

    for k in k2s:
        # Current baseline.
        out["BASE_A12_K2_3RD"] += [(a1,a2,k),(a2,a1,k)]

        # Same ticket count as baseline, but let A3 replace A2 in the second seat.
        out["A1_FIRST_TOP3_SECOND_K2_3RD"] += [(a1,a2,k),(a1,a3,k)]

        # A1/A2 may win; the other Top3 king may fill second.
        for first in top2:
            for second in top3:
                if second!=first:
                    out["TOP2_FIRST_TOP3_SECOND_K2_3RD"].append((first,second,k))

        # Fully open ordered king pair from Top3/Top4, K2 stays third.
        for first,second in itertools.permutations(top3,2):
            out["TOP3_KINGPAIR_K2_3RD"].append((first,second,k))
        for first,second in itertools.permutations(top4,2):
            out["TOP4_KINGPAIR_K2_3RD"].append((first,second,k))

        # Move K2 into second/first while Top3 kings occupy the other two seats.
        for first,third in itertools.permutations(top3,2):
            out["TOP3_KING_K2_2ND"].append((first,k,third))
        for second,third in itertools.permutations(top3,2):
            out["K2_1ST_TOP3_KINGPAIR"].append((k,second,third))

        # Full permutations using one K2 + any two distinct Top3 kings.
        for pair in itertools.combinations(top3,2):
            for perm in itertools.permutations((*pair,k),3):
                out["TOP3_PAIR_K2_ALL_PERMS"].append(perm)

    return {k:uniq(v) for k,v in out.items()}

def max_drawdown(rows):
    cum=peak=maxdd=0.0
    for r in sorted(rows,key=lambda x:(x["race_date"],x["race_id"])):
        cum+=r["profit_yen"]; peak=max(peak,cum); maxdd=max(maxdd,peak-cum)
    return maxdd

def summarize(rows,template,year):
    stake=sum(r["stake_yen"] for r in rows)
    ret=sum(r["return_yen"] for r in rows)
    race_returns=sorted((r["return_yen"] for r in rows),reverse=True)
    top1=race_returns[0] if race_returns else 0.0
    top5=sum(race_returns[:5])
    return {
        "template":template,"year":year,"races":len(rows),
        "hit_races":sum(r["hit"] for r in rows),
        "hit_rate_pct":100*sum(r["hit"] for r in rows)/len(rows) if rows else 0.0,
        "tickets":int(sum(r["tickets"] for r in rows)),
        "avg_tickets_per_race":sum(r["tickets"] for r in rows)/len(rows) if rows else 0.0,
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(rows),
        "top1_return_share_pct":100*top1/ret if ret else None,
        "top5_return_share_pct":100*top5/ret if ret else None,
        "roi_without_top1_pct":100*(ret-top1)/stake if stake else None,
    }

def main():
    a=parse_args(); paths=parse_paths(a.router_year)
    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={y:load_router(paths[y],y) for y in YEARS}
    for y in YEARS:
        missing=set(fixed[y])-set(routers[y])
        if missing: raise SystemExit(f"fixed alerts missing router y={y} sample={sorted(missing)[:5]}")

    date_pairs=defaultdict(list)
    for y in YEARS:
        for rid in fixed[y]:
            d=str(routers[y][rid].get("race_date") or "")[:10]
            if len(d)!=10: raise SystemExit(f"bad date race={rid}")
            date_pairs[d].append((y,rid))

    race_rows=[]
    coverage=defaultdict(int)
    root=Path(a.backfill_root)
    for di,date in enumerate(sorted(date_pairs),1):
        pairs=date_pairs[date]; wanted={rid for _,rid in pairs}
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        for year,rid in sorted(pairs):
            fr=fixed[year][rid]; pack=day.get(rid)
            if pack is None: raise SystemExit(f"missing race pack {rid}")
            k2s=[str(x) for x in fr.get("novel_horse_ids") or []]
            if not k2s:
                coverage[(year,"no_k2")]+=1
                continue
            kings=[str(x) for x in fr.get("seven_consensus_order") or []]
            if len(kings)<4: raise SystemExit(f"king order short {rid}")
            nums=horse_number_map(pack)
            for hid in set(kings[:4]+k2s):
                if hid not in nums: raise SystemExit(f"horse number missing race={rid} horse={hid}")
            payouts,_=payout_map(pack)
            templates=formation_templates(kings,k2s)
            for name,tickets in templates.items():
                ret=0.0; hit=False
                for hids in tickets:
                    comb=tuple(nums[h] for h in hids)
                    val=float(payouts.get(("TRIFECTA",comb),0.0))
                    if val>0:
                        hit=True; ret+=val
                stake=len(tickets)*TICKET_PRICE
                race_rows.append({
                    "year":year,"race_id":rid,"race_date":date,"template":name,
                    "k2_count":len(k2s),"tickets":len(tickets),
                    "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
                    "hit":int(hit),
                })
                coverage[(year,name)]+=1
        if di%25==0:
            print(f"SEAT_ARENA_PROGRESS dates={di}/{len(date_pairs)} rows={len(race_rows)}",flush=True)

    df={}
    for r in race_rows:
        df.setdefault((r["template"],r["year"]),[]).append(r)

    templates=sorted({r["template"] for r in race_rows})
    year_metrics=[]
    for t in templates:
        for y in YEARS:
            rows=df.get((t,y),[])
            if not rows: raise SystemExit(f"missing template-year t={t} y={y}")
            year_metrics.append(summarize(rows,t,y))

    stability=[]
    for t in templates:
        rs=[r for r in year_metrics if r["template"]==t and r["year"] in EVAL_YEARS]
        stake=sum(r["stake_yen"] for r in rs); ret=sum(r["return_yen"] for r in rs)
        stability.append({
            "template":t,
            "roi_2023":next(r["roi_pct"] for r in rs if r["year"]==2023),
            "roi_2024":next(r["roi_pct"] for r in rs if r["year"]==2024),
            "roi_2025":next(r["roi_pct"] for r in rs if r["year"]==2025),
            "min_year_roi_pct":min(r["roi_pct"] or 0.0 for r in rs),
            "combined_roi_pct":100*ret/stake if stake else None,
            "combined_profit_yen":ret-stake,
            "combined_hit_races":sum(r["hit_races"] for r in rs),
            "avg_tickets_per_race":sum(r["tickets"] for r in rs)/sum(r["races"] for r in rs),
            "max_year_drawdown_yen":max(r["max_drawdown_yen"] for r in rs),
            "max_top1_return_share_pct":max((r["top1_return_share_pct"] or 100.0) for r in rs),
            "min_roi_without_top1_pct":min((r["roi_without_top1_pct"] or 0.0) for r in rs),
            "all_years_roi_100plus":int(all((r["roi_pct"] or 0.0)>=100 for r in rs)),
        })
    stability.sort(key=lambda r:(
        -r["all_years_roi_100plus"],
        -(r["min_year_roi_pct"] or -1e9),
        -(r["combined_roi_pct"] or -1e9),
        r["avg_tickets_per_race"],
    ))

    # Baseline-relative hit and spend deltas.
    base={y:next(r for r in year_metrics if r["template"]=="BASE_A12_K2_3RD" and r["year"]==y) for y in EVAL_YEARS}
    deltas=[]
    for t in templates:
        if t=="BASE_A12_K2_3RD": continue
        for y in EVAL_YEARS:
            r=next(x for x in year_metrics if x["template"]==t and x["year"]==y)
            b=base[y]
            deltas.append({
                "template":t,"year":y,
                "extra_hit_races":r["hit_races"]-b["hit_races"],
                "extra_tickets":r["tickets"]-b["tickets"],
                "extra_stake_yen":r["stake_yen"]-b["stake_yen"],
                "extra_return_yen":r["return_yen"]-b["return_yen"],
                "extra_profit_yen":r["profit_yen"]-b["profit_yen"],
            })

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"year-metrics.csv",year_metrics)
    write_csv(out/"stability.csv",stability)
    write_csv(out/"baseline-deltas.csv",deltas)
    write_csv(out/"race-results.csv",race_rows)
    summary={
        "contract":"L2_DANGER_KING_SEAT_ARENA_V1",
        "analysis_years":list(YEARS),"evaluation_years":list(EVAL_YEARS),
        "locked_years":list(LOCKED_YEARS),
        "source_universe":"All L15_FIXED_V1 danger races with >=1 K2 novel.",
        "market_filtering":False,
        "ticket_type":"TRIFECTA only",
        "ticket_price_yen":TICKET_PRICE,
        "templates":templates,
        "evaluation_note":"Fixed formation arena only; no winner is promoted to production from this run.",
        "stability":stability,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Danger King Seat Arena V1\n\n"
        "Tests whether relaxing A1/A2 fixed seats improves the fixed danger/K2 trifecta structure. "
        "No odds completeness filter and no ML selector are used. K2 seat position and king-pair breadth are varied explicitly. "
        "2026 remains sealed.\n",encoding="utf-8"
    )
    print("L2_DANGER_KING_SEAT_ARENA_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
