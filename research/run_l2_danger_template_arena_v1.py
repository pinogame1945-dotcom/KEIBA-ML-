#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import (
    canonical_numbers,
    decode_odds,
    horse_number_map,
    load_day,
    load_fixed_ledgers,
    load_odds_day,
    load_router,
    payout_map,
)

YEARS=(2023,2024,2025)
LOCKED_YEARS=(2026,)

TEMPLATES={
    "QUINELLA_KING_K2":"QUINELLA",
    "EXACTA_KING_TO_K2":"EXACTA",
    "EXACTA_K2_TO_KING":"EXACTA",
    "TRIO_KING_KING_K2":"TRIO",
    "TRIFECTA_KING_KING_K2":"TRIFECTA",
    "TRIFECTA_KING_K2_KING":"TRIFECTA",
    "TRIFECTA_K2_KING_KING":"TRIFECTA",
}

def parse_args():
    p=argparse.ArgumentParser(description="Danger-race coarse ticket-template tournament over frozen L1.5 anchors and all K2 novel horses.")
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_paths(items):
    out={}
    for spec in items:
        year,path=spec.split(":",1)
        out[int(year)]=path
    if set(out)!=set(YEARS):
        raise SystemExit(f"router years mismatch {sorted(out)} expected={list(YEARS)}")
    return out

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def uniq_tickets(bet,rows):
    out=[]; seen=set()
    for nums in rows:
        nums=tuple(int(x) for x in nums)
        if len(set(nums))!=len(nums): continue
        key=canonical_numbers(bet,nums)
        if key in seen: continue
        seen.add(key); out.append(key)
    return out

def make_templates(a1,a2,k2s):
    rows={k:[] for k in TEMPLATES}
    for k in k2s:
        rows["QUINELLA_KING_K2"].extend(((a1,k),(a2,k)))
        rows["EXACTA_KING_TO_K2"].extend(((a1,k),(a2,k)))
        rows["EXACTA_K2_TO_KING"].extend(((k,a1),(k,a2)))
        rows["TRIO_KING_KING_K2"].append((a1,a2,k))
        rows["TRIFECTA_KING_KING_K2"].extend(((a1,a2,k),(a2,a1,k)))
        rows["TRIFECTA_KING_K2_KING"].extend(((a1,k,a2),(a2,k,a1)))
        rows["TRIFECTA_K2_KING_KING"].extend(((k,a1,a2),(k,a2,a1)))
    return {name:uniq_tickets(TEMPLATES[name],vals) for name,vals in rows.items()}

def max_drawdown(race_rows):
    cumulative=0.0; peak=0.0; max_dd=0.0
    for row in sorted(race_rows,key=lambda r:(r["race_date"],r["race_id"])):
        cumulative += row["profit_yen"]
        peak=max(peak,cumulative)
        max_dd=max(max_dd,peak-cumulative)
    return max_dd

def summarize_year(year,template,source_races,race_rows):
    evaluated=len(race_rows)
    stake=sum(r["stake_yen"] for r in race_rows)
    ret=sum(r["return_yen"] for r in race_rows)
    tickets=sum(r["tickets"] for r in race_rows)
    hits=sum(r["return_yen"]>0 for r in race_rows)
    profitable=sum(r["profit_yen"]>0 for r in race_rows)
    returns_desc=sorted(((r["return_yen"],r["stake_yen"]) for r in race_rows),reverse=True)
    top1_ret=returns_desc[0][0] if returns_desc else 0.0
    top5_ret=sum(x[0] for x in returns_desc[:5])
    top1_stake=returns_desc[0][1] if returns_desc else 0.0
    denom_wo=stake-top1_stake
    return {
        "year":year,"template":template,"bet_type":TEMPLATES[template],
        "source_danger_races":source_races,"evaluated_races":evaluated,
        "race_coverage_pct":100*evaluated/source_races if source_races else 0.0,
        "tickets":tickets,"avg_tickets_per_race":tickets/evaluated if evaluated else None,
        "hit_races":hits,"hit_rate_pct":100*hits/evaluated if evaluated else 0.0,
        "profitable_races":profitable,"profitable_race_pct":100*profitable/evaluated if evaluated else 0.0,
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(race_rows),
        "largest_race_return_yen":top1_ret,
        "top1_return_share_pct":100*top1_ret/ret if ret>0 else None,
        "top5_return_share_pct":100*top5_ret/ret if ret>0 else None,
        "roi_without_top1_race_pct":100*(ret-top1_ret)/denom_wo if denom_wo>0 else None,
    }

def combine(rows):
    idx=defaultdict(dict)
    for r in rows: idx[r["template"]][int(r["year"])]=r
    out=[]
    for template,yr in sorted(idx.items()):
        if not all(y in yr for y in YEARS): continue
        stake=sum(yr[y]["stake_yen"] for y in YEARS)
        ret=sum(yr[y]["return_yen"] for y in YEARS)
        races=sum(yr[y]["evaluated_races"] for y in YEARS)
        tickets=sum(yr[y]["tickets"] for y in YEARS)
        out.append({
            "template":template,"bet_type":TEMPLATES[template],
            "roi_2023":yr[2023]["roi_pct"],"roi_2024":yr[2024]["roi_pct"],"roi_2025":yr[2025]["roi_pct"],
            "min_year_roi_pct":min(yr[y]["roi_pct"] for y in YEARS),
            "combined_roi_pct":100*ret/stake if stake else None,
            "combined_profit_yen":ret-stake,"combined_races":races,"combined_tickets":tickets,
            "avg_tickets_per_race":tickets/races if races else None,
            "min_year_coverage_pct":min(yr[y]["race_coverage_pct"] for y in YEARS),
            "hit_rate_2023":yr[2023]["hit_rate_pct"],"hit_rate_2024":yr[2024]["hit_rate_pct"],"hit_rate_2025":yr[2025]["hit_rate_pct"],
            "max_drawdown_2023":yr[2023]["max_drawdown_yen"],"max_drawdown_2024":yr[2024]["max_drawdown_yen"],"max_drawdown_2025":yr[2025]["max_drawdown_yen"],
            "max_top1_return_share_pct":max(yr[y]["top1_return_share_pct"] or 0.0 for y in YEARS),
            "min_roi_without_top1_race_pct":min(yr[y]["roi_without_top1_race_pct"] for y in YEARS),
            "all_years_roi_100plus":int(all(yr[y]["roi_pct"]>=100.0 for y in YEARS)),
            "all_years_coverage_99plus":int(all(yr[y]["race_coverage_pct"]>=99.0 for y in YEARS)),
        })
    out.sort(key=lambda r:(-r["all_years_coverage_99plus"],-r["min_year_roi_pct"],-r["combined_roi_pct"],r["template"]))
    return out

def main():
    a=parse_args(); paths=parse_paths(a.router_year)
    if 2026 not in LOCKED_YEARS or any(y>=2026 for y in YEARS):
        raise SystemExit("2026 lock violated")
    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={y:load_router(paths[y],y) for y in YEARS}
    danger_counts={y:len(fixed[y]) for y in YEARS}
    if not all(danger_counts[y]>0 for y in YEARS):
        raise SystemExit(f"missing fixed alert rows {danger_counts}")

    date_races=defaultdict(list)
    for y in YEARS:
        missing=set(fixed[y])-set(routers[y])
        if missing: raise SystemExit(f"fixed alerts missing router year={y} sample={sorted(missing)[:5]}")
        for rid in fixed[y]:
            date=str(routers[y][rid].get("race_date") or "")[:10]
            if len(date)!=10: raise SystemExit(f"bad race date {rid}")
            date_races[date].append((y,rid))

    root=Path(a.backfill_root)
    race_results=defaultdict(list); incomplete=defaultdict(int); k2_counts=defaultdict(list)
    eligible_counts=defaultdict(int); no_k2_counts=defaultdict(int)
    processed=0
    for di,date in enumerate(sorted(date_races),1):
        pairs=date_races[date]; wanted={rid for _,rid in pairs}
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        odds_day=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for year,rid in sorted(pairs):
            fr=fixed[year][rid]; pack=day.get(rid); oddsrec=odds_day.get(rid)
            if pack is None or oddsrec is None: raise SystemExit(f"missing market pack race={rid}")
            anchors=[str(x) for x in (fr.get("seven_anchor_horse_ids") or [])]
            novel=[str(x) for x in (fr.get("novel_horse_ids") or [])]
            if len(anchors)!=2: raise SystemExit(f"anchor cardinality race={rid} anchors={anchors}")
            if not novel:
                no_k2_counts[year]+=1
                processed+=1
                continue
            eligible_counts[year]+=1
            horse_no=horse_number_map(pack); needed=set(anchors+novel); miss=needed-set(horse_no)
            if miss: raise SystemExit(f"horse number missing race={rid} sample={sorted(miss)[:5]}")
            a1,a2=(horse_no[anchors[0]],horse_no[anchors[1]])
            k2s=[horse_no[h] for h in novel]; k2_counts[year].append(len(k2s))
            odds_map=decode_odds(oddsrec); payouts,present=payout_map(pack)
            generated=make_templates(a1,a2,k2s)
            for template,tickets in generated.items():
                bet=TEMPLATES[template]
                if bet not in present or not tickets:
                    incomplete[(year,template)]+=1; continue
                if any((bet,nums) not in odds_map for nums in tickets):
                    incomplete[(year,template)]+=1; continue
                ret=sum(float(payouts.get((bet,nums),0.0)) for nums in tickets)
                stake=100.0*len(tickets)
                race_results[(year,template)].append({
                    "race_date":date,"race_id":rid,"tickets":len(tickets),
                    "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
                })
            processed+=1
        if di%25==0:
            print(f"DANGER_TEMPLATE_PROGRESS dates={di}/{len(date_races)} races={processed}",flush=True)

    year_rows=[]
    for year in YEARS:
        for template in TEMPLATES:
            year_rows.append(summarize_year(year,template,eligible_counts[year],race_results[(year,template)]))
    stable=combine(year_rows)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"template-year.csv",year_rows)
    write_csv(out/"template-stability.csv",stable)
    write_csv(out/"coverage-audit.csv",[
        {"year":y,"template":t,"all_danger_races":danger_counts[y],
         "k2_eligible_danger_races":eligible_counts[y],
         "danger_races_without_k2_novel":no_k2_counts[y],
         "incomplete_or_unpriced_races":incomplete[(y,t)],
         "evaluated_races":len(race_results[(y,t)])}
        for y in YEARS for t in TEMPLATES
    ])
    summary={
        "contract":"L2_DANGER_TEMPLATE_ARENA_V1",
        "source_l15":"L15_FIXED_V1",
        "analysis_years":list(YEARS),
        "locked_years":list(LOCKED_YEARS),
        "scope":"CONSENSUS_WORLD_GATE alert races with at least one K2 novel horse; use every K2 novel horse; no odds/popularity filter and no ML selector",
        "templates":{
            "QUINELLA_KING_K2":"A1-K2 and A2-K2",
            "EXACTA_KING_TO_K2":"A1/A2 -> K2",
            "EXACTA_K2_TO_KING":"K2 -> A1/A2",
            "TRIO_KING_KING_K2":"A1 + A2 + K2",
            "TRIFECTA_KING_KING_K2":"A1/A2 occupy 1st-2nd, K2 3rd",
            "TRIFECTA_KING_K2_KING":"A1/A2 1st, K2 2nd, other anchor 3rd",
            "TRIFECTA_K2_KING_KING":"K2 1st, A1/A2 occupy 2nd-3rd",
        },
        "all_danger_races":danger_counts,
        "k2_eligible_danger_races":dict(eligible_counts),
        "danger_races_without_k2_novel":dict(no_k2_counts),
        "avg_k2_novel_per_race":{str(y):sum(k2_counts[y])/len(k2_counts[y]) for y in YEARS},
        "ticket_price_yen":100,
        "market_usage":"final odds are used only to verify ticket availability; realized payout is used for deterministic ROI evaluation",
        "production_promotion":False,
        "current_provisional_benchmark":"TRIFECTA_KING_KING_K2 equals DANGER_A12_TOP2_K2_THIRD_KARI_FIX_V1",
        "next_rule":"Only after this coarse structure tournament should formation width (Top2/Top3/Top4) be expanded for the surviving structures.",
        "stability":stable,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Danger Template Arena V1\n\n"
        "Coarse structural tournament for CONSENSUS_WORLD_GATE alert races that contain at least one K2 novel horse. "
        "Danger races with zero K2 novel horses are audited separately and are not silently counted as failed K2 coverage. "
        "Every frozen K2 novel horse is used; there is no odds filter, popularity filter, or ML selection. "
        "The seven ticket structures isolate where K2 sits relative to the frozen A1/A2 anchors across quinella, exacta, trio and trifecta. "
        "Metrics include ROI, profit, hit rate, race coverage, tickets per race, max drawdown and return concentration. "
        "2026 remains sealed and no production rule is promoted by this run.\n",
        encoding="utf-8",
    )
    print("L2_DANGER_TEMPLATE_ARENA_V1_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
