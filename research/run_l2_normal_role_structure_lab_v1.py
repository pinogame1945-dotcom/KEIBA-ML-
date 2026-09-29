#!/usr/bin/env python3
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path

YEARS=(2023,2024,2025)

TEMPLATES={
    "QUINELLA_CORE12":{
        "bet_type":"QUINELLA",
        "role":"A1-A2",
        "meaning":"A1 and A2 only",
    },
    "QUINELLA_AXIS1_KING":{
        "bet_type":"QUINELLA",
        "role":"A1-KING_ALL",
        "meaning":"A1 with every other Seven-King consensus horse",
    },
    "QUINELLA_KING_TOP4_BOX":{
        "bet_type":"QUINELLA",
        "role":"KING_TOP4_BOX",
        "meaning":"Seven-King consensus Top4 box",
    },
    "EXACTA_CORE12_MULTI":{
        "bet_type":"EXACTA",
        "role":"A1_A2_BOTH_ORDERS",
        "meaning":"A1-A2 in both exacta orders",
    },
    "EXACTA_AXIS1_FORWARD":{
        "bet_type":"EXACTA",
        "role":"A1_FIRST_KING_ALL",
        "meaning":"A1 fixed first, every other Seven-King consensus horse second",
    },
    "EXACTA_AXIS1_REVERSE":{
        "bet_type":"EXACTA",
        "role":"KING_ALL_FIRST_A1_SECOND",
        "meaning":"Every other Seven-King consensus horse first, A1 fixed second",
    },
    "EXACTA_AXIS1_MULTI":{
        "bet_type":"EXACTA",
        "role":"A1_KING_ALL_BOTH_DIRECTIONS",
        "meaning":"A1 versus every other Seven-King consensus horse in both orders",
    },
    "TRIO_AXIS12_ALL":{
        "bet_type":"TRIO",
        "role":"A1_A2_PLUS_KING",
        "meaning":"A1 and A2 fixed with every other Seven-King consensus horse",
    },
    "TRIO_AXIS1_ALL":{
        "bet_type":"TRIO",
        "role":"A1_PLUS_TWO_KINGS",
        "meaning":"A1 fixed with every pair of other Seven-King consensus horses",
    },
    "TRIO_KING_TOP6_BOX":{
        "bet_type":"TRIO",
        "role":"KING_TOP6_BOX",
        "meaning":"Seven-King consensus Top6 trio box",
    },
    "TRIFECTA_ANCHOR12_MULTI":{
        "bet_type":"TRIFECTA",
        "role":"A1_A2_MATE_ALL_ORDERS",
        "meaning":"A1, A2 and one other King in all six orders",
    },
    "TRIFECTA_A1_FIRST_ANCHOR2_MATE":{
        "bet_type":"TRIFECTA",
        "role":"A1_FIRST_A2_MATE",
        "meaning":"A1 fixed first; A2 and one other King occupy second/third in both orders",
    },
    "TRIFECTA_ANCHORS_TOP2_MATE":{
        "bet_type":"TRIFECTA",
        "role":"A1_A2_TOP2_MATE_THIRD",
        "meaning":"A1/A2 occupy top two in either order; one other King fixed third",
    },
    "TRIFECTA_KING_TOP4_BOX":{
        "bet_type":"TRIFECTA",
        "role":"KING_TOP4_BOX",
        "meaning":"Seven-King consensus Top4 trifecta box",
    },
}


def parse_args():
    p=argparse.ArgumentParser(description="PASS_SEVEN_ONLY broad role structure audit.")
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def max_drawdown(races):
    rows=sorted(races,key=lambda r:(r["race_date"],r["race_id"]))
    cumulative=0.0; peak=0.0; dd=0.0
    for r in rows:
        cumulative+=r["return_yen"]-r["stake_yen"]
        peak=max(peak,cumulative)
        dd=max(dd,peak-cumulative)
    return dd


def concentration(races):
    if not races:
        return {"top1_return_share_pct":None,"top5_return_share_pct":None,"roi_without_top1_return_pct":None}
    total_ret=sum(r["return_yen"] for r in races)
    total_stake=sum(r["stake_yen"] for r in races)
    vals=sorted([r["return_yen"] for r in races],reverse=True)
    top1=vals[0] if vals else 0.0
    top5=sum(vals[:5])
    return {
        "top1_return_share_pct":100*top1/total_ret if total_ret>0 else None,
        "top5_return_share_pct":100*top5/total_ret if total_ret>0 else None,
        "roi_without_top1_return_pct":100*max(0.0,total_ret-top1)/total_stake if total_stake>0 else None,
    }


def finalize(year,template,races):
    meta=TEMPLATES[template]
    stake=sum(r["stake_yen"] for r in races)
    ret=sum(r["return_yen"] for r in races)
    tickets=sum(r["tickets"] for r in races)
    hits=sum(r["hit"] for r in races)
    positive=sum((r["return_yen"]-r["stake_yen"])>0 for r in races)
    return {
        "year":year,
        "bet_type":meta["bet_type"],
        "template":template,
        "role":meta["role"],
        "meaning":meta["meaning"],
        "normal_races":len(races),
        "tickets":tickets,
        "avg_tickets_per_race":tickets/len(races) if races else None,
        "hit_races":hits,
        "race_hit_rate_pct":100*hits/len(races) if races else None,
        "positive_races":positive,
        "positive_race_rate_pct":100*positive/len(races) if races else None,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "max_drawdown_yen":max_drawdown(races),
        **concentration(races),
    }


def main():
    a=parse_args()
    root=Path(a.dataset_dir)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract")!="L2_BET_KINGS_DATASET_V1":
        raise SystemExit("wrong dataset contract")
    if manifest.get("source_l15")!="L15_FIXED_V1":
        raise SystemExit("wrong L1.5 source")
    if manifest.get("locked_years")!=[2026]:
        raise SystemExit("2026 lock drift")

    metric_rows=[]
    universe=defaultdict(set)

    for template,meta in TEMPLATES.items():
        info=manifest["templates"].get(template)
        if not info:
            raise SystemExit(f"template missing from manifest: {template}")
        if info.get("bet_type")!=meta["bet_type"]:
            raise SystemExit(f"bet type drift: {template}")

        per_race={}
        path=root/info["file"]
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                r=json.loads(line)
                year=int(r["year"])
                if year not in YEARS: continue
                if int(r.get("gate_alert") or 0)!=0: continue
                if int(r.get("ticket_novel_count") or 0)!=0:
                    raise SystemExit(f"novel leakage into PASS_SEVEN_ONLY template={template} race={r['race_id']}")
                rid=str(r["race_id"])
                key=(year,rid)
                universe[year].add(rid)
                z=per_race.setdefault(key,{
                    "year":year,"race_id":rid,"race_date":str(r["race_date"])[:10],
                    "tickets":0,"stake_yen":0.0,"return_yen":0.0,"hit":0,
                })
                z["tickets"]+=1
                z["stake_yen"]+=100.0
                z["return_yen"]+=float(r.get("return_yen_per100") or 0.0)
                z["hit"]=int(z["hit"] or bool(r.get("hit")))

        for year in YEARS:
            races=[v for (y,_),v in per_race.items() if y==year]
            metric_rows.append(finalize(year,template,races))

    # Common normal-race universe should be fully covered by every deterministic template.
    universe_counts={str(y):len(universe[y]) for y in YEARS}
    coverage_rows=[]
    for row in metric_rows:
        denom=universe_counts[str(row["year"])]
        row["normal_universe_races"]=denom
        row["priced_race_coverage_pct"]=100*row["normal_races"]/denom if denom else None
        coverage_rows.append(row)

    # Cross-year structure stability.
    idx=defaultdict(dict)
    for r in coverage_rows:
        idx[r["template"]][int(r["year"])]=r
    stable=[]
    for template,yr in idx.items():
        if not all(y in yr for y in YEARS): continue
        stake=sum(yr[y]["stake_yen"] for y in YEARS)
        ret=sum(yr[y]["return_yen"] for y in YEARS)
        tickets=sum(yr[y]["tickets"] for y in YEARS)
        races=sum(yr[y]["normal_races"] for y in YEARS)
        meta=TEMPLATES[template]
        stable.append({
            "bet_type":meta["bet_type"],
            "template":template,
            "role":meta["role"],
            "meaning":meta["meaning"],
            "roi_2023":yr[2023]["roi_pct"],
            "roi_2024":yr[2024]["roi_pct"],
            "roi_2025":yr[2025]["roi_pct"],
            "profit_2023":yr[2023]["profit_yen"],
            "profit_2024":yr[2024]["profit_yen"],
            "profit_2025":yr[2025]["profit_yen"],
            "hit_rate_2023":yr[2023]["race_hit_rate_pct"],
            "hit_rate_2024":yr[2024]["race_hit_rate_pct"],
            "hit_rate_2025":yr[2025]["race_hit_rate_pct"],
            "avg_tickets_per_race_2023":yr[2023]["avg_tickets_per_race"],
            "avg_tickets_per_race_2024":yr[2024]["avg_tickets_per_race"],
            "avg_tickets_per_race_2025":yr[2025]["avg_tickets_per_race"],
            "combined_races":races,
            "combined_tickets":tickets,
            "combined_stake_yen":stake,
            "combined_return_yen":ret,
            "combined_profit_yen":ret-stake,
            "combined_roi_pct":100*ret/stake if stake else None,
            "min_year_roi_pct":min(yr[y]["roi_pct"] for y in YEARS),
            "all_years_100plus":int(all(yr[y]["roi_pct"]>=100.0 for y in YEARS)),
            "max_top1_return_share_pct":max(yr[y]["top1_return_share_pct"] or 0.0 for y in YEARS),
            "min_roi_without_top1_pct":min(yr[y]["roi_without_top1_return_pct"] or 0.0 for y in YEARS),
        })

    stable.sort(key=lambda r:(
        -r["all_years_100plus"],
        -r["min_year_roi_pct"],
        -r["combined_profit_yen"],
        r["combined_tickets"],
        r["template"],
    ))

    # Navigation only: strongest broad structure inside each bet type, no production promotion.
    leaders=[]
    for bet in ("QUINELLA","EXACTA","TRIO","TRIFECTA"):
        pool=[r for r in stable if r["bet_type"]==bet]
        if not pool: continue
        leaders.append(dict(pool[0]))

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"yearly-metrics.csv",coverage_rows)
    write_csv(out/"structure-stability.csv",stable)
    write_csv(out/"bet-type-leaders.csv",leaders)

    summary={
        "contract":"L2_NORMAL_ROLE_STRUCTURE_LAB_V1",
        "scope":"PASS_SEVEN_ONLY only; no K2 novel",
        "analysis_years":list(YEARS),
        "normal_universe_races":universe_counts,
        "templates":list(TEMPLATES),
        "bet_types":["QUINELLA","EXACTA","TRIO","TRIFECTA"],
        "race_filtering":False,
        "all_normal_races_used":True,
        "ticket_price_yen":100,
        "purpose":"Coarse role/template tournament before formation-width tuning.",
        "2026_locked":True,
        "production_promotion":False,
        "interpretation_guard":"Deterministic structural baselines only. Do not tune race filters or infer a production strategy from this run alone.",
        "all_years_profitable_structures":[r for r in stable if r["all_years_100plus"]==1],
        "bet_type_leaders":leaders,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Normal Role Structure Lab V1\n\n"
        "PASS_SEVEN_ONLY races only. No race filtering and no K2 novel. "
        "This is the first coarse tournament for the roughly 90% normal battlefield: "
        "QUINELLA, EXACTA, TRIO and TRIFECTA using fixed Seven-King role templates. "
        "Every ticket is evaluated at 100 yen. The run compares ROI, total profit, race hit rate, "
        "average tickets per race, drawdown and jackpot concentration. "
        "No formation-width optimization is performed here. 2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_NORMAL_ROLE_STRUCTURE_LAB_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)


if __name__=="__main__":
    main()
