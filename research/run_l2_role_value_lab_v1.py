#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from build_l2_bet_kings_dataset_v1 import (
    canonical_numbers,
    decode_odds,
    load_day,
    load_odds_day,
    payout_map,
)
from run_l2_bet_kings_arena_v1 import load_template
from run_l2_win_edge_audit_v1 import predict_template

YEARS=(2023,2024,2025)
TEMPLATE="WIN_ALL_CANDIDATES"


def parse_args():
    p=argparse.ArgumentParser(description="L2 role-value lab: horse x role x bet type empirical value.")
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


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


def source_role(row):
    if int(float(row.get("s1_is_novel") or 0))==1:
        return "K2_NOVEL"
    if int(float(row.get("s1_is_anchor1") or 0))==1:
        return "SEVEN_A1"
    if int(float(row.get("s1_is_anchor2") or 0))==1:
        return "SEVEN_A2"
    pos=float(row.get("s1_consensus_position") or 99)
    return "SEVEN_TOP6_OTHER" if pos<=6 else "SEVEN_OTHER"


def edge_bucket(edge):
    x=float(edge)
    if x < -0.25: return "NEG_DEEP"
    if x < 0.0: return "NEG_SHALLOW"
    if x < 0.20: return "POS_LOW"
    return "POS_HIGH"


def market_bucket(rank):
    r=int(rank)
    if r==1: return "MKT1"
    if r<=3: return "MKT2_3"
    if r<=6: return "MKT4_6"
    return "MKT7_PLUS"


def uniq_tickets(rows,bet):
    seen=set(); out=[]
    for nums in rows:
        if len(set(nums))!=len(nums):
            continue
        key=canonical_numbers(bet,nums)
        if key in seen:
            continue
        seen.add(key); out.append(key)
    return out


def role_tickets(h, candidates, a1, a2):
    others=[x for x in candidates if x!=h]
    out={}

    out["QUINELLA_HUB_ALL"]=("QUINELLA",[(h,x) for x in others])
    if h!=a1:
        out["QUINELLA_HIMO_A1"]=("QUINELLA",[(a1,h)])
    if h!=a2:
        out["QUINELLA_HIMO_A2"]=("QUINELLA",[(a2,h)])

    out["EXACTA_FIRST_ALL"]=("EXACTA",[(h,x) for x in others])
    out["EXACTA_SECOND_ALL"]=("EXACTA",[(x,h) for x in others])
    if h!=a1:
        out["EXACTA_HIMO_TO_A1_FIRST"]=("EXACTA",[(a1,h)])

    out["TRIO_HUB_ALL"]=("TRIO",[(h,x,y) for x,y in itertools.combinations(others,2)])
    if h!=a1:
        rest=[x for x in candidates if x not in {a1,h}]
        out["TRIO_HIMO_A1"]=("TRIO",[(a1,h,x) for x in rest])
    if h not in {a1,a2}:
        out["TRIO_HIMO_A12"]=("TRIO",[(a1,a2,h)])

    out["TRIFECTA_FIRST_ALL"]=(
        "TRIFECTA",[(h,x,y) for x,y in itertools.permutations(others,2)]
    )
    out["TRIFECTA_SECOND_ALL"]=(
        "TRIFECTA",[(x,h,y) for x,y in itertools.permutations(others,2)]
    )
    out["TRIFECTA_THIRD_ALL"]=(
        "TRIFECTA",[(x,y,h) for x,y in itertools.permutations(others,2)]
    )
    if h!=a1:
        rest=[x for x in candidates if x not in {a1,h}]
        out["TRIFECTA_A1_FIRST_H_SECOND"]=(
            "TRIFECTA",[(a1,h,x) for x in rest]
        )
        out["TRIFECTA_A1_FIRST_H_THIRD"]=(
            "TRIFECTA",[(a1,x,h) for x in rest]
        )
    if h not in {a1,a2}:
        out["TRIFECTA_A12_TOP2_H_THIRD"]=(
            "TRIFECTA",[(a1,a2,h),(a2,a1,h)]
        )
    return out


def segments(src,edge_b,mkt_b,rank,edge):
    out=[
        ("ALL","ALL"),
        ("SOURCE",src),
        ("EDGE",edge_b),
        ("MARKET",mkt_b),
        ("SOURCE_EDGE",f"{src}|{edge_b}"),
        ("SOURCE_MARKET",f"{src}|{mkt_b}"),
        ("EDGE_MARKET",f"{edge_b}|{mkt_b}"),
        ("FULL",f"{src}|{edge_b}|{mkt_b}"),
    ]
    if float(edge)<0 and int(rank)<=3:
        out.append(("FOCUS","POPULAR_NEG_EDGE"))
    if float(edge)<0 and int(rank)==1:
        out.append(("FOCUS","FAVORITE_NEG_EDGE"))
    if src=="SEVEN_A1" and float(edge)<0:
        out.append(("FOCUS","SEVEN_A1_NEG_EDGE"))
    if src=="SEVEN_A1" and float(edge)<0 and int(rank)<=3:
        out.append(("FOCUS","SEVEN_A1_POPULAR_NEG_EDGE"))
    if src=="K2_NOVEL":
        out.append(("FOCUS","K2_NOVEL"))
        if int(rank)>=7:
            out.append(("FOCUS","K2_NOVEL_MKT7_PLUS"))
        if float(edge)>=0.20:
            out.append(("FOCUS","K2_NOVEL_POS_HIGH"))
    return out


def empty_stat():
    return {
        "instances":0,
        "races":set(),
        "tickets":0,
        "ticket_hits":0,
        "portfolio_hits":0,
        "positive_profit_instances":0,
        "stake":0.0,
        "ret":0.0,
    }


def update_stat(stat,race_id,tickets,hit_tickets,ret):
    stake=100.0*tickets
    stat["instances"]+=1
    stat["races"].add(race_id)
    stat["tickets"]+=tickets
    stat["ticket_hits"]+=hit_tickets
    stat["portfolio_hits"]+=int(hit_tickets>0)
    stat["positive_profit_instances"]+=int(ret>stake)
    stat["stake"]+=stake
    stat["ret"]+=ret


def finalize(stats):
    rows=[]
    for (year,role,bet,dim,segment),s in sorted(stats.items()):
        stake=s["stake"]; ret=s["ret"]; inst=s["instances"]
        rows.append({
            "year":year,
            "ticket_role":role,
            "bet_type":bet,
            "segment_dimension":dim,
            "segment":segment,
            "instances":inst,
            "races":len(s["races"]),
            "tickets":s["tickets"],
            "avg_tickets_per_instance":s["tickets"]/inst if inst else None,
            "ticket_hits":s["ticket_hits"],
            "portfolio_hit_instances":s["portfolio_hits"],
            "portfolio_hit_rate_pct":100*s["portfolio_hits"]/inst if inst else None,
            "positive_profit_instances":s["positive_profit_instances"],
            "positive_profit_instance_pct":100*s["positive_profit_instances"]/inst if inst else None,
            "stake_yen":stake,
            "return_yen":ret,
            "profit_yen":ret-stake,
            "roi_pct":100*ret/stake if stake else None,
        })
    return rows


def stability(rows):
    index=defaultdict(dict)
    for r in rows:
        if r["segment_dimension"]!="FOCUS":
            continue
        index[(r["ticket_role"],r["bet_type"],r["segment"])][int(r["year"])]=r
    out=[]
    for (role,bet,seg),yr in sorted(index.items()):
        if not all(y in yr for y in YEARS):
            continue
        stake=sum(yr[y]["stake_yen"] for y in YEARS)
        ret=sum(yr[y]["return_yen"] for y in YEARS)
        out.append({
            "ticket_role":role,
            "bet_type":bet,
            "focus_segment":seg,
            "instances_2023":yr[2023]["instances"],
            "instances_2024":yr[2024]["instances"],
            "instances_2025":yr[2025]["instances"],
            "roi_2023":yr[2023]["roi_pct"],
            "roi_2024":yr[2024]["roi_pct"],
            "roi_2025":yr[2025]["roi_pct"],
            "hit_rate_2023":yr[2023]["portfolio_hit_rate_pct"],
            "hit_rate_2024":yr[2024]["portfolio_hit_rate_pct"],
            "hit_rate_2025":yr[2025]["portfolio_hit_rate_pct"],
            "combined_roi_pct":100*ret/stake if stake else None,
            "combined_profit_yen":ret-stake,
            "min_year_roi_pct":min(yr[y]["roi_pct"] for y in YEARS),
            "all_years_100plus":int(all(yr[y]["roi_pct"]>=100.0 for y in YEARS)),
            "sample_100_each_year":int(all(yr[y]["instances"]>=100 for y in YEARS)),
        })
    return out


def main():
    a=parse_args()
    manifest=json.loads((Path(a.dataset_dir)/"manifest.json").read_text(encoding="utf-8"))
    info=manifest["templates"][TEMPLATE]
    df=load_template(Path(a.dataset_dir)/info["file"])
    preds=predict_template(df,TEMPLATE)

    pred_rows=[]
    date_to_races=defaultdict(set)
    by_race={}
    for year in YEARS:
        pred=preds.get(year)
        if pred is None:
            raise SystemExit(f"missing WIN prediction year={year}")
        for rid,grp in pred.groupby(pred["race_id"].astype(str),sort=False):
            g=grp.copy()
            rows=g.to_dict("records")
            rows.sort(key=lambda r:(float(r.get("s1_candidate_position") or 99),int(float(r["selection_numbers"]))))
            date=str(rows[0]["race_date"])[:10]
            by_race[(year,rid)]=rows
            date_to_races[date].add(rid)
            pred_rows.extend(rows)

    stats=defaultdict(empty_stat)
    root=Path(a.backfill_root)
    processed=0
    for di,date in enumerate(sorted(date_to_races),1):
        wanted=date_to_races[date]
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        odds_day=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in sorted(wanted):
            year=int(str(date)[:4])
            if year not in YEARS:
                continue
            rows=by_race[(year,rid)]
            pack=day.get(rid); oddsrec=odds_day.get(rid)
            if pack is None or oddsrec is None:
                raise SystemExit(f"missing pack race={rid}")
            odds_map=decode_odds(oddsrec)
            payouts,present=payout_map(pack)

            # Full-field WIN market rank.
            win_market=[
                (int(nums[0]),float(odd))
                for (bt,nums),odd in odds_map.items()
                if bt=="WIN" and odd is not None and float(odd)>0
            ]
            win_market.sort(key=lambda x:(x[1],x[0]))
            rank_map={no:i+1 for i,(no,_) in enumerate(win_market)}

            candidate_nos=[int(float(r["selection_numbers"])) for r in rows]
            a1_rows=[r for r in rows if int(float(r.get("s1_is_anchor1") or 0))==1]
            a2_rows=[r for r in rows if int(float(r.get("s1_is_anchor2") or 0))==1]
            if len(a1_rows)!=1 or len(a2_rows)!=1:
                raise SystemExit(f"anchor cardinality race={rid} a1={len(a1_rows)} a2={len(a2_rows)}")
            a1=int(float(a1_rows[0]["selection_numbers"]))
            a2=int(float(a2_rows[0]["selection_numbers"]))

            for r in rows:
                h=int(float(r["selection_numbers"]))
                rank=rank_map.get(h)
                if rank is None:
                    continue
                edge=float(r["edge"])
                src=source_role(r)
                eb=edge_bucket(edge)
                mb=market_bucket(rank)

                for role,(bet,ticket_rows) in role_tickets(h,candidate_nos,a1,a2).items():
                    if bet not in present:
                        continue
                    tickets=uniq_tickets(ticket_rows,bet)
                    valid=[]
                    for nums in tickets:
                        key=(bet,nums)
                        if key in odds_map:
                            valid.append(nums)
                    if not valid:
                        continue
                    ret=sum(float(payouts.get((bet,nums),0.0)) for nums in valid)
                    hit_tickets=sum(float(payouts.get((bet,nums),0.0))>0 for nums in valid)
                    for dim,seg in segments(src,eb,mb,rank,edge):
                        update_stat(stats[(year,role,bet,dim,seg)],rid,len(valid),hit_tickets,ret)
            processed+=1
        if di%50==0:
            print(f"ROLE_VALUE_PROGRESS dates={di}/{len(date_to_races)} races={processed}",flush=True)

    rows=finalize(stats)
    stable=stability(rows)

    # Compact focus ranking for research navigation only.
    focus_rank=[
        r for r in stable
        if r["sample_100_each_year"]==1
    ]
    focus_rank.sort(key=lambda r:(-r["min_year_roi_pct"],-r["combined_roi_pct"],r["ticket_role"],r["focus_segment"]))

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"role-value-summary.csv",rows)
    write_csv(out/"focus-stability.csv",stable)
    write_csv(out/"focus-priority.csv",focus_rank)

    summary={
        "contract":"L2_ROLE_VALUE_LAB_RESULT_V1",
        "source_win_calibration_run":36520915593,
        "analysis_years":[2023,2024,2025],
        "races_processed":processed,
        "horse_candidate_rows":len(pred_rows),
        "ticket_roles":[
            "QUINELLA_HUB_ALL","QUINELLA_HIMO_A1","QUINELLA_HIMO_A2",
            "EXACTA_FIRST_ALL","EXACTA_SECOND_ALL","EXACTA_HIMO_TO_A1_FIRST",
            "TRIO_HUB_ALL","TRIO_HIMO_A1","TRIO_HIMO_A12",
            "TRIFECTA_FIRST_ALL","TRIFECTA_SECOND_ALL","TRIFECTA_THIRD_ALL",
            "TRIFECTA_A1_FIRST_H_SECOND","TRIFECTA_A1_FIRST_H_THIRD",
            "TRIFECTA_A12_TOP2_H_THIRD",
        ],
        "focus_segments":[
            "POPULAR_NEG_EDGE","FAVORITE_NEG_EDGE","SEVEN_A1_NEG_EDGE",
            "SEVEN_A1_POPULAR_NEG_EDGE","K2_NOVEL","K2_NOVEL_MKT7_PLUS","K2_NOVEL_POS_HIGH"
        ],
        "win_phit_market_features":False,
        "2025_is_holdout":False,
        "2026_locked":True,
        "production_promotion":False,
        "interpretation_guard":"Role portfolios are deterministic hindsight evaluation of pre-race-defined horse segments. Overlapping role portfolios double-count tickets across diagnostic views; results are research evidence, not a deployable strategy."
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Role Value Lab V1\n\n"
        "Empirical horse x role x bet-type audit over all 2023-2025 races. "
        "Horse segments use only pre-race information: Seven/K2 role, odds-free WIN P(hit) edge, "
        "and final WIN market rank. Multi-bet role portfolios are evaluated with final odds/payouts. "
        "2025 is development evidence and 2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_ROLE_VALUE_LAB_V1_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)


if __name__=="__main__":
    main()
