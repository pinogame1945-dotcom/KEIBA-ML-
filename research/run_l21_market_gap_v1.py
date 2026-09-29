#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

YEARS=(2022,2023,2024,2025)
DEV_YEARS={2022,2023,2024}
HOLDOUT_YEAR=2025
EXPECTED_PER_YEAR=3456
EXPECTED_TOTAL=13824
EXPECTED_ALERTS=1384


def parse_args():
    p=argparse.ArgumentParser(description="L2.1 Market Gap V1: Seven-King vs closing win market.")
    p.add_argument("--contract",required=True)
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")


def parse_year_paths(items):
    out={}
    for spec in items:
        year,path=spec.split(":",1)
        out[int(year)]=path
    return out


def finite(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None


def load_fixed_ledgers(root):
    root=Path(root)
    out={}
    total=0
    for year in YEARS:
        path=root/f"y{year}.jsonl"
        rows={}
        with open(path,encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row=json.loads(line)
                if row.get("contract")!="L15_FIXED_OUTPUT_V1":
                    raise ValueError(f"unexpected L1.5 contract in {path}")
                rid=str(row.get("race_id") or "")
                if not rid or rid in rows:
                    raise ValueError(f"bad fixed race_id y{year}: {rid}")
                if row.get("gate_alert") is not True:
                    raise ValueError(f"fixed ledger contains non-alert: {rid}")
                rows[rid]=row
                total+=1
        out[year]=rows
    if total!=EXPECTED_ALERTS:
        raise SystemExit(f"fixed Gate count regression: {total} != {EXPECTED_ALERTS}")
    return out


def load_router(path,year):
    rows={}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            if not rid or rid in rows:
                raise ValueError(f"router duplicate/missing race_id y{year}: {rid}")
            rows[rid]=row
    if len(rows)!=EXPECTED_PER_YEAR:
        raise SystemExit(f"router count regression y{year}: {len(rows)} != {EXPECTED_PER_YEAR}")
    return rows


def seven_consensus(router_row):
    experts=router_row.get("experts") or {}
    if len(experts)!=7:
        raise ValueError(f"expected 7 experts race={router_row.get('race_id')} got={len(experts)}")
    stat=defaultdict(lambda:{
        "support":0,"borda":0.0,"top1_votes":0,"best_rank":99,"rank_sum":0.0,
    })
    for expert in experts.values():
        ids=[str(x) for x in (expert.get("top6_horse_ids") or []) if str(x)]
        if not ids:
            raise ValueError(f"empty expert Top6 race={router_row.get('race_id')}")
        for rank,hid in enumerate(ids,1):
            s=stat[hid]
            s["support"]+=1
            s["borda"]+=float(7-rank)
            s["top1_votes"]+=int(rank==1)
            s["best_rank"]=min(s["best_rank"],rank)
            s["rank_sum"]+=rank
    for s in stat.values():
        s["mean_rank"]=s["rank_sum"]/s["support"]
    ordered=sorted(
        stat,
        key=lambda hid:(
            -stat[hid]["borda"],
            -stat[hid]["support"],
            -stat[hid]["top1_votes"],
            stat[hid]["best_rank"],
            stat[hid]["mean_rank"],
            hid,
        ),
    )
    return ordered,dict(stat)


def horse_number_map(race_pack):
    out={}
    for row in race_pack.get("entries") or []:
        hid=str(row.get("horse_id") or "")
        try:
            no=int(row.get("horse_number"))
        except (TypeError,ValueError):
            no=0
        if hid and no>0:
            out[hid]=no
    return out


def result_map(race_pack):
    out={}
    for row in race_pack.get("results") or []:
        hid=str(row.get("horse_id") or "")
        if not hid:
            continue
        finish=finite(row.get("official_finish_position"))
        status=str(row.get("result_status") or "").upper()
        out[hid]={
            "finish":int(finish) if finish is not None else None,
            "finished":status=="FINISHED",
        }
    return out


def final_odds_tuple(raw):
    if not isinstance(raw,list) or len(raw)<3:
        return None
    return raw[3:6] if len(raw)>=6 else raw[:3]


def win_odds_by_number(record):
    out={}
    data=(record.get("odds") or {}).get("1")
    if not isinstance(data,dict):
        return out
    for key,raw in data.items():
        key=str(key)
        if not re.fullmatch(r"\d{1,2}",key):
            continue
        no=int(key)
        tup=final_odds_tuple(raw)
        if not tup:
            continue
        odds=finite(tup[0])
        if odds is not None and odds>0:
            out[no]=odds
    return out


def win_payout_by_number(race_pack):
    out={}
    for row in race_pack.get("payouts") or []:
        if str(row.get("bet_type") or "")!="WIN":
            continue
        nums=[int(x) for x in re.findall(r"\d+",str(row.get("combination") or ""))]
        if len(nums)!=1:
            continue
        payout=finite(row.get("payout_yen"))
        if payout is not None and payout>0:
            out[nums[0]]=payout
    return out


def load_day(path,wanted):
    rows={}
    with gzip.open(path,"rt",encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row=json.loads(line)
            rid=str((row.get("race") or {}).get("race_id") or "")
            if rid in wanted:
                rows[rid]=row
    return rows


def load_odds_day(path,wanted):
    rows={}
    with gzip.open(path,"rt",encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            if rid in wanted:
                rows[rid]=row
    return rows


def pearson(xs,ys):
    if len(xs)<2 or len(xs)!=len(ys):
        return None
    mx=sum(xs)/len(xs)
    my=sum(ys)/len(ys)
    num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx=sum((x-mx)**2 for x in xs)
    dy=sum((y-my)**2 for y in ys)
    if dx<=0 or dy<=0:
        return None
    return num/math.sqrt(dx*dy)


def market_bucket(rank):
    if rank==1:
        return "1"
    if 2<=rank<=3:
        return "2-3"
    if 4<=rank<=6:
        return "4-6"
    return "7+"


def ai_bucket(rank,in_union=True):
    if not in_union:
        return "OUTSIDE_UNION"
    if rank==1:
        return "1"
    if 2<=rank<=3:
        return "2-3"
    if 4<=rank<=6:
        return "4-6"
    return "7+"


def rank_gap_bucket(gap):
    if gap>=4:
        return "AI_BULLISH_4PLUS"
    if gap>=2:
        return "AI_BULLISH_2_3"
    if gap>=-1:
        return "NEAR_1"
    if gap>=-3:
        return "MARKET_BULLISH_2_3"
    return "MARKET_BULLISH_4PLUS"


def labels_for_year(year):
    labels=[str(year),"ALL_2022_2025"]
    if year in DEV_YEARS:
        labels.append("DEV_2022_2024")
    if year==HOLDOUT_YEAR:
        labels.append("HOLDOUT_2025")
    return labels


def segment_labels(gate_alert):
    return ["ALL","GATE_ALERT" if gate_alert else "NON_GATE"]


def update_horse_agg(store,key,row):
    s=store[key]
    s["bets"]+=1
    s["wins"]+=int(row["win"])
    s["top3"]+=int(row["top3"])
    s["stake"]+=100.0
    s["return"]+=float(row["win_return"])
    s["odds_sum"]+=float(row["odds"])
    s["market_prob_sum"]+=float(row["market_prob"])
    s["market_rank_sum"]+=float(row["market_rank"])


def finalize_horse_agg(store,kind):
    rows=[]
    for key,s in sorted(store.items(),key=lambda kv: tuple(map(str,kv[0]))):
        year_label,segment,bucket=key
        bets=s["bets"]
        stake=s["stake"]
        ret=s["return"]
        win_rate=s["wins"]/bets if bets else 0.0
        avg_prob=s["market_prob_sum"]/bets if bets else 0.0
        rows.append({
            "kind":kind,
            "period":year_label,
            "segment":segment,
            "bucket":bucket,
            "bets":bets,
            "wins":s["wins"],
            "top3":s["top3"],
            "win_rate":win_rate,
            "top3_rate":s["top3"]/bets if bets else 0.0,
            "avg_final_odds":s["odds_sum"]/bets if bets else None,
            "avg_normalized_market_probability":avg_prob,
            "win_rate_minus_market_probability_pp":100*(win_rate-avg_prob),
            "stake_yen":stake,
            "return_yen":ret,
            "profit_yen":ret-stake,
            "flat_win_roi_pct":100*ret/stake if stake else None,
            "avg_market_rank":s["market_rank_sum"]/bets if bets else None,
        })
    return rows


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L21_MARKET_GAP_V1":
        raise SystemExit("wrong Market Gap contract")
    if contract["ai"]["odds_used"] is not False:
        raise SystemExit("AI market-independence guard broken")
    if contract["scope"]["locked_years"]!=[2026]:
        raise SystemExit("2026 lock guard broken")

    paths=parse_year_paths(a.router_year)
    if set(paths)!=set(YEARS):
        raise SystemExit(f"router year mismatch: {sorted(paths)}")
    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={y:load_router(paths[y],y) for y in YEARS}

    date_to_races=defaultdict(list)
    owner={}
    for year in YEARS:
        for rid,row in routers[year].items():
            if rid in owner:
                raise ValueError(f"duplicate race_id across years: {rid}")
            date=str(row.get("race_date") or "")[:10]
            if len(date)!=10:
                raise ValueError(f"bad race_date race={rid}")
            owner[rid]=year
            date_to_races[date].append(rid)
    if len(owner)!=EXPECTED_TOTAL:
        raise SystemExit(f"total race regression: {len(owner)} != {EXPECTED_TOTAL}")

    ai_top1_agg=defaultdict(lambda:defaultdict(float))
    market1_agg=defaultdict(lambda:defaultdict(float))
    rank_gap_agg=defaultdict(lambda:defaultdict(float))
    novel_agg=defaultdict(lambda:defaultdict(float))
    signal_agg=defaultdict(lambda:defaultdict(float))
    overview=defaultdict(lambda:{
        "races":0,"top1_agree":0,"top3_overlap_sum":0.0,"top3_jaccard_sum":0.0,
        "spearman_sum":0.0,"spearman_n":0,"ai_top1_market_rank_sum":0.0,
    })
    race_rows=[]
    coverage=defaultdict(int)
    backfill=Path(a.backfill_root)

    for date in sorted(date_to_races):
        wanted=set(date_to_races[date])
        day_path=backfill/"data"/"daily"/f"{date}.jsonl.gz"
        odds_path=backfill/"data"/"odds"/"daily"/f"{date}.jsonl.gz"
        if not day_path.exists() or not odds_path.exists():
            raise SystemExit(f"missing BACKFILL day/odds file: {date}")
        day=load_day(day_path,wanted)
        odds_day=load_odds_day(odds_path,wanted)
        for rid in sorted(wanted):
            year=owner[rid]
            router=routers[year][rid]
            race_pack=day.get(rid)
            odds_record=odds_day.get(rid)
            if race_pack is None or odds_record is None:
                raise SystemExit(f"missing race market row race={rid}")

            seven_order,seven_stats=seven_consensus(router)
            if len(seven_order)<3:
                coverage["seven_lt3"]+=1
                continue
            fixed_row=fixed[year].get(rid)
            gate_alert=fixed_row is not None
            if fixed_row and fixed_row.get("seven_consensus_order")!=seven_order:
                raise ValueError(f"frozen seven consensus drift race={rid}")

            horse_no=horse_number_map(race_pack)
            results=result_map(race_pack)
            win_odds=win_odds_by_number(odds_record)
            win_payout=win_payout_by_number(race_pack)
            number_horse={no:hid for hid,no in horse_no.items()}

            market=[]
            for no,odds in win_odds.items():
                hid=number_horse.get(no)
                if not hid:
                    continue
                market.append((odds,no,hid))
            market.sort(key=lambda x:(x[0],x[1]))
            if len(market)<3:
                coverage["market_lt3"]+=1
                continue
            market_rank={hid:i+1 for i,(_,_,hid) in enumerate(market)}
            market_odds={hid:odds for odds,_,hid in market}
            market_no={hid:no for _,no,hid in market}
            overround=sum(1.0/odds for odds,_,_ in market)
            market_prob={
                hid:(1.0/odds)/overround
                for odds,_,hid in market
            }
            market_order=[hid for _,_,hid in market]
            market_fav=market_order[0]

            seven_pos={hid:i+1 for i,hid in enumerate(seven_order)}
            common=[hid for hid in seven_order if hid in market_rank]
            if len(common)<2:
                coverage["seven_market_common_lt2"]+=1
                continue
            ai_top1=seven_order[0]
            if ai_top1 not in market_rank:
                coverage["ai_top1_missing_market"]+=1
                continue

            xs=[seven_pos[hid] for hid in common]
            ys=[market_rank[hid] for hid in common]
            spearman=pearson(xs,ys)
            ai_top3=set(seven_order[:3])
            market_top3=set(market_order[:3])
            overlap=len(ai_top3&market_top3)
            union=len(ai_top3|market_top3)
            jaccard=overlap/union if union else 0.0

            def horse_row(hid):
                res=results.get(hid) or {}
                finish=res.get("finish")
                no=market_no.get(hid)
                return {
                    "horse_id":hid,
                    "horse_number":no,
                    "odds":market_odds[hid],
                    "market_prob":market_prob[hid],
                    "market_rank":market_rank[hid],
                    "finish":finish,
                    "win":finish==1,
                    "top3":finish is not None and finish<=3,
                    "win_return":float(win_payout.get(no,0.0)),
                }

            ai1=horse_row(ai_top1)
            fav=horse_row(market_fav)
            fav_in_union=market_fav in seven_pos
            fav_ai_rank=seven_pos.get(market_fav,len(market)+1)
            gate_score=finite((fixed_row or {}).get("gate_score")) or 0.0

            for period in labels_for_year(year):
                for segment in segment_labels(gate_alert):
                    ov=overview[(period,segment)]
                    ov["races"]+=1
                    ov["top1_agree"]+=int(ai_top1==market_fav)
                    ov["top3_overlap_sum"]+=overlap
                    ov["top3_jaccard_sum"]+=jaccard
                    ov["ai_top1_market_rank_sum"]+=ai1["market_rank"]
                    if spearman is not None:
                        ov["spearman_sum"]+=spearman
                        ov["spearman_n"]+=1

                    update_horse_agg(
                        ai_top1_agg,
                        (period,segment,market_bucket(ai1["market_rank"])),
                        ai1,
                    )
                    update_horse_agg(
                        market1_agg,
                        (period,segment,ai_bucket(fav_ai_rank,fav_in_union)),
                        fav,
                    )

                    for hid in common:
                        hr=horse_row(hid)
                        gap=market_rank[hid]-seven_pos[hid]
                        update_horse_agg(
                            rank_gap_agg,
                            (period,segment,rank_gap_bucket(gap)),
                            hr,
                        )

                    signal_rows=[
                        ("AI_TOP1_MARKET_4PLUS",ai1,ai1["market_rank"]>=4),
                        ("AGREEMENT_TOP1",ai1,ai1["market_rank"]==1),
                        (
                            "MARKET1_AI_4PLUS_OR_OUTSIDE",
                            fav,
                            (not fav_in_union) or fav_ai_rank>=4,
                        ),
                    ]
                    for hid in seven_order[:3]:
                        if hid in market_rank:
                            hr=horse_row(hid)
                            signal_rows.append((
                                "AI_TOP3_MARKET_4PLUS",
                                hr,
                                hr["market_rank"]>=4,
                            ))
                    for signal,hr,enabled in signal_rows:
                        if enabled:
                            update_horse_agg(signal_agg,(period,segment,signal),hr)

                    if fixed_row:
                        for hid in fixed_row.get("novel_horse_ids") or []:
                            hid=str(hid)
                            if hid not in market_rank:
                                coverage["novel_missing_market"]+=1
                                continue
                            hr=horse_row(hid)
                            update_horse_agg(
                                novel_agg,
                                (period,segment,market_bucket(hr["market_rank"])),
                                hr,
                            )

            winner_ids=[
                hid for hid,res in results.items()
                if res.get("finish")==1
            ]
            race_rows.append({
                "year":year,
                "race_id":rid,
                "race_date":date,
                "gate_alert":gate_alert,
                "gate_score":gate_score,
                "field_market_count":len(market),
                "seven_union_count":len(seven_order),
                "seven_top1_horse_id":ai_top1,
                "seven_top1_horse_number":ai1["horse_number"],
                "seven_top1_market_rank":ai1["market_rank"],
                "seven_top1_final_odds":ai1["odds"],
                "seven_top1_normalized_market_probability":ai1["market_prob"],
                "seven_top1_finish":ai1["finish"],
                "seven_top1_win_return_yen":ai1["win_return"],
                "market_favorite_horse_id":market_fav,
                "market_favorite_horse_number":fav["horse_number"],
                "market_favorite_final_odds":fav["odds"],
                "market_favorite_seven_rank":fav_ai_rank if fav_in_union else None,
                "market_favorite_in_seven_union":fav_in_union,
                "market_favorite_finish":fav["finish"],
                "top1_agreement":ai_top1==market_fav,
                "top3_overlap_count":overlap,
                "top3_jaccard":jaccard,
                "seven_market_spearman":spearman,
                "winner_horse_ids":"|".join(sorted(winner_ids)),
                "k2_novel_count":len((fixed_row or {}).get("novel_horse_ids") or []),
            })
            coverage["races_processed"]+=1

    if coverage["races_processed"]!=EXPECTED_TOTAL:
        raise SystemExit(
            f"race coverage regression: {coverage['races_processed']} != {EXPECTED_TOTAL}; "
            f"coverage={dict(coverage)}"
        )

    overview_rows=[]
    for (period,segment),s in sorted(overview.items()):
        n=s["races"]
        overview_rows.append({
            "period":period,
            "segment":segment,
            "races":n,
            "top1_agreement_rate":s["top1_agree"]/n if n else None,
            "mean_top3_overlap_count":s["top3_overlap_sum"]/n if n else None,
            "mean_top3_jaccard":s["top3_jaccard_sum"]/n if n else None,
            "mean_seven_market_spearman":s["spearman_sum"]/s["spearman_n"] if s["spearman_n"] else None,
            "mean_ai_top1_market_rank":s["ai_top1_market_rank_sum"]/n if n else None,
        })

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"overview.csv",overview_rows)
    write_csv(out/"ai-top1-market-rank.csv",finalize_horse_agg(ai_top1_agg,"AI_TOP1_BY_MARKET_RANK"))
    write_csv(out/"market-favorite-ai-rank.csv",finalize_horse_agg(market1_agg,"MARKET_FAVORITE_BY_AI_RANK"))
    write_csv(out/"rank-gap.csv",finalize_horse_agg(rank_gap_agg,"SEVEN_UNION_RANK_GAP"))
    write_csv(out/"signals.csv",finalize_horse_agg(signal_agg,"PREREGISTERED_SIGNAL"))
    write_csv(out/"k2-novel-market-rank.csv",finalize_horse_agg(novel_agg,"K2_NOVEL_BY_MARKET_RANK"))
    write_csv(out/"race-summary.csv",race_rows)

    summary={
        "contract":"L21_MARKET_GAP_RESULT_V1",
        "upstream":"L15_FIXED_V1",
        "races":coverage["races_processed"],
        "gate_alerts":EXPECTED_ALERTS,
        "development_years":sorted(DEV_YEARS),
        "holdout_year":HOLDOUT_YEAR,
        "holdout_used_for_tuning":False,
        "ai_uses_odds":False,
        "market_price":"historical final WIN odds",
        "market_rank":"ascending final win odds; ties by horse_number",
        "normalized_market_probability":"(1/odds) / sum(1/odds) within race",
        "locked_years":[2026],
        "coverage":dict(coverage),
        "outputs":[
            "overview.csv","ai-top1-market-rank.csv","market-favorite-ai-rank.csv",
            "rank-gap.csv","signals.csv","k2-novel-market-rank.csv","race-summary.csv"
        ],
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    (out/"README.md").write_text(
        "# L2.1 Market Gap V1\n\n"
        "Seven-King ranking is computed without odds/popularity/payout. It is compared only afterward "
        "with historical final WIN odds.\n\n"
        "2022-2024 are the development diagnostic period. 2025 is holdout confirmation and is not "
        "used to tune bucket boundaries. 2026 remains sealed.\n\n"
        "Primary outputs:\n"
        "- overview.csv: agreement/correlation by period and Gate segment\n"
        "- ai-top1-market-rank.csv: where Seven-King Top1 sits in the market and its flat WIN ROI\n"
        "- market-favorite-ai-rank.csv: how Seven-King treats the market favorite\n"
        "- rank-gap.csv: all Seven-union horses grouped by AI-vs-market rank gap\n"
        "- signals.csv: preregistered treasure/danger signal summaries\n"
        "- k2-novel-market-rank.csv: K2 novel horses versus market rank\n"
        "- race-summary.csv: compact per-race audit ledger\n\n"
        "No Market Gap feature is promoted to Bet Router by this run.\n",
        encoding="utf-8",
    )
    print("L21_MARKET_GAP_V1_READY")
    print(json.dumps({
        "races":coverage["races_processed"],
        "gate_alerts":EXPECTED_ALERTS,
        "race_rows":len(race_rows),
        "out_dir":str(out),
    },ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
