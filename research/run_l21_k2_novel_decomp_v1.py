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
EXPECTED_ALERTS=1384
EXPECTED_NOVEL=5577


def parse_args():
    p=argparse.ArgumentParser(description="Decompose K2 novel flat-WIN performance and jackpot concentration.")
    p.add_argument("--contract",required=True)
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--candidate-root",required=True)
    p.add_argument("--race-summary",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def finite(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None


def pipe_list(v):
    return [x for x in str(v or "").split("|") if x]


def read_csv(path):
    with open(path,newline="",encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_fixed(root):
    root=Path(root)
    out={}
    total=0
    novel=0
    for year in YEARS:
        rows={}
        path=root/f"y{year}.jsonl"
        with open(path,encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row=json.loads(line)
                if row.get("contract")!="L15_FIXED_OUTPUT_V1":
                    raise ValueError(f"unexpected L1.5 contract {path}")
                rid=str(row.get("race_id") or "")
                if not rid or rid in rows:
                    raise ValueError(f"duplicate/missing race_id y{year}: {rid}")
                if row.get("gate_alert") is not True or row.get("gate_action")!="INTERVENE_FULL_K2":
                    raise ValueError(f"non-K2 fixed row {rid}")
                if len(row.get("selected_outsiders") or [])!=2:
                    raise ValueError(f"K2 selector drift {rid}")
                rows[rid]=row
                total+=1
                novel+=len(row.get("novel_horse_ids") or [])
        out[year]=rows
    if total!=EXPECTED_ALERTS:
        raise SystemExit(f"alert count regression {total} != {EXPECTED_ALERTS}")
    if novel!=EXPECTED_NOVEL:
        raise SystemExit(f"novel count regression {novel} != {EXPECTED_NOVEL}")
    return out


def load_outsider_top6(root):
    root=Path(root)
    out={}
    for year in YEARS:
        path=root/f"y{year}"/"outsider-top6.csv"
        rows=read_csv(path)
        by_race=defaultdict(dict)
        for row in rows:
            rid=str(row.get("race_id") or "")
            label=str(row.get("label_ja") or "")
            ids=pipe_list(row.get("top6_horse_ids"))
            probs=pipe_list(row.get("top6_probabilities"))
            if not rid or not label:
                continue
            items={}
            for i,hid in enumerate(ids):
                p=finite(probs[i]) if i<len(probs) else None
                items[hid]={"rank":i+1,"prob":p}
            by_race[rid][label]=items
        out[year]=by_race
    return out


def load_race_dates(path):
    out={}
    for row in read_csv(path):
        rid=str(row.get("race_id") or "")
        date=str(row.get("race_date") or "")
        if rid and date:
            out[rid]=date
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


def horse_number_map(pack):
    out={}
    for e in pack.get("entries") or []:
        hid=str(e.get("horse_id") or "")
        try:
            no=int(e.get("horse_number"))
        except (TypeError,ValueError):
            no=0
        if hid and no>0:
            out[hid]=no
    return out


def result_map(pack):
    out={}
    for r in pack.get("results") or []:
        hid=str(r.get("horse_id") or "")
        pos=finite(r.get("official_finish_position"))
        if hid:
            out[hid]=int(pos) if pos is not None else None
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
        tup=final_odds_tuple(raw)
        if not tup:
            continue
        odds=finite(tup[0])
        if odds is not None and odds>0:
            out[int(key)]=odds
    return out


def win_payout_by_number(pack):
    out={}
    for r in pack.get("payouts") or []:
        if str(r.get("bet_type") or "")!="WIN":
            continue
        nums=[int(x) for x in re.findall(r"\d+",str(r.get("combination") or ""))]
        if len(nums)!=1:
            continue
        payout=finite(r.get("payout_yen"))
        if payout is not None and payout>0:
            out[nums[0]]=payout
    return out


def market_rank_bucket(rank):
    if rank==1: return "1"
    if rank<=3: return "2-3"
    if rank<=6: return "4-6"
    return "7+"


def odds_bucket(odds):
    if odds<10: return "LT10"
    if odds<20: return "10-20"
    if odds<50: return "20-50"
    if odds<100: return "50-100"
    if odds<200: return "100-200"
    return "200+"


def gate_bucket(score):
    if score<0.60: return "LT0.60"
    if score<0.65: return "0.60-0.65"
    if score<0.70: return "0.65-0.70"
    return "0.70+"


def novel_pos_bucket(pos):
    if pos==1: return "1"
    if pos==2: return "2"
    if pos==3: return "3"
    return "4+"


def period_labels(year):
    out=[str(year),"ALL_2022_2025"]
    if year in DEV_YEARS:
        out.append("DEV_2022_2024")
    if year==HOLDOUT_YEAR:
        out.append("HOLDOUT_2025")
    return out


def summarize(rows):
    bets=len(rows)
    wins=sum(int(r["win"]) for r in rows)
    top3=sum(int(r["top3"]) for r in rows)
    stake=100.0*bets
    ret=sum(float(r["return_yen"]) for r in rows)
    market_prob=sum(float(r["market_prob"]) for r in rows)/bets if bets else None
    win_rate=wins/bets if bets else None
    return {
        "bets":bets,
        "wins":wins,
        "top3":top3,
        "win_rate":win_rate,
        "top3_rate":top3/bets if bets else None,
        "avg_final_odds":sum(float(r["final_odds"]) for r in rows)/bets if bets else None,
        "avg_market_rank":sum(float(r["market_rank"]) for r in rows)/bets if bets else None,
        "avg_normalized_market_probability":market_prob,
        "win_rate_minus_market_probability_pp":(
            100*(win_rate-market_prob) if bets else None
        ),
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
    }


def add_group_summary(out,rows,period,dimension,bucket):
    s=summarize(rows)
    out.append({"period":period,"dimension":dimension,"bucket":bucket,**s})


def main():
    a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L21_K2_NOVEL_DECOMP_V1":
        raise SystemExit("wrong contract")
    if contract["scope"]["locked_years"]!=[2026]:
        raise SystemExit("2026 lock drift")
    if contract["claim_discipline"]["tuning"] is not False:
        raise SystemExit("tuning must be disabled")

    fixed=load_fixed(a.fixed_ledger_dir)
    outsider=load_outsider_top6(a.candidate_root)
    race_dates=load_race_dates(a.race_summary)

    date_to_races=defaultdict(list)
    owner={}
    for year in YEARS:
        for rid in fixed[year]:
            date=race_dates.get(rid)
            if not date:
                raise ValueError(f"missing Market Gap race date for fixed race={rid}")
            owner[rid]=year
            date_to_races[date].append(rid)

    ledger=[]
    backfill=Path(a.backfill_root)
    for date in sorted(date_to_races):
        wanted=set(date_to_races[date])
        day_path=backfill/"data"/"daily"/f"{date}.jsonl.gz"
        odds_path=backfill/"data"/"odds"/"daily"/f"{date}.jsonl.gz"
        if not day_path.exists() or not odds_path.exists():
            raise SystemExit(f"missing BACKFILL market file date={date}")
        packs=load_day(day_path,wanted)
        odds_rows=load_odds_day(odds_path,wanted)
        for rid in sorted(wanted):
            year=owner[rid]
            fr=fixed[year][rid]
            pack=packs.get(rid)
            odds_rec=odds_rows.get(rid)
            if pack is None or odds_rec is None:
                raise ValueError(f"missing race/odds row race={rid}")

            horse_no=horse_number_map(pack)
            results=result_map(pack)
            win_odds=win_odds_by_number(odds_rec)
            payouts=win_payout_by_number(pack)
            no_horse={no:hid for hid,no in horse_no.items()}

            market=[]
            for no,odds in win_odds.items():
                hid=no_horse.get(no)
                if hid:
                    market.append((odds,no,hid))
            market.sort(key=lambda x:(x[0],x[1]))
            market_rank={hid:i+1 for i,(_,_,hid) in enumerate(market)}
            market_odds={hid:o for o,_,hid in market}
            overround=sum(1.0/o for o,_,_ in market)
            if overround<=0:
                raise ValueError(f"invalid market overround race={rid}")
            market_prob={hid:(1.0/o)/overround for o,_,hid in market}

            selected=[str(x) for x in fr.get("selected_outsiders") or []]
            maps=outsider[year].get(rid) or {}
            novel=[str(x) for x in fr.get("novel_horse_ids") or []]
            for pos,hid in enumerate(novel,1):
                if hid not in horse_no:
                    raise ValueError(f"novel horse number missing race={rid} horse={hid}")
                if hid not in market_rank:
                    raise ValueError(f"novel horse final win odds missing race={rid} horse={hid}")
                sources=[]
                source_ranks={}
                source_probs={}
                for label in selected:
                    item=(maps.get(label) or {}).get(hid)
                    if item:
                        sources.append(label)
                        source_ranks[label]=item.get("rank")
                        source_probs[label]=item.get("prob")
                if not sources:
                    raise ValueError(f"novel horse has no selected K2 source race={rid} horse={hid}")
                no=horse_no[hid]
                finish=results.get(hid)
                ret=float(payouts.get(no,0.0))
                odds=market_odds[hid]
                ledger.append({
                    "year":year,
                    "race_id":rid,
                    "race_date":date,
                    "horse_id":hid,
                    "horse_number":no,
                    "novel_position":pos,
                    "novel_position_bucket":novel_pos_bucket(pos),
                    "gate_score":float(fr.get("gate_score") or 0.0),
                    "gate_score_bucket":gate_bucket(float(fr.get("gate_score") or 0.0)),
                    "selected_outsiders":"|".join(selected),
                    "source_labels":"|".join(sources),
                    "source_count":len(sources),
                    "source_rank_min":min(int(source_ranks[x]) for x in sources),
                    "source_prob_max":max(
                        [float(source_probs[x]) for x in sources if source_probs[x] is not None] or [0.0]
                    ),
                    "market_rank":market_rank[hid],
                    "market_rank_bucket":market_rank_bucket(market_rank[hid]),
                    "final_odds":odds,
                    "odds_bucket":odds_bucket(odds),
                    "market_prob":market_prob[hid],
                    "finish":finish,
                    "win":finish==1,
                    "top3":finish is not None and finish<=3,
                    "return_yen":ret,
                    "profit_yen":ret-100.0,
                })

    if len(ledger)!=EXPECTED_NOVEL:
        raise SystemExit(f"ledger count regression {len(ledger)} != {EXPECTED_NOVEL}")

    bucket_rows=[]
    periods=sorted(set(p for r in ledger for p in period_labels(int(r["year"]))))
    dimensions=(
        ("market_rank_bucket","MARKET_RANK"),
        ("odds_bucket","FINAL_ODDS"),
        ("gate_score_bucket","GATE_SCORE"),
        ("novel_position_bucket","NOVEL_POSITION"),
        ("source_count","SOURCE_COUNT"),
    )
    for period in periods:
        prows=[r for r in ledger if period in period_labels(int(r["year"]))]
        add_group_summary(bucket_rows,prows,period,"ALL","ALL")
        for field,name in dimensions:
            vals=sorted(set(str(r[field]) for r in prows))
            for v in vals:
                sub=[r for r in prows if str(r[field])==v]
                add_group_summary(bucket_rows,sub,period,name,v)

    source_set_rows=[]
    for period in periods:
        prows=[r for r in ledger if period in period_labels(int(r["year"]))]
        for source_set in sorted(set(r["source_labels"] for r in prows)):
            sub=[r for r in prows if r["source_labels"]==source_set]
            add_group_summary(source_set_rows,sub,period,"SOURCE_SET",source_set)

    fractional=defaultdict(lambda:defaultdict(float))
    for r in ledger:
        labels=pipe_list(r["source_labels"])
        n=len(labels)
        for period in period_labels(int(r["year"])):
            for label in labels:
                s=fractional[(period,label)]
                w=1.0/n
                s["bets"]+=w
                s["wins"]+=w*int(r["win"])
                s["top3"]+=w*int(r["top3"])
                s["stake"]+=100.0*w
                s["return"]+=float(r["return_yen"])*w
    fractional_rows=[]
    for (period,label),s in sorted(fractional.items()):
        fractional_rows.append({
            "period":period,
            "label":label,
            "fractional_bets":s["bets"],
            "fractional_wins":s["wins"],
            "fractional_top3":s["top3"],
            "fractional_win_rate":s["wins"]/s["bets"] if s["bets"] else None,
            "fractional_top3_rate":s["top3"]/s["bets"] if s["bets"] else None,
            "fractional_stake_yen":s["stake"],
            "fractional_return_yen":s["return"],
            "fractional_profit_yen":s["return"]-s["stake"],
            "fractional_roi_pct":100*s["return"]/s["stake"] if s["stake"] else None,
        })

    concentration=[]
    removal_counts=contract["concentration_tests"]["remove_highest_return_wins"]
    for period in periods:
        base=[r for r in ledger if period in period_labels(int(r["year"]))]
        for scope in ("ALL_NOVEL","MARKET_RANK_7PLUS"):
            rows=base if scope=="ALL_NOVEL" else [r for r in base if int(r["market_rank"])>=7]
            wins=sorted(
                [r for r in rows if r["win"]],
                key=lambda r:(-float(r["return_yen"]),r["race_id"],r["horse_id"]),
            )
            total_return=sum(float(r["return_yen"]) for r in rows)
            top_shares={}
            for k in (1,3,5,10):
                top=sum(float(r["return_yen"]) for r in wins[:k])
                top_shares[k]=100*top/total_return if total_return>0 else None
            for k in removal_counts:
                removed=wins[:int(k)]
                removed_keys={(r["race_id"],r["horse_id"]) for r in removed}
                kept=[r for r in rows if (r["race_id"],r["horse_id"]) not in removed_keys]
                ks=summarize(kept)
                zero_return=sum(float(r["return_yen"]) for r in rows)-sum(float(r["return_yen"]) for r in removed)
                full_stake=100.0*len(rows)
                concentration.append({
                    "period":period,
                    "scope":scope,
                    "removed_highest_return_wins":k,
                    "original_bets":len(rows),
                    "original_wins":len(wins),
                    "removed_return_yen":sum(float(r["return_yen"]) for r in removed),
                    "drop_ticket_bets":ks["bets"],
                    "drop_ticket_profit_yen":ks["profit_yen"],
                    "drop_ticket_roi_pct":ks["roi_pct"],
                    "zero_return_profit_yen":zero_return-full_stake,
                    "zero_return_roi_pct":100*zero_return/full_stake if full_stake else None,
                    "top1_return_share_pct":top_shares[1],
                    "top3_return_share_pct":top_shares[3],
                    "top5_return_share_pct":top_shares[5],
                    "top10_return_share_pct":top_shares[10],
                })

    winners=sorted(
        [r for r in ledger if r["win"]],
        key=lambda r:(-float(r["return_yen"]),r["race_id"],r["horse_id"]),
    )
    jackpot=[]
    for idx,r in enumerate(winners[:50],1):
        jackpot.append({
            "rank_by_return":idx,
            "year":r["year"],
            "race_id":r["race_id"],
            "race_date":r["race_date"],
            "horse_id":r["horse_id"],
            "horse_number":r["horse_number"],
            "market_rank":r["market_rank"],
            "final_odds":r["final_odds"],
            "return_yen":r["return_yen"],
            "source_labels":r["source_labels"],
            "source_count":r["source_count"],
            "gate_score":r["gate_score"],
            "novel_position":r["novel_position"],
        })

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"novel-ledger.csv",ledger)
    write_csv(out/"bucket-summary.csv",bucket_rows)
    write_csv(out/"outsider-source-set.csv",source_set_rows)
    write_csv(out/"outsider-label-fractional.csv",fractional_rows)
    write_csv(out/"concentration.csv",concentration)
    write_csv(out/"jackpot-winners.csv",jackpot)

    all_dev=[r for r in ledger if int(r["year"]) in DEV_YEARS]
    all_hold=[r for r in ledger if int(r["year"])==HOLDOUT_YEAR]
    seven_dev=[r for r in all_dev if int(r["market_rank"])>=7]
    seven_hold=[r for r in all_hold if int(r["market_rank"])>=7]
    summary={
        "contract":"L21_K2_NOVEL_DECOMP_RESULT_V1",
        "upstream":"L15_FIXED_V1",
        "novel_horses":len(ledger),
        "development_years":sorted(DEV_YEARS),
        "holdout_year":HOLDOUT_YEAR,
        "holdout_used_for_tuning":False,
        "locked_years":[2026],
        "all_novel_dev":summarize(all_dev),
        "all_novel_holdout_2025":summarize(all_hold),
        "market_rank_7plus_dev":summarize(seven_dev),
        "market_rank_7plus_holdout_2025":summarize(seven_hold),
        "concentration_removal_semantics":{
            "drop_ticket":"Remove the highest-return winning ticket(s) entirely from both stake and return.",
            "zero_return":"Keep all stake but set the selected highest-return winning ticket(s) return to zero."
        },
        "outputs":[
            "novel-ledger.csv","bucket-summary.csv","outsider-source-set.csv",
            "outsider-label-fractional.csv","concentration.csv","jackpot-winners.csv"
        ]
    }
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    (out/"README.md").write_text(
        "# L2.1 K2 Novel Decomposition V1\n\n"
        "This run decomposes the 5,577 frozen L15_FIXED_V1 K2 novel horses.\n\n"
        "It does not select or tune a betting rule. Buckets were preregistered before this run. "
        "2022-2024 remain development diagnostics and 2025 remains holdout confirmation.\n\n"
        "Concentration tests report both dropping the largest winning ticket(s) entirely and the "
        "stricter counterfactual of keeping stake while zeroing those returns.\n\n"
        "Outsider attribution is shown as exact source sets and fractional per-label attribution "
        "when both selected outsiders nominated the same novel horse.\n",
        encoding="utf-8",
    )
    print("L21_K2_NOVEL_DECOMP_V1_READY")
    print(json.dumps({
        "novel_horses":len(ledger),
        "wins":sum(int(r["win"]) for r in ledger),
        "dev_roi":summary["all_novel_dev"]["roi_pct"],
        "holdout_roi":summary["all_novel_holdout_2025"]["roi_pct"],
        "dev_7plus_roi":summary["market_rank_7plus_dev"]["roi_pct"],
        "holdout_7plus_roi":summary["market_rank_7plus_holdout_2025"]["roi_pct"],
        "out_dir":str(out),
    },ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
