#!/usr/bin/env python3
import argparse
import csv
import gzip
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

YEARS=(2022,2023,2024,2025)
POLICIES=("BASE","FULL_K2","FULL_K3")
BET_GROUP={
    "WIN":"1",
    "QUINELLA":"4",
    "EXACTA":"6",
    "TRIO":"7",
    "TRIFECTA":"8",
}
ARITY={"WIN":1,"QUINELLA":2,"EXACTA":2,"TRIO":3,"TRIFECTA":3}
UNORDERED={"QUINELLA","TRIO"}
STRATEGY_ORDER=(
    "WIN_CORE2",
    "QUINELLA_AXIS1",
    "EXACTA_MULTI1",
    "TRIO_AXIS12",
    "TRIO_AXIS1",
    "TRIFECTA_MULTI12",
)


def parse_args():
    p=argparse.ArgumentParser(description="Build ticket-level L2 profit-test rows for Seven-King BASE vs Outsider K2/K3.")
    p.add_argument("--candidate-root",required=True)
    p.add_argument("--router-year",action="append",required=True,help="YEAR:PATH")
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--coverage-out",required=True)
    return p.parse_args()


def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")


def pipe_list(value):
    return [x for x in str(value or "").split("|") if x]


def parse_year_paths(items):
    out={}
    for spec in items:
        y,path=spec.split(":",1)
        out[int(y)]=path
    return out


def finite(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None


def mean(values):
    z=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(z)/len(z) if z else 0.0


def truthy(value):
    return str(value or "").strip().lower() in {"1","true","yes"}


def read_csv(path):
    with open(path,newline="",encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_router(path,wanted):
    out={}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            if rid in wanted:
                out[rid]=row
    missing=sorted(wanted-set(out))
    if missing:
        raise ValueError(f"router coverage missing count={len(missing)} sample={missing[:10]}")
    return out


def seven_stats(router_row):
    experts=router_row.get("experts") or {}
    if len(experts)!=7:
        raise ValueError(f"expected 7 experts race={router_row.get('race_id')} got={len(experts)}")
    stat=defaultdict(lambda:{
        "support":0,"borda":0.0,"best_rank":99,"rank_sum":0.0,"top1_votes":0,
    })
    union=set()
    for expert in experts.values():
        ids=[str(x) for x in (expert.get("top6_horse_ids") or []) if str(x)]
        for rank,hid in enumerate(ids,1):
            union.add(hid)
            s=stat[hid]
            s["support"]+=1
            s["borda"]+=float(7-rank)
            s["best_rank"]=min(s["best_rank"],rank)
            s["rank_sum"]+=rank
            if rank==1:
                s["top1_votes"]+=1
    if not union:
        raise ValueError(f"empty seven union race={router_row.get('race_id')}")
    for hid,s in stat.items():
        s["mean_rank"]=s["rank_sum"]/s["support"] if s["support"] else 99.0
    ordered=sorted(
        union,
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


def outsider_maps(rows):
    by_race=defaultdict(dict)
    for row in rows:
        rid=str(row["race_id"])
        label=str(row["label_ja"])
        ids=pipe_list(row.get("top6_horse_ids"))
        nums=pipe_list(row.get("top6_horse_numbers"))
        probs=pipe_list(row.get("top6_probabilities"))
        items={}
        for i,hid in enumerate(ids):
            no=int(nums[i]) if i<len(nums) and nums[i] else None
            prob=finite(probs[i]) if i<len(probs) else None
            items[hid]={"rank":i+1,"horse_no":no,"prob":prob}
        by_race[rid][label]=items
    return by_race


def selected_labels(decision,policy):
    if policy=="BASE":
        return []
    key="FULL__COMBO_K2" if policy=="FULL_K2" else "FULL__COMBO_K3"
    return pipe_list(decision.get(key))


def selected_outsider_stats(label_maps,labels):
    stat=defaultdict(lambda:{
        "support":0,"borda":0.0,"best_rank":99,"prob_sum":0.0,"prob_n":0,"prob_max":0.0,
    })
    for label in labels:
        rows=label_maps.get(label)
        if rows is None:
            raise ValueError(f"selected outsider label missing: {label}")
        for hid,item in rows.items():
            s=stat[hid]
            rank=int(item["rank"])
            s["support"]+=1
            s["borda"]+=float(7-rank)
            s["best_rank"]=min(s["best_rank"],rank)
            prob=finite(item.get("prob"))
            if prob is not None:
                s["prob_sum"]+=prob
                s["prob_n"]+=1
                s["prob_max"]=max(s["prob_max"],prob)
    for s in stat.values():
        s["prob_mean"]=s["prob_sum"]/s["prob_n"] if s["prob_n"] else 0.0
    return dict(stat)


def canonical_numbers(bet_type,values):
    vals=tuple(int(x) for x in values)
    return tuple(sorted(vals)) if bet_type in UNORDERED else vals


def final_odds_tuple(raw):
    if not isinstance(raw,list) or len(raw)<3:
        return None
    return raw[3:6] if len(raw)>=6 else raw[:3]


def decode_odds(record):
    out={}
    root=record.get("odds") or {}
    for bet_type,group in BET_GROUP.items():
        data=root.get(group)
        if not isinstance(data,dict):
            continue
        n=ARITY[bet_type]
        for key,raw in data.items():
            key=str(key)
            nums=[]
            if n==1:
                if not re.fullmatch(r"\d{1,2}",key):
                    continue
                nums=[int(key)]
            else:
                if not re.fullmatch(r"\d{%d}"%(2*n),key):
                    continue
                nums=[int(key[i*2:i*2+2]) for i in range(n)]
            if any(x<=0 for x in nums) or (n>1 and len(set(nums))!=n):
                continue
            tup=final_odds_tuple(raw)
            if not tup:
                continue
            odds=finite(tup[0])
            if odds is None or odds<=0:
                continue
            out[(bet_type,canonical_numbers(bet_type,nums))]=odds
    return out


def payout_map(race_pack):
    out={}
    type_present=set()
    for row in race_pack.get("payouts") or []:
        bet_type=str(row.get("bet_type") or "")
        if bet_type not in BET_GROUP:
            continue
        type_present.add(bet_type)
        nums=[int(x) for x in re.findall(r"\d+",str(row.get("combination") or ""))]
        if len(nums)!=ARITY[bet_type] or any(x<=0 for x in nums):
            continue
        payout=finite(row.get("payout_yen"))
        if payout is None or payout<=0:
            continue
        out[(bet_type,canonical_numbers(bet_type,nums))]=payout
    return out,type_present


def load_market(backfill_root,dates,wanted_ids):
    root=Path(backfill_root)
    races={}
    odds={}
    missing_day=[]
    missing_odds_day=[]
    for date in sorted(dates):
        day=root/"data"/"daily"/f"{date}.jsonl.gz"
        odd=root/"data"/"odds"/"daily"/f"{date}.jsonl.gz"
        if not day.exists():
            missing_day.append(date)
            continue
        with gzip.open(day,"rt",encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row=json.loads(line)
                rid=str((row.get("race") or {}).get("race_id") or "")
                if rid in wanted_ids:
                    races[rid]=row
        if not odd.exists():
            missing_odds_day.append(date)
            continue
        with gzip.open(odd,"rt",encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row=json.loads(line)
                rid=str(row.get("race_id") or "")
                if rid in wanted_ids:
                    odds[rid]=row
    return races,odds,missing_day,missing_odds_day


def horse_number_map(race_pack):
    out={}
    for row in race_pack.get("entries") or []:
        hid=str(row.get("horse_id") or "")
        no=row.get("horse_number")
        try:
            no=int(no)
        except (TypeError,ValueError):
            no=None
        if hid and no and no>0:
            out[hid]=no
    return out


def generate_tickets(pool,anchors):
    if len(anchors)<2:
        return []
    a1,a2=anchors[:2]
    rest=[x for x in pool if x not in {a1,a2}]
    rows=[]
    for hid in (a1,a2):
        rows.append(("WIN_CORE2","WIN",(hid,)))
    for mate in [x for x in pool if x!=a1]:
        rows.append(("QUINELLA_AXIS1","QUINELLA",(a1,mate)))
        rows.append(("EXACTA_MULTI1","EXACTA",(a1,mate)))
        rows.append(("EXACTA_MULTI1","EXACTA",(mate,a1)))
    for mate in rest:
        rows.append(("TRIO_AXIS12","TRIO",(a1,a2,mate)))
        for perm in itertools.permutations((a1,a2,mate),3):
            rows.append(("TRIFECTA_MULTI12","TRIFECTA",perm))
    mates=[x for x in pool if x!=a1]
    for b,c in itertools.combinations(mates,2):
        rows.append(("TRIO_AXIS1","TRIO",(a1,b,c)))
    return rows


def field(rec,key,default=None):
    value=rec
    for part in key.split("."):
        if not isinstance(value,dict):
            return default
        value=value.get(part)
    return default if value is None else value


def horse_feature(hid,anchor1,anchor2,novel,seven,outsider,seven_pos):
    s=seven.get(hid) or {}
    o=outsider.get(hid) or {}
    return {
        "king_support":float(s.get("support",0)),
        "king_borda":float(s.get("borda",0.0)),
        "king_best_rank":float(s.get("best_rank",99)),
        "king_mean_rank":float(s.get("mean_rank",99.0)),
        "king_top1_votes":float(s.get("top1_votes",0)),
        "consensus_position":float(seven_pos.get(hid,99)),
        "outsider_support":float(o.get("support",0)),
        "outsider_borda":float(o.get("borda",0.0)),
        "outsider_best_rank":float(o.get("best_rank",99)),
        "outsider_prob_mean":float(o.get("prob_mean",0.0)),
        "outsider_prob_max":float(o.get("prob_max",0.0)),
        "is_novel":1.0 if hid in novel else 0.0,
        "is_anchor1":1.0 if hid==anchor1 else 0.0,
        "is_anchor2":1.0 if hid==anchor2 else 0.0,
    }


def main():
    a=parse_args()
    candidate_root=Path(a.candidate_root)
    paths=parse_year_paths(a.router_year)
    if set(paths)!=set(YEARS):
        raise SystemExit(f"router years mismatch: {sorted(paths)}")

    decisions={}
    outsiders={}
    seven_declared={}
    gate_ids=set()
    race_year={}
    for year in YEARS:
        ydir=candidate_root/f"y{year}"
        drows=read_csv(ydir/"router-decisions.csv")
        orows=read_csv(ydir/"outsider-top6.csv")
        srows=read_csv(ydir/"seven-union.csv")
        decisions[year]={str(r["race_id"]):r for r in drows}
        outsiders[year]=outsider_maps(orows)
        seven_declared[year]={str(r["race_id"]):set(pipe_list(r["seven_union_horse_ids"])) for r in srows}
        for rid in decisions[year]:
            if rid in gate_ids:
                raise ValueError(f"duplicate Gate race_id across years: {rid}")
            gate_ids.add(rid)
            race_year[rid]=year
    if len(gate_ids)!=1384:
        raise SystemExit(f"Gate alert regression {len(gate_ids)} != 1384")

    routers={}
    dates=set()
    for year in YEARS:
        wanted=set(decisions[year])
        routers[year]=load_router(paths[year],wanted)
        dates.update(str(r.get("race_date") or "")[:10] for r in routers[year].values())

    races,odds_records,missing_day,missing_odds_day=load_market(a.backfill_root,dates,gate_ids)
    missing_race=sorted(gate_ids-set(races))
    if missing_day or missing_race:
        raise SystemExit(
            f"BACKFILL race-pack coverage broken missing_days={missing_day[:10]} "
            f"missing_races={missing_race[:10]} count={len(missing_race)}"
        )

    out_path=Path(a.output)
    out_path.parent.mkdir(parents=True,exist_ok=True)
    counters=defaultdict(int)
    priced_by_policy=defaultdict(int)
    generated_by_policy=defaultdict(int)
    races_by_policy=defaultdict(set)
    missing_odds_races=set()
    payout_type_missing=defaultdict(int)

    with gzip.open(out_path,"wt",encoding="utf-8") as out:
        for year in YEARS:
            for rid,decision in decisions[year].items():
                router=routers[year][rid]
                race_pack=races[rid]
                odds_record=odds_records.get(rid)
                if odds_record is None:
                    missing_odds_races.add(rid)
                    continue
                odds_map=decode_odds(odds_record)
                payouts,payout_types=payout_map(race_pack)
                horse_no=horse_number_map(race_pack)
                seven_order,seven=seven_stats(router)
                declared=seven_declared[year][rid]
                if set(seven_order)!=declared:
                    raise ValueError(
                        f"seven union mismatch race={rid} router={len(seven_order)} declared={len(declared)}"
                    )
                if len(seven_order)<2:
                    counters["too_small_seven_union"]+=1
                    continue
                anchor1,anchor2=seven_order[:2]
                seven_pos={hid:i+1 for i,hid in enumerate(seven_order)}
                label_maps=outsiders[year].get(rid) or {}
                gate_score=finite(decision.get("gate_score")) or 0.0
                blind=truthy(decision.get("blind"))
                race_meta=router.get("race") or {}
                consensus=router.get("consensus") or {}

                for policy in POLICIES:
                    labels=selected_labels(decision,policy)
                    o_stat=selected_outsider_stats(label_maps,labels)
                    novel=set(o_stat)-declared
                    novel_order=sorted(
                        novel,
                        key=lambda hid:(
                            -float((o_stat.get(hid) or {}).get("prob_max",0.0)),
                            -float((o_stat.get(hid) or {}).get("borda",0.0)),
                            float((o_stat.get(hid) or {}).get("best_rank",99)),
                            hid,
                        ),
                    )
                    pool=list(seven_order)+novel_order
                    if len(pool)!=len(set(pool)):
                        raise ValueError(f"duplicate pool horse race={rid} policy={policy}")
                    missing_no=[hid for hid in pool if hid not in horse_no]
                    if missing_no:
                        raise ValueError(
                            f"horse_number missing race={rid} policy={policy} sample={missing_no[:5]}"
                        )
                    generated=generate_tickets(pool,(anchor1,anchor2))
                    generated_by_policy[policy]+=len(generated)
                    races_by_policy[policy].add(rid)
                    horse_features={
                        hid:horse_feature(hid,anchor1,anchor2,novel,seven,o_stat,seven_pos)
                        for hid in pool
                    }
                    for strategy,bet_type,hids in generated:
                        generated_key=(bet_type,canonical_numbers(bet_type,[horse_no[x] for x in hids]))
                        if bet_type not in payout_types:
                            payout_type_missing[(year,bet_type)]+=1
                            continue
                        odd=odds_map.get(generated_key)
                        if odd is None:
                            counters[f"missing_odds_{bet_type}"]+=1
                            continue
                        ret=float(payouts.get(generated_key,0.0))
                        hit=ret>0
                        fs=[horse_features[hid] for hid in hids]
                        nums=[horse_no[hid] for hid in hids]
                        selected_key="-".join(map(str,generated_key[1]))
                        row={
                            "contract":"L2_CANDIDATE_PROFIT_TICKET_V1",
                            "year":year,
                            "race_id":rid,
                            "race_date":str(router.get("race_date") or "")[:10],
                            "policy":policy,
                            "strategy":strategy,
                            "bet_type":bet_type,
                            "selection_key":selected_key,
                            "selection_numbers":"-".join(map(str,nums)),
                            "selection_horse_ids":"|".join(hids),
                            "blind":blind,
                            "hit":hit,
                            "return_yen_per100":ret,
                            "odds":odd,
                            "log_odds":math.log(max(odd,1e-12)),
                            "implied_probability":1.0/odd,
                            "gate_score":gate_score,
                            "pool_size":len(pool),
                            "seven_union_count":len(seven_order),
                            "novel_pool_count":len(novel),
                            "selected_outsider_count":len(labels),
                            "ticket_novel_count":sum(int(x["is_novel"]) for x in fs),
                            "ticket_anchor_count":sum(int(x["is_anchor1"] or x["is_anchor2"]) for x in fs),
                            "king_support_sum":sum(x["king_support"] for x in fs),
                            "king_support_min":min(x["king_support"] for x in fs),
                            "king_support_max":max(x["king_support"] for x in fs),
                            "king_borda_sum":sum(x["king_borda"] for x in fs),
                            "king_borda_mean":mean([x["king_borda"] for x in fs]),
                            "king_top1_votes_sum":sum(x["king_top1_votes"] for x in fs),
                            "outsider_support_sum":sum(x["outsider_support"] for x in fs),
                            "outsider_support_max":max(x["outsider_support"] for x in fs),
                            "outsider_borda_sum":sum(x["outsider_borda"] for x in fs),
                            "outsider_prob_mean":mean([x["outsider_prob_mean"] for x in fs]),
                            "outsider_prob_max":max(x["outsider_prob_max"] for x in fs),
                            "venue_code":race_meta.get("venue_code"),
                            "surface":race_meta.get("surface"),
                            "race_class":race_meta.get("race_class"),
                            "discipline":race_meta.get("discipline"),
                            "direction":race_meta.get("direction"),
                            "weather":race_meta.get("weather"),
                            "track_condition":race_meta.get("track_condition"),
                            "distance_m":finite(race_meta.get("distance_m")) or 0.0,
                            "field_size":finite(race_meta.get("field_size")) or 0.0,
                            "cw_top1_max_vote_share":finite(consensus.get("top1_max_vote_share")) or 0.0,
                            "cw_top3_jaccard":finite(consensus.get("top3_pairwise_jaccard_mean")) or 0.0,
                            "cw_top6_jaccard":finite(consensus.get("top6_pairwise_jaccard_mean")) or 0.0,
                            "cw_rank_diff_mean":finite(consensus.get("pairwise_rank_abs_diff_mean")) or 0.0,
                            "cw_rank_std_mean":finite(consensus.get("horse_rank_std_mean")) or 0.0,
                            "cw_prob_std_mean":finite(consensus.get("horse_probability_std_mean")) or 0.0,
                            "cw_prob_std_max":finite(consensus.get("horse_probability_std_max")) or 0.0,
                        }
                        for idx in range(3):
                            f=fs[idx] if idx<len(fs) else {}
                            prefix=f"s{idx+1}_"
                            for key in (
                                "king_support","king_borda","king_best_rank","king_mean_rank",
                                "king_top1_votes","consensus_position","outsider_support",
                                "outsider_borda","outsider_best_rank","outsider_prob_mean",
                                "outsider_prob_max","is_novel","is_anchor1","is_anchor2",
                            ):
                                row[prefix+key]=float(f.get(key,0.0)) if f else 0.0
                        out.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
                        priced_by_policy[policy]+=1
                        counters["priced_rows"]+=1
                        if hit:
                            counters["hit_rows"]+=1

    coverage={
        "contract":"L2_CANDIDATE_PROFIT_DATASET_V1",
        "gate_alerts":len(gate_ids),
        "years":list(YEARS),
        "policies":list(POLICIES),
        "strategies":list(STRATEGY_ORDER),
        "backfill_race_packs":len(races),
        "historical_odds_races":len(odds_records),
        "missing_odds_races":len(missing_odds_races),
        "missing_odds_race_ids":sorted(missing_odds_races),
        "missing_day_files":missing_day,
        "missing_odds_day_files":missing_odds_day,
        "generated_tickets_by_policy":dict(generated_by_policy),
        "priced_tickets_by_policy":dict(priced_by_policy),
        "races_by_policy":{k:len(v) for k,v in races_by_policy.items()},
        "payout_type_missing_counts":{
            f"{year}:{bet}":count for (year,bet),count in sorted(payout_type_missing.items())
        },
        "counters":dict(counters),
        "ability_uses_odds":False,
        "l2_uses_final_odds":True,
        "odds_timestamp_policy":"FINAL_ODDS_PROXY",
        "locked_years":[2026],
        "place_bets_included":False,
        "wide_bets_included":False,
    }
    cov=Path(a.coverage_out)
    cov.parent.mkdir(parents=True,exist_ok=True)
    cov.write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    if any(priced_by_policy[p]==0 for p in POLICIES):
        raise SystemExit(f"empty priced policy dataset: {dict(priced_by_policy)}")
    print("L2_CANDIDATE_PROFIT_DATASET_READY")
    print(json.dumps({
        "gate_alerts":len(gate_ids),
        "priced":dict(priced_by_policy),
        "missing_odds_races":len(missing_odds_races),
        "output":str(out_path),
        "coverage":str(cov),
    },ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
