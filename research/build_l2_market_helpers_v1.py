#!/usr/bin/env python3
import argparse
import csv
import gzip
import itertools
import json
import math
import re
from collections import defaultdict
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path

YEARS=(2022,2023,2024,2025)
EXPECTED_PER_YEAR=3456
EXPECTED_TOTAL=13824
EXPECTED_ALERTS=1384
BET_GROUP={"WIN":"1","QUINELLA":"4","EXACTA":"6","TRIO":"7","TRIFECTA":"8"}
ARITY={"WIN":1,"QUINELLA":2,"EXACTA":2,"TRIO":3,"TRIFECTA":3}
UNORDERED={"QUINELLA","TRIO"}

TEMPLATE_TO_BET={
    "WIN_ANCHOR1":"WIN",
    "WIN_ANCHORS2":"WIN",
    "WIN_ALL_CANDIDATES":"WIN",
    "QUINELLA_CORE12":"QUINELLA",
    "QUINELLA_AXIS1_KING":"QUINELLA",
    "QUINELLA_AXIS1_NOVEL":"QUINELLA",
    "QUINELLA_AXIS1_ALL":"QUINELLA",
    "QUINELLA_KING_TOP4_BOX":"QUINELLA",
    "EXACTA_CORE12_MULTI":"EXACTA",
    "EXACTA_AXIS1_FORWARD":"EXACTA",
    "EXACTA_AXIS1_REVERSE":"EXACTA",
    "EXACTA_AXIS1_MULTI":"EXACTA",
    "EXACTA_AXIS1_NOVEL_MULTI":"EXACTA",
    "TRIO_AXIS12_ALL":"TRIO",
    "TRIO_AXIS12_NOVEL":"TRIO",
    "TRIO_AXIS1_ALL":"TRIO",
    "TRIO_KING_TOP6_BOX":"TRIO",
    "TRIO_A1_KING_NOVEL":"TRIO",
    "TRIFECTA_ANCHOR12_MULTI":"TRIFECTA",
    "TRIFECTA_ANCHOR12_NOVEL_MULTI":"TRIFECTA",
    "TRIFECTA_A1_FIRST_ANCHOR2_MATE":"TRIFECTA",
    "TRIFECTA_ANCHORS_TOP2_MATE":"TRIFECTA",
    "TRIFECTA_KING_TOP4_BOX":"TRIFECTA",
}


def parse_args():
    p=argparse.ArgumentParser(description="Build L2 Bet Kings Arena V1 ticket datasets from frozen L1.5.")
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
        if isinstance(value,str):
            value=value.replace(",","").strip()
        x=float(value)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None


def mean(values):
    vals=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(vals)/len(vals) if vals else 0.0


def load_fixed_ledgers(root):
    out={}
    count=0
    root=Path(root)
    for year in YEARS:
        path=root/f"y{year}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing fixed L1.5 ledger: {path}")
        year_rows={}
        with open(path,encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row=json.loads(line)
                if row.get("contract")!="L15_FIXED_OUTPUT_V1":
                    raise ValueError(f"unexpected fixed contract in {path}")
                rid=str(row.get("race_id") or "")
                if not rid or rid in year_rows:
                    raise ValueError(f"bad fixed race_id year={year}: {rid}")
                if row.get("gate_alert") is not True or row.get("gate_action")!="INTERVENE_FULL_K2":
                    raise ValueError(f"fixed ledger row is not K2 Gate alert race={rid}")
                if len(row.get("selected_outsiders") or [])!=2:
                    raise ValueError(f"fixed K2 selector count drift race={rid}")
                year_rows[rid]=row
                count+=1
        out[year]=year_rows
    if count!=EXPECTED_ALERTS:
        raise SystemExit(f"fixed alert count regression: {count} != {EXPECTED_ALERTS}")
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
                raise ValueError(f"router duplicate/missing race_id year={year}: {rid}")
            rows[rid]=row
    if len(rows)!=EXPECTED_PER_YEAR:
        raise SystemExit(f"router count regression year={year}: {len(rows)} != {EXPECTED_PER_YEAR}")
    return rows


def seven_stats(router_row):
    experts=router_row.get("experts") or {}
    if len(experts)!=7:
        raise ValueError(f"expected 7 experts race={router_row.get('race_id')} got={len(experts)}")
    stat=defaultdict(lambda:{
        "support":0,"borda":0.0,"best_rank":99,"rank_sum":0.0,"top1_votes":0,
    })
    for expert in experts.values():
        ids=[str(x) for x in (expert.get("top6_horse_ids") or []) if str(x)]
        if not ids:
            raise ValueError(f"empty expert Top6 race={router_row.get('race_id')}")
        for rank,hid in enumerate(ids,1):
            s=stat[hid]
            s["support"]+=1
            s["borda"]+=float(7-rank)
            s["best_rank"]=min(s["best_rank"],rank)
            s["rank_sum"]+=rank
            s["top1_votes"]+=int(rank==1)
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


def canonical_numbers(bet_type,values):
    vals=tuple(int(x) for x in values)
    return tuple(sorted(vals)) if bet_type in UNORDERED else vals


def final_odds_tuple(raw):
    if not isinstance(raw,list) or len(raw)<3:
        return None
    return raw[3:6] if len(raw)>=6 else raw[:3]


@lru_cache(maxsize=16384)
def parse_odds_key(bet_type,key):
    # Combination keys repeat across races; parse each structural key once.
    key=str(key)
    n=ARITY[bet_type]
    if n==1:
        if len(key) not in (1,2) or not key.isascii() or not key.isdigit():
            return None
        nums=(int(key),)
    else:
        if len(key)!=(2*n) or not key.isascii() or not key.isdigit():
            return None
        nums=tuple(int(key[i*2:i*2+2]) for i in range(n))
    if any(x<=0 for x in nums) or (n>1 and len(set(nums))!=n):
        return None
    return canonical_numbers(bet_type,nums)


def iter_decoded_odds(record):
    # Streaming decoder: avoid allocating a large intermediate dict on full-field scans.
    root=record.get("odds") or {}
    for bet_type,group in BET_GROUP.items():
        data=root.get(group)
        if not isinstance(data,dict):
            continue
        for key,raw in data.items():
            nums=parse_odds_key(bet_type,key)
            if nums is None:
                continue
            tup=final_odds_tuple(raw)
            if not tup:
                continue
            price=finite(tup[0])
            if price is None or price<=0:
                continue
            yield bet_type,nums,price


def decode_odds(record):
    # Compatibility wrapper. New large scans should consume iter_decoded_odds directly.
    return {(bet_type,nums):price for bet_type,nums,price in iter_decoded_odds(record)}

def payout_map(race_pack):
    out={}
    present=set()
    for row in race_pack.get("payouts") or []:
        bet_type=str(row.get("bet_type") or "")
        if bet_type not in BET_GROUP:
            continue
        present.add(bet_type)
        nums=[int(x) for x in re.findall(r"\d+",str(row.get("combination") or ""))]
        if len(nums)!=ARITY[bet_type] or any(x<=0 for x in nums):
            continue
        payout=finite(row.get("payout_yen"))
        if payout is None or payout<=0:
            continue
        out[(bet_type,canonical_numbers(bet_type,nums))]=payout
    return out,present


def load_day(path,wanted):
    rows={}
    if not path.exists():
        raise FileNotFoundError(str(path))
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
    if not path.exists():
        raise FileNotFoundError(str(path))
    with gzip.open(path,"rt",encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            if rid in wanted:
                rows[rid]=row
    return rows


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


def uniq_rows(rows,bet_type):
    seen=set()
    out=[]
    for vals in rows:
        if len(vals)!=ARITY[bet_type] or len(set(vals))!=len(vals):
            continue
        key=tuple(sorted(vals)) if bet_type in UNORDERED else tuple(vals)
        if key in seen:
            continue
        seen.add(key)
        out.append(tuple(vals))
    return out


def generate_templates(seven_order,novel,candidates):
    if len(seven_order)<2:
        return {}
    a1,a2=seven_order[:2]
    king_rest=[x for x in seven_order if x!=a1]
    all_rest=[x for x in candidates if x!=a1]
    third_all=[x for x in candidates if x not in {a1,a2}]
    third_king=[x for x in seven_order if x not in {a1,a2}]
    top4=seven_order[:4]
    top6=seven_order[:6]

    out={}
    out["WIN_ANCHOR1"]=[(a1,)]
    out["WIN_ANCHORS2"]=[(a1,),(a2,)]
    out["WIN_ALL_CANDIDATES"]=[(x,) for x in candidates]

    out["QUINELLA_CORE12"]=[(a1,a2)]
    out["QUINELLA_AXIS1_KING"]=[(a1,x) for x in king_rest]
    out["QUINELLA_AXIS1_NOVEL"]=[(a1,x) for x in novel]
    out["QUINELLA_AXIS1_ALL"]=[(a1,x) for x in all_rest]
    out["QUINELLA_KING_TOP4_BOX"]=list(itertools.combinations(top4,2))

    out["EXACTA_CORE12_MULTI"]=[(a1,a2),(a2,a1)]
    out["EXACTA_AXIS1_FORWARD"]=[(a1,x) for x in all_rest]
    out["EXACTA_AXIS1_REVERSE"]=[(x,a1) for x in all_rest]
    out["EXACTA_AXIS1_MULTI"]=[
        row for x in all_rest for row in ((a1,x),(x,a1))
    ]
    out["EXACTA_AXIS1_NOVEL_MULTI"]=[
        row for x in novel for row in ((a1,x),(x,a1))
    ]

    out["TRIO_AXIS12_ALL"]=[(a1,a2,x) for x in third_all]
    out["TRIO_AXIS12_NOVEL"]=[(a1,a2,x) for x in novel if x not in {a1,a2}]
    out["TRIO_AXIS1_ALL"]=[
        (a1,b,c) for b,c in itertools.combinations(all_rest,2)
    ]
    out["TRIO_KING_TOP6_BOX"]=list(itertools.combinations(top6,3))
    out["TRIO_A1_KING_NOVEL"]=[
        (a1,k,n)
        for k in seven_order if k not in {a1}
        for n in novel if n not in {a1,k}
    ]

    anchor12_multi=[]
    anchor12_novel=[]
    a1_first=[]
    anchors_top2=[]
    for mate in third_all:
        vals=(a1,a2,mate)
        anchor12_multi.extend(itertools.permutations(vals,3))
        a1_first.extend(((a1,a2,mate),(a1,mate,a2)))
        anchors_top2.extend(((a1,a2,mate),(a2,a1,mate)))
    for mate in novel:
        if mate in {a1,a2}:
            continue
        anchor12_novel.extend(itertools.permutations((a1,a2,mate),3))
    out["TRIFECTA_ANCHOR12_MULTI"]=anchor12_multi
    out["TRIFECTA_ANCHOR12_NOVEL_MULTI"]=anchor12_novel
    out["TRIFECTA_A1_FIRST_ANCHOR2_MATE"]=a1_first
    out["TRIFECTA_ANCHORS_TOP2_MATE"]=anchors_top2
    out["TRIFECTA_KING_TOP4_BOX"]=list(itertools.permutations(top4,3))

    return {
        template:uniq_rows(rows,TEMPLATE_TO_BET[template])
        for template,rows in out.items()
    }


def horse_feature(hid,seven_stats_map,seven_pos,candidate_pos,novel_pos,a1,a2):
    s=seven_stats_map.get(hid) or {}
    return {
        "king_support":float(s.get("support",0)),
        "king_borda":float(s.get("borda",0.0)),
        "king_best_rank":float(s.get("best_rank",99)),
        "king_mean_rank":float(s.get("mean_rank",99.0)),
        "king_top1_votes":float(s.get("top1_votes",0)),
        "consensus_position":float(seven_pos.get(hid,99)),
        "candidate_position":float(candidate_pos.get(hid,99)),
        "novel_position":float(novel_pos.get(hid,0)),
        "is_novel":1.0 if hid in novel_pos else 0.0,
        "is_anchor1":1.0 if hid==a1 else 0.0,
        "is_anchor2":1.0 if hid==a2 else 0.0,
    }


def main():
    a=parse_args()
    paths=parse_year_paths(a.router_year)
    if set(paths)!=set(YEARS):
        raise SystemExit(f"router years mismatch: {sorted(paths)}")
    fixed=load_fixed_ledgers(a.fixed_ledger_dir)

    routers={}
    race_owner={}
    date_to_races=defaultdict(list)
    total=0
    for year in YEARS:
        routers[year]=load_router(paths[year],year)
        for rid,row in routers[year].items():
            if rid in race_owner:
                raise ValueError(f"duplicate race_id across years: {rid}")
            date=str(row.get("race_date") or "")[:10]
            if len(date)!=10:
                raise ValueError(f"router race_date missing race={rid}")
            race_owner[rid]=year
            date_to_races[date].append(rid)
            total+=1
    if total!=EXPECTED_TOTAL:
        raise SystemExit(f"total router races regression: {total} != {EXPECTED_TOTAL}")

    for year in YEARS:
        missing=set(fixed[year])-set(routers[year])
        if missing:
            raise ValueError(f"fixed alerts missing in router y{year}: {sorted(missing)[:10]}")

    out_dir=Path(a.out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    counters=defaultdict(int)
    template_rows=defaultdict(int)
    template_hits=defaultdict(int)
    template_races=defaultdict(set)
    missing_odds_by_template=defaultdict(int)
    missing_day_files=[]
    missing_odds_files=[]
    processed=set()

    with ExitStack() as stack:
        handles={
            template:stack.enter_context(
                gzip.open(out_dir/f"{template}.jsonl.gz","wt",encoding="utf-8")
            )
            for template in TEMPLATE_TO_BET
        }
        backfill=Path(a.backfill_root)
        for date in sorted(date_to_races):
            wanted=set(date_to_races[date])
            day_path=backfill/"data"/"daily"/f"{date}.jsonl.gz"
            odds_path=backfill/"data"/"odds"/"daily"/f"{date}.jsonl.gz"
            try:
                day=load_day(day_path,wanted)
            except FileNotFoundError:
                missing_day_files.append(date)
                continue
            try:
                odds_day=load_odds_day(odds_path,wanted)
            except FileNotFoundError:
                missing_odds_files.append(date)
                continue
            for rid in sorted(wanted):
                year=race_owner[rid]
                router=routers[year][rid]
                race_pack=day.get(rid)
                odds_record=odds_day.get(rid)
                if race_pack is None:
                    counters["race_pack_row_missing"]+=1
                    continue
                if odds_record is None:
                    counters["odds_race_missing"]+=1
                    continue
                processed.add(rid)

                seven_order,seven_map=seven_stats(router)
                fixed_row=fixed[year].get(rid)
                if fixed_row:
                    if fixed_row["seven_consensus_order"]!=seven_order:
                        raise ValueError(f"fixed seven consensus drift race={rid}")
                    candidates=[str(x) for x in fixed_row["candidate_horse_ids"]]
                    novel=[str(x) for x in fixed_row["novel_horse_ids"]]
                    anchors=[str(x) for x in fixed_row["seven_anchor_horse_ids"]]
                    if anchors!=seven_order[:2]:
                        raise ValueError(f"fixed anchor drift race={rid}")
                    gate_alert=True
                    gate_score=finite(fixed_row.get("gate_score")) or 0.0
                    selected_outsiders=list(fixed_row.get("selected_outsiders") or [])
                else:
                    candidates=list(seven_order)
                    novel=[]
                    anchors=seven_order[:2]
                    gate_alert=False
                    gate_score=0.0
                    selected_outsiders=[]

                if len(anchors)<2:
                    counters["anchor_missing"]+=1
                    continue
                a1,a2=anchors
                horse_no=horse_number_map(race_pack)
                missing_horses=[hid for hid in candidates if hid not in horse_no]
                if missing_horses:
                    raise ValueError(f"candidate horse_number missing race={rid} sample={missing_horses[:5]}")
                odds_map=decode_odds(odds_record)
                payouts,payout_types=payout_map(race_pack)
                templates=generate_templates(seven_order,novel,candidates)

                seven_pos={hid:i+1 for i,hid in enumerate(seven_order)}
                candidate_pos={hid:i+1 for i,hid in enumerate(candidates)}
                novel_pos={hid:i+1 for i,hid in enumerate(novel)}
                hfeat={
                    hid:horse_feature(
                        hid,seven_map,seven_pos,candidate_pos,novel_pos,a1,a2
                    )
                    for hid in candidates
                }
                race_meta=router.get("race") or {}
                consensus=router.get("consensus") or {}
                outs1=selected_outsiders[0] if len(selected_outsiders)>0 else "__NONE__"
                outs2=selected_outsiders[1] if len(selected_outsiders)>1 else "__NONE__"

                for template,tickets in templates.items():
                    bet_type=TEMPLATE_TO_BET[template]
                    if bet_type not in payout_types:
                        counters[f"payout_type_missing_{bet_type}"]+=1
                        continue
                    for hids in tickets:
                        nums=[horse_no[x] for x in hids]
                        key=(bet_type,canonical_numbers(bet_type,nums))
                        odd=odds_map.get(key)
                        if odd is None:
                            missing_odds_by_template[template]+=1
                            continue
                        ret=float(payouts.get(key,0.0))
                        hit=ret>0
                        fs=[hfeat[hid] for hid in hids]
                        row={
                            "contract":"L2_BET_KINGS_TICKET_V1",
                            "year":year,
                            "race_id":rid,
                            "race_date":date,
                            "bet_type":bet_type,
                            "template":template,
                            "selection_key":"-".join(map(str,key[1])),
                            "selection_numbers":"-".join(map(str,nums)),
                            "selection_horse_ids":"|".join(hids),
                            "hit":hit,
                            "return_yen_per100":ret,
                            "odds":odd,
                            "gate_alert":1 if gate_alert else 0,
                            "gate_score_alert_only":gate_score,
                            "selected_outsider_1":outs1,
                            "selected_outsider_2":outs2,
                            "candidate_pool_size":len(candidates),
                            "seven_union_count":len(seven_order),
                            "novel_pool_count":len(novel),
                            "ticket_novel_count":sum(int(f["is_novel"]) for f in fs),
                            "ticket_anchor_count":sum(int(f["is_anchor1"] or f["is_anchor2"]) for f in fs),
                            "king_support_sum":sum(f["king_support"] for f in fs),
                            "king_support_min":min(f["king_support"] for f in fs),
                            "king_support_max":max(f["king_support"] for f in fs),
                            "king_borda_sum":sum(f["king_borda"] for f in fs),
                            "king_borda_mean":mean([f["king_borda"] for f in fs]),
                            "king_top1_votes_sum":sum(f["king_top1_votes"] for f in fs),
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
                            for name in (
                                "king_support","king_borda","king_best_rank","king_mean_rank",
                                "king_top1_votes","consensus_position","candidate_position",
                                "novel_position","is_novel","is_anchor1","is_anchor2",
                            ):
                                row[prefix+name]=float(f.get(name,0.0)) if f else 0.0
                        handles[template].write(
                            json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n"
                        )
                        template_rows[template]+=1
                        template_hits[template]+=int(hit)
                        template_races[template].add(rid)

    if missing_day_files or missing_odds_files:
        raise SystemExit(
            f"market pack files missing race_days={missing_day_files[:10]} "
            f"odds_days={missing_odds_files[:10]}"
        )
    if len(processed)!=EXPECTED_TOTAL:
        missing=sorted(set(race_owner)-processed)
        raise SystemExit(f"market race coverage regression processed={len(processed)} missing={missing[:10]}")

    manifest={
        "contract":"L2_BET_KINGS_DATASET_V1",
        "source_l15":"L15_FIXED_V1",
        "years":list(YEARS),
        "races":len(processed),
        "gate_alerts":sum(len(fixed[y]) for y in YEARS),
        "templates":{},
        "missing_odds_by_template":dict(sorted(missing_odds_by_template.items())),
        "counters":dict(counters),
        "probability_model_uses_odds":False,
        "market_price_stage":"FINAL_ODDS_AFTER_PREDICTION",
        "locked_years":[2026],
        "place_bets":False,
        "wide_bets":False,
    }
    for template in sorted(TEMPLATE_TO_BET):
        manifest["templates"][template]={
            "bet_type":TEMPLATE_TO_BET[template],
            "priced_rows":template_rows[template],
            "hits":template_hits[template],
            "races":len(template_races[template]),
            "file":f"{template}.jsonl.gz",
        }
        if template_rows[template]==0:
            raise SystemExit(f"empty template dataset: {template}")

    (out_dir/"manifest.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("L2_BET_KINGS_DATASET_V1_READY")
    print(json.dumps({
        "races":len(processed),
        "gate_alerts":manifest["gate_alerts"],
        "templates":len(TEMPLATE_TO_BET),
        "priced_rows":sum(template_rows.values()),
        "out_dir":str(out_dir),
    },ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
