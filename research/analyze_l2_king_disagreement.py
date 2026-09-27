#!/usr/bin/env python3
import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    p=argparse.ArgumentParser(description="Decompose disagreement between two persisted L1 expert outputs.")
    p.add_argument("--light", required=True)
    p.add_argument("--core", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")


def read_expert(path):
    races=defaultdict(list)
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row=json.loads(line)
            races[str(row["race_id"])].append(row)
    for race_id, rows in races.items():
        rows.sort(key=lambda x:int(x["predicted_rank"]))
        ranks=[int(x["predicted_rank"]) for x in rows]
        if ranks[0] != 1:
            raise ValueError(f"{race_id}: missing rank 1")
    return races


def distance_band(v):
    try:
        d=int(float(v))
    except (TypeError,ValueError):
        return "UNKNOWN"
    if d <= 1400: return "<=1400"
    if d <= 1800: return "1500-1800"
    if d <= 2200: return "1900-2200"
    return ">=2300"


def field_band(v):
    try:
        n=int(float(v))
    except (TypeError,ValueError):
        return "UNKNOWN"
    if n <= 10: return "<=10"
    if n <= 14: return "11-14"
    return ">=15"


def read_truth(snapshot, wanted):
    truth={}
    with open_text(snapshot) as fh:
        for line in fh:
            if not line.strip():
                continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            if rid not in wanted:
                continue
            t=row.get("target") or {}
            f=row.get("features") or {}
            info=truth.setdefault(rid,{
                "winners":set(),
                "race_date":str(f.get("race_date") or "")[:10],
                "surface":f.get("surface"),
                "distance_m":f.get("distance_m"),
                "venue_code":f.get("venue_code"),
                "race_class":f.get("backfill_race_class_normalized"),
                "field_size":f.get("backfill_field_size"),
                "discipline":f.get("discipline"),
            })
            if t.get("is_win") is True:
                info["winners"].add(str(row.get("horse_id") or ""))
    missing=sorted(wanted-set(truth))
    if missing:
        raise ValueError(f"snapshot missing {len(missing)} requested races")
    no_winner=[rid for rid,x in truth.items() if not x["winners"]]
    if no_winner:
        raise ValueError(f"{len(no_winner)} requested races have no winner target")
    return truth


def hit_within(rows,winners,n):
    return any(str(r["horse_id"]) in winners and int(r["predicted_rank"]) <= n for r in rows)


def group_stats(records,key):
    groups=defaultdict(list)
    for r in records:
        groups[str(r.get(key) or "UNKNOWN")].append(r)
    out=[]
    for name,rows in groups.items():
        n=len(rows)
        light=sum(x["light_top1_hit"] for x in rows)
        core=sum(x["core_top1_hit"] for x in rows)
        neither=sum((not x["light_top1_hit"] and not x["core_top1_hit"]) for x in rows)
        out.append({
            "value":name,
            "races":n,
            "light_top1_hits":light,
            "core_top1_hits":core,
            "light_top1_rate":light/n if n else None,
            "core_top1_rate":core/n if n else None,
            "core_minus_light_pp":((core-light)/n*100) if n else None,
            "neither_rate":neither/n if n else None,
        })
    return sorted(out,key=lambda x:(-x["races"],x["value"]))


def main():
    a=parse_args()
    light=read_expert(a.light)
    core=read_expert(a.core)
    if set(light)!=set(core):
        raise ValueError("expert race coverage differs")
    race_ids=set(light)
    truth=read_truth(a.snapshot,race_ids)

    category=Counter()
    disagreement=[]
    agreement=[]
    chooser=Counter()
    all_rows=[]

    for rid in sorted(race_ids):
        l=light[rid]; c=core[rid]; t=truth[rid]
        winners=t["winners"]
        l1=str(l[0]["horse_id"]); c1=str(c[0]["horse_id"])
        lh=l1 in winners; ch=c1 in winners
        record={
            "race_id":rid,
            **{k:v for k,v in t.items() if k!="winners"},
            "distance_band":distance_band(t.get("distance_m")),
            "field_band":field_band(t.get("field_size") or len(l)),
            "light_top1":l1,
            "core_top1":c1,
            "light_top1_hit":lh,
            "core_top1_hit":ch,
            "light_top3_hit":hit_within(l,winners,3),
            "core_top3_hit":hit_within(c,winners,3),
            "light_top6_hit":hit_within(l,winners,6),
            "core_top6_hit":hit_within(c,winners,6),
            "light_winner_rank":min(int(x["predicted_rank"]) for x in l if str(x["horse_id"]) in winners),
            "core_winner_rank":min(int(x["predicted_rank"]) for x in c if str(x["horse_id"]) in winners),
            "light_gap":float((l[0].get("race_summary") or {}).get("top1_top2_gap") or 0),
            "core_gap":float((c[0].get("race_summary") or {}).get("top1_top2_gap") or 0),
            "light_top1_prob":float(l[0].get("race_normalized_win_probability") or 0),
            "core_top1_prob":float(c[0].get("race_normalized_win_probability") or 0),
        }
        all_rows.append(record)

        if l1 == c1:
            if lh: category["agree_correct"] += 1
            else: category["agree_wrong"] += 1
            agreement.append(record)
        else:
            if lh and ch: category["disagree_both_correct"] += 1
            elif lh: category["disagree_light_only"] += 1
            elif ch: category["disagree_core_only"] += 1
            else: category["disagree_neither"] += 1
            disagreement.append(record)

            # Exploratory gate: choose the expert expressing a larger top1-vs-top2 gap.
            if record["light_gap"] > record["core_gap"]:
                chooser["gap_choose_light"] += 1
                chooser["gap_correct"] += int(lh)
            elif record["core_gap"] > record["light_gap"]:
                chooser["gap_choose_core"] += 1
                chooser["gap_correct"] += int(ch)
            else:
                chooser["gap_tie"] += 1

    n=len(all_rows); nd=len(disagreement); na=len(agreement)
    l_hits=sum(x["light_top1_hit"] for x in disagreement)
    c_hits=sum(x["core_top1_hit"] for x in disagreement)

    result={
        "contract":"L2_KING_DISAGREEMENT_ANALYSIS_V1",
        "races":n,
        "agreement":{
            "races":na,
            "rate":na/n,
            "correct":category["agree_correct"],
            "wrong":category["agree_wrong"],
            "top1_accuracy":category["agree_correct"]/na if na else None,
        },
        "disagreement":{
            "races":nd,
            "rate":nd/n,
            "light_only_correct":category["disagree_light_only"],
            "core_only_correct":category["disagree_core_only"],
            "both_correct_dead_heat_possible":category["disagree_both_correct"],
            "neither_correct":category["disagree_neither"],
            "light_top1_accuracy":l_hits/nd if nd else None,
            "core_top1_accuracy":c_hits/nd if nd else None,
            "either_top1_oracle_ceiling":sum(x["light_top1_hit"] or x["core_top1_hit"] for x in disagreement)/nd if nd else None,
            "light_top3_capture":sum(x["light_top3_hit"] for x in disagreement)/nd if nd else None,
            "core_top3_capture":sum(x["core_top3_hit"] for x in disagreement)/nd if nd else None,
            "top3_union_capture":sum(x["light_top3_hit"] or x["core_top3_hit"] for x in disagreement)/nd if nd else None,
            "light_top6_capture":sum(x["light_top6_hit"] for x in disagreement)/nd if nd else None,
            "core_top6_capture":sum(x["core_top6_hit"] for x in disagreement)/nd if nd else None,
            "top6_union_capture":sum(x["light_top6_hit"] or x["core_top6_hit"] for x in disagreement)/nd if nd else None,
            "mean_light_winner_rank":sum(x["light_winner_rank"] for x in disagreement)/nd if nd else None,
            "mean_core_winner_rank":sum(x["core_winner_rank"] for x in disagreement)/nd if nd else None,
        },
        "exploratory_gap_gate":{
            "note":"Exploratory only; compared on the same evaluation races and therefore not a validated routing rule.",
            "choose_light":chooser["gap_choose_light"],
            "choose_core":chooser["gap_choose_core"],
            "ties":chooser["gap_tie"],
            "correct":chooser["gap_correct"],
            "accuracy_non_ties":chooser["gap_correct"]/(chooser["gap_choose_light"]+chooser["gap_choose_core"]) if (chooser["gap_choose_light"]+chooser["gap_choose_core"]) else None,
        },
        "disagreement_groups":{
            "race_class":group_stats(disagreement,"race_class"),
            "surface":group_stats(disagreement,"surface"),
            "distance_band":group_stats(disagreement,"distance_band"),
            "field_band":group_stats(disagreement,"field_band"),
            "venue_code":group_stats(disagreement,"venue_code"),
        },
    }
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_KING_DISAGREEMENT_ANALYSIS_OK")
    print(json.dumps(result,ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
