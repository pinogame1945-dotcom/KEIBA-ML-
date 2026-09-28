#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

CONTRACT = "ROUTER_FEATURE_SNAPSHOT_V1"
FAMILIES = {
    "OPPONENT": ("opponent_",),
    "NETWORK": ("network_",),
    "LAP": ("lap_",),
    "STYLE": ("style_",),
    "DISTANCE": ("distx_",),
    "BACKFILL": ("backfill_",),
    "AUTO": ("auto_",),
    "PEDIGREE": ("ped_",),
    "ACTOR": ("actor_",),
    "TIME_PACE": ("timepace_",),
}
RACE_KEYS = [
    "venue_code", "surface", "discipline", "direction", "weather",
    "track_condition", "distance_m", "backfill_race_class_normalized",
    "backfill_field_size",
]
FORBIDDEN = {
    "winner_horse_ids", "actual_is_win", "actual_finish_position",
    "finish_position", "target", "payout", "payouts",
    "final_win_odds", "final_popularity",
}


def parse_args():
    p=argparse.ArgumentParser(description="Build pre-race Router Feature Snapshot V1 from persisted L1 expert score tables.")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--expert", action="append", required=True, help="name=path; repeat for every expert")
    p.add_argument("--output", required=True)
    return p.parse_args()


def open_text(path, mode="rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def clean_name(name):
    out=re.sub(r"[^A-Za-z0-9_]+","_",str(name)).strip("_")
    if not out:
        raise ValueError("empty expert name after normalization")
    return out


def finite(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None


def mean(values):
    vals=[x for x in values if x is not None]
    return sum(vals)/len(vals) if vals else None


def pop_std(values):
    vals=[float(x) for x in values if x is not None]
    return statistics.pstdev(vals) if len(vals) >= 2 else 0.0 if len(vals)==1 else None


def read_expert(path):
    races=defaultdict(list)
    meta={"expert_ids":set(),"model_versions":set(),"feature_sets":set()}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row=json.loads(line)
            if row.get("contract") != "L1_TO_L2_OUTPUT_CONTRACT_V1":
                raise ValueError(f"unexpected L1->L2 contract in {path}")
            rid=str(row.get("race_id") or "")
            hid=str(row.get("horse_id") or "")
            if not rid or not hid:
                raise ValueError(f"missing race_id/horse_id in {path}")
            races[rid].append(row)
            meta["expert_ids"].add(str(row.get("expert_id") or ""))
            meta["model_versions"].add(str(row.get("model_version") or ""))
            meta["feature_sets"].add(tuple(row.get("feature_sets") or []))
    if len(meta["expert_ids"]) != 1 or "" in meta["expert_ids"]:
        raise ValueError(f"expert_id mismatch in {path}: {meta['expert_ids']}")
    if len(meta["model_versions"]) != 1:
        raise ValueError(f"model_version mismatch in {path}")
    if len(meta["feature_sets"]) != 1:
        raise ValueError(f"feature_sets mismatch in {path}")
    for rid, rows in races.items():
        rows.sort(key=lambda r:int(r["predicted_rank"]))
        ranks=[int(r["predicted_rank"]) for r in rows]
        if ranks != list(range(1,len(rows)+1)):
            raise ValueError(f"{rid}: ranks must be contiguous 1..N")
    return races, {
        "expert_id":next(iter(meta["expert_ids"])),
        "model_version":next(iter(meta["model_versions"])),
        "feature_sets":list(next(iter(meta["feature_sets"]))),
    }


def read_snapshot(path, wanted):
    races=defaultdict(list)
    with open_text(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row=json.loads(line)
            rid=str(row.get("race_id") or "")
            if rid in wanted:
                races[rid].append(row)
    missing=sorted(wanted-set(races))
    if missing:
        raise ValueError(f"snapshot missing {len(missing)} requested races")
    return races


def jaccard(a,b):
    a=set(a); b=set(b)
    u=a|b
    return len(a&b)/len(u) if u else 1.0


def expert_view(rows, meta):
    probs=[finite(r.get("race_normalized_win_probability")) or 0.0 for r in rows]
    top=rows[0]
    n=len(rows)
    entropy=-sum(p*math.log(max(p,1e-15)) for p in probs)
    norm_entropy=entropy/math.log(n) if n>1 else 0.0
    p1=probs[0]
    p2=probs[1] if n>1 else None
    p3=probs[2] if n>2 else None
    return {
        "expert_id":meta["expert_id"],
        "model_version":meta["model_version"],
        "feature_sets":meta["feature_sets"],
        "top1_horse_id":str(top["horse_id"]),
        "top3_horse_ids":[str(x["horse_id"]) for x in rows[:3]],
        "top6_horse_ids":[str(x["horse_id"]) for x in rows[:6]],
        "top1_probability":p1,
        "top2_probability":p2,
        "top3_probability":p3,
        "top1_top2_gap":p1-p2 if p2 is not None else None,
        "top1_top3_gap":p1-p3 if p3 is not None else None,
        "top3_probability_mass":sum(probs[:3]),
        "top6_probability_mass":sum(probs[:6]),
        "normalized_entropy":norm_entropy,
        "top1_family_abs_share":top.get("family_abs_share") or {},
    }


def data_coverage(snapshot_rows):
    feats=[r.get("features") or {} for r in snapshot_rows]
    prior=[finite(f.get("prior_starts")) for f in feats]
    recent=[finite(f.get("recent_window_starts")) for f in feats]
    known=[x for x in prior if x is not None]
    result={
        "field_size":len(snapshot_rows),
        "prior_starts_mean":mean(prior),
        "prior_starts_zero_rate":sum(x==0 for x in known)/len(known) if known else None,
        "prior_starts_le1_rate":sum(x<=1 for x in known)/len(known) if known else None,
        "prior_starts_le2_rate":sum(x<=2 for x in known)/len(known) if known else None,
        "recent_window_starts_mean":mean(recent),
        "history_known_rate":len(known)/len(feats) if feats else None,
        "family_horse_coverage":{},
    }
    for family,prefixes in FAMILIES.items():
        covered=0
        for f in feats:
            found=False
            for k,v in f.items():
                if any(str(k).startswith(p) for p in prefixes) and v is not None:
                    found=True
                    break
            covered+=int(found)
        result["family_horse_coverage"][family]=covered/len(feats) if feats else None
    return result


def consensus(expert_rows, views):
    names=list(expert_rows)
    top1=[views[n]["top1_horse_id"] for n in names]
    votes=Counter(top1)
    pair_top3=[]
    pair_top6=[]
    rank_diffs=[]
    horse_ranks=defaultdict(list)
    horse_probs=defaultdict(list)
    for name in names:
        for row in expert_rows[name]:
            hid=str(row["horse_id"])
            horse_ranks[hid].append(float(row["predicted_rank"]))
            p=finite(row.get("race_normalized_win_probability"))
            if p is not None:
                horse_probs[hid].append(p)
    for i,a in enumerate(names):
        amap={str(r["horse_id"]):r for r in expert_rows[a]}
        for b in names[i+1:]:
            bmap={str(r["horse_id"]):r for r in expert_rows[b]}
            if set(amap)!=set(bmap):
                raise ValueError("cross-expert horse coverage differs")
            pair_top3.append(jaccard(views[a]["top3_horse_ids"],views[b]["top3_horse_ids"]))
            pair_top6.append(jaccard(views[a]["top6_horse_ids"],views[b]["top6_horse_ids"]))
            for hid in amap:
                rank_diffs.append(abs(int(amap[hid]["predicted_rank"])-int(bmap[hid]["predicted_rank"])))
    rank_stds=[pop_std(v) for v in horse_ranks.values()]
    prob_stds=[pop_std(v) for v in horse_probs.values()]
    return {
        "expert_count":len(names),
        "top1_unique_horses":len(votes),
        "top1_max_vote":max(votes.values()) if votes else 0,
        "top1_max_vote_share":max(votes.values())/len(names) if names else None,
        "top1_full_agreement":len(votes)==1,
        "top3_pairwise_jaccard_mean":mean(pair_top3),
        "top6_pairwise_jaccard_mean":mean(pair_top6),
        "pairwise_rank_abs_diff_mean":mean(rank_diffs),
        "pairwise_rank_abs_diff_max":max(rank_diffs) if rank_diffs else 0,
        "horse_rank_std_mean":mean(rank_stds),
        "horse_rank_std_max":max(rank_stds) if rank_stds else 0,
        "horse_probability_std_mean":mean(prob_stds),
        "horse_probability_std_max":max(prob_stds) if prob_stds else 0,
    }


def assert_no_forbidden(value, path="root"):
    if isinstance(value,dict):
        for k,v in value.items():
            if k in FORBIDDEN:
                raise ValueError(f"forbidden outcome key {k} at {path}")
            assert_no_forbidden(v,path+"."+str(k))
    elif isinstance(value,list):
        for i,v in enumerate(value):
            assert_no_forbidden(v,path+f"[{i}]")


def main():
    a=parse_args()
    specs=[]
    for spec in a.expert:
        if "=" not in spec:
            raise ValueError("--expert must be name=path")
        raw_name,path=spec.split("=",1)
        specs.append((clean_name(raw_name),path))
    names=[x[0] for x in specs]
    if len(names)<2:
        raise ValueError("Router Feature Snapshot requires at least two experts")
    if len(names)!=len(set(names)):
        raise ValueError("duplicate expert names")

    expert_races={}
    metas={}
    for name,path in specs:
        expert_races[name],metas[name]=read_expert(path)
    race_sets=[set(expert_races[n]) for n in names]
    if any(s!=race_sets[0] for s in race_sets[1:]):
        raise ValueError("expert race coverage differs")
    wanted=race_sets[0]
    snapshots=read_snapshot(a.snapshot,wanted)

    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    writer=gzip.open if str(out).endswith(".gz") else open
    count=0
    with writer(out,"wt",encoding="utf-8") as fh:
        for rid in sorted(wanted):
            rows=snapshots[rid]
            first=rows[0].get("features") or {}
            race={
                "venue_code":first.get("venue_code"),
                "surface":first.get("surface"),
                "race_class":first.get("backfill_race_class_normalized"),
                "discipline":first.get("discipline"),
                "direction":first.get("direction"),
                "weather":first.get("weather"),
                "track_condition":first.get("track_condition"),
                "distance_m":first.get("distance_m"),
                "field_size":first.get("backfill_field_size") or len(rows),
            }
            e_rows={n:expert_races[n][rid] for n in names}
            views={n:expert_view(e_rows[n],metas[n]) for n in names}
            date_candidates={str(r.get("race_date") or "")[:10] for n in names for r in e_rows[n]}
            date_candidates.discard("")
            record={
                "contract":CONTRACT,
                "race_id":rid,
                "race_date":sorted(date_candidates)[0] if date_candidates else str(first.get("race_date") or "")[:10],
                "race":race,
                "data_coverage":data_coverage(rows),
                "experts":views,
                "consensus":consensus(e_rows,views),
            }
            assert_no_forbidden(record)
            fh.write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n")
            count+=1
    print("ROUTER_FEATURE_SNAPSHOT_V1_OK")
    print(json.dumps({"races":count,"experts":names,"output":str(out)},ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
