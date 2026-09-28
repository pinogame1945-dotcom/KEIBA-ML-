#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CONTRACT="L15_CONSENSUS_OUTSIDER_JOIN_V1"
EXPECTED_SELECTED=1384
EXPECTED_CAUGHT_BLIND=277
ALL_TEST_BLIND=1366

LABELS={
    "outsider_style":"位置取り型",
    "outsider_lap":"ラップ型",
    "outsider_elo":"Elo型",
    "outsider_network":"Elo型",
    "outsider_distance_detail":"距離詳細型",
    "outsider_distance":"距離詳細型",
    "outsider_race_condition":"レース条件型",
    "outsider_backfill":"レース条件型",
    "outsider_day_trend":"当日傾向型",
    "outsider_daytrend":"当日傾向型",
    "outsider_race_structure":"レース構造型",
    "outsider_raceshape":"レース構造型",
    "outsider_gate_course":"枠・コース型",
    "outsider_gatecourse":"枠・コース型",
    "outsider_jockey":"騎手型",
    "outsider_member_structure":"メンバー構成型",
    "outsider_field":"メンバー構成型",
    "outsider_chimera":"キメラ型",
}

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--selected",required=True)
    p.add_argument("--horse-scorecard",required=True)
    p.add_argument("--nonhorse-scorecard",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def read_selected(path):
    rows=[]
    with open(path,newline="",encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "year":int(r["year"]),
                "race_id":str(r["race_id"]),
                "score":float(r["score"]),
                "blind":str(r["blind"]).strip().lower()=="true",
            })
    ids=[r["race_id"] for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError("duplicate race_id in selected-top10-all")
    if len(rows)!=EXPECTED_SELECTED:
        raise ValueError(f"selected count mismatch: {len(rows)} != {EXPECTED_SELECTED}")
    blind=sum(r["blind"] for r in rows)
    if blind!=EXPECTED_CAUGHT_BLIND:
        raise ValueError(f"caught blind mismatch: {blind} != {EXPECTED_CAUGHT_BLIND}")
    return rows

def discover_rescue_sets(scorecard_path,source):
    obj=json.loads(Path(scorecard_path).read_text(encoding="utf-8"))
    out={}
    yearly=defaultdict(dict)
    for row in obj.get("fold_results") or []:
        if row.get("status")!="success":
            continue
        cand=str(row.get("candidate") or "")
        year=int(row.get("validation_year") or 0)
        if not cand or year not in (2022,2023,2024,2025):
            continue
        raw=((row.get("rescue") or {}).get("rescue_race_ids_by_topn") or {}).get("6") or []
        s=set(map(str,raw))
        yearly[cand][year]=s
    for cand,byyear in yearly.items():
        merged=set()
        for s in byyear.values():
            merged |= s
        out[cand]={
            "candidate":cand,
            "label":LABELS.get(cand,cand),
            "source":source,
            "by_year":byyear,
            "all":merged,
        }
    if not out:
        raise ValueError(f"no candidates discovered in {scorecard_path}")
    return out

def canonicalize_candidates(*sources):
    merged={}
    label_to_key={}
    for source in sources:
        for cand,info in source.items():
            label=info["label"]
            # If aliases map to the same human family, merge them into one canonical candidate.
            if label in label_to_key:
                key=label_to_key[label]
                for y,s in info["by_year"].items():
                    merged[key]["by_year"].setdefault(y,set()).update(s)
                    merged[key]["all"].update(s)
                continue
            key=cand
            label_to_key[label]=key
            merged[key]={
                "candidate":key,
                "label":label,
                "source":info["source"],
                "aliases":[cand],
                "by_year":{y:set(s) for y,s in info["by_year"].items()},
                "all":set(info["all"]),
            }
    return merged

def subset_for_selected(candidate,selected_blind_ids):
    return candidate["all"] & selected_blind_ids

def pairwise(candidates,rescue_sets):
    keys=sorted(candidates,key=lambda k:candidates[k]["label"])
    rows=[]
    for a,b in itertools.combinations(keys,2):
        sa=rescue_sets[a]; sb=rescue_sets[b]
        inter=len(sa&sb); union=len(sa|sb)
        rows.append({
            "a":candidates[a]["label"],
            "b":candidates[b]["label"],
            "a_count":len(sa),
            "b_count":len(sb),
            "intersection":inter,
            "union":union,
            "jaccard":inter/union if union else 0.0,
            "a_unique_vs_b":len(sa-sb),
            "b_unique_vs_a":len(sb-sa),
        })
    return rows

def union_of(keys,rescue_sets):
    s=set()
    for k in keys:
        s |= rescue_sets[k]
    return s

def best_combo(keys,rescue_sets,k):
    best=None
    for combo in itertools.combinations(keys,k):
        u=union_of(combo,rescue_sets)
        labels=tuple(combo)
        row=(len(u),labels,u)
        if best is None or row[0]>best[0] or (row[0]==best[0] and row[1]<best[1]):
            best=row
    return best

def analyze_year(year,rows,candidates):
    selected=[r for r in rows if r["year"]==year]
    blind_ids={r["race_id"] for r in selected if r["blind"]}
    candidate_counts={}
    for k,c in candidates.items():
        s=c["by_year"].get(year,set()) & blind_ids
        candidate_counts[k]=len(s)
    return {
        "year":year,
        "selected":len(selected),
        "blind":len(blind_ids),
        "candidate_counts":candidate_counts,
    }

def main():
    a=parse_args()
    selected=read_selected(a.selected)
    horse=discover_rescue_sets(a.horse_scorecard,"horse")
    nonhorse=discover_rescue_sets(a.nonhorse_scorecard,"nonhorse")
    candidates=canonicalize_candidates(horse,nonhorse)

    if len(candidates)<10:
        raise ValueError(f"too few canonical outsider candidates: {len(candidates)}")

    selected_ids={r["race_id"] for r in selected}
    selected_blind_ids={r["race_id"] for r in selected if r["blind"]}
    rescue_sets={k:subset_for_selected(c,selected_blind_ids) for k,c in candidates.items()}

    # Unique rescue within the selected true-blind population.
    exclusive={}
    for k,s in rescue_sets.items():
        n=0
        ids=[]
        for rid in s:
            supporters=sum(rid in s2 for s2 in rescue_sets.values())
            if supporters==1:
                n+=1; ids.append(rid)
        exclusive[k]={"count":n,"race_ids":sorted(ids)}

    with ThreadPoolExecutor(max_workers=4) as ex:
        yearly=list(ex.map(lambda y:analyze_year(y,selected,candidates),(2022,2023,2024,2025)))
    yearly=sorted(yearly,key=lambda x:x["year"])

    individual=[]
    for k,c in sorted(candidates.items(),key=lambda kv:(-len(rescue_sets[kv[0]]),kv[1]["label"])):
        s=rescue_sets[k]
        individual.append({
            "candidate":k,
            "label":c["label"],
            "source":c["source"],
            "rescued_gate_blind":len(s),
            "rescue_rate_within_277":len(s)/len(selected_blind_ids),
            "rescue_per_100_gate_alerts":100*len(s)/len(selected),
            "end_to_end_blind_rescue_rate":len(s)/ALL_TEST_BLIND,
            "exclusive_rescues_within_11":exclusive[k]["count"],
            "yearly":{
                str(y["year"]):y["candidate_counts"].get(k,0)
                for y in yearly
            },
        })

    keys=sorted(candidates)
    all_union=union_of(keys,rescue_sets)
    horse_keys=[k for k,c in candidates.items() if c["source"]=="horse"]
    nonhorse_keys=[k for k,c in candidates.items() if c["source"]=="nonhorse"]

    label_to_key={c["label"]:k for k,c in candidates.items()}
    primary_labels=["枠・コース型","レース構造型","当日傾向型"]
    primary_keys=[label_to_key[x] for x in primary_labels if x in label_to_key]
    shadow_labels=["メンバー構成型","騎手型"]
    shadow_keys=[label_to_key[x] for x in shadow_labels if x in label_to_key]

    groups={}
    for name,ks in (
        ("horse_union",horse_keys),
        ("nonhorse_union",nonhorse_keys),
        ("primary3_union",primary_keys),
        ("primary3_plus_shadow2",primary_keys+shadow_keys),
        ("all11_union",keys),
    ):
        u=union_of(ks,rescue_sets)
        groups[name]={
            "candidates":[candidates[k]["label"] for k in ks],
            "count":len(u),
            "rescue_rate_within_277":len(u)/len(selected_blind_ids),
            "rescue_per_100_gate_alerts":100*len(u)/len(selected),
            "end_to_end_blind_rescue_rate":len(u)/ALL_TEST_BLIND,
            "remaining_of_277":len(selected_blind_ids-u),
            "race_ids":sorted(u),
        }

    combo_rows=[]
    max_k=min(5,len(keys))
    for k in range(1,max_k+1):
        best=best_combo(keys,rescue_sets,k)
        combo,union_ids=best[1],best[2]
        combo_rows.append({
            "k":k,
            "candidates":[candidates[x]["label"] for x in combo],
            "rescued":len(union_ids),
            "rescue_rate_within_277":len(union_ids)/len(selected_blind_ids),
            "end_to_end_blind_rescue_rate":len(union_ids)/ALL_TEST_BLIND,
            "rescue_per_100_gate_alerts":100*len(union_ids)/len(selected),
        })

    row_by_id={r["race_id"]:dict(r) for r in selected}
    for rid,row in row_by_id.items():
        rescuers=[]
        for k,c in sorted(candidates.items(),key=lambda kv:kv[1]["label"]):
            hit=(rid in rescue_sets[k])
            row["rescue__"+k]=1 if hit else 0
            if hit:
                rescuers.append(c["label"])
        row["rescue_count"]=len(rescuers)
        row["rescuers"]="|".join(rescuers)
        row["any_outsider_rescue"]=bool(rescuers)

    blind_rows=[row_by_id[rid] for rid in sorted(selected_blind_ids)]
    false_alert_rows=[row_by_id[r["race_id"]] for r in selected if not r["blind"]]

    summary={
        "contract":CONTRACT,
        "selected_top10_count":len(selected),
        "selected_blind_count":len(selected_blind_ids),
        "gate_precision":len(selected_blind_ids)/len(selected),
        "all_test_blind_spots":ALL_TEST_BLIND,
        "gate_blind_recall":len(selected_blind_ids)/ALL_TEST_BLIND,
        "candidate_count":len(candidates),
        "candidate_labels":[candidates[k]["label"] for k in sorted(candidates)],
        "individual":individual,
        "groups":groups,
        "best_oracle_combinations":combo_rows,
        "pairwise":pairwise(candidates,rescue_sets),
        "yearly":[
            {
                "year":y["year"],
                "selected":y["selected"],
                "blind":y["blind"],
                "gate_precision":y["blind"]/y["selected"] if y["selected"] else None,
                "candidate_counts":{
                    candidates[k]["label"]:v
                    for k,v in y["candidate_counts"].items()
                },
            } for y in yearly
        ],
        "warnings":[
            "All outsider union/combo figures are hindsight/oracle capacity on the Gate-caught true-blind races, not deployable router performance.",
            "False-positive Gate races do not have outsider 'rescue' labels because rescue is defined only when seven kings miss the winner.",
            "Candidate inflation/harm on the 1,107 false alerts cannot be measured from rescue ledgers alone; outsider Top6 horse lists are needed for that downstream test.",
            "2026 is not used.",
            "Odds are not used.",
        ],
    }

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    # Race-level selected 1,384 CSV.
    ordered_candidates=sorted(candidates,key=lambda k:candidates[k]["label"])
    fields=["year","race_id","score","blind","rescue_count","rescuers","any_outsider_rescue"]+["rescue__"+k for k in ordered_candidates]
    with open(out/"selected-top10-outsider-join.csv","w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()
        for r in sorted(row_by_id.values(),key=lambda x:(x["year"],-x["score"],x["race_id"])):
            w.writerow({k:r.get(k,"") for k in fields})

    with open(out/"caught-blind-outsider-join.csv","w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()
        for r in sorted(blind_rows,key=lambda x:(x["year"],-x["score"],x["race_id"])):
            w.writerow({k:r.get(k,"") for k in fields})

    with open(out/"pairwise-overlap.csv","w",newline="",encoding="utf-8") as fh:
        fields2=["a","b","a_count","b_count","intersection","union","jaccard","a_unique_vs_b","b_unique_vs_a"]
        w=csv.DictWriter(fh,fieldnames=fields2)
        w.writeheader(); w.writerows(summary["pairwise"])

    with open(out/"best-oracle-combinations.csv","w",newline="",encoding="utf-8") as fh:
        fields3=["k","candidates","rescued","rescue_rate_within_277","end_to_end_blind_rescue_rate","rescue_per_100_gate_alerts"]
        w=csv.DictWriter(fh,fieldnames=fields3)
        w.writeheader()
        for row in combo_rows:
            x=dict(row); x["candidates"]="|".join(x["candidates"]); w.writerow(x)

    readme=[
        "# L1.5 Consensus-World × Outsider Join V1",
        "",
        f"- Gate alerts (top10%): {len(selected):,}",
        f"- True seven-king blind spots inside alerts: {len(selected_blind_ids):,}",
        f"- Gate precision: {100*len(selected_blind_ids)/len(selected):.2f}%",
        f"- Gate recall over all 2022–2025 seven-king blind spots: {100*len(selected_blind_ids)/ALL_TEST_BLIND:.2f}%",
        "",
        "## Individual outsider rescue inside Gate-caught blind spots",
        "",
        "| Outsider | Rescue | /277 | Exclusive | End-to-end /1366 |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in individual:
        readme.append(
            f"| {x['label']} | {x['rescued_gate_blind']} | {100*x['rescue_rate_within_277']:.2f}% | "
            f"{x['exclusive_rescues_within_11']} | {100*x['end_to_end_blind_rescue_rate']:.2f}% |"
        )
    readme += [
        "",
        "## Group oracle ceilings",
        "",
        "| Group | Rescue | /277 | End-to-end /1366 | Alerts per rescue |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("primary3_union","primary3_plus_shadow2","nonhorse_union","horse_union","all11_union"):
        x=groups[name]
        alerts_per_rescue=len(selected)/x["count"] if x["count"] else None
        readme.append(
            f"| {name} | {x['count']} | {100*x['rescue_rate_within_277']:.2f}% | "
            f"{100*x['end_to_end_blind_rescue_rate']:.2f}% | {alerts_per_rescue:.2f} |"
        )
    readme += [
        "",
        "## Best hindsight combinations",
        "",
        "| k | Outsiders | Rescue | /277 | End-to-end /1366 |",
        "|---:|---|---:|---:|---:|",
    ]
    for x in combo_rows:
        readme.append(
            f"| {x['k']} | {' + '.join(x['candidates'])} | {x['rescued']} | "
            f"{100*x['rescue_rate_within_277']:.2f}% | {100*x['end_to_end_blind_rescue_rate']:.2f}% |"
        )
    readme += [
        "",
        "> WARNING: outsider union/combinations are hindsight/oracle capacity, not deployable routing performance.",
        "> The next deployable problem is choosing the outsider before the result is known.",
        "",
    ]
    (out/"README.md").write_text("\n".join(readme),encoding="utf-8")

    print("L15_CONSENSUS_OUTSIDER_JOIN_RESULT")
    print(json.dumps({
        "selected":len(selected),
        "blind":len(selected_blind_ids),
        "candidate_count":len(candidates),
        "primary3_union":groups["primary3_union"]["count"],
        "all11_union":groups["all11_union"]["count"],
        "best_combinations":combo_rows,
        "out_dir":str(out),
    },ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
