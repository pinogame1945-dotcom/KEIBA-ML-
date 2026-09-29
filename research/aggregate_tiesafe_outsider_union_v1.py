#!/usr/bin/env python3
import argparse,json
from pathlib import Path

TOPN="6"

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--first",required=True)
    p.add_argument("--second",required=True)
    p.add_argument("--transition",required=True)
    p.add_argument("--output",required=True)
    return p.parse_args()

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def rows_by_year(score):
    out={}
    for row in score.get("fold_results") or []:
        if row.get("status")!="success": continue
        out.setdefault(int(row["validation_year"]),[]).append(row)
    return out

def candidate_sets(rows):
    out={}
    for row in rows:
        rescue=row.get("rescue") or {}
        four=(rescue.get("four_way_race_ids_by_topn") or {}).get(TOPN) or {}
        uni=set().union(*[set(map(str,four.get(k) or [])) for k in ("both_hit","kings_only","outsider_only","both_miss")])
        kings=set(map(str,four.get("both_hit") or []))|set(map(str,four.get("kings_only") or []))
        resc=set(map(str,(rescue.get("rescue_race_ids_by_topn") or {}).get(TOPN) or []))
        out[str(row["candidate"])]={
            "universe":uni,
            "kings_hit":kings,
            "rescues":resc,
        }
    return out

def group_summary(rows):
    cs=candidate_sets(rows)
    if not cs:
        return None
    names=sorted(cs)
    first=cs[names[0]]
    universe=first["universe"]
    kings_hit=first["kings_hit"]
    for name in names[1:]:
        if cs[name]["universe"]!=universe:
            raise SystemExit(f"universe mismatch: {name}")
        if cs[name]["kings_hit"]!=kings_hit:
            raise SystemExit(f"seven-king hit mismatch: {name}")
    blind=universe-kings_hit
    union=set().union(*(cs[n]["rescues"] for n in names))
    bad=union-blind
    if bad:
        raise SystemExit(f"rescue outside blind set: {len(bad)}")
    exclusive={}
    for name in names:
        others=set().union(*(cs[x]["rescues"] for x in names if x!=name)) if len(names)>1 else set()
        exclusive[name]=len(cs[name]["rescues"]-others)
    return {
        "candidate_count":len(names),
        "candidates":names,
        "races":len(universe),
        "seven_hit_count":len(kings_hit),
        "seven_blind_count":len(blind),
        "union_rescue_count":len(union),
        "union_rescue_rate":len(union)/len(blind) if blind else None,
        "both_miss_count":len(blind-union),
        "union_rescue_race_ids":sorted(union),
        "both_miss_race_ids":sorted(blind-union),
        "exclusive_rescue_counts":exclusive,
    }

def main():
    a=parse_args()
    scores=[load(a.first),load(a.second),load(a.transition)]
    by=[rows_by_year(x) for x in scores]
    years=sorted(set().union(*(set(x) for x in by)))
    out={
        "contract":"L1_OUTSIDER_TIESAFE_UNION_V1",
        "topn":6,
        "rank_tie_policy":"score_desc_then_horse_id_asc_v1",
        "years":{},
        "aggregate":{},
    }
    all_groups={"legacy10":[],"new7":[],"all17":[]}
    for year in years:
        first_rows=by[0].get(year,[])
        second_rows=by[1].get(year,[])
        trans_rows=by[2].get(year,[])
        legacy=first_rows+second_rows
        new7=trans_rows
        all17=legacy+new7
        y={
            "legacy10":group_summary(legacy),
            "new7":group_summary(new7),
            "all17":group_summary(all17),
        }
        out["years"][str(year)]=y
        for key in all_groups:
            if y[key]: all_groups[key].append(y[key])

    for key,items in all_groups.items():
        blind=sum(x["seven_blind_count"] for x in items)
        rescue=sum(x["union_rescue_count"] for x in items)
        out["aggregate"][key]={
            "years":len(items),
            "races":sum(x["races"] for x in items),
            "seven_blind_count":blind,
            "union_rescue_count":rescue,
            "union_rescue_rate":rescue/blind if blind else None,
            "both_miss_count":sum(x["both_miss_count"] for x in items),
        }

    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L1_OUTSIDER_TIESAFE_UNION_READY")
    print(json.dumps(out["aggregate"],ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
