#!/usr/bin/env python3
import argparse,gzip,json
from collections import defaultdict
from pathlib import Path

OLD_KINGS={
    "core4_no_pedigree",
    "no_auto_full_pedigree_legacy",
    "jockey_trainer_condition",
    "no_auto_full",
    "no_auto_jockey",
}
NEW_KINGS={
    "no_auto_full_pedigree_v1",
    "jockey_trainer_condition_plus_raceclass",
}
ROLES=("ANCHOR","MAINLINE","COVER")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--year-file",action="append",required=True,
                   help="YEAR:FIVE_PATH:SEVEN_PATH")
    p.add_argument("--output",required=True)
    return p.parse_args()

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def load_groups(path):
    groups=defaultdict(list)
    with open_text(path) as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            key=(str(r["race_id"]),str(r["role"]),int(float(r["top_n"])))
            groups[key].append({
                "expert":str(r["expert_name"]),
                "hit":int(bool(r["role_hit"])),
                "p":float(r["predicted_role_hit_probability"]),
            })
    return groups

def pick(rows):
    return sorted(rows,key=lambda x:(-x["p"],x["expert"]))[0]

def main():
    a=parse_args()
    years=[]
    aggregate=defaultdict(lambda: defaultdict(float))
    by_new_king=defaultdict(lambda: defaultdict(float))

    for spec in a.year_file:
        y_s,five_path,seven_path=spec.split(":",2)
        year=int(y_s)
        g5=load_groups(five_path)
        g7=load_groups(seven_path)
        if set(g5)!=set(g7):
            missing5=len(set(g7)-set(g5)); missing7=len(set(g5)-set(g7))
            raise SystemExit(f"group coverage mismatch y{year}: missing5={missing5} missing7={missing7}")

        per=defaultdict(lambda: defaultdict(float))
        perking=defaultdict(lambda: defaultdict(float))

        for key in sorted(g5):
            rid,role,n=key
            r5=g5[key]; r7=g7[key]
            p5=pick(r5); p7=pick(r7)
            old7=[r for r in r7 if r["expert"] in OLD_KINGS]
            new7=[r for r in r7 if r["expert"] in NEW_KINGS]
            if len({r["expert"] for r in old7})!=5 or len({r["expert"] for r in new7})!=2:
                raise SystemExit(f"expert coverage mismatch y{year} {key}")

            k=f"{role}|{n}"
            d=per[k]
            d["races"]+=1
            d["five_hits"]+=p5["hit"]
            d["seven_hits"]+=p7["hit"]
            changed=int(p5["expert"]!=p7["expert"])
            d["changed"]+=changed

            new_choice=int(p7["expert"] in NEW_KINGS)
            d["new_choice"]+=new_choice
            if new_choice:
                d["new_choice_hits"]+=p7["hit"]
                pk=perking[p7["expert"]]
                pk["choices"]+=1
                pk["hits"]+=p7["hit"]

            rescue=int(p5["hit"]==0 and p7["hit"]==1)
            regression=int(p5["hit"]==1 and p7["hit"]==0)
            d["rescues"]+=rescue
            d["regressions"]+=regression
            if rescue and new_choice:
                d["rescues_by_new"]+=1
                perking[p7["expert"]]["rescues"]+=1
            if regression and new_choice:
                d["regressions_by_new"]+=1
                perking[p7["expert"]]["regressions"]+=1

            old_oracle=max(r["hit"] for r in old7)
            new_oracle=max(r["hit"] for r in new7)
            new_only=int(old_oracle==0 and new_oracle==1)
            d["new_only_oracle_races"]+=new_only
            if new_only and p7["hit"]==1 and new_choice:
                d["captured_new_only"]+=1
                perking[p7["expert"]]["captured_new_only"]+=1

        rows={}
        for k,d0 in sorted(per.items()):
            d=dict(d0); races=int(d["races"])
            role,n=k.split("|")
            row={
                "role":role,"top_n":int(n),"races":races,
                "five_hit_rate":d["five_hits"]/races,
                "seven_hit_rate":d["seven_hits"]/races,
                "router_delta_pp":(d["seven_hits"]-d["five_hits"])*100/races,
                "changed_count":int(d["changed"]),
                "changed_rate":d["changed"]/races,
                "new_choice_count":int(d["new_choice"]),
                "new_choice_rate":d["new_choice"]/races,
                "new_choice_hit_rate":(d["new_choice_hits"]/d["new_choice"]) if d["new_choice"] else None,
                "rescue_count":int(d["rescues"]),
                "regression_count":int(d["regressions"]),
                "net_rescue":int(d["rescues"]-d["regressions"]),
                "rescue_by_new_count":int(d["rescues_by_new"]),
                "regression_by_new_count":int(d["regressions_by_new"]),
                "new_only_oracle_races":int(d["new_only_oracle_races"]),
                "captured_new_only_count":int(d["captured_new_only"]),
                "new_only_capture_rate":(d["captured_new_only"]/d["new_only_oracle_races"]) if d["new_only_oracle_races"] else None,
            }
            rows[k]=row
            for m,v in d.items():
                aggregate[k][m]+=v

        years.append({
            "year":year,
            "rows":rows,
            "new_king_totals":{
                king:{
                    "choices":int(v["choices"]),
                    "hits":int(v["hits"]),
                    "hit_rate":(v["hits"]/v["choices"]) if v["choices"] else None,
                    "rescues":int(v["rescues"]),
                    "regressions":int(v["regressions"]),
                    "captured_new_only":int(v["captured_new_only"]),
                }
                for king,v in sorted(perking.items())
            }
        })
        for king,v in perking.items():
            for m,x in v.items():
                by_new_king[king][m]+=x

    agg_rows={}
    for k,d0 in sorted(aggregate.items()):
        d=dict(d0); races=int(d["races"])
        role,n=k.split("|")
        agg_rows[k]={
            "role":role,"top_n":int(n),"races":races,
            "five_hit_rate":d["five_hits"]/races,
            "seven_hit_rate":d["seven_hits"]/races,
            "router_delta_pp":(d["seven_hits"]-d["five_hits"])*100/races,
            "changed_count":int(d["changed"]),
            "changed_rate":d["changed"]/races,
            "new_choice_count":int(d["new_choice"]),
            "new_choice_rate":d["new_choice"]/races,
            "new_choice_hit_rate":(d["new_choice_hits"]/d["new_choice"]) if d["new_choice"] else None,
            "rescue_count":int(d["rescues"]),
            "regression_count":int(d["regressions"]),
            "net_rescue":int(d["rescues"]-d["regressions"]),
            "rescue_by_new_count":int(d["rescues_by_new"]),
            "regression_by_new_count":int(d["regressions_by_new"]),
            "new_only_oracle_races":int(d["new_only_oracle_races"]),
            "captured_new_only_count":int(d["captured_new_only"]),
            "new_only_capture_rate":(d["captured_new_only"]/d["new_only_oracle_races"]) if d["new_only_oracle_races"] else None,
        }

    out={
        "contract":"L15_7KING_RESCUE_DIAGNOSTIC_V1",
        "years":years,
        "aggregate":agg_rows,
        "new_king_totals":{
            king:{
                "choices":int(v["choices"]),
                "hits":int(v["hits"]),
                "hit_rate":(v["hits"]/v["choices"]) if v["choices"] else None,
                "rescues":int(v["rescues"]),
                "regressions":int(v["regressions"]),
                "net_rescue":int(v["rescues"]-v["regressions"]),
                "captured_new_only":int(v["captured_new_only"]),
            }
            for king,v in sorted(by_new_king.items())
        },
        "notes":[
            "No odds used.",
            "2026 not read.",
            "Rescue = 5K router miss and 7K router hit.",
            "Regression = 5K router hit and 7K router miss.",
            "New-only oracle race = all five old kings miss while at least one new king hits.",
        ],
    }
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L15_7KING_RESCUE_DIAGNOSTIC_V1_OK")
    for k,v in agg_rows.items():
        print(k,
              "delta_pp",round(v["router_delta_pp"],3),
              "new_choice",round(v["new_choice_rate"]*100,2),
              "rescue",v["rescue_count"],
              "regression",v["regression_count"],
              "new_only",v["new_only_oracle_races"],
              "captured",v["captured_new_only_count"])
    print("NEW_KING_TOTALS",json.dumps(out["new_king_totals"],ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
