#!/usr/bin/env python3
import argparse,gzip,json,math
from collections import defaultdict
from pathlib import Path

RC_EXPERT="jockey_trainer_condition_plus_raceclass"

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
            groups[key].append((
                str(r["expert_name"]),
                float(r["predicted_role_hit_probability"]),
                int(bool(r["role_hit"])),
            ))
    return groups

def pick(rows):
    return sorted(rows,key=lambda x:(-x[1],x[0]))[0]

def derive_year(five_path,seven_path):
    g5=load_groups(five_path)
    g7=load_groups(seven_path)
    if set(g5)!=set(g7):
        raise SystemExit(f"group coverage mismatch: five={len(g5)} seven={len(g7)}")
    out=[]
    for key in sorted(g5):
        rid,role,top_n=key
        five_rows=g5[key]; seven_rows=g7[key]
        base=pick(five_rows)
        flat7=pick(seven_rows)
        seven_by_expert={x[0]:x for x in seven_rows}
        if base[0] not in seven_by_expert:
            raise SystemExit(f"5K choice missing from 7K rows: {key} expert={base[0]}")
        if RC_EXPERT not in seven_by_expert:
            raise SystemExit(f"RC expert missing from 7K rows: {key}")
        base7=seven_by_expert[base[0]]
        rc=seven_by_expert[RC_EXPERT]
        five_sorted=sorted(five_rows,key=lambda x:(-x[1],x[0]))
        base_margin=(five_sorted[0][1]-five_sorted[1][1]) if len(five_sorted)>1 else math.nan
        out.append({
            "race_id":rid,
            "role":role,
            "top_n":top_n,
            "base_expert":base[0],
            "base_hit":base[2],
            "base_p5":base[1],
            "base_margin5":base_margin,
            "base_p7":base7[1],
            "rc_hit":rc[2],
            "rc_p7":rc[1],
            "gate_score":rc[1]-base7[1],
            "flat7_expert":flat7[0],
            "flat7_hit":flat7[2],
        })
    return out

def learn_threshold(rows):
    # Default is no override. Lower threshold gradually adds rows with highest RC advantage.
    ranked=sorted(rows,key=lambda r:r["gate_score"],reverse=True)
    best_gain=0
    best_threshold=None
    best_overrides=0
    cumulative_gain=0
    i=0
    while i<len(ranked):
        score=ranked[i]["gate_score"]
        j=i
        while j<len(ranked) and ranked[j]["gate_score"]==score:
            cumulative_gain += ranked[j]["rc_hit"]-ranked[j]["base_hit"]
            j+=1
        overrides=j
        if cumulative_gain>best_gain or (cumulative_gain==best_gain and cumulative_gain>0 and overrides<best_overrides):
            best_gain=cumulative_gain
            best_threshold=score
            best_overrides=overrides
        i=j
    return {
        "enabled":best_threshold is not None and best_gain>0,
        "threshold":best_threshold,
        "train_net_gain":int(best_gain),
        "train_overrides":int(best_overrides),
        "train_rows":len(rows),
    }

def evaluate(rows,policy):
    races=len(rows)
    base_hits=sum(r["base_hit"] for r in rows)
    flat7_hits=sum(r["flat7_hit"] for r in rows)
    rescues_possible=sum(1 for r in rows if r["base_hit"]==0 and r["rc_hit"]==1)
    regressions_possible=sum(1 for r in rows if r["base_hit"]==1 and r["rc_hit"]==0)
    gated_hits=0; overrides=0; rescues=0; regressions=0; harmless=0
    for r in rows:
        override=bool(policy["enabled"] and r["gate_score"]>=policy["threshold"])
        if override:
            overrides+=1
            gated_hits+=r["rc_hit"]
            if r["base_hit"]==0 and r["rc_hit"]==1:
                rescues+=1
            elif r["base_hit"]==1 and r["rc_hit"]==0:
                regressions+=1
            else:
                harmless+=1
        else:
            gated_hits+=r["base_hit"]
    return {
        "races":races,
        "base5_hits":base_hits,
        "base5_hit_rate":base_hits/races if races else None,
        "flat7_hits":flat7_hits,
        "flat7_hit_rate":flat7_hits/races if races else None,
        "gate6_hits":gated_hits,
        "gate6_hit_rate":gated_hits/races if races else None,
        "gate6_vs_base_pp":(gated_hits-base_hits)*100/races if races else None,
        "gate6_vs_flat7_pp":(gated_hits-flat7_hits)*100/races if races else None,
        "override_count":overrides,
        "override_rate":overrides/races if races else None,
        "rescue_count":rescues,
        "regression_count":regressions,
        "net_rescue":rescues-regressions,
        "harmless_override_count":harmless,
        "rc_rescues_possible":rescues_possible,
        "rc_regressions_possible":regressions_possible,
        "rescue_capture_rate":rescues/rescues_possible if rescues_possible else None,
        "perfect_rc_gate_hit_rate":(base_hits+rescues_possible)/races if races else None,
        "perfect_rc_gate_delta_pp":rescues_possible*100/races if races else None,
    }

def main():
    a=parse_args()
    yearly={}
    for spec in a.year_file:
        y_s,five,seven=spec.split(":",2)
        yearly[int(y_s)]=derive_year(five,seven)

    required={2022,2023,2024,2025}
    if set(yearly)!=required:
        raise SystemExit(f"expected years {sorted(required)}, got {sorted(yearly)}")

    folds=[]
    aggregate=defaultdict(lambda:defaultdict(float))
    for test_year in (2023,2024,2025):
        train_years=list(range(2022,test_year))
        train_rows=[r for y in train_years for r in yearly[y]]
        test_rows=yearly[test_year]
        keys=sorted({(r["role"],r["top_n"]) for r in test_rows})
        cells={}
        for role,top_n in keys:
            tr=[r for r in train_rows if r["role"]==role and r["top_n"]==top_n]
            te=[r for r in test_rows if r["role"]==role and r["top_n"]==top_n]
            policy=learn_threshold(tr)
            metrics=evaluate(te,policy)
            cell={"role":role,"top_n":top_n,"train_years":train_years,"test_year":test_year,
                  "policy":policy,"metrics":metrics}
            cells[f"{role}|{top_n}"]=cell
            for k,v in metrics.items():
                if isinstance(v,(int,float)) and v is not None and k not in {
                    "base5_hit_rate","flat7_hit_rate","gate6_hit_rate","gate6_vs_base_pp",
                    "gate6_vs_flat7_pp","override_rate","rescue_capture_rate",
                    "perfect_rc_gate_hit_rate","perfect_rc_gate_delta_pp"
                }:
                    aggregate[f"{role}|{top_n}"][k]+=v
        folds.append({"test_year":test_year,"train_years":train_years,"cells":cells})

    agg={}
    for key,d0 in sorted(aggregate.items()):
        role,top_n=key.split("|")
        d=defaultdict(float,d0); races=int(d["races"])
        agg[key]={
            "role":role,
            "top_n":int(top_n),
            "races":races,
            "base5_hit_rate":d["base5_hits"]/races,
            "flat7_hit_rate":d["flat7_hits"]/races,
            "gate6_hit_rate":d["gate6_hits"]/races,
            "gate6_vs_base_pp":(d["gate6_hits"]-d["base5_hits"])*100/races,
            "gate6_vs_flat7_pp":(d["gate6_hits"]-d["flat7_hits"])*100/races,
            "override_count":int(d["override_count"]),
            "override_rate":d["override_count"]/races,
            "rescue_count":int(d["rescue_count"]),
            "regression_count":int(d["regression_count"]),
            "net_rescue":int(d["net_rescue"]),
            "harmless_override_count":int(d["harmless_override_count"]),
            "rc_rescues_possible":int(d["rc_rescues_possible"]),
            "rc_regressions_possible":int(d["rc_regressions_possible"]),
            "rescue_capture_rate":d["rescue_count"]/d["rc_rescues_possible"] if d["rc_rescues_possible"] else None,
            "perfect_rc_gate_hit_rate":(d["base5_hits"]+d["rc_rescues_possible"])/races,
            "perfect_rc_gate_delta_pp":d["rc_rescues_possible"]*100/races,
        }

    out={
        "contract":"L15_RC_RESCUE_GATE_V1",
        "rc_expert":RC_EXPERT,
        "method":{
            "base":"5K router choice",
            "gate_score":"7K-router P(RC) minus 7K-router P(the 5K-selected expert)",
            "threshold_training":"past unknown-year fold predictions only; choose threshold maximizing train hit-count gain; no positive gain => disabled",
            "walk_forward":[
                {"train":[2022],"test":2023},
                {"train":[2022,2023],"test":2024},
                {"train":[2022,2023,2024],"test":2025},
            ],
            "odds_used":False,
            "locked_years":[2026],
        },
        "folds":folds,
        "aggregate":agg,
        "notes":[
            "This is a proof-of-concept rescue gate using saved fold predictions; it is not the final production gate.",
            "The new pedigree expert is not eligible to override in this experiment.",
            "2026 is never read.",
        ],
    }
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    print("L15_RC_RESCUE_GATE_V1_OK")
    for key,v in agg.items():
        print(key,
              "base",round(v["base5_hit_rate"]*100,3),
              "gate6",round(v["gate6_hit_rate"]*100,3),
              "flat7",round(v["flat7_hit_rate"]*100,3),
              "vs_base_pp",round(v["gate6_vs_base_pp"],3),
              "vs_flat7_pp",round(v["gate6_vs_flat7_pp"],3),
              "overrides",v["override_count"],
              "rescue",v["rescue_count"],
              "regression",v["regression_count"],
              "capture",round((v["rescue_capture_rate"] or 0)*100,2))
    for fold in folds:
        y=fold["test_year"]
        for key,cell in fold["cells"].items():
            pcy=cell["policy"]; m=cell["metrics"]
            print("FOLD",y,key,
                  "enabled",pcy["enabled"],
                  "threshold",None if pcy["threshold"] is None else round(pcy["threshold"],6),
                  "train_gain",pcy["train_net_gain"],
                  "test_delta_pp",round(m["gate6_vs_base_pp"],3),
                  "overrides",m["override_count"],
                  "rescue",m["rescue_count"],
                  "regression",m["regression_count"])

if __name__=="__main__":
    main()
