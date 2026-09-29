#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import pandas as pd

YEARS=(2023,2024,2025)

def parse_args():
    p=argparse.ArgumentParser(description="Diagnose race-level differences between fixed alpha=0 and alpha=1 K2 Top2 policies.")
    p.add_argument("--selected-races",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def write_csv(path,rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        p.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(p,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def parse_ids(v):
    return tuple(x for x in str(v or "").split("|") if x)

def pct(n,d):
    return 100.0*n/d if d else None

def contribution_shares(deltas):
    pos=sorted([float(x) for x in deltas if float(x)>0],reverse=True)
    total=sum(pos)
    return {
        "positive_delta_total_yen":total,
        "top1_positive_delta_share_pct":pct(sum(pos[:1]),total),
        "top3_positive_delta_share_pct":pct(sum(pos[:3]),total),
        "top5_positive_delta_share_pct":pct(sum(pos[:5]),total),
        "delta_without_top1_positive_yen":sum(deltas)-(pos[0] if pos else 0.0),
    }

def summarize(part,label):
    n=len(part)
    changed=part[part["selection_changed"]==1]
    same=part[part["selection_changed"]==0]
    deltas=part["delta_profit_yen"].astype(float).tolist()
    cs=contribution_shares(deltas)
    return {
        "scope":label,
        "races":n,
        "changed_races":len(changed),
        "same_races":len(same),
        "changed_rate_pct":pct(len(changed),n),
        "a0_only_hit_races":int((part["hit_class"]=="A0_ONLY_HIT").sum()),
        "a1_only_hit_races":int((part["hit_class"]=="A1_ONLY_HIT").sum()),
        "both_hit_races":int((part["hit_class"]=="BOTH_HIT").sum()),
        "both_miss_races":int((part["hit_class"]=="BOTH_MISS").sum()),
        "a0_profit_yen":float(part["a0_profit_yen"].sum()),
        "a1_profit_yen":float(part["a1_profit_yen"].sum()),
        "delta_profit_yen":float(part["delta_profit_yen"].sum()),
        "a0_return_yen":float(part["a0_return_yen"].sum()),
        "a1_return_yen":float(part["a1_return_yen"].sum()),
        "positive_delta_races":int((part["delta_profit_yen"]>0).sum()),
        "negative_delta_races":int((part["delta_profit_yen"]<0).sum()),
        "zero_delta_races":int((part["delta_profit_yen"]==0).sum()),
        **cs,
    }

def main():
    a=parse_args()
    src=Path(a.selected_races)
    df=pd.read_csv(src,dtype={"race_id":str})
    needed={"FIXED_A0","FIXED_A1"}
    got=set(df["policy"].astype(str))
    if not needed.issubset(got):
        raise SystemExit(f"missing policies needed={sorted(needed)} got={sorted(got)}")

    a0=df[df["policy"]=="FIXED_A0"].copy()
    a1=df[df["policy"]=="FIXED_A1"].copy()
    keys=["test_year","race_id"]
    for x in (a0,a1):
        if x.duplicated(keys).any():
            raise SystemExit("duplicate race-policy rows")

    cols=["test_year","race_id","race_date","selected_k2_ids","tickets","stake_yen","return_yen","profit_yen","hit"]
    m=a0[cols].merge(
        a1[cols],
        on=keys,
        how="outer",
        suffixes=("_a0","_a1"),
        validate="one_to_one",
        indicator=True,
    )
    if (m["_merge"]!="both").any():
        raise SystemExit("race universe mismatch between alpha0 and alpha1")

    rows=[]
    for r in m.itertuples(index=False):
        ids0=parse_ids(r.selected_k2_ids_a0)
        ids1=parse_ids(r.selected_k2_ids_a1)
        s0=set(ids0); s1=set(ids1)
        hit0=int(r.hit_a0); hit1=int(r.hit_a1)
        if hit0 and hit1: hc="BOTH_HIT"
        elif hit0: hc="A0_ONLY_HIT"
        elif hit1: hc="A1_ONLY_HIT"
        else: hc="BOTH_MISS"
        union=s0|s1
        inter=s0&s1
        rows.append({
            "test_year":int(r.test_year),
            "race_id":str(r.race_id),
            "race_date":str(r.race_date_a0),
            "a0_selected_k2":"|".join(ids0),
            "a1_selected_k2":"|".join(ids1),
            "selection_changed":int(ids0!=ids1),
            "shared_k2_count":len(inter),
            "union_k2_count":len(union),
            "jaccard":len(inter)/len(union) if union else 1.0,
            "a0_only_k2":"|".join(sorted(s0-s1)),
            "a1_only_k2":"|".join(sorted(s1-s0)),
            "a0_hit":hit0,
            "a1_hit":hit1,
            "hit_class":hc,
            "a0_return_yen":float(r.return_yen_a0),
            "a1_return_yen":float(r.return_yen_a1),
            "a0_profit_yen":float(r.profit_yen_a0),
            "a1_profit_yen":float(r.profit_yen_a1),
            "delta_return_yen":float(r.return_yen_a1)-float(r.return_yen_a0),
            "delta_profit_yen":float(r.profit_yen_a1)-float(r.profit_yen_a0),
        })

    outdf=pd.DataFrame(rows).sort_values(["test_year","race_date","race_id"]).reset_index(drop=True)

    summaries=[]
    for y in YEARS:
        part=outdf[outdf["test_year"]==y].copy()
        if part.empty:
            raise SystemExit(f"missing test year {y}")
        summaries.append(summarize(part,str(y)))
        summaries.append(summarize(part[part["selection_changed"]==1].copy(),f"{y}_CHANGED_ONLY"))
    summaries.append(summarize(outdf,"COMBINED"))
    summaries.append(summarize(outdf[outdf["selection_changed"]==1].copy(),"COMBINED_CHANGED_ONLY"))

    changed=outdf[outdf["selection_changed"]==1].copy()
    changed=changed.sort_values(["delta_profit_yen","test_year","race_id"],ascending=[False,True,True])

    contributors=changed[changed["delta_profit_yen"]!=0].copy()
    contributors["abs_delta_profit_yen"]=contributors["delta_profit_yen"].abs()
    contributors=contributors.sort_values(["abs_delta_profit_yen","test_year","race_id"],ascending=[False,True,True])

    combined=next(x for x in summaries if x["scope"]=="COMBINED")
    combined_changed=next(x for x in summaries if x["scope"]=="COMBINED_CHANGED_ONLY")
    by_year={str(y):next(x for x in summaries if x["scope"]==str(y)) for y in YEARS}
    alpha1_only_years=[y for y in YEARS if by_year[str(y)]["a1_only_hit_races"]>0]
    alpha0_only_years=[y for y in YEARS if by_year[str(y)]["a0_only_hit_races"]>0]

    summary={
        "contract":"L2_DANGER_K2_ALPHA_DIFF_V1",
        "source_run":36543847643,
        "comparison":"FIXED_A0 vs FIXED_A1, same fixed BASE seats and K2 Top2 count.",
        "purpose":"Determine whether alpha=1 advantage is repeated across years or dominated by a few jackpot races.",
        "summary_rows":summaries,
        "alpha1_only_hit_years":alpha1_only_years,
        "alpha0_only_hit_years":alpha0_only_years,
        "combined_delta_profit_yen":combined["delta_profit_yen"],
        "combined_changed_only_delta_profit_yen":combined_changed["delta_profit_yen"],
        "production_promotion":False,
    }

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"summary.csv",summaries)
    write_csv(out/"changed-races.csv",changed.to_dict("records"))
    write_csv(out/"delta-contributors.csv",contributors.to_dict("records"))
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Danger K2 Alpha Difference V1\n\n"
        "Pure post-hoc diagnostic of the already sealed 2023-2025 test outputs from run 36543847643. "
        "No new model, no new ticket shape, no data refetch, and no tuning. "
        "It isolates races where fixed alpha=0 and fixed alpha=1 selected different K2 Top2 horses, classifies exclusive hits, "
        "and measures how concentrated alpha=1's incremental profit is by race and year.\n",
        encoding="utf-8",
    )
    print("L2_DANGER_K2_ALPHA_DIFF_V1_READY")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
