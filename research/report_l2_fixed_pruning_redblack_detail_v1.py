#!/usr/bin/env python3
"""Detailed strictly post-run red-to-black audit of ALL 3,040 fixed original lines.

Read existing successful run. No new selection / training / backfill / fitting.
Reports every originally loss-making (2024-25) line flipped to positive by pruning,
year-by-year stability, aggregate distribution, money, and deduped union caveat.
"""
import argparse,csv,gzip,json,collections
from pathlib import Path
POLS=("BASE","PROB_90","PROB_75","PROB_50","PROB_25","PROB_10",
      "MARKET_90","MARKET_75","MARKET_50","MARKET_25","MARKET_10",
      "LEARNED_90","LEARNED_75","LEARNED_50","LEARNED_25","LEARNED_10",
      "HYBRID_90","HYBRID_75","HYBRID_50","HYBRID_25","HYBRID_10")
KEEP=("PROB_10","PROB_25","PROB_50","MARKET_10","LEARNED_50","HYBRID_10","HYBRID_25","HYBRID_50")
def read(path):
    op=gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))
def write(path,rows):
    if not rows:return
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def key(row):
    return tuple(row[c] for c in ("order","k","family","odds_band","policy"))
def asf(row,k):return float(row[k])
def group_counts(rows,group):
    grouped=collections.defaultdict(list)
    for row in rows:grouped[row[group]].append(row)
    summary=[]
    for k,rr in grouped.items():
        both=[x for x in rr if x["both_years_positive"]]
        money=sum(asf(x,"after_profit_yen") for x in rr)
        summary.append({group:k,"flipped_lines":len(rr),
            "both_eval_years_profitable":len(both),
            "both_eval_years_profitable_min_100_tickets_each":sum(x["both_years_positive"] and x["tickets_2024"]>=100 and x["tickets_2025"]>=100 for x in rr),
            "sum_non_dedup_after_profit_yen":round(money),
            "sum_non_dedup_after_stake_yen":round(sum(asf(x,"after_stake_yen") for x in rr)),
            "sum_non_dedup_delta_yen":round(sum(asf(x,"profit_delta_yen") for x in rr))})
    return sorted(summary,key=lambda x:(-x["flipped_lines"],str(x[group])))
def main():
    a=argparse.ArgumentParser();a.add_argument("--source",required=True);a.add_argument("--out",required=True);x=a.parse_args()
    src=Path(x.source);out=Path(x.out);out.mkdir(parents=True,exist_ok=True)
    raw=read(src/"red_lines_changed_to_black_2024_25.csv.gz")
    raw=[r for r in raw if r.get("policy") in POLS and r.get("red_to_black")=="True"]
    lookup={key(r):r for r in raw}
    assert len(lookup)==len(raw)
    yearly={}
    with gzip.open(src/"all_3040_fixed_lines_before_after_2024_2025.csv.gz","rt",newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k=key(r)
            if k in lookup and r["year"] in ("2024","2025"):
                yearly[(r["year"],k)]=r
    assert len(yearly)==len(raw)*2,(len(yearly),len(raw))
    converted=[]
    for r in raw:
        p24=yearly["2024",key(r)];p25=yearly["2025",key(r)]
        item={k:r[k] for k in ("policy","order","k","family","odds_band",
              "before_races","before_tickets","before_hits","before_stake_yen",
              "before_return_yen","before_profit_yen","before_roi_pct",
              "after_races","after_tickets","after_hits","after_stake_yen",
              "after_return_yen","after_profit_yen","after_roi_pct",
              "saved_losing_tickets","lost_winning_tickets","saved_stake_yen",
              "lost_winnings_yen","profit_delta_yen")}
        for y,p in (("2024",p24),("2025",p25)):
            item["before_profit_"+y+"_yen"]=p["before_profit_yen"]
            item["after_profit_"+y+"_yen"]=p["after_profit_yen"]
            item["before_roi_"+y+"_pct"]=p["before_roi_pct"]
            item["after_roi_"+y+"_pct"]=p["after_roi_pct"]
            item["tickets_"+y]=int(p["after_tickets"])
            item["wins_"+y]=int(p["after_hits"])
        item["both_years_positive"]=(
            asf(p24,"after_profit_yen")>0 and asf(p25,"after_profit_yen")>0)
        item["positive_years"]=(int(asf(p24,"after_profit_yen")>0)+
                                 int(asf(p25,"after_profit_yen")>0))
        converted.append(item)
    write(out/"all_red_to_black_converted_lines_with_yearly_profit.csv",converted)
    audit=read(src/"red_to_black_preservation_audit_all_policies.csv")
    pa=[]
    for p in POLS:
        ar=[r for r in converted if r["policy"]==p]
        rr=next(r for r in audit if r["policy"]==p)
        assert len(ar)==int(rr["red_went_black"]),(p,len(ar),rr)
        st=sum(r["both_years_positive"] for r in ar)
        assert st==int(rr["red_went_black_both_individual_test_years"]),(p,st,rr)
        pa.append(dict(policy=p,red_to_black=len(ar),both_years_positive=st,
          both_years_positive_min_100_tickets_each=sum(x["both_years_positive"] and x["tickets_2024"]>=100 and x["tickets_2025"]>=100 for x in ar),
          original_3yr_black_to_red=int(rr["original_black_went_red"]),
          sum_profit_delta_non_dedup_yen=round(sum(asf(z,"profit_delta_yen") for z in ar)),
          sum_after_profit_non_dedup_yen=round(sum(asf(z,"after_profit_yen") for z in ar)),
          median_tickets_converted=sorted((int(z["after_tickets"]) for z in ar))[len(ar)//2] if ar else 0))
    write(out/"policy_red_to_black_summary.csv",pa)
    for p in KEEP:
        s=[r for r in converted if r["policy"]==p]
        write(out/("family_breakdown_"+p+".csv"),group_counts(s,"family"))
        write(out/("odds_band_breakdown_"+p+".csv"),group_counts(s,"odds_band"))
    good=[r for r in converted if r["both_years_positive"]]
    top=sorted(good,key=lambda z:asf(z,"after_profit_yen"),reverse=True)
    write(out/"top_100_both_years_positive_converted_lines.csv",top[:100])
    meaningful=[r for r in good if r["tickets_2024"]>=100 and r["tickets_2025"]>=100]
    write(out/"top_100_both_years_positive_min100tickets_each_year.csv",
          sorted(meaningful,key=lambda z:asf(z,"after_profit_yen"),reverse=True)[:100])
    union=read(src/"deduped_original_245_and_all_tickets_union_before_after.csv")
    union2=[r for r in union if r["year"] in ("2024","2025","2024-2025") and r["policy"] in POLS]
    write(out/"all_policies_deduplicated_union_revenue.csv",union2)
    outcome=dict(contract="READONLY_3040_FIXED_RED_TO_BLACK_DETAIL_V1",
        no_experiment_changed=True,source_run="38692491374",scope="ALL 3040 fixed strategies originally evaluated, black 245 is report-only posthoc partition",
        2024_2025_baseline_negative_to_positive_only=True,
        per_policy=pa,
        top_both_years_min100tickets_each=[{k:r[k] for k in ("policy","order","k","family","odds_band","before_tickets","before_roi_pct","before_profit_yen","after_tickets","after_roi_pct","after_profit_yen","after_profit_2024_yen","after_profit_2025_yen","tickets_2024","tickets_2025")} for r in sorted(meaningful,key=lambda z:asf(z,"after_profit_yen"),reverse=True)[:25]],
        family_selected={p:group_counts([r for r in converted if r["policy"]==p],"family") for p in KEEP},
        odds_selected={p:group_counts([r for r in converted if r["policy"]==p],"odds_band") for p in KEEP},
        years_selection="2023 training only, 2024 evaluated on 2023, 2025 evaluated on 2023-2024",
        does_not_forecast_black_bands=True,
        original_245_are_posthoc_for_reporting_only=True,
        no_net_profit_for_all_3040_union=True,
        not_independent_future_proof="2024/2025 previously inspected; odds are historical FINAL"
        )
    (out/"summary.json").write_text(json.dumps(outcome,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("RED_TO_BLACK_DETAILED_AUDIT_OK",len(converted),"both positive",len(good),
          "both min100",len(meaningful),flush=True)
    for z in pa:print("POLICY",json.dumps(z),flush=True)
    for z in outcome["top_both_years_min100tickets_each"][:15]:print("TOP_LINE",json.dumps(z),flush=True)
if __name__=="__main__":main()
