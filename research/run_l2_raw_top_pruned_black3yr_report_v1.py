#!/usr/bin/env python3
"""Pure reporting: for each 20 ORIGINAL per-ticket reduction policies, select
only those of 3040 *template x disjoint odds band* with 2023-25 TOTAL profit >0;
order all selected lines by 3-year TOTAL ROI descending.

Methodology: retain original walk-forward convention exactly: 2023 is
TRAINING/BASE unpruned for every method, 2024 and 2025 are selected using prior
years only. NEVER describe this as 'each method reduced all 3 years'.
The line totals overlap in physical tickets. Black-only portfolio line-sum ROI
is descriptive/in-sample, NOT de-duplicated bettable portfolio or OOS success.
No historical source download, only the two ALREADY-MATERIALIZED result tables.
"""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

YEARS=("2023","2024","2025")
POLICY_MODES=("PROB","MARKET","LEARNED","HYBRID")
KEEP=(90,75,50,25,10)
POLICIES=("BASE",)+tuple(mode+"_"+str(k) for mode in POLICY_MODES for k in KEEP)
assert len(POLICIES)==21
KEYS=("partner_order","other_horses_k","family","odds_band")
EXPECTED_LINES=3040

def read_csv(path):
    op=gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",encoding="utf-8-sig",newline="") as f:
        yield from csv.DictReader(f)

def key(r):
    return (r["partner_order"],int(r["other_horses_k"]),r["family"],r["odds_band"])

def vals(r,prefix):
    return (int(r[prefix+"tickets"]),int(r[prefix+"hits"]),int(r[prefix+"payout_yen"]))

def baseline_vals(r):
    return (int(r["tickets"]),int(r["hit_tickets"]),int(r["payout_yen"]))

def pct(n,d):
    return round(100*n/d,6) if d else None

def explain(n,h,p):
    assert n>=0 and 0<=h<=n and p>=0,(n,h,p)
    return {"tickets":n,"hit_tickets":h,
            "hit_rate_pct":pct(h,n),
            "stake_yen":n*100,"payout_yen":p,
            "profit_yen":p-n*100,
            "roi_pct":pct(p,n*100)}

def load_original(path):
    data={}
    for r in read_csv(path):
        ky=(r["period"],*key(r))
        if ky in data: raise ValueError("duplicate original raw band "+repr(ky))
        data[ky]=baseline_vals(r)
    assert len(data)==4*EXPECTED_LINES,("baseline row count",len(data))
    for y in YEARS:
        assert sum(yr==y for yr,*_ in data)==EXPECTED_LINES
    return data

def load_pruned(path,raw):
    output={}
    count=Counter()
    for r in read_csv(path):
        yr=r["period"]
        policy=r["policy"]
        assert yr in ("2024","2025") and policy in POLICIES,(yr,policy)
        ky=(policy,yr,*key(r))
        if ky in output:raise ValueError("duplicate ticket policy key "+repr(ky))
        before=(int(r["before_tickets"]),int(r["before_hits"]),
                int(r["before_payout_yen"]))
        assert before==raw[(yr,*key(r))],("wrong phase1 correspondence",ky,before,raw[(yr,*key(r))])
        after=vals(r,"after_")
        assert after[0]<=before[0] and after[1]<=before[1] and after[2]<=before[2],("not subset",ky)
        if policy=="BASE":
            assert after==before,("BASE changed",ky)
        output[ky]=after
        count[(policy,yr)]+=1
    assert len(output)==2*len(POLICIES)*EXPECTED_LINES,("pruned grid size",len(output))
    assert all(count[(p,y)]==EXPECTED_LINES for p in POLICIES for y in ("2024","2025"))
    return output

def write(path,rows):
    assert rows and all(list(x)==list(rows[0]) for x in rows)
    if str(path).endswith(".gz"):
        op=gzip.open(path,"wt",encoding="utf-8",newline="")
    else:
        op=open(path,"w",encoding="utf-8",newline="")
    with op as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def calculate(raw,pruned):
    # Exact lines as frozen phase 1. 2023 must be BASE (prior-year-only),
    # never invent a 2022 model or use the same-year 2023 outcomes for training.
    template_bands=sorted(k[1:] for k in raw if k[0]=="2023")
    assert len(template_bands)==EXPECTED_LINES

    pooled_orig_black=0
    for id_ in template_bands:
        x=tuple(sum(raw[(y,*id_)][i] for y in YEARS) for i in range(3))
        assert x==raw[("2023-2025",*id_)],("3yr base reconstruction",id_)
        pooled_orig_black+=int(x[0]>0 and x[2]>100*x[0])
    assert pooled_orig_black==241,("phase1 black lines drift",pooled_orig_black)

    all_black=[]
    summary=[]
    for policy in POLICIES:
        profitable=[]
        zeros=0
        for id_ in template_bands:
            yr={"2023":raw[("2023",*id_)],
                "2024":pruned[(policy,"2024",*id_)],
                "2025":pruned[(policy,"2025",*id_)]}
            totals=tuple(sum(yr[y][j] for y in YEARS) for j in range(3))
            n,h,p=totals
            if not n: zeros+=1
            if not n or p<=100*n: continue
            orig=tuple(sum(raw[(y,*id_)][j] for y in YEARS) for j in range(3))
            d=explain(n,h,p)
            order,k,family,band=id_
            profitable.append({
                "method":policy,"partner_order":order,"other_horses_k":k,
                "family":family,"odds_band":band,**d,
                "unpruned_3year_tickets":orig[0],
                "unpruned_3year_hits":orig[1],
                "unpruned_3year_roi_pct":pct(orig[2],orig[0]*100),
                "year2023_tickets_BASE":yr["2023"][0],
                "year2023_hits_BASE":yr["2023"][1],
                "year2023_roi_BASE":pct(yr["2023"][2],yr["2023"][0]*100),
                "year2024_tickets_PRUNED":yr["2024"][0],
                "year2024_hits_PRUNED":yr["2024"][1],
                "year2024_roi_PRUNED":pct(yr["2024"][2],yr["2024"][0]*100),
                "year2025_tickets_PRUNED":yr["2025"][0],
                "year2025_hits_PRUNED":yr["2025"][1],
                "year2025_roi_PRUNED":pct(yr["2025"][2],yr["2025"][0]*100),
                "methodological_caveat":"2023 BASE / 2024-25 OOS prune; 3yr black labels post hoc; overlapping physical tickets"
            })
        profitable.sort(key=lambda r:(-r["roi_pct"],-r["tickets"],r["family"],
                                      r["partner_order"],r["other_horses_k"],r["odds_band"]))
        for rank,r in enumerate(profitable,1):
            # ensure same fieldnames in each row, rank at end
            r["roi_rank_within_method"]=rank
        all_black.extend(profitable)

        N=sum(x["tickets"] for x in profitable)
        H=sum(x["hit_tickets"] for x in profitable)
        P=sum(x["payout_yen"] for x in profitable)
        common=explain(N,H,P)
        summary.append({
            "method":policy,
            "three_year_black_lines":len(profitable),
            "three_year_red_or_even_lines":EXPECTED_LINES-zeros-len(profitable),
            "three_year_zero_ticket_lines":zeros,
            "black_only_line_sum_tickets_DUPLICATES":N,
            "black_only_line_sum_hits_DUPLICATES":H,
            "black_only_line_sum_hit_rate_pct_DUPLICATES":common["hit_rate_pct"],
            "black_only_line_sum_stake_yen_DUPLICATES":common["stake_yen"],
            "black_only_line_sum_payout_yen_DUPLICATES":common["payout_yen"],
            "black_only_line_sum_profit_yen_DUPLICATES":common["profit_yen"],
            "black_only_line_sum_roi_pct_DUPLICATES":common["roi_pct"],
            "method_year2023":"UNPRUNED BASE",
            "method_year2024":"OOS prior-year prune" if policy!="BASE" else "BASE",
            "method_year2025":"OOS prior-year prune" if policy!="BASE" else "BASE",
            "profit_selection":"pooled 2023-2025 positive only (descriptive/in-sample)",
        })
        print("BLACK_3YR_POLICY",policy,"LINES",len(profitable),
              "SUM_TICKETS_OVERLAP",N,"ROI_LINE_SUM",common["roi_pct"],flush=True)
    summary.sort(key=lambda r:(-(r["black_only_line_sum_roi_pct_DUPLICATES"] or -1),
                               -r["three_year_black_lines"],r["method"]))
    for rank,r in enumerate(summary,1):r["black_only_roi_rank"]=rank
    assert next(r for r in summary if r["method"]=="BASE")["three_year_black_lines"]==241
    assert len(summary)==21
    assert all(r["profit_yen"]>0 for r in all_black)
    return summary,all_black

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline",required=True)
    ap.add_argument("--pruned",required=True)
    ap.add_argument("--out-dir",required=True)
    a=ap.parse_args()
    root=Path(a.out_dir)
    root.mkdir(parents=True,exist_ok=True)
    old=load_original(Path(a.baseline))
    policy=load_pruned(Path(a.pruned),old)
    summary,black=calculate(old,policy)
    write(root/"methods_ranked_by_3year_black_only_line_sum_roi.csv",summary)
    write(root/"all_methods_3year_black_lines_ranked_within_method.csv.gz",black)
    n={r["method"]:r["three_year_black_lines"] for r in summary}
    assert sum(n.values())==len(black)
    meta={
        "contract":"RAW_TOP1_PRUNED_BLACK3YR_BY_METHOD_V1",
        "base_validation":{"phase1_all_2023_2025_black":241,
                           "phase1_all_lines":3040,
                           "phase2_methods":20,"total_policies_including_BASE":21},
        "definition":"each method: 2023 BASE unpruned + 2024 OOS pruned + 2025 OOS pruned; 3yr pooled profit>0",
        "sort":"method black-only combined line-sum ROI descending; within method individual black lines by ROI descending",
        "black_profitability_is_posthoc_selected_from_same_eval_years":True,
        "portfolio_physical_dedup":False,
        "do_not_claim_profitable_live_portfolio":True,
        "overall_line_sum_double_counts_physical_ticket_across_templates":True,
        "no_2026_data":True,"no_L21_change":True,
        "method_ranks":summary
    }
    (root/"summary.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("THREE_YEAR_BLACK_ONLY_RANKING_COMPLETE",json.dumps({
        "base_black_count":241,"all_profitable_method_rows":len(black),
        "highest_ranked_methods":[{"method":r["method"],"roi":r["black_only_line_sum_roi_pct_DUPLICATES"],"lines":r["three_year_black_lines"],
           "tickets":r["black_only_line_sum_tickets_DUPLICATES"]} for r in summary[:5]]},
        ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
