# !/usr/bin/env python3
"""Fixed MARKET_75 ticket pruning for VALUE_EV rank1, no stability filters."""

from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import pandas as pd
import run_l2_ev_rank2to6_pruning20_best_roi_v1 as ev
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,assert_ok

POLICY="MARKET_75"
SOURCE_METHOD="research-results/l2-raw-20fair-warmup-v1/run-38885047839/summary.json"

def read_original(path):
    out={}
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            scheme,order,k,family=row["買い方ID"].split("|")
            assert_ok(scheme=="EV","historical anchor must be original EV1")
            tid=(order,int(k),family,row["馬券オッズ帯"])
            assert_ok(tid in ev.LINE_KEYS and tid not in out,("bad old EV source",tid))
            out[tid]={}
            for y in YEARS:
                n=int(row[f"{y}_購入点数"])
                h=int(row[f"{y}_的中点数"])
                pay=int(row[f"{y}_払戻円"])
                assert_ok(int(row[f"{y}_投資円"])==100*n,"historical cost drift")
                out[tid][y]=(n,h,pay)
    assert_ok(len(out)==3040,"original 3040 EV source line drift")
    return out

def csv_save(path,rows):
    assert_ok(rows,"empty results")
    with Path(path).open("w",encoding="utf-8",newline="") as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def metrics(entry):
    n=int(entry["n"]);h=int(entry["hits"]);pay=int(entry["payout"])
    return dict(tickets=n,winning_tickets=h,stake_yen=n*100,
                return_yen=pay,profit_yen=pay-100*n,
                roi_pct=100*pay/(100*n) if n else None,
                bought_races=int(entry["races"]))

def main():
    p=argparse.ArgumentParser()
    for key in ("predictions","picks","racewise","original-bands",
                "raw-adoption-summary","rank2to6-summary","backfill-root","out-dir"):
        p.add_argument("--"+key,required=True)
    a=p.parse_args()
    raw=json.loads(Path(a.raw_adoption_summary).read_text(encoding="utf-8"))
    assert_ok(raw["best_method"]==POLICY,
              ("RAW source best policy unexpected",raw["best_method"]))
    assert_ok(raw["January_2023_excluded_from_all_methods_including_BASE"],
              "RAW market adoption used other evaluation period")
    archived=read_original(a.original_bands)
    data,picks=ev.ranklab.load_predictions(a.predictions,a.picks)
    races=pd.read_csv(a.racewise,compression="infer",
                      dtype={"race_id":str,"VALUE_EV_horse":str,"AI_RAW_TOP_horse":str})
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert_ok(len(races)==TOTAL_RACES and races.race_id.is_unique,"archived race drift")
    assert_ok(set(races.year)==set(YEARS),"2026 evaluation sealed")
    root=Path(a.backfill_root);out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    print("EV1_FIXED_MARKET75_NO_RESELECTION_START",flush=True)
    line_stats,thresholds,training,coverage=ev.ledger_pass(
        data,picks,races,root,1,archived)
    assert_ok(coverage["2023"]["training_only_january_races"]==312,
              "2023 warmup mismatch")
    all_selected,_=ev.black_lines(line_stats)
    chosen={p:(all_selected[p] if p in ("BASE",POLICY) else set())
            for p in ev.POLICIES}
    assert_ok(chosen[POLICY] and chosen["BASE"],"no positive historical lines")
    union=ev.physically_dedup_pass(data,picks,races,root,1,chosen,thresholds,training)
    rows=[]
    for policy in ("BASE",POLICY):
        pooled={"n":0,"hits":0,"payout":0,"races":0}
        for year in YEARS:
            stat=union[year,policy]
            for k in pooled:pooled[k]+=int(stat[k])
            rows.append(dict(rank=1,method=policy,period=str(year),
                             black_odds_lines=len(chosen[policy]),**metrics(stat)))
        rows.append(dict(rank=1,method=policy,period="2023-2025",
                         black_odds_lines=len(chosen[policy]),**metrics(pooled)))
    csv_save(out/"rank1_before_stability_by_year.csv",rows)
    selected=next(x for x in rows if x["method"]==POLICY and x["period"]=="2023-2025")
    old=json.loads(Path(a.rank2to6_summary).read_text(encoding="utf-8"))
    assert_ok(old["january_2023_excluded_from_all_policies"]
              and old["evaluation_start_2023"]=="2023-02-01",
              "other EV ranks used different test range")
    assert_ok(len(old["selected_rows"])==5,"other selected ranks not complete")
    six=[dict(ev_rank=1,adopted_method=POLICY,
              black_odds_lines=selected["black_odds_lines"],
              purchased_tickets=selected["tickets"],
              winning_tickets=selected["winning_tickets"],
              stake_yen=selected["stake_yen"],return_yen=selected["return_yen"],
              profit_yen=selected["profit_yen"],roi_pct=selected["roi_pct"])]
    six.extend(old["selected_rows"])
    assert_ok([x["ev_rank"] for x in six]==[1,2,3,4,5,6],"rank comparison drift")
    csv_save(out/"rank1to6_pruning_only_same_window.csv",six)
    result=dict(contract="EV1_FIXED_MARKET75_AND_EV2TO6_PRUNE_ONLY_V1",
        method_applied_to_ev1=POLICY,
        market75_adopted_on_RAW_rank1_not_proven_previously_on_EV1=True,
        method_transferred_to_EV1_without_policy_reselection=True,
        SOURCE_METHOD=SOURCE_METHOD,rank1_anchor="VALUE_EV_EQUALS_ANCHOR_FULL",
        black_line_filter="pooled 2023 Feb-Dec plus full 2024 and 2025 profit positive",
        stability_medium_strict_applied=False,
        exact_original_3040_line_baseline_verified=True,
        common_window_races=10056,
        rank1_by_year=rows,six_rank_comparison=six,
        combined_tickets_before_cross_rank_dedup=sum(x["purchased_tickets"] for x in six),
        per_rank_not_combined_roi=True,
        rank2to6_source_run=38891800300,
        three_year_black_is_same_period_post_hoc=True,
        historical_final_quotes_not_prebet=True,year_2026_sealed=True)
    (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("SIX_RANKS_PRUNING_ONLY_COMPLETE",json.dumps(six,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
