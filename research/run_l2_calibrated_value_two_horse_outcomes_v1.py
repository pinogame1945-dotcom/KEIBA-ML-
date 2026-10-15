#!/usr/bin/env python3
"""Report frozen market-adjusted AI Top1 AND frozen value-AI Top1 joint outcomes.

Read-only; uses precomputed 2023-25 OOS racewise snapshot. 2026 remains sealed.
Crucial: when both models select the same horse, 'both model Top3' means
two predictors are right, NOT two distinct horses arrived in top 3.
Produces both versions separately.
"""
import csv
import gzip
import json
import argparse
from pathlib import Path

RACES_PER_YEAR=3456
YEARS=("2023","2024","2025")
SELECTORS=(
 ("AI_PROB_TOP","VALUE_EV","market_adjusted_AI_vs_expected_value_AI"),
 ("MARKET_TOP","VALUE_EV","market_favorite_vs_expected_value_AI_REFERENCE")
)

def count(rows,a,b,year,scope):
    if year!="2023-2025":rows=[r for r in rows if str(r["year"])==year]
    if scope=="SAME":rows=[r for r in rows if r[a+"_horse"]==r[b+"_horse"]]
    if scope=="DIFFERENT":rows=[r for r in rows if r[a+"_horse"]!=r[b+"_horse"]]
    n=len(rows)
    stats=dict(year=year,scope=scope,selector_a=a,selector_b=b,
       races=n,
       one_of_them_wins=0,
       model_a_wins=0,
       model_b_wins=0,
       both_model_picks_win=0,
       neither_wins=0,
       both_model_picks_top3=0,
       two_distinct_horses_top3=0,
       either_top3=0,
       neither_top3=0,
       model_a_only_top3=0,
       model_b_only_top3=0,
       model_a_top3=0,model_b_top3=0,
       same_horse_selection=0,
       same_selected_horse_top3=0,
       same_selected_horse_win=0,
    )
    for r in rows:
        sw=r[a+"_horse"]==r[b+"_horse"]
        wa=int(r[a+"_win"]);wb=int(r[b+"_win"])
        ta=int(r[a+"_top3"]);tb=int(r[b+"_top3"])
        assert wa in (0,1) and wb in (0,1) and ta in (0,1) and tb in (0,1)
        assert wa<=ta and wb<=tb
        if sw:assert wa==wb and ta==tb
        stats["same_horse_selection"]+=int(sw)
        stats["same_selected_horse_top3"]+=int(sw and ta)
        stats["same_selected_horse_win"]+=int(sw and wa)
        stats["one_of_them_wins"]+=int(bool(wa or wb))
        stats["model_a_wins"]+=wa
        stats["model_b_wins"]+=wb
        stats["both_model_picks_win"]+=int(bool(wa and wb))
        stats["neither_wins"]+=int(not(wa or wb))
        stats["model_a_top3"]+=ta;stats["model_b_top3"]+=tb
        stats["either_top3"]+=int(bool(ta or tb))
        stats["neither_top3"]+=int(not(ta or tb))
        stats["both_model_picks_top3"]+=int(bool(ta and tb))
        stats["two_distinct_horses_top3"]+=int(bool((not sw) and ta and tb))
        stats["model_a_only_top3"]+=int(bool(ta and not tb))
        stats["model_b_only_top3"]+=int(bool(tb and not ta))
    assert stats["either_top3"]+stats["neither_top3"]==n
    assert stats["one_of_them_wins"]+stats["neither_wins"]==n
    assert (stats["two_distinct_horses_top3"]+
      stats["same_selected_horse_top3"]==stats["both_model_picks_top3"])
    for col in ["one_of_them_wins","both_model_picks_win","either_top3",
                "both_model_picks_top3","two_distinct_horses_top3",
                "neither_top3","model_a_wins","model_b_wins",
                "same_horse_selection","same_selected_horse_top3"]:
        stats[col+"_pct"]=round(stats[col]*100/n,6) if n else None
    return stats

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--infile",required=True)
    parser.add_argument("--out-dir",required=True)
    a=parser.parse_args()
    with gzip.open(a.infile,"rt",newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==RACES_PER_YEAR*len(YEARS),len(rows)
    assert {r["year"] for r in rows}==set(YEARS)
    assert all(sum(r["year"]==year for r in rows)==RACES_PER_YEAR for year in YEARS)
    assert len({r["race_id"] for r in rows})==len(rows)
    results=[
        count(rows,x,y,year,scope)
        for x,y,_ in SELECTORS
        for year in ("2023-2025",)+YEARS
        for scope in ("ALL","SAME","DIFFERENT")
    ]
    ref=[r for r in results
         if r["selector_a"]=="MARKET_TOP" and
            r["year"]=="2023-2025" and r["scope"]=="ALL"][0]
    assert ref["same_horse_selection"]==2340,ref
    assert ref["either_top3"]==7940,ref
    assert ref["both_model_picks_top3"]==3344,ref
    # Fixed previously audited reference: market favorite + value EV
    assert ref["one_of_them_wins"]==4417,ref
    o=Path(a.out_dir)
    o.mkdir(parents=True,exist_ok=True)
    with open(o/"all-pairs-year-scope-outcome.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]))
        w.writeheader();w.writerows(results)
    payload={
      "contract":"AI_PROB_TOP_VALUE_EV_PAIRWISE_WIN_TOP3_V1",
      "source_file":a.infile,
      "years":list(YEARS),
      "races":len(rows),
      "market_adjusted_AI_definition":"AI_PROB_TOP: max P__ANCHOR_FULL, calibrated market-anchored win probability",
      "expected_value_AI_definition":"VALUE_EV: max P__ANCHOR_FULL * final_win_odds, frozen ANCHOR_FULL",
      "same_horse_logic":"two_model_predictions_right != two_distinct_horses_top3",
      "pairwise_by_year":[r for r in results if r["selector_a"]=="AI_PROB_TOP" and r["scope"]=="ALL"],
      "market_favorite_reference":ref,
      "not_new_oos":"Study re-reads already inspected 2023-2025 stored racewise predictions and outcomes.",
      "price_warning":"Frozen expected-value selection uses FINAL win odds: not real-time executable",
      "2026_sealed":True,
      "new_training":False,
      "paid_resources":False
    }
    (o/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("CALIBRATED_AI_AND_VALUE_EV_PAIRWISE_REPORT",flush=True)
    for row in payload["pairwise_by_year"]:
        print(json.dumps(row,ensure_ascii=False),flush=True)
    print("ALL_RACE_PAIRWISE_OUTCOME_COMPLETE",flush=True)

if __name__=="__main__":main()
