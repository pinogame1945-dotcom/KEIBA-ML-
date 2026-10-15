#!/usr/bin/env python3
"""Exact 100-yen ROI for EV-anchor agreeing with calibrated probability top 1.

PRE-RACE subgroup selection; normalized *actual* historical payout across
every 2023-25 race. Separate 'value and probability agree' from 'also 1st
market favorite'. No training, no paid runner, no 2026 data.
"""
import argparse, concurrent.futures, json
from collections import Counter,defaultdict
from pathlib import Path
import pandas as pd
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,assert_ok
from build_l2_market_helpers_v1 import load_day,horse_number_map
import run_l2_value_anchor_bet_frontier_v1 as prior

GROUPS=("ALL","EV_EQ_PROB","EV_EQ_PROB_EQ_MARKET","EV_EQ_PROB_NE_MARKET",
        "EV_NE_PROB","EV_EQ_RAW","EV_EQ_PROB_EQ_RAW")
SOURCES=("AI_PROB","MARKET")
PARTNERS=(2,3,5,7)
SINGLES=("WIN","PLACE")

def subset_markers(r):
    ev=r["VALUE_EV_horse"]
    prob=r["AI_PROB_TOP_horse"]
    market=r["MARKET_TOP_horse"]
    raw=r["AI_RAW_TOP_horse"]
    groups=["ALL"]
    if ev==prob:
        groups.append("EV_EQ_PROB")
        groups.append("EV_EQ_PROB_EQ_MARKET" if ev==market else "EV_EQ_PROB_NE_MARKET")
        if ev==raw:groups.append("EV_EQ_PROB_EQ_RAW")
    else:
        groups.append("EV_NE_PROB")
    if ev==raw:groups.append("EV_EQ_RAW")
    return groups

def year_calc(year, races, predictions, root):
    rr=races[races.year.eq(year)].copy()
    pred=predictions[predictions.year.eq(year)].copy()
    pred_groups=pred.groupby("_race_id",sort=False)
    assert_ok(len(rr)==RACES_YEAR,"race count drift "+str(year))
    dates=rr[["race_id","race_date"]].groupby("race_date")["race_id"].apply(list)
    totals=defaultdict(Counter)
    attrs=defaultdict(Counter)
    for day,ids in dates.items():
        packs=load_day(root/"data"/"daily"/f"{day}.jsonl.gz",set(ids))
        for rid in ids:
            row=rr.loc[rr.race_id.eq(rid)].iloc[0]
            pack=packs.get(rid)
            assert_ok(pack is not None,"missing pack "+rid)
            horse_no=horse_number_map(pack)
            axis_id=row["VALUE_EV_horse"]
            market_id=row["MARKET_TOP_horse"]
            assert_ok(axis_id in horse_no and market_id in horse_no,"horse map missing "+rid)
            g=pred_groups.get_group(rid)
            market_sort=g.sort_values(
                ["market_rank","final_win_odds","_horse_id"],
                ascending=[True,True,True],kind="mergesort")["_horse_id"].astype(str).tolist()
            prob_sort=g.sort_values(
                ["P__ANCHOR_FULL","market_rank","_horse_id"],
                ascending=[False,True,True],kind="mergesort")["_horse_id"].astype(str).tolist()
            assert_ok(prob_sort[0]==row["AI_PROB_TOP_horse"],"top probability drift "+rid)
            assert_ok(market_sort[0]==market_id,"market selection drift "+rid)
            paid=prior.payouts(pack,rid)
            groups=subset_markers(row)
            axis_no=horse_no[axis_id]
            mkt_no=horse_no[market_id]
            finish=row["VALUE_EV_finish"]
            bucket={1:"FIRST",2:"SECOND",3:"THIRD"}.get(
                 int(finish) if pd.notna(finish) else None,"FOURTH_PLUS")
            if pd.isna(finish):bucket="NO_FINISH"
            win_ev=paid.get(("WIN",(axis_no,)),0)
            place_ev=paid.get(("PLACE",(axis_no,)),0)
            win_market=paid.get(("WIN",(mkt_no,)),0)
            place_market=paid.get(("PLACE",(mkt_no,)),0)
            p=float(row["VALUE_EV_p"])
            o=float(row["VALUE_EV_odds"])
            group_tickets={}
            for source,ordering in (("AI_PROB",prob_sort),("MARKET",market_sort)):
                others=[horse_no[h] for h in ordering if h!=axis_id]
                for k in PARTNERS:
                    tickets=prior.pair_templates(axis_no,others[:k])
                    for fam,t in tickets.items():
                        group_tickets[(fam,source,k)]=(len(t),sum(paid.get(z,0) for z in t),
                            sum(paid.get(z,0)>0 for z in t))
            for sc in groups:
                at=attrs[sc]
                at["races"]+=1
                at["sum_ai_ev_milli"]+=round(p*o*1000)
                at["sum_odds_milli"]+=round(o*1000)
                at["sum_anchor_favorite"]+=int(axis_id==market_id)
                at["sum_anchor_top3"]+=int(bucket in ("FIRST","SECOND","THIRD"))
                for fam,got in (("WIN",win_ev),("PLACE",place_ev)):
                    prior.update(totals[(sc,"EV",fam,"SINGLE",0)],1,got,bucket)
                for fam,got in (("WIN",win_market),("PLACE",place_market)):
                    prior.update(totals[(sc,"MARKET",fam,"SINGLE",0)],1,got,bucket)
                for (fam,source,k),(count,payout,num_hits) in group_tickets.items():
                    c=totals[(sc,"EV",fam,source,k)]
                    prior.update(c,count,payout,bucket)
                    c["extra_hit_tickets"]+=max(0,num_hits-int(payout>0))
    return totals,attrs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--racewise",required=True)
    ap.add_argument("--backfill-root",required=True)
    ap.add_argument("--out-dir",required=True)
    a=ap.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    col=["year","_race_id","_horse_id","market_rank","final_win_odds","P__ANCHOR_FULL"]
    pred=pd.read_csv(a.predictions,usecols=col,compression="infer",
                     dtype={"_race_id":str,"_horse_id":str})
    pred["year"]=pd.to_numeric(pred["year"],errors="raise").astype(int)
    assert_ok(set(pred.year)==set(YEARS),"2026 sealed or year missing")
    races=pd.read_csv(a.racewise,compression="infer",dtype={"race_id":str})
    races["year"]=pd.to_numeric(races["year"],errors="raise").astype(int)
    assert_ok(set(races.year)==set(YEARS) and len(races)==TOTAL_RACES
              and races.race_id.is_unique,"racewise drift")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        chunks=list(pool.map(lambda y:year_calc(y,races,pred,Path(a.backfill_root)),YEARS))
    merged=defaultdict(Counter)
    m_attrs=defaultdict(Counter)
    for year,(counters,attrs) in zip(YEARS,chunks):
        for key,c in counters.items():
            merged[(str(year),*key)].update(c)
            merged[("2023-2025",*key)].update(c)
        for key,c in attrs.items():
            m_attrs[(str(year),key)].update(c)
            m_attrs[("2023-2025",key)].update(c)
    assert_ok(m_attrs[("2023-2025","ALL")]["races"]==TOTAL_RACES,"incomplete")
    assert_ok(m_attrs[("2023-2025","EV_EQ_PROB")]["races"]==2489,"overlap baseline drift")
    assert_ok(m_attrs[("2023-2025","EV_EQ_PROB_EQ_MARKET")]["races"]==2340,"market overlap drift")
    out_rows=[]
    for (year,sc,axis,fam,src,k),c in merged.items():
        n=c["races"]
        attrs=m_attrs[(year,sc)]
        assert_ok(n==attrs["races"],"template coverage incomplete")
        stake=c["stake_yen"]
        payout=c["return_yen"]
        out_rows.append({
            "year":year,"scope":sc,"anchor":axis,"family":fam,
            "partner_order":src,"other_horses_k":k,"races":n,
            "stake_yen":stake,"return_yen":round(payout,2),
            "profit_yen":round(payout-stake,2),
            "roi_pct":round(100*payout/stake,6) if stake else None,
            "avg_tickets_per_race":round(c["tickets"]/n,5),
            "hit_races":c["hit_races"],
            "hit_rate_pct":round(100*c["hit_races"]/n,6),
            "avg_selected_model_ev":round(attrs["sum_ai_ev_milli"]/(1000*n),6),
            "avg_selected_odds":round(attrs["sum_odds_milli"]/(1000*n),6),
            "anchor_market_favorite_pct":round(100*attrs["sum_anchor_favorite"]/n,6),
            "anchor_top3_pct":round(100*attrs["sum_anchor_top3"]/n,6),
        })
    df=pd.DataFrame(out_rows).sort_values(["year","scope","anchor","family","partner_order","other_horses_k"])
    df.to_csv(out/"conditional-exact-roi-all-82-templates.csv",index=False)
    relevant=df[(df.anchor.eq("EV"))&(
        df.family.isin(SINGLES)|
        (df.family.eq("TRIO")&df.partner_order.eq("AI_PROB")&df.other_horses_k.eq(2))|
        (df.family.eq("WIDE")&df.partner_order.eq("MARKET")&df.other_horses_k.eq(2))|
        (df.family.eq("EXACTA_FIRST")&df.partner_order.eq("MARKET")&df.other_horses_k.eq(2)))]
    relevant.to_csv(out/"conditional-headline-roi.csv",index=False)
    best=(df[df.anchor.eq("EV")].sort_values("roi_pct",ascending=False)
          .groupby(["year","scope"],sort=False).head(5))
    best.to_csv(out/"conditional-best-five-fixed-templates.csv",index=False)
    agree=(relevant[(relevant.year.eq("2023-2025"))&
                    (relevant.scope.isin(("EV_EQ_PROB","EV_EQ_PROB_EQ_MARKET",
                        "EV_EQ_PROB_NE_MARKET","EV_NE_PROB")))]
          .sort_values(["scope","family","partner_order"]))
    print("EXACT_AGREEMENT_ROI_START")
    print(agree.to_csv(index=False),flush=True)
    print("YEAR_SINGLE_AND_TRIO")
    print(relevant[(relevant.scope.eq("EV_EQ_PROB_EQ_MARKET"))&
                   relevant.family.isin(("WIN","PLACE","TRIO"))].to_csv(index=False),flush=True)
    summary={
        "contract":"EV_ANCHOR_AND_PROBABILITY_TOP1_AGREEMENT_ROI_V1",
        "year_scope":[2023,2024,2025],"races":TOTAL_RACES,
        "scopes":{key:int(val["races"]) for (yr,key),val in m_attrs.items() if yr=="2023-2025"},
        "condition_based_on_prerace_predictions_only":True,
        "warning":"FINAL win odds used for retrospective bet choice; cannot treat these figures as live executable ROI",
        "fixed_templates":{"EV_anchor_per_scope":82,"market_single_per_scope":2},
        "all_82_template_roi":True,"lookahead_template_selection_warning":"best five templates ranked after results; not OOS selection",
        "baseline_check":{"ev_all_win_roi":float(df[(df.year.eq("2023-2025"))&
              (df.scope.eq("ALL"))&(df.anchor.eq("EV"))&(df.family.eq("WIN"))].roi_pct.iloc[0]),
              "original_ev_all_trio_roi":float(df[(df.year.eq("2023-2025"))&
              (df.scope.eq("ALL"))&(df.anchor.eq("EV"))&(df.family.eq("TRIO"))&
              (df.partner_order.eq("AI_PROB"))&(df.other_horses_k.eq(2))].roi_pct.iloc[0])},
        "source":"BACKFILL pinned 0d4af461b77f0060dd9c789dd2da646026fada47",
        "2026_sealed":True,
    }
    assert_ok(abs(summary["baseline_check"]["ev_all_win_roi"]-86.898148)<1e-5,
              "original EV WIN ROI not reproduced")
    assert_ok(abs(summary["baseline_check"]["original_ev_all_trio_roi"]-88.888889)<1e-5,
              "original EV TRIO ROI not reproduced")
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print("CONDITIONAL_EXACT_ROI_DONE")

if __name__=="__main__":
    main()
