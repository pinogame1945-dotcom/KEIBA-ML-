#!/usr/bin/env python3
"""EXACT DEDUPED portfolio: original profitable odds lines + red->black converted odds lines.
Historical-only, fixed EV anchor, never reselect horses/templates/bands. Scores trained on prior years.
Retrospective cohort selection by later outcomes is openly flagged, and NOT represented as true future OOS.
"""
import argparse,csv,gzip,json
from collections import defaultdict,Counter
from pathlib import Path
import pandas as pd
import run_l2_fixed_ticket_pruning_3040_v1 as p
import run_l2_ai_market_rank_betfrontier_v1 as rank
from build_l2_market_helpers_v1 import load_day,load_odds_day
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,assert_ok

def readcsv(path):
    op=gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))
def tid(row):
    return row["order"],int(row["k"]),row["family"],row["odds_band"]
def accumulate(dest,year,portfolio,cards):
    a=dest[year,portfolio]
    n=len(cards)
    a["tickets"]+=n
    a["hits"]+=sum(cards[t][1]>0 for t in cards)
    a["return_yen"]+=sum(cards[t][1] for t in cards)
    a["races_bought"]+=int(n>0)
    a["hit_races"]+=int(any(cards[t][1]>0 for t in cards))
def calc_year(year,pred,picks,races,root,stats,limits,history,flips,acc):
    groups=pred[pred.year.eq(year)].groupby("_race_id",sort=False)
    yrows=races[races.year.eq(year)]
    assert len(yrows)==RACES_YEAR
    idx=yrows.set_index("race_id")
    dates=yrows.groupby("race_date")["race_id"].apply(list)
    learn=defaultdict(lambda:[0,0,0.0])
    future=[]
    bought=0
    ids_checked=0
    for date, ids in dates.items():
        packs=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",set(ids))
        odds=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",set(ids))
        for rid in sorted(ids):
            ids_checked+=1
            pack=packs.get(rid);market=odds.get(rid)
            assert_ok(pack is not None and market is not None,("missing",rid))
            assert_ok(market.get("odds_kind")=="FINAL" and market.get("source_status")=="result",("not final",rid))
            rows=defaultdict(set)
            cards={}
            for label,ticket,card in p.create_cards(groups.get_group(rid),pack,market,rid,picks["ANCHOR_FULL"][rid],idx.loc[rid]):
                assert label in history,label
                rows[label].add(ticket)
                if ticket in cards: assert cards[ticket]==card
                cards[ticket]=card
            for card in cards.values():
                p.add_learning(learn,card)
                future.append(card)
            if year==2023:continue
            flags={ticket:p.mask(card,stats,limits) for ticket,card in cards.items()}
            orig=set().union(*(v for key,v in rows.items() if history[key])) if rows else set()
            accumulate(acc,year,"ORIGINAL_BLACK_RAW",orig,cards)
            retained_by_policy={}
            combined_converted_all=set()
            for policy in p.POLICIES[1:]:
                conv_raw=set().union(*(v for key,v in rows.items() if key in flips[policy])) if rows else set()
                conv_kept={t for t in conv_raw if flags[t][policy]}
                orig_kept={t for t in orig if flags[t][policy]}
                retained_by_policy[policy]=orig_kept
                combined_converted_all.update(conv_kept)
                accumulate(acc,year,"CONVERTED_ONLY_PRUNED_"+policy,conv_kept,cards)
                accumulate(acc,year,"ORIGINAL_BLACK_PRUNED_"+policy,orig_kept,cards)
                accumulate(acc,year,"ORIGINAL_RAW_PLUS_CONVERTED_RAW_"+policy,orig | conv_raw,cards)
                accumulate(acc,year,"ORIGINAL_RAW_PLUS_CONVERTED_PRUNED_"+policy,orig | conv_kept,cards)
                accumulate(acc,year,"ORIGINAL_PRUNED_PLUS_CONVERTED_PRUNED_"+policy,orig_kept|conv_kept,cards)
            accumulate(acc,year,"ORIGINAL_RAW_PLUS_ALL_CONVERSIONS_PRUNED",orig|combined_converted_all,cards)
            accumulate(acc,year,"ORIGINAL_LEARNED50_PLUS_ALL_CONVERSIONS_PRUNED",retained_by_policy["LEARNED_50"]|combined_converted_all,cards)
            bought+=1
        if ids_checked%1200==0:print("PORTFOLIO_PROGRESS",year,ids_checked,flush=True)
    assert ids_checked==RACES_YEAR
    print("PORTFOLIO_YEAR_END",year,ids_checked,"race-with-price",bought,flush=True)
    return learn,future

def row(year,name,c):
    n=c["tickets"];r=c["return_yen"]
    return dict(year=str(year),portfolio=name,races_bought=c["races_bought"],
      hit_races=c["hit_races"],tickets=n,winning_tickets=c["hits"],
      stake_yen=n*100,return_yen=r,profit_yen=r-n*100,
      roi_pct=round(r/n,6) if n else None,
      coverage_pct=round(c["races_bought"]/(3456 if year in (2024,2025) else 6912)*100,6))
def csv_out(path,rows):
    assert rows
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    parser=argparse.ArgumentParser()
    for arg in ("predictions","picks","racewise","backfill-root","bands","pruning-results","out-dir"):
        parser.add_argument("--"+arg,required=True)
    a=parser.parse_args()
    result=Path(a.pruning_results);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    bandrows=readcsv(a.bands)
    assert len(bandrows)==3040
    history={}
    for r in bandrows:
        _,order,k,fam=r["買い方ID"].split("|")
        label=order,int(k),fam,r["馬券オッズ帯"]
        history[label]=int(r["2023-2025_損益円"])>0
    assert len(history)==3040 and sum(history.values())==245
    converted=readcsv(result/"red_lines_changed_to_black_2024_25.csv.gz")
    flips={key:set() for key in p.POLICIES[1:]}
    for r in converted:
        if r["red_to_black"]=="True" and r["policy"] in flips:
            label=tid(r)
            assert label in history and not history[label]
            flips[r["policy"]].add(label)
    old_audit=readcsv(result/"red_to_black_preservation_audit_all_policies.csv")
    for rec in old_audit:
        if rec["policy"] in flips:
            assert len(flips[rec["policy"]])==int(rec["red_went_black"]),rec["policy"]
    print("ORIGINAL_BLACK",sum(history.values()),"CONVERSIONS",[(k,len(v)) for k,v in flips.items()],flush=True)
    pred,picks=rank.load_predictions(a.predictions,a.picks)
    races=pd.read_csv(a.racewise,compression="infer",dtype={
        "race_id":str,"VALUE_EV_horse":str,"AI_RAW_TOP_horse":str})
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert len(races)==TOTAL_RACES and races.race_id.is_unique
    stats=defaultdict(lambda:[0,0,0.0]);historical=[];acc=defaultdict(Counter)
    for y in YEARS:
        limits=p.create_thresholds(historical,stats) if y!=2023 else {}
        learned,future=calc_year(y,pred,picks,races,Path(a.backfill_root),stats,limits,history,flips,acc)
        p.merge(stats,learned);historical.extend(future)
    prev=readcsv(result/"deduped_original_245_and_all_tickets_union_before_after.csv")
    def prev_get(year,policy):
        return next(r for r in prev if r["year"]==str(year) and r["segment"]=="ORIGINAL_PLUS_245" and r["policy"]==policy)
    for y in (2024,2025):
        orig=acc[y,"ORIGINAL_BLACK_RAW"]
        for col,pcol in (("tickets","before_tickets"),("hits","before_hits"),
                         ("return_yen","before_return_yen"),("races_bought","before_races")):
            assert orig[col]==int(prev_get(y,"BASE")[pcol]),("baseline mismatch",y,col)
        for policy in p.POLICIES[1:]:
            pruned=acc[y,"ORIGINAL_BLACK_PRUNED_"+policy]
            for col,pcol in (("tickets","after_tickets"),("hits","after_hits"),
                             ("return_yen","after_return_yen"),("races_bought","after_races")):
                assert pruned[col]==int(prev_get(y,policy)[pcol]),("pruning mismatch",y,policy,col,pruned[col],prev_get(y,policy)[pcol])
    keys=sorted({label for (_,label) in acc})
    report=[row(y,label,acc[y,label]) for y in (2024,2025) for label in keys]
    report.extend([row("2024-2025",label,acc[2024,label]+acc[2025,label]) for label in keys])
    csv_out(out/"historical_deduped_original_black_plus_converted_all_policies.csv",report)
    def get(label,year):
        return next(r for r in report if r["portfolio"]==label and r["year"]==str(year))
    base=get("ORIGINAL_BLACK_RAW","2024-2025")
    summ=[]
    for policy in p.POLICIES[1:]:
        label="ORIGINAL_RAW_PLUS_CONVERTED_PRUNED_"+policy
        full=get(label,"2024-2025")
        joint=get("ORIGINAL_PRUNED_PLUS_CONVERTED_PRUNED_"+policy,"2024-2025")
        basepr=get("ORIGINAL_BLACK_PRUNED_"+policy,"2024-2025")
        raw=get("ORIGINAL_RAW_PLUS_CONVERTED_RAW_"+policy,"2024-2025")
        # All marginal values refer to exact deduplicated physical tickets.
        rowout=dict(policy=policy,flipped_lines=len(flips[policy]),
           previous_original_black_roi_pct=base["roi_pct"],
           previous_original_black_profit_yen=base["profit_yen"],
           red_flips_pruned_only_tickets=get("CONVERTED_ONLY_PRUNED_"+policy,"2024-2025")["tickets"],
           combined_raw_black_plus_pruned_flips_races=full["races_bought"],
           combined_raw_black_plus_pruned_flips_coverage_pct=full["coverage_pct"],
           combined_raw_black_plus_pruned_flips_tickets=full["tickets"],
           combined_raw_black_plus_pruned_flips_stake_yen=full["stake_yen"],
           combined_raw_black_plus_pruned_flips_return_yen=full["return_yen"],
           combined_raw_black_plus_pruned_flips_profit_yen=full["profit_yen"],
           combined_raw_black_plus_pruned_flips_roi_pct=full["roi_pct"],
           combined_improvement_over_original_black_yen=full["profit_yen"]-base["profit_yen"],
           combined_original_and_flips_both_pruned_races=joint["races_bought"],
           combined_original_and_flips_both_pruned_roi_pct=joint["roi_pct"],
           combined_original_and_flips_both_pruned_profit_yen=joint["profit_yen"],
           combined_both_pruned_delta_vs_black_pruned_yen=joint["profit_yen"]-basepr["profit_yen"],
           raw_added_lines_no_prune_roi_pct=raw["roi_pct"],
           raw_added_lines_no_prune_profit_yen=raw["profit_yen"],
           combined_raw_black_pruned_flips_profit_2024_yen=get(label,2024)["profit_yen"],
           combined_raw_black_pruned_flips_profit_2025_yen=get(label,2025)["profit_yen"])
        summ.append(rowout)
    csv_out(out/"policy_combined_portfolio_summary.csv",summ)
    lookup={r["portfolio"]:r for r in report if r["year"]=="2024-2025"}
    summary=dict(contract="ORIGINAL_PLUS_CONVERTED_EXACT_PHYSICAL_TICKET_DEDUP_V1",
      source_run="38692491374",years_test=[2024,2025],train_2023=True,
      original_black_lines=245,original_nonplus_lines=2795,original_templates=152,
      original_price_bands=20,black_group_selected_hindsight="2023-2025 results",
      converted_group_selected_hindsight="2024-2025 outcomes after pruning",
      is_prospective_selection=False,
      scoring_uses_only_prior_years=True,
      uses_final_odds_not_trade_time=True,
      exact_baseline_245_union_verified=True,
      exact_prior_pruned_245_union_verified=True,
      original_245_no_prune=base,
      original_raw_plus_all_converted_pruned=lookup["ORIGINAL_RAW_PLUS_ALL_CONVERSIONS_PRUNED"],
      original_learned50_plus_all_converted_pruned=lookup["ORIGINAL_LEARNED50_PLUS_ALL_CONVERSIONS_PRUNED"],
      all_policy_portfolio_summaries=summ)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for r in summ:print("COMBINED_PORTFOLIO",json.dumps(r,ensure_ascii=False),flush=True)
    print("ALL20_COMBINED_PORTFOLIO",json.dumps(summary["original_raw_plus_all_converted_pruned"],ensure_ascii=False),flush=True)
    print("ORIGINAL_BLACK_PLUS_CONVERTED_DEDUP_COMPLETE",flush=True)
if __name__=="__main__":main()
