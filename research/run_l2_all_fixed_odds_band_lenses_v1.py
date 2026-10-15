#!/usr/bin/env python3
"""Odds-band x existing horse-level lenses on FIXED original EV-anchor 152 templates.

7 physical bet types; 20 exact final-quote odds bins; unchanged original
ticket generation. P__ANCHOR_FULL / Correction Risk / L1 ability distill /
horse-rank disagreement are existing OOS horse scores from the archived file.
Council votes/Outsider are explicitly NOT mislabeled as these proxy features.
No trainable model and no fitted thresholds: all gate definitions below are
frozen ex ante. This is an exploratory stratification, not a live ROI claim.
Never use same-year winnings to decide a gate.
"""
from __future__ import annotations
import argparse,csv,gzip,json,math
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path
import pandas as pd

import run_l2_fixed_ticket_pruning_3040_v1 as fixed
import run_l2_ai_market_rank_betfrontier_v1 as ranklab
from build_l2_market_helpers_v1 import load_day,load_odds_day,horse_number_map
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,assert_ok

# Fixed before looking at line ROI. "top" includes all ticket participants,
# with EV anchor horse excluded for companion-only rows.
GATES=("ALL", "RAW_TOP6","RAW_TOP10",
       "CALIBRATED_TOP6","CALIBRATED_TOP10",
       "MARKET_TOP6","MARKET_TOP10",
       "RISK_LT040","RISK_LT060",
       "ABILITY_Z_GE_M1","ABILITY_Z_GE_0",
       "FAIR_RATIO_GE_100","FAIR_RATIO_GE_125",
       "MODEL_WIN_P_GE_003",
       "MARKET_UNDERVALUE_GE_2",
       "AGREEMENT_TOP6","RISK060_AND_FAIR100")
SCOPE=("ALL_3040","ORIGINAL_245_BLACK","ORIGINAL_2795_OTHER")
YEARS_TEXT=("2023","2024","2025")
BET_TYPES=("WIN","PLACE","QUINELLA","WIDE","EXACTA","TRIO","TRIFECTA")

def signature(tid):
    return "|".join(str(x) for x in tid)

def load_original(path):
    with open(path,encoding="utf-8",newline="") as f:rs=list(csv.DictReader(f))
    assert len(rs)==3040
    result={}
    for r in rs:
        _,order,k,fam=r["買い方ID"].split("|")
        tid=(order,int(k),fam,r["馬券オッズ帯"])
        assert tid not in result
        result[tid]=r
    assert sum(int(x["2023-2025_損益円"])>0 for x in result.values())==245
    return result

def frame(pred_path,picks):
    d,p=ranklab.load_predictions(pred_path,picks)
    mandatory=("risk","distill_score_z","market_prob_norm","rank_gap_norm","top3_overlap",
               "distill_ai_rank","P__ANCHOR_FULL","market_rank")
    src=pd.read_csv(pred_path,usecols=["_race_id","_horse_id",*mandatory[:5]],
                    compression="infer",dtype={"_race_id":str,"_horse_id":str},
                    low_memory=False)
    assert not src.duplicated(["_race_id","_horse_id"]).any()
    for col in mandatory[:5]:
        src[col]=pd.to_numeric(src[col],errors="raise")
        assert src[col].notna().all(),("missing source",col)
    d=d.merge(src,on=["_race_id","_horse_id"],validate="one_to_one")
    assert len(d)==len(src)
    assert d["risk"].between(0,1).all()
    assert d["market_prob_norm"].ge(0).all()
    return d,p

def gate_flags(ticket,horse,anchor):
    kind,nums=ticket
    partners=[n for n in nums if n!=anchor]
    participants=partners if partners else [anchor]
    part=[horse[n] for n in participants]
    risk=max(p["risk"] for p in part)
    ability=min(p["distill_score_z"] for p in part)
    fair_ratio=min(p["P__ANCHOR_FULL"]/max(p["market_prob_norm"],1e-9) for p in part)
    pwin=min(p["P__ANCHOR_FULL"] for p in part)
    raw_max=max(p["distill_ai_rank"] for p in part)
    prob_max=max(p["P_rank"] for p in part)
    market_max=max(p["market_rank"] for p in part)
    undervalue=min((p["market_rank"]-p["distill_ai_rank"]) for p in part)
    return dict(ALL=True,
        RAW_TOP6=raw_max<=6,RAW_TOP10=raw_max<=10,
        CALIBRATED_TOP6=prob_max<=6,CALIBRATED_TOP10=prob_max<=10,
        MARKET_TOP6=market_max<=6,MARKET_TOP10=market_max<=10,
        RISK_LT040=risk<.40,RISK_LT060=risk<.60,
        ABILITY_Z_GE_M1=ability>=-1.0,ABILITY_Z_GE_0=ability>=0,
        FAIR_RATIO_GE_100=fair_ratio>=1.0,
        FAIR_RATIO_GE_125=fair_ratio>=1.25,
        MODEL_WIN_P_GE_003=pwin>=.03,
        MARKET_UNDERVALUE_GE_2=undervalue>=2,
        AGREEMENT_TOP6=raw_max<=6 and market_max<=6,
        RISK060_AND_FAIR100=risk<.6 and fair_ratio>=1.0)

def update(c,bet,ret,rid):
    c["tickets"]+=1;c["hit_tickets"]+=int(ret>0);c["return_yen"]+=ret
    if c.get("_last_race")!=rid:
        c["races_bought"]+=1
        c["_last_race"]=rid
    if ret>0 and c.get("_last_hit_race")!=rid:
        c["hit_races"]+=1
        c["_last_hit_race"]=rid

def add_lines(a,b):
    c=Counter()
    for f in ("tickets","hit_tickets","return_yen","races_bought","hit_races"):
        c[f]=a[f]+b[f]
    return c

def cell(c):
    t=int(c["tickets"]); ret=int(c["return_yen"])
    return {"races_bought":int(c["races_bought"]),"tickets":t,
        "hit_tickets":int(c["hit_tickets"]),
        "hit_races":int(c["hit_races"]),"misses":t-int(c["hit_tickets"]),
        "stake_yen":t*100,"return_yen":ret,
        "profit_yen":ret-t*100,
        "hit_rate_exact":f"{int(c['hit_tickets'])}/{t}" if t else "",
        "roi_exact":f"100*{ret}/{t*100}" if t else ""}

def write_csv(path,records):
    assert records
    with (gzip.open(path,"wt",encoding="utf-8",newline="") if path.suffix==".gz"
          else path.open("w",encoding="utf-8",newline="")) as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)

def main():
    ap=argparse.ArgumentParser()
    for a in ("predictions","picks","racewise","backfill-root","bands",
              "old-union","out-dir"):
        ap.add_argument("--"+a,required=True)
    a=ap.parse_args()
    root=Path(a.backfill_root)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    originals=load_original(a.bands)
    pred,picks=frame(a.predictions,a.picks)
    races=pd.read_csv(a.racewise,compression="infer",dtype={
        "race_id":str,"VALUE_EV_horse":str,"AI_RAW_TOP_horse":str})
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert len(races)==TOTAL_RACES and races.race_id.is_unique and set(races.year)==set(YEARS)
    per_year={}
    for year in YEARS:
        line=defaultdict(Counter);fam=defaultdict(Counter)
        union=defaultdict(Counter)
        meta=Counter()
        g=pred[pred.year.eq(year)].groupby("_race_id",sort=False)
        rs=races[races.year.eq(year)]
        index=rs.set_index("race_id")
        dates=rs.groupby("race_date")["race_id"].apply(list)
        for day,ids in dates.items():
            packs=load_day(root/"data"/"daily"/f"{day}.jsonl.gz",set(ids))
            quotes=load_odds_day(root/"data"/"odds"/"daily"/f"{day}.jsonl.gz",set(ids))
            for rid in sorted(ids):
                meta["races"]+=1
                pack=packs.get(rid);market=quotes.get(rid)
                assert pack is not None and market is not None,("missing input",rid)
                gf=g.get_group(rid)
                axis=picks["ANCHOR_FULL"][rid]
                no=horse_number_map(pack)
                an=no[axis]
                pr=gf.sort_values(["P__ANCHOR_FULL","market_rank","_horse_id"],
                    ascending=[False,True,True],kind="mergesort")
                rnk=dict(zip(pr["_horse_id"].astype(str),range(1,len(pr)+1)))
                horse={no[str(row["_horse_id"])]:{
                    "P__ANCHOR_FULL":float(row["P__ANCHOR_FULL"]),
                    "market_prob_norm":float(row["market_prob_norm"]),
                    "risk":float(row["risk"]),
                    "distill_score_z":float(row["distill_score_z"]),
                    "distill_ai_rank":int(row["distill_ai_rank"]),
                    "market_rank":int(row["market_rank"]),
                    "P_rank":int(rnk[str(row["_horse_id"])])
                } for _,row in gf.iterrows()}
                # generate original 152 templates. Any duplicate template ticket
                # is deduplicated only in group portfolio, not in 3040 lines.
                by_ticket={}; all_lines=defaultdict(set)
                obs=dict(index.loc[rid]);obs["race_id"]=rid
                for tid,ticket,card in fixed.create_cards(gf,pack,market,rid,axis,obs):
                    assert tid in originals,("NON_ORIGINAL_LINE",tid)
                    all_lines[tid].add(ticket)
                    by_ticket[ticket]=int(card[1])
                flags={t:gate_flags(t,horse,an) for t in by_ticket}
                for tid,ts in all_lines.items():
                    for ticket in ts:
                        refund=by_ticket[ticket]
                        f=flags[ticket]
                        for gate in GATES:
                            if f[gate]:update(line[gate,tid],ticket,refund,rid)
                # One physical ticket per race in family×individual-price-bin.
                for ticket,refund in by_ticket.items():
                    f=flags[ticket]
                    # use actual quote price's exact original band from
                    # fixed.create_cards (line schema) for every unique ticket
                    _,seq=ticket
                    # look up cached original line band with one existing template
                    # no reprice or payout approximation
                    priceband=next(tid[3] for tid,ts in all_lines.items() if ticket in ts)
                    for gate in GATES:
                        if f[gate]:update(fam[gate,ticket[0],priceband],ticket,refund,rid)
                for scope in SCOPE:
                    tick=set()
                    for tid,ts in all_lines.items():
                        was_plus=int(originals[tid]["2023-2025_損益円"])>0
                        if scope=="ALL_3040" or (scope=="ORIGINAL_245_BLACK")==was_plus:
                            tick.update(ts)
                    for ticket in tick:
                        f=flags[ticket]
                        for gate in GATES:
                            if f[gate]:update(union[gate,scope],ticket,by_ticket[ticket],rid)
                if meta["races"]%1500==0:
                    print("ODDS_LENSES_PROGRESS",year,meta["races"],flush=True)
        assert meta["races"]==RACES_YEAR
        # Exact historical 3040-line invariants across ALL 3 years.
        for tid,old in originals.items():
            c=line["ALL",tid]
            for col,metric in (("購入点数","tickets"),("的中点数","hit_tickets"),
                               ("払戻円","return_yen")):
                assert int(old[f"{year}_{col}"])==c[metric],(
                    "FIXED_LINE_BASELINE_DRIFT",year,tid,col,int(old[f"{year}_{col}"]),c[metric])
        print("BASELINE_3040_EXACT_MATCH",year,flush=True)
        per_year[str(year)]=(line,fam,union)
    # Prior 245 union MUST reproduce original before filtering.
    with open(a.old_union,encoding="utf-8",newline="") as f:
        baseline={r["year"]:r for r in csv.DictReader(f) if r["group"]=="THREE_YEAR_TOTAL_PLUS"}
    for year in YEARS_TEXT:
        c=per_year[year][2]["ALL","ORIGINAL_245_BLACK"];orig=baseline[year]
        for label,col in (("races_bought","races_bought"),
                          ("unique_tickets","tickets"),
                          ("hit_tickets","hit_tickets"),("return_yen","return_yen")):
            assert int(orig[label])==c[col],("OLD_BLACK_BASELINE_DRIFT",year,col,c[col],orig[label])
    print("HISTORICAL_245_UNION_EXACT_MATCH",flush=True)
    periods=(*YEARS_TEXT,"2023-2025")
    def pooled(scope,gate,key):
        return add_lines(add_lines(per_year["2023"][scope][gate,key],
                    per_year["2024"][scope][gate,key]),
                    per_year["2025"][scope][gate,key])
    band_rows=[];line_rows=[];union_rows=[]
    for year in periods:
        for bt in BET_TYPES:
            for band in [fixed.bin_name(x) for x in fixed.BINS]:
                base=(pooled(1,"ALL",(bt,band)) if year=="2023-2025"
                       else per_year[year][1]["ALL",bt,band])
                for gate in GATES:
                    cur=(pooled(1,gate,(bt,band)) if year=="2023-2025"
                         else per_year[year][1][gate,bt,band])
                    basec=cell(base);afterc=cell(cur)
                    band_rows.append(dict(year=year,bet_type=bt,odds_band=band,
                        gate=gate,**{"baseline_"+k:v for k,v in basec.items()},
                        **{"filtered_"+k:v for k,v in afterc.items()},
                        saved_losing_tickets=basec["misses"]-afterc["misses"],
                        lost_winning_tickets=basec["hit_tickets"]-afterc["hit_tickets"],
                        delta_profit_yen=afterc["profit_yen"]-basec["profit_yen"]))
        for scope in SCOPE:
            base=pooled(2,"ALL",scope) if year=="2023-2025" else per_year[year][2]["ALL",scope]
            for gate in GATES:
                cur=pooled(2,gate,scope) if year=="2023-2025" else per_year[year][2][gate,scope]
                basec=cell(base);afterc=cell(cur)
                union_rows.append(dict(year=year,scope=scope,gate=gate,
                    **{"baseline_"+k:v for k,v in basec.items()},
                    **{"filtered_"+k:v for k,v in afterc.items()},
                    saved_losing_tickets=basec["misses"]-afterc["misses"],
                    lost_winning_tickets=basec["hit_tickets"]-afterc["hit_tickets"],
                    delta_profit_yen=afterc["profit_yen"]-basec["profit_yen"],
                    original_3y_black_scope=scope=="ORIGINAL_245_BLACK"))
        for tid,old in originals.items():
            if year=="2023-2025":
                base=pooled(0,"ALL",tid)
            else:base=per_year[year][0]["ALL",tid]
            for gate in GATES:
                cur=pooled(0,gate,tid) if year=="2023-2025" else per_year[year][0][gate,tid]
                b=cell(base);v=cell(cur)
                line_rows.append(dict(year=year,original_line=signature(tid),
                    original_3year_profit=int(old["2023-2025_損益円"]),
                    original_is_black=int(old["2023-2025_損益円"])>0,gate=gate,
                    **{"baseline_"+k:z for k,z in b.items()},
                    **{"filtered_"+k:z for k,z in v.items()},
                    saved_losing_tickets=b["misses"]-v["misses"],
                    lost_winning_tickets=b["hit_tickets"]-v["hit_tickets"],
                    delta_profit_yen=v["profit_yen"]-b["profit_yen"]))
    write_csv(out/"odds_band_x_existing_axes_yearly.csv",band_rows)
    write_csv(out/"all_3040_lines_x_existing_axes_yearly.csv.gz",line_rows)
    write_csv(out/"union_black245_other2795_x_existing_axes_yearly.csv",union_rows)
    # A small diagnostic table in every fixed band: it is NEVER a claim that
    # the hindsight "best" is a deployable profitable policy.
    plus=[x for x in line_rows if x["year"]=="2023-2025"
                  and x["original_is_black"]]
    write_csv(out/"original_245_black_lines_axis_profit_deltas.csv",
              sorted(plus,key=lambda r:(r["original_line"],r["gate"])))
    summary={"contract":"EV_ORIGINAL_152_BY_20_ODDS_BANDS_EXISTING_L1_AXES_V1",
        "years":[2023,2024,2025],"races":TOTAL_RACES,
        "tickets_original_unchanged":True,"fixed_lines":3040,
        "old_black_245_union_exact_verified":True,
        "all_years_3040_exact_verified":True,
        "available_lenses":list(GATES),
        "true_7_council_votes_available":False,
        "true_outsider_votes_available":False,
        "true_l175_trust_tier_available":False,
        "horse_level_columns":"market ranks, rank-probability, OOS L1 ability z, risk, normalized market and corrected p",
        "no_fabricated_votes":"Council, Outsider and L1.75 categorical trust tier NOT synthesized from ranking proxies",
        "method":"retrospective fixed thresholds within exact ticket-final-odds bin, show every win and loss; no selection by labels",
        "live_execution_warning":"The odds categories are from archived FINAL odds, not provably available before cut-off",
        "group_warning":"245/2795 profitability scope uses all-year post-hoc labels, for explanation only",
        "year_2026_sealed":True,
        "cost":"standard free CPU, pinned manifest resolved exact files, no paid artifact"}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("DOLLAR_BOX_ODDS_LENSES_COMPLETE",
          json.dumps({"band_rows":len(band_rows),"line_rows":len(line_rows),
          "union_rows":len(union_rows),"scope_245":[r for r in union_rows
          if r["year"]=="2023-2025" and r["scope"]=="ORIGINAL_245_BLACK"]},
              ensure_ascii=False),flush=True)

if __name__=="__main__":main()
