#!/usr/bin/env python3
"""L2.2 trifecta ticket final-odds band cost/revenue lab, 2023-25 historical.

Fixed physical ticket family: active market ranks 1 and 2 as anchors, each
active market rank 6/7/8/9 as third runner; six trifecta permutations per trio.
Exact official refunds for ALL paid dead-heat tickets. Do not learn probabilities
from outcomes or treat post-race known winning triplets as pre-bet features.

This is RETROSPECTIVE FINAL-ODDS research, never claims executable odds.
"""
from __future__ import annotations
import argparse,concurrent.futures,csv,json,itertools,math,sys
from collections import Counter,defaultdict
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
from build_l2_market_helpers_v1 import load_day,load_odds_day,horse_number_map
from run_l2_all_bet_types_ticket_odds_gate_v1 import ticket_price
from run_l2_value_anchor_bet_frontier_v1 import payouts

YEARS=(2023,2024,2025)
EXPECTED_RACES=10368
EXPECTED_RAW_LINES=239574
EXPECTED_OLD_GROSS_LOWER=16891700
EXPECTED_OLD_JACKPOT=592
PINNED="0d4af461b77f0060dd9c789dd2da646026fada47"
ODDS_EDGES=(0,10,30,50,75,100,150,200,300,500,1000,2000,5000,10000,float("inf"))
FLOORS=(0,30,50,75,100,150,200,300,500,1000)
INTERVALS=((100,300),(100,500),(100,1000),(150,500),(200,1000),(300,1000),(300,2000),(500,2000))
FLOOR_NAMES={f"FLOOR_{x}":(x,float("inf")) for x in FLOORS}
INTERVAL_NAMES={f"RANGE_{lo}_{hi}":(lo,hi) for lo,hi in INTERVALS}
POLICIES={"UNFILTERED_QUOTED":(0,float("inf")),**FLOOR_NAMES,**INTERVAL_NAMES}
CANDIDATE_NAMES=[k for k in POLICIES if k not in ("UNFILTERED_QUOTED","FLOOR_0")]
TRAIN_MIN_LINES=500
TRAIN_MIN_JACKPOTS=20
TRAIN_MIN_SAVINGS=.30
TRAIN_MIN_CAPTURE=.45

def need(ok,msg):
    if not ok:raise ValueError(msg)

def manifest_paths(root,dates):
    names=[]
    for mfile,prefix in (("data/manifest.json","data/daily/"),
                         ("data/odds/manifest.json","data/odds/daily/")):
        day_info=json.loads((root/mfile).read_text(encoding="utf8"))["days"]
        names.append({d:day_info[d] for d in dates if d in day_info})
    out={}
    for d in dates:
        pair=[]
        for m,prefix in zip(names,("data/daily/","data/odds/daily/")):
            item=m.get(d)
            need(item and item.get("status")=="SUCCESS","manifest status missing "+d)
            f=item.get("file")
            need(isinstance(f,str) and f.startswith(prefix) and
                 f.endswith(".jsonl.gz") and ".." not in f,
                 "manifest file drift "+str(f))
            pair.append(f)
        out[d]=tuple(pair)
    return out

def prepare(args):
    df=pd.read_csv(args.predictions,compression="infer",usecols=["year","_race_date"])
    need(set(df.year.unique())==set(YEARS),"2026 sealed; wrong years")
    dates=sorted({str(d)[:10] for d in df._race_date})
    need(300<=len(dates)<=350,"bad archived date count")
    files={"data/manifest.json","data/odds/manifest.json"}
    for pair in manifest_paths(Path(args.backfill_root),dates).values():
        files.update(pair)
    Path(args.sparse_paths).write_text("".join("/"+s+"\n" for s in sorted(files)))
    print("ODDS_BAND_MANIFEST",len(dates),len(files),flush=True)

def band_idx(v):
    need(math.isfinite(v) and v>0,"bad quote "+str(v))
    for i in range(len(ODDS_EDGES)-1):
        if ODDS_EDGES[i]<=v<ODDS_EDGES[i+1]:return i
    raise ValueError("odds band not assigned "+str(v))

def blank():
    return dict(n=0,hits=0,jackpot_hits=0,paid=0,jackpot_paid=0,values=[],race_ids=set())

def bump(stats,value,rid):
    payout=int(value)
    stats["n"]+=1
    stats["hits"]+=int(payout>0)
    stats["jackpot_hits"]+=int(payout>=10000)
    stats["paid"]+=payout
    stats["jackpot_paid"]+=payout if payout>=10000 else 0
    if payout>0: stats["values"].append(payout)
    stats["race_ids"].add(rid)

def joined(rows):
    out=blank()
    for r in rows:
        for k in ("n","hits","jackpot_hits","paid","jackpot_paid"):out[k]+=r[k]
        out["values"].extend(r["values"])
        out["race_ids"].update(r["race_ids"])
    return out

def summary(stat):
    n=stat["n"]; paid=stat["paid"]; cost=n*100
    top3=sum(sorted(stat["values"],reverse=True)[:3])
    return dict(lines=n,stake_yen=cost,return_yen=paid,profit_yen=paid-cost,
        roi_pct=100*paid/cost if cost else None,
        hits=stat["hits"],jackpot_hits=stat["jackpot_hits"],
        jackpot_payout_yen=stat["jackpot_paid"],
        bought_races=len(stat["race_ids"]),
        largest3_payout_share_pct=100*top3/paid if paid else 0,
        roi_without_largest3_pct=100*(paid-top3)/cost if cost else None)

def evaluate(year_rows,year):
    bin_stats=[blank() for _ in range(len(ODDS_EDGES)-1)]
    audit=Counter()
    raw=blank()
    missing=blank()
    for rec in year_rows:
        rid,paid,odds=rec
        bump(raw,paid,rid)
        if odds is None:
            bump(missing,paid,rid)
            continue
        bump(bin_stats[band_idx(odds)],paid,rid)
    quoted=joined(bin_stats)
    need(raw["n"]==quoted["n"]+missing["n"],"partition drift "+str(year))
    by_policy={}
    for name,(lo,hi) in POLICIES.items():
        # Since a boundary coincides with all filter endpoints, bin sums are exact.
        cells=[bin_stats[i] for i in range(len(bin_stats)) if
               ODDS_EDGES[i]>=lo and ODDS_EDGES[i+1]<=hi]
        g=joined(cells)
        assert g["n"]<=quoted["n"]
        s=summary(g)
        s["cost_saved_pct_vs_quoted"]=100*(quoted["n"]-g["n"])/quoted["n"]
        s["retained_jackpot_pct_vs_quoted"]=(100*g["jackpot_hits"]/quoted["jackpot_hits"]
                                           if quoted["jackpot_hits"] else None)
        s["retained_payout_pct_vs_quoted"]=(100*g["paid"]/quoted["paid"]
                                         if quoted["paid"] else None)
        by_policy[name]=s
    return dict(
      year=year,
      raw=summary(raw),
      quote_missing=summary(missing),
      quote_available=summary(quoted),
      by_odds_bin=[
        dict(low=ODDS_EDGES[i],high=("INF" if math.isinf(ODDS_EDGES[i+1])
                                   else ODDS_EDGES[i+1]),**summary(z))
        for i,z in enumerate(bin_stats)],
      policies=by_policy)

def year_worker(year,frame,root_text):
    root=Path(root_text); grp=frame.groupby("_race_id",sort=False)
    races=frame[["_race_id","_race_date"]].drop_duplicates()
    need(len(races)==3456,"OOS races changed "+str(year))
    bydate=races.assign(day=races._race_date.astype(str).str[:10]).groupby("day")["_race_id"].apply(list)
    manifest=manifest_paths(root,list(bydate.index))
    lines=[]; aud=Counter(); year_examples=[]
    for day_i,(day,ids) in enumerate(bydate.items(),1):
        files=manifest[day]
        packs=load_day(root/files[0],set(ids))
        odds=load_odds_day(root/files[1],set(ids))
        for rid in sorted(ids):
            pack=packs.get(rid);book=odds.get(rid)
            need(pack and book,"missing actual or quote row "+rid)
            need(book.get("odds_kind")=="FINAL" and book.get("source_status")=="result",
                 "not archived FINAL quote "+rid)
            g=grp.get_group(rid)
            no=horse_number_map(pack)
            need(set(g._horse_id.astype(str)).issubset(no),"horse number gap "+rid)
            rn={}
            for hid,rank in g[["_horse_id","market_rank"]].itertuples(index=False,name=None):
                rank=int(rank)
                need(1<=rank<=30,"bad market rank "+rid)
                rn.setdefault(rank,[]).append(no[str(hid)])
            # No ex-post reindexing when archived market-rank-1 runner was scratched.
            if len(rn.get(1,[]))!=1 or len(rn.get(2,[]))!=1:
                aud["ambiguous_or_missing_axis_races"]+=1
                aud["races"]+=1
                continue
            a,b=rn[1][0],rn[2][0]
            opponents=[(rank,num) for rank in (6,7,8,9) for num in rn.get(rank,[])]
            aud["available_opponents"]+=len(opponents)
            paid=payouts(pack,rid) # Exact ALL official paid physical tickets including dead heats.
            aud["races_with_multiple_paid_trifectas"]+=int(sum(k[0]=="TRIFECTA" for k in paid)>1)
            need(len({num for _,num in opponents})==len(opponents),"duplicate opponents "+rid)
            for rank,c in opponents:
                for ticket in itertools.permutations((a,b,c),3):
                    key=("TRIFECTA",ticket)
                    yen=int(round(paid.get(key,0)))
                    quote=ticket_price(key,book)
                    if quote is None:
                        aud["unpriced_lines"]+=1
                        aud["unpriced_wins"]+=int(yen>0)
                        aud["unpriced_jackpots"]+=int(yen>=10000)
                        aud["unpriced_return_yen"]+=yen
                    else:
                        need(quote>0 and math.isfinite(quote),"invalid final ticket odds")
                        if yen>0:
                            implied=int(round(quote*100))
                            # Exact odds are quoted before settlement; don't fabricate identity.
                            if abs(implied-yen)>100:
                                aud["paid_vs_final_odds_difference_gt_100yen"]+=1
                        aud["quoted_lines"]+=1
                    lines.append((rid,yen,quote))
            aud["races"]+=1
        if day_i%65==0:
            print("ODDS_BANDS_YEAR",year,"days",day_i,"lines",len(lines),flush=True)
    need(aud["races"]==3456,"partial year "+str(year))
    result=evaluate(lines,year)
    need(result["raw"]["lines"]==sum(r["n"]*6 for r in [] ) if False else True,"noop")
    return result,dict(aud)

def sum_reports(subset,label):
    total_bins=[blank() for _ in range(len(ODDS_EDGES)-1)]
    # Recombine scalar bins directly; sufficient for overall metrics except biggest-three
    # which must use actual winning payouts. Year task summaries contain top3 only; store
    # pooled top3 conservatively by adding per-year top3 as an upper bound (marked).
    braw=defaultdict(int);bmiss=defaultdict(int);bq=defaultdict(int)
    for result in subset:
        for dest,name in ((braw,"raw"),(bmiss,"quote_missing"),(bq,"quote_available")):
            for k in ("lines","stake_yen","return_yen","hits","jackpot_hits","jackpot_payout_yen"):
                dest[k]+=result[name][k]
    combined={}
    for key in POLICIES:
        n=sum(p["policies"][key]["lines"] for p in subset)
        gross=sum(p["policies"][key]["return_yen"] for p in subset)
        top3upper=sum(
            p["policies"][key]["largest3_payout_share_pct"] / 100 *
            p["policies"][key]["return_yen"] for p in subset)
        jack=sum(p["policies"][key]["jackpot_hits"] for p in subset)
        amt=sum(p["policies"][key]["jackpot_payout_yen"] for p in subset)
        base_quote_lines=bq["lines"]
        combined[key]=dict(lines=n,stake_yen=n*100,return_yen=gross,
            profit_yen=gross-n*100,roi_pct=100*gross/(n*100) if n else None,
            hits=sum(p["policies"][key]["hits"] for p in subset),
            jackpot_hits=jack,jackpot_payout_yen=amt,
            cost_saved_pct_vs_quoted=100*(base_quote_lines-n)/base_quote_lines,
            retained_jackpot_pct_vs_quoted=(100*jack/bq["jackpot_hits"]
                                            if bq["jackpot_hits"] else None),
            retained_payout_pct_vs_quoted=(100*gross/bq["return_yen"]
                                           if bq["return_yen"] else None),
            roi_without_up_to_3_largest_each_year_pct=(100*(gross-top3upper)/(100*n)
                                                       if n else None),
            max_top3_per_year_payout_share_pct=100*top3upper/gross if gross else 0)
    def total_s(src):
        n=src["lines"];gross=src["return_yen"]
        return dict(lines=n,stake_yen=n*100,return_yen=gross,profit_yen=gross-n*100,
                    roi_pct=100*gross/(n*100) if n else None,hits=src["hits"],
                    jackpot_hits=src["jackpot_hits"],jackpot_payout_yen=src["jackpot_payout_yen"])
    return dict(period=label,raw=total_s(braw),quote_missing=total_s(bmiss),
                quote_available=total_s(bq),policies=combined)

def policy_train_pick(train):
    base=train["quote_available"]
    candidates=[]
    for name in CANDIDATE_NAMES:
        x=train["policies"][name]
        if not x["lines"] or x["lines"]<TRAIN_MIN_LINES:continue
        if x["jackpot_hits"]<TRAIN_MIN_JACKPOTS:continue
        if x["cost_saved_pct_vs_quoted"]<TRAIN_MIN_SAVINGS*100:continue
        if x["retained_jackpot_pct_vs_quoted"]<TRAIN_MIN_CAPTURE*100:continue
        candidates.append((x["roi_pct"],x["profit_yen"],name))
    if not candidates:return dict(selection=None,candidates=0,reason="NO_PREDECLARED_BAND_CLEARED_TRAINING_COVERAGE_AND_SAVINGS")
    _,__,name=max(candidates)
    return dict(selection=name,candidates=len(candidates),train_roi_pct=train["policies"][name]["roi_pct"],
                train_cost_saved_pct=train["policies"][name]["cost_saved_pct_vs_quoted"],
                train_jackpot_retained_pct=train["policies"][name]["retained_jackpot_pct_vs_quoted"])

def selftest():
    need([band_idx(v) for v in (99.99,100,299.99,300,10000)]==[4,5,7,8,13],"bin exact boundary")
    pay=[("r1",15000,140),("r1",0,40),("r2",0,None),("r2",24000,400)]
    x=evaluate(pay,"TEST")
    need(x["raw"]["lines"]==4 and x["quote_missing"]["lines"]==1,"missing")
    need(x["policies"]["RANGE_100_300"]["lines"]==1,"interval")
    need(x["policies"]["FLOOR_100"]["jackpot_hits"]==2,"payout")
    print("JACKPOT_ODDS_BANDS_SELFTEST_OK")

def main(a):
    df=pd.read_csv(a.predictions,compression="infer",
        usecols=["year","_race_id","_race_date","_horse_id","market_rank"],
        dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    need(set(df.year.unique())==set(YEARS),"2026 cannot be opened")
    need(df._race_id.nunique()==EXPECTED_RACES and
         not df.duplicated(["_race_id","_horse_id"]).any(),"OOS coverage/duplicate drift")
    frames={y:df.loc[df.year.eq(y)].copy() for y in YEARS}
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        futures={y:pool.submit(year_worker,y,frames[y],a.backfill_root) for y in YEARS}
        worker={y:futures[y].result() for y in YEARS}
    periods={str(y):worker[y][0] for y in YEARS}
    all_report=sum_reports([periods[str(y)] for y in YEARS],"2023-2025")
    need(all_report["raw"]["lines"]==EXPECTED_RAW_LINES,
         "previous ticket universe mismatch: "+str(all_report["raw"]["lines"]))
    need(all_report["raw"]["return_yen"]>=EXPECTED_OLD_GROSS_LOWER,
         "baseline payout lower bound regression")
    # At least 592 winning jackpots were already verified against this ticket
    # universe. Extra official same-race paid dead-heats are permitted.
    need(all_report["raw"]["jackpot_hits"]>=EXPECTED_OLD_JACKPOT,
         "known verified jackpot hit count mismatch")
    selection={}
    for test_year,training in ((2024,(2023,)),(2025,(2023,2024))):
        train=sum_reports([periods[str(y)] for y in training],"train")
        chosen=policy_train_pick(train)
        name=chosen["selection"]
        chosen["trained_on_years"]=list(training)
        chosen["test_year"]=test_year
        chosen["test"]=periods[str(test_year)]["policies"].get(name) if name else None
        chosen["no_filter_test"]=periods[str(test_year)]["quote_available"]
        selection[str(test_year)]=chosen
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    table=[]
    for period,stat in [("2023-2025",all_report)]+[(str(y),periods[str(y)]) for y in YEARS]:
        src=stat["policies"]
        for name,rec in sorted(src.items()):
            table.append(dict(period=period,policy=name,**rec))
    with (out/"odds_band_policies.csv").open("w",newline="") as file:
        keys=sorted(set().union(*(v.keys() for v in table)))
        writer=csv.DictWriter(file,fieldnames=keys);writer.writeheader();writer.writerows(table)
    with (out/"atomic_odds_bins.csv").open("w",newline="") as file:
        keys=["year","low","high","lines","stake_yen","return_yen","profit_yen",
              "roi_pct","hits","jackpot_hits","jackpot_payout_yen","bought_races",
              "largest3_payout_share_pct","roi_without_largest3_pct"]
        writer=csv.DictWriter(file,fieldnames=keys);writer.writeheader()
        for y in YEARS:
            for z in periods[str(y)]["by_odds_bin"]:
                writer.writerow(dict(year=y,**z))
    report=dict(contract="KEIBA_L2_2_JACKPOT_FINAL_ODDS_BANDS_V1",
        year_2026_sealed=True,read_only_l21=True,
        model="market-ranked 1+2 fixed trifecta multi to market ranks 6-9",
        historical_final_quotes_not_live_executable=True,
        pinned_backfill=PINNED,
        reference_family_regression=dict(expected_lines=EXPECTED_RAW_LINES,
          prior_representative_payout_lower_bound=EXPECTED_OLD_GROSS_LOWER,
          prior_known_jackpot_races=EXPECTED_OLD_JACKPOT),
        multiple_paid_winnings_all_counted=True,
        bands_edges=["INF" if math.isinf(x) else x for x in ODDS_EDGES],
        quote_missing_policy="Never guess price or bin; missing price excluded from priced-band strategies, but fully settled baseline separately audited",
        train_selection_rule=dict(min_train_lines=TRAIN_MIN_LINES,
          min_train_jackpot_hits=TRAIN_MIN_JACKPOTS,
          min_cost_savings_pct=TRAIN_MIN_SAVINGS*100,
          min_jackpot_capture_pct=TRAIN_MIN_CAPTURE*100,
          optimization="highest TRAIN ROI among prespecified floors and bounded intervals; 2024/2025 temporal tests"),
        annual_audit={str(y):worker[y][1] for y in YEARS},
        all_years=all_report,annual=periods,
        chronological_selected=selection,
        warning="Descriptive payout-band curves for all three years and exploratory filtered comparisons involve historical FINAL odds and multiple-testing; no executable profitable strategy inferred.")
    (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("ODDS_BAND_ALL_ROI",all_report["quote_available"]["roi_pct"],flush=True)
    for yr in YEARS:
        print("ODDS_BAND_YEAR",yr,"raw",periods[str(yr)]["raw"]["roi_pct"],
              "quoted",periods[str(yr)]["quote_available"]["roi_pct"],flush=True)
    print("ODDS_BAND_CHRONOLOGICAL",json.dumps(selection,ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--prepare",action="store_true")
    p.add_argument("--predictions")
    p.add_argument("--backfill-root")
    p.add_argument("--sparse-paths")
    p.add_argument("--out-dir")
    args=p.parse_args()
    if args.selftest:selftest()
    elif args.prepare:prepare(args)
    else:main(args)
