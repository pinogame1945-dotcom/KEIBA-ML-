#!/usr/bin/env python3
"""Exact all-ticket trifecta payout by opponent horse's FINAL win-odds band."""
import argparse,concurrent.futures,csv,itertools,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
from build_l2_market_helpers_v1 import load_day,horse_number_map
from run_l2_value_anchor_bet_frontier_v1 import payouts

YEARS=(2023,2024,2025)
EXPECTED_RACES=3456
EXPECTED_12_6_9={"lines":239574,"hits":940,"return_yen":16936420,"jackpot_hits":594}
EDGES=(0,3,5,7,10,15,20,30,40,50,75,100,150,200,300,500,1000,2000,float("inf"))
PAIRS=((1,2),(1,3),(2,3))
SEGMENTS=("4-5","6-9","10+","4+")
FLOORS=(0,7,10,15,20,30,40,50,75,100)
WINDOWS=((5,15),(7,20),(10,20),(10,30),(10,40),(10,50),
         (10,75),(15,30),(15,40),(15,50),(15,75),(20,40),
         (20,50),(20,75),(20,100),(30,75),(30,100),(30,200),
         (40,100),(50,150),(75,200))
POLICIES={"ALL":(0,float("inf"))}
POLICIES.update({f"MIN_{n}":(n,float("inf")) for n in FLOORS})
POLICIES.update({f"WIN_{lo}_{hi}":(lo,hi) for lo,hi in WINDOWS})
PINNED="0d4af461b77f0060dd9c789dd2da646026fada47"

def need(cond,message):
    if not cond:raise ValueError(message)

def bins(x):
    need(math.isfinite(x) and x>0,"invalid final win odds")
    for i,(lo,hi) in enumerate(zip(EDGES,EDGES[1:])):
        if lo<=x<hi:return i
    raise ValueError(str(x))

def segment(rank):
    return "4-5" if rank<=5 else ("6-9" if rank<=9 else "10+")

def days_manifest(root,dates):
    data=json.loads((root/"data/manifest.json").read_text(encoding="utf8"))["days"]
    result={}
    for day in dates:
        item=data.get(day)
        need(item is not None and item.get("status")=="SUCCESS","missing day "+day)
        file=item.get("file")
        need(isinstance(file,str) and file.startswith("data/daily/")
             and file.endswith(".jsonl.gz") and ".." not in file,"manifest file drift "+day)
        result[day]=file
    return result

def load_predictions(path):
    cols=["year","_race_id","_race_date","_horse_id","market_rank","final_win_odds"]
    df=pd.read_csv(path,compression="infer",usecols=cols,low_memory=False,
       dtype={"_race_id":str,"_horse_id":str})
    need(set(df.year.unique())==set(YEARS),"wrong years or 2026 unsealed")
    need(df._race_id.nunique()==10368,"incomplete race count")
    need(not df.duplicated(["_race_id","_horse_id"]).any(),"duplicate runner")
    need(df.final_win_odds.notna().all() and df.market_rank.notna().all(),"missing rank/odds")
    need((df.final_win_odds>0).all(),"invalid win odds")
    return df

def prepare(args):
    df=load_predictions(args.predictions)
    days=sorted({str(v)[:10] for v in df._race_date})
    need(300<=len(days)<=350,"unexpected days")
    actual=days_manifest(Path(args.backfill_root),days)
    files={"data/manifest.json",*actual.values()}
    Path(args.sparse_paths).write_text("".join("/"+x+"\n" for x in sorted(files)))
    print("OPP_WIN_ODDS_PREPARE",len(days),len(files),flush=True)

def worker(year,frame,root_text):
    root=Path(root_text); groups=frame.groupby("_race_id",sort=False)
    races=frame[["_race_id","_race_date"]].drop_duplicates()
    need(len(races)==EXPECTED_RACES,"missing year race")
    byday=(races.assign(day=races._race_date.astype(str).str[:10])
          .groupby("day")["_race_id"].apply(list))
    manifest=days_manifest(root,list(byday.index))
    totals=defaultdict(Counter)
    audit=Counter()
    for day_i,(day,ids) in enumerate(byday.items(),1):
        result=load_day(root/manifest[day],set(ids))
        for rid in sorted(ids):
            pack=result.get(rid)
            need(pack is not None,"missing pack "+rid)
            group=groups.get_group(rid)
            mapping=horse_number_map(pack)
            need(set(group._horse_id.astype(str)).issubset(mapping),"missing horse id "+rid)
            ranked=defaultdict(list);opps=[]
            for hid,rank,raw_odds in group[["_horse_id","market_rank","final_win_odds"]].itertuples(index=False,name=None):
                rank=int(rank);odds=float(raw_odds)
                need(1<=rank<=30,"market rank invalid "+rid)
                n=mapping[str(hid)];ranked[rank].append(n)
                if rank>=4:opps.append((rank,n,bins(odds)))
            pays=payouts(pack,rid)
            for pair in PAIRS:
                if len(ranked.get(pair[0],[]))!=1 or len(ranked.get(pair[1],[]))!=1:
                    audit["unusable_"+str(pair)]+=1
                    continue
                a,b=(ranked[k][0] for k in pair)
                need(a!=b,"same horse axes")
                for rank,n,bi in opps:
                    need(n not in (a,b),"same horse opp")
                    prices=[int(round(pays.get(("TRIFECTA",order),0)))
                            for order in itertools.permutations((a,b,n),3)]
                    hit=sum(v>0 for v in prices)
                    jackpot=sum(v>=10000 for v in prices)
                    for seg in (segment(rank),"4+"):
                        c=totals[(pair,seg,bi)]
                        c["lines"]+=6;c["opponents"]+=1;c["hits"]+=hit
                        c["jackpot_hits"]+=jackpot;c["pay"]+=sum(prices)
                        c["jackpot_pay"]+=sum(v for v in prices if v>=10000)
                    audit["opponent_sets"]+=1
            audit["races"]+=1
        if day_i%65==0:
            print("WIN_ODDS_YEAR",year,"days",day_i,flush=True)
    need(audit["races"]==EXPECTED_RACES,"incomplete year")
    return dict(totals),dict(audit)

def sum_counters(xs):
    c=Counter()
    for x in xs:c.update(x)
    return c

def report(c):
    lines=c["lines"];return_yen=c["pay"]
    return dict(lines=lines,opponents=c["opponents"],hits=c["hits"],
         jackpot_hits=c["jackpot_hits"],stake_yen=lines*100,return_yen=return_yen,
         jackpot_return_yen=c["jackpot_pay"],profit_yen=return_yen-100*lines,
         roi_pct=100*return_yen/(100*lines) if lines else None)

def make_summary(records,group):
    raw={i:records.get((group[0],group[1],i),Counter())
         for i in range(len(EDGES)-1)}
    baseline=report(sum_counters(raw.values()))
    atomic=[dict(lo=EDGES[i],hi="INF" if math.isinf(EDGES[i+1]) else EDGES[i+1],
                 **report(raw[i])) for i in raw]
    policies={}
    for name,(lo,hi) in POLICIES.items():
        picked=sum_counters(raw[i] for i in raw if EDGES[i]>=lo and EDGES[i+1]<=hi)
        s=report(picked)
        s["saving_pct"]=100*(baseline["lines"]-s["lines"])/baseline["lines"] if baseline["lines"] else 0
        s["jackpot_retention_pct"]=(100*s["jackpot_hits"]/baseline["jackpot_hits"]
                                    if baseline["jackpot_hits"] else None)
        s["payout_retention_pct"]=(100*s["return_yen"]/baseline["return_yen"]
                                   if baseline["return_yen"] else None)
        policies[name]=s
    return dict(all=baseline,atomic=atomic,policies=policies)

def main(args):
    frame=load_predictions(args.predictions)
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        tasks={year:executor.submit(worker,year,frame.loc[frame.year.eq(year)].copy(),
                                    args.backfill_root) for year in YEARS}
        results={year:tasks[year].result() for year in YEARS}
    all_=defaultdict(Counter)
    for year in YEARS:
        for k,v in results[year][0].items():all_[k].update(v)
    records={str(year):results[year][0] for year in YEARS}
    records["2023-2025"]=dict(all_)
    combos=[(pair,seg) for pair in PAIRS for seg in SEGMENTS]
    def k(pair,seg):return f"{pair[0]}+{pair[1]}|{seg}"
    computed={period:{k(pair,seg):make_summary(rows,(pair,seg)) for pair,seg in combos}
              for period,rows in records.items()}
    focus="1+2|6-9"
    need(all(computed["2023-2025"][focus]["all"][field]==value
              for field,value in EXPECTED_12_6_9.items()),
          "fixed 1+2 and 6-9 family disagrees with earlier complete paid ledger")
    def training_pick(trainyears):
        tr=defaultdict(Counter)
        for year in trainyears:
            for k_,v in records[str(year)].items():tr[k_].update(v)
        row=make_summary(tr,((1,2),"6-9"))
        choices=[]
        for name,s in row["policies"].items():
            if name=="ALL" or s["lines"]<1200 or s["hits"]<15:continue
            if s["saving_pct"]<25 or (s["jackpot_retention_pct"] or 0)<40:continue
            choices.append((s["roi_pct"],s["profit_yen"],name))
        if not choices:return dict(rule=None,choices=0)
        _,__,name=max(choices)
        return dict(rule=name,choices=len(choices),training=row["policies"][name])
    chrono={}
    for test,train in ((2024,(2023,)),(2025,(2023,2024))):
        rule=training_pick(train)
        rule.update(train_years=list(train),test_year=test,
            test=computed[str(test)][focus]["policies"].get(rule["rule"]) if rule["rule"] else None,
            test_baseline=computed[str(test)][focus]["all"])
        chrono[str(test)]=rule
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    binsrows=[];rulesrows=[]
    for year,groups in computed.items():
        for group,summary in groups.items():
            axis,segment_=group.split("|")
            for row in summary["atomic"]:
                binsrows.append(dict(period=year,axes=axis,popularity=segment_,**row))
            for policy,row in summary["policies"].items():
                rulesrows.append(dict(period=year,axes=axis,popularity=segment_,policy=policy,**row))
    for filename,rows in (("opponent_win_odds_bins.csv",binsrows),
                          ("opponent_win_odds_windows.csv",rulesrows)):
        with (out/filename).open("w",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    summary=dict(contract="L2_2_OPPONENT_WIN_ODDS_FINAL_V1",years=YEARS,
      frozen_2026=True,original_payout_baseline_reconciled=True,
      prior_family_12_6to9=EXPECTED_12_6_9,pinned_backfill=PINNED,
      quote_type="horse FINAL win odds, not trifecta odds, not live tradable prices",
      grouping="Each horse opponent bought as six 100yen trifecta permutations. Both losers and winners counted.",
      experimental_training="2023 to 2024 and 2023+2024 to 2025; ROI optimized TRAIN only with min saved 25%, min jackpot retention 40%, min 15 hits, min 1200 lines",
      yearly_audit={str(y):results[y][1] for y in YEARS},
      chrono=chrono,
      by_group={year:{group:dict(all=v["all"],policies=v["policies"])
                      for group,v in groups.items()} for year,groups in computed.items()})
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print("OPP_WIN_ODDS_VALIDATED",computed["2023-2025"][focus]["all"],flush=True)
    print("OPP_WIN_ODDS_CHRONOLOGICAL",json.dumps(chrono,ensure_ascii=False),flush=True)

def selftest():
    need([bins(v) for v in (9.9,10,29.9,30,2000)]==[3,4,6,7,17],"odds bin boundary")
    need([segment(v) for v in (4,7,10)]==["4-5","6-9","10+"],"popularity boundaries")
    check={i:Counter(lines=6,pay=12000 if i==5 else 0) for i in range(18)}
    res=make_summary({((1,2),"6-9",i):v for i,v in check.items()},((1,2),"6-9"))
    need(res["policies"]["WIN_10_30"]["lines"]==18,"band arithmetic")
    print("OPP_WIN_ODDS_SELFTEST_OK")

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
