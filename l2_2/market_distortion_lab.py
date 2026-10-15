#!/usr/bin/env python3
"""L2.2: statistical ticket market lab, train 2023->test 2024; train 2023-24->test 2025."""
import argparse, concurrent.futures, csv, gzip, itertools, json, math, sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
from build_l2_market_helpers_v1 import load_day,load_odds_day,horse_number_map
from run_l2_all_bet_types_ticket_odds_gate_v1 import ticket_price
from run_l2_value_anchor_bet_frontier_v1 import payouts

YEARS=(2023,2024,2025)
EXPECTED_L21=(13479,376,1471660)
BACKFILL="0d4af461b77f0060dd9c789dd2da646026fada47"
FREEZE="9655c717333636b5cc931f672e12dbc377e396bc"

def need(ok,message):
    if not ok: raise ValueError(message)

def odds_bin(odds):
    return next((str(b) for b in (10,30,100,300) if odds<b),"300+")

def pl(p,*seq):
    v=1.
    used=0.
    for x in seq:
        v*=p[x]/max(1e-10,1-used)
        used+=p[x]
    return v

def cells(kind,nums,price,rank,p,book):
    ranks=[rank[x] for x in nums]
    b=odds_bin(price)
    span="tight" if max(ranks)-min(ranks)<=4 else "spread"
    result=[("SHAPE",kind,b,str(sum(x<=3 for x in ranks))+span),
            ("CROWD",kind,b,("heavy" if max(p.values())>=.33 else "light")+
             ("fav" if 1 in ranks else "other"))]
    if kind in ("EXACTA","TRIFECTA"):
        result.append(("PLACEMENT",kind,b,str(ranks.index(1)+1) if 1 in ranks else "other"))
        parent_type="QUINELLA" if kind=="EXACTA" else "TRIO"
        parent=ticket_price((parent_type,tuple(sorted(nums))),book)
        if parent:
            permutations=itertools.permutations(nums)
            numerator=pl(p,*nums)
            denominator=sum(pl(p,*t) for t in permutations)
            factor=(.75/.775) if kind=="EXACTA" else (.725/.75)
            theoretical=parent*factor*denominator/max(numerator,1e-12)
            ratio=price/theoretical
            label="low" if ratio<.85 else "mid" if ratio<1.05 else "high" if ratio<1.3 else "very_high"
            result.append(("CROSS_POOL",kind,b,label))
    return result

def candidates(nums):
    for pair in itertools.combinations(nums,2):
        yield "QUINELLA",tuple(sorted(pair))
        for perm in itertools.permutations(pair): yield "EXACTA",perm
    for triple in itertools.combinations(nums,3):
        yield "TRIO",tuple(sorted(triple))
        for perm in itertools.permutations(triple): yield "TRIFECTA",perm

def manifest_paths(root,days):
    a=json.loads((root/"data/manifest.json").read_text())["days"]
    b=json.loads((root/"data/odds/manifest.json").read_text())["days"]
    mapping={}
    for d in days:
        names=[]
        for m,prefix in ((a,"data/daily/"),(b,"data/odds/daily/")):
            info=m.get(d)
            need(info and info.get("status")=="SUCCESS","manifest metadata/status drift "+d)
            name=info.get("file")
            need(isinstance(name,str) and name.startswith(prefix) and
                 name.endswith(".jsonl.gz") and ".." not in name,
                 "manifest remote filename invalid "+str(name))
            names.append(name)
        mapping[d]=names
    return mapping

def prepare(a):
    df=pd.read_csv(a.predictions,compression="infer",usecols=["year","_race_date"])
    need(set(df.year.unique())==set(YEARS),"2026 sealed")
    days=sorted(set(df._race_date.astype(str).str[:10]))
    need(300<=len(days)<=350,"race-day coverage drift")
    mapping=manifest_paths(Path(a.backfill_root),days)
    files={"data/manifest.json","data/odds/manifest.json"}
    for names in mapping.values(): files.update(names)
    Path(a.sparse_paths).write_text("".join("/"+x+"\n" for x in sorted(files)))
    print("L22_METADATA_CONFIRMED",len(days),len(files),flush=True)

def l21(paths):
    out={}
    for path in paths:
        with gzip.open(path,"rt",encoding="utf8",newline="") as f:
            for r in csv.DictReader(f):
                k=(int(r["year"]),r["race_id"],r["bet_type"],
                   tuple(int(n) for n in r["selection"].split("-")))
                need(k not in out,"duplicate frozen ticket "+str(k))
                out[k]=int(r["payout_yen"])
    need((len(out),sum(x>0 for x in out.values()),sum(out.values()))==EXPECTED_L21,
         "frozen L2.1 ledger regression")
    return out

def stats(values):
    n=len(values);wins=sorted((int(x) for x in values if x>0),reverse=True)
    paid=sum(wins);top=sum(wins[:3])
    return dict(n=n,hits=len(wins),paid=paid,profit=paid-100*n,
                roi=paid/(100*n) if n else 0.,
                top3=top, top3_ratio=top/paid if paid else 1.,
                minus3=(paid-top)/(100*n) if n else 0.)

def choose(aggregates):
    merged=defaultdict(lambda:dict(n=0,hits=0,paid=0,top3_upper=0))
    for yearly in aggregates:
        for k,s in yearly.items():
            target=merged[k]
            for field in ("n","hits","paid"):target[field]+=s[field]
            target["top3_upper"]+=s["top3"]
    accepted={}
    for k,s in merged.items():
        n=s["n"];paid=s["paid"]
        if n<300 or s["hits"]<12 or paid<1.04*100*n: continue
        # Safe upper bound on pooled largest-three payouts.
        if s["top3_upper"]>0.8*paid or (paid-s["top3_upper"])<.75*100*n:continue
        accepted[k]=dict(n=n,hits=s["hits"],roi=paid/(100*n))
    return accepted

def year_run(year,frame,history,mode,policy,frozen):
    root=Path(history)
    groups=frame.groupby("_race_id",sort=False)
    days=frame[["_race_id","_race_date"]].drop_duplicates()
    need(len(days)==3456,"all 3456 races required year "+str(year))
    bydate=days.assign(day=days._race_date.astype(str).str[:10]).groupby("day")["_race_id"].apply(list)
    paths=manifest_paths(root,list(bydate.index))
    aggregate=defaultdict(list);port=defaultdict(list)
    audits=defaultdict(int)
    for i,(day,races) in enumerate(bydate.items(),1):
        race_file,odds_file=paths[day]
        packs=load_day(root/race_file,set(races))
        books=load_odds_day(root/odds_file,set(races))
        for rid in races:
            pack=packs.get(rid);book=books.get(rid)
            need(pack is not None and book is not None,"missing history "+rid)
            need(book.get("odds_kind")=="FINAL" and book.get("source_status")=="result",
                 "unknown market status "+rid)
            group=groups.get_group(rid)
            no=horse_number_map(pack)
            need(set(group._horse_id.astype(str)).issubset(no),"horse missing from race "+rid)
            market=group.sort_values(["market_rank","final_win_odds","_horse_id"],
                                      ascending=[True,True,True],kind="mergesort")
            ai=group.sort_values(["P__ANCHOR_FULL","market_rank","_horse_id"],
                                  ascending=[False,True,True],kind="mergesort")
            ids=list(dict.fromkeys(list(market._horse_id.astype(str).iloc[:6])+
                                   list(ai._horse_id.astype(str).iloc[:6])))[:9]
            nums=[no[h] for h in ids]
            need(len(nums)>=3 and len(nums)==len(set(nums)),"invalid candidates "+rid)
            raw={no[h]:1/float(odd) for h,odd in
                 group[["_horse_id","final_win_odds"]].itertuples(index=False,name=None)}
            norm=sum(raw.values())
            need(norm>0 and all(math.isfinite(x) and x>0 for x in raw.values()),
                 "bad market prices "+rid)
            p={n:x/norm for n,x in raw.items()}
            ranks={no[h]:rank for rank,h in enumerate(market._horse_id.astype(str),1)}
            refunds=payouts(pack,rid)
            options={}
            for bet in candidates(nums):
                price=ticket_price(bet,book)
                if price is None:
                    audits["missing_prices"]+=1
                    need(not refunds.get(bet,0),"winner without quote "+rid+str(bet))
                    continue
                audits["priced_tickets"]+=1
                winner=int(refunds.get(bet,0))
                c=cells(bet[0],bet[1],price,ranks,p,book)
                if mode=="training":
                    for cell in c:aggregate[cell].append(winner)
                else:
                    relevant=[policy[x] for x in c if x in policy]
                    if not relevant:continue
                    key=(year,rid,*bet)
                    if key in frozen:
                        audits["overlap_l21"]+=1
                        continue
                    score=max((x["roi"]-1)*math.sqrt(min(x["hits"],50)) for x in relevant)
                    options[bet]=(score,winner)
            if mode=="evaluation":
                port["all"].extend(x[1] for x in options.values())
                best=sorted(options.items(),key=lambda z:(-z[1][0],z[0]))[:8]
                port["cap8"].extend(x[1][1] for x in best)
                audits["races_bought"]+=int(bool(options))
        if i%70==0:
            print("L22_SCAN",year,mode,"days",i,"tickets",audits["priced_tickets"],flush=True)
    if mode=="training":return {k:stats(v) for k,v in aggregate.items()},dict(audits)
    return {k:stats(v) for k,v in port.items()},dict(audits)

def execute(a):
    d=pd.read_csv(a.predictions,compression="infer",usecols=[
       "year","_race_id","_race_date","_horse_id","market_rank","final_win_odds","P__ANCHOR_FULL"],
       dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    need(set(d.year.unique())==set(YEARS) and d._race_id.nunique()==10368,"OOS coverage")
    need(not d.duplicated(["_race_id","_horse_id"]).any(),"duplicate OOS horse")
    baseline=l21((a.joined,a.missing))
    frame={y:d[d.year==y].copy() for y in YEARS}
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        tasks={y:pool.submit(year_run,y,frame[y],a.backfill_root,"training",{},set()) for y in YEARS}
        all_years={y:tasks[y].result() for y in YEARS}
    models={2024:choose([all_years[2023][0]]),
            2025:choose([all_years[2023][0],all_years[2024][0]])}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        tasks={y:pool.submit(year_run,y,frame[y],a.backfill_root,"evaluation",
                             models[y],set(baseline)) for y in (2024,2025)}
        evals={y:tasks[y].result() for y in (2024,2025)}
    dest=Path(a.out_dir);dest.mkdir(parents=True,exist_ok=True)
    with gzip.open(dest/"annual_cells.jsonl.gz","wt",encoding="utf8") as f:
        for year in YEARS:
            for cell,record in sorted(all_years[year][0].items()):
                f.write(json.dumps(dict(year=year,arm=cell[0],kind=cell[1],
                    odds_band=cell[2],cohort=cell[3],**record))+"\n")
    with (dest/"forward_portfolios.csv").open("w",newline="") as f:
        columns=("year,policy,selection_cells,tickets,hits,stake_yen,return_yen,"
                 "profit_yen,roi_pct,base_stake_yen,base_return_yen,"
                 "combined_profit_yen,combined_roi_pct").split(",")
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for year in (2024,2025):
            base=[v for k,v in baseline.items() if k[0]==year]
            for policy,s in evals[year][0].items():
                invested=100*s["n"];base_stake=100*len(base)
                writer.writerow(dict(year=year,policy=policy,selection_cells=len(models[year]),
                     tickets=s["n"],hits=s["hits"],stake_yen=invested,return_yen=s["paid"],
                     profit_yen=s["profit"],roi_pct=100*s["roi"],
                     base_stake_yen=base_stake,base_return_yen=sum(base),
                     combined_profit_yen=sum(base)+s["paid"]-base_stake-invested,
                     combined_roi_pct=100*(sum(base)+s["paid"])/(base_stake+invested)))
    report=dict(contract="KEIBA_L2_2_STATISTICAL_V1",freeze_sha=FREEZE,
                backfill_sha=BACKFILL,year_2026_sealed=True,read_only_l21=True,
                test_train_years={"2024":[2023],"2025":[2023,2024]},
                arms=["CROSS_POOL","PLACEMENT","SHAPE","CROWD"],
                data_caveat="FINAL historical closing odds, not live execution prices",
                multiple_testing_warning=True,candidate_horses="union top6 market/top6 frozen AI, cap9",
                max_new_tickets_per_race=8,
                selected_conditions={str(y):len(models[y]) for y in models},
                training_audit={str(y):all_years[y][1] for y in YEARS},
                test_audit={str(y):evals[y][1] for y in evals},
                forward={str(y):evals[y][0] for y in evals})
    (dest/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("L22_RESULT",json.dumps(report["forward"]),flush=True)

def selftest():
    need(odds_bin(10)=="30" and odds_bin(300)=="300+","odds bins")
    need(len(set(candidates((1,2,3))))==16,"canonical/ordered ticket universe")
    raw={1:.45,2:.35,3:.20}
    book={"odds":{"4":{"0102":[10,0,0]},"6":{"0102":[30,0,0]},
                  "7":{"010203":[35,0,0]},"8":{"010203":[150,0,0]}}}
    for kind,nums,quote in (("EXACTA",(1,2),30),("TRIFECTA",(1,2,3),150)):
        need(any(c[0]=="CROSS_POOL" for c in cells(kind,nums,quote,
              {1:1,2:2,3:3},raw,book)),"cross-market feature missing")
    key=("SHAPE","TRIO","30","2tight")
    v=choose([{key:dict(n=500,hits=30,paid=70000,top3=15000)}])
    need(key in v,"training selection")
    need(key not in choose([{key:dict(n=500,hits=1,paid=70000,top3=15000)}]),
         "min hits enforced")
    print("L22_SELFTEST_OK")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--selftest",action="store_true")
    parser.add_argument("--prepare",action="store_true")
    for x in ("predictions","backfill-root","sparse-paths","joined","missing","out-dir"):
        parser.add_argument("--"+x)
    a=parser.parse_args()
    if a.selftest:selftest()
    elif a.prepare:prepare(a)
    else:execute(a)
